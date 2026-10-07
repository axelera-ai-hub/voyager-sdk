#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Row-wise L2 normalisation in fp32 on RVV (llm.deltanet_l2norm_rvv).

    y = x * rsqrt(sum(x^2) + eps) * scale

The delta rule L2-normalises q and k over the head dim. On Metis that had to be
expressed as rms_norm_kahan with weight 1/sqrt(d) and eps pre-scaled by d, with the
inverse square root coming from a fitted float18 LUT; here it is an exact fp32
reduction and one fsqrt per row.
"""

# mypy: disable-error-code="attr-defined, index, arg-type, call-overload, type-var, misc"

import struct

from axelera.compiler.alto import script as A
import axelera.compiler.alto.axol as axol
from axelera.compiler.alto.axol import intrinsic_registry
from axelera.compiler.alto.axol.intrinsic_registry import FunctionArgument, FunctionArgumentType
from axelera.compiler.alto.compiler.dtypes import MetisTypes
from axelera.compiler.alto.compiler.op_registry import Arg, BufferInfo, ScriptOp, register_op
from axelera.config import HardwareGeneration
import numpy as np
from tvm import tir

from ._rvv import kernel_source, to_scalar, unpacked_shape

_INTRINSIC_NAME = "rvv_deltanet_l2norm"

intrinsic_registry.set(
    name=_INTRINSIC_NAME,
    arguments=[
        FunctionArgument(name="x", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="y", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="eps", type=FunctionArgumentType.UINT64),
        FunctionArgument(name="scale", type=FunctionArgumentType.UINT64),
    ],
    implementation=kernel_source("deltanet_l2norm_rvv"),
)


def _float_bits(value: float) -> int:
    """fp32 bit pattern as an int; the cfunc ABI has no float argument type."""
    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


class DeltaNetL2NormRvv(ScriptOp):
    """L2-normalise each row over the (lane-packed) trailing dim."""

    x = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="(rows, blocks) float32x64; the head dim spans the lane blocks.",
    )
    eps = Arg(float, default=1e-6, kw=True, description="Added to the sum of squares.")
    scale = Arg(
        float,
        default=1.0,
        kw=True,
        description="Applied after normalising; carries the delta rule's d^-0.5 query factor.",
    )
    output = Arg(
        MetisTypes[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="(rows, blocks) float32x64, or None to auto-allocate.",
    )
    core = Arg(int | None, default=None, kw=True, description="Optional AI core to run on.")
    generation = Arg(
        HardwareGeneration | None,
        default=None,
        kw=True,
        description="Optional hardware generation.",
    )

    def validate(self) -> None:
        shape = tuple(int(s) for s in self.x.shape)
        if len(shape) != 2:
            raise ValueError(f"x must be (rows, blocks) lane-packed, got {shape}")
        if self.x.dtype != "float32x64":
            raise ValueError(f"x dtype must be 'float32x64', got '{self.x.dtype}'")

    def infer_outputs(self) -> dict[str, BufferInfo]:
        return {
            "output": BufferInfo(
                shape=tuple(int(s) for s in self.x.shape),
                dtype=self.x.dtype,
                scope=self.x.scope(),
            )
        }

    def implementation(self) -> None:
        rows, d = unpacked_shape(tuple(int(s) for s in self.x.shape))

        x_s = A.unpack(self.x, (rows, d))
        out_s = A.unpack(self.output, (rows, d))

        l1_x = A.alloc_buffer((rows, d), "float32", scope=A.L1)
        l1_out = A.alloc_buffer((rows, d), "float32", scope=A.L1)

        # Each row is normalised independently and a row tile spans whole rows,
        # so every tile is contiguous -- required, as cfunc_dltensor_t has no strides.
        with A.tiles(rows) as tile:
            axol.atomic.copy(x_s.tile(tile, dim=0), output=l1_x.tile(tile, dim=0))
            with A.Invoke(core=self.core):
                A.extern(
                    _INTRINSIC_NAME,
                    l1_x.tile(tile, dim=0),
                    l1_out.tile(tile, dim=0),
                    _float_bits(self.eps),
                    _float_bits(self.scale),
                )
            axol.atomic.copy(l1_out.tile(tile, dim=0), output=out_s.tile(tile, dim=0))

    def emulate(self) -> None:  # type: ignore[override]
        x = to_scalar(self.x).astype(np.float32)
        scale = np.float32(self.scale) / np.sqrt(
            (x * x).sum(axis=-1, keepdims=True) + np.float32(self.eps)
        )
        np.copyto(to_scalar(self.output), (x * scale).astype(np.float32))


deltanet_l2norm_rvv = register_op(
    DeltaNetL2NormRvv,
    "llm.deltanet_l2norm_rvv",
    generation=HardwareGeneration.EUROPA,
    atomic=False,
)

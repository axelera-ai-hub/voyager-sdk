#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Gated RMSNorm tail of the DeltaNet block in fp32 on RVV.

    y = weight * x * rsqrt(mean(x^2) + eps) * silu(gate)

Replaces the float18 inverse-sqrt and SiLU LUTs, and with them the eps rescaling
the DPU path needed: that kernel passed eps' = d*eps so its rsqrt saw
variance + d*eps, and the LUT had to be fitted over that shifted domain. Here eps
matches the torch GatedRMSNorm as written.
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

from ._rvv import kernel_source, sigmoid, to_scalar, unpacked_shape

_INTRINSIC_NAME = "rvv_deltanet_gated_rmsnorm"

intrinsic_registry.set(
    name=_INTRINSIC_NAME,
    arguments=[
        FunctionArgument(name="x", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="gate", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="weight", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="y", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="eps", type=FunctionArgumentType.UINT64),
    ],
    implementation=kernel_source("deltanet_gated_rmsnorm_rvv"),
)


class DeltaNetGatedRmsNormRvv(ScriptOp):
    """RMS-normalise each row over the lane-packed trailing dim, then SiLU-gate it."""

    x = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="(rows, blocks) float32x64 recurrence readout.",
    )
    gate = Arg(MetisTypes[tir.Buffer, "..."], description="(rows, blocks) float32x64 z gate.")
    weight = Arg(MetisTypes[tir.Buffer, "..."], description="(1, blocks) float32x64 norm weight.")
    eps = Arg(float, default=1e-6, kw=True, description="Added to mean(x^2).")
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
        if tuple(int(s) for s in self.gate.shape) != shape:
            raise ValueError(f"gate shape must match x {shape}, got {tuple(self.gate.shape)}")
        if tuple(int(s) for s in self.weight.shape) != (1, shape[1]):
            raise ValueError(
                f"weight must be (1, {shape[1]}), got {tuple(int(s) for s in self.weight.shape)}"
            )
        for name in ("x", "gate", "weight"):
            dtype = getattr(self, name).dtype
            if dtype != "float32x64":
                raise ValueError(f"{name} dtype must be 'float32x64', got '{dtype}'")

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
        gate_s = A.unpack(self.gate, (rows, d))
        weight_s = A.unpack(self.weight, (1, d))
        out_s = A.unpack(self.output, (rows, d))

        l1_weight = A.alloc_buffer((1, d), "float32", scope=A.L1)
        axol.atomic.copy(weight_s, output=l1_weight)

        l1_x = A.alloc_buffer((rows, d), "float32", scope=A.L1)
        l1_gate = A.alloc_buffer((rows, d), "float32", scope=A.L1)
        l1_out = A.alloc_buffer((rows, d), "float32", scope=A.L1)

        # Each row normalises independently, so a row tile is both correct and
        # contiguous -- required, as cfunc_dltensor_t carries no strides.
        with A.tiles(rows) as tile:
            axol.atomic.copy(x_s.tile(tile, dim=0), output=l1_x.tile(tile, dim=0))
            axol.atomic.copy(gate_s.tile(tile, dim=0), output=l1_gate.tile(tile, dim=0))
            with A.Invoke(core=self.core):
                A.extern(
                    _INTRINSIC_NAME,
                    l1_x.tile(tile, dim=0),
                    l1_gate.tile(tile, dim=0),
                    l1_weight,
                    l1_out.tile(tile, dim=0),
                    struct.unpack("<I", struct.pack("<f", float(self.eps)))[0],
                )
            axol.atomic.copy(l1_out.tile(tile, dim=0), output=out_s.tile(tile, dim=0))

    def emulate(self) -> None:  # type: ignore[override]
        x = to_scalar(self.x).astype(np.float32)
        gate = to_scalar(self.gate).astype(np.float32)
        weight = to_scalar(self.weight).astype(np.float32)
        scale = 1.0 / np.sqrt((x * x).mean(axis=-1, keepdims=True) + np.float32(self.eps))
        silu = gate * sigmoid(gate)
        np.copyto(to_scalar(self.output), (weight * x * scale * silu).astype(np.float32))


deltanet_gated_rmsnorm_rvv = register_op(
    DeltaNetGatedRmsNormRvv,
    "llm.deltanet_gated_rmsnorm_rvv",
    generation=HardwareGeneration.EUROPA,
    atomic=False,
)

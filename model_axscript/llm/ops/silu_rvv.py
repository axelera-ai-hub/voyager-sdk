#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Elementwise SiLU in fp32 on RVV (llm.silu_rvv).

    y = x * sigmoid(x)

Replaces the float18 chained-LUT SiLU. The LUT's error budget is stated as an
ABSOLUTE bound (~0.016) over a span of roughly [-25, 40], which is a large relative
error across the bulk of a SiLU output -- the function is near zero for every
negative input.
"""

# mypy: disable-error-code="attr-defined, index, arg-type, call-overload, type-var, misc"

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

_INTRINSIC_NAME = "rvv_silu"

intrinsic_registry.set(
    name=_INTRINSIC_NAME,
    arguments=[
        FunctionArgument(name="x", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="y", type=FunctionArgumentType.DLTENSOR),
    ],
    implementation=kernel_source("silu_rvv"),
)


class SiluRvv(ScriptOp):
    """x * sigmoid(x), elementwise over a lane-packed 2-D buffer."""

    x = Arg(MetisTypes[tir.Buffer, "..."], description="(rows, blocks) float32x64.")
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

        # Elementwise, so a row tile is both correct and contiguous.
        with A.tiles(rows) as tile:
            axol.atomic.copy(x_s.tile(tile, dim=0), output=l1_x.tile(tile, dim=0))
            with A.Invoke(core=self.core):
                A.extern(_INTRINSIC_NAME, l1_x.tile(tile, dim=0), l1_out.tile(tile, dim=0))
            axol.atomic.copy(l1_out.tile(tile, dim=0), output=out_s.tile(tile, dim=0))

    def emulate(self) -> None:  # type: ignore[override]
        x = to_scalar(self.x).astype(np.float32)
        np.copyto(to_scalar(self.output), (x * sigmoid(x)).astype(np.float32))


silu_rvv = register_op(SiluRvv, "llm.silu_rvv", generation=HardwareGeneration.EUROPA, atomic=False)

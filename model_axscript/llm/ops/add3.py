#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Elementwise ``(a + b) + c`` in fp18 as two chained DPU adds (llm.add3).

A datapath has two input units, so the first sum stays in L1 and only the second add
reaches the output: the layer's attention residual is never written to DDR.
"""

# mypy: disable-error-code="attr-defined, index, arg-type"

from axelera.compiler.alto.axol.utils import to_int_tuple
from axelera.compiler.alto.compiler.dtypes import MetisTypes
from axelera.compiler.alto.compiler.op_registry import Arg, BufferInfo, ScriptOp, register_op
import axelera.compiler.alto.script as A
from axelera.config import HardwareGeneration
import numpy as np
from tvm import tir
from tvm.relay.backend.contrib.axelera.utils import float18


def _add(x, y, out, shape):
    """``out = x + y`` in fp18, one datapath per grid point."""
    with A.script():
        for loop_vars in A.grid(*shape, always_tuple=True):
            with A.Datapath():
                sx = A.decl_stream(x.dtype)
                sy = A.decl_stream(y.dtype)
                so = A.decl_stream(out.dtype)
                with A.IFD(num=0):
                    sx.push(x[loop_vars])
                with A.IFD(num=1):
                    sy.push(y[loop_vars])
                with A.DPU():
                    regs = A.array("float18x64")
                    regs[0] = A.cast(sx.pop(), "float18x64")
                    regs[1] = A.cast(sy.pop(), "float18x64")
                    regs[2] = regs[0] + regs[1]
                    so.push(regs[2])
                with A.ODR():
                    out[loop_vars] = so.pop()


class Add3(ScriptOp):
    """Elementwise ``(a + b) + c`` over three same-shape buffers."""

    a = Arg(MetisTypes[tir.Buffer, "..."], description="First summand.")
    b = Arg(MetisTypes[tir.Buffer, "..."], description="Second summand (same shape as a).")
    c = Arg(MetisTypes[tir.Buffer, "..."], description="Third summand (same shape as a).")
    output = Arg(
        MetisTypes[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="Output; defaults to a's shape, dtype and scope.",
    )
    generation = Arg(HardwareGeneration | None, default=None, kw=True)

    def validate(self) -> None:
        shape = to_int_tuple(self.a.shape)
        for name in ("b", "c"):
            other = getattr(self, name)
            if to_int_tuple(other.shape) != shape:
                raise ValueError(f"add3: {name} shape {tuple(other.shape)} != a {shape}")

    def infer_outputs(self) -> dict[str, BufferInfo]:
        return {
            "output": BufferInfo(
                shape=tuple(self.a.shape), dtype=self.a.dtype, scope=self.a.scope()
            )
        }

    def implementation(self) -> None:
        shape = to_int_tuple(self.a.shape)
        ab = A.alloc_buffer(shape, self.a.dtype, scope=A.L1)
        _add(self.a, self.b, ab, shape)
        _add(ab, self.c, self.output, shape)

    def emulate(self) -> None:  # type: ignore[override]
        r = float18.np_round_float32_to_float18
        t = r(r(self.a.astype(np.float32)) + r(self.b.astype(np.float32)))
        self.output[...] = r(t + r(self.c.astype(np.float32))).astype(self.output.dtype)


add3 = register_op(
    Add3,
    "llm.add3",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=True,
)

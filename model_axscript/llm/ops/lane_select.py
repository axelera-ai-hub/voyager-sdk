#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Per-lane broadcast select (llm.lane_select).

Computes out[n, :] = sum_lane(x[n] * onehot) for every row n: a masked
lane-reduce that picks one lane's value out of each row and broadcasts it across
all lanes of the output.

This is the lane->row bridge the gated-DeltaNet layer needs for its per-head
gates.  in_proj_b / in_proj_a project the hidden state to num_v_heads
channels, which a matmul emits heads-in-lanes: x[t] is one PWORD block
whose lane h holds the gate value for head h at timestep t.  The
mixer, by contrast, wants each head's gate as a head-major column (C, 1) (one
scalar per timestep, broadcast across the lanes).  lane_select(x, onehot_h)
with onehot_h = a PWORD that is 1 in lane h and 0 elsewhere extracts head
h's timeseries into exactly that layout.

x streams on IFD0 and the onehot on IFD1 (a single-IFD0 DPU datapath plus a
reduce miscompiles).  The (N,) loop is a cmd-iteration loop (loop var unused
in the body) so the program stays compact.
"""
from axelera.compiler.alto.axol import atomic
from axelera.compiler.alto.compiler.config import HardwareGeneration
from axelera.compiler.alto.compiler.device import PWORD
from axelera.compiler.alto.compiler.dtypes import MetisFloats
from axelera.compiler.alto.compiler.interpreter import emulation as emu
from axelera.compiler.alto.compiler.op_registry import Arg, BufferInfo, ScriptOp, register_op
import axelera.compiler.alto.script as A
import axelera.compiler.backend.tir.ops as tir_ops
import numpy as np
from tvm import tir

# mypy: disable-error-code="attr-defined, index, arg-type"
# flake8: noqa: C901


class LaneSelect(ScriptOp):
    """out[n, :] = sum_lane(x[n] * onehot) (broadcast). See module docstring."""

    x = Arg(MetisFloats[tir.Buffer, "..."], description="(N, 1) rows; lanes hold the values.")
    onehot = Arg(
        MetisFloats[tir.Buffer, "..."], description="(1,) PWORD mask: 1 in the chosen lane."
    )
    out = Arg(MetisFloats[tir.Buffer, "..."] | BufferInfo, default=None, is_output=True, kw=True)
    generation = Arg(HardwareGeneration | None, default=None, kw=True)

    def _n(self) -> int:
        return int(self.x.shape[0])

    def validate(self) -> None:
        if int(self.x.shape[1]) != 1:
            raise ValueError(f"x must be (N, 1); got {tuple(self.x.shape)}.")

    def infer_outputs(self) -> dict[str, BufferInfo]:
        return {"out": BufferInfo(shape=(self._n(), 1), dtype="float32x64", scope=A.DDR)}

    def implementation(self) -> None:
        n = self._n()
        f32, f18 = "float32x64", "float18x64"
        xin = A.alloc_buffer((n, 1), f32, scope=A.L1)
        ohin = A.alloc_buffer((1,), f32, scope=A.L1)
        atomic.copy(self.x, output=xin)
        atomic.copy(self.onehot, output=ohin)

        with A.script():
            with A.Datapath(core=None):
                fx = A.decl_stream(f32)
                fo = A.decl_stream(f32)
                o = A.decl_stream(f32)
                for i in A.serial(n):
                    with A.IFD(num=0):
                        fx.push(xin[i, 0])
                for i in A.serial(n):
                    with A.IFD(num=1):
                        fo.push(ohin[0])
                with A.DPU():
                    x = A.var(f18)
                    r = A.var(f18)
                    for i in A.serial(n):
                        x = A.cast(fx.pop(), f18) * A.cast(fo.pop(), f18)
                        r = tir_ops.dpu_reduce(x, operation=tir_ops.DPU_REDUCE_OP.ADD)
                        o.push(A.cast(r, f32))
                for i in A.serial(n):
                    with A.ODR():
                        self.out[i, 0] = o.pop()

    def emulate(self) -> None:  # type: ignore[override]
        n = self._n()
        x = self.x.reshape(n, PWORD).astype(np.float32)
        oh = self.onehot.reshape(PWORD).astype(np.float32)
        s = emu.rf18(emu.rf18(x) * emu.rf18(oh)).sum(-1)
        np.copyto(self.out, np.repeat(s.reshape(n, 1, 1), PWORD, 2).astype(np.float32))


lane_select = register_op(
    LaneSelect,
    "llm.lane_select",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)

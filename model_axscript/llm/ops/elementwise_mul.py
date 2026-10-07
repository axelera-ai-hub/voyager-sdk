#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Broadcast elementwise multiply (llm.elementwise_mul).

Computes out[n, blk] = a[n, blk] * b[n, blk % bblk] on the DPU, where b is
either full (bblk == blk) or per-row broadcast (bblk == 1).  This is the
glue the gated-DeltaNet layer needs:

* the q/k/v decay products that feed the chunk kernel -- k*beta, v*beta,
  q*exp(g), k*beta*exp(g), k*exp(gc_last-gc) -- are per-row-broadcast
  multiplies (beta / the decay vectors are one scalar per timestep row);
* the gated-RMSNorm gate rmsnorm(x) * silu(z) is a full elementwise multiply
  (combine with llm.pointwise_lut to form silu(z)).

a streams on IFD0 and b on IFD1 (broadcast rows are streamed repeated); the
(N, blk) loop is a cmd-iteration nest (loop vars unused in the body) so the DPU
program stays compact regardless of N.
"""
from axelera.compiler.alto.axol import atomic
from axelera.compiler.alto.compiler.config import HardwareGeneration
from axelera.compiler.alto.compiler.device import PWORD
from axelera.compiler.alto.compiler.dtypes import MetisFloats
from axelera.compiler.alto.compiler.interpreter import emulation as emu
from axelera.compiler.alto.compiler.op_registry import Arg, BufferInfo, ScriptOp, register_op
import axelera.compiler.alto.script as A
import numpy as np
from tvm import tir

# mypy: disable-error-code="attr-defined, index, arg-type"
# flake8: noqa: C901


class ElementwiseMul(ScriptOp):
    """out = a * b with b full or per-row broadcast. See module docstring."""

    a = Arg(MetisFloats[tir.Buffer, "..."], description="(N, blk) left operand.")
    b = Arg(
        MetisFloats[tir.Buffer, "..."], description="(N, bblk) right operand; bblk in {1, blk}."
    )
    out = Arg(MetisFloats[tir.Buffer, "..."] | BufferInfo, default=None, is_output=True, kw=True)
    generation = Arg(HardwareGeneration | None, default=None, kw=True)

    def _n(self) -> int:
        return int(self.a.shape[0])

    def _blk(self) -> int:
        return int(self.a.shape[1])

    def _bblk(self) -> int:
        return int(self.b.shape[1])

    def validate(self) -> None:
        if int(self.b.shape[0]) != self._n():
            raise ValueError("a and b must share the leading dim N.")
        if self._bblk() not in (1, self._blk()):
            raise ValueError(
                f"b's blk dim must be 1 (broadcast) or {self._blk()}; got {self._bblk()}."
            )

    def infer_outputs(self) -> dict[str, BufferInfo]:
        return {"out": BufferInfo(shape=(self._n(), self._blk()), dtype="float32x64", scope=A.DDR)}

    def implementation(self) -> None:
        from axelera.compiler.alto.axol.common import ramp

        n, blk, bblk = self._n(), self._blk(), self._bblk()
        f32, f18 = "float32x64", "float18x64"

        # Staging the whole (n, blk) operands in L1 would pin ~2*n*blk of L1 (several MB
        # for large inputs), blowing the budget. Tile the row axis (A.tiles -> a symbolic
        # loop the memory pass sizes) so only one tile's (tile_rows, blk) is L1-resident.
        ain = A.alloc_buffer((n, blk), f32, scope=A.L1)
        bin = A.alloc_buffer((n, bblk), f32, scope=A.L1)
        oout = A.alloc_buffer((n, blk), f32, scope=A.L1)
        blk_pe = self.a.shape[1]  # PrimExpr (ramp needs a PrimExpr, not a Python int)
        bblk_pe = self.b.shape[1]

        with A.tiles(n) as t:
            at = ain.tile(t, dim=0)
            bt = bin.tile(t, dim=0)
            ot = oout.tile(t, dim=0)
            atomic.copy(self.a[t, ramp(blk_pe)], output=at)
            atomic.copy(self.b[t, ramp(bblk_pe)], output=bt)
            m = t.lanes
            with A.script():
                with A.Datapath(core=None):
                    fa = A.decl_stream(f32)
                    fb = A.decl_stream(f32)
                    o = A.decl_stream(f32)
                    for i in A.serial(m):
                        for j in A.serial(blk):
                            with A.IFD(num=0):
                                fa.push(at[i, j])
                    for i in A.serial(m):
                        for j in A.serial(blk):
                            with A.IFD(num=1):
                                # broadcast: read column 0 for every j when bblk == 1
                                fb.push(bt[i, j if bblk > 1 else 0])
                    with A.DPU():
                        ar = A.var(f18)
                        br = A.var(f18)
                        for i in A.serial(m):
                            for j in A.serial(blk):
                                ar = A.cast(fa.pop(), f18)
                                br = A.cast(fb.pop(), f18)
                                ar = ar * br
                                o.push(A.cast(ar, f32))
                    for i in A.serial(m):
                        for j in A.serial(blk):
                            with A.ODR():
                                ot[i, j] = o.pop()
            atomic.copy(ot, output=self.out[t, ramp(blk_pe)])

    def emulate(self) -> None:  # type: ignore[override]
        n, blk, bblk = self._n(), self._blk(), self._bblk()
        a = self.a.reshape(n, blk, PWORD).astype(np.float32)
        b = self.b.reshape(n, bblk, PWORD).astype(np.float32)
        if bblk == 1:
            b = np.repeat(b, blk, axis=1)
        out = emu.rf18(emu.rf18(a) * emu.rf18(b))
        np.copyto(self.out, out.astype(np.float32))


elementwise_mul = register_op(
    ElementwiseMul,
    "llm.elementwise_mul",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)

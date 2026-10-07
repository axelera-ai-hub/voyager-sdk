#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Pointwise LUT activation with an affine pre/post (llm.pointwise_lut).

Computes out[n, blk] = post * chained_lut(x[n, blk] + bias[n, blk]) on the DPU,
where the chained additive LUT realises an arbitrary scalar nonlinearity (fit on
the host via lut_chain.compute_chained_lut_params(..., func=...)).

This is the gated-DeltaNet decay front-end the layer needs to derive g and
beta from the in_proj activations on-device:

* beta = sigmoid(b)                    -> lut = sigmoid,  bias = 0,        post = 1
* g = -exp(A_log)*softplus(a+dt_bias) -> lut = softplus, bias = dt_bias (per-head,
                                         broadcast), post = -exp(A_log) (per-head)

x streams on IFD0 and bias on IFD1 (a DPU datapath that reads only IFD0 +
a chained LUT miscompiles, and the second stream is needed for the affine anyway).
The (N, blk) loop is a cmd-iteration nest (loop vars unused in the body) so the
DPU program stays compact regardless of N.
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


class PointwiseLut(ScriptOp):
    """out = post * chained_lut(x + bias) elementwise. See module docstring."""

    x = Arg(MetisFloats[tir.Buffer, "..."], description="(N, blk) input values.")
    bias = Arg(
        MetisFloats[tir.Buffer, "..."], description="(N, blk) additive pre-bias (broadcast)."
    )
    lut = Arg(MetisFloats[tir.Buffer, "..."], description="(n_luts,) chained additive LUT params.")
    post = Arg(MetisFloats[tir.Buffer, "..."], description="(1,) post-multiply scalar.")
    out = Arg(MetisFloats[tir.Buffer, "..."] | BufferInfo, default=None, is_output=True, kw=True)
    generation = Arg(HardwareGeneration | None, default=None, kw=True)

    def _n(self) -> int:
        return int(self.x.shape[0])

    def _blk(self) -> int:
        return int(self.x.shape[1])

    def validate(self) -> None:
        if int(self.bias.shape[0]) != self._n() or int(self.bias.shape[1]) != self._blk():
            raise ValueError("bias shape must match x.")

    def infer_outputs(self) -> dict[str, BufferInfo]:
        return {"out": BufferInfo(shape=(self._n(), self._blk()), dtype="float32x64", scope=A.DDR)}

    def implementation(self) -> None:
        from axelera.compiler.alto.axol.common import ramp

        n, blk = self._n(), self._blk()
        f32, f18 = "float32x64", "float18x64"
        n_luts = int(self.lut.shape[0])

        # lut/post are tiny and tile-invariant: stage them once. x/bias/out are (n, blk) --
        # staging them whole would pin 2*(n*blk) of L1 (several MB for large inputs),
        # blowing the L1 budget. Tile the row axis (A.tiles -> a symbolic loop the memory
        # pass sizes) so only one tile's (tile_rows, blk) of x/bias/out is L1-resident.
        lut = A.alloc_buffer((n_luts,), f32, scope=A.L1)
        post = A.alloc_buffer((1,), f32, scope=A.L1)
        atomic.copy(self.lut, output=lut)
        atomic.copy(self.post, output=post)
        xin = A.alloc_buffer((n, blk), f32, scope=A.L1)
        bin = A.alloc_buffer((n, blk), f32, scope=A.L1)
        oout = A.alloc_buffer((n, blk), f32, scope=A.L1)

        blk_pe = self.x.shape[1]  # PrimExpr (ramp needs a PrimExpr, not a Python int)
        with A.tiles(n) as t:
            xt = xin.tile(t, dim=0)
            bt = bin.tile(t, dim=0)
            ot = oout.tile(t, dim=0)
            atomic.copy(self.x[t, ramp(blk_pe)], output=xt)
            atomic.copy(self.bias[t, ramp(blk_pe)], output=bt)
            m = t.lanes
            with A.script():
                with A.Datapath(core=None):
                    fx = A.decl_stream(f32)
                    fb = A.decl_stream(f32)
                    o = A.decl_stream(f32)
                    for i in A.serial(m):
                        for j in A.serial(blk):
                            with A.IFD(num=0):
                                fx.push(xt[i, j])
                    for i in A.serial(m):
                        for j in A.serial(blk):
                            with A.IFD(num=1):
                                fb.push(bt[i, j])
                    with A.DPU():
                        la = A.var(f18)
                        xr = A.var(f18)
                        br = A.var(f18)
                        s = A.var(f18)
                        r = A.var(f18)
                        for i in A.serial(m):
                            for j in A.serial(blk):
                                xr = A.cast(fx.pop(), f18)
                                br = A.cast(fb.pop(), f18)
                                s = xr + br
                                r = tir_ops.lut(s, A.cast(lut[0], f18))
                                for k in A.serial(1, n_luts):
                                    la = tir_ops.lut(s, A.cast(lut[k], f18))
                                    r = r + la
                                r = r * A.cast(post[0], f18)
                                o.push(A.cast(r, f32))
                    for i in A.serial(m):
                        for j in A.serial(blk):
                            with A.ODR():
                                ot[i, j] = o.pop()
            atomic.copy(ot, output=self.out[t, ramp(blk_pe)])

    def emulate(self) -> None:  # type: ignore[override]
        n, blk = self._n(), self._blk()
        x = self.x.reshape(n, blk, PWORD).astype(np.float32)
        bias = self.bias.reshape(n, blk, PWORD).astype(np.float32)
        lut = self.lut.reshape(int(self.lut.shape[0]), PWORD).astype(np.float32)
        post = float(self.post.reshape(-1)[0])

        s = emu.rf18(emu.rf18(x) + emu.rf18(bias))
        flat = s.reshape(1, -1)
        acc = emu.apply_lut(flat, lut[0:1])
        for k in range(1, lut.shape[0]):
            acc = acc + emu.apply_lut(flat, lut[k : k + 1])
        out = emu.rf18(acc.reshape(n, blk, PWORD) * emu.rf18(np.float32(post)))
        np.copyto(self.out, out.astype(np.float32))


pointwise_lut = register_op(
    PointwiseLut,
    "llm.pointwise_lut",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)

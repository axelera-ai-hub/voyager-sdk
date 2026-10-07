#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Fused squared-ReLU + per-channel activation quantize (llm.relu2_quantize).

out[i, j] = round_clip_int8(max(x[i, j], 0)**2 * scale[j] + zero_point[j])

The squared-ReLU MLP's whole middle section in ONE DPU datapath: the Nemotron
(Domyn Small) block is ``down_proj(relu(up_proj(x))**2)``, so between the two
matmuls sits a relu, a square, and the down-projection's input quantize. Spelled
with the atomic ops that would be three passes over the intermediate activation
(``maximum`` -> ``mul`` -> ``llm.quantize``), and the intermediate is the widest
tensor in the layer -- at hidden 4096 / intermediate 16384, prefilling 2048
positions writes 128 MiB per pass. Fusing drops that to one read of the fp
intermediate and one int8 write.

relu(x)**2 is computed exactly (a max and a multiply in DPU registers), NOT via a
LUT: unlike SiLU it never saturates, so a fixed-domain piecewise-linear fit has no
good outer bin -- and the exact form costs two DPU ops.

Layout matches llm.quantize, whose implementation this mirrors: x is flattened to
(N, C_blocks) by the caller (a free view) and ``qparams`` is the PRE-INTERLEAVED
(2*C_blocks,) [scale[0], zp[0], scale[1], zp[1], ...] buffer -- the device cannot do
a strided interleave. x streams on IFD0, qparams on IFD1 (never resident, so the op
carries no per-channel register arrays). The row axis is L1-tiled so an oversized
activation never pins all of L1.
"""
from axelera.compiler.alto.axol import atomic
from axelera.compiler.alto.axol.common import ramp
from axelera.compiler.alto.compiler.config import HardwareGeneration
from axelera.compiler.alto.compiler.device import PWORD
from axelera.compiler.alto.compiler.dtypes import MetisFloats
from axelera.compiler.alto.compiler.interpreter import emulation as emu
from axelera.compiler.alto.compiler.op_registry import Arg, BufferInfo, ScriptOp, register_op
import axelera.compiler.alto.script as A
import numpy as np
from tvm import tir

# mypy: disable-error-code="attr-defined, index, arg-type"


class Relu2Quantize(ScriptOp):
    """out = round_clip_int8(relu(x)**2 * scale + zero_point), per-channel.

    See the module docstring for the layout contract (identical to llm.quantize).
    """

    x = Arg(MetisFloats[tir.Buffer, "..."], description="(N, C_blocks) activation, fp16/fp32.")
    qparams = Arg(
        MetisFloats[tir.Buffer, "..."],
        description=(
            "(2*C_blocks,) interleaved per-channel [scale[0], zp[0], scale[1], zp[1], ...]."
        ),
    )
    out = Arg(MetisFloats[tir.Buffer, "..."] | BufferInfo, default=None, is_output=True, kw=True)
    generation = Arg(HardwareGeneration | None, default=None, kw=True)

    def _n(self) -> int:
        return int(self.x.shape[0])

    def _c(self) -> int:
        return int(self.x.shape[1])

    def validate(self) -> None:
        c = self._c()
        n_qp = int(np.prod([int(s) for s in self.qparams.shape]))
        if n_qp != 2 * c:
            raise ValueError(
                f"qparams must have 2*{c} = {2 * c} PWORD blocks (interleaved scale,zp per "
                f"channel block of x), got shape {tuple(int(s) for s in self.qparams.shape)}."
            )

    def infer_outputs(self) -> dict[str, BufferInfo]:
        return {"out": BufferInfo(shape=(self._n(), self._c()), dtype="int8x64", scope=A.DDR)}

    def implementation(self) -> None:
        n, c = self._n(), self._c()
        f18 = "float18x64"
        xdt, sdt = self.x.dtype, self.qparams.dtype
        c_pe = self.x.shape[1]

        xin = A.alloc_buffer((n, c), xdt, scope=A.L1)
        oout = A.alloc_buffer((n, c), "int8x64", scope=A.L1)
        # Already interleaved by the layer binding: stage contiguously in L1 (no strided
        # DMA) and stream linearly, popping (scale, zp) per channel.
        params_l1 = atomic.copy(self.qparams, scope=A.L1)
        # relu = prelu(x, 0): the DPU's native leaky-relu with a zero negative slope.
        # The coefficient rides in as a one-PWORD DDR constant read as a broadcast
        # scalar inside the DPU block (both IFDs are taken by x and qparams), the same
        # way axol.atomic.prelu feeds its weight.
        zero_const = A.alloc_constant(
            np.zeros((1, PWORD), dtype=np.float32), dtype="float32x64", scope=A.DDR
        )

        with A.tiles(n) as t:
            xt = xin.tile(t, dim=0)
            ot = oout.tile(t, dim=0)
            atomic.copy(self.x[t, ramp(c_pe)], output=xt)
            m = t.lanes
            with A.script():
                # One Datapath PER ROW (as in llm.quantize): each invocation streams only
                # this row's c values + the 2c interleaved params, so the IFD FIFO never
                # accumulates a whole tile (which corrupts the stream on device).
                for i in A.serial(m):
                    with A.Datapath(core=None):
                        fx = A.decl_stream(xdt)
                        fp = A.decl_stream(sdt)
                        o = A.decl_stream("int8x64")
                        for j in A.serial(c):
                            with A.IFD(num=0):
                                fx.push(xt[i, j])
                        for k in A.serial(2 * c):
                            with A.IFD(num=1):
                                fp.push(params_l1[k])
                        with A.DPU():
                            xr = A.var(f18)
                            sr = A.var(f18)
                            zr = A.var(f18)
                            for j in A.serial(c):
                                xr = A.cast(fx.pop(), f18)
                                sr = A.cast(fp.pop(), f18)
                                zr = A.cast(fp.pop(), f18)
                                xr = A.dpu.ops.prelu(xr, A.cast(zero_const[0], f18))
                                xr = xr * xr  # relu(x)**2
                                xr = xr * sr
                                xr = xr + zr
                                o.push(A.cast(xr, "int8x64"))  # round + clip to [-128, 127]
                        for j in A.serial(c):
                            with A.ODR():
                                ot[i, j] = o.pop()
            atomic.copy(ot, output=self.out[t, ramp(c_pe)])

    def emulate(self) -> None:  # type: ignore[override]
        n, c = self._n(), self._c()
        x = emu.rf18(self.x.reshape(n, c, PWORD).astype(np.float32))
        qp = self.qparams.reshape(2 * c, PWORD).astype(np.float32)
        s = emu.rf18(qp[0::2])[None]  # even = scale
        z = emu.rf18(qp[1::2])[None]  # odd = zero_point
        r = np.maximum(x, np.float32(0.0))
        r = emu.rf18(r * r)
        q = emu.rf18(emu.rf18(r * s) + z)
        np.copyto(self.out, np.clip(np.round(q), -128, 127).astype(np.int8))


relu2_quantize = register_op(
    Relu2Quantize,
    "llm.relu2_quantize",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)

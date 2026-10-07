#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Fused per-channel activation quantize (llm.quantize).

out[i, j] = round_clip_int8(x[i, j] * scale[j] + zero_point[j])

One DPU datapath replacing the kernel's mul(scale) -> add(zp) -> dpu_cast(int8)
chain (3 passes over the activation when datapath fusion is off -- which it is,
being buggy). scale/zero_point are per-channel (one value per C block, broadcast
over the rows) and are streamed via IFD1 + popped per output -- never resident, so
the op carries no per-channel register arrays (the pattern quantized_matmul_activation
uses). x streams on IFD0. The row axis is L1-tiled (like llm.elementwise_mul) so an
oversized activation never pins all of L1.

Input is flattened to (N, C_blocks): the caller reshapes its (*leading, C_blocks)
activation to 2D (a free view) and reshapes the int8 result back.
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


class Quantize(ScriptOp):
    """out = round_clip_int8(x * scale + zero_point), per-channel. See module docstring.

    Like quantized_matmul_activation, the per-channel (scale, zero_point) are supplied
    PRE-INTERLEAVED by the layer binding as one buffer qparams of shape (2*C_blocks,):
    qparams[2j] = scale[j], qparams[2j+1] = zero_point[j]. The op streams it linearly and
    pops (scale, zp) per channel -- no on-device interleave (a strided DMA does not lower
    correctly on the device).
    """

    x = Arg(MetisFloats[tir.Buffer, "..."], description="(N, C_blocks) activation, fp16/fp32.")
    qparams = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(2*C_blocks,) interleaved per-channel [scale[0], zp[0], scale[1], zp[1], ...].",
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
        # qparams is ALREADY interleaved [scale[0], zp[0], scale[1], zp[1], ...] by the layer
        # binding -- stage it contiguously in L1 (no strided DMA) and stream it linearly,
        # popping (scale, zp) per channel. Matches quantized_matmul_activation's param feed.
        params_l1 = atomic.copy(self.qparams, scope=A.L1)

        with A.tiles(n) as t:
            xt = xin.tile(t, dim=0)
            ot = oout.tile(t, dim=0)
            atomic.copy(self.x[t, ramp(c_pe)], output=xt)
            m = t.lanes
            with A.script():
                # One Datapath PER ROW (like quantized_matmul_activation): each invocation
                # streams only this row's c values + the 2c interleaved params, so the IFD
                # FIFO never accumulates a whole tile (which corrupts the stream on device).
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
        q = emu.rf18(emu.rf18(x * s) + z)
        np.copyto(self.out, np.clip(np.round(q), -128, 127).astype(np.int8))


quantize = register_op(
    Quantize,
    "llm.quantize",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)

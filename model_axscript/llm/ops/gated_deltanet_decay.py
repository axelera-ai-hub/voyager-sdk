#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""On-device gated-DeltaNet decay quantities (llm.gated_deltanet_decay).

Computes, on the DPU, the per-chunk decay tensors the int8 IMC DeltaNet kernels
(gated_deltanet_chunk_imc_multi / gated_deltanet_decode_imc) otherwise
take host-precomputed -- so the gated-DeltaNet decoder layer can derive them from
the on-device in_proj activations rather than the host.

Given the per-step log-decay g (g_t = -exp(A_log)*softplus(a_t + dt_bias);
the softplus front-end is layered on separately) it produces, for one chunk of
C timesteps (one (head, chunk) of a prefill)::

    gc            = cumsum(g)                       # within-chunk cumulative decay
    decay_mask    = tril(exp(gc_i - gc_j))          # (C, C), packed (C, nblk)
    nsl           = -strict_lower(exp(gc_i - gc_j))  # (C, C), packed (C, nblk)
    vec_out[0::3] = exp(gc)                          # gexp        (C,)
    vec_out[1::3] = exp(gc_last - gc)                # glast_decay (C,)
    vec_out[2::3] = exp(gc_last)                     # decaylast   (scalar, broadcast)

C is any multiple of PWORD (nblk = C / PWORD 64-lane column blocks);
the Qwen3.5-2B prefill kernel ties C = Dk = Dv = 128 (nblk = 2).  The C x C
decay_mask / nsl are packed (C, nblk) (block blk holds columns
blk*64 .. blk*64+63 in its lanes); gexp / glast_decay / decaylast
are per-row scalars (broadcast across lanes) and vec_out interleaves the three.

Mechanics / hardware constraints baked into the datapath structure:

* cumulative sum is a sequential DPU scan;
* vector_transpose(gc, 0, 1) yields gl (PWORD, nblk) with gl[*, blk] the
  lane-vector gc[blk*64 .. blk*64+63] (the same for every output row), so the
  outer difference gc_i - gc_j for column block blk is gl[0, blk] - gc_i;
* every exp is the DPU LUT (fit_exp_lut_params_sdpa_softmax parameterises
  exp(-gap) for gap >= 0, so each output feeds the positive gap);
* each datapath uses at most two input streams, one per physical IFD, and a single
  ODR with a single store; exp(gc_i - gc_j) is produced unmasked in its own
  datapath and the tril / strict-lower 0/1 host masks are applied per-element.
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


class GatedDeltaNetDecay(ScriptOp):
    """On-device decay quantities for one chunk/head. See module docstring."""

    g = Arg(
        MetisFloats[tir.Buffer, "..."], description="(C, 1) per-step log-decay, broadcast per row."
    )
    tril = Arg(
        MetisFloats[tir.Buffer, "..."], description="(C, nblk) 0/1 lower-incl-diagonal mask."
    )
    slt = Arg(MetisFloats[tir.Buffer, "..."], description="(C, nblk) 0/1 strict-lower mask.")
    exp_lut = Arg(
        MetisFloats[tir.Buffer, "..."], description="(n_luts,) exp(-gap) chained LUT params."
    )
    decay_mask = Arg(
        MetisFloats[tir.Buffer, "..."] | BufferInfo, default=None, is_output=True, kw=True
    )
    nsl_out = Arg(
        MetisFloats[tir.Buffer, "..."] | BufferInfo, default=None, is_output=True, kw=True
    )
    vec_out = Arg(
        MetisFloats[tir.Buffer, "..."] | BufferInfo, default=None, is_output=True, kw=True
    )
    generation = Arg(HardwareGeneration | None, default=None, kw=True)

    def _c(self) -> int:
        return int(self.g.shape[0])

    def _nblk(self) -> int:
        return self._c() // PWORD

    def validate(self) -> None:
        if self._c() % PWORD != 0:
            raise ValueError(f"C must be a multiple of {PWORD}; got {self._c()}.")

    def infer_outputs(self) -> dict[str, BufferInfo]:
        c, nblk = self._c(), self._nblk()
        return {
            # C x C decay matrices packed into nblk 64-lane column blocks
            "decay_mask": BufferInfo(shape=(c, nblk), dtype="float32x64", scope=A.DDR),
            "nsl_out": BufferInfo(shape=(c, nblk), dtype="float32x64", scope=A.DDR),
            # interleaved [gexp_i, glast_decay_i, decaylast_i] for a single-store ODR
            "vec_out": BufferInfo(shape=(3 * c, 1), dtype="float32x64", scope=A.DDR),
        }

    def _mul_mask(self, src, mask, negate, out) -> None:
        """out[i,blk] = (-1 if negate) * src[i,blk] * mask[i,blk] (src IFD0, mask IFD1)."""
        c, nblk = self._c(), self._nblk()
        f32, f18 = "float32x64", "float18x64"
        with A.script():
            with A.Datapath(core=None):
                fs = A.decl_stream(f32)
                fm = A.decl_stream(f32)
                o = A.decl_stream(f32)
                for i in A.serial(c):
                    for blk in A.serial(nblk):
                        with A.IFD(num=0):
                            fs.push(src[i, blk])
                for i in A.serial(c):
                    for blk in A.serial(nblk):
                        with A.IFD(num=1):
                            fm.push(mask[i, blk])
                with A.DPU():
                    r = A.var(f18)
                    for i in A.serial(c):
                        for blk in A.serial(nblk):
                            r = A.cast(fs.pop(), f18)
                            r = r * A.cast(fm.pop(), f18)
                            if negate:
                                r = r * A.cast(-1.0, f18)
                            o.push(A.cast(r, f32))
                for i in A.serial(c):
                    for blk in A.serial(nblk):
                        with A.ODR():
                            out[i, blk] = o.pop()

    def implementation(self) -> None:
        c, nblk = self._c(), self._nblk()
        f32 = "float32x64"
        f18 = "float18x64"
        n_luts = int(self.exp_lut.shape[0])

        def L1(cols=1):
            return A.alloc_buffer((c, cols), f32, scope=A.L1)

        gin = L1()
        tril = L1(nblk)
        slt = L1(nblk)
        lut = A.alloc_buffer((n_luts,), f32, scope=A.L1)
        gc = L1()
        gl = A.alloc_buffer((PWORD, nblk), f32, scope=A.L1)
        gclast = L1()
        dm_full = L1(nblk)
        atomic.copy(self.g, output=gin)
        atomic.copy(self.tril, output=tril)
        atomic.copy(self.slt, output=slt)
        atomic.copy(self.exp_lut, output=lut)

        # (1) cumsum g -> gc (sequential scan)
        with A.script():
            with A.Datapath(core=None):
                fi = A.decl_stream(f32)
                o = A.decl_stream(f32)
                for i in A.serial(c):
                    with A.IFD(num=0):
                        fi.push(gin[i, 0])
                with A.DPU():
                    acc = A.var(f18)
                    gi = A.var(f18)
                    acc = A.cast(fi.pop(), f18)
                    o.push(A.cast(acc, f32))
                    for i in A.serial(1, c):
                        gi = A.cast(fi.pop(), f18)
                        acc = acc + gi
                        o.push(A.cast(acc, f32))
                for i in A.serial(c):
                    with A.ODR():
                        gc[i, 0] = o.pop()
        # (2) gc (per-row broadcast) -> gl (PWORD, nblk); gl[*, blk] = gc[blk*64 .. +63] as lanes.
        atomic.vector_transpose(gc, 0, 1, output=gl)
        # (3) broadcast gc_last (= gc[C-1]) to all rows
        with A.script():
            with A.Datapath(core=None):
                fi = A.decl_stream(f32)
                o = A.decl_stream(f32)
                with A.IFD(num=0):
                    fi.push(gc[c - 1, 0])
                with A.DPU():
                    v = A.var(f18)
                    v = A.cast(fi.pop(), f18)
                    for i in A.serial(c):
                        o.push(A.cast(v, f32))
                for i in A.serial(c):
                    with A.ODR():
                        gclast[i, 0] = o.pop()

        # (4) dm_full[i, blk] = exp(gc_i - gc_j); gap = gl[0,blk] - gc_i (gc IFD0, gl IFD1).
        # Nested (i, blk) cmd-iteration loop (loop vars unused in the DPU body so the loop
        # iterates the streams rather than unrolling): gc streamed nblk times, gl tiled
        # across rows -- a single per-element program fits DPU instruction memory regardless
        # of C.
        with A.script():
            with A.Datapath(core=None):
                fg = A.decl_stream(f32)
                fgl = A.decl_stream(f32)
                o = A.decl_stream(f32)
                for i in A.serial(c):
                    for blk in A.serial(nblk):
                        with A.IFD(num=0):
                            fg.push(gc[i, 0])
                for i in A.serial(c):
                    for blk in A.serial(nblk):
                        with A.IFD(num=1):
                            fgl.push(gl[0, blk])
                with A.DPU():
                    la = A.var(f18)
                    gcr = A.var(f18)
                    glr = A.var(f18)
                    gap = A.var(f18)
                    dm = A.var(f18)
                    for i in A.serial(c):
                        for blk in A.serial(nblk):
                            gcr = A.cast(fg.pop(), f18)
                            glr = A.cast(fgl.pop(), f18)
                            gap = glr - gcr
                            dm = tir_ops.lut(gap, A.cast(lut[0], f18))
                            for k in A.serial(1, n_luts):
                                la = tir_ops.lut(gap, A.cast(lut[k], f18))
                                dm = dm + la
                            o.push(A.cast(dm, f32))
                for i in A.serial(c):
                    for blk in A.serial(nblk):
                        with A.ODR():
                            dm_full[i, blk] = o.pop()
        # (5) decay_mask = tril * dm_full ; (6) nsl = -strict_lower * dm_full
        self._mul_mask(dm_full, tril, False, self.decay_mask)
        self._mul_mask(dm_full, slt, True, self.nsl_out)

        # (7) vec_out: gexp = exp(gc); glast_decay = exp(gc_last - gc); decaylast = exp(gc_last)
        with A.script():
            with A.Datapath(core=None):
                fg = A.decl_stream(f32)
                fcl = A.decl_stream(f32)
                o = A.decl_stream(f32)
                for i in A.serial(c):
                    with A.IFD(num=0):
                        fg.push(gc[i, 0])
                for i in A.serial(c):
                    with A.IFD(num=1):
                        fcl.push(gclast[i, 0])
                with A.DPU():
                    la = A.var(f18)
                    gcr = A.var(f18)
                    clr = A.var(f18)
                    zero = A.var(f18)
                    gap = A.var(f18)
                    res = A.var(f18)
                    for i in A.serial(c):
                        gcr = A.cast(fg.pop(), f18)
                        clr = A.cast(fcl.pop(), f18)
                        zero = A.cast(0.0, f18)
                        gap = zero - gcr  # gexp: gap = -gc_i
                        res = tir_ops.lut(gap, A.cast(lut[0], f18))
                        for k in A.serial(1, n_luts):
                            la = tir_ops.lut(gap, A.cast(lut[k], f18))
                            res = res + la
                        o.push(A.cast(res, f32))
                        gap = gcr - clr  # glast_decay: gap = gc_i - gc_last
                        res = tir_ops.lut(gap, A.cast(lut[0], f18))
                        for k in A.serial(1, n_luts):
                            la = tir_ops.lut(gap, A.cast(lut[k], f18))
                            res = res + la
                        o.push(A.cast(res, f32))
                        gap = zero - clr  # decaylast: gap = -gc_last
                        res = tir_ops.lut(gap, A.cast(lut[0], f18))
                        for k in A.serial(1, n_luts):
                            la = tir_ops.lut(gap, A.cast(lut[k], f18))
                            res = res + la
                        o.push(A.cast(res, f32))
                for j in A.serial(3 * c):
                    with A.ODR():
                        self.vec_out[j, 0] = o.pop()

    def emulate(self) -> None:  # type: ignore[override]
        c, nblk = self._c(), self._nblk()
        g = self.g[:, 0, 0].astype(np.float32)  # (C,) broadcast scalar per row
        tril = self.tril[:, :, :].astype(np.float64).reshape(c, nblk * PWORD)[:, :c]
        slt = self.slt[:, :, :].astype(np.float64).reshape(c, nblk * PWORD)[:, :c]
        lut = self.exp_lut.reshape(int(self.exp_lut.shape[0]), PWORD).astype(np.float32)

        def expf(gap):  # exp(-gap), gap >= 0, via the additive chained LUT
            gap = emu.rf18(np.asarray(gap, np.float32))
            acc = emu.apply_lut(gap.reshape(1, -1), lut[0:1])
            for k in range(1, lut.shape[0]):
                acc = acc + emu.apply_lut(gap.reshape(1, -1), lut[k : k + 1])
            return acc.reshape(gap.shape)

        gc = np.cumsum(g).astype(np.float64)  # (C,)
        gap_dm = gc[None, :] - gc[:, None]  # (C, C): gc_j - gc_i
        dm = expf(np.maximum(gap_dm, 0.0).reshape(-1)).reshape(c, c)
        decay_mask = dm * tril
        nsl = -(dm * slt)
        gexp = expf(-gc)  # (C,)
        gc_last = gc[c - 1]
        glast_decay = expf(gc - gc_last)
        decaylast = np.full(c, float(expf(np.array([-gc_last]))[0]))

        np.copyto(self.decay_mask, decay_mask.reshape(c, nblk, PWORD).astype(np.float32))
        np.copyto(self.nsl_out, nsl.reshape(c, nblk, PWORD).astype(np.float32))
        vec = np.zeros((3 * c, 1, PWORD), np.float32)
        for j, v in enumerate((gexp, glast_decay, decaylast)):
            vec[j::3, 0, :] = v.reshape(c, 1)
        np.copyto(self.vec_out, vec)


gated_deltanet_decay = register_op(
    GatedDeltaNetDecay,
    "llm.gated_deltanet_decay",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)

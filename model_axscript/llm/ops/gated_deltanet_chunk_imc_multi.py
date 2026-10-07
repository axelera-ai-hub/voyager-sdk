#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Multi-chunk int8 IMC chunkwise gated delta rule with state carry
(llm.gated_deltanet_chunk_imc_multi).

Full prefill over T = NC * C tokens for a BH grid of independent (head, batch)
sequences: walks NC chunks per head, carrying the (Dk, Dv) recurrent state S in
float32 between chunks. Per chunk (g the within-chunk cumulative log-decay, all
decay factors host-precomputed)::

    L     = -strict_lower((kβ @ kᵀ) ⊙ decay_mask)
    T_wy  = (I - L)⁻¹ ≈ I + L + … + L^NTERMS                  (Neumann series)
    value = T_wy @ (v ⊙ β)
    k_cd  = T_wy @ (kβ ⊙ exp g)                               (intra)
    v_new = value - k_cd @ S                                  (inter: reads S)
    out   = (q ⊙ exp g) @ S  +  ((q @ kᵀ) ⊙ decay_mask) @ v_new
    S     = exp(g_last) · S  +  (k ⊙ exp(g_last - g))ᵀ @ v_new   (state update)

The carried state is initialised from s0 and emitted as state_out, so a long
prefill can be chained across calls (state_out -> s0) and handed off to the decode
kernel; pass s0 = 0 for a fresh prefill.

Every dense matmul is an int8 IMC matmul (load_imc + MVMEXE + mvm_1x1). All B
operands are reshape views in natural row layout, no transpose -- except k_decᵀ which
the host supplies pre-transposed. C = Dk = Dv is any multiple of PWORD (nblk = C/PWORD
blocks; the production head dim 128 = two blocks). Quantization uses per-column dynamic
scales (absmax/127 per column) applied identically here and in the float18-aligned emulate,
so compile_and_run matches interpret. int8 error is chunk-local (the carried state stays
float32): measured ~0.1% vs the float64 recurrence, flat across chunks. Per-chunk inputs
are stacked on a leading NC axis and precomputed by the host (q, kβ, kᵀ, vβ, q⊙exp g,
kβ⊙exp g, k_decᵀ, decay_mask, nsl = -strict_lower·decay_mask, exp(g_last)).
"""
from axelera.compiler.alto.axol import atomic
from axelera.compiler.alto.axol.utils import to_tuple
from axelera.compiler.alto.compiler.config import HardwareGeneration
from axelera.compiler.alto.compiler.device import PWORD
from axelera.compiler.alto.compiler.dtypes import MetisFloats
from axelera.compiler.alto.compiler.interpreter import emulation as emu
from axelera.compiler.alto.compiler.op_registry import Arg, BufferInfo, ScriptOp, register_op
import axelera.compiler.alto.script as A
import axelera.compiler.backend.tir.ops as tir_ops
import numpy as np
from tvm import tir

from .pointwise_lut import pointwise_lut

# mypy: disable-error-code="attr-defined, index, arg-type"
# flake8: noqa: C901

# Quantization scheme (~0.1% chunk rel-L2 vs the fp64 recurrence):
#   * the WY inverse (I-L)^-1 is computed by an EXACT fp18 forward substitution (NOT a
#     Neumann series -- on real activations ||L||~8 so Neumann diverges); see _fwd_solve.
#   * every other GEMM is residual int8 (hi+lo -> ~int16) with per-column dynamic scales:
#     A quantized per-row (scale = rowmax|A|/127), B per-column (colmax|B|/127), product
#     dequantized by the outer product sa[m]*sb[n] post-int32-accumulation. Per-column
#     handles value-channel outliers that a single fixed scale cannot.
_RESID = 256.0  # residual-int8 lo-byte factor: x ~ (hi + lo/256)*s

# qc constant-buffer indices (folded float constants are mis-DMA'd on AICORE_SIM, so the
# fixed factors are read from an input buffer -- see build_qc()).
_I_127, _I_127INV, _I_256, _I_256Q, _I_TWO = 0, 1, 2, 3, 4
_N_QC = 5


def build_qc():
    """The (_N_QC,) constant factors the chunk op reads from its qc input buffer:
    1/127 (dequant), 127 (quant), 1/256 (residual hi-recombine), 256 (residual lo), 2.0 (NR)."""
    v = np.zeros(_N_QC, np.float32)
    v[_I_127], v[_I_127INV], v[_I_256], v[_I_256Q], v[_I_TWO] = 1 / 127, 127.0, 1 / 256, 256.0, 2.0
    return v


def build_abs_lut(n_luts, hi=256.0):
    """(n_luts, PWORD) chained LUT for |x| over [-hi, hi]. abs is exactly piecewise-linear,
    so the fit is exact and extrapolates exactly past hi (the top bin's line)."""
    from ..models.qwen3_5.deltanet_impl import _chained_lut  # lazy: avoid circular import

    return _chained_lut(np.abs, -hi, hi, n_luts).astype(np.float32)


def build_reciprocal_estimate_luts():
    """(3, PWORD) full-range geometric minimax 1/x initial-estimate LUTs (refined on-device
    by 3 Newton-Raphson steps). Bins span [sqrt(smallest_f18), sqrt(largest_f18)] -> the
    reciprocal is calibration-free (no per-operand range tuning)."""
    import math

    from tvm.relay.backend.contrib.axelera.utils.float18 import Float18Params

    num_luts, lut_bins = 3, 16
    mn, mx = math.sqrt(Float18Params.smallest_value), math.sqrt(Float18Params.largest_value)
    rng = (mx / mn) ** (1 / (num_luts * lut_bins - num_luts))
    lim = [mn]
    slp = [Float18Params.largest_value]
    off = [Float18Params.largest_value]
    b0 = mn
    while b0 < mx:
        if len(lim) % lut_bins == 0:
            lim.append(b0)
            slp.append(0)
            off.append(0)
        b1 = min(b0 * rng, mx)
        b = -8 / (b0**2 + 6 * b0 * b1 + b1**2)
        lim.append(b1)
        slp.append(b)
        off.append(-b * (b0 + b1))
        b0 = b1
        if b1 >= mx:
            break
    luts = [np.zeros((PWORD,), np.float32) for _ in range(num_luts)]
    for i in range(num_luts):
        luts[i][0:16] = lim[i * 16 : i * 16 + 16]
        luts[i][16:32] = slp[i * 16 : i * 16 + 16]
        luts[i][32:48] = off[i * 16 : i * 16 + 16]
    return np.array(luts).reshape((num_luts, PWORD)).astype(np.float32)


class GatedDeltaNetChunkImcMulti(ScriptOp):
    """Multi-chunk int8 IMC chunkwise gated delta rule with float32 state carry."""

    q = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(BH, NC, C, nblk) query rows per (head, chunk).",
    )
    kb = Arg(MetisFloats[tir.Buffer, "..."], description="(BH, NC, C, nblk) k*beta rows.")
    knk = Arg(
        MetisFloats[tir.Buffer, "..."], description="(BH, NC, C, nblk) kᵀ rows ([d]=k[:,d])."
    )
    vb = Arg(MetisFloats[tir.Buffer, "..."], description="(BH, NC, C, nblk) v*beta rows.")
    qg = Arg(MetisFloats[tir.Buffer, "..."], description="(BH, NC, C, nblk) q*exp(g) rows.")
    kbg = Arg(MetisFloats[tir.Buffer, "..."], description="(BH, NC, C, nblk) k*beta*exp(g) rows.")
    kdecT = Arg(
        MetisFloats[tir.Buffer, "..."], description="(BH, NC, C, nblk) (k*exp(g_last-g))ᵀ."
    )
    dm = Arg(MetisFloats[tir.Buffer, "..."], description="(BH, NC, C, nblk) decay_mask rows.")
    nsl = Arg(
        MetisFloats[tir.Buffer, "..."], description="(BH, NC, C, nblk) -strict_lower(decay_mask)."
    )
    decaylast = Arg(
        MetisFloats[tir.Buffer, "..."], description="(BH, NC, C, nblk) exp(g_last) broadcast."
    )
    s0 = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(BH, Dk, nblk) entry state per head (Dv in lanes).",
    )
    abs_lut = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(NL,) chained-LUT for |x| (exact 2-piece PWL); used to get per-operand absmax.",
    )
    est_lut = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(3,) full-range reciprocal initial-estimate LUTs (refined by Newton-Raphson). "
        "Calibration-free; see build_reciprocal_estimate_luts().",
    )
    zero = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(C, 2*nblk) zeros: fwd-solve X init, abs-LUT bias, recip bias (sliced).",
    )
    one = Arg(MetisFloats[tir.Buffer, "..."], description="(1,) ones (pointwise_lut post-scale).")
    qc_col = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(C, _N_QC) fixed quant factors broadcast down rows (per-row (C,1) slices "
        "for the a/per-row path); see build_qc(). Buffer, not folded constants (sim-safe).",
    )
    qc_row = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(_N_QC, nblk) fixed quant factors broadcast across cols (per-col (1,nblk) "
        "slices for the b/per-col path + dequant).",
    )
    qc1 = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(_N_QC,) 1-D copy of the quant factors, for DPU direct-access (the 256 "
        "residual-lo factor in _quant_resid; DPU direct-access buffers must be 1-D).",
    )
    output = Arg(
        MetisFloats[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="Per-(head,chunk) readout, auto-inferred (BH, NC, C, nblk) float32x64 DDR.",
    )
    state_out = Arg(
        MetisFloats[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="Final state per head after NC chunks, auto-inferred (BH, Dk, nblk) float32x64 DDR.",
    )
    state_inplace = Arg(
        bool,
        default=False,
        kw=True,
        description="Write the final state back into the s0 buffer in place (one op, "
        "read-then-write per head -- the persistent-cache RMW pattern, cf. causal_conv1d) "
        "and emit no separate state_out. Lets a decoder layer carry the recurrent state "
        "across calls without a second graph output.",
    )
    generation = Arg(HardwareGeneration | None, default=None, kw=True)

    def _bh(self) -> int:
        return int(self.q.shape[0])

    def _nc(self) -> int:
        return int(self.q.shape[1])

    def _c(self) -> int:
        return int(self.q.shape[2])

    def _nblk(self) -> int:
        return self._c() // PWORD

    def validate(self) -> None:
        if self._c() % PWORD != 0:
            raise ValueError(f"C = Dk = Dv must be a multiple of {PWORD}; got {self._c()}.")

    def infer_outputs(self) -> dict[str, BufferInfo]:
        bh, c, nblk = self._bh(), self._c(), self._nblk()
        return {
            "output": BufferInfo(shape=(bh, self._nc(), c, nblk), dtype="float32x64", scope=A.DDR),
            "state_out": BufferInfo(shape=(bh, c, nblk), dtype="float32x64", scope=A.DDR),
        }

    def _maxtree(self, x):
        """(C, nblk) -> (1, nblk): per-lane max over the row axis in one datapath
        (running-max register, vs a ~21-datapath pairwise atomic tree)."""
        f32, f18 = "float32x64", "float18x64"
        c, nblk = int(x.shape[0]), int(x.shape[1])
        out = A.alloc_buffer((1, nblk), f32, scope=A.L1)
        with A.script():
            with A.Datapath(core=None):
                fi = A.decl_stream(f32)
                o = A.decl_stream(f32)
                for i in A.serial(c):
                    for blk in A.serial(nblk):
                        with A.IFD(num=0):
                            fi.push(x[i, blk])
                with A.DPU():
                    amax = A.array(f18, nblk)
                    v = A.var(f18)
                    for blk in A.serial(nblk):  # row 0 inits
                        amax[blk] = A.cast(fi.pop(), f18)
                    for i in A.serial(1, c):
                        for blk in A.serial(nblk):
                            v = A.cast(fi.pop(), f18)
                            amax[blk] = A.max(amax[blk], v)
                    for blk in A.serial(nblk):
                        o.push(A.cast(amax[blk], f32))
                for blk in A.serial(nblk):
                    with A.ODR():
                        out[0, blk] = o.pop()
        return out

    def _rowmax(self, absx):
        """(M, KB) -> (M, 1) broadcast = max over K (max over the KB blocks, then a
        cross-lane dpu_reduce(MAX))."""
        f32, f18 = "float32x64", "float18x64"
        m, kb = int(absx.shape[0]), int(absx.shape[1])
        out = A.alloc_buffer((m, 1), f32, scope=A.L1)
        with A.script():
            with A.Datapath(core=None):
                fi = A.decl_stream(f32)
                o = A.decl_stream(f32)
                for i in A.serial(m):
                    for j in A.serial(kb):
                        with A.IFD(num=0):
                            fi.push(absx[i, j])
                with A.DPU():
                    a = A.var(f18)
                    v = A.var(f18)
                    for i in A.serial(m):
                        a = A.cast(fi.pop(), f18)
                        for j in A.serial(1, kb):
                            v = A.cast(fi.pop(), f18)
                            a = A.max(a, v)
                        a = tir_ops.dpu_reduce(a, operation=tir_ops.DPU_REDUCE_OP.MAX)
                        o.push(A.cast(a, f32))
                for i in A.serial(m):
                    with A.ODR():
                        out[i, 0] = o.pop()
        return out

    def _recip(self, amax, shape):
        """1/amax via the full-range chained-LUT estimate + 3 Newton-Raphson steps
        y*(2 - amax*y), fused into one datapath (estimate + NR in DPU registers).
        Calibration-free (the estimate covers the whole float18 range). 2.0 from qc1
        (1-D, DPU direct-access)."""
        f32, f18 = "float32x64", "float18x64"
        m, n = int(shape[0]), int(shape[1])
        n_est = int(self.est_lut.shape[0])
        inv = A.alloc_buffer(shape, f32, scope=A.L1)
        with A.script():
            with A.Datapath(core=None):
                fi = A.decl_stream(f32)
                o = A.decl_stream(f32)
                for i in A.serial(m):
                    for j in A.serial(n):
                        with A.IFD(num=0):
                            fi.push(amax[i, j])
                with A.DPU():
                    x = A.var(f18)
                    y = A.var(f18)
                    la = A.var(f18)
                    t = A.var(f18)
                    for i in A.serial(m):
                        for j in A.serial(n):
                            x = A.cast(fi.pop(), f18)
                            y = tir_ops.lut(x, A.cast(self.est_lut[0], f18))
                            for k in A.serial(1, n_est):
                                la = tir_ops.lut(x, A.cast(self.est_lut[k], f18))
                                y = y + la
                            for _ in A.serial(3):  # Newton-Raphson y*(2 - x*y). const cast INLINE
                                t = x * y  # (a HOISTED const above the loop breaks the DPU
                                t = (
                                    A.cast(self.qc1[_I_TWO], f18) - t
                                )  # register allocator -> "c0")
                                y = y * t
                            o.push(A.cast(y, f32))
                for i in A.serial(m):
                    for j in A.serial(n):
                        with A.ODR():
                            inv[i, j] = o.pop()
        return inv

    def _quant_resid(self, qx, transpose, hi, lo) -> None:
        """qx already pre-scaled to the int8 range: hi = round(qx),
        lo = round((qx - hi)*256) (residual -> ~int16). transpose writes the int8
        in the (NB, C) block-swapped layout load_imc wants (b operand)."""
        i8, f32, f18 = "int8x64", "float32x64", "float18x64"
        m, n = int(qx.shape[0]), int(qx.shape[1])
        c256 = self.qc1[_I_256Q]  # 256.0; 1-D buffer (DPU direct-access must be 1-D), sim-safe
        with A.script():
            with A.Datapath(core=None):
                fin = A.decl_stream(f32)
                qo = A.decl_stream(i8)
                for i in A.serial(m):
                    for j in A.serial(n):
                        with A.IFD(num=0):
                            fin.push(qx[i, j])
                with A.DPU():
                    h = A.var(f18)
                    for i in A.serial(m):
                        for j in A.serial(n):
                            h = A.cast(fin.pop(), f18)
                            qo.push(A.cast(h, i8))
                for i in A.serial(m):
                    for j in A.serial(n):
                        with A.ODR():
                            if transpose:
                                hi[j, i] = qo.pop()
                            else:
                                hi[i, j] = qo.pop()
            with A.Datapath(core=None):
                fin = A.decl_stream(f32)
                hin = A.decl_stream(i8)
                qo = A.decl_stream(i8)
                for i in A.serial(m):
                    for j in A.serial(n):
                        with A.IFD(num=0):
                            fin.push(qx[i, j])
                for i in A.serial(m):
                    for j in A.serial(n):
                        with A.IFD(num=1):
                            hin.push(hi[j, i] if transpose else hi[i, j])
                with A.DPU():
                    h = A.var(f18)
                    ah = A.var(f18)
                    r = A.var(f18)
                    for i in A.serial(m):
                        for j in A.serial(n):
                            h = A.cast(fin.pop(), f18)
                            ah = A.cast(hin.pop(), f18)
                            r = h - ah
                            r = r * A.cast(c256, f18)
                            qo.push(A.cast(r, i8))
                for i in A.serial(m):
                    for j in A.serial(n):
                        with A.ODR():
                            if transpose:
                                lo[j, i] = qo.pop()
                            else:
                                lo[i, j] = qo.pop()

    def _mm_raw(self, aq, bqT, out, accumulate=False) -> None:
        """int8 IMC matmul aq (M, KB) @ bqT (NB, C) -> out (M, NB) f32, no scale
        (per-row/per-col dequant applied by the _qmm outer product). accumulate: add the
        matmul result to the existing out (read back via IFD1) in the mvm DPU phase --
        fuses an o2+o3 add into the matmul without an extra atomic."""
        i8, f32, f18 = "int8x64", "float32x64", "float18x64"
        m, kb_n, nb_n = int(aq.shape[0]), int(aq.shape[1]), int(bqT.shape[0])
        imc = A.alloc_imc((kb_n, nb_n))
        with A.script():
            atomic.load_imc(buffer=bqT, imc=imc, input_is_transposed=True)
            for i in A.serial(m):
                with A.Datapath(kind=A.DP_MVMEXE, core=None):
                    ifd0 = A.decl_stream(i8)
                    ms = A.decl_stream("int32x64")
                    do = A.decl_stream(f32)
                    pv = A.decl_stream(f32)
                    for nb in A.serial(nb_n):
                        for kb in A.serial(kb_n):
                            with A.IFD(num=0):
                                ifd0.push(aq[i, kb])
                    for nb in A.serial(nb_n):
                        with A.pseudo.MatMul():
                            acc = A.mvm_accumulator()
                            A.reset_accumulator(acc)
                            for kb in A.serial(kb_n):
                                acc += A.mvm_1x1(ifd0.pop(), imc[kb, nb])
                            ms.push(acc)
                    if accumulate:  # IFD1 must come AFTER the MatMul (unit-order rule)
                        for nb in A.serial(nb_n):
                            with A.IFD(num=1):
                                pv.push(out[i, nb])  # existing value (old)
                    with A.DPU():
                        y = A.var(f18)
                        p = A.var(f18)
                        for nb in A.serial(nb_n):
                            y = A.cast(ms.pop(), f18)
                            if accumulate:
                                p = A.cast(pv.pop(), f18)
                                y = y + p
                            do.push(A.cast(y, f32))
                    for nb in A.serial(nb_n):
                        with A.ODR():
                            out[i, nb] = do.pop()

    def _qmm(self, a, b, out) -> None:
        """fp a (M, KB) @ fp b (C, NB) -> fp out (M, NB) via per-column dynamic residual
        int8 (matches emulate's qmm; validated on AICORE_SIM at ~0.07% vs numpy). a quantized
        per-row, b per-col, scales from on-device absmax (abs LUT + max-reduce) and a
        calibration-free NR reciprocal; product reconstructed from 3 int8 IMC matmuls and
        dequanted by the sa[m]*sb[n] outer product."""
        c, nblk = self._c(), self._nblk()
        f32, i8 = "float32x64", "int8x64"
        zc = self.zero[:, 0:nblk]  # (C, nblk) zero bias for the abs LUT
        col127 = self.qc_col[:, _I_127 : _I_127 + 1]  # (C,1) 1/127
        col127inv = self.qc_col[:, _I_127INV : _I_127INV + 1]  # (C,1) 127
        row127 = self.qc_row[_I_127 : _I_127 + 1, :]  # (1,nblk) 1/127
        row127inv = self.qc_row[_I_127INV : _I_127INV + 1, :]  # (1,nblk) 127

        absa = pointwise_lut(a, zc, self.abs_lut, self.one)  # |a|
        amax_a = self._rowmax(absa)  # (C, 1)
        inv_a = self._recip(amax_a, (c, 1))
        sa = atomic.mul(amax_a, col127)  # absmax_a/127 (C, 1)
        qa = atomic.mul(a, atomic.mul(inv_a, col127inv))  # a * 127/absmax_a
        a_hi = A.alloc_buffer((c, nblk), i8, scope=A.L1)
        a_lo = A.alloc_buffer((c, nblk), i8, scope=A.L1)
        self._quant_resid(qa, False, a_hi, a_lo)

        absb = pointwise_lut(b, zc, self.abs_lut, self.one)  # |b|
        amax_b = self._maxtree(absb)  # (1, nblk)
        inv_b = self._recip(amax_b, (1, nblk))
        sb = atomic.mul(amax_b, row127)  # absmax_b/127 (1, nblk)
        qb = atomic.mul(b, atomic.mul(inv_b, row127inv))
        b_hi = A.alloc_buffer((nblk, c), i8, scope=A.L1)
        b_lo = A.alloc_buffer((nblk, c), i8, scope=A.L1)
        self._quant_resid(qb, True, b_hi, b_lo)

        def R():
            return A.alloc_buffer((c, nblk), f32, scope=A.L1)

        row256inv = self.qc_row[_I_256 : _I_256 + 1, :]  # (1,nblk) 1/256
        o1, o23 = R(), R()
        self._mm_raw(a_hi, b_hi, o1)  # hi*hi
        self._mm_raw(a_lo, b_hi, o23)  # lo*hi (residual of a)
        self._mm_raw(a_hi, b_lo, o23, accumulate=True)  # += hi*lo (residual of b), in the mvm DPU
        combined = atomic.add(o1, atomic.mul(o23, row256inv))  # o1 + (o2+o3)/256
        atomic.copy(atomic.mul(atomic.mul(combined, sa), sb), output=out)  # * sa[m] * sb[n]

    def _ew(self, a, b, out, op) -> None:
        """Elementwise a (op) b -> out, op in {'mul', 'add', 'sub'}; (C, nblk)."""
        f32 = "float32x64"
        f18 = "float18x64"
        m = int(a.shape[0])
        nblk = int(a.shape[1])
        with A.script():
            with A.Datapath(core=None):
                fa = A.decl_stream(f32)
                fb = A.decl_stream(f32)
                o = A.decl_stream(f32)
                for i in A.serial(m):
                    for blk in A.serial(nblk):
                        with A.IFD(num=0):
                            fa.push(a[i, blk])
                for i in A.serial(m):
                    for blk in A.serial(nblk):
                        with A.IFD(num=1):
                            fb.push(b[i, blk])
                with A.DPU():
                    x = A.var(f18)
                    y = A.var(f18)
                    for i in A.serial(m):
                        for blk in A.serial(nblk):
                            x = A.cast(fa.pop(), f18)
                            y = A.cast(fb.pop(), f18)
                            if op == "mul":
                                x = x * y
                            elif op == "add":
                                x = x + y
                            else:
                                x = x - y
                            o.push(A.cast(x, f32))
                for i in A.serial(m):
                    for blk in A.serial(nblk):
                        with A.ODR():
                            out[i, blk] = o.pop()

    def _grid_sum(self, x):
        """(rows, W) -> (1, W): pairwise-add tree over the row axis (rows a power of 2)."""
        rows = int(x.shape[0])
        while rows > 1:
            h = rows // 2
            x = atomic.add(atomic.copy(x[0:h, :]), atomic.copy(x[h:rows, :]))
            rows = h
        return x

    def _fwd_solve(self, lmat, rhs):
        """Exact fp18 forward substitution solving (I - lmat) X = rhs, X (C, Wb).

        lmat is the strict-lower (C, nblk) matrix (lane-packed, L[i,j] in lane j); rhs is
        (C, Wb). Left-looking: X[i] = rhs[i] + sum_{j<i} L[i,j]*X[j], X carried in an L1
        buffer. The per-row broadcast of L[i,:] (lane j -> all lanes of row j) is
        vector_transpose(repeat(L[i,:], (PWORD,1))). Rows j>=i contribute 0 (L strict-lower
        + zero-init X). Atomic-ops only."""
        f32, f18 = "float32x64", "float18x64"
        c = self._c()
        wb = int(rhs.shape[1])
        xw = A.alloc_buffer((c, wb), f32, scope=A.L1)
        atomic.copy(self.zero[:, 0:wb], output=xw)  # zero-init (rows >= i read as 0)
        # A.grid HARDWARE loop over tokens (one generated body, replayed C times) keeps the
        # per-chunk program size constant in C. The dependency X[i] <- X[j<i] carries across
        # iterations via the L1 xw buffer; each iteration's accumulator is rebuilt from rhs[i].
        with A.grid(c) as iidx:
            i = to_tuple(iidx)[0]
            li = atomic.vector_transpose(
                atomic.repeat(atomic.copy(lmat[i : i + 1, :]), (PWORD, 1)), 0, 1
            )  # (C, 1), row j = L[i,j] broadcast
            # Fused row: xw[i] = rhs[i] + sum_j li[j]*xw[j] in one datapath (accumulator in
            # registers, xw streamed from L1). Rows j>=i contribute 0 (li strict-lower, xw zero-init).
            with A.script():
                with A.Datapath(core=None):
                    fx = A.decl_stream(f32)  # rhs[i,w] then xw[j,w]  (IFD1: 2-source stream)
                    fl = A.decl_stream(f32)  # li[j] broadcast        (IFD0: single block)
                    o = A.decl_stream(f32)
                    # li -> IFD0 as a SINGLE block; the 2-source fx stream (rhs THEN xw) MUST go
                    # on IFD1. A stream fed by two IFD0 push-blocks from two buffers DEADLOCKS the
                    # device (MSI timeout -> DEVICE_LOST) though it passes AICORE_SIM.
                    for j in A.serial(c):
                        with A.IFD(num=0):
                            fl.push(li[j, 0])
                    for w in A.serial(wb):
                        with A.IFD(num=1):
                            fx.push(rhs[i, w])
                    for j in A.serial(c):
                        for w in A.serial(wb):
                            with A.IFD(num=1):
                                fx.push(xw[j, w])
                    with A.DPU():
                        acc = A.array(f18, wb)
                        m = A.var(f18)
                        t = A.var(f18)
                        for w in A.serial(wb):
                            acc[w] = A.cast(fx.pop(), f18)  # rhs[i,w]
                        for j in A.serial(c):
                            m = A.cast(fl.pop(), f18)
                            for w in A.serial(wb):
                                t = m * A.cast(fx.pop(), f18)
                                acc[w] = acc[w] + t
                        for w in A.serial(wb):
                            o.push(A.cast(acc[w], f32))
                    for w in A.serial(wb):
                        with A.ODR():
                            xw[i, w] = o.pop()
        return xw

    def implementation(self) -> None:
        bh_n = self._bh()
        nc = self._nc()
        c = self._c()
        nblk = self._nblk()
        f32 = "float32x64"

        def L1(cols=nblk):
            return A.alloc_buffer((c, cols), f32, scope=A.L1)

        # A.grid HARDWARE loop over (head, batch) sequences (one generated body, reused BH
        # times) so the kernel scales to many heads without a Python-unroll code blow-up.
        with A.grid(bh_n) as idx:
            bh = to_tuple(idx)[0]
            state = L1()
            atomic.copy(self.s0[bh, :, :], output=state)

            # A.grid HARDWARE loop over chunks (one generated body, replayed NC times) keeps
            # program size constant in context length. The recurrent state carries across
            # iterations via the L1 state buffer (cf. the per-token xw carry in _fwd_solve).
            with A.grid(nc) as cidx:
                ci = to_tuple(cidx)[0]
                q = L1()
                kb = L1()
                vb = L1()
                qg = L1()
                kbg = L1()
                kdect = L1()
                dm = L1()
                nsl = L1()
                dlast = L1()
                knk = L1()
                atomic.copy(self.q[bh, ci, :, :], output=q)
                atomic.copy(self.kb[bh, ci, :, :], output=kb)
                atomic.copy(self.knk[bh, ci, :, :], output=knk)
                atomic.copy(self.vb[bh, ci, :, :], output=vb)
                atomic.copy(self.qg[bh, ci, :, :], output=qg)
                atomic.copy(self.kbg[bh, ci, :, :], output=kbg)
                atomic.copy(self.kdecT[bh, ci, :, :], output=kdect)
                atomic.copy(self.dm[bh, ci, :, :], output=dm)
                atomic.copy(self.nsl[bh, ci, :, :], output=nsl)
                atomic.copy(self.decaylast[bh, ci, :, :], output=dlast)

                # L = (kβ@kᵀ) ⊙ nsl  (int8 GEMM + DPU mask)
                m1 = L1()
                self._qmm(kb, knk, m1)
                lmat = L1()
                self._ew(m1, nsl, lmat, "mul")
                # [value | kcd] = (I - L)⁻¹ @ [vβ | kβ⊙exp g]  -- one fp18 forward solve
                rhs = L1(2 * nblk)
                atomic.copy(vb, output=rhs[:, 0:nblk])
                atomic.copy(kbg, output=rhs[:, nblk : 2 * nblk])
                X = self._fwd_solve(lmat, rhs)
                value = L1()
                kcd = L1()
                atomic.copy(X[:, 0:nblk], output=value)
                atomic.copy(X[:, nblk : 2 * nblk], output=kcd)

                m2 = L1()
                self._qmm(q, knk, m2)
                ai = L1()
                self._ew(m2, dm, ai, "mul")
                vprime = L1()
                self._qmm(kcd, state, vprime)
                vnew = L1()
                self._ew(value, vprime, vnew, "sub")
                ainter = L1()
                self._qmm(qg, state, ainter)
                aiv = L1()
                self._qmm(ai, vnew, aiv)
                out_c = L1()
                self._ew(ainter, aiv, out_c, "add")
                atomic.copy(out_c, output=self.output[bh, ci, :, :])

                # state update: S = exp(g_last)*S + kdecᵀ @ v_new
                upd = L1()
                self._qmm(kdect, vnew, upd)
                sd = L1()
                self._ew(state, dlast, sd, "mul")
                sn = L1()
                self._ew(sd, upd, sn, "add")
                atomic.copy(sn, output=state)
            # Write the final state to exactly one destination. With state_inplace the carry
            # lives in the s0 input buffer (persistent-cache RMW, cf. causal_conv1d) and
            # state_out is NOT written: the decoder layer binds state_out to a dead buffer
            # the codegen sizes to ~1 PWORD, so writing the full (C, NB) state into it
            # overflows the ddr_workspace pool.
            if self.state_inplace:
                atomic.copy(state, output=self.s0[bh, :, :])
            else:
                atomic.copy(state, output=self.state_out[bh, :, :])

    def emulate(self) -> None:  # type: ignore[override]
        bh_n = self._bh()
        nc = self._nc()
        c = self._c()

        def f18(x):
            return emu.rf18(x.astype(np.float32))

        def qmm(a, b):  # per-column dynamic residual int8 -> ~int16
            sa = np.maximum(np.abs(a).max(1, keepdims=True), 1e-12) / 127.0  # (M,1) per-row of A
            sb = np.maximum(np.abs(b).max(0, keepdims=True), 1e-12) / 127.0  # (1,N) per-col of B

            def split(x, s):
                h = f18(f18(x) / f18(s))
                hi = np.clip(np.round(h), -127, 127)
                lo = np.clip(np.round(f18((h - hi) * np.float32(_RESID))), -127, 127)
                return hi, lo

            ah, al = split(a, sa)
            bh, bl = split(b, sb)
            sab = (sa * sb).astype(np.float32)  # (M,N) outer product of output-axis scales

            def mm(x, y, scale):  # int32 accumulate (exact for int8), cast f18, * per-(m,n) scale
                return f18(f18((x @ y).astype(np.float32)) * f18(scale))

            o1 = mm(ah, bh, sab)
            o2 = mm(al, bh, sab / _RESID)
            o3 = mm(ah, bl, sab / _RESID)
            return f18(f18(f18(o1) + f18(o2)) + f18(o3))

        def mul(a, b):
            return f18(f18(a) * f18(b))

        def add(a, b):
            return f18(f18(a) + f18(b))

        def sub(a, b):
            return f18(f18(a) - f18(b))

        def fwd_solve(L, B):
            """Exact (I-L)X=B by forward substitution in float18 (per-step mul-add rounding).
            L (c,c) strict-lower; B (c,W). Mirrors the device DPU scan."""
            X = np.zeros_like(B, dtype=np.float32)
            Lf = f18(L)
            for i in range(c):
                acc = f18(B[i])
                for j in range(i):
                    acc = f18(acc + f18(Lf[i, j] * f18(X[j])))
                X[i] = acc
            return X

        def per(b4):  # (BH, NC, C, nblk, PWORD) -> (BH, NC, C, C): flatten (nblk, PWORD)
            return b4.reshape(b4.shape[:3] + (b4.shape[3] * b4.shape[4],))[..., :c].astype(
                np.float64
            )

        q = per(self.q)
        kb = per(self.kb)
        vb = per(self.vb)
        qg = per(self.qg)
        kbg = per(self.kbg)
        kdecT = per(self.kdecT)
        dm = per(self.dm)
        nsl = per(self.nsl)
        dlast = per(self.decaylast)
        knk = per(self.knk)  # (BH, NC, C, C): knk[bh, ci][d, j] = k[j, d]
        nblk = self._nblk()

        out = np.zeros((bh_n, nc, c, c), np.float32)
        state_final = np.zeros((bh_n, c, c), np.float32)
        for bh in range(bh_n):
            state = (
                self.s0[bh].reshape(c, nblk * PWORD)[:, :c].astype(np.float32).copy()
            )  # (Dk, Dv)
            for ci in range(nc):
                lmat = mul(qmm(kb[bh, ci], knk[bh, ci]), nsl[bh, ci])
                # [value | kcd] = (I-lmat)^-1 @ [vb | kbg]  -- one fp18 forward solve
                X = fwd_solve(lmat, np.concatenate([vb[bh, ci], kbg[bh, ci]], axis=1))
                value, kcd = X[:, :c], X[:, c:]
                ai = mul(qmm(q[bh, ci], knk[bh, ci]), dm[bh, ci])
                vprime = qmm(kcd, state)
                vnew = sub(value, vprime)
                ainter = qmm(qg[bh, ci], state)
                out[bh, ci] = add(ainter, qmm(ai, vnew)).astype(np.float32)
                upd = qmm(kdecT[bh, ci], vnew)
                state = add(mul(state, dlast[bh, ci]), upd)
            state_final[bh] = state.astype(np.float32)
        np.copyto(self.output, out.reshape(bh_n, nc, c, nblk, PWORD))
        packed = state_final.reshape(bh_n, c, nblk, PWORD)
        np.copyto(self.state_out, packed)
        if self.state_inplace:  # also carry the final state in s0 (in place)
            np.copyto(self.s0, packed)


gated_deltanet_chunk_imc_multi = register_op(
    GatedDeltaNetChunkImcMulti,
    "llm.gated_deltanet_chunk_imc_multi",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)

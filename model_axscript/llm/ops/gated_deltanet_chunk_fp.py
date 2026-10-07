#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Chunked fp18 gated-delta recurrence for prefill (llm.gated_deltanet_chunk_fp).

Processes the sequence in chunks of chunk tokens, reading/writing the (Dk,Dv) state
once per chunk, amortising the dominant state-streaming cost by ~chunk vs the per-token
gated_deltanet_recurrent_fp. Pure fp18 with an EXACT forward-substitution solve (a
Neumann-series approximation of the inverse diverges on real activations where ||L||~8).

Chunk algebra (entry state S0, tokens 1..T in the chunk, A_t = prod_{j<=t} alpha_j):

    (I + M) Vnew = RHS,  M strictly-lower UNIT-triangular  -> exact forward substitution
        M[t,i]   = beta_t (A_t/A_i)(k_t . k_i)   (i < t)
        RHS[t]   = beta_t (v_t - A_t (k_t^T S0))
    O   = diag(A)(Q S0) + P Vnew,   P[t,i] = (A_t/A_i)(q_t . k_i)   (i <= t)
    S_T = A_T S0 + (Kgamma)^T Vnew,  Kgamma[i] = (A_T/A_i) k_i

Drop-in interface-compatible with gated_deltanet_recurrent_fp (same q/k/v/ab/s0 ->
output/state, head-major, state carry / state_inplace RMW) plus a chunk size.
"""
from axelera.compiler.alto.axol import atomic
from axelera.compiler.alto.axol.utils import to_tuple
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


def _f18(x):
    return emu.rf18(np.asarray(x, dtype=np.float32))


def _mm18(Amat, Bmat):
    """(I,L)@(L,J) -> (I,J), fp18 per-product and per-add over L (DPU FMA granularity)."""
    Amat = _f18(Amat)
    Bmat = _f18(Bmat)
    L = Amat.shape[1]
    acc = _f18(Amat[:, 0:1] * Bmat[0:1, :])
    for li in range(1, L):
        acc = _f18(acc + _f18(Amat[:, li : li + 1] * Bmat[li : li + 1, :]))
    return acc


def _chunk_fp18(qn, kn, vn, alpha, beta, S0, C):
    """fp18 chunked gated-delta rule for one head. Returns (out (seq,Dv), final S (Dk,Dv))."""
    seq, _Dk = qn.shape
    S = _f18(S0.copy())
    out = np.zeros((seq, vn.shape[1]), np.float32)
    for c0 in range(0, seq, C):
        T = min(C, seq - c0)
        Q, K, V = qn[c0 : c0 + T], kn[c0 : c0 + T], vn[c0 : c0 + T]
        a, b = alpha[c0 : c0 + T], beta[c0 : c0 + T]

        # Decays without LUT/division: A_t = cumprod(alpha) and the ratio matrix
        # R[t,i] = A_t/A_i = prod_{i<j<=t} alpha_j via a pure-MULTIPLY recurrence
        #   R[t] = R[t-1] * alpha_t + onehot[t]   (R[t,t]=1, R[t,i>t]=0)
        # exactly mirroring the device datapath (so emulate stays bit-aligned). fp18 throughout.
        A_t = np.zeros(T, np.float32)
        R = np.zeros((T, T), np.float32)
        Aacc = np.float32(1.0)
        Rprev = np.zeros(T, np.float32)
        af = _f18(a)
        for t in range(T):
            Aacc = _f18(Aacc * af[t])
            A_t[t] = Aacc
            Rt = _f18(Rprev * af[t])
            Rt[t] = np.float32(1.0)
            R[t] = Rt
            Rprev = Rt
        Ltri = np.tril(np.ones((T, T), np.float32))
        SLtri = np.tril(np.ones((T, T), np.float32), -1)

        KS0 = _mm18(K, S)  # (T,Dv)
        QS0 = _mm18(Q, S)
        KK = _mm18(K, K.T)  # (T,T)
        QK = _mm18(Q, K.T)

        M = _f18(_f18(b[:, None] * R) * KK) * SLtri
        RHS = _f18(b[:, None] * _f18(V - _f18(A_t[:, None] * KS0)))
        Vnew = np.zeros((T, vn.shape[1]), np.float32)  # exact fp18 forward substitution
        for t in range(T):
            s = _f18(RHS[t])
            for i in range(t):
                s = _f18(s - _f18(M[t, i] * Vnew[i]))
            Vnew[t] = s

        P = _f18(_f18(R) * QK) * Ltri
        out[c0 : c0 + T] = _f18(_f18(A_t[:, None] * QS0) + _mm18(P, Vnew))

        Kg = _f18(R[T - 1][:, None] * K)  # Kgamma[i] = (A_T/A_i) k_i
        S = _f18(_f18(A_t[T - 1] * S) + _mm18(Kg.T, Vnew))
    return out, S


class GatedDeltaNetChunkFp(ScriptOp):
    """BH-head chunked fp18 gated-delta recurrence (chunk-amortised state streaming)."""

    q = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(BH, seq, 1, NB) lane-packed L2-normed query (Dk in lanes).",
    )
    k = Arg(MetisFloats[tir.Buffer, "..."], description="(BH, seq, 1, NB) lane-packed key.")
    v = Arg(MetisFloats[tir.Buffer, "..."], description="(BH, seq, 1, NB) lane-packed value.")
    ab = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(BH, seq, 1, 2) per-(head,token) [alpha, beta], broadcast across lanes.",
    )
    s0 = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(BH, Dk, NB) entry state S per head (Dk rows, Dv in lanes).",
    )
    chunk = Arg(
        int,
        default=16,
        kw=True,
        description="Chunk length C: tokens processed per state read/write. Smaller C = less "
        "fp18 error in the triangular solve + better conditioning; larger C = less state L1 "
        "traffic. C=16 is the validated sweet spot.",
    )
    output = Arg(
        MetisFloats[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="Per-(head,token) readout, auto-inferred (BH, seq, 1, NB) float32x64 DDR.",
    )
    state_out = Arg(
        MetisFloats[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="Final state per head, auto-inferred (BH, Dk, NB) DDR.",
    )
    state_inplace = Arg(
        bool,
        default=False,
        kw=True,
        description="Write the final state back into s0 in place (persistent-cache RMW) and "
        "emit no separate state_out.",
    )
    generation = Arg(HardwareGeneration | None, default=None, kw=True)

    def _bh(self) -> int:
        return int(self.q.shape[0])

    def _seq(self) -> int:
        return int(self.q.shape[1])

    def _nb(self) -> int:
        return int(self.v.shape[3])

    def _dk(self) -> int:
        return self._nb() * PWORD  # Dk == Dv (square state)

    def _chunk(self) -> int:
        return int(self.chunk)

    def validate(self) -> None:
        if self._dk() // PWORD != self._nb():
            raise ValueError(f"Dk == Dv required: Dk={self._dk()} NB={self._nb()}.")
        if self._seq() % self._chunk() != 0:
            raise ValueError(f"seq ({self._seq()}) must be a multiple of chunk ({self._chunk()}).")

    def infer_outputs(self) -> dict[str, BufferInfo]:
        bh, seq, nb = self._bh(), self._seq(), self._nb()
        return {
            "output": BufferInfo(shape=(bh, seq, 1, nb), dtype="float32x64", scope=A.DDR),
            "state_out": BufferInfo(shape=(bh, self._dk(), nb), dtype="float32x64", scope=A.DDR),
        }

    # DPU datapaths. The whole op is built from ONE workhorse matvec (rd = Σ_l mat[l,:]
    # vrow[l]) -- it does the Grams (mat=Kt, C-in-lanes out), the K/Q·S0 projections
    # (mat=S0, Dv-lanes), AND the solve / P·Vnew reductions (mat=Vnew, contract over chunk)
    # -- plus _per_row, the _spread bridge (C-in-lanes -> broadcast), and _update (state).
    # This device version processes the whole seq as ONE chunk (the host loops chunks,
    # threading state via state_inplace); readout/state use the per-row matvec form.
    def _per_row(self, vec, dst) -> None:
        """lane-packed (1, nb) -> row-packed (Dk, 1) broadcast (dst[dk]=scalar vec[dk])."""
        nb = self._nb()
        for blk in range(nb):
            vb = atomic.reshape(atomic.copy(vec[:, blk : blk + 1]), shape=(1, 1))
            atomic.copy(
                atomic.vector_transpose(atomic.repeat(vb, (PWORD, 1)), 0, 1),
                output=dst[blk * PWORD : (blk + 1) * PWORD, :],
            )

    def _spread(self, src, dst64) -> None:
        """C-in-lanes pword src (1,1) -> (64,1) broadcast (dst64[i]=scalar src lane i).
        Copies src into a fresh (1,1) first so it tolerates a grid-var slice / reshape view."""
        tmp = A.alloc_buffer((1, 1), "float32x64", scope=A.L1)
        atomic.copy(src, output=tmp)
        atomic.copy(atomic.vector_transpose(atomic.repeat(tmp, (PWORD, 1)), 0, 1), output=dst64)

    def _matvec(self, mat, vrow, out, row, L, W) -> None:
        """out[row,0,b] = Σ_{l<L} mat[l,b] * vrow[l,0] (mat a 2D (L,W) buffer).
        Workhorse for the Grams (mat=Kt, W=1) and the K/Q·S0 projections (mat=S0, W=nb).
        mat on IFD0, vrow on IFD1 (both single buffers); stores into out at row.
        Separate from _reduce: the TVM parser parses BOTH branches of any in-body Python
        if, so a shared mat-arity branch mis-ranks the 2D case at parse time."""
        f32, f18 = "float32x64", "float18x64"
        with A.script():
            with A.Datapath(core=None):
                fs = A.decl_stream(f32)
                fv = A.decl_stream(f32)
                o = A.decl_stream(f32)
                for i in A.serial(L):
                    for b in A.serial(W):
                        with A.IFD(num=0):
                            fs.push(mat[i, b])
                for i in A.serial(L):
                    with A.IFD(num=1):
                        fv.push(vrow[i, 0])
                with A.DPU():
                    acc = A.array(f18, W)
                    vv = A.var(f18)
                    t = A.var(f18)
                    vv = A.cast(fv.pop(), f18)
                    for b in A.serial(W):
                        acc[b] = A.cast(fs.pop(), f18) * vv
                    for i in A.serial(1, L):
                        vv = A.cast(fv.pop(), f18)
                        for b in A.serial(W):
                            t = A.cast(fs.pop(), f18) * vv
                            acc[b] = acc[b] + t
                    for b in A.serial(W):
                        o.push(A.cast(acc[b], f32))
                for b in A.serial(W):
                    with A.ODR():
                        out[row, 0, b] = o.pop()

    def _reduce(self, mat, vrow, out, row, L, W) -> None:
        """Same contraction with mat a 3D (L,1,W) buffer (mat[l,0,b]) -- the solve / P·Vnew
        reductions (mat=Vnew, contract over the chunk)."""
        f32, f18 = "float32x64", "float18x64"
        with A.script():
            with A.Datapath(core=None):
                fs = A.decl_stream(f32)
                fv = A.decl_stream(f32)
                o = A.decl_stream(f32)
                for i in A.serial(L):
                    for b in A.serial(W):
                        with A.IFD(num=0):
                            fs.push(mat[i, 0, b])
                for i in A.serial(L):
                    with A.IFD(num=1):
                        fv.push(vrow[i, 0])
                with A.DPU():
                    acc = A.array(f18, W)
                    vv = A.var(f18)
                    t = A.var(f18)
                    vv = A.cast(fv.pop(), f18)
                    for b in A.serial(W):
                        acc[b] = A.cast(fs.pop(), f18) * vv
                    for i in A.serial(1, L):
                        vv = A.cast(fv.pop(), f18)
                        for b in A.serial(W):
                            t = A.cast(fs.pop(), f18) * vv
                            acc[b] = acc[b] + t
                    for b in A.serial(W):
                        o.push(A.cast(acc[b], f32))
                for b in A.serial(W):
                    with A.ODR():
                        out[row, 0, b] = o.pop()

    def _matmul_cc(self, Acol, Bt, dst, M, L) -> None:
        """C-in-lanes matmul in ONE datapath: dst[i] = Σ_l Acol[l,i] * Bt[l]  (i in 0..M-1).
        Folds the whole output-row loop into the DPU -- M physical-register accumulators (one
        C-in-lanes pword/row, M<=64=bank depth), reduction over L a HW loop. Used for the Grams
        KK = K Kᵀ, QK = Q Kᵀ (Bt = Kt, (L,1) transposed key; Acol = (L,M) K/Q broadcast:
        the i-th scalar at [l,i] so the i-inner stream is contiguous). dst is (M,1,1) C-in-lanes.
        """
        f32, f18 = "float32x64", "float18x64"
        with A.script():
            with A.Datapath(core=None):
                fb = A.decl_stream(
                    f32
                )  # Bt[l]        IFD0 (streamed once, held across the i loop)
                fa = A.decl_stream(f32)  # Acol[l,i]    IFD1 (l outer, i inner)
                o = A.decl_stream(f32)
                for li in A.serial(L):
                    with A.IFD(num=0):
                        fb.push(Bt[li, 0])
                for li in A.serial(L):
                    for i in A.serial(M):
                        with A.IFD(num=1):
                            fa.push(Acol[i, li, 0])
                acc = A.dpu.registers(A.dpu.RegBank.B)  # M accumulators persist across blocks
                with A.DPU():  # block 1: l=0 init
                    bl = A.var(f18)
                    ai = A.var(f18)
                    bl = A.cast(fb.pop(), f18)
                    for i in A.serial(M):
                        ai = A.cast(fa.pop(), f18)
                        acc[i] = ai * bl
                with A.DPU():  # block 2: reduction HW loop over l=1..L-1
                    bl = A.var(f18)
                    ai = A.var(f18)
                    t = A.var(f18)
                    for li in A.serial(1, L):
                        bl = A.cast(fb.pop(), f18)
                        for i in A.serial(M):
                            ai = A.cast(fa.pop(), f18)
                            t = ai * bl
                            acc[i] = acc[i] + t
                with (
                    A.DPU()
                ):  # block 3: drain (own block keeps block 2 a HW loop; DEVICE-only -- sim
                    for i in A.serial(M):  # mis-models the input-less register drain)
                        o.push(A.cast(acc[i], f32))
                for i in A.serial(M):
                    with A.ODR():
                        dst[i, 0, 0] = o.pop()

    def _matmul_cw(self, Acol, B, dst, M, L, W, row0=0) -> None:
        """Dv-lane matmul in ONE datapath: dst[row0+i, w] = Σ_l Acol[row0+i,l,0] * B[l,w]  (i in M,
        w in W). B has W lane-columns (B = S0, (L,W) Dv-blocks); each output row is W pwords ->
        acc[i*W+w], M*W <= 64 (bank B depth). Callers tile the C rows (row0 offset) so M*W fits one
        bank (e.g. C=64,W=2 -> two M=32 tiles). Used for KS0 = K S0, QS0 = Q S0. dst is (C,1,W)."""
        f32, f18 = "float32x64", "float18x64"
        with A.script():
            with A.Datapath(core=None):
                fb = A.decl_stream(f32)  # B[l,w]      IFD0
                fa = A.decl_stream(f32)  # Acol[l,i]   IFD1
                o = A.decl_stream(f32)
                for li in A.serial(L):
                    for w in A.serial(W):
                        with A.IFD(num=0):
                            fb.push(B[li, w])
                for li in A.serial(L):
                    for i in A.serial(M):
                        with A.IFD(num=1):
                            fa.push(Acol[row0 + i, li, 0])
                acc = A.dpu.registers(A.dpu.RegBank.B)  # M*W accumulators (tiled to fit)
                with A.DPU():  # block 1: l=0 init
                    bw = A.array(f18, W)
                    ai = A.var(f18)
                    for w in A.serial(W):
                        bw[w] = A.cast(fb.pop(), f18)
                    for i in A.serial(M):
                        ai = A.cast(fa.pop(), f18)
                        for w in A.serial(W):
                            acc[i * W + w] = ai * bw[w]
                with A.DPU():  # block 2: reduction HW loop over l=1..L-1
                    bw = A.array(f18, W)
                    ai = A.var(f18)
                    t = A.var(f18)
                    for li in A.serial(1, L):
                        for w in A.serial(W):
                            bw[w] = A.cast(fb.pop(), f18)
                        for i in A.serial(M):
                            ai = A.cast(fa.pop(), f18)
                            for w in A.serial(W):
                                t = ai * bw[w]
                                acc[i * W + w] = acc[i * W + w] + t
                with A.DPU():  # block 3: drain
                    for i in A.serial(M):
                        for w in A.serial(W):
                            o.push(A.cast(acc[i * W + w], f32))
                for i in A.serial(M):
                    for w in A.serial(W):
                        with A.ODR():
                            dst[row0 + i, 0, w] = o.pop()

    def _solve_offdiag(self, M_bcast, RHS, Vnew, rhsp, b0, rows) -> None:
        """Blocked solve, off-diagonal update: rhsp[b0:b1, :] = RHS[b0:b1, :] - M[b0:b1, 0:b0] @
        Vnew[0:b0, :] (all nb Dv-blocks at once). Vnew_prior on IFD0 (nb pwords/prior-row),
        M[b0+i, j] broadcast on IFD1 (after RHS init); HW loop over b0 prior rows -> constant
        instruction count regardless of chunk size. acc[i*nb+w] = rows*nb regs (<=64)."""
        nb = self._nb()
        f32, f18 = "float32x64", "float18x64"
        with A.script():
            with A.Datapath(core=None):
                fb = A.decl_stream(f32)  # IFD0: Vnew[j, w]  (prior rows x nb)
                fa = A.decl_stream(
                    f32
                )  # IFD1: RHS[b0+i, w] (init) then M[b0+i, j] (j-outer i-inner)
                o = A.decl_stream(f32)
                for j in A.serial(b0):
                    for w in A.serial(nb):
                        with A.IFD(num=0):
                            fb.push(Vnew[j, 0, w])
                for i in A.serial(rows):
                    for w in A.serial(nb):
                        with A.IFD(num=1):
                            fa.push(RHS[b0 + i, 0, w])
                for j in A.serial(b0):
                    for i in A.serial(rows):
                        with A.IFD(num=1):
                            fa.push(M_bcast[b0 + i, j, 0])
                acc = A.dpu.registers(A.dpu.RegBank.B)  # rows*nb
                with A.DPU():  # init acc = RHS[block]
                    for i in A.serial(rows):
                        for w in A.serial(nb):
                            acc[i * nb + w] = A.cast(fa.pop(), f18)
                with (
                    A.DPU()
                ):  # acc[i,w] -= sum_j M[b0+i,j] * Vnew[j,w]  (HW loop over prior rows j)
                    bw = A.array(f18, nb)
                    m = A.var(f18)
                    t2 = A.var(f18)
                    for j in A.serial(b0):
                        for w in A.serial(nb):
                            bw[w] = A.cast(fb.pop(), f18)  # Vnew[j, w]
                        for i in A.serial(rows):
                            m = A.cast(fa.pop(), f18)  # M[b0+i, j]
                            for w in A.serial(nb):
                                t2 = m * bw[w]
                                acc[i * nb + w] = acc[i * nb + w] - t2
                with A.DPU():  # drain
                    for i in A.serial(rows):
                        for w in A.serial(nb):
                            o.push(A.cast(acc[i * nb + w], f32))
                for i in A.serial(rows):
                    for w in A.serial(nb):
                        with A.ODR():
                            rhsp[b0 + i, 0, w] = o.pop()

    def _solve_diag(self, M_bcast, src, Vnew, b0, rows, w) -> None:
        """Blocked solve, diagonal block (one Dv-block w): solve (I+M_kk) Vnew[b0:b1, w] = src[b0:b1, w]
        by forward substitution -- Bs-row register-feedback solve (Bs<=16 unrolled, fits instruction
        memory). M_kk on IFD0, src on IFD1; register feedback acc[i] reads acc[j<i]."""
        f32, f18 = "float32x64", "float18x64"
        with A.script():
            with A.Datapath(core=None):
                fm = A.decl_stream(f32)  # IFD0: M[b0:b1, b0:b1]
                fr = A.decl_stream(f32)  # IFD1: src[b0:b1, w]
                o = A.decl_stream(f32)
                for i in A.serial(rows):
                    for j in A.serial(rows):
                        with A.IFD(num=0):
                            fm.push(M_bcast[b0 + i, b0 + j, 0])
                for i in A.serial(rows):
                    with A.IFD(num=1):
                        fr.push(src[b0 + i, 0, w])
                acc = A.dpu.registers(A.dpu.RegBank.B)  # rows
                with A.DPU():  # forward substitution; acc[j<i] is the register feedback
                    r = A.var(f18)
                    m = A.var(f18)
                    t2 = A.var(f18)
                    for i in A.serial(rows):
                        r = A.cast(fr.pop(), f18)
                        for j in A.serial(rows):
                            m = A.cast(fm.pop(), f18)
                            if j < i:
                                t2 = m * acc[j]
                                r = r - t2
                        acc[i] = r
                with A.DPU():  # drain
                    for i in A.serial(rows):
                        o.push(A.cast(acc[i], f32))
                for i in A.serial(rows):
                    with A.ODR():
                        Vnew[b0 + i, 0, w] = o.pop()

    def _output_matmul(self, P_bcast, AQS0, Vnew, outb, row0, rows, T) -> None:
        """Readout O[t,w] = AQS0[t,w] + sum_i P[t,i] * Vnew[i,w] as one matmul datapath (folds the
        per-row output grid(T)). P broadcast on IFD1 (after AQS0 init), Vnew on IFD0; HW loop over
        the chunk i (P[t,i]=0 for i>t masks the upper part). Tiled over output rows so rows*nb <= 64.
        """
        nb = self._nb()
        f32, f18 = "float32x64", "float18x64"
        with A.script():
            with A.Datapath(core=None):
                fb = A.decl_stream(f32)  # IFD0: Vnew[j, w] (whole chunk x nb)
                fa = A.decl_stream(f32)  # IFD1: AQS0[row0+i, w] (init) then P[row0+i, j] (j-outer)
                o = A.decl_stream(f32)
                for j in A.serial(T):
                    for w in A.serial(nb):
                        with A.IFD(num=0):
                            fb.push(Vnew[j, 0, w])
                for i in A.serial(rows):
                    for w in A.serial(nb):
                        with A.IFD(num=1):
                            fa.push(AQS0[row0 + i, 0, w])
                for j in A.serial(T):
                    for i in A.serial(rows):
                        with A.IFD(num=1):
                            fa.push(P_bcast[row0 + i, j, 0])
                acc = A.dpu.registers(A.dpu.RegBank.B)  # rows*nb
                with A.DPU():  # init acc = AQS0[block]
                    for i in A.serial(rows):
                        for w in A.serial(nb):
                            acc[i * nb + w] = A.cast(fa.pop(), f18)
                with A.DPU():  # acc[i,w] += sum_j P[row0+i,j] * Vnew[j,w]  (HW loop over chunk j)
                    bw = A.array(f18, nb)
                    m = A.var(f18)
                    t2 = A.var(f18)
                    for j in A.serial(T):
                        for w in A.serial(nb):
                            bw[w] = A.cast(fb.pop(), f18)  # Vnew[j, w]
                        for i in A.serial(rows):
                            m = A.cast(fa.pop(), f18)  # P[row0+i, j]
                            for w in A.serial(nb):
                                t2 = m * bw[w]
                                acc[i * nb + w] = acc[i * nb + w] + t2
                with A.DPU():  # drain
                    for i in A.serial(rows):
                        for w in A.serial(nb):
                            o.push(A.cast(acc[i * nb + w], f32))
                for i in A.serial(rows):
                    for w in A.serial(nb):
                        with A.ODR():
                            outb[row0 + i, 0, w] = o.pop()

    def _scan_A(self, alpha, A_buf, T) -> None:
        """Cumulative decay A_t = A_{t-1} * alpha_t carried in one DPU register across a
        serial(T) scan. One push/step, single ODR drain -> A_buf[t] (broadcast scalar).
        alpha on IFD0 (single buffer). Split from _scan_R so each datapath has exactly one
        output buffer (the compiling IFD->DPU->ODR template)."""
        f32, f18 = "float32x64", "float18x64"
        with A.script():
            with A.Datapath(core=None):
                fa = A.decl_stream(f32)
                o = A.decl_stream(f32)
                for t in A.serial(T):
                    with A.IFD(num=0):
                        fa.push(alpha[t, 0, 0])
                with A.DPU():
                    a = A.var(f18)
                    av = A.var(f18)
                    a = A.cast(fa.pop(), f18)  # t=0: A_0 = alpha_0 (A_{-1}=1)
                    o.push(A.cast(a, f32))
                    for t in A.serial(1, T):
                        av = A.cast(fa.pop(), f18)
                        a = a * av
                        o.push(A.cast(a, f32))
                for t in A.serial(T):
                    with A.ODR():
                        A_buf[t, 0, 0] = o.pop()

    def _scan_R(self, alpha, onehots, R_buf, T) -> None:
        """Ratio rows R[t,i] = A_t/A_i via the pure-multiply recurrence R_t = R_{t-1}*alpha_t +
        onehot_t (R_0 = e_0, R[t,t]=1, R[t,i>t]=0), the C-in-lanes row carried in one DPU
        register across a serial(T) scan. alpha on IFD0, onehots on IFD1 (single buffers);
        one push/step, single ODR drain -> R_buf[t]. fp18 throughout (bit-aligned with emulate)."""
        f32, f18 = "float32x64", "float18x64"
        with A.script():
            with A.Datapath(core=None):
                fa = A.decl_stream(f32)
                fo = A.decl_stream(f32)
                o = A.decl_stream(f32)
                for t in A.serial(T):  # push ALL T alphas: a serial start>0 is NOT honored for an
                    with A.IFD(
                        num=0
                    ):  # IFD push loop (it still feeds from 0) -- align in the DPU.
                        fa.push(alpha[t, 0, 0])
                for t in A.serial(T):
                    with A.IFD(num=1):
                        fo.push(onehots[t, 0, 0])
                with A.DPU():
                    r = A.var(f18)
                    av = A.var(f18)
                    r = A.cast(fo.pop(), f18)  # t=0: R_0 = onehot_0
                    av = A.cast(fa.pop(), f18)  # discard alpha_0 (unused by R) to align the stream
                    o.push(A.cast(r, f32))
                    for t in A.serial(1, T):
                        av = A.cast(fa.pop(), f18)
                        r = r * av
                        r = r + A.cast(fo.pop(), f18)
                        o.push(A.cast(r, f32))
                for t in A.serial(T):
                    with A.ODR():
                        R_buf[t, 0, 0] = o.pop()

    def _update(self, state, avn, kl, snew) -> None:
        """snew[dk,b] = avn[0]*state[dk,b] + kl[dk]*avn[1+b]  (rank-1 state accumulate)."""
        dk, nb = self._dk(), self._nb()
        f32, f18 = "float32x64", "float18x64"
        with A.script():
            with A.Datapath(core=None):
                fs = A.decl_stream(f32)
                faux = A.decl_stream(f32)
                oo = A.decl_stream(f32)
                for i in A.serial(dk):
                    for b in A.serial(nb):
                        with A.IFD(num=0):
                            fs.push(state[i, b])
                for j in A.serial(1 + nb):
                    with A.IFD(num=1):
                        faux.push(avn[0, j])
                for i in A.serial(dk):
                    with A.IFD(num=1):
                        faux.push(kl[i, 0])
                with A.DPU():
                    vna = A.array(f18, nb)
                    alv = A.var(f18)
                    kk = A.var(f18)
                    s = A.var(f18)
                    t = A.var(f18)
                    t2 = A.var(f18)
                    alv = A.cast(faux.pop(), f18)
                    for b in A.serial(nb):
                        vna[b] = A.cast(faux.pop(), f18)
                    for i in A.serial(dk):
                        kk = A.cast(faux.pop(), f18)
                        for b in A.serial(nb):
                            s = A.cast(fs.pop(), f18)
                            t = alv * s
                            t2 = kk * vna[b]
                            t = t + t2
                            oo.push(A.cast(t, f32))
                for i in A.serial(dk):
                    for b in A.serial(nb):
                        with A.ODR():
                            snew[i, b] = oo.pop()

    def implementation(self) -> None:
        import numpy as _np

        bh_n, seq = self._bh(), self._seq()
        dk, nb = self._dk(), self._nb()
        T = seq  # device processes the whole seq as one chunk
        f32 = "float32x64"

        # C-in-lanes constants (depend only on T<=64): onehot[t] lane t=1; slt[t] lanes<t=1.
        onehot_np = _np.zeros((T, 1, 1, PWORD), _np.float32)
        slt_np = _np.zeros((T, 1, 1, PWORD), _np.float32)
        for t in range(T):
            onehot_np[t, 0, 0, t] = 1.0
            slt_np[t, 0, 0, :t] = 1.0
        onehots = A.alloc_constant(onehot_np, dtype=f32, scope=A.DDR)
        slts = A.alloc_constant(slt_np, dtype=f32, scope=A.DDR)
        # Underscore-named, not deleted: alloc_constant registers into the block's
        # constant table, so dropping the call would change the emitted IR.
        _one_c = A.alloc_constant(_np.ones((1, 1, PWORD), _np.float32), dtype=f32, scope=A.DDR)
        _zcv = A.alloc_constant(_np.zeros((T, 1, nb, PWORD), _np.float32), dtype=f32, scope=A.DDR)
        _zero11 = A.alloc_constant(_np.zeros((1, 1, PWORD), _np.float32), dtype=f32, scope=A.DDR)
        zero64 = A.alloc_constant(
            _np.zeros((PWORD, 1, PWORD), _np.float32), dtype=f32, scope=A.DDR
        )

        with A.grid(bh_n) as bidx:
            bh = to_tuple(bidx)[0]
            S0 = A.alloc_buffer((dk, nb), f32, scope=A.L1)
            atomic.copy(self.s0[bh, :, :], output=S0)

            K = A.alloc_buffer((T, 1, nb), f32, scope=A.L1)
            Q = A.alloc_buffer((T, 1, nb), f32, scope=A.L1)
            V = A.alloc_buffer((T, 1, nb), f32, scope=A.L1)
            al = A.alloc_buffer((T, 1, 1), f32, scope=A.L1)
            be = A.alloc_buffer((T, 1, 1), f32, scope=A.L1)
            atomic.copy(self.q[bh, :, :, :], output=Q)
            atomic.copy(self.k[bh, :, :, :], output=K)
            atomic.copy(self.v[bh, :, :, :], output=V)
            atomic.copy(self.ab[bh, :, :, 0:1], output=al)  # (seq,1,1) == (T,1,1); no reshape
            atomic.copy(self.ab[bh, :, :, 1:2], output=be)

            # Kt: transpose K (T tokens x Dk) -> (Dk, C-in-lanes). Per Dk-block, pad T->64
            # and vector_transpose (rows<->lanes), so Kt[dk] lane c = K[c, dk] (lanes>=T zero).
            Kt = A.alloc_buffer((dk, 1), f32, scope=A.L1)
            for blk in range(nb):
                Kpad = A.alloc_buffer((PWORD, 1), f32, scope=A.L1)
                atomic.copy(zero64, output=Kpad)
                # copy the strided K-column to a fresh contiguous (T,1,1) buffer first --
                # reshaping the strided slice directly aliases the whole K buffer (T*nb)
                # at a smaller claimed size (T) -> buffer-aliasing error.
                kcol = A.alloc_buffer((T, 1, 1), f32, scope=A.L1)
                atomic.copy(K[:, :, blk : blk + 1], output=kcol)
                atomic.copy(atomic.reshape(kcol, shape=(T, 1)), output=Kpad[0:T, :])
                atomic.copy(
                    atomic.vector_transpose(Kpad, 0, 1),
                    output=Kt[blk * PWORD : (blk + 1) * PWORD, :],
                )

            KKb = A.alloc_buffer((T, 1, 1), f32, scope=A.L1)
            QKb = A.alloc_buffer((T, 1, 1), f32, scope=A.L1)
            KS0 = A.alloc_buffer((T, 1, nb), f32, scope=A.L1)
            QS0 = A.alloc_buffer((T, 1, nb), f32, scope=A.L1)
            Abuf = A.alloc_buffer((T, 1, 1), f32, scope=A.L1)
            Rbuf = A.alloc_buffer((T, 1, 1), f32, scope=A.L1)
            Mbuf = A.alloc_buffer((T, 1, 1), f32, scope=A.L1)
            Pbuf = A.alloc_buffer((T, 1, 1), f32, scope=A.L1)
            RHS = A.alloc_buffer((T, 1, nb), f32, scope=A.L1)
            AQS0 = A.alloc_buffer((T, 1, nb), f32, scope=A.L1)
            Vnew = A.alloc_buffer((T, 1, nb), f32, scope=A.L1)
            outb = A.alloc_buffer((T, 1, nb), f32, scope=A.L1)

            # Front end: KK, QK (C-in-lanes) and KS0, QS0 (Dv-lanes). Each is a full C-row matmul
            # in one datapath (output-row loop folded into the DPU via per-row register accumulators),
            # 4 launches total. Spread K,Q to broadcast columns KB[(T,dk,1)] (KB[i] = row i broadcast);
            # the broadcast operand is T*dk pwords = 2 MiB at C=64. Materialising K and Q at once
            # = 4 MiB = all L1, so reuse one buffer: spread K -> KB, do KK + KS0, then overwrite
            # KB with Q, do QK + QS0.
            KB = A.alloc_buffer((T, dk, 1), f32, scope=A.L1)

            F16L = 16  # float32x16 lanes for VTRSP (f32x64 -> f32x16 alias factor PWORD//F16L)
            SCALE = PWORD // F16L

            def _spread_bcast(src, dst, R, W, S):
                # Fused broadcast-spread src (R,1,S) lane-packed -> dst (R,W,1) (dst[r,l,:]=src val l)
                # in one DWPU datapath. VTRSP needs f32x16, so alias both buffers (scale-4) and work
                # in 16-lane chunks: per (out-chunk c, out-lane-chunk j) push the c-th 16-lane chunk
                # F16L times (repeat) then A.vtrsp the 16x16 tile (ODR, single store/pop) -> row p =
                # broadcast16(lane c*16+p); the SCALE j-tiles fill all 64 output lanes.
                # Used for K/Q (R=T,W=dk,S=nb), M/P (R=T,W=C,S=1), Kgᵀ (R=dk,W=T,S=1).
                f16 = "float32x16"
                src_a = A.alloc_buffer((R, 1, S * SCALE), f16, data=src.data)  # f32x16 aliases
                dst_a = A.alloc_buffer((R, W, SCALE), f16, data=dst.data)
                with A.script():
                    with A.Datapath(kind=A.DP_DWPU):
                        with A.grid(R) as tf:
                            t = to_tuple(tf)[0]
                            with A.grid(W // F16L) as cf:
                                c = to_tuple(cf)[0]
                                with A.grid(SCALE) as jf:
                                    j = to_tuple(jf)[0]
                                    ifd = A.decl_stream(f16)
                                    for p in A.serial(F16L):
                                        with A.IFD(num=0):
                                            ifd.push(src_a[t, 0, c])
                                    for p in A.serial(F16L):
                                        with A.ODR():
                                            dst_a[t, c * F16L + p, j] = A.vtrsp(ifd.pop())

            # _matmul_cw holds M*nb register accumulators (one bank, depth 64); tile the C rows so
            # M*nb <= 64 (C=64,nb=2 -> two M=32 tiles; C<=32 -> a single tile).
            tile = PWORD // nb  # max output rows whose nb-block accumulators fit one bank

            def _proj_tiled(KB, dst):
                for r0 in range(0, T, tile):
                    self._matmul_cw(KB, S0, dst, min(tile, T - r0), dk, nb, r0)

            _spread_bcast(K, KB, T, dk, nb)
            self._matmul_cc(KB, Kt, KKb, T, dk)
            _proj_tiled(KB, KS0)
            _spread_bcast(Q, KB, T, dk, nb)
            self._matmul_cc(KB, Kt, QKb, T, dk)
            _proj_tiled(KB, QS0)

            # Decay scan (two fused carry datapaths): A_t = cumprod(alpha); R[t,i] = A_t/A_i.
            self._scan_A(al, Abuf, T)
            self._scan_R(al, onehots, Rbuf, T)

            # M = beta R KK (strict-lower); P = R QK (lower incl diag); RHS = beta(V - A KS0).
            # Fully vectorised over the T rows (full-buffer elementwise, no per-row slicing).
            atomic.copy(atomic.mul(atomic.mul(atomic.mul(Rbuf, KKb), be), slts), output=Mbuf)
            atomic.copy(atomic.mul(Rbuf, QKb), output=Pbuf)
            atomic.copy(atomic.mul(be, atomic.sub(V, atomic.mul(Abuf, KS0))), output=RHS)
            atomic.copy(atomic.mul(Abuf, QS0), output=AQS0)  # diag(A) Q S0, vectorised

            # Solve (I+M) Vnew = RHS by blocked forward substitution. Broadcast M (Mbuf C-in-lanes
            # -> M_bcast[t,i]) with the fused spread; then per Bs-row block: off-diagonal update
            # (rhsp = RHS - M[block,prior]@Vnew[prior], a HW-loop matmul) + diagonal solve
            # (Bs-row register-feedback solve, per Dv-block). Cross-block feedback via L1.
            M_bcast = A.alloc_buffer((T, T, 1), f32, scope=A.L1)
            _spread_bcast(Mbuf, M_bcast, T, T, 1)
            rhsp = A.alloc_buffer((T, 1, nb), f32, scope=A.L1)
            Bs = 16
            NBk = (T + Bs - 1) // Bs
            sb = [min(kk * Bs, T) for kk in range(NBk + 1)]
            for kk in range(NBk):
                b0, rws = sb[kk], sb[kk + 1] - sb[kk]
                if b0 > 0:
                    self._solve_offdiag(M_bcast, RHS, Vnew, rhsp, b0, rws)
                    for w in range(nb):
                        self._solve_diag(M_bcast, rhsp, Vnew, b0, rws, w)
                else:
                    for w in range(nb):  # block 0: no off-diagonal, solve RHS directly
                        self._solve_diag(M_bcast, RHS, Vnew, b0, rws, w)

            # O = AQS0 + P Vnew: broadcast P then matmul over the chunk with AQS0 init.
            # Tiled over output rows so rows*nb <= 64.
            P_bcast = A.alloc_buffer(
                (T, T, 1), f32, data=M_bcast.data, scope=A.L1
            )  # reuse M_bcast (1MB)
            _spread_bcast(Pbuf, P_bcast, T, T, 1)
            otile = PWORD // nb
            for r0 in range(0, T, otile):
                self._output_matmul(P_bcast, AQS0, Vnew, outb, r0, min(otile, T - r0), T)

            atomic.copy(outb, output=self.output[bh, :, :, :])

            # State: S_T = A_{T-1} S0 + Kgammaᵀ @ Vnew (Kgamma[i] = R[T-1,i] k_i). Build Kgammaᵀ
            # C-in-lanes from Kt*R[T-1], broadcast-spread it, init from A_{T-1} S0, then reuse
            # _output_matmul tiled over dk. KgT_bcast reuses KB's memory (free after front-end).
            rl_src = A.alloc_buffer((1, 1, 1), f32, scope=A.L1)
            atomic.copy(Rbuf[T - 1, :, :], output=rl_src)
            KgT_lanes = A.alloc_buffer(
                (dk, 1), f32, scope=A.L1
            )  # Kgammaᵀ C-in-lanes: lane i = Kg[i,l]
            atomic.copy(atomic.mul(Kt, atomic.reshape(rl_src, shape=(1, 1))), output=KgT_lanes)
            KgT_bcast = A.alloc_buffer((dk, T, 1), f32, data=KB.data, scope=A.L1)  # reuse KB mem
            _spread_bcast(KgT_lanes, KgT_bcast, dk, T, 1)
            ab_src = A.alloc_buffer((1, 1, 1), f32, scope=A.L1)
            atomic.copy(Abuf[T - 1, :, :], output=ab_src)
            AS_init = A.alloc_buffer((dk, 1, nb), f32, scope=A.L1)  # A_{T-1} * S0
            atomic.copy(
                atomic.mul(
                    atomic.reshape(S0, shape=(dk, 1, nb)), atomic.reshape(ab_src, shape=(1, 1, 1))
                ),
                output=AS_init,
            )
            S_out = A.alloc_buffer((dk, 1, nb), f32, scope=A.L1)
            stile = PWORD // nb
            for r0 in range(0, dk, stile):
                self._output_matmul(KgT_bcast, AS_init, Vnew, S_out, r0, min(stile, dk - r0), T)

            if self.state_inplace:
                atomic.copy(atomic.reshape(S_out, shape=(dk, nb)), output=self.s0[bh, :, :])
            else:
                atomic.copy(atomic.reshape(S_out, shape=(dk, nb)), output=self.state_out[bh, :, :])

    def emulate(self) -> None:  # type: ignore[override]
        bh_n, seq = self._bh(), self._seq()
        dk, nb = self._dk(), self._nb()
        dv = nb * PWORD
        C = self._chunk()

        qr = self.q.reshape(bh_n, seq, dk).astype(np.float64)
        kr = self.k.reshape(bh_n, seq, dk).astype(np.float64)
        vv = self.v.reshape(bh_n, seq, dv).astype(np.float64)
        al = self.ab[:, :, 0, 0, 0].astype(np.float64)  # (BH, seq)
        be = self.ab[:, :, 0, 1, 0].astype(np.float64)

        out = np.zeros((bh_n, seq, dv), np.float32)
        state_final = np.zeros((bh_n, dk, dv), np.float32)
        for bh in range(bh_n):
            S0 = self.s0[bh].reshape(dk, dv).astype(np.float32)
            o, S = _chunk_fp18(qr[bh], kr[bh], vv[bh], al[bh], be[bh], S0, C)
            out[bh] = o
            state_final[bh] = S
        np.copyto(self.output, out.reshape(bh_n, seq, 1, nb, PWORD))
        packed = state_final.reshape(bh_n, dk, nb, PWORD)
        np.copyto(self.state_out, packed)
        if self.state_inplace:
            np.copyto(self.s0, packed)


gated_deltanet_chunk_fp = register_op(
    GatedDeltaNetChunkFp,
    "llm.gated_deltanet_chunk_fp",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)

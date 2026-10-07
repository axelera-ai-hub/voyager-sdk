#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Pure-fp18 sequential gated-delta recurrence for prefill (llm.gated_deltanet_recurrent_fp).

The token-mixing core of the gated DeltaNet as the exact sequential recurrence in float18
on the DPU -- no chunking and no int8 anywhere in the core. For BH heads it walks all seq
tokens, carrying that head's (Dk, Dv) recurrent state S in float32 in L1, and per token::

    kS = Σ_Dk(S ⊙ k) ;  qS = Σ_Dk(S ⊙ q) ;  qk = Σ_Dk(q ⊙ k)
    v_new = β·(v − α·kS)
    o     = α·qS + qk·v_new                                  (the readout)
    S     = α·S + k ⊗ v_new                                  (rank-1 state update)

α = exp g is the per-token decay, β = sigmoid b; the qk term folds the within-token
rank-1 update into the readout so S is only updated once per token.

S is stored (Dk, Dv) -- Dk on the row (grid) axis, Dv in the lanes (NB blocks). The
contraction in the readout/outer-product is over Dk, so q/k are consumed row-packed
((Dk, 1), scalar broadcast across 64 lanes); the caller pre-transposes them once so this
op's hot loop is transpose-free. v/v_new/o/kS/qS stay lane-packed (1, NB). Both grids are
A.grid HARDWARE loops (A.grid(BH) outer, A.grid(seq) inner) so program size is constant in
context length. S carries across the token loop in an L1 buffer (NOT a DPU register array
held across the grid -- that corrupts, cf. the decode op). Each token is three hand-fused
DPU datapaths (readout/output/state-update); every IFD reads from a single buffer (two
distinct buffers on one IFD deadlock the device -- see _fwd_solve in the chunk op).

The carried state is initialised from s0 and (with state_inplace) written back into it in
place -- the persistent-cache RMW the decoder layer uses for the recurrent state (cf.
causal_conv1d); otherwise it is emitted as state_out.
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


class GatedDeltaNetRecurrentFp(ScriptOp):
    """BH-head, full-sequence fp18 sequential gated-delta recurrence (A.grid over heads/tokens)."""

    q = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(BH, seq, 1, NB) lane-packed L2-normed query (Dk in lanes); the op "
        "transposes it lane->row per token internally (vector_transpose, like the decode "
        "per_row / chunk fwd-solve idiom), so the caller passes the native head-major layout.",
    )
    k = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(BH, seq, 1, NB) lane-packed L2-normed key (same layout as q).",
    )
    v = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(BH, seq, 1, NB) lane-packed value v_t (Dv in lanes, NB blocks); the "
        "size-1 axis keeps the per-token slice rank-2 for the staging copies.",
    )
    ab = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(BH, seq, 1, 2) per-(head,token) [alpha, beta] = [exp(g), sigmoid(b)], "
        "each broadcast across the 64 lanes (size-1 axis keeps the slice rank-2).",
    )
    s0 = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(BH, Dk, NB) entry state S per head (Dk rows, Dv in lanes).",
    )
    output = Arg(
        MetisFloats[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="Per-(head,token) readout o_t, auto-inferred (BH, seq, 1, NB) float32x64 DDR.",
    )
    state_out = Arg(
        MetisFloats[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="Final state per head after seq tokens, auto-inferred (BH, Dk, NB) DDR.",
    )
    state_inplace = Arg(
        bool,
        default=False,
        kw=True,
        description="Write the final state back into s0 in place (persistent-cache RMW, cf. "
        "causal_conv1d / the chunk op) and emit no separate state_out.",
    )
    token_major = Arg(
        bool,
        default=False,
        kw=True,
        description="Lay q/k/v/output out TOKEN-major (seq, BH, NB) instead of head-major "
        "(BH, seq, 1, NB). The per-head whole-sequence slab then loads as a SINGLE strided 2D "
        "DMA (one descriptor: NB contiguous, row-stride BH*NB, seq rows) instead of a contiguous "
        "block, which lets the caller feed the conv/projection output directly and skip the "
        "explicit head-major transpose (the prefill's to_heads/from_heads). ab/s0/state_out stay "
        "head-major -- the gates are tiny per-token scalars and the state is loaded once per head.",
    )
    generation = Arg(HardwareGeneration | None, default=None, kw=True)

    def _token_major(self) -> bool:
        return bool(self.token_major)

    def _bh(self) -> int:
        return int(self.q.shape[1] if self._token_major() else self.q.shape[0])

    def _seq(self) -> int:
        return int(self.q.shape[0] if self._token_major() else self.q.shape[1])

    def _nb(self) -> int:
        # NOT -1: under emulate the numpy array carries a trailing PWORD lane axis, so -1
        # would grab the lanes. Blocks axis is 2 token-major (seq, BH, NB) / 3 head-major
        # (BH, seq, 1, NB).
        return int(self.q.shape[2] if self._token_major() else self.q.shape[3])

    def _dk(self) -> int:
        return self._nb() * PWORD  # Dk == Dv (square state)

    def validate(self) -> None:
        if self._dk() % PWORD != 0:
            raise ValueError(f"Dk must be a multiple of {PWORD}; got {self._dk()}.")
        if self._dk() // PWORD != self._nb():
            raise ValueError(f"Dk == Dv required (square state): Dk={self._dk()} NB={self._nb()}.")

    def infer_outputs(self) -> dict[str, BufferInfo]:
        bh, seq, nb = self._bh(), self._seq(), self._nb()
        dk = self._dk()
        out_shape = (seq, bh, nb) if self._token_major() else (bh, seq, 1, nb)
        return {
            "output": BufferInfo(shape=out_shape, dtype="float32x64", scope=A.DDR),
            "state_out": BufferInfo(shape=(bh, dk, nb), dtype="float32x64", scope=A.DDR),
        }

    def _per_row(self, vec, dst) -> None:
        """Transpose one lane-packed row vec (1, NB) -> row-packed dst (Dk, 1):
        dst[dk] = scalar vec[dk] broadcast across the 64 lanes. Per NB-block,
        vector_transpose(repeat(vec[:,blk], (64,1))) splats lane j across a pword."""
        nb = self._nb()
        for blk in range(nb):
            vb = atomic.reshape(atomic.copy(vec[:, blk : blk + 1]), shape=(1, 1))
            atomic.copy(
                atomic.vector_transpose(atomic.repeat(vb, (PWORD, 1)), 0, 1),
                output=dst[blk * PWORD : (blk + 1) * PWORD, :],
            )

    # The readout is three single-accumulator datapaths (kS, qS via _matvec; qk via _dot)
    # rather than one fused datapath: a single DPU datapath holding two NB accumulators +
    # qk over the full Dk reduction overflows the DPU instruction memory.
    def _matvec(self, state, vrow, rd, dst_off) -> None:
        """rd[0, dst_off+b] = Σ_dk state[dk,b]*vrow[dk]; vrow (Dk,1) broadcast across lanes.
        S on IFD0 (single buffer), per-row scalar on IFD1 (single buffer); one NB-wide
        accumulator -- the matvec k@S / q@S."""
        dk, nb = self._dk(), self._nb()
        f32, f18 = "float32x64", "float18x64"
        with A.script():
            with A.Datapath(core=None):
                fs = A.decl_stream(f32)
                fv = A.decl_stream(f32)
                o = A.decl_stream(f32)
                for i in A.serial(dk):
                    for b in A.serial(nb):
                        with A.IFD(num=0):
                            fs.push(state[i, b])
                for i in A.serial(dk):
                    with A.IFD(num=1):
                        fv.push(vrow[i, 0])
                with A.DPU():
                    acc = A.array(f18, nb)
                    vv = A.var(f18)
                    t = A.var(f18)
                    vv = A.cast(fv.pop(), f18)  # peel dk=0 to init the accumulator
                    for b in A.serial(nb):
                        acc[b] = A.cast(fs.pop(), f18) * vv
                    for i in A.serial(1, dk):
                        vv = A.cast(fv.pop(), f18)
                        for b in A.serial(nb):
                            t = A.cast(fs.pop(), f18) * vv
                            acc[b] = acc[b] + t
                    for b in A.serial(nb):
                        o.push(A.cast(acc[b], f32))
                for b in A.serial(nb):
                    with A.ODR():
                        rd[0, dst_off + b] = o.pop()

    def _dot(self, krow, qrow, rd, dst) -> None:
        """rd[0, dst] = qk = Σ_dk k[dk]*q[dk] (both (Dk,1) broadcast pwords, so each lane
        carries the scalar product summed over Dk)."""
        dk = self._dk()
        f32, f18 = "float32x64", "float18x64"
        with A.script():
            with A.Datapath(core=None):
                fk = A.decl_stream(f32)
                fq = A.decl_stream(f32)
                o = A.decl_stream(f32)
                for i in A.serial(dk):
                    with A.IFD(num=0):
                        fk.push(krow[i, 0])
                for i in A.serial(dk):
                    with A.IFD(num=1):
                        fq.push(qrow[i, 0])
                with A.DPU():
                    acc = A.var(f18)
                    t = A.var(f18)
                    acc = A.cast(fk.pop(), f18) * A.cast(fq.pop(), f18)
                    for i in A.serial(1, dk):
                        t = A.cast(fk.pop(), f18) * A.cast(fq.pop(), f18)
                        acc = acc + t
                    o.push(A.cast(acc, f32))
                with A.ODR():
                    rd[0, dst] = o.pop()

    def _readout(self, state, kl, ql, rd) -> None:
        """kS -> rd[0:NB], qS -> rd[NB:2NB], qk -> rd[2NB] (kl/ql row-packed (Dk,1))."""
        nb = self._nb()
        self._matvec(state, kl, rd, 0)
        self._matvec(state, ql, rd, nb)
        self._dot(kl, ql, rd, 2 * nb)

    def _output(self, rd, abt, ov) -> None:
        """Fused output: v_new = beta*(v - alpha*kS); o = alpha*qS + qk*v_new.
        rd (1, 3*NB+1) holds [kS | qS | qk | v] (caller copies v into rd after readout,
        so IFD0 reads one buffer); abt (1,2) = [alpha, beta]. Writes ov (1, 2*NB) = [o | v_new]."""
        nb = self._nb()
        f32, f18 = "float32x64", "float18x64"
        with A.script():
            with A.Datapath(core=None):
                fin = A.decl_stream(f32)  # kS, qS, v, qk   IFD0 (single buffer rd)
                fsc = A.decl_stream(f32)  # alpha, beta     IFD1 (single buffer abt)
                oo = A.decl_stream(f32)
                # rd is contiguous [kS(nb) | qS(nb) | qk(1) | v(nb)] -> one IFD0 push loop
                # (multiple separate IFD blocks on one stream hang the device; sim tolerates it).
                for j in A.serial(3 * nb + 1):
                    with A.IFD(num=0):
                        fin.push(rd[0, j])
                for j in A.serial(2):
                    with A.IFD(num=1):
                        fsc.push(abt[0, j])  # alpha, beta
                with A.DPU():
                    kSa = A.array(f18, nb)
                    qSa = A.array(f18, nb)
                    va = A.array(f18, nb)
                    vna = A.array(f18, nb)
                    ova = A.array(f18, nb)
                    alv = A.var(f18)
                    bev = A.var(f18)
                    qkv = A.var(f18)
                    t = A.var(f18)
                    t2 = A.var(f18)
                    for b in A.serial(nb):
                        kSa[b] = A.cast(fin.pop(), f18)
                    for b in A.serial(nb):
                        qSa[b] = A.cast(fin.pop(), f18)
                    qkv = A.cast(fin.pop(), f18)  # qk (before v, matching rd order)
                    for b in A.serial(nb):
                        va[b] = A.cast(fin.pop(), f18)
                    alv = A.cast(fsc.pop(), f18)
                    bev = A.cast(fsc.pop(), f18)
                    for b in A.serial(nb):
                        t = alv * kSa[b]  # alpha*kS  (one op per DPU statement)
                        t = va[b] - t  # v - alpha*kS
                        vna[b] = bev * t  # v_new
                    for b in A.serial(nb):
                        t = alv * qSa[b]  # alpha*qS
                        t2 = qkv * vna[b]  # qk*v_new
                        ova[b] = t + t2  # o
                    for b in A.serial(nb):
                        oo.push(A.cast(ova[b], f32))
                    for b in A.serial(nb):
                        oo.push(A.cast(vna[b], f32))
                for j in A.serial(2 * nb):  # one ODR section: o(nb), v_new(nb)
                    with A.ODR():
                        ov[0, j] = oo.pop()

    def _update(self, state, avn, kl, snew) -> None:
        """Fused rank-1 state update: S_new[dk,nb] = alpha*S[dk,nb] + k[dk]*v_new[nb].
        S on IFD0; avn (1, 1+NB) = [alpha | v_new] then k (kl, (Dk,1)) on IFD1 (the
        validated 2-source pattern). Writes snew (Dk, NB)."""
        dk, nb = self._dk(), self._nb()
        f32, f18 = "float32x64", "float18x64"
        with A.script():
            with A.Datapath(core=None):
                fs = A.decl_stream(f32)  # S[dk,nb]                       IFD0 (single buffer)
                faux = A.decl_stream(f32)  # alpha, v_new[nb], then k[dk]   IFD1 (2 buffers)
                oo = A.decl_stream(f32)
                for i in A.serial(dk):
                    for b in A.serial(nb):
                        with A.IFD(num=0):
                            fs.push(state[i, b])
                for j in A.serial(1 + nb):  # avn = [alpha | v_new], one push loop
                    with A.IFD(num=1):
                        faux.push(avn[0, j])
                for i in A.serial(dk):
                    with A.IFD(num=1):
                        faux.push(kl[i, 0])  # k[dk] (2nd IFD1 source)
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
                            t = alv * s  # alpha*S  (one op per DPU statement)
                            t2 = kk * vna[b]  # k*v_new
                            t = t + t2
                            oo.push(A.cast(t, f32))
                for i in A.serial(dk):
                    for b in A.serial(nb):
                        with A.ODR():
                            snew[i, b] = oo.pop()

    def implementation(self) -> None:
        bh_n, seq = self._bh(), self._seq()
        dk, nb = self._dk(), self._nb()
        tm = self._token_major()
        f32 = "float32x64"

        with A.grid(bh_n) as bidx:
            bh = to_tuple(bidx)[0]

            # Per-head whole-sequence slab of q/k/v -> (seq, 1, nb). Head-major: a contiguous
            # block (self.q[bh]). Token-major: a strided range-slice on the middle (head) axis
            # (self.q[:, bh:bh+1, :]) -> one strided 2D DMA, no DDR transpose upstream. The
            # size-1 head dim keeps the per-token slice rank-2.
            def head_slab(buf):
                return buf[:, bh : bh + 1, :] if tm else buf[bh, :, :, :]

            state = A.alloc_buffer((dk, nb), f32, scope=A.L1)
            atomic.copy(self.s0[bh, :, :], output=state)

            # Bulk-load the whole head sequence into L1 (one DMA each) and accumulate the
            # output in L1 for one bulk store, so the per-token loop touches DDR zero times.
            # Per-token self.q[bh, ci] loads would fragment into BH*seq tiny strided transfers.
            q_h = A.alloc_buffer((seq, 1, nb), f32, scope=A.L1)
            k_h = A.alloc_buffer((seq, 1, nb), f32, scope=A.L1)
            v_h = A.alloc_buffer((seq, 1, nb), f32, scope=A.L1)
            ab_h = A.alloc_buffer((seq, 1, 2), f32, scope=A.L1)
            out_h = A.alloc_buffer((seq, 1, nb), f32, scope=A.L1)
            atomic.copy(head_slab(self.q), output=q_h)
            atomic.copy(head_slab(self.k), output=k_h)
            atomic.copy(head_slab(self.v), output=v_h)
            atomic.copy(self.ab[bh, :, :, :], output=ab_h)  # ab stays head-major

            with A.grid(seq) as tidx:
                ci = to_tuple(tidx)[0]
                kin = A.alloc_buffer((1, nb), f32, scope=A.L1)
                qin = A.alloc_buffer((1, nb), f32, scope=A.L1)
                abt = A.alloc_buffer((1, 2), f32, scope=A.L1)
                kl = A.alloc_buffer((dk, 1), f32, scope=A.L1)
                ql = A.alloc_buffer((dk, 1), f32, scope=A.L1)
                rd = A.alloc_buffer((1, 3 * nb + 1), f32, scope=A.L1)
                ov = A.alloc_buffer((1, 2 * nb), f32, scope=A.L1)
                avn = A.alloc_buffer((1, 1 + nb), f32, scope=A.L1)
                snew = A.alloc_buffer((dk, nb), f32, scope=A.L1)

                # Stage this token's slices from the per-head L1 slabs (contiguous L1->L1)
                # so the helpers take plain (1,NB)/(1,2) buffers.
                atomic.copy(k_h[ci, :, :], output=kin)
                atomic.copy(q_h[ci, :, :], output=qin)
                atomic.copy(ab_h[ci, :, :], output=abt)

                # lane-packed k/q (1,NB) -> row-packed (Dk,1) for the over-Dk reduction.
                self._per_row(kin, kl)
                self._per_row(qin, ql)
                self._readout(state, kl, ql, rd)
                atomic.copy(v_h[ci, :, :], output=rd[:, 2 * nb + 1 : 3 * nb + 1])
                self._output(rd, abt, ov)
                atomic.copy(ov[:, 0:nb], output=out_h[ci, :, :])

                atomic.copy(abt[:, 0:1], output=avn[:, 0:1])  # alpha
                atomic.copy(ov[:, nb : 2 * nb], output=avn[:, 1 : 1 + nb])  # v_new
                self._update(state, avn, kl, snew)
                atomic.copy(snew, output=state)

            atomic.copy(out_h, output=head_slab(self.output))
            if self.state_inplace:
                atomic.copy(state, output=self.s0[bh, :, :])
            else:
                atomic.copy(state, output=self.state_out[bh, :, :])

    def emulate(self) -> None:  # type: ignore[override]
        bh_n, seq = self._bh(), self._seq()
        dk, nb = self._dk(), self._nb()
        dv = nb * PWORD

        def f18(x):
            return emu.rf18(np.asarray(x, dtype=np.float32))

        def mul(a, b):
            return f18(f18(a) * f18(b))

        def add(a, b):
            return f18(f18(a) + f18(b))

        def sub(a, b):
            return f18(f18(a) - f18(b))

        def fsum(terms):  # left-to-right fp18 accumulation (matches the DPU register adds)
            acc = f18(terms[0])
            for t in terms[1:]:
                acc = add(acc, t)
            return acc

        # Flatten q/k/v to (BH, seq, D). Head-major (BH, seq, 1, NB, PWORD) is a plain
        # reshape; token-major (seq, BH, NB, PWORD) reshapes to (seq, BH, D) then transposes
        # the head axis to front (inverse of the layout the device strided-loads per head).
        if self._token_major():
            qr = self.q.reshape(seq, bh_n, dk).transpose(1, 0, 2).astype(np.float64)
            kr = self.k.reshape(seq, bh_n, dk).transpose(1, 0, 2).astype(np.float64)
            vv = self.v.reshape(seq, bh_n, dv).transpose(1, 0, 2).astype(np.float64)
        else:
            qr = self.q.reshape(bh_n, seq, dk).astype(np.float64)  # (BH, seq, Dk)
            kr = self.k.reshape(bh_n, seq, dk).astype(np.float64)
            vv = self.v.reshape(bh_n, seq, dv).astype(np.float64)  # (BH, seq, Dv)
        # ab is always head-major (BH, seq, 1, 2, PWORD): [...,0,0,0]=alpha, [...,0,1,0]=beta.
        al = self.ab[:, :, 0, 0, 0].astype(np.float64)  # (BH, seq)
        be = self.ab[:, :, 0, 1, 0].astype(np.float64)

        out = np.zeros((bh_n, seq, dv), np.float32)
        state_final = np.zeros((bh_n, dk, dv), np.float32)
        for bh in range(bh_n):
            S = self.s0[bh].reshape(dk, dv).astype(np.float32).copy()  # (Dk, Dv)
            for t in range(seq):
                k = kr[bh, t]  # (Dk,)
                q = qr[bh, t]
                v = vv[bh, t]  # (Dv,)
                a = float(al[bh, t])
                b = float(be[bh, t])
                # kS[dv] = Σ_dk S[dk,dv]*k[dk]; qS likewise; qk = Σ_dk q*k -- fp18 sums
                kS = fsum([mul(S[i], k[i]) for i in range(dk)])  # (Dv,)
                qS = fsum([mul(S[i], q[i]) for i in range(dk)])
                qk = fsum([mul(q[i], k[i]) for i in range(dk)])  # scalar (per-lane equal)
                v_new = mul(sub(v, mul(kS, a)), b)
                out[bh, t] = add(mul(qS, a), mul(v_new, qk)).astype(np.float32)
                # S_new[dk,dv] = alpha*S + k[dk]*v_new[dv]
                S = add(mul(S, a), mul(k[:, None], v_new[None, :])).astype(np.float32)
            state_final[bh] = S
        if self._token_major():  # (BH, seq, Dv) -> token-major (seq, BH, NB, PWORD)
            np.copyto(self.output, out.transpose(1, 0, 2).reshape(seq, bh_n, nb, PWORD))
        else:
            np.copyto(self.output, out.reshape(bh_n, seq, 1, nb, PWORD))
        packed = state_final.reshape(bh_n, dk, nb, PWORD)
        np.copyto(self.state_out, packed)
        if self.state_inplace:
            np.copyto(self.s0, packed)


gated_deltanet_recurrent_fp = register_op(
    GatedDeltaNetRecurrentFp,
    "llm.gated_deltanet_recurrent_fp",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)

#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Recurrent gated delta rule -- one linear-attention step (llm.gated_deltanet).

The "gated DeltaNet" layer (Yang et al. 2024, as used in Qwen3-Next / Qwen 3.5
linear-attention blocks) replaces softmax attention with a matrix-valued
recurrent state S of shape (Dk, Dv) per head, updated by the gated
delta rule.  Per timestep t (all quantities are per head)::

    S <- alpha_t * S                               # scalar decay gate (Mamba2-style)
    v_old = k_t @ S            # (Dv,)              # value the state predicts for k_t
    v_new = beta_t * (v_t - v_old)   # (Dv,)        # delta-rule correction, beta_t in (0,1)
    S <- S + outer(k_t, v_new)                      # rank-1 write
    o_t = q_t @ S              # (Dv,)              # read out with the query

with q_t, k_t in R^Dk (L2-normalised upstream), v_t in R^Dv, the decay
alpha_t = exp(g_t) in (0, 1) and the write strength beta_t in (0, 1) both
scalar per (head, timestep).

This kernel runs the recurrence sequentially over the T timesteps it is
handed, carrying S in L1 across the steps, and reads the initial state from
state_in / writes the final state to state_out -- so the caller persists
S across kernel invocations exactly like a KV cache (cf.
axol.core.update_kv_cache).  It therefore covers decode (T = 1) and short
prefill chunks; long prefill wants the chunkwise WY-form algorithm (a matmul
formulation), which is left as future work -- see the note at the bottom.

Layout / packing requirements
-----------------------------
The state is stored Dv-major in the lanes: S is (Dk, Dv_blocks) where
each row k is a vector over the Dv value positions (Dv = Dv_blocks *
PWORD).  Every step is then a sequence of scalar-broadcast FMAs over the
Dk state rows -- v_old += k_t[k] * S[k, :] and S[k, :] += k_t[k] *
v_new -- which the DPU does natively, with no transpose and no cross-lane
reduction.  The price is that q and k must be supplied broadcast-packed:
each logical scalar q_t[k] / k_t[k] is replicated across all PWORD
lanes (device shape (B, H, T, Dk) of float32x64).  alpha and beta
are likewise scalar-broadcast, (B, H, T, 1).  v and the output / state are
ordinary Dv-blocked vectors.

All DPU arithmetic is float18 (state is held fp32 in L1/DDR between steps); the
numpy emulate mirrors the op order with emu.rf18 at every step.

Each timestep lowers to three single-output DPU datapaths (v_old/v_new,
state update, readout).  This split is not just for the one-ODR-per-datapath
rule: a datapath that reads the state with two IFD0 commands (e.g. one
fused update datapath reading S for both the v_old reduction and the rank-1
write) compiles and runs correctly in simulation but hangs the device
(MSI timeout / DEVICE_LOST).  Keeping each datapath to a single IFD0 read --
reading S again in a separate datapath instead -- is what makes it run on
hardware.  Validated on Metis (Omega) for decode and short multi-step prefill.
"""
from axelera.compiler.alto.axol import atomic
from axelera.compiler.alto.axol.utils import to_int_tuple, to_tuple
from axelera.compiler.alto.compiler.config import HardwareGeneration
from axelera.compiler.alto.compiler.dtypes import MetisFloats
from axelera.compiler.alto.compiler.interpreter import emulation as emu
from axelera.compiler.alto.compiler.op_registry import Arg, BufferInfo, ScriptOp, register_op
import axelera.compiler.alto.script as A
import numpy as np
from tvm import tir

# mypy: disable-error-code="attr-defined, index, arg-type"
# flake8: noqa: C901


class GatedDeltaNet(ScriptOp):
    """One (or more) recurrent gated-delta-rule timesteps. See module docstring.

    Operands are batched over (batch, num_heads) and the T timesteps are
    walked sequentially, carrying the (Dk, Dv) state in L1.
    """

    query = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(B, H, T, Dk) broadcast-packed query (each lane = q_t[k]).",
    )
    key = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(B, H, T, Dk) broadcast-packed key (each lane = k_t[k]).",
    )
    value = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(B, H, T, Dv_blocks) value, Dv = Dv_blocks * PWORD.",
    )
    alpha = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(B, H, T, 1) broadcast-packed decay gate exp(g_t) in (0, 1).",
    )
    beta = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(B, H, T, 1) broadcast-packed write strength beta_t in (0, 1).",
    )
    state_in = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(B, H, Dk, Dv_blocks) initial recurrent state S.",
    )

    output = Arg(
        MetisFloats[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="Per-step readout, auto-inferred (B, H, T, Dv_blocks) float32x64 DDR.",
    )
    state_out = Arg(
        MetisFloats[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="Final state after T steps, auto-inferred (B, H, Dk, Dv_blocks) DDR.",
    )
    generation = Arg(
        HardwareGeneration | None,
        default=None,
        kw=True,
        description="Optional hardware generation.",
    )

    # Shape helpers
    def _dims(self) -> tuple[int, int, int, int, int]:
        """(B, H, T, Dk, Dv_blocks) from the operand shapes."""
        b, h, t, dk = to_int_tuple(self.query.shape)
        dv_blocks = int(self.value.shape[-1])
        return b, h, t, dk, dv_blocks

    def validate(self) -> None:
        if len(self.query.shape) != 4:
            raise ValueError(f"query must be (B, H, T, Dk); got {tuple(self.query.shape)}.")
        if to_int_tuple(self.key.shape) != to_int_tuple(self.query.shape):
            raise ValueError(
                f"key {tuple(self.key.shape)} must match query {tuple(self.query.shape)}."
            )
        b, h, t, dk, dv_blocks = self._dims()
        if to_int_tuple(self.value.shape) != (b, h, t, dv_blocks):
            raise ValueError(
                f"value must be (B, H, T, Dv_blocks) = {(b, h, t, dv_blocks)}; "
                f"got {tuple(self.value.shape)}."
            )
        for nm, buf in (("alpha", self.alpha), ("beta", self.beta)):
            if to_int_tuple(buf.shape) != (b, h, t, 1):
                raise ValueError(
                    f"{nm} must be (B, H, T, 1) = {(b, h, t, 1)}; got {tuple(buf.shape)}."
                )
        if to_int_tuple(self.state_in.shape) != (b, h, dk, dv_blocks):
            raise ValueError(
                f"state_in must be (B, H, Dk, Dv_blocks) = {(b, h, dk, dv_blocks)}; "
                f"got {tuple(self.state_in.shape)}."
            )

    def infer_outputs(self) -> dict[str, BufferInfo]:
        b, h, t, dk, dv_blocks = self._dims()
        return {
            "output": BufferInfo(shape=(b, h, t, dv_blocks), dtype="float32x64", scope=A.DDR),
            "state_out": BufferInfo(shape=(b, h, dk, dv_blocks), dtype="float32x64", scope=A.DDR),
        }

    # Implementation
    def implementation(self) -> None:
        b, h, t, dk, dv_blocks = self._dims()
        f32 = "float32x64"

        # Per-head L1 staging for the small per-step operands; the state is
        # double-buffered (read cur / write nxt) so the in-step S read never
        # aliases the S write.
        s_a = A.alloc_buffer((dk, dv_blocks), f32, scope=A.L1)
        s_b = A.alloc_buffer((dk, dv_blocks), f32, scope=A.L1)
        q_l1 = A.alloc_buffer((t, dk), self.query.dtype, scope=A.L1)
        k_l1 = A.alloc_buffer((t, dk), self.key.dtype, scope=A.L1)
        v_l1 = A.alloc_buffer((t, dv_blocks), self.value.dtype, scope=A.L1)
        alpha_l1 = A.alloc_buffer((t, 1), self.alpha.dtype, scope=A.L1)
        beta_l1 = A.alloc_buffer((t, 1), self.beta.dtype, scope=A.L1)
        vnew_l1 = A.alloc_buffer((dv_blocks,), f32, scope=A.L1)
        o_l1 = A.alloc_buffer((t, dv_blocks), f32, scope=A.L1)

        # A recurrent step is three single-output datapaths, each reading the
        # state with a single IFD0 command (two IFD0 reads in one datapath
        # hangs the device even though it works in simulation):
        #   1. _step_vold  -- read cur_S, k, v, alpha, beta; write v_new.
        #   2. _step_state -- read cur_S, k, v_new, alpha; write nxt_S.
        #   3. _step_readout -- read the fresh nxt_S and q; write o_t.
        def _step_vold(cur_S, ti) -> None:
            # v_old = sum_k k_t[k] * (alpha * S[k]) ; v_new = beta * (v_t - v_old)
            with A.Datapath(core=None):
                with (
                    A.decl_stream(f32) as s_stream,
                    A.decl_stream(self.key.dtype) as k_stream,
                    A.decl_stream(self.value.dtype) as v_stream,
                    A.decl_stream(f32) as ab_stream,
                    A.decl_stream(f32) as vnew_stream,
                ):
                    # IFD0: cur_S. IFD1 push/pop order: alpha, beta, v, k.
                    with A.grid(dk, dv_blocks) as idx:
                        kk, vb = to_tuple(idx)
                        with A.IFD(num=0):
                            A.stream_push(s_stream, cur_S[kk, vb])
                    with A.IFD(num=1):
                        A.stream_push(ab_stream, alpha_l1[ti, 0])
                    with A.IFD(num=1):
                        A.stream_push(ab_stream, beta_l1[ti, 0])
                    with A.grid(dv_blocks) as vb:
                        with A.IFD(num=1):
                            A.stream_push(v_stream, v_l1[ti, to_tuple(vb)[0]])
                    with A.grid(dk) as kk:
                        with A.IFD(num=1):
                            A.stream_push(k_stream, k_l1[ti, to_tuple(kk)[0]])

                    with A.DPU():
                        alpha = A.var("float18x64")
                        beta = A.var("float18x64")
                        vt = A.array("float18x64", size=dv_blocks)
                        v_old = A.array("float18x64", size=dv_blocks)
                        kc = A.var("float18x64")
                        sval = A.var("float18x64")
                        dec = A.var("float18x64")
                        prod = A.var("float18x64")

                        A.buffer_store(alpha, A.cast(ab_stream.pop(), "float18x64"), (0,))
                        A.buffer_store(beta, A.cast(ab_stream.pop(), "float18x64"), (0,))
                        with A.unroll(0, dv_blocks) as vb:
                            A.buffer_store(vt, A.cast(v_stream.pop(), "float18x64"), (vb,))

                        # Peel kk == 0 to seed the accumulator.
                        A.buffer_store(kc, A.cast(k_stream.pop(), "float18x64"), (0,))
                        with A.unroll(0, dv_blocks) as vb:
                            A.buffer_store(sval, A.cast(s_stream.pop(), "float18x64"), (0,))
                            A.buffer_store(dec, sval[0] * alpha[0], (0,))
                            A.buffer_store(v_old, kc[0] * dec[0], (vb,))
                        with A.serial(1, dk) as kk:
                            A.buffer_store(kc, A.cast(k_stream.pop(), "float18x64"), (0,))
                            with A.unroll(0, dv_blocks) as vb:
                                A.buffer_store(sval, A.cast(s_stream.pop(), "float18x64"), (0,))
                                A.buffer_store(dec, sval[0] * alpha[0], (0,))
                                A.buffer_store(prod, kc[0] * dec[0], (0,))
                                A.buffer_store(v_old, v_old[vb] + prod[0], (vb,))

                        # v_new = beta * (v_t - v_old)
                        with A.unroll(0, dv_blocks) as vb:
                            A.buffer_store(prod, vt[vb] - v_old[vb], (0,))
                            A.stream_push(vnew_stream, A.cast(beta[0] * prod[0], f32))

                    with A.grid(dv_blocks) as vb:
                        with A.ODR():
                            A.buffer_store(vnew_l1, vnew_stream.pop(), (to_tuple(vb)[0],))

        def _step_state(cur_S, nxt_S, ti) -> None:
            # nxt_S[k] = alpha * S[k] + k_t[k] * v_new
            with A.Datapath(core=None):
                with (
                    A.decl_stream(f32) as s_stream,
                    A.decl_stream(self.key.dtype) as k_stream,
                    A.decl_stream(f32) as vnew_stream,
                    A.decl_stream(f32) as a_stream,
                    A.decl_stream(f32) as snew_stream,
                ):
                    # IFD0: cur_S. IFD1 push/pop order: alpha, v_new, k.
                    with A.grid(dk, dv_blocks) as idx:
                        kk, vb = to_tuple(idx)
                        with A.IFD(num=0):
                            A.stream_push(s_stream, cur_S[kk, vb])
                    with A.IFD(num=1):
                        A.stream_push(a_stream, alpha_l1[ti, 0])
                    with A.grid(dv_blocks) as vb:
                        with A.IFD(num=1):
                            A.stream_push(vnew_stream, vnew_l1[to_tuple(vb)[0]])
                    with A.grid(dk) as kk:
                        with A.IFD(num=1):
                            A.stream_push(k_stream, k_l1[ti, to_tuple(kk)[0]])

                    with A.DPU():
                        alpha = A.var("float18x64")
                        v_new = A.array("float18x64", size=dv_blocks)
                        kc = A.var("float18x64")
                        sval = A.var("float18x64")
                        dec = A.var("float18x64")
                        prod = A.var("float18x64")
                        row = A.var("float18x64")

                        A.buffer_store(alpha, A.cast(a_stream.pop(), "float18x64"), (0,))
                        with A.unroll(0, dv_blocks) as vb:
                            A.buffer_store(v_new, A.cast(vnew_stream.pop(), "float18x64"), (vb,))

                        with A.serial(0, dk) as kk:
                            A.buffer_store(kc, A.cast(k_stream.pop(), "float18x64"), (0,))
                            with A.unroll(0, dv_blocks) as vb:
                                A.buffer_store(sval, A.cast(s_stream.pop(), "float18x64"), (0,))
                                A.buffer_store(dec, sval[0] * alpha[0], (0,))
                                A.buffer_store(prod, kc[0] * v_new[vb], (0,))
                                A.buffer_store(row, dec[0] + prod[0], (0,))
                                A.stream_push(snew_stream, A.cast(row[0], f32))

                    with A.grid(dk, dv_blocks) as idx:
                        kk, vb = to_tuple(idx)
                        with A.ODR():
                            A.buffer_store(nxt_S, snew_stream.pop(), (kk, vb))

        def _step_readout(nxt_S, ti) -> None:
            with A.Datapath(core=None):
                with (
                    A.decl_stream(f32) as s_stream,
                    A.decl_stream(self.query.dtype) as q_stream,
                    A.decl_stream(f32) as o_stream,
                ):
                    with A.grid(dk, dv_blocks) as idx:
                        kk, vb = to_tuple(idx)
                        with A.IFD(num=0):
                            A.stream_push(s_stream, nxt_S[kk, vb])
                    with A.grid(dk) as kk:
                        with A.IFD(num=1):
                            A.stream_push(q_stream, q_l1[ti, to_tuple(kk)[0]])

                    with A.DPU():
                        o = A.array("float18x64", size=dv_blocks)
                        qc = A.var("float18x64")
                        sval = A.var("float18x64")
                        prod = A.var("float18x64")
                        # o = sum_k q_t[k] * S[k]   (peel kk == 0 to seed o)
                        A.buffer_store(qc, A.cast(q_stream.pop(), "float18x64"), (0,))
                        with A.unroll(0, dv_blocks) as vb:
                            A.buffer_store(sval, A.cast(s_stream.pop(), "float18x64"), (0,))
                            A.buffer_store(o, qc[0] * sval[0], (vb,))
                        with A.serial(1, dk) as kk:
                            A.buffer_store(qc, A.cast(q_stream.pop(), "float18x64"), (0,))
                            with A.unroll(0, dv_blocks) as vb:
                                A.buffer_store(sval, A.cast(s_stream.pop(), "float18x64"), (0,))
                                A.buffer_store(prod, qc[0] * sval[0], (0,))
                                A.buffer_store(o, o[vb] + prod[0], (vb,))
                        with A.unroll(0, dv_blocks) as vb:
                            A.stream_push(o_stream, A.cast(o[vb], f32))

                    with A.grid(dv_blocks) as vb:
                        with A.ODR():
                            A.buffer_store(o_l1, o_stream.pop(), (ti, to_tuple(vb)[0]))

        with A.grid(b, h) as idx:
            bi, hi = to_tuple(idx)
            atomic.copy(self.query[bi, hi, :, :], output=q_l1)
            atomic.copy(self.key[bi, hi, :, :], output=k_l1)
            atomic.copy(self.value[bi, hi, :, :], output=v_l1)
            atomic.copy(self.alpha[bi, hi, :, :], output=alpha_l1)
            atomic.copy(self.beta[bi, hi, :, :], output=beta_l1)
            atomic.copy(self.state_in[bi, hi, :, :], output=s_a)

            # Sequential recurrence: ping-pong the state buffer per step.
            bufs = [s_a, s_b]
            for ti in range(t):
                cur_S, nxt_S = bufs[ti % 2], bufs[(ti + 1) % 2]
                _step_vold(cur_S, ti)
                _step_state(cur_S, nxt_S, ti)
                _step_readout(nxt_S, ti)

            atomic.copy(o_l1, output=self.output[bi, hi, :, :])
            atomic.copy(bufs[t % 2], output=self.state_out[bi, hi, :, :])

    # Emulation (float18-stepped, mirrors the DPU op order)
    def emulate(self) -> None:  # type: ignore[override]
        # Do not use _dims() here -- in emulate the operands are numpy arrays
        # with an explicit trailing PWORD lane dim, so value.shape is
        # (B, H, T, Dv_blocks, PWORD), unlike the tir.Buffer in implementation.
        b, h, t, dk = (int(s) for s in self.query.shape[:4])
        pword = int(self.query.shape[-1])
        dv_blocks = int(self.value.shape[-2])
        dv = dv_blocks * pword  # Dv = Dv_blocks * PWORD

        # Strip the broadcast/lane dims to logical numpy shapes.
        q = self.query[..., 0].astype(np.float32)  # (B, H, T, Dk)
        k = self.key[..., 0].astype(np.float32)  # (B, H, T, Dk)
        v = self.value.astype(np.float32).reshape(b, h, t, dv)  # (B, H, T, Dv)
        alpha = self.alpha[..., 0, 0].astype(np.float32)  # (B, H, T)
        beta = self.beta[..., 0, 0].astype(np.float32)  # (B, H, T)
        s0 = self.state_in.astype(np.float32).reshape(b, h, dk, dv)  # (B, H, Dk, Dv)

        out = np.zeros((b, h, t, dv), dtype=np.float32)
        state_final = np.zeros((b, h, dk, dv), dtype=np.float32)

        for bi in range(b):
            for hi in range(h):
                S = emu.rf18(s0[bi, hi])  # (Dk, Dv)
                for ti in range(t):
                    a = emu.rf18(alpha[bi, hi, ti])
                    be = emu.rf18(beta[bi, hi, ti])
                    kc = emu.rf18(k[bi, hi, ti])  # (Dk,)
                    qc = emu.rf18(q[bi, hi, ti])  # (Dk,)
                    vt = emu.rf18(v[bi, hi, ti])  # (Dv,)

                    # pass A: v_old = sum_k k[k] * (a * S[k])
                    v_old = np.zeros(dv, dtype=np.float32)
                    for kk in range(dk):
                        dec = emu.rf18(S[kk] * a)
                        prod = emu.rf18(kc[kk] * dec)
                        v_old = emu.rf18(v_old + prod)
                    v_new = emu.rf18(be * emu.rf18(vt - v_old))

                    # pass B: S[k] = a*S[k] + k[k]*v_new ; o += q[k] * S[k]
                    o = np.zeros(dv, dtype=np.float32)
                    S_new = np.empty_like(S)
                    for kk in range(dk):
                        dec = emu.rf18(S[kk] * a)
                        prod = emu.rf18(kc[kk] * v_new)
                        row = emu.rf18(dec + prod)
                        S_new[kk] = row
                        o = emu.rf18(o + emu.rf18(qc[kk] * row))
                    S = S_new
                    out[bi, hi, ti] = o
                state_final[bi, hi] = S

        np.copyto(self.output, out.reshape(self.output.shape))
        np.copyto(self.state_out, state_final.reshape(self.state_out.shape))


gated_deltanet = register_op(
    GatedDeltaNet,
    "llm.gated_deltanet",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)

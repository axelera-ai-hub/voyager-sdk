#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Chunked gated delta rule in fp32 on RVV (llm.deltanet_chunk_rvv).

The full linear-attention recurrence -- decays, the triangular solve, the readout
and the state carry -- in fp32 on the AI core's RVV unit, replacing the float18 DPU
scan. See deltanet_chunk_rvv.c for the chunk algebra.

Everything runs on RVV here, including the matmuls, which belong on the datapath.
Keeping them here first establishes the fp32 accuracy ceiling, so the cost of moving
each one out can be measured against it rather than guessed at.

Measured at seq=512 with 16 heads and chunk 16, out of 225M cycles: KS0 and QS0 ~42M
each, the two Grams 43.5M and 37.2M, the state update 51.1M and the readout 4.0M --
five matmul-shaped stages for 87% of the kernel, none of which can go faster here,
since a vector op costs ~40 cycles of issue overhead plus vl/128 of work and vl is
capped at Dv=128 by the head dimension. The non-matmul residual is 7.9M (3.5%).
"""

# mypy: disable-error-code="attr-defined, index, arg-type, call-overload, type-var, misc"

from axelera.compiler.alto import script as A
import axelera.compiler.alto.axol as axol
from axelera.compiler.alto.axol import intrinsic_registry
from axelera.compiler.alto.axol.intrinsic_registry import FunctionArgument, FunctionArgumentType
from axelera.compiler.alto.compiler.dtypes import MetisTypes
from axelera.compiler.alto.compiler.op_registry import Arg, BufferInfo, ScriptOp, register_op
from axelera.config import HardwareGeneration
import numpy as np
from tvm import tir

from ._rvv import kernel_source, to_scalar

_INTRINSIC_NAME = "rvv_deltanet_chunk"

intrinsic_registry.set(
    name=_INTRINSIC_NAME,
    arguments=[
        FunctionArgument(name="q", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="k", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="v", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="g", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="beta", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="kkm", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="qkm", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="state", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="out", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="scratch", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="chunk", type=FunctionArgumentType.UINT64),
    ],
    implementation=kernel_source("deltanet_chunk_rvv"),
)


def _scratch_floats(dk: int, dv: int, chunk: int) -> int:
    """Workspace the kernel carves: ks0, qs0, vnew, kkm, qkm, khat, gc, at, ah, ea.

    No transposed-K slab: the Gram matrices read K transposed through a strided load
    instead, since materialising it cost 40% of the kernel's non-GEMM time. No decay
    matrix either -- it is rank one in log space, so it rides in the Grams via khat.
    """
    return 3 * chunk * dv + 2 * chunk * chunk + chunk * dk + 4 * chunk


class DeltaNetChunkRvv(ScriptOp):
    """Per-head chunked gated delta rule with an fp32 recurrent state."""

    q = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="(bh, seq, dk/64) float32x64 L2-normed query, head-major.",
    )
    k = Arg(MetisTypes[tir.Buffer, "..."], description="(bh, seq, dk/64) float32x64 key.")
    v = Arg(MetisTypes[tir.Buffer, "..."], description="(bh, seq, dv/64) float32x64 value.")
    g = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="(bh, seq/64) float32x64 per-token LOG decay.",
    )
    beta = Arg(MetisTypes[tir.Buffer, "..."], description="(bh, seq/64) float32x64 beta.")
    kkm = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="(bh, seq, T/64) float32x64 RAW Gram K@K^T per chunk, from the IMC.",
    )
    qkm = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="(bh, seq, T/64) float32x64 RAW Gram Q@K^T per chunk, from the IMC.",
    )
    state = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="(bh, dk, dv/64) float32x64 recurrent state, updated IN PLACE.",
    )
    chunk = Arg(int, default=64, kw=True, description="Tokens per chunk.")
    output = Arg(
        MetisTypes[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="(bh, seq, dv/64) float32x64 readout, or None to auto-allocate.",
    )
    core = Arg(int | None, default=None, kw=True, description="Optional AI core to run on.")
    generation = Arg(
        HardwareGeneration | None,
        default=None,
        kw=True,
        description="Optional hardware generation.",
    )

    def validate(self) -> None:
        q_shape = tuple(int(s) for s in self.q.shape)
        if len(q_shape) != 3:
            raise ValueError(f"q must be (bh, seq, blocks), got {q_shape}")
        for name in ("k", "v"):
            shape = tuple(int(s) for s in getattr(self, name).shape)
            if shape[:2] != q_shape[:2]:
                raise ValueError(f"{name} shape {shape} must agree with q {q_shape} on (bh, seq)")
        for name in ("q", "k", "v", "g", "beta", "state"):
            dtype = getattr(self, name).dtype
            if dtype != "float32x64":
                raise ValueError(f"{name} dtype must be 'float32x64', got '{dtype}'")

        bh, seq, _ = q_shape
        dk, dv = self._dims()
        if dk != int(self.state.shape[1]) or dv != int(self.state.shape[2]) * 64:
            raise ValueError(
                f"state must be (bh, {dk}, {dv // 64}) for dk={dk} dv={dv}, "
                f"got {tuple(int(s) for s in self.state.shape)}"
            )
        for name in ("g", "beta"):
            shape = tuple(int(s) for s in getattr(self, name).shape)
            if shape != (bh, seq // 64):
                raise ValueError(f"{name} must be ({bh}, {seq // 64}), got {shape}")
        if seq % 64:
            raise ValueError(f"seq must be a multiple of 64, got {seq}")
        if int(self.chunk) <= 0:
            raise ValueError(f"chunk must be positive, got {self.chunk}")

    def _dims(self) -> tuple[int, int]:
        from axelera.compiler.alto.compiler.device import PWORD

        return int(self.q.shape[2]) * PWORD, int(self.v.shape[2]) * PWORD

    def infer_outputs(self) -> dict[str, BufferInfo]:
        return {
            "output": BufferInfo(
                shape=tuple(int(s) for s in self.v.shape),
                dtype=self.v.dtype,
                scope=self.v.scope(),
            )
        }

    def implementation(self) -> None:
        from axelera.compiler.alto.compiler.device import PWORD

        bh, seq, _ = (int(s) for s in self.q.shape)
        dk, dv = self._dims()
        chunk = int(self.chunk)

        q_s = A.unpack(self.q, (bh, seq, dk))
        k_s = A.unpack(self.k, (bh, seq, dk))
        v_s = A.unpack(self.v, (bh, seq, dv))
        g_s = A.unpack(self.g, (bh, seq))
        beta_s = A.unpack(self.beta, (bh, seq))
        gw = int(self.kkm.shape[2]) * PWORD
        kkm_s = A.unpack(self.kkm, (bh, seq, gw))
        qkm_s = A.unpack(self.qkm, (bh, seq, gw))
        state_s = A.unpack(self.state, (bh, dk, dv))
        out_s = A.unpack(self.output, (bh, seq, dv))

        l1_q = A.alloc_buffer((bh, seq, dk), "float32", scope=A.L1)
        l1_k = A.alloc_buffer((bh, seq, dk), "float32", scope=A.L1)
        l1_v = A.alloc_buffer((bh, seq, dv), "float32", scope=A.L1)
        l1_g = A.alloc_buffer((bh, seq), "float32", scope=A.L1)
        l1_beta = A.alloc_buffer((bh, seq), "float32", scope=A.L1)
        l1_kkm = A.alloc_buffer((bh, seq, gw), "float32", scope=A.L1)
        l1_qkm = A.alloc_buffer((bh, seq, gw), "float32", scope=A.L1)
        l1_state = A.alloc_buffer((bh, dk, dv), "float32", scope=A.L1)
        l1_out = A.alloc_buffer((bh, seq, dv), "float32", scope=A.L1)

        # Shaped (rows, PWORD) rather than flat: pool offsets are byte-tight so the
        # size must be a whole lane word anyway, and a single extent past 32767
        # overflows a 16-bit lane count inside CompactBufferAllocation. The kernel
        # treats the data as flat and derives its own offsets from dk/dv/chunk.
        rows = (_scratch_floats(dk, dv, chunk) + PWORD - 1) // PWORD
        l1_scratch = A.alloc_buffer((rows, PWORD), "float32", scope=A.L1)

        # Tiled over heads and then over the token axis. The head tile makes every
        # operand's slice contiguous (all are head-major), and the token tile keeps
        # L1 independent of seq -- staging a whole sequence per head would be
        # 3*seq*dk*4 bytes and would cap chunk_len the way the fp18 scan was capped.
        #
        # The state buffer is allocated outside the token loop and is NOT indexed by
        # it, so the tiling pass leaves it whole and it stays resident across tiles,
        # carrying the recurrence. Whatever token tile the pass picks is safe: the
        # chunk algebra is exact for any chunk length, so a tile smaller than
        # `chunk` simply recurs more often rather than changing the result.
        with A.tiles(bh) as head_tile:
            axol.atomic.copy(
                state_s.tile(head_tile, dim=0), output=l1_state.tile(head_tile, dim=0)
            )
            with A.tiles(seq) as tok_tile:

                def tiled(buf, axis=1):
                    return buf.tile(head_tile, dim=0).tile(tok_tile, dim=axis)

                for src, dst in ((q_s, l1_q), (k_s, l1_k), (v_s, l1_v)):
                    axol.atomic.copy(tiled(src), output=tiled(dst))
                for src, dst in ((g_s, l1_g), (beta_s, l1_beta)):
                    axol.atomic.copy(tiled(src), output=tiled(dst))
                for src, dst in ((kkm_s, l1_kkm), (qkm_s, l1_qkm)):
                    axol.atomic.copy(tiled(src), output=tiled(dst))
                with A.Invoke(core=self.core):
                    A.extern(
                        _INTRINSIC_NAME,
                        tiled(l1_q),
                        tiled(l1_k),
                        tiled(l1_v),
                        tiled(l1_g),
                        tiled(l1_beta),
                        tiled(l1_kkm),
                        tiled(l1_qkm),
                        l1_state.tile(head_tile, dim=0),
                        tiled(l1_out),
                        l1_scratch,
                        chunk,
                    )
                axol.atomic.copy(tiled(l1_out), output=tiled(out_s))
            axol.atomic.copy(
                l1_state.tile(head_tile, dim=0), output=state_s.tile(head_tile, dim=0)
            )

    def emulate(self) -> None:  # type: ignore[override]
        q = to_scalar(self.q).astype(np.float32)
        k = to_scalar(self.k).astype(np.float32)
        v = to_scalar(self.v).astype(np.float32)
        g = to_scalar(self.g).astype(np.float32)
        beta = to_scalar(self.beta).astype(np.float32)
        kkm_all = to_scalar(self.kkm).astype(np.float32)
        qkm_all = to_scalar(self.qkm).astype(np.float32)
        state = to_scalar(self.state)
        out = to_scalar(self.output)
        bh, seq, dv = v.shape
        chunk = int(self.chunk)

        for h in range(bh):
            s = state[h].astype(np.float32)
            c0 = 0
            while c0 < seq:
                # Chunks never straddle a Gram window: the Grams are precomputed on fixed
                # chunk-aligned windows, so t_n is capped at the room left in this one.
                in_win = c0 % chunk
                t_n = min(chunk - in_win, seq - c0)
                gc = np.cumsum(g[h, c0 : c0 + t_n], dtype=np.float32)
                while t_n > 1 and gc[t_n - 1] < -80.0:
                    t_n >>= 1
                sl = slice(c0, c0 + t_n)
                qc, kc, vc = q[h, sl], k[h, sl], v[h, sl]
                be = beta[h, sl]

                # The kernel's factorisation, not the textbook form: the decay ratio
                # A_t/A_i is split about the chunk midpoint, where e^c = sqrt(A_T), so
                # the i half folds into K and the t half is one scalar per row.
                a_t = np.exp(gc[:t_n])
                ec = np.sqrt(a_t[t_n - 1])
                ah = a_t / ec
                ea = ec / a_t
                khat = ea[:, None] * kc
                # The Grams arrive RAW from the IMC (no decay in them), so both halves of
                # each ratio are applied here: ah on the rows, ea on the columns.
                kk = kkm_all[h, sl][:, in_win : in_win + t_n]
                qk = qkm_all[h, sl][:, in_win : in_win + t_n]

                ks0, qs0 = kc @ s, qc @ s
                m = np.tril((ah * be)[:, None] * (ea[None, :] * kk), -1)
                rhs = be[:, None] * (vc - a_t[:, None] * ks0)

                vnew = np.zeros((t_n, dv), np.float32)
                for t in range(t_n):
                    vnew[t] = rhs[t] - m[t, :t] @ vnew[:t]

                out[h, sl] = a_t[:, None] * qs0 + np.tril(ah[:, None] * (ea[None, :] * qk)) @ vnew
                s = a_t[t_n - 1] * s + ec * (khat.T @ vnew)
                c0 += t_n
            state[h] = s


deltanet_chunk_rvv = register_op(
    DeltaNetChunkRvv,
    "llm.deltanet_chunk_rvv",
    generation=HardwareGeneration.EUROPA,
    atomic=False,
)

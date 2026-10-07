#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Chunkwise (WY-form) gated delta rule -- the matmul formulation
(llm.gated_deltanet_chunk).

Mathematically equivalent to the sequential llm.gated_deltanet recurrence,
but it processes the T timesteps in chunks of chunk_size and expresses
each chunk as a sequence of matmuls (Yang et al. 2024, the "UT transform" /
WY representation used by the FLA library).  That matters for the hardware: a
chunk matmul has M = chunk_size rows, so it amortises an int8 IMC weight
load over the whole chunk -- the regime where the in-memory-compute array pays
off -- whereas the sequential decode recurrence is M = 1 and IMC-starved.
This op is the prefill counterpart to the decode-oriented
gated_deltanet / gated_deltanet_quant.

Algorithm (per head, g_t = log(alpha_t) the per-step log-decay).  Within a
chunk of length C the cumulative decay g is prefix-summed; the strictly
lower-triangular system (I - tril(beta KKᵀ, -1)) is inverted by forward
substitution into a C×C matrix T; then T transforms v_beta and
the decayed k_beta so the chunk's contribution is a handful of dense
matmuls.  The carried (Dk, Dv) state S is propagated between chunks
(S <- decay·S + Kᵀ V_new) and read by each chunk (Q@S inter-chunk term).
Keeping S in float across chunks bounds quantization error to be
chunk-local in the int8 variant (the forward-substitution stays on the DPU; the
dense matmuls are what move to the int8 IMC).  See the validated numpy port in
_chunk_gated_delta_rule.

This op is the float reference (the int8 / IMC kernel is the follow-up,
validated against this emulate).  Operand layout matches gated_deltanet
(broadcast-packed query/key, Dv-major value/state, scalar
alpha/beta) so it is a drop-in for comparison and reuses the same test
harness; the IMC kernel may refine the on-device packing.
"""
from axelera.compiler.alto.axol.utils import to_int_tuple
from axelera.compiler.alto.compiler.config import HardwareGeneration
from axelera.compiler.alto.compiler.dtypes import MetisFloats
from axelera.compiler.alto.compiler.op_registry import Arg, BufferInfo, ScriptOp, register_op
import axelera.compiler.alto.script as A
import numpy as np
from tvm import tir

# mypy: disable-error-code="attr-defined, index, arg-type"
# flake8: noqa: C901


def _qmm(a: np.ndarray, b: np.ndarray, quant: bool) -> np.ndarray:
    """Matmul a @ b; if quant, model an int8 IMC matmul.

    Symmetric per-row scale on a (...M,K) and per-column scale on b
    (...K,N) -- the most accurate scheme the IMC's per-channel dequant
    supports (A streamed per-M-row, B the loaded stationary matrix dequantized
    per-N).  int32 accumulation is exact; the dequant is the outer product of the
    two scales.  This is the only operation that becomes int8 on device; the
    surrounding gates, decay, and forward-substitution stay float.
    """
    if not quant:
        return a @ b
    sa = np.maximum(np.abs(a).max(-1, keepdims=True), 1e-12) / 127.0  # (...M,1)
    sb = np.maximum(np.abs(b).max(-2, keepdims=True), 1e-12) / 127.0  # (...1,N)
    aq = np.clip(np.round(a / sa), -127, 127)
    bq = np.clip(np.round(b / sb), -127, 127)
    return (aq @ bq) * sa * sb


def _chunk_gated_delta_rule(
    q: np.ndarray,
    k: np.ndarray,
    v: np.ndarray,
    alpha: np.ndarray,
    beta: np.ndarray,
    s0: np.ndarray,
    chunk_size: int,
    quant: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Chunkwise WY gated delta rule, RAW (no q-scaling / no L2-norm).

    Matches the gated_deltanet recurrence convention: alpha is the decay
    factor (exp(g)), q/k are used as given.  Inputs are logical float arrays:
    q/k (B, H, T, Dk), v (B, H, T, Dv), alpha/beta
    (B, H, T), s0 (B, H, Dk, Dv).  Returns out (B, H, T, Dv)
    and the final state (B, H, Dk, Dv).  Adapted from HF's
    torch_chunk_gated_delta_rule (numpy, float64).

    With quant=True every dense matmul is fake-quantized to int8 (see
    _qmm) while the carried state, gates, and forward-substitution stay
    float, modelling the int8 IMC kernel: error is chunk-local (it does not
    compound through the float state).
    """
    b, h, t, dk = (int(s) for s in q.shape)
    dv = int(v.shape[-1])
    c = int(chunk_size)
    g = np.log(np.clip(alpha, 1e-30, None)).astype(np.float64)  # [B,H,T] log-decay

    pad = int((c - t % c) % c)

    def _pad_t(x: np.ndarray) -> np.ndarray:
        widths = [(0, 0)] * x.ndim
        widths[2] = (0, pad)
        return np.pad(x, widths)

    q, k, v = (
        _pad_t(q.astype(np.float64)),
        _pad_t(k.astype(np.float64)),
        _pad_t(v.astype(np.float64)),
    )
    beta = _pad_t(beta.astype(np.float64))
    g = _pad_t(g)
    tp = t + pad
    nc = tp // c

    v_beta = v * beta[..., None]
    k_beta = k * beta[..., None]

    def _chunks(x: np.ndarray) -> np.ndarray:
        return x.reshape(b, h, nc, c, x.shape[-1])

    q, k, v, k_beta, v_beta = (_chunks(x) for x in (q, k, v, k_beta, v_beta))
    g = g.reshape(b, h, nc, c)
    g = np.cumsum(g, axis=-1)

    tri_incl = np.triu(np.ones((c, c), dtype=bool), 0)  # upper incl. diagonal
    # decay_mask[i, j] = exp(g_i - g_j) for i >= j, else 0
    decay_mask = np.tril(np.exp(g[..., :, None] - g[..., None, :]))

    attn = -np.where(tri_incl, 0.0, _qmm(k_beta, np.swapaxes(k, -1, -2), quant) * decay_mask)
    for i in range(1, c):  # forward substitution -> (I - A)^{-1} lower-tri block
        row = attn[..., i, :i].copy()
        sub = attn[..., :i, :i].copy()
        attn[..., i, :i] = row + (row[..., None] * sub).sum(-2)
    attn = attn + np.eye(c)

    v = _qmm(attn, v_beta, quant)
    k_cumdecay = _qmm(attn, k_beta * np.exp(g)[..., None], quant)

    state = s0.astype(np.float64).copy()
    out = np.zeros((b, h, nc, c, dv), dtype=np.float64)
    for i in range(nc):
        q_i, k_i, v_i = q[:, :, i], k[:, :, i], v[:, :, i]
        attn_intra = _qmm(q_i, np.swapaxes(k_i, -1, -2), quant) * decay_mask[:, :, i]
        v_prime = _qmm(k_cumdecay[:, :, i], state, quant)
        v_new = v_i - v_prime
        attn_inter = _qmm(q_i * np.exp(g[:, :, i, :, None]), state, quant)
        out[:, :, i] = attn_inter + _qmm(attn_intra, v_new, quant)
        decay_last = np.exp(g[:, :, i, -1, None, None])
        k_decayed = k_i * np.exp(g[:, :, i, -1, None] - g[:, :, i])[..., None]
        state = state * decay_last + _qmm(np.swapaxes(k_decayed, -1, -2), v_new, quant)

    out = out.reshape(b, h, -1, dv)[:, :, :t]
    return out, state


class GatedDeltaNetChunk(ScriptOp):
    """Chunkwise WY gated delta rule (float reference). See module docstring."""

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
    chunk_size = Arg(
        int,
        default=64,
        kw=True,
        description="Chunk length C; T is padded up to a multiple of C internally.",
    )
    generation = Arg(
        HardwareGeneration | None,
        default=None,
        kw=True,
        description="Optional hardware generation.",
    )

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
        if self.chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive; got {self.chunk_size}.")

    def infer_outputs(self) -> dict[str, BufferInfo]:
        b, h, t, dk, dv_blocks = self._dims()
        return {
            "output": BufferInfo(shape=(b, h, t, dv_blocks), dtype="float32x64", scope=A.DDR),
            "state_out": BufferInfo(shape=(b, h, dk, dv_blocks), dtype="float32x64", scope=A.DDR),
        }

    def implementation(self) -> None:
        # Per-chunk int8 IMC kernel: quantize the dense matmul operands, run on
        # the IMC (M = chunk_size amortises the load), forward-substitution on
        # the DPU, carried state in float.
        raise NotImplementedError(
            "llm.gated_deltanet_chunk: device kernel not implemented yet; the "
            "chunkwise formulation is currently validated at the emulation level "
            "(interpret) -- see the module docstring."
        )

    # Emulation: the float (float64) chunkwise reference.
    def emulate(self) -> None:  # type: ignore[override]
        # Operands carry an explicit trailing PWORD lane dim here, as in
        # gated_deltanet.emulate.
        b, h, t, dk = (int(s) for s in self.query.shape[:4])
        pword = int(self.query.shape[-1])
        dv_blocks = int(self.value.shape[-2])
        dv = dv_blocks * pword

        q = self.query[..., 0].astype(np.float64)  # (B, H, T, Dk)
        k = self.key[..., 0].astype(np.float64)  # (B, H, T, Dk)
        v = self.value.astype(np.float64).reshape(b, h, t, dv)  # (B, H, T, Dv)
        alpha = self.alpha[..., 0, 0].astype(np.float64)  # (B, H, T)
        beta = self.beta[..., 0, 0].astype(np.float64)  # (B, H, T)
        s0 = self.state_in.astype(np.float64).reshape(b, h, dk, dv)  # (B, H, Dk, Dv)

        out, state_final = _chunk_gated_delta_rule(q, k, v, alpha, beta, s0, self.chunk_size)

        np.copyto(self.output, out.reshape(self.output.shape))
        np.copyto(self.state_out, state_final.reshape(self.state_out.shape))


gated_deltanet_chunk = register_op(
    GatedDeltaNetChunk,
    "llm.gated_deltanet_chunk",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)

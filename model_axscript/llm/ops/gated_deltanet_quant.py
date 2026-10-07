#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Quantized (int8) recurrent gated delta rule -- decode reformulation
(llm.gated_deltanet_quant).

A drop-in numerical sibling of llm.gated_deltanet (same operand layout and
result semantics) that models the int8 decode path: the recurrent state
S is held in int8 with a per-Dv-column scale, and the two per-step
matrix-vector products are int8 matmuls against that state -- the shape the
in-memory-compute (IMC) array accelerates.

Why this is the optimal decode structure
-----------------------------------------
The float18 gated_deltanet reads the (Dk, Dv) state three times per step
(v_old, state update, readout), which dominates a decode that is
state-streaming bound.  Expanding the readout removes one of those reads and,
crucially, lets a single state read feed both matvecs::

    o_t = q_t @ S_new
        = q_t @ (alpha * S_old + outer(k_t, v_new))
        = alpha * (q_t @ S_old) + (q_t . k_t) * v_new

so v_old = alpha * (k_t @ S_old) and q_t @ S_old read the same
S_old.  A kernel therefore loads S_old into the IMC once, streams
k_t and q_t to get both matvecs (int8 -> int32), accumulates the scalar
q_t . k_t, and finishes o_t / v_new with cheap rank-1 corrections in
float18.  The state write S_new = alpha * S_old + outer(k_t, v_new) is the
only full pass over S and is requantized to int8 on the way out.  Net result:
~4x less state traffic (int8 vs float32) and one IMC load/step.

Quantization scheme (what emulate models)
---------------------------------------------
* q/k are L2-normalised upstream, so a single symmetric per-(B,H,T)
  scale quantizes them well.  A matvec reduces over Dk, so q/k must use one
  scale across that axis -- they do.
* S uses a per-Dv-column symmetric scale (varies along the output Dv
  axis, constant along the reduced Dk axis), so sum_k k_q[k] * S_q[k, v]
  is a valid int32 reduction dequantized by s_k * s_col[v].  The scale is
  recomputed (running per-column max) every time S is rewritten.
* v uses a per-(B,H,T) scale.  The decay/write gates alpha/beta are
  one scalar per (head, step) and stay float (never quantized).
* The matmuls are exact int32; every dequant / rank-1 correction is float18
  (emu.rf18), matching the DPU.

The carried state round-trips through int8 every step, so int8 error can
compound over the sequence -- tests/test_gated_deltanet_quant.py sweeps the
step count to characterise that drift.  This op keeps the public I/O float32
(state in/out dequantized) so it is directly comparable to gated_deltanet;
the device kernel (IMC, persistent int8 state + scales across calls) is a
follow-up validated against this emulation.
"""
from axelera.compiler.alto.axol.utils import to_int_tuple
from axelera.compiler.alto.compiler.config import HardwareGeneration
from axelera.compiler.alto.compiler.dtypes import MetisFloats
from axelera.compiler.alto.compiler.interpreter import emulation as emu
from axelera.compiler.alto.compiler.op_registry import Arg, BufferInfo, ScriptOp, register_op
import axelera.compiler.alto.script as A
import numpy as np
from tvm import tir

# mypy: disable-error-code="attr-defined, index, arg-type"
# flake8: noqa: C901


def _quant_sym(x: np.ndarray, axis: int | None) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric int8 fake-quant of x along axis (None = whole array).

    Returns (q, scale) where q is the integer-valued int32 representation
    (clamped to [-127, 127]) and scale is float32 with the reduced axis kept
    as size 1 for broadcasting.  scale is float18-rounded to mirror the
    kernel deriving it on the DPU.
    """
    amax = np.max(np.abs(x), axis=axis, keepdims=True) if axis is not None else np.max(np.abs(x))
    scale = emu.rf18(np.maximum(np.asarray(amax, np.float32), 1e-12) / np.float32(127.0))
    q = np.clip(np.round(x / scale), -127, 127).astype(np.int32)
    return q, np.asarray(scale, np.float32)


class GatedDeltaNetQuant(ScriptOp):
    """Int8 recurrent gated-delta-rule timesteps (decode reformulation).

    Same operands and result as GatedDeltaNet; emulate models the int8
    state / matvecs.  See the module docstring.
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
        description="(B, H, Dk, Dv_blocks) initial recurrent state S (float, dequantized).",
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

    def implementation(self) -> None:
        # IMC kernel: per step, load_imc(S_old) (int8, per-Dv-column scale),
        # stream k and q for the two int8 matvecs (k@S_old, q@S_old) plus the
        # q.k dot product, finish o_t / v_new with float18 rank-1 corrections,
        # then write+requantize S_new.
        raise NotImplementedError(
            "llm.gated_deltanet_quant: device kernel not implemented yet; "
            "the int8 decode reformulation is currently validated at the "
            "emulation level (interpret) -- see the module docstring."
        )

    # Emulation: int8 state + int8 matvecs (decode reformulation), fp18 rounded.
    def emulate(self) -> None:  # type: ignore[override]
        # Operands carry an explicit trailing PWORD lane dim here, as in
        # gated_deltanet.emulate, so value.shape is (B, H, T, Dv_blocks, PWORD).
        b, h, t, dk = (int(s) for s in self.query.shape[:4])
        pword = int(self.query.shape[-1])
        dv_blocks = int(self.value.shape[-2])
        dv = dv_blocks * pword

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
                # Carried state lives in int8 + per-Dv-column scale.
                s_q, s_col = _quant_sym(s0[bi, hi], axis=0)  # (Dk, Dv) int32, (1, Dv)
                s_col = s_col[0]  # (Dv,)
                for ti in range(t):
                    a = emu.rf18(alpha[bi, hi, ti])
                    be = emu.rf18(beta[bi, hi, ti])
                    kf = k[bi, hi, ti]  # fp (Dk,) -- exact, used for the rank-1 write
                    qf = q[bi, hi, ti]
                    vf = emu.rf18(v[bi, hi, ti])  # (Dv,)

                    # int8 quantize q / k (single scale across the reduced Dk axis)
                    q_q, sq = _quant_sym(qf, axis=None)
                    k_q, sk = _quant_sym(kf, axis=None)

                    # Two int8 matvecs against the SAME old state (int32 -> fp18),
                    # dequantized per Dv-column.
                    kS = emu.rf18((k_q @ s_q).astype(np.float32))  # (Dv,)
                    kS = emu.rf18(kS * emu.rf18(sk * s_col))
                    qS = emu.rf18((q_q @ s_q).astype(np.float32))
                    qS = emu.rf18(qS * emu.rf18(sq * s_col))
                    qk = emu.rf18(np.float32(q_q @ k_q) * emu.rf18(sq * sk))  # scalar

                    # Rank-1 corrections in fp18 (the reformulation).
                    v_old = emu.rf18(a * kS)
                    v_new = emu.rf18(be * emu.rf18(vf - v_old))
                    o = emu.rf18(emu.rf18(a * qS) + emu.rf18(qk * v_new))
                    out[bi, hi, ti] = o

                    # State update in fp18 from the dequantized old state, then
                    # requantize per column for the next step.
                    s_old_f = emu.rf18(s_q.astype(np.float32) * s_col)  # (Dk, Dv)
                    s_new_f = emu.rf18(emu.rf18(a * s_old_f) + np.outer(kf, v_new))
                    s_q, s_col = _quant_sym(s_new_f, axis=0)
                    s_col = s_col[0]
                state_final[bi, hi] = emu.rf18(s_q.astype(np.float32) * s_col)

        np.copyto(self.output, out.reshape(self.output.shape))
        np.copyto(self.state_out, state_final.reshape(self.state_out.shape))


gated_deltanet_quant = register_op(
    GatedDeltaNetQuant,
    "llm.gated_deltanet_quant",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)

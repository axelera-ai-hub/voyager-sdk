# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Self-contained Qwen3.5 (text tower) implementation in PyTorch.

A reference of the Qwen/Qwen3.5-2B language model built from torch /
torch.nn only (no transformers), so it can be quantized, exported via
torch.export, and lowered through the compiler pipeline. Qwen3.5 is natively a
vision-language model; this file implements only the qwen3_5_text decoder
tower (the model.language_model.* weights) -- vision and the MTP head are
ignored.

Qwen3.5 is a hybrid decoder: a 3:1 stack of gated DeltaNet linear-attention
blocks and gated full-attention blocks (one full-attention layer every
full_attention_interval layers; see layer_types). Per-block traits:

* Gated full attention -- q_proj emits num_heads * head_dim * 2
  (query concatenated with an output gate), QK-Norm (per-head RMSNorm on Q/K
  before RoPE), partial RoPE (only head_dim * partial_rotary_factor of
  each head is rotated), grouped-query attention, and a sigmoid output gate
  applied before o_proj.
* Gated DeltaNet (Yang et al. 2024) -- a matrix-valued recurrent state updated
  by the gated delta rule, fed by a short depthwise causal convolution; see
  GatedDeltaNet and ops/gated_deltanet.py for the recurrence.

What gets quantized: the full-attention q/k/v/o projections and the SwiGLU MLPs
are W8A8-quantizable (QuantLinear); the DeltaNet blocks are left full
precision for now (plain nn.Linear / nn.Conv1d), so quantize_model
skips them. All RMSNorms are the zero-centred (1 + weight) form; the +1
is folded into the weight at load so the shared RMSNorm (weight * x) and
its export path are reused unchanged.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...checkpoint import load_hf_safetensors
from ...modules import QuantSDPA, RMSNorm, RotaryEmbedding, SwiGLUMLP, repeat_kv, rotate_half
from ...quant import ActivationObserver, QuantConv1d, QuantLinear, RangeObserver
from ...quant import quantize_model as _quantize_model
from ...quant import record_lut_ranges
from ..architecture import LLMArchitecture, register_architecture


def _quant_linear_kwargs(config: "Qwen3_5Config") -> dict:
    """The W8A8 keyword arguments shared by every quantizable linear."""
    return {
        "act_bits": config.activation_quant_bits,
        "weight_bits": config.weight_quant_bits,
        "act_symmetric": config.activation_quant_symmetric,
    }


def smoothquant_groups(module: nn.Module) -> list[list["QuantLinear"]]:
    """Groups of QuantLinear that share an input tensor (Qwen3.5 layout).

    Linears fed by the same activation (q/k/v from input_layernorm; gate/up from
    post_attention_layernorm) must share one smoothing factor so they keep
    identical activation quant params. o_proj / down_proj have their own input,
    so they form singleton groups.

    When quantize_linear_attention is set, the DeltaNet's in_proj_qkv and
    the z/b/a gate projections all read the same (input-layernorm) hidden state, so
    they form ONE SmoothQuant group -- they then share identical activation quant
    params, which is required because the Metis kernel quantizes that hidden state
    once and reuses the int8 result for all four. out_proj reads the gated-norm
    output, so it is a singleton group. The recurrence itself stays full precision
    (see GatedDeltaNet). SmoothQuant is self-contained per group (the x / s fold
    happens inside each member's forward), so this does not perturb the rest.
    """
    groups: list[list[QuantLinear]] = []
    for layer in module.modules():
        if isinstance(layer, Qwen3_5Attention):
            groups.append([layer.q_proj, layer.k_proj, layer.v_proj])
            groups.append([layer.o_proj])
        elif isinstance(layer, SwiGLUMLP):
            groups.append([layer.gate_proj, layer.up_proj])
            groups.append([layer.down_proj])
        elif isinstance(layer, GatedDeltaNet) and isinstance(layer.in_proj_qkv, QuantLinear):
            groups.append([layer.in_proj_qkv, layer.in_proj_z, layer.in_proj_b, layer.in_proj_a])
            groups.append([layer.out_proj])
    # The lm_head is fed by the final norm alone, so it smooths as a singleton. Without a
    # group it would keep smooth = 1 and its qparams would be per-tensor, which breaches the
    # ppl gate.
    head = getattr(module, "lm_head", None)
    if isinstance(head, QuantLinear):
        groups.append([head])
    return groups


def set_decode_recording(module: nn.Module, recording: bool) -> None:
    """Toggle DECODE-scale capture (peak |L2norm q/k| and |state|) on every
    GatedDeltaNet under module. These are the Metis decode kernel's fixed int8
    requant scales, captured during a calibration forward (see Qwen3_5Architecture)."""
    for layer in module.modules():
        if isinstance(layer, GatedDeltaNet):
            layer.record_decode = recording


def set_int8_recurrence(module: nn.Module, enabled: bool) -> None:
    """Toggle the device int8-recurrence model on every GatedDeltaNet under module.

    Diagnostic: isolates the int8 gated-delta recurrence's accuracy cost from the
    LUTs / float18. Requires set_decode_recording to have populated the decode_* scales."""
    for layer in module.modules():
        if isinstance(layer, GatedDeltaNet):
            layer.use_int8_recurrence = enabled


def quantize_model(
    module: nn.Module,
    calibration,
    *,
    smooth_alpha: float | None = 0.8,
    qk_alpha: float | None = 0.5,
) -> None:
    """Calibrate + enable W8A8 (+ SmoothQuant / QK-smoothing) for a Qwen3.5 model.

    Wraps quant.quantize_model with the Qwen3.5 grouping. Must run on a
    freshly-loaded model (the smoothing folds are one-shot). The DeltaNet blocks
    are left untouched (they hold no QuantLinear / QuantSDPA).
    """
    _quantize_model(
        module,
        calibration,
        groups=smoothquant_groups(module),
        smooth_alpha=smooth_alpha,
        qk_alpha=qk_alpha,
    )


@dataclass
class Qwen3_5Config:
    """Hyperparameters for a Qwen3.5 text model.

    Defaults correspond to the Qwen/Qwen3.5-2B text_config.
    """

    vocab_size: int = 248320
    hidden_size: int = 2048
    intermediate_size: int = 6144
    num_hidden_layers: int = 24
    # Full (softmax) attention.
    num_attention_heads: int = 8
    num_key_value_heads: int = 2
    head_dim: int = 256
    # Linear attention (gated DeltaNet).
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    # Hybrid stack: a full-attention layer every full_attention_interval.
    full_attention_interval: int = 4
    # RoPE / norm.
    max_position_embeddings: int = 262144
    rms_norm_eps: float = 1e-6
    rope_theta: float = 10_000_000.0
    partial_rotary_factor: float = 0.25
    tie_word_embeddings: bool = True
    # W8A8 defaults match the production Qwen kernels: per-tensor asymmetric
    # activations (per-channel part comes from SmoothQuant), per-channel symmetric weights.
    activation_quant_bits: int = 8
    weight_quant_bits: int = 8
    activation_quant_symmetric: bool = False
    # When enabled the lm_head is a QuantLinear, so it exports an int8 weight + qparams and
    # runs on device. Smoothed per-channel (per-tensor breaches the ppl gate) and untied from
    # the embedding, since smoothing mutates the weight in place.
    quantize_lm_head: bool = False
    # When enabled, full-attention Q/K/V and softmax weights are fake-quantized.
    quantize_attention: bool = False
    sdpa_quant_bits: int = 8
    # When enabled, the DeltaNet's big static-weight GEMMs (in_proj_qkv, out_proj)
    # become W8A8-quantizable (the IMC-relevant part). The tiny gate projections,
    # the conv, the gated norm, and the recurrent delta rule itself stay full
    # precision: they are either gate-sensitive, not GEMMs, or accumulate state
    # whose error compounds over the sequence, and none map to the weight-
    # stationary IMC anyway.
    quantize_linear_attention: bool = False
    # When enabled, the DeltaNet's depthwise short conv becomes a W8A8 QuantConv1d, so
    # it can run on the int8 DWPU instead of the float18 DPU -- measured at seq=512 with
    # C=6144, K=4, the DPU path costs 106.7M cycles against 2.3M, and the int8 result is
    # the closer of the two to an exact conv. The activation quant is per-channel and
    # SYMMETRIC (QuantConv1d pins it): per-channel because a depthwise conv has no
    # SmoothQuant to migrate outliers, so a per-tensor scale zeros the quiet channels
    # (measured 1.5e-02 against 5.1e-03), and symmetric because the kernel's conv-window
    # cache is zero-initialised, so quantized-zero must be integer 0 for the causal left
    # padding to be right -- with a zero point the leading K-1 outputs are ~47% wrong.
    quantize_conv: bool = False

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def rotary_dim(self) -> int:
        """Number of head-dim channels that RoPE rotates (partial rotary)."""
        return int(self.head_dim * self.partial_rotary_factor)

    def layer_type(self, layer_idx: int) -> str:
        """Return 'full_attention' every full_attention_interval-th layer, else 'linear_attention'.

        Matches HF's layer_types: indices interval-1, 2*interval-1, ... are full.
        """
        is_full = (layer_idx + 1) % self.full_attention_interval == 0
        return "full_attention" if is_full else "linear_attention"

    @classmethod
    def from_hf_dict(cls, cfg: dict) -> "Qwen3_5Config":
        """Build a Qwen3_5Config from a HuggingFace config.json dict.

        Accepts either the top-level VLM config (with a text_config) or a bare text config.
        """
        text = cfg.get("text_config", cfg)
        rope = text.get("rope_parameters", text)
        return cls(
            vocab_size=text["vocab_size"],
            hidden_size=text["hidden_size"],
            intermediate_size=text["intermediate_size"],
            num_hidden_layers=text["num_hidden_layers"],
            num_attention_heads=text["num_attention_heads"],
            num_key_value_heads=text["num_key_value_heads"],
            head_dim=text["head_dim"],
            linear_num_key_heads=text["linear_num_key_heads"],
            linear_num_value_heads=text["linear_num_value_heads"],
            linear_key_head_dim=text["linear_key_head_dim"],
            linear_value_head_dim=text["linear_value_head_dim"],
            linear_conv_kernel_dim=text["linear_conv_kernel_dim"],
            full_attention_interval=text["full_attention_interval"],
            max_position_embeddings=text["max_position_embeddings"],
            rms_norm_eps=text["rms_norm_eps"],
            rope_theta=rope["rope_theta"],
            partial_rotary_factor=rope.get("partial_rotary_factor", 1.0),
            # TOP level first: the VLM config carries this beside text_config, not inside
            # it. Qwen3.5-9B is untied and says so ONLY at the top level, so reading
            # text_config alone silently defaulted it to True and tied the head to the
            # embedding -- the model then ran with its embedding matrix as the output
            # projection and produced fluent-looking nonsense.
            tie_word_embeddings=cfg.get(
                "tie_word_embeddings", text.get("tie_word_embeddings", True)
            ),
        )


def apply_partial_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply RoPE to the first cos.shape[-1] channels of q/k, passing the rest through.

    q/k are [batch, num_heads, seq_len, head_dim]; cos/sin are
    [batch, seq_len, rotary_dim] with rotary_dim <= head_dim. The unrotated tail
    (the NoPE part of partial rotary) is concatenated back unchanged.
    """
    # See apply_rotary_pos_emb: rotate in the activation dtype, not the float32 the tables
    # are built in. A no-op at float32.
    cos = cos.unsqueeze(1).to(q.dtype)  # [batch, 1, seq_len, rotary_dim]
    sin = sin.unsqueeze(1).to(q.dtype)
    rotary_dim = cos.shape[-1]
    q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
    k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
    q_embed = torch.cat([(q_rot * cos) + (rotate_half(q_rot) * sin), q_pass], dim=-1)
    k_embed = torch.cat([(k_rot * cos) + (rotate_half(k_rot) * sin), k_pass], dim=-1)
    return q_embed, k_embed


class Qwen3_5Attention(nn.Module):
    """Gated grouped-query attention with QK-Norm and partial RoPE."""

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_key_value_groups

        # qkv projections are W8A8-quantizable. q_proj emits query AND an output
        # gate (hence the *2); o_proj is also quantizable.
        qkw = _quant_linear_kwargs(config)
        self.q_proj = QuantLinear(config.hidden_size, self.num_heads * self.head_dim * 2, **qkw)
        self.k_proj = QuantLinear(config.hidden_size, self.num_kv_heads * self.head_dim, **qkw)
        self.v_proj = QuantLinear(config.hidden_size, self.num_kv_heads * self.head_dim, **qkw)
        self.o_proj = QuantLinear(self.num_heads * self.head_dim, config.hidden_size, **qkw)

        # QK-Norm: per-head RMSNorm over the full head_dim, applied before RoPE.
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        self.sdpa = QuantSDPA(
            self.head_dim, quantize=config.quantize_attention, bits=config.sdpa_quant_bits
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        batch, seq_len, _ = hidden_states.shape
        head_shape = (batch, seq_len, -1, self.head_dim)

        # q_proj output is [.., num_heads, 2*head_dim] -> split into query and gate.
        q_gate = self.q_proj(hidden_states).view(batch, seq_len, -1, self.head_dim * 2)
        q, gate = torch.chunk(q_gate, 2, dim=-1)
        gate = gate.reshape(batch, seq_len, -1)  # [batch, seq_len, num_heads*head_dim]

        q = self.q_norm(q).transpose(1, 2)  # [batch, num_heads, seq_len, head_dim]
        k = self.k_norm(self.k_proj(hidden_states).view(head_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(head_shape).transpose(1, 2)

        q, k = apply_partial_rotary_pos_emb(q, k, cos, sin)

        # Expand KV heads to match the number of query heads (GQA).
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        attn_output = self.sdpa(q, k, v, attention_mask)  # [batch, heads, seq, head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, -1)
        # Output gate (sigmoid) before the output projection.
        attn_output = attn_output * torch.sigmoid(gate)
        return self.o_proj(attn_output)


def _l2norm(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """L2-normalise over the last dim (matches the FLA kernel's qk normalisation)."""
    return x * torch.rsqrt((x * x).sum(dim=-1, keepdim=True) + eps)


def recurrent_gated_delta_rule(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor | None = None,
    return_state_absmax: bool = False,
):
    """Sequential gated delta-rule reference (full-precision, prefill / no cache).

    query/key/value: [batch, seq, heads, head_dim]; g/beta: [batch, seq, heads].
    Q/K are L2-normalised and Q is scaled by head_dim**-0.5 (the FLA kernel
    convention, use_qk_l2norm_in_kernel=True). Returns the per-step readout
    [batch, seq, heads, head_v_dim]. See ops/gated_deltanet.py for the recurrence.

    When return_state_absmax is set, also returns the peak |state| over the
    scan -- the Metis decode kernel quantises the carried state with a single fixed
    scale s_S = peak|S|/127, so this calibrates it.
    """
    # The scan runs in fp32 whatever the model's dtype: it accumulates the state over the
    # whole sequence, so a narrower type drifts. The RESULT goes back in the caller's dtype
    # so the module is dtype-transparent -- otherwise a bf16 model dies in the next Linear
    # ("expected m1 and m2 to have the same dtype"), which is what blocked bf16 calibration.
    io_dtype = value.dtype
    query = _l2norm(query)
    key = _l2norm(key)
    # -> [batch, heads, seq, dim]
    query, key, value = (x.transpose(1, 2).float() for x in (query, key, value))
    g, beta = (x.transpose(1, 2).float() for x in (g, beta))  # [batch, heads, seq]

    batch, num_heads, seq_len, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    query = query * (k_head_dim**-0.5)

    out = torch.zeros(
        batch, num_heads, seq_len, v_head_dim, dtype=value.dtype, device=value.device
    )
    state = (
        torch.zeros(
            batch, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device
        )
        if initial_state is None
        # .clone(): the loop below updates the state IN PLACE (mul_/baddbmm_), and
        # .float() on a float32 tensor returns the SAME object -- so without a copy a
        # caller's initial_state is destroyed by the call. .contiguous() because the state
        # is later viewed as (batch*heads, k, v), which a non-contiguous input cannot do.
        # (recurrent_gated_delta_rule_int8 rebinds `state` instead, so it needs no copy.)
        else initial_state.float().clone().contiguous()
    )
    state_absmax = torch.zeros((), dtype=value.dtype, device=value.device)
    # In place, and matvecs rather than broadcast-multiply-then-reduce. The naive form
    # allocates FIVE [batch, heads, k_head_dim, v_head_dim] temporaries per token -- 2 MB
    # each at 9B -- and this loop runs once per token per layer for every calibration
    # window, which is the bulk of a CPU calibration (~16 min per pass for a 9B). Same
    # arithmetic: a rank-1 update and two matvecs, differing from the broadcast form only
    # by reduction order (~4e-7 relative).
    state_flat = state.view(batch * num_heads, k_head_dim, v_head_dim)
    for i in range(seq_len):
        q_t = query[:, :, i]  # [batch, heads, k_head_dim]
        k_t = key[:, :, i]
        v_t = value[:, :, i]  # [batch, heads, v_head_dim]
        g_t = g[:, :, i].exp().unsqueeze(-1).unsqueeze(-1)  # [batch, heads, 1, 1]
        beta_t = beta[:, :, i].unsqueeze(-1)  # [batch, heads, 1]

        state.mul_(g_t)
        kv_mem = torch.einsum("bhdv,bhd->bhv", state, k_t)  # [batch, heads, v_head_dim]
        delta = (v_t - kv_mem) * beta_t
        state_flat.baddbmm_(
            k_t.reshape(batch * num_heads, k_head_dim, 1),
            delta.reshape(batch * num_heads, 1, v_head_dim),
        )
        out[:, :, i] = torch.einsum("bhdv,bhd->bhv", state, q_t)
        if return_state_absmax:
            state_absmax = torch.maximum(state_absmax, state.abs().max())

    out = out.transpose(1, 2).contiguous().to(io_dtype)  # [batch, seq, heads, v_head_dim]
    return (out, state_absmax) if return_state_absmax else out


def recurrent_gated_delta_rule_int8(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    s_q: float,
    s_k: float,
    s_S: float,
    initial_state: torch.Tensor | None = None,
):
    """Torch model of the Metis DECODE int8-IMC gated delta rule (gated_deltanet_decode_imc).

    Same recurrence as recurrent_gated_delta_rule, but the kS/qS/qk readout runs in
    int8 exactly as the device does: the L2-normed q/k and the ENTRY state S are quantised
    to int8 with FIXED per-tensor scales (s = absmax/127, calibrated by
    set_decode_recording), the matmuls accumulate the int products, and the result is
    dequantised; the rank-1 state update stays fp. q/k are NOT scaled by head_dim**-0.5
    (the device leaves them unscaled -- the gated RMSNorm tail washes the scale out -- and
    s_q is calibrated on the unscaled L2-norm). Isolates how much of the on-device
    accuracy gap is the int8 recurrence (vs the LUTs / float18) by swapping ONLY this in for
    the fp recurrence in an otherwise-identical fake-quant forward."""
    qmax = 127.0

    def qi8(x, s):
        return torch.clamp(torch.round(x / s), -qmax, qmax)

    query = _l2norm(query)
    key = _l2norm(key)
    io_dtype = value.dtype  # dtype-transparent, as the fp variant above
    query, key, value = (x.transpose(1, 2).float() for x in (query, key, value))
    g, beta = (x.transpose(1, 2).float() for x in (g, beta))
    batch, num_heads, seq_len, k_head_dim = key.shape
    v_head_dim = value.shape[-1]
    out = torch.zeros(
        batch, num_heads, seq_len, v_head_dim, dtype=value.dtype, device=value.device
    )
    state = (
        torch.zeros(
            batch, num_heads, k_head_dim, v_head_dim, dtype=value.dtype, device=value.device
        )
        if initial_state is None
        else initial_state.float()
    )
    for i in range(seq_len):
        ql = query[:, :, i]  # [b, h, Dk]
        kl = key[:, :, i]
        v_t = value[:, :, i]  # [b, h, Dv]
        alpha = g[:, :, i].exp().unsqueeze(-1)  # [b, h, 1]
        beta_t = beta[:, :, i].unsqueeze(-1)  # [b, h, 1]

        q8 = qi8(ql, s_q)
        k8 = qi8(kl, s_k)
        s8 = qi8(state, s_S)  # quantise the ENTRY state (pre-decay), like the device
        # int8 matmul readout against the entry state, then dequant.
        kS = torch.einsum("bhd,bhdv->bhv", k8, s8) * (s_k * s_S)
        qS = torch.einsum("bhd,bhdv->bhv", q8, s8) * (s_q * s_S)
        qk = (q8 * k8).sum(-1, keepdim=True) * (s_q * s_k)  # [b, h, 1]

        v_new = beta_t * (v_t - alpha * kS)
        out[:, :, i] = alpha * qS + qk * v_new
        # fp rank-1 state update: S_new = alpha*S + k (x) v_new (k is the fp L2-norm).
        state = alpha.unsqueeze(-1) * state + kl.unsqueeze(-1) * v_new.unsqueeze(-2)

    return out.transpose(1, 2).contiguous().to(io_dtype)


class GatedRMSNorm(nn.Module):
    """RMSNorm with a SiLU gate: weight * x_normed * silu(gate).

    The DeltaNet output norm. Unlike the decoder RMSNorms this is a plain
    (ones-init) scale, NOT the zero-centred (1 + weight) form.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        # Records the inverse-sqrt input range for the gated-norm LUT. The Metis
        # kernel norms the chunk readout with eps' = D * eps, so its rsqrt sees
        # D * (variance + eps) -- record that scaled range so the fitted LUT covers
        # exactly the device's input domain (eps must be D*eps).
        self.inv_sqrt_range = RangeObserver("inv_sqrt")

    def forward(self, x: torch.Tensor, gate: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        self.inv_sqrt_range(self.hidden_size * (variance + self.eps))
        x = x * torch.rsqrt(variance + self.eps)
        x = self.weight * x.to(input_dtype)
        x = x * F.silu(gate.to(torch.float32))
        return x.to(input_dtype)


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet linear-attention block.

    Projects the input to query/key/value (via a shared in_proj_qkv), a SiLU gate
    z, and the per-head delta-rule gates (b -> beta, a -> decay g); applies a
    short depthwise causal conv + SiLU to q/k/v, runs the recurrent gated delta
    rule, gate-norms the readout, and projects back to hidden_size.

    With config.quantize_linear_attention the two big static-weight GEMMs --
    in_proj_qkv and out_proj -- are W8A8 QuantLinear (so they can run on the int8
    IMC); everything else (the tiny gate projections, the conv, the gated norm, and
    the recurrence) stays full precision. The recurrence is activation x activation
    against a running state, so it neither maps to the weight-stationary IMC nor
    tolerates int8 well (per-step error compounds over the sequence) -- quantizing
    it is a separate investigation.
    """

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_k_heads = config.linear_num_key_heads
        self.num_v_heads = config.linear_num_value_heads
        self.head_k_dim = config.linear_key_head_dim
        self.head_v_dim = config.linear_value_head_dim
        self.key_dim = self.head_k_dim * self.num_k_heads
        self.value_dim = self.head_v_dim * self.num_v_heads
        self.conv_kernel_size = config.linear_conv_kernel_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim

        if config.quantize_conv:
            # QuantConv1d is causal itself (pads by K-1 and truncates to T), so the
            # forward's trailing [..., :seq_len] is a no-op on its output.
            self.conv1d: nn.Module = QuantConv1d(
                self.conv_dim,
                self.conv_kernel_size,
                act_bits=config.activation_quant_bits,
                weight_bits=config.weight_quant_bits,
            )
        else:
            self.conv1d = nn.Conv1d(
                in_channels=self.conv_dim,
                out_channels=self.conv_dim,
                bias=False,
                kernel_size=self.conv_kernel_size,
                groups=self.conv_dim,
                padding=self.conv_kernel_size - 1,
            )
        # in_proj_qkv and out_proj are the IMC-relevant GEMMs: optionally W8A8.
        # The Metis kernel quantizes the (input-layernorm) hidden state ONCE and
        # reuses that int8 activation for in_proj_qkv AND the z/b/a gate projections,
        # so under quantize_linear_attention all four are W8A8 QuantLinear sharing
        # one SmoothQuant group (identical activation quant params); see
        # smoothquant_groups. Gate quant adds ~1.3% to the layer output, far below
        # the int8 recurrence floor. With the flag off, the whole block stays full precision.
        qkv_out = self.key_dim * 2 + self.value_dim
        if config.quantize_linear_attention:
            qkw = _quant_linear_kwargs(config)
            self.in_proj_qkv: nn.Module = QuantLinear(self.hidden_size, qkv_out, **qkw)
            self.out_proj: nn.Module = QuantLinear(self.value_dim, self.hidden_size, **qkw)
            self.in_proj_z: nn.Module = QuantLinear(self.hidden_size, self.value_dim, **qkw)
            self.in_proj_b: nn.Module = QuantLinear(self.hidden_size, self.num_v_heads, **qkw)
            self.in_proj_a: nn.Module = QuantLinear(self.hidden_size, self.num_v_heads, **qkw)
        else:
            self.in_proj_qkv = nn.Linear(self.hidden_size, qkv_out, bias=False)
            self.out_proj = nn.Linear(self.value_dim, self.hidden_size, bias=False)
            self.in_proj_z = nn.Linear(self.hidden_size, self.value_dim, bias=False)
            self.in_proj_b = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)
            self.in_proj_a = nn.Linear(self.hidden_size, self.num_v_heads, bias=False)

        self.dt_bias = nn.Parameter(torch.ones(self.num_v_heads))
        self.A_log = nn.Parameter(torch.zeros(self.num_v_heads))
        self.norm = GatedRMSNorm(self.head_v_dim, eps=config.rms_norm_eps)
        # Records the inverse-sqrt input range for the q/k L2-normalisation LUT. The
        # kernel realises L2-norm as rms_norm_kahan(weight=ones/sqrt(D)), whose rsqrt
        # sees mean(x^2) + eps -- record that for both q and k (see _l2norm).
        self.qk_l2_inv_sqrt_range = RangeObserver("inv_sqrt")
        # Calibrated FIXED int8 requant scales for the Metis DECODE kernel: the peak
        # abs of L2-normed q / L2-normed k / the recurrent state S over calibration
        # (s = absmax/127). Captured only while record_decode is set (see
        # set_decode_recording). Non-persistent (recomputed per session, not in the HF state dict).
        self.record_decode = False
        for name in ("decode_q_absmax", "decode_k_absmax", "decode_S_absmax"):
            self.register_buffer(name, torch.zeros(()), persistent=False)
        # Diagnostic: run the gated delta rule with the device's int8 recurrence
        # (set_int8_recurrence) instead of fp, to isolate the int8-recurrence accuracy
        # cost. Requires the decode_* scales (set_decode_recording) to be populated.
        self.use_int8_recurrence = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # cos/sin/attention_mask are accepted for a uniform decoder-layer signature
        # but unused: linear attention has no RoPE and is implicitly causal.
        batch, seq_len, _ = hidden_states.shape

        mixed_qkv = self.in_proj_qkv(hidden_states).transpose(1, 2)  # [batch, conv_dim, seq]
        z = self.in_proj_z(hidden_states).reshape(batch, seq_len, -1, self.head_v_dim)
        b = self.in_proj_b(hidden_states)  # [batch, seq, num_v_heads]
        a = self.in_proj_a(hidden_states)

        # Depthwise causal conv (drop the right padding) + SiLU.
        mixed_qkv = F.silu(self.conv1d(mixed_qkv)[..., :seq_len]).transpose(1, 2)
        query, key, value = torch.split(
            mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1
        )
        query = query.reshape(batch, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch, seq_len, -1, self.head_v_dim)

        # Record the q/k L2-norm inverse-sqrt input range (mean(x^2) + eps over the
        # head dim) for the kernel's L2-norm LUT; the recurrence L2-normalises q/k
        # internally, but the kernel norms them up front, so capture it here.
        eps = self.norm.eps
        self.qk_l2_inv_sqrt_range(query.float().pow(2).mean(-1) + eps)
        self.qk_l2_inv_sqrt_range(key.float().pow(2).mean(-1) + eps)

        beta = b.sigmoid()
        g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias.float())

        rep = self.num_v_heads // self.num_k_heads
        if rep > 1:
            query = query.repeat_interleave(rep, dim=2)
            key = key.repeat_interleave(rep, dim=2)

        if self.record_decode:
            core, s_absmax = recurrent_gated_delta_rule(
                query, key, value, g=g, beta=beta, return_state_absmax=True
            )
            with torch.no_grad():
                self.decode_q_absmax = torch.maximum(
                    self.decode_q_absmax, _l2norm(query).abs().max()
                )
                self.decode_k_absmax = torch.maximum(
                    self.decode_k_absmax, _l2norm(key).abs().max()
                )
                self.decode_S_absmax = torch.maximum(self.decode_S_absmax, s_absmax)
        elif self.use_int8_recurrence:
            qm = 127.0
            core = recurrent_gated_delta_rule_int8(
                query,
                key,
                value,
                g=g,
                beta=beta,
                s_q=float(self.decode_q_absmax) / qm,
                s_k=float(self.decode_k_absmax) / qm,
                s_S=float(self.decode_S_absmax) / qm,
            )
        else:
            core = recurrent_gated_delta_rule(query, key, value, g=g, beta=beta)
        core = self.norm(core.reshape(-1, self.head_v_dim), z.reshape(-1, self.head_v_dim))
        core = core.reshape(batch, seq_len, -1)
        return self.out_proj(core)


class Qwen3_5DecoderLayer(nn.Module):
    """A single hybrid transformer decoder block (pre-norm).

    The token mixer is gated full attention on full-attention layers and a gated
    DeltaNet on linear-attention layers; the channel mixer is always a SwiGLU MLP.
    """

    def __init__(self, config: Qwen3_5Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_type = config.layer_type(layer_idx)
        if self.layer_type == "full_attention":
            self.self_attn: nn.Module = Qwen3_5Attention(config)
        else:
            self.linear_attn: nn.Module = GatedDeltaNet(config)
        self.mlp = SwiGLUMLP(
            config.hidden_size, config.intermediate_size, **_quant_linear_kwargs(config)
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        mixer = self.self_attn if self.layer_type == "full_attention" else self.linear_attn

        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = mixer(hidden_states, cos, sin, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class Qwen3_5Model(nn.Module):
    """The Qwen3.5 text trunk (embeddings + hybrid decoder layers + final norm)."""

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3_5DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # Partial RoPE: rotary_dim channels of each head are rotated.
        self.rotary_emb = RotaryEmbedding(
            config.rotary_dim, config.max_position_embeddings, config.rope_theta
        )

    def _causal_mask(self, seq_len: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask[None, None, :, :]  # [1, 1, seq_len, seq_len]

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = input_ids.shape[1]
        hidden_states = self.embed_tokens(input_ids)

        if position_ids is None:
            position_ids = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        cos, sin = self.rotary_emb(position_ids)
        attention_mask = self._causal_mask(seq_len, hidden_states.dtype, hidden_states.device)

        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, attention_mask)

        return self.norm(hidden_states)


class Qwen3_5ForCausalLM(nn.Module):
    """Qwen3.5 text model with a language-modeling head."""

    def __init__(self, config: Qwen3_5Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3_5Model(config)
        if config.quantize_lm_head:
            self.lm_head = QuantLinear(
                config.hidden_size, config.vocab_size, **_quant_linear_kwargs(config)
            )
        else:
            self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight
        self._untie_lm_head()

    def _untie_lm_head(self) -> None:
        """Give a quantized head its own weight, on every construction path that (re)ties.

        QuantLinear.apply_smoothing does weight.mul_(s) in place, so a shared parameter would
        silently rescale the embedding table the host still gathers input embeddings from.
        """
        if self.config.quantize_lm_head and self.config.tie_word_embeddings:
            self.lm_head.weight = nn.Parameter(self.lm_head.weight.detach().clone())

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.model(input_ids, position_ids)
        return self.lm_head(hidden_states)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "Qwen/Qwen3.5-2B",
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        **config_overrides,
    ) -> "Qwen3_5ForCausalLM":
        """Load a Qwen3.5 text model with weights from a HuggingFace checkpoint.

        The checkpoint is the full vision-language model; its text-tower weights
        live under model.language_model.* (vision model.visual.* and the
        mtp.* head are ignored). The decoder RMSNorms are zero-centred
        (1 + weight); the +1 is folded into the shared RMSNorm weights
        after loading so the rest of the toolkit sees a plain weight * x norm.
        """
        import dataclasses
        import json

        from ...checkpoint import repo_file

        with open(repo_file(repo_id, "config.json")) as f:
            config = Qwen3_5Config.from_hf_dict(json.load(f))
        if config_overrides:
            config = dataclasses.replace(config, **config_overrides)

        model = cls(config).to(device=device, dtype=dtype)

        # Keep only the text tower and rewrite the model.language_model. prefix
        # to model. so the keys match this module's names.
        #
        # Load shards onto HOST, not device: the checkpoint is the full VLM
        # (text + model.visual.* + mtp.*), so loading it all onto the target device
        # would briefly hold the whole VLM AND the already-allocated text model at once
        # (~18 GB fp32 for the 2B -> OOMs a 12 GB GPU). Loaded on CPU we drop the
        # vision/mtp tensors before they ever touch the device, and load_state_dict
        # copies the kept text tensors into the device model's existing params (a
        # per-param copy_, no second full allocation) -- device peak is just the model itself.
        #
        # The key_filter drops the vision/mtp tensors AS EACH SHARD LOADS, not after
        # load_hf_safetensors returns: filtering a dict comprehension over its result
        # still needs the full VLM (~18 GB fp32) materialized in host RAM first, which
        # is invisible on the GPU path (a separate pool from VRAM) but stacks directly
        # on top of the model's own allocation when device="cpu" -- an OOM there that
        # this same load never causes on cuda.
        prefix = "model.language_model."
        # An untied checkpoint stores lm_head.weight at the TOP level, outside the text
        # tower, so a prefix-only filter drops it -- and allowed_missing below used to
        # swallow that silently, leaving the head tied to the embedding.
        keep = lambda k: k.startswith(prefix) or k == "lm_head.weight"  # noqa: E731
        state_dict = {
            ("model." + k[len(prefix) :] if k.startswith(prefix) else k): v
            for k, v in load_hf_safetensors(
                repo_id, device="cpu", dtype=dtype, key_filter=keep
            ).items()
        }

        # Fold the zero-centred RMSNorm (+1) offset into the weights before loading
        # (input/post-attention/final norms and Q/K norms). GatedRMSNorm is a plain
        # scale and is left as-is.
        rmsnorm_paths = {name for name, m in model.named_modules() if isinstance(m, RMSNorm)}
        for key in list(state_dict):
            module_path = key.rsplit(".weight", 1)[0]
            if module_path in rmsnorm_paths:
                state_dict[key] = state_dict[key] + 1.0

        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        # Only a tied checkpoint legitimately lacks lm_head.weight; for an untied one its
        # absence means the head never loaded, which must not pass silently.
        allowed_missing = ({"lm_head.weight"} if config.tie_word_embeddings else set()) | {
            f"{name}.{buf_name}"
            for name, m in model.named_modules()
            if isinstance(m, ActivationObserver)
            for buf_name, _ in m.named_buffers()
        }
        missing = [k for k in missing if k not in allowed_missing]
        if unexpected:
            raise RuntimeError(f"Unexpected keys in checkpoint: {unexpected}")
        if missing:
            raise RuntimeError(f"Missing keys after loading checkpoint: {missing}")
        if config.tie_word_embeddings:
            # Re-tie, then untie again: a tied checkpoint carries no lm_head.weight
            # (allowed_missing above), so __init__'s untie left a quantized head holding
            # its init values.
            model.lm_head.weight = model.model.embed_tokens.weight
            model._untie_lm_head()
        return model.eval()


class Qwen3_5Architecture(LLMArchitecture):
    """Load + quantize the Qwen3.5 torch reference and export it.

    Knows nothing about Metis kernels. The DeltaNet linear-attention blocks are
    left full precision by default; only the full-attention projections and the
    SwiGLU MLPs are quantized.
    """

    name = "qwen3.5"

    def load_pretrained(self, repo_id, **kwargs):
        # The Metis kernel runs the DeltaNet GEMMs (in_proj_qkv/z/b/a + out_proj) on
        # the int8 IMC, so default quantize_linear_attention=True. A caller can override
        # for the full-precision reference.
        kwargs.setdefault("quantize_linear_attention", True)
        # The short conv runs on the int8 DWPU in the Europa kernel, so quantize it by
        # default too -- otherwise the reference stays float here and the kernel has no
        # calibrated per-channel activation scale to bind.
        kwargs.setdefault("quantize_conv", True)
        return Qwen3_5ForCausalLM.from_pretrained(repo_id, **kwargs)

    def quantize(self, model, calibration, *, smooth_alpha=0.8, qk_alpha=0.5):
        quantize_model(model, calibration, smooth_alpha=smooth_alpha, qk_alpha=qk_alpha)

    def record_lut_ranges(self, model, calibration):
        record_lut_ranges(model, calibration)
        # Also capture the DeltaNet decode kernel's fixed int8 requant scales (peak
        # |L2norm q/k| and |state|): a separate forward pass with decode recording on
        # (it changes the recurrence return signature).
        calibration = list(calibration)
        set_decode_recording(model, True)
        with torch.no_grad():
            for batch in calibration:
                model(batch)
        set_decode_recording(model, False)

    def set_calibration_recording(self, model, on: bool):
        # Fold both the LUT-range recorders and the DeltaNet decode-scale recorders into
        # the calibration passes (--parallel). The decode recorder adds only a
        # side-channel (peak |q/k|/|state|) and returns the recurrence output unchanged,
        # so the activation observers calibrate normally in the same forward.
        from ...quant import set_lut_recording

        set_lut_recording(model, on)
        set_decode_recording(model, on)

    def num_layers(self, model):
        return model.config.num_hidden_layers

    def config_to_dict(self, model_config):
        return asdict(model_config)

    def config_from_dict(self, data):
        return Qwen3_5Config(**data)

    def export_host_io(self, model):
        host = {
            "embed_weight": model.model.embed_tokens.weight.detach()
            .cpu()
            .to(torch.float16)
            .numpy(),
            "final_norm_weight": model.model.norm.weight.detach().cpu().to(torch.float32).numpy(),
            "rms_eps": np.float32(model.config.rms_norm_eps),
            # Scalars the decode loop needs to build per-step RoPE / mask inputs.
            "head_dim": np.int64(model.config.head_dim),
            "rotary_dim": np.int64(model.config.rotary_dim),
            "num_q_heads": np.int64(model.config.num_attention_heads),
            "rope_theta": np.float64(model.config.rope_theta),
        }
        # An untied head is its own weight: the host logits path falls back to embed_weight,
        # which is the WRONG projection once the two are not shared (qwen3.5-9B). Mirrors
        # qwen3's export, including undoing SmoothQuant's in-place weight fold, since the
        # host path feeds the head a plain hidden state with no matching x / s to cancel it.
        if not model.config.tie_word_embeddings:
            weight = model.lm_head.weight.detach().cpu().float()
            if getattr(model.lm_head, "smoothing_enabled", False):
                weight = weight / model.lm_head.smooth_scale.detach().cpu().float()
            host["lm_head_weight"] = weight.to(torch.float16).numpy()
        return host


register_architecture(Qwen3_5Architecture())

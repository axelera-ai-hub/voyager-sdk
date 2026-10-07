# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Reusable decoder-LLM building blocks (torch nn.Module ops).

Architecture-agnostic layers shared across decoder transformers: RMSNorm and
LayerNorm1P, rotary position embeddings (full and partial), a quantizable SDPA, and
the SwiGLU / squared-ReLU MLPs. Depend only on the fake-quant toolkit in quant, not
on any particular model config.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .quant import ActivationObserver, QuantLinear, RangeObserver


class RMSNorm(nn.Module):
    """Root-mean-square layer normalization (x / sqrt(mean(x^2) + eps) * weight).

    Mean over the last dim; reduction in float32 for numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps
        # Records the inverse-sqrt input range for the RMSNorm LUT.
        self.inv_sqrt_range = RangeObserver("inv_sqrt")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(self.inv_sqrt_range(variance + self.eps))
        return self.weight * x.to(input_dtype)


class RotaryEmbedding(nn.Module):
    """Precomputes the cos/sin tables for rotary position embeddings."""

    def __init__(self, head_dim: int, max_position_embeddings: int, theta: float) -> None:
        super().__init__()
        inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [batch, seq_len]
        inv_freq = self.inv_freq[None, :, None].float()  # [1, dim/2, 1]
        position_ids = position_ids[:, None, :].float()  # [batch, 1, seq_len]
        freqs = (inv_freq @ position_ids).transpose(1, 2)  # [batch, seq_len, dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [batch, seq_len, dim]
        return emb.cos(), emb.sin()


# `rotate_half`, `apply_rotary_pos_emb` and `repeat_kv` follow the reference implementation in
# Hugging Face transformers (src/transformers/models/llama/modeling_llama.py), Copyright 2018-
# The Hugging Face team, Apache License 2.0. See licenses/LICENSE-transformers.txt.
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotates the two halves of the last dimension."""
    half = x.shape[-1] // 2
    x1 = x[..., :half]
    x2 = x[..., half:]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Applies RoPE to query and key tensors.

    q/k are shaped [batch, num_heads, seq_len, head_dim] and
    cos/sin are shaped [batch, seq_len, head_dim].
    """
    # The tables are built in float32 for positional accuracy; rotating in that type would
    # hand the rest of the layer a wider activation than it was given (a bf16 calibration then
    # fails at o_proj, the first weight the promoted activation meets). A no-op at float32.
    cos = cos.unsqueeze(1).to(q.dtype)  # [batch, 1, seq_len, head_dim]
    sin = sin.unsqueeze(1).to(q.dtype)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Repeats key/value heads to match the number of query heads (GQA)."""
    batch, num_kv_heads, seq_len, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(batch, num_kv_heads, n_rep, seq_len, head_dim)
    return x.reshape(batch, num_kv_heads * n_rep, seq_len, head_dim)


class QuantSDPA(nn.Module):
    """Scaled dot-product attention with optional fake quantization.

    quantize=False is plain fp attention. quantize=True mirrors the
    production SDPA scheme:

    * Q and K: per-tensor, zero pinned to 0, plus an optional per-head-dim
      QK-smoothing factor that cancels in Q·Kᵀ (K does x/s, Q does
      x·s); see smooth_qk.
    * the softmax numerator exp(scores - max) is fake-quantized per-tensor
      (zero pinned, full int range), but the normalising sum uses the
      *unquantized* weights.
    * V: per-tensor, zero-pinned. Pinning V's zero point keeps the output dequant
      a static scale (a non-zero zero point would make it depend on the per-row
      weight sum); per-tensor matches the kernel's single broadcast-PWORD dequant.
    """

    def __init__(self, head_dim: int, *, quantize: bool = False, bits: int = 8) -> None:
        super().__init__()
        self.scaling = head_dim**-0.5
        self.quantize = quantize
        if not self.quantize:
            return
        # Q/K head-dim channel stats (channel_abs_max) feed the QK-smoothing factor.
        self.q_observer = ActivationObserver(
            bits, pinned_zero=True, signed_full_range=True, num_features=head_dim
        )
        self.k_observer = ActivationObserver(
            bits, pinned_zero=True, signed_full_range=True, num_features=head_dim
        )
        self.w_observer = ActivationObserver(bits, pinned_zero=True, signed_full_range=True)
        self.v_observer = ActivationObserver(bits, pinned_zero=True, signed_full_range=True)
        # QK-smoothing factors (non-persistent; set by smooth_qk).
        self.qk_smoothing_enabled = False
        self.register_buffer("q_smooth_scale", torch.ones(head_dim), persistent=False)
        self.register_buffer("k_smooth_scale", torch.ones(head_dim), persistent=False)
        # LUT input ranges for the softmax exp and reciprocal.
        self.exp_range = RangeObserver("exp")
        self.reciprocal_range = RangeObserver("reciprocal")
        # Optional modelling of the device exp / reciprocal LUTs (set via reference.
        # set_lut_modeling); empty = exact fp softmax (default, unchanged).
        self.lut_sites = frozenset()
        # Chain length of the modelled softmax exp / reciprocal LUTs; device default 2 (LLM path).
        # Callers override per-instance (the dino block uses 1 -- see models/dino/reference.py).
        # Bumping exp (torch-only) also tests how much exp accuracy long-sequence softmax needs.
        self.exp_n_luts = 2
        self.reciprocal_n_luts = 2

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        attn_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.quantize:
            scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
            if attn_mask is not None:
                scores = scores + attn_mask[:, :, :, : k.shape[-2]]
            weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(q.dtype)
            return torch.matmul(weights, v)

        if self.qk_smoothing_enabled:
            q = q / self.q_smooth_scale  # q_smooth_scale = 1/s  ->  q * s
            k = k / self.k_smooth_scale  # k_smooth_scale = s    ->  k / s
        q = self.q_observer(q)
        k = self.k_observer(k)
        scores = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        if attn_mask is not None:
            scores = scores + attn_mask[:, :, :, : k.shape[-2]]
        scores = scores.float()
        exp_input = scores - scores.max(dim=-1, keepdim=True).values
        if self.exp_range.recording:
            # Masked positions underflow exp to 0, so the LUT need not cover them;
            # record only the finite domain.
            finite = exp_input[exp_input > -1e4]
            self.exp_range(finite if finite.numel() else exp_input)
        if "exp" in self.lut_sites:
            # The device exp LUT is fit as exp(-x) over x>=0, so it is fed (max - scores) =
            # -exp_input; ADDITIVE chain reconstructs the softmax numerator.
            from .luts import apply_chained_lut

            weights = apply_chained_lut(
                -exp_input,
                "exp",
                self.exp_range.min_val.item(),
                self.exp_range.max_val.item(),
                n_luts=self.exp_n_luts,
            )
        else:
            weights = torch.exp(exp_input)
        weights_q = self.w_observer(weights)
        v_q = self.v_observer(v)
        out = torch.matmul(weights_q.to(v_q.dtype), v_q)
        denom = self.reciprocal_range(weights.sum(dim=-1, keepdim=True))  # UNQUANTIZED sum
        if "reciprocal" in self.lut_sites:
            from .luts import apply_chained_lut

            recip = apply_chained_lut(
                denom,
                "reciprocal",
                self.reciprocal_range.min_val.item(),
                self.reciprocal_range.max_val.item(),
                n_luts=self.reciprocal_n_luts,
            )
            return (out * recip).to(q.dtype)
        return (out / denom).to(q.dtype)


class SwiGLUMLP(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, hidden_size: int, intermediate_size: int, **quant_kwargs) -> None:
        super().__init__()
        self.gate_proj = QuantLinear(hidden_size, intermediate_size, **quant_kwargs)
        self.up_proj = QuantLinear(hidden_size, intermediate_size, **quant_kwargs)
        self.down_proj = QuantLinear(intermediate_size, hidden_size, **quant_kwargs)
        # Records the SiLU input range for the activation LUT.
        self.silu_range = RangeObserver("silu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.silu_range(self.gate_proj(x)))
        up = self.up_proj(x)
        return self.down_proj(gate * up)


class LayerNorm1P(nn.Module):
    """LayerNorm with a unit-offset weight: F.layer_norm(x, ..., weight + 1, bias, eps).

    The "1P" (one-plus) parameterisation stores the gain as a deviation from 1, so a
    freshly initialised norm is the identity and the checkpoint's `weight` is centred
    on zero. Used by Nemotron-family models (Megatron-LM's `layernorm1p`).

    Unlike RMSNorm this is a *full* LayerNorm: it subtracts the mean and applies a
    bias. Reduction is in float32 for numerical stability.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.normalized_shape = (hidden_size,)
        self.weight = nn.Parameter(torch.zeros(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
        # Records the inverse-sqrt input range for the LayerNorm LUT.
        self.inv_sqrt_range = RangeObserver("inv_sqrt")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        # Spelled out rather than F.layer_norm so the variance flows through the
        # RangeObserver (the device lowers the rsqrt as a LUT, as in RMSNorm).
        mean = x.mean(-1, keepdim=True)
        centered = x - mean
        variance = centered.pow(2).mean(-1, keepdim=True)
        normed = centered * torch.rsqrt(self.inv_sqrt_range(variance + self.eps))
        out = (self.weight + 1.0) * normed + self.bias
        return out.to(input_dtype)


class ReLU2MLP(nn.Module):
    """Squared-ReLU feed-forward network: down_proj(relu(up_proj(x))**2).

    The un-gated MLP used by Nemotron-family models (`hidden_act="relu2"`): a single
    up projection, so it has two thirds of a SwiGLU block's parameters at the same
    intermediate width, and no elementwise gate multiply.
    """

    def __init__(self, hidden_size: int, intermediate_size: int, **quant_kwargs) -> None:
        super().__init__()
        self.up_proj = QuantLinear(hidden_size, intermediate_size, **quant_kwargs)
        self.down_proj = QuantLinear(intermediate_size, hidden_size, **quant_kwargs)
        # Records the relu^2 input range for the activation LUT.
        self.relu2_range = RangeObserver("relu2")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up = self.relu2_range(self.up_proj(x))
        return self.down_proj(torch.square(F.relu(up)))


def apply_partial_rotary_pos_emb(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """RoPE applied to the leading `cos.shape[-1]` channels of each head only.

    Partial rotary embeddings (Nemotron's `partial_rotary_factor`) rotate the first
    `rot_dim` channels of every head and pass the remaining channels through
    unchanged. q/k are [batch, num_heads, seq_len, head_dim]; cos/sin are
    [batch, seq_len, rot_dim] with rot_dim <= head_dim. rot_dim == head_dim
    reduces to apply_rotary_pos_emb.
    """
    rot_dim = cos.shape[-1]
    if rot_dim == q.shape[-1]:
        return apply_rotary_pos_emb(q, k, cos, sin)
    q_rot, q_pass = q[..., :rot_dim], q[..., rot_dim:]
    k_rot, k_pass = k[..., :rot_dim], k[..., rot_dim:]
    q_embed, k_embed = apply_rotary_pos_emb(q_rot, k_rot, cos, sin)
    return torch.cat((q_embed, q_pass), dim=-1), torch.cat((k_embed, k_pass), dim=-1)

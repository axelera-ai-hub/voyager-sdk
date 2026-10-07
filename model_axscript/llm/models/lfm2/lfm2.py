# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Self-contained LFM2 (Liquid Foundation Model 2) reference in PyTorch.

A reference of LiquidAI's LFM2 decoder built from torch / torch.nn only (no
transformers), so it can be quantized, exported via torch.export, and lowered
through the compiler pipeline. Defaults match LiquidAI/LFM2-1.2B.

LFM2 is a hybrid decoder: a stack of short-convolution token mixers with a
handful of grouped-query full-attention layers interleaved (the attention
layers are full_attn_idxs; everything else is a conv block). Per-block
traits:

- Full attention -- grouped-query attention with QK-Norm (per-head RMSNorm on
  Q/K before RoPE), full RoPE, no QKV bias, no gating. Matches Qwen3's attention
  minus the output gate; the projections are named out_proj (not o_proj)
  and the QK-Norms q_layernorm / k_layernorm to mirror the HF checkpoint.
- Short conv (Lfm2ShortConv) -- a double-gated short-range depthwise causal
  convolution. in_proj produces three streams B, C, x; the gated input
  B * x is run through a depthwise causal conv of width conv_L_cache, the
  result is gated again by C, and out_proj maps back to hidden_size.

The channel mixer on every block is a SwiGLU MLP (HF names its linears
w1 / w3 / w2 -- gate / up / down -- which are remapped onto the shared
SwiGLUMLP at load). All RMSNorms are the plain "weight * x" form (ones-init),
so -- unlike Qwen3.5 -- no zero-centred offset folding is needed at load.

What gets quantized: the full-attention q/k/v/out projections and the SwiGLU MLPs
are W8A8 QuantLinear. The short-conv blocks are left full precision by default;
set quantize_conv to make the whole block W8A8 -- its two big GEMMs (in_proj /
out_proj) become QuantLinear and the depthwise conv becomes a QuantConv1d, so the
conv FIR runs on the int8 DWPU rather than the float DPU.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from ...checkpoint import load_hf_safetensors
from ...modules import (
    QuantSDPA,
    RMSNorm,
    RotaryEmbedding,
    SwiGLUMLP,
    apply_rotary_pos_emb,
    repeat_kv,
)
from ...quant import ActivationObserver, QuantConv1d, QuantLinear
from ...quant import quantize_model as _quantize_model
from ...quant import record_lut_ranges
from ..architecture import LLMArchitecture, register_architecture


def _quant_linear_kwargs(config: "LFM2Config") -> dict:
    """The W8A8 keyword arguments shared by every quantizable linear."""
    return {
        "act_bits": config.activation_quant_bits,
        "weight_bits": config.weight_quant_bits,
        "act_symmetric": config.activation_quant_symmetric,
    }


def smoothquant_groups(module: nn.Module) -> list[list["QuantLinear"]]:
    """Groups of QuantLinear that share an input tensor (LFM2 layout).

    q/k/v share the operator_norm output; gate/up share the ffn_norm output.
    out_proj and down_proj have their own input, so they form singleton groups.

    When quantize_conv is set, the short-conv block's big static-weight GEMMs
    (in_proj and out_proj) are QuantLinear and contribute singleton
    groups; the depthwise conv stays full precision. SmoothQuant is self-contained
    per QuantLinear (the x / s fold happens inside its own forward), so
    quantizing the conv block's projections does not perturb anything else.
    """
    groups: list[list[QuantLinear]] = []
    for layer in module.modules():
        if isinstance(layer, Lfm2Attention):
            groups.append([layer.q_proj, layer.k_proj, layer.v_proj])
            groups.append([layer.out_proj])
        elif isinstance(layer, SwiGLUMLP):
            groups.append([layer.gate_proj, layer.up_proj])
            groups.append([layer.down_proj])
        elif isinstance(layer, Lfm2ShortConv):
            groups.extend(
                [m] for m in (layer.in_proj, layer.out_proj) if isinstance(m, QuantLinear)
            )
    return groups


def quantize_model(
    module: nn.Module,
    calibration,
    *,
    smooth_alpha: float | None = 0.8,
    qk_alpha: float | None = 0.5,
) -> None:
    """Calibrate + enable W8A8 (+ SmoothQuant / QK-smoothing) for an LFM2 model.

    Wraps quant.quantize_model with the LFM2 grouping. Must run on a
    freshly-loaded model (the smoothing folds are one-shot). QK-smoothing is only
    active when the attention is itself fake-quantized (quantize_attention); the
    short-conv blocks hold no QuantSDPA, so they are untouched by qk_alpha.
    """
    _quantize_model(
        module,
        calibration,
        groups=smoothquant_groups(module),
        smooth_alpha=smooth_alpha,
        qk_alpha=qk_alpha,
    )


@dataclass
class LFM2Config:
    """Hyperparameters for an LFM2 model. Defaults match LiquidAI/LFM2-1.2B."""

    vocab_size: int = 65536
    hidden_size: int = 2048
    # Raw FFN dim from the checkpoint; the effective MLP width is
    # mlp_intermediate_size (block_auto_adjust_ff_dim rounds it down/up).
    intermediate_size: int = 12288
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 64
    # Hybrid stack: these layer indices are full attention; the rest are conv.
    full_attn_idxs: tuple[int, ...] = (2, 5, 8, 10, 12, 14)
    # Short-conv mixer.
    conv_L_cache: int = 3
    conv_bias: bool = False
    # MLP width adjustment (matches HF Lfm2MLP).
    block_multiple_of: int = 256
    block_ffn_dim_multiplier: float | None = 1.0
    block_auto_adjust_ff_dim: bool = True
    # RoPE / norm.
    max_position_embeddings: int = 128_000
    norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True
    # W8A8: per-tensor asymmetric activations, per-channel symmetric weights.
    activation_quant_bits: int = 8
    weight_quant_bits: int = 8
    activation_quant_symmetric: bool = False
    # When enabled, full-attention Q/K/V and softmax weights are fake-quantized.
    quantize_attention: bool = False
    sdpa_quant_bits: int = 8
    # When enabled, the whole short-conv block is W8A8: the big GEMMs (in_proj,
    # out_proj) become QuantLinear and the depthwise conv becomes a QuantConv1d
    # (its FIR then runs on the int8 DWPU). Requires conv_bias=False.
    quantize_conv: bool = False

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @property
    def mlp_intermediate_size(self) -> int:
        """Effective SwiGLU width after the block_auto_adjust_ff_dim rounding."""
        inter = self.intermediate_size
        if self.block_auto_adjust_ff_dim:
            inter = int(2 * inter / 3)
            if self.block_ffn_dim_multiplier is not None:
                inter = int(self.block_ffn_dim_multiplier * inter)
                inter = self.block_multiple_of * (
                    (inter + self.block_multiple_of - 1) // self.block_multiple_of
                )
        return inter

    def layer_type(self, layer_idx: int) -> str:
        """'full_attention' for indices in full_attn_idxs, else 'conv'."""
        return "full_attention" if layer_idx in self.full_attn_idxs else "conv"

    @classmethod
    def from_hf_dict(cls, cfg: dict) -> "LFM2Config":
        """Builds an LFM2Config from a HuggingFace config.json dict."""
        hidden = cfg["hidden_size"]
        heads = cfg["num_attention_heads"]
        # The original LFM2 config calls the FFN dim block_ff_dim.
        inter = cfg.get("intermediate_size", cfg.get("block_ff_dim"))
        # Derive the attention layer indices from layer_types if present, else
        # full_attn_idxs (older configs), else "every layer is attention".
        layer_types = cfg.get("layer_types")
        if layer_types is not None:
            full_idxs = tuple(i for i, t in enumerate(layer_types) if t == "full_attention")
        else:
            full_idxs = tuple(cfg.get("full_attn_idxs", range(cfg["num_hidden_layers"])))
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=hidden,
            intermediate_size=inter,
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=heads,
            num_key_value_heads=cfg["num_key_value_heads"],
            head_dim=cfg.get("head_dim", hidden // heads),
            full_attn_idxs=full_idxs,
            conv_L_cache=cfg.get("conv_L_cache", 3),
            conv_bias=cfg.get("conv_bias", False),
            block_multiple_of=cfg.get("block_multiple_of", 256),
            block_ffn_dim_multiplier=cfg.get("block_ffn_dim_multiplier", 1.0),
            block_auto_adjust_ff_dim=cfg.get("block_auto_adjust_ff_dim", True),
            max_position_embeddings=cfg["max_position_embeddings"],
            norm_eps=cfg.get("norm_eps", 1e-5),
            rope_theta=cfg.get("rope_theta", cfg.get("theta", 1_000_000.0)),
            tie_word_embeddings=cfg.get("tie_word_embeddings", True),
        )


class Lfm2Attention(nn.Module):
    """Grouped-query attention with QK-Norm and RoPE (no gating)."""

    def __init__(self, config: LFM2Config) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_key_value_groups

        qkw = _quant_linear_kwargs(config)
        self.q_proj = QuantLinear(config.hidden_size, self.num_heads * self.head_dim, **qkw)
        self.k_proj = QuantLinear(config.hidden_size, self.num_kv_heads * self.head_dim, **qkw)
        self.v_proj = QuantLinear(config.hidden_size, self.num_kv_heads * self.head_dim, **qkw)
        self.out_proj = QuantLinear(self.num_heads * self.head_dim, config.hidden_size, **qkw)

        # QK-Norm: per-head RMSNorm over head_dim, applied before RoPE.
        self.q_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)
        self.k_layernorm = RMSNorm(self.head_dim, eps=config.norm_eps)

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
        hidden_shape = (batch, seq_len, -1, self.head_dim)

        # Project, split into heads, QK-Norm, then -> [batch, heads, seq, head_dim].
        q = self.q_layernorm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        k = self.k_layernorm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand KV heads to match the number of query heads (GQA).
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        attn_output = self.sdpa(q, k, v, attention_mask)  # [batch, heads, seq, head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.out_proj(attn_output)


class Lfm2ShortConv(nn.Module):
    """Double-gated short-range depthwise causal convolution token mixer.

    in_proj emits three hidden-width streams B, C, x; the gated input B * x is
    convolved by a depthwise causal conv of width conv_L_cache, gated again by
    C, and projected back by out_proj. With config.quantize_conv the two
    big GEMMs are W8A8 QuantLinear and the depthwise conv is a W8A8 QuantConv1d
    (int8 DWPU); otherwise everything is full precision.
    """

    def __init__(self, config: LFM2Config) -> None:
        super().__init__()
        hidden = config.hidden_size
        self.L_cache = config.conv_L_cache
        if config.quantize_conv:
            qkw = _quant_linear_kwargs(config)
            # The depthwise conv is W8A8 too (runs on the int8 DWPU); QuantConv1d is
            # bias-free, so quantized convs require conv_bias=False.
            if config.conv_bias:
                raise ValueError("quantize_conv does not support conv_bias=True")
            # Pin the conv activation SYMMETRIC (zero-point 0), unlike the linears: the
            # int8 DWPU carries a persistent conv-window cache that is zero-initialised
            # at instance load, and quantized-zero must equal integer 0 for the causal
            # zero-padding / fresh state to be correct (an affine zero-point would need
            # the cache seeded with z_x). Zero-point 0 also drops the dz dequant term.
            self.conv: nn.Module = QuantConv1d(
                hidden, self.L_cache, **{**qkw, "act_symmetric": True}
            )
            self.in_proj: nn.Module = QuantLinear(hidden, 3 * hidden, **qkw)
            self.out_proj: nn.Module = QuantLinear(hidden, hidden, **qkw)
        else:
            self.conv = nn.Conv1d(
                in_channels=hidden,
                out_channels=hidden,
                kernel_size=self.L_cache,
                groups=hidden,
                bias=config.conv_bias,
                padding=self.L_cache - 1,
            )
            self.in_proj = nn.Linear(hidden, 3 * hidden, bias=config.conv_bias)
            self.out_proj = nn.Linear(hidden, hidden, bias=config.conv_bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        # cos/sin/attention_mask accepted for uniform decoder-layer signature but unused:
        # the conv has no RoPE and is implicitly causal (right padding is dropped).
        seq_len = hidden_states.shape[1]
        BCx = self.in_proj(hidden_states)  # [batch, seq, 3 * hidden]
        B, C, x = BCx.chunk(3, dim=-1)
        Bx = (B * x).transpose(1, 2)  # [batch, hidden, seq]
        conv_out = self.conv(Bx)[..., :seq_len].transpose(1, 2)  # [batch, seq, hidden]
        y = C * conv_out
        return self.out_proj(y)


class Lfm2DecoderLayer(nn.Module):
    """A single hybrid transformer decoder block (pre-norm).

    The token mixer is full attention on full-attention layers and a short-conv
    block otherwise; the channel mixer is always a SwiGLU MLP (named
    feed_forward to match the HF checkpoint).
    """

    def __init__(self, config: LFM2Config, layer_idx: int) -> None:
        super().__init__()
        self.layer_type = config.layer_type(layer_idx)
        self.is_attention_layer = self.layer_type == "full_attention"
        if self.is_attention_layer:
            self.self_attn: nn.Module = Lfm2Attention(config)
        else:
            self.conv: nn.Module = Lfm2ShortConv(config)
        self.feed_forward = SwiGLUMLP(
            config.hidden_size, config.mlp_intermediate_size, **_quant_linear_kwargs(config)
        )
        self.operator_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.ffn_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        mixer = self.self_attn if self.is_attention_layer else self.conv

        residual = hidden_states
        hidden_states = mixer(self.operator_norm(hidden_states), cos, sin, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.feed_forward(self.ffn_norm(hidden_states))
        hidden_states = residual + hidden_states
        return hidden_states


class Lfm2Model(nn.Module):
    """The LFM2 trunk (embeddings + hybrid decoder layers + final norm)."""

    def __init__(self, config: LFM2Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Lfm2DecoderLayer(config, i) for i in range(config.num_hidden_layers)]
        )
        self.embedding_norm = RMSNorm(config.hidden_size, eps=config.norm_eps)
        self.rotary_emb = RotaryEmbedding(
            config.head_dim, config.max_position_embeddings, config.rope_theta
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

        return self.embedding_norm(hidden_states)


def _remap_hf_key(key: str) -> str:
    """Rewrite HF LFM2 keys onto this module's names.

    The HF Lfm2MLP names its SwiGLU linears w1/w3/w2 (gate/up/down); the shared
    SwiGLUMLP uses gate_proj/up_proj/down_proj. Everything else already matches.
    """
    return (
        key.replace("feed_forward.w1.", "feed_forward.gate_proj.")
        .replace("feed_forward.w3.", "feed_forward.up_proj.")
        .replace("feed_forward.w2.", "feed_forward.down_proj.")
    )


class Lfm2ForCausalLM(nn.Module):
    """LFM2 with a language-modeling head."""

    def __init__(self, config: LFM2Config) -> None:
        super().__init__()
        self.config = config
        self.model = Lfm2Model(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

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
        repo_id: str = "LiquidAI/LFM2-1.2B",
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        **config_overrides,
    ) -> "Lfm2ForCausalLM":
        """Loads an LFM2 model with weights from a HuggingFace checkpoint.

        HF names match this module's names apart from the SwiGLU linears
        (w1/w3/w2 -> gate/up/down; see _remap_hf_key). config_overrides are applied
        to the derived config (e.g. quantize_attention=True, quantize_conv=True).
        """
        import dataclasses
        import json

        from ...checkpoint import repo_file

        with open(repo_file(repo_id, "config.json")) as f:
            config = LFM2Config.from_hf_dict(json.load(f))
        if config_overrides:
            config = dataclasses.replace(config, **config_overrides)

        model = cls(config).to(device=device, dtype=dtype)
        state_dict = {
            _remap_hf_key(k): v
            for k, v in load_hf_safetensors(repo_id, device=device, dtype=dtype).items()
        }

        # Tied checkpoints omit lm_head.weight; observers add buffers absent from
        # the checkpoint, so strict=False is expected here.
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        allowed_missing = {"lm_head.weight"} | {
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
            model.lm_head.weight = model.model.embed_tokens.weight
        return model.eval()


class Lfm2Architecture(LLMArchitecture):
    """LFM2 architecture: loads and quantizes the torch reference for export.

    Knows nothing about Metis kernels. The short-conv linear-mixer blocks are left
    full precision by default; only the full-attention projections and the SwiGLU
    MLPs are quantized (set quantize_conv to W8A8 the conv blocks' GEMMs too).
    """

    name = "lfm2"

    def load_pretrained(self, repo_id, **kwargs):
        # The only LFM2 Metis implementation (lfm2-metis-opt) runs the short-conv
        # block's in_proj/out_proj as W8A8, so quantize them by default for the
        # export/compile path (callers can still override).
        kwargs.setdefault("quantize_conv", True)
        return Lfm2ForCausalLM.from_pretrained(repo_id, **kwargs)

    def quantize(self, model, calibration, *, smooth_alpha=0.8, qk_alpha=0.5):
        quantize_model(model, calibration, smooth_alpha=smooth_alpha, qk_alpha=qk_alpha)

    def record_lut_ranges(self, model, calibration):
        record_lut_ranges(model, calibration)

    def num_layers(self, model):
        return model.config.num_hidden_layers

    def config_to_dict(self, model_config):
        return asdict(model_config)

    def config_from_dict(self, data):
        return LFM2Config(**data)

    def export_host_io(self, model):
        return {
            "embed_weight": model.model.embed_tokens.weight.detach()
            .cpu()
            .to(torch.float16)
            .numpy(),
            "final_norm_weight": model.model.embedding_norm.weight.detach()
            .cpu()
            .to(torch.float32)
            .numpy(),
            "rms_eps": np.float32(model.config.norm_eps),
            # Scalars the decode loop needs to build per-step RoPE / mask inputs.
            "head_dim": np.int64(model.config.head_dim),
            "num_q_heads": np.int64(model.config.num_attention_heads),
            "rope_theta": np.float64(model.config.rope_theta),
        }


register_architecture(Lfm2Architecture())

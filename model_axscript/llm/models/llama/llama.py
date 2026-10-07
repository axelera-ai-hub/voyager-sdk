# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Self-contained Llama 3.2 reference implementation in PyTorch.

A reference Llama decoder-only transformer built from torch / torch.nn only (no
transformers), so it can be exported and lowered through the compiler pipeline.

Llama traits: no QK-Norm (Q/K go straight from projection to RoPE), grouped-query
attention, no QKV bias, and the "llama3" RoPE frequency rescaling (a piecewise
wavelength interpolation applied to inv_freq). Llama 3.2 1B has head_dim=64
(hidden 2048 / 32 heads), which the Metis kernel handles via block-interleaved
padding (see naive_impl).

The matching hardware kernel and param binding live in naive_impl.
"""

from dataclasses import asdict, dataclass, field

import numpy as np
import torch
import torch.nn as nn

from ...checkpoint import load_hf_safetensors
from ...modules import QuantSDPA, RMSNorm, SwiGLUMLP, apply_rotary_pos_emb, repeat_kv
from ...quant import ActivationObserver, QuantLinear
from ...quant import quantize_model as _quantize_model
from ...quant import record_lut_ranges
from ..architecture import LLMArchitecture, register_architecture


def _quant_linear_kwargs(config: "LlamaConfig") -> dict:
    """The W8A8 keyword arguments shared by every quantizable linear."""
    return {
        "act_bits": config.activation_quant_bits,
        "weight_bits": config.weight_quant_bits,
        "act_symmetric": config.activation_quant_symmetric,
    }


def llama3_inv_freq(head_dim: int, rope_theta: float, scaling: dict | None) -> torch.Tensor:
    """RoPE inverse frequencies with the optional "llama3" wavelength rescaling.

    Standard RoPE inv_freq = 1 / theta**(2i/d). The "llama3" scaling (Llama 3.x)
    leaves high-frequency components untouched, divides low-frequency components
    by `factor`, and smoothly interpolates between the two across a wavelength
    band, so the model extrapolates to longer contexts. With scaling=None this is
    plain RoPE.
    """
    inv_freq = 1.0 / (rope_theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim))
    if scaling is None:
        return inv_freq

    factor = scaling["factor"]
    low_freq_factor = scaling["low_freq_factor"]
    high_freq_factor = scaling["high_freq_factor"]
    old_context_len = scaling["original_max_position_embeddings"]

    low_freq_wavelen = old_context_len / low_freq_factor
    high_freq_wavelen = old_context_len / high_freq_factor

    wavelen = 2 * np.pi / inv_freq
    # Low-frequency (long wavelength) components: divide by factor.
    inv_freq_llama = torch.where(wavelen > low_freq_wavelen, inv_freq / factor, inv_freq)
    # Mid band: smooth interpolation between scaled and unscaled.
    smooth = (old_context_len / wavelen - low_freq_factor) / (high_freq_factor - low_freq_factor)
    smoothed = (1 - smooth) * inv_freq_llama / factor + smooth * inv_freq_llama
    is_medium = (wavelen >= high_freq_wavelen) & (wavelen <= low_freq_wavelen)
    return torch.where(is_medium, smoothed, inv_freq_llama)


class LlamaRotaryEmbedding(nn.Module):
    """RoPE cos/sin tables using the (optionally rescaled) llama3 frequencies."""

    def __init__(
        self, head_dim: int, max_position_embeddings: int, theta: float, scaling: dict | None
    ) -> None:
        super().__init__()
        inv_freq = llama3_inv_freq(head_dim, theta, scaling)
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


def smoothquant_groups(module: nn.Module) -> list[list["QuantLinear"]]:
    """Groups of QuantLinear that share an input tensor (Llama layout).

    q/k/v share the input_layernorm output; gate/up share the
    post_attention_layernorm output. o_proj and down_proj have their own input,
    so they form singleton groups. Identical structure to Qwen3.
    """
    groups: list[list[QuantLinear]] = []
    for layer in module.modules():
        if isinstance(layer, LlamaAttention):
            groups.append([layer.q_proj, layer.k_proj, layer.v_proj])
            groups.append([layer.o_proj])
        elif isinstance(layer, SwiGLUMLP):
            groups.append([layer.gate_proj, layer.up_proj])
            groups.append([layer.down_proj])
    return groups


def quantize_model(
    module: nn.Module,
    calibration,
    *,
    smooth_alpha: float | None = 0.8,
    qk_alpha: float | None = None,
) -> None:
    """Calibrate + enable W8A8 (+ SmoothQuant) for a Llama model.

    QK-smoothing is OFF by default (qk_alpha=None): Llama has no QK-Norm so Q/K go
    straight from projection to RoPE and per-head-dim QK rescaling is not used.
    Attention is still W8A8-quantized (per-tensor Q/K/V). Pass qk_alpha explicitly
    to re-enable QK-smoothing.
    """
    _quantize_model(
        module,
        calibration,
        groups=smoothquant_groups(module),
        smooth_alpha=smooth_alpha,
        qk_alpha=qk_alpha,
    )


@dataclass
class LlamaConfig:
    """Hyperparameters for a Llama 3.2 model. Defaults match Llama-3.2-1B."""

    vocab_size: int = 128256
    hidden_size: int = 2048
    intermediate_size: int = 8192
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 64
    max_position_embeddings: int = 131072
    rms_norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    tie_word_embeddings: bool = True
    # "llama3" RoPE rescaling (None for plain RoPE). Defaults match Llama-3.2-1B.
    rope_scaling: dict | None = field(
        default_factory=lambda: {
            "factor": 32.0,
            "low_freq_factor": 1.0,
            "high_freq_factor": 4.0,
            "original_max_position_embeddings": 8192,
        }
    )
    # W8A8: per-tensor asymmetric activations, per-channel symmetric weights.
    activation_quant_bits: int = 8
    weight_quant_bits: int = 8
    activation_quant_symmetric: bool = False
    # When enabled, Q/K/V and softmax weights are fake-quantized.
    quantize_attention: bool = False
    sdpa_quant_bits: int = 8

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_hf_dict(cls, cfg: dict) -> "LlamaConfig":
        """Builds a LlamaConfig from a HuggingFace config.json dict."""
        head_dim = cfg.get("head_dim", cfg["hidden_size"] // cfg["num_attention_heads"])
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            head_dim=head_dim,
            max_position_embeddings=cfg["max_position_embeddings"],
            rms_norm_eps=cfg["rms_norm_eps"],
            rope_theta=cfg["rope_theta"],
            tie_word_embeddings=cfg.get("tie_word_embeddings", True),
            rope_scaling=cfg.get("rope_scaling"),
        )


class LlamaAttention(nn.Module):
    """Grouped-query attention with RoPE (no QK-Norm)."""

    def __init__(self, config: LlamaConfig) -> None:
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
        self.o_proj = QuantLinear(self.num_heads * self.head_dim, config.hidden_size, **qkw)

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

        # Project and split into heads: [batch, num_heads, seq_len, head_dim].
        q = self.q_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        k = self.k_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        v = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand KV heads to match the number of query heads (GQA).
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        attn_output = self.sdpa(q, k, v, attention_mask)  # [batch, heads, seq, head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(attn_output)


class LlamaDecoderLayer(nn.Module):
    """A single transformer decoder block (pre-norm)."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(config)
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
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states, cos, sin, attention_mask)
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class LlamaModel(nn.Module):
    """The Llama transformer trunk (embeddings + decoder layers + final norm)."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [LlamaDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = LlamaRotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            config.rope_scaling,
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


class LlamaForCausalLM(nn.Module):
    """Llama with a language-modeling head."""

    def __init__(self, config: LlamaConfig) -> None:
        super().__init__()
        self.config = config
        self.model = LlamaModel(config)
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
        repo_id: str = "meta-llama/Llama-3.2-1B",
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        **config_overrides,
    ) -> "LlamaForCausalLM":
        """Loads a Llama model with weights from a HuggingFace checkpoint.

        HF parameter names match this module's names exactly. config_overrides
        are applied to the derived config (e.g. quantize_attention=True).
        """
        import dataclasses
        import json

        from ...checkpoint import repo_file

        with open(repo_file(repo_id, "config.json")) as f:
            config = LlamaConfig.from_hf_dict(json.load(f))
        if config_overrides:
            config = dataclasses.replace(config, **config_overrides)

        model = cls(config).to(device=device, dtype=dtype)
        # Stage the checkpoint on CPU, not `device`: load_state_dict copies it into the
        # already-on-device model param-by-param (CPU->device per tensor), so only ONE
        # full copy of the weights is ever resident on `device`. Loading the state_dict
        # ALSO on `device` would hold a second full copy there during the copy (~2x the
        # model), which OOMs a ~12GB GPU for a 1B fp32 model. (Same fix as Qwen3.5.)
        state_dict = load_hf_safetensors(repo_id, device="cpu", dtype=dtype)

        # Tied checkpoints omit lm_head.weight; ActivationObserver adds buffers absent
        # from the checkpoint, so strict=False is expected here.
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


class LlamaArchitecture(LLMArchitecture):
    """LLMArchitecture implementation for Llama: load + quantize the torch reference and export it."""

    name = "llama"

    def load_pretrained(self, repo_id, **kwargs):
        return LlamaForCausalLM.from_pretrained(repo_id, **kwargs)

    def quantize(self, model, calibration, *, smooth_alpha=0.8, qk_alpha=None):
        # qk_alpha is always forced to None: Llama has no QK-Norm so QK-smoothing
        # is not applicable. Linear SmoothQuant (smooth_alpha) is unaffected.
        quantize_model(model, calibration, smooth_alpha=smooth_alpha, qk_alpha=None)

    def record_lut_ranges(self, model, calibration):
        record_lut_ranges(model, calibration)

    def num_layers(self, model):
        return model.config.num_hidden_layers

    def config_to_dict(self, model_config):
        return asdict(model_config)

    def config_from_dict(self, data):
        return LlamaConfig(**data)

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
            "num_q_heads": np.int64(model.config.num_attention_heads),
            "rope_theta": np.float64(model.config.rope_theta),
            # The precomputed (rescaled) RoPE inverse frequencies, so the host
            # RoPE tables match the model's "llama3" scaling without re-deriving it.
            "rope_inv_freq": llama3_inv_freq(
                model.config.head_dim, model.config.rope_theta, model.config.rope_scaling
            )
            .cpu()
            .to(torch.float64)
            .numpy(),
        }
        # Untied models (e.g. Llama-3.1-8B) project the output through a SEPARATE lm_head; export it
        # so the host lm_head uses the right weights. Tied models reuse embed_weight (the host lm_head
        # falls back to it), so we skip this ~1 GB duplicate there.
        if not model.config.tie_word_embeddings:
            host["lm_head_weight"] = model.lm_head.weight.detach().cpu().to(torch.float16).numpy()
        return host


register_architecture(LlamaArchitecture())

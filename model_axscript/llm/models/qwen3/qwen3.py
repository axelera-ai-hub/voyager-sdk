# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Self-contained Qwen3 implementation in PyTorch.

A reference Qwen3 decoder-only transformer built from torch / torch.nn
only (no transformers), so it can be exported via torch.export and
lowered through the compiler pipeline.

Qwen3-specific traits vs a generic LLaMA-style decoder: QK-Norm (per-head
RMSNorm on Q/K before RoPE — the distinguishing feature over Qwen2), grouped-query
attention, no QKV bias, and a head_dim decoupled from hidden_size.

The matching hardware kernel and param binding live in naive_impl.
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
from ...quant import ActivationObserver, QuantLinear
from ...quant import quantize_model as _quantize_model
from ...quant import record_lut_ranges
from ..architecture import LLMArchitecture, register_architecture


def _quant_linear_kwargs(config: "Qwen3Config") -> dict:
    """The W8A8 keyword arguments shared by every quantizable linear."""
    return {
        "act_bits": config.activation_quant_bits,
        "weight_bits": config.weight_quant_bits,
        "act_symmetric": config.activation_quant_symmetric,
    }


def smoothquant_groups(module: nn.Module) -> list[list["QuantLinear"]]:
    """Groups of QuantLinear that share an input tensor (Qwen3 layout).

    Linears fed by the same activation (q/k/v from input_layernorm; gate/up
    from post_attention_layernorm) must share one smoothing factor so they
    keep identical activation quant params (the shared input is quantized once,
    and the 1/s fold has a single RMSNorm to go into). o_proj /
    down_proj have their own input, so they form singleton groups.
    """
    groups: list[list[QuantLinear]] = []
    for layer in module.modules():
        if isinstance(layer, Qwen3Attention):
            groups.append([layer.q_proj, layer.k_proj, layer.v_proj])
            groups.append([layer.o_proj])
        elif isinstance(layer, SwiGLUMLP):
            groups.append([layer.gate_proj, layer.up_proj])
            groups.append([layer.down_proj])
    # The lm_head is fed by the final norm alone, so it smooths as a singleton. Without a group
    # it would keep smooth = 1 and its qparams would be per-tensor, which breaches the ppl gate.
    head = getattr(module, "lm_head", None)
    if isinstance(head, QuantLinear):
        groups.append([head])
    return groups


def quantize_model(
    module: nn.Module,
    calibration,
    *,
    smooth_alpha: float | None = 0.8,
    qk_alpha: float | None = 0.5,
) -> None:
    """Calibrate + enable W8A8 (+ SmoothQuant / QK-smoothing) for a Qwen3 model.

    Wraps quant.quantize_model with the Qwen3 grouping. Must run on a
    freshly-loaded model (the smoothing folds are one-shot).
    """
    _quantize_model(
        module,
        calibration,
        groups=smoothquant_groups(module),
        smooth_alpha=smooth_alpha,
        qk_alpha=qk_alpha,
    )


@dataclass
class Qwen3Config:
    """Hyperparameters for a Qwen3 model.

    Defaults correspond to the Qwen3-0.6B checkpoint.
    """

    vocab_size: int = 151936
    hidden_size: int = 1024
    intermediate_size: int = 3072
    num_hidden_layers: int = 28
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 40960
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1_000_000.0
    tie_word_embeddings: bool = True
    # W8A8 defaults match the production Qwen kernels: per-tensor asymmetric
    # activations (per-channel part comes from SmoothQuant), per-channel symmetric weights.
    activation_quant_bits: int = 8
    weight_quant_bits: int = 8
    activation_quant_symmetric: bool = False
    # When enabled, Q/K/V and softmax weights are fake-quantized.
    quantize_attention: bool = False
    sdpa_quant_bits: int = 8
    # When enabled the lm_head is a QuantLinear, so it exports an int8 weight + qparams and runs
    # on device. Smoothed per-channel (per-tensor breaches the ppl gate) and untied from the
    # embedding, since smoothing mutates the weight in place.
    quantize_lm_head: bool = False

    @property
    def num_key_value_groups(self) -> int:
        return self.num_attention_heads // self.num_key_value_heads

    @classmethod
    def from_hf_dict(cls, cfg: dict) -> "Qwen3Config":
        """Builds a Qwen3Config from a HuggingFace config.json dict."""
        return cls(
            vocab_size=cfg["vocab_size"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_hidden_layers=cfg["num_hidden_layers"],
            num_attention_heads=cfg["num_attention_heads"],
            num_key_value_heads=cfg["num_key_value_heads"],
            head_dim=cfg["head_dim"],
            max_position_embeddings=cfg["max_position_embeddings"],
            rms_norm_eps=cfg["rms_norm_eps"],
            rope_theta=cfg["rope_theta"],
            tie_word_embeddings=cfg["tie_word_embeddings"],
        )


class Qwen3Attention(nn.Module):
    """Grouped-query attention with QK-Norm and RoPE."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_key_value_groups

        # qkv projections are W8A8-quantizable; o_proj is left full precision.
        qkw = _quant_linear_kwargs(config)
        self.q_proj = QuantLinear(config.hidden_size, self.num_heads * self.head_dim, **qkw)
        self.k_proj = QuantLinear(config.hidden_size, self.num_kv_heads * self.head_dim, **qkw)
        self.v_proj = QuantLinear(config.hidden_size, self.num_kv_heads * self.head_dim, **qkw)
        self.o_proj = QuantLinear(self.num_heads * self.head_dim, config.hidden_size, **qkw)

        # QK-Norm: per-head RMSNorm over head_dim, applied before RoPE.
        self.q_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        # Scaled dot-product attention (optionally fake-quantized).
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
        q = self.q_proj(hidden_states).view(hidden_shape)
        k = self.k_proj(hidden_states).view(hidden_shape)
        v = self.v_proj(hidden_states).view(hidden_shape)
        q = self.q_norm(q).transpose(1, 2)
        k = self.k_norm(k).transpose(1, 2)
        v = v.transpose(1, 2)

        q, k = apply_rotary_pos_emb(q, k, cos, sin)

        # Expand KV heads to match the number of query heads (GQA).
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        attn_output = self.sdpa(q, k, v, attention_mask)  # [batch, heads, seq, head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(attn_output)


class Qwen3DecoderLayer(nn.Module):
    """A single transformer decoder block (pre-norm)."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.self_attn = Qwen3Attention(config)
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


class Qwen3Model(nn.Module):
    """The Qwen3 transformer trunk (embeddings + decoder layers + final norm)."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [Qwen3DecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
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

        return self.norm(hidden_states)


class Qwen3ForCausalLM(nn.Module):
    """Qwen3 with a language-modeling head."""

    def __init__(self, config: Qwen3Config) -> None:
        super().__init__()
        self.config = config
        self.model = Qwen3Model(config)
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
        repo_id: str = "Qwen/Qwen3-0.6B",
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        **config_overrides,
    ) -> "Qwen3ForCausalLM":
        """Loads a Qwen3 model with weights from a HuggingFace checkpoint.

        HF parameter names match this module's names exactly. config_overrides
        are applied to the derived config (e.g. quantize_attention=True).
        """
        import dataclasses
        import json

        from ...checkpoint import repo_file

        with open(repo_file(repo_id, "config.json")) as f:
            config = Qwen3Config.from_hf_dict(json.load(f))
        if config_overrides:
            config = dataclasses.replace(config, **config_overrides)

        model = cls(config).to(device=device, dtype=dtype)
        # Load the checkpoint into host RAM, not onto `device`: the model params are already
        # materialized on `device`, so load_state_dict copies CPU->device tensor-by-tensor.
        # Loading the state dict onto `device` too would keep a second full copy of the
        # weights resident there, ~2x peak that OOMs a fp32 ~2B-param model on a 12GB GPU.
        # (llama / qwen3_5 load_pretrained already stage on CPU for the same reason.)
        state_dict = load_hf_safetensors(repo_id, device="cpu", dtype=dtype)

        # Tied checkpoints omit lm_head.weight (already shared with the
        # embedding) and observers add buffers absent from the checkpoint, so
        # strict=False is expected here.
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
            # Re-tie: __init__'s untie left a quantized head holding its init values, since a
            # tied checkpoint carries no lm_head.weight (allowed_missing above).
            model.lm_head.weight = model.model.embed_tokens.weight
            model._untie_lm_head()
        return model.eval()


class Qwen3Architecture(LLMArchitecture):
    """Step 1 for Qwen3: load + quantize the torch reference and export it.

    Knows nothing about Metis kernels — that is the Qwen3 implementation
    (naive_impl.Qwen3MetisImplementation).
    """

    name = "qwen3"

    def load_pretrained(self, repo_id, **kwargs):
        return Qwen3ForCausalLM.from_pretrained(repo_id, **kwargs)

    def quantize(self, model, calibration, *, smooth_alpha=0.8, qk_alpha=0.5):
        quantize_model(model, calibration, smooth_alpha=smooth_alpha, qk_alpha=qk_alpha)

    def record_lut_ranges(self, model, calibration):
        record_lut_ranges(model, calibration)

    def num_layers(self, model):
        return model.config.num_hidden_layers

    def config_to_dict(self, model_config):
        return asdict(model_config)

    def config_from_dict(self, data):
        return Qwen3Config(**data)

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
        }
        # Untied models need their separate output projection exported; tied ones reuse
        # embed_weight (the host lm_head falls back to it). Qwen3 is tied up to 4B, untied from 8B.
        if not model.config.tie_word_embeddings:
            weight = model.lm_head.weight.detach().cpu().float()
            if getattr(model.lm_head, "smoothing_enabled", False):
                # Undo apply_smoothing's weight.mul_(s): the host logits paths feed the head a
                # plain hidden state, with no matching x / s to cancel the fold.
                weight = weight / model.lm_head.smooth_scale.detach().cpu().float()
            host["lm_head_weight"] = weight.to(torch.float16).numpy()
        return host


register_architecture(Qwen3Architecture())

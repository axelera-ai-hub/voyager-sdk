# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Self-contained Domyn Small (Nemotron) reference implementation in PyTorch.

A reference decoder-only transformer built from torch / torch.nn only (no
transformers), so it can be exported and lowered through the compiler pipeline.
Matches `NemotronForCausalLM` -- the architecture of
https://huggingface.co/domyn/Domyn-Small-v1.0 (10B, `model_type: "nemotron"`).

Three traits set it apart from the Llama/Qwen3 references in this package, and
each one is a place a shared kernel does *not* carry over:

* **LayerNorm1P, not RMSNorm.** Every norm is a full LayerNorm -- it subtracts
  the mean and adds a bias -- with the gain stored as `weight + 1`. That includes
  the final norm, which lives on the host side of the boundary (see
  `DomynArchitecture.export_host_io` and the note there).
* **Squared-ReLU MLP, not SwiGLU.** `down_proj(relu(up_proj(x))**2)`: one up
  projection, no gate multiply. So the MLP's SmoothQuant groups are two
  singletons rather than a {gate, up} pair.
* **Partial RoPE.** `partial_rotary_factor=0.5` rotates only the leading
  `head_dim * 0.5` channels of each head and passes the rest through. The
  inverse frequencies are derived over the *rotary* width, not `head_dim`.

Also note `head_dim` (128) is NOT `hidden_size // num_attention_heads`
(4096 // 48): q_proj widens 4096 -> 6144. GQA is 48 query heads over 8 KV heads.

This module registers the quantizable torch reference only. The device kernel that
realises the three traits above lives beside it in `metis_impl.py` (implementation
`domyn-metis-multicore`), and `export_host_io` below documents the host-boundary
half: a LayerNorm1P final norm and an untied lm_head.
"""

from dataclasses import asdict, dataclass

import numpy as np
import torch
import torch.nn as nn

from ...checkpoint import stream_hf_safetensors_into
from ...modules import LayerNorm1P, QuantSDPA, ReLU2MLP, apply_partial_rotary_pos_emb, repeat_kv
from ...quant import ActivationObserver, QuantLinear
from ...quant import quantize_model as _quantize_model
from ...quant import record_lut_ranges
from ..architecture import LLMArchitecture, register_architecture


def _quant_linear_kwargs(config: "DomynConfig") -> dict:
    """The W8A8 keyword arguments shared by every quantizable linear."""
    return {
        "act_bits": config.activation_quant_bits,
        "weight_bits": config.weight_quant_bits,
        "act_symmetric": config.activation_quant_symmetric,
    }


def rotary_dim(head_dim: int, partial_rotary_factor: float) -> int:
    """The number of head channels RoPE rotates (the rest pass through)."""
    return int(head_dim * partial_rotary_factor)


def nemotron_inv_freq(
    head_dim: int, rope_theta: float, partial_rotary_factor: float
) -> torch.Tensor:
    """RoPE inverse frequencies over the *rotary* width.

    Plain RoPE (no wavelength rescaling -- Domyn Small extends context with YaRN at
    inference time, not with a baked-in `rope_scaling`), but the exponent's
    denominator is the rotary width `int(head_dim * partial_rotary_factor)`, not
    `head_dim`. Getting that wrong silently changes every frequency.
    """
    dim = rotary_dim(head_dim, partial_rotary_factor)
    return 1.0 / (rope_theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))


class DomynRotaryEmbedding(nn.Module):
    """RoPE cos/sin tables over the rotary width (partial rotary embeddings)."""

    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        theta: float,
        partial_rotary_factor: float,
    ) -> None:
        super().__init__()
        inv_freq = nemotron_inv_freq(head_dim, theta, partial_rotary_factor)
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.max_position_embeddings = max_position_embeddings

    @torch.no_grad()
    def forward(self, position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # position_ids: [batch, seq_len]
        inv_freq = self.inv_freq[None, :, None].float()  # [1, rot_dim/2, 1]
        position_ids = position_ids[:, None, :].float()  # [batch, 1, seq_len]
        freqs = (inv_freq @ position_ids).transpose(1, 2)  # [batch, seq_len, rot_dim/2]
        emb = torch.cat((freqs, freqs), dim=-1)  # [batch, seq_len, rot_dim]
        return emb.cos(), emb.sin()


def smoothquant_groups(module: nn.Module) -> list[list["QuantLinear"]]:
    """Groups of QuantLinear that share an input tensor (Nemotron layout).

    q/k/v share the input_layernorm output. Unlike the SwiGLU models there is no
    {gate, up} pair: the squared-ReLU MLP has a single up_proj, so up_proj and
    down_proj are singletons, as is o_proj.
    """
    groups: list[list[QuantLinear]] = []
    for layer in module.modules():
        if isinstance(layer, DomynAttention):
            groups.append([layer.q_proj, layer.k_proj, layer.v_proj])
            groups.append([layer.o_proj])
        elif isinstance(layer, ReLU2MLP):
            groups.append([layer.up_proj])
            groups.append([layer.down_proj])
    return groups


def quantize_model(
    module: nn.Module,
    calibration,
    *,
    smooth_alpha: float | None = 0.8,
    qk_alpha: float | None = None,
) -> None:
    """Calibrate + enable W8A8 (+ SmoothQuant) for a Domyn Small model.

    QK-smoothing is OFF by default (qk_alpha=None): like Llama, Nemotron has no
    QK-Norm -- Q/K go straight from projection to (partial) RoPE -- so per-head-dim
    QK rescaling is not used. Attention is still W8A8-quantized (per-tensor Q/K/V).
    """
    _quantize_model(
        module,
        calibration,
        groups=smoothquant_groups(module),
        smooth_alpha=smooth_alpha,
        qk_alpha=qk_alpha,
    )


@dataclass
class DomynConfig:
    """Hyperparameters for a Domyn Small model. Defaults match Domyn-Small-v1.0."""

    vocab_size: int = 256000
    hidden_size: int = 4096
    intermediate_size: int = 16384
    num_hidden_layers: int = 40
    num_attention_heads: int = 48
    num_key_value_heads: int = 8
    head_dim: int = 128
    max_position_embeddings: int = 32768
    norm_eps: float = 1e-5
    rope_theta: float = 500000.0
    #: Fraction of each head's channels that RoPE rotates (the rest pass through).
    partial_rotary_factor: float = 0.5
    tie_word_embeddings: bool = False
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

    @property
    def rotary_dim(self) -> int:
        return rotary_dim(self.head_dim, self.partial_rotary_factor)

    @classmethod
    def from_hf_dict(cls, cfg: dict) -> "DomynConfig":
        """Builds a DomynConfig from a HuggingFace config.json dict.

        Rejects the Nemotron variants this reference does not model rather than
        loading a silently-wrong graph: a different activation would still fill
        ReLU2MLP, and projection biases would be dropped outright (QuantLinear is
        bias-free), both producing a model that loads cleanly and predicts garbage.
        """
        act = cfg.get("hidden_act", "relu2")
        if act != "relu2":
            raise ValueError(
                f"DomynForCausalLM models the squared-ReLU MLP; got hidden_act={act!r}"
            )
        for key in ("attention_bias", "mlp_bias"):
            if cfg.get(key, False):
                raise ValueError(
                    f"{key}=True is unsupported: QuantLinear is bias-free, so the "
                    "checkpoint's biases would be silently discarded"
                )
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
            # Nemotron spells the norm epsilon "norm_eps" (Llama: "rms_norm_eps").
            norm_eps=cfg.get("norm_eps", cfg.get("rms_norm_eps", 1e-5)),
            rope_theta=cfg["rope_theta"],
            partial_rotary_factor=cfg.get("partial_rotary_factor", 1.0),
            tie_word_embeddings=cfg.get("tie_word_embeddings", False),
        )


class DomynAttention(nn.Module):
    """Grouped-query attention with partial RoPE (no QK-Norm, no projection bias)."""

    def __init__(self, config: DomynConfig) -> None:
        super().__init__()
        self.config = config
        self.head_dim = config.head_dim
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.num_kv_groups = config.num_key_value_groups

        qkw = _quant_linear_kwargs(config)
        # NB: num_heads * head_dim (6144) != hidden_size (4096) for Domyn Small.
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

        q, k = apply_partial_rotary_pos_emb(q, k, cos, sin)

        # Expand KV heads to match the number of query heads (GQA).
        k = repeat_kv(k, self.num_kv_groups)
        v = repeat_kv(v, self.num_kv_groups)

        attn_output = self.sdpa(q, k, v, attention_mask)  # [batch, heads, seq, head_dim]
        attn_output = attn_output.transpose(1, 2).reshape(batch, seq_len, -1)
        return self.o_proj(attn_output)


class DomynDecoderLayer(nn.Module):
    """A single transformer decoder block (pre-norm, LayerNorm1P, squared-ReLU MLP)."""

    def __init__(self, config: DomynConfig) -> None:
        super().__init__()
        self.self_attn = DomynAttention(config)
        self.mlp = ReLU2MLP(
            config.hidden_size, config.intermediate_size, **_quant_linear_kwargs(config)
        )
        self.input_layernorm = LayerNorm1P(config.hidden_size, eps=config.norm_eps)
        self.post_attention_layernorm = LayerNorm1P(config.hidden_size, eps=config.norm_eps)

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


class DomynModel(nn.Module):
    """The Domyn Small transformer trunk (embeddings + decoder layers + final norm)."""

    def __init__(self, config: DomynConfig) -> None:
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [DomynDecoderLayer(config) for _ in range(config.num_hidden_layers)]
        )
        self.norm = LayerNorm1P(config.hidden_size, eps=config.norm_eps)
        self.rotary_emb = DomynRotaryEmbedding(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            config.partial_rotary_factor,
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

        # The tables are built in float32 for precision, then cast back to the
        # activation dtype (as HF's NemotronRotaryEmbedding does). Without the cast a
        # bf16 run silently promotes Q/K to float32 while V stays bf16, and the
        # attention matmul fails on mismatched dtypes.
        cos, sin = (t.to(hidden_states.dtype) for t in self.rotary_emb(position_ids))
        attention_mask = self._causal_mask(seq_len, hidden_states.dtype, hidden_states.device)

        for layer in self.layers:
            hidden_states = layer(hidden_states, cos, sin, attention_mask)

        return self.norm(hidden_states)


class DomynForCausalLM(nn.Module):
    """Domyn Small with a language-modeling head."""

    def __init__(self, config: DomynConfig) -> None:
        super().__init__()
        self.config = config
        self.model = DomynModel(config)
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
        repo_id: str = "domyn/Domyn-Small-v1.0",
        *,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        **config_overrides,
    ) -> "DomynForCausalLM":
        """Loads a Domyn Small model with weights from a HuggingFace checkpoint.

        HF parameter names match this module's names exactly (including the
        LayerNorm1P `weight`/`bias` pairs). config_overrides are applied to the
        derived config (e.g. quantize_attention=True).

        The checkpoint is STREAMED shard by shard into the already-allocated model
        (see checkpoint.stream_hf_safetensors_into). Merging it into one state dict
        first and calling load_state_dict would hold the whole checkpoint AND the
        model at once: ~79 GB at 9.82B parameters in float32, which does not fit a
        62 GB host. Streaming caps the peak at the model plus one shard (~43 GB).
        """
        import dataclasses
        import json

        from ...checkpoint import repo_file

        with open(repo_file(repo_id, "config.json")) as f:
            config = DomynConfig.from_hf_dict(json.load(f))
        if config_overrides:
            config = dataclasses.replace(config, **config_overrides)

        model = cls(config).to(device=device, dtype=dtype)
        missing, unexpected = stream_hf_safetensors_into(model, repo_id, dtype=dtype)

        # ActivationObserver's persistent stats are absent from the checkpoint, and a
        # tied checkpoint would omit lm_head.weight. (Non-persistent buffers -- the RoPE
        # inverse frequencies, the SmoothQuant scales, the RangeObservers -- never reach
        # the missing list; see stream_hf_safetensors_into.)
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


class DomynArchitecture(LLMArchitecture):
    """LLMArchitecture for Domyn Small: load + quantize the torch reference and export it."""

    name = "domyn"

    def load_pretrained(self, repo_id, **kwargs):
        return DomynForCausalLM.from_pretrained(repo_id, **kwargs)

    def quantize(self, model, calibration, *, smooth_alpha=0.8, qk_alpha=None):
        # qk_alpha is always forced to None: no QK-Norm, so QK-smoothing is not
        # applicable. Linear SmoothQuant (smooth_alpha) is unaffected.
        quantize_model(model, calibration, smooth_alpha=smooth_alpha, qk_alpha=None)

    def record_lut_ranges(self, model, calibration):
        record_lut_ranges(model, calibration)

    def num_layers(self, model):
        return model.config.num_hidden_layers

    def config_to_dict(self, model_config):
        return asdict(model_config)

    def config_from_dict(self, data):
        return DomynConfig(**data)

    def export_host_io(self, model):
        """Host-side params for the embedding / final-norm / RoPE boundary.

        Two entries carry a shape the shared host path does not yet handle, and a
        Metis implementation for this architecture has to override it:

        * `final_norm_weight` is the LayerNorm1P gain ALREADY offset (`weight + 1`),
          and `final_norm_bias` accompanies it. The final norm is a full LayerNorm,
          so the host must subtract the mean and add the bias --
          `DecoderLayerImplementation.compute_logits`/`compute_logits_seq` implement
          RMSNorm (no mean, no bias) and would be wrong here.
        * `rope_inv_freq` covers only the rotary width (`rotary_dim`, 64 of the 128
          head channels); `rope_rotary_dim` states that width so the host rotates
          the leading channels and passes the rest through.

        `rms_eps` keeps the contract's name (it is the LayerNorm epsilon here).
        """
        config = model.config
        norm = model.model.norm
        host = {
            "embed_weight": model.model.embed_tokens.weight.detach()
            .cpu()
            .to(torch.float16)
            .numpy(),
            # The 1P offset is folded in here so consumers use the gain directly.
            "final_norm_weight": (norm.weight.detach() + 1.0).cpu().to(torch.float32).numpy(),
            "final_norm_bias": norm.bias.detach().cpu().to(torch.float32).numpy(),
            "rms_eps": np.float32(config.norm_eps),
            # Scalars the decode loop needs to build per-step RoPE / mask inputs.
            "head_dim": np.int64(config.head_dim),
            "num_q_heads": np.int64(config.num_attention_heads),
            "rope_theta": np.float64(config.rope_theta),
            # Precomputed over the ROTARY width, so the host tables match the
            # model's partial rotary factor without re-deriving it.
            "rope_inv_freq": nemotron_inv_freq(
                config.head_dim, config.rope_theta, config.partial_rotary_factor
            )
            .cpu()
            .to(torch.float32)
            .numpy(),
            "rope_rotary_dim": np.int64(config.rotary_dim),
        }
        # Domyn Small is UNTIED, so the host lm_head needs its own weights (~2 GB at fp16);
        # a tied checkpoint would fall back to embed_weight and skip the duplicate.
        if not config.tie_word_embeddings:
            host["lm_head_weight"] = model.lm_head.weight.detach().cpu().to(torch.float16).numpy()
        return host


register_architecture(DomynArchitecture())

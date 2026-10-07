# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""SmolVLM2 architecture: SigLIP vision encoder + pixel-shuffle connector +
SmolLM2 (Llama-architecture) text decoder.

SmolVLM2 = a SigLIP vision tower whose patch features are pixel-shuffled and
projected into the text embedding space, then spliced into the SmolLM2 token
stream at ``<image>`` placeholder positions.

The decoder WEIGHTS are SmolLM2's, lifted out of the VLM checkpoint's
``model.text_model.*`` (see _remap_text_state); no Llama weights are involved.
What is reused is the repo's Llama-architecture *implementation*, because SmolLM2 is
a Llama-shape decoder (RMSNorm, GQA, SwiGLU, plain RoPE, no QK-norm) -- HF says so
itself, resolving text_config to a LlamaConfig with model_type "llama". Every
per-checkpoint value (rope_theta, tie_word_embeddings, vocab_size) is read from that
config rather than inherited from a family default. Only the vision tower, the
connector and the multimodal loader are new.

This module owns the *torch reference* + the ``LLMArchitecture`` contract. The
device kernels live in ``vision_impl`` / ``connector_impl`` and the graph
composition + multicore text impl in ``multicore_impl``.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn

from . import hf as _hf
from ...quant import QuantLinear
from ...quant import quantize_model as _quantize_model
from ...quant import record_lut_ranges
from ..architecture import LLMArchitecture, register_architecture
from ..llama.llama import LlamaConfig, LlamaForCausalLM, llama3_inv_freq
from ..llama.llama import smoothquant_groups as llama_smoothquant_groups
from .vision_reference import (
    SiglipVisionConfig,
    SiglipVisionModel,
)
from .vision_reference import smoothquant_groups as siglip_smoothquant_groups


@dataclass
class SmolVLM2Config:
    """Nested config: a SmolLM2 (Llama-shape) text decoder + a SigLIP vision tower.

    NO defaults, deliberately. Every field varies across the family, and a default set
    would describe a model that does not exist -- a bare instance would have paired
    Llama-3.2-1B's text shape with the tiny unit-test SigLIP at scale_factor 2, when the
    shipped checkpoints are 4/4/3. Both constructors (from_hf_config, config_from_dict)
    have the real values to hand, so requiring them costs nothing.
    """

    text_config: LlamaConfig
    vision_config: SiglipVisionConfig
    scale_factor: int  # pixel-shuffle space-to-depth factor (4 on the 256M/500M, 3 on the 2.2B)
    image_token_id: int  # <image> placeholder id the host splices at

    @classmethod
    def from_hf_config(cls, hf_config) -> "SmolVLM2Config":
        """Build from a loaded HF SmolVLM2 config OBJECT (not the raw json, which omits
        vision fields the HF class fills from its own defaults).

        scale_factor and image_token_id are REQUIRED, not ``getattr`` defaults: the first
        sets image_seq_len (64/tile on the 256M+500M, 81 on the 2.2B) and the second picks
        the splice positions, so a stale value quietly degrades to a text-only answer. The
        argument is resolved, so a miss means transformers renamed the field.
        """
        absent = [k for k in ("scale_factor", "image_token_id") if not hasattr(hf_config, k)]
        if absent:
            raise AttributeError(
                f"resolved HF SmolVLM2 config has no {absent}; refusing to guess (these "
                f"set image_seq_len and the splice positions). Check the transformers "
                f"version -- got {type(hf_config).__name__}."
            )
        return cls(
            text_config=LlamaConfig.from_hf_dict(hf_config.text_config.to_dict()),
            vision_config=_hf.config_from_hf(hf_config.vision_config),
            scale_factor=hf_config.scale_factor,
            image_token_id=hf_config.image_token_id,
        )


def pixel_shuffle(x: torch.Tensor, scale_factor: int) -> torch.Tensor:
    """Space-to-depth by ``scale_factor`` (== HF SmolVLMConnector.pixel_shuffle):
    fold each scale_factor x scale_factor spatial block into the channel dim.
    (bsz, h*w, embed) -> (bsz, h*w/sf^2, embed*sf^2)."""
    bsz, seq, embed_dim = x.size()
    height = width = int(seq**0.5)
    x = x.view(bsz, height, width, embed_dim)
    x = x.view(bsz, height, width // scale_factor, embed_dim * scale_factor)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(bsz, width // scale_factor, height // scale_factor, embed_dim * scale_factor**2)
    x = x.permute(0, 2, 1, 3)
    x = x.reshape(bsz, seq // (scale_factor**2), embed_dim * (scale_factor**2))
    return x


class SmolVLM2Connector(nn.Module):
    """Pixel-shuffle (space-to-depth) + a single bias-free projection Linear into
    the text hidden size (== HF SmolVLMConnector + SmolVLMSimpleMLP)."""

    def __init__(self, config: SmolVLM2Config, **quant_kwargs) -> None:
        super().__init__()
        self.scale_factor = config.scale_factor
        in_dim = config.vision_config.hidden_size * config.scale_factor**2
        out_dim = config.text_config.hidden_size
        # bias-free (SmolVLMSimpleMLP has no bias).
        self.proj = QuantLinear(in_dim, out_dim, **quant_kwargs)

    def forward(self, image_hidden_states: torch.Tensor) -> torch.Tensor:
        x = pixel_shuffle(image_hidden_states, self.scale_factor)
        return self.proj(x)


class SmolVLM2Reference(nn.Module):
    """Float reference: SigLIP vision tower + connector + Llama text decoder.

    ``forward`` runs the multimodal path (vision -> connector -> splice at
    ``image_token_id`` -> text decoder) so calibration/fidelity see the real
    in-distribution activations end to end.
    """

    def __init__(self, config: SmolVLM2Config) -> None:
        super().__init__()
        self.config = config
        self.vision_model = SiglipVisionModel(config.vision_config)
        self.connector = SmolVLM2Connector(config)
        self.text = LlamaForCausalLM(config.text_config)
        # Calibration progress instrumentation: each forward is a full vision+text pipeline
        # over every tile, so a calibration pass takes minutes and needs a heartbeat.
        self._quant_verbose = False
        self._verbose_label = "quant"
        self._fwd_count = 0
        self._quant_t0 = 0.0
        self._quant_t_last = 0.0

    # -- multimodal helpers -------------------------------------------------
    def encode_images(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """pixel_values (num_images, 3, H, W) -> image_embeds
        (num_images, num_tokens_per_image, text_hidden)."""
        feats = torch.cat([self.vision_model(p.unsqueeze(0)) for p in pixel_values], dim=0)
        return self.connector(feats)

    def splice(self, input_ids: torch.Tensor, image_embeds: torch.Tensor) -> torch.Tensor:
        """Embed text tokens and scatter image_embeds into the <image> positions."""
        inputs_embeds = self.text.model.embed_tokens(input_ids)
        mask = input_ids == self.config.image_token_id
        inputs_embeds = inputs_embeds.clone()
        inputs_embeds[mask] = image_embeds.reshape(-1, image_embeds.shape[-1]).to(
            inputs_embeds.dtype
        )
        return inputs_embeds

    def forward(self, batch, position_ids: torch.Tensor | None = None) -> torch.Tensor:
        """Multimodal forward. ``batch`` is either a dict
        ``{"input_ids", "pixel_values"}`` (calibration/fidelity feed a single dict
        via ``module(batch)``) or a bare ``input_ids`` tensor (text-only).

        Running the whole pipeline in one forward lets a single ``quantize_model``
        / ``record_lut_ranges`` pass see every int8 site + LUT range end to end
        (joint PTQ), exactly the way dino/llama calibrate their standalone modules.
        """
        if self._quant_verbose:  # calibration heartbeat; see __init__
            import time

            now = time.perf_counter()
            self._fwd_count += 1
            delta = now - self._quant_t_last if self._fwd_count > 1 else 0.0
            self._quant_t_last = now
            print(
                f"[{self._verbose_label}] forward {self._fwd_count} (vision+text)  "
                f"t=+{now - self._quant_t0:.1f}s  (+{delta:.1f}s)",
                flush=True,
            )
        if isinstance(batch, dict):
            input_ids = batch["input_ids"]
            pixel_values = batch.get("pixel_values")
        else:
            input_ids, pixel_values = batch, None
        if pixel_values is not None:
            image_embeds = self.encode_images(pixel_values)
            inputs_embeds = self.splice(input_ids, image_embeds)
        else:
            inputs_embeds = self.text.model.embed_tokens(input_ids)
        # Run the Llama trunk from inputs_embeds (bypass the embedding lookup).
        return self._text_from_embeds(inputs_embeds, position_ids)

    def _text_from_embeds(
        self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor | None
    ) -> torch.Tensor:
        model = self.text.model
        seq_len = inputs_embeds.shape[1]
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=inputs_embeds.device).unsqueeze(0)
        cos, sin = model.rotary_emb(position_ids)
        attention_mask = model._causal_mask(seq_len, inputs_embeds.dtype, inputs_embeds.device)
        hidden = inputs_embeds
        for layer in model.layers:
            hidden = layer(hidden, cos, sin, attention_mask)
        hidden = model.norm(hidden)
        return self.text.lm_head(hidden)


# --------------------------------------------------------------------------
# HF checkpoint loading
# --------------------------------------------------------------------------
def _remap_text_state(state: dict) -> dict:
    """Rewrite HF SmolVLM2 text-tower keys onto the repo LlamaForCausalLM names.

    HF: ``model.text_model.<...>`` and ``lm_head.weight``.
    Repo: ``model.<...>`` (LlamaModel) and ``lm_head.weight``.
    """
    out = {}
    for k, v in state.items():
        if k.startswith("model.text_model."):
            out["model." + k[len("model.text_model.") :]] = v
        elif k == "lm_head.weight":
            out[k] = v
    return out


def _check_text_state(text: LlamaForCausalLM, state: dict, tied: bool) -> None:
    """Fail loudly on a text state_dict that disagrees with the config.

    ``strict=False`` is required (QuantLinear's observer buffers are not in the checkpoint)
    but it also swallows both ways a wrong ``tie_word_embeddings`` corrupts the model
    silently: tied config + untied checkpoint aliases ``lm_head.weight`` onto the embedding,
    so both load into one tensor and the last write wins; untied config + tied checkpoint
    leaves lm_head at its random init. So assert the invariant directly -- a separate head
    iff untied -- then require every PARAMETER present (buffers are observers, may be absent).
    """
    has_head = "lm_head.weight" in state
    if has_head == tied:
        detail = "clobber the embedding" if has_head else "leave lm_head randomly initialised"
        raise RuntimeError(
            f"checkpoint {'has' if has_head else 'lacks'} a separate lm_head.weight but "
            f"tie_word_embeddings={tied}; loading it as-is would silently {detail}."
        )
    params = dict(text.named_parameters())
    # Tied models share one tensor, which named_parameters() de-duplicates away.
    absent = [n for n in params if n not in state and not (tied and n == "lm_head.weight")]
    if absent:
        raise RuntimeError(f"text checkpoint is missing parameter(s): {sorted(absent)}")
    known = set(params) | set(dict(text.named_buffers()))
    unexpected = [k for k in state if k not in known]
    if unexpected:
        raise RuntimeError(f"text checkpoint has unexpected key(s): {sorted(unexpected)}")


@torch.no_grad()
def load_pretrained(repo_id: str, *, dtype=torch.float32, device="cpu", **config_overrides):
    """Load a HF SmolVLM2 checkpoint into a SmolVLM2Reference.

    The HF model is staged on CPU and copied param-by-param (as in the Llama /
    Qwen3.5 loaders) to avoid a 2x-resident spike.
    """
    # Load the full HF model on CPU, then build the config from its resolved object.
    hf_model = _hf.load_hf_vlm(repo_id, dtype=dtype)
    config = SmolVLM2Config.from_hf_config(hf_model.config)
    # Text-decoder quant knobs (e.g. quantize_attention=True, passed by
    # generate.export_quantized_model) live on the nested LlamaConfig, not on
    # SmolVLM2Config -- route them there. Default quantize_attention ON, matching the
    # multicore convention (the Llama SDPA packer needs the SDPA quant params).
    _text_keys = {"quantize_attention", "sdpa_quant_bits", "activation_quant_symmetric"}
    text_overrides = {
        k: config_overrides.pop(k) for k in list(config_overrides) if k in _text_keys
    }
    text_config = dataclasses.replace(
        config.text_config, quantize_attention=True, **text_overrides
    )
    config = dataclasses.replace(config, text_config=text_config, **config_overrides)

    ref = SmolVLM2Reference(config).to(device=device, dtype=dtype).eval()

    # Text tower -> repo Llama (remap keys, strict=False for observer buffers + tied lm_head).
    text_state = _remap_text_state(hf_model.state_dict())
    _check_text_state(ref.text, text_state, config.text_config.tie_word_embeddings)
    ref.text.load_state_dict(text_state, strict=False)
    if config.text_config.tie_word_embeddings:
        ref.text.lm_head.weight = ref.text.model.embed_tokens.weight

    # Vision tower + connector.
    hf_vision = hf_model.model.vision_model
    _hf.load_full_vision_weights(ref.vision_model, hf_vision)
    ref.connector.proj.weight.copy_(hf_model.model.connector.modality_projection.proj.weight)

    del hf_model
    return ref


class SmolVLM2Architecture(LLMArchitecture):
    """LLMArchitecture for SmolVLM2 (vision + connector + Llama text decoder).

    ``num_layers`` / ``export_host_io`` describe the TEXT decoder (the KV-executor
    graph); the vision encoder is a separate single-pass graph authored by
    ``multicore_impl``. The vision/connector quant params are carried alongside in
    the same QuantizedModel bundle.
    """

    name = "smolvlm2"

    def load_pretrained(self, repo_id, **kwargs):
        return load_pretrained(repo_id, **kwargs)

    def quantize(self, model, calibration, *, smooth_alpha=0.8, qk_alpha=None):
        """Joint W8A8 PTQ of vision + connector + text decoder in one pass.

        ``calibration`` is ``{"input_ids", "pixel_values"}`` dicts, and ``model(dict)`` runs
        the whole multimodal pipeline -- so one ``quantize_model`` call calibrates every int8
        site on its real in-distribution activation and smooths all groups together: the text
        decoder sees spliced multimodal embeddings, the vision tower sees the image path.

        Groups = vision blocks (q/k/v shared, o/fc1/fc2 singletons) + the connector projection
        + the Llama text groups. The patch-embed conv is ungrouped (symmetric per-tensor host
        pixel quant, as in dino). qk_alpha is forced None -- SmolLM2 has no QK-norm.
        """
        groups = (
            siglip_smoothquant_groups(model.vision_model)
            + [[model.connector.proj]]
            + llama_smoothquant_groups(model.text)
        )
        print(
            f"[quant] calibrating {len(calibration)} multimodal samples "
            "(each forward runs the full vision+text pipeline -- slow)...",
            flush=True,
        )
        import time

        model._quant_verbose = True
        model._fwd_count = 0
        model._quant_t0 = model._quant_t_last = time.perf_counter()
        try:
            _quantize_model(
                model, calibration, groups=groups, smooth_alpha=smooth_alpha, qk_alpha=None
            )
        finally:
            model._quant_verbose = False
        print(
            f"[quant] calibration + W8A8 quantization done in "
            f"{time.perf_counter() - model._quant_t0:.1f}s "
            f"({model._fwd_count} forwards)",
            flush=True,
        )

    def record_lut_ranges(self, model, calibration):
        record_lut_ranges(model, calibration)

    def num_layers(self, model):
        return model.config.text_config.num_hidden_layers

    def config_to_dict(self, model_config):
        return {
            "text_config": dataclasses.asdict(model_config.text_config),
            "vision_config": dataclasses.asdict(model_config.vision_config),
            "scale_factor": model_config.scale_factor,
            "image_token_id": model_config.image_token_id,
        }

    def config_from_dict(self, data):
        return SmolVLM2Config(
            text_config=LlamaConfig(**data["text_config"]),
            vision_config=SiglipVisionConfig(**data["vision_config"]),
            scale_factor=data["scale_factor"],
            image_token_id=data["image_token_id"],
        )

    def export_host_io(self, model):
        """Text-decoder host IO (same as Llama) plus the VLM splice scalars."""
        tc = model.config.text_config
        host = {
            "embed_weight": model.text.model.embed_tokens.weight.detach()
            .cpu()
            .to(torch.float16)
            .numpy(),
            "final_norm_weight": model.text.model.norm.weight.detach()
            .cpu()
            .to(torch.float32)
            .numpy(),
            "rms_eps": np.float32(tc.rms_norm_eps),
            "head_dim": np.int64(tc.head_dim),
            "num_q_heads": np.int64(tc.num_attention_heads),
            "rope_theta": np.float64(tc.rope_theta),
            "rope_inv_freq": llama3_inv_freq(tc.head_dim, tc.rope_theta, tc.rope_scaling)
            .cpu()
            .to(torch.float64)
            .numpy(),
            # VLM splice scalars (host embed_tokens override reads these).
            "image_token_id": np.int64(model.config.image_token_id),
            "scale_factor": np.int64(model.config.scale_factor),
        }
        if not tc.tie_word_embeddings:
            host["lm_head_weight"] = (
                model.text.lm_head.weight.detach().cpu().to(torch.float16).numpy()
            )
        return host


register_architecture(SmolVLM2Architecture())

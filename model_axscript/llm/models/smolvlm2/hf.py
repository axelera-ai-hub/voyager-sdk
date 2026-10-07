# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Load real SmolVLM2 SigLIP vision weights from a HuggingFace checkpoint.

Adapted from ``models/dino/hf.py``. Maps one HF ``SmolVLMEncoderLayer`` (a SigLIP
encoder layer) onto the ``SiglipBlock`` reference; the structures line up 1:1
(separate q/k/v/out projections + bias, two LayerNorms with weight+bias, a
gelu-tanh MLP). Relative to DINOv2 there is NO LayerScale and NO CLS token, so
those entries drop from the maps.

In the SmolVLM2 checkpoint the vision tower lives under
``model.vision_model.*`` (encoder layers at
``model.vision_model.encoder.layers.{i}.*``, patch embed at
``model.vision_model.embeddings.*``, final norm at
``model.vision_model.post_layernorm``). This module works from an already
instantiated HF vision module (``hf_vision``), so the caller resolves that
prefix once (see ``smolvlm2.py``).
"""

from __future__ import annotations

import torch

from ...quant import enable_w8a8, quantize_model, record_lut_ranges
from .vision_reference import SiglipBlock, SiglipVisionConfig, smoothquant_groups

# SiglipBlock parameter path -> HF SmolVLMEncoderLayer / SiglipEncoderLayer path.
_WEIGHT_MAP = {
    "ln1.weight": "layer_norm1.weight",
    "ln1.bias": "layer_norm1.bias",
    "q_proj.weight": "self_attn.q_proj.weight",
    "k_proj.weight": "self_attn.k_proj.weight",
    "v_proj.weight": "self_attn.v_proj.weight",
    "o_proj.weight": "self_attn.out_proj.weight",
    "ln2.weight": "layer_norm2.weight",
    "ln2.bias": "layer_norm2.bias",
    "fc1.weight": "mlp.fc1.weight",
    "fc2.weight": "mlp.fc2.weight",
}
# SiglipBlock nn.Parameter attribute -> HF path (the additive biases; no LayerScale).
_ATTR_MAP = {
    "q_bias": "self_attn.q_proj.bias",
    "k_bias": "self_attn.k_proj.bias",
    "v_bias": "self_attn.v_proj.bias",
    "o_bias": "self_attn.out_proj.bias",
    "fc1_bias": "mlp.fc1.bias",
    "fc2_bias": "mlp.fc2.bias",
}


def load_hf_vlm(repo_id: str, *, dtype=None):
    """Load the HF SmolVLM2 model, robust across transformers versions.

    ``AutoModelForImageTextToText`` doesn't always resolve smolvlm, and the top
    level doesn't re-export ``SmolVLMForConditionalGeneration``; the submodule
    always has it. Try the auto class first, then fall back.
    """
    kw = {} if dtype is None else {"dtype": dtype}
    try:
        from transformers import AutoModelForImageTextToText as _Auto

        return _Auto.from_pretrained(repo_id, **kw).eval()
    except Exception:  # noqa: BLE001 - fall back to the concrete class
        import transformers.models.smolvlm as _sv

        return _sv.SmolVLMForConditionalGeneration.from_pretrained(repo_id, **kw).eval()


# HF vision-config attribute -> SiglipVisionConfig field. All REQUIRED: these are read off a
# RESOLVED config object, so a miss means transformers renamed the field, and absorbing that
# into a default would be silent and wrong -- image_size/patch_size set num_patches and hence
# the whole device packing, hidden_act picks the GELU curve, and this class's own defaults are
# the tiny unit-test shape.
_CONFIG_FIELDS = {
    "hidden_size": "hidden_size",
    "num_attention_heads": "num_heads",
    "num_hidden_layers": "num_hidden_layers",
    "intermediate_size": "intermediate_size",
    "layer_norm_eps": "layernorm_eps",
    "hidden_act": "hidden_act",
    "image_size": "image_size",
    "patch_size": "patch_size",
    "num_channels": "num_channels",
}


def config_from_hf(hf_vision_config) -> SiglipVisionConfig:
    """Derive a SiglipVisionConfig from an HF vision config object.

    Raises AttributeError if the object is missing any field in ``_CONFIG_FIELDS``,
    rather than falling back to a default that would silently mis-shape the encoder.
    """
    absent = [k for k in _CONFIG_FIELDS if not hasattr(hf_vision_config, k)]
    if absent:
        raise AttributeError(
            f"resolved HF vision config has no {absent}; refusing to guess (these set the "
            f"patch geometry and activation curve). Check the transformers version -- got "
            f"{type(hf_vision_config).__name__}."
        )
    return SiglipVisionConfig(
        **{field: getattr(hf_vision_config, attr) for attr, field in _CONFIG_FIELDS.items()}
    )


@torch.no_grad()
def load_block_weights(block: SiglipBlock, hf_layer: torch.nn.Module) -> None:
    """Copy one HF SigLIP encoder layer's weights into a SiglipBlock (1:1)."""
    src = dict(hf_layer.named_parameters())
    for dst, name in _WEIGHT_MAP.items():
        block.get_parameter(dst).copy_(src[name])
    for attr, name in _ATTR_MAP.items():
        getattr(block, attr).copy_(src[name])


def full_tile_position_ids(num_patches_per_side: int) -> torch.Tensor:
    """HF SmolVLM position ids for a FULL (all-patches-valid) square tile.

    Replicates ``SmolVLMVisionEmbeddings.forward``'s bucketization EXACTLY: the
    fractional coords are scaled by ``(1 - 1e-6)`` and bucketized with
    ``right=True`` against ``arange(1/nps, 1, 1/nps)``, which for a full tile gives
    per-axis buckets ``[0, 0, 1, 2, ..., nps-2]`` (NOT a plain arange -- the 1e-6
    nudge pushes each coord just below the next boundary). The processor produces
    only full 512x512 tiles, so one baked pos table serves every tile.
    """
    nps = num_patches_per_side
    boundaries = torch.arange(1 / nps, 1.0, 1 / nps)
    frac = torch.arange(nps, dtype=torch.float32) / nps * (1 - 1e-6)
    bucket = torch.bucketize(frac, boundaries, right=True)  # (nps,)
    return (bucket[:, None] * nps + bucket[None, :]).flatten()  # (nps*nps,)


@torch.no_grad()
def load_full_vision_weights(model, hf_vision) -> None:
    """Copy an HF SigLIP vision tower's weights into a vision_reference.SiglipVisionModel:
    each encoder layer, the patch conv (reshaped to the QuantLinear (out, C*P*P)
    layout), the learned absolute position embedding, and the final post-LayerNorm.

    ``hf_vision`` is the HF ``SmolVLMVisionTransformer`` (a.k.a. vision_model).
    """
    for i, block in enumerate(model.encoder.blocks):
        load_block_weights(block, hf_vision.encoder.layers[i])
    conv = hf_vision.embeddings.patch_embedding  # Conv2d(3, hidden, P, P)
    model.patch_embed.proj.weight.copy_(conv.weight.reshape(conv.weight.shape[0], -1))
    model.patch_embed.bias.copy_(conv.bias)
    # Learned absolute position embedding, baked for a full tile: gather the
    # nn.Embedding rows with HF's bucketized full-tile position ids (see
    # full_tile_position_ids -- the ids are NOT arange).
    pos = hf_vision.embeddings.position_embedding.weight  # (num_positions, hidden)
    pos_ids = full_tile_position_ids(model.config.num_patches_per_side)
    model.pos_embed.copy_(pos[pos_ids].unsqueeze(0))
    model.post_layernorm.weight.copy_(hf_vision.post_layernorm.weight)
    model.post_layernorm.bias.copy_(hf_vision.post_layernorm.bias)


@torch.no_grad()
def real_activations(
    hf_vision: torch.nn.Module,
    layer_idx: int,
    *,
    n: int = 8,
    image_size: int = 384,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Capture the real input activations to SigLIP encoder layer *layer_idx* by
    running the HF vision tower on random pixel tensors (for single-block tests)."""
    torch.manual_seed(seed)
    layer = hf_vision.encoder.layers[layer_idx]
    captured: dict = {}
    handle = layer.register_forward_pre_hook(lambda _m, args: captured.__setitem__("x", args[0]))
    acts = []
    try:
        for _ in range(n):
            hf_vision(pixel_values=torch.randn(1, 3, image_size, image_size))
            acts.append(captured["x"].contiguous())
    finally:
        handle.remove()
    return acts


def load_hf_siglip_block(
    hf_vision: torch.nn.Module,
    hf_vision_config,
    *,
    layer_idx: int = 0,
    n_calib: int = 8,
    image_size: int = 384,
    seed: int = 0,
    smooth_alpha: float | None = None,
):
    """Build a calibrated int8 SiglipBlock from a real HF SigLIP layer.

    Returns (block, hf_layer, config): the W8A8 fake-quant block, the HF fp
    reference layer, and the SiglipVisionConfig. Compare via hf_layer's forward
    on the same tokens (SmolVLM/SigLIP encoder layers return a tuple; take [0]).
    """
    config = config_from_hf(hf_vision_config)
    block = SiglipBlock(config)
    load_block_weights(block, hf_vision.encoder.layers[layer_idx])
    block.eval()

    calibration = real_activations(
        hf_vision, layer_idx, n=n_calib, image_size=image_size, seed=seed
    )
    quantize_model(
        block,
        calibration,
        groups=smoothquant_groups(block),
        smooth_alpha=smooth_alpha,
        qk_alpha=None,
    )
    enable_w8a8(block)
    record_lut_ranges(block, calibration)
    return block, hf_vision.encoder.layers[layer_idx], config

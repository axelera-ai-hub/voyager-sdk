# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Load real DINOv2 weights from a HuggingFace checkpoint into a DinoBlock.

Maps one `facebook/dinov2-*` encoder layer onto the DinoBlock reference (the HF
layer structure lines up 1:1 -- separate q/k/v projections, an output dense,
two LayerNorms, two LayerScales, a GELU MLP), derives the block config from the
HF model config, and calibrates the int8 block on real image-derived activations
(the actual hidden-state distribution at that layer). Only the standard MLP
variant is supported (dinov2-small/base/large); the SwiGLU giant is not.

The seq length must be a multiple of 64 (the kernel's PWORD context packing); a
real image gives num_patches+1 tokens (e.g. 257 at 224px), which we truncate to
the requested seq_len. Truncation is harmless for validating one layer: the same
truncated tokens feed both the device block and the HF reference layer, so the
comparison is exact for those tokens.
"""

from __future__ import annotations

import torch

from ...quant import enable_w8a8, quantize_model, record_lut_ranges
from .reference import DinoBlock, DinoBlockConfig, smoothquant_groups

# DinoBlock parameter path -> HF Dinov2Layer parameter path.
_WEIGHT_MAP = {
    "ln1.weight": "norm1.weight",
    "ln1.bias": "norm1.bias",
    "q_proj.weight": "attention.attention.query.weight",
    "k_proj.weight": "attention.attention.key.weight",
    "v_proj.weight": "attention.attention.value.weight",
    "o_proj.weight": "attention.output.dense.weight",
    "ln2.weight": "norm2.weight",
    "ln2.bias": "norm2.bias",
    "fc1.weight": "mlp.fc1.weight",
    "fc2.weight": "mlp.fc2.weight",
}
# DinoBlock attribute (nn.Parameter) -> HF Dinov2Layer parameter path.
_ATTR_MAP = {
    "q_bias": "attention.attention.query.bias",
    "k_bias": "attention.attention.key.bias",
    "v_bias": "attention.attention.value.bias",
    "o_bias": "attention.output.dense.bias",
    "fc1_bias": "mlp.fc1.bias",
    "fc2_bias": "mlp.fc2.bias",
    "ls1": "layer_scale1.lambda1",
    "ls2": "layer_scale2.lambda1",
}


def config_from_hf(hf_config, seq_len: int, image_size: int = 224) -> DinoBlockConfig:
    if getattr(hf_config, "use_swiglu_ffn", False):
        raise NotImplementedError(
            "SwiGLU FFN (dinov2-giant) is not supported; use small/base/large"
        )
    cfg = DinoBlockConfig(
        hidden_size=hf_config.hidden_size,
        num_heads=hf_config.num_attention_heads,
        mlp_ratio=hf_config.mlp_ratio,
        seq_len=seq_len,
        layernorm_eps=hf_config.layer_norm_eps,
        image_size=image_size,
        patch_size=hf_config.patch_size,
        num_channels=hf_config.num_channels,
    )
    cfg.validate()
    return cfg


@torch.no_grad()
def load_block_weights(block: DinoBlock, hf_layer: torch.nn.Module) -> None:
    """Copy one HF Dinov2Layer's weights into a DinoBlock (structures line up 1:1)."""
    src = dict(hf_layer.named_parameters())
    for dst, name in _WEIGHT_MAP.items():
        block.get_parameter(dst).copy_(src[name])
    for attr, name in _ATTR_MAP.items():
        getattr(block, attr).copy_(src[name])


@torch.no_grad()
def real_activations(
    hf_model: torch.nn.Module,
    layer_idx: int,
    seq_len: int,
    *,
    n: int = 8,
    image_size: int = 224,
    seed: int = 0,
) -> list[torch.Tensor]:
    """Capture the real input activations to encoder layer *layer_idx* by running the
    HF model on random pixel tensors; truncate each to seq_len tokens."""
    torch.manual_seed(seed)
    layer = hf_model.encoder.layer[layer_idx]
    captured: dict = {}
    handle = layer.register_forward_pre_hook(lambda _m, args: captured.__setitem__("x", args[0]))
    acts = []
    try:
        for _ in range(n):
            hf_model(pixel_values=torch.randn(1, 3, image_size, image_size))
            acts.append(captured["x"][:, :seq_len, :].contiguous())
    finally:
        handle.remove()
    return acts


@torch.no_grad()
def load_full_weights(model, hf_model, image_size: int) -> None:
    """Copy a HF Dinov2Model's weights into a reference.DinoV2Model: each block, the patch
    conv (reshaped to the QuantLinear's (out, C*P*P) layout), CLS token, the interpolated
    positional embedding for this resolution, and the final LayerNorm."""
    for i, block in enumerate(model.encoder.blocks):
        load_block_weights(block, hf_model.encoder.layer[i])
    conv = hf_model.embeddings.patch_embeddings.projection  # Conv2d(3, hidden, 14, 14)
    model.patch_embed.proj.weight.copy_(conv.weight.reshape(conv.weight.shape[0], -1))
    model.patch_embed.bias.copy_(conv.bias)
    model.cls_token.copy_(hf_model.embeddings.cls_token)
    seq, hidden = model.pos_embed.shape[1], model.pos_embed.shape[2]
    dummy = torch.zeros(1, seq, hidden)
    pos = hf_model.embeddings.interpolate_pos_encoding(dummy, image_size, image_size)
    model.pos_embed.copy_(pos)
    model.final_norm.weight.copy_(hf_model.layernorm.weight)
    model.final_norm.bias.copy_(hf_model.layernorm.bias)


def load_hf_dinov2_block(
    repo_id: str = "facebook/dinov2-small",
    *,
    layer_idx: int = 0,
    seq_len: int = 256,
    n_calib: int = 8,
    image_size: int = 224,
    seed: int = 0,
    smooth_alpha: float | None = None,
):
    """Build a calibrated int8 DinoBlock from a real HF DINOv2 layer.

    smooth_alpha enables SmoothQuant (0.8 is the usual value) to migrate
    activation outliers -- DINOv2's large-norm artifact tokens -- into the
    weights; None keeps plain per-tensor/per-channel W8A8.

    Returns (block, hf_layer, config): the W8A8 fake-quant block, the HF fp
    reference layer (call hf_layer(x)[0] for the fp output on the same tokens),
    and the DinoBlockConfig.
    """
    from transformers import Dinov2Model

    hf_model = Dinov2Model.from_pretrained(repo_id).eval()
    config = config_from_hf(hf_model.config, seq_len)
    block = DinoBlock(config)
    load_block_weights(block, hf_model.encoder.layer[layer_idx])
    block.eval()

    calibration = real_activations(
        hf_model, layer_idx, seq_len, n=n_calib, image_size=image_size, seed=seed
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
    return block, hf_model.encoder.layer[layer_idx], config

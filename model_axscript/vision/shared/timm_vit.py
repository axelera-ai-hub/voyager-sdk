# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""timm weight source for a plain ViT encoder, mapped onto the shared fake-quant reference.

`reference.DinoV2Model` is a pre-norm ViT with a GELU MLP, separate q/k/v projections and a
per-channel LayerScale after each residual branch. A plain ViT (the original architecture, which
every `vit_{small,base,large}_patch16_224.augreg*` checkpoint is an instance of) is the same block
with LayerScale absent, so the reference covers it with `ls1` and `ls2` left at their initial
ones. What differs is the upstream: timm fuses q, k and v into one `qkv` linear, stores a learned
positional embedding for the model's own resolution rather than interpolating one, and carries the
ImageNet classifier in the same checkpoint.

Nothing here is per-checkpoint. Width, depth, head count, MLP ratio and patch size are read off
the loaded model (`config_from_timm`), so one source serves every size of the architecture and a
card supplies only its `repo_id`. The CLASS COUNT is read off the checkpoint's own classifier the
same way, so a fine-tune onto a different label set needs no other change.

`repo_id` names a timm checkpoint OR a local directory holding one, which is what lets a customer
build from weights they trained themselves. timm reads `config.json` out of such a directory (the
file `timm.models.save_for_hf` writes) for the architecture, the eval preprocessing and the class
count, so a local checkpoint answers every question a hub one does.

The state-dict mapping is a pure function (`load_vit_state_dict`) so it is unit-testable without
timm and without the network; `TimmViTSource` only acquires the checkpoint and hands the mapping
its state dict.
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import json
from pathlib import Path

import torch

from .source import Overridable, TransformProcessor
from model_axscript.llm.models.dino.reference import DinoBlockConfig

# Keys whose presence means the checkpoint is NOT the plain-ViT block the reference implements.
# Each would otherwise be dropped in silence and read as a quantization loss on chip.
_UNSUPPORTED = {
    "ls1": "LayerScale (the reference keeps ls1/ls2 at ones for a plain ViT)",
    "ls2": "LayerScale (the reference keeps ls1/ls2 at ones for a plain ViT)",
    "gamma_1": "LayerScale",
    "gamma_2": "LayerScale",
    "q_norm": "QK-norm attention",
    "k_norm": "QK-norm attention",
    "fc1_g": "gated (SwiGLU) MLP",
    "w12": "gated (SwiGLU) MLP",
}


def _reject_unsupported(state_dict: dict) -> None:
    for key in state_dict:
        parts = set(key.split("."))
        for marker, what in _UNSUPPORTED.items():
            if marker in parts:
                raise NotImplementedError(
                    f"checkpoint key {key!r} indicates {what}, which this recipe does not map; "
                    "bring the architecture up as its own reference block instead of letting "
                    "the weight be dropped."
                )


@torch.no_grad()
def load_vit_state_dict(model, state_dict: dict, *, num_layers: int | None = None) -> None:
    """Copy a timm plain-ViT state dict into a `reference.DinoV2Model`.

    Refuses a checkpoint carrying LayerScale, QK-norm or a gated MLP, because the reference block
    has nowhere to put those and a silent drop looks exactly like quantization error later. The
    fused `qkv` weight and bias are split into the reference's three projections; `ls1` / `ls2`
    stay at ones, which is what a plain ViT means.
    """
    _reject_unsupported(state_dict)
    blocks = list(model.encoder.blocks)
    if num_layers is not None and len(blocks) != num_layers:
        raise ValueError(f"reference has {len(blocks)} blocks, checkpoint has {num_layers}")

    conv = state_dict["patch_embed.proj.weight"]  # (hidden, C, P, P)
    model.patch_embed.proj.weight.copy_(conv.reshape(conv.shape[0], -1))
    model.patch_embed.bias.copy_(state_dict["patch_embed.proj.bias"])
    model.cls_token.copy_(state_dict["cls_token"])

    pos = state_dict["pos_embed"]  # (1, num_prefix + num_patches, hidden), this resolution
    if pos.shape[1] != model.pos_embed.shape[1]:
        raise ValueError(
            f"checkpoint pos_embed holds {pos.shape[1]} tokens, the reference expects "
            f"{model.pos_embed.shape[1]}; the run size must match the checkpoint's resolution"
        )
    model.pos_embed.copy_(pos)

    for i, block in enumerate(blocks):
        src = f"blocks.{i}."
        block.ln1.weight.copy_(state_dict[src + "norm1.weight"])
        block.ln1.bias.copy_(state_dict[src + "norm1.bias"])
        # timm fuses the three projections into one (3*hidden, hidden) matrix, in q, k, v order.
        qkv_w = state_dict[src + "attn.qkv.weight"].chunk(3, dim=0)
        qkv_b = state_dict[src + "attn.qkv.bias"].chunk(3, dim=0)
        for proj, bias, w, b in zip(
            (block.q_proj, block.k_proj, block.v_proj),
            (block.q_bias, block.k_bias, block.v_bias),
            qkv_w,
            qkv_b,
        ):
            proj.weight.copy_(w)
            bias.copy_(b)
        block.o_proj.weight.copy_(state_dict[src + "attn.proj.weight"])
        block.o_bias.copy_(state_dict[src + "attn.proj.bias"])
        block.ln2.weight.copy_(state_dict[src + "norm2.weight"])
        block.ln2.bias.copy_(state_dict[src + "norm2.bias"])
        block.fc1.weight.copy_(state_dict[src + "mlp.fc1.weight"])
        block.fc1_bias.copy_(state_dict[src + "mlp.fc1.bias"])
        block.fc2.weight.copy_(state_dict[src + "mlp.fc2.weight"])
        block.fc2_bias.copy_(state_dict[src + "mlp.fc2.bias"])

    if hasattr(model.final_norm, "weight"):  # Identity when the final norm is bypassed
        model.final_norm.weight.copy_(state_dict["norm.weight"])
        model.final_norm.bias.copy_(state_dict["norm.bias"])


def classifier_weights(state_dict: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """The ImageNet head from the same checkpoint: `(weight (classes, hidden), bias (classes,))`."""
    return state_dict["head.weight"].detach().clone(), state_dict["head.bias"].detach().clone()


def config_from_timm(ref, seq_len: int, image_size: int) -> DinoBlockConfig:
    """Derive the per-block shape from a timm `VisionTransformer` instance."""
    attn = ref.blocks[0].attn
    hidden = int(ref.embed_dim)
    cfg = DinoBlockConfig(
        hidden_size=hidden,
        num_heads=int(attn.num_heads),
        mlp_ratio=int(ref.blocks[0].mlp.fc1.out_features // hidden),
        seq_len=seq_len,
        layernorm_eps=float(ref.blocks[0].norm1.eps),
        image_size=image_size,
        patch_size=int(ref.patch_embed.patch_size[0]),
        num_channels=int(ref.patch_embed.proj.weight.shape[1]),
    )
    cfg.validate()
    return cfg


@dataclass(frozen=True)
class TimmViTSource(Overridable):
    """`EncoderSource` backed by a timm `VisionTransformer` checkpoint.

    Same role as `dinov2_vits14.hf.HFSource`, different upstream: the model comes from
    `timm.create_model`, the preprocessing from timm's own resolved eval transform (so the
    checkpoint's normalization is used, not a hand-copied one), and the weights through
    `load_vit_state_dict`.

    `repo_id` is a timm checkpoint id or a local checkpoint directory, the same two things
    `HFSource` accepts, so a card ships pointing at the published weights and a customer points
    it at their own without editing anything else.
    """

    repo_id: str

    @property
    def name(self) -> str:
        return self.repo_id

    def _local_dir(self) -> Path | None:
        """`repo_id` as a local checkpoint directory, or None when it names a hub checkpoint.

        The filesystem is the signal, which is what `transformers.from_pretrained` uses and
        therefore what `HFSource` already inherits.
        """
        path = Path(self.repo_id).expanduser()
        return path if path.is_dir() else None

    def _model_name(self) -> str:
        """What `timm.create_model` is asked for.

        A hub checkpoint is named `<arch>.<tag>` and its hub id prefixes that with `timm/`. A
        local directory is named through timm's `local-dir:` source, which reads `config.json`
        beside the weights for the architecture, the eval preprocessing and the class count, so
        the customer's checkpoint decides all three rather than this card.

        A FILE is refused here, naming what it is missing. It used to fall through to the hub
        branch, and the split below then took the last path segment, so an absolute path lost
        its leading slash and timm answered about a repo id: `Model name
        'tmp/weights/mine.pth' has no source prefix but looks like a Hugging Face Hub repo id
        or a local path`, which names neither the file nor the reason.

        Only a leading `timm/` is stripped, which is what the hub spelling actually prefixes.
        Splitting on the first slash of anything removed a path's first segment as readily as a
        namespace, which is how the message above lost its slash.
        """
        local = self._local_dir()
        if local is not None:
            return f"local-dir:{local}"
        path = Path(self.repo_id).expanduser()
        if path.exists():
            raise ValueError(
                f"{self.repo_id} is a file, not a checkpoint directory. This card reads the "
                f"architecture, the evaluation preprocessing and the class count from a "
                f"`config.json` beside the weights, and a bare state dict names none of the "
                f"three. Point at the directory holding both, which is what "
                f"`timm.models.save_for_hf` writes."
            )
        prefix = "timm/"
        return self.repo_id[len(prefix) :] if self.repo_id.startswith(prefix) else self.repo_id

    @functools.cached_property
    def _pretrained_cfg(self) -> dict:
        """The checkpoint's own data config: input size, interpolation, mean and std.

        Read from the config rather than by instantiating the model: `resolve_data_config`
        accepts the config directly and returns the same dict, while building a ViT-S to throw
        it away costs ~136 ms and a 22M-parameter allocation. That matters because the generic
        run path builds a processor per call (`vision/shared/build.py::_preprocess`), so a loop
        over images paid it per image.

        Cached, because the generic run path builds a processor per call and a LOCAL
        checkpoint would otherwise re-read and re-parse `config.json` off disk once per image;
        the hub branch is a registry lookup and pays nothing either way. The source is frozen,
        so nothing it reads can change between calls.

        A local directory has no entry in timm's registry, so its `config.json` is read
        directly. It is the same file `timm.create_model` reads for a `local-dir:` name, and it
        carries the config under `pretrained_cfg` unless it is one of timm's older single-dict
        files, which are the whole config.
        """
        import timm

        local = self._local_dir()
        if local is None:
            return timm.get_pretrained_cfg(self._model_name()).to_dict()
        config = json.loads((local / "config.json").read_text())
        return config.get("pretrained_cfg", config)

    def load_reference(self):
        import timm

        from model_axscript.shared.checkpoint import hf_access

        with hf_access(self.repo_id):  # a gated repo gets the login hint
            return timm.create_model(self._model_name(), pretrained=True).eval()

    def make_processor(self, image_size: int | None = None):
        """The checkpoint's own eval transform, resolved from its config alone."""
        from timm.data import create_transform, resolve_data_config

        data_config = resolve_data_config(pretrained_cfg=self._pretrained_cfg)
        if image_size is not None:
            data_config["input_size"] = (data_config["input_size"][0], image_size, image_size)
        return TransformProcessor(create_transform(**data_config, is_training=False))

    def seq_len(self, ref, pixel_values) -> int:
        if int(ref.num_prefix_tokens) != 1:
            raise NotImplementedError(
                f"{self.name} carries {ref.num_prefix_tokens} prefix tokens; the reference "
                "embeds exactly one CLS token"
            )
        with torch.no_grad():
            return int(ref.patch_embed(pixel_values).shape[1]) + 1

    def block_config(self, ref, seq_len: int, image_size: int) -> DinoBlockConfig:
        return config_from_timm(ref, seq_len, image_size)

    def num_layers(self, ref) -> int:
        return len(ref.blocks)

    def load_weights(self, model, ref, image_size: int) -> None:
        load_vit_state_dict(model, ref.state_dict(), num_layers=len(ref.blocks))

    def encode(self, ref, pixel_values):
        """fp tokens after the final LayerNorm, matching the device graph's output."""
        return ref.forward_features(pixel_values)

    def export_module(self, ref, image_size: int | None = None):
        """The module a compiler exports, which is NOT the module the fp leg runs.

        timm fuses q, k and v into one projection and the block patterns match HuggingFace's
        separate three, so the timm module offloads nothing. Same weights, same numbers, one
        layout the compiler recognises: `test_timm_vit.py` pins the two modules agreeing.
        """
        return HFViTFeatures(hf_vit_from_timm(ref, image_size))

    def classifier(self, ref) -> tuple[torch.Tensor, torch.Tensor]:
        """The checkpoint's own ImageNet classifier, for the host head."""
        return classifier_weights(ref.state_dict())

    def host_params(self, ref) -> dict:
        """The classifier travels in the artifact, so a run needs no upstream checkpoint."""
        from .imagenet_head import ImageNetHead

        return ImageNetHead.from_reference(self, ref).as_host_params()


# The plain-ViT block axten offloads (`axten/models/vit/block.py::ViTBlockOffload`) matches a
# HuggingFace `ViTLayer`, which projects q, k and v SEPARATELY. timm fuses the three into one
# `qkv` matrix, so a timm block exports to a different subgraph and matches nothing: the whole
# encoder stays on the host. The weights are the same numbers in a different layout, so the fix
# is a layout change and not a different checkpoint.
_HF_BLOCK_KEYS = {
    "norm1": "layernorm_before",
    "norm2": "layernorm_after",
    "attn.proj": "attention.output.dense",
    "mlp.fc1": "intermediate.dense",
    "mlp.fc2": "output.dense",
}


def hf_vit_state_dict(state_dict: dict, num_layers: int) -> dict:
    """A timm plain-ViT state dict, rewritten under HuggingFace `ViTModel`'s parameter names.

    A pure function of the two arguments, so it is unit-testable without timm, without
    transformers and without the network. The ImageNet classifier (`head.*`) is NOT carried over:
    `ViTModel` has no head, and the card runs its own on the host from the same checkpoint.
    """
    _reject_unsupported(state_dict)
    out = {
        "embeddings.cls_token": state_dict["cls_token"],
        "embeddings.position_embeddings": state_dict["pos_embed"],
        "embeddings.patch_embeddings.projection.weight": state_dict["patch_embed.proj.weight"],
        "embeddings.patch_embeddings.projection.bias": state_dict["patch_embed.proj.bias"],
        "layernorm.weight": state_dict["norm.weight"],
        "layernorm.bias": state_dict["norm.bias"],
    }
    for i in range(num_layers):
        src, dst = f"blocks.{i}.", f"encoder.layer.{i}."
        for what in ("weight", "bias"):
            # q, k, v in that order, which is how timm packs the fused projection.
            chunks = state_dict[f"{src}attn.qkv.{what}"].chunk(3, dim=0)
            for name, chunk in zip(("query", "key", "value"), chunks):
                out[f"{dst}attention.attention.{name}.{what}"] = chunk.contiguous()
            for timm_name, hf_name in _HF_BLOCK_KEYS.items():
                out[f"{dst}{hf_name}.{what}"] = state_dict[f"{src}{timm_name}.{what}"]
    return out


@torch.no_grad()
def hf_vit_from_timm(ref, image_size: int | None = None):
    """The same plain ViT as `ref`, as a HuggingFace `ViTModel` carrying the same weights.

    The block layout is what changes and nothing else: the config is read off `ref`, the weights
    are `ref`'s own, and `load_state_dict` is strict, so a key this mapping missed raises here
    rather than becoming a silently untrained layer that reads as quantization loss on chip.

    `image_size` must be the checkpoint's own: a plain ViT stores a learned positional embedding
    for one resolution rather than interpolating, so a different size has no weights to copy.
    """
    from transformers import ViTConfig, ViTModel

    patch_size = int(ref.patch_embed.patch_size[0])
    native = int(ref.patch_embed.img_size[0])
    if image_size is not None and image_size != native:
        raise ValueError(
            f"{native} is the resolution this checkpoint stores a positional embedding for; "
            f"{image_size} would need one this mapping cannot produce"
        )
    hidden = int(ref.embed_dim)
    config = ViTConfig(
        hidden_size=hidden,
        num_hidden_layers=len(ref.blocks),
        num_attention_heads=int(ref.blocks[0].attn.num_heads),
        intermediate_size=int(ref.blocks[0].mlp.fc1.out_features),
        image_size=native,
        patch_size=patch_size,
        num_channels=int(ref.patch_embed.proj.weight.shape[1]),
        layer_norm_eps=float(ref.blocks[0].norm1.eps),
        attn_implementation="sdpa",
    )
    model = ViTModel(config, add_pooling_layer=False)
    model.load_state_dict(hf_vit_state_dict(ref.state_dict(), config.num_hidden_layers))
    return model.eval()


class HFViTFeatures(torch.nn.Module):
    """`forward(pixels) -> last_hidden_state`, which is what `torch.export` can take.

    A HuggingFace encoder returns a dataclass, and export cannot trace one out of the graph.
    """

    def __init__(self, hf: torch.nn.Module):
        super().__init__()
        self.hf = hf

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.hf(pixel_values).last_hidden_state

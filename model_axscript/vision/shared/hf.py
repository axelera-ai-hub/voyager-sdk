# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""HuggingFace as an `EncoderSource`: the family's default upstream, and only the default.

Every mapping here is the imported implementation's own (`llm/models/dino/hf.py`): the
HF-config to block-config derivation and the attribute-path weight copy are the ones that
implementation was written against, so a card built through it reads its weights the way it
expects. What this file adds is the SEAM. The imported `load_dinov2` hardcodes
`Dinov2Model.from_pretrained`, so a card there cannot bring its own upstream; an
`EncoderSource` can, and this is the HuggingFace one.

A different upstream (timm, a local checkpoint, our own export) is a different
`EncoderSource` with its own mapping. The reference module and the device graph are the same
whichever it is.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .source import Overridable


class DinoBackboneGeometry(Overridable):
    """The three shape questions a DINOv2 backbone answers, wherever the source keeps it.

    Every vision source that wraps a DINOv2 backbone owes the builder the same three answers,
    and only differs in how it reaches the backbone: a bare `Dinov2Model` IS the backbone, a
    depth card holds it under `.backbone`, and Depth Anything holds it under `.hf.backbone`.
    Overriding `backbone_of` alone keeps one definition of what the answers mean, so a change
    to any of them cannot reach one source and miss another.
    """

    def backbone_of(self, ref):
        """The DINOv2 backbone inside `ref`. The default is that `ref` already is one."""
        return ref

    def seq_len(self, ref, pixel_values) -> int:
        with torch.no_grad():
            return int(self.backbone_of(ref).embeddings(pixel_values).shape[1])

    def block_config(self, ref, seq_len: int, image_size: int) -> Any:
        from model_axscript.llm.models.dino.hf import config_from_hf

        return config_from_hf(self.backbone_of(ref).config, seq_len, image_size=image_size)

    def num_layers(self, ref) -> int:
        return int(self.backbone_of(ref).config.num_hidden_layers)


@dataclass(frozen=True)
class HFSource(DinoBackboneGeometry):
    """`EncoderSource` backed by a HuggingFace `Dinov2Model` checkpoint."""

    repo_id: str = "facebook/dinov2-small"

    @property
    def name(self) -> str:
        return self.repo_id

    def load_reference(self):
        from transformers import Dinov2Model

        from model_axscript.shared.checkpoint import hf_access

        with hf_access(self.repo_id):  # gated checkpoints get the login hint
            return Dinov2Model.from_pretrained(self.repo_id).eval()

    def make_processor(self, image_size: int | None = None):
        from .calibration import make_processor

        return make_processor(self.repo_id, image_size)

    def load_weights(self, model, ref, image_size: int) -> None:
        from model_axscript.llm.models.dino.hf import load_full_weights

        load_full_weights(model, ref, image_size)

    def encode(self, ref, pixel_values):
        return ref(pixel_values=pixel_values).last_hidden_state

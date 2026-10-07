# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Depth Anything V2 as an `EncoderSource`: a DINOv2 backbone welded to a DPT neck and head.

The other shape a depth card takes. `dinov2_depth.py` composes a backbone and a head this tree
assembles; here HuggingFace ships the pair already joined inside one
`AutoModelForDepthEstimation`, and the source's job is only to make it exportable and to answer
the encoder-shaped questions about the backbone inside it.

Two things this upstream needs that a plain DINOv2 card does not.

**A square input.** The published processor is aspect-ratio preserving, so every image reaches the
model at a different shape, and one compiled graph is ONE shape. `make_processor` therefore turns
that off and pins a square, which is the same resize both legs of the accuracy see. It is not the
upstream's own eval protocol, so an absolute number measured here is not comparable to a published
one; a device-versus-fp gap is unaffected, because the two legs read identical pixels.

**A backbone reached through `.backbone`.** It is a `Dinov2Backbone`, not a `Dinov2Model`: it TAPS
several intermediate layers for the neck to reassemble at four scales. The position embedding is
resized once at load for the same reason every DINOv2 card does it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from .hf import DinoBackboneGeometry


class DepthAnythingDepth(torch.nn.Module):
    """`forward(pixels) -> (1, H, W)` relative depth, as a plain tensor `torch.export` can take."""

    def __init__(self, hf: torch.nn.Module):
        super().__init__()
        self.hf = hf
        self.config = hf.backbone.config  # the ENCODER's shape, which is what the artifact records

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.hf(pixel_values).predicted_depth


@dataclass(frozen=True)
class DepthAnythingSource(DinoBackboneGeometry):
    """`EncoderSource` backed by a `depth-anything/Depth-Anything-V2-*-hf` checkpoint."""

    repo_id: str = "depth-anything/Depth-Anything-V2-Small-hf"
    image_size: int = 224

    @property
    def name(self) -> str:
        return self.repo_id

    def load_reference(self):
        from transformers import AutoModelForDepthEstimation

        from model_axscript.shared.checkpoint import hf_access

        from .axten_encoder import _retarget_image_size

        with hf_access(self.repo_id):  # a gated repo gets the login hint
            hf = AutoModelForDepthEstimation.from_pretrained(self.repo_id).eval()
        _retarget_image_size(hf.backbone, self.image_size)
        return DepthAnythingDepth(hf).eval()

    def make_processor(self, image_size: int | None = None):
        """The upstream's own processor, pinned to ONE square shape.

        `keep_aspect_ratio` off is the load-bearing setting: left on, each image resizes to its own
        shape and there is no single graph to compile, calibrate and score against.
        """
        from transformers import AutoImageProcessor

        size = image_size or self.image_size
        processor = AutoImageProcessor.from_pretrained(self.repo_id)
        processor.size = {"height": size, "width": size}
        processor.keep_aspect_ratio = False
        processor.do_pad = False
        return processor

    def backbone_of(self, ref):
        return ref.hf.backbone

    def load_weights(self, model, ref, image_size: int) -> None:
        raise NotImplementedError(
            "this card has no in-tree device recipe to copy weights into: the DPT neck and head "
            "are not a ViT graph, so the card builds through axten "
            "(implementation='depth-anything-axten') and the compiler reads the weights itself."
        )

    def encode(self, ref, pixel_values):
        """The depth map, which is this card's output on both legs."""
        return ref(pixel_values)

    def export_module(self, ref, image_size: int | None = None):
        """Already a plain-tensor module retargeted at load, so the compiler exports it as it is."""
        return ref

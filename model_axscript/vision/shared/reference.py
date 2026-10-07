# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""The fp32 counterpart of a loaded device, presented with the device's own interface.

A card's accuracy is a PAIR: the device leg beside the fp32 leg, over the same images and
through the same metric, because the claim is the distance between them. The two legs differ
only in what runs the encoder, so the reference is given the interface a loaded device has and
every host stage above it (the checkpoint's classifier, a MixVPR head, a baked segmentation
head) is reused rather than written a second time. A second implementation of the host half
would make the gap a measurement of the host code as much as of quantization.

So a card's `load_reference` composes exactly what its `load_device` composes, swapping this in
for the device backbone, and `infer` is the same call on both.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


def host_device(requested: str = "") -> str:
    """The host device the reference leg runs on: the GPU when there is one, else the CPU.

    A reference leg is fp32 on the HOST, and on a machine with a GPU it belongs there: it is
    the expensive half of an accuracy (thousands of images through torch), and putting it on
    the CPU next to a GPU turns a minute into twenty. `requested` is the card's own
    `cal_device`, honoured when it names one so a machine can be pinned.
    """
    if requested:
        return requested
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class ReferenceEncoder:
    """`encode(pixel_values)` from the upstream checkpoint, in the device's output shape.

    `seg_head` is for a card whose device graph BAKES its head: the device's `encode` then
    returns per-token class logits rather than features, so the reference applies the same
    linear to the same rows to return the same thing. Token 0 stays in place on both legs,
    and whoever reads the output drops it, so the two sides cannot disagree about which rows
    are patches.

    The checkpoint moves to `device` once, not per image, and the result comes back as numpy
    on the host, so a caller sees the same thing whichever device ran it.

    `grid` and `hidden` complete that shape contract: a host head reads them off the device
    backbone to reshape tokens, so a reference standing in for one has to answer the same two
    questions or the head fails on the fp leg alone. `image_size` is the card's, not the
    checkpoint's: DINOv2-small declares 518 and a card running at 224 has a 16x16 grid, not
    37x37, so deriving it from the checkpoint would reshape the fp tokens wrongly.
    """

    def __init__(
        self, source, reference, *, seg_head=None, device: str = "", image_size: int = 0
    ) -> None:
        self.source = source
        self.seg_head = seg_head
        self.device = host_device(device)
        self.reference = reference.to(self.device) if hasattr(reference, "to") else reference
        self._image_size = image_size

    @property
    def hidden(self) -> int:
        """Feature width, under the name the device backbone uses for it."""
        return int(self.reference.config.hidden_size)

    @property
    def grid(self) -> int:
        """Patch-grid side at the size THIS card runs."""
        if not self._image_size:
            raise ValueError(
                "this ReferenceEncoder was built with no image_size, so it cannot report the "
                "patch grid a host head needs. Pass image_size=config.image_size where it is "
                "constructed; the checkpoint's own default is a different size."
            )
        return int(self._image_size // self.reference.config.patch_size)

    def encode(self, pixel_values) -> np.ndarray:
        import numpy as np

        if hasattr(pixel_values, "to"):
            pixel_values = pixel_values.to(self.device)
        tokens = self.source.encode(self.reference, pixel_values)
        if hasattr(tokens, 'detach'):
            tokens = tokens.detach().float().cpu().numpy()
        tokens = np.asarray(tokens, dtype=np.float32)
        if self.seg_head is None:
            return tokens
        from .seg_head import seg_logits

        return seg_logits(tokens[0], self.seg_head)[None]

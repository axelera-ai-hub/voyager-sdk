# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Upstream weight source for a vision encoder -- the seam that keeps HuggingFace ONE
source, not a hard requirement.

The generic build / run / eval flow never imports `transformers` directly: it acquires
the fp reference model, its config, its preprocessing, its per-resolution weights, and
its features through the configured `EncoderSource`. HuggingFace is one implementation
(`dinov2_vits14.hf.HFSource`); a customer brings their own upstream -- torchvision, a
local checkpoint, or our export + SoT flow -- by implementing this Protocol. The
on-chip recipe (the torch fake-quant `reference.DinoV2Model` + the ALTO graph) is
unchanged whatever the source; only where the fp weights come from differs.

Return types are left as `Any` here so this contract stays model-agnostic: a concrete
source returns the model's own `DinoBlockConfig` / reference module.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Protocol, runtime_checkable


class SourceOverrideError(Exception):
    """An override this source cannot take, carrying the reason a reader can act on.

    Raised by the source rather than decided by the caller, so the command line asks the
    object what it accepts instead of testing it for a field name. The front door turns this
    into a refusal naming the flag; nothing else catches it.
    """


class Overridable:
    """How the two customer overrides reach a source's own fields.

    `--weights` and `--head` are written through these, so a source that names its weights by
    something other than `repo_id`, or that composes a head, says so ONCE here instead of the
    front door growing a branch per source. That is the same reason `EncoderSource` exists at
    all: the door knows the protocol, never the fields behind it.
    """

    def with_weights(self, spec: str) -> "EncoderSource":
        """This source, pointed at the caller's weights.

        The default is "my weights are `repo_id`", which every upstream that resolves a hub id
        or a checkpoint directory satisfies. A source whose weights are, say, a bare `.pt`
        file overrides this and repoints its own field.
        """
        return self._repointed("repo_id", spec, flag="--weights", method="with_weights")

    def with_head(self, spec: str) -> "EncoderSource":
        """This source, composing the caller's output head.

        The default is identity, because for most cards the head is named on the CONFIG and
        baked by the recipe: the source supplies a backbone and knows nothing about a head. A
        source that ASSEMBLES backbone and head into one reference module (a depth card)
        overrides this so both copies of the head move together.
        """
        return self

    def _repointed(self, field: str, spec: str, *, flag: str, method: str) -> "EncoderSource":
        names = (
            {f.name for f in dataclasses.fields(self)} if dataclasses.is_dataclass(self) else set()
        )
        if field not in names:
            raise SourceOverrideError(
                f"{type(self).__name__} does not name what {flag} replaces by `{field}`. Give "
                f"the card a source that does, or override `{method}` on this one (see "
                f"model_axscript/vision/shared/source.py)."
            )
        return dataclasses.replace(self, **{field: spec})


@runtime_checkable
class EncoderSource(Protocol):
    """Contract an upstream must satisfy to feed a vision-encoder bringup.

    `ref` is whatever object `load_reference` returns (the upstream's fp model); every
    other method takes it back so a source stays stateless.

    The two `with_*` methods are what a customer's `--weights` and `--head` reach. `Overridable`
    supplies both, so a source inherits them rather than implementing them, and overrides one
    only where its own answer differs.
    """

    name: str
    """Identity for meta.json / artifact naming (the HF repo id for `HFSource`)."""

    def with_weights(self, spec: str) -> "EncoderSource":
        """This source, pointed at the caller's weights, or `SourceOverrideError`."""

    def with_head(self, spec: str) -> "EncoderSource":
        """This source, composing the caller's head; identity where the config owns it."""

    def load_reference(self) -> Any:
        """The upstream fp reference encoder, in eval mode."""

    def make_processor(self, image_size: int | None = None) -> Any:
        """The preprocessing transform (resize/crop/normalize); `image_size` overrides
        the native crop size."""

    def seq_len(self, ref: Any, pixel_values: Any) -> int:
        """Real token count the embedding produces for one `(1, 3, H, W)` batch."""

    def block_config(self, ref: Any, seq_len: int, image_size: int) -> Any:
        """The per-block `DinoBlockConfig` (also carries num_channels / patch_size)."""

    def num_layers(self, ref: Any) -> int:
        """Number of encoder blocks."""

    def load_weights(self, model: Any, ref: Any, image_size: int) -> None:
        """Copy the upstream fp weights into the torch fake-quant `reference.DinoV2Model`."""

    def encode(self, ref: Any, pixel_values: Any) -> Any:
        """The fp last-hidden-state features for `pixel_values` (for fidelity / kNN)."""


@dataclass(frozen=True)
class TransformProcessor:
    """A plain image transform, presented as the image processor the generic flow calls.

    `preprocess_image` and the eval harnesses call
    `processor(images=..., return_tensors="pt").pixel_values`, which is the HuggingFace
    image-processor surface. An upstream outside HuggingFace (timm, torchvision) hands out a
    bare callable instead, one PIL image to one `(C, H, W)` tensor. Wrapping it here keeps the
    adapter in one place: a source for such an upstream returns
    `TransformProcessor(transform)` from `make_processor` and the rest of the flow sees no
    difference.

    The transform owns the resize, crop and normalization, so the upstream's own eval
    preprocessing is what reaches both the calibration set and the run path.
    """

    transform: Any

    def __call__(self, images: Any, return_tensors: str = "pt") -> SimpleNamespace:
        """One image or a list of images -> `.pixel_values` of shape `(N, C, H, W)`."""
        import torch  # deferred: the runtime tox env installs no torch

        if return_tensors != "pt":
            raise ValueError(
                f"TransformProcessor returns torch tensors; got return_tensors={return_tensors!r}"
            )
        batch = list(images) if isinstance(images, (list, tuple)) else [images]
        return SimpleNamespace(pixel_values=torch.stack([self.transform(img) for img in batch]))

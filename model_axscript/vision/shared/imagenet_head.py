# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""The host-side ImageNet classifier: one linear projection of the CLS token.

Shared by every plain-ViT ImageNet card, whatever its width: the encoder runs on the AIPU
and emits tokens, and the classifier is a single (classes, hidden) matrix applied to token 0.
At 1000 x 1024, the widest of the three, that is ~1 MFLOP for one image, so it stays a host
stage: putting it on device would cost a graph tail and a second quantization boundary for a
projection the host runs in microseconds. Its weights travel in the artifact's `host_params`,
so a run needs no upstream checkpoint.

The class names come from timm's bundled ImageNet metadata, which is already a dependency of
this card's upstream. A missing timm degrades to the class index rather than failing a run, and
so does a head that is not an ImageNet-1k one: a fine-tune onto a different label set predicts
the customer's classes, which no file here carries.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

_HEAD_WEIGHT, _HEAD_BIAS = "head_weight", "head_bias"
_IMAGENET_CLASSES = 1000


def class_label(index: int) -> str:
    """The human-readable ImageNet-1k description for a class index."""
    try:
        from timm.data import ImageNetInfo

        return ImageNetInfo().index_to_description(int(index))
    except Exception:  # noqa: BLE001 - a label lookup must not fail a run
        return f"class {int(index)}"


def softmax(logits: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last axis."""
    shifted = np.asarray(logits, dtype=np.float32) - np.max(logits, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


@dataclass(frozen=True)
class ImageNetHead:
    """`logits = cls @ weight.T + bias`, run on the host in float32."""

    weight: np.ndarray  # (classes, hidden)
    bias: np.ndarray  # (classes,)

    def __post_init__(self) -> None:
        if self.weight.ndim != 2 or self.bias.shape != (self.weight.shape[0],):
            raise ValueError(
                f"head shapes disagree: weight {self.weight.shape}, bias {self.bias.shape}"
            )

    @property
    def num_classes(self) -> int:
        return int(self.weight.shape[0])

    def logits(self, tokens: np.ndarray) -> np.ndarray:
        """(1, seq, hidden) encoder tokens (or a bare (hidden,) CLS) -> (classes,) logits."""
        cls = np.asarray(tokens, dtype=np.float32)
        if cls.ndim == 3:
            cls = cls[0, 0]
        elif cls.ndim == 2:
            cls = cls[0]
        if cls.shape != (self.weight.shape[1],):
            raise ValueError(
                f"CLS feature {cls.shape} does not match head input {self.weight.shape[1]}"
            )
        return cls @ self.weight.T + self.bias

    @property
    def is_imagenet(self) -> bool:
        """Is this the ImageNet-1k classifier, whose indices have published names?

        A head of another width came from a fine-tune onto the customer's own label set, which
        no file in this tree carries, so anything this tree says about those indices is
        guesswork. Both the labels and the line that introduces them read this.
        """
        return self.num_classes == _IMAGENET_CLASSES

    def label(self, index: int) -> str:
        """The name of class `index`: ImageNet's, or the index itself for another label set."""
        return class_label(index) if self.is_imagenet else f"class {index}"

    def top_k(self, tokens: np.ndarray, k: int = 5) -> list[tuple[int, str, float]]:
        """The k highest-scoring classes as (index, label, probability), best first."""
        logits = self.logits(tokens)
        probs = softmax(logits)
        order = np.argsort(-logits)[: min(k, self.num_classes)]
        return [(int(i), self.label(int(i)), float(probs[i])) for i in order]

    def as_host_params(self) -> dict[str, np.ndarray]:
        """The arrays this head contributes to the artifact's `host_params`."""
        return {
            _HEAD_WEIGHT: self.weight.astype(np.float32),
            _HEAD_BIAS: self.bias.astype(np.float32),
        }

    @classmethod
    def from_host_params(cls, artifact_dir) -> ImageNetHead:
        """Load the head an artifact carries (written at build time by the source's hook)."""
        path = Path(artifact_dir) / "host_params.npz"
        with np.load(path) as params:
            if _HEAD_WEIGHT not in params:
                raise KeyError(
                    f"{path} carries no {_HEAD_WEIGHT}; the artifact was built without the "
                    "classifier head, so it cannot be run as a classifier"
                )
            return cls(params[_HEAD_WEIGHT], params[_HEAD_BIAS])

    @classmethod
    def from_reference(cls, source, ref) -> ImageNetHead:
        """The head from the upstream checkpoint (the build path and the eval harness)."""
        weight, bias = source.classifier(ref)
        return cls(weight.float().numpy(), bias.float().numpy())

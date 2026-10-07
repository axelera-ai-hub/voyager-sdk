# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""The segmentation head a card bakes into its device graph: Meta's, or the customer's own.

The published head is a BatchNorm over the backbone's embedding dimension followed by a 1x1
conv onto the 150 ADE20K classes, released per backbone size as a `.pth` on
`dl.fbaipublicfiles.com`. It is NOT on HuggingFace, which is why a segmentation card cannot be
reached through `from_pretrained` at all: the weights are assembled from a backbone plus this
file. A card that names a path instead loads that file and fetches nothing, which is how a
customer runs their own head, trained on their own label set, on the same backbone.

Nothing here hardcodes an embedding dimension or a class count. For a published head the URL
and the cache filename are DERIVED from the size the card declares; either way the dimension
and the class count are READ from the head that arrives, and the dimension is then checked
against the backbone. A `vitb14` head on a `vits14` backbone otherwise reaches the first matmul
and fails as a shape error that names neither the head nor the backbone.

The same fold lives in `llm/models/dino/seg_demo.py`. It is written out again rather than
imported, because a shipped card must not depend on a file named `*_demo.py`; the four lines
are cheap and `test_seg_head.py` pins them against the unfolded form.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import head_weights
from .head_weights import local_file

_BN_EPS = 1e-5  # mmsegmentation BatchNorm2d default, which trained these heads

#: Meta's ADE20K linear head, and the file layout a card of this task can compile. Everything a
#: reader needs about where it comes from and what it must contain is here; `head_weights` owns
#: what to DO with it, so the depth head beside this one differs only in this declaration.
PUBLISHED = head_weights.PublishedHead(
    what="segmentation",
    base="https://dl.fbaipublicfiles.com/dinov2",
    stem="dinov2_{size}",
    filename="{stem}_ade20k_linear_head.pth",
    cache_leaf="seg_heads",
    digests={
        "vits14": "67e10225e0bf1e2c6e8bc9e07020211ab58cbb4aa14efbaa32c52914931c4ade",
        "vitb14": "7c3545c3a6f79ccc130d075dfe66e84f8c32d72d2acd31fc218f8a15cd2c7b23",
        "vitl14": "59137f319d7e83c79f3253ab7bf63041f1f884e54ba865a2c7ef0046655306a0",
    },
    layout=(
        "decode_head.bn.weight",
        "decode_head.bn.bias",
        "decode_head.bn.running_mean",
        "decode_head.bn.running_var",
        "decode_head.conv_seg.weight",
        "decode_head.conv_seg.bias",
    ),
    pointwise="decode_head.conv_seg.weight",
    topology=(
        "This card compiles a LINEAR segmentation head: a BatchNorm over the backbone's "
        "embedding dimension, then a 1x1 convolution onto your classes."
    ),
)

#: Backbone sizes Meta publishes an ADE20K linear head for, verified against the URLs above.
SIZES = PUBLISHED.sizes

# The ADE20K label set in mmsegmentation ordering, which is the order the head's conv_seg
# outputs are in, so index i of a logit row is ADE20K_CLASSES[i].
ADE20K_CLASSES = [
    "wall",
    "building",
    "sky",
    "floor",
    "tree",
    "ceiling",
    "road",
    "bed ",
    "windowpane",
    "grass",
    "cabinet",
    "sidewalk",
    "person",
    "earth",
    "door",
    "table",
    "mountain",
    "plant",
    "curtain",
    "chair",
    "car",
    "water",
    "painting",
    "sofa",
    "shelf",
    "house",
    "sea",
    "mirror",
    "rug",
    "field",
    "armchair",
    "seat",
    "fence",
    "desk",
    "rock",
    "wardrobe",
    "lamp",
    "bathtub",
    "railing",
    "cushion",
    "base",
    "box",
    "column",
    "signboard",
    "chest of drawers",
    "counter",
    "sand",
    "sink",
    "skyscraper",
    "fireplace",
    "refrigerator",
    "grandstand",
    "path",
    "stairs",
    "runway",
    "case",
    "pool table",
    "pillow",
    "screen door",
    "stairway",
    "river",
    "bridge",
    "bookcase",
    "blind",
    "coffee table",
    "toilet",
    "flower",
    "book",
    "hill",
    "bench",
    "countertop",
    "stove",
    "palm",
    "kitchen island",
    "computer",
    "swivel chair",
    "boat",
    "bar",
    "arcade machine",
    "hovel",
    "bus",
    "towel",
    "light",
    "truck",
    "tower",
    "chandelier",
    "awning",
    "streetlight",
    "booth",
    "television receiver",
    "airplane",
    "dirt track",
    "apparel",
    "pole",
    "land",
    "bannister",
    "escalator",
    "ottoman",
    "bottle",
    "buffet",
    "poster",
    "stage",
    "van",
    "ship",
    "fountain",
    "conveyer belt",
    "canopy",
    "washer",
    "plaything",
    "swimming pool",
    "stool",
    "barrel",
    "basket",
    "waterfall",
    "tent",
    "bag",
    "minibike",
    "cradle",
    "oven",
    "ball",
    "food",
    "step",
    "tank",
    "trade name",
    "microwave",
    "pot",
    "animal",
    "bicycle",
    "lake",
    "dishwasher",
    "screen",
    "blanket",
    "sculpture",
    "hood",
    "sconce",
    "vase",
    "traffic light",
    "tray",
    "ashcan",
    "fan",
    "pier",
    "crt screen",
    "plate",
    "monitor",
    "bulletin board",
    "shower",
    "radiator",
    "glass",
    "clock",
    "flag",
]
ADE20K_NUM_CLASSES = len(ADE20K_CLASSES)


@dataclass(frozen=True)
class SegHead:
    """The head's parameters, plus the shapes they imply, read from the file itself."""

    bn_weight: np.ndarray
    bn_bias: np.ndarray
    bn_running_mean: np.ndarray
    bn_running_var: np.ndarray
    conv_weight: np.ndarray  # (num_classes, dim)
    conv_bias: np.ndarray

    @property
    def dim(self) -> int:
        """Embedding dimension the head expects, i.e. the backbone's hidden size."""
        return int(self.conv_weight.shape[1])

    @property
    def num_classes(self) -> int:
        return int(self.conv_weight.shape[0])


def head_file(head: str, *, cache_dir=None) -> Path:
    """The `.pth` `head` names: your own file, or Meta's ADE20K head for a backbone size."""
    return PUBLISHED.file(head, cache_dir=cache_dir)


def load_seg_head(head: str, *, cache_dir=None) -> SegHead:
    """The linear segmentation head `head` names, as the parameters a card bakes into its graph.

    `head` is a published backbone size or a path to your own head file, and `PUBLISHED`
    resolves which and refuses a file that is not this head. The class count is never declared:
    it is `conv_seg`'s output width, whatever that is.
    """
    state = PUBLISHED.state_dict(head, cache_dir=cache_dir)
    return SegHead(
        bn_weight=state["decode_head.bn.weight"].numpy(),
        bn_bias=state["decode_head.bn.bias"].numpy(),
        bn_running_mean=state["decode_head.bn.running_mean"].numpy(),
        bn_running_var=state["decode_head.bn.running_var"].numpy(),
        conv_weight=state["decode_head.conv_seg.weight"]
        .numpy()
        .reshape(-1, len(state["decode_head.bn.weight"])),
        conv_bias=state["decode_head.conv_seg.bias"].numpy(),
    )


def check_backbone(
    head: SegHead, *, head_spec: str, hidden_size: int, num_classes: int | None = None
):
    """Refuse a head paired with a backbone it was not trained on, naming both sides.

    The class count is checked only against something that declares one: `num_classes` when a
    caller passes it, and otherwise ADE20K's 150 for a PUBLISHED head, where the count is a
    property of the file Meta ships. A customer's head predicts their own label set, so its
    count is read off `conv_seg` and there is nothing to check it against.
    """
    if head.dim != hidden_size:
        raise ValueError(
            f"the segmentation head {head_spec} expects a {head.dim}-wide backbone, and this "
            f"one is {hidden_size} wide. Pair the head with the backbone it was trained on."
        )
    if num_classes is None:
        if local_file(head_spec) is not None:
            return
        num_classes = ADE20K_NUM_CLASSES
    if head.num_classes != num_classes:
        raise ValueError(
            f"the head {head_spec} predicts {head.num_classes} classes, not {num_classes}; "
            f"it is not the ADE20K linear head."
        )


def class_names(head_spec, head: SegHead) -> list[str]:
    """Labels for what `head` predicts: the ADE20K vocabulary, or indices for your own head.

    A customer's head predicts their own label set, which no file in this tree carries, and
    printing ADE20K's first N names beside it is a wrong answer rather than a missing one.
    """
    if local_file(head_spec) is None:
        return ADE20K_CLASSES
    return [f"class {i}" for i in range(head.num_classes)]


def seg_logits(patch_tokens: np.ndarray, head: SegHead, eps: float = _BN_EPS) -> np.ndarray:
    """(N, dim) patch tokens -> (N, num_classes) logits: BatchNorm, then the 1x1 conv."""
    normalized = (patch_tokens - head.bn_running_mean) / np.sqrt(head.bn_running_var + eps)
    return (normalized * head.bn_weight + head.bn_bias) @ head.conv_weight.T + head.conv_bias


def folded_head(head: SegHead, eps: float = _BN_EPS) -> tuple[np.ndarray, np.ndarray]:
    """Fold the BatchNorm into the 1x1 conv -> one (weight, bias) linear, fp32.

    BN(x) = a * x + c, so the pair applied after it becomes W' = conv_weight * a and
    b' = conv_weight @ c + conv_bias. Used to bake the head into the device graph, and it is
    the same linear `seg_logits` applies on the host.
    """
    a = head.bn_weight / np.sqrt(head.bn_running_var + eps)
    c = head.bn_bias - head.bn_running_mean * a
    weight = (head.conv_weight * a[None, :]).astype(np.float32)
    bias = (head.conv_weight @ c + head.conv_bias).astype(np.float32)
    return weight, bias

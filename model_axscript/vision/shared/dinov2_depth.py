# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""DINOv2 plus Meta's NYU linear depth head, as one module the compiler can offload whole.

The counterpart of `seg_head.py` + `ade20k_seg.py` on the depth axis. Meta publishes the head
alongside each backbone as `dinov2_vit*14_nyu_linear_head.pth`; it is reconstructed here without
its mmcv/mmseg framework, the way the ADE20K head already is. A card can name its own head file
in place of a published size, and the bin count then comes from that file's conv.

Despite Meta calling it `BNHead` the head carries NO BatchNorm (it is commented out upstream), so
there is nothing to fold. What it does is:

  * read the LAST encoder block's tokens BEFORE the backbone's final LayerNorm (Meta's config sets
    `out_indices=[11]`, `final_norm=False`), which is why `encode` asks for `output_hidden_states`;
  * broadcast the CLS token across the patch grid and concatenate it on the channel axis, 2C wide;
  * bilinear-upsample the grid 4x (the config's `upsample=4`);
  * a 1x1 conv to `n_bins` depth-bin logits, then Adabins' "linear" normalisation (relu, `+0.1`,
    sum-to-one over bins) and a weighted sum against uniform bin centres in
    `[_MIN_DEPTH, _MAX_DEPTH]`, giving one depth value per position;
  * a bilinear resize back to the input grid.

Every one of those ops has an offload pattern, so the whole card, encoder and head, compiles into
ONE device dispatch through `axten_encoder`. There is no `imported_dino` equivalent: that builder
authors a ViT graph with an optional baked LINEAR head, and this head is not linear.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from . import head_weights
from .hf import DinoBackboneGeometry

#: Meta's NYU linear depth head, declared the way the ADE20K head beside it is. The two differ
#: in this block and in nothing else: `head_weights` owns resolving a size to a URL, a cache
#: path and a digest, and refusing a file that is not this head.
PUBLISHED = head_weights.PublishedHead(
    what="depth",
    base="https://dl.fbaipublicfiles.com/dinov2",
    stem="dinov2_{size}",
    filename="{stem}_nyu_linear_head.pth",
    cache_leaf="depth_heads",
    digests={
        "vits14": "6062f6789bf73ac06458c1e6c6cb067790da09c78004a33245207f0ad5f7e93c",
        "vitb14": "2ce8ccb154437ca5761d16d0609144828ed273afeb73bb776a388650e5e0b339",
        "vitl14": "a853f7dcea0c1e1243cb21b4629b8c61787be9bda51b1f22678aeb8307d82cbf",
    },
    layout=("decode_head.conv_depth.weight", "decode_head.conv_depth.bias"),
    pointwise="decode_head.conv_depth.weight",
    topology=(
        "This card compiles a LINEAR depth head: a single 1x1 convolution onto your depth "
        "bins, over the CLS token concatenated onto every patch. Despite Meta's `BNHead` name "
        "it carries no BatchNorm."
    ),
)

#: Backbone sizes Meta publishes an NYU linear depth head for, verified against the URLs above.
SIZES = PUBLISHED.sizes

# The NYU "linear" config's own values. The bin COUNT is not here: it is read off the weights,
# so a head file that ships a different one is used as it is rather than reshaped to fit.
_MIN_DEPTH, _MAX_DEPTH, _UPSAMPLE = 0.001, 10.0, 4
_ADABINS_EPS = 0.1  # the "+eps" of Adabins' linear normalisation, before the sum-to-one


def _safe_globals() -> list:
    """The numpy data constructors the head file's training metadata unpickles through.

    Named one by one rather than turning `weights_only` off: every entry builds a numpy value and
    nothing here can execute anything, where `weights_only=False` restores the arbitrary-code path
    for a file fetched over the network.

    The scalar constructor is registered under its LEGACY name. numpy 2 moved it to `numpy._core`,
    and torch keys its allowlist off the live object's own module, so the object alone would be
    registered as `numpy._core.multiarray.scalar` and never match the `numpy.core.multiarray.scalar`
    this pickle was written with.
    """
    import numpy as np

    dtypes = [getattr(np.dtypes, name) for name in dir(np.dtypes) if name.endswith("DType")]
    scalar = getattr(np, "_core", np).multiarray.scalar
    return [np.dtype, *dtypes, (scalar, "numpy.core.multiarray.scalar")]


def head_file(head: str, *, cache_dir=None) -> Path:
    """The `.pth` `head` names: your own file, or Meta's NYU linear head for a backbone size."""
    return PUBLISHED.file(head, cache_dir=cache_dir)


def load_depth_head_weights(head: str, *, cache_dir=None) -> tuple[torch.Tensor, torch.Tensor]:
    """The linear depth head `head` names -> its 1x1 conv `(weight, bias)`.

    `head` is a published backbone size or a path to your own head file, and `head_file`
    resolves which. The bin COUNT is never declared: it is the conv's output width, so a head
    trained for a different number of depth bins is used as it is rather than reshaped to fit.

    `weights_only=True` like every other checkpoint this tree loads: the file is a pickle, and
    the trust boundary must not depend on whichever torch default is resolved. The two tensors
    are all this head is; Meta's training metadata rides alongside them and is not read.

    That metadata is why `_safe_globals` exists. The file carries numpy SCALARS (a training epoch,
    a best score) next to the tensors, and the weights-only unpickler refuses the numpy scalar
    constructor by default. Allowlisting exactly that constructor keeps the load restricted to
    data, where `weights_only=False` would restore the arbitrary-code path for a remote file.
    """
    state = PUBLISHED.state_dict(head, cache_dir=cache_dir, safe_globals=_safe_globals)
    return state["decode_head.conv_depth.weight"], state["decode_head.conv_depth.bias"]


def check_backbone(weight: torch.Tensor, *, head_spec: str, hidden_size: int) -> None:
    """Refuse a head paired with a backbone it was not trained on, naming both sides.

    The head's input is TWICE the backbone width, because the CLS token is concatenated onto
    every patch. Reaching the conv with the wrong pairing is a shape error naming neither side.
    """
    expected = 2 * hidden_size
    if int(weight.shape[1]) != expected:
        raise ValueError(
            f"the depth head {head_spec} expects {int(weight.shape[1])} input channels, and a "
            f"{hidden_size}-wide backbone gives it {expected} (the patch tokens plus the "
            f"broadcast CLS). Pair the head with the backbone it was trained on."
        )


class Dinov2LinearDepth(torch.nn.Module):
    """`forward(pixels) -> (1, H, W)` metric depth: a HuggingFace DINOv2 backbone + Meta's head.

    Emits a bare tensor rather than HuggingFace's dataclass, which is what `torch.export` takes
    and what both legs of the accuracy read.
    """

    def __init__(self, backbone: torch.nn.Module, weight: torch.Tensor, bias: torch.Tensor):
        super().__init__()
        self.backbone = backbone
        self.config = backbone.config  # so a caller can ask the composite the backbone's shape
        self.patch_size = int(backbone.config.patch_size)
        n_bins, in_channels = int(weight.shape[0]), int(weight.shape[1])
        self.conv_depth = torch.nn.Conv2d(in_channels, n_bins, kernel_size=1)
        self.conv_depth.weight = torch.nn.Parameter(weight, requires_grad=False)
        self.conv_depth.bias = torch.nn.Parameter(bias, requires_grad=False)
        # Shaped (1, n_bins, 1, 1) so the weighted sum is a plain per-channel constant broadcast,
        # which `MulOffload` matches; a reshaped view of a flat tensor is not a `get_attr`.
        self.register_buffer(
            "bins", torch.linspace(_MIN_DEPTH, _MAX_DEPTH, n_bins).reshape(1, -1, 1, 1)
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        import torch.nn.functional as F

        height, width = pixel_values.shape[-2:]
        # hidden_states[-1] is the last block's output BEFORE the final LayerNorm, which is what
        # Meta's `final_norm=False` reads. last_hidden_state would be the normed one.
        tokens = self.backbone(pixel_values, output_hidden_states=True).hidden_states[-1]
        cls, patches = tokens[:, 0], tokens[:, 1:]
        batch, _, channels = patches.shape
        grid_h, grid_w = height // self.patch_size, width // self.patch_size
        patches = patches.permute(0, 2, 1).reshape(batch, channels, grid_h, grid_w)
        cls = cls[:, :, None, None].expand(batch, channels, grid_h, grid_w)
        features = torch.cat([patches, cls], dim=1)
        features = F.interpolate(
            features,
            size=(grid_h * _UPSAMPLE, grid_w * _UPSAMPLE),
            mode="bilinear",
            align_corners=False,
        )
        logit = torch.relu(self.conv_depth(features)) + _ADABINS_EPS
        logit = logit / logit.sum(dim=1, keepdim=True)
        depth = (logit * self.bins).sum(dim=1, keepdim=True)
        depth = F.interpolate(depth, size=(height, width), mode="bilinear", align_corners=False)
        return depth.squeeze(1)


@dataclass(frozen=True)
class Dinov2DepthSource(DinoBackboneGeometry):
    """`EncoderSource` for a DINOv2 backbone under Meta's NYU linear depth head.

    `load_reference` returns the COMPOSITE, backbone and head, because the head is part of what
    this card is: the device graph runs both and so must the fp leg it is compared against. The
    encoder-shaped questions (`seq_len`, `block_config`, `num_layers`) are answered for the
    backbone inside it, which is what the artifact's geometry records.

    `image_size` is a field rather than an argument because a DINOv2 position embedding is resized
    ONCE at load: doing it per forward pass is the interpolate subgraph the embed pattern cannot
    match, and the whole encoder would fall back to the host.
    """

    repo_id: str
    #: Which depth head to run: a backbone size Meta publishes one for (`SIZES`), or a path to
    #: your own head file. Not named for a size, because it holds either.
    head: str
    image_size: int = 224
    cache_dir: Path | None = field(default=None)

    @property
    def name(self) -> str:
        return self.repo_id

    def with_head(self, spec: str):
        """This source, composing the caller's depth head instead of the card's own.

        Overridden because this source ASSEMBLES the head with the backbone into one reference
        module, so the config's declaration alone would leave the fp leg running the published
        head against a device graph compiled for the caller's.
        """
        return self._repointed("head", spec, flag="--head", method="with_head")

    def load_reference(self):
        from transformers import Dinov2Model

        from model_axscript.shared.checkpoint import hf_access

        from .axten_encoder import _retarget_image_size

        with hf_access(self.repo_id):  # a gated repo gets the login hint
            backbone = Dinov2Model.from_pretrained(self.repo_id).eval()
        _retarget_image_size(backbone, self.image_size)
        weight, bias = load_depth_head_weights(self.head, cache_dir=self.cache_dir)
        check_backbone(weight, head_spec=self.head, hidden_size=int(backbone.config.hidden_size))
        return Dinov2LinearDepth(backbone, weight, bias).eval()

    def make_processor(self, image_size: int | None = None):
        from .calibration import make_processor

        return make_processor(self.repo_id, image_size or self.image_size)

    def backbone_of(self, ref):
        return ref.backbone

    def load_weights(self, model, ref, image_size: int) -> None:
        raise NotImplementedError(
            "this card has no in-tree device recipe to copy weights into: the depth head is not "
            "the linear one `imported_dino` can bake, so the card builds through axten "
            "(implementation='dinov2-depth-axten') and the compiler reads the weights itself."
        )

    def encode(self, ref, pixel_values):
        """The depth map, which is this card's output on both legs."""
        return ref(pixel_values)

    def export_module(self, ref, image_size: int | None = None):
        """Already a plain-tensor module retargeted at load, so the compiler exports it as it is."""
        return ref

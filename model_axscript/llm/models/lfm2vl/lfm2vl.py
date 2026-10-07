# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Self-contained LFM2.5-VL reference (SigLIP2 vision tower + LFM2 decoder) in PyTorch.

LFM2.5-VL (`LFM2 Technical Report <https://arxiv.org/abs/2511.23404>`_, §5) is an
LFM2 decoder with a SigLIP2 NaFlex encoder attached through a *lightweight
connector*: a PixelUnshuffle that folds each 2x2 patch block into one token,
then a two-layer MLP into the language hidden size. Two releases are supported and
differ only in their widths and depths (:data:`SUPPORTED_MODELS`): LFM2.5-VL-450M
pairs the Base-86M encoder with the LFM2.5-350M backbone, LFM2.5-VL-3B the So400m
encoder with LFM2.5-2.6B.

Both towers are shapes this tree already compiles:

* the decoder is the hybrid short-conv / full-attention LFM2 stack of ``models/lfm2``,
  which supplies both layer kernels and both buffer packers unchanged;
* a SigLIP2 encoder layer is a plain pre-norm ViT block -- LayerNorm, three separate
  biased projections, bidirectional attention, biased GELU MLP -- so the fused block
  kernel and binder in ``models/dino`` apply to it as written.

The vision tower carries **no rotary embedding** (SigLIP2 learns an absolute position
grid and resamples it per resolution) and the decoder's rotary positions are plain 1-D,
so the decode loop's tables are the backbone's, unchanged. Three things did need care.

**The resampled position grid.** SigLIP2 NaFlex stores one learned ``num_patches``
(16x16) position grid and bilinearly resamples it -- with antialiasing -- onto whatever
patch grid an image produced. The device graph bakes one grid, so the resample is a host
constant computed at load (:func:`resize_position_embedding`).

**PixelUnshuffle is a reshape only in merge-block order.** The connector folds patches
``(2h+i, 2w+j)`` into one row whose channels are ordered ``(0,0), (0,1), (1,0), (1,1)``
(see :func:`merge_permutation` for the derivation). Those four rows are *not* adjacent
in the processor's row-major patch order, so the fold would be a gather. Instead the
patch rows are permuted into merge-block order on the host, once, before the tower --
and the position grid with them. Nothing else in the tower is position-dependent (no
rotary embedding, bidirectional attention, one uniform mask), so the permutation is
exact, and it makes the unshuffle a free reshape of the packed buffer. The merged rows
come out in the row-major order the placeholders expect.

**Where the LayerNorms are.** The encoder's ``post_layernorm`` runs at vision width
*before* the unshuffle; the connector's own LayerNorm (when the checkpoint has one --
both supported releases set ``projector_use_layernorm: false``) runs at merged width
*after* it. Both live in :class:`Lfm2VlConnector` so one device kernel covers the pair.

**The So400m widths.** Its head_dim (72) and MLP width (4304) are not PWORD multiples,
so the block kernel lays them out padded (a 128-lane head stride, 4352) with zeros in
the pad -- see ``models/dino/optimized_impl``. That is a device layout only: this
reference always runs the checkpoint's own widths.

Layout of the pieces:
  lfm2vl.py     -- this file: configs, the torch fake-quant reference, HF loading, the
                   host-side geometry (position resample, merge-block order).
  vision_impl.py -- the alto-script vision kernels + the single-dispatch device graph.
  text_impl.py  -- the decoder implementation (the LFM2 layer kernels reused as-is)
                   plus the VLM host hook that splices the image embeddings in.
  pipeline.py   -- preprocessing (fit, dynamic tiling), build_artifacts, the pipeline.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...checkpoint import load_hf_safetensors
from ...quant import ActivationObserver, QuantLinear, RangeObserver
from ...quant import quantize_model as _quantize_model
from ...quant import record_lut_ranges as _record_lut_ranges
from ..architecture import LLMArchitecture, register_architecture
from ..dino.reference import DinoBlock, DinoBlockConfig, QuantLayerNorm
from ..dino.reference import smoothquant_groups as _vision_block_groups
from ..lfm2.lfm2 import LFM2Config, Lfm2DecoderLayer, _remap_hf_key
from ..lfm2.lfm2 import smoothquant_groups as _text_groups
from ...modules import RMSNorm, RotaryEmbedding

PWORD = 64


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


@dataclass
class Lfm2VlVisionConfig:
    """SigLIP2 encoder + connector hyperparameters, and the compile-pinned geometry.

    Defaults are LFM2.5-VL-450M's Base-86M encoder at a 32x32 patch grid (512x512
    pixels -- the checkpoint's own ``tile_size`` and its 256-token ceiling), which is
    1024 encoder tokens in and 256 image tokens out.

    The grid is part of the config because the device graph is compiled for one token
    count: the patch count, the resampled position grid and the attention mask are all
    baked at it. A different resolution is a different artifact -- but *tiling* is not,
    because every tile is exactly one compiled grid (see pipeline.tile_image).
    """

    depth: int = 12
    hidden_size: int = 768
    intermediate_size: int = 3072
    num_heads: int = 12
    num_channels: int = 3
    patch_size: int = 16
    #: The learned NaFlex position grid, resampled onto grid_h x grid_w at load.
    num_position_embeddings: int = 256
    layernorm_eps: float = 1e-6

    # --- connector (HF's multi_modal_projector) -------------------------------------
    downsample_factor: int = 2
    projector_hidden_size: int = 2048
    projector_bias: bool = True
    projector_hidden_act: str = "gelu"
    #: Whether the connector carries its own LayerNorm at merged width. Neither released
    #: LFM2.5-VL does (the encoder's post_layernorm is the only one); the transformers
    #: implementation always does, which is what the default follows.
    projector_use_layernorm: bool = True
    #: The decoder's hidden size -- what the connector projects into.
    out_hidden_size: int = 1024

    # --- compiled geometry ------------------------------------------------------------
    #: patch grid (rows, cols) of the compiled resolution -- image_size / patch_size.
    grid_h: int = 32
    grid_w: int = 32
    #: How a picture is forced onto that one grid: "letterbox" scales it to fit and pads
    #: with mid-grey (which the 0.5/0.5 normalization sends to ~0), keeping the aspect
    #: ratio; "stretch" resizes anamorphically, using every patch at the cost of
    #: distorting. A BUILD-time property, so calibration sees the geometry inference will.
    fit_mode: str = "stretch"

    # --- dynamic tiling (host-side; recorded so a run reproduces the build) ------------
    #: Tile edge in pixels. Tiling is only offered when it equals the compiled square,
    #: since a tile has to be exactly one compiled dispatch.
    tile_size: int = 512
    min_tiles: int = 2
    max_tiles: int = 10
    #: Append a whole-image thumbnail (the report's "coarse global representation").
    use_thumbnail: bool = True
    #: Whether a high-resolution picture is split at all. Off makes every picture one
    #: dispatch, which is the cheapest and the least detailed.
    do_image_splitting: bool = True

    # --- LUT chain lengths --------------------------------------------------------------
    # How many hardware LUTs each nonlinearity's additive chain gets: the softmax's
    # exp/reciprocal, the GELUs (encoder MLP and connector), the LayerNorms' inv_sqrt.
    # They are THIS tower's, not the shared DINO_* environment defaults, because the two
    # encoders do not want the same thing: an approximation that is within noise on the
    # Base-86M's activations is not automatically within noise on So400m's, whose residual
    # stream is two orders of magnitude wider. One number per nonlinearity drives both the
    # kernel's LUT buffer shape and what the torch reference models, so they cannot drift.
    sdpa_n_luts: int = 1
    gelu_n_luts: int = 1
    norm_n_luts: int = 2

    # --- properties -------------------------------------------------------------------

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    # The three widths the DEVICE lays the encoder block out at. They are the
    # checkpoint's own on the Base-86M encoder (head_dim 64, intermediate 3072) and
    # rounded up to whole PWORDs on So400m (72 -> a 128-lane head stride, 4304 -> 4352),
    # where the pad is zero on both sides of every contraction. Delegated to the block
    # config so the reference, the kernel and the binder cannot disagree about them.

    @property
    def padded_head_dim(self) -> int:
        return self.block_config().padded_head_dim

    @property
    def qkv_dim(self) -> int:
        """q/k/v/o_proj width on device: the padded heads, laid out head-major."""
        return self.block_config().qkv_dim

    @property
    def padded_intermediate(self) -> int:
        return self.block_config().padded_intermediate

    @property
    def pads_widths(self) -> bool:
        """Whether this encoder's device layout carries any zero padding at all."""
        return self.block_config().pads_widths

    @property
    def patch_dim(self) -> int:
        """Flattened patch length K == C * P * P -- the processor's feature axis.

        SigLIP2 NaFlex patchifies on the host and projects with a plain Linear, so the
        patches arrive flattened and there is no im2col to undo.
        """
        return self.num_channels * self.patch_size**2

    @property
    def num_patches(self) -> int:
        return self.grid_h * self.grid_w

    @property
    def seq_len(self) -> int:
        """Patch count padded up to a PWORD multiple (the compiled token axis)."""
        return _round_up(self.num_patches, PWORD)

    @property
    def merge_unit(self) -> int:
        """Patches folded into one connector token (the PixelUnshuffle factor squared)."""
        return self.downsample_factor**2

    @property
    def merged_dim(self) -> int:
        """Connector input width: hidden_size * merge_unit."""
        return self.hidden_size * self.merge_unit

    @property
    def num_merged(self) -> int:
        """Image tokens the LLM sees per dispatch (patches after the unshuffle)."""
        return self.num_patches // self.merge_unit

    @property
    def image_height(self) -> int:
        return self.grid_h * self.patch_size

    @property
    def image_width(self) -> int:
        return self.grid_w * self.patch_size

    @property
    def image_shape(self) -> tuple[int, int]:
        """(height, width) in pixels of the compiled geometry.

        The grid need not be square: the position resample and the merge order are both
        built from the grid_h/grid_w pair. Tiling, however, needs a square that matches
        ``tile_size`` -- see :attr:`tiling_supported`.
        """
        return self.image_height, self.image_width

    @property
    def image_geometry(self) -> str:
        """The compiled geometry for a log line: "512" if square, else "288x512" (HxW)."""
        height, width = self.image_shape
        return str(height) if height == width else f"{height}x{width}"

    @property
    def num_grid_per_side(self) -> int:
        return int(round(self.num_position_embeddings**0.5))

    @property
    def tiling_supported(self) -> bool:
        """Whether this artifact can tile: a tile has to BE one compiled dispatch.

        The processor cuts ``tile_size`` squares, so the compiled grid must be that same
        square. At the default (512 px, a 32x32 grid) it is -- which is why 512 is the
        default rather than something cheaper.
        """
        height, width = self.image_shape
        return height == width == self.tile_size

    def validate(self) -> None:
        # head_dim and intermediate_size are free to be unaligned: the block kernel lays
        # them out at padded_head_dim / padded_intermediate and zeroes the pad, which is
        # what lets the So400m encoder (72, 4304) compile. Every width below is one a
        # kernel indexes directly, with no padding to hide behind.
        for name in (
            "hidden_size",
            "out_hidden_size",
            "patch_dim",
            "merged_dim",
            "projector_hidden_size",
        ):
            value = getattr(self, name)
            if value % PWORD:
                raise NotImplementedError(f"vision {name}={value} must be a multiple of {PWORD}")
        merge = self.downsample_factor
        if self.grid_h % merge or self.grid_w % merge:
            raise ValueError(
                f"patch grid {self.grid_h}x{self.grid_w} must divide by the connector's "
                f"downsample factor {merge}"
            )
        if self.num_grid_per_side**2 != self.num_position_embeddings:
            raise ValueError(
                f"num_position_embeddings {self.num_position_embeddings} is not a square "
                "grid; SigLIP2 stores the learned positions as one square table"
            )

    def block_config(self) -> DinoBlockConfig:
        """The block-shape record one encoder layer is built from.

        A SigLIP2 encoder layer is a plain pre-norm ViT block -- LayerNorm, separate
        biased q/k/v, bidirectional attention, biased GELU MLP -- which is the shape the
        shared ViT block in ``models/dino`` is written against, LayerScale pinned at 1.

        It also owns the device widths: Base-86M's are the checkpoint's own, and So400m's
        head_dim (72) and intermediate (4304) are padded to whole PWORDs there.
        """
        return DinoBlockConfig(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            # So400m: 4304 is not a whole multiple of hidden 1152, and head_dim 72 is not
            # a PWORD. The reference runs both unpadded; the device pads them.
            mlp_ratio=self.intermediate_size // self.hidden_size,
            mlp_hidden_size=self.intermediate_size,
            head_dim_is_pword=self.head_dim % PWORD == 0,
            sdpa_n_luts=self.sdpa_n_luts,
            gelu_n_luts=self.gelu_n_luts,
            norm_n_luts=self.norm_n_luts,
            seq_len=self.seq_len,
            layernorm_eps=self.layernorm_eps,
            # image_size feeds a conv patch embed this tower does not have; SigLIP2
            # projects patches the processor already flattened, so nothing reads it.
            # Recorded as the height, one number being unable to describe a non-square grid.
            image_size=self.image_height,
            patch_size=self.patch_size,
            num_channels=self.num_channels,
        )

    @classmethod
    def from_hf_dict(cls, cfg: dict, *, grid_h: int, grid_w: int) -> "Lfm2VlVisionConfig":
        """Build from a HuggingFace Lfm2VlConfig dict (the whole config, not the sub-dict).

        The connector's shape lives at the TOP level of the HF config and the encoder's
        under ``vision_config``, so both are read here.
        """
        vision = cfg["vision_config"]
        image = cfg.get("image_processor") or {}
        return cls(
            depth=vision["num_hidden_layers"],
            hidden_size=vision["hidden_size"],
            intermediate_size=vision["intermediate_size"],
            num_heads=vision["num_attention_heads"],
            num_channels=vision.get("num_channels", 3),
            patch_size=vision["patch_size"],
            num_position_embeddings=vision["num_patches"],
            layernorm_eps=vision.get("layer_norm_eps", 1e-6),
            downsample_factor=cfg.get("downsample_factor", 2),
            projector_hidden_size=cfg["projector_hidden_size"],
            projector_bias=cfg.get("projector_bias", True),
            projector_hidden_act=cfg.get("projector_hidden_act", "gelu"),
            # transformers' Lfm2VlMultiModalProjector always carries the norm, so a
            # checkpoint that does not want one has to say so; hence the True default.
            projector_use_layernorm=cfg.get("projector_use_layernorm", True),
            out_hidden_size=cfg["text_config"]["hidden_size"],
            grid_h=grid_h,
            grid_w=grid_w,
            tile_size=cfg.get("tile_size", image.get("tile_size", 512)),
            min_tiles=cfg.get("min_tiles", image.get("min_tiles", 2)),
            max_tiles=cfg.get("max_tiles", image.get("max_tiles", 10)),
            use_thumbnail=cfg.get("use_thumbnail", image.get("use_thumbnail", True)),
            do_image_splitting=cfg.get(
                "do_image_splitting", image.get("do_image_splitting", True)
            ),
        )


#: The releases this port targets, and the encoder each one carries. Both are SigLIP2
#: NaFlex towers on an LFM2.5 backbone, which is what makes one implementation cover
#: them: the same block, the same connector, the same decoder kernels at other widths.
SUPPORTED_MODELS = {
    "LFM2.5-VL-450M": "SigLIP2 NaFlex Base-86M (hidden 768, head_dim 64, MLP 3072) "
    "on the LFM2.5-350M backbone",
    "LFM2.5-VL-3B": "SigLIP2 NaFlex So400m (hidden 1152, head_dim 72, MLP 4304) "
    "on the LFM2.5-2.6B backbone",
}


def check_supported(cfg: dict) -> None:
    """Refuse a checkpoint these kernels are not written for, at the config, by name.

    What they cover is a SigLIP2 NaFlex encoder whose hidden size is a PWORD multiple.
    head_dim and the MLP width are NOT part of that: the block kernel lays both out at
    padded widths and zeroes the pad, which is what the So400m encoder needs (72 and
    4304, neither PWORD-aligned). Everything else -- another encoder family, or a hidden
    size the kernels index directly and cannot pad -- would fail deep in the geometry
    validation with a message about PWORDs rather than about the model, so it is named
    here instead. Checked structurally rather than by name, so a FINE-TUNE of a supported
    release (its own weights, its own directory) still loads.
    """
    vision = cfg["vision_config"]
    kind = vision.get("model_type", "siglip2_vision_model")
    hidden = vision["hidden_size"]
    reasons = []
    if kind != "siglip2_vision_model":
        reasons.append(f"a {kind} encoder")
    if hidden % PWORD:
        reasons.append(f"hidden_size {hidden} (not a multiple of {PWORD})")
    if reasons:
        supported = "; ".join(f"{name}, {what}" for name, what in SUPPORTED_MODELS.items())
        raise NotImplementedError(
            f"this implementation targets {supported}. This checkpoint has "
            f"{' and '.join(reasons)}, which the fused encoder-block kernel and its binder "
            "do not cover."
        )


@dataclass
class Lfm2VlConfig:
    """The whole model: an LFM2 text config + the vision/connector config + token ids."""

    text: LFM2Config = field(default_factory=LFM2Config)
    vision: Lfm2VlVisionConfig = field(default_factory=Lfm2VlVisionConfig)
    image_token_id: int = 396

    @property
    def num_hidden_layers(self) -> int:
        """Decoder depth -- what the layer-per-kind machinery counts."""
        return self.text.num_hidden_layers

    def __post_init__(self) -> None:
        # dataclass fields survive a JSON round-trip as plain dicts; rebuild them.
        if isinstance(self.text, dict):
            self.text = LFM2Config(**self.text)
        if isinstance(self.vision, dict):
            self.vision = Lfm2VlVisionConfig(**self.vision)
        self.text.full_attn_idxs = tuple(self.text.full_attn_idxs)

    @classmethod
    def from_hf_dict(cls, cfg: dict, *, grid_h: int, grid_w: int) -> "Lfm2VlConfig":
        check_supported(cfg)
        text_cfg = dict(cfg["text_config"])
        # LFM2.5 moved the rotary settings into a rope_parameters block; LFM2 kept them
        # flat. LFM2Config.from_hf_dict reads the flat spelling, so lift them first.
        rope = text_cfg.get("rope_parameters") or {}
        if rope:
            if rope.get("rope_type", "default") != "default":
                raise NotImplementedError(
                    f"only default (unscaled) RoPE is supported; got {rope['rope_type']!r}"
                )
            text_cfg.setdefault("rope_theta", rope["rope_theta"])
        text = LFM2Config.from_hf_dict(text_cfg)
        # The only LFM2 Metis kernels run the short-conv block's projections as W8A8.
        text = replace(text, quantize_conv=True)
        return cls(
            text=text,
            vision=Lfm2VlVisionConfig.from_hf_dict(cfg, grid_h=grid_h, grid_w=grid_w),
            image_token_id=cfg.get("image_token_id", 396),
        )


# --------------------------------------------------------------------------------------
# Host-side geometry: what the device graph bakes in.
# Pure functions of the config (+ the checkpoint's position table), so the torch
# reference and the compiled pipeline call the same code.
# --------------------------------------------------------------------------------------


def merge_permutation(grid_h: int, grid_w: int, merge: int) -> torch.Tensor:
    """Row-major patch index of each token in MERGE-BLOCK order, as a (num_patches,) index.

    HF's connector reshapes the encoder output to ``(1, grid_h, grid_w, C)`` and applies
    a PixelUnshuffle whose net effect is::

        out[h2, w2, i * 2C + j * C + c] = in[2 * h2 + i, 2 * w2 + j, c]

    -- i.e. merged row ``h2 * (grid_w / 2) + w2`` is the concatenation of the four patch
    rows ``(2h2, 2w2), (2h2, 2w2 + 1), (2h2 + 1, 2w2), (2h2 + 1, 2w2 + 1)``, in that
    order. Reordering the tower's tokens this way makes the fold a pure reshape of the
    packed buffer: four consecutive rows become one ``merge_unit * hidden`` row.

    Returned as the permutation to APPLY to a row-major ``(num_patches, ...)`` buffer
    (``patches[merge_permutation(...)]``), so the same index reorders the patches, the
    resampled position grid, and anything else laid out per patch.
    """
    mh, mw = grid_h // merge, grid_w // merge
    rows = torch.arange(mh).view(mh, 1, 1, 1) * merge + torch.arange(merge).view(1, 1, merge, 1)
    cols = torch.arange(mw).view(1, mw, 1, 1) * merge + torch.arange(merge).view(1, 1, 1, merge)
    rows = rows.expand(mh, mw, merge, merge)
    cols = cols.expand(mh, mw, merge, merge)
    return (rows * grid_w + cols).reshape(-1)


def to_tower_order(patches: torch.Tensor, cfg: Lfm2VlVisionConfig) -> torch.Tensor:
    """Reorder one or more images' patch rows from the processor's order into merge order.

    ``patches`` is ``(n_images * num_patches, patch_dim)`` in the processor's row-major
    patch order; each image is permuted independently (they are separate encodings).
    """
    index = merge_permutation(cfg.grid_h, cfg.grid_w, cfg.downsample_factor)
    rows = torch.as_tensor(patches)
    if rows.shape[0] % cfg.num_patches:
        raise ValueError(
            f"{rows.shape[0]} patch rows is not a whole number of {cfg.num_patches}-patch "
            "images"
        )
    images = rows.reshape(-1, cfg.num_patches, *rows.shape[1:])
    return images[:, index].reshape(rows.shape)


def resize_position_embedding(
    cfg: Lfm2VlVisionConfig, position_weight: torch.Tensor
) -> torch.Tensor:
    """Resample the learned square position grid onto this patch grid, in tower order.

    Port of ``Siglip2VisionEmbeddings.resize_positional_embeddings`` for a single image:
    bilinear with antialiasing, computed in float32 (as HF does on CPU), returning
    ``(num_patches, hidden)`` already permuted into merge-block order.
    """
    side = cfg.num_grid_per_side
    grid = position_weight.detach().to(torch.float32).reshape(side, side, -1)
    grid = grid.permute(2, 0, 1).unsqueeze(0)  # (1, hidden, side, side)
    resized = F.interpolate(
        grid, size=(cfg.grid_h, cfg.grid_w), mode="bilinear", align_corners=False, antialias=True
    )
    flat = resized.reshape(resized.shape[1], cfg.num_patches).transpose(0, 1).contiguous()
    return flat[merge_permutation(cfg.grid_h, cfg.grid_w, cfg.downsample_factor)].contiguous()


def patch_grid(image_size: int | tuple[int, int], patch: int) -> tuple[int, int]:
    """(grid_h, grid_w) for a square side or an (H, W) pair of pixel sizes.

    The one spelling of the geometry argument every entry point takes. Both sides must
    divide by ``patch``; that the GRID divides by the connector's downsample factor is
    ``validate``'s job.
    """
    height, width = (image_size, image_size) if isinstance(image_size, int) else image_size
    for name, value in (("height", height), ("width", width)):
        if value <= 0 or value % patch:
            raise ValueError(
                f"image {name} {value} must be a positive multiple of patch_size {patch}"
            )
    return height // patch, width // patch


def config_from_pretrained(
    repo_id: str,
    *,
    image_size: int | tuple[int, int] = 512,
    vision_overrides: dict | None = None,
) -> Lfm2VlConfig:
    """The config for a HF repo or local checkpoint dir at a pinned geometry (no weights).

    ``image_size`` is a square side or an (height, width) pair. Cheap enough for the
    host-side tools that need the geometry without loading the parameters.

    ``vision_overrides`` replaces fields of the vision sub-config -- the build's own
    choices, as opposed to the checkpoint's. It goes here rather than onto the loaded
    model because the reference's modules are BUILT from these (a LayerNorm's LUT chain
    length is a constructor argument), so setting them afterwards would change the config
    the artifact records without changing the model calibration saw.
    """
    import json

    from ...checkpoint import repo_file

    with open(repo_file(repo_id, "config.json")) as f:
        hf_config = json.load(f)
    grid_h, grid_w = patch_grid(image_size, hf_config["vision_config"]["patch_size"])
    config = Lfm2VlConfig.from_hf_dict(hf_config, grid_h=grid_h, grid_w=grid_w)
    if vision_overrides:
        config = replace(config, vision=replace(config.vision, **vision_overrides))
    return config


# --------------------------------------------------------------------------------------
# Torch fake-quant reference
# --------------------------------------------------------------------------------------


class Lfm2VlPatchEmbed(nn.Module):
    """The SigLIP2 patch projection as a symmetric-input QuantLinear + a float bias.

    NaFlex patchifies on the host, so this is already a dense matmul over each patch's
    C*P*P features -- no im2col. Input quant is symmetric (zp = 0) so the host can
    quantize pixels with a single scale, and the bias is added after dequant; both match
    the device kernel (see vision_impl.make_vision_embed_kernel).
    """

    def __init__(self, cfg: Lfm2VlVisionConfig, **quant_kwargs) -> None:
        super().__init__()
        self.proj = QuantLinear(cfg.patch_dim, cfg.hidden_size, act_symmetric=True, **quant_kwargs)
        self.bias = nn.Parameter(torch.zeros(cfg.hidden_size))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # (num_patches, patch_dim) -> (1, num_patches, hidden)
        return self.proj(patches.unsqueeze(0)) + self.bias


class Lfm2VlVisionBlock(DinoBlock):
    """A SigLIP2 encoder layer: the shared ViT block, with the tanh GELU.

    Structurally it IS the shared pre-norm ViT block (LayerNorm, separate biased q/k/v,
    bidirectional attention, biased GELU MLP), so it inherits the module layout the fused
    device kernel and its binder are written against. LayerScale (ls1/ls2) stays at its
    1.0 init -- SigLIP2 has none, and the binder folds it into a dequant scale where it
    disappears.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # The base block records the range for the exact (erf) GELU; SigLIP2's MLP
        # evaluates `gelu_pytorch_tanh`. The observer's function name is what
        # layout.fit_lut builds the device table from, so renaming it is the whole
        # change; the connector below, whose activation really is erf, keeps "gelu".
        self.gelu_range = RangeObserver("gelu_tanh")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        x = self.ln1(hidden)
        q = self._heads(self.q_proj(x) + self.q_bias)
        k = self._heads(self.k_proj(x) + self.k_bias)
        v = self._heads(self.v_proj(x) + self.v_bias)
        attn = self.sdpa(q, k, v, None)  # bidirectional: every patch sees every patch
        batch, _, seq, _ = attn.shape
        attn = attn.transpose(1, 2).reshape(batch, seq, cfg.hidden_size)
        res1 = hidden + (self.o_proj(attn) + self.o_bias) * self.ls1
        y = self.ln2(res1)
        h1 = F.gelu(self.gelu_range(self.fc1(y) + self.fc1_bias), approximate="tanh")
        return res1 + (self.fc2(h1) + self.fc2_bias) * self.ls2


class Lfm2VlConnector(nn.Module):
    """post_layernorm -> PixelUnshuffle -> [LayerNorm] -> GELU MLP -> decoder hidden size.

    HF splits this across two modules -- the encoder's ``post_layernorm`` and
    ``multi_modal_projector`` -- but on device they are one kernel, because the norm at
    vision width and the fold that follows it share a datapath. Both norms are declared
    here so the binder has one module to read.

    ``norm`` is the encoder's post_layernorm, applied BEFORE the fold at vision width.
    ``merged_norm`` is the connector's own, applied AFTER it at ``merge_unit * hidden``
    width; both released LFM2.5-VLs have none (``projector_use_layernorm: false``), and
    this is then None.

    The input is expected in merge-block token order (see :func:`merge_permutation`), so
    the unshuffle is the reshape it is on device.
    """

    def __init__(self, cfg: Lfm2VlVisionConfig, **quant_kwargs) -> None:
        super().__init__()
        merged = cfg.merged_dim
        self.merge_unit = cfg.merge_unit
        self.norm = QuantLayerNorm(cfg.hidden_size, cfg.layernorm_eps, cfg.norm_n_luts)
        self.merged_norm = (
            QuantLayerNorm(merged, cfg.layernorm_eps, cfg.norm_n_luts)
            if cfg.projector_use_layernorm
            else None
        )
        self.linear_fc1 = QuantLinear(merged, cfg.projector_hidden_size, **quant_kwargs)
        self.fc1_bias = nn.Parameter(torch.zeros(cfg.projector_hidden_size))
        # projector_hidden_act is "gelu": the exact (erf) form, and the encoder block's
        # is the tanh approximation, so the two fit different device tables.
        self.gelu_range = RangeObserver("gelu")
        self.linear_fc2 = QuantLinear(
            cfg.projector_hidden_size, cfg.out_hidden_size, **quant_kwargs
        )
        self.fc2_bias = nn.Parameter(torch.zeros(cfg.out_hidden_size))

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch, seq, hidden = hidden_states.shape
        x = self.norm(hidden_states)
        x = x.reshape(batch, seq // self.merge_unit, hidden * self.merge_unit)
        if self.merged_norm is not None:
            x = self.merged_norm(x)
        x = F.gelu(self.gelu_range(self.linear_fc1(x) + self.fc1_bias))
        return self.linear_fc2(x) + self.fc2_bias


class Lfm2VlVisionModel(nn.Module):
    """Patch embed (+ resampled position grid) -> N encoder blocks -> connector.

    Returns the image embeddings the decoder consumes. Geometry is fixed by the config,
    so the whole tower is a function of the flattened patches alone -- which is also
    what makes it callable straight from the calibration loop.
    """

    def __init__(self, cfg: Lfm2VlVisionConfig, **quant_kwargs) -> None:
        super().__init__()
        cfg.validate()
        self.config = cfg
        block_cfg = cfg.block_config()
        self.patch_embed = Lfm2VlPatchEmbed(cfg, **quant_kwargs)
        self.blocks = nn.ModuleList(
            [Lfm2VlVisionBlock(block_cfg, **quant_kwargs) for _ in range(cfg.depth)]
        )
        self.connector = Lfm2VlConnector(cfg, **quant_kwargs)
        # Filled by load_hf_weights (the resampled grid for this resolution, in tower order).
        self.register_buffer(
            "pos_embed", torch.zeros(cfg.num_patches, cfg.hidden_size), persistent=False
        )

    def _encode_one(self, patches: torch.Tensor) -> torch.Tensor:
        """One image's patches (already in tower order) -> (1, num_merged, out_hidden)."""
        hidden = self.patch_embed(patches) + self.pos_embed
        for block in self.blocks:
            hidden = block(hidden)
        return self.connector(hidden)

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        """``(n_images * num_patches, patch_dim)`` -> ``(1, n_images * num_merged, out_hidden)``.

        Patches arrive in the processor's row-major order and are permuted here (see
        :func:`to_tower_order`). Images are encoded ONE AT A TIME and concatenated, which
        is not a shortcut: HF runs the tower per image and unpads each one separately, so
        a patch never attends across an image boundary. Every image shares the compiled
        geometry, hence the same position grid.
        """
        cfg = self.config
        ordered = to_tower_order(patches, cfg)
        chunks = [
            self._encode_one(ordered[i : i + cfg.num_patches])
            for i in range(0, ordered.shape[0], cfg.num_patches)
        ]
        return torch.cat(chunks, dim=1)


class Lfm2VlTextModel(nn.Module):
    """The LFM2 hybrid decoder stack, driven by embeddings rather than ids.

    Identical to ``models.lfm2.Lfm2Model`` except that it takes ``inputs_embeds`` (the
    caller splices in the image embeddings), so the leaf names still map 1:1 onto the
    checkpoint's ``model.language_model.*``.
    """

    def __init__(self, cfg: LFM2Config) -> None:
        super().__init__()
        self.config = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList(
            [Lfm2DecoderLayer(cfg, i) for i in range(cfg.num_hidden_layers)]
        )
        self.embedding_norm = RMSNorm(cfg.hidden_size, eps=cfg.norm_eps)
        self.rotary_emb = RotaryEmbedding(
            cfg.head_dim, cfg.max_position_embeddings, cfg.rope_theta
        )

    def forward(
        self, inputs_embeds: torch.Tensor, position_ids: torch.Tensor | None = None
    ) -> torch.Tensor:
        seq_len = inputs_embeds.shape[1]
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        if position_ids is None:
            position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
        cos, sin = self.rotary_emb(position_ids)
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=1)[None, None]
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden, cos, sin, mask)
        return self.embedding_norm(hidden)


class Lfm2VlForConditionalGeneration(nn.Module):
    """The full VLM: SigLIP2 tower + connector + LFM2 decoder + (tied) LM head.

    ``forward`` takes ONE sample dict so the shared PTQ toolkit (which calls
    ``module(batch)``) can calibrate both towers jointly on real multimodal inputs::

        {"input_ids": (1, T) int64,
         "patches":   (n_images * num_patches, patch_dim) float or None}
    """

    def __init__(self, config: Lfm2VlConfig, **quant_kwargs) -> None:
        super().__init__()
        self.config = config
        self.visual = Lfm2VlVisionModel(config.vision, **quant_kwargs)
        self.text_model = Lfm2VlTextModel(config.text)
        self.lm_head = nn.Linear(config.text.hidden_size, config.text.vocab_size, bias=False)
        if config.text.tie_word_embeddings:
            self.lm_head.weight = self.text_model.embed_tokens.weight

    @property
    def device(self) -> torch.device:
        return self.text_model.embed_tokens.weight.device

    def input_embeddings(self, sample: dict) -> torch.Tensor:
        """Token embeddings with the image embeddings scattered over the placeholders."""
        cfg = self.config
        input_ids = torch.as_tensor(sample["input_ids"]).reshape(1, -1).to(self.device)
        embeds = self.text_model.embed_tokens(input_ids)
        patches = sample.get("patches")
        if patches is None:
            return embeds
        merged = self.visual(torch.as_tensor(patches).to(device=self.device, dtype=embeds.dtype))
        mask = (input_ids == cfg.image_token_id).unsqueeze(-1).expand_as(embeds)
        return embeds.masked_scatter(mask, merged.reshape(-1, embeds.shape[-1]))

    def forward(self, sample) -> torch.Tensor:
        if torch.is_tensor(sample):  # text-only convenience form
            sample = {"input_ids": sample}
        return self.lm_head(self.text_model(self.input_embeddings(sample)))

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "LiquidAI/LFM2.5-VL-450M",
        *,
        image_size: int | tuple[int, int] = 512,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        quantize_attention: bool = True,
        vision_overrides: dict | None = None,
        **config_overrides,
    ) -> "Lfm2VlForConditionalGeneration":
        """Load a HF LFM2-VL checkpoint, pinning the vision tower to ``image_size``.

        ``quantize_attention`` fake-quantizes the DECODER's SDPA (the vision tower's is
        always quantized). It defaults on because the device layer kernel binds
        calibrated SDPA scales -- an unquantized reference cannot fill them.

        ``repo_id`` is a hub id or a local checkpoint directory: custom weights are the
        same call with a path, as they are everywhere else in HuggingFace.
        """
        config = config_from_pretrained(
            repo_id, image_size=image_size, vision_overrides=vision_overrides
        )
        config = replace(config, text=replace(config.text, quantize_attention=quantize_attention))
        if config_overrides:
            config = replace(config, **config_overrides)
        model = cls(config).to(device=device, dtype=dtype)
        # The checkpoint stages on the HOST in its OWN dtype (bf16, not `dtype`): the
        # parameters are already allocated, so every load is a widening copy_ and an
        # upcast would only buy a second full-size model.
        model.load_weights(repo_id)
        return model.eval()

    def load_weights(self, repo_id: str) -> "Lfm2VlForConditionalGeneration":
        """Load a HF checkpoint's weights into this (already configured) model.

        The weight half of ``from_pretrained``, on its own, so a caller can swap in their
        own finetune -- a hub id or a local directory of the SAME architecture -- without
        rebuilding the module.
        """
        load_hf_weights(self, load_hf_safetensors(repo_id, device="cpu", dtype=None))
        return self


# --------------------------------------------------------------------------------------
# HuggingFace checkpoint -> reference module
# --------------------------------------------------------------------------------------

#: encoder block parameter (relative) -> HF name under
#: model.vision_tower.vision_model.encoder.layers.<i>.
_BLOCK_MAP = {
    "ln1.weight": "layer_norm1.weight",
    "ln1.bias": "layer_norm1.bias",
    "ln2.weight": "layer_norm2.weight",
    "ln2.bias": "layer_norm2.bias",
    "q_proj.weight": "self_attn.q_proj.weight",
    "q_bias": "self_attn.q_proj.bias",
    "k_proj.weight": "self_attn.k_proj.weight",
    "k_bias": "self_attn.k_proj.bias",
    "v_proj.weight": "self_attn.v_proj.weight",
    "v_bias": "self_attn.v_proj.bias",
    "o_proj.weight": "self_attn.out_proj.weight",
    "o_bias": "self_attn.out_proj.bias",
    "fc1.weight": "mlp.fc1.weight",
    "fc1_bias": "mlp.fc1.bias",
    "fc2.weight": "mlp.fc2.weight",
    "fc2_bias": "mlp.fc2.bias",
}
#: connector parameter (relative) -> HF name. ``norm`` is the ENCODER's post_layernorm,
#: which this module owns because the device kernel fuses it with the fold that follows.
_CONNECTOR_MAP = {
    "norm.weight": "model.vision_tower.vision_model.post_layernorm.weight",
    "norm.bias": "model.vision_tower.vision_model.post_layernorm.bias",
    "linear_fc1.weight": "model.multi_modal_projector.linear_1.weight",
    "fc1_bias": "model.multi_modal_projector.linear_1.bias",
    "linear_fc2.weight": "model.multi_modal_projector.linear_2.weight",
    "fc2_bias": "model.multi_modal_projector.linear_2.bias",
}
_CONNECTOR_NORM_MAP = {
    "merged_norm.weight": "model.multi_modal_projector.layer_norm.weight",
    "merged_norm.bias": "model.multi_modal_projector.layer_norm.bias",
}
_VISION_PREFIX = "model.vision_tower.vision_model."


@torch.no_grad()
def load_hf_weights(model: Lfm2VlForConditionalGeneration, state: dict) -> None:
    """Copy a HF LFM2-VL state dict into the reference model.

    The text tower maps 1:1 apart from the SwiGLU leaf names (w1/w3/w2 -> gate/up/down,
    shared with ``models.lfm2``); the vision tower needs two rewrites: the learned
    position grid is resampled to this resolution (and permuted into merge-block order),
    and the encoder's post_layernorm moves into the connector module.
    """
    cfg = model.config
    vision = cfg.vision

    def get(name: str) -> torch.Tensor:
        try:
            return state[name]
        except KeyError:
            raise KeyError(f"missing checkpoint tensor {name!r}") from None

    # --- text tower: model.language_model.* -> text_model.* (leaf names via _remap_hf_key).
    text_state = {
        _remap_hf_key(key[len("model.language_model.") :]): value
        for key, value in state.items()
        if key.startswith("model.language_model.")
    }
    missing, unexpected = model.text_model.load_state_dict(text_state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected text-tower keys: {unexpected}")
    # Observers own non-persistent buffers absent from any checkpoint.
    observers = {
        name
        for name, module in model.text_model.named_modules()
        if isinstance(module, ActivationObserver)
    }
    missing = [key for key in missing if key.rsplit(".", 1)[0] not in observers]
    if missing:
        raise RuntimeError(f"missing text-tower keys: {missing}")
    if cfg.text.tie_word_embeddings:
        model.lm_head.weight = model.text_model.embed_tokens.weight
    else:
        model.lm_head.weight.copy_(get("lm_head.weight"))

    # --- patch embed + the resampled position grid.
    model.visual.patch_embed.proj.weight.copy_(
        get(_VISION_PREFIX + "embeddings.patch_embedding.weight")
    )
    model.visual.patch_embed.bias.copy_(get(_VISION_PREFIX + "embeddings.patch_embedding.bias"))
    model.visual.pos_embed.copy_(
        resize_position_embedding(
            vision, get(_VISION_PREFIX + "embeddings.position_embedding.weight")
        )
    )

    # --- encoder blocks (1:1 names).
    for i, block in enumerate(model.visual.blocks):
        prefix = f"{_VISION_PREFIX}encoder.layers.{i}."
        for dst, src in _BLOCK_MAP.items():
            block.get_parameter(dst).copy_(get(prefix + src))

    # --- connector (+ the encoder's post_layernorm, which lives in it here).
    mapping = dict(_CONNECTOR_MAP)
    if model.visual.connector.merged_norm is not None:
        mapping.update(_CONNECTOR_NORM_MAP)
    for dst, src in mapping.items():
        model.visual.connector.get_parameter(dst).copy_(get(src))


# --------------------------------------------------------------------------------------
# Quantization
# --------------------------------------------------------------------------------------


def smoothquant_groups(model: Lfm2VlForConditionalGeneration) -> list[list[QuantLinear]]:
    """Input-sharing linear groups across both towers.

    Encoder blocks reuse the shared ViT grouping (q/k/v share ln1; o/fc1/fc2 alone); the
    connector's two linears each have their own input; the text tower reuses the LFM2
    grouping (q/k/v, out_proj, the conv block's in/out projections, gate/up, down). The
    patch embed is deliberately left ungrouped, so the host pixel quantize stays a plain
    symmetric per-tensor scale.
    """
    groups = _vision_block_groups(model.visual)
    connector = model.visual.connector
    groups += [[connector.linear_fc1], [connector.linear_fc2]]
    return groups + _text_groups(model.text_model)


def quantize_model(model, calibration, *, smooth_alpha=0.8, qk_alpha=0.5) -> None:
    """Calibrate + enable W8A8 over the whole VLM in one pass.

    One joint pass matters here: the decoder's activation scales must see the image
    embeddings the vision tower actually produces, not text-only statistics.
    """
    _quantize_model(
        model,
        calibration,
        groups=smoothquant_groups(model),
        smooth_alpha=smooth_alpha,
        qk_alpha=qk_alpha,
    )


class Lfm2VlArchitecture(LLMArchitecture):
    """Step 1 for LFM2.5-VL: load + quantize the torch reference.

    The device side is split in two (see the module docstring): the vision tower is one
    graph of its own (vision_impl) and the decoder is the standard per-layer LLM artifact
    (text_impl), so this architecture only owns the quantized bundle they share.
    """

    name = "lfm2vl"

    def load_pretrained(self, repo_id, **kwargs):
        """Build the float reference.

        NB the generic ``llm compile`` path (generate.export_quantized_model) then
        calibrates on TEXT tokens only, which leaves the vision tower's observers empty
        and fails the export. Build an LFM2-VL artifact through
        ``pipeline.build_artifacts``, which calibrates both towers on real
        (image, question) pairs.
        """
        return Lfm2VlForConditionalGeneration.from_pretrained(repo_id, **kwargs)

    def quantize(self, model, calibration, *, smooth_alpha=0.8, qk_alpha=0.5):
        quantize_model(model, calibration, smooth_alpha=smooth_alpha, qk_alpha=qk_alpha)

    def record_lut_ranges(self, model, calibration):
        _record_lut_ranges(model, calibration)

    def num_layers(self, model):
        return model.config.text.num_hidden_layers

    def config_to_dict(self, model_config):
        return asdict(model_config)

    def config_from_dict(self, data):
        return Lfm2VlConfig(**data)

    def export_host_io(self, model):
        text = model.config.text
        host = {
            "embed_weight": model.text_model.embed_tokens.weight.detach()
            .cpu()
            .to(torch.float16)
            .numpy(),
            "final_norm_weight": model.text_model.embedding_norm.weight.detach()
            .cpu()
            .to(torch.float32)
            .numpy(),
            "rms_eps": np.float32(text.norm_eps),
            # Scalars the decode loop needs to build per-step RoPE / mask inputs.
            "head_dim": np.int64(text.head_dim),
            "num_q_heads": np.int64(text.num_attention_heads),
            "rope_theta": np.float64(text.rope_theta),
            "image_token_id": np.int64(model.config.image_token_id),
        }
        if not text.tie_word_embeddings:
            host["lm_head_weight"] = model.lm_head.weight.detach().cpu().to(torch.float16).numpy()
        return host


register_architecture(Lfm2VlArchitecture())

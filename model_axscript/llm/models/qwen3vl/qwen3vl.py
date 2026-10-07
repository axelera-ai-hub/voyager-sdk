# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Self-contained Qwen3-VL reference (vision tower + text tower) in PyTorch.

Qwen3-VL (https://arxiv.org/pdf/2511.21631) is a Qwen3 decoder with a ViT vision
tower bolted on. Three things distinguish it from "Qwen3 + a ViT":

  * **interleaved MRoPE** -- the decoder's rotary positions are 3-D (t, h, w).
    Instead of the Qwen2-VL chunked [TTT..HHH..WWW] frequency split, the sections
    are interleaved across the frequency axis, so every section keeps the full
    frequency spectrum. Image tokens get their (h, w) grid position; text tokens
    get t == h == w, which makes the decode step degenerate to ordinary 1-D RoPE.
  * **DeepStack** -- three intermediate vision-block outputs are merged and added
    into the FIRST three decoder layers' residual streams at the image-token
    positions (not just once at the input).
  * **timestamp-aligned video** -- a clip is a run of per-frame placeholder blocks
    separated by literal ``<X.X seconds>`` text, so the frames go through the vision
    tower exactly like images and the temporal position lives in those labels
    (split_video_grids, mrope_position_ids).

Everything else is Qwen3: GQA, QK-Norm, SwiGLU, RMSNorm. So the text tower reuses
``models.qwen3`` verbatim (Qwen3DecoderLayer) and the vision block reuses the
DINOv2 ViT block (``models.dino.reference.DinoBlock`` -- LayerNorm + separate
q/k/v + biased linears + GELU MLP, exactly Qwen3-VL's vision block modulo the
2-D RoPE this module adds, and modulo LayerScale which stays pinned at 1).

Layout of the pieces:
  qwen3vl.py     -- this file: configs, the torch fake-quant reference, HF loading,
                    the host-side geometry helpers (2-D vision RoPE, interpolated
                    position embedding, MRoPE position ids), and the architecture.
  vision_impl.py -- the alto-script vision kernels + the single-dispatch device graph.
  text_impl.py   -- the decoder implementation (the qwen3 optimized layer kernel
                    reused as-is) plus the VLM host hooks (MRoPE + image embeddings).

Sizes: the 2B tower (depth 24, hidden 1024, head_dim 64) packs cleanly onto PWORDs. The
8B tower (depth 27, hidden 1152, head_dim 72, intermediate 4304) has two widths that are
not PWORD multiples; both are zero-padded on device (see the padding geometry below).
The MoE variants (30B-A3B, 235B-A22B) need expert routing in the decoder and are rejected
with an explicit error.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...checkpoint import load_hf_safetensors
from ...modules import RMSNorm
from ...quant import ActivationObserver, QuantLinear, RangeObserver
from ...quant import quantize_model as _quantize_model
from ...quant import record_lut_ranges as _record_lut_ranges
from ..architecture import LLMArchitecture, register_architecture
from ..dino.reference import DinoBlock, DinoBlockConfig, QuantLayerNorm
from ..dino.reference import smoothquant_groups as _vision_block_groups
from ..qwen3.qwen3 import Qwen3Config, Qwen3DecoderLayer
from ..qwen3.qwen3 import smoothquant_groups as _text_groups

PWORD = 64
#: RoPE base of the vision tower's 2-D rotary embedding (transformers'
#: VisionRotaryEmbedding default; unrelated to the decoder's rope_theta).
VISION_ROPE_THETA = 10000.0


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


@dataclass
class Qwen3VLVisionConfig:
    """Vision-tower hyperparameters + the compile-pinned image geometry.

    Defaults are the Qwen3-VL-2B tower at a 256x256 image (a 16x16 patch grid,
    i.e. 256 tokens == 4 PWORDs of sequence, no padding).

    The grid is part of the config because the device graph is compiled for one
    token count: the patch count, the position-embedding constant, the 2-D RoPE
    tables and the attention mask are all baked at that geometry. A different
    resolution is a different artifact.
    """

    depth: int = 24
    hidden_size: int = 1024
    intermediate_size: int = 4096
    num_heads: int = 16
    in_channels: int = 3
    patch_size: int = 16
    temporal_patch_size: int = 2
    spatial_merge_size: int = 2
    out_hidden_size: int = 2048
    num_position_embeddings: int = 2304
    deepstack_visual_indexes: tuple[int, ...] = (5, 11, 17)
    layernorm_eps: float = 1e-6
    #: patch grid (rows, cols) of the compiled resolution -- image_size / patch_size.
    grid_h: int = 16
    grid_w: int = 16
    #: how a picture is forced onto that one grid: "letterbox" scales it to fit and pads the
    #: rest with mid-grey (which the 0.5/0.5 normalization sends to ~0), keeping the aspect
    #: ratio; "stretch" resizes anamorphically, using every patch at the cost of distorting.
    #: A BUILD-time property, so calibration sees the geometry inference will.
    fit_mode: str = "stretch"

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def patch_dim(self) -> int:
        """Flattened patch length K == C * T * P * P (the processor's feature axis)."""
        return self.in_channels * self.temporal_patch_size * self.patch_size**2

    @property
    def num_patches(self) -> int:
        return self.grid_h * self.grid_w

    @property
    def image_height(self) -> int:
        return self.grid_h * self.patch_size

    @property
    def image_width(self) -> int:
        return self.grid_w * self.patch_size

    @property
    def image_shape(self) -> tuple[int, int]:
        """(height, width) in pixels of the compiled geometry.

        The grid need not be square: everything geometric (2-D RoPE, the interpolated
        position embeddings, the merge order) is built from the grid_h/grid_w pair, so a
        288x512 picture compiles at its own aspect ratio as an 18x32 grid.
        """
        return self.image_height, self.image_width

    @property
    def image_geometry(self) -> str:
        """The compiled geometry for a log line: "256" if square, else "288x512" (HxW)."""
        height, width = self.image_shape
        return str(height) if height == width else f"{height}x{width}"

    @property
    def seq_len(self) -> int:
        """Patch count padded up to a PWORD multiple (the compiled token axis)."""
        return _round_up(self.num_patches, PWORD)

    @property
    def merge_unit(self) -> int:
        return self.spatial_merge_size**2

    @property
    def num_merged(self) -> int:
        """Image tokens the LLM sees (patches after the 2x2 spatial merge)."""
        return self.num_patches // self.merge_unit

    @property
    def num_grid_per_side(self) -> int:
        return int(round(self.num_position_embeddings**0.5))

    # --- device padding geometry ---------------------------------------------------
    # Two of the 8B tower's dimensions are not PWORD multiples: head_dim 72 and
    # intermediate 4304. Both are zero-padded on device rather than reshaped.
    #
    # head_dim pads PER HALF, because RoPE rotates x1 against x2: head_dim 72 (halves of
    # 36) becomes 128, with x1 in block 0 and x2 in block 1. That is exactly what
    # axol.core.rope consumes, so padding the head also replaces the 2B tower's
    # lane-swizzle RoPE with the plain block-level op. The padded lanes carry zeros end to
    # end and o_proj's padded input columns are zero, so nothing downstream sees them.

    @property
    def padded_head_dim(self) -> int:
        """head_dim rounded so each RoPE half fills whole PWORDs (== head_dim if aligned)."""
        if self.head_dim % PWORD == 0:
            return self.head_dim
        return 2 * _round_up(self.head_dim // 2, PWORD)

    @property
    def head_dim_pw(self) -> int:
        return self.padded_head_dim // PWORD

    @property
    def pads_head_dim(self) -> bool:
        return self.padded_head_dim != self.head_dim

    @property
    def qkv_dim(self) -> int:
        """Width of the q/k/v projections on device (padded heads, laid out head-major)."""
        return self.num_heads * self.padded_head_dim

    @property
    def padded_intermediate(self) -> int:
        """MLP width on device: the checkpoint's, rounded up to a PWORD multiple."""
        return _round_up(self.intermediate_size, PWORD)

    def validate(self) -> None:
        if self.head_dim % 2:
            raise NotImplementedError(
                f"vision head_dim must be even (RoPE rotates its halves); got {self.head_dim}"
            )
        if self.head_dim % PWORD and self.head_dim // 2 > PWORD:
            raise NotImplementedError(
                f"vision head_dim {self.head_dim}: an unaligned half wider than {PWORD} lanes "
                "would need multi-block RoPE halves, which the padding does not cover"
            )
        for name in ("hidden_size", "out_hidden_size", "patch_dim"):
            value = getattr(self, name)
            if value % PWORD:
                raise NotImplementedError(f"vision {name}={value} must be a multiple of {PWORD}")
        if self.grid_h % self.spatial_merge_size or self.grid_w % self.spatial_merge_size:
            raise ValueError(
                f"patch grid {self.grid_h}x{self.grid_w} must divide by the spatial merge "
                f"size {self.spatial_merge_size}"
            )

    def block_config(self) -> DinoBlockConfig:
        """The DinoBlockConfig describing one vision block (shape only)."""
        return DinoBlockConfig(
            hidden_size=self.hidden_size,
            num_heads=self.num_heads,
            mlp_ratio=self.intermediate_size // self.hidden_size,
            # 8B: intermediate 4304 is not a whole multiple of hidden 1152, and head_dim
            # 72 is not a PWORD -- the reference runs both unpadded, the device pads.
            mlp_hidden_size=self.intermediate_size,
            head_dim_is_pword=self.head_dim % PWORD == 0,
            seq_len=self.seq_len,
            layernorm_eps=self.layernorm_eps,
            # DinoBlockConfig.image_size feeds dino's own conv patch-embed only; Qwen3-VL
            # projects patches the processor already flattened, so nothing here reads it.
            # Recorded as the height because one number cannot describe a non-square grid.
            image_size=self.image_height,
            patch_size=self.patch_size,
            num_channels=self.in_channels,
        )

    @classmethod
    def from_hf_dict(cls, cfg: dict, *, grid_h: int, grid_w: int) -> "Qwen3VLVisionConfig":
        return cls(
            depth=cfg["depth"],
            hidden_size=cfg["hidden_size"],
            intermediate_size=cfg["intermediate_size"],
            num_heads=cfg["num_heads"],
            in_channels=cfg.get("in_channels", 3),
            patch_size=cfg["patch_size"],
            temporal_patch_size=cfg.get("temporal_patch_size", 2),
            spatial_merge_size=cfg["spatial_merge_size"],
            out_hidden_size=cfg["out_hidden_size"],
            num_position_embeddings=cfg.get("num_position_embeddings", 2304),
            deepstack_visual_indexes=tuple(cfg["deepstack_visual_indexes"]),
            grid_h=grid_h,
            grid_w=grid_w,
        )


@dataclass
class Qwen3VLConfig:
    """The whole model: a Qwen3 text config + the vision config + the token ids."""

    text: Qwen3Config = field(default_factory=Qwen3Config)
    vision: Qwen3VLVisionConfig = field(default_factory=Qwen3VLVisionConfig)
    image_token_id: int = 151655
    video_token_id: int = 151656
    vision_start_token_id: int = 151652
    vision_end_token_id: int = 151653
    #: Per-section frequency budget of the interleaved MRoPE (t, h, w); sums to head_dim/2.
    mrope_section: tuple[int, int, int] = (24, 20, 20)

    @property
    def num_hidden_layers(self) -> int:
        """Decoder depth -- what the layer-per-kind machinery counts."""
        return self.text.num_hidden_layers

    def __post_init__(self) -> None:
        # dataclass fields survive a JSON round-trip as plain dicts/lists; rebuild them.
        if isinstance(self.text, dict):
            self.text = Qwen3Config(**self.text)
        if isinstance(self.vision, dict):
            self.vision = Qwen3VLVisionConfig(**self.vision)
        self.vision.deepstack_visual_indexes = tuple(self.vision.deepstack_visual_indexes)
        self.mrope_section = tuple(self.mrope_section)

    @classmethod
    def from_hf_dict(cls, cfg: dict, *, grid_h: int, grid_w: int) -> "Qwen3VLConfig":
        text_cfg = dict(cfg["text_config"])
        rope_scaling = text_cfg.get("rope_scaling") or {}
        if not rope_scaling.get("mrope_interleaved", True):
            raise NotImplementedError("only interleaved MRoPE (Qwen3-VL) is supported")
        if text_cfg.get("num_experts"):
            raise NotImplementedError(
                "MoE decoders (Qwen3-VL-30B-A3B / 235B-A22B) need expert routing in the "
                "layer kernel; only the dense variants (2B, 8B) are supported"
            )
        # The text tower is a plain Qwen3 stack; reuse its config verbatim.
        text = Qwen3Config(
            vocab_size=text_cfg["vocab_size"],
            hidden_size=text_cfg["hidden_size"],
            intermediate_size=text_cfg["intermediate_size"],
            num_hidden_layers=text_cfg["num_hidden_layers"],
            num_attention_heads=text_cfg["num_attention_heads"],
            num_key_value_heads=text_cfg["num_key_value_heads"],
            head_dim=text_cfg["head_dim"],
            max_position_embeddings=text_cfg["max_position_embeddings"],
            rms_norm_eps=text_cfg["rms_norm_eps"],
            rope_theta=text_cfg["rope_theta"],
            tie_word_embeddings=bool(cfg.get("tie_word_embeddings", False)),
        )
        return cls(
            text=text,
            vision=Qwen3VLVisionConfig.from_hf_dict(
                cfg["vision_config"], grid_h=grid_h, grid_w=grid_w
            ),
            image_token_id=cfg.get("image_token_id", 151655),
            video_token_id=cfg.get("video_token_id", 151656),
            vision_start_token_id=cfg.get("vision_start_token_id", 151652),
            vision_end_token_id=cfg.get("vision_end_token_id", 151653),
            mrope_section=tuple(rope_scaling.get("mrope_section", (24, 20, 20))),
        )


# --------------------------------------------------------------------------------------
# Host-side geometry: everything the device graph bakes in or the runtime feeds it.
# Pure functions of the config (+ the checkpoint's pos_embed table), so both the torch
# reference and the compiled pipeline call the same code.
# --------------------------------------------------------------------------------------


def merge_block_order(grid_h: int, grid_w: int, merge: int) -> torch.Tensor:
    """Row/col index of each patch in the processor's merge-block-major token order.

    The image processor emits patches grouped by 2x2 merge block ((block row, block
    col, intra row, intra col)), because the merger later folds each consecutive
    group of `merge**2` tokens into one. Returns (num_patches, 2) int64 of the
    full-resolution (row, col) of each token, matching
    Qwen3VLVisionModel.rot_pos_emb.
    """
    mh, mw = grid_h // merge, grid_w // merge
    rows = torch.arange(mh)[:, None, None, None] * merge + torch.arange(merge)[None, None, :, None]
    cols = torch.arange(mw)[None, :, None, None] * merge + torch.arange(merge)[None, None, None, :]
    rows = rows.expand(mh, mw, merge, merge).reshape(-1)
    cols = cols.expand(mh, mw, merge, merge).reshape(-1)
    return torch.stack((rows, cols), dim=-1)


def vision_rope_cos_sin(cfg: Qwen3VLVisionConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """The vision tower's 2-D rotary tables, (num_patches, head_dim) each.

    head_dim/4 frequencies for the row position and head_dim/4 for the column
    position are concatenated into the head_dim/2 unique frequencies, then
    duplicated (the half-rotation convention: cos == cat(f, f)).
    """
    dim = cfg.head_dim // 2  # VisionRotaryEmbedding(head_dim // 2)
    inv_freq = 1.0 / (VISION_ROPE_THETA ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    pos_ids = merge_block_order(cfg.grid_h, cfg.grid_w, cfg.spatial_merge_size)
    table = torch.outer(torch.arange(max(cfg.grid_h, cfg.grid_w), dtype=torch.float32), inv_freq)
    freqs = table[pos_ids].flatten(1)  # (num_patches, head_dim/2)
    emb = torch.cat((freqs, freqs), dim=-1)  # (num_patches, head_dim)
    return emb.cos(), emb.sin()


def interpolated_pos_embed(cfg: Qwen3VLVisionConfig, pos_embed_weight: torch.Tensor):
    """Bilinearly resample the learned (48x48) position grid onto this patch grid.

    Port of Qwen3VLVisionModel.fast_pos_embed_interpolate for a single image
    (t == 1), returning (num_patches, hidden) already permuted into the
    merge-block-major token order.
    """
    side = cfg.num_grid_per_side
    h, w, merge = cfg.grid_h, cfg.grid_w, cfg.spatial_merge_size
    h_idx = torch.linspace(0, side - 1, h)
    w_idx = torch.linspace(0, side - 1, w)
    h_floor, w_floor = h_idx.int(), w_idx.int()
    h_ceil = (h_floor + 1).clip(max=side - 1)
    w_ceil = (w_floor + 1).clip(max=side - 1)
    dh, dw = h_idx - h_floor, w_idx - w_floor
    base, base_ceil = h_floor * side, h_ceil * side
    corners = [
        (base[None].T + w_floor[None]).flatten(),
        (base[None].T + w_ceil[None]).flatten(),
        (base_ceil[None].T + w_floor[None]).flatten(),
        (base_ceil[None].T + w_ceil[None]).flatten(),
    ]
    weights = [
        ((1 - dh)[None].T * (1 - dw)[None]).flatten(),
        ((1 - dh)[None].T * dw[None]).flatten(),
        (dh[None].T * (1 - dw)[None]).flatten(),
        (dh[None].T * dw[None]).flatten(),
    ]
    pos = sum(
        pos_embed_weight[idx.long()] * weight[:, None] for idx, weight in zip(corners, weights)
    )  # (h*w, hidden), row-major patch order
    # Row-major -> merge-block-major (the processor's token order).
    return (
        pos.view(h // merge, merge, w // merge, merge, -1)
        .permute(0, 2, 1, 3, 4)
        .reshape(h * w, -1)
        .contiguous()
    )


def split_video_grids(grids) -> list[tuple[int, int, int]]:
    """A video's (t, h, w) -> t per-frame (1, h, w) grids.

    Qwen3-VL separates video frames with timestamp TEXT tokens
    (``<0.5 seconds><vision_start>…<vision_end>``), so each frame is its own vision
    block with its own placeholder run and t == 1; the temporal position lives in
    those timestamps, not in the rotary t section. Same first step as
    Qwen3VLModel.get_rope_index.
    """
    return [(1, int(h), int(w)) for t, h, w in grids for _ in range(int(t))]


def join_vision_rows(vision_mask: torch.Tensor, parts) -> torch.Tensor:
    """Interleave per-modality vision rows into token order.

    ``parts`` is [(modality mask, rows)]. Each modality's rows land at its positions WITHIN
    the union of vision positions: concatenating all image rows then all video rows is only
    right when every image precedes every video, which a prompt is free not to do. Same
    construction as HF's joint DeepStack buffer (Qwen3VLModel.forward).
    """
    reference = parts[0][1]
    joint = reference.new_zeros((int(vision_mask.sum()), reference.shape[-1]))
    for mask, rows in parts:
        joint[mask[vision_mask]] = rows.reshape(-1, joint.shape[-1]).to(joint.dtype)
    return joint


def mrope_position_ids(
    input_ids,
    cfg: Qwen3VLConfig,
    grids: list[tuple[int, int, int]],
    video_grids: list[tuple[int, int, int]] | None = None,
) -> tuple[np.ndarray, int]:
    """3-D (t, h, w) rotary positions for one prompt, plus the decode-time delta.

    Port of Qwen3VLModel.get_rope_index for a single un-padded sequence, images and
    videos. Text runs advance all three sections together; a vision block occupies one
    t-step and lays its merged grid out over (h, w). Blocks are consumed in the order
    their placeholders appear, each from its own modality's grid list -- exactly HF's
    two-index walk. ``video_grids`` are the processor's per-video (t, h, w), split into
    frames here. Returns (position_ids (3, T) int64, delta) where the next generated
    token sits at absolute position ``len(input_ids) + delta`` in all three sections.
    """
    # Through torch rather than np.asarray: calibration runs on the GPU where there is one,
    # and numpy refuses a CUDA tensor outright. The positions themselves stay on the host by
    # design, and the caller moves the tables built from them onto the model's device.
    ids = torch.as_tensor(input_ids, dtype=torch.int64).reshape(-1).tolist()
    merge = cfg.vision.spatial_merge_size
    queues = {
        cfg.image_token_id: list(grids or []),
        cfg.video_token_id: split_video_grids(video_grids or []),
    }
    chunks: list[torch.Tensor] = []
    start = 0
    while True:
        # Whichever modality's placeholder comes first (and still has a grid left).
        nexts = [
            (ids.index(token, start), token)
            for token, queue in queues.items()
            if queue and token in ids[start:]
        ]
        if not nexts:
            break
        end, token = min(nexts)
        t, h, w = queues[token].pop(0)
        gh, gw = h // merge, w // merge
        text_len = end - start
        offset = int(chunks[-1].max()) + 1 if chunks else 0
        chunks.append(torch.arange(text_len).view(1, -1).expand(3, -1) + offset)
        # t_index is 0 for a single frame; h/w index the merged grid.
        t_idx = torch.arange(t).view(-1, 1).expand(-1, gh * gw).flatten()
        h_idx = torch.arange(gh).view(1, -1, 1).expand(t, -1, gw).flatten()
        w_idx = torch.arange(gw).view(1, 1, -1).expand(t, gh, -1).flatten()
        chunks.append(torch.stack([t_idx, h_idx, w_idx]) + text_len + offset)
        start = end + t * gh * gw
    if start < len(ids):
        offset = int(chunks[-1].max()) + 1 if chunks else 0
        chunks.append(torch.arange(len(ids) - start).view(1, -1).expand(3, -1) + offset)
    positions = torch.cat(chunks, dim=1).reshape(3, -1)
    if positions.shape[1] != len(ids):
        raise ValueError(
            f"built {positions.shape[1]} positions for {len(ids)} tokens: the vision "
            "placeholder runs do not match image_grid_thw / video_grid_thw"
        )
    delta = int(positions.max()) + 1 - len(ids)
    return positions.numpy(), delta


def mrope_cos_sin(
    position_ids: np.ndarray, head_dim: int, rope_theta: float, mrope_section
) -> tuple[np.ndarray, np.ndarray]:
    """cos/sin over the head_dim/2 unique frequencies -- the device rope layout.

    Interleaved MRoPE: section t owns every frequency index by default; sections h and w
    take back indices ``offset::3`` up to ``3 * mrope_section[i]`` (offset 1 for h, 2 for
    w). So the three positional axes are woven through the frequency spectrum rather than
    split into three contiguous bands (Qwen2-VL's chunked layout).
    """
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    pos = np.asarray(position_ids, dtype=np.float64).reshape(3, -1)
    freqs = pos[:, :, None] * inv_freq[None, None, :]  # (3, T, head_dim/2)
    angles = freqs[0].copy()
    for dim, offset in enumerate((1, 2), start=1):
        idx = slice(offset, mrope_section[dim] * 3, 3)
        angles[..., idx] = freqs[dim][..., idx]
    return np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)


# --------------------------------------------------------------------------------------
# Torch fake-quant reference
# --------------------------------------------------------------------------------------


class Qwen3VLPatchEmbed(nn.Module):
    """The Conv3d patch projection as a QuantLinear over the processor's flat patches.

    Stride == kernel, so the conv is exactly a dense matmul over each patch's
    C*T*P*P features (which is the layout the image processor already emits).
    Input quant is symmetric (zp = 0) so the host can quantize pixels with a
    single scale, and the conv bias is added after dequant -- both matching the
    device kernel (see vision_impl.make_vision_embed_kernel).
    """

    def __init__(self, cfg: Qwen3VLVisionConfig, **quant_kwargs) -> None:
        super().__init__()
        self.proj = QuantLinear(cfg.patch_dim, cfg.hidden_size, act_symmetric=True, **quant_kwargs)
        self.bias = nn.Parameter(torch.zeros(cfg.hidden_size))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # (num_patches, patch_dim) -> (1, num_patches, hidden)
        return self.proj(patches.unsqueeze(0)) + self.bias


class Qwen3VLPatchMerger(nn.Module):
    """LayerNorm -> 2x2 spatial merge -> GELU MLP into the decoder's hidden size.

    postshuffle=False (the output merger) normalises per vision token and then
    folds 4 tokens into one; postshuffle=True (the DeepStack mergers) folds first
    and normalises the merged 4*hidden vector.
    """

    def __init__(
        self, cfg: Qwen3VLVisionConfig, *, postshuffle: bool = False, **quant_kwargs
    ) -> None:
        super().__init__()
        dim = cfg.hidden_size * cfg.merge_unit
        self.merge_unit = cfg.merge_unit
        self.postshuffle = postshuffle
        self.norm = QuantLayerNorm(dim if postshuffle else cfg.hidden_size, cfg.layernorm_eps)
        self.linear_fc1 = QuantLinear(dim, dim, **quant_kwargs)
        self.fc1_bias = nn.Parameter(torch.zeros(dim))
        self.gelu_range = RangeObserver("gelu")
        self.linear_fc2 = QuantLinear(dim, cfg.out_hidden_size, **quant_kwargs)
        self.fc2_bias = nn.Parameter(torch.zeros(cfg.out_hidden_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq, hidden = x.shape
        merged = (batch, seq // self.merge_unit, hidden * self.merge_unit)
        x = self.norm(x.reshape(merged)) if self.postshuffle else self.norm(x).reshape(merged)
        # nn.GELU() in HF: the exact (erf) form, unlike the block MLP's tanh approximation.
        x = F.gelu(self.gelu_range(self.linear_fc1(x) + self.fc1_bias))
        return self.linear_fc2(x) + self.fc2_bias


class Qwen3VLVisionBlock(DinoBlock):
    """A DINOv2 ViT block with 2-D RoPE on q/k -- the Qwen3-VL vision block.

    Structurally identical to DinoBlock (LayerNorm, separate biased q/k/v, bidirectional
    attention, biased GELU MLP), so it inherits the module layout the fused device kernel
    and its binder are written against. LayerScale (ls1/ls2) stays at its 1.0 init:
    Qwen3-VL has none, and the binder folds it into a dequant scale where it disappears.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # DinoBlock records this range for the exact (erf) GELU table; HF's Qwen3-VL block
        # MLP evaluates the TANH approximation, so fit that curve instead. The observer's
        # function name is what layout.fit_lut builds the device table from, so renaming it
        # is the whole change -- and DINO (and this model's merger) stay on erf.
        self.gelu_range = RangeObserver("gelu_tanh")

    def forward(self, hidden: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
        cfg = self.config
        x = self.ln1(hidden)
        q = self._heads(self.q_proj(x) + self.q_bias)
        k = self._heads(self.k_proj(x) + self.k_bias)
        v = self._heads(self.v_proj(x) + self.v_bias)
        q, k = apply_vision_rope(q, k, cos, sin)
        attn = self.sdpa(q, k, v, None)  # bidirectional: every patch sees every patch
        batch, _, seq, _ = attn.shape
        attn = attn.transpose(1, 2).reshape(batch, seq, cfg.hidden_size)
        res1 = hidden + (self.o_proj(attn) + self.o_bias) * self.ls1
        y = self.ln2(res1)
        # gelu_pytorch_tanh, as HF does here, and the device fits the same curve via the
        # "gelu_tanh" LUT this block's observer names (see __init__). The two forms are
        # ~5e-4 apart, under the table's own fit error, so this buys no accuracy -- it buys
        # the device and the reference evaluating ONE curve.
        h1 = F.gelu(self.gelu_range(self.fc1(y) + self.fc1_bias), approximate="tanh")
        return res1 + (self.fc2(h1) + self.fc2_bias) * self.ls2


def apply_vision_rope(q, k, cos, sin):
    """2-D RoPE for (1, heads, seq, head_dim) q/k with (seq, head_dim) tables."""
    cos = cos[None, None].to(q.dtype)
    sin = sin[None, None].to(q.dtype)

    def rotate_half(x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    return q * cos + rotate_half(q) * sin, k * cos + rotate_half(k) * sin


class Qwen3VLVisionModel(nn.Module):
    """Patch embed (+ interpolated position embedding) -> N blocks -> patch merger.

    Returns the merged image embeddings the decoder consumes, plus the DeepStack
    features taken at ``deepstack_visual_indexes``. Geometry is fixed by the
    config, so the whole tower is a function of the flattened patches alone --
    which is also what makes it callable straight from the calibration loop.
    """

    def __init__(self, cfg: Qwen3VLVisionConfig, **quant_kwargs) -> None:
        super().__init__()
        cfg.validate()
        self.config = cfg
        block_cfg = cfg.block_config()
        self.patch_embed = Qwen3VLPatchEmbed(cfg, **quant_kwargs)
        self.blocks = nn.ModuleList(
            [Qwen3VLVisionBlock(block_cfg, **quant_kwargs) for _ in range(cfg.depth)]
        )
        self.merger = Qwen3VLPatchMerger(cfg, postshuffle=False, **quant_kwargs)
        self.deepstack_mergers = nn.ModuleList(
            [
                Qwen3VLPatchMerger(cfg, postshuffle=True, **quant_kwargs)
                for _ in cfg.deepstack_visual_indexes
            ]
        )
        cos, sin = vision_rope_cos_sin(cfg)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)
        # Filled by load_hf_weights (the resampled position grid for this resolution).
        self.register_buffer(
            "pos_embed", torch.zeros(cfg.num_patches, cfg.hidden_size), persistent=False
        )

    def _encode_one(self, patches: torch.Tensor):
        """One image's patches -> (merged, [deepstack…]), all (1, tokens, out_hidden)."""
        cfg = self.config
        hidden = self.patch_embed(patches) + self.pos_embed
        deepstack = []
        for i, block in enumerate(self.blocks):
            hidden = block(hidden, self.rope_cos, self.rope_sin)
            if i in cfg.deepstack_visual_indexes:
                slot = cfg.deepstack_visual_indexes.index(i)
                deepstack.append(self.deepstack_mergers[slot](hidden))
        return self.merger(hidden), deepstack

    def forward(self, patches: torch.Tensor):
        """(num_images * num_patches, patch_dim) -> merged embeddings + DeepStack features.

        Images are encoded ONE AT A TIME and concatenated, which is not a shortcut:
        HF blocks vision attention per image via cu_seqlens, so a patch never
        attends across an image boundary. Every image shares the compiled geometry,
        hence the same position embedding and rotary tables.
        """
        cfg = self.config
        if patches.shape[0] % cfg.num_patches:
            raise ValueError(
                f"{patches.shape[0]} patch rows is not a whole number of "
                f"{cfg.num_patches}-patch images"
            )
        chunks = [
            self._encode_one(patches[i : i + cfg.num_patches])
            for i in range(0, patches.shape[0], cfg.num_patches)
        ]
        merged = torch.cat([m for m, _ in chunks], dim=1)
        deepstack = [
            torch.cat([ds[slot] for _, ds in chunks], dim=1)
            for slot in range(len(cfg.deepstack_visual_indexes))
        ]
        return merged, deepstack


class Qwen3VLTextModel(nn.Module):
    """The Qwen3 decoder stack, driven by pre-computed MRoPE tables.

    Identical to models.qwen3.Qwen3Model except that (a) it takes embeddings
    rather than ids (the caller splices in the image embeddings) and rotary
    tables rather than positions (MRoPE lives on the host), and (b) it adds the
    DeepStack features into the first ``len(deepstack)`` layers' outputs at the
    image-token positions.
    """

    def __init__(self, cfg: Qwen3Config) -> None:
        super().__init__()
        self.config = cfg
        self.embed_tokens = nn.Embedding(cfg.vocab_size, cfg.hidden_size)
        self.layers = nn.ModuleList([Qwen3DecoderLayer(cfg) for _ in range(cfg.num_hidden_layers)])
        self.norm = RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps)

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        *,
        deepstack: list[torch.Tensor] | None = None,
        image_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = inputs_embeds.shape[1]
        dtype, device = inputs_embeds.dtype, inputs_embeds.device
        mask = torch.full((seq_len, seq_len), torch.finfo(dtype).min, dtype=dtype, device=device)
        mask = torch.triu(mask, diagonal=1)[None, None]
        hidden = inputs_embeds
        for i, layer in enumerate(self.layers):
            hidden = layer(hidden, cos, sin, mask)
            if deepstack and i < len(deepstack):
                hidden = hidden.clone()
                hidden[image_mask] = hidden[image_mask] + deepstack[i].reshape(
                    -1, hidden.shape[-1]
                )
        return self.norm(hidden)


class Qwen3VLForConditionalGeneration(nn.Module):
    """The full VLM: vision tower + Qwen3 decoder + (tied) LM head.

    ``forward`` takes ONE sample dict so the shared PTQ toolkit (which calls
    ``module(batch)``) can calibrate the vision and text towers jointly on real
    multimodal inputs:

        {"input_ids": (1, T) int64,
         "patches":   (num_patches, patch_dim) float or None,
         "grids":     [(t, h, w), ...]}
    """

    def __init__(self, config: Qwen3VLConfig, **quant_kwargs) -> None:
        super().__init__()
        self.config = config
        self.visual = Qwen3VLVisionModel(config.vision, **quant_kwargs)
        self.text_model = Qwen3VLTextModel(config.text)
        self.lm_head = nn.Linear(config.text.hidden_size, config.text.vocab_size, bias=False)
        if config.text.tie_word_embeddings:
            self.lm_head.weight = self.text_model.embed_tokens.weight

    @property
    def device(self) -> torch.device:
        return self.text_model.embed_tokens.weight.device

    def input_embeddings(self, sample: dict):
        """(embeds, deepstack, vision_mask, position_ids) for one sample.

        Images and videos go through the SAME tower -- a video's temporal patches are
        independent frame encodings (HF blocks vision attention per frame via cu_seqlens),
        so the only difference is which placeholder rows the result lands on. The DeepStack
        features follow the same order, so the per-layer injection ignores modality.
        """
        cfg = self.config
        input_ids = torch.as_tensor(sample["input_ids"]).reshape(1, -1).to(self.device)
        embeds = self.text_model.embed_tokens(input_ids)
        vision_mask = torch.zeros_like(input_ids, dtype=torch.bool)
        encoded: list[tuple[torch.Tensor, list[torch.Tensor]]] = []
        for key, token_id in (
            ("patches", cfg.image_token_id),
            ("video_patches", cfg.video_token_id),
        ):
            patches = sample.get(key)
            if patches is None:
                continue
            mask = input_ids == token_id
            merged, features = self.visual(patches.to(device=self.device, dtype=embeds.dtype))
            embeds = embeds.masked_scatter(
                mask.unsqueeze(-1).expand_as(embeds), merged.reshape(-1, embeds.shape[-1])
            )
            vision_mask = vision_mask | mask
            encoded.append((mask, features))
        deepstack = [
            join_vision_rows(vision_mask, [(mask, features[slot]) for mask, features in encoded])
            for slot in range(len(encoded[0][1]) if encoded else 0)
        ]
        position_ids, _ = mrope_position_ids(
            input_ids[0], cfg, sample.get("grids") or [], sample.get("video_grids") or []
        )
        return embeds, deepstack, vision_mask, position_ids

    def forward(self, sample) -> torch.Tensor:
        cfg = self.config
        if torch.is_tensor(sample):  # text-only convenience form
            sample = {"input_ids": sample}
        embeds, deepstack, image_mask, position_ids = self.input_embeddings(sample)
        cos, sin = mrope_cos_sin(
            position_ids, cfg.text.head_dim, cfg.text.rope_theta, cfg.mrope_section
        )
        # The tables hold the head_dim/2 unique frequencies; torch RoPE wants the duplicated
        # head_dim form. Built in numpy, so they follow the model (calibration may be on GPU).
        to = {"device": embeds.device, "dtype": embeds.dtype}
        cos_t = torch.from_numpy(np.concatenate([cos, cos], -1))[None].to(**to)
        sin_t = torch.from_numpy(np.concatenate([sin, sin], -1))[None].to(**to)
        hidden = self.text_model(embeds, cos_t, sin_t, deepstack=deepstack, image_mask=image_mask)
        return self.lm_head(hidden)

    @classmethod
    def from_pretrained(
        cls,
        repo_id: str = "Qwen/Qwen3-VL-2B-Instruct",
        *,
        image_size: int | tuple[int, int] = 256,
        dtype: torch.dtype = torch.float32,
        device: str | torch.device = "cpu",
        quantize_attention: bool = True,
        **config_overrides,
    ) -> "Qwen3VLForConditionalGeneration":
        """Load a HF Qwen3-VL checkpoint, pinning the vision tower to image_size.

        quantize_attention fake-quantizes the DECODER's SDPA (the vision tower's is
        always quantized). It defaults on because the device layer kernel binds
        calibrated SDPA scales -- an unquantized reference cannot fill them.

        ``repo_id`` is a hub id or a local checkpoint directory -- custom weights are
        the same call with a path, as they are everywhere else in HuggingFace.
        """
        config = config_from_pretrained(repo_id, image_size=image_size)
        config = replace(config, text=replace(config.text, quantize_attention=quantize_attention))
        if config_overrides:
            config = replace(config, **config_overrides)
        model = cls(config).to(device=device, dtype=dtype)
        # The checkpoint is staged on the HOST and in its OWN dtype (bf16, not `dtype`): the
        # parameters are already allocated, so every load is a widening copy_ and an upcast
        # would only buy a second full-size model. At 8B that is 32 GB on top of 32 GB --
        # enough to get a 31 GB host OOM-killed mid-load.
        model.load_weights(repo_id)
        return model.eval()

    def load_weights(self, repo_id: str) -> "Qwen3VLForConditionalGeneration":
        """Load a HF checkpoint's weights into this (already configured) model.

        The weight half of `from_pretrained`, on its own, so a caller can swap in their
        own finetune -- a hub id or a local directory of the SAME architecture -- without
        rebuilding the module. `from_pretrained` is this plus the config.
        """
        load_hf_weights(self, load_hf_safetensors(repo_id, device="cpu", dtype=None))
        return self


def patch_grid(image_size: int | tuple[int, int], patch: int) -> tuple[int, int]:
    """(grid_h, grid_w) for a square side or an (H, W) pair of pixel sizes.

    The one spelling of the geometry argument every entry point takes. Both sides must
    divide by `patch`; that the GRID divides by the spatial merge size is `validate`'s job.
    """
    height, width = (image_size, image_size) if isinstance(image_size, int) else image_size
    for name, value in (("height", height), ("width", width)):
        if value <= 0 or value % patch:
            raise ValueError(
                f"image {name} {value} must be a positive multiple of patch_size {patch}"
            )
    return height // patch, width // patch


def config_from_pretrained(
    repo_id: str, *, image_size: int | tuple[int, int] = 256
) -> Qwen3VLConfig:
    """The config for a HF repo or local checkpoint dir at a pinned geometry (no weights).

    `image_size` is a square side or an (height, width) pair. Cheap enough for the
    host-side tools that need the geometry without loading 2B parameters.
    """
    import json

    from ...checkpoint import repo_file

    with open(repo_file(repo_id, "config.json")) as f:
        hf_config = json.load(f)
    grid_h, grid_w = patch_grid(image_size, hf_config["vision_config"]["patch_size"])
    return Qwen3VLConfig.from_hf_dict(hf_config, grid_h=grid_h, grid_w=grid_w)


# --------------------------------------------------------------------------------------
# HuggingFace checkpoint -> reference module
# --------------------------------------------------------------------------------------

#: vision block parameter (relative) -> HF name under model.visual.blocks.<i>.
_BLOCK_MAP = {
    "ln1.weight": "norm1.weight",
    "ln1.bias": "norm1.bias",
    "ln2.weight": "norm2.weight",
    "ln2.bias": "norm2.bias",
    "o_proj.weight": "attn.proj.weight",
    "o_bias": "attn.proj.bias",
    "fc1.weight": "mlp.linear_fc1.weight",
    "fc1_bias": "mlp.linear_fc1.bias",
    "fc2.weight": "mlp.linear_fc2.weight",
    "fc2_bias": "mlp.linear_fc2.bias",
}
#: merger parameter (relative) -> HF name under the merger module.
_MERGER_MAP = {
    "norm.weight": "norm.weight",
    "norm.bias": "norm.bias",
    "linear_fc1.weight": "linear_fc1.weight",
    "fc1_bias": "linear_fc1.bias",
    "linear_fc2.weight": "linear_fc2.weight",
    "fc2_bias": "linear_fc2.bias",
}


@torch.no_grad()
def load_hf_weights(model: Qwen3VLForConditionalGeneration, state: dict) -> None:
    """Copy a HF Qwen3-VL state dict into the reference model.

    The text tower maps 1:1 (a plain prefix rename); the vision tower needs three
    rewrites: the Conv3d patch kernel flattens to a matmul, the fused qkv weight
    splits into three, and the learned position grid is resampled to this
    resolution.
    """
    cfg = model.config
    vis = cfg.vision

    def get(name: str) -> torch.Tensor:
        try:
            return state[name]
        except KeyError:
            raise KeyError(f"missing checkpoint tensor {name!r}") from None

    # --- text tower: model.language_model.* -> text_model.* (identical leaf layout).
    text_state = {
        key[len("model.language_model.") :]: value
        for key, value in state.items()
        if key.startswith("model.language_model.")
    }
    missing, unexpected = model.text_model.load_state_dict(text_state, strict=False)
    if unexpected:
        raise RuntimeError(f"unexpected text-tower keys: {unexpected}")
    # Observers/range recorders own non-persistent buffers absent from any checkpoint.
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

    # --- patch embed: Conv3d (hidden, C, T, P, P) -> (hidden, C*T*P*P) matmul.
    conv = get("model.visual.patch_embed.proj.weight")
    model.visual.patch_embed.proj.weight.copy_(conv.reshape(conv.shape[0], -1))
    model.visual.patch_embed.bias.copy_(get("model.visual.patch_embed.proj.bias"))
    model.visual.pos_embed.copy_(interpolated_pos_embed(vis, get("model.visual.pos_embed.weight")))

    # --- blocks: split the fused qkv, then the 1:1 names.
    hidden = vis.hidden_size
    for i, block in enumerate(model.visual.blocks):
        prefix = f"model.visual.blocks.{i}."
        qkv_w = get(prefix + "attn.qkv.weight")
        qkv_b = get(prefix + "attn.qkv.bias")
        for slot, proj in enumerate(("q", "k", "v")):
            rows = slice(slot * hidden, (slot + 1) * hidden)
            getattr(block, f"{proj}_proj").weight.copy_(qkv_w[rows])
            getattr(block, f"{proj}_bias").copy_(qkv_b[rows])
        for dst, src in _BLOCK_MAP.items():
            block.get_parameter(dst).copy_(get(prefix + src))

    # --- mergers (output + DeepStack).
    for dst, src in _MERGER_MAP.items():
        model.visual.merger.get_parameter(dst).copy_(get("model.visual.merger." + src))
    for i, merger in enumerate(model.visual.deepstack_mergers):
        for dst, src in _MERGER_MAP.items():
            merger.get_parameter(dst).copy_(get(f"model.visual.deepstack_merger_list.{i}." + src))


# --------------------------------------------------------------------------------------
# Quantization
# --------------------------------------------------------------------------------------


def smoothquant_groups(model: Qwen3VLForConditionalGeneration) -> list[list[QuantLinear]]:
    """Input-sharing linear groups across both towers.

    Vision blocks reuse the DINOv2 grouping (q/k/v share ln1; o/fc1/fc2 alone);
    the mergers' two linears each have their own input; the text tower reuses the
    Qwen3 grouping (q/k/v, o, gate/up, down).
    """
    groups = _vision_block_groups(model.visual)
    for merger in [model.visual.merger, *model.visual.deepstack_mergers]:
        groups += [[merger.linear_fc1], [merger.linear_fc2]]
    return groups + _text_groups(model.text_model)


def quantize_model(model, calibration, *, smooth_alpha=0.8, qk_alpha=0.5) -> None:
    """Calibrate + enable W8A8 over the whole VLM in one pass.

    One joint pass matters here: the decoder's activation scales must see the
    image embeddings the vision tower actually produces, not text-only stats.
    """
    _quantize_model(
        model,
        calibration,
        groups=smoothquant_groups(model),
        smooth_alpha=smooth_alpha,
        qk_alpha=qk_alpha,
    )


class Qwen3VLArchitecture(LLMArchitecture):
    """Step 1 for Qwen3-VL: load + quantize the torch reference.

    The device side is split in two (see the module docstring): the vision tower
    is one graph of its own (vision_impl) and the decoder is the standard
    per-layer LLM artifact (text_impl), so this architecture only owns the
    quantized bundle they share.
    """

    name = "qwen3vl"

    def load_pretrained(self, repo_id, **kwargs):
        """Build the float reference.

        NB the generic `llm compile` path (generate.export_quantized_model) then
        calibrates on TEXT tokens only, which leaves the vision tower's observers
        empty and fails the export. Build a Qwen3-VL artifact through
        pipeline.build_artifacts, which calibrates both towers on real
        (image, question) pairs.
        """
        return Qwen3VLForConditionalGeneration.from_pretrained(repo_id, **kwargs)

    def quantize(self, model, calibration, *, smooth_alpha=0.8, qk_alpha=0.5):
        quantize_model(model, calibration, smooth_alpha=smooth_alpha, qk_alpha=qk_alpha)

    def record_lut_ranges(self, model, calibration):
        _record_lut_ranges(model, calibration)

    def num_layers(self, model):
        return model.config.text.num_hidden_layers

    def config_to_dict(self, model_config):
        return asdict(model_config)

    def config_from_dict(self, data):
        return Qwen3VLConfig(**data)

    def export_host_io(self, model):
        text = model.config.text
        host = {
            "embed_weight": model.text_model.embed_tokens.weight.detach()
            .cpu()
            .to(torch.float16)
            .numpy(),
            "final_norm_weight": model.text_model.norm.weight.detach()
            .cpu()
            .to(torch.float32)
            .numpy(),
            "rms_eps": np.float32(text.rms_norm_eps),
            "head_dim": np.int64(text.head_dim),
            "num_q_heads": np.int64(text.num_attention_heads),
            "rope_theta": np.float64(text.rope_theta),
            # MRoPE section widths -- the runtime needs them to build the prompt's
            # rotary tables (decode positions are 1-D, but the prompt's are not).
            "mrope_section": np.asarray(model.config.mrope_section, dtype=np.int64),
            "image_token_id": np.int64(model.config.image_token_id),
            # DeepStack: how many decoder layers take a vision feature map. The runtime
            # feeds one per-layer input per injected layer (text_impl.deepstack_positions).
            "deepstack_layers": np.int64(len(model.config.vision.deepstack_visual_indexes)),
        }
        if not text.tie_word_embeddings:
            host["lm_head_weight"] = model.lm_head.weight.detach().cpu().to(torch.float16).numpy()
        return host


register_architecture(Qwen3VLArchitecture())

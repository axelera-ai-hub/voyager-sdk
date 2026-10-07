# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Torch W8A8 fake-quant reference for a DINOv2-style ViT encoder block.

The block is composed from the shared fake-quant toolkit (quant.QuantLinear /
QuantSDPA / observers) so the exact same export helpers the LLM path uses
(export.export_linear_quant_params / export_sdpa_quant_params /
export_lut_ranges) apply unchanged -- guaranteeing the int8 math matches the
device datapath. The pieces the decoder-LLM path lacks and this block adds:

  * QuantLayerNorm (mean-subtract + gamma + beta), vs the LLM's scale-only
    RMSNorm -- mirrors the device axol.core.layer_norm_two_pass_kahan.
  * per-linear additive bias (DINOv2 uses qkv/out/fc1/fc2 biases), added in
    float AFTER dequant, exactly as the device does (atomic.add of a bias
    buffer) -- so bias lives outside the bias-free QuantLinear.
  * LayerScale (per-channel learned scale after attn / MLP).
  * bidirectional attention -- QuantSDPA is called with attn_mask=None (full
    attention over all patches), the device analogue being an all-zero mask.
  * a GELU MLP activation (RangeObserver("gelu") -> the "gelu" LUT fitter).
"""

from __future__ import annotations

from dataclasses import dataclass
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules import QuantSDPA
from ...quant import QuantLinear, RangeObserver

#: Lanes in one PWORD -- the granularity every device buffer's channel axis is laid out at.
PWORD = 64

# Default chained-PWL LUT lengths. These are DINOv2's, tuned on its own encoder and kept
# where they were -- including the env overrides, which are how that tower has always been
# A/B'd. They are only DEFAULTS now: the length is a field of DinoBlockConfig, so a tower
# whose activations need a finer chain (the SigLIP2 So400m encoder does) raises its own
# without moving anyone else's. The kernel's LUT BUFFER SHAPE and the reference's modelled
# chain both read the config, so the two cannot drift apart.
#
# Why 1 each, measured on DINOv2's own encoder. GELU: ~31% shorter epilogue chain / -24%
# single-core fc1, and over the tight [-3,3] gelu range as accurate as the chained-2 (kNN gap
# +0.020 vs +0.026). Softmax exp/reciprocal: -508us/block @518, the biggest per-op lever, and
# within noise of the chained-2 on that encoder (CLS 0.9676 vs 0.9667).
_DEFAULT_SDPA_NLUTS = int(os.environ.get("DINO_SDPA_NLUTS", "1"))
_DEFAULT_GELU_NLUTS = int(os.environ.get("DINO_GELU_NLUTS", "1"))
#: The LayerNorm's inv_sqrt chain: 2 everywhere it has been measured, hence the default.
_DEFAULT_NORM_NLUTS = 2


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


# --- Optional modelling of the device's chained PWL LUTs in the torch reference --------------
# The fake-quant reference computes exp / 1x / inv_sqrt / gelu in exact fp; the device
# approximates each with a 2-LUT ADDITIVE chain (llm.luts) fit over the recorded range.
# Modelling those LUTs here lets us isolate LUT error from true device bugs and reproduce
# device-only degradation off-device. Enabled per-site via set_lut_modeling(); the default
# (empty lut_sites) keeps the exact path, so nothing changes unless opted in.


def set_lut_modeling(module: nn.Module, sites) -> None:
    """Turn on LUT modelling for the given site functions (a set/iterable of
    {"inv_sqrt", "exp", "reciprocal", "gelu"}) on every relevant submodule; pass an empty
    set to restore the exact path. Requires record_lut_ranges to have been run."""
    sites = frozenset(sites)
    for m in module.modules():
        if isinstance(m, (QuantLayerNorm, DinoBlock)) or type(m).__name__ == "QuantSDPA":
            m.lut_sites = sites


@dataclass
class DinoBlockConfig:
    """Shape of one ViT encoder block.

    hidden_size and seq_len must be multiples of 64. head_dim defaults to exactly one
    PWORD (64); a tower whose head_dim is not a PWORD (the SigLIP2 So400m encoder of
    Qwen3-VL 8B and LFM2.5-VL-3B: 72) clears head_dim_is_pword, and the DEVICE widths
    below round it -- and an intermediate that is not a PWORD multiple -- up to whole
    PWORDs. This reference always runs the checkpoint's own widths; the padding is a
    weight-layout property of the kernel, and the zeros in it cannot reach an output.
    """

    hidden_size: int = 128
    num_heads: int = 2
    mlp_ratio: int = 4
    seq_len: int = 64
    layernorm_eps: float = 1e-6
    # Patch-embedding geometry (only used by the on-device embed kernel).
    image_size: int = 224
    patch_size: int = 14
    num_channels: int = 3
    #: Exact MLP width, for towers whose intermediate is not a whole multiple of
    #: hidden_size (Qwen3-VL 8B: 1152 -> 4304). None keeps hidden_size * mlp_ratio.
    mlp_hidden_size: int | None = None
    #: Whether head_dim must be exactly one PWORD. True for the device kernels written
    #: against a one-block head (dino itself); the Qwen3-VL 8B tower sets it False and
    #: pads head_dim on device instead (see Qwen3VLVisionConfig.padded_head_dim).
    head_dim_is_pword: bool = True
    #: How many hardware LUTs each nonlinearity's ADDITIVE chain uses -- the softmax's
    #: exp/reciprocal, the MLP's GELU, the two LayerNorms' inv_sqrt. A chain of n costs n
    #: passes and buys resolution: 16 segments fitted over the recorded range each. These
    #: are the kernel's buffer shapes AND what the reference models, so one number drives
    #: both. The defaults are DINOv2's; a tower whose activations span a wider range needs
    #: more (see the SigLIP2 So400m encoder in models/lfm2vl).
    sdpa_n_luts: int = _DEFAULT_SDPA_NLUTS
    gelu_n_luts: int = _DEFAULT_GELU_NLUTS
    norm_n_luts: int = _DEFAULT_NORM_NLUTS

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def intermediate_size(self) -> int:
        return self.mlp_hidden_size or self.hidden_size * self.mlp_ratio

    # --- device widths ------------------------------------------------------------
    # What the fused kernel lays each width out at: the checkpoint's own whenever it is
    # already a PWORD multiple (so an aligned tower never sees the padding at all), and
    # the next whole PWORD when it is not. Rounding head_dim to a whole PWORD -- rather
    # than to a whole HALF, as the Qwen3-VL tower does -- is enough here because nothing
    # in this block is position-dependent: it has no rotary embedding whose halves have
    # to start on a block boundary.

    @property
    def padded_head_dim(self) -> int:
        """The head stride on device: head_dim rounded up to a whole PWORD."""
        return _round_up(self.head_dim, PWORD)

    @property
    def head_dim_pw(self) -> int:
        return self.padded_head_dim // PWORD

    @property
    def qkv_dim(self) -> int:
        """Width of q/k/v/o_proj on device: the padded heads, laid out head-major."""
        return self.num_heads * self.padded_head_dim

    @property
    def padded_intermediate(self) -> int:
        """MLP width on device: the checkpoint's, rounded up to a PWORD multiple."""
        return _round_up(self.intermediate_size, PWORD)

    @property
    def pads_widths(self) -> bool:
        """Whether this block's device layout carries any padding at all."""
        return (self.padded_head_dim, self.padded_intermediate) != (
            self.head_dim,
            self.intermediate_size,
        )

    def validate(self) -> None:
        if self.hidden_size % 64 != 0:
            raise ValueError(f"hidden_size {self.hidden_size} must be a multiple of 64")
        if self.seq_len % 64 != 0:
            raise ValueError(f"seq_len {self.seq_len} must be a multiple of 64 (mask packing)")
        if self.head_dim_is_pword and self.head_dim != 64:
            raise ValueError(f"head_dim must be 64 (one PWORD); got {self.head_dim}")
        if self.head_dim % 2:
            raise ValueError(f"head_dim must be even (RoPE halves it); got {self.head_dim}")


class QuantLayerNorm(nn.Module):
    """LayerNorm with an inv_sqrt LUT-range recorder.

    Mirrors the device axol.core.layer_norm_two_pass_kahan: subtract the mean,
    scale by 1/sqrt(var + eps) (the LUT-approximated part -- the range recorder
    watches var + eps), then apply the learned gamma (weight) and beta (bias).

    ``n_luts`` is the length of that LUT's chain on device; it only matters when
    ``set_lut_modeling`` turns the approximation on, and it has to match the kernel's
    LUT buffer, which is why both come from the block config.
    """

    def __init__(
        self, hidden_size: int, eps: float = 1e-6, n_luts: int = _DEFAULT_NORM_NLUTS
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
        self.n_luts = int(n_luts)
        self.inv_sqrt_range = RangeObserver("inv_sqrt")
        self.lut_sites = frozenset()  # set by set_lut_modeling to model the device inv_sqrt LUT

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        mean = x.mean(-1, keepdim=True)
        var = self.inv_sqrt_range((x - mean).pow(2).mean(-1, keepdim=True) + self.eps)
        if "inv_sqrt" in self.lut_sites:
            # Deferred, as `modules.py` defers it: `luts` imports axelera.compiler and tvm at
            # module scope, and LUT modelling is opt-in (`lut_sites`). At module scope it made
            # the whole reference model unimportable without a compiler, which is what gated
            # the weight-mapping test -- pure state-dict logic that needs neither.
            from ...luts import apply_chained_lut

            inv = apply_chained_lut(
                var,
                "inv_sqrt",
                self.inv_sqrt_range.min_val.item(),
                self.inv_sqrt_range.max_val.item(),
                n_luts=self.n_luts,
            )
        else:
            inv = torch.rsqrt(var)
        xn = (x - mean) * inv
        return (self.weight * xn + self.bias).to(dtype)


class DinoBlock(nn.Module):
    """One DINOv2-style encoder block (pre-norm, LayerScale, GELU MLP).

    Attention uses three separate QuantLinear projections (q/k/v) rather than a
    fused qkv, matching the device kernel's three matmuls; each carries an
    additive bias applied after dequant.
    """

    def __init__(self, config: DinoBlockConfig, **quant_kwargs) -> None:
        super().__init__()
        config.validate()
        self.config = config
        h = config.hidden_size
        inter = config.intermediate_size

        self.ln1 = QuantLayerNorm(h, config.layernorm_eps, config.norm_n_luts)
        self.q_proj = QuantLinear(h, h, **quant_kwargs)
        self.k_proj = QuantLinear(h, h, **quant_kwargs)
        self.v_proj = QuantLinear(h, h, **quant_kwargs)
        self.q_bias = nn.Parameter(torch.zeros(h))
        self.k_bias = nn.Parameter(torch.zeros(h))
        self.v_bias = nn.Parameter(torch.zeros(h))
        self.sdpa = QuantSDPA(config.head_dim, quantize=True)
        # The softmax's exp/reciprocal chains, from the config: DINOv2's default is a single
        # LUT each (device -508us/block, kNN within noise), and a tower that needs finer
        # asks for it there rather than through a shared environment variable.
        self.sdpa.exp_n_luts = config.sdpa_n_luts
        self.sdpa.reciprocal_n_luts = config.sdpa_n_luts
        self.o_proj = QuantLinear(h, h, **quant_kwargs)
        self.o_bias = nn.Parameter(torch.zeros(h))
        self.ls1 = nn.Parameter(torch.ones(h))

        self.ln2 = QuantLayerNorm(h, config.layernorm_eps, config.norm_n_luts)
        self.fc1 = QuantLinear(h, inter, **quant_kwargs)
        self.fc1_bias = nn.Parameter(torch.zeros(inter))
        self.gelu_range = RangeObserver("gelu")
        self.fc2 = QuantLinear(inter, h, **quant_kwargs)
        self.fc2_bias = nn.Parameter(torch.zeros(h))
        self.ls2 = nn.Parameter(torch.ones(h))
        self.lut_sites = frozenset()  # set by set_lut_modeling to model the device gelu LUT

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        # (1, seq, hidden) -> (1, num_heads, seq, head_dim)
        b, s, _ = x.shape
        x = x.view(b, s, self.config.num_heads, self.config.head_dim)
        return x.transpose(1, 2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        # --- Attention sub-block ---
        x = self.ln1(hidden)
        q = self._heads(self.q_proj(x) + self.q_bias)
        k = self._heads(self.k_proj(x) + self.k_bias)
        v = self._heads(self.v_proj(x) + self.v_bias)
        attn = self.sdpa(q, k, v, None)  # bidirectional: no mask
        b, _, s, _ = attn.shape
        attn = attn.transpose(1, 2).reshape(b, s, cfg.hidden_size)
        o = self.o_proj(attn) + self.o_bias
        o = o * self.ls1
        res1 = hidden + o
        # --- MLP sub-block ---
        y = self.ln2(res1)
        h1 = self.gelu_range(self.fc1(y) + self.fc1_bias)
        if "gelu" in self.lut_sites:
            # The config's chain length, which is also the kernel's gelu_lut buffer shape,
            # so calibration models exactly the approximation the device will run.
            from ...luts import apply_chained_lut  # deferred: see the inv_sqrt site above

            h1 = apply_chained_lut(
                h1,
                "gelu",
                self.gelu_range.min_val.item(),
                self.gelu_range.max_val.item(),
                n_luts=cfg.gelu_n_luts,
            )
        else:
            h1 = F.gelu(h1)
        h2 = self.fc2(h1) + self.fc2_bias
        h2 = h2 * self.ls2
        return res1 + h2


def smoothquant_groups(module: nn.Module) -> list[list[QuantLinear]]:
    """SmoothQuant grouping for one block or a whole encoder.

    Per block: q/k/v share one group (they share the ln1 input, so the fused kernel
    can quantize it ONCE -> identical input quant params); o/fc1/fc2 have distinct
    inputs, so each is its own singleton group. For a multi-block module the groups
    from every block are concatenated -- the whole model is calibrated in one
    quantize_model call, exactly like the LLM Architecture.quantize path."""
    blocks = (
        [module]
        if isinstance(module, DinoBlock)
        else [m for m in module.modules() if isinstance(m, DinoBlock)]
    )
    groups: list[list[QuantLinear]] = []
    for b in blocks:
        groups += [[b.q_proj, b.k_proj, b.v_proj], [b.o_proj], [b.fc1], [b.fc2]]
    return groups


class DinoV2Encoder(nn.Module):
    """The repeated encoder: a stack of DinoBlocks quantized jointly. Calibrating this
    whole module (quantize_model) makes each block's observers see the real FP activation
    at its depth in one forward pass -- the same joint PTQ the LLM path uses."""

    def __init__(self, config: DinoBlockConfig, num_layers: int) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([DinoBlock(config) for _ in range(num_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class QuantPatchEmbed(nn.Module):
    """Non-overlapping patch projection (== the HF Conv2d) as a symmetric-input QuantLinear
    over im2col'd patches. Because stride==kernel, the conv is exactly this matmul; casting it
    as a QuantLinear lets it calibrate with the shared toolkit and its int8 weight + scales
    pack straight into the device conv. Input quant is SYMMETRIC (zp=0) to match the device
    conv (no dz term); bias is added after dequant (the conv bias). NOT SmoothQuant-grouped, so
    the host pixel quant stays a plain symmetric per-tensor scale (smoothing would need
    per-input-channel pixel scaling on host)."""

    def __init__(self, num_channels: int, hidden_size: int, patch_size: int, **quant_kwargs):
        super().__init__()
        self.num_channels = num_channels
        self.patch_size = patch_size
        patch_dim = num_channels * patch_size * patch_size
        self.proj = QuantLinear(patch_dim, hidden_size, act_symmetric=True, **quant_kwargs)
        self.bias = nn.Parameter(torch.zeros(hidden_size))

    def _im2col(self, pixel_values: torch.Tensor) -> torch.Tensor:
        # (1,C,H,W) -> (1, num_patches, C*P*P), flatten order (c,kh,kw) == conv weight flatten.
        b, c, h, w = pixel_values.shape
        p = self.patch_size
        x = pixel_values.reshape(b, c, h // p, p, w // p, p)  # b,c,ph,kh,pw,kw
        x = x.permute(0, 2, 4, 1, 3, 5)  # b,ph,pw,c,kh,kw
        return x.reshape(b, (h // p) * (w // p), c * p * p)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.proj(self._im2col(pixel_values)) + self.bias


class DinoV2Model(nn.Module):
    """Full DINOv2: quantizable patch embed (+ CLS + interpolated positional embeddings) ->
    encoder -> final QuantLayerNorm. Calibrated as ONE module on real pixel_values, so every
    int8 site (patch conv, blocks) sees its real activation and the final-norm LUT range is
    recorded. cls_token / pos_embed are host constants baked into the device embed kernel."""

    def __init__(
        self,
        config: DinoBlockConfig,
        num_layers: int,
        num_channels: int,
        patch_size: int,
        seq_real: int,
    ) -> None:
        super().__init__()
        self.patch_embed = QuantPatchEmbed(num_channels, config.hidden_size, patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.hidden_size))
        # interpolated pos-embed for this resolution (filled from HF in the loader).
        self.register_buffer(
            "pos_embed", torch.zeros(1, seq_real, config.hidden_size), persistent=False
        )
        self.encoder = DinoV2Encoder(config, num_layers)
        self.final_norm = QuantLayerNorm(
            config.hidden_size, config.layernorm_eps, config.norm_n_luts
        )

    def encoder_output(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Patch-embed + CLS + pos-embed + encoder blocks, BEFORE `final_norm`.

        A weight-mapping check for a source whose own `encode` also skips the final norm
        (DinoMixSource, since the MixVPR head was trained on pre-norm tokens) needs this exact
        value, not `forward`'s post-norm one, to compare like for like.
        """
        x = self.patch_embed(pixel_values)  # (1, num_patches, hidden)
        cls = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls, x), dim=1) + self.pos_embed  # (1, seq_real, hidden)
        return self.encoder(x)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.final_norm(self.encoder_output(pixel_values))

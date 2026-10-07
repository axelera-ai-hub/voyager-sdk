# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Torch W8A8 fake-quant reference for the SmolVLM2 SigLIP vision encoder.

Adapted from ``models/dino/reference.py``. Same fake-quant toolkit
(quant.QuantLinear / QuantSDPA / observers, QuantLayerNorm) so the shared export
helpers apply unchanged and the int8 math matches the device datapath.

SigLIP is a *simpler* bidirectional ViT than DINOv2 -- the pieces we DROP relative
to the DINOv2 reference:

  * no CLS token (sequence is exactly num_patches),
  * no register tokens,
  * no LayerScale (ls1/ls2) -- each sub-block output goes straight into the
    residual,
  * learned ABSOLUTE position embeddings (nn.Embedding indexed by position_ids),
    not the interpolated grid embedding DINOv2 uses.

What we KEEP (identical to DINOv2): QuantLayerNorm (mean-subtract + gamma + beta),
separate q/k/v/o QuantLinears each with an additive bias applied after dequant,
bidirectional attention (QuantSDPA with attn_mask=None), and a two-linear MLP.
The one MLP change: SigLIP uses the *tanh-approximation* GELU
(``gelu_pytorch_tanh``), so the reference activation is ``F.gelu(x, approximate=
"tanh")`` and the device fits its ``gelu`` LUT to that curve.

Unlike the DINOv2 reference (which fixes head_dim==64 for clean PWORD packing),
this reference does NOT constrain head_dim: SigLIP SO400M has head_dim=72. The
head-dim packing is purely a device-kernel concern (padding 72 -> 128), so the
torch reference stays exact for any head_dim.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...modules import QuantSDPA
from ...quant import QuantLinear, RangeObserver, enable_w8a8, quantize_model, record_lut_ranges


def set_lut_modeling(module: nn.Module, sites) -> None:
    """Turn on device-LUT modelling for the given sites ({"inv_sqrt","exp",
    "reciprocal","gelu"}) on every relevant submodule; empty set restores the
    exact path. Requires record_lut_ranges to have been run first."""
    sites = frozenset(sites)
    for m in module.modules():
        if isinstance(m, (SiglipLayerNorm, SiglipBlock)) or type(m).__name__ == "QuantSDPA":
            m.lut_sites = sites


# HF ``vision_config`` json key -> SiglipVisionConfig field, for from_hf_dict.
_HF_DICT_FIELDS = {
    "hidden_size": "hidden_size",
    "num_attention_heads": "num_heads",
    "num_hidden_layers": "num_hidden_layers",
    "intermediate_size": "intermediate_size",
    "layer_norm_eps": "layernorm_eps",
    "hidden_act": "hidden_act",
    "image_size": "image_size",
    "patch_size": "patch_size",
    "num_channels": "num_channels",
}


@dataclass
class SiglipVisionConfig:
    """Shape of the SigLIP vision encoder. Defaults are small (unit tests); real
    models come from ``hf.config_from_hf`` (the resolved HF config object).
    SmolVLM2-2.2B uses SigLIP SO400M (hidden 1152, 27 layers, 16 heads,
    intermediate 4304, patch 14, image 384)."""

    hidden_size: int = 128
    num_heads: int = 2
    num_hidden_layers: int = 2
    intermediate_size: int = 512
    layernorm_eps: float = 1e-6
    hidden_act: str = "gelu_pytorch_tanh"
    # Patch-embedding geometry.
    image_size: int = 224
    patch_size: int = 14
    num_channels: int = 3
    #: LUTs in each nonlinearity's ADDITIVE chain -- softmax exp/reciprocal, MLP GELU,
    #: LayerNorm inv_sqrt. Each is the kernel's LUT buffer shape AND what this reference
    #: emulates, so one number drives both. This tower's own, not the shared DINO_*_NLUTS
    #: knobs (as ``Lfm2VlVisionConfig``): a chain is fitted to one encoder's range.
    sdpa_n_luts: int = 1
    gelu_n_luts: int = 1
    norm_n_luts: int = 2

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_heads

    @property
    def num_patches_per_side(self) -> int:
        return self.image_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.num_patches_per_side**2

    def validate(self) -> None:
        if self.hidden_size % self.num_heads != 0:
            raise ValueError(
                f"hidden_size {self.hidden_size} not divisible by num_heads {self.num_heads}"
            )

    @classmethod
    def from_hf_dict(cls, cfg: dict) -> "SiglipVisionConfig":
        """Build from a HF ``vision_config`` dict (SmolVLM/SigLIP).

        Every field is required. A raw ``config.json`` may legitimately omit them, and
        that is precisely the case to reject: this class's defaults are the small
        unit-test shape, so absorbing an omission builds a silently mis-shaped encoder.
        Real checkpoints go through the resolved config object instead
        (``hf.config_from_hf``), where HF has already applied its own class defaults.
        """
        absent = [k for k in _HF_DICT_FIELDS if k not in cfg]
        if absent:
            raise KeyError(
                f"HF vision_config omits {absent}; refusing to fall back to "
                f"{cls.__name__}'s unit-test defaults. Build from the RESOLVED HF config "
                f"object via hf.config_from_hf instead of the raw json."
            )
        return cls(**{field: cfg[key] for key, field in _HF_DICT_FIELDS.items()})


class SiglipLayerNorm(nn.Module):
    """LayerNorm with an inv_sqrt LUT-range recorder (identical to DINOv2's
    QuantLayerNorm). Mirrors the device layer_norm_two_pass_kahan: subtract mean,
    scale by 1/sqrt(var+eps) (the LUT-approximated part), apply learned gamma/beta."""

    def __init__(self, hidden_size: int, eps: float = 1e-6, n_luts: int = 2) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.eps = eps
        #: == the kernel's ln*_lut / pn_lut shape. Default is apply_chained_lut's own 2.
        self.n_luts = int(n_luts)
        self.inv_sqrt_range = RangeObserver("inv_sqrt")
        self.lut_sites = frozenset()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        dtype = x.dtype
        x = x.to(torch.float32)
        mean = x.mean(-1, keepdim=True)
        var = self.inv_sqrt_range((x - mean).pow(2).mean(-1, keepdim=True) + self.eps)
        if "inv_sqrt" in self.lut_sites:
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


_TANH_GELU_ACTS = ("gelu_pytorch_tanh", "gelu_tanh", "gelu_new")
_ERF_GELU_ACTS = ("gelu", "gelu_none")


def _gelu(x: torch.Tensor, hidden_act: str) -> torch.Tensor:
    if hidden_act in _TANH_GELU_ACTS:
        return F.gelu(x, approximate="tanh")
    return F.gelu(x)


def gelu_lut_function(hidden_act: str) -> str:
    """The luts.generate_lut_table function name for this hidden_act.

    The device table is fit from the RangeObserver's function name (layout.fit_lut), so this
    is what keeps the fitted curve and the one ``_gelu`` evaluates in agreement. All three
    SmolVLM2 checkpoints are "gelu_pytorch_tanh"; DINOv2 / HF ViT are erf. An unrecognised
    act raises rather than falling back, which is the mismatch this mapping prevents.
    """
    if hidden_act in _TANH_GELU_ACTS:
        return "gelu_tanh"
    if hidden_act in _ERF_GELU_ACTS:
        return "gelu"
    raise ValueError(
        f"unsupported SigLIP hidden_act {hidden_act!r}; expected one of "
        f"{_TANH_GELU_ACTS + _ERF_GELU_ACTS}"
    )


class SiglipBlock(nn.Module):
    """One SigLIP encoder block (pre-norm, NO LayerScale, gelu-tanh MLP).

    Three separate q/k/v QuantLinears (matching the device kernel's three
    matmuls), each with an additive bias applied after dequant; bidirectional
    attention (no mask)."""

    def __init__(self, config: SiglipVisionConfig, **quant_kwargs) -> None:
        super().__init__()
        config.validate()
        self.config = config
        h = config.hidden_size
        inter = config.intermediate_size

        self.ln1 = SiglipLayerNorm(h, config.layernorm_eps, config.norm_n_luts)
        self.q_proj = QuantLinear(h, h, **quant_kwargs)
        self.k_proj = QuantLinear(h, h, **quant_kwargs)
        self.v_proj = QuantLinear(h, h, **quant_kwargs)
        self.q_bias = nn.Parameter(torch.zeros(h))
        self.k_bias = nn.Parameter(torch.zeros(h))
        self.v_bias = nn.Parameter(torch.zeros(h))
        self.sdpa = QuantSDPA(config.head_dim, quantize=True)
        # The kernel's sdpa_exp_lut / sdpa_inv_lut shape.
        self.sdpa.exp_n_luts = int(config.sdpa_n_luts)
        self.sdpa.reciprocal_n_luts = int(config.sdpa_n_luts)
        self.o_proj = QuantLinear(h, h, **quant_kwargs)
        self.o_bias = nn.Parameter(torch.zeros(h))

        self.ln2 = SiglipLayerNorm(h, config.layernorm_eps, config.norm_n_luts)
        self.fc1 = QuantLinear(h, inter, **quant_kwargs)
        self.fc1_bias = nn.Parameter(torch.zeros(inter))
        self.gelu_range = RangeObserver(gelu_lut_function(config.hidden_act))
        self.fc2 = QuantLinear(inter, h, **quant_kwargs)
        self.fc2_bias = nn.Parameter(torch.zeros(h))
        self.lut_sites = frozenset()

    def _heads(self, x: torch.Tensor) -> torch.Tensor:
        b, s, _ = x.shape
        x = x.view(b, s, self.config.num_heads, self.config.head_dim)
        return x.transpose(1, 2)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        cfg = self.config
        # --- Attention sub-block (no LayerScale) ---
        x = self.ln1(hidden)
        q = self._heads(self.q_proj(x) + self.q_bias)
        k = self._heads(self.k_proj(x) + self.k_bias)
        v = self._heads(self.v_proj(x) + self.v_bias)
        attn = self.sdpa(q, k, v, None)  # bidirectional: no mask
        b, _, s, _ = attn.shape
        attn = attn.transpose(1, 2).reshape(b, s, cfg.hidden_size)
        o = self.o_proj(attn) + self.o_bias
        res1 = hidden + o
        # --- MLP sub-block (gelu-tanh, no LayerScale) ---
        y = self.ln2(res1)
        h1 = self.gelu_range(self.fc1(y) + self.fc1_bias)
        if "gelu" in self.lut_sites:
            from ...luts import apply_chained_lut  # deferred: `luts` reads the compiler

            h1 = apply_chained_lut(
                h1,
                # the observer's own function (gelu_tanh for SigLIP), so the emulated LUT
                # is the same table fit_lut builds for the device
                self.gelu_range.function,
                self.gelu_range.min_val.item(),
                self.gelu_range.max_val.item(),
                n_luts=int(cfg.gelu_n_luts),  # == the kernel's gelu_lut buffer shape
            )
        else:
            h1 = _gelu(h1, cfg.hidden_act)
        h2 = self.fc2(h1) + self.fc2_bias
        return res1 + h2


def smoothquant_groups(module: nn.Module) -> list[list[QuantLinear]]:
    """SmoothQuant grouping for one block or a whole encoder. Per block: q/k/v
    share one group (share the ln1 input -> quantized once); o/fc1/fc2 are
    singletons. Multi-block modules concatenate every block's groups."""
    blocks = (
        [module]
        if isinstance(module, SiglipBlock)
        else [m for m in module.modules() if isinstance(m, SiglipBlock)]
    )
    groups: list[list[QuantLinear]] = []
    for b in blocks:
        groups += [[b.q_proj, b.k_proj, b.v_proj], [b.o_proj], [b.fc1], [b.fc2]]
    return groups


class SiglipEncoder(nn.Module):
    """The repeated encoder stack, quantized jointly (one PTQ pass)."""

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([SiglipBlock(config) for _ in range(config.num_hidden_layers)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class SiglipPatchEmbed(nn.Module):
    """Non-overlapping patch projection (== the HF Conv2d) as a symmetric-input
    QuantLinear over im2col'd patches (stride==kernel, so conv==matmul). Input
    quant is SYMMETRIC (zp=0) to match the device conv; bias added after dequant."""

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
        gh, gw = h // p, w // p
        # The tile side need NOT be a multiple of the patch size: HF's stride==kernel Conv2d
        # silently drops a partial trailing patch. The 2.2B's 384px tiles at patch 14 emit
        # 27x27 patches over [0, 378), ignoring the last 6 rows/columns (729, not 784), so
        # crop to match. No-op at 512/16.
        if gh * p != h or gw * p != w:
            pixel_values = pixel_values[..., : gh * p, : gw * p]
        x = pixel_values.reshape(b, c, gh, p, gw, p)
        x = x.permute(0, 2, 4, 1, 3, 5)
        return x.reshape(b, gh * gw, c * p * p)

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.proj(self._im2col(pixel_values)) + self.bias


class SiglipVisionModel(nn.Module):
    """Full SigLIP vision tower: quantizable patch embed (+ learned absolute pos
    embed) -> encoder -> final post-LayerNorm. Calibrated as ONE module on real
    pixel_values so every int8 site sees its real activation. pos_embed is a host
    constant baked into the device embed kernel (NO CLS, NO interpolation)."""

    def __init__(self, config: SiglipVisionConfig) -> None:
        super().__init__()
        self.config = config
        self.patch_embed = SiglipPatchEmbed(
            config.num_channels, config.hidden_size, config.patch_size
        )
        # Learned absolute position embedding, one row per patch (no CLS row).
        self.pos_embed = nn.Parameter(torch.zeros(1, config.num_patches, config.hidden_size))
        self.encoder = SiglipEncoder(config)
        self.post_layernorm = SiglipLayerNorm(
            config.hidden_size, config.layernorm_eps, config.norm_n_luts
        )

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        x = self.patch_embed(pixel_values)  # (1, num_patches, hidden)
        x = x + self.pos_embed
        x = self.encoder(x)
        return self.post_layernorm(x)


def build_calibrated_block(
    config: SiglipVisionConfig, *, seed: int = 0, n_calib: int = 8
) -> SiglipBlock:
    """A self-contained int8 SigLIP block from random weights (no HF checkpoint):
    calibrate -> enable fake-quant -> record LUT ranges. For unit tests."""
    torch.manual_seed(seed)
    block = SiglipBlock(config)
    for p in block.parameters():
        nn.init.normal_(p, std=0.02) if p.ndim > 1 else nn.init.normal_(p, std=0.05)
    block.eval()

    seq = config.num_patches
    calibration = [torch.randn(1, seq, config.hidden_size) for _ in range(n_calib)]
    quantize_model(
        block, calibration, groups=smoothquant_groups(block), smooth_alpha=None, qk_alpha=None
    )
    enable_w8a8(block)
    record_lut_ranges(block, calibration)
    return block

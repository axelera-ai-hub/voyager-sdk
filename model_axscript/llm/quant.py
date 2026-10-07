# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Architecture-agnostic W8A8 + SmoothQuant fake-quantization toolkit.

Provides activation observers, a quantizable linear, SmoothQuant / QK-smoothing,
a LUT input-range recorder, and the calibrate/smooth/recalibrate/enable
orchestration for post-training-quantizing a decoder LLM.
"""

from __future__ import annotations

import enum
import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class QuantMode(enum.Enum):
    """Operating mode of an ActivationObserver."""

    DISABLED = "disabled"  # passthrough; the tensor is returned unchanged
    OBSERVE = "observe"  # collect statistics, but do not modify the tensor
    HIST = "hist"  # accumulate a histogram over the observed range (percentile calib)
    FAKE_QUANT = "fake_quant"  # quantize-dequantize using the collected scale


class ActivationObserver(nn.Module):
    """Per-tensor minmax activation observer with optional fake-quant.

    Symmetric (zero_point = 0, q in [-qmax, qmax]) or asymmetric/affine (learned
    zero_point, q in [qmin, qmax]); the affine zero-point lets the int range
    straddle a non-zero-centred distribution (e.g. SiLU output) without wasting
    codes (the per-channel part comes from SmoothQuant). FAKE_QUANT does a
    quantize-dequantize round-trip; DISABLED (default) is a no-op; OBSERVE tracks
    running min/max and, when num_features is given, a per-input-channel abs_max
    for SmoothQuant.
    """

    def __init__(
        self,
        num_bits: int = 8,
        symmetric: bool = True,
        *,
        num_features: int | None = None,
        pinned_zero: bool | None = None,
        signed_full_range: bool | None = None,
        eps: float = 1e-12,
        calib_method: str = "minmax",
        percentile: float = 99.99,
        n_bins: int = 2048,
    ) -> None:
        super().__init__()
        self.num_bits = num_bits
        # Calibration for the activation clip range: "minmax" (running min/max) or
        # "percentile" (clip both tails at the (100-percentile)th / percentile-th
        # value of a histogram accumulated over the observed range -- a second HIST
        # pass; the ViT-standard fix for heavy activation outliers, cf. FQ-ViT/TensorRT).
        self.calib_method = calib_method
        self.percentile = percentile
        self.n_bins = n_bins
        # symmetric couples zero-pinning and restricted range; the two knobs can be
        # overridden independently for decoupled cases (e.g. attn_weight: zero
        # pinned but full int range).
        self.pinned_zero = symmetric if pinned_zero is None else pinned_zero
        self.signed_full_range = (
            (not symmetric) if signed_full_range is None else signed_full_range
        )
        self.eps = eps
        self.mode = QuantMode.DISABLED
        self.register_buffer("min_val", torch.zeros(()))
        self.register_buffer("max_val", torch.zeros(()))
        self.register_buffer("num_batches", torch.zeros((), dtype=torch.long))
        # Per-input-channel running abs-max for SmoothQuant (optional), plus the SIGNED
        # per-channel range. The signed pair is what lets the post-smoothing per-tensor
        # range be derived instead of re-observed: smoothing divides the input by a
        # positive per-channel s, so the smoothed tensor's min/max is just the min/max
        # over channels of (channel_min/s, channel_max/s). See rescale_from_channels.
        if num_features is not None:
            self.register_buffer("channel_abs_max", torch.zeros(num_features))
            self.register_buffer("channel_min", torch.zeros(num_features))
            self.register_buffer("channel_max", torch.zeros(num_features))
        else:
            self.channel_abs_max = None
            self.channel_min = None
            self.channel_max = None
        # Histogram over [hist_lo, hist_hi] for percentile calibration (HIST pass).
        self.register_buffer("hist", torch.zeros(n_bins), persistent=False)
        self.register_buffer("hist_lo", torch.zeros(()), persistent=False)
        self.register_buffer("hist_hi", torch.zeros(()), persistent=False)

    @property
    def qmin(self) -> int:
        # Restricted symmetric range [-(2^(b-1)-1), .] vs full signed [-2^(b-1), .].
        return (
            -(2 ** (self.num_bits - 1))
            if self.signed_full_range
            else -(2 ** (self.num_bits - 1) - 1)
        )

    @property
    def qmax(self) -> int:
        return 2 ** (self.num_bits - 1) - 1

    @property
    def abs_max(self) -> torch.Tensor:
        """Observed maximum absolute value (drives the symmetric scale)."""
        return torch.maximum(self.min_val.abs(), self.max_val.abs())

    def qparams(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Static (scale, zero_point) from the observed min/max (or percentile clip)."""
        if self.calib_method == "percentile" and float(self.hist.sum()) > 0:
            a = (100.0 - self.percentile) / 100.0
            lo = self._hist_quantile(a)
            hi = self._hist_quantile(1.0 - a)
            return self._range_to_qparams(lo, hi)
        return self._range_to_qparams(self.min_val, self.max_val)

    def can_rescale(self) -> bool:
        """Whether the post-smoothing range can be derived rather than re-observed.

        Needs the signed per-channel range, and a min/max clip: a percentile clip reads a
        HISTOGRAM of the smoothed tensor, which per-channel extremes cannot reconstruct.
        """
        return self.channel_min is not None and self.calib_method == "minmax"

    def rescale_from_channels(self, smooth_scale: torch.Tensor) -> None:
        """Set the per-tensor range to what OBSERVING x / smooth_scale would have given.

        SmoothQuant is output-neutral -- it folds s into the weight and divides the input
        by it -- so after smoothing every layer sees the identical input it saw before, and
        the only thing a second calibration pass learns is this range. s is positive, so
        dividing preserves order per channel and the tensor extremes are just the extremes
        over channels of the divided per-channel ones.
        """
        if not self.can_rescale():
            raise RuntimeError("observer cannot derive a post-smoothing range; re-observe")
        s = smooth_scale.to(self.channel_min.dtype)
        self.min_val = torch.minimum((self.channel_min / s).min(), (self.channel_max / s).min())
        self.max_val = torch.maximum((self.channel_min / s).max(), (self.channel_max / s).max())

    def _init_hist(self) -> None:
        """Freeze the histogram range to the observed min/max and zero the counts."""
        lo, hi = self.min_val.detach().clone(), self.max_val.detach().clone()
        if float(hi - lo) <= 0:  # degenerate (constant tensor): give it a tiny width
            hi = lo + self.eps
        self.hist_lo, self.hist_hi = lo, hi
        self.hist = torch.zeros(self.n_bins, device=lo.device)

    def _hist_quantile(self, q: float) -> torch.Tensor:
        """Value at cumulative fraction q of the accumulated histogram."""
        cdf = torch.cumsum(self.hist, 0)
        target = q * float(cdf[-1])
        idx = int(
            torch.searchsorted(cdf, torch.tensor(target, device=cdf.device)).clamp(
                0, self.n_bins - 1
            )
        )
        width = (self.hist_hi - self.hist_lo) / self.n_bins
        return self.hist_lo + (idx + 0.5) * width

    @property
    def scale(self) -> torch.Tensor:
        return self.qparams()[0]

    @property
    def zero_point(self) -> torch.Tensor:
        return self.qparams()[1]

    def _range_to_qparams(self, min_v: torch.Tensor, max_v: torch.Tensor):
        if self.pinned_zero:
            # Zero pinned to 0: a single symmetric scale from the abs-max.
            scale = (torch.maximum(min_v.abs(), max_v.abs()) / self.qmax).clamp_min(self.eps)
            return scale, torch.zeros_like(scale)
        # Asymmetric/affine: include 0 in the range so it is exactly representable.
        min_v = torch.minimum(min_v, torch.zeros_like(min_v))
        max_v = torch.maximum(max_v, torch.zeros_like(max_v))
        scale = ((max_v - min_v) / (self.qmax - self.qmin)).clamp_min(self.eps)
        zero_point = torch.round(self.qmin - min_v / scale).clamp(self.qmin, self.qmax)
        return scale, zero_point

    def reset(self) -> None:
        self.min_val.zero_()
        self.max_val.zero_()
        self.num_batches.zero_()
        if self.channel_abs_max is not None:
            self.channel_abs_max.zero_()
            self.channel_min.zero_()
            self.channel_max.zero_()
        self.hist.zero_()
        self.hist_lo.zero_()
        self.hist_hi.zero_()

    def _fake_quant(self, x, scale, zero_point):
        q = torch.clamp(torch.round(x / scale) + zero_point, self.qmin, self.qmax)
        return (q - zero_point) * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode is QuantMode.OBSERVE:
            xd = x.detach()
            self.min_val = torch.minimum(self.min_val, xd.min().to(self.min_val.dtype))
            self.max_val = torch.maximum(self.max_val, xd.max().to(self.max_val.dtype))
            if self.channel_abs_max is not None:
                reduce_dims = tuple(range(xd.ndim - 1))
                channel_max = xd.abs().amax(dim=reduce_dims).to(self.channel_abs_max.dtype)
                self.channel_abs_max = torch.maximum(self.channel_abs_max, channel_max)
                self.channel_min = torch.minimum(
                    self.channel_min, xd.amin(dim=reduce_dims).to(self.channel_min.dtype)
                )
                self.channel_max = torch.maximum(
                    self.channel_max, xd.amax(dim=reduce_dims).to(self.channel_max.dtype)
                )
            self.num_batches += 1
            return x
        if self.mode is QuantMode.HIST:
            xd = x.detach().flatten().float()
            lo, hi = self.hist_lo, self.hist_hi
            idx = ((xd - lo) / (hi - lo) * self.n_bins).long().clamp_(0, self.n_bins - 1)
            self.hist += torch.bincount(idx, minlength=self.n_bins).to(self.hist.dtype)
            return x
        if self.mode is QuantMode.FAKE_QUANT:
            scale, zero_point = self.qparams()
            return self._fake_quant(x, scale.to(x.dtype), zero_point.to(x.dtype))
        return x


def set_quant_mode(module: nn.Module, mode: QuantMode) -> None:
    """Sets the QuantMode on every observer under module."""
    for observer in module.modules():
        if isinstance(observer, ActivationObserver):
            observer.mode = mode


def reset_observers(module: nn.Module) -> None:
    """Clears the collected statistics on every observer under module."""
    for observer in module.modules():
        if isinstance(observer, ActivationObserver):
            observer.reset()


class RangeObserver(nn.Module):
    """Pass-through module recording its input's running min/max.

    Captures the input range of a LUT-approximated nonlinearity (inverse-sqrt,
    SiLU, exp, reciprocal) so a downstream component can fit a table over that
    range. Records only while recording is set. Buffers are non-persistent
    (recomputed per session, kept out of the HF state dict).
    """

    def __init__(self, function: str) -> None:
        super().__init__()
        self.function = function
        self.recording = False
        self.register_buffer("min_val", torch.full((), float("inf")), persistent=False)
        self.register_buffer("max_val", torch.full((), float("-inf")), persistent=False)
        self.register_buffer("num_batches", torch.zeros((), dtype=torch.long), persistent=False)

    def reset(self) -> None:
        self.min_val.fill_(float("inf"))
        self.max_val.fill_(float("-inf"))
        self.num_batches.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.recording:
            xd = x.detach()
            self.min_val = torch.minimum(self.min_val, xd.min().to(self.min_val.dtype))
            self.max_val = torch.maximum(self.max_val, xd.max().to(self.max_val.dtype))
            self.num_batches += 1
        return x


def set_lut_recording(module: nn.Module, recording: bool) -> None:
    """Toggles range recording on every RangeObserver under module."""
    for observer in module.modules():
        if isinstance(observer, RangeObserver):
            observer.recording = recording


def reset_lut_ranges(module: nn.Module) -> None:
    """Clears the recorded ranges on every RangeObserver."""
    for observer in module.modules():
        if isinstance(observer, RangeObserver):
            observer.reset()


def record_lut_ranges(module: nn.Module, calibration) -> None:
    """Records LUT-input ranges by running calibration through module.

    Run on the *quantized* model (after quantize_model) so the captured
    ranges match the values the hardware LUTs will see.
    """
    reset_lut_ranges(module)
    set_lut_recording(module, True)
    with torch.no_grad():
        for batch in calibration:
            module(batch)
    set_lut_recording(module, False)


#: Elements of a weight that quantization arithmetic will materialise at once. Past this,
#: callers walk output-row bands so the temporaries stay bounded; 64M elements is 256 MB per
#: fp32 copy, small beside a calibration GPU and far above every layer weight in the models
#: here (only a vocabulary projection reaches it).
_QUANT_BAND_ELEMS = 64 << 20


def row_bands(weight: torch.Tensor, band_elems: int = _QUANT_BAND_ELEMS) -> list[slice]:
    """Output-row bands to quantize ``weight`` in, one slice covering it all when it is small.

    Quantizing a weight costs several temporaries its own size (the abs, the divide, the
    round, the int32 cast for the zero-point sum). At a 248320x2048 lm_head that is 1.9 GiB
    each, enough to OOM a 12 GB calibration GPU; banding bounds the peak to one band. The
    scale is per output channel and the zero-point sum is per row, so no reduction crosses a
    band and the quantization is unchanged by the split. ``band_elems`` overrides the budget
    for a caller running where less memory is free.
    """
    rows_per_band = max(1, band_elems // weight.shape[1])
    if weight.shape[0] <= rows_per_band:
        return [slice(None)]
    return [
        slice(lo, min(lo + rows_per_band, weight.shape[0]))
        for lo in range(0, weight.shape[0], rows_per_band)
    ]


class QuantLinear(nn.Module):
    """A bias-free linear layer with optional W8A8 fake quantization.

    Owns its weight parameter directly so a HuggingFace checkpoint maps onto
    <path>.weight as for an nn.Linear. Combines two independently-
    toggled pieces (both off by default, so it is initially identical to an
    unquantized linear):

    * an ActivationObserver on the input (the "A8" half; per-tensor
      asymmetric to match the production kernels), and
    * per-output-channel symmetric weight fake-quant (the "W8" half). Weights are
      static, so the scale is computed directly (one per output row); per-channel
      absorbs per-row magnitude variation a per-tensor scale would average over.

    Supports SmoothQuant: apply_smoothing folds a per-input-channel s
    into the weight (W <- W * s) and divides the input by s in
    forward. Since (x / s) @ (W * s).T == x @ W.T this is exact in
    float but moves activation outliers into the easy-to-quantize weights.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        act_bits: int = 8,
        weight_bits: int = 8,
        act_symmetric: bool = False,
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # nn.Linear default

        self.act_observer = ActivationObserver(
            num_bits=act_bits, symmetric=act_symmetric, num_features=in_features
        )
        self.weight_bits = weight_bits
        self.weight_quant = False  # toggled via set_weight_quant
        self.eps = eps

        # SmoothQuant per-input-channel factor; ones until apply_smoothing runs.
        # Non-persistent: it is folded into the (persistent) weight, so it need
        # not be checkpointed, and this keeps it out of HF state-dict loading.
        self.smoothing_enabled = False
        self.register_buffer("smooth_scale", torch.ones(in_features), persistent=False)

    @property
    def weight_qmax(self) -> int:
        return 2 ** (self.weight_bits - 1) - 1

    def weight_scale(self, rows: slice = slice(None)) -> torch.Tensor:
        """Per-output-channel symmetric scale, shape [out_features, 1].

        ``rows`` restricts it to a band of output channels. The scale is per output channel,
        so a band's scale is exactly the corresponding slice of the whole one.
        """
        abs_max = self.weight[rows].detach().abs().amax(dim=1, keepdim=True)
        return (abs_max / self.weight_qmax).clamp_min(self.eps)

    def _quantized_weight(self, rows: slice = slice(None)) -> torch.Tensor:
        weight = self.weight[rows]
        if not self.weight_quant:
            return weight
        scale = self.weight_scale(rows).to(weight.dtype)
        q = torch.clamp(torch.round(weight / scale), -self.weight_qmax, self.weight_qmax)
        return q * scale

    @torch.no_grad()
    def apply_smoothing(self, s: torch.Tensor) -> None:
        """Folds a per-input-channel SmoothQuant factor s ([in_features])
        into the weight. The fold is in-place and irreversible (run at most once).
        """
        if self.smoothing_enabled:
            raise RuntimeError("smoothing already applied; smooth_quant must run only once")
        s = s.to(self.weight.dtype)
        self.weight.mul_(s)  # W[:, j] *= s[j]
        self.smooth_scale = s
        self.smoothing_enabled = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.smoothing_enabled:
            x = x / self.smooth_scale
        x = self.act_observer(x)
        # See row_bands: a vocabulary-sized head cannot be fake-quantized in one shot on a
        # calibration GPU. Banding leaves the quantization unchanged, though the result is
        # not bitwise equal to the one-shot path -- float32 F.linear picks different kernels
        # per weight shape (~1e-6, reproduced with no quantization in play at all).
        bands = row_bands(self.weight) if self.weight_quant else [slice(None)]
        if len(bands) == 1:
            return F.linear(x, self._quantized_weight())
        out = x.new_empty((*x.shape[:-1], self.weight.shape[0]))
        for band in bands:
            out[..., band] = F.linear(x, self._quantized_weight(band))
        return out


class PerChannelActivationObserver(ActivationObserver):
    """Per-channel symmetric activation observer (for the depthwise conv input).

    The depthwise conv input is [B, C, L] and each channel is convolved
    independently, so a single per-tensor activation scale (what QuantLinear uses,
    tamed there by SmoothQuant) crushes the low-magnitude channels of the gated
    B*x to zero and destroys the model. This instead tracks abs-max per channel
    along channel_axis and produces a per-channel symmetric scale (zero-point 0);
    quant/dequant broadcast it back over the batch/length axes. Subclasses
    ActivationObserver so calibrate / enable_w8a8 still discover and drive it.
    """

    def __init__(
        self, num_channels: int, channel_axis: int = 1, *, num_bits: int = 8, eps: float = 1e-12
    ) -> None:
        super().__init__(num_bits=num_bits, symmetric=True, eps=eps)
        self.channel_axis = channel_axis
        # Widen the (scalar) running min/max to one entry per channel.
        self.register_buffer("min_val", torch.zeros(num_channels))
        self.register_buffer("max_val", torch.zeros(num_channels))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode is QuantMode.OBSERVE:
            xd = x.detach()
            dims = tuple(d for d in range(xd.ndim) if d != self.channel_axis)
            self.min_val = torch.minimum(self.min_val, xd.amin(dim=dims).to(self.min_val.dtype))
            self.max_val = torch.maximum(self.max_val, xd.amax(dim=dims).to(self.max_val.dtype))
            self.num_batches += 1
            return x
        if self.mode is QuantMode.FAKE_QUANT:
            scale, zero_point = self.qparams()  # per-channel, zero_point == 0
            shape = [1] * x.ndim
            shape[self.channel_axis] = -1
            s = scale.reshape(shape).to(x.dtype)
            return self._fake_quant(x, s, torch.zeros_like(s))
        return x


class QuantConv1d(nn.Module):
    """A bias-free depthwise causal Conv1d with optional W8A8 fake quantization.

    The depthwise (groups == channels) analogue of QuantLinear for the LFM2
    short-conv token mixer: each channel owns K FIR taps (weight shape
    [channels, 1, K]) and there is no cross-channel mixing. Owns its weight
    directly so a HuggingFace nn.Conv1d checkpoint maps onto <path>.weight.
    Combines the same two independently-toggled pieces as QuantLinear (both off
    by default, so it is initially identical to an unquantized conv):

    * a PerChannelActivationObserver on the input (the "A8" half): per-channel
      symmetric, because the depthwise conv has no SmoothQuant to migrate the
      gated B*x's per-channel outliers, so a per-tensor scale would zero out the
      quiet channels, and
    * per-output-channel symmetric weight fake-quant (the "W8" half): one scale
      per channel over its K taps.

    Causal like torch.nn.Conv1d(..., padding=K-1)[..., :T]: left-pad by K-1 and
    drop the trailing K-1 outputs. Symmetric (zero-point 0) activation keeps the
    int8 DWPU dequant a pure per-channel scale and lets the zero-init conv-state
    cache represent real-zero exactly.
    """

    def __init__(
        self,
        channels: int,
        kernel_size: int,
        *,
        act_bits: int = 8,
        weight_bits: int = 8,
        act_symmetric: bool = True,  # accepted for kwarg-compat; per-channel is always symmetric
        eps: float = 1e-12,
    ) -> None:
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = nn.Parameter(torch.empty(channels, 1, kernel_size))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # nn.Conv1d default

        # Per-channel symmetric activation quant over the channel axis (conv input is
        # [B, C, L], so axis 1). Per-channel is essential for a depthwise conv (see
        # PerChannelActivationObserver); per-tensor here breaks the model.
        self.act_observer = PerChannelActivationObserver(
            channels, channel_axis=1, num_bits=act_bits
        )
        self.weight_bits = weight_bits
        self.weight_quant = False  # toggled via set_weight_quant
        self.eps = eps

    @property
    def weight_qmax(self) -> int:
        return 2 ** (self.weight_bits - 1) - 1

    def weight_scale(self) -> torch.Tensor:
        """Per-output-channel symmetric scale, shape [channels, 1, 1]."""
        abs_max = self.weight.detach().abs().amax(dim=(1, 2), keepdim=True)
        return (abs_max / self.weight_qmax).clamp_min(self.eps)

    def _quantized_weight(self) -> torch.Tensor:
        if not self.weight_quant:
            return self.weight
        scale = self.weight_scale().to(self.weight.dtype)
        q = torch.clamp(torch.round(self.weight / scale), -self.weight_qmax, self.weight_qmax)
        return q * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-1]
        y = F.conv1d(
            self.act_observer(x),
            self._quantized_weight(),
            groups=self.channels,
            padding=self.kernel_size - 1,
        )
        return y[..., :seq_len]


def set_weight_quant(module: nn.Module, enabled: bool = True) -> None:
    """Enables or disables weight fake-quant on every QuantLinear / QuantConv1d."""
    for layer in module.modules():
        if isinstance(layer, (QuantLinear, QuantConv1d)):
            layer.weight_quant = enabled


def enable_w8a8(module: nn.Module) -> None:
    """Turn on 8-bit weights and activation fake-quant.

    Per-token (dynamic) activations need no calibration; per_tensor ones must
    be calibrated first in OBSERVE mode (after SmoothQuant, if used).
    """
    set_quant_mode(module, QuantMode.FAKE_QUANT)
    set_weight_quant(module, True)


def disable_quant(module: nn.Module) -> None:
    """Convenience: return the model to full-precision behaviour."""
    set_quant_mode(module, QuantMode.DISABLED)
    set_weight_quant(module, False)


def _is_qk_smoothable(module: nn.Module) -> bool:
    """Whether module is a quantized attention supporting QK-smoothing.

    Duck-typed so the toolkit needs no architecture import.
    """
    return (
        getattr(module, "quantize", False)
        and hasattr(module, "q_observer")
        and hasattr(module, "k_observer")
        and hasattr(module, "q_smooth_scale")
        and hasattr(module, "k_smooth_scale")
        and hasattr(module, "qk_smoothing_enabled")
    )


def apply_smooth_quant(
    groups: list[list[QuantLinear]], alpha: float = 0.8, eps: float = 1e-5
) -> None:
    """Applies SmoothQuant to each group of input-sharing QuantLinear.

    For each group, computes one per-input-channel factor

        s_j = max|X_j|**alpha / max|W_j|**(1 - alpha)

    (max|X_j| the shared calibrated activation max, max|W_j| the max over
    all the group's weights) and folds it into each member, migrating activation
    outliers into the weights so cheap per-tensor activation quant is accurate.
    alpha is the migration strength (0.8 matches the production
    recommendation). Requires calibrated observers. Members of a group share one
    set of activation quant params; the grouping is the caller's responsibility.
    """
    for group in groups:
        # Shared per-channel activation max across members; elementwise max to be safe.
        act_max = None
        weight_max = None
        for layer in group:
            cam = layer.act_observer.channel_abs_max
            if cam is None or int(layer.act_observer.num_batches.item()) == 0:
                raise RuntimeError("calibrate before smooth_quant (no activation stats)")
            wm = layer.weight.detach().abs().amax(dim=0)  # [in_features]
            act_max = cam if act_max is None else torch.maximum(act_max, cam)
            weight_max = wm if weight_max is None else torch.maximum(weight_max, wm)
        s = (
            act_max.clamp_min(eps).pow(alpha) / weight_max.clamp_min(eps).pow(1 - alpha)
        ).clamp_min(eps)
        for layer in group:
            layer.apply_smoothing(s)


def smooth_qk(module: nn.Module, alpha: float = 0.5, eps: float = 1e-5) -> None:
    """Applies QK-smoothing to every QK-smoothable attention under module.

    Computes a per-head-dim factor

        s_d = max|K_d|**alpha / max|Q_d|**(1 - alpha)

    from the calibrated Q/K channel maxima, split between K (x/s) and Q
    (x*s) so it cancels in Q*K^T while balancing their per-channel
    magnitudes for quantization. alpha defaults to 0.5 (production
    recommendation). Requires calibrated attention observers.
    """
    for sdpa in module.modules():
        if not _is_qk_smoothable(sdpa):
            continue
        if sdpa.qk_smoothing_enabled:
            raise RuntimeError("QK-smoothing already applied; smooth_qk must run only once")
        max_q = sdpa.q_observer.channel_abs_max
        max_k = sdpa.k_observer.channel_abs_max
        if int(sdpa.q_observer.num_batches.item()) == 0:
            raise RuntimeError("calibrate before smooth_qk (no Q/K stats)")
        s = (max_k.clamp_min(eps).pow(alpha) / max_q.clamp_min(eps).pow(1 - alpha)).clamp_min(eps)
        sdpa.k_smooth_scale = s  # K: x / s
        sdpa.q_smooth_scale = 1.0 / s  # Q: x / (1/s) = x * s
        sdpa.qk_smoothing_enabled = True


def set_calibration_method(
    module: nn.Module, method: str, *, percentile: float = 99.99, only_asymmetric: bool = True
) -> int:
    """Set the activation clip calibration ("minmax" | "percentile") on observers.

    only_asymmetric restricts it to the affine (non-zero-pinned) observers -- the
    Linear activations the percentile fix targets -- leaving the softmax/V observers
    (pinned-zero, bounded ranges) on min-max. Returns the count of observers changed.
    """
    n = 0
    for obs in module.modules():
        if isinstance(obs, ActivationObserver) and not (only_asymmetric and obs.pinned_zero):
            obs.calib_method = method
            obs.percentile = percentile
            n += 1
    return n


def calibrate(module: nn.Module, calibration) -> None:
    """Run calibration through module in OBSERVE mode (resetting first).

    Percentile observers get a second HIST pass: the OBSERVE pass fixes each one's
    histogram range to its min/max, then HIST accumulates the distribution so qparams
    can clip the tails at the requested percentile.
    """
    reset_observers(module)
    set_quant_mode(module, QuantMode.OBSERVE)
    with torch.no_grad():
        for batch in calibration:
            module(batch)
    hist_obs = [
        o
        for o in module.modules()
        if isinstance(o, ActivationObserver) and o.calib_method == "percentile"
    ]
    if hist_obs:
        set_quant_mode(module, QuantMode.DISABLED)  # others: passthrough, keep their min/max
        for o in hist_obs:
            o._init_hist()
            o.mode = QuantMode.HIST
        with torch.no_grad():
            for batch in calibration:
                module(batch)


def _smoothed_observers(module: nn.Module) -> list:
    """(observer, smooth_scale) for every observer whose input smoothing rescaled.

    An observer that was NOT smoothed needs nothing: smoothing is output-neutral, so its
    input is unchanged and its first-pass range still holds.
    """
    pairs = []
    for m in module.modules():
        if isinstance(m, QuantLinear) and getattr(m, "smoothing_enabled", False):
            pairs.append((m.act_observer, m.smooth_scale))
        if getattr(m, "qk_smoothing_enabled", False):
            pairs.append((m.q_observer, m.q_smooth_scale))
            pairs.append((m.k_observer, m.k_smooth_scale))
    return pairs


def quantize_model(
    module: nn.Module,
    calibration,
    *,
    groups: list[list[QuantLinear]] | None = None,
    smooth_alpha: float | None = 0.8,
    qk_alpha: float | None = 0.5,
) -> None:
    """Calibrate and enable W8A8 (+ SmoothQuant / QK-smoothing) in one call.

    Runs the order-sensitive PTQ sequence: calibrate, smooth (linears via
    smooth_alpha/groups, attention via qk_alpha; each skipped when its
    alpha is None or nothing is smoothable), re-calibrate on the smoothed
    activations, then switch to fake-quant. groups is the input-sharing
    grouping (per-layer singletons when None).

    Must run on a freshly-loaded model — the smoothing folds are one-shot.
    """
    calibration = list(calibration)
    if groups is None:
        groups = [[m] for m in module.modules() if isinstance(m, QuantLinear)]

    calibrate(module, calibration)
    smoothed = False
    if smooth_alpha is not None:
        apply_smooth_quant(groups, alpha=smooth_alpha)
        smoothed = True
    if qk_alpha is not None and any(_is_qk_smoothable(m) for m in module.modules()):
        smooth_qk(module, alpha=qk_alpha)
        smoothed = True
    if smoothed:
        # Second pass, or derive it. SmoothQuant is output-neutral (it folds s into the
        # weight and divides the input by it), so every layer sees the SAME input it saw in
        # the first pass and the only thing this pass learns is each smoothed observer's
        # own range, over x / s. That is recoverable from the signed per-channel range the
        # first pass already recorded -- so a whole extra sweep of the corpus, half of
        # calibration, is avoidable. Falls back to re-observing whenever any smoothed
        # observer cannot derive it (a percentile clip needs a histogram of the smoothed
        # tensor, which per-channel extremes cannot reconstruct).
        smoothed_observers = _smoothed_observers(module)
        if smoothed_observers and all(o.can_rescale() for o, _ in smoothed_observers):
            for observer, scale in smoothed_observers:
                observer.rescale_from_channels(scale)
        else:
            calibrate(module, calibration)  # per-tensor scales must see smoothed inputs
    enable_w8a8(module)

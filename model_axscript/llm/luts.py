# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Build the kernel's per-site lookup tables from recorded input ranges.

Each nonlinearity (RMSNorm rsqrt, MLP silu, softmax exp / 1/x) is
approximated on hardware by a chained piecewise-linear LUT. This turns a
(function, [min, max]) pair into the packed (n_luts, 64) table the kernel
consumes. A table's first dim is the number of chained LUTs; each row is one
64-element PWORD ([16 bin_delims, 16 slopes, 16 offsets, 16 zeros]).
"""

from __future__ import annotations

import functools
import math

from axelera.compiler.alto.axol.utils.lut_chain import (
    ChainedLUTMethod,
    compute_chained_lut_params,
    fit_exp_lut_params_sdpa_softmax,
    fit_inv_lut_params,
    pack_lut_params_to_pword,
)
import numpy as np
from tvm.relay.backend.contrib.axelera.utils.float18 import (
    Float18Params,
    np_round_float32_to_float18,
)
from tvm.relay.backend.contrib.axelera.utils.lut import LUTParams

# Smallest positive domain start for the functions that blow up at 0.
_POS_EPS = 1e-4

# The int8 rounding floor for softmax weights is exp(-5.545) = 0.5/127 (5.545 = 8*ln2, i.e.
# exp = 2^-8): a *weight* past that gap quantises to zero in the int8 P@V numerator.
#
# BUT the softmax DENOMINATOR sums the *unquantized* exp weights, and the sub-floor tail -- while
# negligible per weight -- has an aggregate mass that GROWS with sequence length (~4% of the
# denominator at seq 257, ~10% at seq 1408). Capping the exp domain at the int8 floor zeroes that
# tail, under-counting the denominator and distorting long-sequence softmax; the per-layer error is
# small but compounds across layers (it collapsed DINOv2 @518 features to a gray PCA). So extend the
# exp domain past the floor to retain the tail -- exp(-10) ~ 4.5e-5 is negligible even summed over
# thousands of keys. The tail bins must be fit NON-quant-aware (see the exp branch below), since the
# quant-aware fit is degenerate below the int8 floor.
_EXP_DOMAIN_MAX = 10.0


def generate_lut_table(function: str, lo: float, hi: float, n_luts: int) -> np.ndarray:
    """Return the (n_luts, 64) float32 LUT table for *function* over [lo, hi].

    *function* is one of "inv_sqrt", "silu", "gelu", "exp", "reciprocal".
    Functions singular at the origin clamp the lower bound to a small positive
    value; exp flips its recorded s <= 0 range to [0, -lo] to match
    the fitter's exp(-x) parameterisation.
    """
    lo, hi = float(lo), float(hi)
    if function == "exp":
        # Recorded range is for exp(s), s in [lo, hi] with hi ~ 0, lo < 0. The fitter parameterises
        # exp(-x) over x in [0, -lo]. Extend the domain past the int8 floor to _EXP_DOMAIN_MAX so the
        # sub-floor tail is retained for the softmax denominator (see _EXP_DOMAIN_MAX above). Fit
        # NON-quant-aware: the quant-aware offset optimiser is degenerate past the int8 floor (every
        # value rounds to 0 -> flat objective -> garbage/negative tail offsets), whereas the plain
        # secant+midpoint fit keeps the small positive tail values the denominator needs. (Verified:
        # non-quant-aware @ cap 10 restores per-layer softmax to ~1.0 at both seq 257 and 1408,
        # vs 0.96 at seq 1408 for the old capped-at-floor quant-aware fit.)
        hi_x = min(max(-lo, _POS_EPS), _EXP_DOMAIN_MAX)
        luts = compute_chained_lut_params(
            0.0,
            hi_x,
            n_luts,
            fit_fn=functools.partial(fit_exp_lut_params_sdpa_softmax, quant_aware=False),
        )
    elif function == "reciprocal":
        # Softmax denominator: 1/x with x = sum of exp weights (>= 1).
        luts = compute_chained_lut_params(
            max(lo, 1.0), max(hi, 1.0 + _POS_EPS), n_luts, fit_fn=fit_inv_lut_params
        )
        luts = _extrapolate_last_bin(luts)
    elif function == "inv_sqrt":
        # Geometric spacing + error-centering, fit over the observed range. Out-of-range HIGH
        # variance (deep-layer / out-of-distribution activations that exceed the calibrated max)
        # must be SATURATED to the small positive endpoint 1/sqrt(var_max), NOT linearly
        # extrapolated: 1/sqrt is decreasing, so the last bin has a negative slope, and extending
        # that line past ~2.7x var_max drives 1/sqrt NEGATIVE. A negative LayerNorm 1/std sign-flips
        # AND amplifies (instead of normalising), seeding a layer-to-layer runaway that blows the
        # activations to +-thousands -- this collapsed DINOv2 @518 on native-detail images whose
        # variances exceed the range calibrated on smoother images. Saturating keeps it bounded and
        # positive, while still not zeroing (which would kill the activation).
        luts = compute_chained_lut_params(
            max(lo, 1e-8), max(hi, 2e-8), n_luts, fit_fn=_fit_invsqrt
        )
        luts = _saturate_high_tail(luts)
    elif function == "silu":
        # Fit over the calibrated range (clamped to at least +/-6 so the curve minimum
        # near x=-1.3 and a reasonable tail are always covered) rather than a fixed
        # +/-12: the wide fixed domain wasted the 31 bins and left a systematic bias
        # near the SiLU minimum that compounds across layers (the dominant device-ppl
        # error for LFM2.5). Tails extrapolate via _extrapolate_last_bin.
        slo = -6.0
        shi = 6.0
        luts = compute_chained_lut_params(
            slo, shi, n_luts, func=_silu, method=ChainedLUTMethod.ADDITIVE
        )
        # Anchor both tails to silu's asymptotes so out-of-range inputs stay correct
        # (lets the tight domain above stay safe for wider-range models, e.g. qwen3):
        # left -> 0 (constant asymptote), right -> ~x (last segment's slope-1 line).
        luts = _saturate_low_tail(luts)
        luts = _extrapolate_last_bin(luts)
    elif function in ("gelu", "gelu_tanh"):
        # GELU (the ViT/DINOv2 MLP activation) has the same shape as SiLU -- a
        # constant 0 left asymptote and a slope-1 right asymptote, essentially done
        # by |x| ~ 6. Clamp the calibrated range INTO +/-3 so a few outlier activations
        # can't stretch the fit domain and starve the ~+/-3 region (where almost all
        # post-LayerNorm gelu inputs live) of LUT resolution. Beyond +/-3 the asymptote
        # tails handle it (left -> 0, right -> ~x) via the same helpers as silu.
        #
        # Two curves, one fitter: "gelu" is the exact (erf) form (DINOv2, Qwen3-VL's patch
        # merger), "gelu_tanh" the tanh approximation HF evaluates in the Qwen3-VL vision
        # block MLP. Over +/-3 they differ by ~5e-4, under this table's own fit error, so
        # the point is matching the reference curve, not accuracy.
        slo = max(lo, -3.0)
        shi = min(hi, 3.0)
        luts = compute_chained_lut_params(
            slo,
            shi,
            n_luts,
            func=_gelu_tanh if function == "gelu_tanh" else _gelu,
            method=ChainedLUTMethod.ADDITIVE,
        )
        luts = _saturate_low_tail(luts)
        luts = _extrapolate_last_bin(luts)
    else:
        raise ValueError(f"unknown LUT function {function!r}")

    table = np.stack([pack_lut_params_to_pword(p) for p in luts]).astype(np.float32)
    assert table.shape == (n_luts, 64), table.shape
    return table


@functools.lru_cache(maxsize=512)
def _cached_lut_table(function: str, lo: float, hi: float, n_luts: int) -> np.ndarray:
    return generate_lut_table(function, lo, hi, n_luts)


def apply_chained_lut(x, function: str, lo: float, hi: float, n_luts: int = 2):
    """Evaluate the device's chained PWL LUT for *function* on torch tensor *x*, matching the
    kernel: ADDITIVE chaining sums each hardware LUT's piecewise-linear output, with float18
    rounding (via the production ``compute_lut_numpy_single_batch`` primitive). Returns a torch
    tensor. Lets the torch fake-quant reference model the device's exp / 1x / inv_sqrt / gelu
    approximations instead of exact fp, isolating LUT error from true device bugs."""
    import numpy as np
    import torch
    from tvm.relay.backend.contrib.axelera.utils.lut import compute_lut_numpy_single_batch

    table = _cached_lut_table(function, float(lo), float(hi), n_luts)
    xn = x.detach().to(torch.float32).cpu().numpy()
    out = np.zeros_like(xn, dtype=np.float32)
    for row in table:
        out += compute_lut_numpy_single_batch(xn, row[0:16], row[16:32], row[32:48], True)
    return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)


def _saturate_low_tail(luts: list) -> list:
    """Flatten the chain's first bin to 0 so inputs below the fitted range read ~0
    instead of extrapolating its (nonzero-slope) segment off in the wrong direction.

    Low-side mirror of _extrapolate_last_bin. The hardware applies bin 0's linear
    segment for every x below the first delimiter. For silu -- whose LEFT asymptote is
    a constant 0 (slope 0), unlike the positive tail's slope-1 line -- the segment fit
    near the low bound has a small NEGATIVE slope, so far-negative inputs drift the wrong
    way (e.g. x=-80 -> +1.05) instead of -> 0. Zeroing bin 0's slope+offset anchors the
    whole (-inf, delim0] sub-range to 0, matching silu(-inf)=0. That is what lets a tight
    fit domain (e.g. the +/-6 floor) stay safe for models whose activations run past it,
    rather than trading curve resolution for tail coverage (see generate_lut_table)."""
    p0 = luts[0]
    slopes = np.asarray(p0.slope_values, dtype=np.float32).copy()
    offsets = np.asarray(p0.offset_values, dtype=np.float32).copy()
    slopes[0] = np.float32(0.0)
    offsets[0] = np.float32(0.0)
    return [LUTParams(np.asarray(p0.bin_delim_values), slopes, offsets)] + list(luts[1:])


def _extrapolate_last_bin(luts: list) -> list:
    """Extend the chain's last real bin to +inf so out-of-range inputs use its
    linear segment instead of being zeroed.

    The hardware LUT outputs 0 beyond its last delimiter — fine for exp
    (decays to 0) but catastrophic for inv_sqrt / reciprocal / silu,
    which stay large past the calibrated range (e.g. deep-layer RMSNorm variance
    exceeds the recorded range, giving a dead activation). Pushing the last
    delimiter to largest_value keeps the bulk fit and extrapolates beyond it.
    """
    last = luts[-1]
    delims = np.asarray(last.bin_delim_values, dtype=np.float32).copy()
    real = np.flatnonzero(delims < Float18Params.largest_value)
    if real.size:
        delims[real[-1]] = np.float32(Float18Params.largest_value)
    return luts[:-1] + [LUTParams(delims, last.slope_values, last.offset_values)]


def _saturate_high_tail(luts: list) -> list:
    """High-side counterpart of _extrapolate_last_bin, for DECREASING positive functions like
    1/sqrt (LayerNorm 1/std). _extrapolate_last_bin keeps the last bin's slope and extends the line
    to +inf; but 1/sqrt's last bin has a NEGATIVE slope, so past ~2.7x var_max the line crosses zero
    and goes negative. A negative 1/std makes LayerNorm sign-flip and amplify instead of normalise,
    which seeds a layer-to-layer runaway that blows activations to +-thousands on inputs whose
    variance exceeds the calibrated range. Instead, flatten EACH chained LUT's last real bin to its
    value at var_max and extend it to +inf, so out-of-range-high inputs saturate to the small,
    positive endpoint 1/sqrt(var_max): bounded, never negative, and not zero (which would kill the
    activation). Costs ~1 bin of resolution at the highest-variance end, where 1/sqrt is flattest.

    Like _extrapolate_last_bin, this modifies ONLY the last chained LUT (extending any earlier LUT
    would inject a spurious constant across the later LUTs' sub-ranges and corrupt the in-range
    fit)."""
    last = luts[-1]
    delims = np.asarray(last.bin_delim_values, dtype=np.float32).copy()
    slopes = np.asarray(last.slope_values, dtype=np.float32).copy()
    offsets = np.asarray(last.offset_values, dtype=np.float32).copy()
    real = np.flatnonzero(delims < Float18Params.largest_value)
    if real.size:
        j = int(real[-1])
        cap = np.float32(slopes[j] * float(delims[j]) + offsets[j])  # value at var_max (this LUT)
        slopes[j] = np.float32(0.0)
        offsets[j] = cap
        delims[j] = np.float32(Float18Params.largest_value)
    return luts[:-1] + [LUTParams(delims, slopes, offsets)]


def _fit_invsqrt(x_min: float, x_max: float, N: int, round_f18: bool = True) -> LUTParams:
    """Geometric-spaced, error-centered PWL fit of 1/sqrt(x) over [x_min, x_max].

    Ported from production fit_invsqrt: geometric breakpoints give equal
    relative error per segment, and subtracting half the per-segment peak error
    centers the otherwise-high bias of a convex function's secant approximation.
    """
    x_max = max(x_max, x_min * 1.0001)
    bp = np.geomspace(x_min, x_max, N + 1)
    delims = bp[1:].astype(np.float32)
    slopes = np.zeros(N, dtype=np.float32)
    offsets = np.zeros(N, dtype=np.float32)
    for i in range(N):
        x0, x1 = float(bp[i]), float(bp[i + 1])
        y0, y1 = 1.0 / np.sqrt(x0), 1.0 / np.sqrt(x1)
        slope = (y1 - y0) / (x1 - x0)
        offset = y0 - slope * x0
        x_mid = (x0 + x1) / 2
        max_error = (slope * x_mid + offset) - 1.0 / np.sqrt(x_mid)
        offset = max(offset - max_error / 2, 0.0)
        slopes[i] = slope
        offsets[i] = offset
    if round_f18:
        delims = np_round_float32_to_float18(delims)
        slopes = np_round_float32_to_float18(slopes)
        offsets = np_round_float32_to_float18(offsets)
    return LUTParams(delims, slopes, offsets)


def _silu(x: np.ndarray) -> np.ndarray:
    # Clip the exponent to keep exp() in range (silu(-large) -> 0 either way).
    return x / (1.0 + np.exp(-np.clip(x, -80.0, 80.0)))


# Exact (erf-based) GELU, matching torch nn.GELU(approximate="none") -- the
# DINOv2 / HF ViT default. math.erf is scalar-only, so vectorise it.
_erf = np.vectorize(math.erf)


def _gelu(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return (0.5 * x * (1.0 + _erf(x / np.sqrt(2.0)))).astype(np.float32)


def _gelu_tanh(x: np.ndarray) -> np.ndarray:
    """The tanh approximation -- torch's nn.GELU(approximate="tanh"), HF's
    `gelu_pytorch_tanh`, which Qwen3-VL's vision block MLP evaluates."""
    x = np.asarray(x, dtype=np.float64)
    inner = np.sqrt(2.0 / np.pi) * (x + 0.044715 * x**3)
    return (0.5 * x * (1.0 + np.tanh(inner))).astype(np.float32)

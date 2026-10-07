# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Export calibrated quantization parameters as plain {name: tensors} dicts."""

from __future__ import annotations

import torch

from .modules import LayerNorm1P, QuantSDPA, RMSNorm
from .quant import QuantConv1d, QuantLinear, RangeObserver, row_bands

#: Elements of a weight the export quantizes at once. Smaller than quant's own budget: this
#: runs at the end of a build, when the calibration GPU is at its fullest. 4M elements is
#: 16 MB per fp32 copy.
_EXPORT_BAND_ELEMS = 4 << 20


@torch.no_grad()
def export_linear_quant_params(module: torch.nn.Module) -> dict[str, dict]:
    """Returns {linear_path: {qs, qz, w, ds, dz, qmin, qmax}} for every QuantLinear.

    Folds SmoothQuant into the activation scale (qs = 1/(smooth*act_scale)).
    The module must have been calibrated with activation observers holding ranges.
    """
    params: dict[str, dict] = {}
    for name, layer in module.named_modules():
        if not isinstance(layer, QuantLinear):
            continue
        if int(layer.act_observer.num_batches.item()) == 0:
            raise RuntimeError(f"{name}: calibrate before exporting (no activation stats)")
        act_scale, act_zp = layer.act_observer.qparams()
        act_scale = act_scale.reshape(())  # per-tensor scalar
        qz = int(act_zp.item())
        smooth = layer.smooth_scale  # [in_features]
        wqmax = layer.weight_qmax
        # In output-row bands, landing each band on the HOST as it is produced. Export runs
        # once and its result goes to the host anyway (build_quantized_model wraps this in
        # _cpu_tree), while on the device every temporary here -- and the int8 result itself
        # -- is the size of the weight: 1.9 GiB and 486 MiB for a 248320x2048 lm_head, and by
        # export time the calibration GPU has neither free. Bands leave the scales (a few
        # floats per output row) where they were; only the big tensors move.
        bands = row_bands(layer.weight, _EXPORT_BAND_ELEMS)
        w_int = torch.empty(layer.weight.shape, dtype=torch.int8, device="cpu")
        scales, dz_bands = [], []
        for band in bands:
            band_scale = layer.weight_scale(band)  # [band, 1]
            scales.append(band_scale.reshape(-1))
            q = torch.clamp(torch.round(layer.weight[band] / band_scale), -wqmax, wqmax).to(
                torch.int8
            )
            w_int[band] = q.cpu()
            dz_bands.append(q.to(torch.int32).sum(dim=1).cpu())
        weight_scale = torch.cat(scales)  # [out_features]

        in_features = smooth.numel()
        params[name] = {
            # Activation: q = round(x * qs + qz), clamped to [qmin, qmax].
            "qs": (1.0 / (smooth * act_scale)).contiguous(),  # [in_features]
            "qz": torch.full((in_features,), qz, dtype=torch.int32),  # [in_features]
            "w": w_int,  # [out_features, in_features] int8
            # Output dequant: out = (acc - dz) * ds, dz = qz * sum_j w.
            "ds": (act_scale * weight_scale).contiguous(),  # [out_features]
            "dz": qz * torch.cat(dz_bands),  # [out_features], summed per band above
            "qmin": layer.act_observer.qmin,
            "qmax": layer.act_observer.qmax,
        }
    return params


@torch.no_grad()
def export_conv_weights(module: torch.nn.Module) -> dict[str, dict]:
    """Returns {conv_path: {"weight": weight}} for every full-precision depthwise
    nn.Conv1d (the DeltaNet short-conv mixer). The weight is a per-channel FIR tap
    kept as a plain float tensor (groups == channels), shape [channels, 1,
    kernel_size]; the kernel repacks it channel-major. Quantized convs (QuantConv1d)
    are handled by export_conv_quant_params instead.
    """
    return {
        name: {"weight": layer.weight.detach()}
        for name, layer in module.named_modules()
        if isinstance(layer, torch.nn.Conv1d)
    }


@torch.no_grad()
def export_conv_quant_params(module: torch.nn.Module) -> dict[str, dict]:
    """Returns {conv_path: {qs, w, ds, qmin, qmax}} for every QuantConv1d.

    The W8A8 depthwise short-conv (LFM2): a PER-CHANNEL SYMMETRIC activation scale
    (zero-point 0; per-channel because the depthwise conv has no SmoothQuant, so a
    per-tensor scale would zero out the quiet channels of the gated B*x) and a per-
    output-channel symmetric weight scale (one per channel over its K FIR taps). The
    int8 DWPU computes per channel acc = sum_j q_x[t+j]*q_w[j]; with a zero activation
    zero-point the dequant is just out = acc * ds, ds = s_x*s_w (no dz correction).
    qs and ds are therefore per-channel [channels]; the int8 weight stays
    [channels, 1, K] for the channel-major packer.
    """
    params: dict[str, dict] = {}
    for name, layer in module.named_modules():
        if not isinstance(layer, QuantConv1d):
            continue
        if int(layer.act_observer.num_batches.item()) == 0:
            raise RuntimeError(f"{name}: calibrate before exporting (no activation stats)")
        act_scale, act_zp = layer.act_observer.qparams()  # per-channel [channels]
        if bool((act_zp != 0).any()):
            raise RuntimeError(
                f"{name}: QuantConv1d activation must be symmetric (zero-point 0) for the "
                f"DWPU conv; got a non-zero zero-point."
            )
        act_scale = act_scale.reshape(-1)  # [channels]
        weight_scale = layer.weight_scale().reshape(-1)  # [channels]
        wqmax = layer.weight_qmax
        w_int = torch.clamp(
            torch.round(layer.weight / weight_scale[:, None, None]), -wqmax, wqmax
        ).to(
            torch.int8
        )  # [channels, 1, K]

        params[name] = {
            # Activation: q = round(x * qs), clamped to [qmin, qmax] (per-channel, zp=0).
            "qs": (1.0 / act_scale).contiguous(),  # [channels]
            "w": w_int,  # [channels, 1, K] int8
            "ds": (act_scale * weight_scale).contiguous(),  # [channels] output dequant scale
            "qmin": layer.act_observer.qmin,
            "qmax": layer.act_observer.qmax,
        }
    return params


@torch.no_grad()
def export_deltanet_params(module: torch.nn.Module) -> dict[str, dict]:
    """Returns {deltanet_path: {dt_bias, A_log, norm_weight}} for every GatedDeltaNet.

    The DeltaNet's per-block scalars not covered by the generic linear/conv/norm
    exports: the delta-rule timestep bias (dt_bias) and log-decay (A_log,
    kept in log form -- the kernel forms -exp(A_log)), and the gated output
    norm's learned scale (a GatedRMSNorm, distinct from the shared RMSNorm
    that export_norm_weights captures). Duck-typed so this stays model-neutral.
    """
    params: dict[str, dict] = {}
    for name, layer in module.named_modules():
        if all(hasattr(layer, a) for a in ("dt_bias", "A_log", "norm", "in_proj_qkv")):
            entry = {
                "dt_bias": layer.dt_bias.detach().clone(),
                "A_log": layer.A_log.detach().clone(),
                "norm_weight": layer.norm.weight.detach().clone(),
            }
            # The Metis decode kernel's FIXED int8 requant scales (s = absmax/127),
            # calibrated during record_lut_ranges (set_decode_recording). Present iff
            # the calibration captured them; q/k are L2-normed and S decays, so fixed
            # scales suffice (no on-device max-reduce).
            qm = getattr(layer, "decode_q_absmax", None)
            if qm is not None and float(qm) > 0:
                qmax = 127.0
                entry["decode_s_q"] = (layer.decode_q_absmax / qmax).detach().clone()
                entry["decode_s_k"] = (layer.decode_k_absmax / qmax).detach().clone()
                entry["decode_s_S"] = (layer.decode_S_absmax / qmax).detach().clone()
            params[name] = entry
    return params


@torch.no_grad()
def export_sdpa_quant_params(module: torch.nn.Module) -> dict[str, dict]:
    """Returns the SDPA quant params for every quantized QuantSDPA.

    Q, K, attn_weight and V are zero-pinned (zp = 0), so each is a pure
    multiplicative scale. The QK-smoothing factor is folded into the Q/K scales
    (and cancels in Q·Kᵀ).
    """
    params: dict[str, dict] = {}
    for name, sdpa in module.named_modules():
        if not (isinstance(sdpa, QuantSDPA) and sdpa.quantize):
            continue
        if int(sdpa.q_observer.num_batches.item()) == 0:
            raise RuntimeError(f"{name}: calibrate before exporting (no SDPA stats)")
        q_scale = sdpa.q_observer.qparams()[0].reshape(())
        k_scale = sdpa.k_observer.qparams()[0].reshape(())
        w_scale = sdpa.w_observer.qparams()[0].reshape(())
        v_scale = sdpa.v_observer.qparams()[0].reshape(())  # per-tensor scalar
        params[name] = {
            "qs_q": (1.0 / (sdpa.q_smooth_scale * q_scale)).contiguous(),
            "qs_k": (1.0 / (sdpa.k_smooth_scale * k_scale)).contiguous(),
            "qs_w": 1.0 / w_scale,
            "qs_v": 1.0 / v_scale,
            "scores_dq_scale": q_scale * k_scale,
            "output_dq_scale": w_scale * v_scale,
            "scale": sdpa.scaling,
            "qmin": sdpa.q_observer.qmin,
            "qmax": sdpa.q_observer.qmax,
        }
    return params


@torch.no_grad()
def export_norm_weights(module: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Returns {norm_path: weight} for every RMSNorm, plus LayerNorm1P weight/bias.

    Only the learned affine parameters; the rsqrt is handled by a LUT (see
    export_lut_ranges).

    An RMSNorm has a single tensor, keyed by its bare module path
    (`...input_layernorm`). A LayerNorm1P has two, so they take explicit `.weight` /
    `.bias` suffixes (`...input_layernorm.weight`); the unit offset is folded in, so
    the exported gain is `weight + 1` and the device consumes it directly. Both
    survive layer_subset's re-keying unchanged.

    Matching RMSNorm ONLY was a silent bug waiting for the first non-RMSNorm model:
    every one of Domyn Small's 81 LayerNorm1P norms would have been dropped from the
    bundle with nothing to say so, and a dropped affine reads as quantization loss
    three steps later.
    """
    params: dict[str, torch.Tensor] = {}
    for name, norm in module.named_modules():
        if isinstance(norm, RMSNorm):
            params[name] = norm.weight.detach().clone()
        elif isinstance(norm, LayerNorm1P):
            params[f"{name}.weight"] = (norm.weight.detach() + 1.0).clone()
            params[f"{name}.bias"] = norm.bias.detach().clone()
    return params


@torch.no_grad()
def export_lut_ranges(module: torch.nn.Module) -> dict[str, dict]:
    """Returns {site_path: {function, min, max}} for each LUT nonlinearity.

    Only sites that recorded data are returned (so an unquantized-attention model
    omits exp / reciprocal). Run record_lut_ranges first.
    """
    ranges: dict[str, dict] = {}
    for name, observer in module.named_modules():
        if isinstance(observer, RangeObserver) and int(observer.num_batches.item()) > 0:
            ranges[name] = {
                "function": observer.function,
                "min": observer.min_val.item(),
                "max": observer.max_val.item(),
            }
    return ranges

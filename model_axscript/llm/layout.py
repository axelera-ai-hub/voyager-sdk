# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Helpers for mapping a torch model's exported params onto kernel buffers.

Reusable mechanics only: picking a layer's params, the x64 packer, and the
norm/LUT helpers. The mapping itself is plain code in the implementation (see
models/qwen3/naive_impl.decode_layer_buffers).
"""

from __future__ import annotations

import json
from pathlib import Path
import re

import torch

# A kernel buffer's target layout: (shape, dtype-string).
BufferSpec = tuple[tuple[int, ...], str]

# PWORD width: a norm weight buffer is (dim, PWORD) with each scalar broadcast
# across its row (see axol/core/norm/common.py); a LUT row is one PWORD.
PWORD = 64

_TORCH_DTYPE = {
    "int8": torch.int8,
    "int16": torch.int16,
    "int32": torch.int32,
    "float16": torch.float16,
    "float32": torch.float32,
    "uint64": torch.int64,
}


def extract_layer_buffer_spec(layer_model_json: str | Path) -> dict[str, BufferSpec]:
    """Read {logical_name: (shape, dtype)} from a compiled layer's model.json.

    Strips the trailing _<n> from input names (q_proj_qs_5 -> q_proj_qs).
    """
    model = json.loads(Path(layer_model_json).read_text())
    spec: dict[str, BufferSpec] = {}
    for buf in model.get("inputs", []):
        logical = re.sub(r"_\d+$", "", buf["name"])
        spec[logical] = (tuple(buf["shape"]), buf["dtype"])
    return spec


# Picking a layer's exported params


def layer_linears(linear_params: dict, layer_index: int) -> dict:
    """This layer's linear params, keyed by submodule leaf name (q_proj ...)."""
    tag = f".layers.{layer_index}."
    return {path.split(".")[-1]: p for path, p in linear_params.items() if tag in path}


def layer_sdpa(sdpa_params: dict, layer_index: int) -> dict:
    """This layer's SDPA params (a single {field: value} dict, or {})."""
    tag = f".layers.{layer_index}."
    for path, params in sdpa_params.items():
        if tag in path:
            return params
    return {}


def layer_conv(conv_params: dict, layer_index: int) -> dict:
    """This layer's depthwise-conv weights, keyed by submodule leaf name (conv ...)."""
    tag = f".layers.{layer_index}."
    return {path.split(".")[-1]: w for path, w in conv_params.items() if tag in path}


def pack_channel_major_conv(weight: torch.Tensor, spec: BufferSpec) -> torch.Tensor:
    """Pack a depthwise Conv1d weight [channels, 1, K] into the kernel's
    channel-major (K, C_blocks, PWORD) layout (lane j of block cb = channel
    cb*PWORD+j's tap), matching llm.causal_conv1d's weight contract.
    """
    channels, one, k = weight.shape
    if one != 1:
        raise ValueError(f"expected a depthwise conv weight [C, 1, K]; got {tuple(weight.shape)}")
    # [C, 1, K] -> [K, C] -> [K, C_blocks, PWORD].
    taps = weight.reshape(channels, k).transpose(0, 1).reshape(k, channels // PWORD, PWORD)
    return pack(taps, spec)


def pack_dw2d_conv_weight(weight: torch.Tensor, spec: BufferSpec) -> torch.Tensor:
    """Pack a depthwise Conv1d weight [channels, 1, K] into axol.core.dw_convolution_2d's
    channel-major (C_blocks, K, 1, 1) layout (lane j of block cb = channel cb*PWORD+j's
    tap), matching the int8 llm.causal_conv1d weight contract (which passes the taps
    straight to dw_convolution_2d as KH=K, KW=1). Host-side transpose: a device-side
    (K, C_blocks) -> (C_blocks, K, 1, 1) transpose mis-lowers for C_blocks > 1.
    """
    channels, one, k = weight.shape
    if one != 1:
        raise ValueError(f"expected a depthwise conv weight [C, 1, K]; got {tuple(weight.shape)}")
    # [C, 1, K] -> [C_blocks, PWORD, K] -> [C_blocks, K, PWORD].
    taps = weight.reshape(channels // PWORD, PWORD, k).transpose(1, 2)
    return pack(taps, spec)


def layer_subset(params: dict, layer_index: int) -> dict:
    """This layer's entries, re-keyed by the path relative to the layer.

    model.layers.3.input_layernorm -> input_layernorm. Used for norm
    weights and LUT ranges.
    """
    tag = f".layers.{layer_index}."
    return {path.split(tag, 1)[1]: value for path, value in params.items() if tag in path}


# Producing buffers


def pack(tensor: torch.Tensor, spec: BufferSpec) -> torch.Tensor:
    """Reshape/cast a tensor into a kernel buffer's x64 layout.

    A per-tensor scalar is broadcast to fill the buffer; otherwise the element
    count must already match spec's shape.

    Raises if the cast to the buffer's declared dtype turns a finite value infinite --
    narrowing a param table to fp16 is a legitimate optimization, but a dequant zero of
    ~1e5 (dz' = dz - bias/ds, which is large exactly when ds is small) overflows fp16 and
    the device then produces NaNs with nothing to point at. Fail at the bind instead.
    """
    shape, dtype = spec
    expected = 1
    for d in shape:
        expected *= d
    if tensor.numel() == 1 and expected > 1:
        tensor = tensor.reshape(1).expand(expected).contiguous()
    if tensor.numel() != expected:
        raise ValueError(
            f"cannot pack {tuple(tensor.shape)} ({tensor.numel()} elems) into {shape}"
        )
    out = tensor.reshape(shape).to(_TORCH_DTYPE[dtype]).contiguous()
    if out.is_floating_point():
        lost = torch.isfinite(tensor.reshape(shape)) & ~torch.isfinite(out)
        if lost.any():
            worst = tensor.reshape(shape)[lost].abs().max().item()
            raise ValueError(
                f"packing into {dtype} overflows: {int(lost.sum())} of {expected} values are "
                f"out of range (largest |x| = {worst:.3e}). Declare this buffer fp32."
            )
    return out


def zeros(spec: BufferSpec) -> torch.Tensor:
    """A buffer pinned to zeros (e.g. a zero-pinned SDPA zero point)."""
    shape, dtype = spec
    return torch.zeros(shape, dtype=_TORCH_DTYPE[dtype])


def broadcast_to_pword(weight: torch.Tensor) -> torch.Tensor:
    """Broadcast a [dim] per-feature scale to the (dim, PWORD) row layout."""
    dim = weight.numel()
    return weight.reshape(dim, 1).expand(dim, PWORD).contiguous()


def fit_lut(rng: dict, spec: BufferSpec) -> torch.Tensor:
    """Fit + pack the LUT table for a recorded nonlinearity range into spec.

    rng is {"function", "min", "max"}; the number of chained LUTs is the
    buffer's first dimension.
    """
    from .luts import generate_lut_table  # lazy import keeps layout alto-free

    (shape, _) = spec
    table = generate_lut_table(rng["function"], rng["min"], rng["max"], shape[0])
    return pack(torch.from_numpy(table), spec)

# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Llama 3.2 Metis implementation: the alto-script decoder-layer kernel + param
binding, using design-P block-interleaved padding for head_dim=64.

Llama 3.2 1B has head_dim=64, whose RoPE rotate_half pairs elements within a
single 64-lane PWORD — which the lane-aligned DPU cannot reach. Design P pads
each head to 2 PWORD blocks (block0 lanes[0:32]=elems 0..31, block1
lanes[0:32]=elems 32..63, pad lanes=0), reducing the kernel to the validated
head_dim_pw=2 regime. The padding is done entirely in host-side weight/table
packing (decode_layer_buffers); the torch model stays at the real head_dim. The
SDPA scale and RoPE frequencies use the LOGICAL head_dim.

Kept separate from the PyTorch reference: TVM Script can't run under
from __future__ import annotations, and its dtype-string annotations trip F821.
"""

# flake8: noqa: F821  -- A.Buffer dtype-string annotations look like F821 names

from typing import TYPE_CHECKING

import numpy as np

from ...implementation import (
    HostKernel,
    LayerKernel,
    UniformDecoderImplementation,
    register_implementation,
)

if TYPE_CHECKING:
    from .llama import LlamaConfig

PWORD = 64
# Llama 3.2 RoPE wavelength rescaling (used by the host RoPE tables at run time).
_LLAMA32_ROPE_SCALING = {
    "factor": 32.0,
    "low_freq_factor": 1.0,
    "high_freq_factor": 4.0,
    "original_max_position_embeddings": 8192,
}


def _design_p_dims(head_dim: int) -> dict:
    """Physical (padded) head dims for the design-P layout from a logical head_dim.

    half = head_dim/2 freqs occupy rope_dim_pw PWORD blocks; the head is padded to
    head_dim_pw = 2*rope_dim_pw blocks. For head_dim that is already a multiple of
    128 this is a no-op (padded == real).
    """
    half = head_dim // 2
    rope_dim_pw = (half + PWORD - 1) // PWORD
    head_dim_pw = 2 * rope_dim_pw
    return {
        "half": half,
        "rope_dim_pw": rope_dim_pw,
        "head_dim_pw": head_dim_pw,
        "padded_head_dim": head_dim_pw * PWORD,
    }


def make_layer_kernel(config: "LlamaConfig", ctx_len: int, phase: str):
    """Build the LlamaLayerFunctional IRModule for one Llama decoder layer.

    Dims come from config; head_dim is padded per design P. ctx_len sizes the KV
    cache / mask; phase is "decode" (seq_len 1) or "prefill" (seq_len = ctx_len).
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol.common import BufferInfo
    from axelera.compiler.alto.axol.utils.lut_chain import ChainedLUTMethod

    hidden_dim = config.hidden_size
    num_q_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    logical_head_dim = config.head_dim
    intermediate_dim = config.intermediate_size

    d = _design_p_dims(logical_head_dim)
    head_dim_pw = d["head_dim_pw"]
    rope_dim_pw = d["rope_dim_pw"]
    padded_head_dim = d["padded_head_dim"]

    hidden_dim_pw = hidden_dim // 64
    q_out_dim = num_q_heads * padded_head_dim
    q_out_dim_pw = q_out_dim // 64
    kv_out_dim = num_kv_heads * padded_head_dim
    kv_out_dim_pw = kv_out_dim // 64
    intermediate_dim_pw = intermediate_dim // 64

    context_len = ctx_len
    seq_len = 1 if phase == "decode" else ctx_len

    @A.ir_module
    class LlamaLayerFunctional:
        @A.prim_func
        def main(
            seq_pos: A.Buffer((1,), "uint64x64", scope=A.DDR),
            hidden_states: A.Buffer((1, seq_len, hidden_dim_pw), "float32x64", scope=A.DDR),
            rms1_w: A.Buffer((hidden_dim,), "float32x64", scope=A.DDR),
            rms1_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            q_proj_qs: A.Buffer((1, 1, hidden_dim_pw), "float32x64", scope=A.DDR),
            q_proj_qz: A.Buffer((1, 1, hidden_dim_pw), "float32x64", scope=A.DDR),
            q_proj_w: A.Buffer((hidden_dim, q_out_dim_pw), "int8x64", scope=A.DDR),
            q_proj_dz: A.Buffer((1, 1, q_out_dim_pw), "float32x64", scope=A.DDR),
            q_proj_ds: A.Buffer((1, 1, q_out_dim_pw), "float32x64", scope=A.DDR),
            k_proj_qs: A.Buffer((1, 1, hidden_dim_pw), "float32x64", scope=A.DDR),
            k_proj_qz: A.Buffer((1, 1, hidden_dim_pw), "float32x64", scope=A.DDR),
            k_proj_w: A.Buffer((hidden_dim, kv_out_dim_pw), "int8x64", scope=A.DDR),
            k_proj_dz: A.Buffer((1, 1, kv_out_dim_pw), "float32x64", scope=A.DDR),
            k_proj_ds: A.Buffer((1, 1, kv_out_dim_pw), "float32x64", scope=A.DDR),
            v_proj_qs: A.Buffer((1, 1, hidden_dim_pw), "float32x64", scope=A.DDR),
            v_proj_qz: A.Buffer((1, 1, hidden_dim_pw), "float32x64", scope=A.DDR),
            v_proj_w: A.Buffer((hidden_dim, kv_out_dim_pw), "int8x64", scope=A.DDR),
            v_proj_dz: A.Buffer((1, 1, kv_out_dim_pw), "float32x64", scope=A.DDR),
            v_proj_ds: A.Buffer((1, 1, kv_out_dim_pw), "float32x64", scope=A.DDR),
            k_rope_sin: A.Buffer((1, seq_len, rope_dim_pw), "float32x64", scope=A.DDR),
            k_rope_cos: A.Buffer((1, seq_len, rope_dim_pw), "float32x64", scope=A.DDR),
            q_rope_sin: A.Buffer((1, seq_len, rope_dim_pw), "float32x64", scope=A.DDR),
            q_rope_cos: A.Buffer((1, seq_len, rope_dim_pw), "float32x64", scope=A.DDR),
            k_cache_qs: A.Buffer((1, 1, 1, head_dim_pw), "float32x64", scope=A.DDR),
            k_cache_qz: A.Buffer((1, 1, 1, head_dim_pw), "float32x64", scope=A.DDR),
            v_cache_qs: A.Buffer((1, 1, 1, head_dim_pw), "float16x64", scope=A.DDR),
            v_cache_qz: A.Buffer((1, 1, 1, head_dim_pw), "float16x64", scope=A.DDR),
            k_cache: A.Buffer((1, num_kv_heads, context_len, head_dim_pw), "int8x64", scope=A.DDR),
            v_cache: A.Buffer((1, num_kv_heads, context_len, head_dim_pw), "int8x64", scope=A.DDR),
            q_attn_qs: A.Buffer((1, 1, 1, head_dim_pw), "float32x64", scope=A.DDR),
            q_attn_qz: A.Buffer((1, 1, 1, head_dim_pw), "float32x64", scope=A.DDR),
            sdpa_exp_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            sdpa_inv_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            sdpa_scores_dq_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            sdpa_weights_q_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            sdpa_output_dq_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            attn_mask: A.Buffer(
                (1, num_q_heads, seq_len, context_len // 64), "float32x64", scope=A.DDR
            ),
            o_proj_qs: A.Buffer((1, 1, q_out_dim_pw), "float16x64", scope=A.DDR),
            o_proj_qz: A.Buffer((1, 1, q_out_dim_pw), "float16x64", scope=A.DDR),
            o_proj_w: A.Buffer((q_out_dim, hidden_dim_pw), "int8x64", scope=A.DDR),
            o_proj_dz: A.Buffer((1, 1, hidden_dim_pw), "float32x64", scope=A.DDR),
            o_proj_ds: A.Buffer((1, 1, hidden_dim_pw), "float32x64", scope=A.DDR),
            residual_in: A.Buffer((1, seq_len, hidden_dim_pw), "float32x64", scope=A.DDR),
            rms2_w: A.Buffer((hidden_dim,), "float32x64", scope=A.DDR),
            rms2_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            gate_proj_qs: A.Buffer((1, 1, hidden_dim_pw), "float32x64", scope=A.DDR),
            gate_proj_qz: A.Buffer((1, 1, hidden_dim_pw), "float32x64", scope=A.DDR),
            gate_proj_w: A.Buffer((hidden_dim, intermediate_dim_pw), "int8x64", scope=A.DDR),
            gate_proj_dz: A.Buffer((1, 1, intermediate_dim_pw), "float32x64", scope=A.DDR),
            gate_proj_ds: A.Buffer((1, 1, intermediate_dim_pw), "float32x64", scope=A.DDR),
            silu_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            up_proj_qs: A.Buffer((1, 1, hidden_dim_pw), "float32x64", scope=A.DDR),
            up_proj_qz: A.Buffer((1, 1, hidden_dim_pw), "float32x64", scope=A.DDR),
            up_proj_w: A.Buffer((hidden_dim, intermediate_dim_pw), "int8x64", scope=A.DDR),
            up_proj_dz: A.Buffer((1, 1, intermediate_dim_pw), "float32x64", scope=A.DDR),
            up_proj_ds: A.Buffer((1, 1, intermediate_dim_pw), "float32x64", scope=A.DDR),
            down_proj_qs: A.Buffer((1, 1, intermediate_dim_pw), "float32x64", scope=A.DDR),
            down_proj_qz: A.Buffer((1, 1, intermediate_dim_pw), "float32x64", scope=A.DDR),
            down_proj_w: A.Buffer((intermediate_dim, hidden_dim_pw), "int8x64", scope=A.DDR),
            down_proj_dz: A.Buffer((1, 1, hidden_dim_pw), "float32x64", scope=A.DDR),
            down_proj_ds: A.Buffer((1, 1, hidden_dim_pw), "float32x64", scope=A.DDR),
        ):  # noqa: F821
            eps_const = A.alloc_constant(
                [9.9999999747524271e-07] * 64, dtype="float32x64", scope=A.DDR
            )
            sdpa_weights_q_zp = A.alloc_constant(
                np.array([0] * 64).reshape((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR
            )
            sdpa_weights_q_qmin = A.alloc_constant(
                np.array([-128] * 64).reshape((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR
            )
            sdpa_weights_q_qmax = A.alloc_constant(
                np.array([127] * 64).reshape((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR
            )
            sdpa_query_q_zp = A.alloc_constant(
                np.array([0] * 64).reshape((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR
            )
            sdpa_value_q_zp = A.alloc_constant(
                np.array([0] * 64).reshape((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR
            )

            # RMS Norm 1
            rms1 = axol.core.rms_norm_kahan(
                hidden_states,
                rms1_lut,
                eps_const[0:1],
                (hidden_dim,),
                weight_buffer=rms1_w,
                invsqrt_method="additive",
            )

            # Q projection (output padded to head_dim_pw per head)
            q0 = axol.atomic.mul(rms1, q_proj_qs)
            q1 = axol.atomic.add(q0, q_proj_qz)
            q2 = axol.atomic.dpu_cast(q1, output=BufferInfo(dtype="int8x64"))
            q3 = axol.core.matmul(q2, q_proj_w)
            q4 = axol.atomic.sub(q3, q_proj_dz)
            q5 = axol.atomic.mul(q4, q_proj_ds)
            q5_4d = A.alloc_buffer(
                (1, seq_len, num_q_heads, head_dim_pw), "float32x64", data=q5.data, scope=A.DDR
            )
            q6 = axol.atomic.dpu_cast(q5_4d, output=BufferInfo(dtype="float16x64"))
            q_heads = axol.core.transpose(q6, dims=(0, 2, 1, 3))

            # K projection
            k0 = axol.atomic.mul(rms1, k_proj_qs)
            k1 = axol.atomic.add(k0, k_proj_qz)
            k2 = axol.atomic.dpu_cast(k1, output=BufferInfo(dtype="int8x64"))
            k3 = axol.core.matmul(k2, k_proj_w)
            k4 = axol.atomic.sub(k3, k_proj_dz)
            k5 = axol.atomic.mul(k4, k_proj_ds)
            k5_4d = A.alloc_buffer(
                (1, seq_len, num_kv_heads, head_dim_pw), "float32x64", data=k5.data, scope=A.DDR
            )
            k6 = axol.atomic.dpu_cast(k5_4d, output=BufferInfo(dtype="float16x64"))
            k_heads = axol.core.transpose(k6, dims=(0, 2, 1, 3))

            # V projection
            v0 = axol.atomic.mul(rms1, v_proj_qs)
            v1 = axol.atomic.add(v0, v_proj_qz)
            v2 = axol.atomic.dpu_cast(v1, output=BufferInfo(dtype="int8x64"))
            v3 = axol.core.matmul(v2, v_proj_w)
            v4 = axol.atomic.sub(v3, v_proj_dz)
            v5 = axol.atomic.mul(v4, v_proj_ds)
            v5_4d = A.alloc_buffer(
                (1, seq_len, num_kv_heads, head_dim_pw), "float32x64", data=v5.data, scope=A.DDR
            )
            v6 = axol.atomic.dpu_cast(v5_4d, output=BufferInfo(dtype="float16x64"))
            v_heads = axol.core.transpose(v6, dims=(0, 2, 1, 3))

            # RoPE (no QK-Norm for Llama). trig: [sin_block | cos_block], head_dim/2
            # freqs in the low lanes of each block (design-P padded head layout).
            trig_k = axol.core.concat(k_rope_sin, k_rope_cos, dim=2)
            k_rope = axol.core.rope(k_heads, trig_k)
            trig_q = axol.core.concat(q_rope_sin, q_rope_cos, dim=2)
            q_rope = axol.core.rope(q_heads, trig_q)

            # Quantize K for cache (zero-pinned: pad lanes stay zero)
            kq0 = axol.atomic.mul(k_rope, k_cache_qs)
            kq1 = axol.atomic.add(kq0, k_cache_qz)
            k_update = axol.atomic.dpu_cast(kq1, output=BufferInfo(dtype="int8x64"))

            # Quantize V for cache
            vq0 = axol.atomic.mul(v_heads, v_cache_qs)
            vq1 = axol.atomic.add(vq0, v_cache_qz)
            vq2 = axol.atomic.dpu_cast(vq1)
            v_update = axol.atomic.dpu_cast(vq2, output=BufferInfo(dtype="int8x64"))

            k_out, v_out = axol.core.update_kv_cache(k_update, v_update, k_cache, v_cache, seq_pos)

            # Quantize Q for attention
            qq0 = axol.atomic.mul(q_rope, q_attn_qs)
            qq1 = axol.atomic.add(qq0, q_attn_qz)
            q_attn = axol.atomic.dpu_cast(qq1, output=BufferInfo(dtype="int8x64"))

            # SDPA — scale uses the LOGICAL head_dim (padding adds only zeros).
            attn_out = axol.core.scaled_dot_product_attention(
                q_attn,
                k_out,
                v_out,
                sdpa_scores_dq_scale,
                sdpa_exp_lut,
                sdpa_inv_lut,
                sdpa_weights_q_scale,
                sdpa_weights_q_zp,
                sdpa_weights_q_qmin,
                sdpa_weights_q_qmax,
                sdpa_output_dq_scale,
                sdpa_query_q_zp,
                sdpa_value_q_zp,
                attn_mask,
                enable_gqa=True,
                use_newton_raphson=False,
                scale=1.0 / (logical_head_dim**0.5),
                neg_exp_method=ChainedLUTMethod.ADDITIVE,
                inv_method=ChainedLUTMethod.ADDITIVE,
            )

            attn_t = axol.core.transpose(attn_out, dims=(0, 2, 1, 3))
            attn_c = axol.atomic.dpu_cast(attn_t, output=BufferInfo(dtype="float16x64"))
            attn_flat = A.alloc_buffer(
                (1, seq_len, q_out_dim_pw), "float16x64", data=attn_c.data, scope=A.DDR
            )

            # Output projection (input padded; pad columns are zeroed in packing)
            o0 = axol.atomic.mul(attn_flat, o_proj_qs)
            o1 = axol.atomic.add(o0, o_proj_qz)
            o2 = axol.atomic.dpu_cast(o1, output=BufferInfo(dtype="float32x64"))
            o3 = axol.atomic.dpu_cast(o2, output=BufferInfo(dtype="int8x64"))
            o4 = axol.core.matmul(o3, o_proj_w)
            o5 = axol.atomic.sub(o4, o_proj_dz)
            o6 = axol.atomic.mul(o5, o_proj_ds)

            residual = axol.atomic.add(residual_in, o6)

            # RMS Norm 2
            rms2 = axol.core.rms_norm_kahan(
                residual,
                rms2_lut,
                eps_const[0:1],
                (hidden_dim,),
                weight_buffer=rms2_w,
                invsqrt_method="additive",
            )

            # MLP gate + SiLU
            g0 = axol.atomic.mul(rms2, gate_proj_qs)
            g1 = axol.atomic.add(g0, gate_proj_qz)
            g2 = axol.atomic.dpu_cast(g1, output=BufferInfo(dtype="int8x64"))
            g3 = axol.core.matmul(g2, gate_proj_w)
            g4 = axol.atomic.sub(g3, gate_proj_dz)
            g5 = axol.atomic.mul(g4, gate_proj_ds)
            gate = axol.core.piecewise_linear_approximation(g5, silu_lut, "additive")

            # MLP up
            u0 = axol.atomic.mul(rms2, up_proj_qs)
            u1 = axol.atomic.add(u0, up_proj_qz)
            u2 = axol.atomic.dpu_cast(u1, output=BufferInfo(dtype="int8x64"))
            u3 = axol.core.matmul(u2, up_proj_w)
            u4 = axol.atomic.sub(u3, up_proj_dz)
            up = axol.atomic.mul(u4, up_proj_ds)

            # Gate * Up + down
            dd0 = axol.atomic.mul(gate, up)
            dd1 = axol.atomic.mul(dd0, down_proj_qs)
            dd2 = axol.atomic.add(dd1, down_proj_qz)
            dd3 = axol.atomic.dpu_cast(dd2, output=BufferInfo(dtype="int8x64"))
            dd4 = axol.core.matmul(dd3, down_proj_w)
            dd5 = axol.atomic.sub(dd4, down_proj_dz)
            dd6 = axol.atomic.mul(dd5, down_proj_ds)

            A.output(axol.atomic.add(residual, dd6))

    return LlamaLayerFunctional


# Projection names match the kernel buffer prefixes and torch submodule leaf names.
_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# q/k/v_proj pad their OUTPUT (head) dim; o_proj pads its INPUT (head) dim.
_HEAD_OUT_PROJ = ("q_proj", "k_proj", "v_proj")
_HEAD_IN_PROJ = ("o_proj",)
# SDPA per-head-dim quant scales: kernel buffer prefix -> exported SDPA field.
_SDPA_QS = {"q_attn": "qs_q", "k_cache": "qs_k", "v_cache": "qs_v"}
# SDPA per-tensor scales the kernel takes as broadcast single-PWORD inputs.
_SDPA_SCALARS = {
    "sdpa_scores_dq_scale": "scores_dq_scale",
    "sdpa_weights_q_scale": "qs_w",
    "sdpa_output_dq_scale": "output_dq_scale",
}
# RMSNorm-weight buffer -> the RMSNorm site (no QK-Norm for Llama).
_NORM_SITES = {"rms1_w": "input_layernorm", "rms2_w": "post_attention_layernorm"}
# LUT buffer -> the RangeObserver site whose recorded range the table is fit over.
_LUT_SITES = {
    "rms1_lut": "input_layernorm.inv_sqrt_range",
    "rms2_lut": "post_attention_layernorm.inv_sqrt_range",
    "sdpa_exp_lut": "self_attn.sdpa.exp_range",
    "sdpa_inv_lut": "self_attn.sdpa.reciprocal_range",
    "silu_lut": "mlp.silu_range",
}


def _interleave_indices(n_heads: int, head_dim: int, padded_head_dim: int) -> np.ndarray:
    """Map real head element (h, e) -> padded physical channel index.

    block = e // half, lane = e % half ; physical = h*padded + block*PWORD + lane.
    """
    half = head_dim // 2
    idx = np.empty(n_heads * head_dim, dtype=np.int64)
    for h in range(n_heads):
        for e in range(head_dim):
            idx[h * head_dim + e] = h * padded_head_dim + (e // half) * PWORD + (e % half)
    return idx


def _pad_axis(t, idx: np.ndarray, padded_len: int, axis: int, fill=0.0):
    """Scatter t's axis (length == idx.size, the real head channels) into a new
    axis of length padded_len at positions idx; other positions are fill."""
    import torch

    shape = list(t.shape)
    shape[axis] = padded_len
    out = torch.full(shape, fill, dtype=t.dtype, device=t.device)
    out.index_copy_(axis, torch.from_numpy(idx).to(t.device), t)
    return out


def decode_layer_buffers(bundle, spec, layer):
    """Fill the kernel's per-layer constant buffers for one layer from bundle.

    Mirrors the Qwen3 naive packer, but applies design-P block-interleaved
    padding to the head dimension: q/k/v_proj outputs and the SDPA per-head-dim
    scales are padded on their output/head axis; o_proj's input (head) axis and
    its activation quant are padded on the input axis. Pad channels carry zero
    weight, so they dequantize to zero and never reach the output.
    """
    # Deferred: packing a torch checkpoint onto kernel buffers is BUILD work, and the
    # run path resolves this module by name without torch installed.
    from ...layout import (
        broadcast_to_pword,
        fit_lut,
        layer_linears,
        layer_sdpa,
        layer_subset,
        pack,
        zeros,
    )

    lin = layer_linears(bundle.linear, layer)
    sdpa = layer_sdpa(bundle.sdpa, layer)
    norm = layer_subset(bundle.norm, layer)
    rng = layer_subset(bundle.lut_ranges, layer)

    num_q_heads = int(bundle.host_scalars["num_q_heads"])
    head_dim = int(bundle.host_scalars["head_dim"])
    num_kv_heads = int(bundle.config["num_key_value_heads"])
    d = _design_p_dims(head_dim)
    padded_head_dim = d["padded_head_dim"]
    q_idx = _interleave_indices(num_q_heads, head_dim, padded_head_dim)
    kv_idx = _interleave_indices(num_kv_heads, head_dim, padded_head_dim)
    hd_idx = _interleave_indices(1, head_dim, padded_head_dim)  # single head_dim vector
    q_plen = num_q_heads * padded_head_dim
    kv_plen = num_kv_heads * padded_head_dim

    buffers = {}

    def put(name, value):
        if name in spec:
            buffers[name] = pack(value, spec[name])

    for proj in _PROJECTIONS:
        p = lin[proj]
        if proj in _HEAD_OUT_PROJ:
            # Pad the OUTPUT (head) dim: weight rows, ds, dz. qs/qz are per-input.
            idx, plen = (q_idx, q_plen) if proj == "q_proj" else (kv_idx, kv_plen)
            put(f"{proj}_qs", p["qs"])
            put(f"{proj}_qz", p["qz"])
            put(f"{proj}_w", _pad_axis(p["w"], idx, plen, axis=0).transpose(0, 1))
            put(f"{proj}_ds", _pad_axis(p["ds"], idx, plen, axis=0))
            put(f"{proj}_dz", _pad_axis(p["dz"], idx, plen, axis=0))
        elif proj in _HEAD_IN_PROJ:
            # Pad the INPUT (head) dim: weight columns + activation quant qs/qz.
            put(f"{proj}_qs", _pad_axis(p["qs"], q_idx, q_plen, axis=0, fill=1.0))
            put(f"{proj}_qz", _pad_axis(p["qz"], q_idx, q_plen, axis=0))
            put(f"{proj}_w", _pad_axis(p["w"], q_idx, q_plen, axis=1).transpose(0, 1))
            put(f"{proj}_ds", p["ds"])  # per-output (hidden): unchanged
            put(f"{proj}_dz", p["dz"])
        else:  # gate/up/down: no head dim, standard packing
            put(f"{proj}_qs", p["qs"])
            put(f"{proj}_qz", p["qz"])
            put(f"{proj}_w", p["w"].transpose(0, 1))
            put(f"{proj}_ds", p["ds"])
            put(f"{proj}_dz", p["dz"])

    for prefix, field in _SDPA_QS.items():
        if f"{prefix}_qs" in spec:
            val = sdpa[field]
            # qs_q / qs_k are per-head-dim ([head_dim]) -> interleave-pad to the
            # 2-block layout (pad lanes -> 1.0; they multiply zero rope/V lanes, so
            # the value is irrelevant). qs_v is a per-tensor scalar -> pack
            # broadcasts it across all lanes (V's pad lanes are zero regardless).
            if val.numel() == head_dim:
                val = _pad_axis(val, hd_idx, padded_head_dim, axis=0, fill=1.0)
            buffers[f"{prefix}_qs"] = pack(val, spec[f"{prefix}_qs"])
        if f"{prefix}_qz" in spec:
            buffers[f"{prefix}_qz"] = zeros(spec[f"{prefix}_qz"])  # SDPA is zero-pinned
    for buf, field in _SDPA_SCALARS.items():
        put(buf, sdpa[field])
    for buf, site in _NORM_SITES.items():
        put(buf, broadcast_to_pword(norm[site]))
    for buf, site in _LUT_SITES.items():
        if buf in spec and site in rng:
            buffers[buf] = fit_lut(rng[site], spec[buf])
    return buffers


class LlamaMetisImplementation(UniformDecoderImplementation):
    """Standard Metis implementation of Llama 3.2: host embed -> device layer ->
    host lm_head, using design-P head-dim padding for head_dim < 128."""

    name = "llama-metis"
    architecture = "llama"

    def kernels(self):
        return [
            HostKernel("embedding", params=("embed_weight",)),
            LayerKernel(
                "decode_layer",
                build=make_layer_kernel,
                layer_buffers=decode_layer_buffers,
                phases=("prefill", "decode"),
                shared_inputs=self.shared_inputs,
                state=frozenset({"k_out", "v_out"}),
            ),
            HostKernel("lm_head", params=("final_norm_weight", "embed_weight", "rms_eps")),
        ]

    def _rope_tables(self, position, head_dim, rope_theta):
        """Design-P RoPE cos/sin for head_dim<128 with llama3 frequency rescaling.

        Produces (1, n_pos, rope_dim_pw, 64) with the head_dim/2 unique freqs in
        the low lanes of each block (pad lanes 0), matching the block-interleaved
        padded head layout the kernel consumes.
        """
        from .llama import llama3_inv_freq

        d = _design_p_dims(int(head_dim))
        half = d["half"]
        rope_dim_pw = d["rope_dim_pw"]
        inv_freq = llama3_inv_freq(int(head_dim), float(rope_theta), _LLAMA32_ROPE_SCALING).numpy()
        pos = np.atleast_1d(np.asarray(position, dtype=np.float64))
        angles = pos[:, None] * inv_freq[None, :]  # (n_pos, half)
        n_pos = pos.size
        cos = np.zeros((1, n_pos, rope_dim_pw, PWORD), dtype=np.float32)
        sin = np.zeros((1, n_pos, rope_dim_pw, PWORD), dtype=np.float32)
        flat_cos = np.cos(angles).astype(np.float32)
        flat_sin = np.sin(angles).astype(np.float32)
        for b in range(rope_dim_pw):
            lo, hi = b * PWORD, min((b + 1) * PWORD, half)
            cos[0, :, b, : hi - lo] = flat_cos[:, lo:hi]
            sin[0, :, b, : hi - lo] = flat_sin[:, lo:hi]
        return cos, sin


register_implementation(LlamaMetisImplementation())

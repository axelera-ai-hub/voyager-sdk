# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Qwen3 Metis implementation: the alto-script decoder-layer kernel + param binding.

Kept separate from the PyTorch reference: TVM Script can't run under from
__future__ import annotations, and its A.Buffer(..., "float32x64", ...)
dtype-string annotations trip pyflakes F821 (hence the module-wide noqa below).
"""

# flake8: noqa: F821  -- A.Buffer dtype-string annotations look like F821 names

from typing import TYPE_CHECKING

from ...implementation import (
    HostKernel,
    LayerKernel,
    UniformDecoderImplementation,
    register_implementation,
)

if TYPE_CHECKING:
    from .qwen3 import Qwen3Config


def make_layer_kernel(config: "Qwen3Config", ctx_len: int, phase: str):
    """Build the QwenLayerFunctional IRModule for one Qwen3 decoder layer.

    Dims come from config. ctx_len sizes the KV cache / mask; phase is
    "decode" (seq_len 1) or "prefill" (seq_len = ctx_len). Alto imports
    are deferred here so the float reference model can import without the compiler.
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol.common import BufferInfo
    from axelera.compiler.alto.axol.utils.lut_chain import ChainedLUTMethod
    import numpy as np

    hidden_dim = config.hidden_size
    num_q_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    head_dim = config.head_dim
    intermediate_dim = config.intermediate_size

    hidden_dim_pw = hidden_dim // 64
    head_dim_pw = head_dim // 64
    rope_dim_pw = head_dim_pw // 2
    q_out_dim = num_q_heads * head_dim
    q_out_dim_pw = q_out_dim // 64
    kv_out_dim = num_kv_heads * head_dim
    kv_out_dim_pw = kv_out_dim // 64
    intermediate_dim_pw = intermediate_dim // 64

    context_len = ctx_len
    seq_len = 1 if phase == "decode" else ctx_len

    @A.ir_module
    class QwenLayerFunctional:
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
            q_norm_w: A.Buffer((head_dim,), "float32x64", scope=A.DDR),
            q_norm_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            k_norm_w: A.Buffer((head_dim,), "float32x64", scope=A.DDR),
            k_norm_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
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
            # Constants
            eps_const = A.alloc_constant(
                [9.9999999747524271e-07] * 64, dtype="float32x64", scope=A.DDR
            )
            # SDPA zero-points + int8 range are zero-pinned constants (the dequant
            # scales themselves arrive as per-layer input buffers).
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

            # Q projection
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

            # RMS Norm for Q and K heads
            q_norm = axol.core.rms_norm_kahan(
                q_heads,
                q_norm_lut,
                eps_const[0:1],
                (head_dim,),
                weight_buffer=q_norm_w,
                invsqrt_method="additive",
            )
            k_norm = axol.core.rms_norm_kahan(
                k_heads,
                k_norm_lut,
                eps_const[0:1],
                (head_dim,),
                weight_buffer=k_norm_w,
                invsqrt_method="additive",
            )

            # RoPE
            trig_k = axol.core.concat(k_rope_sin, k_rope_cos, dim=2)
            k_rope = axol.core.rope(k_norm, trig_k)
            trig_q = axol.core.concat(q_rope_sin, q_rope_cos, dim=2)
            q_rope = axol.core.rope(q_norm, trig_q)

            # Quantize K for cache
            kq0 = axol.atomic.mul(k_rope, k_cache_qs)
            kq1 = axol.atomic.add(kq0, k_cache_qz)
            k_update = axol.atomic.dpu_cast(kq1, output=BufferInfo(dtype="int8x64"))

            # Quantize V for cache
            vq0 = axol.atomic.mul(v_heads, v_cache_qs)
            vq1 = axol.atomic.add(vq0, v_cache_qz)
            vq2 = axol.atomic.dpu_cast(vq1)
            v_update = axol.atomic.dpu_cast(vq2, output=BufferInfo(dtype="int8x64"))

            # Update KV cache
            k_out, v_out = axol.core.update_kv_cache(k_update, v_update, k_cache, v_cache, seq_pos)

            # Quantize Q for attention
            qq0 = axol.atomic.mul(q_rope, q_attn_qs)
            qq1 = axol.atomic.add(qq0, q_attn_qz)
            q_attn = axol.atomic.dpu_cast(qq1, output=BufferInfo(dtype="int8x64"))

            # SDPA
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
                scale=1.0 / (head_dim**0.5),
                neg_exp_method=ChainedLUTMethod.ADDITIVE,
                inv_method=ChainedLUTMethod.ADDITIVE,
            )

            # Transpose back and reshape to flat
            attn_t = axol.core.transpose(attn_out, dims=(0, 2, 1, 3))
            attn_c = axol.atomic.dpu_cast(attn_t, output=BufferInfo(dtype="float16x64"))
            attn_flat = A.alloc_buffer(
                (1, seq_len, q_out_dim_pw), "float16x64", data=attn_c.data, scope=A.DDR
            )

            # Output projection
            o0 = axol.atomic.mul(attn_flat, o_proj_qs)
            o1 = axol.atomic.add(o0, o_proj_qz)
            o2 = axol.atomic.dpu_cast(o1, output=BufferInfo(dtype="float32x64"))
            o3 = axol.atomic.dpu_cast(o2, output=BufferInfo(dtype="int8x64"))
            o4 = axol.core.matmul(o3, o_proj_w)
            o5 = axol.atomic.sub(o4, o_proj_dz)
            o6 = axol.atomic.mul(o5, o_proj_ds)

            # Residual connection
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

            # MLP gate projection + SiLU
            g0 = axol.atomic.mul(rms2, gate_proj_qs)
            g1 = axol.atomic.add(g0, gate_proj_qz)
            g2 = axol.atomic.dpu_cast(g1, output=BufferInfo(dtype="int8x64"))
            g3 = axol.core.matmul(g2, gate_proj_w)
            g4 = axol.atomic.sub(g3, gate_proj_dz)
            g5 = axol.atomic.mul(g4, gate_proj_ds)
            gate = axol.core.piecewise_linear_approximation(g5, silu_lut, "additive")

            # MLP up projection
            u0 = axol.atomic.mul(rms2, up_proj_qs)
            u1 = axol.atomic.add(u0, up_proj_qz)
            u2 = axol.atomic.dpu_cast(u1, output=BufferInfo(dtype="int8x64"))
            u3 = axol.core.matmul(u2, up_proj_w)
            u4 = axol.atomic.sub(u3, up_proj_dz)
            up = axol.atomic.mul(u4, up_proj_ds)

            # Gate * Up + down projection
            d0 = axol.atomic.mul(gate, up)
            d1 = axol.atomic.mul(d0, down_proj_qs)
            d2 = axol.atomic.add(d1, down_proj_qz)
            d3 = axol.atomic.dpu_cast(d2, output=BufferInfo(dtype="int8x64"))
            d4 = axol.core.matmul(d3, down_proj_w)
            d5 = axol.atomic.sub(d4, down_proj_dz)
            d6 = axol.atomic.mul(d5, down_proj_ds)

            # Final residual connection
            A.output(axol.atomic.add(residual, d6))

    return QwenLayerFunctional


# Projection names match both the kernel buffer prefixes (q_proj_*) and the
# torch submodule leaf names (self_attn.q_proj).
_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")
# SDPA per-head-dim quant scales: kernel buffer prefix -> exported SDPA field.
_SDPA_QS = {"q_attn": "qs_q", "k_cache": "qs_k", "v_cache": "qs_v"}
# SDPA per-tensor scales the kernel takes as broadcast single-PWORD inputs.
_SDPA_SCALARS = {
    "sdpa_scores_dq_scale": "scores_dq_scale",  # q_scale * k_scale
    "sdpa_weights_q_scale": "qs_w",  # softmax-weight quant scale
    "sdpa_output_dq_scale": "output_dq_scale",  # w_scale * v_scale
}
# RMSNorm-weight buffer -> the RMSNorm site (relative module path within a layer).
_NORM_SITES = {
    "rms1_w": "input_layernorm",
    "rms2_w": "post_attention_layernorm",
    "q_norm_w": "self_attn.q_norm",
    "k_norm_w": "self_attn.k_norm",
}
# LUT buffer -> the RangeObserver site whose recorded range the table is fit over.
_LUT_SITES = {
    "rms1_lut": "input_layernorm.inv_sqrt_range",
    "rms2_lut": "post_attention_layernorm.inv_sqrt_range",
    "q_norm_lut": "self_attn.q_norm.inv_sqrt_range",
    "k_norm_lut": "self_attn.k_norm.inv_sqrt_range",
    "sdpa_exp_lut": "self_attn.sdpa.exp_range",
    "sdpa_inv_lut": "self_attn.sdpa.reciprocal_range",
    "silu_lut": "mlp.silu_range",
}


def decode_layer_buffers(bundle, spec, layer):
    """Fill the kernel's per-layer constant buffers for one layer from bundle.

    Each exported tensor is mutated as the kernel wants (transpose weights, broadcast
    norm scales, zero-pin SDPA zero points, fit LUTs) and packed to the buffer's
    x64 layout. put skips names the spec doesn't ask for, leaving the non-
    constant buffers (KV cache, scratch, host inputs) alone.
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

    buffers = {}

    def put(name, value):
        if name in spec:
            buffers[name] = pack(value, spec[name])

    for proj in _PROJECTIONS:
        p = lin[proj]
        put(f"{proj}_qs", p["qs"])
        put(f"{proj}_qz", p["qz"])
        put(f"{proj}_w", p["w"].transpose(0, 1))  # [out, in] -> the matmul's [in, out]
        put(f"{proj}_ds", p["ds"])
        put(f"{proj}_dz", p["dz"])
    for prefix, field in _SDPA_QS.items():
        put(f"{prefix}_qs", sdpa[field])
        if f"{prefix}_qz" in spec:
            buffers[f"{prefix}_qz"] = zeros(spec[f"{prefix}_qz"])  # SDPA is zero-pinned
    for buf, field in _SDPA_SCALARS.items():
        put(buf, sdpa[field])
    for buf, site in _NORM_SITES.items():
        put(buf, broadcast_to_pword(norm[site]))
    for buf, site in _LUT_SITES.items():
        if buf in spec and site in rng:  # un-quantized attention omits exp/recip ranges
            buffers[buf] = fit_lut(rng[site], spec[buf])
    return buffers


class Qwen3MetisImplementation(UniformDecoderImplementation):
    """The standard Metis implementation of Qwen3: host embed -> device layer -> host lm_head."""

    name = "qwen3-metis"
    architecture = "qwen3"

    def kernels(self):
        return [
            HostKernel("embedding", params=("embed_weight",)),
            LayerKernel(
                "decode_layer",
                build=make_layer_kernel,
                layer_buffers=decode_layer_buffers,
                # The same layer kernel compiles for both phases (prefill = full
                # sequence, decode = one token); they share the weights + KV cache.
                phases=("prefill", "decode"),
                shared_inputs=self.shared_inputs,
                state=frozenset({"k_out", "v_out"}),
            ),
            HostKernel(
                "lm_head",
                params=("final_norm_weight", "embed_weight", "rms_eps"),
            ),
        ]


register_implementation(Qwen3MetisImplementation())

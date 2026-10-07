# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Optimized Llama 3.2 Metis implementation (registered as llama-metis-opt).

Uses NK llm.matmul (b_transposed) for projections, quantized_matmul_activation
fusing output dequant, and the fused gate||up SwiGLU op. Uses design-P
block-interleaved head padding for head_dim=64 (no QK-Norm).

Projection weights are declared NK (N_BLOCKS, K) and bound transposed; the
design-P head padding is inherited from the naive packer by binding through a
KN view of the spec, then swapping back to NK (see optimized_decode_layer_buffers).
"""

# flake8: noqa: F821  -- A.Buffer dtype-string annotations look like F821 names

from typing import TYPE_CHECKING

import numpy as np

from ...implementation import HostKernel, LayerKernel, register_implementation
from .naive_impl import (
    _PROJECTIONS,
    LlamaMetisImplementation,
    _design_p_dims,
    _interleave_indices,
    _pad_axis,
    decode_layer_buffers,
)

if TYPE_CHECKING:
    from .llama import LlamaConfig


def make_optimized_layer_kernel(config: "LlamaConfig", ctx_len: int, phase: str):
    """Build the optimized LlamaLayerFunctional IRModule for one decoder layer."""
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol.common import BufferInfo
    from axelera.compiler.alto.axol.utils.lut_chain import ChainedLUTMethod

    from ...ops.fused_gate_up_swiglu import fused_gate_up_swiglu
    from ...ops.quantized_matmul_activation import quantized_matmul_activation

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

    # Both prefill and decode use the core SDPA op (same operands/result).
    sdpa = axol.core.scaled_dot_product_attention

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
            q_proj_w: A.Buffer((q_out_dim_pw, hidden_dim), "int8x64", scope=A.DDR),
            q_proj_dq: A.Buffer((1, 1, 2 * q_out_dim_pw), "float32x64", scope=A.DDR),
            k_proj_w: A.Buffer((kv_out_dim_pw, hidden_dim), "int8x64", scope=A.DDR),
            k_proj_dq: A.Buffer((1, 1, 2 * kv_out_dim_pw), "float32x64", scope=A.DDR),
            v_proj_w: A.Buffer((kv_out_dim_pw, hidden_dim), "int8x64", scope=A.DDR),
            v_proj_dq: A.Buffer((1, 1, 2 * kv_out_dim_pw), "float32x64", scope=A.DDR),
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
            o_proj_w: A.Buffer((hidden_dim_pw, q_out_dim), "int8x64", scope=A.DDR),
            o_proj_dq: A.Buffer((1, 1, 2 * hidden_dim_pw), "float32x64", scope=A.DDR),
            residual_in: A.Buffer((1, seq_len, hidden_dim_pw), "float32x64", scope=A.DDR),
            rms2_w: A.Buffer((hidden_dim,), "float32x64", scope=A.DDR),
            rms2_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            gate_proj_qs: A.Buffer((1, 1, hidden_dim_pw), "float32x64", scope=A.DDR),
            gate_proj_qz: A.Buffer((1, 1, hidden_dim_pw), "float32x64", scope=A.DDR),
            silu_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            # Composed gate||up weight (NK, block-interleaved) for the fused SwiGLU op.
            gate_up_proj_w: A.Buffer(
                (2 * intermediate_dim_pw, hidden_dim), "int8x64", scope=A.DDR
            ),
            # [gate_dz, gate_ds, up_dz, up_ds, down_qs, down_qz] per intermediate channel.
            mlp_dq_params: A.Buffer((1, 1, 6 * intermediate_dim_pw), "float32x64", scope=A.DDR),
            down_proj_w: A.Buffer((hidden_dim_pw, intermediate_dim), "int8x64", scope=A.DDR),
            # [down_dz, down_ds] per hidden channel (down output dequant).
            down_dq_params: A.Buffer((1, 1, 2 * hidden_dim_pw), "float32x64", scope=A.DDR),
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

            # Q/K/V share one SmoothQuant group: identical input qs/qz, so quantize
            # the RMSNorm output to int8 once and reuse it for all three.
            qkv0 = axol.atomic.mul(rms1, q_proj_qs)
            qkv1 = axol.atomic.add(qkv0, q_proj_qz)
            qkv_q = axol.atomic.dpu_cast(qkv1, output=BufferInfo(dtype="int8x64"))

            # Q projection (NK matmul + fused dequant; output head-padded)
            q5 = quantized_matmul_activation(
                qkv_q, q_proj_w, q_proj_dq, output=BufferInfo(dtype="float16x64")
            )
            q5_4d = axol.atomic.reshape(q5, shape=(1, seq_len, num_q_heads, head_dim_pw))
            q_heads = axol.core.transpose(q5_4d, dims=(0, 2, 1, 3))

            # K projection (reuses the shared Q/K/V int8 activation)
            k5 = quantized_matmul_activation(
                qkv_q, k_proj_w, k_proj_dq, output=BufferInfo(dtype="float16x64")
            )
            k5_4d = axol.atomic.reshape(k5, shape=(1, seq_len, num_kv_heads, head_dim_pw))
            k_heads = axol.core.transpose(k5_4d, dims=(0, 2, 1, 3))

            # V projection (reuses the shared Q/K/V int8 activation)
            v5 = quantized_matmul_activation(
                qkv_q, v_proj_w, v_proj_dq, output=BufferInfo(dtype="float16x64")
            )
            v5_4d = axol.atomic.reshape(v5, shape=(1, seq_len, num_kv_heads, head_dim_pw))
            v_heads = axol.core.transpose(v5_4d, dims=(0, 2, 1, 3))

            # RoPE (no QK-Norm for Llama)
            trig_k = axol.core.concat(k_rope_sin, k_rope_cos, dim=2)
            k_rope = axol.core.rope(k_heads, trig_k)
            trig_q = axol.core.concat(q_rope_sin, q_rope_cos, dim=2)
            q_rope = axol.core.rope(q_heads, trig_q)

            # Quantize K for cache
            kq0 = axol.atomic.mul(k_rope, k_cache_qs)
            kq1 = axol.atomic.add(kq0, k_cache_qz)
            k_update = axol.atomic.dpu_cast(kq1, output=BufferInfo(dtype="int8x64"))

            # Quantize V for cache
            vq0 = axol.atomic.mul(v_heads, v_cache_qs)
            vq1 = axol.atomic.add(vq0, v_cache_qz)
            v_update = axol.atomic.dpu_cast(vq1, output=BufferInfo(dtype="int8x64"))

            k_out, v_out = axol.core.update_kv_cache(k_update, v_update, k_cache, v_cache, seq_pos)

            # Quantize Q for attention
            qq0 = axol.atomic.mul(q_rope, q_attn_qs)
            qq1 = axol.atomic.add(qq0, q_attn_qz)
            q_attn = axol.atomic.dpu_cast(qq1, output=BufferInfo(dtype="int8x64"))

            # SDPA — scale uses the LOGICAL head_dim.
            attn_out = sdpa(
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
            attn_flat = axol.atomic.reshape(attn_c, shape=(1, seq_len, q_out_dim_pw))

            # Output projection (NK; input head-padded, pad columns zeroed in packing)
            o0 = axol.atomic.mul(attn_flat, o_proj_qs)
            o1 = axol.atomic.add(o0, o_proj_qz)
            o3 = axol.atomic.dpu_cast(o1, output=BufferInfo(dtype="int8x64"))
            o6 = quantized_matmul_activation(
                o3, o_proj_w, o_proj_dq, output=BufferInfo(dtype="float32x64")
            )

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

            # gate/up share one SmoothQuant group: quantize rms2 to int8 once.
            gu0 = axol.atomic.mul(rms2, gate_proj_qs)
            gu1 = axol.atomic.add(gu0, gate_proj_qz)
            gu_q = axol.atomic.dpu_cast(gu1, output=BufferInfo(dtype="int8x64"))

            # Fused gate||up matmul + SwiGLU + down-input requantize.
            mlp_split = fused_gate_up_swiglu(gu_q, gate_up_proj_w, mlp_dq_params, luts=silu_lut)

            # Down projection consumes the split-N int8 directly and fuses the down dequant.
            d6 = quantized_matmul_activation(
                mlp_split,
                down_proj_w,
                down_dq_params,
                matrix_a_is_split=True,
                output=BufferInfo(dtype="float32x64"),
            )

            A.output(axol.atomic.add(residual, d6))

    return LlamaLayerFunctional


def optimized_decode_layer_buffers(bundle, spec, layer):
    """Bind layer constants, repacking projection weights to NK layout.

    The optimized kernel declares weights NK (N_BLOCKS, K), whereas the naive
    packer produces KN (K, N_BLOCKS) (and applies the design-P head padding).
    Bind via the naive packer using a KN view of the spec, then swap the
    in/out-block axes of each packed weight back to NK (the PWORD lane is
    unchanged), inheriting the head padding for free.
    """
    weight_names = [f"{proj}_w" for proj in _PROJECTIONS]
    kn_spec = dict(spec)
    for name in weight_names:
        if name in kn_spec:
            (n_blocks, k, lane), dtype = kn_spec[name]  # NK: (out/64, in, 64)
            kn_spec[name] = ((k, n_blocks, lane), dtype)  # KN view: (in, out/64, 64)

    buffers = decode_layer_buffers(bundle, kn_spec, layer)

    for name in weight_names:
        if name in buffers:
            # KN-packed (in, out/64, 64) -> NK (out/64, in, 64); lane unchanged.
            buffers[name] = buffers[name].permute(1, 0, 2).contiguous()

    _add_fused_buffers(buffers, spec, bundle, layer)
    return buffers


def _add_fused_buffers(buffers, spec, bundle, layer) -> None:
    """Build the fused dequant tables the optimized ops consume.

    q/k/v/o use quantized_matmul_activation with an interleaved [dz, ds] table per
    output channel. q/k/v outputs are head-padded (design P), so their dz/ds are
    interleave-padded first; o_proj's output is the hidden dim (unpadded). The MLP
    fuses gate||up into one NK matmul + a combined dequant/requant table, and the
    down dequant into another table.
    """
    import torch

    from ...layout import layer_linears

    lin = layer_linears(bundle.linear, layer)
    num_q_heads = int(bundle.host_scalars["num_q_heads"])
    head_dim = int(bundle.host_scalars["head_dim"])
    num_kv_heads = int(bundle.config["num_key_value_heads"])
    pad = _design_p_dims(head_dim)["padded_head_dim"]
    q_idx = _interleave_indices(num_q_heads, head_dim, pad)
    kv_idx = _interleave_indices(num_kv_heads, head_dim, pad)
    q_plen, kv_plen = num_q_heads * pad, num_kv_heads * pad

    def _nk(w):  # [out, in] int8 -> NK (out//64, in, 64): [nb, k, lane] = w[nb*64+lane, k]
        out, inn = w.shape
        return w.reshape(out // 64, 64, inn).permute(0, 2, 1).contiguous().to(torch.int8)

    def _blocks(v):  # (C,) -> (C//64, 64) float32 (per-channel value packed in the lane)
        return torch.as_tensor(v).to(torch.float32).reshape(-1, 64).contiguous()

    # q/k/v/o interleaved [dz, ds] output-dequant tables. q/k/v outputs head-padded.
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        name = f"{proj}_dq"
        if name not in spec:
            continue
        p = lin[proj]
        dz, ds = p["dz"], p["ds"]
        if proj == "q_proj":
            dz, ds = _pad_axis(dz, q_idx, q_plen, 0), _pad_axis(ds, q_idx, q_plen, 0)
        elif proj in ("k_proj", "v_proj"):
            dz, ds = _pad_axis(dz, kv_idx, kv_plen, 0), _pad_axis(ds, kv_idx, kv_plen, 0)
        opw = dz.numel() // 64  # output blocks (padded for q/k/v; hidden for o)
        dq = torch.empty((2 * opw, 64), dtype=torch.float32)
        dq[0::2] = _blocks(dz)
        dq[1::2] = _blocks(ds)
        buffers[name] = dq.reshape(1, 1, 2 * opw, 64).contiguous()

    # Fused MLP (no head dim — identical to the qwen3 optimized packer).
    g, u, dn = lin["gate_proj"], lin["up_proj"], lin["down_proj"]
    ipw = g["w"].shape[0] // 64  # intermediate_dim_pw

    if "gate_up_proj_w" in spec:
        gate_nk, up_nk = _nk(g["w"]), _nk(u["w"])  # (ipw, hidden, 64) each
        composed = torch.empty((2 * ipw, gate_nk.shape[1], 64), dtype=torch.int8)
        composed[0::2] = gate_nk  # block-row 2j = gate row j
        composed[1::2] = up_nk  # block-row 2j+1 = up row j
        buffers["gate_up_proj_w"] = composed.contiguous()

    if "mlp_dq_params" in spec:
        # per intermediate channel: [gate_dz, gate_ds, up_dz, up_ds, down_qs, down_qz]
        mlp_dq = torch.empty((6 * ipw, 64), dtype=torch.float32)
        mlp_dq[0::6] = _blocks(g["dz"])
        mlp_dq[1::6] = _blocks(g["ds"])
        mlp_dq[2::6] = _blocks(u["dz"])
        mlp_dq[3::6] = _blocks(u["ds"])
        mlp_dq[4::6] = _blocks(dn["qs"])  # down input quant
        mlp_dq[5::6] = _blocks(dn["qz"])
        buffers["mlp_dq_params"] = mlp_dq.reshape(1, 1, 6 * ipw, 64).contiguous()

    if "down_dq_params" in spec:
        # per hidden channel: [down_dz, down_ds] (down output dequant)
        hpw = dn["w"].shape[0] // 64
        down_dq = torch.empty((2 * hpw, 64), dtype=torch.float32)
        down_dq[0::2] = _blocks(dn["dz"])
        down_dq[1::2] = _blocks(dn["ds"])
        buffers["down_dq_params"] = down_dq.reshape(1, 1, 2 * hpw, 64).contiguous()


class LlamaOptimizedImplementation(LlamaMetisImplementation):
    """Optimized Metis implementation of Llama 3.2 (NK matmul + fused MLP)."""

    name = "llama-metis-opt"

    def kernels(self):
        return [
            HostKernel("embedding", params=("embed_weight",)),
            LayerKernel(
                "decode_layer",
                build=make_optimized_layer_kernel,
                layer_buffers=optimized_decode_layer_buffers,
                phases=("prefill", "decode"),
                shared_inputs=self.shared_inputs,
                state=frozenset({"k_out", "v_out"}),
                # Promote workspace buffers to L2: stages intermediates off DDR.
                enable_buffer_promotion=True,
            ),
            HostKernel("lm_head", params=("final_norm_weight", "embed_weight", "rms_eps")),
        ]


register_implementation(LlamaOptimizedImplementation())

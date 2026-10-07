# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Optimized Qwen3 Metis implementation (registered as qwen3-metis-opt).

Uses the custom NK matmul (ops/matmul.py, llm.matmul with b_transposed=True):
projection weights are declared in NK layout (N_BLOCKS, K) and bound transposed
(see optimized_decode_layer_buffers).
"""

# flake8: noqa: F821  -- A.Buffer dtype-string annotations look like F821 names

from typing import TYPE_CHECKING

from ...implementation import (
    HostKernel,
    LayerKernel,
    UniformDecoderImplementation,
    register_implementation,
)
from .naive_impl import _PROJECTIONS, decode_layer_buffers

if TYPE_CHECKING:
    from .qwen3 import Qwen3Config


def make_optimized_layer_kernel(config: "Qwen3Config", ctx_len: int, phase: str):
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

    from ...ops.fused_gate_up_swiglu import fused_gate_up_swiglu
    from ...ops.quantized_matmul_activation import quantized_matmul_activation

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

    sdpa = axol.core.scaled_dot_product_attention

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
            q_proj_w: A.Buffer((q_out_dim_pw, hidden_dim), "int8x64", scope=A.DDR),
            q_proj_dq: A.Buffer((1, 1, 2 * q_out_dim_pw), "float32x64", scope=A.DDR),
            k_proj_w: A.Buffer((kv_out_dim_pw, hidden_dim), "int8x64", scope=A.DDR),
            k_proj_dq: A.Buffer((1, 1, 2 * kv_out_dim_pw), "float32x64", scope=A.DDR),
            v_proj_w: A.Buffer((kv_out_dim_pw, hidden_dim), "int8x64", scope=A.DDR),
            v_proj_dq: A.Buffer((1, 1, 2 * kv_out_dim_pw), "float32x64", scope=A.DDR),
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
            # SDPA zero-points + int8 range are zero-pinned constants; dequant
            # scales arrive as per-layer input buffers.
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

            rms1 = axol.core.rms_norm_kahan(
                hidden_states,
                rms1_lut,
                eps_const[0:1],
                (hidden_dim,),
                weight_buffer=rms1_w,
                invsqrt_method="additive",
            )

            # Q/K/V share one SmoothQuant group (identical input qs/qz), so quantize
            # the RMSNorm output to int8 once and reuse for all three.
            qkv0 = axol.atomic.mul(rms1, q_proj_qs)
            qkv1 = axol.atomic.add(qkv0, q_proj_qz)
            qkv_q = axol.atomic.dpu_cast(qkv1, output=BufferInfo(dtype="int8x64"))

            q5 = quantized_matmul_activation(
                qkv_q, q_proj_w, q_proj_dq, output=BufferInfo(dtype="float16x64")
            )
            q5_4d = axol.atomic.reshape(q5, shape=(1, seq_len, num_q_heads, head_dim_pw))
            q_heads = axol.core.transpose(q5_4d, dims=(0, 2, 1, 3))

            k5 = quantized_matmul_activation(
                qkv_q, k_proj_w, k_proj_dq, output=BufferInfo(dtype="float16x64")
            )
            k5_4d = axol.atomic.reshape(k5, shape=(1, seq_len, num_kv_heads, head_dim_pw))
            k_heads = axol.core.transpose(k5_4d, dims=(0, 2, 1, 3))

            v5 = quantized_matmul_activation(
                qkv_q, v_proj_w, v_proj_dq, output=BufferInfo(dtype="float16x64")
            )
            v5_4d = axol.atomic.reshape(v5, shape=(1, seq_len, num_kv_heads, head_dim_pw))
            v_heads = axol.core.transpose(v5_4d, dims=(0, 2, 1, 3))

            # Merge (heads, seq) into a single batch axis so the per-head norm
            # tiles as a few large register-batched tiles (q: 16->4, k: 8->2
            # launches) that fully use bank B, instead of one launch per head
            # each paying copy/transpose/datapath-launch overhead. The norm is
            # per-row over head_dim, so the merge is a pure view change.
            q_heads_flat = axol.atomic.reshape(
                q_heads, shape=(1, num_q_heads * seq_len, head_dim_pw)
            )
            q_norm_flat = axol.core.rms_norm_kahan(
                q_heads_flat,
                q_norm_lut,
                eps_const[0:1],
                (head_dim,),
                weight_buffer=q_norm_w,
                invsqrt_method="additive",
            )
            q_norm = axol.atomic.reshape(q_norm_flat, shape=(1, num_q_heads, seq_len, head_dim_pw))
            k_heads_flat = axol.atomic.reshape(
                k_heads, shape=(1, num_kv_heads * seq_len, head_dim_pw)
            )
            k_norm_flat = axol.core.rms_norm_kahan(
                k_heads_flat,
                k_norm_lut,
                eps_const[0:1],
                (head_dim,),
                weight_buffer=k_norm_w,
                invsqrt_method="additive",
            )
            k_norm = axol.atomic.reshape(
                k_norm_flat, shape=(1, num_kv_heads, seq_len, head_dim_pw)
            )

            trig_k = axol.core.concat(k_rope_sin, k_rope_cos, dim=2)
            k_rope = axol.core.rope(k_norm, trig_k)
            trig_q = axol.core.concat(q_rope_sin, q_rope_cos, dim=2)
            q_rope = axol.core.rope(q_norm, trig_q)

            kq0 = axol.atomic.mul(k_rope, k_cache_qs)
            kq1 = axol.atomic.add(kq0, k_cache_qz)
            k_update = axol.atomic.dpu_cast(kq1, output=BufferInfo(dtype="int8x64"))

            vq0 = axol.atomic.mul(v_heads, v_cache_qs)
            vq1 = axol.atomic.add(vq0, v_cache_qz)
            v_update = axol.atomic.dpu_cast(vq1, output=BufferInfo(dtype="int8x64"))

            k_out, v_out = axol.core.update_kv_cache(k_update, v_update, k_cache, v_cache, seq_pos)

            qq0 = axol.atomic.mul(q_rope, q_attn_qs)
            qq1 = axol.atomic.add(qq0, q_attn_qz)
            q_attn = axol.atomic.dpu_cast(qq1, output=BufferInfo(dtype="int8x64"))

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
                scale=1.0 / (head_dim**0.5),
                neg_exp_method=ChainedLUTMethod.ADDITIVE,
                inv_method=ChainedLUTMethod.ADDITIVE,
            )

            attn_t = axol.core.transpose(attn_out, dims=(0, 2, 1, 3))
            attn_c = axol.atomic.dpu_cast(attn_t, output=BufferInfo(dtype="float16x64"))
            attn_flat = axol.atomic.reshape(attn_c, shape=(1, seq_len, q_out_dim_pw))

            o0 = axol.atomic.mul(attn_flat, o_proj_qs)
            o1 = axol.atomic.add(o0, o_proj_qz)
            o3 = axol.atomic.dpu_cast(o1, output=BufferInfo(dtype="int8x64"))
            o6 = quantized_matmul_activation(
                o3, o_proj_w, o_proj_dq, output=BufferInfo(dtype="float32x64")
            )

            residual = axol.atomic.add(residual_in, o6)

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

            # Fused gate||up matmul + SwiGLU: one composed NK matmul, the whole
            # SwiGLU chain in DPU registers, int8 out in split-N layout.
            mlp_split = fused_gate_up_swiglu(gu_q, gate_up_proj_w, mlp_dq_params, luts=silu_lut)

            # Down projection consumes split-N int8 directly (matrix_a_is_split),
            # fusing the down dequant. Emits float32 for the residual add.
            d6 = quantized_matmul_activation(
                mlp_split,
                down_proj_w,
                down_dq_params,
                matrix_a_is_split=True,
                output=BufferInfo(dtype="float32x64"),
            )

            A.output(axol.atomic.add(residual, d6))

    return QwenLayerFunctional


def optimized_decode_layer_buffers(bundle, spec, layer):
    """Bind layer constants, repacking projection weights to NK layout.

    The optimized kernel declares weights as (N_BLOCKS, K) (out-block major)
    for the b_transposed matmul. Bind via the naive packer using a KN view of the spec
    (swap the weight's first two dims), then swap the in / out-block axes of each
    packed weight back to NK. The PWORD lane (last axis) is unchanged, and all
    non-weight buffers are bound exactly as the naive packer does.
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

    # gate/up weights are composed into one NK matmul; gate/up output-dequant +
    # down input-quant are packed into one table. gate/up share their input quant
    # (gate_proj_qs/qz).
    _add_fused_mlp_buffers(buffers, spec, bundle, layer)
    return buffers


def _add_fused_mlp_buffers(buffers, spec, bundle, layer) -> None:
    """Build the composed gate||up weight + combined dequant tables that
    fused_gate_up_swiglu / quantized_matmul_activation consume.

    Only emits buffers the spec asks for.
    """
    import torch

    from ...layout import layer_linears

    def _nk(w):  # [out, in] int8 -> NK (out//64, in, 64): [nb, k, lane] = w[nb*64+lane, k]
        out, inn = w.shape
        return w.reshape(out // 64, 64, inn).permute(0, 2, 1).contiguous().to(torch.int8)

    def _blocks(v):  # (C,) -> (C//64, 64) float32 (per-channel value packed in the lane)
        return torch.as_tensor(v).to(torch.float32).reshape(-1, 64).contiguous()

    lin = layer_linears(bundle.linear, layer)

    # q/k/v/o: fuse output dequant via interleaved [dz, ds] table per output channel.
    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        name = f"{proj}_dq"
        if name in spec:
            p = lin[proj]
            opw = p["w"].shape[0] // 64  # output blocks
            dq = torch.empty((2 * opw, 64), dtype=torch.float32)
            dq[0::2] = _blocks(p["dz"])
            dq[1::2] = _blocks(p["ds"])
            buffers[name] = dq.reshape(1, 1, 2 * opw, 64).contiguous()

    g, u, d = lin["gate_proj"], lin["up_proj"], lin["down_proj"]
    ipw = g["w"].shape[0] // 64  # intermediate_dim_pw

    if "gate_up_proj_w" in spec:
        gate_nk, up_nk = _nk(g["w"]), _nk(u["w"])
        composed = torch.empty((2 * ipw, gate_nk.shape[1], 64), dtype=torch.int8)
        composed[0::2] = gate_nk  # block-row 2j = gate row j
        composed[1::2] = up_nk  # block-row 2j+1 = up row j
        buffers["gate_up_proj_w"] = composed.contiguous()

    if "mlp_dq_params" in spec:
        # Per intermediate channel: [gate_dz, gate_ds, up_dz, up_ds, down_qs, down_qz].
        mlp_dq = torch.empty((6 * ipw, 64), dtype=torch.float32)
        mlp_dq[0::6] = _blocks(g["dz"])
        mlp_dq[1::6] = _blocks(g["ds"])
        mlp_dq[2::6] = _blocks(u["dz"])
        mlp_dq[3::6] = _blocks(u["ds"])
        mlp_dq[4::6] = _blocks(d["qs"])  # down input quant
        mlp_dq[5::6] = _blocks(d["qz"])
        buffers["mlp_dq_params"] = mlp_dq.reshape(1, 1, 6 * ipw, 64).contiguous()

    if "down_dq_params" in spec:
        # Per hidden channel: [down_dz, down_ds] (down output dequant).
        hpw = d["w"].shape[0] // 64
        down_dq = torch.empty((2 * hpw, 64), dtype=torch.float32)
        down_dq[0::2] = _blocks(d["dz"])
        down_dq[1::2] = _blocks(d["ds"])
        buffers["down_dq_params"] = down_dq.reshape(1, 1, 2 * hpw, 64).contiguous()


class Qwen3OptimizedImplementation(UniformDecoderImplementation):
    """Optimized Metis implementation of Qwen3: host embed -> device layer -> host lm_head."""

    name = "qwen3-metis-opt"
    architecture = "qwen3"

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
                # Buffer promotion stages workspace off DDR into L2 (~11% prefill speedup).
                enable_buffer_promotion=True,
            ),
            HostKernel(
                "lm_head",
                params=("final_norm_weight", "embed_weight", "rms_eps"),
            ),
        ]


register_implementation(Qwen3OptimizedImplementation())

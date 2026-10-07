# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Optimized LFM2 Metis implementation (registered as lfm2-metis-opt).

LFM2 is a hybrid stack, so unlike the uniform Llama/Qwen3 implementations this
realises TWO decoder-layer kinds (see implementation.DecoderLayerImplementation):

* full_attention -- structurally the Qwen3 optimized attention layer: QK-Norm,
  RoPE, GQA, int8 SDPA + KV cache, then a SwiGLU MLP. (LFM2 attention is Qwen3
  attention minus the output gate; the projections are named out_proj and the
  QK-Norms q_layernorm / k_layernorm to match the checkpoint.)
* conv -- the LFM2 short-conv token mixer: in_proj (W8A8) splits into
  B, C, x; B*x runs through a depthwise causal conv (the llm.causal_conv1d
  op, carrying a persistent conv_state window like a KV cache), gated again by
  C, then out_proj (W8A8); followed by the same SwiGLU MLP.

Both kinds reuse the optimized matmul stack (the NK quantized_matmul_activation
and the fused gate||up SwiGLU of ops/), so this requires quantize_conv=True so
the conv block's in_proj / out_proj are W8A8 (the depthwise conv itself
always stays full precision -- its FIR taps ride in as a plain float constant).

Each kind has ONE builder, ``make_*_layer_kernel_multicore``, taking a core count that
``DecoderLayerImplementation._compile_phase`` forwards from ``GenerationConfig.num_cores``
(it shards a kind iff the builder's signature accepts ``cores``). Both ``lfm2-metis-opt``
and lfm2vl bind it, so the ``lfm2*-4core`` entries of ``model_axscript/llm/regression/manifest.toml``
run the sharded program at their declared ``num_cores = 4``; their ``ppl`` / ``ppl_gap``
goldens describe that program and must be re-measured alongside any change of sharding.

The sharded builder shards three things. Every one is a rearrangement of the same
arithmetic, not a different computation:

* **SDPA head-shards** across all cores (attention layers);
* **q/k/v** -- and the conv mixer's B/C/x, which are likewise three independent matmuls
  over one shared int8 input -- run as a **parallel group**, one whole projection per
  core. That needs three cores; sharding a single projection instead is fixed-overhead
  bound and buys nothing;
* the **SwiGLU MLP** N-shards its gate||up matmul, and down_proj consumes the resulting
  split K directly.

The depthwise conv itself stays single-core -- its state is a carried window, not a
partitionable reduction -- and so do the RMSNorms and the two output projections. Each
shard is chosen at build time by a helper called during tracing, so ``cores == 1`` emits
exactly the single-core program and a single-core artifact is unaffected. (An M-shard of
out_proj/o_proj and its residual add is the obvious next one; it does not fit the
in-body ``debug`` branch, whose taps read the unfused projection output, and both arms
of that branch are traced.)
"""

# flake8: noqa: F821  -- A.Buffer dtype-string annotations look like F821 names

from typing import TYPE_CHECKING

import numpy as np

from ...implementation import (
    DecoderLayerImplementation,
    HostKernel,
    LayerKernel,
    make_state_scale,
    make_valid_len,
    register_implementation,
)

# Design-P head padding (head_dim=64 -> 2 PWORD blocks); reused from Llama.
from ..llama.naive_impl import _design_p_dims, _interleave_indices, _pad_axis
from ..multicore_layer import NormFold, ddr1_split_shards

if TYPE_CHECKING:
    from .lfm2 import LFM2Config

# LFM2 names its two hidden RMSNorms operator_norm / ffn_norm, where the shared backbone
# folds (multicore_layer.HIDDEN_NORM_FOLDS) name qwen3's input_layernorm /
# post_attention_layernorm, so the folds are declared here. hidden_norm_quant checks the
# buffer name it is handed against the fold, so a renamed half is a trace-time error.
ATTN_NORM_FOLD = NormFold("q_proj_qparams", "operator_norm")  # q/k/v share the quant
CONV_NORM_FOLD = NormFold("in_proj_qparams", "operator_norm")  # B/C/x share the quant
FFN_NORM_FOLD = NormFold("gate_proj_qparams", "ffn_norm")  # gate/up share the quant

# LUT rows + 1/D + eps in one buffer (see multicore_layer.norm_consts): the LFM2 norms fit
# the two-row chained inv-sqrt LUT the Kahan norm used, so every consts buffer is (2+2,).
_NORM_CONSTS_SHAPE = (4,)


def _hidden_norm_dims(hidden_dim: int, seq_len: int):
    """The dims view ``multicore_layer.hidden_norm_quant`` dispatches on.

    A hidden norm uses the single-pass rms_norm_dpu_reduce (the activation quantize fused
    in-op) when its D blocks fit the DPU register file; a wider hidden dim would overflow
    it and falls back to the transpose-reduce Kahan norm plus a streaming llm.quantize.
    Every shipping LFM2 size takes the fused path (hidden 1024 / 1536 / 2048 -> 16 / 24 /
    32 blocks, all well inside the limit).
    """
    from types import SimpleNamespace

    from ..multicore_layer import DPU_REDUCE_MAX_D_BLOCKS

    hidden_dim_pw = hidden_dim // 64
    return SimpleNamespace(
        hidden_dim=hidden_dim,
        hidden_dim_pw=hidden_dim_pw,
        seq_len=seq_len,
        use_dpu_hidden_norm=hidden_dim_pw <= DPU_REDUCE_MAX_D_BLOCKS,
    )


# Kernel builds


def _head_projections(qkv_q, weights, dequants, reshapes, group_cores):
    """(q, k, v) as head-major (1, heads, seq, head_dim_pw) buffers.

    Three independent matmuls over the same normed int8 input, so with a core each they
    run concurrently -- one whole projection per core, its head reshape and transpose
    riding along inside that core's own region. GQA makes the three outputs different
    widths, which the group op takes per projection.

    Called from a kernel body during tracing, so the untaken branch is never parsed as
    TVMScript and ``group_cores=None`` emits exactly the sequential program.
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto.axol.common import BufferInfo

    from ...ops.parallel_matmul_group import parallel_matmul_group
    from ...ops.quantized_matmul_activation import quantized_matmul_activation

    if group_cores:
        return parallel_matmul_group(
            qkv_q,
            weights,
            dequants,
            cores=group_cores,
            output_dtype="float16x64",
            reshape_shape=reshapes,
            transpose_dims=[(0, 2, 1, 3)] * len(weights),
        )
    heads = []
    for weight, dequant, shape in zip(weights, dequants, reshapes):
        proj = quantized_matmul_activation(
            qkv_q, weight, dequant, output=BufferInfo(dtype="float16x64")
        )
        heads.append(
            axol.core.transpose(axol.atomic.reshape(proj, shape=shape), dims=(0, 2, 1, 3))
        )
    return heads


def _stream_projections(activation, weights, dequants, group_cores):
    """The short-conv mixer's three hidden-wide streams B, C, x.

    in_proj emits them from one input, which is three independent matmuls sharing one
    int8 activation -- a matmul group when there are three cores to give it, the same
    three sequential matmuls otherwise. (Realised as three projections rather than one
    matmul plus a slice: slicing a quantized_matmul_activation output does not lower.)
    """
    from axelera.compiler.alto.axol.common import BufferInfo

    from ...ops.parallel_matmul_group import parallel_matmul_group
    from ...ops.quantized_matmul_activation import quantized_matmul_activation

    if group_cores:
        return parallel_matmul_group(
            activation, weights, dequants, cores=group_cores, output_dtype="float32x64"
        )
    return [
        quantized_matmul_activation(
            activation, weight, dequant, output=BufferInfo(dtype="float32x64")
        )
        for weight, dequant in zip(weights, dequants)
    ]


def make_attention_layer_kernel_multicore(
    config: "LFM2Config",
    ctx_len: int,
    phase: str,
    chunk_len: int | None = None,
    debug: bool = False,
    cores: int = 1,
):
    """The full_attention decoder layer: QK-Norm + RoPE + GQA SDPA + SwiGLU MLP.

    Mirrors the Qwen3 optimized attention layer (NK matmuls + fused MLP); LFM2 has
    no attention output gate and names its projection out_proj / its QK norms
    q_norm / k_norm here (bound from self_attn.q_layernorm etc).

    Chunked prefill (chunk_len set): the prefill graph processes chunk_len query
    positions while the KV cache and causal mask stay at full context_len; the host
    threads the cache by writing each chunk's K/V at seq_pos=base_pos.
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol.common import BufferInfo
    from axelera.compiler.alto.axol.utils.lut_chain import ChainedLUTMethod
    import numpy as np

    from ...ops.fused_gate_up_swiglu import fused_gate_up_swiglu
    from ...ops.quantized_matmul_activation import quantized_matmul_activation
    from ...ops.rms_norm_dpu_reduce import rms_norm_dpu_reduce
    from ..multicore_layer import hidden_norm_quant, quantize_activation

    hidden_dim = config.hidden_size
    num_q_heads = config.num_attention_heads
    num_kv_heads = config.num_key_value_heads
    logical_head_dim = config.head_dim
    intermediate_dim = config.mlp_intermediate_size

    # head_dim=64 (every LFM2 size) gets design-P block-interleaved padding so each
    # head occupies head_dim_pw>=2 PWORD blocks (rope_dim_pw>=1); for head_dim>=128
    # this is a no-op. The torch model / SDPA scale / RoPE freqs use logical_head_dim.
    d = _design_p_dims(logical_head_dim)
    head_dim_pw = d["head_dim_pw"]
    rope_dim_pw = d["rope_dim_pw"]
    padded_head_dim = d["padded_head_dim"]
    # (QK-Norm over a design-P head must divide by the LOGICAL head_dim, not the padded
    # one. That correction lives entirely in the packed weight and consts, so nothing
    # here depends on it -- see _put_qk_norm.)
    hidden_dim_pw = hidden_dim // 64
    q_out_dim = num_q_heads * padded_head_dim
    q_out_dim_pw = q_out_dim // 64
    kv_out_dim = num_kv_heads * padded_head_dim
    kv_out_dim_pw = kv_out_dim // 64
    intermediate_dim_pw = intermediate_dim // 64

    context_len = ctx_len
    # Chunked prefill: chunk_len query positions against the full context_len KV cache.
    seq_len = 1 if phase == "decode" else (chunk_len or ctx_len)

    # Multicore: which shards apply is resolved here (build time), so the body holds one
    # unconditional call per op and cores == 1 emits exactly the single-core program.
    multicore = bool(cores and cores > 1)
    sdpa_cores_kw = {"cores": cores} if multicore else {}
    # One whole q/k/v projection per core, so the group needs three of them.
    qkv_group_cores = tuple(range(3)) if int(cores or 1) >= 3 else None
    norm_dims = _hidden_norm_dims(hidden_dim, seq_len)
    qkv_reshapes = [
        (1, seq_len, num_q_heads, head_dim_pw),
        (1, seq_len, num_kv_heads, head_dim_pw),
        (1, seq_len, num_kv_heads, head_dim_pw),
    ]

    @A.ir_module
    class Lfm2AttentionLayer:
        @A.prim_func
        def main(
            seq_pos: A.Buffer((1,), "uint64x64", scope=A.DDR),
            hidden_states: A.Buffer((1, seq_len, hidden_dim_pw), "float32x64", scope=A.DDR),
            # [inv-sqrt LUT rows, 1/D, eps] in one buffer: see pack_norm_consts. The norm
            # carries NO weight buffer -- the packer folds operator_norm's affine weight
            # into the quantize scale below (see multicore_layer.hidden_norm_quant).
            rms1_consts: A.Buffer(_NORM_CONSTS_SHAPE, "float32x64", scope=A.DDR),
            # Interleaved [scale, zp, ...] for the q/k/v input quantize, fused INTO the
            # norm (one datapath, int8 out) rather than run as a separate mul/add/cast.
            q_proj_qparams: A.Buffer((2 * hidden_dim_pw,), "float32x64", scope=A.DDR),
            q_proj_w: A.Buffer((q_out_dim_pw, hidden_dim), "int8x64", scope=A.DDR),
            q_proj_dq: A.Buffer((1, 1, 2 * q_out_dim_pw), "float32x64", scope=A.DDR),
            k_proj_w: A.Buffer((kv_out_dim_pw, hidden_dim), "int8x64", scope=A.DDR),
            k_proj_dq: A.Buffer((1, 1, 2 * kv_out_dim_pw), "float32x64", scope=A.DDR),
            v_proj_w: A.Buffer((kv_out_dim_pw, hidden_dim), "int8x64", scope=A.DDR),
            v_proj_dq: A.Buffer((1, 1, 2 * kv_out_dim_pw), "float32x64", scope=A.DDR),
            # D-major QK-Norm weights (head_dim in the PWORD lanes, one block per
            # input D block) for rms_norm_dpu_reduce -- NOT broadcast-PWORD.
            q_norm_w: A.Buffer((head_dim_pw,), "float32x64", scope=A.DDR),
            q_norm_consts: A.Buffer(_NORM_CONSTS_SHAPE, "float32x64", scope=A.DDR),
            k_norm_w: A.Buffer((head_dim_pw,), "float32x64", scope=A.DDR),
            k_norm_consts: A.Buffer(_NORM_CONSTS_SHAPE, "float32x64", scope=A.DDR),
            k_rope_sin: A.Buffer((1, seq_len, rope_dim_pw), "float32x64", scope=A.DDR),
            k_rope_cos: A.Buffer((1, seq_len, rope_dim_pw), "float32x64", scope=A.DDR),
            q_rope_sin: A.Buffer((1, seq_len, rope_dim_pw), "float32x64", scope=A.DDR),
            q_rope_cos: A.Buffer((1, seq_len, rope_dim_pw), "float32x64", scope=A.DDR),
            # Interleaved [scale, zp, ...] per-head-dim quant params for the fused
            # llm.quantize datapath (zp is zero-pinned for the SDPA quants).
            k_cache_qparams: A.Buffer((2 * head_dim_pw,), "float32x64", scope=A.DDR),
            v_qparams: A.Buffer((2 * head_dim_pw,), "float16x64", scope=A.DDR),
            k_cache: A.Buffer((1, num_kv_heads, context_len, head_dim_pw), "int8x64", scope=A.DDR),
            v_cache: A.Buffer((1, num_kv_heads, context_len, head_dim_pw), "int8x64", scope=A.DDR),
            q_attn_qparams: A.Buffer((2 * head_dim_pw,), "float32x64", scope=A.DDR),
            sdpa_exp_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            sdpa_inv_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            sdpa_scores_dq_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            sdpa_weights_q_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            sdpa_output_dq_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            # 1-head int8 ({0, -127}) causal mask, broadcast across q heads by the SDPA
            # op (as in llama/qwen3-metis-multicore) rather than a materialised
            # (num_q_heads, seq_len, context_len) fp32 mask. At chunk_len=1024 /
            # context_len=4096 that is 4 MiB instead of 512 MiB per chunk to build on
            # the host, upload, and re-stream from DDR in every attention layer -- the
            # dominant chunked-prefill cost at long context.
            attn_mask: A.Buffer((1, 1, seq_len, context_len // 64), "int8x64", scope=A.DDR),
            o_qparams: A.Buffer((2 * q_out_dim_pw,), "float16x64", scope=A.DDR),
            o_proj_w: A.Buffer((hidden_dim_pw, q_out_dim), "int8x64", scope=A.DDR),
            o_proj_dq: A.Buffer((1, 1, 2 * hidden_dim_pw), "float32x64", scope=A.DDR),
            residual_in: A.Buffer((1, seq_len, hidden_dim_pw), "float32x64", scope=A.DDR),
            rms2_consts: A.Buffer(_NORM_CONSTS_SHAPE, "float32x64", scope=A.DDR),
            # Interleaved [scale, zp, ...] for the gate/up input quantize (fused into
            # the ffn_norm; gate and up share one SmoothQuant group, and ffn_norm's
            # affine weight is folded into this scale).
            gate_proj_qparams: A.Buffer((2 * hidden_dim_pw,), "float32x64", scope=A.DDR),
            silu_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            gate_up_proj_w: A.Buffer(
                (2 * intermediate_dim_pw, hidden_dim), "int8x64", scope=A.DDR
            ),
            mlp_dq_params: A.Buffer((1, 1, 6 * intermediate_dim_pw), "float32x64", scope=A.DDR),
            down_proj_w: A.Buffer((hidden_dim_pw, intermediate_dim), "int8x64", scope=A.DDR),
            down_dq_params: A.Buffer((1, 1, 2 * hidden_dim_pw), "float32x64", scope=A.DDR),
        ):  # noqa: F821
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

            # Attention norm + the shared q/k/v input quantize in ONE single-pass
            # dpu_reduce datapath (int8 out): q/k/v share one SmoothQuant group, so the
            # norm is quantized once, and folding the quantize into the norm drops the
            # fp32 intermediate and halves the write.
            qkv_q = hidden_norm_quant(
                hidden_states, rms1_consts, q_proj_qparams, norm_dims, ATTN_NORM_FOLD
            )

            q_heads, k_heads, v_heads = _head_projections(
                qkv_q,
                [q_proj_w, k_proj_w, v_proj_w],
                [q_proj_dq, k_proj_dq, v_proj_dq],
                qkv_reshapes,
                qkv_group_cores,
            )

            # Per-head QK-Norm (flatten heads+seq into one batch axis to tile big).
            # Single-pass DPU add-reduce with the head_dim packed D-major into the lanes:
            # at head_dim_pw=2 every lane carries a real position, where the Kahan norm's
            # DNn packing idles most of them, and there is no NDd<->DNn transpose. The
            # /padded-head correction lives in the packed consts and weight, not here
            # (see _put_qk_norm).
            q_heads_flat = axol.atomic.reshape(
                q_heads, shape=(1, num_q_heads * seq_len, head_dim_pw)
            )
            q_norm_flat = rms_norm_dpu_reduce(
                q_heads_flat,
                q_norm_consts,
                normalized_shape=(padded_head_dim,),
                weight_buffer=q_norm_w,
                invsqrt_method="additive",
            )
            q_norm = axol.atomic.reshape(q_norm_flat, shape=(1, num_q_heads, seq_len, head_dim_pw))
            k_heads_flat = axol.atomic.reshape(
                k_heads, shape=(1, num_kv_heads * seq_len, head_dim_pw)
            )
            k_norm_flat = rms_norm_dpu_reduce(
                k_heads_flat,
                k_norm_consts,
                normalized_shape=(padded_head_dim,),
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

            # Fuse the int8 SDPA-input quant (scale + zero-point + cast) into one
            # llm.quantize datapath per K/V/Q. The element-wise mul/add/cast triple
            # this replaces materialised per-head (1, heads, seq, head_dim_pw) DDR
            # intermediates whose L1 staging split the innermost head_dim_pw axis ->
            # one transfer per sequence position (non-contiguous DMA at large ctx).
            # quantize_activation works on a (heads*seq, head_dim_pw) view, keeping
            # head_dim_pw contiguous. (Same fused op the multicore attention uses.)
            k_update = quantize_activation(
                k_rope, k_cache_qparams, (num_kv_heads, seq_len), head_dim_pw
            )
            v_update = quantize_activation(
                v_heads, v_qparams, (num_kv_heads, seq_len), head_dim_pw
            )
            k_out, v_out = axol.core.update_kv_cache(k_update, v_update, k_cache, v_cache, seq_pos)

            q_attn = quantize_activation(
                q_rope, q_attn_qparams, (num_q_heads, seq_len), head_dim_pw
            )

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
                **sdpa_cores_kw,
            )

            attn_t = axol.core.transpose(attn_out, dims=(0, 2, 1, 3))
            attn_c = axol.atomic.dpu_cast(attn_t, output=BufferInfo(dtype="float16x64"))
            attn_flat = axol.atomic.reshape(attn_c, shape=(1, seq_len, q_out_dim_pw))

            # o_proj input quant, same fused datapath (over a (seq, q_out_dim_pw) view).
            o3 = quantize_activation(attn_flat, o_qparams, (seq_len,), q_out_dim_pw)
            o6 = quantized_matmul_activation(
                o3, o_proj_w, o_proj_dq, output=BufferInfo(dtype="float32x64")
            )

            residual = axol.atomic.add(residual_in, o6)
            # debug: in-context fp18 taps for per-op device-vs-fp32 attribution
            # (layer_out first so the non-debug output slot [0] is stable).
            if debug:
                attn_layer_out, rms2, d6 = _swiglu_mlp_tail(
                    axol,
                    BufferInfo,
                    fused_gate_up_swiglu,
                    quantized_matmul_activation,
                    residual,
                    # eps_const / rms2_w / rms2_lut and gate_proj_qs / _qz are the Kahan
                    # arm's operands (qwen3.5); the fused arm below takes rms2_consts and
                    # gate_proj_qparams instead.
                    None,
                    None,
                    None,
                    hidden_dim,
                    None,
                    None,
                    gate_up_proj_w,
                    mlp_dq_params,
                    silu_lut,
                    down_proj_w,
                    down_dq_params,
                    debug=True,
                    cores=cores,
                    gate_proj_qparams=gate_proj_qparams,
                    rms2_consts=rms2_consts,
                    norm_dims=norm_dims,
                )
                A.output(attn_layer_out, qkv_q, o6, residual, rms2, d6)
            else:
                A.output(
                    _swiglu_mlp_tail(
                        axol,
                        BufferInfo,
                        fused_gate_up_swiglu,
                        quantized_matmul_activation,
                        residual,
                        # See the debug arm above: the Kahan operands go unused here.
                        None,
                        None,
                        None,
                        hidden_dim,
                        None,
                        None,
                        gate_up_proj_w,
                        mlp_dq_params,
                        silu_lut,
                        down_proj_w,
                        down_dq_params,
                        cores=cores,
                        gate_proj_qparams=gate_proj_qparams,
                        rms2_consts=rms2_consts,
                        norm_dims=norm_dims,
                    )
                )

    return Lfm2AttentionLayer


def make_conv_layer_kernel_multicore(
    config: "LFM2Config",
    ctx_len: int,
    phase: str,
    chunk_len: int | None = None,
    debug: bool = False,
    cores: int = 1,
):
    """The conv decoder layer: the LFM2 short-conv token mixer + SwiGLU MLP.

    in_proj (W8A8) -> split B/C/x -> B*x -> depthwise causal conv
    (llm.causal_conv1d, persistent conv_state window) -> C*conv ->
    out_proj (W8A8) -> residual -> SwiGLU MLP. No RoPE / mask / seq_pos: the
    conv is implicitly causal and its only state is the K-1 sample window.

    The conv window (conv_state) threads across calls like a KV cache; the host
    supplies a per-call state_scale gate (0.0 at a sequence start -- full prefill,
    chunked prefill chunk 0, or the first decode step -- else 1.0) so the accumulator
    resets between prompts/windows without recompiling or reloading, and continues
    within a sequence. See implementation.make_state_scale.
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol.common import BufferInfo

    from ...ops.causal_conv1d import causal_conv1d
    from ...ops.fused_gate_up_swiglu import fused_gate_up_swiglu
    from ...ops.quantized_matmul_activation import quantized_matmul_activation
    from ..multicore_layer import hidden_norm_quant

    hidden_dim = config.hidden_size
    intermediate_dim = config.mlp_intermediate_size
    k = config.conv_L_cache

    hidden_dim_pw = hidden_dim // 64
    intermediate_dim_pw = intermediate_dim // 64
    # Chunked prefill compiles the prefill graph for chunk_len timesteps; the conv
    # window state (conv_state) threads the K-1 carry across chunks.
    seq_len = 1 if phase == "decode" else (chunk_len or ctx_len)

    norm_dims = _hidden_norm_dims(hidden_dim, seq_len)

    # Multicore, resolved here so the body holds one unconditional call per op and
    # cores == 1 emits exactly the single-core program. The depthwise conv stays on one
    # core: its state is a carried K-1 window, not a partitionable reduction.
    bcx_group_cores = tuple(range(3)) if int(cores or 1) >= 3 else None

    @A.ir_module
    class Lfm2ConvLayer:
        @A.prim_func
        def main(
            hidden_states: A.Buffer((1, seq_len, hidden_dim_pw), "float32x64", scope=A.DDR),
            # [inv-sqrt LUT rows, 1/D, eps] in one buffer: see pack_norm_consts. No weight
            # buffer -- operator_norm's affine weight is folded into the quantize scale.
            rms1_consts: A.Buffer(_NORM_CONSTS_SHAPE, "float32x64", scope=A.DDR),
            # Interleaved [scale, zp, ...] for the B/C/x input quantize, fused INTO the
            # norm (one datapath, int8 out) rather than run as a separate mul/add/cast.
            in_proj_qparams: A.Buffer((2 * hidden_dim_pw,), "float32x64", scope=A.DDR),
            b_proj_w: A.Buffer((hidden_dim_pw, hidden_dim), "int8x64", scope=A.DDR),
            b_proj_dq: A.Buffer((1, 1, 2 * hidden_dim_pw), "float32x64", scope=A.DDR),
            c_proj_w: A.Buffer((hidden_dim_pw, hidden_dim), "int8x64", scope=A.DDR),
            c_proj_dq: A.Buffer((1, 1, 2 * hidden_dim_pw), "float32x64", scope=A.DDR),
            x_proj_w: A.Buffer((hidden_dim_pw, hidden_dim), "int8x64", scope=A.DDR),
            x_proj_dq: A.Buffer((1, 1, 2 * hidden_dim_pw), "float32x64", scope=A.DDR),
            # W8A8 depthwise conv on the DWPU: conv_qs quantizes bx to int8 (symmetric),
            # int8 taps + int8 window state, conv_ds is the per-channel dequant scale.
            conv_qs: A.Buffer((1, 1, hidden_dim_pw), "float32x64", scope=A.DDR),
            conv_weight: A.Buffer((hidden_dim_pw, k, 1, 1), "int8x64", scope=A.DDR),
            conv_state: A.Buffer((1, k - 1, hidden_dim_pw), "int8x64", scope=A.DDR),
            conv_ds: A.Buffer((1, 1, hidden_dim_pw), "float32x64", scope=A.DDR),
            state_scale: A.Buffer((1, 1, 1), "float32x64", scope=A.DDR),
            valid_len: A.Buffer((1,), "uint64x64", scope=A.DDR),
            out_proj_qs: A.Buffer((1, 1, hidden_dim_pw), "float32x64", scope=A.DDR),
            out_proj_qz: A.Buffer((1, 1, hidden_dim_pw), "float32x64", scope=A.DDR),
            out_proj_w: A.Buffer((hidden_dim_pw, hidden_dim), "int8x64", scope=A.DDR),
            out_proj_dq: A.Buffer((1, 1, 2 * hidden_dim_pw), "float32x64", scope=A.DDR),
            residual_in: A.Buffer((1, seq_len, hidden_dim_pw), "float32x64", scope=A.DDR),
            rms2_consts: A.Buffer(_NORM_CONSTS_SHAPE, "float32x64", scope=A.DDR),
            # Interleaved [scale, zp, ...] for the gate/up input quantize (fused into
            # the ffn_norm; gate and up share one SmoothQuant group, and ffn_norm's
            # affine weight is folded into this scale).
            gate_proj_qparams: A.Buffer((2 * hidden_dim_pw,), "float32x64", scope=A.DDR),
            silu_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            gate_up_proj_w: A.Buffer(
                (2 * intermediate_dim_pw, hidden_dim), "int8x64", scope=A.DDR
            ),
            mlp_dq_params: A.Buffer((1, 1, 6 * intermediate_dim_pw), "float32x64", scope=A.DDR),
            down_proj_w: A.Buffer((hidden_dim_pw, intermediate_dim), "int8x64", scope=A.DDR),
            down_dq_params: A.Buffer((1, 1, 2 * hidden_dim_pw), "float32x64", scope=A.DDR),
        ):  # noqa: F821
            # Operator norm + the shared B/C/x input quantize in ONE single-pass
            # dpu_reduce datapath (int8 out): the three streams share one input quant
            # (see _stream_projections for the layout and the core split), so the norm
            # is quantized once.
            iq = hidden_norm_quant(
                hidden_states, rms1_consts, in_proj_qparams, norm_dims, CONV_NORM_FOLD
            )
            b_gate, c_gate, x_in = _stream_projections(
                iq,
                [b_proj_w, c_proj_w, x_proj_w],
                [b_proj_dq, c_proj_dq, x_proj_dq],
                bcx_group_cores,
            )
            bx = axol.atomic.mul(b_gate, x_in)

            # Quantize bx to int8 for the DWPU conv. Per-channel symmetric activation
            # (conv_qs is per-channel, zero-point 0): a pure per-channel scale, no
            # zero-point add, and the int8 window/halo of zeros is exactly real-zero
            # (correct causal padding / fresh state). Per-channel is essential -- the
            # depthwise conv has no SmoothQuant, so a per-tensor bx scale zeros the
            # quiet channels and destroys accuracy.
            bxq = axol.atomic.mul(bx, conv_qs)
            bx_q = axol.atomic.dpu_cast(bxq, output=BufferInfo(dtype="int8x64"))

            # Depthwise causal conv on the DWPU (conv_state is the persistent int8
            # window, updated in place like a KV cache); the op returns the int32
            # accumulation cast to float32. state_scale is the host-supplied per-call
            # reset gate: 0.0 zeros the carried window at a sequence start (position 0 --
            # full prefill, chunk 0, or the first decode step), 1.0 continues it across
            # subsequent chunks/steps. Deriving it from position gives multi-prompt
            # correctness for free (each new prompt/window prefills from position 0)
            # without recompiling or reloading, and is the sequence-boundary hook a
            # future incremental chat sets explicitly.
            # valid_len is how many of this call's seq_len query rows are REAL tokens.
            # A right-padded prefill chunk would otherwise persist a window taken from the
            # pad, and the next decode step -- plus every KV entry it writes -- would be
            # computed from it. See ops/causal_conv1d.valid_len.
            conv_raw = causal_conv1d(
                bx_q, conv_weight, conv_state, state_scale=state_scale, valid_len=valid_len
            )
            # Per-channel dequant (symmetric -> pure scale ds = s_x*s_w), then gate by C.
            conv_out = axol.atomic.mul(conv_raw, conv_ds)
            y = axol.atomic.mul(c_gate, conv_out)

            # out_proj (W8A8): hidden -> hidden.
            o0 = axol.atomic.mul(y, out_proj_qs)
            o1 = axol.atomic.add(o0, out_proj_qz)
            oq = axol.atomic.dpu_cast(o1, output=BufferInfo(dtype="int8x64"))
            o6 = quantized_matmul_activation(
                oq, out_proj_w, out_proj_dq, output=BufferInfo(dtype="float32x64")
            )

            residual = axol.atomic.add(residual_in, o6)
            # debug: emit in-context fp18 intermediates for per-op device-vs-fp32
            # attribution (layer_out first, so the non-debug output slot [0] is stable).
            if debug:
                layer_out, rms2, d6 = _swiglu_mlp_tail(
                    axol,
                    BufferInfo,
                    fused_gate_up_swiglu,
                    quantized_matmul_activation,
                    residual,
                    # eps_const / rms2_w / rms2_lut and gate_proj_qs / _qz are the Kahan
                    # arm's operands (qwen3.5); the fused arm below takes rms2_consts and
                    # gate_proj_qparams instead.
                    None,
                    None,
                    None,
                    hidden_dim,
                    None,
                    None,
                    gate_up_proj_w,
                    mlp_dq_params,
                    silu_lut,
                    down_proj_w,
                    down_dq_params,
                    debug=True,
                    cores=cores,
                    gate_proj_qparams=gate_proj_qparams,
                    rms2_consts=rms2_consts,
                    norm_dims=norm_dims,
                )
                A.output(
                    layer_out, iq, b_gate, c_gate, x_in, bx, conv_out, y, o6, residual, rms2, d6
                )
            else:
                A.output(
                    _swiglu_mlp_tail(
                        axol,
                        BufferInfo,
                        fused_gate_up_swiglu,
                        quantized_matmul_activation,
                        residual,
                        # See the debug arm above: the Kahan operands go unused here.
                        None,
                        None,
                        None,
                        hidden_dim,
                        None,
                        None,
                        gate_up_proj_w,
                        mlp_dq_params,
                        silu_lut,
                        down_proj_w,
                        down_dq_params,
                        cores=cores,
                        gate_proj_qparams=gate_proj_qparams,
                        rms2_consts=rms2_consts,
                        norm_dims=norm_dims,
                    )
                )

    return Lfm2ConvLayer


def _swiglu_mlp_tail(
    axol,
    BufferInfo,
    fused_gate_up_swiglu,
    quantized_matmul_activation,
    residual,
    eps_const,
    rms2_w,
    rms2_lut,
    hidden_dim,
    gate_proj_qs,
    gate_proj_qz,
    gate_up_proj_w,
    mlp_dq_params,
    silu_lut,
    down_proj_w,
    down_dq_params,
    cores: int = 1,
    debug: bool = False,
    generation=None,
    gate_up_matrix_b1=None,
    down_matrix_b1=None,
    gate_proj_qparams=None,
    rms2_consts=None,
    norm_dims=None,
):
    """Shared post-mixer SwiGLU MLP tail (ffn_norm + fused gate||up + down + residual).

    Shared by both layer kinds: ffn_norm, one composed NK gate||up matmul + SwiGLU in
    DPU registers (int8 split-N out), down projection consuming the split int8, then
    the second residual add. Returns the layer output.

    ``cores`` > 1 N-shards both matmuls: gate||up distributes its output tiles across the
    cores and down_proj reads the resulting split K directly, so nothing is re-assembled
    in between. At ``cores == 1`` the calls are exactly the single-core ones.

    gate_up_matrix_b1/down_matrix_b1 stream the rest of a K-split weight from a second
    DDR region (see upper_shard_scope); None keeps the weight in one region, as before.

    ``norm_dims`` opts into the fused ffn_norm (LFM2): the norm and the gate/up input
    quantize collapse into one single-pass dpu_reduce datapath (int8 out), reading the
    packed ``rms2_consts`` (LUT rows, 1/D, eps) and the interleaved ``gate_proj_qparams``,
    which carries the zero-point AND ffn_norm's affine weight folded into its scale. That
    arm therefore uses none of ``eps_const``/``rms2_w``/``rms2_lut``/``gate_proj_qs``/
    ``gate_proj_qz``, and the ``rms2`` debug tap becomes the int8 output rather than the
    fp32 norm. Left None (qwen3.5) the tail keeps the Kahan norm + a separate
    mul/add/cast over those operands, unchanged.
    """
    mlp_cores_kw = {"cores": cores} if cores and cores > 1 else {}
    if mlp_cores_kw:
        from ...ops.quantized_matmul_activation import quantized_matmul_activation_multicore

        down_matmul = quantized_matmul_activation_multicore
    else:
        down_matmul = quantized_matmul_activation
    if norm_dims is not None:
        from ..multicore_layer import hidden_norm_quant

        gu_q = hidden_norm_quant(
            residual, rms2_consts, gate_proj_qparams, norm_dims, FFN_NORM_FOLD
        )
        rms2 = gu_q
    else:
        rms2 = axol.core.rms_norm_kahan(
            residual,
            rms2_lut,
            eps_const[0:1],
            (hidden_dim,),
            weight_buffer=rms2_w,
            invsqrt_method="additive",
        )
        gu0 = axol.atomic.mul(rms2, gate_proj_qs)
        gu1 = axol.atomic.add(gu0, gate_proj_qz)
        gu_q = axol.atomic.dpu_cast(gu1, output=BufferInfo(dtype="int8x64"))
    # generation is threaded so a EUROPA caller gets the EXP2/RECIP SiLU inside the
    # fused datapath; left unset it resolves to OMEGA's LUT chain as before.
    mlp_split = fused_gate_up_swiglu(
        gu_q,
        gate_up_proj_w,
        mlp_dq_params,
        matrix_b1=gate_up_matrix_b1,
        luts=silu_lut,
        generation=generation,
        **mlp_cores_kw,
    )
    d6 = down_matmul(
        mlp_split,
        down_proj_w,
        down_dq_params,
        **mlp_cores_kw,
        matrix_a_is_split=True,
        matrix_b1=down_matrix_b1,
        output=BufferInfo(dtype="float32x64"),
    )
    out = axol.atomic.add(residual, d6)
    # debug returns extra taps (rms2, d6) for per-op attribution; default keeps the
    # single-output contract (qwen3.5 deltanet also calls this helper).
    return (out, rms2, d6) if debug else out


# Buffer packing


def _nk(w):
    """[out, in] int8 -> NK (out//64, in, 64): block-row nb lane j = w[nb*64+j, k]."""
    import torch

    out, inn = w.shape
    return w.reshape(out // 64, 64, inn).permute(0, 2, 1).contiguous().to(torch.int8)


def _blocks(v):
    """(C,) -> (C//64, 64) float32 (per-channel value packed in the lane)."""
    import torch

    return torch.as_tensor(v).to(torch.float32).reshape(-1, 64).contiguous()


def _dq_table(p):
    """Interleaved [dz, ds] per output channel for quantized_matmul_activation.
    table[2j] = dz block j, table[2j+1] = ds block j. Shape (1, 1, 2*opw, 64)."""
    import torch

    opw = p["w"].shape[0] // 64
    dq = torch.empty((2 * opw, 64), dtype=torch.float32)
    dq[0::2] = _blocks(p["dz"])
    dq[1::2] = _blocks(p["ds"])
    return dq.reshape(1, 1, 2 * opw, 64).contiguous()


def _slice_out_rows(p, lo, hi):
    """A sub-projection over output rows [lo, hi) of a QuantLinear export dict
    (splitting a fused in_proj into its B/C/x parts). w/dz/ds are per-output."""
    return {"w": p["w"][lo:hi], "dz": p["dz"][lo:hi], "ds": p["ds"][lo:hi]}


def _put_qma_proj(buffers, spec, p, prefix):
    """Bind a W8A8 NK quantized_matmul_activation projection: qs, qz, NK w, dq.

    Splits the weight's K across DDR/DDR1 when the kernel declared a "<prefix>_w1"
    buffer (see ddr1_split_shards); a no-op otherwise.
    """
    # Deferred: packing a torch checkpoint onto kernel buffers is BUILD work, and the
    # run path resolves this module by name without torch installed.
    from ...layout import pack

    if f"{prefix}_qs" in spec:
        buffers[f"{prefix}_qs"] = pack(p["qs"], spec[f"{prefix}_qs"])
    if f"{prefix}_qz" in spec:
        buffers[f"{prefix}_qz"] = pack(p["qz"], spec[f"{prefix}_qz"])
    if f"{prefix}_w" in spec:
        buffers[f"{prefix}_w"] = _nk(p["w"])
        ddr1_split_shards(buffers, spec, (f"{prefix}_w",))
    if f"{prefix}_dq" in spec:
        buffers[f"{prefix}_dq"] = _dq_table(p)


def _put_fused_mlp(buffers, spec, lin):
    """Bind the composed gate||up NK weight + combined dequant tables for the fused
    SwiGLU + split down projection.

    Splits gate_up_proj_w / down_proj_w's K across DDR/DDR1 when the kernel declared
    the matching "<name>1" buffer (see ddr1_split_shards); a no-op otherwise.
    """
    # Deferred: packing a torch checkpoint onto kernel buffers is BUILD work, and the
    # run path resolves this module by name without torch installed.
    from ...layout import pack

    import torch

    g, u, d = lin["gate_proj"], lin["up_proj"], lin["down_proj"]
    ipw = g["w"].shape[0] // 64

    if "gate_proj_qs" in spec:  # gate/up share their (smoothed) input quant
        buffers["gate_proj_qs"] = pack(g["qs"], spec["gate_proj_qs"])
    if "gate_proj_qz" in spec:
        buffers["gate_proj_qz"] = pack(g["qz"], spec["gate_proj_qz"])

    if "gate_up_proj_w" in spec:
        gate_nk, up_nk = _nk(g["w"]), _nk(u["w"])
        composed = torch.empty((2 * ipw, gate_nk.shape[1], 64), dtype=torch.int8)
        composed[0::2] = gate_nk  # block-row 2j = gate row j
        composed[1::2] = up_nk  # block-row 2j+1 = up row j
        buffers["gate_up_proj_w"] = composed.contiguous()
        ddr1_split_shards(buffers, spec, ("gate_up_proj_w",))
    if "mlp_dq_params" in spec:
        mlp_dq = torch.empty((6 * ipw, 64), dtype=torch.float32)
        mlp_dq[0::6] = _blocks(g["dz"])
        mlp_dq[1::6] = _blocks(g["ds"])
        mlp_dq[2::6] = _blocks(u["dz"])
        mlp_dq[3::6] = _blocks(u["ds"])
        mlp_dq[4::6] = _blocks(d["qs"])  # down input quant
        mlp_dq[5::6] = _blocks(d["qz"])
        buffers["mlp_dq_params"] = mlp_dq.reshape(1, 1, 6 * ipw, 64).contiguous()
    if "down_proj_w" in spec:
        buffers["down_proj_w"] = _nk(d["w"])
        ddr1_split_shards(buffers, spec, ("down_proj_w",))
    if "down_dq_params" in spec:
        hpw = d["w"].shape[0] // 64
        down_dq = torch.empty((2 * hpw, 64), dtype=torch.float32)
        down_dq[0::2] = _blocks(d["dz"])
        down_dq[1::2] = _blocks(d["ds"])
        buffers["down_dq_params"] = down_dq.reshape(1, 1, 2 * hpw, 64).contiguous()


def _put_hidden_norm_consts(buffers, spec, bundle, layer, hidden_dim):
    """Bind rms1_consts / rms2_consts: each hidden norm's [LUT rows, 1/D, eps] buffer.

    The norms themselves carry no weight buffer -- ``_put_hidden_norm_folds`` folds each
    affine weight into the quantize scale that follows it.
    """
    from ..multicore_layer import add_norm_consts

    add_norm_consts(
        buffers,
        spec,
        bundle,
        layer,
        {
            "rms1_consts": ("operator_norm.inv_sqrt_range", hidden_dim),
            "rms2_consts": ("ffn_norm.inv_sqrt_range", hidden_dim),
        },
    )


def _put_hidden_norm_folds(buffers, spec, norm, folds):
    """Bind each fused norm's qparams: the shared input quant interleaved [scale, zp, ...]
    with the preceding RMSNorm's affine weight folded into the scale.

    ``folds`` maps a NormFold to the exported QuantLinear whose (per-input, shared) qs/qz
    it carries. check_norm_folds then asserts every declared fold produced a buffer, since
    a missing one would silently drop that norm's weight.
    """
    from ..multicore_layer import check_norm_folds, interleave_qparams

    for fold, p in folds.items():
        if fold.qparams in spec:
            buffers[fold.qparams] = interleave_qparams(
                p["qs"], p["qz"], spec[fold.qparams], norm_weight=norm[fold.site]
            )
    check_norm_folds(buffers, tuple(folds))


def _put_norms_and_luts(buffers, spec, norm, rng, norm_sites, lut_sites):
    """Bind the RMSNorm weights (broadcast to PWORD rows) and fit the LUT tables.

    LFM2's own norms pass no ``norm_sites``: its hidden norms fold their weight into the
    following quantize scale and its QK-Norms are D-major (see _put_qk_norm). The sites
    are for qwen3.5, which still binds broadcast Kahan weights through this helper.
    """
    # Deferred: packing a torch checkpoint onto kernel buffers is BUILD work, and the
    # run path resolves this module by name without torch installed.
    from ...layout import broadcast_to_pword, fit_lut, pack

    for buf, site in norm_sites.items():
        if buf in spec and site in norm:
            buffers[buf] = pack(broadcast_to_pword(norm[site]), spec[buf])
    for buf, site in lut_sites.items():
        if buf in spec and site in rng:
            buffers[buf] = fit_lut(rng[site], spec[buf])


# The one LUT both kinds fit through the shared helper; every norm LUT now rides in
# that norm's consts buffer instead.
_MLP_LUT_SITES = {"silu_lut": "feed_forward.silu_range"}
# SDPA per-head-dim quant scales: kernel qparams buffer -> exported SDPA field.
_SDPA_QPARAMS = {"q_attn_qparams": "qs_q", "k_cache_qparams": "qs_k", "v_qparams": "qs_v"}
_SDPA_SCALARS = {
    "sdpa_scores_dq_scale": "scores_dq_scale",
    "sdpa_weights_q_scale": "qs_w",
    "sdpa_output_dq_scale": "output_dq_scale",
}


def attention_layer_buffers(bundle, spec, layer):
    """Pack one full-attention layer's constants, with design-P head padding.

    head_dim=64 (every LFM2 size) is padded to head_dim_pw=2 PWORD blocks: q/k/v
    pad their OUTPUT (head) axis, o_proj pads its INPUT (head) axis, the SDPA
    per-head-dim scales pad on the head axis, and QK-Norm divides by the padded
    head -- corrected back to the logical head via weight (x sqrt(r)) and the
    inv-sqrt LUT input range (x r), where r = logical/padded (== 1 for head_dim>=128).
    """
    # Deferred: packing a torch checkpoint onto kernel buffers is BUILD work, and the
    # run path resolves this module by name without torch installed.
    from ...layout import fit_lut, layer_linears, layer_sdpa, layer_subset, pack

    from ..multicore_layer import interleave_qparams

    lin = layer_linears(bundle.linear, layer)
    sdpa = layer_sdpa(bundle.sdpa, layer)
    norm = layer_subset(bundle.norm, layer)
    rng = layer_subset(bundle.lut_ranges, layer)

    num_q_heads = int(bundle.host_scalars["num_q_heads"])
    head_dim = int(bundle.host_scalars["head_dim"])
    num_kv_heads = int(bundle.config["num_key_value_heads"])
    d = _design_p_dims(head_dim)
    padded = d["padded_head_dim"]
    r = head_dim / padded
    q_idx = _interleave_indices(num_q_heads, head_dim, padded)
    kv_idx = _interleave_indices(num_kv_heads, head_dim, padded)
    hd_idx = _interleave_indices(1, head_dim, padded)  # one head_dim vector
    q_plen, kv_plen = num_q_heads * padded, num_kv_heads * padded

    buffers: dict = {}

    # q/k/v: pad the OUTPUT (head) axis of w / dz / ds; qs/qz are per-input.
    for proj in ("q_proj", "k_proj", "v_proj"):
        p = lin[proj]
        idx, plen = (q_idx, q_plen) if proj == "q_proj" else (kv_idx, kv_plen)
        padded_p = {
            "w": _pad_axis(p["w"], idx, plen, axis=0),
            "dz": _pad_axis(p["dz"], idx, plen, axis=0),
            "ds": _pad_axis(p["ds"], idx, plen, axis=0),
        }
        if f"{proj}_qs" in spec:
            buffers[f"{proj}_qs"] = pack(p["qs"], spec[f"{proj}_qs"])
        if f"{proj}_qz" in spec:
            buffers[f"{proj}_qz"] = pack(p["qz"], spec[f"{proj}_qz"])
        if f"{proj}_w" in spec:
            buffers[f"{proj}_w"] = _nk(padded_p["w"])
        if f"{proj}_dq" in spec:
            buffers[f"{proj}_dq"] = _dq_table(padded_p)

    # o_proj (LFM2 out_proj): pad the INPUT (head) axis of w + the activation quant.
    o = lin["out_proj"]
    if "o_qparams" in spec:  # interleaved [scale, zp, ...] for the fused o-input quantize
        buffers["o_qparams"] = interleave_qparams(
            _pad_axis(o["qs"], q_idx, q_plen, axis=0, fill=1.0),
            _pad_axis(o["qz"], q_idx, q_plen, axis=0),
            spec["o_qparams"],
        )
    if "o_proj_w" in spec:
        buffers["o_proj_w"] = _nk(_pad_axis(o["w"], q_idx, q_plen, axis=1))
    if "o_proj_dq" in spec:  # dz/ds are per-output (hidden) -- unchanged
        buffers["o_proj_dq"] = _dq_table(o)

    _put_fused_mlp(buffers, spec, lin)

    # Hidden norms: consts only (no weight buffer). q/k/v share one SmoothQuant group and
    # so do gate/up, so each shared input quant is bound once, interleaved, with the norm
    # ahead of it folded into its scale. (The per-projection qs/qz above are per-input and
    # stay unbound for k/v, which are not in the kernel signature.)
    _put_hidden_norm_consts(buffers, spec, bundle, layer, int(bundle.config["hidden_size"]))
    _put_hidden_norm_folds(
        buffers, spec, norm, {ATTN_NORM_FOLD: lin["q_proj"], FFN_NORM_FOLD: lin["gate_proj"]}
    )
    # QK-Norm keeps its own weight buffer, corrected for the design-P /padded reduction.
    _put_qk_norm(
        buffers, spec, bundle, norm, rng, "q_norm", "self_attn.q_layernorm", hd_idx, padded, r
    )
    _put_qk_norm(
        buffers, spec, bundle, norm, rng, "k_norm", "self_attn.k_layernorm", hd_idx, padded, r
    )
    _put_norms_and_luts(buffers, spec, norm, rng, {}, _MLP_LUT_SITES)
    # SDPA softmax exp / reciprocal LUTs (present when quantize_attention).
    for buf, site in (
        ("sdpa_exp_lut", "self_attn.sdpa.exp_range"),
        ("sdpa_inv_lut", "self_attn.sdpa.reciprocal_range"),
    ):
        if buf in spec and site in rng:
            buffers[buf] = fit_lut(rng[site], spec[buf])

    # SDPA quant: interleaved [scale, zp=0, ...] qparams for the fused quantize. qs_q/qs_k
    # are per-head-dim -> design-P pad (pad lanes -> 1.0, they multiply the zero rope/V pad
    # lanes); qs_v is per-tensor (interleave_qparams broadcasts). zp is zero-pinned (None).
    for buf, fld in _SDPA_QPARAMS.items():
        if buf in spec:
            val = sdpa[fld]
            if int(val.numel()) == head_dim:
                val = _pad_axis(val, hd_idx, padded, axis=0, fill=1.0)
            buffers[buf] = interleave_qparams(val, None, spec[buf])
    for buf, fld in _SDPA_SCALARS.items():
        if buf in spec:
            buffers[buf] = pack(sdpa[fld], spec[buf])
    return buffers


def _put_qk_norm(buffers, spec, bundle, norm, rng, prefix, site, hd_idx, padded, r):
    """Bind a QK-Norm weight + [LUT rows, 1/D, eps] consts, corrected for the design-P
    /padded-head reduction: the op scales the sum of squares by the 1/D the consts carry,
    which is 1/padded here, so weight x sqrt(r), eps x r and LUT input range x r bring the
    result back to a /logical-head norm (r = logical/padded, == 1 for head_dim>=128).
    """
    # Deferred: packing a torch checkpoint onto kernel buffers is BUILD work, and the
    # run path resolves this module by name without torch installed.
    from ...layout import fit_lut, pack

    import math

    from ...ops.rms_norm_dpu_reduce import pack_norm_consts

    w_buf, consts_buf = f"{prefix}_w", f"{prefix}_consts"
    if w_buf in spec and site in norm:
        w = _pad_axis(norm[site], hd_idx, padded, axis=0) * math.sqrt(r)
        # D-major for rms_norm_dpu_reduce: pack reshapes (padded_head_dim,) ->
        # (head_dim_pw, PWORD), i.e. row k lane j = w[k*64 + j]. _pad_axis already put
        # the weight in the head's PHYSICAL lane order, so this is lane-aligned with the
        # input and the design-P pad lanes (zero input) stay inert.
        buffers[w_buf] = pack(w, spec[w_buf])
    rng_site = f"{site}.inv_sqrt_range"
    if consts_buf in spec and rng_site in rng:
        scaled = {
            **rng[rng_site],
            "min": rng[rng_site]["min"] * r,
            "max": rng[rng_site]["max"] * r,
        }
        (shape, dtype) = spec[consts_buf]
        n_luts = shape[0] - 2
        lut = fit_lut(scaled, ((n_luts, *shape[1:]), dtype))
        buffers[consts_buf] = pack_norm_consts(
            lut, (padded,), float(bundle.host_scalars["rms_eps"]) * r
        )


def conv_layer_buffers(bundle, spec, layer):
    """Pack one conv layer's constants: in/out_proj (W8A8), depthwise conv FIR
    taps (channel-major), and the shared SwiGLU MLP."""
    # Deferred: packing a torch checkpoint onto kernel buffers is BUILD work, and the
    # run path resolves this module by name without torch installed.
    from ...layout import layer_conv, layer_linears, layer_subset, pack, pack_dw2d_conv_weight

    lin = layer_linears(bundle.linear, layer)
    norm = layer_subset(bundle.norm, layer)
    rng = layer_subset(bundle.lut_ranges, layer)
    conv = layer_conv(bundle.conv, layer)

    buffers: dict = {}
    # in_proj emits B||C||x (each hidden-wide); the kernel runs three matmuls
    # sharing the input quant, so split the one in_proj into three sub-projections.
    in_proj = lin["in_proj"]
    hidden = int(in_proj["w"].shape[1])
    b_p, c_p, x_p = (_slice_out_rows(in_proj, i * hidden, (i + 1) * hidden) for i in range(3))
    for prefix, sub in (("b_proj", b_p), ("c_proj", c_p), ("x_proj", x_p)):
        if f"{prefix}_w" in spec:
            buffers[f"{prefix}_w"] = _nk(sub["w"])
        if f"{prefix}_dq" in spec:
            buffers[f"{prefix}_dq"] = _dq_table(sub)
    _put_qma_proj(buffers, spec, lin["out_proj"], "out_proj")
    _put_fused_mlp(buffers, spec, lin)
    # Hidden norms: consts only. in_proj's qs/qz are per-input and shared by B/C/x, and
    # gate/up share theirs, so each is bound once as one interleaved [scale, zp, ...]
    # buffer with the norm ahead of it folded into its scale.
    _put_hidden_norm_consts(buffers, spec, bundle, layer, int(bundle.config["hidden_size"]))
    _put_hidden_norm_folds(
        buffers, spec, norm, {CONV_NORM_FOLD: in_proj, FFN_NORM_FOLD: lin["gate_proj"]}
    )
    _put_norms_and_luts(buffers, spec, norm, rng, {}, _MLP_LUT_SITES)
    # W8A8 depthwise conv (int8 DWPU): activation quant scale (conv_qs, symmetric so
    # no zero-point), int8 channel-major taps (conv_weight), per-channel dequant scale
    # (conv_ds = s_x*s_w). See export_conv_quant_params / ops.causal_conv1d.
    conv_p = conv["conv"]
    if "conv_qs" in spec:
        buffers["conv_qs"] = pack(conv_p["qs"], spec["conv_qs"])
    if "conv_weight" in spec:
        buffers["conv_weight"] = pack_dw2d_conv_weight(conv_p["w"], spec["conv_weight"])
    if "conv_ds" in spec:
        buffers["conv_ds"] = pack(conv_p["ds"], spec["conv_ds"])
    return buffers


# Implementation


class Lfm2OptimizedImplementation(DecoderLayerImplementation):
    """Optimized hybrid Metis implementation of LFM2: host embed -> per-layer
    full_attention / conv device kernels -> host lm_head."""

    name = "lfm2-metis-opt"
    architecture = "lfm2"

    #: Union of every kind's shared_inputs: the attention kind's RoPE/mask/seq_pos set
    #: plus the conv kind's per-call state_scale reset gate. Drives the host input
    #: builders (prefill/step/chunk); each kind's graph binds only its own subset.
    shared_inputs = DecoderLayerImplementation.shared_inputs | {"state_scale", "valid_len"}

    #: conv_L_cache is 3 for every LFM2 size, so the carried window is K-1 = 2 rows and a
    #: padded chunk must hold at least that many real tokens; a 1-token remainder replays
    #: through decode instead. See ops/causal_conv1d.valid_len.
    min_padded_remainder = 2

    #: Measured on OMEGA at the default 1024-wide chunk: one padded prefill chunk beats
    #: replaying the remainder through decode from 8 real tokens up, i.e. chunk_len // 128.
    #: The base default of 4 would replay every prompt up to 256 tokens -- which is every
    #: prompt an `axzoo benchmark --workload generate` run sends -- through decode.
    padded_chunk_breakeven_ratio = 128

    chunked_remainder_via_padded_chunk = True

    #: Matches the attention kernel's attn_mask buffer: one int8 row broadcast across
    #: q heads by SDPA, not num_q_heads fp32 rows (see
    #: make_attention_layer_kernel_multicore).
    mask_num_heads = 1
    mask_dtype = np.int8

    def _mixer_state_inputs(self, base_position: int, n_real: int) -> dict:
        # The conv mixer's carried K-1 window is a sequential accumulator (unlike the
        # position-indexed KV cache), so it must reset at a sequence start. state_scale
        # = 0.0 at position 0 (fresh) else 1.0 (continue); see make_state_scale.
        return {
            "state_scale": make_state_scale(base_position),
            # n_real < seq_len only for a right-padded chunk; the op then persists the
            # window from the last REAL rows instead of the buffer tail.
            "valid_len": make_valid_len(n_real),
        }

    def host_kernels(self):
        return [
            HostKernel("embedding", params=("embed_weight",)),
            HostKernel("lm_head", params=("final_norm_weight", "embed_weight", "rms_eps")),
        ]

    def layer_kernels(self, model_config):
        attn = LayerKernel(
            "full_attention",
            build=make_attention_layer_kernel_multicore,
            layer_buffers=attention_layer_buffers,
            phases=("prefill", "decode"),
            # Attention binds RoPE/mask/seq_pos but not the conv-only state_scale.
            shared_inputs=DecoderLayerImplementation.shared_inputs,
            state=frozenset({"k_out", "v_out"}),
            enable_buffer_promotion=True,
            # Fusion merges the quant/dequant chains around the projections and SDPA into
            # single datapath invocations, cutting DDR round trips; verified correct and a
            # significant prefill win on this kernel (unlike the hand-built mixers, which
            # is why the LayerKernel default stays off).
            enable_datapath_fusion=False,
        )
        conv = LayerKernel(
            "conv",
            build=make_conv_layer_kernel_multicore,
            layer_buffers=conv_layer_buffers,
            phases=("prefill", "decode"),
            # The conv mixer has no RoPE / mask / seq_pos; its only runtime input is the
            # per-call state_scale reset gate for its window accumulator.
            shared_inputs=frozenset({"state_scale", "valid_len"}),
            state=frozenset({"conv_state"}),
            # The chunked causal_conv1d stages an extended-input DDR scratch that the
            # buffer-promotion pass mis-groups (assertion in AnnotatePromotion); the
            # conv mixer is weight-stream bound, not L2-residency bound, so promotion
            # buys little here. Disabled until the conv scratch is promotion-safe.
            enable_buffer_promotion=False,
            # Fusion is safe on the conv mixer's generated (not hand-built) datapaths.
            enable_datapath_fusion=True,
        )
        return {"full_attention": attn, "conv": conv}

    def layer_kinds(self, model_config):
        return [
            "full_attention" if model_config.layer_type(i) == "full_attention" else "conv"
            for i in range(model_config.num_hidden_layers)
        ]

    def _rope_tables(self, position, head_dim, rope_theta):
        """Plain RoPE cos/sin in the design-P padded-head layout.

        head_dim=64: rope_dim_pw blocks with head_dim/2 freqs in the low lanes,
        pad lanes 0. For head_dim>=128 this matches the default half-rotation tables.
        """
        import numpy as np

        d = _design_p_dims(int(head_dim))
        half, rope_dim_pw = d["half"], d["rope_dim_pw"]
        inv_freq = 1.0 / (
            float(rope_theta) ** (np.arange(0, int(head_dim), 2, dtype=np.float64) / int(head_dim))
        )
        pos = np.atleast_1d(np.asarray(position, dtype=np.float64))
        angles = pos[:, None] * inv_freq[None, :]  # (n_pos, half)
        cos = np.zeros((1, pos.size, rope_dim_pw, 64), dtype=np.float32)
        sin = np.zeros((1, pos.size, rope_dim_pw, 64), dtype=np.float32)
        flat_cos, flat_sin = np.cos(angles).astype(np.float32), np.sin(angles).astype(np.float32)
        for b in range(rope_dim_pw):
            lo, hi = b * 64, min((b + 1) * 64, half)
            cos[0, :, b, : hi - lo] = flat_cos[:, lo:hi]
            sin[0, :, b, : hi - lo] = flat_sin[:, lo:hi]
        return cos, sin


register_implementation(Lfm2OptimizedImplementation())

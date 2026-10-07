# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Llama 3.2 Metis multicore implementation (registered as llama-metis-multicore).

Llama-specific decoder layer on the shared ``models.multicore_layer`` host contract +
helpers. Llama bits supplied here: the kernel op sequence (NO per-head q/k RMSNorm),
design-P head padding (logical 64 -> padded 128), the shadow-spec buffer packer that
inherits the design-P padding from the Llama optimized packer, and the llama3-rescaled
RoPE table (``_trig`` override -> ``self._rope_tables``).

Adopts the qwen3 optimizations: int8 ({0,-127}) 1-head causal mask (was fp16),
conditional hidden-norm dispatch (was always-dpu), and core SDPA for both phases (the
decode-specialized sdpa_decode is no longer used). Multicore via cores>1.
"""

# flake8: noqa: F821  -- A.Buffer dtype-string annotations look like F821 names

from typing import TYPE_CHECKING

import numpy as np

from ...implementation import register_implementation
from ..multicore_layer import (
    HIDDEN_NORM_FOLDS,
    INPUT_NORM_FOLD,
    POST_ATTN_NORM_FOLD,
    PWORD,
    ModelSpec,
    MulticoreDecoderImplementation,
    add_norm_consts,
    check_norm_folds,
    hidden_norm_quant,
    layer_dims,
    quantize_activation,
)
from .naive_impl import LlamaMetisImplementation, _design_p_dims
from .optimized_impl import optimized_decode_layer_buffers

if TYPE_CHECKING:
    from .llama import LlamaConfig

LLAMA_SPEC = ModelSpec(has_qk_norm=False, has_lm_head=True)


def make_llama_layer_kernel(
    config: "LlamaConfig",
    ctx_len: int,
    phase: str,
    chunk_len: int | None = None,
    cores: int = 1,
):
    """Build the LlamaLayerFunctional IRModule for one Llama decoder layer.

    Design-P padding (logical head_dim 64 -> padded 128) via _design_p_dims; the
    attention scale uses the logical head_dim (the padding lanes are zeros). cores>1
    distributes SDPA heads + MLP matmuls across AI cores.
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol.common import BufferInfo
    from axelera.compiler.alto.axol.utils.lut_chain import ChainedLUTMethod
    import numpy as np

    from ...ops.fused_gate_up_swiglu import fused_gate_up_swiglu
    from ...ops.quantized_matmul_activation import (
        quantized_matmul_activation,
        quantized_matmul_activation_multicore,
    )

    head_dim_pw = _design_p_dims(config.head_dim)["head_dim_pw"]
    d = layer_dims(
        config,
        head_dim_pw=head_dim_pw,
        logical_head_dim=config.head_dim,
        ctx_len=ctx_len,
        phase=phase,
        chunk_len=chunk_len,
        spec=LLAMA_SPEC,
    )
    seq_len, hidden_dim, hidden_dim_pw = d.seq_len, d.hidden_dim, d.hidden_dim_pw
    logical_head_dim, head_dim_pw = d.logical_head_dim, d.head_dim_pw
    num_q_heads, num_kv_heads, context_len = d.num_q_heads, d.num_kv_heads, d.context_len
    q_out_dim, q_out_dim_pw, kv_out_dim_pw = d.q_out_dim, d.q_out_dim_pw, d.kv_out_dim_pw
    intermediate_dim, intermediate_dim_pw = d.intermediate_dim, d.intermediate_dim_pw

    _multicore = bool(cores and cores > 1)
    _sdpa_cores_kw = {"cores": cores} if _multicore else {}
    _gu_cores_kw = {"cores": cores} if _multicore else {}
    _down_matmul = (
        quantized_matmul_activation_multicore if _multicore else quantized_matmul_activation
    )
    _down_cores_kw = {"cores": cores} if _multicore else {}

    @A.ir_module
    class LlamaLayerFunctional:
        @A.prim_func
        def main(
            seq_pos: A.Buffer((1,), "uint64x64", scope=A.DDR),
            hidden_states: A.Buffer((1, seq_len, hidden_dim_pw), "float32x64", scope=A.DDR),
            # [inv-sqrt LUT rows, 1/D, eps] in one buffer: see pack_norm_consts.
            rms1_consts: A.Buffer((4,), "float32x64", scope=A.DDR),
            q_proj_qparams: A.Buffer((2 * hidden_dim_pw,), "float32x64", scope=A.DDR),
            q_proj_w: A.Buffer((q_out_dim_pw, hidden_dim), "int8x64", scope=A.DDR),
            q_proj_dq: A.Buffer((1, 1, 2 * q_out_dim_pw), "float32x64", scope=A.DDR),
            k_proj_w: A.Buffer((kv_out_dim_pw, hidden_dim), "int8x64", scope=A.DDR),
            k_proj_dq: A.Buffer((1, 1, 2 * kv_out_dim_pw), "float32x64", scope=A.DDR),
            v_proj_w: A.Buffer((kv_out_dim_pw, hidden_dim), "int8x64", scope=A.DDR),
            v_proj_dq: A.Buffer((1, 1, 2 * kv_out_dim_pw), "float32x64", scope=A.DDR),
            rope_trig: A.Buffer((1, seq_len, head_dim_pw), "float32x64", scope=A.DDR),
            k_cache_qparams: A.Buffer((2 * head_dim_pw,), "float32x64", scope=A.DDR),
            v_qparams: A.Buffer((2 * head_dim_pw,), "float16x64", scope=A.DDR),
            k_cache: A.Buffer((1, num_kv_heads, context_len, head_dim_pw), "int8x64", scope=A.DDR),
            v_cache: A.Buffer((1, num_kv_heads, context_len, head_dim_pw), "int8x64", scope=A.DDR),
            q_attn_qparams: A.Buffer((2 * head_dim_pw,), "float32x64", scope=A.DDR),
            sdpa_exp_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            sdpa_inv_lut: A.Buffer((1,), "float32x64", scope=A.DDR),
            sdpa_scores_dq_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            sdpa_weights_q_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            sdpa_output_dq_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            attn_mask: A.Buffer((1, 1, seq_len, context_len // 64), "int8x64", scope=A.DDR),
            o_qparams: A.Buffer((2 * q_out_dim_pw,), "float16x64", scope=A.DDR),
            o_proj_w: A.Buffer((hidden_dim_pw, q_out_dim), "int8x64", scope=A.DDR),
            o_proj_dq: A.Buffer((1, 1, 2 * hidden_dim_pw), "float32x64", scope=A.DDR),
            residual_in: A.Buffer((1, seq_len, hidden_dim_pw), "float32x64", scope=A.DDR),
            rms2_consts: A.Buffer((4,), "float32x64", scope=A.DDR),
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

            qkv_q = hidden_norm_quant(
                hidden_states, rms1_consts, q_proj_qparams, d, INPUT_NORM_FOLD
            )

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

            # q/k head path (no QK-Norm for Llama): RoPE -> standalone quantize.
            k_roped = axol.core.rope(k_heads, rope_trig)
            k_update = quantize_activation(
                k_roped, k_cache_qparams, (num_kv_heads, seq_len), head_dim_pw
            )
            v_update = quantize_activation(
                v_heads, v_qparams, (num_kv_heads, seq_len), head_dim_pw
            )

            k_out, v_out = axol.core.update_kv_cache(k_update, v_update, k_cache, v_cache, seq_pos)

            q_roped = axol.core.rope(q_heads, rope_trig)
            q_attn = quantize_activation(
                q_roped, q_attn_qparams, (num_q_heads, seq_len), head_dim_pw
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
                scale=1.0 / (logical_head_dim**0.5),  # logical head_dim (padding is zeros)
                neg_exp_method=ChainedLUTMethod.ADDITIVE,
                inv_method=ChainedLUTMethod.ADDITIVE,
                # Fuse the o_proj-input activation quantize into SDPA's P@V write -> int8
                # output (drops the separate dpu_cast + quantize; the transpose runs int8).
                attention_output_quantizer=o_qparams,
                **_sdpa_cores_kw,
            )

            attn_t = axol.core.transpose(attn_out, dims=(0, 2, 1, 3))
            attn_flat = axol.atomic.reshape(attn_t, shape=(1, seq_len, q_out_dim_pw))

            o6 = quantized_matmul_activation(
                attn_flat, o_proj_w, o_proj_dq, output=BufferInfo(dtype="float32x64")
            )

            residual = axol.atomic.add(residual_in, o6)

            gu_q = hidden_norm_quant(
                residual, rms2_consts, gate_proj_qparams, d, POST_ATTN_NORM_FOLD
            )

            mlp_split = fused_gate_up_swiglu(
                gu_q, gate_up_proj_w, mlp_dq_params, luts=silu_lut, **_gu_cores_kw
            )

            d6 = _down_matmul(
                mlp_split,
                down_proj_w,
                down_dq_params,
                **_down_cores_kw,
                matrix_a_is_split=True,
                output=BufferInfo(dtype="float32x64"),
            )

            A.output(axol.atomic.add(residual, d6))

    return LlamaLayerFunctional


# claude qparams buffer -> (qs name, qz name, leading) the Llama packers produce. These
# names are exactly what optimized_decode_layer_buffers populates (q/k/v/o input quants
# + the design-P-padded SDPA per-head scales), so the head padding is inherited; we just
# interleave the packed qs/qz pairs into one buffer.
_QPARAM_QSZ = {
    "q_proj_qparams": ("q_proj_qs", "q_proj_qz", (1, 1)),
    "gate_proj_qparams": ("gate_proj_qs", "gate_proj_qz", (1, 1)),
    "o_qparams": ("o_proj_qs", "o_proj_qz", (1, 1)),
    "k_cache_qparams": ("k_cache_qs", "k_cache_qz", (1, 1, 1)),
    "v_qparams": ("v_cache_qs", "v_cache_qz", (1, 1, 1)),
    "q_attn_qparams": ("q_attn_qs", "q_attn_qz", (1, 1, 1)),
}


def llama_layer_buffers(bundle, spec, layer):
    """Pack one Llama layer's constants.

    Runs the tested Llama optimized packer over a SHADOW spec that uses the baseline
    qs/qz buffer names (so all design-P head padding is applied exactly as in
    llama-metis-opt), then interleaves each packed qs/qz pair into the single
    [scale, zp, ...] qparams buffer the fused ops consume, folding the hidden RMSNorm
    weights into the two scales that follow them (HIDDEN_NORM_FOLDS).
    """
    import torch

    from ...layout import layer_subset, pack

    shadow = {k: v for k, v in spec.items() if k not in _QPARAM_QSZ}
    for qp_name, (qs_name, qz_name, lead) in _QPARAM_QSZ.items():
        if qp_name not in spec:
            continue
        (qp_shape, qp_dtype) = spec[qp_name]
        blocks = qp_shape[0] // 2  # qparams is (2*blocks, PWORD)
        qsz_shape = (*lead, blocks, PWORD)
        shadow[qs_name] = (qsz_shape, qp_dtype)
        shadow[qz_name] = (qsz_shape, qp_dtype)

    buffers = optimized_decode_layer_buffers(bundle, shadow, layer)
    hidden_dim = int(bundle.config["hidden_size"])
    add_norm_consts(
        buffers,
        spec,
        bundle,
        layer,
        {
            "rms1_consts": ("input_layernorm.inv_sqrt_range", hidden_dim),
            "rms2_consts": ("post_attention_layernorm.inv_sqrt_range", hidden_dim),
        },
    )

    # Interleave each packed qs/qz pair -> qparams: qp[0::2]=scale, qp[1::2]=zp.
    norm = layer_subset(bundle.norm, layer)
    folded = {f.qparams: f.site for f in HIDDEN_NORM_FOLDS}
    for qp_name, (qs_name, qz_name, _lead) in _QPARAM_QSZ.items():
        if qp_name not in spec:
            continue
        qs_b = buffers.pop(qs_name).reshape(-1, PWORD).float()
        qz_b = buffers.pop(qz_name).reshape(-1, PWORD)
        site = folded.get(qp_name)
        if site is not None:
            qs_b = qs_b * torch.as_tensor(norm[site]).float().reshape_as(qs_b)
        qp = torch.empty(2 * qs_b.shape[0], PWORD, dtype=qs_b.dtype)
        qp[0::2] = qs_b
        qp[1::2] = qz_b
        buffers[qp_name] = pack(qp, spec[qp_name])

    check_norm_folds(buffers)
    return buffers


class LlamaMulticoreImplementation(MulticoreDecoderImplementation, LlamaMetisImplementation):
    """Llama 3.2 Metis multicore implementation (llama3-rescaled RoPE; no QK-Norm).

    MRO: the shared multicore base supplies the int8-mask host inputs + kernels(); the
    Llama base supplies llama3 _rope_tables (used by the _trig override) + embed_tokens.
    """

    name = "llama-metis-multicore"
    architecture = "llama"
    spec = LLAMA_SPEC
    layer_build = staticmethod(make_llama_layer_kernel)
    layer_buffers_fn = staticmethod(llama_layer_buffers)

    def _trig(self, positions, host_params):
        # Design-P + llama3-rescaled cos/sin (inherited _rope_tables), combined sin||cos.
        cos, sin = self._rope_tables(
            positions, int(host_params["head_dim"]), float(host_params["rope_theta"])
        )
        return np.ascontiguousarray(np.concatenate([sin, cos], axis=2))


register_implementation(LlamaMulticoreImplementation())

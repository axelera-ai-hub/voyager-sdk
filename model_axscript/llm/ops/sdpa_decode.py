#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Decode-optimized scaled dot-product attention (llm.sdpa_decode).

A drop-in alternative to axol.core.scaled_dot_product_attention specialised
for the decode phase (short query, seq_len small).  Same operands, same result
(emulation delegates to the core SDPA reference), different schedule:

The softmax denominator (1/sum(exp)) is stashed per query row in a DPU bank-B
register (denom_reg[m]) in the QK+softmax datapath, then reused in the P@V
datapath (scaled * denom_reg[m]) to apply the normalization during the output
matmul -- avoiding a separate normalize pass over the attention weights.  This
register handoff relies on bank B persisting across the two MVMEXE datapaths
on the same core.

Registered under its own name (llm.sdpa_decode); it does not replace the core
op.  The optimized qwen kernel selects it only when building the decode phase.
Numerics match standard SDPA, so emulate reuses the core reference.
"""
import logging

from axelera.compiler.alto.axol.common import sliced
from axelera.compiler.alto.axol.core.scaled_dot_product_attention import (
    _check_python_args,
    _sdpa_emulation,
    atomic,
    core,
)
from axelera.compiler.alto.axol.utils.lut_chain import ChainedLUTMethod, chained_lut_dpu
from axelera.compiler.alto.compiler.config import HardwareGeneration
from axelera.compiler.alto.compiler.dtypes import MetisTypes
from axelera.compiler.alto.compiler.op_registry import Arg, BufferInfo, ScriptOp, register_op
import axelera.compiler.alto.script as A
import axelera.compiler.backend.tir.ops as tir_ops
from axelera.compiler.config import get_compiler_config
import numpy as np
from tvm import tir
from tvm.tir import BufferLoad, BufferSlice, Ramp

# mypy: disable-error-code="attr-defined, index, arg-type"
# flake8: noqa: C901

PWORD = get_compiler_config().pword_len
_logger = logging.getLogger(__name__)


def _head_slice(buf, head, dim0_extent, dim1_extent):
    return BufferSlice(BufferLoad(buf, [head, Ramp(0, 1, dim0_extent), Ramp(0, 1, dim1_extent)]))


# Datapath 1: QK^T matmul + softmax. Stores 1/denominator in bank B per row.
@sliced
def _fused_matmul_softmax_decode(
    query: tir.BufferSlice,
    imc_k: tir.BufferSlice,
    attention_mask: tir.BufferSlice,
    attention_scores_dequantizer_scale: tir.BufferSlice,
    attention_weights_quantizer_scale: tir.BufferSlice,
    attention_weights_quantizer_zero_point: tir.BufferSlice,
    exp_lut: tir.BufferSlice,
    inv_lut: tir.BufferSlice,
    n_exp_luts: int,
    n_inv_luts: int,
    exp_method: ChainedLUTMethod,
    inv_method: ChainedLUTMethod,
    use_newton_raphson: bool,
    scale: float,
    quantized_scores: tir.BufferSlice,
    scale_const: tir.BufferSlice,
    newton_const: tir.BufferSlice,
    attention_scores_dequantizer_zero: tir.BufferSlice,
) -> None:
    seq_len = query.shape[0]
    context_len_blocks = imc_k.shape[1]
    head_dim_blocks = imc_k.shape[0]

    attention_scores_dequantizer_scale = A.alloc_buffer(
        tuple([attention_scores_dequantizer_scale.shape[-1]]),
        attention_scores_dequantizer_scale.dtype,
        scope=attention_scores_dequantizer_scale.scope(),
        data=attention_scores_dequantizer_scale.buffer.data,
    )
    attention_weights_quantizer_scale = A.alloc_buffer(
        tuple([attention_weights_quantizer_scale.shape[-1]]),
        attention_weights_quantizer_scale.dtype,
        scope=attention_weights_quantizer_scale.scope(),
        data=attention_weights_quantizer_scale.buffer.data,
    )
    attention_weights_quantizer_zero_point = A.alloc_buffer(
        tuple([attention_weights_quantizer_zero_point.shape[-1]]),
        attention_weights_quantizer_zero_point.dtype,
        scope=attention_weights_quantizer_zero_point.scope(),
        data=attention_weights_quantizer_zero_point.buffer.data,
    )

    with A.script():
        for m in A.serial(seq_len):
            with A.Datapath(kind=A.DP_MVMEXE):
                ifd_stream = A.decl_stream("float32x64")
                mask_stream = A.decl_stream(attention_mask.dtype)
                matmul_stream = A.decl_stream("int32x64")
                dpu_stream = A.decl_stream("int8x64")
                for n, r in A.grid(context_len_blocks, head_dim_blocks):
                    with A.IFD(num=0):
                        ifd_stream.push(query[m, r])
                for n in A.serial(context_len_blocks):
                    with A.pseudo.MatMul():
                        acc = A.mvm_accumulator()
                        A.reset_accumulator(acc)
                        for r in A.serial(0, head_dim_blocks):
                            acc += A.mvm_1x1(ifd_stream.pop(), imc_k[r, n])
                        matmul_stream.push(acc)
                for n in A.serial(context_len_blocks):
                    with A.IFD(num=1):
                        mask_stream.push(attention_mask[m, n])
                with A.DPU():
                    denom_reg = A.dpu.registers(A.dpu.RegBank.B)
                    x = A.array("float18x64", context_len_blocks)
                    y = A.array("float18x64", context_len_blocks)
                    max_var = A.var("float18x64")
                    sum_exp = A.var("float18x64")
                    temp = A.var("float18x64")
                    lut_acc = A.var("float18x64")
                    for i in A.serial(context_len_blocks):
                        x[i] = matmul_stream.pop()
                        x[i] = x[i] * A.cast(attention_scores_dequantizer_scale[0], "float18x64")
                        x[i] = x[i] + A.cast(attention_scores_dequantizer_zero[0], "float18x64")
                        x[i] = x[i] * A.cast(scale_const[0], "float18x64")
                        x[i] = x[i] + A.cast(mask_stream.pop(), "float18x64")
                    max_var = x[0]
                    for i in A.serial(1, context_len_blocks):
                        max_var = A.max(max_var, x[i])
                    max_var = tir_ops.dpu_reduce(max_var, operation=tir_ops.DPU_REDUCE_OP.MAX)
                    for i in A.serial(0, context_len_blocks):
                        y[i] = max_var - x[i]
                    if exp_method is ChainedLUTMethod.ADDITIVE and n_exp_luts > 1:
                        for i in A.serial(0, context_len_blocks):
                            x[i] = y[i]
                            y[i] = tir_ops.lut(x[i], A.cast(exp_lut[0], "float18x64"))
                            for k in A.serial(1, n_exp_luts):
                                lut_acc = tir_ops.lut(x[i], A.cast(exp_lut[k], "float18x64"))
                                y[i] = y[i] + lut_acc
                    else:
                        for i in A.serial(0, context_len_blocks):
                            y[i] = chained_lut_dpu(
                                y[i],
                                exp_lut,
                                n_exp_luts,
                                method=ChainedLUTMethod.COMPOSITION,
                                tmp=lut_acc,
                            )

                    denominator = A.var("float18x64")
                    denominator = y[0]
                    for i in A.serial(1, context_len_blocks):
                        denominator = denominator + y[i]
                    denominator = tir_ops.dpu_reduce(
                        denominator, operation=tir_ops.DPU_REDUCE_OP.ADD
                    )
                    if use_newton_raphson:
                        sum_exp = denominator
                    if inv_method is ChainedLUTMethod.ADDITIVE and n_inv_luts > 1:
                        lut_acc = denominator
                        denominator = tir_ops.lut(lut_acc, A.cast(inv_lut[0], "float18x64"))
                        for k in A.serial(1, n_inv_luts):
                            temp = tir_ops.lut(lut_acc, A.cast(inv_lut[k], "float18x64"))
                            denominator = denominator + temp
                    else:
                        denominator = chained_lut_dpu(
                            denominator,
                            inv_lut,
                            n_inv_luts,
                            method=ChainedLUTMethod.COMPOSITION,
                            tmp=lut_acc,
                        )
                    if use_newton_raphson:
                        temp = sum_exp * denominator
                        temp = A.cast(newton_const[0], "float18x64") - temp
                        denominator = denominator * temp
                    denom_reg[m] = denominator

                    for i in A.serial(context_len_blocks):
                        y[i] = y[i] * A.cast(attention_weights_quantizer_scale[0], "float18x64")
                        y[i] = y[i] + A.cast(
                            attention_weights_quantizer_zero_point[0], "float18x64"
                        )
                        dpu_stream.push(A.cast(y[i], "int8x64"))
                for n in A.serial(context_len_blocks):
                    with A.ODR():
                        quantized_scores[m, n] = dpu_stream.pop()


# Datapath 2: P@V matmul, applying 1/denominator from bank B during the matmul.
@sliced
def _fused_matmul_denominator_decode(
    quantized_scores: tir.BufferSlice,
    imc_v: tir.BufferSlice,
    attention_output_dequantizer_scale: tir.BufferSlice,
    output_buffer: tir.BufferSlice,
) -> None:
    seq_len = quantized_scores.shape[0]
    context_len_blocks = quantized_scores.shape[1]
    head_dim_blocks = imc_v.shape[1]
    attention_output_dequantizer_scale = A.alloc_buffer(
        tuple([attention_output_dequantizer_scale.shape[-1]]),
        attention_output_dequantizer_scale.dtype,
        scope=attention_output_dequantizer_scale.scope(),
        data=attention_output_dequantizer_scale.buffer.data,
    )
    with A.script():
        for m, n in A.grid(seq_len, head_dim_blocks):
            with A.Datapath(kind=A.DP_MVMEXE):
                ifd_stream = A.decl_stream("int8x64")
                matmul_stream = A.decl_stream("int32x64")
                dpu_stream = A.decl_stream("float32x64")
                for r in A.serial(context_len_blocks):
                    with A.IFD(num=0):
                        ifd_stream.push(quantized_scores[m, r])
                with A.pseudo.MatMul():
                    acc = A.mvm_accumulator()
                    A.reset_accumulator(acc)
                    for r in A.serial(context_len_blocks):
                        acc += A.mvm_1x1(ifd_stream.pop(), imc_v[r, n])
                    matmul_stream.push(acc)
                with A.DPU():
                    denom_reg = A.dpu.registers(A.dpu.RegBank.B)
                    scaled = A.var("float18x64")
                    scaled = A.cast(matmul_stream.pop(), "float18x64")
                    scaled = scaled * A.cast(attention_output_dequantizer_scale[0], "float18x64")
                    scaled = scaled * denom_reg[m]
                    dpu_stream.push(scaled)
                with A.ODR():
                    output_buffer[m, n] = dpu_stream.pop()


class SDPADecode(ScriptOp):
    """Decode-specialised scaled dot-product attention. See module docstring.

    Operands mirror axol.core.scaled_dot_product_attention so the optimized
    kernel can select this op (decode) or the core op (prefill) interchangeably.
    """

    query = Arg(MetisTypes[tir.Buffer, "..."], description="(batch, head, seq_q, embed_blocks).")
    key = Arg(MetisTypes[tir.Buffer, "..."], description="(batch, kv_heads, ctx, embed_blocks).")
    value = Arg(MetisTypes[tir.Buffer, "..."], description="(batch, kv_heads, ctx, embed_blocks).")
    attention_scores_dequantizer_scale = Arg(
        MetisTypes[tir.Buffer, "..."], description="q*k scale."
    )
    exp_lut = Arg(
        MetisTypes[tir.Buffer, "..."], description="exp(-x) LUT params (n_exp_luts, PWORD)."
    )
    inv_lut = Arg(MetisTypes[tir.Buffer, "..."], description="1/x LUT params (n_inv_luts, PWORD).")
    attention_weights_quantizer_scale = Arg(
        MetisTypes[tir.Buffer, "..."], description="weight q-scale."
    )
    attention_weights_quantizer_zero_point = Arg(
        MetisTypes[tir.Buffer, "..."], description="weight q-zp."
    )
    attention_weights_quantizer_qmin = Arg(
        MetisTypes[tir.Buffer, "..."], description="weight qmin (-128)."
    )
    attention_weights_quantizer_qmax = Arg(
        MetisTypes[tir.Buffer, "..."], description="weight qmax (127)."
    )
    attention_output_dequantizer_scale = Arg(
        MetisTypes[tir.Buffer, "..."], description="fq*v scale."
    )
    query_quantizer_zero_point = Arg(
        MetisTypes[tir.Buffer, "..."], description="query q-zp (must be 0)."
    )
    value_quantizer_zero_point = Arg(
        MetisTypes[tir.Buffer, "..."], description="value q-zp (must be 0)."
    )
    attention_mask = Arg(
        MetisTypes[tir.Buffer, "..."], description="(*leading, seq_q, ctx_blocks)."
    )

    is_causal = Arg(bool, default=False, kw=True, description="Unsupported on HW; must be False.")
    enable_gqa = Arg(bool, default=False, kw=True, description="Grouped-query attention.")
    use_newton_raphson = Arg(
        bool, default=False, kw=True, description="One Newton step after the inv LUT."
    )
    scale = Arg(float, default=1.0, kw=True, description="Dot-product scaling factor.")
    exp_method = Arg(
        ChainedLUTMethod,
        default=ChainedLUTMethod.ADDITIVE,
        kw=True,
        description="exp LUT chaining.",
    )
    inv_method = Arg(
        ChainedLUTMethod,
        default=ChainedLUTMethod.ADDITIVE,
        kw=True,
        description="inv LUT chaining.",
    )
    output = Arg(
        MetisTypes[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="Output, auto-inferred (batch, head, seq_q, embed_v_blocks) float32x64 DDR.",
    )
    generation = Arg(
        HardwareGeneration | None,
        default=None,
        kw=True,
        description="Optional hardware generation.",
    )

    def validate(self) -> None:
        _check_python_args(
            is_causal=bool(self.is_causal),
            enable_gqa=bool(self.enable_gqa),
            num_query_heads=int(self.query.shape[1]),
            num_kv_heads=int(self.key.shape[1]),
        )

    def infer_outputs(self) -> dict[str, BufferInfo]:
        batch, head, seq_len_q, _ = self.query.shape
        embed_v_blocks = self.value.shape[-1]
        return {
            "output": BufferInfo(
                shape=(batch, head, seq_len_q, embed_v_blocks),
                dtype="float32x64",
                scope=A.DDR,
            )
        }

    # ------------------------------------------------------------------ #
    def implementation(self) -> None:
        query = self.query
        key = self.key
        value = self.value
        attention_scores_dequantizer_scale = self.attention_scores_dequantizer_scale
        exp_lut = self.exp_lut
        inv_lut = self.inv_lut
        attention_weights_quantizer_scale = self.attention_weights_quantizer_scale
        attention_weights_quantizer_zero_point = self.attention_weights_quantizer_zero_point
        attention_output_dequantizer_scale = self.attention_output_dequantizer_scale
        attention_mask = self.attention_mask
        output = self.output

        enable_gqa = bool(self.enable_gqa)
        use_newton_raphson = bool(self.use_newton_raphson)
        scale = float(self.scale)
        exp_method = ChainedLUTMethod(self.exp_method)
        inv_method = ChainedLUTMethod(self.inv_method)

        _check_python_args(
            is_causal=bool(self.is_causal),
            enable_gqa=enable_gqa,
            num_query_heads=int(query.shape[1]),
            num_kv_heads=int(key.shape[1]),
        )

        batch_size = query.shape[0]
        num_heads = int(query.shape[1])
        num_kv_heads = int(key.shape[1])
        seq_len = int(query.shape[2])
        context_len = int(key.shape[2])
        context_len_padded = ((context_len + PWORD - 1) // PWORD) * PWORD
        context_pad = context_len_padded - context_len
        context_len_blocks = context_len_padded // PWORD
        mask_context_len_blocks = int(attention_mask.shape[-1])
        mask_pad_blocks = context_len_blocks - mask_context_len_blocks
        head_dim_blocks = key.shape[3]
        kv_groups = num_heads // num_kv_heads

        n_exp_luts = int(exp_lut.shape[0])
        n_inv_luts = int(inv_lut.shape[0])

        _n_mask_leading = len(attention_mask.shape) - 2
        _mask_broadcasts_head = _n_mask_leading >= 2 and int(attention_mask.shape[1]) == 1
        _mask_broadcasts_batch = _n_mask_leading >= 1 and int(attention_mask.shape[0]) == 1
        _mask_dedup_at_kv_h = _mask_broadcasts_head
        _mask_dedup_at_batch = _mask_broadcasts_head and _mask_broadcasts_batch

        imc_k = A.alloc_imc((head_dim_blocks, context_len_blocks))
        imc_v = A.alloc_imc((context_len_blocks, head_dim_blocks))
        key_T = A.alloc_buffer(
            (head_dim_blocks * PWORD, context_len_blocks), dtype=key.dtype, scope=A.L1
        )

        key_copy_l1 = A.alloc_buffer((context_len, head_dim_blocks), dtype=key.dtype, scope=A.L1)
        key_l1 = key_copy_l1
        if context_pad > 0:
            key_l1 = A.alloc_buffer(
                (context_len_padded, head_dim_blocks), dtype=key.dtype, scope=A.L1
            )

        value_copy_l1 = A.alloc_buffer(
            (context_len, head_dim_blocks), dtype=value.dtype, scope=A.L1
        )
        value_l1 = value_copy_l1
        if context_pad > 0:
            value_l1 = A.alloc_buffer(
                (context_len_padded, head_dim_blocks), dtype=value.dtype, scope=A.L1
            )

        query_l1_group = A.alloc_buffer(
            (kv_groups, seq_len, head_dim_blocks), dtype=query.dtype, scope=A.L1
        )
        output_l1_group = A.alloc_buffer(
            (kv_groups, seq_len, output.shape[-1]), dtype=output.dtype, scope=A.L1
        )

        mask_copy_l1 = A.alloc_buffer(
            (seq_len, mask_context_len_blocks), dtype=attention_mask.dtype, scope=A.L1
        )
        mask_l1 = mask_copy_l1
        if mask_pad_blocks > 0:
            mask_l1 = A.alloc_buffer(
                (seq_len, context_len_blocks), dtype=attention_mask.dtype, scope=A.L1
            )

        quantized_scores_list = [
            A.alloc_buffer((seq_len, context_len_blocks), "int8x64", scope=A.L1)
            for _ in range(kv_groups)
        ]

        scores_deq_l1 = A.alloc_buffer(
            attention_scores_dequantizer_scale.shape,
            attention_scores_dequantizer_scale.dtype,
            scope=A.L1,
        )
        weights_q_scale_l1 = A.alloc_buffer(
            attention_weights_quantizer_scale.shape,
            attention_weights_quantizer_scale.dtype,
            scope=A.L1,
        )
        weights_q_zp_l1 = A.alloc_buffer(
            attention_weights_quantizer_zero_point.shape,
            attention_weights_quantizer_zero_point.dtype,
            scope=A.L1,
        )
        output_deq_l1 = A.alloc_buffer(
            attention_output_dequantizer_scale.shape,
            attention_output_dequantizer_scale.dtype,
            scope=A.L1,
        )

        scale_const_ddr = A.alloc_constant(
            np.array([[scale] * PWORD], dtype=np.float32), dtype="float32x64"
        )
        newton_const_ddr = A.alloc_constant(
            np.array([[2.0] * PWORD], dtype=np.float32), dtype="float32x64"
        )
        zero_const_ddr = A.alloc_constant(
            np.array([[0.0] * PWORD], dtype=np.float32), dtype="float32x64"
        )
        scale_const_l1 = A.alloc_buffer((1,), "float32x64", scope=A.L1)
        newton_const_l1 = A.alloc_buffer((1,), "float32x64", scope=A.L1)
        zero_const_l1 = A.alloc_buffer((1,), "float32x64", scope=A.L1)

        with A.serial(batch_size) as batch:
            atomic.copy(attention_scores_dequantizer_scale, output=scores_deq_l1)
            atomic.copy(attention_weights_quantizer_scale, output=weights_q_scale_l1)
            atomic.copy(attention_weights_quantizer_zero_point, output=weights_q_zp_l1)
            atomic.copy(attention_output_dequantizer_scale, output=output_deq_l1)

            atomic.copy(scale_const_ddr, output=scale_const_l1)
            atomic.copy(newton_const_ddr, output=newton_const_l1)
            atomic.copy(zero_const_ddr, output=zero_const_l1)

            if _mask_dedup_at_batch:
                _loop_refs = [0, 0]
                _mask_leading = tuple(
                    _loop_refs[i] if int(attention_mask.shape[i]) > 1 else 0
                    for i in range(_n_mask_leading)
                )
                atomic.copy(
                    attention_mask[_mask_leading + (slice(None), slice(None))],
                    output=mask_copy_l1,
                )
                if mask_pad_blocks > 0:
                    atomic.pad(
                        mask_copy_l1,
                        pad=((0, 0), (0, mask_pad_blocks)),
                        mode="constant",
                        value=-10000,
                        output=mask_l1,
                    )

            with A.serial(num_kv_heads) as kv_h:
                atomic.copy(key[batch, kv_h, :, :], output=key_copy_l1)
                if context_pad > 0:
                    atomic.pad(
                        key_copy_l1,
                        pad=((0, context_pad), (0, 0)),
                        mode="constant",
                        value=0,
                        output=key_l1,
                    )
                core.transpose(key_l1, (1, 0), key_T)

                atomic.copy(value[batch, kv_h, :, :], output=value_copy_l1)
                if context_pad > 0:
                    atomic.pad(
                        value_copy_l1,
                        pad=((0, context_pad), (0, 0)),
                        mode="constant",
                        value=0,
                        output=value_l1,
                    )

                if _mask_dedup_at_kv_h and not _mask_dedup_at_batch:
                    _loop_refs = [batch, 0]
                    _mask_leading = tuple(
                        _loop_refs[i] if int(attention_mask.shape[i]) > 1 else 0
                        for i in range(_n_mask_leading)
                    )
                    atomic.copy(
                        attention_mask[_mask_leading + (slice(None), slice(None))],
                        output=mask_copy_l1,
                    )
                    if mask_pad_blocks > 0:
                        atomic.pad(
                            mask_copy_l1,
                            pad=((0, 0), (0, mask_pad_blocks)),
                            mode="constant",
                            value=-10000,
                            output=mask_l1,
                        )

                atomic.copy(
                    query[batch, kv_h * kv_groups : (kv_h + 1) * kv_groups, :, :],
                    output=query_l1_group,
                )

                atomic.load_imc(key_T, imc_k)

                for g in range(kv_groups):
                    head = kv_h * kv_groups + g

                    if not _mask_dedup_at_kv_h:
                        _loop_refs = [batch, head]
                        _mask_leading = tuple(
                            _loop_refs[i] if int(attention_mask.shape[i]) > 1 else 0
                            for i in range(_n_mask_leading)
                        )
                        atomic.copy(
                            attention_mask[_mask_leading + (slice(None), slice(None))],
                            output=mask_copy_l1,
                        )
                        if mask_pad_blocks > 0:
                            atomic.pad(
                                mask_copy_l1,
                                pad=((0, 0), (0, mask_pad_blocks)),
                                mode="constant",
                                value=-10000,
                                output=mask_l1,
                            )

                    _fused_matmul_softmax_decode(
                        query=_head_slice(query_l1_group, g, seq_len, head_dim_blocks),
                        imc_k=imc_k,
                        attention_mask=mask_l1,
                        attention_scores_dequantizer_scale=scores_deq_l1,
                        attention_weights_quantizer_scale=weights_q_scale_l1,
                        attention_weights_quantizer_zero_point=weights_q_zp_l1,
                        exp_lut=exp_lut,
                        inv_lut=inv_lut,
                        n_exp_luts=n_exp_luts,
                        n_inv_luts=n_inv_luts,
                        exp_method=exp_method,
                        inv_method=inv_method,
                        use_newton_raphson=use_newton_raphson,
                        scale=scale,
                        quantized_scores=quantized_scores_list[g],
                        scale_const=scale_const_l1,
                        newton_const=newton_const_l1,
                        attention_scores_dequantizer_zero=zero_const_l1,
                    )

                atomic.load_imc(value_l1, imc_v)

                for g in range(kv_groups):
                    _fused_matmul_denominator_decode(
                        quantized_scores=quantized_scores_list[g],
                        imc_v=imc_v,
                        attention_output_dequantizer_scale=output_deq_l1,
                        output_buffer=_head_slice(output_l1_group, g, seq_len, output.shape[-1]),
                    )

                atomic.copy(
                    output_l1_group,
                    output=output[batch, kv_h * kv_groups : (kv_h + 1) * kv_groups, :, :],
                )

    # ------------------------------------------------------------------ #
    def emulate(self) -> None:  # type: ignore[override]
        # Numerically identical to standard SDPA; reuse the core reference.
        _sdpa_emulation(
            self.query,
            self.key,
            self.value,
            self.attention_scores_dequantizer_scale,
            self.exp_lut,
            self.inv_lut,
            self.attention_weights_quantizer_scale,
            self.attention_weights_quantizer_zero_point,
            self.attention_weights_quantizer_qmin,
            self.attention_weights_quantizer_qmax,
            self.attention_output_dequantizer_scale,
            self.query_quantizer_zero_point,
            self.value_quantizer_zero_point,
            self.attention_mask,
            self.output,
            is_causal=bool(self.is_causal),
            enable_gqa=bool(self.enable_gqa),
            use_newton_raphson=bool(self.use_newton_raphson),
            scale=float(self.scale),
            exp_method=ChainedLUTMethod(self.exp_method).value,
            inv_method=ChainedLUTMethod(self.inv_method).value,
            generation=self.generation,
        )


sdpa_decode = register_op(
    SDPADecode,
    "llm.sdpa_decode",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)

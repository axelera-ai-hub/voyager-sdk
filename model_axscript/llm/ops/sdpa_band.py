#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Decode SDPA with the GQA group as one launch per stage (llm.sdpa_band).

The core op (``axol.core.scaled_dot_product_attention``) runs every query head of a KV
group on its own: per head a QK+softmax launch, a wait, a DPU cast launch, a wait, a DMA of
that head's output quant params and a P@V launch. This op keeps the core op's datapath code
per row (same DPU programs, same arithmetic, same register footprint) but feeds all
``kv_groups`` heads of the core's band through one launch of each stage: the query band is
the M dimension of the QK+softmax datapath, the cast covers the band, and P@V streams the
band's output quant params per (row, block) through IFD1. Decode only (batch 1, seq_len 1),
GQA, additive LUT chains. Bit-identical to the core op; ``emulate`` delegates to its
reference. The op runs one core's band of KV heads inside a region the caller has opened
(``band=(core, kv_lo, kv_hi)``); the caller partitions the heads and places the barriers.

The small operands are staged as explicit L1 copies, then the K/V stream, then the K
transpose and the two IMC programs. With ``packed_consts`` the scalar constants and LUT rows
arrive as one copy.
"""
# mypy: disable-error-code="attr-defined, index, arg-type"
# flake8: noqa: C901

from typing import Any

from axelera.compiler.alto.axol import atomic
from axelera.compiler.alto.axol import core as axcore
from axelera.compiler.alto.axol.common import BufferInfo, sliced
from axelera.compiler.alto.axol.core.scaled_dot_product_attention import (
    _check_python_args,
    _make_partitioned_imc_map,
    _sdpa_emulation,
)
from axelera.compiler.alto.axol.utils.lut_chain import ChainedLUTMethod
from axelera.compiler.alto.compiler.config import get_alto_config
from axelera.compiler.alto.compiler.device import IMC_WEIGHTSETS, PWORD
from axelera.compiler.alto.compiler.dtypes import MetisTypes
from axelera.compiler.alto.compiler.op_registry import Arg, ScriptOp, register_op
import axelera.compiler.alto.script as A
import axelera.compiler.backend.tir.ops as tir_ops
from axelera.config import HardwareGeneration
import numpy as np
from tvm import tir
from tvm.ir import Array

from .quantized_matmul_activation import bs

_tensor: Any = MetisTypes[tir.Buffer, "..."]

_LOAD_K_CHANNEL = 0
_LOAD_V_CHANNEL = 1
# The packed softmax program needs one DPU register more than the core op's, which is at the
# limit at 64 context blocks; wider contexts stage the separate constants.
PACKED_MAX_BLOCKS = 32

# Rows of the packed per-op constant buffer (float32x64 rows) the datapaths read.
CST_SCORES_DQ, CST_WQ_SCALE, CST_WQ_ZP, CST_OUT_DQ, CST_SCALE, CST_ZERO = range(6)
CST_EXP0 = 6


def cst_layout(n_exp_luts: int, n_inv_luts: int) -> tuple[int, int]:
    """(inv_lut slot, total rows) for a packed constant buffer with these LUT counts."""
    return CST_EXP0 + n_exp_luts, CST_EXP0 + n_exp_luts + n_inv_luts


def packed_rows(n_exp_luts: int, n_inv_luts: int) -> int:
    """Rows of the ``packed_consts`` buffer for these LUT counts (the kernel input's shape)."""
    return cst_layout(n_exp_luts, n_inv_luts)[1]


def pack_sdpa_consts(scores_dq, wq_scale, wq_zp, out_dq, scale: float, exp_lut, inv_lut):
    """Host-side packer for the ``packed_consts`` operand: rows of PWORD float32."""

    def row(v):
        v = np.asarray(v, dtype=np.float32).reshape(-1)
        return v[:PWORD] if v.size >= PWORD else np.full(PWORD, v[0], np.float32)

    rows = [row(v) for v in (scores_dq, wq_scale, wq_zp, out_dq)]
    rows += [np.full(PWORD, scale, np.float32), np.zeros(PWORD, np.float32)]
    rows.extend(np.asarray(exp_lut, dtype=np.float32).reshape(-1, PWORD))
    rows.extend(np.asarray(inv_lut, dtype=np.float32).reshape(-1, PWORD))
    return np.stack(rows).astype(np.float32)


def _alias(src, shape: tuple):
    """A same-data view of ``src`` with another shape (the core op's constant re-view)."""
    data = (src.buffer if isinstance(src, tir.BufferSlice) else src).data
    return A.alloc_buffer(shape, src.dtype, scope=src.scope(), data=data)


def _flat(buf):
    """``buf`` viewed 1-D."""
    return _alias(buf, (buf.shape[-1],))


@sliced
def _qk_softmax_band(
    query: tir.BufferSlice,
    imc_k: tir.BufferSlice,
    attention_mask: tir.BufferSlice,
    scores_with_stats: tir.BufferSlice,
    n_exp_luts: int,
    n_inv_luts: int,
    *,
    scores_dq,
    wq_scale,
    wq_zp,
    exp_lut,
    inv_lut,
    scale,
    zero,
) -> None:
    """The core op's ``_fused_matmul_softmax`` over a (heads, head_dim_blocks) query band,
    one mask row shared by every band row: one launch, ``heads`` hardware iterations. Each
    constant is a ``(1-D buffer, row)`` pair; the LUT rows follow their first row."""
    band = query.shape[0]
    context_len_blocks = imc_k.shape[1]
    head_dim_blocks = imc_k.shape[0]
    dq, dq_row = scores_dq
    wq, wq_row = wq_scale
    wz, wz_row = wq_zp
    exp_lut, exp0 = exp_lut
    inv_lut, inv0 = inv_lut
    scale, scale_row = scale
    zero, zero_row = zero
    with A.script():
        for m in A.serial(band):
            with A.Datapath(kind=A.DP_MVMEXE, id="sdpa_band_qk"):
                ifd_stream = A.decl_stream(query.dtype)
                mask_stream = A.decl_stream(attention_mask.dtype)
                matmul_stream = A.decl_stream("int32x64")
                dpu_stream = A.decl_stream("float16x64")
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
                        mask_stream.push(attention_mask[0, n])
                with A.DPU():
                    x = A.array("float18x64", context_len_blocks)
                    y = A.array("float18x64", context_len_blocks)
                    denominator = A.var("float18x64")
                    max_var = A.var("float18x64")
                    temp = A.var("float18x64")
                    lut_acc = A.var("float18x64")
                    for i in A.serial(context_len_blocks):
                        x[i] = matmul_stream.pop()
                        x[i] = x[i] * A.cast(dq[dq_row], "float18x64")
                        x[i] = x[i] + A.cast(zero[zero_row], "float18x64")
                        x[i] = x[i] * A.cast(scale[scale_row], "float18x64")
                        x[i] = x[i] + A.cast(mask_stream.pop(), "float18x64")
                    max_var = x[0]
                    for i in A.serial(1, context_len_blocks):
                        max_var = A.max(max_var, x[i])
                    max_var = tir_ops.dpu_reduce(max_var, operation=tir_ops.DPU_REDUCE_OP.MAX)
                    for i in A.serial(0, context_len_blocks):
                        y[i] = max_var - x[i]
                    for i in A.serial(0, context_len_blocks):
                        x[i] = y[i]
                        y[i] = tir_ops.lut(x[i], A.cast(exp_lut[exp0], "float18x64"))
                        for k in A.serial(1, n_exp_luts):
                            lut_acc = tir_ops.lut(x[i], A.cast(exp_lut[exp0 + k], "float18x64"))
                            y[i] = y[i] + lut_acc
                    denominator = y[0]
                    for i in A.serial(1, context_len_blocks):
                        denominator = denominator + y[i]
                    denominator = tir_ops.dpu_reduce(
                        denominator, operation=tir_ops.DPU_REDUCE_OP.ADD
                    )
                    lut_acc = denominator
                    denominator = tir_ops.lut(lut_acc, A.cast(inv_lut[inv0], "float18x64"))
                    for k in A.serial(1, n_inv_luts):
                        temp = tir_ops.lut(lut_acc, A.cast(inv_lut[inv0 + k], "float18x64"))
                        denominator = denominator + temp
                    for i in A.serial(context_len_blocks):
                        y[i] = y[i] * A.cast(wq[wq_row], "float18x64")
                        y[i] = y[i] + A.cast(wz[wz_row], "float18x64")
                        dpu_stream.push(y[i])
                    dpu_stream.push(denominator)
                for n in A.serial(context_len_blocks + 1):
                    with A.ODR():
                        scores_with_stats[m, n] = dpu_stream.pop()


@sliced
def _pv_band(
    quantized_scores: tir.BufferSlice,
    imc_v: tir.BufferSlice,
    scores_with_stats: tir.BufferSlice,
    output_quantizer: tir.BufferSlice,
    output_buffer: tir.BufferSlice,
    out_dq,
) -> None:
    """The core op's ``_fused_matmul_denominator`` over the band, as two chained launches of
    (heads x head_dim_blocks) iterations each: ``acc * out_dq * (1/sum)`` with the row's
    1/sum on IFD1 into an L1 float32 temporary (float18 values are exact there, so the
    round trip is lossless), then ``* scale + zp`` with the block's interleaved [scale, zp]
    pair on IFD1. Two launches because a datapath has one operand per IFD stream per launch
    (the tensorizer replicates the grid per IFD block, so two IFD1 operands would not
    interleave). ``out_dq`` is a ``(1-D buffer, row)`` pair."""
    band = quantized_scores.shape[0]
    context_len_blocks = quantized_scores.shape[1]
    head_dim_blocks = imc_v.shape[1]
    out_dq, out_dq_row = out_dq
    # The band's interleaved [scale, zp] params viewed (head, block, 2).
    oq = _alias(output_quantizer, (band, head_dim_blocks, 2))
    with A.script():
        pv = A.alloc_buffer((band, head_dim_blocks), "float32x64", scope=A.L1)
        for m, n in A.grid(band, head_dim_blocks):
            with A.Datapath(kind=A.DP_MVMEXE, id="sdpa_band_pv"):
                ifd_stream = A.decl_stream("int8x64")
                den_stream = A.decl_stream("float16x64")
                pv_stream = A.decl_stream("float32x64")
                matmul_stream = A.decl_stream("int32x64")
                for r in A.serial(context_len_blocks):
                    with A.IFD(num=0):
                        ifd_stream.push(quantized_scores[m, r])
                with A.pseudo.MatMul():
                    acc = A.mvm_accumulator()
                    A.reset_accumulator(acc)
                    for r in A.serial(context_len_blocks):
                        acc += A.mvm_1x1(ifd_stream.pop(), imc_v[r, n])
                    matmul_stream.push(acc)
                with A.IFD(num=1):
                    den_stream.push(scores_with_stats[m, context_len_blocks])
                with A.DPU():
                    scaled = A.var("float18x64")
                    scaled = A.cast(matmul_stream.pop(), "float18x64")
                    scaled = scaled * A.cast(out_dq[out_dq_row], "float18x64")
                    scaled = scaled * A.cast(den_stream.pop(), "float18x64")
                    pv_stream.push(A.cast(scaled, "float32x64"))
                with A.ODR():
                    pv[m, n] = pv_stream.pop()
        for m, n in A.grid(band, head_dim_blocks):
            with A.Datapath(id="sdpa_band_oq"):
                pv_in = A.decl_stream("float32x64")
                oq_stream = A.decl_stream(oq.dtype)
                dpu_stream = A.decl_stream("int8x64")
                with A.IFD(num=0):
                    pv_in.push(pv[m, n])
                for j in A.grid(2):
                    with A.IFD(num=1):
                        oq_stream.push(oq[m, n, j])
                with A.DPU():
                    scaled = A.var("float18x64")
                    scaled = A.cast(pv_in.pop(), "float18x64")
                    scaled = scaled * A.cast(oq_stream.pop(), "float18x64")
                    scaled = scaled + A.cast(oq_stream.pop(), "float18x64")
                    dpu_stream.push(A.cast(scaled, "int8x64"))
                with A.ODR():
                    output_buffer[m, n] = dpu_stream.pop()


class SDPABand(ScriptOp):
    """Decode SDPA, one launch per stage for the KV group. See the module docstring."""

    query = Arg(_tensor, description="Query, (batch, head, 1, embed_q_blocks).")
    key = Arg(_tensor, description="Key, (batch, kv_heads, ctx, embed_k_blocks).")
    value = Arg(_tensor, description="Value, (batch, kv_heads, ctx, embed_v_blocks).")
    attention_scores_dequantizer_scale = Arg(_tensor, description="q_scale * k_scale.")
    neg_exp_lut = Arg(_tensor, description="Packed exp(-x) LUT params, (n_exp_luts, PWORD).")
    inv_lut = Arg(_tensor, description="Packed 1/x LUT params, (n_inv_luts, PWORD).")
    attention_weights_quantizer_inverse_scale = Arg(_tensor, description="P quant scale.")
    attention_weights_quantizer_zero_point = Arg(_tensor, description="P quant zero point.")
    attention_weights_quantizer_qmin = Arg(_tensor, description="Must be -128.")
    attention_weights_quantizer_qmax = Arg(_tensor, description="Must be 127.")
    attention_output_dequantizer_scale = Arg(_tensor, description="fq_scale * v_scale.")
    query_quantizer_zero_point = Arg(_tensor, description="Must be zero.")
    value_quantizer_zero_point = Arg(_tensor, description="Must be zero.")
    attention_mask = Arg(_tensor, description="(1, 1, 1, context_len_blocks), broadcast.")
    attention_output_quantizer = Arg(
        _tensor,
        description="Interleaved per-channel [scale, zp] params of the int8 output.",
    )
    output = Arg(
        _tensor | BufferInfo | None,
        default=None,
        is_output=True,
        description="Output; auto-inferred (batch, head, 1, embed_v_blocks) int8.",
    )

    scale = Arg(float, default=1.0, kw=True, description="Dot-product scaling factor.")
    band = Arg(
        tuple | Array,
        kw=True,
        description="(core, kv_lo, kv_hi): this core's KV heads, run inside the caller's open "
        "AICore region (no barriers).",
    )
    before_kv = Arg(
        object,
        default=None,
        kw=True,
        description="Called once per band with the operands staged, before the K/V launches, "
        "for a caller that streams its own DMAs under SDPA.",
    )
    kv_landed = Arg(
        object,
        default=None,
        kw=True,
        description="Called once K and V are waited, before the transposes.",
    )
    end = Arg(
        object,
        default=None,
        kw=True,
        description="Called after the last datapath, before the output copy.",
    )
    packed_consts = Arg(
        _tensor | None,
        default=None,
        kw=True,
        description="Optional (rows, PWORD) float32 buffer with every scalar constant and LUT "
        "row packed (pack_sdpa_consts / cst_layout): one load instead of eight.",
    )
    generation = Arg(HardwareGeneration | None, default=None, kw=True)

    def validate(self) -> None:
        _check_python_args(
            is_causal=False,
            enable_gqa=True,
            num_query_heads=int(self.query.shape[1]),
            num_kv_heads=int(self.key.shape[1]),
        )
        if int(self.query.shape[2]) != 1 or int(self.query.shape[0]) != 1:
            raise NotImplementedError("sdpa_band is the decode (batch 1, seq_len 1) schedule")
        context_len = int(self.key.shape[2])
        if context_len % PWORD != 0:
            raise NotImplementedError(
                f"sdpa_band needs a PWORD-multiple context, got {context_len}"
            )
        if int(self.attention_mask.shape[-1]) != context_len // PWORD:
            raise NotImplementedError("sdpa_band needs a mask spanning the full context")
        if any(int(s) != 1 for s in self.attention_mask.shape[:-1]):
            raise NotImplementedError("sdpa_band needs a broadcast (1, 1, 1, blocks) mask")
        if len(tuple(self.band)) != 3:
            raise ValueError("band must be (core, kv_lo, kv_hi)")
        if self.packed_consts is not None:
            n_exp_luts, n_inv_luts = int(self.neg_exp_lut.shape[0]), int(self.inv_lut.shape[0])
            if int(self.packed_consts.shape[0]) != packed_rows(n_exp_luts, n_inv_luts):
                raise ValueError(f"packed_consts has {int(self.packed_consts.shape[0])} rows")
            if context_len // PWORD > PACKED_MAX_BLOCKS:
                raise ValueError(
                    f"packed_consts is limited to {PACKED_MAX_BLOCKS} context blocks (DPU "
                    f"registers); got {context_len // PWORD}"
                )

    def infer_outputs(self) -> dict[str, BufferInfo]:
        batch, head, seq_len_q, _ = self.query.shape
        _, _, _, embed_v_blocks = self.value.shape
        return {
            "output": BufferInfo(
                shape=(batch, head, seq_len_q, embed_v_blocks), dtype="int8x64", scope=A.DDR
            )
        }

    def implementation(self) -> None:
        query, key, value, output = self.query, self.key, self.value, self.output
        attention_mask = self.attention_mask
        output_quantizer = self.attention_output_quantizer
        generation = self.generation or get_alto_config().generation

        num_heads = int(query.shape[1])
        context_len = int(key.shape[2])
        context_len_blocks = context_len // PWORD
        head_dim_blocks = int(key.shape[3])
        dv_blocks = int(output.shape[-1])
        num_kv_heads = int(key.shape[1])
        kv_groups = num_heads // num_kv_heads
        n_exp_luts = int(self.neg_exp_lut.shape[0])
        n_inv_luts = int(self.inv_lut.shape[0])
        inv0, _ = cst_layout(n_exp_luts, n_inv_luts)
        n_mask_leading = len(attention_mask.shape) - 2

        core, kv_lo, kv_hi = (int(x) for x in self.band)
        packed = self.packed_consts
        before_kv, kv_landed, end = (
            fn or (lambda: None) for fn in (self.before_kv, self.kv_landed, self.end)
        )

        scale_const = A.alloc_constant(
            np.array([[float(self.scale)] * PWORD], dtype=np.float32), dtype="float32x64"
        )
        zero_const = A.alloc_constant(
            np.array([[0.0] * PWORD], dtype=np.float32), dtype="float32x64"
        )
        oq_flat = _alias(
            output_quantizer, (int(np.prod([int(s) for s in output_quantizer.shape])),)
        )

        half_ws = IMC_WEIGHTSETS[generation] // 2
        imc_k = A.alloc_imc(
            (head_dim_blocks, context_len_blocks),
            map=_make_partitioned_imc_map(
                (head_dim_blocks, context_len_blocks), weightset_offset=0, n_weightsets=half_ws
            ),
        )
        imc_v = A.alloc_imc(
            (context_len_blocks, dv_blocks),
            map=_make_partitioned_imc_map(
                (context_len_blocks, dv_blocks), weightset_offset=half_ws, n_weightsets=half_ws
            ),
        )
        key_l1 = A.alloc_buffer((context_len, head_dim_blocks), dtype=key.dtype, scope=A.L1)
        key_T = A.alloc_buffer(
            (head_dim_blocks * PWORD, context_len_blocks), dtype=key.dtype, scope=A.L1
        )
        value_l1 = A.alloc_buffer((context_len, dv_blocks), dtype=value.dtype, scope=A.L1)
        mask_l1 = A.alloc_buffer((1, context_len_blocks), dtype=attention_mask.dtype, scope=A.L1)
        scores_with_stats = A.alloc_buffer(
            (kv_groups, context_len_blocks + 1), "float16x64", scope=A.L1
        )
        quantized_scores = A.alloc_buffer((kv_groups, context_len_blocks), "int8x64", scope=A.L1)
        query_band_l1 = A.alloc_buffer(
            (kv_groups, 1, head_dim_blocks), dtype=query.dtype, scope=A.L1
        )
        output_band_l1 = A.alloc_buffer((kv_groups, 1, dv_blocks), dtype=output.dtype, scope=A.L1)
        oq_l1 = A.alloc_buffer((2 * kv_groups * dv_blocks,), output_quantizer.dtype, scope=A.L1)

        # Straight-line (batch 1): a loop over batch would let the loop-invariant
        # operand loads and the K/V sync be hoisted apart from the K/V loads.
        batch = 0
        # A user tiling scope (as the core op has): the hardware-constraint tiling leaves
        # it alone. Outside one, its re-emission of the multi-dim L1 loads put them behind
        # the wait that closes them ("token HP_DMA_0->SW not produced").
        with A.tiles(1):
            # The head-invariant operands, staged once per band as explicit L1 copies (a
            # software wait each; an operand handed to the datapath would be staged by the
            # compiler with a hardware token): the one packed buffer, or the eight constants
            # the core op stages the same way.
            if packed is not None:
                cst_l1 = atomic.copy(packed, scope=A.L1, core=core)
            else:
                dq_l1, wq_l1, wz_l1, out_dq_l1, exp_l1, inv_l1, scale_l1, zero_l1 = (
                    atomic.copy(s, scope=A.L1, core=core)
                    for s in (
                        self.attention_scores_dequantizer_scale,
                        self.attention_weights_quantizer_inverse_scale,
                        self.attention_weights_quantizer_zero_point,
                        self.attention_output_dequantizer_scale,
                        self.neg_exp_lut,
                        self.inv_lut,
                        scale_const,
                        zero_const,
                    )
                )

            def _consts():
                """The datapaths' constants as (1-D buffer, row) pairs: rows of ONE view of the
                packed copy (one view, so the reads stage as one operand), or row 0 of each
                copy. Made per head, where the datapaths that read them are."""
                if packed is not None:
                    cst = _flat(cst_l1)
                    rows = dict(
                        scores_dq=CST_SCORES_DQ,
                        wq_scale=CST_WQ_SCALE,
                        wq_zp=CST_WQ_ZP,
                        exp_lut=CST_EXP0,
                        inv_lut=inv0,
                        scale=CST_SCALE,
                        zero=CST_ZERO,
                    )
                    return {k: (cst, row) for k, row in rows.items()}, (_flat(cst_l1), CST_OUT_DQ)
                consts = dict(
                    scores_dq=(_flat(dq_l1), 0),
                    wq_scale=(_flat(wq_l1), 0),
                    wq_zp=(_flat(wz_l1), 0),
                    exp_lut=(exp_l1, 0),
                    inv_lut=(inv_l1, 0),
                    scale=(scale_l1, 0),
                    zero=(zero_l1, 0),
                )
                return consts, (_flat(out_dq_l1), 0)

            mask_src = attention_mask[(0,) * n_mask_leading + (slice(None), slice(None))]
            atomic.copy(mask_src, output=mask_l1, core=core)
            k_id, v_id = f"sdpa_band_k_{core}", f"sdpa_band_v_{core}"

            for kv_h in range(kv_lo, kv_hi):
                head_base, head_end = kv_h * kv_groups, (kv_h + 1) * kv_groups
                oq_lo = 2 * head_base * dv_blocks
                first_head, last_head = kv_h == kv_lo, kv_h == kv_hi - 1

                atomic.copy(
                    query[batch, head_base:head_end, :, :], output=query_band_l1, core=core
                )
                query_op = bs(
                    query_band_l1,
                    tir.Ramp(0, 1, kv_groups),
                    0,
                    tir.Ramp(0, 1, head_dim_blocks),
                )
                atomic.copy(
                    oq_flat[tir.Ramp(oq_lo, 1, 2 * kv_groups * dv_blocks)],
                    output=oq_l1,
                    core=core,
                )
                if first_head:
                    before_kv()
                atomic.copy(
                    key[batch, kv_h, :, :],
                    output=key_l1,
                    core=core,
                    channel=_LOAD_K_CHANNEL,
                    id=k_id,
                )
                atomic.copy(
                    value[batch, kv_h, :, :],
                    output=value_l1,
                    core=core,
                    channel=_LOAD_V_CHANNEL,
                    id=v_id,
                )
                A.waits(k_id, v_id)
                if first_head:
                    kv_landed()

                axcore.transpose(key_l1, (1, 0), key_T)
                atomic.load_imc(key_T, imc_k, core=core, generation=generation)
                atomic.load_imc(value_l1, imc_v, core=core, generation=generation)

                consts, out_dq = _consts()
                _qk_softmax_band(
                    query_op,
                    imc_k,
                    mask_l1,
                    scores_with_stats,
                    n_exp_luts,
                    n_inv_luts,
                    **consts,
                )
                atomic.dpu_cast(
                    scores_with_stats[0:kv_groups, 0:context_len_blocks],
                    quantized_scores,
                    core=core,
                )
                _pv_band(
                    quantized_scores,
                    imc_v,
                    scores_with_stats,
                    oq_l1,
                    bs(output_band_l1, tir.Ramp(0, 1, kv_groups), 0, tir.Ramp(0, 1, dv_blocks)),
                    out_dq,
                )
                if last_head:
                    end()
                atomic.copy(
                    output_band_l1, output=output[batch, head_base:head_end, :, :], core=core
                )

    def emulate(self) -> None:  # type: ignore[override]
        additive = ChainedLUTMethod.ADDITIVE.value
        _sdpa_emulation(
            self.query,
            self.key,
            self.value,
            self.attention_scores_dequantizer_scale,
            self.neg_exp_lut,
            self.inv_lut,
            self.attention_weights_quantizer_inverse_scale,
            self.attention_weights_quantizer_zero_point,
            self.attention_weights_quantizer_qmin,
            self.attention_weights_quantizer_qmax,
            self.attention_output_dequantizer_scale,
            self.query_quantizer_zero_point,
            self.value_quantizer_zero_point,
            self.attention_mask,
            self.output,
            False,
            True,
            False,
            self.scale,
            additive,
            additive,
            self.generation,
            output_quantizer=self.attention_output_quantizer,
        )


sdpa_band = register_op(
    SDPABand,
    "llm.sdpa_band",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)

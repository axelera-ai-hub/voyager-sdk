# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Fused DINOv2 ViT block kernel (the optimized counterpart to naive_impl).

Takes the same inspiration as qwen3/optimized_impl and multicore_layer. Every
linear collapses from a 7-op dequant chain (mul, add, cast, matmul, sub, mul,
add-bias) into a SINGLE llm.quantized_matmul_activation, which keeps the whole
(matmul - dz) * ds (+ LUT activation) inside one DPU datapath. On top of that:

  * the GELU MLP activation is FUSED into the fc1 matmul (luts=gelu_lut) -- no
    separate piecewise_linear_approximation pass;
  * the per-output-channel BIAS is folded into the dequant zero-point
    (dz' = dz - bias/ds), so (matmul - dz') * ds == matmul_dequant + bias -- no
    separate atomic.add per linear;
  * every input quantize (mul-scale, add-zp, cast-int8) is ONE llm.quantize
    datapath instead of three atomic passes over the activation (datapath fusion
    is off, being buggy) -- the multicore_layer pattern;
  * the LayerNorm output is quantized ONCE and reused by q/k/v (they share a
    SmoothQuant group, so their input quant is identical).

Weights move to NK layout (N_BLOCKS, K) as core.matmul(b_transposed) wants; the
dequant params are the interleaved [dz', ds] table quantized_matmul_activation
streams, and each input quant reads an interleaved [scale, zp] table.

Kept separate from reference.py for the same TVMScript/F821 reasons as naive_impl.
"""

# flake8: noqa: F821  -- A.Buffer dtype-string annotations look like F821 names

import os
from typing import TYPE_CHECKING

import torch

from ...export import export_linear_quant_params, export_lut_ranges, export_sdpa_quant_params
from ...layout import fit_lut, pack

# Carry the inter-block residual stream -- the block's hidden_states input/output AND the in-block
# res1 accumulator -- in fp16 instead of fp32. Halves the block's DDR I/O floor + the res1 read/write
# passes (-771us/block @518 c4, measured; kNN device-vs-fp gap unchanged -- no accuracy regression,
# no fp16 overflow over 12 blocks). This is the ACCUMULATOR (unlike the o6/d6 _resid_dtype, already
# fp16); embed output + final_norm input match dtype so the graph threads consistently. Landed ON by
# default; set DINO_HIDDEN_FP16=0 to fall back to the fp32 residual stream.
_HIDDEN_FP16 = os.environ.get("DINO_HIDDEN_FP16", "1") == "1"
_HIDDEN_DTYPE = "float16x64" if _HIDDEN_FP16 else "float32x64"

# Carry the per-row IFD1 quant PARAMS (dequant/quant scales, LUT coeffs, LayerNorm affine) in fp16
# instead of fp32.  IFD1 delivers 1 byte/cycle, so an fp32x64 param pop costs 4 cycles; fp16 halves
# that -- directly cutting the DPU unroll's stream-pop tax on the param-heavy epilogues (measured on
# the qma proxy: identical schedule, stall 10->2 cyc/iter, -41us).  DINO_QPARAM_FP16=1 to enable.
_QPARAM_FP16 = os.environ.get("DINO_QPARAM_FP16", "0") == "1"
_QP = "float16x64" if _QPARAM_FP16 else "float32x64"

if TYPE_CHECKING:
    from .reference import DinoBlock, DinoBlockConfig


def _mshard_mlp_residual(
    f2,
    fc1_w,
    fc1_dq,
    gelu_lut,
    fc2_w,
    fc2_dq,
    res1,
    *,
    seq,
    hidden_pw,
    cores,
    resid_dtype,
    out_dtype,
):
    """EXPERIMENT: M-shard the WHOLE MLP (fc1+GELU+fc2+residual) over `cores` seq row-bands.

    Each core owns a contiguous band of rows and computes fc1 (dequant+bias+GELU+fc2-input quant,
    fused) -> fc2 -> banded residual add for that band only, all in one AICore datapath. Because the
    band's fc1 output is the whole intermediate row (full N), fc2 reads its own band's full-K
    activation -- no split-matrix (matrix_a_is_split) handling. Both weights are staged into shared
    L2 once (each core then reads the FULL fc1_w AND fc2_w from L2). Requires seq % cores == 0."""
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol import atomic
    from axelera.compiler.alto.axol.common import BufferInfo
    from tvm import tir

    from ...ops.quantized_matmul_activation import (
        _b_l2_stage_max_bytes,
        slice_n,
        quantized_matmul_activation,
    )

    core_ids = tuple(range(cores))
    band = seq // cores
    out = A.alloc_buffer((1, seq, hidden_pw), dtype=out_dtype, scope=A.DDR)
    out_bs = tir.BufferSlice(out)

    def _stage(w):
        wb = int(w.shape[-2]) * int(w.shape[-1]) * 64  # int8x64: 64 B/elem
        if wb <= _b_l2_stage_max_bytes():
            return A.alloc_buffer(tuple(int(s) for s in w.shape), dtype=w.dtype, scope=A.L2)
        return None

    fc1_l2 = _stage(fc1_w)
    fc2_l2 = _stage(fc2_w)

    for idx, core in enumerate(core_ids):
        lo, hi = idx * band, (idx + 1) * band
        with A.AICore(core=core):
            A.barrier(*core_ids)
            w1, w2 = fc1_w, fc2_w
            if idx == 0:
                if fc1_l2 is not None:
                    atomic.copy(fc1_w, scope=A.L2, output=fc1_l2)
                if fc2_l2 is not None:
                    atomic.copy(fc2_w, scope=A.L2, output=fc2_l2)
            A.barrier(*core_ids)  # order the single DDR->L2 stage before any core reads it
            if fc1_l2 is not None:
                w1 = fc1_l2
            if fc2_l2 is not None:
                w2 = fc2_l2
            f2_band = slice_n(f2, 1, lo, hi)  # slice the seq/M axis
            d2_band = quantized_matmul_activation(
                f2_band,
                w1,
                fc1_dq,
                luts=gelu_lut,
                fuse_output_quant=True,
                output=BufferInfo(dtype="int8x64"),
                core=core,
                stage_weight_l2=False,
            )
            d6_band = quantized_matmul_activation(
                d2_band,
                w2,
                fc2_dq,
                output=BufferInfo(dtype=resid_dtype),
                core=core,
                stage_weight_l2=False,
            )
            axol.atomic.add(slice_n(res1, 1, lo, hi), d6_band, output=slice_n(out_bs, 1, lo, hi))
            A.barrier(*core_ids)
    return out


def _mlp_out(
    f2,
    fc1_w,
    fc1_dq,
    gelu_lut,
    fc2_w,
    fc2_dq,
    res1,
    *,
    mlp_mshard,
    mlp_cores,
    fc_cores,
    fc1_split,
    fc2_ms_cores,
    seq,
    hidden_pw,
    resid_dtype,
    out_dtype,
):
    """Emit the MLP sub-block (fc1+GELU -> fc2 -> residual add) and return its output buffer.

    Dispatched in plain Python (runs during tracing) so the untaken branch is NOT parsed as
    TVMScript. mlp_mshard=True -> whole-MLP M-shard (fc2 reads its own band, no split). Else the
    N-shard fc1 (contiguous split output) + M-shard fc2 (matrix_a_is_split) path."""
    from axelera.compiler.alto.axol.common import BufferInfo

    from ...ops.mshard_matmul_residual import mshard_matmul_residual
    from ...ops.quantized_matmul_activation import quantized_matmul_activation_multicore

    if mlp_mshard:
        return _mshard_mlp_residual(
            f2,
            fc1_w,
            fc1_dq,
            gelu_lut,
            fc2_w,
            fc2_dq,
            res1,
            seq=seq,
            hidden_pw=hidden_pw,
            cores=mlp_cores,
            resid_dtype=resid_dtype,
            out_dtype=out_dtype,
        )
    d2 = quantized_matmul_activation_multicore(
        f2,
        fc1_w,
        fc1_dq,
        cores=fc_cores,
        luts=gelu_lut,
        fuse_output_quant=True,
        split_output=fc1_split,
        output=BufferInfo(dtype="int8x64"),
    )
    return mshard_matmul_residual(
        d2,
        fc2_w,
        fc2_dq,
        res1,
        seq=seq,
        n_blocks=hidden_pw,
        cores=(fc2_ms_cores or 1),
        matrix_a_is_split=fc1_split,
        resid_dtype=resid_dtype,
        out_dtype=out_dtype,
    )


def _fc1_shard_cores(inter_pw: int, cores: int) -> int:
    """How many cores to N-shard fc1 across: the most that leave a power-of-two N band.

    The band each core owns is one IMC tile, so a band that is not a power of two sub-tiles and
    gives the shard back (inter_pw=48 across 4 cores is 12, which sub-tiles into 2x6, while across
    3 cores it is 16, one clean tile). The count is therefore read off the weight rather than
    fixed: inter_pw=24 and 48 pick 3, inter_pw=64 picks 4. Returns 1 when no count qualifies,
    which selects the plain single-core MLP path.
    """
    for candidate in range(cores, 1, -1):
        band, rem = divmod(inter_pw, candidate)
        if rem == 0 and band & (band - 1) == 0:
            return candidate
    return 1


def make_optimized_block_kernel(
    config: "DinoBlockConfig", seq_len: int | None = None, cores: int = 1
):
    """Build the fused IRModule for one DINOv2-style encoder block.

    cores>1 head-shards SDPA across AI cores (scaled_dot_product_attention(cores=)); all
    matmuls stay single-core because N-sharding them was measured to be a net loss for
    these encoder shapes (see llm/tests/_diag_dino_matmul_perf.py).

    Every width the buffers are declared at is the config's DEVICE width, which is the
    checkpoint's own wherever it is already a PWORD multiple. A tower whose head_dim or
    intermediate is not (the SigLIP2 So400m encoder: 72 and 4304) is laid out at the
    padded widths instead, and ``optimized_block_buffers`` fills the pad with zeros: a
    padded output channel is written zero (dequant scale 0) and a padded input channel is
    contracted against a zero weight column, so no pad lane can reach a real output. The
    softmax normalisation stays 1/sqrt(REAL head_dim) -- the pad adds nothing to a score
    and must not enter its scaling either.
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol.utils.lut_chain import ChainedLUTMethod
    import numpy as np

    from ...ops.layer_norm_dpu_reduce import layer_norm_dpu_reduce_mshard
    from ...ops.mshard_matmul_residual import mshard_matmul_residual
    from ...ops.parallel_matmul_group import parallel_matmul_group

    hidden = config.hidden_size
    num_heads = config.num_heads
    head_dim = config.head_dim  # the REAL one: what the softmax normalises by
    eps = float(config.layernorm_eps)

    # LUT chain lengths, from the config rather than this module's DINO_* defaults: they
    # are the shapes of the LUT buffers below AND what the torch reference models, and a
    # tower that needs a finer chain must be able to say so without moving DINOv2's.
    sdpa_n_luts = config.sdpa_n_luts
    gelu_n_luts = config.gelu_n_luts
    norm_n_luts = config.norm_n_luts

    # Device widths (== the checkpoint's when they are PWORD multiples already).
    qkv_dim = config.qkv_dim  # heads * padded_head_dim: q/k/v out, o_proj in
    inter = config.padded_intermediate
    hidden_pw = hidden // 64
    qkv_pw = qkv_dim // 64
    head_dim_pw = config.head_dim_pw
    inter_pw = inter // 64
    seq = int(seq_len if seq_len is not None else config.seq_len)
    ctx_blocks = seq // 64

    # Multicore: SDPA is head-sharded across all `cores`. fc1 (the biggest matmul) is ALSO
    # N-sharded, across the core count DERIVED from the weight's block count by
    # `_fc1_shard_cores` (inter_pw=48 picks 3, inter_pw=64 picks 4). fc1 emits a
    # CONTIGUOUS split output (split_output=True) and fc2 consumes it via matrix_a_is_split, so
    # there is no strided per-row output DMA (the ~1.7x N-shard win) and no re-assembly. The other
    # matmuls (q/k/v/o) stay single-core: they were measured a net loss to N-shard for these shapes
    # (see llm/tests/_diag_dino_matmul_perf.py). fc_cores==1 keeps the plain single-core MLP path.
    _mc = bool(cores and cores > 1)
    _sdpa_kw = {"cores": cores} if _mc else {}
    fc_cores = _fc1_shard_cores(inter_pw, int(cores)) if _mc else 1
    _fc1_split = fc_cores > 1
    _fc2_split_kw = {"matrix_a_is_split": True} if _fc1_split else {}
    # q/k/v are three independent matmuls over the same ln1 output -- run one whole projection per
    # core (cores 0/1/2) concurrently when we have >=3 cores. Sharding a single projection is flat
    # (fixed-overhead bound); op-level concurrency parallelises the group (~617us->~11us of matmul
    # work @518). See ops/parallel_matmul_group.py and llm/tests/_diag_dino_qkv_parallel.py.
    _qkv_cores = (0, 1, 2) if (_mc and int(cores) >= 3) else None
    # LayerNorm reduces over D per row -> rows are independent, weight is tiny (no replication
    # penalty), so M-shard ln1/ln2 across ALL cores (embarrassingly parallel, contiguous row-band
    # output). See ops/layer_norm_dpu_reduce.py:layer_norm_dpu_reduce_mshard.
    _ln_cores = tuple(range(int(cores))) if _mc else None
    # fc2 / o_proj M-shard: split the seq rows across all cores + row-partitioned residual add, weight
    # staged to shared L2. Requires seq % cores == 0. See ops/mshard_matmul_residual.py. None -> single-core
    # matmul + separate residual add. (o_proj's matmul is fixed-overhead-bound like q/k/v so its M-shard
    # win, if any, is the BANDED residual-add fusion, not the matmul.)
    _fc2_ms_cores = int(cores) if (_mc and (seq % int(cores) == 0)) else None
    _oproj_ms_cores = _fc2_ms_cores
    # Whole-MLP M-shard: shard fc1 too (over the same seq row-bands as fc2), fusing
    # fc1+GELU+fc2+residual into one per-core datapath. Each core owns a contiguous row-band and
    # computes the WHOLE MLP for it (full N/K), so fc2 reads its own band's full-K activation --
    # the split-matrix (matrix_a_is_split) plumbing disappears. Cost: each core reads the FULL fc1
    # and fc2 weights from shared L2 (vs 1/4 each in the N-shard), staged once. So it's a WIN only
    # once each core's row-band has enough compute to amortize that full-weight read: measured
    # +1.2% at seq=1408 (@518) but -4.7% at seq=320 (@224, weight-BW-bound). Gate on a large seq;
    # DINO_MLP_MSHARD=1 forces on, =0 forces off (overrides the gate). Requires seq%cores==0.
    _MLP_MSHARD_MIN_SEQ = 1024
    _mlp_env = os.environ.get("DINO_MLP_MSHARD")  # None -> auto (seq-gated)
    _mlp_mshard = (
        _mc
        and (seq % int(cores) == 0)
        and (seq >= _MLP_MSHARD_MIN_SEQ if _mlp_env is None else _mlp_env == "1")
    )
    # o_proj/fc2 outputs (o6/d6, and thus the o7/d7 LayerScale products) written as fp16 instead of
    # fp32: at seq 1408 these are 4.3MB buffers, and halving them saves the matmul-write + mul + add
    # passes over them (-444us/block @518 c4, measured). res1/output/hidden stay fp32 so the residual
    # ACCUMULATOR (incl. the inter-block hidden) keeps full precision -- block cosine 1.0000, exact.
    _resid_dtype = "float16x64"

    @A.ir_module
    class DinoBlockFused:
        @A.prim_func
        def main(
            hidden_states: A.Buffer((1, seq, hidden_pw), _HIDDEN_DTYPE, scope=A.DDR),
            ln1_w: A.Buffer((hidden_pw,), "float32x64", scope=A.DDR),  # D-major (dpu-reduce norm)
            ln1_lut: A.Buffer((norm_n_luts,), "float32x64", scope=A.DDR),
            # q/k/v share one interleaved [scale, zp] input-quant table (one SmoothQuant group).
            qkv_qparams: A.Buffer((2 * hidden_pw,), _QP, scope=A.DDR),
            # NK weights (N_BLOCKS, K) + 4-wide dequant/output-quant table:
            # [dz', ds, q_attn_scale, 0] per block. The projection's output int8-quantize
            # (the SDPA input quant) is FUSED into the matmul write (fuse_output_quant), so
            # q/k/v come out int8 and feed SDPA directly -- no separate quantize pass, no fp16
            # intermediate, and the head transpose then moves int8 (4x less than fp16).
            # Their width is qkv_dim: hidden when head_dim is a PWORD, heads*padded_head_dim
            # when it is not (the padded rows have dequant scale 0, so they write zeros).
            q_proj_w: A.Buffer((qkv_pw, hidden), "int8x64", scope=A.DDR),
            q_proj_dq: A.Buffer((1, 1, 4 * qkv_pw), _QP, scope=A.DDR),
            k_proj_w: A.Buffer((qkv_pw, hidden), "int8x64", scope=A.DDR),
            k_proj_dq: A.Buffer((1, 1, 4 * qkv_pw), _QP, scope=A.DDR),
            v_proj_w: A.Buffer((qkv_pw, hidden), "int8x64", scope=A.DDR),
            v_proj_dq: A.Buffer((1, 1, 4 * qkv_pw), _QP, scope=A.DDR),
            sdpa_exp_lut: A.Buffer((sdpa_n_luts,), "float32x64", scope=A.DDR),
            sdpa_inv_lut: A.Buffer((sdpa_n_luts,), "float32x64", scope=A.DDR),
            sdpa_scores_dq_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            sdpa_weights_q_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            sdpa_output_dq_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            # int8, 1-head broadcast padding mask ({0, -127}): the SDPA op broadcasts it
            # over all query heads (leading dims all 1 -> L1-hoisted, read once not per-head)
            # and uses -127 as the exp-negligible masked fill. ~48x smaller than per-head fp32.
            attn_mask: A.Buffer((1, 1, seq, ctx_blocks), "int8x64", scope=A.DDR),
            # o_proj contracts over the (padded) head layout: qkv_dim in, hidden out.
            o_proj_qparams: A.Buffer((2 * qkv_pw,), _QP, scope=A.DDR),
            o_proj_w: A.Buffer((hidden_pw, qkv_dim), "int8x64", scope=A.DDR),
            o_proj_dq: A.Buffer((1, 1, 2 * hidden_pw), _QP, scope=A.DDR),
            ln2_w: A.Buffer((hidden_pw,), "float32x64", scope=A.DDR),  # D-major (dpu-reduce norm)
            ln2_lut: A.Buffer((norm_n_luts,), "float32x64", scope=A.DDR),
            fc1_qparams: A.Buffer((2 * hidden_pw,), _QP, scope=A.DDR),
            fc1_w: A.Buffer((inter_pw, hidden), "int8x64", scope=A.DDR),
            # 4-wide [dz', ds, fc2_in_scale, fc2_in_zp]: fc1's dequant+GELU AND fc2's input
            # quantize fused into one write (fuse_output_quant), so fc1 emits int8 straight to
            # fc2 -- drops the separate quantize pass and the ~2 MiB fp16 h1 materialization.
            fc1_dq: A.Buffer((1, 1, 4 * inter_pw), _QP, scope=A.DDR),
            gelu_lut: A.Buffer((gelu_n_luts,), "float32x64", scope=A.DDR),
            fc2_w: A.Buffer((hidden_pw, inter), "int8x64", scope=A.DDR),
            fc2_dq: A.Buffer((1, 1, 2 * hidden_pw), _QP, scope=A.DDR),
        ):  # noqa: F821
            eps_const = A.alloc_constant([eps] * 64, dtype="float32x64", scope=A.DDR)
            sdpa_w_zp = A.alloc_constant(
                np.array([0] * 64).reshape((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR
            )
            sdpa_w_qmin = A.alloc_constant(
                np.array([-128] * 64).reshape((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR
            )
            sdpa_w_qmax = A.alloc_constant(
                np.array([127] * 64).reshape((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR
            )
            sdpa_q_zp = A.alloc_constant(
                np.array([0] * 64).reshape((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR
            )
            sdpa_v_zp = A.alloc_constant(
                np.array([0] * 64).reshape((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR
            )

            # --- Attention sub-block ---
            # Fused LayerNorm 1 + q/k/v input quantize in ONE single-pass DPU-reduce datapath
            # (int8 out): replaces the two-pass Kahan norm + a separate llm.quantize (~3.5x
            # faster standalone, and the quantize is free). q/k/v share the int8 result (one
            # SmoothQuant group). ln1_w/ln1_b are D-major packed; qkv_qparams is the fused quant.
            # NB: ln1 bias is FOLDED into qkv_qparams' zero-point host-side (qz' = bias*qs + qz), so
            # NO bias_buffer here -- a bias stream + the quant stream = two per-row IFD1 streams,
            # which corrupt on device (interpret/sim don't catch it). See optimized_block_buffers.
            qkv_q = layer_norm_dpu_reduce_mshard(
                hidden_states,
                ln1_lut,
                eps_const[0:1],
                normalized_shape=(hidden,),
                cores=_ln_cores,
                weight_buffer=ln1_w,
                quant_params=qkv_qparams,
                output_dtype="int8x64",
            )

            # Projection + dequant + SDPA-input int8 quantize fused (fuse_output_quant): q/k/v
            # come out int8 in (1, seq, hidden_pw); reshape to heads and transpose (now on int8,
            # 4x less traffic) to the (1, num_heads, seq, head_dim_pw) SDPA input. The attention
            # input quant is per-head-dim (uniform across heads, zp=0), so quantizing before vs
            # after the transpose is identical -- the scale is folded into q/k/v_proj_dq.
            # q/k/v projections (fuse_output_quant int8) + head transpose to
            # (1, num_heads, seq, head_dim_pw): run one WHOLE projection+transpose per core (0/1/2)
            # concurrently when _qkv_cores is set, else sequentially on the default core. All three
            # share qkv_q; the transpose is folded into each core's region so it parallelises too.
            q8, k8, v8 = parallel_matmul_group(
                qkv_q,
                [q_proj_w, k_proj_w, v_proj_w],
                [q_proj_dq, k_proj_dq, v_proj_dq],
                cores=_qkv_cores,
                fuse_output_quant=True,
                output_dtype="int8x64",
                reshape_shape=(1, seq, num_heads, head_dim_pw),
                transpose_dims=(0, 2, 1, 3),
            )

            attn = axol.core.scaled_dot_product_attention(
                q8,
                k8,
                v8,
                sdpa_scores_dq_scale,
                sdpa_exp_lut,
                sdpa_inv_lut,
                sdpa_weights_q_scale,
                sdpa_w_zp,
                sdpa_w_qmin,
                sdpa_w_qmax,
                sdpa_output_dq_scale,
                sdpa_q_zp,
                sdpa_v_zp,
                attn_mask,
                is_causal=False,
                enable_gqa=False,
                scale=1.0 / (head_dim**0.5),
                neg_exp_method=ChainedLUTMethod.ADDITIVE,
                inv_method=ChainedLUTMethod.ADDITIVE,
                # Fuse the o_proj INPUT quantize into SDPA's P@V write: SDPA emits int8 directly
                # (o_proj_qparams is per-hidden-block = head-major, exactly what the quantizer
                # indexes), so the head transpose then moves int8 (4x less) and the separate
                # dpu_cast + o3 quantize are gone.
                attention_output_quantizer=o_proj_qparams,
                **_sdpa_kw,  # cores=cores head-shards SDPA when multicore
            )
            attn_t = axol.core.transpose(attn, dims=(0, 2, 1, 3))  # int8
            o3 = axol.atomic.reshape(attn_t, shape=(1, seq, qkv_pw))  # int8 o_proj input
            # o_proj (LayerScale 1 folded into o_proj_dq -- o6 is the LayerScaled attention output)
            # + residual add res1 = hidden + o6. M-shard across cores when enabled: each core does its
            # row-band's o_proj + banded residual add (input o3 is normal -> matrix_a_is_split=False).
            res1 = mshard_matmul_residual(
                o3,
                o_proj_w,
                o_proj_dq,
                hidden_states,
                seq=seq,
                n_blocks=hidden_pw,
                cores=(_oproj_ms_cores or 1),
                matrix_a_is_split=False,
                resid_dtype=_resid_dtype,
                out_dtype=_HIDDEN_DTYPE,
            )

            # --- MLP sub-block ---
            # Fused LayerNorm 2 + fc1 input quantize (int8 out) in one DPU-reduce datapath.
            # ln2 bias folded into fc1_qparams' zero-point (see ln1 note); no bias_buffer.
            f2 = layer_norm_dpu_reduce_mshard(
                res1,
                ln2_lut,
                eps_const[0:1],
                normalized_shape=(hidden,),
                cores=_ln_cores,
                weight_buffer=ln2_w,
                quant_params=fc1_qparams,
                output_dtype="int8x64",
            )
            # fc1 matmul + dequant + bias + GELU + fc2-input int8 quantize (fuse_output_quant) ->
            # fc2 (LayerScale 2 folded into fc2_dq's scale) + residual add. _mlp_out dispatches in
            # plain Python (not parsed as TVMScript) between two layouts: (a) N-shard fc1 (contiguous
            # split output) + M-shard fc2 via matrix_a_is_split, or (b) DINO_MLP_MSHARD whole-MLP
            # M-shard where each core owns a seq row-band end-to-end (fc2 reads its own band, no
            # split). Both fuse the residual add banded into the per-core tail.
            A.output(
                _mlp_out(
                    f2,
                    fc1_w,
                    fc1_dq,
                    gelu_lut,
                    fc2_w,
                    fc2_dq,
                    res1,
                    mlp_mshard=_mlp_mshard,
                    mlp_cores=int(cores),
                    fc_cores=fc_cores,
                    fc1_split=_fc1_split,
                    fc2_ms_cores=_fc2_ms_cores,
                    seq=seq,
                    hidden_pw=hidden_pw,
                    resid_dtype=_resid_dtype,
                    out_dtype=_HIDDEN_DTYPE,
                )
            )

    return DinoBlockFused


def compile_block(
    config: "DinoBlockConfig",
    out_dir,
    *,
    target=None,
    seq_len: int | None = None,
    make_kernel=None,
    enable_buffer_promotion: bool = False,
    cores: int = 1,
    dpu_unroll: bool = False,
):
    """Compile a block kernel for *target*. Returns (mod, model_json).

    make_kernel defaults to the fused make_optimized_block_kernel. target defaults to
    AICORE_SIM (the functional simulator); pass Target.DEVICE for the Metis silicon.
    enable_buffer_promotion stages the per-op workspace off DDR into L2 (a device perf
    lever). cores>1 compiles the multicore block (MLP N-shard + SDPA head-shard) and
    configures that many AI cores. The runtime reads the target back from the emitted
    alto_config.json, so the same run() drives either.
    """
    from pathlib import Path

    from axelera.compiler.alto.compiler.config import AltoConfig, HardwareGeneration, Target
    from axelera.compiler.alto.compiler.driver import compile as alto_compile

    if target is None:
        target = Target.AICORE_SIM
    if make_kernel is None:
        make_kernel = make_optimized_block_kernel
    out_dir = Path(out_dir)
    mod = make_kernel(config, seq_len, cores)
    alto_config = AltoConfig(
        output_dir=out_dir,
        generation=HardwareGeneration.OMEGA,
        target=target,
        num_cores=cores,  # AI cores the system is configured for (sim is single-core)
        enable_tiling_injection=True,
        enable_buffer_promotion=enable_buffer_promotion,
        enable_llm_support=True,
        assign_unique_input_pools=True,
        unpack_axm_archive=True,
        clean_output_dir=True,
        emit_atex_artifacts=True,
        emit_alto_config=True,
        dpu_unroll=dpu_unroll,
    )
    alto_compile(mod, alto_config)
    model_jsons = sorted(out_dir.rglob("model.json"))
    if not model_jsons:
        raise RuntimeError(f"no model.json produced under {out_dir}")
    return mod, model_jsons[0]


# ---------------------------------------------------------------------------
# Zero padding, for a tower whose widths are not PWORD multiples
#
# Every helper here is the IDENTITY when the checkpoint's widths already are, so an
# aligned tower (dino itself, the SigLIP2 Base-86M encoder) never meets the padding.
# What makes a padded layout exact, whatever the pad lanes end up holding, is a pair of
# invariants the binder maintains with them:
#
#   * every padded OUTPUT channel is written zero -- its dequant scale is zero, and
#   * every padded INPUT channel is contracted against a zero WEIGHT column.
# ---------------------------------------------------------------------------


def _head_pad_index(heads: int, head_dim: int, padded_head_dim: int) -> torch.Tensor:
    """Where each real head-dim channel lands in the padded, head-major layout.

    Head h's real channels come first in its stride and the pad fills the tail. (The
    Qwen3-VL tower splits its pad per RoPE half instead, so each half starts on a block
    boundary; nothing in THIS block is position-dependent, so the tail is enough.)
    Returns a flat (heads*head_dim,) index into the heads*padded_head_dim axis.
    """
    within = torch.arange(head_dim)
    return (torch.arange(heads).reshape(-1, 1) * padded_head_dim + within).reshape(-1)


def _scatter_axis(t: torch.Tensor, axis: int, index: torch.Tensor, width: int) -> torch.Tensor:
    """Scatter `t` along `axis` into a zero buffer `width` wide (identity if same width)."""
    if t.shape[axis] == width:
        return t
    shape = list(t.shape)
    shape[axis] = width
    return torch.zeros(shape, dtype=t.dtype).index_copy_(axis, index, t)


def _pad_axis(t: torch.Tensor, axis: int, width: int) -> torch.Tensor:
    """Zero-pad the tail of `axis` out to `width` (identity if already that wide)."""
    return _scatter_axis(t, axis, torch.arange(t.shape[axis]), width)


def _pad_channels(values, width: int, index: torch.Tensor | None = None) -> torch.Tensor:
    """A per-channel vector widened to a device width, the pad left at ZERO.

    A per-tensor scalar is broadcast over the REAL channels first, so it cannot leak into
    the pad -- a zero dequant scale is exactly what pins a padded output to zero.
    """
    t = torch.as_tensor(values).to(torch.float32).reshape(-1)
    if t.numel() == width:
        return t
    if t.numel() == 1:
        t = t.expand(width if index is None else index.numel()).contiguous()
    return _pad_axis(t, 0, width) if index is None else _scatter_axis(t, 0, index, width)


# Kernel weight/dq buffer prefix -> torch submodule leaf name (identical here).
_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2")
# dq buffer -> the module attribute holding the float bias vector (folded into dz').
_PROJ_BIAS = {
    "q_proj": "q_bias",
    "k_proj": "k_bias",
    "v_proj": "v_bias",
    "o_proj": "o_bias",
    "fc1": "fc1_bias",
    "fc2": "fc2_bias",
}
_SDPA_SCALARS = {
    "sdpa_scores_dq_scale": "scores_dq_scale",
    "sdpa_weights_q_scale": "qs_w",
    "sdpa_output_dq_scale": "output_dq_scale",
}
_NORM_W = {"ln1_w": "ln1.weight", "ln2_w": "ln2.weight"}
_NORM_B = {"ln1_b": "ln1.bias", "ln2_b": "ln2.bias"}
# LayerScale (per-output-channel) folds into the following projection's dequant SCALE ds -- ls1 after
# o_proj, ls2 after fc2 -- since (matmul-dz)*ds*ls == LayerScale(deq). Removes the two elementwise
# muls; no separate ls1/ls2 buffers. (q/k/v/fc1 have no LayerScale.)
_PROJ_LAYERSCALE = {"o_proj": "ls1", "fc2": "ls2"}
_LUT_SITES = {
    "ln1_lut": "ln1.inv_sqrt_range",
    "ln2_lut": "ln2.inv_sqrt_range",
    "gelu_lut": "gelu_range",
    "sdpa_exp_lut": "sdpa.exp_range",
    "sdpa_inv_lut": "sdpa.reciprocal_range",
}
# Interleaved input-quant [scale, zp] buffer -> the linear providing qs/qz. q/k/v share
# one quantize (qkv_qparams), so all three read q_proj's params.
_INPUT_QP = {
    "qkv_qparams": "q_proj",
    "o_proj_qparams": "o_proj",
    "fc1_qparams": "fc1",
    # fc2's input quant is fused into fc1's write (fuse_output_quant), so no fc2_qparams input.
}


def optimized_block_buffers(block: "DinoBlock", spec: dict) -> dict:
    """Bind the fused kernel's buffers: NK weights, bias-folded interleaved dequant
    tables, interleaved input-quant tables, norms, LayerScales, and LUTs.

    The device widths come off the kernel's own spec, so a tower whose head_dim or MLP
    width is not a PWORD multiple is bound at the padded layout, with zeros in the pad
    (see the padding helpers above). ``block`` always holds the checkpoint's own widths.
    """
    lin = export_linear_quant_params(block)
    sdpa = export_sdpa_quant_params(block)["sdpa"]
    rng = export_lut_ranges(block)
    buffers: dict = {}

    def put(name, value):
        if name in spec:
            buffers[name] = pack(value, spec[name])

    def nk(w):  # [out, in] int8 -> NK (out//64, in, 64)
        out, inn = w.shape
        return w.reshape(out // 64, 64, inn).permute(0, 2, 1).contiguous().to(torch.int8)

    def blocks(v):  # (C,) -> (C//64, 64)
        return torch.as_tensor(v).to(torch.float32).reshape(-1, 64).contiguous()

    def interleave(scale, zp, c_pw):  # -> (2*c_pw, 64): [scale0, zp0, scale1, zp1, ...]
        qp = torch.empty((2 * c_pw, 64), dtype=torch.float32)
        qp[0::2] = blocks(scale)
        qp[1::2] = blocks(zp)
        return qp

    def interleave4(dz, ds, qs, qz, c_pw):  # -> (4*c_pw, 64): [dz, ds, q_scale, q_zp] per block
        qp = torch.empty((4 * c_pw, 64), dtype=torch.float32)
        qp[0::4], qp[1::4], qp[2::4], qp[3::4] = blocks(dz), blocks(ds), blocks(qs), blocks(qz)
        return qp

    # The downstream int8-quantize is fused into each projection/fc1 matmul write
    # (fuse_output_quant), so the dequant table is 4-wide [dz', ds, q_scale, q_zp]:
    #   q/k/v_proj -> the SDPA-input quant (per-head-dim scale, zero-pinned), tiled over the
    #     projection's output blocks (uniform across heads);
    #   fc1        -> fc2's per-channel input quant ([scale, zp] from lin["fc2"]).
    num_heads = int(block.config.num_heads)
    head_dim = int(block.config.head_dim)
    hidden = int(block.config.hidden_size)

    # --- device widths, and the zero padding that reaches them --------------------------
    # Taken off the KERNEL's own spec, so this binds whatever geometry the kernel was built
    # for: the checkpoint's own widths on an aligned tower (dino, the 450M SigLIP2 Base
    # encoder), the padded ones on a So400m encoder (head_dim 72 -> a 128-lane head stride,
    # intermediate 4304 -> 4352). A partial spec -- the unit tests bind one buffer at a
    # time -- falls back to the checkpoint's, which is the same thing when nothing pads.
    def _device_width(spec_name: str, proj: str) -> int:
        if spec_name in spec:
            return int(spec[spec_name][0][0]) * 64
        return int(lin[proj]["w"].shape[0])

    qkv_dim = _device_width("q_proj_w", "q_proj")  # heads * padded head stride
    inter = _device_width("fc1_w", "fc1")
    head_index = _head_pad_index(num_heads, head_dim, qkv_dim // num_heads)

    # Per-output-channel device width of each projection, and (head-major ones) its scatter.
    _OUT_WIDTH = {
        "q_proj": qkv_dim,
        "k_proj": qkv_dim,
        "v_proj": qkv_dim,
        "o_proj": hidden,
        "fc1": inter,
        "fc2": hidden,
    }
    _OUT_INDEX = {"q_proj": head_index, "k_proj": head_index, "v_proj": head_index}

    def pad_weight(proj, w):
        """[out, in] -> the device's [out, in], zeros in every padded row/column."""
        if proj in _OUT_INDEX:  # q/k/v scatter their OUTPUT rows into the head stride
            return _scatter_axis(w, 0, head_index, qkv_dim)
        if proj == "o_proj":  # ...and o_proj/fc2 zero the matching INPUT columns
            return _scatter_axis(w, 1, head_index, qkv_dim)
        if proj == "fc1":
            return _pad_axis(w, 0, inter)
        if proj == "fc2":
            return _pad_axis(w, 1, inter)
        return w

    def _attn_scale(field):
        """An SDPA input scale as a per-output-channel vector over the padded heads.

        It is per head-dim and uniform across heads, so it tiles over them; a per-tensor
        scale (v) is broadcast over the head first.
        """
        s = torch.as_tensor(sdpa[field]).to(torch.float32).reshape(-1)
        if s.numel() == 1:
            s = s.expand(head_dim).contiguous()
        return _pad_channels(s.repeat(num_heads), qkv_dim, head_index)

    _FUSED_OQ = {"q_proj": "qs_q", "k_proj": "qs_k", "v_proj": "qs_v"}

    def _output_quant(proj):
        """(q_scale, q_zp) folded into this proj's write, or None if it writes float."""
        if proj in _FUSED_OQ:
            return _attn_scale(_FUSED_OQ[proj]), torch.zeros(qkv_dim)
        if proj == "fc1":
            fq = lin["fc2"]  # fc1's output feeds fc2, so its output quant == fc2's input quant
            # The pad's quantized value is whatever this leaves there; fc2's zero weight
            # columns are what make that irrelevant.
            return _pad_channels(fq["qs"], inter), _pad_channels(fq["qz"], inter)
        return None  # o_proj / fc2 write float to the residual -- no fused output quant

    # NK weights + bias-folded dequant tables (4-wide where the output quant is fused).
    for proj in _PROJECTIONS:
        p = lin[proj]
        if f"{proj}_w" in spec:
            buffers[f"{proj}_w"] = nk(pad_weight(proj, p["w"]))
        name = f"{proj}_dq"
        if name in spec:
            width = _OUT_WIDTH[proj]
            opw = width // 64  # output blocks, padded
            ds = p["ds"].to(torch.float32)
            bias = getattr(block, _PROJ_BIAS[proj]).detach().to(torch.float32)
            dz_folded = p["dz"].to(torch.float32) - bias / ds  # (matmul - dz')*ds = deq + bias
            # Fold LayerScale into ds (o_proj->ls1, fc2->ls2): (matmul-dz')*(ds*ls) = (deq+bias)*ls
            # = LayerScale(output). dz_folded keeps the ORIGINAL ds (bias/ds), so the bias fold is
            # unaffected. Eliminates the downstream elementwise mul.
            if proj in _PROJ_LAYERSCALE:
                ls = getattr(block, _PROJ_LAYERSCALE[proj]).detach().to(torch.float32).reshape(-1)
                ds = ds * ls
            # Widened only now: the fold divides by ds, so a padded (zero) ds must not
            # exist while it happens.
            index = _OUT_INDEX.get(proj)
            dz_folded, ds = (_pad_channels(v, width, index) for v in (dz_folded, ds))
            oq = _output_quant(proj)
            if oq is not None:
                qs, qz = oq
                buffers[name] = (
                    interleave4(dz_folded, ds, qs, qz, opw).reshape(1, 1, 4 * opw, 64).contiguous()
                )
            else:
                buffers[name] = (
                    interleave(dz_folded, ds, opw).reshape(1, 1, 2 * opw, 64).contiguous()
                )

    # Interleaved [scale, zp] input-quant tables. qkv_qparams / fc1_qparams feed the FUSED
    # layer_norm_dpu_reduce, so the LayerNorm bias is folded into their zero-point
    # (qz' = ln_bias*qs + qz) -- the norm then needs no bias stream (two per-row IFD1 streams
    # corrupt on device). o_proj_qparams (SDPA-output quant) has no norm bias, but IS indexed
    # per input channel over the padded head layout, so it takes the head scatter.
    # fc2_qparams is gone (folded into fc1's matmul write). Guarded by `buf in spec`.
    _QP_BIAS_FOLD = {"qkv_qparams": "ln1.bias", "fc1_qparams": "ln2.bias"}
    for buf, proj in _INPUT_QP.items():
        if buf in spec:
            p = lin[proj]
            qs = p["qs"].to(torch.float32).reshape(-1)
            qz = p["qz"].to(torch.float32).reshape(-1)
            if buf in _QP_BIAS_FOLD:
                bias = (
                    block.get_parameter(_QP_BIAS_FOLD[buf]).detach().to(torch.float32).reshape(-1)
                )
                qz = qz + bias * qs  # fold LayerNorm bias into the quant zero-point
            if proj == "o_proj":  # per-INPUT-channel: the (padded) head layout
                qs, qz = (_pad_channels(v, qkv_dim, head_index) for v in (qs, qz))
            buffers[buf] = interleave(qs, qz, qs.numel() // 64)

    for buf, field in _SDPA_SCALARS.items():
        put(buf, sdpa[field])
    # LayerNorm weight is D-major for layer_norm_dpu_reduce: pack reshapes the flat (hidden,)
    # parameter into the (hidden_pw, 64) buffer (block k lane j = param[k*64+j]), NOT
    # broadcast_to_pword. Bias is folded into the qparams above, so no ln1_b/ln2_b buffers.
    for buf, path in _NORM_W.items():
        put(buf, block.get_parameter(path).detach())
    # LayerScale (ls1/ls2) is folded into the o_proj/fc2 dequant scale above -- no separate buffers.
    for buf, site in _LUT_SITES.items():
        if buf in spec and site in rng:
            buffers[buf] = fit_lut(rng[site], spec[buf])
    # When DINO_QPARAM_FP16, the matmul dequant/quant-param buffers are declared fp16 (_QP) to
    # halve their per-row IFD1 pop cost -- so their host fills must be fp16 to match.  ONLY these
    # 9 (NOT the LUTs, which must stay fp32 -- fp16 LUT coeffs fault the device; nor norm/scale).
    if _QPARAM_FP16:
        _MM_QP = {
            "qkv_qparams",
            "q_proj_dq",
            "k_proj_dq",
            "v_proj_dq",
            "o_proj_qparams",
            "o_proj_dq",
            "fc1_qparams",
            "fc1_dq",
            "fc2_dq",
        }
        for b in _MM_QP & buffers.keys():
            buffers[b] = buffers[b].to(torch.float16)
    return buffers


# ---------------------------------------------------------------------------
# Patch-embedding kernel (graph kind "embed") — space-to-depth matmul + CLS + pos.
# ---------------------------------------------------------------------------
def make_embed_kernel(config: "DinoBlockConfig", seq_len: int | None = None, cores: int = 1):
    """Build the patch-embed IRModule as a space-to-depth matmul (NOT a padded Conv2d).

    The DINOv2 patch embed is a non-overlapping patch projection (stride == kernel == P), so it
    is equivalent to flattening each P*P*C patch into the channel dim and running one dense
    matmul. Doing that instead of the naive Conv2d avoids padding the 3 input channels out to a
    full PWORD (64): the old conv streamed a (1,H,W,1) int8x64 image (61/64 lanes zero) and
    reduced over P*P*64=12544, whereas here the host im2col's each patch into K=C*P*P (588,
    padded to a PWORD multiple -> 640), a ~14-20x cut in both input DMA and MVM reduction.

    Pipeline: host-im2col'd int8 pixels (1, num_patches, K_BLOCKS) -> quantized_matmul_activation
    (matmul + per-out-channel dequant, with the conv bias FOLDED into the dequant zero-point,
    dz' = dz - bias/ds; no LUT activation, fp32 out) -> concat CLS + zero pad -> add positional
    embedding. Output (1, seq, hidden_pw) fp32 feeds block 0. `cores` is ignored (single-core).
    Graph input `pixels` is (1, num_patches, K_BLOCKS) int8x64 (host space-to-depth + sym quant).
    """
    from math import ceil

    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol.common import BufferInfo
    import numpy as np

    from ...ops.quantized_matmul_activation import quantized_matmul_activation

    hidden = config.hidden_size
    hidden_pw = hidden // 64
    H = W = config.image_size
    P = config.patch_size
    C = config.num_channels
    gh, gw = H // P, W // P
    num_patches = gh * gw
    patch_dim = C * P * P  # flattened patch length (K); matches host im2col + weight K order
    k = ceil(patch_dim / 64) * 64  # K padded up to a PWORD multiple (the pad zeros contribute 0)
    k_pw = k // 64
    seq = int(seq_len if seq_len is not None else config.seq_len)
    n_pad = seq - num_patches - 1  # zero rows after CLS + patches

    @A.ir_module
    class DinoEmbed:
        @A.prim_func
        def main(
            pixels: A.Buffer((1, num_patches, k_pw), "int8x64", scope=A.DDR),
            patch_w: A.Buffer((hidden_pw, k), "int8x64", scope=A.DDR),  # NK (N_BLOCKS, K)
            # interleaved per-out-channel [dz', ds] (bias folded into dz'): (matmul - dz')*ds
            patch_dq: A.Buffer((1, 1, 2 * hidden_pw), "float32x64", scope=A.DDR),
            cls_const: A.Buffer((1, 1, hidden_pw), "float32x64", scope=A.DDR),
            pos_const: A.Buffer((1, seq, hidden_pw), "float32x64", scope=A.DDR),
        ):  # noqa: F821
            p2 = quantized_matmul_activation(
                pixels, patch_w, patch_dq, output=BufferInfo(dtype="float32x64")
            )  # (1, num_patches, hidden_pw) fp32 == patch conv dequantized + bias
            zpad = A.alloc_constant(np.zeros((1, n_pad, hidden_pw, 64)), "float32x64", scope=A.DDR)
            tok = axol.core.concat(cls_const, p2, zpad, dim=1)  # (1, seq, hidden_pw)
            # + positional embedding -> feeds block 0 (output dtype matches the block hidden stream)
            A.output(axol.atomic.add(tok, pos_const, output=BufferInfo(dtype=_HIDDEN_DTYPE)))

    return DinoEmbed


def embed_buffers(model, spec: dict) -> dict:
    """Pack the space-to-depth embed kernel's constants from a calibrated DinoV2Model: the patch
    weight in NK layout (int8, K = C*P*P padded to a PWORD multiple), the interleaved [dz', ds]
    dequant table with the conv bias folded into dz', the CLS token, and the padded
    positional-embedding constant."""
    from math import ceil

    import torch

    lin = export_linear_quant_params(model)
    p = lin["patch_embed.proj"]
    hidden = int(model.patch_embed.proj.out_features)
    hidden_pw = hidden // 64
    ps = int(model.patch_embed.patch_size)
    c = int(model.patch_embed.num_channels)
    patch_dim = c * ps * ps
    k = ceil(patch_dim / 64) * 64  # K padded up to a PWORD multiple (matches make_embed_kernel)

    def blocks(v):  # (C,) -> (C//64, 64)
        return torch.as_tensor(v).to(torch.float32).reshape(-1, 64).contiguous()

    # NK weight: w (hidden, C*P*P) int8, K flatten order (c, kh, kw) == host im2col. Pad K to a
    # PWORD multiple, then reshape to (hidden_pw, k, 64) as core.matmul(b_transposed) wants.
    w = p["w"].to(torch.int8)  # (hidden, patch_dim)
    wpad = torch.zeros(hidden, k, dtype=torch.int8)
    wpad[:, :patch_dim] = w
    nk = wpad.reshape(hidden_pw, 64, k).permute(0, 2, 1).contiguous()  # (hidden_pw, k, 64)

    # Interleaved [dz', ds] with the conv bias folded into the dequant zero-point:
    # (matmul - dz')*ds == matmul*ds + bias, since dz' = dz - bias/ds (dz == 0 for symmetric qz).
    ds = p["ds"].to(torch.float32)
    bias = model.patch_embed.bias.detach().to(torch.float32)
    dz_folded = p["dz"].to(torch.float32) - bias / ds
    dq = torch.empty((2 * hidden_pw, 64), dtype=torch.float32)
    dq[0::2], dq[1::2] = blocks(dz_folded), blocks(ds)

    buffers = {"patch_w": pack(nk, spec["patch_w"])}
    buffers["patch_dq"] = pack(dq.reshape(1, 1, 2 * hidden_pw, 64).contiguous(), spec["patch_dq"])
    buffers["cls_const"] = pack(model.cls_token.detach(), spec["cls_const"])
    # pos_embed (1, seq_real, hidden) -> pad rows to seq_pad with zeros (masked/pad positions).
    seq_pad = spec["pos_const"][0][1]
    pos = model.pos_embed.detach().to(torch.float32)
    padded = torch.zeros(1, seq_pad, hidden, dtype=torch.float32)
    padded[:, : pos.shape[1]] = pos
    buffers["pos_const"] = pack(padded, spec["pos_const"])
    return buffers


# ---------------------------------------------------------------------------
# Final-LayerNorm kernel (graph kind "final_norm").
# ---------------------------------------------------------------------------
def make_final_norm_kernel(config: "DinoBlockConfig", seq_len: int | None = None, cores: int = 1):
    """Build the final-LayerNorm IRModule over the last block's output. Uses the single-pass
    DPU-reduce LayerNorm, M-sharded across `cores` (rows are independent; the fn_w/fn_b weight is
    tiny -- 3 KiB -- so the per-core re-read needs no L2 staging). fn_w/fn_b are D-major packed."""
    from axelera.compiler.alto import script as A

    from ...ops.layer_norm_dpu_reduce import layer_norm_dpu_reduce_mshard

    hidden = config.hidden_size
    hidden_pw = hidden // 64
    eps = float(config.layernorm_eps)
    seq = int(seq_len if seq_len is not None else config.seq_len)
    fn_cores = tuple(range(int(cores))) if (cores and cores > 1) else None

    @A.ir_module
    class DinoFinalNorm:
        @A.prim_func
        def main(
            hidden_states: A.Buffer((1, seq, hidden_pw), _HIDDEN_DTYPE, scope=A.DDR),
            fn_w: A.Buffer((hidden_pw,), "float32x64", scope=A.DDR),  # D-major
            fn_b: A.Buffer((hidden_pw,), "float32x64", scope=A.DDR),
            fn_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
        ):  # noqa: F821
            eps_const = A.alloc_constant([eps] * 64, dtype="float32x64", scope=A.DDR)
            A.output(
                layer_norm_dpu_reduce_mshard(
                    hidden_states,
                    fn_lut,
                    eps_const[0:1],
                    normalized_shape=(hidden,),
                    cores=fn_cores,
                    weight_buffer=fn_w,
                    bias_buffer=fn_b,
                    invsqrt_method="additive",
                )
            )

    return DinoFinalNorm


def final_norm_buffers(model, spec: dict) -> dict:
    """Pack the final-norm gamma/beta (D-major) + the inv_sqrt LUT (fit over the recorded
    final-norm range)."""
    rng = export_lut_ranges(model)
    buffers = {
        "fn_w": pack(model.final_norm.weight.detach(), spec["fn_w"]),  # D-major reshape via pack
        "fn_b": pack(model.final_norm.bias.detach(), spec["fn_b"]),
    }
    buffers["fn_lut"] = fit_lut(rng["final_norm.inv_sqrt_range"], spec["fn_lut"])
    return buffers


# ---------------------------------------------------------------------------
# Fused final-LayerNorm + linear segmentation head (graph kind "final_norm", seg variant).
# ---------------------------------------------------------------------------
def make_final_seg_kernel(config, seq_len=None, cores=1, *, n_classes=150):
    """Final-LayerNorm fused with a linear segmentation head, in ONE kernel: the LN output int8
    quantize is fused in (quant_params), and the head is an int8 matmul (768 -> n_classes) via
    quantized_matmul_activation -- so the graph emits per-token class LOGITS directly (pixels ->
    seg logits in one dispatch), not features. The head's BatchNorm is folded into the linear
    host-side (see final_seg_buffers). n_classes is padded up to a PWORD multiple; the host slices
    back to n_classes and argmaxes."""
    from math import ceil

    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol.common import BufferInfo

    from ...ops.layer_norm_dpu_reduce import layer_norm_dpu_reduce_mshard
    from ...ops.quantized_matmul_activation import quantized_matmul_activation

    hidden = config.hidden_size
    hidden_pw = hidden // 64
    eps = float(config.layernorm_eps)
    seq = int(seq_len if seq_len is not None else config.seq_len)
    n_cls_pad = ceil(n_classes / 64) * 64
    n_cls_pw = n_cls_pad // 64
    fn_cores = tuple(range(int(cores))) if (cores and cores > 1) else None

    @A.ir_module
    class DinoFinalSeg:
        @A.prim_func
        def main(
            hidden_states: A.Buffer((1, seq, hidden_pw), _HIDDEN_DTYPE, scope=A.DDR),
            fn_w: A.Buffer((hidden_pw,), "float32x64", scope=A.DDR),  # D-major gamma
            fn_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            # LN output int8-quant [scale, zp] D-major (beta FOLDED into zp): (2*hidden_pw,)
            fn_qp: A.Buffer((2 * hidden_pw,), "float32x64", scope=A.DDR),
            head_w: A.Buffer((n_cls_pw, hidden), "int8x64", scope=A.DDR),  # NK head weight
            head_dq: A.Buffer((1, 1, 2 * n_cls_pw), "float32x64", scope=A.DDR),  # [dz', ds]
        ):  # noqa: F821
            eps_const = A.alloc_constant([eps] * 64, dtype="float32x64", scope=A.DDR)
            ln_i8 = layer_norm_dpu_reduce_mshard(
                hidden_states,
                fn_lut,
                eps_const[0:1],
                normalized_shape=(hidden,),
                cores=fn_cores,
                weight_buffer=fn_w,
                quant_params=fn_qp,
                invsqrt_method="additive",
            )  # int8 (1, seq, hidden_pw): normalised then activation-quantized to the head input scale
            A.output(
                quantized_matmul_activation(
                    ln_i8, head_w, head_dq, output=BufferInfo(dtype="float32x64")
                )  # (1, seq, n_cls_pw) fp32 class logits
            )

    return DinoFinalSeg


def final_seg_buffers(model, spec: dict, *, head_w, head_b, feat_scale) -> dict:
    """Pack the fused final-norm + seg-head constants.

    head_w/head_b are the BN-folded head (n_classes x hidden float, n_classes bias); feat_scale is
    the calibrated symmetric int8 scale for the final-norm output (the head's activation input).
    Builds: gamma (D-major), inv_sqrt LUT, the LN output-quant table (scale=1/feat_scale, beta
    folded into zp), the per-output-channel int8 head weight (NK, n_classes padded to a PWORD
    multiple), and the [dz', ds] dequant table (ds = feat_scale * weight_scale, bias folded into dz').
    """
    from math import ceil

    import torch

    hidden = int(model.final_norm.weight.numel())
    hidden_pw = hidden // 64
    rng = export_lut_ranges(model)

    def blocks(v):
        return torch.as_tensor(v).to(torch.float32).reshape(-1, 64).contiguous()

    def interleave(scale, zp, c_pw):  # (2*c_pw, 64): [scale0, zp0, scale1, zp1, ...]
        qp = torch.empty((2 * c_pw, 64), dtype=torch.float32)
        qp[0::2], qp[1::2] = blocks(scale), blocks(zp)
        return qp

    # LN output int8-quant: int8 = round(out * (1/feat_scale) + beta/feat_scale) == round((norm*gamma
    # + beta)/feat_scale). beta folded into zp so only quant_params (no bias_buffer) -> one IFD stream.
    beta = model.final_norm.bias.detach().to(torch.float32)
    fs = float(feat_scale)
    fn_qp = interleave(torch.full((hidden,), 1.0 / fs), beta / fs, hidden_pw)

    # Head weight: BN-folded W' (n_classes, hidden) -> per-output-channel symmetric int8, padded to a
    # PWORD multiple, NK layout. Bias b' folded into the dequant zero (dz' = -b'/ds; feat quant is
    # symmetric so dz==0), ds = feat_scale * weight_scale.
    W = torch.as_tensor(head_w).to(torch.float32)  # (n_classes, hidden)
    b = torch.as_tensor(head_b).to(torch.float32)  # (n_classes,)
    n_cls = W.shape[0]
    n_pad = ceil(n_cls / 64) * 64
    Wp = torch.zeros(n_pad, hidden)
    Wp[:n_cls] = W
    bp = torch.zeros(n_pad)
    bp[:n_cls] = b
    ds_w = (Wp.abs().amax(dim=1) / 127.0).clamp_min(1e-12)  # (n_pad,) per-out-channel
    W_i8 = torch.clamp(torch.round(Wp / ds_w[:, None]), -127, 127).to(torch.int8)
    n_cls_pw = n_pad // 64
    nk = W_i8.reshape(n_cls_pw, 64, hidden).permute(0, 2, 1).contiguous()  # (n_cls_pw, hidden, 64)
    ds = fs * ds_w  # (n_pad,)
    dz_folded = -bp / ds
    dq = torch.empty((2 * n_cls_pw, 64), dtype=torch.float32)
    dq[0::2], dq[1::2] = blocks(dz_folded), blocks(ds)

    buffers = {
        "fn_w": pack(model.final_norm.weight.detach(), spec["fn_w"]),
        "fn_lut": fit_lut(rng["final_norm.inv_sqrt_range"], spec["fn_lut"]),
        "fn_qp": pack(fn_qp, spec["fn_qp"]),
        "head_w": pack(nk, spec["head_w"]),
        "head_dq": pack(dq.reshape(1, 1, 2 * n_cls_pw, 64).contiguous(), spec["head_dq"]),
    }
    return buffers

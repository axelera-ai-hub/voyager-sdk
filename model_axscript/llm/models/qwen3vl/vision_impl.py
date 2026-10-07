# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Fused Qwen3-VL vision tower in alto script: patch embed -> N blocks -> patch merger.

The whole tower is authored as ONE device graph (the LLM graph mechanism, as in
models/dino/model.py) and runs in a single dispatch: flattened int8 patches in,
merged image embeddings out. Only the image-processor normalization, the int8
patch quantize and an output slice stay on the host.

The block kernel is the DINOv2 fused block (see models/dino/optimized_impl.py --
same LayerNorm/int8-matmul/SDPA/GELU datapath, same buffer contract) plus the one
thing Qwen3-VL adds: **2-D RoPE on q/k**.

Why RoPE needs a different shape here than in the decoder: ``axol.core.rope``
implements the half-rotation on PWORD BLOCKS -- it wants the two halves of a head
in separate blocks, i.e. head_dim >= 128. The 2B vision head_dim is 64, so both
halves live in one PWORD and the op cannot express it. Instead the rotation is
built from two primitives that do work at lane granularity::

    rotate_half(x) = sign * swizzle(x, lane j <- lane (j + 32) % 64)
    y              = x * cos + swizzle(x) * (sign * sin)

``sign`` (-1 on the low half, +1 on the high half) is folded into the sin table on
the host, so the device sees one lane-permute plus a multiply-add. cos/sin are
compile-time constants: the patch grid is pinned by the config, so the rotary
tables, the interpolated position embedding and the padding mask are all fixed.

The 8B tower (head_dim 72, intermediate 4304) takes the other route: both are
ZERO-PADDED to PWORD multiples, and head_dim pads PER HALF, so head h's x1 lands in
one whole block and its x2 in the next -- which IS ``axol.core.rope``'s contract, so
the padded tower uses the plain block-level op and no swizzle. The padding lives in
the projections' weight layout: q/k/v scatter their rows into the padded head stride,
o_proj and fc2 get zero WEIGHT columns at the pad positions, and that zero column is
what makes the result exact whatever the padded lanes hold. See
qwen3vl.Qwen3VLVisionConfig for the geometry.

DeepStack rides on the same graph. Its three mergers are TAPS (KindContract.tap):
each reads the hidden state after its block in ``deepstack_visual_indexes`` without
consuming it, and writes an extra graph output -- so the dispatch returns three
feature maps plus the merged embeddings. They are the output merger with the merge
reshape moved BEFORE the norm (HF's postshuffle=True), which is why one builder
covers both (see make_vision_merge_kernel).
"""

# flake8: noqa: F821  -- A.Buffer dtype-string annotations look like F821 names
#
# NB: no `from __future__ import annotations` here. TVMScript reads the prim_func's
# parameter annotations at runtime, and PEP 563 would hand it strings instead of the
# A.Buffer objects it parses (same reason qwen3/dino's kernel modules omit it).

from dataclasses import asdict
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from ...export import export_linear_quant_params, export_lut_ranges, export_sdpa_quant_params
from ...layout import broadcast_to_pword, extract_layer_buffer_spec, fit_lut, pack

if TYPE_CHECKING:
    from .qwen3vl import Qwen3VLVisionConfig, Qwen3VLVisionModel

#: The inter-block residual stream (and the q/k projections feeding RoPE) carried as fp16:
#: halves the block's DDR floor, and the DPU computes in float18 regardless.
_HIDDEN_DTYPE = "float16x64"
#: Carry the per-row IFD1 activation-QUANTIZE params as fp16, halving the stream-pop tax on
#: the param-heavy epilogues. OFF by default (dino's DINO_QPARAM_FP16 equivalent); set
#: `QWEN3VL_QPARAM_FP16=1` to opt in. Quantize params ONLY: the dequant tables, the LUT
#: coefficients and the norm weights all overflow in fp16, and layout.pack raises on that.
_QPARAM_FP16 = os.environ.get("QWEN3VL_QPARAM_FP16", "0") == "1"
_QP = "float16x64" if _QPARAM_FP16 else "float32x64"
#: int8 mask fill for padded key columns -- exp-negligible after score scaling.
_MASK_NEG = -127
#: Chained-LUT lengths; must match the torch reference's modelled chains
#: (DinoBlock sets QuantSDPA.exp_n_luts / reciprocal_n_luts from DINO_SDPA_NLUTS).
_SDPA_NLUTS = 1
_GELU_NLUTS = 1


def _fit_cores(blocks: int, cores: int | None) -> int:
    """The largest core count up to `cores` that divides `blocks` evenly.

    Every N-shard here writes a contiguous split output, which cannot be ragged, so a
    matmul can only use as many cores as divide its own output-block count. Taking the
    largest divisor rather than a fixed number lets each tower size use what it has (the
    2B intermediate is 64 blocks, the 8B's padded 68).
    """
    return max(c for c in range(1, int(cores or 1) + 1) if blocks % c == 0)


def rotate_half_perm(head_dim: int) -> np.ndarray:
    """The swizzle weight for ``out[j] = in[(j + head_dim/2) % head_dim]``.

    Combined with the sign folded into the sin table this is exactly
    ``rotate_half([x1, x2]) = [-x2, x1]`` within one PWORD.
    """
    perm = np.zeros((head_dim, head_dim), dtype=np.int8)
    for j in range(head_dim):
        perm[j, (j + head_dim // 2) % head_dim] = 1
    return perm


def rope_constants(cfg: "Qwen3VLVisionConfig") -> tuple[np.ndarray, np.ndarray]:
    """(cos, sin*sign) baked into the block kernel, shaped (1, seq, 1, 64).

    The unpadded (2B) layout, for the lane-swizzle rotation. Padded token rows get
    cos=1/sin=0 (identity rotation); they are masked out of attention anyway, and an
    identity keeps them finite.
    """
    from .qwen3vl import vision_rope_cos_sin

    cos, sin = vision_rope_cos_sin(cfg)  # (num_patches, head_dim)
    head_dim = cfg.head_dim
    sign = np.concatenate([-np.ones(head_dim // 2), np.ones(head_dim // 2)]).astype(np.float32)
    cos_pad = np.ones((cfg.seq_len, head_dim), dtype=np.float32)
    sin_pad = np.zeros((cfg.seq_len, head_dim), dtype=np.float32)
    cos_pad[: cfg.num_patches] = cos.numpy()
    sin_pad[: cfg.num_patches] = sin.numpy() * sign
    return (
        cos_pad.reshape(1, cfg.seq_len, head_dim // 64, 64),
        sin_pad.reshape(1, cfg.seq_len, head_dim // 64, 64),
    )


def rope_trig_constant(cfg: "Qwen3VLVisionConfig") -> np.ndarray:
    """The padded-head layout: sin||cos over the HALF width, (1, seq, 2*half_pw, 64).

    ``axol.core.rope``'s trig contract is trig[..., :blocks] = sin, trig[..., blocks:] = cos
    over the half width. The op computes the true rotate_half (y1 = x1*cos - x2*sin,
    y2 = x2*cos + x1*sin), so unlike the swizzle path there is no sign to fold. Padded lanes
    are don't-care (the projections zero the matching x lanes) but carry the identity anyway.
    """
    from .qwen3vl import vision_rope_cos_sin

    cos, sin = vision_rope_cos_sin(cfg)  # (num_patches, head_dim), halves duplicated
    half, phalf = cfg.head_dim // 2, cfg.padded_head_dim // 2
    cos_pad = np.ones((cfg.seq_len, phalf), dtype=np.float32)
    sin_pad = np.zeros((cfg.seq_len, phalf), dtype=np.float32)
    cos_pad[: cfg.num_patches, :half] = cos.numpy()[:, :half]
    sin_pad[: cfg.num_patches, :half] = sin.numpy()[:, :half]
    trig = np.concatenate([sin_pad, cos_pad], axis=-1)  # (seq, 2*phalf)
    return trig.reshape(1, cfg.seq_len, 2 * (phalf // 64), 64)


# --------------------------------------------------------------------------------------
# Kernels
#
# The helpers below run during TRACING (they are called from a prim_func body but are
# plain Python), which is how a branch or a reshape-chain stays out of the TVMScript
# parser -- the same pattern as models/dino/optimized_impl._mlp_out.
# --------------------------------------------------------------------------------------


def _pad_tokens(embedded, n_pad: int, hidden_pw: int):
    """Append n_pad zero rows so the token axis is a PWORD multiple (no-op when aligned)."""
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A

    if not n_pad:
        return embedded
    zpad = A.alloc_constant(np.zeros((1, n_pad, hidden_pw, 64)), "float32x64", scope=A.DDR)
    return axol.core.concat(embedded, zpad, dim=1)


def _to_heads(x, seq: int, heads: int, head_dim_pw: int):
    """(1, seq, qkv_pw) -> (1, heads, seq, head_dim_pw).

    Head-major, so a head's (padded) blocks are adjacent -- which is what the
    projections' weight layout already produces.
    """
    from axelera.compiler.alto import axol

    return axol.core.transpose(
        axol.atomic.reshape(x, shape=(1, seq, heads, head_dim_pw)), dims=(0, 2, 1, 3)
    )


def _rope(x, cos, sin, perm: np.ndarray):
    """y = x * cos + rotate_half(x) * sin, with rotate_half as a lane permute.

    The sign of the low half is folded into `sin` on the host (see rope_constants),
    so the whole rotation is one swizzle plus a multiply-add.
    """
    from axelera.compiler.alto import axol

    rotated = axol.core.swizzle(x, perm)
    return axol.atomic.add(axol.atomic.mul(x, cos), axol.atomic.mul(rotated, sin))


def _qkv_projections(qkv_q, weights, dequants, *, group_cores):
    """q/k/v for one vision block: one whole projection per core, or three in sequence.

    Each projection is fixed-overhead / weight-load bound, so overlapping all three beats
    splitting any one of them, and it replicates no weights. The group mixes output dtypes:
    q and k reach RoPE in float, v fuses its SDPA-input quantize and emits int8.

    A plain helper, not an inline branch, because TVMScript parses a body-level `if` as a
    TIR statement -- the same reason dino's `_mlp_out` dispatches its layouts here.
    """
    from axelera.compiler.alto.axol.common import BufferInfo

    from ...ops.parallel_matmul_group import parallel_matmul_group
    from ...ops.quantized_matmul_activation import quantized_matmul_activation

    if group_cores:
        return parallel_matmul_group(
            qkv_q,
            weights,
            dequants,
            cores=group_cores,
            output_dtype=[_HIDDEN_DTYPE, _HIDDEN_DTYPE, "int8x64"],
            fuse_output_quant=[False, False, True],
        )
    q = quantized_matmul_activation(
        qkv_q, weights[0], dequants[0], output=BufferInfo(dtype=_HIDDEN_DTYPE)
    )
    k = quantized_matmul_activation(
        qkv_q, weights[1], dequants[1], output=BufferInfo(dtype=_HIDDEN_DTYPE)
    )
    v8 = quantized_matmul_activation(
        qkv_q,
        weights[2],
        dequants[2],
        fuse_output_quant=True,
        output=BufferInfo(dtype="int8x64"),
    )
    return q, k, v8


def _rope_mshard(x, cos, sin, perm: np.ndarray, *, cores: int, heads: int):
    """`_rope`, with the multiply-add split head-band per core.

    Every head rotates independently, so the split is exact. It was worth doing because
    RoPE measured 0.606 ms of a 2.82 ms block at four cores -- the same 0.60 ms it cost at
    ONE -- the largest operator getting no benefit from the other three cores. Banding it
    is worth 0.301 ms/block.

    The HEAD axis, not seq: cos/sin are (1, seq, head_dim/64) and broadcast over heads, so
    each core reads the whole table and slices nothing. The ops take no `core=`, so
    placement comes from the enclosing `A.AICore` region, as in `mshard_matmul_residual`.

    The swizzle stays whole and single-core: it reinterprets its operand's storage
    (`data=input.data`), so it needs a base pointer and cannot take a band. Sharding it
    inside the op was tried and is a dead end twice over -- slicing an alias drops the
    offset (only band 0 came out right), and in isolation the whole permute is 0.094 ms
    while a 4-core split costs 0.140 ms, so barrier overhead exceeds the work.

    Falls back to unsharded `_rope` at one core or when heads do not divide evenly.
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A
    from tvm import tir

    from ...ops.quantized_matmul_activation import slice_n

    if not cores or cores <= 1 or heads % cores:
        return _rope(x, cos, sin, perm)

    core_ids = tuple(range(int(cores)))
    band = heads // int(cores)
    out = A.alloc_buffer(tuple(int(dim) for dim in x.shape), dtype=x.dtype, scope=A.DDR)
    out_slice = tir.BufferSlice(out)
    rotated = axol.core.swizzle(x, perm)  # whole-buffer, see the docstring
    for index, core in enumerate(core_ids):
        lo, hi = index * band, (index + 1) * band
        with A.AICore(core=core):
            A.barrier(*core_ids)
            axol.atomic.add(
                axol.atomic.mul(slice_n(x, 1, lo, hi), cos),
                axol.atomic.mul(slice_n(rotated, 1, lo, hi), sin),
                output=slice_n(out_slice, 1, lo, hi),
            )
            A.barrier(*core_ids)
    return out


def _make_rope(cfg: "Qwen3VLVisionConfig", cores: int = 1):
    """This tower's RoPE as an apply(x) built at kernel-build time.

    Padded heads (8B) carry the two halves in separate blocks, which is
    ``axol.core.rope``'s contract; the unpadded 2B tower keeps both in one PWORD and uses
    the lane-swizzle identity. The branch is taken here rather than in the prim_func body,
    so the untaken path is never parsed as TVMScript; the constants are allocated on first
    use (inside the body, where they belong) and reused by q and k.
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A

    tables: dict = {}

    if cfg.pads_head_dim:
        trig_np = rope_trig_constant(cfg)

        def apply(x):
            if not tables:
                tables["trig"] = A.alloc_constant(trig_np, "float32x64", scope=A.DDR)
            return axol.core.rope(x, tables["trig"])

        return apply

    cos_np, sin_np = rope_constants(cfg)
    perm = rotate_half_perm(cfg.head_dim)

    def apply(x):
        if not tables:
            tables["cos"] = A.alloc_constant(cos_np, "float32x64", scope=A.DDR)
            tables["sin"] = A.alloc_constant(sin_np, "float32x64", scope=A.DDR)
        return _rope_mshard(
            x, tables["cos"], tables["sin"], perm, cores=int(cores), heads=cfg.num_heads
        )

    return apply


def _merger_norm(
    hidden_states,
    weight,
    lut,
    eps,
    *,
    postshuffle: bool,
    hidden: int,
    merged_dim: int,
    merged_seq: int,
    merged_pw: int,
    ln_cores,
):
    """The patch merger's LayerNorm, in the merged (merged_seq, merged_pw) layout.

    Pre-shuffle (the output merger): normalise at vision width with the single-pass
    dpu_reduce norm, then fold 4 tokens into one row -- a pure view, the tokens being
    adjacent. Post-shuffle (the DeepStack mergers): fold first and normalise the whole
    4*hidden vector, too wide for the register-resident norm, so the transpose-reduce Kahan
    one. Called during tracing, keeping the branch out of the TVMScript parser.
    """
    from axelera.compiler.alto import axol

    from ...ops.layer_norm_dpu_reduce import layer_norm_dpu_reduce_mshard

    if postshuffle:
        # Deliberately single-core. An M-shard over the merged tokens was tried and is
        # bit-exact (each row's Kahan sum stays whole), but measured 1.799 ms against 1.800
        # -- no gain. The ~2x a DeepStack merger costs over the output merger is a ONE-core
        # observation; at four cores the merger's time is its fc1/fc2, already sharded.
        return axol.core.layer_norm_two_pass_kahan(
            axol.atomic.reshape(hidden_states, shape=(merged_seq, merged_pw)),
            lut,
            eps,
            normalized_shape=(merged_dim,),
            weight_buffer=weight,
            invsqrt_method="additive",
        )
    normed = layer_norm_dpu_reduce_mshard(
        hidden_states,
        lut,
        eps,
        normalized_shape=(hidden,),
        cores=ln_cores,
        weight_buffer=weight,
        invsqrt_method="additive",
    )
    return axol.atomic.reshape(normed, shape=(merged_seq, merged_pw))


def make_vision_embed_kernel(cfg: "Qwen3VLVisionConfig", cores: int = 1):
    """Patch projection: int8 patches -> hidden + interpolated position embedding.

    The Conv3d patch embed has stride == kernel, so it is one dense matmul over
    each patch's C*T*P*P features -- which is the layout the image processor
    already emits, so there is no im2col on device. The conv bias is folded into
    the dequant zero point (dz' = dz - bias/ds). Padding rows (patch count up to
    the PWORD multiple) are appended as zeros.
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol.common import BufferInfo

    from ...ops.quantized_matmul_activation import quantized_matmul_activation_multicore

    hidden_pw = cfg.hidden_size // 64
    k_pw = cfg.patch_dim // 64
    n_patches, seq = cfg.num_patches, cfg.seq_len
    n_pad = seq - n_patches
    # The output stays unified (the position add and the token pad read it whole), so this
    # is the strided-write N-shard: worth little on its own (embed is ~0.3% of a frame), but
    # it stops the kernel ignoring `cores` outright.
    proj_cores = _fit_cores(hidden_pw, cores)

    @A.ir_module
    class Qwen3VLEmbed:
        @A.prim_func
        def main(
            patches: A.Buffer((1, n_patches, k_pw), "int8x64", scope=A.DDR),
            patch_w: A.Buffer((hidden_pw, cfg.patch_dim), "int8x64", scope=A.DDR),  # NK
            patch_dq: A.Buffer((1, 1, 2 * hidden_pw), "float32x64", scope=A.DDR),  # [dz', ds]
            pos_const: A.Buffer((1, seq, hidden_pw), "float32x64", scope=A.DDR),
        ):  # noqa: F821
            embedded = quantized_matmul_activation_multicore(
                patches,
                patch_w,
                patch_dq,
                cores=proj_cores,
                output=BufferInfo(dtype="float32x64"),
            )
            tokens = _pad_tokens(embedded, n_pad, hidden_pw)
            A.output(axol.atomic.add(tokens, pos_const, output=BufferInfo(dtype=_HIDDEN_DTYPE)))

    return Qwen3VLEmbed


def make_vision_block_kernel(cfg: "Qwen3VLVisionConfig", cores: int = 1):
    """One fused Qwen3-VL vision block (LayerNorm -> RoPE attention -> GELU MLP).

    Fusions carried over from the DINOv2 block: the LayerNorm emits int8 directly
    (its output quantize is fused into the norm datapath, with the norm's beta
    folded into the quant zero point), every projection is one
    quantized_matmul_activation with the bias folded into the dequant zero, the
    SDPA output quantize is fused into P@V (so o_proj reads int8), and fc1 fuses
    dequant + GELU + fc2's input quantize into its write.

    q/k differ from DINOv2: they must reach RoPE in float, so their matmuls emit
    fp16 (2-wide dequant table) and are quantized after the rotation with the
    calibrated per-head-dim SDPA scales. v keeps the fused int8 write.
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol.common import BufferInfo
    from axelera.compiler.alto.axol.utils.lut_chain import ChainedLUTMethod

    from ...ops.layer_norm_dpu_reduce import layer_norm_dpu_reduce_mshard
    from ...ops.mshard_matmul_residual import mshard_matmul_residual
    from ..dino.optimized_impl import _mlp_out

    hidden = cfg.hidden_size
    # Device widths: == the checkpoint's on 2B, zero-padded to PWORD multiples on
    # 8B (head_dim 72 -> 128 per padded head, intermediate 4304 -> 4352).
    inter = cfg.padded_intermediate
    heads, head_dim = cfg.num_heads, cfg.head_dim
    qkv_dim = cfg.qkv_dim
    hidden_pw, inter_pw = hidden // 64, inter // 64
    qkv_pw = qkv_dim // 64
    head_dim_pw = cfg.head_dim_pw
    seq = cfg.seq_len
    ctx_blocks = seq // 64
    eps = float(cfg.layernorm_eps)
    rope = _make_rope(cfg, cores)
    # Multicore, following models/dino/optimized_impl (same block shape, same helpers):
    #   * SDPA head-sharded across every core;
    #   * LayerNorm M-sharded (rows are independent, the weight is tiny);
    #   * fc1 N-sharded over min(cores, 3) with a CONTIGUOUS split output, which fc2
    #     then consumes via matrix_a_is_split (no strided per-row output DMA);
    #   * fc2 and o_proj M-sharded with the residual add banded into the per-core tail.
    # q/k/v run as a parallel group (one whole projection per core) once there are three
    # cores to put them on -- they are fixed-overhead-bound, so overlapping them is worth
    # more than splitting any one of them. Needs exactly three cores' worth of placement;
    # a fourth core has no fourth projection to take.
    multicore = bool(cores and cores > 1)
    qkv_group_cores = (0, 1, 2) if (cores or 0) >= 3 else None
    ln_cores = tuple(range(int(cores))) if multicore else None
    sdpa_kw = {"cores": cores} if multicore else {}
    # fc1 N-shards over its intermediate blocks (dino hard-codes 3 here instead).
    fc_cores = _fit_cores(inter_pw, cores) if multicore else 1
    fc1_split = fc_cores > 1
    mshard_cores = int(cores) if (multicore and seq % int(cores) == 0) else None
    # Whole-MLP M-shard only pays once each core's row band amortizes reading the FULL
    # fc1+fc2 weights from L2 (dino measures +1.2% at seq 1408, -4.7% at seq 320).
    mlp_mshard = bool(mshard_cores) and seq >= 1024

    @A.ir_module
    class Qwen3VLVisionBlockFused:
        @A.prim_func
        def main(
            hidden_states: A.Buffer((1, seq, hidden_pw), _HIDDEN_DTYPE, scope=A.DDR),
            ln1_w: A.Buffer((hidden_pw,), "float32x64", scope=A.DDR),  # D-major
            ln1_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            # q/k/v share one interleaved [scale, zp] input quant (one SmoothQuant group);
            # ln1's beta is folded into zp so the norm needs no second IFD1 stream.
            qkv_qparams: A.Buffer((2 * hidden_pw,), _QP, scope=A.DDR),
            # q/k/v are hidden -> qkv_dim: hidden on 2B, heads*padded_head_dim on 8B.
            q_proj_w: A.Buffer((qkv_pw, hidden), "int8x64", scope=A.DDR),  # NK
            q_proj_dq: A.Buffer((1, 1, 2 * qkv_pw), "float32x64", scope=A.DDR),
            k_proj_w: A.Buffer((qkv_pw, hidden), "int8x64", scope=A.DDR),
            k_proj_dq: A.Buffer((1, 1, 2 * qkv_pw), "float32x64", scope=A.DDR),
            v_proj_w: A.Buffer((qkv_pw, hidden), "int8x64", scope=A.DDR),
            # 4-wide [dz', ds, q_scale, q_zp]: v's SDPA-input quantize is fused into its write.
            v_proj_dq: A.Buffer((1, 1, 4 * qkv_pw), "float32x64", scope=A.DDR),
            # Post-RoPE SDPA input scales (per head-dim, zero-pinned).
            q_attn_qs: A.Buffer((1, 1, 1, head_dim_pw), "float32x64", scope=A.DDR),
            k_attn_qs: A.Buffer((1, 1, 1, head_dim_pw), "float32x64", scope=A.DDR),
            sdpa_exp_lut: A.Buffer((_SDPA_NLUTS,), "float32x64", scope=A.DDR),
            sdpa_inv_lut: A.Buffer((_SDPA_NLUTS,), "float32x64", scope=A.DDR),
            sdpa_scores_dq_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            sdpa_weights_q_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            sdpa_output_dq_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            # 1-head-broadcast int8 padding mask ({0, -127}) over the padded token axis.
            attn_mask: A.Buffer((1, 1, seq, ctx_blocks), "int8x64", scope=A.DDR),
            # o_proj contracts over the (padded) head layout: qkv_dim in, hidden out.
            o_proj_qparams: A.Buffer((2 * qkv_pw,), _QP, scope=A.DDR),
            o_proj_w: A.Buffer((hidden_pw, qkv_dim), "int8x64", scope=A.DDR),
            o_proj_dq: A.Buffer((1, 1, 2 * hidden_pw), "float32x64", scope=A.DDR),
            ln2_w: A.Buffer((hidden_pw,), "float32x64", scope=A.DDR),
            ln2_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            fc1_qparams: A.Buffer((2 * hidden_pw,), _QP, scope=A.DDR),
            fc1_w: A.Buffer((inter_pw, hidden), "int8x64", scope=A.DDR),
            # 4-wide [dz', ds, fc2_in_scale, fc2_in_zp]: dequant + GELU + fc2's input
            # quantize all fused into fc1's write, so fc1 emits int8 straight to fc2.
            fc1_dq: A.Buffer((1, 1, 4 * inter_pw), "float32x64", scope=A.DDR),
            gelu_lut: A.Buffer((_GELU_NLUTS,), "float32x64", scope=A.DDR),
            fc2_w: A.Buffer((hidden_pw, inter), "int8x64", scope=A.DDR),
            fc2_dq: A.Buffer((1, 1, 2 * hidden_pw), "float32x64", scope=A.DDR),
        ):  # noqa: F821
            eps_const = A.alloc_constant([eps] * 64, dtype="float32x64", scope=A.DDR)
            # SDPA zero points are pinned to 0 (q/k/v and the softmax weights are all
            # symmetric); the int8 range bounds are constants too.
            w_zp = A.alloc_constant(np.zeros((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR)
            q_zp = A.alloc_constant(np.zeros((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR)
            v_zp = A.alloc_constant(np.zeros((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR)
            w_qmin = A.alloc_constant(np.full((1, 1, 1, 1, 64), -128.0), "float32x64", scope=A.DDR)
            w_qmax = A.alloc_constant(np.full((1, 1, 1, 1, 64), 127.0), "float32x64", scope=A.DDR)

            # --- Attention: LayerNorm (+ fused int8 quantize) -> q/k/v -> RoPE -> SDPA.
            qkv_q = layer_norm_dpu_reduce_mshard(
                hidden_states,
                ln1_lut,
                eps_const[0:1],
                normalized_shape=(hidden,),
                cores=ln_cores,
                weight_buffer=ln1_w,
                quant_params=qkv_qparams,
                output_dtype="int8x64",
            )
            q, k, v8 = _qkv_projections(
                qkv_q,
                [q_proj_w, k_proj_w, v_proj_w],
                [q_proj_dq, k_proj_dq, v_proj_dq],
                group_cores=qkv_group_cores,
            )

            q_rope = rope(_to_heads(q, seq, heads, head_dim_pw))
            k_rope = rope(_to_heads(k, seq, heads, head_dim_pw))
            # SDPA input quantize (zero-pinned, so scale only).
            q8 = axol.atomic.dpu_cast(
                axol.atomic.mul(q_rope, q_attn_qs), output=BufferInfo(dtype="int8x64")
            )
            k8 = axol.atomic.dpu_cast(
                axol.atomic.mul(k_rope, k_attn_qs), output=BufferInfo(dtype="int8x64")
            )

            attn = axol.core.scaled_dot_product_attention(
                q8,
                k8,
                _to_heads(v8, seq, heads, head_dim_pw),
                sdpa_scores_dq_scale,
                sdpa_exp_lut,
                sdpa_inv_lut,
                sdpa_weights_q_scale,
                w_zp,
                w_qmin,
                w_qmax,
                sdpa_output_dq_scale,
                q_zp,
                v_zp,
                attn_mask,
                is_causal=False,
                enable_gqa=False,
                # The REAL head_dim: the padded lanes are zero on both sides, so they
                # add nothing to a score and must not enter its normalisation either.
                scale=1.0 / (head_dim**0.5),
                neg_exp_method=ChainedLUTMethod.ADDITIVE,
                inv_method=ChainedLUTMethod.ADDITIVE,
                # Fuse o_proj's input quantize into P@V so the head transpose moves int8.
                attention_output_quantizer=o_proj_qparams,
                **sdpa_kw,
            )
            o3 = axol.atomic.reshape(
                axol.core.transpose(attn, dims=(0, 2, 1, 3)), shape=(1, seq, qkv_pw)
            )
            # o_proj + the residual add, M-sharded when multicore (each core does its own
            # row band and its banded add); plain matmul + add on one core.
            res1 = mshard_matmul_residual(
                o3,
                o_proj_w,
                o_proj_dq,
                hidden_states,
                seq=seq,
                n_blocks=hidden_pw,
                cores=(mshard_cores or 1),
                matrix_a_is_split=False,
                resid_dtype=_HIDDEN_DTYPE,
                out_dtype=_HIDDEN_DTYPE,
            )

            # --- MLP: LayerNorm (+ fused quantize) -> fc1 + GELU -> fc2 -> residual.
            f2 = layer_norm_dpu_reduce_mshard(
                res1,
                ln2_lut,
                eps_const[0:1],
                normalized_shape=(hidden,),
                cores=ln_cores,
                weight_buffer=ln2_w,
                quant_params=fc1_qparams,
                output_dtype="int8x64",
            )
            # fc1 (+GELU, +fc2's input quant) -> fc2 -> residual, dispatched in plain
            # Python between the N-shard/M-shard layouts (dino's _mlp_out).
            A.output(
                _mlp_out(
                    f2,
                    fc1_w,
                    fc1_dq,
                    gelu_lut,
                    fc2_w,
                    fc2_dq,
                    res1,
                    mlp_mshard=mlp_mshard,
                    mlp_cores=int(cores),
                    fc_cores=fc_cores,
                    fc1_split=fc1_split,
                    fc2_ms_cores=mshard_cores,
                    seq=seq,
                    hidden_pw=hidden_pw,
                    resid_dtype=_HIDDEN_DTYPE,
                    out_dtype=_HIDDEN_DTYPE,
                )
            )

    return Qwen3VLVisionBlockFused


def make_vision_merge_kernel(
    cfg: "Qwen3VLVisionConfig", cores: int = 1, *, postshuffle: bool = False
):
    """Patch merger: LayerNorm -> 2x2 spatial merge -> GELU MLP -> decoder hidden size.

    The spatial merge is free: the processor emits patches in merge-block order,
    so folding 4 consecutive tokens into one 4*hidden row is a pure reshape of the
    packed buffer. The LayerNorm runs at vision width (pre-shuffle, as in the
    checkpoint) and writes float; the int8 quantize is a separate pass at merged
    width because fc1's per-channel scales span all 4*hidden channels (the norm's
    beta is folded into that quantizer's zero point, tiled over the 4 sub-tokens).

    ``postshuffle`` builds the DeepStack variant instead (HF's
    Qwen3VLPatchMerger(postshuffle=True)): the merge reshape comes FIRST and the norm runs
    over the whole 4*hidden vector. That width would need ~3x64 resident registers in the
    single-pass dpu_reduce norm, well past the DPU register file, so it uses the
    transpose-reduce Kahan LayerNorm -- whose gamma is broadcast-per-lane rather than
    D-major (see merge_buffers). Everything after the norm is identical.
    """
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol.common import BufferInfo

    from ...ops.quantize import quantize
    from ...ops.quantized_matmul_activation import quantized_matmul_activation_multicore

    hidden = cfg.hidden_size
    hidden_pw = hidden // 64
    merged_dim = hidden * cfg.merge_unit
    merged_pw = merged_dim // 64
    out_pw = cfg.out_hidden_size // 64
    seq = cfg.seq_len
    merged_seq = seq // cfg.merge_unit
    eps = float(cfg.layernorm_eps)
    ln_cores = tuple(range(int(cores))) if cores and cores > 1 else None
    # Both matmuls N-shard over their own output blocks. fc1's split output feeds fc2 as
    # split K (matrix_a_is_split), so there is no un-split in between.
    fc1_cores = _fit_cores(merged_pw, cores)
    fc2_cores = _fit_cores(out_pw, cores)
    fc1_split = fc1_cores > 1
    # The norm's width and hence its gamma buffer: vision width (D-major, dpu_reduce)
    # pre-shuffle, merged width (broadcast, Kahan) post-shuffle.
    norm_w_shape = (merged_dim,) if postshuffle else (hidden_pw,)

    @A.ir_module
    class Qwen3VLMerge:
        @A.prim_func
        def main(
            hidden_states: A.Buffer((1, seq, hidden_pw), _HIDDEN_DTYPE, scope=A.DDR),
            mnorm_w: A.Buffer(norm_w_shape, "float32x64", scope=A.DDR),
            mnorm_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            # [scale, zp] per merged channel, beta folded into zp (tiled over the 4 sub-tokens).
            mfc1_qparams: A.Buffer((2 * merged_pw,), _QP, scope=A.DDR),
            mfc1_w: A.Buffer((merged_pw, merged_dim), "int8x64", scope=A.DDR),  # NK
            # 4-wide [dz', ds, fc2_in_scale, fc2_in_zp] -- GELU + fc2 quantize fused.
            mfc1_dq: A.Buffer((1, 1, 4 * merged_pw), "float32x64", scope=A.DDR),
            mgelu_lut: A.Buffer((_GELU_NLUTS,), "float32x64", scope=A.DDR),
            mfc2_w: A.Buffer((out_pw, merged_dim), "int8x64", scope=A.DDR),
            mfc2_dq: A.Buffer((1, 1, 2 * out_pw), "float32x64", scope=A.DDR),
        ):  # noqa: F821
            eps_const = A.alloc_constant([eps] * 64, dtype="float32x64", scope=A.DDR)
            merged = _merger_norm(
                hidden_states,
                mnorm_w,
                mnorm_lut,
                eps_const[0:1],
                postshuffle=postshuffle,
                hidden=hidden,
                merged_dim=merged_dim,
                merged_seq=merged_seq,
                merged_pw=merged_pw,
                ln_cores=ln_cores,
            )
            q = quantize(merged, mfc1_qparams)
            h1 = quantized_matmul_activation_multicore(
                q,
                mfc1_w,
                mfc1_dq,
                cores=fc1_cores,
                luts=mgelu_lut,
                fuse_output_quant=True,
                split_output=fc1_split,
                output=BufferInfo(dtype="int8x64"),
            )
            A.output(
                quantized_matmul_activation_multicore(
                    h1,
                    mfc2_w,
                    mfc2_dq,
                    cores=fc2_cores,
                    matrix_a_is_split=fc1_split,
                    output=BufferInfo(dtype="float32x64"),
                )
            )

    return Qwen3VLMerge


# --------------------------------------------------------------------------------------
# Parameter binding
# --------------------------------------------------------------------------------------


def _nk(weight: torch.Tensor) -> torch.Tensor:
    """[out, in] int8 -> NK (out//64, in, 64), the layout core.matmul(b_transposed) wants."""
    out, inn = weight.shape
    return weight.reshape(out // 64, 64, inn).permute(0, 2, 1).contiguous().to(torch.int8)


def _interleave(*channels) -> torch.Tensor:
    """Interleave per-channel vectors on the channel axis: [a0, b0, a1, b1, ...].

    Each is first laid out (C,) -> (C//64, 64): one per-channel value per PWORD lane.
    """
    packed = [torch.as_tensor(c).to(torch.float32).reshape(-1, 64).contiguous() for c in channels]
    stride = len(packed)
    out = torch.empty((stride * packed[0].shape[0], 64), dtype=torch.float32)
    for i, part in enumerate(packed):
        out[i::stride] = part
    return out


#
# ---- 8B zero padding -----------------------------------------------------------------
#
# Every helper below is the identity when the checkpoint's widths are already PWORD
# multiples, so the 2B path never sees the padding at all.
#
# Two invariants carry the correctness:
#   * every padded OUTPUT channel is written as zero (its dequant scale is zero), and
#   * every padded INPUT channel is contracted against a zero WEIGHT column,
# so whatever a padded lane holds, it cannot reach a real output.


def _head_pad_index(heads: int, head_dim: int, padded_head_dim: int) -> torch.Tensor:
    """Where each real head-dim channel lands in the padded, half-aligned layout.

    Head h's first half goes to h*padded, its second to h*padded + padded/2 -- so each
    RoPE half starts on a PWORD boundary. Returns a flat (heads*head_dim,) index into
    the padded heads*padded_head_dim axis.
    """
    half, phalf = head_dim // 2, padded_head_dim // 2
    within = torch.cat([torch.arange(half), phalf + torch.arange(half)])
    return (torch.arange(heads).reshape(-1, 1) * padded_head_dim + within.reshape(1, -1)).reshape(
        -1
    )


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
    """A per-output-channel vector, widened to the padded channel count.

    Per-tensor scalars are expanded to the REAL width first, so the padded lanes stay
    zero rather than inheriting the scalar (a zero dequant scale is what pins a padded
    output to zero).
    """
    t = torch.as_tensor(values).to(torch.float32).reshape(-1)
    if t.numel() == width:
        return t
    if t.numel() == 1:  # per-tensor: broadcast over the REAL channels only
        t = t.expand(width if index is None else index.numel()).contiguous()
    if index is None:
        return _pad_axis(t, 0, width)
    return _scatter_axis(t, 0, index, width)


def _dequant_table(params: dict, bias: torch.Tensor, scale: torch.Tensor | None = None):
    """[dz', ds] with the linear's bias folded into the dequant zero.

    (matmul - dz') * ds == dequant(matmul) + bias for dz' = dz - bias/ds. `scale`
    is an optional extra per-output-channel factor (LayerScale) folded into ds --
    it must not touch dz', which keeps the original ds in the bias term.
    """
    ds = params["ds"].to(torch.float32)
    dz = params["dz"].to(torch.float32) - bias.to(torch.float32) / ds
    return dz, (ds if scale is None else ds * scale.to(torch.float32))


def embed_buffers(model: "Qwen3VLVisionModel", spec: dict) -> dict:
    """Patch weight (NK int8), bias-folded dequant table, and the position constant."""
    # Export just the patch embed: the tower-wide export would int8-quantize every
    # block's weights again for one buffer.
    params = export_linear_quant_params(model.patch_embed)["proj"]
    hidden = int(model.config.hidden_size)
    hidden_pw = hidden // 64
    dz, ds = _dequant_table(params, model.patch_embed.bias.detach())
    seq_pad = spec["pos_const"][0][1]
    pos = torch.zeros(seq_pad, hidden)
    pos[: model.config.num_patches] = model.pos_embed.detach().to(torch.float32)
    return {
        "patch_w": pack(_nk(params["w"]), spec["patch_w"]),
        "patch_dq": pack(_interleave(dz, ds).reshape(1, 1, 2 * hidden_pw, 64), spec["patch_dq"]),
        "pos_const": pack(pos, spec["pos_const"]),
    }


#: block projection -> the module attribute holding its float bias (folded into dz').
_PROJ_BIAS = {
    "q_proj": "q_bias",
    "k_proj": "k_bias",
    "v_proj": "v_bias",
    "o_proj": "o_bias",
    "fc1": "fc1_bias",
    "fc2": "fc2_bias",
}


def block_buffers(block, spec: dict) -> dict:
    """Bind one vision block: NK weights, folded dequant tables, quant params, LUTs.

    Follows models/dino/optimized_impl.optimized_block_buffers, with q/k emitting
    float (2-wide table, no fused output quant) because RoPE runs between the
    projection and the SDPA quantize.
    """
    lin = export_linear_quant_params(block)
    sdpa = export_sdpa_quant_params(block)["sdpa"]
    ranges = export_lut_ranges(block)
    heads, head_dim = block.config.num_heads, block.config.head_dim
    hidden = block.config.hidden_size
    buffers: dict = {}

    # The device widths come off the KERNEL's own spec, so this binds whatever geometry
    # the kernel was built for: identical to the checkpoint on 2B, zero-padded on
    # 8B (head_dim 72 -> 128 per head, intermediate 4304 -> 4352). A spec that omits a
    # weight (a partial bind) falls back to the checkpoint's own width.
    def _width(spec_name: str, proj: str) -> int:
        if spec_name in spec:
            return spec[spec_name][0][0] * 64
        return int(lin[proj]["w"].shape[0])

    qkv_dim = _width("q_proj_w", "q_proj")
    inter = _width("fc1_w", "fc1")
    head_index = _head_pad_index(heads, head_dim, qkv_dim // heads)
    # Per-output-channel width of each projection, and the scatter for the head-major ones.
    out_width = {"q_proj": qkv_dim, "k_proj": qkv_dim, "v_proj": qkv_dim}
    out_width.update({"o_proj": hidden, "fc1": inter, "fc2": hidden})
    out_index = {"q_proj": head_index, "k_proj": head_index, "v_proj": head_index}

    def pad_weight(proj: str, w: torch.Tensor) -> torch.Tensor:
        """[out, in] -> the device's [out, in], zeros in every padded row/column.

        q/k/v scatter their OUTPUT rows into the padded head stride; o_proj and fc2 pad
        the matching INPUT columns with zeros, which is what makes a padded lane's
        contents irrelevant downstream.
        """
        if proj in out_index:
            return _scatter_axis(w, 0, out_index[proj], qkv_dim)
        if proj == "o_proj":
            return _scatter_axis(w, 1, head_index, qkv_dim)
        if proj == "fc1":
            return _pad_axis(w, 0, inter)
        if proj == "fc2":
            return _pad_axis(w, 1, inter)
        return w

    def put(name, value):
        if name in spec:
            buffers[name] = pack(value, spec[name])

    # LayerNorm gammas are D-major (pack reshapes); their betas fold into the
    # following quantizer's zero point, so there is no bias buffer.
    put("ln1_w", block.ln1.weight.detach())
    put("ln2_w", block.ln2.weight.detach())

    for name, proj, norm_bias in (
        ("qkv_qparams", "q_proj", block.ln1.bias),
        ("o_proj_qparams", "o_proj", None),
        ("fc1_qparams", "fc1", block.ln2.bias),
    ):
        if name not in spec:
            continue
        params = lin[proj]
        qs = params["qs"].to(torch.float32).reshape(-1)
        qz = params["qz"].to(torch.float32).reshape(-1)
        if norm_bias is not None:
            qz = qz + norm_bias.detach().to(torch.float32).reshape(-1) * qs
        if proj == "o_proj":  # per-INPUT-channel: the (padded) head layout
            qs, qz = (_pad_channels(v, qkv_dim, head_index) for v in (qs, qz))
        buffers[name] = pack(_interleave(qs, qz), spec[name])

    # v_proj's SDPA-input quantize is fused into its write; q/k are quantized after RoPE.
    # The attention scale is per head-dim and uniform across heads, so it tiles over
    # the projection's output blocks (one block per head unpadded, two when padded).
    v_scale = torch.as_tensor(sdpa["qs_v"]).to(torch.float32).reshape(-1)
    if v_scale.numel() == 1:  # per-tensor
        v_scale = v_scale.expand(head_dim)
    v_scale = _pad_channels(v_scale.reshape(-1).repeat(heads), qkv_dim, head_index)

    for proj, bias_attr in _PROJ_BIAS.items():
        params = lin[proj]
        if f"{proj}_w" in spec:
            buffers[f"{proj}_w"] = _nk(pad_weight(proj, params["w"]))
        name = f"{proj}_dq"
        if name not in spec:
            continue
        bias = getattr(block, bias_attr).detach()
        # LayerScale folds into the following dequant scale (pinned at 1 for Qwen3-VL).
        layer_scale = {"o_proj": block.ls1, "fc2": block.ls2}.get(proj)
        dz, ds = _dequant_table(
            params, bias, None if layer_scale is None else layer_scale.detach().reshape(-1)
        )
        # Padded output channels get ds = 0, so the projection writes zeros there
        # ((acc - 0) * 0) whatever the accumulator holds.
        width = out_width[proj]
        dz, ds = (_pad_channels(v, width, out_index.get(proj)) for v in (dz, ds))
        out_pw = width // 64
        if proj == "v_proj":  # fused SDPA-input quantize (zero-pinned)
            table = _interleave(dz, ds, v_scale, torch.zeros(out_pw * 64))
            buffers[name] = pack(table, spec[name])
        elif proj == "fc1":  # fused GELU + fc2 input quantize
            fc2 = lin["fc2"]
            table = _interleave(
                dz, ds, _pad_channels(fc2["qs"], width), _pad_channels(fc2["qz"], width)
            )
            buffers[name] = pack(table, spec[name])
        else:
            buffers[name] = pack(_interleave(dz, ds), spec[name])

    # Post-RoPE SDPA scales: per head-dim, so they take the WITHIN-head scatter.
    one_head = _head_pad_index(1, head_dim, qkv_dim // heads)
    put("q_attn_qs", _pad_channels(sdpa["qs_q"], qkv_dim // heads, one_head))
    put("k_attn_qs", _pad_channels(sdpa["qs_k"], qkv_dim // heads, one_head))
    put("sdpa_scores_dq_scale", torch.as_tensor(sdpa["scores_dq_scale"]))
    put("sdpa_weights_q_scale", torch.as_tensor(sdpa["qs_w"]))
    put("sdpa_output_dq_scale", torch.as_tensor(sdpa["output_dq_scale"]))
    for name, site in (
        ("ln1_lut", "ln1.inv_sqrt_range"),
        ("ln2_lut", "ln2.inv_sqrt_range"),
        ("gelu_lut", "gelu_range"),
        ("sdpa_exp_lut", "sdpa.exp_range"),
        ("sdpa_inv_lut", "sdpa.reciprocal_range"),
    ):
        if name in spec and site in ranges:
            buffers[name] = fit_lut(ranges[site], spec[name])
    return buffers


def merge_buffers(merger, spec: dict) -> dict:
    """Bind a patch merger: gamma/LUT, merged-width input quant (beta folded), MLP.

    Covers both merger flavours unchanged. The output merger normalises at vision width,
    so its beta repeats across the 4 sub-tokens of a merge block and its gamma is
    D-major; the DeepStack (post-shuffle) mergers normalise at merged width, so the beta
    repeat degenerates to 1 and the gamma is the Kahan norm's broadcast-per-lane layout.
    Both are read off the kernel's own buffer spec.
    """
    lin = export_linear_quant_params(merger)
    ranges = export_lut_ranges(merger)
    fc1, fc2 = lin["linear_fc1"], lin["linear_fc2"]

    beta = merger.norm.bias.detach().to(torch.float32).reshape(-1)
    repeat = merger.linear_fc1.in_features // beta.numel()
    qs = fc1["qs"].to(torch.float32).reshape(-1)
    qz = fc1["qz"].to(torch.float32).reshape(-1) + beta.repeat(repeat) * qs

    gamma = merger.norm.weight.detach().to(torch.float32)
    # (dim, PWORD) spec == the Kahan norm's broadcast weight; (dim/64, PWORD) == D-major.
    if spec["mnorm_w"][0][0] == gamma.numel():
        gamma = broadcast_to_pword(gamma)

    dz1, ds1 = _dequant_table(fc1, merger.fc1_bias.detach())
    dz2, ds2 = _dequant_table(fc2, merger.fc2_bias.detach())
    buffers = {
        "mnorm_w": pack(gamma, spec["mnorm_w"]),
        "mnorm_lut": fit_lut(ranges["norm.inv_sqrt_range"], spec["mnorm_lut"]),
        "mfc1_qparams": pack(_interleave(qs, qz), spec["mfc1_qparams"]),
        "mfc1_w": pack(_nk(fc1["w"]), spec["mfc1_w"]),
        "mfc1_dq": pack(_interleave(dz1, ds1, fc2["qs"], fc2["qz"]), spec["mfc1_dq"]),
        "mgelu_lut": fit_lut(ranges["gelu_range"], spec["mgelu_lut"]),
        "mfc2_w": pack(_nk(fc2["w"]), spec["mfc2_w"]),
        "mfc2_dq": pack(_interleave(dz2, ds2), spec["mfc2_dq"]),
    }
    return buffers


# --------------------------------------------------------------------------------------
# Graph authoring + device driver
# --------------------------------------------------------------------------------------


def compile_kernel(
    module,
    out_dir: Path,
    *,
    target=None,
    cores: int = 1,
    promote: bool = True,
    generation: str = "OMEGA",
):
    """Compile one alto module; returns (model.json, model.atex).

    ``generation`` ("OMEGA" | "EUROPA") changes the core count, the memory sizes and hence
    the schedule, so it must be the same for every kernel of one artifact.

    ``resources`` is pinned to this kernel's fair share of device memory
    (``num_cores / NUM_AI_CORES``) rather than the default 1.0: a VLM keeps TWO artifacts
    loaded on one device, so neither may budget as if it owned all of it. It is also the
    minimum the config validator accepts for a given core count. Buffer promotion stays ON
    (measured: 3.89 vs 4.63 ms per vision block at one core).
    """
    from axelera.compiler.alto.compiler.config import AltoConfig, HardwareGeneration, Target
    from axelera.compiler.alto.compiler.device.hw_specs import NUM_AI_CORES
    from axelera.compiler.alto.compiler.driver import compile as alto_compile

    out_dir = Path(out_dir)
    hw = HardwareGeneration[str(generation).upper()]
    alto_compile(
        module,
        AltoConfig(
            output_dir=out_dir,
            generation=hw,
            target=Target.AICORE_SIM if target is None else target,
            num_cores=cores,
            resources=max(cores, 1) / NUM_AI_CORES[hw],
            enable_tiling_injection=True,
            enable_buffer_promotion=promote,
            enable_llm_support=True,
            assign_unique_input_pools=True,
            unpack_axm_archive=True,
            clean_output_dir=True,
            emit_atex_artifacts=True,
            emit_alto_config=True,
        ),
    )
    model_jsons = sorted(out_dir.rglob("model.json"))
    atexes = sorted(out_dir.rglob("model.atex"))
    if not model_jsons or not atexes:
        raise RuntimeError(f"compile produced no model.json/model.atex under {out_dir}")
    return model_jsons[0], atexes[0]


def build_vision_graph(
    model: "Qwen3VLVisionModel",
    out_dir,
    *,
    target=None,
    cores=1,
    promote=True,
    generation: str = "OMEGA",
):
    """Author the whole tower as one graph: embed -> depth x block -> merge (+ DeepStack).

    The activation threads inputs_embeds -> hidden_a -> hidden_b -> ... -> merged output,
    ping-ponging the two spine slots so depth costs 2 live buffers rather than one per block;
    the padding mask is a shared input to the blocks; every kernel's constants are bound as
    its own outlined weights. ``generation`` goes to every kernel compile and to the graph
    renderer.
    Returns (out_dir, specs).

    The DeepStack mergers sit on the same graph as TAPS (KindContract.tap): each reads the
    hidden state right after its block without consuming it, so one dispatch still produces
    everything the decoder needs -- the merged embeddings plus one feature map per injected
    decoder layer.
    """
    from ...graph_descriptor import KindContract, author_graph_model

    out_dir = Path(out_dir)
    cfg = model.config
    ds_indexes = tuple(cfg.deepstack_visual_indexes)

    def _compile(name, module, ncores=1):
        model_json, atex = compile_kernel(
            module,
            out_dir / f"_{name}",
            target=target,
            cores=ncores,
            promote=promote,
            generation=generation,
        )
        return atex, extract_layer_buffer_spec(model_json)

    # Every kernel gets the core count, not just the block: compiling `embed`, `merge` and
    # the DeepStack mergers at one core left ~12% of a frame single-core (the three
    # DeepStack mergers alone are ~3.8 ms each).
    embed_atex, embed_spec = _compile("embed", make_vision_embed_kernel(cfg, cores), cores)
    block_atex, block_spec = _compile("block", make_vision_block_kernel(cfg, cores), cores)
    merge_atex, merge_spec = _compile("merge", make_vision_merge_kernel(cfg, cores), cores)
    kernel_atex = {
        ("encode", "embed"): embed_atex,
        ("encode", "block"): block_atex,
        ("encode", "merge"): merge_atex,
    }
    contracts = {
        "embed": KindContract(shared_inputs=frozenset({"patches"}), state=frozenset()),
        "block": KindContract(shared_inputs=frozenset({"attn_mask"}), state=frozenset()),
        "merge": KindContract(shared_inputs=frozenset(), state=frozenset()),
    }
    if ds_indexes:
        ds_atex, ds_spec = _compile(
            "deepstack", make_vision_merge_kernel(cfg, cores, postshuffle=True), cores
        )
        kernel_atex[("encode", "deepstack")] = ds_atex
        contracts["deepstack"] = KindContract(tap=True)

    # Kinds and per-position constants in spine order, so a tap lands directly after
    # the block whose output it reads.
    kinds = ["embed"]
    buffers = {0: embed_buffers(model, embed_spec)}
    for i, block in enumerate(model.blocks):
        buffers[len(kinds)] = block_buffers(block, block_spec)
        kinds.append("block")
        if i in ds_indexes:
            buffers[len(kinds)] = merge_buffers(
                model.deepstack_mergers[ds_indexes.index(i)], ds_spec
            )
            kinds.append("deepstack")
    buffers[len(kinds)] = merge_buffers(model.merger, merge_spec)
    kinds.append("merge")

    author_graph_model(
        out_dir,
        "qwen3vl_vision",
        kernel_atex,
        kinds,
        contracts,
        buffers,
        threaded_inputs=frozenset({"hidden_states"}),
        generation=str(generation).lower(),
    )
    # The few scalars a run needs that are not in the graph: the geometry, the host
    # pixel-quant scale, the mask shape and the generation this was built for. With these the
    # artifact reloads and runs with no torch model -- see Qwen3VLVisionDevice.load.
    (out_dir / "meta.json").write_text(
        json.dumps(
            {
                "config": asdict(cfg),
                "patch_scale": float(
                    model.patch_embed.proj.act_observer.qparams()[0].reshape(-1)[0]
                ),
                "mask_shape": list(block_spec["attn_mask"][0]),
                "generation": str(generation).upper(),
                "cores": int(cores),
            },
            indent=2,
        )
    )
    specs = {"embed": embed_spec, "block": block_spec, "merge": merge_spec}
    if ds_indexes:
        specs["deepstack"] = ds_spec
    return out_dir, specs


class Qwen3VLVisionDevice:
    """The compiled vision tower: flattened patches -> merged image embeddings.

    ``encode`` is one device dispatch returning the merged embeddings AND the DeepStack
    feature maps (the tap outputs, in ``deepstack_visual_indexes`` order). The host does
    the int8 patch quantize (one calibrated symmetric scale, the same the patch-embed
    matmul was calibrated with) and slices the padded rows off every output.
    """

    def _bind(self, graph_dir, config, patch_scale, mask_shape, generation, share_from=None):
        """Everything both construction paths set: geometry, padding mask, loaded runtime.

        The connection claims one sub-device per core the graph places work on, which
        RuntimeModel reads off the artifact -- unless it is sharing one, the first tower
        loaded having claimed them already.
        """
        from ...generate import RuntimeModel

        self.graph_dir = Path(graph_dir)
        self.config = config
        self.generation = str(generation).upper()
        self.patch_scale = float(patch_scale)
        self.mask_shape = tuple(mask_shape)
        self._mask = self._padding_mask()
        self.runtime = RuntimeModel(self.graph_dir / "model.json", share_from=share_from)
        # What the last encode cost, readable before one has run (a prompt with no
        # picture reports a zero frame rather than an AttributeError).
        self.last_encode: tuple[int, float] = (0, 0.0)
        self.last_quantize_s = 0.0
        self.last_dispatch_s: list[float] = []

    def __init__(
        self, model: "Qwen3VLVisionModel", out_dir, *, target=None, cores=1, generation="OMEGA"
    ):
        graph_dir, specs = build_vision_graph(
            model, out_dir, target=target, cores=cores, generation=generation
        )
        self._bind(
            graph_dir,
            model.config,
            model.patch_embed.proj.act_observer.qparams()[0].reshape(-1)[0],
            specs["block"]["attn_mask"][0],
            generation,
        )

    @classmethod
    def load(cls, graph_dir, *, share_from=None, generation="OMEGA"):
        """Reload a compiled tower (graph + meta.json) with no torch model or calibration.

        The core count comes off meta.json -- what the tower was compiled for is what the
        connection claims -- so a run needs no flag to repeat it.
        """
        from .qwen3vl import Qwen3VLVisionConfig

        meta = json.loads((Path(graph_dir) / "meta.json").read_text())
        self = cls.__new__(cls)
        self._bind(
            graph_dir,
            Qwen3VLVisionConfig(**meta["config"]),
            meta["patch_scale"],
            meta["mask_shape"],
            generation,
            share_from,
        )
        return self

    @property
    def num_deepstack(self) -> int:
        """How many DeepStack feature maps this compiled tower emits (its tap outputs).

        Read off the artifact rather than the config: a tower compiled before DeepStack
        existed has none, and the pipeline has to notice that rather than assume.
        """
        return sum(name != "output" for name in self.runtime.output_names("encode"))

    def _padding_mask(self) -> np.ndarray:
        """int8 1-head-broadcast mask: 0 for real patches, -127 for padding."""
        shape = self.mask_shape  # (1, 1, seq, ctx_blocks, 64)
        keys = np.arange(shape[-2] * 64).reshape(shape[-2], 64)
        row = np.where(keys < self.config.num_patches, 0, _MASK_NEG).astype(np.int8)
        return np.broadcast_to(row, shape).copy()

    def quantize_patches(self, patches) -> np.ndarray:
        """(num_patches, patch_dim) float -> (1, num_patches, patch_dim/64, 64) int8."""
        cfg = self.config
        values = torch.as_tensor(np.asarray(patches)).float()
        if values.shape != (cfg.num_patches, cfg.patch_dim):
            raise ValueError(
                f"expected {(cfg.num_patches, cfg.patch_dim)} patches for the compiled "
                f"{cfg.grid_h}x{cfg.grid_w} grid; got {tuple(values.shape)}"
            )
        quantized = torch.clamp(torch.round(values / self.patch_scale), -127, 127).to(torch.int8)
        return quantized.numpy().reshape(1, cfg.num_patches, cfg.patch_dim // 64, 64)

    def encode(self, patches) -> tuple[np.ndarray, list[np.ndarray]]:
        """Patches -> (merged embeddings, DeepStack features), all (tokens, out_hidden).

        The processor concatenates every image's patches, so N images arrive as
        N * num_patches rows; each is one dispatch (the graph is compiled for one
        image's token count, and HF blocks vision attention per image anyway), and
        the results concatenate in image order -- which is the order the decoder's
        image-placeholder rows expect.

        The dispatch returns one buffer per graph output: the tap outputs (the DeepStack
        mergers, in tower order) and then the merged embeddings. They are told apart by
        name rather than by position.
        """
        from time import perf_counter

        cfg = self.config
        rows = np.asarray(patches)
        if rows.shape[0] % cfg.num_patches:
            raise ValueError(
                f"{rows.shape[0]} patch rows is not a whole number of "
                f"{cfg.num_patches}-patch images"
            )
        names = self.runtime.output_names("encode")
        per_image = []
        # (dispatches, seconds) of the last encode -- the tower's own cost, separate from
        # whatever host work the caller wraps it in. Read by the demo's RunStats.
        self.last_encode = (0, 0.0)
        # The two costs that used to hide inside that one number: the host int8 patch
        # quantize (a frame is billed to the accelerator, so the host share must be
        # separable) and the per-dispatch times, so a multi-image prompt can be read on a
        # median rather than a mean the cold first dispatch drags around. Separate
        # attributes, so existing `last_encode` readers are unaffected.
        self.last_quantize_s = 0.0
        self.last_dispatch_s: list[float] = []
        started = perf_counter()
        for start in range(0, rows.shape[0], cfg.num_patches):
            quantize_started = perf_counter()
            patches = self.quantize_patches(rows[start : start + cfg.num_patches])
            self.last_quantize_s += perf_counter() - quantize_started
            dispatch_started = perf_counter()
            outs = self.runtime.run(
                "encode",
                {"patches": patches, "attn_mask": self._mask},
            )
            self.last_dispatch_s.append(perf_counter() - dispatch_started)
            outs = outs if isinstance(outs, list) else [outs]
            per_image.append(
                {
                    name: out.reshape(-1, cfg.out_hidden_size)[: cfg.num_merged]
                    for name, out in zip(names, outs)
                }
            )
        self.last_encode = (len(per_image), perf_counter() - started)
        merged = np.concatenate([image["output"] for image in per_image], axis=0)
        deepstack = [
            np.concatenate([image[name] for image in per_image], axis=0)
            for name in names
            if name != "output"
        ]
        return merged, deepstack

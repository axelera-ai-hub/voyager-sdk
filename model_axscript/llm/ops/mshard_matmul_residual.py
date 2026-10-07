# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""M-sharded quantized matmul with a banded residual add: one fused tail across cores.

Split the row (M) axis across cores and give each core its contiguous band: the matmul
for those rows, then the residual add for those same rows. Rows are independent, so
there is no cross-core reduction, no split output and no un-split -- and the residual
add is naturally banded, so it parallelises for free instead of running as one
full-width op afterwards.

The weight is the whole point of the M-shard: every core needs ALL of it (only the rows
of A are split), so it is staged into shared L2 once rather than re-read per core. That
is also the limit of the trick -- it only pays once each core's band is large enough to
amortize reading the full weight (dino measures +1.2% at seq 1408, -4.7% at seq 320).

Written for a ViT block's ``fc2``/``o_proj`` tail (models/dino, models/qwen3vl) and used
by the decoder's ``o_proj`` (models/qwen3), which is why it lives here rather than in a
model file. Prefill only in a decoder: decode has M = 1 and nothing to split.
"""

# flake8: noqa: F821


def mshard_matmul_residual(
    matrix_a,
    matrix_b,
    dequant_params,
    residual,
    *,
    seq,
    n_blocks,
    cores,
    matrix_a_is_split=False,
    resid_dtype,
    out_dtype,
):
    """``out[:, lo:hi] = residual[:, lo:hi] + (matrix_a[:, lo:hi] @ matrix_b)`` per core.

    ``seq`` is the row count (must divide by ``cores``), ``n_blocks`` the output's PWORD
    block count; ``matrix_a_is_split`` passes through to the matmul for a producer that
    wrote the split (per-core K band) layout. ``cores`` in {None, 0, 1} emits the plain
    matmul + add. Runs during tracing (called from a prim_func body), so the untaken
    branch is never parsed as TVMScript. Returns the full ``(1, seq, n_blocks)`` output.
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol import atomic
    from axelera.compiler.alto.axol.common import BufferInfo
    from tvm import tir

    from .quantized_matmul_activation import (
        _b_l2_stage_max_bytes,
        slice_n,
        quantized_matmul_activation,
    )

    kw_split = {"matrix_a_is_split": True} if matrix_a_is_split else {}
    if not cores or cores <= 1:
        out = quantized_matmul_activation(
            matrix_a, matrix_b, dequant_params, output=BufferInfo(dtype=resid_dtype), **kw_split
        )
        return axol.atomic.add(residual, out, output=BufferInfo(dtype=out_dtype))

    core_ids = tuple(range(cores))
    band = seq // cores
    out = A.alloc_buffer((1, seq, n_blocks), dtype=out_dtype, scope=A.DDR)
    out_bs = tir.BufferSlice(out)
    w_bytes = int(matrix_b.shape[-2]) * int(matrix_b.shape[-1]) * 64  # int8x64: 64 B/elem
    stage = w_bytes <= _b_l2_stage_max_bytes()
    if stage:
        w_l2 = A.alloc_buffer(
            tuple(int(s) for s in matrix_b.shape), dtype=matrix_b.dtype, scope=A.L2
        )

    for idx, core in enumerate(core_ids):
        lo, hi = idx * band, (idx + 1) * band
        with A.AICore(core=core):
            A.barrier(*core_ids)
            w = matrix_b
            if stage:
                if idx == 0:
                    atomic.copy(matrix_b, scope=A.L2, output=w_l2)
                A.barrier(*core_ids)  # order the single DDR->L2 stage before any core reads it
                w = w_l2
            a_band = slice_n(matrix_a, 1, lo, hi)  # slice the seq/M axis
            out_band = quantized_matmul_activation(
                a_band,
                w,
                dequant_params,
                output=BufferInfo(dtype=resid_dtype),
                core=core,
                stage_weight_l2=False,
                **kw_split,
            )
            axol.atomic.add(
                slice_n(residual, 1, lo, hi), out_band, output=slice_n(out_bs, 1, lo, hi)
            )
            A.barrier(*core_ids)
    return out

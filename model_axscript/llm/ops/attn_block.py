#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""The decode attention block, o_proj and the MLP as ONE core region per core.

Per core: the k/q norm+RoPE+quant and the v quant for ITS heads, ITS KV-cache slot write,
SDPA (band schedule) over ITS KV group, its o_proj N band -- with the o_proj weights streamed
into a region-local L1 buffer while the attention work runs, so o_proj programs the IMC from
resident weights and streams only what is left -- then gate_up and down_proj, with down's tile
0 streamed during gate_up's last tile. Each core's KV group is self-contained up to o_proj, so
no barrier separates the glue from SDPA, and the DDR channels that idle during the glue and the
softmax/PV compute carry the o_proj band instead.

Channel discipline (the compiler keeps a per-unit FIFO of software-waited DMAs -- alto
``get_token_task_map`` -- and a wait retires the OLDEST launch on its channel, so a DMA that
some op waits on must never sit behind a foreign launch on its channel):
  * during the glue and SDPA's operand staging every op DMA is on channel 0, so the first
    o_proj parts stream on channel 1 and are waited right before SDPA launches V on it;
  * between SDPA's K/V landing and its output copy no op DMA is issued, so more parts stream on
    both channels and are waited before that copy;
  * two more parts go out on channel 1 across the o6 all-gather barrier (o_proj's own prologue
    copies are channel 0's) and o_proj waits them at its core scope;
  * o_proj queues the remaining parts on both channels and waits each tile's before its program.
Even cores start with the DDR1 shards, odd cores with the DDR0 shards, so both regions stream.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from axelera.compiler.alto.axol import atomic
from axelera.compiler.alto.axol.common import sliced
from axelera.compiler.alto.axol.core.matmul._common import shards
import axelera.compiler.alto.script as A
from tvm import tir

from .fused_gate_up_swiglu import FusedGateUpSwiGLU, fused_gate_up_swiglu, launch_gu_tile0
from .quantize import quantize
from .quantized_matmul_activation import (
    ShardPlan,
    bs,
    caller_buffer,
    core_ids,
    inflight_part_id,
    slice_n,
    stream_shard,
)
from .rms_norm_dpu_reduce import rms_norm_dpu_reduce
from .sdpa_band import sdpa_band
from .update_kv_cache_2d import update_kv_cache_2d

# o_proj parts (one shard each, two per tile) streamed on channel 1 under the glue, on both
# channels under SDPA, and on channel 1 across the o6 barrier; a narrower band truncates the
# later lists.
_PARTS_GLUE = 3
_PARTS_SDPA = 6
_PARTS_PRE_OPROJ = 2


@dataclass(frozen=True, eq=False)
class MlpOperands:
    """What one layer's MLP block computes on. ``matrix_a`` only carries gate_up's activation
    shape: ``a_prologue(core)`` computes the activation inside the core's region.
    ``gate_up_output`` and ``output`` (down_proj's) are caller-owned buffers."""

    matrix_a: Any
    gate_up_w: Any
    gate_up_dq: Any
    silu_lut: Any
    gate_up_w1: Any
    down_w: Any
    down_dq: Any
    down_w1: Any
    gate_up_output: Any
    output: Any
    a_prologue: Callable[[int], Any]


def _gate_up_probe(matrix_a, gate_up_w, gate_up_dq, silu_lut, gate_up_w1) -> FusedGateUpSwiGLU:
    return FusedGateUpSwiGLU(
        matrix_a, gate_up_w, gate_up_dq, matrix_b1=gate_up_w1, luts=silu_lut, pipeline_weights=True
    )


@sliced
def gate_up_output_info(matrix_a, gate_up_w, gate_up_dq, silu_lut, gate_up_w1):
    """(shape, dtype) of gate_up's split output, for a caller that owns it."""
    info = _gate_up_probe(matrix_a, gate_up_w, gate_up_dq, silu_lut, gate_up_w1).infer_outputs()
    return tuple(int(x) for x in info["output"].shape), info["output"].dtype


def _bands(total: int, cores) -> list:
    """``shards`` over every core: (core, lo, hi) per core."""
    parts = list(shards(total, cores))
    if len(parts) != len(cores):
        raise ValueError(f"{total} cannot be sharded over the cores {cores}")
    return parts


@sliced
def fused_attn_oproj_multicore(
    q_heads,
    k_heads,
    v_heads,
    k_cache,
    v_cache,
    seq_pos,
    rope_trig,
    sdpa_consts,
    o_qparams,
    kvq,
    attn_out,
    o_w,
    o_dq,
    *,
    o_w1,
    cores,
    output,
    k_norm_consts,
    k_norm_w,
    k_cache_qparams,
    v_qparams,
    q_norm_consts,
    q_norm_w,
    q_attn_qparams,
    head_dim,
    scale,
    mlp: MlpOperands,
    tail_launch,
    tail_body,
    packed_consts=None,
):
    """Attention glue + KV write + SDPA (band) + o_proj + MLP, one region per core; returns
    down_proj's output, the caches updated in place.

    ``k_cache``/``v_cache`` are the caches the slot write targets and SDPA reads.
    ``q_heads``/``k_heads``/``v_heads`` are the (1, heads, 1, D) views of the projections;
    ``sdpa_consts`` = (scores_dq, exp_lut, inv_lut, wq_scale, wq_zp,
    wq_qmin, wq_qmax, out_dq, q_zp, v_zp, attn_mask) in the core op's order; ``kvq`` is the
    caller's (3, heads, 1, D) int8 DDR scratch (plane 0 = q, 1 = k, 2 = v; k/v use the first
    kv-head rows); ``attn_out`` (1, heads, 1, D) int8 and ``output`` (1, 1, hidden_pw) are
    caller-owned. ``tail_launch(core, part_idx)`` is emitted after the down band before its
    closing barrier and ``tail_body(core, part_idx, n_lo, n_hi, d6)`` after that barrier, still
    inside the core's region; ``n_lo``/``n_hi`` is the core's down_proj N band.
    """
    all_cores = core_ids(cores)
    if len(all_cores) <= 1:
        raise ValueError("fused_attn_oproj_multicore needs more than one core")
    num_q_heads, hd_pw = int(q_heads.shape[1]), int(q_heads.shape[3])
    num_kv_heads = int(k_heads.shape[1])
    if num_kv_heads != len(all_cores) or num_q_heads % num_kv_heads:
        raise ValueError(
            f"the fused attention region runs one KV head per core ({num_kv_heads} KV heads on "
            f"{len(all_cores)} cores) and needs the query heads ({num_q_heads}) to divide by them"
        )
    if o_w1 is None or mlp.gate_up_w1 is None or mlp.down_w1 is None:
        raise ValueError(
            "the fused attention region streams every weight from two K shards (DDR and "
            "DDR1): o_w1, mlp.gate_up_w1 and mlp.down_w1 are required"
        )
    kv_parts = _bands(num_kv_heads, all_cores)
    kv_groups = num_q_heads // num_kv_heads

    # o_proj: its activation is SDPA's output viewed as one row.
    attn_flat = atomic.reshape(attn_out, shape=(1, 1, num_q_heads * hd_pw))
    plan = ShardPlan(
        attn_flat, o_w, o_dq, output, False, dict(matrix_b1=o_w1, pipeline_weights=True), all_cores
    )
    o_parts = _bands(plan.n_blocks, all_cores)
    k_ext0 = int(o_w.shape[-1])

    # The glue scratch (q plane for SDPA's query, k/v planes for the cache write), indexed by
    # global head as the multicore SDPA does.
    kvq_buf, k_cache, v_cache, v_heads_buf = (
        caller_buffer(b) for b in (kvq, k_cache, v_cache, v_heads)
    )
    if None in (kvq_buf, k_cache, v_cache, v_heads_buf):
        raise ValueError("kvq, k_cache, v_cache and v_heads must be whole buffers")
    kvq_heads = atomic.reshape(kvq, shape=(1, 3 * num_q_heads, 1, hd_pw))
    one, d_all = tir.Ramp(0, 1, 1), tir.Ramp(0, 1, hd_pw)
    q_all = bs(kvq_buf, one, tir.Ramp(0, 1, num_q_heads), one, d_all)

    def _plane(plane, lo, hi):
        return bs(kvq_buf, tir.Ramp(plane, 1, 1), tir.Ramp(lo, 1, hi - lo), one, d_all)

    def _rows(buf, plane, h):
        return bs(buf, plane, h, one, d_all)

    # The MLP's tiling: gate_up's tile bands, down_proj's shard plan and N bands, and the
    # (pair_per_tile, K) slot gate_up's tile 0 is streamed into.
    gu = _gate_up_probe(mlp.matrix_a, mlp.gate_up_w, mlp.gate_up_dq, mlp.silu_lut, mlp.gate_up_w1)
    gu_tiles, _ = gu.n_outer_tiles()
    pair_per_tile, gu_k = int(mlp.gate_up_w.shape[-2]) // gu_tiles, int(gu._b_k())
    down = ShardPlan(
        mlp.gate_up_output,
        mlp.down_w,
        mlp.down_dq,
        mlp.output,
        False,
        dict(matrix_b1=mlp.down_w1, matrix_a_is_split=True, pipeline_weights=True),
        all_cores,
    )
    gu_parts = _bands(gu_tiles, all_cores)
    down_parts = _bands(down.n_blocks, all_cores)

    def _core_region(part_idx, core, kv_lo, kv_hi):
        _, n_lo, n_hi = o_parts[part_idx]
        n_per_tile, k = plan.tile_plan(core, part_idx, n_lo, n_hi)
        band_rows = n_hi - n_lo
        if band_rows % n_per_tile:
            raise ValueError(
                f"o_proj band {band_rows} is not a whole number of {n_per_tile}-row tiles"
            )
        num_tiles = band_rows // n_per_tile
        q_lo, q_hi = kv_lo * kv_groups, kv_hi * kv_groups
        # Parts as 2 * tile + shard, tile-major; even cores take a tile's DDR1 shard first, odd
        # cores its DDR0 shard, so both regions stream while every core follows the same
        # schedule.
        first, second = (1, 0) if core % 2 == 0 else (0, 1)
        order = [2 * t + sh for t in range(num_tiles) for sh in (first, second)]
        glue_parts = order[:_PARTS_GLUE]
        sdpa_parts = order[_PARTS_GLUE : _PARTS_GLUE + _PARTS_SDPA]
        pre_parts = order[_PARTS_GLUE + _PARTS_SDPA : _PARTS_GLUE + _PARTS_SDPA + _PARTS_PRE_OPROJ]

        with A.AICore(core=core):
            resident = A.alloc_buffer((band_rows, k), dtype=o_w.dtype, scope=A.L1)

            def launch(part, channel):
                t, sh = divmod(part, 2)
                src, k_off = (o_w, 0) if sh == 0 else (o_w1, k_ext0)
                rows = tir.Ramp(n_lo + t * n_per_tile, 1, n_per_tile)
                dst = resident.tile(tir.Ramp(t * n_per_tile, 1, n_per_tile), dim=0)
                stream_shard(
                    src,
                    rows,
                    dst,
                    k_off,
                    core=core,
                    channel=channel,
                    id=inflight_part_id(core, part),
                )

            def wait(part):
                A.waits(inflight_part_id(core, part))

            # --- o_proj parts on channel 1, under the glue (whose DMAs are all on channel 0) ---
            for part in glue_parts:
                launch(part, 1)

            # --- the glue for this core's heads: v quant, k chain, q chain, then the cache
            # write of the k/v planes the datapaths just wrote ---
            for h in range(kv_lo, kv_hi):
                quantize(_rows(v_heads_buf, 0, h), v_qparams, out=_rows(kvq_buf, 2, h))
            rms_norm_dpu_reduce(
                slice_n(k_heads, 1, kv_lo, kv_hi),
                k_norm_consts,
                normalized_shape=(int(head_dim),),
                weight_buffer=k_norm_w,
                rope_trig=rope_trig,
                quant_params=k_cache_qparams,
                invsqrt_method="additive",
                output=_plane(1, kv_lo, kv_hi),
            )
            rms_norm_dpu_reduce(
                slice_n(q_heads, 1, q_lo, q_hi),
                q_norm_consts,
                normalized_shape=(int(head_dim),),
                weight_buffer=q_norm_w,
                rope_trig=rope_trig,
                quant_params=q_attn_qparams,
                invsqrt_method="additive",
                output=_plane(0, q_lo, q_hi),
            )
            update_kv_cache_2d(
                kvq_heads,
                kvq_heads,
                k_cache,
                v_cache,
                seq_pos,
                core=core,
                head_lo=kv_lo,
                heads=kv_hi - kv_lo,
                k_src_head_lo=num_q_heads + kv_lo,
                v_src_head_lo=2 * num_q_heads + kv_lo,
            )

            # --- SDPA over this core's KV group, o_proj parts streamed around it ---
            def wait_glue():
                for part in glue_parts:  # channel 1 must be clear before V's launch
                    wait(part)

            def launch_sdpa():
                for i, part in enumerate(sdpa_parts):
                    launch(part, i % 2)

            def wait_sdpa():
                for part in sdpa_parts:
                    wait(part)

            sdpa_band(
                q_all,
                k_cache,
                v_cache,
                *sdpa_consts,
                scale=scale,
                attention_output_quantizer=o_qparams,
                output=attn_out,
                band=(core, kv_lo, kv_hi),
                before_kv=wait_glue,
                kv_landed=launch_sdpa,
                end=wait_sdpa,
                packed_consts=packed_consts,
            )
            # More parts on channel 1 across the o6 all-gather barrier (o_proj's own prologue
            # copies are on channel 0); o_proj waits them before their tiles' programs.
            for part in pre_parts:
                launch(part, 1)
            # o_proj reads every core's SDPA output.
            A.barrier(*all_cores)
            plan.emit(
                core,
                part_idx,
                n_lo,
                n_hi,
                resident=resident,
                resident_parts=tuple(glue_parts + sdpa_parts),
                inflight_parts=tuple(pre_parts),
            )

            # --- the MLP: gate_up's tile 0 streams under o_proj's output write, the o6
            # all-gather barrier and the region overhead; it is waited before the norm2
            # prologue (whose own DMAs would otherwise share the channels) ---
            gu_slot = A.alloc_buffer((pair_per_tile, gu_k), dtype=mlp.gate_up_w.dtype, scope=A.L1)
            _, t_lo, t_hi = gu_parts[part_idx]
            launch_gu_tile0(
                mlp.gate_up_w, mlp.gate_up_w1, t_lo * pair_per_tile, pair_per_tile, gu_slot, core
            )
            A.barrier(*all_cores)
            _, dn_lo, dn_hi = down_parts[part_idx]
            dn_per_tile, dn_k = down.tile_plan(core, part_idx, dn_lo, dn_hi)
            dn_slot = A.alloc_buffer((dn_per_tile, dn_k), dtype=mlp.down_w.dtype, scope=A.L1)
            fused_gate_up_swiglu(
                mlp.matrix_a,
                mlp.gate_up_w,
                mlp.gate_up_dq,
                matrix_b1=mlp.gate_up_w1,
                luts=mlp.silu_lut,
                pipeline_weights=True,
                output=mlp.gate_up_output,
                core=core,
                band_tiles=(t_lo, t_hi),
                a_prologue=mlp.a_prologue,
                prefetched=gu_slot,
                xop_b=mlp.down_w,
                xop_b1=mlp.down_w1,
                xop_rows=(dn_lo, dn_per_tile),
                xop_slot=dn_slot,
            )
            A.barrier(*all_cores)
            down.emit(core, part_idx, dn_lo, dn_hi, prefetched=dn_slot)
            # After the down band, before its closing barrier: every DMA of the down op has
            # been waited, so the tail's launch streams through the barrier skew.
            tail_launch(core, part_idx)
            A.barrier(*all_cores)
            tail_body(core, part_idx, dn_lo, dn_hi, down.out)

    for part_idx, (core, kv_lo, kv_hi) in enumerate(kv_parts):
        _core_region(part_idx, core, kv_lo, kv_hi)
    return down.out

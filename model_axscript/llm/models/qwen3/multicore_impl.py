# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Qwen3 Metis multicore implementation (registered as qwen3-metis-multicore).

The optimized, multicore-capable Qwen3 decoder layer. The host-side contract, the
fused norm/quantize helpers, and the buffer-size resolution are shared via
``models.multicore_layer``; this file supplies the Qwen3 kernel op sequence (Qwen3 has
per-head q/k RMSNorm) and buffer packer. Standard RoPE (the backbone default).

cores>1 decode on Europa runs one fused region per core: attention glue, KV write, band
SDPA, o_proj, MLP and the next layer's norm+qkv (ops/attn_block.py),
``default_layers_per_kernel`` layers per kernel (stacked_kernel.py). Prefill on Europa
row-shards the glue over the cores. Omega and single-core builds run the segment-per-op
body.
"""

# flake8: noqa: F821  -- A.Buffer dtype-string annotations look like F821 names

from types import SimpleNamespace
from typing import TYPE_CHECKING

from ...implementation import register_implementation
from ..multicore_layer import (
    PWORD,
    HIDDEN_NORM_FOLDS,
    INPUT_NORM_FOLD,
    POST_ATTN_NORM_FOLD,
    ModelSpec,
    MulticoreDecoderImplementation,
    add_norm_consts,
    check_norm_folds,
    ddr1_split_shards,
    ddr1_unsharded_spec,
    hidden_norm_quant,
    interleave_qparams,
    layer_dims,
    quantize_activation,
    upper_shard_scope,
)
from .optimized_impl import optimized_decode_layer_buffers
from .stacked_kernel import (
    SDPA_LUT_ROWS,
    default_layers_per_kernel,
    grouped_module,
    layer_params,
    shared_params,
)

if TYPE_CHECKING:
    from axelera.compiler.alto.compiler.config import HardwareGeneration

    from .qwen3 import Qwen3Config

QWEN3_SPEC = ModelSpec(has_qk_norm=True, has_lm_head=False)


def _qkv_band_spans_three(num_q_heads: int, num_kv_heads: int, cores: int) -> bool:
    """Whether a core's equal share of q||k||v holds q, ALL of k and v (head units):
    ``FusedQkvPlan`` emits a band over at most two projections, so such a geometry (32 q /
    8 kv heads on 2 cores) keeps the segment-per-op body."""
    total = num_q_heads + 2 * num_kv_heads
    if total % cores:
        return False
    band = total // cores
    q_end, k_end = num_q_heads, num_q_heads + num_kv_heads
    return any(c * band < q_end and k_end < (c + 1) * band for c in range(cores))


def make_qwen3_layer_kernel(
    config: "Qwen3Config",
    ctx_len: int,
    phase: str,
    chunk_len: int | None = None,
    cores: int = 1,
    layers_per_kernel: int | None = None,
    generation: "HardwareGeneration | None" = None,
):
    """Build the QwenLayerFunctional IRModule of one or more Qwen3 decoder layers.

    ctx_len sizes the KV cache / mask; phase is "decode" (seq_len 1) or "prefill"
    (seq_len = chunk_len or ctx_len). Chunked prefill builds the query side for
    chunk_len positions while the KV cache stays full ctx_len; the host threads the
    cache via seq_pos. cores>1 distributes SDPA heads + MLP matmuls across AI cores.

    layers_per_kernel (decode only) stacks that many consecutive layers into the kernel
    (see stacked_kernel.grouped_module); the default is ``default_layers_per_kernel`` for
    the fused multicore schedule and 1 otherwise.
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol.common import BufferInfo
    from axelera.compiler.alto.axol.utils.lut_chain import ChainedLUTMethod
    from axelera.compiler.alto.compiler.config import HardwareGeneration, get_alto_config
    import numpy as np
    from tvm import tir

    from ...ops.add3 import add3
    from ...ops.attn_block import MlpOperands, fused_attn_oproj_multicore, gate_up_output_info
    from ...ops.fused_gate_up_swiglu import fused_gate_up_swiglu
    from ...ops.quantize import quantize
    from ...ops.quantized_matmul_activation import (
        FusedQkvPlan,
        bs,
        caller_buffer,
        quantized_matmul_activation_multicore,
        quantized_matmul_activation_multicore_fused,
        slice_n,
    )
    from ...ops.rms_norm_dpu_reduce import rms_norm_dpu_reduce
    from ...ops.sdpa_band import PACKED_MAX_BLOCKS, packed_rows

    d = layer_dims(
        config,
        head_dim_pw=config.head_dim // PWORD,
        logical_head_dim=config.head_dim,
        ctx_len=ctx_len,
        phase=phase,
        chunk_len=chunk_len,
        spec=QWEN3_SPEC,
    )
    seq_len, hidden_dim_pw = d.seq_len, d.hidden_dim_pw
    head_dim, head_dim_pw = config.head_dim, d.head_dim_pw
    num_q_heads, num_kv_heads, context_len = d.num_q_heads, d.num_kv_heads, d.context_len
    q_out_dim_pw, kv_out_dim_pw = d.q_out_dim_pw, d.kv_out_dim_pw
    generation = generation or get_alto_config().generation
    # Omega has one DDR region, so the upper shards fall back into it (see upper_shard_scope).
    _ddr1 = upper_shard_scope(generation)
    _multicore = cores > 1
    _europa = generation == HardwareGeneration.EUROPA
    # The N-tile / head sharding of the ops that take one; the projections take ``cores``
    # outright (a one-core shard is the plain op).
    _cores_kw = {"cores": cores} if _multicore else {}
    # Every core owns exactly one KV group: the attention region's channel discipline
    # (ops/attn_block.py) holds for a one-head band, and a second head's copies retire the
    # o_proj parts' tokens ahead of its own K/V. Its fused q||k||v band may straddle at most
    # two projections. Omega and other core counts keep the segment-per-op body.
    _fused_decode = (
        _multicore
        and phase == "decode"
        and _europa
        and d.use_dpu_hidden_norm
        and num_kv_heads == cores
        and num_q_heads % num_kv_heads == 0
        and not _qkv_band_spans_three(num_q_heads, num_kv_heads, cores)
    )
    # Prefetch tile i+1's weights under tile i's datapath. Decode only: at tiled M the
    # per-M-tile A reload contends for the token line the prefetch holds open.
    _pipeline_weights = phase == "decode"
    # N-shard q/k/v and o into a unified output. At M > 1 the band write is one DMA
    # descriptor per row, which Omega pays for (so it shards at decode only) but Europa's 2D
    # DMA takes in one transfer.
    _proj_cores = cores if _multicore and (phase == "decode" or _europa) else 1
    _qkv_splits = (q_out_dim_pw, kv_out_dim_pw, kv_out_dim_pw)
    # Europa prefill glue (hidden norms, head chains, transposes, v quant, residual adds)
    # row-sharded over the cores; Omega keeps the single-core ops (its per-row DMA
    # descriptors would eat the gain).
    _prefill_glue = _multicore and _europa and phase == "prefill" and seq_len % cores == 0
    _core_ids = tuple(range(cores))
    _band = seq_len // cores

    def _qkv_args(p):
        """The qkv projection's operands: q||k||v stacked on N, so sharding the concatenated
        N gives every core one weight stream, one dequant copy and one IMC program."""
        return {
            "matrix_b": p.qkv_proj_w,
            "dequant_params": (p.q_proj_dq, p.k_proj_dq, p.v_proj_dq),
            "cores": _proj_cores,
            "splits": _qkv_splits,
            "matrix_b1": p.qkv_proj_w1,
            "pipeline_weights": _pipeline_weights,
        }

    def _project_qkv(activation, p, prologue=None):
        """(q5, k5, v5) from the normed activation, as ONE N-concatenated matmul; at cores=1
        three unsharded matmuls over N-slices of the same weight."""
        # pinned: region-written.
        outputs = (
            tuple(_region_output((1, seq_len, n), "float16x64") for n in _qkv_splits)
            if _prefill_glue
            else (BufferInfo(dtype="float16x64"),) * 3
        )
        return quantized_matmul_activation_multicore_fused(
            activation, outputs=outputs, prologue=prologue, **_qkv_args(p)
        )

    def _sharded_proj_output():
        """The output of a sharded projection whose consumer is the sharded residual add:
        region-written and region-read, so pinned."""
        if _prefill_glue:
            return _region_output((1, seq_len, hidden_dim_pw), "float32x64")
        return BufferInfo(dtype="float32x64")

    def _final_residual(residual, x):
        """A residual add: row-sharded over the cores at prefill, one op otherwise."""
        if not _prefill_glue:
            return axol.atomic.add(residual, x)
        out = _region_output((1, seq_len, hidden_dim_pw), "float32x64")
        ob = tir.BufferSlice(out)
        _row_sharded(
            lambda lo, hi: axol.atomic.add(
                slice_n(residual, 1, lo, hi),
                slice_n(x, 1, lo, hi),
                output=slice_n(ob, 1, lo, hi),
            )
        )
        return out

    def _row_sharded(body):
        """Run ``body(lo, hi)`` on every core's band of positions, between one barrier pair."""
        for idx, core in enumerate(_core_ids):
            with A.AICore(core=core):
                A.barrier(*_core_ids)
                body(idx * _band, (idx + 1) * _band)
                A.barrier(*_core_ids)

    def _shape_only(shape, dtype):
        """A buffer that carries a shape for an op's inference and is never touched."""
        return tir.decl_buffer(shape, dtype, name="shape_only", scope=A.DDR)

    def _norm_quant_l1(x, consts, qparams, fold):
        """The hidden norm+quant into an L1 buffer of the core whose region this runs in."""
        out = A.alloc_buffer((1, seq_len, hidden_dim_pw), "int8x64", scope=A.L1)
        hidden_norm_quant(x, consts, qparams, d, fold, output=out)
        return out

    # A buffer written and read only inside core regions is pinned so the memory analyzer
    # sees its live range: one PWORD written at root scope on allocation and read again at
    # root scope right after its last consumer (`_release`).
    _pinned: dict = {}

    def _region_output(shape, dtype, marker_src=None):
        """``marker_src``: a same-dtype buffer whose first PWORD is copied in as the pin
        write (one DMA) instead of the datapath fill."""
        out = A.alloc_buffer(shape, dtype, scope=A.DDR)
        first = tir.BufferSlice(tir.BufferLoad(out, [0] * (len(shape) - 1) + [tir.Ramp(0, 1, 1)]))
        if marker_src is None:
            axol.atomic.full((1,), 0, output=first)
        else:
            src = tir.BufferSlice(
                tir.BufferLoad(marker_src, [0] * (len(marker_src.shape) - 1) + [tir.Ramp(0, 1, 1)])
            )
            axol.atomic.copy(src, output=first)
        _pinned[id(out)] = first
        return out

    def _release(*bufs):
        """End the live range of pinned buffers (no-op for anything else)."""
        for buf in bufs:
            first = _pinned.pop(id(buf), None)
            if first is not None:
                axol.atomic.copy(first, scope=A.L1)

    def _pin_region_outputs():
        for first in list(_pinned.values()):
            axol.atomic.copy(first, scope=A.L1)
        _pinned.clear()

    def _hidden_norm_quant(x, consts, qparams, fold):
        """hidden_norm_quant, row-sharded at prefill (dpu_reduce path only)."""
        if not (_prefill_glue and d.use_dpu_hidden_norm):
            return hidden_norm_quant(x, consts, qparams, d, fold)
        out = _region_output((1, seq_len, hidden_dim_pw), "int8x64")
        ob = tir.BufferSlice(out)
        _row_sharded(
            lambda lo, hi: hidden_norm_quant(
                slice_n(x, 1, lo, hi),
                consts,
                qparams,
                d,
                fold,
                output=slice_n(ob, 1, lo, hi),
                force_tiled=True,
            )
        )
        return out

    def _heads_dma(x, heads, split):
        """(1, seq, heads*D) -> (1, heads, seq, D) (``split``) or back, at prefill: on each
        core's band, one 2D DMA per head between the two real buffers (slicing a reshape
        VIEW would drop the view's offset)."""
        shape3, shape4 = (1, seq_len, heads * head_dim_pw), (1, heads, seq_len, head_dim_pw)
        out = _region_output(shape4 if split else shape3, x.dtype)
        x3, x4 = (x, out) if split else (out, x)

        def body(lo, hi):
            rows = tir.Ramp(lo, 1, hi - lo)
            for h in range(heads):
                row_view = bs(x3, 0, rows, tir.Ramp(h * head_dim_pw, 1, head_dim_pw))
                head_view = bs(x4, 0, h, rows, tir.Ramp(0, 1, head_dim_pw))
                src, dst = (row_view, head_view) if split else (head_view, row_view)
                axol.atomic.copy(src, output=dst)

        _row_sharded(body)
        return out

    def _to_heads(x3, heads):
        """The (1, heads, seq, D) view of a projection output."""
        if _prefill_glue:
            return _heads_dma(x3, heads, split=True)
        x4 = axol.atomic.reshape(x3, shape=(1, seq_len, heads, head_dim_pw))
        return _swap_seq_heads(x4, (1, heads, seq_len, head_dim_pw))

    def _from_heads(x4, heads):
        """(1, heads, seq, D) -> (1, seq, heads*D) for the o projection."""
        if _prefill_glue:
            return _heads_dma(x4, heads, split=False)
        x = _swap_seq_heads(x4, (1, seq_len, heads, head_dim_pw))
        return axol.atomic.reshape(x, shape=(1, seq_len, heads * head_dim_pw))

    def _quant_heads_sharded(x, qparams, heads):
        """quantize_activation over (1, heads, seq, D), row-sharded: per core and head, one
        contiguous (band, D) slab in and out."""
        out = _region_output((1, heads, seq_len, head_dim_pw), "int8x64")
        qp1 = axol.atomic.reshape(qparams, shape=(2 * head_dim_pw,))
        x_buf = caller_buffer(x)
        d_all = tir.Ramp(0, 1, head_dim_pw)

        def body(lo, hi):
            rows = tir.Ramp(lo, 1, hi - lo)
            for h in range(heads):
                quantize(bs(x_buf, 0, h, rows, d_all), qp1, out=bs(out, 0, h, rows, d_all))

        _row_sharded(body)
        return out

    def _norm_rope_quant(x, consts, weight_buffer, quant_params, trig, heads):
        """RMSNorm -> RoPE -> quantize for one head path: one fused dpu_reduce datapath at
        decode, where every extra op costs a barrier-delimited segment; three ops at prefill,
        whose trig would overflow L1 in the fused form."""
        if seq_len == 1:
            return rms_norm_dpu_reduce(
                x,
                consts,
                normalized_shape=(head_dim,),
                weight_buffer=weight_buffer,
                rope_trig=trig,
                quant_params=quant_params,
                invsqrt_method="additive",
                output=BufferInfo(dtype="int8x64"),
            )
        if _prefill_glue:
            # Per band the trig slice is small, so the fused datapath fits L1 here too.
            out = _region_output((1, heads, seq_len, head_dim_pw), "int8x64")
            ob = tir.BufferSlice(out)

            def body(lo, hi):
                rms_norm_dpu_reduce(
                    slice_n(x, 2, lo, hi),
                    consts,
                    normalized_shape=(head_dim,),
                    weight_buffer=weight_buffer,
                    rope_trig=slice_n(trig, 1, lo, hi),
                    quant_params=quant_params,
                    invsqrt_method="additive",
                    output=slice_n(ob, 2, lo, hi),
                    force_tiled=True,
                )

            _row_sharded(body)
            return out
        normed = rms_norm_dpu_reduce(
            x,
            consts,
            normalized_shape=(head_dim,),
            weight_buffer=weight_buffer,
            invsqrt_method="additive",
        )
        return quantize_activation(
            axol.core.rope(normed, trig), quant_params, (heads, seq_len), head_dim_pw
        )

    def _cache_update(k_update, v_update, k_cache, v_cache, seq_pos):
        """The KV-cache write; with row-sharded glue it runs in a core-0 region of its own,
        between barriers, so the allocator sees the region-written updates consumed."""
        if not _prefill_glue:
            return axol.core.update_kv_cache(k_update, v_update, k_cache, v_cache, seq_pos)
        for core in _core_ids:
            with A.AICore(core=core):
                A.barrier(*_core_ids)
                if core == _core_ids[0]:
                    axol.core.update_kv_cache(
                        k_update, v_update, k_cache, v_cache, seq_pos, core=core
                    )
                A.barrier(*_core_ids)
        return k_cache, v_cache

    def _swap_seq_heads(x, shape):
        """(1, S, H, D) <-> (1, H, S, D), as a free view whenever S is 1.

        Swapping a length-1 axis leaves every element at the same flat offset, so at decode
        the reshape moves no data, where the transpose would still cost its own
        barrier-delimited segment. Prefill keeps the real transpose.
        """
        if seq_len == 1:
            return axol.atomic.reshape(x, shape=shape)
        return axol.core.transpose(x, dims=(0, 2, 1, 3))

    _zp_consts: list = []

    def _sdpa_zp_consts():
        """(wq_zp, wq_qmin, wq_qmax, q_zp, v_zp): read-only, so one set per kernel serves
        every sub-layer."""
        if not _zp_consts:
            _zp_consts.extend(
                A.alloc_constant(np.full((1, 1, 1, 1, PWORD), v), "float32x64", scope=A.DDR)
                for v in (0, -128, 127, 0, 0)
            )
        return _zp_consts

    def _layer_fused(q5, k5, v5, residual_in, seq_pos, rope_trig, sdpa_consts, p, next_layer):
        """`_layer` from the projections on as the fused decode schedule: the attention
        glue, the KV write, SDPA, o_proj, gate_up and down_proj as one region per core
        (ops/attn_block.py), then the layer's residual add per core on its down band and,
        with ``next_layer``, that layer's input norm and qkv band in the same region, its
        qkv tile 0 streamed across the layer boundary."""
        # Read by the norm2 prologue and the add3 band inside the fused region: pinned.
        o6 = _region_output((1, seq_len, hidden_dim_pw), "float32x64")
        q_heads = _to_heads(q5, num_q_heads)
        k_heads = _to_heads(k5, num_kv_heads)
        v_heads = _to_heads(v5, num_kv_heads)

        def _scratch(shape, fill):
            """A pinned int8 DDR scratch buffer the cores fill bands of; its pin write copies
            one PWORD from a constant of its own (one DMA, no datapath fill)."""
            marker = A.alloc_constant(
                np.full((*shape, PWORD), fill, dtype=np.int8), "int8x64", scope=A.DDR
            )
            return _region_output(shape, "int8x64", marker_src=marker)

        # SDPA's output, and the roped/quantized q with the k/v updates packed into ONE
        # scratch (plane 0 = q, 1 = k, 2 = v; k/v use the first num_kv_heads rows). In DDR
        # so SDPA's query staging and the cache-write intrinsic read DDR, not a core's L1.
        attn_out = _scratch((1, num_q_heads, seq_len, head_dim_pw), 1)
        kvq = _scratch((3, num_q_heads, seq_len, head_dim_pw), 2)

        def prologue(core):
            """The attention residual add and post-attn norm+quant, per core in its region."""
            res = A.alloc_buffer((1, seq_len, hidden_dim_pw), "float32x64", scope=A.L1)
            axol.atomic.add(residual_in, o6, output=res)
            return _norm_quant_l1(res, p.rms2_consts, p.gate_proj_qparams, POST_ATTN_NORM_FOLD)

        gu_q = _shape_only((1, seq_len, hidden_dim_pw), "int8x64")
        mlp_split = _region_output(
            *gate_up_output_info(
                gu_q, p.gate_up_proj_w, p.mlp_dq_params, p.silu_lut, p.gate_up_proj_w1
            )
        )
        mlp_ops = MlpOperands(
            matrix_a=gu_q,
            gate_up_w=p.gate_up_proj_w,
            gate_up_dq=p.mlp_dq_params,
            silu_lut=p.silu_lut,
            gate_up_w1=p.gate_up_proj_w1,
            down_w=p.down_proj_w,
            down_dq=p.down_dq_params,
            down_w1=p.down_proj_w1,
            gate_up_output=mlp_split,
            output=BufferInfo(dtype="float32x64"),
            a_prologue=prologue,
        )
        # The layer tail: this layer's residual add per core on its down band, then the next
        # layer's input norm and qkv band, its tile 0 streamed on channel 1 across the
        # down->add3 barrier. layer_out is region-written and region-read: pinned.
        layer_out = _region_output((1, seq_len, hidden_dim_pw), "float32x64")
        next_plan = None
        if next_layer is not None:
            n = SimpleNamespace(**next_layer)
            next_plan = FusedQkvPlan(
                _shape_only((1, seq_len, hidden_dim_pw), "int8x64"),
                outputs=(BufferInfo(dtype="float16x64"),) * 3,
                **_qkv_args(n),
            )
        # The tile-0 slot each core's tail launch streamed into, for its body.
        tail_slots: dict = {}

        def tail_launch(core, part_idx):
            """After the down band, before its closing barrier: the next qkv tile 0."""
            if next_plan is None:
                return
            _, lo, hi = next_plan.partition[part_idx]
            slot = next_plan.alloc_tile0_slot(core, lo, hi)
            if slot is not None:
                next_plan.launch_tile0(core, lo, hi, slot)
            tail_slots[core] = slot

        def tail_body(core, part_idx, n_lo, n_hi, d6):
            """After that barrier: the residual add on this core's band, then the next
            layer's input norm and qkv band."""
            out_bs = tir.BufferSlice(layer_out)
            add3(
                slice_n(residual_in, 2, n_lo, n_hi),
                slice_n(o6, 2, n_lo, n_hi),
                slice_n(d6, 2, n_lo, n_hi),
                output=slice_n(out_bs, 2, n_lo, n_hi),
            )
            if next_plan is None:
                return
            A.barrier(*_core_ids)
            _, lo, hi = next_plan.partition[part_idx]
            core_a = _norm_quant_l1(layer_out, n.rms1_consts, n.q_proj_qparams, INPUT_NORM_FOLD)
            slot = tail_slots.get(core)
            next_plan.emit(core, lo, hi, core_a, pre_slot=slot)
            A.barrier(*_core_ids)

        fused_attn_oproj_multicore(
            q_heads,
            k_heads,
            v_heads,
            p.k_out,
            p.v_out,
            seq_pos,
            rope_trig,
            sdpa_consts,
            p.o_qparams,
            kvq,
            attn_out,
            p.o_proj_w,
            p.o_proj_dq,
            o_w1=p.o_proj_w1,
            cores=cores,
            output=o6,
            k_norm_consts=p.k_norm_consts,
            k_norm_w=p.k_norm_w,
            k_cache_qparams=p.k_cache_qparams,
            v_qparams=p.v_qparams,
            q_norm_consts=p.q_norm_consts,
            q_norm_w=p.q_norm_w,
            q_attn_qparams=p.q_attn_qparams,
            head_dim=head_dim,
            scale=1.0 / (head_dim**0.5),
            packed_consts=p.sdpa_packed,
            mlp=mlp_ops,
            tail_launch=tail_launch,
            tail_body=tail_body,
        )
        _release(mlp_split)
        # The add3 band is o6's region read, so its pin needs no release copy.
        _pinned.pop(id(o6))
        _pin_region_outputs()
        if next_plan is None:
            return layer_out
        return (layer_out, *next_plan.outs)

    def _layer(
        hidden_states,
        residual_in,
        seq_pos,
        attn_mask,
        rope_trig,
        *,
        sdpa_packed=None,
        next_layer=None,
        prev_qkv=None,
        **per_layer,
    ):
        """One decoder layer, hidden_states/residual_in -> layer_out; the caches (k_out,
        v_out) are updated in place.

        Fused decode: ``prev_qkv`` is this layer's (q5, k5, v5), which the previous
        sub-layer's region tail computed; ``next_layer`` is the next sub-layer's buffers,
        whose norm+qkv this layer's tail computes, returning its (q5, k5, v5) after
        layer_out."""
        wq_zp, wq_qmin, wq_qmax, q_zp, v_zp = _sdpa_zp_consts()
        p = SimpleNamespace(sdpa_packed=sdpa_packed, **per_layer)
        # The SDPA op's constant operands, in its order.
        sdpa_consts = (
            p.sdpa_scores_dq_scale,
            p.sdpa_exp_lut,
            p.sdpa_inv_lut,
            p.sdpa_weights_q_scale,
            wq_zp,
            wq_qmin,
            wq_qmax,
            p.sdpa_output_dq_scale,
            q_zp,
            v_zp,
            attn_mask,
        )

        if prev_qkv is not None:
            q5, k5, v5 = prev_qkv
        elif _fused_decode:
            # The input norm repeated per core inside the projection's region.
            q5, k5, v5 = _project_qkv(
                _shape_only((1, seq_len, hidden_dim_pw), "int8x64"),
                p,
                prologue=lambda core: _norm_quant_l1(
                    hidden_states, p.rms1_consts, p.q_proj_qparams, INPUT_NORM_FOLD
                ),
            )
        else:
            qkv_q = _hidden_norm_quant(
                hidden_states, p.rms1_consts, p.q_proj_qparams, INPUT_NORM_FOLD
            )
            q5, k5, v5 = _project_qkv(qkv_q, p)
            _release(qkv_q)

        if _fused_decode:
            return _layer_fused(
                q5, k5, v5, residual_in, seq_pos, rope_trig, sdpa_consts, p, next_layer
            )

        q_heads = _to_heads(q5, num_q_heads)
        k_heads = _to_heads(k5, num_kv_heads)
        # Both head chains are authored AFTER all three projections: k_heads still live
        # across the V projection overflows L1 (4.01 MiB against 4 MiB) at chunk 512.
        v_heads = _to_heads(v5, num_kv_heads)
        _release(q5, k5, v5)

        k_update = _norm_rope_quant(
            k_heads, p.k_norm_consts, p.k_norm_w, p.k_cache_qparams, rope_trig, num_kv_heads
        )
        v_update = (
            _quant_heads_sharded(v_heads, p.v_qparams, num_kv_heads)
            if _prefill_glue
            else quantize_activation(v_heads, p.v_qparams, (num_kv_heads, seq_len), head_dim_pw)
        )
        q_attn = _norm_rope_quant(
            q_heads, p.q_norm_consts, p.q_norm_w, p.q_attn_qparams, rope_trig, num_q_heads
        )
        _release(q_heads, k_heads, v_heads)

        k_out, v_out = _cache_update(k_update, v_update, p.k_out, p.v_out, seq_pos)
        _release(k_update, v_update)

        attn_out = axol.core.scaled_dot_product_attention(
            q_attn,
            k_out,
            v_out,
            *sdpa_consts,
            enable_gqa=True,
            use_newton_raphson=False,
            scale=1.0 / (head_dim**0.5),
            neg_exp_method=ChainedLUTMethod.ADDITIVE,
            inv_method=ChainedLUTMethod.ADDITIVE,
            # The o_proj-input quantize fused into SDPA's P@V write: int8 out, int8 transpose.
            attention_output_quantizer=p.o_qparams,
            **_cores_kw,
        )
        _release(q_attn)

        attn_flat = _from_heads(attn_out, num_q_heads)
        _release(attn_out)
        o6 = quantized_matmul_activation_multicore(
            attn_flat,
            p.o_proj_w,
            p.o_proj_dq,
            cores=_proj_cores,
            matrix_b1=p.o_proj_w1,
            pipeline_weights=_pipeline_weights,
            output=_sharded_proj_output(),
        )
        _release(attn_flat)

        residual = _final_residual(residual_in, o6)
        _release(o6)
        gu_q = _hidden_norm_quant(
            residual, p.rms2_consts, p.gate_proj_qparams, POST_ATTN_NORM_FOLD
        )
        mlp_split = fused_gate_up_swiglu(
            gu_q,
            p.gate_up_proj_w,
            p.mlp_dq_params,
            matrix_b1=p.gate_up_proj_w1,
            luts=p.silu_lut,
            pipeline_weights=_pipeline_weights,
            **_cores_kw,
        )
        _release(gu_q)
        d6 = quantized_matmul_activation_multicore(
            mlp_split,
            p.down_proj_w,
            p.down_dq_params,
            cores=cores,
            matrix_b1=p.down_proj_w1,
            matrix_a_is_split=True,
            pipeline_weights=_pipeline_weights,
            output=_sharded_proj_output(),
        )
        layer_out = _final_residual(residual, d6)
        _release(residual, d6)
        _pin_region_outputs()
        return layer_out

    # The band SDPA reads its eleven scalar constants and LUT rows from ONE packed buffer
    # while its softmax program has a DPU register to spare for it; a longer context stages
    # the separate constants, so the packed buffer is declared only where it is read.
    _sdpa_packed_rows = (
        packed_rows(SDPA_LUT_ROWS, SDPA_LUT_ROWS)
        if _fused_decode and context_len // PWORD <= PACKED_MAX_BLOCKS
        else None
    )
    n_layers = (
        1
        if phase != "decode"
        else layers_per_kernel
        or (default_layers_per_kernel(config.num_hidden_layers) if _fused_decode else 1)
    )
    return grouped_module(
        n_layers,
        _layer,
        shared_params(d),
        layer_params(d, sdpa_packed_rows=_sdpa_packed_rows),
        {"A": A, "_ddr1": _ddr1},
        chain=_fused_decode,
    )


# RMSNorm weights using rms_norm_dpu_reduce (D-major packing).
_DPU_QK_NORM_WEIGHTS = ("q_norm_w", "k_norm_w")

# Weights whose K is split over the two DDR regions: "<name>" holds the lower K
# range in DDR, "<name>1" the upper in DDR1.
_DDR1_SHARDED_WEIGHTS = ("gate_up_proj_w", "down_proj_w", "qkv_proj_w", "o_proj_w")


def qwen3_layer_buffers(bundle, spec, layer):
    """Pack one Qwen3 layer's constants (QK-norm weights D-major for dpu_reduce; interleaved
    per-channel quantize params for the fused ops, with the hidden RMSNorm weights folded
    into the two scales that follow them)."""
    from ...layout import layer_linears, layer_sdpa, layer_subset, pack
    from .naive_impl import _NORM_SITES

    # Hide the dpu_reduce norm names from the optimized packer (its broadcast_to_pword
    # would produce D*PWORD elems and fail to pack into the (D_blocks,) spec).
    base_spec = {
        k: v
        for k, v in ddr1_unsharded_spec(spec, _DDR1_SHARDED_WEIGHTS).items()
        if k not in _DPU_QK_NORM_WEIGHTS and k != "sdpa_packed"
    }
    buffers = optimized_decode_layer_buffers(bundle, base_spec, layer)

    norm = layer_subset(bundle.norm, layer)
    for name in _DPU_QK_NORM_WEIGHTS:
        if name in spec:
            buffers[name] = pack(norm[_NORM_SITES[name]], spec[name])

    _add_qkv_weight(buffers, spec, bundle, layer)
    hidden_dim = int(bundle.config["hidden_size"])
    head_dim = int(bundle.host_scalars["head_dim"])
    add_norm_consts(
        buffers,
        spec,
        bundle,
        layer,
        {
            "rms1_consts": ("input_layernorm.inv_sqrt_range", hidden_dim),
            "rms2_consts": ("post_attention_layernorm.inv_sqrt_range", hidden_dim),
            "q_norm_consts": ("self_attn.q_norm.inv_sqrt_range", head_dim),
            "k_norm_consts": ("self_attn.k_norm.inv_sqrt_range", head_dim),
        },
    )

    # Interleave per-channel scale/zp into the [scale, zp, ...] qparams buffers. The
    # matmul-input quants carry the SmoothQuant zero-point; the SDPA quants are zero-pinned.
    lin = layer_linears(bundle.linear, layer)
    sdpa = layer_sdpa(bundle.sdpa, layer)
    qparam_sites = {
        "q_proj_qparams": (lin["q_proj"]["qs"], lin["q_proj"]["qz"]),
        "gate_proj_qparams": (lin["gate_proj"]["qs"], lin["gate_proj"]["qz"]),
        "o_qparams": (lin["o_proj"]["qs"], lin["o_proj"]["qz"]),
        "q_attn_qparams": (sdpa["qs_q"], None),
        "k_cache_qparams": (sdpa["qs_k"], None),
        "v_qparams": (sdpa["qs_v"], None),
    }
    folded = {f.qparams: f.site for f in HIDDEN_NORM_FOLDS}
    for name, (qs, qz) in qparam_sites.items():
        if name in spec:
            site = folded.get(name)
            buffers[name] = interleave_qparams(
                qs, qz, spec[name], norm_weight=norm[site] if site else None
            )
    check_norm_folds(buffers)
    if "sdpa_packed" in spec:
        buffers["sdpa_packed"] = _pack_sdpa_consts(buffers, spec["sdpa_packed"], head_dim)
    ddr1_split_shards(buffers, spec, _DDR1_SHARDED_WEIGHTS)
    return buffers


def _pack_sdpa_consts(buffers, spec_entry, head_dim: int):
    """The band SDPA's packed constant buffer, from the same per-layer scalars and LUT rows the
    separate constant buffers carry (so both paths compute on identical values)."""
    import numpy as np
    import torch

    from ...layout import pack
    from ...ops.sdpa_band import pack_sdpa_consts

    def _np(name):
        return buffers[name].detach().cpu().float().numpy()

    rows = pack_sdpa_consts(
        _np("sdpa_scores_dq_scale"),
        _np("sdpa_weights_q_scale"),
        0.0,
        _np("sdpa_output_dq_scale"),
        1.0 / (head_dim**0.5),
        _np("sdpa_exp_lut"),
        _np("sdpa_inv_lut"),
    )
    if rows.shape[0] != spec_entry[0][0]:
        raise ValueError(f"sdpa_packed rows {rows.shape[0]} != spec {spec_entry[0][0]}")
    return pack(torch.from_numpy(np.ascontiguousarray(rows)), spec_entry)


def _add_qkv_weight(buffers, spec, bundle, layer) -> None:
    """Stack q/k/v into the one N-concatenated weight the fused projection consumes."""
    # Deferred: packing a torch checkpoint onto kernel buffers is BUILD work, and the run
    # path resolves this module by name without torch installed.
    import torch

    from ...layout import layer_linears

    if "qkv_proj_w" not in spec:
        return
    lin = layer_linears(bundle.linear, layer)
    bands = []
    for proj in ("q_proj", "k_proj", "v_proj"):
        w = lin[proj]["w"]  # [out, in] int8 -> NK (out//PWORD, in, PWORD)
        out, inn = w.shape
        bands.append(w.reshape(out // PWORD, PWORD, inn).permute(0, 2, 1).to(torch.int8))
    buffers["qkv_proj_w"] = torch.cat(bands, dim=0).contiguous()


class Qwen3MulticoreImplementation(MulticoreDecoderImplementation):
    """Qwen3 Metis multicore implementation (standard RoPE; per-head q/k RMSNorm)."""

    name = "qwen3-metis-multicore"
    architecture = "qwen3"
    spec = QWEN3_SPEC
    layer_build = staticmethod(make_qwen3_layer_kernel)
    layer_buffers_fn = staticmethod(qwen3_layer_buffers)


register_implementation(Qwen3MulticoreImplementation())

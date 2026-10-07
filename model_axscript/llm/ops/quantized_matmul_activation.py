#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#

# mypy: disable-error-code="attr-defined, index, arg-type, type-var, assignment, call-overload"

"""Fused int8 matmul + per-channel dequantize + (optional) LUT activation.

Registered as llm.quantized_matmul_activation.  Pairs with
llm.fused_gate_up_swiglu: its matrix_a_is_split mode consumes that op's
split-N int8 output directly (no un-split), and the down-projection dequant is
fused into this op.
"""

import itertools

from axelera.compiler.alto.axol import atomic
from axelera.compiler.alto.axol.common import ramp, sliced
from axelera.compiler.alto.axol.core.matmul._common import pipeline_dims, shards
from axelera.compiler.alto.axol.utils.lut_chain import ChainedLUTMethod, chained_lut_dpu
from axelera.compiler.alto.compiler.config import HardwareGeneration, get_alto_config
from axelera.compiler.alto.compiler.device import MEMORY_SIZES, PWORD, BufferScope
from axelera.compiler.alto.compiler.dtypes import (
    Float32x64,
    Int8x64,
    MetisFloats,
    MetisTypes,
    get_dtype_size,
)
from axelera.compiler.alto.compiler.interpreter import emulation as emu
from axelera.compiler.alto.compiler.op_registry import Arg, BufferInfo, ScriptOp, register_op
import axelera.compiler.alto.script as A
import numpy as np
from tvm import ir, tir

from .matmul import (
    PIPELINE_MIN_TILES,
    _imc_max_blocks,
    _nk_schedule,
    _sync_id,
    _tile_at,
    _try_int,
)

# One HP DMA channel sustains only 4 outstanding transactions where a DDR region
# needs 5-6, so the two weight shards take a channel each.
_LOAD_B_CHANNEL = 0
_LOAD_B1_CHANNEL = 1
_LOAD_B1_ID = "qmm_act_load_b1"
_LOAD_B_ID = "qma_load_b"
_LOAD_IMC_ID = "qma_load_imc"


def prefetch_first(generation=None) -> bool:
    """Whether tile i+1's weight DMAs go out BEFORE tile i's IMC program: Europa waits the
    weight DMA in software, so the next tile's DMAs can be launched before the IMC program;
    Omega's hardware token forbids the reorder."""
    return (generation or get_alto_config().generation) == HardwareGeneration.EUROPA


def inflight_part_id(core, part: int) -> str:
    """Sync id of a resident-band part a caller launched ahead of the op (see inflight_parts)."""
    return f"qma_inflight_{core}_{part}"


def xop_ids(core, slot: int = 0) -> tuple[str, str]:
    """Sync ids of the two cross-op tile-0 shard DMAs into a core's slot, in launch order.

    ``slot`` is the launching loop arm of the first shard (each arm pairs with its own wait).
    """
    return _sync_id("xop_load_first", core, slot), _sync_id("xop_load_second", core, 0)


def stream_shard(src, rows, dst, k_off, *, core, channel, id, where=None):
    """DMA ``rows`` of one K shard of a weight into columns [k_off, k_off + K) of the L1
    buffer ``dst``, on ``channel`` under the software-waited sync ``id``."""
    ext = int(src.shape[-1])
    atomic.copy(
        src[(rows, tir.Ramp(0, 1, ext))],
        scope=A.L1,
        output=dst.tile(tir.Ramp(k_off, 1, ext), dim=1),
        core=core,
        channel=channel,
        id=id,
        where=where,
    )


def _b_l2_stage_max_bytes(generation: HardwareGeneration | None = None) -> int:
    """Largest matrix_b worth staging in L2: 28 MiB of Omega's 32 MiB, 112 MiB of Europa's 128.

    The spare eighth is headroom for the activation and output buffers sharing L2. A weight
    over budget (e.g. Llama-3.1-8B down_proj ~56 MiB) streams from DDR instead.
    """
    l2_bytes = MEMORY_SIZES[generation or get_alto_config().generation][BufferScope.L2]
    return l2_bytes * 7 // 8


class QuantizedMatMulActivation(ScriptOp):
    """Fused int8 matmul + per-channel dequantize + (optional) LUT activation.

    Computes activation((matmul(A, B) - dz) * ds) end-to-end inside a single
    Datapath, so the post-matmul chain stays in DPU registers.

    Layout: matrix_b is always in NK layout (b_transposed=True semantics from
    core.matmul).  matrix_b.shape[-2] is N_BLOCKS; matrix_b.shape[-1] is
    K_BLOCKS * PWORD.

    Dequant params layout: dequant_params carries the per-output-channel dequant
    zero (dz) and scale (ds) interleaved on the channel axis::

        dequant_params[..., 0]   = dz[0]
        dequant_params[..., 1]   = ds[0]
        dequant_params[..., 2]   = dz[1]
        dequant_params[..., 3]   = ds[1]
        ...

    dequant_params.shape[-1] == 2 * N_BLOCKS.  The DPU streams them via IFD1
    and pops one (dz, ds) pair per output column, so only the current pair plus
    LUT-chain transients live in DPU registers.

    Activation is optional:
    - luts=None: no activation, output is the dequantized matmul.
    - luts provided: chained LUT approximation (same as core.piecewise_linear_approximation).

    Schedule: reuses _nk_schedule from core.matmul, including the pow-of-2
    n_per_tile constraint.  Only m_outer compute is supported; n_outer is not yet
    implemented.
    """

    matrix_a = Arg(
        Int8x64[tir.Buffer, "..."],
        description="Quantized A operand, layout (*batch, M, K_BLOCKS), int8x64.",
    )
    matrix_b = Arg(
        Int8x64[tir.Buffer, "..."],
        description=(
            "Quantized B operand in NK layout: (*batch, N_BLOCKS, K_BLOCKS*PWORD), "
            "int8x64.  Same layout as core.matmul with b_transposed=True."
        ),
    )
    dequant_params = Arg(
        MetisFloats[tir.Buffer, "..."],
        description=(
            "Interleaved per-channel dequantize params, last dim is "
            "2 * N_BLOCKS carrying [dz[0], ds[0], dz[1], ds[1], ...].  "
            "Leading dims must be broadcast-compatible with the matmul output "
            "(typically (1, ..., 1) for per-tensor or (1, 1, 2*N_BLOCKS) for "
            "per-output-channel, since dz/ds don't depend on M)."
        ),
    )

    # Keyword args
    matrix_b1 = Arg(
        Int8x64[tir.Buffer, "..."] | None,
        default=None,
        kw=True,
        description=(
            "Optional second shard of matrix_b holding the upper K range, same "
            "layout and N extent, the two K extents summing to matrix_a's K. Both "
            "shards stream concurrently on separate DMA channels, so placing them "
            "in different DDR regions (A.DDR and A.DDR1 on Europa) doubles the "
            "weight bandwidth. Requires the pipelined N loop and no L2 staging."
        ),
    )
    luts = Arg(
        Float32x64[tir.Buffer, "n_luts"] | None,
        default=None,
        kw=True,
        description=(
            "Optional packed LUT parameters (n_luts x PWORD float32) for the "
            "fused activation; same format as core.piecewise_linear_approximation. "
            "None skips the activation step."
        ),
    )
    method = Arg(
        ChainedLUTMethod,
        default=ChainedLUTMethod.ADDITIVE,
        kw=True,
        description="Chaining strategy for multi-LUT activations (ADDITIVE or COMPOSITION).",
    )
    output = Arg(
        MetisTypes[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description=(
            "Output buffer.  None auto-infers shape (*batch, M, N_BLOCKS) and "
            "dtype float16x64 in matrix_a's scope."
        ),
    )
    nk_schedule = Arg(
        str | None,
        default=None,
        kw=True,
        description=(
            "Override the NK schedule choice (m_outer or n_outer); see "
            "core.matmul.  Only m_outer compute is implemented; n_outer raises "
            "NotImplementedError."
        ),
    )
    matrix_a_is_split = Arg(
        bool,
        default=False,
        kw=True,
        description=(
            "When True, matrix_a has an extra leading axis that splits "
            "its K_BLOCKS dimension into uniform tiles: shape "
            "(*batch, num_k_splits, M, K_BLOCKS // num_k_splits).  "
            "This matches the split-N layout emitted by "
            "core.matmul_split_n and core.fused_gate_up_swiglu, so "
            "each per-K-split DMA into L1 is a single contiguous read "
            "(no per-row stride through the wider K axis).  Otherwise "
            "matrix_a is read in canonical (*batch, M, K_BLOCKS) layout."
        ),
    )
    core = Arg(
        int | None,
        default=None,
        kw=True,
        description="Optional AIPU core to run on.",
    )
    generation = Arg(
        HardwareGeneration | None,
        default=None,
        kw=True,
        description="Optional hardware generation.",
    )
    stage_weight_l2 = Arg(
        bool | None,
        default=None,
        kw=True,
        description=(
            "Force the matrix_b -> L2 staging decision. None = auto (stage iff this op's weight "
            "fits the L2 budget). The multicore N-shard sets this from the FULL weight size, since "
            "each core stages its band into the SHARED L2 -- a per-band size can pass the gate "
            "while the aggregate overflows (e.g. Llama-3.1-8B down_proj: 4 x 14 MiB = 56 MiB)."
        ),
    )
    pipeline_weights = Arg(
        bool,
        default=False,
        kw=True,
        description=(
            "Software-pipeline the N loop so tile i+1's weight DMA runs under tile i's IMC load "
            "and datapath. Without it the two are serialized: the compiled stream makes every "
            "weight DMA after the first consume the previous tile's m_odr token, because matrix_b "
            "is staged into one L1 band that the IMC load is still reading. Pipelining stages it "
            "two tiles deep instead, so the write-after-read is gone. Needs a tiled N "
            "(num_n_tiles > 1); ignored otherwise."
        ),
    )
    prefetched = Arg(
        Int8x64[tir.Buffer, "..."] | None,
        default=None,
        kw=True,
        description=(
            "Caller-owned L1 slot (n_per_tile, K) holding tile 0's K shards, launched with ids "
            "xop_ids(core) and not waited: the op waits them and uses it as pipeline slot 0. "
            "Pipelined uniform N tiles only."
        ),
    )
    prefetched_both = Arg(
        bool,
        default=False,
        kw=True,
        description=(
            "With ``prefetched``: both K shards are in flight (launched back to back on one "
            "channel), so the op waits both in launch order."
        ),
    )
    output_splits = Arg(
        tuple | None,
        default=None,
        kw=True,
        description=(
            "Band-local block-column ranges written to separate buffers, flat (lo0, hi0, lo1, "
            "hi1): the first to ``output``, the second to ``output2``, for a band straddling two "
            "projections of a concatenated weight."
        ),
    )
    output2 = Arg(
        MetisTypes[tir.Buffer, "..."] | None,
        default=None,
        kw=True,
        description="With output_splits: the second range's caller-owned output slice (the "
        "first range goes to ``output``).",
    )
    inflight_parts = Arg(
        tuple | None,
        default=None,
        kw=True,
        description=(
            "With ``resident``: parts (2*tile + shard) the caller launched (ids "
            "inflight_part_id(core, part)) and not waited; the op waits them before their tile "
            "and streams only the rest. Launch order per channel is tile order (attn_block's "
            "channel discipline)."
        ),
    )
    resident = Arg(
        Int8x64[tir.Buffer, "..."] | None,
        default=None,
        kw=True,
        description=(
            "Caller-owned L1 band (n_blocks, K) holding this op's whole weight: each tile's IMC "
            "program reads its rows and the op streams only the parts not in ``resident_parts``. "
            "Pipelined uniform N tiles only."
        ),
    )
    resident_parts = Arg(
        tuple | None,
        default=None,
        kw=True,
        description=(
            "With ``resident``: the parts (2*tile + shard; shard 0 = matrix_b, 1 = matrix_b1) "
            "already landed. The rest are launched up front in tile order over both channels "
            "and waited per tile. None = all resident."
        ),
    )
    fuse_output_quant = Arg(
        bool,
        default=False,
        kw=True,
        description=(
            "Fuse the DOWNSTREAM int8 activation-quantize into this op's output write, so the "
            "next matmul consumes int8 straight from L1 -- no separate llm.quantize pass and no "
            "materialized fp16 intermediate. When True: (1) output is int8x64; (2) dequant_params "
            "carries FOUR values per output block instead of two, block-interleaved as "
            "[dz, ds, q_scale, q_zp, ...] (last dim 4*N_BLOCKS), so the single IFD1 stream feeds "
            "both the dequant and the output-quant with one regular contiguous walk (no second "
            "iterator on the port). The DPU applies out = round_clip_int8((activation) * q_scale "
            "+ q_zp) after the dequant (and optional LUT) -- identical math to llm.quantize."
        ),
    )

    # ------------------------------------------------------------------ #
    # Layout helpers (NK only)                                            #
    # ------------------------------------------------------------------ #

    def _param_width(self) -> int:
        """Values per output block in dequant_params: 2 = [dz, ds]; 4 = [dz, ds, q_scale, q_zp]
        when fuse_output_quant folds the downstream quantize into this op's output."""
        return 4 if self.fuse_output_quant else 2

    def _b_n_blocks(self) -> tir.PrimExpr:
        return self.matrix_b.shape[-2]

    def _b_k(self) -> tir.PrimExpr:
        """Full K, summing both shards when matrix_b1 carries the upper range."""
        if self.matrix_b1 is None:
            return self.matrix_b.shape[-1]
        return self.matrix_b.shape[-1] + self.matrix_b1.shape[-1]

    def _a_batch_shape(self) -> tuple:
        """matrix_a's batch dims (all leading axes before M and K_BLOCKS),
        excluding any split axis introduced by matrix_a_is_split."""
        if self.matrix_a_is_split:
            return tuple(self.matrix_a.shape[:-3])
        return tuple(self.matrix_a.shape[:-2])

    def _a_m(self):
        """matrix_a's M dim."""
        return self.matrix_a.shape[-2]

    def _a_k_blocks_total(self):
        """Total K_BLOCKS along matrix_a's reduction axis.  In split layout
        this is num_k_splits * K_per_split; otherwise matrix_a.shape[-1]."""
        if self.matrix_a_is_split:
            return self.matrix_a.shape[-3] * self.matrix_a.shape[-1]
        return self.matrix_a.shape[-1]

    def _resolve_batch_effective(self):
        """Like _resolve_batch but uses matrix_a's batch axes with the
        split axis folded out."""
        a_batch = self._a_batch_shape()
        b_batch = tuple(self.matrix_b.shape[:-2])
        broadcast_a = None
        broadcast_b = None
        if a_batch != b_batch:
            from functools import reduce
            from operator import mul

            if reduce(mul, a_batch, 1) == 1:
                broadcast_a = (0,) * len(a_batch)
            if reduce(mul, b_batch, 1) == 1:
                broadcast_b = (0,) * len(b_batch)
            if broadcast_a is None and broadcast_b is None:
                raise ValueError(
                    f"Batch dimensions of matrix A {a_batch} are not "
                    f"broadcastable with matrix B {b_batch}."
                )
        batch_dims = max(a_batch, b_batch)
        return batch_dims, broadcast_a, broadcast_b

    # ------------------------------------------------------------------ #

    def _validate_matrix_b1(self) -> None:
        """Check the upper-K shard against matrix_b; K extents are checked by validate."""
        if self.matrix_b1.dtype != "int8x64":
            raise ValueError(f"matrix_b1 dtype must be int8x64, got {self.matrix_b1.dtype}.")
        if tuple(self.matrix_b1.shape[:-2]) != tuple(self.matrix_b.shape[:-2]):
            raise ValueError(
                f"Batch dimensions of matrix_b1 {self.matrix_b1.shape[:-2]} do not match "
                f"matrix_b {self.matrix_b.shape[:-2]}."
            )
        if not ir.structural_equal(self.matrix_b1.shape[-2], self.matrix_b.shape[-2]):
            raise ValueError(
                f"N extent of matrix_b1 ({self.matrix_b1.shape[-2]}) does not match "
                f"matrix_b ({self.matrix_b.shape[-2]})."
            )

    def validate(self) -> None:
        min_a_dims = 3 if self.matrix_a_is_split else 2
        if len(self.matrix_a.shape) < min_a_dims:
            raise ValueError(
                f"matrix_a must have at least {min_a_dims} dimensions "
                f"(matrix_a_is_split={self.matrix_a_is_split}), got "
                f"{len(self.matrix_a.shape)}."
            )
        if len(self.matrix_b.shape) < 2:
            raise ValueError(
                f"matrix_b must have at least 2 dimensions, got {len(self.matrix_b.shape)}."
            )
        # Validates broadcast compatibility (raises if invalid).
        self._resolve_batch_effective()

        k_a = self._a_k_blocks_total() * PWORD
        if not ir.structural_equal(k_a, self._b_k()):
            raise ValueError(
                f"Inner dimensions of matrix A ({k_a}) and matrix B "
                f"({self._b_k()}) do not match."
            )
        if self.matrix_a.dtype != "int8x64":
            raise ValueError(f"matrix_a dtype must be int8x64, got {self.matrix_a.dtype}.")
        if self.matrix_b.dtype != "int8x64":
            raise ValueError(f"matrix_b dtype must be int8x64, got {self.matrix_b.dtype}.")
        if self.matrix_b1 is not None:
            self._validate_matrix_b1()

        # dequant_params: last dim must equal WIDTH * N_BLOCKS (WIDTH=2 for [dz,ds],
        # 4 for [dz,ds,q_scale,q_zp] when fusing the output quant); all leading dims
        # must be 1 (we currently flat-view this buffer for IFD1 staging).
        n_blocks = self._b_n_blocks()
        width = self._param_width()
        if not ir.structural_equal(self.dequant_params.shape[-1], n_blocks * width):
            raise ValueError(
                f"dequant_params last dim must be {width}*N_BLOCKS = {n_blocks * width} "
                f"(fuse_output_quant={self.fuse_output_quant}), got "
                f"{self.dequant_params.shape[-1]}."
            )
        for i, dim in enumerate(self.dequant_params.shape[:-1]):
            if not ir.structural_equal(dim, 1):
                raise ValueError(
                    f"dequant_params leading dims must all be 1 (per-channel "
                    f"broadcast over batch and M); got dim {i} = {dim}."
                )

        if isinstance(self.output, tir.Buffer):
            if self.output.dtype not in ("int8x64", "float16x64", "float32x64"):
                raise ValueError(
                    f"output dtype {self.output.dtype} is not supported, "
                    f"must be int8x64, float16x64 or float32x64."
                )
            if self.fuse_output_quant and self.output.dtype != "int8x64":
                raise ValueError(
                    f"fuse_output_quant requires an int8x64 output, got {self.output.dtype}."
                )

        if self.nk_schedule is not None and self.nk_schedule not in ("m_outer", "n_outer"):
            raise ValueError(
                f"nk_schedule must be None, 'm_outer', or 'n_outer'; got {self.nk_schedule!r}"
            )

    def infer_outputs(self) -> dict[str, BufferInfo]:
        batch_dims, _, _ = self._resolve_batch_effective()
        m = self._a_m()
        n_blocks = self._b_n_blocks()
        return {
            "output": BufferInfo(
                shape=tuple(batch_dims) + (m, n_blocks),
                dtype="int8x64" if self.fuse_output_quant else "float16x64",
                scope=self.matrix_a.scope(),
            )
        }

    # ------------------------------------------------------------------ #
    # Implementation                                                       #
    # ------------------------------------------------------------------ #

    def _pipeline_plan(self) -> tuple[int | None, int | None, bool, int]:
        """The N/M tiling implementation() uses: (num_n_tiles, num_m_tiles, pipeline_n, n_per_tile).

        Also what a caller streaming this op's tile 0 ahead of it (``prefetched``) sizes the
        slot by, so both must come from this one place.
        """
        m = self._a_m()
        k_blocks = self._a_k_blocks_total()
        n_blocks = self.matrix_b.shape[-2]

        # Pre-stage the full 2*N_BLOCKS dequant table in L1.  Reserve those
        # bytes against the schedule's L1 budget so tile sizing is accurate.
        n_blocks_static = _try_int(n_blocks)
        width = self._param_width()
        dq_bytes = 0
        if n_blocks_static is not None:
            # get_dtype_size reports the lane-vector size per element
            # (e.g. 256 B for float32x64), so byte count = num_elements * itemsize.
            dq_bytes = width * n_blocks_static * get_dtype_size(self.dequant_params.dtype)
        # Only m_outer is implemented, so the schedule is pinned regardless of
        # the nk_schedule arg.
        _, num_n_tiles, num_m_tiles = _nk_schedule(
            m,
            k_blocks,
            n_blocks,
            self.generation,
            force="m_outer",
            out_dtype=self.output.dtype,
            extra_l1_bytes=dq_bytes,
            b_slots=2 if self.pipeline_weights else 1,
            # The prefetch needs several uniform tiles to double-buffer between; without this
            # the widest fit collapses N to one tile (or a ragged several) and pipeline_n
            # below silently comes out False, discarding the pipelining the caller asked for.
            min_n_tiles=PIPELINE_MIN_TILES if self.pipeline_weights else 1,
        )

        num_n_tiles = num_n_tiles if (num_n_tiles or 0) > 1 else None
        num_m_tiles = num_m_tiles if (num_m_tiles or 0) > 1 else None
        # A ragged last tile miscomputes once the pipeline stages overlap.
        # Uniform tiles only; the same shape split evenly is bit-identical.
        pipeline_n = bool(
            self.pipeline_weights
            and num_n_tiles is not None
            and n_blocks_static is not None
            and n_blocks_static % num_n_tiles == 0
        )
        n_per_tile_max = n_blocks_static // num_n_tiles if pipeline_n else n_blocks
        return num_n_tiles, num_m_tiles, pipeline_n, n_per_tile_max

    def implementation(self) -> None:
        batch_dims, broadcast_a, broadcast_b = self._resolve_batch_effective()

        m = self._a_m()
        k_blocks = self._a_k_blocks_total()
        n_blocks = self.matrix_b.shape[-2]
        k = self._b_k()
        width = self._param_width()
        num_n_tiles, num_m_tiles, pipeline_n, n_per_tile_max = self._pipeline_plan()

        # Size the M-dimension L1 buffers to one tile's worth instead of the
        # full M.  The allocator does not fold M-tiled buffers down per
        # iteration, so full-M allocations would exceed the L1 budget even
        # when the per-tile live region fits.  m_tile.base is only used for
        # DDR-side indexing; in L1 we always address from base 0 via a
        # local_m_ramp of width m_tile.lanes.
        m_static = _try_int(m)
        if m_static is None:
            raise ValueError("quantized_matmul_activation requires a statically-known M.")
        if num_m_tiles is None:
            m_tile_max = m_static
        else:
            m_tile_max = (m_static + num_m_tiles - 1) // num_m_tiles

        # Two views of the same L1 staging buffer:
        #   a_l1_storage: natural shape — 2D (m_tile_max, k_blocks) in
        #     canonical mode, 3D (num_k_splits, m_tile_max, k_per_split)
        #     in split mode — so the atomic.copy ndim check matches.
        #   a_l1: 3D alias of the same memory for the IFD0 walk, uniform
        #     across both modes (num_k_splits=1 collapses in canonical mode).
        if self.matrix_a_is_split:
            num_k_splits = _try_int(self.matrix_a.shape[-3])
            k_per_split_int = _try_int(self.matrix_a.shape[-1])
            a_l1_storage = A.alloc_buffer(
                (num_k_splits, m_tile_max, k_per_split_int),
                dtype=self.matrix_a.dtype,
                scope=A.L1,
            )
            a_l1 = a_l1_storage
        else:
            num_k_splits = 1
            k_per_split_int = _try_int(k_blocks)
            a_l1_storage = A.alloc_buffer(
                (m_tile_max, k_blocks),
                dtype=self.matrix_a.dtype,
                scope=A.L1,
            )
            a_l1 = A.alloc_buffer(
                (1, m_tile_max, k_per_split_int),
                dtype=self.matrix_a.dtype,
                scope=A.L1,
                data=a_l1_storage.data,
            )
        # Pipelining stages the weights two tiles deep instead of banding the whole N: tile i+1's
        # DMA targets the slot tile i is not reading, so it can run under tile i's IMC load.
        if self.pipeline_weights and num_m_tiles is not None:
            # A tiled M re-reads A and flushes its tile inside the compute the prefetch
            # runs under, and those DMAs contend for the token line it holds open.
            raise ValueError(
                "quantized_matmul_activation cannot pipeline weights with a tiled M "
                f"(num_m_tiles={num_m_tiles}); M must fit one L1 tile."
            )
        # One SEPARATE allocation per pipeline slot: a command's hazard region is the
        # whole backing buffer, so two slots in one make the prefetch collide.
        resident = caller_buffer(self.resident)
        if resident is not None:
            if not pipeline_n:
                raise ValueError("resident needs the pipelined N loop (uniform N tiles).")
            want = (int(n_blocks), int(k))
            if tuple(int(x) for x in resident.shape) != want:
                raise ValueError(f"resident band shape {tuple(resident.shape)} != {want}")
            if self.prefetched is not None:
                raise ValueError("resident and prefetched are alternatives")
        b_l1 = (
            []
            if resident is not None
            else [
                A.alloc_buffer((n_per_tile_max, k), dtype=self.matrix_b.dtype, scope=A.L1)
                for _ in range(2 if pipeline_n else 1)
            ]
        )
        prefetched = caller_buffer(self.prefetched)
        if prefetched is not None:
            # The op before this one streamed tile 0 into the caller's buffer: it is slot 0.
            if not pipeline_n:
                raise ValueError("prefetched needs the pipelined N loop (uniform N tiles).")
            want = (int(n_per_tile_max), int(k))
            if tuple(int(x) for x in prefetched.shape) != want:
                raise ValueError(f"prefetched slot shape {tuple(prefetched.shape)} != {want}")
            b_l1[0] = prefetched
        out_l1 = A.alloc_buffer(
            (m_tile_max, n_blocks),
            dtype=self.output.dtype,
            scope=A.L1,
        )
        # Stage the full interleaved dequant table in L1 once per matmul (tiny:
        # WIDTH*N_BLOCKS * 64 * dtype_bytes, usually well under 100 KB). WIDTH is 2 for
        # [dz,ds] or 4 for [dz,ds,q_scale,q_zp] when the output quant is fused in.
        dq_l1 = A.alloc_buffer((n_blocks * width,), dtype=self.dequant_params.dtype, scope=A.L1)
        # Flat re-view of dequant_params: leading dims are all 1 (validated),
        # so the data is a contiguous (WIDTH*N_BLOCKS,)-shaped table.
        dq_flat = A.alloc_buffer(
            (n_blocks * width,),
            dtype=self.dequant_params.dtype,
            scope=self.dequant_params.scope(),
            data=self.dequant_params.buffer.data,
        )
        # The resident path is a Python-unrolled tile loop the memory-constraint pass cannot
        # tile, so it gets a one-tile IMC operand reused per tile (the pipelined path's full
        # (K, N) operand is tiled by that pass along its A.tiles loop).
        imc = A.alloc_imc((k_blocks, int(n_per_tile_max) if resident is not None else n_blocks))

        # Stage the interleaved dequant table once (broadcast over batch and M).
        atomic.copy(dq_flat, scope=A.L1, output=dq_l1, generation=self.generation)
        if self.inflight_parts is not None and resident is not None:
            # Parts the caller launched ahead of this op (channel 1, across the barrier): waited
            # here at the core's scope, where the caller's launches can pair (the N loop below
            # sits in the M-tiles scope); after the dq copy, which is channel 0's.
            # One wait per part: several ids on one channel in a single A.waits collapse into a
            # single token consume (the sync's targets are a set of units).
            for p in self.inflight_parts:
                A.waits(inflight_part_id(self.core, int(p)))

        if prefetched is not None:
            # Tile 0's second shard is in flight into slot 0 since the previous op (the first
            # was waited there). Its launch pairs with this wait across the ops because both
            # sit at the core's scope: outside any loop, so before the batch grid and M loop.
            if self.prefetched_both:
                # Both shards are still in flight (one channel, no wait between the launches):
                # wait them in launch order, as the FIFO pairing expects.
                A.waits(xop_ids(self.core, 0)[0])
            A.waits(xop_ids(self.core, 0)[1])

        if batch_dims:
            with A.grid(*batch_dims) as batch_vars_:
                batch_vars = A.to_tuple(batch_vars_)
                ba = tuple(broadcast_a) if broadcast_a is not None else batch_vars
                bb = tuple(broadcast_b) if broadcast_b is not None else batch_vars
                self._compute_m_outer(
                    ba,
                    bb,
                    batch_vars,
                    a_l1,
                    a_l1_storage,
                    b_l1,
                    dq_l1,
                    out_l1,
                    imc,
                    num_n_tiles,
                    num_m_tiles,
                    pipeline_n,
                    resident,
                )
        else:
            self._compute_m_outer(
                (),
                (),
                (),
                a_l1,
                a_l1_storage,
                b_l1,
                dq_l1,
                out_l1,
                imc,
                num_n_tiles,
                num_m_tiles,
                pipeline_n,
                resident,
            )

    def _compute_m_outer(
        self,
        ba,
        bb,
        bo,
        a_l1,
        a_l1_storage,
        b_l1,
        dq_l1,
        out_l1,
        imc,
        num_n_tiles,
        num_m_tiles,
        pipeline_n,
        resident=None,
    ) -> None:
        """M outer, N inner.  For each N tile, fuse matmul + dequant + LUT
        activation into a single Datapath block per M row in the M tile.
        """
        # a_l1 is always 3D (num_k_splits, m, k_per_split); split mode
        # collapses a_l1.shape[0] * a_l1.shape[2] into K_BLOCKS.
        m = self._a_m()
        k_blocks = self._a_k_blocks_total()
        num_k_splits = int(a_l1.shape[0])
        # From matrix_b, not b_l1: with pipelining b_l1 holds one N tile per slot.
        n_blocks = self.matrix_b.shape[-2]
        k = self._b_k()

        # Stage matrix_b in L2 only when M is actually tiled (num_m_tiles > 1): the L2 copy
        # amortizes the per-N-tile weight re-read across M tiles (with a single M tile, e.g. decode
        # M=1, the L2 hop is pure overhead) -- AND only when the full weight fits the L2 budget,
        # else it streams from DDR (the missing byte cap previously caused an OOM).
        if self.stage_weight_l2 is not None:
            b_fits_l2 = self.stage_weight_l2  # forced (e.g. by the multicore N-shard, see below)
        else:
            b_nb, b_k = _try_int(self.matrix_b.shape[-2]), _try_int(k)
            b_bytes = (
                None
                if b_nb is None or b_k is None
                else b_nb * b_k * get_dtype_size(self.matrix_b.dtype)
            )
            b_fits_l2 = b_bytes is None or b_bytes <= _b_l2_stage_max_bytes(self.generation)
        # ``b1_src`` set means the N loop streams the two shards itself, one per region; staging
        # clears it, joining them in L2 once so the inner loop has a single full-K source.
        b1_src = self.matrix_b1
        if num_m_tiles is not None and num_m_tiles > 1 and b_fits_l2:
            if self.matrix_b1 is None:
                b_src = atomic.copy(self.matrix_b, scope=A.L2)
            else:
                b_src = self._stage_shards_l2(n_blocks, k)
                b1_src = None
        else:
            b_src = self.matrix_b

        with A.tiles(m, num_tiles=num_m_tiles) as m_tile:
            # a_l1 / out_l1 are sized to one M-tile; in L1 address from
            # base 0.  m_tile.base only appears on the DDR side.
            local_m_ramp = tir.Ramp(0, 1, m_tile.lanes)

            if self.matrix_a_is_split:
                k_per_split_expr = self.matrix_a.shape[-1]
                if num_m_tiles is None:
                    # A single M tile spans the full M of every split, so the whole
                    # (num_k_splits, M, k_per_split) block is contiguous on both sides.
                    atomic.copy(
                        self.matrix_a[ba + ramp((num_k_splits, m, k_per_split_expr))],
                        scope=A.L1,
                        output=a_l1_storage,
                        generation=self.generation,
                    )
                else:
                    # An M tile leaves gaps between splits, so one contiguous DMA per
                    # split; the 1-lane Ramp keeps the destination's leading singleton dim.
                    for k_split_idx in range(num_k_splits):
                        ks_ramp = tir.Ramp(k_split_idx, 1, 1)
                        atomic.copy(
                            self.matrix_a[ba + (ks_ramp, m_tile, ramp(k_per_split_expr))],
                            scope=A.L1,
                            output=a_l1_storage.tile(ks_ramp, dim=0).tile(local_m_ramp, dim=1),
                            generation=self.generation,
                        )
            else:
                # Canonical layout — single contiguous DMA into 2D storage.
                # The 3D a_l1 view aliases this memory for the IFD0 walk.
                atomic.copy(
                    self.matrix_a[ba + (m_tile, ramp(k_blocks))],
                    scope=A.L1,
                    output=a_l1_storage.tile(local_m_ramp, dim=0),
                    generation=self.generation,
                )
            if resident is not None:
                self._n_resident(resident, a_l1, dq_l1, out_l1, imc, local_m_ramp, num_n_tiles)
            elif pipeline_n:
                self._n_pipeline(
                    b_src, bb, a_l1, b_l1, dq_l1, out_l1, imc, local_m_ramp, k, num_n_tiles, b1_src
                )
            else:
                with A.tiles(n_blocks, num_tiles=num_n_tiles) as n_tile:
                    if b1_src is None:
                        atomic.copy(
                            b_src[bb + (n_tile, ramp(k))],
                            scope=A.L1,
                            output=b_l1[0].tile(n_tile, dim=0),
                            generation=self.generation,
                        )
                    else:
                        # Each shard is densely packed, so both DDR reads stay
                        # contiguous and only the L1 write is strided.
                        k_lo = self.matrix_b.shape[-1]
                        atomic.copy(
                            b_src[bb + (n_tile, ramp(k_lo))],
                            scope=A.L1,
                            output=b_l1[0].tile(n_tile, dim=0).tile(tir.Ramp(0, 1, k_lo), dim=1),
                            core=self.core,
                            channel=_LOAD_B_CHANNEL,
                            generation=self.generation,
                            id=_LOAD_B_ID,
                        )
                        atomic.copy(
                            b1_src[bb + (n_tile, ramp(k - k_lo))],
                            scope=A.L1,
                            output=b_l1[0]
                            .tile(n_tile, dim=0)
                            .tile(tir.Ramp(k_lo, 1, k - k_lo), dim=1),
                            core=self.core,
                            channel=_LOAD_B1_CHANNEL,
                            generation=self.generation,
                            id=_LOAD_B1_ID,
                        )
                        # One sync closing both launches: an unsynced launch gets its own
                        # wait injected after it, which breaks up the two regions' overlap.
                        A.waits(_LOAD_B_ID, _LOAD_B1_ID)
                    atomic.load_imc(
                        buffer=b_l1[0].tile(n_tile, dim=0),
                        imc=imc.tile(n_tile, dim=1),
                        input_is_transposed=True,
                        core=self.core,
                        generation=self.generation,
                    )
                    # a_l1 is always 3D (num_k_splits, m_tile_max, k_per_split).
                    # Slice m on dim=1 with the local ramp (base 0) to give
                    # _fused_inner a uniform (num_k_splits, m_tile_lanes, k_per_split) view.
                    self._fused_inner(
                        a_tile=a_l1.tile(local_m_ramp, dim=1),
                        imc_tile=imc.tile(n_tile, dim=1),
                        dq_tile=dq_l1,  # full dq_l1; index using n_tile.base
                        n_tile=n_tile,
                        out_tile=out_l1.tile(local_m_ramp, dim=0).tile(n_tile, dim=1),
                    )

            # Output index is `bo + (m_tile, N)`, matching an output of rank len(bo)+2. A per-core
            # sliced output (e.g. an M-shard/N-shard slab (1, M, N) from a rank-preserving slice)
            # carries extra leading singleton dims; when matrix_a_is_split makes batch_dims=() (so
            # bo=()) those don't line up and the index under-fills the buffer. Pad with leading 0s to
            # the output's actual rank (no-op when they already match, e.g. fc1's normal-input shard).
            if self.output_splits is not None:
                # One pipeline over a band that spans two projections: each part of out_l1 goes
                # to its own destination (two small writes instead of one).
                ranges = [int(x) for x in self.output_splits]
                for idx, dst in enumerate((self.output, self.output2)[: len(ranges) // 2]):
                    c_lo, c_hi = ranges[2 * idx], ranges[2 * idx + 1]
                    dst_index = bo + (m_tile, tir.Ramp(0, 1, c_hi - c_lo))
                    n_extra = len(dst.shape) - len(dst_index)
                    if n_extra > 0:
                        dst_index = (0,) * n_extra + dst_index
                    atomic.copy(
                        out_l1.tile(local_m_ramp, dim=0).tile(
                            tir.Ramp(c_lo, 1, c_hi - c_lo), dim=1
                        ),
                        scope=A.DDR,
                        output=dst[dst_index],
                        generation=self.generation,
                    )
                return
            out_index = bo + (m_tile, ramp(n_blocks))
            n_extra = len(self.output.shape) - len(out_index)
            if n_extra > 0:
                out_index = (0,) * n_extra + out_index
            atomic.copy(
                out_l1.tile(local_m_ramp, dim=0),
                scope=A.DDR,
                output=self.output[out_index],
                generation=self.generation,
            )

    def _stage_shards_l2(self, n_blocks, k):
        """Join the two DDR shards into one full-K L2 copy, one transfer per region."""
        k_lo = self.matrix_b.shape[-1]
        b_l2 = A.alloc_buffer((n_blocks, k), dtype=self.matrix_b.dtype, scope=A.L2)
        atomic.copy(
            self.matrix_b,
            scope=A.L2,
            output=b_l2.tile(tir.Ramp(0, 1, k_lo), dim=1),
            channel=_LOAD_B_CHANNEL,
            generation=self.generation,
            id=_LOAD_B_ID,
        )
        atomic.copy(
            self.matrix_b1,
            scope=A.L2,
            output=b_l2.tile(tir.Ramp(k_lo, 1, k - k_lo), dim=1),
            channel=_LOAD_B1_CHANNEL,
            generation=self.generation,
            id=_LOAD_B1_ID,
        )
        # One sync closing both launches: an unsynced launch gets its own wait
        # injected after it, which breaks up the two regions' overlap.
        A.waits(_LOAD_B_ID, _LOAD_B1_ID)
        return b_l2

    def _weight_shards(self, b_src, b1_src, k):
        """Per K shard of the weight: (DDR operand, K offset, K extent, DMA channel, id)."""
        if b1_src is None:
            return [(b_src, 0, k, None, _LOAD_B_ID)]
        k_lo = self.matrix_b.shape[-1]
        return [
            (b_src, 0, k_lo, _LOAD_B_CHANNEL, _LOAD_B_ID),
            (b1_src, k_lo, k - k_lo, _LOAD_B1_CHANNEL, _LOAD_B1_ID),
        ]

    def _n_pipeline(
        self, b_src, bb, a_l1, b_l1, dq_l1, out_l1, imc, local_m_ramp, k, num_n_tiles, b1_src=None
    ) -> None:
        """N-tile loop with tile i+1's weight DMAs in flight under tile i's compute."""
        n_blocks = int(self.matrix_b.shape[-2])  # the band, not the (split) output
        n_per_tile = n_blocks // num_n_tiles
        core = self.core
        weight_shards = self._weight_shards(b_src, b1_src, k)

        def _load_b(n_idx, slot, where=None):
            """Stage N tile ``n_idx`` into pipeline slot ``slot`` of b_l1."""
            # Each shard is densely packed, so both DDR reads stay
            # contiguous and only the L1 write is strided.
            for src, k_offset, extent, channel, id_ in weight_shards:
                atomic.copy(
                    src[bb + (n_idx, ramp(extent))],
                    scope=A.L1,
                    output=b_l1[slot].tile(tir.Ramp(k_offset, 1, extent), dim=1),
                    core=core,
                    channel=channel,
                    generation=self.generation,
                    id=_sync_id(id_, core, slot),
                    where=where,
                )

        def _load_imc(n_tile, slot, where=None):
            """Program the IMC columns of ``n_tile`` from pipeline slot ``slot``."""
            atomic.load_imc(
                buffer=b_l1[slot],
                imc=imc.tile(n_tile, dim=1),
                input_is_transposed=True,
                core=core,
                generation=self.generation,
                id=_sync_id(_LOAD_IMC_ID, core, slot),
                where=where,
            )

        def _wait_b(slot):
            """Retire the loads that filled pipeline slot ``slot``."""
            A.waits(*(_sync_id(id_, core, slot) for _, _, _, _, id_ in weight_shards))

        # Tile 0 is peeled so the datapath is never predicated: a predicated datapath
        # strands an IFD1 descriptor in the FIFO and hangs the board.
        dma_first = prefetch_first(self.generation)
        if self.prefetched is None:
            _load_b(_tile_at(0, n_per_tile), 0)
            if dma_first:
                # One joint wait for both shards: an unpaired launch gets its own wait injected
                # right after it, which serialises the two regions' reads of tile 0.
                _wait_b(0)
        # else: tile 0 is in flight into slot 0 since the previous op, waited in implementation().
        with A.tiles(n_blocks, num_tiles=num_n_tiles) as n_tile:
            step, _, count = pipeline_dims(n_tile, n_blocks)
            # A.tiles clamps the last tile's extent; every tile here is full, and the
            # slot buffers are sized for a full one, so rebuild the ramp from the index.
            tile = _tile_at(step, n_per_tile)
            is_more = step + 1 < count
            # NOTE: where on blocks, not A.when (aka if): TVM's scheduling infra does not
            # support the latter. The loop is unrolled, so each arm's parity test folds away.
            # The next tile's DMAs go out before this tile's program (prefetch_first) or after
            # its wait: ReduceBufferLiveness would hoist the prefetch up to the IMC load reading
            # its slot, and the sync between them is a reorder boundary that pins it.
            for slot in range(2):
                on_slot = step % 2 == slot

                def _prefetch(slot=slot, on_slot=on_slot):
                    _load_b(
                        _tile_at(A.min(step + 1, count - 1), n_per_tile),
                        1 - slot,
                        where=tir.And(is_more, on_slot),
                    )

                if dma_first:
                    _prefetch()
                _load_imc(tile, slot, where=on_slot)
                with A.when(on_slot):
                    A.waits(_sync_id(_LOAD_IMC_ID, core, slot))
                if not dma_first:
                    _prefetch()
            self._fused_inner(
                a_tile=a_l1.tile(local_m_ramp, dim=1),
                imc_tile=imc.tile(tile, dim=1),
                dq_tile=dq_l1,
                n_tile=tile,
                out_tile=out_l1.tile(local_m_ramp, dim=0).tile(tile, dim=1),
            )
            # Waits after this tile's compute so both shard DMAs stay in flight across it;
            # every launch in the body precedes every wait, so each slot pairs with its own.
            for slot in range(2):
                with A.when(tir.And(is_more, step % 2 == slot)):
                    _wait_b(1 - slot)

    def _n_resident(self, resident, a_l1, dq_l1, out_l1, imc, local_m_ramp, num_n_tiles) -> None:
        """N-tile loop over a weight band (partly) resident in L1: per tile an IMC program
        from the band's rows, its wait, the fused datapath. The parts not in
        ``resident_parts`` are launched up front in tile order, alternating the channels so
        both DDR regions stream, and each tile waits its own right before its program (the
        waits retire a channel's launches in order). Unrolled: every tile is full."""
        n_blocks = int(self.matrix_b.shape[-2])  # the band, not the (split) output
        n_per_tile = n_blocks // num_n_tiles
        core = self.core
        shards = self._weight_shards(self.matrix_b, self.matrix_b1, self._b_k())
        # Parts the caller launched ahead of the op were waited in implementation(): landed.
        landed = set(
            range(2 * num_n_tiles)
            if self.resident_parts is None
            else map(int, self.resident_parts)
        ) | set(map(int, self.inflight_parts or ()))
        missing = [
            part
            for part in range(2 * num_n_tiles)
            if part % 2 < len(shards) and part not in landed
        ]

        def _id(part):
            return _sync_id(shards[part % 2][4], core, part // 2)

        for i, part in enumerate(missing):
            t, sh = divmod(part, 2)
            src, k_offset, _, _, _ = shards[sh]
            rows = tir.Ramp(t * n_per_tile, 1, n_per_tile)
            stream_shard(
                src,
                rows,
                resident.tile(rows, dim=0),
                k_offset,
                core=core,
                channel=i % 2,
                id=_id(part),
            )
        for step in range(num_n_tiles):
            tile = _tile_at(step, n_per_tile)
            ids = [_id(part) for part in (2 * step, 2 * step + 1) if part in missing]
            if ids:
                A.waits(*ids)
            atomic.load_imc(
                buffer=resident.tile(tir.Ramp(step * n_per_tile, 1, n_per_tile), dim=0),
                imc=imc,
                input_is_transposed=True,
                core=core,
                generation=self.generation,
                id=_sync_id(_LOAD_IMC_ID, core, step),
            )
            A.waits(_sync_id(_LOAD_IMC_ID, core, step))
            self._fused_inner(
                a_tile=a_l1.tile(local_m_ramp, dim=1),
                imc_tile=imc,
                dq_tile=dq_l1,
                n_tile=tile,
                out_tile=out_l1.tile(local_m_ramp, dim=0).tile(tile, dim=1),
            )

    def _fused_inner(self, a_tile, imc_tile, dq_tile, n_tile, out_tile) -> None:
        """Per-(m_tile, n_tile) fused datapath dispatcher.

        Dispatches on n_luts at the Python level — TVM Script's parser walks
        both branches of any if inside A.script() and would try to evaluate
        the LUT path even when luts is None.  Each branch has its own context.
        """
        luts = self.luts
        n_luts = int(luts.shape[0]) if luts is not None else 0
        q = self.fuse_output_quant
        if n_luts == 0:
            fn = self._fused_inner_no_lut_q if q else self._fused_inner_no_lut
        elif n_luts == 1:
            fn = self._fused_inner_single_lut_q if q else self._fused_inner_single_lut
        else:
            fn = self._fused_inner_chained_lut_q if q else self._fused_inner_chained_lut
        fn(a_tile, imc_tile, dq_tile, n_tile, out_tile)

    def _a_tile_dims(self, a_tile):
        """Return (m_tile_size, k_blocks) for the 3D (num_k_splits, m, k_per_split)
        a_tile layout used in both modes (canonical mode has num_k_splits=1)."""
        num_k_splits = int(a_tile.shape[0])
        m_tile_size = a_tile.shape[1]
        k_per_split = int(a_tile.shape[2])
        return m_tile_size, num_k_splits * k_per_split

    def _fused_inner_no_lut(self, a_tile, imc_tile, dq_tile, n_tile, out_tile) -> None:
        m_tile_size, k_blocks = self._a_tile_dims(a_tile)
        num_k_splits = int(a_tile.shape[0])
        k_per_split = int(a_tile.shape[2])
        n_per_tile = imc_tile.shape[1]
        out_dtype = self.output.dtype
        dq_dtype = self.dequant_params.dtype

        with A.script():
            for m in A.serial(m_tile_size):
                with A.Datapath(kind=A.DP_MVMEXE, core=self.core):
                    ifd0_stream = A.decl_stream("int8x64")
                    ifd1_stream = A.decl_stream(dq_dtype)
                    matmul_stream = A.decl_stream("int32x64")
                    dpu_stream = A.decl_stream(out_dtype)

                    # Walk a_tile's 3D layout: outer ks (1 in canonical mode),
                    # inner k_within.  Push order preserves K=0..k_blocks-1.
                    for n in A.serial(n_per_tile):
                        for ks in A.serial(num_k_splits):
                            for k_within in A.serial(k_per_split):
                                with A.IFD(num=0):
                                    ifd0_stream.push(a_tile[ks, m, k_within])

                    for n in A.serial(n_per_tile):
                        with A.pseudo.MatMul():
                            acc = A.mvm_accumulator()
                            A.reset_accumulator(acc)
                            for r in A.serial(k_blocks):
                                acc += A.mvm_1x1(ifd0_stream.pop(), imc_tile[r, n])
                            matmul_stream.push(acc)

                    for c in A.serial(2 * n_per_tile):
                        with A.IFD(num=1):
                            ifd1_stream.push(dq_tile[2 * n_tile.base + c])

                    with A.DPU():
                        regs = A.array("float18x64", 3)
                        for n in A.serial(n_per_tile):
                            regs[0] = A.cast(matmul_stream.pop(), "float18x64")
                            regs[1] = A.cast(ifd1_stream.pop(), "float18x64")
                            regs[2] = A.cast(ifd1_stream.pop(), "float18x64")
                            # DPU compiler requires single-op RHS per BufferStore.
                            regs[0] = regs[0] - regs[1]
                            regs[0] = regs[0] * regs[2]
                            dpu_stream.push(A.cast(regs[0], out_dtype))

                    for n in A.serial(n_per_tile):
                        with A.ODR():
                            out_tile[m, n] = dpu_stream.pop()

    def _fused_inner_single_lut(self, a_tile, imc_tile, dq_tile, n_tile, out_tile) -> None:
        m_tile_size, k_blocks = self._a_tile_dims(a_tile)
        num_k_splits = int(a_tile.shape[0])
        k_per_split = int(a_tile.shape[2])
        n_per_tile = imc_tile.shape[1]
        out_dtype = self.output.dtype
        dq_dtype = self.dequant_params.dtype
        luts = self.luts
        method = self.method

        with A.script():
            for m in A.serial(m_tile_size):
                with A.Datapath(kind=A.DP_MVMEXE, core=self.core):
                    ifd0_stream = A.decl_stream("int8x64")
                    ifd1_stream = A.decl_stream(dq_dtype)
                    matmul_stream = A.decl_stream("int32x64")
                    dpu_stream = A.decl_stream(out_dtype)

                    for n in A.serial(n_per_tile):
                        for ks in A.serial(num_k_splits):
                            for k_within in A.serial(k_per_split):
                                with A.IFD(num=0):
                                    ifd0_stream.push(a_tile[ks, m, k_within])

                    for n in A.serial(n_per_tile):
                        with A.pseudo.MatMul():
                            acc = A.mvm_accumulator()
                            A.reset_accumulator(acc)
                            for r in A.serial(k_blocks):
                                acc += A.mvm_1x1(ifd0_stream.pop(), imc_tile[r, n])
                            matmul_stream.push(acc)

                    for c in A.serial(2 * n_per_tile):
                        with A.IFD(num=1):
                            ifd1_stream.push(dq_tile[2 * n_tile.base + c])

                    with A.DPU():
                        regs = A.array("float18x64", 4)
                        for n in A.serial(n_per_tile):
                            regs[0] = A.cast(matmul_stream.pop(), "float18x64")
                            regs[1] = A.cast(ifd1_stream.pop(), "float18x64")
                            regs[2] = A.cast(ifd1_stream.pop(), "float18x64")
                            # DPU compiler requires single-op RHS per BufferStore.
                            regs[0] = regs[0] - regs[1]
                            regs[0] = regs[0] * regs[2]
                            regs[3] = chained_lut_dpu(regs[0], luts, 1, method)
                            dpu_stream.push(A.cast(regs[3], out_dtype))

                    for n in A.serial(n_per_tile):
                        with A.ODR():
                            out_tile[m, n] = dpu_stream.pop()

    def _fused_inner_chained_lut(self, a_tile, imc_tile, dq_tile, n_tile, out_tile) -> None:
        m_tile_size, k_blocks = self._a_tile_dims(a_tile)
        num_k_splits = int(a_tile.shape[0])
        k_per_split = int(a_tile.shape[2])
        n_per_tile = imc_tile.shape[1]
        out_dtype = self.output.dtype
        dq_dtype = self.dequant_params.dtype
        luts = self.luts
        n_luts = int(luts.shape[0])
        method = self.method

        with A.script():
            for m in A.serial(m_tile_size):
                with A.Datapath(kind=A.DP_MVMEXE, core=self.core):
                    ifd0_stream = A.decl_stream("int8x64")
                    ifd1_stream = A.decl_stream(dq_dtype)
                    matmul_stream = A.decl_stream("int32x64")
                    dpu_stream = A.decl_stream(out_dtype)

                    for n in A.serial(n_per_tile):
                        for ks in A.serial(num_k_splits):
                            for k_within in A.serial(k_per_split):
                                with A.IFD(num=0):
                                    ifd0_stream.push(a_tile[ks, m, k_within])

                    for n in A.serial(n_per_tile):
                        with A.pseudo.MatMul():
                            acc = A.mvm_accumulator()
                            A.reset_accumulator(acc)
                            for r in A.serial(k_blocks):
                                acc += A.mvm_1x1(ifd0_stream.pop(), imc_tile[r, n])
                            matmul_stream.push(acc)

                    for c in A.serial(2 * n_per_tile):
                        with A.IFD(num=1):
                            ifd1_stream.push(dq_tile[2 * n_tile.base + c])

                    with A.DPU():
                        regs = A.array("float18x64", 4)
                        tmp = A.var("float18x64")
                        acc_lut = A.var("float18x64")
                        for n in A.serial(n_per_tile):
                            regs[0] = A.cast(matmul_stream.pop(), "float18x64")
                            regs[1] = A.cast(ifd1_stream.pop(), "float18x64")
                            regs[2] = A.cast(ifd1_stream.pop(), "float18x64")
                            # DPU compiler requires single-op RHS per BufferStore.
                            regs[0] = regs[0] - regs[1]
                            regs[0] = regs[0] * regs[2]
                            regs[3] = chained_lut_dpu(
                                regs[0],
                                luts,
                                n_luts,
                                method,
                                tmp.buffer,
                                acc_lut.buffer,
                            )
                            dpu_stream.push(A.cast(regs[3], out_dtype))

                    for n in A.serial(n_per_tile):
                        with A.ODR():
                            out_tile[m, n] = dpu_stream.pop()

    # ------------------------------------------------------------------ #
    # Output-quant-fused datapaths (fuse_output_quant=True)                #
    #                                                                      #
    # Identical to the plain variants but IFD1 streams FOUR values per     #
    # output block ([dz, ds, q_scale, q_zp], block-interleaved) from the   #
    # one dq buffer, and the DPU applies out = (activation) * q_scale +    #
    # q_zp before the int8 round/clip cast -- folding llm.quantize into    #
    # this op's write. The pop order dz, ds, q_scale, q_zp matches the     #
    # contiguous push, so the single IFD1 iterator stays regular.          #
    # ------------------------------------------------------------------ #
    def _fused_inner_no_lut_q(self, a_tile, imc_tile, dq_tile, n_tile, out_tile) -> None:
        m_tile_size, k_blocks = self._a_tile_dims(a_tile)
        num_k_splits = int(a_tile.shape[0])
        k_per_split = int(a_tile.shape[2])
        n_per_tile = imc_tile.shape[1]
        out_dtype = self.output.dtype  # int8x64 (validated)
        dq_dtype = self.dequant_params.dtype

        with A.script():
            for m in A.serial(m_tile_size):
                with A.Datapath(kind=A.DP_MVMEXE, core=self.core):
                    ifd0_stream = A.decl_stream("int8x64")
                    ifd1_stream = A.decl_stream(dq_dtype)
                    matmul_stream = A.decl_stream("int32x64")
                    dpu_stream = A.decl_stream(out_dtype)

                    for n in A.serial(n_per_tile):
                        for ks in A.serial(num_k_splits):
                            for k_within in A.serial(k_per_split):
                                with A.IFD(num=0):
                                    ifd0_stream.push(a_tile[ks, m, k_within])

                    for n in A.serial(n_per_tile):
                        with A.pseudo.MatMul():
                            acc = A.mvm_accumulator()
                            A.reset_accumulator(acc)
                            for r in A.serial(k_blocks):
                                acc += A.mvm_1x1(ifd0_stream.pop(), imc_tile[r, n])
                            matmul_stream.push(acc)

                    # 4 params per output block: [dz, ds, q_scale, q_zp].
                    for c in A.serial(4 * n_per_tile):
                        with A.IFD(num=1):
                            ifd1_stream.push(dq_tile[4 * n_tile.base + c])

                    with A.DPU():
                        regs = A.array("float18x64", 5)
                        for n in A.serial(n_per_tile):
                            regs[0] = A.cast(matmul_stream.pop(), "float18x64")
                            regs[1] = A.cast(ifd1_stream.pop(), "float18x64")  # dz
                            regs[2] = A.cast(ifd1_stream.pop(), "float18x64")  # ds
                            # DPU compiler requires single-op RHS per BufferStore.
                            regs[0] = regs[0] - regs[1]
                            regs[0] = regs[0] * regs[2]
                            regs[3] = A.cast(ifd1_stream.pop(), "float18x64")  # q_scale
                            regs[4] = A.cast(ifd1_stream.pop(), "float18x64")  # q_zp
                            regs[0] = regs[0] * regs[3]
                            regs[0] = regs[0] + regs[4]
                            dpu_stream.push(A.cast(regs[0], out_dtype))  # round + clip int8

                    for n in A.serial(n_per_tile):
                        with A.ODR():
                            out_tile[m, n] = dpu_stream.pop()

    def _fused_inner_single_lut_q(self, a_tile, imc_tile, dq_tile, n_tile, out_tile) -> None:
        m_tile_size, k_blocks = self._a_tile_dims(a_tile)
        num_k_splits = int(a_tile.shape[0])
        k_per_split = int(a_tile.shape[2])
        n_per_tile = imc_tile.shape[1]
        out_dtype = self.output.dtype
        dq_dtype = self.dequant_params.dtype
        luts = self.luts
        method = self.method

        with A.script():
            for m in A.serial(m_tile_size):
                with A.Datapath(kind=A.DP_MVMEXE, core=self.core):
                    ifd0_stream = A.decl_stream("int8x64")
                    ifd1_stream = A.decl_stream(dq_dtype)
                    matmul_stream = A.decl_stream("int32x64")
                    dpu_stream = A.decl_stream(out_dtype)

                    for n in A.serial(n_per_tile):
                        for ks in A.serial(num_k_splits):
                            for k_within in A.serial(k_per_split):
                                with A.IFD(num=0):
                                    ifd0_stream.push(a_tile[ks, m, k_within])

                    for n in A.serial(n_per_tile):
                        with A.pseudo.MatMul():
                            acc = A.mvm_accumulator()
                            A.reset_accumulator(acc)
                            for r in A.serial(k_blocks):
                                acc += A.mvm_1x1(ifd0_stream.pop(), imc_tile[r, n])
                            matmul_stream.push(acc)

                    for c in A.serial(4 * n_per_tile):
                        with A.IFD(num=1):
                            ifd1_stream.push(dq_tile[4 * n_tile.base + c])

                    with A.DPU():
                        regs = A.array("float18x64", 6)
                        for n in A.serial(n_per_tile):
                            regs[0] = A.cast(matmul_stream.pop(), "float18x64")
                            regs[1] = A.cast(ifd1_stream.pop(), "float18x64")  # dz
                            regs[2] = A.cast(ifd1_stream.pop(), "float18x64")  # ds
                            regs[0] = regs[0] - regs[1]
                            regs[0] = regs[0] * regs[2]
                            regs[3] = chained_lut_dpu(regs[0], luts, 1, method)  # activation
                            regs[4] = A.cast(ifd1_stream.pop(), "float18x64")  # q_scale
                            regs[5] = A.cast(ifd1_stream.pop(), "float18x64")  # q_zp
                            regs[3] = regs[3] * regs[4]
                            regs[3] = regs[3] + regs[5]
                            dpu_stream.push(A.cast(regs[3], out_dtype))

                    for n in A.serial(n_per_tile):
                        with A.ODR():
                            out_tile[m, n] = dpu_stream.pop()

    def _fused_inner_chained_lut_q(self, a_tile, imc_tile, dq_tile, n_tile, out_tile) -> None:
        m_tile_size, k_blocks = self._a_tile_dims(a_tile)
        num_k_splits = int(a_tile.shape[0])
        k_per_split = int(a_tile.shape[2])
        n_per_tile = imc_tile.shape[1]
        out_dtype = self.output.dtype
        dq_dtype = self.dequant_params.dtype
        luts = self.luts
        n_luts = int(luts.shape[0])
        method = self.method

        with A.script():
            for m in A.serial(m_tile_size):
                with A.Datapath(kind=A.DP_MVMEXE, core=self.core):
                    ifd0_stream = A.decl_stream("int8x64")
                    ifd1_stream = A.decl_stream(dq_dtype)
                    matmul_stream = A.decl_stream("int32x64")
                    dpu_stream = A.decl_stream(out_dtype)

                    for n in A.serial(n_per_tile):
                        for ks in A.serial(num_k_splits):
                            for k_within in A.serial(k_per_split):
                                with A.IFD(num=0):
                                    ifd0_stream.push(a_tile[ks, m, k_within])

                    for n in A.serial(n_per_tile):
                        with A.pseudo.MatMul():
                            acc = A.mvm_accumulator()
                            A.reset_accumulator(acc)
                            for r in A.serial(k_blocks):
                                acc += A.mvm_1x1(ifd0_stream.pop(), imc_tile[r, n])
                            matmul_stream.push(acc)

                    for c in A.serial(4 * n_per_tile):
                        with A.IFD(num=1):
                            ifd1_stream.push(dq_tile[4 * n_tile.base + c])

                    with A.DPU():
                        regs = A.array("float18x64", 6)
                        tmp = A.var("float18x64")
                        acc_lut = A.var("float18x64")
                        for n in A.serial(n_per_tile):
                            regs[0] = A.cast(matmul_stream.pop(), "float18x64")
                            regs[1] = A.cast(ifd1_stream.pop(), "float18x64")  # dz
                            regs[2] = A.cast(ifd1_stream.pop(), "float18x64")  # ds
                            regs[0] = regs[0] - regs[1]
                            regs[0] = regs[0] * regs[2]
                            regs[3] = chained_lut_dpu(
                                regs[0],
                                luts,
                                n_luts,
                                method,
                                tmp.buffer,
                                acc_lut.buffer,
                            )
                            regs[4] = A.cast(ifd1_stream.pop(), "float18x64")  # q_scale
                            regs[5] = A.cast(ifd1_stream.pop(), "float18x64")  # q_zp
                            regs[3] = regs[3] * regs[4]
                            regs[3] = regs[3] + regs[5]
                            dpu_stream.push(A.cast(regs[3], out_dtype))

                    for n in A.serial(n_per_tile):
                        with A.ODR():
                            out_tile[m, n] = dpu_stream.pop()

    # ------------------------------------------------------------------ #
    # Emulation                                                            #
    # ------------------------------------------------------------------ #

    def emulate(self) -> None:  # type: ignore[override]
        # numpy buffer shapes (PWORD trailing dim from int8x64 unpacking):
        #   matrix_a:       (..., M, K_BLOCKS, PWORD)
        #   matrix_b (NK):  (..., N_BLOCKS, K_BLOCKS*PWORD, PWORD)
        #   dequant_params: (..., 2*N_BLOCKS, PWORD)
        #   output:         (..., M, N_BLOCKS, PWORD)
        pword = self.matrix_a.shape[-1]

        # Flatten matrix_a's last two dims: (..., M, K_BLOCKS, PWORD) -> (..., M, K).
        # In split layout, matrix_a is (..., num_k_splits, M, K_per_split, PWORD);
        # move the split axis (-4) next to M (-3) so the two K axes are adjacent,
        # then fold: (..., M, num_k_splits, K_per_split, PWORD).
        if self.matrix_a_is_split:
            mat_a = np.moveaxis(self.matrix_a, -4, -3)
            mat_a = mat_a.reshape(
                mat_a.shape[:-3] + (mat_a.shape[-3] * mat_a.shape[-2], mat_a.shape[-1])
            )
        else:
            mat_a = self.matrix_a
        a_flat = mat_a.reshape(mat_a.shape[:-2] + (mat_a.shape[-2] * mat_a.shape[-1],))
        # NK matrix_b numpy: (..., N_BLOCKS, K, PWORD) where PWORD packs N.
        # Move N_BLOCKS adjacent to PWORD so they flatten into N=N_BLOCKS*PWORD,
        # leaving K as the matmul inner dim: -> (..., K, N_BLOCKS, PWORD) -> (..., K, N).
        ndim_b = self.matrix_b.ndim
        perm = list(range(ndim_b - 3)) + [ndim_b - 2, ndim_b - 3, ndim_b - 1]
        # The shards split K, so the reference matmul sees them rejoined.
        matrix_b = (
            self.matrix_b
            if self.matrix_b1 is None
            else np.concatenate((self.matrix_b, self.matrix_b1), axis=-2)
        )
        b_kn = np.transpose(matrix_b, perm)
        b_flat = b_kn.reshape(b_kn.shape[:-2] + (b_kn.shape[-2] * b_kn.shape[-1],))

        # int32 matmul accumulation, then cast to fp18 to match the AIPU path
        # (the int32 MVM accumulator is cast to fp18 before dequant arithmetic).
        # Every subsequent op is rf18-wrapped to match the kernel rounding.
        result = np.matmul(a_flat.astype(np.int32), b_flat.astype(np.int32))
        result_f = emu.rf18(result.astype(np.float32))  # shape (..., M, N)

        # De-interleave dequant_params (..., WIDTH*N_BLOCKS, PWORD) into per-channel
        # dz/ds (..., N_BLOCKS, PWORD), then flatten to align with result_f's
        # last dim (N=N_BLOCKS*PWORD).  WIDTH=4 also carries [q_scale, q_zp] for the
        # fused output quant (applied after the LUT below).
        width = self._param_width()
        dz = self.dequant_params[..., 0::width, :]
        ds = self.dequant_params[..., 1::width, :]
        dz_flat = dz.reshape(dz.shape[:-2] + (dz.shape[-2] * dz.shape[-1],))
        ds_flat = ds.reshape(ds.shape[:-2] + (ds.shape[-2] * ds.shape[-1],))
        # dequant_params has all-1 leading dims (validated); squeeze so
        # broadcast against result_f works regardless of batch shape.
        dz_b = dz_flat.reshape((1,) * (result_f.ndim - 1) + (dz_flat.shape[-1],))
        ds_b = ds_flat.reshape((1,) * (result_f.ndim - 1) + (ds_flat.shape[-1],))

        # Per-step fp18 rounding to match the DPU's float18 arithmetic.
        dz_f18 = emu.rf18(dz_b.astype(np.float32))
        ds_f18 = emu.rf18(ds_b.astype(np.float32))
        dequantized = emu.rf18(result_f - dz_f18)
        dequantized = emu.rf18(dequantized * ds_f18)

        # Optional LUT activation.  Order of operations matches the kernel so
        # rf18 rounding is element-by-element identical.
        if self.luts is not None and self.luts.shape[0] > 0:
            n_luts = int(self.luts.shape[0])
            # luts numpy: (n_luts, PWORD); apply_lut expects (1, PWORD) per call.
            luts_np = self.luts.reshape(n_luts, -1)

            def _apply(x, lut_idx):
                return emu.apply_lut(x, luts_np[lut_idx : lut_idx + 1])

            if n_luts == 1:
                dequantized = _apply(dequantized, 0)
            elif self.method == ChainedLUTMethod.ADDITIVE:
                acc = _apply(dequantized, 0)
                for k in range(1, n_luts):
                    acc = emu.rf18(acc + _apply(dequantized, k))
                dequantized = acc
            else:  # COMPOSITION
                for k in range(n_luts):
                    dequantized = _apply(dequantized, k)

        # Fused output quant: out = round_clip_int8((activation) * q_scale + q_zp), with the
        # same element-by-element fp18 rounding as llm.quantize (round/clip happens in the
        # int8 cast below).  q_scale/q_zp are the odd-pair of the WIDTH=4 dequant table.
        if self.fuse_output_quant:
            qs = self.dequant_params[..., 2::width, :]
            qz = self.dequant_params[..., 3::width, :]
            qs_flat = qs.reshape(qs.shape[:-2] + (qs.shape[-2] * qs.shape[-1],))
            qz_flat = qz.reshape(qz.shape[:-2] + (qz.shape[-2] * qz.shape[-1],))
            qs_b = qs_flat.reshape((1,) * (dequantized.ndim - 1) + (qs_flat.shape[-1],))
            qz_b = qz_flat.reshape((1,) * (dequantized.ndim - 1) + (qz_flat.shape[-1],))
            qs_f18 = emu.rf18(qs_b.astype(np.float32))
            qz_f18 = emu.rf18(qz_b.astype(np.float32))
            dequantized = emu.rf18(emu.rf18(dequantized * qs_f18) + qz_f18)

        # Reshape (..., M, N) -> (..., M, N_BLOCKS, PWORD) to match the output buffer.
        n_blocks_dim = int(self.matrix_b.shape[-3])
        dequantized = dequantized.reshape(dequantized.shape[:-1] + (n_blocks_dim, pword))
        if self.output.dtype == np.int8:
            np.copyto(self.output, emu.round_clip_cast(dequantized.astype(np.float32)))
        else:
            np.copyto(self.output, dequantized.astype(self.output.dtype))


quantized_matmul_activation = register_op(
    QuantizedMatMulActivation,
    "llm.quantized_matmul_activation",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)


def slice_n(buf, axis: int, lo: int, hi: int):
    """RANK-PRESERVING BufferSlice narrowing one axis to [lo:hi] (full ramps elsewhere)."""
    # slice_2d returns a 2D view, dropping the batch dims that ops re-derive their rank from.
    extents = [int(s) for s in buf.shape]
    offsets = [0] * len(extents)
    axis %= len(extents)
    extents[axis] = hi - lo
    offsets[axis] = lo
    return tir.BufferSlice(buf[ramp(tuple(extents), tuple(offsets))])


def bs(buf, *idx) -> tir.BufferSlice:
    """BufferSlice of ``buf`` at ``idx``: a Ramp keeps its dim, anything else indexes one away."""
    return tir.BufferSlice(tir.BufferLoad(buf, list(idx)))


def _dq_band(dequant_params, width: int, lo: int, hi: int):
    """This band's dequant table in a FRESH (offset-0) buffer."""
    # The op re-views dequant_params at offset 0, so a sliced view would read the wrong
    # scales; a standalone same-rank copy makes that re-alias correct.
    shape = [int(s) for s in dequant_params.shape]
    shape[-1] = width * (hi - lo)
    band = A.alloc_buffer(tuple(shape), dtype=dequant_params.dtype, scope=dequant_params.scope())
    atomic.copy(
        slice_n(dequant_params, len(dequant_params.shape) - 1, width * lo, width * hi),
        output=band,
    )
    return band


def caller_buffer(output):
    """A caller-owned output buffer, or None: a ``tir.Buffer`` (whole) or the whole-buffer
    ``BufferSlice`` that ``@sliced`` turns it into."""
    if isinstance(output, tir.Buffer):
        return output
    if isinstance(output, tir.BufferSlice) and [int(x) for x in output.shape] == [
        int(x) for x in output.buffer.shape
    ]:
        return output.buffer
    return None


def _output_info(matrix_a, matrix_b, dequant_params, output, kwargs):
    """The op's own (shape, dtype, scope) inference, honoring a caller dtype override."""
    probe = QuantizedMatMulActivation(matrix_a, matrix_b, dequant_params, output=output, **kwargs)
    info = probe.infer_outputs()["output"]
    dtype = output.dtype if (output is not None and getattr(output, "dtype", None)) else info.dtype
    return tuple(info.shape), dtype, info.scope


@sliced
def quantized_matmul_activation_multicore(
    matrix_a, matrix_b, dequant_params, *, cores, output=None, split_output=False, **kwargs
):
    """N-sharded ``quantized_matmul_activation``: split the output columns
    (N_BLOCKS) across ``cores``.

    Each core runs the unmodified single-core op (pinned via its ``core=`` arg) on
    its contiguous N-band of B / dequant / output, inside an ``A.AICore`` region
    bracketed by ``A.barrier`` -- the same shape as ``core.matmul``'s N-shard. The
    matmul output columns are independent, so this needs no cross-core reduction.
    ``cores`` in {None, 1} (or a single id) falls back to the plain op (unchanged).

    ``split_output`` (default False, so the interface is unchanged) controls the OUTPUT
    LAYOUT:

    - False: one unified ``(*batch, M, N_BLOCKS)`` output; each core writes its column
      band ``out[..., lo:hi]`` -- a STRIDED last-axis slice.  Since Metis has no 2D DMA,
      that write lowers to one DMA descriptor PER M-ROW (num_m_rows x cores tiny copies),
      which dominates and pins the N-shard speedup to ~1.2x even when compute parallelises.
    - True: a ``(cores, M, N_BLOCKS // cores)`` output with the CORE as the OUTER axis, so
      each core writes one CONTIGUOUS ``(M, band)`` slab (a few large DMAs, not per-row).
      Measured ~1.7x faster for the fc1 shape (3-core 1.21x -> 2.05x).  The result is the
      split layout ``core.matmul``'s / this op's ``matrix_a_is_split=True`` consumes with
      NO un-split (the next matmul reads split ``K`` directly), so the win survives into the
      consumer (~+1% ingest cost).  Requires N_BLOCKS % cores == 0 and a (1,)/scalar batch.
    """
    ids = core_ids(cores)

    if len(ids) <= 1:
        return quantized_matmul_activation(
            matrix_a, matrix_b, dequant_params, output=output, **kwargs
        )

    plan = ShardPlan(matrix_a, matrix_b, dequant_params, output, split_output, kwargs, ids)
    _emit_shard_region([plan], ids)
    return plan.out


def _split_n(matrix_b, splits):
    """The per-projection views of an N-concatenated weight (all None for a missing shard)."""
    if matrix_b is None:
        return [None] * len(splits)
    n_axis = len(matrix_b.shape) - 2
    bases = tuple(itertools.accumulate(splits, initial=0))
    if bases[-1] != int(matrix_b.shape[n_axis]):
        raise ValueError(
            f"splits {tuple(splits)} sum to {bases[-1]} blocks, but the weight has "
            f"{int(matrix_b.shape[n_axis])}"
        )
    return [slice_n(matrix_b, n_axis, base, base + n) for base, n in zip(bases, splits)]


class FusedQkvPlan:
    """One N-CONCATENATED weight feeding several outputs, sharded over ``cores``: the shared
    preparation of ``quantized_matmul_activation_multicore_fused`` plus per-core band emission
    a caller can run inside a region of its own. ``matrix_b`` stacks the projections' weights
    along N in ``splits`` block order, while ``dequant_params`` and ``outputs`` stay per
    projection."""

    def __init__(
        self, matrix_a, matrix_b, dequant_params, *, cores, splits, outputs=None, **kwargs
    ):
        self.matrix_a = matrix_a
        self.matrix_b = matrix_b
        self.dequant_params = tuple(dequant_params)
        self.core_ids = core_ids(cores)
        outputs = outputs if outputs is not None else (None,) * len(splits)
        self.splits = tuple(splits)
        self.bases = tuple(itertools.accumulate(splits, initial=0))
        self.n_axis = len(matrix_b.shape) - 2
        self.weights = _split_n(matrix_b, splits)
        self.matrix_b1 = kwargs.pop("matrix_b1", None)
        self.kwargs = kwargs
        # Decode (the pipelined weight stream): a core whose band straddles two projections
        # runs it as ONE pipeline (concatenated dequant table, per-projection output writes);
        # prefill keeps one op per projection.
        self.pipelined = bool(kwargs.get("pipeline_weights", False))
        self.b1_axis = None if self.matrix_b1 is None else len(self.matrix_b1.shape) - 2
        self.shards_b1 = _split_n(self.matrix_b1, splits)
        self.outs = []
        for b, b1, dq, out in zip(self.weights, self.shards_b1, self.dequant_params, outputs):
            # Shape inference validates K, so it needs the shard too.
            shape, dtype, scope = _output_info(matrix_a, b, dq, out, _with_b1(kwargs, b1))
            if caller_buffer(out) is not None:
                if tuple(int(x) for x in out.shape) != tuple(int(x) for x in shape):
                    raise ValueError(f"output shape {tuple(out.shape)} != {tuple(shape)}")
                self.outs.append(caller_buffer(out))
            else:
                self.outs.append(A.alloc_buffer(shape, dtype=dtype, scope=scope))
        # Stage iff the WHOLE fused weight fits the shared L2 budget, not one band (see ShardPlan).
        # K is the SUM of both regions when the weight is split across DDR/DDR1: staging joins the
        # shards in L2, so matrix_b's K alone is half the real footprint. This verdict is forced on
        # the op via stage_weight_l2, which bypasses its own _b_k() check -- so it has to agree.
        full_k = int(matrix_b.shape[-1]) + (
            0 if self.matrix_b1 is None else int(self.matrix_b1.shape[-1])
        )
        full_b_bytes = self.bases[-1] * full_k * get_dtype_size(matrix_b.dtype)
        self.stage_weight_l2 = full_b_bytes <= _b_l2_stage_max_bytes()
        self.width = 4 if kwargs.get("fuse_output_quant", False) else 2
        # A band that does not straddle a split costs ONE weight stream, dequant copy and IMC
        # program sequence whatever it spans; sharding the projections apart pays each per split.
        self.partition = list(shards(self.bases[-1], self.core_ids))
        if len(self.partition) != len(self.core_ids):
            raise ValueError(f"{self.bases[-1]} N blocks cannot be sharded over {self.core_ids}")

    def parts(self, lo, hi):
        return [
            (base, n, dq, out, b1)
            for base, n, dq, out, b1 in zip(
                self.bases, self.splits, self.dequant_params, self.outs, self.shards_b1
            )
            if max(lo, base) < min(hi, base + n)
        ]

    def straddles(self, lo, hi) -> bool:
        """Whether the band [lo, hi) is emitted as ONE pipeline over two projections."""
        return len(self.parts(lo, hi)) > 1 and self.pipelined

    def first_op_band(self, lo, hi):
        """N range of the band's FIRST op (the whole band when it straddles, else the first part)."""
        if self.straddles(lo, hi):
            return lo, hi
        base, n, _, _, _ = self.parts(lo, hi)[0]
        return max(lo, base), min(hi, base + n)

    def tile0(self, core, lo, hi):
        """(n_per_tile, k_lo, k_hi) of the band's first op, for a caller streaming its tile 0
        ahead (``launch_tile0``), or None when that op is not pipelined."""
        if self.matrix_b1 is None or len(self.matrix_b.shape) != 2:
            return None
        op_lo, op_hi = self.first_op_band(lo, hi)
        band_b = slice_n(self.matrix_b, self.n_axis, op_lo, op_hi)
        band_b1 = slice_n(self.matrix_b1, self.b1_axis, op_lo, op_hi)
        # A shape probe only (never emitted): its dequant table is a view of the widest
        # projection's (the plan depends on the table's dtype and width, not its values).
        dq = max(self.dequant_params, key=lambda d: int(d.shape[-1]))
        out = max(self.outs, key=lambda o: int(o.shape[-1]))
        probe = QuantizedMatMulActivation(
            self.matrix_a,
            band_b,
            slice_n(dq, len(dq.shape) - 1, 0, self.width * (op_hi - op_lo)),
            output=slice_n(tir.BufferSlice(out), len(out.shape) - 1, 0, op_hi - op_lo),
            core=core,
            stage_weight_l2=self.stage_weight_l2,
            **_with_b1(self.kwargs, band_b1),
        )
        _, _, pipeline_n, n_per_tile = probe._pipeline_plan()
        if not pipeline_n:
            return None
        return int(n_per_tile), int(band_b.shape[-1]), int(band_b1.shape[-1])

    def alloc_tile0_slot(self, core, lo, hi):
        """The caller-owned L1 slot for this band's tile 0 (or None)."""
        plan = self.tile0(core, lo, hi)
        if plan is None:
            return None
        n_per_tile, k_lo, k_hi = plan
        return A.alloc_buffer((n_per_tile, k_lo + k_hi), dtype=self.matrix_b.dtype, scope=A.L1)

    def launch_tile0(self, core, lo, hi, slot):
        """Stream the band's tile 0 (both K shards, back to back on ONE channel, ids
        xop_ids(core, 0) in that order) into ``slot``; ``emit(..., pre_slot=slot)`` waits them
        in that order. One channel so an op DMA on the other channel between the launch and
        the wait cannot retire them (token FIFO)."""
        op_lo, _ = self.first_op_band(lo, hi)
        rows = tir.Ramp(op_lo, 1, int(slot.shape[0]))
        first_id, second_id = xop_ids(core, 0)
        k_lo = int(self.matrix_b.shape[-1])
        stream_shard(
            self.matrix_b, rows, slot, 0, core=core, channel=_LOAD_B1_CHANNEL, id=first_id
        )
        stream_shard(
            self.matrix_b1, rows, slot, k_lo, core=core, channel=_LOAD_B1_CHANNEL, id=second_id
        )

    def emit(self, core, lo, hi, core_a, pre_slot=None):
        """This core's band [lo, hi) inside an already-open AICore region; ``core_a`` is the
        activation (a region prologue's L1 buffer, or the shared matrix_a); ``pre_slot`` a
        caller-streamed tile 0 with both shards in flight (see launch_tile0)."""
        matrix_b, n_axis, width, kwargs = self.matrix_b, self.n_axis, self.width, self.kwargs
        extra_pre = (
            {"prefetched": pre_slot, "prefetched_both": True} if pre_slot is not None else {}
        )
        parts = self.parts(lo, hi)
        if self.straddles(lo, hi):
            # The band straddles projections: one pipeline over the whole contiguous band of
            # the concatenated weight, its dequant table the parts' tables side by side, its
            # output written per part.
            if len(parts) != 2:
                raise ValueError("a band may straddle at most two projections")
            dq_cat = A.alloc_buffer(
                (1, 1, width * (hi - lo)), dtype=self.dequant_params[0].dtype, scope=A.DDR
            )
            dq_cat_bs = tir.BufferSlice(dq_cat)
            ranges, split_outs, b1_cat = [], [], None
            for base, n, dq, out, b1 in parts:
                part_lo, part_hi = max(lo, base), min(hi, base + n)
                atomic.copy(
                    slice_n(
                        dq, len(dq.shape) - 1, width * (part_lo - base), width * (part_hi - base)
                    ),
                    output=slice_n(dq_cat_bs, 2, width * (part_lo - lo), width * (part_hi - lo)),
                )
                ranges += [part_lo - lo, part_hi - lo]
                split_outs.append(
                    slice_n(
                        tir.BufferSlice(out), len(out.shape) - 1, part_lo - base, part_hi - base
                    )
                )
            if self.matrix_b1 is not None:
                b1_cat = slice_n(self.matrix_b1, self.b1_axis, lo, hi)
            quantized_matmul_activation(
                core_a,
                slice_n(matrix_b, n_axis, lo, hi),
                dq_cat,
                output=split_outs[0],
                core=core,
                stage_weight_l2=self.stage_weight_l2,
                output_splits=tuple(ranges),
                output2=split_outs[1],
                **_with_b1(kwargs, b1_cat),
                **extra_pre,
            )
            return
        for i, (base, n, dq, out, b1) in enumerate(parts):
            part_lo, part_hi = max(lo, base), min(hi, base + n)
            band_b1 = (
                None
                if b1 is None
                else slice_n(b1, len(b1.shape) - 2, part_lo - base, part_hi - base)
            )
            extra = extra_pre if i == 0 else {}
            quantized_matmul_activation(
                core_a,
                slice_n(matrix_b, n_axis, part_lo, part_hi),
                _dq_band(dq, width, part_lo - base, part_hi - base),
                output=slice_n(
                    tir.BufferSlice(out), len(out.shape) - 1, part_lo - base, part_hi - base
                ),
                core=core,
                stage_weight_l2=self.stage_weight_l2,
                **_with_b1(kwargs, band_b1),
                **extra,
            )


@sliced
def quantized_matmul_activation_multicore_fused(
    matrix_a, matrix_b, dequant_params, *, cores, splits, outputs=None, prologue=None, **kwargs
):
    """One N-CONCATENATED weight feeding several outputs, sharded over ``cores``.

    ``prologue(core)``, if given, computes the activation inside each core's region;
    ``matrix_a`` then only carries the shape for inference.
    """
    if len(core_ids(cores)) <= 1:
        outputs = outputs if outputs is not None else (None,) * len(splits)
        shards_b1 = _split_n(kwargs.pop("matrix_b1", None), splits)
        return tuple(
            quantized_matmul_activation(matrix_a, b, dq, output=out, **_with_b1(kwargs, b1))
            for b, b1, dq, out in zip(
                _split_n(matrix_b, splits), shards_b1, dequant_params, outputs
            )
        )

    plan = FusedQkvPlan(
        matrix_a, matrix_b, dequant_params, cores=cores, splits=splits, outputs=outputs, **kwargs
    )
    for core, lo, hi in plan.partition:
        with A.AICore(core=core):
            A.barrier(*plan.core_ids)
            plan.emit(core, lo, hi, prologue(core) if prologue is not None else matrix_a)
            A.barrier(*plan.core_ids)
    return tuple(plan.outs)


def _with_b1(kwargs: dict, matrix_b1) -> dict:
    """``kwargs`` plus the upper-K shard, omitted entirely when there is none."""
    if matrix_b1 is None:
        return kwargs
    return {**kwargs, "matrix_b1": matrix_b1}


class ShardPlan:
    """One matmul's pre-allocated output plus everything its per-core band needs."""

    def __init__(self, matrix_a, matrix_b, dequant_params, output, split_output, kwargs, core_ids):
        self.matrix_a = matrix_a
        self.matrix_b = matrix_b
        self.dequant_params = dequant_params
        self.split_output = split_output
        self.kwargs = kwargs
        self.n_blocks = int(matrix_b.shape[-2])
        # Each core stages its N-band into the SHARED L2, so the budget check must be on the FULL
        # weight: a per-band check would pass while the aggregate overflows. K is the SUM of both
        # regions when the weight is split across DDR/DDR1 -- staging joins the shards in L2, so
        # matrix_b's K alone is half the real footprint, and this verdict is forced on the op via
        # stage_weight_l2, which bypasses its own _b_k() check.
        b1 = kwargs.get("matrix_b1")
        full_k = int(matrix_b.shape[-1]) + (0 if b1 is None else int(b1.shape[-1]))
        full_b_bytes = self.n_blocks * full_k * get_dtype_size(matrix_b.dtype)
        self.stage_weight_l2 = full_b_bytes <= _b_l2_stage_max_bytes()
        # dequant_params carries `width` values per output block, so a core's N-band spans
        # width*[lo:hi] along the interleaved last axis.
        self.width = 4 if kwargs.get("fuse_output_quant", False) else 2

        # Pre-allocate the output. Reuse the op's own shape inference (batch / M / N);
        # honor the caller's requested output dtype (the op infers float16x64).
        out_shape, out_dtype, out_scope = _output_info(
            matrix_a, matrix_b, dequant_params, output, kwargs
        )
        if split_output:
            # (cores, M, band) with core as the OUTER axis: each core's write is one contiguous
            # slab. Needs even N sharding and a (1,)/scalar batch to match the op's output rank.
            if self.n_blocks % len(core_ids) != 0:
                raise ValueError(
                    f"split_output requires N_BLOCKS ({self.n_blocks}) divisible by cores "
                    f"({len(core_ids)})."
                )
            batch = tuple(int(s) for s in out_shape[:-2])
            if batch not in ((), (1,)):
                raise ValueError(f"split_output requires a (1,) or scalar batch; got {batch}.")
            band = self.n_blocks // len(core_ids)
            self.out = A.alloc_buffer(
                (len(core_ids), int(out_shape[-2]), band), dtype=out_dtype, scope=out_scope
            )
        elif caller_buffer(output) is not None:
            # A caller-owned output (e.g. one the caller pins so the memory analyzer sees a
            # lifetime for a buffer that is otherwise only touched inside core regions).
            if tuple(int(x) for x in output.shape) != tuple(int(x) for x in out_shape):
                raise ValueError(f"output shape {tuple(output.shape)} != {tuple(out_shape)}")
            self.out = caller_buffer(output)
        else:
            self.out = A.alloc_buffer(out_shape, dtype=out_dtype, scope=out_scope)
        self.out_bs = tir.BufferSlice(self.out)

        # What partition() needs to price a band: the same values the per-core op will hand
        # _nk_schedule, so both agree on the tile width a band admits.
        self.m = _try_int(matrix_a.shape[-2])
        if kwargs.get("matrix_a_is_split", False):
            splits, k_per_split = _try_int(matrix_a.shape[-3]), _try_int(matrix_a.shape[-1])
            self.k_blocks = None if splits is None or k_per_split is None else splits * k_per_split
        else:
            self.k_blocks = _try_int(matrix_a.shape[-1])
        self.generation = kwargs.get("generation")
        self.pipeline_weights = kwargs.get("pipeline_weights", False)
        self.out_dtype = out_dtype
        self.dq_dtype = dequant_params.dtype

    def _band_tiles(self, band: int) -> int | None:
        """N tiles the per-core schedule gives a band of ``band`` blocks, None if it cannot fit."""
        _, num_n_tiles, _ = _nk_schedule(
            self.m,
            self.k_blocks,
            band,
            self.generation,
            force="m_outer",
            out_dtype=self.out_dtype,
            extra_l1_bytes=self.width * band * get_dtype_size(self.dq_dtype),
            b_slots=2,
            min_n_tiles=PIPELINE_MIN_TILES,
        )
        return num_n_tiles

    def partition(self, core_ids: tuple[int, ...]) -> list[tuple[int, int, int]]:
        """(core, lo, hi) N bands: tile-aligned where that beats an even split.

        ``shards`` divides N_BLOCKS evenly, which says nothing about the IMC tile width the
        per-core schedule can then use -- and a pipelined N loop needs UNIFORM tiles, so the
        width must divide the band. Qwen3.5-9B's head splits 3880 blocks into 485 per core,
        and 485 = 5 * 97 has no divisor within the IMC ceiling of 4 blocks, leaving every tile
        one block wide: 485 tiles per core where 122 would do, and 4x the tiles to schedule at
        compile time. Moving the boundary to 488/484 buys that 4x for 0.8% of load balance,
        which costs nothing real here -- the cores share one DDR and the total bytes are
        identical either way, so a core that finishes a tile early releases its bandwidth.

        Only when pipelining: otherwise the schedule takes the widest fit and tolerates a
        ragged last tile, so alignment gains nothing. And only when an aligned partition
        strictly lowers the busiest core's tile count, so a shape the even split already
        schedules well keeps the exact bands it has today (Qwen3's 151936-token vocabulary is
        2374 blocks = 2 * 1187 with 1187 prime, so its only aligned width is 2, which is worse
        than the 296/297 split already achieves).
        """
        even = list(shards(self.n_blocks, core_ids))
        if not self.pipeline_weights or self.m != 1 or self.k_blocks is None:
            return even
        # Widest tile the IMC can hold. Below 2 there is no width to align to, and at 0 (a K
        # deeper than the IMC) pricing a band would raise rather than answer.
        max_width = _imc_max_blocks(self.generation) // max(1, self.k_blocks)
        if max_width < 2:
            return even
        baseline = max((self._band_tiles(hi - lo) or 0) for _, lo, hi in even)
        best = None
        for width in range(max_width, 1, -1):
            if self.n_blocks % width or self.n_blocks // width < len(core_ids):
                continue
            # Shard the TILE count, not the block count, so every band is a whole number of
            # tiles and the cores keep the balance shards() would have given them.
            bands = [(hi - lo) * width for _, lo, hi in shards(self.n_blocks // width, core_ids)]
            tiles = [self._band_tiles(band) for band in bands]
            if any(t is None for t in tiles) or min(tiles) < PIPELINE_MIN_TILES:
                continue
            key = (max(tiles), sum(tiles))
            if best is None or key < best[0]:
                best = (key, bands)
        if best is None or not 0 < best[0][0] < baseline:
            return even
        offsets = itertools.accumulate(best[1], initial=0)
        return [(core, lo, lo + band) for core, lo, band in zip(core_ids, offsets, best[1])]

    def _band_args(self, core: int, part_idx: int, lo: int, hi: int, dq_band):
        """The single-core op's (args, kwargs) for this core's N band."""
        kwargs = dict(self.kwargs)
        matrix_b1 = kwargs.pop("matrix_b1", None)
        if matrix_b1 is not None:
            kwargs["matrix_b1"] = slice_n(matrix_b1, len(matrix_b1.shape) - 2, lo, hi)
        # split_output writes this core's result to its own contiguous outer-axis slab out[core];
        # otherwise the strided last-axis N band of the unified output.
        core_out = (
            slice_n(self.out_bs, 0, part_idx, part_idx + 1)
            if self.split_output
            else slice_n(self.out_bs, len(self.out_bs.shape) - 1, lo, hi)
        )
        args = (
            self.matrix_a,
            slice_n(self.matrix_b, len(self.matrix_b.shape) - 2, lo, hi),  # N is dim -2 (NK)
            dq_band,
        )
        kwargs.update(
            output=core_out,
            core=core,
            stage_weight_l2=self.stage_weight_l2,  # decided on the FULL weight (shared L2)
        )
        return args, kwargs

    def tile_plan(self, core: int, part_idx: int, lo: int, hi: int) -> tuple[int, int]:
        """(n_per_tile, K) of this core's band: the shape of a slot the caller streams tile 0 into."""
        # The band's dequant table is only needed for its shape here: a view, no copy.
        dq_view = slice_n(
            self.dequant_params,
            len(self.dequant_params.shape) - 1,
            self.width * lo,
            self.width * hi,
        )
        args, kwargs = self._band_args(core, part_idx, lo, hi, dq_view)
        probe = QuantizedMatMulActivation(*args, **kwargs)
        _, _, pipeline_n, n_per_tile = probe._pipeline_plan()
        if not pipeline_n:
            raise ValueError("a tile-0 prefetch needs the pipelined N loop (uniform N tiles).")
        return int(n_per_tile), int(probe._b_k())

    def emit(self, core: int, part_idx: int, lo: int, hi: int, **extra) -> None:
        """Emit this core's N band, inside an already-open AICore region; ``extra`` goes to
        the single-core op (e.g. ``prefetched=`` for a tile 0 the previous op streamed)."""
        dq_band = _dq_band(self.dequant_params, self.width, lo, hi)
        args, kwargs = self._band_args(core, part_idx, lo, hi, dq_band)
        quantized_matmul_activation(*args, **kwargs, **extra)


def core_ids(cores) -> tuple[int, ...]:
    if isinstance(cores, int):
        return tuple(range(cores))
    if cores is None:
        return ()
    return tuple(int(c) for c in cores)


def _emit_shard_region(plans: list[ShardPlan], core_ids: tuple[int, ...]) -> None:
    """Run every plan's band for each core between one barrier pair."""
    partitions = [plan.partition(core_ids) for plan in plans]
    participants = [core for core, _, _ in partitions[0]]
    for bands in partitions[1:]:
        if [core for core, _, _ in bands] != participants:
            raise ValueError(
                "grouped matmuls must shard over the same cores; got "
                f"{[[core for core, _, _ in b] for b in partitions]}"
            )
    for part_idx, core in enumerate(participants):
        with A.AICore(core=core):
            A.barrier(*participants)
            for plan, bands in zip(plans, partitions):
                plan.emit(core, part_idx, bands[part_idx][1], bands[part_idx][2])
            A.barrier(*participants)

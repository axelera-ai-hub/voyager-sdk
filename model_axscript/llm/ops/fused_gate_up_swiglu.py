#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#

# mypy: disable-error-code="attr-defined, index, arg-type, type-var, assignment, call-overload"

"""Fused gate||up qmm + SwiGLU + final quantize (registered as llm.fused_gate_up_swiglu).

Combines the two parallel projections (gate and up) into a single int8 matmul
over a composed (2*N, K) weight matrix.  At each output column the DPU pairs
the consecutive matmul outputs (gate, up), runs SiLU(dequant(gate)) * dequant(up)
end-to-end in registers, then quantizes the product back to int8 — all in one
Datapath, no L1 round-trip between steps.
"""

import contextlib

from axelera.compiler.alto.axol import atomic
from axelera.compiler.alto.axol.common import ramp
from axelera.compiler.alto.axol.core.matmul._common import pipeline_dims, shards
from axelera.compiler.alto.axol.utils.lut_chain import ChainedLUTMethod, chained_lut_dpu
from axelera.compiler.alto.compiler.config import HardwareGeneration
from axelera.compiler.alto.compiler.device import PWORD
from axelera.compiler.alto.compiler.dtypes import Float32x64, Int8x64, MetisFloats, get_dtype_size
from axelera.compiler.alto.compiler.interpreter import emulation as emu
from axelera.compiler.alto.compiler.op_registry import Arg, BufferInfo, ScriptOp, register_op
import axelera.compiler.alto.script as A
import numpy as np
from tvm import ir, tir

from .matmul import PIPELINE_MIN_TILES, _nk_schedule, _resolve_batch, _sync_id, _tile_at, _try_int
from .quantized_matmul_activation import (
    caller_buffer,
    prefetch_first,
    slice_n,
    stream_shard,
    xop_ids,
)

# A DMA's fixed setup cost dominates below this many bytes, so buffering a whole
# band of N tiles in L1 to flush it as one transfer pays for the extra L1.
_BAND_FLUSH_MAX_BYTES = 64 * 1024

# One HP DMA channel sustains only 4 outstanding transactions where a DDR region
# needs 5-6, so the two weight shards take a channel each.
_LOAD_B_CHANNEL = 0
_LOAD_B1_CHANNEL = 1
_LOAD_B_ID = "gu_load_b"
_LOAD_B1_ID = "gu_load_b1"
_LOAD_IMC_ID = "gu_load_imc"


# log2(e). EXP2 is the hardware exponential, so exp(-x) = EXP2(-x * log2(e)).
_LOG2E = 1.44269504088896340736


def gu_pre_ids(core) -> tuple[str, str]:
    """Sync ids of a caller-prefetched gate_up tile 0 (lower K shard, upper K shard)."""
    return f"gu_pre_{core}_lo", f"gu_pre_{core}_hi"


def launch_gu_tile0(gate_up_w, gate_up_w1, pair_row0: int, pair_per_tile: int, slot, core) -> None:
    """Stream gate_up rows [pair_row0, pair_row0 + pair_per_tile) of the FULL weight (both K
    shards, channel 0 / channel 1) into ``slot``, with the ids the op waits (prefetched=)."""
    rows = tir.Ramp(pair_row0, 1, pair_per_tile)
    lo_id, hi_id = gu_pre_ids(core)
    k_lo = int(gate_up_w.shape[-1])
    stream_shard(gate_up_w, rows, slot, 0, core=core, channel=_LOAD_B_CHANNEL, id=lo_id)
    stream_shard(gate_up_w1, rows, slot, k_lo, core=core, channel=_LOAD_B1_CHANNEL, id=hi_id)


def _silu_expr(x, silu_consts, luts, n_luts, method, tmp, acc_lut, is_europa):
    """SiLU(x) emitted inside an existing ``A.DPU()`` block; returns the result expr.

    A *dispatched Python function*, like chained_lut_dpu, and for the same two reasons.
    An inline ``if`` would not work: inside a ``with A.script()`` body TVMScript parses
    ``if`` as IR and evaluates BOTH arms, so it would dereference the constants only
    EUROPA stages. And the statements cannot be folded into one expression: the DPU TIR
    parser requires a single operation per BufferStore RHS, hence the explicit stores
    through ``tmp``.

    EUROPA uses x * RECIP(1 + EXP2(-x*log2e)) -- the hardware exponential and
    reciprocal, so no LUT table and no calibrated range, and it stays inside the fused
    datapath rather than costing an L1 round-trip on the 2*intermediate-wide tensor.
    """
    if is_europa:
        neg_log2e = A.cast(silu_consts[0], "float18x64")
        one = A.cast(silu_consts[1], "float18x64")
        A.buffer_store(tmp, x * neg_log2e, [0])
        A.buffer_store(tmp, A.dpu.ops.exp2(tmp[0]), [0])
        A.buffer_store(tmp, tmp[0] + one, [0])
        A.buffer_store(tmp, A.dpu.ops.recip(tmp[0]), [0])
        return x * tmp[0]
    return chained_lut_dpu(x, luts, n_luts, method, tmp, acc_lut)


class FusedGateUpSwiGLU(ScriptOp):
    """Fused dual-projection matmul + SwiGLU + int8 output quantize.

    Compute (per output element)::

        gate_int = matmul(A, gate_W)[m, j]
        up_int   = matmul(A, up_W)[m, j]
        gate_dq  = (gate_int - gate_dz[j]) * gate_ds[j]
        up_dq    = (up_int   - up_dz[j])   * up_ds[j]
        out[m, j] = round_clip_cast(
            (SiLU(gate_dq) * up_dq) * down_qs[j] + down_qz[j]
        )

    All arithmetic between the matmul and the final cast stays in DPU
    registers (fp18), so the dequants / SiLU / product / quantize chain
    only touches L1 once at the int8 store.

    Layout requirements:

    matrix_b (the composed weight):
        NK layout, shape (*batch, 2 * N_BLOCKS, K_BLOCKS * PWORD).
        Row 2*j is gate_W row j; row 2*j+1 is up_W row j.  Build
        caller-side with e.g.::

            composed = np.empty((2 * N_BLOCKS, K, 64), int8)
            composed[0::2] = gate_w
            composed[1::2] = up_w

    dequant_params (the combined dequant + output-quantize table):
        Shape (*ones, 6 * N_BLOCKS), layout per logical output j::

            [gate_dz[j], gate_ds[j], up_dz[j], up_ds[j],
             down_qs[j], down_qz[j]]

        contiguous in j.  Matmul dequant (4 floats per logical N) and
        output-quantize (2 floats per logical N) are combined so a single
        IFD1 stream feeds all of them in a single command.

    luts:
        SiLU LUT in standard PWLA format, shape (n_luts,).

    Schedule: NK n_outer with split-N output materialisation (same trick
    as core.matmul_split_n).  The output's leading N-tile axis is
    materialised so each per-tile DMA write is one contiguous slab —
    avoiding the per-row N-stride that the canonical (M, N) layout forces.
    n_per_tile (in matmul-output units, on the 2 * N_BLOCKS scale) must be
    even so each (gate, up) pair stays within the same N tile, and
    pair_blocks must be divisible by num_n_tiles so every tile is uniform.

    Output shape is (*batch, num_n_tiles, M, N_BLOCKS // num_n_tiles).
    The downstream consumer (typically a qmm_act on the down projection)
    must accept matrix_a in this split layout — see
    quantized_matmul_activation's matrix_a_is_split kwarg.
    """

    matrix_a = Arg(
        Int8x64[tir.Buffer, "..."],
        description="Quantized A operand, layout (*batch, M, K_BLOCKS), int8x64.",
    )
    matrix_b = Arg(
        Int8x64[tir.Buffer, "..."],
        description=(
            "Composed gate||up weight in NK layout: "
            "(*batch, 2*N_BLOCKS, K_BLOCKS*PWORD), int8x64.  Row 2*j "
            "is gate_W row j; row 2*j+1 is up_W row j."
        ),
    )
    dequant_params = Arg(
        MetisFloats[tir.Buffer, "..."],
        description=(
            "Combined matmul-dequant + output-quantize table, last dim "
            "6 * N_BLOCKS carrying [gate_dz, gate_ds, up_dz, up_ds, "
            "down_qs, down_qz] per logical output channel.  Streamed as "
            "a single IFD1 command.  All leading dims must be 1."
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
            "weight bandwidth."
        ),
    )
    pipeline_weights = Arg(
        bool,
        default=False,
        kw=True,
        description=(
            "Software-pipeline the N loop so tile i+1's weight DMAs run under tile i's "
            "datapath. b_l1 then holds two prefetch slots rather than the band, so the "
            "schedule is charged for them via _nk_schedule's b_slots. Requires a single "
            "M tile: a tiled M issues its own DMAs under the compute and they contend "
            "for the token line the prefetch holds open."
        ),
    )
    luts = Arg(
        Float32x64[tir.Buffer, "n_luts"],
        kw=True,
        description="SwiGLU's SiLU LUT in standard PWLA format, (n_luts,) float32x64.",
    )
    method = Arg(
        ChainedLUTMethod,
        default=ChainedLUTMethod.ADDITIVE,
        kw=True,
        description="Chaining strategy for multi-LUT SiLU (typically ADDITIVE).",
    )
    output = Arg(
        Int8x64[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description=(
            "int8 output in split-N layout: "
            "(*batch, num_n_tiles, M, N_BLOCKS // num_n_tiles).  "
            "N_BLOCKS is half matrix_b.shape[-2] (one int8 per "
            "(gate, up) pair).  num_n_tiles is chosen by the n_outer "
            "schedule heuristic and must divide N_BLOCKS."
        ),
    )
    core = Arg(
        int | None,
        default=None,
        kw=True,
        description="Optional AIPU core to run on.",
    )
    cores = Arg(
        int | tuple[int, ...] | None,
        default=None,
        kw=True,
        description="AIPU cores to distribute the output N-tiles across (int -> 0..n-1, "
        "tuple -> ids, None/<=1 -> single core). Each core owns whole num_n_tiles tiles.",
    )
    a_prologue = Arg(
        object,
        default=None,
        kw=True,
        description="Band mode: ``a_prologue(core)`` runs inside the core's region and returns "
        "the activation it reads (an L1 buffer shaped like matrix_a, which then only carries "
        "the shape).",
    )
    band_tiles = Arg(
        tuple[int, ...] | None,
        default=None,
        kw=True,
        description=(
            "Band mode: run only this core's N tiles [lo, hi) inside the caller's open AICore "
            "region, reading ``a_prologue``'s activation and writing the caller's split ``output``."
        ),
    )
    xop_b = Arg(
        Int8x64[tir.Buffer, "..."] | None,
        default=None,
        kw=True,
        description=(
            "Cross-op prefetch: the NEXT matmul's weight (lower K shard). Its tile 0 on this "
            "core (rows xop_rows) is streamed into xop_slot during this op's last tile."
        ),
    )
    xop_b1 = Arg(
        Int8x64[tir.Buffer, "..."] | None,
        default=None,
        kw=True,
        description="Cross-op prefetch: the next matmul's upper K shard.",
    )
    xop_rows = Arg(
        tuple[int, ...] | None,
        default=None,
        kw=True,
        description="Cross-op prefetch: (first row, rows) of xop_b/xop_b1 that are the next tile 0.",
    )
    xop_slot = Arg(
        Int8x64[tir.Buffer, "..."] | None,
        default=None,
        kw=True,
        description="Cross-op prefetch: the next matmul's L1 slot 0, (rows, K_next).",
    )
    prefetched = Arg(
        Int8x64[tir.Buffer, "..."] | None,
        default=None,
        kw=True,
        description=(
            "Band mode: caller-owned L1 slot (pair_per_tile, K) holding this band's tile 0 (both "
            "K shards, ids gu_pre_ids(core)), waited where tile 0 would have loaded and used as "
            "pipeline slot 0."
        ),
    )
    generation = Arg(
        HardwareGeneration | None,
        default=None,
        kw=True,
        description="Optional hardware generation.",
    )

    # ------------------------------------------------------------------ #
    # Shape helpers                                                        #
    # ------------------------------------------------------------------ #

    def _b_pair_blocks(self) -> tir.PrimExpr:
        """matrix_b's N axis — the number of matmul-output PWORD-blocks
        (2 * N_BLOCKS, since gate and up are interleaved)."""
        return self.matrix_b.shape[-2]

    def _b_k(self) -> tir.PrimExpr:
        """Full K, summing both shards when matrix_b1 carries the upper range."""
        if self.matrix_b1 is None:
            return self.matrix_b.shape[-1]
        return self.matrix_b.shape[-1] + self.matrix_b1.shape[-1]

    def _n_blocks(self) -> tir.PrimExpr:
        """Logical N_BLOCKS of the output (half _b_pair_blocks)."""
        return self.matrix_b.shape[-2] // 2

    def _stage_lut_l1(self):
        """Copy the SiLU LUT DDR->L1 once and return the L1 buffer.  Passed to
        _fused_inner so the DPU reads the resident copy; otherwise the datapath
        references the DDR LUT and the stager re-loads it (a DMA + blocking sync)
        on every N tile.

        Returns None on EUROPA, whose datapath computes SiLU from EXP2/RECIP and never
        reads the table -- staging it there would be a DMA for nothing."""
        if self._europa_silu():
            return None
        lut_l1 = A.alloc_buffer(tuple(self.luts.shape), dtype=self.luts.dtype, scope=A.L1)
        atomic.copy(self.luts, scope=A.L1, output=lut_l1, generation=self.generation)
        return lut_l1

    def _europa_silu(self) -> bool:
        """Whether SiLU is computed from EXP2/RECIP instead of the LUT chain: decided by
        ``self.generation`` (the explicit one, else the configured one at construction; the two
        paths differ numerically)."""
        return self.generation == HardwareGeneration.EUROPA

    def _stage_silu_consts_l1(self):
        """Copy [-log2(e), 1.0] -- the EUROPA EXP2/RECIP SiLU's constants -- DDR->L1 once
        and return the L1 buffer, for the same reason the LUT is staged: left in DDR the
        datapath references it in place and the stager re-loads it on every N tile.

        Those per-tile reloads are not merely wasteful, they are incorrect under
        pipeline_weights: the prefetched weight DMA is deliberately still in flight
        across the tile body, and a second DMA there produces the same
        (HP_DMA_0 -> SW) token, which the ATEX token validator rejects.

        The constant is allocated in DDR and DMA'd into a plain L1 buffer, NOT allocated
        with scope=A.L1: an L1 CONSTANT makes the codegen emit an `l1_const_data[]` array
        that the graph codegen names identically in every kernel (get_const_data_c_array
        takes a variable_name no call site sets), so two kernels in one graph model that
        both have L1 constants fail to link with "redefinition of 'l1_const_data'" -- and
        the MLP is in both the linear_attention and full_attention kernels."""
        if not self._europa_silu():
            return None
        consts_ddr = A.alloc_constant(
            np.array([[-_LOG2E] * PWORD, [1.0] * PWORD], np.float32),
            dtype="float32x64",
            scope=A.DDR,
        )
        if not self.pipeline_weights:
            return consts_ddr
        consts_l1 = A.alloc_buffer((2,), dtype="float32x64", scope=A.L1)
        atomic.copy(consts_ddr, scope=A.L1, output=consts_l1, generation=self.generation)
        return consts_l1

    def _b_slots(self) -> int:
        """B tiles live at once: two when the N loop prefetches the next tile."""
        return 2 if self.pipeline_weights else 1

    def _weight_shards(
        self,
    ) -> list[tuple[tir.Buffer, tir.PrimExpr, tir.PrimExpr, int | None, str]]:
        """Per K shard of the weight: (DDR operand, K offset, K extent, DMA channel, id)."""
        k_lo = self.matrix_b.shape[-1]
        if self.matrix_b1 is None:
            return [(self.matrix_b, 0, k_lo, None, _LOAD_B_ID)]
        return [
            (self.matrix_b, 0, k_lo, _LOAD_B_CHANNEL, _LOAD_B_ID),
            (self.matrix_b1, k_lo, self.matrix_b1.shape[-1], _LOAD_B1_CHANNEL, _LOAD_B1_ID),
        ]

    def _a_source(self, a_src, single_m_tile: bool):
        """A's source: the multicore L2 copy, DDR for a single M tile, else an L2 stage."""
        if a_src is not None:
            return a_src
        if single_m_tile:
            return self.matrix_a
        # A tiled M is staged so its per-tile reloads are L2 hits rather than DDR re-reads.
        return atomic.copy(self.matrix_a, scope=A.L2)

    def _alloc_b_l1(self, rows) -> list[tir.Buffer]:
        """L1 weight staging, one SEPARATE buffer per pipeline slot of ``rows`` N blocks."""
        # A command's hazard region is the whole backing buffer, so slots sharing an
        # allocation make the prefetch collide with the IMC load it should run under.
        return [
            A.alloc_buffer((rows, self._b_k()), dtype=self.matrix_b.dtype, scope=A.L1)
            for _ in range(self._b_slots())
        ]

    def n_outer_tiles(self) -> tuple[int, int | None]:
        """(num_n_tiles, num_m_tiles) of the n_outer schedule, L1 reserved for the dq+q table.

        ``num_m_tiles`` is None when one M tile fits. Every tile is the same width (the split
        output's slabs are uniform) and holds whole (gate, up) pairs."""
        m, k_blocks, pair_blocks = (
            self.matrix_a.shape[-2],
            self.matrix_a.shape[-1],
            self.matrix_b.shape[-2],
        )
        pair_blocks_int = _try_int(pair_blocks)
        if pair_blocks_int is None:
            raise ValueError(
                "fused_gate_up_swiglu requires a statically-known pair_blocks (= 2 * N_BLOCKS)."
            )
        # get_dtype_size reports the lane-vector size per element (e.g. 256 B for float32x64),
        # so byte count = num_elements * itemsize.
        dq_bytes = 6 * (pair_blocks_int // 2) * get_dtype_size(self.dequant_params.dtype)
        schedule, num_n_tiles, num_m_tiles = _nk_schedule(
            m,
            k_blocks,
            pair_blocks,
            self.generation,
            force="n_outer",
            out_dtype="int8x64",
            extra_l1_bytes=dq_bytes,
            min_n_per_tile=2,  # each (gate, up) pair must stay inside one tile
            b_slots=self._b_slots(),
            require_divisible=True,
            min_n_tiles=PIPELINE_MIN_TILES if self.pipeline_weights else 1,
        )
        if schedule != "n_outer" or num_n_tiles is None:
            raise ValueError(
                "fused_gate_up_swiglu requires a static n_outer schedule, "
                "but the heuristic could not produce one for shapes "
                f"M={m}, K_BLOCKS={k_blocks}, pair_blocks={pair_blocks}."
            )
        if pair_blocks_int % num_n_tiles != 0:
            raise ValueError(
                f"pair_blocks ({pair_blocks_int}) must be divisible by "
                f"num_n_tiles ({num_n_tiles}) so all tiles are uniform."
            )
        if (pair_blocks_int // num_n_tiles) % 2 != 0:
            raise ValueError(
                f"pair_blocks per N tile ({pair_blocks_int // num_n_tiles}) must be even "
                f"so each (gate, up) pair stays within one tile."
            )
        return int(num_n_tiles), num_m_tiles

    def _band_flush_tiles(self, m, band_n_tiles: int, num_m_tiles, n_per_tile: int) -> int:
        """N tiles buffered in L1 before the output is flushed: the whole band, or 1."""
        # A single M tile makes the band one contiguous slab of the split output, so it can
        # leave as ONE DMA while the slab is small enough for per-tile DMAs to be setup bound.
        m_int = _try_int(m)
        if num_m_tiles is not None or m_int is None:
            return 1
        band_bytes = m_int * band_n_tiles * n_per_tile * get_dtype_size("int8x64")
        return band_n_tiles if band_bytes <= _BAND_FLUSH_MAX_BYTES else 1

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
        if len(self.matrix_a.shape) < 2:
            raise ValueError(f"matrix_a must have >= 2 dims, got {len(self.matrix_a.shape)}.")
        if len(self.matrix_b.shape) < 2:
            raise ValueError(f"matrix_b must have >= 2 dims, got {len(self.matrix_b.shape)}.")
        _resolve_batch(self.matrix_a, self.matrix_b)

        if not ir.structural_equal(self.matrix_a.shape[-1] * PWORD, self._b_k()):
            raise ValueError(
                f"Inner dimensions of matrix A ({self.matrix_a.shape[-1] * PWORD}) "
                f"and matrix B ({self._b_k()}) do not match."
            )
        if self.matrix_a.dtype != "int8x64":
            raise ValueError(f"matrix_a dtype must be int8x64, got {self.matrix_a.dtype}.")
        if self.matrix_b.dtype != "int8x64":
            raise ValueError(f"matrix_b dtype must be int8x64, got {self.matrix_b.dtype}.")
        if self.matrix_b1 is not None:
            self._validate_matrix_b1()
        if (self.band_tiles is None) != (self.a_prologue is None):
            raise ValueError("band_tiles and a_prologue come together")
        xop = (self.xop_b, self.xop_b1, self.xop_rows, self.xop_slot)
        if any(x is not None for x in xop):
            if any(x is None for x in xop):
                raise ValueError("xop_b, xop_b1, xop_rows and xop_slot come together")
            if self.band_tiles is None:
                raise ValueError(
                    "a cross-op prefetch (xop_slot) needs the banded body (band_tiles)"
                )

        # matrix_b's N axis must be even (the gate/up pairing requires it).
        pair_blocks = _try_int(self._b_pair_blocks())
        if pair_blocks is None or pair_blocks % 2 != 0:
            raise ValueError(
                f"matrix_b's penultimate dim must be even (2 * N_BLOCKS); "
                f"got {self._b_pair_blocks()}."
            )

        # dequant_params: last dim == 6 * N_BLOCKS (4 dq + 2 q per output).
        n_blocks = pair_blocks // 2
        expected_dq = 6 * n_blocks
        if not ir.structural_equal(self.dequant_params.shape[-1], expected_dq):
            raise ValueError(
                f"dequant_params last dim must be 6 * N_BLOCKS = "
                f"{expected_dq}, got {self.dequant_params.shape[-1]}."
            )
        for i, dim in enumerate(self.dequant_params.shape[:-1]):
            if not ir.structural_equal(dim, 1):
                raise ValueError(
                    f"dequant_params leading dims must all be 1; " f"dim {i} = {dim}."
                )

        if isinstance(self.output, tir.Buffer) and self.output.dtype != "int8x64":
            raise ValueError(f"output dtype must be int8x64, got {self.output.dtype}.")

        if self.luts is None or _try_int(self.luts.shape[0]) in (0, None):
            raise ValueError("luts is required for the SwiGLU SiLU activation.")

    def infer_outputs(self) -> dict[str, BufferInfo]:
        batch_dims, _, _ = _resolve_batch(self.matrix_a, self.matrix_b)
        m = self.matrix_a.shape[-2]
        n_blocks_int = _try_int(self._n_blocks())
        if n_blocks_int is None:
            raise ValueError("fused_gate_up_swiglu requires a statically-known N_BLOCKS.")
        num_n_tiles, _ = self.n_outer_tiles()
        n_per_tile = n_blocks_int // num_n_tiles
        return {
            "output": BufferInfo(
                shape=tuple(batch_dims) + (num_n_tiles, m, n_per_tile),
                dtype="int8x64",
                scope=self.matrix_a.scope(),
            )
        }

    # ------------------------------------------------------------------ #
    # Implementation                                                       #
    # ------------------------------------------------------------------ #

    def implementation(self) -> None:
        batch_dims, broadcast_a, broadcast_b = _resolve_batch(self.matrix_a, self.matrix_b)

        m = self.matrix_a.shape[-2]
        k_blocks = self.matrix_a.shape[-1]
        pair_blocks = self.matrix_b.shape[-2]  # 2 * N_BLOCKS

        # n_outer schedule with split-N output materialisation: B/IMC are
        # loaded once per N tile, M is the inner loop, and the per-tile
        # DMA write goes to a contiguous slab in the output's split layout.
        # n_outer_tiles reserves L1 for the combined dq+q table.
        num_n_tiles, num_m_tiles = self.n_outer_tiles()
        num_m_tiles = num_m_tiles if (num_m_tiles or 0) > 1 else None
        if self.pipeline_weights and num_m_tiles is not None:
            # A tiled M re-reads A and flushes its tile inside the compute the prefetch
            # runs under, and those DMAs contend for the token line it holds open.
            raise ValueError(
                "fused_gate_up_swiglu cannot pipeline weights with a tiled M "
                f"(num_m_tiles={num_m_tiles}); M must fit one L1 tile."
            )

        pair_blocks_int = _try_int(pair_blocks)
        pair_per_tile = pair_blocks_int // num_n_tiles
        n_per_tile = pair_per_tile // 2  # logical output channels per N tile

        # Cross-check against the resolved output buffer's split axis.
        out_split = _try_int(self.output.shape[-3])
        if out_split != num_n_tiles:
            raise ValueError(
                f"output split axis ({out_split}) does not match the "
                f"scheduled num_n_tiles ({num_n_tiles})."
            )

        # Multicore: distribute the output N-tiles across cores. Each core owns whole
        # tiles [t_lo, t_hi) -> a contiguous N-band of the (full) matrix_b / output, with
        # band-local L1 buffers and a fresh per-core dequant band. The single-core path
        # below is left byte-identical (the offsets/core default to 0 / self.core).
        # ScriptOp wraps a Python-int arg as a tir IntImm (not a Python int), so test
        # for iterability rather than isinstance(int): scalar count -> 0..n-1; iterable
        # (tuple / tir Array) -> ids; None -> single-core.
        if self.cores is None:
            core_ids = ()
        else:
            try:
                core_ids = tuple(int(c) for c in self.cores)
            except TypeError:
                core_ids = tuple(range(int(self.cores)))
        if len(core_ids) > 1:
            self._implementation_multicore(
                core_ids,
                batch_dims,
                broadcast_a,
                broadcast_b,
                m,
                k_blocks,
                pair_blocks,
                pair_blocks_int,
                num_n_tiles,
                num_m_tiles,
                pair_per_tile,
                n_per_tile,
            )
            return
        if self.band_tiles is not None:
            # One core's band of the multicore schedule, emitted into the caller's region.
            t_lo, t_hi = (int(t) for t in self.band_tiles)
            self._emit_band(
                self.core,
                t_lo,
                t_hi,
                None,
                self._dq_flat_full(pair_blocks_int // 2),
                batch_dims,
                broadcast_a,
                broadcast_b,
                m,
                k_blocks,
                num_m_tiles,
                pair_per_tile,
                n_per_tile,
            )
            return

        a_l1 = A.alloc_buffer((m, k_blocks), dtype=self.matrix_a.dtype, scope=A.L1)
        b_l1 = self._alloc_b_l1(pair_per_tile if self.pipeline_weights else pair_blocks)
        n_blocks = pair_blocks // 2
        # out_l1 holds one flush's tiles back to back, so its per-row stride matches the DDR
        # split slot's stride and the flush is a single contiguous DMA rather than one per M row.
        flush_tiles = self._band_flush_tiles(m, num_n_tiles, num_m_tiles, n_per_tile)
        out_l1 = A.alloc_buffer(
            (flush_tiles * m, n_per_tile),
            dtype="int8x64",
            scope=A.L1,
        )

        # Stage the combined dq+q table in L1 (tiny; broadcast over M).
        # One buffer means a single IFD1 stream feeds all 6 floats per
        # logical output to the DPU.
        dq_l1 = A.alloc_buffer(
            (n_blocks * 6,),
            dtype=self.dequant_params.dtype,
            scope=A.L1,
        )
        dq_flat = A.alloc_buffer(
            (n_blocks * 6,),
            dtype=self.dequant_params.dtype,
            scope=self.dequant_params.scope(),
            data=self.dequant_params.buffer.data,
        )
        atomic.copy(dq_flat, scope=A.L1, output=dq_l1, generation=self.generation)

        # Stage the SiLU LUT into L1 ONCE (tiny, N-loop-invariant) -- otherwise the
        # datapath references the DDR LUT and the stager re-loads it every N tile.
        lut_l1 = self._stage_lut_l1()
        silu_c = self._stage_silu_consts_l1()

        imc = A.alloc_imc((k_blocks, pair_blocks))

        if batch_dims:
            with A.grid(*batch_dims) as batch_vars_:
                batch_vars = A.to_tuple(batch_vars_)
                ba = tuple(broadcast_a) if broadcast_a is not None else batch_vars
                bb = tuple(broadcast_b) if broadcast_b is not None else batch_vars
                self._compute_n_outer(
                    ba,
                    bb,
                    batch_vars,
                    a_l1,
                    b_l1,
                    dq_l1,
                    lut_l1,
                    out_l1,
                    imc,
                    num_n_tiles,
                    num_m_tiles,
                    pair_per_tile,
                    n_per_tile,
                    silu_c=silu_c,
                )
        else:
            self._compute_n_outer(
                (),
                (),
                (),
                a_l1,
                b_l1,
                dq_l1,
                lut_l1,
                out_l1,
                imc,
                num_n_tiles,
                num_m_tiles,
                pair_per_tile,
                n_per_tile,
                silu_c=silu_c,
            )

    def _implementation_multicore(
        self,
        core_ids,
        batch_dims,
        broadcast_a,
        broadcast_b,
        m,
        k_blocks,
        pair_blocks,
        pair_blocks_int,
        num_n_tiles,
        num_m_tiles,
        pair_per_tile,
        n_per_tile,
    ) -> None:
        """Shard the num_n_tiles output tiles across cores. Each core runs the same
        n_outer compute as the single-core path on its band of tiles, reading a
        contiguous N-band of the full matrix_b (offset) and writing its band of the
        full output (tile-offset), with band-local L1 and a fresh per-core dq band."""
        n_blocks = pair_blocks_int // 2
        dq_flat_full = self._dq_flat_full(n_blocks)
        partitions = list(shards(num_n_tiles, core_ids))
        participants = [c for c, _, _ in partitions]
        first_core = participants[0]
        # The activation A is needed in full by every N-shard (N-sharding splits the
        # output columns, not the contraction).  Reading A from DDR once per core would
        # put core_count x |A| of redundant traffic on the shared 24 GB/s DDR bus -- a
        # cost that GROWS with the core count and, past a handful of cores, exceeds the
        # (N-sharded, read-once) weight traffic.  Instead stage A into L2 EXACTLY ONCE
        # (on the first participant) and let every core read it from the per-core,
        # ~50 GB/s L2 bus.  The barrier after the stage orders the single DDR->L2 copy
        # before any core reads it.
        a_l2 = A.alloc_buffer(tuple(self.matrix_a.shape), dtype=self.matrix_a.dtype, scope=A.L2)
        for core, t_lo, t_hi in partitions:
            with A.AICore(core=core):
                A.barrier(*participants)
                if core == first_core:
                    atomic.copy(self.matrix_a, scope=A.L2, output=a_l2, generation=self.generation)
                A.barrier(*participants)
                self._emit_band(
                    core,
                    t_lo,
                    t_hi,
                    a_l2,
                    dq_flat_full,
                    batch_dims,
                    broadcast_a,
                    broadcast_b,
                    m,
                    k_blocks,
                    num_m_tiles,
                    pair_per_tile,
                    n_per_tile,
                )
                A.barrier(*participants)

    def _dq_flat_full(self, n_blocks):
        """Full 1D re-view of the dequant table (offset 0), sliced per core into the band."""
        return A.alloc_buffer(
            (n_blocks * 6,),
            dtype=self.dequant_params.dtype,
            scope=self.dequant_params.scope(),
            data=self.dequant_params.buffer.data,
        )

    def _emit_band(
        self,
        core,
        t_lo,
        t_hi,
        a_l2,
        dq_flat_full,
        batch_dims,
        broadcast_a,
        broadcast_b,
        m,
        k_blocks,
        num_m_tiles,
        pair_per_tile,
        n_per_tile,
    ) -> None:
        """One core's band of N tiles [t_lo, t_hi), inside an already-open AICore region;
        ``a_l2`` is the L2 copy of A every core reads, or None with an ``a_prologue``."""
        band_n_tiles = t_hi - t_lo
        band_pair = band_n_tiles * pair_per_tile
        band_nblk = band_n_tiles * n_per_tile
        b_l1 = self._alloc_b_l1(pair_per_tile if self.pipeline_weights else band_pair)
        pre_slot = caller_buffer(self.prefetched)
        if pre_slot is not None:
            if not self.pipeline_weights:
                raise ValueError("prefetched needs pipeline_weights")
            want = (int(pair_per_tile), int(self._b_k()))
            if tuple(int(x) for x in pre_slot.shape) != want:
                raise ValueError(f"prefetched slot shape {tuple(pre_slot.shape)} != {want}")
            b_l1[0] = pre_slot
            # Only the channel-0 shard is waited before the prologue (whose own DMAs are
            # channel 0's, and a wait retires the oldest launch on its channel); the
            # channel-1 shard keeps streaming under the prologue and is waited after it,
            # before the first IMC program.
            A.waits(gu_pre_ids(core)[0])
        a_src = self.a_prologue(core) if self.a_prologue is not None else a_l2
        if pre_slot is not None:
            A.waits(gu_pre_ids(core)[1])
        a_l1 = A.alloc_buffer((m, k_blocks), dtype=self.matrix_a.dtype, scope=A.L1)
        flush_tiles = self._band_flush_tiles(m, band_n_tiles, num_m_tiles, n_per_tile)
        out_l1 = A.alloc_buffer((flush_tiles * m, n_per_tile), dtype="int8x64", scope=A.L1)
        dq_l1 = A.alloc_buffer((band_nblk * 6,), dtype=self.dequant_params.dtype, scope=A.L1)
        # Fresh per-core dq band (offset 0 in L1) — _fused_inner indexes it
        # band-relative via out_base = n_tile.base // 2.
        atomic.copy(
            slice_n(dq_flat_full, 0, 6 * t_lo * n_per_tile, 6 * t_hi * n_per_tile),
            scope=A.L1,
            output=dq_l1,
            generation=self.generation,
        )
        # Per-core L1 copy of the SiLU LUT, staged once (see _stage_lut_l1).
        lut_l1 = self._stage_lut_l1()
        silu_c = self._stage_silu_consts_l1()
        imc = A.alloc_imc((k_blocks, band_pair))
        pair_offset = t_lo * pair_per_tile
        if batch_dims:
            with A.grid(*batch_dims) as batch_vars_:
                batch_vars = A.to_tuple(batch_vars_)
                ba = tuple(broadcast_a) if broadcast_a is not None else batch_vars
                bb = tuple(broadcast_b) if broadcast_b is not None else batch_vars
                self._compute_n_outer(
                    ba,
                    bb,
                    batch_vars,
                    a_l1,
                    b_l1,
                    dq_l1,
                    lut_l1,
                    out_l1,
                    imc,
                    band_n_tiles,
                    num_m_tiles,
                    pair_per_tile,
                    n_per_tile,
                    pair_offset=pair_offset,
                    silu_c=silu_c,
                    tile_offset=t_lo,
                    core=core,
                    a_src=a_src,
                    pre_slot=pre_slot,
                )
        else:
            self._compute_n_outer(
                (),
                (),
                (),
                a_l1,
                b_l1,
                dq_l1,
                lut_l1,
                out_l1,
                imc,
                band_n_tiles,
                num_m_tiles,
                pair_per_tile,
                n_per_tile,
                pair_offset=pair_offset,
                silu_c=silu_c,
                tile_offset=t_lo,
                core=core,
                a_src=a_src,
                pre_slot=pre_slot,
            )
        if self.xop_slot is not None:
            # The next matmul's second shard runs through the barrier and that op's prologue;
            # it waits on it before its first IMC program. Launched here, after the batch
            # grid and the flush: a launch pairs with a wait only in the same scope, and the
            # wait sits at the core's scope in the next op.
            self._xop_launch(core, 1)

    def _xop_launch(self, core, order: int, where=None, slot: int = 0) -> None:
        """Shard ``order`` (launch order) of the next matmul's tile 0 into its slot, on
        channel 1 (channel 0 stays free for the flush and the next op's small copies). Odd
        cores read the upper (DDR1) shard first so both regions stream the whole time."""
        n_lo, rows = (int(x) for x in self.xop_rows)
        k_lo = int(self.xop_b.shape[-1])
        shards = [(self.xop_b, 0), (self.xop_b1, k_lo)]
        if core % 2:
            shards.reverse()
        src, k_off = shards[order]
        stream_shard(
            src,
            tir.Ramp(n_lo, 1, rows),
            caller_buffer(self.xop_slot),
            k_off,
            core=core,
            channel=_LOAD_B1_CHANNEL,
            id=xop_ids(core, slot)[order],
            where=where,
        )

    def _compute_n_outer(
        self,
        ba,
        bb,
        bo,
        a_l1,
        b_l1,
        dq_l1,
        lut_l1,
        out_l1,
        imc,
        num_n_tiles,
        num_m_tiles,
        pair_per_tile,
        n_per_tile,
        pair_offset: int = 0,
        tile_offset: int = 0,
        core=None,
        a_src=None,
        silu_c=None,
        pre_slot=None,
    ) -> None:
        """N outer, M inner.  Per N tile, load B and IMC once, then loop
        over M tiles.  Each M row's Datapath runs matmul + fused SwiGLU +
        quantize entirely in DPU registers, writing int8 to out_l1.  After
        the M loop the per-tile slab is flushed contiguously to the output's
        split slot.

        ``pair_offset`` / ``tile_offset`` / ``core`` support the multicore N-tile
        shard: b_l1 / imc / dq_l1 are band-local, so this loops the band's tiles
        (num_n_tiles = band tiles) but reads the FULL matrix_b at the band's pair
        offset and writes the FULL output at the band's tile offset. All three
        default to the single-core values (0 / 0 / self.core).

        ``a_src`` is the buffer the activation A is read from.  In the multicore
        path the caller passes a single full-A copy already staged in L2 (read by
        every core off the high-bandwidth, per-core L2 bus rather than each core
        re-reading A from the shared 24 GB/s DDR -- the redundant DDR reads would
        otherwise grow with the core count).  None (single core) reads A from DDR
        directly, staging through L2 only when M is tiled.  ``pre_slot`` is a caller-streamed
        tile 0 already in slot 0 (``prefetched``)."""
        _core = self.core if core is None else core
        m = a_l1.shape[0]
        k_blocks = a_l1.shape[1]
        # b_l1 is the whole band, or just two prefetch slots; either way the band's
        # extent is the tile count times the tile width.
        pair_blocks = num_n_tiles * pair_per_tile

        # When M fits in a single L1 tile (num_m_tiles is None after the
        # >1-or-None normalisation in implementation()), A is loop-invariant
        # across the N-tile loop: load it into L1 ONCE and reuse it for every
        # N tile.  The previous schedule staged A to L2 and re-copied it into
        # L1 once per N tile (num_n redundant reloads) -- pure overhead here,
        # and for a single token (decode) it also avoided an unnecessary L2 hop.
        # Only when M is genuinely tiled (num_m_tiles > 1) do we keep the L2
        # staging + per-(N,M)-tile reload (n_outer's reason to exist).
        single_m_tile = num_m_tiles is None

        # With a single M tile the band's output tiles are contiguous in the split
        # output, so they are buffered in out_l1 and flushed together after the loop.
        flush_tiles = self._band_flush_tiles(m, num_n_tiles, num_m_tiles, n_per_tile)
        band_l1 = (
            None
            if flush_tiles == 1
            else A.alloc_buffer(
                (flush_tiles, m, n_per_tile),
                dtype=out_l1.dtype,
                scope=A.L1,
                data=out_l1.data,
            )
        )

        weight_shards = self._weight_shards()

        def _load_b(n_idx, rows, slot, where=None):
            """Stage N tile ``n_idx`` into ``rows`` of pipeline slot ``slot`` of b_l1."""
            # Each shard is densely packed, so both DDR reads stay
            # contiguous and only the L1 write is strided.
            b_n_idx = (
                n_idx if not pair_offset else tir.Ramp(n_idx.base + pair_offset, 1, n_idx.lanes)
            )
            for src, k_offset, extent, channel, id_ in weight_shards:
                atomic.copy(
                    src[bb + (b_n_idx, ramp(extent))],
                    scope=A.L1,
                    output=b_l1[slot].tile(rows, dim=0).tile(tir.Ramp(k_offset, 1, extent), dim=1),
                    core=_core,
                    channel=channel,
                    generation=self.generation,
                    id=_sync_id(id_, _core, slot),
                    where=where,
                )

        def _load_imc(n_tile, rows, slot, where=None):
            """Program the IMC columns of ``n_tile`` from pipeline slot ``slot`` of b_l1."""
            atomic.load_imc(
                buffer=b_l1[slot].tile(rows, dim=0),
                imc=imc.tile(n_tile, dim=1),
                input_is_transposed=True,
                core=_core,
                generation=self.generation,
                id=_sync_id(_LOAD_IMC_ID, _core, slot),
                where=where,
            )

        def _wait_b(slot):
            """Retire the loads that filled pipeline slot ``slot``."""
            A.waits(*(_sync_id(id_, _core, slot) for _, _, _, _, id_ in weight_shards))

        def _flush_band():
            """One DMA for the band: its tiles are adjacent slots of the split output."""
            atomic.copy(
                band_l1,
                scope=A.DDR,
                output=self.output[
                    bo
                    + (
                        tir.Ramp(tile_offset, 1, flush_tiles),
                        ramp(m),
                        ramp(self.output.shape[-1]),
                    )
                ],
                generation=self.generation,
            )

        def _compute_tile(load_a_tile, n_tile, tile_local):
            """Run the fused datapath for ``n_tile`` and place its int8 output."""
            if band_l1 is not None:
                load_a_tile(n_tile, out_l1.tile(tir.Ramp(tile_local * m, 1, m), dim=0))
                return
            load_a_tile(n_tile, out_l1)
            atomic.copy(
                out_l1,
                scope=A.DDR,
                output=self.output[
                    bo + (tile_local + tile_offset, ramp(m), ramp(self.output.shape[-1]))
                ],
                generation=self.generation,
            )

        def _n_loop_pipelined(load_a_tile):
            """N-tile loop with tile i+1's weight DMAs in flight under tile i's compute."""
            slot_rows = A.Ramp(0, 1, pair_per_tile)
            # Tile 0 is peeled so the datapath is never predicated: gate_up's dequant walk lowers
            # to TWO IFD1 descriptors per tile and under a predicate one escapes and stalls.
            # A prefetched tile 0 landed in slot 0 already (waited in _emit_band).
            dma_first = prefetch_first(self.generation)
            xop = self.xop_slot is not None
            # Like _n_loop: a one-tile band leaves the peeled load and the loop below with
            # no user tiling scope, and the hardware-constraint tiling then re-emits them
            # after their own wait.
            outer = A.tiles(1) if num_n_tiles == 1 else contextlib.nullcontext()
            with outer:
                if pre_slot is None:
                    _load_b(_tile_at(0, pair_per_tile), slot_rows, 0)
                    if dma_first:
                        # One joint wait for both shards (see _n_pipeline).
                        _wait_b(0)
                with A.tiles(pair_blocks, num_tiles=num_n_tiles) as n_tile:
                    step, _, count = pipeline_dims(n_tile, pair_blocks)
                    # A.tiles clamps the last tile's extent; every tile here is full, and the
                    # slot buffers are sized for a full one, so rebuild the ramp from the index.
                    tile = _tile_at(step, pair_per_tile)
                    is_more = step + 1 < count
                    is_last = step + 1 == count
                    # NOTE: where on blocks, not A.when (aka if): TVM's scheduling infra does not
                    # support the latter. The loop is unrolled, so each arm's parity test folds away.
                    # The next tile's DMAs go out before this tile's program (prefetch_first) or
                    # after its wait: ReduceBufferLiveness would hoist the prefetch up to the IMC
                    # load reading its slot, and the sync between them is a reorder boundary.
                    for slot in range(2):
                        on_slot = step % 2 == slot

                        def _prefetch(slot=slot, on_slot=on_slot):
                            _load_b(
                                _tile_at(A.min(step + 1, count - 1), pair_per_tile),
                                slot_rows,
                                1 - slot,
                                where=tir.And(is_more, on_slot),
                            )

                        if dma_first:
                            _prefetch()
                        _load_imc(tile, slot_rows, slot, where=on_slot)
                        if xop:
                            # The last tile has no prefetch: the next op's first shard streams
                            # under its program and compute instead (waited below, in the same
                            # body: a launch pairs only with a wait in its own loop body).
                            self._xop_launch(_core, 0, where=tir.And(is_last, on_slot), slot=slot)
                        with A.when(on_slot):
                            A.waits(_sync_id(_LOAD_IMC_ID, _core, slot))
                        if not dma_first:
                            _prefetch()
                    _compute_tile(load_a_tile, tile, step)
                    # Waits after this tile's compute so both shard DMAs stay in flight across it;
                    # every launch in the body precedes every wait, so each slot pairs with its own.
                    for slot in range(2):
                        with A.when(tir.And(is_more, step % 2 == slot)):
                            _wait_b(1 - slot)
                    if xop:
                        # Per arm, under the launch's own predicate, like _wait_b above.
                        for slot in range(2):
                            with A.when(tir.And(is_last, step % 2 == slot)):
                                A.waits(xop_ids(_core, slot)[0])
                if band_l1 is not None:
                    _flush_band()

        def _n_loop(load_a_tile):
            """Body of the N-tile loop; ``load_a_tile`` stages A and runs the fused datapath."""
            # A one-tile band unrolls the loop below away; with no user tiling scope left,
            # the hardware-constraint tiling re-emits the weight load after its own wait.
            outer = A.tiles(1) if num_n_tiles == 1 else contextlib.nullcontext()
            with outer:
                with A.tiles(pair_blocks, num_tiles=num_n_tiles) as n_tile:
                    # b_l1 is a single band-wide slot here, indexed by the band-local tile.
                    _load_b(n_tile, n_tile, 0)
                    _wait_b(0)
                    _load_imc(n_tile, n_tile, 0)
                    _compute_tile(load_a_tile, n_tile, n_tile.base // pair_per_tile)
                if band_l1 is not None:
                    _flush_band()

        _run_n_loop = _n_loop_pipelined if self.pipeline_weights else _n_loop

        a_source = self._a_source(a_src, single_m_tile)

        if single_m_tile:
            # Load full A into L1 once, reuse every N tile (A is N-loop-invariant).
            atomic.copy(
                a_source[ba + (ramp(m), ramp(k_blocks))],
                scope=A.L1,
                output=a_l1,
                generation=self.generation,
            )

            def _load_a(n_tile, out_view):
                self._fused_inner(
                    a_tile=a_l1,
                    imc_tile=imc.tile(n_tile, dim=1),
                    dq_tile=dq_l1,
                    n_tile=n_tile,
                    out_tile=out_view,
                    core=_core,
                    luts=lut_l1,
                    silu_consts=silu_c,
                )

            _run_n_loop(_load_a)
        else:
            # M is tiled: reload the active M tile per (N, M) tile from a_source.
            def _load_a(n_tile, out_view):
                with A.tiles(m, num_tiles=num_m_tiles) as m_tile:
                    atomic.copy(
                        a_source[ba + (m_tile, ramp(k_blocks))],
                        scope=A.L1,
                        output=a_l1.tile(m_tile, dim=0),
                        generation=self.generation,
                    )
                    self._fused_inner(
                        a_tile=a_l1.tile(m_tile, dim=0),
                        imc_tile=imc.tile(n_tile, dim=1),
                        dq_tile=dq_l1,
                        n_tile=n_tile,
                        out_tile=out_view.tile(m_tile, dim=0),
                        core=_core,
                        luts=lut_l1,
                        silu_consts=silu_c,
                    )

            _run_n_loop(_load_a)

    def _fused_inner(
        self,
        a_tile,
        imc_tile,
        dq_tile,
        n_tile,
        out_tile,
        core=None,
        luts=None,
        silu_consts=None,
    ) -> None:
        """Per-(m_tile, n_tile) fused datapath: pair gate/up matmul outputs
        and run the SwiGLU + quantize chain in DPU registers.

        ``luts`` is the SiLU LUT the DPU reads; callers pass an L1-resident copy
        staged once before the N-tile loop (else the datapath references the DDR
        buffer and the stager re-loads it per tile).  Defaults to ``self.luts``.
        ``silu_consts`` is the staged [-log2e, 1.0] pair EUROPA's EXP2/RECIP SiLU reads
        instead (see _stage_silu_consts_l1)."""
        _core = self.core if core is None else core
        is_europa = self._europa_silu()
        if is_europa and silu_consts is None:
            raise ValueError("EUROPA SiLU needs the staged [-log2e, 1.0] constants.")
        m_tile_size = a_tile.shape[0]
        k_blocks = a_tile.shape[1]
        n_per_tile_pair = imc_tile.shape[1]  # in matmul-output units (2*output)
        method = self.method
        luts = self.luts if luts is None else luts
        n_luts = int(luts.shape[0])

        # out_tile is sized to one N tile (m_tile.lanes, out_per_tile),
        # reused across all N tiles; writes happen at column j without a
        # per-tile base offset.  The split-axis tile_idx is applied at the
        # DDR flush.
        out_per_tile = n_per_tile_pair // 2  # logical output channels per N tile

        # The IFD1 dq table slice for this N tile starts at 6 * out_base
        # in the combined dq buffer, where out_base is the logical column
        # offset of the current N tile (matmul-output base // 2).
        out_base = n_tile.base // 2

        with A.script():
            for m in A.serial(m_tile_size):
                with A.Datapath(kind=A.DP_MVMEXE, core=_core):
                    ifd0_stream = A.decl_stream("int8x64")
                    ifd1_stream = A.decl_stream(dq_tile.dtype)
                    matmul_stream = A.decl_stream("int32x64")
                    dpu_stream = A.decl_stream("int8x64")

                    # IFD0: stream the M row's a_tile values.
                    for n, r in A.grid(n_per_tile_pair, k_blocks):
                        with A.IFD(num=0):
                            ifd0_stream.push(a_tile[m, r])

                    # MatMul: one int32 accumulation per matmul-output column.
                    for n in A.serial(n_per_tile_pair):
                        with A.pseudo.MatMul():
                            acc = A.mvm_accumulator()
                            A.reset_accumulator(acc)
                            for r in A.serial(k_blocks):
                                acc += A.mvm_1x1(ifd0_stream.pop(), imc_tile[r, n])
                            matmul_stream.push(acc)

                    # IFD1: 6 floats per logical output (gate_dz, gate_ds,
                    # up_dz, up_ds, down_qs, down_qz), streamed contiguously.
                    # Tile slice starts at 6 * out_base in the combined dq buffer.
                    for c in A.serial(6 * out_per_tile):
                        with A.IFD(num=1):
                            ifd1_stream.push(dq_tile[6 * out_base + c])

                    with A.DPU():
                        regs = A.array("float18x64", 8)
                        tmp = A.var("float18x64")
                        acc_lut = A.var("float18x64")

                        # Cluster n_per_tile_pair matmul outputs into out_per_tile (gate, up) pairs.
                        for j in A.serial(out_per_tile):
                            # gate_int32, up_int32 → fp18.
                            regs[0] = A.cast(matmul_stream.pop(), "float18x64")  # gate
                            regs[1] = A.cast(matmul_stream.pop(), "float18x64")  # up
                            # dq pops: gate_dz, gate_ds, up_dz, up_ds.
                            regs[2] = A.cast(ifd1_stream.pop(), "float18x64")
                            regs[3] = A.cast(ifd1_stream.pop(), "float18x64")
                            regs[4] = A.cast(ifd1_stream.pop(), "float18x64")
                            regs[5] = A.cast(ifd1_stream.pop(), "float18x64")
                            # gate dequant: (gate - gate_dz) * gate_ds.
                            regs[0] = regs[0] - regs[2]
                            regs[0] = regs[0] * regs[3]
                            # SiLU(gate).
                            regs[0] = _silu_expr(
                                regs[0],
                                silu_consts,
                                luts,
                                n_luts,
                                method,
                                tmp.buffer,
                                acc_lut.buffer,
                                is_europa,
                            )
                            # up dequant: (up - up_dz) * up_ds.
                            regs[1] = regs[1] - regs[4]
                            regs[1] = regs[1] * regs[5]
                            # SwiGLU product.
                            regs[0] = regs[0] * regs[1]
                            # Final quantize: prod * down_qs + down_qz, cast int8.
                            regs[6] = A.cast(ifd1_stream.pop(), "float18x64")  # qs
                            regs[7] = A.cast(ifd1_stream.pop(), "float18x64")  # qz
                            regs[0] = regs[0] * regs[6]
                            regs[0] = regs[0] + regs[7]
                            dpu_stream.push(A.cast(regs[0], "int8x64"))

                    for j in A.serial(out_per_tile):
                        with A.ODR():
                            out_tile[m, j] = dpu_stream.pop()

    # ------------------------------------------------------------------ #
    # Emulation                                                            #
    # ------------------------------------------------------------------ #

    def emulate(self) -> None:  # type: ignore[override]
        # numpy buffer shapes (PWORD trailing dim from int8x64 unpacking):
        #   matrix_a:       (..., M, K_BLOCKS, PWORD)
        #   matrix_b (NK):  (..., 2*N_BLOCKS, K, PWORD)  -- composed gate||up
        #   dequant_params: (..., 6*N_BLOCKS, PWORD) -- combined dq + q
        #   output:         (..., num_n_tiles, M, N_per_tile_BLOCKS, PWORD)
        #     -- split layout from the n_outer schedule
        a_flat = self.matrix_a.reshape(
            self.matrix_a.shape[:-2] + (self.matrix_a.shape[-2] * self.matrix_a.shape[-1],)
        )
        ndim_b = self.matrix_b.ndim
        perm = list(range(ndim_b - 3)) + [ndim_b - 2, ndim_b - 3, ndim_b - 1]
        # The shards split K, so the reference matmul sees them rejoined.
        matrix_b = (
            self.matrix_b
            if self.matrix_b1 is None
            else np.concatenate((self.matrix_b, self.matrix_b1), axis=-2)
        )
        b_kn = np.transpose(matrix_b, perm)
        # (..., K, 2*N_BLOCKS, PWORD) -> (..., K, 2*N) where 2*N = 2*N_BLOCKS*PWORD
        b_flat = b_kn.reshape(b_kn.shape[:-2] + (b_kn.shape[-2] * b_kn.shape[-1],))

        # int32 matmul accumulation, cast to fp18.
        result = np.matmul(a_flat.astype(np.int32), b_flat.astype(np.int32))
        result_f = emu.rf18(result.astype(np.float32))  # (..., M, 2*N)

        # Split alternating gate (even) and up (odd) matmul outputs along the
        # final axis.  result_f's last dim is 2*N = 2*N_BLOCKS*PWORD,
        # interleaved PWORD-block-wise: positions [0..PWORD) are gate[0],
        # [PWORD..2*PWORD) are up[0], etc.
        pword = int(self.matrix_a.shape[-1])
        n_blocks = int(self.matrix_b.shape[-3]) // 2
        # Reshape (..., M, 2*N) -> (..., M, 2*N_BLOCKS, PWORD), then split.
        result_blocks = result_f.reshape(result_f.shape[:-1] + (2 * n_blocks, pword))
        gate_int = result_blocks[..., 0::2, :].reshape(result_f.shape[:-1] + (n_blocks * pword,))
        up_int = result_blocks[..., 1::2, :].reshape(result_f.shape[:-1] + (n_blocks * pword,))

        # Split the combined dq + q table: dequant_params shape
        # (..., 6*N_BLOCKS, PWORD) with [gate_dz, gate_ds, up_dz, up_ds,
        # down_qs, down_qz] interleaved per logical N.
        dq = self.dequant_params
        dq_flat = dq.reshape((6 * n_blocks, pword))
        gate_dz = dq_flat[0::6].reshape((n_blocks * pword,))
        gate_ds = dq_flat[1::6].reshape((n_blocks * pword,))
        up_dz = dq_flat[2::6].reshape((n_blocks * pword,))
        up_ds = dq_flat[3::6].reshape((n_blocks * pword,))
        down_qs = dq_flat[4::6].reshape((n_blocks * pword,))
        down_qz = dq_flat[5::6].reshape((n_blocks * pword,))

        # Broadcast 1D channel vectors against (..., M, N=N_BLOCKS*PWORD).
        gate_dz_b = gate_dz.reshape((1,) * (gate_int.ndim - 1) + (gate_dz.size,))
        gate_ds_b = gate_ds.reshape((1,) * (gate_int.ndim - 1) + (gate_ds.size,))
        up_dz_b = up_dz.reshape((1,) * (up_int.ndim - 1) + (up_dz.size,))
        up_ds_b = up_ds.reshape((1,) * (up_int.ndim - 1) + (up_ds.size,))
        qs_b = down_qs.reshape((1,) * (gate_int.ndim - 1) + (down_qs.size,))
        qz_b = down_qz.reshape((1,) * (gate_int.ndim - 1) + (down_qz.size,))

        # rf18-step the parameters to match the cast at IFD1 pop.
        gate_dz_f = emu.rf18(gate_dz_b.astype(np.float32))
        gate_ds_f = emu.rf18(gate_ds_b.astype(np.float32))
        up_dz_f = emu.rf18(up_dz_b.astype(np.float32))
        up_ds_f = emu.rf18(up_ds_b.astype(np.float32))
        qs_f = emu.rf18(qs_b.astype(np.float32))
        qz_f = emu.rf18(qz_b.astype(np.float32))

        # gate dequant + SiLU.
        gate_dq = emu.rf18(gate_int - gate_dz_f)
        gate_dq = emu.rf18(gate_dq * gate_ds_f)
        if self._europa_silu():
            # Mirror the EXP2/RECIP datapath statement for statement, rf18 at each
            # step: numpy's exact exp2/reciprocal stand in for the hardware ops.
            # Emulating the LUT here instead would stop modelling what runs.
            u = emu.rf18(gate_dq * emu.rf18(np.float32(-_LOG2E)))
            sig = emu.rf18(np.exp2(u))
            sig = emu.rf18(sig + np.float32(1.0))
            sig = emu.rf18(np.reciprocal(sig))
            gate_silu = emu.rf18(gate_dq * sig)
        else:
            # SiLU via the LUT chain (ADDITIVE or COMPOSITION mode).
            luts_np = self.luts.reshape(int(self.luts.shape[0]), -1)
            gate_silu = emu.apply_lut(gate_dq, luts_np[0:1])
            if self.method == ChainedLUTMethod.ADDITIVE:
                for k_lut in range(1, int(self.luts.shape[0])):
                    gate_silu = emu.rf18(
                        gate_silu + emu.apply_lut(gate_dq, luts_np[k_lut : k_lut + 1])
                    )
            elif self.method == ChainedLUTMethod.COMPOSITION:
                for k_lut in range(1, int(self.luts.shape[0])):
                    gate_silu = emu.apply_lut(gate_silu, luts_np[k_lut : k_lut + 1])

        # up dequant.
        up_dq = emu.rf18(up_int - up_dz_f)
        up_dq = emu.rf18(up_dq * up_ds_f)

        # SwiGLU product.
        prod = emu.rf18(gate_silu * up_dq)

        # Final quantize: prod * qs + qz, cast int8.
        scaled = emu.rf18(prod * qs_f)
        shifted = emu.rf18(scaled + qz_f)
        int8 = emu.round_clip_cast(shifted.astype(np.float32))

        # Reshape (..., M, N=N_BLOCKS*PWORD) -> split layout
        # (..., num_n_tiles, M, N_per_tile_BLOCKS, PWORD) to match the
        # n_outer schedule's contiguous-flush format.
        out_shape = self.output.shape
        num_n_tiles = int(out_shape[-4])
        n_per_tile_blocks = n_blocks // num_n_tiles
        n_per_tile_n = n_per_tile_blocks * pword
        # Split the last N axis into (num_n_tiles, n_per_tile_n), then
        # move the tile axis in front of M.
        int8 = int8.reshape(int8.shape[:-1] + (num_n_tiles, n_per_tile_n))
        int8 = np.moveaxis(int8, -2, -3)
        int8 = int8.reshape(out_shape)
        np.copyto(self.output, int8)


fused_gate_up_swiglu = register_op(
    FusedGateUpSwiGLU,
    "llm.fused_gate_up_swiglu",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)

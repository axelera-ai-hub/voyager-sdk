#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Custom LLM matmul op (registered as llm.matmul).

An alternative to axol.core.matmul with explicit L1 staging and analytic
KN/NK tiling schedules. Registered under the llm. namespace so it lives
alongside the core op rather than overriding it; the optimized Qwen3 kernel
injects it in place of axol.core.matmul.
"""

# mypy: disable-error-code="attr-defined, index, arg-type, type-var, assignment, call-overload"

from functools import reduce
import math
from operator import mul

from axelera.compiler.alto.axol import atomic
from axelera.compiler.alto.axol.common import ramp
from axelera.compiler.alto.compiler.config import HardwareGeneration, get_alto_config
from axelera.compiler.alto.compiler.device import IMC_SIZE, PWORD
from axelera.compiler.alto.compiler.dtypes import Int8x64, MetisTypes, get_dtype_size
from axelera.compiler.alto.compiler.interpreter import emulation as emu
from axelera.compiler.alto.compiler.op_registry import Arg, BufferInfo, ScriptOp, register_op
import axelera.compiler.alto.script as A
import numpy as np
from tvm import ir, tir


def _imc_max_blocks(generation: HardwareGeneration | None) -> int:
    """Largest K_BLOCKS * N_BLOCKS_per_tile an IMC operand may use (Omega 256, Europa 288)."""
    # infer_outputs() runs before the registry resolves the generation Arg, so it can still be None.
    return IMC_SIZE[generation or get_alto_config().generation]


# Tile count a weight-pipelining M=1 matmul aims for, and the point past which widening the
# tile is worth losing pipeline depth for. Below it the schedule keeps one block per tile, the
# narrow default: MEASURED on Qwen3.5 decode, widening gate_up from 12 tiles to 8 cost 2.4 ms
# across 24 layers (28.83 -> 31.24 ms), because at a small N the prefetch depth is worth more
# than the handful of IMC reloads it saves. Above it the tile count is pathological and the
# reverse holds: the lm_head's 485 blocks at one per tile ran 3.0452 ms and compiled in 1283 s,
# against 2.9878 ms and 277 s at 97 uniform tiles. A shape that cannot reach the target (N too
# small, or no divisor under the IMC ceiling) takes as many uniform tiles as it can.
PIPELINE_MIN_TILES = 32

# Conservative L1 workspace budget (excluding IMC region) for staging the
# A copy, B copy, and accumulated output.  Real workspace is larger (~2-3 MB
# depending on configuration), but staying under this avoids the
# ValidateMemory pre-allocation check.
L1_WORKSPACE_BUDGET = 4 * 1024 * 1024


def _sync_id(id_: str, core, slot: int) -> str:
    """Sync id per core and pipeline slot, so each parity arm pairs with its own wait."""
    return f"{id_}_{core}_{slot}"


def _tile_at(index, width) -> tir.Ramp:
    """Ramp of the band's N tile ``index``; every tile is uniform by construction."""
    return A.Ramp(index * width, 1, width)


def _try_int(x) -> int | None:
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _pow2_floor(n: int) -> int:
    """Largest power of 2 <= n; returns 1 if n < 1."""
    if n < 1:
        return 1
    return 1 << (n.bit_length() - 1)


def _m1_schedule(
    schedule: str,
    nb: int,
    kb: int,
    k_bytes: int,
    out_block_bytes: int,
    l1_budget: int,
    min_n_per_tile: int,
    max_n_per_tile: int,
    b_slots: int = 1,
    require_divisible: bool = False,
    min_n_tiles: int = 1,
):
    """The whole (schedule, num_n, num_m) at M = 1: the widest weight tile that fits, up to
    the IMC ceiling, so N splits into as few tiles as possible.

    A single M row loads the IMC once per N tile, so the reload count is exactly the tile
    count -- minimising it is the whole game at M=1, and the widest tile the L1 budget
    allows always does that. (A previous, narrower-first version picked the smallest
    DMA-efficient tile instead, which for a small K and a large N -- the lm_head's
    vocabulary projection -- left N split into hundreds of 1-block tiles: several times
    more IMC reloads than the IMC needed. Measured on Europa at N=2048/8192 (32/128
    blocks), kb=40: 3.1x/3.5x faster decode, on top of the compile-time win from fewer
    scheduled blocks.)

    require_divisible restricts the search to tiles that divide N evenly: fused_gate_up_swiglu
    needs every tile the same width, for its (gate, up) pair addressing, and rejects a ragged
    last tile outright -- other callers (e.g. quantized_matmul_activation) tolerate one fine, so
    forcing it on them too would cost real compile time (more, narrower tiles) for no reason.
    It also constrains the width to a MULTIPLE of min_n_per_tile, since a caller that needs
    its groups whole needs whole groups per tile, not merely two blocks' worth. min_n_per_tile
    is itself always a valid divisor (nb is a multiple of it by the caller's own contract --
    gate_up's pair_blocks is always even because min_n_per_tile=2), so the scan can fall back
    on it and still be valid.

    min_n_tiles floors the TILE COUNT, which is what a weight-pipelining caller needs: the
    prefetch double-buffers tile i+1's DMA under tile i's IMC load, so it needs at least two
    tiles, and uniform ones (a ragged last tile miscomputes once the stages overlap). The
    widest fit alone satisfies neither -- at Qwen3.5's decode shapes it collapses N to a
    single tile for qkv/out/down_proj and to a ragged 54 for the lm_head, and
    quantized_matmul_activation then drops the pipelining it was asked for WITHOUT
    complaining. Measured on the lm_head at vocab 248320 / 8 cores: 485 uniform tiles
    pipelined 3.0452 ms, 54 ragged unpipelined 3.3228 ms -- the widest schedule is 9% SLOWER
    there, because losing the DMA/compute overlap costs more than 431 saved IMC reloads gain.

    So the schedule takes the widest uniform tile that still leaves min_n_tiles of them, or
    all of them when N is smaller than that: the lm_head goes 485 -> 97 tiles (n_per_tile 5,
    the widest divisor of 485 under the IMC ceiling of 9), while every layer projection keeps
    one block per tile exactly as before, which is what they measured fastest at.

    A shape with no pipelinable option (N prime and above the IMC ceiling, or N == 1) falls
    back to min_n_per_tile -- the NARROWEST legal width, not the widest -- and runs
    unpipelined. That is the safe end to land on: min_n_per_tile always divides nb by the
    caller's own contract, so the fallback stays uniform for a require_divisible or
    min_n_tiles caller, where the widest fit would not.
    """
    if min_n_per_tile > min(max_n_per_tile, nb):
        # Clamping below the floor would hand the caller a tile it cannot use, and the
        # error would surface as a shape complaint from the op rather than from here.
        raise ValueError(
            f"K_BLOCKS={kb} leaves at most {min(max_n_per_tile, nb)} N blocks per IMC tile, "
            f"below the caller's floor of {min_n_per_tile}."
        )
    # Uniform tiles are required either by the caller's addressing (require_divisible) or by
    # a pipeline that needs every stage the same width.
    uniform = require_divisible or min_n_tiles > 1
    # N cannot be cut into more tiles than it has blocks, so a small N asks for all of them --
    # which is the narrow one-block-per-tile schedule, and what the layer projections want.
    target_tiles = min(nb, min_n_tiles)
    n_per_tile = None
    for candidate in range(min(max_n_per_tile, nb), min_n_per_tile - 1, -1):
        if uniform and nb % candidate:
            continue
        # A floor alone does not keep a caller's groups intact: gate_up's pairs are two
        # blocks, and a width of 3 divides 96 and clears a floor of 2 while still cutting
        # every pair across two tiles.
        if require_divisible and candidate % min_n_per_tile:
            continue
        if nb // candidate < target_tiles:
            continue
        live = b_slots * candidate * k_bytes * PWORD + k_bytes + nb * out_block_bytes
        if live <= l1_budget:
            n_per_tile = candidate
            break
    if n_per_tile is None:
        n_per_tile = min_n_per_tile
    live = b_slots * n_per_tile * k_bytes * PWORD + k_bytes + nb * out_block_bytes
    # The narrowest tile has the smallest b_l1, so a miss here means no tiling fits at all.
    if live > l1_budget:
        return schedule, None, None
    return schedule, math.ceil(nb / n_per_tile), 1


def _n_per_tile_candidates(max_value: int, min_value: int = 1) -> list[int]:
    """Pow-of-2 candidates for n_per_tile, descending from _pow2_floor(max_value) to min_value.

    The MVM mapping algorithm (mvm_calc_mapping in the AIPU backend)
    treats non-pow-of-2 column widths poorly: _cut_slice floors to
    the nearest pow-of-2 and leaves a remainder slice that maps to a
    differently-shaped MVM region than the bulk, producing a lopsided
    HPIDA reduction.  Restricting the schedule's n_per_tile to a
    power of 2 keeps each matmul tile on the algorithm's happy path:
    every full-sized tile maps uniformly, and only the residual tile
    (when nb itself is not a multiple of the chosen n_per_tile)
    can require a cut.
    """
    if max_value < 1:
        return [1]
    candidates: list[int] = []
    p = _pow2_floor(max_value)
    while p >= max(1, min_value):
        candidates.append(p)
        p //= 2
    return candidates


def _kn_num_k_tiles(k_blocks, n_blocks, generation: HardwareGeneration | None) -> int | None:
    """Number of K tiles required for the KN schedule to fit the IMC.

    Picks the smallest num_k such that ceil(K_BLOCKS/num_k) * N_BLOCKS fits the IMC.
    Returns 1 when the full (K, N) IMC operand fits in one shot.

    Returns None if either dim is not statically known (caller falls
    back to autotiling).  K tiling produces partial-sum accumulation;
    the caller uses this to gate K tiling on a real capacity need rather
    than letting the autotiler split K speculatively.
    """
    kb = _try_int(k_blocks)
    nb = _try_int(n_blocks)
    if kb is None or nb is None:
        return None
    if nb <= 0:
        return 1
    k_per_tile_max = max(1, _imc_max_blocks(generation) // nb)
    return max(1, math.ceil(kb / k_per_tile_max))


def _nk_schedule(
    m,
    k_blocks,
    n_blocks,
    generation: HardwareGeneration | None,
    force: str | None = None,
    out_dtype: str = "float32x64",
    extra_l1_bytes: int = 0,
    min_n_per_tile: int = 1,
    b_slots: int = 1,
    require_divisible: bool = False,
    min_n_tiles: int = 1,
):
    """Decide loop ordering and tiling for the NK matmul schedule.

    Both orderings flush output once per outer-loop iteration so the DMA
    is a single contiguous copy per iteration — Metis has no 2D DMA.

    - "m_outer": load A row once, then loop over N tiles per row.
      Output flushed per M tile, copying (M_tile, N).  Optimal when
      the full M fits in L1 with one M tile.  Falls back to tiling M when
      it doesn't fit, at the cost of (num_m - 1) * num_n extra IMC reloads.

    - "n_outer": load B/IMC once per N tile, then loop over M tiles.
      Output flushed per N tile, copying (M, N_per_tile).  Worth it
      whenever M would otherwise need to be tiled — saves the IMC reloads
      m_outer would incur.  N tile size is chosen as large as possible
      (= smallest num_n) subject to both the IMC capacity bound and
      the L1 budget for out_l1 + b_l1.

    force pins the schedule to "m_outer" or "n_outer" regardless of which
    the heuristic would prefer; tile sizing is still computed analytically.
    None (default) picks whichever fits.

    extra_l1_bytes reserves L1 for callee-owned allocations alongside the
    matmul staging buffers (e.g. a fused kernel that pre-stages quantization
    params in L1).  Subtracted from L1_WORKSPACE_BUDGET before tile fits are
    computed, so both schedules trade off against the reduced budget.

    Above M = 1 the N tile size is chosen from _n_per_tile_candidates (powers of 2
    only) to keep the downstream MVM mapping uniform. At M = 1 _m1_schedule scans
    every width instead: minimising IMC reloads (and keeping the tiles uniform for
    the weight prefetch) is worth more there than the mapping's pow-2 happy path,
    which is measured, not assumed -- see _m1_schedule.

    min_n_per_tile floors the N tile width, for callers whose layout needs more than
    one block per tile (fused_gate_up_swiglu keeps each (gate, up) pair in one tile).

    b_slots is how many B tiles the caller keeps live at once (2 when the N loop prefetches
    tile i+1); it scales the b_l1 term of every fit so the extra slot is charged for.

    min_n_tiles floors the M=1 tile count so a weight-pipelining caller gets tiles it can
    actually double-buffer; see _m1_schedule.

    require_divisible restricts the M=1 tile search to widths that divide N evenly (see
    _m1_schedule); only fused_gate_up_swiglu needs this, for its (gate, up) pair addressing.
    Ignored above M=1, where n_outer already shrinks from a power-of-2 ceiling and m_outer
    tiles M instead of N.

    Returns (schedule, num_n_tiles, num_m_tiles).  A tile count of None means
    "let the autotiler pick" — used for symbolic shapes or when the schedule
    cannot fit analytically.
    """
    if force not in (None, "m_outer", "n_outer"):
        raise ValueError(f"force must be None, 'm_outer', or 'n_outer'; got {force!r}")

    m_int = _try_int(m)
    kb = _try_int(k_blocks)
    nb = _try_int(n_blocks)
    if m_int is None or kb is None or nb is None:
        return (force or "m_outer"), None, None

    k_bytes = kb * PWORD
    # Bytes per PWORD-block of the output dtype (e.g. 256 for float32x64,
    # 128 for float16x64, 64 for int8x64).
    out_block_bytes = get_dtype_size(out_dtype)

    # Maximum N_per_tile allowed by IMC capacity.
    n_per_tile_imc_max = max(1, _imc_max_blocks(generation) // max(1, kb))

    # L1 budget after deducting caller-reserved bytes (e.g. fused kernels
    # that pre-stage quantization params in L1).
    if extra_l1_bytes < 0:
        raise ValueError(f"extra_l1_bytes must be >= 0, got {extra_l1_bytes}.")
    l1_budget = L1_WORKSPACE_BUDGET - extra_l1_bytes

    def m_outer_tiles():
        """Pick (num_n, num_m) for m_outer minimizing num_n * num_m.

        m_outer reloads IMC once per (M tile, N tile), so total IMC loads
        equal num_n * num_m.  Enumerate n_per_tile from largest (smallest
        num_n) downwards, recording the best total seen.  Decreasing
        n_per_tile shrinks b_l1 and frees L1 budget for a larger m_tile,
        so num_m monotonically decreases; once it hits 1, smaller n_per_tile
        only grows num_n, so we stop.

        Live working set per inner iteration: a_l1 (m_tile rows) + b_l1
        (n_per_tile rows) + out_l1 (m_tile rows of full N, flushed per M
        tile).  Lex tie-break: (total, num_m, num_n) — smaller num_m wins
        ties (fewer A loads and output flushes), then smaller num_n.

        Returns None if no n_per_tile fits even m_tile=1.
        """
        # a_l1 + out_l1 per M row, using the configured output dtype.
        cost_per_m = k_bytes + nb * out_block_bytes
        best = None  # (key, num_n, num_m), key = (total, num_m, num_n)
        n_per_tile_max = min(nb, n_per_tile_imc_max)
        for n_per_tile in _n_per_tile_candidates(n_per_tile_max, min_n_per_tile):
            b_l1 = b_slots * n_per_tile * k_bytes * PWORD
            avail = l1_budget - b_l1
            if avail < cost_per_m:
                continue
            m_tile_max = max(1, avail // cost_per_m)
            num_n = max(1, math.ceil(nb / n_per_tile))
            num_m = max(1, math.ceil(m_int / m_tile_max))
            total = num_n * num_m
            key = (total, num_m, num_n)
            if best is None or key < best[0]:
                best = (key, num_n, num_m)
            if num_m == 1:
                # Smaller n_per_tile would keep num_m=1 but grow num_n.
                break
        if best is None:
            return None
        return best[1], best[2]

    def n_outer_tiles():
        """Pick (num_n, num_m) for n_outer; shrinks N_per_tile until it fits.

        Only pow-of-2 n_per_tile values are considered — see _n_per_tile_candidates.
        """
        a_l1_min = k_bytes  # one M row of A
        for n_per_tile in _n_per_tile_candidates(min(nb, n_per_tile_imc_max), min_n_per_tile):
            b_l1 = b_slots * n_per_tile * k_bytes * PWORD
            out_l1 = m_int * n_per_tile * out_block_bytes
            if b_l1 + out_l1 + a_l1_min <= l1_budget:
                num_n = max(1, math.ceil(nb / n_per_tile))
                m_tile_max = max(1, (l1_budget - b_l1 - out_l1) // k_bytes)
                num_m = max(1, math.ceil(m_int / m_tile_max))
                return num_n, num_m
        return None

    # Only at M = 1: above it n_outer flushes output per N tile, so narrowing multiplies DMAs.
    if m_int == 1:
        return _m1_schedule(
            force or "m_outer",
            nb,
            kb,
            k_bytes,
            out_block_bytes,
            l1_budget,
            min_n_per_tile,
            n_per_tile_imc_max,
            b_slots,
            require_divisible,
            min_n_tiles,
        )

    if force == "m_outer":
        result = m_outer_tiles()
        return ("m_outer", *result) if result is not None else ("m_outer", None, None)

    if force == "n_outer":
        result = n_outer_tiles()
        return ("n_outer", *result) if result is not None else ("n_outer", None, None)

    # Default: prefer m_outer when the full M fits in one tile (no IMC
    # reload penalty), else n_outer (saves IMC reloads when M tiles).
    result = m_outer_tiles()
    if result is not None and result[1] == 1:
        return ("m_outer", *result)

    result = n_outer_tiles()
    if result is not None:
        return ("n_outer", *result)

    # Even N_per_tile=1 with M_tile=1 doesn't fit. Bail to autotiler.
    return "m_outer", None, None


def _resolve_batch(matrix_a: tir.Buffer, matrix_b: tir.Buffer):
    """Resolve broadcast-compatible batch dims for matrix_a @ matrix_b.

    Either operand's batch may broadcast against the other if its product is 1.
    Returns (batch_dims, broadcast_a, broadcast_b) where batch_dims is the
    resolved output batch shape and each broadcast_* is either None (no
    broadcasting) or a zero-tuple of indices for the singleton operand.
    """
    batch_a = tuple(matrix_a.shape[:-2])
    batch_b = tuple(matrix_b.shape[:-2])

    broadcast_a: tuple[int, ...] | None = None
    broadcast_b: tuple[int, ...] | None = None

    if batch_a != batch_b:
        if reduce(mul, batch_a, 1) == 1:
            broadcast_a = (0,) * len(batch_a)
        if reduce(mul, batch_b, 1) == 1:
            broadcast_b = (0,) * len(batch_b)
        if broadcast_a is None and broadcast_b is None:
            raise ValueError(
                f"Batch dimensions of matrix A {batch_a} are not broadcastable "
                f"with matrix B {batch_b}."
            )

    batch_dims = max(batch_a, batch_b)
    return batch_dims, broadcast_a, broadcast_b


class LLMMatMul(ScriptOp):
    """Batched matrix multiplication with explicit L1 staging.

    Computes matrix_a @ matrix_b over arbitrary leading batch dims (with
    optional broadcasting of one operand's batch).

    Two B layouts are supported, controlled by b_transposed:

    KN (default, b_transposed=False):

        matrix_a:  (*batch, M,                K_BLOCKS),  int8x64
        matrix_b:  (*batch, K_BLOCKS * PWORD, N_BLOCKS),  int8x64

        Schedule tiles M and K (the reduction dim).  Partial matmul results
        accumulate into a float32 L1 buffer.  Handles arbitrarily large K
        via reduction tiling.

    NK (b_transposed=True):

        matrix_a:  (*batch, M,                  K_BLOCKS),       int8x64
        matrix_b:  (*batch, N_BLOCKS, K_BLOCKS * PWORD),         int8x64

        Schedule tiles M and N.  The full reduction dim of B is loaded into
        IMC for each N tile (using load_imc(input_is_transposed=True)),
        so each atomic.matmul writes its final value directly — no
        partial-sum accumulator, no float32 staging.  Output dtype is
        produced inline by the matmul.  Requires K <= IMC capacity
        (e.g. 16384 elements on OMEGA).
    """

    matrix_a = Arg(
        Int8x64[tir.Buffer, "..."],
        description="Matrix A, layout (*batch, M, K_BLOCKS), int8x64.",
    )
    matrix_b = Arg(
        Int8x64[tir.Buffer, "..."],
        description=(
            "Matrix B. KN layout (default): (*batch, K_BLOCKS*PWORD, N_BLOCKS). "
            "NK layout (b_transposed=True): (*batch, N_BLOCKS, K_BLOCKS*PWORD). "
            "Always int8x64 (PWORD packing on the N axis)."
        ),
    )

    output = Arg(
        MetisTypes[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="""
        Output buffer specification. Can be:
            - BufferInfo to specify shape/dtype/scope constraints
            - tir.Buffer for pre-allocated output
            - None to auto-infer (shape=(*batch, M, N_BLOCKS),
              dtype=float32x64, scope=matrix_a.scope())""",
    )
    b_transposed = Arg(
        bool,
        default=False,
        kw=True,
        description=(
            "If True, matrix_b is in (*batch, N_BLOCKS, K) layout (blockwise "
            "transposed on the outer two dims; PWORD packing on N is unchanged). "
            "Enables a tile-N schedule that avoids partial-sum accumulation."
        ),
    )
    nk_schedule = Arg(
        str | None,
        default=None,
        kw=True,
        description=(
            "Override the NK schedule choice (m_outer or n_outer). "
            "None lets the heuristic pick. Only applies when b_transposed=True."
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

    # ------------------------------------------------------------------ #
    # Layout helpers                                                       #
    # ------------------------------------------------------------------ #

    def _b_k(self) -> tir.PrimExpr:
        """Reduction dim K of matrix_b (in elements, not blocks)."""
        return self.matrix_b.shape[-1] if self.b_transposed else self.matrix_b.shape[-2]

    def _b_n_blocks(self) -> tir.PrimExpr:
        """Number of N PWORD-blocks in matrix_b."""
        return self.matrix_b.shape[-2] if self.b_transposed else self.matrix_b.shape[-1]

    # ------------------------------------------------------------------ #

    def validate(self) -> None:
        if len(self.matrix_a.shape) < 2:
            raise ValueError(
                f"matrix_a must have at least 2 dimensions, got {len(self.matrix_a.shape)}."
            )
        if len(self.matrix_b.shape) < 2:
            raise ValueError(
                f"matrix_b must have at least 2 dimensions, got {len(self.matrix_b.shape)}."
            )
        # Validates broadcast compatibility (raises if invalid).
        _resolve_batch(self.matrix_a, self.matrix_b)

        if not ir.structural_equal(self.matrix_a.shape[-1] * PWORD, self._b_k()):
            raise ValueError(
                f"Inner dimensions of matrix A ({self.matrix_a.shape[-1] * PWORD}) "
                f"and matrix B ({self._b_k()}) do not match."
            )

        if self.matrix_a.dtype != "int8x64":
            raise ValueError(
                f"matrix_a dtype {self.matrix_a.dtype} is not supported, must be int8x64."
            )
        if self.matrix_b.dtype != "int8x64":
            raise ValueError(
                f"matrix_b dtype {self.matrix_b.dtype} is not supported, must be int8x64."
            )

        if isinstance(self.output, tir.Buffer):
            if self.output.dtype not in ("int8x64", "float16x64", "float32x64"):
                raise ValueError(
                    f"output dtype {self.output.dtype} is not supported, "
                    f"must be int8x64, float16x64 or float32x64."
                )

        if self.nk_schedule is not None and self.nk_schedule not in ("m_outer", "n_outer"):
            raise ValueError(
                f"nk_schedule must be None, 'm_outer', or 'n_outer'; got {self.nk_schedule!r}"
            )

    def infer_outputs(self) -> dict[str, BufferInfo]:
        batch_dims, _, _ = _resolve_batch(self.matrix_a, self.matrix_b)
        m = self.matrix_a.shape[-2]
        n_blocks = self._b_n_blocks()
        return {
            "output": BufferInfo(
                shape=tuple(batch_dims) + (m, n_blocks),
                dtype="float32x64",
                scope=self.matrix_a.scope(),
            )
        }

    def implementation(self) -> None:
        if self.b_transposed:
            self._impl_nk()
        else:
            self._impl_kn()

    # ------------------------------------------------------------------ #
    # KN schedule                                                          #
    # ------------------------------------------------------------------ #

    def _impl_kn(self) -> None:
        # m_only requires the full (K_BLOCKS, N_BLOCKS) IMC operand to fit
        # in one shot.  When it does, prefer it — a single matmul per m_tile
        # writes the final value directly in the output dtype, no float32
        # accumulator, no K tiling.  Otherwise k_tiling splits K and
        # accumulates partial sums in a float32 L1 buffer.
        k_blocks = self.matrix_a.shape[-1]
        n_blocks = self.matrix_b.shape[-1]
        kb = _try_int(k_blocks)
        nb = _try_int(n_blocks)
        use_m_only = (
            kb is not None and nb is not None and kb * nb <= _imc_max_blocks(self.generation)
        )

        if use_m_only:
            self._impl_kn_m_only()
        else:
            self._impl_kn_k_tiling()

    def _impl_kn_k_tiling(self) -> None:
        """KN schedule with K tiling and a float32 partial-sum accumulator."""
        batch_dims, broadcast_a, broadcast_b = _resolve_batch(self.matrix_a, self.matrix_b)

        m = self.matrix_a.shape[-2]
        k_blocks = self.matrix_a.shape[-1]
        k = self.matrix_b.shape[-2]
        n_blocks = self.matrix_b.shape[-1]

        a_l1 = A.alloc_buffer((m, k_blocks), dtype=self.matrix_a.dtype, scope=A.L1)
        b_l1 = A.alloc_buffer((k, n_blocks), dtype=self.matrix_b.dtype, scope=A.L1)
        accumulator = A.alloc_buffer((m, n_blocks), dtype="float32x64", scope=A.L1)
        imc = A.alloc_imc((k_blocks, n_blocks))

        cast_l1 = None
        if self.output.dtype != "float32x64":
            cast_l1 = A.alloc_buffer((m, n_blocks), dtype=self.output.dtype, scope=A.L1)

        if batch_dims:
            with A.grid(*batch_dims) as batch_vars_:
                batch_vars = A.to_tuple(batch_vars_)
                ba = tuple(broadcast_a) if broadcast_a is not None else batch_vars
                bb = tuple(broadcast_b) if broadcast_b is not None else batch_vars
                self._kn_compute(ba, bb, batch_vars, a_l1, b_l1, accumulator, imc, cast_l1)
        else:
            self._kn_compute((), (), (), a_l1, b_l1, accumulator, imc, cast_l1)

    def _impl_kn_m_only(self) -> None:
        """KN with M-only tiling: load full (K, N) into IMC once, then one
        matmul per m_tile writing directly to the output dtype.  No K
        tiling, no partial-sum accumulator."""
        batch_dims, broadcast_a, broadcast_b = _resolve_batch(self.matrix_a, self.matrix_b)

        m = self.matrix_a.shape[-2]
        k_blocks = self.matrix_a.shape[-1]
        k = self.matrix_b.shape[-2]
        n_blocks = self.matrix_b.shape[-1]

        a_l1 = A.alloc_buffer((m, k_blocks), dtype=self.matrix_a.dtype, scope=A.L1)
        b_l1 = A.alloc_buffer((k, n_blocks), dtype=self.matrix_b.dtype, scope=A.L1)
        out_l1 = A.alloc_buffer((m, n_blocks), dtype=self.output.dtype, scope=A.L1)
        imc = A.alloc_imc((k_blocks, n_blocks))

        if batch_dims:
            with A.grid(*batch_dims) as batch_vars_:
                batch_vars = A.to_tuple(batch_vars_)
                ba = tuple(broadcast_a) if broadcast_a is not None else batch_vars
                bb = tuple(broadcast_b) if broadcast_b is not None else batch_vars
                self._kn_compute_m_only(ba, bb, batch_vars, a_l1, b_l1, out_l1, imc)
        else:
            self._kn_compute_m_only((), (), (), a_l1, b_l1, out_l1, imc)

    def _kn_compute_m_only(self, ba, bb, bo, a_l1, b_l1, out_l1, imc) -> None:
        """M-outer KN with no K reduction tiling.  atomic.matmul writes its
        final value directly into out_l1 in the configured output dtype."""
        m = a_l1.shape[0]
        k_blocks = a_l1.shape[1]
        k = b_l1.shape[0]
        n_blocks = b_l1.shape[1]

        # Stage full B once outside the m_tile loop and push to IMC.
        atomic.copy(
            self.matrix_b[bb + (ramp(k), ramp(n_blocks))],
            scope=A.L1,
            output=b_l1,
            generation=self.generation,
        )
        atomic.load_imc(
            buffer=b_l1,
            imc=imc,
            core=self.core,
            generation=self.generation,
        )

        with A.tiles(m) as m_tile:
            atomic.copy(
                self.matrix_a[ba + (m_tile, ramp(k_blocks))],
                scope=A.L1,
                output=a_l1.tile(m_tile, dim=0),
                generation=self.generation,
            )
            atomic.matmul(
                matrix_ifd=a_l1.tile(m_tile, dim=0),
                matrix_imc=imc,
                output=out_l1.tile(m_tile, dim=0),
                core=self.core,
                generation=self.generation,
            )
            atomic.copy(
                out_l1.tile(m_tile, dim=0),
                scope=A.DDR,
                output=self.output[bo + (m_tile, ramp(n_blocks))],
                generation=self.generation,
            )

    def _kn_compute(self, ba, bb, bo, a_l1, b_l1, accumulator, imc, cast_l1) -> None:
        m = a_l1.shape[0]
        k_blocks = a_l1.shape[1]
        n_blocks = b_l1.shape[1]

        # Only tile K when IMC capacity forces it.  K tiling produces
        # partial-sum accumulation; for the common case where K*N fits in
        # the IMC we prefer a single K iteration.  Forcing num_tiles=1
        # emits a degenerate T.unroll the autotiler can't simplify, so
        # pass None when the IMC fits and the analytic minimum only when >1.
        num_k_tiles = _kn_num_k_tiles(k_blocks, n_blocks, self.generation)
        num_k_tiles = num_k_tiles if (num_k_tiles or 0) > 1 else None

        with A.tiles(m) as m_tile:
            atomic.full(
                (m_tile.lanes, n_blocks),
                0.0,
                scope=A.L1,
                output=accumulator.tile(m_tile, dim=0),
                core=self.core,
                generation=self.generation,
            )
            atomic.copy(
                self.matrix_a[ba + (m_tile, ramp(k_blocks))],
                scope=A.L1,
                output=a_l1.tile(m_tile, dim=0),
                generation=self.generation,
            )
            with A.tiles(k_blocks, num_tiles=num_k_tiles) as k_tile:
                k_ramp = tir.Ramp(k_tile.base * PWORD, 1, k_tile.lanes * PWORD)
                atomic.copy(
                    self.matrix_b[bb + (k_ramp, ramp(n_blocks))],
                    scope=A.L1,
                    output=b_l1.tile(k_ramp, dim=0),
                    generation=self.generation,
                )
                atomic.load_imc(
                    buffer=b_l1.tile(k_ramp, dim=0),
                    imc=imc.tile(k_tile, dim=0),
                    core=self.core,
                    generation=self.generation,
                )
                atomic.matmul(
                    matrix_ifd=a_l1.tile(m_tile, dim=0).tile(k_tile, dim=1),
                    matrix_imc=imc.tile(k_tile, dim=0),
                    partial=accumulator.tile(m_tile, dim=0),
                    output=accumulator.tile(m_tile, dim=0),
                    core=self.core,
                    generation=self.generation,
                )

            if cast_l1 is None:
                atomic.copy(
                    accumulator.tile(m_tile, dim=0),
                    scope=A.DDR,
                    output=self.output[bo + (m_tile, ramp(n_blocks))],
                    generation=self.generation,
                )
            else:
                atomic.dpu_cast(
                    accumulator.tile(m_tile, dim=0),
                    output=cast_l1.tile(m_tile, dim=0),
                    generation=self.generation,
                )
                atomic.copy(
                    cast_l1.tile(m_tile, dim=0),
                    scope=A.DDR,
                    output=self.output[bo + (m_tile, ramp(n_blocks))],
                    generation=self.generation,
                )

    # ------------------------------------------------------------------ #
    # NK schedule: tile M and N, full K in IMC, no partial sums.         #
    # ------------------------------------------------------------------ #

    def _impl_nk(self) -> None:
        batch_dims, broadcast_a, broadcast_b = _resolve_batch(self.matrix_a, self.matrix_b)

        m = self.matrix_a.shape[-2]
        k_blocks = self.matrix_a.shape[-1]
        n_blocks = self.matrix_b.shape[-2]
        k = self.matrix_b.shape[-1]

        schedule, num_n_tiles, num_m_tiles = _nk_schedule(
            m,
            k_blocks,
            n_blocks,
            self.generation,
            force=self.nk_schedule,
            out_dtype=self.output.dtype,
        )
        # Only pass a forced tile count when it's >1.  Forcing num_tiles=1
        # emits a degenerate T.unroll loop that InjectTiling re-tiles via
        # SizeVars, which AssignTilingFactors cannot simplify.
        num_n_tiles = num_n_tiles if (num_n_tiles or 0) > 1 else None
        num_m_tiles = num_m_tiles if (num_m_tiles or 0) > 1 else None

        a_l1 = A.alloc_buffer((m, k_blocks), dtype=self.matrix_a.dtype, scope=A.L1)
        b_l1 = A.alloc_buffer((n_blocks, k), dtype=self.matrix_b.dtype, scope=A.L1)
        # Output staging in the final dtype; atomic.matmul casts inline.
        out_l1 = A.alloc_buffer((m, n_blocks), dtype=self.output.dtype, scope=A.L1)
        imc = A.alloc_imc((k_blocks, n_blocks))

        compute = self._nk_compute_n_outer if schedule == "n_outer" else self._nk_compute_m_outer

        if batch_dims:
            with A.grid(*batch_dims) as batch_vars_:
                batch_vars = A.to_tuple(batch_vars_)
                ba = tuple(broadcast_a) if broadcast_a is not None else batch_vars
                bb = tuple(broadcast_b) if broadcast_b is not None else batch_vars
                compute(ba, bb, batch_vars, a_l1, b_l1, out_l1, imc, num_n_tiles, num_m_tiles)
        else:
            compute((), (), (), a_l1, b_l1, out_l1, imc, num_n_tiles, num_m_tiles)

    def _nk_compute_m_outer(
        self, ba, bb, bo, a_l1, b_l1, out_l1, imc, num_n_tiles, num_m_tiles
    ) -> None:
        """M outer, N inner.  A loaded once per M tile; B/IMC reloaded per N tile.
        Use when full M fits in L1 with the m_outer out_l1 footprint."""
        m = a_l1.shape[0]
        k_blocks = a_l1.shape[1]
        n_blocks = b_l1.shape[0]
        k = b_l1.shape[1]
        # Stage B in L2 only when M is actually tiled (num_m_tiles > 1): the
        # L2 copy amortizes the per-N-tile weight read across M tiles.  With
        # a single M tile (e.g. decode M=1) the L2 hop is pure overhead, so
        # read B straight from DDR.
        if num_m_tiles is not None and num_m_tiles > 1:
            b_src = atomic.copy(self.matrix_b, scope=A.L2)
        else:
            b_src = self.matrix_b

        with A.tiles(m, num_tiles=num_m_tiles) as m_tile:
            atomic.copy(
                self.matrix_a[ba + (m_tile, ramp(k_blocks))],
                scope=A.L1,
                output=a_l1.tile(m_tile, dim=0),
                generation=self.generation,
            )
            with A.tiles(n_blocks, num_tiles=num_n_tiles) as n_tile:
                atomic.copy(
                    b_src[bb + (n_tile, ramp(k))],
                    scope=A.L1,
                    output=b_l1.tile(n_tile, dim=0),
                    generation=self.generation,
                )
                atomic.load_imc(
                    buffer=b_l1.tile(n_tile, dim=0),
                    imc=imc.tile(n_tile, dim=1),
                    input_is_transposed=True,
                    core=self.core,
                    generation=self.generation,
                )
                atomic.matmul(
                    matrix_ifd=a_l1.tile(m_tile, dim=0),
                    matrix_imc=imc.tile(n_tile, dim=1),
                    output=out_l1.tile(m_tile, dim=0).tile(n_tile, dim=1),
                    core=self.core,
                    generation=self.generation,
                )

            atomic.copy(
                out_l1.tile(m_tile, dim=0),
                scope=A.DDR,
                output=self.output[bo + (m_tile, ramp(n_blocks))],
                generation=self.generation,
            )

    def _nk_compute_n_outer(
        self, ba, bb, bo, a_l1, b_l1, out_l1, imc, num_n_tiles, num_m_tiles
    ) -> None:
        """N outer, M inner.  B/IMC loaded once per N tile and reused across
        all M tiles; A reloaded per (n_tile, m_tile).  The output flush is
        done once per N tile (outside the M loop) so the DMA covers the full
        (M, N_per_tile) slice in one contiguous copy — Metis has no 2D DMA.

        Trade-off: out_l1 must hold (M, N_per_tile) instead of
        (M_tile, N_per_tile).  _nk_schedule accounts for this when choosing
        num_m_tiles.
        """
        m = a_l1.shape[0]
        k_blocks = a_l1.shape[1]
        n_blocks = b_l1.shape[0]
        k = b_l1.shape[1]

        with A.tiles(n_blocks, num_tiles=num_n_tiles) as n_tile:
            atomic.copy(
                self.matrix_b[bb + (n_tile, ramp(k))],
                scope=A.L1,
                output=b_l1.tile(n_tile, dim=0),
                generation=self.generation,
            )
            atomic.load_imc(
                buffer=b_l1.tile(n_tile, dim=0),
                imc=imc.tile(n_tile, dim=1),
                input_is_transposed=True,
                core=self.core,
                generation=self.generation,
            )
            with A.tiles(m, num_tiles=num_m_tiles) as m_tile:
                atomic.copy(
                    self.matrix_a[ba + (m_tile, ramp(k_blocks))],
                    scope=A.L1,
                    output=a_l1.tile(m_tile, dim=0),
                    generation=self.generation,
                )
                atomic.matmul(
                    matrix_ifd=a_l1.tile(m_tile, dim=0),
                    matrix_imc=imc.tile(n_tile, dim=1),
                    output=out_l1.tile(m_tile, dim=0).tile(n_tile, dim=1),
                    core=self.core,
                    generation=self.generation,
                )

            # One larger contiguous copy per N tile (outside the M loop).
            atomic.copy(
                out_l1.tile(n_tile, dim=1),
                scope=A.DDR,
                output=self.output[bo + (ramp(m), n_tile)],
                generation=self.generation,
            )

    # ------------------------------------------------------------------ #

    def emulate(self) -> None:  # type: ignore[override]
        # numpy int8x64 buffers expose the PWORD packing as a trailing dim of
        # length 64. For KN: matrix_b is (..., K, N_BLOCKS, 64) -> flatten
        # last two dims gives (..., K, N).  For NK: matrix_b is
        # (..., N_BLOCKS, K, 64); transpose+flatten so np.matmul sums over K.
        a_flat = self.matrix_a.reshape(self.matrix_a.shape[:-2] + (self.matrix_a.shape[-2], -1))
        if self.b_transposed:
            # NK: (..., N_BLOCKS, K, 64) -> (..., K, N_BLOCKS, 64) -> (..., K, N)
            ndim = self.matrix_b.ndim
            perm = list(range(ndim - 3)) + [ndim - 2, ndim - 3, ndim - 1]
            b_kn = np.transpose(self.matrix_b, perm)
            b_flat = b_kn.reshape(b_kn.shape[:-2] + (b_kn.shape[-2], -1))
        else:
            b_flat = self.matrix_b.reshape(
                self.matrix_b.shape[:-2] + (self.matrix_b.shape[-2], -1)
            )
        result = np.matmul(a_flat.astype(np.int32), b_flat.astype(np.int32))
        result = result.reshape(self.output.shape)
        if self.output.dtype == np.int8:
            np.copyto(self.output, emu.round_clip_cast(result.astype(np.float32)))
        else:
            np.copyto(self.output, result.astype(self.output.dtype))


matmul = register_op(
    LLMMatMul,
    "llm.matmul",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)

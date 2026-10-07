#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Flash scaled dot-product attention (llm.sdpa_flash).

A drop-in alternative to axol.core.scaled_dot_product_attention that tiles BOTH
the **context / KV dimension** and (optionally) the **query / seq dimension**, so
L1 + IMC footprint becomes independent of context length and of seq length. This
is the structural fix for long-context prefill, where the core op (which holds the
FULL K, V and the (seq, context) score row resident per head) overflows L1 / IMC.

Schedule (seq-outer, KV-mid, seq-row-inner)
-------------------------------------------
For each head we loop over query/seq tiles of ``seq_tile`` rows (``seq_tile=0`` ->
the whole seq, no tiling), and within each, over KV tiles of ``kv_tile_blocks``
PWORD-blocks. Per KV tile we DMA only that tile's K, V and mask slice (all
O(kv_tile)) and run an **online softmax**, carrying per-query-row running state in
L1 -- sized to the seq tile, so resident state is O(seq_tile * kv_tile):

    m  running max       (seq_tile, 1) float32
    l  running denom      (seq_tile, 1) float32
    O  output accumulator (seq_tile, dv_blocks) float32

Query blocks are independent (no cross-block state); the cost of seq tiling is
re-streaming K/V per seq block. Tiling either axis trades L1 footprint against
re-loads / datapath count -- e.g. a bigger seq_tile lets kv_tile shrink and vice
versa.

Per KV tile ``j`` (online softmax, cf. Dao et al. FlashAttention) -- 3 datapaths:

    S_j      = Q @ K_jᵀ            (dequant, ·scale, + mask tile)        [1 qk_softmax]
    m_new    = max(m_old, rowmax S_j)
    alpha    = exp(m_old - m_new)         # neg-exp LUT, arg = m_new - m_old >= 0
    P_j      = exp(S_j - m_new)           # neg-exp LUT, arg = m_new - S_j   >= 0
    l_new    = alpha·l_old + rowsum P_j
    O_new    = alpha·O_old + (quant(P_j)·V_j)·output_dq                  [2 pv_accumulate]
and after the loop ``O_out = O · (1/l)``                                 [3 normalize]

The KV loop is a single uniform ``A.serial`` runtime loop (constant program
source; the compiler unrolls it). There is NO first-tile special case in the
datapath: the carry is initialised to ``m=-BIG, l=0, O=0`` (``atomic.full`` before
the loop), so the general update handles tile 0 for free -- ``m_new = max(-BIG,
rowmax) = rowmax``; ``alpha = exp(m_old-m_new) ~ 0`` multiplies only the zero
O_old/l_old, so ``l_new = s`` and ``O = contrib``. This is the standard flash-
attention initialisation, and it matters here beyond elegance: an earlier version
branched on the (unrolled) tile index with ``A.Select``/``A.If``, but the compiler
folds that branch to a constant per unrolled copy and there is no intra-DPU scalar
DCE before register allocation, so the folded-away branch's producers became dead
DPU instructions the allocator could not place (compile failed at kv>=4). A branch-
free body has no dead code to begin with.

Keeping the body uniform (one K/V/mask load per iteration) is also what keeps it
correct against ``InjectDMAs``: an earlier peel-first/peel-last version was wrong on
device because ``InjectDMAs`` (which runs on the rolled loop, before unrolling)
hoisted the peeled tile's key load into the shared staging buffer ahead of the
middle tiles, clobbering it. With no peeled block there is nothing to hoist.

The per-row running state (m, l, alpha) is carried in **L1** (cf. the core op's
scores_with_stats), NOT in DPU register banks, so there is no seq-tile cap. That
costs extra IFD inputs, and the IFD has only two ports -- each backed by a SINGLE
physical hardware stream whose address generator runs ONE pattern (a contiguous/
constant-stride run). Two separate IR streams on a port collide on that one channel
(it compiles but silently mis-addresses every row but the first); so does a stream
that advances for one push then reads a fixed offset for the next. But two logical
inputs CAN share a port if they are laid out contiguously in ONE buffer and read as
a single run. That is how the two extra port-1 inputs are folded back in so the
algorithm needs only 3 datapaths/tile, not 5:
  * QK reads ``qk_port1[m, :] = [mask_0..mask_{kvtb-1}, m_old, l_old]`` (one f32 run;
    the int8 mask is cast to f32 at staging) on port 1, query MVM on port 0.
  * P@V runs ONE datapath per query row (re-streaming P per output block, as the QK
    matmul re-streams the query) and reads ``out_state[m, 0:dv+1] = [O_old.., alpha]``
    (alpha in the last column) as one contiguous port-1 run.

``max()`` keeps both neg-exp LUT arguments >= 0 (the LUT's valid domain).

K/V/mask are addressed as a leading index into reshaped views (context dim split
as ``(n_tiles, kv_tile_ctx)``) so a runtime-``t`` slice ``view[..., t, ...]``
lowers cleanly; this requires context_len and the mask to be exact multiples of
the tile (else NotImplementedError -- a partial-last-tile path is future work).

Numerics
--------
The online rescale (alpha multiplies + per-tile P re-quantization) adds float18
rounding the one-shot softmax does not have, so -- unlike sdpa_decode -- emulate
does NOT delegate to the core reference; it models the online algorithm in
float18 with the same LUTs / reductions / accumulation order as the datapaths.
The online recombination tracks the one-shot schedule to ~1e-2 relative for
ctx >= 512 (test_sdpa_flash.py validates emulation vs the fakequant reference and
AICORE_SIM vs emulation).
"""
import logging

from axelera.compiler.alto.axol.common import sliced
from axelera.compiler.alto.axol.core.matmul._common import shards
from axelera.compiler.alto.axol.core.scaled_dot_product_attention import (
    _check_python_args,
    atomic,
    core,
)
from axelera.compiler.alto.axol.utils.lut_chain import ChainedLUTMethod, chained_lut_dpu
from axelera.compiler.alto.compiler import emulation as emu
from axelera.compiler.alto.compiler.config import HardwareGeneration
from axelera.compiler.alto.compiler.dtypes import MetisTypes
from axelera.compiler.alto.compiler.op_registry import Arg, BufferInfo, ScriptOp, register_op
import axelera.compiler.alto.script as A
import axelera.compiler.backend.tir.ops as tir_ops
from axelera.compiler.config import get_compiler_config
import numpy as np
from tvm import tir
from tvm.tir import BufferLoad, BufferSlice, Ramp
from tvm.topi.axelera.softmax import _reduce_add_f18, _reduce_adder_tree

# mypy: disable-error-code="attr-defined, index, arg-type"
# flake8: noqa: C901

PWORD = get_compiler_config().pword_len
_logger = logging.getLogger(__name__)

_F18 = "float18x64"
_F32 = "float32x64"
# Per-row online-softmax state (P/m/l/alpha + the fused mask|m|l run) is stored in L1 as
# float16, not float32, to halve the two dominant L1 buffers (scores_stats, qk_port1). f16
# has the same 10-bit mantissa as the fp18 the DPU computes in (so P, m, alpha are exact and
# the int8 P re-quantization is unaffected); l (the denom, up to ctx) keeps ~3-4 sig figs,
# well inside the op's ~2% tolerance. The O accumulator stays f32 (accumulates across tiles).
_F16 = "float16x64"

# Max KV tile, in PWORD-blocks. The OMEGA IMC is (4 weightsets, 8 rows, 8 cols)
# PWORD-blocks: kv_tile_blocks <= 8 fits ONE weightset (no spill), <= 32 the full IMC.
_MAX_KV_TILE_BLOCKS = 32

# Auto-tiling cap on kv_tile_blocks (== the IMC cap). With the datapaths now free of dead
# code (the branch-free softmax update + the mask-synth advance-at-start fix), the whole
# kv in {1..32} range compiles and is bit-exact; MAXLIVE stays well under the 72-register
# A+B budget (kv=32 -> ~35). kv>8 spills across IMC weightsets but the backend handles it.
_AUTOTILE_MAX_KV_BLOCKS = 32

# Omega L1 budget in bytes (cf. the allocator's 'global.l1': 4194304 constraint).
_L1_BYTES = 4 * 1024 * 1024


def _dtype_lane_bytes(dtype: str) -> int:
    """Bytes per buffer element for a 'TxN' dtype (int8x64 -> 64, float32x64 -> 256)."""
    base, _, lanes = dtype.partition("x")
    size = {
        "int8": 1,
        "uint8": 1,
        "float16": 2,
        "bfloat16": 2,
        "int16": 2,
        "int32": 4,
        "uint32": 4,
        "float32": 4,
    }.get(base)
    if size is None:
        raise ValueError(f"unknown dtype {dtype!r}")
    return (int(lanes) if lanes else 1) * size


def estimate_l1_footprint(
    seq_tile: int,
    kv_tile_blocks: int,
    *,
    head_dim_blocks: int = 1,
    dv_blocks: int = 1,
    qkv_dtype: str = "int8x64",
    pword: int = None,  # type: ignore[assignment]
) -> dict:
    """Predicted L1 footprint (bytes) of the sdpa_flash resident buffers.

    Mirrors the ``alloc_buffer(... scope=A.L1)`` calls in ``implementation()`` for a given
    (seq_tile, kv_tile_blocks) choice -- the lever for fitting long-context prefill in L1.
    Returns a per-buffer breakdown plus ``total``, ``budget`` and ``fits``. K/V DDR views and
    the IMC weightsets are NOT L1 and are excluded. The two dominant buffers (qk_port1,
    scores_stats) are f16; the O accumulator stays f32. The causal mask is synthesised in-DPU
    from a tiny per-row coeff tile (no DDR mask buffer).

    seq_tile=0 means "no seq tiling" -- pass the full seq_len in that case.
    """
    pword = PWORD if pword is None else pword
    kv_tile_ctx = kv_tile_blocks * pword
    f32 = _dtype_lane_bytes(_F32)
    f16 = _dtype_lane_bytes(_F16)
    qkv = _dtype_lane_bytes(qkv_dtype)

    bd = {
        # Per-KV-tile staging -- O(kv_tile), independent of seq:
        "key_T": head_dim_blocks * pword * kv_tile_blocks * qkv,
        "key_tile": kv_tile_ctx * head_dim_blocks * qkv,
        "value_tile": kv_tile_ctx * head_dim_blocks * qkv,
        # Seq-resident per-row state -- O(seq_tile * kv_tile):
        "qk_port1": seq_tile * (kv_tile_blocks + 2) * f16,
        "quantized_scores": seq_tile * kv_tile_blocks * qkv,
        "scores_stats": seq_tile * (kv_tile_blocks + 3) * f16,
        "ml_carry": seq_tile * 2 * f16,
        "query": seq_tile * head_dim_blocks * qkv,
        "out_state": seq_tile * (dv_blocks + 1) * f32,
        "consts": 3 * f32,
        "coeff": seq_tile * 2 * f32,  # in-DPU causal-mask synth coeffs [seed | qo] per row
    }
    total = sum(bd.values())
    bd["total"] = total
    bd["budget"] = _L1_BYTES
    bd["fits"] = total <= _L1_BYTES
    return bd


def _divisors(n: int) -> list:
    """Sorted divisors of ``n`` (candidate seq_tile values: seq_len % seq_tile == 0)."""
    ds = set()
    i = 1
    while i * i <= n:
        if n % i == 0:
            ds.add(i)
            ds.add(n // i)
        i += 1
    return sorted(ds)


def choose_tiles(
    seq_len: int,
    context_len: int,
    *,
    head_dim_blocks: int = 1,
    dv_blocks: int = 1,
    qkv_dtype: str = "int8x64",
    max_kv_tile_blocks: int = _AUTOTILE_MAX_KV_BLOCKS,
    budget: int = _L1_BYTES,
    pword: int = None,  # type: ignore[assignment]
) -> tuple:
    """Pick ``(seq_tile, kv_tile_blocks)`` that **minimises the total tile count**
    (``n_seq_tiles * n_kv_tiles``) subject to the L1 budget and the op's divisibility
    constraints. Fewer tiles == fewer datapath invocations, IMC reloads and K/V re-streams.

    Total tiles ``= (seq_len/seq_tile) * (context_len/(kv_tile_blocks*PWORD))``, so minimising
    it is the same as maximising the product ``seq_tile * kv_tile_blocks`` -- but the L1
    footprint ``~= seq_tile*(a + b*kv) + c*kv`` couples the two, giving an interior optimum
    (neither "max seq_tile" nor "max kv" alone is right), so we brute-force the (small)
    feasible grid via ``estimate_l1_footprint``.

    Constraints enforced by ``implementation()`` and mirrored here:
      * ``seq_len % seq_tile == 0``       (seq_tile is a divisor of seq_len),
      * ``context_len % (kv_tile_blocks*PWORD) == 0`` (kv tile divides the context in blocks),
      * ``1 <= kv_tile_blocks <= max_kv_tile_blocks`` (<=8 one IMC weightset, <=32 full IMC),
      * the resident buffers fit ``budget`` (via ``estimate_l1_footprint``).

    Ties (equal tile count) are broken toward the BALANCED split -- minimum ``n_seq_tiles +
    n_kv_tiles``. Device profiling (1k-query / 4k-context, all four equal-tile splits) is
    U-shaped in the seq/kv ratio: kv=16/seq_tile=256 (n_seq=n_kv=4) is fastest, beating both
    the seq-heavy kv=4/seq_tile=1024 (-22%) and the kv-heavy kv=32/seq_tile=128 (-3%). A linear
    fit ``cost ~ a*n_seq + b*n_kv`` (a~351, b~427 cycles) puts the optimum where the two tile
    counts are ~equal, i.e. minimum sum at fixed product. Raises ``ValueError`` if nothing fits
    (only for pathologically large head_dim/dv; seq_tile=1, kv_tile_blocks=1 is otherwise tiny).
    """
    pword = PWORD if pword is None else pword
    if context_len % pword != 0:
        raise ValueError(
            f"choose_tiles: context_len ({context_len}) must be a multiple of PWORD ({pword})."
        )
    seq_divs = _divisors(seq_len)
    kv_cands = [kv for kv in range(1, max_kv_tile_blocks + 1) if context_len % (kv * pword) == 0]
    best = None  # ((total_tiles, n_seq + n_kv), seq_tile, kv)
    for kv in kv_cands:
        n_kv = context_len // (kv * pword)
        for st in seq_divs:
            fp = estimate_l1_footprint(
                st,
                kv,
                head_dim_blocks=head_dim_blocks,
                dv_blocks=dv_blocks,
                qkv_dtype=qkv_dtype,
                pword=pword,
            )
            if fp["total"] > budget:
                continue
            # Minimise tile count; tie-break toward the BALANCED split (min n_seq + n_kv).
            n_seq = seq_len // st
            key = (n_seq * n_kv, n_seq + n_kv)
            if best is None or key < best[0]:
                best = (key, st, kv)
    if best is None:
        raise ValueError(
            f"choose_tiles: no (seq_tile, kv_tile_blocks) fits the {budget / 2**20:.0f} MB L1 "
            f"budget for seq_len={seq_len}, context_len={context_len}, "
            f"head_dim_blocks={head_dim_blocks}, dv_blocks={dv_blocks}."
        )
    return best[1], best[2]


def _mask_fill_value(dtype: str) -> float:
    """Additive sentinel for masked positions, representable in *dtype* (cf. core SDPA).

    int8 can't hold -10000 but -127 is exp-negligible; lets the mask be int8.
    """
    return -127.0 if dtype.split("x")[0] == "int8" else -10000.0


def _head_slice(buf, head, dim0_extent, dim1_extent):
    return BufferSlice(BufferLoad(buf, [head, Ramp(0, 1, dim0_extent), Ramp(0, 1, dim1_extent)]))


def _emit_chained_exp(arg, exp_lut, n_exp_luts, exp_method, acc):
    """Emit the (possibly chained) neg-exp LUT applied to one fp18 value `arg`.

    Mirrors the core op's exp handling but for a single scalar (used for alpha).
    Returns the resulting fp18 expression.
    """
    if exp_method is ChainedLUTMethod.ADDITIVE and n_exp_luts > 1:
        out = tir_ops.lut(arg, A.cast(exp_lut[0], _F18))
        for k in A.serial(1, n_exp_luts):
            acc = tir_ops.lut(arg, A.cast(exp_lut[k], _F18))
            out = out + acc
        return out
    return chained_lut_dpu(arg, exp_lut, n_exp_luts, method=ChainedLUTMethod.COMPOSITION, tmp=acc)


# IFD has only TWO ports, each backed by a SINGLE physical hardware stream whose address
# generator does ONE pattern (a contiguous/constant-stride run). Two separate decl_streams
# on one port collide; a fixed-offset second push on a stream that's also advancing does
# too. But two logical inputs CAN share a port if they're laid out contiguously in ONE
# buffer and read as a single run. So each datapath keeps port 1 to one contiguous stream:
# the QK reads [mask | m_old | l_old] fused (qk_port1), and the P@V reads [O_old.. | alpha]
# fused (out_state's last col). m/l/alpha ride L1 (cf. the core op's scores_with_stats),
# NOT bank B, so there is no seq cap.


# Datapath 0 (optional): in-DPU causal-mask synthesis. Instead of streaming a tile of the
# O(seq*ctx) DDR mask (a strided 2D-slice DMA -- the dominant non-contiguous DMA), generate
# the (seq_tile, kv_tile) additive causal mask tile in the DPU from tiny per-row coords, and
# write it straight into the mask region of qk_port1 (an ODR -- strided is free, no DMA).
# Per (query row m, key block n, lane j): t = keypos - querypos, computed in the fp18-exact
# SPLIT form t = 64*(keyblock - qb) + (lane - qo) [keyblock = tile_base_blocks + n]; the host
# folds the tile base into `seed = 64*(tile_base_blocks - qb)` so the DPU streams only
# (seed, qo) per row. step_lut(t) -> 0 (visible, incl. diagonal) if t<=0 else -10000.
# core=None (DPU-only, no matmul) -- the same envelope as _flash_normalize (direct-const +
# LUT), which sidesteps the MVMEXE const-preload/bypass blockers that stalled this in the
# core op. (cf. memory sdpa-causal-mask-synthesis: synthesis works + is bit-exact; it was
# abandoned for the dense op only because mask bandwidth isn't the latency bottleneck there
# -- but here it kills the dominant non-contiguous mask DMA, a compile + footprint win.)
@sliced
def _flash_mask_synth(
    coeff: tir.BufferSlice,
    lane_ramp: tir.BufferSlice,
    c64_const: tir.BufferSlice,
    step_lut: tir.BufferSlice,
    mask_out: tir.BufferSlice,
) -> None:
    # coeff: (seq_tile, 2) f32 = [seed = 64*(tile_base_blocks - qb), qo] per query row.
    # lane_ramp: (1, PWORD) f32, row 0 = [0, 1, ..., PWORD-1] (within-block lane index).
    # c64_const: (1,) f32 = float(PWORD) (the per-block stride of keypos, == 64).
    # step_lut: (1, PWORD) f32 packed [16 delims | 16 slopes | 16 offsets | 16 zeros];
    #   delims[0]=0.5 (t<=0 -> bin 0 -> offset 0 visible), delims[1:]=1e9, offsets[1:]=-10000.
    # mask_out: (seq_tile, kv_tile) f32 additive mask tile (= qk_port1[:, 0:kv_tile]).
    seq_tile = coeff.shape[0]
    kv_tile_blocks = mask_out.shape[1]
    with A.script():
        for m in A.serial(seq_tile):
            with A.Datapath(core=None, id="flash_mask_synth"):
                coeff_stream = A.decl_stream(_F32)
                out_stream = A.decl_stream(_F16)  # writes the f16 mask region of qk_port1
                for c in A.serial(2):
                    with A.IFD(num=0):
                        coeff_stream.push(coeff[m, c])
                with A.DPU():
                    y = A.array(_F18, kv_tile_blocks)
                    seed = A.var(_F18)
                    qo = A.var(_F18)
                    base_v = A.var(_F18)
                    bt = A.var(_F18)
                    t_v = A.var(_F18)
                    seed = A.cast(coeff_stream.pop(), _F18)  # 64*(tile_base_blocks - qb)
                    qo = A.cast(coeff_stream.pop(), _F18)
                    base_v = A.cast(lane_ramp[0], _F18) - qo  # lane - qo (per-lane, small/exact)
                    # Advance bt at the START of each iteration (init seed - 64) rather than the
                    # end, so the final increment is consumed by the last t_v. An end-of-loop
                    # advance leaves a dead `bt += 64` after the last use; with no intra-DPU
                    # scalar DCE before register allocation, that dead dst is left unplaced once
                    # MAXLIVE exceeds bank A (kv>=16) and the bank-B allocator takes over.
                    bt = seed - A.cast(c64_const[0], _F18)  # so bt = seed at n=0 after the advance
                    for n in A.serial(kv_tile_blocks):
                        bt = bt + A.cast(
                            c64_const[0], _F18
                        )  # advance one key block (bt = seed + n*64)
                        t_v = bt + base_v  # = 64*(keyblock - qb) + (lane - qo) = keypos - querypos
                        y[n] = tir_ops.lut(t_v, A.cast(step_lut[0], _F18))
                    for n in A.serial(kv_tile_blocks):
                        out_stream.push(A.cast(y[n], _F16))
                for n in A.serial(kv_tile_blocks):
                    with A.ODR():
                        mask_out[m, n] = out_stream.pop()


# Datapath 1: QK^T matmul + online-softmax update of (m, l, alpha) + quantized P tile.
# port 0 = query (MVM); port 1 = one contiguous fused stream qk_port1[m, :] holding the
# mask tile then m_old, l_old (the mask cast to f32 at staging so the run is single-dtype).
@sliced
def _flash_qk_softmax(
    query: tir.BufferSlice,
    imc_k: tir.BufferSlice,
    qk_port1: tir.BufferSlice,
    scores_stats: tir.BufferSlice,
    attention_scores_dequantizer_scale: tir.BufferSlice,
    attention_weights_quantizer_scale: tir.BufferSlice,
    attention_weights_quantizer_zero_point: tir.BufferSlice,
    exp_lut: tir.BufferSlice,
    n_exp_luts: int,
    exp_method: ChainedLUTMethod,
    scale_const: tir.BufferSlice,
    zero_const: tir.BufferSlice,
) -> None:
    # Branch-free online-softmax update -- no tile_idx (the carry init makes the first tile
    # fall out of the general recurrence; see the update comment below + implementation()).
    # scores_stats is (seq, kv_tile_blocks + 3): [0:kvtb] = scaled (pre-int8) weights P,
    # [kvtb]=m, [kvtb+1]=l, [kvtb+2]=alpha; ONE ODR writes the row, a later dpu_cast
    # quantizes the P cols. qk_port1 = (seq, kvtb+2) f32: mask[0:kvtb] then m_old, l_old.
    seq_len = query.shape[0]
    kv_tile_blocks = imc_k.shape[1]
    head_dim_blocks = imc_k.shape[0]

    dq_scale = A.alloc_buffer(
        tuple([attention_scores_dequantizer_scale.shape[-1]]),
        attention_scores_dequantizer_scale.dtype,
        scope=attention_scores_dequantizer_scale.scope(),
        data=attention_scores_dequantizer_scale.buffer.data,
    )
    w_q_scale = A.alloc_buffer(
        tuple([attention_weights_quantizer_scale.shape[-1]]),
        attention_weights_quantizer_scale.dtype,
        scope=attention_weights_quantizer_scale.scope(),
        data=attention_weights_quantizer_scale.buffer.data,
    )
    w_q_zp = A.alloc_buffer(
        tuple([attention_weights_quantizer_zero_point.shape[-1]]),
        attention_weights_quantizer_zero_point.dtype,
        scope=attention_weights_quantizer_zero_point.scope(),
        data=attention_weights_quantizer_zero_point.buffer.data,
    )

    with A.script():
        for m in A.serial(seq_len):
            with A.Datapath(kind=A.DP_MVMEXE, id="flash_qk"):
                ifd_stream = A.decl_stream(_F32)  # query activations (port 0, MVM)
                port1_stream = A.decl_stream(_F16)  # mask + m_old + l_old (port 1, fused, f16)
                matmul_stream = A.decl_stream("int32x64")
                out_stream = A.decl_stream(_F16)  # packed [P|m|l|alpha] row -> scores_stats (f16)

                for n, r in A.grid(kv_tile_blocks, head_dim_blocks):
                    with A.IFD(num=0):
                        ifd_stream.push(query[m, r])
                for n in A.serial(kv_tile_blocks):
                    with A.pseudo.MatMul():
                        acc = A.mvm_accumulator()
                        A.reset_accumulator(acc)
                        for r in A.serial(0, head_dim_blocks):
                            acc += A.mvm_1x1(ifd_stream.pop(), imc_k[r, n])
                        matmul_stream.push(acc)
                for c in A.serial(kv_tile_blocks + 2):
                    with A.IFD(num=1):
                        port1_stream.push(qk_port1[m, c])

                with A.DPU():
                    x = A.array(_F18, kv_tile_blocks)
                    y = A.array(_F18, kv_tile_blocks)
                    aux = A.array(_F18, 1)  # scratch for the alpha LUT argument
                    tile_max = A.var(_F18)
                    m_new = A.var(_F18)
                    m_old = A.var(_F18)
                    l_old = A.var(_F18)
                    alpha = A.var(_F18)
                    s_tile = A.var(_F18)
                    l_new = A.var(_F18)
                    lut_acc = A.var(_F18)

                    # Dequantise / scale / mask the score tile. The mask is the first
                    # kvtb elements of the fused port-1 stream.
                    for i in A.serial(kv_tile_blocks):
                        x[i] = matmul_stream.pop()
                        x[i] = x[i] * A.cast(dq_scale[0], _F18)
                        x[i] = x[i] + A.cast(zero_const[0], _F18)
                        x[i] = x[i] * A.cast(scale_const[0], _F18)
                        x[i] = x[i] + A.cast(port1_stream.pop(), _F18)

                    # Tile row-max.
                    tile_max = x[0]
                    for i in A.serial(1, kv_tile_blocks):
                        tile_max = A.max(tile_max, x[i])
                    tile_max = tir_ops.dpu_reduce(tile_max, operation=tir_ops.DPU_REDUCE_OP.MAX)

                    # Online-softmax update -- UNIFORM across tiles (no tile_idx branch). The
                    # carry is initialised to m_old=-BIG, l_old=0, O_old=0 (see implementation()),
                    # so the first tile falls out of the general recurrence: m_new =
                    # max(-BIG, tile_max) = tile_max; alpha = exp(m_old - m_new) ~ 0, but it only
                    # ever multiplies l_old / O_old (both 0 on tile 0), so l_new = s_tile and
                    # O_new = contrib regardless of alpha's exact value. Keeping the body
                    # branch-free is what avoids dead code: an A.If / A.Select on tile_idx leaves
                    # the folded-away first-tile branch's producers (or its input loads) with no
                    # consumer, and there is no intra-DPU scalar DCE before register allocation,
                    # so the allocator left those dead dsts unplaced (check_allocation / the
                    # scheduler then failed at kv>=4). The LUT arg must be a buffer element, so
                    # (m_new - m_old) goes through aux[0]; max() keeps it >= 0 (the LUT domain).
                    m_old = A.cast(port1_stream.pop(), _F18)
                    l_old = A.cast(port1_stream.pop(), _F18)
                    m_new = A.max(m_old, tile_max)
                    aux[0] = m_new - m_old
                    alpha = _emit_chained_exp(aux[0], exp_lut, n_exp_luts, exp_method, lut_acc)

                    # P_j = exp(x - m_new) = neg_exp_lut(m_new - x), arg >= 0.
                    if exp_method is ChainedLUTMethod.ADDITIVE and n_exp_luts > 1:
                        for i in A.serial(0, kv_tile_blocks):
                            x[i] = m_new - x[i]
                            y[i] = tir_ops.lut(x[i], A.cast(exp_lut[0], _F18))
                            for k in A.serial(1, n_exp_luts):
                                lut_acc = tir_ops.lut(x[i], A.cast(exp_lut[k], _F18))
                                y[i] = y[i] + lut_acc
                    else:
                        for i in A.serial(0, kv_tile_blocks):
                            x[i] = m_new - x[i]
                            y[i] = chained_lut_dpu(
                                x[i],
                                exp_lut,
                                n_exp_luts,
                                method=ChainedLUTMethod.COMPOSITION,
                                tmp=lut_acc,
                            )

                    # Tile denom (rowsum P_j).
                    s_tile = y[0]
                    for i in A.serial(1, kv_tile_blocks):
                        s_tile = s_tile + y[i]
                    s_tile = tir_ops.dpu_reduce(s_tile, operation=tir_ops.DPU_REDUCE_OP.ADD)

                    # l_new = alpha*l_old + s_tile (uniform; tile 0 has l_old=0 -> l_new=s_tile).
                    l_new = alpha * l_old  # one op per DPU statement
                    l_new = l_new + s_tile

                    # Scale P (int8 round/clip done later by atomic.dpu_cast) and emit the
                    # packed row [P_0..P_{kvtb-1}, m_new, l_new, alpha] in ONE ODR.
                    for i in A.serial(kv_tile_blocks):
                        y[i] = y[i] * A.cast(w_q_scale[0], _F18)
                        y[i] = y[i] + A.cast(w_q_zp[0], _F18)
                        out_stream.push(A.cast(y[i], _F16))
                    out_stream.push(A.cast(m_new, _F16))
                    out_stream.push(A.cast(l_new, _F16))
                    out_stream.push(A.cast(alpha, _F16))

                for n in A.serial(kv_tile_blocks + 3):
                    with A.ODR():
                        scores_stats[m, n] = out_stream.pop()


# Datapath 2: P@V matmul-accumulate -- O_new = alpha*O_old + (P@V)*output_dq, in one
# datapath, ONE per query row (not per dv-block). P feeds the MVM (port 0, re-streamed
# per output block exactly as the QK matmul re-streams the query). The whole O row plus
# the per-row alpha are read on port 1 as a SINGLE contiguous stream out_state[m, 0:dv+1]
# (alpha in the last column) -- a single address-generator run, which is what the one
# physical stream per port supports (a fixed-offset 2nd push does NOT). Folding the alpha
# rescale in here drops both the separate o_rescale datapath AND the per-dv-block grid.
@sliced
def _flash_pv_accumulate(
    quantized_scores: tir.BufferSlice,
    imc_v: tir.BufferSlice,
    out_state: tir.BufferSlice,
    attention_output_dequantizer_scale: tir.BufferSlice,
) -> None:
    seq_len = quantized_scores.shape[0]
    kv_tile_blocks = quantized_scores.shape[1]
    head_dim_blocks = imc_v.shape[1]  # dv blocks; out_state is (seq, dv+1), alpha @ col dv
    output_dq = A.alloc_buffer(
        tuple([attention_output_dequantizer_scale.shape[-1]]),
        attention_output_dequantizer_scale.dtype,
        scope=attention_output_dequantizer_scale.scope(),
        data=attention_output_dequantizer_scale.buffer.data,
    )
    with A.script():
        for m in A.serial(seq_len):
            with A.Datapath(kind=A.DP_MVMEXE, id="flash_pv"):
                ifd_stream = A.decl_stream("int8x64")
                ovec_stream = A.decl_stream(_F32)  # O_old[0:dv] + alpha, one contiguous run
                matmul_stream = A.decl_stream("int32x64")
                dpu_stream = A.decl_stream(_F32)
                for n, r in A.grid(head_dim_blocks, kv_tile_blocks):
                    with A.IFD(num=0):
                        ifd_stream.push(quantized_scores[m, r])
                for n in A.serial(head_dim_blocks):
                    with A.pseudo.MatMul():
                        acc = A.mvm_accumulator()
                        A.reset_accumulator(acc)
                        for r in A.serial(kv_tile_blocks):
                            acc += A.mvm_1x1(ifd_stream.pop(), imc_v[r, n])
                        matmul_stream.push(acc)
                for c in A.serial(head_dim_blocks + 1):
                    with A.IFD(num=1):
                        ovec_stream.push(out_state[m, c])
                with A.DPU():
                    ov = A.array(_F18, head_dim_blocks + 1)  # O_old[0:dv], then alpha
                    contrib = A.var(_F18)
                    o_new = A.var(_F18)
                    a_f = A.var(_F18)
                    for c in A.serial(head_dim_blocks + 1):
                        ov[c] = A.cast(ovec_stream.pop(), _F18)
                    a_f = ov[head_dim_blocks]  # alpha (=0 on tile 0 -> O = contrib)
                    for n in A.serial(head_dim_blocks):
                        contrib = A.cast(matmul_stream.pop(), _F18)
                        contrib = contrib * A.cast(output_dq[0], _F18)
                        o_new = a_f * ov[n]  # one op per DPU statement
                        o_new = o_new + contrib
                        dpu_stream.push(A.cast(o_new, _F32))
                for n in A.serial(head_dim_blocks):
                    with A.ODR():
                        out_state[m, n] = dpu_stream.pop()


# Datapath 3: post-loop normalize O *= 1/l. Per (row, dv-block): reads the running O
# (IFD0) and the final per-row denom l from scores_stats (IFD1) -- 2 streams, no
# matmul, so a plain DPU-only (core=None) datapath. l is in L1 (not a bank), so this
# is readable here regardless of datapath kind.
@sliced
def _flash_normalize(
    out_state: tir.BufferSlice,
    scores_stats: tir.BufferSlice,
    inv_lut: tir.BufferSlice,
    n_inv_luts: int,
    inv_method: ChainedLUTMethod,
    newton_const: tir.BufferSlice,
    use_newton_raphson: bool,
    output_buffer: tir.BufferSlice,
) -> None:
    seq_len = out_state.shape[0]
    dv_blocks = out_state.shape[1]
    l_col = int(scores_stats.shape[1]) - 2  # [P | m | l | alpha]
    with A.script():
        for m, n in A.grid(seq_len, dv_blocks):
            with A.Datapath(core=None, id="flash_norm"):
                o_stream = A.decl_stream(_F32)  # O from f32 accumulator
                l_stream = A.decl_stream(_F16)  # l from f16 scores_stats
                dpu_stream = A.decl_stream(_F32)  # normalized O written fp32 straight to DDR
                with A.IFD(num=0):
                    o_stream.push(out_state[m, n])
                with A.IFD(num=1):
                    l_stream.push(scores_stats[m, l_col])
                with A.DPU():
                    o_f = A.var(_F18)
                    inv_l = A.var(_F18)
                    sum_l = A.var(_F18)
                    temp = A.var(_F18)
                    lut_acc = A.var(_F18)
                    sum_l = A.cast(l_stream.pop(), _F18)
                    if inv_method is ChainedLUTMethod.ADDITIVE and n_inv_luts > 1:
                        lut_acc = sum_l
                        inv_l = tir_ops.lut(lut_acc, A.cast(inv_lut[0], _F18))
                        for k in A.serial(1, n_inv_luts):
                            temp = tir_ops.lut(lut_acc, A.cast(inv_lut[k], _F18))
                            inv_l = inv_l + temp
                    else:
                        inv_l = chained_lut_dpu(
                            sum_l,
                            inv_lut,
                            n_inv_luts,
                            method=ChainedLUTMethod.COMPOSITION,
                            tmp=lut_acc,
                        )
                    if use_newton_raphson:
                        temp = sum_l * inv_l
                        temp = A.cast(newton_const[0], _F18) - temp
                        inv_l = inv_l * temp
                    o_f = A.cast(o_stream.pop(), _F18)
                    o_f = o_f * inv_l
                    dpu_stream.push(A.cast(o_f, _F32))
                with A.ODR():
                    output_buffer[m, n] = dpu_stream.pop()


class SDPAFlash(ScriptOp):
    """Flash (KV-tiled) scaled dot-product attention. See module docstring.

    Operands mirror axol.core.scaled_dot_product_attention so a kernel can swap
    this in for long-context prefill where the core op overflows L1 / IMC.
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
        MetisTypes[tir.Buffer, "..."],
        description="(*leading, seq_q, ctx_blocks). MUST be the standard causal mask (queries "
        "= last seq positions): sdpa_flash is causal-only and SYNTHESISES the mask in-DPU from "
        "positions -- this operand's values are NOT streamed (used only by emulate() as the "
        "reference). A non-causal/padding mask would be silently ignored on device.",
    )

    is_causal = Arg(bool, default=False, kw=True, description="Unsupported on HW; must be False.")
    enable_gqa = Arg(bool, default=False, kw=True, description="Grouped-query attention.")
    use_newton_raphson = Arg(
        bool, default=False, kw=True, description="One Newton step after the inv LUT."
    )
    scale = Arg(float, default=1.0, kw=True, description="Dot-product scaling factor.")
    kv_tile_blocks = Arg(
        int,
        default=0,
        kw=True,
        description="Context tile size in PWORD-blocks. 0 (default) auto-picks BOTH "
        "kv_tile_blocks and seq_tile via choose_tiles() to minimise the total tile count "
        "within the L1 budget (overriding seq_tile). A non-zero value is used as-is "
        "(<=8 fits one IMC weightset, <=32 fits the full IMC).",
    )
    seq_tile = Arg(
        int,
        default=0,
        kw=True,
        description="Query/seq tile size in rows (0 = full seq, no seq tiling). Tiling the "
        "seq axis bounds the L1 footprint (per-row state is O(seq_tile*kv_tile)) so large "
        "seq fits without shrinking kv_tile_blocks; K/V are re-streamed per seq tile.",
    )
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
    cores = Arg(
        int | tuple[int, ...] | None,
        default=None,
        kw=True,
        description="AIPU cores to distribute the query heads over. int -> cores 0..n-1; "
        "tuple -> explicit ids; None (default) -> single core.",
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
        output = self.output

        enable_gqa = bool(self.enable_gqa)
        use_newton_raphson = bool(self.use_newton_raphson)
        scale = float(self.scale)
        exp_method = ChainedLUTMethod(self.exp_method)
        inv_method = ChainedLUTMethod(self.inv_method)
        kv_tile_blocks = int(self.kv_tile_blocks)

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
        head_dim_blocks = int(key.shape[3])
        dv_blocks = int(output.shape[-1])
        kv_groups = num_heads // num_kv_heads

        n_exp_luts = int(self.exp_lut.shape[0])
        n_inv_luts = int(self.inv_lut.shape[0])

        # Cores to distribute the query heads over (empty / single -> single-core path).
        # The ScriptOp arg arrives as a scalar (int / tir IntImm) core *count*, an
        # iterable (tuple / tir Array) of core ids, or None.
        _cores = self.cores
        if _cores is None:
            core_ids: tuple[int, ...] = ()
        else:
            try:
                core_ids = tuple(int(c) for c in _cores)  # iterable of ids
            except TypeError:
                core_ids = tuple(range(int(_cores)))  # scalar count -> 0..n-1

        # kv_tile_blocks == 0 -> auto: pick BOTH tiles (overriding seq_tile) to minimise the
        # total tile count within the L1 budget (see choose_tiles). An explicit kv_tile_blocks
        # keeps caller control (and seq_tile=0 then still means "full seq, no seq tiling").
        if kv_tile_blocks == 0:
            seq_tile, kv_tile_blocks = choose_tiles(
                seq_len,
                context_len,
                head_dim_blocks=head_dim_blocks,
                dv_blocks=dv_blocks,
                qkv_dtype=str(query.dtype),
            )
            _logger.info(
                "sdpa_flash auto-tiled: seq_tile=%d, kv_tile_blocks=%d (seq_len=%d, context_len=%d)",
                seq_tile,
                kv_tile_blocks,
                seq_len,
                context_len,
            )
        else:
            # Seq (query) tiling: process seq_tile query rows at a time so the per-row L1 state
            # is O(seq_tile * kv_tile) instead of O(seq * kv_tile). 0 -> full seq (no tiling).
            seq_tile = int(self.seq_tile) or seq_len

        kv_tile_ctx = kv_tile_blocks * PWORD
        n_kv_tiles = (context_len + kv_tile_ctx - 1) // kv_tile_ctx

        if seq_len % seq_tile != 0:
            raise NotImplementedError(
                f"sdpa_flash requires seq_len ({seq_len}) to be a multiple of seq_tile "
                f"({seq_tile}); a partial-last-seq-tile path is future work."
            )
        n_seq_tiles = seq_len // seq_tile

        # In-DPU causal-mask synthesis constants + per-(seq-tile, kv-tile, row) coeffs.
        # coeff[s, t, m] = [seed = 64*(t*kv_tile_blocks - qb), qo] where the query's absolute
        # position is querypos = (context_len - seq_len) + s*seq_tile + m, qb = querypos//64,
        # qo = querypos%64. Folding the tile base into seed lets the DPU stream just 2 scalars
        # per row. Built host-side (O(seq*n_kv_tiles), tiny), sliced per (s, t) -- contiguous.
        _qpos = (context_len - seq_len) + np.arange(seq_len, dtype=np.int64)
        _qb = _qpos // PWORD
        _qo = (_qpos % PWORD).astype(np.float32)
        _coeff = np.zeros((n_seq_tiles, n_kv_tiles, seq_tile, 2, PWORD), dtype=np.float32)
        for _s in range(n_seq_tiles):
            for _t in range(n_kv_tiles):
                _rows = slice(_s * seq_tile, (_s + 1) * seq_tile)
                _seed = (PWORD * (_t * kv_tile_blocks - _qb[_rows])).astype(np.float32)
                _coeff[_s, _t, :, 0, :] = _seed[:, None]
                _coeff[_s, _t, :, 1, :] = _qo[_rows][:, None]
        _lane_ramp = np.arange(PWORD, dtype=np.float32).reshape(1, PWORD)
        _c64 = np.array([[float(PWORD)] * PWORD], dtype=np.float32)
        _step = np.zeros((1, PWORD), dtype=np.float32)
        _step[0, 0:16] = 1e9  # bin delimiters
        _step[0, 0] = 0.5  # t<=0 -> bin 0 (visible, incl. diagonal)
        _step[0, 32:48] = -10000.0  # offsets (masked)
        _step[0, 32] = 0.0  # bin 0 offset: visible
        # NB: the host numpy arrays for the DDR constants are built ONCE here, but the
        # A.alloc_constant() calls live PER CORE inside _run_head_range -- a constant
        # allocated at this (shared) scope is only reachable from core 0, so non-zero
        # cores would read garbage (e.g. a fully-masked tile -> Inf rows).
        # Early, actionable L1 check (the allocator's failure is cryptic + ~60s in).
        _l1 = estimate_l1_footprint(
            seq_tile,
            kv_tile_blocks,
            head_dim_blocks=head_dim_blocks,
            dv_blocks=dv_blocks,
            qkv_dtype=str(query.dtype),
        )
        _logger.info(
            "sdpa_flash L1 footprint: %.2f MB / %.2f MB (seq_tile=%d, kv_tile_blocks=%d)",
            _l1["total"] / 2**20,
            _l1["budget"] / 2**20,
            seq_tile,
            kv_tile_blocks,
        )
        if not _l1["fits"]:
            _big = ", ".join(
                f"{k}={v / 2**20:.2f}MB"
                for k, v in sorted(
                    _l1.items(), key=lambda kv: -kv[1] if isinstance(kv[1], int) else 0
                )
                if k not in ("total", "budget", "fits") and isinstance(v, int) and v > 2**18
            )
            raise ValueError(
                f"sdpa_flash L1 footprint {_l1['total'] / 2**20:.2f} MB exceeds the "
                f"{_l1['budget'] / 2**20:.0f} MB budget for seq_tile={seq_tile}, "
                f"kv_tile_blocks={kv_tile_blocks}. Reduce kv_tile_blocks and/or set a "
                f"smaller seq_tile. Largest buffers: {_big}."
            )

        _scale_arr = np.array([[scale] * PWORD], dtype=np.float32)
        _newton_arr = np.array([[2.0] * PWORD], dtype=np.float32)
        _zero_arr = np.array([[0.0] * PWORD], dtype=np.float32)
        # Carry init for the branch-free recurrence: [m=-BIG, l=0] per row, copied (contiguous)
        # into ml_carry each seq tile. A DDR const + copy (not a strided atomic.full on the m
        # column) so the write is a single contiguous run -- a strided 1-column fill mis-writes.
        _ml_init = np.zeros((seq_tile, 2, PWORD), dtype=np.float32)
        _ml_init[:, 0, :] = -30000.0

        # Single uniform A.serial loop over KV tiles (no peeled first/last block). The
        # per-tile differences are expressed with A.Select on the loop var inside the
        # datapaths, so every iteration has an identical structure / one load per tile.
        # That is what avoids the InjectDMAs hoist hazard a peeled-last block triggered
        # (it would load the last tile into the shared staging buffer ahead of the
        # middle tiles). Requires context_len and the mask to be exact multiples of the
        # tile so K/V/mask can be addressed as a leading index into reshaped views
        # view[..., t, ...] (the slice form the DMA lowering accepts for a runtime t).
        if context_len % kv_tile_ctx != 0:
            raise NotImplementedError(
                "sdpa_flash currently requires context_len to be an exact multiple of kv_tile "
                f"({kv_tile_ctx} tokens / {kv_tile_blocks} blocks); got context_len={context_len}."
            )

        def _run_head_range(h_lo, h_hi):
            # Per-core working set: each core allocates its own L1 / IMC (L1 is per-core)
            # and runs its band of query heads. Single-core (cores=None) calls this once.
            # Per-core DDR constants (built from the shared host numpy arrays above) --
            # each core MUST allocate its own copy or non-zero cores read garbage.
            coeff_const = A.alloc_constant(
                _coeff.reshape(n_seq_tiles, n_kv_tiles, seq_tile, 2, PWORD), dtype=_F32
            )
            lane_ramp_const = A.alloc_constant(_lane_ramp, dtype=_F32)
            c64_const = A.alloc_constant(_c64, dtype=_F32)
            step_lut_const = A.alloc_constant(_step, dtype=_F32)
            scale_const_ddr = A.alloc_constant(_scale_arr, dtype=_F32)
            newton_const_ddr = A.alloc_constant(_newton_arr, dtype=_F32)
            zero_const_ddr = A.alloc_constant(_zero_arr, dtype=_F32)
            ml_init_ddr = A.alloc_constant(_ml_init, dtype=_F16)
            # Reshaped views of the I/O DDR buffers, PER CORE: each core declares its own
            # alias so the multicore DMA dependency tracking treats the per-core (disjoint
            # head-slice) reads/writes as independent. A single shared aliased view written
            # by all cores raced/poisoned the non-zero cores' output channels. The query
            # seq axis is split into (n_seq_tiles, seq_tile) so a runtime-s slice
            # view[..., s, :, ...] lowers cleanly (the same leading-index trick K/V use).
            key_view = A.alloc_buffer(
                (batch_size, num_kv_heads, n_kv_tiles, kv_tile_ctx, head_dim_blocks),
                dtype=key.dtype,
                scope=key.scope(),
                data=key.buffer.data,
            )
            value_view = A.alloc_buffer(
                (batch_size, num_kv_heads, n_kv_tiles, kv_tile_ctx, head_dim_blocks),
                dtype=value.dtype,
                scope=value.scope(),
                data=value.buffer.data,
            )
            query_view = A.alloc_buffer(
                (batch_size, num_heads, n_seq_tiles, seq_tile, head_dim_blocks),
                dtype=query.dtype,
                scope=query.scope(),
                data=query.buffer.data,
            )
            output_view = A.alloc_buffer(
                (batch_size, num_heads, n_seq_tiles, seq_tile, dv_blocks),
                dtype=output.dtype,
                scope=output.scope(),
                data=output.buffer.data,
            )
            # Per-tile IMC + staging buffers (sized to one KV tile -> O(kv_tile)).
            coeff_l1 = A.alloc_buffer((seq_tile, 2), _F32, scope=A.L1)
            imc_k = A.alloc_imc((head_dim_blocks, kv_tile_blocks))
            imc_v = A.alloc_imc((kv_tile_blocks, head_dim_blocks))
            key_T = A.alloc_buffer(
                (head_dim_blocks * PWORD, kv_tile_blocks), dtype=key.dtype, scope=A.L1
            )
            key_tile_l1 = A.alloc_buffer(
                (kv_tile_ctx, head_dim_blocks), dtype=key.dtype, scope=A.L1
            )
            value_tile_l1 = A.alloc_buffer(
                (kv_tile_ctx, head_dim_blocks), dtype=value.dtype, scope=A.L1
            )
            qk_port1_tile = A.alloc_buffer((seq_tile, kv_tile_blocks + 2), _F16, scope=A.L1)
            quantized_scores_tile = A.alloc_buffer(
                (seq_tile, kv_tile_blocks), "int8x64", scope=A.L1
            )
            scores_stats_tile = A.alloc_buffer((seq_tile, kv_tile_blocks + 3), _F16, scope=A.L1)
            ml_carry = A.alloc_buffer((seq_tile, 2), _F16, scope=A.L1)
            query_l1 = A.alloc_buffer((seq_tile, head_dim_blocks), dtype=query.dtype, scope=A.L1)
            out_state = A.alloc_buffer((seq_tile, dv_blocks + 1), _F32, scope=A.L1)
            scale_const_l1 = A.alloc_buffer((1,), _F32, scope=A.L1)
            newton_const_l1 = A.alloc_buffer((1,), _F32, scope=A.L1)
            zero_const_l1 = A.alloc_buffer((1,), _F32, scope=A.L1)

            with A.serial(batch_size) as batch:
                atomic.copy(scale_const_ddr, output=scale_const_l1)
                atomic.copy(newton_const_ddr, output=newton_const_l1)
                atomic.copy(zero_const_ddr, output=zero_const_l1)

                with A.serial(h_hi - h_lo) as head_local:
                    head = h_lo + head_local
                    kv_h = head // kv_groups

                    # Outer query/seq-tile loop: each block of seq_tile rows runs its own,
                    # independent online-softmax over all KV tiles (no cross-block state), so
                    # the resident per-row L1 state is bounded by seq_tile. K/V are re-streamed
                    # per seq tile (the cost of seq tiling); the query/mask/output are addressed
                    # via the (n_seq_tiles, seq_tile) split views.
                    with A.serial(n_seq_tiles) as s:
                        q_src = BufferSlice(
                            BufferLoad(
                                query_view,
                                [
                                    batch,
                                    head,
                                    s,
                                    Ramp(0, 1, seq_tile),
                                    Ramp(0, 1, head_dim_blocks),
                                ],
                            )
                        )
                        atomic.copy(q_src, output=query_l1)

                        # Carry init for the branch-free recurrence (see _flash_qk_softmax): the
                        # O accumulator and l carry start at 0, and the m carry starts at -BIG so
                        # tile 0 falls out of the general update -- max(-BIG, tile_max) = tile_max,
                        # and alpha = exp(m_old - m_new) ~ 0 multiplies only the (zero) O_old / l_old.
                        # -BIG must be below any possible tile row-max (the mask fill is -10000), so
                        # -30000 is safe; it is exact enough in f16 (precision is irrelevant here).
                        atomic.full((seq_tile, dv_blocks + 1), 0.0, output=out_state)
                        atomic.copy(ml_init_ddr, output=ml_carry)

                        with A.serial(n_kv_tiles) as t:
                            key_src = BufferSlice(
                                BufferLoad(
                                    key_view,
                                    [
                                        batch,
                                        kv_h,
                                        t,
                                        Ramp(0, 1, kv_tile_ctx),
                                        Ramp(0, 1, head_dim_blocks),
                                    ],
                                )
                            )
                            value_src = BufferSlice(
                                BufferLoad(
                                    value_view,
                                    [
                                        batch,
                                        kv_h,
                                        t,
                                        Ramp(0, 1, kv_tile_ctx),
                                        Ramp(0, 1, head_dim_blocks),
                                    ],
                                )
                            )

                            atomic.copy(key_src, output=key_tile_l1)
                            core.transpose(key_tile_l1, (1, 0), key_T)
                            atomic.load_imc(key_T, imc_k)
                            atomic.copy(value_src, output=value_tile_l1)

                            # Fill the mask region of qk_port1 (cols 0:kvtb): synthesize the causal
                            # mask in-DPU from tiny per-row coords (no DDR mask, no strided mask DMA).
                            coeff_src = BufferSlice(
                                BufferLoad(
                                    coeff_const, [s, t, Ramp(0, 1, seq_tile), Ramp(0, 1, 2)]
                                )
                            )
                            atomic.copy(coeff_src, output=coeff_l1)
                            _flash_mask_synth(
                                coeff=coeff_l1,
                                lane_ramp=lane_ramp_const,
                                c64_const=c64_const,
                                step_lut=step_lut_const,
                                mask_out=qk_port1_tile[:, 0:kv_tile_blocks],
                            )
                            # Carried m/l into qk_port1's [kvtb:kvtb+2] band -- DPU (dpu_cast),
                            # not atomic.copy, so the strided write rides the ODR iterator (free).
                            atomic.dpu_cast(
                                ml_carry, qk_port1_tile[:, kv_tile_blocks : kv_tile_blocks + 2]
                            )

                            # 1: Q@K + dequant/scale/mask + online-softmax update (m,l,alpha) + P.
                            _flash_qk_softmax(
                                query=query_l1,
                                imc_k=imc_k,
                                qk_port1=qk_port1_tile,
                                scores_stats=scores_stats_tile,
                                attention_scores_dequantizer_scale=self.attention_scores_dequantizer_scale,
                                attention_weights_quantizer_scale=self.attention_weights_quantizer_scale,
                                attention_weights_quantizer_zero_point=(
                                    self.attention_weights_quantizer_zero_point
                                ),
                                exp_lut=self.exp_lut,
                                n_exp_luts=n_exp_luts,
                                exp_method=exp_method,
                                scale_const=scale_const_l1,
                                zero_const=zero_const_l1,
                            )
                            # Quantize the P columns of the packed stats buffer to int8.
                            atomic.dpu_cast(
                                scores_stats_tile[:, 0:kv_tile_blocks], quantized_scores_tile
                            )
                            # Stage this tile's alpha into out_state's last column so the P@V
                            # datapath can read O_old + alpha as a single fused port-1 stream.
                            # DPU (dpu_cast), not atomic.copy: it's a strided 1-col write.
                            atomic.dpu_cast(
                                scores_stats_tile[:, kv_tile_blocks + 2 : kv_tile_blocks + 3],
                                out_state[:, dv_blocks : dv_blocks + 1],
                            )
                            # 2: P@V matmul-accumulate -> O_new = alpha*O_old + (P@V)*output_dq.
                            atomic.load_imc(value_tile_l1, imc_v)
                            _flash_pv_accumulate(
                                quantized_scores=quantized_scores_tile,
                                imc_v=imc_v,
                                out_state=out_state,
                                attention_output_dequantizer_scale=(
                                    self.attention_output_dequantizer_scale
                                ),
                            )
                            # Snapshot this tile's m/l into the carry buffer for the next tile.
                            # DPU (dpu_cast), not atomic.copy: it's a strided column-band read.
                            atomic.dpu_cast(
                                scores_stats_tile[:, kv_tile_blocks : kv_tile_blocks + 2],
                                ml_carry,
                            )

                        # Separate normalize pass: O *= 1/l (final l from scores_stats). Reads the
                        # fp16 O accumulator and writes the normalized result as fp32 STRAIGHT to the
                        # DDR output (folds the old trailing atomic.copy, and bridges fp16->fp32).
                        out_dst = BufferSlice(
                            BufferLoad(
                                output_view,
                                [batch, head, s, Ramp(0, 1, seq_tile), Ramp(0, 1, dv_blocks)],
                            )
                        )
                        _flash_normalize(
                            out_state=out_state[:, 0:dv_blocks],
                            scores_stats=scores_stats_tile,
                            inv_lut=self.inv_lut,
                            n_inv_luts=n_inv_luts,
                            inv_method=inv_method,
                            newton_const=newton_const_l1,
                            use_newton_raphson=use_newton_raphson,
                            output_buffer=out_dst,
                        )

        if len(core_ids) > 1:
            # Distribute the query heads across the requested cores (heads are
            # independent: own Q/K/V slices + L1 state, no cross-head reduction).
            # Same AICore + barrier pattern as the multicore matmul / core SDPA.
            partitions = list(shards(num_heads, core_ids))
            participants = [core_id for core_id, _, _ in partitions]
            for core_id, h_lo, h_hi in partitions:
                with A.AICore(core=core_id):
                    A.barrier(*participants)
                    _run_head_range(h_lo, h_hi)
                    A.barrier(*participants)
        else:
            _run_head_range(0, num_heads)

    # ------------------------------------------------------------------ #
    def emulate(self) -> None:  # type: ignore[override]
        """Float18 model of the online algorithm (NOT the core one-shot reference).

        Mirrors the datapaths' LUTs, reductions and left-to-right accumulation so
        device output matches.
        """
        query = self.query
        key = self.key
        value = self.value
        output = self.output

        scale = float(self.scale)
        use_newton_raphson = bool(self.use_newton_raphson)
        exp_method = ChainedLUTMethod(self.exp_method)
        inv_method = ChainedLUTMethod(self.inv_method)
        kv_tile_blocks = int(self.kv_tile_blocks)
        if kv_tile_blocks == 0:  # auto: resolve kv tile exactly as implementation() does
            _, kv_tile_blocks = choose_tiles(
                int(query.shape[2]),
                int(key.shape[2]),
                head_dim_blocks=int(key.shape[3]),
                dv_blocks=int(output.shape[-1]),
                qkv_dtype=str(query.dtype),
            )
        kv_tile_ctx = kv_tile_blocks * PWORD

        pword = query.shape[-1]
        kv_groups = query.shape[1] // key.shape[1]
        q = np.reshape(query, query.shape[0:-2] + (query.shape[-2] * query.shape[-1],))
        k = np.reshape(key, key.shape[0:-2] + (key.shape[-2] * key.shape[-1],)).repeat(
            kv_groups, axis=1
        )
        v = np.reshape(value, value.shape[0:-2] + (value.shape[-2] * value.shape[-1],)).repeat(
            kv_groups, axis=1
        )
        context_len = k.shape[2]

        mask = self.attention_mask
        _n_mask_leading = mask.ndim - 3
        _mask_batch = mask.shape[0] if _n_mask_leading >= 2 else 1
        _mask_heads = mask.shape[_n_mask_leading - 1] if _n_mask_leading >= 1 else 1
        mask = mask.reshape(_mask_batch, _mask_heads, mask.shape[-3], -1)
        _emu_fill = _mask_fill_value(str(mask.dtype))
        if mask.shape[-1] > context_len:
            mask = mask.copy()
            mask[..., context_len:] = _emu_fill
        if mask.shape[-1] < context_len:
            pad = context_len - mask.shape[-1]
            widths = [(0, 0)] * (mask.ndim - 1) + [(0, pad)]
            mask = np.pad(mask, widths, constant_values=_emu_fill)

        def _vec(buf):
            return np.reshape(buf, (-1, int(buf.shape[-2]) * pword))[0]

        dq_f18 = emu.rf18(_vec(self.attention_scores_dequantizer_scale)[0])
        q_scale_f18 = emu.rf18(_vec(self.attention_weights_quantizer_scale)[0])
        q_zp_f18 = emu.rf18(_vec(self.attention_weights_quantizer_zero_point)[0])
        out_dq_f18 = emu.rf18(_vec(self.attention_output_dequantizer_scale)[0])
        scale_f18 = emu.rf18(scale)

        exp_lut = np.reshape(self.exp_lut, (-1, pword))
        inv_lut = np.reshape(self.inv_lut, (-1, pword))

        def apply_exp(arr):
            if exp_lut.shape[0] == 1:
                return emu.apply_lut(arr, exp_lut)
            if exp_method is ChainedLUTMethod.COMPOSITION:
                out = arr
                for j in range(exp_lut.shape[0]):
                    out = emu.apply_lut(out, exp_lut[j : j + 1])
                return out
            out = emu.apply_lut(arr, exp_lut[0:1])
            for j in range(1, exp_lut.shape[0]):
                out = emu.rf18(out + emu.apply_lut(arr, exp_lut[j : j + 1]))
            return out

        def apply_inv(arr):
            if inv_lut.shape[0] == 1:
                return emu.apply_lut(arr, inv_lut)
            if inv_method is ChainedLUTMethod.COMPOSITION:
                out = arr
                for j in range(inv_lut.shape[0]):
                    out = emu.apply_lut(out, inv_lut[j : j + 1])
                return out
            out = emu.apply_lut(arr, inv_lut[0:1])
            for j in range(1, inv_lut.shape[0]):
                out = emu.rf18(out + emu.apply_lut(arr, inv_lut[j : j + 1]))
            return out

        num_heads = q.shape[1]
        batch_size = q.shape[0]
        out_flat = output.reshape(batch_size, num_heads, q.shape[2], -1)
        for b in range(batch_size):
            for h in range(num_heads):
                q_h = q[b, h]  # (seq, dim)
                k_h = k[b, h]  # (ctx, dim)
                v_h = v[b, h]  # (ctx, dv)
                mask_h = mask[min(b, _mask_batch - 1), h if _mask_heads > 1 else 0]  # (seq, ctx)
                seq = q_h.shape[0]
                dv = v_h.shape[1]

                # Per-row running state as (seq, 1) column vectors (matches the
                # adder-tree reduction's keepdims output); o_run is (seq, dv).
                m_run = None
                l_run = np.zeros((seq, 1), np.float32)
                o_run = np.zeros((seq, dv), np.float32)
                for t in range((context_len + kv_tile_ctx - 1) // kv_tile_ctx):
                    g0 = t * kv_tile_ctx
                    g1 = min(g0 + kv_tile_ctx, context_len)
                    s = emu.rf18(q_h.astype("int32") @ k_h[g0:g1].T.astype("int32"))
                    s = emu.rf18(emu.rf18(emu.rf18(s * dq_f18) * scale_f18))
                    s = emu.rf18(s + emu.rf18(mask_h[:, g0:g1]))  # (seq, tile)
                    tile_max = np.max(s, axis=-1, keepdims=True)  # (seq, 1)
                    if t == 0:
                        m_new = tile_max
                        alpha = None
                    else:
                        m_new = np.maximum(m_run, tile_max)
                        alpha = apply_exp(emu.rf18(m_new - m_run))  # exp(m_old - m_new), (seq,1)
                    p = apply_exp(emu.rf18(m_new - s))  # exp(s - m_new), (seq, tile)
                    s_outer = _reduce_add_f18(p.reshape(p.shape[0:-1] + (-1, PWORD)), axis=(-2,))
                    s_tile = _reduce_adder_tree(s_outer, axis=-1).squeeze(-1)  # (seq, 1)
                    pq = emu.round_clip_cast(emu.rf18(emu.rf18(p * q_scale_f18) + q_zp_f18))
                    contrib = emu.rf18(
                        emu.rf18(pq.astype("int32") @ v_h[g0:g1].astype("int32")) * out_dq_f18
                    )
                    if t == 0:
                        l_run = s_tile
                        o_run = contrib
                    else:
                        l_run = emu.rf18(emu.rf18(alpha * l_run) + s_tile)
                        o_run = emu.rf18(emu.rf18(alpha * o_run) + contrib)
                    m_run = m_new

                inv_l = apply_inv(l_run)  # (seq, 1)
                if use_newton_raphson:
                    two = emu.rf18(2.0)
                    inv_l = emu.rf18(inv_l * emu.rf18(two - emu.rf18(l_run * inv_l)))
                out_flat[b, h] = emu.rf18(o_run * inv_l)
        return None


sdpa_flash = register_op(
    SDPAFlash,
    "llm.sdpa_flash",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)

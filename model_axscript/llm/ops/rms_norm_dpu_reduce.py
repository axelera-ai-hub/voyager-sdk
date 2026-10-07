#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""DPU-reduce RMSNorm variant with D-major lane packing (llm.rms_norm_dpu_reduce).

A single-pass alternative to axol.core.rms_norm_kahan that packs the D
(reduction) axis into the PWORD lanes instead of the N (batch) axis. Per
leading position:

    1. Pop D_blocks input vectors into a register array (held resident).
    2. Per-lane accumulate sum(x^2) across the blocks.
    3. add_reduce -- hardware add-tree across the 64 lanes -> scalar total.
    4. mean_plus_eps = sum_sq * (1/D) + eps.
    5. one_over_rms = chained_lut(mean_plus_eps) (+ optional Newton step).
    6. output[k] = x_buf[k] * one_over_rms * weight[k] for k in 0..D_blocks.

Versus Kahan this needs no NDd<->DNn transpose (works in native NDd order),
is single-pass (input stays in registers, no sum_x_sq L1 round-trip), and
wastes no lanes when N is small (decode, N=1): every lane carries a useful D
position. The trade is no Kahan compensation (a per-block fp18 sum + the
add-tree) and D_blocks extra live registers, so it suits small-D norms
(QK-norm) better than large hidden norms.

D-major weight. Unlike Kahan's DNn lanes (which share one D-weight across
the N lanes), here each lane is a distinct D position, so the weight must be
blocked exactly like the input: weight.reshape(D_blocks, PWORD) -- row k,
lane j = weight[k*64 + j] (no broadcast). The kernel holds the D_blocks
weight vectors in registers (loaded once in init; weight is invariant across N)
and multiplies them block-for-block, lane-aligned with the input.  Pass the
weight pre-packed this way (the natural reshape(-1, 64) blocking, not
broadcast_to_pword).

Dispatch. When input+output fit in ~half of L1 the kernel runs as one
Datapath over the full N grid (no L1 staging, init once -- lowest overhead).
Otherwise it tiles N_dims[1] (the axis after batch), keeping the inner N
axes whole per tile, so the Datapath count is B x ceil(N_dims[1]/tile)
rather than prod(N_dims[:-1]) -- it does not blow up per-head.
"""
from contextlib import nullcontext
import math

from axelera.compiler.alto.axol import atomic
from axelera.compiler.alto.axol.common import buffer_slice_with_defaults
from axelera.compiler.alto.axol.utils import to_int_tuple, to_tuple
from axelera.compiler.alto.axol.utils.lut_chain import ChainedLUTMethod, chained_lut_dpu
from axelera.compiler.alto.compiler.config import HardwareGeneration
from axelera.compiler.alto.compiler.device import MEMORY_SIZES, PWORD, MemoryScope
from axelera.compiler.alto.compiler.dtypes import MetisTypes
from axelera.compiler.alto.compiler.interpreter import emulation as emu
from axelera.compiler.alto.compiler.op_registry import Arg, BufferInfo, ScriptOp, register_op
import axelera.compiler.alto.script as A
from axelera.compiler.alto.utils import parse_vector_dtype
import numpy as np
from tvm import tir

# mypy: disable-error-code="attr-defined, index, arg-type"
# flake8: noqa: C901

# Fraction of L1 the single-dispatch path may consume for input+output combined.
# EnforceMemoryConstraints starts complaining as total data nears L1; 0.55 keeps
# a safety cushion. Cells beyond that fall through to the tiled path.
_SINGLE_DISPATCH_RATIO = 0.55


def _vec_bytes(dtype: str) -> int:
    scalar, _ = parse_vector_dtype(dtype)
    return PWORD * np.dtype(scalar).itemsize


def _data_bytes(N_dims, D_blocks, input_dtype, output_dtype) -> int:
    """Bytes for input + output buffers across all N_dims."""
    return math.prod(N_dims) * D_blocks * (_vec_bytes(input_dtype) + _vec_bytes(output_dtype))


def _compute_tile_size(
    tile_axis_size,
    inner_n_count,
    D_blocks,
    input_dtype,
    output_dtype,
    num_consts,
    generation,
    extra_l1_bytes=0,
):
    """Return (tile_size, num_full_tiles, remainder) for the tile axis.

    Each tile spans tile_size x inner_n_count N positions; mem per tile-axis
    unit = inner_n_count x D_blocks x (in+out vec). extra_l1_bytes reserves L1
    for whole-buffer residents the tile must coexist with (e.g. the fused-RoPE
    trig table, which spans the full seq regardless of the tile).
    """
    mem_per_unit = inner_n_count * D_blocks * (_vec_bytes(input_dtype) + _vec_bytes(output_dtype))
    fixed_overhead = num_consts * PWORD * 4 + extra_l1_bytes
    l1_budget = MEMORY_SIZES[generation][MemoryScope.L1] - fixed_overhead
    if mem_per_unit <= 0 or l1_budget <= 0:
        raise ValueError(
            f"dpu_reduce: invalid memory budget (inner_n_count={inner_n_count}, "
            f"D_blocks={D_blocks}, l1_budget={l1_budget}, mem_per_unit={mem_per_unit})"
        )
    max_tile = l1_budget // mem_per_unit
    if max_tile <= 0:
        raise ValueError(
            f"dpu_reduce: a single tile-axis unit ({inner_n_count} inner N x {D_blocks} D blocks) "
            f"exceeds the L1 budget ({l1_budget} bytes); inner N axes too large."
        )
    tile_size = min(tile_axis_size, max_tile)
    return tile_size, tile_axis_size // tile_size, tile_axis_size % tile_size


# init_consts rows after the LUT: 1/D, then eps.
_NUM_SCALAR_CONSTS = 2


def pack_norm_consts(lut, normalized_shape: tuple[int, ...], eps: float):
    """The op's init_consts buffer: LUT rows, then 1/D, then eps, PWORD-broadcast."""
    import torch

    lut = torch.as_tensor(np.asarray(lut), dtype=torch.float32).reshape(-1, PWORD)
    scalars = torch.tensor(
        [[1.0 / math.prod(normalized_shape)] * PWORD, [eps] * PWORD], dtype=torch.float32
    )
    return torch.cat([lut, scalars]).contiguous()


class RMSNormDPUReduce(ScriptOp):
    """DPU add-reduce RMS normalization with D-major lane packing.

    Mirrors axol.core.rms_norm_kahan's operands so call sites can swap,
    except weight must be D-major packed (reshape(D_blocks, PWORD)),
    not broadcast-PWORD. See module docstring.
    """

    input = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="Input, layout (*N_dims, *D_blocks), int8x64/float16x64/float32x64.",
    )
    init_consts = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="The three constants the DPU pops once in init, as ONE "
        "(n_luts + 2,) float32x64 buffer: the chained inv-sqrt LUT rows, then 1/D, then "
        "eps. IFD1 is a single FIFO and these are already streamed back to back, so one "
        "operand streams them identically while issuing ONE DMA where three buffers issue "
        "three. Build it with ``pack_norm_consts``, which is what fixes the order.",
    )

    normalized_shape = Arg(
        tuple[int, ...],
        kw=True,
        description="Logical shape of the reduction axis (product must be a multiple of PWORD).",
    )
    weight_buffer = Arg(
        MetisTypes[tir.Buffer, "..."] | None,
        default=None,
        kw=True,
        description="D-major packed affine weight, (D_blocks,); row k lane j = weight[k*64+j].",
    )
    quant_params = Arg(
        MetisTypes[tir.Buffer, "..."] | None,
        default=None,
        kw=True,
        description="Optional INTERLEAVED per-channel quantize params, (2*D_blocks,) D-major: "
        "[scale[0], zp[0], scale[1], zp[1], ...]. If set, the normalized output is quantized "
        "(out * scale + zp) and rounded/clipped to int8 in-register -- fusing the post-norm "
        "activation quantize. Output must then be int8x64. Pre-interleaved by the layer binding "
        "(NOT on device: a strided DMA does not lower correctly), streamed linearly + popped as "
        "(scale, zp) per block -- same contract as quantized_matmul_activation's dequant params. "
        "With rope_trig the params are instead held in registers, since IFD1 carries only one "
        "per-row operand (see quant_resident).",
    )
    rope_trig = Arg(
        MetisTypes[tir.Buffer, "..."] | None,
        default=None,
        kw=True,
        description="Optional half-rotation RoPE applied to the normalized row BEFORE the quantize "
        "(fuses norm->rope->quantize for the q/k head tensors). Shape (1, seq, D_blocks): per "
        "position, [:D_blocks//2] = sin, [D_blocks//2:] = cos. Streamed per row indexed by the LAST "
        "N (position) axis; the reduction (head_dim) must split into a sin/cos half (D_blocks even). "
        "The full-seq trig is staged in L1 once and its footprint is reserved in the tile-size "
        "budget (extra_l1_bytes), so it works on both the single-dispatch and tiled paths. NB the "
        "DPU keeps fp18 through norm->rope->quant in registers (vs the unfused chain widening to "
        "fp32 between ops), so device output can differ from the chain by a few int8 levels -- "
        "interpret models both identically (bit-exact); confirm real-data impact via perplexity.",
    )
    newton_const = Arg(
        MetisTypes[tir.Buffer, "..."] | None,
        default=None,
        kw=True,
        description="Optional Newton-Raphson constants (2,) float32x64: [3.0, 0.5].",
    )
    invsqrt_method = Arg(
        str,
        default="additive",
        kw=True,
        description="Chained-LUT method; only 'additive' is supported.",
    )
    output = Arg(
        MetisTypes[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="""
        Output buffer specification. Can be:
            - BufferInfo to override dtype/scope
            - tir.Buffer for a pre-allocated output
            - None to auto-infer (shape=input.shape, dtype=input.dtype,
              scope=input.scope())""",
    )
    generation = Arg(
        HardwareGeneration | None,
        default=None,
        kw=True,
        description="Optional hardware generation.",
    )
    force_tiled = Arg(
        bool,
        default=False,
        kw=True,
        description="Take the L1-tiled path even when the whole input fits one dispatch (a "
        "band of a row-sharded op).",
    )

    # ------------------------------------------------------------------ #
    # Layout helpers                                                       #
    # ------------------------------------------------------------------ #
    def _dims(self):
        """(N_dims, D_outer_dims, D_blocks) from the (vectorized) input shape."""
        normalized_shape = to_int_tuple(self.normalized_shape)
        input_shape = to_int_tuple(self.input.shape)
        nd = len(normalized_shape)
        N_dims = input_shape[:-nd] if nd < len(input_shape) else ()
        D_outer_dims = input_shape[len(N_dims) :]
        D_blocks = math.prod(D_outer_dims) if D_outer_dims else 1
        return N_dims, D_outer_dims, D_blocks

    def validate(self) -> None:
        if self.invsqrt_method != "additive":
            raise ValueError(
                f"rms_norm_dpu_reduce only supports invsqrt_method='additive', "
                f"got {self.invsqrt_method!r}"
            )
        normalized_shape = to_int_tuple(self.normalized_shape)
        if len(normalized_shape) <= 0:
            raise ValueError("normalized_shape must have at least one dimension.")
        d = math.prod(normalized_shape)
        if d % PWORD != 0:
            raise ValueError(
                f"rms_norm_dpu_reduce requires prod(normalized_shape) ({d}) to be a "
                f"multiple of PWORD ({PWORD})."
            )
        n_rows = int(math.prod(int(s) for s in self.init_consts.shape))
        if n_rows <= _NUM_SCALAR_CONSTS:
            raise ValueError(
                f"init_consts needs at least one LUT row before its 1/D and eps rows; got "
                f"{n_rows} rows."
            )
        N_dims, _, D_blocks = self._dims()
        if not N_dims:
            raise NotImplementedError(
                "rms_norm_dpu_reduce requires at least one N (leading) axis; got input "
                f"shape {to_int_tuple(self.input.shape)} with normalized_shape {normalized_shape}."
            )
        if self.weight_buffer is not None:
            n_blocks = math.prod(int(s) for s in self.weight_buffer.shape)
            if n_blocks != D_blocks:
                raise ValueError(
                    f"D-major weight_buffer must have {D_blocks} PWORD blocks (one per input D "
                    f"block), got shape {tuple(self.weight_buffer.shape)} ({n_blocks} blocks)."
                )
        if self.quant_params is not None:
            n_qp = math.prod(int(s) for s in self.quant_params.shape)
            if n_qp != 2 * D_blocks:
                raise ValueError(
                    f"quant_params must have 2*{D_blocks} = {2 * D_blocks} PWORD blocks "
                    f"(interleaved scale,zp per D block), got shape {tuple(self.quant_params.shape)}."
                )
            if self.output is not None:
                out_dtype = (
                    self.output.dtype
                    if hasattr(self.output, "dtype")
                    else self.output.get("dtype")
                )
                if out_dtype not in ("int8x64", None):
                    raise ValueError(f"fused quantize requires int8x64 output, got {out_dtype}.")

    def infer_outputs(self) -> dict[str, BufferInfo]:
        # Fused quantize emits int8; otherwise the output matches the input dtype.
        dtype = "int8x64" if self.quant_params is not None else self.input.dtype
        return {
            "output": BufferInfo(
                shape=tuple(self.input.shape),
                dtype=dtype,
                scope=self.input.scope(),
            )
        }

    # ------------------------------------------------------------------ #
    def implementation(self) -> None:
        generation = self.generation or HardwareGeneration.OMEGA
        input = self.input
        output_buffer = self.output
        init_consts = self.init_consts
        weight_buffer = self.weight_buffer
        newton_const = self.newton_const
        has_weight = weight_buffer is not None
        quant_params = self.quant_params
        has_quant = quant_params is not None
        rope_trig = self.rope_trig
        has_rope = rope_trig is not None
        # IFD1 is one FIFO and RoPE already streams trig per row, so the quantize params must
        # instead be popped once into registers; viable only because RoPE fixes D_blocks.
        quant_resident = has_quant and has_rope

        N_dims, D_outer_dims, D_blocks = self._dims()
        rope_half = D_blocks // 2  # sin/cos half; rope mixes block k with block k+rope_half

        n_luts = int(init_consts.shape[0]) - _NUM_SCALAR_CONSTS
        num_consts = (
            int(init_consts.shape[0])
            + (int(newton_const.shape[0]) if newton_const is not None else 0)
            + (D_blocks if has_weight else 0)
            + (2 * D_blocks if has_quant else 0)
        )

        consts_dtype = "float32x64"
        input_dtype = "int32x64" if input.dtype == "int8x64" else input.dtype
        output_dtype = str(output_buffer.dtype)

        # Stage the (small, position-indexed) RoPE trig in L1 once so the per-row
        # streaming reads from L1 rather than re-indexing the DDR buffer across the
        # tiled grid (which the memory-constraint pass can't tile). The trig spans the
        # FULL seq regardless of tiling, so its L1 footprint must be reserved up front.
        trig_l1 = atomic.copy(rope_trig, scope=A.L1) if has_rope else None
        trig_l1_bytes = (
            int(math.prod(int(s) for s in rope_trig.shape)) * _vec_bytes(rope_trig.dtype)
            if has_rope
            else 0
        )

        l1_size = MEMORY_SIZES[generation][MemoryScope.L1]
        fits_in_one_dispatch = not bool(self.force_tiled) and (
            _data_bytes(N_dims, D_blocks, input_dtype, output_dtype)
            <= int(l1_size * _SINGLE_DISPATCH_RATIO) - trig_l1_bytes
        )

        # Per-Datapath kernel. in_source/out_dest are buffer-likes
        # (DDR-resident for single-dispatch, L1 tiles otherwise);
        # n_grid_extents is the N grid for this dispatch.
        def _run_kernel(in_source, out_dest, n_grid_extents):
            with A.Datapath():
                with (
                    A.decl_stream(input_dtype) as input_stream,
                    A.decl_stream(consts_dtype) as init_stream,
                    (
                        A.decl_stream(consts_dtype) if newton_const is not None else nullcontext()
                    ) as newton_stream,
                    (
                        A.decl_stream(weight_buffer.dtype) if has_weight else nullcontext()
                    ) as weight_stream,
                    A.decl_stream(quant_params.dtype) if has_quant else nullcontext() as qp_stream,
                    A.decl_stream(rope_trig.dtype) if has_rope else nullcontext() as trig_stream,
                    A.decl_stream(output_buffer.dtype) as output_stream,
                ):
                    # IFD0: input vectors in NDd order (outer N, inner D).
                    with A.grid(*n_grid_extents) as n_loop_vars:
                        n_loop_vars = to_tuple(n_loop_vars)
                        with A.grid(*D_outer_dims) as d_loop_vars:
                            d_loop_vars = to_tuple(d_loop_vars)
                            with A.IFD(num=0):
                                A.stream_push(input_stream, in_source[n_loop_vars + d_loop_vars])

                    # IFD1: constants -- LUT params, 1/D, eps, optional Newton, weight.
                    # The first three arrive packed, so they cost one DMA rather than three.
                    with A.grid(*init_consts.shape) as i:
                        with A.IFD(num=1):
                            A.stream_push(init_stream, init_consts[i])
                    if newton_const is not None:
                        with A.grid(*newton_const.shape) as i:
                            with A.IFD(num=1):
                                A.stream_push(newton_stream, newton_const[i])
                    if has_weight:
                        # D-major weight: one vector per D block, distinct per lane,
                        # invariant across N; pushed once, popped once in init.
                        with A.grid(*weight_buffer.shape) as i:
                            with A.IFD(num=1):
                                A.stream_push(weight_stream, weight_buffer[i])
                    if has_quant:
                        # Pre-interleaved [scale[0], zp[0], ...] on ONE IFD1 stream: two separate
                        # streams or an on-device strided interleave both misalign on the device.
                        if quant_resident:
                            with A.grid(*quant_params.shape) as k:
                                with A.IFD(num=1):
                                    A.stream_push(qp_stream, quant_params[k])
                        else:
                            # Kept out of registers: a 2*D_blocks array overflows the DPU file on
                            # the hidden norm, so the small param buffer is re-read per row.
                            with A.grid(*n_grid_extents):
                                with A.grid(*quant_params.shape) as k:
                                    with A.IFD(num=1):
                                        A.stream_push(qp_stream, quant_params[k])
                    if has_rope:
                        # Per-row RoPE trig, indexed by POSITION (the last N axis): for each
                        # N-row push its (sin || cos) blocks. rope_trig is (1, seq, D_blocks);
                        # trig[0, pos, :] -- re-read per head (broadcast across heads).
                        with A.grid(*n_grid_extents) as n_loop_vars:
                            n_loop_vars = to_tuple(n_loop_vars)
                            pos = n_loop_vars[-1]
                            with A.grid(D_blocks) as kk:
                                with A.IFD(num=1):
                                    A.stream_push(trig_stream, trig_l1[0, pos, kk])

                    with A.grid(*n_grid_extents):
                        with A.DPU():
                            # Resident input row (D_blocks slots); reused for output.
                            x_buf = A.array("float18x64", size=D_blocks)
                            mean_plus_eps = A.array("float18x64")
                            one_over_rms = A.array("float18x64")
                            if newton_const is not None:
                                invsqrt_pre = A.array("float18x64")

                            lut_regs = A.array("float18x64", size=n_luts)
                            one_over_d_reg = A.array("float18x64", size=1)
                            eps_reg = A.array("float18x64", size=1)
                            if newton_const is not None:
                                newton_regs = A.array("float18x64", size=2)
                            if has_weight:
                                weight_regs = A.array("float18x64", size=D_blocks)
                            if quant_resident:
                                qp_regs = A.array("float18x64", size=2 * D_blocks)
                            if has_rope:
                                # Normalized row held resident so RoPE can mix the two
                                # head_dim halves (block k with block k+rope_half) before quant.
                                norm_buf = A.array("float18x64", size=D_blocks)
                                sin_r = A.array("float18x64", size=rope_half)
                                cos_r = A.array("float18x64", size=rope_half)

                            sum_sq = A.var("float18x64")
                            sq_tmp = A.var("float18x64")
                            out_tmp = A.var("float18x64")
                            newton_tmp = A.var("float18x64")
                            lut_acc = A.var("float18x64")
                            # Streamed quantize params (one scalar each, reused per block).
                            qs_tmp = A.var("float18x64")
                            qz_tmp = A.var("float18x64")
                            rope_t1 = A.var("float18x64")
                            rope_t2 = A.var("float18x64")

                            with A.init():
                                with A.unroll(0, n_luts) as k:
                                    A.buffer_store(
                                        lut_regs, A.cast(init_stream.pop(), "float18x64"), (k,)
                                    )
                                A.buffer_store(
                                    one_over_d_reg,
                                    A.cast(init_stream.pop(), "float18x64"),
                                    (0,),
                                )
                                A.buffer_store(
                                    eps_reg, A.cast(init_stream.pop(), "float18x64"), (0,)
                                )
                                if newton_const is not None:
                                    with A.unroll(0, 2) as i:
                                        A.buffer_store(
                                            newton_regs,
                                            A.cast(newton_stream.pop(), "float18x64"),
                                            (i,),
                                        )
                                if has_weight:
                                    with A.unroll(0, D_blocks) as k:
                                        A.buffer_store(
                                            weight_regs,
                                            A.cast(weight_stream.pop(), "float18x64"),
                                            (k,),
                                        )
                                if quant_resident:
                                    with A.unroll(0, 2 * D_blocks) as k:
                                        A.buffer_store(
                                            qp_regs, A.cast(qp_stream.pop(), "float18x64"), (k,)
                                        )

                            # Per-N body
                            with A.unroll(0, D_blocks) as k:
                                A.buffer_store(
                                    x_buf, A.cast(input_stream.pop(), "float18x64"), (k,)
                                )

                            A.buffer_store(sum_sq, x_buf[0] * x_buf[0], (0,))
                            if D_blocks > 1:
                                with A.unroll(1, D_blocks) as k:
                                    A.buffer_store(sq_tmp, x_buf[k] * x_buf[k], (0,))
                                    A.buffer_store(sum_sq, sum_sq[0] + sq_tmp[0], (0,))

                            A.buffer_store(sum_sq, A.dpu.ops.add_reduce(sum_sq[0]), (0,))

                            A.buffer_store(mean_plus_eps, sum_sq[0] * one_over_d_reg[0], (0,))
                            A.buffer_store(mean_plus_eps, mean_plus_eps[0] + eps_reg[0], (0,))

                            invsqrt_dst = invsqrt_pre if newton_const is not None else one_over_rms
                            A.buffer_store(
                                invsqrt_dst,
                                chained_lut_dpu(
                                    value=mean_plus_eps[0],
                                    lut_buf=lut_regs,
                                    n_luts=n_luts,
                                    method=ChainedLUTMethod.ADDITIVE,
                                    tmp=newton_tmp,
                                    acc=lut_acc,
                                ),
                                (0,),
                            )

                            if newton_const is not None:
                                A.buffer_store(newton_tmp, invsqrt_pre[0] * invsqrt_pre[0], (0,))
                                A.buffer_store(newton_tmp, mean_plus_eps[0] * newton_tmp[0], (0,))
                                A.buffer_store(newton_tmp, newton_regs[0] - newton_tmp[0], (0,))
                                A.buffer_store(invsqrt_pre, invsqrt_pre[0] * newton_tmp[0], (0,))
                                A.buffer_store(one_over_rms, invsqrt_pre[0] * newton_regs[1], (0,))

                            if has_rope:
                                # 1. normalized row -> norm_buf (x * inv_rms * weight).
                                with A.unroll(0, D_blocks) as k:
                                    A.buffer_store(norm_buf, x_buf[k] * one_over_rms[0], (k,))
                                    if has_weight:
                                        A.buffer_store(
                                            norm_buf, norm_buf[k] * weight_regs[k], (k,)
                                        )
                                # 2. this row's trig: sin (rope_half blocks) then cos.
                                with A.unroll(0, rope_half) as k:
                                    A.buffer_store(
                                        sin_r, A.cast(trig_stream.pop(), "float18x64"), (k,)
                                    )
                                with A.unroll(0, rope_half) as k:
                                    A.buffer_store(
                                        cos_r, A.cast(trig_stream.pop(), "float18x64"), (k,)
                                    )
                                # 3. half-rotation RoPE in place: y_k = n_k*cos - n_{k+h}*sin,
                                #    y_{k+h} = n_{k+h}*cos + n_k*sin. Products go through temps
                                #    (the DPU lowers a*b then -/+ , not a compound a*b - c*d);
                                #    read both halves before storing back.
                                with A.unroll(0, rope_half) as k:
                                    A.buffer_store(out_tmp, norm_buf[k] * cos_r[k], (0,))
                                    A.buffer_store(
                                        rope_t1, norm_buf[k + rope_half] * sin_r[k], (0,)
                                    )
                                    A.buffer_store(rope_t2, out_tmp[0] - rope_t1[0], (0,))  # y_k
                                    A.buffer_store(
                                        out_tmp, norm_buf[k + rope_half] * cos_r[k], (0,)
                                    )
                                    A.buffer_store(rope_t1, norm_buf[k] * sin_r[k], (0,))
                                    A.buffer_store(
                                        norm_buf, out_tmp[0] + rope_t1[0], (k + rope_half,)
                                    )  # y_{k+h}
                                    A.buffer_store(norm_buf, rope_t2[0], (k,))
                                # 4. per-channel quantize + int8 cast, in block order.
                                with A.unroll(0, D_blocks) as k:
                                    A.buffer_store(out_tmp, norm_buf[k], (0,))
                                    if has_quant:
                                        A.buffer_store(out_tmp, out_tmp[0] * qp_regs[2 * k], (0,))
                                        A.buffer_store(
                                            out_tmp, out_tmp[0] + qp_regs[2 * k + 1], (0,)
                                        )
                                    A.stream_push(
                                        output_stream, A.cast(out_tmp[0], output_buffer.dtype)
                                    )
                            else:
                                with A.unroll(0, D_blocks) as k:
                                    A.buffer_store(out_tmp, x_buf[k] * one_over_rms[0], (0,))
                                    if has_weight:
                                        # lane j of block k == D-index k*64+j, lane-aligned.
                                        A.buffer_store(out_tmp, out_tmp[0] * weight_regs[k], (0,))
                                    if has_quant:
                                        # Fused per-channel quantize: out * scale + zero_point,
                                        # params streamed (popped) per block; the int8 cast below
                                        # rounds + clips to [-128, 127].
                                        A.buffer_store(
                                            qs_tmp, A.cast(qp_stream.pop(), "float18x64"), (0,)
                                        )
                                        A.buffer_store(
                                            qz_tmp, A.cast(qp_stream.pop(), "float18x64"), (0,)
                                        )
                                        A.buffer_store(out_tmp, out_tmp[0] * qs_tmp[0], (0,))
                                        A.buffer_store(out_tmp, out_tmp[0] + qz_tmp[0], (0,))
                                    A.stream_push(
                                        output_stream, A.cast(out_tmp[0], output_buffer.dtype)
                                    )

                    # ODR: drain to the destination buffer.
                    with A.grid(*n_grid_extents) as n_loop_vars:
                        n_loop_vars = to_tuple(n_loop_vars)
                        with A.grid(*D_outer_dims) as d_loop_vars:
                            d_loop_vars = to_tuple(d_loop_vars)
                            with A.ODR():
                                A.buffer_store(
                                    out_dest.buffer,
                                    output_stream.pop(),
                                    out_dest.sliced_indices(n_loop_vars + d_loop_vars),
                                )

        # Single-dispatch path: fits in ~half L1, no L1 staging.
        if fits_in_one_dispatch:
            _run_kernel(input, output_buffer, N_dims)
            return

        # Tiled path: tile N_dims[1] (the axis after batch); keep inner N
        # axes whole per tile.  Walk outward if the inner-N slab doesn't fit,
        # until _compute_tile_size succeeds.
        def _try_tile_axis(idx):
            leading = N_dims[:idx]
            inner = N_dims[idx + 1 :]
            inner_count = math.prod(inner) if inner else 1
            try:
                ts, nft, rem = _compute_tile_size(
                    N_dims[idx],
                    inner_count,
                    D_blocks,
                    input_dtype,
                    output_dtype,
                    num_consts,
                    generation,
                    extra_l1_bytes=trig_l1_bytes,
                )
            except ValueError:
                return None
            return leading, inner, ts, nft, rem

        chosen = None
        if len(N_dims) >= 2:
            for tile_idx in range(1, len(N_dims)):
                chosen = _try_tile_axis(tile_idx)
                if chosen is not None:
                    break

        if chosen is None:
            leading_N_dims = ()
            inner_N_dims = ()
            tile_size, num_full_tiles, remainder = _compute_tile_size(
                N_dims[0],
                1,
                D_blocks,
                input_dtype,
                output_dtype,
                num_consts,
                generation,
                extra_l1_bytes=trig_l1_bytes,
            )
        else:
            leading_N_dims, inner_N_dims, tile_size, num_full_tiles, remainder = chosen

        def _process_tile(leading_indices, tile_start, tile_count):
            tile_idx = (
                leading_indices
                + (slice(tile_start, tile_start + tile_count),)
                + (slice(None),) * len(inner_N_dims)
                + (slice(None),) * len(D_outer_dims)
            )
            input_tile_l1 = atomic.copy(input[tile_idx], scope=A.L1)
            output_tile_l1, _ = buffer_slice_with_defaults(
                None,
                shape=(tile_count,) + inner_N_dims + D_outer_dims,
                dtype=output_buffer.dtype,
                scope=A.L1,
            )
            _run_kernel(input_tile_l1, output_tile_l1, (tile_count,) + inner_N_dims)
            atomic.copy(output_tile_l1, output=output_buffer[tile_idx])

        def _dispatch_tiles(leading_indices):
            if num_full_tiles > 0:
                with A.grid(num_full_tiles) as t:
                    _process_tile(leading_indices, t * tile_size, tile_size)
            if remainder > 0:
                _process_tile(leading_indices, num_full_tiles * tile_size, remainder)

        if leading_N_dims:
            with A.grid(*leading_N_dims) as leading_vars:
                _dispatch_tiles(to_tuple(leading_vars))
        else:
            _dispatch_tiles(())

    # ------------------------------------------------------------------ #

    def emulate(self) -> None:  # type: ignore[override]
        """NumPy reference: float18-rounded, chained additive inv-sqrt LUT, D-major weight."""
        normalized_shape = to_int_tuple(self.normalized_shape)
        d = math.prod(normalized_shape)
        out = self.output
        out_dtype = out.dtype

        # Flatten to (N_total, D): the trailing (D_blocks, 64) pack contiguously to D.
        x = emu.rf18(self.input.astype(np.float32).reshape(-1, d))

        sum_sq = emu.rf18(x * x).sum(axis=-1, keepdims=True)
        mean_plus_eps = emu.rf18(sum_sq * emu.rf18(float(self.init_consts[-2, 0])))
        mean_plus_eps = emu.rf18(mean_plus_eps + emu.rf18(float(self.init_consts[-1, 0])))

        lut_2d = self.init_consts[:-_NUM_SCALAR_CONSTS].reshape(-1, PWORD)
        one_over_rms = emu.apply_lut(mean_plus_eps, lut_2d[0:1])
        for k in range(1, lut_2d.shape[0]):
            one_over_rms = emu.rf18(one_over_rms + emu.apply_lut(mean_plus_eps, lut_2d[k : k + 1]))

        if self.newton_const is not None:
            three = emu.rf18(float(self.newton_const[0, 0]))
            half = emu.rf18(float(self.newton_const[1, 0]))
            tmp = emu.rf18(one_over_rms * one_over_rms)
            tmp = emu.rf18(mean_plus_eps * tmp)
            tmp = emu.rf18(three - tmp)
            one_over_rms = emu.rf18(one_over_rms * tmp)
            one_over_rms = emu.rf18(one_over_rms * half)

        normalized = emu.rf18(x * one_over_rms)

        if self.weight_buffer is not None:
            # D-major: weight packs (D_blocks, 64) -> D, one distinct value per lane.
            w = emu.rf18(self.weight_buffer.astype(np.float32).reshape(-1))
            normalized = emu.rf18(normalized * w[None, :])

        if self.rope_trig is not None:
            # Half-rotation RoPE, per-row by position (last N axis -> row r maps to pos
            # r % seq for the (heads, seq) layout). trig (1, seq, D): [:d/2]=sin, [d/2:]=cos.
            seq = int(self.rope_trig.shape[1])
            trig = self.rope_trig.astype(np.float32).reshape(seq, d)
            half = d // 2
            pos = np.arange(normalized.shape[0]) % seq
            sin = emu.rf18(trig[:, :half][pos])
            cos = emu.rf18(trig[:, half:][pos])
            x1, x2 = normalized[:, :half], normalized[:, half:]
            y1 = emu.rf18(emu.rf18(x1 * cos) - emu.rf18(x2 * sin))
            y2 = emu.rf18(emu.rf18(x2 * cos) + emu.rf18(x1 * sin))
            normalized = np.concatenate([y1, y2], axis=-1)

        if self.quant_params is not None:
            # Fused per-channel quantize: out * scale + zp. quant_params is interleaved
            # [scale[0], zp[0], ...] D-major (lane = D-index); de-interleave for the reference.
            qp = self.quant_params.astype(np.float32).reshape(2 * (d // PWORD), PWORD)
            qs = emu.rf18(qp[0::2].reshape(-1))  # even = scale
            qz = emu.rf18(qp[1::2].reshape(-1))  # odd = zero_point
            normalized = emu.rf18(normalized * qs[None, :])
            normalized = emu.rf18(normalized + qz[None, :])

        normalized = normalized.reshape(self.input.shape)
        if np.dtype(out_dtype) == np.int8:
            normalized = np.clip(np.round(normalized), -128, 127)
        normalized = normalized.astype(out_dtype)
        if np.issubdtype(np.dtype(out_dtype), np.floating):
            finfo = np.finfo(out_dtype)
            normalized[(normalized != 0) & (np.abs(normalized) < finfo.smallest_normal)] = 0

        np.copyto(out, normalized)


rms_norm_dpu_reduce = register_op(
    RMSNormDPUReduce,
    "llm.rms_norm_dpu_reduce",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)

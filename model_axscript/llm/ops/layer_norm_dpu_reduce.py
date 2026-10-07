#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""DPU-reduce LayerNorm variant with D-major lane packing (llm.layer_norm_dpu_reduce).

The LayerNorm counterpart of llm.rms_norm_dpu_reduce (see that op for the D-major / lane-packing
rationale and tiling). Single-pass over the D (reduction) axis packed into the 64 PWORD lanes; per
leading position, in one DPU datapath:

    1. Pop D_blocks input vectors into resident registers (x_buf).
    2. Per-lane accumulate sum(x) AND sum(x^2) across the blocks.
    3. add_reduce (hardware add-tree over the 64 lanes) -> scalar sum_x, sum_sq.
    4. mean = sum_x / D ; var = sum_sq / D - mean^2 + eps   (one-pass variance).
    5. inv_std = chained_lut(var)  (the inv_sqrt LUT; + optional Newton step).
    6. out[k] = (x_buf[k] - mean) * inv_std * weight[k] + bias[k], optionally fused-quantized to int8.

Versus axol.core.layer_norm_two_pass_kahan this is single-pass (no mean pass + variance pass, no
NDd<->DNn transpose, no sum-of-squares L1 round-trip) and can FUSE the post-norm activation quantize
(quant_params) into the same datapath -- so it replaces both the norm and the following llm.quantize.
The trade is the one-pass E[x^2]-mean^2 variance (no Kahan compensation) and 3*D_blocks live registers
(x_buf + weight + bias), so it suits small/medium hidden (validate numerics before trusting it).

weight AND bias are D-major packed (reshape(D_blocks, PWORD); row k lane j = param[k*64+j]), NOT
broadcast_to_pword -- each lane is a distinct D position, lane-aligned with the input (same as the
RMSNorm op's weight).
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

_SINGLE_DISPATCH_RATIO = 0.55


def _vec_bytes(dtype: str) -> int:
    scalar, _ = parse_vector_dtype(dtype)
    return PWORD * np.dtype(scalar).itemsize


def _data_bytes(N_dims, D_blocks, input_dtype, output_dtype) -> int:
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
    mem_per_unit = inner_n_count * D_blocks * (_vec_bytes(input_dtype) + _vec_bytes(output_dtype))
    fixed_overhead = num_consts * PWORD * 4 + extra_l1_bytes
    l1_budget = MEMORY_SIZES[generation][MemoryScope.L1] - fixed_overhead
    if mem_per_unit <= 0 or l1_budget <= 0:
        raise ValueError(
            f"dpu_reduce: invalid memory budget (mem_per_unit={mem_per_unit}, "
            f"l1_budget={l1_budget})"
        )
    max_tile = l1_budget // mem_per_unit
    if max_tile <= 0:
        raise ValueError("dpu_reduce: a single tile-axis unit exceeds the L1 budget.")
    tile_size = min(tile_axis_size, max_tile)
    return tile_size, tile_axis_size // tile_size, tile_axis_size % tile_size


class LayerNormDPUReduce(ScriptOp):
    """DPU add-reduce LayerNorm with D-major lane packing. See module docstring."""

    input = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="Input, layout (*N_dims, *D_blocks), int8x64/float16x64/float32x64.",
    )
    invsqrt_lut = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="Chained inv-sqrt LUT params, (n_luts,) float32x64.",
    )
    eps_const = Arg(
        MetisTypes[tir.Buffer, "..."], description="Variance epsilon as a (1,) float32x64 buffer."
    )

    normalized_shape = Arg(
        tuple[int, ...], kw=True, description="Reduction-axis shape (product a multiple of PWORD)."
    )
    weight_buffer = Arg(
        MetisTypes[tir.Buffer, "..."] | None,
        default=None,
        kw=True,
        description="D-major packed affine weight (gamma), (D_blocks,).",
    )
    bias_buffer = Arg(
        MetisTypes[tir.Buffer, "..."] | None,
        default=None,
        kw=True,
        description="D-major packed affine bias (beta), (D_blocks,). Added after scale.",
    )
    quant_params = Arg(
        MetisTypes[tir.Buffer, "..."] | None,
        default=None,
        kw=True,
        description="Optional INTERLEAVED per-channel quantize params, (2*D_blocks,) D-major: "
        "[scale[0], zp[0], ...]. If set, the normalized output is quantized (out*scale + zp) and "
        "rounded/clipped to int8 in-register (fuses the post-norm activation quantize). Output must "
        "then be int8x64. Same contract as rms_norm_dpu_reduce.quant_params.",
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
        description="Output buffer / BufferInfo / None (auto: input shape+dtype+scope).",
    )
    generation = Arg(
        HardwareGeneration | None,
        default=None,
        kw=True,
        description="Optional hardware generation.",
    )
    core = Arg(
        int | None,
        default=None,
        kw=True,
        description="Optional AI core to pin the datapath to (for M-sharding across cores; "
        "rows are independent so no cross-core reduction). None = default/unpinned.",
    )

    # ------------------------------------------------------------------ #
    def _dims(self):
        normalized_shape = to_int_tuple(self.normalized_shape)
        input_shape = to_int_tuple(self.input.shape)
        nd = len(normalized_shape)
        N_dims = input_shape[:-nd] if nd < len(input_shape) else ()
        D_outer_dims = input_shape[len(N_dims) :]
        D_blocks = math.prod(D_outer_dims) if D_outer_dims else 1
        return N_dims, D_outer_dims, D_blocks

    def validate(self) -> None:
        if self.invsqrt_method != "additive":
            raise ValueError("layer_norm_dpu_reduce only supports invsqrt_method='additive'.")
        normalized_shape = to_int_tuple(self.normalized_shape)
        if len(normalized_shape) <= 0:
            raise ValueError("normalized_shape must have at least one dimension.")
        d = math.prod(normalized_shape)
        if d % PWORD != 0:
            raise ValueError(
                f"prod(normalized_shape) ({d}) must be a multiple of PWORD ({PWORD})."
            )
        N_dims, _, D_blocks = self._dims()
        if not N_dims:
            raise NotImplementedError(
                "layer_norm_dpu_reduce requires at least one N (leading) axis."
            )
        if self.bias_buffer is not None and self.quant_params is not None:
            # bias and quant_params would each be a per-row IFD1 stream; TWO per-row streams on one
            # IFD port MISALIGN on device (sim/interpret don't model it -> silent garbage). Fold the
            # bias into the quant zero-point instead: qz' = bias*scale + zp, pass quant_params only.
            raise ValueError(
                "bias_buffer + quant_params together need two per-row IFD1 streams (device-corrupting)"
                "; fold the bias into the quant zero-point (qz' = bias*scale + zp) and drop bias_buffer."
            )
        for name, buf in (
            ("weight_buffer", self.weight_buffer),
            ("bias_buffer", self.bias_buffer),
        ):
            if buf is not None:
                n_blocks = math.prod(int(s) for s in buf.shape)
                if n_blocks != D_blocks:
                    raise ValueError(
                        f"D-major {name} must have {D_blocks} PWORD blocks, "
                        f"got shape {tuple(buf.shape)} ({n_blocks} blocks)."
                    )
        if self.quant_params is not None:
            n_qp = math.prod(int(s) for s in self.quant_params.shape)
            if n_qp != 2 * D_blocks:
                raise ValueError(
                    f"quant_params must have 2*{D_blocks} PWORD blocks, "
                    f"got shape {tuple(self.quant_params.shape)}."
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
        dtype = "int8x64" if self.quant_params is not None else self.input.dtype
        return {
            "output": BufferInfo(
                shape=tuple(self.input.shape), dtype=dtype, scope=self.input.scope()
            )
        }

    # ------------------------------------------------------------------ #
    def implementation(self) -> None:
        generation = self.generation or HardwareGeneration.OMEGA
        input = self.input
        output_buffer = self.output
        invsqrt_lut = self.invsqrt_lut
        eps_const = self.eps_const
        weight_buffer = self.weight_buffer
        bias_buffer = self.bias_buffer
        newton_const = self.newton_const
        has_weight = weight_buffer is not None
        has_bias = bias_buffer is not None
        quant_params = self.quant_params
        has_quant = quant_params is not None

        normalized_shape = to_int_tuple(self.normalized_shape)
        N_dims, D_outer_dims, D_blocks = self._dims()
        d = math.prod(normalized_shape)
        one_over_d = A.constant(
            np.array([[1.0 / d] * PWORD], dtype=np.float32), dtype="float32x64"
        )

        n_luts = int(invsqrt_lut.shape[0])
        num_consts = (
            n_luts
            + int(one_over_d.shape[0])
            + int(eps_const.shape[0])
            + (int(newton_const.shape[0]) if newton_const is not None else 0)
            + (D_blocks if has_weight else 0)
            + (D_blocks if has_bias else 0)
            + (2 * D_blocks if has_quant else 0)
        )

        consts_dtype = "float32x64"
        input_dtype = "int32x64" if input.dtype == "int8x64" else input.dtype
        output_dtype = str(output_buffer.dtype)

        l1_size = MEMORY_SIZES[generation][MemoryScope.L1]
        fits_in_one_dispatch = _data_bytes(N_dims, D_blocks, input_dtype, output_dtype) <= int(
            l1_size * _SINGLE_DISPATCH_RATIO
        )

        core = self.core

        def _run_kernel(in_source, out_dest, n_grid_extents):
            with A.Datapath(core=core):
                with (
                    A.decl_stream(input_dtype) as input_stream,
                    A.decl_stream(consts_dtype) as lut_stream,
                    A.decl_stream(consts_dtype) as one_over_d_stream,
                    A.decl_stream(consts_dtype) as eps_stream,
                    (
                        A.decl_stream(consts_dtype) if newton_const is not None else nullcontext()
                    ) as newton_stream,
                    (
                        A.decl_stream(weight_buffer.dtype) if has_weight else nullcontext()
                    ) as weight_stream,
                    A.decl_stream(bias_buffer.dtype) if has_bias else nullcontext() as bias_stream,
                    A.decl_stream(quant_params.dtype) if has_quant else nullcontext() as qp_stream,
                    A.decl_stream(output_buffer.dtype) as output_stream,
                ):
                    # IFD0: input vectors in NDd order.
                    with A.grid(*n_grid_extents) as n_loop_vars:
                        n_loop_vars = to_tuple(n_loop_vars)
                        with A.grid(*D_outer_dims) as d_loop_vars:
                            d_loop_vars = to_tuple(d_loop_vars)
                            with A.IFD(num=0):
                                A.stream_push(input_stream, in_source[n_loop_vars + d_loop_vars])

                    # IFD1: constants (LUT, 1/D, eps, optional Newton, weight, bias).
                    with A.grid(n_luts) as k:
                        with A.IFD(num=1):
                            A.stream_push(lut_stream, invsqrt_lut[k])
                    with A.grid(*one_over_d.shape) as i:
                        with A.IFD(num=1):
                            A.stream_push(one_over_d_stream, one_over_d[i])
                    with A.grid(*eps_const.shape) as i:
                        with A.IFD(num=1):
                            A.stream_push(eps_stream, eps_const[i])
                    if newton_const is not None:
                        with A.grid(*newton_const.shape) as i:
                            with A.IFD(num=1):
                                A.stream_push(newton_stream, newton_const[i])
                    if has_weight:
                        with A.grid(*weight_buffer.shape) as i:
                            with A.IFD(num=1):
                                A.stream_push(weight_stream, weight_buffer[i])
                    if has_bias:
                        # Bias re-streamed PER ROW (kept out of resident registers to stay under
                        # the DPU register cap: resident bias would push x_buf+weight+bias to
                        # 3*D_blocks and overflow the file). Invariant across N, so each row pushes
                        # the same D_blocks vectors; popped one per block in the output loop.
                        with A.grid(*n_grid_extents):
                            with A.grid(*bias_buffer.shape) as k:
                                with A.IFD(num=1):
                                    A.stream_push(bias_stream, bias_buffer[k])
                    if has_quant:
                        with A.grid(*n_grid_extents):
                            with A.grid(*quant_params.shape) as k:
                                with A.IFD(num=1):
                                    A.stream_push(qp_stream, quant_params[k])

                    with A.grid(*n_grid_extents):
                        with A.DPU():
                            x_buf = A.array("float18x64", size=D_blocks)
                            mean = A.array("float18x64")
                            var = A.array("float18x64")
                            inv_std = A.array("float18x64")
                            if newton_const is not None:
                                invsqrt_pre = A.array("float18x64")

                            lut_regs = A.array("float18x64", size=n_luts)
                            one_over_d_reg = A.array("float18x64", size=1)
                            eps_reg = A.array("float18x64", size=1)
                            if newton_const is not None:
                                newton_regs = A.array("float18x64", size=2)
                            if has_weight:
                                weight_regs = A.array("float18x64", size=D_blocks)

                            sum_x = A.var("float18x64")
                            bias_tmp = A.var("float18x64")
                            sum_sq = A.var("float18x64")
                            sq_tmp = A.var("float18x64")
                            mean_sq = A.var("float18x64")
                            out_tmp = A.var("float18x64")
                            newton_tmp = A.var("float18x64")
                            lut_acc = A.var("float18x64")
                            qs_tmp = A.var("float18x64")
                            qz_tmp = A.var("float18x64")

                            with A.init():
                                with A.unroll(0, n_luts) as k:
                                    A.buffer_store(
                                        lut_regs, A.cast(lut_stream.pop(), "float18x64"), (k,)
                                    )
                                A.buffer_store(
                                    one_over_d_reg,
                                    A.cast(one_over_d_stream.pop(), "float18x64"),
                                    (0,),
                                )
                                A.buffer_store(
                                    eps_reg, A.cast(eps_stream.pop(), "float18x64"), (0,)
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
                            # Pop the D_blocks input vectors, accumulate per-lane sum(x) and sum(x^2).
                            with A.unroll(0, D_blocks) as k:
                                A.buffer_store(
                                    x_buf, A.cast(input_stream.pop(), "float18x64"), (k,)
                                )

                            A.buffer_store(sum_x, x_buf[0], (0,))
                            A.buffer_store(sum_sq, x_buf[0] * x_buf[0], (0,))
                            if D_blocks > 1:
                                with A.unroll(1, D_blocks) as k:
                                    A.buffer_store(sum_x, sum_x[0] + x_buf[k], (0,))
                                    A.buffer_store(sq_tmp, x_buf[k] * x_buf[k], (0,))
                                    A.buffer_store(sum_sq, sum_sq[0] + sq_tmp[0], (0,))

                            # add-tree over the 64 lanes -> scalar totals (broadcast to all lanes).
                            A.buffer_store(sum_x, A.dpu.ops.add_reduce(sum_x[0]), (0,))
                            A.buffer_store(sum_sq, A.dpu.ops.add_reduce(sum_sq[0]), (0,))

                            # mean = sum_x/D ; var = sum_sq/D - mean^2 + eps
                            A.buffer_store(mean, sum_x[0] * one_over_d_reg[0], (0,))
                            A.buffer_store(var, sum_sq[0] * one_over_d_reg[0], (0,))  # E[x^2]
                            A.buffer_store(mean_sq, mean[0] * mean[0], (0,))
                            A.buffer_store(var, var[0] - mean_sq[0], (0,))
                            A.buffer_store(var, var[0] + eps_reg[0], (0,))

                            invsqrt_dst = invsqrt_pre if newton_const is not None else inv_std
                            A.buffer_store(
                                invsqrt_dst,
                                chained_lut_dpu(
                                    value=var[0],
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
                                A.buffer_store(newton_tmp, var[0] * newton_tmp[0], (0,))
                                A.buffer_store(newton_tmp, newton_regs[0] - newton_tmp[0], (0,))
                                A.buffer_store(invsqrt_pre, invsqrt_pre[0] * newton_tmp[0], (0,))
                                A.buffer_store(inv_std, invsqrt_pre[0] * newton_regs[1], (0,))

                            # out[k] = (x - mean) * inv_std * weight[k] + bias[k], optional quant.
                            with A.unroll(0, D_blocks) as k:
                                A.buffer_store(out_tmp, x_buf[k] - mean[0], (0,))
                                A.buffer_store(out_tmp, out_tmp[0] * inv_std[0], (0,))
                                if has_weight:
                                    A.buffer_store(out_tmp, out_tmp[0] * weight_regs[k], (0,))
                                if has_bias:
                                    A.buffer_store(
                                        bias_tmp, A.cast(bias_stream.pop(), "float18x64"), (0,)
                                    )
                                    A.buffer_store(out_tmp, out_tmp[0] + bias_tmp[0], (0,))
                                if has_quant:
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

        if fits_in_one_dispatch:
            _run_kernel(input, output_buffer, N_dims)
            return

        # Tiled path (tile N_dims[1], keep inner N whole) -- same shape as the RMSNorm op.
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
                N_dims[0], 1, D_blocks, input_dtype, output_dtype, num_consts, generation
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
        """NumPy reference: fp18-rounded one-pass LayerNorm, chained additive inv-sqrt, D-major w/b."""
        normalized_shape = to_int_tuple(self.normalized_shape)
        d = math.prod(normalized_shape)
        out = self.output
        out_dtype = out.dtype

        x = emu.rf18(self.input.astype(np.float32).reshape(-1, d))
        inv_d = emu.rf18(1.0 / d)
        mean = emu.rf18(emu.rf18(x).sum(axis=-1, keepdims=True) * inv_d)
        ex2 = emu.rf18(emu.rf18(x * x).sum(axis=-1, keepdims=True) * inv_d)
        var = emu.rf18(
            emu.rf18(ex2 - emu.rf18(mean * mean)) + emu.rf18(float(self.eps_const[0, 0]))
        )

        lut_2d = self.invsqrt_lut.reshape(-1, PWORD)
        inv_std = emu.apply_lut(var, lut_2d[0:1])
        for k in range(1, lut_2d.shape[0]):
            inv_std = emu.rf18(inv_std + emu.apply_lut(var, lut_2d[k : k + 1]))
        if self.newton_const is not None:
            three = emu.rf18(float(self.newton_const[0, 0]))
            half = emu.rf18(float(self.newton_const[1, 0]))
            tmp = emu.rf18(inv_std * inv_std)
            tmp = emu.rf18(var * tmp)
            tmp = emu.rf18(three - tmp)
            inv_std = emu.rf18(inv_std * tmp)
            inv_std = emu.rf18(inv_std * half)

        normalized = emu.rf18(emu.rf18(x - mean) * inv_std)
        if self.weight_buffer is not None:
            w = emu.rf18(self.weight_buffer.astype(np.float32).reshape(-1))
            normalized = emu.rf18(normalized * w[None, :])
        if self.bias_buffer is not None:
            b = emu.rf18(self.bias_buffer.astype(np.float32).reshape(-1))
            normalized = emu.rf18(normalized + b[None, :])
        if self.quant_params is not None:
            qp = self.quant_params.astype(np.float32).reshape(2 * (d // PWORD), PWORD)
            qs = emu.rf18(qp[0::2].reshape(-1))
            qz = emu.rf18(qp[1::2].reshape(-1))
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


layer_norm_dpu_reduce = register_op(
    LayerNormDPUReduce,
    "llm.layer_norm_dpu_reduce",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)


def _row_bands(m, n):
    """Split m rows into n contiguous, as-even-as-possible bands: list of (lo, hi)."""
    base, rem = divmod(m, n)
    out, lo = [], 0
    for i in range(n):
        hi = lo + base + (1 if i < rem else 0)
        out.append((lo, hi))
        lo = hi
    return out


def layer_norm_dpu_reduce_mshard(
    input,
    invsqrt_lut,
    eps_const,
    *,
    normalized_shape,
    cores,
    m_axis=1,
    weight_buffer=None,
    bias_buffer=None,
    quant_params=None,
    invsqrt_method="additive",
    output_dtype=None,
):
    """M-sharded ``layer_norm_dpu_reduce``: split the row axis (``m_axis``, default the seq axis of a
    ``(1, seq, D_blocks)`` input) across ``cores`` and normalise each contiguous row-band on its own
    core, concurrently. LayerNorm reduces over the trailing ``normalized_shape`` PER ROW, so rows are
    independent -- no cross-core reduction -- and the affine weight/bias/quant params are tiny and
    read in full by each core (negligible replication, unlike a matmul weight). The row-band output
    is contiguous (M is an outer axis), so each core writes one slab. Falls back to a single unpinned
    call when ``cores`` has <= 1 entry. Returns the full output buffer.
    """
    from axelera.compiler.alto import script as A

    from .quantized_matmul_activation import slice_n

    core_ids = tuple(int(c) for c in cores) if cores is not None else ()
    out_dtype = output_dtype or ("int8x64" if quant_params is not None else input.dtype)

    def _ln(x, out, core):
        layer_norm_dpu_reduce(
            x,
            invsqrt_lut,
            eps_const,
            normalized_shape=normalized_shape,
            weight_buffer=weight_buffer,
            bias_buffer=bias_buffer,
            quant_params=quant_params,
            invsqrt_method=invsqrt_method,
            output=out,
            core=core,
        )

    if len(core_ids) <= 1:
        core = core_ids[0] if core_ids else None
        out = A.alloc_buffer(
            tuple(int(s) for s in input.shape), dtype=out_dtype, scope=input.scope()
        )
        _ln(input, out, core)
        return out

    out = A.alloc_buffer(tuple(int(s) for s in input.shape), dtype=out_dtype, scope=input.scope())
    bands = _row_bands(int(input.shape[m_axis]), len(core_ids))
    for i, (lo, hi) in enumerate(bands):
        with A.AICore(core=core_ids[i]):
            A.barrier(*core_ids)
            _ln(slice_n(input, m_axis, lo, hi), slice_n(out, m_axis, lo, hi), core_ids[i])
            A.barrier(*core_ids)
    return out

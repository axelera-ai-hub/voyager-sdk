#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Single-token gated delta rule in fp32 on RVV (llm.deltanet_step_rvv).

The decode counterpart of deltanet_chunk_rvv: one token, so the chunk form's decay
pivot, triangular solve and Gram matrices all collapse away and what is left is two
passes over the recurrent state. See deltanet_step_rvv.c for the algebra.

The gates ride along rather than coming from deltanet_gates_rvv, which packs the
sequence into lanes and so needs seq a multiple of PWORD; at one token every head's
gate is a single lane word.
"""

# mypy: disable-error-code="attr-defined, index, arg-type, call-overload, type-var, misc"

from axelera.compiler.alto import script as A
import axelera.compiler.alto.axol as axol
from axelera.compiler.alto.axol import intrinsic_registry
from axelera.compiler.alto.axol.intrinsic_registry import FunctionArgument, FunctionArgumentType
from axelera.compiler.alto.compiler.dtypes import MetisTypes
from axelera.compiler.alto.compiler.op_registry import Arg, BufferInfo, ScriptOp, register_op
from axelera.config import HardwareGeneration
import numpy as np
from tvm import tir

from ._rvv import kernel_source, sigmoid, to_scalar

_INTRINSIC_NAME = "rvv_deltanet_step"

intrinsic_registry.set(
    name=_INTRINSIC_NAME,
    arguments=[
        FunctionArgument(name="q", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="k", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="v", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="a", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="b", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="pos_a", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="dt_bias", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="state", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="out", type=FunctionArgumentType.DLTENSOR),
    ],
    implementation=kernel_source("deltanet_step_rvv"),
)


class DeltaNetStepRvv(ScriptOp):
    """One decode step of the gated delta rule, with an fp32 recurrent state."""

    q = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="(bh, dk/64) float32x64 L2-normed query, head-major, one token.",
    )
    k = Arg(MetisTypes[tir.Buffer, "..."], description="(bh, dk/64) float32x64 key.")
    v = Arg(MetisTypes[tir.Buffer, "..."], description="(bh, dv/64) float32x64 value.")
    a = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="(1, 1) float32x64 in_proj_a output, head in lane.",
    )
    b = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="(1, 1) float32x64 in_proj_b output, head in lane.",
    )
    pos_a = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="(1, 1) float32x64 exp(A_log) per head, head in lane.",
    )
    dt_bias = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="(1, 1) float32x64 per-head softplus bias, head in lane.",
    )
    state = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="(bh, dk, dv/64) float32x64 recurrent state, updated IN PLACE.",
    )
    num_heads = Arg(int, kw=True, description="Number of valid head lanes.")
    output = Arg(
        MetisTypes[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="(bh, dv/64) float32x64 readout, or None to auto-allocate.",
    )
    core = Arg(int | None, default=None, kw=True, description="Optional AI core to run on.")
    cores = Arg(
        int | tuple[int, ...] | None,
        default=None,
        kw=True,
        description="AI cores to distribute the heads across (int -> 0..n-1, tuple -> ids, "
        "None/<=1 -> single core). Each core owns a contiguous band of heads.",
    )
    generation = Arg(
        HardwareGeneration | None,
        default=None,
        kw=True,
        description="Optional hardware generation.",
    )

    def validate(self) -> None:
        from axelera.compiler.alto.compiler.device import PWORD

        q_shape = tuple(int(s) for s in self.q.shape)
        if len(q_shape) != 2:
            raise ValueError(f"q must be (bh, blocks) lane-packed, got {q_shape}")
        for name in ("k", "v"):
            shape = tuple(int(s) for s in getattr(self, name).shape)
            if shape[0] != q_shape[0]:
                raise ValueError(f"{name} shape {shape} must agree with q {q_shape} on bh")
        for name in ("q", "k", "v", "a", "b", "pos_a", "dt_bias", "state"):
            dtype = getattr(self, name).dtype
            if dtype != "float32x64":
                raise ValueError(f"{name} dtype must be 'float32x64', got '{dtype}'")
        for name in ("a", "b", "pos_a", "dt_bias"):
            shape = tuple(int(s) for s in getattr(self, name).shape)
            if shape != (1, 1):
                raise ValueError(f"{name} must be (1, 1) lane-packed, got {shape}")
        bh = q_shape[0]
        dk, dv = self._dims()
        state_shape = tuple(int(s) for s in self.state.shape)
        if state_shape != (bh, dk, dv // PWORD):
            raise ValueError(
                f"state must be ({bh}, {dk}, {dv // PWORD}) for dk={dk} dv={dv}, "
                f"got {state_shape}"
            )
        if not 0 < int(self.num_heads) <= min(bh, PWORD):
            raise ValueError(f"num_heads must be in [1, {min(bh, PWORD)}], got {self.num_heads}")
        # The whole state is staged in L1 -- it is read twice per call, and re-DMAing it
        # for the second pass would double the traffic that dominates this op.
        l1_bytes = (bh * dk * dv + 4 * bh * dv) * 4
        if l1_bytes > 3 * 1024 * 1024:
            raise ValueError(
                f"bh={bh} dk={dk} dv={dv} needs {l1_bytes / 2**20:.1f} MB of L1; too large"
            )

    def _dims(self) -> tuple[int, int]:
        from axelera.compiler.alto.compiler.device import PWORD

        return int(self.q.shape[1]) * PWORD, int(self.v.shape[1]) * PWORD

    def infer_outputs(self) -> dict[str, BufferInfo]:
        return {
            "output": BufferInfo(
                shape=tuple(int(s) for s in self.v.shape),
                dtype=self.v.dtype,
                scope=self.v.scope(),
            )
        }

    def implementation(self) -> None:
        from axelera.compiler.alto.axol.core.matmul._common import shards

        bh = int(self.q.shape[0])
        # ScriptOp wraps a Python-int arg as a tir IntImm, so test for iterability rather
        # than isinstance(int): scalar count -> 0..n-1, iterable -> ids, None -> one core.
        if self.cores is None:
            core_ids = ()
        else:
            try:
                core_ids = tuple(int(c) for c in self.cores)
            except TypeError:
                core_ids = tuple(range(int(self.cores)))

        if len(core_ids) <= 1:
            self._emit_heads(0, bh, self.core)
            return

        # Heads are fully independent -- each carries its own dk*dv state slice and there
        # is no reduction across them -- so a head band per core needs only the barriers
        # that order the shared L1/DDR traffic, no cross-core exchange. Every operand is
        # head-major, so a band is contiguous and the DMAs stay whole.
        partitions = list(shards(bh, core_ids))
        participants = [c for c, _, _ in partitions]
        for core, lo, hi in partitions:
            with A.AICore(core=core):
                A.barrier(*participants)
                self._emit_heads(lo, hi, core)
                A.barrier(*participants)

    def _emit_heads(self, lo: int, hi: int, core) -> None:
        """Emit the recurrence for heads [lo, hi) on ``core``."""
        from axelera.compiler.alto.compiler.device import PWORD

        bh = int(self.q.shape[0])
        dk, dv = self._dims()

        # The gates arrive with the heads in LANES, which cannot be tiled over heads --
        # so turn each one into per-head ROWS first: repeat the lane word down 64 rows
        # and transpose, which leaves row h holding head h's value across every lane.
        # One transpose per gate word, not the 64 the DPU decode path needed per head.
        def per_head(word):
            rows = axol.atomic.vector_transpose(
                axol.atomic.repeat(axol.atomic.reshape(word, shape=(1, 1)), (PWORD, 1)), 0, 1
            )
            return axol.atomic.copy(rows[0:bh, :])

        gates = [per_head(getattr(self, n)) for n in ("a", "b", "pos_a", "dt_bias")]

        q_s = A.unpack(self.q, (bh, dk))
        k_s = A.unpack(self.k, (bh, dk))
        v_s = A.unpack(self.v, (bh, dv))
        state_s = A.unpack(self.state, (bh, dk, dv))
        out_s = A.unpack(self.output, (bh, dv))
        gates_s = [A.unpack(g, (bh, PWORD)) for g in gates]

        band = hi - lo
        l1_q = A.alloc_buffer((band, dk), "float32", scope=A.L1)
        l1_k = A.alloc_buffer((band, dk), "float32", scope=A.L1)
        l1_v = A.alloc_buffer((band, dv), "float32", scope=A.L1)
        l1_state = A.alloc_buffer((band, dk, dv), "float32", scope=A.L1)
        l1_out = A.alloc_buffer((band, dv), "float32", scope=A.L1)
        l1_gates = [A.alloc_buffer((band, PWORD), "float32", scope=A.L1) for _ in gates_s]

        # Tiled over this core's band of heads. Every operand is head-major so each tile
        # is contiguous, and the state -- 4*dk*dv bytes per head, the bulk of the traffic
        # -- only ever needs one tile resident. Untiled, the read-only operands' live
        # ranges end at the extern and the next op's staging is free to land on top of
        # them while this kernel is still reading; a tile scope keeps each one live
        # across its own invoke.
        #
        # A.tiles(lo, hi)'s Ramp is always based at loop_var * tile_size (only the
        # extent comes from hi - lo), so tiling the full-width source buffers with it
        # directly addresses [0, band) on every core, not its own [lo, hi) -- and
        # .tile() needs a real Buffer, not the BufferRegion a plain slice produces, so
        # the fix is to tile 0-based over the band and shift the Ramp by lo by hand
        # before applying it to the full-width source buffers. The L1 destinations are
        # already band-sized, so they tile against the unshifted 0-based ramp.
        with A.tiles(band) as head_tile:
            src_tile = tir.Ramp(head_tile.base + lo, head_tile.stride, head_tile.lanes)

            def tiled_src(buf):
                return buf.tile(src_tile, dim=0)

            def tiled(buf):
                return buf.tile(head_tile, dim=0)

            for src, dst in zip(
                (*gates_s, q_s, k_s, v_s, state_s), (*l1_gates, l1_q, l1_k, l1_v, l1_state)
            ):
                axol.atomic.copy(tiled_src(src), output=tiled(dst))
            with A.Invoke(core=core):
                A.extern(
                    _INTRINSIC_NAME,
                    tiled(l1_q),
                    tiled(l1_k),
                    tiled(l1_v),
                    *[tiled(g) for g in l1_gates],
                    tiled(l1_state),
                    tiled(l1_out),
                )
            axol.atomic.copy(tiled(l1_out), output=tiled_src(out_s))
            axol.atomic.copy(tiled(l1_state), output=tiled_src(state_s))

    def emulate(self) -> None:  # type: ignore[override]
        q = to_scalar(self.q).astype(np.float32)
        k = to_scalar(self.k).astype(np.float32)
        v = to_scalar(self.v).astype(np.float32)
        a, b, pos_a, dt_bias = (
            to_scalar(getattr(self, n))[0].astype(np.float32)
            for n in ("a", "b", "pos_a", "dt_bias")
        )
        state = to_scalar(self.state)
        out = to_scalar(self.output)

        x = a + dt_bias
        softplus = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
        alpha = np.exp(-pos_a * softplus)
        beta = sigmoid(b)

        for h in range(int(self.num_heads)):
            s = state[h].astype(np.float32)
            vnew = beta[h] * (v[h] - alpha[h] * (k[h] @ s))
            out[h] = alpha[h] * (q[h] @ s) + (q[h] @ k[h]) * vnew
            state[h] = alpha[h] * s + np.outer(k[h], vnew)


deltanet_step_rvv = register_op(
    DeltaNetStepRvv,
    "llm.deltanet_step_rvv",
    generation=HardwareGeneration.EUROPA,
    atomic=False,
)

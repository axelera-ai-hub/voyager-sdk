#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Fused L2-norm(q) + L2-norm(k) + gated-delta step + gated RMSNorm (llm.deltanet_mixer_core_rvv).

The mixer chains these four RVV ops 1:1 -- L2-norm(q)/L2-norm(k) feed the recurrence,
whose readout feeds the gated RMSNorm -- and none of the three intermediates
(normalized q, normalized k, the step readout) are needed for anything else. Run
separately, each is its own kernel launch: its own DDR round trip in and out, its own
dcache_invalidate_buffer pass, and (deltanet_l2norm_rvv / deltanet_gated_rmsnorm_rvv
specifically) no multicore sharding at all -- only deltanet_step_rvv shards today.

Fusing them keeps the three intermediates in L1 scratch (never touching DDR) and lets
one head-band multicore partition cover all four computations' work, not just the
step's. See deltanet_step_rvv.c for the algebra this reuses unchanged, and
deltanet_l2norm_rvv.c / deltanet_gated_rmsnorm_rvv.c for the two norms'.
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

_INTRINSIC_NAME = "rvv_deltanet_mixer_core"

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
        FunctionArgument(name="z", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="gnorm_weight", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="state", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="qn", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="kn", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="ot", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="out", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="q_scale", type=FunctionArgumentType.UINT64),
        FunctionArgument(name="k_scale", type=FunctionArgumentType.UINT64),
        FunctionArgument(name="l2norm_eps", type=FunctionArgumentType.UINT64),
        FunctionArgument(name="gnorm_eps", type=FunctionArgumentType.UINT64),
    ],
    implementation=kernel_source("deltanet_mixer_core_rvv"),
)


def _float_bits(value: float) -> int:
    import struct

    return struct.unpack("<I", struct.pack("<f", float(value)))[0]


class DeltaNetMixerCoreRvv(ScriptOp):
    """One decode step's q/k L2-norm + gated-delta recurrence + gated RMSNorm, fused."""

    q = Arg(MetisTypes[tir.Buffer, "..."], description="(bh, dk/64) float32x64 raw query.")
    k = Arg(MetisTypes[tir.Buffer, "..."], description="(bh, dk/64) float32x64 raw key.")
    v = Arg(MetisTypes[tir.Buffer, "..."], description="(bh, dv/64) float32x64 value.")
    a = Arg(
        MetisTypes[tir.Buffer, "..."], description="(1, 1) float32x64 in_proj_a, head in lane."
    )
    b = Arg(
        MetisTypes[tir.Buffer, "..."], description="(1, 1) float32x64 in_proj_b, head in lane."
    )
    pos_a = Arg(
        MetisTypes[tir.Buffer, "..."], description="(1, 1) float32x64 exp(A_log), head in lane."
    )
    dt_bias = Arg(
        MetisTypes[tir.Buffer, "..."], description="(1, 1) float32x64 softplus bias, head in lane."
    )
    z = Arg(
        MetisTypes[tir.Buffer, "..."], description="(bh, dv/64) float32x64 gated-RMSNorm gate."
    )
    gnorm_weight = Arg(
        MetisTypes[tir.Buffer, "..."], description="(1, dv/64) float32x64, shared across heads."
    )
    state = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="(bh, dk, dv/64) float32x64 recurrent state, updated IN PLACE.",
    )
    q_eps = Arg(float, default=1e-6, kw=True, description="L2-norm eps, shared by q and k.")
    q_scale = Arg(float, default=1.0, kw=True, description="Applied after normalising q.")
    k_scale = Arg(float, default=1.0, kw=True, description="Applied after normalising k.")
    gnorm_eps = Arg(float, default=1e-6, kw=True, description="Gated-RMSNorm eps.")
    num_heads = Arg(int, kw=True, description="Number of valid head lanes.")
    output = Arg(
        MetisTypes[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="(bh, dv/64) float32x64 gated-RMSNorm output, or None to auto-allocate.",
    )
    core = Arg(int | None, default=None, kw=True, description="Optional AI core to run on.")
    cores = Arg(
        int | tuple[int, ...] | None,
        default=None,
        kw=True,
        description="AI cores to distribute the heads across, as deltanet_step_rvv.",
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
        for name in ("k", "v", "z"):
            shape = tuple(int(s) for s in getattr(self, name).shape)
            if shape[0] != q_shape[0]:
                raise ValueError(f"{name} shape {shape} must agree with q {q_shape} on bh")
        for name in ("q", "k", "v", "a", "b", "pos_a", "dt_bias", "z", "gnorm_weight", "state"):
            dtype = getattr(self, name).dtype
            if dtype != "float32x64":
                raise ValueError(f"{name} dtype must be 'float32x64', got '{dtype}'")
        for name in ("a", "b", "pos_a", "dt_bias"):
            shape = tuple(int(s) for s in getattr(self, name).shape)
            if shape != (1, 1):
                raise ValueError(f"{name} must be (1, 1) lane-packed, got {shape}")
        bh = q_shape[0]
        dk, dv = self._dims()
        if tuple(int(s) for s in self.gnorm_weight.shape) != (1, dv // PWORD):
            raise ValueError(
                f"gnorm_weight must be (1, {dv // PWORD}), got {tuple(self.gnorm_weight.shape)}"
            )
        state_shape = tuple(int(s) for s in self.state.shape)
        if state_shape != (bh, dk, dv // PWORD):
            raise ValueError(
                f"state must be ({bh}, {dk}, {dv // PWORD}) for dk={dk} dv={dv}, "
                f"got {state_shape}"
            )
        if not 0 < int(self.num_heads) <= min(bh, PWORD):
            raise ValueError(f"num_heads must be in [1, {min(bh, PWORD)}], got {self.num_heads}")
        # state + qn/kn/ot scratch, all staged in L1 for the whole call (see deltanet_step_rvv).
        l1_bytes = (bh * dk * dv + bh * (2 * dk + 2 * dv) + 4 * bh * dv) * 4
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
        if self.cores is None:
            core_ids = ()
        else:
            try:
                core_ids = tuple(int(c) for c in self.cores)
            except TypeError:
                core_ids = tuple(range(int(self.cores)))

        # The gate transpose is whole-bh and identical for every core -- compute it ONCE,
        # here, rather than redundantly inside each core's own band (see
        # deltanet_step_rvv.py's identical per_head, which redoes this per core).
        def per_head(word):
            from axelera.compiler.alto.compiler.device import PWORD

            rows = axol.atomic.vector_transpose(
                axol.atomic.repeat(axol.atomic.reshape(word, shape=(1, 1)), (PWORD, 1)), 0, 1
            )
            return axol.atomic.copy(rows[0:bh, :])

        gates = [per_head(getattr(self, n)) for n in ("a", "b", "pos_a", "dt_bias")]

        if len(core_ids) <= 1:
            self._emit_heads(0, bh, self.core, gates)
            return

        partitions = list(shards(bh, core_ids))
        participants = [c for c, _, _ in partitions]
        for core, lo, hi in partitions:
            with A.AICore(core=core):
                A.barrier(*participants)
                self._emit_heads(lo, hi, core, gates)
                A.barrier(*participants)

    def _emit_heads(self, lo: int, hi: int, core, gates) -> None:
        """Emit the fused computation for heads [lo, hi) on ``core``."""
        from axelera.compiler.alto.compiler.device import PWORD

        bh = int(self.q.shape[0])
        dk, dv = self._dims()

        q_s = A.unpack(self.q, (bh, dk))
        k_s = A.unpack(self.k, (bh, dk))
        v_s = A.unpack(self.v, (bh, dv))
        z_s = A.unpack(self.z, (bh, dv))
        state_s = A.unpack(self.state, (bh, dk, dv))
        out_s = A.unpack(self.output, (bh, dv))
        gates_s = [A.unpack(g, (bh, PWORD)) for g in gates]
        gnorm_weight_s = A.unpack(self.gnorm_weight, (1, dv))

        band = hi - lo
        l1_q = A.alloc_buffer((band, dk), "float32", scope=A.L1)
        l1_k = A.alloc_buffer((band, dk), "float32", scope=A.L1)
        l1_v = A.alloc_buffer((band, dv), "float32", scope=A.L1)
        l1_z = A.alloc_buffer((band, dv), "float32", scope=A.L1)
        l1_state = A.alloc_buffer((band, dk, dv), "float32", scope=A.L1)
        l1_qn = A.alloc_buffer((band, dk), "float32", scope=A.L1)
        l1_kn = A.alloc_buffer((band, dk), "float32", scope=A.L1)
        l1_ot = A.alloc_buffer((band, dv), "float32", scope=A.L1)
        l1_out = A.alloc_buffer((band, dv), "float32", scope=A.L1)
        l1_gates = [A.alloc_buffer((band, PWORD), "float32", scope=A.L1) for _ in gates_s]
        l1_gnorm_weight = A.alloc_buffer((1, dv), "float32", scope=A.L1)
        axol.atomic.copy(gnorm_weight_s, output=l1_gnorm_weight)

        # See deltanet_step_rvv.py: A.tiles(lo, hi)'s Ramp is always 0-based, so tile the
        # band 0-based and shift the Ramp by lo before applying it to full-width sources.
        with A.tiles(band) as head_tile:
            src_tile = tir.Ramp(head_tile.base + lo, head_tile.stride, head_tile.lanes)

            def tiled_src(buf):
                return buf.tile(src_tile, dim=0)

            def tiled(buf):
                return buf.tile(head_tile, dim=0)

            for src, dst in zip(
                (*gates_s, q_s, k_s, v_s, z_s, state_s),
                (*l1_gates, l1_q, l1_k, l1_v, l1_z, l1_state),
            ):
                axol.atomic.copy(tiled_src(src), output=tiled(dst))
            with A.Invoke(core=core):
                A.extern(
                    _INTRINSIC_NAME,
                    tiled(l1_q),
                    tiled(l1_k),
                    tiled(l1_v),
                    *[tiled(g) for g in l1_gates],
                    tiled(l1_z),
                    l1_gnorm_weight,
                    tiled(l1_state),
                    tiled(l1_qn),
                    tiled(l1_kn),
                    tiled(l1_ot),
                    tiled(l1_out),
                    _float_bits(self.q_scale),
                    _float_bits(self.k_scale),
                    _float_bits(self.q_eps),
                    _float_bits(self.gnorm_eps),
                )
            axol.atomic.copy(tiled(l1_out), output=tiled_src(out_s))
            axol.atomic.copy(tiled(l1_state), output=tiled_src(state_s))

    def emulate(self) -> None:  # type: ignore[override]
        q = to_scalar(self.q).astype(np.float32)
        k = to_scalar(self.k).astype(np.float32)
        v = to_scalar(self.v).astype(np.float32)
        z = to_scalar(self.z).astype(np.float32)
        gnorm_weight = to_scalar(self.gnorm_weight).astype(np.float32)[0]
        a, b, pos_a, dt_bias = (
            to_scalar(getattr(self, n))[0].astype(np.float32)
            for n in ("a", "b", "pos_a", "dt_bias")
        )
        state = to_scalar(self.state)
        out = to_scalar(self.output)

        # The scalar Args arrive as TVM FloatImm under interpret(), not Python floats,
        # and numpy ufuncs reject them ("FloatImm has no attribute sqrt"). Coerce once
        # here so the op is usable in host-side interpret() validation, not just on device.
        q_scale = float(self.q_scale)
        k_scale = float(self.k_scale)
        q_eps = float(self.q_eps)
        gnorm_eps = float(self.gnorm_eps)

        x = a + dt_bias
        softplus = np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)
        alpha = np.exp(-pos_a * softplus)
        beta = sigmoid(b)

        for h in range(int(self.num_heads)):
            qn = q[h] * q_scale / np.sqrt((q[h] * q[h]).sum() + q_eps)
            kn = k[h] * k_scale / np.sqrt((k[h] * k[h]).sum() + q_eps)

            s = state[h].astype(np.float32)
            vnew = beta[h] * (v[h] - alpha[h] * (kn @ s))
            ot = alpha[h] * (qn @ s) + (qn @ kn) * vnew
            state[h] = alpha[h] * s + np.outer(kn, vnew)

            gscale = 1.0 / np.sqrt((ot * ot).mean() + gnorm_eps)
            silu = z[h] * sigmoid(z[h])
            out[h] = gnorm_weight * ot * gscale * silu


deltanet_mixer_core_rvv = register_op(
    DeltaNetMixerCoreRvv,
    "llm.deltanet_mixer_core_rvv",
    generation=HardwareGeneration.EUROPA,
    atomic=False,
)

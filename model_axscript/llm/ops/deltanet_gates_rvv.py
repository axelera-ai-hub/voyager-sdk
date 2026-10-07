#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Gated-DeltaNet per-token gates in fp32 on RVV (llm.deltanet_gates_rvv).

Replaces the three chained float18 LUTs the Metis path uses for this step
(sigmoid, softplus, exp) with exact fp32 arithmetic, and emits the LOG decay so
the chunked recurrence can work in log space:

    g    = -exp(A_log) * softplus(a + dt_bias)
    beta = sigmoid(b)

exp(A_log) is folded on the host -- A_log is a weight, so there is no reason to
spend an exponential per token on it. See _rvv_prelude.c for the fp32 softplus
and sigmoid; there is no libm on the AI core.
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

from ._rvv import kernel_source, sigmoid, to_scalar, unpacked_shape

_INTRINSIC_NAME = "rvv_deltanet_gates"

intrinsic_registry.set(
    name=_INTRINSIC_NAME,
    arguments=[
        FunctionArgument(name="a", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="b", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="pos_a", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="dt_bias", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="mask", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="g", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="beta", type=FunctionArgumentType.DLTENSOR),
        FunctionArgument(name="heads", type=FunctionArgumentType.UINT64),
    ],
    implementation=kernel_source("deltanet_gates_rvv"),
)


class DeltaNetGatesRvv(ScriptOp):
    """Per-(head, token) log decay and beta from the a/b gate projections."""

    a = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="(seq, 1) float32x64 in_proj_a output, head in lane.",
    )
    b = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="(seq, 1) float32x64 in_proj_b output, head in lane.",
    )
    pos_a = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="(1, 1) float32x64 exp(A_log) per head, head in lane.",
    )
    dt_bias = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="(1, 1) float32x64 per-head softplus bias, head in lane.",
    )
    mask = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="(1, seq/64) float32x64: 1.0 per real token, 0.0 for chunk padding.",
    )
    num_heads = Arg(int, kw=True, description="Number of valid head lanes.")
    g = Arg(
        MetisTypes[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="(heads, seq/64) float32x64 log decay, or None to auto-allocate.",
    )
    beta = Arg(
        MetisTypes[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="(heads, seq/64) float32x64 sigmoid(b), or None to auto-allocate.",
    )
    core = Arg(int | None, default=None, kw=True, description="Optional AI core to run on.")
    generation = Arg(
        HardwareGeneration | None,
        default=None,
        kw=True,
        description="Optional hardware generation.",
    )

    def validate(self) -> None:
        a_shape = tuple(int(s) for s in self.a.shape)
        if len(a_shape) != 2 or a_shape[1] != 1:
            raise ValueError(f"a must be (seq, 1) lane-packed, got {a_shape}")
        if tuple(int(s) for s in self.b.shape) != a_shape:
            raise ValueError(f"a and b must match, got {a_shape} and {tuple(self.b.shape)}")
        for name in ("a", "b", "pos_a", "dt_bias", "mask"):
            dtype = getattr(self, name).dtype
            if dtype != "float32x64":
                raise ValueError(f"{name} dtype must be 'float32x64', got '{dtype}'")
        for name in ("pos_a", "dt_bias"):
            shape = tuple(int(s) for s in getattr(self, name).shape)
            if shape != (1, 1):
                raise ValueError(f"{name} must be (1, 1) lane-packed, got {shape}")
        from axelera.compiler.alto.compiler.device import PWORD

        if not 0 < int(self.num_heads) <= PWORD:
            raise ValueError(f"num_heads must be in [1, {PWORD}], got {self.num_heads}")
        seq = int(self.a.shape[0])
        if seq % PWORD:
            raise ValueError(f"seq must be a multiple of {PWORD} to pack head-major, got {seq}")
        # The op transposes, so neither side tiles contiguously and it runs
        # untiled; a/b dominate the footprint at 4 bytes per lane per token.
        l1_bytes = 2 * seq * PWORD * 4 + 2 * int(self.num_heads) * seq * 4
        if l1_bytes > 3 * 1024 * 1024:
            raise ValueError(f"seq={seq} needs {l1_bytes / 2**20:.1f} MB of L1; too large")

    def infer_outputs(self) -> dict[str, BufferInfo]:
        from axelera.compiler.alto.compiler.device import PWORD

        info = BufferInfo(
            shape=(int(self.num_heads), int(self.a.shape[0]) // PWORD),
            dtype=self.a.dtype,
            scope=self.a.scope(),
        )
        return {"g": info, "beta": info}

    def implementation(self) -> None:
        seq, lanes = unpacked_shape(tuple(int(s) for s in self.a.shape))
        gate_shape = unpacked_shape(tuple(int(s) for s in self.g.shape))
        const_shape = unpacked_shape((1, 1))

        a_s = A.unpack(self.a, (seq, lanes))
        b_s = A.unpack(self.b, (seq, lanes))
        pos_a_s = A.unpack(self.pos_a, const_shape)
        dt_bias_s = A.unpack(self.dt_bias, const_shape)
        mask_s = A.unpack(self.mask, (1, seq))
        g_s = A.unpack(self.g, gate_shape)
        beta_s = A.unpack(self.beta, gate_shape)

        l1_a = A.alloc_buffer((seq, lanes), "float32", scope=A.L1)
        l1_b = A.alloc_buffer((seq, lanes), "float32", scope=A.L1)
        l1_pos_a = A.alloc_buffer(const_shape, "float32", scope=A.L1)
        l1_dt_bias = A.alloc_buffer(const_shape, "float32", scope=A.L1)
        l1_mask = A.alloc_buffer((1, seq), "float32", scope=A.L1)
        l1_g = A.alloc_buffer(gate_shape, "float32", scope=A.L1)
        l1_beta = A.alloc_buffer(gate_shape, "float32", scope=A.L1)

        axol.atomic.copy(a_s, output=l1_a)
        axol.atomic.copy(b_s, output=l1_b)
        axol.atomic.copy(pos_a_s, output=l1_pos_a)
        axol.atomic.copy(dt_bias_s, output=l1_dt_bias)
        axol.atomic.copy(mask_s, output=l1_mask)
        with A.Invoke(core=self.core):
            A.extern(
                _INTRINSIC_NAME,
                l1_a,
                l1_b,
                l1_pos_a,
                l1_dt_bias,
                l1_mask,
                l1_g,
                l1_beta,
                int(self.num_heads),
            )
        axol.atomic.copy(l1_g, output=g_s)
        axol.atomic.copy(l1_beta, output=beta_s)

    def emulate(self) -> None:  # type: ignore[override]
        heads = int(self.num_heads)
        a = to_scalar(self.a).astype(np.float32)[:, :heads]
        b = to_scalar(self.b).astype(np.float32)[:, :heads]
        pos_a = to_scalar(self.pos_a).astype(np.float32)[0, :heads]
        dt_bias = to_scalar(self.dt_bias).astype(np.float32)[0, :heads]

        mask = to_scalar(self.mask).astype(np.float32)[0][:, None]  # (seq, 1)

        x = a + dt_bias
        softplus = np.maximum(x, 0.0) + np.log1p(np.exp(-np.abs(x)))
        np.copyto(to_scalar(self.g), (-pos_a * softplus * mask).T.astype(np.float32))
        np.copyto(to_scalar(self.beta), (sigmoid(b) * mask).T.astype(np.float32))


deltanet_gates_rvv = register_op(
    DeltaNetGatesRvv,
    "llm.deltanet_gates_rvv",
    generation=HardwareGeneration.EUROPA,
    atomic=False,
)

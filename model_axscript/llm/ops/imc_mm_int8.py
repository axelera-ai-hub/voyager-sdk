#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Batched int8 IMC matmul with on-device quantization (llm.imc_mm_int8).

C[g] = A[g] @ B[g] for a batch of independent fp32 matmuls, with both operands quantized
on the RVV unit, the product accumulated in int32 on the IMC, and the result dequantized
by the outer product sa[m]*sb[n] on the DPU.

This is the piece the DeltaNet chunk op needs: five of its six matmuls are 82% of that
kernel and cannot go faster on RVV, because a vector op costs ~40 cycles of issue overhead
plus vl/128 of work and vl is capped at the head dimension. Flash attention already runs
2.4x more matmul work than the whole chunk op in a fifth of the time by putting it here.

Both operands are activations -- there is no static weight -- so the IMC array is reloaded
per matmul. That is what flash does for K and V too, and the array is small enough that it
is not the constraint: the region holds 256 int8 blocks and the largest operand here is 4.

Accuracy, measured in numpy against the fp64 recurrence: per-row/per-column int8 costs
1.4e-2 over all six of the chunk op's matmuls, dominated by Q@S0 (1.3e-2) because that
one lands directly in the output rather than being attenuated through the solve. The DPU's
float18 dequant adds ~1e-3 on top, below the int8 term.
"""

# mypy: disable-error-code="attr-defined, index, arg-type, call-overload, type-var, misc"

from axelera.compiler.alto import script as A
import axelera.compiler.alto.axol as axol
from axelera.compiler.alto.axol import intrinsic_registry
from axelera.compiler.alto.axol.intrinsic_registry import FunctionArgument, FunctionArgumentType
from axelera.compiler.alto.axol.common import sliced
from axelera.compiler.alto.compiler.device import PWORD
from axelera.compiler.alto.compiler.dtypes import MetisTypes
from axelera.compiler.alto.compiler.op_registry import Arg, BufferInfo, ScriptOp, register_op
from axelera.config import HardwareGeneration
import numpy as np
from tvm import tir

from ._rvv import kernel_source, to_scalar

_INTRINSIC_NAME = "rvv_imc_mm_quant"
_F18 = "float18x64"
_F32 = "float32x64"

intrinsic_registry.set(
    name=_INTRINSIC_NAME,
    arguments=[
        FunctionArgument(name=n, type=FunctionArgumentType.DLTENSOR)
        for n in ("a", "b", "qa", "qb", "sc")
    ],
    implementation=kernel_source("_imc_mm_quant"),
)


@sliced
def _imc_matmul(
    qa: tir.BufferSlice,
    imc_b: tir.BufferSlice,
    sc: tir.BufferSlice,
    out: tir.BufferSlice,
) -> None:
    """One (M, K) @ (K, N) int8 MVM per row, dequantized by sc's row and column scales.

    sc[m] is ONE contiguous run: the row scale (lane-broadcast, since the DPU has no
    scalar broadcast on this path) followed by the N column scales. It has to be one
    buffer -- port 1 delivers a single contiguous stream per row, and feeding it from two
    buffers misaligns it, which measured as 2.2e-01 against an expected 9.1e-03.
    """
    m_rows = qa.shape[0]
    k_blocks = imc_b.shape[0]
    n_blocks = imc_b.shape[1]

    with A.script():
        for mi in A.serial(m_rows):
            with A.Datapath(kind=A.DP_MVMEXE, id="imc_mm"):
                ifd = A.decl_stream("int8x64")
                scale_stream = A.decl_stream(_F32)
                matmul_stream = A.decl_stream("int32x64")
                dpu_stream = A.decl_stream(_F32)

                for n, r in A.grid(n_blocks, k_blocks):
                    with A.IFD(num=0):
                        ifd.push(qa[mi, r])
                for n in A.serial(n_blocks):
                    with A.pseudo.MatMul():
                        acc = A.mvm_accumulator()
                        A.reset_accumulator(acc)
                        for r in A.serial(0, k_blocks):
                            acc[0] += A.mvm_1x1(ifd.pop(), imc_b[r, n])
                        matmul_stream.push(acc[0])
                for c in A.serial(n_blocks + 1):
                    with A.IFD(num=1):
                        scale_stream.push(sc[mi, c])

                with A.DPU():
                    row = A.var(_F18)
                    val = A.var(_F18)
                    row = A.cast(scale_stream.pop(), _F18)
                    for n in A.serial(n_blocks):
                        val = A.cast(matmul_stream.pop(), _F18)
                        val = val * row
                        val = val * A.cast(scale_stream.pop(), _F18)
                        dpu_stream.push(A.cast(val, _F32))
                for n in A.serial(n_blocks):
                    with A.ODR():
                        out[0, mi, n] = dpu_stream.pop()


class ImcMmInt8(ScriptOp):
    """Batched int8 IMC matmul, quantized on RVV and dequantized on the DPU."""

    a = Arg(MetisTypes[tir.Buffer, "..."], description="(G, M, K/64) float32x64 activation.")
    b = Arg(
        MetisTypes[tir.Buffer, "..."],
        description="(G, K, N/64) float32x64 IMC operand, already in (K, N) layout.",
    )
    output = Arg(
        MetisTypes[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="(G, M, N/64) float32x64, or None to auto-allocate.",
    )
    core = Arg(int | None, default=None, kw=True, description="Optional AI core to run on.")
    generation = Arg(
        HardwareGeneration | None, default=None, kw=True, description="Hardware generation."
    )

    def _dims(self):
        g, m, kb = (int(s) for s in self.a.shape)
        _, k, nb = (int(s) for s in self.b.shape)
        return g, m, k, kb * PWORD, nb

    def validate(self) -> None:
        g, m, k, kk, nb = self._dims()
        if k != kk:
            raise ValueError(f"b's row count {k} must equal a's K = {kk}")
        if int(self.b.shape[0]) != g:
            raise ValueError(f"batch mismatch: a has {g}, b has {int(self.b.shape[0])}")
        for name in ("a", "b"):
            if getattr(self, name).dtype != _F32:
                raise ValueError(f"{name} must be float32x64, got {getattr(self, name).dtype}")

    def infer_outputs(self) -> dict[str, BufferInfo]:
        g, m, _, _, nb = self._dims()
        return {"output": BufferInfo(shape=(g, m, nb), dtype=_F32, scope=self.a.scope())}

    def implementation(self) -> None:
        g, m, k, kk, nb = self._dims()
        kb = kk // PWORD
        n = nb * PWORD

        a_s = A.unpack(self.a, (g, m, kk))
        b_s = A.unpack(self.b, (g, k, n))
        out_s = A.unpack(self.output, (g, m, n))

        # num_tiles=g forces one matmul per tile, so the IMC load and the MVM below see
        # exactly one (K, N) operand. The tile slice is still 3-D, so it is reshaped to
        # drop the unit batch axis before staging.
        with A.tiles(g, num_tiles=g) as tile:
            # Allocated at full extent and tiled, so the tiling pass sizes the actual L1
            # region; a fixed leading 1 mismatches the symbolic partial-tile extent.
            l1_a = A.alloc_buffer((g, m, kk), "float32", scope=A.L1)
            l1_b = A.alloc_buffer((g, k, n), "float32", scope=A.L1)
            # The quantized operands are allocated PACKED, because load_imc and the MVM
            # streams consume lane words; the extern gets scalar views of the same memory.
            qa_p = A.alloc_buffer((m, kb), "int8x64", scope=A.L1)
            qb_p = A.alloc_buffer((k, nb), "int8x64", scope=A.L1)
            sc_p = A.alloc_buffer((m, 1 + nb), _F32, scope=A.L1)
            out_p = A.alloc_buffer((g, m, nb), _F32, scope=A.L1)

            axol.atomic.copy(a_s.tile(tile, dim=0), output=l1_a.tile(tile, dim=0))
            axol.atomic.copy(b_s.tile(tile, dim=0), output=l1_b.tile(tile, dim=0))
            with A.Invoke(core=self.core):
                A.extern(
                    _INTRINSIC_NAME,
                    l1_a.tile(tile, dim=0),
                    l1_b.tile(tile, dim=0),
                    A.unpack(qa_p, (m, kk)),
                    A.unpack(qb_p, (k, n)),
                    A.unpack(sc_p, (m, (1 + nb) * PWORD)),
                )
            imc_b = A.alloc_imc((kb, nb))
            axol.atomic.load_imc(qb_p, imc_b)
            _imc_matmul(qa=qa_p, imc_b=imc_b, sc=sc_p, out=out_p.tile(tile, dim=0))
            # out_p stays PACKED for the datapath (it writes lane words); the copy back
            # goes through a scalar view so both sides of it agree.
            axol.atomic.copy(
                A.unpack(out_p, (g, m, n)).tile(tile, dim=0), output=out_s.tile(tile, dim=0)
            )

    def emulate(self) -> None:  # type: ignore[override]
        a = to_scalar(self.a).astype(np.float64)
        b = to_scalar(self.b).astype(np.float64)
        out = to_scalar(self.output)
        for gi in range(a.shape[0]):
            sa = np.maximum(np.abs(a[gi]).max(axis=1, keepdims=True), 1e-30) / 127.0
            sb = np.maximum(np.abs(b[gi]).max(axis=0, keepdims=True), 1e-30) / 127.0
            qa = np.rint(np.clip(a[gi] / sa, -127, 127))
            qb = np.rint(np.clip(b[gi] / sb, -127, 127))
            out[gi] = (qa @ qb) * (sa * sb)


imc_mm_int8 = register_op(
    ImcMmInt8,
    "llm.imc_mm_int8",
    generation=HardwareGeneration.EUROPA,
    atomic=False,
)

#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Int8 chunkwise gated delta rule (llm.gated_deltanet_chunk_quant).

The int8 sibling of llm.gated_deltanet_chunk: same chunkwise WY algorithm
and operand layout, but every dense chunk matmul is an int8 matmul (the shape
the IMC accelerates, M = chunk_size amortising the array load) while the
carried (Dk, Dv) state, the per-step gates, and the C×C
forward-substitution stay float.  Keeping the state float makes the int8
error chunk-local -- it does not compound through the recurrence -- so a long
prefill quantizes about as well as a single chunk.

emulate reuses _chunk_gated_delta_rule(..., quant=True), which models
each IMC matmul as symmetric per-row-A / per-col-B int8 with int32 accumulation
(see _qmm).  Measured error vs the float64 recurrence is ~1.5% relative L2,
flat from 64 to 512 tokens at the production dims (Dk=Dv=128, C=64).

This is the int8 numerical contract; the AxeleraScript IMC kernel
(implementation) is the follow-up, validated against this emulate.
"""
from axelera.compiler.alto.compiler.config import HardwareGeneration
from axelera.compiler.alto.compiler.op_registry import register_op
import numpy as np

from .gated_deltanet_chunk import GatedDeltaNetChunk, _chunk_gated_delta_rule

# mypy: disable-error-code="attr-defined, index, arg-type"
# flake8: noqa: C901


class GatedDeltaNetChunkQuant(GatedDeltaNetChunk):
    """Int8 chunkwise WY gated delta rule. See module docstring.

    Inherits the operands / validate / infer_outputs of GatedDeltaNetChunk;
    only the emulation differs (int8 dense matmuls).
    """

    def implementation(self) -> None:
        # Per-chunk int8 IMC kernel: quantize the dense matmul operands, run on
        # the IMC (M = chunk_size amortises the load), forward-substitution on
        # the DPU, carried state in float.
        raise NotImplementedError(
            "llm.gated_deltanet_chunk_quant: device kernel not implemented yet; "
            "the int8 chunkwise formulation is currently validated at the "
            "emulation level (interpret) -- see the module docstring."
        )

    def emulate(self) -> None:  # type: ignore[override]
        b, h, t, dk = (int(s) for s in self.query.shape[:4])
        pword = int(self.query.shape[-1])
        dv_blocks = int(self.value.shape[-2])
        dv = dv_blocks * pword

        q = self.query[..., 0].astype(np.float64)  # (B, H, T, Dk)
        k = self.key[..., 0].astype(np.float64)  # (B, H, T, Dk)
        v = self.value.astype(np.float64).reshape(b, h, t, dv)  # (B, H, T, Dv)
        alpha = self.alpha[..., 0, 0].astype(np.float64)  # (B, H, T)
        beta = self.beta[..., 0, 0].astype(np.float64)  # (B, H, T)
        s0 = self.state_in.astype(np.float64).reshape(b, h, dk, dv)  # (B, H, Dk, Dv)

        out, state_final = _chunk_gated_delta_rule(
            q, k, v, alpha, beta, s0, self.chunk_size, quant=True
        )

        np.copyto(self.output, out.reshape(self.output.shape))
        np.copyto(self.state_out, state_final.reshape(self.state_out.shape))


gated_deltanet_chunk_quant = register_op(
    GatedDeltaNetChunkQuant,
    "llm.gated_deltanet_chunk_quant",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)

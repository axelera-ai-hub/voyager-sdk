#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Int8 IMC chunkwise gated delta rule -- single chunk, zero initial state
(llm.gated_deltanet_chunk_imc).

Int8 IMC implementation of the chunkwise (WY) gated delta rule for one chunk with
a zero initial state::

    L     = -strict_lower((kβ @ kᵀ) ⊙ decay_mask)
    T     = (I - L)⁻¹  ≈ I + L + L² + … + L^NTERMS      (Neumann series)
    value = T @ (v ⊙ β)
    out   = ((q @ kᵀ) ⊙ decay_mask) @ value

Every dense matmul is an int8 IMC matmul (load_imc + MVMEXE + mvm_1x1). The
forward-substitution that a textbook WY transform would use is replaced by the
Neumann series, which is pure matmuls and avoids the lane-vs-scalar problem of
the sequential solve.

Layout (C = Dk = Dv = 64 = one lane block, no padding): every matmul is
(M,1) @ (1,C) -> (M,1) with K = C. The B operands need no transpose -- T /
value / vβ reshape (C,1) -> (1,C) (same memory), and kᵀ is supplied by the host.
Quantization uses a single fixed scale (invb = 1/scale, dqb = scale²) supplied
by the caller and applied identically here and in emulate (no on-device max-reduce).

Decay quantities (decay_mask, nsl = -strict_lower·decay_mask, imat = identity)
and kβ/vβ/kᵀ are precomputed by the host. The emulate is float18-aligned
(emu.rf18 at every dequant/elementwise step, int32 accumulator rounded to float18
before dequant, mirroring the DPU) so compile_and_run matches interpret bit-for-bit.
"""
from axelera.compiler.alto.axol import atomic
from axelera.compiler.alto.compiler.config import HardwareGeneration
from axelera.compiler.alto.compiler.device import PWORD
from axelera.compiler.alto.compiler.dtypes import MetisFloats
from axelera.compiler.alto.compiler.interpreter import emulation as emu
from axelera.compiler.alto.compiler.op_registry import Arg, BufferInfo, ScriptOp, register_op
import axelera.compiler.alto.script as A
import numpy as np
from tvm import tir

# mypy: disable-error-code="attr-defined, index, arg-type"
# flake8: noqa: C901


class GatedDeltaNetChunkImc(ScriptOp):
    """Single-chunk (zero-state) int8 IMC chunkwise gated delta rule. See module docstring."""

    q = Arg(MetisFloats[tir.Buffer, "..."], description="(C, 1) query rows (Dk in lanes).")
    kb = Arg(MetisFloats[tir.Buffer, "..."], description="(C, 1) k*beta rows.")
    knk = Arg(MetisFloats[tir.Buffer, "..."], description="(1, C) kᵀ in NK layout ([0,d]=k[:,d]).")
    vb = Arg(MetisFloats[tir.Buffer, "..."], description="(C, 1) v*beta rows (Dv in lanes).")
    dm = Arg(MetisFloats[tir.Buffer, "..."], description="(C, 1) decay_mask rows.")
    nsl = Arg(MetisFloats[tir.Buffer, "..."], description="(C, 1) -strict_lower(decay_mask) rows.")
    imat = Arg(MetisFloats[tir.Buffer, "..."], description="(C, 1) identity rows.")
    invb = Arg(MetisFloats[tir.Buffer, "..."], description="(1,) quantize scale 1/s.")
    dqb = Arg(MetisFloats[tir.Buffer, "..."], description="(1,) dequant scale s².")
    output = Arg(
        MetisFloats[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="Chunk readout, auto-inferred (C, 1) float32x64 DDR.",
    )
    nterms = Arg(
        int, default=8, kw=True, description="Neumann series length (I + L + … + L^nterms)."
    )
    generation = Arg(HardwareGeneration | None, default=None, kw=True)

    def _c(self) -> int:
        return int(self.q.shape[0])

    def validate(self) -> None:
        c = self._c()
        if c != PWORD:
            raise ValueError(f"this op currently supports C = Dk = Dv = {PWORD} only; got {c}.")
        for nm, buf, shape in (
            ("kb", self.kb, (c, 1)),
            ("vb", self.vb, (c, 1)),
            ("dm", self.dm, (c, 1)),
            ("nsl", self.nsl, (c, 1)),
            ("imat", self.imat, (c, 1)),
            ("knk", self.knk, (1, c)),
        ):
            if tuple(int(s) for s in buf.shape) != shape:
                raise ValueError(f"{nm} must be {shape}; got {tuple(buf.shape)}.")

    def infer_outputs(self) -> dict[str, BufferInfo]:
        return {"output": BufferInfo(shape=(self._c(), 1), dtype="float32x64", scope=A.DDR)}

    def _qmm(self, a, b, out) -> None:
        """fp a (M,1) @ fp b (1,C) -> fp out (M,1): fixed-scale int8 IMC."""
        c = self._c()
        i8 = "int8x64"
        f32 = "float32x64"
        f18 = "float18x64"
        m = int(a.shape[0])
        aq = A.alloc_buffer((m, 1), i8, scope=A.L1)
        bq = A.alloc_buffer((1, c), i8, scope=A.L1)
        imc = A.alloc_imc((1, 1))
        inv = self._inv
        dq = self._dq
        with A.script():
            with A.Datapath(core=None):  # quantize a
                fin = A.decl_stream(f32)
                qo = A.decl_stream(i8)
                for i in A.serial(m):
                    with A.IFD(num=0):
                        fin.push(a[i, 0])
                with A.DPU():
                    x = A.var(f18)
                    for i in A.serial(m):
                        x = A.cast(fin.pop(), f18)
                        x = x * A.cast(inv[0], f18)
                        qo.push(A.cast(x, i8))
                for i in A.serial(m):
                    with A.ODR():
                        aq[i, 0] = qo.pop()
            with A.Datapath(core=None):  # quantize b
                fin = A.decl_stream(f32)
                qo = A.decl_stream(i8)
                for kk in A.serial(c):
                    with A.IFD(num=0):
                        fin.push(b[0, kk])
                with A.DPU():
                    x = A.var(f18)
                    for kk in A.serial(c):
                        x = A.cast(fin.pop(), f18)
                        x = x * A.cast(inv[0], f18)
                        qo.push(A.cast(x, i8))
                for kk in A.serial(c):
                    with A.ODR():
                        bq[0, kk] = qo.pop()
            atomic.load_imc(buffer=bq, imc=imc, input_is_transposed=True)
            for i in A.serial(m):
                with A.Datapath(kind=A.DP_MVMEXE, core=None):
                    ifd0 = A.decl_stream(i8)
                    ms = A.decl_stream("int32x64")
                    do = A.decl_stream(f32)
                    with A.IFD(num=0):
                        ifd0.push(aq[i, 0])
                    with A.pseudo.MatMul():
                        acc = A.mvm_accumulator()
                        A.reset_accumulator(acc)
                        acc += A.mvm_1x1(ifd0.pop(), imc[0, 0])
                        ms.push(acc)
                    with A.DPU():
                        y = A.var(f18)
                        y = A.cast(ms.pop(), f18)
                        y = y * A.cast(dq[0], f18)
                        do.push(A.cast(y, f32))
                    with A.ODR():
                        out[i, 0] = do.pop()

    def _ew(self, a, b, out, op) -> None:
        """Elementwise a (op) b -> out, shape (M,1), op in {'mul', 'add'}."""
        f32 = "float32x64"
        f18 = "float18x64"
        m = int(a.shape[0])
        with A.script():
            with A.Datapath(core=None):
                fa = A.decl_stream(f32)
                fb = A.decl_stream(f32)
                o = A.decl_stream(f32)
                for i in A.serial(m):
                    with A.IFD(num=0):
                        fa.push(a[i, 0])
                for i in A.serial(m):
                    with A.IFD(num=1):
                        fb.push(b[i, 0])
                with A.DPU():
                    x = A.var(f18)
                    y = A.var(f18)
                    for i in A.serial(m):
                        x = A.cast(fa.pop(), f18)
                        y = A.cast(fb.pop(), f18)
                        if op == "mul":
                            x = x * y
                        else:
                            x = x + y
                        o.push(A.cast(x, f32))
                for i in A.serial(m):
                    with A.ODR():
                        out[i, 0] = o.pop()

    def _reshape_1c(self, buf):
        return A.alloc_buffer((1, self._c()), buf.dtype, scope=A.L1, data=buf.data)

    def implementation(self) -> None:
        c = self._c()
        f32 = "float32x64"
        self._inv = A.alloc_buffer((1,), f32, scope=A.L1)
        self._dq = A.alloc_buffer((1,), f32, scope=A.L1)
        atomic.copy(self.invb, output=self._inv)
        atomic.copy(self.dqb, output=self._dq)

        def L1():
            return A.alloc_buffer((c, 1), f32, scope=A.L1)

        q = L1()
        kb = L1()
        vb = L1()
        dm = L1()
        nsl = L1()
        imat = L1()
        knk = A.alloc_buffer((1, c), f32, scope=A.L1)
        atomic.copy(self.q, output=q)
        atomic.copy(self.kb, output=kb)
        atomic.copy(self.vb, output=vb)
        atomic.copy(self.dm, output=dm)
        atomic.copy(self.nsl, output=nsl)
        atomic.copy(self.imat, output=imat)
        atomic.copy(self.knk, output=knk)

        m1 = L1()
        self._qmm(kb, knk, m1)  # kβ @ kᵀ
        lmat = L1()
        self._ew(m1, nsl, lmat, "mul")  # L = -(kβ@kᵀ ⊙ decay_mask) strict-lower
        tmat = L1()
        atomic.copy(imat, output=tmat)  # T = I
        for _ in range(int(self.nterms)):
            tb = self._reshape_1c(tmat)
            tmp = L1()
            self._qmm(lmat, tb, tmp)  # L @ T
            tn = L1()
            self._ew(tmp, imat, tn, "add")  # I + L@T
            tmat = tn
        vbb = self._reshape_1c(vb)
        value = L1()
        self._qmm(tmat, vbb, value)  # T @ vβ
        m2 = L1()
        self._qmm(q, knk, m2)  # q @ kᵀ
        ai = L1()
        self._ew(m2, dm, ai, "mul")  # attn_intra
        valb = self._reshape_1c(value)
        out_l1 = L1()
        self._qmm(ai, valb, out_l1)  # attn_intra @ value
        atomic.copy(out_l1, output=self.output)

    def emulate(self) -> None:  # type: ignore[override]
        c = self._c()
        inv = np.float32(self.invb.reshape(-1)[0])
        dq = np.float32(self.dqb.reshape(-1)[0])

        inv18 = emu.rf18(inv)

        def q8(x):
            # Cast x->f18, multiply by f18(inv) in f18, then cast->int8 (mirrors the kernel).
            prod = emu.rf18(emu.rf18(x.astype(np.float32)) * inv18)
            return np.clip(np.round(prod), -127, 127)

        def qmm(a, b):
            acc = emu.rf18((q8(a) @ q8(b)).astype(np.float32))
            return emu.rf18(acc * dq)

        def mul(a, b):
            return emu.rf18(emu.rf18(a.astype(np.float32)) * emu.rf18(b.astype(np.float32)))

        def add(a, b):
            return emu.rf18(emu.rf18(a.astype(np.float32)) + emu.rf18(b.astype(np.float32)))

        q = self.q[:, 0, :c].astype(np.float64)
        kb = self.kb[:, 0, :c].astype(np.float64)
        kT = self.knk[0, :, :c].astype(np.float64)  # (C, C): [d, j] = k[j, d]
        vb = self.vb[:, 0, :c].astype(np.float64)
        dm = self.dm[:, 0, :c].astype(np.float64)
        nsl = self.nsl[:, 0, :c].astype(np.float64)
        ident = self.imat[:, 0, :c].astype(np.float64)

        lmat = mul(qmm(kb, kT), nsl)
        tmat = ident.copy()
        for _ in range(int(self.nterms)):
            tmat = add(qmm(lmat, tmat), ident)
        value = qmm(tmat, vb)
        ai = mul(qmm(q, kT), dm)
        out = qmm(ai, value)
        np.copyto(self.output, out.reshape(c, 1, PWORD).astype(np.float32))


gated_deltanet_chunk_imc = register_op(
    GatedDeltaNetChunkImc,
    "llm.gated_deltanet_chunk_imc",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)

#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Multi-head int8 IMC gated-delta decode-step readout (llm.gated_deltanet_decode_imc_multi).

One decode timestep for BH heads, walked by an A.grid HARDWARE loop (one generated body,
reused serially per head). The A.grid loop serialises the per-head IMC loads, which is
required because unrolled instances contend for the single physical IMC array and the
scheduler can overlap their load_imc calls, clobbering across heads.

Per head it computes the readout and the delta::

    kS = k @ S ;  qS = q @ S ;  qk = q . k
    v_new = beta * (v - alpha * kS)
    o     = alpha * qS + qk * v_new

and emits o + v_new per head. The rank-1 state update S_new = alpha * S + k (x) v_new is
applied by the caller in fp on the carried fp32 state (so the state never round-trips through
int8, avoiding compounding error; it also sidesteps an A.grid-incompatible in-op state
datapath where a register array held across the d-loop corrupts). Inputs carry a leading BH
axis; the state S is Sᵀ in NK int8 layout, the scalars folded per head.
"""
from axelera.compiler.alto.axol import atomic
from axelera.compiler.alto.axol.utils import to_tuple
from axelera.compiler.alto.compiler.config import HardwareGeneration
from axelera.compiler.alto.compiler.device import PWORD
from axelera.compiler.alto.compiler.dtypes import Int8x64, MetisFloats
from axelera.compiler.alto.compiler.op_registry import Arg, BufferInfo, ScriptOp, register_op
import axelera.compiler.alto.script as A
import axelera.compiler.backend.tir.ops as tir_ops
import numpy as np
from tvm import tir

# mypy: disable-error-code="attr-defined, index, arg-type"
# flake8: noqa: C901


class GatedDeltaNetDecodeImcMulti(ScriptOp):
    """BH-head int8-IMC gated-delta decode-step readout (A.grid over heads)."""

    q8 = Arg(Int8x64[tir.Buffer, "..."], description="(BH, 1, Dk_blocks) int8 query, Dk in lanes.")
    k8 = Arg(Int8x64[tir.Buffer, "..."], description="(BH, 1, Dk_blocks) int8 key, Dk in lanes.")
    s0nk = Arg(
        Int8x64[tir.Buffer, "..."],
        description="(BH, Dv_blocks, Dk) int8 state Sᵀ (NK: N=Dv in lanes, indexed by K=Dk).",
    )
    v = Arg(MetisFloats[tir.Buffer, "..."], description="(BH, 1, Dv_blocks) float value v_t.")
    scals = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(BH, 6) folded scalars [s_kS, s_qS, s_qk, alpha, beta, s_S] per head "
        "(only s_kS/s_qS/s_qk/alpha/beta are used; s_S is unused here).",
    )
    output = Arg(
        MetisFloats[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="Readout o_t per head, auto-inferred (BH, 1, Dv_blocks) float32x64 DDR.",
    )
    v_new = Arg(
        MetisFloats[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="The delta v_new = beta*(v - alpha*kS) per head, auto-inferred "
        "(BH, 1, Dv_blocks) float32x64 DDR. The rank-1 state update S_new = alpha*S + "
        "k (x) v_new is done by the caller in fp (the carried state stays fp32 -- no int8 "
        "compounding -- and it sidesteps an A.grid-incompatible in-op state datapath).",
    )
    generation = Arg(HardwareGeneration | None, default=None, kw=True)

    def _bh(self) -> int:
        return int(self.q8.shape[0])

    def _dims(self) -> tuple[int, int]:
        dk = int(self.q8.shape[2]) * PWORD
        dv = int(self.v.shape[2]) * PWORD
        return dk, dv

    def validate(self) -> None:
        bh = self._bh()
        dk, dv = self._dims()
        if int(self.k8.shape[0]) != bh or int(self.k8.shape[2]) * PWORD != dk:
            raise ValueError("q8 and k8 must share (BH, Dk).")
        if (
            int(self.s0nk.shape[0]) != bh
            or int(self.s0nk.shape[1]) * PWORD != dv
            or int(self.s0nk.shape[2]) != dk
        ):
            raise ValueError(f"s0nk must be (BH, Dv_blocks, Dk) = ({bh}, {dv // PWORD}, {dk}).")
        if int(self.scals.shape[0]) != bh or int(self.scals.shape[1]) != 6:
            raise ValueError("scals must be (BH, 6) [s_kS, s_qS, s_qk, alpha, beta, s_S].")

    def infer_outputs(self) -> dict[str, BufferInfo]:
        bh = self._bh()
        dk, dv = self._dims()
        return {
            "output": BufferInfo(shape=(bh, 1, dv // PWORD), dtype="float32x64", scope=A.DDR),
            "v_new": BufferInfo(shape=(bh, 1, dv // PWORD), dtype="float32x64", scope=A.DDR),
        }

    def implementation(self) -> None:
        i8 = "int8x64"
        f32 = "float32x64"
        f18 = "float18x64"
        bh_n = self._bh()
        dk, dv = self._dims()
        KB = dk // PWORD
        NB = dv // PWORD

        # One body reused serially per head: each IMC load completes before the next head's.
        with A.grid(bh_n) as idx:
            bh = to_tuple(idx)[0]
            q = A.alloc_buffer((1, KB), i8, scope=A.L1)
            k = A.alloc_buffer((1, KB), i8, scope=A.L1)
            snk = A.alloc_buffer((NB, dk), i8, scope=A.L1)
            vl = A.alloc_buffer((1, NB), f32, scope=A.L1)
            sc = A.alloc_buffer((6,), f32, scope=A.L1)
            kS = A.alloc_buffer((1, NB), f32, scope=A.L1)
            qS = A.alloc_buffer((1, NB), f32, scope=A.L1)
            qk = A.alloc_buffer((1,), f32, scope=A.L1)
            vnew = A.alloc_buffer((1, NB), f32, scope=A.L1)
            out_l1 = A.alloc_buffer((1, NB), f32, scope=A.L1)
            imc = A.alloc_imc((KB, NB))
            atomic.copy(self.q8[bh, :, :], output=q)
            atomic.copy(self.k8[bh, :, :], output=k)
            atomic.copy(self.s0nk[bh, :, :], output=snk)
            atomic.copy(self.v[bh, :, :], output=vl)
            atomic.copy(self.scals[bh, :], output=sc)
            atomic.load_imc(buffer=snk, imc=imc, input_is_transposed=True)

            with A.script():
                # kS = (k @ S) * s_kS  -- int8 IMC matvec (M=1)
                with A.Datapath(kind=A.DP_MVMEXE, core=None):
                    ifd0 = A.decl_stream(i8)
                    ms = A.decl_stream("int32x64")
                    do = A.decl_stream(f32)
                    for n in A.serial(NB):
                        for r in A.serial(KB):
                            with A.IFD(num=0):
                                ifd0.push(k[0, r])
                    for n in A.serial(NB):
                        with A.pseudo.MatMul():
                            acc = A.mvm_accumulator()
                            A.reset_accumulator(acc)
                            for r in A.serial(KB):
                                acc += A.mvm_1x1(ifd0.pop(), imc[r, n])
                            ms.push(acc)
                    with A.DPU():
                        y = A.var(f18)
                        for n in A.serial(NB):
                            y = A.cast(ms.pop(), f18)
                            y = y * A.cast(sc[0], f18)
                            do.push(A.cast(y, f32))
                    for n in A.serial(NB):
                        with A.ODR():
                            kS[0, n] = do.pop()
                # qS = (q @ S) * s_qS  -- reuses the same IMC load of S
                with A.Datapath(kind=A.DP_MVMEXE, core=None):
                    ifd0 = A.decl_stream(i8)
                    ms = A.decl_stream("int32x64")
                    do = A.decl_stream(f32)
                    for n in A.serial(NB):
                        for r in A.serial(KB):
                            with A.IFD(num=0):
                                ifd0.push(q[0, r])
                    for n in A.serial(NB):
                        with A.pseudo.MatMul():
                            acc = A.mvm_accumulator()
                            A.reset_accumulator(acc)
                            for r in A.serial(KB):
                                acc += A.mvm_1x1(ifd0.pop(), imc[r, n])
                            ms.push(acc)
                    with A.DPU():
                        y = A.var(f18)
                        for n in A.serial(NB):
                            y = A.cast(ms.pop(), f18)
                            y = y * A.cast(sc[1], f18)
                            do.push(A.cast(y, f32))
                    for n in A.serial(NB):
                        with A.ODR():
                            qS[0, n] = do.pop()
                # qk = reduce_add(q*k) * s_qk
                with A.Datapath(core=None):
                    fq = A.decl_stream(i8)
                    fk = A.decl_stream(i8)
                    o = A.decl_stream(f32)
                    for r in A.serial(KB):
                        with A.IFD(num=0):
                            fq.push(q[0, r])
                    for r in A.serial(KB):
                        with A.IFD(num=1):
                            fk.push(k[0, r])
                    with A.DPU():
                        acc = A.var(f18)
                        prod = A.var(f18)
                        aa = A.var(f18)
                        bb = A.var(f18)
                        aa = A.cast(fq.pop(), f18)
                        bb = A.cast(fk.pop(), f18)
                        acc = aa * bb
                        for r in A.serial(1, KB):
                            aa = A.cast(fq.pop(), f18)
                            bb = A.cast(fk.pop(), f18)
                            prod = aa * bb
                            acc = acc + prod
                        acc = tir_ops.dpu_reduce(acc, operation=tir_ops.DPU_REDUCE_OP.ADD)
                        acc = acc * A.cast(sc[2], f18)
                        o.push(A.cast(acc, f32))
                    with A.ODR():
                        qk[0] = o.pop()
                # v_new = beta * (v - alpha*kS)
                with A.Datapath(core=None):
                    fkS = A.decl_stream(f32)
                    fv = A.decl_stream(f32)
                    o = A.decl_stream(f32)
                    for n in A.serial(NB):
                        with A.IFD(num=0):
                            fkS.push(kS[0, n])
                    for n in A.serial(NB):
                        with A.IFD(num=1):
                            fv.push(vl[0, n])
                    with A.DPU():
                        kSv = A.var(f18)
                        vv = A.var(f18)
                        vo = A.var(f18)
                        vn = A.var(f18)
                        for n in A.serial(NB):
                            kSv = A.cast(fkS.pop(), f18)
                            vo = kSv * A.cast(sc[3], f18)
                            vv = A.cast(fv.pop(), f18)
                            vn = vv - vo
                            vn = vn * A.cast(sc[4], f18)
                            o.push(A.cast(vn, f32))
                    for n in A.serial(NB):
                        with A.ODR():
                            vnew[0, n] = o.pop()
                # o = alpha*qS + qk*v_new
                with A.Datapath(core=None):
                    fqS = A.decl_stream(f32)
                    fvn = A.decl_stream(f32)
                    o = A.decl_stream(f32)
                    for n in A.serial(NB):
                        with A.IFD(num=0):
                            fqS.push(qS[0, n])
                    for n in A.serial(NB):
                        with A.IFD(num=1):
                            fvn.push(vnew[0, n])
                    with A.DPU():
                        qSv = A.var(f18)
                        vnv = A.var(f18)
                        res = A.var(f18)
                        tmp = A.var(f18)
                        for n in A.serial(NB):
                            qSv = A.cast(fqS.pop(), f18)
                            res = qSv * A.cast(sc[3], f18)
                            vnv = A.cast(fvn.pop(), f18)
                            tmp = A.cast(qk[0], f18) * vnv
                            res = res + tmp
                            o.push(A.cast(res, f32))
                    for n in A.serial(NB):
                        with A.ODR():
                            out_l1[0, n] = o.pop()
            # Emit readout o + the delta v_new; the caller applies the rank-1 state update.
            atomic.copy(out_l1, output=self.output[bh, :, :])
            atomic.copy(vnew, output=self.v_new[bh, :, :])

    def emulate(self) -> None:  # type: ignore[override]
        bh_n = self._bh()
        dk, dv = self._dims()
        NB = dv // PWORD
        for bh in range(bh_n):
            q8 = self.q8[bh, 0].astype(np.float64).reshape(-1)[:dk]
            k8 = self.k8[bh, 0].astype(np.float64).reshape(-1)[:dk]
            snk = self.s0nk[bh].astype(np.float64)  # (NB, Dk, PWORD)
            S = np.concatenate([snk[nb] for nb in range(NB)], axis=1)  # (Dk, Dv)
            v = self.v[bh].astype(np.float64).reshape(-1)[:dv]
            s_kS, s_qS, s_qk, alpha, beta, _s_S = (
                float(x) for x in self.scals[bh].reshape(-1)[::PWORD][:6]
            )
            kS = (k8 @ S) * s_kS
            qS = (q8 @ S) * s_qS
            qk = float(q8 @ k8) * s_qk
            v_new = beta * (v - alpha * kS)
            o = alpha * qS + qk * v_new
            np.copyto(self.output[bh], o.reshape(1, NB, PWORD).astype(np.float32))
            np.copyto(self.v_new[bh], v_new.reshape(1, NB, PWORD).astype(np.float32))


gated_deltanet_decode_imc_multi = register_op(
    GatedDeltaNetDecodeImcMulti,
    "llm.gated_deltanet_decode_imc_multi",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)

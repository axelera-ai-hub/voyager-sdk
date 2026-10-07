#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Int8 IMC gated-delta decode-step readout (llm.gated_deltanet_decode_imc).

One decode timestep of the gated delta rule with int8 IMC: the two state
matrix-vector products are int8 matmuls against the recurrent state S loaded once
into the in-memory-compute array (both k@S and q@S reuse the same load)::

    kS = k @ S ;  qS = q @ S ;  qk = q . k
    v_new = beta * (v - alpha * kS)
    o     = alpha * qS + qk * v_new

(equivalently o = q @ (alpha*S + outer(k, v_new))). Inputs are pre-quantized int8
(q8/k8, state S as Sᵀ in NK int8 layout) plus the broadcast fp key kbcast (for the
rank-1 write) and the folded dequant/gate scalars (incl. the state scale s_S); v is
float. Emits both the readout o and the updated state state_out = alpha*S + outer(k,
v_new) in NK layout (fp32). Quantization scales are provided by the caller (fixed per
invocation).
"""
from axelera.compiler.alto.axol import atomic
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


class GatedDeltaNetDecodeImc(ScriptOp):
    """One int8-IMC gated-delta decode-step readout. See module docstring."""

    q8 = Arg(Int8x64[tir.Buffer, "..."], description="(1, Dk_blocks) int8 query, Dk in lanes.")
    k8 = Arg(Int8x64[tir.Buffer, "..."], description="(1, Dk_blocks) int8 key, Dk in lanes.")
    s0nk = Arg(
        Int8x64[tir.Buffer, "..."],
        description="(Dv_blocks, Dk) int8 state Sᵀ (NK layout: N=Dv in lanes, indexed by K=Dk).",
    )
    v = Arg(MetisFloats[tir.Buffer, "..."], description="(1, Dv_blocks) float value v_t.")
    kbcast = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(Dk, 1) float key broadcast per lane ([d] = k_t[d]); for the rank-1 state write.",
    )
    scals = Arg(
        MetisFloats[tir.Buffer, "..."],
        description="(6,) folded scalars [s_kS, s_qS, s_qk, alpha, beta, s_S] (broadcast per lane).",
    )
    output = Arg(
        MetisFloats[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="Readout o_t, auto-inferred (1, Dv_blocks) float32x64 DDR.",
    )
    state_out = Arg(
        MetisFloats[tir.Buffer, "..."] | BufferInfo,
        default=None,
        is_output=True,
        kw=True,
        description="Updated state Sᵀ (NK), auto-inferred (Dv_blocks, Dk) float32x64 DDR.",
    )
    generation = Arg(HardwareGeneration | None, default=None, kw=True)

    def _dims(self) -> tuple[int, int]:
        dk = int(self.q8.shape[1]) * PWORD
        dv = int(self.v.shape[1]) * PWORD
        return dk, dv

    def validate(self) -> None:
        dk, dv = self._dims()
        if int(self.k8.shape[1]) * PWORD != dk:
            raise ValueError("q8 and k8 must share Dk.")
        if int(self.s0nk.shape[0]) * PWORD != dv or int(self.s0nk.shape[1]) != dk:
            raise ValueError(f"s0nk must be (Dv_blocks, Dk) = ({dv // PWORD}, {dk}).")
        if int(self.kbcast.shape[0]) != dk:
            raise ValueError(f"kbcast must be (Dk, 1) = ({dk}, 1).")
        if int(self.scals.shape[0]) != 6:
            raise ValueError("scals must have 6 entries [s_kS, s_qS, s_qk, alpha, beta, s_S].")

    def infer_outputs(self) -> dict[str, BufferInfo]:
        dk, dv = self._dims()
        return {
            "output": BufferInfo(shape=(1, dv // PWORD), dtype="float32x64", scope=A.DDR),
            "state_out": BufferInfo(shape=(dv // PWORD, dk), dtype="float32x64", scope=A.DDR),
        }

    def implementation(self) -> None:
        i8 = "int8x64"
        f32 = "float32x64"
        f18 = "float18x64"
        dk, dv = self._dims()
        KB = dk // PWORD
        NB = dv // PWORD
        q = A.alloc_buffer((1, KB), i8, scope=A.L1)
        k = A.alloc_buffer((1, KB), i8, scope=A.L1)
        snk = A.alloc_buffer((NB, dk), i8, scope=A.L1)
        vl = A.alloc_buffer((1, NB), f32, scope=A.L1)
        kbc = A.alloc_buffer((dk, 1), f32, scope=A.L1)
        self._sc = A.alloc_buffer((6,), f32, scope=A.L1)
        kS = A.alloc_buffer((1, NB), f32, scope=A.L1)
        qS = A.alloc_buffer((1, NB), f32, scope=A.L1)
        qk = A.alloc_buffer((1,), f32, scope=A.L1)
        vnew = A.alloc_buffer((1, NB), f32, scope=A.L1)
        out_l1 = A.alloc_buffer((1, NB), f32, scope=A.L1)
        state_l1 = A.alloc_buffer((NB, dk), f32, scope=A.L1)
        imc = A.alloc_imc((KB, NB))
        atomic.copy(self.q8, output=q)
        atomic.copy(self.k8, output=k)
        atomic.copy(self.s0nk, output=snk)
        atomic.copy(self.v, output=vl)
        atomic.copy(self.kbcast, output=kbc)
        atomic.copy(self.scals, output=self._sc)
        atomic.load_imc(buffer=snk, imc=imc, input_is_transposed=True)
        sc = self._sc

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
            # state update: S_new[d, :] = alpha*S_old[d, :] + k[d]*v_new[:], written
            # in NK layout state_out[vb, d].  IFD0 = old state snk; IFD1 carries
            # v_new (NB, held in registers) then the broadcast key (Dk).
            with A.Datapath(core=None):
                fs = A.decl_stream(i8)
                f1 = A.decl_stream(f32)
                o = A.decl_stream(f32)
                for d in A.serial(dk):
                    for n in A.serial(NB):
                        with A.IFD(num=0):
                            fs.push(snk[n, d])
                for n in A.serial(NB):
                    with A.IFD(num=1):
                        f1.push(vnew[0, n])
                for d in A.serial(dk):
                    with A.IFD(num=1):
                        f1.push(kbc[d, 0])
                with A.DPU():
                    vn = A.array(f18, NB)
                    kd = A.var(f18)
                    sv = A.var(f18)
                    rr = A.var(f18)
                    for n in A.serial(NB):
                        vn[n] = A.cast(f1.pop(), f18)
                    for d in A.serial(dk):
                        kd = A.cast(f1.pop(), f18)
                        for n in A.serial(NB):
                            sv = A.cast(fs.pop(), f18)
                            sv = sv * A.cast(sc[5], f18)  # dequant: * s_S
                            sv = sv * A.cast(sc[3], f18)  # decay: * alpha
                            rr = kd * vn[n]  # k[d] * v_new
                            rr = rr + sv
                            o.push(A.cast(rr, f32))
                for d in A.serial(dk):
                    for n in A.serial(NB):
                        with A.ODR():
                            state_l1[n, d] = o.pop()
        atomic.copy(out_l1, output=self.output)
        atomic.copy(state_l1, output=self.state_out)

    def emulate(self) -> None:  # type: ignore[override]
        dk, dv = self._dims()
        NB = dv // PWORD
        q8 = self.q8[0].astype(np.float64).reshape(-1)[:dk]  # (Dk,)
        k8 = self.k8[0].astype(np.float64).reshape(-1)[:dk]
        # s0nk[nb, d, vlane] = S[d, nb*PWORD + vlane]  ->  reconstruct S (Dk, Dv).
        snk = self.s0nk.astype(np.float64)  # (NB, Dk, PWORD)
        S = np.concatenate([snk[nb] for nb in range(NB)], axis=1)  # (Dk, Dv)
        v = self.v.astype(np.float64).reshape(-1)[:dv]
        kbc = self.kbcast[:, 0, 0].astype(np.float64)  # (Dk,) fp key (lane 0 = broadcast value)
        s_kS, s_qS, s_qk, alpha, beta, s_S = (
            float(x) for x in self.scals.reshape(-1)[::PWORD][:6]
        )
        kS = (k8 @ S) * s_kS
        qS = (q8 @ S) * s_qS
        qk = float(q8 @ k8) * s_qk
        v_new = beta * (v - alpha * kS)
        o = alpha * qS + qk * v_new
        np.copyto(self.output, o.reshape(1, NB, PWORD).astype(np.float32))
        # state update: S_new = alpha * (S * s_S) + outer(k, v_new), in NK layout.
        s_new = alpha * (S * s_S) + np.outer(kbc, v_new)  # (Dk, Dv)
        state_nk = np.stack([s_new[:, nb * PWORD : (nb + 1) * PWORD] for nb in range(NB)], axis=0)
        np.copyto(self.state_out, state_nk.astype(np.float32))  # (NB, Dk, PWORD)


gated_deltanet_decode_imc = register_op(
    GatedDeltaNetDecodeImc,
    "llm.gated_deltanet_decode_imc",
    generation=[HardwareGeneration.OMEGA, HardwareGeneration.EUROPA],
    atomic=False,
)

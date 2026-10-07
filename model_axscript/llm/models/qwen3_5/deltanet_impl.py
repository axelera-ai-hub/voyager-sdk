#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Metis implementation for the Qwen3.5 hybrid decoder.

Qwen3.5 is a 3:1 hybrid stack (config.layer_type(i)): linear_attention
gated-DeltaNet layers and a periodic full_attention layer. This module wires
the gated-DeltaNet linear-attention layer and registers a two-kind
Implementation via the per-kind LayerKernel infra.

make_deltanet_layer_kernel assembles the device-validated token mixer: normed
hidden -> quantized in_proj_qkv IMC -> causal_conv1d -> silu -> split q/k/v ->
head-major layout -> {L2-norm q/k, quantized z/b/a gates via lane_select,
beta = sigmoid(b), g = softplus(a + dt_bias) * -exp(A_log), fp18 sequential
recurrence (gated_deltanet_recurrent_fp), gated RMSNorm tail with eps = D*rms_eps}
-> out_proj IMC. All five GEMMs are W8A8 (gate quant adds ~1.3% to the output).

make_deltanet_layer_kernel is a full conforming decoder layer (threaded
hidden_states/residual_in -> output: input_layernorm -> deltanet mixer ->
+residual -> post_attn_norm -> SwiGLU MLP -> +residual) for BOTH phases: PREFILL
(chunked fp18 scan) and DECODE (single-token fp readout + rank-1 state update).
recurrent_state + conv_state are in-place persistent cache IO (the KV-cache
pattern; prefill via state_inplace=, conv via causal_conv1d).
deltanet_layer_buffers binds every constant for both phases (decode adds the
fixed calibrated requant scales sq/sk/sS + posA = exp(A_log)).

The companion full_attention kind (gated dense attention layer + query/gate split +
partial RoPE + sigmoid output gate) is make_full_attention_layer_kernel. The impl
is registered as qwen3.5-metis.
"""
# flake8: noqa: F821  -- A.Buffer dtype-string annotations look like F821 names
from typing import TYPE_CHECKING

from ...implementation import (
    DecoderLayerImplementation,
    HostKernel,
    LayerKernel,
    register_implementation,
)

if TYPE_CHECKING:
    from .qwen3_5 import Qwen3_5Config


# linear_attention (gated DeltaNet) layer kernel
def make_deltanet_layer_kernel(
    config: "Qwen3_5Config",
    ctx_len: int,
    phase: str,
    debug: bool = False,
    chunk_len: int | None = None,
):
    """Build the gated-DeltaNet linear-attention token-mixer IRModule.

    Assembles the device-validated pipeline: normed hidden -> quantized in_proj_qkv
    IMC -> causal_conv1d -> silu -> split q/k/v -> head-major layout -> {L2-norm q/k,
    quantized z/b/a gates via lane_select, beta/g, fp18 sequential recurrence,
    gated RMSNorm} -> out_proj IMC.

    Boundary is the torch GatedDeltaNet.forward (in: pre-mixer hidden, out:
    out_proj(core)); the input RMSNorm, residual add and SwiGLU MLP that wrap
    it follow the qwen3 pattern and are composed by the per-kind decoder infra.

    Built as a single un-wrapped prim_func (no A.block("root") wrapper) because
    rms_norm_kahan owns the root block. All five projections are W8A8
    (gate quant adds ~1.3% to the output).
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol.common import BufferInfo

    from ...ops.causal_conv1d import causal_conv1d
    from ...ops.elementwise_mul import elementwise_mul
    from ...ops.fused_gate_up_swiglu import fused_gate_up_swiglu
    from ...ops.gated_deltanet_recurrent_fp import gated_deltanet_recurrent_fp
    from ...ops.lane_select import lane_select
    from ...ops.pointwise_lut import pointwise_lut
    from ...ops.quantized_matmul_activation import quantized_matmul_activation
    from ..lfm2.optimized_impl import _swiglu_mlp_tail

    PWORD = 64
    BH = config.linear_num_value_heads
    if config.linear_num_key_heads != BH:
        raise NotImplementedError(
            "make_deltanet_layer_kernel: GQA (num_key_heads != num_value_heads) "
            "needs query/key head repeat_interleave; not yet wired."
        )
    D = config.linear_key_head_dim  # = value_head_dim; also the chunk length C
    C = D
    if C % PWORD != 0:
        raise ValueError(f"linear_key_head_dim must be a multiple of {PWORD}; got {C}")
    NB = D // PWORD
    HID = config.hidden_size
    KEY = BH * D
    VAL = BH * D
    CONV = 2 * KEY + VAL
    Kc = config.linear_conv_kernel_dim
    HB, CB, VB, KB = HID // PWORD, CONV // PWORD, VAL // PWORD, KEY // PWORD
    intermediate_dim = config.intermediate_size
    intermediate_dim_pw = intermediate_dim // PWORD
    NL, NRMS = 4, 2

    def quant(x, qs, qz):
        return axol.atomic.dpu_cast(
            axol.atomic.add(axol.atomic.mul(x, qs), qz), output=BufferInfo(dtype="int8x64")
        )

    def quant_sym(x, inv):  # symmetric (qz=0) int8 quant: round(x / s) via mul by 1/s
        return axol.atomic.dpu_cast(axol.atomic.mul(x, inv), output=BufferInfo(dtype="int8x64"))

    def l2norm(
        x_2d, lut, w, eps_buf
    ):  # (M, NB) -> L2-normed rows (rms_norm over D, weight 1/sqrt(D))
        M = int(x_2d.shape[0])
        n = axol.core.rms_norm_kahan(
            axol.atomic.reshape(x_2d, shape=(1, M, NB)),
            lut,
            eps_buf[0:1],
            (D,),
            weight_buffer=w,
            invsqrt_method="additive",
        )
        return axol.atomic.reshape(n, shape=(M, NB))

    def _build_decode_kernel():
        # Single-token gated-DeltaNet decode: in_proj -> conv with state carry -> gates
        # (beta=sigmoid(b), alpha=exp(g)) -> per head: L2-norm q/k (unscaled, d^-0.5
        # washes out via the gated norm), fp readout + rank-1 state update, write the
        # state back -> gated RMSNorm -> out_proj -> residual + SwiGLU MLP.
        # The carried recurrent_state holds S as (Dk, Dv_blk); the readout transposes
        # the two non-lane axes per Dv-block via a pword-grid transpose, generalising
        # to head_dim==128 (Dv_blk>1).
        def to_heads_dec(x):  # (1, 1, BH*NB) -> (BH, NB) head-major rows (one token)
            return axol.atomic.reshape(x, shape=(BH, NB))

        def from_heads_dec(core):  # (BH, NB) -> (1, 1, BH*NB)
            return axol.atomic.reshape(core, shape=(1, 1, VB))

        def per_row(vec_bh, dst):  # (1, NB) row of k/q -> dst (Dk,1): vec[dk] per row
            # kbc[dk] = scalar k[dk] broadcast across all 64 lanes. vector_transpose spreads
            # one Dv-block's 64 lanes across 64 rows, so build it a block at a time.
            for blk in range(NB):
                vb = axol.atomic.reshape(axol.atomic.copy(vec_bh[:, blk : blk + 1]), shape=(1, 1))
                axol.atomic.copy(
                    axol.atomic.vector_transpose(axol.atomic.repeat(vb, (PWORD, 1)), 0, 1),
                    output=dst[blk * PWORD : (blk + 1) * PWORD, :],
                )

        def grid_sum(x):  # (rows, W) -> (1, W); rows a power of 2 (pairwise-add tree)
            rows = int(x.shape[0])
            while rows > 1:
                h = rows // 2
                x = axol.atomic.add(axol.atomic.copy(x[0:h, :]), axol.atomic.copy(x[h:rows, :]))
                rows = h
            return x

        @A.ir_module
        class DeltaNetDecodeLayer:
            @A.prim_func
            def main(
                hidden_states: A.Buffer((1, 1, HB), "float32x64", scope=A.DDR),
                residual_in: A.Buffer((1, 1, HB), "float32x64", scope=A.DDR),
                rms1_w: A.Buffer((HID,), "float32x64", scope=A.DDR),
                rms1_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
                qs_h: A.Buffer((1, 1, HB), "float32x64", scope=A.DDR),
                qz_h: A.Buffer((1, 1, HB), "float32x64", scope=A.DDR),
                w_qkv: A.Buffer((CB, HID), "int8x64", scope=A.DDR),
                dq_qkv: A.Buffer((1, 1, 2 * CB), "float32x64", scope=A.DDR),
                w_z: A.Buffer((VB, HID), "int8x64", scope=A.DDR),
                dq_z: A.Buffer((1, 1, 2 * VB), "float32x64", scope=A.DDR),
                w_b: A.Buffer((1, HID), "int8x64", scope=A.DDR),
                dq_b: A.Buffer((1, 1, 2), "float32x64", scope=A.DDR),
                w_a: A.Buffer((1, HID), "int8x64", scope=A.DDR),
                dq_a: A.Buffer((1, 1, 2), "float32x64", scope=A.DDR),
                conv_w: A.Buffer((Kc, CB), "float32x64", scope=A.DDR),
                conv_state: A.Buffer((1, Kc - 1, CB), "float32x64", scope=A.DDR),
                onehots: A.Buffer((BH, 1), "float32x64", scope=A.DDR),
                dtb_all: A.Buffer((BH, 1), "float32x64", scope=A.DDR),
                posA_all: A.Buffer((BH, 1), "float32x64", scope=A.DDR),
                sig_l: A.Buffer((NL,), "float32x64", scope=A.DDR),
                sp_l: A.Buffer((NL,), "float32x64", scope=A.DDR),
                e_l: A.Buffer((NL,), "float32x64", scope=A.DDR),
                silu_l: A.Buffer((NL,), "float32x64", scope=A.DDR),
                l2_l: A.Buffer((NRMS,), "float32x64", scope=A.DDR),
                gn_l: A.Buffer((NRMS,), "float32x64", scope=A.DDR),
                l2w: A.Buffer((D,), "float32x64", scope=A.DDR),
                gnw: A.Buffer((D,), "float32x64", scope=A.DDR),
                eps: A.Buffer((1,), "float32x64", scope=A.DDR),
                epsg: A.Buffer((1,), "float32x64", scope=A.DDR),
                recurrent_state: A.Buffer((BH, C, NB), "float32x64", scope=A.DDR),
                sqi: A.Buffer((1,), "float32x64", scope=A.DDR),
                ski: A.Buffer((1,), "float32x64", scope=A.DDR),
                sSi: A.Buffer((1,), "float32x64", scope=A.DDR),
                scfix_bh: A.Buffer((BH, 6), "float32x64", scope=A.DDR),
                zerocv: A.Buffer((1, CB), "float32x64", scope=A.DDR),
                zerob: A.Buffer((BH, 1), "float32x64", scope=A.DDR),
                zerog: A.Buffer((BH, NB), "float32x64", scope=A.DDR),
                one: A.Buffer((1,), "float32x64", scope=A.DDR),
                qs_core: A.Buffer((1, 1, VB), "float32x64", scope=A.DDR),
                qz_core: A.Buffer((1, 1, VB), "float32x64", scope=A.DDR),
                w_oproj: A.Buffer((HB, VAL), "int8x64", scope=A.DDR),
                dq_oproj: A.Buffer((1, 1, 2 * HB), "float32x64", scope=A.DDR),
                rms2_w: A.Buffer((HID,), "float32x64", scope=A.DDR),
                rms2_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
                gate_proj_qs: A.Buffer((1, 1, HB), "float32x64", scope=A.DDR),
                gate_proj_qz: A.Buffer((1, 1, HB), "float32x64", scope=A.DDR),
                silu_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
                gate_up_proj_w: A.Buffer((2 * intermediate_dim_pw, HID), "int8x64", scope=A.DDR),
                mlp_dq_params: A.Buffer(
                    (1, 1, 6 * intermediate_dim_pw), "float32x64", scope=A.DDR
                ),
                down_proj_w: A.Buffer((HB, intermediate_dim), "int8x64", scope=A.DDR),
                down_dq_params: A.Buffer((1, 1, 2 * HB), "float32x64", scope=A.DDR),
            ):  # noqa: F821
                eps_const = A.alloc_constant(
                    [9.9999999747524271e-07] * 64, dtype="float32x64", scope=A.DDR
                )
                rms1 = axol.core.rms_norm_kahan(
                    hidden_states,
                    rms1_lut,
                    eps_const[0:1],
                    (HID,),
                    weight_buffer=rms1_w,
                    invsqrt_method="additive",
                )
                hq = quant(rms1, qs_h, qz_h)
                mqkv = quantized_matmul_activation(
                    hq, w_qkv, dq_qkv, output=BufferInfo(dtype="float32x64")
                )
                conv = causal_conv1d(mqkv, conv_w, conv_state, zero_initial_state=False)
                cs = pointwise_lut(axol.atomic.reshape(conv, shape=(1, CB)), zerocv, silu_l, one)
                cs3 = axol.atomic.reshape(cs, shape=(1, 1, CB))
                q_h = to_heads_dec(axol.atomic.copy(cs3[:, :, 0:KB]))  # (BH, NB)
                k_h = to_heads_dec(axol.atomic.copy(cs3[:, :, KB : 2 * KB]))
                v_h = to_heads_dec(axol.atomic.copy(cs3[:, :, 2 * KB : 3 * KB]))

                z_all = to_heads_dec(
                    quantized_matmul_activation(
                        hq, w_z, dq_z, output=BufferInfo(dtype="float32x64")
                    )
                )
                b_l = axol.atomic.reshape(
                    quantized_matmul_activation(
                        hq, w_b, dq_b, output=BufferInfo(dtype="float32x64")
                    ),
                    shape=(1, 1),
                )
                a_l = axol.atomic.reshape(
                    quantized_matmul_activation(
                        hq, w_a, dq_a, output=BufferInfo(dtype="float32x64")
                    ),
                    shape=(1, 1),
                )
                b_all = A.alloc_buffer((BH, 1), "float32x64", scope=A.DDR)
                a_all = A.alloc_buffer((BH, 1), "float32x64", scope=A.DDR)
                for h in range(BH):
                    oh = axol.atomic.reshape(axol.atomic.copy(onehots[h : h + 1, :]), shape=(1,))
                    # b_all[h:h+1, :] has a size-1 leading dim squeezed to (1,), so reshape
                    # the (1, 1) lane_select result to (1,) to match.
                    axol.atomic.copy(
                        axol.atomic.reshape(lane_select(b_l, oh), shape=(1,)),
                        output=b_all[h : h + 1, 0],
                    )
                    axol.atomic.copy(
                        axol.atomic.reshape(lane_select(a_l, oh), shape=(1,)),
                        output=a_all[h : h + 1, 0],
                    )

                # gates: beta = sigmoid(b); neg_g = posA*softplus(a+dt_bias) (the
                # per-head posA=exp(A_log) is a separate elementwise_mul, since
                # pointwise_lut's post is a scalar); alpha = exp(-neg_g) via e_l.
                beta_all = pointwise_lut(b_all, zerob, sig_l, one)  # (BH, 1)
                neg_g = elementwise_mul(pointwise_lut(a_all, dtb_all, sp_l, one), posA_all)
                alpha_all = pointwise_lut(neg_g, zerob, e_l, one)  # (BH, 1)

                # L2-norm q/k BATCHED over all heads (one rms_norm_kahan each): calling
                # it per head corrupts via device scratch aliasing.
                ql_all = l2norm(q_h, l2_l, l2w, eps)  # (BH, NB)
                kl_all = l2norm(k_h, l2_l, l2w, eps)

                # FP readout + rank-1 state update per head. The decode matvecs are tiny
                # (Dk=Dv=128, a single token), so the readout is done in fp on the DPU.
                # kbc/qbc = k/q broadcast "per row" (Dk,1): kbc[dk] = scalar k[dk] across
                # all 64 lanes, built by a 64x64 vector_transpose per Dv-block. Each readout is
                # a grid-axis sum of an elementwise product:
                #   kS = Σ_Dk(S ⊙ kbc),  qS = Σ_Dk(S ⊙ qbc),  qk = Σ_Dk(qbc ⊙ kbc)
                # grid_sum reduces the Dk rows by a pairwise-add tree (Dk is a power of 2).
                # (per_row / grid_sum are defined above, at the kernel scope.)
                kbc = A.alloc_buffer((D, 1), "float32x64", scope=A.DDR)
                qbc = A.alloc_buffer((D, 1), "float32x64", scope=A.DDR)
                o_all = A.alloc_buffer((BH, NB), "float32x64", scope=A.DDR)
                for bh in range(BH):
                    s_bh = axol.atomic.copy(recurrent_state[bh, :, :])  # (Dk, NB) entry state
                    kl = axol.atomic.reshape(
                        axol.atomic.copy(kl_all[bh : bh + 1, :]), shape=(1, NB)
                    )
                    ql = axol.atomic.reshape(
                        axol.atomic.copy(ql_all[bh : bh + 1, :]), shape=(1, NB)
                    )
                    per_row(kl, kbc)
                    per_row(ql, qbc)
                    alpha_bh = axol.atomic.reshape(
                        axol.atomic.copy(alpha_all[bh : bh + 1, :]), shape=(1, 1)
                    )
                    beta_bh = axol.atomic.reshape(
                        axol.atomic.copy(beta_all[bh : bh + 1, :]), shape=(1, 1)
                    )
                    kS = grid_sum(elementwise_mul(s_bh, kbc))  # (1, NB)
                    qS = grid_sum(elementwise_mul(s_bh, qbc))  # (1, NB)
                    qk = grid_sum(elementwise_mul(qbc, kbc))  # (1, 1)
                    v_bh = axol.atomic.reshape(
                        axol.atomic.copy(v_h[bh : bh + 1, :]), shape=(1, NB)
                    )
                    # v_new = beta*(v - alpha*kS);  o = alpha*qS + qk*v_new
                    v_new = axol.atomic.mul(
                        axol.atomic.sub(v_bh, axol.atomic.mul(kS, alpha_bh)), beta_bh
                    )
                    o_bh = axol.atomic.add(
                        axol.atomic.mul(qS, alpha_bh), axol.atomic.mul(v_new, qk)
                    )
                    axol.atomic.copy(
                        axol.atomic.reshape(o_bh, shape=(NB,)), output=o_all[bh : bh + 1, :]
                    )
                    # rank-1 fp state update: S_new = alpha*S + k (x) v_new.
                    outer = elementwise_mul(axol.atomic.repeat(v_new, (D, 1)), kbc)  # (Dk, NB)
                    aS = axol.atomic.mul(s_bh, alpha_bh)
                    axol.atomic.copy(axol.atomic.add(aS, outer), output=recurrent_state[bh, :, :])

                rr = axol.atomic.reshape(
                    axol.core.rms_norm_kahan(
                        axol.atomic.reshape(o_all, shape=(1, BH, NB)),
                        gn_l,
                        epsg[0:1],
                        (D,),
                        weight_buffer=gnw,
                        invsqrt_method="additive",
                    ),
                    shape=(BH, NB),
                )
                core = elementwise_mul(rr, pointwise_lut(z_all, zerog, silu_l, one))
                out = quantized_matmul_activation(
                    quant(from_heads_dec(core), qs_core, qz_core),
                    w_oproj,
                    dq_oproj,
                    output=BufferInfo(dtype="float32x64"),
                )
                residual = axol.atomic.add(residual_in, out)
                _tail = _swiglu_mlp_tail(
                    axol,
                    BufferInfo,
                    fused_gate_up_swiglu,
                    quantized_matmul_activation,
                    residual,
                    eps_const,
                    rms2_w,
                    rms2_lut,
                    HID,
                    gate_proj_qs,
                    gate_proj_qz,
                    gate_up_proj_w,
                    mlp_dq_params,
                    silu_lut,
                    down_proj_w,
                    down_dq_params,
                )
                # debug: emit dataflow intermediates for device-vs-fp32 bisection
                # (mqkv=in_proj, cs=conv+silu, o_all=recurrence readout, core=gated-norm*z, out=out_proj).
                if debug:
                    A.output(_tail, mqkv, cs, o_all, core, out)
                else:
                    A.output(_tail)

        return DeltaNetDecodeLayer

    if phase == "decode":
        return _build_decode_kernel()

    # Chunked prefill: process chunk_len tokens per call, carrying recurrent_state + conv_state
    # across chunks (state persists, zeroed at instance load). Bounds the compile size to
    # chunk_len regardless of total prompt length.
    seq_len = chunk_len if chunk_len else ctx_len
    if seq_len % C != 0:
        raise ValueError(f"prefill seq_len ({seq_len}) must be a multiple of chunk len C={C}.")
    NC = seq_len // C  # chunks; the recurrent state is carried across them inside chunk_imc
    BHS = BH * seq_len

    def to_heads(x):  # (1, seq, heads*nblk) -> (BH, NC, C, NB) head-major
        r = axol.atomic.reshape(x, shape=(1, NC, C, BH, NB))
        return axol.atomic.reshape(
            axol.core.transpose(r, dims=(0, 3, 1, 2, 4)), shape=(BH, NC, C, NB)
        )

    def from_heads(core):  # (BH, NC, C, NB) -> (1, seq, heads*nblk)
        r = axol.atomic.reshape(core, shape=(1, BH, NC, C, NB))
        return axol.atomic.reshape(
            axol.core.transpose(r, dims=(0, 2, 3, 1, 4)), shape=(1, seq_len, BH * NB)
        )

    @A.ir_module
    class DeltaNetLayer:
        @A.prim_func
        def main(
            hidden_states: A.Buffer((1, seq_len, HB), "float32x64", scope=A.DDR),
            residual_in: A.Buffer((1, seq_len, HB), "float32x64", scope=A.DDR),
            rms1_w: A.Buffer((HID,), "float32x64", scope=A.DDR),
            rms1_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            qs_h: A.Buffer((1, 1, HB), "float32x64", scope=A.DDR),
            qz_h: A.Buffer((1, 1, HB), "float32x64", scope=A.DDR),
            w_qkv: A.Buffer((CB, HID), "int8x64", scope=A.DDR),
            dq_qkv: A.Buffer((1, 1, 2 * CB), "float32x64", scope=A.DDR),
            w_z: A.Buffer((VB, HID), "int8x64", scope=A.DDR),
            dq_z: A.Buffer((1, 1, 2 * VB), "float32x64", scope=A.DDR),
            w_b: A.Buffer((1, HID), "int8x64", scope=A.DDR),
            dq_b: A.Buffer((1, 1, 2), "float32x64", scope=A.DDR),
            w_a: A.Buffer((1, HID), "int8x64", scope=A.DDR),
            dq_a: A.Buffer((1, 1, 2), "float32x64", scope=A.DDR),
            conv_w: A.Buffer((Kc, CB), "float32x64", scope=A.DDR),
            conv_state: A.Buffer((1, Kc - 1, CB), "float32x64", scope=A.DDR),
            onehots: A.Buffer((BH, 1), "float32x64", scope=A.DDR),
            dtb_all: A.Buffer((BHS, 1), "float32x64", scope=A.DDR),
            posA_all: A.Buffer((BHS, 1), "float32x64", scope=A.DDR),
            l2_l: A.Buffer((NRMS,), "float32x64", scope=A.DDR),
            gn_l: A.Buffer((NRMS,), "float32x64", scope=A.DDR),
            sig_l: A.Buffer((NL,), "float32x64", scope=A.DDR),
            sp_l: A.Buffer((NL,), "float32x64", scope=A.DDR),
            e_l: A.Buffer((NL,), "float32x64", scope=A.DDR),
            silu_l: A.Buffer((NL,), "float32x64", scope=A.DDR),
            l2w: A.Buffer((D,), "float32x64", scope=A.DDR),
            gnw: A.Buffer((D,), "float32x64", scope=A.DDR),
            eps: A.Buffer((1,), "float32x64", scope=A.DDR),
            epsg: A.Buffer((1,), "float32x64", scope=A.DDR),
            recurrent_state: A.Buffer((BH, C, NB), "float32x64", scope=A.DDR),
            zero_conv: A.Buffer((seq_len, CB), "float32x64", scope=A.DDR),
            zerob: A.Buffer((BHS, 1), "float32x64", scope=A.DDR),
            zerog: A.Buffer((BHS, NB), "float32x64", scope=A.DDR),
            one: A.Buffer((1,), "float32x64", scope=A.DDR),
            qs_core: A.Buffer((1, 1, VB), "float32x64", scope=A.DDR),
            qz_core: A.Buffer((1, 1, VB), "float32x64", scope=A.DDR),
            w_oproj: A.Buffer((HB, VAL), "int8x64", scope=A.DDR),
            dq_oproj: A.Buffer((1, 1, 2 * HB), "float32x64", scope=A.DDR),
            rms2_w: A.Buffer((HID,), "float32x64", scope=A.DDR),
            rms2_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            gate_proj_qs: A.Buffer((1, 1, HB), "float32x64", scope=A.DDR),
            gate_proj_qz: A.Buffer((1, 1, HB), "float32x64", scope=A.DDR),
            silu_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            gate_up_proj_w: A.Buffer((2 * intermediate_dim_pw, HID), "int8x64", scope=A.DDR),
            mlp_dq_params: A.Buffer((1, 1, 6 * intermediate_dim_pw), "float32x64", scope=A.DDR),
            down_proj_w: A.Buffer((HB, intermediate_dim), "int8x64", scope=A.DDR),
            down_dq_params: A.Buffer((1, 1, 2 * HB), "float32x64", scope=A.DDR),
        ):  # noqa: F821
            eps_const = A.alloc_constant(
                [9.9999999747524271e-07] * 64, dtype="float32x64", scope=A.DDR
            )
            # input_layernorm -> the mixer consumes the normed hidden (in_proj's
            # activation quant was calibrated on the input_layernorm output).
            rms1 = axol.core.rms_norm_kahan(
                hidden_states,
                rms1_lut,
                eps_const[0:1],
                (HID,),
                weight_buffer=rms1_w,
                invsqrt_method="additive",
            )

            # in_proj_qkv (quantize once, shared) -> conv -> silu -> split q/k/v
            hq = quant(rms1, qs_h, qz_h)
            mqkv = quantized_matmul_activation(
                hq, w_qkv, dq_qkv, output=BufferInfo(dtype="float32x64")
            )
            # Chunked prefill carries the conv window across chunks (conv_state persists, zeroed at
            # load so chunk 0 is fresh); a single full-prefill call uses zero_initial_state=True.
            conv = causal_conv1d(mqkv, conv_w, conv_state, zero_initial_state=(chunk_len is None))
            cs = pointwise_lut(
                axol.atomic.reshape(conv, shape=(seq_len, CB)), zero_conv, silu_l, one
            )
            cs3 = axol.atomic.reshape(cs, shape=(1, seq_len, CB))
            # q|k|v are three contiguous BH*NB-wide slices of the conv output
            # (CB == 3*BH*NB) sharing the (BH, NB) head layout. Head-majorise all three
            # in ONE transpose (3x fewer ops / DMA descriptor setups, no per-slice copy).
            qkv_hm = axol.core.transpose(
                axol.atomic.reshape(cs3, shape=(1, NC, C, 3, BH, NB)), dims=(0, 4, 3, 1, 2, 5)
            )  # (1, BH, 3, NC, C, NB)
            q_km = axol.atomic.reshape(
                axol.atomic.copy(qkv_hm[:, :, 0:1, :, :, :]), shape=(BH, NC, C, NB)
            )
            k_km = axol.atomic.reshape(
                axol.atomic.copy(qkv_hm[:, :, 1:2, :, :, :]), shape=(BH, NC, C, NB)
            )
            v_km = axol.atomic.reshape(
                axol.atomic.copy(qkv_hm[:, :, 2:3, :, :, :]), shape=(BH, NC, C, NB)
            )

            # gates: z (head-major), b/a (heads-in-lanes -> lane_select per head)
            z5 = quantized_matmul_activation(hq, w_z, dq_z, output=BufferInfo(dtype="float32x64"))
            z_all = axol.atomic.reshape(to_heads(z5), shape=(BHS, NB))
            b_T1 = axol.atomic.reshape(
                quantized_matmul_activation(hq, w_b, dq_b, output=BufferInfo(dtype="float32x64")),
                shape=(seq_len, 1),
            )
            a_T1 = axol.atomic.reshape(
                quantized_matmul_activation(hq, w_a, dq_a, output=BufferInfo(dtype="float32x64")),
                shape=(seq_len, 1),
            )
            b_all = A.alloc_buffer((BHS, 1), "float32x64", scope=A.DDR)
            a_all = A.alloc_buffer((BHS, 1), "float32x64", scope=A.DDR)
            for h in range(BH):
                oh = axol.atomic.reshape(axol.atomic.copy(onehots[h : h + 1, :]), shape=(1,))
                axol.atomic.copy(
                    lane_select(b_T1, oh), output=b_all[h * seq_len : (h + 1) * seq_len, :]
                )
                axol.atomic.copy(
                    lane_select(a_T1, oh), output=a_all[h * seq_len : (h + 1) * seq_len, :]
                )

            # mixer: batched front-end (rms_norm MUST be batched, not per-head)
            ql_all = axol.atomic.reshape(
                axol.core.rms_norm_kahan(
                    axol.atomic.reshape(q_km, shape=(1, BHS, NB)),
                    l2_l,
                    eps[0:1],
                    (D,),
                    weight_buffer=l2w,
                    invsqrt_method="additive",
                ),
                shape=(BHS, NB),
            )
            kl_all = axol.atomic.reshape(
                axol.core.rms_norm_kahan(
                    axol.atomic.reshape(k_km, shape=(1, BHS, NB)),
                    l2_l,
                    eps[0:1],
                    (D,),
                    weight_buffer=l2w,
                    invsqrt_method="additive",
                ),
                shape=(BHS, NB),
            )
            # Per-token gates: beta = sigmoid(b); g = -exp(A_log)·softplus(a+dt_bias).
            # neg_g = posA·softplus >= 0, so alpha = exp(-neg_g) reuses the exp(-x) LUT (e_l).
            beta_all = pointwise_lut(b_all, zerob, sig_l, one)  # (BHS, 1)
            neg_g_all = elementwise_mul(pointwise_lut(a_all, dtb_all, sp_l, one), posA_all)
            alpha_all = pointwise_lut(neg_g_all, zerob, e_l, one)  # (BHS, 1)

            # Sequential fp18 gated-delta recurrence: q/k/v lane-packed per (head, token)
            # (BH, seq, 1, NB), gates packed as ab = [alpha | beta] (BH, seq, 1, 2).
            # The op carries the recurrent state in L1 across the seq HW-loop and writes
            # it back into recurrent_state in place (state_inplace=True; zero-initialised
            # on a fresh instance).
            v_all = axol.atomic.reshape(v_km, shape=(BHS, NB))
            ln = (BH, seq_len, 1, NB)  # (BHS, NB) -> lane-packed per (head, token)
            ab_all = A.alloc_buffer((BHS, 2), "float32x64", scope=A.DDR)
            axol.atomic.copy(alpha_all, output=ab_all[:, 0:1])
            axol.atomic.copy(beta_all, output=ab_all[:, 1:2])
            o, _carry = gated_deltanet_recurrent_fp(
                axol.atomic.reshape(ql_all, shape=ln),
                axol.atomic.reshape(kl_all, shape=ln),
                axol.atomic.reshape(v_all, shape=ln),
                axol.atomic.reshape(ab_all, shape=(BH, seq_len, 1, 2)),
                recurrent_state,
                state_inplace=True,
            )

            # gated RMSNorm tail (eps = D * rms_eps) and out_proj
            rr = axol.atomic.reshape(
                axol.core.rms_norm_kahan(
                    axol.atomic.reshape(o, shape=(1, BHS, NB)),
                    gn_l,
                    epsg[0:1],
                    (D,),
                    weight_buffer=gnw,
                    invsqrt_method="additive",
                ),
                shape=(BHS, NB),
            )
            core = elementwise_mul(rr, pointwise_lut(z_all, zerog, silu_l, one))
            core_3 = from_heads(axol.atomic.reshape(core, shape=(BH, NC, C, NB)))
            out = quantized_matmul_activation(
                quant(core_3, qs_core, qz_core),
                w_oproj,
                dq_oproj,
                output=BufferInfo(dtype="float32x64"),
            )

            # recurrent_state was updated in place by gated_deltanet_recurrent_fp;
            # conv_state likewise by causal_conv1d -- both persistent cache IO.

            # residual add + the shared SwiGLU MLP tail.
            residual = axol.atomic.add(residual_in, out)
            _tail = _swiglu_mlp_tail(
                axol,
                BufferInfo,
                fused_gate_up_swiglu,
                quantized_matmul_activation,
                residual,
                eps_const,
                rms2_w,
                rms2_lut,
                HID,
                gate_proj_qs,
                gate_proj_qz,
                gate_up_proj_w,
                mlp_dq_params,
                silu_lut,
                down_proj_w,
                down_dq_params,
            )
            A.output(_tail)

    return DeltaNetLayer


# Fixed LUT domains for the elementwise nonlinearities (the gate activations are
# bounded; calibrating them buys nothing -- only the two inv_sqrt norms need ranges).
_SIG_RANGE = (-12.0, 12.0)  # sigmoid is bounded; this span covers it to <1e-7
# Real Qwen3.5-2B activations have wide ranges (conv-silu input reaches ~34,
# gate-silu ~16, softplus ~17). Curvature-equidistributed knots (_chained_lut) cover
# these spans at <0.016 max error without coarsening the bulk -- a uniform grid over
# the same range would be catastrophic (coarse softplus -> alpha>1 -> recurrence diverges).
# The silu LUT is shared by both silu sites, so it covers their union.
_SOFTPLUS_RANGE = (-18.0, 22.0)
_SILU_RANGE = (-25.0, 40.0)


def _sigmoid(x):
    import numpy as np

    return 1.0 / (1.0 + np.exp(-x))


def _softplus(x):
    import numpy as np

    return np.logaddexp(0.0, x)


def _silu(x):
    import numpy as np

    return x / (1.0 + np.exp(-x))


def _equidist_pwl(func, lo: float, hi: float, n_bins: int, *, power: float = 0.5):
    """Piecewise-linear fit of func over [lo, hi] with non-uniform knots.

    Places the n_bins+1 breakpoints by the equidistribution principle (de
    Boor): the chord error of a segment of width h is ~(1/8)|f''|h^2, so making
    every segment carry equal worst-case error means the knot density should be
    proportional to |f''(x)|^power -- power=1/2 minimises the L-infinity (max)
    error, power=2/5 the L2 error. Integrates that density and inverts it to get
    the knots, then least-squares-fits a line per bin. The hardware LUT stores
    explicit per-bin delimiters, so non-uniform spacing is free at inference.
    Returns an LUTParams.
    """
    from axelera.compiler.alto.axol.utils import lut_chain as LC
    import numpy as np

    xs = np.linspace(lo, hi, 8192, dtype=np.float64)
    ys = np.asarray(func(xs), dtype=np.float64)
    d2 = np.abs(np.gradient(np.gradient(ys, xs), xs))
    rho = (d2 + 1e-9) ** power
    # cumulative density Phi(x); knots evenly spaced in Phi -> equal error/segment
    phi = np.concatenate([[0.0], np.cumsum(0.5 * (rho[1:] + rho[:-1]) * np.diff(xs))])
    phi /= phi[-1]
    edges = np.interp(np.linspace(0.0, 1.0, n_bins + 1), phi, xs)
    edges = np.maximum.accumulate(edges)  # guard against ties from flat regions

    slopes = np.zeros(n_bins, dtype=np.float32)
    offsets = np.zeros(n_bins, dtype=np.float32)
    idx = np.clip(np.digitize(xs, edges) - 1, 0, n_bins - 1)
    for i in range(n_bins):
        m = idx == i
        if m.sum() >= 2:
            a, b = np.polyfit(xs[m], ys[m], 1)
        elif m.sum() == 1:
            a, b = 0.0, float(ys[m][0])
        else:  # empty bin (shouldn't happen with monotone edges) -> hold value
            a, b = 0.0, float(func(edges[i]))
        slopes[i], offsets[i] = a, b

    bin_delims = edges[1:].astype(np.float32)  # per-bin upper delimiter
    # Extend the final delimiter to +inf so inputs at/above the last knot use the
    # top bin's line (extrapolated) rather than falling into the zero-padding bin
    # and reading 0 -- otherwise x == hi exactly reads 0 (sigmoid(12) -> 0 not 1).
    bin_delims[-1] = LC.Float18Params.largest_value
    return LC.LUTParams(
        LC.np_round_float32_to_float18(bin_delims),
        LC.np_round_float32_to_float18(slopes),
        LC.np_round_float32_to_float18(offsets),
    )


def _chained_lut(func, lo: float, hi: float, n_luts: int):
    """Build the (n_luts, 64) additive chained-LUT table fitting func over [lo, hi].

    Uses curvature-equidistributed knots (see _equidist_pwl) rather than uniform
    spacing: at the effective bin budget of an n_luts chain this fits the real (wide)
    activation range ~50x more accurately than uniform bins.
    """
    import functools

    from axelera.compiler.alto.axol.utils import lut_chain as LC
    import numpy as np

    params = LC.compute_chained_lut_params(
        lo,
        hi,
        n_luts,
        method=LC.ChainedLUTMethod.ADDITIVE,
        fit_fn=functools.partial(_equidist_pwl, func),
    )
    return np.stack([LC.pack_lut_params_to_pword(p) for p in params]).astype(np.float32)


def _exp_lut(n_luts: int):
    """The (n_luts, 64) chained exp(-x) table over [0, 24] (the decay LUT)."""
    from axelera.compiler.alto.axol.utils import lut_chain as LC
    import numpy as np

    params = LC.compute_chained_lut_params(
        0.0, 24.0, n_luts, fit_fn=LC.fit_exp_lut_params_sdpa_softmax
    )
    return np.stack([LC.pack_lut_params_to_pword(p) for p in params]).astype(np.float32)


def deltanet_layer_buffers(bundle, spec, layer):
    """Bind one gated-DeltaNet layer's constants from the quantized model.

    The five W8A8 GEMMs (in_proj_qkv + z/b/a gates + out_proj) become NK int8
    weights plus interleaved [dz, ds] dequant tables (the qwen3-opt scheme); the
    depthwise conv is repacked channel-major; dt_bias / exp(A_log) are expanded
    per (head, timestep); the gated-norm scale, the ones/sqrt(D) L2 weight, and the
    sigmoid / softplus / exp / silu / inv_sqrt LUTs round it out.

    Only emits buffers the spec asks for, so it tolerates the kernel growing extra IO
    (e.g. the wrapping RMSNorm + SwiGLU MLP, added by _add_decoder_wrapper_buffers).
    """
    import numpy as np
    import torch

    from ...layout import (
        broadcast_to_pword,
        layer_conv,
        layer_linears,
        layer_subset,
        pack,
        pack_channel_major_conv,
    )
    from ...luts import generate_lut_table
    from ..architecture import get_architecture

    config = get_architecture(bundle.architecture).config_from_dict(bundle.config)
    PWORD = 64
    BH = config.linear_num_value_heads
    D = config.linear_key_head_dim
    eps = float(config.rms_norm_eps)
    NL = spec["sig_l"][0][0] if "sig_l" in spec else 4
    NRMS = spec["l2_l"][0][0] if "l2_l" in spec else 2

    lin = layer_linears(bundle.linear, layer)
    conv = layer_conv(bundle.conv, layer)
    dn = next(iter(layer_subset(bundle.deltanet, layer).values()))  # the GatedDeltaNet params
    rng = layer_subset(bundle.lut_ranges, layer)
    # Prefill processes ctx_len positions; decode is a single token (no zero_conv,
    # uses the (1, CB) zerocv instead).
    is_decode = "zero_conv" not in spec
    seq_len = 1 if is_decode else spec["zero_conv"][0][0]

    buffers: dict = {}

    def put(name, value):
        if name in spec:
            buffers[name] = pack(value, spec[name])

    def nk(w):  # [out, in] int8 -> NK (out//64, in, 64)
        out, inn = w.shape
        return w.reshape(out // PWORD, PWORD, inn).permute(0, 2, 1).contiguous().to(torch.int8)

    def blocks(v):  # (C,) -> (C//64, 64)
        return torch.as_tensor(v).to(torch.float32).reshape(-1, PWORD)

    def dq_table(p):  # interleaved [dz, ds] per output channel -> (1, 1, 2*opw, 64)
        opw = p["w"].shape[0] // PWORD
        t = torch.empty((2 * opw, PWORD), dtype=torch.float32)
        t[0::2] = blocks(p["dz"])
        t[1::2] = blocks(p["ds"])
        return t.reshape(1, 1, 2 * opw, PWORD)

    def pad_out(w, dz, ds):  # pad a <64-output projection up to one block (zeros)
        out, inn = w.shape
        wp = torch.zeros((PWORD, inn), dtype=w.dtype)
        wp[:out] = w
        dzp = torch.zeros(PWORD, dtype=torch.float32)
        dzp[:out] = torch.as_tensor(dz)
        dsp = torch.zeros(PWORD, dtype=torch.float32)
        dsp[:out] = torch.as_tensor(ds)
        return wp, {"w": wp, "dz": dzp, "ds": dsp}

    # in_proj_qkv: the shared activation quant (qs_h/qz_h) is reused for z/b/a too
    # (one SmoothQuant group -> identical qs/qz).
    qkv = lin["in_proj_qkv"]
    put("qs_h", qkv["qs"])
    put("qz_h", qkv["qz"])
    if "w_qkv" in spec:
        buffers["w_qkv"] = pack(nk(qkv["w"]), spec["w_qkv"])
    if "dq_qkv" in spec:
        buffers["dq_qkv"] = pack(dq_table(qkv), spec["dq_qkv"])

    z = lin["in_proj_z"]
    if "w_z" in spec:
        buffers["w_z"] = pack(nk(z["w"]), spec["w_z"])
    if "dq_z" in spec:
        buffers["dq_z"] = pack(dq_table(z), spec["dq_z"])

    # b/a emit num_v_heads (<64) channels -> pad to one block (heads in low lanes).
    for proj, wname, dname in (("in_proj_b", "w_b", "dq_b"), ("in_proj_a", "w_a", "dq_a")):
        p = lin[proj]
        wp, pp = pad_out(p["w"], p["dz"], p["ds"])
        if wname in spec:
            buffers[wname] = pack(nk(wp), spec[wname])
        if dname in spec:
            buffers[dname] = pack(dq_table(pp), spec[dname])

    # out_proj (its own activation quant: input is the gated-norm output).
    o = lin["out_proj"]
    put("qs_core", o["qs"])
    put("qz_core", o["qz"])
    if "w_oproj" in spec:
        buffers["w_oproj"] = pack(nk(o["w"]), spec["w_oproj"])
    if "dq_oproj" in spec:
        buffers["dq_oproj"] = pack(dq_table(o), spec["dq_oproj"])

    # Depthwise causal conv (channel-major), zero conv state (prefill).
    if "conv_w" in spec:
        buffers["conv_w"] = pack_channel_major_conv(conv["conv1d"]["weight"], spec["conv_w"])

    # Per-head gates expanded over the sequence (one value per (head, timestep),
    # broadcast across the 64 lanes). posA = exp(A_log) per head: both phases use
    # neg_g = posA·softplus >= 0 and alpha = exp(-neg_g).
    dt_bias = dn["dt_bias"].detach().to(torch.float32).cpu().numpy()  # [BH]
    posA = dn["A_log"].detach().float().exp().cpu().numpy()  # [BH]
    put("dtb_all", broadcast_to_pword(torch.from_numpy(np.repeat(dt_bias, seq_len))))
    put("posA_all", broadcast_to_pword(torch.from_numpy(np.repeat(posA, seq_len))))

    # Per-head lane one-hots (identity in the low BH lanes).
    put("onehots", torch.from_numpy(np.eye(BH, PWORD, dtype=np.float32)))

    # Norm weights + the inv_sqrt LUTs (calibrated ranges already include eps).
    put("l2w", broadcast_to_pword(torch.full((D,), 1.0 / (D**0.5))))
    put("gnw", broadcast_to_pword(dn["norm_weight"].detach().to(torch.float32).cpu()))
    put("eps", torch.tensor(eps))
    put("epsg", torch.tensor(D * eps))
    if "l2_l" in spec:
        r = rng["linear_attn.qk_l2_inv_sqrt_range"]
        buffers["l2_l"] = pack(
            torch.from_numpy(generate_lut_table("inv_sqrt", r["min"], r["max"], NRMS)),
            spec["l2_l"],
        )
    if "gn_l" in spec:
        r = rng["linear_attn.norm.inv_sqrt_range"]
        buffers["gn_l"] = pack(
            torch.from_numpy(generate_lut_table("inv_sqrt", r["min"], r["max"], NRMS)),
            spec["gn_l"],
        )

    # Elementwise-nonlinearity LUTs (fixed domains) + the decay exp LUT.
    if "sig_l" in spec:
        buffers["sig_l"] = pack(
            torch.from_numpy(_chained_lut(_sigmoid, *_SIG_RANGE, NL)), spec["sig_l"]
        )
    if "sp_l" in spec:
        buffers["sp_l"] = pack(
            torch.from_numpy(_chained_lut(_softplus, *_SOFTPLUS_RANGE, NL)), spec["sp_l"]
        )
    if "silu_l" in spec:
        buffers["silu_l"] = pack(
            torch.from_numpy(_chained_lut(_silu, *_SILU_RANGE, NL)), spec["silu_l"]
        )
    if "e_l" in spec:
        buffers["e_l"] = pack(torch.from_numpy(_exp_lut(NL)), spec["e_l"])

    # Zero pads (the pointwise_lut bias slots). recurrent_state + conv_state are
    # NOT here -- they are codegen-allocated persistent cache IO (zero-initialised on
    # a fresh instance, carried across calls), not per-layer constants.
    for name in ("zero_conv", "zerocv", "zerob", "zerog"):
        if name in spec:
            buffers[name] = torch.zeros(tuple(spec[name][0]), dtype=torch.float32)
    put("one", torch.tensor(1.0))

    # Decode-phase scalars: the fixed requant scales (1/s_q, 1/s_k, 1/s_S) and folded
    # decode constants are decode-only. s_q/s_k/s_S are calibrated decode scales on the
    # deltanet group (bounded: q/k L2-normed, S decayed).
    if is_decode:
        if "sqi" in spec:
            s_q, s_k, s_S = (float(dn[f"decode_s_{x}"]) for x in ("q", "k", "S"))
            put("sqi", torch.tensor(1.0 / s_q))
            put("ski", torch.tensor(1.0 / s_k))
            put("sSi", torch.tensor(1.0 / s_S))
            # fixed scale columns per head: [s_kS, s_qS, s_qk, 0(alpha), 0(beta), 0(s_S unused)].
            # The kernel overwrites cols 3/4 with the on-device alpha/beta.
            row = torch.tensor([s_k * s_S, s_q * s_S, s_q * s_k, 0.0, 0.0, 0.0])
            put("scfix_bh", row.reshape(1, 6, 1).expand(BH, 6, PWORD).contiguous())

    # The wrapping decoder layer (input/post norms + SwiGLU MLP) reuses the qwen3 binding.
    _add_decoder_wrapper_buffers(buffers, spec, bundle, layer)
    return buffers


# RMSNorm-weight / LUT sites (relative module path within the decoder layer):
# the input/post-attention norms + the SwiGLU silu, wrapping the deltanet mixer.
_WRAP_NORM_SITES = {"rms1_w": "input_layernorm", "rms2_w": "post_attention_layernorm"}
_WRAP_LUT_SITES = {
    "rms1_lut": "input_layernorm.inv_sqrt_range",
    "rms2_lut": "post_attention_layernorm.inv_sqrt_range",
    "silu_lut": "mlp.silu_range",
}


def _add_decoder_wrapper_buffers(buffers, spec, bundle, layer) -> None:
    """Bind the input/post RMSNorms + fused SwiGLU MLP that wrap the mixer.

    Reuses LFM2's fused-MLP packer + norm/LUT binding (covers gate_proj_qs/qz,
    the composed gate||up weight, the combined dequant tables, and down_proj_w).
    No-op when none of the wrapper buffers are in the spec."""
    from ...layout import layer_linears, layer_subset
    from ..lfm2.optimized_impl import _put_fused_mlp, _put_norms_and_luts

    lin = layer_linears(bundle.linear, layer)
    norm = layer_subset(bundle.norm, layer)
    rng = layer_subset(bundle.lut_ranges, layer)
    if "gate_proj" not in lin:
        return  # bare-mixer kernel (no wrapping decoder layer) -- nothing to bind

    _put_fused_mlp(buffers, spec, lin)
    _put_norms_and_luts(buffers, spec, norm, rng, _WRAP_NORM_SITES, _WRAP_LUT_SITES)


# full_attention layer kernel (the periodic dense-attention layer)
def _partial_rope_weights(head_dim_pw: int, rotary_pw: int):
    """Build the two swizzle gather matrices for partial RoPE on the first rotary_pw blocks of a head.

    half = rotary_pw*PWORD/2 freqs; the rotary slice is rearranged design-P
    (lane i<half -> block0, lane i>=half -> block1 low lanes) so the device rope op's
    whole-block pairing realises the (i, i+half) half-rotation, then gathered back.
    """
    import numpy as np

    PWORD = 64
    if rotary_pw != 1:
        raise NotImplementedError(
            f"partial-rope swizzle is wired for rotary_pw==1 (rotary_dim==64, Qwen3.5); "
            f"got rotary_pw={rotary_pw}."
        )
    rot = PWORD  # rotary channels (one block); half = the rotary_dim/2 freqs
    half = rot // 2
    # forward: rotary block -> 2 design-P blocks (block0 lane i<half <- q[i]; block1 <- q[i+half];
    # lanes >=half are don't-care -- discarded after rope, which is per-lane independent).
    w_fwd = np.zeros((2 * rot, rot), np.int8)
    for j in range(2 * rot):
        blk, lane = j // PWORD, j % PWORD
        w_fwd[j, (lane + (half if blk == 1 else 0)) if lane < half else 0] = 1
    # backward: roped 2 blocks -> 1 block (lane j<half <- block0[j]; else <- block1[j-half]).
    w_bwd = np.zeros((rot, 2 * rot), np.int8)
    for j in range(rot):
        w_bwd[j, j if j < half else (PWORD + (j - half))] = 1
    return w_fwd, w_bwd


def make_full_attention_layer_kernel(
    config: "Qwen3_5Config", ctx_len: int, phase: str, chunk_len: int | None = None
):
    """Build the gated full-attention decoder layer kernel.

    Three Qwen3.5-specific additions to the qwen3 optimized attention:
    (1) q_proj emits num_heads*head_dim*2 -> split query/gate;
    (2) partial RoPE on the first rotary_dim channels via the swizzle design-P trick;
    (3) sigmoid(gate) output gate before o_proj.
    KV cache, int8 SDPA, and the SwiGLU MLP tail are the standard qwen3/LFM2
    attention layer (head_dim=256 >= 128, no design-P padding needed).
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol.common import BufferInfo
    from axelera.compiler.alto.axol.utils.lut_chain import ChainedLUTMethod
    import numpy as np

    from ...ops.elementwise_mul import elementwise_mul
    from ...ops.fused_gate_up_swiglu import fused_gate_up_swiglu
    from ...ops.pointwise_lut import pointwise_lut
    from ...ops.quantized_matmul_activation import quantized_matmul_activation
    from ..lfm2.optimized_impl import _swiglu_mlp_tail

    PWORD = 64
    hidden_dim = config.hidden_size
    num_q = config.num_attention_heads
    num_kv = config.num_key_value_heads
    head_dim = config.head_dim
    intermediate_dim = config.intermediate_size
    rotary_dim = config.rotary_dim

    hidden_pw = hidden_dim // PWORD
    head_pw = head_dim // PWORD
    rotary_pw = rotary_dim // PWORD
    if rotary_dim % PWORD != 0:
        raise NotImplementedError(f"rotary_dim must be a multiple of {PWORD}; got {rotary_dim}")
    q_out = num_q * head_dim
    q_out_pw = q_out // PWORD
    kv_out = num_kv * head_dim
    kv_out_pw = kv_out // PWORD
    inter_pw = intermediate_dim // PWORD
    # ctx = KV-cache size (full context); seq = query positions processed per call.
    # Chunked prefill runs chunk_len queries against the full-ctx cache (seq_pos offsets
    # the KV write), attending to all earlier chunks via the (num_q, seq, ctx) mask.
    ctx = ctx_len
    seq = 1 if phase == "decode" else (chunk_len if chunk_len else ctx_len)
    w_fwd, w_bwd = _partial_rope_weights(head_pw, rotary_pw)
    sdpa = axol.core.scaled_dot_product_attention

    def partial_rope(x_heads, n_heads, sin, cos):
        # x_heads (1, n_heads, seq, head_pw): rope the first rotary_pw block(s), pass the rest.
        trig = axol.core.concat(sin, cos, dim=2)  # (1, seq, 2*rotary_pw)
        x_rot = axol.atomic.copy(x_heads[:, :, :, 0:rotary_pw])
        dp = axol.core.swizzle(x_rot, w_fwd)  # (1, n_heads, seq, 2*rotary_pw)
        roped = axol.core.rope(dp, trig)
        rout = axol.core.swizzle(roped, w_bwd)  # (1, n_heads, seq, rotary_pw)
        return axol.core.concat(rout, axol.atomic.copy(x_heads[:, :, :, rotary_pw:head_pw]), dim=3)

    @A.ir_module
    class Qwen35AttentionLayer:
        @A.prim_func
        def main(
            seq_pos: A.Buffer((1,), "uint64x64", scope=A.DDR),
            hidden_states: A.Buffer((1, seq, hidden_pw), "float32x64", scope=A.DDR),
            rms1_w: A.Buffer((hidden_dim,), "float32x64", scope=A.DDR),
            rms1_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            q_proj_qs: A.Buffer((1, 1, hidden_pw), "float32x64", scope=A.DDR),
            q_proj_qz: A.Buffer((1, 1, hidden_pw), "float32x64", scope=A.DDR),
            q_proj_w: A.Buffer((2 * q_out_pw, hidden_dim), "int8x64", scope=A.DDR),
            q_proj_dq: A.Buffer((1, 1, 2 * 2 * q_out_pw), "float32x64", scope=A.DDR),
            k_proj_w: A.Buffer((kv_out_pw, hidden_dim), "int8x64", scope=A.DDR),
            k_proj_dq: A.Buffer((1, 1, 2 * kv_out_pw), "float32x64", scope=A.DDR),
            v_proj_w: A.Buffer((kv_out_pw, hidden_dim), "int8x64", scope=A.DDR),
            v_proj_dq: A.Buffer((1, 1, 2 * kv_out_pw), "float32x64", scope=A.DDR),
            q_norm_w: A.Buffer((head_dim,), "float32x64", scope=A.DDR),
            q_norm_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            k_norm_w: A.Buffer((head_dim,), "float32x64", scope=A.DDR),
            k_norm_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            k_rope_sin: A.Buffer((1, seq, rotary_pw), "float32x64", scope=A.DDR),
            k_rope_cos: A.Buffer((1, seq, rotary_pw), "float32x64", scope=A.DDR),
            q_rope_sin: A.Buffer((1, seq, rotary_pw), "float32x64", scope=A.DDR),
            q_rope_cos: A.Buffer((1, seq, rotary_pw), "float32x64", scope=A.DDR),
            k_cache_qs: A.Buffer((1, 1, 1, head_pw), "float32x64", scope=A.DDR),
            k_cache_qz: A.Buffer((1, 1, 1, head_pw), "float32x64", scope=A.DDR),
            v_cache_qs: A.Buffer((1, 1, 1, head_pw), "float16x64", scope=A.DDR),
            v_cache_qz: A.Buffer((1, 1, 1, head_pw), "float16x64", scope=A.DDR),
            k_cache: A.Buffer((1, num_kv, ctx, head_pw), "int8x64", scope=A.DDR),
            v_cache: A.Buffer((1, num_kv, ctx, head_pw), "int8x64", scope=A.DDR),
            q_attn_qs: A.Buffer((1, 1, 1, head_pw), "float32x64", scope=A.DDR),
            q_attn_qz: A.Buffer((1, 1, 1, head_pw), "float32x64", scope=A.DDR),
            sdpa_exp_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            sdpa_inv_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            sdpa_scores_dq_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            sdpa_weights_q_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            sdpa_output_dq_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            attn_mask: A.Buffer((1, num_q, seq, ctx // 64), "float32x64", scope=A.DDR),
            gate_sig_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            o_proj_qs: A.Buffer((1, 1, q_out_pw), "float16x64", scope=A.DDR),
            o_proj_qz: A.Buffer((1, 1, q_out_pw), "float16x64", scope=A.DDR),
            o_proj_w: A.Buffer((hidden_pw, q_out), "int8x64", scope=A.DDR),
            o_proj_dq: A.Buffer((1, 1, 2 * hidden_pw), "float32x64", scope=A.DDR),
            residual_in: A.Buffer((1, seq, hidden_pw), "float32x64", scope=A.DDR),
            rms2_w: A.Buffer((hidden_dim,), "float32x64", scope=A.DDR),
            rms2_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            gate_proj_qs: A.Buffer((1, 1, hidden_pw), "float32x64", scope=A.DDR),
            gate_proj_qz: A.Buffer((1, 1, hidden_pw), "float32x64", scope=A.DDR),
            silu_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            gate_up_proj_w: A.Buffer((2 * inter_pw, hidden_dim), "int8x64", scope=A.DDR),
            mlp_dq_params: A.Buffer((1, 1, 6 * inter_pw), "float32x64", scope=A.DDR),
            down_proj_w: A.Buffer((hidden_pw, intermediate_dim), "int8x64", scope=A.DDR),
            down_dq_params: A.Buffer((1, 1, 2 * hidden_pw), "float32x64", scope=A.DDR),
            # The output-gate LUT's post-scale (== 1.0). Must be a runtime INPUT buffer,
            # not an in-kernel alloc_constant: pointwise_lut stages its post into L1, and
            # the compiler folds a CONST source into a global l1_const_data which is not
            # namespaced per kernel -- duplicate l1_const_data across prefill + decode
            # kernels collides and axkernelcc fails. A bound input stays in DDR and is
            # copied to L1 at runtime (same pattern as the deltanet's one buffer).
            one: A.Buffer((1,), "float32x64", scope=A.DDR),
        ):  # noqa: F821
            eps_c = A.alloc_constant(
                [9.9999999747524271e-07] * 64, dtype="float32x64", scope=A.DDR
            )
            z_zp = A.alloc_constant(np.zeros((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR)
            qmin_c = A.alloc_constant(np.full((1, 1, 1, 1, 64), -128), "float32x64", scope=A.DDR)
            qmax_c = A.alloc_constant(np.full((1, 1, 1, 1, 64), 127), "float32x64", scope=A.DDR)
            gate_zero = A.alloc_constant(np.zeros((seq, q_out_pw, 64)), "float32x64", scope=A.DDR)

            rms1 = axol.core.rms_norm_kahan(
                hidden_states,
                rms1_lut,
                eps_c[0:1],
                (hidden_dim,),
                weight_buffer=rms1_w,
                invsqrt_method="additive",
            )
            qkv0 = axol.atomic.mul(rms1, q_proj_qs)
            qkv_q = axol.atomic.dpu_cast(
                axol.atomic.add(qkv0, q_proj_qz), output=BufferInfo(dtype="int8x64")
            )
            # q_proj emits query/gate per head: (1, seq, num_q, 2*head_pw) -> split.
            qg = axol.atomic.reshape(
                quantized_matmul_activation(
                    qkv_q, q_proj_w, q_proj_dq, output=BufferInfo(dtype="float16x64")
                ),
                shape=(1, seq, num_q, 2 * head_pw),
            )
            q5 = axol.atomic.copy(qg[:, :, :, 0:head_pw])
            gate = axol.atomic.dpu_cast(
                axol.atomic.reshape(
                    axol.atomic.copy(qg[:, :, :, head_pw : 2 * head_pw]),
                    shape=(1, seq, q_out_pw),
                ),
                output=BufferInfo(dtype="float32x64"),
            )
            q_heads = axol.core.transpose(q5, dims=(0, 2, 1, 3))

            k5 = axol.atomic.reshape(
                quantized_matmul_activation(
                    qkv_q, k_proj_w, k_proj_dq, output=BufferInfo(dtype="float16x64")
                ),
                shape=(1, seq, num_kv, head_pw),
            )
            k_heads = axol.core.transpose(k5, dims=(0, 2, 1, 3))
            v5 = axol.atomic.reshape(
                quantized_matmul_activation(
                    qkv_q, v_proj_w, v_proj_dq, output=BufferInfo(dtype="float16x64")
                ),
                shape=(1, seq, num_kv, head_pw),
            )
            v_heads = axol.core.transpose(v5, dims=(0, 2, 1, 3))

            q_norm = axol.atomic.reshape(
                axol.core.rms_norm_kahan(
                    axol.atomic.reshape(q_heads, shape=(1, num_q * seq, head_pw)),
                    q_norm_lut,
                    eps_c[0:1],
                    (head_dim,),
                    weight_buffer=q_norm_w,
                    invsqrt_method="additive",
                ),
                shape=(1, num_q, seq, head_pw),
            )
            k_norm = axol.atomic.reshape(
                axol.core.rms_norm_kahan(
                    axol.atomic.reshape(k_heads, shape=(1, num_kv * seq, head_pw)),
                    k_norm_lut,
                    eps_c[0:1],
                    (head_dim,),
                    weight_buffer=k_norm_w,
                    invsqrt_method="additive",
                ),
                shape=(1, num_kv, seq, head_pw),
            )
            q_rope = partial_rope(q_norm, num_q, q_rope_sin, q_rope_cos)
            k_rope = partial_rope(k_norm, num_kv, k_rope_sin, k_rope_cos)

            k_update = axol.atomic.dpu_cast(
                axol.atomic.add(axol.atomic.mul(k_rope, k_cache_qs), k_cache_qz),
                output=BufferInfo(dtype="int8x64"),
            )
            v_update = axol.atomic.dpu_cast(
                axol.atomic.add(axol.atomic.mul(v_heads, v_cache_qs), v_cache_qz),
                output=BufferInfo(dtype="int8x64"),
            )
            k_out, v_out = axol.core.update_kv_cache(k_update, v_update, k_cache, v_cache, seq_pos)
            q_attn = axol.atomic.dpu_cast(
                axol.atomic.add(axol.atomic.mul(q_rope, q_attn_qs), q_attn_qz),
                output=BufferInfo(dtype="int8x64"),
            )
            attn_out = sdpa(
                q_attn,
                k_out,
                v_out,
                sdpa_scores_dq_scale,
                sdpa_exp_lut,
                sdpa_inv_lut,
                sdpa_weights_q_scale,
                z_zp,
                qmin_c,
                qmax_c,
                sdpa_output_dq_scale,
                z_zp,
                z_zp,
                attn_mask,
                enable_gqa=True,
                use_newton_raphson=False,
                scale=1.0 / (head_dim**0.5),
                neg_exp_method=ChainedLUTMethod.ADDITIVE,
                inv_method=ChainedLUTMethod.ADDITIVE,
            )
            attn_t = axol.core.transpose(attn_out, dims=(0, 2, 1, 3))
            attn_flat = axol.atomic.reshape(
                axol.atomic.dpu_cast(attn_t, output=BufferInfo(dtype="float32x64")),
                shape=(1, seq, q_out_pw),
            )
            # sigmoid output gate (delta 3): attn_flat * sigmoid(gate), before o_proj.
            sig = pointwise_lut(
                axol.atomic.reshape(gate, shape=(seq, q_out_pw)), gate_zero, gate_sig_lut, one
            )
            gated = axol.atomic.reshape(
                elementwise_mul(axol.atomic.reshape(attn_flat, shape=(seq, q_out_pw)), sig),
                shape=(1, seq, q_out_pw),
            )
            o6 = quantized_matmul_activation(
                axol.atomic.dpu_cast(
                    axol.atomic.add(axol.atomic.mul(gated, o_proj_qs), o_proj_qz),
                    output=BufferInfo(dtype="int8x64"),
                ),
                o_proj_w,
                o_proj_dq,
                output=BufferInfo(dtype="float32x64"),
            )
            residual = axol.atomic.add(residual_in, o6)
            A.output(
                _swiglu_mlp_tail(
                    axol,
                    BufferInfo,
                    fused_gate_up_swiglu,
                    quantized_matmul_activation,
                    residual,
                    eps_c,
                    rms2_w,
                    rms2_lut,
                    hidden_dim,
                    gate_proj_qs,
                    gate_proj_qz,
                    gate_up_proj_w,
                    mlp_dq_params,
                    silu_lut,
                    down_proj_w,
                    down_dq_params,
                )
            )

    return Qwen35AttentionLayer


def full_attention_layer_buffers(bundle, spec, layer):
    """Bind one gated full-attention layer's constants (head_dim>=128, no design-P).

    Reuses the LFM2/qwen3 optimized helpers: q/k/v/o are NK quantized_matmul_activation
    projections (q_proj has 2x the output for query/gate), the SwiGLU MLP is the fused
    gate||up packer, the norms/SDPA mirror the qwen3 attention layer. Qwen3.5 extras:
    q_proj output is 2*q_out (transparent to the NK packer) and a sigmoid LUT (gate_sig_lut)
    for the output gate. The partial-RoPE swizzle weights are compile-time constants in the
    kernel; the rope sin/cos tables are host inputs.
    """
    import torch

    from ...layout import fit_lut, layer_linears, layer_sdpa, layer_subset, pack, zeros
    from ..lfm2.optimized_impl import _put_fused_mlp, _put_norms_and_luts, _put_qma_proj

    lin = layer_linears(bundle.linear, layer)
    sdpa = layer_sdpa(bundle.sdpa, layer)
    norm = layer_subset(bundle.norm, layer)
    rng = layer_subset(bundle.lut_ranges, layer)
    buffers: dict = {}

    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        _put_qma_proj(buffers, spec, lin[proj], proj)
    _put_fused_mlp(buffers, spec, lin)
    _put_norms_and_luts(
        buffers,
        spec,
        norm,
        rng,
        {
            "rms1_w": "input_layernorm",
            "rms2_w": "post_attention_layernorm",
            "q_norm_w": "self_attn.q_norm",
            "k_norm_w": "self_attn.k_norm",
        },
        {
            "rms1_lut": "input_layernorm.inv_sqrt_range",
            "rms2_lut": "post_attention_layernorm.inv_sqrt_range",
            "q_norm_lut": "self_attn.q_norm.inv_sqrt_range",
            "k_norm_lut": "self_attn.k_norm.inv_sqrt_range",
            "silu_lut": "mlp.silu_range",
        },
    )
    # SDPA quant (per-head-dim qs for q/k/v, zero-pinned zp; per-tensor scalars).
    for prefix, fld in (("q_attn", "qs_q"), ("k_cache", "qs_k"), ("v_cache", "qs_v")):
        if f"{prefix}_qs" in spec:
            buffers[f"{prefix}_qs"] = pack(sdpa[fld], spec[f"{prefix}_qs"])
        if f"{prefix}_qz" in spec:
            buffers[f"{prefix}_qz"] = zeros(spec[f"{prefix}_qz"])
    for buf, fld in (
        ("sdpa_scores_dq_scale", "scores_dq_scale"),
        ("sdpa_weights_q_scale", "qs_w"),
        ("sdpa_output_dq_scale", "output_dq_scale"),
    ):
        if buf in spec:
            buffers[buf] = pack(sdpa[fld], spec[buf])
    for buf, site in (
        ("sdpa_exp_lut", "self_attn.sdpa.exp_range"),
        ("sdpa_inv_lut", "self_attn.sdpa.reciprocal_range"),
    ):
        if buf in spec and site in rng:
            buffers[buf] = fit_lut(rng[site], spec[buf])
    # The output-gate sigmoid LUT (fixed domain; the gate is bounded).
    if "gate_sig_lut" in spec:
        buffers["gate_sig_lut"] = pack(
            torch.from_numpy(_chained_lut(_sigmoid, *_SIG_RANGE, spec["gate_sig_lut"][0][0])),
            spec["gate_sig_lut"],
        )
    # The output-gate LUT post-scale (1.0), bound as a runtime input (see one param comment
    # in make_full_attention_layer_kernel for why it cannot be an in-kernel const).
    if "one" in spec:
        buffers["one"] = pack(torch.tensor(1.0), spec["one"])
    return buffers


# Implementation: hybrid two-kind stack
class Qwen3_5Implementation(DecoderLayerImplementation):
    """Qwen3.5 Metis impl: host embed -> hybrid device layers -> host lm_head.

    Two layer kinds (per config.layer_type): linear_attention (gated DeltaNet)
    and full_attention (gated dense attention).
    """

    name = "qwen3.5-metis"
    architecture = "qwen3.5"

    def host_kernels(self) -> list[HostKernel]:
        return [
            HostKernel("embedding", params=("embed_weight",)),
            HostKernel("lm_head", params=("final_norm_weight", "embed_weight", "rms_eps")),
        ]

    def layer_kernels(self, model_config) -> dict[str, LayerKernel]:
        return {
            "linear_attention": LayerKernel(
                "linear_attention",
                build=make_deltanet_layer_kernel,
                layer_buffers=deltanet_layer_buffers,
                phases=("prefill", "decode"),
                shared_inputs=frozenset(),  # no rope / mask / seq_pos for linear attention
                # Persistent per-layer cache (zero-initialised, carried across calls):
                # the gated-delta recurrent state and the depthwise-conv window.
                state=frozenset({"recurrent_state", "conv_state"}),
                enable_buffer_promotion=False,  # PromoteBuffers asserts on this op chain
                # Also the LayerKernel default; stated explicitly because it is a
                # correctness constraint: fusion reorders the hand-built IMC/IFD datapaths.
                enable_datapath_fusion=False,
            ),
            "full_attention": LayerKernel(
                "full_attention",
                build=make_full_attention_layer_kernel,
                layer_buffers=full_attention_layer_buffers,
                phases=("prefill", "decode"),
                shared_inputs=self.shared_inputs,
                state=frozenset({"k_out", "v_out"}),
                # PromoteBuffers asserts on the query/gate / partial-rope / output-gate
                # datapaths (annotate_promotion dma_group assertion). Promotion is a
                # perf-only L2 staging; disable for correctness.
                enable_buffer_promotion=False,
            ),
        }

    def layer_kinds(self, model_config) -> list[str]:
        return [model_config.layer_type(i) for i in range(model_config.num_hidden_layers)]

    def _rope_tables(self, position, head_dim, rope_theta):
        """Partial-RoPE cos/sin for the full_attention layers, packed for the design-P swizzle.

        Only rotary_dim channels of each head are rotated, pairing channel i with
        i + rotary_dim/2 (HF apply_partial_rotary_pos_emb with rotate_half).
        make_full_attention_layer_kernel realises that with a single-block (rotary_pw == 1)
        swizzle, so it wants the rotary_dim/2 angles in the LOW lanes of one pword:
        build (1, n_positions, 1, 64) cos/sin with lanes [0:half) and the [half:rotary_dim)
        duplicate (as in the reference's cat(freqs, freqs) -- the swizzle discards lanes >= half).
        inv_freq uses the rotary_dim base, NOT head_dim. rotary_dim == 64 for Qwen3.5
        (partial_rotary_factor 0.25 * head_dim 256), which is the kernel's rotary_pw == 1
        invariant (_partial_rope_weights)."""
        import numpy as np

        PWORD = 64
        rotary_dim = PWORD  # == 64; the kernel's swizzle requires rotary_pw == 1.
        half = rotary_dim // 2
        pos = np.atleast_1d(np.asarray(position, dtype=np.float64))
        inv_freq = 1.0 / (
            rope_theta ** (np.arange(0, rotary_dim, 2, dtype=np.float64) / rotary_dim)
        )
        angles = pos[:, None] * inv_freq[None, :]  # (n, half)
        cos = np.zeros((1, pos.size, 1, PWORD), np.float32)
        sin = np.zeros((1, pos.size, 1, PWORD), np.float32)
        cos[0, :, 0, :half] = cos[0, :, 0, half : 2 * half] = np.cos(angles)
        sin[0, :, 0, :half] = sin[0, :, 0, half : 2 * half] = np.sin(angles)
        return cos, sin


register_implementation(Qwen3_5Implementation())
__all__ = ["Qwen3_5Implementation", "make_deltanet_layer_kernel", "deltanet_layer_buffers"]

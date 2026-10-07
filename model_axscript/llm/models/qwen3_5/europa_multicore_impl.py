#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""EUROPA multicore implementation for the Qwen3.5 hybrid decoder.

Cloned from deltanet_impl.py (qwen3.5-metis), then adapted for EUROPA. The
full_attention kind uses llm.sdpa_flash_europa (multi-AICore flash attention,
sharding query heads across cores) and BOTH linear_attention phases run their
recurrence in fp32 on the AI core's RVV unit.

Qwen3.5 is a 3:1 hybrid stack (config.layer_type(i)): linear_attention
gated-DeltaNet layers and a periodic full_attention layer. This module wires
the gated-DeltaNet linear-attention layer and registers a two-kind
Implementation via the per-kind LayerKernel infra.

make_deltanet_layer_kernel assembles the token mixer: normed hidden -> quantized
in_proj_qkv IMC -> causal_conv1d -> silu -> split q/k/v -> head-major layout ->
{L2-norm q/k, quantized z/b/a gates, beta = sigmoid(b),
g = softplus(a + dt_bias) * -exp(A_log), the chunked gated-delta recurrence,
gated RMSNorm tail} -> out_proj IMC. All five GEMMs are W8A8 (gate quant adds
~1.3% to the output).

The four braced steps are fp32 RVV C kernels in both phases
(llm.deltanet_l2norm_rvv, llm.deltanet_gated_rmsnorm_rvv, and then
llm.deltanet_gates_rvv + llm.deltanet_chunk_rvv for prefill against
llm.deltanet_mixer_core_rvv, which folds all four in, for decode). That removes six
calibrated float18 LUT evaluations -- sigmoid, softplus, exp, two inverse square
roots and the gate SiLU -- along with the 32 lane_select ops that scattered b/a
per head, and the eps rescaling the LUT indirection needed. Error against the
torch reference is ~5e-7 and, unlike the fp18 scan, does not grow as the
recurrent state is carried across chunks or across decode steps.

Because both phases run the same arithmetic, the recurrent state one leaves is the
state the other picks up; the fp18 decode scan agreed with prefill only to the
accuracy of its fitted tables.

make_deltanet_layer_kernel is a full conforming decoder layer (threaded
hidden_states/residual_in -> output: input_layernorm -> deltanet mixer ->
+residual -> post_attn_norm -> SwiGLU MLP -> +residual) for BOTH phases: PREFILL
(chunked recurrence) and DECODE (single token).
recurrent_state + conv_state are in-place persistent cache IO (the KV-cache
pattern; prefill via state_inplace=, conv via causal_conv1d).
deltanet_layer_buffers binds every constant; both phases ask for the same set.

The depthwise short conv runs W8A8 on the DWPU rather than float18 on the DPU: measured at
seq=512 with C=6144, K=4, the DPU path costs 106.7M cycles against 2.3M, and the int8
result is the closer of the two to an exact conv (2.1e-04 vs 3.8e-04) because the DPU
carries only 10 mantissa bits. That needs the reference model to quantize the conv too, so
Qwen3_5 defaults quantize_conv=True and the scales come from QuantConv1d via
export_conv_quant_params -- the same path LFM2's conv block uses. A bundle without them is
rejected rather than papered over with a guessed scale.

The companion full_attention kind (gated dense attention layer + query/gate split +
partial RoPE + sigmoid output gate) is make_full_attention_layer_kernel. The impl
is registered as qwen3.5-europa-multicore.
"""
# flake8: noqa: F821  -- A.Buffer dtype-string annotations look like F821 names
from functools import partial
from typing import TYPE_CHECKING

from ...implementation import (
    DecoderLayerImplementation,
    HostKernel,
    LayerKernel,
    register_implementation,
)
from ..multicore_layer import ModelSpec, NormFold

if TYPE_CHECKING:
    from .qwen3_5 import Qwen3_5Config


def _nshard_qma(cores, a, w, dq, **kw):
    """A W8A8 projection N-sharded across cores: each core owns a band of the OUTPUT
    channels, so it streams only its 1/cores of the weight, and the bands concatenate
    into one buffer with no cross-core reduction. Downstream reads the whole buffer and
    does not know it was sharded.

    Decode is weight-bandwidth bound (58.7 MB per layer against ~1 MB of recurrent
    state), and each core has its own path to DDR. Measured on EUROPA with a
    decode-shaped 25 MB projection: 18.2 GB/s on one core, 29.1 on two, 36.0 on four,
    33.6 on eight -- DDR saturates at four, so more cores past that spend power for
    nothing. Falls back to one core when the output blocks do not divide evenly (the
    per-head b/a gates are a single block).

    The unified (banded) output layout is kept rather than split_output: its per-M-row
    DMA descriptors are what makes banding slow, and at M=1 there is only one row. The
    decode output is also ~1/500th of the weight traffic, and measured no slower.

    Shared by both layer kinds -- takes cores explicitly rather than closing over it,
    since each kind's kernel builder has its own.
    """
    from ...ops.quantized_matmul_activation import (
        quantized_matmul_activation,
        quantized_matmul_activation_multicore,
    )

    n_blocks = int(w.shape[0])
    if cores <= 1 or n_blocks % cores:
        return quantized_matmul_activation(a, w, dq, **kw)
    return quantized_matmul_activation_multicore(a, w, dq, cores=cores, **kw)


def _nshard_gate_up(cores, a, w, dq, **kw):
    """The fused gate||up projection N-sharded across cores.

    gate_up_proj is 25.2 MB, 43% of the layer's decode weight traffic, and the fused op
    already shards its own output N-tiles (each core owns whole tiles), so this only has
    to hand it the core count. The output stays in the split-N layout down_proj consumes
    via matrix_a_is_split, so nothing un-splits and the shard is invisible downstream.

    Shared by both layer kinds -- see _nshard_qma.
    """
    from ...ops.fused_gate_up_swiglu import fused_gate_up_swiglu

    return fused_gate_up_swiglu(a, w, dq, cores=(cores if cores > 1 else None), **kw)


# The deltanet layer's input RMSNorm, whose affine weight the packer folds into the
# quantize scale that follows it. qwen3's INPUT_NORM_FOLD targets "q_proj_qparams"; this
# layer's normed hidden feeds in_proj_qkv (and w_z/w_b/w_a, which share the same
# SmoothQuant group and therefore the same scale), so the fold targets that buffer.
DELTANET_INPUT_NORM_FOLD = NormFold("qkv_proj_qparams", "input_layernorm")


# linear_attention (gated DeltaNet) layer kernel
def make_deltanet_layer_kernel(
    config: "Qwen3_5Config",
    ctx_len: int,
    phase: str,
    chunk_len: int | None = None,
    cores: int = 1,
):
    """Build the gated-DeltaNet linear-attention token-mixer IRModule.

    Assembles the pipeline: normed hidden -> quantized in_proj_qkv IMC ->
    causal_conv1d -> silu -> split q/k/v -> head-major layout -> {L2-norm q/k,
    quantized z/b/a gates, beta/g, the gated-delta recurrence, gated RMSNorm} ->
    out_proj IMC. In PREFILL the braced steps are fp32 RVV kernels; DECODE still
    runs them on the DPU in fp18.

    Boundary is the torch GatedDeltaNet.forward (in: pre-mixer hidden, out:
    out_proj(core)); the input RMSNorm, residual add and SwiGLU MLP that wrap
    it follow the qwen3 pattern and are composed by the per-kind decoder infra.

    Built as a single un-wrapped prim_func (no A.block("root") wrapper) because
    rms_norm_kahan owns the root block. All five projections are W8A8
    (gate quant adds ~1.3% to the output).
    """
    from types import SimpleNamespace

    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol.common import BufferInfo
    from axelera.compiler.alto.compiler.config import HardwareGeneration

    from ...ops.causal_conv1d import causal_conv1d
    from ...ops.deltanet_chunk_rvv import deltanet_chunk_rvv
    from ...ops.deltanet_gated_rmsnorm_rvv import deltanet_gated_rmsnorm_rvv
    from ...ops.deltanet_gates_rvv import deltanet_gates_rvv
    from ...ops.deltanet_l2norm_rvv import deltanet_l2norm_rvv
    from ...ops.deltanet_mixer_core_rvv import deltanet_mixer_core_rvv
    from ...ops.fused_gate_up_swiglu import fused_gate_up_swiglu
    from ...ops.quantized_matmul_activation import quantized_matmul_activation
    from ...ops.imc_mm_int8 import imc_mm_int8
    from ...ops.silu_rvv import silu_rvv
    from ..multicore_layer import POST_ATTN_NORM_FOLD, hidden_norm_quant

    # Tokens per recurrence chunk, independent of C: the recurrence takes q/k/v flat
    # in seq and chunks internally. Must be a multiple of PWORD, as the IMC Gram
    # output is lane-packed.
    RVV_CHUNK = 64
    # _compile_phase traces this kernel inside alto_config_context(generation=...), so an
    # unset generation Arg resolves to the AMBIENT generation, not to the OMEGA default an
    # earlier revision of this file relied on. The shared DPU ops all register for
    # [OMEGA, EUROPA], so they resolve either way; the RVV ops are EUROPA-only and say so at
    # the call site regardless, which is what keeps this kernel independent of the ambient
    # value. NB the projections built through _nshard_qma do NOT pass generation and so now
    # bind EUROPA here where they once bound OMEGA -- see _compile_phase's own note.
    EUROPA = HardwareGeneration.EUROPA

    PWORD = 64
    # Lane-packed width of the per-token gate mask (see token_mask): one lane per token.
    TOKMASK = max(1, (1 if phase == "decode" else (chunk_len or ctx_len)) // PWORD)
    # BH (value heads) is the wide count the recurrence, gates and norm run at; BHK
    # (key heads) is the possibly narrower count in_proj_qkv emits for q/k. GQA
    # repeat_interleaves each key head across REP value heads; REP == 1 is identity.
    BH = config.linear_num_value_heads
    BHK = config.linear_num_key_heads
    if BH % BHK != 0:
        raise ValueError(
            f"make_deltanet_layer_kernel: linear_num_value_heads ({BH}) must be a "
            f"multiple of linear_num_key_heads ({BHK}) to repeat_interleave q/k over it."
        )
    REP = BH // BHK
    D = config.linear_key_head_dim  # = value_head_dim; also the chunk length C
    C = D
    if C % PWORD != 0:
        raise ValueError(f"linear_key_head_dim must be a multiple of {PWORD}; got {C}")
    NB = D // PWORD
    rms_eps = float(config.rms_norm_eps)
    HID = config.hidden_size
    KEY = BHK * D  # q/k width (narrow, pre-repeat)
    VAL = BH * D  # v width (wide)
    CONV = 2 * KEY + VAL
    Kc = config.linear_conv_kernel_dim
    HB, CB, VB, KB = HID // PWORD, CONV // PWORD, VAL // PWORD, KEY // PWORD
    intermediate_dim = config.intermediate_size
    intermediate_dim_pw = intermediate_dim // PWORD

    # Prefetch tile i+1's weights under tile i's compute, as qwen3's multicore_impl does.
    _pipeline_weights = phase == "decode"
    nshard_qma = partial(_nshard_qma, cores, pipeline_weights=_pipeline_weights)
    nshard_gate_up = partial(
        _nshard_gate_up, cores, pipeline_weights=_pipeline_weights, generation=EUROPA
    )
    _mlp_qma = partial(_nshard_qma, cores, pipeline_weights=_pipeline_weights)
    _mlp_gate_up = nshard_gate_up

    # qwen3's post-attention-norm dims for hidden_norm_quant's fused RMSNorm+quantize
    # (see the MLP tail below); HB <= DPU_REDUCE_MAX_D_BLOCKS (64) at this model's
    # hidden_size, so this always takes the fast rms_norm_dpu_reduce path.
    _mlp_dims = SimpleNamespace(
        hidden_dim=HID,
        hidden_dim_pw=HB,
        seq_len=(1 if phase == "decode" else (chunk_len if chunk_len else ctx_len)),
        use_dpu_hidden_norm=(HB <= 64),
    )

    # Half of the heaviest weights' K lives in DDR1 so both regions stream at once
    # (decode is weight-bandwidth bound; see _nshard_qma). w_z/w_b/w_a stay whole --
    # their traffic is a small fraction of the rest. Applies to BOTH phases: the
    # authoring pipeline packs one constant per (layer, buffer name) shared by every
    # phase declaring it, so a decode-only split would bind prefill's same-named w_qkv
    # to decode's half-width tensor. A.DDR1 directly rather than upper_shard_scope(): this
    # kernel is only ever built for EUROPA, so naming the region is exact and needs no
    # generation lookup at all. (upper_shard_scope() would now resolve correctly, since the
    # trace runs inside alto_config_context -- but it did not when this was written, and
    # returning plain DDR silently collapsed both shards into one region.)
    _ddr1 = A.DDR1
    qkv_k_lo = HID // 2
    o_k_lo = VAL // 2
    gu_k_lo = HID // 2
    down_k_lo = intermediate_dim // 2

    def quant(x, qs, qz):
        return axol.atomic.dpu_cast(
            axol.atomic.add(axol.atomic.mul(x, qs), qz), output=BufferInfo(dtype="int8x64")
        )

    def quant_sym(x, inv):  # symmetric (qz=0) int8 quant: round(x / s) via mul by 1/s
        return axol.atomic.dpu_cast(axol.atomic.mul(x, inv), output=BufferInfo(dtype="int8x64"))

    def _build_decode_kernel():
        # Single-token decode running the same fp32 pipeline as prefill, so the recurrent
        # state one phase leaves is the state the other picks up. One token needs no
        # head-major transpose: q|k|v are contiguous heads*NB-wide slices of the conv
        # output already in (heads, NB) layout, so to_heads is a reshape.
        def to_heads_dec(x, heads):  # (1, 1, heads*NB) -> (heads, NB), one token
            return axol.atomic.reshape(x, shape=(heads, NB))

        def _repeat_kv_heads(x):  # (BHK, NB) -> (BH, NB): repeat_interleave over heads
            if REP == 1:
                return x
            tiled = axol.atomic.repeat(
                axol.atomic.reshape(x, shape=(BHK, 1, NB)), nb_repetitions=(1, REP, 1)
            )
            return axol.atomic.reshape(tiled, shape=(BH, NB))

        def _mixer_step_and_norm(
            q_h, k_h, v_h, a_l, b_l, pos_a_w, dt_bias_w, z_all, gnorm_w, recurrent_state
        ):
            # L2-norm(q) + L2-norm(k) + the gated-delta step + gated RMSNorm, fused into one
            # RVV kernel. The fused op requires q/k already repeated to BH heads so they agree
            # with v; an L2-norm is row-wise, so repeating before or after it is equivalent.
            return deltanet_mixer_core_rvv(
                _repeat_kv_heads(q_h),
                _repeat_kv_heads(k_h),
                v_h,
                a_l,
                b_l,
                pos_a_w,
                dt_bias_w,
                z_all,
                gnorm_w,
                recurrent_state,
                q_eps=rms_eps,
                q_scale=float(D) ** -0.5,
                k_scale=1.0,
                gnorm_eps=rms_eps,
                num_heads=BH,
                generation=EUROPA,
                cores=(cores if cores > 1 else None),
                output=BufferInfo(dtype="float32x64"),
            )

        @A.ir_module
        class DeltaNetDecodeLayer:
            @A.prim_func
            def main(
                hidden_states: A.Buffer((1, 1, HB), "float32x64", scope=A.DDR),
                residual_in: A.Buffer((1, 1, HB), "float32x64", scope=A.DDR),
                # Input RMSNorm + activation quantize, fused. The affine weight is folded into the
                # quantize scale by the packer (DELTANET_INPUT_NORM_FOLD), hence no rms1_w.
                rms1_consts: A.Buffer((4,), "float32x64", scope=A.DDR),
                qkv_proj_qparams: A.Buffer((2 * HB,), "float32x64", scope=A.DDR),
                w_qkv: A.Buffer((CB, qkv_k_lo), "int8x64", scope=A.DDR),
                dq_qkv: A.Buffer((1, 1, 2 * CB), "float32x64", scope=A.DDR),
                w_z: A.Buffer((VB, HID), "int8x64", scope=A.DDR),
                dq_z: A.Buffer((1, 1, 2 * VB), "float32x64", scope=A.DDR),
                w_b: A.Buffer((1, HID), "int8x64", scope=A.DDR),
                dq_b: A.Buffer((1, 1, 2), "float32x64", scope=A.DDR),
                w_a: A.Buffer((1, HID), "int8x64", scope=A.DDR),
                dq_a: A.Buffer((1, 1, 2), "float32x64", scope=A.DDR),
                # Depthwise conv runs in float on the DPU, so the taps arrive already fake-quantized
                # (w_int8 * s_w): the weight matches the DWPU path bit for bit and only the
                # activation quantize is gone, which is why there is no conv_qs/conv_ds. conv_state
                # is phase-shared, so prefill uses the same dtype and layout.
                conv_w: A.Buffer((Kc, CB), "float32x64", scope=A.DDR),
                conv_state: A.Buffer((1, Kc - 1, CB), "float32x64", scope=A.DDR),
                # 1.0 per real token, 0.0 for padding on a short final prefill chunk. Zeroing
                # both gates makes a padded position an exact no-op on the recurrent state
                # (beta=0 -> vnew=0, g=0 -> alpha=1, so S_new = S), which is what lets the
                # remainder be one prefill call instead of a token-by-token decode replay.
                token_mask: A.Buffer((1, TOKMASK), "float32x64", scope=A.DDR),
                pos_a_w: A.Buffer((1, 1), "float32x64", scope=A.DDR),
                dt_bias_w: A.Buffer((1, 1), "float32x64", scope=A.DDR),
                gnorm_w: A.Buffer((1, NB), "float32x64", scope=A.DDR),
                recurrent_state: A.Buffer((BH, C, NB), "float32x64", scope=A.DDR),
                qs_core: A.Buffer((1, 1, VB), "float32x64", scope=A.DDR),
                qz_core: A.Buffer((1, 1, VB), "float32x64", scope=A.DDR),
                w_oproj: A.Buffer((HB, o_k_lo), "int8x64", scope=A.DDR),
                dq_oproj: A.Buffer((1, 1, 2 * HB), "float32x64", scope=A.DDR),
                rms2_consts: A.Buffer((4,), "float32x64", scope=A.DDR),
                gate_proj_qparams: A.Buffer((2 * HB,), "float32x64", scope=A.DDR),
                silu_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
                gate_up_proj_w: A.Buffer(
                    (2 * intermediate_dim_pw, gu_k_lo), "int8x64", scope=A.DDR
                ),
                mlp_dq_params: A.Buffer(
                    (1, 1, 6 * intermediate_dim_pw), "float32x64", scope=A.DDR
                ),
                down_proj_w: A.Buffer((HB, down_k_lo), "int8x64", scope=A.DDR),
                down_dq_params: A.Buffer((1, 1, 2 * HB), "float32x64", scope=A.DDR),
                # DDR1 upper shards of the four K-split weights above. Declared last:
                # the runtime binds inputs pool-grouped, so this keeps each weight's two
                # halves' pools contiguous with the rest of a phase's inputs.
                w_qkv1: A.Buffer((CB, HID - qkv_k_lo), "int8x64", scope=_ddr1),
                w_oproj1: A.Buffer((HB, VAL - o_k_lo), "int8x64", scope=_ddr1),
                gate_up_proj_w1: A.Buffer(
                    (2 * intermediate_dim_pw, HID - gu_k_lo), "int8x64", scope=_ddr1
                ),
                down_proj_w1: A.Buffer((HB, intermediate_dim - down_k_lo), "int8x64", scope=_ddr1),
            ):  # noqa: F821
                hq = hidden_norm_quant(
                    hidden_states,
                    rms1_consts,
                    qkv_proj_qparams,
                    _mlp_dims,
                    DELTANET_INPUT_NORM_FOLD,
                )
                mqkv = nshard_qma(
                    hq, w_qkv, dq_qkv, matrix_b1=w_qkv1, output=BufferInfo(dtype="float32x64")
                )
                conv = causal_conv1d(mqkv, conv_w, conv_state, zero_initial_state=False)
                cs = silu_rvv(axol.atomic.reshape(conv, shape=(1, CB)), generation=EUROPA)
                cs3 = axol.atomic.reshape(cs, shape=(1, 1, CB))
                # q|k are KB blocks wide (BHK heads); v is VB blocks wide (BH heads).
                q_h = to_heads_dec(axol.atomic.copy(cs3[:, :, 0:KB]), BHK)
                k_h = to_heads_dec(axol.atomic.copy(cs3[:, :, KB : 2 * KB]), BHK)
                v_h = to_heads_dec(axol.atomic.copy(cs3[:, :, 2 * KB : 2 * KB + VB]), BH)

                z_all = to_heads_dec(
                    nshard_qma(hq, w_z, dq_z, output=BufferInfo(dtype="float32x64")), BH
                )
                # b/a keep the heads in lanes; the step op reads the head lane directly,
                # so the 32 lane_select ops that scattered them per head are gone.
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

                core = _mixer_step_and_norm(
                    q_h,
                    k_h,
                    v_h,
                    a_l,
                    b_l,
                    pos_a_w,
                    dt_bias_w,
                    z_all,
                    gnorm_w,
                    recurrent_state,
                )
                out = nshard_qma(
                    quant(axol.atomic.reshape(core, shape=(1, 1, VB)), qs_core, qz_core),
                    w_oproj,
                    dq_oproj,
                    matrix_b1=w_oproj1,
                    output=BufferInfo(dtype="float32x64"),
                )
                residual = axol.atomic.add(residual_in, out)
                # SwiGLU MLP tail: qwen3's multicore_impl pattern (fused norm+quant,
                # weight-pipelined N-sharded matmuls), not LFM2's separate
                # norm/mul/add/cast chain -- see _mlp_dims and _nshard_qma/_nshard_gate_up.
                gu_q = hidden_norm_quant(
                    residual, rms2_consts, gate_proj_qparams, _mlp_dims, POST_ATTN_NORM_FOLD
                )
                mlp_split = _mlp_gate_up(
                    gu_q,
                    gate_up_proj_w,
                    mlp_dq_params,
                    matrix_b1=gate_up_proj_w1,
                    luts=silu_lut,
                )
                d6 = _mlp_qma(
                    mlp_split,
                    down_proj_w,
                    down_dq_params,
                    matrix_a_is_split=True,
                    matrix_b1=down_proj_w1,
                    output=BufferInfo(dtype="float32x64"),
                )
                _tail = axol.atomic.add(residual, d6)
                A.output(_tail)

        return DeltaNetDecodeLayer

    if phase == "decode":
        return _build_decode_kernel()

    # Chunked prefill: process chunk_len tokens per call, carrying recurrent_state + conv_state
    # across chunks (state persists, zeroed at instance load). Bounds the compile size to
    # chunk_len regardless of total prompt length.
    seq_len = chunk_len if chunk_len else ctx_len
    # Gram shapes, computed HERE rather than in the prim_func body: TVMScript turns a name
    # assigned inside the body into a TIR expression, so `BH * (seq_len // RVV_CHUNK)`
    # would come out a TIR Mul and reshape's int() on it fails.
    GRAM_A_SHAPE = (BH * (seq_len // RVV_CHUNK), RVV_CHUNK, NB)
    GRAM_OUT_SHAPE = (BH, seq_len, RVV_CHUNK // PWORD)
    if seq_len % C != 0:
        raise ValueError(f"prefill seq_len ({seq_len}) must be a multiple of chunk len C={C}.")
    BHS = BH * seq_len

    # The token axis is kept as ONE seq dimension through the transposes rather than
    # split into (NC, C). axol.core.transpose tiles along the first non-unit input dim,
    # so with (1, NC, C, ...) it can only tile NC -- and even at NC=1 the residency is
    # C*3*BH*NB (3.1 MB here) regardless of sequence length. Against a single seq axis
    # it tiles down to individual tokens instead.
    def to_heads(x, heads):  # (1, seq, heads*nblk) -> (heads, seq, NB) head-major
        r = axol.atomic.reshape(x, shape=(1, seq_len, heads, NB))
        return axol.atomic.reshape(
            axol.core.transpose(r, dims=(0, 2, 1, 3)), shape=(heads, seq_len, NB)
        )

    def from_heads(core):  # (BH, seq, NB) -> (1, seq, heads*nblk)
        r = axol.atomic.reshape(core, shape=(1, BH, seq_len, NB))
        return axol.atomic.reshape(
            axol.core.transpose(r, dims=(0, 2, 1, 3)), shape=(1, seq_len, BH * NB)
        )

    def _repeat_kv_heads_seq(x):  # (BHK, seq, NB) -> (BH, seq, NB): repeat_interleave heads
        if REP == 1:
            return x
        tiled = axol.atomic.repeat(
            axol.atomic.reshape(x, shape=(BHK, 1, seq_len, NB)), nb_repetitions=(1, REP, 1, 1)
        )
        return axol.atomic.reshape(tiled, shape=(BH, seq_len, NB))

    @A.ir_module
    class DeltaNetLayer:
        @A.prim_func
        def main(
            hidden_states: A.Buffer((1, seq_len, HB), "float32x64", scope=A.DDR),
            residual_in: A.Buffer((1, seq_len, HB), "float32x64", scope=A.DDR),
            # Fused input norm + quantize -- see the decode kernel's comment.
            rms1_consts: A.Buffer((4,), "float32x64", scope=A.DDR),
            qkv_proj_qparams: A.Buffer((2 * HB,), "float32x64", scope=A.DDR),
            w_qkv: A.Buffer((CB, qkv_k_lo), "int8x64", scope=A.DDR),
            dq_qkv: A.Buffer((1, 1, 2 * CB), "float32x64", scope=A.DDR),
            w_z: A.Buffer((VB, HID), "int8x64", scope=A.DDR),
            dq_z: A.Buffer((1, 1, 2 * VB), "float32x64", scope=A.DDR),
            w_b: A.Buffer((1, HID), "int8x64", scope=A.DDR),
            dq_b: A.Buffer((1, 1, 2), "float32x64", scope=A.DDR),
            w_a: A.Buffer((1, HID), "int8x64", scope=A.DDR),
            dq_a: A.Buffer((1, 1, 2), "float32x64", scope=A.DDR),
            # Float DPU conv -- see the decode kernel's comment. conv_state is shared
            # with decode (phase-independent persistent state), so both must agree.
            conv_w: A.Buffer((Kc, CB), "float32x64", scope=A.DDR),
            conv_state: A.Buffer((1, Kc - 1, CB), "float32x64", scope=A.DDR),
            token_mask: A.Buffer((1, TOKMASK), "float32x64", scope=A.DDR),
            pos_a_w: A.Buffer((1, 1), "float32x64", scope=A.DDR),
            dt_bias_w: A.Buffer((1, 1), "float32x64", scope=A.DDR),
            gnorm_w: A.Buffer((1, NB), "float32x64", scope=A.DDR),
            recurrent_state: A.Buffer((BH, C, NB), "float32x64", scope=A.DDR),
            qs_core: A.Buffer((1, 1, VB), "float32x64", scope=A.DDR),
            qz_core: A.Buffer((1, 1, VB), "float32x64", scope=A.DDR),
            w_oproj: A.Buffer((HB, o_k_lo), "int8x64", scope=A.DDR),
            dq_oproj: A.Buffer((1, 1, 2 * HB), "float32x64", scope=A.DDR),
            rms2_consts: A.Buffer((4,), "float32x64", scope=A.DDR),
            gate_proj_qparams: A.Buffer((2 * HB,), "float32x64", scope=A.DDR),
            silu_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            gate_up_proj_w: A.Buffer((2 * intermediate_dim_pw, gu_k_lo), "int8x64", scope=A.DDR),
            mlp_dq_params: A.Buffer((1, 1, 6 * intermediate_dim_pw), "float32x64", scope=A.DDR),
            down_proj_w: A.Buffer((HB, down_k_lo), "int8x64", scope=A.DDR),
            down_dq_params: A.Buffer((1, 1, 2 * HB), "float32x64", scope=A.DDR),
            # DDR1 upper shards, matching the decode kernel's split (see the comment
            # above qkv_k_lo): must be identical in both phases, not just declared here.
            w_qkv1: A.Buffer((CB, HID - qkv_k_lo), "int8x64", scope=_ddr1),
            w_oproj1: A.Buffer((HB, VAL - o_k_lo), "int8x64", scope=_ddr1),
            gate_up_proj_w1: A.Buffer(
                (2 * intermediate_dim_pw, HID - gu_k_lo), "int8x64", scope=_ddr1
            ),
            down_proj_w1: A.Buffer((HB, intermediate_dim - down_k_lo), "int8x64", scope=_ddr1),
        ):  # noqa: F821
            # input_layernorm -> the mixer consumes the normed hidden (in_proj's
            # activation quant was calibrated on the input_layernorm output).
            # in_proj_qkv (quantize once, shared) -> conv -> silu -> split q/k/v
            hq = hidden_norm_quant(
                hidden_states, rms1_consts, qkv_proj_qparams, _mlp_dims, DELTANET_INPUT_NORM_FOLD
            )
            mqkv = quantized_matmul_activation(
                hq, w_qkv, dq_qkv, matrix_b1=w_qkv1, output=BufferInfo(dtype="float32x64")
            )
            # conv_state persists across chunks (zeroed at load); a single full-prefill call
            # uses zero_initial_state=True. The float DPU path is faster than int8 on the DWPU
            # but less accurate (float18 accumulates in 10 mantissa bits); that is the trade.
            conv = causal_conv1d(mqkv, conv_w, conv_state, zero_initial_state=(chunk_len is None))
            # fp32 SiLU. A float18 LUT carries an absolute error budget over a span of roughly
            # [-25, 40], which is a large relative error where SiLU is near zero.
            cs = silu_rvv(axol.atomic.reshape(conv, shape=(seq_len, CB)), generation=EUROPA)
            cs3 = axol.atomic.reshape(cs, shape=(1, seq_len, CB))
            # q|k are KB*2 blocks wide (BHK heads, narrow under GQA); v is VB blocks
            # wide (BH heads). q/k share one transpose (their widths match); v needs
            # its own since its width can differ once BHK < BH.
            qk_hm = axol.core.transpose(
                axol.atomic.reshape(
                    axol.atomic.copy(cs3[:, :, 0 : 2 * KB]), shape=(1, seq_len, 2, BHK, NB)
                ),
                dims=(0, 3, 2, 1, 4),
            )  # (1, BHK, 2, seq, NB)
            q_km_narrow = axol.atomic.reshape(
                axol.atomic.copy(qk_hm[:, :, 0:1, :, :]), shape=(BHK, seq_len, NB)
            )
            k_km_narrow = axol.atomic.reshape(
                axol.atomic.copy(qk_hm[:, :, 1:2, :, :]), shape=(BHK, seq_len, NB)
            )
            v_km = to_heads(axol.atomic.copy(cs3[:, :, 2 * KB : 2 * KB + VB]), BH)

            # gates: z (head-major), b/a stay heads-in-lanes -- the RVV gate op reads
            # the head lane directly, so the 32 lane_select ops the DPU path needed
            # to scatter b/a into per-head buffers are gone.
            z5 = quantized_matmul_activation(hq, w_z, dq_z, output=BufferInfo(dtype="float32x64"))
            z_all = axol.atomic.reshape(to_heads(z5, BH), shape=(BHS, NB))
            b_T1 = axol.atomic.reshape(
                quantized_matmul_activation(hq, w_b, dq_b, output=BufferInfo(dtype="float32x64")),
                shape=(seq_len, 1),
            )
            a_T1 = axol.atomic.reshape(
                quantized_matmul_activation(hq, w_a, dq_a, output=BufferInfo(dtype="float32x64")),
                shape=(seq_len, 1),
            )

            # L2-norm is row-wise, so normalising the BHK heads actually produced is equivalent
            # to normalising after the repeat below, and cheaper. q additionally carries the
            # delta rule's d^-0.5 factor; the Metis path instead lets the gated norm absorb it
            # via eps = D*eps, which is equivalent only because that norm rescales eps to match.
            ql_narrow = deltanet_l2norm_rvv(
                axol.atomic.reshape(q_km_narrow, shape=(BHK * seq_len, NB)),
                eps=rms_eps,
                scale=float(D) ** -0.5,
                generation=EUROPA,
            )
            kl_narrow = deltanet_l2norm_rvv(
                axol.atomic.reshape(k_km_narrow, shape=(BHK * seq_len, NB)),
                eps=rms_eps,
                generation=EUROPA,
            )
            # GQA: broadcast each key head across its REP value heads (torch's
            # query/key.repeat_interleave(rep, dim=2)); identity when REP == 1.
            ql_all = axol.atomic.reshape(
                _repeat_kv_heads_seq(axol.atomic.reshape(ql_narrow, shape=(BHK, seq_len, NB))),
                shape=(BHS, NB),
            )
            kl_all = axol.atomic.reshape(
                _repeat_kv_heads_seq(axol.atomic.reshape(kl_narrow, shape=(BHK, seq_len, NB))),
                shape=(BHS, NB),
            )

            # Per-token gates in fp32: beta = sigmoid(b), g = -exp(A_log)*softplus(a+dt_bias).
            # g is the LOG decay and stays that way -- the recurrence forms exp(gc_t - gc_i)
            # itself -- so the sigmoid, softplus AND exp LUTs all go away.
            g_all, beta_all = deltanet_gates_rvv(
                a_T1, b_T1, pos_a_w, dt_bias_w, token_mask, num_heads=BH, generation=EUROPA
            )

            # fp32 chunked gated-delta recurrence. The state is L1-resident per head and updated
            # in place, carried across chunks within the call and across calls via
            # recurrent_state.
            #
            # The two Gram matmuls run on the int8 IMC: they are the only matmuls in the chunk
            # algebra free of the sequential state dependency (kkm = K@K^T and qkm = Q@K^T need
            # a chunk's own q/k and nothing else), so they batch over every (head, chunk) at
            # once. They take RAW L2-normed operands, not khat, which keeps the decay out of the
            # int8 entirely -- the recurrence applies it to the Gram output as the rank-one
            # product ah[t]*ea[i]. That costs no accuracy: khat's dynamic range lies along
            # khat^T's columns, which is exactly what a per-column int8 scale absorbs.
            q_g = axol.atomic.reshape(ql_all, shape=GRAM_A_SHAPE)
            k_g = axol.atomic.reshape(kl_all, shape=GRAM_A_SHAPE)
            k_t = axol.core.transpose(k_g, dims=(0, 2, 1))
            kkm = axol.atomic.reshape(
                imc_mm_int8(k_g, k_t, generation=EUROPA), shape=GRAM_OUT_SHAPE
            )
            qkm = axol.atomic.reshape(
                imc_mm_int8(q_g, k_t, generation=EUROPA), shape=GRAM_OUT_SHAPE
            )
            o = deltanet_chunk_rvv(
                axol.atomic.reshape(ql_all, shape=(BH, seq_len, NB)),
                axol.atomic.reshape(kl_all, shape=(BH, seq_len, NB)),
                v_km,
                g_all,
                beta_all,
                kkm,
                qkm,
                recurrent_state,
                chunk=RVV_CHUNK,
                generation=EUROPA,
            )

            # gated RMSNorm tail + SiLU gate in one fp32 op (was inv_sqrt LUT +
            # silu LUT + elementwise_mul, with eps pre-scaled by D).
            core = deltanet_gated_rmsnorm_rvv(
                axol.atomic.reshape(o, shape=(BHS, NB)),
                z_all,
                gnorm_w,
                eps=rms_eps,
                generation=EUROPA,
            )
            core_3 = from_heads(axol.atomic.reshape(core, shape=(BH, seq_len, NB)))
            out = quantized_matmul_activation(
                quant(core_3, qs_core, qz_core),
                w_oproj,
                dq_oproj,
                matrix_b1=w_oproj1,
                output=BufferInfo(dtype="float32x64"),
            )

            # recurrent_state was updated in place by deltanet_chunk_rvv; conv_state
            # likewise by causal_conv1d -- both persistent cache IO.

            # residual add + the SwiGLU MLP tail (qwen3's fused norm+quant pattern; one
            # core for these at prefill, it is compute- not weight-bandwidth-bound).
            residual = axol.atomic.add(residual_in, out)
            gu_q = hidden_norm_quant(
                residual, rms2_consts, gate_proj_qparams, _mlp_dims, POST_ATTN_NORM_FOLD
            )
            mlp_split = fused_gate_up_swiglu(
                gu_q,
                gate_up_proj_w,
                mlp_dq_params,
                matrix_b1=gate_up_proj_w1,
                luts=silu_lut,
                generation=EUROPA,
            )
            d6 = quantized_matmul_activation(
                mlp_split,
                down_proj_w,
                down_dq_params,
                matrix_a_is_split=True,
                matrix_b1=down_proj_w1,
                output=BufferInfo(dtype="float32x64"),
            )
            _tail = axol.atomic.add(residual, d6)
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
    depthwise conv is repacked channel-major; exp(A_log) / dt_bias and the gated-norm
    scale go in with the heads in lanes.

    Both phases run the same fp32 kernel and so ask for the same buffers; nothing here
    is phase-specific.

    Only emits buffers the spec asks for, so it tolerates the kernel growing extra IO
    (e.g. the wrapping RMSNorm + SwiGLU MLP, added by _add_decoder_wrapper_buffers).
    """
    import torch

    from ...layout import (
        layer_conv,
        layer_linears,
        layer_subset,
        pack,
    )
    from ..architecture import get_architecture
    from ..multicore_layer import ddr1_split_shards, ddr1_unsharded_spec

    config = get_architecture(bundle.architecture).config_from_dict(bundle.config)
    PWORD = 64
    BH = config.linear_num_value_heads
    D = config.linear_key_head_dim

    lin = layer_linears(bundle.linear, layer)
    conv = layer_conv(bundle.conv, layer)
    dn = next(iter(layer_subset(bundle.deltanet, layer).values()))  # the GatedDeltaNet params

    # w_qkv/w_oproj's K is split over DDR/DDR1 at decode (see _build_decode_kernel);
    # pack against the full-K spec, then cut the DDR1 shard off the packed result.
    ddr1_names = ("w_qkv", "w_oproj")
    full_spec = ddr1_unsharded_spec(spec, ddr1_names)

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

    # in_proj_qkv: the shared activation quant is reused for z/b/a too (one SmoothQuant
    # group -> identical qs/qz). It is bound as the interleaved, norm-folded
    # qkv_proj_qparams in _add_decoder_wrapper_buffers, where input_layernorm is in
    # scope; hidden_norm_quant applies it, so there are no separate qs_h/qz_h.
    qkv = lin["in_proj_qkv"]
    if "w_qkv" in spec:
        buffers["w_qkv"] = pack(nk(qkv["w"]), full_spec["w_qkv"])
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
        buffers["w_oproj"] = pack(nk(o["w"]), full_spec["w_oproj"])
    if "dq_oproj" in spec:
        buffers["dq_oproj"] = pack(dq_table(o), spec["dq_oproj"])
    ddr1_split_shards(buffers, spec, ddr1_names)

    # Depthwise causal conv weights. QuantConv1d pins the activation quant per-channel
    # and SYMMETRIC, so the dequant is a pure scale ds = s_x*s_w with no dz correction
    # and an int8 zero window is a real zero, which the causal left padding needs.
    conv_p = conv["conv1d"]
    if "w" not in conv_p:
        raise RuntimeError(
            "this kernel needs a QuantConv1d export (qs/w/ds) to recover the conv weight "
            "scale; this bundle carries a float conv weight. Build the model with "
            "quantize_conv=True (the default via Qwen3_5 load_pretrained). Guessing the "
            "activation scale from the in-projection's dequant scale is not an option: it "
            "compiles cleanly and saturates the conv input."
        )
    if "conv_w" in spec:
        # The conv runs in float on the DPU, so bind the FAKE-QUANTIZED float weight
        # (w_int8 * s_w) in the float path's channel-major (K, C_blocks) layout. QuantConv1d
        # exports w (int8), qs (=1/s_x) and ds (=s_x*s_w), so s_w = ds * qs.
        w_i8 = conv_p["w"].detach().to(torch.float32).cpu()  # [C, 1, K]
        ds = conv_p["ds"].detach().to(torch.float32).cpu().reshape(-1)  # [C]
        qs = conv_p["qs"].detach().to(torch.float32).cpu().reshape(-1)  # [C]
        s_w = ds * qs  # [C]
        w_f = w_i8 * s_w.reshape(-1, 1, 1)
        channels, _, k_taps = w_f.shape
        # [C, 1, K] -> [C_blocks, PWORD, K] -> [K, C_blocks, PWORD]
        taps = w_f.reshape(channels // PWORD, PWORD, k_taps).permute(2, 0, 1).contiguous()
        buffers["conv_w"] = pack(taps, spec["conv_w"])
    if "conv_qs" in spec:
        buffers["conv_qs"] = pack(conv_p["qs"], spec["conv_qs"])
    if "conv_ds" in spec:
        buffers["conv_ds"] = pack(conv_p["ds"], spec["conv_ds"])

    # exp(A_log) and dt_bias per head, heads in the low lanes: the RVV gate paths index
    # the head lane directly.
    dt_bias = dn["dt_bias"].detach().to(torch.float32).cpu().numpy()  # [BH]
    posA = dn["A_log"].detach().float().exp().cpu().numpy()  # [BH]

    def head_lanes(v):
        t = torch.zeros((1, 1, PWORD), dtype=torch.float32)
        t[0, 0, :BH] = torch.as_tensor(v).to(torch.float32)
        return t

    put("pos_a_w", head_lanes(posA))
    put("dt_bias_w", head_lanes(dt_bias))
    put(
        "gnorm_w",
        dn["norm_weight"].detach().to(torch.float32).cpu().reshape(1, D // PWORD, PWORD),
    )

    # recurrent_state + conv_state are NOT here -- they are codegen-allocated persistent
    # cache IO (zero-initialised on a fresh instance, carried across calls), not constants.

    # The wrapping decoder layer (input/post norms + SwiGLU MLP) reuses the qwen3 binding.
    _add_decoder_wrapper_buffers(buffers, spec, bundle, layer)
    return buffers


# RMSNorm-weight / LUT sites (module path relative to the decoder layer). Empty of
# norms: both the input and post-attention affine weights are folded into the
# quantize scale that follows them (qkv_proj_qparams / gate_proj_qparams) and their
# inv-sqrt LUTs are carried by rms1_consts / rms2_consts via add_norm_consts.
_WRAP_NORM_SITES: dict[str, str] = {}
_WRAP_LUT_SITES = {"silu_lut": "mlp.silu_range"}


def _add_decoder_wrapper_buffers(buffers, spec, bundle, layer) -> None:
    """Bind the input RMSNorm + fused SwiGLU MLP tail that wrap the mixer.

    Reuses LFM2's fused-MLP packer (covers the composed gate||up weight, the combined
    dequant tables, and down_proj_w) but qwen3's hidden_norm_quant packing for the
    post-attention norm + its quantize scale (rms2_consts + gate_proj_qparams), not
    LFM2's separate weight-buffer + qs/qz scheme.
    No-op when none of the wrapper buffers are in the spec."""
    from ...layout import layer_linears, layer_subset
    from ..lfm2.optimized_impl import _put_fused_mlp, _put_norms_and_luts
    from ..multicore_layer import (
        POST_ATTN_NORM_FOLD,
        add_norm_consts,
        check_norm_folds,
        interleave_qparams,
    )

    lin = layer_linears(bundle.linear, layer)
    norm = layer_subset(bundle.norm, layer)
    rng = layer_subset(bundle.lut_ranges, layer)
    if "gate_proj" not in lin:
        return  # bare-mixer kernel (no wrapping decoder layer) -- nothing to bind

    _put_fused_mlp(buffers, spec, lin)
    _put_norms_and_luts(buffers, spec, norm, rng, _WRAP_NORM_SITES, _WRAP_LUT_SITES)

    hidden_dim = int(bundle.config["hidden_size"])
    add_norm_consts(
        buffers,
        spec,
        bundle,
        layer,
        {
            "rms1_consts": ("input_layernorm.inv_sqrt_range", hidden_dim),
            "rms2_consts": ("post_attention_layernorm.inv_sqrt_range", hidden_dim),
        },
    )
    if "qkv_proj_qparams" in spec:
        # in_proj_qkv's activation scale with the input_layernorm weight folded in, so
        # hidden_norm_quant needs no weight buffer. z/b/a share this scale (one
        # SmoothQuant group), so folding it once covers every consumer of the normed
        # hidden.
        q = lin["in_proj_qkv"]
        buffers["qkv_proj_qparams"] = interleave_qparams(
            q["qs"], q["qz"], spec["qkv_proj_qparams"], norm_weight=norm["input_layernorm"]
        )
        check_norm_folds(buffers, folds=(DELTANET_INPUT_NORM_FOLD,))
    if "gate_proj_qparams" in spec:
        g = lin["gate_proj"]
        buffers["gate_proj_qparams"] = interleave_qparams(
            g["qs"],
            g["qz"],
            spec["gate_proj_qparams"],
            norm_weight=norm["post_attention_layernorm"],
        )
        check_norm_folds(buffers, folds=(POST_ATTN_NORM_FOLD,))


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
    config: "Qwen3_5Config",
    ctx_len: int,
    phase: str,
    chunk_len: int | None = None,
    cores: int = 1,
):
    """Build the gated full-attention decoder layer kernel.

    Three Qwen3.5-specific additions to the qwen3 optimized attention:
    (1) q_proj emits num_heads*head_dim*2 -> split query/gate;
    (2) partial RoPE on the first rotary_dim channels via the swizzle design-P trick;
    (3) sigmoid(gate) output gate before o_proj.
    KV cache, int8 SDPA, and the SwiGLU MLP tail are the standard qwen3/LFM2
    attention layer (head_dim=256 >= 128, no design-P padding needed).

    Attention is llm.sdpa_flash_europa rather than axol.core.scaled_dot_product_attention:
    the core op holds the FULL K/V of one head resident in the IMC, which needs
    head_dim_blocks * ctx_blocks slots and so cannot map ctx beyond ~2304 for this
    model's head_dim=256 (the MVM mapping assert). The flash op tiles the KV axis, making
    the IMC/L1 footprint independent of context length, and shards query heads over
    `cores`. It also drops the exp/inv LUT operands (EXP2/RECIP hardware instructions).
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol.common import BufferInfo
    from axelera.compiler.alto.compiler.config import HardwareGeneration
    import numpy as np

    from types import SimpleNamespace

    from ...ops.elementwise_mul import elementwise_mul
    from ...ops.fused_gate_up_swiglu import fused_gate_up_swiglu
    from ...ops.pointwise_lut import pointwise_lut
    from ...ops.quantized_matmul_activation import quantized_matmul_activation
    from ...ops.sdpa_flash_europa import sdpa_flash_europa
    from ..multicore_layer import POST_ATTN_NORM_FOLD, hidden_norm_quant

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

    # qwen3's post-attention-norm dims for hidden_norm_quant's fused RMSNorm+quantize
    # (see the MLP tail below).
    _mlp_dims = SimpleNamespace(
        hidden_dim=hidden_dim,
        hidden_dim_pw=hidden_pw,
        seq_len=seq,
        use_dpu_hidden_norm=(hidden_pw <= 64),
    )

    # Half of the heaviest weights' K lives in DDR1 so both regions stream at once, and
    # A.DDR1 directly rather than upper_shard_scope() -- see the deltanet kernel for
    # both. Applies to BOTH phases: one constant is packed per (layer, buffer name) and
    # shared by every phase declaring it, so a decode-only split would bind prefill's
    # same-named q_proj_w to decode's half-width tensor.
    _ddr1 = A.DDR1
    q_k_lo = hidden_dim // 2
    k_k_lo = hidden_dim // 2
    v_k_lo = hidden_dim // 2
    o_k_lo = q_out // 2
    gu_k_lo = hidden_dim // 2
    dn_k_lo = intermediate_dim // 2

    def partial_rope(x_heads, n_heads, sin, cos):
        # x_heads (1, n_heads, seq, head_pw): rope the first rotary_pw block(s), pass the rest.
        trig = axol.core.concat(sin, cos, dim=2)  # (1, seq, 2*rotary_pw)
        x_rot = axol.atomic.copy(x_heads[:, :, :, 0:rotary_pw])
        dp = axol.core.swizzle(x_rot, w_fwd)  # (1, n_heads, seq, 2*rotary_pw)
        roped = axol.core.rope(dp, trig)
        rout = axol.core.swizzle(roped, w_bwd)  # (1, n_heads, seq, rotary_pw)
        return axol.core.concat(rout, axol.atomic.copy(x_heads[:, :, :, rotary_pw:head_pw]), dim=3)

    def _build_decode_kernel():
        # Decode-only: N-shard the projections across cores (the SDPA op already does,
        # via cores= above), on top of the DDR1 split every phase gets (above). Prefill
        # keeps one core for these -- it is compute- not weight-bandwidth-bound.
        #
        # Weight-pipelined, as qwen3's multicore_impl does.
        _qma = partial(_nshard_qma, cores, pipeline_weights=True)
        _gate_up = partial(
            _nshard_gate_up, cores, pipeline_weights=True, generation=HardwareGeneration.EUROPA
        )
        _mlp_qma = _qma
        _mlp_gate_up = _gate_up

        @A.ir_module
        class Qwen35AttentionDecodeLayer:
            @A.prim_func
            def main(
                seq_pos: A.Buffer((1,), "uint64x64", scope=A.DDR),
                hidden_states: A.Buffer((1, seq, hidden_pw), "float32x64", scope=A.DDR),
                rms1_w: A.Buffer((hidden_dim,), "float32x64", scope=A.DDR),
                rms1_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
                q_proj_qs: A.Buffer((1, 1, hidden_pw), "float32x64", scope=A.DDR),
                q_proj_qz: A.Buffer((1, 1, hidden_pw), "float32x64", scope=A.DDR),
                q_proj_w: A.Buffer((2 * q_out_pw, q_k_lo), "int8x64", scope=A.DDR),
                q_proj_dq: A.Buffer((1, 1, 2 * 2 * q_out_pw), "float32x64", scope=A.DDR),
                k_proj_w: A.Buffer((kv_out_pw, k_k_lo), "int8x64", scope=A.DDR),
                k_proj_dq: A.Buffer((1, 1, 2 * kv_out_pw), "float32x64", scope=A.DDR),
                v_proj_w: A.Buffer((kv_out_pw, v_k_lo), "int8x64", scope=A.DDR),
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
                sdpa_scores_dq_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
                sdpa_weights_q_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
                sdpa_output_dq_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
                attn_mask_positions: A.Buffer((seq, 2), "float32x64", scope=A.DDR),
                gate_sig_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
                o_proj_qs: A.Buffer((1, 1, q_out_pw), "float16x64", scope=A.DDR),
                o_proj_qz: A.Buffer((1, 1, q_out_pw), "float16x64", scope=A.DDR),
                o_proj_w: A.Buffer((hidden_pw, o_k_lo), "int8x64", scope=A.DDR),
                o_proj_dq: A.Buffer((1, 1, 2 * hidden_pw), "float32x64", scope=A.DDR),
                residual_in: A.Buffer((1, seq, hidden_pw), "float32x64", scope=A.DDR),
                rms2_consts: A.Buffer((4,), "float32x64", scope=A.DDR),
                gate_proj_qparams: A.Buffer((2 * hidden_pw,), "float32x64", scope=A.DDR),
                silu_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
                gate_up_proj_w: A.Buffer((2 * inter_pw, gu_k_lo), "int8x64", scope=A.DDR),
                mlp_dq_params: A.Buffer((1, 1, 6 * inter_pw), "float32x64", scope=A.DDR),
                down_proj_w: A.Buffer((hidden_pw, dn_k_lo), "int8x64", scope=A.DDR),
                down_dq_params: A.Buffer((1, 1, 2 * hidden_pw), "float32x64", scope=A.DDR),
                one: A.Buffer((1,), "float32x64", scope=A.DDR),
                # DDR1 upper shards of the six K-split weights above. Declared last, as in
                # the deltanet decode kernel: the runtime binds inputs pool-grouped.
                q_proj_w1: A.Buffer((2 * q_out_pw, hidden_dim - q_k_lo), "int8x64", scope=_ddr1),
                k_proj_w1: A.Buffer((kv_out_pw, hidden_dim - k_k_lo), "int8x64", scope=_ddr1),
                v_proj_w1: A.Buffer((kv_out_pw, hidden_dim - v_k_lo), "int8x64", scope=_ddr1),
                o_proj_w1: A.Buffer((hidden_pw, q_out - o_k_lo), "int8x64", scope=_ddr1),
                gate_up_proj_w1: A.Buffer(
                    (2 * inter_pw, hidden_dim - gu_k_lo), "int8x64", scope=_ddr1
                ),
                down_proj_w1: A.Buffer(
                    (hidden_pw, intermediate_dim - dn_k_lo), "int8x64", scope=_ddr1
                ),
            ):  # noqa: F821
                eps_c = A.alloc_constant(
                    [9.9999999747524271e-07] * 64, dtype="float32x64", scope=A.DDR
                )
                z_zp = A.alloc_constant(np.zeros((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR)
                qmin_c = A.alloc_constant(
                    np.full((1, 1, 1, 1, 64), -128), "float32x64", scope=A.DDR
                )
                qmax_c = A.alloc_constant(
                    np.full((1, 1, 1, 1, 64), 127), "float32x64", scope=A.DDR
                )
                gate_zero = A.alloc_constant(
                    np.zeros((seq, q_out_pw, 64)), "float32x64", scope=A.DDR
                )

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
                    _qma(
                        qkv_q,
                        q_proj_w,
                        q_proj_dq,
                        matrix_b1=q_proj_w1,
                        output=BufferInfo(dtype="float16x64"),
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
                    _qma(
                        qkv_q,
                        k_proj_w,
                        k_proj_dq,
                        matrix_b1=k_proj_w1,
                        output=BufferInfo(dtype="float16x64"),
                    ),
                    shape=(1, seq, num_kv, head_pw),
                )
                k_heads = axol.core.transpose(k5, dims=(0, 2, 1, 3))
                v5 = axol.atomic.reshape(
                    _qma(
                        qkv_q,
                        v_proj_w,
                        v_proj_dq,
                        matrix_b1=v_proj_w1,
                        output=BufferInfo(dtype="float16x64"),
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
                k_out, v_out = axol.core.update_kv_cache(
                    k_update, v_update, k_cache, v_cache, seq_pos
                )
                q_attn = axol.atomic.dpu_cast(
                    axol.atomic.add(axol.atomic.mul(q_rope, q_attn_qs), q_attn_qz),
                    output=BufferInfo(dtype="int8x64"),
                )
                attn_out = sdpa_flash_europa(
                    q_attn,
                    k_out,
                    v_out,
                    sdpa_scores_dq_scale,
                    sdpa_weights_q_scale,
                    z_zp,
                    qmin_c,
                    qmax_c,
                    sdpa_output_dq_scale,
                    z_zp,
                    z_zp,
                    enable_gqa=True,
                    use_newton_raphson=False,
                    scale=1.0 / (head_dim**0.5),
                    generation=HardwareGeneration.EUROPA,
                    cores=cores,
                    mask_query_positions=attn_mask_positions,
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
                o6 = _qma(
                    axol.atomic.dpu_cast(
                        axol.atomic.add(axol.atomic.mul(gated, o_proj_qs), o_proj_qz),
                        output=BufferInfo(dtype="int8x64"),
                    ),
                    o_proj_w,
                    o_proj_dq,
                    matrix_b1=o_proj_w1,
                    output=BufferInfo(dtype="float32x64"),
                )
                residual = axol.atomic.add(residual_in, o6)
                # SwiGLU MLP tail: qwen3's multicore_impl pattern (fused norm+quant,
                # weight-pipelined N-sharded matmuls) -- see _mlp_dims and _qma/_gate_up.
                gu_q = hidden_norm_quant(
                    residual, rms2_consts, gate_proj_qparams, _mlp_dims, POST_ATTN_NORM_FOLD
                )
                mlp_split = _mlp_gate_up(
                    gu_q,
                    gate_up_proj_w,
                    mlp_dq_params,
                    matrix_b1=gate_up_proj_w1,
                    luts=silu_lut,
                )
                d6 = _mlp_qma(
                    mlp_split,
                    down_proj_w,
                    down_dq_params,
                    matrix_a_is_split=True,
                    matrix_b1=down_proj_w1,
                    output=BufferInfo(dtype="float32x64"),
                )
                A.output(axol.atomic.add(residual, d6))

        return Qwen35AttentionDecodeLayer

    if phase == "decode":
        return _build_decode_kernel()

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
            q_proj_w: A.Buffer((2 * q_out_pw, q_k_lo), "int8x64", scope=A.DDR),
            q_proj_dq: A.Buffer((1, 1, 2 * 2 * q_out_pw), "float32x64", scope=A.DDR),
            k_proj_w: A.Buffer((kv_out_pw, k_k_lo), "int8x64", scope=A.DDR),
            k_proj_dq: A.Buffer((1, 1, 2 * kv_out_pw), "float32x64", scope=A.DDR),
            v_proj_w: A.Buffer((kv_out_pw, v_k_lo), "int8x64", scope=A.DDR),
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
            sdpa_scores_dq_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            sdpa_weights_q_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            sdpa_output_dq_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            # Per-query-row causal-mask positions, not a dense mask: sdpa_flash_europa
            # regenerates the mask in-DPU from these. A runtime input, so one compiled kernel
            # serves every chunk and decode step.
            attn_mask_positions: A.Buffer((seq, 2), "float32x64", scope=A.DDR),
            gate_sig_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            o_proj_qs: A.Buffer((1, 1, q_out_pw), "float16x64", scope=A.DDR),
            o_proj_qz: A.Buffer((1, 1, q_out_pw), "float16x64", scope=A.DDR),
            o_proj_w: A.Buffer((hidden_pw, o_k_lo), "int8x64", scope=A.DDR),
            o_proj_dq: A.Buffer((1, 1, 2 * hidden_pw), "float32x64", scope=A.DDR),
            residual_in: A.Buffer((1, seq, hidden_pw), "float32x64", scope=A.DDR),
            rms2_consts: A.Buffer((4,), "float32x64", scope=A.DDR),
            gate_proj_qparams: A.Buffer((2 * hidden_pw,), "float32x64", scope=A.DDR),
            silu_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            gate_up_proj_w: A.Buffer((2 * inter_pw, gu_k_lo), "int8x64", scope=A.DDR),
            mlp_dq_params: A.Buffer((1, 1, 6 * inter_pw), "float32x64", scope=A.DDR),
            down_proj_w: A.Buffer((hidden_pw, dn_k_lo), "int8x64", scope=A.DDR),
            down_dq_params: A.Buffer((1, 1, 2 * hidden_pw), "float32x64", scope=A.DDR),
            # The output-gate LUT's post-scale (== 1.0). Must be a runtime INPUT, not an
            # in-kernel alloc_constant: pointwise_lut stages its post into L1, and the compiler
            # folds a CONST source into a global l1_const_data that is not namespaced per
            # kernel, so prefill's and decode's would collide at link time.
            one: A.Buffer((1,), "float32x64", scope=A.DDR),
            # DDR1 upper shards, matching the decode kernel's split (see the comment
            # above q_k_lo): must be identical in both phases, not just declared here.
            q_proj_w1: A.Buffer((2 * q_out_pw, hidden_dim - q_k_lo), "int8x64", scope=_ddr1),
            k_proj_w1: A.Buffer((kv_out_pw, hidden_dim - k_k_lo), "int8x64", scope=_ddr1),
            v_proj_w1: A.Buffer((kv_out_pw, hidden_dim - v_k_lo), "int8x64", scope=_ddr1),
            o_proj_w1: A.Buffer((hidden_pw, q_out - o_k_lo), "int8x64", scope=_ddr1),
            gate_up_proj_w1: A.Buffer(
                (2 * inter_pw, hidden_dim - gu_k_lo), "int8x64", scope=_ddr1
            ),
            down_proj_w1: A.Buffer(
                (hidden_pw, intermediate_dim - dn_k_lo), "int8x64", scope=_ddr1
            ),
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
                    qkv_q,
                    q_proj_w,
                    q_proj_dq,
                    matrix_b1=q_proj_w1,
                    output=BufferInfo(dtype="float16x64"),
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
                    qkv_q,
                    k_proj_w,
                    k_proj_dq,
                    matrix_b1=k_proj_w1,
                    output=BufferInfo(dtype="float16x64"),
                ),
                shape=(1, seq, num_kv, head_pw),
            )
            k_heads = axol.core.transpose(k5, dims=(0, 2, 1, 3))
            v5 = axol.atomic.reshape(
                quantized_matmul_activation(
                    qkv_q,
                    v_proj_w,
                    v_proj_dq,
                    matrix_b1=v_proj_w1,
                    output=BufferInfo(dtype="float16x64"),
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
            # No dense mask operand: sdpa_flash_europa synthesises the causal mask in-DPU from
            # attn_mask_positions. An unread input would still be transferred, forfeiting it.
            attn_out = sdpa_flash_europa(
                q_attn,
                k_out,
                v_out,
                sdpa_scores_dq_scale,
                sdpa_weights_q_scale,
                z_zp,
                qmin_c,
                qmax_c,
                sdpa_output_dq_scale,
                z_zp,
                z_zp,
                enable_gqa=True,
                use_newton_raphson=False,
                scale=1.0 / (head_dim**0.5),
                generation=HardwareGeneration.EUROPA,
                cores=cores,
                mask_query_positions=attn_mask_positions,
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
                matrix_b1=o_proj_w1,
                output=BufferInfo(dtype="float32x64"),
            )
            residual = axol.atomic.add(residual_in, o6)
            # SwiGLU MLP tail: qwen3's fused norm+quant pattern (one core at prefill --
            # compute- not weight-bandwidth-bound, see the decode kernel's _qma/_gate_up).
            gu_q = hidden_norm_quant(
                residual, rms2_consts, gate_proj_qparams, _mlp_dims, POST_ATTN_NORM_FOLD
            )
            mlp_split = fused_gate_up_swiglu(
                gu_q,
                gate_up_proj_w,
                mlp_dq_params,
                matrix_b1=gate_up_proj_w1,
                luts=silu_lut,
                generation=HardwareGeneration.EUROPA,
            )
            d6 = quantized_matmul_activation(
                mlp_split,
                down_proj_w,
                down_dq_params,
                matrix_a_is_split=True,
                matrix_b1=down_proj_w1,
                output=BufferInfo(dtype="float32x64"),
            )
            A.output(axol.atomic.add(residual, d6))

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
    from ..multicore_layer import (
        POST_ATTN_NORM_FOLD,
        add_norm_consts,
        check_norm_folds,
        interleave_qparams,
    )

    lin = layer_linears(bundle.linear, layer)
    sdpa = layer_sdpa(bundle.sdpa, layer)
    norm = layer_subset(bundle.norm, layer)
    rng = layer_subset(bundle.lut_ranges, layer)
    buffers: dict = {}

    for proj in ("q_proj", "k_proj", "v_proj", "o_proj"):
        _put_qma_proj(buffers, spec, lin[proj], proj)
    _put_fused_mlp(buffers, spec, lin)
    # The post-attention norm is NOT here: hidden_norm_quant folds its affine weight
    # into gate_proj_qparams's quantize scale instead of a separate weight buffer
    # (qwen3's pattern -- see the add_norm_consts/interleave_qparams call below).
    _put_norms_and_luts(
        buffers,
        spec,
        norm,
        rng,
        {
            "rms1_w": "input_layernorm",
            "q_norm_w": "self_attn.q_norm",
            "k_norm_w": "self_attn.k_norm",
        },
        {
            "rms1_lut": "input_layernorm.inv_sqrt_range",
            "q_norm_lut": "self_attn.q_norm.inv_sqrt_range",
            "k_norm_lut": "self_attn.k_norm.inv_sqrt_range",
            "silu_lut": "mlp.silu_range",
        },
    )
    hidden_dim = int(bundle.config["hidden_size"])
    add_norm_consts(
        buffers,
        spec,
        bundle,
        layer,
        {"rms2_consts": ("post_attention_layernorm.inv_sqrt_range", hidden_dim)},
    )
    if "gate_proj_qparams" in spec:
        g = lin["gate_proj"]
        buffers["gate_proj_qparams"] = interleave_qparams(
            g["qs"],
            g["qz"],
            spec["gate_proj_qparams"],
            norm_weight=norm["post_attention_layernorm"],
        )
        check_norm_folds(buffers, folds=(POST_ATTN_NORM_FOLD,))
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
class Qwen3_5EuropaMulticoreImplementation(DecoderLayerImplementation):
    """Qwen3.5 EUROPA multicore impl: host embed -> hybrid device layers -> host lm_head.

    Two layer kinds (per config.layer_type): linear_attention (gated DeltaNet)
    and full_attention (gated dense attention).
    """

    name = "qwen3.5-europa-multicore"
    architecture = "qwen3.5"

    #: The full_attention kernel takes per-row query POSITIONS, not a dense mask:
    #: sdpa_flash_europa regenerates the causal mask in-DPU from them (see
    #: make_mask_query_positions). Swapped rather than added, because a declared-but-unread
    #: input is still transferred -- keeping attn_mask would forfeit the bandwidth saving.
    shared_inputs = (DecoderLayerImplementation.shared_inputs - {"attn_mask"}) | {
        "attn_mask_positions",
        "token_mask",
    }

    #: The deltanet recurrence is a sequential accumulator, so a padded final prefill
    #: chunk is only safe with the gate mask below zeroing the pad positions; the conv's
    #: K-1 window additionally needs the last real embeddings repeated in the tail slots.
    chunked_remainder_via_padded_chunk = True
    #: linear_conv_kernel_dim - 1: the depthwise short conv's carried window width.
    conv_window_replay = 3
    #: The same K-1: with fewer real rows than the window needs, the window the conv
    #: persists straddles the carried halo, so a shorter remainder must replay through
    #: decode instead. The break-even floor happens to be stricter at every chunk_len we
    #: ship, but that is a perf heuristic and this is the correctness bound.
    min_padded_remainder = 3

    def _token_mask_inputs(self, valid_len: int, width: int) -> dict:
        from ...implementation import make_token_mask

        return {"token_mask": make_token_mask(valid_len, width)}

    def _mask_inputs(self, positions, host_params, context_len: int) -> dict:
        """Per-row query positions in place of the base class's dense causal mask.

        Carries exactly the same information the dense mask did -- the absolute query
        positions, supplied per call -- so chunked prefill and decode keep working with one
        compiled kernel; it is only encoded in 2 scalars per row instead of ctx/64 blocks.
        """
        from ...implementation import make_mask_query_positions

        return {"attn_mask_positions": make_mask_query_positions(positions)}

    #: The head is a QuantLinear on device by default -- see tail_kernel. Nothing has to be
    #: passed at build time; --device-lm-head false opts back out to the host gemv.
    provides_device_head = True

    #: What the shared lm_head kernel is parameterized over: has_qk_norm describes the
    #: full-attention layers (which do carry q/k norms), and it takes the same DPU-reduce
    #: norm arm the layers do.
    spec = ModelSpec(has_qk_norm=True)

    def host_kernels(self) -> list[HostKernel]:
        # The lm_head entry declares the params compute_logits' host gemv reads, which is what
        # a --device-lm-head false build falls back to; _pipeline_stages drops it from the
        # manifest when the device head replaces it.
        return [
            HostKernel("embedding", params=("embed_weight",)),
            HostKernel("lm_head", params=("final_norm_weight", "embed_weight", "rms_eps")),
        ]

    def tail_kernel(self, config):
        """The device lm_head, unless the build opted out (config.device_lm_head=False).

        Decode phase only: the head is on the decode critical path, while prefill scores its
        logits through compute_logits_seq, off that path.

        Worth 19.5 ms/token at Qwen3.5-2B. On the host the head is a 248320x2048 fp16 gemv,
        1.02 GB read per token, measured at 22.50 ms -- 44% of a 51.6 ms decode step, against
        28.83 ms for all 24 device layers together. On device it is int8 and K-split across
        both DDR regions (254.3 MB each), and measures 3.0452 ms at cores=8 (DEVICE_REPORT,
        sd 10 us): 508.6 MB at 167 GB/s, i.e. ~3x the 57 GB/s a decoder layer achieves, since
        this is one N-sharded matmul streaming both regions rather than a dependency chain.
        Bit-exact against its own cores=1 build under either sync mode.
        """
        if not self.uses_device_head(config):
            return None
        from axelera.compiler.alto.compiler.config import SyncMode

        from ..lm_head_kernel import LM_HEAD_KIND, lm_head_buffers, make_lm_head_kernel

        return LayerKernel(
            LM_HEAD_KIND,
            build=partial(make_lm_head_kernel, spec=self.spec),
            layer_buffers=lm_head_buffers,
            phases=("decode",),
            enable_buffer_promotion=True,
            # SOFTWARE sync, as the layer kernels use: the head has the same single-core-norm
            # into N-sharded-8-core-matmul shape that InjectHWTokens miscompiles.
            auto_sync_mode=SyncMode.SOFTWARE,
        )

    def layer_kernels(self, model_config) -> dict[str, LayerKernel]:
        from axelera.compiler.alto.compiler.config import SyncMode

        # SyncMode.SOFTWARE, not the compiler's default HARDWARE, for both kinds. HARDWARE
        # runs the ATEX InjectHWTokens pass, which derives hardware token lines from
        # per-core hazard analysis (transform_atex_kernel runs per AICore kernel and the
        # pass has no notion of cores), leaving cross-core ordering to the barriers alone.
        # These kernels break it because they mix core participant sets -- the RVV ops on
        # one core or a capped subset, the N-sharded matmuls on all of them --
        # so per-core barrier counts differ and linear_attention decode comes out
        # numerically wrong at cores=8. SOFTWARE is bit-exact there at no measured cost.
        #
        # Buffer promotion moves each layer's workspace from DDR to L2, so a cross-core
        # gather is read from L2 rather than broadcast-read from DDR by all 8 cores at once.
        # For deltanet decode it is also what avoids the DMA token hang that DDR-resident
        # workspace shows under a long soak.
        return {
            "linear_attention": LayerKernel(
                "linear_attention",
                build=make_deltanet_layer_kernel,
                layer_buffers=deltanet_layer_buffers,
                phases=("prefill", "decode"),
                # No rope / mask / seq_pos; its only runtime input is the gate mask.
                shared_inputs=frozenset({"token_mask"}),
                # Persistent per-layer cache (zero-initialised, carried across calls):
                # the gated-delta recurrent state and the depthwise-conv window.
                state=frozenset({"recurrent_state", "conv_state"}),
                enable_buffer_promotion=True,
                enable_datapath_fusion=False,  # fusion reorders the hand-built datapaths
                auto_sync_mode=SyncMode.SOFTWARE,  # InjectHWTokens miscompiles this
            ),
            "full_attention": LayerKernel(
                "full_attention",
                build=make_full_attention_layer_kernel,
                layer_buffers=full_attention_layer_buffers,
                phases=("prefill", "decode"),
                # Gated attention binds RoPE / mask positions / seq_pos, but not the
                # deltanet-only token_mask.
                shared_inputs=self.shared_inputs - {"token_mask"},
                state=frozenset({"k_out", "v_out"}),
                enable_buffer_promotion=True,  # see the comment above
                auto_sync_mode=SyncMode.SOFTWARE,  # see the comment above
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


register_implementation(Qwen3_5EuropaMulticoreImplementation())
__all__ = [
    "Qwen3_5EuropaMulticoreImplementation",
    "make_deltanet_layer_kernel",
    "deltanet_layer_buffers",
]

# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Domyn Small (Nemotron) Metis implementation, registered as ``domyn-metis``.

The device half of the bring-up: the alto-script decoder-layer kernel plus the
per-layer constant binding. Built on the shared multicore backbone
(``models.multicore_layer``), so the host contract is the production one -- a single
precombined RoPE trig (sin||cos), an int8 ``{0, -127}`` 1-head broadcast causal mask,
and ``seq_pos`` -- and ``cores>1`` head-shards SDPA.

Three traits set this kernel apart from the Llama/Qwen3 layers, and each is handled
without a new compiler-level primitive:

* **LayerNorm1P, not RMSNorm.** Every norm subtracts the mean and adds a bias, so the
  norm is ``axol.core.layer_norm_two_pass_kahan`` rather than ``rms_norm_kahan``, and
  its **affine is folded into the following activation quantize** instead of riding in
  the norm op -- see ``fold_norm_affine_into_quant``. The single-pass
  ``rms_norm_dpu_reduce`` / ``layer_norm_dpu_reduce`` variants are not usable at this
  width: they hold ``~3 * D_blocks`` vectors resident and hidden 4096 is
  ``D_blocks = 64``, well past the ~32 that fits the DPU register banks (see
  ``multicore_layer.DPU_REDUCE_MAX_D_BLOCKS``). So the norm result is quantized by a
  separate ``llm.quantize`` datapath. ``DOMYN_SPEC`` pins
  ``dpu_reduce_max_d_blocks=0`` so a *small* config (a tiny test model) takes the same
  path as the real one rather than silently diverging.

* **Squared-ReLU MLP, not SwiGLU.** One up projection, no gate, so there is no
  ``fused_gate_up_swiglu`` composed weight. ``relu(x)**2`` and the down-projection's
  input quantize fuse into ``llm.relu2_quantize`` (exact -- a DPU relu and a
  multiply, no LUT), leaving the MLP as matmul -> fused relu²+quantize -> matmul.

* **Partial RoPE** (``partial_rotary_factor``), handled entirely on the host as a
  channel permutation plus a padded frequency table -- see
  ``partial_rope_head_permutation``. No new RoPE kernel.

The final norm is a LayerNorm too, and it lives on the host side of the boundary;
``DomynMetisImplementation.compute_logits``/``compute_logits_seq`` override the
base's RMSNorm.
"""

# flake8: noqa: F821  -- A.Buffer dtype-string annotations look like F821 names

from typing import TYPE_CHECKING

import numpy as np

from ...implementation import HostKernel, register_implementation
from ..multicore_layer import (
    ModelSpec,
    MulticoreDecoderImplementation,
    interleave_qparams,
    layer_dims,
    quantize_activation,
)

if TYPE_CHECKING:
    from .domyn import DomynConfig

PWORD = 64

#: has_qk_norm=False (Nemotron has no QK-Norm: Q/K go projection -> RoPE);
#: has_lm_head=True (the checkpoint is untied, so the host lm_head is its own stage);
#: dpu_reduce_max_d_blocks=0 forces the Kahan norm at every width -- this kernel only
#: builds the two-pass LayerNorm, so the dpu_reduce D-major weight layout must never
#: be selected (it would mis-pack the norm weights for a small test config).
DOMYN_SPEC = ModelSpec(has_qk_norm=False, has_lm_head=True, dpu_reduce_max_d_blocks=0)


# Partial RoPE as a channel permutation


def partial_rope_head_permutation(head_dim: int, rotary_dim: int) -> np.ndarray:
    """Reorder a head's channels so hardware half-rotation RoPE realises *partial* RoPE.

    Returns ``perm`` with ``perm[device_position] = reference_channel``: the row order
    to give ``q_proj`` / ``k_proj`` so that ``axol.core.rope`` -- which is a plain
    half-rotation over PWORD *blocks*, pairing device channel ``p`` with ``p +
    head_dim/2`` -- reproduces Nemotron's partial rotation of only the leading
    ``rotary_dim`` channels.

    Why a permutation is needed at all: the reference pairs channel ``i`` with ``i +
    rotary_dim/2`` *within* the rotary block. At ``head_dim=128``,
    ``rotary_dim=64`` those 64 rotary channels occupy exactly ONE PWORD, so the
    partner of lane ``i`` is lane ``i+32`` of the *same* PWORD -- and the RoPE op can
    only pair whole blocks, never lanes inside one. Re-laying the channels out moves
    each pair's two halves into blocks ``head_dim/2`` apart, which is what the op does
    pair.

    The layout, for ``h = head_dim/2`` and ``r = rotary_dim/2``:

    * device ``p`` in ``[0, r)``   <- reference ``p``       (rotary first half)
    * device ``p + h``             <- reference ``p + r``   (rotary second half)
    * device ``p`` in ``[r, h)``   <- a pass-through channel, paired with another
      pass-through channel at ``p + h``; the frequency table is 0 there, so
      ``cos = 1``, ``sin = 0`` and the pair passes through untouched.

    Attention is unaffected: ``q . k`` sums over all ``head_dim`` channels, and q and
    k are permuted identically, so the score is unchanged. V and ``o_proj`` are not
    permuted (RoPE never touches them). ``rotary_dim == head_dim`` returns the
    identity, i.e. plain RoPE.
    """
    if head_dim % (2 * PWORD) != 0:
        raise ValueError(
            f"head_dim must be a multiple of {2 * PWORD} so head_dim/2 is a whole number "
            f"of PWORDs (the RoPE trig width); got head_dim={head_dim}"
        )
    if rotary_dim % 2 or not 0 < rotary_dim <= head_dim:
        raise ValueError(
            f"rotary_dim must be even and in (0, head_dim]; got rotary_dim={rotary_dim}, "
            f"head_dim={head_dim}"
        )
    half, rot_half = head_dim // 2, rotary_dim // 2
    n_pass_pairs = half - rot_half  # identity pairs holding the pass-through channels
    perm = np.empty(head_dim, dtype=np.int64)
    perm[:rot_half] = np.arange(rot_half)  # rotary, first half
    perm[half : half + rot_half] = np.arange(rot_half, rotary_dim)  # rotary, second half
    # The head_dim - rotary_dim pass-through channels fill the identity pairs in order.
    perm[rot_half:half] = rotary_dim + np.arange(n_pass_pairs)
    perm[half + rot_half :] = rotary_dim + n_pass_pairs + np.arange(n_pass_pairs)
    return perm


def permute_head_channels(tensor, perm: np.ndarray, head_dim: int):
    """Apply ``perm`` to the head-channel axis of a head-major output-channel tensor.

    ``tensor`` is a projection weight ``[num_heads * head_dim, in_features]`` or a
    per-output-channel vector ``[num_heads * head_dim]``; the output channel index is
    ``head * head_dim + channel``, so the permutation applies inside each head.
    """
    import torch

    t = torch.as_tensor(tensor)
    if t.shape[0] % head_dim:
        raise ValueError(f"leading dim {t.shape[0]} is not a multiple of head_dim {head_dim}")
    index = torch.as_tensor(perm, dtype=torch.long)
    return t.reshape(t.shape[0] // head_dim, head_dim, *t.shape[1:])[:, index].reshape(t.shape)


def fold_norm_affine_into_quant(norm_weight, norm_bias, act_scale, act_zero):
    """Fold a LayerNorm's affine into the per-channel quantize that consumes it.

    Returns the ``(scale, zero_point)`` for which

        quantize(normalized) = round(normalized * scale + zero_point)

    equals ``round(layer_norm(x) * act_scale + act_zero)``, using
    ``scale = norm_weight * act_scale`` and ``zero_point = norm_bias * act_scale +
    act_zero``. Algebraically exact -- both are the same affine composed -- and one
    rounding step shorter on the DPU.

    This is not (only) an optimization. ``axol.core.layer_norm_two_pass_kahan`` streams
    its weight and bias buffers, each ``normalized_shape`` PWORD rows: 1 MiB apiece at
    hidden 4096, so 2 MiB of fixed L1 overhead against a 4 MiB L1. That leaves no room
    for even a single PWORD-wide data tile and the op refuses to tile ("a single
    PWORD-wide tile exceeds L1 memory"). Folding removes both buffers from the kernel,
    and the ``llm.quantize`` that follows every norm here applies the affine for free.

    Every norm in this architecture is immediately quantized for the next matmul
    (input_layernorm -> the shared q/k/v quant, post_attention_layernorm -> up_proj),
    so nothing consumes an un-affined norm output. ``norm_weight`` already carries the
    LayerNorm1P unit offset; ``export_norm_weights`` folds it in.
    """
    import torch

    def _vec(v):
        return torch.as_tensor(v).to(torch.float32).reshape(-1)

    weight, bias = _vec(norm_weight), _vec(norm_bias)
    scale, zero = _vec(act_scale), _vec(act_zero)
    return weight * scale, bias * scale + zero


def make_domyn_layer_kernel(
    config: "DomynConfig",
    ctx_len: int,
    phase: str,
    chunk_len: int | None = None,
    cores: int = 1,
):
    """Build the DomynLayerFunctional IRModule for one Domyn Small decoder layer.

    ``ctx_len`` sizes the KV cache / mask; ``phase`` is "decode" (seq_len 1) or
    "prefill" (seq_len = chunk_len or ctx_len). Chunked prefill builds the query side
    for chunk_len positions while the KV cache stays full ctx_len; the host threads the
    cache via ``seq_pos``. ``cores>1`` head-shards SDPA across AI cores.
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol.common import BufferInfo
    from axelera.compiler.alto.axol.utils.lut_chain import ChainedLUTMethod

    from ...ops.quantized_matmul_activation import (
        quantized_matmul_activation,
        quantized_matmul_activation_multicore,
    )
    from ...ops.relu2_quantize import relu2_quantize

    d = layer_dims(
        config,
        head_dim_pw=config.head_dim // PWORD,
        logical_head_dim=config.head_dim,
        ctx_len=ctx_len,
        phase=phase,
        chunk_len=chunk_len,
        spec=DOMYN_SPEC,
    )
    seq_len, hidden_dim, hidden_dim_pw = d.seq_len, d.hidden_dim, d.hidden_dim_pw
    head_dim, head_dim_pw = config.head_dim, d.head_dim_pw
    num_q_heads, num_kv_heads, context_len = d.num_q_heads, d.num_kv_heads, d.context_len
    q_out_dim, q_out_dim_pw, kv_out_dim_pw = d.q_out_dim, d.q_out_dim_pw, d.kv_out_dim_pw
    intermediate_dim, intermediate_dim_pw = d.intermediate_dim, d.intermediate_dim_pw

    # The host RoPE tables and the q/k weight permutation must agree with the kernel on
    # the rotary width, and the permutation only exists for an even rotary_dim inside a
    # head whose half is a whole number of PWORDs. Validate once, here, so a checkpoint
    # this kernel cannot rotate fails at compile time rather than predicting garbage.
    partial_rope_head_permutation(head_dim, config.rotary_dim)

    # Multicore (cores>1): head-shard SDPA and N-shard the two MLP matmuls, which are
    # 134 MiB of the layer's 193 MiB of int8 weights -- and decode is weight-bandwidth
    # bound, so that is where the cores pay. Op + cores kwarg are selected here (trace
    # time) so the body has a single unconditional call per op.
    #
    # split_output is left False (one unified output): the split-N layout would need
    # relu2_quantize to index its per-channel table per core band, and the unified
    # write's cost -- one DMA descriptor per M-row per core -- is 4 descriptors at
    # decode's M=1, i.e. nothing where it matters most.
    _multicore = bool(cores and cores > 1)
    _sdpa_cores_kw = {"cores": cores} if _multicore else {}
    _mlp_matmul = (
        quantized_matmul_activation_multicore if _multicore else quantized_matmul_activation
    )
    _mlp_cores_kw = {"cores": cores} if _multicore else {}
    # LayerNorm epsilon from the checkpoint (Nemotron spells it norm_eps), not a
    # hardcoded constant -- Domyn Small uses 1e-5 where Qwen3/Llama use 1e-6.
    norm_eps = float(config.norm_eps)

    @A.ir_module
    class DomynLayerFunctional:
        @A.prim_func
        def main(
            seq_pos: A.Buffer((1,), "uint64x64", scope=A.DDR),
            hidden_states: A.Buffer((1, seq_len, hidden_dim_pw), "float32x64", scope=A.DDR),
            ln1_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            q_proj_qparams: A.Buffer((2 * hidden_dim_pw,), "float32x64", scope=A.DDR),
            q_proj_w: A.Buffer((q_out_dim_pw, hidden_dim), "int8x64", scope=A.DDR),
            q_proj_dq: A.Buffer((1, 1, 2 * q_out_dim_pw), "float32x64", scope=A.DDR),
            k_proj_w: A.Buffer((kv_out_dim_pw, hidden_dim), "int8x64", scope=A.DDR),
            k_proj_dq: A.Buffer((1, 1, 2 * kv_out_dim_pw), "float32x64", scope=A.DDR),
            v_proj_w: A.Buffer((kv_out_dim_pw, hidden_dim), "int8x64", scope=A.DDR),
            v_proj_dq: A.Buffer((1, 1, 2 * kv_out_dim_pw), "float32x64", scope=A.DDR),
            rope_trig: A.Buffer((1, seq_len, head_dim_pw), "float32x64", scope=A.DDR),
            k_cache_qparams: A.Buffer((2 * head_dim_pw,), "float32x64", scope=A.DDR),
            v_qparams: A.Buffer((2 * head_dim_pw,), "float16x64", scope=A.DDR),
            k_cache: A.Buffer((1, num_kv_heads, context_len, head_dim_pw), "int8x64", scope=A.DDR),
            v_cache: A.Buffer((1, num_kv_heads, context_len, head_dim_pw), "int8x64", scope=A.DDR),
            q_attn_qparams: A.Buffer((2 * head_dim_pw,), "float32x64", scope=A.DDR),
            sdpa_exp_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            sdpa_inv_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            sdpa_scores_dq_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            sdpa_weights_q_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            sdpa_output_dq_scale: A.Buffer((1, 1, 1, 1), "float32x64", scope=A.DDR),
            attn_mask: A.Buffer((1, 1, seq_len, context_len // 64), "int8x64", scope=A.DDR),
            o_qparams: A.Buffer((2 * q_out_dim_pw,), "float16x64", scope=A.DDR),
            o_proj_w: A.Buffer((hidden_dim_pw, q_out_dim), "int8x64", scope=A.DDR),
            o_proj_dq: A.Buffer((1, 1, 2 * hidden_dim_pw), "float32x64", scope=A.DDR),
            residual_in: A.Buffer((1, seq_len, hidden_dim_pw), "float32x64", scope=A.DDR),
            ln2_lut: A.Buffer((2,), "float32x64", scope=A.DDR),
            up_proj_qparams: A.Buffer((2 * hidden_dim_pw,), "float32x64", scope=A.DDR),
            up_proj_w: A.Buffer((intermediate_dim_pw, hidden_dim), "int8x64", scope=A.DDR),
            up_proj_dq: A.Buffer((1, 1, 2 * intermediate_dim_pw), "float32x64", scope=A.DDR),
            down_proj_qparams: A.Buffer((2 * intermediate_dim_pw,), "float32x64", scope=A.DDR),
            down_proj_w: A.Buffer((hidden_dim_pw, intermediate_dim), "int8x64", scope=A.DDR),
            down_proj_dq: A.Buffer((1, 1, 2 * hidden_dim_pw), "float32x64", scope=A.DDR),
        ):  # noqa: F821
            eps_const = A.alloc_constant([norm_eps] * 64, dtype="float32x64", scope=A.DDR)
            # SDPA zero-points + int8 range are zero-pinned constants; the dequant
            # scales arrive as per-layer input buffers.
            sdpa_weights_q_zp = A.alloc_constant(
                np.array([0] * 64).reshape((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR
            )
            sdpa_weights_q_qmin = A.alloc_constant(
                np.array([-128] * 64).reshape((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR
            )
            sdpa_weights_q_qmax = A.alloc_constant(
                np.array([127] * 64).reshape((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR
            )
            sdpa_query_q_zp = A.alloc_constant(
                np.array([0] * 64).reshape((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR
            )
            sdpa_value_q_zp = A.alloc_constant(
                np.array([0] * 64).reshape((1, 1, 1, 1, 64)), "float32x64", scope=A.DDR
            )

            # LayerNorm1P, affine-free: two passes (mean, then variance about it) with
            # Kahan compensation. hidden 4096 is too wide for the register-resident
            # single-pass variant, and too wide to stream the norm's own weight/bias
            # through L1 -- so the gain and bias are folded into the quantize below
            # (fold_norm_affine_into_quant), which is where they are applied.
            ln1 = axol.core.layer_norm_two_pass_kahan(
                hidden_states,
                ln1_lut,
                eps_const[0:1],
                (hidden_dim,),
                invsqrt_method="additive",
            )
            # q/k/v share one SmoothQuant group (identical input qs/qz), so quantize
            # the norm output to int8 once and feed all three matmuls. q_proj_qparams
            # carries input_layernorm's gain/bias folded in.
            qkv_q = quantize_activation(ln1, q_proj_qparams, (seq_len,), hidden_dim_pw)

            q5 = quantized_matmul_activation(
                qkv_q, q_proj_w, q_proj_dq, output=BufferInfo(dtype="float16x64")
            )
            q5_4d = axol.atomic.reshape(q5, shape=(1, seq_len, num_q_heads, head_dim_pw))
            q_heads = axol.core.transpose(q5_4d, dims=(0, 2, 1, 3))

            k5 = quantized_matmul_activation(
                qkv_q, k_proj_w, k_proj_dq, output=BufferInfo(dtype="float16x64")
            )
            k5_4d = axol.atomic.reshape(k5, shape=(1, seq_len, num_kv_heads, head_dim_pw))
            k_heads = axol.core.transpose(k5_4d, dims=(0, 2, 1, 3))

            # No QK-Norm: projection -> RoPE directly. The K chain is authored BEFORE
            # the V projection so the k_heads transpose output is consumed and freed
            # before V stages its in/weight/out, rather than stranding in L1 across the
            # whole V projection (the ordering the Qwen3 multicore kernel needs too).
            # rope_trig's zero-frequency lanes leave the non-rotary channels untouched;
            # the q/k weight permutation put each rotary pair's halves a half-head apart.
            k_roped = axol.core.rope(k_heads, rope_trig)
            k_update = quantize_activation(
                k_roped, k_cache_qparams, (num_kv_heads, seq_len), head_dim_pw
            )

            v5 = quantized_matmul_activation(
                qkv_q, v_proj_w, v_proj_dq, output=BufferInfo(dtype="float16x64")
            )
            v5_4d = axol.atomic.reshape(v5, shape=(1, seq_len, num_kv_heads, head_dim_pw))
            v_heads = axol.core.transpose(v5_4d, dims=(0, 2, 1, 3))
            v_update = quantize_activation(
                v_heads, v_qparams, (num_kv_heads, seq_len), head_dim_pw
            )

            k_out, v_out = axol.core.update_kv_cache(k_update, v_update, k_cache, v_cache, seq_pos)

            q_roped = axol.core.rope(q_heads, rope_trig)
            q_attn = quantize_activation(
                q_roped, q_attn_qparams, (num_q_heads, seq_len), head_dim_pw
            )

            attn_out = axol.core.scaled_dot_product_attention(
                q_attn,
                k_out,
                v_out,
                sdpa_scores_dq_scale,
                sdpa_exp_lut,
                sdpa_inv_lut,
                sdpa_weights_q_scale,
                sdpa_weights_q_zp,
                sdpa_weights_q_qmin,
                sdpa_weights_q_qmax,
                sdpa_output_dq_scale,
                sdpa_query_q_zp,
                sdpa_value_q_zp,
                attn_mask,
                enable_gqa=True,
                use_newton_raphson=False,
                scale=1.0 / (head_dim**0.5),
                neg_exp_method=ChainedLUTMethod.ADDITIVE,
                inv_method=ChainedLUTMethod.ADDITIVE,
                # Fuse the o_proj-input quantize into SDPA's P@V write -> int8 out.
                attention_output_quantizer=o_qparams,
                **_sdpa_cores_kw,
            )

            attn_t = axol.core.transpose(attn_out, dims=(0, 2, 1, 3))
            attn_flat = axol.atomic.reshape(attn_t, shape=(1, seq_len, q_out_dim_pw))

            o6 = quantized_matmul_activation(
                attn_flat, o_proj_w, o_proj_dq, output=BufferInfo(dtype="float32x64")
            )

            residual = axol.atomic.add(residual_in, o6)

            ln2 = axol.core.layer_norm_two_pass_kahan(
                residual,
                ln2_lut,
                eps_const[0:1],
                (hidden_dim,),
                invsqrt_method="additive",
            )
            # up_proj_qparams carries post_attention_layernorm's gain/bias folded in.
            up_q = quantize_activation(ln2, up_proj_qparams, (seq_len,), hidden_dim_pw)

            # float16 out, not float32: this is the widest matmul in the layer
            # (N = intermediate_dim/64 blocks), and its L1 output tile at float32 is
            # 2.7 MiB of a 4 MiB L1 -- enough to make prefill codegen infeasible at any
            # chunk length. Precision is not the constraint here: relu2_quantize squares
            # in float18 (float32's exponent range) and the result is quantized to int8,
            # so float16's 10 mantissa bits on `up` are far finer than the int8 grid it
            # lands on, and float16's range covers this MLP's activations with orders of
            # magnitude to spare.
            up = _mlp_matmul(
                up_q,
                up_proj_w,
                up_proj_dq,
                **_mlp_cores_kw,
                output=BufferInfo(dtype="float16x64"),
            )
            # relu(up)**2 + the down-projection input quantize in one datapath. The
            # reshapes are free views; the intermediate is the widest tensor in the
            # layer, so each avoided pass over it is worth a lot of DDR bandwidth.
            up_2d = axol.atomic.reshape(up, shape=(seq_len, intermediate_dim_pw))
            mlp_q_2d = relu2_quantize(up_2d, down_proj_qparams)
            mlp_q = axol.atomic.reshape(mlp_q_2d, shape=(1, seq_len, intermediate_dim_pw))

            d6 = _mlp_matmul(
                mlp_q,
                down_proj_w,
                down_proj_dq,
                **_mlp_cores_kw,
                output=BufferInfo(dtype="float32x64"),
            )

            A.output(axol.atomic.add(residual, d6))

    return DomynLayerFunctional


# Per-layer constant binding


#: Projection names: both the kernel buffer prefixes (q_proj_*) and the torch
#: submodule leaf names. No gate_proj -- the squared-ReLU MLP has a single up_proj.
_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj")
#: Projections whose output channels carry RoPE, and so are permuted (see
#: partial_rope_head_permutation). V and o_proj are untouched by RoPE.
_ROPE_PROJECTIONS = ("q_proj", "k_proj")
#: qparams buffer -> (norm module path, the projection whose input quant it feeds).
#: Each LayerNorm1P's gain and bias are folded into that quantize rather than bound as
#: their own buffers (see fold_norm_affine_into_quant). export_norm_weights gives a
#: LayerNorm1P explicit .weight / .bias suffixes; an RMSNorm has a bare path.
_FOLDED_NORM_QUANTS = {
    "q_proj_qparams": ("input_layernorm", "q_proj"),  # shared by q/k/v
    "up_proj_qparams": ("post_attention_layernorm", "up_proj"),
}
#: LUT buffer -> the RangeObserver site whose recorded range the table is fit over.
_LUT_SITES = {
    "ln1_lut": "input_layernorm.inv_sqrt_range",
    "ln2_lut": "post_attention_layernorm.inv_sqrt_range",
    "sdpa_exp_lut": "self_attn.sdpa.exp_range",
    "sdpa_inv_lut": "self_attn.sdpa.reciprocal_range",
}
#: SDPA per-tensor scales the kernel takes as broadcast single-PWORD inputs.
_SDPA_SCALARS = {
    "sdpa_scores_dq_scale": "scores_dq_scale",  # q_scale * k_scale
    "sdpa_weights_q_scale": "qs_w",  # softmax-weight quant scale
    "sdpa_output_dq_scale": "output_dq_scale",  # w_scale * v_scale
}


def domyn_layer_buffers(bundle, spec, layer):
    """Bind one Domyn Small layer's constants into the kernel's buffers.

    Weights go in NK layout ``(out//64, in, 64)`` for the b_transposed matmul, output
    dequant as an interleaved ``[dz, ds]`` table, activation quant as an interleaved
    ``[scale, zp]`` table (with each LayerNorm's gain/bias folded into the quant that
    follows it), and the inv_sqrt / softmax LUTs fit over the recorded ranges.

    ``q_proj`` and ``k_proj`` are permuted along their head-channel axis first (weight
    AND per-output-channel dequant together), which is what turns the hardware's
    half-rotation RoPE into Nemotron's partial rotation.
    """
    import torch

    from ...layout import fit_lut, layer_linears, layer_sdpa, layer_subset, pack

    lin = layer_linears(bundle.linear, layer)
    sdpa = layer_sdpa(bundle.sdpa, layer)
    norm = layer_subset(bundle.norm, layer)
    rng = layer_subset(bundle.lut_ranges, layer)

    head_dim = int(bundle.config["head_dim"])
    rotary_dim = int(head_dim * float(bundle.config.get("partial_rotary_factor", 1.0)))
    perm = partial_rope_head_permutation(head_dim, rotary_dim)

    buffers: dict = {}

    def put(name, value):
        if name in spec:
            buffers[name] = pack(value, spec[name])

    def _nk(w):  # [out, in] int8 -> NK (out//64, in, 64): [nb, k, lane] = w[nb*64+lane, k]
        out, inn = w.shape
        return w.reshape(out // PWORD, PWORD, inn).permute(0, 2, 1).contiguous().to(torch.int8)

    def _blocks(v):  # (C,) -> (C//64, 64): a per-channel value packed into the lane
        return torch.as_tensor(v).to(torch.float32).reshape(-1, PWORD).contiguous()

    for proj in _PROJECTIONS:
        p = lin[proj]
        w, dz, ds = p["w"], p["dz"], p["ds"]
        if proj in _ROPE_PROJECTIONS:
            # Permute the weight rows and their per-output-channel dequant together;
            # dropping either half would silently mis-rotate (or mis-scale) the head.
            w = permute_head_channels(w, perm, head_dim)
            dz = permute_head_channels(dz, perm, head_dim)
            ds = permute_head_channels(ds, perm, head_dim)
        if f"{proj}_w" in spec:
            buffers[f"{proj}_w"] = _nk(w)
        name = f"{proj}_dq"
        if name in spec:
            opw = w.shape[0] // PWORD
            dq = torch.empty((2 * opw, PWORD), dtype=torch.float32)
            dq[0::2] = _blocks(dz)
            dq[1::2] = _blocks(ds)
            buffers[name] = dq.reshape(1, 1, 2 * opw, PWORD).contiguous()

    for buf, field in _SDPA_SCALARS.items():
        put(buf, sdpa[field])
    for buf, site in _LUT_SITES.items():
        if buf in spec and site in rng:  # un-quantized attention omits exp/recip ranges
            buffers[buf] = fit_lut(rng[site], spec[buf])

    # Interleaved [scale, zp] activation-quant tables. The matmul-input quants carry
    # the SmoothQuant zero-point; the SDPA quants are zero-pinned (qz=None). The two
    # post-norm quants additionally carry their LayerNorm's gain and bias.
    qparam_sites = {
        "down_proj_qparams": (lin["down_proj"]["qs"], lin["down_proj"]["qz"]),
        "o_qparams": (lin["o_proj"]["qs"], lin["o_proj"]["qz"]),
        "q_attn_qparams": (sdpa["qs_q"], None),
        "k_cache_qparams": (sdpa["qs_k"], None),
        "v_qparams": (sdpa["qs_v"], None),
    }
    for buf, (site, proj) in _FOLDED_NORM_QUANTS.items():
        qparam_sites[buf] = fold_norm_affine_into_quant(
            norm[f"{site}.weight"], norm[f"{site}.bias"], lin[proj]["qs"], lin[proj]["qz"]
        )
    for name, (qs, qz) in qparam_sites.items():
        if name in spec:
            buffers[name] = interleave_qparams(qs, qz, spec[name])
    return buffers


# Implementation


class DomynMetisImplementation(MulticoreDecoderImplementation):
    """Domyn Small on Metis: host embed -> N device decoder layers -> host lm_head."""

    name = "domyn-metis"
    architecture = "domyn"
    spec = DOMYN_SPEC
    layer_build = staticmethod(make_domyn_layer_kernel)
    layer_buffers_fn = staticmethod(domyn_layer_buffers)

    def kernels(self):
        """As the multicore backbone, but the lm_head stage reads the LayerNorm bias
        and the model's own (untied) lm_head weight rather than the tied embedding."""
        ks = [k for k in super().kernels() if k.name != "lm_head"]
        return ks + [
            HostKernel(
                "lm_head",
                params=("final_norm_weight", "final_norm_bias", "lm_head_weight", "rms_eps"),
            )
        ]

    def _trig(self, positions, host_params):
        """Precombined sin||cos RoPE tables for *partial* rotary embeddings.

        ``host_params["rope_inv_freq"]`` holds the inverse frequencies over the ROTARY
        width (``rotary_dim/2`` of them, exported by
        ``DomynArchitecture.export_host_io``). The table is ``head_dim/2`` lanes wide,
        so the tail is padded with frequency **zero** -- which gives ``cos = 1``,
        ``sin = 0``, i.e. an exact identity rotation for the channel pairs that hold
        the pass-through channels. Combined with the q/k head-channel permutation
        (see partial_rope_head_permutation) this makes the plain half-rotation RoPE
        kernel reproduce Nemotron's partial rotation, with no partial-RoPE kernel.
        """
        head_dim = int(host_params["head_dim"])
        inv_freq = np.asarray(host_params["rope_inv_freq"], dtype=np.float64).reshape(-1)
        half = head_dim // 2
        if inv_freq.size > half:
            raise ValueError(
                f"rope_inv_freq has {inv_freq.size} frequencies, more than the {half} "
                f"lanes of a head_dim={head_dim} trig table"
            )
        freqs = np.zeros(half, dtype=np.float64)  # zero freq => identity pair
        freqs[: inv_freq.size] = inv_freq
        pos = np.atleast_1d(np.asarray(positions, dtype=np.float64))
        angles = pos[:, None] * freqs[None, :]
        rope_pw = half // PWORD
        cos = np.cos(angles).astype(np.float32).reshape(1, pos.size, rope_pw, PWORD)
        sin = np.sin(angles).astype(np.float32).reshape(1, pos.size, rope_pw, PWORD)
        return np.ascontiguousarray(np.concatenate([sin, cos], axis=2))

    # The final norm is a LayerNorm1P, not an RMSNorm: it subtracts the mean and adds a
    # bias. The base class's _final_norm implements RMSNorm, which would drop both --
    # and that seam is ALL this card overrides, so the lm_head gemv, its cached fp16
    # cast, the numpy fallback and the device-head branch all stay inherited.

    def _final_norm(self, rows: np.ndarray, host_params: dict) -> np.ndarray:
        """LayerNorm1P over the last axis of ``rows`` [.., hidden].

        ``final_norm_weight`` already carries the 1P unit offset (export_host_io folds it
        in), so it is applied as a plain gain. Variance is the biased mean of squared
        deviations, matching modules.LayerNorm1P (and torch.nn.functional.layer_norm).
        """
        eps = float(host_params["rms_eps"])  # the LayerNorm epsilon (contract name)
        weight = host_params["final_norm_weight"].astype(np.float32)
        bias = host_params["final_norm_bias"].astype(np.float32)
        centered = rows - rows.mean(-1, keepdims=True)
        inv_std = 1.0 / np.sqrt((centered**2).mean(-1, keepdims=True) + eps)
        return centered * inv_std * weight + bias


register_implementation(DomynMetisImplementation())

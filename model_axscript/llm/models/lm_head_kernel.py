# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""The lm_head as a DEVICE kernel: final RMSNorm + quantize + int8 vocab projection.

The alternative is `DecoderLayerImplementation.compute_logits`, an fp16 gemv on the host, whose
cost is additive: at batch 1 the step is a serial chain, since the head needs the last layer's
hidden state and the next step needs the sampled token.

Structurally this is one decoder-layer stage, not a new kind of computation: the final RMSNorm
feeds a per-channel activation quantize whose scale absorbs the norm's affine weight (the same
fold a layer's hidden norms take, see NormFold), then one N-sharded int8 matmul over the
vocabulary. So it reuses hidden_norm_quant and quantized_matmul_activation_multicore unchanged.

Two properties of the vocabulary drive the layout:

  * The N shard does NOT have to divide evenly. `shards` (axol.core.matmul._common) partitions
    N_BLOCKS by divmod and hands the first `remainder` cores one extra block. Nothing is padded,
    so there are no pad rows to produce a spurious logit and the host needs no slice before
    sampling. Only PWORD alignment is required, which every target vocab already satisfies
    (Qwen3 151936 = 2374 * 64, Llama 128256 = 2004 * 64); `split_output=True` would require even
    division, so the head must stay on the default unified-output path.
  * The head is the one weight that must stay int8 under W4A8: per-output-channel int4 breaches
    the perplexity gate, and collapses outright once the per-channel activation scale is folded
    in (the folded weight's dynamic range needs more than 15 levels).
"""

# flake8: noqa: F821  -- A.Buffer dtype-string annotations look like F821 names

from .multicore_layer import PWORD, ModelSpec, NormFold, layer_dims

#: Descriptor kind name (and kernel-symbol stem) for the device head.
LM_HEAD_KIND = "lm_head"

#: Torch path of the model's final RMSNorm, the norm the head folds into its qparams.
FINAL_NORM_PATH = "model.norm"

#: The host-fed input: the last layer's hidden state, as `main` declares it.
LM_HEAD_INPUT = "head_hidden"

#: The final norm folds into the head's quantize scale, exactly as a layer's hidden norms do.
FINAL_NORM_FOLD = NormFold("lm_head_qparams", FINAL_NORM_PATH)


def vocab_blocks(vocab_size: int) -> int:
    """The vocabulary as PWORD output blocks (the matmul's N_BLOCKS)."""
    if vocab_size % PWORD:
        raise ValueError(
            f"vocab_size {vocab_size} is not a multiple of PWORD ({PWORD}); the head would need "
            "pad rows, whose zero weight yields a logit of 0 rather than -inf"
        )
    return vocab_size // PWORD


def make_lm_head_kernel(
    config,
    ctx_len: int,
    phase: str,
    chunk_len=None,
    cores: int = 1,
    generation=None,
    *,
    spec: ModelSpec,
):
    """Build the lm_head IRModule: hidden -> final RMSNorm -> int8 quantize -> logits.

    ``phase`` sizes the query width exactly as a decoder layer does: "decode" is seq_len 1,
    a prefill phase is chunk_len (or ctx_len) positions. The weight and its dequant table are
    phase-independent. ``spec`` is the implementation's, so the head takes the same norm arm
    (dpu_reduce vs Kahan) its decoder layers do. ``generation`` is the compile target, which
    only the caller knows at trace time -- see upper_shard_scope.
    """
    from axelera.compiler.alto import script as A
    from axelera.compiler.alto.axol.common import BufferInfo

    from ..ops.quantized_matmul_activation import (
        quantized_matmul_activation,
        quantized_matmul_activation_multicore,
    )
    from .multicore_layer import hidden_norm_quant, upper_shard_scope

    # The head consumes the hidden state, so only the hidden dims matter; head_dim is
    # irrelevant here but layer_dims requires it, and 1 keeps use_dpu_hidden_norm honest.
    d = layer_dims(
        config,
        head_dim_pw=1,
        logical_head_dim=PWORD,
        ctx_len=ctx_len,
        phase=phase,
        chunk_len=chunk_len,
        spec=spec,
    )
    seq_len, hidden_dim, hidden_dim_pw = d.seq_len, d.hidden_dim, d.hidden_dim_pw
    vocab_pw = vocab_blocks(config.vocab_size)

    _ddr1 = upper_shard_scope(generation)
    _multicore = bool(cores and cores > 1)
    _matmul = quantized_matmul_activation_multicore if _multicore else quantized_matmul_activation
    _cores_kw = {"cores": cores} if _multicore else {}
    # Half of K in DDR1 so both regions stream the vocabulary at once.
    head_k_lo = hidden_dim // 2

    @A.ir_module
    class LMHeadFunctional:
        @A.prim_func
        def main(
            head_hidden: A.Buffer((1, seq_len, hidden_dim_pw), "float32x64", scope=A.DDR),
            final_norm_consts: A.Buffer((4,), "float32x64", scope=A.DDR),
            lm_head_qparams: A.Buffer((2 * hidden_dim_pw,), "float32x64", scope=A.DDR),
            lm_head_w: A.Buffer((vocab_pw, head_k_lo), "int8x64", scope=A.DDR),
            lm_head_dq: A.Buffer((1, 1, 2 * vocab_pw), "float32x64", scope=A.DDR),
            lm_head_w1: A.Buffer((vocab_pw, hidden_dim - head_k_lo), "int8x64", scope=_ddr1),
        ):  # noqa: F821

            head_q = hidden_norm_quant(
                head_hidden, final_norm_consts, lm_head_qparams, d, FINAL_NORM_FOLD
            )
            logits = _matmul(
                head_q,
                lm_head_w,
                lm_head_dq,
                **_cores_kw,
                matrix_b1=lm_head_w1,
                pipeline_weights=phase == "decode",
                output=BufferInfo(dtype="float32x64"),
            )
            A.output(logits)

    return LMHeadFunctional


def lm_head_buffers(bundle, spec, layer=0):
    """Pack the head's constants: NK int8 weight, [dz, ds] dequant table, and the activation
    qparams with the final RMSNorm's affine weight folded into the scale.

    ``layer`` is unused -- the head is a single stage, not a per-layer kind -- but the signature
    matches LayerKernel.layer_buffers so the descriptor binding path is shared.
    """
    import torch

    from .multicore_layer import check_norm_folds, interleave_qparams, norm_consts

    head = bundle.linear.get("lm_head")
    if head is None:
        raise ValueError(
            "bundle has no quantized 'lm_head'; rebuild with quantize_lm_head enabled"
        )

    buffers: dict = {}

    if "lm_head_w" in spec:
        (vocab_pw, k_lo, lane), _ = spec["lm_head_w"]
        k_hi = spec["lm_head_w1"][0][1] if "lm_head_w1" in spec else 0
        in_features = k_lo + k_hi
        # NK layout (out//64, in, 64): [nb, k, lane] = w[nb*64 + lane, k].
        packed = (
            head["w"]
            .reshape(vocab_pw, lane, in_features)
            .permute(0, 2, 1)
            .contiguous()
            .to(torch.int8)
        )
        buffers["lm_head_w"] = packed[:, :k_lo].contiguous()
        if k_hi:
            buffers["lm_head_w1"] = packed[:, k_lo:].contiguous()

    if "lm_head_dq" in spec:
        (_, _, two_vocab_pw, lane), _ = spec["lm_head_dq"]
        vocab_pw = two_vocab_pw // 2

        def _blocks(v):
            return torch.as_tensor(v).to(torch.float32).reshape(vocab_pw, lane).contiguous()

        dq = torch.empty((two_vocab_pw, lane), dtype=torch.float32)
        dq[0::2] = _blocks(head["dz"])
        dq[1::2] = _blocks(head["ds"])
        buffers["lm_head_dq"] = dq.reshape(1, 1, two_vocab_pw, lane).contiguous()

    if "lm_head_qparams" in spec:
        # The final RMSNorm has no weight buffer on device: fold its affine weight into the
        # quantize scale, exactly as a layer's hidden norms fold into q_proj / gate_proj.
        buffers["lm_head_qparams"] = interleave_qparams(
            head["qs"],
            head["qz"],
            spec["lm_head_qparams"],
            norm_weight=bundle.norm[FINAL_NORM_FOLD.site],
        )

    if "final_norm_consts" in spec:
        buffers["final_norm_consts"] = norm_consts(
            bundle,
            bundle.lut_ranges[f"{FINAL_NORM_PATH}.inv_sqrt_range"],
            (int(bundle.config["hidden_size"]),),
            spec["final_norm_consts"],
        )

    check_norm_folds(buffers, (FINAL_NORM_FOLD,))
    return buffers

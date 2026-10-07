# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""End-to-end LLM generation orchestrator: drives a compiled decoder LLM (Qwen3)
through prefill + autoregressive decode on the Axelera runtime executor."""

from __future__ import annotations

from typing import TYPE_CHECKING

from dataclasses import replace

import json
import logging
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .export import (
    export_conv_quant_params,
    export_conv_weights,
    export_deltanet_params,
    export_linear_quant_params,
    export_lut_ranges,
    export_norm_weights,
    export_sdpa_quant_params,
)
from .implementation import get_implementation
from .models.architecture import get_architecture
from .quantized_model import QuantizedModel, architecture_conflict

# Re-exported: the config every model package returns is plain data and lives in a
# stdlib-only module, so reading what a model declares does not import this orchestrator.
from .shared.config import GenerationConfig

# Stdlib-only dataclass (no torch/numpy), so importing the prefill sink here costs nothing
# and keeps this module's import weight unchanged.
from .shared.run_stats import RunStats

# Sub-devices to claim per hardware generation: Metis (OMEGA) has 4, Europa 8. Defined in
# `shared/hardware.py`, which imports nothing, so the front door can refuse an unknown
# generation without importing this module; this is the runtime's name for it.
from ..shared.hardware import CORES_PER_GENERATION as NUM_SUB_DEVICES

if TYPE_CHECKING:
    from ..shared.runtime import RuntimeModel

logger = logging.getLogger(__name__)


def build_quantized_model(architecture: str, model) -> QuantizedModel:
    """Bundle an already-quantized model + its exports into a QuantizedModel.

    Splits the architecture's host params into tensors (embedding / final norm) vs
    scalars (rms_eps / RoPE dims).
    """
    arch = get_architecture(architecture)
    host_tensors: dict = {}
    host_scalars: dict = {}
    for name, value in arch.export_host_io(model).items():
        array = np.asarray(value)
        if array.ndim >= 1:
            host_tensors[name] = torch.from_numpy(np.ascontiguousarray(array))
        else:
            host_scalars[name] = array.item()

    # The exported bundle is serialized to disk + later converted to numpy, so it must be
    # CPU-resident: when the reference model is on cuda the export helpers return cuda tensors,
    # and a downstream .numpy() then raises. Coerce every exported tensor to CPU here
    # (one place) rather than per-consumer.
    def _cpu_tree(obj):
        if hasattr(obj, "detach"):  # torch tensor
            return obj.detach().cpu()
        if isinstance(obj, dict):
            return {k: _cpu_tree(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return type(obj)(_cpu_tree(v) for v in obj)
        return obj

    return QuantizedModel(
        architecture=architecture,
        config=arch.config_to_dict(model.config),
        num_layers=arch.num_layers(model),
        linear=_cpu_tree(export_linear_quant_params(model)),
        sdpa=_cpu_tree(export_sdpa_quant_params(model)),
        norm=_cpu_tree(export_norm_weights(model)),
        lut_ranges=_cpu_tree(export_lut_ranges(model)),
        host_tensors=host_tensors,
        host_scalars=host_scalars,
        # conv holds {path: {field: value}}: full-precision convs export {"weight": w},
        # W8A8 QuantConv1d export {w, qs, ds, ...}. A given conv is one or the other,
        # so the paths never collide.
        conv=_cpu_tree({**export_conv_weights(model), **export_conv_quant_params(model)}),
        deltanet=_cpu_tree(export_deltanet_params(model)),
    )


def export_quantized_model(config: GenerationConfig) -> QuantizedModel:
    """Load + quantize the reference model, returning the exportable bundle.

    The head is quantized into the bundle exactly when this build runs it on device, so the
    bundle and the compile cannot disagree about where the head lives: a compile that binds
    the device head reads `bundle.linear["lm_head"]`, and a bundle without it aborts the
    build after the whole calibration has been paid for.

    The condition is read off the IMPLEMENTATION and the overridden field lives on the
    ARCHITECTURE's config, which are two objects and not one: `qwen3-metis-opt` has the
    field and no device head, and an architecture could carry a device head with no field.
    Passing the override only when the head is on device is what keeps the first case
    working; `test_every_device_head_architecture_accepts_the_override` is what keeps the
    second from arriving unnoticed.
    """
    implementation = get_implementation(config.implementation)
    architecture = implementation.architecture
    arch = get_architecture(architecture)
    overrides = {"quantize_attention": True}
    if implementation.uses_device_head(config):
        overrides["quantize_lm_head"] = True
    model = arch.load_pretrained(
        config.repo_id, dtype=torch.float32, device=config.cal_device, **overrides
    )
    calib = _tokenize_calibration(config)
    arch.quantize(model, calib, smooth_alpha=config.smooth_alpha, qk_alpha=config.qk_alpha)
    arch.record_lut_ranges(model, calib)
    return build_quantized_model(architecture, model)


def _log_context_memory(context_len: int, model_config: dict) -> None:
    """Log what this context_len costs in DDR, so a later DEVICE_LOST is not a surprise.

    Dims come from the exported bundle's own config dict, which is the same thing the kernel
    is built from. Best-effort: a config missing a dim gets no line rather than a failed
    build, since this is reporting, not a gate.
    """
    from .shared.memory import context_memory

    try:
        num_layers = int(model_config["num_hidden_layers"])
        head_dim = int(
            model_config.get("head_dim")
            or model_config["hidden_size"] // model_config["num_attention_heads"]
        )
        memory = context_memory(
            context_len,
            num_layers=num_layers,
            num_kv_heads=int(model_config["num_key_value_heads"]),
            num_q_heads=int(model_config["num_attention_heads"]),
            head_dim=head_dim,
        )
    except (KeyError, TypeError, ValueError, ZeroDivisionError) as e:
        logger.debug("skipped the context-memory estimate (%s)", e)
        return
    logger.info("[axscript] %s", memory.summary())


def build_artifact(
    config: GenerationConfig,
    *,
    quantized_model: Path | None = None,
    save_quantized_model: bool = True,
) -> Path:
    """Export the quantized model, then compile the artifact; returns its directory.

    `save_quantized_model` (default True) persists the intermediate QuantizedModel under
    artifact_dir/quantized_model, so a later build can point `quantized_model` at it
    instead of recalibrating. Pass False to skip that disk cost for a build you know you
    will not reuse.

    Either way this then gates the compile output: a structural cross-layer contract check
    (always) and a runtime load-probe (when a device/simulator is present).

    `quantized_model`, when given, is an existing build's `quantized_model/` directory
    to load instead of quantizing the model.
    """
    from ..shared.artifact_contract import (
        AxScriptArtifactError,
        checkpoint_conflict,
        validate_artifact,
        validate_run_size,
        write_artifact_metadata,
    )
    from ..shared.runtime import probe_runtime_load

    # Before the multi-minute export + compile, not after.
    validate_run_size(config.context_len, config.supported_context_lens, "context_len")
    if quantized_model is not None:
        conflict = checkpoint_conflict(
            Path(quantized_model).parent, config.repo_id
        ) or architecture_conflict(
            quantized_model, get_implementation(config.implementation).architecture
        )
        if conflict:
            raise AxScriptArtifactError(conflict)
        logger.info("[axscript] reusing calibrated quantized_model from %s", quantized_model)
        bundle = QuantizedModel.load(quantized_model)
    else:
        bundle = export_quantized_model(config)
    _log_context_memory(config.context_len, bundle.config)
    if save_quantized_model:
        bundle_dir = Path(config.artifact_dir) / "quantized_model"
        bundle.save(bundle_dir)

        # writes a serialized version of the quant params
        # to store next to the bundle, such that it can be identified
        from .evaluation.metrics import EvalContext

        ctx = EvalContext(config=config)
        EvalContext._write_cache_key(bundle_dir, ctx._quant_key())
    artifact_dir = get_implementation(config.implementation).compile(bundle, config)
    # What this directory was actually compiled at. The eval driver reads it back
    # rather than a build config's declared default, which may not describe THIS artifact.
    write_artifact_metadata(
        artifact_dir, compiled_seq_lens=[config.context_len], model_name=config.repo_id
    )
    validate_artifact(artifact_dir, known_dtypes=_NP_DTYPE.keys())
    probe_runtime_load(artifact_dir)
    return artifact_dir


# model.json dtype strings -> numpy dtypes for buffer allocation.
_NP_DTYPE = {
    "int8": np.int8,
    "int16": np.int16,
    "int32": np.int32,
    "float16": np.float16,
    "float32": np.float32,
    "uint64": np.uint64,
}


def artifact_repo_id(artifact_dir) -> str | None:
    """The HF repo_id a built artifact was made from, so downstream tools can default --repo-id.

    The read itself is `artifact_contract.artifact_checkpoint`, which both entry points use:
    one place decides which files record a checkpoint and in what order. This name stays
    because it is what the eval, chat and benchmark tools already call.
    """
    from ..shared.artifact_contract import artifact_checkpoint

    return artifact_checkpoint(artifact_dir)


def artifact_generation(artifact_dir) -> str:
    """The hardware generation ("OMEGA"/"EUROPA") a built artifact targets.

    Reads pipeline.json's `generation` (written at compile time); defaults to "OMEGA" for
    artifacts built before it was recorded there (OMEGA was always the CLI default).
    """
    pipeline = json.loads((Path(artifact_dir) / "pipeline.json").read_text())
    return pipeline.get("generation") or "OMEGA"


_HOST_PARAMS_CACHE: dict[str, dict] = {}


def _load_host_params(artifact_dir: Path, pipeline: dict) -> dict:
    """Load (and memoize) an artifact's host_params.npz, keyed by path.

    Reused across warmup + timed generation + perplexity calls so the (311 MB) npz is
    read once and downstream caches keyed by array identity (e.g. the fp32 lm_head
    weight) stay warm across calls. A fresh process / recompiled artifact reloads.
    """
    path = Path(artifact_dir) / pipeline["host_params"]
    cached = _HOST_PARAMS_CACHE.get(str(path))
    if cached is None:
        cached = dict(np.load(path))
        _HOST_PARAMS_CACHE[str(path)] = cached
    return cached


#: Run-shape fields the COMPILED artifact decides, which a build-intent config only states
#: an intent for. Both reach the device as a buffer shape, so a disagreement is not a
#: preference but a mis-sized argument.
_COMPILED_RUN_FIELDS = ("context_len", "chunk_len")


def _compiled_run_config(config, pipeline: dict):
    """Return `config` with its run-shape fields taken from the artifact, not the intent.

    `pipeline.json` is the compiled truth; the config is what the card DECLARES it ships.
    The two legitimately differ for any artifact built at a non-default size (an
    `axzoo build --seq-len 1024`, or a context sweep), and the directory layout is the
    same shape either way, so nothing downstream can tell them apart.

    It matters because the host derives buffer shapes from this number:
    `make_causal_mask` reshapes to `(..., context_len // 64, 64)`. Reading the intent for
    llama-3.1-8b -- shipped `context_len` 4096, artifact compiled at 1024 -- built a 4 MiB
    mask for a graph whose input is 1 MiB, and the device rejected the call by name:

        failed to find input handler for arg 3 with size 4194304 offset 0
        AXR_ERROR_INVALID_ARGUMENT: Argument validation failed

    This is the same rule the rest of the run path already follows (the implementation and
    the host params are resolved from `pipeline.json` two lines below); `context_len` was
    the one compiled parameter still being re-specified from the config.
    """
    compiled = {
        f: pipeline[f]
        for f in _COMPILED_RUN_FIELDS
        if pipeline.get(f) is not None and getattr(config, f, None) != pipeline[f]
    }
    if not compiled:
        return config
    stated = {f: getattr(config, f, None) for f in compiled}
    if "chunk_len" in compiled:
        # The card's per-size mapping is a build-time DECLARATION: `GenerationConfig` resolves
        # `chunk_len` from it on every construction, `replace` included, so leaving the list in
        # place here would re-derive the card's width and discard the artifact's. Dropped only
        # when the artifact states a chunk of its own, which is the case this file exists for
        # (an `axzoo build --chunk-len`, or a size the card has since re-mapped).
        compiled["chunk_lens"] = None
    logger.debug("run config follows the artifact: %s (config stated %s)", compiled, stated)
    return replace(config, **compiled)


def _resolve_generation(runtime, tokenizer, artifact_dir, prompt, config, timing=None):
    """Resolve the implementation + host params from the artifact, returning its
    token generator (the implementation owns the decode loop).

    `timing` is the optional `RunStats` prefill sink; it is filled when the returned
    generator is first advanced, not here, because that is when the prefill runs.
    """
    artifact_dir = Path(artifact_dir)
    pipeline = json.loads((artifact_dir / "pipeline.json").read_text())
    host_params = _load_host_params(artifact_dir, pipeline)
    impl = get_implementation(pipeline["implementation"])
    config = _compiled_run_config(config, pipeline)
    return impl.generate(
        runtime, tokenizer, prompt, config, host_params=host_params, timing=timing
    )


def __getattr__(name):
    """Re-export RuntimeModel from the shared runtime, resolved on first access.

    Deferred rather than imported at module level because this tree is vendored and is
    imported under a different package root in each checkout. Here it is
    `model_axscript.llm`, so `..` is `model_axscript`; in the compiler checkout it is
    `tests.integration.alto.llm`, where `..shared` does not exist. Resolving on access keeps
    the upward reference out of import time, so their checkout is unaffected and a re-port of
    this file conflicts visibly rather than breaking quietly.
    """
    if name == "RuntimeModel":
        from ..shared.runtime import RuntimeModel

        return RuntimeModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def run_generation(runtime: RuntimeModel, tokenizer, artifact_dir, prompt: str, config):
    """Drive generation from the compiled artifact; yields decoded text per token."""
    return _resolve_generation(runtime, tokenizer, artifact_dir, prompt, config)


def benchmark_generation(
    runtime, tokenizer, artifact_dir, prompt: str, config, *, warmup: bool = True
) -> dict:
    """Time a generation run: time-to-first-token (s), tokens generated, and
    steady-state decode throughput (tok/s, measured after the first token so it
    excludes prefill). _run_graph is synchronous, so wall-clock here is the real
    end-to-end latency (device step + host embedding / lm_head / sampling).

    When warmup is set (default), a throwaway generation primes the device first so
    TTFT reflects steady prefill latency rather than one-time first-launch overhead
    (kernel setup / buffer allocation). The warmup re-runs prefill from position 0,
    which the timed run then overwrites, so it does not affect the measured output.

    `prefill_s` and its three parts come from the `RunStats` sink the implementation
    fills, so they are the prefill ALONE -- not the `ttft_s` bracket, which additionally
    pays the tokenizer, the argmax and the first `tokenizer.decode` because this loop
    consumes a lazy generator and all of that lands on the first `next()`. The sink is
    fresh per call and never handed the warmup run, whose cold dispatch is the cost the
    warmup exists to exclude.
    """
    if warmup:
        warm = _resolve_generation(runtime, tokenizer, artifact_dir, prompt, config)
        for i, _ in enumerate(warm):
            if i >= 1:  # prefill + one decode step is enough to prime both graphs
                break
        warm.close()
    stats = RunStats()
    gen = _resolve_generation(runtime, tokenizer, artifact_dir, prompt, config, timing=stats)
    start = perf_counter()
    ttft = first_done = None
    tokens = 0
    # The implementation yields CUMULATIVE decoded text, so the last value is the whole
    # answer; it was being discarded. Kept rather than re-generated: a second run to read
    # the text back would cost another prefill+decode and could not be guaranteed to be
    # the run that was timed. Empty string when nothing was yielded at all.
    text = ""
    for tokens, text in enumerate(gen, start=1):
        if ttft is None:
            ttft = perf_counter() - start
            first_done = perf_counter()
    decode_s = perf_counter() - first_done if first_done is not None else 0.0
    return {
        "ttft_s": ttft or 0.0,
        "tokens": tokens,
        "decode_tok_per_s": (tokens - 1) / decode_s if tokens > 1 and decode_s > 0 else 0.0,
        "response": text,
        "prefill_s": stats.prefill_s,
        "prefill_device_s": stats.prefill_device_s,
        "prefill_prep_s": stats.prefill_prep_s,
        "prefill_host_s": stats.prefill_host_s,
    }


_PERPLEXITY_BLOCK = 512  # positions per host lm_head matmul (bounds the [block, vocab] memory)


def device_perplexity(runtime, artifact_dir, input_ids, config) -> float:
    """Teacher-forced perplexity of the compiled (on-device) model over input_ids.

    Scored in windows of the compiled context length (each window starts at
    position 0 and overwrites the KV cache). When a prefill graph is present, each
    window is one device call (the prefill output gives every position's hidden,
    causally conditioned); otherwise it falls back to the decode graph, one device
    call per token (far slower). Reflects the real on-device numerics (LUTs +
    float18), unlike the fake-quant torch proxy.

    `$AXSCRIPT_SCORE_GRAPH=decode` forces the decode graph even where a prefill graph
    exists. The two graphs are meant to agree - a healthy build has prefill-vs-decode
    top-1 agreement of 1.000 - so this is a DIAGNOSTIC, not a second way to report a
    number: it exists because a build whose PREFILL kernel is broken still has a sound
    decode kernel, and scoring through it says what the quantized weights are worth while
    the prefill bug is outstanding (qwen3.5-2b: prefill cosine ~0 against fp, decode
    0.984). Label any number taken this way as decode-scored - it is NOT the path the
    shipped prefill artifact uses, and it costs one device call per token.
    """
    artifact_dir = Path(artifact_dir)
    pipeline = json.loads((artifact_dir / "pipeline.json").read_text())
    host_params = _load_host_params(artifact_dir, pipeline)
    impl = get_implementation(pipeline["implementation"])
    ctx_len = config.context_len

    if hasattr(input_ids, "detach"):  # torch tensor (possibly on cuda) -> host numpy
        input_ids = input_ids.detach().cpu().numpy()
    ids = np.asarray(input_ids).reshape(-1)
    scorable = max(len(ids) - 1, 0)
    import os as _os

    forced = _os.environ.get("AXSCRIPT_SCORE_GRAPH", "").strip().lower()
    if forced not in ("", "prefill", "decode"):
        raise ValueError(f"AXSCRIPT_SCORE_GRAPH={forced!r}: expected 'prefill', 'decode' or unset")
    use_prefill = impl._prefill_graph_name() in getattr(runtime, "graph_names", ())
    if forced == "decode":
        use_prefill = False
    elif forced == "prefill" and not use_prefill:
        raise ValueError("AXSCRIPT_SCORE_GRAPH=prefill but this build has no prefill graph")
    logger.info(
        "on-device perplexity: scoring %d tokens via the %s graph%s",
        scorable,
        "prefill" if use_prefill else "decode",
        (
            " (FORCED by $AXSCRIPT_SCORE_GRAPH; a decode-scored number is not the "
            "shipped prefill path)"
            if forced
            else ""
        ),
    )
    total_nll, n_tokens = 0.0, 0
    for start in range(0, len(ids), ctx_len):
        window = ids[start : start + ctx_len]
        if len(window) < 2:
            continue
        if use_prefill:
            n_window = _score_window_prefill(
                impl, runtime, host_params, window, ctx_len, getattr(config, "chunk_len", None)
            )
        else:
            n_window = _score_window_decode(impl, runtime, host_params, window, ctx_len)
        total_nll += n_window[0]
        n_tokens += n_window[1]
        logger.info("  scored %d / %d tokens", n_tokens, scorable)
    if n_tokens == 0:
        raise ValueError("need at least 2 tokens to measure perplexity")
    return math.exp(total_nll / n_tokens)


def _block_nll(logits: np.ndarray, targets: np.ndarray) -> float:
    """Summed -log_softmax(logits)[targets] over a [n, vocab] block."""
    logits = np.asarray(logits, dtype=np.float64)
    m = logits.max(axis=-1, keepdims=True)
    log_z = m[:, 0] + np.log(np.exp(logits - m).sum(axis=-1))
    tgt = logits[np.arange(len(targets)), targets.astype(int)]
    return float((log_z - tgt).sum())


def _score_window_prefill(
    impl, runtime, host_params, window, ctx_len, chunk_len=None
) -> tuple[float, int]:
    """Score one window with a (possibly chunked) prefill + chunked lm_head."""
    hidden = impl.prefill_hidden(
        runtime, host_params, [int(t) for t in window], ctx_len, chunk_len
    )
    n_pred = len(window) - 1  # positions 0..n_pred-1 predict tokens 1..n_pred
    total = 0.0
    for lo in range(0, n_pred, _PERPLEXITY_BLOCK):
        hi = min(lo + _PERPLEXITY_BLOCK, n_pred)
        logits = impl.compute_logits_seq(
            hidden[:, lo:hi], host_params, runtime=runtime
        )  # [hi-lo, vocab]
        total += _block_nll(logits, window[lo + 1 : hi + 1])
    return total, n_pred


def _score_window_decode(impl, runtime, host_params, window, ctx_len) -> tuple[float, int]:
    """Score one window with the decode graph, one device call per token."""
    # Each window is an INDEPENDENT sequence starting at position 0; the deltanet recurrent/conv
    # state are accumulators (not position-indexed like the KV cache), so reset them or state leaks
    # from the previous window and corrupts the score. (DECODE_RESET=0 disables, for A/B testing.)
    import os as _os

    if _os.environ.get("DECODE_RESET", "1") == "1" and hasattr(runtime, "reset_state"):
        runtime.reset_state()
    total = 0.0
    for pos in range(len(window) - 1):
        logits = impl.decode_step(runtime, host_params, int(window[pos]), pos, ctx_len)
        total += _block_nll(np.asarray(logits).reshape(1, -1), window[pos + 1 : pos + 2])
    return total, len(window) - 1


def _logit_consistency(pre: np.ndarray, dec: np.ndarray) -> tuple[float, float, bool]:
    """One position: rel_l2, KL(softmax(pre)||softmax(dec)), top-1 (argmax) agreement."""
    pre = np.asarray(pre, dtype=np.float64).reshape(-1)
    dec = np.asarray(dec, dtype=np.float64).reshape(-1)
    rel_l2 = float(np.linalg.norm(dec - pre) / (np.linalg.norm(pre) + 1e-12))
    p = np.exp(pre - pre.max())
    p /= p.sum()
    q = np.exp(dec - dec.max())
    q /= q.sum()
    kl = float(np.sum(p * (np.log(p + 1e-12) - np.log(q + 1e-12))))
    return rel_l2, kl, bool(np.argmax(pre) == np.argmax(dec))


def decode_prefill_consistency(
    runtime,
    artifact_dir,
    input_ids,
    config,
    max_positions: int = 1024,
    bins: int = 8,
    verbose: bool = False,
    reset_recurrent_state: bool = False,
) -> dict[str, float]:
    """Prefill<->decode logit-consistency check (decode numerical soundness, no reference).

    device_perplexity validates only the PREFILL kernel (it takes the prefill graph when
    present). This drives the DECODE kernel over the same window and compares its per-position
    next-token logits against the prefill kernel's: any divergence is purely the decode path
    (sdpa_decode + incremental int8 KV cache).

    Decode runs FIRST on the fresh (zero-init) cache, exactly how real generation runs it,
    then prefill provides the reference. (Decoding over a prefill-dirtied cache would mis-test a
    state generation never sees, and reset_state OOMs L2 at large ctx.) We also score BOTH paths'
    top-1 against the actual next token, which tells us which path is sound in absolute terms (a
    healthy model lands ~0.4-0.55 greedy top-1 on natural text), so a cross-path disagreement
    can't be misread.

    Returns flat scalars (suitable for the eval suite): rel_l2 mean/p95/max, KL mean/p95, cross
    top-1 agreement, each path's top-1-vs-actual accuracy, and drift indicators.
    """
    artifact_dir = Path(artifact_dir)
    pipeline = json.loads((artifact_dir / "pipeline.json").read_text())
    host_params = _load_host_params(artifact_dir, pipeline)
    impl = get_implementation(pipeline["implementation"])
    ctx_len = config.context_len
    if impl._prefill_graph_name() not in getattr(runtime, "graph_names", ()):
        raise ValueError("decode_prefill_consistency needs both prefill and decode graphs")

    if hasattr(input_ids, "detach"):
        input_ids = input_ids.detach().cpu().numpy()
    ids = np.asarray(input_ids).reshape(-1)
    k = min(max_positions, len(ids) - 1, ctx_len - 1)
    if k < 2:
        raise ValueError("need at least 2 tokens for the consistency check")
    window = ids[: k + 1]
    targets = window[1 : k + 1]  # actual next token at each position

    # --- DECODE on the fresh cache (the instance is zero-init at load; matches generation). ---
    if reset_recurrent_state and hasattr(runtime, "reset_state"):
        runtime.reset_state()
    dec_logits = None
    dec_top1_actual = np.empty(k, dtype=bool)
    for pos in range(k):
        d = np.asarray(
            impl.decode_step(runtime, host_params, int(window[pos]), pos, ctx_len)
        ).reshape(-1)
        if dec_logits is None:
            dec_logits = np.empty((k, d.shape[0]), dtype=np.float32)
        dec_logits[pos] = d
        dec_top1_actual[pos] = int(np.argmax(d)) == int(targets[pos])
        if verbose and (pos + 1) % 256 == 0:
            logger.info(
                "  decode %d/%d  top1-vs-actual~%.3f",
                pos + 1,
                k,
                dec_top1_actual[: pos + 1].mean(),
            )

    # --- PREFILL reference (one call; overwrites the cache, but decode is already captured). ---
    chunk_len = getattr(config, "chunk_len", None)
    hidden = impl.prefill_hidden(
        runtime, host_params, [int(t) for t in window[:k]], ctx_len, chunk_len
    )
    # Through the SAME head decode used, so what follows is the decode path alone.
    pre = np.asarray(
        impl.compute_logits_seq(hidden[:, :k], host_params, runtime=runtime)
    )  # [k, vocab]
    head = _head_consistency(impl, pre, hidden[:, :k], host_params, runtime)

    rel = np.empty(k)
    kls = np.empty(k)
    top1 = np.empty(k, dtype=bool)
    pre_top1_actual = np.empty(k, dtype=bool)
    for pos in range(k):
        rel[pos], kls[pos], top1[pos] = _logit_consistency(pre[pos], dec_logits[pos])
        pre_top1_actual[pos] = int(np.argmax(pre[pos])) == int(targets[pos])

    edges = np.linspace(0, k, bins + 1, dtype=int)
    first_bin = rel[edges[0] : edges[1]].mean() if edges[1] > edges[0] else rel.mean()
    last_lo = edges[bins - 1]
    if verbose:
        logger.info(
            "decode/prefill consistency over %d positions  (prefill top1-vs-actual=%.3f, "
            "decode top1-vs-actual=%.3f):",
            k,
            pre_top1_actual.mean(),
            dec_top1_actual.mean(),
        )
        for b in range(bins):
            lo, hi = edges[b], edges[b + 1]
            if hi > lo:
                logger.info(
                    "  pos %5d-%-5d  rel_l2=%.4f KL=%.4f cross-top1=%.1f%% "
                    "(pre-acc=%.1f%% dec-acc=%.1f%%)",
                    lo,
                    hi,
                    rel[lo:hi].mean(),
                    kls[lo:hi].mean(),
                    top1[lo:hi].mean() * 100,
                    pre_top1_actual[lo:hi].mean() * 100,
                    dec_top1_actual[lo:hi].mean() * 100,
                )
    return {
        "decode_consistency_positions": float(k),
        "decode_rel_l2_mean": float(rel.mean()),
        "decode_rel_l2_p95": float(np.percentile(rel, 95)),
        "decode_rel_l2_max": float(rel.max()),
        "decode_kl_mean": float(kls.mean()),
        "decode_kl_p95": float(np.percentile(kls, 95)),
        "cross_top1_agreement": float(top1.mean()),
        "prefill_top1_vs_actual": float(pre_top1_actual.mean()),
        "decode_top1_vs_actual": float(dec_top1_actual.mean()),
        "cross_top1_last_bin": float(top1[last_lo:].mean()),
        "decode_rel_l2_last_over_first": float(rel[last_lo:].mean() / (first_bin + 1e-12)),
        **head,
    }


def _head_consistency(impl, dev, hidden, host_params, runtime) -> dict[str, float]:
    """The device lm_head (``dev``, already scored) against the host fp16 gemv on the same hidden.

    Its own signal: the decode<->prefill comparison runs both sides through whichever head
    the artifact has, so the head's quantization goes unmeasured unless reported here.
    Empty when the artifact has no device head.
    """
    if not impl.has_device_head(runtime):
        return {}
    host = np.asarray(impl.compute_logits_seq(hidden, host_params))
    k = dev.shape[0]
    rel = np.empty(k)
    top1 = np.empty(k, dtype=bool)
    for pos in range(k):
        rel[pos], _kl, top1[pos] = _logit_consistency(host[pos], dev[pos])
    return {
        "head_rel_l2_mean": float(rel.mean()),
        "head_rel_l2_max": float(rel.max()),
        "head_top1_agreement": float(top1.mean()),
    }


def load_tokenizer(repo_id: str):
    """AutoTokenizer.from_pretrained with a fallback for checkpoints whose
    tokenizer_config names a tokenizer_class this transformers version doesn't have
    (e.g. LFM2.5's "TokenizersBackend"). The tokenizer.json itself is a standard fast
    tokenizer, so load it directly and re-attach the special tokens from the config so
    eos/bos detection still works for generation.
    """
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(repo_id)
    except ValueError as e:
        if "does not exist or is not currently imported" not in str(e):
            raise
        import json

        from transformers import PreTrainedTokenizerFast

        from .checkpoint import repo_file

        special = {}
        chat_template = None
        try:  # carry eos/bos/pad/unk so the decode loop can detect stop tokens
            cfg = json.load(open(repo_file(repo_id, "tokenizer_config.json")))
            for key in ("eos_token", "bos_token", "pad_token", "unk_token"):
                val = cfg.get(key)
                tok = val.get("content") if isinstance(val, dict) else val
                if tok:
                    special[key] = tok
            # Carry model_input_names too: PreTrainedTokenizerFast defaults to including
            # token_type_ids, which a decoder-only LM does not accept, so `generate(**tok(
            # text))` raises "model_kwargs are not used by the model: ['token_type_ids']"
            # -- the exact call --run and the chat path make. LFM2.5-350M declares
            # ["input_ids", "attention_mask"]; honour whatever the checkpoint says.
            if cfg.get("model_input_names"):
                special["model_input_names"] = cfg["model_input_names"]
            chat_template = cfg.get("chat_template")  # may be inlined here
        except Exception:  # noqa: BLE001 - special tokens are best-effort
            pass
        if chat_template is None:
            # Newer checkpoints (e.g. LFM2.5) ship the template as a separate
            # chat_template.jinja rather than inlined in tokenizer_config.json. Without
            # it AutoTokenizer's fallback loses chat formatting, which not only blocks
            # --apply-chat-template but silently degrades calibration (build_calibration
            # gates chat sources on tokenizer.chat_template), leaving the assistant-turn
            # / eos regime out-of-distribution and mis-quantising the stop token.
            try:
                chat_template = open(repo_file(repo_id, "chat_template.jinja")).read()
            except Exception:  # noqa: BLE001 - not all checkpoints have one
                pass
        return PreTrainedTokenizerFast(
            tokenizer_file=repo_file(repo_id, "tokenizer.json"),
            chat_template=chat_template,
            **special,
        )


def _tokenize_calibration(config: GenerationConfig, tokenizer=None) -> list:
    """Build calibration samples as [1, <=window] input tensors.

    When config.calibration_texts is set, splits that raw text into
    config.num_calib_samples windows of config.calib_window tokens. Otherwise builds
    the diverse, chat-aware mix from config.calib_sources (see calibration.py).
    """
    if tokenizer is None:
        tokenizer = load_tokenizer(config.repo_id)

    window = config.calib_window
    if config.calibration_texts:
        ids = tokenizer("\n\n".join(config.calibration_texts), return_tensors="pt").input_ids[0]
        n = min(config.num_calib_samples, ids.numel() // window)
        if n == 0:  # corpus shorter than one window
            samples = [ids.unsqueeze(0)]
        else:
            samples = [ids[i * window : (i + 1) * window].unsqueeze(0) for i in range(n)]
        return [b.to(config.cal_device) for b in samples]

    from .calibration import build_calibration, distribute_budget

    sources = (
        distribute_budget(config.calib_budget, config.calib_sources)
        if config.calib_budget
        else config.calib_sources
    )
    samples = build_calibration(sources, tokenizer, window)
    if not samples:  # every source unavailable -> minimal raw fallback
        import datasets

        ds = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
        ids = tokenizer("\n\n".join(t for t in ds["text"] if t.strip()), return_tensors="pt")
        samples = [ids.input_ids[0][:window].unsqueeze(0)]
    return [b.to(config.cal_device) for b in _stack(samples, config.calib_batch)]


def _stack(samples: list, batch: int) -> list:
    """Group same-length calibration windows into [batch, window] tensors.

    Done here rather than in calibrate() so every consumer of the list benefits -- the two
    calibration passes, the LUT-range recording and the decode-scale recording all iterate
    it. Only equal-length windows are grouped; a short tail (a source that ran out) stays
    on its own rather than being padded, since padding would feed the observers values the
    model will never see.
    """
    if batch <= 1:
        return samples
    by_len: dict = {}
    for s in samples:
        by_len.setdefault(tuple(s.shape[1:]), []).append(s)
    out = []
    for group in by_len.values():
        for i in range(0, len(group), batch):
            chunk = group[i : i + batch]
            out.append(torch.cat(chunk, dim=0) if len(chunk) > 1 else chunk[0])
    return out


def main(config: GenerationConfig, prompt: str | None = None) -> str:
    from ..shared.runtime import RuntimeModel

    # None takes the model's own declared seed prompt rather than a literal of this
    # module's, so there is one place a default prompt is written down.
    if prompt is None:
        prompt = config.prompt
    artifact_dir = build_artifact(config)
    tokenizer = load_tokenizer(config.repo_id)
    runtime = RuntimeModel(artifact_dir / "model.json", NUM_SUB_DEVICES[config.generation])
    text = ""
    for text in run_generation(runtime, tokenizer, artifact_dir, prompt, config):
        pass
    return text


if __name__ == "__main__":
    print(main(GenerationConfig()))

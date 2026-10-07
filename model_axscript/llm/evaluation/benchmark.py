# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Head-to-head performance benchmark of registered implementations.

Quantizes the reference model once with minimal calibration (a single forward),
saves the QuantizedModel bundle, then compiles and times each implementation from
that one bundle. The only variable across runs is the implementation, so TTFT and
decode tok/s differences are attributable to it.

Two ways to run, both sharing the exact same quantized weights:

- Both in one process (sequential; device handle released between them):

      python -m ...llm.evaluation.benchmark \\
          --implementations qwen3-metis qwen3-metis-opt --max-new-tokens 64

- One impl per process (sidesteps the AICORE_SIM / device singleton). The first
  run quantizes and saves the bundle; later runs reuse it:

      python -m ...llm.evaluation.benchmark --implementations qwen3-metis
      python -m ...llm.evaluation.benchmark --implementations qwen3-metis-opt
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import gc
import json
import logging
from pathlib import Path

from ..generate import (
    NUM_SUB_DEVICES,
    GenerationConfig,
    RuntimeModel,
    benchmark_generation,
    export_quantized_model,
)
from ..implementation import get_implementation
from ..quantized_model import QuantizedModel

logger = logging.getLogger(__name__)

# A single short calibration forward (no dataset download); yields valid activation
# scales to compile and run, but is not accuracy-faithful. Use evaluation.suite for perplexity.
_MINIMAL_CALIB_TEXT = "The quick brown fox jumps over the lazy dog."

_DEFAULT_IMPLEMENTATIONS = ("qwen3-metis", "qwen3-metis-opt")


def _shared_bundle(
    config: GenerationConfig, bundle_dir: Path, *, requantize: bool
) -> QuantizedModel:
    """Quantize the reference model once (minimal calibration) and persist it.

    Reuses a previously saved bundle at bundle_dir unless requantize is set,
    so repeated invocations (or per-impl processes) share identical quantized weights.
    """
    bundle_dir = Path(bundle_dir)
    if (bundle_dir / "quantized_model.json").exists() and not requantize:
        logger.info("reusing saved quantized bundle at %s", bundle_dir)
        return QuantizedModel.load(bundle_dir)
    logger.info("quantizing reference model (single-forward calibration)")
    bundle = export_quantized_model(config)
    bundle.save(bundle_dir)
    logger.info("saved quantized bundle to %s", bundle_dir)
    return bundle


def _compile_and_time(impl_name, bundle, cfg, tokenizer, prompt, recompile) -> dict:
    # Resolved HERE rather than at the caller: `cfg` is the per-implementation config, so a
    # `None` prompt follows whatever each one declares instead of whichever was built first.
    """Compile one implementation from the shared bundle, then time generation.

    The runtime (device handle) is released before returning so the next
    implementation in a same-process run can connect cleanly.
    """
    impl = get_implementation(impl_name)
    art_dir = Path(cfg.artifact_dir)
    if (art_dir / "model.json").exists() and not recompile:
        logger.info("[%s] reusing compiled artifact at %s", impl_name, art_dir)
    else:
        logger.info(
            "[%s] compiling (context_len=%d, phases=%s) -- this can take a while",
            impl_name,
            cfg.context_len,
            tuple(cfg.phases),
        )
        art_dir = impl.compile(bundle, cfg)
    runtime = RuntimeModel(art_dir / "model.json", NUM_SUB_DEVICES[cfg.generation])
    try:
        logger.info("[%s] timing generation", impl_name)
        result = benchmark_generation(
            runtime, tokenizer, art_dir, cfg.prompt if prompt is None else prompt, cfg
        )
        logger.info("  [%s] %s", impl_name, ", ".join(f"{k}={v:.4g}" for k, v in result.items()))
        return result
    finally:
        del runtime
        gc.collect()  # drop the device handle before the next implementation


def benchmark_implementations(
    implementations,
    *,
    repo_id: str = "Qwen/Qwen3-0.6B",
    target: str = "device",
    context_len: int = 2048,
    max_new_tokens: int = 64,
    phases=("prefill", "decode"),
    prompt: str | None = None,  # None => the model's declared `GenerationConfig.prompt`
    artifact_root="build/qwen3-bench",
    requantize: bool = False,
    recompile: bool = False,
    generation: str = "OMEGA",
) -> dict:
    """Quantize once, then compile and time each implementation from the shared bundle.

    Returns {implementation: result-dict} with {"error": ...} for any failed impl.
    Implementations must share an architecture (the bundle is architecture-specific).
    Per-impl failures are captured so the remaining implementations still run.
    """
    artifact_root = Path(artifact_root)
    archs = {get_implementation(i).architecture for i in implementations}
    if len(archs) != 1:
        raise ValueError(
            f"implementations span multiple architectures {archs}; the shared "
            "quantized bundle is architecture-specific"
        )

    # The quantize/calibration config is implementation-independent (keyed by architecture).
    base = GenerationConfig(
        implementation=implementations[0],
        repo_id=repo_id,
        target=target,
        context_len=context_len,
        max_new_tokens=max_new_tokens,
        phases=tuple(phases),
        calibration_texts=[_MINIMAL_CALIB_TEXT],
        num_calib_samples=1,
        generation=generation,
    )
    bundle = _shared_bundle(base, artifact_root / "quantized_model", requantize=requantize)

    from ..generate import load_tokenizer

    tokenizer = load_tokenizer(repo_id)

    results: dict[str, dict] = {}
    for impl_name in implementations:
        cfg = replace(base, implementation=impl_name, artifact_dir=artifact_root / impl_name)
        try:
            results[impl_name] = _compile_and_time(
                impl_name, bundle, cfg, tokenizer, prompt, recompile
            )
        except Exception as exc:  # noqa: BLE001 -- record + continue to the next impl
            logger.warning("  %s: failed (%s)", impl_name, exc)
            results[impl_name] = {"error": f"{type(exc).__name__}: {exc}"}
    return results


def format_report(results: dict) -> str:
    """Render the benchmark results as a text table with speedup vs the first implementation."""
    lines = ["benchmark (TTFT + steady-state decode throughput):"]
    impls = list(results)
    baseline = next((results[i] for i in impls if "error" not in results[i]), None)
    baseline_name = next((i for i in impls if "error" not in results[i]), None)
    for impl, res in results.items():
        if "error" in res:
            lines.append(f"  {impl}: unavailable ({res['error']})")
            continue
        line = (
            f"  {impl}: ttft={res['ttft_s']:.4g}s, "
            f"decode={res['decode_tok_per_s']:.4g} tok/s, tokens={res['tokens']}"
        )
        if baseline and res is not baseline and baseline.get("decode_tok_per_s"):
            line += f"  ({res['decode_tok_per_s'] / baseline['decode_tok_per_s']:.3g}x vs {baseline_name})"
        lines.append(line)
    return "\n".join(lines)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m ...llm.evaluation.benchmark",
        description="Benchmark registered LLM implementations from one shared quantized model.",
    )
    p.add_argument(
        "--implementations",
        nargs="+",
        default=list(_DEFAULT_IMPLEMENTATIONS),
        help="registered implementation names to compare (must share an architecture)",
    )
    p.add_argument("--repo-id", default="Qwen/Qwen3-0.6B", help="HuggingFace checkpoint id")
    p.add_argument("--context-len", type=int, default=2048, help="static context length")
    p.add_argument("--max-new-tokens", type=int, default=64, help="tokens to generate when timing")
    p.add_argument("--target", choices=("device", "sim"), default="device")
    p.add_argument(
        "--generation",
        default="OMEGA",
        choices=("OMEGA", "EUROPA"),
        help="target hardware generation; sets the number of sub-devices",
    )
    p.add_argument(
        "--phases",
        nargs="+",
        default=["prefill", "decode"],
        help="device-kernel phases to build; pass 'decode' to skip the prefill kernel",
    )
    p.add_argument(
        "--prompt",
        default=None,
        help="seed prompt for timing (default: the model's declared GenerationConfig.prompt)",
    )
    p.add_argument(
        "--artifact-root", default="build/qwen3-bench", help="root for bundle + artifacts"
    )
    p.add_argument(
        "--requantize",
        action="store_true",
        help="re-quantize even if a saved bundle exists (default: reuse it)",
    )
    p.add_argument(
        "--recompile",
        action="store_true",
        help="recompile even if an implementation's artifact already exists",
    )
    p.add_argument("--json", action="store_true", help="emit JSON instead of a text report")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="status logging verbosity",
    )
    return p


def main(argv=None) -> dict:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger().setLevel(args.log_level)
    logger.info(
        "benchmarking %s (repo=%s, context_len=%d, target=%s, phases=%s, max_new_tokens=%d)",
        args.implementations,
        args.repo_id,
        args.context_len,
        args.target,
        args.phases,
        args.max_new_tokens,
    )
    results = benchmark_implementations(
        args.implementations,
        repo_id=args.repo_id,
        target=args.target,
        context_len=args.context_len,
        max_new_tokens=args.max_new_tokens,
        phases=tuple(args.phases),
        prompt=args.prompt,
        artifact_root=args.artifact_root,
        requantize=args.requantize,
        recompile=args.recompile,
        generation=args.generation,
    )
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print(format_report(results))
    return results


if __name__ == "__main__":
    main()

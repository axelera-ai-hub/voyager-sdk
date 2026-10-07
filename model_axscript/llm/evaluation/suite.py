# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Run a set of metrics against a registered implementation and report results."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import warnings

from ..generate import GenerationConfig
from .metrics import (
    DeviceDecodeConsistencyMetric,
    EvalContext,
    Metric,
    OnDevicePerplexityMetric,
    PerformanceMetric,
    PerplexityMetric,
)

logger = logging.getLogger(__name__)

# Every metric the CLI can run, keyed by name. device_perplexity is prefill-
# accelerated when a prefill graph is present.
_METRICS: dict[str, Metric] = {
    m.name: m
    for m in (
        PerplexityMetric(),
        PerformanceMetric(),
        OnDevicePerplexityMetric(),
        DeviceDecodeConsistencyMetric(),
    )
}
_DEFAULT_NAMES = ("perplexity", "performance", "device_perplexity")
DEFAULT_METRICS: tuple[Metric, ...] = tuple(_METRICS[n] for n in _DEFAULT_NAMES)


@dataclass
class EvalReport:
    implementation: str
    context_len: int
    metrics: dict[str, dict]  # metric name -> results, or {"error": ...} if it failed

    def __str__(self) -> str:
        lines = [f"evaluation: {self.implementation} (context_len={self.context_len})"]
        for name, res in self.metrics.items():
            if "error" in res:
                lines.append(f"  {name}: unavailable ({res['error']})")
            else:
                lines.append(f"  {name}: " + ", ".join(f"{k}={v:.4g}" for k, v in res.items()))
        return "\n".join(lines)


def _evaluate_metric(metric, ctx, results: dict) -> None:
    """Run one metric, logging + capturing its result/error into ``results`` (so one metric
    failing -- e.g. no device for performance -- never aborts the rest)."""
    logger.info("running metric: %s", metric.name)
    try:
        res = metric.evaluate(ctx)
        logger.info("  %s: %s", metric.name, ", ".join(f"{k}={v:.4g}" for k, v in res.items()))
        results[metric.name] = res
    except Exception as exc:  # noqa: BLE001 - report it, don't abort the suite
        logger.warning("  %s: unavailable (%s)", metric.name, exc)
        results[metric.name] = {"error": f"{type(exc).__name__}: {exc}"}


def _run_scheduled(ctx, config, metrics) -> dict:
    """parallel: run the resource graph and the metrics as one job graph, so a metric overlaps
    the build of resources it doesn't need (perplexity scores while the device artifact
    compiles). Device metrics (requires="runtime") are chained so the single Metis device is
    used by one at a time."""
    from .scheduler import Job, run_jobs

    needs_runtime = any(getattr(m, "requires", "runtime") == "runtime" for m in metrics)
    jobs = ctx.resource_jobs(build_runtime=needs_runtime)
    results: dict[str, dict] = {}
    prev_device: str | None = None
    for metric in metrics:
        req = getattr(metric, "requires", "runtime")
        deps = [req]  # wait on the "bundle" or "runtime" resource job
        if req == "runtime" and prev_device is not None:
            deps.append(prev_device)  # serialize device metrics onto the one device
        name = f"metric:{metric.name}"
        jobs.append(Job(name, tuple(deps), lambda _i, m=metric: _evaluate_metric(m, ctx, results)))
        if req == "runtime":
            prev_device = name
    logger.info(
        "parallel eval: %d metric(s) + %d codegen unit(s) scheduled (metrics overlap the build)",
        len(metrics),
        sum(1 for j in jobs if j.in_process),
    )
    run_jobs(jobs)
    return results


def run_evaluation(
    config: GenerationConfig, metrics=DEFAULT_METRICS, **context_kwargs
) -> EvalReport:
    """Evaluate config.implementation with each metric.

    With --parallel, the resources and the metrics run as one dependency graph so a metric
    overlaps the build of resources it doesn't need (e.g. perplexity scores while the device
    artifact compiles). Per-metric failures are captured in the report so the remaining
    metrics still run.
    """
    ctx = EvalContext(config, **context_kwargs)
    if getattr(config, "parallel", False):
        # No serial fallback: a parallel-build failure should surface (it usually signals a real
        # parallel-specific bug), not be masked by a silent, slower sequential re-run.
        return EvalReport(
            config.implementation, config.context_len, _run_scheduled(ctx, config, metrics)
        )
    results: dict[str, dict] = {}
    for metric in metrics:
        _evaluate_metric(metric, ctx, results)
    return EvalReport(config.implementation, config.context_len, results)


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m ...llm.evaluation",
        description="Evaluate a registered LLM implementation (perplexity + performance).",
    )
    p.add_argument(
        "--implementation", default="qwen3-metis", help="registered implementation name"
    )
    p.add_argument("--repo-id", default="Qwen/Qwen3-0.6B", help="HuggingFace checkpoint id")
    p.add_argument("--context-len", type=int, default=2048, help="static context length")
    p.add_argument("--max-new-tokens", type=int, default=64, help="tokens to generate when timing")
    p.add_argument("--target", choices=("device", "sim"), default="device")
    p.add_argument(
        "--device",
        default="cpu",
        help="torch device for the reference model — calibration/quantization and the "
        "fake-quant perplexity forwards (e.g. cpu, cuda, cuda:0). Independent of "
        "--target (the Metis/sim backend the compiled artifact runs on).",
    )
    p.add_argument(
        "--phases",
        nargs="+",
        default=["prefill", "decode"],
        help="device-kernel phases to build; pass 'decode' to skip the prefill kernel",
    )
    p.add_argument(
        "--chunk-len",
        type=int,
        default=None,
        help="chunked prefill: compile the prefill graph for this many query positions and ingest "
        "the prompt in chunk_len-token chunks (threading recurrent/conv/KV state), bounding the "
        "prefill program/compile regardless of context length; default = single full-context prefill",
    )
    p.add_argument(
        "--device-lm-head",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="run the lm_head on device as an int8 vocab projection instead of the host fp16 "
        "gemv; quantizes the head into the bundle (recalibrates) and adds one decode-width "
        "graph, at the cost of vocab*hidden bytes of device DDR. Default: the host gemv, as "
        "the shipped cards and their goldens",
    )
    p.add_argument(
        "--num-cores",
        type=int,
        default=1,
        help="AIPU cores to compile for; >1 distributes the SDPA attention heads across "
        "cores (and sets AltoConfig.num_cores). 1 (default) = single core.",
    )
    p.add_argument("--artifact-dir", default="build/qwen3-0.6b")
    p.add_argument(
        "--calib-budget",
        type=int,
        default=None,
        help="total calibration samples (forwards), split across the source mix by "
        "weight; default uses the full production mix (~4232)",
    )
    # The quantized bundle and the compiled artifact are cached under --artifact-dir and
    # auto-reused when their inputs are unchanged (bundle key = repo/calib/smoothing/device_lm_head;
    # artifact key = bundle + implementation/context_len/chunk_len/phases/target).
    p.add_argument(
        "--parallel",
        action="store_true",
        help="run the eval as a dependency graph: overlap device codegen with calibration, "
        "compile the per-kind/phase kernels in parallel, and score metrics that don't need the "
        "device artifact (perplexity) concurrently with the build; off by default",
    )
    p.add_argument(
        "--rebuild",
        choices=("quant", "artifact", "all"),
        default=None,
        help="force a fresh build despite a valid cache: 'artifact' recompiles, 'quant' "
        "re-calibrates (and recompiles), 'all' = quant. Default: reuse whatever's still valid.",
    )
    p.add_argument(
        "--reuse",
        action="store_true",
        help="DEPRECATED / no-op: reuse is automatic + keyed. Use --rebuild to force a rebuild.",
    )
    p.add_argument(
        "--reuse-quant",
        action="store_true",
        help="DEPRECATED / no-op: the bundle is always cached + auto-reused. Use --rebuild quant.",
    )
    p.add_argument(
        "--metrics",
        nargs="+",
        choices=sorted(_METRICS),
        default=list(_DEFAULT_NAMES),
        help="which metrics to run",
    )
    p.add_argument("--prompt", default=None, help="prompt (seed text) for the performance run")
    p.add_argument(
        "--prompt-tokens",
        type=int,
        default=None,
        help="benchmark a synthesized prompt of this many tokens (TTFT vs prompt length)",
    )
    p.add_argument(
        "--eval-max-tokens", type=int, default=None, help="cap on the perplexity corpus tokens"
    )
    p.add_argument("--json", action="store_true", help="emit JSON instead of a text report")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="status logging verbosity",
    )
    return p


def _config_from_args(args: argparse.Namespace) -> GenerationConfig:
    return GenerationConfig(
        implementation=args.implementation,
        repo_id=args.repo_id,
        target=args.target,
        device=args.device,
        context_len=args.context_len,
        max_new_tokens=args.max_new_tokens,
        phases=tuple(args.phases),
        chunk_len=args.chunk_len,
        device_lm_head=args.device_lm_head,
        num_cores=args.num_cores,
        parallel=args.parallel,
        artifact_dir=Path(args.artifact_dir),
        calib_budget=args.calib_budget,
    )


def _quiet_huggingface() -> None:
    """Silence the HuggingFace / tokenizers startup chatter that's irrelevant to eval.

    Most of it is transformers' own logger (pad-token notices, legacy-tokenizer
    warnings, attention-mask advisories); the rest are UserWarning/FutureWarning the
    libraries emit at load. The env knobs must precede the first tokenizers/transformers
    import -- our transformers imports are lazy (inside the metrics), so setting them at
    the top of main() runs first.
    """
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    for mod in (r"transformers(\..*)?", r"huggingface_hub(\..*)?", r"tokenizers(\..*)?"):
        warnings.filterwarnings("ignore", category=UserWarning, module=mod)
        warnings.filterwarnings("ignore", category=FutureWarning, module=mod)

    try:  # belt-and-suspenders if transformers is already imported
        from transformers.utils import logging as hf_logging

        hf_logging.set_verbosity_error()
    except Exception:
        pass


def main(argv=None) -> EvalReport:
    args = _parser().parse_args(argv)
    _quiet_huggingface()
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger().setLevel(args.log_level)  # in case logging was already configured
    if args.device.startswith("cuda"):
        import torch

        if not torch.cuda.is_available():
            raise SystemExit("--device cuda requested but torch reports no CUDA device available")
    logger.info(
        "evaluating %s (repo=%s, context_len=%d, target=%s, device=%s, phases=%s, metrics=%s)",
        args.implementation,
        args.repo_id,
        args.context_len,
        args.target,
        args.device,
        args.phases,
        args.metrics,
    )
    context_kwargs = {}
    if args.rebuild:
        # Rebuilding the quant bundle invalidates the artifact downstream, so it implies artifact.
        context_kwargs["rebuild"] = (
            frozenset({"quant", "artifact"})
            if args.rebuild in ("quant", "all")
            else frozenset({"artifact"})
        )
    if args.reuse or args.reuse_quant:
        logger.warning(
            "--reuse / --reuse-quant are deprecated no-ops: artifacts are cached and auto-reused "
            "by key. Use --rebuild {quant,artifact,all} to force a rebuild."
        )
    if args.prompt is not None:
        context_kwargs["prompt"] = args.prompt
    if args.prompt_tokens is not None:
        context_kwargs["prompt_tokens"] = args.prompt_tokens
    if args.eval_max_tokens is not None:
        context_kwargs["eval_max_tokens"] = args.eval_max_tokens
    report = run_evaluation(
        _config_from_args(args), metrics=[_METRICS[n] for n in args.metrics], **context_kwargs
    )
    if args.json:
        print(
            json.dumps(
                {
                    "implementation": report.implementation,
                    "context_len": report.context_len,
                    "metrics": report.metrics,
                },
                indent=2,
            )
        )
    else:
        print(report)
    return report


if __name__ == "__main__":
    main()

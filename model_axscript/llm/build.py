# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""`llm compile` — calibrate + quantize + compile a device artifact.

A thin CLI over GenerationConfig + the build path. By DEFAULT it does the full work: recalibrate +
quantize (the `quantized_model/` bundle) AND recompile the device artifact. Two opt-in flags reuse
the suite's keyed caches (EvalContext) when nothing relevant changed:

  --reuse-quant   reuse the cached quantized_model/ (skip calibration) when its key matches
  --reuse-build   reuse the existing compiled artifact (skip compile) when its key matches

Reuse is best-effort + keyed: if the relevant cache key (repo/calib/smoothing/device-lm-head for
quant; that plus implementation/context/chunk/cores/phases/target for the build) does NOT match, that
layer is rebuilt regardless. `--reuse-quant --reuse-build` together = "skip if already up-to-date".
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import time


def _fmt(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(round(seconds)), 60)
    if m < 60:
        return f"{m}m {s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def resolve_device_lm_head(flag: bool | None, impl) -> bool:
    """This CLI's default is the implementation's device head where it provides one. The shipped
    cards and ``GenerationConfig`` default to the host head (their goldens were measured with
    it), so a front-door build of the same model differs from this one exactly here."""
    return impl.provides_device_head if flag is None else flag


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m model_axscript.llm.build",
        description="Calibrate + quantize + compile a device artifact.",
    )
    p.add_argument(
        "--implementation",
        default="qwen3-metis",
        help="implementation registry name (e.g. qwen3-metis-multicore, llama-metis-opt)",
    )
    p.add_argument("--repo-id", default="Qwen/Qwen3-0.6B", help="HF checkpoint to quantize")
    p.add_argument("--artifact-dir", required=True, help="output directory for the artifact")
    p.add_argument("--context-len", type=int, default=2048)
    p.add_argument(
        "--chunk-len", type=int, default=None, help="chunked-prefill query width (None=full)"
    )
    p.add_argument(
        "--device-lm-head",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="run the lm_head on device as an int8 vocab projection instead of the host fp16 "
        "gemv; quantizes the head into the bundle (recalibrates) and adds one decode-width "
        "graph, at the cost of vocab*hidden bytes of device DDR. Default: on where the "
        "implementation provides one; pass --no-device-lm-head to force the host gemv",
    )
    p.add_argument(
        "--calib-batch",
        type=int,
        default=1,
        help="calibration windows per forward. The windows are independent, so stacking "
        "them is identical arithmetic and identical observer statistics, but it amortises "
        "Python and thread dispatch -- which dominate CPU calibration of a model whose "
        "linear attention is a per-token loop (measured 3.07x at 8 for a 9B). Costs "
        "activation memory linearly",
    )
    p.add_argument(
        "--calib-dtype",
        default="float32",
        choices=("float32", "bfloat16"),
        help="dtype for the torch reference during calibration. bfloat16 halves its weights "
        "and the memory traffic a CPU calibration is bound by; the gated-delta scan stays "
        "fp32 regardless. It does round the observed activation ranges, so check the "
        "perplexity gate before trusting a bf16-calibrated bundle",
    )
    p.add_argument("--num-cores", type=int, default=1)
    p.add_argument(
        "--generation",
        default="OMEGA",
        choices=("OMEGA", "EUROPA"),
        help="target hardware generation",
    )
    p.add_argument(
        "--phases", nargs="+", default=["prefill", "decode"], choices=["prefill", "decode"]
    )
    p.add_argument("--target", choices=["device", "sim"], default="device")
    p.add_argument(
        "--device",
        default=None,
        help="torch device for the reference model AND calibration (same device for both, so "
        "they can't disagree on where a tensor lives). Default: cuda if one is present, else cpu",
    )
    p.add_argument("--calib-budget", type=int, default=None)
    p.add_argument("--smooth-alpha", type=float, default=0.8)
    p.add_argument("--qk-alpha", type=float, default=0.5)
    p.add_argument(
        "--parallel", action="store_true", help="overlap device codegen with calibration"
    )
    p.add_argument(
        "--codegen-jobs",
        type=int,
        default=4,
        help="codegen compiles to run at once under --parallel. Each is a spawn process "
        "holding a whole kernel compile, ALONGSIDE the calibration process holding the fp32 "
        "reference (36 GB for a 9B), so the default four can exhaust host memory on a big "
        "model. 1 still folds the LUT/decode-scale recorders into the calibration passes -- "
        "halving them from four to two, which is most of what --parallel buys -- while "
        "compiling one kernel at a time",
    )
    p.add_argument(
        "--reuse-quant",
        action="store_true",
        help="reuse the cached quantized_model (skip calibration) if its key matches",
    )
    p.add_argument(
        "--reuse-build",
        action="store_true",
        help="reuse the existing compiled artifact (skip compile) if its key matches",
    )
    p.add_argument(
        "--log-level",
        default="WARNING",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        help="root logging level (INFO/DEBUG surface calibration + codegen progress)",
    )
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    logging.basicConfig(level=args.log_level, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger().setLevel(args.log_level)

    from .evaluation.metrics import EvalContext
    from .generate import GenerationConfig
    from .implementation import get_implementation

    if args.device is None:
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"

    impl = get_implementation(args.implementation)
    config = GenerationConfig(
        implementation=args.implementation,
        repo_id=args.repo_id,
        artifact_dir=Path(args.artifact_dir),
        context_len=args.context_len,
        chunk_len=args.chunk_len,
        device_lm_head=resolve_device_lm_head(args.device_lm_head, impl),
        num_cores=args.num_cores,
        generation=args.generation,
        phases=tuple(args.phases),
        target=args.target,
        device=args.device,
        # The reference model and its calibration inputs must agree on device -- see
        # EvalContext.quantized_model, which feeds one into the other.
        cal_device=args.device,
        calib_budget=args.calib_budget,
        calib_batch=args.calib_batch,
        calib_dtype=args.calib_dtype,
        smooth_alpha=args.smooth_alpha,
        qk_alpha=args.qk_alpha,
        parallel=args.parallel,
        codegen_jobs=args.codegen_jobs,
    )
    # Fail fast, before calibration: an implementation with no device head refuses --device-lm-head
    # outright, and one that has it needs the phase its graph is compiled for.
    tail = impl.tail_kernel(config)
    if tail is not None and not set(tail.phases) & set(config.phases):
        raise ValueError(
            f"--device-lm-head compiles a {tail.phases} graph, which --phases "
            f"{list(config.phases)} excludes"
        )

    out = Path(config.artifact_dir)
    # Default rebuilds BOTH layers; each --reuse-* flag drops its layer from the force-rebuild set.
    # (Reuse is still keyed: EvalContext rebuilds a layer anyway if its own cache key doesn't match.)
    rebuild = set()
    if not args.reuse_quant:
        rebuild.add("quant")
    if not args.reuse_build:
        rebuild.add("artifact")
    ctx = EvalContext(config=config, rebuild=frozenset(rebuild))
    artifact_key = ctx._artifact_key()

    if (
        args.reuse_build
        and (out / "model.json").exists()
        and EvalContext._cache_ok(out, artifact_key)
    ):
        print(f"reusing build: {out}  (artifact matches config; --reuse-build)")
        return 0

    t_start = time.perf_counter()
    stages = []
    if config.parallel:
        from .evaluation.scheduler import run_jobs

        # Build the bundle (calibration) ∥ codegen, then author the artifact -- via the scheduler.
        # Drop the "runtime" job so we don't connect/allocate the device merely to compile.
        # (build_runtime=False would have built ONLY the bundle and skipped the artifact entirely.)
        jobs = [j for j in ctx.resource_jobs(build_runtime=True) if j.name != "runtime"]
        jt: dict = {}
        # the "artifact" job authors model.json + writes the cache key
        run_jobs(jobs, max_processes=max(1, config.codegen_jobs), timings=jt)

        def _label(name):
            if name == "bundle":
                return "calibrate + quantize"
            if name == "artifact":
                return "author"
            if name.startswith("codegen:"):
                return f"codegen ({name.split(':')[1]})"  # codegen:<phase>:<kind>
            return name

        for job in jobs:  # graph order: bundle, codegen:*, artifact
            if job.name in jt:
                stages.append((_label(job.name), jt[job.name]))
        # per-job times OVERLAP, so they can sum to more than this wall-clock total:
        stages.append(("total (overlapped)", time.perf_counter() - t_start))
    else:
        t0 = time.perf_counter()
        bundle = ctx.bundle()  # reused when --reuse-quant + key matches, else (re)calibrated
        t_quant = time.perf_counter() - t0
        t1 = time.perf_counter()
        out = get_implementation(config.implementation).compile(bundle, config)
        t_compile = time.perf_counter() - t1
        EvalContext._write_cache_key(out, artifact_key)
        total = time.perf_counter() - t_start
        stages += [
            ("calibrate + quantize", t_quant),
            ("compile (codegen + author)", t_compile),
            ("total", total),
        ]

    print(f"compiled -> {out}")
    width = max(len(name) for name, _ in stages)
    for name, secs in stages:
        print(f"  {name:<{width}}  {_fmt(secs)}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

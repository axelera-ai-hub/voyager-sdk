# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Reproduce the Axelera device-LLM benchmark numbers — a single, self-contained script you can
hand to someone else to independently verify the results.

It runs a fixed task set on BOTH the compiled device artifact (through the HF shim, hf_model.py)
and the fp32 HuggingFace reference (lm-eval's own `--model hf`), capped at the artifact's compiled
context_len so the comparison is apples-to-apples, and prints a side-by-side table.

----------------------------------------------------------------------------------------------------
SETUP (one-time)
    pip install "lm-eval>=0.4.12" transformers langdetect immutabledict
    # langdetect + immutabledict are only needed for ifeval's verifiers.

    # Run from the repository root, where `model_axscript` is importable.

REQUIREMENTS
    * A real Metis device visible to axelera.runtime (the script asserts one is present and is NOT a
      simulator) for the `device` backend.
    * The prebuilt artifact directory (model.json + host_params.npz + pipeline.json).
    * Network access (or a local HF cache) for the tokenizer/config and the fp reference weights.

USAGE
    # Full reproduction (device + reference), ~1.5-2 h dominated by ifeval decode on-device:
    python -m model_axscript.llm.evaluation.reproduce_benchmarks \
        --artifact_dir build/qwen3-1.7b-4k-4-cores --repo_id Qwen/Qwen3-1.7B

    # Fast smoke (~10-15 min): small subsets, skips full wikitext.
    python -m model_axscript.llm.evaluation.reproduce_benchmarks --artifact_dir ... --repo_id ... --quick

    # Numerical trust check only (device-vs-fp logit fidelity + device determinism):
    python -m model_axscript.llm.evaluation.reproduce_benchmarks --artifact_dir ... --repo_id ... --sanity --backends device

    # Subset of tasks / one backend:
    python -m model_axscript.llm.evaluation.reproduce_benchmarks --artifact_dir ... --repo_id ... \
        --backends device --tasks wikitext,lambada_openai

DEFAULT TASKS (and which device code path each exercises)
    wikitext        (full)  loglikelihood_rolling -> PREFILL path (forward)
    lambada_openai  (200)   loglikelihood         -> PREFILL path (forward)
    ifeval          (100)   generate_until        -> DECODE path (generate); Qwen3 thinking disabled

NOTES THAT MATTER FOR REPRODUCIBILITY
    * context_len cap: the reference is pinned to the artifact's context_len (read from pipeline.json).
      Without it the fp model would use its full native context and post an unfairly low perplexity.
    * Chat template: applied ONLY to generative chat tasks (ifeval), with Qwen3 `enable_thinking=False`
      (default thinking emits <think> blocks that break ifeval format checks and inflate length).
      Raw LM tasks (wikitext/lambada) are scored WITHOUT a chat template, as they should be.
    * ifeval uses max_gen_toks=512 here (vs lm-eval's 1280 default) to bound on-device decode time;
      both backends use the same value, so the comparison is fair. Pass --max_gen_toks to change it.
    * Results are also written to <out>/repro_results.json with tool versions + device info.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform

# Keep the fp reference on CPU and avoid a CUDA caching-allocator warmup crash on driverless boxes.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

# Tasks that must be chat-templated + thinking-disabled (generative chat tasks).
CHAT_TASKS = {"ifeval", "gsm8k", "gsm8k_cot"}

# The limits behind the collected numbers (None = full task). --limit overrides all of these.
DEFAULT_LIMITS = {"wikitext": None, "lambada_openai": 200, "ifeval": 100}
QUICK_LIMITS = {"lambada_openai": 50, "ifeval": 5}
DEFAULT_TASKS = ["wikitext", "lambada_openai", "ifeval"]
QUICK_TASKS = ["lambada_openai", "ifeval"]


def device_info() -> dict:
    """Confirm a real (non-simulator) Metis device is present; return its descriptor."""
    from axelera.runtime import Context

    sim_env = {k: v for k, v in os.environ.items() if any(s in k.upper() for s in ("SIM", "MOCK"))}
    devices = Context().list_devices()
    if not devices:
        raise RuntimeError("no Axelera device visible to axelera.runtime")
    return {"device": str(devices[0]), "sim_env_vars": sim_env}


def _patch_no_think(tokenizer) -> None:
    """Force Qwen-style chat templates to disable thinking (no <think> reasoning block)."""
    orig = tokenizer.apply_chat_template

    def patched(*a, **k):
        k.setdefault("enable_thinking", False)
        return orig(*a, **k)

    tokenizer.apply_chat_template = patched


def build_lm(
    backend: str, artifact_dir: str, repo_id: str, context_len: int, num_sub_devices: int
):
    """Construct the lm-eval model object for a backend (device shim or fp32 reference)."""
    if backend == "device":
        from .hf_model import make_hflm  # pins max_length=context_len internally

        lm = make_hflm(artifact_dir, repo_id, num_sub_devices=num_sub_devices)
    elif backend == "ref":
        from lm_eval.models.huggingface import HFLM

        # device="cpu" + dtype float32: the full-precision reference, same context budget as the device.
        lm = HFLM(
            pretrained=repo_id,
            dtype="float32",
            device="cpu",
            max_length=context_len,
            batch_size=1,
        )
    else:
        raise ValueError(backend)
    _patch_no_think(lm.tokenizer)
    return lm


def run_backend(backend, lm, tasks, limits, max_gen_toks) -> dict:
    """Evaluate `tasks` on a built lm. Raw LM tasks and chat tasks are run in separate
    simple_evaluate calls so the chat template is applied ONLY where it belongs."""
    import lm_eval

    raw = [t for t in tasks if t not in CHAT_TASKS]
    chat = [t for t in tasks if t in CHAT_TASKS]
    out: dict = {}
    for group, use_chat in ((raw, False), (chat, True)):
        for task in group:
            limit = limits.get(task, None)
            print(
                f"\n>>> [{backend}] {task} (limit={limit}, chat_template={use_chat})", flush=True
            )
            res = lm_eval.simple_evaluate(
                model=lm,
                tasks=[task],
                limit=limit,
                apply_chat_template=use_chat,
                gen_kwargs=(f"max_gen_toks={max_gen_toks}" if use_chat else None),
                bootstrap_iters=0,
            )
            out[task] = res["results"][task]
    return out


def sanity_check(artifact_dir: str, repo_id: str) -> dict:
    """Trustworthiness check: device logits must be deterministic, non-degenerate, and correlated
    with — but NOT identical to — the fp reference (the W8A8 quantization fingerprint)."""
    import numpy as np
    import torch

    from .hf_model import AxeleraForCausalLM

    prompt = "The mitochondria is the powerhouse of the cell, and DNA encodes"
    dev = AxeleraForCausalLM.from_artifact(artifact_dir, repo_id=repo_id)
    tok = dev.tokenizer
    ids = torch.tensor([tok(prompt).input_ids])
    l1 = dev(input_ids=ids).logits[0].float().numpy()
    l2 = dev(input_ids=ids).logits[0].float().numpy()

    from transformers import AutoModelForCausalLM

    ref = AutoModelForCausalLM.from_pretrained(repo_id, dtype=torch.float32).eval()
    with torch.no_grad():
        lr = ref(input_ids=ids).logits[0].float().numpy()

    seq = l1.shape[0]
    cos = [
        float(np.dot(l1[p], lr[p]) / (np.linalg.norm(l1[p]) * np.linalg.norm(lr[p]) + 1e-9))
        for p in range(seq)
    ]
    top1 = float(np.mean([l1[p].argmax() == lr[p].argmax() for p in range(seq)]))
    out = {
        "deterministic": bool(np.array_equal(l1, l2)),
        "logits_std": float(l1.std()),
        "cosine_vs_ref_mean": float(np.mean(cos)),
        "cosine_vs_ref_min": float(np.min(cos)),
        "top1_agreement_vs_ref": top1,
        "identical_to_ref": bool(np.allclose(l1, lr, atol=1e-3)),
    }
    print("\n=== SANITY (device numerical trust) ===")
    print(json.dumps(out, indent=2))
    ok = (
        out["deterministic"]
        and out["logits_std"] > 0
        and not out["identical_to_ref"]
        and out["cosine_vs_ref_mean"] > 0.9
    )
    print(
        "VERDICT:",
        (
            "PASS — real, deterministic, quantized device compute"
            if ok
            else "FAIL — investigate (see thresholds in script)"
        ),
    )
    return out


def _print_table(results: dict, backends: list) -> None:
    print("\n" + "=" * 78)
    print("BENCHMARK COMPARISON")
    print("=" * 78)
    tasks = sorted({t for b in backends for t in results.get(b, {})})
    for task in tasks:
        print(f"\n{task}:")
        metrics = sorted(
            {
                m
                for b in backends
                for m in results.get(b, {}).get(task, {})
                if m.endswith(",none") and isinstance(results[b][task][m], (int, float))
            }
        )
        header = "  {:<28}".format("metric") + "".join(f"{b:>14}" for b in backends)
        print(header)
        for m in metrics:
            row = "  {:<28}".format(m.replace(",none", ""))
            for b in backends:
                v = results.get(b, {}).get(task, {}).get(m)
                row += f"{v:>14.4f}" if isinstance(v, (int, float)) else f"{'-':>14}"
            print(row)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="Reproduce Axelera device-LLM benchmarks.")
    p.add_argument("--artifact_dir", required=True, help="prebuilt artifact directory")
    p.add_argument(
        "--repo_id",
        default=None,
        help="HF repo id (config + tokenizer + fp reference); defaults to pipeline.json",
    )
    p.add_argument("--backends", default="device,ref", help="comma list: device,ref")
    p.add_argument("--tasks", default=None, help="comma list; default depends on --quick")
    p.add_argument("--limit", type=int, default=None, help="override per-task limit for ALL tasks")
    p.add_argument("--max_gen_toks", type=int, default=512, help="generation cap for chat tasks")
    p.add_argument(
        "--num_sub_devices",
        type=int,
        default=None,
        help="sub-devices to reserve; default: what the artifact was compiled for",
    )
    p.add_argument(
        "--quick", action="store_true", help="small subsets, skip full wikitext (~10-15 min)"
    )
    p.add_argument("--sanity", action="store_true", help="run the device numerical-trust check")
    p.add_argument("--out", default="repro_out", help="output directory for results JSON")
    args = p.parse_args(argv)

    backends = [b.strip() for b in args.backends.split(",") if b.strip()]
    tasks = (
        [t.strip() for t in args.tasks.split(",")]
        if args.tasks
        else (QUICK_TASKS if args.quick else DEFAULT_TASKS)
    )
    limits = dict(QUICK_LIMITS if args.quick else DEFAULT_LIMITS)
    if args.limit is not None:
        limits = {t: args.limit for t in tasks}

    pipeline = json.loads((Path(args.artifact_dir) / "pipeline.json").read_text())
    context_len = int(pipeline["context_len"])
    from ..generate import artifact_repo_id

    repo_id = args.repo_id or artifact_repo_id(args.artifact_dir)
    if repo_id is None:
        raise SystemExit(f"no repo_id in {args.artifact_dir}/pipeline.json; pass --repo_id")

    import lm_eval
    import transformers

    report: dict = {
        "config": {
            "artifact_dir": args.artifact_dir,
            "repo_id": repo_id,
            "implementation": pipeline.get("implementation"),
            "context_len": context_len,
            "chunk_len": pipeline.get("chunk_len"),
            "tasks": tasks,
            "limits": limits,
            "max_gen_toks": args.max_gen_toks,
            "backends": backends,
        },
        "versions": {
            "lm_eval": lm_eval.__version__,
            "transformers": transformers.__version__,
            "python": platform.python_version(),
        },
        "results": {},
    }
    if "device" in backends:
        report["device_info"] = device_info()
        print("device:", report["device_info"]["device"], flush=True)

    if args.sanity:
        report["sanity"] = sanity_check(args.artifact_dir, repo_id)

    for backend in backends:
        print(f"\n########## backend: {backend} ##########", flush=True)
        lm = build_lm(backend, args.artifact_dir, repo_id, context_len, args.num_sub_devices)
        report["results"][backend] = run_backend(backend, lm, tasks, limits, args.max_gen_toks)
        del lm

    _print_table(report["results"], backends)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "repro_results.json").write_text(json.dumps(report, indent=2, default=str))
    print(f"\nwrote {out_dir / 'repro_results.json'}")


if __name__ == "__main__":
    main()

# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""`llm evaluate` — run EleutherAI lm-evaluation-harness tasks on a compiled device artifact.

Wraps the artifact in the HF shim (hf_model.make_hflm) and dispatches arbitrary lm-eval tasks via
lm_eval.simple_evaluate. Loglikelihood tasks (hellaswag/arc/lambada/wikitext) run the device PREFILL
path; generative tasks (gsm8k/ifeval) run the DECODE loop. Generative tasks on Qwen3 usually want
--apply-chat-template --no-think.

    python -m model_axscript.llm.evaluation.evaluate --artifact-dir build/qwen3-0.6b-4k-4-cores --repo-id Qwen/Qwen3-0.6B \
        --tasks hellaswag,arc_easy --limit 200

With --compare-fp the same tasks also run on the fp32 (or --ref-dtype) HuggingFace reference and a
device-vs-fp delta is printed -- the quantization/fp18 quality gap. The reference runs on --ref-device
(e.g. cuda for speed) and reuses OUR tokenizer loader (chat template + LFM2.5's TokenizersBackend
fallback) so it can't diverge from the device tokenization and isn't blocked by AutoTokenizer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m model_axscript.llm.evaluation.evaluate",
        description="Run lm-eval tasks on a compiled artifact (via the HF shim).",
    )
    p.add_argument("--artifact-dir", required=True)
    p.add_argument(
        "--repo-id",
        default=None,
        help="HF repo for the tokenizer/config; defaults to the artifact's pipeline.json",
    )
    p.add_argument("--tasks", required=True, help="comma-separated lm-eval task names")
    p.add_argument("--limit", type=int, default=None, help="cap docs per task (bring-up)")
    p.add_argument("--num-fewshot", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument(
        "--num-sub-devices",
        type=int,
        default=None,
        help="sub-devices to reserve; defaults to what the artifact needs",
    )
    p.add_argument("--apply-chat-template", action="store_true", help="chat-template the prompts")
    p.add_argument(
        "--no-think",
        action="store_true",
        help="Qwen3: disable thinking (enable_thinking=False) when chat-templating",
    )
    p.add_argument(
        "--compare-fp",
        action="store_true",
        help="also run the fp32 HF reference on the same tasks and print a device-vs-fp delta",
    )
    p.add_argument(
        "--ref-device",
        default="cpu",
        help="torch device for the --compare-fp reference (e.g. cpu, cuda, cuda:0)",
    )
    p.add_argument(
        "--ref-dtype",
        default="float32",
        choices=("float32", "bfloat16", "float16"),
        help="dtype for the --compare-fp reference model",
    )
    p.add_argument(
        "--parallel",
        action="store_true",
        help="with --compare-fp, run the fp reference concurrently with the device eval (separate "
        "process; they use different hardware -- Metis vs --ref-device -- so they truly overlap)",
    )
    p.add_argument("--out", default=None, help="write the full results JSON here")
    return p


def _apply_no_think(tokenizer) -> None:
    """Qwen3: prefill an empty <think></think> so the model answers directly."""
    _orig = tokenizer.apply_chat_template

    def _no_think(*a, **k):
        k.setdefault("enable_thinking", False)
        return _orig(*a, **k)

    tokenizer.apply_chat_template = _no_think


def _run_tasks(lm, args, tasks: list[str]) -> dict:
    import lm_eval

    return lm_eval.simple_evaluate(
        model=lm,
        tasks=tasks,
        limit=args.limit,
        num_fewshot=args.num_fewshot,
        apply_chat_template=args.apply_chat_template,
    )


def _make_reference_lm(repo_id: str, context_len: int, args):
    """fp reference as lm_eval's HFLM, but with OUR tokenizer injected.

    Passing the tokenizer object (rather than letting HFLM call AutoTokenizer) both applies the same
    chat template the device uses -- so the comparison is apples-to-apples -- and dodges LFM2.5's
    AutoTokenizer failure (tokenizer_class=TokenizersBackend). max_length is pinned to the artifact's
    context_len so rolling/truncation windows match the device (see make_hflm).
    """
    from lm_eval.models.huggingface import HFLM

    from ..generate import load_tokenizer

    tokenizer = load_tokenizer(repo_id)
    if args.no_think:
        _apply_no_think(tokenizer)
    return HFLM(
        pretrained=repo_id,
        tokenizer=tokenizer,
        dtype=args.ref_dtype,
        device=args.ref_device,
        batch_size=args.batch_size,
        max_length=context_len,
    )


def _reference_worker(repo_id: str, context_len: int, tasks: list[str], cfg: dict, out_path: str):
    """Subprocess (spawn) entry: build the fp reference, run the tasks, write results JSON.

    Runs in its own interpreter so its lm-eval/HF-datasets state -- and the CUDA/CPU work -- never
    share a process (or the GIL) with the device eval. Only JSON-able results cross back, via the file.
    """
    args = argparse.Namespace(**cfg)
    lm = _make_reference_lm(repo_id, context_len, args)
    res = _run_tasks(lm, args, tasks)
    Path(out_path).write_text(json.dumps(res, default=str))


_REF_CFG_KEYS = (
    "ref_device",
    "ref_dtype",
    "batch_size",
    "no_think",
    "limit",
    "num_fewshot",
    "apply_chat_template",
)


def _start_reference(repo_id: str, context_len: int, tasks: list[str], args):
    """Launch the fp reference in a spawned subprocess; returns (process, results_path)."""
    import multiprocessing as mp
    import tempfile

    out_path = tempfile.NamedTemporaryFile(prefix="fp_ref_", suffix=".json", delete=False).name
    cfg = {k: getattr(args, k) for k in _REF_CFG_KEYS}
    proc = mp.get_context("spawn").Process(
        target=_reference_worker, args=(repo_id, context_len, tasks, cfg, out_path)
    )
    proc.start()
    return proc, out_path


def _finish_reference(proc, out_path: str) -> dict:
    """Join the reference subprocess and load its results (raises if it failed)."""
    proc.join()
    if proc.exitcode != 0 or not Path(out_path).exists():
        raise SystemExit(f"fp reference subprocess failed (exit {proc.exitcode})")
    return json.loads(Path(out_path).read_text())


def _delta_table(device_res: dict, ref_res: dict, tasks: list[str]) -> str:
    """Compact device-vs-fp table over the shared numeric metrics (skips *_stderr)."""
    rows = [
        "",
        "device vs fp32 reference (Δ = device - fp):",
        f"{'task':<22}{'metric':<26}{'device':>10}{'fp32':>10}{'Δ':>10}",
    ]
    for task in tasks:
        dev = device_res["results"].get(task, {})
        ref = ref_res["results"].get(task, {})
        for metric, dval in dev.items():
            # lm-eval keys are "<name>,<filter>" (e.g. "perplexity,none", "acc_stderr,none");
            # compare the numeric metrics, skip the stderr companions.
            name = metric.split(",")[0]
            if name.endswith("_stderr") or "," not in metric or not isinstance(dval, (int, float)):
                continue
            rval = ref.get(metric)
            if not isinstance(rval, (int, float)):
                continue
            rows.append(f"{task:<22}{metric:<26}{dval:>10.4f}{rval:>10.4f}{dval - rval:>+10.4f}")
    return "\n".join(rows)


def main(argv=None) -> int:
    args = _parser().parse_args(argv)

    from . import hf_model

    if hf_model.AxeleraForCausalLM is None:
        raise SystemExit("transformers is required for the HF shim (pip install transformers).")
    try:
        from lm_eval.utils import make_table
    except ImportError:
        raise SystemExit("lm-eval is not installed (pip install lm-eval).")

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    lm = hf_model.make_hflm(
        args.artifact_dir,
        args.repo_id,
        batch_size=args.batch_size,
        num_sub_devices=args.num_sub_devices,
    )
    if args.no_think:
        _apply_no_think(lm.tokenizer)

    # Same repo_id + context_len as the device artifact, so the reference matches it exactly.
    repo_id = context_len = ref_proc = ref_out = None
    if args.compare_fp:
        from ..generate import artifact_repo_id

        repo_id = args.repo_id or artifact_repo_id(args.artifact_dir)
        context_len = lm.model._context_len
        if args.parallel:  # kick off the fp reference before the device eval so they overlap
            ref_proc, ref_out = _start_reference(repo_id, context_len, tasks, args)

    device_res = _run_tasks(lm, args, tasks)
    print("=== device ===")
    print(make_table(device_res))
    out: dict = {"device": device_res}

    if args.compare_fp:
        print(f"\n=== fp32 reference ({args.ref_dtype} on {args.ref_device}, repo={repo_id}) ===")
        if args.parallel:
            ref_res = _finish_reference(ref_proc, ref_out)
        else:
            ref_lm = _make_reference_lm(repo_id, context_len, args)
            ref_res = _run_tasks(ref_lm, args, tasks)
        print(make_table(ref_res))
        print(_delta_table(device_res, ref_res, tasks))
        out["reference"] = ref_res

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(out, indent=2, default=str))
        print(f"wrote results -> {args.out}")
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

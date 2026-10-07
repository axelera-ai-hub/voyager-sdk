# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Fast regression check for a compiled device LLM artifact.

Target <=5 min. It exercises BOTH device code paths and reports continuous, low-variance fidelity
signals + performance, rather than noisy task accuracy.

THREE MODES (chosen by flags):
  * GOLDEN GATE -- `--compare-to PATH`: gate device-vs-golden (paired; device is deterministic so an
    unchanged build => exactly 0, a recompile that perturbs numerics shows up). Exits nonzero on breach.
  * FP REFERENCE (DEFAULT, when no --compare-to): score an fp32 (or bf16) HF reference and report
    device-vs-fp quality (ppl gap + per-token mean/p95/max |Δ|). The quantization gap itself is
    informational -- it is expected -- but the signals that need no reference (decode<->prefill,
    determinism) ARE gated here, and exit nonzero on breach.
  * PERF-ONLY -- `--perf-only`: report only the performance numbers; skip all fidelity/reference
    scoring. Nothing is checked, so its numbers are not attached to a correct result.

SIGNALS (device-side, computed in GOLDEN and FP modes):
  * PERPLEXITY over a fixed, pinned, out-of-calibration corpus (regression_corpus.txt) -- PREFILL.
  * per-token |Δlogprob| mean / p95 / max (max trips first on a localized failure) -- vs golden or fp.
  * DECODE<->PREFILL consistency over a short window -- the ONLY signal that exercises the DECODE
    path / KV cache, and gated in both fidelity modes. Perplexity drives prefill, so no
    decode-only change (the fused decode schedule, layer stacking) can move it.
    Both sides go through whatever lm_head the artifact carries, so on a device-head build the
    head gets its own line (device head vs the host fp16 gemv) rather than distorting this one.
  * DETERMINISM (two corpus forwards bit-identical).
  * PERFORMANCE: prefill TTFT + tok/s, and decode tok/s, at powers of 2 from 1024 to context_len
    (decode skipped at the top size -- no room left). In GOLDEN mode it is loosely gated vs golden.

RESULTS are plain JSON you can write/read ANYWHERE (not tied to the build dir): `--out PATH` writes this
run's full results (in FP mode only when the floors passed: a failing run is not a golden, see
`write_results`); `--compare-to PATH` reads a previously-saved results file (a "golden" is just a saved
run you trust; legacy flat files also work). Schema: {metadata, measured{ppl, corpus_token_logprobs, perf,
decode_consistency, deterministic}, comparison?, reference_fp?}.
Run (from the alto integration dir):
    pip install "lm-eval>=0.4.12" transformers
    # default: device vs fp32 reference (quality report) -- and save a baseline for later gating:
    python -m model_axscript.llm.evaluation.regression --artifact_dir build/qwen3-0.6b-4k-4-cores --repo_id Qwen/Qwen3-0.6B \
        --out goldens/qwen3-0.6b.json
    # per-commit gate against that golden (exits nonzero on breach) + archive this run:
    python -m model_axscript.llm.evaluation.regression --artifact_dir ... --repo_id ... \
        --compare-to goldens/qwen3-0.6b.json --out runs/$(date +%F-%H%M).json
    # just performance:
    python -m model_axscript.llm.evaluation.regression --artifact_dir ... --repo_id ... --perf-only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys
import time

HERE = Path(__file__).parent
CORPUS_PATH = HERE / "regression_corpus.txt"

# Golden tolerances. Numerical signals are tight (device is deterministic); perf is looser (timing noise).
DEFAULT_TOL = {
    "ppl_rel": 0.02,  # corpus perplexity relative increase
    "logprob_mean_abs": 0.03,  # mean |Δ logprob| vs golden (nats)
    "logprob_max_abs": 0.50,  # max  |Δ logprob| vs golden (nats)
    "decode_rel_l2": 0.10,  # prefill<->decode rel-L2 mean
    "decode_top1": 0.95,  # prefill<->decode top-1 agreement (floor)
    "head_rel_l2": 0.10,  # device lm_head vs host fp16 head, rel-L2 mean (device-head builds)
    "head_top1": 0.90,  # device lm_head vs host fp16 head, top-1 agreement (floor)
    # decode drift: an intermittent fault reads as exact first positions then divergence to
    # the end, so the onset is gated on its own (see self_consistency), not only the mean.
    "decode_drift_ratio": 100.0,  # last-over-first position-bin rel-L2 ratio that flags an onset
    "decode_drift_max": 0.01,  # ... and the rel-L2 max the onset has to reach to count
    "perf_slow": 0.25,  # allowed TTFT increase / tok-s decrease vs golden
}


def load_corpus() -> str:
    return CORPUS_PATH.read_text()


def _target_logprobs(logits, ids, start):
    """Per-token logprobs of ids[start:] under `logits` ([seq,V]); token i predicted by logits[i-1]."""
    import torch

    logp = torch.log_softmax(logits.float(), dim=-1)
    pos = torch.arange(start - 1, len(ids) - 1)
    return logp[pos, ids[start:]]


def score_corpus(logits_fn, tok, context_len: int) -> dict:
    """Perplexity + per-token logprobs over the fixed corpus (one window). The gate's prefill signal."""
    import numpy as np
    import torch

    ids = tok(load_corpus(), return_tensors="pt").input_ids[0][:context_len]
    with torch.no_grad():
        logits = logits_fn(ids.unsqueeze(0))[0]
    tok_lp = _target_logprobs(logits, ids, 1)
    return {
        "ppl": float(np.exp(-tok_lp.mean().item())),
        "corpus_token_logprobs": tok_lp.float().tolist(),
    }


# ----- performance ------------------------------------------------------------------------


def _perf_sizes(ctx: int) -> list[int]:
    sizes, L = [], 128
    while L <= ctx:
        sizes.append(L)
        L *= 2
    return sizes or [ctx]


def perf_numbers(shim, n_decode: int = 16) -> list[dict]:
    """Prefill TTFT + tok/s, and decode tok/s, at powers of 2 from 1024 to context_len.

    TTFT uses the real generation path (chunked prefill + single-position lm_head), NOT the
    teacher-forcing all-positions lm_head. Decode tok/s is n_decode steps from position L (skipped
    when L + n_decode > context_len). One warmup pass first (first device launch carries setup cost).
    """
    impl, rt, hp = shim._impl, shim._runtime, shim._host_params
    ctx, chunk = shim._context_len, shim._chunk_len
    tid = min(100, int(shim.config.vocab_size) - 1)  # any valid id; content doesn't affect timing
    # An artifact built with --phases prefill has no decode graph. That is a legitimate
    # configuration (perplexity is teacher-forced, so prefill alone answers the quality
    # question), so report the prefill numbers rather than dying on a KeyError in the
    # warmup -- which ignores n_decode.
    has_decode = "decode" in getattr(rt, "_spec", {}).get("graphs", {"decode": None})
    if not has_decode:
        n_decode = 0

    warm = [tid] * min(chunk or 1024, ctx)  # warmup: 1 chunk prefill + a few decode (discard)
    h = impl.prefill_hidden(rt, hp, warm, ctx, chunk)
    impl.compute_logits(h, hp, position=len(warm) - 1, runtime=rt)
    if has_decode:
        for p in range(len(warm), min(len(warm) + 3, ctx)):
            impl.decode_step(rt, hp, tid, p, ctx)

    out = []
    for L in _perf_sizes(ctx):
        prompt = [tid] * L
        t0 = time.perf_counter()
        h = impl.prefill_hidden(rt, hp, prompt, ctx, chunk)
        # real TTFT = device prefill + 1-position lm_head (on device when the artifact has one)
        impl.compute_logits(h, hp, position=L - 1, runtime=rt)
        ttft = time.perf_counter() - t0
        rec = {"ctx": L, "ttft_s": ttft, "prefill_tok_s": L / ttft}
        if n_decode and L + n_decode <= ctx:
            t1 = time.perf_counter()
            for j in range(n_decode):
                impl.decode_step(rt, hp, tid, L + j, ctx)
            rec["decode_tok_s"] = n_decode / (time.perf_counter() - t1)
        out.append(rec)
    return out


# ----- backends ---------------------------------------------------------------------------


def build_device(artifact_dir, repo_id, num_sub_devices):
    from .hf_model import AxeleraForCausalLM

    shim = AxeleraForCausalLM.from_artifact(
        artifact_dir, repo_id=repo_id, num_sub_devices=num_sub_devices
    )
    return shim, (lambda ids: shim(input_ids=ids).logits)


def build_fp(repo_id, device, dtype, architecture: str | None = None):
    """The fp reference for `repo_id`, as (tokenizer, ids -> logits).

    `AutoModelForCausalLM` is the normal path. It RAISES for a checkpoint whose `model_type`
    the pinned transformers does not carry (`qwen3_5` is the current case), and supporting
    architectures ahead of that pin is the point of this tree - so fall back to the tree's
    own reference wrapper, which loads the same weights through the model tree instead of
    the HF registry. Without this the regression gate cannot establish a golden for such a
    model at all: it dies before reaching the device, on the leg that needs no device.

    `architecture` names the model tree's architecture (the artifact's
    `pipeline.json['architecture']`); the fallback needs it and cannot infer it from a
    checkpoint transformers already refused to parse.
    """
    import os

    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    import torch
    from transformers import AutoModelForCausalLM

    from ..generate import load_tokenizer

    dt = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype]
    tok = load_tokenizer(repo_id)
    try:
        model = AutoModelForCausalLM.from_pretrained(repo_id, dtype=dt).to(device).eval()
    except (ValueError, KeyError) as e:
        from ..huggingface import _UNKNOWN_ARCH
        from .hf_model import ReferenceForCausalLM

        if _UNKNOWN_ARCH not in str(e) or architecture is None or ReferenceForCausalLM is None:
            raise
        # The reference leg is fp32 by definition; `dtype` is not honoured here because
        # `from_architecture` pins float32 deliberately (a bf16 "reference" is not one).
        model = ReferenceForCausalLM.from_architecture(architecture, repo_id, device=device)
    return tok, (lambda ids: model(input_ids=ids.to(device)).logits.float().cpu())


def decode_consistency_device(shim, artifact_dir, max_positions):
    from ..generate import decode_prefill_consistency

    ids = shim.tokenizer(load_corpus(), return_tensors="np").input_ids[0][: max_positions + 1]
    return decode_prefill_consistency(
        shim._runtime, artifact_dir, ids, shim._gen_config, max_positions=max_positions
    )


# ----- comparisons ------------------------------------------------------------------------


def _delta_logprobs(cur, ref):
    import numpy as np

    a, b = np.asarray(cur, float), np.asarray(ref, float)
    n = min(len(a), len(b))
    d = np.abs(a[:n] - b[:n])
    return float(d.mean()), float(np.percentile(d, 95)), float(d.max())


def compare(now, base, tol):
    """Gate measured `now` against measured `base` (fidelity + decode + determinism; perf is gated
    separately by _perf_gate). Returns (rows, ok); each row is (name, detail, status)."""
    rows, ok = [], True

    def chk(name, passed, detail):
        nonlocal ok
        ok = ok and passed
        rows.append((name, detail, "PASS" if passed else "FAIL"))

    ppl_rel = (now["ppl"] - base["ppl"]) / base["ppl"]
    chk(
        "perplexity",
        ppl_rel <= tol["ppl_rel"],
        f"{now['ppl']:.4f}   ({ppl_rel:+.2%}, tol +{tol['ppl_rel']:.0%})",
    )
    mean, p95, mx = _delta_logprobs(now["corpus_token_logprobs"], base["corpus_token_logprobs"])
    chk(
        "|Δlogprob| mean",
        mean <= tol["logprob_mean_abs"],
        f"{mean:.4f} nats   (tol {tol['logprob_mean_abs']})",
    )
    rows.append(("|Δlogprob| p95", f"{p95:.4f} nats", "----"))  # informational
    chk(
        "|Δlogprob| max",
        mx <= tol["logprob_max_abs"],
        f"{mx:.4f} nats   (tol {tol['logprob_max_abs']})",
    )
    # The reference-free floors (decode<->prefill, and the device head when there is one)
    # are absolute, so the golden gate and the fp-reference run share one implementation.
    sc_rows, sc_ok = self_consistency(now.get("decode_consistency"), tol, now["deterministic"])
    rows.extend(sc_rows)
    return rows, ok and sc_ok


def self_consistency(dc, tol, deterministic: bool) -> tuple[list[tuple[str, str, str]], bool]:
    """PASS/FAIL rows for the signals that need no reference: decode<->prefill, the drift
    onset, the device head where there is one, and determinism."""
    rows: list[tuple[str, str, str]] = []
    ok = True

    def chk(name, passed, detail):
        nonlocal ok
        ok = ok and bool(passed)
        rows.append((name, detail, "PASS" if passed else "FAIL"))

    if dc is None:
        # --decode_positions 0. Say so: an empty gate reads as "checked and fine", and this
        # is the only check that sees the decode path at all.
        rows.append(("decode↔prefill", "NOT MEASURED (--decode_positions 0)", "----"))
    else:
        # Decode runs on the fresh cache, so exact is the healthy reading. A fault that
        # corrupts one KV slot reads exact at first and divergent to the end -- a small mean
        # with a huge last-over-first ratio -- so the onset is gated on its own, not only the
        # mean.
        chk(
            "decode rel_l2",
            dc["decode_rel_l2_mean"] <= tol["decode_rel_l2"],
            f"{dc['decode_rel_l2_mean']:.4f}   (tol {tol['decode_rel_l2']})",
        )
        ratio = dc.get("decode_rel_l2_last_over_first", 0.0)
        onset = (
            ratio > tol["decode_drift_ratio"] and dc["decode_rel_l2_max"] > tol["decode_drift_max"]
        )
        if ratio > 0:
            detail = f"max {dc['decode_rel_l2_max']:.4f}, last/first bin {ratio:.0f}"
            if onset:
                detail += "   INTERMITTENT KV FAULT: exact at first, divergent to the end"
        else:
            detail = "none (exact at every position)"
        chk("decode drift", not onset, detail)
        chk(
            "decode top-1",
            dc["cross_top1_agreement"] >= tol["decode_top1"],
            f"{dc['cross_top1_agreement']:.3f}   (floor {tol['decode_top1']})",
        )
        if "head_top1_agreement" in dc:
            chk(
                "head rel_l2",
                dc["head_rel_l2_mean"] <= tol["head_rel_l2"],
                f"{dc['head_rel_l2_mean']:.4f}   (tol {tol['head_rel_l2']})",
            )
            chk(
                "head top-1",
                dc["head_top1_agreement"] >= tol["head_top1"],
                f"{dc['head_top1_agreement']:.3f}   (floor {tol['head_top1']})",
            )
    chk("determinism", deterministic, "bit-identical" if deterministic else "FORWARDS DIFFER")
    return rows, ok


def _perf_gate(perf, gperf, tol):
    """Single perf verdict vs baseline perf: worst TTFT-slowdown / decode-slowdown across sizes."""
    if not gperf:
        return True, "no baseline perf to compare"
    gmap = {r["ctx"]: r for r in gperf}
    worst, worst_s = -1.0, ""
    for r in perf:
        g = gmap.get(r["ctx"])
        if not g:
            continue
        ts = (r["ttft_s"] - g["ttft_s"]) / g["ttft_s"]
        if ts > worst:
            worst, worst_s = ts, f"TTFT @{r['ctx']} {ts:+.1%}"
        if "decode_tok_s" in r and "decode_tok_s" in g:
            ds = (g["decode_tok_s"] - r["decode_tok_s"]) / g["decode_tok_s"]
            if ds > worst:
                worst, worst_s = ds, f"decode @{r['ctx']} {-ds:+.1%}"
    return worst <= tol["perf_slow"], f"worst {worst_s}   (tol {tol['perf_slow']:.0%})"


def _measured(corpus, dc, perf, deterministic) -> dict:
    """The raw measured values — this is what gets written by --out and read back by --compare-to."""
    return {
        "ppl": corpus["ppl"],
        "corpus_token_logprobs": corpus["corpus_token_logprobs"],
        "perf": perf,
        "decode_consistency": dc,
        "deterministic": deterministic,
    }


def _read_baseline(path: str) -> dict:
    """The measured values from a saved results file (supports the nested or legacy-flat schema)."""
    d = json.loads(Path(path).read_text())
    return d.get("measured", d)


def _read_baseline_metadata(path: str) -> dict:
    """The conditions a saved results file was measured under, or {} for the legacy-flat schema."""
    d = json.loads(Path(path).read_text())
    return d.get("metadata", {}) if isinstance(d.get("metadata"), dict) else {}


def _artifact_num_cores(artifact_dir) -> int | None:
    """The core count the artifact was built for, from its own pipeline.json.

    None for an artifact built before the field existed, which is not the same as 1: a golden
    taken then cannot say what it spanned, and `_cores_gate` treats the two differently.

    A VLM keeps one artifact per tower, and the decoder's is the one these numbers describe.
    """
    for candidate in (
        Path(artifact_dir) / "pipeline.json",
        Path(artifact_dir) / "text" / "pipeline.json",
    ):
        try:
            recorded = json.loads(candidate.read_text()).get("num_cores")
        except (OSError, ValueError):
            continue
        if recorded is not None:
            return int(recorded)
    return None


def _cores_gate(now: int | None, baseline: int | None) -> tuple[bool, str]:
    """Whether these two runs are comparable at all, and why.

    A different core count changes both the numbers and what they mean, so a mismatch has to
    stop the comparison rather than be reported as a difference: a rebuild at another shard
    width and a real regression look identical in every per-metric row. Asked before the
    device is loaded, so a run that cannot be compared is not measured at all.

    Either side unknown passes with a note. Every golden recorded before the field existed is
    that case, and refusing on them would gate nothing while failing everything.
    """
    if now is None or baseline is None:
        missing = "this run" if now is None else "the golden"
        return True, f"core count not comparable: {missing} does not record one"
    if now != baseline:
        return False, (
            f"{now} cores now against {baseline} in the golden: not comparable. Rebuild at "
            f"{baseline} cores, or re-measure the golden at {now} and commit it."
        )
    return True, f"{now} cores, as the golden"


def _metadata(artifact_dir, repo_id, context_len) -> dict:
    """Self-describing provenance for a results file (so a dump anywhere is interpretable)."""
    import datetime
    import platform
    import subprocess

    import lm_eval
    import torch
    import transformers

    def _git():
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=str(HERE),
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        except Exception:
            return None

    dev, dev_full = None, None
    try:
        from axelera.runtime import Context

        devs = Context().list_devices()
        if devs:
            dev_full = str(devs[0])
            dev = getattr(devs[0], "name", None) or dev_full
    except Exception:
        pass
    return {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "artifact_dir": artifact_dir,
        "repo_id": repo_id,
        "context_len": context_len,
        "num_cores": _artifact_num_cores(artifact_dir),
        "git_commit": _git(),
        "device": dev,
        "device_full": dev_full,
        "versions": {
            "lm_eval": lm_eval.__version__,
            "transformers": transformers.__version__,
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
    }


# ----- rendering ----------------------------------------------------------------------------

_TTY = sys.stdout.isatty()
_ANSI = re.compile(r"\x1b\[[0-9;]*m")


def _c(s, code):  # ANSI color, auto-disabled when piped (so logs/CI stay clean)
    return f"\033[{code}m{s}\033[0m" if _TTY else str(s)


def _vis(s):  # visible width, ignoring ANSI codes (so color doesn't break table alignment)
    return len(_ANSI.sub("", str(s)))


def _pad(s, w, align):
    s = str(s)
    gap = w - _vis(s)
    if gap <= 0:
        return s
    if align == ">":
        return " " * gap + s
    if align == "^":
        return " " * (gap // 2) + s + " " * (gap - gap // 2)
    return s + " " * gap


def _table(headers, rows, aligns=None, title=None, note=None):
    """Render a box-drawing table. headers=[] (or all-empty) => no header row."""
    cols = max(len(headers), *(len(r) for r in rows)) if rows else len(headers)
    headers = list(headers) + [""] * (cols - len(headers))
    aligns = aligns or ["<"] * cols
    w = [_vis(headers[i]) for i in range(cols)]
    for r in rows:
        for i in range(cols):
            w[i] = max(w[i], _vis(r[i]))

    def edge(left, mid, right):
        return left + mid.join("─" * (w[i] + 2) for i in range(cols)) + right

    def line(cells):
        return "│ " + " │ ".join(_pad(cells[i], w[i], aligns[i]) for i in range(cols)) + " │"

    if title:
        print("\n" + _c(title, "1") + (_c("   " + note, "2") if note else ""))
    print(edge("┌", "┬", "┐"))
    if any(_vis(h) for h in headers):
        print(line(headers))
        print(edge("├", "┼", "┤"))
    for r in rows:
        print(line(r))
    print(edge("└", "┴", "┘"))


_SYM = {"PASS": ("✓", "32"), "FAIL": ("✗", "31"), "----": ("·", "2")}


def _sym(status):
    s, code = _SYM[status]
    return _c(s, code)


def _print_header(meta, mode_label):
    bits = [meta["repo_id"], f"ctx {meta['context_len']}"]
    if meta.get("device"):
        bits.append(meta["device"])
    if meta.get("git_commit"):
        bits.append(f"@{meta['git_commit']}")
    _table(
        [],
        [
            [_c("REGRESSION", "1"), _c(Path(meta["artifact_dir"]).name, "1")],
            ["target", "  ·  ".join(bits)],
            ["mode", f"{mode_label}  ·  {meta['timestamp']}"],
        ],
        aligns=["<", "<"],
    )


def _print_perf(perf, gperf=None):
    gmap = {r["ctx"]: r for r in (gperf or [])}
    rows = []
    for r in perf:
        g = gmap.get(r["ctx"])
        ttft = f"{r['ttft_s']:.3f}" + (
            f"  ({(r['ttft_s']-g['ttft_s'])/g['ttft_s']:+.1%})" if g else ""
        )
        if "decode_tok_s" in r:
            dec = f"{r['decode_tok_s']:.2f}"
            if g and "decode_tok_s" in g:
                dec += f"  ({(r['decode_tok_s']-g['decode_tok_s'])/g['decode_tok_s']:+.1%})"
        else:
            dec = "—"
        rows.append([str(r["ctx"]), ttft, f"{r['prefill_tok_s']:.1f}", dec])
    _table(
        ["prompt tok", "TTFT (s)", "prefill tok/s", "decode tok/s"],
        rows,
        aligns=[">", ">", ">", ">"],
        title="PERFORMANCE",
        note="warmup'd · 16 decode tokens",
    )


def _print_gate(rows, note):
    trows = [[_sym(status), name, detail] for name, detail, status in rows]
    _table(["", "check", "value"], trows, aligns=["^", "<", "<"], title="GATE", note=note)


def _print_fidelity(device_ppl, fp_ppl, ppl_gap, cmean, cp95, cmax, ref_label):
    _table(
        [],
        [
            ["perplexity", f"{device_ppl:.4f}   vs fp {fp_ppl:.4f}   {ppl_gap:+.2%}"],
            ["|Δlogprob| (nats)", f"mean {cmean:.4f}   p95 {cp95:.4f}   max {cmax:.4f}"],
        ],
        aligns=["<", "<"],
        title="FIDELITY",
        note=f"device vs {ref_label}",
    )


def _print_device(dc, deterministic):
    rows = []
    if dc is not None:
        rows.append(
            [
                "decode↔prefill",
                f"rel_l2 {dc['decode_rel_l2_mean']:.4f}   "
                f"top-1 {dc['cross_top1_agreement']:.3f}",
            ]
        )
        if "head_top1_agreement" in dc:
            rows.append(
                [
                    "device↔host head",
                    f"rel_l2 {dc['head_rel_l2_mean']:.4f}   "
                    f"top-1 {dc['head_top1_agreement']:.3f}",
                ]
            )
    rows.append(
        [
            "determinism",
            _c("bit-identical ✓", "32") if deterministic else _c("FORWARDS DIFFER ✗", "31"),
        ]
    )
    _table([], rows, aligns=["<", "<"], title="DEVICE")


def _print_summary(ppl, ppl_gap, perf, verdict):
    """One-line digest table: accuracy (ppl, +gap vs fp) · avg prefill tok/s · avg decode tok/s · verdict."""
    parts = []
    if ppl is not None:
        s = f"ppl {ppl:.2f}"
        if ppl_gap is not None:
            s += f" ({ppl_gap:+.1%} vs fp)"
        parts.append(s)
    pre, dec = _avg(perf, "prefill_tok_s"), _avg(perf, "decode_tok_s")
    if pre is not None:
        parts.append(f"prefill {pre:.0f} tok/s")
    if dec is not None:
        parts.append(f"decode {dec:.1f} tok/s")
    if verdict:
        parts.append(verdict)
    _table([], [["SUMMARY", "   ·   ".join(parts)]], aligns=["<", "<"])


def _avg(perf, key):
    vals = [r[key] for r in perf if key in r]
    return sum(vals) / len(vals) if vals else None


def main(argv=None) -> int:
    import numpy as np
    import torch

    p = argparse.ArgumentParser(description="Fast device-LLM regression check.")
    p.add_argument(
        "artifact_dir",
        nargs="?",
        default=None,
        help="prebuilt artifact directory (positional; or use --artifact_dir)",
    )
    p.add_argument(
        "--artifact_dir",
        dest="artifact_dir_flag",
        default=None,
        help="prebuilt artifact directory (alternative to the positional)",
    )
    p.add_argument(
        "--repo_id",
        default=None,
        help="HF repo (tokenizer + fp reference); defaults to the artifact's pipeline.json",
    )
    p.add_argument(
        "--generation",
        default=None,
        choices=("OMEGA", "EUROPA"),
        help="target hardware generation; defaults to the artifact's pipeline.json",
    )
    p.add_argument(
        "--out",
        default=None,
        help="write this run's full results JSON to PATH (anywhere; serves as a future baseline)",
    )
    p.add_argument(
        "--compare-to",
        dest="compare_to",
        default=None,
        help="GOLDEN GATE: gate against a saved results JSON at PATH (else default = fp reference)",
    )
    p.add_argument(
        "--perf-only",
        dest="perf_only",
        action="store_true",
        help="report only performance numbers; skip all fidelity/reference scoring",
    )
    p.add_argument(
        "--ref-device",
        dest="ref_device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="where to run the fp reference in DEFAULT (fp) mode",
    )
    p.add_argument(
        "--ref-dtype",
        dest="ref_dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
    )
    p.add_argument(
        "--no-ref",
        dest="no_ref",
        action="store_true",
        help="in golden-gate mode, skip the fp reference comparison (faster gate)",
    )
    p.add_argument(
        "--decode_positions", type=int, default=128, help="window for decode<->prefill (0=skip)"
    )
    p.add_argument("--n_decode", type=int, default=16, help="decode steps per perf point")
    args = p.parse_args(argv)

    from ..generate import NUM_SUB_DEVICES, artifact_generation, artifact_repo_id

    artifact_dir = args.artifact_dir or args.artifact_dir_flag
    if not artifact_dir:
        p.error("artifact_dir is required (positional, or --artifact_dir)")
    repo_id = args.repo_id or artifact_repo_id(artifact_dir)
    if repo_id is None:
        raise SystemExit(f"no repo_id in {artifact_dir}/pipeline.json; pass --repo_id")
    generation = args.generation or artifact_generation(artifact_dir)
    cores_note = None
    if args.compare_to:
        # Asked BEFORE the device is touched. A comparison across two core counts cannot mean
        # anything, so there is no reason to spend a measurement finding that out, and no
        # results file is written: nothing was measured.
        cores_ok, cores_note = _cores_gate(
            _artifact_num_cores(artifact_dir),
            _read_baseline_metadata(args.compare_to).get("num_cores"),
        )
        if not cores_ok:
            print(_c(f"REFUSED  {cores_note}", "31"))
            return 2
    shim, dev_logits = build_device(artifact_dir, repo_id, NUM_SUB_DEVICES[generation])
    ctx = shim._context_len
    meta = _metadata(artifact_dir, repo_id, ctx)
    mode_label = (
        "perf-only" if args.perf_only else ("golden gate" if args.compare_to else "fp-reference")
    )
    _print_header(meta, mode_label)

    # ---- PERF-ONLY mode ----
    if args.perf_only:
        perf = perf_numbers(shim, args.n_decode)
        _print_perf(perf)
        _print_summary(None, None, perf, None)
        if args.out:
            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            Path(args.out).write_text(
                json.dumps({"metadata": meta, "measured": {"perf": perf}}, indent=2)
            )
            print(f"wrote results -> {args.out}")
        return 0

    # ---- full device measurement (GOLDEN + FP modes) ----
    # Before score_corpus: the check needs a zero-init cache (generate.decode_prefill_consistency),
    # and prefill rewrites every slot it needs. A --phases prefill artifact has no decode
    # graph to check.
    _graphs = getattr(shim._runtime, "_spec", {}).get("graphs", {"decode": None})
    dc = (
        decode_consistency_device(shim, artifact_dir, args.decode_positions)
        if args.decode_positions and "decode" in _graphs
        else None
    )

    # Recurrent/conv models carry persistent state that prefill threads and does not
    # overwrite (unlike the position-indexed KV cache): zero it before each corpus forward,
    # so the decode check above cannot leak into the score and the two forwards start alike.
    def fresh_state():
        if hasattr(shim._runtime, "reset_state"):
            shim._runtime.reset_state()

    fresh_state()
    corpus = score_corpus(dev_logits, shim.tokenizer, ctx)
    _ids = shim.tokenizer(load_corpus(), return_tensors="pt").input_ids[0][
        :ctx
    ]  # determinism re-forward
    fresh_state()
    with torch.no_grad():
        _lp2 = _target_logprobs(dev_logits(_ids.unsqueeze(0))[0], _ids, 1).float().tolist()
    deterministic = bool(np.array_equal(corpus["corpus_token_logprobs"], _lp2))
    perf = perf_numbers(shim, args.n_decode)
    measured = _measured(corpus, dc, perf, deterministic)
    results = {"metadata": meta, "measured": measured}

    # fp reference: always in fp mode; in golden mode too, unless --no-ref (it's shown either way).
    run_fp = (not args.compare_to) or (not args.no_ref)
    fp_c = ppl_gap = cmean = cp95 = cmax = None
    if run_fp:
        print(
            "\n" + _c(f"· scoring fp reference ({args.ref_device}, {args.ref_dtype}) …", "2"),
            flush=True,
        )
        # The artifact names its architecture, which is what lets the reference fall back to
        # the model tree for a checkpoint transformers cannot parse.
        fp_tok, fp_logits = build_fp(
            repo_id,
            args.ref_device,
            args.ref_dtype,
            architecture=(getattr(shim, "_pipeline", None) or {}).get("architecture"),
        )
        fp_c = score_corpus(fp_logits, fp_tok, ctx)
        cmean, cp95, cmax = _delta_logprobs(
            corpus["corpus_token_logprobs"], fp_c["corpus_token_logprobs"]
        )
        ppl_gap = (corpus["ppl"] - fp_c["ppl"]) / fp_c["ppl"]
        results["reference_fp"] = {
            "ref_device": args.ref_device,
            "ref_dtype": args.ref_dtype,
            "fp_ppl": fp_c["ppl"],
            "ppl_gap": ppl_gap,
            "mean_abs_dlogprob": cmean,
            "p95_abs_dlogprob": cp95,
            "max_abs_dlogprob": cmax,
        }
    ref_label = f"fp {args.ref_dtype} reference ({args.ref_device})"

    rc = 0
    if args.compare_to:
        # ---- GOLDEN GATE (vs golden) + fp reference (informational) ----
        base = _read_baseline(args.compare_to)
        rows, fid_ok = compare(measured, base, DEFAULT_TOL)
        # The condition the comparison was allowed under, so a PASS says what it compared.
        rows.append(("core count", cores_note, "PASS"))
        perf_ok, perf_note = _perf_gate(perf, base.get("perf"), DEFAULT_TOL)
        rows.append(("performance", perf_note, "PASS" if perf_ok else "FAIL"))
        ok = fid_ok and perf_ok
        _print_gate(rows, f"vs golden · {args.compare_to}")
        if run_fp:
            _print_fidelity(corpus["ppl"], fp_c["ppl"], ppl_gap, cmean, cp95, cmax, ref_label)
        _print_perf(perf, base.get("perf"))
        verdict = _c("PASS", "32") if ok else _c("FAIL", "31")
        results["comparison"] = {
            "baseline": args.compare_to,
            "result": "PASS" if ok else "FAIL",
            "rows": [{"name": n, "detail": d, "status": s} for n, d, s in rows],
        }
        rc = 0 if ok else 1
    else:
        # ---- FP REFERENCE mode: the quantization gap is informational, but the
        # reference-free signals are gated. They are what catches a decode-path defect,
        # which perplexity structurally cannot see.
        _print_fidelity(corpus["ppl"], fp_c["ppl"], ppl_gap, cmean, cp95, cmax, ref_label)
        _print_device(dc, deterministic)
        _print_perf(perf)
        rows, ok = self_consistency(dc, DEFAULT_TOL, deterministic)
        _print_gate(rows, "no golden · absolute floors only")
        results["self_consistency"] = {
            "result": "PASS" if ok else "FAIL",
            "rows": [{"name": n, "detail": d, "status": st} for n, d, st in rows],
        }
        verdict = _c("PASS", "32") if ok else _c("FAIL", "31")
        rc = 0 if ok else 1

    _print_summary(corpus["ppl"], ppl_gap, perf, verdict)

    if args.out:
        write_results(results, Path(args.out), rc=rc, golden_mode=bool(args.compare_to))
    return rc


def write_results(results: dict, out: Path, *, rc: int, golden_mode: bool) -> Path | None:
    """Write this run's results JSON to ``out``; in FP-reference mode only a passing run, since
    `run_tier` takes that file as the golden and a failing one would gate every later build
    against a breached artifact. Golden mode archives every run. Returns the path, or None."""
    if not golden_mode and rc != 0:
        print(f"not writing {out}: the run failed its own floors, so it is not a golden")
        return None
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"wrote results -> {out}")
    return out


if __name__ == "__main__":
    import sys

    sys.exit(main())

# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Context-length sweep across the shipped AxScript LLMs.

Rebuilds a model at each requested `context_len`, measuring ttft, decode
throughput, on-device perplexity, and the KV-cache DDR estimate, then writes
the measured rows to `<--rows-dir>/<stem>.yaml`.

This tool MEASURES; it does not commit evidence. Filing a measured row into the
committed report tree is a separate step, where a row is filed beside the card
status it is evidence for, merged into what is already committed: the hand-written
comment headers, the `incomplete` and `undeclared_reason` judgements, and the
failure rows an earlier attempt recorded. A second writer into that tree would
overwrite precisely the human judgements it exists to hold, so the rows this tool
leaves behind are measurement output, and taking one into the tree is that
separate step's job.

`GenerationConfig.supported_context_lens` gates a build to sizes already
measured on the target part (`shared/artifact_contract.py::validate_run_size`): this
tool is what produces those measurements. A size outside a model's shipped set
is not silently widened for everyone: `_widen_for_sweep` extends only the
throwaway `GenerationConfig` this sweep builds, one size at a time right
before that size's build, and logs the override. The shipped `make_config()`
default is untouched, so a plain `--build` still refuses anything past the
shipped set until a human reviews a row here and widens the model's own
`supported_context_lens`.

A failed size is a row, not a gap: it is recorded with `status` + `note`
rather than skipped, because that failure is exactly what justifies the
shipped set staying narrow. A size this tool never got to measure at all
(the process never completed even once, across repeated attempts) is a
different case, `status: BLOCKED` rather than `DEVICE_LOST`/`FAILED`: this
harness's own exception handling never ran, since nothing was ever caught
in-process, so that row is written by hand describing what was observed
and what is not established about why.

Usage (from a source checkout root):

    python model_axscript/llm/shared/context_sweep.py --model qwen2_5_0_5b
    python model_axscript/llm/shared/context_sweep.py --model all --context-lens 128,512
    python model_axscript/llm/shared/context_sweep.py --model gemma_3_1b --device metis-0:2c:0
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import sys
from datetime import date
from pathlib import Path
from time import perf_counter

import numpy as np
import yaml

# model_axscript is a source dir at the repo root (not pip-installed), same as each
# model's own __main__.py.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model_axscript.shared.artifact_contract import (  # noqa: E402
    artifact_toolchain,
    validate_artifact,
    validate_run_size,
)
from model_axscript.shared.checkpoint import hf_access  # noqa: E402
from model_axscript.shared.quantized_model import MANIFEST  # noqa: E402
from model_axscript.shared.runtime import (  # noqa: E402
    _NP_DTYPE,
    _WEDGED_DEVICE_MARKERS,
    RuntimeModel,
    probe_runtime_load,
    required_sub_devices,
)
from model_axscript.llm.generate import (  # noqa: E402
    benchmark_generation,
    device_perplexity,
    export_quantized_model,
)
from model_axscript.llm.implementation import get_implementation  # noqa: E402
from model_axscript.llm.shared.memory import context_memory  # noqa: E402

logger = logging.getLogger(__name__)

STEMS = ()  # no bundled models remain; the next tree lands its own stems here
_DEFAULT_CONTEXT_LENS = (128, 512, 1024, 2048)
# Fixed across context_lens so perplexity rows are comparable to each other; the test
# split (never calibration's train split, see shared/calibration.py) keeps eval held out.
_EVAL_TOKENS = 2048
_MAX_NEW_TOKENS = 64  # a throughput measurement, not a full generation
_DEFAULT_OUTPUT_ROOT = Path("build") / "context-sweep"


class PreparationFailure(RuntimeError):
    """The model could not be prepared, so no context_len was compiled or measured.

    Everything before the per-size compile: the checkpoint, its tokenizer, the eval corpus,
    the quantization. None of it reads context_len, so a failure here says nothing about any
    requested size, and recording one row per size would claim each was attempted when none
    was. A caller that files these rows as evidence needs that apart from "this size did not
    run": they are different findings and warrant different messages. A gated checkpoint is
    the common case, and `hf_access` has already turned it into a message naming the repo
    and how to get a token by the time it arrives here.
    """


def _load_model(stem: str):
    """Import a model package by its directory stem; returns (make_config, MODEL_TAG)."""
    mod = importlib.import_module(f"model_axscript.llm.{stem}")
    return mod.make_config, mod.MODEL_TAG


def _widen_for_sweep(config, context_len: int) -> None:
    """Opt exactly this measurement build out of validate_run_size's gate.

    Mutates only the harness's own throwaway GenerationConfig instance, never the
    model's make_config() default, so a plain build stays gated at the shipped set.
    """
    # Resolved BEFORE the set is widened, so the fallback in `chunk_len_for` still sees the
    # sizes the CARD declared and picks the widest chunk this model is known to compile.
    # Assigned here because the config was constructed at another size: `chunk_len` is derived
    # from `chunk_lens` at construction, and a size moved afterwards has to move it too, or the
    # sweep prefills a 128-token context with a 1024-wide graph.
    chunk_len = config.chunk_len_for(context_len)
    config.context_len = context_len
    config.chunk_len = chunk_len
    if context_len not in config.supported_context_lens:
        logger.warning(
            "[context-sweep] measuring context_len=%d, outside the shipped supported set "
            "%s; this widens only this sweep's own config, not the shipped default",
            context_len,
            config.supported_context_lens,
        )
        chunks = dict(zip(config.supported_context_lens, config.chunk_lens or ()))
        config.supported_context_lens = tuple(
            sorted({*config.supported_context_lens, context_len})
        )
        if config.chunk_lens is not None:
            # Kept positionally aligned with the set it was widened alongside; the new size
            # takes the width resolved above.
            chunks[context_len] = chunk_len
            config.chunk_lens = tuple(chunks[n] for n in config.supported_context_lens)


def _kv_cache_mib(model_config: dict, context_len: int) -> float | None:
    """Total KV-cache DDR (MiB, all layers) at context_len; None if a dim is missing.

    Same extraction generate.py._log_context_memory uses, reusing the same
    context_memory arithmetic rather than re-deriving it.
    """
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
    except (KeyError, TypeError, ValueError, ZeroDivisionError):
        return None
    return round(memory.kv_cache_mib * memory.num_layers, 1)


def _compiled_ddr_mib(artifact_dir: Path) -> float:
    """Total DDR footprint (MiB) of the compiled artifact: weights + KV cache + prefill
    mask + workspace, everything - the sum of model.json's own `memory_pools` declaration.

    This is the compiler's actual buffer plan, not a re-derivation of it: it sidesteps
    guessing whether a term (e.g. the prefill mask, a `shared_input` bound to the same
    tensor name on every layer's KernelCall) is allocated once or once per layer, which
    `_kv_cache_mib` above cannot settle from `memory.py`'s formulas alone.
    """
    spec = json.loads((artifact_dir / "model.json").read_text())
    total = sum(p["size"] for p in spec["memory_pools"] if p.get("memory") == "ddr")
    return round(total / (1024 * 1024), 2)


_METIS_TIERS_GIB = (2, 4, 8, 16)
# MEASURED via a ceiling bisection on qwen2.5-0.5b (16GiB Metis, every rung cross-device
# liveness-verified): ctx3072 (1385.8 MiB static) passed, ctx3584 (1648.33 MiB) passed,
# ctx4096 (1938.85 MiB) failed - runtime-vs-static overhead factor brackets to 8.45x-9.94x.
# Applied to ctx2048's own 944.74 MiB static footprint that bracket means 7.80-9.17 GiB of
# actual DRAM in practice, straddling the 8GiB/16GiB tier boundary (~8.67x) - one more rung
# would only narrow the bracket, not resolve which side of 8.67x the true factor lands on
# without a lot more of them. RECOMMENDATION (not yet confirmed): stop bisecting here and
# use the conservative tier; 12.0 is the upper end rounded up from 9.94x, since under-
# declaring a customer's required DRAM is the dangerous direction and over-declaring only
# costs some capability on a smaller part. This is a global default calibrated on ONE
# model (qwen); do not tighten it toward 9.94x on the strength of another model's rows
# without bisecting that model too - a different attention/KV-cache shape can have a
# different runtime-vs-static factor.
_DDR_HEADROOM = 12.0


def _min_dram_tier_gib(ddr_mib: float) -> int | None:
    """Smallest Metis DRAM tier that fits ddr_mib with `_DDR_HEADROOM` margin.

    DERIVED, not measured: labelled separately from `ddr_mib` in the report row so the
    two are never conflated (the compiled footprint is a fact; this bound is inferred
    from it and the stated headroom). None means it exceeds every shipped tier.
    """
    required = ddr_mib * _DDR_HEADROOM
    return next((t for t in _METIS_TIERS_GIB if t * 1024 >= required), None)


def _build_one(config, bundle) -> Path:
    """Compile the artifact for config.context_len, reusing an already-quantized bundle.

    Quantization does not depend on context_len (only the device-kernel compile
    does), so the caller quantizes once per model and calls this once per size.
    Mirrors generate.build_artifact's body, minus the export step it reuses.
    """
    validate_run_size(config.context_len, config.supported_context_lens, "context_len")
    out_dir = Path(config.artifact_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    bundle.save(out_dir / "quantized_model")
    artifact_dir = get_implementation(config.implementation).compile(bundle, config)
    validate_artifact(artifact_dir, known_dtypes=_NP_DTYPE.keys())
    probe_runtime_load(artifact_dir)
    return artifact_dir


def _exported_config(artifact_dir: Path, bundle) -> dict:
    """The exported model's dims: from the bundle in hand, else the one it left on disk.

    A reused artifact was quantized by an earlier run, so there is no bundle here. Its
    saved manifest carries the same dict (and no tensors, so reading it is cheap), which
    keeps a reused row carrying the same `kv_cache_mib` a freshly built one carries instead
    of quietly dropping the field on a re-measurement.
    """
    if bundle is not None:
        return bundle.config
    manifest = Path(artifact_dir) / "quantized_model" / MANIFEST
    try:
        return json.loads(manifest.read_text()).get("config", {})
    except (OSError, ValueError):
        return {}


def _already_built(artifact_dir: Path, context_len: int) -> bool:
    """True when artifact_dir holds a compiled artifact for THIS context_len.

    The artifact's own pipeline.json is what is asked, never the directory name: a
    directory says what someone meant to build there, and reusing a hit without checking
    what it actually is would measure one size and file the numbers under another.
    """
    pipeline_path = Path(artifact_dir) / "pipeline.json"
    if not pipeline_path.is_file():
        return False
    try:
        compiled = json.loads(pipeline_path.read_text()).get("context_len")
    except (OSError, ValueError):
        return False
    return compiled == context_len


def _eval_ids(tokenizer, n_tokens: int) -> np.ndarray:
    """First n_tokens of wikitext-2 TEST (held out from calibration's TRAIN split)."""
    import datasets

    ds = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(t for t in ds["text"] if t.strip())
    ids = tokenizer(text, return_tensors="np").input_ids[0]
    return ids[:n_tokens]


def _classify_failure(exc: Exception) -> str:
    """DEVICE_LOST for a device-execution failure, FAILED for everything else.

    An `axelera.runtime.objects.ExecutionError` (a job that reached the device and did not
    complete) is the reliable signal - its message never contains distinguishing text (see
    runtime.py's device_job), so type is what device_job itself now matches on. The marker
    strings remain a fallback for this harness's own cross-device-reproduction message
    (constructed below, a plain RuntimeError that deliberately contains "DEVICE_LOST").
    """
    try:
        from axelera.runtime.objects import ExecutionError

        if isinstance(exc, ExecutionError):
            return "DEVICE_LOST"
    except ImportError:
        pass
    if any(marker in str(exc) for marker in _WEDGED_DEVICE_MARKERS):
        return "DEVICE_LOST"
    return "FAILED"


def _measure_on_device(runtime_device, artifact_dir, tokenizer, config, eval_ids):
    """Connect to runtime_device and take one context_len's measurements.

    A fresh RuntimeModel per call (own Context + connection), released before returning
    so the next call (same device or a fallback) can connect cleanly.

    Perplexity is scored exactly as the shared eval path scores it: no BOS token is
    prepended to a window, so a sweep number is comparable with a gate number.
    """
    spec = json.loads((artifact_dir / "model.json").read_text())
    runtime = RuntimeModel(artifact_dir / "model.json", device=runtime_device)
    try:
        device_name = runtime._device.name
        # What the run actually RESERVED, not the compiled span: a build whose L2 arena is
        # wider than its span reserves more (see shared.runtime.required_sub_devices), and a
        # report row saying `cores: 1` for a run the banner called `cores: 3` is not evidence
        # of the run that happened. Needs the DeviceInfo, so it is read off the live runtime.
        cores = required_sub_devices(spec, runtime._device)
        # `benchmark_generation` owns the timing and reports only timing. The prompt length is
        # the caller's to state (as the zoo's own benchmark writer states it), so the caller
        # keeps it: `sweep_one` computes it once and writes it into the row itself, rather than
        # this function stashing a key in another function's return value for a reader sixty
        # lines away. Reading a `prompt_tokens` key off the bench dict failed every LLM perf
        # row with a KeyError, which is what that arrangement cost the first time.
        bench = benchmark_generation(runtime, tokenizer, artifact_dir, config.prompt, config)
        ppl = device_perplexity(runtime, artifact_dir, eval_ids, config)
    finally:
        runtime.instance.release()
        runtime.connection.release()
        runtime.ctx.release()
    return bench, ppl, device_name, cores


def sweep_one(
    stem: str,
    context_lens,
    device,
    fallback_device,
    cal_device: str,
    out_root: Path,
    on_row=lambda block: None,
    bundle=None,
    reuse_existing: bool = False,
) -> tuple[dict, object]:
    """Build + measure one model across context_lens; returns (report block, bundle).

    A DEVICE_LOST on `device` is re-run once on `fallback_device` before being trusted: a
    wedge looks identical to a genuine size-driven DEVICE_LOST, and recording a mid-sweep
    hardware fault as a size limit would put a fabricated ceiling into a committed report.
    If it does not reproduce on the fallback, that row's measurement (from the fallback)
    is kept and the sweep's active device switches to the fallback for the remaining
    sizes; if it does reproduce, the size is genuinely the limit and `device` stays active.

    `on_row` runs after every row (success or failure), given the report block so far, so
    a caller can write it to disk immediately - a long sweep must not leave nothing on disk
    until every context_len finishes; a partial result is still an honest, inspectable one.

    `bundle`: an already-quantized QuantizedModel to reuse instead of quantizing from
    scratch (quantization is context_len-independent). Also returned, so a caller making
    several separate sweep_one calls for the same model (e.g. an adaptive bisection, where
    the next context_len isn't known until the previous one's result is seen) can pass it
    into the next call rather than pay calibration again each time.

    `reuse_existing`: measure an artifact already compiled at that size instead of
    rebuilding it (and, when every size is already built, never quantize at all). OFF by
    default, because a sweep exists to measure what the current source compiles to, and a
    directory left by an earlier one would silently make it measure old code. A caller that
    wants an artifact reused states so. A row carries `build_s` only when this call built
    it; a reused one was not timed here, and there is no honest number to put there.

    A failure to prepare the model (checkpoint, tokenizer, eval corpus, quantization) raises
    `PreparationFailure` rather than being recorded once per size: none of it is
    size-dependent, so no requested size was ever attempted.
    """
    # `load_tokenizer`, not `AutoTokenizer`: see the note in `entry.py`. The sweep is what
    # `axzoo report` drives, so a checkpoint with an unimportable `tokenizer_class` would
    # fail its perf rows rather than the build that produced them.
    from ..generate import load_tokenizer

    make_config, model_tag = _load_model(stem)
    base_config = make_config(out_root / stem, cal_device=cal_device)
    try:
        with hf_access(base_config.repo_id):
            tokenizer = load_tokenizer(base_config.repo_id)
        eval_ids = _eval_ids(tokenizer, _EVAL_TOKENS)
        # How many tokens the request carries, which every row records as `seq_len`. A
        # property of the prompt and the tokenizer, so it is the same at every context_len
        # and belongs here rather than inside the per-size measurement.
        # `base_config.prompt`, so every size in the sweep measures the same request: the
        # per-size configs below are derived from this one and never restate it.
        prompt_tokens = len(tokenizer(base_config.prompt).input_ids)
    except Exception as exc:
        raise PreparationFailure(str(exc)) from exc

    runs: list[dict] = []
    active_device = device
    device_name: str | None = None
    cores: int | None = None
    # What COMPILED the artifacts these rows are measured on, read back off each one rather
    # than asked of this shell: `reuse_existing` measures a directory an earlier shell may
    # have built, and this shell's package versions would then describe the wrong compile.
    toolchain: dict[str, str] = {}
    block = {
        "model_tag": model_tag,
        "device": device_name,
        "cores": cores,
        "toolchain": toolchain,
        "runs": runs,
    }
    for context_len in context_lens:
        config = make_config(out_root / stem / f"ctx{context_len}", cal_device=cal_device)
        config.max_new_tokens = _MAX_NEW_TOKENS
        row: dict = {"context_len": context_len}
        build_s: int | None = None
        try:
            # INSIDE the try: widening can now refuse a size (a context under every chunk width
            # this model declares, see `GenerationConfig.chunk_len_for`), and a refusal is this
            # size's result, not the end of the sweep. Above the try it took the remaining
            # sizes down with it and left no row saying why.
            _widen_for_sweep(config, context_len)
            # The prefill is ingested in chunk_len-token chunks, so it is a divisor of the
            # TTFT beside it: the same artifact at the same context_len reports a different
            # first-token latency at a different chunk. Stated per row rather than inferred
            # from the shipped config, which a later commit moves without touching the row.
            #
            # Written HERE and not beside the timings, because the chunk is the knob this
            # build was compiled with and is known before anything can fail: a size that ends
            # as a status row still states it. Written on the success path only, a DEVICE_LOST
            # re-measurement replaced a committed row and dropped the value with it, and no
            # carry-forward can put back a key the sweep never wrote for the VLM family.
            if config.chunk_len is not None:
                row["chunk_len"] = config.chunk_len
            if reuse_existing and _already_built(config.artifact_dir, context_len):
                logger.info(
                    "[context-sweep] %s context_len=%d: measuring the artifact already at %s",
                    stem,
                    context_len,
                    config.artifact_dir,
                )
                artifact_dir = Path(config.artifact_dir)
            else:
                started = perf_counter()
                if bundle is None:  # quantize once per model, reuse across every context_len
                    try:
                        bundle = export_quantized_model(config)
                    except Exception as exc:
                        raise PreparationFailure(str(exc)) from exc
                artifact_dir = _build_one(config, bundle)
                build_s = round(perf_counter() - started)
            # Off the DIRECTORY, so a size whose measurement failed still says what compiled
            # the artifact it failed on, and a reused artifact reports the shell that built
            # it rather than this one.
            toolchain = toolchain or artifact_toolchain(artifact_dir)
            try:
                bench, ppl, row_device, row_cores = _measure_on_device(
                    active_device, artifact_dir, tokenizer, config, eval_ids
                )
            except Exception as exc:
                if fallback_device is None:
                    raise
                # A wedged card and a genuine device-side failure (e.g. exceeding DRAM at
                # prefill) raise the IDENTICAL axelera.runtime exception message here (both
                # surface as "AXR_ERROR_RUNTIME_ERROR: Failed to execute command lists" - the
                # distinguishing driver text like "DEVICE_LOST"/"MSI" is printed straight to
                # stderr by the C library, never part of the Python exception), so there is
                # no string to pre-filter on. Always re-run on the fallback instead of trying
                # to classify first.
                logger.warning(
                    "[context-sweep] %s context_len=%d failed on %s (%s); re-running on "
                    "fallback %s to tell a size limit from a mid-sweep wedge",
                    stem,
                    context_len,
                    active_device,
                    exc,
                    fallback_device,
                )
                try:
                    bench, ppl, row_device, row_cores = _measure_on_device(
                        fallback_device, artifact_dir, tokenizer, config, eval_ids
                    )
                except Exception as exc2:
                    # Reproduced on a 2nd (presumed-healthy) device: not a single wedged
                    # card, so call it DEVICE_LOST regardless of what either message said.
                    raise RuntimeError(
                        f"DEVICE_LOST: reproduced on both {active_device} ({exc}) and "
                        f"{fallback_device} ({exc2})"
                    ) from exc2
                row["note"] = (
                    f"measured on fallback {fallback_device} ({active_device} failed here "
                    "but not on the fallback: a mid-sweep wedge, not a size limit)"
                )
                active_device = fallback_device
            if device_name is not None and row_device != device_name:
                note = f"device switched from {device_name} to {row_device}"
                row["note"] = f"{row['note']}; {note}" if "note" in row else note
            device_name = row_device
            if cores is not None and row_cores != cores:
                note = f"cores={row_cores} at this size (previously {cores})"
                row["note"] = f"{row['note']}; {note}" if "note" in row else note
            cores = row_cores
            kv_mib = _kv_cache_mib(_exported_config(artifact_dir, bundle), context_len)
            row["seq_len"] = prompt_tokens
            row["perplexity"] = round(ppl, 3)
            # Which of the two perplexities this is, said in the row. `device_perplexity`
            # scores the sweep's own corpus and is token-denominated; the card's eval figure is
            # lm-eval wikitext word_perplexity and is word-denominated. Nothing else in the row
            # distinguishes them, and the gap between them reads as a regression to anyone who
            # differences a report against a card.
            row["perplexity_harness"] = "device-probe"
            row["ttft_ms"] = round(bench["ttft_s"] * 1000, 1)
            row["decode_tok_s"] = round(bench["decode_tok_per_s"], 1)
            row["ddr_mib"] = _compiled_ddr_mib(artifact_dir)
            # `_min_dram_tier_gib` is deliberately NOT written into the row. It multiplies the
            # footprint by `_DDR_HEADROOM`, a factor bisected on ONE model and labelled a
            # recommendation, and the answers contradict the row they would sit in: every model
            # here RAN on the part named in `device`, while the derived bound says three of them
            # need 16 GiB and the rest need more than any tier ships. A reader has only the row,
            # `min_dram_gib` reads as measured next to `ddr_mib`, and there is no value -- `null`
            # included, which means "fits no tier" -- that is true of these builds. It comes back
            # when the factor is bisected per model; until then `ddr_mib` is the fact, on its own.
            if kv_mib is not None:
                row["kv_cache_mib"] = kv_mib
            if build_s is not None:
                row["build_s"] = build_s
            # The request and its answer, last in the row so the numbers above stay readable
            # at a glance. Written only under GREEDY decoding: at a non-zero temperature the
            # continuation is one draw, and a reviewer differencing two reports would read the
            # sampler as a numerics change. The condition lives here because this is the only
            # place that can see the config the row was measured with.
            if config.temperature == 0:
                row["prompt"] = config.prompt
                row["response"] = bench["response"]
        except PreparationFailure:
            # Nothing was attempted at this size, so it gets no row.
            raise
        except Exception as exc:  # noqa: BLE001 - a failed size is a recorded row, not a crash
            logger.error("[context-sweep] %s context_len=%d failed: %s", stem, context_len, exc)
            row["status"] = _classify_failure(exc)
            row["note"] = str(exc).splitlines()[0][:200]
        runs.append(row)
        block = {
            "model_tag": model_tag,
            "device": device_name,
            "cores": cores,
            "toolchain": toolchain,
            "runs": runs,
        }
        on_row(block)

    return block, bundle


def write_measured_rows(stem: str, block: dict, out_dir: Path, command: str) -> Path:
    """Write this sweep's rows to `<out_dir>/<stem>.yaml`, with the provenance they need.

    A row on its own says nothing checkable, so the device, the core count, the date and
    the exact command that took it are written beside it: a reader of these numbers has to
    be able to tell what took them and repeat it.

    Every row is written, the failures included, and no card status is: a status is a claim
    about a model that a human makes on a card, and this file is a measurement. Filing any
    of this as evidence, under the status the card carries, is a separate, later step.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{stem}.yaml"
    data = {
        "model_tag": block["model_tag"],
        "date": date.today().isoformat(),
        "device": block["device"],
        "cores": block["cores"],
        "toolchain": block.get("toolchain") or {},
        "command": command,
        "runs": block["runs"],
    }
    path.write_text(yaml.safe_dump(data, sort_keys=False, default_flow_style=False))
    return path


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--model", choices=(*STEMS, "all"), default="all", help="which model(s) to sweep"
    )
    parser.add_argument(
        "--context-lens",
        default=",".join(str(n) for n in _DEFAULT_CONTEXT_LENS),
        help="comma-separated context lengths to rebuild + measure",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Metis device to run on (name or index); default is the first device",
    )
    parser.add_argument(
        "--fallback-device",
        default=None,
        help=(
            "a second known-good Metis device: on a DEVICE_LOST, the same size is "
            "re-run here once to tell a size-driven limit from a mid-sweep wedge on "
            "--device. Strongly recommended whenever device health is not freshly verified."
        ),
    )
    parser.add_argument("--cal-device", default="", help="torch device for calibration")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=_DEFAULT_OUTPUT_ROOT,
        help="where to compile the swept artifacts",
    )
    parser.add_argument(
        "--rows-dir",
        type=Path,
        default=None,
        help=(
            "where to write the measured rows, as <rows-dir>/<stem>.yaml; defaults to "
            "--output-root, beside the artifacts they describe. Never the committed "
            "reports tree: filing a row there is a separate step, merged with what a "
            "human already wrote in it"
        ),
    )
    args = parser.parse_args(argv)
    rows_dir = args.rows_dir or args.output_root

    context_lens = [int(n) for n in args.context_lens.split(",")]
    stems = STEMS if args.model == "all" else (args.model,)
    command = (
        f"python model_axscript/llm/shared/context_sweep.py --model {args.model} "
        f"--context-lens {args.context_lens}"
        + (f" --device {args.device}" if args.device is not None else "")
        + (
            f" --fallback-device {args.fallback_device}"
            if args.fallback_device is not None
            else ""
        )
    )

    for stem in stems:

        def on_row(block, stem=stem):
            path = write_measured_rows(stem, block, rows_dir, command)
            print(f"[context-sweep] wrote {path} ({len(block['runs'])} row(s))")

        try:
            sweep_one(
                stem,
                context_lens,
                args.device,
                args.fallback_device,
                args.cal_device,
                args.output_root,
                on_row=on_row,
            )
        except PreparationFailure as exc:
            # No size was attempted, so this model contributes no rows; the remaining
            # models are unaffected and still worth sweeping.
            logger.error("[context-sweep] %s: could not be prepared, no rows: %s", stem, exc)
    return 0


if __name__ == "__main__":
    sys.exit(main())

# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""What each verb DOES for the LLM family, once its door has parsed the command line.

`python -m model_axscript.llm` owns the flags; this module owns the behaviour. It declares
the verbs and the options the family honours, so the door can refuse anything else by name
without knowing what an LLM is.

Nothing here is per model: a model package declares its tag and its `make_config`, and every
step below reads the returned configuration. That is why seven models need no code between
the command line and `generate.py`.
"""

from __future__ import annotations

import dataclasses
import importlib
import logging
from pathlib import Path

from model_axscript import registry
from model_axscript.shared.artifact_contract import artifact_checkpoint, checkpoint_conflict
from model_axscript.shared.hardware import CORES_PER_GENERATION

VERBS = ("build", "run", "benchmark", "eval")
# Every option the front door declares except the media flags, which belong to the
# vision-language driver beside this one: a text model has no picture to be given.
OPTIONS = frozenset(
    {
        "output_dir",
        "device",
        "cal_device",
        "num_cal",
        "weights",
        "calibration",
        "seq_len",
        "interactive",
        "tasks",
        "num_eval_samples",
        "engine",
        "generation",
        "num_cores",
        "quantized_model",
        "save_quantized_model",
    }
)

# The input shape per verb, which the front door binds and refuses against. A prompt is
# optional because this family has a sensible default for one; an image never is.
INPUTS = {"run": ("prompt*",), "benchmark": ("prompt*",)}


# Defaults for --eval. wikitext is a rolling log-likelihood over plain text, so it needs no
# few-shot setup and stays meaningful at any compiled context length.
EVAL_TASKS = ("wikitext",)
# The two legs a score can be taken on: the fp reference, and the compiled model on the
# device. With neither named, --eval runs both and prints the gap between them.
EVAL_ENGINES = ("host", "aipu")


def default_output_dir(model_tag: str) -> Path:
    return Path("build") / model_tag / "axscript"


def _config(package, output_dir: Path, args, reused_checkpoint: str | None = None):
    """The model's own configuration, with the per-build overrides the caller named applied.

    Applied here rather than passed into `make_config`, so a model declares one factory and
    the overrides mean the same thing for every model in the family. `--seq-len` is the
    compiled capacity, which the artifact contract still refuses if the model has not
    measured it, so an unmeasured value fails by name rather than as a device hang.

    `reused_checkpoint` is the checkpoint an artifact that is NOT being rebuilt was compiled
    from (see `_reused_checkpoint`); given, it wins over the card's, because what is loaded
    is that artifact and not what the card would build today.
    """
    config = package.make_config(output_dir, cal_device=args.cal_device)
    overrides = {}
    if args.num_cal is not None:
        overrides["calib_budget"] = args.num_cal
    if reused_checkpoint or args.weights:
        # `repo_id` IS the weights source: everywhere in HuggingFace a hub id and a local
        # directory are the same call, and `checkpoint.repo_file` already takes either. So a
        # customer's own checkpoint is this flag, not an edit to the card. One branch for the
        # two, because `_reused_checkpoint` has already refused a pair that disagrees.
        overrides["repo_id"] = reused_checkpoint or args.weights
    if args.calibration is not None:
        # One source, the caller's, replacing the card's mix rather than joining it: a build
        # that quietly kept a share of WikiText would report scales the caller did not ask
        # for. The count comes from --num-cal, or the source's default budget.
        overrides["calib_sources"] = ((args.calibration, args.num_cal or 1024),)
    if args.seq_len is not None:
        overrides["context_len"] = args.seq_len
    if args.generation is not None:
        # Already validated in `run`, before anything was built.
        generation = str(args.generation).upper()
        overrides["generation"] = generation
        if args.num_cores is None:
            # The core count follows the generation, so the pair cannot be set half-way. A
            # model bakes in the count for the part it ships on, so overriding the generation
            # alone would otherwise build for Europa across an Omega-sized 4 cores and label
            # the artifact EUROPA -- a claim about the part that the binary does not meet.
            # `--num-cores` still wins when given, for a build that deliberately leaves cores
            # free for another workload.
            overrides["num_cores"] = CORES_PER_GENERATION[generation]
    if args.num_cores is not None:
        overrides["num_cores"] = args.num_cores
    return dataclasses.replace(config, **overrides) if overrides else config


def _reused_checkpoint(output_dir: Path, verb: str, weights) -> str | None:
    """The checkpoint an artifact this call will NOT rebuild was made from, or None.

    A `run`, `benchmark` or `eval` over an existing directory has to tokenize with the
    checkpoint that directory was compiled from: the card's default describes what a build
    WOULD use, and `--weights` is precisely the case where the two differ, so taking the
    config's would decode a customer's model with our vocabulary and report it as a result.
    None means there is nothing built here to read a checkpoint off, or nothing recorded it.

    A `--weights` that disagrees is refused rather than ignored, the same rule the VLM entry
    applies to a `--seq-len` against a built artifact: both are compiled in, and honouring
    the old artifact would hand back a number measured on weights nobody asked about. The
    read and the refusal text come from the artifact contract, which both families share.
    """
    # The same condition the caller below builds on, so the two cannot disagree about
    # whether this run compiles: `build` always does, and any verb does when nothing is there.
    if verb == "build" or not (output_dir / "model.json").is_file():
        return None
    conflict = checkpoint_conflict(output_dir, weights)
    if conflict:
        raise SystemExit(conflict)
    return artifact_checkpoint(output_dir)


def run(tag: str, verb: str, args) -> int:
    """Carry out `verb` for the model `tag` names. Returns a process exit status."""
    # The runtime logs the run banner (device / cores / compiled run size) at INFO; every
    # number printed below is only comparable against another at the same settings, so the
    # banner is part of the output, not debug chatter.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    # Refused before anything is imported or compiled -- the mapping is a leaf module, so
    # saying so costs no torch. `GenerationConfig` takes the name as given and the core count
    # is read back only when the artifact is loaded, so a typo would otherwise surface as a
    # KeyError after a multi-minute build.
    if args.generation is not None and str(args.generation).upper() not in CORES_PER_GENERATION:
        raise SystemExit(
            f"unsupported hardware generation {args.generation!r}; supported: "
            f"{', '.join(CORES_PER_GENERATION)}"
        )
    package = importlib.import_module(registry.package(tag))
    output_dir = args.output_dir or default_output_dir(tag)
    # Read before anything is imported or compiled, so a --weights that cannot be honoured
    # is refused by name rather than after a load.
    reused = _reused_checkpoint(output_dir, verb, args.weights)
    # Deferred so the front door can read VERBS / OPTIONS, and refuse a verb or a flag this
    # family does not have, without paying for torch and the compiler to say so.
    from model_axscript.llm.generate import (
        RuntimeModel,
        benchmark_generation,
        build_artifact,
        run_generation,
    )

    if args.interactive:
        # The REPL is reached rather than reimplemented: one chat loop however a user arrives.
        # Called as the MODULE, never through `llm/cli.py`: that dispatcher is an upstream file
        # we carry for resync parity and do not maintain, so nothing of ours may depend on it.
        from model_axscript.llm.evaluation.chat import main as chat

        # The REPL takes no device: it would pick any healthy one, so a run asked for a
        # NAMED device and silently measured another. Refused by name, per this front
        # door's rule that a flag it cannot honour is not accepted.
        if args.device:
            raise SystemExit(
                "--device is not honoured with --interactive: the chat REPL takes no device "
                "and would reserve whichever one is free. Drop --device, or run the verb "
                "that accepts it."
            )
        return chat(["--artifact-dir", str(output_dir)]) or 0

    # --eval never builds: the reference leg needs no artifact, and the compiled leg says so
    # by name when one is missing.
    if verb == "eval":
        if args.engine is not None and args.engine not in EVAL_ENGINES:
            raise SystemExit(
                f"--engine {args.engine!r} is not a leg to score; the legs are "
                f"{', '.join(EVAL_ENGINES)}"
            )
        _score(
            _config(package, output_dir, args, reused),
            output_dir,
            tasks=tuple(args.tasks) if args.tasks else EVAL_TASKS,
            num_eval_samples=args.num_eval_samples,
            engine=args.engine,
            device=args.device,
        )
        return 0

    # `build` is the expensive, writing action, so it is only ever taken when named or when
    # there is nothing to load: selecting a different --device never triggers a recompile.
    if verb == "build" or not (output_dir / "model.json").is_file():
        if args.quantized_model is not None:
            # `--quantized-model` loads a bundle that was already calibrated; these
            # flags only affect a fresh calibration pass, so honouring both would either
            # silently drop the calibration flags or contradict "reuse this bundle as-is".
            conflicting = [
                f"--{flag}"
                for flag, value in (
                    ("num-cal", args.num_cal),
                    ("calibration", args.calibration),
                )
                if value is not None
            ]
            if args.cal_device:
                conflicting.append("--cal-device")
            if conflicting:
                raise SystemExit(
                    f"{', '.join(conflicting)} has no effect with --quantized-model: "
                    "the quantized bundle is loaded as-is instead of recalibrated. Drop "
                    "--quantized-model to recalibrate, or drop the calibration flags "
                    "to reuse the bundle."
                )
        print(f"[build] {tag} -> {output_dir}")
        build_artifact(
            _config(package, output_dir, args),
            quantized_model=args.quantized_model,
            save_quantized_model=args.save_quantized_model,
        )
    if verb == "build":
        return 0

    # `load_tokenizer`, not `AutoTokenizer`: it owns the fallback for a checkpoint whose
    # `tokenizer_config.json` names a `tokenizer_class` the pinned transformers cannot
    # import (LFM2.5's `TokenizersBackend`), and it carries `model_input_names` through.
    # `AutoTokenizer.from_pretrained` RAISES for those cards, so a model that builds
    # would not run. Same reason the build and eval paths use it.
    from ..generate import load_tokenizer

    # Config first: the seed prompt is one of the things a model declares
    # (`GenerationConfig.prompt`), so it cannot be resolved before the config exists.
    config = _config(package, output_dir, args, reused)
    prompt = " ".join(args.inputs["prompt"]) or config.prompt
    tokenizer = load_tokenizer(config.repo_id)
    runtime = RuntimeModel(output_dir / "model.json", device=args.device)
    if verb == "run":
        text = ""
        for text in run_generation(runtime, tokenizer, output_dir, prompt, config):
            pass
        print(text)
        return 0
    metrics = benchmark_generation(runtime, tokenizer, output_dir, prompt, config)
    print(
        f"ttft_s {metrics['ttft_s']:.3f}  "
        f"decode_tok_per_s {metrics['decode_tok_per_s']:.2f}  "
        f"generated_tokens {metrics['tokens']}"
    )
    return 0


def _score(config, artifact_dir: Path, *, tasks, num_eval_samples, engine, device=None) -> None:
    """Score accuracy through the same driver the zoo uses, and print the rows.

    Deliberately not a local implementation. Both backends of a model score through this one
    driver, so a number taken here is comparable with the number taken by tag, and with the
    other backend of the same model. A second driver would make three numbers that agree
    only by coincidence.

    The zoo import is function-local: nothing else in this tree depends on it, so a missing
    install fails here, naming what to install, rather than breaking every other entry path.
    """
    try:
        from axelera.zoo.build.config import LlmEvalConfig
        from axelera.zoo.eval.llm_runner import axscript_target_for, run_llm_eval
    except ImportError as error:
        raise SystemExit(
            "--eval scores through the shared zoo eval driver; install `axelera-zoo[llm-eval]`."
        ) from error

    # The capacity comes from the artifact, the same source the driver reads for the other
    # backend, so an entry scores against what THIS directory was built at rather than what
    # the config declares; a context sweep compiles other lengths under the same config.
    result = run_llm_eval(
        axscript_target_for(config, artifact_dir, device=device),
        LlmEvalConfig(tasks=tuple(tasks)),
        engine_override=engine,
        num_eval_samples=num_eval_samples,
    )
    print(f'engines: {", ".join(result.engines)}   max_length: {result.max_length}')
    for row in result.rows:
        scores = "  ".join(f'{p}={row.scores.get(p, float("nan")):.4f}' for p in result.engines)
        delta = "" if row.delta is None else f"  delta={row.delta:+.4f}"
        print(f"  {row.task:<24} {row.metric:<20} {scores}{delta}")

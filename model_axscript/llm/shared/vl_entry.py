# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""The command line every vision-language model in this family shares.

The four verbs are the ones every other model in the tree answers to -- ``--build``,
``--run``, ``--benchmark``, ``--eval`` -- because `axzoo` and
``python -m model_axscript.llm`` drive a model through that set and neither knows or cares
which family it reached. A VLM that spelled its actions differently would be a model the
zoo can only build.

What differs from :mod:`model_axscript.llm.shared.entry` is only what is behind the verbs.
A text model's build is one call to ``generate.build_artifact``; a VLM compiles two towers
through its own driver, and its run takes a picture or a clip as well as a prompt. So this
parser owns the four verbs and the flags the tree standardizes (``--output-dir``,
``--device``, ``--cal-device``, the eval set) and forwards EVERY other flag straight to the
model's own run / benchmark CLI. That is deliberate: ``--image``, ``--video``, ``--frames``,
``--max-new-tokens`` and the sampler flags are documented by the driver that implements
them, and re-declaring them here is how two copies of one flag set drift apart.

Reached as ``python -m model_axscript.llm <verb> <tag>``. The models declare
``DRIVER = "vl_entry"`` and own no entry script, so these are the whole surface:

    python -m model_axscript.llm build <tag>                       # compile both towers
    python -m model_axscript.llm run <tag> "What is this?" --image photo.jpg
    python -m model_axscript.llm run <tag> --interactive            # /image /video /reset
    python -m model_axscript.llm benchmark <tag>
    python -m model_axscript.llm eval <tag> --tasks wikitext
"""

from __future__ import annotations

import argparse
import ast
import importlib
import logging
import os
import sys
import textwrap
from pathlib import Path

import model_axscript
from model_axscript import registry
from model_axscript.shared.artifact_contract import (
    checkpoint_conflict,
    validate_run_size,
    artifact_metadata,
    write_artifact_metadata,
)

from .vl_config import CORES_PER_GENERATION, geometry_label

#: Scored by default. wikitext is a rolling log-likelihood over plain text: no few-shot
#: setup, and meaningful at any compiled context length. The same default the text-only
#: family uses, so a VLM's decoder and a text model's are scored on one task list.
#: The front door reads these three off whichever driver a model declares, before anything
#: runs, to narrow `--help`, refuse a verb by name and bind the positional input.
VERBS = ("build", "run", "benchmark", "eval")
#: Every shared flag this entry honours, under the SAME names the front door declares. A
#: quantity that means one thing is spelled one way the whole distance, so `_argv` forwards
#: rather than translates and two spellings of one knob cannot drift apart. None of them
#: bypasses the run-size gate: an unmeasured context is still refused by `validate_run_size`
#: at build time, by name, rather than hanging the device.
OPTIONS = frozenset(
    {
        "output_dir",
        "device",
        "cal_device",
        "interactive",
        "tasks",
        "num_eval_samples",
        "engine",
        "image",
        "video",
        "frames",
        "seq_len",
        "num_cal",
        "weights",
        "generation",
        "num_cores",
        "quantized_model",
        "max_new_tokens",
        "greedy",
        "temperature",
        "top_p",
        "top_k",
        "min_p",
        "repetition_penalty",
    }
)
#: Flags only one driver here has (`--tiles`, `--no-stream`) fit no shared surface, so the
#: front door hands over what it did not recognise and `_argv` passes it on. Read by
#: `__main__.main`, which refuses unknown flags for a driver that declares nothing.
FORWARDS_UNKNOWN = True

#: The two forwarded flags that are REFUSED rather than passed on, and what to use instead.
#: Both name which artifact a run is taken from, and this entry has already answered that:
#: `_forward` appends the passthrough AFTER its own `--artifact-dir`, and argparse keeps the
#: last occurrence, so a forwarded copy silently retargets the run away from `--output-dir`.
#: `--repo-id` is the same question one level down -- it points the config, processor and
#: tokenizer at a checkpoint the artifact was not built from. Both are ROW-visible: a
#: benchmark labelled with one artifact would have been run on another.
RESOLVED_HERE = {
    "--artifact-dir": "--output-dir names the directory, and this entry passes it down",
    "--repo-id": "a run reads the checkpoint its artifact was built from, off the artifact",
}

#: The prompt stays positional, as it is for the text family; the picture or clip arrives by
#: flag OR as a positional alongside the prompt. Both are optional: with neither, the model
#: runs its own built-in tour.
INPUTS = {"run": ("prompt*",), "benchmark": ("prompt*",)}

#: A positional that names one of these is media, not a word of the prompt. `axzoo run`
#: forwards its INPUT positional -- documented as "image, video or stream to run over" --
#: into `prompt*`, so without this a VLM asked about a picture answers about the built-in
#: tour images instead and nothing says the picture was dropped.
#:
#: Classified by suffix ALONE, deliberately: a path that does not exist has to reach
#: `open_media` to be refused by name. Demoting it to prompt text here would restore the
#: silent-wrong-answer this exists to prevent.
_VIDEO_SUFFIXES = (".mp4", ".webm", ".mkv", ".avi", ".mov", ".m4v", ".gif")
_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff", ".ppm")


def _split_media(words: list[str]) -> tuple[list[str], list[str], str]:
    """Positional input -> (pictures, clips, prompt): what is a file, and what is the question."""
    images, videos, prompt = [], [], []
    for word in words:
        suffix = Path(word.split("?")[0]).suffix.lower()  # a URL's query is not its suffix
        if suffix in _IMAGE_SUFFIXES:
            images.append(word)
        elif suffix in _VIDEO_SUFFIXES:
            videos.append(word)
        else:
            prompt.append(word)
    return images, videos, " ".join(prompt)


#: Scored when `--tasks` names nothing: BOTH towers. `docvqa` is the vision-language metric
#: this model is judged by upstream and the only one that can see a vision tower that
#: quantized badly; `wikitext` is the decoder's rolling log-likelihood, and the row that
#: lines this model up with every text model in the tree. A default of wikitext alone let a
#: VLM report a clean score while half of it was never run.
EVAL_TASKS = ("docvqa", "wikitext")
#: The two legs a score can be taken on, named by WHERE they run, as the zoo's eval runner
#: and `entry.py` name them. One vocabulary end to end: what `--pipe` takes is what
#: `engine_override` accepts and what the printed rows are labelled with, so nothing has to
#: be translated and a VLM's rows line up with every other LLM's in a report.
EVAL_ENGINES = ("host", "aipu")


def default_output_dir(model_tag: str) -> Path:
    """Where a build lands when the caller names no directory.

    The same layout `axzoo` resolves (`build/<tag>/<backend>`), so a model built by hand and
    the same model built by tag are one artifact rather than two.
    """
    return Path("build") / model_tag / "axscript"


def _declared(config, run_module: str | None = None, actions: bool = True) -> str:
    """The configuration this model is released at, read from its own config.

    A customer should not have to open the build config to find out what is supported, and a
    hand-written list here would drift from the set the build enforces. Both run sizes are
    printed because a VLM has two: the decoder's context and the vision tower's patch grid,
    each baked at build time and each gated by ``validate_run_size``.

    `actions` describes THIS parser's spelling, so the front door drops it: there they are
    verbs (`run <tag>`), and `--run` would document a command line nobody is typing.
    """
    contexts = ", ".join(str(n) for n in config.supported_context_lens)
    geometries = ", ".join(geometry_label(s) for s in config.supported_image_sizes)
    return (
        (
            "Every action is named: --build compiles (minutes, the only writing action),\n"
            "--run and --benchmark load the artifact and build first only if it is\n"
            "missing. No action prints this help.\n\n"
            if actions
            else ""
        )
        + f"Validated context lengths: {contexts}. Validated image geometries: {geometries}.\n"
        "Other values are refused at build time - they have not been measured on chip, and an\n"
        "unmeasured size fails as a device hang rather than an error.\n"
        # Wrapped rather than hand-broken: the sentence interpolates a geometry, a context and
        # a core count whose widths differ per model, so fixed line breaks read raggedly on
        # every model but the one they were laid out for.
        + textwrap.fill(
            f"Shipped build: {geometry_label(config.image_size)} at {config.context_len} tokens"
            f" on {config.num_cores} AI core(s) of {config.generation}. --seq-len,"
            " --generation and --num-cores pick otherwise, within the sets above. A build"
            " reserves one sub-device per core, so it needs a part with at least that many,"
            " and it must hold BOTH towers at once - they are resident together for every"
            " prompt.",
            width=88,
        )
        + "\n\n"
        + _passthrough_note(run_module)
    )


def _passthrough_note(run_module: str | None) -> str:
    """Where to look up the flags this parser forwards rather than declares.

    The module is DERIVED from the run entry the model handed in, never hardcoded: the
    media / prompt / sampler flags are documented by whichever driver implements them, and
    this family has more than one architecture. A fixed name here would send every model's
    user to one model's --help.
    """
    note = (
        "Media, prompt, token-budget and sampler flags are forwarded to the model's own\n"
        "run / benchmark CLI, which declares them. Flags listed above are declared here; any\n"
        "other flag is handed to that CLI unchanged, so this model's own are typed the same\n"
        "way -- they differ per driver, and a typo among them is refused before any build"
    )
    return (
        f"{note}; run\n`python -m {run_module} --help` to see them." if run_module else f"{note}."
    )


def _run_module(run) -> str | None:
    """The importable module whose ``main`` a ``--help`` should send the reader to.

    Declared by the entry package as ``RUN_MODULE``, never derived from ``run`` itself:
    ``run`` is a forwarder defined IN that package, so its ``__module__`` is the package --
    which has no ``__main__.py`` and so is not runnable with ``-m`` at all.
    """
    package = sys.modules.get(getattr(run, "__module__", "") or "")
    return getattr(package, "RUN_MODULE", None)


def _parser(
    model_tag: str, config, display_name: str, run_module: str | None = None
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"python -m model_axscript.llm <verb> {model_tag}",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=_declared(config, run_module),
        description=f"Build, then run, {display_name} (axscript).",
    )
    parser.add_argument(
        "--build",
        action="store_true",
        help="compile both towers (the only action that writes; takes minutes)",
    )
    parser.add_argument(
        "--run",
        nargs="?",
        const="",
        default=None,
        metavar="PROMPT",
        help="answer PROMPT about the attached --image / --video (builds first only if the "
        "artifact is missing). With no PROMPT and no media, the model's own demo runs.",
    )
    parser.add_argument(
        "--benchmark",
        nargs="?",
        const="",
        default=None,
        metavar="PROMPT",
        help="vision fps + ttft + decode tok/s (builds first only if the artifact is missing)",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--seq-len",
        type=int,
        default=None,
        metavar="N",
        help=f"compile the decoder's KV cache at N tokens (default {config.context_len}); must "
        "be one of the validated lengths listed below. Baked into the artifact, so changing it "
        "is a rebuild and not a run-time switch.",
    )
    parser.add_argument(
        "--generation",
        choices=tuple(CORES_PER_GENERATION),
        default=None,
        help=f"hardware generation the artifact targets (default {config.generation}). Also "
        "sets the core count unless --num-cores says otherwise.",
    )
    parser.add_argument(
        "--num-cores",
        type=int,
        default=None,
        metavar="N",
        help="AI cores both towers shard over; default is the whole part for the chosen "
        f"generation ({', '.join(f'{g}={n}' for g, n in CORES_PER_GENERATION.items())}). A build "
        "reserves one sub-device per core and BOTH towers are resident at once, so this is a "
        "hardware requirement of the artifact and not only a speed.",
    )
    parser.add_argument(
        "--num-cal",
        type=int,
        default=None,
        metavar="N",
        help=f"DocVQA entries to calibrate on (default {config.num_calibration_samples}). "
        "Calibration is the bulk of a build, so a small N is how you get a runnable artifact "
        "quickly -- at the cost of a worse-quantized model. For exercising the pipeline, not "
        "for producing a number.",
    )
    parser.add_argument(
        "--weights",
        default=None,
        metavar="DIR-OR-ID",
        help="build both towers from your own checkpoint instead of the card's: a local "
        f"HuggingFace directory or a hub id (default {config.repo_id}). A build parameter, "
        "so a run reads the checkpoint off the artifact rather than taking this again.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Metis device to run on: name (e.g. metis-0:2d:0) or index; default is the first "
        "device, or $AXELERA_METIS_DEVICE",
    )
    parser.add_argument(
        "--cal-device",
        default="",
        help="torch device for build-time calibration: 'cpu' / 'cuda' (default auto-detects a GPU)",
    )
    parser.add_argument(
        "--eval",
        action="store_true",
        help="score lm-eval accuracy tasks on the decoder tower (both legs by default; see --engine)",
    )
    parser.add_argument(
        "--tasks", nargs="+", default=None, help=f"lm-eval tasks to score (default {EVAL_TASKS})"
    )
    parser.add_argument(
        "--num-eval-samples", type=int, default=None, help="per-task sample cap for --eval"
    )
    parser.add_argument(
        "--engine",
        choices=EVAL_ENGINES,
        default=None,
        help="score one leg only: host (fp reference) or aipu (the compiled model); "
        "default scores both and reports the gap",
    )
    return parser


def build(config, compile_towers) -> dict:
    """Compile both towers into ``config.artifact_dir`` and record what was built.

    ``compile_towers`` is the model's own two-tower builder (``pipeline.build_artifacts``),
    handed in by its entry rather than looked up here.

    The two run sizes are checked BEFORE the compile, not after: an unmeasured context or
    geometry does not fail at compile time, it fails on chip minutes later as a hang, so the
    declared set is the gate.

    ``metadata.json`` is written last, at the artifact ROOT, because it describes the whole
    build rather than one tower. It is the seam every zoo verb keys on: `eval` bounds the
    scored sequence with it and `benchmark` labels its row with it, and each RAISES rather
    than falling back to the config -- a config states intent, and a sweep or an override can
    compile a different length into the same shape of directory.
    """
    validate_run_size(config.context_len, tuple(config.supported_context_lens), "context_len")
    validate_run_size(tuple(config.image_size), tuple(config.supported_image_sizes), "image_size")
    built = compile_towers(
        config.repo_id,
        config.artifact_dir,
        num_samples=config.num_calibration_samples,
        image_size=tuple(config.image_size),
        fit_mode=config.fit_mode,
        context_len=config.context_len,
        target=config.target,
        num_cores=config.num_cores,
        generation=config.generation,
        smooth_alpha=config.smooth_alpha,
        qk_alpha=config.qk_alpha,
        device=config.cal_device,
    )
    # The calibration count and the geometry are recorded because THIS build knows them and
    # no later reader can recover them: the DocVQA eval scores rows the build did not
    # calibrate on, and how many rows that is belongs to the directory rather than to
    # whatever config the reader happens to hold. Same doctrine as `compiled_seq_lens`.
    write_artifact_metadata(
        config.artifact_dir,
        compiled_seq_lens=[config.context_len],
        model_name=config.repo_id,
        num_calibration_samples=config.num_calibration_samples,
        image_size=tuple(config.image_size),
    )
    return built


def _built_at(output_dir: Path) -> int | None:
    """The context length ``output_dir`` was actually compiled at, or None if nothing is.

    Read from ``metadata.json`` rather than from the config, for the reason the file exists:
    a config states what a build WOULD use, and `--seq-len` is precisely the case where
    the two differ. None means no finished build to contradict the request -- the same file
    the caller above treats as the "already built" marker, so the two cannot disagree about
    whether there is an artifact here.
    """
    lengths = artifact_metadata(output_dir).get("compiled_seq_lens") or []
    return int(lengths[0]) if len(lengths) == 1 else None


def _forward(action_main, output_dir: Path, prompt: str, passthrough: list[str]) -> int:
    """Drive the model's own run / benchmark CLI, which owns the flags and the timing.

    The measurement is NOT reimplemented here. The driver's numbers are the ones its README
    quotes and the ones the zoo's benchmark writer reshapes, so a third timing loop in this
    file would produce a fourth number nobody could line up with the other three.

    `--artifact-dir` is FIRST and the passthrough follows it, which is last-wins order: the
    flags that would win it back are refused in `main` (`RESOLVED_HERE`) rather than ordered
    around here, so nothing depends on where this appends.
    """
    argv = ["--artifact-dir", str(output_dir), *passthrough]
    if prompt and "--prompt" not in passthrough:
        argv += ["--prompt", prompt]
    return action_main(argv) or 0


def _argv(verb: str, args) -> list[str]:
    """One front-door verb, spelled as this entry's own flags.

    The translation is here rather than in the front door because these flags are this
    family's: the front door declares a verb and a shared flag surface, and each driver
    says what that means for it.
    """
    positional_images, positional_videos, prompt = _split_media(
        list(args.inputs.get("prompt", []) or [])
    )
    argv: list[str] = {
        "build": ["--build"],
        "run": ["--run", prompt],
        "benchmark": ["--benchmark", prompt],
        "eval": ["--eval"],
    }[verb]
    if args.output_dir is not None:
        argv += ["--output-dir", str(args.output_dir)]
    if args.device is not None:
        argv += ["--device", str(args.device)]
    if args.cal_device:
        argv += ["--cal-device", str(args.cal_device)]
    if getattr(args, "tasks", None):
        argv += ["--tasks", *args.tasks]
    if getattr(args, "num_eval_samples", None) is not None:
        argv += ["--num-eval-samples", str(args.num_eval_samples)]
    if getattr(args, "generation", None) is not None:
        argv += ["--generation", str(args.generation)]
    if getattr(args, "num_cores", None) is not None:
        argv += ["--num-cores", str(args.num_cores)]
    # The front door's names for the two build knobs this entry spells differently.
    if getattr(args, "seq_len", None) is not None:
        argv += ["--seq-len", str(args.seq_len)]
    if getattr(args, "num_cal", None) is not None:
        argv += ["--num-cal", str(args.num_cal)]
    if getattr(args, "weights", None) is not None:
        argv += ["--weights", str(args.weights)]
    # A positional picture and `--image` are the same request spelled two ways, so they add
    # up rather than one silently winning. Positional CLIPS go down the same channel: the
    # driver concatenates `--image` and `--video` into one source list that `open_media`
    # classifies by suffix, so a clip named positionally is still a clip, and two of them,
    # or a clip mixed with a picture, is refused there, by name, where `--video` (one value,
    # last wins) would have dropped one without saying so.
    sources = [str(one) for one in getattr(args, "image", None) or []]
    sources += positional_images + positional_videos
    if sources:
        argv += ["--image", *sources]
    if getattr(args, "video", None) is not None:
        argv += ["--video", str(args.video)]
    if getattr(args, "frames", None) is not None:
        argv += ["--frames", str(args.frames)]
    if getattr(args, "engine", None) is not None:
        # Passed straight through: one flag name and one pair of values from the front door
        # to the zoo's runner, so there is nothing to map and nothing to keep in step.
        argv += ["--engine", str(args.engine)]
    # `--interactive` is this model's OWN session (the turn carries the attached picture or
    # clip, and /image /video swap it), not the text family's chat REPL over one artifact.
    if getattr(args, "interactive", False):
        argv.append("--interactive")
    # Forwarded ONLY when typed: a default carried here would overrule the driver's own.
    # `main` declares none of them, so they reach the model the way `--image` does --
    # `parse_known_args` into `passthrough`, then `_forward`.
    for dest, flag in (
        ("max_new_tokens", "--max-new-tokens"),
        ("temperature", "--temperature"),
        ("top_p", "--top-p"),
        ("top_k", "--top-k"),
        ("min_p", "--min-p"),
        ("repetition_penalty", "--repetition-penalty"),
    ):
        value = getattr(args, dest, None)
        if value is not None:
            argv += [flag, str(value)]
    if getattr(args, "greedy", False):
        argv.append("--greedy")
    # Whatever the front door did not recognise, unread: naming the driver-specific flags
    # here would be a second copy of each driver's list. `_validate_extra` has vetted them
    # for typos, and `main` refuses the `RESOLVED_HERE` pair among them before it builds.
    argv += list(getattr(args, "extra", []) or [])
    return argv


def describe(tag: str) -> str:
    """This model's declared configuration, for the front door's `--help`.

    Read off the config the build enforces, so the sizes shown are the ones it refuses against.
    """
    package = importlib.import_module(registry.package(tag))
    config = package.make_config(default_output_dir(tag))
    return _declared(config, _run_module(package.run), actions=False)


def _declared_options(source_path: Path) -> frozenset[str]:
    """Every option string a module hands to `add_argument`, read rather than imported.

    Same bargain as `registry._declared`: vetting a flag name must not cost a compile.
    Importing `RUN_MODULE` pulls `llm/luts.py` and with it `axelera.compiler`, which the
    framework tox lane does not provide. Option strings built at runtime rather than written
    as literals are not seen, so the caller treats an empty result as nothing to check.
    """
    try:
        tree = ast.parse(source_path.read_text())
    except (OSError, SyntaxError):
        return frozenset()
    return frozenset(
        argument.value
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        for argument in node.args
        if isinstance(argument, ast.Constant)
        and isinstance(argument.value, str)
        and argument.value.startswith("-")
    )


def _validate_extra(package, verb: str, extra: list[str]) -> None:
    """Refuse a typo among the forwarded flags BEFORE anything compiles.

    `main` builds when no artifact is there, so a flag handed straight down would fail on the
    far side of a multi-minute compile. Judged against the option strings the run CLI named by
    `RUN_MODULE` declares; a package declaring none is simply not checked, and nor is
    `benchmark`.
    """
    if not extra or verb != "run":
        return
    module_name = getattr(package, "RUN_MODULE", None)
    if not module_name:
        return
    top, *address = module_name.split(".")
    if top != model_axscript.__name__ or not address:
        return
    # Located from the top-level package, which imports nothing. Importing the run module's
    # OWN package is no cheaper than importing the module: `models/<arch>/__init__.py`
    # registers the architecture, which pulls the compiler in.
    root = Path(model_axscript.__file__).resolve().parent
    source_path = root.joinpath(*address).with_suffix(".py")
    # argparse installs these itself, so they are never literals in the source.
    known = _declared_options(source_path) | {"-h", "--help"}
    if known == {"-h", "--help"}:
        return
    unknown = [
        token for token in extra if token.startswith("-") and token.split("=")[0] not in known
    ]
    if unknown:
        # Exit 2, as `parser.error` does everywhere else: this is a usage error.
        print(
            f"error: unrecognized arguments: {' '.join(unknown)}; "
            f"`python -m {module_name} --help` lists the flags this model's run CLI takes",
            file=sys.stderr,
        )
        raise SystemExit(2)


def run(tag: str, verb: str, args) -> int:
    """Carry out one door verb for a two-tower model. The family-driver seam.

    Reached from `llm/shared/entry.py` for a model that declares `DRIVER`, so a VLM answers
    `python -m model_axscript.llm <verb> <tag>` like every other model in the tree, with no entry
    script of its own. The three callables come off the model package, which declares them.
    """
    package = importlib.import_module(registry.package(tag))

    # Loading from a pre-quantized bundle does not work for VLMs yet: vision and text are
    # calibrated in one joint pass (the decoder's activation scales must see real
    # vision-tower output), and the vision tower compiles from the live calibrated module,
    # not from the exported bundle. So there is no reuse path that would skip that pass.
    if args.quantized_model:
        raise SystemExit(f"--quantized-model is not supported for {package.DISPLAY_NAME} yet.")
    _validate_extra(package, verb, list(getattr(args, "extra", []) or []))
    return main(
        package.MODEL_TAG,
        package.make_config,
        package.DISPLAY_NAME,
        compile_towers=package.compile_towers,
        run=package.run,
        benchmark=package.benchmark,
        argv=_argv(verb, args),
    )


def main(
    model_tag: str,
    make_config,
    display_name: str,
    *,
    compile_towers,
    run,
    benchmark,
    argv=None,
) -> int:
    """Run one VLM's entry, against the three entry points its driver implements.

    The three are passed in rather than looked up by module name, so the coupling is written
    down in the model's own ``__main__.py``: this file owns the family's command line, and
    the family's second architecture supplies its own three without editing it.
    """
    # The runtime logs the run banner (device / cores / compiled run size) at INFO; every
    # number this prints is comparable only with another taken at the same settings, so the
    # banner is part of the output rather than debug chatter.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = _parser(
        model_tag,
        make_config(default_output_dir(model_tag)),
        display_name,
        _run_module(run),
    )
    args, passthrough = parser.parse_known_args(argv)
    # Wrong the moment it is typed, so it is refused before anything is resolved or compiled.
    # Printed rather than `parser.error`, which would append THIS parser's usage: reached
    # through the front door the actions are verbs (`run <tag>`), not the `--run` spelled here.
    refused = sorted({token.split("=")[0] for token in passthrough} & RESOLVED_HERE.keys())
    if refused:
        for flag in refused:
            print(f"error: {flag} cannot be forwarded: {RESOLVED_HERE[flag]}", file=sys.stderr)
        raise SystemExit(2)  # a usage error, as `_validate_extra` reports one

    # No action requested: show the commands rather than starting a multi-minute compile.
    # Building is the expensive, writing action, so it is named explicitly.
    if not (args.build or args.run is not None or args.benchmark is not None or args.eval):
        parser.print_help()
        return 0

    output_dir = args.output_dir or default_output_dir(model_tag)
    # Only passed on when the caller named one, so a model's own default stays the single
    # place the shipped context length is written down.
    overrides = {
        name: value
        for name, value in (
            ("context_len", args.seq_len),
            ("generation", args.generation),
            ("num_cores", args.num_cores),
            ("num_calibration_samples", args.num_cal),
            # `repo_id` IS the weights source: a hub id and a local directory are the same
            # argument everywhere in HuggingFace, and `checkpoint.repo_file` plus the
            # processor loads take either. So a customer's own checkpoint is this flag,
            # not an edit to the card, and both towers plus the calibration follow it.
            ("repo_id", args.weights),
        )
        if value is not None
    }

    if args.device is not None:
        # ONE mechanism for this flag, for every verb this entry has. `run` and `benchmark`
        # are forwarded to the family's own main, which parses its own argv and declares no
        # device flag, so the environment is the only channel that reaches them; `--eval`
        # then reads the same variable through `_select_device`, rather than this file
        # carrying a second way to say the same thing. Selecting a part never recompiles:
        # the artifact is device-agnostic. (The zoo's own eval passes the part as an
        # argument instead, because it sets no environment.)
        os.environ["AXELERA_METIS_DEVICE"] = str(args.device)

    if args.eval:
        # --eval never builds: the reference leg needs no artifact, and the compiled leg says
        # so by name when one is missing.
        _score(
            make_config(output_dir, **overrides),
            output_dir,
            tasks=tuple(args.tasks) if args.tasks else EVAL_TASKS,
            num_eval_samples=args.num_eval_samples,
            engine=args.engine,
        )
        return 0

    run_or_bench = args.run is not None or args.benchmark is not None
    # The marker of "already built" is metadata.json at the artifact ROOT, which `build`
    # writes LAST -- after both towers have compiled. Nothing inside a tower can stand in
    # for it: the decoder's own model.json is written before the vision compile even starts,
    # so a build that died in the vision tower would leave a directory every later --run
    # took for finished and then failed to load.
    have_artifact = (output_dir / "metadata.json").is_file()
    # Context length is COMPILED IN, so an override that disagrees with what is already in
    # output_dir is a request to rebuild, not a run-time setting. Saying so beats running the
    # old artifact under the new number, which would misreport every measurement taken from it.
    # Only an EXPLICIT --seq-len can conflict: with none given the caller has asked for
    # whatever is there, and the model's own default is not a claim about an existing build.
    already = _built_at(output_dir)
    stale = args.seq_len is not None and already is not None and already != args.seq_len
    # Weights are compiled in as hard as the context length, and `build` records which
    # checkpoint they came from, so a disagreement is comparable the way `stale` is. The
    # comparison and its message live in the artifact contract, so this family and the text
    # one refuse it in the same words.
    other_weights = checkpoint_conflict(output_dir, args.weights)
    # The other three build knobs are compiled in just as hard -- `compile_towers` takes
    # num_samples, num_cores and generation -- but metadata.json records only the compiled
    # length, so there is nothing to compare them against the way `stale` compares --seq-len.
    # Unable to check and unable to honour, this refuses them BY NAME rather than computing
    # `overrides` and dropping it: a caller who asks for a 3-sample recalibration and is
    # handed the old artifact plus a success has been told something untrue. Same rule as the
    # front door's --device-with---interactive refusal: a flag it cannot honour is not
    # accepted. When metadata.json grows these three, they join `stale` and this goes away.
    unhonoured = [
        flag
        for flag, value in (
            ("--num-cal", args.num_cal),
            ("--num-cores", args.num_cores),
            ("--generation", args.generation),
        )
        if value is not None
    ]
    # --build compiles, whatever is already there and whichever other verb it is paired
    # with. It is the flag that ASKS for a compile, so honouring the old artifact instead --
    # built at another --num-cal, another geometry, another core count -- would hand
    # back something nobody asked for and report success.
    if args.build or not have_artifact:
        print(f"[build] {model_tag} -> {output_dir}")
        config = make_config(output_dir, cal_device=args.cal_device, **overrides)
        built = build(config, compile_towers)
        print(
            f"  vision  {built['vision']}\n  text    {built['text']}\n  bundle  {built['bundle']}"
        )
    elif stale:
        raise SystemExit(
            f"{output_dir} was compiled at context_len={already}, not "
            f"{args.seq_len}. Add --build to recompile it there, or pass --output-dir to "
            "keep both."
        )
    elif other_weights:
        raise SystemExit(other_weights)
    elif unhonoured:
        raise SystemExit(
            f"cannot honour {', '.join(unhonoured)} against the artifact already in "
            f"{output_dir}: compiled in. Add --build to recompile it there, or pass "
            "--output-dir to keep both."
        )
    if not run_or_bench:
        return 0

    status = 0
    if args.run is not None:
        status = _forward(run, output_dir, args.run, passthrough)
    if args.benchmark is not None and status == 0:
        status = _forward(benchmark, output_dir, args.benchmark, passthrough)
    return status


def _score(config, artifact_dir: Path, *, tasks, num_eval_samples, engine, device=None) -> None:
    """Score accuracy through the same driver the zoo uses, and print the rows.

    Deliberately not a local implementation, and deliberately not a VLM-specific one. What is
    scored is the DECODER tower -- the adapter resolves it out of the multi-tower artifact --
    on log-likelihood tasks, which is the one accuracy path every LLM in this tree takes. A
    second harness for this model would produce a perplexity comparable with nothing, which
    is the failure `tests/axscript/test_llm_eval_contract.py` exists to prevent.

    The log-likelihood tasks are not the whole model's accuracy: a VLM's own task metric is a
    vision-language one, and `docvqa` is scored HERE too, through the zoo's vision-language
    runner. Which runner a task goes to is decided by the task name, so a caller writes
    `--tasks docvqa wikitext` and gets both towers measured without naming a second flag.

    The zoo import is function-local: nothing else in this tree depends on it, so a missing
    install fails here, naming what to install, rather than breaking every other entry path.
    """
    try:
        from axelera.zoo.build.config import VLM_TASKS, LlmEvalConfig, VlmEvalConfig
        from axelera.zoo.eval.llm_runner import axscript_target_for, run_llm_eval
        from axelera.zoo.eval.vlm_runner import run_vlm_eval, vlm_target_for
    except ImportError as error:
        raise SystemExit(
            "--eval scores through the shared zoo eval driver; install `axelera-zoo[llm-eval]`."
        ) from error

    vision_tasks = tuple(task for task in tasks if task in VLM_TASKS)
    text_tasks = tuple(task for task in tasks if task not in VLM_TASKS)
    if vision_tasks:
        vision = run_vlm_eval(
            vlm_target_for(config, artifact_dir),
            VlmEvalConfig(tasks=vision_tasks),
            engine_override=engine,
            num_eval_samples=num_eval_samples,
        )
        _print_rows(vision)
    if not text_tasks:
        return
    result = run_llm_eval(
        axscript_target_for(config, artifact_dir),
        LlmEvalConfig(tasks=text_tasks),
        engine_override=engine,
        num_eval_samples=num_eval_samples,
    )
    _print_rows(result)


def _print_rows(result) -> None:
    """Print one eval's drift rows.

    These rows sit beside every other LLM's in a zoo report, and a leg has to be called the
    same thing across them or the numbers cannot be lined up. Identical to `entry.py`'s
    block, and shared by both runners because `run_vlm_eval` returns the same result type.
    """
    print(f'engines: {", ".join(result.engines)}   max_length: {result.max_length}')
    for row in result.rows:
        scores = "  ".join(f'{p}={row.scores.get(p, float("nan")):.4f}' for p in result.engines)
        delta = "" if row.delta is None else f"  delta={row.delta:+.4f}"
        print(f"  {row.task:<24} {row.metric:<20} {scores}{delta}")

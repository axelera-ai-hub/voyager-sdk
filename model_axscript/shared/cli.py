# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""The command line every family door shares: one flag surface, one family per door.

    python -m model_axscript.vision build     dinov2-vits14
    python -m model_axscript.vision run       dinov2-vits14 cat.jpg
    python -m model_axscript.llm    run       qwen3-0.6b "The capital of France is"
    python -m model_axscript.llm    eval      qwen3-0.6b --tasks wikitext
    python -m model_axscript.llm    list

A door per family, because one verb set spanning every family says less than it looks like
it says. `build` reaches a different compiler for a picture model than for a decoder, and
`run <tag> cat.jpg` binds a photograph for one family and a sentence for the other. At a
door, `--help` offers the flags that apply to the models behind it and no others.

The MACHINERY stays here, in one file, for the property the doors exist to keep. Every flag
in the tree is declared once, in `_FLAGS`, and a family opts in by naming dests in its
driver's OPTIONS, so a flag cannot exist on six models and be missing from the seventh. The
rules that cross flags live here for the same reason: `--frames` needs `--video`, a sampler
flag shapes one answer so it is `run` only, and `--generation` / `--num-cores` are compiled
in so `eval`, which never builds, refuses them. Copied into each door those rules would
drift, and a rule that holds at one door and not the other is the failure a user cannot see.

A model package declares its configuration and the seams a driver calls, never a parser of
its own. Each family's driver (`<family>/shared/entry.py`, plus any a model names through
`DRIVER`) names the verbs, the options and the INPUT SHAPE that family honours, and anything
else is refused BY NAME rather than accepted and ignored. The input shape is what lets the
families disagree about their positionals without a driver guessing: vision names one or
more images, an LLM names an optional prompt, and a family that wants an image and then a
prompt declares both.

`--help` narrows a second time as soon as the tag is typed, from the family's whole surface
to that one model's, so a card's own flags and its own input shape are what a user reads.
"""

from __future__ import annotations

import argparse
import importlib
from pathlib import Path
import re
import sys

from model_axscript import registry

#: The tree root, which is also the directory a family sits directly below.
_ROOT = Path(__file__).resolve().parents[1]

#: Every verb any family in the tree answers to. A door offers the subset its own drivers
#: declare; this is the vocabulary, so a verb one family lacks can be refused by name while
#: a word no family knows is still reported as the unknown token it is.
_VERBS = ("build", "run", "benchmark", "eval")

# Every flag in the tree, declared once: dest -> (the flag a user types, its parser options).
# A family opts in by naming the dest in its driver's OPTIONS.
_FLAGS: dict[str, tuple[str, dict]] = {
    "output_dir": (
        "--output-dir",
        dict(
            type=Path,
            default=None,
            help="artifact directory (default build/<tag>/axscript, the layout axzoo resolves)",
        ),
    ),
    "device": (
        "--device",
        dict(
            default=None,
            help="Metis device to run on: name (e.g. metis-0:2d:0) or index; default is the first",
        ),
    ),
    "cal_device": (
        "--cal-device",
        dict(
            default="",
            help="torch device for build-time calibration: 'cpu' / 'cuda' (default auto-detects a GPU)",
        ),
    ),
    # `--num-cal` and `--seq-len` are spelled as `axzoo build` spells them, and mean the same
    # thing, because a customer reads a generated recipe and a hand-authored one side by side.
    "num_cal": (
        "--num-cal",
        dict(
            type=int,
            default=None,
            metavar="N",
            help="calibration sample count for this build only (default: the model's own)",
        ),
    ),
    "seq_len": (
        "--seq-len",
        dict(
            type=int,
            default=None,
            metavar="N",
            help="compile sequence length for this build only, where the family has one",
        ),
    ),
    # Build from your OWN weights. Each names a file or a directory you trained, and each is
    # read only when a build happens, so passing one to `run` compiles rather than reloading.
    # `--weights` rather than `--repo-id` because the value is a directory as often as it is
    # a hub id, and rather than `--backbone` because for a classification fine-tune the head
    # is in the same checkpoint: it names what is being replaced, not what is inside it.
    "weights": (
        "--weights",
        dict(
            default=None,
            metavar="ID_OR_DIR",
            help="weights to build from: a Hugging Face repo id, or a local directory in "
            "checkpoint layout, a config file beside the weights. A bare state dict is not "
            "one, and is refused naming the config file it could not find "
            "(default: the weights the model ships)",
        ),
    ),
    "head": (
        "--head",
        dict(
            default=None,
            metavar="PATH",
            help="your own output head weights, where the model has a head of its own. This "
            "one IS a bare state dict of the head's layers, in the published UNFOLDED form, "
            "not a head already collapsed into a single matrix. The class count is read from "
            "it, so a head predicting a different number of classes needs nothing else "
            "(default: the head the model ships)",
        ),
    ),
    "calibration": (
        "--calibration",
        dict(
            default=None,
            metavar="DIR_OR_ID",
            help="calibration data to fit the int8 scales and the activation ranges on: a "
            "local directory or a Hugging Face dataset id. Match it to the data you will "
            "deploy on, because out-of-domain calibration does not error, it quietly costs "
            "accuracy (default: the model's own calibration set)",
        ),
    ),
    "interactive": (
        "--interactive",
        dict(
            action="store_true",
            help="`run` only: a multi-turn session, where the family has one",
        ),
    ),
    "quantized_model": (
        "--quantized-model",
        dict(
            type=Path,
            default=None,
            metavar="DIR",
            help="`build` only: reuse a calibrated `quantized_model/` dir from a prior build "
            "instead of recalibrating",
        ),
    ),
    "save_quantized_model": (
        "--no-save-quantized-model",
        dict(
            action="store_false",
            dest="save_quantized_model",
            default=True,
            help="`build` only: skip persisting the calibrated `quantized_model/` bundle "
            "beside the artifact. A build saves it by default, so a later build can point "
            "--quantized-model at it without recalibrating; pass this flag to skip "
            "that disk cost for a build you know you will not reuse",
        ),
    ),
    "tasks": (
        "--tasks",
        dict(nargs="+", default=None, help="`eval` only: the accuracy tasks to score"),
    ),
    "num_eval_samples": (
        "--num-eval-samples",
        dict(type=int, default=None, help="`eval` only: per-task sample cap"),
    ),
    "engine": (
        "--engine",
        dict(
            default=None,
            help="`eval` only: score one leg instead of both; the driver names the legs it has",
        ),
    ),
    # Which part the artifact targets, and how much of it. Declared without `choices` for the
    # same reason `--engine` is: the driver names the values it has, and the config refuses an
    # unknown generation by name. Left unset, the core count follows the generation (a whole
    # Omega part is 4 cores, a whole Europa 8), so the pair cannot be set half-way.
    "generation": (
        "--generation",
        dict(
            default=None,
            metavar="NAME",
            help="`build` / `run` / `benchmark`: hardware generation the artifact targets, e.g. OMEGA / EUROPA; applies to the build, so it is read only when one happens",
        ),
    ),
    "num_cores": (
        "--num-cores",
        dict(
            type=int,
            default=None,
            metavar="N",
            help="`build` / `run` / `benchmark`: AI cores the build spans; default is the whole part for the "
            "chosen generation",
        ),
    ),
    # Media for `run` / `benchmark`, which only a vision-language driver declares. Flags
    # rather than positionals because the prompt is already positional: `run <tag> what is
    # this` would otherwise bind `what` as the picture. Spelled as the model's own run CLI
    # spells them, so nothing has to be translated or sniffed on the way down.
    "image": (
        "--image",
        dict(
            action="append",
            default=None,
            metavar="PATH_OR_URL",
            help="`run` only: picture to ask about; repeat the flag for more than one",
        ),
    ),
    "video": (
        "--video",
        dict(
            default=None,
            metavar="PATH_OR_URL",
            help="`run` only: clip to ask about, instead of --image",
        ),
    ),
    "frames": (
        "--frames",
        dict(type=int, default=None, help="frames to sample from --video, spread over the clip"),
    ),
    # Answer length and sampling, spelled identically by all three vision-language drivers.
    # Every default is None so an untyped knob is not forwarded and the driver's own default
    # stands: they disagree, and an unset sampler flag never reaches `generate()` at all.
    "max_new_tokens": (
        "--max-new-tokens",
        dict(
            type=int,
            default=None,
            metavar="N",
            help="`run` / `benchmark`: answer length cap (default: the driver's own)",
        ),
    ),
    "greedy": (
        "--greedy",
        dict(
            action="store_true",
            help="`run` only: turn sampling off, for a reproducible answer",
        ),
    ),
    "temperature": (
        "--temperature",
        dict(type=float, default=None, help="`run` only: sampling temperature"),
    ),
    "top_p": (
        "--top-p",
        dict(type=float, default=None, help="`run` only: nucleus sampling threshold"),
    ),
    "top_k": (
        "--top-k",
        dict(type=int, default=None, metavar="N", help="`run` only: top-k sampling cutoff"),
    ),
    "min_p": (
        "--min-p",
        dict(type=float, default=None, help="`run` only: minimum-probability sampling floor"),
    ),
    "repetition_penalty": (
        "--repetition-penalty",
        dict(type=float, default=None, help="`run` only: penalty applied to repeated tokens"),
    ),
}

# dest -> the flag a user typed, for a refusal that names what they wrote.
_FLAG_FOR_DEST = {dest: flag for dest, (flag, _) in _FLAGS.items()}

# An input-shape token is a name, optionally suffixed: `name` is exactly one, `name?` at most
# one, `name+` one or more, `name*` any number. Only the last token may repeat, which is what
# keeps the split positional and unambiguous.
_REPEATS = "+*"

#: How a driver announces itself. A module in `<family>/shared/` that declares the verbs it
#: answers IS a driver of that family, so the doors need no table of them.
_DECLARES_VERBS = re.compile(r"^VERBS\s*=", re.M)


def door(family: str) -> str:
    """The command a user types to reach one family."""
    return f"python -m model_axscript.{family}"


def _family_of(directory: Path) -> str:
    """The family a model DIRECTORY sits in, which is the door that answers for it.

    The same rule `registry.family` applies to a tag, read straight off a path the caller
    already holds. Both bulk helpers below use it rather than the registry call, because
    that call re-reads the whole tree per tag and a list of thirty models would pay for
    thirty walks to answer one question.
    """
    return directory.relative_to(_ROOT).parts[0]


def tags(family: str) -> dict[str, Path]:
    """Every tag one door answers for, mapped to its model directory, sorted by tag."""
    return {
        tag: directory
        for tag, directory in registry.models().items()
        if _family_of(directory) == family
    }


def doors() -> list[str]:
    """Every family with a command line of its own, sorted."""
    return sorted({_family_of(directory) for directory in registry.models().values()})


def tag_in(argv: list[str]) -> str | None:
    """The registered tag among what was typed, for a `--help` narrowed to one model.

    Reads the registry, which parses declarations rather than importing packages, so naming a
    tag never costs a torch import. Every tag in the tree is a candidate, not only this
    door's, so a tag typed at the wrong door is answered with the right door rather than with
    "unknown model".
    """
    registered = registry.models()
    return next((token for token in argv if token in registered), None)


def _placeholder(token: str) -> str:
    name = token.rstrip("?+*")
    if token.endswith("+"):
        return f"<{name}> [{name} ...]"
    if token.endswith("*"):
        return f"[{name} ...]"
    if token.endswith("?"):
        return f"[{name}]"
    return f"<{name}>"


def _given(count: int) -> str:
    return f"{count} was given" if count == 1 else f"{count} were given"


def _shape(spec: tuple[str, ...]) -> str:
    """The declared input shape, spelled the way a usage line spells it."""
    return " ".join(_placeholder(token) for token in spec)


def _input_names(drivers: list) -> str:
    """What this family calls its positional input, over the drivers it has.

    The untargeted help has no tag and so no shape, but it does know the family, which is
    enough to stop a vision door offering a prompt and a language door offering an image.
    """
    names = sorted(
        {token.rstrip("?+*") for d in drivers for spec in d.INPUTS.values() for token in spec}
    )
    return " or ".join(names) or "nothing"


def _input_help(inputs: dict[str, tuple[str, ...]] | None, family_input: str = "input") -> str:
    if not inputs:
        return f"{family_input} for `run` / `benchmark`; the family's default when omitted"
    shapes = "; ".join(f"{verb} {_shape(spec)}" for verb, spec in inputs.items() if spec)
    return f"this model's input: {shapes}" if shapes else "this model takes no input"


def _parser(
    family: str,
    options: set[str] | frozenset[str] | None = None,
    verbs: tuple[str, ...] | None = None,
    inputs: dict[str, tuple[str, ...]] | None = None,
    declared: str | None = None,
    usage: str | None = None,
    family_input: str = "input",
) -> argparse.ArgumentParser:
    """The command line at one door. The arguments narrow it further, for a model's help.

    Narrowing is for `--help` only. A real parse uses the whole surface, so a flag that
    belongs to another family is refused by name rather than reported as an unrecognized
    argument, which does not say which family has it. `declared` is what the driver says
    about the ONE model being asked about, which no shared flag list can carry.

    `usage` is how the parse that accepts everything still PRINTS one family's line: the
    union is the surface a refusal has to reach, and it is also the wall of flags the doors
    exist to stop a user reading.
    """
    parser = argparse.ArgumentParser(
        prog=door(family),
        usage=usage,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description=f"Build, run and measure the {family} models in this tree, by tag.",
        epilog="\n\n".join(
            filter(
                None,
                (
                    declared,
                    f"`list` prints every {family} tag with the directory that declares it.",
                ),
            )
        ),
    )
    parser.add_argument("verb", choices=(*(verbs or _VERBS), "list"))
    # The example comes from this family, because a vision user shown an LLM tag reads it as
    # a tag they could type here, and the door would refuse it.
    example = next(iter(tags(family)), "<tag>")
    parser.add_argument("tag", nargs="?", help=f"model tag, e.g. {example} (see `list`)")
    parser.add_argument("input", nargs="*", help=_input_help(inputs, family_input))
    for dest, (flag, kwargs) in _FLAGS.items():
        if options is None or dest in options:
            parser.add_argument(flag, **kwargs)
    return parser


def _declared_by(driver, tag: str) -> str | None:
    """What this model declares about itself, for its own `--help`.

    Optional: a driver with nothing model-specific to add declares no `describe`.
    """
    describe = getattr(driver, "describe", None)
    return describe(tag) if describe else None


def _driver(family: str):
    """The family's DEFAULT driver: what a verb does for a model that names no other.

    Imported by family name rather than selected from a table, so adding a family is adding
    a directory. A driver names its own verbs, options and input shapes, which is what lets
    this module own the whole flag surface without knowing what any family can do.
    """
    name = f"model_axscript.{family}.shared.entry"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as missing:
        if missing.name != name:  # a real import error inside the driver, not a missing driver
            raise
        raise SystemExit(
            f"family {family!r} has no driver: expected {name.replace('.', '/')}.py"
        ) from missing


def _drivers(family: str) -> list:
    """Every driver a family has, for the surface its door offers before a tag is typed.

    Found by the declaration a driver already makes rather than by walking the models: the
    driver a model names is read with a registry call, and the untargeted help would pay one
    of those for every model in the family to answer a question about the family.
    """
    shared = _ROOT / family / "shared"
    names = sorted(
        path.stem
        for path in shared.glob("*.py")
        if _DECLARES_VERBS.search(path.read_text(encoding="utf-8"))
    )
    if not names:
        raise SystemExit(f"family {family!r} has no driver: nothing in {shared} declares VERBS")
    return [importlib.import_module(f"model_axscript.{family}.shared.{name}") for name in names]


def surface(family: str) -> tuple[tuple[str, ...], frozenset[str], str]:
    """The verbs, the options and the input name one door offers, over the family's drivers.

    The union, not the default driver's share of it: a family whose vision-language models
    take a picture must offer `--image` in its own `--help`, or the flag is unreadable until
    a tag is typed. Ordered by `_VERBS` so the usage line is stable whatever the directory
    listing returns.
    """
    drivers = _drivers(family)
    verbs = tuple(verb for verb in _VERBS if any(verb in d.VERBS for d in drivers))
    return verbs, frozenset().union(*(d.OPTIONS for d in drivers)), _input_names(drivers)


def _driver_for(tag: str):
    """The driver that answers for one MODEL: the one it declares, else its family's.

    Resolved HERE rather than inside the family driver, because the narrowed `--help`, the
    verb refusal and the input shape are all read off the driver before anything runs. A
    model whose surface differs from its family's needs to be known at this point or the
    door offers and refuses the wrong flags. A vision-language model takes a picture and a
    clip; the text model beside it takes neither.

    Falls through to `_driver` for the common case, so a family with one driver needs no
    declaration and the family lookup stays the single place a driver is imported.
    """
    declared = registry.declares(tag, "DRIVER", None)
    family = registry.family(tag)
    if not declared:
        return _driver(family)
    name = f"model_axscript.{family}.shared.{declared}"
    try:
        return importlib.import_module(name)
    except ModuleNotFoundError as missing:
        if missing.name != name:  # a real import error inside the driver, not a missing one
            raise
        raise SystemExit(
            f"{tag} declares driver {declared!r}, but "
            f"{name.replace('.', '/')}.py does not exist"
        ) from missing


def _wrong_door(family: str, tag: str) -> str | None:
    """Why this tag belongs at another door, or None when it belongs at this one.

    A tag is unique across the tree, so the door it was typed at is enough to know which one
    it should have been. Naming that door beats refusing the tag as unknown, which is what a
    per-family registry lookup would have said and is false.
    """
    home = registry.family(tag)
    if home == family:
        return None
    return (
        f"{tag} belongs to the {home} family, and this is the {family} command line; "
        f"build and run it with `{door(home)}`"
    )


def _typed(args, parser, dests) -> str:
    """The flags among `dests` the caller actually typed, comma-joined, or "" for none.

    Compared against the parser's OWN default rather than against None, so a store_true flag
    answers the same question the others do and a default that moves keeps the check honest.
    """
    return ", ".join(
        _FLAG_FOR_DEST[dest] for dest in dests if getattr(args, dest) != parser.get_default(dest)
    )


def _unsupported(driver, args, parser) -> list[str]:
    """The flags the caller typed that this family does not honour."""
    return [
        flag
        for dest, flag in _FLAG_FOR_DEST.items()
        if dest not in driver.OPTIONS and getattr(args, dest) != parser.get_default(dest)
    ]


def _bind_inputs(spec: tuple[str, ...], values: list[str], parser, verb: str, tag: str) -> dict:
    """Bind the positionals to the names the family's driver declared for this verb.

    A bag of strings left every driver to guess which token was which, and gave a family that
    needs two kinds of input no way to say so. The names come from the driver, so the guess is
    a declaration and the arity is refused here, before a driver imports torch to complain.
    """
    if not spec:
        if values:
            parser.error(f"`{verb} {tag}` takes no input, and {_given(len(values))}")
        return {}
    usage = f"`{verb} {tag} {_shape(spec)}`"
    bound: dict[str, str | list[str] | None] = {}
    rest = list(values)
    for token in spec:
        name = token.rstrip("?+*")
        if token[-1] in _REPEATS:
            if token.endswith("+") and not rest:
                parser.error(f"{usage} needs {name}; there is no default")
            bound[name], rest = rest, []
        elif rest:
            bound[name] = rest.pop(0)
        elif token.endswith("?"):
            bound[name] = None
        else:
            parser.error(f"{usage} needs {name}; there is no default")
    if rest:
        parser.error(f"{usage} takes {len(spec)}, and {_given(len(values))}")
    return bound


def _refuse_unsupported_verb(argv: list[str], tag: str, driver) -> None:
    """Refuse a verb this family lacks, in this module's words rather than argparse's.

    The narrowed `--help` parser is built from the family's own VERBS, so an unsupported verb
    reaches it as an invalid-choice error naming the ones it does accept. That is the same
    refusal the full parse makes below, minus the reason, so it is made here first: asking for
    help must not replace an explanation with a shorter one.

    A verb no family has is left alone, so argparse still reports it as the unknown token it is.
    """
    verb = next((token for token in argv if not token.startswith("-")), None)
    if verb is None or verb in driver.VERBS or verb not in _VERBS:
        return
    raise SystemExit(
        f"`{verb}` is not available for {tag}: the {registry.family(tag)} family supports "
        f"{', '.join(driver.VERBS)}"
    )


def main(family: str, argv=None) -> int:
    """Carry out one command at the `family` door. Returns a process exit status."""
    argv = list(sys.argv[1:] if argv is None else argv)

    # Narrowed help, before the full parse: once the tag is typed there is one model, and its
    # own flags and input shape are what a user needs rather than the union over its family.
    #
    # The narrowed parser knows only that family's VERBS, so an unsupported verb reaches it as
    # an argparse invalid-choice. That is a worse message than this module's own refusal, and
    # asking for --help must not downgrade the explanation, so the verb is checked first.
    if "-h" in argv or "--help" in argv:
        tag = tag_in(argv)
        if tag is not None:
            elsewhere = _wrong_door(family, tag)
            if elsewhere:
                raise SystemExit(elsewhere)
            driver = _driver_for(tag)
            _refuse_unsupported_verb(argv, tag, driver)
            _parser(
                family, driver.OPTIONS, driver.VERBS, driver.INPUTS, _declared_by(driver, tag)
            ).parse_args(argv)
        family_verbs, family_options, family_input = surface(family)
        _parser(family, family_options, family_verbs, family_input=family_input).parse_args(argv)

    # The full surface, so a flag another family owns is refused by name below rather than
    # reported as an unrecognized argument, which does not say which door has it. The usage
    # line stays this family's: a user who mistyped a vision command must not be shown the
    # whole tree's flags, which is what the doors were split up to stop.
    verbs, options, family_input = surface(family)
    parser = _parser(
        family,
        usage=_parser(family, options, verbs).format_usage().removeprefix("usage: "),
        family_input=family_input,
    )
    # Collected, not refused, so a FORWARDS_UNKNOWN driver can be handed them; everyone else
    # gets the same usage error a few lines down.
    args, extra = parser.parse_known_args(argv)

    if args.verb == "list":
        # No tag, so no driver, so nothing to forward an unknown flag to: it can only be a
        # typo here, and the refusal below is never reached for this verb.
        if extra:
            parser.error(f"unrecognized arguments: {' '.join(extra)}")
        for tag, directory in tags(family).items():
            print(f"{tag:<16} {directory}")
        return 0
    if args.tag is None:
        parser.error(f"`{args.verb}` needs a model tag; `list` prints them")
    for dest in ("num_cal", "seq_len", "num_cores"):
        value = getattr(args, dest)
        if value is not None and value <= 0:
            parser.error(f"{_FLAG_FOR_DEST[dest]} must be positive")

    try:
        registry.model_dir(args.tag)
    except KeyError as unknown_tag:
        parser.error(str(unknown_tag).strip('"'))
    elsewhere = _wrong_door(family, args.tag)
    if elsewhere:
        parser.error(elsewhere)

    driver = _driver_for(args.tag)
    if args.verb not in driver.VERBS:
        parser.error(
            f"`{args.verb}` is not available for {args.tag}: the {family} family supports "
            f"{', '.join(driver.VERBS)}"
        )
    unsupported = _unsupported(driver, args, parser)
    if unsupported:
        parser.error(
            f"{', '.join(unsupported)}: not available for {args.tag}; the {family} family "
            f"honours {', '.join(sorted(_FLAG_FOR_DEST[d] for d in driver.OPTIONS))}"
        )
    # Opt-in, because the other reading of an unknown flag is a typo: a driver wrapping its
    # own CLI receives them, and a family that forwards nothing must still fail here.
    if extra and not getattr(driver, "FORWARDS_UNKNOWN", False):
        parser.error(f"unrecognized arguments: {' '.join(extra)}")
    args.extra = extra
    # Build parameters, so every verb that can build takes them: `run` and `benchmark` compile
    # first when nothing is built. `eval` never does, so there they could only be dropped.
    baked = _typed(
        args,
        parser,
        (
            "generation",
            "num_cores",
            "weights",
            "calibration",
            "quantized_model",
            "save_quantized_model",
        ),
    )
    if baked and args.verb == "eval":
        parser.error(
            f"{baked}: `eval` never builds, so it "
            "cannot choose the part an artifact is compiled for; build it first"
        )
    # Media and --interactive are `run`-only. `benchmark` times a fixed prompt through the
    # model's own vision loop and its CLI takes no picture, so a forwarded --image would be
    # refused inside the driver, on a device, minutes in. Both are checked after the
    # not-available refusal above, so a family that honours neither is told that rather than
    # being told which verb a flag it cannot use belongs to.
    if args.interactive and args.verb != "run":
        parser.error("--interactive applies to `run`")
    media = _typed(args, parser, ("image", "video", "frames"))
    if media and args.verb != "run":
        parser.error(f"{media}: only `run` takes media")
    # --frames is a property OF a video (how much of the clip to sample), so on its own it
    # describes nothing. Silently ignoring it is how a run gets timed at a frame count nobody
    # asked for.
    if args.frames is not None and args.video is None:
        parser.error("--frames samples from --video; name a video to sample, or drop --frames")
    # Sampling shapes ONE answer, so `run` alone, like media: no benchmark CLI declares a
    # sampler flag. --max-new-tokens is absent here because both CLIs take it.
    sampling = _typed(
        args, parser, ("greedy", "temperature", "top_p", "top_k", "min_p", "repetition_penalty")
    )
    if sampling and args.verb != "run":
        parser.error(f"{sampling}: only `run` samples")
    # Neither `build` nor `eval` generates a free answer, so a length cap names nothing.
    if args.max_new_tokens is not None and args.verb not in ("run", "benchmark"):
        parser.error("--max-new-tokens applies to `run` and `benchmark`")
    args.inputs = _bind_inputs(
        driver.INPUTS.get(args.verb, ()), args.input, parser, args.verb, args.tag
    )
    return driver.run(args.tag, args.verb, args)

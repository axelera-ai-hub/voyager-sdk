# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""What each verb DOES for the vision family, once its door has parsed the command line.

`python -m model_axscript.vision` owns the flags; this module owns the behaviour, and it
declares the verbs and options the family honours so the door can refuse anything else by
name.

Six things stay per model, because a bare encoder and a place-recognition head genuinely
differ, and each is a few lines in the model's own package:

    build(config)                       compile the artifact this configuration describes
    load_device(config)                 the loaded device, compiling first if nothing is built
    load_reference(config)              the fp32 counterpart, with the device's interface
    prepare(image, config)              the host-side input for one image
    infer(device, prepared)             the model's output for one prepared input
    report_run(device, images, config)  the result a reader of this task can judge

`load_reference` exists because a published accuracy is a PAIR, the device leg beside the fp32
one, and both legs owe the same host stages: `infer` is the same call on either. No verb here
uses it (a run and a benchmark are about the device), so it is the zoo's accuracy path that
reaches it. See `shared/reference.py`.

`prepare` is separate from `infer` so `benchmark` can hoist the host resize / crop / normalize
OUT of the timed loop: leaving it in measures the JPEG decoder as much as the model, and every
published number for these cards was taken with it hoisted.

`report_run` exists because a feature vector has no readable result of its own: a shape and a
dtype prove only that something well-formed came back, so each model reports agreement with
its own reference instead.
"""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import logging
import os
from pathlib import Path

from model_axscript import registry

from .config import artifact_fields
from .source import SourceOverrideError

VERBS = ("build", "run", "benchmark")
# Scoring is absent on purpose: a metric belongs to the zoo, keyed by task, so `axzoo eval
# <card>` scores a model and this door builds and runs one. A second door onto an accuracy is
# how a card ends up with two numbers. Naming only what is honoured is what makes the front
# door refuse the rest by name instead of accepting a flag it silently drops.
OPTIONS = frozenset(
    {
        "output_dir",
        "device",
        "cal_device",
        "num_cal",
        "weights",
        # `--head` is not paired with `--weights` on purpose: a customer's head over the
        # shipped backbone is a real and common case.
        "head",
        "calibration",
    }
)
# The input shape per verb, which the front door binds and refuses against. `run` reports
# every image named; `benchmark` times exactly one, because a mean over different images
# measures the images as much as the model.
INPUTS = {"run": ("image+",), "benchmark": ("image",)}


def default_output_dir(tag: str) -> Path:
    """Where a build lands when no `--output-dir` is given: `build/<tag>/axscript`.

    The backend is a path component because one tag serves every backend of a card, so a
    vision card with a codegen backend would otherwise have the two builds overwrite each
    other. It is also the path `axelera.zoo` resolves, which is what lets `axzoo eval` and
    `axzoo benchmark` score an artifact this command built.
    """
    return Path("build") / tag / "axscript"


def _require_field(config, name: str, flag: str, tag: str):
    """The configuration field `flag` writes, refused by name when the card declares none.

    Returns its current value, which is what `--weights` then repoints. A card without the
    field otherwise fails as a bare `TypeError` out of `dataclasses.replace`, or an
    `AttributeError` on the read, several frames down and naming neither the flag nor the card.
    Same standard as `_repointed` below, which refuses a source that cannot take an override.
    """
    fields = (
        {f.name for f in dataclasses.fields(config)} if dataclasses.is_dataclass(config) else set()
    )
    if name not in fields:
        raise SystemExit(
            f"{flag} cannot be applied to {tag}: the configuration it declares "
            f"({type(config).__name__}) has no {name}. Set it in the card's own make_config "
            "(see model_axscript/README.md)."
        )
    return getattr(config, name)


#: Which `EncoderSource` method each customer override is written through. A flag added
#: later names its method here rather than growing a branch in `_repointed`.
_APPLY = {"--weights": "with_weights", "--head": "with_head"}


def _repointed(source, spec: str, flag: str, tag: str):
    """The card's own source, carrying what the caller named, or a refusal against `flag`.

    The SOURCE owns the weights for this family: it is what knows how to build the
    architecture and map a state dict onto it. So this repoints the card's source rather than
    substituting a different one, and the architecture the shapes are read off stays the
    card's. Covers the case the README documents as needing no code, a fine-tune of the same
    architecture published as a HuggingFace repo or sitting in a local directory.

    The door ASKS the source (`with_weights` / `with_head`) rather than testing it for a field
    name: a source that spells its weights differently answers for itself, in one place, and
    adding one never edits this function.
    """
    method = _APPLY[flag]
    apply = getattr(source, method, None)
    if apply is None:
        raise SystemExit(
            f"{flag} cannot be applied to {tag}: {type(source).__name__} is not an "
            f"EncoderSource this door can override, because it has no `{method}`. Inherit "
            f"`Overridable` or implement the method (model_axscript/vision/shared/source.py)."
        )
    try:
        return apply(spec)
    except SourceOverrideError as reason:
        raise SystemExit(f"{flag} cannot be applied to {tag}: {reason}") from None


def _output_dir_for(base: Path, config, shipped) -> Path:
    """Where this build lands: the card's own directory, or one keyed to what was overridden.

    A card's artifact directory is fixed, and `builder.is_built` accepts whatever sits there as
    long as it records the same implementation. So without this, a build from your own weights
    OVERWRITES the shipped build, and a run afterwards reports the shipped model's numbers as
    yours. Keyed on the whole configuration rather than on the flag list, so the two cannot
    drift apart: any override that changes what is compiled moves the directory, and a
    configuration equal to the one the card ships lands exactly where it always did.
    """
    mine = artifact_fields(config)
    if mine == artifact_fields(shipped):
        return base
    return base.with_name(f"{base.name}-{hashlib.sha1(repr(mine).encode()).hexdigest()[:8]}")


def _head_override(shipped, source, head: str, tag: str, package) -> dict:
    """The override `--head` writes: the config's head, and the source's copy where it has one.

    ONE question decides whether the flag applies, asked of the card's own configuration: does
    it declare a head. A card that bakes one into its device graph does; a classification card,
    a feature-extraction card and a host-head card do not, and there is nothing for the flag to
    replace. `with_head` is identity for a source whose head is named on the config, so both
    kinds are written the same way here.

    `source` is whatever `--weights` has already put in the overrides, or the card's own when
    that flag was not given, so naming both flags does not drop the weights half.
    """
    if shipped.head is None:
        raise SystemExit(_no_head(tag, package))
    return {"head": head, "source": _repointed(source, head, "--head", tag)}


def _no_head(tag: str, package) -> str:
    """Why `--head` does not apply to this card, in the card's own terms where it has any.

    The shared half states the rule. A card whose head exists but is out of the flag's reach
    DECLARES why, as `NO_HEAD_REASON`, rather than this message naming cards one by one: a
    reader is told where their head actually is, and adding a card never edits a string here.
    """
    shared = (
        f"--head: {tag} compiles no head of its own, so there is nothing for the flag to "
        f"replace. A card that bakes a head into its device graph declares one in its "
        f"configuration, and this one declares none. A classification card carries its "
        f"classifier inside the checkpoint, so bring it with --weights, and a "
        f"feature-extraction card has no head at all."
    )
    reason = getattr(package, "NO_HEAD_REASON", None)
    return f"{shared} {reason}" if reason else shared


def run(tag: str, verb: str, args) -> int:
    """Carry out `verb` for the model `tag` names. Returns a process exit status."""
    # The runtime logs the run banner (device / cores / compiled run size) at INFO; every
    # number printed below is only comparable against another at the same settings, so the
    # banner is part of the output, not debug chatter.
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    package = importlib.import_module(registry.package(tag))
    output_dir = args.output_dir or default_output_dir(tag)
    shipped = package.make_config(output_dir, cal_device=args.cal_device)
    config = shipped
    overrides = {}
    if args.num_cal is not None:
        _require_field(config, "n_calib", "--num-cal", tag)
        overrides["n_calib"] = args.num_cal
    if args.calibration is not None:
        # A HuggingFace dataset id or a local image directory, read recursively and sorted, so
        # two builds of the same folder observe the same scales.
        _require_field(config, "calibration_dataset", "--calibration", tag)
        overrides["calibration_dataset"] = args.calibration
    if args.weights is not None:
        source = _require_field(config, "source", "--weights", tag)
        overrides["source"] = _repointed(source, args.weights, "--weights", tag)
        # `repo_id` is a SECOND copy of the weights address, and it is the identity written
        # into meta.json. Repointing only the source compiles the caller's weights and then
        # records them under the shipped model's name.
        _require_field(config, "repo_id", "--weights", tag)
        overrides["repo_id"] = args.weights
    if args.head is not None:
        source = overrides.get("source", getattr(config, "source", None))
        overrides.update(_head_override(shipped, source, args.head, tag, package))
    # Applied here, not passed into make_config, so an override means the same thing for every
    # model in the family however each one builds its configuration.
    if overrides:
        config = dataclasses.replace(config, **overrides)
    if args.output_dir is None:
        # Only the DEFAULT directory is keyed. A caller who named one asked for that one.
        output_dir = _output_dir_for(output_dir, config, shipped)
        config = dataclasses.replace(config, artifact_dir=output_dir)

    if verb == "build":
        print(f"[build] {tag} -> {output_dir}")
        package.build(config)
        return 0

    if args.device is not None:  # RuntimeModel._select_device reads this; no recompile
        os.environ["AXELERA_METIS_DEVICE"] = str(args.device)

    device = package.load_device(config)
    if verb == "run":
        print(package.report_run(device, args.inputs["image"], config))
        return 0

    from .build import timeit

    # Hoisted deliberately: what is timed is the model, not the image decoder.
    prepared = package.prepare(args.inputs["image"], config)
    print(timeit(lambda: package.infer(device, prepared)))
    return 0

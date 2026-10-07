# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""The shape of an AxScript model package. Copy this folder and fill it in.

A model package is DECLARATIVE. It names the tag it answers to and the configuration it
builds from, and it defines the seams its family's driver calls; it never parses a command
line of its own. `python -m model_axscript.<family> <verb> <tag>` is the command line in this
tree, and it resolves this package by reading `MODEL_TAG` out of this file WITHOUT importing
it, so listing the models costs nothing. Keep torch, the compiler and every heavy import
inside the function that needs them.

The seams below are the vision family's set, which is the general shape: build the artifact,
load it, turn one input into what the model consumes, run it, and report a result a reader of
this task can judge. A family declares its own set in `<family>/shared/entry.py`, so check
that file for the family you are joining and delete the seams it does not call.
"""

from pathlib import Path

MODEL_TAG = "your-model-tag"


def make_config(artifact_dir, cal_device: str = ""):
    """Everything the build and the run both need: checkpoint, shapes, calibration, output.

    One configuration for both, so a run reads the same declaration the artifact was compiled
    from. `cal_device` is a compile-only knob (the torch device calibration runs on); the run
    path ignores it.

    Declare the run sizes this model has been MEASURED at. A size outside that set is refused
    before the compile starts, because an unmeasured one does not fail at compile time: it
    fails on chip, minutes later, as a hang or a saturated activation table.
    """
    raise NotImplementedError(
        "return your family's config dataclass, e.g. "
        "`vision.shared.config.EncoderConfig(..., artifact_dir=Path(artifact_dir))`"
    )


def build(config) -> None:
    """Compile the artifact `config` describes into its `artifact_dir`."""
    raise NotImplementedError("call your family's build orchestrator with `config`")


def load_device(config):
    """The loaded model, compiling first only when nothing is built at `artifact_dir`."""
    if not (Path(config.artifact_dir) / "model.json").is_file():
        print(f"[build] {MODEL_TAG} -> {config.artifact_dir}")
        build(config)
    raise NotImplementedError("return your family's loaded device for `config.artifact_dir`")


def prepare(model_input, config):
    """The host-side form of ONE input: the resize, crop, normalize or tokenize this model does.

    Separate from `infer` so `benchmark` can hoist it out of the timed loop. Leaving it inside
    measures the image decoder or the tokenizer as much as the model.
    """
    raise NotImplementedError("preprocess one input through this model's own upstream")


def infer(device, prepared):
    """The model's output for one prepared input, on the loaded device."""
    raise NotImplementedError("run one dispatch and return its output")


def report_run(device, model_inputs, config) -> str:
    """The result a reader of THIS task can judge, never a tensor shape.

    A shape and a dtype prove only that something well-formed came back. Report the labels, the
    text, or the agreement with this model's own fp reference, so a run says whether the model
    is working rather than whether it returned.
    """
    raise NotImplementedError("report a task-meaningful result for each input")

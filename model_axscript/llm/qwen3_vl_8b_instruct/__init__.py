# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL-8B-Instruct on the AxScript path: the build+run config for this model.

The same two towers as :mod:`model_axscript.llm.qwen3_vl_2b_instruct`, with different
numbers: the text side scales by config alone, and the vision side is depth 27 / hidden
1152, whose two non-PWORD widths (head_dim 72, intermediate 4304) are zero-padded on
device by ``vision_impl``. No kernel is new, so this package is configuration only.
"""

from pathlib import Path

MODEL_TAG = "qwen3-vl-8b-instruct"

#: See the 2B package: two towers answer every verb differently, and this model's multi-turn
#: session is its own rather than the text family's chat REPL. Read by `registry` without
#: importing the package.
DRIVER = "vl_entry"


#: The context this model ships at, and what ``make_config`` returns when the caller names
#: none. Kept as a module constant so `--context-len`'s help can print the default without
#: constructing a config (which imports torch).
DEFAULT_CONTEXT_LEN = 2048

#: DocVQA (image, question) pairs a shipped build calibrates on. A module constant so the
#: entry can print the default in --help without constructing a config.
DEFAULT_CALIBRATION_SAMPLES = 48

#: The part a build targets when the caller names nothing, set to Omega with the rest of the
#: family so one default holds across every VLM here. READ THE MEMORY ARITHMETIC IN
#: ``make_config`` FIRST: at 6.95 GB of resident int8 decoder weights this is the one size in
#: the family a 4 GiB Metis board cannot hold, so an Omega build COMPILES and then fails at
#: LOAD on such a part. `--generation EUROPA` is what these weights were sized for, and a
#: larger-memory Metis board is the other way this default becomes loadable.
DEFAULT_GENERATION = "OMEGA"


def make_config(
    artifact_dir,
    cal_device: str = "",
    context_len: int | None = None,
    generation: str | None = None,
    num_cores: int | None = None,
    num_calibration_samples: int | None = None,
):
    """Per-model build+run config (checkpoint, geometry, context length, cores).

    Used by the family driver for both build and run, and by the zoo for the checkpoint id and
    the declared run sizes. ``cal_device`` is a compile-only knob (the torch device used for
    calibration); the runtime path ignores it.

    ``context_len`` overrides the shipped default for one build, which is how ``--context-len``
    reaches the compile. It does NOT bypass the gate: ``vl_entry.build`` still puts the value
    through ``validate_run_size`` against ``supported_context_lens`` below, so an override is
    a choice among the sizes measured on chip rather than a way around them. Widening that
    set is a separate, deliberate edit backed by a measurement.

    ``generation`` picks the part the artifact targets -- OMEGA (Metis) or EUROPA -- and
    ``num_cores`` how much of it the build spans; left unset, the core count follows the
    generation (``vl_config.CORES_PER_GENERATION``: 4 on Omega, 8 on Europa) so the pair cannot be
    set inconsistently by forgetting one of them.

    ``num_calibration_samples`` trades accuracy for build time. Calibration is the bulk of a
    build, so cutting it is the way to get a runnable artifact quickly -- and the result is a
    WORSE-quantized model, not merely a faster-built one, since the activation scales are
    fitted on whatever few samples were seen. Use it to exercise the pipeline, never to
    produce a number.
    """
    # deferred: avoids an import cycle
    from ..shared.vl_config import VlmConfig

    return VlmConfig(
        implementation="qwen3vl-metis-multicore",
        repo_id="Qwen/Qwen3-VL-8B-Instruct",
        artifact_dir=Path(artifact_dir),
        # WHERE THIS SET COMES FROM, and how it differs from the 2B's. 8B's resident int8
        # decoder weights are 6.95 GB (36 layers x 192.9 M) against the 4 GiB a PCIe Metis
        # board reports, and both towers must be resident at once, so this size does not fit
        # on a 4 GiB Omega part. It defaults to Omega with the rest of the family regardless,
        # and that is a knowing trade rather than an oversight: the build will COMPILE, and it
        # is the DEVICE that will not hold the result -- a load-time failure, not a build-time
        # one. `--generation EUROPA` (eight cores) is what these weights were sized for; a
        # larger-memory Metis board is the other part that can hold them.
        #
        # None of the set below is backed by an on-chip measurement -- there is no Europa
        # part here -- which is why the card stays `pending` and its report carries no
        # accuracy row. The declared sets deliberately match the 2B's so that one
        # --context-len or --image-size means the same thing across the family; that makes
        # them build intent, not evidence. Record measurements here as they are taken.
        # Read from the argument or the shipped default, never from the environment. `axzoo
        # build` forwards --seq-len / --num-cal to the front door as ARGUMENTS on this backend
        # (`_cli/build.py`); AXZOO_COMPILE_SEQ_LEN / AXZOO_NUM_CAL cross that boundary only for
        # the generated codegen recipes, which run as their own subprocess. Reading them here too
        # made `axzoo show-config` answer differently under an exported shell variable, for these
        # two models and no others. What a model declares is not environment-dependent.
        context_len=DEFAULT_CONTEXT_LEN if context_len is None else int(context_len),
        supported_context_lens=(512, 2048),
        # 288x512 is an 18x32 patch grid -> 144 image tokens: a page-shaped geometry rather
        # than a square, which is what a document-VQA model is calibrated on. The same pair of
        # sets the 2B declares, so one --image-size / --context-len works across the family.
        image_size=(288, 512),
        supported_image_sizes=((256, 256), (288, 512)),
        fit_mode="stretch",
        num_cores=num_cores,
        generation=DEFAULT_GENERATION if generation is None else generation,
        num_calibration_samples=(
            DEFAULT_CALIBRATION_SAMPLES
            if num_calibration_samples is None
            else int(num_calibration_samples)
        ),
        cal_device=cal_device,
    )


#: The display name the family command line prints. A literal, so the front door and the
#: zoo can read it without importing this package.
DISPLAY_NAME = "Qwen3-VL-8B-Instruct"


def compile_towers(*args, **kwargs):
    """Compile both towers. The family driver's `build` seam.

    Imported here rather than at module scope: naming what answers a verb must cost nothing,
    and `pipeline` pulls torch and the compiler in.
    """
    from ..models.qwen3vl.pipeline import build_artifacts

    return build_artifacts(*args, **kwargs)


#: The driver module behind `run`, for the `--help` epilog to point at. Declared
#: rather than derived: `run` below is a forwarder defined HERE, so its `__module__`
#: is this package -- which has no `__main__.py` and cannot be run with `-m`.
RUN_MODULE = "model_axscript.llm.models.qwen3vl.qwen3vl_run"


def run(*args, **kwargs):
    """Answer `run`: the model's own CLI, which owns the run flags and prints the output."""
    from ..models.qwen3vl.qwen3vl_run import main

    return main(*args, **kwargs)


def benchmark(*args, **kwargs):
    """Answer `benchmark`: the model's own timing loop, whose numbers its README quotes."""
    from ..models.qwen3vl.benchmark import main

    return main(*args, **kwargs)

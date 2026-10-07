# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL-2B-Instruct on the AxScript path: the build+run config for this model.

The model code itself lives in ``model_axscript/llm/models/qwen3vl/`` (both towers, the
torch reference, the HuggingFace-shaped run API) and the family command line in
``model_axscript/llm/shared/vl_entry.py``; this package holds the per-model
configuration and the seams the family driver calls.
"""

from pathlib import Path

MODEL_TAG = "qwen3-vl-2b-instruct"

#: The driver in ``llm/shared/`` that answers this model's verbs. Two towers compile, run,
#: time and score differently at every verb, and this model's multi-turn session is its own
#: (`/image`, `/video`, `/reset` against the attached media) rather than the text family's
#: chat REPL. A literal, because `registry` reads it without importing the package.
DRIVER = "vl_entry"


#: The context this model ships at, and what ``make_config`` returns when the caller names
#: none. Kept as a module constant so `--context-len`'s help can print the default without
#: constructing a config (which imports torch).
DEFAULT_CONTEXT_LEN = 2048

#: DocVQA (image, question) pairs a shipped build calibrates on. A module constant so the
#: entry can print the default in --help without constructing a config.
DEFAULT_CALIBRATION_SAMPLES = 48

#: The part this model is released against. Both generations are supported -- `--generation`
#: picks -- but this is the one its resident weights were sized for, and the core count
#: follows from it rather than being repeated here.
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
        # num_cores > 1 resolves the multicore decoder kernel; the tag is what lands in the
        # artifact's pipeline.json.
        implementation="qwen3vl-metis-multicore",
        repo_id="Qwen/Qwen3-VL-2B-Instruct",
        artifact_dir=Path(artifact_dir),
        # Shipped configuration: an 18x32 patch grid (288x512 -> 144 image tokens) and a
        # 2048-token context, both towers spanning the whole Omega part (4 cores, from
        # DEFAULT_GENERATION). The page-shaped grid keeps a document's aspect ratio far better
        # than a square does, which is what this model is calibrated on, and 2048 is the
        # context the driver itself defaults to.
        #
        # WHAT IS MEASURED. Both declared sizes have now been run on chip, on metis-0:1:0,
        # firmware v1.9.0-a1-4-g07dba233f6da. Both entries are here because a supported size
        # with no recorded measurement behind it is an unbacked claim.
        #
        #   SHIPPED, 288x512 / 2048 on four cores (18x32 grid, 576 patches, 144 image tokens):
        #
        #     vision tower   126.5 ms/frame (best 119.6, 10 runs), 7.9 img/s
        #     TTFT           1.31 s   (vision 124 ms + prefill 1182 ms, 160 tokens, 135 tok/s)
        #     decode         7.85 tok/s (127.4 ms/token; 7.75 wall)
        #
        #   256x256 / 512 on four cores, the size this model first shipped at:
        #
        #     vision tower   83.3 ms/frame (one dispatch + 3 DeepStack maps), 11.5 img/s
        #     TTFT           0.281 s
        #     decode         10.3 tok/s
        #
        # Reading the gap between them, because both numbers are correct and neither is a
        # regression. TTFT is a function of the COMPILED CONTEXT, not of the prompt -- the
        # prefill graph ingests every compiled position whatever it is sent -- so 4x the
        # context costs ~4.6x the TTFT, and that is the price of the larger window rather
        # than a slower part. Vision grows with patches, 2.25x of them for 1.52x the time.
        # Decode is slower at 2048 because each step reads a KV cache 4x the size.
        #
        # Decode is DDR-bound on the weights and, on this host, on the lm_head: 36.2 ms of a
        # 127 ms step is the host fp lm_head gemv (28%), so 10.97 tok/s is what the device
        # alone sustains. That is a host-memory cost, not device DDR.
        #
        # Still missing for the shipped size: a wikitext row (`--eval`). Take it before this
        # model's card moves past `pending`, and do not widen either supported set further
        # without the same four numbers.
        # Read from the argument or the shipped default, never from the environment. `axzoo
        # build` forwards --seq-len / --num-cal to the front door as ARGUMENTS on this backend
        # (`_cli/build.py`); AXZOO_COMPILE_SEQ_LEN / AXZOO_NUM_CAL cross that boundary only for
        # the generated codegen recipes, which run as their own subprocess. Reading them here too
        # made `axzoo show-config` answer differently under an exported shell variable, for these
        # two models and no others. What a model declares is not environment-dependent.
        context_len=DEFAULT_CONTEXT_LEN if context_len is None else int(context_len),
        supported_context_lens=(512, 2048),
        image_size=(288, 512),
        supported_image_sizes=((256, 256), (288, 512)),
        # The PAIRS a report row can be filed under, which is not the product of the two sets
        # above: those gate one axis each, and between them they admit four combinations while
        # two have been compiled and measured. A declared identity with no measured row fails
        # the report's both-halves check for a card at `validated` or above -- widening the
        # set is a measurement, not an edit -- so only the measured pairs are stated.
        #
        # 256x256 is still a supported GEOMETRY and still gated as one: it is the size this
        # model first shipped at, and its numbers are in the comment above. It rejoins this
        # tuple when an artifact at it is built and measured again.
        supported_inputs=(
            VlmConfig.inputs_for((288, 512), 512),
            VlmConfig.inputs_for((288, 512), 2048),
        ),
        fit_mode="stretch",
        num_cores=num_cores,
        generation=DEFAULT_GENERATION if generation is None else generation,
        # The full DocVQA mix a shipped build takes. The regression tier deliberately differs,
        # using a smaller budget to keep a nightly run short.
        num_calibration_samples=(
            DEFAULT_CALIBRATION_SAMPLES
            if num_calibration_samples is None
            else int(num_calibration_samples)
        ),
        cal_device=cal_device,
    )


#: The display name the family command line prints. A literal, so the front door and the
#: zoo can read it without importing this package.
DISPLAY_NAME = "Qwen3-VL-2B-Instruct"


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

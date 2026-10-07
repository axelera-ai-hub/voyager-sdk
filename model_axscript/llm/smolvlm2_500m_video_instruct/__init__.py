# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""SmolVLM2-500M-Video-Instruct on the AxScript path: the build+run config for this model.

The model code itself lives in ``model_axscript/llm/models/smolvlm2/`` (both towers, the
torch reference, the HuggingFace-shaped run API) and the family command line in
``model_axscript/llm/shared/vl_entry.py``; this package holds only the per-model
configuration that ``__main__.py`` drives.

All three SmolVLM2 sizes build through one recipe -- every shape is read from the resolved
HuggingFace config -- so what distinguishes this file is the checkpoint and the geometry
that follows from it, not a different code path.
"""

from pathlib import Path

MODEL_TAG = "smolvlm2-500m-video-instruct"

#: The driver in ``llm/shared/`` that answers this model's verbs. Two towers compile, run,
#: time and score differently at every verb, and this model's multi-turn session is its own
#: (`/image`, `/video`, `/reset` against the attached media) rather than the text family's
#: chat REPL. A literal, because `registry` reads it without importing the package.
DRIVER = "vl_entry"


#: The context this model ships at, and what ``make_config`` returns when the caller names
#: none. Kept as a module constant so ``--seq-len``'s help can print the default without
#: constructing a config (which imports torch).
#:
#: SmolVLM2 SPLITS a picture into tiles rather than resizing it to one grid, so the prompt
#: cost scales with tile count: 64 tokens per tile here, and one tiled
#: photograph is already ~1100 tokens. That is why the shipped context is 2048 rather than
#: the 512 a text-only model of this size would ship at -- a single image plus its question
#: has to fit alongside the answer.
DEFAULT_CONTEXT_LEN = 2048

#: Post-training-quantization calibration budget, in DocVQA ENTRIES. Each entry is spent
#: twice -- once as a (document image, question) still and once as its (question, answer) chat
#: turn with the page left out -- so the pass costs twice this many forwards and both halves
#: come from one source. Both are load-bearing: the stills set the vision path's int8 scales,
#: the text turns stop the eos logit being mis-scaled by a prompt that is >90% image tokens.
#: A module constant so the entry can print the default in --help without constructing a
#: config. The floor is 1: one entry reaches both regimes.
DEFAULT_CALIBRATION_SAMPLES = 26

#: The part a build targets when the caller names nothing; ``--generation`` picks the other,
#: and the core count follows from it (4 on Omega) rather than being repeated here.
#: Omega (Metis) is the default across every VLM in this family: it is the part on
#: hand, so a default build is one that can be loaded where it was made.
#: The on-chip figures recorded below were taken on EUROPA at eight cores -- they are the
#: measurements this model HAS, not measurements of this default.
DEFAULT_GENERATION = "OMEGA"

#: The tile geometry this checkpoint's vision tower is compiled at, in pixels.
#:
#: NOT a knob. Unlike the family's other VLM, whose tower is pinned to a patch grid the
#: caller chooses, SmolVLM2's tower is built at the tile size baked into the checkpoint
#: (512/16 here) and a picture is split into tiles to match. ``pipeline.build_artifacts``
#: validates a request against the checkpoint's own config and refuses anything else, so this
#: is a declaration of what the checkpoint IS.
#:
#: Shares the 256M's vision tower bit for bit, so only the decoder differs: same
#: tile geometry, same tokens per tile, a wider and deeper text trunk.
TILE_SIZE = (512, 512)


def make_config(
    artifact_dir,
    cal_device: str = "",
    context_len: int | None = None,
    generation: str | None = None,
    num_cores: int | None = None,
    num_calibration_samples: int | None = None,
):
    """Per-model build+run config (checkpoint, geometry, context length, cores).

    Used by ``__main__.py`` for both build and run, and by the zoo for the checkpoint id and
    the declared run sizes. Every knob here is a field of ``shared.vl_config.VlmConfig`` and
    is documented there, per field -- including why ``context_len`` is gated to the sizes
    measured on chip (``artifact_contract.validate_run_size``) rather than free, and why the
    core count follows ``generation`` (``shared.hardware.CORES_PER_GENERATION``) when left
    unset.

    ``num_calibration_samples`` counts DocVQA ENTRIES here, and each is spent both as a still
    and as a text-only turn, so the floor is 1: one document already reaches both regimes.
    """
    # deferred: avoids an import cycle
    from ..shared.vl_config import VlmConfig

    return VlmConfig(
        # num_cores > 1 resolves the multicore decoder kernel; the tag is what lands in the
        # artifact's manifest.
        implementation="smolvlm2-metis-multicore",
        repo_id="HuggingFaceTB/SmolVLM2-500M-Video-Instruct",
        artifact_dir=Path(artifact_dir),
        # Shipped configuration: 768 wide x 12 layers @512/16 vision tower +
        # 960 wide x 32 layers, 15q/5kv decoder, both spanning the whole
        # part the build targets (8 cores on the Europa default).
        #
        # MEASURED ON CHIP, ON EUROPA -- NOT the part this entry defaults to, which is now
        # Omega; these are the numbers this model has, and a re-measure on Omega is what
        # would replace them rather than a relabel. europa-0:2:0,
        # firmware v1.9.0-a2, 8 cores of EUROPA, context 2048, --num-cal 1. Warm medians
        # from `python -m model_axscript.llm benchmark smolvlm2-500m-video-instruct`.
        #
        #   vision       19.0 fps (52.5 ms/frame: 48.2 encoder + 4.4 shuffle+connector)
        #   decode       34.1 tok/s
        #   TTFT         475 ms at 2048 positions (and 475 ms at 5, so it is the
        #                context that is being read and not the prompt)
        #   artifact     112 MB vision + 786 MB text (+ 487 MB quantized bundle, not uploaded)
        #
        # WHAT IS MEASURED: every entry below has been built and run on chip; the numbers
        # are recorded in the zoo report block for this stem. A supported size with no
        # recorded measurement behind it is an unbacked claim, so this set stays as narrow
        # as the evidence -- widening it is an on-chip measurement, not an edit.
        context_len=DEFAULT_CONTEXT_LEN if context_len is None else int(context_len),
        supported_context_lens=(2048,),
        image_size=TILE_SIZE,
        supported_image_sizes=(TILE_SIZE,),
        # SmolVLM2 tiles rather than stretching or letterboxing; "tile" is the truthful
        # spelling for this architecture and `build_artifacts` refuses the other two.
        fit_mode="tile",
        num_cores=num_cores,
        generation=DEFAULT_GENERATION if generation is None else generation,
        num_calibration_samples=(
            DEFAULT_CALIBRATION_SAMPLES
            if num_calibration_samples is None
            else int(num_calibration_samples)
        ),
        # SmolLM2 has no QK-norm, so there is no QK smoothing to apply. Stated here rather
        # than left at the family default, which would record a setting the build did not use.
        qk_alpha=None,
        cal_device=cal_device,
    )


#: The display name the family command line prints. A literal, so the front door and the
#: zoo can read it without importing this package.
DISPLAY_NAME = "SmolVLM2-500M-Video-Instruct"


def compile_towers(*args, **kwargs):
    """Compile both towers. The family driver's `build` seam.

    Imported here rather than at module scope: naming what answers a verb must cost nothing,
    and `pipeline` pulls torch and the compiler in.
    """
    from ..models.smolvlm2.pipeline import build_artifacts

    return build_artifacts(*args, **kwargs)


#: The driver module behind `run`, for the `--help` epilog to point at. Declared
#: rather than derived: `run` below is a forwarder defined HERE, so its `__module__`
#: is this package -- which has no `__main__.py` and cannot be run with `-m`.
RUN_MODULE = "model_axscript.llm.models.smolvlm2.smolvlm2_run"


def run(*args, **kwargs):
    """Answer `run`: the model's own CLI, which owns the run flags and prints the output."""
    from ..models.smolvlm2.smolvlm2_run import main

    return main(*args, **kwargs)


def benchmark(*args, **kwargs):
    """Answer `benchmark`: the model's own timing loop, whose numbers this entry quotes."""
    from ..models.smolvlm2.benchmark import main

    return main(*args, **kwargs)

# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""LFM2.5-VL-450M on the AxScript path: the build+run config for this model.

The model code itself lives in ``model_axscript/llm/models/lfm2vl/`` (both towers, the
torch reference, the HuggingFace-shaped run API) and the family command line in
``model_axscript/llm/shared/vl_entry.py``; this package holds only the per-model
configuration that ``__main__.py`` drives.

Both released LFM2.5-VL sizes share that model code -- every width and depth is read from
the resolved HuggingFace config -- so this file differs from its 3B sibling only in the
checkpoint it names and what follows from it.
"""

from pathlib import Path

MODEL_TAG = "lfm2.5-vl-450m"

#: The driver in ``llm/shared/`` that answers this model's verbs. Two towers compile, run,
#: time and score differently at every verb, and this model's multi-turn session is its own
#: (`/image`, `/video`, `/reset` against the attached media) rather than the text family's
#: chat REPL. A literal, because `registry` reads it without importing the package.
DRIVER = "vl_entry"


#: The context this model ships at, and what ``make_config`` returns when the caller names
#: none. Kept as a module constant so ``--seq-len``'s help can print the default
#: without constructing a config (which imports torch).
DEFAULT_CONTEXT_LEN = 2048

#: DocVQA (image, question) pairs a shipped build calibrates on. A module constant so the
#: entry can print the default in --help without constructing a config.
DEFAULT_CALIBRATION_SAMPLES = 48

#: The part a build targets when the caller names nothing; ``--generation`` picks the other,
#: and the core count follows from it (4 on Omega) rather than being repeated here.
#: Omega (Metis) is the default across every VLM in this family: it is the part on
#: hand, so a default build is one that can be loaded where it was made.
#: The headline figures recorded below were taken on EUROPA at eight cores; the earlier Omega
#: run kept beside them is the one this default now produces. At 0.38 GB for both towers
#: together this size has room on either part.
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

    Used by ``__main__.py`` for both build and run, and by the zoo for the checkpoint id and
    the declared run sizes. Every knob here is a field of ``shared.vl_config.VlmConfig`` and
    is documented there, per field -- including why ``context_len`` is gated to the sizes
    measured on chip (``artifact_contract.validate_run_size``) rather than free, and why the
    core count follows ``generation`` (``shared.hardware.CORES_PER_GENERATION``) when left
    unset.
    """
    # deferred: avoids an import cycle
    from ..shared.vl_config import VlmConfig

    return VlmConfig(
        # The LFM2.5-VL decoder kernel takes the core count as a parameter, so there is no
        # separate multicore name to select here; the tag is what lands in the artifact's
        # pipeline.json.
        implementation="lfm2vl-metis-opt",
        repo_id="LiquidAI/LFM2.5-VL-450M",
        artifact_dir=Path(artifact_dir),
        # Shipped configuration: the checkpoint's own 512x512 tile size (a 32x32 patch grid
        # -> 256 image tokens) and a 2048-token context, both towers spanning the whole
        # part the build targets. 512 is what lets a build TILE at all -- a tile IS the
        # compiled grid, so a high-resolution picture is cut into copies of it plus a thumbnail.
        #
        # MEASURED ON CHIP: `axzoo report lfm2.5-vl-450m --backend axscript`, recorded in
        # `axelera_zoo/reports/llm/lfm2_5_vl_450m.yaml`. The numbers live there, not here.
        #
        # THE RUN ITSELF, verbatim, because the numbers say how fast it is and only the
        # transcript says there was an answer at all:
        #
        #   python -m model_axscript.llm run lfm2.5-vl-450m "What is in this picture?" \
        #       --image <a photograph of the Statue of Liberty> --no-tiles --greedy
        #
        #   > This picture shows a futuristic cityscape with a prominent statue in the
        #     foreground. The statue appears to be a tall, slender figure, possibly a woman,
        #     standing on a pedestal. It's surrounded by a circular area with a small platform
        #
        # On natural photographs it reads the scene (two cats on a pink couch, correctly, on
        # another). It is weakest on tight close-ups: a peeled banana filling the frame came
        # back as "a broken piece of white tissue paper" unsplit, and "a white, fibrous object
        # ... resembling a piece of fruit or vegetable" tiled. That is this size's own limit
        # against a DocVQA-calibrated quantization, not a geometry or a tiling defect -- the
        # calibration set is documents and the clip is fitted on them.
        #
        # The wikitext row is WORD-denominated (lm-eval counts on the pre-detokenization
        # page). `device_perplexity` is token-denominated, so the two differ by a
        # multiplicative factor and must never go in one table -- differencing them
        # manufactures a regression or hides one. It also scores the DECODER ALONE, which is
        # most of the parameters and all of the generation but is not this model's
        # vision-language accuracy; that is a DocVQA ANLS number against a dataset harness
        # this tree does not ship yet. `--pipe torch` cannot supply the fp reference leg for
        # ANY VLM here: the shared eval driver loads it with AutoModelForCausalLM, and an
        # Lfm2VlConfig is not a causal-LM config. So the compiled number is obtainable and
        # the torch-vs-compiled drift is not -- the same gap Qwen3-VL and SmolVLM2 have.
        #
        # Read TTFT against the COMPILED CONTEXT, not the prompt: the prefill graph ingests
        # every compiled position whatever it is sent, so most of it is the 2048-token
        # window and not the 274-token prompt. Tiling is the other half of the cost -- the
        # same picture split into 7 dispatches is 1792 image tokens and an 1817-token prompt,
        # which multiplies the vision time and lengthens TTFT with it.
        #
        # ONE entry in each set, because one is what has been measured. The architecture
        # also builds at 256x256 (a 16x16 grid, 64 image tokens, four times cheaper per
        # dispatch and unable to tile -- `Lfm2VlVisionConfig.tiling_supported` says so) and
        # at other context lengths, and `--seq-len` would compile them.
        # Neither is declared here: `validate_run_size` refuses what is not in these sets,
        # and that is deliberate -- a declared size is a claim that it was run on chip.
        # Widening either set is a measurement recorded below, not an edit to make a build
        # pass.
        context_len=DEFAULT_CONTEXT_LEN if context_len is None else int(context_len),
        supported_context_lens=(2048,),
        image_size=(512, 512),
        supported_image_sizes=((512, 512),),
        # How a non-square picture meets the square grid. "stretch" spends every patch on
        # content and distorts; "letterbox" keeps the aspect ratio and pads with mid-grey,
        # which the 0.5/0.5 normalization sends to ~0. Baked into the artifact, so
        # inference reproduces the geometry calibration saw.
        fit_mode="stretch",
        num_cores=num_cores,
        generation=DEFAULT_GENERATION if generation is None else generation,
        # The full DocVQA mix a shipped build takes. Calibration is flat in the core count
        # and is the dominant QUALITY knob: a 4-sample build answers in vague generalities
        # where a 48-sample one reads the scene.
        num_calibration_samples=(
            DEFAULT_CALIBRATION_SAMPLES
            if num_calibration_samples is None
            else int(num_calibration_samples)
        ),
        cal_device=cal_device,
    )


#: The display name the family command line prints. A literal, so the front door and the
#: zoo can read it without importing this package.
DISPLAY_NAME = "LFM2.5-VL-450M"


def compile_towers(*args, **kwargs):
    """Compile both towers. The family driver's `build` seam.

    Imported here rather than at module scope: naming what answers a verb must cost nothing,
    and `pipeline` pulls torch and the compiler in.
    """
    from ..models.lfm2vl.pipeline import build_artifacts

    return build_artifacts(*args, **kwargs)


#: The driver module behind `run`, for the `--help` epilog to point at. Declared
#: rather than derived: `run` below is a forwarder defined HERE, so its `__module__`
#: is this package -- which has no `__main__.py` and cannot be run with `-m`.
RUN_MODULE = "model_axscript.llm.models.lfm2vl.lfm2vl_run"


def run(*args, **kwargs):
    """Answer `run`: the model's own CLI, which owns the run flags and prints the output."""
    from ..models.lfm2vl.lfm2vl_run import main

    return main(*args, **kwargs)


def benchmark(*args, **kwargs):
    """Answer `benchmark`: the model's own timing loop, whose numbers this entry quotes."""
    from ..models.lfm2vl.benchmark import main

    return main(*args, **kwargs)

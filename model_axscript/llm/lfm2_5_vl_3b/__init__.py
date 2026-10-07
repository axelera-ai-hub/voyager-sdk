# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""LFM2.5-VL-3B on the AxScript path: the build+run config for this model.

The model code itself lives in ``model_axscript/llm/models/lfm2vl/`` (both towers, the
torch reference, the HuggingFace-shaped run API) and the family command line in
``model_axscript/llm/shared/vl_entry.py``; this package holds only the per-model
configuration that ``__main__.py`` drives.

Both released LFM2.5-VL sizes share that model code -- every width and depth is read from
the resolved HuggingFace config -- so this file differs from its 450M sibling only in the
checkpoint it names and what follows from it. The difference in KIND is inside the shared
code and needs nothing here: this checkpoint's SigLIP2 NaFlex So400m encoder has a
head_dim of 72 and an MLP width of 4304, neither a multiple of the 64-lane PWORD every
device buffer's channel axis is laid out at, so the fused encoder-block kernel lays a head
out at a 128-lane stride and pads fc1's output tail. The padding lives entirely in the
weight layout; the torch reference, the calibration and the exported quant parameters all
run the checkpoint's own widths.
"""

from pathlib import Path

MODEL_TAG = "lfm2.5-vl-3b"

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
#: and the core count follows from it (4 on Omega). Omega (Metis) is the default across every
#: VLM in this family: it is the part on hand, so a default build is one that can be loaded
#: where it was made.
#: The on-chip figures recorded below were taken on EUROPA, at eight cores -- they are the
#: measurements this model HAS, not measurements of this default.
#: See the memory arithmetic in
#: ``make_config``: it fits a 4 GiB part and not a 1 GB one, and the refusal for a part that
#: cannot hold both towers is explicit at load.
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
        repo_id="LiquidAI/LFM2.5-VL-3B",
        artifact_dir=Path(artifact_dir),
        # WHAT THIS SIZE COSTS, and why that is the first thing to check. Both towers are
        # resident on the device at once, so an artifact only runs on a part whose memory
        # holds the pair. Int8 weights are one byte each, so the arithmetic over this
        # checkpoint's widths is: ~0.55 GB of encoder (27 So400m blocks at the padded
        # widths) beside ~2.4 GB of decoder (30 layers: 22 conv, 8 attention), so ~3.0 GB
        # resident. It fits the 4 GiB a PCIe Metis board reports; it does not fit a 1 GB
        # part, and that failure is explicit at load -- the run names both sizes and refuses
        # before anything is dispatched. The host carries a further ~0.52 GB, the fp16
        # embedding table doubling as the tied lm_head, which is host memory and not the
        # device's. `--compile-seq-len` moves only the KV cache (8 attention layers x 2 x
        # 2048 x 512 B is ~17 MB at this context), so it is not a lever on the fit.
        #
        # MEASURED ON CHIP, ON EUROPA -- NOT the part this entry defaults to, which is now
        # Omega; these are the numbers this model has, and a re-measure on Omega is what
        # would replace them rather than a relabel. europa-0:2:0,
        # firmware v1.9.0-a2, 8 cores of EUROPA, context 2048, 512x512 tiles, --num-cal 1.
        # Warm medians from `python -m model_axscript.llm benchmark lfm2.5-vl-3b`.
        #
        #   vision       81.68 ms/picture (12.24 pictures/s), one dispatch, 256 image tokens
        #   decode       13.8 tok/s (60.0 ms device + 12.7 ms host lm_head, 17 %)
        #   TTFT         1449 ms at 2048 positions; 1.62 s end to end, vision included
        #   artifact     606 MB vision + 3431 MB text (+ 3.2 GB quantized bundle, not uploaded)
        #
        # The ENCODER was also checked against the torch W8A8 reference it was quantized
        # from: the real 27-block So400m tower at 512x512, cosine mean 0.9852 with percentile
        # clipping, which is why percentile is the vision tower's default.
        #
        # THE DECLARED SETS ARE BUILD INTENT BEYOND THAT ONE CONFIGURATION. They deliberately
        # match the 450M's, so one --seq-len means the same thing across the family.
        #
        # The BUILD wants host memory, not device memory: the fp32 torch reference is
        # ~11.8 GB on its own (a 5.9 GB bf16 checkpoint upcast) and stays resident through
        # both towers, because `pipeline.build_artifacts` runs `build_vision_graph` AFTER the
        # decoder compile, so peak RSS lands near 14 GB. `--num-cal` is not a lever on that --
        # the sample count does not change the resident model, which is the bulk.
        context_len=DEFAULT_CONTEXT_LEN if context_len is None else int(context_len),
        supported_context_lens=(2048,),
        # 512 is the checkpoint's own tile size, a 32x32 patch grid -> 256 image tokens,
        # and it is what lets a build TILE at all: a tile IS the compiled grid, so a
        # high-resolution picture is cut into copies of it plus a thumbnail.
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
DISPLAY_NAME = "LFM2.5-VL-3B"


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

# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""SmolVLM2 on the family's two-tower build contract.

`model_axscript.llm.shared.vl_entry` drives every VLM in this family through one
`compile_towers(repo_id, out_dir, **knobs) -> {"vision", "text", "bundle"}` call, so that
`axzoo build`, the model's own `__main__.py` and the eval driver all reach the same builder.
SmolVLM2's own builder (`multicore_impl.build_artifact`) predates that contract and speaks
`GenerationConfig`, so this module is the seam between the two: it translates the family's
knobs, refuses the ones SmolVLM2 cannot honour rather than silently dropping them, and
reports the three directories a build wrote.

Two of the family's knobs do NOT apply to this architecture, and saying so here is the
point of the module -- accepting and ignoring them would let a caller believe a build was
made at a geometry or a smoothing setting it was not:

* ``image_size`` is NOT a free knob. Qwen3-VL pins its tower to a patch grid the caller
  chooses; SmolVLM2 instead SPLITS a picture into fixed-size tiles (plus a global
  thumbnail) and its tower is compiled at the tile size baked into the checkpoint -- 512/16
  on the 256M and 500M, 384/14 on the 2.2B. So the value is validated against the
  checkpoint's own geometry and never used to reshape anything.
* ``qk_alpha`` is forced None: SmolLM2 has no QK-norm, so there is no QK smoothing to
  apply (see ``smolvlm2.SmolVLM2Architecture.quantize``).
"""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#: The decoder implementation a build registers, and what lands in the artifact's manifest.
IMPLEMENTATION = "smolvlm2-metis-multicore"

#: How SmolVLM2 makes a picture fit its tower: by tiling, not by resizing to one grid. Named
#: so the family's ``fit_mode`` has a truthful value for this architecture rather than
#: borrowing "stretch"/"letterbox", neither of which its processor does.
FIT_MODE = "tile"


def native_image_size(repo_id: str) -> tuple[int, int]:
    """The tile geometry ``repo_id``'s vision tower is built at, from its HF config alone.

    Config-only on purpose: this is called to VALIDATE a requested geometry before a build
    commits to loading weights, so it must not cost the checkpoint download.
    """
    from transformers import AutoConfig

    vision = AutoConfig.from_pretrained(repo_id).vision_config
    side = int(vision.image_size)
    return (side, side)


def fit_image(config, image):
    """Return the picture this family will show BOTH legs, as RGB. Geometry untouched.

    Every other family in the tree resizes here, because its tower is compiled for one patch
    grid and the reference leg has to be shown the same pixels for the delta to mean
    anything. This family does not have one grid (``FIT_MODE`` above): its processor SPLITS a
    picture into tiles of the size baked into the checkpoint plus a global thumbnail, and
    dispatches the one compiled tile graph once per tile. Both legs load THAT processor out
    of the artifact, so the compiled geometry is already applied to both, identically, and a
    resize here would show them a picture the shipped model never sees -- on a document scan,
    it would throw away exactly the high-resolution crops the question is about and report
    the ANLS that earns.

    So the only thing this normalizes is the MODE. The tower is 3-channel by construction and
    the image processor identifies the channel axis by looking for a 1 or a 3 in the shape, so
    a 4-channel screenshot raises "Unable to infer channel dimension format" from inside
    transformers, naming neither alpha nor the file. Guarded rather than unconditional
    (`convert` copies, and a DocVQA row is already RGB), and a raw HxWx3 array is accepted for
    the same reason the other families accept one.

    `config` is the artifact's reconstructed config, unused for the geometry and taken for the
    signature every family's `fit_image` shares -- the eval resolves this function off the
    model's own config and must not learn a per-family call shape.
    """
    from PIL import Image

    if not hasattr(image, "convert"):
        import numpy as np

        image = Image.fromarray(np.asarray(image, dtype=np.uint8))
    return image if image.mode == "RGB" else image.convert("RGB")


def build_artifacts(
    repo_id: str,
    out_dir,
    *,
    calibration=None,
    num_samples: int | None = None,
    image_size: int | tuple[int, int] | None = None,
    fit_mode: str = FIT_MODE,
    context_len: int = 2048,
    target: str = "device",
    num_cores: int = 4,
    generation: str = "OMEGA",
    smooth_alpha: float = 0.8,
    qk_alpha: float | None = None,
    device: str = "cpu",
) -> dict:
    """Quantize both towers jointly, compile them into ``out_dir``, and say what was written.

    Returns ``{"vision": dir, "text": dir, "bundle": dir}`` -- the family contract, so the
    entry can report a build without knowing which architecture produced it.

    ``num_samples`` is the TOTAL calibration budget, split across DocVQA stills, video
    clips and text-only turns; None takes the recipe's own default. ``calibration``
    overrides the mix with a caller's own sample list (see ``calibration.py``).

    ``device`` is the TORCH device the calibration forwards run on ("cpu" / "cuda"),
    distinct from ``generation`` / ``target`` (the silicon the artifact is compiled for) and
    from the Metis device a run selects. Calibration is the bulk of a build, so a GPU is
    worth having. The compile shape is the same either way, so the artifact is cached under
    one key; the quantization scales come from the calibration forwards, which a CPU and a
    CUDA device do not compute bit-identically.
    """
    from ...generate import GenerationConfig
    from .multicore_impl import _BUNDLE_SUBDIR, _TEXT_SUBDIR, _VISION_SUBDIR, build_artifact

    out_dir = Path(out_dir)

    if fit_mode != FIT_MODE:
        raise ValueError(
            f"SmolVLM2 fits a picture by tiling it, so fit_mode must be {FIT_MODE!r}, not "
            f"{fit_mode!r}. 'stretch' and 'letterbox' describe resizing to a single patch "
            "grid, which is the sibling architecture's geometry, not this one's."
        )
    if qk_alpha is not None:
        raise ValueError(
            f"qk_alpha must be None for SmolVLM2, not {qk_alpha!r}: SmolLM2 has no QK-norm, "
            "so there is no QK smoothing to apply. Passing a value would record a smoothing "
            "setting the build did not use."
        )

    # The geometry is the checkpoint's, so a request is CHECKED rather than applied. Done
    # before the weights load: a mismatch here is a config error, and it should not cost a
    # multi-GB download to hear about it.
    native = native_image_size(repo_id)
    if image_size is not None:
        wanted = (image_size, image_size) if isinstance(image_size, int) else tuple(image_size)
        if wanted != native:
            raise ValueError(
                f"{repo_id} tiles at {native[0]}x{native[1]} (its vision config's image_size), "
                f"so it cannot be built at {wanted[0]}x{wanted[1]}. SmolVLM2's tower is "
                "compiled at the checkpoint's tile size and a picture is split into tiles to "
                "match; the geometry is not a build knob. Declare the native size in the "
                "model's config."
            )

    config = GenerationConfig(
        implementation=IMPLEMENTATION,
        repo_id=repo_id,
        artifact_dir=out_dir,
        context_len=context_len,
        # The prompt (tiles + text) is ingested in one prefill call, as the family's other
        # VLM does; chunked prefill is a text-only path here.
        chunk_len=None,
        num_cores=int(num_cores),
        generation=generation,
        target=target,
        device=device,
        smooth_alpha=smooth_alpha,
    )
    logger.info(
        "building %s at %dx%d tiles, context %d, %d core(s) of %s",
        repo_id,
        native[0],
        native[1],
        context_len,
        num_cores,
        generation,
    )
    build_artifact(
        repo_id,
        config,
        out_dir,
        calib_n=num_samples,
        cores=int(num_cores),
        calibration=calibration,
    )
    return {
        "vision": out_dir / _VISION_SUBDIR,
        "text": out_dir / _TEXT_SUBDIR,
        "bundle": out_dir / _BUNDLE_SUBDIR,
    }


def artifacts(artifact_dir) -> dict:
    """The three directories ``build_artifacts`` writes, as the loaders want them."""
    from .multicore_impl import _BUNDLE_SUBDIR, _TEXT_SUBDIR, _VISION_SUBDIR

    artifact_dir = Path(artifact_dir)
    return {
        "vision": artifact_dir / _VISION_SUBDIR,
        "text": artifact_dir / _TEXT_SUBDIR,
        "bundle": artifact_dir / _BUNDLE_SUBDIR,
    }

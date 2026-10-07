# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Depth Anything V2 Small: a DINOv2-S/14 backbone under a DPT neck and head, as ONE device graph.

The other shape a depth card takes. Where `dinov2_depth_vits14_nyu_linear` composes a backbone and
a head this tree assembles, HuggingFace ships this pair already joined, and the neck taps four
intermediate encoder layers rather than reading only the last one. The taps, the reassemble
convolutions, the bilinear fusions and the head all have offload patterns, so the whole model
compiles through `axten` into a single dispatch; there is no in-tree recipe for it, so this card
has ONE builder.

The tag drops the checkpoint's `-hf` suffix, which marks the transformers FORMAT of the weights
rather than a different model.

The published processor preserves each image's aspect ratio, so every image would reach the model
at its own shape and one compiled graph is one shape. This card pins a square instead
(`DepthAnythingSource.make_processor`). That is not the upstream's own eval protocol, so an
absolute number measured here is not comparable to a published one; the device-versus-fp gap is
unaffected, because both legs read identical pixels.
"""

from pathlib import Path

from ...shared.depth import (  # noqa: F401 - the seams the vision driver calls
    build,
    infer,
    load_device,
    load_reference,
    prepare,
    report_run,
)

MODEL_TAG = "depth_anything_v2_small"

_REPO_ID = "depth-anything/Depth-Anything-V2-Small-hf"

#: 224 is a native grid for the ViT-S/14 backbone (16x16 patches) and the size measured here.
_IMAGE_SIZE = 224


def make_config(artifact_dir, cal_device: str = ""):
    """Per-model build and run config (checkpoint, input size, calibration).

    Shared by the build and the run, so a run reads the same declaration the artifact was
    compiled from. `cal_device` is a compile-only knob (the torch calibration device).
    """
    from ...shared.config import EncoderConfig  # deferred: avoids an import cycle
    from ...shared.depth_anything import DepthAnythingSource  # deferred: torch-dependent

    return EncoderConfig(
        implementation="depth-anything-axten",
        repo_id=_REPO_ID,
        source=DepthAnythingSource(_REPO_ID, image_size=_IMAGE_SIZE),
        load_dtype="float32",
        image_size=_IMAGE_SIZE,
        supported_image_sizes=(_IMAGE_SIZE,),
        artifact_dir=Path(artifact_dir),
        cal_device=cal_device,
        # Indoor depth, for the reason the sibling card's docstring gives: a depth model
        # calibrated on the outdoor-heavy default clips where its own output lives.
        calibration_dataset="tanganke/nyuv2",
    )

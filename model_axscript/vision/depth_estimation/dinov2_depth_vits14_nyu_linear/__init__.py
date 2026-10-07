# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""DINOv2-S/14 plus Meta's NYU linear depth head, as ONE device graph.

The same ViT-S/14 backbone the feature-extraction and segmentation cards ship, under the depth
head Meta released with it. The head is not a linear map, so it is not the head `imported_dino`
can bake: it broadcasts the CLS token onto every patch, upsamples the grid, runs a 1x1 conv to
256 depth bins and reduces those bins to one distance per position. Every one of those ops has an
offload pattern, so the whole card compiles through `axten` into a single dispatch, and this card
therefore has ONE builder rather than a choice of two.

`nyu` records what the head was trained and is scored on (the NYU Depth V2 indoor set), which is
the zoo's naming convention for a depth card; `linear` names which of Meta's two published heads
it is (the other is a DPT decoder).
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

MODEL_TAG = "dinov2_depth_vits14_nyu_linear"

#: The backbone this head was trained on. Names both the checkpoint and the head file.
_BACKBONE = "vits14"
_REPO_ID = "facebook/dinov2-small"

#: Meta's own eval crops NYU to 480x640 and the head runs at the backbone's native grid. 224 is
#: the size measured here: 16x16 patches, upsampled 4x by the head to a 64x64 depth grid before
#: the resize back to the input. A different size needs its own measurement before it is listed.
_IMAGE_SIZE = 224


def make_config(artifact_dir, cal_device: str = ""):
    """Per-model build and run config (checkpoint, head, input size, calibration).

    Shared by the build and the run, so a run reads the same declaration the artifact was
    compiled from. `cal_device` is a compile-only knob (the torch calibration device).
    """
    from ...shared.config import EncoderConfig  # deferred: avoids an import cycle
    from ...shared.dinov2_depth import Dinov2DepthSource  # deferred: torch-dependent

    return EncoderConfig(
        implementation="dinov2-depth-axten",
        repo_id=_REPO_ID,
        source=Dinov2DepthSource(_REPO_ID, _BACKBONE, image_size=_IMAGE_SIZE),
        # Declared here as well as on the source: this is the field that says the card
        # bakes a head, and `--head` moves both through `EncoderSource.with_head`.
        head=_BACKBONE,
        load_dtype="float32",
        image_size=_IMAGE_SIZE,
        supported_image_sizes=(_IMAGE_SIZE,),
        artifact_dir=Path(artifact_dir),
        cal_device=cal_device,
        # Indoor scenes, because that is where this head's depth range lives. Calibrating a depth
        # model on the outdoor-heavy default sets int8 ranges that clip where the output actually
        # is, and no evaluator's scale alignment recovers a clipped value.
        calibration_dataset="tanganke/nyuv2",
    )

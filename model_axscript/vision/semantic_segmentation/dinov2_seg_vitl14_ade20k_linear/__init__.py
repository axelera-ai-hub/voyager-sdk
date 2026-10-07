# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""DINOv2-L/14 plus Meta's ADE20K linear head, as ONE device graph.

The backbone is the same ViT-L/14 the zoo ships on its other backend; the head is a
BatchNorm and a 1x1 conv onto the 150 ADE20K classes, folded into a single linear and baked
into the final kernel. So one dispatch takes pixels to per-token class logits, with no host
head and no second model.

Everything here except the checkpoint and the head file is shared with the other two sizes
(`...shared/ade20k_seg.py`), which is why this package is a configuration and nothing else.
"""

from pathlib import Path

from ...shared.ade20k_seg import (  # noqa: F401 - the seams the vision driver calls
    build,
    infer,
    load_device,
    load_reference,
    prepare,
    report_run,
)

MODEL_TAG = "dinov2_seg_vitl14_ade20k_linear"

#: The backbone this head was trained on. Names both the checkpoint and the head file.
_BACKBONE = "vitl14"
_REPO_ID = "facebook/dinov2-large"


def make_config(artifact_dir, cal_device: str = ""):
    """Per-model build and run config (checkpoint, head, input size, calibration).

    Shared by the build and the run, so a run reads the same declaration the artifact was
    compiled from. `cal_device` is a compile-only knob (the torch calibration device).
    """
    from ...shared.config import EncoderConfig  # deferred: avoids an import cycle
    from ...shared.hf import HFSource  # deferred: torch-dependent

    return EncoderConfig(
        implementation="dinov2-axten",
        repo_id=_REPO_ID,
        source=HFSource(_REPO_ID),
        head=_BACKBONE,
        load_dtype="float32",
        # 518 is DINOv2's native resolution and the one the head was trained at: 37x37 patches
        # rather than 16x16, which is the class map a segmentation consumer upsamples.
        image_size=518,
        supported_image_sizes=(518,),
        artifact_dir=Path(artifact_dir),
        cal_device=cal_device,
    )

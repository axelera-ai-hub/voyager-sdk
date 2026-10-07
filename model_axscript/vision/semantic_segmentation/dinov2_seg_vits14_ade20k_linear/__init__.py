# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""DINOv2-S/14 plus Meta's ADE20K linear head, as ONE device graph.

The backbone is the same ViT-S/14 the feature-extraction card ships; the head is a
BatchNorm and a 1x1 conv onto the 150 ADE20K classes, folded into a single linear and baked
into the final kernel. So one dispatch takes pixels to per-token class logits, with no host
head and no second model.

The recipe is built on the DINOv2 implementation under `model_axscript/llm/models/dino/`
rather than on a copy of it, reached through this family's own `EncoderSource`: on the same
checkpoint, calibration set and device, that implementation agrees with the fp reference at
CLS cosine 0.98 where the copy reaches 0.92.
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

MODEL_TAG = "dinov2_seg_vits14_ade20k_linear"

#: The backbone this head was trained on. Names both the checkpoint and the head file.
_BACKBONE = "vits14"
_REPO_ID = "facebook/dinov2-small"


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
        # rather than 16x16, which is the class map a segmentation consumer upsamples. 224 is
        # measured too and costs less, at a fifth of the map.
        image_size=518,
        supported_image_sizes=(224, 518),
        artifact_dir=Path(artifact_dir),
        cal_device=cal_device,
    )

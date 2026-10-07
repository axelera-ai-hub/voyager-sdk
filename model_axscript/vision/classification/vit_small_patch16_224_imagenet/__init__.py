# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""ViT-S/16 (`vit_small_patch16_224.augreg_in21k_ft_in1k`) on the AxScript (ALTO-direct) path.

The same model the zoo already ships on the axgraph backend, brought up as a second backend on
card MC-201. It reuses the shared vision encoder recipe: no new ALTO kernel, a timm weight source
(`TimmViTSource`) for the upstream, and the checkpoint's own ImageNet classifier as a host stage
(`ImageNetHead`).

Everything except the checkpoint is shared with the other two patch-16 sizes
(`...shared/imagenet_vit.py`), which is why this package is a configuration and nothing else.
Like every model package it stays import-light: the shared seams reach the torch-dependent
driver only from inside their bodies, so a caller that just wants `MODEL_TAG` (the registry, a
listing) does not pay for torch.
"""

from pathlib import Path

from ...shared.imagenet_vit import (  # noqa: F401 - the seams the vision driver calls
    build,
    infer,
    load_device,
    load_reference,
    prepare,
    report_run,
)

# Authored, not derived: this card joins the EXISTING zoo card for the same model, so the tag is
# the one that card already publishes (see axelera_zoo configs/classification/). Deriving it from
# the timm checkpoint id would mint a second identity for one model.
MODEL_TAG = "vit_small_patch16_224-imagenet"

REPO_ID = "timm/vit_small_patch16_224.augreg_in21k_ft_in1k"


def make_config(artifact_dir, cal_device: str = ""):
    """Per-model build+run config (implementation, checkpoint, dtypes, image size).

    Shared by the build and the run, so a run reads the same declaration the artifact was
    compiled from. ``cal_device`` is a compile-only knob (the torch calibration device); the
    runtime path ignores it.
    """
    from ...shared.config import EncoderConfig  # deferred: avoids an import cycle
    from ...shared.timm_vit import TimmViTSource  # deferred: torch-dependent

    return EncoderConfig(
        implementation="vit-axten",
        repo_id=REPO_ID,
        source=TimmViTSource(REPO_ID),
        load_dtype="float32",
        image_size=224,
        artifact_dir=Path(artifact_dir),
        cal_device=cal_device,
    )

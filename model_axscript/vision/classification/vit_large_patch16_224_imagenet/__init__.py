# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""ViT-L/16 (`vit_large_patch16_224.augreg_in21k_ft_in1k`) on the AxScript (ALTO-direct) path.

The same model the zoo already ships on the axgraph backend, brought up as a second backend on
the card it already has. It reuses the shared vision encoder recipe: no new ALTO kernel, a timm
weight source (`TimmViTSource`) for the upstream, and the checkpoint's own ImageNet classifier
as a host stage (`ImageNetHead`).

Everything except the checkpoint is shared with the other two patch-16 sizes
(`...shared/imagenet_vit.py`): 24 blocks of width 1024 here, read off the checkpoint
rather than declared, which is why this package is a configuration and nothing else.
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
MODEL_TAG = "vit_large_patch16_224-imagenet"

REPO_ID = "timm/vit_large_patch16_224.augreg_in21k_ft_in1k"


def make_config(artifact_dir, cal_device: str = ""):
    """Per-model build+run config (implementation, checkpoint, dtypes, image size).

    Shared by the build and the run, so a run reads the same declaration the artifact was
    compiled from. ``cal_device`` is a compile-only knob (the torch calibration device); the
    runtime path ignores it.
    """
    from ...shared.config import EncoderConfig  # deferred: avoids an import cycle
    from ...shared.timm_vit import TimmViTSource

    return EncoderConfig(
        implementation="vit-axten",
        repo_id=REPO_ID,
        source=TimmViTSource(REPO_ID),
        load_dtype="float32",
        image_size=224,
        # TWO cores, where every other card in this family takes the default four. axten
        # N-shards the fc1 matmul across `min(cores, 3)` cores, and ViT-L's 4096-wide MLP is 64
        # blocks of 64: 64 is not divisible by 3, so a four-core build is refused at compile time
        # with "split_output requires N_BLOCKS (64) divisible by cores (3)". Two cores makes the
        # shard count divide the block count and compiles clean. The `min(cores, 3)` choice is the
        # compiler's, so lifting this back to four is a change on that side, not here.
        shard=2,
        artifact_dir=Path(artifact_dir),
        cal_device=cal_device,
    )

# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Image calibration set for DINOv2 PTQ -- the vision analog of llm/calibration.py.

ViT PTQ is calibrated on a small set of real, in-distribution natural images
(unlabeled); the standard is a random sample of ImageNet-1k train (~32-256 images
is plenty for min/max + SmoothQuant observers). DINOv2 was trained on the private
LVD-142M, so ImageNet is the representative proxy. Images are preprocessed with the
model's own eval transform (the HF image processor: resize/center-crop + ImageNet
normalization) -- exactly what inference sees.

CRITICAL: the calibration images must have the resolution/detail of the DEPLOYMENT input, so
they must be DOWNSCALED to the model's input res, not upscaled. The old default,
``zh-plus/tiny-imagenet``, is 64px -> upscaling it to 518 produces smooth, low-frequency images
whose activation VARIANCES are far below real native-518 images. That under-covered the recorded
LayerNorm inv_sqrt ranges, so realistic @518 images went out-of-range and blew up (a ±10^4 runaway
via the inv_sqrt LUT's out-of-range extrapolation -- see llm/luts.py). So the default is now
``sayakpaul/coco-30-val-2014`` (30k diverse natural images, ~454px median): downscaled to 518 they
retain real detail and their variances match deployment. Point ``dataset`` at ImageNet if gated
access is available. (For @224 either source is fine; the mismatch only bites at high res.)
"""

from __future__ import annotations

import torch

# 30k diverse natural images at ~454px median -> downscaled (not upscaled) to the input res.
_DEFAULT_DATASET = "sayakpaul/coco-30-val-2014"


def load_calibration_images(n: int = 32, *, dataset: str = _DEFAULT_DATASET, split: str = "train"):
    """Return n real RGB PIL images sampled from a natural-image dataset."""
    from datasets import load_dataset

    # Over-fetch a little so we can skip any non-RGB / decode failures.
    ds = load_dataset(dataset, split=f"{split}[:{max(n * 2, n + 8)}]")
    images = []
    for example in ds:
        img = example.get("image") if isinstance(example, dict) else example["image"]
        try:
            images.append(img.convert("RGB"))
        except Exception:  # noqa: BLE001 - skip undecodable samples
            continue
        if len(images) >= n:
            break
    if not images:
        raise RuntimeError(f"no calibration images loaded from {dataset!r}")
    return images


def make_processor(repo_id: str, image_size: int | None = None):
    """The model's HF image processor, optionally overridden to a target square size (the
    DINOv2 processor defaults to a 224 crop; pass e.g. 518 for native resolution)."""
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(repo_id)
    if image_size is not None:
        processor.size = {"shortest_edge": image_size}
        processor.crop_size = {"height": image_size, "width": image_size}
    return processor


def image_calibration(
    repo_id: str,
    *,
    n: int = 32,
    dataset: str = _DEFAULT_DATASET,
    split: str = "train",
    image_size: int | None = None,
) -> list[torch.Tensor]:
    """Preprocessed pixel_values [(1, 3, H, W), ...] for n real images, using the model's
    own HF image processor (same transform as inference). image_size overrides the crop size."""
    processor = make_processor(repo_id, image_size)
    images = load_calibration_images(n, dataset=dataset, split=split)
    return [processor(images=img, return_tensors="pt").pixel_values for img in images]

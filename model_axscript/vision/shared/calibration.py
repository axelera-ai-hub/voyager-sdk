# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Image calibration set for DINOv2 PTQ -- the vision analog of llm/shared/calibration.py.

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
LayerNorm inv_sqrt ranges, so realistic @518 images went out-of-range and blew up (a +-10^4 runaway
via the inv_sqrt LUT's out-of-range extrapolation -- see shared/luts.py). So the default is now
``sayakpaul/coco-30-val-2014`` (30k diverse natural images, ~454px median): downscaled to 518 they
retain real detail and their variances match deployment. Point ``dataset`` at ImageNet if gated
access is available. (For @224 either source is fine; the mismatch only bites at high res.)
"""

from __future__ import annotations

from pathlib import Path

import torch

# 30k diverse natural images at ~454px median -> downscaled (not upscaled) to the input res.
_DEFAULT_DATASET = "sayakpaul/coco-30-val-2014"


_IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".ppm", ".tif", ".tiff")


def _images_from_dir(directory: Path, n: int) -> list:
    """Return up to n RGB images from a local directory, recursively, sorted for repeatability.

    Calibration data is domain data, and a customer's images sit on disk far more often than on
    a hub. Sorted rather than shuffled so two builds of the same directory observe the same
    scales; take a subset yourself if you want a different sample.
    """
    from PIL import Image

    files = sorted(p for p in directory.rglob("*") if p.suffix.lower() in _IMAGE_SUFFIXES)
    images = []
    for path in files:
        try:
            with Image.open(path) as img:
                images.append(img.convert("RGB"))
        except Exception:  # noqa: BLE001 - skip an unreadable file, the set is a sample
            continue
        if len(images) >= n:
            break
    if not images:
        raise RuntimeError(
            f"no readable images in {directory} (looked for {', '.join(_IMAGE_SUFFIXES)})"
        )
    return images


def as_pil(value):
    """One dataset row's image as a PIL image, whether the column holds one or an array.

    A hub dataset does not always store JPEG bytes. `tanganke/nyuv2` stores a decoded CHW float
    tensor, which is what an indoor-depth calibration set is available as, and the plain
    `.convert("RGB")` path skipped every row of it in silence and then reported that the set was
    empty. Floats are read as the 0..1 range every such column uses; an integer array is taken as
    8-bit already.
    """
    from PIL import Image

    if hasattr(value, "convert"):  # already a PIL image
        return value

    import numpy as np

    array = np.asarray(value)
    if array.ndim != 3:
        raise ValueError(f"expected an image with 3 axes, got shape {array.shape}")
    if array.shape[0] in (1, 3) and array.shape[-1] not in (1, 3):
        array = np.transpose(array, (1, 2, 0))  # CHW -> HWC
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    if np.issubdtype(array.dtype, np.floating):
        array = np.clip(array, 0.0, 1.0) * 255.0
    return Image.fromarray(array.astype(np.uint8))


def load_calibration_images(n: int = 32, *, dataset: str = _DEFAULT_DATASET, split: str = "train"):
    """Return n real RGB PIL images, from a local directory or a HuggingFace dataset id.

    A path that exists on disk is read directly, so pointing `calibration_dataset` at your own
    images needs no hub account and no upload. Anything else is treated as a hub id.
    """
    local = Path(dataset).expanduser()
    if local.is_dir():
        return _images_from_dir(local, n)

    from datasets import load_dataset

    # Over-fetch a little so we can skip any non-RGB / decode failures.
    ds = load_dataset(dataset, split=f"{split}[:{max(n * 2, n + 8)}]")
    images = []
    for example in ds:
        img = example.get("image") if isinstance(example, dict) else example["image"]
        try:
            images.append(as_pil(img).convert("RGB"))
        except Exception:  # noqa: BLE001 - skip undecodable samples
            continue
        if len(images) >= n:
            break
    if not images:
        raise RuntimeError(f"no calibration images loaded from {dataset!r}")
    return images


def make_processor(repo_id: str, image_size: int | None = None):
    """The model's HF image processor, optionally overridden to a target square size (the
    DINOv2 processor defaults to a 224 crop; pass e.g. 518 for native resolution).

    HF-specific helper used by `HFSource.make_processor`; the generic flow gets its
    processor via the configured `EncoderSource`, not this function directly."""
    from transformers import AutoImageProcessor

    processor = AutoImageProcessor.from_pretrained(repo_id)
    if image_size is not None:
        processor.size = {"shortest_edge": image_size}
        processor.crop_size = {"height": image_size, "width": image_size}
    return processor


def image_calibration(
    source,
    *,
    n: int = 32,
    dataset: str = _DEFAULT_DATASET,
    split: str = "train",
    image_size: int | None = None,
) -> list[torch.Tensor]:
    """Preprocessed pixel_values [(1, 3, H, W), ...] for n real images, using the source's
    own preprocessing (same transform as inference). image_size overrides the crop size."""
    processor = source.make_processor(image_size)
    images = load_calibration_images(n, dataset=dataset, split=split)
    return [processor(images=img, return_tensors="pt").pixel_values for img in images]

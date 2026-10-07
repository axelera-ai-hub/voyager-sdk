# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""kNN classification accuracy for the on-device DINOv2 vs the HF fp reference.

The vision analog of the LLM eval stack: DINOv2 is a frozen feature extractor with no
output distribution (no perplexity), so its accuracy is measured on the features. kNN is
the standard, training-free DINOv2 metric -- extract CLS features for a labeled index set,
nearest-neighbor classify a val set, report top-1. We run it twice (device int8 vs HF fp)
and report both + the quantization gap.

Cost is one batch-1 `encode()` dispatch per image (~60 ms base, no batching), so the CLI
offers a `quick` proxy (2000 index / 500 val, ~a few min) and a `full` run (10000/2000,
~15 min). The device-vs-fp GAP converges far faster than absolute top-1; absolute numbers
are low because tiny-imagenet (the self-contained, labelled ImageNet-200 proxy; real
ImageNet-1k is gated) is 64px upscaled to 224 -- the gap is the meaningful quantity.

The fp reference runs on GPU (falls back to CPU) so it doesn't bottleneck the device pass.

    python -m tests.integration.alto.llm.models.dino.knn_eval --repo-id facebook/dinov2-small
    python -m tests.integration.alto.llm.models.dino.knn_eval --mode full --repo-id facebook/dinov2-base --n-calib 64
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

_DATASET = "zh-plus/tiny-imagenet"
_MODES = {"quick": (2000, 500), "full": (10000, 2000)}


def load_labeled_images(n: int, split: str, *, dataset: str = _DATASET, seed: int = 0):
    """Return (images, labels): n shuffled RGB PIL images + int class labels from `split`."""
    from datasets import load_dataset

    ds = load_dataset(dataset, split=split).shuffle(seed=seed).select(range(n))
    images = [ex["image"].convert("RGB") for ex in ds]
    labels = np.array([int(ex["label"]) for ex in ds], dtype=np.int64)
    return images, labels


def _pixels(processor, image):
    return processor(images=image, return_tensors="pt").pixel_values


@torch.no_grad()
def extract_device_cls(dev, images, processor) -> np.ndarray:
    """CLS features (N, hidden) from the on-device DinoV2Device (one dispatch per image)."""
    feats = [dev.encode(_pixels(processor, img))[0, 0] for img in images]
    return np.stack(feats).astype(np.float32)


@torch.no_grad()
def extract_fp_cls(hf_model, images, processor, device: str) -> np.ndarray:
    """CLS features (N, hidden) from the HF fp model (on `device`, e.g. cuda)."""
    hf_model = hf_model.to(device)
    feats = []
    for img in images:
        out = hf_model(pixel_values=_pixels(processor, img).to(device)).last_hidden_state
        feats.append(out[0, 0].float().cpu().numpy())
    hf_model.to("cpu")
    return np.stack(feats).astype(np.float32)


def knn_top1(
    index_f: np.ndarray,
    index_y: np.ndarray,
    val_f: np.ndarray,
    val_y: np.ndarray,
    *,
    k: int = 20,
    temperature: float = 0.07,
) -> float:
    """DINO-style cosine kNN top-1: temperature-weighted vote over the k nearest neighbours."""
    idx = index_f / (np.linalg.norm(index_f, axis=1, keepdims=True) + 1e-9)
    val = val_f / (np.linalg.norm(val_f, axis=1, keepdims=True) + 1e-9)
    sims = val @ idx.T  # (V, N) cosine
    knn = np.argsort(-sims, axis=1)[:, :k]
    n_classes = int(max(index_y.max(), val_y.max())) + 1
    correct = 0
    for i in range(len(val_f)):
        w = np.exp(sims[i, knn[i]] / temperature)
        votes = np.zeros(n_classes, dtype=np.float64)
        np.add.at(votes, index_y[knn[i]], w)
        correct += int(votes.argmax() == val_y[i])
    return correct / len(val_f)


def evaluate_knn(
    repo_id: str = "facebook/dinov2-small",
    *,
    n_index: int = 2000,
    n_val: int = 500,
    k: int = 20,
    n_calib: int = 16,
    cores: int = 4,
    out_dir: Path | None = None,
    fp_device: str | None = None,
    seed: int = 0,
    cache_dir: Path | None = None,
) -> dict:
    """Build the device model, extract device + fp CLS features for the index/val sets, and
    return {device_top1, fp_top1, gap, cls_cosine, ...}."""
    from transformers import AutoImageProcessor

    from .model import load_dino_device

    if fp_device is None:
        fp_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reuse a cached compiled+calibrated artifact if present (skips the build); want_fp gives the
    # fp reference the kNN comparison needs.
    dev, hf_model = load_dino_device(
        repo_id,
        cache_dir=cache_dir or out_dir,
        cores=cores,
        n_calib=n_calib,
        calib_device=fp_device,
        want_fp=True,
    )
    processor = AutoImageProcessor.from_pretrained(repo_id)

    idx_imgs, idx_y = load_labeled_images(n_index, "train", seed=seed)
    val_imgs, val_y = load_labeled_images(n_val, "valid", seed=seed)

    dev_idx = extract_device_cls(dev, idx_imgs, processor)
    dev_val = extract_device_cls(dev, val_imgs, processor)
    fp_idx = extract_fp_cls(hf_model, idx_imgs, processor, fp_device)
    fp_val = extract_fp_cls(hf_model, val_imgs, processor, fp_device)

    dev_top1 = knn_top1(dev_idx, idx_y, dev_val, val_y, k=k)
    fp_top1 = knn_top1(fp_idx, idx_y, fp_val, val_y, k=k)
    # CLS feature fidelity on the val set (the cheap intrinsic proxy).
    cos = np.sum(dev_val * fp_val, 1) / (
        np.linalg.norm(dev_val, axis=1) * np.linalg.norm(fp_val, axis=1) + 1e-9
    )
    return {
        "repo_id": repo_id,
        "n_index": n_index,
        "n_val": n_val,
        "k": k,
        "device_top1": dev_top1,
        "fp_top1": fp_top1,
        "gap": fp_top1 - dev_top1,
        "cls_cosine_mean": float(cos.mean()),
    }


def main(argv=None) -> dict:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default="facebook/dinov2-small")
    p.add_argument("--mode", default="quick", choices=list(_MODES))
    p.add_argument("--n-index", type=int, default=None)
    p.add_argument("--n-val", type=int, default=None)
    p.add_argument("--k", type=int, default=20)
    p.add_argument("--n-calib", type=int, default=16)
    p.add_argument("--cores", type=int, default=4)
    p.add_argument("--fp-device", default=None)
    p.add_argument(
        "--cache-dir",
        default=None,
        help="reuse/build the compiled artifact here (default: keyed cache under "
        "$AXELERA_DINO_CACHE or ~/.cache/axelera_dino)",
    )
    args = p.parse_args(argv)

    n_index, n_val = _MODES[args.mode]
    res = evaluate_knn(
        args.repo_id,
        n_index=args.n_index or n_index,
        n_val=args.n_val or n_val,
        k=args.k,
        n_calib=args.n_calib,
        cores=args.cores,
        fp_device=args.fp_device,
        cache_dir=args.cache_dir,
    )
    print(
        f"{res['repo_id']} kNN(k={res['k']}) on {res['n_index']} index / {res['n_val']} val: "
        f"device top-1 {res['device_top1']:.4f} | fp top-1 {res['fp_top1']:.4f} | "
        f"gap {res['gap']:+.4f} | CLS cosine {res['cls_cosine_mean']:.4f}"
    )
    return res


if __name__ == "__main__":
    main()

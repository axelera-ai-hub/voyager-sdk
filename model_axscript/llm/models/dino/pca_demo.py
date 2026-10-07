# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""PCA-of-patch-features demo for DINOv2 -- the iconic self-supervised-ViT visualization,
run through BOTH the fp reference and our int8 on-device model to show the quantized,
single-dispatch device model preserves the rich features.

The top-3 PCA components of the patch tokens are mapped to RGB. A single PCA basis is fit
on the fp features across all demo images and applied to BOTH fp and device, so: (a) the
same colour means the same feature direction everywhere -> object parts get consistent
colours across images (DINOv2's part-correspondence), and (b) fp vs device use identical
colouring, so if quantization preserved the features the two columns look the same.

    python -m tests.integration.alto.llm.models.dino.pca_demo --repo-id facebook/dinov2-base \
        --n-calib 256 --cores 4 --out /tmp/dinov2_pca_demo.png
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch


def _demo_images(n: int, seed: int = 0):
    """n high-res RGB pet photos (cats + dogs) -- part-correspondence shows across them."""
    from datasets import load_dataset

    ds = load_dataset("microsoft/cats_vs_dogs", split="train").shuffle(seed=seed).select(range(n))
    return [ex["image"].convert("RGB") for ex in ds]


def _display_rgb(image, size: int):
    """Resize + center-crop a PIL image to size x size for display (matches the patch grid)."""
    from PIL import Image

    w, h = image.size
    s = size / min(w, h)
    image = image.resize((round(w * s), round(h * s)), Image.BICUBIC)
    w, h = image.size
    left, top = (w - size) // 2, (h - size) // 2
    return np.asarray(image.crop((left, top, left + size, size + top))) / 255.0


def _pca_fit(x: np.ndarray, k: int = 3):
    mean = x.mean(0)
    _, _, vt = np.linalg.svd(x - mean, full_matrices=False)
    return mean, vt[:k]


def _l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def _to_rgb(proj: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    return np.clip((proj - lo) / (hi - lo + 1e-9), 0, 1)


def _foreground_mask(proj: np.ndarray, grid: int) -> np.ndarray:
    """Clean foreground mask (1=object, 0=background) from the first PCA component (which
    separates fg/bg in DINOv2). The background level is the image-border mean; the object is
    the PC1 side the image CENTER leans toward (fixes the arbitrary PC1 sign). A morphological
    close + interior hole-fill + largest-connected-component then makes the object SOLID -- so
    low-contrast interior patches (e.g. uniform dark fur, whose PC1 sits near background) don't
    punch black holes inside the animal."""
    from scipy import ndimage

    pc1 = proj[:, 0].reshape(grid, grid)
    border = np.concatenate([pc1[0], pc1[-1], pc1[:, 0], pc1[:, -1]])
    bg = float(border.mean())
    q = grid // 4
    center = float(pc1[q : grid - q, q : grid - q].mean())
    sign = 1.0 if center > bg else -1.0
    fg = (pc1 - bg) * sign > 0.0  # object side of the background level
    fg = ndimage.binary_closing(fg, iterations=1)
    fg = ndimage.binary_fill_holes(fg)
    labels, n = ndimage.label(fg)
    if n > 1:  # drop stray background patches on the object side
        sizes = ndimage.sum(np.ones_like(labels), labels, index=range(1, n + 1))
        fg = labels == (1 + int(np.argmax(sizes)))
    return fg.astype(np.float32).ravel()


@torch.no_grad()
def _fp_patches(hf_model, pixel_values, device):
    out = hf_model(pixel_values=pixel_values.to(device)).last_hidden_state
    return out[0, 1:].float().cpu().numpy()  # drop CLS -> (num_patches, hidden)


def run_demo(
    repo_id="facebook/dinov2-base",
    *,
    n_images=6,
    n_calib=256,
    cores=4,
    out="dinov2_pca_demo.png",
    seed=0,
    work_dir=None,
    image_size=None,
    fg_keep=0.6,
    cache_dir=None,
):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .calibration import make_processor
    from .model import load_dino_device

    fp_device = "cuda" if torch.cuda.is_available() else "cpu"

    # Reuses a cached compiled+calibrated artifact if present (skips the build); want_fp loads the
    # fp reference the PCA basis + comparison column need.
    dev, hf_model = load_dino_device(
        repo_id,
        cache_dir=cache_dir,
        cores=cores,
        n_calib=n_calib,
        image_size=image_size,
        calib_device=fp_device,
        want_fp=True,
    )
    processor = make_processor(repo_id, image_size)
    hidden = dev.hidden
    grid = dev.grid  # patches per side (16 for 224px)
    crop = dev.image_size

    images = _demo_images(n_images, seed)
    disp, fp_feats, dev_feats = [], [], []
    hf_model = hf_model.to(fp_device)
    for img in images:
        px = processor(images=img, return_tensors="pt").pixel_values  # (1,3,crop,crop)
        disp.append(_display_rgb(img, crop))
        fp_feats.append(_fp_patches(hf_model, px, fp_device))  # (P, hidden)
        dev_feats.append(dev.encode(px)[0, 1:].astype(np.float32))  # device patch tokens
    hf_model.to("cpu")

    # Mask basis: PCA on RAW fp patches -- PC0 carries the fg/bg magnitude split the mask needs.
    fp_all = np.concatenate(fp_feats, 0)
    mask_mean, mask_comps = _pca_fit(fp_all, 3)
    # Per-image foreground masks (threshold the first PCA component). fg_keep=1.0 disables masking.
    masks = (
        None
        if fg_keep >= 1.0
        else [_foreground_mask((f - mask_mean) @ mask_comps.T, grid) for f in fp_feats]
    )
    # Colour basis: the standard DINOv2-PCA refinement -- re-run PCA on the FOREGROUND patches
    # ONLY (thresholded above). The background no longer dominates the components, so the object's
    # colours get much crisper. Fit on L2-NORMALIZED features (direction only -> low-texture
    # patches don't collapse to one colour). One basis over all fp fg patches -> shared colours
    # across images + fp/device comparability.
    fg_all = (
        fp_all if masks is None else np.concatenate([f[m > 0] for f, m in zip(fp_feats, masks)], 0)
    )
    col_mean, col_comps = _pca_fit(_l2norm(fg_all), 3)
    col_proj = (_l2norm(fg_all) - col_mean) @ col_comps.T
    lo, hi = np.percentile(col_proj, 2, 0), np.percentile(col_proj, 98, 0)

    def rgb_map(feats, mask=None):
        proj = (_l2norm(feats) - col_mean) @ col_comps.T
        rgb = _to_rgb(proj, lo, hi).reshape(grid, grid, 3)
        if mask is not None:  # black out background so its amplified texture doesn't render
            rgb = rgb * mask.reshape(grid, grid, 1)
        return rgb

    # Render: rows = images, cols = [original, fp PCA, int8 device PCA].
    n = len(images)
    fig, axes = plt.subplots(n, 3, figsize=(9, 3 * n))
    axes = np.atleast_2d(axes)
    col_titles = [
        "input",
        "DINOv2 fp\n(PCA of patch features)",
        "DINOv2 int8 on Metis\n(single-dispatch device)",
    ]
    for j, t in enumerate(col_titles):
        axes[0, j].set_title(t, fontsize=12)
    for i in range(n):
        # Foreground mask (computed above) applied to BOTH columns so they stay comparable.
        mask = None if masks is None else masks[i]
        panels = [disp[i], rgb_map(fp_feats[i], mask), rgb_map(dev_feats[i], mask)]
        for j, p in enumerate(panels):
            ax = axes[i, j]
            ax.imshow(p if j == 0 else np.asarray(_upsample(p, crop)))
            ax.set_xticks([])
            ax.set_yticks([])
    fig.suptitle(
        f"{repo_id.split('/')[-1]}  —  patch-feature PCA: fp vs int8 on-device (Metis)",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.98))
    out = Path(out)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out}  ({n} images, grid {grid}x{grid}, hidden {hidden})")
    return out


def _upsample(rgb_grid: np.ndarray, size: int):
    """Bilinear-upsample a (g,g,3) PCA map to (size,size,3) for overlay."""
    from PIL import Image

    im = Image.fromarray((np.clip(rgb_grid, 0, 1) * 255).astype(np.uint8))
    return np.asarray(im.resize((size, size), Image.BILINEAR)) / 255.0


_DEMO_DIR = Path(__file__).parent / "demo"


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default="facebook/dinov2-base")
    p.add_argument("--n-images", type=int, default=6)
    p.add_argument("--n-calib", type=int, default=256)
    p.add_argument("--cores", type=int, default=4)
    p.add_argument("--out", default=str(_DEMO_DIR / "dinov2_pca_demo.png"))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="override input resolution (e.g. 518 for DINOv2 native; default 224 crop)",
    )
    p.add_argument(
        "--fg-keep",
        type=float,
        default=0.6,
        help="<1.0 masks the background to black (clean object segmentation via PC1 + "
        "hole-fill); 1.0 disables the mask (show raw PCA on every patch)",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="reuse/build the compiled artifact here (default: keyed cache under "
        "$AXELERA_DINO_CACHE or ~/.cache/axelera_dino)",
    )
    args = p.parse_args(argv)
    run_demo(
        args.repo_id,
        n_images=args.n_images,
        n_calib=args.n_calib,
        cores=args.cores,
        out=args.out,
        seed=args.seed,
        image_size=args.image_size,
        fg_keep=args.fg_keep,
        cache_dir=args.cache_dir,
    )


if __name__ == "__main__":
    main()

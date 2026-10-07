# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Real-time video PCA-RGB demo for DINOv2 on Metis -- the moving-picture version of
``pca_demo.py`` (the iconic self-supervised-ViT visualization), run frame-by-frame through
our int8, single-dispatch on-device model with a live FPS/latency HUD.

Each output frame is ``input | DINOv2 int8 PCA (on Metis)`` (add ``--show-fp`` for a middle fp
column). The patch tokens' top-3 PCA components map to RGB with ONE basis fit across all
frames, so colours are temporally stable and the same colour = the same feature direction --
object parts keep consistent colours as the animal moves (DINOv2 part-correspondence). The
background is masked to black (like the reference). Every frame is timed on the device, and the
HUD shows the measured FPS + ms/frame; the GIF is written with per-frame durations equal to the
measured device latency, so it plays back at the model's *actual* speed (pre-rendered, but the
timing is real).

    python -m tests.integration.alto.llm.models.dino.video_demo \
        --repo-id facebook/dinov2-base --n-calib 256 --cores 4 --n-frames 60

With no --video it fetches a CC-BY clip (fluffy kitten, (c) Nesnad, Wikimedia Commons).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time

import numpy as np
import torch

from .pca_demo import _display_rgb, _l2norm, _pca_fit, _to_rgb

# Default demo clip: single fluffy kitten filling the frame (close-up motion, clean bg).
# "Kitten playing - Tokyo - Jan 7 2020", (c) Nesnad, CC BY 4.0, via Wikimedia Commons.
_DEFAULT_CLIP_URL = (
    "https://upload.wikimedia.org/wikipedia/commons/f/f3/"
    "Kitten_playing_-_Tokyo_-_Jan_7_2020.webm"
)
_DEFAULT_CLIP_CREDIT = "Kitten clip (c) Nesnad, CC BY 4.0 (Wikimedia Commons)"
_DEMO_DIR = Path(__file__).parent / "demo"


def _fetch_default_clip() -> Path:
    """Download the default CC-BY demo clip (cached under the artifact cache root, one-time)."""
    import urllib.request

    from .model import default_cache_root

    dst = default_cache_root() / "clips" / "kitten_tokyo.webm"
    if dst.exists() and dst.stat().st_size > 0:
        return dst
    dst.parent.mkdir(parents=True, exist_ok=True)
    req = urllib.request.Request(_DEFAULT_CLIP_URL, headers={"User-Agent": "dino-demo/1.0"})
    with urllib.request.urlopen(req) as r, open(dst, "wb") as f:  # noqa: S310 (trusted host)
        f.write(r.read())
    print(f"fetched demo clip -> {dst}  [{_DEFAULT_CLIP_CREDIT}]")
    return dst


def _read_frames(path, *, start=0, n=60, stride=1):
    """Decode up to n RGB frames from a video (cv2), starting at frame `start`, taking every
    `stride`-th frame. Returns (frames[list of HxWx3 uint8 RGB], source_fps)."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"could not open video {path!r}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start)
    frames, idx = [], 0
    while len(frames) < n:
        ok, bgr = cap.read()
        if not ok:
            break
        if idx % stride == 0:
            frames.append(bgr[:, :, ::-1].copy())  # BGR -> RGB
        idx += 1
    cap.release()
    if not frames:
        raise RuntimeError(f"no frames decoded from {path!r}")
    return frames, fps


def _fg_mask(pc0_grid, k=0.0):
    """Robust foreground mask (1=object, 0=background) from the first mask-PCA component.

    Two-sided: an object can sit on EITHER side of the background level (a bright OR a dark
    subject both deviate from bland background), so we threshold |PC0 - bg| -- unlike a one-sided
    split this catches e.g. a black and a white dog at once. A LOW threshold grabs the textured
    OUTLINE (feather/fur/edge patches), then fill-holes fills the enclosed uniform interior (a
    solid blue bird body / dark fur, whose own features are low-information and near background),
    and the largest connected component drops thin background structure (cage bars, fences)."""
    from scipy import ndimage

    border = np.concatenate([pc0_grid[0], pc0_grid[-1], pc0_grid[:, 0], pc0_grid[:, -1]])
    bg = float(np.median(border))
    dev = np.abs(pc0_grid - bg)
    mad = np.median(np.abs(dev - np.median(dev))) + 1e-6
    fg = dev > (np.median(dev) + k * 1.4826 * mad)
    fg = ndimage.binary_dilation(fg, iterations=1)
    fg = ndimage.binary_fill_holes(fg)
    fg = ndimage.binary_erosion(fg, iterations=1)
    lab, n = ndimage.label(fg)
    if n > 1:  # keep the largest blob = the subject; drop thin cage/fence structure
        sizes = ndimage.sum(np.ones_like(lab), lab, index=range(1, n + 1))
        fg = lab == (1 + int(np.argmax(sizes)))
    return fg.astype(np.float32).ravel()


def _fg_mask_kmeans(feats, grid):
    """Foreground mask by 2-means clustering of the L2-normalized patch features.

    More robust than thresholding a single PCA component when the background is *feature-textured*
    (e.g. grass): there the object/background split is NOT the top variance axis (the majority-area
    background dominates every top PC), so no single component isolates the object -- but the two
    feature CLUSTERS still separate object-from-background cleanly (a standard DINOv2 segmentation
    trick). Foreground = the cluster that is LESS present on the image border (border ~ background);
    then fill-holes + keep the largest connected blob for a solid mask."""
    from scipy import ndimage
    from sklearn.cluster import KMeans

    fn = feats / (np.linalg.norm(feats, axis=1, keepdims=True) + 1e-9)
    lab = KMeans(n_clusters=2, n_init=5, random_state=0).fit_predict(fn).reshape(grid, grid)
    border = np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])
    fg = lab == int(np.argmin([(border == c).mean() for c in (0, 1)]))  # cluster least on border
    fg = ndimage.binary_fill_holes(fg)
    labeled, n = ndimage.label(fg)
    if n > 1:
        sizes = ndimage.sum(np.ones_like(labeled), labeled, index=range(1, n + 1))
        fg = labeled == (1 + int(np.argmax(sizes)))
    return fg.astype(np.float32).ravel()


def _load_font(size: int):
    from PIL import ImageFont

    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(p, size)
        except Exception:  # noqa: BLE001
            continue
    return ImageFont.load_default()


def _to_pil(arr01: np.ndarray, size: int):
    """float[0,1] HxWx3 -> PIL RGB upscaled to size x size (nearest keeps the patch grid crisp)."""
    from PIL import Image

    im = Image.fromarray((np.clip(arr01, 0, 1) * 255).astype(np.uint8))
    return im.resize((size, size), Image.NEAREST if arr01.shape[0] <= 64 else Image.BICUBIC)


def _compose(disp01, pca01, *, panel, fps, ms, cores, res, model_name, credit, show_fp, fp01=None):
    """Assemble one output frame: [input | (fp) | device PCA] with titles + an FPS/latency HUD
    overlaid on the device panel. Returns a PIL RGB image."""
    from PIL import Image, ImageDraw

    gap, head, foot = 14, 46, 26
    panels = [("input", _to_pil(disp01, panel))]
    if show_fp and fp01 is not None:
        panels.append(("DINOv2 fp", _to_pil(fp01, panel)))
    panels.append((f"{model_name} int8 - Metis", _to_pil(pca01, panel)))

    w = len(panels) * panel + (len(panels) - 1) * gap
    canvas = Image.new("RGB", (w, head + panel + foot), (16, 16, 20))
    d = ImageDraw.Draw(canvas)
    f_title, f_big, f_small = _load_font(19), _load_font(34), _load_font(15)

    x = 0
    for i, (title, im) in enumerate(panels):
        canvas.paste(im, (x, head))
        tw = d.textlength(title, font=f_title)
        d.text((x + (panel - tw) / 2, 12), title, font=f_title, fill=(225, 225, 230))
        x += panel + gap

    # HUD on the device (last) panel: measured FPS + ms/frame + config.
    hx, hy = w - panel + 12, head + 10
    d.rectangle([hx - 6, hy - 6, hx + 172, hy + 62], fill=(0, 0, 0))
    d.text((hx, hy), f"{fps:5.1f} FPS", font=f_big, fill=(80, 235, 120))
    d.text(
        (hx, hy + 40),
        f"{ms:4.0f} ms/frame  -  {cores} cores  -  {res}px",
        font=f_small,
        fill=(200, 200, 205),
    )
    fc = f_small  # shrink the credit line so it never clips the canvas width
    while d.textlength(credit, font=fc) > w - 16 and fc.size > 9:
        fc = _load_font(fc.size - 1)
    d.text((8, head + panel + 6), credit, font=fc, fill=(120, 120, 128))
    return canvas


@torch.no_grad()
def _fp_patches(hf_model, px, device):
    out = hf_model(pixel_values=px.to(device)).last_hidden_state
    return out[0, 1:].float().cpu().numpy()


def run_video_demo(
    video=None,
    repo_id="facebook/dinov2-base",
    *,
    n_frames=60,
    start=0,
    stride=1,
    n_calib=256,
    cores=4,
    image_size=None,
    out=None,
    fg_keep=0.6,
    show_fp=False,
    panel=384,
    cache_dir=None,
    credit=None,
    dpu_unroll=False,
):
    from PIL import Image

    from .calibration import make_processor
    from .model import load_dino_device

    fp_device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(out or (_DEMO_DIR / "dinov2_video_demo.gif"))
    # Resolve the input clip + its attribution (default clip has a known CC-BY credit; a custom
    # --video carries only the credit the caller supplies, else just its filename).
    if video:
        video = Path(video)
        clip_credit = credit or f"Video: {video.name}"
    else:
        video = _fetch_default_clip()
        clip_credit = _DEFAULT_CLIP_CREDIT

    # Runnable int8 device model -- reuses a cached compiled+calibrated artifact if present
    # (skips the multi-minute build), else builds it once into the cache. hf_model returned
    # only when --show-fp needs the fp reference column.
    dev, hf_model = load_dino_device(
        repo_id,
        cache_dir=cache_dir,
        cores=cores,
        n_calib=n_calib,
        image_size=image_size,
        calib_device=fp_device,
        want_fp=show_fp,
        dpu_unroll=dpu_unroll,
    )
    processor = make_processor(repo_id, image_size)
    crop = dev.image_size
    grid = dev.grid
    panel = max(
        grid, (panel // grid) * grid
    )  # whole multiple of grid -> uniform crisp patch blocks
    model_name = repo_id.split("/")[-1]

    frames_rgb, _ = _read_frames(video, start=start, n=n_frames, stride=stride)
    print(f"decoded {len(frames_rgb)} frames from {video.name}")

    # Warm up the device (first calls include graph load / device init -- excluded from timing).
    px0 = processor(images=Image.fromarray(frames_rgb[0]), return_tensors="pt").pixel_values
    for _ in range(2):
        dev.encode(px0)

    disp, dev_feats, fp_feats, latencies = [], [], [], []
    fp_hf = hf_model.to(fp_device) if show_fp else None
    for rgb in frames_rgb:
        pil = Image.fromarray(rgb)
        px = processor(images=pil, return_tensors="pt").pixel_values
        t0 = time.perf_counter()
        feat = dev.encode(px)  # host quant + one device dispatch + slice
        latencies.append(time.perf_counter() - t0)
        dev_feats.append(feat[0, 1:].astype(np.float32))
        if show_fp:
            fp_feats.append(_fp_patches(fp_hf, px, fp_device))
        disp.append(_display_rgb(pil, crop))
    if show_fp:
        hf_model.to("cpu")

    lat = np.array(latencies)
    print(
        f"device latency: median {lat.mean()*1e3:.1f} ms  ({1/lat.mean():.1f} FPS mean, "
        f"{1/np.median(lat):.1f} FPS median)"
    )

    # Per-frame foreground masks by 2-means clustering the basis features (fp with --show-fp, else
    # device); applied to BOTH columns so fp/device stay comparable. Clustering (not a PC0 threshold)
    # so feature-textured backgrounds like grass segment cleanly -- the object/bg split isn't the top
    # variance axis there, but the two feature clusters still separate it.
    basis_src = fp_feats if show_fp else dev_feats
    masks = None
    if fg_keep < 1.0:
        masks = [_fg_mask_kmeans(f, grid) for f in basis_src]
    # Colour basis on the L2-normalized FOREGROUND patches only -> the background doesn't dominate
    # the palette and per-frame colours stay comparable (uniform-content frames don't collapse to
    # one hue). ONE basis shared across all frames = temporally stable colours (no flicker); with
    # --show-fp, identical colouring for fp vs device.
    if masks is not None:
        fg_feats = np.concatenate([f[m > 0] for f, m in zip(basis_src, masks)], 0)
    else:
        fg_feats = np.concatenate(basis_src, 0)
    col_mean, col_comps = _pca_fit(_l2norm(fg_feats), 3)
    col_proj = (_l2norm(fg_feats) - col_mean) @ col_comps.T
    lo, hi = np.percentile(col_proj, 2, 0), np.percentile(col_proj, 98, 0)

    def rgb_map(feats, mask):
        proj = (_l2norm(feats) - col_mean) @ col_comps.T
        rgb = _to_rgb(proj, lo, hi).reshape(grid, grid, 3)
        if mask is not None:
            rgb = rgb * mask.reshape(grid, grid, 1)
        return rgb

    # Compose frames; HUD FPS is a short EMA of the instantaneous rate (feels "live").
    credit = f"{clip_credit}  |  DINOv2 int8 single-dispatch on Axelera Metis"
    out_frames, durations, ema = [], [], None
    for i in range(len(frames_rgb)):
        mask = masks[i] if masks is not None else None
        # Pass the raw (grid,grid,3) map -> _to_pil nearest-upsamples it to `panel`, so each patch
        # is a crisp solid block (no anti-aliasing / bilinear blur on the feature map).
        pca = rgb_map(dev_feats[i], mask)
        fp_pca = rgb_map(fp_feats[i], mask) if show_fp else None
        inst = 1.0 / latencies[i]
        ema = inst if ema is None else 0.7 * ema + 0.3 * inst
        out_frames.append(
            _compose(
                disp[i],
                pca,
                panel=panel,
                fps=ema,
                ms=latencies[i] * 1e3,
                cores=cores,
                res=crop,
                model_name=model_name,
                credit=credit,
                show_fp=show_fp,
                fp01=fp_pca,
            )
        )
        durations.append(max(20, int(round(latencies[i] * 1e3))))  # real device latency

    out.parent.mkdir(parents=True, exist_ok=True)
    out_frames[0].save(
        out,
        save_all=True,
        append_images=out_frames[1:],
        duration=durations,
        loop=0,
        disposal=2,
        optimize=True,
    )
    print(f"wrote {out}  ({len(out_frames)} frames, plays at real device speed)")
    _write_mp4(out.with_suffix(".mp4"), out_frames, fps=1.0 / float(np.median(lat)))
    return out


def _write_mp4(path: Path, frames, fps: float):
    """Write a constant-fps mp4 (small, shareable) at the median device FPS."""
    import cv2

    w, h = frames[0].size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), max(fps, 1.0), (w, h))
    if not vw.isOpened():
        print(f"(skipped mp4: no encoder for {path})")
        return
    for im in frames:
        vw.write(np.asarray(im)[:, :, ::-1])  # RGB -> BGR
    vw.release()
    print(f"wrote {path}  ({fps:.1f} fps, real device speed)")


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--video", default=None, help="input video (default: fetch a CC-BY kitten clip)"
    )
    p.add_argument("--repo-id", default="facebook/dinov2-base")
    p.add_argument("--n-frames", type=int, default=60)
    p.add_argument("--start", type=int, default=0, help="first source frame to use")
    p.add_argument("--stride", type=int, default=1, help="take every Nth source frame")
    p.add_argument("--n-calib", type=int, default=256)
    p.add_argument("--cores", type=int, default=4)
    p.add_argument("--image-size", type=int, default=None, help="e.g. 518 for native res")
    p.add_argument("--panel", type=int, default=384, help="per-panel pixel size in the output")
    p.add_argument("--fg-keep", type=float, default=0.6, help="<1.0 masks background to black")
    p.add_argument("--show-fp", action="store_true", help="add a middle fp-reference column")
    p.add_argument(
        "--cache-dir",
        default=None,
        help="reuse/build the compiled artifact here (default: keyed cache under "
        "$AXELERA_DINO_CACHE or ~/.cache/axelera_dino)",
    )
    p.add_argument(
        "--credit",
        default=None,
        help="attribution line for a custom --video (shown in the footer)",
    )
    p.add_argument("--out", default=None, help="output .gif (an .mp4 sibling is also written)")
    p.add_argument(
        "--dpu-unroll",
        action="store_true",
        help="compile with AltoConfig dpu_unroll=True (DPU loop-unroll; distinct artifact)",
    )
    args = p.parse_args(argv)
    run_video_demo(
        video=args.video,
        repo_id=args.repo_id,
        n_frames=args.n_frames,
        start=args.start,
        stride=args.stride,
        n_calib=args.n_calib,
        cores=args.cores,
        image_size=args.image_size,
        panel=args.panel,
        fg_keep=args.fg_keep,
        show_fp=args.show_fp,
        out=args.out,
        cache_dir=args.cache_dir,
        credit=args.credit,
        dpu_unroll=args.dpu_unroll,
    )


if __name__ == "__main__":
    main()

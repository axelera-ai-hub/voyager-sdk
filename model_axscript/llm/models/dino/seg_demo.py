# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""DINOv2 semantic segmentation on Metis (device-only).

The device runs the DINOv2 ViT in ONE dispatch (int8, see ``model.DinoV2Device``); the host runs
a tiny pretrained linear head -- Meta's ADE20K ``linear`` head (BatchNorm + 1x1 conv over the
768-d final-layer patch tokens) -- then upsamples/colorizes/overlays the 150-class map. No
recompile: the head consumes exactly ``encode()``'s features (validated: it expects the
final-LayerNorm patch tokens, which is what the device graph emits). @518 (37x37 patch grid) is
the natural resolution; the fast dpu_unroll artifact runs it at ~16 FPS.

    python -m tests.integration.alto.llm.models.dino.seg_demo                  # still-image panel
    python -m tests.integration.alto.llm.models.dino.seg_demo --video clip.mp4 # frame-by-frame video

The still panel is ``input | device int8 seg overlay`` per image with an ADE20K class legend; the
video is the same overlay played at the measured device speed with an FPS/latency HUD.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import time
import urllib.request

import numpy as np

# ADE20K (150 classes) -- mmsegmentation ordering, matching the head's conv_seg outputs.
ADE20K_CLASSES = [
    "wall",
    "building",
    "sky",
    "floor",
    "tree",
    "ceiling",
    "road",
    "bed ",
    "windowpane",
    "grass",
    "cabinet",
    "sidewalk",
    "person",
    "earth",
    "door",
    "table",
    "mountain",
    "plant",
    "curtain",
    "chair",
    "car",
    "water",
    "painting",
    "sofa",
    "shelf",
    "house",
    "sea",
    "mirror",
    "rug",
    "field",
    "armchair",
    "seat",
    "fence",
    "desk",
    "rock",
    "wardrobe",
    "lamp",
    "bathtub",
    "railing",
    "cushion",
    "base",
    "box",
    "column",
    "signboard",
    "chest of drawers",
    "counter",
    "sand",
    "sink",
    "skyscraper",
    "fireplace",
    "refrigerator",
    "grandstand",
    "path",
    "stairs",
    "runway",
    "case",
    "pool table",
    "pillow",
    "screen door",
    "stairway",
    "river",
    "bridge",
    "bookcase",
    "blind",
    "coffee table",
    "toilet",
    "flower",
    "book",
    "hill",
    "bench",
    "countertop",
    "stove",
    "palm",
    "kitchen island",
    "computer",
    "swivel chair",
    "boat",
    "bar",
    "arcade machine",
    "hovel",
    "bus",
    "towel",
    "light",
    "truck",
    "tower",
    "chandelier",
    "awning",
    "streetlight",
    "booth",
    "television receiver",
    "airplane",
    "dirt track",
    "apparel",
    "pole",
    "land",
    "bannister",
    "escalator",
    "ottoman",
    "bottle",
    "buffet",
    "poster",
    "stage",
    "van",
    "ship",
    "fountain",
    "conveyer belt",
    "canopy",
    "washer",
    "plaything",
    "swimming pool",
    "stool",
    "barrel",
    "basket",
    "waterfall",
    "tent",
    "bag",
    "minibike",
    "cradle",
    "oven",
    "ball",
    "food",
    "step",
    "tank",
    "trade name",
    "microwave",
    "pot",
    "animal",
    "bicycle",
    "lake",
    "dishwasher",
    "screen",
    "blanket",
    "sculpture",
    "hood",
    "sconce",
    "vase",
    "traffic light",
    "tray",
    "ashcan",
    "fan",
    "pier",
    "crt screen",
    "plate",
    "monitor",
    "bulletin board",
    "shower",
    "radiator",
    "glass",
    "clock",
    "flag",
]
ADE20K_PALETTE = np.array(
    [
        [120, 120, 120],
        [180, 120, 120],
        [6, 230, 230],
        [80, 50, 50],
        [4, 200, 3],
        [120, 120, 80],
        [140, 140, 140],
        [204, 5, 255],
        [230, 230, 230],
        [4, 250, 7],
        [224, 5, 255],
        [235, 255, 7],
        [150, 5, 61],
        [120, 120, 70],
        [8, 255, 51],
        [255, 6, 82],
        [143, 255, 140],
        [204, 255, 4],
        [255, 51, 7],
        [204, 70, 3],
        [0, 102, 200],
        [61, 230, 250],
        [255, 6, 51],
        [11, 102, 255],
        [255, 7, 71],
        [255, 9, 224],
        [9, 7, 230],
        [220, 220, 220],
        [255, 9, 92],
        [112, 9, 255],
        [8, 255, 214],
        [7, 255, 224],
        [255, 184, 6],
        [10, 255, 71],
        [255, 41, 10],
        [7, 255, 255],
        [224, 255, 8],
        [102, 8, 255],
        [255, 61, 6],
        [255, 194, 7],
        [255, 122, 8],
        [0, 255, 20],
        [255, 8, 41],
        [255, 5, 153],
        [6, 51, 255],
        [235, 12, 255],
        [160, 150, 20],
        [0, 163, 255],
        [140, 140, 140],
        [250, 10, 15],
        [20, 255, 0],
        [31, 255, 0],
        [255, 31, 0],
        [255, 224, 0],
        [153, 255, 0],
        [0, 0, 255],
        [255, 71, 0],
        [0, 235, 255],
        [0, 173, 255],
        [31, 0, 255],
        [11, 200, 200],
        [255, 82, 0],
        [0, 255, 245],
        [0, 61, 255],
        [0, 255, 112],
        [0, 255, 133],
        [255, 0, 0],
        [255, 163, 0],
        [255, 102, 0],
        [194, 255, 0],
        [0, 143, 255],
        [51, 255, 0],
        [0, 82, 255],
        [0, 255, 41],
        [0, 255, 173],
        [10, 0, 255],
        [173, 255, 0],
        [0, 255, 153],
        [255, 92, 0],
        [255, 0, 255],
        [255, 0, 245],
        [255, 0, 102],
        [255, 173, 0],
        [255, 0, 20],
        [255, 184, 184],
        [0, 31, 255],
        [0, 255, 61],
        [0, 71, 255],
        [255, 0, 204],
        [0, 255, 194],
        [0, 255, 82],
        [0, 10, 255],
        [0, 112, 255],
        [51, 0, 255],
        [0, 194, 255],
        [0, 122, 255],
        [0, 255, 163],
        [255, 153, 0],
        [0, 255, 10],
        [255, 112, 0],
        [143, 255, 0],
        [82, 0, 255],
        [163, 255, 0],
        [255, 235, 0],
        [8, 184, 170],
        [133, 0, 255],
        [0, 255, 92],
        [184, 0, 255],
        [255, 0, 31],
        [0, 184, 255],
        [0, 214, 255],
        [255, 0, 112],
        [92, 255, 0],
        [0, 224, 255],
        [112, 224, 255],
        [70, 184, 160],
        [163, 0, 255],
        [153, 0, 255],
        [71, 255, 0],
        [255, 0, 163],
        [255, 204, 0],
        [255, 0, 143],
        [0, 255, 235],
        [133, 255, 0],
        [255, 0, 235],
        [245, 0, 255],
        [255, 0, 122],
        [255, 245, 0],
        [10, 190, 212],
        [214, 255, 0],
        [0, 204, 255],
        [20, 0, 255],
        [255, 255, 0],
        [0, 153, 255],
        [0, 41, 255],
        [0, 255, 204],
        [41, 0, 255],
        [41, 255, 0],
        [173, 0, 255],
        [0, 245, 255],
        [71, 0, 255],
        [122, 0, 255],
        [0, 255, 184],
        [0, 92, 255],
        [184, 255, 0],
        [0, 133, 255],
        [255, 214, 0],
        [25, 194, 194],
        [102, 255, 0],
        [92, 0, 255],
    ],
    dtype=np.uint8,
)

_HEAD_URL = (
    "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/dinov2_vitb14_ade20k_linear_head.pth"
)
_DEMO_DIR = Path(__file__).parent / "demo"


def load_seg_head(cache_dir=None) -> dict:
    """Download (once) + load Meta's ADE20K linear head as numpy params.

    Head = BatchNorm over 768 channels then a 1x1 conv (768 -> 150). Cached under
    ``<dino cache>/seg_heads`` so the download is paid once.
    """
    import torch

    from .model import default_cache_root

    cache_dir = Path(cache_dir) if cache_dir else default_cache_root() / "seg_heads"
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = cache_dir / "dinov2_vitb14_ade20k_linear_head.pth"
    if not path.exists():
        print(f"[seg] downloading ADE20K head -> {path}")
        urllib.request.urlretrieve(_HEAD_URL, path)
    sd = torch.load(path, map_location="cpu", weights_only=True)
    if "state_dict" in sd:
        sd = sd["state_dict"]
    n_cls = len(ADE20K_CLASSES)
    return {
        "bn_w": sd["decode_head.bn.weight"].numpy(),
        "bn_b": sd["decode_head.bn.bias"].numpy(),
        "bn_rm": sd["decode_head.bn.running_mean"].numpy(),
        "bn_rv": sd["decode_head.bn.running_var"].numpy(),
        "cw": sd["decode_head.conv_seg.weight"].numpy().reshape(n_cls, -1),
        "cb": sd["decode_head.conv_seg.bias"].numpy(),
    }


def seg_logits(patch_tokens: np.ndarray, head: dict, eps: float = 1e-5) -> np.ndarray:
    """(N, 768) patch tokens -> (N, 150) class logits (BatchNorm then 1x1 conv)."""
    y = (patch_tokens - head["bn_rm"]) / np.sqrt(head["bn_rv"] + eps) * head["bn_w"] + head["bn_b"]
    return y @ head["cw"].T + head["cb"]


def folded_head(head: dict = None, eps: float = 1e-5):
    """Fold the head's BatchNorm into the 1x1 conv -> (W' (n_classes, 768), b' (n_classes,)) fp32.

    Used to bake the head into the device graph (model.load_dino_device(seg_head=True)); the same
    linear the host path applies, minus the separate BN (BN(x)=a*x+c, so W'=cw*a, b'=cw@c+cb).
    """
    if head is None:
        head = load_seg_head()
    a = head["bn_w"] / np.sqrt(head["bn_rv"] + eps)
    c = head["bn_b"] - head["bn_rm"] * a
    W = (head["cw"] * a[None, :]).astype(np.float32)
    b = (head["cw"] @ c + head["cb"]).astype(np.float32)
    return W, b


def class_map(dev, pixel_values, head: dict = None) -> np.ndarray:
    """One device dispatch -> (grid, grid) int class map (ADE20K indices).

    With an on-device head (``dev.seg_head``) ``encode`` already returns per-token logits; otherwise
    the host head is applied to the returned features (``head`` required in that case).
    """
    out = dev.encode(pixel_values)[0, 1:]  # drop CLS -> (grid*grid, D)
    logits = out if getattr(dev, "seg_head", False) else seg_logits(out, head)
    return logits.argmax(1).reshape(dev.grid, dev.grid)


def colorize(cls_map: np.ndarray) -> np.ndarray:
    """(grid, grid) ADE20K indices -> (grid, grid, 3) uint8 via the ADE20K palette."""
    return ADE20K_PALETTE[cls_map]


def overlay(base_rgb: np.ndarray, cls_map: np.ndarray, *, alpha: float = 0.55) -> np.ndarray:
    """base_rgb (H, W, 3) uint8 + (grid, grid) class map -> overlaid RGB at base res (NEAREST)."""
    from PIL import Image

    h, w = base_rgb.shape[:2]
    seg = np.asarray(Image.fromarray(colorize(cls_map)).resize((w, h), Image.NEAREST))
    return (base_rgb * (1 - alpha) + seg * alpha).astype(np.uint8)


def top_classes(cls_map: np.ndarray, k: int = 8):
    """The k most-present ADE20K classes -> [(name, (r,g,b)), ...] for a legend."""
    idx, cnt = np.unique(cls_map, return_counts=True)
    order = idx[np.argsort(-cnt)][:k]
    return [(ADE20K_CLASSES[i], tuple(int(c) for c in ADE20K_PALETTE[i])) for i in order]


# --- rendering helpers -----------------------------------------------------------------------


def _load_font(size: int):
    from PIL import ImageFont

    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _legend_strip(entries, width: int, *, height: int, bg=(20, 24, 28)) -> np.ndarray:
    """A horizontal legend: colour swatch + class name for each (name, rgb) entry."""
    from PIL import Image, ImageDraw

    img = Image.new("RGB", (width, height), bg)
    d = ImageDraw.Draw(img)
    font = _load_font(max(12, height // 3))
    x, y, sw = 12, height // 4, height // 2
    for name, rgb in entries:
        d.rectangle([x, y, x + sw, y + sw], fill=rgb)
        d.text((x + sw + 6, y), name, fill=(235, 235, 235), font=font)
        x += sw + 12 + int(d.textlength(name, font=font)) + 22
        if x > width - 80:
            break
    return np.asarray(img)


def _panel(
    base_rgb: np.ndarray,
    ov: np.ndarray,
    *,
    title_l="input",
    title_r="dinov2 int8 seg - Metis",
    hud=None,
) -> np.ndarray:
    """Compose a single [input | overlay] row with titles and an optional HUD line."""
    from PIL import Image, ImageDraw

    h, w = base_rgb.shape[:2]
    bar = max(28, h // 12)
    canvas = Image.new("RGB", (w * 2, h + bar), (20, 24, 28))
    canvas.paste(Image.fromarray(base_rgb), (0, bar))
    canvas.paste(Image.fromarray(ov), (w, bar))
    d = ImageDraw.Draw(canvas)
    font = _load_font(max(14, bar // 2))
    d.text((10, bar // 6), title_l, fill=(235, 235, 235), font=font)
    d.text((w + 10, bar // 6), title_r, fill=(235, 235, 235), font=font)
    if hud:
        big = _load_font(max(20, bar))
        small = _load_font(max(12, bar // 2))
        d.text((w + 10, bar + 6), hud[0], fill=(80, 235, 120), font=big)
        d.text((w + 10, bar + 10 + max(20, bar)), hud[1], fill=(210, 210, 210), font=small)
    return np.asarray(canvas)


# --- image sources ---------------------------------------------------------------------------


def _demo_scene_images(n: int, seed: int = 0):
    """Diverse natural scenes (COCO val) -- good ADE20K coverage (people/animals/vehicles/scenes)."""
    from datasets import load_dataset

    ds = (
        load_dataset("sayakpaul/coco-30-val-2014", split="train")
        .shuffle(seed=seed)
        .select(range(n))
    )
    return [ex["image"].convert("RGB") for ex in ds]


def _fit(img, size: int):
    """Center-crop-resize a PIL image to a square `size` (for display panels)."""
    from PIL import Image

    w, h = img.size
    s = size / min(w, h)
    img = img.resize((round(w * s), round(h * s)), Image.BICUBIC)
    w, h = img.size
    left, top = (w - size) // 2, (h - size) // 2
    return np.asarray(img.crop((left, top, left + size, top + size)))


# --- still-image panel -----------------------------------------------------------------------


def run_image_demo(
    repo_id="facebook/dinov2-base",
    *,
    n_images=6,
    cores=4,
    n_calib=64,
    image_size=518,
    dpu_unroll=True,
    panel=384,
    out=None,
    seed=0,
    alpha=0.55,
    on_device_head=False,
):
    from PIL import Image

    from .calibration import make_processor
    from .model import load_dino_device

    out = Path(out or (_DEMO_DIR / "dinov2_seg_demo.png"))
    dev, _ = load_dino_device(
        repo_id,
        cores=cores,
        n_calib=n_calib,
        image_size=image_size,
        calib_device="cuda",
        want_fp=False,
        dpu_unroll=dpu_unroll,
        seg_head=on_device_head,
    )
    head = None if on_device_head else load_seg_head()
    proc = make_processor(repo_id, image_size)
    images = _demo_scene_images(n_images, seed)

    rows = []
    for img in images:
        px = proc(images=img, return_tensors="pt").pixel_values
        cm = class_map(dev, px, head)
        base = _fit(img, panel)
        ov = overlay(base, cm, alpha=alpha)
        row = _panel(base, ov)
        legend = _legend_strip(top_classes(cm, 8), row.shape[1], height=max(26, panel // 12))
        rows.append(np.concatenate([row, legend], axis=0))
    Image.fromarray(np.concatenate(rows, axis=0)).save(out)
    print(f"wrote {out}  ({len(images)} images, device int8 seg @ {dev.image_size})")
    return out


# --- video ------------------------------------------------------------------------------------


def run_video_demo(
    video,
    repo_id="facebook/dinov2-base",
    *,
    n_frames=60,
    start=0,
    stride=1,
    cores=4,
    n_calib=64,
    image_size=518,
    dpu_unroll=True,
    panel=384,
    out=None,
    credit=None,
    alpha=0.55,
    on_device_head=False,
):
    from PIL import Image

    from .calibration import make_processor
    from .model import load_dino_device
    from .video_demo import _read_frames, _write_mp4

    out = Path(out or (_DEMO_DIR / "dinov2_seg_video.gif"))
    dev, _ = load_dino_device(
        repo_id,
        cores=cores,
        n_calib=n_calib,
        image_size=image_size,
        calib_device="cuda",
        want_fp=False,
        dpu_unroll=dpu_unroll,
        seg_head=on_device_head,
    )
    head = None if on_device_head else load_seg_head()
    proc = make_processor(repo_id, image_size)
    frames_rgb, _ = _read_frames(Path(video), start=start, n=n_frames, stride=stride)

    out_frames, durations, ema = [], [], None
    for fr in frames_rgb:
        img = Image.fromarray(fr)
        px = proc(images=img, return_tensors="pt").pixel_values
        t0 = time.perf_counter()
        cm = class_map(dev, px, head)
        dt = time.perf_counter() - t0
        ema = dt if ema is None else 0.9 * ema + 0.1 * dt
        base = _fit(img, panel)
        row = _panel(
            base,
            overlay(base, cm, alpha=alpha),
            hud=(
                f"{1.0 / ema:.1f} FPS",
                f"{dt * 1e3:.0f} ms/frame  -  {cores} cores  -  {dev.image_size}px",
            ),
        )
        legend = _legend_strip(top_classes(cm, 8), row.shape[1], height=max(26, panel // 12))
        out_frames.append(Image.fromarray(np.concatenate([row, legend], axis=0)))
        durations.append(max(20, int(round(dt * 1e3))))
    lat = np.array([d / 1e3 for d in durations])
    print(
        f"device latency: median {np.median(lat)*1e3:.1f} ms ({1/np.median(lat):.1f} FPS median)"
    )
    out_frames[0].save(
        out, save_all=True, append_images=out_frames[1:], duration=durations, loop=0
    )
    _write_mp4(out.with_suffix(".mp4"), out_frames, 1.0 / float(np.median(lat)))
    print(f"wrote {out} (+ .mp4)  ({len(out_frames)} frames, real device speed)")
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo-id", default="facebook/dinov2-base")
    p.add_argument("--video", default=None, help="run a clip frame-by-frame (else still panel)")
    p.add_argument("--n-images", type=int, default=6)
    p.add_argument("--n-frames", type=int, default=60)
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--stride", type=int, default=1)
    p.add_argument("--cores", type=int, default=4)
    p.add_argument("--n-calib", type=int, default=64)
    p.add_argument("--image-size", type=int, default=518)
    p.add_argument("--panel", type=int, default=384)
    p.add_argument("--alpha", type=float, default=0.55, help="overlay opacity")
    p.add_argument(
        "--no-dpu-unroll", action="store_true", help="disable dpu_unroll (slower artifact)"
    )
    p.add_argument(
        "--on-device-head",
        action="store_true",
        help="bake the ADE20K head into the graph (pixels->logits, one dispatch; "
        "distinct seg-specific artifact)",
    )
    p.add_argument("--credit", default=None)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    common = dict(
        repo_id=args.repo_id,
        cores=args.cores,
        n_calib=args.n_calib,
        image_size=args.image_size,
        dpu_unroll=not args.no_dpu_unroll,
        panel=args.panel,
        out=args.out,
        alpha=args.alpha,
        on_device_head=args.on_device_head,
    )
    if args.video:
        run_video_demo(
            args.video,
            n_frames=args.n_frames,
            start=args.start,
            stride=args.stride,
            credit=args.credit,
            **common,
        )
    else:
        run_image_demo(n_images=args.n_images, **common)


if __name__ == "__main__":
    main()

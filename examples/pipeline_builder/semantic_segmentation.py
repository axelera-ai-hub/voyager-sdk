#!/usr/bin/env python3
# Copyright Axelera AI, 2026
"""Semantic segmentation (YOLO26n-sem, pixel-level class map output).

Ultralytics YOLO26n-sem trained on Cityscapes (19 classes), compiled at
1024x1024. The .axm emits (1, 19, H, W) per-class logits;
decode_semantic_segmentation() argmaxes over the class axis to a (1, H, W)
class map, which ax_semantic_segmentation() wraps for palette rendering.

Preprocessing follows the Ultralytics convention (RGB, resize to 1024x1024,
HWC->CHW scaled to [0, 1] via to_tensor); the model has no inline
normalization. Swap the model path, resize dims, and num_classes for other
semantic-segmentation models.

Auto-generated from rt_demo.py's @publicdemo `semantic_segmentation` by
tools/export_pipeline_builder_demo.py.  Edit the original demo in rt_demo.py
rather than this file -- regenerate with
`python tools/export_pipeline_builder_demo.py semantic_segmentation`.

Usage:
    python semantic_segmentation.py PATH [--display opencv|console|none|auto] [--no-wait]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from axelera.runtime import op, cv, display

# Populate this directory once with the .axm files referenced below
# (`axdownloadmodel --axm <name>` inside this directory).
MODEL_DIR = str(Path.home() / ".cache" / "axelera" / "runtime2")


def main(args):
    with display.App(renderer=args.display) as visualizer:
        vis = visualizer.create_window(
            'semantic_segmentation', (args.window_width, args.window_height)
        )
        pipeline = op.seq(
            op.color_convert('RGB'),
            op.resize(width=1024, height=1024),
            op.to_tensor(),
            op.load(f'{MODEL_DIR}/yolo26n-sem.axm'),
            op.decode_semantic_segmentation(num_classes=19),
            op.ax_semantic_segmentation(),
        )

        for img in cv.create_source(args.input):
            if vis.is_closed:
                break

            result = pipeline(img)
            vis(img, result)
        # Headless backend ('none') never sets is_closed -- wait_for_close
        # would block forever -- so skip the wait there even when --wait is set.
        if args.wait and args.display != "none":
            vis.wait_for_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Semantic segmentation (YOLO26n-sem, pixel-level class map output).'
    )
    parser.add_argument("input", help="Image or video file to process")
    parser.add_argument(
        "--display",
        choices=["none", "opencv", "console", "iterm2", "auto"],
        default="auto",
        help="Display backend (default: auto)",
    )
    parser.add_argument("--window-width", type=int, default=800)
    parser.add_argument("--window-height", type=int, default=500)
    parser.add_argument(
        "-w",
        "--wait",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep the window open after processing until the user closes "
        "it (default). Use --no-wait to exit as soon as the input ends; "
        "useful for batch/video runs where you do not need to inspect "
        "the final frame.",
    )
    parser.add_argument(
        "--backend",
        choices=["ffmpeg", "opencv"],
        default="ffmpeg",
        help="Video decode backend passed to cv.create_source (default: %(default)s)",
    )
    main(parser.parse_args())

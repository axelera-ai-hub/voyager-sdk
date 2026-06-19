#!/usr/bin/env python3
# Copyright Axelera AI, 2026
"""YOLO OBB (Oriented Bounding Box) detection pipeline.

Detects objects with rotation using YOLO11n-obb model trained on DOTA dataset
(15 classes: plane, ship, vehicle, etc.).

Usage:
    rt-demo.py obb data/P0019_0_2304.png --no-display --save-dir rtout

Auto-generated from rt_demo.py's @publicdemo `obb` by
tools/export_pipeline_builder_demo.py.  Edit the original demo in rt_demo.py
rather than this file -- regenerate with
`python tools/export_pipeline_builder_demo.py obb`.

Usage:
    python obb.py PATH [--display opencv|console|none|auto] [--no-wait]
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
        vis = visualizer.create_window('obb', (args.window_width, args.window_height))
        pipeline = op.seq(
            op.color_convert('BGR', 'I420'),
            op.letterbox(1024, 1024),
            op.to_tensor(),
            op.load(f'{MODEL_DIR}/yolo11n-obb-dotav1-onnx.axm'),
            op.decode_obb(num_classes=15, confidence_threshold=0.25),
            op.nms(iou_threshold=0.45, max_boxes=300, box_format='xywhr'),
            op.to_image_space(box_format='xywhr'),
            op.ax_obb(class_id_type=op.DotaClasses),
        )
        for img in cv.create_source(args.input):
            if vis.is_closed:
                break

            detections = pipeline(img)
            vis(img, detections)
        # Headless backend ('none') never sets is_closed -- wait_for_close
        # would block forever -- so skip the wait there even when --wait is set.
        if args.wait and args.display != "none":
            vis.wait_for_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='YOLO OBB (Oriented Bounding Box) detection pipeline.'
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
    main(parser.parse_args())

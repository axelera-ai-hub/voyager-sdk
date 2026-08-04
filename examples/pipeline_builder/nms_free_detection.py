#!/usr/bin/env python3
# Copyright Axelera AI, 2026
"""NMS-free YOLO detection (yolo26).

NMS-free models output (1, 300, 6) with all candidates -- decode_detections
filters by confidence, no nms needed. Pipeline: load -> decode -> to_image_space -> ax_detection.

Supports both hardware (.axm) and ONNX paths:
    rt-demo.py nms_free_detection image.jpg
    rt-demo.py nms_free_detection image.jpg --onnx

Auto-generated from rt_demo.py's @publicdemo `nms_free_detection` by
tools/export_pipeline_builder_demo.py.  Edit the original demo in rt_demo.py
rather than this file -- regenerate with
`python tools/export_pipeline_builder_demo.py nms_free_detection`.

Usage:
    python nms_free_detection.py PATH [--display opencv|console|none|auto] [--no-wait]
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
            'nms_free_detection', (args.window_width, args.window_height)
        )
        pipeline = op.seq(
            op.color_convert('RGB'),
            op.letterbox(640, 640),
            op.to_tensor(),
            op.load(f'{MODEL_DIR}/yolo26n-coco-onnx.axm'),
            # NMS-free: decode_detections handles confidence filtering + squeeze, no nms needed
            op.decode_detections(algo='yolo26', num_classes=80, confidence_threshold=0.4),
            op.to_image_space(),
            op.ax_detection(class_id_type=op.CocoClasses),
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
    parser = argparse.ArgumentParser(description='NMS-free YOLO detection (yolo26).')
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

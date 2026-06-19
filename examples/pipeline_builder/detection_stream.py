#!/usr/bin/env python3
# Copyright Axelera AI, 2026
"""YOLOv8 object detection pipeline using a pipelined stream.

Detects objects using YOLOv8n model trained on COCO dataset (80 classes).
Applies NMS to remove duplicate detections and outputs DetectedObject instances.

Note: For custom models with different classes, use:
      MyClasses = op.load_classes('my_labels.txt')

Auto-generated from rt_demo.py's @publicdemo `detection_stream` by
tools/export_pipeline_builder_demo.py.  Edit the original demo in rt_demo.py
rather than this file -- regenerate with
`python tools/export_pipeline_builder_demo.py detection_stream`.

Usage:
    python detection_stream.py PATH [--display opencv|console|none|auto] [--no-wait]
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
        vis = visualizer.create_window('detection_stream', (args.window_width, args.window_height))
        pipeline = op.seq(
            op.color_convert('RGB', src='BGR'),
            # YOLO preprocessing: letterbox maintains aspect ratio unlike resize
            op.letterbox(640, 640),
            op.to_tensor(),
            # Detection postprocessing: decode -> NMS -> to_image_space -> DetectedObject
            op.load(f'{MODEL_DIR}/yolov8n-coco.axm'),
            op.decode_detections(algo='yolov8', num_classes=80, confidence_threshold=0.25),
            op.nms(iou_threshold=0.45, max_boxes=300),
            op.to_image_space(),
            op.ax_detection(class_id_type=op.CocoClasses),
        )

        with cv.create_source(args.input) as source:
            for img, detections in pipeline.stream(source):
                if vis.is_closed:
                    break
                vis(img, detections)
        # Headless backend ('none') never sets is_closed -- wait_for_close
        # would block forever -- so skip the wait there even when --wait is set.
        if args.wait and args.display != "none":
            vis.wait_for_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='YOLOv8 object detection pipeline using a pipelined stream.'
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

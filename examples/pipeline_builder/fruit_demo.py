#!/usr/bin/env python3
# Copyright Axelera AI, 2026
"""Fruit cascade demo: YOLOv8l-pose → segmentation on ROIs + full-frame fruit detection.

Three-stage pipeline (mirrors ax_models/reference/cascade/fruit-demo.yaml):
- master_detections: YOLOv8l-pose on full frame to locate persons/objects
- segmentations: YOLOv8s-seg on top-5 ROIs by bbox area, filtered for banana/apple/orange
- object_detections: YOLOv8s on full frame, filtered for banana/apple/orange

Auto-generated from rt_demo.py's @publicdemo `fruit_demo` by
tools/export_pipeline_builder_demo.py.  Edit the original demo in rt_demo.py
rather than this file -- regenerate with
`python tools/export_pipeline_builder_demo.py fruit_demo`.

Usage:
    python fruit_demo.py PATH [--display opencv|console|none|auto] [--no-wait]
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
        vis = visualizer.create_window('fruit_demo', (args.window_width, args.window_height))
        FRUIT_CLASSES = [op.CocoClasses.banana, op.CocoClasses.apple, op.CocoClasses.orange]

        pipeline = op.seq(
            op.color_convert('RGB'),
            op.letterbox(640, 640),
            op.to_tensor(),
            op.par(
                op.seq(
                    op.load(f'{MODEL_DIR}/yolov8lpose-coco-onnx.axm'),
                    op.decode_pose(algo='yolov8', num_keypoints=17),
                    op.nms(iou_threshold=0.45, max_boxes=300),
                    op.to_image_space(keypoint_cols=range(6, 57, 3)),
                    op.ax_pose(),
                    op.sort_by(by='area', top=5),
                    op.for_each(
                        'fruit_segs',
                        op.crop_roi(property='bbox'),
                        op.letterbox(640, 640),
                        op.to_tensor(),
                        op.load(f'{MODEL_DIR}/yolov8sseg-coco-onnx.axm'),
                        op.decode_segmentation(algo='yolov8', num_classes=80),
                        op.par(
                            op.seq(op.itemgetter(0), op.nms(iou_threshold=0.45, max_boxes=300)),
                            op.itemgetter(1),
                        ),
                        op.par(
                            op.seq(op.pack(), op.itemgetter(0), op.to_image_space()),
                            op.proto_to_mask(),
                        ),
                        op.ax_segmentation(class_id_type=op.CocoClasses),
                        op.filter(class_ids=FRUIT_CLASSES),
                    ),
                ),
                op.seq(
                    op.load(f'{MODEL_DIR}/yolov8s-coco-onnx.axm'),
                    op.decode_detections(algo='yolov8', num_classes=80, confidence_threshold=0.25),
                    op.nms(iou_threshold=0.45, max_boxes=300),
                    op.to_image_space(),
                    op.ax_detection(class_id_type=op.CocoClasses),
                    op.filter(class_ids=FRUIT_CLASSES),
                ),
            ),
        )

        with cv.create_source(args.input, backend=args.backend) as source:
            for img, results in pipeline.stream(source):
                if vis.is_closed:
                    break
                vis(img, results)
        # Headless backend ('none') never sets is_closed -- wait_for_close
        # would block forever -- so skip the wait there even when --wait is set.
        if args.wait and args.display != "none":
            vis.wait_for_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Fruit cascade demo: YOLOv8l-pose → segmentation on ROIs + full-frame fruit detection.'
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

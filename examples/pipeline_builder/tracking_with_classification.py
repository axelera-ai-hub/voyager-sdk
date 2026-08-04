#!/usr/bin/env python3
# Copyright Axelera AI, 2026
"""Track vehicles and classify each tracked object - demonstrates cascade after tracking.

Shows that TrackedObject can be used as a parent in cascade operations:
- Detection -> Filtering -> Tracking -> Foreach(Classification)

Pattern: Any object with a .bbox property can be used with croproi()
TrackedObject implements the HasBBox protocol via its .bbox property.

Auto-generated from rt_demo.py's @publicdemo `tracking_with_classification` by
tools/export_pipeline_builder_demo.py.  Edit the original demo in rt_demo.py
rather than this file -- regenerate with
`python tools/export_pipeline_builder_demo.py tracking_with_classification`.

Usage:
    python tracking_with_classification.py PATH [--display opencv|console|none|auto] [--no-wait]
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
            'tracking_with_classification', (args.window_width, args.window_height)
        )
        pipeline = op.seq(
            op.color_convert('RGB'),
            op.letterbox(640, 640),
            op.to_tensor(),
            op.load(f'{MODEL_DIR}/yolov8n-coco.axm'),
            op.decode_detections(algo='yolov8', num_classes=80, confidence_threshold=0.25),
            op.nms(iou_threshold=0.45, max_boxes=300),
            op.to_image_space(),
            op.ax_detection(class_id_type=op.CocoClasses),
            op.filter(
                class_ids=[
                    op.CocoClasses.car,
                    op.CocoClasses.truck,
                    op.CocoClasses.bus,
                    op.CocoClasses.motorcycle,
                ]
            ),
            op.tracker(algo='oc-sort'),
            # foreach: classify each tracked vehicle
            op.for_each(
                'vehicle_types',
                op.crop_roi(property='bbox'),  # TrackedObject has .bbox property
                op.resize(size=256, half_pixel_centers=True),
                op.center_crop((224, 224)),
                op.to_tensor(),
                op.normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                op.load(f'{MODEL_DIR}/squeezenet1.0-imagenet.axm'),
                op.softmax(),
                op.ax_classification(class_id_type=op.ImagenetClasses),
                op.top_k(k=1),
            ),
        )
        for img in cv.create_source(args.input):
            if vis.is_closed:
                break

            with op.frame_context(img):
                result = pipeline(img)
            tracked_objects = result.input
            vis(img, tracked_objects)
        # Headless backend ('none') never sets is_closed -- wait_for_close
        # would block forever -- so skip the wait there even when --wait is set.
        if args.wait and args.display != "none":
            vis.wait_for_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Track vehicles and classify each tracked object - demonstrates cascade after tracking.'
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

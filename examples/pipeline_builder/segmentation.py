#!/usr/bin/env python3
# Copyright Axelera AI, 2026
"""YOLOv8 instance segmentation pipeline with explicit tuple data flow.

Detects objects and generates per-instance segmentation masks using
prototype-based mask prediction. Uses par + itemgetter pattern for explicit
data flow between operators.

Data flow:
    decode_segmentation -> (detections, protos) as a tuple
    par(itemgetter+nms, itemgetter) -> (filtered_dets, protos) unpacked
    par(pack+itemgetter+to_image_space, proto_to_mask) -> (img_dets, masks)
    axsegmentation -> list[SegmentedObject]

Auto-generated from rt_demo.py's @publicdemo `segmentation` by
tools/export_pipeline_builder_demo.py.  Edit the original demo in rt_demo.py
rather than this file -- regenerate with
`python tools/export_pipeline_builder_demo.py segmentation`.

Usage:
    python segmentation.py PATH [--display opencv|console|none|auto] [--no-wait]
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
        vis = visualizer.create_window('segmentation', (args.window_width, args.window_height))
        pipeline = op.seq(
            op.color_convert('RGB'),
            op.letterbox(640, 640),
            op.to_tensor(),
            op.load(f'{MODEL_DIR}/yolov8nseg-coco.axm'),
            # decode_segmentation returns (detections, protos) as a tuple
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
        )

        for img in cv.create_source(args.input):
            if vis.is_closed:
                break

            segments = pipeline(img)
            vis(img, segments)
        # Headless backend ('none') never sets is_closed -- wait_for_close
        # would block forever -- so skip the wait there even when --wait is set.
        if args.wait and args.display != "none":
            vis.wait_for_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='YOLOv8 instance segmentation pipeline with explicit tuple data flow.'
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

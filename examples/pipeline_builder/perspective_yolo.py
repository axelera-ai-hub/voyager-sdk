#!/usr/bin/env python3
# Copyright Axelera AI, 2026
"""Demonstrates perspective transform followed by YOLO detection on the input stream.

split_at='undistort' runs color_convert + perspective on this thread and hands
the corrected frame straight to frame_callback (here the window) before inference
is submitted, so the display keeps up with the decoder instead of waiting on the
model; the boxes are drawn onto the same frame a beat later, when the result
arrives. Splitting without a frame_callback would pay the cost and buy nothing.

Usage:
    python perspective_yolo.py PATH [--display opencv|console|none|auto] [--no-wait]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from axelera.runtime import op, cv, display
import math

# Populate this directory once with the .axm files referenced below
# (`axdownloadmodel --axm <name>` inside this directory).
MODEL_DIR = str(Path.home() / ".cache" / "axelera" / "runtime2")


def main(args):
    with display.App(renderer=args.display) as visualizer:
        vis = visualizer.create_window('perspective_yolo', (args.window_width, args.window_height))
        angle = math.radians(45)  # Convert 45 degrees to radians from center pivot
        center_x, center_y = 320, 240  # Assuming image dimensions are 640x480
        rotation_matrix = [
            math.cos(angle),
            -math.sin(angle),
            center_x * (1 - math.cos(angle)) + center_y * math.sin(angle),
            math.sin(angle),
            math.cos(angle),
            center_y * (1 - math.cos(angle)) - center_x * math.sin(angle),
            0.0,
            0.0,
            1.0,
        ]

        pipeline = op.seq(
            op.color_convert('RGB'),
            # Name the geometric-correction op so stream(split_at=...) splits here: the
            # prefix (color_convert + perspective) runs in the stream thread, the rest
            # on the worker pool.
            op.perspective(matrix=rotation_matrix, width=640, height=480, name='undistort'),
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

        for img, result in pipeline.stream(args.input, split_at='undistort', frame_callback=vis):
            if vis.is_closed:
                break
            vis(img, result)
        # Headless backend ('none') never sets is_closed -- wait_for_close
        # would block forever -- so skip the wait there even when --wait is set.
        if args.wait and args.display != "none":
            vis.wait_for_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Demonstrates perspective transform followed by YOLO detection on the input stream.'
    )
    parser.add_argument("input", help="Image or video file to process")
    parser.add_argument(
        "--display",
        choices=["none", "opencv", "vulkan", "console", "iterm2", "auto"],
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

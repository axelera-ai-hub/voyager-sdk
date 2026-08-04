#!/usr/bin/env python3
# Copyright Axelera AI, 2026
"""Bottle cascade demo: detect bottles -> OC-Sort tracker -> ResNet50 classification per tracked ROI.

Three-stage pipeline (mirrors ax_models/reference/cascade/with_tracker/ssd-mobilenetv1-resnet50.yaml):
- detections: YOLOv8n on full frame, filtered for bottles with confidence >= 0.4.
  Note: the reference YAML uses SSD-MobileNetV1 (300x300) whose decoder has no rt2 equivalent;
  YOLOv8n is used instead with matching confidence and NMS thresholds.
- tracking: OC-Sort tracker maintains identity of detected bottles across frames.
- classifications: ResNet50 ImageNet classification on the 10 tracked bottles closest to
  the frame centre, selected via op.sort_by(by='center_distance', descending=False, top=10).
  This directly mirrors the YAML's 'which: CENTER, top_k: 10' using the center_distance
  sort key.

Auto-generated from rt_demo.py's @publicdemo `bottle_demo` by
tools/export_pipeline_builder_demo.py.  Edit the original demo in rt_demo.py
rather than this file -- regenerate with
`python tools/export_pipeline_builder_demo.py bottle_demo`.

Usage:
    python bottle_demo.py PATH [--display opencv|console|none|auto] [--no-wait]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from axelera.runtime import op, cv, display
import cv2

# Populate this directory once with the .axm files referenced below
# (`axdownloadmodel --axm <name>` inside this directory).
MODEL_DIR = str(Path.home() / ".cache" / "axelera" / "runtime2")


def main(args):
    with display.App(renderer=args.display) as visualizer:
        vis = visualizer.create_window('bottle_demo', (args.window_width, args.window_height))
        BOX_COLOR = (100, 200, 220)  # BGR for tracked bottles

        pipeline = op.seq(
            op.color_convert('RGB'),
            op.letterbox(640, 640),
            op.to_tensor(),
            op.load(f'{MODEL_DIR}/yolov8n-coco.axm'),
            op.decode_detections(algo='yolov8', num_classes=80, confidence_threshold=0.4),
            op.nms(iou_threshold=0.5, max_boxes=300),
            op.to_image_space(),
            op.ax_detection(class_id_type=op.CocoClasses),
            op.filter(class_ids=[op.CocoClasses.bottle]),
            op.tracker(algo='oc-sort'),
            op.sort_by(by='center_distance', descending=False, top=10),
            op.for_each(
                'classifications',
                op.crop_roi(property='bbox'),
                op.resize(width=224, height=224),
                op.to_tensor(),
                op.normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                op.load(f'{MODEL_DIR}/resnet50-imagenet.axm'),
                op.softmax(),
                op.ax_classification(class_id_type=op.ImagenetClasses),
                op.top_k(k=1),
            ),
        )

        with cv.create_source(args.input, backend=args.backend) as source:
            for img, result in pipeline.stream(source):
                if vis.is_closed:
                    break
                frame = img.convert('BGR').to_numpy().copy()
                h, w = frame.shape[:2]
                for track, top1 in zip(result.input, result.classifications):
                    x0, y0, x1, y1 = track.predicted_bbox.frame_pixels(w, h)
                    cv2.rectangle(frame, (x0, y0), (x1, y1), BOX_COLOR, 2)
                    cls_name = top1[0].class_id.name if top1 else 'unknown'
                    label = (
                        f'#{track.track_id} {cls_name} {top1[0].score:.2f}'
                        if top1
                        else f'#{track.track_id}'
                    )
                    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 0.45, 2
                    (tw, th), baseline = cv2.getTextSize(label, font, scale, thickness)
                    ty = max(20, y0 - 6)
                    cv2.rectangle(
                        frame,
                        (x0, ty - th - baseline),
                        (x0 + tw, ty + baseline),
                        (0, 0, 0),
                        cv2.FILLED,
                    )
                    cv2.putText(frame, label, (x0, ty), font, scale, BOX_COLOR, thickness)
                vis(frame, [])
        # Headless backend ('none') never sets is_closed -- wait_for_close
        # would block forever -- so skip the wait there even when --wait is set.
        if args.wait and args.display != "none":
            vis.wait_for_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Bottle cascade demo: detect bottles -> OC-Sort tracker -> ResNet50 classification per tracked ROI.'
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

#!/usr/bin/env python3
# Copyright Axelera AI, 2026
"""Object tracking with full state lifecycle, detection correlation, and filtering.

Demonstrates key tracker features:

1. op.filter(): Protocol-based class filtering
   - Filter detections to track only vehicles (cars, trucks, buses, motorcycles)
   - Filter works with any object implementing HasClassId protocol

2. return_all_states=True: Expose full track lifecycle
   - Visualization uses dual-property strategy:
     * COLOR = track identity (consistent color per track_id)
     * ALPHA = track state (new=70%, tracked=100%, lost=40%, removed=20%)
   - See new/lost/removed tracks for debugging
   - Default (False) returns only active tracks for MOT evaluation

3. latest_det_id: O(1) detection correlation
   - Each TrackedObject links back to its matched detection
   - Access detection metadata (confidence, class, etc.)
   - Value >= 0: valid detection match
   - Value -1: no detection (lost/removed track)

Use cases:
- Filter specific object classes before tracking
- Debug tracking behavior (why did track get lost?)
- Analyze detection-to-track assignments
- Access raw detection scores for post-processing

Auto-generated from rt_demo.py's @publicdemo `tracking` by
tools/export_pipeline_builder_demo.py.  Edit the original demo in rt_demo.py
rather than this file -- regenerate with
`python tools/export_pipeline_builder_demo.py tracking`.

Usage:
    python tracking.py PATH [--display opencv|console|none|auto] [--no-wait]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from axelera.runtime import op, cv, display
from collections import Counter

# Populate this directory once with the .axm files referenced below
# (`axdownloadmodel --axm <name>` inside this directory).
MODEL_DIR = str(Path.home() / ".cache" / "axelera" / "runtime2")


def main(args):
    with display.App(renderer=args.display) as visualizer:
        vis = visualizer.create_window('tracking', (args.window_width, args.window_height))
        detect_pipeline = op.seq(
            op.color_convert('RGB'),
            op.letterbox(640, 640),
            op.to_tensor(),
            op.load(f'{MODEL_DIR}/yolov8n-coco.axm'),
            op.decode_detections(algo='yolov8', num_classes=80, confidence_threshold=0.25),
            op.nms(iou_threshold=0.45, max_boxes=300),
            op.to_image_space(),
            op.ax_detection(class_id_type=op.CocoClasses),
            # Filter to track only vehicles - demonstrates protocol-based filtering
            op.filter(
                class_ids=[
                    op.CocoClasses.car,
                    op.CocoClasses.truck,
                    op.CocoClasses.bus,
                    op.CocoClasses.motorcycle,
                ]
            ),
        )
        tracker = op.tracker(algo='bytetrack', return_all_states=True)

        for img in cv.create_source(args.input):
            if vis.is_closed:
                break

            with op.frame_context(img):
                detections = detect_pipeline(img)
                tracked = tracker(detections)

            # Show state distribution
            state_counts = Counter(t.state for t in tracked)

            print(f'  State breakdown: {dict(state_counts)}')

            # Show detection correlation for first few tracks
            print('  Track -> Detection mapping:')
            for obj in tracked[:5]:
                if obj.latest_det_id >= 0 and obj.latest_det_id < len(detections):
                    det = detections[obj.latest_det_id]
                    print(
                        f'    Track {obj.track_id} ({obj.state.name}): '
                        f'det[{obj.latest_det_id}] score={det.score:.2f} class={det.class_id.name}'
                    )
                else:
                    print(f'    Track {obj.track_id} ({obj.state.name}): no detection (lost)')

            vis(img, tracked)
        # Headless backend ('none') never sets is_closed -- wait_for_close
        # would block forever -- so skip the wait there even when --wait is set.
        if args.wait and args.display != "none":
            vis.wait_for_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Object tracking with full state lifecycle, detection correlation, and filtering.'
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

#!/usr/bin/env python3
# Copyright Axelera AI, 2026
"""Standard ImageNet classification pipeline.
Preprocesses images using ImageNet normalization and outputs top-5 predictions.

Auto-generated from rt_demo.py's @publicdemo `classification` by
tools/export_pipeline_builder_demo.py.  Edit the original demo in rt_demo.py
rather than this file -- regenerate with
`python tools/export_pipeline_builder_demo.py classification`.

Usage:
    python classification.py PATH [--display opencv|console|none|auto] [--no-wait]
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
        vis = visualizer.create_window('classification', (args.window_width, args.window_height))
        pipeline = op.seq(
            op.color_convert('RGB'),
            # Standard ImageNet preprocessing
            op.resize(size=256, half_pixel_centers=True),
            op.center_crop((224, 224)),
            op.to_tensor(),
            op.normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            # Inference
            op.load(f'{MODEL_DIR}/squeezenet1.0-imagenet.axm'),
            op.softmax(),
            op.ax_classification(class_id_type=op.ImagenetClasses),
            op.top_k(k=5),
        )

        label_layer = None
        for img in cv.create_source(args.input):
            if vis.is_closed:
                break

            top5 = pipeline(img)
            lines = [
                f'{rank}. {c.class_id.name} {c.score * 100:.1f}%' for rank, c in enumerate(top5, 1)
            ]
            label_layer = vis.text(
                '2%, 8%',
                '\n'.join(lines),
                anchor_x='left',
                anchor_y='top',
                font_size=24,
                existing=label_layer,
            )
            vis(img, None)
        # Headless backend ('none') never sets is_closed -- wait_for_close
        # would block forever -- so skip the wait there even when --wait is set.
        if args.wait and args.display != "none":
            vis.wait_for_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Standard ImageNet classification pipeline.')
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

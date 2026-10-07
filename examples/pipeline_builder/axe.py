#!/usr/bin/env python3
# Copyright Axelera AI, 2026
"""Package a whole pipeline into one .axe file, then load it back and run it.

The round trip:
* Build a detection pipeline with named operators
* Save it as a portable .axe file
* Load the .axe and read what it says about itself
* Retune a named operator in the loaded pipeline
* Run inference on it

An .axe is a ZIP holding a `pipeline.toml` plus the model files it names:
the operator sequence (preprocessing, model, postprocessing), the embedded
`.axm`, and the metadata (name, description, task, input shape).

It holds the pipeline as it was built. `pipeline.optimized()` picks fused
kernels for the machine it runs on, so `save_axe` refuses an optimized
pipeline; call `.optimized()` after loading, on the machine that will run it.

Both formats, and what an .axe cannot carry, are described in
docs/reference/pipeline-builder/model-formats.md.

Usage:
    python axe.py PATH [--display opencv|console|none|auto] [--no-wait]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from axelera.runtime import op, cv, display
import os

# Populate this directory once with the .axm files referenced below
# (`axdownloadmodel --axm <name>` inside this directory).
MODEL_DIR = str(Path.home() / ".cache" / "axelera" / "runtime2")


def main(args):
    with display.App(renderer=args.display) as visualizer:
        vis = visualizer.create_window('axe', (args.window_width, args.window_height))
        import tempfile

        pipeline = op.seq(
            op.color_convert('RGB'),
            op.letterbox(640, 640),
            op.to_tensor(),
            op.load(f'{MODEL_DIR}/yolov8n-coco.axm'),
            op.decode_detections(
                algo='yolov8',
                num_classes=80,
                confidence_threshold=0.25,
                name='decode',
            ),
            op.nms(iou_threshold=0.45, max_boxes=300, name='nms'),
            op.to_image_space(),
            op.ax_detection(class_id_type=op.CocoClasses),
        )

        # TemporaryDirectory removes the .axe (and its dir) automatically on exit,
        # avoiding mktemp's deprecated name race and the manual cleanup below.
        with tempfile.TemporaryDirectory() as tmpdir:
            axe_path = os.path.join(tmpdir, 'pipeline.axe')
            print(f'Saving pipeline to {axe_path}...')
            pipeline.save_axe(
                axe_path,
                name='YOLOv8n COCO Detector',
                description='Object detection using YOLOv8n trained on COCO dataset',
                task='detection',
            )

            print(f'Loading the .axe back: {axe_path}')
            # An .axe stores the pipeline unoptimized on purpose, because fusion picks
            # kernels for the machine that runs it. Optimizing here is that machine.
            detector = op.load(axe_path).optimized()

            print('What the .axe says about itself...')
            info = detector.info()
            print(f'  Name: {info["name"]}')
            print(f'  Description: {info["description"]}')
            print(f'  Task: {info["task"]}')
            print(f'  Schema version: {info["schema_version"]}')
            print(f'  Input shape: {info["input_shape"]} ({info["input_format"]})')
            print()

            # A loaded pipeline is still a pipeline: reach a named operator with
            # node(name=...) and set the field, the same as before it was saved.
            nms = detector.node(name='nms')
            print(f'Retuning the loaded pipeline: nms.max_boxes {nms.max_boxes} -> 100')
            nms.max_boxes = 100
            print()

            print('Running inference on the loaded pipeline...')

            for img in cv.create_source(args.input):
                if vis.is_closed:
                    break
                detections = detector(img)
                print(f'  {img.source}: {len(detections)} detections')
                for det in detections[:5]:  # Show first 5
                    print(f'    - {det.class_id.name}: {det.score:.2%} at {det.bbox}')
                if len(detections) > 5:
                    print(f'    ... and {len(detections) - 5} more')

                vis(img, detections)
        # Headless backend ('none') never sets is_closed -- wait_for_close
        # would block forever -- so skip the wait there even when --wait is set.
        if args.wait and args.display != "none":
            vis.wait_for_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Package a whole pipeline into one .axe file, then load it back and run it.'
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

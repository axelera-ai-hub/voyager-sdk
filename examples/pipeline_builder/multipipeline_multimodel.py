#!/usr/bin/env python3
# Copyright Axelera AI, 2026
"""Five different models on five streams, one distinct pipeline each.

The other multipipeline_* demos run one pipeline copied N times
(pipeline.copy()), so every stream runs the same model. This one instead
builds five independent pipelines, each with its own model and task, and
runs them together via as_available(): YOLOv8n detection, YOLOv8s
detection, YOLOv8l pose, YOLOv8s segmentation, and ResNet-50
classification. Five distinct models share the one AIPU core pool, so the
scheduler time-shares cores across them (see scheduler.py core allocation).

Fewer inputs than pipelines are reused round-robin, so every model gets a
stream even from a single video; inputs beyond the five pipelines are unused.
Comma-separate the input paths to drive each stream from a different source.

Usage:
    python multipipeline_multimodel.py PATH[,PATH...] [--display opencv|console|none|auto] [--no-wait] [--frames N]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from axelera.runtime import op, cv, display, scheduler
import threading
import contextlib
import itertools

# Populate this directory once with the .axm files referenced below
# (`axdownloadmodel --axm <name>` inside this directory).
MODEL_DIR = str(Path.home() / ".cache" / "axelera" / "runtime2")


class _MultiWindow:
    """Lazily opens one display window per stream index."""

    def __init__(self, app, base, size):
        self._app = app
        self._size = size
        self._windows = [base]
        self._ended = set()
        self._lock = threading.Lock()

    def __getitem__(self, i):
        with self._lock:
            while i >= len(self._windows):
                self._windows.append(
                    self._app.create_window(f"Stream {len(self._windows)}", self._size)
                )
        return self._windows[i]

    def close_if_open(self, i):
        # Stream i ran out of frames.  Close its window and stop counting it
        # towards is_closed: a stream reaching its end is not the user asking
        # to quit, and must not cut the still-running streams short.
        with self._lock:
            self._ended.add(i)
            if i < len(self._windows):
                self._windows[i].close()

    @property
    def is_closed(self):
        # True as soon as the user closes the window of ANY stream that is
        # still running; reporting window 0 alone would ignore every other one.
        with self._lock:
            live = [w for i, w in enumerate(self._windows) if i not in self._ended]
        return any(w.is_closed for w in live)

    def wait_for_close(self):
        for wnd in list(self._windows):
            wnd.wait_for_close()

    def __call__(self, img, *result_args):
        return self._windows[0](img, *result_args)

    def __getattr__(self, name):
        # Guard every private attribute, not just _windows: __getattr__ reads
        # self._windows, so a miss on any of them would recurse.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._windows[0], name)


def main(args):
    with display.App(renderer=args.display) as visualizer:
        vis = _MultiWindow(
            visualizer,
            visualizer.create_window(
                'multipipeline_multimodel', (args.window_width, args.window_height)
            ),
            (args.window_width, args.window_height),
        )
        pipelines = [
            (
                'detection-n',
                op.seq(
                    op.color_convert('RGB'),
                    op.letterbox(640, 640),
                    op.to_tensor(),
                    op.load(f'{MODEL_DIR}/yolov8n-coco.axm'),
                    op.decode_detections(algo='yolov8', num_classes=80, confidence_threshold=0.25),
                    op.nms(iou_threshold=0.45, max_boxes=300),
                    op.to_image_space(),
                    op.ax_detection(class_id_type=op.CocoClasses),
                ),
            ),
            (
                'detection-s',
                op.seq(
                    op.color_convert('RGB'),
                    op.letterbox(640, 640),
                    op.to_tensor(),
                    op.load(f'{MODEL_DIR}/yolov8s-coco-onnx.axm'),
                    op.decode_detections(algo='yolov8', num_classes=80, confidence_threshold=0.25),
                    op.nms(iou_threshold=0.45, max_boxes=300),
                    op.to_image_space(),
                    op.ax_detection(class_id_type=op.CocoClasses),
                ),
            ),
            (
                'pose',
                op.seq(
                    op.color_convert('RGB'),
                    op.letterbox(640, 640),
                    op.to_tensor(),
                    op.load(f'{MODEL_DIR}/yolov8lpose-coco-onnx.axm'),
                    op.decode_pose(algo='yolov8', num_keypoints=17),
                    op.nms(iou_threshold=0.45, max_boxes=300),
                    op.to_image_space(keypoint_cols=range(6, 57, 3)),
                    op.ax_pose(),
                ),
            ),
            (
                'segmentation',
                op.seq(
                    op.color_convert('RGB'),
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
                ),
            ),
            (
                'classification',
                op.seq(
                    op.color_convert('RGB'),
                    op.resize(size=256, half_pixel_centers=True),
                    op.center_crop((224, 224)),
                    op.to_tensor(),
                    op.normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                    op.load(f'{MODEL_DIR}/resnet50-imagenet.axm'),
                    op.softmax(),
                    op.ax_classification(class_id_type=op.ImagenetClasses),
                    op.top_k(k=5),
                ),
            ),
        ]
        # Title each stream's window with its model/task so the display makes clear
        # which window shows which model. Indexing vis[i] lazily creates window i
        # (default title 'Stream i'); options(0, title=...) overrides that label.
        for i, (label, _) in enumerate(pipelines):
            print(f'stream {i}: {label}')
            vis[i].options(
                0,
                title=label,
                title_anchor_x='center',
                title_position='50%, 0%',
                # Draw the geometry only: an empty format suppresses the per-box class
                # name and score. The metadata still reaches the caller in `result`,
                # it just isn't painted over five tiled windows.
                bbox_label_format='',
            )

        paths = [p.strip() for p in args.input.split(',')]
        # Unlike the sibling multipipeline_* demos, the stream count here is fixed by
        # the pipeline list, not by the inputs, so surplus paths are never opened.
        # Say so rather than dropping them silently.
        if len(paths) > len(pipelines):
            print(
                f'note: {len(paths)} inputs given but only {len(pipelines)} pipelines; '
                f'ignoring {", ".join(paths[len(pipelines) :])}'
            )
        # One source per pipeline; reuse inputs round-robin when fewer are given so
        # every model still gets a stream (a single video drives all five).
        sources = [
            cv.create_source(paths[i % len(paths)], backend=args.backend)
            for i in range(len(pipelines))
        ]
        max_frames = args.frames if getattr(args, 'frames', 0) else None

        # Each pipeline is already distinct, so stream it directly; no copy() as in
        # the single-model multipipeline_* demos.
        streamables = [
            pipe.stream(itertools.islice(source, max_frames) if max_frames else source)
            for (_, pipe), source in zip(pipelines, sources)
        ]

        try:
            for stream_id, img, result in scheduler.current().as_available(streamables):
                # Stop draining once the user closes the display (see multipipeline_aa).
                if vis.is_closed:
                    break
                if img is None:
                    vis.close_if_open(stream_id)
                    continue
                vis[stream_id](img, result)
        finally:
            # Tear down each stream generator and its background decoder thread;
            # leaving a decoder running risks a GIL abort at interpreter shutdown.
            # Both closes are idempotent and independent, so suppress per item.
            # (Inlined rather than calling the _close_multistream helper so this
            # demo exports self-contained via export_pipeline_builder_demo.py.)
            for streamable in streamables:
                with contextlib.suppress(Exception):
                    streamable.close()
            for source in sources:
                with contextlib.suppress(Exception):
                    source.close()
        # Headless backend ('none') never sets is_closed -- wait_for_close
        # would block forever -- so skip the wait there even when --wait is set.
        if args.wait and args.display != "none":
            vis.wait_for_close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Five different models on five streams, one distinct pipeline each.'
    )
    parser.add_argument(
        "input",
        help="Image or video file to process. Comma-separate several paths to run one stream per source, each in its own window.",
    )
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
    parser.add_argument(
        "--frames",
        type=int,
        default=0,
        help="Process at most this many frames per stream (default: no limit)",
    )
    main(parser.parse_args())

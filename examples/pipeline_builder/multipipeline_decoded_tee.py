#!/usr/bin/env python3
# Copyright Axelera AI, 2026
"""YOLOv8 detection with a decoded-frame tee for low-latency display.

Shows frame_callback plus split_at: split_at names an operator that splits
the pipeline into a prefix (runs on the calling thread) and a main tail
(runs on the worker pool); frame_callback fires right after the prefix,
with the value that operator returned, before inference starts.

split_at is generic: it can name any top-level operator (this is only one
example), and frame_callback receives whatever that operator returns --
image, tensor, or any other type -- the same value that continues on to
the main tail, so it cannot diverge from what the model sees.

Two rules when choosing a split point: split_at matches top-level
operators only (it does not descend into a nested op.seq(...)), and a
coordinate-recording operator (op.letterbox, op.resize) must come after
the split, in the main tail, so its geometry reaches op.to_image_space().

What to watch for: each printed line reports display_ahead_by, the gap in
milliseconds between the frame currently on screen and the frame whose
detections just arrived. With more streams than the AIPU has spare
capacity for, this number stays positive and playback never stalls
waiting for a slow stream's inference; boxes appear a beat behind the
live picture, but the video itself does not stutter.

Usage:
    python multipipeline_decoded_tee.py PATH[,PATH...] [--display opencv|console|none|auto] [--no-wait]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from axelera.runtime import op, cv, display, scheduler
import threading
import contextlib
import math

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
                'multipipeline_decoded_tee', (args.window_width, args.window_height)
            ),
            (args.window_width, args.window_height),
        )
        from functools import partial

        pipeline = op.seq(
            op.color_convert('RGB'),
            op.perspective(
                matrix=[
                    math.cos(math.radians(45)),
                    -math.sin(math.radians(45)),
                    320 * (1 - math.cos(math.radians(45))) + 240 * math.sin(math.radians(45)),
                    math.sin(math.radians(45)),
                    math.cos(math.radians(45)),
                    240 * (1 - math.cos(math.radians(45))) - 320 * math.sin(math.radians(45)),
                    0,
                    0,
                    1,
                ],
                width=640,
                height=480,
                # Name the perspective op so stream(split_at=...) can split the
                # pipeline here: the prefix (color_convert + perspective) runs in the
                # stream thread and its output is what the frame_callback tee sees.
                name='undistort',
            ),
            op.letterbox(640, 640),
            op.to_tensor(),
            op.load(f'{MODEL_DIR}/yolov8n-coco.axm'),
            op.decode_detections(algo='yolov8', num_classes=80, confidence_threshold=0.25),
            op.nms(iou_threshold=0.45, max_boxes=300),
            op.to_image_space(),
            op.ax_detection(class_id_type=op.CocoClasses),
        )
        paths = [p.strip() for p in args.input.split(',')]
        sources = [cv.create_source(path, backend=args.backend) for path in paths]

        # Presentation timestamp of the most recently decoded frame per stream,
        # recorded by the tee. Comparing it against the pts of the frame whose
        # inference result just arrived shows how far display runs ahead of inference.
        last_decoded_pts = {}

        def frame_callback(stream_id, frame):
            # Called pre-inference on the stream thread with the preprocessed frame:
            # because we split at 'undistort', this is the perspective-corrected image,
            # not the raw decoded frame. Runs on the frame's critical path, so keep it
            # light and fast: record the frame's pts and hand it to the display.
            last_decoded_pts[stream_id] = frame.presentation_timestamp
            if not vis.is_closed:
                vis[stream_id](frame)

        streamables = [
            pipeline.copy().stream(
                source, frame_callback=partial(frame_callback, i), split_at='undistort'
            )
            for i, source in enumerate(sources)
        ]

        try:
            for stream_id, img, result in scheduler.current().as_available(streamables):
                # Stop draining once the user closes the display; breaking runs the
                # generator's teardown, which stops the stream threads.
                if vis.is_closed:
                    break
                if img is None:
                    vis.close_if_open(stream_id)
                    continue
                pts = img.presentation_timestamp
                n_det = len(result) if result else 0
                # How far the tee (display) has run ahead of inference: pts of the
                # latest decoded frame minus pts of the frame this result belongs to.
                # With multiple real streams sharing one AIPU pool this stays above
                # zero; that gap is the decoupling the tee buys, measured with pts alone.
                lag_ms = (last_decoded_pts[stream_id] - pts) / 1000
                # Overlay boxes on the frame this result belongs to (the tee already
                # showed it undecorated; this is inference catching up).
                vis[stream_id](img, result)
                print(
                    f'stream={stream_id}  pts={pts}us  detections={n_det}  '
                    f'display_ahead_by={lag_ms:.0f}ms'
                )
        finally:
            # Tear down each stream generator and its background decoder thread;
            # leaving a decoder running risks a GIL abort at interpreter shutdown.
            # Both closes are idempotent and independent, so suppress per item.
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
        description='YOLOv8 detection with a decoded-frame tee for low-latency display.'
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
    main(parser.parse_args())

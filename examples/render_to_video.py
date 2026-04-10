#!/usr/bin/env python
# Copyright Axelera AI, 2025
"""Render an inference stream using OpenGL, and

This demo shows how to:
  * Save a composite output video of multiple input sources
  * Poll the latest rendered frames via Surface.latest
  * Write 600 frames to a file at 60fps (about 10 seconds output) with cv2
"""

from __future__ import annotations

import queue
import threading
import time

from axelera import types
from axelera.app import create_inference_stream, display, config

framework = config.env.framework
NETWORK = "yolov5m-v7-coco-tracker"
SOURCES = [
    str(framework / "media/traffic2_480p.mp4@60"),
    str(framework / "media/traffic3_480p.mp4@60"),
]
SURFACE_SIZE = (848 * 2, 480)


import cv2


class Writer:
    def __init__(self, surface, filename, fps, frame_size):
        cc = cv2.VideoWriter_fourcc(*'mp4v')
        self._writer = cv2.VideoWriter(filename, cc, fps, frame_size)
        self._fps = fps
        self._q = queue.Queue()
        self._thread = threading.Thread(target=self._worker)
        self._stop_event = threading.Event()
        self._surface = surface

    def _worker(self):
        last_frame = time.time()
        n = 0
        while not self._stop_event.is_set():
            self._writer.write(self._surface.latest.asarray(types.ColorFormat.BGR))
            now = time.time()
            time.sleep(max(0, 1 / self._fps - (now - last_frame)))
            last_frame = now
            if n >= 600:
                self._stop_event.set()
            n += 1

        self._writer.release()

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop_event.set()
        self._thread.join()


def inference_worker(surface: display.Surface, stream):
    try:
        max_fps = max(s.fps for s in stream.sources.values())
        print(f"Source max FPS: {max_fps}")
        with Writer(surface, "output.mp4", fps=max_fps, frame_size=SURFACE_SIZE) as writer:
            for frame_result in stream:
                surface.push(frame_result.image, frame_result.meta, frame_result.stream_id)
                if writer._stop_event.is_set():
                    break
    finally:
        stream.stop()


def main():
    stream = create_inference_stream(network=NETWORK, sources=SOURCES)
    with display.App(renderer=True) as app:
        surface = app.create_surface(SURFACE_SIZE)
        app.start_thread(inference_worker, (surface, stream), name="InferenceThread")
        app.run(interval=1 / 60)


if __name__ == "__main__":
    main()

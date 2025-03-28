#!/usr/bin/env python
# Copyright Axelera AI, 2025
# The simplest demo application within 50 lines of code

import os

import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst

framework = os.environ.get("AXELERA_FRAMEWORK", '.')

from axelera.app import logging_utils
from axelera.app.display import App
from axelera.app.stream import create_inference_stream

logging_utils.configure_logging(logging_utils.Config(logging_utils.INFO))

# TODO: generate a test video for this tutorial
source = os.path.join(framework, "media/traffic1_1080p.mp4")

stream = create_inference_stream(
    network="t3-learn-axtaskmeta",
    sources=[source],
    pipe_type='torch-aipu',
)


def main(window, stream):
    for frame_result in stream:
        window.show(frame_result.image, frame_result.meta)

        # Print metadata to terminal
        print(
            f"Classified as {frame_result.meta.classifier.class_id} with score {frame_result.meta.classifier.score}"
        )


with App(visible=True, opengl=stream.manager.hardware_caps.opengl) as app:
    wnd = app.create_window("Business logic demo", (900, 600))
    app.start_thread(main, (wnd, stream), name='InferenceThread')
    app.run(interval=1 / 10)
stream.stop()

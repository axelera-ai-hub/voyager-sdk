#!/usr/bin/env python
# Copyright Axelera AI, 2024
# The simplest demo application within 50 lines of code

from axelera.app import config, display
from axelera.app.stream import create_inference_stream

source = str(config.env.framework / "media/traffic1_1080p.mp4")

stream = create_inference_stream(
    network="t3-learn-axtaskmeta",
    sources=[source],
    pipe_type='torch-aipu',
)


def main(window, stream):
    for frame_result in stream:
        window.show(frame_result.image, frame_result.meta)

        # Print metadata to terminal
        class_id, score = frame_result.meta["classifications"].get_result(0)
        print(f"Classified as {class_id} with score {score}")


with display.App(renderer=True, opengl=stream.hardware_caps.opengl) as app:
    wnd = app.create_window("Business logic demo", (900, 600))
    app.start_thread(main, (wnd, stream), name='InferenceThread')
    app.run(interval=1 / 10)
stream.stop()

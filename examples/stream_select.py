#!/usr/bin/env python
# Copyright Axelera AI, 2025

'''
This example runs all 4 streams and the user can remove and add streams from keyboard.
The benefit of this approach is the we don't need to set pipeline to paused state so adding and removing streams is instant, but we need to know RTSP, and they need to be live.
'''


import threading

import gi

gi.require_version('Gst', '1.0')
from gi.repository import Gst

from axelera.app import config, logging_utils
from axelera.app.display import App
from axelera.app.stream import create_inference_stream

# Define all stream sources
source = {
    0: "rtsp://127.0.0.1:8554/0",
    1: "rtsp://127.0.0.1:8554/1",
    2: "rtsp://127.0.0.1:8554/0",
    3: "rtsp://127.0.0.1:8554/1",
}

streams = [0, 1, 2, 3]

hw_caps = config.HardwareCaps(
    config.HardwareEnable.detect,
    config.HardwareEnable.detect,
    config.HardwareEnable.detect,
)


def control_func(stream):
    while True:
        print(f"Current streams: {streams}")
        user_input = input(
            f"Select stream to toggle [{min(source.keys())}-{max(source.keys())}]: "
        )

        if not user_input.isdigit() or int(user_input) not in source:
            print(f"Select one of {list(source.keys())}")
            continue

        idx = int(user_input)

        if idx in streams:
            streams.remove(idx)
        else:
            streams.append(idx)
        stream.stream_select(','.join(map(str, streams)))


def main(window, stream):
    control = threading.Thread(
        target=control_func, args=(stream,), name="ControlThread", daemon=True
    )
    control.start()
    for frame_result in stream:
        if frame_result.image:
            window.show(frame_result.image, frame_result.meta, frame_result.stream_id)


if __name__ == '__main__':
    stream = create_inference_stream(
        network="yolov8n-license-plate",
        sources=list(source.values()),
        pipe_type='gst',  # Only gst pipe is supported for this example
        log_level=logging_utils.INFO,
        hardware_caps=config.HardwareCaps(
            vaapi=config.HardwareEnable.detect,
            opencl=config.HardwareEnable.detect,
            opengl=config.HardwareEnable.detect,
        ),
    )
    try:
        with App(visible=True, opengl=stream.manager.hardware_caps.opengl) as app:
            wnd = app.create_window("Stream select demo", (900, 600))
            app.start_thread(main, (wnd, stream), name='InferenceThread')
            app.run(interval=1 / 10)

    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        stream.stop()

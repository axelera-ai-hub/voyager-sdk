#!/usr/bin/env python
# Copyright Axelera AI, 2025

'''
To test it we need to run an RTSP server with at least 2 streams

./examples/stream_toggle.py

This example initially runs only 1 stream, and from keyboard input user can toggle max of 4 streams on the fly.
In this example by adding a stream, new branch in pipeline is getting created while pipeline is in paused state.
After transitioning to playing state all rtspsrc elements renegotiate with cameras or rtsp_server using RTP protocol, and after that the cameras or RTSP server and start streaming, then we transition all elements of the pipeline to playing state.
With this approach we have a delay of aproximately 3 seconds in the handshake protocol between cameras and pipeline
'''

import threading

from axelera.app import config, logging_utils
from axelera.app.display import App
from axelera.app.stream import create_inference_stream

source = {
    0: "rtsp://127.0.0.1:8554/0",  # Initial one
    1: "rtsp://127.0.0.1:8554/1",
    2: "rtsp://127.0.0.1:8554/2",
    3: "rtsp://127.0.0.1:8554/3",
}

source_map = {0: 0}


def control_func(stream):
    while True:
        user_input = input(
            f"Select stream to toggle [{min(source.keys())}-{max(source.keys())}]: "
        )

        if not user_input.isdigit() or int(user_input) not in source:
            print(f"Select one of {list(source.keys())}")
            continue

        idx = int(user_input)
        if idx in source_map:
            stream.remove_source(source_map[idx])
            del source_map[idx]
        else:
            source_map[idx] = stream.add_source(source[idx], idx)


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
        sources=[source[0]],
        pipe_type='gst',  # Only gst pipe is supported
        log_level=logging_utils.INFO,
        hardware_caps=config.HardwareCaps(
            vaapi=config.HardwareEnable.detect,
            opencl=config.HardwareEnable.detect,
            opengl=config.HardwareEnable.detect,
        ),
    )
    try:
        with App(visible=True, opengl=stream.manager.hardware_caps.opengl) as app:
            wnd = app.create_window("Stream toggle demo", (900, 600))
            app.start_thread(main, (wnd, stream), name='InferenceThread')
            app.run(interval=1 / 10)
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f'Error: {e}')
    finally:
        stream.stop()

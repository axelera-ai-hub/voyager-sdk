#!/usr/bin/env python
# Copyright Axelera AI, 2025
# Extended app with additional config, showing advanced usage of metadata
import time

from axelera.app import config, display, inf_tracers, logging_utils
from axelera.app.stream import create_inference_stream

framework = config.env.framework
tracers = inf_tracers.create_tracers('core_temp', 'end_to_end_fps', 'cpu_usage')
stream = create_inference_stream(
    network="yolov5m-v7-coco-tracker",
    sources=[
        str(framework / "media/traffic1_1080p.mp4"),
        str(framework / "media/traffic2_1080p.mp4"),
    ],
    pipe_type='gst',
    log_level=logging_utils.INFO,  # INFO, DEBUG, TRACE
    hardware_caps=config.HardwareCaps(
        vaapi=config.HardwareEnable.detect,
        opencl=config.HardwareEnable.detect,
        opengl=config.HardwareEnable.detect,
    ),
    tracers=tracers,
    specified_frame_rate=10,
    # rtsp_latency=500,
)


def main(window, stream):
    window.options(0, title="Traffic 1")
    window.options(1, title="Traffic 2")
    last_temp_report = time.time()
    CLASS = stream.manager.detections.classes
    for frame_result in stream:
        window.show(frame_result.image, frame_result.meta, frame_result.stream_id)
        core_temp = stream.get_all_metrics()['core_temp']
        end_to_end_fps = stream.get_all_metrics()['end_to_end_fps']
        cpu_usage = stream.get_all_metrics()['cpu_usage']
        if (now := time.time()) - last_temp_report > 1:
            last_temp_report = now
            metrics = [
                f"Core temp: {core_temp.value}°C",
                f"End-to-end FPS: {end_to_end_fps.value:.1f}",
                f"CPU usage: {cpu_usage.value:.1f}%",
            ]
            print('='.center(90, '='))
            print(' | '.join(metrics).center(90))
            print('='.center(90, '='))

        # # Print car, vehicle and person count to terminal
        print(f"Found {sum(d.is_car for d in frame_result.detections)} car(s)")
        VEHICLE = ('car', 'truck', 'motorcycle')
        print(f"Found {sum(d.is_a(VEHICLE) for d in frame_result.detections)} vehicle(s)")
        # (d.class_id == 0) equivalent to (d.label == CLASS.person)
        print(f"Found {sum(d.label == CLASS.person for d in frame_result.detections)} person(s)")


with display.App(
    visible=True,
    opengl=stream.manager.hardware_caps.opengl,
    buffering=not stream.is_single_image(),
) as app:
    wnd = app.create_window("Advanced usage demo", (900, 600))
    app.start_thread(main, (wnd, stream), name='InferenceThread')
    app.run(interval=1 / 10)
stream.stop()

---
title: "Extended Application with Metrics and Render Control"
---
# Extended Application with Metrics and Render Control

This example builds on the basic application pattern to demonstrate advanced SDK features: hardware capability detection, performance tracers (core temperature, FPS, CPU usage), dynamic render configuration, and programmatic access to detection metadata such as class filtering and on-screen text overlays.

Use this when you need runtime telemetry, hardware-accelerated decoding, or dynamic control over what annotations are displayed.

## What you'll learn

- How to configure hardware capabilities (VA-API, OpenCL, OpenGL)
- How to create and read performance tracers (temperature, FPS, CPU)
- How to dynamically toggle render settings at runtime
- How to filter detections by class and display custom text overlays
- How to set log levels and frame rate limits

## Prerequisites

- Voyager SDK installed and activated
- Sample media files available in `media/` (included with the SDK)
- GStreamer pipeline support (`pipe_type='gst'`)

## Source

> [!TIP]
> **Download**
> This example is included in the SDK at `examples/application_extended.py`.


```python
#!/usr/bin/env python
# Copyright Axelera AI, 2024
# Extended app with additional config, showing advanced usage of metadata
import time

from axelera.app import config, display, inf_tracers, logging_utils
from axelera.app.stream import create_inference_stream

framework = config.env.framework
tracers = inf_tracers.create_tracers('core_temp', 'end_to_end_fps', 'cpu_usage', pipe_type='gst')
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
    allow_hardware_codec=False,
    tracers=tracers,
    specified_frame_rate=10,
    # rtsp_latency=500,
    render_config=config.RenderConfig(
        detections=config.TaskRenderConfig(
            show_annotations=False,
            show_labels=False,
        ),
    ),
)


def toggle_task_render_config(stream, start_time, interval=30, show_time=5):
    """Dynamically toggle render settings for demo/testing
    Toggle render settings every 'interval' seconds, showing labels for 'show_time' seconds.
    """
    now = time.time()
    elapsed = now - start_time
    if (elapsed % interval) < show_time:
        stream.manager.detections.set_render(
            show_annotations=False,
            show_labels=True,
        )
        print(f"Render settings toggled: detection labels shown for {show_time}s")
    else:
        stream.manager.detections.set_render(
            show_annotations=False,
            show_labels=False,
        )


def main(window, stream):
    window.options(0, title="Traffic 1")
    window.options(1, title="Traffic 2")
    counter = window.text(
        '20px, 10%',
        "Vehicles: 00",
    )
    last_temp_report = time.time()
    CLASS = stream.manager.detections.classes
    render_toggle_start = time.time()
    for frame_result in stream:
        toggle_task_render_config(stream, render_toggle_start)
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

        # Print car, vehicle and person count to terminal,
        # and show vehicle count on the window
        VEHICLE = ('car', 'truck', 'motorcycle')
        vehicles = sum(d.is_a(VEHICLE) for d in frame_result.detections)
        counter["text"] = f"Vehicles: {vehicles:02}"
        print(f"Found {sum(d.is_car for d in frame_result.detections)} car(s)")
        print(f"Found {vehicles} vehicle(s)")
        # (d.class_id == 0) equivalent to (d.label == CLASS.person)
        print(f"Found {sum(d.label == CLASS.person for d in frame_result.detections)} person(s)")

        if window.is_closed:
            break


with display.App(
    renderer=True,
    opengl=stream.hardware_caps.opengl,
    buffering=not stream.is_single_image(),
) as app:
    wnd = app.create_window("Advanced usage demo", (900, 600))
    app.start_thread(main, (wnd, stream), name='InferenceThread')
    app.run()
stream.stop()
```

## Key concepts

**Performance tracers** are created with `inf_tracers.create_tracers()` and passed to the stream. Once running, `stream.get_all_metrics()` returns a dictionary of live measurements. This example tracks core temperature, end-to-end FPS, and CPU usage, printing a summary to the terminal every second.

**Hardware capability detection** is configured through `config.HardwareCaps`. Setting each capability to `HardwareEnable.detect` lets the SDK probe the system and use hardware acceleration (VA-API for video decode, OpenCL/OpenGL for rendering) when available. The `allow_hardware_codec=False` flag forces software decoding, which is useful for debugging or environments where hardware codecs are unreliable.

**Dynamic render control** is demonstrated by the `toggle_task_render_config` function, which periodically shows and hides detection labels via `stream.manager.detections.set_render()`. This pattern is useful for dashboards that need to reduce visual clutter or cycle between different annotation views.

**Detection filtering** uses the `is_a()` method and named class attributes (`CLASS.person`, `d.is_car`) to count objects by category. The `window.text()` overlay displays a live vehicle count directly on the video stream.

---

---
title: "Basic Application with Tracking"
---
# Basic Application with Tracking

This example demonstrates the simplest way to build a complete inference application with the Voyager SDK. It runs a YOLOv5 tracker on two video streams, displays the results in a window, and prints vehicle tracking information to the terminal.

Use this as your starting point for any new application that needs object detection and tracking with visual output.

## What you'll learn

- How to create an inference stream with multiple video sources
- How to iterate over frame results and access tracker metadata
- How to use the display module to show annotated frames in a window
- How to extract bounding box history from tracked objects

## Prerequisites

- Voyager SDK installed and activated
- Sample media files available in `media/` (included with the SDK)

## Source

> [!TIP]
> **Running this example**
> This example is included in the SDK at `examples/application.py`. From the SDK directory with your environment activated:
>
> ```bash
> source venv/bin/activate
> python examples/application.py
> ```


```python
#!/usr/bin/env python
# Copyright Axelera AI, 2025
from axelera.app import config, create_inference_stream, display

stream = create_inference_stream(
    network="yolov5m-v7-coco-tracker",
    sources=[
        config.env.framework / "media/traffic1_1080p.mp4",
        config.env.framework / "media/traffic2_1080p.mp4",
    ],
)


def main(window, stream):
    window.options(0, title="Traffic 1")
    window.options(1, title="Traffic 2")
    VEHICLE = ('car', 'truck', 'motorcycle')
    center = lambda box: ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)
    for frame_result in stream:
        window.show(frame_result.image, frame_result.meta, frame_result.stream_id)
        for veh in frame_result.pedestrian_and_vehicle_tracker:
            print(
                f"{veh.label.name} {veh.track_id}: {center(veh.history[0])} → {center(veh.history[-1])} @ stream {frame_result.stream_id}"
            )

        if window.is_closed:
            break


with display.App(renderer=True) as app:
    wnd = app.create_window("Business logic demo", (900, 600))
    app.start_thread(main, (wnd, stream), name='InferenceThread')
    app.run()
stream.stop()
```

## Key concepts

The `create_inference_stream` function is the main entry point for building inference applications. It accepts a network name (here `yolov5m-v7-coco-tracker`) and a list of video sources. The SDK handles model compilation, pipeline setup, and frame decoding automatically.

Each `frame_result` yielded by the stream contains the annotated image, metadata, and a `stream_id` indicating which source produced the frame. The `pedestrian_and_vehicle_tracker` attribute provides tracked detections, each with a `track_id` and a `history` of bounding boxes that lets you trace an object's movement over time.

The display module follows a thread-based pattern: the `App` context manager owns the GUI event loop, while the inference logic runs in a separate thread started via `app.start_thread`. The `window.show()` call routes each frame to the correct panel based on its `stream_id`, and `window.is_closed` provides a clean shutdown signal.

---

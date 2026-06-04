# Run Inference in Python

Integrate Metis inference into your own Python application. You'll have a working detection loop in about 20 lines of code.

## Quickstart

```python
from axelera.app.stream import create_inference_stream

stream = create_inference_stream(
    network="yolov5s-v7-coco",
    sources=["media/traffic1_1080p.mp4"],
)

for frame_result in stream:
    for obj in frame_result.detections:
        print(f"{obj.label.name}: {obj.score:.2f} at {obj.bbox}")

stream.stop()
```

---

## Prerequisites

- SDK installed and virtual environment activated (see [Install the SDK](../user-guides/sdk-install.md))
- You can run `inference.py` successfully (see [First Inference](../user-guides/first-inference.md))

> [!CAUTION]
> **Before every session**
> ```bash
> source venv/bin/activate
> ```


---

## Step 1: Create an inference stream

`create_inference_stream` is the entry point for all Python application integration. It takes a model name and one or more input sources, and returns an iterable stream.

```python
from axelera.app import config
from axelera.app.stream import create_inference_stream

stream = create_inference_stream(
    network="yolov5m-v7-coco-tracker",
    sources=[
        str(config.env.framework / "media/traffic1_1080p.mp4"),
        str(config.env.framework / "media/traffic2_1080p.mp4"),
    ],
)
```

`config.env.framework` is the path to your Voyager SDK installation. You can also use plain strings for absolute paths.

**Supported source types:**

| Source | Example |
|--------|---------|
| Video file | `"media/traffic1_1080p.mp4"` |
| USB camera | `"usb:0"` |
| RTSP stream | `"rtsp://user:pwd@192.168.1.10:8554/stream"` |

Multiple sources run in parallel, each sharing the AIPU for inference. See [Video Sources](video-sources.md) for the full list.

---

## Step 2: Iterate over frames

The stream is a Python iterator. Each iteration yields a `FrameResult` for one frame from one source.

```python
for frame_result in stream:
    # frame_result.image       — the raw video frame
    # frame_result.stream_id   — which source this frame came from (0, 1, ...)
    # frame_result.<task_name> — inference results for each task in the pipeline
```

### Accessing detection results

The attribute name for results matches the **task name** in your pipeline YAML. For `yolov5m-v7-coco-tracker`, the YAML defines two tasks: `detections` and `pedestrian_and_vehicle_tracker`.

```python
for frame_result in stream:
    # Object detection results
    for obj in frame_result.detections:
        print(f"{obj.label.name}: {obj.score:.2f}")

    # Tracker results (only if pipeline includes a tracker)
    for tracked in frame_result.pedestrian_and_vehicle_tracker:
        print(f"ID {tracked.track_id}: {tracked.label.name}")
```

**Result types by task category:**

| task_category in YAML | Attribute type | Key properties |
|-----------------------|----------------|----------------|
| `ObjectDetection` | `ObjectDetectionMeta` | `bbox`, `score`, `label`, `class_id` |
| `ObjectTracking` | `TrackerMeta` | `track_id`, `history`, `bbox`, `label` |
| `Classification` | `ClassificationMeta` | `label`, `score` |
| `KeypointDetection` | `KeypointDetectionMeta` | `keypoints`, `bbox`, `score` |
| `InstanceSegmentation` | `InstanceSegmentationMeta` | `mask`, `bbox`, `label` |

See [InferenceStream API](../reference/apis/inference-stream.md) for the full property reference.

---

## Step 3: Display results

To render inference results in a window, use the `display` module alongside the stream.

```python
from axelera.app import display
from axelera.app.stream import create_inference_stream

stream = create_inference_stream(
    network="yolov5m-v7-coco-tracker",
    sources=["media/traffic1_1080p.mp4"],
)

def main(window, stream):
    for frame_result in stream:
        window.show(frame_result.image, frame_result.meta, frame_result.stream_id)

with display.App(renderer=True) as app:
    wnd = app.create_window("My App", (900, 600))
    app.start_thread(main, (wnd, stream), name="InferenceThread")
    app.run()

stream.stop()
```

`window.show()` draws all inference metadata (bounding boxes, labels, etc.) over the frame and renders it. The pipeline runs in a background thread while the display loop handles the window.

### Customizing the display

```python
# Per-stream display options
window.options(0, title="Camera 1", grayscale=0.8)
window.options(1, title="Camera 2", bbox_class_colors={"person": (0, 255, 0, 200)})
```

### Adding text overlays

```python
# Create overlay once before the loop
counter = window.text("20px, 10%", "Vehicles: 0")

for frame_result in stream:
    count = sum(1 for obj in frame_result.detections if obj.label.name in ("car", "truck"))
    counter["text"] = f"Vehicles: {count}"
    window.show(frame_result.image, frame_result.meta, frame_result.stream_id)
```

Overlays created with `window.text()` and `window.image()` are updated in place without copying the frame — lower overhead than drawing with OpenCV.

---

## Step 4: Run headless (no display)

For server deployments or batch processing without a screen:

```python
from axelera.app import display

display.set_backend("empty")

stream = create_inference_stream(
    network="yolov5s-v7-coco",
    sources=["media/traffic1_1080p.mp4"],
)

for frame_result in stream:
    for obj in frame_result.detections:
        print(f"{obj.label.name}: {obj.score:.2f}")

stream.stop()
```

---

## Step 5: Save rendered output

To save frames with visualizations to disk or a video file, use `create_surface` instead of `create_window`:

```python
with display.App(renderer=True) as app:
    surface = app.create_surface((1280, 720))

    for frame_result in stream:
        rendered = surface.render(
            frame_result.image,
            frame_result.meta,
            frame_result.stream_id,
        )
        rendered.save("output_frame.jpg")
```

---

## Advanced options

All options from `inference.py` can also be passed to `create_inference_stream`. The most commonly used:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `pipe_type` | string | `'gst'` | Pipeline backend: `'gst'`, `'torch'`, or `'torch-aipu'` |
| `log_level` | constant | `INFO` | Verbosity: `logging_utils.INFO`, `DEBUG`, `TRACE` |
| `specified_frame_rate` | int | `0` | Frame rate control — see table below |
| `rtsp_latency` | int | `500` | RTSP buffer latency in ms — see note below |

### Frame rate control

| `specified_frame_rate` value | Behavior |
|------------------------------|-----------|
| `0` | Match the source frame rate (default) |
| `N` (positive integer) | Cap throughput to N FPS — useful to reduce CPU load |
| `-1` | Downstream-leaky mode: drop frames when the application loop is too slow to keep up. Prevents queue buildup when processing is slower than the source. |

### RTSP latency

RTSP streams buffer incoming packets to handle network jitter. The `rtsp_latency` option controls the buffer size in milliseconds:

- **Too low** → frames arrive before they are buffered, causing choppy playback or dropped frames on unstable networks
- **Too high** → introduces end-to-end delay between the physical scene and your application's view

500 ms is a safe default for most networks. For low-latency applications over reliable LAN, try 50–100 ms. For streams over WAN or WiFi with significant jitter, increase to 1000–2000 ms.

> [!NOTE]
> High jitter is often a sign of network congestion or poor WiFi signal rather than a buffer size problem. If increasing `rtsp_latency` doesn't help, investigate the network path first.


### Hardware accelerator control

The GStreamer pipeline uses hardware accelerators (VA-API, OpenCL, OpenGL) where available. You can detect and override these at runtime:

```python
from axelera.app import config

caps = config.HardwareCaps()

# Detect what is available
print(caps.vaapi)   # True if VA-API is supported
print(caps.opencl)  # True if OpenCL is supported
print(caps.opengl)  # True if OpenGL is supported

# Override: disable VA-API (forces software decode)
caps.vaapi = False

stream = create_inference_stream(
    network="yolov5s-v7-coco",
    sources=["media/traffic1_1080p.mp4"],
    hardware_caps=caps,
)
```

Disabling hardware accelerators is useful when debugging accuracy differences between `gst` and `torch-aipu` pipeline modes — if VA-API resize produces slightly different pixel values than the training pipeline, disabling it isolates the effect.

### Raw tensor output

For advanced use cases, you can access the raw output tensors directly:

```python
for frame_result in stream:
    tensors = frame_result.meta['detections'].tensors
    raw = tensors[0]   # numpy array of int8 values, shape depends on model output
```

> [!WARNING]
> The raw tensor path bypasses the decoder (NMS, bbox scaling, etc.) and returns the model's quantized int8 output. Dequantise using the scale and zero_point from the tensor metadata. This path is **slower** than the high-level `frame_result.detections` API because it forces data to be copied across the device boundary on every frame. Only use it if you need the raw values.


```python
from axelera.app import logging_utils

stream = create_inference_stream(
    network="yolov5m-v7-coco-tracker",
    sources=["usb:0"],
    pipe_type="gst",
    log_level=logging_utils.DEBUG,
    specified_frame_rate=15,
)
```

### Accessing live metrics

```python
import time

last_report = time.time()

for frame_result in stream:
    if time.time() - last_report > 1.0:
        metrics = stream.get_all_metrics()
        print(f"FPS: {metrics['end_to_end_fps'].value:.1f}")
        print(f"Core temp: {metrics['core_temp'].value}°C")
        last_report = time.time()
```

Pass `tracers` when creating the stream to enable metrics collection:

```python
from axelera.app import inf_tracers

tracers = inf_tracers.create_tracers("end_to_end_fps", "core_temp", "cpu_usage")

stream = create_inference_stream(
    network="yolov5m-v7-coco-tracker",
    sources=["media/traffic1_1080p.mp4"],
    tracers=tracers,
)
```

Available tracers: `end_to_end_fps`, `core_temp`, `cpu_usage`, `stream_timing`.

---

## Example applications

The SDK ships ready-to-run examples in `examples/`:

| File | What it shows |
|------|---------------|
| `examples/application.py` | Vehicle tracker with display and terminal output |
| `examples/application_extended.py` | Adds tracers, hardware caps, custom overlays |
| `examples/application_tensor.py` | Access raw output tensors alongside high-level results |
| `examples/cross_line_count.py` | Cross-line vehicle counting using tracker history |
| `examples/remote_cross_line_monitor.py` | Cross-line counting with remote monitoring over the network |
| `examples/fruit_demo.py` | Classification demo with custom result rendering |
| `examples/render_to_video.py` | Save rendered output to a video file |
| `examples/render_to_ui.py` | Integrate into a wxPython UI |

Run any example from inside the SDK directory with the virtual environment active:

```bash
python examples/application.py
```

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: axelera` | Activate the virtual environment: `source venv/bin/activate` |
| `No Axelera device found` | Run `axdevice` to check hardware. See [Verify Setup](../getting-started/verify-setup.md) |
| Window doesn't appear | Check your display server; or run headless with `display.set_backend("empty")` |
| High CPU usage | Set `specified_frame_rate=-1` (downstream-leaky mode) to prevent queue buildup |

---

## Next steps

- [InferenceStream API](../reference/apis/inference-stream.md) — full reference for `create_inference_stream`, `FrameResult`, and metadata types
- [Video Sources](video-sources.md) — cameras, files, RTSP, multi-source
- [Measure Accuracy](measure-accuracy.md) — benchmark your model
- [Model Zoo](../reference/models/model-zoo.md) — all available models and their task categories

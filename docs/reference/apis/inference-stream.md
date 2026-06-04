---
title: "InferenceStream API"
---
# InferenceStream API

The Python API for integrating Metis inference into your own applications. See [Run Inference in Python](../../tutorials/run-inference-in-python.md) for a step-by-step guide.

---

## create_inference_stream

```python
from axelera.app.stream import create_inference_stream

stream = create_inference_stream(network, sources, **options)
```

Creates and starts an inference pipeline. Returns a `Stream` object.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `network` | string | Yes | Model name (as in Model Zoo), or absolute path to a YAML pipeline file |
| `sources` | list of strings | Yes | One or more input sources (file paths, `usb:N`, or RTSP URLs) |
| `pipe_type` | string | No | Pipeline backend: `'gst'` (default), `'torch'`, `'torch-aipu'` |
| `log_level` | constant | No | Verbosity level. Default: `logging_utils.INFO` |
| `specified_frame_rate` | int | No | Frame rate control (see below). Default: `0` |
| `rtsp_latency` | int | No | RTSP buffer in milliseconds. Default: `500` |
| `hardware_caps` | `HardwareCaps` | No | GPU acceleration settings (see below) |
| `tracers` | list | No | Metrics collectors from `inf_tracers.create_tracers()` |
| `render_config` | `RenderConfig` | No | Controls how annotations (boxes, labels) are rendered (see below) |

All `inference.py` command-line options are also accepted as keyword arguments.

### frame_rate values

| Value | Behavior |
|-------|-----------|
| `0` | Match the input frame rate (default) |
| `N > 0` | Produce exactly N frames per second |
| `-1` | Downstream-leaky: drop frames if the application loop is slow |

Use `-1` when your application has variable processing time and you want to avoid latency buildup.

### hardware_caps

```python
from axelera.app import config

stream = create_inference_stream(
    network="yolov5s-v7-coco",
    sources=["usb:0"],
    hardware_caps=config.HardwareCaps(
        vaapi=config.HardwareEnable.detect,    # VA-API video decode
        opencl=config.HardwareEnable.detect,  # OpenCL for pre/post-processing
        opengl=config.HardwareEnable.detect,  # OpenGL for display
    ),
)
```

`HardwareEnable` values: `detect` (auto), `enable` (force on), `disable` (force off).

### render_config

Controls per-task annotation rendering:

```python
from axelera.app import config

stream = create_inference_stream(
    network="yolov5s-v7-coco",
    sources=["usb:0"],
    render_config=config.RenderConfig(
        detections=config.TaskRenderConfig(
            show_annotations=False,
            show_labels=False,
        ),
    ),
)
```

`TaskRenderConfig` options:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `show_annotations` | bool | `True` | Draw bounding boxes / masks |
| `show_labels` | bool | `True` | Draw class labels and scores |

The key in `RenderConfig` (e.g., `detections=`) matches the task name in your pipeline YAML.

---

## Stream

The object returned by `create_inference_stream`.

### Iteration

```python
for frame_result in stream:
    ...
```

Iterating yields one `FrameResult` per input frame per source, in arrival order.

### Methods

| Method | Returns | Description |
|--------|---------|-------------|
| `stream.stop()` | None | Stop the pipeline and release all resources. Always call this when done. |
| `stream.get_all_metrics()` | dict | Current tracer values, keyed by tracer name. Returns empty dict if no tracers configured. |

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `stream.sources` | dict | Maps stream ID (int) to source string |

---

## FrameResult

Yielded by the stream iterator. Contains the frame image, inference metadata, and per-task results.

### Properties

| Property | Type | Description |
|----------|------|-------------|
| `frame_result.image` | `Image` | The raw input frame at original resolution |
| `frame_result.meta` | `MetaMap` | All inference results from all tasks, keyed by task name |
| `frame_result.stream_id` | int | Index of the source this frame came from (0-based) |
| `frame_result.<task_name>` | metadata object | Direct attribute access to a task's results by its YAML task name |

### Task name attributes

Task names come from your pipeline YAML. If your YAML defines:

```yaml
pipeline:
  - detections:
      model_name: yolov5s
  - tracks:
      model_name: oc_sort
```

Then `frame_result.detections` and `frame_result.tracks` are available as attributes.

---

## Image

Accessed via `frame_result.image`.

| Method / Property | Returns | Description |
|-------------------|---------|-------------|
| `image.asarray(format=None)` | `numpy.ndarray` | Frame as NumPy array. Format: `'RGB'`, `'BGR'`, `'GRAY'`, `'BGRA'`. Defaults to input format. |
| `image.aspil()` | `PIL.Image` | Frame as a PIL Image object |
| `image.color_format` | string | Color format of the raw image (`'RGB'`, `'BGR'`, etc.) |

---

## Metadata types

The type of `frame_result.<task_name>` depends on the `task_category` in the pipeline YAML.

### ObjectDetectionMeta

Returned when `task_category: ObjectDetection`.

```python
for obj in frame_result.detections:   # iterate over detected objects
    obj.bbox      # (x1, y1, x2, y2) in pixels
    obj.score     # float, confidence 0.0–1.0
    obj.class_id  # int, class index
    obj.label     # label enum value (if classlabels_file was set)
    obj.label.name  # string label name
```

### TrackerMeta

Returned when `task_category: ObjectTracking`.

```python
for obj in frame_result.tracks:       # iterate over tracked objects
    obj.track_id      # int, unique persistent ID across frames
    obj.history       # list of past bboxes: history[0]=first seen, history[-1]=current
    obj.bbox          # current (x1, y1, x2, y2)
    obj.score         # float, confidence
    obj.label         # label enum value
    obj.is_a(labels)  # bool — True if obj.label.name is in the given list/tuple
```

`history` length is controlled by the `history_length` option in the tracker's YAML config.

### ClassificationMeta

Returned when `task_category: Classification`.

```python
for obj in frame_result.classification:
    obj.label     # top predicted label
    obj.score     # confidence of top prediction
```

### KeypointDetectionMeta

Returned when `task_category: KeypointDetection`.

```python
for obj in frame_result.poses:
    obj.keypoints   # list of (x, y) or (x, y, visibility) tuples
    obj.bbox        # bounding box (if available)
    obj.score       # confidence
```

### InstanceSegmentationMeta

Returned when `task_category: InstanceSegmentation`.

```python
for obj in frame_result.segments:
    obj.mask    # pixel-level segmentation mask
    obj.bbox    # bounding box
    obj.label   # class label
    obj.score   # confidence
```

---

## Display API

### display.App

Context manager for the display system.

```python
from axelera.app import display

with display.App(renderer=True) as app:
    wnd = app.create_window("Title", (width, height))
    app.start_thread(my_function, (wnd, stream), name="InferenceThread")
    app.run()
```

| Parameter | Description |
|-----------|-------------|
| `renderer=True` | Enable frame rendering (bounding boxes, overlays) |
| `renderer=False` | Disable rendering — use when only accessing raw results |

### Window

Created by `app.create_window(title, size)`.

| Method | Description |
|--------|-------------|
| `window.show(image, meta, stream_id)` | Render inference results onto the frame and display |
| `window.options(stream_id, **kwargs)` | Configure display options for a source stream |
| `window.text(position, text, **kwargs)` | Create a text overlay layer; returns a handle |
| `window.image(position, image, **kwargs)` | Create an image overlay layer; returns a handle |

**window.options kwargs:**

| Option | Description |
|--------|-------------|
| `title` | Label shown above the stream |
| `grayscale` | Float 0–1: how gray to render the background frame (results stay colored) |
| `bbox_class_colors` | Dict mapping label name → `(R, G, B, A)` color tuple |

**window.text / window.image position format:**

Position is a CSS-style string: `"20px, 10%"` means 20 pixels from the left, 10% from the top. Use `stream_id=N` to position relative to a specific stream; use `stream_id=-1` (default) to position relative to the whole window.

### Headless mode

```python
display.set_backend("empty")
```

Call this before creating an `App` to run without a display server. Required for servers and CI environments.

### Surface (saving rendered frames)

Created by `app.create_surface(size)`. Use instead of `create_window` when you want rendered frames for saving or custom UIs.

| Method | Returns | Description |
|--------|---------|-------------|
| `surface.render(image, meta, stream_id)` | `Image` | Render immediately and return the result |
| `surface.push(image, meta, stream_id)` | None | Add to render queue (non-blocking) |
| `surface.pop_latest()` | `Image` or None | Get the latest rendered frame (None if not yet ready) |
| `surface.latest` | `Image` | Most recently rendered frame |

`pop_latest()` returns each new frame exactly once. `surface.latest` may return the same frame on multiple calls.

---

## Tracers

```python
from axelera.app import inf_tracers

tracers = inf_tracers.create_tracers("end_to_end_fps", "core_temp", "cpu_usage")

stream = create_inference_stream(..., tracers=tracers)

metrics = stream.get_all_metrics()
print(metrics["end_to_end_fps"].value)
```

**Available tracers:**

| Name | What it measures |
|------|-----------------|
| `end_to_end_fps` | Frames per second through the complete pipeline |
| `core_temp` | Metis AIPU core temperature in °C |
| `cpu_usage` | Host CPU utilization percentage |
| `stream_timing` | Per-frame latency and jitter |

---

## See also

- [Run Inference in Python](../../tutorials/run-inference-in-python.md) — tutorial with worked examples
- [Pipelines — How Inference Works](../pipeline/pipeline-basics.md) — pipeline concepts
- [Model Zoo](../models/model-zoo.md) — available models and their task categories
- [Video Sources](../../tutorials/video-sources.md) — input source reference

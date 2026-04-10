![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Application integration tutorial

## Contents

- [Application integration tutorial](#application-integration-tutorial)
  - [Contents](#contents)
  - [Prerequisites](#prerequisites)
  - [Level](#level)
  - [Example computer vision pipeline](#example-computer-vision-pipeline)
    - [Understanding how YAML maps to Python](#understanding-how-yaml-maps-to-python)
  - [Simple application integration example](#simple-application-integration-example)
    - [Cross-line counting example](#cross-line-counting-example)
    - [Broadcasting metadata to remote clients](#broadcasting-metadata-to-remote-clients)
  - [Advanced features](#advanced-features)
    - [Tracing](#tracing)
    - [Hardware caps](#hardware-caps)
    - [Frame rate control](#frame-rate-control)
    - [RTSP latency](#rtsp-latency)
  - [Customizing visualization](#customizing-visualization)
    - [Custom Visual Overlays](#custom-visual-overlays)
  - [Accessing Rendered Images](#accessing-rendered-images)
  - [Using Raw Tensor Output in Your Application](#using-raw-tensor-output-in-your-application)
    - [Pipeline configuration for raw tensor output](#pipeline-configuration-for-raw-tensor-output)
      - [Performance optimization (advanced)](#performance-optimization-advanced)
    - [Performance note](#performance-note)
    - [Example: Extracting the tensor in Python](#example-extracting-the-tensor-in-python)
  - [List of example applications](#list-of-example-applications)
  - [Next Steps](#next-steps)
  - [Related Documentation](#related-documentation)
  - [Further support](#further-support)

## Prerequisites
- Complete [Quick Start Guide](quick_start_guide.md) - verify you can run inference with `inference.py`
- SDK installed and virtual environment activated (`source ~/voyagersdk/bin/activate`)
- Basic understanding of Python or C++ programming
- Familiarity with object detection concepts (bounding boxes, confidence scores)

## Level
**Intermediate** - Requires understanding of pipeline execution and writing application code


Voyager application integration APIs cleanly separate the task of developing computer vision
pipelines from the task of developing inferencing applications based on these pipelines.

## Example computer vision pipeline

A Voyager YAML file describes an inferencing pipeline including image pre-processing,
a deep learning model and post-processing. This tutorial is based on a
[YOLOv5m-based pipeline with tracker](/ax_models/reference/cascade/with_tracker/yolov5m-v7-coco-tracker.yaml),
but it can be applied to any YAML pipeline that outputs tracked bounding boxes.
The code fragment below shows the complete pipeline definition.

```yaml
name: yolov5m-v7-coco-tracker

pipeline:
  - detections:                          # Task name (first task)
      model_name: yolov5m
      preprocess:
        - letterbox:
            width: 640
            height: 640
        - torch-totensor:
      postprocess:
        - decodeyolo:
            box_format: xywh
            normalized_coord: False
            label_filter: ['car', 'truck', 'motorcycle', 'bus', 'person', 'bicycle', 'cat', 'dog']
            conf_threshold: 0.3
            nms_iou_threshold: 0.5

  - pedestrian_and_vehicle_tracker:      # Task name (second task)
      model_name: oc_sort
      cv_process:
        - tracker:
            algorithm: oc-sort
            bbox_task_name: detections
            history_length: 30
            algo_params:
              max_age: 50
              iou_threshold: 0.3

models:
  yolov5m:
    task_category: ObjectDetection       # Determines Python Meta type
  oc_sort:
    task_category: ObjectTracking        # Determines Python Meta type
```

This pipeline includes a YOLOv5m object detection model followed by an OC-SORT tracker. The pipeline has
**two tasks**, each with a user-defined name: `detections` for the object detection task and
`pedestrian_and_vehicle_tracker` for the tracking task.

The input images are first pre-processed by letterboxing and converted to tensor format. The YOLOv5m model
outputs are decoded into bounding box metadata. The tracker then processes these detections to maintain
consistent object IDs across frames. Fields like `conf_threshold` and `nms_iou_threshold` are tunable
parameters with default values that can be modified at runtime.

The Voyager toolchain generates optimized implementations of YAML pipelines for a target system
comprising a host processor and one or more Metis devices. You can use the application integration APIs
to configure and run pipelines from either Python or C++ applications and by writing only a few lines of code.

YAML pipelines do not include input operators. Instead, you specify your required
[video sources](/docs/tutorials/video_sources.md) at runtime when instantiating a pipeline
in your application.

### Understanding how YAML maps to Python

Understanding the relationship between your YAML pipeline and the Python API is essential for working with
inference results in your application.

**Key concepts:**

1. **Task names are user-defined**: You choose any name for your tasks (e.g., `detections`, `master_detections`, `my_custom_task`)
2. **task_category determines metadata type**: The `task_category` field in the models section determines which Python metadata class is returned
3. **Access results via task name**: Use `frame_result.{your_task_name}` to access results for each task

**The mapping pattern:**

| YAML Element | Location | Purpose | Example |
|--------------|----------|---------|---------|
| Task name | `pipeline:` list | User-defined identifier for accessing results | `detections`, `pedestrian_and_vehicle_tracker` |
| Model name | Task's `model_name:` field | References entry in `models:` section | `yolov5m`, `oc_sort` |
| task_category | `models.{model_name}.task_category` | Determines Python metadata class type | `ObjectDetection`, `ObjectTracking` |

**How it works in this example:**

```python
# Task name: "detections" with task_category: ObjectDetection
frame_result.detections              # Access via task name from YAML
# Returns: ObjectDetectionMeta       # Type determined by task_category
# Contains: DetectedObject items     # Individual detections with bbox, score, label

# Task name: "pedestrian_and_vehicle_tracker" with task_category: ObjectTracking
frame_result.pedestrian_and_vehicle_tracker  # Access via task name from YAML
# Returns: TrackerMeta                        # Type determined by task_category
# Contains: TrackedObject items               # Tracked objects with track_id, history
```

**Common task_category mappings:**

| task_category | Meta Class | Object Class | Typical Use |
|---------------|------------|--------------|-------------|
| `ObjectDetection` | `ObjectDetectionMeta` | `DetectedObject` | Bounding box detection (YOLO, SSD, etc.) |
| `ObjectTracking` | `TrackerMeta` | `TrackedObject` | Multi-frame object tracking |
| `KeypointDetection` | `KeypointDetectionMeta` | `KeypointObject` | Pose estimation, facial landmarks |
| `InstanceSegmentation` | `InstanceSegmentationMeta` | Segment objects | Pixel-level instance masks |
| `Classification` | `ClassificationMeta` | `ClassifiedObject` | Image or ROI classification |

> [!TIP]
> For a complete list of supported task categories and their uses, see the
> [Model definition reference](/docs/tutorials/custom_weights.md#model-definition).

**Example with different task names:**

The fruit-demo pipeline demonstrates how flexible task naming is:

```yaml
pipeline:
  - master_detections:               # User chose this name
      model_name: yolov8lpose
  - segmentations:                   # User chose this name
      model_name: yolov8sseg
  - object_detections:               # User chose this name
      model_name: yolov8s

models:
  yolov8lpose:
    task_category: KeypointDetection
  yolov8sseg:
    task_category: InstanceSegmentation
  yolov8s:
    task_category: ObjectDetection
```

Access in Python:
```python
frame_result.master_detections     # KeypointDetectionMeta
frame_result.segmentations         # InstanceSegmentationMeta
frame_result.object_detections     # ObjectDetectionMeta
```


## Simple application integration example

The file [application.py](/examples/application.py) shows how to configure a pipeline
with multiple video input sources and then run it, obtaining a sequence of images and inference metadata.
The application renders the results visually in a window and outputs a basic analysis of
all tracked vehicles to the terminal.

The first few lines of code import the Voyager application integration libraries needed by this example.

```python
from axelera.app import config, display
from axelera.app.stream import create_inference_stream
```

The application defines a *stream* comprising the YOLOv5m-based pipeline and two input video files.

```python
stream = create_inference_stream(
    network="yolov5m-v7-coco-tracker",
    sources=[
        str(config.env.framework / "media/traffic1_1080p.mp4"),
        str(config.env.framework / "media/traffic2_1080p.mp4"),
    ],
)
```

The first argument to `create_inference_stream` is the name of the pipeline as it appears in the YAML file (field: `name`).
If your YAML file is located outside of the Voyager SDK repository, you can provide an absolute path to it instead.

The second argument is an array of input sources, in this case two video files in the `media` directory of the
Voyager SDK repository. (The variable `config.env.framework` is the path to the Voyager repository for the activated environment.)
Many different input sources are supported.

| Source | Description | Example |
| :----- | :---------- | :------ |
| `/path/to/file` | [Path to an image or video file](/docs/tutorials/video_sources.md#local-files) | `str(config.env.framework / "media/traffic1_1080p.mp4")` |
| `usb:<device_id>` | [USB camera](/docs/tutorials/video_sources.md#usb-cameras) | `usb:0` |
| `rtsp://<ip_address>:<port>/<stream_name>` | [RTSP camera](/docs/tutorials/video_sources.md#rtsp-cameras) | `rtsp://user:pwd@127.0.0.1:8554/1` |

You can freely mix and match different video sources and input formats. Using the same type of source, color format and video resolution often results in the highest possible performance.

> [!TIP]
> You can download the sample videos used in this tutorial by running the command `./install.sh --media` from the root of the Voyager SDK repository.

The application creates a window to display rendered inference results. For performance reasons,
it runs the pipeline in a separate thread from the main application.
A final call to `stream.stop()` is needed to terminate the pipeline and
release all of its allocated resources.

```python
with display.App(renderer=True) as app:
    wnd = app.create_window("Business logic demo", (900, 600))
    app.start_thread(main, (wnd, stream), name='InferenceThread')
    app.run()
stream.stop()
```

The main application arranges the display window with two tiles, one for each input video.
It defines a `VEHICLE` as a list of class labels, defines the function `center`
for analyzing object movement with center coordinates, and then starts iterating over
the inference stream using Voyager libraries to analyze and visualise the results.

```python
def main(window, stream):
    window.options(0, title="Traffic 1")
    window.options(1, title="Traffic 2")

    VEHICLE = ('car', 'truck', 'motorcycle')
    center = lambda box: ((box[0] + box[2]) // 2, (box[1] + box[3]) // 2)

    for frame_result in stream:
        window.show(frame_result.image, frame_result.meta, frame_result.stream_id)

        # Access tracker results (task name: pedestrian_and_vehicle_tracker)
        # Returns TrackerMeta with TrackedObject items
        for tracked_obj in frame_result.pedestrian_and_vehicle_tracker:
            if tracked_obj.is_a(VEHICLE):
                print(
                    f"{tracked_obj.label.name} {tracked_obj.track_id}: "
                    f"{center(tracked_obj.history[0])} → {center(tracked_obj.history[-1])} "
                    f"@ stream {frame_result.stream_id}"
                )
```

The `stream` object is an iterator that yields a `frame_result` for each input frame. This object contains:

- `frame_result.image`: The input source image at its original resolution
- `frame_result.meta`: All inference metadata from all tasks
- `frame_result.detections`: Direct access to the "detections" task results (ObjectDetectionMeta)
- `frame_result.pedestrian_and_vehicle_tracker`: Direct access to the tracker task results (TrackerMeta)
- `frame_result.stream_id`: The input stream identifier

The task names (`detections`, `pedestrian_and_vehicle_tracker`) come directly from your YAML pipeline definition.
Each task name becomes an attribute on `frame_result`, returning the appropriate metadata type based on its `task_category`.

The method `window.show()` overlays the frame metadata on the original image and renders the result.

The helper function `is_a()` makes it easy to filter objects by category. In this example,
`tracked_obj.is_a(VEHICLE)` determines if a tracked object belongs to any of the defined vehicle categories.

**Understanding object properties:**

Different object types provide different properties based on their task_category:

**TrackedObject** (from ObjectTracking tasks):
- All detection properties: `bbox`, `score`, `class_id`, `label`
- `track_id`: Unique identifier maintained across frames
- `history`: List of previous bounding boxes showing movement trajectory
  - `history[0]`: First observed position
  - `history[-1]`: Current position

**DetectedObject** (from ObjectDetection tasks):
- `bbox`: Bounding box coordinates
- `score`: Confidence score
- `class_id`: Numeric class identifier
- `label`: Label enum value (if labels provided)

**KeypointObject** (from KeypointDetection tasks):
- `keypoints`: Array of (x, y) or (x, y, visibility) coordinates
- `box`, `score`: Bounding box and confidence (if available)

Sample output from running this application is shown below.

```
car 2: (398, 312) → (415, 312) @ stream 1
truck 70028: (720, 311) → (754, 310) @ stream 1
car 70027: (1647, 1039) → (1644, 1040) @ stream 0
```

> [!TIP]
> You can also implement your own methods to manipulate and display images. To obtain
> a PIL Image object, call `frame_result.image.aspil()`, and call `frame_result.image.color_format`
> to determine its color format.
> To obtain the image as a NumPy array use `frame_result.image.asarray()` optionally specifying the required color
> format as an argument (`RGB`, `BGR`, `GRAY` or `BGRA`). If unspecified, the color format is determined
> based on the input video.

### Cross-line counting example

The earlier snippet prints tracker IDs and labels. [`examples/cross_line_count.py`](/examples/cross_line_count.py)
builds on that by showing how to use the tracker history (`veh.history`) to implement real business logic — here, counting vehicles as they cross a virtual line.

- Text overlays for displaying counts are created using `window.text()` (see [Custom Visual Overlays](#custom-visual-overlays)) before the loop, then updated each frame.
- The first iteration computes the line geometry based on the input frame size and caches the slope and intercept.
- For each tracked object, it checks whether the center of the bounding box moved from one side of the line to the other and keeps running totals as well as a short-term history window to show recently crossed IDs.
- The example draws the counting line on the image using OpenCV and feeds the modified frame back to the display with `types.Image.fromarray(...)`.

This example demonstrates a concise pattern for processing trajectory history (`meta.pedestrian_and_vehicle_tracker`) within the main iteration loop.

### Broadcasting metadata to remote clients

[`examples/remote_cross_line_monitor.py`](/examples/remote_cross_line_monitor.py) extends the previous example by publishing the per-frame counts and tracked objects over a TCP socket. This pattern is useful when you need to ship metadata to a remote dashboard or control system while keeping rendering local.

- Like the previous example, it uses `window.text()` to display live counts on screen.
- The script spins up a lightweight socket server (`RemoteBroadcaster`) that accepts multiple clients and broadcasts a JSON document per frame.
- The broadcast payload includes cumulative counts, the list of IDs that crossed in the last N frames, and the current tracker objects (track ID, class ID, bounding box).
- Connect with any TCP client, e.g. `nc localhost 8765`, to watch the stream in real time or pipe it into your own monitoring service.

Because the broadcaster runs in a background thread and uses only standard library modules, it drops cleanly into more complex applications without extra dependencies.

When running headless (e.g. on a server without a display), switch the display backend to "empty" before creating the `App`:

```python
from axelera.app import display

display.set_backend("empty")

with display.App(renderer=False) as app:
    ...
```

Sample payload emitted by the broadcaster:

```json
{"frame": 1336, "crossed_up": 7, "crossed_down": 0, "recent_up": [], "recent_down": [], "objects": [{"track_id": 40, "class_id": 2, "bbox": [518, 370, 590, 478]}, {"track_id": 34, "class_id": 2, "bbox": [355, 208, 365, 218]}, {"track_id": 39, "class_id": 2, "bbox": [495, 235, 513, 247]}, {"track_id": 41, "class_id": 2, "bbox": [475, 236, 490, 247]}, {"track_id": 30, "class_id": 2, "bbox": [596, 269, 637, 294]}, {"track_id": 43, "class_id": 2, "bbox": [451, 227, 464, 236]}, {"track_id": 20, "class_id": 7, "bbox": [514, 372, 592, 478]}]}
```

Typical use cases include feeding live dashboards, storing counts in a time-series database, or triggering automation workflows (e.g. gate control, alerts) directly from the streamed metadata.

## Advanced features

The file [application_extended.py](/examples/application_extended.py) uses a number of
advanced integration features.

The first few lines of code import the Voyager application integration libraries needed by this example.

```python
import time

from axelera.app import config, display, inf_tracers, logging_utils
from axelera.app.stream import create_inference_stream

framework = config.env.framework
```

The application creates a YOLOv5m inferencing pipeline and with a number of advanced
options configured.

```python
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
    render_config=config.RenderConfig(
        detections=config.TaskRenderConfig(
            show_annotations=False,
            show_labels=False,
        ),
    ),
)
```

The `pipe_type` argument to `create_inference_stream` specifies the type of pipeline
to build and run. The default `gst` option represents a pipeline running end-to-end
across the host processor and Metis device. Other supported values include 
`torch` and `torch-aipu` (see [`--pipe` option](/docs/reference/deploy.md) for further
details).

The `log_level` argument to `create_inference_stream` controls the verbosity of output
to the terminal. The default verbosity level is `INFO` while `DEBUG` and `TRACE` levels
provide additional information useful for debugging.

> [!TIP]
> All options supported by [`inference.py`](/docs/reference/inference.md) can also be passed to
> `create_inference_stream`, enabling you to easily switch between evaluation and
> development environments.

### Tracing

Tracers provide real-time metrics that help you better understand
device resource utilization, performance and thermal characteristics.

The function `create_tracers` shown in the example below takes a list of metrics as its arguments and returns a list of
tracers to provide the `tracers` argument in the function `create_inference_stream`.  The available tracers are:

* `end_to_end_fps` - collects end-to-end fps metric
* `core_temp` - collects the temperature of the metis core
* `cpu_usage` - collects information about the host CPU usage
* `stream_timing`  - collects metrics about stream latency and jitter

The main application periodically queries these tracers using the method `stream.get_all_metrics()` and prints
the returned metrics to the terminal.

```python
def main(window, stream):
    # ... other code
    for frame_result in stream:
        # Process frames...
        
        # Get current metrics
        metrics = stream.get_all_metrics()
        core_temp = metrics['core_temp']
        end_to_end_fps = metrics['end_to_end_fps']
        cpu_usage = metrics['cpu_usage']
        
        # Report metrics periodically
        if (now := time.time()) - last_temp_report > 1:
            last_temp_report = now
            metrics = [
                f"Core temp: {core_temp.value}°C",
                f"End-to-end FPS: {end_to_end_fps.value:.1f}",
                f"CPU usage: {cpu_usage.value:.1f}%",
            ]
            print(' | '.join(metrics))
```

### Hardware caps

The `hardware_caps` argument to `create_inference_stream` is used to enable or disable use of the host runtime acceleration
libraries, which can accelerate image pre-processing and post-processing operations on GPU hardware embedded on the host
processor. Set to `detect` for automatic detection, `enable` to force usage and `disable` to prevent usage.

### Frame rate control

The `specified_frame_rate` argument to `create_inference_stream` lets you fine-tune pipeline frame rate control behavior.

| Value | Description |
| :---- | :---------- |
| Positive integer | The pipeline produces precisely the specified frames per second, dropping or duplicating frames as needed
| `0` | The pipeline produces frames at the a rate matching the input frame rate
| `-1` | The pipeline operates in downstream-leaky mode, dropping frames if the application is unable to consume them before the next frame is ready

In general, if the application can process frames faster than the input frame rate then a value of `0` can be specified. If
not and your application loop runs with predictable time, a positive value can be specified. If your application requires
periods of intense processing, downstream-leaky mode can help prevent your pipeline queues from filling up during these periods,
which would otherwise lead to a buildup of latency and potential instabilities.

### RTSP latency

The `rtsp_latency` argument to `create_inference_stream` lets you balance system latency
and stability when working with IP network cameras.
Setting a low latency enables the application to respond quicker to detections, but
may also lead to problems such as:

- inability to smooth out network jitter resulting in packet loss
- choppy and stuttering playback during transmission delays
- poor synchronization of audio and video streams
- loss of connection due to insufficient margin for packet retransmission

In general, specifying a higher latency value such as 2000ms can help avoid these
issues under poor network conditions, while the default latency value of 500ms provides
a good tradeoff between latency and reliability in many typical network conditions.

## Customizing visualization

The file [fruit_demo.py](/examples/demos/fruit_demo.py) shows how to configure the visualiser in a number
of different ways.

```python
def main(window, stream):
    for n, source in stream.sources.items():
        window.options(
            n,
            title=f"Video {n} {source}",
            title_size=24,
            grayscale=0.9,
            bbox_class_colors={
                'banana': (255, 255, 0, 125),
                'apple': (255, 0, 0, 125),
                'orange': (255, 127, 0, 125),
            },
        )
    for frame_result in stream:
        window.show(frame_result.image, frame_result.meta, frame_result.stream_id)
```

The method `window.options()` supports the following configuration options:

- `title`: Adds a descriptive title to the video stream
- `grayscale`: Sets a grayness level for the original image. Inference metadata such as bounding boxes and segments continue to be rendered in color
- `bbox_class_colors`: Specifies the color to be used to render each specified class label
- `bbox_label_format`: See the [`Options` class definition](/axelera/app/display.py#L75) for further details

> [!TIP]
> Consider reducing rendering overheads by calling `window.show()` only when needed, or by setting `renderer=False` to
> disable rendering if not required by your application.

### Custom Visual Overlays

The file [application_extended.py](/examples/application_extended.py) demonstrates various advanced features including tracers, hardware acceleration settings, render configuration, and custom visual overlays. Real-world applications often need to display dynamic information such as vehicle counts, performance metrics (FPS, latency, temperature), application state, alerts, or branding elements (logos, timestamps).

**When to use each approach:**

- **Use OpenCV** (`cv2.line`, `cv2.circle`, `cv2.rectangle`, etc.) when drawing **geometric annotations** that are part of the scene analysis, such as counting lines, region boundaries, detection zones, or measurement overlays. These are integral to your computer vision logic and should be drawn directly on the frame.

- **Use the layering API** (`window.text`, `window.image`) when adding **information overlays** like text labels, counters, status indicators, logos, or watermarks. These are display elements separate from the actual image content.

For adding text information overlays, you can use either OpenCV or Voyager's layering API:

**Using OpenCV:**
```python
import cv2
from axelera import types

image = frame_result.image.asarray().copy()
cv2.putText(image, f"Count: {count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)
window.show(types.Image.fromarray(image, frame_result.image.color_format), ...)
```

**Using the layering API (recommended):**
```python
# Create overlay once before the loop
counter = window.text('20px, 10%', "Vehicles: 00")

for frame_result in stream:
    # Update text without modifying the frame
    counter["text"] = f"Vehicles: {count:02}"
    window.show(frame_result.image, frame_result.meta, frame_result.stream_id)
```

The layering API offers several advantages: no image copying overhead, better performance, cleaner code separation, and support for transparency and animations.

The method `window.text` creates a text overlay on the window, and returns a handle to the text overlay which can be used for future modification.

- The text overlay will be located at the given css-style string position `'20px, 10%'`. This means the anchor point of the text overlay (default top left) will be located 20 screen pixels from the left of the window, and 10% down from the top. This illustrates the two methods of providing layer locations: in absolute pixel terms, or relative to the window/stream size.
  - Note that the counter example is constructed with `stream_id=0`. When a `stream_id` is provided, the layer will be rendered to that specific stream. As such, the positions will be relative to the indicated stream's canvas. If `stream_id=-1` (default), the layer belongs to the window and positions are relative to the entire window.
- The text overlay will contain the text "Vehicles: 00" after initialization, to serve as a counter of the number of vehicles currently in the frame.
- Colors use RGBA format (red, green, blue, alpha), where the alpha channel controls transparency (0=fully transparent, 255=fully opaque).

Later, when processing the stream, we update the text in the text overlay to whatever the current vehicle count is for this frame. This is done by setting the field by name like a dictionary, `counter["text"]` in this case. This will update the text in-place on the next rendered frame.

> [!TIP]
> You can also update fields by kwarg with the `set` method on the handle, including multiple fields at once.
> Position coordinates can be provided in tuple form, like `('20px', '10%')`.
> Other layers, such as images, may be displayed in a similar fashion using `window.image()`.
> Additional customization options include `bgcolor` (background color with transparency), `font_size`, `anchor_x`/`anchor_y` (for custom anchor points), and fade animations (`fadein_by`, `fadein_for`, `fadeout_from`, `fadeout_for`).
> There are many other customizations and methods provided by the handle. See the [`Window` and `LayerHandle` class definitions](/axelera/app/display.py) for further details.

## Accessing Rendered Images

If you need to access the rendered frames (with visualizations) for custom processing, saving, or integration with
other UI frameworks, use `create_surface()` instead of `create_window()`:

```python
with display.App(renderer=True) as app:
    surface = app.create_surface((900, 600))

    for frame_result in stream:
        # Get rendered image with all visualizations applied
        rendered_image = surface.render(
            frame_result.image,
            frame_result.meta,
            frame_result.stream_id
        )
        # Now you can save it, process it, display in custom UI, etc.
        rendered_image.save("output.jpg")
```

Note that if you do not need the specific returned image immediately, such as when streaming to a custom UI, you should use `surface.pop_latest(...)`

```python
with display.App(renderer=True) as app:
    surface = app.create_surface((900, 600))

    for frame_result in stream:
        # Add this image + metadata to the rendering queue
        surface.push(
            frame_result.image,
            frame_result.meta,
            frame_result.stream_id
        )
        # The latest NEW frame will be available here
        new = surface.pop_latest()
        # The latest rendered frame will be available here
        latest = surface.latest
```

> [!TIP]
> `surface.latest` and `surface.pop_latest()` are similar, except `pop_latest()` will only return the latest frame once.
> This is useful when streaming output, as each time you call `pop_latest()`, if the return is not None, it is
> guaranteed to be a new frame. See [render_to_ui.py](/examples/render_to_ui.py) for example usage.

When to use Surface vs Window:
- Use `create_window()` when you want Voyager to display results in a window
- Use `create_surface()` when you want to access rendered images for:
  - Saving to disk or streaming
  - Integration with custom UI frameworks (Qt, wxPython, web apps)
  - Custom processing pipelines

See [render_to_video.py](/examples/render_to_video.py) for a complete example of saving rendered frames to video, and [render_to_ui.py](/examples/render_to_ui.py) for wxPython UI integration.

## Using Raw Tensor Output in Your Application

In some cases, you may want your pipeline to return the raw output tensor from your model, rather than ready-to-use detection objects. This is useful if:
- You are working with a custom model or a model not supported by the standard postprocessing operators.
- You want full control over postprocessing (e.g., custom decoding, NMS, or analytics).
- You want to experiment with new postprocessing algorithms or integrate with your own pipeline.

### Pipeline configuration for raw tensor output

To receive the raw tensor output that matches your original ONNX or PyTorch model, configure your YAML pipeline like this:

```yaml
inference:
  handle_all: True  # Default: produces output matching your original model
postprocess:
  - get-raw-tensor:
```

This gives you the same tensor output as your original ONNX or PyTorch model - fully dequantized, with correct dimensions and layout.

#### Performance optimization (advanced)

For improved performance, especially with smaller models, you can handle processing steps yourself:

**Option 1 - Handle nothing (maximum performance potential):**
```yaml
inference:
  handle_all: False  # You handle all processing steps yourself
postprocess:
  - get-raw-tensor:
```

**Option 2 - Fine-grained control:**
```yaml
inference:
  # Don't set handle_all, configure individual steps
  handle_dequantization_and_depadding: True
  handle_transpose: False
  handle_postamble: False
postprocess:
  - get-raw-tensor:
```

With `handle_all: False`, you take responsibility for all processing steps yourself, but this can enable additional optimizations in the pipeline.

This will produce an `.axnet` where the output meta contains a tensor, not decoded detections.

### Performance note

The raw tensor output path is intended for flexibility, experimentation, and integration - not for maximum speed. If you want maximum performance in production applications, consider building a standard decoder as done for models in the Axelera model zoo, which allows the pipeline to use fully optimized postprocessing.

### Example: Extracting the tensor in Python

When you run inference, you can extract the tensor from the frame result like this:

```python
for frame_result in stream:
    tensor_wrapper = frame_result.meta['detections']
    tensor = tensor_wrapper.tensors[0]  # This is a numpy array
    # Now you can implement your own postprocessing, e.g.:
    # detections = postprocess_yolov8(tensor, tensor.shape, ...)
```

> **Note:**
> - The output is a tensor (e.g., shape `[1, 84, 8400]` for YOLOv8), not a list of detection objects.
> - You are responsible for all postprocessing, including decoding, NMS, and mapping to image coordinates.
> - **For InferenceStream, we highly recommend using the standard pipeline path for production and real applications.** The raw tensor output path is intended primarily for educational purposes, experimentation, or quick prototyping, and it significantly reduces performance compared to the standard pipeline.
> - You can use the raw tensor output path for quick prototyping or exploration. After that, you can either:
>   - Move to a production solution with AxInferenceNet ([AxInferenceNet C++ Integration Tutorial](/docs/tutorials/axinferencenet.md)), or
>   - Finish your implementation and use InferenceStream properly with a standard pipeline ([Tutorials](/ax_models/tutorials/general/tutorials.md)).


See the [axinferencenet_tensor.cpp](/examples/axinferencenet/axinferencenet_tensor.cpp) and [application_tensor.py](/examples/application_tensor.py) examples for more details on working with raw tensor output.


## List of example applications

| Example | Description |
| :------ | :---------- |
| [`/examples/application.py`](/examples/application.py) | Simple integration of vehicle tracker into an application including visualization and basic analytics |
| [`/examples/application_extended.py`](/examples/application_extended.py) | Adds advanced customization and monitoring to the simple vehicle tracker example |
| [`/examples/demos/fruit_demo.py`](/examples/demos/fruit_demo.py) | Renders segmentation of fruits held by people in color against a grayscale background |
| [`/examples/application_tensor.py`](/examples/application_tensor.py) | Demonstrates how to extract and postprocess the raw tensor output from a YOLOv8 model |
| [`/examples/cross_line_count.py`](/examples/cross_line_count.py) | Counts vehicles crossing a virtual line using tracker metadata and overlays live statistics on the rendered frame |
| [`/examples/remote_cross_line_monitor.py`](/examples/remote_cross_line_monitor.py) | Extends the cross-line counter with a TCP broadcast server that streams JSON updates to remote clients |
| [`/examples/render_to_ui.py`](/examples/render_to_ui.py) | Demonstrates how to stream rendered images in a custom UI framework, wxPython in this case |
| [`/examples/render_to_video.py`](/examples/render_to_video.py) | Demonstrates how to save rendered images to a video file |


## Next Steps
- **Test with your own video sources**: [Video Sources Tutorial](video_sources.md)
- **Measure performance**: [Benchmarking Tutorial](benchmarking.md)
- **Deploy custom models**: [Custom Weights Tutorial](custom_weights.md)
- **Chain multiple models**: [Cascaded Models Tutorial](cascaded_model.md)
- **Switch to C++ for production**: [AxInferenceNet Tutorial](axinferencenet.md)

## Related Documentation
**Tutorials:**
- [Video Sources](video_sources.md) - Configure cameras, video files, and RTSP streams
- [AxInferenceNet Tutorial](axinferencenet.md) - C++ integration approach
- [Benchmarking](benchmarking.md) - Measure performance metrics
- [Cascaded Models](cascaded_model.md) - Chain multiple models together

**References:**
- [AxRuntime API](../reference/axruntime.md) - Python pipeline management
- [AxRunModel API](../reference/axrunmodel.md) - Model metadata inspection
- [AxDevice API](../reference/axdevice.md) - Device enumeration and capabilities
- [Adapters Reference](../reference/adapters.md) - Input source configuration details
- [AxInferenceNet C++ API](../reference/axinferencenet.md) - C++ class reference

**Examples:**
- [application.py](../../examples/application.py) - Basic detection loop with visualization
- [application_extended.py](../../examples/application_extended.py) - Advanced features (hardware caps, frame rate control)
- [application_tensor.py](../../examples/application_tensor.py) - Direct tensor access for custom postprocessing
- [classification_example.py](../../examples/classification_example.py) - Classification workflow
- [multiple_pipelines.py](../../examples/multiple_pipelines.py) - Concurrent pipeline execution
- [cross_line_count.py](../../examples/cross_line_count.py) - Practical analytics example
- [remote_cross_line_monitor.py](../../examples/remote_cross_line_monitor.py) - TCP broadcast server for remote monitoring
- [axinferencenet_example.cpp](../../examples/axinferencenet/axinferencenet_example.cpp) - C++ basic inference
- [axinferencenet_cascaded.cpp](../../examples/axinferencenet/axinferencenet_cascaded.cpp) - C++ multi-model pipeline
- [axinferencenet_tensor.cpp](../../examples/axinferencenet/axinferencenet_tensor.cpp) - C++ tensor access



## Further support

For blog posts, projects and technical support please visit [Axelera AI Community](https://community.axelera.ai/).

For technical documents and guides please visit [Customer Portal](https://support.axelera.ai/).

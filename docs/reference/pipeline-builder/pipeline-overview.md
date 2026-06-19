---
title: "Pipeline Overview"
---
# Pipeline Overview

> [!IMPORTANT]
> **Alpha**
> Core operators (detection, classification, pose, segmentation, tracking) are stable. Cascade (`op.for_each`, `op.crop_roi`) and streaming APIs are still in development.


> [!TIP]
> **New here?**
> Start with the [Quickstart](quickstart.md) for an overview,
> or [Model Compilation](model-compilation.md) if you need to compile a model first.
> This page is the full pipeline reference.


This guide explains how pipeline stages fit together. All examples use
`from axelera.runtime import op`.

## Getting Started

```python
from axelera.runtime import op
import numpy as np

# Build a detection pipeline
pipeline = op.seq(
    op.letterbox(640, 640),
    op.to_tensor(),
    op.load('yolov8n-coco.axm'),
    op.decode_detections(algo='yolov8', num_classes=80),
    op.nms(),
    op.to_image_space(),
    op.ax_detection(class_id_type=op.CocoClasses),
)

# Run on an image (numpy array, HxWxC, uint8, BGR or RGB)
image = np.zeros((480, 640, 3), dtype=np.uint8)  # replace with your image
detections = pipeline(image)  # -> list[DetectedObject]

for det in detections:
    print(f"{det.class_id.name}: {det.score:.0%} at {det.bbox}")
```

---

## Pipeline Stages

Every inference pipeline follows the same pattern:

```
Image -> Transforms -> Model Inference -> Decode -> NMS -> Coordinate Transform -> Result Wrapper
         (preprocess)   (op.load)         (parse     (filter   (to_image_space)    (ax_detection,
                                          raw tensor  overlaps)                     ax_pose, etc.)
                                          output)
```

| Stage | What it does | Example operator |
|-------|-------------|-----------------|
| **Transforms** | Prepare image for model input (resize, normalize, etc.) | `op.letterbox()`, `op.to_tensor()`, `op.normalize()` |
| **Model Inference** | Run the neural network | `op.load('model.axm')` or `op.onnx_model('model.onnx')` |
| **Decode** | Parse raw tensor into structured array | `op.decode_detections()`, `op.decode_pose()`, `op.decode_segmentation()` |
| **NMS** | Remove duplicate overlapping detections | `op.nms()` |
| **Coordinate Transform** | Map from model input space to original image pixels | `op.to_image_space()` |
| **Result Wrapper** (optional) | Convert array to typed Python objects. Without this step you get raw `np.ndarray` which is perfectly usable. | `op.ax_detection()`, `op.ax_pose()`, `op.ax_segmentation()` |

---

## Pipeline Form vs Step-by-Step

There are two ways to use operators: **pipeline form** (`op.seq`) and **step-by-step**.

**Pipeline form** is the recommended approach. `op.seq` chains operators together
into a pipeline. Frame context is managed automatically.

Calling `pipeline.optimized()` analyzes the chain and fuses adjacent operations
(e.g., merging letterbox + to_tensor + normalize into a single step that runs inside
the `.axm` model execution). This reduces memory copies and speeds up inference.
The pipeline works without `optimized()` -- it just runs faster with it.

```python
pipeline = op.seq(
    op.letterbox(640, 640),
    op.to_tensor(),
    op.load('yolov8n-coco.axm'),
    op.decode_detections(algo='yolov8', num_classes=80),
    op.nms(),
    op.to_image_space(),
    op.ax_detection(class_id_type=op.CocoClasses),
)
optimized = pipeline.optimized()   # fuse ops for speed (optional)
detections = optimized(image)
```

**Step-by-step** gives maximum flexibility -- useful for debugging, mixing custom
Python logic between operators, or inspecting intermediate values. You must manage
the frame context manually with `op.frame_context(image)`.

```python
with op.frame_context(image):
    x = op.letterbox(640, 640)(image)
    x = op.to_tensor()(x)
    x = op.load('yolov8n-coco.axm')(x)
    print(f"Raw output shape: {x.shape}")  # inspect intermediate
    x = op.decode_detections(algo='yolov8', num_classes=80)(x)
    x = op.nms()(x)
    x = op.to_image_space()(x)
    detections = op.ax_detection(class_id_type=op.CocoClasses)(x)
```

**Mixing both:** You can use pipeline form for the model portion and step-by-step
for custom surrounding logic:

```python
model_pipeline = op.seq(
    op.load('yolov8n-coco.axm'),
    op.decode_detections(algo='yolov8', num_classes=80),
    op.nms(),
)

with op.frame_context(image):
    preprocessed = op.letterbox(640, 640)(image)
    preprocessed = op.to_tensor()(preprocessed)
    detections_raw = model_pipeline(preprocessed)
    # custom logic here ...
    detections = op.to_image_space()(detections_raw)
    result = op.ax_detection(class_id_type=op.CocoClasses)(detections)
```

---

## Detection Pipeline

```python
pipeline = op.seq(
    # Preprocessing
    op.letterbox(640, 640),                 # Resize with padding, maintain aspect ratio
    op.to_tensor(),                          # HWC uint8 -> CHW float32 [0,1]

    # Model inference
    op.load('yolov8n-coco.axm'),            # Run model, returns raw tensor

    # Postprocessing
    op.decode_detections(                   # Parse raw tensor -> (N, 6) array
        algo='yolov8',                      # [x0, y0, x1, y1, score, class_id]
        num_classes=80,
        confidence_threshold=0.25,
    ),
    op.nms(iou_threshold=0.45, max_boxes=300),  # Remove overlapping boxes
    op.to_image_space(),                    # MODEL_PIXEL -> IMAGE_PIXEL coordinates
    op.ax_detection(class_id_type=op.CocoClasses),  # -> list[DetectedObject]
)

# Use the pipeline
detections = pipeline(image)  # list[DetectedObject]
for det in detections:
    print(f"{det.class_id.name}: {det.score:.0%} at {det.bbox}")
```

**Step-by-step equivalent** (same result, but each step visible):

```python
with op.frame_context(image):
    x = op.letterbox(640, 640)(image)
    x = op.to_tensor()(x)
    x = op.load('yolov8n-coco.axm')(x)
    x = op.decode_detections(algo='yolov8', num_classes=80, confidence_threshold=0.25)(x)
    x = op.nms(iou_threshold=0.45, max_boxes=300)(x)
    x = op.to_image_space()(x)
    detections = op.ax_detection(class_id_type=op.CocoClasses)(x)
```

---

## Classification Pipeline

```python
pipeline = op.seq(
    op.resize(size=256, half_pixel_centers=True),   # Resize smaller edge to 256
    op.center_crop((224, 224)),                       # Center crop to model input
    op.to_tensor(),
    op.normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),

    op.load('squeezenet1.0-imagenet.axm'),
    op.ax_classification(class_id_type=op.ImagenetClasses),  # -> list[Classification]
    op.top_k(k=5),                                           # -> top 5 classifications
)
```

---

## Pose Pipeline

Pose detection adds keypoints to each detection. The keypoint columns flow through
NMS naturally alongside the bounding box.

```python
pipeline = op.seq(
    op.letterbox(640, 640),
    op.to_tensor(),
    op.load('yolov8npose-coco.axm'),
    op.decode_pose(algo='yolov8', num_keypoints=17),  # -> (N, 57) array
    op.nms(iou_threshold=0.45, max_boxes=300),
    op.to_image_space(keypoint_cols=range(6, 57, 3)),  # Transform bbox AND keypoint coords
    op.ax_pose(),                                        # -> list[PoseObject]
)
```

The `keypoint_cols` parameter tells `to_image_space` which columns contain x-coordinates
of keypoints (every 3rd column starting at 6), so they get mapped to image space too.

---

## Segmentation Pipeline

Instance segmentation returns two outputs from the model: detections and prototype masks.
This requires tuple data flow using `par` and `itemgetter`.

```python
pipeline = op.seq(
    op.letterbox(640, 640),
    op.to_tensor(),
    op.load('yolov8nseg-coco.axm'),

    # decode_segmentation returns a plain tuple: (detections, protos)
    op.decode_segmentation(algo='yolov8', num_classes=80),

    # Process detections and protos separately, then recombine
    op.par(
        op.seq(op.itemgetter(0), op.nms(iou_threshold=0.45)),  # NMS on detections
        op.itemgetter(1),                                        # Pass protos through
    ),
    # After unnamed par: detections and protos are separate positional args

    op.par(
        op.seq(op.pack(), op.itemgetter(0), op.to_image_space()),  # Transform det coords
        op.proto_to_mask(),                                         # Compute masks from det+protos
    ),

    op.ax_segmentation(class_id_type=op.CocoClasses),  # -> list[SegmentedObject]
)
```

**Why par + itemgetter?** `decode_segmentation` returns a tuple `(detections, protos)`.
We need to apply NMS only to detections while keeping protos intact. `par` runs two
branches on the same input; `itemgetter(0)` extracts detections, `itemgetter(1)` extracts protos.

**Step-by-step equivalent** (often clearer for segmentation):

```python
with op.frame_context(image):
    x = op.letterbox(640, 640)(image)
    x = op.to_tensor()(x)
    det_raw, proto_raw = op.load('yolov8nseg-coco.axm')(x)
    detections, protos = op.decode_segmentation(algo='yolov8', num_classes=80)(det_raw, proto_raw)
    detections = op.nms(iou_threshold=0.45, max_boxes=300)(detections)
    masks = op.proto_to_mask()(detections, protos)
    detections = op.to_image_space()(detections)
    segments = op.ax_segmentation()(detections, masks)
```

---

## Cascade Pipeline (for_each + crop_roi)

> **Work in progress:** Cascade support (`op.for_each`, `op.crop_roi`) is not yet complete.
> The API shape shown below reflects the planned design but may change.

Cascade pipelines run a second-stage model on each detection from the first stage.
`for_each` iterates over a list, and `crop_roi` extracts the image region for each detection.

```python
pipeline = op.seq(
    # First stage: detect objects
    op.letterbox(640, 640),
    op.to_tensor(),
    op.load('yolov8n-coco.axm'),
    op.decode_detections(algo='yolov8', num_classes=80),
    op.nms(),
    op.to_image_space(),
    op.ax_detection(class_id_type=op.CocoClasses),

    # Second stage: classify each detected region
    op.for_each(
        'classifications',                    # Name for output field
        op.crop_roi(property='bbox'),          # Crop image region from detection bbox
        op.resize(224, 224),
        op.to_tensor(),
        op.load('classifier.axm'),
        op.ax_classification(),
    ),
)
# Result: NamedTuple(input=[DetectedObject, ...], classifications=[Classification, ...])
```

---

## Tracker Integration

Tracking adds persistent identity to detections across video frames.
Place `op.tracker()` after `op.ax_detection()`:

```python
pipeline = op.seq(
    op.letterbox(640, 640),
    op.to_tensor(),
    op.load('yolov8n-coco.axm'),
    op.decode_detections(algo='yolov8', num_classes=80),
    op.nms(),
    op.to_image_space(),
    op.ax_detection(class_id_type=op.CocoClasses),
    op.tracker(algo='bytetrack'),  # -> list[TrackedObject]
)

# TrackedObject has:
#   .track_id       - persistent ID across frames
#   .state          - lifecycle state (see table below)
#   .tracked        - the DetectedObject it matched this frame
#   .predicted_bbox - Kalman-filtered bbox (smoother than raw detection)
```

By default only active tracks are returned. Pass `return_all_states=True` to
also receive `lost` and `removed` tracks -- useful for visualisation, but adds
~7% overhead and breaks MOT evaluation metrics.

### Track States

| State | Meaning |
|-------|---------|
| `new` | Just initialised; not yet confirmed (below `min_hits`) |
| `tracked` | Active and matched by a detection this frame |
| `lost` | No detection matched this frame; held alive for recovery |
| `removed` | Lost too long; will not be returned again (unless `return_all_states=True`) |

### Choosing an Algorithm

| Algorithm | Key Strength | Paper |
|-----------|-------------|-------|
| `'tracktrack'` | Iterative matching with track-aware NMS and appearance-based association; CMC on by default (SOTA) | CVPR 2025 |
| `'oc-sort'` | Observation-centric re-update + virtual trajectory; optional ReID embedding and CMC; boundary-based ID recovery | CVPR 2023 |
| `'bytetrack'` | Simple, robust; handles low-confidence detections via dual-threshold cascade | ECCV 2022 |
| `'sort'` | Pure IoU baseline; minimal overhead | ICIP 2016 |

`tracktrack` gives the highest accuracy with appearance-based association and
CMC on by default. `oc-sort` is a good alternative when you also need ReID
embedding or boundary-based ID recovery. `bytetrack` is a familiar choice if
you are coming from other frameworks, with only two tunable parameters. `sort`
is the lightest option when ID stability is less important.

### Tracking with Pose and Segmentation

The tracker also works after `op.ax_pose()` and `op.ax_segmentation()`.
`TrackedObject.tracked` holds the corresponding pose or segmentation object:

```python
pose_pipeline = op.seq(
    ...,
    op.ax_pose(class_id_type=op.CocoClasses),
    op.tracker(algo='tracktrack'),  # TrackedObject.tracked is a PoseObject
)
```

For the full parameter reference, see
[Tracker](api/axelera.runtime.op.tracker.md).

---

## Model Formats

Two file formats are used for models:

**`.axm` (Axelera Model)** -- A compiled neural network for the Axelera AIPU. This is
the output of `deploy.py` (the compilation step). When loaded with `op.load()`, it
runs the model and returns raw numpy output tensors. You build the surrounding
pipeline (preprocessing, decoding, NMS, etc.) yourself. See
[Model Compilation](model-compilation.md) for how to produce `.axm` files from
PyTorch, ONNX, or Ultralytics models.

**`.axe` (Axelera Executable)** -- A complete pipeline package. It is a ZIP archive
containing a `pipeline.toml` (which describes the full operator chain) and an embedded
`.axm`. When loaded with `op.load()`, you get a ready-to-run pipeline -- no need to
add preprocessing or postprocessing.

`op.load()` handles both formats automatically:

```python
# .axm -- you build the pipeline around it
detector = op.seq(
    op.letterbox(640, 640),
    op.to_tensor(),
    op.load('yolov8n-coco.axm'),       # just the model, returns raw tensors
    op.decode_detections(algo='yolov8', num_classes=80),
    op.nms(),
    op.to_image_space(),
    op.ax_detection(class_id_type=op.CocoClasses),
)

# .axe -- everything is bundled, just call it
detector = op.load('yolov8n-coco.axe')  # complete pipeline
detections = detector(image)
```

To save a pipeline you built as an `.axe` file for later reuse:

```python
pipeline.save_axe('yolov8n-coco.axe')
```

---

## Running a Pipeline

### Input Formats

The primary input format is a **numpy array** (H, W, C), dtype uint8. Torch tensors
and PIL Images are also accepted and converted automatically.

```python
import numpy as np
import cv2

# From file
image = cv2.imread('image.jpg')   # BGR, HxWx3, uint8

# From camera
ret, image = cap.read()           # BGR by default with OpenCV
```

#### The `Image` type

`rt.Image` (`axelera.runtime.Image`) is the runtime's own image type. Unlike a bare
numpy array, it carries its **color format** as metadata, so downstream operators know
whether pixels are RGB, BGR, I420, etc. without being told. `cv.create_source` yields
`Image` frames, and `op.color_convert` *returns* an `Image` so the rest of the pipeline
can track the format.

```python
import axelera.runtime as rt

# Wrap an array, declaring its format (so no fallback/auto-detect is needed)
img = rt.Image.from_array(cv2.imread('photo.jpg'), 'BGR')
img.color_format        # ColorFormat.BGR
img.shape               # (H, W, 3)
img.to_numpy()          # back to an ndarray (zero-copy where possible)
img.convert('RGB')      # -> new Image in RGB
```

`Image.from_any(x, fallback_color_format=...)` accepts an ndarray, torch tensor, PIL
image, or existing `Image`; the `fallback_color_format` is consulted only for formatless
inputs (numpy/torch). A format-carrying input (`Image`/PIL) always wins. Passing a plain
array straight to a pipeline works too — operators wrap it internally — but supplying an
`Image` (or an explicit `src=` on `op.color_convert`) avoids format guesswork.

### Streaming Over Video and Image Folders

Calling the pipeline directly (above) handles a single frame. To run a pipeline
over every frame of a **video file**, a **folder of images**, or a live **USB or
RTSP camera**, use `pipeline.stream`. It reads frames from the source, runs the
pipeline on each, and yields `(image, result)` pairs in source order:

```python
from axelera.runtime import op

pipeline = op.seq(
    op.color_convert('rgb', src='bgr'),
    op.letterbox(640, 640),
    op.to_tensor(),
    op.load('yolov8n-coco.axm'),
    op.decode_detections(algo='yolov8', num_classes=80),
    op.nms(),
    op.to_image_space(),
    op.ax_detection(class_id_type=op.CocoClasses),
)

# The source may be:
#   'clip.mp4'        a video file
#   'frames/'         a directory of images
#   '/dev/video0'     a USB camera
#   'rtsp://host/...' an RTSP/RTP/RTMP/SRT stream URL
#   or a generator that returns Image objects
for image, detections in pipeline.stream('rtsp://camera.local/stream'):
    for det in detections:
        print(f"{det.class_id.name}: {det.score:.0%} at {det.bbox}")
```

Results are yielded in source order, and many frames may be in flight at once.
Frame context is managed automatically, just as when you call the pipeline
directly.

A camera or RTSP stream is **unbounded**: `pipeline.stream` keeps yielding frames
until the stream ends or you break out of the loop. Live sources stay real-time by
dropping frames when the consumer falls behind, whereas a file blocks so no frame
is skipped. Break out of the loop at any time to stop reading the source early.

### Video Sources (`cv.create_source`)

`pipeline.stream` accepts a path directly, but when you want the frames themselves —
plus metadata like frame rate and frame count — open the source explicitly with
`cv.create_source`. It returns a `VideoSource`: an iterable of `Image` frames that also
works as a context manager.

```python
from axelera.runtime import cv

with cv.create_source('clip.mp4') as source:
    print(source.fps, source.frame_count)   # metadata available up front
    for image in source:                     # each frame is an rt.Image
        result = pipeline(image)
```

Key arguments:

- `backend` — `"ffmpeg"` (default) or `"opencv"`. FFmpeg gives the widest format support;
  OpenCV is the fallback.
- `buffer_size` — decode queue depth (default 30); larger absorbs bursts at the cost of memory.
- `live_source` — override the file-vs-live classification. RTSP/RTP/RTMP/SRT/UDP/MJPEG URIs
  and `/dev/video*` are auto-detected as live (drop frames to stay real-time); everything else
  blocks. Pass `live_source=True` for HTTP live streams (HLS), or `False` to force blocking.
- `event_callback(code, message)` — decoder events/errors (FFmpeg backend only).

A `VideoSource` can be handed straight to `pipeline.stream(source)` to get pipelined
`(image, result)` pairs instead of iterating frames yourself.

### Batching (`pipeline.batch`)

To submit several inputs at once — letting the scheduler fill the AIPU more fully than
one-at-a-time calls — collect frames and pass them to `pipeline.batch`. It returns a list
of results in input order. Call `pipeline.optimized()` first so the preprocess/postprocess
ops are fused for the batched path.

```python
pipeline = pipeline.optimized()

queued = [img1, img2, img3, img4]
for img, result in zip(queued, pipeline.batch(queued)):
    ...
```

`pipeline.batch`, `pipeline.stream`, and `pipeline.optimized` all run via the active
scheduler; see [`scheduler`](api/README.md) for device selection.

### Color Handling

Models are typically trained on **RGB** images, but OpenCV reads **BGR**. There are
two approaches:

**Convert before the pipeline** (explicit):

```python
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
detections = pipeline(image_rgb)
```

**Add color conversion inside the pipeline** (self-contained):

```python
pipeline = op.seq(
    op.color_convert('rgb', src='bgr'),  # first operator converts color
    op.letterbox(640, 640),
    op.to_tensor(),
    op.load('yolov8n-coco.axm'),
    ...
)
detections = pipeline(image)  # pass BGR directly
```

Note the color convert here takes a src color, this is only used when the given
image does not have color format embedded e.g. numpy or torch tensor.  rt.Image or
pillow images both have an embedded color format, and in those cases the src color
is ignored.

### Output Types

Result wrappers (`op.ax_detection`, `op.ax_pose`, etc.) are **optional**. They convert
raw numpy arrays into typed Python objects (`DetectedObject`, `PoseObject`, etc.).

**Without wrappers** -- you get raw `np.ndarray` directly, which is perfectly fine
for custom processing or when you want full control:

```python
pipeline_raw = op.seq(
    op.letterbox(640, 640),
    op.to_tensor(),
    op.load('yolov8n-coco.axm'),
    op.decode_detections(algo='yolov8', num_classes=80),
    op.nms(),
    op.to_image_space(),
    # no ax_detection -- returns np.ndarray (N, 6)
)
raw = pipeline_raw(image)  # np.ndarray: [x0, y0, x1, y1, score, class_id]
```

**With wrappers** -- you get typed objects with named attributes (`det.bbox`,
`det.score`, `det.class_id`). Use wrappers when you want:
- **Cascade pipelines**: `op.for_each` + `op.crop_roi` read the `.bbox` attribute from typed objects
- **Built-in rendering**: typed objects have a `.draw()` method for visualization
- **Cleaner code**: `det.class_id.name` instead of `int(row[5])`

```python
detections = pipeline(image)   # list[DetectedObject]
for det in detections:
    print(det.class_id, det.score, det.bbox)
```

For step-by-step use, `frame_context` tracks the original image for coordinate mapping:

```python
with op.frame_context(image) as fc:
    ...
    # fc.input is the original image
    # fc.saved contains letterbox metadata
```

---

## Supported Models

Currently supported model families:

- **YOLO object detection**: YOLOv5, YOLOv7, YOLOv8, YOLOv9, YOLOv10, YOLO11, YOLO26
- **YOLO pose estimation**: YOLOv8-pose, YOLO11-pose
- **YOLO instance segmentation**: YOLOv8-seg, YOLO11-seg
- **Classifiers**: Any model with softmax output (ImageNet-style)

More architectures (the model-zoo models) are coming soon.

---

## Custom Operators

You can create custom operators by subclassing `op.Operator` and implementing `__call__`:

```python
from axelera.runtime import op
import numpy as np

class ScaleScores(op.Operator):
    """Scale all detection scores by a constant factor."""
    factor: float = 1.0

    def __call__(self, detections: np.ndarray) -> np.ndarray:
        result = detections.copy()
        result[:, 4] *= self.factor  # column 4 is score
        return result

# Use in a pipeline
pipeline = op.seq(
    op.load('yolov8n-coco.axm'),
    op.decode_detections(algo='yolov8', num_classes=80),
    ScaleScores(factor=0.9),   # custom operator
    op.nms(),
    op.to_image_space(),
    op.ax_detection(class_id_type=op.CocoClasses),
)
```

Custom operators work with typed result objects too. This example logs
detection counts and passes them through:

```python
class LogDetections(op.Operator):
    """Print how many detections were found."""
    label: str = ''

    def __call__(self, detections: list) -> list:
        print(f"[{self.label}] {len(detections)} detections")
        return detections

pipeline = op.seq(
    ...,
    op.ax_detection(class_id_type=op.CocoClasses),
    LogDetections(label='cam-0'),
)
```

Torch tensors are automatically converted to numpy before reaching `__call__`.

Custom operators are useful for business logic (zone counting, alerts),
drawing, debug logging, or domain-specific post-processing. Where a built-in
operator exists (e.g. `op.filter()` for class/score/bbox filtering), prefer it:
built-in operators are optimized and eligible for future pipeline fusion, while
custom operators are opaque to the optimizer
(see [Postprocess](api/axelera.runtime.op.postprocess.md#filter)).

Custom operators work at runtime but are not saved to `.axe` files unless
registered:

```python
from axelera.runtime.op import OperatorRegistry
OperatorRegistry.register('log_detections', LogDetections)
```

---

## Combinators Quick Reference

| Combinator | Signature | What it does | When to use |
|-----------|-----------|-------------|-------------|
| `op.seq(a, b, c)` | `x -> a(x) -> b(...) -> c(...)` | Execute in order, pipe output to next | Building any pipeline |
| `op.par(a, b)` | `x -> (a(x), b(x))` | Run multiple ops on same input | Processing tuple elements separately |
| `op.for_each(name, ops...)` | `[x1, x2] -> NamedTuple(input, name)` | Apply ops to each list element | Cascade: second-stage model on each detection |
| `op.pack()` | `a, b -> (a, b)` | Collect positional args into tuple | After unnamed par, before itemgetter |
| `op.unpack()` | `(a, b) -> a, b` | Mark tuple for arg unpacking | When operator returns tuple but next expects separate args |
| `op.itemgetter(i)` | `(a, b) -> a` (if i=0) | Extract element from tuple | Selecting from decode output or par results |
| `op.identity(x)` | `x -> x` | Pass through unchanged | Placeholder in par branches |

**Key rule:** Unnamed `par` automatically unpacks its result. Named `par` (all operators have names) returns a NamedTuple that is NOT unpacked.

---
title: "Cascade Tutorial"
---
# Cascade Tutorial

Run a second model on each detection the first one found. A detector says
"there is a bottle here"; a classifier then says which bottle it is. The same
shape covers plate reading behind a vehicle detector, pose behind a person
detector, and segmentation behind anything.

This page builds one cascade from a plain detection pipeline, one step at a
time. Every pipeline here is taken from a runnable demo in
`examples/pipeline_builder/`, named at the end of each step.

**Before you start:** work through the [Quickstart](quickstart.md) so a
single-model pipeline runs. Then fetch the compiled models the demos use:

```bash
python examples/pipeline_builder/download_axm.py
```

They land in `~/.cache/axelera/runtime2/`, which is what `MODEL_DIR` points at
below.

## Step 1: the detector on its own

A cascade starts as an ordinary detection pipeline. Nothing here is
cascade-specific yet.

```python
from pathlib import Path
from axelera.runtime import op, cv

MODEL_DIR = str(Path.home() / ".cache" / "axelera" / "runtime2")

pipeline = op.seq(
    op.color_convert('RGB'),
    op.letterbox(640, 640),
    op.to_tensor(),
    op.load(f'{MODEL_DIR}/yolov8n-coco.axm'),
    op.decode_detections(algo='yolov8', num_classes=80, confidence_threshold=0.4),
    op.nms(iou_threshold=0.5, max_boxes=300),
    op.to_image_space(),
    op.ax_detection(class_id_type=op.CocoClasses),
)
```

It returns a list of `DetectedObject`, each carrying a `.bbox`. That `.bbox` is
the only thing the second stage needs.

## Step 2: add the second stage

`op.for_each` runs the operators you give it once per element of the list it
receives. `op.crop_roi` is what turns an element into an image: it reads the
named property, `bbox` here, and crops that region out of the frame the
pipeline is working on.

Append this to the `op.seq` above:

```python
    op.for_each(
        'classifications',
        op.crop_roi(property='bbox'),
        op.resize(width=224, height=224),
        op.to_tensor(),
        op.normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        op.load(f'{MODEL_DIR}/resnet50-imagenet.axm'),
        op.softmax(),
        op.ax_classification(class_id_type=op.ImagenetClasses),
        op.top_k(k=1),
    ),
```

The body is a whole pipeline of its own: preprocess, model, postprocess. It
sees one crop at a time, so it is written exactly as a single-image
classification pipeline would be.

The first argument, `'classifications'`, names the output. For what else
`for_each` accepts, including renaming the input field with `iter=`, see the
[Combinators reference](../../reference/pipeline-builder/api/axelera.runtime.op.combinators.md).

## Step 3: read the result

A cascade returns both stages, so you can line them up:

```python
result = pipeline(image)

for detection, top1 in zip(result.input, result.classifications):
    label = top1[0].class_id.name if top1 else 'unknown'
    print(label, top1[0].score if top1 else 0.0)
```

`result.input` is the detector's list, unchanged. `result.classifications` is
the second stage's output, one entry per element of `result.input`, in the same
order. `op.top_k(k=1)` returns a list, hence `top1[0]`, and that list is empty
when the classifier returned nothing, so guard it.

Feed it one image, or a whole video or camera:

```python
result = pipeline(img)

with cv.create_source('clip.mp4') as source:
    for img, result in pipeline.stream(source):
        ...
```

`crop_roi` cuts from the frame the pipeline was called with. Both calls record
that frame for you; you open an `op.frame_context(img)` yourself only when you
call operators one at a time instead of through `op.seq`.

The nearest runnable demo is `examples/pipeline_builder/bottle_demo.py`. It is
this pipeline plus the two refinements in the next steps.

## Step 4: send fewer ROIs to the second stage

The second stage runs once per detection, so a frame with eighty detections
runs it eighty times. Two operators cut that down, both placed before
`for_each`:

```python
    op.filter(class_ids=[op.CocoClasses.bottle]),
    op.sort_by(by='center_distance', descending=False, top=10),
```

`op.filter` keeps the classes you care about (it also takes `min_score`,
`min_width` and `min_height`). `op.sort_by` orders the survivors by a numeric
attribute and keeps the first `top` of them.

If you are moving a YAML pipeline over, this is where its ROI selection goes.
`which: CENTER` with `top_k: 10` becomes
`op.sort_by(by='center_distance', descending=False, top=10)`; `which: AREA`
becomes `by='area'` with the default `descending=True`; `which: SCORE` becomes
`by='score'`. The YAML `label_filter` becomes `op.filter(class_ids=[...])`.

## Step 5: cascade behind a tracker

Anything with a `.bbox` can drive a cascade, and a `TrackedObject` has one. Put
`op.tracker` before `for_each` and the second stage runs per track instead of
per raw detection:

```python
    op.filter(class_ids=[op.CocoClasses.car, op.CocoClasses.truck,
                         op.CocoClasses.bus, op.CocoClasses.motorcycle]),
    op.tracker(algo='oc-sort'),
    op.for_each(
        'vehicle_types',
        op.crop_roi(property='bbox'),
        op.resize(size=256, half_pixel_centers=True),
        op.center_crop((224, 224)),
        op.to_tensor(),
        op.normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        op.load(f'{MODEL_DIR}/squeezenet1.0-imagenet.axm'),
        op.softmax(),
        op.ax_classification(class_id_type=op.ImagenetClasses),
        op.top_k(k=1),
    ),
```

`result.input` now holds `TrackedObject`, so each row has a `track_id` you can
carry across frames. A `TrackedObject`'s `bbox` is its Kalman-filtered
`predicted_bbox`, so that is the region `crop_roi` cuts.

Runnable: `examples/pipeline_builder/tracking_with_classification.py`.

## Step 6: nesting and branching

A `for_each` body is an ordinary pipeline, so it can hold another `for_each`,
and a cascade can sit inside one branch of an `op.par` while another branch
works on the full frame:

```python
pipeline = op.seq(
    op.color_convert('RGB'),
    op.letterbox(640, 640),
    op.to_tensor(),
    op.par(
        op.seq(
            # Branch one: pose, then segment inside each of the five largest people.
            op.load(f'{MODEL_DIR}/yolov8lpose-coco-onnx.axm'),
            ...,
            op.sort_by(by='area', top=5),
            op.for_each('fruit_segs', op.crop_roi(property='bbox'), ...),
        ),
        op.seq(
            # Branch two: plain detection on the same frame.
            op.load(f'{MODEL_DIR}/yolov8s-coco-onnx.axm'),
            ...,
        ),
    ),
)
```

Each level of nesting multiplies the number of second-stage calls, so trim with
`filter` and `sort_by` at every level, not only the first.

Runnable: `examples/pipeline_builder/fruit_demo.py`.

## Drawing the results on the original frame

A crop's coordinates are relative to the crop, not to the frame it came from.
The pipeline composes them back for you, and the accessor to use is
`frame_pixels()` rather than `to_pixels()`:

```python
x0, y0, x1, y1 = child.bbox.frame_pixels(width, height)
```

The [Coordinate System Tutorial](coordinate-system-tutorial.md) is the full
account: what each accessor returns, how composition works at three levels or
more, and how to compose coordinates by hand if you are not using `op.seq`.

## What a cascade costs

The second stage runs once per detection, so its per-call time is not its cost.
A row with the largest mean in a profile can be the cheapest part of the run if
it executes once a frame while the row beneath it executes twenty times. Read
the call count beside the mean, and rank by the per-model totals.

[Profiling](profiling.md) works through a real cascade report and the trap in
detail.

## The demos, in order of difficulty

| Demo | Cascade shape |
|---|---|
| `bottle_demo.py` | detect, track, select the ten closest to centre, classify each |
| `tracking_with_classification.py` | detect, filter to vehicles, track, classify each track |
| `fruit_demo.py` | two parallel branches, one of them a pose stage with segmentation nested inside it |

Copy one and edit it rather than starting from an empty file. The
[examples README](../../../examples/pipeline_builder/README.md) lists every demo
and how to run it.

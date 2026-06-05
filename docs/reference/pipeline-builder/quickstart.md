---
title: "Pipeline Builder Quickstart"
---
# Pipeline Builder Quickstart

> [!IMPORTANT]
> **Alpha**
> Core operators (detection, classification, pose, segmentation, tracking)
> are stable. Cascade (`op.foreach`, `op.croproi`) and streaming APIs are still in development.


The Voyager SDK Pipeline Builder runs ML pipelines on the Axelera Metis AIPU.
Two steps: compile your model to a `.axm` file, then build a pipeline
around it. Pre- and postprocessing run as C/C++ operators dispatched
across the AIPU and host; you compose them in Python without managing
hardware placement.

## Quick Example

Two packages, two roles:

- `axelera-devkit` is only needed to **compile** (Step 1). Install it on
  your dev machine.
- `axelera-rt` is the lightweight runtime that **runs** `.axm` files
  (Step 2). Install it anywhere you want to deploy.

Compile once, copy the `.axm` to any target with `axelera-rt` installed,
and run. The devkit does not need to be present at deployment.

For this walkthrough you need both Axelera packages plus `opencv-python`
(used for image I/O). The Axelera packages come from our Artifactory
index; `opencv-python` comes from public PyPI:

```bash
pip install --no-cache-dir \
    --extra-index-url https://software.axelera.ai/artifactory/api/pypi/axelera-pypi/simple \
    axelera-rt axelera-devkit opencv-python
```

Ultralytics is optional. The Quick Example below uses it to export a
YOLO to `.axm` in Step 1, but any compile path (ONNX/PyTorch via the
compiler API, or a pre-built `.axm`) works. Install it only if you want
the YOLO export shortcut:

```bash
pip install ultralytics
```

Ultralytics can also install the Axelera packages on demand during
`export`/`predict`, but with no progress output it's easy to mistake for
a hang, so installing up front is usually better. See
[Install the Voyager SDK](../../user-guides/sdk-install.md) for the full
guide, including runtime-only installs for target devices.

```python
# --- Step 1: Compile (one-time) ---
# Easiest path: Ultralytics export
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.export(format="axelera")
# Output: yolo11n_axelera_model/yolo11n.axm

# --- Step 2: Run (your application) ---
from axelera.runtime2 import op
import urllib.request
import cv2

pipeline = op.seq(
    op.colorconvert("RGB", src="BGR"),  # OpenCV gives BGR; YOLO wants RGB
    op.letterbox(640, 640),
    op.totensor(),
    op.load("yolo11n_axelera_model/yolo11n.axm"),
    op.decode_detections(algo="yolov8", num_classes=80),
    op.nms(),
    op.to_image_space(),
    op.axdetection(class_id_type=op.CocoClasses),
)

# Grab Ultralytics' sample image (swap for your own file anytime).
urllib.request.urlretrieve("https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg", "bus.jpg")
image = cv2.imread("bus.jpg")  # BGR, which the pipeline converts above
detections = pipeline(image)

# `det.class_id` is a CocoClasses enum member because we passed
# `class_id_type=op.CocoClasses` above; `.name` gives the readable label.
# Without `class_id_type`, `det.class_id` is a plain int.
for det in detections:
    print(f"{det.class_id.name}: {det.score:.2f} at {det.bbox}")
```

## Using Your Own YOLO

`model.export(format="axelera")` runs on whatever `.pt` you load, so
custom weights compile the same way:

```python
from ultralytics import YOLO

YOLO("my_trained.pt").export(format="axelera")
# Output: my_trained_axelera_model/my_trained.axm
```

Two things in the runtime snippet then change:

1. **`num_classes`** in `op.decode_detections(...)` must match your model.
2. **Class names.** `op.CocoClasses` only applies to the 80-class COCO
   set. For your own labels, load them from a file with `op.load_classes`
   and pass the result as `class_id_type`:

   ```python
   # Ultralytics dataset YAML (the same one you trained with) works
   # directly: the `names:` field is read whether it's a list or a
   # {id: name} dict. A plain .txt / .names file (one label per line)
   # also works.
   MyClasses = op.load_classes("my_dataset.yaml")

   pipeline = op.seq(
       op.colorconvert("RGB", src="BGR"),
       op.letterbox(640, 640),
       op.totensor(),
       op.load("my_trained_axelera_model/my_trained.axm"),
       op.decode_detections(algo="yolov8", num_classes=len(MyClasses)),
       op.nms(),
       op.to_image_space(),
       op.axdetection(class_id_type=MyClasses),
   )
   ```

`class_id_type` is optional. Omit it and `det.class_id` is a plain int,
which is fine if you only need IDs or will look up names yourself.

## `.axm` vs `.axe`

Two file formats you'll encounter:

- `.axm` is just the compiled model. `op.load()` returns raw output
  tensors; you build the pre/postprocessing around it (as in the Quick
  Example above).
- `.axe` is a bundled pipeline (model + pre/postprocessing). A single
  `op.load("...axe")` gives you a ready-to-run callable.

See [Model Formats](../pipeline/model-formats.md) for the full
description, including how to save a pipeline you've built as an `.axe`.

## Where to Go Next

Already have a `.axm` file? Jump to the
[Pipeline Overview](pipeline-overview.md) for full examples covering detection,
classification, pose estimation, segmentation, tracking, and cascades.

Need to compile first? See [Model Compilation](model-compilation.md) for the
Ultralytics path, the generic ONNX/PyTorch path, and how to validate accuracy
before deploying.

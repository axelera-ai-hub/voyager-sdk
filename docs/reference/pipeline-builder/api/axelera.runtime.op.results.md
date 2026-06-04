---
title: "axelera.runtime.op.results"
---
# `axelera.runtime.op.results`


Result wrapper operators that convert tensor data into typed result objects.

This module contains operators that wrap tensor data into typed result objects:
- AxDetection: Converts detection tensors to list[DetectedObject]
- AxClassification: Converts classification tensors to list[Classification]
- AxPose: Converts pose tensors to list[PoseObject]
- AxSegmentation: Converts segmentation tensors to list[SegmentedObject]

These operators are distinct from postprocessing operators (decode, nms, etc.) as they
perform the final step of wrapping numeric data into domain objects.

Note: These wrappers are optional. Without them, your pipeline returns raw numpy arrays
which you can inspect, process, or pass to other tools directly. Reasons to use the
wrappers: they enable cascade pipelines (op.foreach + op.croproi read the typed bbox
attribute), accuracy measurement tools, and high-performance rendering via the draw()
method -- all planned or available features.

## Summary

| Name | Description |
|------|-------------|
| [AxClassification](#axclassification) | Decode classification results into list[Classification]. |
| [AxDetection](#axdetection) | Convert detection array to list of DetectedObject instances. |
| [AxPose](#axpose) | Convert pose detection array to list of PoseObject instances. |
| [AxSegmentation](#axsegmentation) | Convert segmentation data to list of SegmentedObject instances. |

---

### AxClassification

Decode classification results into list[Classification].

Takes either a raw np.ndarray of class scores or a (values, indices) tuple
from topk and returns a list[Classification].

**Args:**

- **class_id_type**: Enum type for class IDs (e.g., ImagenetClasses, CocoClasses).

**Examples:**

```python
# Pattern 1: All classes (less common)
op.seq(op.load('model'), op.axclassification(...))

# Pattern 2: Top-k classes (recommended, matches detection pattern)
op.seq(
    op.load('model'),
    op.topk(k=5),              # -> (values, indices)
    op.axclassification(...),  # -> list[Classification] for top 5
)
```

**Constructor:**

```python
__init__(class_id_type: type = int)
```

---

### AxDetection

Convert detection array to list of DetectedObject instances.

Takes an np.ndarray with coordinates in IMAGE_PIXEL space and returns a
list[DetectedObject] with index, class_id, bbox, and score.

**Args:**

- **class_id_type**: Type to cast class IDs to (default: int). Common: op.CocoClasses, op.ImagenetClasses, etc.

**Examples:**

```python
# Standard detection pipeline
op.seq(
    op.letterbox(640, 640),
    op.totensor(),
    op.load('yolov8n-coco'),
    op.decode_detections(algo='yolov8', num_classes=80, confidence_threshold=0.25),
    op.nms(),
    op.to_image_space(),  # MODEL_PIXEL -> IMAGE_PIXEL
    op.axdetection(class_id_type=op.CocoClasses),
)
# Input: np.ndarray (N, 6) in IMAGE_PIXEL -> Output: list[DetectedObject]
```

**Note:** Call `to_image_space()` before `axdetection()` to convert coordinates from MODEL_PIXEL to IMAGE_PIXEL.

**Constructor:**

```python
__init__(class_id_type: type = int)
```

---

### AxPose

Convert pose detection array to list of PoseObject instances.

Takes an np.ndarray (N, 6+K*3) with coordinates in IMAGE_PIXEL space and
returns a list[PoseObject] with index, bbox, keypoints, score, and class_id.

**Args:**

- **num_keypoints**: Number of keypoints (e.g., 17 for COCO, 5 for face).
- **keypoint_names**: Optional list of keypoint names (e.g., COCO_KEYPOINT_NAMES).
- **class_id_type**: Type to cast class IDs to (default: int).

**Examples:**

```python
# Standard pose pipeline
op.seq(
    op.letterbox(640, 640),
    op.load('yolov8npose-coco'),
    op.decode_pose(algo='yolov8', num_keypoints=17),
    op.nms(),
    op.to_image_space(keypoint_cols=range(6, 57, 3)),  # MODEL_PIXEL -> IMAGE_PIXEL
    op.axpose(num_keypoints=17, keypoint_names=COCO_KEYPOINT_NAMES),
)
# Input: np.ndarray (M, 57) in IMAGE_PIXEL -> Output: list[PoseObject]
```

**Note:** Call `to_image_space()` with the `keypoint_cols` parameter before `axpose()` to convert coordinates from MODEL_PIXEL to IMAGE_PIXEL.

**Constructor:**

```python
__init__(num_keypoints: int = 17, keypoint_names: list[str] | None = None, class_id_type: type = int)
```

---

### AxSegmentation

Convert segmentation data to list of SegmentedObject instances.

Takes (detections, masks) where detections is an (M, 38) array in
IMAGE_PIXEL space and masks is a list of binary mask arrays. Returns
a list[SegmentedObject].

**Args:**

- **class_id_type**: Type to cast class IDs to (default: int).

**Examples:**

```python
# Explicit segmentation pipeline with tuple data flow
op.seq(
    op.load('yolov8n-seg'),
    op.decode_segmentation(algo='yolov8', num_classes=80),
    op.par(op.seq(op.itemgetter(0), op.nms()), op.itemgetter(1)),
    op.par(
        op.seq(op.pack(), op.itemgetter(0)),
        op.proto_to_mask(),
    ),
    op.par(
        op.seq(op.pack(), op.itemgetter(0), op.to_image_space()),
        op.seq(op.pack(), op.itemgetter(1)),
    ),
    op.axsegmentation(class_id_type=op.CocoClasses),
)
```

Note: Accepts (array, masks) as two separate args or single tuple.

**Constructor:**

```python
__init__(class_id_type: type = int)
```

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
wrappers: they enable cascade pipelines (op.for_each + op.crop_roi read the typed bbox
attribute), accuracy measurement tools, and high-performance rendering via the draw()
method -- all planned or available features.

## Summary

| Name | Description |
|------|-------------|
| [AxClassification](#axclassification) | Decode classification results into list[Classification]. |
| [AxDepthMap](#axdepthmap) | Convert model output tensor to a DepthMap instance. |
| [AxDetection](#axdetection) | Convert detection array to list of DetectedObject instances. |
| [AxObb](#axobb) | Convert OBB detection array to list of OrientedObject instances. |
| [AxPose](#axpose) | Convert pose detection array to list of PoseObject instances. |
| [AxSegmentation](#axsegmentation) | Convert segmentation data to list of SegmentedObject instances. |

---

### AxClassification

**Alias:** `ax_classification`

Decode classification results into list[Classification].

Takes either a raw np.ndarray of class scores or a (values, indices) tuple
from top_k and returns a list[Classification].

**Args:**

- **class_id_type**: Enum type for class IDs (e.g., ImagenetClasses, CocoClasses).

**Examples:**

```python
# Pattern 1: All classes (less common)
op.seq(op.load('model'), op.ax_classification(...))

# Pattern 2: Top-k classes (recommended, matches detection pattern)
op.seq(
    op.load('model'),
    op.top_k(k=5),              # -> (values, indices)
    op.ax_classification(...),  # -> list[Classification] for top 5
)
```

**Constructor:**

```python
__init__(class_id_type: type = int)
```

---

### AxDepthMap

**Alias:** `ax_depth_map`

Convert model output tensor to a DepthMap instance.

Squeezes all size-1 dimensions from the model output and validates the
result is a 2D (H, W) depth array. Raises ValueError if the squeezed
result is not 2D (e.g., a 3-channel model output).

**Examples:**

```python
op.seq(
    op.color_convert('RGB', 'BGR'),
    op.resize((224, 224)),
    op.totensor(),
    op.load('fastdepth-nyudepthv2-onnx.axm'),
    op.ax_depth_map(),
)
# Input: np.ndarray (1, 1, 224, 224) -> Output: DepthMap
```

---

### AxDetection

**Alias:** `ax_detection`

Convert detection array to list of DetectedObject instances.

Takes an np.ndarray with coordinates in NORMALIZED [0,1] space and returns a
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
    op.to_image_space(),  # MODEL_PIXEL -> NORMALIZED
    op.ax_detection(class_id_type=op.CocoClasses),
)
# Input: np.ndarray (N, 6) in NORMALIZED [0,1] -> Output: list[DetectedObject]
```

**Note:**

This operator expects NORMALIZED [0,1] coordinates. Use to_image_space()
before ax_detection() to convert from MODEL_PIXEL to NORMALIZED.

**Constructor:**

```python
__init__(class_id_type: type = int)
```

---

### AxObb

**Alias:** `ax_obb`

Convert OBB detection array to list of OrientedObject instances.

Takes an np.ndarray (N, 7) with columns [cx, cy, w, h, score, class_id, angle]
in NORMALIZED [0,1] space and returns a list[OrientedObject].

**Args:**

- **class_id_type**: Type to cast class IDs to (default: int). Common: op.DotaClasses.

**Examples:**

```python
op.seq(
    op.load('yolo11n-obb.axm'),
    op.decode_obb(num_classes=15),
    op.nms(box_format='xywhr'),
    op.to_image_space(box_format='xywhr'),
    op.ax_obb(class_id_type=op.DotaClasses),
)
```

**Constructor:**

```python
__init__(class_id_type: type = int)
```

---

### AxPose

**Alias:** `ax_pose`

Convert pose detection array to list of PoseObject instances.

Takes an np.ndarray (N, 6+K*3) with coordinates in NORMALIZED [0,1] space and
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
    op.to_image_space(keypoint_cols=range(6, 57, 3)),  # MODEL_PIXEL -> NORMALIZED
    op.ax_pose(num_keypoints=17, keypoint_names=COCO_KEYPOINT_NAMES),
)
# Input: np.ndarray (M, 57) in NORMALIZED [0,1] -> Output: list[PoseObject]
```

**Note:**

This operator expects NORMALIZED [0,1] coordinates. Use to_image_space()
with keypoint_cols parameter before ax_pose() to convert from MODEL_PIXEL.

**Constructor:**

```python
__init__(num_keypoints: int = 17, keypoint_names: list[str] | None = None, class_id_type: type = int)
```

---

### AxSegmentation

**Alias:** `ax_segmentation`

Convert segmentation data to list of SegmentedObject instances.

Takes (detections, masks) where detections is an (M, 38) array in
NORMALIZED [0,1] space and masks is a list of binary mask arrays. Returns
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
    op.ax_segmentation(class_id_type=op.CocoClasses),
)
```

Note: Accepts (array, masks) as two separate args or single tuple.

**Constructor:**

```python
__init__(class_id_type: type = int)
```

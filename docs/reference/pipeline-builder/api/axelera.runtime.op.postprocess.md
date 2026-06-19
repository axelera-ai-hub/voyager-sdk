---
title: "axelera.runtime.op.postprocess"
---
# `axelera.runtime.op.postprocess`


Postprocessing operators for converting raw model outputs to structured results.

Pipeline stages after model inference:
    1. Decode (DecodeDetections/DecodePose/DecodeSegmentation):
       Parse raw model tensors into standardized array format
    2. Nms: Remove duplicate overlapping detections
    3. ToImageSpace: Map coordinates from model input space to original image pixels
    4. (In results module) AxDetection/AxPose/AxSegmentation: Wrap into typed Python objects

Additional operators: Filter, TopK, Top1, ProtoToMask, Split.

## Summary

| Name | Description |
|------|-------------|
| [DecodeDetections](#decodedetections) | Parse raw detection model output into a standardized bounding box array. |
| [DecodeObb](#decodeobb) | Parse raw OBB (Oriented Bounding Box) model output into a standardized array. |
| [DecodePose](#decodepose) | Parse raw pose model output into a structured array with bounding boxes and keypoints. |
| [DecodeSegmentation](#decodesegmentation) | Parse raw segmentation model output into detection and prototype arrays. |
| [Nms](#nms) | Non-maximum suppression to remove duplicate overlapping detections. |
| [ToImageSpace](#toimagespace) | Map coordinates from model input space to normalized [0,1] image-relative values. |
| [ProtoToMask](#prototomask) | Compute per-detection binary masks from mask coefficients and prototype features. |
| [Filter](#filter) | Filter detections based on class IDs, score, or custom function. |
| [Split](#split) | Split detections into two lists: matching and non-matching. |
| [TopK](#topk) | Return the k largest/smallest elements along a given dimension. |
| [Top1](#top1) | Convenience operator returning only the top-1 element (k=1). |

---

### DecodeDetections

**Alias:** `decode_detections`

Parse raw detection model output into a standardized bounding box array.

Takes the raw tensor from a detection model (YOLOv5/v7/v8/v9/v10/v11/v26) and
converts it into a clean (N, 6) array: [x0, y0, x1, y1, score, class_id].
This is the first postprocessing step after model inference.

**Args:**

- **algo**: Detection algorithm ('yolov5', 'yolov7', 'yolov8', 'yolov9', 'yolov10', 'yolo11').
- **num_classes**: Number of classes (required).
- **confidence_threshold**: Minimum confidence score (default: 0.25).
- **max_boxes_pre_nms**: Maximum boxes to keep before NMS (default: 30000).

**Examples:**

```python
# Standard YOLOv8 detection pipeline
op.seq(
    op.load('yolov8n-coco'),
    op.decode_detections(algo='yolov8', num_classes=80, confidence_threshold=0.25),
    op.nms(),
    op.to_image_space(),
    op.ax_detection(class_id_type=op.CocoClasses),
)
# Input: raw model output -> Output: (N, 6) array -> list[DetectedObject]
```

**Note:** Filters boxes below the confidence threshold before returning results.

**Constructor:**

```python
__init__(algo: str, num_classes: int, confidence_threshold: float = field(default=0.25, metadata={'range': (0.0, 1.0)}), max_boxes_pre_nms: int = field(default=30000, metadata={'range': (1, None)}), input_format: str = 'auto')
```

---

### DecodeObb

**Alias:** `decode_obb`

Parse raw OBB (Oriented Bounding Box) model output into a standardized array.

Takes the raw tensor from an OBB model (YOLOv8/v11-obb) and converts it into
a (N, 7) array: [cx, cy, w, h, score, class_id, angle].

The input format is [cx, cy, w, h, class_scores..., angle] per anchor.
Both the ONNX postamble and the optimized UltralyticsYoloPostamble produce
this format.

**Args:**

- **num_classes**: Number of classes (required).
- **confidence_threshold**: Minimum confidence score (default: 0.25).
- **max_boxes_pre_nms**: Maximum boxes to keep before NMS (default: 30000).

**Examples:**

```python
op.seq(
    op.load('yolo11n-obb.axm'),
    op.decode_obb(num_classes=15, confidence_threshold=0.25),
    op.nms(iou_threshold=0.45, box_format='xywhr'),
    op.to_image_space(box_format='xywhr'),
    op.ax_obb(class_id_type=op.DotaClasses),
)
```

**Constructor:**

```python
__init__(num_classes: int, confidence_threshold: float = field(default=0.25, metadata={'range': (0.0, 1.0)}), max_boxes_pre_nms: int = field(default=30000, metadata={'range': (1, None)}))
```

---

### DecodePose

**Alias:** `decode_pose`

Parse raw pose model output into a structured array with bounding boxes and keypoints.

Takes the raw tensor from a pose model and converts it into a (N, 6+K*3) array
where each row has a bounding box, score, class ID, and K keypoint triplets
(x, y, confidence).

**Args:**

- **algo**: Pose algorithm ('yolov8', 'yolo11').
- **num_keypoints**: Number of keypoints (e.g., 17 for COCO body, 5 for face).
- **confidence_threshold**: Minimum confidence score (default: 0.25).
- **max_boxes_pre_nms**: Maximum boxes to keep before NMS (default: 30000).
- **input_format**: Coordinate format - 'auto', 'xyxy', 'cxcywh' (default: 'auto').

**Examples:**

```python
# COCO body pose (17 keypoints)
op.seq(
    op.load('yolov8n-pose'),
    op.decode_pose(algo='yolov8', num_keypoints=17),
    op.nms(),  # Works unchanged - preserves keypoint columns
    op.ax_pose(num_keypoints=17),
)
# Input: raw model output -> (N, 57) -> (M, 57) -> list[PoseObject]
```

**Note:**

NMS operates on columns 0-4 and returns full rows, preserving keypoint columns
automatically.

**Constructor:**

```python
__init__(algo: str, num_keypoints: int, confidence_threshold: float = field(default=0.25, metadata={'range': (0.0, 1.0)}), max_boxes_pre_nms: int = field(default=30000, metadata={'range': (1, None)}), input_format: str = 'auto')
```

---

### DecodeSegmentation

**Alias:** `decode_segmentation`

Parse raw segmentation model output into detection and prototype arrays.

Takes the raw tensor outputs from an instance segmentation model and returns
a tuple (detections, protos) where detections is (N, 38) and protos is
(32, H, W).

**Args:**

- **algo**: Segmentation algorithm ('yolov8', 'yolo11').
- **num_classes**: Number of classes.
- **num_mask_coeffs**: Number of mask coefficients (default: 32).
- **confidence_threshold**: Minimum confidence score (default: 0.25).
- **max_boxes_pre_nms**: Maximum boxes to keep before NMS (default: 30000).
- **input_format**: Coordinate format - 'auto', 'xyxy', 'cxcywh' (default: 'auto').

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

**Note:**

Uses the `par` + `itemgetter` pattern to process detections through NMS while
passing protos through unchanged.

**Constructor:**

```python
__init__(algo: str, num_classes: int, num_mask_coeffs: int = 32, confidence_threshold: float = field(default=0.25, metadata={'range': (0.0, 1.0)}), max_boxes_pre_nms: int = field(default=30000, metadata={'range': (1, None)}), input_format: str = 'auto')
```

---

### Nms

**Alias:** `nms`

Non-maximum suppression to remove duplicate overlapping detections.

Takes an np.ndarray (N, M) where columns 0:4 are boxes, column 4 is
the score, and columns 5+ are pass-through data. Returns the same format
with duplicate/overlapping boxes removed.

**Args:**

- **iou_threshold**: IOU threshold for suppression (default: 0.45). Higher = more boxes kept, lower = more aggressive suppression.
- **class_agnostic**: If False, apply NMS per-class (default: False). If True, apply NMS across all classes together.
- **max_boxes**: Maximum boxes to return after NMS (default: 300).
- **backend**: NMS implementation - 'opencv' or 'torch' (default: 'opencv').
- **box_format**: Box format - 'xyxy' (default) or 'xywhr' for OBB. When 'xywhr', uses rotated NMS with angle from column 6.

**Examples:**

```python
# Standard detection pipeline
op.nms(iou_threshold=0.45, max_boxes=300)

# OBB (oriented bounding box) pipeline
op.nms(iou_threshold=0.45, box_format='xywhr')
```

**Note:**

Extra columns beyond the box and score are passed through unchanged. Works in any
coordinate space -- typically used in MODEL_PIXEL space before
`to_image_space()` conversion.

**Constructor:**

```python
__init__(iou_threshold: float = field(default=0.45, metadata={'range': (0.0, 1.0)}), class_agnostic: bool = False, max_boxes: int = field(default=300, metadata={'range': (1, None)}), backend: str = 'opencv', box_format: str = 'xyxy')
```

---

### ToImageSpace

**Alias:** `to_image_space`

Map coordinates from model input space to normalized [0,1] image-relative values.

After model inference, all coordinates are in MODEL_PIXEL space -- the resized and
letterboxed input (e.g., 640x640). Without this operator, bounding boxes shown on the
original image will be misaligned because they reference the wrong resolution and include
letterbox padding offsets. This operator reverses the letterbox transform and scales
coordinates to normalized [0,1] values relative to the original image.

Takes an np.ndarray with MODEL_PIXEL coordinates and returns an np.ndarray with
NORMALIZED [0,1] coordinates. Letterbox metadata is read from the frame context (set
automatically by op.letterbox()).

**Args:**

- **box_cols**: Tuple (start, end) for box coordinates. Default: (0, 4).
- **keypoint_cols**: Column indices for keypoint x,y pairs. Default: None.

**Examples:**

```python
# Detection
transformed = to_image_space(detections)  # -> np.ndarray

# Pose
transformed = to_image_space(pose_data, keypoint_cols=range(6, 57, 3))

# Segmentation (use with par + pack + itemgetter pattern)
op.par(
    op.seq(op.pack(), op.itemgetter(0), op.to_image_space()),
    op.seq(op.pack(), op.itemgetter(1)),
)
```

**Note:** Use in combination with proto_to_mask for segmentation pipelines.

**Constructor:**

```python
__init__(box_cols: tuple[int, int] | None = (0, 4), box_format: str = 'xyxy', keypoint_cols: list[int] | range | None = None)
```

---

### ProtoToMask

**Alias:** `proto_to_mask`

Compute per-detection binary masks from mask coefficients and prototype features.

Instance segmentation models (e.g., YOLOv8-seg) output two tensors: per-detection
mask coefficients (one 32-element vector per detection) and a prototype feature map
(32, H, W). This operator combines them: for each detection it computes
coefficients @ prototypes.reshape(32, H*W), applies sigmoid, then crops to the
detection's bounding box in prototype space and thresholds to produce a binary mask.

Takes (detections, protos) where detections is an (M, 38) array and protos
is a (32, H, W) prototype tensor. Returns a list of M binary mask arrays.

Must be called BEFORE to_image_space() because mask cropping relies on MODEL_PIXEL
bounding box coordinates to index into the prototype grid correctly.

**Args:**

- **mask_threshold**: Threshold for binary mask creation (default: 0.5).
- **num_mask_coeffs**: Number of mask coefficients per detection (default: 32).

**Examples:**

```python
# Explicit segmentation pipeline with tuple data flow
op.seq(
    op.decode_segmentation(...),
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

**Note:**

Mask cropping operates in prototype space. Call this BEFORE `to_image_space()`
transforms coordinates.

**Constructor:**

```python
__init__(mask_threshold: float = field(default=0.5, metadata={'range': (0.0, 1.0)}), num_mask_coeffs: int = 32)
```

---

### Filter

**Alias:** `filter`

Filter detections based on class IDs, score, or custom function.

Takes a list[DetectedObject] and returns only those matching the filter
criteria.

**Args:**

- **class_ids**: Keep only these class IDs (optional).
- **min_score**: Minimum confidence score (optional).
- **fn**: Custom filter function (optional).

**Examples:**

```python
# Filter by class - keep only people
op.filter(class_ids=[op.CocoClasses.person])

# Filter by score - keep high-confidence detections
op.filter(min_score=0.90)

# Combine filters - high-confidence people
op.filter(class_ids=[op.CocoClasses.person], min_score=0.90)

# In cascade pipeline
op.seq(
    op.load('yolov8n-coco'),
    op.decode_detections(...),
    op.nms(),
    op.to_image_space(),
    op.ax_detection(class_id_type=op.CocoClasses),
    op.filter(class_ids=[op.CocoClasses.person]),  # Only process people
    op.for_each('crops', op.crop_roi(property='bbox'), ...),
)
```

Note: Multiple filters are AND'd together -- all must match.

---

### Split

**Alias:** `split`

Split detections into two lists: matching and non-matching.

Like Filter, but returns both halves as a tuple (matching, non_matching)
instead of discarding non-matching detections.

---

### TopK

**Alias:** `top_k`

Return the k largest/smallest elements along a given dimension.

Follows PyTorch's torch.top_k API. For np.ndarray input, returns a tuple
of (values, indices). For list[Classification] input, returns the top-k
Classification objects.

**Args:**

- **k**: Number of top elements to return.
- **dim**: Dimension to sort along (None for flattened array).
- **largest**: If True, return k largest; if False, k smallest (default: True).
- **sorted**: Whether to return elements in sorted order (default: True).

**Examples:**

```python
# Recommended: top_k -> ax_classification (matches detection pattern)
op.seq(
    op.load('model'),
    op.top_k(k=5),                    # -> (values, indices)
    op.ax_classification(...),        # -> list[Classification]
)
```

**Constructor:**

```python
__init__(k: int = field(metadata={'range': (1, None)}), dim: int | None = None, largest: bool = True, sorted: bool = True)
```

---

### Top1

**Alias:** `top1`

Convenience operator returning only the top-1 element (k=1).

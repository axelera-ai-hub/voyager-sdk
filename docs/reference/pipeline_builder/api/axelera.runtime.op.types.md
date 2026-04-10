# `axelera.runtime.op.types`

Data types for the op package: BBox, DetectedObject, TrackedObject, Classification, etc.

## Summary

| Name | Description |
|------|-------------|
| [BBox](#bbox) | Bounding box in XYXY format (x0, y0, x1, y1). |
| [DetectedObject](#detectedobject) | A detected object with bounding box and classification. |
| [PoseObject](#poseobject) | A detected pose with keypoints. |
| [SegmentedObject](#segmentedobject) | An instance segmentation with mask. |
| [TrackedObject](#trackedobject) | A tracked object with persistent track ID and lifecycle state. |
| [Classification](#classification) | A classification result with class ID and confidence score. |
| [Keypoint](#keypoint) | A single keypoint with position and confidence. |
| [CoordFormat](#coordformat) | Coordinate format for bounding boxes. |
| [CoordSpace](#coordspace) | Coordinate space for bounding box values. |
| [TrackedObjectState](#trackedobjectstate) | Lifecycle state of a tracked object across video frames. |

---

### BBox

Bounding box in XYXY format (x0, y0, x1, y1).

The internal storage is always XYXY. Use the format methods to get
coordinates in other formats without modifying the instance.

**Constructor:**

```python
__init__(x0: float, y0: float, x1: float, y1: float)
```

---

### DetectedObject

A detected object with bounding box and classification.

**Attributes:**

- **index** (`int`): Sequential index of this detection in the current frame.
- **class_id** (`int | enum.Enum | None`): Class identifier (int or enum like CocoClasses).
- **bbox** (`BBox | None`): Bounding box in IMAGE_PIXEL coordinates (original image space).
- **score** (`float | None`): Detection confidence score [0-1].

Note: Coordinates are in pixel space, ready for visualization.

**Constructor:**

```python
__init__(index: int = 0, class_id: int | enum.Enum | None = None, bbox: BBox | None = None, score: float | None = None)
```

---

### PoseObject

A detected pose with keypoints.

Supports variable keypoint counts for different pose formats
(COCO body 17, face 5/68, hand 21, etc.).

**Attributes:**

- **index** (`int`): Detection index within the frame
- **keypoints** (`list[Keypoint]`): List of keypoints in IMAGE_PIXEL coordinates
- **bbox** (`BBox | None`): Bounding box in IMAGE_PIXEL coordinates
- **score** (`float | None`): Confidence score [0,1]
- **class_id** (`int | enum.Enum | None`): Class identifier (optional, for multi-class pose)

**Note:** Use to_image_space() before axpose() to convert from MODEL_PIXEL.

**Constructor:**

```python
__init__(index: int = 0, keypoints: list[Keypoint] = field(default_factory=list), bbox: BBox | None = None, score: float | None = None, class_id: int | enum.Enum | None = None)
```

---

### SegmentedObject

An instance segmentation with mask.

**Attributes:**

- **index** (`int`): Detection index within the frame
- **class_id** (`int | enum.Enum | None`): Class identifier (int or enum)
- **mask** (`np.ndarray | None`): Binary mask (H, W) in prototype resolution
- **bbox** (`BBox | None`): Bounding box in IMAGE_PIXEL coordinates
- **score** (`float | None`): Confidence score [0,1]

**Note:** Use to_image_space() before axsegmentation() to convert from MODEL_PIXEL.

**Constructor:**

```python
__init__(index: int = 0, class_id: int | enum.Enum | None = None, mask: np.ndarray | None = None, bbox: BBox | None = None, score: float | None = None)
```

---

### TrackedObject

A tracked object with persistent track ID and lifecycle state.

**Attributes:**

- **track_id** (`int`): Unique identifier for this track, persistent across frames.
- **predicted_bbox** (`BBox`): Kalman-filtered predicted bounding box.
- **state** (`TrackedObjectState`): Current lifecycle state (new, tracked, lost, removed).
- **tracked** (`DetectedObject`): The DetectedObject this track is associated with.
- **latest_det_id** (`int`): Index into the CURRENT frame's detection list. - >= 0: Valid index, can be used to access detections[latest_det_id] - -1: No detection in current frame (lost/removed track) Note: This refers to the current frame only. For lost/removed tracks, this is -1 because there is no matching detection in this frame.

**Constructor:**

```python
__init__(track_id: int, predicted_bbox: BBox, state: TrackedObjectState, tracked: DetectedObject, latest_det_id: int = -1, bbox: BBox | None, class_id: int | enum.Enum | None, score: float | None)
```

---

### Classification

A classification result with class ID and confidence score.

**Attributes:**

- **class_id** (`int`): Class identifier (int or enum like ImagenetClasses).
- **score** (`float`): Classification confidence score [0-1].

**Constructor:**

```python
__init__(class_id: int, score: float)
```

---

### Keypoint

A single keypoint with position and confidence.

Coordinates are in the same space as the parent PoseObject (IMAGE_PIXEL).

**Attributes:**

- **x** (`float`): X coordinate in IMAGE_PIXEL space
- **y** (`float`): Y coordinate in IMAGE_PIXEL space
- **confidence** (`float`): Detection confidence [0,1]
- **name** (`str | None`): Optional keypoint name (e.g., 'nose', 'left_eye')

**Constructor:**

```python
__init__(x: float, y: float, confidence: float)
```

---

### CoordFormat

Coordinate format for bounding boxes.

XYXY: (x0, y0, x1, y1) - top-left and bottom-right corners (exclusive)
XYWH: (x_center, y_center, width, height) - center point with dimensions
LTWH: (left, top, width, height) - top-left corner with dimensions

**Constructor:**

```python
__init__(XYXY='xyxy', XYWH='xywh', LTWH='ltwh')
```

---

### CoordSpace

Coordinate space for bounding box values.

**Normalized:** This is the standard storage format for all DetectedObject.bbox values.

MODEL_PIXEL: Pixel coordinates in model input space (e.g., 0-640 for 640x640 model).
    This is typically what decoders output before normalization.
IMAGE_PIXEL: Pixel coordinates in original image space.
    Final output for visualization after calling to_pixel_coords().

**Constructor:**

```python
__init__(NORMALIZED='normalized', MODEL_PIXEL='model_pixel', IMAGE_PIXEL='image_pixel')
```

---

### TrackedObjectState

Lifecycle state of a tracked object across video frames.

**Attributes:**

- **new**: Track was just initialized; not yet confirmed (hit count below min_hits).
- **tracked**: Track is active and confirmed by consistent detections.
- **lost**: Track had no matching detection this frame; held alive for recovery. Only returned when return_all_states=True.
- **removed**: Track was lost for too long and will no longer be returned. Only returned when return_all_states=True.

**Constructor:**

```python
__init__(new=0, tracked=1, lost=2, removed=3)
```

---
title: "axelera.runtime.op.types"
---
# `axelera.runtime.op.types`


Data types for the op package: BBox, DetectedObject, TrackedObject, Classification, etc.

## Summary

| Name | Description |
|------|-------------|
| [BBox](#bbox) | Bounding box in XYXY format (x0, y0, x1, y1). |
| [DepthMap](#depthmap) | A depth estimation result holding a 2D depth array. |
| [DetectedObject](#detectedobject) | A detected object with bounding box and classification. |
| [OrientedObject](#orientedobject) | An oriented (rotated) detected object with XYWHR bounding box. |
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

Coordinates are normalized [0,1] relative to the local region.
Use `to_pixels(w, h)` for pixel conversion within the local region,
or `frame_pixels(w, h)` for frame-level pixel coordinates when nested.

**Constructor:**

```python
__init__(x0: float, y0: float, x1: float, y1: float)
```

**Properties:**

- **`width`** (`float`)
- **`height`** (`float`)

**Methods:**

#### in_frame_of

```python
in_frame_of(parent: BBox) -> BBox
```

Map local normalized coordinates into the parent's coordinate space.

Pure arithmetic composition: `parent.x0 + self.x0 * parent.width`.
The returned BBox has no `_frame` set (caller decides what to attach).

#### to_pixels

```python
to_pixels(w: int, h: int) -> tuple[int, int, int, int]
```

Convert normalized [0,1] coordinates to clamped pixel values.

#### frame_pixels

```python
frame_pixels(w: int, h: int) -> tuple[int, int, int, int]
```

Return frame-space pixel coordinates, using `_frame` if populated.

#### xyxy

```python
xyxy() -> tuple[float, float, float, float]
```

Return bbox as (x0, y0, x1, y1) tuple.

#### xywh

```python
xywh() -> tuple[float, float, float, float]
```

Return bbox as (center_x, center_y, width, height) tuple.

#### ltwh

```python
ltwh() -> tuple[float, float, float, float]
```

Return bbox as (left, top, width, height) tuple.

#### classmethod from_xywh

```python
from_xywh(cx: float, cy: float, w: float, h: float) -> BBox
```

Create BBox from center-width-height format.

#### classmethod from_ltwh

```python
from_ltwh(left: float, top: float, w: float, h: float) -> BBox
```

Create BBox from left-top-width-height format.

---

### DepthMap

A depth estimation result holding a 2D depth array.

The depth array is stored at model output resolution (e.g., 224x224).
Measurement/evaluation accesses the raw array directly via `depth`.
Rendering (via `draw()`) handles normalization and resize to canvas
using the INFERNO colormap.

**Attributes:**

- **depth** (`np.ndarray`): 2D float32 array (H, W) of depth values at model resolution.

**Constructor:**

```python
__init__(depth: np.ndarray = field(default_factory=(lambda: np.empty(0))))
```

---

### DetectedObject

A detected object with bounding box and classification.

**Attributes:**

- **index** (`int`): Sequential index of this detection in the current frame.
- **class_id** (`int | enum.Enum | None`): Class identifier (int or enum like CocoClasses).
- **bbox** (`BBox | None`): Bounding box with normalized [0,1] coordinates.   Use `bbox.to_pixels(w, h)` for local pixel coords or   `bbox.frame_pixels(w, h)` for frame-level pixel coords.
- **score** (`float | None`): Detection confidence score [0-1].

**Constructor:**

```python
__init__(index: int = 0, class_id: int | enum.Enum | None = None, bbox: BBox | None = None, score: float | None = None)
```

---

### OrientedObject

An oriented (rotated) detected object with XYWHR bounding box.

Separate from DetectedObject because OBB geometry is fundamentally different:
center-format boxes with rotation angle, no xyxy corners. Downstream operators
that assume xyxy (crop_roi, tracker) cannot work with OBB.

Coordinates are normalized [0,1] relative to the local region.
Use `to_corners_pixels(w, h)` for pixel conversion within the local region,
or `frame_corners(w, h)` for frame-level pixel coordinates when nested.

**Attributes:**

- **index** (`int`): Sequential index of this detection in the current frame.
- **class_id** (`int | enum.Enum | None`): Class identifier (int or enum like DotaClasses).
- **cx** (`float`): Center x in NORMALIZED [0,1] coordinates.
- **cy** (`float`): Center y in NORMALIZED [0,1] coordinates.
- **w** (`float`): Width in NORMALIZED [0,1] coordinates.
- **h** (`float`): Height in NORMALIZED [0,1] coordinates.
- **angle** (`float`): Rotation angle in radians.
- **score** (`float | None`): Detection confidence score [0-1].

**Constructor:**

```python
__init__(index: int = 0, class_id: int | enum.Enum | None = None, cx: float = 0.0, cy: float = 0.0, w: float = 0.0, h: float = 0.0, angle: float = 0.0, score: float | None = None)
```

**Properties:**

- **`xywhr`** (`tuple[float, float, float, float, float]`) — Return box as (cx, cy, w, h, angle) tuple.
- **`corners`** (`list[tuple[float, float]]`) — Compute 4 corner points of the rotated rectangle in normalized coords.
- **`bbox`** (`BBox`) — Axis-aligned bounding box (AABB) enclosing the rotated rectangle.

**Methods:**

#### to_corners_pixels

```python
to_corners_pixels(w: int, h: int) -> list[tuple[int, int]]
```

Pixel-space corners of the rotated rectangle, clamped to (w, h).

Rotation is applied in pixel coordinates so the rectangle stays a
true rectangle on non-square images. Rotating in normalized [0,1]
space first and then scaling anisotropically by (w, h) would shear
the box into a parallelogram.

#### frame_corners

```python
frame_corners(w: int, h: int) -> list[tuple[int, int]]
```

Return frame-level pixel corners, using _frame_bbox if populated.

When nested inside a ForEach, the OBB's normalized coords are relative
to the crop region. This method maps them to the original frame by
composing through the parent's frame box. Rotation is applied in
pixel space (see to_corners_pixels) for the same reason.

---

### PoseObject

A detected pose with keypoints.

Supports variable keypoint counts for different pose formats
(COCO body 17, face 5/68, hand 21, etc.).

**Attributes:**

- **index** (`int`): Detection index within the frame
- **keypoints** (`list[Keypoint]`): List of keypoints with normalized [0,1] coordinates
- **bbox** (`BBox | None`): Bounding box with normalized [0,1] coordinates
- **score** (`float | None`): Confidence score [0,1]
- **class_id** (`int | enum.Enum | None`): Class identifier (optional, for multi-class pose)

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
- **bbox** (`BBox | None`): Bounding box with normalized [0,1] coordinates
- **score** (`float | None`): Confidence score [0,1]

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
__init__(track_id: int, predicted_bbox: BBox, state: TrackedObjectState, tracked: DetectedObject, latest_det_id: int = -1)
```

**Properties:**

- **`bbox`** (`BBox | None`) — Return predicted_bbox for protocol compatibility with Filter/Split.
- **`class_id`** (`int | enum.Enum | None`) — Return class_id from tracked detection for protocol compatibility.
- **`score`** (`float | None`) — Return score from tracked detection for protocol compatibility.

**Methods:**

#### draw

```python
draw(draw)
```

Draw the tracked object using the unified visualization strategy.

Visualization Strategy (see display.draw_tracked_box for full details):
- COLOR = track identity: Each track_id gets a unique color, allowing
  viewers to follow specific objects across frames ("the red car").
- ALPHA = state: Opacity indicates lifecycle (tracked=100%, lost=40%, etc.)

This dual-property approach shows both identity and state simultaneously.

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

Coordinates are normalized [0,1] relative to the image region.
Use `frame_x(w)` and `frame_y(h)` for frame-level pixel coordinates
when nested inside a ForEach cascade.

**Attributes:**

- **x** (`float`): X coordinate, normalized [0,1]
- **y** (`float`): Y coordinate, normalized [0,1]
- **confidence** (`float`): Detection confidence [0,1]
- **name** (`str | None`): Optional keypoint name (e.g., 'nose', 'left_eye')

**Constructor:**

```python
__init__(x: float, y: float, confidence: float)
```

**Methods:**

#### frame_x

```python
frame_x(w: int) -> int
```

Return frame-level pixel x, using _frame_x if populated.

#### frame_y

```python
frame_y(h: int) -> int
```

Return frame-level pixel y, using _frame_y if populated.

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
    Produced by fc.map_bbox() for power-user raw-tensor workflows;
    pipeline operators output NORMALIZED instead.

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

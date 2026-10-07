# Coordinate System Tutorial

How coordinates work in Pipeline Builder pipelines, from single-model detection to multi-level cascades.

## Quick Reference

```python
# Get pixel coordinates for drawing (detection, pose, segmentation)
x0, y0, x1, y1 = det.bbox.to_pixels(img_width, img_height)

# Get frame-level pixels (works at any nesting depth)
x0, y0, x1, y1 = det.bbox.frame_pixels(img_width, img_height)

# OBB: get frame-level rotated corners
points = obb_det.frame_corners(img_width, img_height)

# Keypoints: get frame-level pixel coordinates
px, py = kp.frame_x(img_width), kp.frame_y(img_height)

# Manual coordinate composition (power-user)
frame_bbox = child_bbox.in_frame_of(parent_bbox)
```

**Key rule:** BBox stores normalized [0,1] coordinates. Call `to_pixels()` or `frame_pixels()` when you need integers for drawing.

---

## BBox API

### Storage: Normalized [0,1]

All BBox coordinates are normalized relative to their local region:

```python
det.bbox.x0  # float in [0, 1]: left edge as fraction of region width
det.bbox.y0  # float in [0, 1]: top edge as fraction of region height
det.bbox.x1  # float in [0, 1]: right edge
det.bbox.y1  # float in [0, 1]: bottom edge
```

Why normalized? Because normalized coordinates compose via pure arithmetic.
`parent.x0 + child.x0 * parent.width` gives you the child's position in the
parent's coordinate system, with no image dimensions needed until you want pixels.
This is what makes multi-level cascades work without passing image sizes around.

### `to_pixels(w, h)`: convert to pixel integers

```python
x0, y0, x1, y1 = bbox.to_pixels(img_width, img_height)
# Returns clamped integers: (int, int, int, int)
```

Multiplies normalized coords by image dimensions, rounds, and clamps to `[0, w]` / `[0, h]`.
Use this when you have a BBox relative to a known image and need pixel coordinates.

### `frame_pixels(w, h)`: frame-level pixel coordinates

```python
x0, y0, x1, y1 = bbox.frame_pixels(img_width, img_height)
```

Inside a cascade, the pipeline also records each box composed all the way up to
the original frame, and `frame_pixels()` uses that. At root level there is
nothing to compose, and it returns the same values as `to_pixels()`.

This is the function to use in rendering code. It returns correct pixel
positions in the original frame regardless of nesting depth.

The frame-level box travels with the bbox, not with the object that holds it,
so it works the same for `DetectedObject`, `PoseObject` and `SegmentedObject`.
It is not shown in `repr()` and does not take part in `==`.

### `in_frame_of(parent)`: compose coordinates

```python
frame_bbox = child_bbox.in_frame_of(parent_bbox)
```

Computes the child's position within the parent's coordinate frame:
```
result.x0 = parent.x0 + child.x0 * parent.width
result.y0 = parent.y0 + child.y0 * parent.height
result.x1 = parent.x0 + child.x1 * parent.width
result.y1 = parent.y0 + child.y1 * parent.height
```

Returns a new BBox with no frame-level box recorded on it, so read it with
`to_pixels()`. Use this for manual coordinate arithmetic (see Line-by-Line
Patterns below).

### `width` / `height` properties

```python
bbox.width   # x1 - x0 (normalized width)
bbox.height  # y1 - y0 (normalized height)
```

---

## Pipeline Patterns (op.seq users)

For regular pipeline users, coordinate composition is automatic.

### Single-level detection

```python
pipeline = op.seq(
    op.color_convert('RGB', 'BGR'),
    op.letterbox(640, 640),
    op.to_tensor(),
    op.load('yolov8n-coco.axm'),
    op.decode_detections(algo='yolov8', num_classes=80),
    op.nms(),
    op.to_image_space(),      # model pixels -> normalized [0,1]
    op.ax_detection(),         # creates BBox with normalized coords
)

detections = pipeline(img)

for det in detections:
    x0, y0, x1, y1 = det.bbox.to_pixels(w, h)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
```

At root level (no `for_each`), `to_pixels()` and `frame_pixels()` return the
same values.

### Two-level cascade (nested detection)

```python
pipeline = op.seq(
    op.color_convert('RGB', 'BGR'),
    # Level 1: Detect vehicles in full frame
    op.letterbox(640, 640),
    op.to_tensor(),
    op.load('yolov8n-coco.axm'),
    op.decode_detections(algo='yolov8', num_classes=80),
    op.nms(),
    op.to_image_space(),
    op.ax_detection(class_id_type=op.CocoClasses),
    # Level 2: For each vehicle, detect plates
    op.for_each(
        'plates',
        op.crop_roi(property='bbox'),
        op.resize(width=640, height=640),
        op.to_tensor(),
        op.load('plate-detector.axm', name='plates'),
        op.decode_detections(algo='yolov8', num_classes=1),
        op.nms(),
        op.to_image_space(),
        op.ax_detection(),
    ),
)
```

What happens automatically:
1. `for_each` hands each vehicle to its body, and remembers the vehicle's frame-level box
2. `crop_roi` crops the vehicle out of the current image
3. `ax_detection` stores each plate's box relative to the crop, and composes it with the vehicle's box into a frame-level box

To render plate detections on the original frame:
```python
result = pipeline(img)
for plates in result.plates:          # one list of plates per vehicle
    for plate_det in plates:
        # frame_pixels() maps back to the original image
        x0, y0, x1, y1 = plate_det.bbox.frame_pixels(w, h)
        cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 1)
```

### Three or more levels

The same pattern extends to arbitrary depth. Each `for_each` / `crop_roi` /
`ax_detection` cycle composes one more level of coordinates, and
`frame_pixels()` always maps back to the original frame.

The other result operators do the same: `ax_pose` also composes its keypoints
(read them with `kp.frame_x(w)` / `kp.frame_y(h)`), `ax_segmentation` composes
its boxes, and `ax_obb` composes its rotated box (read the corners with
`frame_corners(w, h)`).

---

## Line-by-Line Patterns (power users)

If you write detection loops manually instead of using `for_each`, you can
compose coordinates explicitly with `in_frame_of()`.

### Manual two-level cascade

```python
# Level 1
vehicles = detect_pipeline(img)

for vehicle in vehicles:
    # Crop the vehicle region
    vx0, vy0, vx1, vy1 = vehicle.bbox.to_pixels(w, h)
    roi = img[vy0:vy1, vx0:vx1]

    # Level 2: detect plates in the ROI; boxes come back relative to the ROI
    plates = plate_pipeline(roi)

    # Compose coordinates manually
    for plate in plates:
        frame_bbox = plate.bbox.in_frame_of(vehicle.bbox)
        px0, py0, px1, py1 = frame_bbox.to_pixels(w, h)
        cv2.rectangle(frame, (px0, py0), (px1, py1), color, 1)
```

The key difference from pipeline usage: you call `in_frame_of()` explicitly,
and you use `to_pixels()` on the composed result (not `frame_pixels()`, since
nothing records a frame-level box in manual mode).

Call each pipeline on its own image, and not inside an `op.frame_context` you
opened for a different image. A pipeline called inside an existing frame
context reuses it, so `plate_pipeline(roi)` inside `with op.frame_context(img):`
maps the plates against the wrong image.

---

## Migration Guide (IMAGE_PIXEL -> NORMALIZED)

### Before (IMAGE_PIXEL)

```python
# BBox stored pixel integers
det.bbox.x0  # e.g., 320 (pixels)
cv2.rectangle(frame, (int(det.bbox.x0), int(det.bbox.y0)),
              (int(det.bbox.x1), int(det.bbox.y1)), color, 2)
```

### After (NORMALIZED)

```python
# BBox stores normalized [0,1] floats
det.bbox.x0  # e.g., 0.5 (50% of image width)
x0, y0, x1, y1 = det.bbox.frame_pixels(w, h)
cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
```

### Search patterns for code that needs updating

```bash
# Direct pixel access (needs frame_pixels/to_pixels)
grep -rn 'int(.*bbox\.x0\|int(.*bbox\.y0\|int(.*box\.x0' .

# BBox with pixel integer values (needs normalization)
grep -rn 'BBox([0-9][0-9]' .

# Old coordinate format checks
grep -rn 'IMAGE_PIXEL' .
```

### Key changes

| Before | After |
|--------|-------|
| `int(det.bbox.x0)` | `det.bbox.frame_pixels(w, h)[0]` |
| `BBox(100, 200, 300, 400)` | `BBox(0.15, 0.4, 0.45, 0.8)` |
| `CoordSpace.IMAGE_PIXEL` | `CoordSpace.NORMALIZED` |
| Pixel coords in tests | Normalized [0,1] coords in tests |

---

## Checking the result

When you draw a cascade's results on the original frame:

- **Level-2 boxes should appear inside their level-1 parent box.** Boxes piled
  in the top-left corner mean the coordinates were not composed.
- **Keypoints should align with visible body parts.** Dots clustered at (0,0)
  mean the keypoint frame mapping is not being used.
- **Level-2 boxes should not be tiny.** If they are, you are drawing
  crop-relative coordinates with `to_pixels()` where `frame_pixels()` was needed.

[`bottle_demo.py`](../../../examples/pipeline_builder/bottle_demo.py) draws a
cascade's results with `frame_pixels()`.

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
det.bbox.x0  # float in [0, 1] -- left edge as fraction of region width
det.bbox.y0  # float in [0, 1] -- top edge as fraction of region height
det.bbox.x1  # float in [0, 1] -- right edge
det.bbox.y1  # float in [0, 1] -- bottom edge
```

Why normalized? Because normalized coordinates compose via pure arithmetic.
`parent.x0 + child.x0 * parent.width` gives you the child's position in the
parent's coordinate system -- no image dimensions needed until you want pixels.
This is what makes multi-level cascades work without passing image sizes around.

### `to_pixels(w, h)` -- Convert to pixel integers

```python
x0, y0, x1, y1 = bbox.to_pixels(img_width, img_height)
# Returns clamped integers: (int, int, int, int)
```

Multiplies normalized coords by image dimensions and clamps to `[0, w)` / `[0, h)`.
Use this when you have a BBox relative to a known image and need pixel coordinates.

### `frame_pixels(w, h)` -- Frame-level pixel coordinates

```python
x0, y0, x1, y1 = bbox.frame_pixels(img_width, img_height)
```

If `bbox._frame` is set (the pipeline populated frame-level coordinates), uses
`_frame` to compute pixels. Otherwise falls back to `to_pixels()`.

This is the function to use in rendering code. It returns correct pixel
positions in the original frame regardless of nesting depth.

### `in_frame_of(parent)` -- Compose coordinates

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

Returns a new BBox. The `_frame` field is NOT set on the result -- use this
for manual coordinate arithmetic (see Line-by-Line Patterns below).

### `width` / `height` properties

```python
bbox.width   # x1 - x0 (normalized width)
bbox.height  # y1 - y0 (normalized height)
```

### `_frame` -- Pre-computed frame coordinates

```python
bbox._frame  # BBox | None
```

Set automatically by the pipeline (AxDetection, AxPose, AxSegmentation) when
running inside a ForEach. Contains the bbox composed all the way up to the
original frame level. Not visible in `repr()` or `==` comparison.

Why on BBox and not on DetectedObject? Because `_frame` travels WITH the bbox.
If someone replaces `det.bbox`, the old `_frame` detaches naturally. And it
works for all object types (DetectedObject, PoseObject, SegmentedObject) without
modifying each one.

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
    op.to_image_space(),      # MODEL_PIXEL -> NORMALIZED [0,1]
    op.ax_detection(),         # creates BBox with normalized coords
)

with op.frame_context(img):
    detections = pipeline(img)

for det in detections:
    x0, y0, x1, y1 = det.bbox.to_pixels(w, h)
    cv2.rectangle(frame, (x0, y0), (x1, y1), color, 2)
```

At root level (no ForEach), `_frame` is None. `to_pixels()` and `frame_pixels()`
return the same values.

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
        op.resize(size=(640, 640)),
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
1. ForEach sets `_parent_frame_box_var` from the vehicle's `bbox._frame` (or `bbox`)
2. CropRoi reads `_source_image_var` (the vehicle ROI), crops the plate region
3. AxDetection reads `_parent_frame_box_var` and calls `bbox.in_frame_of(parent)`
4. The plate's `bbox._frame` now holds frame-level coordinates

To render plate detections on the original frame:
```python
for plate_det in result.plates:
    # frame_pixels() uses _frame to map back to original image
    x0, y0, x1, y1 = plate_det.bbox.frame_pixels(w, h)
    cv2.rectangle(frame, (x0, y0), (x1, y1), (0, 255, 255), 1)
```

See `rt-demo.py: nested_detection()` for a complete runnable example.

### Three or more levels

The same pattern extends to arbitrary depth. Each ForEach/CropRoi/AxDetection
cycle composes one more level of coordinates. The `_frame` field always holds
the full composition back to the original frame.

---

## Line-by-Line Patterns (power users)

If you write detection loops manually instead of using op.seq, you can compose
coordinates explicitly with `in_frame_of()`.

### Manual two-level cascade

```python
with op.frame_context(img):
    # Level 1
    vehicles = detect_pipeline(img)

    for vehicle in vehicles:
        # Crop the vehicle region
        vx0, vy0, vx1, vy1 = vehicle.bbox.to_pixels(w, h)
        roi = img[vy0:vy1, vx0:vx1]

        # Level 2: detect plates in ROI
        with op.frame_context(roi):
            plates = plate_pipeline(roi)

        # Compose coordinates manually
        for plate in plates:
            frame_bbox = plate.bbox.in_frame_of(vehicle.bbox)
            px0, py0, px1, py1 = frame_bbox.to_pixels(w, h)
            cv2.rectangle(frame, (px0, py0), (px1, py1), color, 1)
```

The key difference from pipeline usage: you call `in_frame_of()` explicitly,
and you use `to_pixels()` on the composed result (not `frame_pixels()`, since
`_frame` is not set in manual mode).

See `rt-demo.py: debug_nested_detection()` for a complete runnable example with
coordinate values printed at each level.

---

## Internals

### ContextVars for nesting state

Two module-level ContextVars in `op/_core.py` carry nesting state:

- `_source_image_var: ContextVar[np.ndarray | None]` -- The image for the current nesting level
- `_parent_frame_box_var: ContextVar[BBox | None]` -- The parent's frame-level bbox

Why ContextVars instead of FrameContext? ForEach is the ONLY operator that
creates nesting, so it should own the nesting state. ContextVars have proper
set/reset semantics with try/finally. FrameContext stays at 182 lines and
unchanged -- it does not need to know about nesting.

### ForEach manages the vars

For each item in the collection, ForEach:
1. Saves the current var values (for restore after iteration)
2. Sets `_source_image_var` to the current image (from var or `fc.input`)
3. Sets `_parent_frame_box_var` to the item's `bbox._frame` (or `bbox`)
4. Runs the inner operators
5. Restores the saved values (via try/finally)

Items without a `bbox` attribute set `_parent_frame_box_var` to None.

### CropRoi reads and updates the var

CropRoi in object-mode:
1. Reads `_source_image_var.get()` for the source image (falls back to `fc.input`)
2. Converts normalized bbox to pixels via `bbox.to_pixels(img_w, img_h)`
3. Crops the region
4. Sets `_source_image_var` to the cropped image (so nested CropRoi gets the right source)

### Result operators populate `_frame`

AxDetection, AxPose, AxSegmentation:
1. Read `_parent_frame_box_var.get()`
2. If set, call `bbox.in_frame_of(parent_frame_box)` and store in `bbox._frame`
3. If None (root level), `_frame` stays None

AxObb (oriented bounding boxes):
1. Read `_parent_frame_box_var.get()`
2. If set, create an axis-aligned BBox from the OBB center/size, compose via
   `in_frame_of()`, and store in `_frame_bbox`
3. `frame_corners(w, h)` maps the rotated corners through `_frame_bbox`

AxPose (keypoints):
1. Read `_parent_frame_box_var.get()`
2. If set, compute frame-level keypoint coords: `frame_x = parent.x0 + kp.x * parent.width`
3. Store in `kp._frame_x` / `kp._frame_y`
4. `kp.frame_x(w)` / `kp.frame_y(h)` return frame-level pixel coords

### Zero overhead for non-cascade pipelines

`ContextVar.get(None)` costs ~20ns. ForEach only sets the vars when it runs.
Non-cascade pipelines never touch the nesting machinery.

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

## Validation

Run these demos to verify nested coordinate composition works for each task type.
All commands use `--no-display --save-dir rtout` so they work headless; check
the output images in `rtout/` to confirm correct rendering.

```bash
source containerless.sh

# 1. Nested detection (det -> det) -- the baseline
python3 rt-demo.py nested_detection data/coco/images/val2017 --no-display --save-dir rtout
# Expect: "Level 1: N detections, Level 2: M sub-detections"
# Check rtout/: green L1 boxes + cyan L2 boxes at correct frame positions

# 2. Nested pose (det -> pose with keypoints)
python3 rt-demo.py nested_pose data/coco/images/val2017 --no-display --save-dir rtout
# Expect: "L1: N detections, L2: M poses, K keypoints"
# Check rtout/: green L1 boxes + yellow L2 pose boxes + cyan keypoint dots at frame positions

# 3. Nested segmentation (det -> instance segmentation)
python3 rt-demo.py nested_segmentation data/coco/images/val2017 --no-display --save-dir rtout
# Expect: "L1: N detections, L2: M segmented objects"
# Check rtout/: green L1 boxes + yellow L2 segmentation boxes at frame positions

# 4. Nested OBB (obb -> det via AABB crop)
python3 rt-demo.py nested_obb data/P0019_0_2304.png --no-display --save-dir rtout
# Expect: "L1: N OBB, L2: M sub-detections"
# Check rtout/: green L1 rotated boxes + yellow L2 axis-aligned boxes
```

What to look for in the output images:
- **L2 boxes should appear INSIDE their L1 parent box** -- if L2 boxes are in the
  wrong position (e.g., top-left corner), the frame coordinate composition is broken
- **Keypoints should align with visible body parts** -- if dots cluster at (0,0),
  the keypoint frame mapping is not working
- **L2 boxes should NOT be tiny** -- if they are, the coordinates may still be in
  ROI-local space instead of frame space

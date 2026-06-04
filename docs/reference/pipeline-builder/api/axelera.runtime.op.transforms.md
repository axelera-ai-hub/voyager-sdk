---
title: "axelera.runtime.op.transforms"
---
# `axelera.runtime.op.transforms`


Image transforms: Resize, CenterCrop, Letterbox, Normalize, etc.

## Summary

| Name | Description |
|------|-------------|
| [Resize](#resize) | Resize an image to specified dimensions or scale. |
| [CenterCrop](#centercrop) | Crop center region of specified size from an image. |
| [CropRoi](#croproi) | Extract a region of interest (ROI) from the input image using a bounding box. |
| [Letterbox](#letterbox) | Resize image to fit target size with padding to maintain aspect ratio. |
| [ColorConvert](#colorconvert) | Convert image to a target color format with optional auto-detection. |
| [ToTensor](#totensor) | Convert image from HWC format to CHW format and normalize to [0, 1] range. |
| [ToImage](#toimage) | Convert image to CHW tensor format without value scaling. |
| [ToDtype](#todtype) | Convert tensor dtype with optional value scaling. |
| [Normalize](#normalize) | Normalize image tensor using mean and standard deviation per channel. |

---

### Resize

Resize an image to specified dimensions or scale.

Takes an np.ndarray image with shape (H, W, C) or (H, W) and returns a
resized np.ndarray.

**Args:**

- **width**: Target width for exact size (must specify both width and height).
- **height**: Target height for exact size (must specify both width and height).
- **size**: Scale smaller edge to this size, preserving aspect ratio (alternative to width/height).
- **half_pixel_centers**: Use half-pixel center alignment (default: False). Currently only True is implemented (uses OpenCV).
- **interpolation**: Interpolation mode - 'nearest', 'bilinear', 'bicubic', 'lanczos' (default: 'bilinear').

**Examples:**

```python
# Fixed size resize
op.resize(width=640, height=480)
# Input: (1080, 1920, 3) -> Output: (480, 640, 3)

# Aspect-preserving resize (smaller edge to size)
op.resize(size=256, half_pixel_centers=True)
# Input: (1080, 1920, 3) -> Output: (256, 455, 3)

# Typical classification preprocessing
op.seq(
    op.resize(size=256, half_pixel_centers=True),  # Resize smaller edge
    op.centercrop(224),                            # Center crop to 224x224
    op.totensor(),
)
```

**Note:** Currently `half_pixel_centers=True` is required (only implemented mode).

**Constructor:**

```python
__init__(width: int = 0, height: int = 0, size: int = 0, half_pixel_centers: bool = False, interpolation: InterpolationMode = InterpolationMode.bilinear)
```

---

### CenterCrop

Crop center region of specified size from an image.

Takes an np.ndarray image with shape (H, W, C) or (H, W) and returns the
center-cropped region with shape (crop_h, crop_w, C) or (crop_h, crop_w).

**Args:**

- **size**: Crop dimensions as int (square), (height, width), or [height, width].

**Examples:**

```python
# Square crop (common for classification models)
op.centercrop(224)  # Crops center 224x224 region
# Input: (256, 256, 3) -> Output: (224, 224, 3)

# Rectangular crop
op.centercrop((224, 320))
# Input: (480, 640, 3) -> Output: (224, 320, 3)

# Typical classification preprocessing
op.seq(
    op.resize(size=256),      # Resize smaller edge to 256
    op.centercrop(224),       # Center crop to 224x224
    op.totensor(),
    op.normalize(...),
)
```

**Raises:**

- **ValueError**: If crop size is larger than image dimensions.

**Constructor:**

```python
__init__(size: int | Sequence[int])
```

---

### CropRoi

Extract a region of interest (ROI) from the input image using a bounding box.

Takes an Object with a bbox property (e.g., DetectedObject, TrackedObject)
or an np.ndarray tensor with indices parameter. Returns the cropped image
region as np.ndarray.

**Args:**

- **property**: Name of the attribute containing the BBox (typically 'bbox').
- **indices**: For tensor mode - tuple of (x0_idx, y0_idx, x1_idx, y1_idx).
- **format**: Coordinate format when using indices (default: XYXY).

**Examples:**

```python
# Extract detected person regions in cascade pipeline
op.foreach(
    'crops',
    op.croproi(property='bbox'),  # Extract each detected object's bbox
    op.resize(size=256),
    op.classify(...),
)
# Input: DetectedObject -> Output: np.ndarray (cropped region)

# Tensor mode for efficient cascade without object wrappers
op.croproi(indices=(0, 1, 2, 3), format=CoordFormat.XYXY)
```

**Raises:**

- **ValueError**: If bbox is invalid (negative dimensions or out of bounds).
- **TypeError**: If used outside a pipeline without frame_context.

**Note:** Object mode (property='bbox') expects bbox coords in PIXEL SPACE
(already mapped to original image coordinates by AxDetection/Tracker).
Tensor mode (indices=...) expects coords in MODEL SPACE and
automatically maps them to original image space using frame context.

**Constructor:**

```python
__init__(property: str = None, indices: tuple[int, ...] = None, format: CoordFormat = CoordFormat.XYXY)
```

---

### Letterbox

Resize image to fit target size with padding to maintain aspect ratio.

Takes an np.ndarray image with shape (H, W, C) or (H, W) and returns a
letterboxed image with shape (height, width, C) or (height, width).

**Args:**

- **width**: Target width (padded dimension).
- **height**: Target height (padded dimension).

**Examples:**

```python
# Standard YOLO preprocessing
op.letterbox(640, 640)
# Input: (1080, 1920, 3) -> Output: (640, 640, 3) with black padding

# Detection pipeline
op.seq(
    op.letterbox(640, 640),
    op.totensor(),
    op.load('yolov8n-coco'),
    op.decode_detections(...),
)
```

**Note:** Letterbox metadata is stored automatically, allowing `to_image_space()` to map bounding boxes back to original image coordinates.

**Constructor:**

```python
__init__(width: int, height: int, fill_color=(114, 114, 114))
```

---

### ColorConvert

Convert image to a target color format with optional auto-detection.

`dst` is required. `src` is optional. Behavior depends on whether `src`
is given and whether the input carries format metadata:

|           | Known format input        | Unknown format input       |
|-----------|---------------------------|----------------------------|
| src=None  | Auto-detect, convert dst  | ERROR: unknown format      |
| src given | Validate src, convert dst | Trust user: src->dst       |

Takes np.ndarray, types.Image, or PIL Image and returns np.ndarray in the
target color format.

**Args:**

- **dst**: Target color format ('BGR', 'RGB', 'GRAY', 'BGRA', 'RGBA').
- **src**: Source color format. If None, auto-detect from input or error.

**Examples:**

```python
# Auto-detect from types.Image (src omitted)
img = types.Image.fromarray(cv2.imread('photo.jpg'), 'BGR')
pipeline = op.seq(
    op.colorconvert('RGB'),   # auto-detects BGR, converts to RGB
    op.letterbox(640, 640),
)

# Explicit conversion with np.ndarray (src given)
img = cv2.imread('photo.jpg')  # BGR ndarray
pipeline = op.seq(
    op.colorconvert('RGB', src='BGR'),  # trust user: BGR->RGB
    op.letterbox(640, 640),
)

# Validation: types.Image + explicit src (must match)
img = types.Image.fromarray(data, 'RGB')
op.colorconvert('RGB', src='BGR')(img)  # ERROR: image is RGB but src says BGR
```

**Note:** Supported conversions include BGRA ↔ RGBA, BGR ↔ BGRA, and RGB ↔ RGBA.

**Constructor:**

```python
__init__(dst: str, src: str | None = None)
```

---

### ToTensor

Convert image from HWC format to CHW format and normalize to [0, 1] range.

DEPRECATED in TorchVision v2: Use op.seq(op.toimage(), op.todtype(scale=True)) instead.
This operator is kept for backwards compatibility.

Takes an np.ndarray image with shape (H, W, C) in uint8 [0-255] range and
returns an np.ndarray tensor with shape (C, H, W) in float32 [0.0-1.0] range.

**Examples:**

```python
# Legacy approach (still works)
op.totensor()

# Modern approach (recommended, aligns with torchvision.transforms.v2)
op.seq(
    op.toimage(),              # Convert to CHW format without scaling
    op.todtype(scale=True),    # Scale to [0.0, 1.0] range
)
```

**Note:** Converts to the CHW format expected by most deep learning frameworks, and scales pixel values from [0, 255] to [0.0, 1.0]. Matches the behavior of the deprecated `torchvision.transforms.v2.ToTensor`.

---

### ToImage

Convert image to CHW tensor format without value scaling.

Modern replacement for ToTensor (along with ToDtype). Takes an np.ndarray
image with shape (H, W, C) in any numeric dtype and returns an np.ndarray
tensor with shape (C, H, W) in the same dtype (no scaling).

**Examples:**

```python
# Modern approach (recommended, matches torchvision.transforms.v2)
op.seq(
    op.toimage(),                    # HWC -> CHW, no scaling
    op.todtype(scale=True),          # Scale to [0.0, 1.0]
    op.normalize(...),
)

# Equivalent to legacy ToTensor
op.seq(op.toimage(), op.todtype(scale=True))  # Same as op.totensor()
```

**Note:** Transposes from HWC to CHW format without scaling values. Use with `ToDtype(scale=True)` to also scale values. Matches `torchvision.transforms.v2.ToImage` behavior.

---

### ToDtype

Convert tensor dtype with optional value scaling.

Modern replacement for ToTensor's scaling behavior (along with ToImage).
Takes an np.ndarray tensor in any dtype and returns an np.ndarray in the
specified dtype, optionally scaled.

**Args:**

- **dtype**: Target numpy dtype (default: np.float32).
- **scale**: If True, scale values based on source dtype: uint8 [0, 255] -> float [0.0, 1.0], int16 [-32768, 32767] -> float [-1.0, 1.0], otherwise convert without scaling.

**Examples:**

```python
# Modern approach with scaling (replaces ToTensor)
op.seq(
    op.toimage(),              # HWC -> CHW
    op.todtype(scale=True),    # uint8 -> float32 with scaling
)

# Convert dtype without scaling
op.todtype(dtype=np.float32, scale=False)

# Full preprocessing pipeline
op.seq(
    op.letterbox(640, 640),
    op.toimage(),                    # HWC -> CHW format
    op.todtype(scale=True),          # Scale to [0.0, 1.0]
    op.normalize(mean=[...], std=[...]),
)
```

**Note:** Applies appropriate scaling to normalize values to standard ranges. Matches `torchvision.transforms.v2.ToDtype` behavior -- `ToDtype(dtype=torch.float32, scale=True)` is the recommended replacement for `ConvertImageDtype`.

**Constructor:**

```python
__init__(dtype: type = np.float32, scale: bool = False)
```

---

### Normalize

Normalize image tensor using mean and standard deviation per channel.

Takes an np.ndarray tensor with shape (C, H, W) in float [0.0-1.0] range
and returns a normalized np.ndarray with shape (C, H, W) where each channel
is computed as (channel - mean) / std.

**Args:**

- **mean**: Tuple of mean values for each channel (e.g., [0.485, 0.456, 0.406]).
- **std**: Tuple of standard deviation values for each channel (e.g., [0.229, 0.224, 0.225]).

**Examples:**

```python
# ImageNet normalization (most common for pretrained models)
op.normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

# Complete preprocessing pipeline
op.seq(
    op.letterbox(640, 640),
    op.totensor(),      # Convert to CHW format and scale to [0, 1]
    op.normalize(       # Normalize using ImageNet statistics
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    op.load('model'),
)
```

**Note:** Each channel is normalized as `(channel - mean) / std`. The `inplace` parameter is not implemented; normalization is always out-of-place.

**Constructor:**

```python
__init__(mean: tuple[float, ...], std: tuple[float, ...])
```

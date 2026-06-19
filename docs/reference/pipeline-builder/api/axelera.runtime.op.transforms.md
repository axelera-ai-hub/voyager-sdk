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
| [CropRotatedRoi](#croprotatedroi) | Extract the rotated region from an OrientedObject via affine warp. |
| [Letterbox](#letterbox) | Resize image to fit target size with padding to maintain aspect ratio. |
| [ColorConvert](#colorconvert) | Convert image to a target color format with optional auto-detection. |
| [ToTensor](#totensor) | Convert image from HWC format to CHW format and normalize to [0, 1] range. |
| [ToImageTensor](#toimagetensor) | Convert image to CHW tensor format without value scaling. |
| [ToDtype](#todtype) | Convert tensor dtype with optional value scaling. |
| [Normalize](#normalize) | Per-channel normalization: `(image - mean) / std`. |
| [LinearScaling](#linearscaling) | Linear scaling: `image / scale + shift` per channel. |
| [ContrastNormalize](#contrastnormalize) | Contrast stretching (min-max normalization) to [0, 1] range. |

---

### Resize

**Alias:** `resize`

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
op.resize(size=256)
# Input: (1080, 1920, 3) -> Output: (256, 455, 3)

# Typical classification preprocessing
op.seq(
    op.resize(size=256),          # Resize smaller edge
    op.center_crop(224),           # Center crop to 224x224
    op.totensor(),
)
```

**Note:**

Specify either (width AND height) OR size, not both.
Currently `half_pixel_centers=True` is required (only implemented mode).

**Constructor:**

```python
__init__(width: int = 0, height: int = 0, size: int = 0, half_pixel_centers: bool = True, interpolation: InterpolationMode = InterpolationMode.bilinear)
```

---

### CenterCrop

**Alias:** `center_crop`

Crop center region of specified size from an image.

Takes an np.ndarray image with shape (H, W, C) or (H, W) and returns the
center-cropped region with shape (crop_h, crop_w, C) or (crop_h, crop_w).

**Args:**

- **size**: Crop dimensions as int (square), (height, width), or [height, width].

**Examples:**

```python
# Square crop (common for classification models)
op.center_crop(224)  # Crops center 224x224 region
# Input: (256, 256, 3) -> Output: (224, 224, 3)

# Rectangular crop
op.center_crop((224, 320))
# Input: (480, 640, 3) -> Output: (224, 320, 3)

# Typical classification preprocessing
op.seq(
    op.resize(size=256),      # Resize smaller edge to 256
    op.center_crop(224),       # Center crop to 224x224
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

**Alias:** `crop_roi`

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
op.for_each(
    'crops',
    op.crop_roi(property='bbox'),  # Extract each detected object's bbox
    op.resize(size=256),
    op.classify(...),
)
# Input: DetectedObject -> Output: np.ndarray (cropped region)

# Tensor mode for efficient cascade without object wrappers
op.crop_roi(indices=(0, 1, 2, 3), format=CoordFormat.XYXY)
```

**Raises:**

- **ValueError**: If bbox is invalid (negative dimensions or out of bounds).
- **TypeError**: If used outside a pipeline without frame_context.

**Note:**

Object mode (property='bbox') expects bbox coords in PIXEL SPACE
(already mapped to original image coordinates by AxDetection/Tracker).
Tensor mode (indices=...) expects coords in MODEL SPACE and
automatically maps them to original image space using frame context.

**Constructor:**

```python
__init__(property: str = None, indices: tuple[int, ...] = None, format: CoordFormat = CoordFormat.XYXY)
```

---

### CropRotatedRoi

**Alias:** `crop_rotated_roi`

Extract the rotated region from an OrientedObject via affine warp.

Uses cv2.warpAffine to rotate the source image so the OBB becomes
axis-aligned, then crops the rectangle. This produces a tighter crop
than the AABB (which CropRoi uses via OrientedObject.bbox).

Use this instead of CropRoi when the second-stage model benefits from
seeing only the object pixels (e.g., OCR on rotated text).

**Examples:**

```python
op.for_each(
    'crops',
    op.crop_rotated_roi(),
    op.resize(640, 640),
    op.totensor(),
    op.load('classifier.axm'),
    ...
)
```

---

### Letterbox

**Alias:** `letterbox`

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

**Note:**

Letterbox metadata is stored automatically, allowing `to_image_space()` to map
bounding boxes back to original image coordinates.

**Constructor:**

```python
__init__(width: int, height: int, fill_color=(114, 114, 114))
```

---

### ColorConvert

**Alias:** `color_convert`

Convert image to a target color format with optional auto-detection.

`dst` is required. `src` is optional. Behavior depends on whether `src`
is given and whether the input carries format metadata:

|           | Known format input        | Unknown format input       |
|-----------|---------------------------|----------------------------|
| src=None  | Auto-detect, convert dst  | ERROR: unknown format      |
| src given | Validate src, convert dst | Trust user: src->dst       |

Takes np.ndarray, types.Image, or PIL Image and returns Image in the target
color format.  Image is returned because it allows the downstream operators
to know the color format.

**Args:**

- **dst**: Target color format (see ColorFormat for valid values.)
- **src**: Source color format. If None, auto-detect from input or error.

**Examples:**

```python
# Auto-detect from Image (src omitted)
img = rt.Image.from_array(cv2.imread('photo.jpg'), 'BGR')
pipeline = op.seq(
    op.color_convert('RGB'),   # auto-detects BGR, converts to RGB
    op.letterbox(640, 640),
)

# Explicit conversion with np.ndarray (src given)
img = cv2.imread('photo.jpg')  # BGR ndarray
pipeline = op.seq(
    op.color_convert('RGB', src='BGR'),  # trust user: BGR->RGB
    op.letterbox(640, 640),
)

# Validation: types.Image + explicit src, the input format overrides src
img = rt.Image.from_array(data, 'RGB')
rgb = op.color_convert('RGB', src='BGR')(img)  # src='BGR' is ignored, no conversion needed
```

**Note:**

The actual conversion is delegated to `Image.convert`; see its docstring
for the full support matrix of which `ColorFormat` pairs can be converted
(or call `img.is_convert_available`). In short: RGB/BGR/RGBA/BGRA/GRAY are
fully interconvertible, YUV formats decode to those but cannot be encoded
to, and PACKED (unknown) input cannot be converted.

**Constructor:**

```python
__init__(dst: ColorFormat | str, src: ColorFormat | str | None = None)
```

---

### ToTensor

**Aliases:** `to_tensor`, `totensor`

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

**Note:**

Converts to the CHW format expected by most deep learning frameworks, and scales
pixel values from [0, 255] to [0.0, 1.0]. Matches the behavior of the deprecated
`torchvision.transforms.v2.ToTensor`.

---

### ToImageTensor

**Aliases:** `to_image_tensor`, `toimage`

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

**Note:**

Transposes from HWC to CHW format without scaling values. Use with
`ToDtype(scale=True)` to also scale values. Matches
`torchvision.transforms.v2.ToImage` behavior.

---

### ToDtype

**Aliases:** `to_dtype`, `todtype`

Convert tensor dtype with optional value scaling.

Modern replacement for ToTensor's scaling behavior (along with ToImageTensor).
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

**Note:**

Applies appropriate scaling to normalize values to standard ranges. Matches
`torchvision.transforms.v2.ToDtype` behavior -- `ToDtype(dtype=torch.float32,
scale=True)` is the recommended replacement for `ConvertImageDtype`.

**Constructor:**

```python
__init__(dtype: type = np.float32, scale: bool = False)
```

---

### Normalize

**Alias:** `normalize`

Per-channel normalization: `(image - mean) / std`.

**Args:**

- **mean**: Per-channel mean (e.g., `[0.485, 0.456, 0.406]`).
- **std**: Per-channel standard deviation (e.g., `[0.229, 0.224, 0.225]`).
- **layout**: ``'CHW'` (default, after `op.totensor()`) or `'HWC'` (for NHWC models that skip `totensor``).

**Examples:**

```python
# CHW pipeline (after totensor)
op.seq(
    op.totensor(),
    op.normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    op.load('model.axm'),
)

# HWC pipeline (NHWC preamble model, no to_tensor, HWC-layout normalize)
op.seq(
    op.letterbox(224, 224),
    op.to_dtype(scale=False),
    op.normalize(mean=[103.939, 116.779, 123.68], std=[1, 1, 1], layout='HWC'),
    op.load('nhwc-preamble.axm'),
)
```

**Constructor:**

```python
__init__(mean: tuple[float, ...], std: tuple[float, ...], layout: TensorLayout = TensorLayout.CHW)
```

---

### LinearScaling

**Alias:** `linear_scaling`

Linear scaling: `image / scale + shift` per channel.

Computes `image / scale + shift`. Commonly used for TensorFlow-style
preprocessing::

    # TF-mode: x / 127.5 - 1  (maps [0, 255] to [-1, 1])
    op.linearscaling(scale=[127.5], shift=[-1.0])

    # Caffe-mode BGR mean subtraction on HWC data
    op.linearscaling(scale=[1, 1, 1], shift=[-103.939, -116.779, -123.68], layout='HWC')

**Args:**

- **scale**: Per-channel divisor.
- **shift**: Per-channel additive bias after division (default: `[0.0]`).
- **layout**: ``'CHW'` (default) or `'HWC'``.

**Constructor:**

```python
__init__(scale: tuple[float, ...], shift: tuple[float, ...] = (0.0,), layout: TensorLayout = TensorLayout.CHW)
```

---

### ContrastNormalize

**Alias:** `contrast_normalize`

Contrast stretching (min-max normalization) to [0, 1] range.

Rescales each image so minimum maps to 0 and maximum maps to 1.
Layout-agnostic. Output is always float32.

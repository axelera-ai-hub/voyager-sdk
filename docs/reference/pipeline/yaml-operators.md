---
title: "YAML Pipeline Operators"
---
# YAML Pipeline Operators

Reference for all pre- and post-processing operators that can be used in the `pipeline:` section of a network YAML file.

These operators handle the CPU-side work: loading inputs, transforming images before the model, and decoding outputs after it. They are distinct from the [GStreamer operators](gst-operators.md), which run in the live-video pipeline.

---

## Input operators

### `input`

Loads the input image.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `color_format` | string | `RGB` | Color format of the source image. One of: `RGB`, `BGR`, `Gray`. |

Setting `color_format` here can eliminate a separate `convert-color` step later — the color conversion is absorbed into the load.

### `input-from-roi`

Loads a region of interest from the input image. Used in cascaded pipelines where the first model outputs a bounding box.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `color_format` | string | `RGB` | Color format of the ROI source image. One of: `RGB`, `BGR`, `Gray`. |

---

## Preprocess operators

### Geometric transformations

#### `centercrop`

Crops the input image from the center. Equivalent to `torchvision.transforms.CenterCrop()`.

#### `letterbox`

Resizes the image while preserving aspect ratio, padding the remainder with a constant value to reach the target dimensions. Used by YOLO models.

#### `resize`

Resizes the input image to specified dimensions.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `width` | integer | — | Target width. |
| `height` | integer | — | Target height. |
| `size` | integer | — | Scale the shorter edge to this size while preserving aspect ratio. Do not combine with `width`/`height`. |
| `half_pixel_centers` | boolean | `false` | Use half-pixel centers (OpenCV backend only). |
| `interpolation` | string | `bilinear` | One of: `nearest`, `bilinear`, `bicubic`, `lanczos`. Falls back with a warning if unsupported. |

---

### Pixel value transformations

#### `contrast-normalize`

Stretches contrast by linearly mapping pixel values to the full available range.

**Formula:** `(input − min) / (max − min)`

Use when images have poor contrast and no fixed input range is known.

#### `linear-scaling`

Applies a linear scale-and-shift transformation.

**Formula:** `output = input × scale + shift`

| Parameter | Type | Description |
|-----------|------|-------------|
| `scale` | float | Multiplicative factor. |
| `shift` | float | Additive offset applied after scaling. |

Common uses:

| Goal | scale | shift |
|------|-------|-------|
| `[0, 255] → [0, 1]` | `0.00392` (1/255) | `0` |
| `[0, 255] → [−1, 1]` | `0.00784` (1/127.5) | `−1` |

#### `normalize`

Standardizes pixel values using per-channel mean and standard deviation.

**Formula:** `output[c] = (input[c] − mean[c]) / std[c]`

| Parameter | Type | Description |
|-----------|------|-------------|
| `mean` | float or list | Per-channel mean. Single value applies to all channels. |
| `std` | float or list | Per-channel standard deviation. |

ImageNet standard values: `mean: [0.485, 0.456, 0.406]`, `std: [0.229, 0.224, 0.225]`

---

### Color space transformations

#### `convert-color`

Converts the color space of the input image. Follows OpenCV `cvtColor` conventions.

| Parameter | Type | Description |
|-----------|------|-------------|
| `conversion` | string | Conversion code, e.g. `RGB2BGR`. |

Supported conversions: `RGB2GRAY`, `GRAY2RGB`, `RGB2BGR`, `BGR2RGB`, `BGR2GRAY`, `GRAY2BGR`.

---

### Tensor conversion

#### `torch-totensor`

Converts the image to a PyTorch tensor, permuting dimensions to NCHW format.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `scale` | boolean | `true` | If `true`, divides pixel values by 255 to normalize to `[0, 1]`. |

Set `scale: false` when you have already scaled values via `linear-scaling` or `normalize` — otherwise you get double-scaling.

---

## Postprocess operators

| Operator | Description |
|----------|-------------|
| `decode-ssd-mobilenet` | Decodes SSD MobileNet detection output. |
| `decodeyolo` | Decodes YOLO detection output. |
| `topk` | Returns the top-K class indices and scores from a classification output. |
| Multi-object tracker (SORT / OC-SORT / ByteTrack) | Associates detections across frames and assigns track IDs. |

---

## Mapping training transforms to YAML operators

When deploying a model, the YAML preprocessing must exactly replicate the transforms used during training. Here are the most common patterns:

### Pattern 1 — PyTorch `ToTensor()`

```python
transforms.ToTensor()
# or manually: img = img / 255.0; img = np.transpose(img, (2,0,1))
```

```yaml
- torch-totensor:
    scale: true   # default, can be omitted
```

---

### Pattern 2 — Scale to `[−1, 1]`

```python
img = img / 127.5 - 1
img = np.transpose(img, (2, 0, 1))
```

```yaml
- linear-scaling:
    scale: 0.00784   # 1/127.5
    shift: -1
- torch-totensor:
    scale: false     # values already scaled — do NOT divide by 255 again
```

---

### Pattern 3 — ImageNet normalization (`ToTensor` + `Normalize`)

```python
transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
```

```yaml
- torch-totensor:
    scale: true
- normalize:
    mean: [0.485, 0.456, 0.406]
    std: [0.229, 0.224, 0.225]
```

---

### Pattern 4 — Offset scaling (`−127.5` then `× 1/128`)

```python
img -= 127.5
img *= 0.0078125   # 1/128
img = np.transpose(img, (2, 0, 1))
```

```yaml
- normalize:
    mean: 127.5
    std: 128
- torch-totensor:
    scale: false   # values already in final range
```

---

## Key rules

- `torch-totensor` always permutes to NCHW — place it last in the preprocess chain.
- `torch-totensor` with `scale: true` divides by 255. If you apply `linear-scaling` or `normalize` first in integer space, set `scale: false`.
- `normalize` expects values in `[0, 1]` when used after `torch-totensor` with `scale: true`. When used before `torch-totensor`, ensure `scale: false`.
- Use `convert-color` when the model expects a different channel order than your input source (e.g. camera outputs BGR, model expects RGB).

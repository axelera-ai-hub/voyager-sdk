---
title: "GStreamer Operators"
---
# GStreamer Operators

The Voyager SDK ships a set of built-in GStreamer operators that cover the most common pre-processing, post-processing, and utility tasks. These operators appear in the pipeline YAML files inside `ax_models/` and can be combined to build custom pipelines.

See [Pipelines — How Inference Works](pipeline-basics.md) for an overview of how operators fit into a pipeline.

---

## How operators are specified

Each operator entry in a pipeline YAML has three fields:

```yaml
- instance: axtransform          # GStreamer element type
  lib: libtransform_resize.so    # shared library implementing the operator
  options: width:640;height:640  # semicolon-separated key:value pairs
```

Lists within options use commas: `mean:0.485,0.456,0.406`

---

## Pre-processing operators

These operators prepare each video frame before it reaches the AI model.

### transform_resize

Resizes a video frame. Supports letterboxing (preserving aspect ratio with padding) or stretching.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `width` | int | — | Output width in pixels |
| `height` | int | — | Output height in pixels |
| `size` | int | — | Alternative to width/height: scales the shorter edge to this value |
| `letterbox` | int | `1` | `1` = preserve aspect ratio with padding; `0` = stretch to fit |
| `padding` | int | — | Pixel fill value for letterbox padding (e.g. `114` for gray) |
| `to_tensor` | int | `0` | `1` = output as NHWC tensor instead of video frame (requires RGBA input) |

```yaml
- instance: axtransform
  lib: libtransform_resize.so
  options: width:640;height:640;padding:114;letterbox:1
```

---

### transform_resizeratiocropexcess

Resizes while keeping aspect ratio, then crops excess. Common for classification models that expect a square input.

| Option | Type | Description |
|--------|------|-------------|
| `resize_size` | int | Resize so the shorter edge equals this value |
| `final_size_after_crop` | int | Optional: center-crop to this square size after resize |

```yaml
- instance: axtransform
  lib: libtransform_resizeratiocropexcess.so
  options: resize_size:256;final_size_after_crop:224
```

---

### transform_cropresize

Crops a region of interest (ROI) from the frame and resizes it. Used in cascaded pipelines where a detector's output feeds a classifier.

| Option | Type | Description |
|--------|------|-------------|
| `meta_key` | string | Key in the metadata map containing bounding box ROIs |
| `width` | int | Output width |
| `height` | int | Output height |
| `respect_aspectratio` | int | `1` = preserve aspect ratio and crop excess; `0` = stretch |

```yaml
- instance: axtransform
  lib: libtransform_cropresize.so
  options: meta_key:detections;width:224;height:224;respect_aspectratio:1
```

---

### transform_totensor

Converts a video frame to a tensor for model input.

| Option | Type | Description |
|--------|------|-------------|
| `type` | string | `int8` — NHWC format, copies uint8 values; `float32` — NCHW format, normalizes to [0, 1] |

```yaml
- instance: axtransform
  lib: libtransform_totensor.so
  options: type:int8
```

---

### inplace_normalize

Normalizes a tensor by applying mean/std scaling and optional quantization. Used to prepare float inputs for int8 quantized models.

Formula: `x = (x - mean) / std`, then optionally quantize.

| Option | Type | Description |
|--------|------|-------------|
| `mean` | float list | Per-channel mean values (comma-separated) |
| `std` | float list | Per-channel standard deviation values (comma-separated) |
| `quant_scale` | float | Quantization scale parameter |
| `quant_zeropoint` | float | Quantization zero-point parameter |
| `simd` | string | SIMD acceleration: `avx2` or `avx512` (int8 only) |

```yaml
- instance: axinplace
  lib: libinplace_normalize.so
  mode: write
  options: mean:0.485,0.456,0.406;std:0.229,0.224,0.225;quant_scale:0.01863;quant_zeropoint:-14
```

---

### transform_dequantize

Converts a quantized int8 tensor to float32. Formula: `y = scale × (x − zero_point)`.

| Option | Type | Description |
|--------|------|-------------|
| `dequant_scale` | float list | Dequantisation scale per tensor (comma-separated) |
| `dequant_zeropoint` | int list | Dequantisation zero-point per tensor (comma-separated) |
| `transpose` | int | `1` = transpose NHWC → NCHW; `0` = leave as-is |

```yaml
- instance: axtransform
  lib: libtransform_dequantize.so
  options: dequant_scale:0.1304;dequant_zeropoint:-70
```

---

### transform_padding

Adds padding to a tensor. Padding values are available in `manifest.json` after model compilation.

| Option | Type | Description |
|--------|------|-------------|
| `padding` | int list | Padding before and after each dimension, comma-separated (length = 2 × number of dimensions) |
| `fill` | int | Fill value for padded bytes (typically the quantization zero-point) |
| `input_shape` | int list | Optional reshape before padding |
| `output_shape` | int list | Optional reshape after padding |

```yaml
- instance: axtransform
  lib: libtransform_padding.so
  options: padding:0,0,0,0,0,8,0,0;fill:114
```

---

### transform_yolopreproc

Executes the first layer of a YOLO network on the host CPU before sending to the AIPU. Reshapes 2×2 pixel patches into single pixels with 4× channels (e.g. 640×640×3 → 320×320×12), then adds required tensor padding.

| Option | Type | Description |
|--------|------|-------------|
| `padding` | int list | Padding at beginning and end of each dimension |
| `fill` | int | Fill value for padded bytes |

```yaml
- instance: axtransform
  lib: libtransform_yolopreproc.so
  options: padding:0,0,0,0,0,0,0,52;fill:114
```

---

## Post-processing operators

These operators convert raw model output tensors into structured results (bounding boxes, class labels, etc.).

### decode_yolov5

Decodes YOLOv5 and YOLOv7/v8-compatible model output into object detection bounding boxes. Handles dequantisation, sigmoid activation, and anchor-based decoding.

| Option | Type | Description |
|--------|------|-------------|
| `meta_key` | string | Key in the metadata map to store detection results |
| `anchors` | float list | Anchor values from `model_info.json` |
| `classes` | int | Number of object classes the model detects |
| `confidence_threshold` | float | Minimum confidence to keep a detection (e.g. `0.25`) |
| `topk` | int | Maximum number of boxes before NMS |
| `multiclass` | int | `1` = consider all classes per box; `0` = top class only |
| `sigmoid_in_postprocess` | int | `1` = apply sigmoid here; `0` = model already applied it |
| `transpose` | int | `1` = transpose NHWC → NCHW |
| `scales` | float list | Dequantisation scales from `manifest.json` |
| `zero_points` | int list | Dequantisation zero-points from `manifest.json` |
| `label_filter` | int list | Keep only these class IDs (optional) |

```yaml
- instance: decode_muxer
  lib: libdecode_yolov5.so
  options: meta_key:detections;classes:80;confidence_threshold:0.25;topk:100;
    multiclass:0;sigmoid_in_postprocess:1;transpose:1;
    anchors:1.25,1.625,2.0,3.75,4.125,2.875;
    scales:0.0814,0.0950,0.0929;zero_points:70,82,66
```

---

### decode_ssd2

Decodes SSD (Single Shot MultiBox Detector) model output. Handles sigmoid/softmax activation, anchor generation, dequantisation.

| Option | Type | Description |
|--------|------|-------------|
| `meta_key` | string | Key in the metadata map to store detection results |
| `classes` | int | Number of object classes |
| `confidence_threshold` | float | Minimum confidence to keep a detection |
| `topk` | int | Maximum number of boxes |
| `class_agnostic` | int | `1` = NMS across all classes; `0` = per-class NMS |
| `transpose` | int | `1` = transpose NHWC → NCHW |
| `softmax` | int | `1` = apply softmax; `0` = apply sigmoid |
| `scales` | float list | Dequantisation scales from `manifest.json` |
| `zero_points` | int list | Dequantisation zero-points from `manifest.json` |
| `saved_anchors` | string | Path to anchor file, or omit to auto-generate |
| `label_filter` | int list | Keep only these class IDs (optional) |

```yaml
- instance: decode_muxer
  lib: libdecode_ssd2.so
  options: meta_key:detections;classes:90;confidence_threshold:0.4;
    topk:1000;class_agnostic:1;transpose:1;scales:0.9;zero_points:0
```

---

### decode_classification

Decodes classifier model output into top-K class predictions.

| Option | Type | Description |
|--------|------|-------------|
| `meta_key` | string | Key in the metadata map to store classification results |
| `classlabels_file` | string | Path to a text file with one label per line |
| `top_k` | int | Number of top predictions to keep |
| `sorted` | int | `1` = sort results by confidence |
| `largest` | int | `1` = highest-confidence label first |
| `softmax` | int | `1` = apply softmax before selecting top-K |
| `box_meta` | string | If set, attach classification to existing detection boxes instead of creating new metadata |

```yaml
- instance: decode_muxer
  lib: libdecode_classification.so
  options: meta_key:classification;classlabels_file:labels.txt;top_k:5;softmax:0
```

---

### inplace_nms

Non-Maximum Suppression. Removes duplicate bounding boxes, keeping only the most confident detection for each object.

| Option | Type | Description |
|--------|------|-------------|
| `meta_key` | string | Key in the metadata map containing detections to filter |
| `max_boxes` | int | Maximum number of boxes to keep after NMS |
| `nms_threshold` | float | IoU threshold above which overlapping boxes are suppressed (e.g. `0.45`) |
| `class_agnostic` | int | `1` = suppress across all classes; `0` = suppress per class |
| `location` | string | `CPU` or `GPU` (OpenCL) |

```yaml
- instance: axinplace
  lib: libinplace_nms.so
  options: meta_key:detections;max_boxes:300;nms_threshold:0.45;class_agnostic:0;location:CPU
```

---

## Utility operators

### inplace_draw

Draws inference results (bounding boxes, labels, classification overlays) onto the video frame. Iterates over all metadata entries and calls each one's draw function.

No options required.

```yaml
- instance: axinplace
  lib: libinplace_draw.so
  mode: write
```

---

### inplace_tracker

Assigns persistent tracking IDs to detected objects across frames. Supports multiple tracking algorithms.

| Option | Type | Description |
|--------|------|-------------|
| `algorithm` | string | `sort`, `oc-sort`, `bytetrack`, or `scalarmot` |
| `detection_meta_key` | string | Key in metadata map for input detections |
| `tracking_meta_key` | string | Key in metadata map to store tracker output |
| `streamid_meta_key` | string | Key for stream ID in multi-stream applications |
| `history_length` | int | Number of time steps to retain per tracked object |
| `algo_params_json` | string | Path to JSON file with additional algorithm parameters |

```yaml
- instance: axinplace
  lib: libinplace_tracker.so
  options: detection_meta_key:detections;tracking_meta_key:tracks;algorithm:sort
```

---

### inplace_addstreamid

Adds a stream ID to frame metadata. Required in multi-stream pipelines where multiple input streams are merged before inference.

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `stream_id` | int | — | The ID to assign to this stream |
| `meta_key` | string | `stream_id` | Key in the metadata map |

```yaml
- instance: axinplace
  lib: libinplace_addstreamid.so
  mode: meta
  options: stream_id:0
```

---

## See also

- [Pipelines — How Inference Works](pipeline-basics.md) — pipeline concepts and structure
- [Video Sources](../../tutorials/video-sources.md) — configuring input sources
- [First Inference](../../user-guides/first-inference.md) — run a complete pipeline
- [Model Zoo](../models/model-zoo.md) — pre-built models with ready-to-use pipelines

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

Additional operators: Filter, SortBy, TopK, Top1, Softmax, ProtoToMask, Split.

## Summary

| Name | Description |
|------|-------------|
| [DecodeDetections](#decodedetections) | Parse raw detection model output into a standardized bounding box array. |
| [DecodeLPRNet](#decodelprnet) | Decode LPRNet logits/token IDs into license plate text. |
| [DecodeObb](#decodeobb) | Parse raw OBB (Oriented Bounding Box) model output into a standardized array. |
| [DecodePose](#decodepose) | Parse raw pose model output into a structured array with bounding boxes and keypoints. |
| [DecodeRetinaFace](#decoderetinaface) | Decode RetinaFace 3-head output into an `op.Nms`-ready array. |
| [DecodeSegmentation](#decodesegmentation) | Parse raw segmentation model output into detection and prototype arrays. |
| [DecodeSemanticSegmentation](#decodesemanticsegmentation) | Reduce dense per-class segmentation logits to a 2D class-id map. |
| [DecodeYolact](#decodeyolact) | Decode YOLACT's four head tensors into scored candidates plus prototypes. |
| [MatchGallery](#matchgallery) | Match an embedding against a gallery of named reference embeddings. |
| [Nms](#nms) | Non-maximum suppression to remove duplicate overlapping detections. |
| [ToImageSpace](#toimagespace) | Map coordinates from model input space to normalized [0,1] image-relative values. |
| [ProtoToMask](#prototomask) | Compute per-detection binary masks from mask coefficients and prototype features. |
| [Filter](#filter) | Filter detections based on class IDs, score, or custom function. |
| [Split](#split) | Split detections into two lists: matching and non-matching. |
| [SortBy](#sortby) | Order a list of objects by a named numeric attribute; optionally take top-n. |
| [TopK](#topk) | Return the k largest/smallest elements along a given dimension. |
| [Top1](#top1) | Convenience operator returning only the top-1 element (k=1). |
| [Softmax](#softmax) | Apply softmax to convert logits to probabilities. |

---

### DecodeDetections

**Alias:** `decode_detections`

Parse raw detection model output into a standardized bounding box array.

Takes the raw tensor from a detection model (yolov5, yolov7, yolox, yolov8,
yolov9, yolo11, yolov10 or yolo26) and converts it into a clean (N, 6)
array: [x0, y0, x1, y1, score, class_id]. This is the first postprocessing
step after model inference.

**Args:**

- **algo**: Detection algorithm. It selects the decode form: 'yolov5', 'yolov7' and 'yolox' carry an objectness column at index 4, 'yolov8', 'yolov9' and 'yolo11' carry per-class scores from index 4, and 'yolov10' and 'yolo26' emit already suppressed [x0, y0, x1, y1, score, class_id] rows.
- **num_classes**: Number of classes (required). Together with algo it fixes the column count the input must have.
- **confidence_threshold**: Minimum confidence score (default: 0.25).
- **max_boxes_pre_nms**: Maximum boxes to keep before NMS (default: 30000). No effect on the NMS-free form, whose rows the head has already capped.
- **use_multi_label**: If True, emit one detection per (anchor, class) pair whose combined score exceeds `confidence_threshold` (same box repeated across qualifying classes); class-aware NMS resolves overlap. Enable for eval to match the Ultralytics COCO-mAP convention. Default False (argmax over classes -- single label per anchor).

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

**Note:**

Filters boxes below the confidence threshold before returning results.
For an algo in the supported set, an input whose width contradicts the
form that algo declares raises ValueError, rather than being decoded as
whatever form that width happens to match. An unrecognised algo is not
validated: it warns and falls back to sniffing the width.

**Constructor:**

```python
__init__(algo: str, num_classes: int, confidence_threshold: float = field(default=0.25, metadata={'range': (0.0, 1.0)}), max_boxes_pre_nms: int = field(default=30000, metadata={'range': (1, None)}), use_multi_label: bool = False, input_format: str = 'auto')
```

---

### DecodeLPRNet

**Aliases:** `decode_lpr_net`, `decode_lprnet`

Decode LPRNet logits/token IDs into license plate text.

Performs greedy CTC decoding with optional repeat collapsing and blank
suppression. Supports common LPRNet layouts:
- (B, C, T)
- (B, T, C)
- (C, T)
- (T, C)

**Args:**

- **charset**: Ordered character set (without blank token).
- **blank_index**: Index of the CTC blank token.
- **collapse_repeats**: Collapse consecutive repeated symbols.
- **remove_blank**: Drop blank tokens from decoded text.

**Returns:** list[str], one decoded plate string per sample.

**Constructor:**

```python
__init__(charset: str | list[str], blank_index: int | None, collapse_repeats: bool, remove_blank: bool)
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
    op.nms(iou_threshold=0.45, box_format='xywhr', backend='opencv'),
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

### DecodeRetinaFace

**Alias:** `decode_retina_face`

Decode RetinaFace 3-head output into an `op.Nms`-ready array.

Accepts an iterable of tensors identified by last-dim shape, in any order:
`loc` has last dim 4, `conf` has last dim 2, `landmarks` has last dim
10.

Emits `(N, 6)` -- `x0, y0, x1, y1, score, class` -- and with
`with_landmarks=True` appends the model's five landmark points, giving
`(N, 16)`: `..., class, kx0, ky0, kx1, ky1, ... kx4, ky4`. Upstream
Pytorch_Retinaface concatenates its decoded boxes, score and ten landmark
values into one row the same way; the class column at index 5 is what the
rest of the pipeline reads, since `op.Nms` slices it as the class on its
opencv and torch backends and `op.AxDetection` and `op.AxPose` both
expect the box/score/class prefix. RetinaFace detects one class, so that
column is always 0, exactly as it is for single-class pose.

The five points are the two eye centres, the nose tip and the two mouth
corners, which is what face alignment consumes. They carry no visibility
channel, hence two columns per point rather than pose's three.

Landmarks are off by default: columns past the box and score are
pass-through data for `op.Nms`, but `op.AxDetection` reads a row wider
than 6 as one score per class, so a pipeline ending there wants the plain
`(N, 6)`. Pair `with_landmarks=True` with `op.AxPose(num_keypoints=5,
keypoint_stride=2)`.

Boxes and landmarks alike come out in MODEL_PIXEL space, which is
`op.Nms`'s input contract.

The default `cfg` covers both the ResNet-50 and MobileNet-0.25 variants
of the upstream Pytorch_Retinaface repository -- they share the anchor
configuration; only `model_width` / `model_height` differ.

**Args:**

- **model_width**: Width of model input (default 640 for MobileNet; ResNet-50 uses 840).
- **model_height**: Height of model input.
- **confidence_threshold**: Drop boxes whose face-class probability is below this. Legacy default 0.2.
- **cfg**: Anchor configuration. Defaults to the upstream Pytorch_Retinaface `{min_sizes, steps, variance, clip}` matching both backbones.
- **with_landmarks**: Append the five decoded landmark points, widening each row from 6 to 16 columns. Requires the landmark tensor.

**Examples:**

```python
op.seq(
    op.color_convert('BGR'),
    op.resize(width=640, height=640),
    # NOT op.to_tensor() -- that would /255 the pixels into [0, 1] and
    # then subtracting ~104 below would saturate the input.  RetinaFace
    # was trained on raw BGR pixel values minus an ImageNet-BGR mean,
    # so keep pixels in [0, 255] through normalize().
    op.toimage(),  # HWC -> CHW, no scaling
    op.todtype(dtype=np.float32),
    op.normalize(mean=[104, 117, 123], std=[1, 1, 1]),
    op.load('retinaface-mobilenet0.25-widerface-onnx.axm'),
    op.DecodeRetinaFace(model_width=640, model_height=640),
    op.Nms(iou_threshold=0.4, class_agnostic=True),
    op.ToImageSpace(),
    op.AxDetection(class_id_type=op.FaceClasses),
)

# Carrying the five landmark points through as well
op.seq(
    ...,
    op.DecodeRetinaFace(model_width=640, model_height=640, with_landmarks=True),
    op.Nms(iou_threshold=0.4, class_agnostic=True),
    op.ToImageSpace(keypoint_cols=range(6, 16, 2)),
    op.AxPose(num_keypoints=5, keypoint_stride=2, class_id_type=op.FaceClasses),
)
```

**Constructor:**

```python
__init__(model_width: int = 640, model_height: int = 640, confidence_threshold: float = field(default=0.2, metadata={'range': (0.0, 1.0)}), cfg: dict = field(default_factory=lambda: dict(_RETINAFACE_DEFAULT_CFG)), with_landmarks: bool = False)
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
__init__(algo: str, num_classes: int, num_mask_coeffs: int = 32, confidence_threshold: float = field(default=0.25, metadata={'range': (0.0, 1.0)}), max_boxes_pre_nms: int = field(default=30000, metadata={'range': (1, None)}), input_format: str = 'auto', use_multi_label: bool = False)
```

---

### DecodeSemanticSegmentation

**Alias:** `decode_semantic_segmentation`

Reduce dense per-class segmentation logits to a 2D class-id map.

Takes a single model output of per-class scores/logits in NCHW layout
(`(N, C, H, W)` or `(C, H, W)`) and applies argmax over the class axis,
yielding an `(N, H, W)` / `(H, W)` class-id map that
ax_semantic_segmentation() then wraps into a SemanticSegmentation result.

This is the semantic-segmentation counterpart of decode_detections /
decode_segmentation: it owns the model-specific decode step (argmax),
keeping ax_semantic_segmentation a pure result wrapper. Use it for models
(e.g. Ultralytics YOLO*-sem) whose .axm emits raw class logits rather than
a pre-argmaxed class map.

**Args:**

- **num_classes**: Length of the class axis; validated against the channel dimension so a mis-wired tensor fails loudly instead of silently producing garbage.

Some models argmax inside their own graph and emit a 1-channel class map rather
than logits. This op drops that unit class axis instead of argmaxing again, so it
is safe to keep in the pipeline either way. A model with a single
genuine class is not treated as one: its logits also carry a unit class axis and
must still be argmaxed. Drop the op only if you want raw logits downstream.

**Examples:**

```python
op.seq(
    op.color_convert('RGB'),
    op.resize(width=1024, height=1024),
    op.to_tensor(),
    op.load('yolo26n-sem-cityscapes-logits.axm'),     # -> (1, num_classes, H, W) logits
    op.decode_semantic_segmentation(num_classes=19),  # -> (1, H, W) class map
    op.ax_semantic_segmentation(),                    # -> SemanticSegmentation
)
```

An export that reduces the class axis in its own graph loads as
`(1, H, W)` uint8 instead, and this op squeezes that to `(H, W)` rather
than argmaxing it. `yolo26n-sem.axm` is such a build.

**Constructor:**

```python
__init__(num_classes: int)
```

---

### DecodeYolact

**Alias:** `decode_yolact`

Decode YOLACT's four head tensors into scored candidates plus prototypes.

YOLACT is an SSD-style one-stage instance segmentation network: its heads
emit box offsets against fixed anchor priors, class logits and per-anchor
mask coefficients, while a separate branch emits a small basis of prototype
mask channels. This operator does the part that is specific to YOLACT in a native
C++ kernel (prior box decode with variances, softmax with the background class
dropped, and the confidence prefilter), then hands the shared postprocess chain
`(candidates, protos)`. Suppression is `op.nms(algo='fast')` and mask
assembly is `op.proto_to_mask()`, exactly as for the other instance
segmentation models: neither is specific to YOLACT, so neither lives here.

The four tensors may arrive in any order, since runtime unpacks a model's
multi-output tuple into separate arguments. They are identified by shape:
`loc` has last dim 4, `conf` has last dim `num_classes + 1`, the mask
coefficients have last dim `num_mask_coeffs`, and the prototypes are the
3D `(H, W, num_mask_coeffs)` tensor. Prototypes are emitted transposed to
the `(num_mask_coeffs, H, W)` layout `op.proto_to_mask` expects.

`candidates` carries one row per (surviving anchor, class) pair in the
shared `[x0, y0, x1, y1, score, class, coefficients...]` layout, boxes in
MODEL_PIXEL space. Fast NMS ranks and suppresses within a class, so the pair
expansion is what upstream's `(num_classes, N)` score matrix expresses.

**Args:**

- **model_width**: Width of the model input. Boxes are emitted in MODEL_PIXEL space so `op.to_image_space()` can map them onto the frame.
- **model_height**: Height of the model input.
- **num_classes**: Foreground classes; the conf head carries one more for background.
- **num_mask_coeffs**: Number of prototype channels (upstream `mask_dim`).
- **confidence_threshold**: Drop an anchor whose best foreground probability does not exceed this (upstream `nms_conf_thresh`).
- **cfg**: Anchor configuration. The default covers the published YOLACT ResNet-50, ResNet-101 and DarkNet-53 configs, which share it; only `model_width` / `model_height` change with input resolution.

**Examples:**

```python
op.seq(
    op.color_convert('RGB'),
    # YOLACT stretch-resizes to a square, it does not letterbox.
    op.resize(width=550, height=550),
    op.to_tensor(),
    op.normalize(mean=[123.68, 116.78, 103.94], std=[58.40, 57.12, 57.38]),
    op.load('yolact-resnet50-coco.axm'),
    op.decode_yolact(model_width=550, model_height=550),
    op.par(
        op.seq(op.itemgetter(0), op.nms(algo='fast', iou_threshold=0.5, max_boxes=100)),
        op.itemgetter(1),
    ),
    op.par(
        op.seq(op.pack(), op.itemgetter(0)),
        op.proto_to_mask(encoding='binary', crop_bounds='ceil_padded'),
    ),
    op.par(
        op.seq(op.pack(), op.itemgetter(0), op.to_image_space()),
        op.seq(op.pack(), op.itemgetter(1)),
    ),
    op.ax_segmentation(class_id_type=op.CocoClasses),
)
```

**Note:**

Greedy NMS would silently diverge from the reference: YOLACT specifies
fast NMS, which is non-transitive, hence `algo='fast'` above. Masks
come out in prototype space, so `op.to_image_space()` applies to the
detections only.

**Constructor:**

```python
__init__(model_width: int = 550, model_height: int = 550, num_classes: int = 80, num_mask_coeffs: int = 32, confidence_threshold: float = field(default=0.05, metadata={'range': (0.0, 1.0)}), cfg: dict = field(default_factory=lambda: dict(_YOLACT_DEFAULT_CFG)))
```

---

### MatchGallery

**Alias:** `match_gallery`

Match an embedding against a gallery of named reference embeddings.

Computes cosine similarity between the input embedding and every gallery
entry, picks the maximum, and emits `Recognition(name, score)`. If
`score < threshold` the returned name is `'Unknown'` but `score`
still carries the actual top similarity for downstream inspection.

The gallery can be supplied either as an in-memory `dict[str,
Sequence[float]]` (preferred when the caller already loaded / built it)
or as a `path` to a JSON file with the same shape -- the JSON path is
loaded lazily on first call and cached.

**Args:**

- **gallery**: Mapping of `name -> embedding vector`. The vectors are normalized on first use; passing pre-normalized vectors is fine.
- **path**: Path to a JSON file holding the same mapping. Used only when `gallery` is empty; loaded lazily.
- **threshold**: Cosine-similarity cutoff for a positive match. Default 0.352 mirrors the legacy face-recognition tutorial's threshold.

**Examples:**

```python
gallery = {'alice': [...], 'bob': [...]}
op.match_gallery(gallery=gallery, threshold=0.352)
```

**Constructor:**

```python
__init__(gallery: dict = field(default_factory=dict), path: str = '', threshold: float = field(default=0.352, metadata={'range': (-1.0, 1.0)}))
```

---

### Nms

**Alias:** `nms`

Non-maximum suppression to remove duplicate overlapping detections.

Takes an np.ndarray (N, M) where columns 0:4 are boxes, column 4 is
the score, and columns 5+ are pass-through data. Returns the same format
with duplicate/overlapping boxes removed.

**Args:**

- **iou_threshold**: IOU threshold for suppression (default: 0.45). Higher = more boxes kept, lower = more aggressive suppression. The boundary follows each metric's own upstream reference, so it is not the same on both: iou_metric='iou' keeps a box whose overlap is exactly the threshold, matching torchvision and cv2.dnn.NMSBoxes, and iou_metric='probiou' suppresses it, matching ultralytics.
- **class_agnostic**: If False, apply NMS per-class (default: False). If True, apply NMS across all classes together.
- **max_boxes**: Maximum boxes to return after NMS (default: 300).
- **algo**: Suppression algorithm - 'greedy' or 'fast'. Left unset it resolves to what the box format is measured against: 'fast' on oriented boxes under iou_metric='probiou', 'greedy' everywhere else. The field keeps what you passed, including None; `effective_algo` reports what the resolution settled on, and is what the suppression actually runs. 'greedy' takes the best box left and drops what it overlaps, repeatedly, so a dropped box suppresses nothing itself. 'fast' judges every overlap against the original per-class ranking, so a box can be dropped by one that was itself dropped. These are different algorithms, not approximations of each other, and the models that specify 'fast' (YOLACT and its descendants) are evaluated against it. The oriented default follows ultralytics, which is what these models are measured against: at 8.4.121 rotated NMS is `TorchNMS.fast_nms(..., iou_func=batch_probiou)`. 'greedy' is what the GStreamer NMS element runs and stays available by asking for it; it is also the quicker of the two on oriented boxes, since it skips the pairs of a dropped box. The axis-aligned default stays 'greedy', matching torchvision and `cv2.dnn.NMSBoxes`.
- **top_k**: Candidates ranked per class before that class is compared (default: 0, meaning all). algo='fast' on box_format='xyxy' only; any other combination rejects it.
- **min_score**: Drop detections scoring at or below this before suppression (default: 0.0, no filtering). The cut is exclusive, matching the reference decoders' `score > threshold`, where `op.filter(min_score=...)` is inclusive. Where it runs cannot change the result: a suppressor always outscores its victim, so cutting low scores can never remove a survivor's suppressor, and both this cut and the max_boxes cap take a prefix of the same score order.
- **backend**: NMS implementation - 'native', 'opencv' or 'torch' (default: 'native'). 'native' is a C++ kernel running the same algorithms as the GStreamer NMS element; the other two are kept for cross-checking it.
- **box_format**: Box format - 'xyxy' (default) or 'xywhr' for OBB. When 'xywhr', suppression is rotation-aware and reads the angle from column 6.
- **iou_metric**: How overlap is measured - 'iou' (default) or 'probiou'. 'iou' is the exact intersection over union of the two shapes; for box_format='xywhr' that means rotated-rectangle polygon intersection, which only backend='opencv' provides. 'probiou' models each oriented box as a 2D Gaussian and measures the Bhattacharyya distance between the two, which the native kernel computes; it requires box_format='xywhr'. These are different measures, so iou_threshold does not carry across between them -- a threshold tuned against one has to be retuned for the other.

**Examples:**

```python
# Standard detection pipeline
op.nms(iou_threshold=0.45, max_boxes=300)

# OBB (oriented bounding box) pipeline, exact rotated-rectangle IoU
op.nms(iou_threshold=0.45, box_format='xywhr', backend='opencv')

# OBB with probabilistic IoU
op.nms(iou_threshold=0.45, box_format='xywhr', iou_metric='probiou')

# Instance segmentation, one row per (anchor, class) pair
op.nms(algo='fast', iou_threshold=0.5, top_k=200, max_boxes=100)
```

**Note:**

Extra columns beyond the box and score are passed through unchanged. Works in any
coordinate space -- typically used in MODEL_PIXEL space before
`to_image_space()` conversion.

**Constructor:**

```python
__init__(iou_threshold: float = field(default=0.45, metadata={'range': (0.0, 1.0)}), class_agnostic: bool = False, max_boxes: int = field(default=300, metadata={'range': (1, None)}), algo: str | None = None, top_k: int = field(default=0, metadata={'range': (0, None)}), min_score: float = field(default=0.0, metadata={'range': (0.0, 1.0)}), backend: str = 'native', box_format: str = 'xyxy', iou_metric: str = 'iou')
```

**Properties:**

- **`effective_algo`** (`str`) — The algorithm this NMS actually runs.

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

Instance segmentation models (e.g., YOLOv8-seg) output two tensors: per-
detection mask coefficients (one 32-element vector per detection) and a
prototype feature map `(32, H, W)`. This operator computes
`coefficients @ prototypes.reshape(32, H*W)`, snaps each bbox into
prototype space, crops the mask to that bbox, and encodes it as uint8 via
`clip(raw_logit * 255, 0, 255)`. The downstream evaluator kernel binarizes
with `> 0` (effective threshold `raw_logit > 1/255 ~= 0.004`), the
display renderer treats the same uint8 array as an alpha-like intensity.

Takes `(detections, protos)` where `detections` is an `(M, 38)`
array and `protos` is a `(32, H, W)` prototype tensor. Returns a
list of `M` uint8 mask arrays, each cropped to its detection's bbox
in prototype space.

Must be called BEFORE `to_image_space()` because mask cropping relies
on MODEL_PIXEL bounding box coordinates to index into the prototype grid.

**Args:**

- **num_mask_coeffs**: Number of mask coefficients per detection (default: 32).
- **encoding**: How the raw logit becomes uint8 - 'alpha' (default) or 'binary'. 'alpha' emits `clip(logit * 255, 0, 255)`, so the array doubles as a renderer alpha channel and binarizes downstream at `logit > 1/255`. 'binary' emits 255 where `logit > 0` and 0 elsewhere, which is the exact threshold a model specifying `sigmoid(logit) > 0.5` asks for, since the two are the same test.
- **crop_bounds**: Which prototype cells the box covers - 'round' (default) or 'ceil_padded'. 'round' takes the nearest cell to each edge, the rounding a downstream evaluator repeats when it derives the canvas placement rectangle. 'ceil_padded' takes `[ceil(edge - 1), ceil(edge + 1))`, which is the rectangle the YOLACT family's own `crop` retains and is two cells wider per axis; a YOLACT model cropped by 'round' emits masks that do not match its reference.

**Examples:**

```python
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

Mask cropping operates in prototype space. Call this BEFORE
`to_image_space()` transforms coordinates.

**Constructor:**

```python
__init__(num_mask_coeffs: int = 32, encoding: str = 'alpha', crop_bounds: str = 'round', model_width: int = 0, model_height: int = 0)
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

### SortBy

**Alias:** `sort_by`

Order a list of objects by a named numeric attribute; optionally take top-n.

Serializable equivalent of `sorted(objs, key=attrgetter(by), reverse=descending)[:top]`.

**Args:**

- **by**: Attribute name to sort by (e.g. `'score'`, `'area'`, `'center_distance'`). Any numeric attribute or property on the objects. `getattr` raises loudly on a bad name instead of silently returning 0.0. `'center_distance'` uses `BBox.center_distance` — the Euclidean distance from the bbox centre to the image centre (0.5, 0.5) in normalised coords. Use `descending=False` to get the closest-to-centre objects first.
- **descending**: If True (default), sort largest-first.
- **top**: If set, return only the first `top` elements after sorting.

**Examples:**

```python
op.seq(
    ...,
    op.ax_pose(),
    op.sort_by(by='area', top=5),                               # top-5 by bbox area
    op.sort_by(by='center_distance', descending=False, top=10), # top-10 closest to centre
    op.for_each('rois', ...),
)
```

**Constructor:**

```python
__init__(by: str, descending: bool = True, top: int | None = None)
```

---

### TopK

**Alias:** `top_k`

Return the k largest/smallest elements along a given dimension.

Follows PyTorch's torch.topk API. For np.ndarray input, returns a plain
(values, indices) tuple; spread it into the next op with op.unpack(). For
list[Classification] input, returns the top-k Classification objects.

**Args:**

- **k**: Number of top elements to return.
- **dim**: Dimension to rank along (default -1 = last dim, like torch.topk).
- **largest**: If True, return k largest; if False, k smallest (default: True).
- **sorted**: Whether to return elements in sorted order (default: True).

**Examples:**

```python
# op.unpack spreads top_k's (values, indices) into ax_classification's args
op.seq(
    op.load('model'),
    op.top_k(k=5),                    # -> (values, indices)
    op.unpack(),
    op.ax_classification(...),        # -> list[Classification]
)
```

**Constructor:**

```python
__init__(k: int = field(metadata={'range': (1, None)}), dim: int = -1, largest: bool = True, sorted: bool = True)
```

---

### Top1

**Alias:** `top1`

Convenience operator returning only the top-1 element (k=1).

**Constructor:**

```python
__init__(largest: bool = True, *, name: str = '')
```

---

### Softmax

**Alias:** `softmax`

Apply softmax to convert logits to probabilities.

**Args:**

- **axis**: Axis along which softmax is computed (default: -1).

**Examples:**

```python
op.seq(
    op.load('classifier.axm'),
    op.softmax(),
    op.ax_classification(class_id_type=op.ImagenetClasses),
    op.top_k(k=5),
)
```

**Constructor:**

```python
__init__(axis: int = -1)
```

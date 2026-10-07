# Postambles: running a model's tail natively

When you compile a model, almost all of it runs on the AIPU. The model-specific
arithmetic at the very end, which turns the device's raw numbers into the tensors
the source model declares, stays on the host: box decoding for a detector, a
softmax for a classifier. The compiler leaves that piece as a small graph inside
the `.axm`, and `op.load` runs it on onnxruntime (or PyTorch, for a `.pt2`
sidecar). That graph is the model's **postamble**, the mirror of the *preamble*
in front of the network. [Model Formats](model-formats.md) explains how the two
differ from the preprocessing and postprocessing you write yourself.

The onnxruntime tail is always correct, and it is often the slowest part of an
otherwise native pipeline: it needs an onnxruntime session per model, a call into
it on every frame, and full float tensors materialised to feed it. A **postamble
kernel** is a native implementation of that tail, and you name one when you load
the model.

This page covers Pipeline Builder. If you drive the low-level `axelera.runtime`
C-runtime API yourself, running the postamble graph by hand is covered in
[Post-Processing with ONNX Postamble Graphs](../../tutorials/axruntime-python/postamble-processing.md).

## Use a built-in kernel

Pass a `Postamble` to `op.load`:

```python
from axelera.runtime import Postamble, op

pipeline = op.seq(
    preprocess,
    op.load(
        'yolov8n-coco.axm',
        using_postamble=Postamble(
            fused_kernel='UltralyticsYoloPostamble', num_classes=80, dfl_bins=16
        ),
    ),
    op.decode_detections(...),
)
```

Everything besides the slot and the kernel name is the kernel's own parameter,
passed to its constructor. Every built-in kernel except `RetinaFacePostamble` and
`YolactPostamble` requires `num_classes`, because a wrong class count mis-decodes
every frame without an error.

A postamble is a runtime choice: the same `.axm` works with or without one, and
you can add, change or remove it without recompiling. Leave it out while you
bring a model up, and keep the onnxruntime tail as the reference you check a
kernel against.

The class reference, including how a kernel is named and what an empty
`Postamble()` means, is [`axelera.runtime.postamble`](api/axelera.runtime.postamble.md).
The `using_postamble` argument is documented on
[`op.load`](api/axelera.runtime.op.inference.md).

## `kernel=` or `fused_kernel=`

The slot decides what your kernel is handed:

| slot | the kernel receives | it replaces |
|---|---|---|
| `fused_kernel=` | the AIPU's own outputs: padded, quantized, NHWC, exactly as the device wrote them | the postamble graph **and** the depad, dequantize and align steps in front of it |
| `kernel=` | float32 tensors, already unpadded, dequantized and reshaped to the source model's output shapes | the postamble graph only |

`fused_kernel=` is faster, because no full float tensor is built first: the
kernel does the depad and dequantize itself, in the same pass as the decode.
`kernel=` is easier, because you work on ordinary float tensors in the shapes the
original model declared. Filling both slots raises `ValueError`.

A kernel is written against one slot. Every built-in kernel is a
`fused_kernel=`, and naming one in the other slot is refused:

```
ValueError: UltralyticsYoloPostamble must be declared as fused_kernel=, not kernel=.
```

## What a postamble may do

A postamble restores the tensors the source model declares. The compiler cuts the
graph where the device stops, which leaves depad, dequantize, transpose,
concatenate and whatever activation a head ends in (YOLACT's mask `tanh`, a
classifier's softmax) to run on the host. Replacing that with a kernel changes how
those tensors are computed, never what they are.

It is not a place for postprocessing. `op.load` carries the source model's
signature, so a compiled model stays substitutable for the ONNX or PyTorch model it
was exported from, and a numerics check, an accuracy run or a golden output keeps
comparing like with like. A kernel that also decodes changes the tensors `op.load`
emits, and those comparisons stop holding. Decode, suppression and mask assembly
stay in the operator chain after `op.load`.

When you time a kernel, compare it with the whole path it replaces, not with the
graph alone. For a `fused_kernel=` that includes the depad, dequantize and align
steps: on YOLACT at 550x550 the graph took 0.98 ms and the steps in front of it
7.71 ms, so a kernel timed against the graph alone is judged on an eighth of its
job.

## The built-in kernels

All are `fused_kernel=`. `op.load` always supplies the output descriptors and the
model's input size; the parameters below are the ones you write.

| kernel | for | required | optional |
|---|---|---|---|
| `UltralyticsYoloPostamble` | split-head YOLO: YOLOv8/v11 detect, pose, segment, OBB; YOLOv9/GELAN and YOLOv6; anchor-free YOLOv5u/v7u re-exports; YOLOX (`has_objectness=True`) | `num_classes` | `dfl_bins`, `num_keypoints`, `num_mask_coeffs`, `has_obb`, `has_objectness`, `tensor_roles`, `backend` |
| `AnchoredYoloPostamble` | anchor-based YOLOv5/v7 (upstream exports, three anchors per level) | `num_classes`, `anchors` | `backend` |
| `YoloNmsFreePostamble` | NMS-free YOLO (YOLO26, YOLOv10) | `num_classes` | `dfl_bins`, `num_keypoints`, `num_mask_coeffs`, `has_obb`, `max_detections`, `tensor_roles`, `backend` |
| `SsdPostamble` | SSD-MobileNet-style heads: a loc/cls tensor pair per FPN level, priors generated from the tensor shapes | `num_classes` | `min_scale`, `max_scale`, `aspect_ratios`, `reduce_lowest_layer`, `interpolated_scale_aspect_ratio`, `variance_xy`, `variance_wh` |
| `RetinaFacePostamble` | RetinaFace: a box, class and landmark head per FPN level, concatenated across levels with the class pair softmaxed; the deltas stay undecoded for `op.decode_retina_face` | | `backend` |
| `SemanticSegmentationPostamble` | semantic segmentation: one dense output resized to the model resolution, emitted as per-class logits or as the uint8 class map an in-graph `ArgMax` leaves | `num_classes` | `output`, `backend` |
| `ClassificationPostamble` | classifiers whose tail is a softmax | `num_classes` | |
| `YolactPostamble` | YOLACT and its descendants | | `num_classes`, `num_mask_coeffs`, `num_anchors`, `backend` |

Choosing between the YOLO kernels:

- **Split head or packed head.** `UltralyticsYoloPostamble` needs a split head:
  per FPN level, a box tensor of `4 * dfl_bins` channels and a separate
  class-score tensor of `num_classes` channels, found by channel count. That is
  why it also covers YOLOv9/GELAN (`dfl_bins=16`) and YOLOv6 (`dfl_bins=17`). An
  upstream YOLOv5 or YOLOv7 export packs `num_anchors * (5 + num_classes)`
  channels into one tensor per level, so it needs `AnchoredYoloPostamble` and the
  pixel `(w, h)` anchor pairs from the model's own configuration. Naming the wrong
  kernel fails in the constructor, because the channel counts contradict the
  parameters.
- **`tensor_roles`**: one role per output, in output order (`'box'`, `'class'`,
  `'keypoint'`, `'mask'`, `'proto'`, `'angle'`, `'objectness'`). You need it only
  when two outputs have the same width and the kernel says it cannot tell them
  apart.
- **`backend`**: leave it at `'auto'`. `AnchoredYoloPostamble` accepts it but
  decodes scalar whichever you name, and logs a warning for a vector backend.

Notes on the others:

- `SsdPostamble` takes the prior-generation *rule* rather than the priors: the
  TF `MultiscaleGridAnchorGenerator` parameters above, with the TF Object
  Detection API defaults. It emits a pair, boxes as normalized unclamped
  `y1, x1, y2, x2` and per-class sigmoid scores without the background class,
  exactly as the model's own postprocess graph does. `op.DecodeSsd` turns the
  pair into detections.
- `ClassificationPostamble` emits probabilities, so the pipeline after it must
  not softmax again. Softmax twice flattens the distribution: top-1 survives, the
  confidences are wrong, and nothing is logged.

Every recipe in [`model_recipes/`](../../../model_recipes/README.md) that has a
kernel declares it as `POSTAMBLE`, so the recipes are the worked examples:
[`detection/yolov8n_coco_axrelay.py`](../../../model_recipes/detection/yolov8n_coco_axrelay.py),
[`detection/yolov5s_coco_axrelay.py`](../../../model_recipes/detection/yolov5s_coco_axrelay.py) (`anchors`),
[`detection/yolox_s_coco_axrelay.py`](../../../model_recipes/detection/yolox_s_coco_axrelay.py) (`has_objectness=True`),
[`pose/yolo11n_pose_coco_axrelay.py`](../../../model_recipes/pose/yolo11n_pose_coco_axrelay.py),
[`instance_segmentation/yolo11n_seg_coco_axrelay.py`](../../../model_recipes/instance_segmentation/yolo11n_seg_coco_axrelay.py),
[`classification/densenet121_imagenet_axrelay.py`](../../../model_recipes/classification/densenet121_imagenet_axrelay.py).

### Selected for you: models exported by Ultralytics

An `.axm` exported by the Ultralytics frontend (`yolo export format=axelera`)
carries its task metadata, and `.optimized()` picks the matching kernel from it
when you passed no `Postamble`. Models built by the Model Zoo carry no such
metadata, so nothing is picked for them. To keep the onnxruntime tail on an
Ultralytics export, pass an empty `Postamble()`.

## Check what ran

List the operators of the loaded model:

```python
>>> [type(o).__name__ for o in op.load(AXM).operators]
['NchwToNhwc', 'Quant', 'Pad', 'AxRuntimeModel', 'Depad', 'Dequant',
 '_AlignToModelOutput', '_ListUnpack', 'OnnxModel']

>>> p = Postamble(fused_kernel='UltralyticsYoloPostamble', num_classes=80)
>>> [type(o).__name__ for o in op.load(AXM, using_postamble=p).operators]
['NchwToNhwc', 'Quant', 'Pad', 'AxRuntimeModel', '_ListUnpack', 'UltralyticsYoloPostamble']
```

The first is the default: depad, dequantize, align, then the graph. In the second,
the fused kernel has replaced all four. `op.load` places a declared kernel before
any optimization, and `.optimized()` leaves a declared tail alone.

## In a recipe or a Model Zoo config

A recipe declares its kernel once, as a module-level `POSTAMBLE`, and passes it to
`op.load` in both its demo and its accuracy pipeline; see
[Model recipes](../../../model_recipes/README.md#postamble). An `axzoo` config
declares the same object as a module-level `postamble = Postamble(...)` next to
`preprocess`, and the build writes it into the generated recipe as `POSTAMBLE`.

## Where the built-in kernels live

The built-in kernels are ordinary `Operator` classes in the installed package. The
module is `mega`, reached through `axelera.runtime`:

```bash
python -c "from axelera.runtime import mega; print(mega.__file__)"
```

The file it prints is readable Python. For a Model Zoo model, `axzoo info <model>`
prints the model's `Postamble` and the file that defines its kernel in one step.
Each kernel class there shows its constructor contract, the parameter validation,
and the refusals: what it checks against the tensors it is given, and the errors it
raises. `ClassificationPostamble` is pure NumPy from end to end and is the shortest
complete example. The other kernels call into compiled extension modules for their
per-frame loop, and the C++ source of those modules is not distributed.

## Write your own kernel

### The contract

`op.load` builds every kernel the same way:

```python
kernel_cls(output_infos, model_width, model_height, **kwargs)
```

- `output_infos`: one `axelera.runtime.TensorInfo` per model output, describing
  what your slot receives. Each has `shape`, `dtype`, `padding` (a
  `(before, after)` pair per axis), `scale` and `zero_point`. Read the geometry
  from here instead of hard-coding it.
  - For `fused_kernel=`: the device tensors. `scale` and `zero_point` are arrays
    when the output is quantized per channel.
  - For `kernel=`: the source model's output shapes, float32, no padding.
- `model_width`, `model_height`: the model's **input** size. For a model with a
  preamble this is the size in front of the preamble, not the AIPU's input (YOLOX
  reports 640, not 320), which is what coordinate decoding needs.
- `**kwargs`: the parameters on the `Postamble`. Take the ones you need
  explicitly, and do not infer them from channel counts: `num_classes=64` and a
  64-channel DFL box tensor look the same.

`__call__` receives one positional NumPy array per model output, in output order:
`def __call__(self, x)` for a single-output model, `def __call__(self, *tensors)`
for several. It returns what the postamble graph returns, meaning the source
model's output tensors, with the same shapes and dtypes. For one output, return
the array. For several, the built-in kernels return them as separate positional
values for the next operator, which a custom kernel cannot do through a public
type; return a tuple instead, and put `op.unpack()` right after `op.load(...)` in
your pipeline.

### A NumPy kernel

A `kernel=` kernel that applies a softmax to a single-output classifier:

```python
# my_kernels.py
import numpy as np
from axelera.runtime import TensorInfo, op


class SoftmaxTail(op.Operator):
    postamble_slot = 'kernel'

    infos: list[TensorInfo]
    model_width: int
    model_height: int
    axis: int

    def __init__(self, infos, model_width, model_height, *, axis=-1, name=''):
        if len(infos) != 1:
            raise ValueError(f"SoftmaxTail reads one output, this model has {len(infos)}")
        self.infos = infos
        self.model_width = model_width
        self.model_height = model_height
        self.axis = axis
        super().__init__(name=name)

    def __call__(self, x):
        e = np.exp(x - x.max(axis=self.axis, keepdims=True))
        return e / e.sum(axis=self.axis, keepdims=True)
```

```python
from axelera.runtime import Postamble, op

model = op.load('my-model.axm', using_postamble=Postamble(kernel='my_kernels:SoftmaxTail'))
```

What each part is for:

- `postamble_slot` names the slot the kernel is written for, so `op.load` refuses
  the other slot by name. Leave it out and the kernel is not checked.
- The class-level annotations are the kernel's fields. They are what the printed
  pipeline shows. Keep the constructor keyword-only after the three positional
  arguments, and call `super().__init__(name=name)`.
- Every `Operator` subclass is registered by its snake_case class name when the
  class is defined, so the name must not collide with an existing operator.

### A C++ kernel

To run the per-frame loop in C++, build your own Python extension module (for
example with pybind11 or nanobind) and call it from the kernel's `__call__`. The
`Operator` class stays in Python: it holds the parameters, validates them against
`output_infos` in its constructor, and passes the arrays to your extension.

```python
import numpy as np
from axelera.runtime import TensorInfo, op

import my_decode_ext  # your compiled module


class MyFusedKernel(op.Operator):
    postamble_slot = 'fused_kernel'

    infos: list[TensorInfo]
    model_width: int
    model_height: int
    num_classes: int

    def __init__(self, infos, model_width, model_height, *, num_classes, name=''):
        # validate num_classes against infos here, then precompute what the loop needs
        self.infos = infos
        self.model_width = model_width
        self.model_height = model_height
        self.num_classes = num_classes
        super().__init__(name=name)

    def __call__(self, *tensors):
        return my_decode_ext.decode(
            [np.ascontiguousarray(t) for t in tensors],
            [(i.shape, i.padding, float(i.scale), int(i.zero_point)) for i in self.infos],
            self.num_classes,
        )
```

The constraints that matter:

- Your extension receives NumPy arrays: for `fused_kernel=`, the padded quantized
  NHWC buffers that `output_infos` describes; for `kernel=`, float32 in the source
  model's shapes. Pass a contiguous array if your code assumes one.
- It must return NumPy arrays in the shapes and dtypes the postamble graph would
  have produced, following the return rule above.
- The `"module:ClassName"` string is imported when `op.load` runs, so the module
  and your extension must be importable in that process (installed, or on
  `sys.path`).
- A pipeline saved with `save_axe()` stores the `Postamble` itself, the kernel
  name and its parameters, and `op.load` builds the kernel again when the `.axe`
  is loaded. So keep the parameters to plain values (numbers, strings, booleans,
  lists, dicts), and make the kernel module available wherever the `.axe` runs.
- A class passed to `Postamble` directly must be defined at module scope in an
  importable module: it is stored as its import path, so one defined in `__main__`
  or inside a function is refused.

### Refuse what you cannot do

Check the tensors you are given against what the kernel supports, in the
constructor, and raise with a message that says what to do instead. For example, a
kernel that dequantizes with one scale per tensor must refuse a per-channel output
rather than use the first element of the scale array:

```python
for info in infos:
    if np.size(info.scale) != 1 or np.size(info.zero_point) != 1:
        raise ValueError(
            "needs a per-tensor output quantization; drop the Postamble to run the onnx tail"
        )
```

Per-channel output quantization occurs in real models (a ViT classifier head can
carry 1000 scales). Refusing costs the user only the speed-up, because the default
tail dequantizes per channel correctly.

## Verify a kernel

Write the unfused arithmetic in plain NumPy and compare both on the same buffer.
This needs no device:

```python
import numpy as np
from axelera.runtime import TensorInfo, mega

# 1000 classes padded out to 1024, as the device writes them.
info = TensorInfo(
    (1, 1, 1, 1024),
    np.int8,
    padding=[(0, 0), (0, 0), (0, 0), (0, 24)],
    scale=0.125,
    zero_point=-37,
)


def reference(raw, info, num_classes):
    """depad, dequantize, reshape, softmax"""
    x = raw[tuple(slice(b, -e if e else None) for b, e in info.padding)]
    x = (x.astype(np.float32) - info.zero_point) * info.scale
    x = x.reshape(1, num_classes)
    e = np.exp(x - x.max(axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


raw = np.random.default_rng(0).integers(-128, 128, size=info.shape, dtype=np.int8)
got = mega.ClassificationPostamble([info], 224, 224, num_classes=1000)(raw)

np.testing.assert_allclose(got, reference(raw, info, 1000), rtol=1e-5, atol=1e-6)
```

On hardware, run the same input through `op.load(AXM)`, which uses the
onnxruntime tail, and through `op.load(AXM, using_postamble=...)`, and compare the
outputs. Test the refusals as well, since those are what users will hit.

## When it goes wrong

| symptom | cause |
|---|---|
| `Postamble takes kernel= or fused_kernel=, not both` | fill exactly one slot |
| `must be declared as fused_kernel=, not kernel=` | the kernel is written for the other slot |
| `is not a built-in kernel` | the bare name is misspelled; the message lists the valid ones |
| `did not resolve to a class` | the `"module:ClassName"` string is wrong, or the module is not importable when `op.load` runs |
| `cannot be named importably` | the class passed to `Postamble` is defined in `__main__` or inside a function |
| `Type name '...' is already registered` | your kernel class has the same name as an existing operator; rename it |
| `missing 1 required keyword-only argument: 'num_classes'` | the kernel requires a declared class count |
| `num_classes=... contradicts the model` | the declared count disagrees with the output tensor's unpadded channels |
| `No output tensor carries ... channels` | the declared `num_classes` or `dfl_bins` matches no output tensor's channel count |
| `cannot be told apart by width` | two outputs have the same width; declare `tensor_roles` |
| `needs a per-tensor output quantization` | the output is quantized per channel; drop the `Postamble` to run the onnxruntime tail |
| `TypeError: __call__() takes 2 positional arguments but 10 were given` | a multi-output model and a single-argument `__call__`; take `*tensors` |
| the next operator receives a tuple | a multi-output custom kernel returned a tuple; add `op.unpack()` after `op.load(...)` |
| confidences look flat, but the ranking is right | softmax applied both in the kernel and in the pipeline |
| accuracy a few points low, no error | a step that is not order-preserving is applied twice, most often a normalization or a color conversion in both the preprocessing and the pipeline |

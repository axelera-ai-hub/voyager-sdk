---
title: Model Formats
---

# Model Formats: `.axm` and `.axe`

Two file formats carry a model to the AIPU. An `.axm` is the compiled model on
its own. An `.axe` is a whole pipeline, the model included, in one file.

| | `.axm` | `.axe` |
|---|---|---|
| Holds | one compiled model | preprocessing, model, postprocessing, metadata |
| Produced by | the compiler (see [Model Compilation](../../tutorials/pipeline-builder/model-compilation.md)) | `pipeline.save_axe(...)` |
| Loaded by | `op.load('model.axm')` | `op.load('pipeline.axe')` |
| You still write | the pre and postprocessing around it | nothing, it is ready to call |
| Runs on | the AIPU | the AIPU |

Both go through the same `op.load()`, which picks the format from the file. It
takes these two and nothing else: for an ONNX model use `op.onnx_model()`, and
for a `.pt2` sidecar use `op.pt2_model()`.

> [!NOTE]
> Older documentation calls the compiled model an `.axmodel`. The compiler
> writes `.axm`, and that is the name used throughout the Pipeline Builder.

## `.axm`, the compiled model

**An `.axm` takes and returns what the source model did.** That is the property
the rest of this page rests on: the pipeline you write around an `.axm` is the
pipeline you would write around the ONNX or PyTorch model it was compiled from,
so a pipeline that was correct before compilation stays correct after it.

Holding that costs the runtime real work, all of it inside `op.load()`. The
chip is fed padded int8 in its own layout and answers in kind, so the loader
quantizes and pads on the way in, then depads, dequantizes, and restores each
output to the model's logical shape on the way out. It also runs the parts of
the network the compiler could not put on the AIPU. What reaches you is
unpadded float32 in the source model's shapes.

> [!NOTE]
> One case breaks the guarantee. An `.axm` compiled before the manifest
> recorded the original output shapes cannot be realigned, and `op.load()` says
> so: `Output shapes may not match ONNX. Re-compile the model to fix this.`
> Recompile rather than reshaping by hand downstream.

What comes out is the model's output, not decoded results, so you supply the
stages around it:

```python
pipeline = op.seq(
    op.color_convert('RGB'),
    op.letterbox(640, 640),
    op.to_tensor(),
    op.load('yolov8n-coco.axm'),
    op.decode_detections(algo='yolov8', num_classes=80),
    op.nms(iou_threshold=0.45, max_boxes=300, name='nms'),
    op.to_image_space(),
    op.ax_detection(class_id_type=op.CocoClasses),
)
```

An `.axm` is also accepted as a directory of the same content, which is the only
form the native runtime loads on Windows.

### Preamble and postamble: the parts that stayed on the host

Two words appear in the runtime's messages that are easy to read as something
they are not.

> A **preamble** and a **postamble** are pieces of *the model itself* that the
> compiler could not place on the AIPU, so it left them as small graphs inside
> the `.axm` to run on the host: the preamble in front of the network, the
> postamble after it. A typical postamble is the model's own decode or final
> activation.

They are not your preprocessing and postprocessing. Those are operators **you**
write around the model, in your pipeline; the preamble and postamble are inside
the `.axm`, put there by the compiler, and run whether you asked for them or
not. Which is why the shape guarantee above holds: between them and the
depad/dequantize step, an `.axm` still answers like the model it came from.

**What runs them.** By default the postamble graph runs on **onnxruntime**, or
on PyTorch when the compiler left a `.pt2` sidecar. You can name a native kernel
to run in its place when you load the model, without recompiling:
[Postambles](postamble.md) is the guide, and
[`Postamble`](api/axelera.runtime.postamble.md) the class reference.

**Or leave them out entirely.** If you cut the graph yourself before compiling,
so the trailing decode is not part of the model you hand the compiler, the
`.axm` has no postamble at all and that work becomes ordinary operators in your
own postprocessing. That is a legitimate design, and sometimes the faster one.
Which split wins depends on the model and the host, so it is worth measuring
rather than assuming.

## `.axe`, the whole pipeline

`save_axe()` writes that pipeline, the `.axm` inside it, and the metadata into
one file:

```python
pipeline.save_axe('yolov8n-coco.axe', name='YOLOv8n COCO', description='...')
detector = op.load('yolov8n-coco.axe')
detections = detector(image)
```

It is a ZIP holding a `pipeline.toml` manifest plus every model file the
manifest names:

```
yolov8n-coco.axe
|-- pipeline.toml      metadata, model files, the operator sequence
`-- yolov8n-coco.axm   every model the pipeline loads, embedded
```

`op.load()` returns the pipeline ready to call. It is still a pipeline, so a
named operator can be retuned after loading:

```python
detector = op.load('yolov8n-coco.axe')
detector.node(name='nms').max_boxes = 100
```

`info()` reads the metadata back: the name, description, task, schema version,
and the input shape and layout the model declares.

A runnable end to end example of the whole round trip is
[`examples/pipeline_builder/axe.py`](../../../examples/pipeline_builder/axe.py): it saves a detection pipeline, loads it
back, reads the metadata, retunes a named operator and runs it.

## Which one to use

Use an `.axm` while you are building the pipeline, and whenever the pipeline
code lives with the application that runs it.

Use an `.axe` to hand a finished pipeline to someone else, or to another
machine, without also handing over the code that built it. Three things follow
from the whole pipeline being in the file rather than only the model:

- **It runs the pipeline you tested**, not one reconstructed from a README. The
  operator sequence and every parameter travel with it.
- **It can be optimized as a whole.** Fusion is not confined to the model: the
  preprocessing chain folds into the model's own quantize and pad step, and the
  model's output tail folds into the decode. A receiver holding only an `.axm`
  gets whatever fusion the code they wrote around it happens to allow.
- **It says what it needs.** The manifest records the model files, the input
  shape, the task, and any operator the reader has to import. The options given
  to `op.load` (`using_postamble`, `core_allocation`, `device_config`) are
  recorded too, and apply again when the package is loaded.

Because the second point is done on load, not on save, an `.axe` is stored
unoptimized on purpose. The next section says why, and what to call.

## Two rules the `.axe` format follows

**It holds the pipeline as it was built.** `pipeline.optimized()` picks fused
kernels for the machine it runs on. An OpenCL kernel chosen on one host falls
back to software on a host with no OpenCL device, rather than the kernel that
host would have picked for itself, and `.optimized()` cannot be undone. So
`save_axe()` refuses an optimized pipeline, and the machine that will run it
optimizes it on load. **This call is not optional:** without it the pipeline
runs the unfused operator sequence the file stores.

```python
detector = op.load('yolov8n-coco.axe').optimized()
```

Retuning a named operator still works afterwards, so the order is yours.

**A field is written only when it was read.** The input shape and layout come
from the model in the package; a pipeline with no model states no shape, rather
than a default. The task comes from the pipeline's own `ax_<task>` operator, and
is empty when the pipeline does not say.

## Limitations

- **A custom operator needs its module on the far side.** Your own operator is
  saved like any other, and the manifest records which module defines it, but
  the operator registry is per process: the process that loads the file must
  import that module first. `save_axe()` warns and names them at save time, and
  the load fails with the module name rather than an unknown operator type. An
  anonymous operator (`op.operator(lambda x: x)`) cannot be saved at all.
- **Operator names are global.** Two operators cannot register the same name in
  one process; the second raises at import.
- **One name per model file.** Two models whose paths differ but whose file
  names match cannot be packaged together, because a pipeline refers to a model
  by that name. `save_axe()` refuses and names both paths.
- **A `.pt2` sidecar is not embedded.** A pipeline holding an `op.pt2_model`
  saves without complaint and the manifest describes it, but the file itself is
  not put in the archive, so the package only loads where that file already is.
- **Not byte-reproducible.** The manifest records when it was written, so saving
  the same pipeline twice produces different bytes. Identify a package by the
  model inside it, not by a hash of the file.
- **Every load extracts.** Loading writes the embedded models to a temporary
  directory, which lives as long as the loaded pipeline. Loading one package N
  times costs N copies on disk.
- **Schema versions.** The `.axe` schema is at version 1.0. A file whose major
  version is newer than the runtime refuses to load; a newer minor version warns
  and continues.

## See also

- [Model Compilation](../../tutorials/pipeline-builder/model-compilation.md) for producing an `.axm`.
- [Pipeline Overview](pipeline-overview.md) for building the pipeline you save.
- [Inference operators](api/axelera.runtime.op.inference.md) for `op.load`,
  `op.onnx_model` and `op.pt2_model`.

---
title: "Custom Model Architecture"
---
# Custom Model Architecture

> [!WARNING]
> **Experimental**
> This feature is in beta. Some parts of the workflow require implementing SDK-internal Python classes. API stability is not guaranteed between releases.


How to deploy a completely new model architecture — one that isn't in the Model Zoo and requires a custom decoder, dataset adapter, or evaluator.

**Start here first:** If your model uses a standard architecture (YOLO, ResNet, etc.) with custom weights, use [Deploy Custom Weights](custom-weights.md) instead — it's much simpler and doesn't require implementing any SDK classes.

---

## When you need this

You need the custom model workflow when:
- Your model architecture is not in the Model Zoo
- Your output tensor format requires a custom decoder (to convert raw tensors to bounding boxes, class labels, etc.)
- Your dataset uses a labeling format not supported by the built-in adapters
- You need a custom accuracy metric

---

## The Voyager framework APIs

The SDK provides five extension points. You only implement the ones you need:

| API | What you implement | Example in SDK |
|-----|-------------------|----------------|
| `types.Model` | A deployable PyTorch or ONNX model class. Referenced in YAML `models` section. | `AxUltralyticsYOLO` |
| `types.DataAdapter` | A dataset adapter: outputs images and ground truth in `AxTaskMeta` format. Referenced in YAML `datasets` section. | `ObjDataAdapter` |
| `types.Evaluator` | Compares inference results against ground truth to compute accuracy metrics (e.g. mAP). Defined as a property of a `DataAdapter`. | `ObjectEvaluator` |
| `AxOperator` | A host-side pipeline element: pre-processing or post-processing (resize, normalize, etc.). | `Resize` |
| `AxOperator` decoder | An `AxOperator` that converts raw model output tensors to `AxTaskMeta` metadata. Referenced in YAML `operators` section. | `YoloDecode` |

All components communicate through `AxTaskMeta` — the SDK's common metadata representation for a task category (object detection, classification, etc.).

---

## Workflow overview

1. **Define your model class** — subclass `types.Model`, implement the required methods, point `weight_path` at your weights file.
2. **Define a decoder** — subclass `AxOperator`, implement the tensor-to-metadata conversion for your model's output format.
3. **Define a data adapter** (if needed) — subclass `types.DataAdapter` to load your dataset.
4. **Wire it up in YAML** — reference your classes in the `models`, `datasets`, and `operators` sections.
5. **Deploy** — run `./deploy.py` as with any other model.

> [!WARNING]
> **Image operators are stripped during compilation**
> The Axelera model compiler removes any image pre-processing operators from the source model definition before compiling for Metis hardware. You **must** specify those operations as pipeline operators in your YAML file. If you don't, the end-to-end pipeline will run without the preprocessing your model was trained with, silently producing incorrect results.


---

## Getting started

The best starting point is the source of an existing Model Zoo model that's architecturally similar to yours. Browse:

- `ax_models/yolo/ax_yolo.py` — YOLO model class
- `ax_datasets/objdataadapter.py` — object detection data adapter
- `ax_evaluators/obj_eval.py` — mAP evaluator
- `ax_models/decoders/yolo.py` — YOLO decoder operator

---

## See also

- [Deploy Custom Weights](custom-weights.md) — simpler path for standard architectures with custom weights
- [Dataset Adapters](../reference/models/adapters.md) — built-in adapters you can extend or reuse
- [GStreamer Operators](../reference/pipeline/gst-operators.md) — built-in pre/post-processing operators
- [Compiler Python API](../reference/compiler/compiler-api.md) — programmatic compilation

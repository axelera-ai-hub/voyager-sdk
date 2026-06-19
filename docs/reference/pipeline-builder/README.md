---
title: Pipeline Builder
---
# Pipeline Builder **\[Experimental\]**

The Pipeline Builder is the Pythonic API for building ML inference pipelines on the Axelera Metis AIPU. Where [YAML pipelines](../pipeline/README.md) give you optimized GStreamer throughput for production deployment, the Pipeline Builder gives you composable Python operators for custom inter-stage logic, rapid prototyping, and workflows that go beyond standard detect-and-track patterns.

> [!IMPORTANT]
> **Experimental**
> Core operators (detection, classification, pose, segmentation, tracking) are stable. Cascade (`op.for_each`, `op.crop_roi`) and streaming APIs are still in development. Optimized fused kernels from the YAML pipeline path have not yet been ported — each release closes this gap.


## Start here

- [Quickstart](quickstart.md) — your first pipeline in under a minute
- [Model Compilation](model-compilation.md) — compile your model to `.axm` (Ultralytics, ONNX, or PyTorch)
- [Pipeline Overview](pipeline-overview.md) — full reference with examples for every task type

## Operator reference

- [Transforms](api/axelera.runtime.op.transforms.md) — image preprocessing (resize, letterbox, normalize, color convert)
- [Postprocess](api/axelera.runtime.op.postprocess.md) — decode raw model output, NMS, coordinate transform, filter
- [Results](api/axelera.runtime.op.results.md) — convert arrays to typed objects (DetectedObject, PoseObject, etc.)
- [Inference](api/axelera.runtime.op.inference.md) — model loading (`op.load`, `op.onnx_model`)
- [Tracker](api/axelera.runtime.op.tracker.md) — multi-object tracking (ByteTrack, OC-SORT, SORT, TrackTrack)
- [Combinators](api/axelera.runtime.op.combinators.md) — pipeline composition (`op.seq`, `op.par`, `op.for_each`)

## Type reference

- [Types](api/axelera.runtime.op.types.md) — BBox, DetectedObject, PoseObject, SegmentedObject, TrackedObject, Classification

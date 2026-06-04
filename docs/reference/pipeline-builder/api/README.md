---
title: Pipeline Builder API
---
# Pipeline Builder API

> [!IMPORTANT]
> **Preview**
> Core operators (detection, classification, pose, segmentation, tracking) are stable. Cascade (`op.foreach`, `op.croproi`) and streaming APIs are still in development.


The Pipeline Builder API is the Pythonic interface for building ML inference pipelines on the Axelera Metis AIPU. It replaces YAML-based pipeline configuration with composable Python operators.

## Getting started

- [Quickstart](../quickstart.md) — Overview and quick start
- [Model Compilation](../model-compilation.md) — Compile your model to `.axm`
- [Pipeline Overview](../pipeline-overview.md) — Build pipelines with full examples for every task type

## Operator Reference

- [Transforms](axelera.runtime.op.transforms.md) — Image preprocessing (resize, letterbox, normalize, color convert)
- [Postprocess](axelera.runtime.op.postprocess.md) — Decode raw model output, NMS, coordinate transform, filter
- [Results](axelera.runtime.op.results.md) — Convert arrays to typed objects (DetectedObject, PoseObject, etc.)
- [Inference](axelera.runtime.op.inference.md) — Model loading (`op.load`, `op.onnx_model`)
- [Tracker](axelera.runtime.op.tracker.md) — Multi-object tracking (ByteTrack, OC-SORT, SORT, TrackTrack)
- [Combinators](axelera.runtime.op.combinators.md) — Pipeline composition (`op.seq`, `op.par`, `op.foreach`)

## Type Reference

- [Types](axelera.runtime.op.types.md) — Data types: BBox, DetectedObject, PoseObject, SegmentedObject, TrackedObject, Classification

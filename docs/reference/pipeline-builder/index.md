# API Reference

> **Alpha:** Core operators (detection, classification, pose, segmentation, tracking)
> are stable. Cascade (`op.for_each`, `op.crop_roi`) and streaming APIs are still in development.

## Tutorials

- [Getting Started](quickstart.md) -- Overview and quick start
- [Model Compilation](model-compilation.md) -- Compile your model to `.axm`
- [Pipeline Overview](pipeline-overview.md) -- Build pipelines with full examples for every task type

## Operator Reference

- [Transforms](api/axelera.runtime.op.transforms.md) - Image preprocessing (resize, letterbox, normalize, color convert, etc.)
- [Postprocess](api/axelera.runtime.op.postprocess.md) - Decode raw model output, NMS, coordinate transform, filter
- [Results](api/axelera.runtime.op.results.md) - Convert arrays to typed objects (AxDetection, AxPose, AxSegmentation, AxClassification)
- [Inference](api/axelera.runtime.op.inference.md) - Model loading (load, onnx_model)
- [Tracker](api/axelera.runtime.op.tracker.md) - Multi-object tracking (ByteTrack, OC-SORT, SORT, TrackTrack)
- [Combinators](api/axelera.runtime.op.combinators.md) - Pipeline composition (seq, par, for_each, pack, unpack, itemgetter)

## Type Reference

- [Types](api/axelera.runtime.op.types.md) - Data types: BBox, DetectedObject, PoseObject, SegmentedObject, TrackedObject, Classification, Keypoint

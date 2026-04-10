# API Reference

> **Beta / Experimental:** This API is under active development and may change.
> It is shared early to show the direction and gather feedback.

## Operator Reference

- [Transforms](axelera.runtime.op.transforms.md) - Image preprocessing (resize, letterbox, normalize, color convert, etc.)
- [Postprocess](axelera.runtime.op.postprocess.md) - Decode raw model output, NMS, coordinate transform, filter
- [Results](axelera.runtime.op.results.md) - Convert arrays to typed objects (AxDetection, AxPose, AxSegmentation, AxClassification)
- [Inference](axelera.runtime.op.inference.md) - Model loading (load, onnx_model)
- [Tracker](axelera.runtime.op.tracker.md) - Multi-object tracking (ByteTrack, OC-SORT, SORT, TrackTrack)
- [Combinators](axelera.runtime.op.combinators.md) - Pipeline composition (seq, par, foreach, pack, unpack, itemgetter)

## Type Reference

- [Types](axelera.runtime.op.types.md) - Data types: BBox, DetectedObject, PoseObject, SegmentedObject, TrackedObject, Classification, Keypoint

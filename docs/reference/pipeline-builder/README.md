---
title: Pipeline Builder
---

# Pipeline Builder **\[Beta\]**

The Pipeline Builder is the Pythonic API for building ML inference pipelines on the Axelera Metis AIPU. Where [YAML pipelines](../pipeline/README.md) describe a fixed GStreamer pipeline in configuration, the Pipeline Builder gives you composable Python operators for custom inter-stage logic, rapid prototyping, and workflows that go beyond standard detect-and-track patterns.

> [!IMPORTANT]
> **Beta**
> The Pipeline Builder is in beta. Its concepts are in place, and many models already reach their target performance with it. Some models still have significant room to improve; that work is planned for the 2.0 release. Compared with YAML pipelines, the Pipeline Builder is more modular and more flexible, and a pipeline is easy to scale. If you already run a YAML pipeline in production, you do not need to move it.

## Start here

- [Quickstart](../../tutorials/pipeline-builder/quickstart.md): your first pipeline in under a minute
- [Model Compilation](../../tutorials/pipeline-builder/model-compilation.md): compile your model to `.axm` (Ultralytics, ONNX, or PyTorch)
- [Model Formats](model-formats.md): `.axm` and `.axe`, what each holds, when to use which, and the limits
- [Pipeline Overview](pipeline-overview.md): full reference with examples for every task type
- [Cascade Tutorial](../../tutorials/pipeline-builder/cascade-tutorial.md): run a second model on each detection of the first
- [Coordinate System Tutorial](../../tutorials/pipeline-builder/coordinate-system-tutorial.md): where a box's coordinates live, and how a cascade composes them
- [Postamble](postamble.md): run the model's postamble with a built-in kernel in place of onnxruntime
- [Profiling](../../tutorials/pipeline-builder/profiling.md): read a pipeline's per-operator report and find the bottleneck

## API reference

The [API reference](api/README.md) is generated from the source docstrings.

### Operators

- [Transforms](api/axelera.runtime.op.transforms.md): image preprocessing (resize, letterbox, normalize, color convert)
- [Image Preprocessing](api/axelera.runtime.op.img_preprocess.md): geometric correction (`op.perspective`, `op.barrel_correct`)
- [Postprocess](api/axelera.runtime.op.postprocess.md): decode raw model output, NMS, coordinate transform, filter, top-k, softmax
- [Results](api/axelera.runtime.op.results.md): convert arrays to typed objects (`op.ax_detection`, `op.ax_pose`, `op.ax_segmentation`, `op.ax_classification`, and more)
- [Inference](api/axelera.runtime.op.inference.md): model loading (`op.load`, `op.onnx_model`, `op.pt2_model`)
- [AXE packages](api/axelera.runtime.op.axe.md): load a saved `.axe` pipeline package (`op.load_axe`, `AxeOperator`, `IncompatibleAxeError`)
- [Tracker](api/axelera.runtime.op.tracker.md): multi-object tracking (ByteTrack, OC-SORT, SORT, TrackTrack)
- [Combinators](api/axelera.runtime.op.combinators.md): pipeline composition (`op.seq`, `op.par`, `op.for_each`, `op.pack`, `op.unpack`, `op.itemgetter`)

### Model and device settings

- [Postamble](api/axelera.runtime.postamble.md): name a kernel for the model's postamble, in place of the onnxruntime tail (see also the [Postamble guide](postamble.md))
- [Device config](api/axelera.runtime.device_config.md): board settings a model needs before it runs, such as the 110m MVM limiter

### Running and measuring pipelines

- [Sources](api/axelera.runtime.cv.source.md): video, camera and image input (`cv.create_source`)
- [Scheduler](api/axelera.runtime.scheduler.md): device selection and execution behind `pipeline.stream` and `pipeline.batch`
- [Benchmark](api/axelera.runtime.bench.md): latency, throughput and device throughput (`bench.benchmark_pipeline`)
- [Profiler](api/axelera.runtime.profiler.md): per-operator and per-stage timing (`OperatorProfiler`, `ResourceMonitor`)

### Types

- [Types](api/axelera.runtime.op.types.md): BBox, DetectedObject, PoseObject, Keypoint, SegmentedObject, TrackedObject, Classification, and more

---
title: Pipeline Builder API
---
# Pipeline Builder API

> [!IMPORTANT]
> **Beta**
> The Pipeline Builder is in beta: its concepts are in place, and some models still have room to improve in performance. See [Pipeline Builder](../README.md) for what beta means for your project.


The Pipeline Builder API is the Pythonic interface for building ML inference pipelines on the Axelera Metis AIPU. It builds a pipeline from composable Python operators, in place of a YAML pipeline configuration.

The pages below are generated from the source docstrings.

**Note:**

The `Operator` classes each have a snake_case alias. e.g.:
- `op.Seq` can also be used as `op.seq`
- `op.CenterCrop` can also be used as `op.center_crop`
- etc.

## Getting started

- [Quickstart](../../../tutorials/pipeline-builder/quickstart.md): overview and quick start
- [Model Compilation](../../../tutorials/pipeline-builder/model-compilation.md): compile your model to `.axm`
- [Model Formats](../model-formats.md): `.axm` and `.axe`, what each holds and when to use which
- [Pipeline Overview](../pipeline-overview.md): build pipelines with full examples for every task type
- [Cascade Tutorial](../../../tutorials/pipeline-builder/cascade-tutorial.md): run a second model on each detection of the first
- [Coordinate System Tutorial](../../../tutorials/pipeline-builder/coordinate-system-tutorial.md): where a box's coordinates live, and how a cascade composes them

## Operator Reference

- [Transforms](axelera.runtime.op.transforms.md): image preprocessing (resize, letterbox, normalize, color convert)
- [Image Preprocessing](axelera.runtime.op.img_preprocess.md): geometric correction (`op.perspective`, `op.barrel_correct`)
- [Postprocess](axelera.runtime.op.postprocess.md): decode raw model output, NMS, coordinate transform, filter, top-k, softmax
- [Results](axelera.runtime.op.results.md): convert arrays to typed objects (DetectedObject, PoseObject, etc.)
- [Inference](axelera.runtime.op.inference.md): model loading (`op.load`, `op.onnx_model`, `op.pt2_model`)
- [AXE packages](axelera.runtime.op.axe.md): load a saved `.axe` pipeline package (`op.load_axe`, `AxeOperator`, `IncompatibleAxeError`)
- [Tracker](axelera.runtime.op.tracker.md): multi-object tracking (ByteTrack, OC-SORT, SORT, TrackTrack)
- [Combinators](axelera.runtime.op.combinators.md): pipeline composition (`op.seq`, `op.par`, `op.for_each`, `op.pack`, `op.unpack`, `op.itemgetter`)

## Model and Device Settings

- [Postamble](axelera.runtime.postamble.md): name a kernel for the model's postamble, in place of the onnxruntime tail (see also the [Postamble guide](../postamble.md))
- [Device config](axelera.runtime.device_config.md): board settings a model needs before it runs, such as the 110m MVM limiter

## Running Pipelines

- [Sources](axelera.runtime.cv.source.md): video, camera and image input (`cv.create_source`, `VideoSource`, `Image`)
- [Scheduler](axelera.runtime.scheduler.md): device selection and execution behind `pipeline.stream` and `pipeline.batch`

## Measuring

- [Benchmark](axelera.runtime.bench.md): latency, throughput and device throughput (`bench.benchmark_pipeline`, behind a recipe's `--benchmark`)
- [Profiler](axelera.runtime.profiler.md): per-operator and per-stage timing behind that report (`OperatorProfiler`, `ResourceMonitor`)
- [Profiling](../../../tutorials/pipeline-builder/profiling.md): how to read the report those fields produce

## Type Reference

- [Types](axelera.runtime.op.types.md): data types such as BBox, DetectedObject, PoseObject, Keypoint, SegmentedObject, TrackedObject, Classification

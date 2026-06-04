---
title: "APIs"
---

# APIs

Python interfaces for building applications that run inference on Metis hardware.

| API | What it covers |
|-----|---------------|
| [InferenceStream API](inference-stream.md) | The recommended Python API for inference applications. `create_inference_stream`, `FrameResult`, metadata types (detection, tracking, classification, etc.), display, and tracers. |
| [axelera.runtime API](axruntime-py.md) | The low-level Python API for direct AIPU access. `Context`, `Connection`, `Model`, tensor management. Use this when you need fine-grained control over device selection, buffer management, or multi-instance execution. |
| [Pipeline Builder API](../pipeline-builder/index.md) | The Pythonic API for composable ML pipelines using `axelera.runtime.op`. Build pipelines with operators for transforms, inference, postprocessing, tracking, and more. Includes quickstart, model compilation guide, and full operator reference. |

For most applications, start with the [InferenceStream API](inference-stream.md) and the [Run Inference in Python](../../tutorials/run-inference-in-python.md) tutorial.

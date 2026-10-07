---
title: "APIs"
---

# APIs

Python and C/C++ interfaces for building applications that run inference on Metis hardware.

## Python

| API | What it covers |
|-----|---------------|
| [InferenceStream API](inference-stream.md) | The recommended Python API for inference applications. `create_inference_stream`, `FrameResult`, metadata types (detection, tracking, classification, etc.), display, and tracers. |
| [axelera.runtime API](axruntime-py.md) | The low-level Python API for direct AIPU access. `Context`, `Connection`, `Model`, tensor management. Use this when you need fine-grained control over device selection, buffer management, or multi-instance execution. |
| [Pipeline Builder API](../pipeline-builder/README.md) | The Pythonic API for composable ML pipelines using `axelera.runtime.op`. Build pipelines with operators for transforms, inference, postprocessing, tracking, and more. Includes quickstart, model compilation guide, and full operator reference. |

## C/C++

| API | What it covers |
|-----|---------------|
| [AxInferenceNet C++ Reference](axinferencenet.md) | The recommended C++ API for inference applications. An asynchronous pipeline handling multiple streams and models, with preprocessing, dequantization and decoding plugins. Usable standalone or through the `axinferencenet` GStreamer element. |
| [AxRuntime C/C++ API](axruntime-cpp.md) | The low-level C API for direct AIPU access, callable from C or C++. The same object model as `axelera.runtime`: context, connection, model, model instance, tensor and device management. |

For most applications, start with the [InferenceStream API](inference-stream.md) and the [Run Inference in Python](../../tutorials/run-inference-in-python.md) tutorial. For C++ applications, start with [AxInferenceNet](axinferencenet.md) and the [AxInferenceNet C++ Example](../../tutorials/examples/axinferencenet-basic.md).

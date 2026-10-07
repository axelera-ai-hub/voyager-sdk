# Reference

Technical reference for all Voyager SDK components, organized by layer — from the tools you run, through the pipeline internals, to the compiler and system configuration.

| Section | What it covers |
|---------|---------------|
| [Tools](tools/README.md) | CLI tools: `axdevice`, `inference.py`, `deploy.py`, `axrunmodel` |
| [Pipeline](pipeline/README.md) | Pipeline structure, model file formats, and the full GStreamer operator reference |
| [APIs](apis/README.md) | Python APIs: InferenceStream (high-level), axelera.runtime (low-level), and Pipeline Builder. C/C++ APIs: AxInferenceNet (pipeline) and AxRuntime (low-level) |
| [Pipeline Builder](pipeline-builder/README.md) | Pythonic API for composable ML pipelines using `axelera.runtime.op` — quickstart, model compilation, operator reference |
| [Models](models/model-zoo.md) | Model Zoo, dataset adapters, and additional models |
| [Compiler](compiler/README.md) | Compiler CLI, Python API, multi-core configuration, and ONNX operator support |
| [AxScript](axscript/axelerascript_guide.md) | AxeleraScript language guide for writing custom kernels and models: programming model and language reference |
| [Measurement](measurement/README.md) | Performance metrics, bottleneck analysis, and accuracy measurement |
| [System](system/README.md) | Hardware architecture and thermal management |

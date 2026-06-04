# Reference

Technical reference for all Voyager SDK components, organized by layer — from the tools you run, through the pipeline internals, to the compiler and system configuration.

| Section | What it covers |
|---------|---------------|
| [Tools](tools/README.md) | CLI tools: `install.sh`, `axdevice`, `inference.py`, `deploy.py`, `axrunmodel` |
| [Pipeline](pipeline/README.md) | Pipeline structure, model file formats, and the full GStreamer operator reference |
| [APIs](apis/README.md) | Python APIs: InferenceStream (high-level), axelera.runtime (low-level), and Pipeline Builder |
| [Pipeline Builder](pipeline-builder/README.md) | Pythonic API for composable ML pipelines using `axelera.runtime.op` — quickstart, model compilation, operator reference |
| [Models](models/model-zoo.md) | Model Zoo, dataset adapters, and additional models |
| [Compiler](compiler/README.md) | Compiler CLI, Python API, multi-core configuration, and ONNX operator support |
| [Measurement](measurement/README.md) | Performance metrics, bottleneck analysis, and accuracy measurement |
| [System](system/README.md) | Hardware architecture, virtual environments, environment variables, and thermal management |

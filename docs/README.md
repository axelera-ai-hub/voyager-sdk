# Voyager SDK

The Voyager SDK makes it easy to build high-performance inferencing applications with Axelera AI Metis devices.

> [!IMPORTANT]
> This is a production-ready release of Voyager SDK. Software components and features that are in development are marked **\[Beta\]** indicating tested functionality that will continue to grow in future releases or **\[Experimental\]** indicating early-stage feature with limited testing.


## Finding your way around

**[Getting Started](getting-started/hardware-install.md)** walks you through hardware installation and verifying your setup. Start here if this is your first time with a Metis device.

**[User Guides](user-guides/sdk-install.md)** cover day-to-day tasks: installing the SDK, running your first inference, updating firmware, working with LLMs, and monitoring your device. These are step-by-step and assume no prior experience with the SDK.

**[Tutorials](tutorials/README.md)** go deeper: video sources, custom weights, cascaded pipelines, Python API usage, and code examples you can run and modify. Come here once you're up and running and want to build something.

**[Model Zoo](reference/models/model-zoo.md)** lists every pre-trained model in this release with performance data, supported tasks, and licensing. Use it to find the right model for your application.

**[Reference](reference/README.md)** is the technical detail: CLI tools, pipeline configuration, compiler options, API specifications, and system internals. Look things up here when you need exact flags, parameters, or architecture details.

**[Glossary](glossary.md)** defines the terms used throughout — AIPU, pipeline, mAP, and other SDK-specific vocabulary.

---

## Install SDK and get started

| Document | Description |
| :--- | :--- |
| [Hardware installation](getting-started/hardware-install.md) | Install your Metis M.2, PCIe or Compute Board |
| [SDK installation](user-guides/sdk-install.md) | Clone the repo, run the installer, activate your environment |
| [Verify setup](getting-started/verify-setup.md) | Confirm the device is detected and the stack is working |
| [Windows setup](user-guides/windows-setup.md) | Install Voyager SDK and run a model in Windows 11 (WSL2 + native) |
| [AxDevice manual](reference/tools/axdevice.md) | Lists all Metis boards connected to your system and configures their settings |
| [Firmware update](user-guides/firmware-update.md) | Update your board firmware |

## Deploy models on Metis devices

| Document | Description |
| :--- | :--- |
| [Model zoo](reference/models/model-zoo.md) | All models supported by this release of the Voyager SDK |
| [Deployment manual (`deploy.py`)](reference/tools/deploy-py.md) | All options provided by the command-line deployment tool |
| [Custom weights](tutorials/custom-weights.md) | Deploy a model using your own weights |
| [Custom model](tutorials/custom-model.md) | Deploy a custom model architecture |
| [Compiler CLI](reference/compiler/compiler-cli.md) | Compiler Command Line Interface \[beta\] |
| [Compiler API](reference/compiler/compiler-api.md) | Python Compiler API \[experimental\] |
| [Compiler configuration](reference/compiler/compiler-configs.md) | TOML and Python configuration options |

## Run models on Metis devices

| Document | Description |
| :--- | :--- |
| [First inference](user-guides/first-inference.md) | Run object detection on a camera or video file |
| [Run inference in Python](tutorials/run-inference-in-python.md) | InferenceStream API with worked examples |
| [Video sources](tutorials/video-sources.md) | Cameras, RTSP streams, video files, and multiple inputs |
| [Measure accuracy](tutorials/measure-accuracy.md) | Benchmark a model against a validation dataset |
| [Inferencing manual (`inference.py`)](reference/tools/inference-py.md) | All options provided by the command-line inferencing tool |
| [Cascaded pipelines](tutorials/cascaded-pipelines.md) | Chain models together (e.g. detect then classify) |
| [LLM inference](user-guides/llm.md) | Run Language Models on Metis devices \[experimental\] |

## Two ways to build pipelines

The Voyager SDK provides two pipeline approaches. Use whichever fits your workflow, or combine them.

| | [YAML Pipeline](reference/pipeline/README.md) | [Pipeline Builder](reference/pipeline-builder/README.md) **\[Experimental\]** |
| :--- | :--- | :--- |
| **Best for** | Production deployment, standard workflows | Custom inter-stage logic, rapid prototyping |
| **Define pipelines in** | YAML configuration files | Python code (`axelera.runtime.op`) |
| **Strengths** | Optimized GStreamer throughput, battle-tested | Composable operators, Jupyter-friendly, full Python control |
| **Maturity** | Stable — production systems run on this today | Core operators stable; cascade and streaming APIs in development |

**Hybrid approach:** Many teams use YAML pipelines for primary inference (detection, segmentation) via InferenceStream, then hand off to Pipeline Builder operators for tracking, filtering, and custom business logic in Python.

## Application integration APIs

The Voyager SDK allows you to develop inferencing pipelines and end-user applications at different levels of abstraction.

| API | Description |
| :--- | :--- |
| [Pipeline Builder](reference/pipeline-builder/README.md) | Pythonic API for composable ML pipelines using `axelera.runtime.op` operators \[experimental\] |
| [InferenceStream](reference/apis/inference-stream.md) (high level) | Python library for reading pipeline image and inference metadata from within your application |
| [AxRuntime](reference/apis/axruntime-py.md) (low level) | Python API for manually constructing, configuring and executing pipelines |
| [GStreamer plugins](reference/pipeline/gst-operators.md) | Plugins for integrating Metis inferencing within a GStreamer pipeline |

## Code examples

Complete, runnable examples demonstrating different integration patterns.

| Example | What it shows |
| :--- | :--- |
| [Application (basic)](tutorials/examples/application.md) | Simplest InferenceStream integration |
| [Application (extended)](tutorials/examples/application-extended.md) | Runtime telemetry, hardware decoding, dynamic render settings |
| [Application (tensor)](tutorials/examples/application-tensor.md) | Direct tensor access for custom post-processing |
| [AxInferenceNet (basic)](tutorials/examples/axinferencenet-basic.md) | C++ low-level model integration |
| [AxInferenceNet (cascaded)](tutorials/examples/axinferencenet-cascaded.md) | C++ cascaded model pipeline |
| [AxInferenceNet (tensor)](tutorials/examples/axinferencenet-tensor.md) | C++ direct tensor access |
| [Classification](tutorials/examples/classification.md) | Image classification with generator-based input |
| [Cross-line counting](tutorials/examples/cross-line-count.md) | Vehicle counting across a virtual line |
| [Multiple pipelines](tutorials/examples/multiple-pipelines.md) | Dynamic pipeline management and hot-swapping |
| [Remote monitor](tutorials/examples/remote-monitor.md) | TCP broadcast of real-time JSON telemetry |

## Reference

| Document | Description |
| :--- | :--- |
| [Pipeline basics](reference/pipeline/pipeline-basics.md) | How inference pipelines work |
| [Model formats](reference/pipeline/model-formats.md) | The `model.json` file and compiled output structure |
| [YAML operators](reference/pipeline/yaml-operators.md) | Pipeline operators for YAML configuration |
| [GStreamer operators](reference/pipeline/gst-operators.md) | GStreamer pipeline plugins reference |
| [Inference configuration](reference/pipeline/inference-configuration.md) | Advanced inference settings |
| [Compiler configuration reference](reference/compiler/compiler-config-ref.md) | Full compiler config field reference |
| [ONNX operator support](reference/compiler/onnx-support.md) | ONNX operators supported by the Axelera AI compiler |
| [Model adapters](reference/models/adapters.md) | Custom dataset adapters and evaluators |
| [Additional models](reference/models/additional-models.md) | Models beyond the standard zoo |
| [AxRunmodel](reference/tools/axrunmodel.md) | Run deployed models with DMA buffers, double buffering, multiple cores |
| [install.sh](reference/tools/install-sh.md) | Installer options and what it installs |
| [Hardware](reference/system/hardware.md) | Metis hardware specifications and capabilities |
| [Environment variables](reference/system/environment-variables.md) | SDK environment variables reference |
| [Virtual environments](reference/system/virtual-environments.md) | Why activation is required and how it works |
| [Thermal and power](user-guides/thermal.md) | Thermal behavior, power management, and monitoring |
| [AxMonitor](user-guides/axmonitor.md) | Real-time device monitoring tool |
| [Performance](reference/measurement/performance.md) | Performance benchmarking methodology |
| [Accuracy metrics](reference/measurement/accuracy-metrics.md) | Understanding mAP, precision, recall |
| [Glossary](glossary.md) | Definitions for terms used throughout the SDK docs |

## Support

- [Axelera AI Community](https://community.axelera.ai/) — forums, projects, technical support
- [Customer Portal](https://support.axelera.ai/) — technical documents and support tickets
- [GitHub Issues](https://github.com/axelera-ai-hub/voyager-sdk/issues) — SDK bugs and feature requests

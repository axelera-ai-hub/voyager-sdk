![image](/docs/images/Ax_Voyager_SDK_Repo_Banner_1600x457_01.png)

# Voyager SDK repository
v1.6: [Release notes](/RELEASE_NOTES.md)

- [Voyager SDK repository](#voyager-sdk-repository)
  - [Release Qualification](#release-qualification)
  - [Install SDK and get started](#install-sdk-and-get-started)
  - [Deploy models on Metis devices](#deploy-models-on-metis-devices)
  - [Run models on Metis devices](#run-models-on-metis-devices)
  - [Application integration APIs](#application-integration-apis)
  - [Reference pipelines](#reference-pipelines)
  - [\[Alpha\] Pipeline Builder API](#alpha-pipeline-builder-api)
  - [Additional documentation](#additional-documentation)
  - [Further support](#further-support)

The Voyager SDK makes it easy to build high-performance inferencing applications with Axelera AI Metis devices. The sections below provide links to code examples, tutorials and reference documentation.

## Release Qualification

This is a production-ready release of Voyager SDK. Software components and features that are in development are marked with one of the following maturity labels:

- **Experimental:** May change or be removed without notice; no support guarantees.
- **Alpha:** Usable but incomplete; breaking changes possible.
- **Beta:** Feature-complete but not fully stable; committed to developing this further in future releases.


## Install SDK and get started

| Document | Description |
| :--------------------- | :---------- |
| [Installation guide - original `install.sh`](/docs/tutorials/install.md) | Explains how to set up the Voyager SDK repository and toolchain on your development system using `install.sh` |
| [Installation guide - new standalone Python wheels](/docs/tutorials/install_pip.md) | Explains how to set up the Voyager SDK repository and toolchain on your development system using new standalone Python wheels |
| [Quick start guide](/docs/tutorials/quick_start_guide.md) | Explains how to deploy and run your first model |
| [Windows getting started guide](/docs/tutorials/windows/windows_getting_started.md) | Explains how to install Voyager SDK and run a model in Windows 11|
| [AxDevice manual](/docs/reference/axdevice.md) | AxDevice is a tool that lists all Metis boards connected to your system and can configure their settings |
| [Board firmware update guide](/docs/tutorials/firmware_flash_update.md) | Explains how to update your board firmware (for customers with older boards who have received instructions) |

## Deploy models on Metis devices

| Document | Description |
| :--------------------- | :---------- |
| [Model zoo](/docs/reference/model_zoo.md) | Lists all models supported by this release of the Voyager SDK |
| [Deployment manual (`deploy.py`)](/docs/reference/deploy.md) | Explains all options provided by the command-line deployment tool |
| [AxMO deployment example (`deploy_huggingface_classifier.py`)](/docs/tutorials/deploy_with_axmo.md) | Deploy HuggingFace classifiers using AxMO quantization [experimental] |
| [Custom weights tutorial](/docs/tutorials/custom_weights.md) | Explains how to deploy a model using your own weights |
| [Custom model tutorial](/docs/tutorials/custom_model.md) | Explains how to deploy a custom model |

## Run models on Metis devices

| Document | Description |
| :--------------------- | :---------- |
| [Benchmarking guide](/docs/tutorials/benchmarking.md) | Explains how to measure end-to-end performance and accuracy |
| [Inferencing manual (`inference.py`)](/docs/reference/inference.md) | Explains all options provided by command-line interencing tool |
| [Application integration tutorial (high level)](/docs/tutorials/application.md) | Explains how to integrate a YAML pipeline within your application |
| [Application integration tutorial (low level)](/docs/tutorials/axinferencenet.md) | Explains how to integrate an AxInferenceNet model within your application |

## Application integration APIs

The Voyager SDK allows you to develop inferencing pipelines and end-user applications
at different levels of abstraction.

| API | Description |
| :--------------------- | :---------- |
| InferenceStream (high level) | [Library](/docs/tutorials/application.md) for directly reading pipeline image and inference metadata from within your application |
| AxInferenceNet (middle level) | [C/C++](/docs/reference/axinferencenet.md) API reference for integrating model inferencing and pipeline construction directly within an application |
| AxRuntime (low level) | [Python](/docs/reference/axelera.runtime.md) and [C/C++](/docs/reference/axruntime.md) APIs for manually constructing, configuring and executing pipelines |
| GStreamer | [Plugins](/docs/reference/pipeline_operators.md) for integrating Metis inferencing within a GStreamer pipeline

The InferenceStream library is the easiest to use and enables most users to achieve the highest performance. The lower-level APIs enable expert users
to integrate Metis within existing video streaming frameworks.

## Reference pipelines

The Voyager SDK makes it easy to construct pipelines that combine multiple models in different ways. A number of
end-to-end reference pipelines are provided, which you can use as templates for your own projects.

| Directory | Description |
| :--------------------- | :---------- |
| [`/ax_models/reference/parallel`](/ax_models/reference/parallel) | Multiple pipelines running in parallel |
| [`/ax_models/reference/cascade`](/ax_models/reference/cascade) | Cascaded pipelines in which the output of one model is input to a secondary model |
| [`/ax_models/reference/cascade/with_tracker`](/ax_models/reference/cascade) | Cascaded pipelines in which the output of the first model is tracked prior to being input to a secondary model |
| [`/ax_models/reference/image_preprocess`](/ax_models/reference/image_preprocess) | Pipelines in which the camera input is first preprocessed prior to being used for inferencing |

## \[Alpha\] Pipeline Builder API

A new Python-native API for building, running, and packaging ML inference pipelines. The entire pipeline, from model loading through post-processing and tracking, can be expressed as a composable Python expression.

- **Composable operators**: `op.seq()` for sequential, `op.par()` for parallel, `op.foreach()` for cascade (per-object) processing.
- **Data routing**: `op.select(i)` to extract from tuples, `op.pack()` / `op.unpack()` for explicit tuple conversion.
- **30+ operators** across preprocessing, inference, postprocessing, filtering, tracking, and result types.
- **Model loading**:
    - `op.load('model.axm')` - hardware inference on AIPU.
    - `op.onnx_model('model.onnx')` - CPU inference, no AIPU required.
    - `op.load('pipeline.axe')` - portable pipeline package (new .axe format).
- **Tracker integration**: `op.tracker(algo='bytetrack')` - supports ByteTrack, OC-SORT, SORT, TrackTrack. Full lifecycle states via `return_all_states=True` (new, tracked, lost, removed).
- **Pipeline optimizer**: Automatic SIMD-accelerated fusion of operator chains (e.g., NchwToNhwc + Quant + Pad into single QuantizeTransposePad).
- **Typed result objects**: `DetectedObject`, `PoseObject`, `SegmentedObject`, `TrackedObject`, `Classification` with protocol-based interfaces and `.draw()` visualization.

## Additional documentation

This section provides links to additional documentation available in the Voyager SDK repository.

| Document                                                                   | Description                                                                                                                                                                            |
|:---------------------------------------------------------------------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Advanced deployment tutorials](/ax_models/tutorials/general/tutorials.md) | Advanced deployment options [experimental]                                                                                                                                             |
| [AxRunmodel manual](/docs/reference/axrunmodel.md)                         | AxRunModel is a tool that can run deployed models on Metis hardware using different features available in the AxRuntime API (such as DMA buffers, double buffering and multiple cores) |
| [Compiler CLI](/docs/reference/compiler_cli.md)                            | Compiler Command Line Interface [beta]                                                                                                                                                 |
| [Compiler API](/docs/reference/compiler_api.md)                            | Python Compiler API [experimental]                                                                                                                                                     |
| [ONNX operator support](/docs/reference/onnx-opset17-support.md)           | List of ONNX operators supported by the Axelera AI compiler                                                                                                                            |
| [Thermal and Power Guide](/docs/reference/thermal_and_power_guide.md)      | Document detailing the thermal behavior and power management for Metis and instructions to make changes                                                                                |
| [SLM/LLM inference tutorial](/docs/tutorials/llm.md)                       | Explains how to run Language Models on Metis devices [experimental]                                                                                                                    |

## Further support

For blog posts, projects and technical support please visit [Axelera AI Community](https://community.axelera.ai/).

For technical documents and guides please visit [Customer Portal](https://support.axelera.ai/).

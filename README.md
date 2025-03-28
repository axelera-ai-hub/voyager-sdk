![image](/docs/images/Ax_Voyager_SDK_Repo_Banner_1600x457_01.png)

# Voyager SDK repository

- [Voyager SDK repository](#voyager-sdk-repository)
  - [Install SDK and get started](#install-sdk-and-get-started)
  - [Deploy models on Metis devices](#deploy-models-on-metis-devices)
  - [Run models on Metis devices](#run-models-on-metis-devices)
  - [Application integration APIs](#application-integration-apis)
  - [Reference pipelines](#reference-pipelines)
  - [Additional documentation](#additional-documentation)

The Voyager SDK makes it easy to build high-performance inferencing applications with Axelera AI Metis devices. The sections below provide links to code examples, tutorials and reference documentation.

> [!IMPORTANT]
> This is a production-ready release of Voyager SDK. Software components and features that are in development are marked "[Beta]"
> indicating tested functionality that will continue to grow in future releases or "[Experimental]" indicating early-stage feature with limited testing.

## Install SDK and get started

| Document | Description |
| :--------------------- | :---------- |
| [Installation guide](/docs/tutorials/install.md) | Explains how to setup the Voyager SDK repository and toolchain on your development system |
| [Quick start guide](/docs/tutorials/quick_start_guide.md) | Explains how to deploy and run your first model |
| [AxDevice manual](/docs/reference/axdevice.md) | AxDevice is a tool that lists all Metis boards connected to your system and can configure their settings |
| [Board firmware update guide](/docs/tutorials/firmware_flash_update.md) | Explains how to update your board firmware (for customers with older boards who have received instructions) |

## Deploy models on Metis devices

| Document | Description |
| :--------------------- | :---------- |
| [Model zoo](/docs/reference/model_zoo.md) | Lists all models supported by this release of the Voyager SDK |
| [Deployment manual (`deploy.py`)](/docs/reference/deploy.md) | Explains all options provided by the command-line deployment tool |
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

An experimental [inference server](/docs/reference/inference_server.md) is also provided, which lets you declare `InferenceStream` objects as part of a
host application and offload the associated pipeline processing to a networked server.

## Reference pipelines

The Voyager SDK makes it easy to construct pipelines that combine multiple models in different ways. A number of
end-to-end reference pipelines are provided, which you can use as templates for your own projects.

| Directory | Description |
| :--------------------- | :---------- |
| [`/ax_models/reference/parallel`](/ax_models/reference/parallel) | Multiple pipelines running in parallel |
| [`/ax_models/reference/cascade`](/ax_models/reference/cascade) | Cascaded pipelines in which the output of one model is input to a secondary model |
| [`/ax_models/reference/cascade/with_tracker`](/ax_models/reference/cascade) | Cascaded pipelines in which the output of the first model is tracked prior to being input to a secondary model |
| [`/ax_models/reference/image_preprocess`](/ax_models/reference/image_preprocess) | Pipelines in which the camera input is first preprocessed prior to being used for inferencing |

## Additional documentation

This section provides links to additional documentation available in the Voyager SDK repository.

| Document | Description |
| :--------------------- | :---------- |
| [Advanced deployment tutorials](/ax_models/tutorials/general/tutorials.md) | Advanced deployment options [experimental] |
| [AxRunmodel manual](/docs/reference/axrunmodel.md) | AxRunModel is a tool that can run deployed models on Metis hardware using different features available in the AxRuntime API (such as DMA buffers, double buffering and multiple cores) |
| [ONNX operator support](/docs/reference/onnx-opset14-support.md) | List of ONNX operators supported by the Axelera AI compiler

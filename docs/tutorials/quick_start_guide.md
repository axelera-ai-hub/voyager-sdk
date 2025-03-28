![image](/docs/images/Ax_Page_Banner_2500x168_01.png)

# Voyager SDK: Quick Start Guide

- [Voyager SDK: Quick Start Guide](#voyager-sdk-quick-start-guide)
  - [Introduction](#introduction)
  - [Development environment](#development-environment)
  - [Runtime environment](#runtime-environment)
    - [Define an ``InferenceStream`` object in your application](#define-an-inferencestream-object-in-your-application)
    - [Launch an inference server](#launch-an-inference-server)
    - [Build a GStreamer pipeline](#build-a-gstreamer-pipeline)
    - [Use the AxRuntime API to inteface with Metis directly](#use-the-axruntime-api-to-inteface-with-metis-directly)
  - [Setup](#setup)
  - [Run a Metis-accelerated pipeline](#run-a-metis-accelerated-pipeline)
    - [Measure model accuracy](#measure-model-accuracy)
  - [Next steps](#next-steps)
  - [API documentation](#api-documentation)

## Introduction

The Voyager SDK makes it easy to build high-performance applications for Axelera's Metis AI
processing unit (AIPU).

<p align="center">
  <img src="/docs/images/voyager-overview.png"
  alt="Axelera Optimized Deployment"/>
</p>

## Development environment

The fastest way to get started is
to define a YAML-based pipeline with one or more models from the [Voyager model zoo](/docs/reference/model_zoo.md)
and build it using the AI Pipeline Builder. The model zoo contains many industry-standard models,
most of which can be configured with your own [custom weights/dataset](/docs/tutorials/custom_weights.md)
by updating only a few fields in the model YAML file (and without having to write any code).

The [AI Pipeline Builder](/docs/reference/deploy.md) converts the YAML pipeline description into executable code
for a specified target platform. A target platform usually comprises a host CPU with an integrated video decoder and
embedded GPU, which is interfaced over PCIe to a Metis device. The pipeline build steps include calling
the model compiler to generate Metis-executable code, and mapping the image pre-processing and
post-processing elements onto a library of optimized OpenCL kernels for GPU acceleration.
Numerous optimization techniques are automatically applied during pipeline deployment, such as
fusing consecutive processing layers together to reduce the amount of
intermediate pipeline data generated in external memory. The generated pipeline is based on
a set of foundational Axelera plugins including ``AxInference`` for executing models on Metis
and ``AxTransform`` for executing OpenCL kernels on the host, both of which interoperate
to manage the dataflow (buffer allocations and synchronization) between different hardware optimally.

The final step of connecting the generated pipeline to one or more specific input and output
devices (such as cameras, files and displays) is deferred to runtime, as this provides the greatest
flexibility when incorporating the pipeline in an application.

## Runtime environment

Axelera provides the command-line evaluation tool [``inference.py``](/docs/reference/inference.md)
that lets you run deployed models and pipelines with many different configurations of input
and output device including files, live video streams and model validation datasets. This tool provides
specific options for [benchmarking end-to-end model performance and accuracy](/docs/tutorials/benchmarking.md),
and it can generate visually compelling proof-of-concept demos that save you time when prototyping new solutions.

Once you are satisfied with the overall performance of your pipeline, you can integrate this
within your application in a number of different ways.

### Define an ``InferenceStream`` object in your application

Pipelines built using the AI Pipeline Builder can be easily incorporated within
Python and C/C++ applications using libraries that provide an `InferenceStream` object.
An InferenceStream can be configured at runtime with
one or more input sources, and it returns a sequence of input images and
associated inference metadata. The Voyager SDK provides additional libraries to parse,
analyse and render metadata visually.

The sample file [``application.py``](/examples/application.py) and
[associated tutorial](/docs/tutorials/application.md) gives a minimal working
example of how to create an InferenceStream and integrate it within your own application.

### Launch an inference server

The Axelera inference server makes it easy to offload inferencing from an application
to a seprarate Metis-based host, which the client application can then interface with using
standard protocols such as gRPC.

A [tutorial](/docs/reference/inference_server.md) is provided based on the sample files
[``inference_server.py``](/inference_server.py) and
[``inference_cilent.py``](/inference_client.py), which provide a minimal working
example of setting up a client-server solution.

Note that the inference server is currently available as an early proof-of-concept and
will be productised in future releases.

### Build a GStreamer pipeline

Users working with GStreamer pipelines can directly utilize Axelera plugins
such as ``AxInferenceNet`` to integrate a deployed model directly within a
GStreamer pipeline.

[Reference documentation](/docs/reference/pipeline_operators.md) is provided for
all of the foundational Axelera plugins available for use in GStreamer pipelines.

### Use the AxRuntime API to inteface with Metis directly

Advanced users can directly use the [``AxRuntime``](/docs/reference/axruntime.md) C API
or the [axelera.runtime](/docs/reference/axelera.runtime.md) Python API, which provides
functions for selecting a Metis device, loading models, and performing inference
on the device.

Note that additional documentation explaining how deploy model YAML files and
perform inference with AxRuntime will be provided in a later release.

## Setup

The Voyager SDK is released as a GitHub repository. The
[installation guide](/docs/tutorials/install.md) covers the steps
needed to download, install and activate a development environment on your
system.

## Run a Metis-accelerated pipeline

To launch a Metis-accelerated example pipeline with the object detector model YOLOv5s-v7 and
a single USB camera, run the following command:

```bash
./inference.py yolov5s-v7-coco usb:0
```
If you don't have a USB camera, you can use a different source to evaluate, such as a local video file:

```bash
./inference.py yolov5s-v7-coco media/traffic1_1080p.mp4
```

The first time you launch a new model, the pipeline compiler is invoked to build the model
and compilation progress is indicated in the progress bar shown in the terminal window.
By default, the compiler checks for availability of host image acceleration hardware
(such as VA-API and OpenCL) and compiles for the most suitable hardware found.

Following a successful build, input from the USB camera is displayed in a new window with
bounding boxes overlaid on detected objects. An instrumentation panel is also shown in the bottom left of this window, which
provides visual representation of the following data:

1. **System throughput**: the measured end-to-end performance of the complete pipeline
including all pre-processing, inferencing and post-processing elements.

2. **Device throughput**: indicates the maximum Metis device throughput possible if all
other pipeline elements are able to produce and consume data sufficiently quickly for
Metis to process.

3. **CPU utilization**: provides an indication of how effectively the pipeline is offloaded
from the host CPU to the hardware accelerators on the target system.

Once the pipeline is finished running, the average accumulated metric values are displayed
as a summary in the terminal.

### Measure model accuracy

To measure the accuracy of the YOLOv5s-v7 model running on the target hardware, run
the following command:

```
./inference.py yolov5s-v7-coco dataset --no-display
```

This command launches the pipeline with all images from the model's default validation dataset
(in this case, COCO2017) calculating the accuracy based on the output of the inference predictions
and the ground truth data. On completion, the mean average precision (mAP) for
the entire dataset is output to the terminal.

## Next steps

If you've made it here, great! That means that your Axelera AI platform is setup and functioning
correctly and you're now ready to start a more in-depth evaluation or even start your Proof of Concept with Metis.

As a next step in your evaluation, we recommend this guide on [how to work with different video sources](/docs/tutorials/video_sources.md), 
including running inference on multiple streams simultaneously and using different video sources. 

Any Voyager SDK pipeline you run using `inference.py` can be easily integrated [within your own application](/docs/tutorials/application.md).
This allows you to focus on building your business logic while confidently entrusting the AI model inference tasks to the Voyager SDK.

You can explore a wide variety of models well-integrated with the Voyager SDK in the [Axelera Model Zoo](/docs/reference/model_zoo.md) and directly benefit from highly optimized model pipelines. You might also need to [deploy a model using your own weights](/docs/tutorials/custom_weights.md) and [benchmark its performance and accuracy](/docs/tutorials/benchmarking.md). In some cases, you may wish to integrate your model within a larger [cascaded pipeline](/docs/tutorials/cascaded_model.md). Once your pipeline is complete and performing well, you can integrate your custom pipeline with application.py through simple configurations.

Beta support is also provided for deploying your own [custom model](/docs/tutorials/custom_model.md).

## API documentation

API reference material:
- [YAML pipeline operators](/docs/reference/yaml_operators.md)
- [Axelera GStreamer pipeline elements](/docs/reference/pipeline_operators.md)
- [Inference Server](/docs/reference/inference_server.md)

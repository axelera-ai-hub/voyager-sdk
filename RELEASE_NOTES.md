![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Voyager SDK release notes v1.5

- [Voyager SDK release notes v1.5](#voyager-sdk-release-notes-v15)
  - [Voyager SDK release notes v1.5.2](#voyager-sdk-release-notes-v152)
  - [Fixed Issues Since v1.5.1](#fixed-issues-since-v151)
  - [New Features / Support (v1.5.2)](#new-features--support-v152)
  - [Document Updates](#document-updates)
  - [Voyager SDK release notes v1.5.1](#voyager-sdk-release-notes-v151)
  - [Release Qualification](#release-qualification)
  - [New Features / Support (v1.5.1)](#new-features--support-v151)
    - [New Axelera AI Cards and Systems](#new-axelera-ai-cards-and-systems)
    - [New Platforms](#new-platforms)
    - [New Networks Supported](#new-networks-supported)
      - [New models for Image Classification](#new-models-for-image-classification)
      - [New models for Oriented Bounding Boxes Object Detection](#new-models-for-oriented-bounding-boxes-object-detection)
      - [New models for Instance Segmentation](#new-models-for-instance-segmentation)
      - [New models for Keypoint Detection](#new-models-for-keypoint-detection)
    - [End-to-End Pipelines](#end-to-end-pipelines)
    - [Installation](#installation)
    - [AI Pipeline Builder](#ai-pipeline-builder)
    - [Beta Model Compiler](#beta-model-compiler)
    - [Runtime](#runtime)
    - [Tools](#tools)
    - [Firmware](#firmware)
  - [Breaking Changes](#breaking-changes)
  - [Fixed Issues Since Last Release](#fixed-issues-since-last-release)
  - [Known Issues and Limitations](#known-issues-and-limitations)
  - [System Requirement](#system-requirement)
    - [Development Environment](#development-environment)
    - [Runtime Environment](#runtime-environment)
  - [Further Support](#further-support)

## Voyager SDK release notes v1.5.2
This release addresses several issues found in v1.5.1 and delivers targeted improvements to stability, compatibility, and developer experience.

## Fixed Issues Since v1.5.1
- Fixed build failures for ImageNet networks using HINT instructions.
- Fixed `NormaliseCL` to correctly handle non‑4‑channel inputs.
- Prevented segmentation faults in YOLO decoders when receiving an unexpected number of input tensors.

## New Features / Support (v1.5.2)
- Runtime now supports executing AXM files directly, streamlining deployment.
- Added selectable network protocol for RTSP sources in `inference.py` to improve input flexibility.
- Added `create_inference_net` overloads to allow existing code to compile without a context parameter.

## Document Updates
- General refinements across tutorials and references aligned with the above changes.

## Voyager SDK release notes v1.5.1

- [Voyager SDK release notes v1.5.1](#voyager-sdk-release-notes-v151)
- Support for Ubuntu 24.04 with Python 3.12 for development and running inference.
- A new computer vision task Oriented Bounding Boxes Object Detection added to the model zoo.
- New models added for image classification, instance segmentation and keypoint detection.
- Enhanced support on Windows including running LLMs and full functionality of `axmonitor`.

## Release Qualification
This is a production-ready release of Voyager SDK. Software components and features that are in
development are marked "\[Beta\]" indicating tested functionality that will continue to grow in
future releases or "\[Experimental\]" indicating early-stage feature with limited testing.

## New Features / Support (v1.5.1)

### New Axelera AI Cards and Systems
- The release adds support for
  [Metis 4-chip PCIe Cards](https://store.axelera.ai/collections/ai-acceleration-cards/products/pcie-ai-accelerator-card-powered-by-4-metis-aipu) including both the 16GB and the 64GB variants.

### New Platforms
- \[Beta\] Nvidia Jetson Orin Nano (Arm Cortex-A78AE).

### New Networks Supported

Voyager SDK model zoo includes computer vision tasks and LLMs. For a full list of supported models
and data about their performance and accuracy see [here](/docs/reference/model_zoo.md).

Models that are supported but not included in the model zoo are documented
[here](/docs/reference/additional_models.md).

For convenience, pre-compiled models are available to download by running `axdownloadmodel` in the
parent folder of Voyager SDK.

#### New models for Image Classification

| Model Name                                                                                     | Resolution | Format        |
| :--------------------------------------------------------------------------------------------- | :--------- | :------------ |
| [ResNeXt50_32x4d](/ax_models/zoo/torchvision/classification/resnext50_32x4d-imagenet.yaml)     | 224x224    | Pytorch, ONNX |
| [Wide ResNet-50](/ax_models/zoo/torchvision/classification/wide_resnet50-imagenet.yaml)        | 224x224    | Pytorch, ONNX |
| [MobilenetV3-large](/ax_models/zoo/torchvision/classification/mobilenetv3_large-imagenet.yaml) | 224x224    | Pytorch, ONNX |
| [DenseNet-121](/ax_models/zoo/torchvision/classification/densenet121-imagenet.yaml)            | 224x224    | Pytorch, ONNX |
| [RegNetX-1_6GF](/ax_models/zoo/torchvision/classification/regnet_x_1_6gf-imagenet.yaml)        | 224x224    | Pytorch, ONNX |
| [RegNetX-400MF](/ax_models/zoo/torchvision/classification/regnet_x_400mf-imagenet.yaml)        | 224x224    | Pytorch, ONNX |
| [RegNetY-1_6GF](/ax_models/zoo/torchvision/classification/regnet_y_1_6gf-imagenet.yaml)        | 224x224    | Pytorch, ONNX |
| [RegNetY-400MF](/ax_models/zoo/torchvision/classification/regnet_y_400mf-imagenet.yaml)        | 224x224    | Pytorch, ONNX |

#### New models for Oriented Bounding Boxes Object Detection

| Model Name                                                                      | Resolution | Format |
| :------------------------------------------------------------------------------ | :--------- | :----- |
| [Yolov8n-obb](/ax_models/zoo/yolo/obb_detection/yolov8n-obb-dotav1-onnx.yaml)   | 640x640    | ONNX   |
| [Yolov8l-obb](/ax_models/zoo/yolo/obb_detection/yolov8l-obb-dotav1-onnx.yaml)   | 640x640    | ONNX   |
| [Yolov11n-obb](/ax_models/zoo/yolo/obb_detection/yolo11n-obb-dotav1-onnx.yaml)  | 640x640    | ONNX   |
| [Yolov11l-obb](/ax_models/zoo/yolo/obb_detection/yolo11l-obb-dotav1-onnx.yaml)  | 640x640    | ONNX   |

#### New models for Instance Segmentation
| Model Name                                                                          | Resolution | Format          |
| :---------------------------------------------------------------------------------- | :--------- | :-------------- |
| [Yolov8m-seg](/ax_models/zoo/yolo/instance_segmentation/yolov8mseg-coco.yaml)       | 640x640    | Pytorch, ONNX   |

#### New models for Keypoint Detection
| Model Name                                                                   | Resolution | Format        |
| :--------------------------------------------------------------------------- | :--------- | :------------ |
| [Yolov8m-pose](/ax_models/zoo/yolo/keypoint_detection/yolov8mpose-coco.yaml) | 640x640    | Pytorch, ONNX |

### End-to-End Pipelines
- New YAML files for all new models offered in our model zoo in this release (see tables above).
- Enhanced multi-object tracking features and examples:
  - \[Experimental\] New Track ID Recovery Mechanism: Introduced a memory bank for track ID
    recovery, restoring a person's ID after they leave and subsequently reappear.
    See: [yolox-deep-oc-sort-osnet-membank.yaml](/ax_models/reference/cascade/with_tracker/yolox-deep-oc-sort-osnet-membank.yaml).
  - Multi-Object Tracking (MOT) with Re-ID example in C++ showcasing the built-in OC-SORT using
    `AxInferenceNet`. This allows flexible configuration to enable Deep-OC-SORT (using OSNet for
    Re-ID) and further activate the memory bank via tracking parameters.
    Refer to: [axinferencenet_tracker.cpp](/examples/axinferencenet/axinferencenet_tracker.cpp).
  - A new example [cross_line_count.py](/examples/cross_line_count.py) has been added which
    demonstrates how to use a tracker in application code. Another example
    [remote_cross_line_monitor.py](/examples/remote_cross_line_monitor.py) shows how the line
    crossing events can be made available via a simple TCP server.

### Installation
- Stability improvements in Windows installation.
- Ubuntu 24.04 native installation supported. Ubuntu 22.04 remains supported.

### AI Pipeline Builder
- Reduced latency when using OpenCL by changing how asynchronous workloads are implemented. This
  optimisation is always enabled.
- New optional low latency mode that trades-off FPS for low latency added.
  - This mode can be enabled using `--low-latency` on the command line for example with
    `inference.py`, or using `low_latency=True` in pipeline construction. 
  - Performance impact when using this mode compared to normal mode depends on the pipeline (see
    tables below for examples). Smaller and faster models are more significantly impacted. Latency
    statistics taken across all frames in the inference run are output. 

    `./inference.py yolov8n-resnet50  media/traffic1_1080p.mp4@30 --no-display --low-latency`    

    **Video res 1080p, 30 frame rate**
    |                       | Latency (ms) | Throughput (FPS) |
    | :-------------------- | :----------- | :--------------- |
    | v1.5 low latency mode | 12           | 30               |
    | v1.5 normal mode      | 403          | 30               |

    `./inference.py yolov8n-resnet50  media/traffic1_1080p.mp4 --no-display --low-latency` 

    **Video res 1080p, unrestricted frame rate**
    |                       | Latency (ms) | Throughput (FPS) |
    | :-------------------- | :----------- | :--------------- |
    | v1.5 low latency mode | 117          | 170              |
    | v1.5 normal mode      | 125          | 185              |

- Support for tiled inference on high-resolution video streams (4K or 8K) resulting in
  high-accuracy detection and pose estimation for large numbers of small objects in each frame. The 
  tiling size is configurable, while fine-grained configuration of tile size and location using a
  JSON file allows users to have more tiles in areas of interest or where the objects are further
  away. A [demo](/examples/demos/8k_demo.py) application is available.
- Support for rendering to images without windows is added for embedding rendered results into
  other UI frameworks. See examples [render_to_ui.py](/examples/render_to_ui.py) and
  [render_to_video.py](/examples/render_to_video.py).
- Multiple source/streams as inputs are supported in application framework. 
- New Polar transform operator can be used to allow inference on fish eye lenses.
- A more general crop operator has been added.
- The centre crop operator now allows for non-square outputs.

### Beta Model Compiler
- \[Beta\] The Compiler CLI default for the `resources_used` option has been changed from 1.0 to
  0.25. With this option set to the default 0.25 on compilation, it compiles models for single-core
  using 1/4 of the available memory resources. This allows to easily decide at execution time how
  many cores are used for execution (1 to 4). The best performance on multiple cores (2, 3, 4) is
  achieved by compiling with aipu_cores_used set to 4 and `resources_used` set to the appropriate
  amount of memory for all cores (which optimises the cache configuration), but this means you
  cannot run on fewer cores without recompiling. See multi-core-modes for more details on the
  different multi-core modes.
- List of supported operators documented [here](/docs/reference/onnx-opset17-support.md) will grow
  in future releases. For technical assistance on compiling your own model please turn to the
  [Axelera Community](https://community.axelera.ai/).

### Runtime
- Support for running LLMs on Windows platforms with `axllm` tool.

### Tools
- `interactive_flash_update` supports automatic firmware update of all Metis devices simultaneously.
  Refer to the [firmware flash update documentation](/docs/tutorials/firmware_flash_update.md).
- Enhanced functionality of `axmonitor`:
  - New metrics - DDR size, DDR utilization per-context, PCIe utilization per channel (4 read and
    4 writes channels).
  - Device and system timestamps recorded per sampling point.
  - System setup and device configuration provided e.g. firmware version, core frequency.
  - Full functionality supported on Windows platforms.
- New `axmodelinfo` tool to report model information in areas such as model structure (inputs,
  outputs, tensor layouts), quantization details, padding and pre-/post-processing info, version
  and compatibility checks.
- All features of `axdevice` supported on Windows platforms.

### Firmware
- Improved compatibility and UX with a range of hosts by significantly reducing Metis boot time
  resulting in successful enumeration in the first attempt.

## Breaking Changes
- Model format: The model format is updated to support new LLM model formats and future AIPU
  versions. All computer vision models compiled with SDK versions older than v1.5 need to be
  re-deployed. Similarly, downloaded pre-compiled models (computer vision as well as LLMs) need to
  be re-downloaded.

## Fixed Issues Since Last Release
- Fixed memory leak issues with discrete GPUs (SDK-8171) and axinferencenet with GStreamer.
- Improved handling of pipeline shutdown and RTSP source errors.
- Fixed incorrect cropping and face alignment issues.

## Known Issues and Limitations
- **Installer tool's docker option not working (SDK-8083):** Running `install.sh --docker` fails on
  certain configurations, for example on Firefly ITX-3588J motherboard.
- **Higher RAM required for compiling `Real-ESRGAN-x4plus`:**
  Compiling the model Real-ESRGAN-x4plus requires a machine with at least 128GB of memory.
- **Device monitoring with AxMonitor is not supported on single-MSI hosts (SDK-6581):** 
  For some systems with single-MSI hosts, device monitoring with `AxMonitor` does not display any
  data. An example of a host with this issue is Arduino Portenta X8 Mini.

## System Requirement
### Development Environment
For model compiling purposes, these are the host requirements:

| Requirement               | Detail                                                                 |
| :------------------------ | :--------------------------------------------------------------------- |
| OS                        | Linux Ubuntu 22.04, Docker (on Windows or Linux), Windows + WSL/Ubuntu |
| CPU architecture          | ARM64, x86, x86\_64                                                    |
| Recommended CPU           | Intel Core-i5 or equivalent                                            |
| Minimum System Memory     | 16GB (large models may require swap partition)                         |
| Recommended System Memory | 32 GB                                                                  |

### Runtime Environment
This release is expected to work with Intel Core-i 12th and 13th generations (x86), AMD Ryzen (x86)
and Arm64 host CPUs. Please find the list of platforms Axelera AI has tested with Metis M.2 Card
[here](https://support.axelera.ai/hc/en-us/articles/25437844422418-Metis-M-2-Tested-Host-PCs) and
Metis PCIe Card
[here](https://support.axelera.ai/hc/en-us/articles/25437554693138-Metis-PCIe-Tested-Host-PCs).

## Further Support
- For blog posts, projects and technical support please visit
[Axelera AI Customer Portal](https://support.axelera.ai).
- For technical documents and guides please visit [Customer Portal](https://support.axelera.ai/).

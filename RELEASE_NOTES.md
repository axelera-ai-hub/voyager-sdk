# Voyager SDK release notes v1.7

- [Voyager SDK release notes v1.7](#voyager-sdk-release-notes-v17)
  - [Voyager SDK release notes v1.7.0](#voyager-sdk-release-notes-v170)
    - [Release Qualification](#release-qualification)
  - [Installation and Compatibility](#installation-and-compatibility)
    - [Installation](#installation)
    - [Release Compatibility Matrix](#release-compatibility-matrix)
    - [Metis M.2 Max upgrade notes](#metis-m2-max-upgrade-notes)
  - [New Features / Support](#new-features--support)
    - [New Axelera AI Cards and Systems](#new-axelera-ai-cards-and-systems)
    - [Host Platform Support](#host-platform-support)
      - [Validated hardware platforms](#validated-hardware-platforms)
      - [Operating Systems](#operating-systems)
      - [Virtualization support](#virtualization-support)
    - [New Networks Supported](#new-networks-supported)
      - [New models for Image Classification](#new-models-for-image-classification)
      - [New models for Object Detection](#new-models-for-object-detection)
      - [New models for Semantic Segmentation](#new-models-for-semantic-segmentation)
    - [AI Pipeline Builder](#ai-pipeline-builder)
      - [\[Alpha\] Pipeline Builder API](#alpha-pipeline-builder-api)
      - [Video decode and sources](#video-decode-and-sources)
      - [New task types](#new-task-types)
      - [YAML Pipeline Builder](#yaml-pipeline-builder)
    - [\[Beta\] Model Compiler](#beta-model-compiler)
    - [Tools](#tools)
  - [Breaking Changes](#breaking-changes)
  - [Known Issues and Limitations](#known-issues-and-limitations)
    - [IMPORTANT - Memory leak in GStreamer software video decode (`gst-libav` 1.24.2)](#important---memory-leak-in-gstreamer-software-video-decode-gst-libav-1242)
    - [Other](#other)
  - [System Requirement](#system-requirement)
    - [Development Environment](#development-environment)
    - [Runtime Environment](#runtime-environment)
  - [Further Support](#further-support)

## Voyager SDK release notes v1.7.0

Voyager SDK v1.7.0 expands the pipeline builder capabilities, adds new computer vision models, and broadens hardware and platform support.

- Support for Metis 1-chip PCIe Rev2 2 GB and 8 GB DDR variants.
- New computer vision models including YOLOv4, YOLOv4-CSP-Leaky, and YOLO26 semantic segmentation family.
- We introduce capabilities in the Pipeline Builder API such as support for streaming video, cascaded model pipelines and a scheduling API.
- DMA support in virtual machines (Linux and Windows).
- New documentation portal launched at [docs.axelera.ai](https://docs.axelera.ai/), covering Voyager SDK and all Axelera AI hardware products.

### Release Qualification

This is a production-ready release of Voyager SDK. Software components and features that are in development are marked with one of the following maturity labels:

- **Experimental:** May change or be removed without notice; no support guarantees.
- **Alpha:** Usable but incomplete; breaking changes possible.
- **Beta:** Feature-complete but not fully stable; committed to developing this further in future releases.

## Installation and Compatibility

### Installation

- `pip install` as root now supported. Installation via pip with superuser privileges is now fixed.
- `axelera-devkit` now supports PyTorch versions 2.7–2.12 (previously 2.7–2.10).
- `axelera-devkit` now supports NumPy versions 1.x and 2.x. Note: `axelera-devkit[all]` still pins NumPy to `< 2.0.0`.

### Release Compatibility Matrix

The following compatibility matrix describes the recommended and supported versions of firmware and driver per Voyager SDK release. Consult it before installing or upgrading a card or Voyager SDK release.

> **Tip!** `axversion` outputs the SDK version and `axversion --driver` outputs the driver version. `axdevice` outputs the firmware and board controller firmware version.

Recommended = version shipped with the SDK release.

Supported = versions tested with the SDK release.

Note: Other versions may work but are not actively tested. An upgrade of the card's flashed firmware and board controller firmware to a compatible version using `axdevice interactive_flash_update` script is advised.

| Release | Board controller (Recommended) | Board controller (Supported) | Flashed Firmware (Recommended) | Flashed Firmware (Supported) | PCIe driver - Linux (Recommended) | PCIe driver - Linux (Supported) | PCIe driver - Windows (Recommended) | PCIe driver - Windows (Supported) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| v1.6.0 | 7.4 | 7.0 | 1.6.0 | 1.5.0<br>1.4.0 | 1.4.16 | 1.4.10<br>1.4.4 | 1.3.4 | 1.3.1<br>1.3.0 |
| v1.6.1 | 7.4 | 7.0 | 1.6.0 | 1.5.0<br>1.4.0 | 1.4.17 | 1.4.16<br>1.4.10<br>1.4.4 | 1.3.11 | 1.3.4<br>1.3.1<br>1.3.0 |
| v1.7.0 | 7.4 | 7.0 | 1.7.0 | 1.6.0<br>1.5.0<br>1.4.0 | 1.5.7 | 1.5.5 | 1.3.11 | 1.3.5<br>1.3.4<br>1.3.1<br>1.3.0 |

### Metis M.2 Max upgrade notes

1. [Metis M.2 Max](https://axelera.ai/ai-accelerators/metis-m2-ai-acceleration-card) is required to be updated to recommended board controller and firmware versions. The `axdevice interactive_flash_update` script handles board variant selection automatically. Enable the average power controller if deploying on hosts with limited power delivery. See the [Power Management Guide](docs/user-guides/thermal.md).

## New Features / Support

### New Axelera AI Cards and Systems

- Support for Metis 1-chip PCIe Rev 2.0 cards with 2GB and 8GB DDR memory configurations.
- Default device power limit on Metis M.2 Max lowered from 11.0 W to 8.5 W for broader host compatibility. When paired with hosts which are compliant with PCIe SIG M.2 Specification Revision 4.0 power rating or higher, increasing the power limit setting results in potential performance gains. Users are recommended to experiment in steps of 0.1 W up to 11 W. On the other hand, configuring the power limit setting to a lower value enables M.2 Max in power-constrained hosts by trading off performance e.g. a power limit setting of 4 W when paired with embedded SBCs.

### Host Platform Support

#### Validated hardware platforms

- OnLogic K801 (12th Gen Intel Core-i)

The full list of Validated Host Systems for Axelera Metis Cards is available [here](https://www.axelera.ai/metis-evaluation-kit).

#### Operating Systems

- Windows IoT native support for Metis on x86 host machines.
- The standalone `axelera-rt` wheel with ManyLinux support remains the recommended installation path for Yocto images. Refer to the [meta-axelera](https://github.com/axelera-ai-hub/meta-axelera) Yocto layer for the recommended recipe structure.

#### Virtualization support

- Production ready for Metis PCIe passthrough in VMs, with the entire runtime stack including the driver running inside the guest. The PCIe driver now supports DMA when running inside a virtual machine. This enables PCIe passthrough for Metis even in VMs that have not been configured with multi-MSI support.

### New Networks Supported

Voyager SDK model zoo includes computer vision tasks and LLMs. For a full list of supported models and data about their performance and accuracy see [here](docs/reference/models/model-zoo.md).

Models that are supported but not included in the model zoo are documented [here](docs/reference/models/additional-models.md).

For convenience, pre-compiled models are available to download by running `axdownloadmodel` in the parent folder of Voyager SDK.

The release includes new YAML files for all new models offered in our model zoo in this release (see tables below).

#### New models for Image Classification

| Model Name | Resolution | Format |
| --- | --- | --- |
| [MobileNetV3-small](ax_models/zoo/torchvision/classification/mobilenetv3_small-imagenet.yaml) | 224x224 | PyTorch |
| [MobileNetV3-large](ax_models/zoo/torchvision/classification/mobilenetv3_large-imagenet.yaml) | 224x224 | PyTorch |

#### New models for Object Detection

| Model Name | Resolution | Format |
| --- | --- | --- |
| [YOLOv4](ax_models/zoo/yolo/object_detection/yolov4-416-coco.yaml) | 416x416 | Darknet |
| [YOLOv4-CSP-Leaky](ax_models/zoo/yolo/object_detection/yolov4-csp-leaky-coco.yaml) | 640x640 | Darknet |

The YAML model deployment and Pipeline Builder now support the **Darknet** model format.

#### New models for Semantic Segmentation

| Model Name | Resolution | Format |
| --- | --- | --- |
| [YOLO26n-Seg](ax_models/zoo/yolo/semantic_segmentation/yolo26nsem-cityscapes-onnx.yaml) | 1024x1024 | ONNX |
| [YOLO26s-Seg](ax_models/zoo/yolo/semantic_segmentation/yolo26ssem-cityscapes-onnx.yaml) | 1024x1024 | ONNX |
| [YOLO26m-Seg](ax_models/zoo/yolo/semantic_segmentation/yolo26msem-cityscapes-onnx.yaml) | 1024x1024 | ONNX |
| [YOLO26l-Seg](ax_models/zoo/yolo/semantic_segmentation/yolo26lsem-cityscapes-onnx.yaml) | 1024x1024 | ONNX |
| [YOLO26x-Seg](ax_models/zoo/yolo/semantic_segmentation/yolo26xsem-cityscapes-onnx.yaml) | 1024x1024 | ONNX |

Trained on the Cityscapes dataset. Note: model zoo FP32 accuracy uses a 1024×1024 input dimension aligned with Cityscapes, which differs from the Ultralytics default of 1024×2048; reported accuracy numbers will differ accordingly.

The YAML Pipeline Builder supports both Accuracy mode and Performance mode for semantic segmentation models.

### AI Pipeline Builder

#### \[Alpha\] Pipeline Builder API

The Python-native Pipeline Builder API that was introduced in v1.6 now gains a streaming runtime and many new capabilities.

**Streaming and scheduling**

- a new `Scheduler` owns model instances and connections, caches loaded models, and distributes work across the available AIPU cores.
- `pipeline.stream` pipelines frames across cores automatically. Per-model priority can be tuned via the `core_allocation` argument to `op.load`. A single-core option runs each frame end-to-end in the calling thread for easy operator stepping/debugging.
- `pipeline.batch` allows optimised scheduling of multiple inputs without having to use asynchronous APIs.

#### Video decode and sources

- A built-in video decoder binding for ffmpeg and OpenCV with a zero-copy streaming API. `cv.create_source` enables accelerated decoding into a new `Image` class that enables zero-copy whilst facilitating access via a `to_numpy()` read only view.

#### New task types

- Depth estimation
- Oriented Bounding Boxes (OBB)
- Re-identification

**Other**

- Multi-level cascade pipelines now track coordinates correctly through each stage, extended to OBB and pose keypoints, so per-object crops map back to the original frame.
- New and expanded API documentation, getting-started and model-compilation tutorials, and a coordinate-system tutorial, as well as a new set of examples in `examples/pipeline_builder/`.

**Demo Scripts**

New standalone runnable demo scripts are self-contained and depend only on `axelera.runtime`, providing a ready starting point for building custom inference pipelines.

| Demo Script | Description | Model |
| --- | --- | --- |
| `classification.py` | ImageNet top-5 classification with standard ImageNet preprocessing. | `squeezenet1.0-imagenet.axm` |
| `detection.py` | YOLOv8 object detection on COCO (80 classes) with NMS. | `yolov8n-coco.axm` |
| `detection_vary_confidence.py` | Object detection with runtime confidence threshold variation every 60 frames. | `yolov8n-coco.axm` |
| `pose_detection.py` | YOLOv8 human pose estimation with 17 COCO keypoints. | `yolov8npose-coco.axm` |
| `segmentation.py` | YOLOv8 instance segmentation with prototype-based mask prediction. | `yolov8nseg-coco.axm` |
| `depth_estimation.py` | Monocular depth estimation with FastDepth (NYU Depth V2). | `fastdepth-nyudepthv2-onnx.axm` |
| `obb.py` | YOLO11n oriented-bounding-box detection on DOTA (15 classes). | `yolo11n-obb-dotav1-onnx.axm` |
| `tracking.py` | Multi-object tracking with state lifecycle, detection correlation, and class filtering. | `yolov8n-coco.axm` |
| `tracking_with_classification.py` | Detection → filtering → tracking → per-track classification cascade. | `yolov8n-coco.axm`, `squeezenet1.0-imagenet.axm` |

#### YAML Pipeline Builder

- Per-stream crop: crop configuration can be set independently per stream via the usage `./inference.py yolov8n-coco crop[left,top,width,height]:rtsp://...`
- Robust ONNX preamble: NHWC support, new transform operators, output validation, and an optimizer fix, so extracted preprocessing matches the original model more reliably.
- Decoder improvements: faster fused semantic-segmentation decoder, additionally improves accuracy over the decoder in 1.6.
- fp16 (half-precision) support added to OpenCL kernels where hardware support is available.
- DMA-buf import/export via a Khronos extension `cl_khr_external_memory` for copy-free OpenCL interoperability on supported platforms.
- Multiplanar image support enabled for OpenCL.
- OpenCL 3.0 compatibility.

### \[Beta\] Model Compiler

- Framework and dependency support expanded:
    - PyTorch 2.10–2.12 support added across the compiler toolchain; supported range is now 2.7–2.12.
    - NumPy 2.x compatibility across `axelera-tvm`, `qtools`, and `onnx2torch`.
    - Python 3.13 wheel validation and aarch64 wheel builds.
- Unique `.bin` file paths during compilation, preventing collisions when multiple models are compiled into the same output directory.
- Quantization accuracy improvement: Hardswish LUT updated to v4 (uniform 16-bin) for better activation-quantization accuracy.

### Tools

- `axWinGrantLargePages.exe` (Windows): New utility that grants the Large Pages privilege to improve DMA performance. `libaxldev` will warn at runtime if this has not been run.
- `axmonitor` improvements:
    - Per-sensor power statistics now available for Metis 4-chip PCIe card and Metis Compute Board.
    - MVM utilization and stack usage now reported per AI core.

## Breaking Changes

None

## Known Issues and Limitations

### IMPORTANT - Memory leak in GStreamer software video decode (`gst-libav` 1.24.2)

Pipelines that perform software H.264 decoding via GStreamer's `avdec_h264` element (`gst-libav` 1.24.2, as shipped with GStreamer 1.24.2 on Ubuntu 24) leak approximately **104 bytes per decoded video frame, per decoder**. This is a defect in `gst-libav` itself (`avdec`/`avviddec` does not free the per-frame `AVPacket`), not in the Voyager SDK, and the SDK cannot workaround it.
 
The leak grows linearly with runtime and scales with the number of decoded streams, so it is most noticeable in long running, multi-stream deployments. For example, 16 streams at 30 FPS leak roughly 50 KiB/s in aggregate.
 
**Impact:** affects only pipelines that use GStreamer software video decode. Hardware-accelerated decode paths are not affected.
 
**Mitigation:** the only fix is to upgrade GStreamer / `gst-libav` to **1.28.3 or later**, which resolves the underlying `AVPacket` leak. Or alternatively where possible, prefer hardware-accelerated decode over software `avdec_h264`. Restarting long-running pipelines periodically bounds the resident memory growth.


### Other

- Performance variability is observed on certain hosts. Inference-only performance (FPS) drops of up to 5-10% is observed on `yolox-x-crowdhuman-onnx` and `yolo11l-obb-dotav1-onnx` compared to SDK Release v1.5.
- Numpy compatibility: `axelera-devkit` supports only NumPy 1.x APIs and is not compatible with NumPy 2.x. `axelera-rt` supports both NumPy 1.x and 2.x but constrains the `numpy` dependency to prevent breaking `axelera-devkit` on Linux. The Windows runtime environment supports NumPy 2.x.
- Python 3.13 incompatibility with wheel installer on Ubuntu 24.04. Not reproducible with Python 3.12 (default for Ubuntu 24.04).
- Metis Compute Board video output rendering is choppy (1–2 FPS). This impacts rendering to display only, inference performance is not impacted.
- YOLO26\* models (e.g. `yolo26x-obb-dotav1-onnx`) sometimes fail to deploy unexpectedly.
- MobileNetV3 may yield degraded accuracy on some combinations of hosts and cards.
- Device monitoring with AxMonitor is not supported on single-MSI hosts. For some systems with single-MSI hosts, device monitoring with `AxMonitor` does not display any data. An example of a host with this issue is Arduino Portenta X8 Mini.
- On hosts with multiple Axelera cards, you may see the error `[libaxldev_linux.c:1889] AXL_IOCTL_FWTRACE_OPEN_SESSION failed: Cannot allocate memory`. If you encounter this, upgrade your device driver by running `axdevice driver --install 1.5.7`.
- Cards with flashed firmware v1.3.2 and board controller firmware v1.4 must be upgraded to the v1.7.0 recommended versions (see the [Release Compatibility Matrix](#release-compatibility-matrix)).
- Since v1.7.0 was released, a new `scipy` release relaxed its NumPy version constraints such that NumPy 2.4+ may be installed (whether this happens depends on the other packages in your environment and your Python version). This can cause errors during model compilation, either via `scipy` or directly in the Axelera model compiler, for example:
  ```
  ERROR   : module 'numpy' has no attribute 'long'
  ERROR   : TypeError: only 0-dimensional arrays can be converted to Python scalars
  ```
  The workaround is to explicitly install an older NumPy version:
  ```bash
  pip install "numpy<2.4"
  ```

## System Requirement

### Development Environment

For model compiling purposes, these are the host requirements:

| Requirement | Detail |
| --- | --- |
| OS | Linux Ubuntu 22.04, Ubuntu 24.04, Docker (on Windows or Linux), Windows + WSL/Ubuntu |
| CPU architecture | ARM64, x86, x86_64 |
| Recommended CPU | Intel Core-i5 or equivalent |
| Minimum System Memory | 16 GB (large models may require swap partition) |
| Recommended System Memory | 32 GB |

### Runtime Environment

This release is expected to work with Intel (x86), AMD (x86) and Arm64 host CPUs. See [here](https://support.axelera.ai/hc/en-us/articles/34274775900178-Validated-Host-Systems-for-Axelera-Metis-AI-Accelerator-Cards) for a list of validated host systems for Axelera Metis AI Accelerator Cards.

## Further Support

- For blog posts, projects and technical support please visit [Axelera AI Customer Portal](https://support.axelera.ai/).
- For technical documents and guides please visit [docs.axelera.ai](https://docs.axelera.ai/).

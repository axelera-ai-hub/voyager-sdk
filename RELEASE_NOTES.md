# Voyager SDK release notes v1.6

- [Voyager SDK release notes v1.6](#voyager-sdk-release-notes-v16)
    - [Voyager SDK release notes v1.6.1](#voyager-sdk-release-notes-v161)
        - [Fixed Issues Since v1.6.0](#fixed-issues-since-v160)
        - [New Features / Support Since v1.6.0](#new-features--support-since-v160)
            - [Windows](#windows)
            - [Metis Hardware](#metis-hardware)
            - [Pipeline Builder](#pipeline-builder)
            - [Tools](#tools)
        - [Document Updates Since v1.6.0](#document-updates-since-v160)
    - [Voyager SDK release notes v1.6.0](#voyager-sdk-release-notes-v160)
    - [Release Qualification](#release-qualification)
    - [New Features / Support](#new-features--support)
        - [New Axelera AI Cards and Systems](#new-axelera-ai-cards-and-systems)
        - [Host Platform Support](#host-platform-support)
        - [New Networks Supported](#new-networks-supported)
            - [New models for Object Detection](#new-models-for-object-detection)
            - [New models for Instance Segmentation](#new-models-for-instance-segmentation)
            - [New models for Keypoint / Pose Detection](#new-models-for-keypoint--pose-detection)
            - [New models for Oriented Bounding Boxes Object Detection](#new-models-for-oriented-bounding-boxes-object-detection)
            - [New models for Re-Identification](#new-models-for-re-identification)
        - [End-to-End Pipelines](#end-to-end-pipelines)
        - [Installation](#installation)
        - [AI Pipeline Builder](#ai-pipeline-builder)
            - [YAML Pipeline Builder](#yaml-pipeline-builder)
            - [\[Alpha\] Pipeline Builder API](#alpha-pipeline-builder-api)
            - [AxLLM](#axllm)
        - [Beta Model Compiler](#beta-model-compiler)
        - [Runtime](#runtime)
        - [Tools](#tools-1)
        - [Firmware](#firmware)
    - [Breaking Changes](#breaking-changes)
    - [Fixed Issues Since Last Release](#fixed-issues-since-last-release)
    - [Known Issues and Limitations](#known-issues-and-limitations)
    - [System Requirement](#system-requirement)
        - [Development Environment](#development-environment)
        - [Runtime Environment](#runtime-environment)
    - [Further Support](#further-support)

## Voyager SDK release notes v1.6.1

This release is a patch on top of v1.6.0, delivering improvements to Windows performance, expanded hardware support, pipeline stability fixes, and a new documentation portal at [docs.axelera.ai](https://docs.axelera.ai/).

### Fixed Issues Since v1.6.0

- Fixed closed-loop power control on Metis M.2 Max:
    - The feature no longer results in unexpected performance degradation.
    - The user-set power limit (e.g. via `axdevice --set-power-limit`) now matches values reported by `axmonitor` at runtime. Previously, reported values could clamp below the configured limit.
    - Default power limit setting is changed from 11 W to 8.5 W (see [Metis Hardware](#metis-hardware)).
- Fixed performance degradation when writing output video to disk. Multi-stream video output can now be written without throughput drop, including cases with uneven per-stream frame distribution.
- Fixed multiple-pipelines example to work with USB/video sources.
- Fix for `libaxldev` "Failed to detach int ioctl" errors on some ARM hosts.
- Fixed `axmonitor` to report the correct Avg/Min/Max power from the on-board power sensor on 4-chip PCIe cards.
- Restored the `--transform` option of `axcompile`.
- Fixed OpenCL paths producing out-of-bounds reads/writes when the upstream element passes multi-memory `gst_memory` buffers from `OpenCLAllocator`.
- Fixed the Face-recognition pipeline which was failing with "Failed to set kernel argument 7, Invalid arg size".
- Fixed segmentation fault at end of inference with tiled inference on 4-chip PCIe with multiple streams.
- Fixed error case on Windows where `axrunmodel yolox-x-crowdhuman-onnx` triggered "AssertionError: Arrays are not equal".
- Fixed accuracy testing on private datasets where the `.ax_dataset_complete` stamp was not being created.
- Fixed the examples `cross_line_count.py` and `remote_cross_line_monitor.py` terminal hangs when the display window was closed via the X button.
- Fixed OpenGL renderer failing on NV16 image sources; corrected rendering for YUY2 and grayscale sources.
- Fixed `render_to_wx.py` performance; previously it was extremely slow (~4 min when deploying a model) when adding a stream.
- Removed the examples `examples/stream_select.py` and `examples/multiple_pipeline.py` in favour of `render_to_wx.py`.
- Fixed multiple semantic segmentation issues: gst-path thresholding on multi-class segmentation, incorrect output-tensor transpose and dataloader sizes on single-class segmentation, incorrect default sigmoid in the gst decoder, "Invalid Value" being printed for a valid accuracy_mean, missing prediction field in binary segmentation, and a missing resize in preprocessing.
- Fixed blank streams when saving the output of multiple streams; sources beyond the first were saved as blank due to a source-ID mismatch in the save-output draw.
- Fixed `pipeline.add_source` failing because of a stale reference to a legacy env var.
- Fixed architecture detection when building operators on Debian 12.
- Fixed firmware upgrade on Raspberry Pi 5 with Metis M.2.

### New Features / Support Since v1.6.0

#### Windows

- Performance improvements in the PCIe driver for Windows and in `libaxldev`, closing the gap with Linux.
- Improved logging and tracing infrastructure for higher-resolution tracing and higher stability, including a utility for viewing traces.
- **\[Breaking change\] Driver compatibility**: Voyager SDK on Windows now requires Metis Driver 1.3.2 or newer. Drivers v1.3.1 and earlier are no longer supported. This release ships with Windows driver 1.3.11.

#### Metis Hardware

- **Metis M.2 Max host dependent setting:** When using a Metis M.2 Max card, the closed loop power control is enabled with a default power limit of 8.5 W. This default value is biased towards the average workstation host PC. When paired with hosts which are compliant with PCIe SIG M.2 Specification Revision 4.0 power rating or higher, increasing the power limit setting results in potential performance gains. Users are recommended to experiment in steps of 0.1 W up to 11 W. On the other hand, configuring the power limit setting to a lower value enables M.2 Max in power-constrained hosts by trading off performance e.g. a power limit setting of 4 W when paired with embedded SBCs.
- Metis PCIe Rev.2 boards with **2 GB** and **8 GB** DDR memory configurations are now supported.

#### Pipeline Builder

- `axstreamer` VideoDecode now supports automatic colour format selection (`color_format=auto`), enabling the decoder to select its native output format. This avoids unnecessary colour conversions and enables fusion of colour-convert into resize operators.
- Added support for downstream OpenCL-based Resize and Colour Conversion plugins for I420-formatted images.

#### Tools

- `axrunmodel` now displays a live progress bar when stdin is a TTY (via `alive_progress`).

### Document Updates Since v1.6.0

- New documentation portal launched at [docs.axelera.ai](https://docs.axelera.ai/), covering Voyager SDK and all Axelera AI hardware products.

## Voyager SDK release notes v1.6.0

Voyager SDK v1.6.0 introduces support for new Axelera hardware, host platforms and operating systems. The release expands installation flexibility, pipeline APIs, the model zoo and end-to-end pipelines, and development tools.

- Metis M.2 Max support, delivering PCIe-class performance in the M.2 form factor.
- Standalone Python wheels with ManyLinux support for broader Linux distribution coverage.
- New Pipeline Builder API for defining and executing pipelines from Python.

## Release Qualification

This is a production-ready release of Voyager SDK. Software components and features that are in development are marked with one of the following maturity labels:

- **Experimental:** May change or be removed without notice; no support guarantees.
- **Alpha:** Usable but incomplete; breaking changes possible.
- **Beta:** Feature-complete but not fully stable; committed to developing this further in future releases.

## New Features / Support

### New Axelera AI Cards and Systems

- The release adds \[Beta\] support for [Metis M.2 Max](https://axelera.ai/ai-accelerators/metis-m2-ai-acceleration-card) engineering samples.

### Host Platform Support

#### Validated hardware platforms

- Dell Pro Slim Plus XE5 (Intel Core Ultra Series 2).
- AsRock NUC Box-125 (Intel Core Ultra Series 1).
- Lenovo P3 Tiny Gen2 (Intel Core i)
- Kontron KISS 1U V4 ADL 1U (Intel Core i)
- Supermicro SYS-322GA-NR (Intel Xeon 6)

The full list of Validated Host Systems for Axelera Metis Cards is available [here](https://support.axelera.ai/hc/en-us/articles/34274775900178-Validated-Host-Systems-for-Axelera-Metis-AI-Accelerator-Cards).

#### Operating Systems

- Yocto layer (`meta-axelera`) for integrating Axelera AI hardware in custom Yocto distributions available at [meta-axelera](https://github.com/axelera-ai-hub/meta-axelera), removing the need for manual integration.
- Yocto build sources for the Metis Compute Board are available at [axelera-aisbc-bsp](https://github.com/axelera-ai-hub/axelera-aisbc-bsp), enabling users to build a custom Linux image for the single-board computer.
- \[Beta\] Debian 12 and 13 on x86 host systems.
- \[Beta\] Red Hat Enterprise Linux 9 and 10 on x86 host systems.
- \[Alpha\] Runtime environment supported (refer to `axelera-rt` in Installation section) on Yocto-based Linux using the new Pipeline Builder API.

#### Virtualization support

\[Beta\] PCIe passthrough in KVM virtual machines. Metis devices can be passed through from the host to a VM, with the entire runtime stack including the host driver running inside the guest.

### New Networks Supported

Voyager SDK model zoo includes computer vision tasks and LLMs. For a full list of supported models and data about their performance and accuracy see [here](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/docs/reference/model_zoo.md).

Models that are supported but not included in the model zoo are documented [here](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/docs/reference/additional_models.md).

For convenience, pre-compiled models are available to download by running `axdownloadmodel` in the parent folder of Voyager SDK.

#### New models for Object Detection

| Model Name | Resolution | Format |
| --- | --- | --- |
| [GELAN-S](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/zoo/yolo/object_detection/gelan-s-coco-onnx.yaml) | 640x640 | ONNX |
| [GELAN-M](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/zoo/yolo/object_detection/gelan-m-coco-onnx.yaml) | 640x640 | ONNX |
| [GELAN-C](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/zoo/yolo/object_detection/gelan-c-coco-onnx.yaml) | 640x640 | ONNX |
| [YOLO26-X](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/zoo/yolo/object_detection/yolo26x-coco-onnx.yaml) | 640x640 | ONNX |
| [YOLO-NAS S](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/zoo/yolo/object_detection/yolonas-s-coco-onnx.yaml) | 640x640 | ONNX |
| [YOLO-NAS M](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/zoo/yolo/object_detection/yolonas-m-coco-onnx.yaml) | 640x640 | ONNX |
| [YOLO-NAS L](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/zoo/yolo/object_detection/yolonas-l-coco-onnx.yaml) | 640x640 | ONNX |

The GELAN family (Generalized Efficient Layer Aggregation Network) is the architecture underlying YOLOv9. YOLO-NAS (Neural Architecture Search) models are Deci AI's accuracy-optimised architecture with quantisation-aware blocks.

#### New models for Instance Segmentation

| Model Name | Resolution | Format |
| --- | --- | --- |
| [YOLO26-N Seg](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/zoo/yolo/instance_segmentation/yolo26nseg-coco-onnx.yaml) | 640x640 | ONNX |
| [YOLO26-S Seg](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/zoo/yolo/instance_segmentation/yolo26sseg-coco-onnx.yaml) | 640x640 | ONNX |
| [YOLO26-M Seg](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/zoo/yolo/instance_segmentation/yolo26mseg-coco-onnx.yaml) | 640x640 | ONNX |
| [YOLO26-L Seg](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/zoo/yolo/instance_segmentation/yolo26lseg-coco-onnx.yaml) | 640x640 | ONNX |
| [YOLO26-X Seg](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/zoo/yolo/instance_segmentation/yolo26xseg-coco-onnx.yaml) | 640x640 | ONNX |

#### New models for Keypoint / Pose Detection

| Model Name | Resolution | Format |
| --- | --- | --- |
| [YOLO26-N Pose](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/zoo/yolo/keypoint_detection/yolo26npose-coco-onnx.yaml) | 640x640 | ONNX |
| [YOLO26-S Pose](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/zoo/yolo/keypoint_detection/yolo26spose-coco-onnx.yaml) | 640x640 | ONNX |
| [YOLO26-M Pose](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/zoo/yolo/keypoint_detection/yolo26mpose-coco-onnx.yaml) | 640x640 | ONNX |
| [YOLO26-L Pose](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/zoo/yolo/keypoint_detection/yolo26lpose-coco-onnx.yaml) | 640x640 | ONNX |
| [YOLO26-X Pose](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/zoo/yolo/keypoint_detection/yolo26xpose-coco-onnx.yaml) | 640x640 | ONNX |

#### New models for Oriented Bounding Boxes Object Detection

| Model Name | Resolution | Format |
| --- | --- | --- |
| [YOLO26-N OBB](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/zoo/yolo/obb_detection/yolo26n-obb-dotav1-onnx.yaml) | 640x640 | ONNX |
| [YOLO26-S OBB](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/zoo/yolo/obb_detection/yolo26s-obb-dotav1-onnx.yaml) | 640x640 | ONNX |
| [YOLO26-M OBB](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/zoo/yolo/obb_detection/yolo26m-obb-dotav1-onnx.yaml) | 640x640 | ONNX |
| [YOLO26-L OBB](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/zoo/yolo/obb_detection/yolo26l-obb-dotav1-onnx.yaml) | 640x640 | ONNX |
| [YOLO26-X OBB](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/zoo/yolo/obb_detection/yolo26x-obb-dotav1-onnx.yaml) | 640x640 | ONNX |

#### New models for Re-Identification

| Model Name | Resolution | Format |
| --- | --- | --- |
| [SBS-S50](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/zoo/torch/sbs-s50-market1501-onnx.yaml) | 256x128 | ONNX |

SBS-S50 is a Re-ID backbone used with the Deep-OC-SORT tracker, enabling the full re-identification tracking pipeline on Axelera hardware.

### End-to-End Pipelines

- **New YAML files:** The release includes new YAML files for all new models offered in our model zoo in this release (see tables above).
- **New parallel multi-model reference pipelines:** These demonstrate how to run multiple detection, segmentation, and face detection models simultaneously on a single input stream.
    - [parallel-yolo11-pose-seg-retinaface.yaml](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/reference/parallel/parallel-yolo11-pose-seg-retinaface.yaml) - simultaneous detection, pose, segmentation, and face detection.
    - [parallel-fastsams-retinaface-yolo11.yaml](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/reference/parallel/parallel-fastsams-retinaface-yolo11.yaml) - simultaneous segmentation, face detection, and object detection.
- **New examples:**
    - [render_to_wxpython.py](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/examples/render_to_wxpython.py) - GUI for streaming inference with live network switching (replaces render_to_ui). Supports switching between models without restarting the pipeline.
    - [axinferencenet_local_plugin.cpp](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/examples/axinferencenet/axinferencenet_local_plugin.cpp) - raw tensor access for custom postprocessing in C++.
- **Enhanced multi-object tracking features:**
    - TrackTrack - a state-of-the-art multi-object tracking algorithm using iterative matching with track-aware NMS (CVPR 2025). Full implementation in C++ with Python bindings.
    - Re-identification pipeline: Deep-OC-SORT with SBS-S50 re-identification backbone. See [deep-oc-sort-sbs50-onnx.yaml](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/model_cards/torch/deep-oc-sort-sbs50-onnx.yaml).
    - Camera motion compensation (CMC) support for (Deep-)OC-SORT, improving tracking under camera movement. See [yolox-deep-oc-sort-cmc-osnet.yaml](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/reference/cascade/with_tracker/yolox-deep-oc-sort-cmc-osnet.yaml).
    - \[Experimental\] Memory Bank: Enables the tracker to restore a person's ID after they leave the scene and later reappear. See [yolox-deep-oc-sort-osnet-membank.yaml](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/ax_models/reference/cascade/with_tracker/yolox-deep-oc-sort-osnet-membank.yaml).

### Installation

This release introduces new installation options including standalone Python wheels and a unified PyPI index (see tutorials/install_pip.md). The existing `install.sh` installer remains stable and available as a fallback.

- **Python wheels:** The SDK is now split into runtime and compilation environments delivered as standalone Python wheels: `axelera-rt` and `axelera-devkit` respectively.
    - `axelera-rt` - runtime environment (inference, device management, monitoring).
    - `axelera-devkit` - compilation and quantization environment.
    - Supported Python versions: 3.10, 3.11, 3.12, 3.13.
    - ManyLinux support: the wheels can be installed across different Linux distributions (Debian 12 & 13, Red Hat Enterprise Linux 9 & 10) and Yocto images without the Axelera installer script.
- **New PyPI index:** Python packages are now indexed by a new, single Axelera-hosted index (`https://software.axelera.ai/artifactory/axelera-pypi/`) which replaces the previous two indexes (`axelera-dev-pypi` and `axelera-runtime-pypi`). Any installation scripts or pip configuration files referencing the old URLs must be updated.
- **Expanded PyTorch support:** The `axelera-devkit` now supports PyTorch versions 2.7 through 2.10 (previously limited to 2.9 and below).

#### New and improved driver

- PCIe Linux driver source is publicly available at [axelera-driver](https://github.com/axelera-ai-hub/axelera-driver), enabling building the kernel module.
- \[Beta\] The host driver supports Debian 12 and 13 kernels.
- \[Beta\] The host driver supports Red Hat Enterprise Linux 9 and 10 kernels. The user may build RPM packages from the driver repository.
- See driver install using `axdevice` under the Tools section.

### AI Pipeline Builder

#### YAML Pipeline Builder

New features have been added to the YAML Pipeline Builder:

- **Multi-stream tiling**:
    - Tiling pipelines now support multiple camera sources simultaneously, enabling tiled inference for each stream for high-resolution input processing.
    - `tile[...]:source` syntax for different tiling configurations per camera.
    - Automatic pipeline construction for tiled multi-stream scenarios.
- **OpenCL acceleration**: Face alignment, color conversion (NV16, BGR/RGB), polar transforms, and ROI cropping on GPU.
- **DMA buffer passthrough**: `AxInferenceNet` accepts dmabuf inputs directly, avoiding memory copies on ARM. This is particularly valuable for Metis Compute Board deployments where camera and display share DMA buffers.
- **YOLO decoder unification**: Consolidated YOLOv8/seg/OBB decoders into single implementation, reducing code duplication. Future YOLO decoder improvements now benefit all YOLO task types.
- **ONNX preamble integration**: Automatic extraction of preprocessing constants from ONNX graphs into pipeline operators. This ensures the pipeline exactly matches the original model's preprocessing without manual transcription.
- **Buffer pool management**: Improved sizing for batched workloads, queue fixes for Nvidia Jetson Orin platforms.
- **Performance optimizations** result in up to 50% end-to-end performance improvement for high-throughput model pipelines, as measured on an Intel Core-i5 platform.
- **Loop and Frame Rate on images** `image_dir` inputs: Users can now enable looping and set a frame rate limit, making it easier to run continuous benchmarks on a fixed image dataset.

#### \[Alpha\] Pipeline Builder API

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

#### AxLLM

- Added support for Gradio 6.x (in addition to Gradio 5.x), ensuring the LLM demo UI compatibility for both versions.

### \[Beta\] Model Compiler

- **TOML configuration format**: The compiler CLI now generates default configuration files in TOML format, making model-specific configurations more readable and editable. The `compile --generate-config` command outputs `default_conf.toml`. Existing JSON configs remain supported for backwards compatibility; use `compile --generate-config --config-format json` to generate JSON.
- Model configurations are now grouped by model family, simplifying multi-model setups.
- **AXM output format**: Added support for generating AXM files (self-contained archive) as a compilation output, simplifying model deployment.

### Runtime

### Tools

- `axdownloadmedia`: A new command-line utility to list and download Axelera-provided media files (test videos, images) from cloud storage.
- `axdevice` improvements:
    - `axdevice driver --install`: New option for Debian-based systems to automatically install or download the Axelera PCIe device driver, simplifying first-time setup. Without `--install` the tool shows the current status of the driver.
    - `axdevice` firmware flashing: `axdevice` now includes firmware flash subcommands (`axdevice interactive_flash_update`, `axdevice interactive_flash_downgrade`), integrating firmware flashing into the standard `axdevice` workflow.
    - `axdevice` power limit display improvements: `--set-power-limit` now accepts `off` or `0` to explicitly disable throttling. Relevant only for M.2 Max boards.
- `axmonitor` improvements:
    - `axmonitor` DDR bandwidth measurement is now available with a plot in the OVERVIEW page, helping users understand whether workloads are DDR-bandwidth-bound vs. compute-bound.
    - `axmonitor` power measurements extended (M.2 Max and PCIe Rev2 boards only) with min, max, and average values over a 1-second window.

### Firmware

- \[Beta\] New board controller firmware release supporting Metis M.2 Max.
- \[Beta\] **Closed loop power control** supported for M.2 Max. This mechanism trades off peak compute throughput for predictable power envelope, allowing safe operation in thermally or electrically constrained environments such as M.2 form-factor boards. Users can set the power limit with `axdevice --set-power-limit LIMIT` where `LIMIT` is an integer representing watts. See the [Power Management Guide](https://github.com/axelera-ai/application.framework/blob/main/docs/reference/thermal_and_power_guide.md#2-power-management-guide).
- **Power saving when idle**: When an AI-core is idle for 1 second or more, the core frequency is automatically reduced. This saves ~0.6W total across all four cores with no performance impact.

## Breaking Changes

- **New PyPI index**: The previous two PyPI indexes (`axelera-dev-pypi` and `axelera-runtime-pypi`) have been replaced by a single index (`https://software.axelera.ai/artifactory/axelera-pypi/`). Any installation scripts or pip configuration files referencing the old URLs must be updated.
- **TOML default for compiler configuration**: The compiler CLI now generates TOML configuration files by default instead of JSON. Existing JSON configs remain supported; use `compile --generate-config --config-format json` to generate JSON.

## Fixed Issues Since Last Release

- Installer tool's docker option not working (SDK-8083): Running `install.sh --docker` fails on certain configurations, for example on Firefly ITX-3588J motherboard.

## Known Issues and Limitations

- Performance variability is observed on certain hosts. Inference-only performance (FPS) drops of up to 5-10% is observed on `yolox-x-crowdhuman-onnx` and `yolo11l-obb-dotav1-onnx` compared to SDK Release v1.5.
- Numpy compatibility: `axelera-devkit` supports only NumPy 1.x APIs and is not compatible with NumPy 2.x. `axelera-rt` supports both NumPy 1.x and 2.x but constrains the `numpy` dependency to prevent breaking `axelera-devkit` on Linux. The Windows runtime environment supports NumPy 2.x.
- Python 3.13 incompatibility with wheel installer on Ubuntu 24.04. Not reproducible with Python 3.12 (default for Ubuntu 24.04).
- When using `inference.py`, performance degrades when writing output video to disk.
- Metis Compute Board video output rendering is choppy (1–2 FPS). This impacts rendering to display only, inference performance is not impacted.
- YOLO X models (e.g. `yolo26x-obb-dotav1-onnx`) sometimes fails to deploy unexpectedly.
- MobileNetV3 may yield degraded accuracy on some combinations of hosts and cards.
- \[Beta\] Closed-loop power control on M.2 Max may cause unexpected performance degradation. For maximum performance, use a host system whose M.2 slot is rated to PCI-SIG M.2 Specification Revision 4.0 or higher (e.g. AsRock NUC Box-125, Seco Titan 300), where closed-loop power control is not required. On hosts with lower M.2 slot power delivery, set the MVM utilization limit via the environment variable e.g. `AXELERA_CONFIGURE_BOARD=,20` will set the MVM utilization limit to 20%.
- Device monitoring with AxMonitor is not supported on single-MSI hosts. For some systems with single-MSI hosts, device monitoring with `AxMonitor` does not display any data. An example of a host with this issue is Arduino Portenta X8 Mini.

## System Requirement

### Development Environment

For model compiling purposes, these are the host requirements:

| Requirement | Detail |
| --- | --- |
| OS | Linux Ubuntu 22.04, Ubuntu 24.04, Docker (on Windows or Linux), Windows + WSL/Ubuntu |
| CPU architecture | ARM64, x86, x86_64 |
| Recommended CPU | Intel Core-i5 or equivalent |
| Minimum System Memory | 16GB (large models may require swap partition) |
| Recommended System Memory* | 32 GB |

\*Compiling the model Real-ESRGAN-x4plus requires a machine with at least 128GB of memory.

### Runtime Environment

This release is expected to work with Intel (x86), AMD (x86) and Arm64 host CPUs. See [here](https://support.axelera.ai/hc/en-us/articles/34274775900178-Validated-Host-Systems-for-Axelera-Metis-AI-Accelerator-Cards) for a list of validated host systems for Axelera Metis AI Accelerator Cards.

## Further Support

- For blog posts, projects and technical support please visit [Axelera AI Customer Portal](https://support.axelera.ai/).
- For technical documents and guides please visit [Customer Portal](https://support.axelera.ai/).

# Voyager SDK release notes v1.8

- [Voyager SDK release notes v1.8](#voyager-sdk-release-notes-v18)
  - [Voyager SDK release notes v1.8.0](#voyager-sdk-release-notes-v180)
    - [Release Qualification](#release-qualification)
  - [Installation and Compatibility](#installation-and-compatibility)
    - [IMPORTANT - AxModel version 5: models must be rebuilt](#important---axmodel-version-5-models-must-be-rebuilt)
    - [IMPORTANT - Driver upgrade is mandatory](#important---driver-upgrade-is-mandatory)
    - [Installation](#installation)
    - [Release Compatibility Matrix](#release-compatibility-matrix)
  - [New Features / Support](#new-features--support)
    - [New Axelera AI Cards and Systems](#new-axelera-ai-cards-and-systems)
    - [Host Platform Support](#host-platform-support)
      - [Operating systems and kernels](#operating-systems-and-kernels)
    - [\[Alpha\] Model recipes for the Model Zoo](#alpha-model-recipes-for-the-model-zoo)
    - [New Networks Supported](#new-networks-supported)
    - [\[Alpha\] Pipeline Builder API](#alpha-pipeline-builder-api)
      - [New task types and operators](#new-task-types-and-operators)
      - [Video decode and sources](#video-decode-and-sources)
      - [Display and rendering](#display-and-rendering)
    - [YAML Pipeline Builder](#yaml-pipeline-builder)
      - [Demo scripts and examples](#demo-scripts-and-examples)
    - [RISC-V toolchain unification](#risc-v-toolchain-unification)
    - [Tools](#tools)
    - [Packaging](#packaging)
  - [Breaking Changes](#breaking-changes)
  - [Fixed Issues Since v1.7.0](#fixed-issues-since-v170)
  - [Known Issues and Limitations](#known-issues-and-limitations)
    - [Memory leak in GStreamer software video decode (`gst-libav` 1.24.2)](#memory-leak-in-gstreamer-software-video-decode-gst-libav-1242)
    - [Other](#other)
  - [System Requirement](#system-requirement)
    - [Development Environment](#development-environment)
    - [Runtime Environment](#runtime-environment)
  - [Further Support](#further-support)

## Voyager SDK release notes v1.8.0

- **Metis 4-chip PCIe with 8GB RAM** and **Metis chip-down designs** are supported in this release.
- **Pipeline builder API \[Alpha\]** has a more finalized API surface with video streaming and model cascading support as well as new task types and operators. The Pipeline Builder is still "alpha" with no performance guarantees; performance optimizations will be delivered in future releases.
- **Pre-compiled kernel binaries are now mandatory.** The AxModel format moves to major version 5 and the RISC-V toolchain is no longer part of a runtime installation. Existing models must be re-deployed or re-downloaded. See [Breaking Changes](#breaking-changes).
- **New: standalone Python model recipes (Alpha).** Selected Model Zoo models now ship a readable, self-contained build script under `model_recipes/` with no zoo runtime dependency, so users can copy and adapt it directly.
- **Linux kernel support extended down to 5.4** (Yocto dunfell) and up to 6.17.

### Release Qualification

This is a production-ready release of Voyager SDK. Software components and features that are in development are marked with one of the following maturity labels:

- **Experimental:** May change or be removed without notice; no support guarantees.
- **Alpha:** Usable but incomplete; breaking changes possible.
- **Beta:** Feature-complete but not fully stable; committed to developing this further in future releases.

## Installation and Compatibility

### IMPORTANT - AxModel version 5: models must be rebuilt

v1.8.0 bumps the AxModel major version from 4 to 5. Every AxModel now contains a pre-compiled kernel, which allows the runtime to execute inference without compiling kernel source at load time - and therefore allows the RISC-V toolchain to be dropped from the runtime environment entirely.

Consequence: AxModels produced by v1.7 or earlier cannot be run or inspected by v1.8. `axrunmodel`, `inference.py` and `axmodeltool` will fail with:

```
Unsupported model version: 4.0, expected at least 5.0
Please re-deploy or re-download the model.
```

Action required: re-deploy models with the v1.8 toolchain, or re-download the pre-compiled models with `axdownloadmodel`.

### IMPORTANT - Driver upgrade is mandatory

For both Linux and Windows, the minimum *supported* driver version in v1.8.0 is equal to the shipped version, so no earlier driver is accepted:

- Linux: install `metis-dkms` 1.6.2
- Windows: install MetisDriver-1.3.14. This driver version is certified by Microsoft.

### Installation

- Windows now requires only three packages: Axelera Device Package, Axelera Runtime, and Axelera Services. The `axelera-win-toolchain-deps-installer.exe` is no longer built or shipped, and "RISC-V Toolchain and Dependencies for Axelera" will no longer appear in Add/Remove Programs.
- The RISC-V toolchain has moved out of the runtime install and into the development kit.
- `numpy` is no longer pinned to `< 2` by the installer configurations; resolution is left to the SDK wheels. The NumPy `< 2` constraint documented for `axelera-devkit` in v1.7 therefore no longer applies at install time, and NumPy 2.x incompatibilities in the detection and segmentation evaluators have been fixed.
- New system packages are required for the Vulkan renderer: `glslang-tools` and `libvulkan1` (installed by `install-dependencies.sh`).

> **Deprecation notice:** the classic SDK Installer is deprecated as of v1.8.0 and will be removed in a future release. Installation via `pip` is the recommended path for both new deployments and existing environments.

### Release Compatibility Matrix

Recommended and supported board controller, flashed firmware and PCIe driver versions for each Voyager SDK release are maintained in the [Release Compatibility Matrix](RELEASE_COMPATIBILITY_MATRIX.md). Consult it before installing or upgrading a card or a Voyager SDK release.

For v1.8.0 the minimum supported driver equals the shipped version - see [IMPORTANT - Driver upgrade is mandatory](#important---driver-upgrade-is-mandatory) above.

## New Features / Support

### New Axelera AI Cards and Systems

- Metis 4-chip PCIe with 8GB RAM is supported.
- Chip-down Metis support, in two variants: with a board controller, and without a board controller. For chip-down designs the LPDDR geometry (size, ranks, bus width) is read directly from GPIO pins at boot, enabling support for new Metis boards without changes to Metis firmware.

### Host Platform Support

#### Operating systems and kernels

- **Linux kernel support now spans 5.4 to 6.17.** Kernel 5.4 support enables Yocto **dunfell** embedded targets. A kernel warning on 6.17 has been eliminated.
- The `axelera-rt` wheel with ManyLinux support remains the recommended installation path for Yocto images. Refer to the [meta-axelera](https://github.com/axelera-ai-hub/meta-axelera) Yocto layer for the recommended recipe structure.

The full list of Validated Host Systems for Axelera Metis Cards is available [here](https://docs.axelera.ai/docs/hardware/metis/common/validated-host-systems).

### \[Alpha\] Model recipes for the Model Zoo

Selected Model Zoo models now ship a standalone Python recipe under `model_recipes/`: a self-contained, readable script that quantizes and compiles the model, with no dependency on zoo tooling at runtime. Users can copy a recipe, point it at their own weights and calibration data, and run it directly with `python`. See `model_recipes/README.md`.

Recipes are authored and regenerated from a new Python configuration system, `axzoo` (shipped as the `axelera-zoo` wheel). This is additive - the existing YAML model zoo, `deploy.py` and `inference.py` are unchanged and remain fully supported. Most application developers can work directly from a recipe; `axzoo` can be used for regenerating recipes, reproducing builds, and managing model catalogs over time.

```
axzoo list                                   # packaged configs, grouped by task
axzoo info detection/yolov8n_coco            # display name, license, metric
axzoo build detection/yolov8n_coco           # quantize + compile to .axm
axzoo fork detection/yolov8n_coco --script   # get the standalone editable recipe
axzoo benchmark detection/yolov8n_coco --video input.mp4
```

Command groups: **Discovery** (`list`, `check`, `show-config`, `info`), **Build** (`build`, `export`, `drift-check`, `check-assets`), **Run** (`benchmark`, `eval`), **Tools** (`fork`, `completion`). Shell tab completion is available via `axzoo completion install`.

Recipes build through the `axrelay` backend (`--backend axrelay`), the established quantization and compilation path, which ingests both `.onnx` and `.pt2`.

Other notes:

- `axzoo benchmark` reports a latency distribution (min/mean/p50/p95/p99/p99.9/max/stddev/jitter), a per-operator breakdown, per-frame device-versus-host split, throughput with `--cores N`, and host CPU and peak-memory usage. `--json` is supported.
- `.pt2` (`torch.export` ExportedProgram) is a new first-class source format alongside `.onnx`. Bare `.pt` state dictionaries are explicitly rejected.
- Model provenance (upstream framework, version and SPDX license) is now recorded in the `.axm` `manifest.json`.
- Named pre-compile passes replace ad-hoc YAML flags: `replace_focus_layer`, `gemm_to_conv`, `disable_ceil_mode_in_avgpool`, `yolo26_seg_insert_proto_identity`.
- Model exports run in isolated per-framework virtual environments so the SDK environment is never mutated.

The following models already present in v1.7 are now also available as recipes (built via `axzoo`): `resnet18-imagenet`, `resnet50-imagenet`, `yolov8n-coco`, `yolo11l-coco`, `yolox-s-coco`, `yolo11n-seg-coco`, `yolo11n-pose-coco`.

### New Networks Supported

Voyager SDK model zoo includes computer vision tasks and LLMs. For a full list of supported models and data about their performance and accuracy see [here](docs/reference/models/model-zoo.md).

Models that are supported but not included in the model zoo are documented [here](docs/reference/models/additional-models.md).

For convenience, pre-compiled models are available to download by running `axdownloadmodel` in the parent folder of Voyager SDK.

The YAML model zoo is unchanged in this release. New models in v1.8 are delivered through `axzoo`.

### \[Alpha\] Pipeline Builder API

Users are advised not to use this API for workloads where performance is critical, such as benchmarks or production workloads. Pipeline Builder will reach top performance and become the recommended API for running models on Axelera hardware in the next Voyager SDK release(s).

- **OpenCL (GPU) fast paths** are now used automatically for `op.resize`, `op.letterbox` and `op.color_convert`, with transparent fallback to the CPU implementation if the GPU path is unavailable.
- **The `Scheduler` no longer connects to hardware at construction.** Connection is established lazily on first use, and new `Scheduler.is_connected` and `Scheduler.opencl_context` properties are available. A pipeline with no `op.load` node never touches the device - previously, merely referencing the scheduler claimed every AI core.
- **Optimizer policy control:** `Seq.optimized()` and `LoadedSeq.optimized()` accept a `policy` argument (`'auto'` or `'none'`), with new context managers `optimize_policy(policy)` and `optimization_disabled()`.
- `OperatorProfiler` **now times leaf operators only**, so nested containers no longer double-count their children.
- **New `ResourceMonitor` context manager** reports process CPU percentage, core count and peak RSS.
- **Nested `for_each` geometry is now correctly scoped**, so `to_image_space` inside a `for_each` after a `resize` maps coordinates back correctly.
- `Image` **additions:** `decoded_timestamp`, `color_range` and `color_matrix`.
- Reusable pipeline factories are available under `axelera.zoo.pipelines` for the YOLO detection, pose and segmentation families, YOLOX and classifiers, and a benchmarking helper under `axelera.zoo.bench`.

#### New task types and operators

- **Semantic segmentation is now a first-class task**: new `op.SemanticSegmentation` result type, `op.ax_semantic_segmentation()` and `op.decode_semantic_segmentation(num_classes=…)`, with palette-based rendering.
- **Optimized int8 semantic segmentation decode** using Intel AVX2 and Arm NEON with a fallback on scalar if neither is available.
- **Face detection and recognition:** `op.decode_retina_face()`, `op.Embedding` with `op.ax_embedding()`, and `op.Recognition` with `op.match_gallery()` for cosine-similarity gallery matching.
- **Licence plate recognition:** `op.decode_lprnet()` performs greedy CTC decoding to plate strings.
- `op.sort_by(by=…, descending=…, top=…)` orders results by a named numeric attribute, supported by new `BBox.area`, `BBox.center_distance`, `Object.area` and `Object.center_distance` properties.
- `op.observer()` **and** `op.dumper()` run a side-effect callback, or write each input array to disk, and pass inputs through unchanged.

#### Video decode and sources

- **Frame pacing:** a trailing `@<fps>` or `@auto` on an input specification paces a file as if it were a live camera, for example `./inference.py yolov8n-coco video.mp4@30`.
- **New colour formats** end to end: `Y444` (planar YUV 4:4:4) and `Y42B` (planar 4:2:2), plus corrected `GRAY8`, `RGBA`, `BGRA` and `NV16` paths.
- **Colour range and colour matrix metadata** are now carried per frame from the decoder, so the correct YUV-to-RGB conversion matrix is selected automatically.
- `decoded_timestamp` is stamped by the decoder on each frame and shares a clock epoch with Python's `time.monotonic_ns()` on Linux.
- The maximum number of inference children was raised from 4 to 8.

#### Display and rendering

- New `Surface.rectangle()` layer type, `App.start_thread()` and `App.run(interval=…)`.
- `display.Options` gained `show_bounding_boxes`, `show_keypoints`, `show_segmentation`, `show_trajectory` and `show_tiles`, and the label format `{track_id}`.
- Screen resolution is now detected via CoreGraphics on macOS rather than only `xrandr`.

### YAML Pipeline Builder

- **New `pillow_bilinear` interpolation mode** for `Resize`, matching PIL/Pillow anti-aliased downscaling. The OpenCL resize element gained a corresponding `interpolation` option, and the composite OpenCL preprocessing operators now use it.
- **\[Alpha\] New Vulkan renderer**, selectable with `--display vulkan`. Shaders are compiled on the fly and cached, and headless operation is supported with a suitable GPU. Set `AXELERA_VULKAN_DEBUG=1` or `2` for validation-layer diagnostics (requires `vulkan-validationlayers`).

    Note: the Vulkan renderer is new in this release and is not yet at parity with `--display opengl`. Not yet supported: multi-resolution layout, frame style settings, the grayscale option, image-overlay and heatmap metadata types, the startup logo and progress bar, and window resizing.

#### Demo scripts and examples

| Demo Script | Description | Model |
| --- | --- | --- |
| `semantic_segmentation.py` | YOLO26n semantic segmentation on Cityscapes (19 classes) at 1024x1024. | `yolo26n-sem.axm` |
| `fruit_demo.py` | Three-stage cascade: pose estimation, then segmentation on the largest regions, then detection, filtered to fruit classes. | `yolov8l-pose`, `yolov8s-seg`, `yolov8s` |
| `bottle_demo.py` | Detection, tracking, then per-track classification of the bottles nearest the frame centre. | `yolov8n-coco.axm`, `resnet50-imagenet.axm` |

All examples under `examples/pipeline_builder/` gained a shared `--backend {ffmpeg,opencv}` option.

> **Note:** `semantic_segmentation.py` requires `yolo26n-sem.axm`, which is not in the public pre-compiled catalog. Build it locally with `download_axm.py --deploy-missing`.

### RISC-V toolchain unification

- **LLVM 20 bare-metal toolchain for RISC-V** replaces the previous GNU newlib toolchain. As a result of this the wheel shrinks from ~200 MiB to ~20 MiB (~900 MiB less on disk), with up to ~10% inference performance improvement, especially on small, control-bound networks.

### Tools

- `axddr` **(new)** is a high-level tool for LPDDR verification, usable both on Axelera boards and by chip-down customers to facilitate and speed up LPDDR design validation. It performs LPDDR memory validation over PCIe, with no JTAG or UART required; it loads the stage0 firmware on demand. Tests available: `sequential`, `random`, `aliasing`, `databus`. `axddr -i` prints device name, DDR size, rank, bus width and firmware versions without running a test.
- `axrunmodel`: reworked with a low-latency executor, outputs verification and statistics; resources are auto-selected from the model's L2 footprint; new transient `--set-power-limit` and `--set-mvm-limit` overrides.
- `axmonitor`:
    - New post-processing mode: `axmonitor --post-proc <file.jsonl>` generates per-device SVG plots (temperature, power, KPS, DDR bandwidth, PCIe bandwidth, CPU utilisation), a temperature-versus-KPS plot, an average-power-versus-KPS plot, and a Markdown summary.
    - Power-during-inference metrics are included in the post-processing report.
- `axcmd`: new `--get-mvm-limits <aicore>` reports the user-set, power-control-loop, service-monitor and effective MVM limits separately, so it is clear *why* MVM utilisation is being capped. `--get-ddr-size` now works on chip-down boards.
- Power limit configuration now persists across device resets.
- `tools/tile_config.py`: actionable install hints when `wx` or `cv2` are missing, and corrected control sizing.

### Packaging

Newly shipped wheels:

| Package | Available in |
| --- | --- |
| `axelera-zoo` | `axelera-rt` and `axelera-devkit` |
| `axelera-riscv-llvm-toolchain-minimal` | `axelera-devkit` |

Removed: `axelera-riscv-gnu-newlib-toolchain`, the `axelera-riscv-openocd` Debian package, and `axelera-win-toolchain-deps-installer.exe`.

## Breaking Changes

1. **AxModel major version 4 to 5 - models must be rebuilt.** AxModels built with v1.7 or earlier cannot be run or inspected. See [above](#important---axmodel-version-5-models-must-be-rebuilt).
2. **Minimum supported PCIe driver raised to the shipped version:** Linux **1.6.2** and Windows **1.3.14**. Older drivers are rejected.
3. **Linux device node names changed** from colons to hyphens, for example `metis-0:1:0` becomes `metis-0-1-0`. udev backwards-compatibility symlinks preserve the old form, but scripts that glob or parse the colon form should be reviewed.
4. **Metis Compute Board (AISBC) running BSP 1.3.3 requires a patch for SDK v1.8.0.** This consists of updating the Metis kernel module to 1.6.2 and applying a `.deb` package that symlinks the device node name change described above. Download both packages from the Axelera software portal: the [Metis kernel module 1.6.2 package](https://software.axelera.ai/ui/native/axelera-bsp/voyager/bsp/aisbc/1.3.x/kernel-module-metis-6.1.148-rockchip-standard_v1.8.0-r0_arm64.deb) and the [name-change symlink package](https://software.axelera.ai/ui/native/axelera-bsp/voyager/bsp/aisbc/1.3.x/axelera-container_1.0-r0_arm64.deb). To apply:

```
su root                                                            # root password for the AISBC BSP image
mount -o remount,rw /
dpkg -i kernel-module-metis-6.1.148-rockchip-standard_v1.8.0-r0_arm64.deb   # Metis kernel upgrade
dpkg -i axelera-container_1.0-r0_arm64.deb                                  # device name change symlink
mount -o remount,ro /                                              # if this fails, simply run `reboot`
reboot
```

## Fixed Issues Since v1.7.0

- Fixed a **chip inference abort** on models with hand-written hardsigmoid or hardswish, for example MobileNetV3-Small, which previously failed with "fewer padding configurations than tensors".
- Fixed an intermittent crash when streaming with more than one core, caused by a detached decoder thread using freed contexts.
- Fixed `color_convert()` crashing on **odd-width** of YUV input formats.
- Fixed the **last few frames of long videos being dropped**, and made the progress display respect `--frames`.
- Fixed `op.top_k` crashing on standard classification head shapes.
- Fixed NumPy 2.x incompatibilities in the detection and segmentation evaluators.
- Fixed a Darknet model's `.cfg` path not being expanded from its portable form.
- Fixed the Level Zero out-of-memory message, which now names the memory region and reports human-readable sizes and totals.
- Fixed **kernel 5.4 build failures** and a kernel warning on 6.17.

## Known Issues and Limitations

### Memory leak in GStreamer software video decode (`gst-libav` 1.24.2)

Pipelines that perform software H.264 decoding via GStreamer's `avdec_h264` element (`gst-libav` 1.24.2, as shipped with GStreamer 1.24.2 on Ubuntu 24) leak approximately **104 bytes per decoded video frame, per decoder**. This is a defect in `gst-libav` itself (`avdec`/`avviddec` does not free the per-frame `AVPacket`), not in the Voyager SDK, and the SDK cannot workaround it.

The leak grows linearly with runtime and scales with the number of decoded streams, so it is most noticeable in long running, multi-stream deployments. For example, 16 streams at 30 FPS leak roughly 50 KiB/s in aggregate.

**Impact:** affects only pipelines that use GStreamer software video decode. Hardware-accelerated decode paths are not affected.

**Mitigation:** the only fix is to upgrade GStreamer / `gst-libav` to **1.28.3 or later**, which resolves the underlying `AVPacket` leak. Or alternatively where possible, prefer hardware-accelerated decode over software `avdec_h264`. Restarting long-running pipelines periodically bounds the resident memory growth.

### Other

- The `axzoo` documentation is currently shipped with the `axelera-zoo` package rather than being part of the documentation portal.
- Performance variability is observed on certain hosts. Inference-only performance (FPS) drops of up to 5-10% are observed on `yolox-x-crowdhuman-onnx` and `yolo11l-obb-dotav1-onnx` compared to SDK Release v1.5.
- Python 3.13 incompatibility with the wheel installer on Ubuntu 24.04. Not reproducible with Python 3.12 (default for Ubuntu 24.04).
- Metis Compute Board video output rendering is choppy (1-2 FPS). This impacts rendering to display only; inference performance is not impacted.
- YOLO26\* models (e.g. `yolo26x-obb-dotav1-onnx`) sometimes fail to deploy unexpectedly.
- MobileNetV3 may yield degraded accuracy on some combinations of hosts and cards.
- Device monitoring with AxMonitor is not supported on single-MSI hosts. For some systems with single-MSI hosts, device monitoring with `AxMonitor` does not display any data. An example of a host with this issue is Arduino Portenta X8 Mini.

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
| Python | 3.10 to 3.13 |

### Runtime Environment

This release is expected to work with Intel (x86), AMD (x86) and Arm64 host CPUs. See [here](https://docs.axelera.ai/docs/hardware/metis/common/validated-host-systems) for a list of validated host systems for Axelera Metis AI Accelerator Cards.

## Further Support

- For blog posts, projects and technical support please visit the [Axelera AI Community](https://community.axelera.ai/).
- For technical documents and guides please visit [docs.axelera.ai](https://docs.axelera.ai/).

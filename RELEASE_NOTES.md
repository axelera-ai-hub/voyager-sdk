# Voyager SDK release notes v1.9.0

- [Voyager SDK release notes v1.9.0](#voyager-sdk-release-notes-v190)
  - [Voyager SDK release notes v1.9.0](#voyager-sdk-release-notes-v190-1)
    - [Release Qualification](#release-qualification)
  - [Installation and Compatibility](#installation-and-compatibility)
    - [Installation](#installation)
    - [Deployed model compatibility](#deployed-model-compatibility)
    - [Release Compatibility Matrix](#release-compatibility-matrix)
  - [New Features / Support](#new-features--support)
    - [New Axelera AI Cards and Systems](#new-axelera-ai-cards-and-systems)
    - [Host Platform Support](#host-platform-support)
      - [Operating systems and kernels](#operating-systems-and-kernels)
      - [Windows native Pipeline Builder](#windows-native-pipeline-builder)
    - [New Neural Networks Supported](#new-neural-networks-supported)
      - [Beta: axzoo and model recipes](#beta-axzoo-and-model-recipes)
      - [New models in the YAML zoo](#new-models-in-the-yaml-zoo)
      - [Beta: ViTs](#beta-vits)
      - [Beta: LLMs and VLMs](#beta-llms-and-vlms)
    - [Beta: Pipeline Builder API](#beta-pipeline-builder-api)
      - [New task types and operators](#new-task-types-and-operators)
      - [Video decode and sources](#video-decode-and-sources)
      - [Display and rendering](#display-and-rendering)
    - [YAML Pipelines](#yaml-pipelines)
      - [Demo scripts and examples](#demo-scripts-and-examples)
    - [Model Compiler](#model-compiler)
      - [Alpha: AxeleraScript (AxScript)](#alpha-axelerascript-axscript)
    - [Generative-model enablement](#generative-model-enablement)
      - [Alpha: AxScript LLM/VLM toolkit](#alpha-axscript-llmvlm-toolkit)
    - [Runtime](#runtime)
    - [Tools](#tools)
  - [Breaking Changes](#breaking-changes)
  - [Fixed Issues Since v1.8.0](#fixed-issues-since-v180)
  - [Known Issues and Limitations](#known-issues-and-limitations)
    - [Memory leak in GStreamer software video decode (gst-libav 1.24.2)](#memory-leak-in-gstreamer-software-video-decode-gst-libav-1242)
    - [New models supported](#new-models-supported)
    - [Pipeline Builder](#pipeline-builder)
    - [Runtime, driver, firmware](#runtime-driver-firmware)
    - [Other](#other)
  - [System Requirement](#system-requirement)
    - [Development Environment](#development-environment)
    - [Runtime Environment](#runtime-environment)
  - [Further Support](#further-support)


## Voyager SDK release notes v1.9.0
Voyager SDK v1.9.0 is a content rich release: it greatly improves the Model Zoo coverage with a single build/run/eval/report workflow, makes the Pipeline Builder substantially faster and more capable, ships a native Windows runtime and re-brands the Metis product line.

- **Generative AI and transformer Support:** compilation and execution of generative AI and transformer-based models (incl. LLMs, VLMs and ViTs). Our toolchain is augmented with **AxScript** and with **AxGraph** to help users port their models to our platform.
  - **AxScript:** a Python-based Domain Specific Language which can be used to program Axelera AI platforms at a low level, giving the developer full power and flexibility. AxScript can be used to implement Generative AI models, CNNs, ViTs etc. directly (either by hand, or with the help of an AI agent). See [Axelera script blog](https://axelera.ai/blog/axelerascript-one-language-every-inference-workload-solved) for an overview and benefits and the [AxScript Language Guide](docs/reference/axscript/axelerascript_guide.md) included in this release.
  - **AxGraph**: our new Graph Compiler based on MLIR, with preliminary support for generative AI and transformer-based models is Experimental.
  - The existing compiler is now referred to as **AxRelay**, in all documentation and code examples. It supports CNNs and is still the recommended production path for the models it supports.
  - The release adds support for several state-of-the-art LLMs, VLMs and ViTs. See [Model Zoo](docs/reference/models/model-zoo.md).
- **Pipeline Builder API matures to Beta** offering better stability, higher performance and a wealth of new features. For example, native post-processing kernels replace the onnxruntime "postamble" tail, a new fusion pass runs pre-processing as a single fused OpenCL kernel (up to 10x faster than OpenCV), new scheduler defaults raise YOLOv8n 720p throughput from 377 to 682 fps on 4 cores, and a portable `.axe` pipeline format records a pipeline end to end.
See [Pipeline Builder API documentaion](docs/reference/pipeline-builder/api/README.md) 
- **Model Zoo migration to the Pipeline Builder API**: The zoo's models now reach you as standalone Python recipes instead of YAML, in `model_recipes/`. One file per model, self-contained, showing the whole path a production deployment takes: calibration, quantization, compilation with the model's tuned settings, and the end-to-end inference pipeline wired in a few lines with Pipeline Builder. Nothing is hidden in a catalog, so every knob that shaped the published numbers is a line you can read and change.
  - **Run it, measure it, reproduce it.** Run a recipe to get throughput and a rendered result window straight away. Add `--eval`, with `pip install axelera-zoo[eval]`, to reproduce the published accuracy on the same scoring path the figures were taken with.
  - **Start from the closest model.** `axzoo list` shows the catalog, `axzoo info <model>` shows where a model comes from upstream and how it is exported, and `axzoo fork <model>` copies the recipe, its config or its export script into your own project, so a reference model becomes the starting point for your own weights.
- **AxScript models ship as code**, in `model_axscript/`, one Python package per model with its kernels. A `model_recipes/` file is a model's resolved configuration; an AxScript recipe is its implementation, so the part you change is the kernel. Build, run, benchmark and eval with `python -m model_axscript.vision` or `python -m model_axscript.llm`, and since quantization runs on your machine from the published checkpoint, a fine-tune of a shipped architecture rebuilds by changing one line. See [Model Recipes documentation](model_recipes/README.md).
- **Native Windows support for the Pipeline Builder**: an MSVC `win_amd64` wheel of `axelera-runtime2` (Python 3.10 to 3.14). Includes a Vulkan-based renderer, and trackers are supported. Pipelines run natively; model compilation remains Linux/WSL2.
- **Product re-branding.** `axdevice` and the application framework now use the new board names: Axelera Embedded **110m / 113m / 111c**, Axelera Edge **130p / 232p**, Axelera Server **150p**. `--board` replaces `--metis` in the application framework. Product names starting with "1" have a Metis chip, "2" have a Europa chip (note: Europa is not yet supported in this SDK release). See [Axelera updated naming blog](https://axelera.ai/blog/same-amazing-products-updated-naming) for more information. Scripts that parse `axdevice` output must be reviewed.
- **Driver upgrade is mandatory**: Linux `metis-dkms` **1.7.15** (minimum 1.7.0) and Windows `MetisDriver` **1.4.5** (minimum 1.4.0). No Metis firmware re-flash is required.

### Release Qualification

This is a production-ready release of Voyager SDK. Software components and features that are in development are marked with one of the following maturity labels:

- **Experimental:** May change or be removed without notice; no support guarantees.
- **Alpha:** Usable but incomplete; breaking changes possible.
- **Beta:** Feature-complete but not fully stable; committed to developing this further in future releases.

## Installation and Compatibility

**IMPORTANT — Driver upgrade is mandatory**

The minimum supported PCIe driver is raised on both Linux and Windows. Older drivers are rejected by the runtime.

- Linux: install `metis-dkms` **1.7.15** (minimum supported: 1.7.0). `axdevice driver --install` downloads and installs the version pinned by the SDK.
- Windows: install **MetisDriver-1.4.5** (minimum supported: 1.4.0).

### Installation

- `quick-start.sh` (new) installs the SDK from the version recorded in `VERSION.TXT`, which is now shipped in the SDK installer directory.
- **Pip install:** Installation via pip is now the supported path. As in previous releases, the full SDK installation requires both `axelera-devkit[all]` and `axelera-rt` packages. A runtime-only environment requires only `axelera-rt` but will not include `inference.py`, visual rendering and the `--save-output` option; those can be enabled by installing `axelera-rt[all]`.
- New system packages: `libpci-dev` (axcmd) and `python3-pip` are installed by `install-dependencies.sh`. `libzip` is no longer required by the runtime.
- **Windows**: 1.9.x installers, plus `pip install axelera-runtime2` for native pipelines. The Vulkan renderer needs a GLSL compiler (`glslangValidator`) on PATH; see the updated `windows-setup.md`.
- `pip` must stay below 26.2 in SDK environments (26.2 breaks `.pth`-based isolated builds); the SDK images and constraints pin it.
- **Classic installer** `install.sh` moved from pipenv to uv. `--gen-pipfile` is a deprecated alias for `--gen-uv-input`. `uv` is now an installer prerequisite. The classic installer remains deprecated.

> **Deprecation notice**: the classic SDK Installer (`install.sh`) has been deprecated and is no longer available. Installation via `pip` is the supported path for both new deployments and existing environments.

### Deployed model compatibility

A model deployed with v1.8.0 still loads on the v1.9.0 runtime: the `.axm` format version is unchanged. Re-deploying with the v1.9.0 toolchain is recommended to pick up the compiler improvements listed below (default memory allocator, fused requantize, new kernels). The manifest inside a `.axm` adds version 2.0, recording quantization parameters, padding and axis order per tensor instead of as parallel arrays; the runtime still reads 1.1 manifests and ignores manifest fields it does not represent.

### Release Compatibility Matrix

Recommended and supported board controller, flashed firmware and PCIe driver versions for each Voyager SDK release are maintained in the Voyager SDK Release Compatibility Matrix [here](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/RELEASE_COMPATIBILITY_MATRIX.md). Consult it before installing or upgrading a card or a Voyager SDK release.

## New Features / Support

### New Axelera AI Cards and Systems

This release does not add support for any new Axelera cards.

### Host Platform Support

#### Operating systems and kernels

- **Linux kernel support remains 5.4 to 6.17**; driver now supports Yocto releases kirkstone, scarthgap and wrynose.
- **Python 3.10 to 3.13** for SDK environments; the native Windows `axelera-runtime2` wheel is built for 3.10 to 3.14.
- **Windows**: dynamic large-page allocation in `libaxldev` falls back to regular pages with a warning instead of failing when the lock-pages privilege is missing.

#### Windows native Pipeline Builder

- `axelera-runtime2` is available as a native MSVC `win_amd64` wheel (Python 3.10 to 3.14) with trackers and the Vulkan renderer (Win32 surface). Pipelines and recipes run natively on Windows; model compilation support is still limited to Linux (or WSL2).

The full list of Validated Host Systems for Axelera Metis Cards is available [here](https://docs.axelera.ai/docs/hardware/metis/common/validated-host-systems).

### New Neural Networks Supported

Voyager SDK model zoo includes computer vision tasks and LLMs. The model zoo is moving from YAML to `axzoo`. Both ship in v1.9.0:

- `axzoo` - each model is a Python recipe built on Pipeline Builder. `axzoo` does not depend on `axelera.app`, so a recipe runs standalone. From the next release, new models are added to `axzoo` only.
- YAML zoo - `axelera.app` deploy using `deploy.py` and run as a GStreamer pipeline using `inference.py` remain supported.

#### Beta: axzoo and model recipes

The Python model zoo introduced in v1.8 as an Alpha is now the primary way models are built, run and evaluated.

- **Recipes:**
  - An `axzoo` model recipe replaces a YAML declaration with the code that runs. Nothing is resolved behind a schema: the calibration, the compiler call and the pipeline are lines you read, change and diff.
- **Compile:**
  - `axzoo` models can be compiled with:
    - AxRelay, the compiler behind the YAML zoo CNNs
    - AxGraph, the new graph compiler
    - AxScript, which allows models and kernels to be created using a low-level API directly.

Not every model is released on every compilation path. `axzoo list` shows which paths a model is released on, and `axzoo info <model>` names the one it builds by default. Individual model recipes make this explicit. They also read well to a coding agent, which can learn the toolchain from a few examples instead of relying on SDK internals.

Pre-compiled models are available with `axdownloadmodel`; use `axdownloadmodel --target metis` to select the device generation.

See [AxScript LLM/VLM toolchain](#alpha-axscript-llmvlm-toolkit) for `axzoo` features on such as run, eval and benchmark.

- **Published Model Cards and Recipes:**

109 model cards (86 vision, 11 text LLM, 7 VLM) and 65 standalone recipes under `model_recipes/` are available. 79 YAML zoo cards are migrated.

| Task | axzoo model cards released recipes |
| --- | --- |
| Image classification (top-1) | resnet18, resnet34, resnet50, resnet101, resnet152, resnext50_32x4d, wide_resnet50, densenet121, efficientnet_b0, mobilenetv2, mobilenetv3_large, mobilenetv3_small, yolo11n_cls |
| Object detection (COCO mAP50-95) | yolov5n, yolov5s, yolov5s_v5, yolov5m, yolov5l, yolov7, yolov8n, yolov8s, yolov8m, yolov8l, yolov9t, yolov9s, yolov9m, yolov9c, yolov10n, yolov10s, yolov10b, yolo11n, yolo11s, yolo11m, yolo11l, yolo11x, yolo26n, yolo26s, yolo26m, yolo26l, yolox_s, yolox_m |
| Pose estimation (COCO mAP50-95) | yolov8n_pose, yolov8s_pose, yolov8m_pose, yolov8l_pose, yolo11n_pose, yolo11l_pose |
| Instance segmentation (COCO mAP50-95) | yolov8n_seg, yolov8s_seg, yolov8m_seg, yolov8l_seg, yolo11n_seg, yolo11l_seg, yolact_resnet50 |
| Semantic segmentation (Cityscapes mIoU) | yolo26n_sem, yolo26s_sem, yolo26m_sem, yolo26l_sem, yolo26x_sem, yolo26n_sem_logits |
| Monocular depth (NYU delta1) | yolo26n_depth, yolo26s_depth, yolo26m_depth, yolo26l_depth, yolo26x_depth |

#### New models in the YAML zoo

For a full list of supported YAML zoo models and data about their performance and accuracy see [here](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/docs/reference/models/model-zoo.md).

Models that are supported but not included in the model zoo are documented [here](https://github.com/axelera-ai-hub/voyager-sdk/blob/latest/docs/reference/models/additional-models.md).

| Model Name | Resolution | Source Format |
| --- | --- | --- |
| YOLACT ResNet-50 (yolact-resnet50-coco-onnx) | 550x550 | ONNX |
| YOLO26 {n,s,m,l,x} depth (yolo26*-depth-nyudepthv2-onnx) | 640x640 | ONNX |

> **Warning:** The YAML-based model zoo, along with `deploy.py` and `inference.py` will be deprecated in a future release; existing scripts and application code should be migrated to use model recipes and the Pipeline Builder API. The deprecation will not happen before the Pipeline Builder API is GA (Generally Available).

- MobileNetV4-small accuracy. Activations feeding depthwise convolutions are now observed per channel, cutting the int8 top-1 loss from 3.25 to 1.51 points (72.23 against a 73.74 FP32 reference, full ImageNet validation split).
- YOLO26 semantic segmentation now tracks the current Ultralytics export. The recipes pin 8.4.146, where Ultralytics reduces the class axis inside the graph, so the model declares a (1, 1024, 1024) uint8 class map instead of (1, 19, 1024, 1024) float32 logits. Same weights, same accuracy; what changes is the tail the postamble reproduces. `yolo26n-sem-cityscapes-logits` stays on the 8.4.52 export for consumers that need the per-class scores.

For convenience, pre-compiled models are available to download by running `axdownloadmodel` in the parent folder of Voyager SDK.

#### Beta: ViTs

The following models are available as AxScript models. Run with `python -m model_axscript.vision {list,build,run,benchmark} <tag>`.

- ViT-S/16, ViT-B/16, ViT-L/16 (top-1 0.8084 / 0.8362 / 0.8521)
- DINOv2-S/14
- DINOv2 segmentation S/B/L (ADE20K)
- DINOv2 depth (NYU, linear head)
- Depth-Anything-V2-small

#### Beta: LLMs and VLMs

**LLMs**

| Model | Max Context Length |
| --- | --- |
| qwen3-0.6b | 4096 |
| qwen3-1.7b | 4096 |
| qwen3-4b | 4096 |
| qwen3-8b | 4096 |
| llama-3.2-1b | 4096 |
| llama-3.2-3b | 4096 |
| llama-3.1-8b | 4096 |
| lfm2-700m | 4096 |
| lfm2.5-350m | 4096 |
| lfm2.5-1.2b | 4096 |
| domyn-10b | 4096 |

**VLMs**

| Model |
| --- |
| qwen3-vl-2b-instruct |
| qwen3-vl-8b-instruct |
| smolvlm2-256m |
| smolvlm2-500m |
| smolvlm2-2.2b |
| lfm2.5-vl-450m |
| lfm2.5-vl-3b |

> **Note**: The ViT, LLM & VLM models are not supported under Windows OS in this release 

See **Generative-model enablement** below for the model table.

### Beta: Pipeline Builder API

This release brings the Pipeline Builder API to Beta level of maturity, stability and a wealth of new features. The Pipeline Builder API generates pipelines which are faster end-to-end and gives the developer control over the kernels implementing the model postamble.

Highlights:

- **Native kernels for the model's postamble.** Previously model postamble subgraph ran in onnxruntime; in this release the developer can supply their own kernels via the `using_postable` parameter of `op.load()`.
- **Pre-process fusion:** `.optimized()` folds color conversion, resize, letterbox, centre-crop, normalization and quantization into one kernel resulting in up to 10.4x higher performance; `--optimize {auto,cl,simd,none}` forces a backend, and a forced kernel that could not be applied now warns instead of quietly running on the CPU; `collect_declines()` tells you why fusion could not be applied.
- **Improved pipeline scheduling and throughput.** Default queue depths are now tuned per AIPU core, with asynchronous frame submission/completion resulting in higher throughput, for example +81% for yolov8n-coco at 720p.
- **Improved multi-threaded scaling.** Kernel operators no longer serialize on the Python interpreter lock (GIL). An 8-thread preprocessing workload at 640x640 improved from 0.2x to 7.8x scaling. OpenCL operators are also now safe to share across threads.
- **Improved video pipelining** through zero-copy buffer import. Image resize from 1080p → 640x640 and 4K -> 640x640 is 3x faster.
- **Multi-stream**: `Scheduler.as_available`, `round_robin`, `with_callback` and `multistream` share one core pool across streams. A new `multipipeline_multimodel` example runs five models across five streams on one core pool.
- **Profiling**: A new profiler attributes each operator to its model and its role, preamble or postamble, instead of merging same-named ops from different models into one row; it adds a Faults column to separate work from waiting and reports device occupancy, host cores used and achieved in-flight depth.
- **Packaged pipelines (.axe)**: A pipeline can be saved to a file and loaded back with its real input shape and format, its per-task metadata, and the custom operators it needs declared and checked at load time. See `examples/pipeline_builder/axe.py`.
- **Loading**: `op.load` takes an unpacked `.axm` directory, the only form the native runtime loads on Windows. An empty or corrupted `.axm`, typically a failed download, is detected and re-fetched.

#### New task types and operators

- **Face detection**: RetinaFace MobileNet-0.25 and ResNet-50 on WIDERFACE, with a native postamble and optional landmark output.
- **Instance segmentation:** YOLACT ResNet-50 supported at its native 550x550.
- **YOLOv5 and v7** are supported on runtime2, through the anchored YOLO postamble.
- **Fast NMS**: `op.nms(algo='fast', top_k=N)`, a different suppression algorithm rather than an approximation, resulting in 1.6x against numpy at 17k-29k rows, 4.5x at 3200.
- **Decoders for every shipped head**: `op.DecodeDetections`, `op.DecodeSegmentation`, `op.DecodeSsd`, `op.DecodeRetinaFace`, `op.AxPose(keypoint_stride=)`, `op.DecodeYolact`, and `op.DecodeSemanticSegmentation`, with `tensor_roles` to disambiguate aliasing tensor shapes.
- **Lens and viewpoint correction**: `op.perspective()` and `op.barrel_correct()` in OpenCL with a CPU fallback.
- **Fused preprocess kernel**. `ResizeClOperator` does color conversion, resize or letterbox, and normalise or quantize in one OpenCL pass; `AxResizeCl` folds resize and centre-crop together.

#### Video decode and sources

- **Live sources keep the freshest frame**: A source that falls behind now drops stale queued frames rather than the newest arrival, so it catches back up in real time instead of falling permanently behind. `source.dropped` and `source.may_drop` report what happened.
- **Playback pacing**: `create_source('path@fps')` or `'@auto'` plays a file at a chosen or natural rate.
- **Iteration helpers**: `cv.iter_results`, `cv.frame_count`, `cv.is_single_image`, and `cv.Source` are public for writing your own source.

#### Display and rendering

- **Vulkan** is the default renderer (`--display {none,opencv,vulkan,console,iterm2,auto}`) with a single hardware-native copy.
- **Low-latency preview**: `stream(frame_callback=, split_at=)` lets a display run ahead of inference for low-latency preview. `split_at` names an operator in your pipeline, and everything up to it runs on the thread driving the stream; `frame_callback` then receives that frame before inference is submitted.
- **Windows**: Vulkan renders through a Win32 surface; the OpenCV renderer is offered, and a requested Vulkan renderer fails loudly rather than silently going offscreen.

### YAML Pipelines

- To align with new product names, `--board` (110m, 113m, 130p, 150p, 232p, 250p, auto, none, pcie, m2, europa) replaces `--metis`, which is kept as a deprecated alias. `config.Board`, `add_board_arg`, `get_board_type()`, `detect_board_type()`.
- The asynchronous executor is now the default for batch-1 models and GStreamer queues shrink from 4 to 1.
- `AXELERA_SLOG_LEVEL` (default `err`) controls the device system log verbosity; leaving it at the previous default resulted in limited throughput in some pipelines.
- `AXELERA_ENABLE_AUTOMATIC_REPORTS=1` enables the crash reporter (`axdevice --report`). `--cl-platform` implies `--enable-opencl`.

#### Demo scripts and examples

| Example | Description |
| --- | --- |
| `examples/pipeline_builder/axe.py` | Save and reload a built pipeline as `.axe`. |
| `examples/pipeline_builder/multipipeline_decoded_tee.py` | One decoded stream feeding several pipelines. |
| `examples/pipeline_builder/multipipeline_multimodel.py` | Several models sharing one scheduler. |
| `examples/pipeline_builder/perspective_yolo.py` | Perspective correction before detection. |
| `examples/pipeline_builder/barrel_correct_yolo.py` | Lens (barrel) correction before detection. |

New documentation: three AxScript tutorials (quickstart, HuggingFace API, custom weights), the `.axe` format reference, API pages for the postamble, profiler, bench and image-preprocess operators, a C++ axruntime reference, and an LLM and VLM guide. Removed: `examples/llm/llm_compile.py` and `llm_run.py`, replaced by `axzoo build` and the LLM recipes.

### Model Compiler

**Experimental: AxGraph**

AxGraph is a graph-based compilation flow. Alongside the legacy axrelay path (ONNX or `.pt2` through `axelera.compiler.quantize/compile`), the SDK carries a second, graph-native toolchain that ingests `torch.export` programs (`.pt2`). It is made of the following packages which work as one pipeline:

- Axelera Transformers (`axelera-transformers`) provides Axelera-ready transformer architectures and the Hugging Face Optimum integration, so a checkpoint such as Qwen3, Llama 3, LFM2 or SmolVLM is exported into a graph the rest of the flow understands, including KV-cache and multimodal wrappers.
- The Model Optimizer (AxMO) (`axelera-model-optimizer`) takes that exported graph, annotates every operator against a per-generation operator database assigns each node to an engine, and quantizes it using post-training quantization techniques (incl. post-training calibration with SmoothQuant-style outlier removal).
- The graph compiler (`axelera-graph-compiler`, MLIR-based) lowers the optimized torch graph through the Axelera Operator Library (AxOL) dialect, whose operators correspond to AxeleraScript kernels, applies graph-level optimizations, and hands AxeleraScript to the ALTO backend to produce the `.axm`.

The flow is driven from the model zoo with `axzoo build --backend axgraph` and `axzoo eval`, and every recipe splits into a resumable `quantize()` and `compile_to_axm()` step.

#### Alpha: AxeleraScript (AxScript)

AxeleraScript (AxScript) is a Python-embedded, hardware-specific language for writing kernels and models for Axelera AI cores, and v1.9.0 is the first release in which it is a documented, supported authoring path. An AI core is a spatial pipeline of stream units (input feeders, the MVM and depthwise compute engines, the integer/activation arithmetic units and the store unit) connected by hardware FIFOs; AxScript exposes exactly an abstraction of that structure.

A kernel is written as a Python function but it is parsed into the compiler's IR rather than executed: the language constructs and ordinary `for`/`if`/`with` syntax describe which unit does what and how data streams between them, while the compiler fills in tiling, DMA, synchronization and unit bypass unless the author takes control of them explicitly.

New in this release:

- The AxScript Language Guide which covers the language specification and programming model, the datapath and block contracts, and an index of error messages by symptom.
- Support for multi-core programs.
- Support for asynchronous scheduling.
- Typed dtypes validated at construction (e.g., `int8`, `int8x64`).
- A rejected-constructs list (`while`, `assert`, `A.parallel`) that turns silent miscompiles into parse errors and source-located diagnostics that point every error, including out-of-memory, at the offending statement.
- Hardware-constraint validators.

AxScript is also the third build backend of the model zoo (`axzoo build --backend axscript`) and the foundation of the LLM/VLM toolkit (see below).

### Generative-model enablement

#### Alpha: AxScript LLM/VLM toolkit

- `model_axscript/` brings AxScript-based LLMs into the model zoo as a third `axzoo` backend, with its own command lines: `python -m model_axscript.llm {list,build,run,benchmark,eval} <tag>` and `python -m model_axscript.vision {list,build,run,benchmark} <tag>`. Vision scoring goes through `axzoo eval <card>`, where the metric lives.
- **HuggingFace-style API**: `register_auto()` teaches the Auto classes our model type (`ax_causal_lm`), after which `AutoModelForCausalLM.from_pretrained(<artifact>)` uses transformers' own `generate`; `AutoModelForImageTextToText` does the same for VLMs. Engine primitives `Implementation.ingest_prompt` / `decode_step` / `compute_logits` are available directly, and there is a torch-free run path.
- **Custom weights**: `--weights`, `--head` (segmentation/depth heads), `--calibration`, `--cal-device cuda` GPU calibration, keyed customer artifacts. Tutorials: `axscript-quickstart.md`, `axscript-llm-huggingface.md`, `axscript-custom-weights.md`.
- **Multi-core chunked prefill** provides top performance by dividing compute across multiple AI cores
- **Multi context lengths support** - 128, 256 and 512 join the existing 1024, 2048 and 4096.
- Custom prompts for benchmarking, and `--interactive` on `model_axscript run` for an interactive session.
- **Precompiled SLM Demo** models from previous releases marked as deprecated but still available to users.

### Runtime

- **Beta:** New Executor V2 introduced on the libaxldev job-ring stack supports two modes
  - synchronous `run()` with optional double buffering
  - asynchronous `run_async()` with separate device- and host-side buffer depths
- `AXR_API_VERSION` 3.2 to **3.3**: `axr_device_l2_size()` / `DeviceInfo.l2_size`; enums are `typedef enum` so `axruntime.h` compiles in C mode again.

### Tools

- `axdevice`: [new product names](https://axelera.ai/blog/same-amazing-products-updated-naming) and listing layout; `-d/--devices` filters the printed list; a clear error when no device is enumerated; fixes in `--report`.
  - The listing layout changed to `Device N: <name> <model> <memory>[ <grade>] flver=... bcver=... clock=... mvm=...` (model before memory; the 113m shows its temperature grade `std` after the memory size).

| Old display name | New display name | Axelera Product Name |
| --- | --- | --- |
| pcie-r2 | 130p | Axelera Edge 130p |
| m2 | 110m | Axelera Embedded 110m |
| std-temp m2-max | 113m (+ grade std) | Axelera Embedded 113m |
| compute-board | 111c | Axelera Embedded 111c |
| part of 4-metis-pcie | part of 150p | Axelera Server 150p |
| pcie | pcie (unchanged) | Axelera Edge 130p Rev 1.1 |

- `axrunmodel`: new product names replace old ones (see above)
- `axmodeltool`: `-g/--generation` and target inference from the ELF `hw_gen`; graph-model support in memory, tensor and kernel-version views.
- `axmonitor`: Improved GUI appearance, Enhanced terminal-mode output with additional PCIe link information reporting: **PCIe link generation, width and maximum bandwidth** for Linux.

## Breaking Changes

- **Board names re-branded.** Update scripts that parse `axdevice` output due to the change documented in [Tools](#tools).
- **Minimum supported PCIe driver raised.** For Linux from 1.6.2 to **1.7.0** (1.7.15 recommended) and for Windows from 1.3.14 to **1.4.0** (1.4.1 recommended). Host user space, driver and device firmware must come from the same release (HDIF context-area ABI 1.0, `CMD_HW_GEN` removed).
- `--metis` is deprecated in favour of `--board` in `axelera.app`; `config.Metis` is an alias of `config.Board`, `get/detect_metis_type` became `get/detect_board_type`, `override_metis=` is `override_board=` (no alias).
- `op.nms` defaults changed: backend native (pass `backend='opencv'|'torch'` for the old behaviour); oriented boxes default to `algo='fast'` (about 12 % fewer boxes; `algo='greedy'` restores); `xywhr` with `iou_metric='iou'` needs `backend='opencv'`; `max_boxes` semantics differ per algorithm.
- **Postamble kernels require** `num_classes`; softmax auto-detection was removed (declare `ClassificationPostamble`); `SemanticSegmentationDecode` was removed; `DecodeDetections` raises when width and algo contradict.
- **Pipeline Builder runtime defaults**: scheduler defaults (2 workers, 2 frames per core, asynchronous submission, batch 1 only on the async path) raise dmabuf residency about 4x; live-source `buffer_size` default 30 to 2; Vulkan is the default renderer in runtime2 and `axelera.runtime2` is required to render from `inference.py`; `Seq.optimized(None)` is deprecated; `save_axe` refuses optimized pipelines; `Perspective`/`BarrelCorrect` moved to `op.img_preprocess`.
- **Model Zoo CLI**: `--pipe` became `--engine {host,aipu}`; `--cores` became `--in-flight`; recipes require explicit `--build`, `--cal-device` and `CALIBRATION_DIR`; `axelera.zoo.passes` and zoo helpers were removed from the recipe surface; compiler TOMLs renamed per family
- **libaxmonitor protocol** 1.3 to 1.6: `AxeHostMessage.class` renamed `msg_class` in the `.proto` (wire-compatible).

## Fixed Issues Since v1.8.0

- Fixed `axdevice -d/--devices` listing every device instead of the selected ones, and `axdevice --report` crashing on `num_cores`.
- Fixed `axrunmodel` reporting false output mismatches
- Fixed `axdevice --reboot` and communication errors after a cold boot (virtualisation area refresh), and the BC update failing instead of being skipped on boards without a board controller.
- Fixed the Windows installer crash under certain regional settings and the Windows dynamic-memory ioctl.
- Fixed Linux driver stability and robustness by resolving defects around device probe, removal, interrupt handling, asynchronous DMA, firmware tracing and PCIe link recovery
- Fixed a quantization calibration defect that caused certain models to produce out-of-range zero points. Affected models now calibrate correctly and stay on device.
- Fixed bilinear resize using `align_corners` which was exported with different coordinate semantics without warning. A warning is now raised so the model can be re-exported correctly.
- Fixed the YOLO26 semantic-segmentation head running its output Resize on device
- Fixed YAML pipeline defects: cascade caps negotiation, the YOLO26 decoder for 5 to 63 classes, `--tiled` crash on yolo26/yolov10, `resize_cl` letterbox padding and YUV tint, dmabuf fd ownership, OpenGL grid positions, YOLACT mask placement.
- Fixed Model Zoo defects: `--optimize` dropped on eval, `show-config` crash, `--seq-len`/`--num-cal` silently ignored, device not released between report sizes
- Fix memory leak for cascade pipelines by using YAML/GStreamer

## Known Issues and Limitations

### Memory leak in GStreamer software video decode (gst-libav 1.24.2)

Unchanged from v1.8.0: pipelines that decode H.264 in software through GStreamer's `avdec_h264` (gst-libav 1.24.2 on Ubuntu 24.04) leak about 104 bytes per decoded frame per decoder. This is a gst-libav defect. Mitigation: upgrade gst-libav to 1.28.3 or later, prefer hardware decode, or restart long-running pipelines periodically.

### New models supported

- Running LLM, VLM and Vision Transformer models is not supported on Windows. It is supported on Linux only.

### Pipeline Builder

- Optimizations in pipeline builder API such as fusing the model tail and the pipeline's decode are not included in this release.
- Asynchronous mode in Executor v2 is on by default, but its depth is not yet auto-tuned. The default suits larger models; on small, fast models the limitation is higher latency and for high number of streams higher buffer memory per stream.
- OpenCL preprocess operators do not scale across host threads. The cause is in the driver and is unresolved; adding threads to an OpenCL-heavy preprocess does not add throughput.

### Runtime, driver, firmware

- `Fail to dma alloc 400000`: the driver needs a 4 MiB contiguous buffer per device; on fragmented hosts probe aborts and no `/dev/metis-*` node appears. Workaround: `echo 1 > /proc/sys/vm/compact_memory` and reload the module, or reboot.
- Windows: `axl_get_dma_link_info` (PCIe link info) is not implemented; large pages need the lock-pages privilege, otherwise slower regular pages are used.

### Other

- Metis Compute Board video output rendering is choppy (1–2 FPS). This impacts rendering to display only; inference performance is not impacted.
- MobileNetV3 may yield degraded accuracy on some combinations of hosts and cards.
- Device monitoring with AxMonitor is not supported on single-MSI hosts (for example Arduino Portenta X8 Mini).

## System Requirement

### Development Environment

For model compiling purposes, these are the host requirements:

| Requirement | Detail |
| --- | --- |
| OS | Linux Ubuntu 22.04, Ubuntu 24.04, Docker (on Windows or Linux), Windows + WSL/Ubuntu |
| CPU architecture | ARM64, x86, x86_64 |
| Recommended CPU | Intel Core-i5 or equivalent |
| Minimum System Memory | 16 GB (large models may require swap partition) |
| Recommended System Memory | 32 GB [1] |
| Python | 3.10 to 3.13 |

[1] The hardware requirements above are adequate for compilation of the models in our Model Zoo with the exception of Qwen3-VL-8B-Instruct which requires more than 32 GB System Memory. Larger models may require a larger system memory and/or a more powerful CPU.

### Runtime Environment

This release is expected to work with Intel (x86), AMD (x86) and Arm64 host CPUs on Linux, and natively on Windows for the runtime and the Pipeline Builder (`axelera-runtime2`). See [here](https://docs.axelera.ai/docs/hardware/metis/common/validated-host-systems) for a list of validated host systems for Axelera Metis AI Accelerator Cards.

## Further Support

- For blog posts, projects and technical support please visit the [Axelera AI Community](https://community.axelera.ai/).
- For technical documents and guides please visit [docs.axelera.ai](https://docs.axelera.ai/).

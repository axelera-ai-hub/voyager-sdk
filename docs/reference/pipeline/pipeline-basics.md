---
title: "Pipelines"
---
# Pipelines — How Inference Works

When you run `inference.py yolov5s-v7-coco usb:0`, you're running a **pipeline** — a chain of processing steps that takes video in and produces AI results out.

## What a pipeline does

```
Video Source → Pre-processing → AI Model (AIPU) → Post-processing → Display
```

Each step in the chain:

| Step | What happens | Where it runs |
|------|-------------|---------------|
| **Video source** | Reads frames from camera, file, or stream | CPU |
| **Pre-processing** | Resizes frames, converts color space, normalizes values | CPU (or GPU) |
| **AI Model** | Runs the neural network on the prepared frame | [AIPU](../../glossary.md#aipu) |
| **Post-processing** | Interprets raw model output (e.g., filters detections, applies NMS) | CPU |
| **Display** | Draws bounding boxes, labels, overlays on the frame | CPU/GPU |

## Pipeline types

The SDK supports three pipeline backends:

| Pipeline | Flag | What it uses | Best for |
|----------|------|-------------|----------|
| GStreamer | `--pipe gst` (default) | GStreamer framework | Production use, best performance |
| PyTorch | `--pipe torch` | PyTorch + ONNXRuntime | Debugging, CPU-only testing |
| PyTorch+AIPU | `--pipe torch-aipu` | PyTorch with AIPU offload | Hybrid debugging |

For most users, the default GStreamer pipeline is the right choice.

## The GStreamer pipeline

GStreamer is a media processing framework. The SDK uses it to build efficient video processing chains. Each step in the pipeline is a GStreamer **operator** (also called an element).

### Common operators you'll see

| Operator | What it does |
|----------|-------------|
| `transform_resize` | Resizes video frames to the model's expected input size |
| `transform_totensor` | Converts video frames to tensor format for the AI model |
| `decode_yolov5` | Interprets YOLOv5 model output into detection results |
| `inplace_nms` | Removes duplicate detections (Non-Maximum Suppression) |
| `inplace_draw` | Draws bounding boxes and labels on the video frame |
| `inplace_tracker` | Assigns tracking IDs to objects across frames |

For the full list of operators with all options, see [GStreamer Operators](gst-operators.md).

### How operators connect

```
camera → resize → totensor → [AIPU] → decode → nms → draw → display
```

Each operator receives data from the previous one and passes its output to the next. The SDK handles all of this automatically when you specify a model from the [Model Zoo](../models/model-zoo.md).

## YAML pipeline files

Every model in the Model Zoo has a YAML file that defines its pipeline. These files live in `ax_models/` and specify:

- Input resolution
- Pre-processing steps
- Model architecture and weights
- Post-processing steps
- Output format

You don't need to edit these for standard use. But if you want to customize behavior (e.g., change input resolution, add tracking), the YAML files are where you do it.

## Multi-source pipelines

You can run inference on multiple video sources simultaneously:

```bash
./inference.py yolov5s-v7-coco usb:0 usb:1 media/traffic.mp4
```

Each source gets its own pipeline instance, sharing the AIPU for model execution. See [Video Sources](../../tutorials/video-sources.md) for all source types.

## Building the GStreamer operators

All GStreamer plugins live under the `operators/` directory. Build them with:

```bash
source containerless.sh   # exports toolchain paths
make -C operators gst_ops_install
```

`gst_ops_install` does more than compile — it also:
- Downloads the matching ONNX Runtime binary distribution into `operators/onnxruntime/`
- Configures CMake/Ninja under `operators/<Debug|Release>/`
- Installs the resulting `.so` files and pkg-config stubs into `operators/lib` so they can be found by `GST_PLUGIN_PATH`

### Enabling CUDA (optional)

The operators build uses ONNX Runtime's CPU binaries by default. To enable the CUDA execution provider:

1. Ensure `nvidia-smi` shows the GPU, `nvcc` is installed, and `libcudnn8-dev` headers are present. The CUDA-enabled ONNX Runtime tarball is only available for Linux x86_64.
2. Rebuild with the CUDA flag:

```bash
make -C operators CUDA_AVAILABLE=1 gst_ops_install
```

`CUDA_AVAILABLE=1` tells the Makefile to download the CUDA tarball and registers the CUDA execution provider. Leaving the flag unset keeps the default CPU execution path.

---

## Performance considerations

| Factor | Effect | How to check |
|--------|--------|-------------|
| Input resolution | Higher resolution = slower pre-processing | Check model YAML for expected size |
| Model complexity | More parameters = longer AIPU execution | Compare FPS between models |
| Post-processing | NMS and drawing add CPU overhead | Use `--show-stream-timing` |
| Number of sources | More sources = more CPU load | Monitor with `--show-host-fps` |

## See also

- [First Inference](../../user-guides/first-inference.md) — run your first pipeline
- [inference.py](../tools/inference-py.md) — command-line options for controlling pipelines
- [Model Zoo](../models/model-zoo.md) — available models and their pipelines
- [GStreamer Operators](gst-operators.md) — full reference for all built-in operators
- [Model Formats](model-formats.md) — ONNX, .axmodel, model.json explained
- [Glossary: Pipeline](../../glossary.md#pipeline) — definition

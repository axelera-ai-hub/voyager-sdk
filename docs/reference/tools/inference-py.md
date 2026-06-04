# inference.py

The SDK's command-line tool for running [models](../../glossary.md#model) on [Metis](../../glossary.md#metis) hardware. It handles [compilation](../../glossary.md#compilation-model), execution, display, and performance reporting.

## Basic usage

```bash
./inference.py <model-name> <source> [options]
```

| Argument | What it is | Example |
|----------|-----------|---------|
| `<model-name>` | A model from the [Model Zoo](../../glossary.md#model-zoo) | `yolov5s-v7-coco` |
| `<source>` | Where the video comes from (see [Video Sources](../../tutorials/video-sources.md)) | `usb:0`, `media/traffic1_1080p.mp4` |

### Examples

Run object detection on a USB camera:
```bash
./inference.py yolov5s-v7-coco usb:0
```

Run classification on a video file with no display:
```bash
./inference.py resnet50-imagenet media/traffic1_1080p.mp4 --no-display
```

Measure accuracy against a validation [dataset](../../glossary.md#dataset):
```bash
./inference.py yolov5s-v7-coco dataset --no-display
```

Run on multiple sources simultaneously:
```bash
./inference.py yolov8s-coco-onnx usb:0 usb:1 media/traffic1_1080p.mp4
```

## What happens when you run it

1. **First run only:** The [pipeline compiler](../../glossary.md#compilation-model) builds the model for your hardware. This takes a few minutes and shows a progress bar. The result is cached.
2. **Every run:** The [pipeline](../../glossary.md#pipeline) starts — [pre-processing](../../glossary.md#pre-processing), [inference](../../glossary.md#inference) on the [AIPU](../../glossary.md#aipu), [post-processing](../../glossary.md#post-processing).
3. **Display:** A window shows the video with results overlaid (bounding boxes for detection, labels for classification) and performance metrics.
4. **On completion:** Average [throughput](../../glossary.md#fps) and CPU usage are printed to the terminal.

> [!NOTE]
> **Single images**
> If you pass a single image instead of a video, the window stays open until you press `q`. No end-of-run summary is printed.


## Options

### Display

| Option | What it does |
|--------|-------------|
| `--no-display` | Run headless. No window, just terminal output. Use for benchmarking or remote sessions. |
| `--display opengl` | Force OpenGL renderer (default if available, most efficient) |
| `--display opencv` | Force OpenCV renderer (slower, works on more systems) |
| `--display console` | Render to terminal using ANSI colors. Useful over SSH. |
| `--window-size WxH` | Set window size, e.g., `--window-size 1920x1080` |
| `--window-size fullscreen` | Fullscreen display |

### Performance

| Option | What it does |
|--------|-------------|
| `--frames N` | Stop after N frames (across all sources). Default: run all frames. |
| `--aipu-cores N` | Use N [AIPU](../../glossary.md#aipu) cores (default: all available, typically 4). Useful for testing multi-model scenarios. |
| `--show-host-fps` | Display host-specific FPS alongside the default metrics. |
| `--show-stream-timing` | Show latency and jitter information during the run. |

### Pipeline

| Option | What it does |
|--------|-------------|
| `--pipe gst` | Use [GStreamer](../../glossary.md#gstreamer) pipeline (default). Runs on AIPU. |
| `--pipe torch` | Use PyTorch pipeline with ONNXRuntime. Runs on CPU. |
| `--pipe torch-aipu` | Use PyTorch pipeline with the model offloaded to AIPU. |

> [!TIP]
> The default `gst` pipeline is almost always what you want. The `torch` option is useful for comparing AIPU results against CPU-only execution.


### Hardware acceleration

| Option | What it does |
|--------|-------------|
| `--enable-hardware-codec` | Prefer hardware video decoding. Default uses software decoding (better pipeline performance on most systems). |
| `--enable-vaapi` / `--disable-vaapi` | Control Intel VA-API acceleration for pre-processing. Auto-detected by default. |
| `--enable-opencl` / `--disable-opencl` | Control OpenCL acceleration for pre-processing. Auto-detected by default. |
| `--enable-opengl` / `--disable-opengl` | Control OpenGL for rendering. Auto-detected by default. |

> [!NOTE]
> VAAPI requires a VA-API compatible driver (iHD for Intel, radeonsi for AMD) and is not available on hosts without a supported GPU or VA-API driver

### Output

| Option | What it does |
|--------|-------------|
| `--save-output path.mp4` | Save the rendered output to an MP4 file. |
| `--save-output output%02d.mp4` | Save multiple streams separately (e.g., `output00.mp4`, `output01.mp4`). |

> [!NOTE]
> When saving output, all frames must be rendered. This may reduce system [FPS](../../glossary.md#fps) as the pipeline waits for video encoding.


## Output metrics

When a run completes, inference.py reports:

| Metric | What it means |
|--------|--------------|
| **System FPS** | End-to-end [throughput](../../glossary.md#system-throughput) including all processing and display |
| **Device FPS** | Raw [AIPU](../../glossary.md#aipu) [throughput](../../glossary.md#device-throughput) (what the hardware can do) |
| **CPU usage** | How much host CPU the pipeline consumes |
| **mAP** | [Accuracy](../../glossary.md#map) score (only when using `dataset` source) |

## Full option list

For all available options:

```bash
./inference.py --help
```

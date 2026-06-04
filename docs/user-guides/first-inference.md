# First Inference

Run object detection on your Metis hardware in under five minutes. This guide uses `inference.py`, the SDK's command-line tool for running any model from the [Model Zoo](../reference/models/model-zoo.md) with a single command. For other ways to run inference (Python API, C++ API, GStreamer), see the [Tutorials](../tutorials/README.md) section.

> [!CAUTION]
> **Before every session**
> Activate your environment first:
> ```bash
> source venv/bin/activate
> ```


## Quickstart

```bash
./inference.py yolov5s-v7-coco usb:0
```

No USB camera? Use a sample video:

```bash
./inference.py yolov5s-v7-coco media/traffic1_1080p.mp4
```

---

## Prerequisites

- SDK installed and environment activated (see [Install the SDK](sdk-install.md))
- Setup verified (see [Verify Your Setup](../getting-started/verify-setup.md))

## Step 1: Run object detection

With a USB camera:

```bash
./inference.py yolov5s-v7-coco usb:0
```

Or with a sample video file:

```bash
./inference.py yolov5s-v7-coco media/traffic1_1080p.mp4
```

> [!NOTE]
> **First run**
> The first time you run a model, the pipeline compiler builds it for your hardware. This takes a few minutes and shows a progress bar. Subsequent runs start immediately. See [Model Formats](../reference/pipeline/model-formats.md) for what gets built and where it's stored.


## Step 2: Read the output

A window opens showing the video feed with bounding boxes on detected objects. An instrumentation panel in the bottom-left displays:

| Metric | What it means |
|--------|--------------|
| **System throughput** | End-to-end performance of the complete pipeline |
| **Device throughput** | Maximum Metis device throughput (if all other pipeline elements keep up) |
| **CPU utilization** | How effectively the pipeline is offloaded from the host CPU |

When the pipeline finishes (or you press Ctrl+C), average metrics are printed to the terminal.

## Step 3: Try a different model

The SDK ships with many pre-optimized models. Try a classification model:

```bash
./inference.py resnet50-imagenet media/traffic1_1080p.mp4
```

Or a different object detector:

```bash
./inference.py yolov8s-coco-onnx usb:0
```

Browse the full list in the [Model Zoo](../reference/models/model-zoo.md).

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `command not found: inference.py` | Activate environment: `source venv/bin/activate` |
| No display window appears | Add `--no-display` to run headless, or check your display server |
| `No Axelera device found` | Run `axdevice` to check detection. See [Verify Setup](../getting-started/verify-setup.md) |
| Missing media files | Run `./install.sh --media` to download sample videos |

---

## Next steps

- [Video sources](../tutorials/video-sources.md) — cameras, RTSP streams, files, and multiple inputs
- [Measure accuracy](../tutorials/measure-accuracy.md) — benchmark a model against a validation dataset

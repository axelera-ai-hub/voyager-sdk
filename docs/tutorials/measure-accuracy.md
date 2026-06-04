# Measure Accuracy

Benchmark a model's accuracy against its validation dataset to verify it performs correctly on Metis hardware.

## Quickstart

```bash
source venv/bin/activate
./inference.py yolov5s-v7-coco dataset --no-display
```

---

## Prerequisites

- SDK installed and environment activated (see [Install the SDK](../user-guides/sdk-install.md))
- Internet connection (datasets are downloaded on first use)

> [!CAUTION]
> **Before every session**
> ```bash
> source venv/bin/activate
> ```


## Step 1: Run accuracy measurement

```bash
./inference.py yolov5s-v7-coco dataset --no-display
```

- `dataset` tells the tool to use the model's default validation dataset (COCO2017 for YOLO models, ImageNet for ResNet, etc.)
- `--no-display` runs headless — no video window, just metrics

> [!NOTE]
> **First run**
> The validation dataset is downloaded automatically on first use. This may take several minutes depending on your connection.


## Step 2: Read the results

On completion, the mean average precision (mAP) is printed to the terminal:

```
[INFO] Accuracy results:
  mAP@0.5: 0.XXX
  mAP@0.5:0.95: 0.XXX
```

Compare these values against the expected accuracy listed in the [Model Zoo](../reference/models/model-zoo.md) to verify your hardware is performing correctly.

## Step 3: Try other models

```bash
./inference.py resnet50-imagenet dataset --no-display
./inference.py yolov8s-coco-onnx dataset --no-display
```

> [!TIP]
> Dataset validation uses individual images rather than video, so throughput numbers will be lower than with a video source. This is expected — the purpose of this mode is accuracy verification, not performance benchmarking.


## Troubleshooting

| Symptom | Fix |
|---------|-----|
| Download fails or times out | Check internet connection, retry |
| Accuracy significantly below expected | Check firmware version matches SDK version with `axdevice` |
| Out of disk space | Datasets can be large (COCO is ~20GB). Free up space and retry |

---

## Next steps

- [Accuracy Metrics](../reference/measurement/accuracy-metrics.md) — understanding mAP, precision, recall, and FPS
- [Performance Metrics](../reference/measurement/performance.md) — measuring throughput and latency
- [Model Zoo](../reference/models/model-zoo.md) — browse available models and expected accuracy values
- [inference.py](../reference/tools/inference-py.md) — full command reference for benchmarking options

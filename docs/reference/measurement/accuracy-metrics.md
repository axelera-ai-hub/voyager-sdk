---
title: "Accuracy Metrics"
---
# Accuracy Metrics — Understanding the Numbers

When you run [accuracy benchmarking](../../tutorials/measure-accuracy.md), the SDK reports metrics like **mAP**, **precision**, and **recall**. This page explains what each number means in plain terms.

## mAP (Mean Average Precision)

[mAP](../../glossary.md#map) is the standard metric for object detection accuracy. It answers: **how well does the model find and correctly label objects?**

| mAP range | What it means |
|-----------|---------------|
| 0.90+ | Excellent — finds almost everything correctly |
| 0.70-0.90 | Good — reliable for most applications |
| 0.50-0.70 | Moderate — may miss some objects or make mistakes |
| Below 0.50 | Poor — needs improvement or different model |

### mAP@50 vs mAP@50:95

- **mAP@50** — Counts a detection as correct if the bounding box overlaps the real object by at least 50%. More lenient.
- **mAP@50:95** — Averages accuracy across overlap thresholds from 50% to 95%. Much stricter — the standard benchmark metric.

When you see "mAP" without qualification, it usually means mAP@50:95.

## Precision and Recall

These are the two components that make up mAP:

| Metric | Question it answers | High value means |
|--------|-------------------|------------------|
| **Precision** | Of the things the model detected, how many were real? | Few false alarms |
| **Recall** | Of the real objects, how many did the model find? | Few missed objects |

### The trade-off

You can't maximise both. Increasing sensitivity (recall) means more false positives (lower precision). The confidence threshold controls this balance:

```bash
# Lower threshold = more detections (higher recall, lower precision)
# Higher threshold = fewer but more confident detections (higher precision, lower recall)
```

## FPS (Frames Per Second)

Performance metrics reported during inference:

| Metric | What it measures |
|--------|-----------------|
| **System FPS** | End-to-end throughput including pre/post-processing |
| **Device FPS** | How fast the [AIPU](../../glossary.md#aipu) processes frames (model execution only) |
| **Host FPS** | CPU-side processing speed (shown with `--show-host-fps`) |

### Which FPS matters?

- **System FPS** is what your application will actually achieve — it's the real-world number.
- **Device FPS** shows the AIPU's raw capability — useful for understanding where the bottleneck is.
- If Device FPS >> System FPS, the bottleneck is in pre/post-processing (CPU-side).
- If Device FPS ≈ System FPS, the model is the bottleneck.

## Reading benchmark output

When you run:

```bash
./inference.py yolov5s-v7-coco dataset --frames 5000 --no-display
```

The output includes:

```
System: 125.3 fps  Device: 142.7 fps  CPU: 23%  mAP@50: 0.547  mAP@50:95: 0.371
```

| Field | Meaning |
|-------|---------|
| `System: 125.3 fps` | End-to-end throughput |
| `Device: 142.7 fps` | AIPU processing speed |
| `CPU: 23%` | Host CPU utilization |
| `mAP@50: 0.547` | Accuracy at 50% overlap threshold |
| `mAP@50:95: 0.371` | Accuracy averaged across thresholds (the headline number) |

## Comparing models

When choosing between models, consider both accuracy and speed:

| Model | mAP@50:95 | System FPS | Use case |
|-------|-----------|-----------|----------|
| YOLOv5s | Good | Very fast | Real-time applications, edge deployment |
| YOLOv8s | Better | Fast | Balanced accuracy/speed |
| YOLOv8l | Best | Slower | When accuracy matters most |

Smaller models (s = small) are faster. Larger models (l = large) are more accurate. Choose based on your application's requirements.

## See also

- [Measure Accuracy](../../tutorials/measure-accuracy.md) — how to run benchmarks
- [Model Zoo](../models/model-zoo.md) — available models with performance data
- [Glossary: mAP](../../glossary.md#map) — definition

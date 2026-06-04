---
title: "Performance Metrics"
---
# Performance Metrics

How to measure throughput, latency, and accuracy across different pipeline modes, and how to identify bottlenecks.

---

## Pipeline modes

The SDK supports three pipeline modes, each measuring something different.

| Mode | What runs where | Use it to measure |
|------|----------------|-------------------|
| `gst` (default) | GStreamer pipeline + AIPU inference | End-to-end real-world throughput |
| `torch-aipu` | Python pre/post-processing + AIPU inference | Quantization accuracy loss |
| `torch` | Everything on CPU, FP32 | Original model accuracy (baseline) |

### Isolating quantization accuracy loss

Run the same model in all three modes on a validation dataset:

```bash
# FP32 baseline (original model accuracy)
./inference.py yolov5s-v7-coco dataset --no-display --pipe=torch

# AIPU with Python pipeline (isolates quantization effect)
./inference.py yolov5s-v7-coco dataset --no-display --pipe=torch-aipu --aipu-cores=1

# Full end-to-end (GStreamer pipeline + AIPU)
./inference.py yolov5s-v7-coco dataset --no-display
```

The difference between `torch` and `torch-aipu` is the accuracy cost of int8 quantization alone. This is the number to use when comparing Metis against other hardware.

The `gst` mode adds hardware accelerated pre-processing (VA-API, OpenCL), which may introduce small additional differences compared to the training pipeline.

---

## Metrics

Four metrics are available during inference runs.

### Host FPS

Throughput at the point where frames are dispatched to the Metis accelerator. Because transfers to and from the device are pipelined, this reflects the maximum throughput the AIPU can sustain for this model.

If Host FPS ≈ System FPS, inference is the bottleneck. If Host FPS >> System FPS, something else in the pipeline is slower.

### System FPS

End-to-end throughput across the entire pipeline. This is `1 / time_for_slowest_element` — the most useful single number for real-world performance.

### CPU Usage

Host CPU utilization as a percentage of total available compute (all cores). High CPU usage with low System FPS typically points to a pre- or post-processing bottleneck.

### Stream Timing

Per-frame latency (time from source to application code) and jitter (variation in latency). Enabled with `--show-stream-timing`.

High jitter is usually improved by increasing buffer size (see `--rtsp-latency` in [inference.py](../tools/inference-py.md)).

---

## Controlling which metrics are shown

```bash
./inference.py yolov5s-v7-coco usb:0 \
  --show-host-fps \
  --show-system-fps \
  --show-cpu-usage \
  --show-stream-timing
```

All four are on by default except `--show-stream-timing`.

---

## Pipeline stages

Each frame moves through four stages before reaching your application:

| Stage | Where it runs | What it does |
|-------|--------------|--------------|
| **Input conversion** | Host CPU / VA-API | Decode compressed video (H.264/HEVC), convert color space (YUV → RGB), resize to model input resolution |
| **Tensor preparation** | Host CPU / OpenCL | Normalize pixel values, apply padding, layout conversion (HWC → CHW), quantize to int8 |
| **AIPU inference** | Metis AIPU | Execute the compiled model graph on hardware |
| **Output decode** | Host CPU | Convert raw output tensors to structured results (bounding boxes, class labels, scores), apply NMS |

In the `--show-stats` table, these stages correspond to:
- `axtransform-colorconvert0` and similar — input conversion
- `inference-task0:libtransform_*` elements — tensor preparation
- `inference-task0:inference` — AIPU execution
- `inference-task0:libdecode_*` and `libinplace_nms_*` — output decode

The `Inference latency` row is the wall-clock time from when the first byte enters the AxInferenceNet element to when decoded results emerge — it spans all four stages end-to-end for a single frame. `Total latency` includes any upstream buffering from the source.

---

## Identifying bottlenecks

When System FPS is lower than expected, use `--show-stats` to get a per-element timing breakdown. Disable OpenCL double-buffering first so the profiler captures GPU kernel times accurately:

```bash
AXELERA_USE_CL_DOUBLE_BUFFER=0 ./inference.py yolov5s-v7-coco media/traffic3_720p.mp4 \
  --show-stats --no-display
```

Example output:

```
========================================================================
Element                                         Time(μs)   Effective FPS
========================================================================
qtdemux0                                              14        68,221.1
h264parse0                                            22        43,575.6
capsfilter0                                            8       115,892.2
decodebin-link0                                       18        53,824.9
axtransform-colorconvert0                            464         2,154.9
inference-task0:libtransform_resize_cl_0             441         2,265.7
inference-task0:libtransform_padding_0               434         2,300.8
inference-task0:inference                          1,217           821.5
inference-task0:Inference latency                 28,718             n/a
inference-task0:libdecode_yolov5_0                   251         3,975.7
inference-task0:libinplace_nms_0                      21        45,654.3
inference-task0:Postprocessing latency               642             n/a
inference-task0:Total latency                     37,057             n/a
========================================================================
End-to-end average measurement                                     809.3
========================================================================
```

**Reading the table:**

- The lowest **Effective FPS** row identifies the bottleneck. In this example, `inference` at 821.5 FPS is the limiting element — everything else is faster.
- Elements prefixed `inference-task0:` are operators inside the [AxInferenceNet](../pipeline/pipeline-basics.md) GStreamer element (resize, padding, the AIPU inference itself, decode, NMS).
- `Inference latency` and `Total latency` rows show cumulative latency in microseconds, not throughput — the `n/a` in the FPS column is expected.
- The early GStreamer elements (`qtdemux0`, `h264parse0`) operate on compressed data, so their timings are not meaningful for per-frame analysis.

> [!TIP]
> If `axtransform-colorconvert0` appears to be a bottleneck, check whether VA-API hardware decode is enabled. For large resolutions (1080p+), color conversion can become significant.


---

## Performance vs accuracy trade-off

Hardware-accelerated pipelines (`gst` mode) typically show slightly lower accuracy than CPU pipelines (`torch` mode). This is expected: hardware accelerators like VA-API support a limited set of resize and color conversion algorithms, which may differ slightly from the Python libraries used during model training.

The accuracy difference is usually very small and acceptable for production use. Use `torch-aipu` mode to quantify the quantization-only contribution before attributing any gap to hardware pre-processing.

---

## See also

- [inference.py](../tools/inference-py.md) — all command-line flags including `--show-stats`
- [Accuracy Metrics](accuracy-metrics.md) — mAP and classification accuracy measurement
- [GStreamer Operators](../pipeline/gst-operators.md) — the operators that appear in the element timing table

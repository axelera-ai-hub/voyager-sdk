---
title: "axrunmodel"
---
# axrunmodel

Runs a compiled model directly against the AIPU with minimal overhead — useful for measuring peak hardware throughput without the full inference pipeline.

```bash
axrunmodel <path/to/model.json>
```

Point it at the `model.json` file in your compiled model directory (see [Model Formats](../pipeline/model-formats.md)).

**Example:**
```bash
axrunmodel build/yolov8l-coco-onnx/yolov8l-coco-onnx/1/model.json
```

`axrunmodel` uses the same input data for every frame, which removes input bottlenecks and measures maximum AIPU throughput in ideal conditions.

---

## Output

After the run completes, `axrunmodel` reports:

| Metric | What it means |
|--------|---------------|
| **Device FPS** | 1 / execution_time at the device level, including data transfer. Not a throughput number — closer to "max instantaneous rate". |
| **Host FPS** | 1 / execution_time at the host level, including PCIe transfer overhead |
| **System FPS** | total_frames / total_time — the meaningful throughput number |

---

## Options

| Option | Default | Description |
|--------|---------|-------------|
| `-d N`, `--devices N` | all | Comma-separated device indices to use (e.g., `-d 0,1`). Run `axdevice` to list device indices. |
| `--seconds N` | `10` | Run for N seconds |
| `--aipu-cores N` | all (4) | Number of AIPU cores to use. Reduce to test single-core or leave headroom for other models. |
| `--throttle-fps N` | none | Cap System FPS at N frames per second |
| `--double-buffer` / `--no-double-buffer` | enabled | Enable/disable double-buffering optimization |
| `--input-dmabuf` / `--no-input-dmabuf` | enabled | Enable/disable DMA input buffers |
| `--output-dmabuf` / `--no-output-dmabuf` | enabled | Enable/disable DMA output buffers |
| `--show-bar-chart` | off | Display FPS per frame as a horizontal bar chart over time — useful for spotting ramp-up or variability |
| `--show-histogram` | off | Display FPS distribution as a histogram — shows how consistent performance is |

Full option list:
```bash
axrunmodel --help
```

---

## See also

- [Model Formats](../pipeline/model-formats.md) — where `model.json` comes from
- [Performance Metrics](../measurement/performance.md) — full pipeline benchmarking with `inference.py`
- [axdevice](axdevice.md) — list device indices for use with `--devices`

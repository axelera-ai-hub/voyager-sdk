---
title: "Hardware Overview"
---
# Hardware Overview

What Metis is, how it works, and what numbers like "TOPS" and "4 cores" mean in practice.

---

## Metis: the AIPU

Metis is an **AIPU** (AI Processing Unit) designed specifically for neural network inference. Unlike a CPU or GPU, it is purpose-built for the matrix-vector multiplications that dominate neural network computation.

### Key specifications

| Specification | Value |
|---------------|-------|
| Architecture | In-Memory Compute (IMC) |
| AI cores | 4 |
| Precision | INT8 (weights and activations) |
| Clock speed | 800 MHz (default) |
| Peak compute | ~214 TOPS (INT8) |
| DDR memory | 4 GB (PCIe / M.2) |
| Form factors | PCIe, M.2, Compute Board (SBC) |
| Interface | PCIe Gen 3 x4 |

---

## What "In-Memory Compute" means

Traditional processors move data between separate memory and compute units. Metis uses In-Memory Compute: the matrix-vector multiply (MVM) happens inside the SRAM arrays where the model weights are stored. This eliminates the memory bandwidth bottleneck that limits GPU efficiency on small-to-medium models.

The practical effect: Metis achieves high throughput on the neural network core while consuming much less power than a GPU for the same task.

---

## The 4 cores

Metis has 4 AI cores that share the on-chip SRAM. Each core can run an independent model instance, or the 4 cores can work together on a single model (sharing memory). This determines your compilation strategy:

- **Independent cores** (`resources_used: 0.25` per model): Lower latency, more flexible — good for real-time pipelines with a single model type.
- **Shared-memory batch** (`resources_used: 1.0`, `aipu_cores_used: 4`): Higher throughput on memory-intensive models — good for throughput-first workloads.

See [Compiler Configuration](../compiler/compiler-configs.md) for how to configure this.

---

## The host pipeline

Metis does not work alone. A typical inference pipeline involves both the host CPU and the AIPU:

```
Camera / File
    ↓
Host: decode compressed video (H.264, etc.)
    ↓
Host: color convert, resize, letterbox
Host: normalize, quantize to INT8
    ↓
PCIe transfer → Metis AIPU
    ↓
AIPU: neural network forward pass (INT8)
    ↓
PCIe transfer ← results
    ↓
Host: dequantise, decode output tensors
Host: NMS, bounding box extraction
    ↓
Application: FrameResult with metadata
```

Host-side acceleration (VA-API for video decode, OpenCL for pre/post-processing, OpenGL for rendering) can be enabled to reduce CPU load. See [inference.py](../tools/inference-py.md#hardware-acceleration) for the relevant flags.

---

## What TOPS means

TOPS = Tera Operations Per Second. For INT8 matrix multiply:
- 1 TOPS = 10¹² multiply-accumulate operations per second

Metis delivers ~214 INT8 TOPS. This is the theoretical peak for INT8 matrix-vector multiply across all 4 cores at 800 MHz.

Real-world throughput depends on the model: how much of the computation fits in on-chip memory, how much PCIe transfer is needed, and how efficient the pipeline stages are. Use `--show-stats` with `inference.py` to measure actual pipeline throughput rather than relying on peak TOPS numbers.

---

## Form factors

| Form factor | Typical use |
|-------------|-------------|
| PCIe | Workstation, server, edge server |
| M.2 | Embedded, compact edge devices |
| Compute Board (SBC) | Standalone evaluation and prototyping; mini-ITX with ARM host onboard |

All form factors use the same AIPU die and software stack. Each is available in multiple variants differing by AIPU count (1× or 4×), memory, and cooling (active or passive) — see the product datasheets for specific configurations.

---

## Operating environment

- Ambient temperature range: −20°C to +70°C
- Software throttling: configurable (default: disabled — see [Thermal Management](../../user-guides/thermal.md))
- Hardware shutdown: 120°C junction temperature (automatic, requires power cycle to recover)

---

## See also

- [Thermal Management](../../user-guides/thermal.md) — temperature monitoring and throttling
- [axdevice](../tools/axdevice.md) — inspect connected devices and firmware versions
- [Compiler Configuration](../compiler/compiler-configs.md) — how to use 1 vs 4 cores
- [Performance Metrics](../measurement/performance.md) — measuring real throughput

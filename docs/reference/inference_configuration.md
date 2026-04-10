# Inference Configuration Guide

This document explains how to configure the inference pipeline using `handle_*` flags in your model YAML files. These flags control data transformations and optimize runtime performance.

## Prerequisites

**Before using this guide, you must deploy your model first:**
```bash
./deploy.py <network-name>
```

Deployment creates the compiled model artifacts (quantized model, preamble.onnx, postamble.onnx, quantization parameters). This guide explains how to configure the RUNTIME behavior when you run inference on your deployed model.

**See also:**
- **[deploy.md](deploy.md)** - How to deploy and compile models
- **[inference.md](inference.md)** - How to run `./inference.py` CLI tool

---

## Quick Navigation

- [The Challenge](#the-fundamental-challenge)
- [Data Flow](#complete-data-flow)
- [Configuration Reference](#configuration-flags-reference)
- [Flag Details](#detailed-flag-descriptions)
- [Examples](#pipeline-scenarios)
- [Recommendations](#recommendations)

---

## The Fundamental Challenge

**AI models work with float32 data, but Axelera hardware uses quantized int8 with special memory alignment.**

```
Model Space:     float32, no padding, any layout
                       ↕
AIPU Hardware:   int8, padded, NHWC layout
```

**Required transformations:**
- Quantization (float32 → int8) before AIPU
- Padding (alignment bytes) before AIPU
- Dequantization (int8 → float32) after AIPU
- Depadding (remove alignment) after AIPU
- Transpose (layout fix) after AIPU

**The `handle_*` flags control WHO handles these: framework or decoder.**

---

## Complete Data Flow

```
Input (float32)
    ↓
┌─────────────────────────────────┐
│  PREAMBLE (optional)            │  ← CPU preprocessing
│  + QUANTIZE (float32 → int8)    │  ← handle_preamble
│  + PAD (alignment)              │     controls this
└─────────────────────────────────┘
    ↓
┌═════════════════════════════════┐
║  AIPU HARDWARE (Metis)          ║  ← Quantized inference
┗═════════════════════════════════┛
    ↓ (int8, padded output)
┌─────────────────────────────────┐
│  DEPAD + DEQUANTIZE + TRANSPOSE │  ← handle_* flags
│  + POSTAMBLE (optional)         │     control this
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  DECODER (task-specific)        │  ← Receives float32
└─────────────────────────────────┘     or int8+params
    ↓
Output (float32 results)
```

---

## Configuration Flags Reference

### YAML Structure

```yaml
inference:
  # Convenience flag (sets all 4 below)
  handle_all: True/False/None  # default: None

  # Individual flags (default: True for each)
  handle_dequantization_and_depadding: True/False
  handle_transpose: True/False
  handle_postamble: True/False
  handle_preamble: True/False

  # Performance tuning
  dequantize_using_lut: True/False  # default: True

  # ONNX Runtime config (for postamble)
  postamble_onnxruntime_intra_op_num_threads: 4
  postamble_onnxruntime_inter_op_num_threads: 4
  postamble_onnx: "path/to/postamble.onnx"
```

### Quick Reference

| Flag | True (Framework) | False (Decoder) |
|------|------------------|-----------------|
| `handle_all` | Sets all to True | Sets all to False |
| `handle_dequantization_and_depadding` | Adds transform → float32 | Decoder gets int8+params |
| `handle_transpose` | Fixes layout | Decoder handles |
| `handle_postamble` | Runs postamble.onnx | Decoder handles |
| `handle_preamble` | Special preprocessing | Standard padding |

---

## Detailed Flag Descriptions

### handle_all (Convenience Flag)

**Purpose:** Sets all 4 individual flags at once

**Values:**
- **`None` (default):** Use individual flag settings
- **`True`:** Framework handles all → decoder gets float32
- **`False`:** Decoder handles all → decoder gets int8+params

**When to use:**
- **Starting?** Use `True` - simple, works immediately
- **Optimizing?** Use `False` - fuse operations in decoder
- **Fine-tuning?** Omit `handle_all` and set individual flags

**Important:** `handle_all` (True/False) and individual flags are mutually exclusive. Setting both raises an error. Use either `handle_all` OR individual flags, not both.

**Rule of thumb:**
- Small models: `False` may help (fusion opportunities)
- Large models: `True` works well (transform overhead is small)

---

### handle_dequantization_and_depadding

**Purpose:** Controls data format from inference block

**`True` (default):**
```
AIPU (int8, padded) → libtransform_paddingdequantize.so
  → Decoder receives: float32 arrays
```

**`False`:**
```
AIPU (int8, padded) → (no transform)
  → Decoder receives: int8 + quantization parameters
  → Decoder must depad and dequantize
```

**Why False?** Working with int8 directly is faster for simple operations (finding max, comparing). Can fuse with decoder logic.

---

### handle_transpose

**Purpose:** Controls dimension reordering

**Background:** AIPU outputs NHWC [N,H,W,C], some decoders expect NCHW [N,C,H,W]

**`True` (default):** Framework transposes → decoder gets expected layout
**`False`:** Decoder receives raw NHWC → decoder handles if needed

**Why False?** Transpose is slow! If decoder can work with NHWC directly, skip it.

---

### handle_postamble

**Purpose:** Controls compiler-extracted postprocessing operations

**Background:** Compiler extracts CPU-friendly operations to `postamble.onnx`

**`True` (default):**
```
libtransform_postamble.so bundles:
  - DEPAD
  - DEQUANTIZE
  - TRANSPOSE
  - POSTAMBLE (runs postamble.onnx via ONNX Runtime)
→ Decoder gets final float32 results
```

**`False`:**
```
Decoder must handle postamble operations
  Option A: Implement in C++ (can fuse with decoding)
  Option B: Use postamble_onnx parameter (run via ORT)
```

**Why False?** Fuse postamble with decoder operations for better performance.

**Dependency:** When `handle_postamble=True`, the framework also requires `handle_dequantization_and_depadding=True` and `handle_transpose=True`. If they are set to `False`, they will be automatically forced to `True` with a warning.

---

### handle_preamble

**Purpose:** Controls preprocessing before AIPU

**`True` (default):**
```
If special preprocessing pattern detected in preamble:
  libtransform_yolopreproc.so (pattern-specific preprocessing)
Else:
  libtransform_padding.so (standard)
```

**`False`:** Standard padding only

**Current Status:** ⚠️ Partially implemented
- Compiler extracts preamble.onnx (automatic)
- Runtime detects and handles specific preprocessing patterns
- Does NOT fully load/execute arbitrary preamble.onnx (yet)

**For most models:** Use `True` (default) - special preamble patterns are uncommon.

---

### dequantize_using_lut

**Purpose:** HOW to perform dequantization

**`True` (default):** Lookup Table
- Pre-compute 256 int8 → float32 mappings
- Fast for batch_size = 1

**`False`:** Calculate on-the-fly
- Compute: `(int8 - zero_point) * scale`
- Better for batch_size > 1

---

### postamble_onnx

**Purpose:** Specify postamble file for separate execution

**When needed:** Only if `handle_postamble=False` AND you want ONNX Runtime execution

```yaml
inference:
  handle_postamble: False
  postamble_onnx: "build/model/quantized/postamble.onnx"
```

---

### ONNX Runtime Threading

**`postamble_onnxruntime_intra_op_num_threads`:** Threads WITHIN one operation (default: 4)
**`postamble_onnxruntime_inter_op_num_threads`:** Operations running in PARALLEL (default: 4)

---

## Pipeline Scenarios

### Scenario 1: handle_all = True (Simple)

```yaml
inference:
  handle_all: True
```

**Pipeline:**
```
Input → Padding+Quantize → AIPU → Postamble Transform → Decoder
                                   (depad+dequant+
                                    transpose+postamble)
```

**Characteristics:**
- ✅ Simplest - works immediately
- ✅ No custom code needed
- ⚠️ Multiple pipeline stages

**Best for:** Prototyping, large models

---

### Scenario 2: handle_all = False (Optimized)

```yaml
inference:
  handle_all: False
```

**Pipeline:**
```
Input → Padding+Quantize → AIPU → Decoder
                                   (depad+dequant+transpose+
                                    postamble+decode, all fused)
```

**Characteristics:**
- ✅ Fastest - fewer stages, operations fused
- ⚠️ Requires full C++ decoder implementation

**Best for:** Production, small models, real-time

---

### Scenario 3: Fine-Tuned (Hybrid)

```yaml
inference:
  handle_all: None
  handle_dequantization_and_depadding: True  # Framework
  handle_transpose: False  # Decoder (can fuse)
  handle_postamble: False  # Decoder (can fuse)
```

**Pipeline:**
```
Input → Preamble → AIPU → Depad+Dequant → Decoder
                          (framework)     (transpose+postamble+
                                           decode, fused)
```

**Characteristics:**
- ✅ Balanced - framework handles complex dequant
- ✅ Decoder optimizes specific operations

**Best for:** Production after profiling

---

## Recommendations

### Starting Out
```yaml
inference:
  handle_all: True  # Maximum simplicity
```

### Production: Small Models
```yaml
inference:
  handle_all: False  # Maximum performance
```

### Production: Large Models
```yaml
inference:
  handle_all: True  # Simplicity wins
```

### After Profiling
```yaml
inference:
  handle_all: None
  handle_dequantization_and_depadding: True
  handle_transpose: False
  handle_postamble: False
  dequantize_using_lut: True  # if batch_size=1
```

---

## Trade-offs

| Approach | Simplicity | Performance | Custom Code |
|----------|-----------|-------------|-------------|
| **handle_all: True** | ✅✅✅ | ⚠️ Good | ❌ None |
| **handle_all: False** | ⚠️ Complex | ✅✅✅ | ✅✅ Full |
| **Fine-tuned** | ⚠️⚠️ | ✅✅ | ✅ Selective |

---

## Key Takeaways

1. **Start simple:** `handle_all: True` for development
2. **Profile first:** Measure before optimizing
3. **WHO not WHETHER:** Flags control who handles transformations (framework vs decoder)
4. **Bundled operations:** `handle_postamble=True` bundles multiple operations efficiently
5. **Preamble special:** Pattern detection for certain preprocessing operations
6. **LUT for singles:** Use for batch_size=1
7. **Large models:** Transform overhead is small

**Golden Rule:** Start with `handle_all: True`, optimize only if measurements show it's needed!

---

## See Also

- **[deploy.md](deploy.md)** - Model deployment and compilation
- **[inference.md](inference.md)** - Inference CLI tool usage
- **[Pipeline Templates](../../pipeline-template/)** - Pre-configured examples

---
title: "Inference Configuration"
description: Configure handle_* YAML flags to control how data transformations are split between the framework and your decoder
---
# Inference Configuration

How to tune the data transformation pipeline between the AIPU and your decoder using `handle_*` flags in model YAML files.

> [!NOTE]
> **Prerequisites**
> You must [deploy your model](../tools/deploy-py.md) before configuring inference. Deployment creates the compiled model artifacts (quantized model, preamble, postamble). This page covers the **runtime** behavior when you run inference on that deployed model.


---

## The problem these flags solve

AI models work with float32. The Metis AIPU works with quantized int8 in padded, NHWC layout. Between model output and your decoder, several transformations are needed:

| Transformation | Direction | What it does |
|---|---|---|
| Quantize | Before AIPU | float32 → int8 |
| Pad | Before AIPU | Add alignment bytes |
| Depad | After AIPU | Remove alignment bytes |
| Dequantize | After AIPU | int8 → float32 |
| Transpose | After AIPU | NHWC → NCHW (if decoder expects it) |
| Postamble | After AIPU | Run compiler-extracted postprocessing |

The `handle_*` flags control **who** does each transformation — the framework (automatic) or your decoder (manual, but faster if you can fuse operations).

---

## Data flow

```
Input (float32)
    ↓
┌─────────────────────────────────┐
│  PREAMBLE (optional)            │  ← handle_preamble
│  + QUANTIZE + PAD               │
└─────────────────────────────────┘
    ↓
┌═════════════════════════════════┐
║  AIPU (Metis)                   ║  ← int8 inference
┗═════════════════════════════════┛
    ↓ (int8, padded)
┌─────────────────────────────────┐
│  DEPAD + DEQUANTIZE + TRANSPOSE │  ← handle_* flags
│  + POSTAMBLE (optional)         │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│  DECODER (task-specific)        │  ← float32 or int8+params
└─────────────────────────────────┘
    ↓
Output (results)
```

---

## Quick start

For most users, the defaults work fine — you don't need to set anything. If you want to tune performance, start here:

```yaml
# Option 1: Framework handles everything (default behavior)
inference:
  handle_all: True

# Option 2: Decoder handles everything (best performance, requires custom decoder)
inference:
  handle_all: False

# Option 3: Fine-tune individual flags
inference:
  handle_dequantization_and_depadding: True
  handle_transpose: False
  handle_postamble: False
```

**Rule of thumb:** Start with `handle_all: True` (or just omit the flags entirely). Only switch to `False` or fine-tune after profiling shows a bottleneck.

---

## YAML reference

```yaml
inference:
  # Convenience flag — sets all 4 individual flags at once
  handle_all: True/False/None     # default: None (use individual settings)

  # Individual flags (each defaults to True)
  handle_dequantization_and_depadding: True/False
  handle_transpose: True/False
  handle_postamble: True/False
  handle_preamble: True/False

  # Performance tuning
  dequantize_using_lut: True/False  # default: True

  # ONNX Runtime config (only relevant when handle_postamble is False)
  postamble_onnx: "path/to/postamble.onnx"
  postamble_onnxruntime_intra_op_num_threads: 4
  postamble_onnxruntime_inter_op_num_threads: 4
```

> [!WARNING]
> `handle_all` and individual flags are mutually exclusive. Setting both raises an error — use one approach or the other.


---

## Flag reference

### handle_all

Sets all 4 individual flags at once.

| Value | Effect | Use when |
|---|---|---|
| `None` (default) | Each flag uses its own setting | Fine-tuning individual flags |
| `True` | Framework handles all transforms → decoder gets float32 | Starting out, large models |
| `False` | Decoder handles all transforms → decoder gets int8 + params | Production, performance-critical |

### handle_dequantization_and_depadding

| Value | Decoder receives | When to use |
|---|---|---|
| `True` (default) | float32 arrays | Most cases |
| `False` | int8 + quantization parameters | Decoder can fuse dequant with its own logic |

### handle_transpose

| Value | Decoder receives | When to use |
|---|---|---|
| `True` (default) | Expected layout (NCHW if needed) | Most cases |
| `False` | Raw NHWC from AIPU | Decoder works with NHWC directly |

Transpose is expensive — skip it if your decoder doesn't need NCHW.

### handle_postamble

| Value | Decoder receives | When to use |
|---|---|---|
| `True` (default) | Final float32 results (postamble already applied) | Most cases |
| `False` | Pre-postamble output (decoder handles the rest) | Fusing postamble with decode logic |

> [!NOTE]
> **Dependency**
> When `handle_postamble=True`, the framework also forces `handle_dequantization_and_depadding=True` and `handle_transpose=True` (with a warning if you set them to False).


### handle_preamble

| Value | Effect | When to use |
|---|---|---|
| `True` (default) | Detects and applies special preprocessing patterns | Most models |
| `False` | Standard padding only | Override pattern detection |

Currently partially implemented — the runtime detects specific preprocessing patterns (e.g. YOLO preprocessing) but does not execute arbitrary preamble ONNX files.

### dequantize_using_lut

| Value | Method | Best for |
|---|---|---|
| `True` (default) | Lookup table (pre-computed 256 int8 → float32 mappings) | batch_size = 1 |
| `False` | Calculate on-the-fly: `(int8 - zero_point) * scale` | batch_size > 1 |

---

## Pipeline scenarios

### Simple: `handle_all: True`

```
Input → Preamble+Quantize+Pad → AIPU → Depad+Dequant+Transpose+Postamble → Decoder
                                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
                                        framework handles all of this
```

Best for: prototyping, large models, no custom decoder needed.

### Optimized: `handle_all: False`

```
Input → Preamble+Quantize+Pad → AIPU → Decoder
                                        ^^^^^^^
                                        decoder does everything (fused)
```

Best for: production, small models, real-time. Requires a custom C++ decoder.

### Hybrid: individual flags

```yaml
inference:
  handle_dequantization_and_depadding: True   # framework handles the complex part
  handle_transpose: False                      # decoder skips transpose (works with NHWC)
  handle_postamble: False                      # decoder fuses postamble with decode
```

Best for: production after profiling. Framework handles the hard part, decoder optimizes what it can.

---

## Trade-off summary

| Approach | Simplicity | Performance | Custom code needed |
|---|---|---|---|
| `handle_all: True` | High | Good | None |
| `handle_all: False` | Low | Best | Full C++ decoder |
| Fine-tuned | Medium | Better | Selective |

**Start with `handle_all: True`. Optimize only if measurements show it's needed.**

---

## See also

- [deploy.py](../tools/deploy-py.md) — model deployment and compilation
- [inference.py](../tools/inference-py.md) — inference CLI tool
- [Pipelines](./pipeline-basics.md) — how the inference pipeline works

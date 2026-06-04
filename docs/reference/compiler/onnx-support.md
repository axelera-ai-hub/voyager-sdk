---
title: "ONNX Operator Support"
---
# ONNX Operator Support

Which ONNX operators the Metis AIPU supports natively, and which run on the host CPU.

> [!NOTE]
> **Auto-generated tables**
> The detailed per-operator constraint tables (attribute restrictions, dtype limits) are auto-generated and live in the SDK at `docs/reference/onnx-opset{14,15,16,17}-support.md`. This page provides the consolidated summary.


---

## Support levels

| Level | Meaning |
|-------|---------|
| **Supported** | Fully accelerated on the AIPU with no restrictions |
| **Constrained** | Accelerated on the AIPU for specific configurations (see per-opset tables for attribute/dtype restrictions) |
| **Not supported** | Falls back to host CPU via ONNX Runtime preamble/postamble |

A "Constrained" operator is still hardware-accelerated — it just has attribute limits (e.g. specific kernel sizes, data types, padding modes). If your model uses a constrained operator outside its supported configuration, that layer falls back to CPU.

---

## Supported operators (opsets 14–17)

The following operators are supported or constrained across opsets 14–17. Operators added in later opsets are noted.

| Operator | Opset 14 | Opset 15 | Opset 16 | Opset 17 |
|----------|:--------:|:--------:|:--------:|:--------:|
| Add | Constrained | Constrained | Constrained | Constrained |
| AveragePool | Constrained | Constrained | Constrained | Constrained |
| BatchNormalization | Supported | Supported | Supported | Supported |
| Clip | Constrained | Constrained | Constrained | Constrained |
| Concat | Constrained | Constrained | Constrained | Constrained |
| Conv | Constrained | Constrained | Constrained | Constrained |
| ConvTranspose | Constrained | Constrained | Constrained | Constrained |
| Flatten | Constrained | Constrained | Constrained | Constrained |
| Gemm | Constrained | Constrained | Constrained | Constrained |
| GlobalAveragePool | Supported | Supported | Supported | Supported |
| GlobalMaxPool | Supported | Supported | Supported | Supported |
| HardSigmoid | Constrained | Constrained | Constrained | Constrained |
| HardSwish | Supported | Supported | Supported | Supported |
| LeakyRelu | Supported | Supported | Supported | Supported |
| MatMul | Constrained | Constrained | Constrained | Constrained |
| MaxPool | Constrained | Constrained | Constrained | Constrained |
| Mul | Constrained | Constrained | Constrained | Constrained |
| PRelu | Constrained | Constrained | Constrained | Constrained |
| Pad | Constrained | Constrained | Constrained | Constrained |
| Relu | Supported | Supported | Supported | Supported |
| Reshape | Constrained | Constrained | Constrained | Constrained |
| Resize | Constrained | Constrained | Constrained | Constrained |
| Selu | Constrained | Constrained | Constrained | Constrained |
| Sigmoid | Supported | Supported | Supported | Supported |
| Slice | Constrained | Constrained | Constrained | Constrained |
| Softmax | Constrained | Constrained | Constrained | Constrained |
| Split | Constrained | Constrained | Constrained | Constrained |
| Squeeze | Constrained | Constrained | Constrained | Constrained |
| Sub | Constrained | Constrained | Constrained | Constrained |
| Tanh | Supported | Supported | Supported | Supported |
| Transpose | Constrained | Constrained | Constrained | Constrained |
| Unsqueeze | — | Constrained | Constrained | Constrained |
| Gelu | — | — | Constrained | Constrained |
| GroupNormalization | — | — | — | Constrained |
| LayerNormalization | — | — | — | Constrained |
| LSTM | — | — | — | Constrained |
| Mish | — | — | — | Supported |
| NegativeLogLikelihoodLoss | — | — | — | Constrained |

---

## Operators that fall back to CPU

Any operator not in the table above will be executed on the host CPU using ONNX Runtime, as part of the model's preamble or postamble sections. The compiler handles this automatically — you don't need to manually split the model.

Common CPU-fallback scenarios:
- Non-standard activation functions (e.g. `Erf`, `Gelu` in opset < 16)
- Dynamic shape operations
- String operations, sequence ops
- Operators with data types not supported by the AIPU (e.g. FP32 in the core model path)

---

## Checking your model

The compiler will report which operators are accelerated and which fall back to CPU in the compilation output. An unsupported operator does not prevent compilation — it just means that layer runs on the host.

If a significant portion of your model falls back to CPU, the performance gap between AIPU and CPU inference narrows. Use `--pipe=torch` and `--pipe=torch-aipu` to measure how much of your model's compute is AIPU-accelerated:

```bash
# CPU baseline
./inference.py my-model dataset --no-display --pipe=torch

# AIPU with Python pipeline
./inference.py my-model dataset --no-display --pipe=torch-aipu

# Full GStreamer + AIPU (production)
./inference.py my-model dataset --no-display
```

---

## Recommended opset

The compiler defaults to opset 17 for PyTorch-to-ONNX export:

```python
config = CompilerConfig(onnx_opset_version=17)
```

Opset 17 has the broadest operator support including `LayerNormalization`, `GroupNormalization`, and `LSTM` on-AIPU.

---

## See also

- [Compiler CLI](compiler-cli.md) — `--input-shape` and custom config for edge cases
- [Compiler Python API](compiler-api.md) — `CompilerConfig.onnx_opset_version`
- [Model Formats](../pipeline/model-formats.md) — how the compiled model handles CPU-fallback layers (preamble/postamble)

---
title: "Compiler"
---

# Compiler

Tools and APIs for compiling custom ONNX models to run on Metis hardware. Compilation converts a floating-point model to an int8 binary optimized for the AIPU.

| Page | What it covers |
|------|---------------|
| [Compiler CLI](compiler-cli.md) | The `compile` command-line tool. Covers basic usage, quantize-only mode, real-image calibration, output artifacts, and error status codes. |
| [Compiler Python API](compiler-api.md) | The `compiler.quantize()` and `compiler.compile()` Python API. Covers `CompilerConfig`, the two-step quantize → compile workflow, and usage examples. |
| [Compiler Configuration](compiler-configs.md) | Multi-core compilation modes: Batch-1 (independent cores, lower latency) vs Batch-4 (shared memory, higher throughput). Includes resource allocation for multi-model pipelines. |
| [ONNX Operator Support](onnx-support.md) | Which ONNX operators are accelerated on the AIPU (opsets 14–17), which are constrained, and what falls back to CPU. |
| [CompilerConfig Reference](compiler-config-ref.md) | Full property listing for `CompilerConfig`: all quantization, scheduling, memory, and hardware parameters with types, defaults, and enum values. |

See the [Deploy Custom Weights](../../tutorials/custom-weights.md) tutorial for the end-to-end workflow.

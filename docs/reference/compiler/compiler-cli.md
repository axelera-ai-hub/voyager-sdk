---
title: "Compiler CLI"
---
# Compiler CLI

The `axcompile` command converts an ONNX model into a compiled `.axmodel` binary ready to run on Metis hardware. It handles quantization (FP32 → int8) and hardware-specific optimization in one step.

See [Model Formats](../pipeline/model-formats.md) for an explanation of what compilation produces and where the output files land.

---
 
## Basic usage

```bash
axcompile -i /path/to/model.onnx -o /path/to/output/
```

This compiles the ONNX model using default settings and writes all artifacts to the output directory.

---

## Options

### Custom compiler configuration

Generate a default configuration file, edit it, then pass it to the compiler:

```bash
# Write default_conf.json to the output directory
axcompile --generate-config /path/to/output/

# Compile using the modified config
axcompile -i model.onnx --conf /path/to/output/default_conf.json -o /path/to/output/
```

### Quantize only

Run quantization without the full compilation step. Useful for validating quantization accuracy before committing to a full compile (which takes longer).

```bash
# Produces quantized_model_manifest.json
axcompile -i model.onnx --quantize-only -o /path/to/output/

# Later: compile from the saved quantized model
axcompile -i /path/to/output/quantized_model/quantized_model_manifest.json -o /path/to/compiled/
```

### Models with dynamic input shapes

If the ONNX model has dynamic batch or spatial dimensions, provide a fixed shape for compilation:

```bash
axcompile -i model.onnx --input-shape 1,3,224,224 -o /path/to/output/
```

Shape format: `N,C,H,W` (batch, channels, height, width).

### Real images for calibration

By default the compiler uses random data for calibration. For better quantization accuracy, provide representative images:

```bash
axcompile -i model.onnx \
  --input-shape 1,224,224,3 \
  --imageset /path/to/images/ \
  --transform /path/to/preprocess.py \
  --input-data-layout NHWC \
  --color-format BGR \
  --imreader-backend OPENCV \
  -o /path/to/output/
```

The `--transform` argument accepts a path to a Python file, optionally followed by `:function_name` to select a specific function:

```bash
--transform preprocess.py                # loads ‘get_preprocess_transform’ (default)
--transform preprocess.py:my_func        # loads ‘my_func’
```

The transform file must define a function with the signature:
 
 ```python
def my_func(image: PIL.Image.Image | np.ndarray) -> torch.Tensor:
     ...
     return tensor
 ```
 
If no `:function_name` is given, the function `get_preprocess_transform` is loaded by default.

| Option | Values | Description |
|--------|--------|-------------|
| `--color-format` | `RGB` (default), `BGR`, `GRAY` | Color format of input images |
| `--imreader-backend` | `PIL` (default), `OPENCV` | Library used to load images |
| `--input-data-layout` | `NCHW`, `NHWC` | Tensor layout |

### Reusing CLI arguments

Every compilation saves a `cli_args.json` to the output directory. Pass it to a later run to reproduce the same settings:

```bash
axcompile -i new_model.onnx --cli-args /path/to/previous/cli_args.json -o /new/output/
```

Any flags passed explicitly in the current invocation override values in `cli_args.json`.

### All options

```bash
axcompile --help
```

---

## Output artifacts

```
output/
├── cli_args.json                       # CLI arguments used (reusable)
├── conf.json                           # Compiler configuration used
├── input_model/
│   └── fp32_model.onnx                 # Copy of the input model
├── quantized_model/
│   ├── quantized_model_manifest.json   # Saved quantized model (reusable as input)
│   ├── quantized_model.json
│   ├── quantized_model.txt
│   └── report.json
├── compiled_model/
│   ├── manifest.json                   # Quantization parameters (scales, zero-points)
│   ├── model.json                      # Pipeline descriptor
│   ├── quantized_model.json
│   ├── kernel_function.c
│   ├── pool_l2_const.bin
│   └── report.json
├── compilation_report.json             # Status and error information
└── compilation_log.txt                 # Full compiler log
```

The `compiled_model/` directory is what you point to when running inference. See [Model Formats](../pipeline/model-formats.md) for how these files are used.

---

## Compilation status codes

`compilation_report.json` includes a `status` field. If compilation fails, this identifies where in the pipeline it stopped:

| Status | Description |
|--------|-------------|
| `SUCCEEDED` | Compilation completed successfully |
| `INITIALIZE_CLI_ERROR` | Error during CLI initialization |
| `ONNX_GRAPH_CLEANER_ERROR` | Error during ONNX graph cleaning |
| `QTOOLS_ERROR` | Error during quantization |
| `GRAPH_EXPORTER_ERROR` | Error during graph export to TorchScript |
| `TVM_FRONTEND_ERROR` | Error in TVM frontend |
| `TOP_LEVEL_QUANTIZE_ERROR` | Uncategorised error during `quantize()` |
| `LOWER_FRONTEND_ERROR` | Error in compiler frontend lowering |
| `FRONTEND_TO_MIDEND_ERROR` | Error in frontend-to-midend conversion |
| `LOWER_MIDEND_ERROR` | Error in midend lowering |
| `MIDEND_TO_TIR_ERROR` | Error in midend-to-TIR conversion |
| `LOWER_TIR_ERROR` | Error in TIR lowering |
| `TIR_TO_RUNTIME_ERROR` | Error in TIR-to-runtime conversion |
| `TOP_LEVEL_LOWER_ERROR` | Uncategorised error during `lower()` |

---

## See also

- [Compiler Python API](compiler-api.md) — programmatic compilation from Python
- [Model Formats](../pipeline/model-formats.md) — what the compiled artifacts are
- [First Inference](../../user-guides/first-inference.md) — run a compiled model

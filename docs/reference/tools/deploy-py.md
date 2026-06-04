---
title: "deploy.py"
---
# deploy.py

Compiles and deploys a model (or full pipeline) from a YAML definition file. Used when deploying [custom weights](../../tutorials/custom-weights.md) or new model architectures.

```bash
./deploy.py <network-name-or-yaml-path>
```

After deployment, the compiled pipeline is cached in `build/`. Subsequent calls to `inference.py` with the same network name use the cached build without recompiling.

---

## Output

Artifacts are written to `build/<network-name>/<model-name>/`:

```
build/
└── yolov8n-licenseplate/
    └── yolov8n-licenseplate/
        └── 1/
            ├── model.json         # pipeline descriptor
            ├── model.axmodel      # compiled AIPU binary
            └── manifest.json      # quantization parameters
```

See [Model Formats](../pipeline/model-formats.md) for what each file contains.

---

## Options

| Option | Description |
|--------|-------------|
| `--build-root <path>` | Write compiled output here instead of `build/` |
| `--data-root <path>` | Look for datasets here instead of `data/` |
| `--num-cal-images <N>` | Number of calibration images for quantization (default: 200; range: 100–400) |
| `--aipu-cores <N>` | Target N AIPU cores (1–4). Only valid for single-model networks. |
| `--pipe <type>` | Pipeline type: `gst` (default, AIPU), `torch` (CPU/ONNX), `torch-aipu` (Python + AIPU) |
| `--mode <mode>` | Deployment mode (see below) |
| `--model <model>` | Compile only the named model within the network; skip pipeline deployment |
| `--models-only` | Compile all models in the network; skip pipeline deployment |
| `--pipeline-only` | Deploy the pipeline only; skip model compilation (uses pre-compiled models) |
| `--export` | Produce a zip archive. For `QUANTIZE` mode: `<model>-prequantized.zip`. For other modes: `<model>.zip`. Saved to `exported/`. |

### --mode values

| Mode | Description |
|------|-------------|
| `PREQUANTIZED` | *(default)* Use a pre-quantized model if available; otherwise quantize then compile |
| `QUANTIZE` | Quantize only — do not compile. Outputs `quantized_model_manifest.json`. |
| `QUANTCOMPILE` | Quantize then compile in one step |

---

## Examples

Deploy a custom YAML:
```bash
./deploy.py customers/mymodels/yolov8n-licenseplate.yaml
```

Increase calibration images for better quantization accuracy:
```bash
./deploy.py customers/mymodels/yolov8n-licenseplate.yaml --num-cal-images 400
```

Quantize only (inspect before committing to full compile):
```bash
./deploy.py customers/mymodels/yolov8n-licenseplate.yaml --mode QUANTIZE
```

All options:
```bash
./deploy.py --help
```

---

## What happens during deployment

When you run `./deploy.py <network-name>`, the following steps execute:

### 1. Calibration data preparation

The YAML `preprocess:` section (Resize, Normalize, etc.) is applied to calibration images from the dataset (default: 200 images). These preprocessed images are used for the quantization step.

```yaml
preprocess:
  - letterbox:
      width: 640
      height: 640
  - torch-totensor:
  - normalize:
      mean: [0.485, 0.456, 0.406]
      std: [0.229, 0.224, 0.225]
```

> [!NOTE]
> YAML preprocessing is used **only during deployment** to prepare calibration data. It is not part of the runtime inference pipeline — the GStreamer operators handle runtime pre-processing independently.


### 2. Compilation

The compiler takes the ONNX model plus calibration data and produces:
- **Quantized model** — weights converted from FP32 to INT8
- **AIPU binary** (`.axmodel`) — optimized for Metis hardware
- **Preamble/postamble** — CPU-side transforms extracted by the compiler
- **Manifest** — quantization parameters for runtime dequantisation

### 3. Pipeline deployment

The compiled model is wrapped into a GStreamer pipeline descriptor (`model.json`) that the runtime can load directly.

---

## See also

- [Deploy Custom Weights](../../tutorials/custom-weights.md) — end-to-end walkthrough
- [Compiler CLI](../compiler/compiler-cli.md) — lower-level compiler without the YAML pipeline
- [Model Formats](../pipeline/model-formats.md) — output file structure

---
title: "Compiler Python API"
---
# Compiler Python API

The Python API for compiling custom models programmatically. Use this when you need to integrate compilation into a training or evaluation script.

```python
from axelera import compiler
from axelera.compiler import CompilerConfig
```

---

## Overview

Compilation is a two-step process:

1. **Quantize** — Convert an FP32 model to int8 using calibration data. Returns an `AxeleraQuantizedModel` that can run on CPU for accuracy validation.
2. **Compile** — Optimize the quantized model for Metis hardware and write deployment artifacts to disk.

```python
config = CompilerConfig()

quantized = compiler.quantize(model="model.onnx", calibration_dataset=data, config=config)
compiler.compile(model=quantized, config=config, output_dir=Path("./compiled/"))
```

---

## quantize()

```python
compiler.quantize(
    model: Union[torch.nn.Module, onnx.ModelProto, str],
    calibration_dataset: Iterator,
    config: CompilerConfig,
    transform_fn: Optional[Callable] = None,
) -> AxeleraQuantizedModel
```

Quantizes a model to int8 using calibration data.

### Parameters

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | `torch.nn.Module`, `onnx.ModelProto`, or `str` | Yes | PyTorch model, ONNX model object, or path to `.onnx` file |
| `calibration_dataset` | iterator | Yes | Yields samples used to determine quantization scales |
| `config` | `CompilerConfig` | Yes | Quantization and compilation settings |
| `transform_fn` | callable | No | Preprocessing function applied to each item from `calibration_dataset` before it reaches the model |

### Returns

`AxeleraQuantizedModel` — a quantized model that supports CPU inference for accuracy validation before hardware deployment.

### How the quantized model works

The model is split into three parts:

- **Preamble** (optional): Operations that cannot run on the AIPU, executed on CPU via ONNX Runtime before the core model
- **Core**: The main int8 model, compiled for and executed on Metis
- **Postamble** (optional): Remaining operations executed on CPU after the core model

You can call the returned model directly to validate accuracy on CPU:

```python
output = quantized_model(input_tensor)
```

Internally this runs: preamble → quantize inputs → int8 core → dequantize outputs → postamble.

---

## compile()

```python
compiler.compile(
    model: AxeleraQuantizedModel,
    config: CompilerConfig,
    output_dir: Path,
) -> None
```

Compiles a quantized model for deployment on Metis hardware.

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | `AxeleraQuantizedModel` | Output of `quantize()` |
| `config` | `CompilerConfig` | Compilation settings (cores, resource allocation, etc.) |
| `output_dir` | `Path` | Directory where compiled artifacts are written |

The compiled artifacts in `output_dir` are what you pass to `inference.py` or `create_inference_stream()`. See [Model Formats](../pipeline/model-formats.md) for the output structure.

---

## CompilerConfig

Controls both quantization and compilation behavior. All parameters are optional — defaults work for most models.

```python
from axelera.compiler import CompilerConfig

config = CompilerConfig(
    ptq_scheme="per_tensor_histogram",  # Quantization scheme
    aipu_cores=4,                        # Number of AIPU cores (1–4)
    resources=1.0,                       # Memory fraction (0.0–1.0)
    save_error_artifact=True,            # Save artifacts on failure
)
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ptq_scheme` | `"per_tensor_min_max"` | Quantization calibration scheme: `per_tensor_min_max` or `per_tensor_histogram` |
| `aipu_cores` | all available | Number of AIPU cores to target (1–4) |
| `resources` | `1.0` | Fraction of AIPU memory to use (reduce for multi-model scenarios) |
| `save_error_artifact` | `False` | Preserve intermediate files when compilation fails |

---

## Examples

### Raw images with preprocessing

Use this when calibration data is a directory of image files.

```python
import cv2
import numpy as np
import glob
from pathlib import Path
from axelera import compiler
from axelera.compiler import CompilerConfig

def calibration_images():
    for path in glob.glob("calib_images/*.jpg")[:100]:
        yield cv2.imread(path)   # uint8 BGR

def preprocess(img: np.ndarray) -> np.ndarray:
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (640, 640))
    img = img.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))   # HWC → CHW
    return np.expand_dims(img, 0)         # add batch dim

config = CompilerConfig(ptq_scheme="per_tensor_min_max")

quantized = compiler.quantize(
    model="yolo.onnx",
    calibration_dataset=calibration_images(),
    config=config,
    transform_fn=preprocess,
)

compiler.compile(quantized, config, Path("./compiled/"))
```

### Pre-processed data iterator

When your pipeline already outputs model-ready tensors, omit `transform_fn`.

```python
def preprocessed_data():
    for path in glob.glob("calib_images/*.jpg")[:100]:
        img = cv2.imread(path)
        img = cv2.resize(img, (640, 640)).astype(np.float32) / 255.0
        yield np.expand_dims(np.transpose(img, (2, 0, 1)), 0)

quantized = compiler.quantize(
    model="yolo.onnx",
    calibration_dataset=preprocessed_data(),
    config=CompilerConfig(),
    # no transform_fn needed
)
```

### PyTorch DataLoader

Integrate with an existing PyTorch training pipeline.

```python
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

dataset = datasets.ImageFolder("calibration_images/", transform=transform)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

def extract_images(batch):
    images, labels = batch
    return images.numpy()

quantized = compiler.quantize(
    model=pytorch_model,   # torch.nn.Module
    calibration_dataset=loader,
    config=CompilerConfig(),
    transform_fn=extract_images,
)
```

### Save and reload a quantized model

Separate the quantization and compilation steps — useful for validating accuracy before deploying.

```python
# Step 1: quantize and save
quantized = compiler.quantize(model="resnet50.onnx", ...)
quantized.export("resnet50_quantized/")

# Step 2: validate on CPU (separate script / later session)
from axelera.compiler.quantized_model import AxeleraQuantizedModel

quantized = AxeleraQuantizedModel.load("resnet50_quantized/")

# Step 3: compile for hardware
config = CompilerConfig(aipu_cores=4, resources=1.0)
compiler.compile(quantized, config, Path("resnet50_compiled/"))
```

---

## See also

- [Compiler CLI](compiler-cli.md) — command-line interface for the same workflow
- [Model Formats](../pipeline/model-formats.md) — what the compiled output contains
- [Custom Weights](../../tutorials/custom-weights.md) — end-to-end guide for deploying a custom model

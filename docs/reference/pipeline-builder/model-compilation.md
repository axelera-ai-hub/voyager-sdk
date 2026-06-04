---
title: "Model Compilation"
---
# Model Compilation

> [!IMPORTANT]
> **Alpha**
> Core operators (detection, classification, pose, segmentation, tracking) are stable. Cascade (`op.foreach`, `op.croproi`) and streaming APIs are still in development.


The Voyager SDK compiler quantizes your model to mixed-precision and compiles it for the Metis
AIPU, producing an `.axm` file. This page covers two paths: through a supported third-party integration (e.g.
Ultralytics, where compilation is handled for you) or directly via the compiler
API for any ONNX or PyTorch model.

## Ultralytics Integration

If your model is trained with [Ultralytics](https://docs.ultralytics.com/), a single
call handles quantization and compilation:

```python
from ultralytics import YOLO

model = YOLO("yolo11n.pt")
model.export(format="axelera")
# Output: yolo11n_axelera_model/yolo11n.axm
```

The output directory `yolo11n_axelera_model/` contains the `.axm` file ready for
`op.load()`.

To validate accuracy on the AIPU vs the original model, use `yolo val`:

```bash
yolo val model=yolo11n_axelera_model format=axelera
```

For full details, see the
[Ultralytics Axelera integration guide](https://docs.ultralytics.com/integrations/axelera/).

### What Happens Under the Hood

This shows what the Ultralytics integration does internally — useful if you're
curious how the exporter works, or if you want to build a similar integration
for another framework.

The Ultralytics exporter calls the same compiler API you can use directly:

```python
from axelera import compiler
from axelera.compiler import CompilerConfig
from axelera.compiler.config.model_specific import extract_ultralytics_metadata

# 1. Extract Ultralytics-specific metadata (task type, class names, keypoint shape, etc.)
#    Generic models: skip this step and omit model_metadata from CompilerConfig.
metadata = extract_ultralytics_metadata(model)

# 2. Configure the compiler
config = CompilerConfig(
    model_metadata=metadata,
    aipu_cores_used=1,
    output_axm_format=True,
    model_name=model_name,
)

# 3. Quantize and compile
qmodel = compiler.quantize(
    model="yolo11n.onnx",
    calibration_dataset=calibration_images(),
    config=config,
    transform_fn=transform_fn,
)
compiler.compile(model=qmodel, config=config, output_dir=export_path)
```

The exporter does two things your own code needs to handle manually:

1. **`model_metadata`** -- Ultralytics-specific. The exporter extracts task type,
   class names, and keypoint shape and embeds them in the `.axm`; at runtime,
   `op.load()` reads this to auto-select optimized C++ postprocessing.
   **For generic models, omit `model_metadata`** and wire up your own postprocessing
   pipeline instead.

2. **Auto-tuned CompilerConfig** -- the exporter picks the right `resources_used`,
   `quantization_scheme`, `tiling_depth`, etc. for each model architecture. When you
   compile your own model, you set these yourself (sensible defaults work for most
   models).

## Generic Path: From ONNX or PyTorch

For models not trained with Ultralytics, use the compiler API directly.

> **Required settings for the pipeline builder:**
> `output_axm_format=True` must be set so the compiler produces an `.axm` file
> that `op.load()` can consume. The pipeline builder supports
> `aipu_cores_used=1` only.

### From an ONNX Model

```python
from pathlib import Path
from axelera import compiler
from axelera.compiler import CompilerConfig

config = CompilerConfig(
    model_name="my_detector",
    aipu_cores_used=1,
    resources_used=0.25,        # fraction of device memory to use
    output_axm_format=True,     # required for op.load()
)

# Provide a calibration dataset: an iterator yielding numpy arrays
# matching the model's input shape and dtype (typically float32 NCHW)
def calibration_data():
    for path in Path("calibration_images/").glob("*.jpg"):
        img = preprocess(path)  # your preprocessing: resize, normalize, etc.
        yield img

qmodel = compiler.quantize(
    model="model.onnx",
    calibration_dataset=calibration_data(),
    config=config,
)

compiler.compile(
    model=qmodel,
    config=config,
    output_dir=Path("compiled_output/"),
)
```

For model-specific tuning (e.g., `quantization_scheme`, `tiling_depth`), see the
[Full Compiler Config Reference](../compiler/compiler-config-ref.md).

### From a PyTorch Model with DataLoader

```python
from pathlib import Path
from torch.utils.data import DataLoader
from axelera import compiler
from axelera.compiler import CompilerConfig

config = CompilerConfig(
    model_name="my_classifier",
    aipu_cores_used=1,
    resources_used=0.25,
    output_axm_format=True,     # required for op.load()
)
loader = DataLoader(my_dataset, batch_size=1)

def extract_images(batch):
    images, labels = batch
    return images

qmodel = compiler.quantize(
    model=torch_model,
    calibration_dataset=loader,
    config=config,
    transform_fn=extract_images,
)

compiler.compile(
    model=qmodel,
    config=config,
    output_dir=Path("compiled_output/"),
)
```

## Finding Your .axm

After compilation, the `.axm` file is named after `model_name` in your
`CompilerConfig`. The `output_dir` you pass to `compiler.compile()` receives
intermediate build artifacts (manifests, quantized graphs, etc.), while the
`.axm` is placed relative to it -- typically in the current working directory.

For example, with `model_name="my_detector"`:

```
$ ls *.axm
my_detector.axm
```

The Ultralytics exporter automatically organizes output into
`<model_name>_axelera_model/`. When using the API directly, you handle
file placement yourself.

Once you have the `.axm`, see the [Pipeline Overview](pipeline-overview.md)
for how to build inference pipelines around it.

## Validate Before Deploying

After quantization, you can run the quantized model on CPU to check accuracy
before compiling for hardware:

```python
import numpy as np

# Run a sample through the quantized model on CPU
sample = np.random.randn(1, 3, 640, 640).astype(np.float32)
output = qmodel(sample)
# Compare output against the original model to verify quantization quality
```

## Next Steps

- [Pipeline Overview](pipeline-overview.md) — Build pipelines around your `.axm`
  with examples for detection, classification, pose, segmentation, and tracking
- [Compiler API Reference](../compiler/compiler-api.md) — Full API details for
  `compiler.quantize()` and `compiler.compile()`
- [Compiler Configuration Reference](../compiler/compiler-configs.md) —
  All `CompilerConfig` options

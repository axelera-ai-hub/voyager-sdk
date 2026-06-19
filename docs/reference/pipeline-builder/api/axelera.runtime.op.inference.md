---
title: "axelera.runtime.op.inference"
---
# `axelera.runtime.op.inference`


Model loading and inference: AxRuntimeModel, OnnxModel, load().

## Summary

| Name | Description |
|------|-------------|
| [load](#load) | Load a compiled Axelera model (`.axm`) or pipeline package (`.axe`). |
| [onnx_model](#onnx_model) | Run standalone ONNX model inference. |
| [OnnxModel](#onnxmodel) | ONNX model inference operator. |

---

### load

```python
load(file: str, *, name: str = '', core_allocation: int | str | None = None) -> Operator
```

Load a compiled Axelera model (`.axm`) or pipeline package (`.axe`).

Takes preprocessed np.ndarray input(s) and returns model output(s).

**Args:**

- **file**: path to .axm or .axe file.
- **name**: Optional name for the operator in pipeline.
- **core_allocation**: How many AIPU cores to give this model. `None` (default) shares cores equally with the other models in the pipeline; an int assigns that many cores absolutely (e.g. `2`); a percentage string (e.g. `'50%'`) takes that share of the cores left after absolute allocations, so it scales with the core count of the target hardware.

**Examples:**

```python
# Detection model (.axm)
op.seq(
    op.letterbox(640, 640),
    op.totensor(),
    op.normalize(mean=[0.0, 0.0, 0.0], std=[1.0, 1.0, 1.0]),
    op.load('yolov8n-coco.axm'),  # Returns raw model output
    op.decode_detections(...),
)

# Classification model (.axm)
op.seq(
    op.resize(size=256, half_pixel_centers=True),
    op.center_crop(224),
    op.totensor(),
    op.normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    op.load('resnet50-imagenet'),  # Returns raw logits
    op.top_k(k=5),
    op.ax_classification(...),
)

# Load .axe file (complete pipeline)
detector = op.load('yolov8n-coco.axe')
detections = detector(image)
```

**Note:**

`op.load()` automatically handles quantization, padding, model execution, depadding,
and dequantization. Preamble and postamble ONNX graphs are also applied if present
in the model configuration.

---

### onnx_model

```python
onnx_model(path: str | Path, *, provider: str | None = None, name: str = '') -> OnnxModel
```

Run standalone ONNX model inference.

Creates an operator that runs ONNX model inference using onnxruntime.
By default, automatically selects the best available execution provider
(CUDA > MPS > OpenVINO > CPU). Use the `provider` parameter to force
a specific provider.

Input (when called):
    np.ndarray - model input(s) as positional args

Output (when called):
    np.ndarray for single-output models, unpack args for multi-output

**Args:**

- **path**: Path to .onnx file
- **provider**: Execution provider ('cuda', 'cpu', 'openvino', 'mps', or None for auto)
- **name**: Optional name for the operator (used in serialization and debugging)

**Returns:** `OnnxModel` -- OnnxModel operator instance

**Raises:**

- **FileNotFoundError**: If the ONNX file does not exist
- **ImportError**: If onnxruntime is not installed
- **ValueError**: If the specified provider is not available

**Examples:**

```python
# Auto-select best available provider
pipeline = op.seq(
    op.letterbox(640, 640),
    op.totensor(),
    op.onnx_model('yolov8n.onnx'),
    op.decode_detections(algo='yolov8', num_classes=80),
)

# Force CPU execution
op.onnx_model('model.onnx', provider='cpu')

# Force CUDA GPU execution
op.onnx_model('model.onnx', provider='cuda')
```

**Note:**

Available providers depend on your onnxruntime installation:
- `'cuda'`: Requires onnxruntime-gpu with CUDA support
- `'mps'`: Available on macOS with Apple Silicon
- `'openvino'`: Requires onnxruntime-openvino
- `'cpu'`: Always available

Use `_internal.get_available_onnx_providers()` to check available providers.

---

### OnnxModel

**Alias:** `onnx_model`

ONNX model inference operator.

Runs ONNX model inference using onnxruntime with automatic or explicit
execution provider selection.

**Args:**

- **file**: Path to .onnx file or filename within .axm archive.
- **axm_path**: Optional path to .axm archive containing the ONNX model.
- **provider**: Execution provider ('cuda', 'cpu', 'openvino', 'mps', or None for auto).

**Constructor:**

```python
__init__(axm_path: str | Path = str(axm_path), provider: str | None = provider, file=str(file))
```

**Methods:**

#### serialize

```python
serialize() -> dict[str, Any]
```

Serialize OnnxModel for AXE format.

Returns dict with type and file path. The ONNX file will be embedded
in the AXE archive using just the filename (not full path).

#### classmethod deserialize_impl

```python
deserialize_impl(operator_cls, spec: dict[str, Any], zf: zipfile.ZipFile, base_path: Path) -> OnnxModel
```

Deserialize OnnxModel from AXE format.

Extracts the ONNX file from the archive and creates an OnnxModel.

---
title: "axelera.runtime.op.inference"
---
# `axelera.runtime.op.inference`


Model loading and inference: AxRuntimeModel, OnnxModel, Pt2Model, load().

## Summary

| Name | Description |
|------|-------------|
| [load](#load) | Load a compiled Axelera model (`.axm`) or pipeline package (`.axe`). |
| [onnx_model](#onnx_model) | Run standalone ONNX model inference. |
| [OnnxModel](#onnxmodel) | ONNX model inference operator. |
| [pt2_model](#pt2_model) | Run a standalone `.pt2` (torch.export) model. |
| [Pt2Model](#pt2model) | torch.export ExportedProgram inference operator. |

---

### load

```python
load(file: str, *, name: str = '', core_allocation: int | str | None = None, using_postamble: Postamble | None = None, device_config: DeviceConfig | None = None) -> Operator
```

Load a compiled Axelera model (`.axm`) or pipeline package (`.axe`).

Takes preprocessed np.ndarray input(s) and returns model output(s).

**Args:**

- **file**: path to .axm or .axe file.
- **name**: Optional name for the operator in pipeline.
- **core_allocation**: How many AIPU cores to give this model. `None` (default) shares cores equally with the other models in the pipeline; an int assigns that many cores absolutely (e.g. `2`); a percentage string (e.g. `'50%'`) takes that share of the cores left after absolute allocations, so it scales with the core count of the target hardware.
- **using_postamble**: a `Postamble` naming a kernel to run the model's post-processing in place of the default onnxruntime tail. See `Postamble`. `None` (default) keeps the onnx tail.
- **device_config**: board settings the cores running this model must hold, e.g. `DeviceConfig({'110m': BoardConfig(mvm_limitation=57)})`. `None` (default) leaves the device alone. See `axelera.runtime.device_config`.

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
    op.load('resnet50-imagenet.axm'),  # Returns raw logits
    op.softmax(),
    op.ax_classification(class_id_type=op.ImagenetClasses),
    op.top_k(k=5),
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

`onnxruntime.get_available_providers()` lists what your installation has,
under onnxruntime's own names (e.g. `'CUDAExecutionProvider'`).

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
__init__(file: str | Path = '', axm_path: str | Path = '', provider: str | None = None, *, name: str = '', save: str = '')
```

**Methods:**

#### model_input_spec

```python
model_input_spec() -> tuple[list[int], str]
```

Return the declared input shape and its layout, empty if it is dynamic.

The shape comes from the ONNX graph, which is NCHW. A dimension the graph
left symbolic is not a shape a package can state.

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

---

### pt2_model

```python
pt2_model(path: str | Path, *, name: str = '') -> Pt2Model
```

Run a standalone `.pt2` (torch.export) model.

Mirrors `onnx_model` but for `torch.export.ExportedProgram` files.

---

### Pt2Model

**Alias:** `pt2_model`

torch.export ExportedProgram inference operator.

Loads a `.pt2` file or a member of an `.axm` archive and runs
the exported program as a sequence operator.

**Args:**

- **file**: Path to .pt2 file or member name within an .axm archive.
- **axm_path**: Optional path to .axm archive containing the .pt2 member.

**Constructor:**

```python
__init__(file: str | Path = '', axm_path: str | Path = '', *, name: str = '', save: str = '')
```

**Properties:**

- **`input_shape`** (`tuple[int, ...] | None`) — Shape of the exported program's first input tensor, or `None`.

**Methods:**

#### model_input_spec

```python
model_input_spec() -> tuple[list[int], str]
```

Return the exported program's input shape and its layout, NCHW.

---
title: "axelera.runtime.op.axe"
---
# `axelera.runtime.op.axe`


AXE format support for model pipeline packaging.

This module provides functionality for saving and loading .axe files,
which are ZIP archives containing complete inference pipelines
(preprocessing + model + postprocessing) in a portable format.

An .axe holds the pipeline as it was built. Fusion picks its kernels for the
host it runs on, so `save_axe` refuses a pipeline that `.optimized()` rewrote,
and whoever loads the package optimizes it for their own machine.

Example:
    pipeline = op.seq(
        op.letterbox(640, 640),
        op.totensor(),
        op.load('model.axm'),
        op.decode_detections(algo='yolov8', name='decode'),
        op.nms(name='nms'),
        op.to_image_space(),
        op.ax_detection(class_id_type=op.CocoClasses),
    )
    pipeline.save_axe('detector.axe', name='YOLOv8n COCO', task='detection')

    # Load, optimize for this host, and use
    detector = op.load('detector.axe').optimized()
    detector.node(name='nms').max_boxes = 200  # Retune a named operator
    detections = detector(image)

## Summary

| Name | Description |
|------|-------------|
| [AxeOperator](#axeoperator) | Pipeline loaded from .axe file. |
| [IncompatibleAxeError](#incompatibleaxeerror) | Raised when .axe file has incompatible schema version. |
| [load_axe](#load_axe) | Load an .axe pipeline package. |

---

### AxeOperator

**Alias:** `axe_operator`

Pipeline loaded from .axe file.

`op.load('pipeline.axe')` and `load_axe` return one; you do not construct it
directly.

**Attributes:**

- **op**: Reconstructed root Operator
- **name** (`str`): Pipeline name (from manifest or load param)
- **file**: Source .axe file path
- **manifest** (`dict[str, Any]`): Original pipeline.toml content

**Constructor:**

```python
__init__(op: Operator, manifest: dict[str, Any], name: str = '', file: str = '')
```

**Methods:**

#### optimized

```python
optimized(policy: str | None = 'inherit') -> AxeOperator
```

Return the package's pipeline with the registered fusions applied.

An .axe stores its pipeline unoptimized, so the process that loads it
optimizes it for its own host with this call.

#### info

```python
info() -> dict[str, Any]
```

Get pipeline metadata without running inference.

**Returns:** `dict[str, Any]` -- Metadata dict with name, description, input_shape, etc.

---

### IncompatibleAxeError

Raised when .axe file has incompatible schema version.

**Attributes:**

- **file_version**: Version in the .axe file
- **supported_version**: Maximum version this runtime supports
- **message**: Human-readable error message

**Constructor:**

```python
__init__(file_version: str, supported_version: str)
```

---

### load_axe

```python
load_axe(path: str, *, name: str = '') -> AxeOperator
```

Load an .axe pipeline package.

**Args:**

- **path**: Path to .axe file
- **name**: Optional name override

**Returns:** `AxeOperator` -- AxeOperator instance

**Raises:**

- **FileNotFoundError**: If file doesn't exist
- **IncompatibleAxeError**: If schema version incompatible

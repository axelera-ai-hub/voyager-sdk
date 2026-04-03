# Architecture Reference

Reference with: @docs/architecture.md

---

## Layer map

```
Agent (Claude / LangChain / any)
        │  tool_use / MCP protocol
        ▼
┌───────────────────────────────┐
│       MCP Server              │  mcp/server.py
│  detect_objects               │  fastmcp, stdio transport
│  classify_image               │
│  segment_image                │
│  list_models                  │
└──────────────┬────────────────┘
               │ Python import
               ▼
┌───────────────────────────────┐
│       REST API                │  api/main.py
│  POST /v1/detect              │  FastAPI + uvicorn
│  POST /v1/classify            │  Pydantic validation
│  POST /v1/segment             │
│  GET  /v1/models              │
│  GET  /v1/health              │
└──────────────┬────────────────┘
               │ Python import
               ▼
┌───────────────────────────────┐
│     Python Wrapper            │  axelera/inference.py
│  detect(uri, model, thresh)   │  Typed returns
│  classify(uri, model)         │  Error translation
│  segment(uri, model, thresh)  │
└──────────────┬────────────────┘
               │ SDK call
               ▼
┌───────────────────────────────┐
│     Voyager SDK               │  Axelera proprietary
│     Metis AIPU hardware       │  PCIe or M.2
└───────────────────────────────┘
```

---

## Canonical response schema

All wrapper functions and API routes return this exact shape:

```python
# Detection response
{
    "ok": True,
    "model": "yolo11s-coco-onnx",
    "timing_ms": 13.5,
    "detections": [
        {"label": "person", "score": 0.97, "bbox": [x, y, w, h]},
        {"label": "car",    "score": 0.84, "bbox": [x, y, w, h]}
    ],
    "error": None
}

# Error response (any layer)
{
    "ok": False,
    "model": "yolo11s-coco-onnx",
    "timing_ms": 0.0,
    "detections": [],
    "error": "ModelNotFoundError: model 'yolo11x' is not installed"
}
```

---

## API request schema

```python
# POST /v1/detect  (and /classify, /segment)
{
    "input_uri": "file:///tmp/image.jpg",   # required, must be file:// URI
    "model": "yolo11s-coco-onnx",           # optional, has default
    "threshold": 0.4                        # optional, float 0.0–1.0
}
```

---

## Error HTTP mapping

| Exception class    | HTTP status | When |
|--------------------|-------------|------|
| ValidationError    | 422         | Bad input_uri, threshold out of range |
| ModelNotFoundError | 404         | Model name not in zoo |
| DeviceNotFoundError| 503         | Metis hardware not detected |
| InferenceError     | 500         | Runtime failure during inference |

---

## MCP tool schemas (authoritative)

### detect_objects
```json
{
  "name": "detect_objects",
  "description": "Run object detection on an image using Axelera Metis AIPU. Returns labelled bounding boxes with confidence scores.",
  "input_schema": {
    "type": "object",
    "properties": {
      "input_uri": {"type": "string", "description": "file:// URI to the image"},
      "model":     {"type": "string", "default": "yolo11s-coco-onnx"},
      "threshold": {"type": "number", "default": 0.4, "minimum": 0.0, "maximum": 1.0}
    },
    "required": ["input_uri"]
  }
}
```

### classify_image
```json
{
  "name": "classify_image",
  "description": "Classify an image into ImageNet categories using Axelera Metis AIPU.",
  "input_schema": {
    "type": "object",
    "properties": {
      "input_uri": {"type": "string", "description": "file:// URI to the image"},
      "model":     {"type": "string", "default": "resnet50-imagenet"}
    },
    "required": ["input_uri"]
  }
}
```

### segment_image
```json
{
  "name": "segment_image",
  "description": "Run instance segmentation on an image using Axelera Metis AIPU.",
  "input_schema": {
    "type": "object",
    "properties": {
      "input_uri": {"type": "string", "description": "file:// URI to the image"},
      "model":     {"type": "string", "default": "yolov8n-seg-coco-onnx"},
      "threshold": {"type": "number", "default": 0.4, "minimum": 0.0, "maximum": 1.0}
    },
    "required": ["input_uri"]
  }
}
```

### list_models
```json
{
  "name": "list_models",
  "description": "List all available models in the Axelera model zoo, optionally filtered by task.",
  "input_schema": {
    "type": "object",
    "properties": {
      "task": {
        "type": "string",
        "enum": ["detection", "classification", "segmentation"],
        "description": "Optional filter by task type"
      }
    }
  }
}
```

---

## What is explicitly out of scope (v1)

- Multi-device orchestration
- Model registry / model upload
- OpenTelemetry / distributed tracing
- OAuth / API key authentication
- Rate limiting
- Async/streaming inference
- LLM inference (Metis SLM support is separate roadmap item)
- Windows support

These may be added post-Gate-6 based on real usage feedback.

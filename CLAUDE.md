# Axelera Tool Layer — Claude Code Context

## What this project is
A production-grade MCP server and Python SDK that wraps the Axelera Voyager SDK,
exposing Metis AIPU computer vision inference as callable tools for AI agents.
The primary consumer is Claude (via MCP tool_use) but the REST API is framework-agnostic.

## Non-negotiable architecture rules
- Nothing gets built on top of something that hasn't been proven to work below it
- Every phase ends with a runnable deliverable, not a diagram
- No abstraction without a passing test beneath it
- Error contracts are defined BEFORE the happy path is extended

## Repo layout
```
axelera-tool-layer/
├── CLAUDE.md                  ← you are here
├── axelera/
│   ├── __init__.py
│   ├── inference.py           ← thin Voyager SDK wrapper (Phase 1)
│   └── exceptions.py          ← typed error hierarchy
├── api/
│   └── main.py                ← FastAPI REST layer (Phase 2)
├── mcp/
│   └── server.py              ← MCP server (Phase 3)
├── examples/
│   └── agent_demo.py          ← end-to-end agent workflow (Phase 4)
├── tests/
│   ├── test_inference.py
│   ├── test_api.py
│   └── test_mcp.py
├── docs/
│   ├── architecture.md        ← @docs/architecture.md
│   ├── phases.md              ← @docs/phases.md
│   ├── model_zoo.md           ← @docs/model_zoo.md
│   └── decisions.md          ← @docs/decisions.md
├── Dockerfile
├── requirements.txt
└── README.md
```

## Tech stack
- Python 3.11+
- Voyager SDK (installed at /opt/axelera or via pip axelera-voyager)
- FastAPI + uvicorn for REST
- fastmcp for MCP server
- anthropic Python SDK for agent demo
- pytest for tests

## Commands Claude must know
```bash
# Validate SDK works (run this FIRST before anything else)
python tests/test_inference.py

# Run REST API locally
uvicorn api.main:app --reload --port 8000

# Run MCP server
python mcp/server.py

# Run all tests
pytest tests/ -v

# Run agent demo
python examples/agent_demo.py --image path/to/image.jpg

# Build Docker image
docker build -t axelera-tool-layer:latest .

# Health check
curl http://localhost:8000/v1/health
```

## Response schema contract (never deviate from this)
Every inference function returns:
```python
{
    "ok": bool,
    "model": str,
    "timing_ms": float,
    "results": list,   # detections / classes / segments
    "error": str | None
}
```

## Error hierarchy (axelera/exceptions.py)
```
AxeleraError (base)
├── ModelNotFoundError
├── DeviceNotFoundError
├── InferenceError
└── ValidationError
```

## Phase gates — do NOT proceed past a gate without passing it
- Gate 1: `python tests/test_inference.py` prints real detections with timing_ms
- Gate 2: `curl localhost:8000/v1/health` returns {"status": "ok", "device": ...}
- Gate 3: MCP Inspector can call detect_objects and return structured JSON
- Gate 4: `python examples/agent_demo.py --image test.jpg` prints agent natural language summary
- Gate 5: `docker run axelera-tool-layer:latest` starts without errors

## What Claude should NEVER do in this project
- Add a new abstraction layer before the layer below it has a passing test
- Return base64-encoded images in any API response (use file URIs)
- Swallow exceptions silently — always re-raise as typed AxeleraError
- Add OpenTelemetry, OAuth, rate limiting, or multi-device support before Gate 5
- Write a mock that hides a real Voyager SDK failure

## Voyager SDK call pattern (verified baseline)
```python
# This is the ground truth. All wrappers must produce identical results.
from axelera import sdk as voyager
pipeline = voyager.load_pipeline("yolo11s-coco-onnx")
result = pipeline.infer("file:///path/to/image.jpg")
# result.detections → list of Detection(label, score, bbox)
# result.timing_ms  → float
```

## MCP tool schemas — canonical definitions
```python
# detect_objects
input:  {input_uri: str, model: str = "yolo11s-coco-onnx", threshold: float = 0.4}
output: {ok: bool, model: str, timing_ms: float, detections: list, error: str|None}

# classify_image
input:  {input_uri: str, model: str = "resnet50-imagenet"}
output: {ok: bool, model: str, timing_ms: float, classes: list, error: str|None}

# segment_image
input:  {input_uri: str, model: str = "yolov8n-seg-coco-onnx", threshold: float = 0.4}
output: {ok: bool, model: str, timing_ms: float, segments: list, error: str|None}

# list_models
input:  {task: str | None}   # "detection" | "classification" | "segmentation" | None
output: {models: list[{name: str, task: str, format: str}]}
```

## Current status
See @docs/phases.md for live phase progress and decisions log.

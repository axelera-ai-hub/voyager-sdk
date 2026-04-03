# API Curl Test Suite — Gate 3

Run all tests: `bash docs/curl_tests.md` won't work — copy-paste each block.
Or run: `pytest tests/test_api.py -v` which automates all of these.

---

## Health check
```bash
curl -s http://localhost:8000/v1/health | python3 -m json.tool
# Expected: {"status": "ok", "device": "Metis AIPU", "sdk_version": "..."}
```

---

## Detect — happy path
```bash
curl -s -X POST http://localhost:8000/v1/detect \
  -H "Content-Type: application/json" \
  -d '{"input_uri": "file:///tmp/test.jpg", "model": "yolo11s-coco-onnx", "threshold": 0.4}' \
  | python3 -m json.tool
# Expected: {"ok": true, "model": "yolo11s-coco-onnx", "timing_ms": ..., "detections": [...]}
```

---

## Detect — bad image path (422)
```bash
curl -s -X POST http://localhost:8000/v1/detect \
  -H "Content-Type: application/json" \
  -d '{"input_uri": "file:///nonexistent/image.jpg"}' \
  | python3 -m json.tool
# Expected: HTTP 422, {"ok": false, "error": "ValidationError: ..."}
```

---

## Detect — unknown model (404)
```bash
curl -s -X POST http://localhost:8000/v1/detect \
  -H "Content-Type: application/json" \
  -d '{"input_uri": "file:///tmp/test.jpg", "model": "doesnotexist-v99"}' \
  | python3 -m json.tool
# Expected: HTTP 404, {"ok": false, "error": "ModelNotFoundError: ..."}
```

---

## List models
```bash
curl -s http://localhost:8000/v1/models | python3 -m json.tool
# Expected: {"models": [{name, task, format}, ...]}

# With filter
curl -s "http://localhost:8000/v1/models?task=detection" | python3 -m json.tool
```

---

## Classify — happy path
```bash
curl -s -X POST http://localhost:8000/v1/classify \
  -H "Content-Type: application/json" \
  -d '{"input_uri": "file:///tmp/test.jpg"}' \
  | python3 -m json.tool
# Expected: {"ok": true, "classes": [{"label": "...", "score": ...}]}
```

---

## Segment — happy path
```bash
curl -s -X POST http://localhost:8000/v1/segment \
  -H "Content-Type: application/json" \
  -d '{"input_uri": "file:///tmp/test.jpg"}' \
  | python3 -m json.tool
# Expected: {"ok": true, "segments": [...]}
```

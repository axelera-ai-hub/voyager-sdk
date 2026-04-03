# Phase Progress & Task Tracker

Claude Code: read this at the start of every session and update checkboxes as tasks complete.
Reference with: @docs/phases.md

---

## Phase 1 — Voyager SDK validation (Hr 0–2)
Goal: prove the hardware works before writing a single line of wrapper code.

- [ ] Run Voyager SDK sample script against a real image
- [ ] Confirm timing_ms is populated (proves hardware is accelerating, not CPU fallback)
- [ ] Document exact model name format (e.g. `yolo11s-coco-onnx` not `yolo11s`)
- [ ] Document input types accepted (file:// URI vs raw path vs bytes)
- [ ] Document full output shape (Detection fields, score range, bbox format)
- [ ] Log at least 3 known SDK edge cases (missing model, wrong path, device not found)
- [ ] Write `tests/test_sdk_baseline.py` — raw SDK call, no wrapper, prints result
- [ ] **GATE 1 PASS:** test_sdk_baseline.py prints detections with timing_ms > 0

---

## Phase 2 — Python wrapper (Hr 2–6)
Goal: thin, typed, testable wrapper that hides Voyager internals.

- [ ] Create `axelera/exceptions.py` with full error hierarchy
- [ ] Create `axelera/inference.py` with `detect()` function
- [ ] Create `axelera/inference.py` with `classify()` function
- [ ] Create `axelera/inference.py` with `segment()` function
- [ ] All three functions return canonical response schema
- [ ] All SDK exceptions caught and re-raised as typed AxeleraError
- [ ] Write `tests/test_inference.py` — one happy path test per function
- [ ] Write `tests/test_inference.py` — one error path test (bad image path)
- [ ] `pytest tests/test_inference.py -v` passes all 8 tests
- [ ] **GATE 2 PASS:** wrapper callable from plain Python, no FastAPI needed

---

## Phase 3 — FastAPI REST layer (Hr 6–11)
Goal: HTTP interface with real error contracts, testable with curl.

- [ ] Create `api/main.py` with `POST /v1/detect`
- [ ] Create `api/main.py` with `POST /v1/classify`
- [ ] Create `api/main.py` with `POST /v1/segment`
- [ ] Create `api/main.py` with `GET /v1/models`
- [ ] Create `api/main.py` with `GET /v1/health`
- [ ] Define Pydantic request model (InferenceRequest)
- [ ] Define Pydantic response model (InferenceResponse)
- [ ] Define error response schema for 400, 422, 500
- [ ] Test `POST /v1/detect` with valid image → 200 + detections
- [ ] Test `POST /v1/detect` with bad path → 422 + structured error
- [ ] Test `POST /v1/detect` with unknown model → 404 + ModelNotFoundError
- [ ] Test `GET /v1/health` → 200 + device status
- [ ] **GATE 3 PASS:** all 4 curl tests in docs/curl_tests.md pass

---

## Phase 4 — MCP server (Hr 11–16)
Goal: Claude and any MCP-compatible agent can call inference tools.

- [ ] Install fastmcp (or mcp-python), verify version in requirements.txt
- [ ] Create `mcp/server.py` with `detect_objects` tool + full JSON schema
- [ ] Verify `detect_objects` schema matches canonical definition in CLAUDE.md
- [ ] Test detect_objects via MCP Inspector CLI
- [ ] Add `classify_image` tool
- [ ] Add `segment_image` tool
- [ ] Add `list_models` tool
- [ ] All 4 tools discoverable and callable from MCP Inspector
- [ ] Error responses return structured JSON (not plain text exceptions)
- [ ] **GATE 4 PASS:** MCP Inspector calls all 4 tools successfully

---

## Phase 5 — Agent workflow (Hr 16–21)
Goal: Claude receives an image, calls MCP tools, returns natural language.

- [ ] Create `examples/agent_demo.py` with CLI arg `--image`
- [ ] Wire Anthropic SDK with detect_objects as available tool
- [ ] Agent successfully calls detect_objects via tool_use
- [ ] Agent interprets detections and returns natural language summary
- [ ] Test on easy image (person in frame)
- [ ] Test on hard image (crowded scene, 10+ objects)
- [ ] Test on edge case (empty/dark/corrupted image)
- [ ] Add single retry if tool call fails with InferenceError
- [ ] Log all tool calls and results to `logs/agent_run_{timestamp}.json`
- [ ] **GATE 5 PASS:** `python examples/agent_demo.py --image test.jpg` works end-to-end

---

## Phase 6 — Ship (Hr 21–24)
Goal: containerised, documented, demo-ready.

- [ ] Write `Dockerfile` (multi-stage: build layer caches SDK, runtime layer minimal)
- [ ] `docker build` succeeds without errors
- [ ] `docker run` starts API on port 8000
- [ ] `docker run` health check passes
- [ ] Pin all dependencies in `requirements.txt` with exact versions
- [ ] Write `README.md` — prerequisites, 3-step quickstart, curl example, agent example
- [ ] Record 90-second terminal demo (API call + agent output)
- [ ] Push to GitHub with clean commit history (one commit per phase)
- [ ] **GATE 6 PASS:** someone else can clone + docker run and get a working API

---

## Decisions log
| Date | Decision | Reason |
|------|----------|--------|
| — | fastmcp over raw mcp-python | Higher-level API, less boilerplate |
| — | file:// URIs not base64 | Avoids payload size issues, matches Voyager SDK native |
| — | uvicorn not gunicorn | Single-process sufficient for prototype, simpler config |

Update this table when architectural decisions are made during development.

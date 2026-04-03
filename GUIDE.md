# Axelera Tool Layer — Claude Code Build Guide
## From zero to production in 24 hours

---

## Prerequisites (before opening Claude Code)

1. Metis hardware connected and detected (`lspci | grep -i axelera`)
2. Voyager SDK installed (`pip show axelera-voyager` or check `/opt/axelera`)
3. Claude Code installed (`npm install -g @anthropic-ai/claude-code`)
4. Python 3.11+ (`python3 --version`)
5. One test image on disk (e.g. `/tmp/test.jpg`)

---

## Project setup

```bash
# Clone or create the project
mkdir axelera-tool-layer && cd axelera-tool-layer

# Copy the CLAUDE.md and docs/ folder from this package into your project root
# The file structure should look like:
# axelera-tool-layer/
# ├── CLAUDE.md
# └── docs/
#     ├── phases.md
#     ├── architecture.md
#     ├── model_zoo.md
#     └── curl_tests.md

# Launch Claude Code
claude
```

---

## How to run each phase with Claude Code

### Phase 1 — SDK validation (say this exactly)

```
Read @docs/phases.md. We are starting Phase 1.

My Voyager SDK is installed. Before writing any wrapper code, write a 
test script at tests/test_sdk_baseline.py that calls the Voyager SDK 
directly on file:///tmp/test.jpg using yolo11s-coco-onnx. Print the 
full result including timing_ms. Run it and show me the output.

Do not proceed past this until the output shows real detections with 
timing_ms greater than zero.
```

---

### Phase 2 — Python wrapper (say this)

```
Gate 1 is passed. Read @docs/phases.md and @docs/architecture.md.

Build Phase 2:
1. Create axelera/exceptions.py with the full error hierarchy from CLAUDE.md
2. Create axelera/inference.py with detect(), classify(), segment() 
   following the canonical response schema in @docs/architecture.md
3. Write tests/test_inference.py covering happy path and error path for each function
4. Run pytest tests/test_inference.py -v and show me the output

All SDK exceptions must be caught and re-raised as typed AxeleraErrors.
Check off completed tasks in @docs/phases.md as you go.
```

---

### Phase 3 — REST API (say this)

```
Gate 2 is passed. Read @docs/phases.md and @docs/architecture.md.

Build Phase 3:
1. Create api/main.py with all 5 routes using FastAPI + Pydantic
2. Define the full error HTTP mapping from @docs/architecture.md
3. Run uvicorn and then run all curl tests from @docs/curl_tests.md
4. All 7 curl tests must pass before marking Gate 3 done

Do not add auth, rate limiting, or Docker yet.
```

---

### Phase 4 — MCP server (say this)

```
Gate 3 is passed. Read @docs/phases.md and @docs/architecture.md.

Build Phase 4:
1. Install fastmcp and add to requirements.txt with pinned version
2. Create mcp/server.py exposing all 4 tools with schemas exactly as defined 
   in @docs/architecture.md
3. Start the MCP server and verify all 4 tools are discoverable via MCP Inspector
4. Each tool must return structured JSON matching the canonical schema, 
   including error cases

Check off tasks in @docs/phases.md as you go.
```

---

### Phase 5 — Agent demo (say this)

```
Gate 4 is passed. Read @docs/phases.md.

Build Phase 5:
1. Create examples/agent_demo.py that:
   - Takes --image as a CLI argument
   - Uses the Anthropic Python SDK with detect_objects as an available tool
   - Calls detect_objects via tool_use
   - Interprets the detections and returns a natural language summary
2. Add a single retry if tool call returns an InferenceError
3. Log all tool calls and responses to logs/agent_run_{timestamp}.json
4. Run it on /tmp/test.jpg and show me the agent's output

The agent summary should read like a human describing what's in the image,
not just echoing raw JSON.
```

---

### Phase 6 — Ship (say this)

```
Gate 5 is passed. Read @docs/phases.md.

Build Phase 6:
1. Write a multi-stage Dockerfile: build layer caches SDK install, 
   runtime layer is minimal. Test that docker build succeeds.
2. Test that docker run starts the API and the health check passes.
3. Pin all dependencies in requirements.txt with exact versions (pip freeze)
4. Write README.md covering: prerequisites, 3-step quickstart, 
   one curl example, one agent example. Nothing else.
5. Check off all Phase 6 tasks in @docs/phases.md

When all gates are checked, we are production-ready v1.
```

---

## Useful Claude Code commands during the build

```bash
# See what Claude remembers about the project
/memory

# Start fresh context for a new phase (don't use /compact)
/clear

# Reference a doc file inline
@docs/phases.md
@docs/architecture.md
@docs/model_zoo.md
@docs/curl_tests.md

# Force Claude to think harder on a hard problem
ultrathink about the error handling strategy for the MCP server
```

---

## When things go wrong

**SDK not finding the model:**
```
The Voyager SDK is throwing ModelNotFoundError. 
Read @docs/model_zoo.md and check what the exact model name format should be.
Try listing available models with the SDK's built-in list command.
```

**MCP Inspector can't connect:**
```
The MCP server is not connecting to MCP Inspector.
Read the fastmcp docs for stdio transport setup.
Check that the server is running on the correct transport (stdio not HTTP).
```

**Docker build fails:**
```
The Docker build is failing at the SDK install step.
The Voyager SDK may require specific system packages.
Check axelera's installation docs and add apt-get steps to the build layer.
```

**Agent isn't calling the tool:**
```
The agent is not calling detect_objects — it's answering from its own knowledge.
Check that the tool is passed in the tools= array to the Anthropic API.
Check that the system prompt tells the agent to use available tools for image analysis.
```

---

## What production v1 looks like

When Gate 6 is complete, you have:

- `tests/` — 15+ passing tests covering all layers
- `api/main.py` — REST API with full error contracts
- `mcp/server.py` — 4 MCP tools, schema-validated
- `axelera/inference.py` — typed wrapper, no silent failures
- `examples/agent_demo.py` — end-to-end agent workflow
- `Dockerfile` — containerised, runnable by anyone
- `README.md` — 3-step quickstart
- `docs/` — full architecture + decision record
- `logs/` — agent run traces for debugging

Total: a GitHub-ready, demo-able, extensible CV inference toolkit 
that any agent can call as a first-class tool.

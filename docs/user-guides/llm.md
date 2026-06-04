---
title: "LLM Inference"
---
# LLM Inference

Running Small Language Models (SLMs) on Metis hardware using the `axllm` command.

> [!NOTE]
> All models are precompiled. You cannot load arbitrary LLMs without prior compilation for the Metis platform. See the [Model Zoo — Large Language Models](../reference/models/model-zoo.md#large-language-models) for the full list of available precompiled language models and RAM requirements.


---

## Quick start

```bash
# Single prompt
axllm llama-3-2-1b-1024-4core-static --prompt "Give me a joke"

# Interactive chat session
axllm llama-3-2-1b-1024-4core-static

# Web UI (opens Gradio interface)
axllm llama-3-2-1b-1024-4core-static --ui
```

---

## Pipeline modes

| Mode | Description |
|------|-------------|
| `--pipeline=transformers-aipu` | *(default)* Runs the model on the Metis AIPU. All models are precompiled. |
| `--pipeline=transformers` | Runs on CPU/GPU via Hugging Face Transformers. For development and testing only — not for benchmarking. |

Some models require a 16GB AIPU card. Check the [Model Zoo — LLM RAM requirements](../reference/models/model-zoo.md#large-language-models) before selecting a model. The system will also tell you at model load time if you don't have sufficient memory.

---

## Usage modes

### Single prompt
```bash
axllm <model-name> --prompt "Your prompt here"
```

### Interactive CLI
```bash
axllm <model-name>
```
Type your messages at the prompt. The model maintains conversation history within the session. Type `exit` to quit.

### Rich CLI (formatted output)
```bash
axllm <model-name> --rich-cli
```
Colorized output with Markdown formatting and chat bubbles.

### Web UI
```bash
axllm <model-name> --ui          # public URL (accessible outside localhost)
axllm <model-name> --ui local    # local only
```
Launches a Gradio web interface. The terminal prints the URL — open it in a browser.

> [!NOTE]
> The public URL changes on each launch.


---

## Options

### Performance statistics

```bash
axllm <model-name> --show-stats --prompt "Hello"
```

Output:
```
Tokenization: 0.4ms | Prefill: 3.1us | TTFT: 0.573s | Gen: 2.111s | Tokens/sec: 9.95 | Tokens: 21
CPU %: 1.8%
Core Temp: 34.0°C
```

| Metric | Description |
|--------|-------------|
| `Tokenization` | Time to tokenise the input |
| `Prefill` | Time for context setup |
| `TTFT` | Time to first token (includes model startup) |
| `Gen` | Total generation time |
| `Tokens/sec` | Generation throughput |

### System prompt

```bash
axllm <model-name> --system-prompt "You are a helpful assistant that always answers in bullet points."
```

### Temperature

Controls randomness. `0` = deterministic. Higher = more creative.

```bash
axllm <model-name> --temperature 0.7 --prompt "Tell me a story."
```

Suggested range: 0.2 (focused) to 1.0 (creative). Default: `0`.

### Combined example

```bash
axllm llama-3-2-1b-1024-4core-static \
  --system-prompt "You are a pirate." \
  --temperature 0.8 \
  --rich-cli
```

---

## All options

```bash
axllm --help
```

---

## Embedding files

Embedding files are generated offline by Axelera AI and bundled with each precompiled model. The `axextractembeddings` tool is for Axelera-internal model preparation — end users do not need to run it.

---

## See also

- [Model Zoo](../reference/models/model-zoo.md) — available precompiled language models
- [axmonitor](axmonitor.md) — monitor memory and temperature during LLM inference
- [axdevice](../reference/tools/axdevice.md) — verify your hardware has sufficient AIPU memory

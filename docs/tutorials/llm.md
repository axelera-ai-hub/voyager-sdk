![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# SLM Inference on Axelera AI Platform

## Contents
- [SLM Inference on Axelera AI Platform](#slm-inference-on-axelera-ai-platform)
  - [Contents](#contents)
  - [Prerequisites](#prerequisites)
  - [Level](#level)
  - [Overview](#overview)
  - [Environment Setup](#environment-setup)
  - [Supported Pipelines](#supported-pipelines)
  - [Usage Modes](#usage-modes)
    - [1. Single Prompt Mode](#1-single-prompt-mode)
    - [2. Interactive CLI Mode](#2-interactive-cli-mode)
    - [3. Beautiful CLI (Rich) Mode](#3-beautiful-cli-rich-mode)
    - [4. Web UI Mode](#4-web-ui-mode)
  - [Showing Performance Statistics](#showing-performance-statistics)
  - [Customizing System Prompt and Temperature](#customizing-system-prompt-and-temperature)
    - [\`--system-prompt\`](#--system-prompt)
    - [\`--temperature\`](#--temperature)
  - [Chat Memory (History)](#chat-memory-history)
  - [Embedding File Generation](#embedding-file-generation)
  - [Additional Notes](#additional-notes)
  - [Next Steps](#next-steps)
  - [Related Documentation](#related-documentation)
  - [Further support](#further-support)


## Prerequisites
- Complete [Installation Guide](install.md)
- Understanding of language model concepts (tokens, prompts, generation)
- At least 4GB AIPU memory (some models require 16GB)

## Level
**Intermediate** - Requires understanding of LLM concepts and environment management

## Overview

Axelera AI is excited to offer support for Small Language Models (SLMs) on our Metis platform. The Voyager SDK enables a selection of popular, license-free SLMs, allowing users to experience language model inference on our hardware.
You can try several precompiled models to explore the capabilities available today. This feature demonstrates the potential of running efficient language model inference on Axelera AI's hardware architecture.

---

## Environment Setup

To run LLM inference, please run the installer first. See [Installation Guide](/docs/tutorials/install.md).

Then activate the environment as usual:
```sh
source venv/bin/activate
```
---

## Supported Pipelines

- `--pipeline=transformers-aipu` (default):
  - Runs the model on Axelera AIPU hardware. Some models require a 16GB AIPU card (the system will indicate memory requirements during model loading).
  - Fast inference, but **all models must be precompiled**.
- `--pipeline=transformers`:
  - Runs the model using the Hugging Face Transformers library on CPU or GPU (uses GPU if available).
  - **Developer/testing only:** This is for development and testing, not for production or performance benchmarking.

> [!IMPORTANT]  
> All models are precompiled for now. You cannot load arbitrary models unless they have been compiled for the platform.

---

## Usage Modes

### 1. Single Prompt Mode
Run a single prompt and exit:

```sh
axllm llama-3-2-1b-1024-4core-static --prompt "Give me a joke"
```

### 2. Interactive CLI Mode
Chat interactively with the model in your terminal:

```sh
axllm llama-3-2-1b-1024-4core-static
```

### 3. Beautiful CLI (Rich) Mode
Enable a modern, colorized, chat-bubble CLI with Markdown formatting:

```sh
axllm llama-3-2-1b-1024-4core-static --rich-cli
```

### 4. Web UI Mode
Launch a Gradio web interface for chat:

```sh
axllm llama-3-2-1b-1024-4core-static --ui
```

By default, this shares the UI publicly. To run locally only:

```sh
axllm llama-3-2-1b-1024-4core-static --ui local
```

> [!NOTE]  
> When you launch the UI mode, the console will print a URL such as:
> - `Running on local URL:  http://127.0.0.1:7860` (for local mode)
> - `Running on public URL: https://xxxx.gradio.live` (for public/share mode)
> 
> The public URL will be different each time you launch. Open the printed URL in your browser to access the chat UI.

> [!WARNING]  
> The time to first token (TTFT) may be high, especially on AIPU, due to model prefill and hardware startup.

---

## Showing Performance Statistics

You can use the `--show-stats` flag to print detailed performance statistics after each response:

```sh
axllm llama-3-2-1b-1024-4core-static --show-stats --prompt "Give me a joke"
```

Example output:
```
INFO    : Model already downloaded and verified: /home/ubuntu/voyager-sdk/build/llama-3-2-1b-1024-4core-static/model/model.json
INFO    : Found PCI device: 01:00.0 Processing accelerators: Device 1f9d:1100
INFO    : Found AIPU driver: metis                  90112  0
INFO    : Firmware version matches: v1.2.0-rc1+bl1-42-ga6b90faa5af6-dirty
INFO    : Loaded embeddings: vocab_size=128256, embedding_dim=2048
INFO    : EmbeddingProcessor initialized with file: /home/ubuntu/voyager-sdk/build/llama-3-2-1b-1024-4core-static/llama_3_2_1b_embeddings.npz
INFO    : AxInstance initialized with model: /home/ubuntu/voyager-sdk/build/llama-3-2-1b-1024-4core-static/model/model.json
Why did the AI program go to therapy? Because it had a lot of "algorithmic" issues.
INFO    : Tokenization: 0.4ms | Prefill: 3.1us | TTFT: 0.573s | Gen: 2.111s | Tokens/sec: 9.95 | Tokens: 21
INFO    : CPU %: 1.8%
INFO    : Core Temp: 34.0°C
```

**Statistics explained:**
- `Tokenization`: Time to tokenize the input.
- `Prefill`: Time for model prefill (context setup).
- `TTFT`: Time to first token (how long before the first word appears).
- `Gen`: Total generation time.
- `Tokens/sec`: Generation speed.
- `Tokens`: Number of tokens generated.
- `CPU %`: CPU usage during inference.
- `Core Temp`: Temperature of the AIPU.

---

## Customizing System Prompt and Temperature

You can control the model's behavior and creativity with the following flags:

### `--system-prompt`
Sets the system prompt (the "persona" or instructions for the assistant). Useful for customizing the assistant's style, role, or constraints.

**Example:**
```sh
axllm llama-3-2-1b-1024-4core-static --system-prompt "You are a helpful assistant that always answers in haiku."
```

### `--temperature`
Controls the randomness/creativity of the model's responses.
- `0` = deterministic (always the same answer)
- Higher values (e.g. `0.7`, `1.0`) = more creative/random
- **Suggested range:** 0.2 (more focused) to 1.0 (more creative). Most users find 0.7 a good balance.
- **Default:** `0` (fully deterministic, lowest creativity).

**Example:**
```sh
axllm llama-3-2-1b-1024-4core-static --temperature 0.7 --prompt "Write a creative story about a robot."
```

**Combined example:**
```sh
axllm llama-3-2-1b-1024-4core-static --system-prompt "You are a pirate." --temperature 1.0 --rich-cli
```

---

## Chat Memory (History)

The CLI supports a simple chat memory system: the assistant remembers previous turns in the current session, so you can have a multi-turn conversation.

Example:
```
axllm llama-3-2-1b-1024-4core-static
INFO    : Welcome to LLM CLI. Type 'exit' to quit.
User: give me a joke
Assistant: Why did the AI program go to the doctor? Because it had a glitch and was feeling a little "programmed to be happy."
User: what did I ask
Assistant: You asked me to give you a joke.
User: exit
INFO    : Goodbye!
```

---

## Embedding File Generation

The `axextractembeddings` tool is used to generate embedding files **offline**. These embedding files are required at runtime for fast inference, but are **not** regenerated each time you run the model. You only need to generate the embedding file once (typically done by Axelera AI or during model preparation).

- **Command:** `axextractembeddings` (`--help` for details)
- **Purpose:** Extracts and saves the model's embedding matrix to a file for use by the runtime.

> [!TIP]
> End users typically do not need to run this command unless preparing a new model for deployment.

---

## Additional Notes

- The CLI and UI both support streaming output, but TTFT may be high due to model prefill.
- For more advanced usage, see the help: `axllm --help`

---

## Next Steps
- **Explore available models**: Browse LLM section in [Model Zoo](../reference/model_zoo.md)
- **Monitor resource usage**: Use [AxMonitor](axmonitor.md) to track memory consumption

## Related Documentation
**Tutorials:**
- [Installation Guide](install.md) - Create the separate voyagersdk virtual environment
- [Quick Start Guide](quick_start_guide.md) - General SDK concepts apply to LLM workflow

**References:**
- [Model Zoo](../reference/model_zoo.md) - Lists available precompiled LLM models
- [AxDevice API](../reference/axdevice.md) - Verify hardware has sufficient memory

--- 
## Further support

For blog posts, projects and technical support please visit [Axelera AI Community](https://community.axelera.ai/).

For technical documents and guides please visit [Customer Portal](https://support.axelera.ai/).

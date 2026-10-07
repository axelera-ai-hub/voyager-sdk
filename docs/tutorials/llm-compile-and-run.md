---
title: "Compile and Run an LLM"
description: Quantize, compile and run HuggingFace causal-LMs on Axelera Metis using the axelera-transformers package.
---
# Compile and Run an LLM

> [!WARNING]
> **Experimental.**
> LLM support is in early access. The set of supported architectures and sequence lengths will change.

This page covers quantizing a HuggingFace causal-LM, compiling it for Metis, and running or benchmarking the compiled artifact through `axzoo build` and the generated LLM recipe.

> [!NOTE]
> **Two LLM paths, and which page to read.**
> This page is the **axelera-transformers** path, which compiles through AxMO and the graph
> compiler. [Serve an AxScript LLM through HuggingFace, or Your Own Front End](axscript/llm-huggingface.md) is
> the **AxScript** path, whose device kernels are hand-authored and whose KV cache lives on the
> AIPU. Both present the same HuggingFace API. Use the page that matches the artifact you have.

---

## Supported models

The following checkpoints are tested end-to-end. Other models from the same families up to 8B parameters typically work, but accuracy and performance have not been validated.

| Family | Model | Max context length |
|---|---|---|
| Llama | [meta-llama/Llama-3.2-1B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) | 4096 |
| Llama | [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) | 4096 |
| Llama | [meta-llama/Llama-3.1-8B-Instruct](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | 2048 |
| Qwen3 | [Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B) | 4096 |
| Qwen3 | [Qwen/Qwen3-1.7B](https://huggingface.co/Qwen/Qwen3-1.7B) | 4096 |
| Qwen3 | [Qwen/Qwen3-4B](https://huggingface.co/Qwen/Qwen3-4B) | 4096 |

---

## Prerequisites

Install `axelera-devkit[all]` by following the [PIP installation guide](../user-guides/sdk-install.md#python-pip-installation). That extra provides the AxMO quantizer, the `axelera-graph-compiler` backend, and the `axelera-transformers` package. 

Activate the virtual environment:

```bash
source axelera-env/bin/activate
```

Calibration uses CUDA automatically when PyTorch detects it and otherwise uses CPU. To override that selection, run the standalone recipe with `--cal-device`, for example `python model_recipes/llm/llama-3.2-1b.py --cal-device cuda`; `axzoo build` does not expose a device override. Compilation itself always runs on CPU, and compilation of 3B and 8B models requires a host machine with at least 128 GB of RAM.

---

## Calibration defaults

The generated LLM recipe builds calibration data with `get_multi_dataset_calibration_dataloader`, using a mix of Wikitext, MMLU, and HellaSwag. The defaults reflect the configuration used to validate the supported models above:

| Setting | Default | Flag |
|---|---|---|
| Calibration samples | 1024 | `axzoo build --num-cal` or `AXZOO_NUM_CAL` |
| Calibration batch size | 1 | `AXZOO_CALIBRATION_BATCH_SIZE` |
| SmoothQuant alpha | 0.8 | Baked in `LlmBuildConfig` |
| Compile sequence length | Per config | `axzoo build --seq-len` or recipe `--seq-len` |

The compile sequence length doubles as the calibration block size; each calibration sample is tokenized to that same length.

---

## Compile a Llama model

Llama checkpoints quantize cleanly with defaults:

```bash
# 1B at 4k context; output directory defaults to ./build/llama-3.2-1b
axzoo build llama-3.2-1b

# 8B at 2k context
axzoo build llama-3.1-8b --output ./compiled_llama_8b/
```

Compiling the 8B model can take several hours. Calibration on CUDA is much faster than CPU; the compile step itself always runs on CPU.

## Compile a Qwen3 model

Qwen3 attention is more sensitive to quantization than Llama's, so the Qwen3 zoo configs bake `smooth_qk_alpha=0.5`; no CLI flag is needed because it is part of the config.

```bash
axzoo build qwen3-1.7b --output ./compiled_qwen3-1.7b/
```

### Build arguments

| Argument | Default | Description |
|---|---|---|
| `axzoo build <llm>` | Built-in config | Selects the HuggingFace model and baked quality constants |
| `--output DIR` | `./build/<config>` | Compiled artifact directory |
| `--num-cal N` | config default | Calibration samples for this build only |
| `--seq-len N` | config default | Max sequence length, also used as calibration block size |
| `axzoo fork <llm> --script` | `./<model-id>.py` | Writes an editable recipe copy |

---

## Run inference

The compiled model exposes the standard HuggingFace causal-LM API. The generated recipe includes deterministic `--run` and chat-template `--chat` smoke modes:

```bash
axzoo fork qwen3-1.7b --script --force
python qwen3-1.7b.py --run \
    --output-dir ./compiled_qwen3-1.7b/ \
    --device-id 0 \
    --prompt "Large language models are neural networks" \
    --max-new-tokens 50
```

### Embedding in your own code

```python
import axelera.transformers
import transformers

model = axelera.transformers.AxLMTransformerForCausalLM.load_compiled(
    "./compiled_qwen3-1.7b/", device_id=0,
)
tokenizer = transformers.AutoTokenizer.from_pretrained("./compiled_qwen3-1.7b/")

inputs = tokenizer("The future of AI is", return_tensors="pt").to(model.device)
output_ids = model.generate(**inputs, max_new_tokens=50, do_sample=False)
print(tokenizer.decode(output_ids[0], skip_special_tokens=True))
```

### Benchmark

```bash
axzoo benchmark qwen3-1.7b \
    --output-dir ./compiled_qwen3-1.7b/ \
    --device-ids 0 1 2 3 \
    --prompt "Explain what an AI accelerator is in 3 sentences." \
    --max-new-tokens 50 \
    --warmup 1 \
    --iterations 10 \
    --output-csv llm_results.csv
```

The standalone recipe exposes the same benchmark action:

```bash
python qwen3-1.7b.py --benchmark \
    --output-dir ./compiled_qwen3-1.7b/ \
    --device-ids 0 1 2 3 \
    --prompt "Explain what an AI accelerator is in 3 sentences." \
    --max-new-tokens 50 \
    --warmup 1 \
    --iterations 10
```

---

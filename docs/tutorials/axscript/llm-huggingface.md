---
title: "Serve an AxScript LLM through HuggingFace, or Your Own Front End"
description: Load a compiled AxScript LLM or VLM with AutoModelForCausalLM.from_pretrained, generate with the sampling parameters you already use, or drive the same engine from a front end of your own.
---
# Serve an AxScript LLM through HuggingFace, or Your Own Front End

> [!WARNING]
> **Experimental.**
> LLM support is in early access, and the set of supported architectures and context lengths
> will change. Run `python -m model_axscript.llm list` to see which cards your install carries.

A compiled AxScript artifact is a HuggingFace model directory. You load it with the Auto
classes, generate with the parameters you already pass, and stream tokens with the streamers
you already use. Nothing about the call is Axelera-specific except the path.

**HuggingFace is a plugin here, not the interface.** The engine underneath takes token ids in and
returns logits, on numpy, and it names no adapter and branches on none. `transformers` reaches it
through ONE adapter file, `llm/huggingface.py`, and a second front end is added BESIDE that one
rather than by changing anything below it: an OpenAI-shaped endpoint, another serving framework,
or an edge target carrying neither torch nor transformers. If adding one ever needs an engine
change, the boundary has already been crossed and that is the defect.
[Writing your own front end](#writing-your-own-front-end) is the four primitives your adapter
would call.

> [!IMPORTANT]
> **This page is the text and vision-language families.** A vision card has no HuggingFace
> serving adapter: nothing registers an Auto class for it, and its output is a tensor its own
> `run` reports rather than generated tokens. Vision does use HuggingFace, but on the other side
> of the build, as one place a card's WEIGHTS can come from, which is
> [Build an AxScript Model from Your Own Weights](custom-weights.md). To run a vision
> card, use [Compile and Run an AxScript Model](quickstart.md), which is also where to
> start if you have not built an artifact yet: this page begins from one that already exists.

> [!NOTE]
> **Two LLM paths, and which page to read.**
> This page is the **AxScript** path: hand-authored device kernels, the whole decoder and its
> KV cache on the AIPU. [Compile and Run an LLM](../llm-compile-and-run.md) is the
> **axelera-transformers** path, which compiles through AxMO and the graph compiler. They
> present the same HuggingFace API. Use the page that matches the artifact you have.

---

## Prerequisites

The AxScript LLM path needs PyTorch, `transformers` and `safetensors` on top of the runtime.
It is a build-and-run toolkit, not part of the minimal runtime install.

```bash
pip install "torch>=2.8" "transformers>=4.57,<4.58" safetensors
```

The version window is not cosmetic: the published accuracy numbers were measured on 4.57.x,
and the tokenizer fallback this tree carries exists for checkpoints that version cannot load
on its own.

**Building and running need different things**, which matters if you deploy somewhere smaller
than you build:

| | `numpy` + `axelera.runtime` | PyTorch | `transformers` |
|---|---|---|---|
| **build** (quantize and compile) | required | **required**: the reference model, the calibration observers and SmoothQuant are all torch | **required**: the tokenizer, and the chat templates the calibration mix uses |
| **eval** (score an artifact) | required | **required**: the fp32 reference leg is a torch module, and the device leg is a torch `PreTrainedModel` | required |
| **run** (generate from a built artifact) | required | optional, and worth having | only if you use the HuggingFace API on this page |

On the run path torch is an accelerator rather than a dependency. The host lm_head uses its
fp16 CPU gemv when torch is installed (about 14 ms per step against about 29 ms for the fp32
alternative, argmax-identical) and falls back to numpy when it is not, so a machine without
torch runs slower rather than not at all.

---

## How it fits together

HuggingFace is a **plugin over the engine**, not the engine itself:

```
   your script / notebook / pipeline          a future OpenAI-style endpoint, another
                   |                          serving framework, or your own front end
             transformers                                      |
                   |                                           |
       llm/huggingface.py  <------- ONE ADAPTER, replaceable -------->  (a sibling adapter)
   forward() + prepare_inputs_for_generation             no engine change to add one
                   |
       llm/implementation.py    THE ENGINE: token ids in, logits out, on numpy.
                   |            Names no adapter. Branches on no adapter.
           axelera.runtime      KV and recurrent state live HERE, on the AIPU:
                                not inspectable, not reorderable, not duplicable
```

The figure shows the text path. A vision-language model has an adapter of its own
(`models/qwen3vl/`, `models/smolvlm2/`, `models/lfm2vl/`), presenting the same HuggingFace API
over the same engine.

Two things follow, and both matter to you.

Everything on this page that the device **refuses** comes from the bottom line, not from a
gap in the adapter. One KV cache on the accelerator is why beam search and batches raise
rather than run, and no version of the adapter changes that.

And the adapter is **replaceable**. The engine names no adapter and branches on no adapter, so
if the HuggingFace API is not the interface you want, you write a second adapter beside ours
rather than forking the engine. See "Writing your own front end" below.

---

## Build an artifact

```bash
python -m model_axscript.llm list                       # every tag, and where it lives
python -m model_axscript.llm build qwen3-0.6b           # compile; artifact -> build/qwen3-0.6b/axscript
python -m model_axscript.llm build qwen3-0.6b --seq-len 2048
```

`list` is the source of truth for which tags your install carries, and the examples below use
`qwen3-0.6b` only as a stand-in. Substitute a tag `list` prints.

The context length is compiled in. The KV cache and the prefill kernel are sized from it, and
the prefill attention mask grows with its square, so a larger context is a different artifact
rather than a run-time choice. `--seq-len` accepts only the sizes the card DECLARES; anything
else is refused by name before the build starts, because on hardware an undeclared size would
surface as a device hang minutes in rather than as an error. Not every declared size has been
measured on a device: where a card declares small sizes (128, 256, 512) ahead of that
measurement, treat their latency and accuracy as unestablished. The refusal message prints the
live set, and `supported_context_lens` in the card's `make_config` is where it is declared.

The build writes `config.json`, the tokenizer (chat template included) and
`generation_config.json` into the artifact directory, which is what makes the next section
work with no hub and no network.

---

## Run it

```python
from model_axscript.llm.huggingface import register_auto
from transformers import AutoModelForCausalLM, AutoTokenizer

register_auto()   # once per process, before the first from_pretrained

model = AutoModelForCausalLM.from_pretrained("build/qwen3-0.6b/axscript")
tokenizer = AutoTokenizer.from_pretrained("build/qwen3-0.6b/axscript")

inputs = tokenizer("The capital of France is", return_tensors="pt")
ids = model.generate(**inputs, max_new_tokens=32, do_sample=False)
print(tokenizer.decode(ids[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

`register_auto()` is what teaches the Auto classes the artifact's model type, `ax_causal_lm`. A
stock transformers cannot discover a type that another package defines, and this artifact
carries no `auto_map`, which is exactly what lets it load with no `trust_remote_code`. Without
that call the load fails with "Transformers does not recognize this architecture". Importing
the module is not enough on purpose: registration builds the model class, which imports torch,
and the engine is meant to load on a machine that has none.

Every family has the same call, on the module that carries its adapter, so moving between a
text model and a vision-language one does not mean learning a second spelling.

On a host with more than one Metis part, name the one to run on. The default is
`$AXELERA_METIS_DEVICE`, and then the first part, so two processes would otherwise land on the
same one:

```python
model = AutoModelForCausalLM.from_pretrained(
    "build/qwen3-0.6b/axscript", device=1        # or a name: "metis-0:2d:0"
)
```

Generation is `transformers.GenerationMixin.generate` itself, so every sampling parameter is
the library's own:

```python
ids = model.generate(
    **inputs,
    max_new_tokens=128,
    do_sample=True,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    min_p=0.05,
    repetition_penalty=1.1,
)
```

Chat templates, streaming and the text-generation pipeline work the same way:

```python
messages = [{"role": "user", "content": "Name three primary colours."}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
)

from transformers import TextIteratorStreamer
streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
model.generate(**inputs, max_new_tokens=128, streamer=streamer)
for chunk in streamer:
    print(chunk, end="", flush=True)
```

```python
from transformers import pipeline

generate = pipeline("text-generation", model="build/qwen3-0.6b/axscript")
print(generate("The capital of France is", max_new_tokens=32))
```

Or from the command line, with no Python of your own:

```bash
python -m model_axscript.llm run       qwen3-0.6b "The capital of France is"
python -m model_axscript.llm run       qwen3-0.6b --interactive     # multi-turn chat
python -m model_axscript.llm benchmark qwen3-0.6b                   # time to first token, tok/s
```

### A vision-language model

Same shape, through the image-text-to-text Auto class. The vision tower is compiled for one
patch grid, so pictures are resized to it first, which is the one Axelera-specific line:

```python
from model_axscript.llm.models.qwen3vl.hf_model import register_auto
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

register_auto()   # once per process, as for a text model

model = AutoModelForImageTextToText.from_pretrained("build/qwen3-vl-2b-instruct/axscript")
processor = AutoProcessor.from_pretrained("build/qwen3-vl-2b-instruct/axscript")

image = model.fit_images(Image.open("photo.jpg"))          # the compiled grid; skip it and the check tells you the size
messages = [{"role": "user", "content": [
    {"type": "image", "image": image}, {"type": "text", "text": "Describe it."}]}]
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt"
)
ids = model.generate(**inputs, max_new_tokens=64)
print(processor.batch_decode(ids[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True))
```

---

## Your own weights

`python -m model_axscript.llm build <tag> --weights <dir-or-id>` builds a card from your
fine-tune of the architecture it implements, and it is the same flag at the vision door. The
walkthrough is one page for all three families:
[Build an AxScript Model from Your Own Weights](custom-weights.md).

`--weights` is a BUILD parameter. Once an artifact exists, the checkpoint it was built from is
recorded in its `metadata.json`, and a later `run`, `benchmark` or `eval` reads it from there,
so the Auto classes load your tokenizer from the artifact without you naming it again.

---

## Your own calibration data

`--calibration` takes a text corpus for a text card, and it is written up with the flag
walkthrough:
[Your own calibration data](custom-weights.md#your-own-calibration-data).

A vision-language card refuses the flag by name, because its two towers are calibrated
together: a sample is an (image, question, answer) row and not a picture, so the decoder's
scales see a real prompt attached to the embeddings the vision tower produced. Pairing your
images with one invented question would fit the scales to a prompt distribution nobody runs.

That makes it a Python API rather than a flag, which is why it lives on this page. It is one
call per model:

```python
from model_axscript.llm.models.qwen3vl import pipeline

samples = pipeline.multimodal_calibration(processor, config, images=my_images, prompts=my_questions)
pipeline.build_artifacts(repo_id, out_dir, calibration=samples, image_size=..., context_len=...)
```

---

## Writing your own front end

You do not have to go through `transformers`. Our adapter is a thin layer over four engine
primitives, and yours can call the same ones:

| Primitive | What it does |
|---|---|
| `Implementation.ingest_prompt(runtime, host_params, prompt, context_len, chunk_len)` | seeds the device cache with a prompt; returns the hidden state, the row holding the last real token, and the position the next token is written at |
| `Implementation.decode_step(runtime, host_params, token_id, position, context_len)` | one token in, next-token logits out |
| `Implementation.compute_logits(hidden, host_params, position=-1)` | the host lm_head at one position |
| `Implementation.compute_logits_seq(hidden, host_params)` | the host lm_head at every position, for teacher-forced scoring |

`RuntimeModel(model_json, num_sub_devices=None, device=None)` from
`model_axscript.shared.runtime` loads the artifact and reserves the cores the build spans. Read `model_axscript/llm/huggingface.py` as the worked example: it is one file,
and the two methods that matter are `forward` and `prepare_inputs_for_generation`.

Two things your adapter owns, because the engine deliberately does not:

* **Sampling.** The engine carries a greedy choice and nothing more. `transformers` brings its
  own samplers, and so would any other framework, so growing the engine's would give it a
  dependency on one front end in behaviour if not in imports.
* **The write cursor.** The device holds the cache but the host tracks where the next token
  goes. Our adapter takes that from HuggingFace's `cache_position`; yours tracks it however it
  likes, as long as it advances one position per decoded token.

> [!NOTE]
> A front end needs neither `transformers` nor torch: the engine imports without either. Torch is
> still worth installing on the run host, because the host lm_head uses its fp16 gemv when it is
> present (about 14 ms per step against about 29 ms for the numpy fp32 fallback).

---

## What this device cannot do

The AIPU holds one KV cache per loaded model, and it cannot be inspected, reordered or
duplicated. Anything that needs several caches raises with the reason rather than returning a
plausible answer from the wrong distribution:

| You ask for | What happens |
|---|---|
| `num_beams > 1` | `NotImplementedError`: beam search needs several caches |
| `num_return_sequences > 1` | `NotImplementedError`: same reason |
| a batch of prompts | `NotImplementedError`: loop the prompts, or use `batch_size=1` |
| `assistant_model=...` | `NotImplementedError`: assisted decoding needs a cache that can be rolled back |
| `penalty_alpha=...` | `NotImplementedError`: contrastive search needs candidate hidden states and a reorderable cache |
| a prompt longer than the compiled context | `ValueError` naming the compiled length |
| `model.save_pretrained(...)` | `NotImplementedError`: the model IS the artifact directory, so copy it |

`past_key_values` comes back as `None`. HuggingFace only threads it through, and this cache
never leaves the device.

One consequence worth planning around rather than discovering: **each `generate()` call
re-reads the whole prompt.** The adapter keeps no prefix cache across calls, so a long
multi-turn conversation pays for its history every turn.

---

## Scoring it

Accuracy goes through the zoo's own evaluation driver, so a number here is comparable with a
number taken for any other model. It needs `pip install "axelera-zoo[llm-eval]"`:

```bash
python -m model_axscript.llm eval qwen3-0.6b --tasks wikitext
python -m model_axscript.llm eval qwen3-0.6b --tasks wikitext --engine host   # the fp32 reference leg
```

With no `--engine`, both legs run and the gap between them is printed, which is the number
that tells you what quantization cost.

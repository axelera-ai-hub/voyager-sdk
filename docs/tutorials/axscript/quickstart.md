---
title: "Compile and Run an AxScript Model"
description: Build one of the AxScript cards this SDK ships, run it on a Metis device, and read the numbers it prints, before changing anything about it.
---
# Compile and Run an AxScript Model

> [!WARNING]
> **Experimental.**
> AxScript is in early access. The set of supported architectures will change.

AxScript is a set of hand-authored recipes that compile a model straight to a Metis artifact,
with no pipeline YAML and no graph compiler in between. Each recipe is a CARD, addressed by a
tag, and each card ships pointing at a public checkpoint. This page reproduces a card exactly as
it ships. Change nothing yet: a plain build is the baseline every later number is read against.

There are two doors, one per family. `vision` answers for the picture models, `llm` for the text
models and the vision-language ones, which share a decoder.

```bash
python -m model_axscript.vision list
python -m model_axscript.llm list
```

`list` is the source of truth for what your install carries, and the tags below are stand-ins.
Substitute one that `list` prints.

```
dinov2-vits14                    .../vision/feature_extraction/dinov2_vits14
dinov2_seg_vits14_ade20k_linear  .../vision/semantic_segmentation/dinov2_seg_vits14_ade20k_linear
vit_small_patch16_224-imagenet   .../vision/classification/vit_small_patch16_224_imagenet
```

`python -m model_axscript` with no family is not a command line. It prints the door each family
uses and exits non-zero, so an old command teaches the new one rather than continuing to work in
silence.

---

## Before you start

**Text and vision-language** need PyTorch, `transformers` and `safetensors` on top of the
runtime. Building is a toolkit, not part of the minimal runtime install:

```bash
pip install "torch>=2.8" "transformers>=4.57,<4.58" safetensors
```

The two ends are there for different reasons. The floor is an import:
`transformers.models.qwen3_vl`, which the Qwen3-VL card loads, first ships in 4.57.0, so an
install below it succeeds and then fails at import. The ceiling is the version the published
numbers were measured at, and every result this toolkit writes records the `transformers`
version beside it, so running a different one is a different measurement rather than an error.

**Vision** compiles through the compiler's own vision transformer offload,
`axelera.compiler.axten`. Not every compiler build carries it. If yours does not, the first build
stops before it downloads or calibrates anything and says so:

```
ModuleNotFoundError: this compiler build has no `axelera.compiler.axten`, which the axten
vision builder needs to compile a ViT or DINOv2 backbone. Use a compiler build that
provides it. `pip show axelera-compiler` reports the build you are on.
```

Nothing on the vision side of this page will work until that is resolved.

---

## Build it

```bash
python -m model_axscript.vision build vit_small_patch16_224-imagenet
python -m model_axscript.llm    build qwen3-0.6b
```

The build downloads the card's checkpoint, quantizes it against the card's own calibration set,
compiles the device graph and writes the artifact to `build/<tag>/axscript`. It is a compile, not
a download: expect minutes, and expect the calibration pass to use a GPU when one is present
(`--cal-device cpu` forces CPU).

What lands there is a self-contained artifact. `model.json` is the device graph, `pipeline.json`
names the stages, `host_params.npz` carries whatever the host computes (a classifier, a pixel
scale), and `meta.json` records what the build was given. A run needs none of the upstream
checkpoint.

**One configuration per card.** The input size for a vision card and the context length for a
text one are COMPILED IN: the device buffers and the attention mask are sized from them, so a
different value is a different artifact rather than a run-time choice. Each card ships the one
configuration its published numbers were measured at, and a value it has not measured is refused
by name before the build starts, because on hardware an unmeasured size surfaces as a device hang
minutes in rather than as an error.

`build` is not something you run twice. `run` and `benchmark` build first when nothing is there,
so the commands below work from a clean tree.

---

## Run it

A vision card takes one or more images, positionally. The images come BEFORE the flags, because
`run` accepts any number of them and a flag in between would swallow the rest of the line.

```bash
python -m model_axscript.vision run vit_small_patch16_224-imagenet my-photo.jpg
```

The runtime logs a banner first (device, cores, the compiled run size), and every number below is
only comparable against another taken at the same settings, so the banner is part of the output
rather than debug chatter. Then:

```
my-photo.jpg: top-5 ImageNet classes
  48.09%  bannister, banister, balustrade, balusters, handrail (421)
  28.03%  knee pad (615)
   1.03%  unicycle, monocycle (880)
   0.95%  ping-pong ball (722)
   0.62%  coil, spiral, volute, whorl, helix (506)
  fp reference: top-1 agrees, logit cosine 0.9933
```

The last line is the one worth reading. It runs the SAME weights in floating point on the host
and compares, so it isolates what compiling cost you from what the model was always going to get
wrong. A top-1 that agrees and a cosine near 1 mean the artifact is faithful; the model being
wrong about the picture is a different question.

A card whose head runs on the device reports the same way, in the units of its own task:

```bash
python -m model_axscript.vision run dinov2_seg_vits14_ade20k_linear my-photo.jpg
```

```
my-photo.jpg: 81.6% of 1369 patches take the fp reference's label, mean patch cosine 0.9783
  predicted: wall 46%, floor 26%, person 15%, bench 5%, tree 2%
```

A text model takes a prompt instead, and falls back to the card's own when you give none:

```bash
python -m model_axscript.llm run qwen3-0.6b "The capital of France is"
python -m model_axscript.llm run qwen3-0.6b --interactive
python -m model_axscript.llm run qwen3-vl-2b-instruct "what is this?" --image photo.jpg
```

---

## Time it

`benchmark` times the model with the host-side decoding lifted out of the timed loop, so what it
reports is the artifact and not the image decoder. It takes exactly one image, because a mean
over different images measures the images as much as the model:

```bash
python -m model_axscript.vision benchmark vit_small_patch16_224-imagenet my-photo.jpg
```

```
{'ms': 8.23035350185819, 'fps': 121.50146403483487}
```

---

## Choose a device

`--device` names one Metis part, by index or by name. The artifact is device-agnostic, so this
never recompiles:

```bash
python -m model_axscript.vision run vit_small_patch16_224-imagenet my-photo.jpg --device 1
python -m model_axscript.llm    run qwen3-0.6b --device metis-0:2d:0
```

On a host with several parts this is worth passing rather than leaving to the default, so two
processes do not land on the same one.

---

## What is fixed, and what you still choose

| Fixed when the artifact is built | Chosen at run time |
|---|---|
| the weights that were compiled | which Metis part runs it (`--device`) |
| the input size or context length | the input: images, or a prompt |
| the kernels and the quantization scheme | every generation parameter, for a text model |
| the number of AI cores the build spans | |

Asking at run time for something in the left column is refused naming the bound and the rebuild,
never served as a silently different thing.

---

## Score it

Scoring belongs to whoever owns the task, so it is not one verb. A text card has `eval`, which
scores through the zoo's evaluation driver and needs `pip install "axelera-zoo[llm-eval]"`:

```bash
python -m model_axscript.llm eval qwen3-0.6b --tasks wikitext
```

A vision card does not, and the vision door does not offer one. Score it through the zoo, which
owns the metric per task:

```bash
axzoo eval vit_small_patch16_224-imagenet --backend axscript
```

---

## Where to go next

* [Serve an AxScript LLM through HuggingFace, or Your Own Front End](llm-huggingface.md) loads a built
  text or vision-language artifact with the Auto classes, so your existing `generate()` code runs
  it unchanged. It also shows how to drive the engine from a front end of your own.
* [Build an AxScript Model from Your Own Weights](custom-weights.md) points a card at
  your fine-tune, your output head, and your calibration data.

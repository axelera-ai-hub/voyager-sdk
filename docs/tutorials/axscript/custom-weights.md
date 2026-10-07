---
title: "Build an AxScript Model from Your Own Weights"
description: Compile a supported AxScript architecture for Metis from weights you trained yourself, across the vision, text and vision-language families, including a head with a different number of classes.
---
# Build an AxScript Model from Your Own Weights

> [!WARNING]
> **Experimental.**
> AxScript is in early access. The set of supported architectures will change.

AxScript is a set of hand-authored recipes that compile a model straight to a Metis artifact.
Each recipe is addressed by a tag, and each one ships pointing at a public checkpoint. This
page covers pointing one at your weights instead.

The starting point this page assumes: you took an architecture the SDK already supports, you
fine-tuned it on your own data, and your model may predict a different number of classes, or
speak a different domain, than the public one does.

> [!NOTE]
> **Build the card as it ships first.** [Compile and Run an AxScript
> Model](quickstart.md) is one command per step and gives you the baseline every number
> on this page is read against. Come back here once that runs.

One page for all three families, because it is one concept and the flags are spelled the same
way at every door. Where a family differs, the difference is called out rather than split into
a page of its own.

---

## Before you start

There are two doors, one per family, and each answers for its own tags:

```bash
python -m model_axscript.vision list    # picture models
python -m model_axscript.llm list       # text models and vision-language models
```

`list` is the source of truth for which tags your install carries. The tags below are
stand-ins; substitute one that `list` prints.

**Text and vision-language** need PyTorch, `transformers` and `safetensors` on top of the
runtime. Building is a toolkit, not part of the minimal runtime install:

```bash
pip install "torch>=2.8" "transformers>=4.57,<4.58" safetensors
```

**Vision** compiles through the compiler's own vision transformer offload,
`axelera.compiler.axten`. Not every compiler build carries it. If yours does not, the first
build stops before it downloads or calibrates anything and tells you so:

```
ModuleNotFoundError: this compiler build has no `axelera.compiler.axten`, which the axten
vision builder needs to compile a ViT or DINOv2 backbone. Use a compiler build that
provides it. `pip show axelera-compiler` reports the build you are on.
```

Nothing on the vision side of this page will work until that is resolved.

---

## What you can replace

One command builds a model, and three flags decide what it builds from. Omit a flag and the
model uses what it ships with, so a plain `build` always reproduces the published model.

| Flag | Takes | Omit it and you get |
|---|---|---|
| `--weights` | A Hugging Face repo id, or a local directory in checkpoint layout | The checkpoint the model ships |
| `--head` | A path to your own output head weights | The head the model ships |
| `--calibration` | Your own calibration data: images or text, as a directory or a hub id | The model's own calibration set |

Not every family has all three, because not every family has the thing the flag replaces. A
door refuses a flag it does not honour by name, before anything is downloaded or compiled:

| Family | `--weights` | `--head` | `--calibration` |
|---|---|---|---|
| Vision | yes | yes, for a card that compiles one (segmentation, depth) | yes |
| Text | yes | no head to replace | yes |
| Vision-language | yes, both towers | no head to replace | no: see [Vision-language calibration](#vision-language-calibration) |

All three are BUILD parameters. `eval`, which never builds, refuses them by name.

---

## What is yours, and what the recipe fixes

An AxScript model is a hand-authored recipe. The architecture, the kernels, the input size and,
where a model has one, the head's topology are all decided in the SDK and measured on a device
before the model ships. That is what makes it a one-command build, and it is also the boundary
of what you can change:

| Yours | Fixed by the recipe |
|---|---|
| The weight VALUES, backbone and head alike | The architecture those weights load into |
| How many classes or bins you predict | The layers your head is made of |
| The calibration data the 8-bit ranges are fitted to | The input size, the kernels, and what runs on the device |

So read `list` as the supported set. A tag is a model this path can build, and an architecture
with no tag has no recipe: there is no flag, configuration field or plugin that adds one, because
a model here IS a compiled graph plus the measurement behind it. If yours is not in the list,
that is a request for a new model rather than something to work around on this page.

The same holds one level down for a head. A model that compiles a head compiles exactly one
topology for its task, and there is no route today for a head of another shape: not on the
device, and not as a host-side stage this path would assemble for you. What a head must be is
under [What the head cannot be](#what-the-head-cannot-be).

---

## Build from your own checkpoint

`--weights` takes a Hugging Face repo id or a local directory, the way
`from_pretrained("Qwen/...")` and `from_pretrained("./my-finetune")` are one call upstream, so
a fine-tune that was never pushed to the hub needs no push and no hub account.

```bash
python -m model_axscript.vision build vit_small_patch16_224-imagenet --weights /path/to/my-finetune
python -m model_axscript.llm    build qwen3-0.6b   --weights ./my-qwen3-finetune
python -m model_axscript.llm    build qwen3-0.6b   --weights my-org/qwen3-0.6b-my-domain
python -m model_axscript.llm    build qwen3-vl-2b-instruct --weights ./my-vlm-finetune   # both towers
```

The card still decides the architecture, the kernels and the compiled context or input size.
Only the weights come from you, and quantization runs on your weights, so the int8 scales are
fitted to them and not to ours.

What the flag takes is a Hugging Face checkpoint LAYOUT: `config.json`, the weight files and,
for a text model, the tokenizer files. A bare `state_dict` (`.pt` / `.pth`) is not a
checkpoint, so save it as one first, which is also what makes it loadable by anything else:

```python
model.save_pretrained("./my-qwen3-finetune")
tokenizer.save_pretrained("./my-qwen3-finetune")
```

For a classification card the classifier lives in the same checkpoint as the backbone, so
`--weights` brings both, and the class count is read out of what you supplied. A model with
seven classes instead of a thousand needs no extra flag.

Your fine-tune must be the architecture the card implements. The graph is never read from the
checkpoint: a card names a reference block and the kernels compiled for it, and the
checkpoint's own `config.json` supplies the sizes. A different architecture needs a new card,
not a new checkpoint.

---

## Build with your own head

Vision only, and only for a card that compiles a head: segmentation and depth. Their head runs
ON THE DEVICE, in the same single dispatch as the backbone, so a head that predicts a different
number of classes is a different compiled artifact rather than a different setting at run time.

You supply weights. You do not cut the graph, choose what is offloaded, or touch the compiler:
the device recipe for that head is authored in the SDK, and your file feeds it.

```bash
python -m model_axscript.vision build dinov2_seg_vits14_ade20k_linear --head /path/to/my-head.pth
```

```
[build] dinov2_seg_vits14_ade20k_linear -> build/dinov2_seg_vits14_ade20k_linear/axscript-dddd8785
```

Your weights and your class count are yours. The head's SHAPE is not: the recipe implements one
head topology per task, a BatchNorm and a 1x1 convolution for segmentation, and a file of another
shape is refused rather than compiled into something it is not. See the refusals below.

`--head` and `--weights` are not paired: either works on its own, because your head over the
shipped backbone is a normal thing to want.

The file is a PyTorch state dictionary of the head's layers, in the same unfolded form the
public head is published in: the normalization parameters and the final convolution, under
their own names. Give it as the layers, not as a head you have already collapsed into a single
matrix. The number of classes comes from the shape of that final convolution, so you never
declare it twice.

If the file is not a head of that shape, the build stops before it compiles anything, names the
parameters it could not find, and says what this card's head IS:

```
ValueError: my-head.pth is not a segmentation head this card can compile: missing
['decode_head.bn.weight', 'decode_head.bn.bias', 'decode_head.bn.running_mean',
'decode_head.bn.running_var', 'decode_head.conv_seg.bias']. This card compiles a LINEAR
segmentation head: a BatchNorm over the backbone's embedding dimension, then a 1x1
convolution onto your classes. Save your head as those layers, not as one you have
already collapsed.
```

If the head was trained on a backbone of a different width, the build stops before compiling
and names both widths:

```
ValueError: the segmentation head my-head.pth expects a 768-wide backbone, and this
one is 384 wide. Pair the head with the backbone it was trained on.
```

### What the head cannot be

The topology belongs to the recipe, so each of these is refused before anything is downloaded or
compiled, rather than supported in a degraded form:

| | |
|---|---|
| A multi-layer or multi-scale decoder | DPT, UPerNet, FPN, PSP, and anything else that stacks layers behind the final convolution. The refusal lists the layers this model has no device graph for. |
| A convolution wider than 1x1 | A 3x3 final convolution is refused. Read as-is it would report nine times your real class count. |
| A head already folded into a single matrix | Save the layers, not their product. See the snippet below. |
| Your own head on the host instead | Not available. No flag keeps a custom head off the device, and no model assembles a host-side one for you. |

If your head is one of those, what you need is a model that compiles it, not a flag here.

> [!WARNING]
> **A depth head has a fixed output range.** The depth model reduces its bins against distances
> of 0.001 to 10 metres, the indoor range the published head was trained for. A head trained on
> another range has the same layer names and the same width, so it is accepted and compiled, and
> every distance it returns comes back linearly remapped onto that range. Nothing detects this.
> If your depth data is not indoor-scale, the numbers will be wrong and will look plausible.

**Your head is quantized with the backbone.** It is compiled into the same 8-bit graph, so your
head's accuracy is a property of the artifact rather than of the file you supplied. The fp
comparison `run` prints, on your own images, is the check available today.

### Save your head so the build can read it

The flag is the easy part. What has to be right is the FILE, and the shortest way to get it right
is to save the head as its own layers rather than as whatever your training script kept in
memory:

```python
import torch

# `head` is the trained decode head: the BatchNorm over the backbone's embedding dimension,
# and the 1x1 conv onto your classes. Save the parameters under the names the recipe reads.
torch.save(
    {
        "decode_head.bn.weight": head.bn.weight.detach().cpu(),
        "decode_head.bn.bias": head.bn.bias.detach().cpu(),
        "decode_head.bn.running_mean": head.bn.running_mean.detach().cpu(),
        "decode_head.bn.running_var": head.bn.running_var.detach().cpu(),
        "decode_head.conv_seg.weight": head.conv_seg.weight.detach().cpu(),  # (classes, dim, 1, 1)
        "decode_head.conv_seg.bias": head.conv_seg.bias.detach().cpu(),      # (classes,)
    },
    "my-head.pth",
)
```

A checkpoint saved by mmsegmentation already carries these names under `state_dict`, and the
loader reads that too, so an mmseg checkpoint needs no conversion.

Check the file before you spend a compile on it. This costs a second and reads back exactly what
the build will read:

```python
from model_axscript.vision.shared.seg_head import load_seg_head

head = load_seg_head("my-head.pth")
print(head.dim, head.num_classes)   # the backbone width it needs, and your class count
```

### A different number of classes needs nothing else

No flag, no field, no edit. The count is read from the shape of the final convolution, reaches
the compiler from there, and the artifact records it. A twelve-class head against the shipped
model's one hundred and fifty:

```bash
cat build/dinov2_seg_vits14_ade20k_linear/axscript-dddd8785/meta.json
```

```json
{
  "image_size": 518,
  "hidden": 384,
  "seg_head": true,
  "seg_num_classes": 12,
  "head_source": "/path/to/my-head.pth"
}
```

Run it and twelve is what comes back, named by index because the labels are yours:

```
my-photo.jpg: 76.8% of 1369 patches take the fp reference's label, mean patch cosine 0.9669
  predicted: class 0 42%, class 8 20%, class 10 11%, class 7 6%, class 1 6%
```

Same backbone, same input size, same calibration set, a different compiled graph.

Every other card refuses the flag and says where its head actually is:

```
$ python -m model_axscript.vision build vit_small_patch16_224-imagenet --head my-head.pth
--head: vit_small_patch16_224-imagenet compiles no head of its own, so there is nothing for
the flag to replace. A card that bakes a head into its device graph declares one in its
configuration, and this one declares none. A classification card carries its classifier inside
the checkpoint, so bring it with --weights, and a feature-extraction card has no head at all.
```

---

## Your own calibration data

The artifact runs in 8-bit integer arithmetic, and the ranges it uses are measured by running
a small set of real samples through the model at build time. That set is the calibration set.
It needs no labels.

`--calibration` REPLACES the card's set rather than joining it: a build that quietly kept a
share of the default corpus would report int8 scales nobody asked for. `--num-cal N` sets how
many samples are observed.

What matters is that the data looks like what you will deploy on. Calibrating out of domain
does not fail. It quietly costs accuracy, which is the worst way for it to go wrong.

**Vision** takes a directory of images or a Hugging Face dataset id. A directory is read
recursively and sorted, so two builds of the same directory measure the same ranges. Use the
same kind of scene as your deployment, at the same resolution or higher:

```bash
python -m model_axscript.vision build vit_small_patch16_224-imagenet \
    --weights /path/to/my-finetune \
    --calibration /path/to/my-images
```

**Text** takes a `.txt`, a `.md`, a directory of either, or a `.jsonl` with a `text` field per
line:

```bash
python -m model_axscript.llm build qwen3-0.6b --calibration ./my-corpus.jsonl --num-cal 256
python -m model_axscript.llm build qwen3-0.6b --calibration ./my-corpus-dir
```

### Vision-language calibration

A vision-language card refuses `--calibration` by name. Its two towers are calibrated
together, so a sample is an (image, question, answer) row and not a picture: the decoder's
scales have to see a real prompt attached to the embeddings the vision tower produced, and
pairing your images with one invented question would fit the scales to a prompt distribution
nobody runs.

Build the samples yourself and hand them to the builder, which is one call per model. That is
a Python API rather than a flag, and it is written up with the rest of the HuggingFace API in
[Serve an AxScript LLM through HuggingFace, or Your Own Front End](llm-huggingface.md#your-own-calibration-data).

---

## Where the artifact lands, and how to tell it is yours

A plain build goes to `build/<tag>/axscript`. Both families keep your build off the shipped
build, and they do it differently, because a vision override changes the compiled graph while
a text checkpoint is read back at run time:

| | Vision | Text and vision-language |
|---|---|---|
| Default directory | keyed to what you overrode: `build/<tag>/axscript-07cd8358` | `build/<tag>/axscript`, unkeyed |
| What records the inputs | `meta.json` | `metadata.json` |
| A `run` naming a different checkpoint | loads the artifact keyed to it, and builds it if absent | refused by name |

The vision suffix comes from the settings themselves, so it is not a value to memorise: change
the checkpoint, the head, the calibration set or the sample count and you get a different one.

```
build/vit_small_patch16_224-imagenet/axscript              the model as it ships
build/vit_small_patch16_224-imagenet/axscript-07cd8358     built from your own checkpoint
```

Read the recorded inputs to confirm you have your model and not the stock one:

```bash
cat build/vit_small_patch16_224-imagenet/axscript-*/meta.json
```

| Field | What it tells you |
|---|---|
| `repo_id` | The checkpoint the artifact was built from. Your path appears here when you passed `--weights`. |
| `head_source` | The head the artifact was built with, or `null` when the model has no head of its own. |
| `calibration_dataset` | The images the 8-bit ranges were measured on. |
| `n_calib` | How many of them were used. |
| `seg_num_classes` | For a model whose head runs on the device, how many classes it predicts. |
| `image_size` | The input size the artifact was compiled for. |

For a text or vision-language model the checkpoint is a property of the artifact rather than
of the command: a later `run`, `benchmark` or `eval` reads it back and tokenizes with it, so
you do not repeat `--weights` and cannot decode your model with our tokenizer. Naming a
different one against a built directory is refused rather than silently honoured, because
running the artifact that is there and labelling it with the checkpoint that was asked for is
how a number gets attributed to weights nobody measured:

```
build/qwen3-0.6b/axscript was built from Qwen/Qwen3-0.6B, not ./my-qwen3-finetune. Run
`build` to compile it from your checkpoint, or pass --output-dir to keep both.
```

`--output-dir` chooses the directory yourself, in either family, and turns the vision keying
off: a caller who named a directory asked for that one.

---

## Run and measure it

Point `run` at the same model and give it the input its family takes. For vision that is one
or more images, and the images come BEFORE the flags, because `run` takes any number of them
and a flag in between would swallow the rest of the line:

```bash
python -m model_axscript.vision run vit_small_patch16_224-imagenet my-photo.jpg \
    --weights /path/to/my-finetune
```

Pass the same build flags you built with. They tell the command which artifact to load, and
for a model with its own head they also tell it which head to compare against. The output
names each predicted class, and beside it how far the compiled model is from the same weights
running in floating point on the host, which is the only number that isolates what compiling
cost you:

```
my-photo.jpg: top-5 of 7 classes
  53.98%  class 5 (5)
  16.37%  class 1 (1)
  13.85%  class 3 (3)
   5.06%  class 0 (0)
   4.70%  class 4 (4)
  fp reference: top-1 agrees, logit cosine 0.9985
```

Classes are named by index rather than by a word, because your label set is yours: the SDK has
no file that maps your class 5 to a name.

For a text model the prompt is positional and the checkpoint is already recorded, so there is
nothing to repeat:

```bash
python -m model_axscript.llm run qwen3-0.6b "The capital of France is"
```

`benchmark` times the model, with the host-side decoding lifted out of the timed loop:

```bash
python -m model_axscript.vision benchmark vit_small_patch16_224-imagenet my-photo.jpg \
    --weights /path/to/my-finetune
```

```
{'ms': 8.29615248949267, 'fps': 120.53780367062087}
```

Add `--device` to either verb to choose which Metis device to run on. It does not recompile.

Scoring differs by family, because the metric belongs to whoever owns the task. A text model
has an `eval` verb (`python -m model_axscript.llm eval qwen3-0.6b --tasks wikitext`, which
needs `axelera-zoo[llm-eval]`). A vision card does not: score it with
`axzoo eval vit_small_patch16_224-imagenet --backend axscript`.

---

## See also

* [Compile and Run an AxScript Model](quickstart.md) is the step before this one: build
  a card as it ships, run it, and read what it prints.
* [Serve an AxScript LLM through HuggingFace, or Your Own Front End](llm-huggingface.md)
  loads a built text or vision-language artifact with the Auto classes, and carries the Python
  calibration API a vision-language card needs.
* [Deploy Custom Weights](../custom-weights.md) is the same goal on the YAML pipeline path, with
  `deploy.py` rather than AxScript.
* `model_axscript/README.md` is the reference for the tree itself: what each command does, and
  how to author or adapt a card.

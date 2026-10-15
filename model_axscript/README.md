# AxScript models

Recipes for models compiled on the **AxScript** path, which drives the ALTO backend
directly instead of going through the graph compiler. Each model here is a folder of code: the
compile is a program you read, run and change, not a pipeline you configure. That is the point,
because the interesting knobs are the ones you reach by editing the recipe.

```bash
python -m model_axscript.vision list                       # every vision model, and where it lives
python -m model_axscript.vision build dinov2-vits14        # compile it
python -m model_axscript.vision run   dinov2-vits14 image.jpg   # run it on a Metis device
python -m model_axscript.llm    run   qwen3-0.6b "The capital of France is"
```

Nothing here needs the zoo. Building, running and benchmarking need the compiler, the runtime
and the checkpoint the model names. Scoring accuracy (`eval`) additionally needs the shared
accuracy driver, `pip install axelera-zoo[llm-eval]`, so a number taken here is the number the
published figures were taken with; the command says so if it is missing.

The code in this directory is licensed under the Apache License 2.0 (`LICENSE` and `NOTICE`
here), separately from the rest of the SDK. That covers the recipes, not what they build.
Every checkpoint a recipe names is a third-party model under its publisher's own licence, and
accepting that licence is the user's responsibility, not the recipe's. Building and compiling do
not change it: an artifact built here is a derivative of the upstream weights and inherits their
terms, including any notice that must travel with copies. `licenses/Credits.txt` at the release
root lists every upstream model, its publisher and its licence, and names the file under
`licenses/` that carries the full text where a licence asks to be shipped verbatim. A model
folder's README states anything that goes beyond attribution for that one model.

Read this in order if you are new: what happens to a model (below), then how to point it at
your own weights and data, then where the code lives.

## 1. What happens to a model

Three stages, and each one is a real function you can call.

### The input is weights, not a graph

An AxScript recipe imports no graph. It builds the model in torch from a `state_dict`, then
authors the device graph itself, so its input is a checkpoint plus the code in this tree.

```
  a card declares ONE source                     what the recipe does with it
  ------------------------------------------     ------------------------------------------
  HFSource("facebook/dinov2-small")          ->  load_reference()   torch nn.Module, fp32
  TimmViTSource("timm/vit_small_patch16...") ->  block_config()     width, depth, heads, patch
                                                 load_weights()     state_dict -> our block
                                                 encode()           the fp output to compare
```

The seam is `EncoderSource` (`vision/shared/source.py`): seven methods, two implementations
shipped.

| implementation | upstream | how the weights arrive |
|---|---|---|
| `vision/shared/hf.py::HFSource` | HuggingFace `transformers` | `from_pretrained(repo_id)` |
| `vision/shared/timm_vit.py::TimmViTSource` | `timm` | `timm.create_model(name, pretrained=True)` |

Both download from the upstream hub on the first build and cache it, so nothing in the zoo's
asset bucket is involved. That is also why `axzoo export` plays no part in an AxScript build:
it produces the `.pt2` a codegen backend ingests, and `axzoo export <card> --backend axscript`
refuses by name for that reason. On a card that also has a codegen backend, two commands let
you see and take over how that source is produced:

```bash
axzoo export <card> --format pt2   # produce the source artifact the zoo publishes
axzoo fork   <card> --export       # an editable copy of the script that produced it
```

What the recipe accepts, and what it does not:

| input | supported | why |
|---|---|---|
| HF or timm checkpoint | yes | the two shipped sources instantiate it and hand over its `state_dict` |
| your own `state_dict` (`.pt`, `.pth`) | write a source | instantiate the architecture we implement, load the tensors, derive the shapes |
| `.pt2` (`torch.export`) | write a source | `torch.export.load(p).module()` for `encode`, `.state_dict()` for the weights |
| `.onnx` | no | there is no torch module to fake-quantize; it would need onnx2torch first |
| a pickled whole module | no | every `torch.load` here pins `weights_only=True`, enforced by an authoring gate |
| a checkpoint of a DIFFERENT architecture | no | see below |

The last row is the real boundary. The graph is never read, so what a source does is map a
`state_dict` into OUR block: pre-norm ViT, GELU MLP, separate q/k/v projections, LayerScale
optional. A checkpoint that deviates is refused by name rather than silently dropped, which is
what `timm_vit.py::_UNSUPPORTED` does for QK-norm (`q_norm`, `k_norm`), a gated MLP (`fc1_g`,
`w12`) and LayerScale on a plain ViT. Those weights have nowhere to go, and dropping them looks
exactly like quantization loss three steps later. A modified architecture therefore needs a new
reference block and a new kernel, not a new source. Fine-tunes of an architecture we implement
need neither: change `repo_id` and rebuild.

### Quantization and compilation (`build`)

One command runs the whole chain, and every step below is a real function you can call:

```
python -m model_axscript.vision build <tag>
  |
  |-- vision/shared/imported_dino.py::load_calibrated(config)
  |     _source(config)                     the card's EncoderSource
  |     source.load_reference()             upstream fp32 torch module
  |     source.block_config(...)            width, depth, heads, patch, seq_len, READ off it
  |     calibration_pixels(config, source)  real images (COCO by default), the card's own
  |                                         preprocessing, `n_calib` of them
  |     llm/models/dino/reference.py::DinoV2Model
  |                                         our torch model of the same block, with observers
  |     source.load_weights(model, ...)     upstream state_dict -> that model
  |     check_reference_against_upstream    our fp output must match the upstream's
  |     llm/quant.py::set_calibration_method(model, "minmax" | "percentile", percentile=)
  |     llm/quant.py::quantize_model(model, calibration, groups=smoothquant_groups(model),
  |                                  smooth_alpha=)   observe int8 scales, migrate activation
  |                                                   outliers into the weights
  |     llm/quant.py::enable_w8a8(model)    switch the observed model to W8A8 fake quant
  |     llm/quant.py::record_lut_ranges(model, calibration)
  |                                         per-site ranges for the activation LUTs
  |     -> (reference, model, block_config, seq_real)
  |
  |-- llm/models/dino/model.py::build_encoder_graph(reference, model, block_config, out_dir,
  |                                                cores=, final_norm=, seg_head_w=, ...)
  |     embed  -> N x block -> final_norm, ONE graph, one dispatch
  |     kernels are AxScript, in llm/models/dino/optimized_impl.py
  |     compile_block(config, out_dir, target=Target.DEVICE, cores=N)
  |       -> axelera.compiler.alto.compiler.driver.compile      the ALTO compiler
  |     llm/graph_descriptor.py::author_graph_model(...)
  |       -> axelera.codegen.axmodel_graph._engine.render_graph_model
  |     the block kernel is compiled ONCE and invoked per layer, each invocation bound to its
  |     own outlined weights, so depth costs weights and not code
  |
  |-- shared/artifact_contract.py::validate_run_size / validate_artifact
  |-- shared/runtime.py::probe_runtime_load     the artifact is loaded once before it is kept
  |
  v
build/<tag>/axscript/     model.json, meta.json, pipeline.json, host_params.npz,
                          dinov2.elf, per-kind AI-core ELFs, outlined weight blobs
```

The checkpoint is fp32 and stays fp32 on the way in. Nothing loads a pre-quantized model: int8
is produced here, by observing our own torch copy of the model while it runs the calibration
images. Before any observation, `check_reference_against_upstream` compares our copy's fp output
against the upstream's, so a wrong weight mapping fails there instead of surfacing later as
quantization loss.

The calibration set is a card declaration, not a flag. `EncoderConfig.calibration_dataset`
defaults to `sayakpaul/coco-30-val-2014` and `n_calib` to 32 images, and those images are
preprocessed by the SOURCE's own processor, so the scales are observed on the normalization the
checkpoint was trained with. `quantize_model` runs the model over them to fill the observers,
`smooth_alpha` (0.5 by default) migrates activation outliers into the weights over
`smoothquant_groups(model)`, and `record_lut_ranges` takes a second pass for the activation LUT
ranges. `calib_method="percentile"` with `percentile=99.99` replaces min/max on the affine
observers, which is the first thing to try when a card's int8 gap is larger than its siblings'.

The artifact is compile OUTPUT and is built where you run the command. It is not downloaded
and it is not served from an asset bucket: what ships is this code plus the published input it
reads, which is what makes the recipe adaptable. Point `--weights` at a fine-tune, or
`--calibration` at your own images, and the same chain compiles your model without an edit.

Knobs live on one dataclass, `vision/shared/config.py::EncoderConfig`: `image_size`, `shard`
(AI cores one inference is split across), `n_calib`, `calibration_dataset`, `calib_method`,
`percentile`, `smooth_alpha`, `final_norm`, `seg_head`, `cal_device`. A card's
`make_config` returns the ONE configuration its published numbers were measured at.

### Run

```
python -m model_axscript.vision run <tag> <image>
  |
  |-- <card>.load_device(config)      -> vision/shared/imported_dino.py::load_device
  |     shared/runtime.py::RuntimeModel(model_json, num_sub_devices=required_cores(spec))
  |       axelera.runtime Context -> device_connect -> load_model -> load_model_instance
  |-- <card>.prepare(image, config)   the upstream's own resize / crop / normalize
  |-- <card>.infer(device, prepared)  RuntimeModel.run("encode", inputs) -> one dispatch
  v
task output: features, ImageNet logits, or per-token class logits, per the card
```

`RuntimeModel.run` wraps each dispatch in `device_job`, which turns a wedged part into an
instruction rather than a bare "Failed to execute command lists". The artifact records the
core count it was compiled at and the loader reserves exactly that, so a run never picks a
topology the build did not use.

## 2. Adapt it to your model

In order of how much work each case is.

### Same architecture, your own weights

If you trained one of the architectures above, your weights are a `state_dict` and the shipped
source already knows the rest, so you override one method:

```python
# my_card/source.py
from dataclasses import dataclass

import torch

from model_axscript.vision.shared.timm_vit import TimmViTSource


@dataclass(frozen=True)
class MyCheckpointSource(TimmViTSource):
    """The shipped timm source, with the weights coming from a local file."""

    checkpoint: str = ""

    def load_reference(self):
        import timm

        model = timm.create_model(self._model_name(), pretrained=False)
        state = torch.load(self.checkpoint, weights_only=True)
        state = state.get("state_dict", state.get("model", state))  # bare dict or a wrapper
        model.load_state_dict(state)
        return model.eval()
```

```python
# my_card/__init__.py
def make_config(artifact_dir, cal_device=""):
    from ...shared.config import EncoderConfig

    from .source import MyCheckpointSource

    repo = "timm/vit_small_patch16_224.augreg_in21k_ft_in1k"   # the ARCHITECTURE, not the weights
    return EncoderConfig(
        implementation="vit-metis",
        repo_id=repo,
        source=MyCheckpointSource(repo, checkpoint="/path/to/my_vit_s16.pt"),
        image_size=224,
        artifact_dir=Path(artifact_dir),
        cal_device=cal_device,
    )
```

`repo_id` still names the timm architecture, because that is what tells `create_model` which
model to build and what the shapes are read off; only the weights come from your file. Nothing
downloads. Everything else, the preprocessing, the shape derivation, the weight mapping, the fp
reference for the fidelity check, is inherited. Verified end to end on ViT-S/16: `seq_real` 197,
hidden 384, 6 heads, patch 16, weights mapped, classifier `(1000, 384)`.

If your fine-tune is ON the HuggingFace hub, no subclass is needed at all: point the shipped
source at your repo, in the same `make_config`, and change the calibration set while you are
there.

```python
def make_config(artifact_dir, cal_device=""):
    from ...shared.config import EncoderConfig
    from ...shared.hf import HFSource

    repo = "my-org/dinov2-small-my-domain"       # your fine-tune, not ours
    return EncoderConfig(
        implementation="dinov2-metis",
        repo_id=repo,
        source=HFSource(repo),                   # TimmViTSource(repo) for a timm ViT
        calibration_dataset="my-org/my-domain-images",   # observe int8 scales on YOUR data
        n_calib=64,
        calib_method="percentile",               # "minmax" is the default
        percentile=99.99,
        image_size=224,
        artifact_dir=Path(artifact_dir),
        cal_device=cal_device,
    )
```

Match the calibration data to your domain: the int8 scales and the activation LUT ranges are
fitted from whatever you supply, so a medical or thermal model calibrated on COCO gives away
accuracy for no reason. `calibration_dataset` takes either a HuggingFace dataset id or a
**local directory**, read recursively and sorted so two builds of the same folder observe the
same scales:

```python
calibration_dataset="/data/my-domain/calib",   # jpg, png, bmp, webp, ppm, tif
n_calib=64,
```

The LLM side works the same way; see "A different calibration set" below.

For a local DINOv2 checkpoint, subclass `hf.py::HFSource` the same way as above and load with
`Dinov2Model.from_pretrained("/path/to/my/checkpoint")`, which takes a directory.

A `.pt2` takes more than an override. `torch.export.load(path)` gives you an
`ExportedProgram`: `.module()` is callable, which answers `encode`, and `.state_dict()` holds
the tensors, which answers `load_weights`. What it does NOT give you is `ref.blocks[0].attn` or
`ref.patch_embed`, so `block_config`, `seq_len` and `num_layers` cannot be inherited: derive
them from the tensor shapes instead (`blocks.<n>.attn.qkv.weight` gives depth and width,
`patch_embed.proj.weight` gives the patch size and channels). That is a real implementation of
four methods rather than one, which is why the shipped cards take a checkpoint and not a `.pt2`.

### A different calibration set

Both families take a local corpus, because domain data lives on disk. Vision:
`calibration_dataset="/data/my-domain/calib"`, a directory read recursively and sorted, as
above. LLM: calibration sources are `(name, count)` pairs, where a name is one of `wikitext`,
`gsm8k`, `humaneval`, `alpaca`, `mmlu`, or a path to your own corpus: a `.txt` / `.md` file, a
directory of them, or a `.jsonl` with a `text` field per line. A local corpus is cut into raw
windows exactly as WikiText is. A name that is neither a known source nor an existing path is
an error, because skipping it would leave the model calibrated on less data than you asked for.
From the command line, `--calibration <path>` does the same for one build (part 4).

### A different architecture

Say a `state_dict` arrives with QK-norm, a SwiGLU MLP, a different residual structure, or a
window-attention backbone. No source can rescue that, because there is nowhere in our block to
put those tensors, and the mapping refuses them by name rather than dropping them. What that
model needs is its own port: a torch reference block with fake-quant observers, and an
AxScript kernel for it. Both exist here to copy, `llm/models/dino/reference.py` and
`llm/models/dino/optimized_impl.py`, and `_template/` is the package skeleton. It is a
bounded job rather than a research one, roughly a few hundred lines for a transformer block
whose ops are already in `llm/ops/`, and it is well suited to an AI agent working from those
two files plus the AxScript language guide the compiler wheel ships. What you cannot do is
route the new architecture through the DINOv2 block and hope: the refusal you get instead is
the honest answer.

Whichever case you are in, `build_artifact` validates the compile output against the
runtime-loadable contract (`shared/artifact_contract.py`), so a recipe that emits a malformed
artifact fails at build time rather than at someone else's first load. `_template/` is the
package skeleton, and `_template/README.md` walks through the files a new model needs.

## 3. Where the code lives


The two families are laid out differently, and the difference is real rather than tidy.

```
model_axscript/
  README.md                        you are here
  registry.py                      which model a tag names, and where its code lives
  __main__.py                      says where each family's command line went, and exits
  <family>/__main__.py             that family's command line, `python -m model_axscript.<family>`
  shared/cli.py                   the flags and the rules both doors share, declared once
  _template/                       copy this to start your own model
  shared/                         used by both families: the artifact contract, the
                                   runtime wrapper, checkpoint helpers, the compiler probe

  llm/                             FLAT: one model per directory, because LLM is one task
    qwen3_0_6b/  llama_3_2_1b/     a model: declarations, its kernels if it has any, README
    lfm2_5_vl_450m/                a VISION-LANGUAGE model: two towers, so it declares a
    smolvlm2_256m_video_instruct/  VlmConfig rather than a GenerationConfig, and answers
    qwen3_vl_2b_instruct/          the same four verbs through vl_entry
    shared/entry.py               the LLM driver: what each verb does for this family
    shared/vl_entry.py            the vision-language driver, and the config a VLM
    shared/vl_config.py           declares; a VLM owns no entry script either
    models/<arch>/                 ARCHITECTURES, not models: qwen3, llama, lfm2, domyn, dino,
                                   and the VL towers lfm2vl, smolvlm2, qwen3vl.
                                   `dino` is here and every VISION card authors from it
    ops/  quant.py  export.py      the ALTO op library and toolkit
    layout.py  luts.py  modules.py
    graph_descriptor.py            the KindContract graph engine both families author with

  vision/                          GROUPED BY TASK, because vision has several
    classification/<model>/        vit_small_patch16_224_imagenet, ...
    semantic_segmentation/<model>/ dinov2_seg_vits14_ade20k_linear, ...
    feature_extraction/<model>/    dinov2_vits14
    depth_estimation/<model>/      depth_anything_v2_small, dinov2_depth_vits14_nyu_linear
    shared/entry.py               the vision driver
    shared/imagenet_vit.py        one recipe, three ImageNet ViT cards
    shared/ade20k_seg.py          one recipe, three segmentation cards
    shared/imported_dino.py       how a vision card reaches the toolkit under `llm/`
```

Two things about this are worth saying plainly rather than defending. `llm/` is not just the
LLM family: it is where the ALTO toolkit lives, so its root holds
`quant.py`, `export.py`, `layout.py`, `luts.py`, `graph_descriptor.py` and `ops/`, which BOTH
families compile through. And `llm/models/` holds architectures rather than models, which is
why `llm/models/dino/` is the reference block and kernel every vision card uses. Reading
`llm/` as "the LLM family" therefore gets you lost; read it as "the toolkit, plus the LLM
family's own models".

The nesting depth differs for the same reason a directory listing does: `vision/` groups by
the zoo's task axis, so `vision/semantic_segmentation/<model>/` lines up with
`model_recipes/<task>/`, while `llm/` stays flat because a task level there would read
`llm/llm/`. A tag never encodes the depth, and `registry.py`
resolves either, so regrouping renames nothing.

**One door per FAMILY, one command line behind them.** `python -m model_axscript.<family>
<verb> <tag>` is the only command line a model has: no model package declares a parser of its
own. The door is per family because the same word means different things to different
families, and a user who has not typed a tag yet should read only the flags that can apply to
their model: `build` reaches a different compiler for a picture model than for a decoder, and
`run <tag> cat.jpg` binds a photograph for one family and a sentence for the other.

The doors share `shared/cli.py` and declare no flag of their own, so the flags are declared
once and one cannot exist on six models and be missing from the seventh, and a verb or flag a
family does not have is refused by name instead of accepted and dropped. A tag typed at the
wrong door is answered with the door that has it. `axzoo build --backend axscript` shells the
door the card's `model_dir` names, so the zoo path and the customer path compile identically.

A family's driver declares three things and the shared command line enforces all three: its
verbs, the flags it honours, and its INPUT SHAPE. The shape is what lets the families disagree about their
positionals without a driver guessing at them. Vision declares `{"run": ("image+",),
"benchmark": ("image",)}` and the LLM family declares an optional prompt. Before the shape
existed the positionals were a bag of strings that each driver reinterpreted, so `build <tag>
cat.jpg` accepted the image and threw it away.

The vision-language family takes both, and takes its media BY FLAG rather than by shape: it
declares `{"run": ("prompt*",)}` and reads the picture off `--image`, the clip off `--video`.
The prompt is already the positional, so `run <tag> what is this` would bind `what` as the
picture: a shape of `("image", "prompt*")` can only be typed one way round, and the way a
user types a question is prompt-first. The flags are spelled as the model's own run CLI
spells them, so nothing is translated or sniffed on the way down.

Developing a model needs a configuration the front door deliberately does not offer, and reaches it by
module rather than through a second door: part 4 lists them.

The artifact is never downloaded, at any status. It is compiled output, so both paths compile
it, and re-running a build reuses what is already there. What a release publishes is the
recipe and the checkpoint it reads, not the compiled result.

**`__init__.py` is declarative, on purpose.** It names `MODEL_TAG` and `make_config`,
and `registry.py` reads that tag by parsing the file, never by importing the package.
An import pulls in torch and the compiler, and listing the models must cost nothing.
So a model's declarations stay where they can be read statically; anything that needs
torch is imported inside the function that needs it.

**There is one toolkit.** Every model authors through `export`, `luts`, `quant`, `layout`,
`modules` and `graph_descriptor` under `llm/`. The root `shared/` holds only what both families
share and the toolkit does not own: the artifact contract, the runtime, the checkpoint helpers
and the compiler probe.

### Toolkit, driver, model

Three tiers, so nothing forks:

- **The toolkit** (`llm/`, plus `shared/`): the AxScript
  machinery every model shares regardless of task - the `KindContract` graph engine, the
  ALTO `ops/` library, quant/layout/luts/export/modules. `KindContract` is what lets one
  engine author both a stateful decoder LLM and a stateless vision encoder, so there is no
  reason to fork it per task, and the vision family reaches it through
  `vision/shared/imported_dino.py`. `shared/` holds the pieces the toolkit does not
  own: the artifact contract, the runtime, the checkpoint helpers, the compiler probe.
- **The driver** (`<family>/shared/`, one per family): the build/run *shape* that a
  family genuinely shares but differs across families - LLM is stateful
  prefill+decode with a KV cache (`GenerationConfig`/`DecoderLayerImplementation`);
  vision is a single stateless `encode` (`EncoderConfig`/kNN). This split is real
  difference, not drift.
- **Model-specific** code (registration, configuration, the seams the driver calls,
  README) goes in `<family>/<model>/` or `<family>/<task>/<model>/`. This is the
  reusable recipe. A card that reuses an existing architecture is a
  configuration and nothing else; one that needs a new device recipe carries its torch
  reference and its ALTO kernel here too.

When several cards share one recipe, the seams live once in a per-recipe module beside the
driver and each card package is a `make_config` and a re-export. `vision/shared/ade20k_seg.py`
serves the three ADE20K segmentation cards, and `vision/shared/imagenet_vit.py` serves the
three plain-ViT ImageNet classifiers, which differ only in the checkpoint their configuration
names: width, depth, head count and patch size are read off the checkpoint, so a fourth size is
a new package of about fifty lines and no new code.

Adding a model must not fork the shared tiers: give the toolkit or driver a hook
with a no-op default instead of a per-model branch.

Model code imports `shared` + `<family>/shared` plus the compiler packages, and the dependency
runs one way only: nothing outside `model_axscript/` imports a model, so discovering a model costs
no compiler import. A single-graph model with no family needs no `shared`: it is just a
`<family>/<model>/` folder like `_template/`.

See [`llm/lfm2_5_350m/`](llm/lfm2_5_350m/) for the worked LLM example and
[`vision/feature_extraction/dinov2_vits14/README.md`](vision/feature_extraction/dinov2_vits14/README.md)
for the worked vision (encoder) example.

### I want to change X, which file?

A high-level index for "I want to change X, which file?". The AxScript *language*
itself and the *supported-op list* are COMPILER-owned (see the Language Guide below),
not this tree.

The first two rows are the common case, and both are edits to YOUR model's own
`__init__.py`, e.g. `vision/classification/vit_base_patch16_224_imagenet/__init__.py`. The
rest name the implementation behind a seam, for when you are changing how it works rather
than which inputs it takes.

| I want to... | Where I edit it |
|---|---|
| swap the upstream weight source | `make_config` in your model's `__init__.py`: `repo_id=` and `source=TimmViTSource(...)` or `HFSource(...)`. The zoo card declares the same id (`source=UpstreamCheckpoint(...)`), so change both. A different upstream entirely means implementing the seam in `vision/shared/source.py`, which is seven methods |
| change calibration (data and observers) | `make_config` in your model's `__init__.py`: `calibration_dataset=`, `n_calib=`, `calib_method=`, `percentile=`, `smooth_alpha=`. The implementations are `vision/shared/calibration.py` for the image set and `llm/quant.py` for the observers and SmoothQuant |
| add another size of a recipe that exists | a new model package with `MODEL_TAG`, `REPO_ID` and `make_config`, re-exporting the seams from the recipe's shared module (`vision/shared/ade20k_seg.py`, `vision/shared/imagenet_vit.py`) |
| change a hosted output head | `vision/shared/imagenet_head.py` (ImageNet classifier) / `vision/shared/seg_head.py` (ADE20K linear head, baked into the device graph) |
| change quantization / scales / LUTs | `llm/quant.py` (W8A8 fake-quant), `llm/export.py` (export scales), `llm/luts.py` (per-site LUT ranges) |
| author the device graph / kernels | `llm/graph_descriptor.py` (`KindContract` engine) + `llm/ops/` + per-model `optimized_impl.py` / `naive_impl.py` |
| map params onto kernel buffers | `llm/layout.py` |
| compile / build the artifact | `vision/shared/imported_dino.py::build_artifact` / `llm/generate.py` |
| check the artifact contract | `shared/artifact_contract.py` |
| run on device / runtime plumbing | `shared/runtime.py` (`RuntimeModel`, device select, load probe) |
| measure performance (latency / fps) | `vision/shared/build.py::timeit` behind the `benchmark` verb; LLM `generate.benchmark_generation` |
| measure accuracy | vision: `axzoo eval <card> --backend axscript`, through the zoo's dataset + evaluator registries; LLM perplexity in `llm/evaluation/` |
| the AxScript DSL syntax / supported ops | COMPILER-owned: the Language Guide + the AIPU partition table (not here) |

### The AxScript language guide

The kernels (`optimized_impl.py` / `naive_impl.py` / `llm/ops/`) are written in AxScript,
the ALTO device DSL. Its canonical, low-level reference ships with
the compiler wheel at `axelera/compiler/alto/docs/axelerascript_guide.md`. Refer to it for language syntax and the supported-op surface; it is
updated upstream, so link to it and never copy it here.

## 4. Every command

**The family doors, by tag.** The command lines this tree owns, one per family: `vision` for
the picture models, `llm` for the text models and the vision-language ones, which share a
decoder. One flag surface behind both, so a flag cannot exist on six models and be missing
from the seventh, and a verb, a flag or an input a family does not have is refused by name
rather than accepted and dropped. `--help` at a door lists the flags that family honours, and
listing them again for one tag narrows to that card.

```bash
python -m model_axscript.vision list                                 # every vision tag
python -m model_axscript.vision build     <tag>                      # compile the artifact
python -m model_axscript.vision run       <tag> <image>              # build if needed, then run
python -m model_axscript.vision benchmark <tag> <image>              # + fps / latency
python -m model_axscript.vision run       <tag> <image> --device <dev>   # a device, no recompile
python -m model_axscript.vision build     <tag> --weights <dir-or-id>    # your own checkpoint
python -m model_axscript.vision build     <tag> --calibration <path>     # your own calibration data
python -m model_axscript.vision build     <tag> --head <path>            # your own head, where a card compiles one
python -m model_axscript.vision --help                               # the flags a card takes

python -m model_axscript.llm list                                    # every text and VLM tag
python -m model_axscript.llm build     <tag>                         # compile the artifact
python -m model_axscript.llm run       <tag> "<prompt>"              # build if needed, then run
python -m model_axscript.llm benchmark <tag>                         # + tokens/s, latency
python -m model_axscript.llm run       <tag> --interactive           # multi-turn, text models
python -m model_axscript.llm run       <tag> "what is this?" --image photo.jpg   # a VLM
python -m model_axscript.llm run       <tag> "describe it" --video clip.mp4 --frames 8
python -m model_axscript.llm build     <tag> --generation EUROPA --num-cores 8   # which part
python -m model_axscript.llm build     <tag> --weights <dir-or-id>    # your own checkpoint
python -m model_axscript.llm build     <tag> --calibration <path>     # your own calibration data
python -m model_axscript.llm eval      <tag> --tasks wikitext        # score it, text
python -m model_axscript.llm run       <tag> --help                  # this card's flags and input
```

A vision card has no `eval` verb, which is why the vision door does not offer one; score it
with `axzoo eval <card> --backend axscript` instead (the "measure accuracy" row above).

`python -m model_axscript` with no family is not a command line. It prints the door each
family uses and exits non-zero, so an old command teaches the new one rather than continuing
to work in silence.

**Building from your own weights, by family.** Three flags, declared once in `shared/cli.py`
so a flag cannot exist on six models and be missing from the seventh, and refused BY NAME at a
door that does not honour one:

| Flag | Vision | Text | Vision-language |
|---|---|---|---|
| `--weights` a hub id or a local checkpoint directory | yes | yes | yes, both towers |
| `--head` your own output head | a card that compiles one (segmentation, depth) | no head to replace | no head to replace |
| `--calibration` your own images or corpus | yes | yes | no: two towers calibrated together |

All three are BUILD parameters, so `eval`, which never builds, refuses them by name. A
vision-language card's calibration sample is an (image, question, answer) row rather than a
picture, so its door is the builder itself:
`multimodal_calibration(processor, config, images=..., prompts=...)` into
`pipeline.build_artifacts(calibration=...)`, per model.

Two rules the tree enforces rather than documents. `--calibration` REPLACES the card's set
rather than joining it, because a build that quietly kept a share of the default corpus would
report int8 scales nobody asked for. And the checkpoint is a property of the ARTIFACT, not of
the command: vision keys the default artifact directory to everything that changes what is
compiled, so a build from your own weights cannot overwrite the shipped one; text and vision-language
record it in `metadata.json` and refuse a later `--weights` that names something else, because
running the artifact that is there and labelling it with the checkpoint that was asked for is
how a number gets attributed to weights nobody measured.

The tutorials are three, in the order a reader meets them, and this file does not
repeat them: [`docs/tutorials/axscript/quickstart.md`](../docs/tutorials/axscript/quickstart.md) builds and runs a card as it ships,
[`docs/tutorials/axscript/llm-huggingface.md`](../docs/tutorials/axscript/llm-huggingface.md) serves a built text or VLM artifact, and
[`docs/tutorials/axscript/custom-weights.md`](../docs/tutorials/axscript/custom-weights.md) is the walkthrough for all three flags across the
families. The last carries what each flag takes (a HuggingFace checkpoint LAYOUT, not a bare
`state_dict`), how to save a head file the build can read, and how to read back what a build was
given.

**Building and running have different dependencies.** Build and eval need PyTorch and
`transformers` unavoidably: the reference model, the calibration observers and SmoothQuant are
torch, and both eval legs are torch modules. Running a built artifact needs `numpy` and
`axelera.runtime`; torch is an accelerator there rather than a requirement (the host lm_head
takes its fp16 CPU gemv when present, about 14 ms per step against about 29 for the fp32
alternative, and falls back to numpy when absent), and `transformers` is needed only if you
drive the model through the HuggingFace API. A test holds that line by importing the run path
with both libraries blocked. One gap remains:
resolving a model by tag imports the torch reference models, so the command line above still
needs torch even to run.

**A text model also loads through the HuggingFace API.** A built LLM or VLM artifact is a
HuggingFace model directory, so after one `register_auto()` (which teaches the Auto classes our
model type, since a stock transformers cannot discover one another package defines)
`AutoModelForCausalLM.from_pretrained(<artifact-dir>)` and
`generate(..., top_p=..., min_p=...)` work, and the sampling, streaming and stopping are all
`transformers`' own (`llm/huggingface.py`). That adapter is ONE front end over an engine that
names none, so a second is added beside it rather than by changing the engine. Usage, what one
device KV cache refuses, and the four primitives another front end would call are in
[`docs/tutorials/axscript/llm-huggingface.md`](../docs/tutorials/axscript/llm-huggingface.md). A vision card has no such adapter: its output is a
tensor its own `run` reports, and HuggingFace reaches vision on the other side of the build, as
one place a card's weights come from. This file stays about authoring and adapting a card.

`<input>` is what the family declared, not a convention: `run dinov2-vits14 <image> [image
...]`, `benchmark dinov2-vits14 <image>` because a mean over different images measures the
images as much as the model, and an optional prompt for a text model, which falls back to the
family's default when omitted. Name the tag and `--help` narrows to that card, so a vision
user is not offered `--tasks`, `--engine`, `--seq-len` or `--interactive` to be refused by them
a moment later.

Calibration uses the GPU automatically; `--cal-device cpu` forces CPU and `--num-cal N`
overrides the sample count, whatever the calibration data is.

The verb sets differ by family and the front door enforces that:
an LLM honours all four verbs, including `eval`, whose regime is DECLARED so it takes
`--tasks`; vision honours only `build`, `run` and `benchmark` here, and `eval` is refused by
name because scoring a vision card is `axzoo eval`, not a verb on this front door. A card
ships ONE configuration, the input size or context length and the shard width its published
numbers were measured at, and the front door runs that one.

**Developer modules, for the text family.** Not a second designed surface. Each is a module
with its own parser, run directly, and it takes any configuration rather than the shipped one,
so a number from here is not the number a published figure was taken with. Vision has none:
score a vision card with `axzoo eval` (the "measure accuracy" row above).

```bash
python -m model_axscript.llm.build                 --help   # compile any configuration
python -m model_axscript.llm.evaluation.evaluate   --help   # lm-eval tasks on an artifact
python -m model_axscript.llm.evaluation.chat       --help   # interactive REPL on an artifact
python -m model_axscript.llm.evaluation.regression --help   # perplexity and perf of an artifact,
                                                            # recorded once and compared later
```

Everything above compiles through the same code, so the front door and a module cannot
disagree about what an artifact is; they disagree only about who chooses the configuration.
Quote a card's own numbers, not a developer module's.

Each model folder's README gives its exact commands, its checkpoint, and any prerequisite such
as a HuggingFace token for a gated one.

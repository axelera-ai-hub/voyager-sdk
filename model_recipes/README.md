# Model recipes

Model recipes are standalone Python scripts for selected Model Zoo models. They
show how to compile a model with either backend (the established `axrelay`
path, or the newer AxMO + graph-compiler `axgraph` path), and how to wire the
compiled artifact into a high-efficiency end-to-end pipeline in just a few
lines with Pipeline Builder.

A recipe never needs `axelera-zoo`. Everything the catalog knew about the
model, the compiler configuration, the weight URL and checksum, and the
calibration source, is baked into the file as a literal, so the recipe *is*
the resolved catalog entry. Once you have it, the catalog is out of the
picture:

| Command | Requires |
|---|---|
| `python yolov8n_coco_axrelay.py` | `axelera-compiler` |
| `python yolov8n_coco_axrelay.py --run <input>` | `axelera-runtime` |
| `python yolov8n_coco_axrelay.py --benchmark <input>` | `axelera-runtime` |
| `python yolov8n_coco_axrelay.py --eval` | `axelera-zoo[eval]` |

Only `--eval` needs the zoo, because the evaluators, the datasets and the
ground truth live there. Without it installed, `--eval` stops and names the
package rather than failing on an import; every other command is unaffected.
If you score the pipeline in your own eval harness, you never install it.
Downloading the weights the recipe names is a cached fetch with a checksum,
not a catalog lookup.

What a vision recipe does NOT repeat is the part that is the same in all of
them. The command line, the weight download and the accuracy legs live in
[`shared/`](shared/), which ships beside the recipes and travels with a fork.
So a recipe reads as this model: its preprocess, its pipelines, and the
compiler call that builds it.

The LLM recipes under [`llm/`](llm/) are the exception: a decoder's build,
prompt handling and eval have little in common with a vision pipeline, so each
carries its own command line and imports no `shared/`. Nothing below about
copying that directory applies to them, and `axzoo fork --script` writes none
beside one.

## Measure accuracy

`--eval` scores one leg and prints its metrics:

```
$ python detection/yolov8n_coco_axrelay.py --eval
Evaluating COCO (aipu): 5000 samples
yolov8n-coco COCO/val on aipu (preprocess: none): mAP_box=<value>, mAP50_box=<value>, ...
key metric: mAP_box
```

That is the compiled model on the device. On its own it cannot tell you whether
a low number came from the model, the dataset, the pipeline, or from
quantization and compilation. `--engine host` answers that: it runs the fp32
weight the build consumes, on this machine, through the same preprocess,
decode tail, dataset and evaluator.

```
$ python detection/yolov8n_coco_axrelay.py --eval --engine host
Evaluating COCO (host): 5000 samples
yolov8n-coco COCO/val on host (preprocess: none): mAP_box=<value>, ...
```

The difference between the two is what deployment cost, and nothing else. Each
run reports one engine, so run the verb twice for the pair; `--engine host`
needs no `.axm` and so no compile, which means you can take the reference
before you build. Add `--frames N` for a quick check, remembering that a split
stored in class order is truncated rather than sampled, and `--data-root DIR`
to point at your own copy of the dataset.

`preprocess:` on those lines is the second axis of the number. `--run` and
`--benchmark` fuse the preprocess by default (`--optimize auto`), which replaces
the declared resize with an OpenCL or SIMD kernel; `--eval` does not, because
that kernel moves pixels and can move the metric, so the reported figure is the
op sequence this file declares. To score what you deploy instead, name the
policy:

```
$ python detection/yolov8n_coco_axrelay.py --eval --optimize auto
yolov8n-coco COCO/val on aipu (preprocess: auto): mAP_box=<value>, ...
```

Both figures are valid and they are not the same measurement, which is why the
line says which one it is.

If you would rather not run the recipe at all, `axzoo eval` takes the same two
legs from the packaged config and reports them in the same shape:

```
$ axzoo eval yolov8n-coco --engine host
$ axzoo eval yolov8n-coco --engine aipu
```

Both front doors call the same evaluator on the same dataset through the same
pipeline, so their numbers are comparable with each other and with the ones in
the Model Zoo's published accuracy data.

The Model Zoo remains the source of truth for supported models, benchmark
numbers and accuracy data. Existing YAML model definitions remain supported.
Pipeline Builder is Beta in this release. Its concepts are all in place, and
compared with a YAML pipeline it is more modular, more flexible and easier to
scale. It already performs well on many models; some still have significant
room to improve, and that work is planned for the 2.0 release. If you already
have a production pipeline on an existing YAML Model Zoo model, there is no
need to move it: the YAML/GStreamer flow remains the most performant path for
those models today. If you need a model only available through `axgraph` (ViT
classifiers, DINOv2 segmentation), start on Pipeline Builder now: it is the
only runnable path for those models today.

Recipe coverage is expanding: every model currently released only as a YAML
definition is on track to get a recipe here too, so "selected" above will
eventually mean the full Model Zoo catalog.

## Choose a workflow

Most application developers can work directly from a recipe. Model Zoo partners,
distributors, FAEs, and support teams may also want to learn the `axzoo` config
system because it helps manage model catalogs, regenerate recipes, reproduce
customer builds, and maintain long-lived model variants.

### Copy and adapt a recipe

Use this path when an existing recipe is close to your model architecture or
pipeline.

1. Copy a vision recipe anywhere, for example
   [`detection/yolov8n_coco_axrelay.py`](detection/yolov8n_coco_axrelay.py),
   together with [`shared/`](shared/). The destination doesn't matter, but the
   two travel together: the recipe declares the model, and `shared/` holds what
   every recipe does the same way (the command line, the asset fetch, the
   accuracy legs). The few recipes that declare `PRE_COMPILE_PASSES` also read
   those rewrites from [`shared/passes/`](shared/passes/), so copy
   that directory alongside too. `axzoo fork --script` does all of it for you,
   and a recipe copied without `shared/` says so and names the directory.
2. Point `load_source()` at your exported `.onnx` or `.pt2` file.
3. Replace `CALIBRATION_DIR` with 100 to 400 images representative of your
   deployment scene, drawn from your training set or a held-out calibration/
   validation split.
4. Update `preprocess` so calibration and runtime inference use the same image
   transforms.
5. Adjust the compiler configuration and, when useful for checks, update
   `build_pipeline(axm_path)` for your model's post-processing and task type.
6. Run the recipe directly:

   ```bash
   python my_model_axrelay.py
   python my_model_axrelay.py --benchmark input.mp4
   ```

The `--run`/`--benchmark` path uses the recipe's Pipeline Builder graph. Treat
it as a correctness sanity check, not a reproduction of published Model Zoo
performance numbers (see above for the production-throughput picture, and the
[custom weights tutorial](../docs/tutorials/custom-weights.md) for an
existing `axrelay` production pipeline). The numbers `--benchmark` reports
are real, though: they show where Pipeline Builder currently stands, and
will become the published Model Zoo numbers once it reaches target
performance.

### Generate an editable recipe with `axzoo fork`

Use this path when you want a clean standalone script generated from the packaged
zoo configuration installed with `axelera-devkit`.

```bash
axzoo fork detection/yolov8n_coco --script
```

`axzoo fork --script` regenerates the recipe from the installed `axelera-zoo`
package and writes an editable copy into your current directory. This is useful
when you want the current template for your installed SDK version, need to choose
a backend or weight format, or do not want to copy from the source tree by hand.

### Use recipes as AI reference material

You can also ask an AI coding assistant to build a script for your model using
the released recipes as reference material. Point it at the closest recipe and
ask it to adapt the source loading, calibration, preprocessing, compiler config,
and optional `build_pipeline(axm_path)` check path for your model.

Review the generated script before running it. In particular, check that
calibration uses representative data, preprocessing matches runtime inference,
and the pipeline post-processing matches the model outputs.

## How to read a recipe

Each recipe follows the same broad structure:

| Section | What to look for |
| --- | --- |
| Provenance block | The packaged model, upstream framework, weight source, and license. |
| `preprocess` | The runtime image transforms; calibration uses this same sequence. |
| `build_pipeline(axm_path)` | The readable Pipeline Builder graph used for run/benchmark checks: preprocess, `op.load()`, post-processing, and result conversion. |
| `CALIBRATION_DIR` | The representative data used for quantization. Replace this for your deployment scene (`CALIBRATION_URL` is the packaged set it falls back to). |
| `load_source()` | Where the `.onnx` or `.pt2` source artifact comes from. |
| `POSTAMBLE` | The native kernel that runs the model's postamble, when one covers the model. See [Postamble](#postamble). |
| `DEVICE_CONFIG` | Board settings this model needs before it runs, when it needs any. See [Pipeline runtime settings](#pipeline-runtime-settings). Most recipes have no such section. |
| `PRE_COMPILE_PASSES` | Graph rewrites applied before compiling, when the model needs any: each name is a file in [`shared/passes/`](shared/passes/) working around a pattern the compiler does not yet read. Most recipes have no such section. [`shared/passes/README.md`](shared/passes/README.md) says how to tell whether your model needs one, and how to write your own. |
| Compiler calls | The `CompilerConfig` (or AxMO config, for `axgraph`) and the quantize/compile calls that build the `.axm`: exactly what `axzoo` runs for this model. |
| `build_eval_pipeline(model_op)` | The pipeline `--eval` scores. Same preprocess and model as above, different tail: presenting a result and scoring one are different jobs. |
| `EVAL_TARGET` | The task, dataset and split this model's accuracy is measured against. |
| `main()` | Hands the above to [`shared/cli.py`](shared/cli.py), which is where each verb is carried out. |
| `--run` / `--benchmark` | Optional development checks for the recipe's Pipeline Builder path. |
| `--eval` | Accuracy, on the device (`aipu`) or against the fp32 source (`host`). See **Measure accuracy** above. |

YAML models also carry per-model settings, but `deploy.py` auto-picks how to
apply them behind the scenes depending on the case. A recipe has no such
auto-picking: it's what `axzoo` actually runs to build that model, written
out as a plain script, so the `CompilerConfig` and quantize/compile calls you
see are exactly what executes, not a resolved-at-runtime abstraction.

`preprocess` is a plain Python callable: it receives one HWC uint8 image and
returns a tensor. Recipes default to `op.seq(...)` (Pipeline Builder) because
it lets one definition serve both calibration and the `--run`/`--benchmark`
pipeline, not because it's required. If you already have a preprocessing
function from your training code (for example a
`torchvision.transforms.Compose`), assign it to `preprocess` directly;
anything that isn't already a Pipeline Builder operator is auto-wrapped when
it's composed inside `build_pipeline`.

## Postamble

The compiler leaves the last host-side piece of a model, its postamble (for
example a detector's box decoding or a classifier's softmax), as a small graph
inside the `.axm`. By default `op.load` runs that graph on onnxruntime. A recipe
whose model is covered by a built-in native kernel names it once, as
`POSTAMBLE`, and passes it to `op.load` in both `build_pipeline` and the
accuracy leg, so the demo, the benchmark and `--eval` run the same tail:

```python
POSTAMBLE = Postamble(fused_kernel='UltralyticsYoloPostamble', num_classes=80, dfl_bins=16)
```

The kernel changes how the model's output tensors are computed, not what they
are, and it changes nothing about the build: delete `POSTAMBLE` and its
`using_postamble=` argument and the same `.axm` runs the onnxruntime tail. When
you adapt a recipe to a model with a different class count or head, update the
kernel's parameters to match, or remove it while you bring the model up.
[Postambles](../docs/reference/pipeline-builder/postamble.md) lists the built-in
kernels and their parameters, and shows how to write your own.

## Pipeline runtime settings

A few settings are neither part of the build nor part of the pipeline graph:
they are device settings the card has to be holding while the model runs.
A recipe that needs one declares it as `DEVICE_CONFIG` and hands it to
`op.load`, and Pipeline Builder writes it to the device before the first frame.

```python
DEVICE_CONFIG = DeviceConfig({'110m': BoardConfig(mvm_limitation=57)})


def build_pipeline(axm_path):
    return op.seq(
        preprocess,
        op.load(str(axm_path), using_postamble=POSTAMBLE, device_config=DEVICE_CONFIG),
        ...
    )
```

The key is the **board model**, the name `axdevice` prints for your card
(`110m`, `113m`, `111c`, `130p`, `150p`, `232p`, or `pcie` for the legacy
single-chip card), and not the form factor. That is the point rather than a
detail: an Axelera Embedded 110m and a 113m are both M.2 cards, and a setting
meant for one of them must not reach the other. A board the config does not
name is left exactly as it is, so one recipe runs unchanged everywhere, and a
name that is not a board model is rejected where you wrote it.

| Setting | Declared for | What it does |
| --- | --- | --- |
| `mvm_limitation` | `110m` | Caps MVM utilisation at that percentage (1 to 100). Some models draw more power at full utilisation than a 110m can dissipate. The compiled artifact is the same either way, so the limit belongs to the run and not to the build. |

**Declared for** is where a packaged model asks for the setting, not a claim
about which boards support it. The setting exists on other boards too, and
what a given board can sustain depends on your own enclosure, airflow and power
budget. If you want to cap a board the recipe says nothing about, or run a
different cap on the one it does, set `AXELERA_CONFIGURE_BOARD` (below) rather
than editing the recipe: it applies to whatever board is present.

A recipe that declares `DEVICE_CONFIG` explains, in the comment above it, what
each setting does and why the board is listed for it.

A setting applies to the AI cores running that model, not to the whole card, so
two models in one pipeline each keep their own and a model that declares
nothing runs at full utilisation beside one that does. This is the same rule
the YAML/GStreamer flow uses for its static core ranges.

It is not applied the same way, though, and the difference costs a little.
The YAML flow binds a core to one model for the whole run, so it writes each
setting once at startup and never again. Pipeline Builder shares cores between
models, so when a model moves to a core last used by a model wanting a
different value, the setting is written again: a few milliseconds, on the
frame that causes the move. A pipeline whose models all want the same value, or
none, pays that at most once per core. If you need the switching cost gone
entirely, give each model its own cores with `op.load(core_allocation=...)`.

Values are not written to the card's configuration file, and every core this
pipeline changed is put back when it finishes, so whatever you run next sees
the card as it was. A board whose firmware owns these settings is left alone
and says so.

### Overriding from the command line

`AXELERA_CONFIGURE_BOARD` decides this for the whole process, and what you set
there beats what the recipe asks for. It reads the same way it does on the
YAML/GStreamer path:

| Value | Effect |
| --- | --- |
| unset, or `1` | Apply what the model asked for. The default. |
| `0` | Leave the device alone. The model's request is dropped. |
| `[clock],[mvm]` | Apply your values instead, on whatever board is present. Either part may be empty: `,90` caps MVM at 90% and leaves the clock, `400` sets the core clock to 400 MHz and leaves MVM to the model. |

```bash
AXELERA_CONFIGURE_BOARD=,90 python detection/yolov8l_coco_axrelay.py --benchmark input.mp4
```

## Backend names

Recipes include their backend in the file name, for example `_axrelay.py` or
`_axgraph.py`. Start from the backend used by the recipe you are adapting
unless you have a specific reason to move the model to another compiler
path.

### `axrelay`

The quantize/compile path behind `deploy.py` and the existing YAML model
flow. Every recipe committed in this tree today uses it. `load_source()`
accepts either `.onnx` or `.pt2`, and a single `main()` takes the source
through calibration, quantize, and compile in one pass via
`axelera.compiler.quantize`/`compile`, for example
[`detection/yolov8n_coco_axrelay.py`](detection/yolov8n_coco_axrelay.py).

### `axgraph`

The AxMO + `axelera-graph-compiler` path: the newer compiler stack currently
used for ViT-family compilation (classifiers, and DINOv2 backbones).
`load_source()` accepts `.pt2` only, no ONNX today. No `axgraph` recipe is
committed here yet (this tree only ships a committed recipe for a
`ready`-lifecycle model); once one ships, `axzoo fork <model> --script`
regenerates it the same way it does for `axrelay` recipes today.

Compiler calls split into two functions instead of `axrelay`'s single
`main()`: `quantize()` (AxMO) and `compile_to_axm()` (`axelera.graph_compiler`),
so a build can stop after quantization and resume into compile later. The
shape below (subject to change until a recipe is committed here):

```python
def quantize():
    from axelera.model_optimizer import get_default_config, import_exported_program
    from axelera.model_optimizer.api import (
        finalize_optimized_model,
        prepare_model_for_optimization,
    )
    from axelera.model_optimizer.trainer.calibration import calibrate_model

    model = import_exported_program(torch.export.load(str(load_source())))
    axmo_cfg = get_default_config(...)  # model-specific AxMO settings
    prepare_model_for_optimization(model, axmo_cfg)
    calibrate_model(model, cal_loader)
    finalize_optimized_model(model, axmo_cfg)
    return model


def compile_to_axm(model):
    from axelera.graph_compiler.api import compile_single_graph
    from axelera.graph_compiler.config import CompilerConfig
    from axelera.graph_compiler.types import DType, Pipeline, TensorType

    input_types = (TensorType(shape=tuple(INPUT_SIZE), dtype=DType.FLOAT32),)
    compile_single_graph(model, CompilerConfig(pipeline=Pipeline.GENERIC, ...), input_types)
```

## PyTorch model, ONNX, or `.pt2`?

Prefer the PyTorch/`.pt2` path over ONNX even on `axrelay`, where both are
accepted: it carries more of the original model structure through to the
compiler and reaches a higher optimization level than an ONNX-converted
model. On `axgraph` it isn't a choice at all; `.pt2` is the only format it
ingests.

`.pt2` itself isn't really the point: it's a serialized
`torch.export.ExportedProgram`, the file form of "instantiate your model and
feed it in." If your model is already loaded in your SDK working venv, skip
the file entirely: call `torch.export.export(model, args=(...))` and pass the
resulting module straight into the compiler call, the same way a recipe's
`.pt2` branch does.

Where `.pt2` earns its keep is moving a model between environments: export it
under your training framework's own dependencies, then quantize/compile it in
the SDK venv, without either environment needing the other's packages.
`axzoo export <model> --format pt2` automates that hop: it runs the model's
exporter in its own throwaway venv (cached under `_export_venvs/`, one per
exporter/framework version), so your SDK venv never needs your training
framework's dependencies installed. Once you have the `.pt2` file, that
export venv has done its job; `rm -rf` it and go back to your SDK venv to
quantize and compile. There is no need to resolve the source framework's
dependencies in your working venv just to get one file out of it.

## Where `axzoo` fits

The `axzoo` command is installed with `axelera-devkit` through the
`axelera-zoo` package. You can work from recipes without it. It is useful to
find a model, see where it came from, and reproduce or adapt how its source
artifact was exported. `axzoo <command> --help` documents every option.

`axzoo` resolves a model by name; a recipe has already been resolved. That is
the whole difference, and it is why the recipe carries no zoo dependency:

- `axzoo build yolov8n-coco` looks the model up in the catalog, fetches the
  published weights and the calibration images, then compiles.
- `python yolov8n_coco_axrelay.py` compiles the same model from the URL and
  checksum already written into the script.

Accuracy evaluation is the exception on both paths, because the evaluators and
datasets are part of the zoo rather than of any one model.

| Command | What it gives you |
| --- | --- |
| `axzoo list [GROUP]` | The packaged models grouped by task (`detection`, `classification`, ...), each with the compiler backends it is released on. `--backend` filters by backend. |
| `axzoo info <model>` | The model card: display name, task, license, upstream (framework, model and version, or source repository and pinned ref), the recipe it generates, default backend, calibration source, and the upstream and expected accuracy. For a model that compiles from a hub checkpoint (the LLM and AxScript models) it names the checkpoint, for example `Source: huggingface Qwen/Qwen3-0.6B`. It also prints the model's `Postamble` exactly as its recipe declares it, with the file that defines the kernel (`none` when the onnxruntime tail runs). For a model whose source artifact is exported, it prints the `axzoo export` command and the absolute path of the export script in your install (a torchvision model has no export script; `info` gives the `torchvision.models` call that loads the same weights instead), and, when the model is built from an upstream repository, the build hook and the `axzoo fork --export` command. |
| `axzoo info --axm <path>` | The build snapshot recorded in a `.axm` that `axzoo` built: weight format, compiler backend and version, compiler configuration, content hash, pre-compile passes applied, and upstream provenance. It prints only what the build recorded. |
| `axzoo export <model> --format pt2` | Produces the model's source artifact (`pt2` or `onnx`, as `axzoo info` lists) the same way the packaged weight was produced, in an isolated venv. `--to PATH` chooses the destination. See [PyTorch model, ONNX, or `.pt2`?](#pytorch-model-onnx-or-pt2). |
| `axzoo fork <model> --export` | An editable copy of the script that builds the source artifact from an upstream repository, with a header that records the repository, the pinned ref, the formats and the build parameters. Edit its build function to point at your own checkpoint or variant. Only for models built from an upstream repository; for the others the command says so. |
| `axzoo fork <model> --script` | The standalone recipe for the model, regenerated from the installed package (see [Generate an editable recipe](#generate-an-editable-recipe-with-axzoo-fork)). |

To export your own weights for a packaged architecture, start from
`axzoo info <model>`: it prints the path of the export script that produced the
packaged weight, so you can open it. For a model built from an upstream
repository, `axzoo fork <model> --export` gives you an editable copy of the
build. Point the recipe's `load_source()` at the file your export produces.

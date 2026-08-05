# Model recipes

Model recipes are standalone Python scripts for selected Model Zoo models. They
show how to compile a model with either backend (the established `axrelay`
path, or the newer AxMO + graph-compiler `axgraph` path), and how to wire the
compiled artifact into a high-efficiency end-to-end pipeline in just a few
lines with Pipeline Builder.

Neither compiling a model (`axelera.compiler`/AxMO APIs) nor running it
end-to-end (`axelera.runtime2.op`) needs `axelera-zoo`. The committed recipes
import it anyway, for shared asset/calibration-resolution helpers and so
`--run`/`--benchmark` numbers match `axzoo benchmark`, but none of that is
structurally required; you could write the same compile-and-run script with
zero zoo imports.

The Model Zoo remains the source of truth for supported models, benchmark
numbers and accuracy data. Existing YAML model definitions remain supported.
Pipeline Builder is Alpha in this release; full production-level throughput
is planned for an upcoming release. If you already have a production
pipeline on an existing YAML Model Zoo model, no need to move it: the
YAML/GStreamer flow remains the most performant path today. If you need a
model only available through `axgraph` (ViT classifiers, DINOv2
segmentation), start on Pipeline Builder now: throughput is on its way, and
it's the only runnable path for those models today.

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

1. Copy a recipe anywhere, for example
   [`detection/yolov8n_coco_axrelay.py`](detection/yolov8n_coco_axrelay.py).
   The destination doesn't matter; the recipe is a self-contained script.
2. Point `load_source()` at your exported `.onnx` or `.pt2` file.
3. Replace `CALIBRATION` with 100 to 400 images representative of your
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
| `build_pipeline(axm_path)` | The readable Alpha Pipeline Builder graph used for run/benchmark checks: preprocess, `op.load()`, post-processing, and result conversion. |
| `CALIBRATION` | The representative data used for quantization. Replace this for your deployment scene. |
| `load_source()` | Where the `.onnx` or `.pt2` source artifact comes from. |
| Compiler calls | The `CompilerConfig` (or AxMO config, for `axgraph`) and the quantize/compile calls that build the `.axm`: exactly what `axzoo` runs for this model. |
| `--run` / `--benchmark` | Optional development checks for the recipe's Pipeline Builder path. |

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
`axelera-zoo` package. General application developers can usually work from
recipes directly. `axzoo` is useful when you want to discover packaged configs,
generate a fresh recipe, build from a managed config, or maintain a model catalog
over time.

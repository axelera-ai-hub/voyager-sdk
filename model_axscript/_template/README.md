# AxScript model template

Copy this folder to `model_axscript/<family>/<your-model>/`, or to
`model_axscript/<family>/<task>/<your-model>/` where the family groups its models by task,
and fill in `__init__.py`. That one file is the whole package: what a model DECLARES, plus
the seams its family's driver calls.

```
<your-model>/
  README.md      # what this model is, how to build and run it, the sizes it is measured at
  __init__.py    # MODEL_TAG + make_config + the seams the family driver calls
```

There is no entry script and no argument parser here, on purpose. Your family door
is the only command line in this tree, it declares every flag once, and it refuses a verb or a
flag a family does not have BY NAME rather than accepting one and dropping it. A per-model
parser is how a flag comes to exist on six models and be missing from the seventh.

## Steps to adapt

1. **Tag.** Set `MODEL_TAG` to the id this model is published under. The registry reads it
   straight out of this file, so it must be a plain string literal, and it must match the id
   the zoo config for this model declares: one model, one public id.
2. **Configuration.** Fill in `make_config` with your family's config dataclass, including the
   run sizes this model has been MEASURED at. Widening that set is a measurement, never an
   edit to make a build pass.
3. **Seams.** Implement the seams your family's driver calls. Open
   `model_axscript/<family>/shared/entry.py` to see which ones that is and which verbs the
   family honours; delete the seams it does not call. A genuinely NEW family also writes that
   file, declaring its `VERBS` and `OPTIONS`.
4. **Run it.**

   ```bash
   python -m model_axscript.<family> list              # your tag should appear here
   python -m model_axscript.<family> build <your-tag>
   python -m model_axscript.<family> run <your-tag> <input>
   ```

   Confirm the output reads as a RESULT for your task, not as a tensor shape.

5. **README.** Rewrite this file for your model: what it is, where its weights come from, how
   it is compiled, and the input sizes or context lengths it is validated at.

`model_axscript/vision/semantic_segmentation/dinov2_seg_vits14_ade20k_linear/` is a worked
example of a task-grouped model whose package is a configuration and nothing else, and
`model_axscript/vision/classification/vit_small_patch16_224_imagenet/` is one that also
carries a weight source and a host head of its own.

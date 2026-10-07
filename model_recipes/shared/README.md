# `shared/` - what every recipe does the same way

A recipe is a teaching script. What is worth reading in one is the model it
builds: the preprocess it needs, the pipelines it runs, the compiler settings
it was built with. What is not worth reading 54 times is argparse, an HTTPS
download, and the loop that feeds a dataset to an evaluator.

Those three live here. A recipe declares what is true of its model and hands
that over.

| File | What it holds | Needs `axelera-zoo`? |
| --- | --- | --- |
| [`cli.py`](cli.py) | The four verbs every recipe offers, and what each one does | no |
| [`assets.py`](assets.py) | The weight download, its checksum, the calibration image walk | no |
| [`accuracy.py`](accuracy.py) | The two `--eval` legs, over the Model Zoo's datasets | yes |

## Why the split is drawn there

`--build`, `--run` and `--benchmark` must work with `axelera-zoo` uninstalled.
That is the recipe contract, and it is what makes a recipe usable by someone
who never learns the catalog. So `cli.py` and `assets.py` import no zoo at all,
and a test reads their source to prove it rather than trusting an import check.

`--eval` is the one exception the contract allows, because the datasets, the
ground truth and the per-task evaluators live in the zoo and nowhere else.
`accuracy.py` reaches for them inside its functions, so this package stays
importable without the zoo, and a missing install produces a message naming the
package instead of an `ImportError`.

## Reading it

Start with `cli.py`. `run(recipe)` is the whole dispatch: it builds the parser
from what the recipe handed over, resolves the `.axm` (building it when needed),
and carries out each verb the caller named. `Recipe` and `Accuracy` are the two
things a recipe fills in.

`accuracy.py` is worth reading if you want to know what an accuracy number
means here. Two legs, `aipu` and `host`, and the pair is the point: the same
dataset, preprocess, decode tail and evaluator on both, with only the model
stage differing, so the difference between the two numbers is what deployment
cost rather than a difference in how they were measured.

## How a recipe reaches it

Each recipe opens with six lines: it puts its own directory, and the one above
it, at the front of `sys.path` when either holds a `shared/`, then `import
shared` and `shared.check_beside(__file__)`. Python already puts a script's own
directory on `sys.path`, so the prepend is what lets a recipe sitting one level
down in the SDK tree reach `model_recipes/shared/`.

`import shared` is a plain top-level import, which means one copy answers for
the whole process. That is right when every recipe in it came from one tree,
and wrong when a tool loads an SDK recipe and a forked one together: the second
would silently run the first one's copy, and an edit you made to your fork would
appear not to take effect. `check_beside` is what turns that into an error
naming both directories.

A process that deliberately imports several trees (the zoo renders a recipe per
config) sets the cached copy aside for each import, with
`axelera.zoo.run.isolated_shared`.

`shared/passes/` travels inside this directory, so it needs nothing of its own.

## Editing it

This directory travels with a recipe. `axzoo fork --script` and `axzoo build`
copy it next to the script they emit, and the wheel carries a copy for an
install with no source checkout. So a fork of yours is yours to change: nothing
regenerates it, and `axzoo drift-check` does not touch it.

In the SDK's own tree it is hand-written and shared by all 54 recipes, so a
change here changes every one of them.

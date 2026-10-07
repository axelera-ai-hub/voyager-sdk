# Copyright Axelera AI, 2026
"""What every recipe does the same way, kept out of every recipe.

A recipe is a teaching script. What is worth reading in one is this model: the
preprocess it needs, the pipeline it runs, the compiler settings it was built
with. What is not worth reading 54 times is argparse, an HTTPS fetch with a
checksum, and the loop that feeds a dataset to an evaluator. Those live here.

The split is by what carries information about the model:

    assets.py     download and checksum a weight, walk calibration images
    cli.py        the command line every recipe offers, and its dispatch
    accuracy.py   the `--eval` legs, over the Model Zoo's datasets

`assets` and `cli` import no zoo, because a recipe must build, run and
benchmark with `axelera-zoo` uninstalled. `accuracy` is the exception the
recipe contract allows, and it imports the zoo inside its functions so this
package stays importable without it.

This directory travels with a recipe: `axzoo fork --script` and `axzoo build`
copy it next to the script they emit, and the wheel carries it. A recipe
`cp`-ed out on its own says so and names this directory.
"""

from __future__ import annotations

from pathlib import Path

from . import accuracy, assets, cli, passes

__all__ = ['accuracy', 'assets', 'check_beside', 'cli', 'passes']


def check_beside(recipe_file: str) -> None:
    """Raise unless this is the copy that ships with `recipe_file`.

    A recipe imports `shared` by name, so the first copy imported in a process
    wins for every recipe after it. That is right when they all come from one
    tree, which is the normal case. It is wrong, and silently so, when a tool
    loads a recipe from the SDK tree and a forked one from somewhere else in the
    same process: the second would run the first one's copy, and an edit the
    customer made to their fork would appear not to take effect.

    So each recipe says which copy it expects, and a mismatch is an error naming
    both directories rather than a wrong answer.
    """
    ours = Path(__file__).resolve().parent
    here = Path(recipe_file).resolve().parent
    if ours.parent not in (here, here.parent):
        raise SystemExit(
            f'{Path(recipe_file).name} ships with the shared/ directory in {here}, but '
            f'{ours} was imported first in this process and is what it would use. Run the '
            f'two recipes separately, or copy them into one tree.'
        )

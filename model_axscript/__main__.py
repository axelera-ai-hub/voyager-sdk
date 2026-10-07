# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""The tree has one command line per family, and this is neither of them.

`python -m model_axscript` served every family at once, and the union it offered said less
than it looked like it said. The same word meant two things: `build` reaches a different
compiler for a picture model than for a decoder, and `run <tag> cat.jpg` binds a photograph
for one family and a sentence for the other. A user who typed no tag was offered the union
over every family, most of which their model refuses. Each family has a door of its own now,
so `--help` lists the flags that apply.

This prints where each family went and exits non-zero. Forwarding would have been shorter
and would have left the old command working, which is how a team keeps typing it for a year
and never reads the door that replaced it.
"""

from __future__ import annotations

import sys

from model_axscript import registry
from model_axscript.shared import cli


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    # The tag says which door was wanted, so the suggestion is one runnable command rather
    # than a menu. Without one, every door is a candidate and they are all named.
    tag = cli.tag_in(argv)
    families = [registry.family(tag)] if tag else cli.doors()
    typed = " ".join(argv)
    print(
        "`python -m model_axscript` is no longer a command line: each family has one of its "
        "own, so a door offers the flags that apply to the models behind it.",
        file=sys.stderr,
    )
    for family in families:
        print(f"  {f'{cli.door(family)} {typed}'.rstrip()}", file=sys.stderr)
    print(
        "Run a door with `list` to see the tags it answers for, or with `--help` for its flags.",
        file=sys.stderr,
    )
    return 2


if __name__ == "__main__":
    sys.exit(main())

# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""The vision command line: build, run and benchmark a picture model, by tag.

    python -m model_axscript.vision build     dinov2-vits14
    python -m model_axscript.vision run       dinov2-vits14 cat.jpg
    python -m model_axscript.vision benchmark dinov2-vits14 cat.jpg
    python -m model_axscript.vision list

Scoring is absent because the metric belongs to the zoo, keyed by task: `axzoo eval <card>`
scores a vision model and this door builds and runs one.

The door names its family and nothing else. Every flag, and every rule that spans two of
them, is declared once in `model_axscript.shared.cli`, which both doors import, so a flag
cannot exist on six models and be missing from the seventh.
"""

from __future__ import annotations

import sys

from model_axscript.shared import cli

if __name__ == "__main__":
    sys.exit(cli.main("vision"))

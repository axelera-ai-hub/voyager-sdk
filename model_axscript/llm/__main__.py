# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""The language command line: build, run, benchmark and score a decoder, by tag.

    python -m model_axscript.llm build     qwen3-0.6b
    python -m model_axscript.llm run       qwen3-0.6b "The capital of France is"
    python -m model_axscript.llm run       qwen3-0.6b --interactive
    python -m model_axscript.llm benchmark qwen3-0.6b
    python -m model_axscript.llm eval      qwen3-0.6b --tasks wikitext
    python -m model_axscript.llm run       qwen3-vl-2b-instruct "What is this?" --image cat.jpg
    python -m model_axscript.llm list

Text models and vision-language models share this door, because they share a decoder: a VLM
is that decoder with a vision tower in front of it, scored on the same tasks and compiled for
the same part. What differs is the input, and a VLM says so by declaring `DRIVER = "vl_entry"`
so its picture and clip flags are offered for it and refused by name for a text model.

The door names its family and nothing else. Every flag, and every rule that spans two of
them, is declared once in `model_axscript.shared.cli`, which both doors import, so a flag
cannot exist on six models and be missing from the seventh.
"""

from __future__ import annotations

import sys

from model_axscript.shared import cli

if __name__ == "__main__":
    sys.exit(cli.main("llm"))

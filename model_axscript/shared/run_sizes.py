# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""The run-size vocabulary, in a module with no dependencies.

A run size is the compiled buffer a result is only comparable within: `context_len` for an
LLM, `image_size` for a vision encoder. The names appear in three places that must agree,
the compiled artifact's `pipeline.json`, the config attribute declaring the measured set
(`supported_context_lens` / `supported_image_sizes`), and a committed report row's key, so
they are defined once and imported rather than spelled out per reader.

Deliberately a leaf module importing nothing. The obvious home is `runtime.py`, which reads
these keys back off an artifact, but that module needs numpy, and a reader wanting only the
vocabulary should not have to install a numerical stack to learn two strings. The zoo-only
test environment is exactly such a reader.
"""

from __future__ import annotations

RUN_SIZE_KEYS = ("context_len", "image_size")

INPUTS_KEY = "inputs"
INPUTS_DECLARED_ATTR = "supported_inputs"
"""The general form of the same identity, for a model no single number describes.

A run-size key is a scalar, which is right while ONE number sizes the compiled buffers: a
context length, or a square single-input image. It cannot state a 640x384 input, a two-input
model, or a tiled one, and a pair written as a bare list leaves every reader guessing the
order of the pair.

So a model whose identity is not one number states `inputs` instead: a mapping of graph-input
name to logical shape with the dims explicit, and declares its measured set under
`INPUTS_DECLARED_ATTR`. A model uses ONE of the two kinds and never both, so no reader has to
ask which key is authoritative for a row.

Logical shapes, not the packed device shapes in `model.json`: that file declares this encoder's
input as `[1, 196, 12, 64]`, which is the host's im2col layout and says nothing a reader could
compare against another model. The logical shape lives in `pipeline.json` because the builder
writes it there, which is also what keeps a row READ rather than typed.
"""

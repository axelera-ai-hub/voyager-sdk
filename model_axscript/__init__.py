# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0

"""AxScript models: the hand-authored compile programs, one package per model.

A regular package, not a namespace one, because the tree is entered two ways. Without this
file pytest walks up from `llm/tests/`, stops here, and names the module `llm.tests.<x>` with
this directory on `sys.path`, so every `..` inside `llm/` escapes the top-level package while
the same import resolves fine in production as `model_axscript.llm.<x>`. One empty marker
makes both entries agree.
"""

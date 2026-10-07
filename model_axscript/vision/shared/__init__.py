# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""What the vision models share. Declares no `MODEL_TAG`, so it is not a model.

`registry._candidate_dirs` walks this depth and lets `MODEL_TAG` decide, the same
way it does for the toolkit packages under `llm/`. The leading underscore this
directory used to carry made it unreachable by that walk; the name now matches
`model_recipes/shared/`, and declaring nothing is what keeps it out of the tag
list. Keep this file empty of declarations.
"""

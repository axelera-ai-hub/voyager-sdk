# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Which module builds and loads a card's device graph, resolved from its `implementation`.

Two builders exist for the same ViT encoder. `imported_dino` authors the device graph in this
tree, on `model_axscript/llm/models/dino/`; `axten_encoder` hands the upstream module to
`axelera.compiler.axten` and lets the compiler match and lower it. A card names one and changes
nothing else, and the name it declares is the one written into `pipeline.json`, so an artifact
records what built it.

THE IMPORT IS DEFERRED ON PURPOSE, not for load time. Both builders register the alto op
`llm.matmul`, `model_axscript/llm/ops/matmul.py` on one side and
`axten/ops/transformers/matmul.py` on the other, and the second registration raises
`Operation llm.matmul is already registered`. One process can therefore hold ONE builder.
Importing only the resolved one is what keeps that true; comparing the two means two processes.
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path

# `implementation` -> the module in this package that owns the graph. The prefix names the model
# family, as it already did, and the suffix names who implements the device graph.
_BUILDERS = {
    "dinov2-metis": "imported_dino",
    "vit-metis": "imported_dino",
    "dinov2-axten": "axten_encoder",
    "vit-axten": "axten_encoder",
    # The two depth families have no in-tree recipe at all: neither head is the linear one
    # `imported_dino` bakes, so axten is the only builder that can produce them.
    "dinov2-depth-axten": "axten_encoder",
    "depth-anything-axten": "axten_encoder",
}


def builder_for(config):
    """The builder module for `config.implementation`, imported now and not before."""
    try:
        module = _BUILDERS[config.implementation]
    except KeyError:
        raise ValueError(
            f"unknown implementation {config.implementation!r}; this family builds through "
            f"{sorted(_BUILDERS)}"
        ) from None
    return importlib.import_module(f".{module}", __package__)


def build_artifact(config):
    """Compile `config`'s artifact through the builder it names."""
    return builder_for(config).build_artifact(config)


def load_device(config):
    """The loaded device for an artifact built through the builder `config` names."""
    return builder_for(config).load_device(config)


#: The artifact's entry point, and the file that says one is there at all. Every builder writes
#: it at the artifact root; the graph model itself sits wherever its compiler puts it
#: (`imported_dino` beside it, axten one directory down), so `model.json` is NOT the check.
ARTIFACT_ENTRY = "pipeline.json"


def artifact_exists(artifact_dir) -> bool:
    """Is there an artifact of any builder at `artifact_dir`?"""
    return (Path(artifact_dir) / ARTIFACT_ENTRY).is_file()


def release(config, device) -> None:
    """Give the part back through the builder that loaded `device`.

    Resolved from the config, like `build_artifact` and `load_device`, rather than by probing the
    device object: the two builders hold their reservation in different places, and which one is
    in hand is a fact the config already carries.
    """
    builder_for(config).release(device)


def device_name(config, device) -> str | None:
    """The device `device` actually opened, through the builder that loaded it."""
    return builder_for(config).device_name(device)


def is_built(config) -> bool:
    """Is there an artifact at `config.artifact_dir` that THIS configuration's builder wrote?

    `pipeline.json` is the check because it is the contract's own entry point and every builder
    writes it in the same place, while the graph model itself sits wherever its compiler puts it
    (`imported_dino` beside it, axten one directory down).

    The implementation it records has to match, because the two builders lay their artifacts out
    differently. A card whose `implementation` changed with a build already on disk therefore
    rebuilds, rather than handing one builder's directory to the other's loader and failing on a
    missing file several frames down.
    """
    if not artifact_exists(config.artifact_dir):
        return False
    pipeline = Path(config.artifact_dir) / ARTIFACT_ENTRY
    built_by = json.loads(pipeline.read_text()).get("implementation")
    if built_by == config.implementation:
        return True
    print(
        f"[build] {config.artifact_dir} was built by {built_by!r} and this configuration asks "
        f"for {config.implementation!r}; rebuilding"
    )
    return False

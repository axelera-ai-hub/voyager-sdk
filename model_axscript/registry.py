# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Which model a tag names, and where its code lives.

A tag (`qwen3-0.6b`, `dinov2-vits14`) is what a user types and what a card, a build
directory and a report are keyed on. The package that declares it is the model's own
`__init__.py`, through `MODEL_TAG`. This resolves the one from the other by reading the
declaration, never by importing the package: an import pulls torch and the compiler in, and
listing the models must cost nothing.

A model sits at `<family>/<model>/` or at `<family>/<task>/<model>/`. Both depths are
registered the same way, because the depth is a navigation choice per family (a family with
one task has nothing to group by) and a tag must not encode it.
"""

from __future__ import annotations

import ast
from pathlib import Path

_ROOT = Path(__file__).resolve().parent


def _declared(init_path: Path, name: str):
    """The literal a model package assigns to `name` at module scope, or None.

    Read rather than imported, for the reason the module docstring gives: importing a model
    package pulls torch and the compiler in, and answering "what does this model declare"
    must cost nothing. A non-literal value reads as None -- a declaration this can only
    learn by executing the package is not a declaration this promises to see.
    """
    for node in ast.parse(init_path.read_text()).body:
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Name) and target.id == name:
                try:
                    return ast.literal_eval(node.value)
                except ValueError:
                    return None
    return None


def _declared_tag(init_path: Path) -> str | None:
    """The `MODEL_TAG = "..."` a model package declares, or None when it declares none."""
    tag = _declared(init_path, "MODEL_TAG")
    return None if tag is None else str(tag)


def _is_public(part: str) -> bool:
    """Whether a path component can be part of a model's address."""
    return not part.startswith((".", "_")) and part != "skills"


def _candidate_dirs() -> list[Path]:
    """Directories that may declare a model, at either depth the tree uses.

    Nothing here decides whether a directory IS a model: `MODEL_TAG` does. So the toolkit's
    own packages (`llm/ops`, `llm/models/<arch>`) are candidates and simply declare nothing.
    """
    candidates = (*_ROOT.glob("*/*/__init__.py"), *_ROOT.glob("*/*/*/__init__.py"))
    return sorted(
        init.parent
        for init in candidates
        if all(_is_public(part) for part in init.relative_to(_ROOT).parts[:-1])
    )


def models() -> dict[str, Path]:
    """Every registered tag, mapped to its model directory, sorted by tag."""
    found: dict[str, Path] = {}
    for model in _candidate_dirs():
        tag = _declared_tag(model / "__init__.py")
        if tag is not None:
            found[tag] = model
    return dict(sorted(found.items()))


def model_dir(tag: str) -> Path:
    """The directory of the model `tag` names, or KeyError naming what is registered."""
    found = models()
    if tag not in found:
        known = ", ".join(found) or "no models are registered in this tree"
        raise KeyError(f"unknown model {tag!r}; registered tags: {known}")
    return found[tag]


def _address(tag: str) -> tuple[str, ...]:
    """The path components of the model's directory below `model_axscript`."""
    return model_dir(tag).relative_to(_ROOT).parts


def family(tag: str) -> str:
    """The driver family the model belongs to (`llm`, `vision`).

    The FIRST component below `model_axscript`, not the parent directory: where a family
    groups its models by task the parent is the task, and it is the family that selects
    which driver runs the model.
    """
    return _address(tag)[0]


def task(tag: str) -> str:
    """The zoo TASK the model is grouped under, which selects a vision model's eval harness.

    The SECOND component below `model_axscript`, where `family()` is the first. A model at the
    flat depth has no task level to read, and that is said by name rather than guessed at: a
    guess here would score one task's dataset under another task's metric.
    """
    address = _address(tag)
    if len(address) < 3:
        raise KeyError(
            f"{tag} is not grouped by task: it sits at {'/'.join(address[:-1])}, so there is "
            f"no task level to read. Group it as <family>/<task>/<model>/ to give it one."
        )
    return address[1]


def package(tag: str) -> str:
    """The importable package path of the model `tag` names."""
    return ".".join((_ROOT.name, *_address(tag)))


def declares(tag: str, name: str, default=None):
    """What the model `tag` declares for `name` at module scope, or `default`.

    The generalisation of the `MODEL_TAG` read, for the handful of facts the front door
    needs BEFORE deciding what to run -- and so before it can afford to import anything.
    `DRIVER` is the one today: a model whose verbs are answered by a driver other than its
    family's default names it, and the family driver hands every verb to that one. A
    two-tower VLM declares `vl_entry` this way.
    """
    value = _declared(model_dir(tag) / "__init__.py", name)
    return default if value is None else value

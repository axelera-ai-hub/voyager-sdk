# Copyright Axelera AI, 2026
"""Graph rewrites a model needs before the compiler will accept it.

Each pass is one file in this directory. A recipe names the ones its model
needs in `PRE_COMPILE_PASSES` and calls `apply`; nothing else here is part of
the recipe's surface. README.md says how to tell whether your own model needs
a pass, and how to write one.
"""

from __future__ import annotations

import ast
import importlib
from pathlib import Path
from typing import Any

__all__ = ['apply']

# A leading `_` marks a shared helper, `test_` a test. The zoo's catalogue and
# its wheel-build hook hold the same rule; a test pins the three equal.
_NOT_A_PASS_PREFIXES = ('_', 'test_')

# What `apply` reads of the six README section 4 lists. A pass missing one of
# these can be neither run nor skipped, so it is refused by name.
_REQUIRED_DECLARATIONS = ('SOURCE_KINDS', 'DEPRECATED_IN', 'DESCRIPTION')


def _available() -> list[str]:
    """Pass names in this directory. A leading `_` marks a shared helper."""
    return sorted(
        p.stem
        for p in Path(__file__).parent.glob('*.py')
        if not p.name.startswith(_NOT_A_PASS_PREFIXES)
    )


def _declarations(path: Path) -> dict[str, Any]:
    """A pass's module-level literal constants, read WITHOUT importing it.

    Reading a declaration must not execute the pass. The ONNX rewrites here
    `import onnx` at module scope, and a build in another weight format skips
    them, so importing one first would make the skip need the very toolchain it
    was skipping. Only literals are read, which is all a declaration may be.
    """
    declared: dict[str, Any] = {}
    for node in ast.parse(path.read_text(encoding='utf-8')).body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if not isinstance(target, ast.Name) or not target.id.isupper():
            continue
        try:
            declared[target.id] = ast.literal_eval(node.value)
        except ValueError:
            continue
    return declared


def _source_kind(model: Any) -> str:
    """The weight format `model` is in: an ONNX path or graph, else a torch module.

    A pass tried out by hand is usually handed the loaded ONNX graph rather than
    a path, and calling that torch would skip an ONNX-only pass and blame its
    declaration. Testing `graph` alone would misread a torch module that happens
    to have one, so both ONNX-only fields are required.
    """
    if isinstance(model, (str, Path)):
        return '.onnx'
    if hasattr(model, 'graph') and hasattr(model, 'opset_import'):
        return '.onnx'
    return '.pt2'


def apply(model: Any, names: list[str] | tuple[str, ...]) -> Any:
    """Run each named pass over `model`, in order, and return the result.

    `model` is an ONNX file path or a torch module, exactly as the recipe
    loaded it. A pass declaring other `SOURCE_KINDS` is skipped rather than
    forced, so a recipe can list a rewrite that applies to one of its weight
    formats and not the other.

    A pass returns `None` to say it found nothing to rewrite; that is reported
    and is not an error. `SUPPORTS` is deliberately not read here: it names the
    compiler backends a rewrite is valid for, and a recipe is generated for one
    backend, so the pair is checked once, where the recipe is generated.
    """
    source_kind = _source_kind(model)
    directory = Path(__file__).parent
    for name in names:
        path = directory / f'{name}.py'
        if not path.is_file():
            raise SystemExit(
                f'{name!r} is not a pass in {directory}. Available: '
                f'{", ".join(_available()) or "none"}.'
            )
        if name.startswith(_NOT_A_PASS_PREFIXES):
            raise SystemExit(
                f'{name!r} is a helper module in {directory}, not a pass. Available: '
                f'{", ".join(_available()) or "none"}.'
            )
        declared = _declarations(path)
        missing = [field for field in _REQUIRED_DECLARATIONS if field not in declared]
        if missing:
            raise SystemExit(
                f'{path} declares no {", ".join(missing)} as a literal. Every pass declares '
                f'the six constants README.md section 4 lists.'
            )
        if source_kind not in declared['SOURCE_KINDS']:
            print(f'[pass] {name}: declared for {" ".join(declared["SOURCE_KINDS"])}, skipped')
            continue
        if declared['DEPRECATED_IN'] is not None:
            print(
                f'[pass] {name}: absorbed into the compiler in {declared["DEPRECATED_IN"]}, '
                f'skipped. Remove it from PRE_COMPILE_PASSES.'
            )
            continue
        try:
            module = importlib.import_module(f'{__package__}.{name}')
        except ModuleNotFoundError as exc:
            # The file is there, so the missing module is one the pass imports.
            # "Not a pass" would send the reader to check a spelling that is right.
            raise SystemExit(
                f'{name!r} imports {exc.name!r}, which is not installed. The rewrites here '
                f"run in the compiler's environment; install it there and build again."
            ) from None
        if not hasattr(module, 'run'):
            raise SystemExit(
                f'{path} defines no run(), so it is not a pass. Available: '
                f'{", ".join(_available()) or "none"}.'
            )
        if isinstance(model, (str, Path)):
            # The recipe hands `quantize` an .onnx path; a graph rewrite needs
            # the loaded graph. Load once, before the first pass that runs.
            import onnx

            model = onnx.load(str(model))
        location = Path(module.__file__)
        # Before `run`, so a pass that raises is attributable; the outcome
        # follows, so a stage that rewrote nothing does not read like one that did.
        print(
            f'[pass] {name}: {declared["DESCRIPTION"]}  ({location.parent.name}/{location.name})'
        )
        rewritten = module.run(model)
        if rewritten is None:
            print(f'[pass] {name}: found nothing to rewrite in this model')
        else:
            model = rewritten
    return model

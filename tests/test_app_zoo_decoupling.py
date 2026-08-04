# Copyright Axelera AI, 2026
"""Guards that axelera.app does not depend on axelera.zoo.

axelera.zoo ships as a separate, wheel-only package; the framework must run
without it installed.
"""
import ast
from pathlib import Path

import pytest

_APP_ROOT = Path(__file__).resolve().parents[1] / "axelera" / "app"


def _app_py_files():
    return [p for p in _APP_ROOT.rglob("*.py") if "tests" not in p.parts]


def test_no_app_module_imports_zoo():
    offenders = []
    for path in _app_py_files():
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            if any(n == "axelera.zoo" or n.startswith("axelera.zoo.") for n in names):
                offenders.append(f"{path.relative_to(_APP_ROOT.parent.parent)}:{node.lineno}")
    assert not offenders, "axelera.app must not import axelera.zoo: " + ", ".join(offenders)


@pytest.mark.parametrize(
    "module",
    [
        "axelera.app.logging_utils",
        "axelera.app.data_utils",
        "axelera.app.config",
        "axelera.app.utils",
    ],
)
def test_app_logging_and_data_modules_import(module):
    import importlib

    importlib.import_module(module)


def test_logging_config_is_app_local():
    from axelera.app.config import LoggingConfig

    cfg = LoggingConfig()
    assert hasattr(cfg, "console_level")
    assert hasattr(cfg, "compiler_level")

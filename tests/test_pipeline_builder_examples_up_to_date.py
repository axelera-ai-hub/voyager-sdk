# Copyright Axelera AI, 2026
"""Drift guard for the auto-generated examples/pipeline_builder/*.py.

Those files are committed generated artifacts produced by
``tools/export_pipeline_builder_demo.py`` from the ``@publicdemo``
functions in ``internal-apps/axelera/internal/apps/rt_demo.py``.  Their
file header tells users "Edit the original demo in rt_demo.py rather
than this file" but nothing else enforces it: edits to either
``rt_demo.py`` or the exporter that aren't followed by a regeneration
leave the checked-in examples silently stale.

This test re-runs the exporter into a tmp directory and asserts each
generated file is byte-identical to the committed copy.  CI fails
whenever the working tree drifts from what the exporter would produce.
"""

from __future__ import annotations

import difflib
import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXPORTER_PATH = REPO_ROOT / 'tools' / 'export_pipeline_builder_demo.py'
EXAMPLES_DIR = REPO_ROOT / 'examples' / 'pipeline_builder'


def _load_exporter():
    spec = importlib.util.spec_from_file_location('_pb_exporter', EXPORTER_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover
        raise RuntimeError(f'could not load exporter from {EXPORTER_PATH}')
    mod = importlib.util.module_from_spec(spec)
    sys.modules['_pb_exporter'] = mod
    spec.loader.exec_module(mod)
    return mod


_exp = _load_exporter()

# Loading the demo list cascades into ``from axelera.runtime import cv,
# display, op`` (via rt_demo.py).  Those names only exist on
# ``axelera-runtime >= 1.7``; the framework CI's unittests env is pinned
# to an older version that omits them.  Skip the whole module when the
# environment cannot satisfy that import -- the drift guard is meaningful
# only in envs that can actually re-run the exporter (smoke harness,
# internal-apps wheel build env, dev venvs).
try:
    _PUBLIC_DEMOS = _exp.list_public_demos()
except ImportError as e:
    pytest.skip(
        f'drift guard needs axelera.runtime with cv/display/op '
        f'(axelera-runtime >= 1.7); current env: {e}',
        allow_module_level=True,
    )


@pytest.mark.parametrize('demo_name', _PUBLIC_DEMOS)
def test_pipeline_builder_example_up_to_date(demo_name, tmp_path):
    committed_path = EXAMPLES_DIR / f'{demo_name}.py'
    if not committed_path.is_file():
        pytest.fail(
            f'missing checked-in example: {committed_path}. '
            f'Generate with: python tools/export_pipeline_builder_demo.py {demo_name}'
        )

    source, tree = _exp._read_rt_demo_source()
    regen_path = tmp_path / f'{demo_name}.py'
    _exp.export_demo(demo_name, output_path=regen_path, source=source, tree=tree)

    committed = committed_path.read_text()
    regenerated = regen_path.read_text()
    if committed == regenerated:
        return

    diff = '\n'.join(
        difflib.unified_diff(
            committed.splitlines(),
            regenerated.splitlines(),
            fromfile=f'examples/pipeline_builder/{demo_name}.py (committed)',
            tofile=f'examples/pipeline_builder/{demo_name}.py (regenerated)',
            lineterm='',
        )
    )
    pytest.fail(
        f'examples/pipeline_builder/{demo_name}.py is stale -- rt_demo.py or '
        f'the exporter was edited without regenerating.  Regenerate with:\n'
        f'  python tools/export_pipeline_builder_demo.py {demo_name}\n'
        f'or, for all of them:\n'
        f'  python tools/export_pipeline_builder_demo.py all\n\n'
        f'Diff (committed vs regenerated):\n{diff}'
    )

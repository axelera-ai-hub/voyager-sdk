#!/usr/bin/env python3
# Copyright Axelera AI, 2026
"""Download .axm files referenced by the public pipeline_builder demos.

Scans every ``examples/pipeline_builder/*.py`` for ``{MODEL_DIR}/<name>.axm``
references and downloads each one into ``~/.cache/axelera/runtime2/`` via
``axdownloadmodel``.  Models not in the public catalog can be compiled
locally with ``--deploy-missing`` (calls ``yolo export model=<stem>.pt
format=axelera`` -- requires ultralytics + axelera-devkit).

Usage:
    python download_axm.py                    # download every referenced .axm
    python download_axm.py --version <ver>    # pin axdownloadmodel to a model version
    python download_axm.py --deploy-missing   # also yolo-export anything missing
    python download_axm.py --list             # list referenced .axm and exit
"""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
import subprocess
import sys

EXAMPLE_DIR = Path(__file__).resolve().parent
MODEL_CACHE_DIR = Path.home() / '.cache' / 'axelera' / 'runtime2'

# Filename of this script -- skipped when scanning EXAMPLE_DIR so the
# download tool does not try to ingest its own source.
_SELF_NAME = Path(__file__).name

# Matches `{MODEL_DIR}/<stem>.axm` inside an f-string.  The exported demos
# all follow this exact form (see tools/export_pipeline_builder_demo.py's
# template), so anchoring on the literal `{MODEL_DIR}/` placeholder keeps
# the regex specific enough to avoid false positives on, say, comments.
_AXM_REF = re.compile(r'\{MODEL_DIR\}/([A-Za-z0-9._-]+)\.axm')


def collect_axm_stems(example_dir: Path = EXAMPLE_DIR) -> list[str]:
    """Return the sorted set of .axm stems referenced under ``example_dir``."""
    stems: set[str] = set()
    for py in sorted(example_dir.glob('*.py')):
        if py.name == _SELF_NAME:
            continue
        for m in _AXM_REF.finditer(py.read_text()):
            stems.add(m.group(1))
    return sorted(stems)


def _run_or_warn(cmd: list[str], *, missing_msg: str) -> subprocess.CompletedProcess | None:
    """Run ``cmd`` in MODEL_CACHE_DIR.  Returns the CompletedProcess on
    success or non-zero exit, or ``None`` if the executable is missing
    (after printing ``missing_msg`` to stderr).
    """
    try:
        return subprocess.run(
            cmd, cwd=str(MODEL_CACHE_DIR), check=False, capture_output=True, text=True
        )
    except FileNotFoundError:
        print(f'    {missing_msg}', file=sys.stderr)
        return None


def _log_failure_tail(result: subprocess.CompletedProcess) -> None:
    detail = (result.stderr or result.stdout or '').strip().splitlines()
    print(f'    failed: {detail[-1] if detail else "(no output)"}', file=sys.stderr)


def _download_via_axdownloadmodel(stems: list[str], version: str | None = None) -> None:
    """Run a single ``axdownloadmodel --axm <stem1> <stem2> ...`` for every
    stem in ``stems``.  Callers are expected to re-scan MODEL_CACHE_DIR to
    decide which stems actually landed (``axdownloadmodel`` exits 0 even
    for unknown models, so per-stem file-existence is the only reliable
    signal).

    When ``version`` is given it is forwarded as ``--version <version>``
    to pin a specific model revision in the public catalog.
    """
    if not stems:
        return
    cmd = ['axdownloadmodel', '--axm', *stems]
    if version is not None:
        cmd.extend(['--version', version])
    print(f'  [axdl] {" ".join(cmd)}')
    result = _run_or_warn(cmd, missing_msg='axdownloadmodel not on PATH')
    if result is not None and result.returncode != 0:
        _log_failure_tail(result)


def _compile_via_yolo_export(stem: str) -> bool:
    """Compile ``<stem>.axm`` via ``yolo export model=<stem>.pt format=axelera``.

    Writes its artefacts to ``MODEL_CACHE_DIR/<stem>_axelera_model/<stem>.axm``;
    this function then copies that up to ``MODEL_CACHE_DIR/<stem>.axm`` so
    the demo finds it at the canonical path.  A previously-compiled nested
    artefact is reused without recompiling.
    """
    target = MODEL_CACHE_DIR / f'{stem}.axm'
    nested = MODEL_CACHE_DIR / f'{stem}_axelera_model' / f'{stem}.axm'
    if not nested.is_file():
        print(
            f'  [yolo] yolo export model={stem}.pt format=axelera (this may take several minutes)'
        )
        result = _run_or_warn(
            ['yolo', 'export', f'model={stem}.pt', 'format=axelera'],
            missing_msg='yolo (ultralytics) CLI not on PATH; install with `pip install ultralytics`',
        )
        if result is None:
            return False
        if not nested.is_file():
            _log_failure_tail(result)
            return False
    shutil.copy(nested, target)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        '--list',
        '-l',
        action='store_true',
        help='List the .axm stems referenced by the public demos and exit.',
    )
    parser.add_argument(
        '--version',
        default=None,
        metavar='VERSION',
        help='Pass-through to `axdownloadmodel --version <VERSION>` to pin a '
        'specific model revision in the public catalog. Affects only the '
        'axdownloadmodel path; the `--deploy-missing` yolo-export fallback '
        'ignores this value.',
    )
    parser.add_argument(
        '--deploy-missing',
        action='store_true',
        help='For any .axm not retrievable via axdownloadmodel, fall back '
        'to `yolo export model=<stem>.pt format=axelera`. Requires '
        'ultralytics + axelera-devkit; compilation can take several '
        'minutes per model.',
    )
    args = parser.parse_args()

    stems = collect_axm_stems()
    if args.list:
        for s in stems:
            print(s)
        return 0

    MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f'Target: {MODEL_CACHE_DIR}')
    print(f'Demos reference {len(stems)} .axm file(s):')
    for s in stems:
        print(f'  - {s}.axm')
    print()

    def _missing(names: list[str]) -> list[str]:
        return [s for s in names if not (MODEL_CACHE_DIR / f'{s}.axm').is_file()]

    for s in stems:
        if s not in _missing(stems):
            print(f'  [skip] {s}.axm already present')

    # One axdownloadmodel invocation for every stem still missing on disk.
    _download_via_axdownloadmodel(_missing(stems), version=args.version)

    # Per-stem yolo fallback for whatever axdownloadmodel did not produce.
    if args.deploy_missing:
        for stem in _missing(stems):
            _compile_via_yolo_export(stem)

    missing = _missing(stems)
    print()
    if missing:
        print(f'Missing after all attempts: {", ".join(missing)}', file=sys.stderr)
        if not args.deploy_missing:
            print(
                '  Retry with --deploy-missing to attempt local compilation via `yolo export`.',
                file=sys.stderr,
            )
        return 1
    print(f'All {len(stems)} .axm file(s) present in {MODEL_CACHE_DIR}.')
    return 0


if __name__ == '__main__':
    sys.exit(main())

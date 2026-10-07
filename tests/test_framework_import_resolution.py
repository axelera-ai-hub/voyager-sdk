# Copyright Axelera AI, 2026
"""The tests must import the tree they are checked out in.

`axelera.zoo` is installed editable, and the `.pth` records whichever checkout was
pip-installed. Run from a git worktree, a plain import therefore returns the OTHER
tree's copy, and the suite silently tests code that is not the code under test. That
has produced fake failures, a fake "install axelera-zoo" error, and a fake clean run.

`pythonpath` in the pytest config prevents it by putting this tree first. These tests
fail if that entry is ever dropped.
"""

from pathlib import Path
import subprocess
import sys
import tempfile
import warnings

import pytest

_TREE = Path(__file__).resolve().parents[1]


def _run_git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", *args],
        cwd=repo,
        check=check,
        text=True,
        capture_output=True,
    )


def _resolve_git_path(repo: Path, path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else (repo / candidate).resolve()


def _is_primary_checkout(repo: Path) -> bool:
    git_dir = _run_git(repo, "rev-parse", "--git-dir").stdout.strip()
    common_dir = _run_git(repo, "rev-parse", "--git-common-dir").stdout.strip()
    return _resolve_git_path(repo, git_dir) == _resolve_git_path(repo, common_dir)


def test_axelera_zoo_resolves_inside_this_tree():
    if not (_TREE / "axelera_zoo" / "python").is_dir():
        pytest.skip("not a source checkout; axelera.zoo is installed as a package")
    zoo = pytest.importorskip("axelera.zoo")
    assert Path(zoo.__file__).resolve().is_relative_to(_TREE), (
        f"axelera.zoo resolved to {zoo.__file__}, outside the tree under test ({_TREE}). "
        "The pythonpath entry in pyproject.toml has been dropped, so this run is testing "
        "another checkout."
    )


def test_import_resolution_from_ephemeral_worktree_when_primary_checkout():
    if not (_TREE / "axelera_runtime2" / "python").is_dir():
        pytest.skip("not a source checkout; import-resolution worktree smoke test is inapplicable")

    try:
        if not _is_primary_checkout(_TREE):
            pytest.skip("already running inside a linked worktree")
    except (FileNotFoundError, subprocess.CalledProcessError) as exc:
        pytest.skip(f"cannot determine worktree state: {exc}")

    with tempfile.TemporaryDirectory(prefix="af-import-resolution-") as tmp_dir:
        worktree = Path(tmp_dir) / "wt"
        try:
            _run_git(_TREE, "worktree", "add", "--detach", str(worktree), "HEAD")
        except (FileNotFoundError, subprocess.CalledProcessError) as exc:
            warnings.warn(
                f"cannot create temporary worktree for import-resolution smoke test: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
            pytest.skip("environment cannot create a temporary git worktree")
        cleanup_error = None
        try:
            run = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "pytest",
                    "tests/test_framework_import_resolution.py::test_axelera_zoo_resolves_inside_this_tree",
                    "axelera_runtime2/tests/python/test_runtime2_import_resolution.py::"
                    "test_axelera_runtime2_resolves_inside_this_tree",
                    "-q",
                ],
                cwd=worktree,
                text=True,
                capture_output=True,
            )
        finally:
            remove = _run_git(_TREE, "worktree", "remove", "--force", str(worktree), check=False)
            if remove.returncode != 0:
                cleanup_error = (
                    f"failed to remove temporary worktree {worktree}:\n"
                    f"stdout:\n{remove.stdout}\n"
                    f"stderr:\n{remove.stderr}"
                )

        if cleanup_error:
            pytest.fail(cleanup_error)

        assert run.returncode == 0, (
            "worktree import-resolution smoke test failed.\n"
            f"stdout:\n{run.stdout}\n"
            f"stderr:\n{run.stderr}"
        )

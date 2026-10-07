# Copyright Axelera AI, 2026
# The containerless checks are bash, because what they check is bash. Nothing invoked them,
# so they could not fail: run them from the suite CI already runs.
import pathlib
import subprocess

import pytest

SCRIPTS = sorted(pathlib.Path(__file__).parent.glob("test_containerless_*.sh"))


def test_the_shell_tests_were_found():
    assert SCRIPTS, "no tests/test_containerless_*.sh found; this shim would pass vacuously"


@pytest.mark.parametrize("script", SCRIPTS, ids=lambda p: p.stem)
def test_containerless_shell_test(script):
    done = subprocess.run(["bash", str(script)], capture_output=True, text=True)
    assert done.returncode == 0, f"{script.name} failed:\n{done.stdout}\n{done.stderr}"

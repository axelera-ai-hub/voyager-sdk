# Copyright Axelera AI, 2026
"""Vendored subset of the ALTO test-environment presets.

Source: axelera-ai/host.compiler.alto, axelera_compiler/tests/development/pytest_presets.py,
revision ca88e4d224b63a7b00b074715810d0eea0c54a2a.

Upstream builds these as `pytest.param(generation, marks=[pytest.mark.test_enabled(runs_on=...,
device_generation=...)])`, backed by the ax_pytest_helpers environment-filter plugin (not
installed in this repo -- this repo's conftest instead gives every test a permissive
`test_enabled()` by default, see model_axscript/llm/tests/conftest.py).

The vendored presets keep only what the migrated tests actually need from it: the
HardwareGeneration parameters for `@pytest.mark.parametrize("generation", [...])`.
Every test that uses it here also targets Target.AICORE_SIM via `compile_and_run`, i.e. the
ALTO functional simulator ("spike"), which this dev machine does not have on PATH -- so the
param carries the repo's own `slow` marker (see pyproject.toml) instead of the upstream
environment-filter marks.
"""

from axelera.compiler.alto.compiler.config import HardwareGeneration
import pytest

AICORESIM_OMEGA = pytest.param(HardwareGeneration.OMEGA, marks=[pytest.mark.slow])
AICORESIM_EUROPA = pytest.param(HardwareGeneration.EUROPA, marks=[pytest.mark.slow])

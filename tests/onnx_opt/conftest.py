# Copyright Axelera AI, 2026
# Configuration for ONNX optimization tests
#
# NOTE: Tests in this directory are excluded from py310-runtime tox environment because:
# - onnx_optimizations.py is only used during deployment/compilation phase
# - ONNX dependencies are not needed at runtime
# - See tox.ini [testenv:py{310,312}-runtime] for exclusion

import pytest

# Apply onnx_opt marker to all tests in this directory
# This allows selective running: pytest -m "not onnx_opt" to skip, pytest -m onnx_opt to run only these
pytestmark = pytest.mark.onnx_opt

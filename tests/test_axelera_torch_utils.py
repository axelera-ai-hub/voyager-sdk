# Copyright Axelera AI, 2024
from unittest.mock import Mock, patch

import pytest

from axelera.app import torch_utils


def mock_availability(available):
    """return a mock torch that reports availability of cuda or mps"""
    m = Mock()
    m.cuda.is_available.return_value = available == "cuda"
    m.backends.mps.is_available.return_value = available == "mps"
    m.device = lambda device: Mock(type=device)
    return m


@pytest.mark.parametrize("desired", ["cpu", "mps", "cuda"])
def test_device_name_explicit(desired):
    with patch.object(torch_utils, 'torch', mock_availability("")):
        assert desired == torch_utils.device_name(desired)


@pytest.mark.parametrize("available", ["mps", "cuda"])
def test_device_name_something_available(available):
    with patch.object(torch_utils, 'torch', mock_availability(available)):
        assert available == torch_utils.device_name()
        assert available == torch_utils.device_name("auto")


def test_device_name_nothing_available():
    with patch.object(torch_utils, 'torch', mock_availability("")):
        assert "cpu" == torch_utils.device_name()
        assert "cpu" == torch_utils.device_name("auto")

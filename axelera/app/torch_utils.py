# Copyright Axelera AI, 2024
# Utilities to avoid importing torch unless necessary.

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Protocol, Sequence

from . import config
from .logging_utils import getLogger

LOG = getLogger(__name__)

if TYPE_CHECKING:
    import torch
else:
    try:
        import torch
    except ImportError:

        class Torch:
            def __getattr__(self, name):
                raise ImportError("torch not available")

        torch = Torch()

try:
    from torch.utils import data
except ImportError:

    class DataLoader(Protocol):
        def __len__(self) -> int:
            ...

        def __iter__(self) -> Iterable[Any]:
            ...

    class Dataset(Protocol):
        def __init__(self, data: Sequence[Any]):
            ...

        def __len__(self):
            ...

        def __getitem__(self, idx):
            ...

    class data:
        DataLoader = DataLoader
        Dataset = Dataset


TORCH_DEVICE_NAMES = ['auto', 'cuda', 'mps', 'cpu']


def device_name(desired_device_name: str = 'auto') -> str:
    '''Return the name of the backend to use for torch.device.

    `desired_device_name` can be one of 'auto', 'cuda', 'cpu', 'mps'.  If auto
    then either cuda or mps will be used if available, otherwise cpu as a
    fallback.
    '''
    assert desired_device_name in TORCH_DEVICE_NAMES
    if desired_device_name == 'auto':
        if device := config.env.torch_device:
            return device
        if torch.cuda.is_available():
            LOG.info("Using CUDA based torch")
            return 'cuda'
        elif (mps := getattr(torch.backends, 'mps', None)) and mps.is_available():
            LOG.info("Using MPS based torch")
            return 'mps'
        LOG.info("Using CPU based torch")
        return 'cpu'
    return desired_device_name

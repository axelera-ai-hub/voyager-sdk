# Copyright Axelera AI, 2025
# Utilities to avoid importing torch unless necessary.

from __future__ import annotations

import importlib
from os import PathLike
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Dict,
    Iterable,
    List,
    Optional,
    Protocol,
    Sequence,
    Union,
)

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
        def __len__(self) -> int: ...

        def __iter__(self) -> Iterable[Any]: ...

    class Dataset(Protocol):
        def __init__(self, data: Sequence[Any]): ...

        def __len__(self): ...

        def __getitem__(self, idx): ...

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


def safe_torch_load(
    file_path: str | PathLike,
    map_location: Optional[Union[str, torch.device, Dict[str, str], Callable]] = None,
    allowed_classes: Optional[List[type]] = None,
) -> Any:
    """
    Safely load a PyTorch model with compatibility across PyTorch versions.

    This function handles the differences between PyTorch 1.x and 2.6+ serialization APIs.
    PyTorch 2.6+ introduced stricter deserialization with weights_only=True by default,
    requiring explicit allowlisting of custom classes through safe_globals.

    Args:
        file_path: Path to the model file to load
        map_location: Optional device mapping for tensor locations (e.g., 'cpu', 'cuda')
        allowed_classes: Optional list of custom classes to allow when deserializing with PyTorch 2.6+
                         If None and an error occurs, will attempt to extract the class from error messages

    Returns:
        The loaded PyTorch object (model, state_dict, etc.)

    Raises:
        ImportError: If PyTorch is not installed
        FileNotFoundError: If the model file doesn't exist
        RuntimeError: If the model cannot be loaded due to compatibility issues
    """
    if not isinstance(file_path, (str, PathLike)):
        raise TypeError(f"Expected file_path to be a string or PathLike, got {type(file_path)}")

    # Check if file exists to provide better error message
    import os

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Model file not found: {file_path}")

    # First try with default settings (compatible with PyTorch 1.x)
    try:
        import torch

        return torch.load(file_path, map_location=map_location)
    except Exception as e:
        original_error = str(e)
        if "weights_only" in original_error or "was not an allowed global" in original_error:
            LOG.debug(f"Using compatibility mode for PyTorch 2.6+ due to: {original_error}")

            # For PyTorch 2.6+, try with weights_only=False
            try:
                # Try adding the required class to the safe_globals list if available
                torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
                if torch_version >= (2, 6):
                    extracted_classes = []

                    # Try to extract the module and class from the error message
                    if allowed_classes is None and "was not an allowed global" in original_error:
                        # Parse class name from error message like:
                        # "WeightsUnpickler error: Unsupported global: GLOBAL models.yolo.Model was not an allowed global"
                        import re

                        # Look for all global classes mentioned in the error
                        matches = re.finditer(r'GLOBAL\s+([^\s]+)\s+was not', original_error)
                        for match in matches:
                            class_path = match.group(1)
                            LOG.debug(
                                f"Attempting to allowlist {class_path} extracted from error message"
                            )
                            try:
                                # Split into module path and class name
                                last_dot = class_path.rfind('.')
                                if last_dot > 0:
                                    module_path = class_path[:last_dot]
                                    class_name = class_path[last_dot + 1 :]

                                    # Import the module and get the class
                                    module = importlib.import_module(module_path)
                                    cls = getattr(module, class_name)
                                    extracted_classes.append(cls)
                            except (ImportError, AttributeError) as err:
                                LOG.warning(
                                    f"Could not automatically import class {class_path}: {err}"
                                )

                    if allowed_classes:
                        extracted_classes.extend(allowed_classes)

                    try:
                        # For PyTorch 2.6+
                        import torch.serialization

                        if extracted_classes:
                            LOG.debug(
                                f"Using safe_globals with allowed classes: {extracted_classes}"
                            )
                            with torch.serialization.safe_globals(extracted_classes):
                                return torch.load(
                                    file_path, map_location=map_location, weights_only=False
                                )
                        else:
                            # If no classes to allowlist, try without allowlisting but with weights_only=False
                            LOG.debug(
                                "No allowed classes specified, trying with weights_only=False"
                            )
                            return torch.load(
                                file_path, map_location=map_location, weights_only=False
                            )
                    except (ImportError, AttributeError) as err:
                        LOG.warning(f"safe_globals context manager not available: {err}")
                        # Fallback if safe_globals context manager is not available
                        return torch.load(file_path, map_location=map_location, weights_only=False)
                else:
                    # For older PyTorch versions
                    return torch.load(file_path, map_location=map_location)
            except Exception as e2:
                LOG.error(f"Failed to load model with compatibility mode: {e2}")
                raise RuntimeError(
                    f"Failed to load model with compatibility mode: {e2}. Original error: {original_error}"
                )
        else:
            # If it's another type of error, re-raise it
            raise


def set_random_seed(seed=42):
    """To ensure deterministic ordering across different machines and runs for reproducibility."""
    import os
    import random

    import numpy as np

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    # For complete determinism (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Python hash seed for consistent ordering in subprocesses (no-op in current
    # process; the interpreter already sampled PYTHONHASHSEED at startup).
    os.environ['PYTHONHASHSEED'] = str(seed)

"""
Utility functions for axelera.runtime examples.

This package contains YOLO-specific preprocessing and postprocessing utilities,
separating complex model-specific logic from Axelera runtime API usage.

The Axelera-specific helper functions (quantize, pad, depad, dequantize) are
intentionally kept in the example files themselves to make them self-contained
and easier to follow as tutorials.
"""

from . import yolo_utils

__all__ = ['yolo_utils']

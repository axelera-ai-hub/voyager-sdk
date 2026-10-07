#
# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
#
"""Shared plumbing for the Europa RVV C kernels."""

from pathlib import Path

import numpy as np

_HERE = Path(__file__).parent


def kernel_source(name: str) -> str:
    """C for one RVV kernel, prefixed with the shared fp32 math prelude.

    Every kernel's source is concatenated into one translation unit per model, so
    the prelude is include-guarded and safe to prepend to each kernel that needs it.
    """
    prelude = (_HERE / "_rvv_prelude.c").read_text()
    return f"{prelude}\n{(_HERE / f'{name}.c').read_text()}"


def unpacked_shape(packed_shape: tuple[int, ...]) -> tuple[int, ...]:
    """Packed buffer shape ``(..., d_last)`` -> scalar ``(..., d_last * PWORD)``."""
    from axelera.compiler.alto.compiler.device import PWORD

    return (*packed_shape[:-1], packed_shape[-1] * PWORD)


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Overflow-free sigmoid, mirroring ax_vsigmoid_f32m8 in _rvv_prelude.c.

    The naive 1/(1+exp(-x)) overflows for very negative x; the kernel keeps the
    exponential bounded by 1 and selects by sign, so emulation must too or it stops
    modelling the code it is meant to check.
    """
    z = np.exp(-np.abs(x))
    return np.where(x < 0, z / (1.0 + z), 1.0 / (1.0 + z))


def to_scalar(array: np.ndarray) -> np.ndarray:
    """Packed array ``(..., blocks, PWORD)`` -> scalar view ``(..., blocks * PWORD)``.

    Used in emulation, where the lane axis is materialized as a trailing dim.
    """
    return array.reshape((*array.shape[:-2], array.shape[-2] * array.shape[-1]))

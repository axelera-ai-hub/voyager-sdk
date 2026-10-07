# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""What a build checks about the compiler it is about to use, before it uses it."""

from __future__ import annotations


def require_alto_dtypes() -> None:
    """Fail with an actionable message when the compiler predates `alto.compiler.dtypes`.

    Every AxScript op module imports `axelera.compiler.alto.compiler.dtypes`, which exists
    only on compiler builds at or after the rename from `types`. On an older wheel the bare
    `ModuleNotFoundError` names a module path and nothing else, which reads as a broken recipe
    rather than an out-of-date environment, and it surfaces mid-build, after calibration.
    """
    import importlib.util

    if importlib.util.find_spec("axelera.compiler.alto.compiler.dtypes") is not None:
        return
    raise ModuleNotFoundError(
        "this compiler build has no `axelera.compiler.alto.compiler.dtypes`, which every "
        "AxScript op needs (older builds call it `types`). Use a compiler build that provides "
        "it. `pip show axelera-compiler` reports the build you are on."
    )


def require_axten() -> None:
    """Fail with an actionable message when the compiler ships no `axten`.

    The axten builder compiles a ViT or DINOv2 backbone through the shared transformer
    kernels in `axelera.compiler.axten`, which older compiler builds do not carry at all. The
    bare `ModuleNotFoundError` names a module path three frames deep and reads as a broken
    card rather than an out-of-date environment.
    """
    import importlib.util

    try:
        # find_spec raises rather than returning None when the PARENT package is missing,
        # which is the same "wrong environment" answer and gets the same message.
        if importlib.util.find_spec("axelera.compiler.axten") is not None:
            return
    except ModuleNotFoundError:
        pass
    raise ModuleNotFoundError(
        "this compiler build has no `axelera.compiler.axten`, which the axten vision builder "
        "needs to compile a ViT or DINOv2 backbone. Use a compiler build that provides it. "
        "`pip show axelera-compiler` reports the build you are on."
    )

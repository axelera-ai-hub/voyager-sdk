# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Custom alto ops for the LLM kernels.

A home for LLM-specific axops -- fused or specialised operators the decoder
kernels use that aren't in the core library (axelera.compiler.alto.axol.core).
Each op lives in its own module here and registers itself on import; importing
this package imports every (non-underscore) module, so all custom ops are
registered together.

Because registering an axop imports the alto compiler, import this package
lazily -- from inside a kernel build (like naive_impl.make_layer_kernel defers
its axol / A imports) -- not at module top level, so the float reference model
keeps importing without the compiler.

Add an op: drop ops/<name>.py in (it's auto-discovered) following the
axol.core pattern -- a public @sliced emitter that calls A.axop("llm.<name>",
...), a @register_axop("llm.<name>", ...) expansion into atomics, and a
@register_axop_emulation("llm.<name>", ...) numpy reference. Prefix helper
modules with _ to keep them out of auto-discovery.
"""

import importlib
import pkgutil


def _register_ops() -> None:
    """Import every (non-underscore) op module so each registers its axop."""
    for module in pkgutil.iter_modules(__path__):
        if not module.name.startswith("_"):
            importlib.import_module(f"{__name__}.{module.name}")


_register_ops()

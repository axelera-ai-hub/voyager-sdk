# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers: which accelerators these tools support, and what to call them.

One table for the build CLI, the run CLI and the page, so `--generation`'s choices, a
title and a build's core ceiling cannot drift apart.
"""

from __future__ import annotations

#: Supported hardware generations, and the name each one's product goes by. A generation
#: the compiler gains is added here with its product name -- never guessed from its own.
DEVICE_NAMES = {"OMEGA": "Metis", "EUROPA": "Europa"}


def supported_generation(generation: str) -> str:
    """`generation` upper-cased, or a ValueError naming what is supported."""
    generation = str(generation).upper()
    if generation not in DEVICE_NAMES:
        raise ValueError(
            f"unsupported hardware generation {generation!r}; this model supports "
            f"{', '.join(DEVICE_NAMES)}"
        )
    return generation


def device_name(generation: str) -> str:
    """The device an artifact was compiled for, as a person says it: "Metis", "Europa"."""
    return DEVICE_NAMES[supported_generation(generation)]


def num_ai_cores(generation: str) -> int:
    """AI cores a whole card of that generation has -- four on Omega, eight on Europa.

    From the compiler's hw_specs, the same number `compile_kernel` sizes each tower's
    memory share against (`cores / NUM_AI_CORES`), so a copy here could only disagree.
    """
    from axelera.compiler.alto.compiler.config import HardwareGeneration
    from axelera.compiler.alto.compiler.device.hw_specs import NUM_AI_CORES

    return int(NUM_AI_CORES[HardwareGeneration[supported_generation(generation)]])

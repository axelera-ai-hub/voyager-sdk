# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Shared helpers: which accelerators these tools support, and what to call them.

One table for the build CLI, the run CLI and the pipeline, so ``--generation``'s choices,
a title and a build's core ceiling cannot drift apart.
"""

from __future__ import annotations

#: Supported hardware generations, and the name each one's product goes by. A generation
#: the compiler gains is added here with its product name -- never guessed from its own.
DEVICE_NAMES = {"OMEGA": "Metis", "EUROPA": "Europa"}

#: What a DIRECT caller of this package gets when it names nothing: Omega (Metis) is the
#: shipped part. The shipped ENTRIES do not reach this -- each declares its own
#: ``DEFAULT_GENERATION`` -- but they now agree with it: every one picks Omega, so "when the
#: caller names nothing" gives the same answer above and below the entries. Much of their
#: recorded on-chip evidence was taken on Europa, which each model states for itself rather
#: than here. An artifact carries its generation into its ELF and the
#: runtime refuses a mismatch, so this is a real choice and not a label -- see
#: ``attached_generation``, which warns when a build targets a device this machine lacks.
DEFAULT_GENERATION = "OMEGA"


def supported_generation(generation: str) -> str:
    """``generation`` upper-cased, or a ValueError naming what is supported."""
    generation = str(generation).upper()
    if generation not in DEVICE_NAMES:
        raise ValueError(
            f"unsupported hardware generation {generation!r}; this model supports "
            f"{', '.join(DEVICE_NAMES)}"
        )
    return generation


def device_name(generation: str) -> str:
    """The device an artifact was compiled for, as a person says it."""
    return DEVICE_NAMES[supported_generation(generation)]


def attached_generation() -> str | None:
    """The generation of the board in this machine, or None if there is not one.

    An artifact carries the generation it was compiled for into its ELF and the runtime
    refuses a mismatch outright, so this is what lets a build say up front that it is
    targeting a device this machine does not have. Never raises: no driver, no board, or
    a runtime that cannot be imported all mean "nothing to compare against".
    """
    try:
        from axelera.runtime import Context

        devices = Context().list_devices()
    except Exception:  # noqa: BLE001 - runtime/driver unavailable is not an error here
        return None
    if not devices:
        return None
    marks = f"{devices[0].name} {getattr(devices[0], 'board_type', '')}".lower()
    return "EUROPA" if "europa" in marks else "OMEGA"


def num_ai_cores(generation: str) -> int:
    """AI cores a whole card of that generation has.

    From the compiler's hw_specs, the same number ``compile_kernel`` sizes each tower's
    memory share against (``cores / NUM_AI_CORES``), so a copy here could only disagree.
    """
    from axelera.compiler.alto.compiler.config import HardwareGeneration
    from axelera.compiler.alto.compiler.device.hw_specs import NUM_AI_CORES

    return int(NUM_AI_CORES[HardwareGeneration[supported_generation(generation)]])

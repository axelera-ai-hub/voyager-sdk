# Copyright Axelera AI, 2026
"""Clear `ceil_mode` on every average-pooling layer of a torch model.

WHAT IT LOOKS LIKE WITHOUT THE PASS
    The build stops on an `AvgPool2d` the codegen cannot emit, naming
    `ceil_mode`. `ceil_mode=True` asks the pooling window to round its output
    size up rather than down, which means the last window hangs off the edge
    of the feature map. The AIPU pooling unit has no such window.

WHAT THE PASS DOES
    Sets `ceil_mode = False` on every `torch.nn.AvgPool2d` in the model.

WHICH MODELS NEED IT
    torchvision-style ResNets and the timm classifiers built on them, which
    ship `ceil_mode=True` on a pool whose input size the flag never actually
    changes. Check yours before assuming it is harmless: on an input size
    where the height or width is not a whole number of windows, dropping
    `ceil_mode` drops the partial window, and the model then sees a slightly
    smaller feature map than it was trained on. This pass is safe for the
    input sizes the zoo configs declare, and it is on you to confirm it is
    safe for yours.

WHY THERE IS NO NUMERIC GUARD HERE
    Unlike the two ONNX rewrites in this directory, this one is not an
    identity: it is a deliberate, model-specific decision that the dropped
    partial window does not matter. There is nothing to assert equal, so the
    check is yours to make, by evaluating accuracy with and without it.
"""

from __future__ import annotations

from typing import Any

# What happens if you do not run this pass. Free text, not a category: a build
# that stops is the easy case, and the ones that quietly cost you something are
# the reason this line exists. The README table repeats it verbatim, and a test
# keeps the two equal.
WITHOUT_IT = 'the build stops: the codegen has no ceil-mode pooling window'

# Compiler backends this rewrite is valid for. It edits the torch graph
# before either backend ingests it, so both accept it.
SUPPORTS = ('axrelay', 'axgraph')

# Weight formats it operates on. A build in another format skips it.
SOURCE_KINDS = ('.pt2',)

# One line, printed by `apply` when the pass runs.
DESCRIPTION = 'AvgPool2d ceil_mode cleared, which the AIPU pooling unit cannot express'

# What retires this pass. Every pass here is a workaround with an end date.
RETIREMENT_CONDITION = 'the AIPU pooling unit implements a ceil-mode window'

# Set to the SDK version that absorbed the fix to start the one-release grace
# period: the pass is skipped with a warning instead of running.
DEPRECATED_IN = None


def run(model: Any) -> Any | None:
    """Clear `ceil_mode` on every `AvgPool2d` in `model`, in place.

    Returns `None` when no pool carried the flag, which is the runner's
    "nothing to rewrite" answer. This pass edits the model in place, so the
    object it hands back cannot by itself say whether anything changed.
    """
    import torch

    cleared = 0
    for module in model.modules():
        if isinstance(module, torch.nn.AvgPool2d) and getattr(module, 'ceil_mode', False):
            module.ceil_mode = False
            cleared += 1
    return model if cleared else None

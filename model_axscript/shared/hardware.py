# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Which hardware a build uses -- the part it targets, the host it calibrates on -- with no
dependencies.

A build compiles FOR a generation and shards across the AI cores that part has: Omega
(Metis) is a 4-core part, Europa an 8-core one. Three readers need that mapping and must agree
on it -- `generate.py` claims that many sub-devices at run time, and `shared/entry.py` and
`shared/vl_config.py` default a build to the whole part, one per family -- so it is written
down once, here rather than per reader: a build labelled EUROPA that spanned Omega's 4 cores
would be a claim about the part the binary does not meet.

Deliberately a leaf module importing nothing, for the reason `run_sizes.py` gives about
run-size keys. The obvious home for the mapping is `llm/generate.py`, but that module imports
torch and the compiler, and a caller asking only WHICH generations exist -- the front door
refusing a typo before it builds anything, a `--help` printing the choices, a test reading a
config -- should not have to install a numerical stack to be told two names. `default_cal_device`
is here for the same reason from the other side: it is the one place that needs torch to answer,
so it answers without it when there is none rather than making every config-reader carry one.
"""

from __future__ import annotations

#: AI cores a build spans per hardware generation, which is also the number of sub-devices a
#: run claims: Omega (Metis) has 4, Europa 8. "All of it" is the default either way -- both
#: towers of a VLM must be resident together, and giving a build fewer cores than the part has
#: does not make it fit anything smaller, only makes it slower. A caller who wants to leave
#: cores free for another workload names a smaller count.
#:
#: The generation names are the ones `models/qwen3vl/helpers.py::DEVICE_NAMES` defines and
#: `pipeline.build_artifacts` accepts.
CORES_PER_GENERATION = {"OMEGA": 4, "EUROPA": 8}


def default_cal_device() -> str:
    """The torch device post-training calibration should run on: CUDA when there is one.

    Resolved eagerly by both config classes, so every consumer sees a concrete device rather
    than one path defaulting and another raising. With no torch installed there is no CUDA to
    find either, so the answer is "cpu" rather than an ImportError: reading what a model
    declares is not a build, and the build itself fails on the first real torch call, naming
    torch, rather than here on a field it never used.
    """
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"

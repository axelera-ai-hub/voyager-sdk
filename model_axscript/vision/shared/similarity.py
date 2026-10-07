# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Model-agnostic L2-normalize + cosine helpers shared by the vision eval paths (kNN, VPR)."""

from __future__ import annotations

import numpy as np


def l2_normalize(f: np.ndarray, axis: int = 1, eps: float = 1e-9) -> np.ndarray:
    """Row-normalize `f` to unit L2 norm along `axis`."""
    f = np.asarray(f)
    return f / (np.linalg.norm(f, axis=axis, keepdims=True) + eps)


def cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(Q, D), (N, D) -> (Q, N) cosine similarity."""
    return l2_normalize(a) @ l2_normalize(b).T


def mean_cosine(a: np.ndarray, b: np.ndarray) -> float:
    """Mean per-row cosine between paired (N, D) arrays `a` and `b`."""
    return float(np.sum(l2_normalize(a) * l2_normalize(b), axis=1).mean())


def mean_pairwise_cosine(descriptors: np.ndarray) -> float:
    """Mean cosine over all DISTINCT pairs of rows in `descriptors`.

    The "are these the same place?" question, for a set of descriptors with no ground truth.
    Distinct from `mean_cosine`, which pairs row i of `a` with row i of `b` - passing the same
    array to that returns 1.0 whatever the inputs, which reads as a perfect match rather than
    as a bug. Needs at least two rows.
    """
    n = len(descriptors)
    if n < 2:
        raise ValueError(f"need at least 2 descriptors to compare, got {n}")
    sims = cosine_matrix(descriptors, descriptors)
    off_diagonal = ~np.eye(n, dtype=bool)
    return float(sims[off_diagonal].mean())


def descriptor_cosine_gap(device_descriptors, reference_descriptors) -> dict:
    """Per-descriptor device-versus-fp cosine, and the gap `1 - mean cosine`.

    A place-recognition descriptor has no readable meaning on its own, so what a single run of
    such a card can say is how far its descriptor sits from the reference's for the same image.
    Recall@k is the scored quantity and lives in the zoo's evaluator; this is the run path's
    fidelity line.
    """
    mean = mean_cosine(device_descriptors, reference_descriptors)
    return {"cosine_mean": mean, "gap": 1.0 - mean}

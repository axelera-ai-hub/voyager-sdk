# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Evaluation suite for registered model implementations (accuracy + performance)."""

from .metrics import (
    EvalContext,
    Metric,
    OnDevicePerplexityMetric,
    PerformanceMetric,
    PerplexityMetric,
)
from .scoring import fidelity, perplexity, token_nll
from .suite import DEFAULT_METRICS, EvalReport, run_evaluation

__all__ = [
    "DEFAULT_METRICS",
    "EvalContext",
    "EvalReport",
    "Metric",
    "OnDevicePerplexityMetric",
    "PerformanceMetric",
    "PerplexityMetric",
    "fidelity",
    "perplexity",
    "run_evaluation",
    "token_nll",
]

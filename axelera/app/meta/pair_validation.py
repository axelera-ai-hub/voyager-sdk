# Copyright Axelera AI, 2024
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

from ..eval_interfaces import PairValidationEvalSample
from .base import AggregationNotRequiredForEvaluation, AxTaskMeta


@dataclass(frozen=True)
class PairValidationMeta(AxTaskMeta):
    _results: List[np.ndarray] = field(default_factory=list, init=False)

    def add_result(self, result: np.ndarray) -> bool:
        '''Return True if the result is now full.'''
        if len(self._results) >= 2:
            raise ValueError("Both result1 and result2 already have data.")

        if result.ndim == 1:
            result = result.reshape(1, -1)

        self._results.append(result)
        return len(self._results) == 2

    @property
    def result1(self) -> np.ndarray:
        return self._results[0] if len(self._results) > 0 else np.array([])

    @property
    def result2(self) -> np.ndarray:
        return self._results[1] if len(self._results) > 1 else np.array([])

    def to_evaluation(self):
        return PairValidationEvalSample.from_numpy(self.result1, self.result2)

    def draw(self, draw):
        raise RuntimeError("Pair Verification does not support drawing")

    @classmethod
    def aggregate(cls, meta_list: List['PairValidationMeta']) -> 'PairValidationMeta':
        raise AggregationNotRequiredForEvaluation(cls)

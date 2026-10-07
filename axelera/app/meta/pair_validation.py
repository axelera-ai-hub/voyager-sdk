# Copyright Axelera AI, 2024
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, List

import numpy as np

from ..eval_interfaces import PairValidationEvalSample
from .base import AggregationNotRequiredForEvaluation, AxTaskMeta

if TYPE_CHECKING:
    from .classification import EmbeddingsMeta


@dataclass(frozen=True)
class PairValidationMeta(AxTaskMeta):
    results: List[np.ndarray] = field(default_factory=list)

    def add_result(self, result: np.ndarray) -> bool:
        '''Return True if the result is now full.'''
        if len(self.results) >= 2:
            raise ValueError("Both result1 and result2 already have data.")

        if result.ndim == 1:
            result = result.reshape(1, -1)

        self.results.append(result)
        return len(self.results) == 2

    @property
    def result1(self) -> np.ndarray:
        return self.results[0] if len(self.results) > 0 else np.array([])

    @property
    def result2(self) -> np.ndarray:
        return self.results[1] if len(self.results) > 1 else np.array([])

    def to_evaluation(self):
        return PairValidationEvalSample.from_numpy(self.result1, self.result2)

    def draw(self, draw):
        # raise RuntimeError("Pair Verification does not support drawing")
        pass

    @classmethod
    def aggregate(cls, meta_list: List['PairValidationMeta']) -> 'PairValidationMeta':
        raise AggregationNotRequiredForEvaluation(cls)

    @classmethod
    def decode(cls, data: dict[str, bytes | bytearray]) -> EmbeddingsMeta:
        buffer = data.get("data", b"")
        num_of_embeddings = data.get("num_of_embeddings", b"")
        embeddings = np.frombuffer(buffer, dtype=np.float32)
        num_of_embeddings = np.frombuffer(num_of_embeddings, dtype=np.int32)[0]
        # Reshape to (num_of_embeddings, M) where M is the vector length of each embedding
        meta = cls()
        meta.add_result(embeddings.reshape(num_of_embeddings, -1))
        return meta

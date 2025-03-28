# Copyright Axelera AI, 2025
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Union

import numpy as np
import torch
import torch.nn.functional as TF

from axelera import types
from axelera.app import gst_builder, logging_utils, meta
from axelera.app.operators import AxOperator, EvalMode

LOG = logging_utils.getLogger(__name__)


class MyClassificationEvalSample(types.BaseEvalSample):
    def __init__(self, class_ids: np.ndarray):
        self.class_ids = class_ids

    @property
    def data(self) -> Any:
        return self.class_ids


# Tutorial-8: Implement your own to_evaluation method
@dataclass(frozen=True)
class MyAxTaskMeta(meta.AxTaskMeta):
    class_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    scores: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))

    def __post_init__(self):
        if not isinstance(self.class_ids, np.ndarray):
            raise ValueError("class_ids must be a numpy array")
        if not isinstance(self.scores, np.ndarray):
            raise ValueError("scores must be a numpy array")
        if not self.class_ids.shape == self.scores.shape:
            raise ValueError("class_ids and scores must have the same shape")
        if not self.class_ids.dtype == np.int32:
            raise ValueError("class_ids must be a numpy array of int32")
        if not self.scores.dtype == np.float32:
            raise ValueError("scores must be a numpy array of float32")
        if self.class_ids.ndim != 1 or self.scores.ndim != 1:
            raise ValueError("class_ids and scores must be 1-D arrays")

    @classmethod
    def decode(cls, data: Dict[str, Union[bytes, bytearray]]) -> MyAxTaskMeta:
        # TODO: send something else from C++ to here
        scores = data.get("scores", b"")
        scores = np.frombuffer(scores, dtype=np.float32)
        classes = data.get("classes", b"")
        classes = np.frombuffer(classes, dtype=np.int32)
        model_meta = cls()
        model_meta.add_result(classes, scores)
        return model_meta

    def to_evaluation(self):
        if not (ground_truth := self.access_ground_truth()):
            raise ValueError("Ground truth is not set")

        from axelera.app.eval_interfaces import ClassificationGroundTruthSample

        if not isinstance(ground_truth, ClassificationGroundTruthSample):
            raise ValueError("Ground truth is not a ClassificationGroundTruthSample")

        return MyClassificationEvalSample(
            class_ids=self.class_ids,
        )


class TopKDecoderWithMyAxTaskMeta(AxOperator):
    k: int = 1
    largest: bool = True
    sorted: bool = True
    softmax: bool = False

    def _post_init(self):
        if self.eval_mode == EvalMode.EVAL:
            # This dict will be passed as "custom_config" to the evaluator.
            # It is useful for passing parameters from operators, but it is
            # not a strict requirement for implementing the evaluator and the operator
            self.register_validation_params(
                {
                    'top_k': self.k,
                }
            )
        # elif self.eval_mode == EvalMode.PAIR_EVAL:
        #   we have a path to do pair evaluation; it can be useful if your task is a recognition task outputting embeddings

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        pass

    def exec_torch(self, image, predict, axmeta):
        if self.softmax:
            predict = TF.softmax(predict, dim=1)
        top_scores, top_ids = torch.topk(
            predict, k=self.k, largest=self.largest, sorted=self.sorted
        )
        top_scores = top_scores.cpu().detach().numpy()[0]
        top_ids = top_ids.cpu().detach().numpy()[0]

        model_meta = MyAxTaskMeta(
            scores=top_scores.astype(np.float32),
            class_ids=top_ids.astype(np.int32),
        )

        axmeta.add_instance(self.task_name, model_meta)
        return image, predict, axmeta

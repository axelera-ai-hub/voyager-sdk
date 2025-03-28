# Copyright Axelera AI, 2025
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Union

import numpy as np
import torch

from axelera import types
from axelera.app import gst_builder, logging_utils, meta
from axelera.app.operators import AxOperator, PipelineContext

LOG = logging_utils.getLogger(__name__)


# Tutorial-7: Implement your own AxTaskMeta
@dataclass(frozen=True)
class MyPyClassificationMeta(meta.AxTaskMeta):
    class_ids: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))
    scores: np.ndarray = field(default_factory=lambda: np.array([], dtype=float))
    num_classes: np.ndarray = field(default_factory=lambda: np.array([], dtype=int))

    # Only required if we want to use the meta data in the evaluation interface
    def to_evaluation(self):
        if not (ground_truth := self.access_ground_truth()):
            raise ValueError("Ground truth is not set")

        from axelera.app.eval_interfaces import (
            ClassificationEvalSample,
            ClassificationGroundTruthSample,
        )

        if isinstance(ground_truth, ClassificationGroundTruthSample):
            pred_data = ClassificationEvalSample(num_classes=self.num_classes[0])
            pred_data.class_ids = self.class_ids
            pred_data.scores = self.scores
            return pred_data
        else:
            raise NotImplementedError(
                f"Ground truth type {type(ground_truth).__name__} is not supported"
            )

    # Only required if we want to use the meta data in the evaluation interface
    def get_result(self, index: int = 0):
        return self.class_ids, self.scores

    # We learn to visualize the results in a later tutorial
    def draw(self, draw):
        pass

    @classmethod
    def decode(cls, data: Dict[str, Union[bytes, bytearray]]) -> 'MyPyClassificationMeta':
        scores = data.get("scores_subtype", b"")
        scores = np.frombuffer(scores, dtype=np.float32)
        classes = data.get("classes_subtype", b"")
        classes = np.frombuffer(classes, dtype=np.int32)
        num_classes = data.get("num_classes_subtype", b"")
        num_classes = np.frombuffer(num_classes, dtype=np.int32)
        model_meta = cls(class_ids=classes, scores=scores, num_classes=num_classes)
        return model_meta


class TopKDecoderWithMySimplifiedGstPluginAndMyAxTaskMeta(AxOperator):
    def _post_init(self):
        pass

    # Only required because we want to pass num_classes to the evaluation interface
    # We do this by looping num_classes through GStreamer back to Python
    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        where: str,
        compiled_model_dir: Path,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, where, compiled_model_dir
        )
        self.num_classes = model_info.num_classes

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        gst.decode_muxer(
            lib='libdecode_tu7.so',
            options=f'meta_key:{str(self.task_name)};' f'num_classes:{self.num_classes};',
        )

    def exec_torch(self, image, predict, axmeta):
        top_scores, top_ids = torch.topk(predict, k=1)
        top_scores = top_scores.cpu().detach().numpy()[0]
        top_ids = top_ids.cpu().detach().numpy()[0]

        model_meta = MyPyClassificationMeta(
            num_classes=np.array([self.num_classes]),
            class_ids=top_ids,
            scores=top_scores,
        )

        axmeta.add_instance(self.task_name, model_meta)
        return image, predict, axmeta

# Copyright Axelera AI, 2024
# Metadata for classifier
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union

import numpy as np

from .. import plot_utils
from .base import AggregationNotRequiredForEvaluation, AxTaskMeta, MetaObject

if TYPE_CHECKING:
    from .. import display


class ClassifiedObject(MetaObject):
    @property
    def box(self):
        return self._meta.get_result(self._index)[2]

    @property
    def score(self):
        return self._meta.get_result(self._index)[1]

    @property
    def class_id(self):
        return self._meta.get_result(self._index)[0]


@dataclass(frozen=True)
class ClassificationMeta(AxTaskMeta):
    """Metadata for classification task
    class_ids, scores, and boxes are lists of lists. The inner lists correspond
    to the Top-K results for a single ROI, and the outer lists contain the results
    for different ROIs in a single image. To add new results to the metadata, use
    the add_result method, which allows for easy management of the data. get_result
    method allows for easy retrieval of the data of a single ROI for a specified index.
    For measurements, it is the special case where an image is a single ROI, but still
    keeps the class_ids and scores as lists of lists, with the outer list having a length
    of 1.
    """

    Object: ClassVar[MetaObject] = ClassifiedObject

    _class_ids: list[list[int]] = field(default_factory=list, init=False)
    _scores: list[list[float]] = field(default_factory=list, init=False)
    labels: Optional[list] = field(default_factory=lambda: None, repr=False)
    num_classes: Optional[int] = field(default_factory=lambda: None, repr=False)
    extra_info: Union[Dict[str, Any], MappingProxyType] = field(default_factory=dict)

    def add_result(
        self,
        class_ids: Union[int, list[int], np.ndarray],
        scores: Union[float, list[float], np.ndarray],
    ):
        """add classification result of a single ROI."""
        if isinstance(class_ids, int) or isinstance(class_ids, float):
            class_ids = [class_ids]
        elif isinstance(class_ids, np.ndarray):
            class_ids = class_ids.flatten().tolist()
        elif not isinstance(class_ids, list):
            raise ValueError(
                f"class_ids must be an int, list or numpy array, got {type(class_ids)}"
            )
        if len(class_ids) > 0 and not isinstance(class_ids[0], int):
            raise TypeError(f"class_ids must be a list of int, got {type(class_ids[0])}")
        class_ids = [int(item) for item in class_ids]

        if isinstance(scores, float):
            scores = [scores]
        elif isinstance(scores, np.ndarray):
            scores = scores.flatten().tolist()
        elif not isinstance(scores, list):
            raise ValueError(f"scores must be a float, list or numpy array, got {type(scores)}")
        if len(scores) > 0 and not isinstance(scores[0], (int, float)):
            raise TypeError(f"scores must be a list of int or float, got {type(scores[0])}")

        assert len(class_ids) == len(
            scores
        ), f"class_ids and scores must have the same length, got {len(class_ids)} and {len(scores)}"

        self._class_ids.append(class_ids)
        self._scores.append(scores)

    def get_result(self, index: int = 0):
        """Retrieve classification result of a single ROI."""
        assert index < len(
            self._class_ids
        ), f"Index out of range, got {index}, but the length of class_ids is {len(self._class_ids)}"
        return self._class_ids[index], self._scores[index]

    def __len__(self):
        return len(self._class_ids)

    def transfer_data(self, other: ClassificationMeta):
        """Transfer data from another ClassificationMeta instance without creating intermediate copies."""
        if not isinstance(other, ClassificationMeta):
            raise TypeError("other must be an instance of ClassificationMeta")
        self._class_ids.extend(other._class_ids)
        self._scores.extend(other._scores)

    def draw(self, draw: display.Draw):
        boxes = [None] * len(self._class_ids)
        if self.master_meta_name:
            boxes = self.get_master_meta().boxes[self.subframe_index]
            if len(boxes) == 4 and len(self._class_ids) == 1:
                boxes = [boxes]

        for i, box in enumerate(boxes):
            if box is None:
                # If no boxes, use the entire image with a small indent
                box = [5, 5, draw.canvas_size[0] - 5, draw.canvas_size[1] - 5]

            score = self._scores[i][0]
            class_id = self._class_ids[i][0]
            self._draw_single_box(draw, box, class_id, score)

    def _draw_single_box(self, draw, box, class_id, score):
        try:
            _ = box[0][0]
            box = box[0]
        except (TypeError, IndexError):
            pass

        # assign image size if box w/h is not provided
        if box[2] == -1:
            box[2] = draw.canvas_size[0]
        if box[3] == -1:
            box[3] = draw.canvas_size[1]

        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        if isinstance(class_id, str):
            color = plot_utils.get_color(0)
            txt = class_id
        else:
            class_id = int(class_id)
            color = plot_utils.get_color(class_id)
            if self.labels:
                try:
                    txt = f'{self.labels(class_id).name}'
                except TypeError:
                    txt = f'{self.labels[class_id]}'
            else:
                txt = f'{class_id}'
        txt += f' {score:.2f}'

        draw.labelled_box(p1, p2, txt, color)

    def to_evaluation(self):
        if not (ground_truth := self.access_ground_truth()):
            raise ValueError("Ground truth is not set")

        from ..eval_interfaces import ClassificationEvalSample, ClassificationGroundTruthSample

        if isinstance(ground_truth, ClassificationGroundTruthSample):
            pred_data = ClassificationEvalSample(num_classes=self.num_classes)
            if len(self._class_ids) == 1:
                pred_data.class_ids = self._class_ids[0]
                pred_data.scores = self._scores[0]
            else:
                raise ValueError(f"Expected only one class_id, but got {len(self._class_ids)}")

            return pred_data
        else:
            raise NotImplementedError(
                f"Ground truth type {type(ground_truth).__name__} is not supported"
            )

    @classmethod
    def decode(cls, data: Dict[str, Union[bytes, bytearray]]) -> 'ClassificationMeta':
        scores = data.get("scores", b"")
        scores = np.frombuffer(scores, dtype=np.float32)
        classes = data.get("classes", b"")
        classes = np.frombuffer(classes, dtype=np.int32)
        model_meta = cls()
        model_meta.add_result(classes, scores)
        return model_meta

    @classmethod
    def aggregate(cls, meta_list: List['ClassificationMeta']) -> 'ClassificationMeta':
        raise AggregationNotRequiredForEvaluation(cls)

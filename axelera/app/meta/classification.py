# Copyright Axelera AI, 2023
# Metadata for classifier
from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union

import numpy as np

from .. import exceptions, plot_utils
from .base import (
    AggregationNotRequiredForEvaluation,
    AxTaskMeta,
    MetaObject,
    class_as_label,
    safe_label_format,
)

if TYPE_CHECKING:
    from .. import display

BOX_INDENT = 5


class ClassifiedObject(MetaObject):
    '''Access a classification result using a class interface.

    The default object returns the top-1 result, but the other
    results can be accessed using the property `all_results`.

    For example:

      stream = create_inference_stream(network="resnet18-imagenet", sources=sys.argv[1:])
      for result in stream:
          m = result.classifications[0]
          print(f"Image {result.meta.image_id} is a {m.label.name} with {m.score:.2f}% confidence")
          for topn, x in enumerate(m.topk[1:], 1):
              print(f"  or alternative {topn}: {x.label.name} with {x.score:.2f}%")

    This objects acts as a view of the metadata to avoid copying data.

    Args:
        meta: The metadata containing meta for at least this object
        index: The index of this object in the metadata's fields.
        subindex: The subindex of this object.
    '''

    __slots__ = ('_subindex',)

    def __init__(self, meta, index: int, subindex: int = 0):
        super().__init__(meta, index)
        self._subindex = subindex

    def _get_meta_result_element(self, element_index: int):
        result = self._meta.get_result(self._index)
        return (
            result[element_index][self._subindex]
            if result and len(result) > element_index
            else None
        )

    @property
    def box(self):
        '''Return the bounding box of the classified object with the highest score as a list [x1, y1, x2, y2].'''
        return self._get_meta_result_element(2)

    @property
    def score(self):
        return self._get_meta_result_element(1)

    @property
    def class_id(self):
        return self._get_meta_result_element(0)

    @property
    def topk(self) -> list[ClassifiedObject]:
        '''Return all classification results for this object as a list of ClassifiedObject instances.

        This is useful for accessing the top-K results.
        '''
        return [
            ClassifiedObject(self._meta, self._index, n)
            for n in range(len(self._meta.get_result(self._index)[0]))
        ]


@dataclass(frozen=True)
class EmbeddingsMeta(AxTaskMeta):
    """Metadata for embeddings tasks"""

    embedding: list[list[float]] = field(default_factory=list, init=False)

    def add_results(self, data: Union[list[float], np.ndarray]):
        if isinstance(data, np.ndarray):
            self.embedding.append(data.tolist())
        else:
            self.embedding.extend(data)

    def draw(self, draw: display.Draw):
        raise exceptions.NotSupportedForTask("EmbeddingsMeta", "draw")

    @classmethod
    def decode(cls, data: Dict[str, Union[bytes, bytearray]]) -> EmbeddingsMeta:
        buffer = data.get("data", b"")
        num_of_embeddings = data.get("num_of_embeddings", b"")
        embeddings = np.frombuffer(buffer, dtype=np.float32)
        num_of_embeddings = np.frombuffer(num_of_embeddings, dtype=np.int32)[0]
        # Reshape to (num_of_embeddings, M) where M is the vector length of each embedding
        meta = cls()
        meta.add_results(embeddings.reshape(num_of_embeddings, -1).tolist())
        return meta

    def to_evaluation(self):
        from ..eval_interfaces import ReIdEvalSample

        return ReIdEvalSample(embedding=self.embedding)


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
        # Make sure we have scores for this index
        if index < len(self._scores):
            return self._class_ids[index], self._scores[index]
        return self._class_ids[index], None

    def __len__(self):
        return len(self._class_ids)

    def transfer_data(self, other: ClassificationMeta):
        """Transfer data from another ClassificationMeta instance without creating intermediate copies."""
        if not isinstance(other, ClassificationMeta):
            raise TypeError("other must be an instance of ClassificationMeta")
        self._class_ids.extend(other._class_ids)
        self._scores.extend(other._scores)
        self.extra_info.update(other.extra_info)

    def draw(self, draw: display.Draw):
        if draw.options.show_bounding_boxes is False:
            return
        boxes = [None] * len(self._class_ids)
        if self.master_meta_name:
            boxes = self.get_master_meta().boxes[self.subframe_index]
            if len(boxes) == 4 and len(self._class_ids) == 1:
                boxes = [boxes]

        for i, box in enumerate(boxes):
            if box is None:
                # If no boxes, use the entire image with a small indent
                box = [5, 5, draw.image_size[0] - BOX_INDENT, draw.image_size[1] - BOX_INDENT]

            if i < len(self._class_ids):
                score = self._scores[i][0] if self._scores[i] else 0
                class_id = self._class_ids[i][0] if self._class_ids[i] else 0
                self._draw_single_box(
                    draw,
                    box,
                    class_id,
                    score,
                    self.task_render_config.show_labels,
                    self.task_render_config.show_annotations,
                )

    def _draw_single_box(
        self, draw, box, class_id, score, show_labels=True, show_annotations=True
    ):
        if show_annotations:
            try:
                _ = box[0][0]
                box = box[0]
            except (TypeError, IndexError):
                pass

            # assign image size if box w/h is not provided
            if box[2] == -1:
                box[2] = draw.image_size[0] - BOX_INDENT
            if box[3] == -1:
                box[3] = draw.image_size[1] - BOX_INDENT

            p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        else:
            p1, p2 = ((0, 0), (0, 0))

        cmap = draw.options.bbox_class_colors
        fmt = safe_label_format(draw.options.bbox_label_format)
        if show_labels:
            if isinstance(class_id, str):
                color = cmap.get(0, plot_utils.get_color(0))
                # blank format disables label - see display.Options
                txt = class_id if fmt != "" else ""
            else:
                if isinstance(class_id, list):
                    class_id = class_id[0]
                class_id = int(class_id)

                color = cmap.get(class_id, plot_utils.get_color(class_id))
                label = class_as_label(self.labels, class_id)
                # We can only display score as a percent with softmax enabled
                softmax = self.extra_info.get("softmax", False)
                scorep = score * 100 if softmax else score
                percent = '%' if softmax else ''
                txt = fmt.format(label=label, score=score, scorep=scorep, scoreunit=percent)
        else:
            color = cmap.get(0, plot_utils.get_color(0))
            txt = ""

        if show_annotations or show_labels:
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

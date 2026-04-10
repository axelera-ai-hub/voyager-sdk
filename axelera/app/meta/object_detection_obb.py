# Copyright Axelera AI, 2023
# Oriented bounding boxes for object detection task
from __future__ import annotations

import atexit
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union

import numpy as np

from axelera import types

from .. import logging_utils, utils
from ..model_utils.box import convert
from .base import AxTaskMeta, MetaObject, draw_bounding_boxes
from .gst_decode_utils import decode_bbox_obb

if TYPE_CHECKING:
    from .. import display

LOG = logging_utils.getLogger(__name__)

total_num_detections_obb = None


def _print_total_num_detections_obb():
    LOG.debug(f"Total number of OBB detections: {total_num_detections_obb}")


class DetectedObjectOBB(MetaObject):
    @property
    def box(self):
        return self._meta.boxes[self._index]

    @property
    def score(self):
        return self._meta.scores[self._index]

    @property
    def class_id(self):
        return self._meta.class_ids[self._index]


@dataclass(frozen=True)
class ObjectDetectionMetaOBB(AxTaskMeta):
    """Metadata for oriented bounding box object detection task
    Boxes format is XYWHR or XYXYXYXY in the original pixel coordinate.
    The data will be readable-only after initialization."""

    Object: ClassVar[MetaObject] = DetectedObjectOBB

    boxes: np.ndarray
    scores: np.ndarray
    class_ids: np.ndarray
    labels: Optional[Union[tuple, utils.FrozenIntEnum]] = None
    extra_info: Union[Dict[str, Any], MappingProxyType] = field(default_factory=dict)

    def __post_init__(self):
        assert (
            len(self.boxes) == len(self.scores) == len(self.class_ids)
        ), "boxes, scores, and class_ids must have the same length {} {} {}"
        assert isinstance(self.boxes, np.ndarray), "boxes must be a numpy array"
        assert isinstance(self.scores, np.ndarray), "scores must be a numpy array"
        assert isinstance(self.class_ids, np.ndarray), "class_ids must be a numpy array"
        assert (
            self.boxes.shape[1] == 5 or self.boxes.shape[1] == 8
        ), "OBB format: expected 5 or 8 values per box"
        self.boxes.flags.writeable = False
        self.scores.flags.writeable = False
        self.class_ids.flags.writeable = False
        global total_num_detections_obb
        if total_num_detections_obb is None:
            atexit.register(_print_total_num_detections_obb)
            total_num_detections_obb = 0
        total_num_detections_obb += len(self.boxes)

    @classmethod
    def create_immutable_meta(
        cls,
        boxes,
        scores,
        class_ids,
        labels=None,
        extra_info={},
        make_extra_info_mutable=False,
    ):
        """A helper function to create an immutable instance of ObjectDetectionMetaOBB
        with expected data types. Be careful to enable make_extra_info_mutable,
        since the extra_info can be modified by users."""
        if labels:
            if not isinstance(labels, utils.FrozenIntEnumMeta):
                safe_labels = tuple(labels)
            else:
                safe_labels = labels
        else:
            safe_labels = None
        if not make_extra_info_mutable and extra_info:
            safe_extra_info = MappingProxyType(extra_info)
        else:
            safe_extra_info = extra_info

        return cls(
            boxes=np.array(boxes, np.float32),
            scores=np.array(scores),
            class_ids=np.array(class_ids, np.int32),
            labels=safe_labels,
            extra_info=safe_extra_info,
        )

    def __len__(self):
        return len(self.class_ids)

    @property
    def bboxes(self):
        return [self.boxes, self.scores, self.class_ids]

    def empty(self):
        return len(self.class_ids) == 0

    def to_xyxyxyxy(self):
        if self.boxes.shape[1] == 8:  # OBB format: 8 coordinates (4 corners)
            return self.boxes
        elif self.boxes.shape[1] == 5:  # OBB format: 5 coordinates (x, y, w, h, r)
            return convert(self.boxes, types.BoxFormat.XYWHR, types.BoxFormat.XYXYXYXY)
        else:
            raise ValueError(f"Unknown OBB format with {self.boxes.shape[1]} coordinates")

    def to_xywhr(self):
        if self.boxes.shape[1] == 5:  # OBB format: 5 coordinates (x, y, w, h, r)
            return self.boxes
        elif self.boxes.shape[1] == 8:  # OBB format: 8 coordinates (4 corners)
            return convert(self.boxes, types.BoxFormat.XYXYXYXY, types.BoxFormat.XYWHR)
        else:
            raise ValueError(f"Unknown OBB format with {self.boxes.shape[1]} coordinates")

    def draw(self, draw: display.Draw):
        if draw.options.show_bounding_boxes:
            draw_bounding_boxes(
                self,
                draw,
                self.task_render_config.show_labels,
                self.task_render_config.show_annotations,
            )

    def to_evaluation(self):
        if not (ground_truth := self.access_ground_truth()):
            raise ValueError("Ground truth is not set")

        from .. import eval_interfaces

        if isinstance(ground_truth, eval_interfaces.ObjDetGroundTruthSample):
            if len(self.class_ids) > 0:
                # For evaluation, convert OBB to XYWHR format
                prediction = eval_interfaces.ObjDetEvalSample.from_numpy(
                    boxes=np.copy(self.to_xywhr()),
                    labels=np.copy(self.class_ids),
                    scores=np.copy(self.scores),
                )
            else:
                prediction = eval_interfaces.ObjDetEvalSample()
            return prediction
        else:
            raise NotImplementedError(
                f"Ground truth is {type(ground_truth)} which is not supported yet"
            )

    @classmethod
    def decode(cls, data: Dict[str, Union[bytes, bytearray]]) -> 'ObjectDetectionMetaOBB':
        # Decode OBB format - expecting 5 coordinates per box
        boxes = decode_bbox_obb(data)
        if boxes.shape[1] != 5 and boxes.shape[1] != 8:
            raise ValueError(f"Expected boxes with 5 or 8 coordinates, got {boxes.shape[1]}")
        if buffer := data.get('classes', b''):
            classes = np.frombuffer(buffer, dtype=np.int32)
        else:
            classes = np.zeros(boxes.shape[0], dtype=np.int32)

        if scores := data.get('scores', b''):
            scores = np.frombuffer(scores, dtype=np.float32)
        else:
            scores = np.ones(boxes.shape[0], dtype=np.float32)

        return cls(boxes=boxes, scores=scores, class_ids=classes)

    @classmethod
    def aggregate(cls, meta_list: List['ObjectDetectionMetaOBB']) -> 'ObjectDetectionMetaOBB':
        if not all(isinstance(meta, cls) for meta in meta_list):
            raise TypeError(f"All metas must be instances of {cls.__name__}")

        return cls(
            boxes=np.concatenate([meta.boxes for meta in meta_list]),
            scores=np.concatenate([meta.scores for meta in meta_list]),
            class_ids=np.concatenate([meta.class_ids for meta in meta_list]),
            labels=meta_list[0].labels,
        )

# Copyright Axelera AI, 2024
# Metadata for keypoint and landmark detection
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional, Union

import numpy as np
from typing_extensions import Self

from axelera import types

from .. import display, logging_utils, plot_utils
from ..model_utils import box
from .base import AxTaskMeta, MetaObject, draw_bounding_boxes
from .gst_decode_utils import decode_bbox

LOG = logging_utils.getLogger(__name__)


class KeypointObject(MetaObject):
    @property
    def keypoints(self):
        return self._meta.keypoints[self._index]

    @property
    def score(self):
        return self._meta.scores[self._index]


class KeypointObjectWithBbox(KeypointObject):
    @property
    def box(self):
        return self._meta.boxes[self._index]


@dataclass(frozen=True)
class KeypointDetectionMeta(AxTaskMeta):
    """Base class for keypoint detection metadata.
    Keypoint detection can be categorized into the top-down and bottom-up methods.
    Top-down methods first find a specific object such as person, and then detect
    keypoints on the object, while bottom-up methods find all keypoints first and
    then group them into individuals.

    To tailor the data structures and methods to fit the specific needs of each
    method, we subclass this class into TopDownKeypointDetectionMeta and
    BottomUpKeypointDetectionMeta.
    """

    Object: ClassVar[MetaObject] = KeypointObject

    extra_info: Dict[str, Any] = field(default_factory=dict)

    def __len__(self):
        raise NotImplementedError("Implement this method to create keypoint objects")

    def to_evaluation(self):
        raise NotImplementedError(
            "Each keypoint detection model should implement this according to your data format"
        )

    def add_result(
        self,
        keypoints: Union[tuple, np.ndarray],
        boxes: Optional[np.ndarray] = None,
        scores: Optional[float] = None,
    ):
        raise NotImplementedError(
            f"Implement this method if you want to add results object-by-object"
        )


@dataclass(frozen=True)
class TopDownKeypointDetectionMeta(KeypointDetectionMeta):
    """Metadata for top-down keypoint detection task.
    - keypoints: a list of np.ndarray, shape=(K, 2 or 3), where K is the number of keypoints,
      and the first 2 are the x, y, and the last one is the optional visibility.
    - boxes: a list of np.ndarray, shape=(4,), where 4 is the x1, y1, x2, y2 of the box
      bounding the object.
    - scores: a list of float, where length is the number of objects.
    """

    _keypoints: List[np.ndarray] = field(default_factory=list)
    _boxes: List[np.ndarray] = field(default_factory=list)
    _scores: List[float] = field(default_factory=list)

    def __len__(self):
        return len(self._keypoints)

    def add_result(
        self,
        keypoints: Union[tuple, np.ndarray],
        boxes: Optional[np.ndarray] = None,
        scores: Optional[float] = None,
    ):
        if isinstance(keypoints, tuple):
            if len(keypoints) not in (2, 3) or not all(
                isinstance(k, (int, float)) for k in keypoints
            ):
                raise ValueError(
                    "keypoints must be a tuple of two or three int or float values (x, y) or (x, y, score)"
                )
            keypoint_array = np.array(keypoints).reshape(1, -1)
        elif isinstance(keypoints, np.ndarray):
            if (
                keypoints.ndim != 2
                or keypoints.shape[1] not in (2, 3)
                or not np.issubdtype(keypoints.dtype, np.number)
            ):
                raise ValueError(
                    "keypoints must be a 2D numpy array with shape (N, 2) or (N, 3) containing int or float values"
                )
            keypoint_array = keypoints
        else:
            raise ValueError("keypoints must be either a tuple or a numpy array")

        if len(self._keypoints) > 0 and keypoint_array.shape != self._keypoints[0].shape:
            raise ValueError("keypoints must be of the same shape as the existing ones")

        if boxes is not None:
            if not isinstance(boxes, np.ndarray) or boxes.shape != (4,):
                raise ValueError("boxes must be a 1D numpy array with shape (4,)")
            self._boxes.append(boxes)
            if scores is not None:
                if not np.isscalar(scores):
                    raise ValueError("scores must be a single scalar value")
                self._scores.append(scores)

        self._keypoints.append(keypoint_array)

    @property
    def keypoints(self):
        return self._keypoints

    @property
    def boxes(self):
        return self._boxes

    @property
    def scores(self):
        return self._scores

    def to_evaluation(self):
        keypoints = np.stack(self._keypoints, axis=0)
        boxes = np.stack(self._boxes, axis=0)
        scores = np.array(self._scores)
        raise NotImplementedError("Evaluation logic for top-down keypoint detection")

    def xyxy(self):
        if self._boxes:
            return np.stack(self._boxes, axis=0)
        else:
            return np.zeros((0, 4))

    def xywh(self):
        if self._boxes:
            boxes = self.xyxy()
            return box.convert(boxes, types.BoxFormat.XYXY, types.BoxFormat.XYWH)
        else:
            return np.zeros((0, 4))

    def ltwh(self):
        if self._boxes:
            boxes = self.xyxy()
            return box.convert(boxes, types.BoxFormat.XYXY, types.BoxFormat.LTWH)
        else:
            return np.zeros((0, 4))


@dataclass(frozen=True)
class BottomUpKeypointDetectionMeta(KeypointDetectionMeta):
    """Metadata for bottom-up keypoint detection task.
    - keypoints: np.ndarray, shape=(N, K, 2 or 3), where N is the number of objects,
      K is the number of keypoints, and the first 2 are the x, y, and the last one is
      the optional visibility.
    - boxes: np.ndarray, shape=(N, 4), where N is the number of objects, and 4 is
      the x1, y1, x2, y2 of the box bounding the object.
    - scores: np.ndarray, shape=(N,), where N is the number of objects.
    """

    keypoints: np.ndarray = field(default_factory=lambda: np.array([]))
    boxes: np.ndarray = field(default_factory=lambda: np.array([]))
    scores: np.ndarray = field(default_factory=lambda: np.array([]))

    def __post_init__(self):
        if isinstance(self.keypoints, list):
            raise TypeError("keypoints must be a numpy array, expected dimensions (N, K, 2 or 3)")
        if isinstance(self.boxes, list):
            raise TypeError("boxes must be a numpy array, expected dimensions (N, 4)")
        if isinstance(self.scores, list):
            raise TypeError("scores must be a numpy array, expected dimensions (N,)")
        if not (self.keypoints.shape[0] == self.boxes.shape[0] == self.scores.shape[0]):
            raise ValueError("First dimension of keypoints, boxes, and scores must be the same")
        if self.keypoints.shape[0] == 0:
            return
        if self.keypoints.ndim != 3:
            raise ValueError("keypoints must be a 3D numpy array")
        if self.boxes.ndim != 2:
            raise ValueError("boxes must be a 2D numpy array")
        if self.scores.ndim != 1:
            raise ValueError("scores must be a 1D numpy array")
        if self.keypoints.shape[2] != 2 and self.keypoints.shape[2] != 3:
            LOG.warning(
                f"keypoints typically have 2 or 3 fields, but got {self.keypoints.shape[2]}"
            )
        if self.boxes.shape[1] != 4:
            raise ValueError("boxes must have 4 fields")

    def __len__(self):
        return len(self.keypoints)

    @classmethod
    def from_list(cls, keypoints: List[np.ndarray], boxes: List[np.ndarray], scores: List[float]):
        return cls(
            keypoints=np.stack(keypoints, axis=0),
            boxes=np.stack(boxes, axis=0),
            scores=np.array(scores),
        )

    @classmethod
    def decode(cls, data: Dict[str, Union[bytes, bytearray]]) -> Self:
        values_per_kpt_to_datatype = {
            2: np.dtype([('x', np.int32), ('y', np.int32)]),
            3: np.dtype([('x', np.int32), ('y', np.int32), ('visibility', np.float32)]),
        }

        boxes = decode_bbox(data)
        kpts = data.get("kpts", b"")
        if kpts_shape := data.get("kpts_shape", b""):
            [kpts_per_box, values_per_kpt] = np.frombuffer(kpts_shape, dtype=np.int32)
        else:
            if len(boxes) == 0:
                raise ValueError("Cannot infer keypoints shape from empty boxes")
            values_per_kpt = 3
            kpts_per_box = len(kpts) / (
                boxes.shape[0] * values_per_kpt_to_datatype[values_per_kpt].itemsize
            )
        kpts = np.frombuffer(kpts, dtype=values_per_kpt_to_datatype[values_per_kpt])
        if values_per_kpt == 2 or cls != CocoBodyKeypointsMeta:
            kpts = np.vstack([kpts['x'], kpts['y']])
            kpts = kpts.T.astype(float).reshape(-1, kpts_per_box, 2)
        elif values_per_kpt == 3:
            kpts = np.vstack([kpts['x'], kpts['y'], kpts['visibility']])
            kpts = kpts.T.astype(float).reshape(-1, kpts_per_box, values_per_kpt)
        else:
            raise ValueError(f"Unsupported number of values per keypoint: {values_per_kpt}")
        scores = data.get('scores', b'')
        scores = np.frombuffer(scores, dtype=np.float32)
        return cls(keypoints=kpts, boxes=boxes, scores=scores)

    def to_evaluation(self):
        from ..eval_interfaces import KptDetEvalSample, KptDetGroundTruthSample

        if not (ground_truth := self.access_ground_truth()):
            raise ValueError("Ground truth is not set")
        if isinstance(ground_truth, KptDetGroundTruthSample):
            if len(self.keypoints) > 0:
                prediction = KptDetEvalSample.from_numpy(
                    boxes=self.xyxy(),
                    keypoints=self.keypoints,
                    scores=self.scores,
                )
            else:
                prediction = KptDetEvalSample.empty()
            return prediction
        else:
            raise NotImplementedError(
                f"Ground truth is {type(ground_truth)} which is not supported yet"
            )

    def add_result(
        self,
        keypoints: Union[tuple, np.ndarray],
        boxes: Optional[np.ndarray] = None,
        scores: Optional[float] = None,
    ):
        raise ValueError(
            f"{self.__class__.__name__} does not support adding results object-by-object"
        )

    def xyxy(self):
        return self.boxes

    def xywh(self):
        return box.convert(self.boxes, types.BoxFormat.XYXY, types.BoxFormat.XYWH)

    def ltwh(self):
        return box.convert(self.boxes, types.BoxFormat.XYXY, types.BoxFormat.LTWH)


_red = (255, 0, 0, 255)
_yellow = (255, 255, 0, 255)


class CocoBodyKeypointsMeta(BottomUpKeypointDetectionMeta):
    # COCO Body Keypoints order:
    # 0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
    # 5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
    # 9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
    # 13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle

    Object: ClassVar[MetaObject] = KeypointObjectWithBbox

    keypoints_shape = [17, 3]
    _point_bounds = [
        (15, 13, 11, 5, 6, 12, 14, 16),  # both legs + shoulders
        (5, 7, 9),  # left arm
        (6, 8, 10),  # right arm
    ]

    def draw(self, draw: display.Draw):
        draw_bounding_boxes(self, draw, show_class=False)
        if len(self.keypoints) == 0:
            return
        lines = []
        for det_pts in self.keypoints:
            for l in self._point_bounds:
                # Hardcoded value is visibility score, draw lines only if visibility score is big enough
                line = [det_pts[kp][:2] for kp in l if det_pts[kp][2] > 0.5]
                if len(line) > 1:
                    lines.append(line)

            if det_pts[5][2] > 0.5 and det_pts[6][2] > 0.5 and det_pts[0][2] > 0.5:
                lines.append(
                    [det_pts[0][:2], np.array(display.midpoint(det_pts[5][:2], det_pts[6][:2]))]
                )  # nose to middle of shoulders
            for x, y, v in det_pts:
                if v > 0.5:
                    draw.keypoint((x, y), _red, 6)
        if lines:
            draw.polylines(lines, False, _yellow, 2)


class FaceLandmarkLocalizationMeta(BottomUpKeypointDetectionMeta):
    # COCO Face Landmark Localization order:
    # 0: left_eye, 1: right_eye, 2: nose, 3: left_mouth, 4: right_mouth

    Object: ClassVar[MetaObject] = KeypointObjectWithBbox
    keypoints_shape = [5, 2]

    def to_evaluation(self):
        from ..eval_interfaces import GeneralSample, KptDetEvalSample

        if not (ground_truth := self.access_ground_truth()):
            raise ValueError("Ground truth is not set")
        if isinstance(ground_truth, GeneralSample):
            if len(self.keypoints) > 0:
                prediction = KptDetEvalSample.from_numpy(
                    boxes=self.xyxy(),
                    keypoints=self.keypoints,
                    scores=self.scores,
                )
            else:
                prediction = KptDetEvalSample()
            return prediction
        else:
            raise NotImplementedError(
                f"Ground truth is {type(ground_truth)} which is not supported yet"
            )

    def draw(self, draw: display.Draw):
        draw_bounding_boxes(self, draw, show_class=False)

        for det_pts in self.keypoints:
            for x, y in det_pts:
                draw.keypoint((x, y), _red, 6)


class FaceLandmarkTopDownMeta(TopDownKeypointDetectionMeta):
    # Face Landmark Localization order:
    # 0: left_eye, 1: right_eye, 2: nose, 3: left_mouth, 4: right_mouth
    # Top-down method, so no boxes if not with a cascade pipeline
    keypoints_shape = [5, 3]  # x, y, score

    def draw(self, draw: display.Draw):
        for det_pts in self._keypoints:
            for x, y, _ in det_pts:
                draw.keypoint((x, y), _red, 6)

    @classmethod
    def decode(cls, data: Dict[str, Union[bytes, bytearray]]) -> Self:

        if kpts_shape := data.get("kpts_shape", b""):
            kpts_shape = np.frombuffer(kpts_shape, dtype=np.int32)
        else:
            kpts_shape = cls.keypoints_shape
        if kpts := data.get("kpts", b""):
            kpts = np.frombuffer(
                kpts,
                dtype=np.dtype([('x', np.int32), ('y', np.int32), ('visibility', np.float32)]),
            )
            kpts = (
                np.vstack([kpts['x'], kpts['y'], kpts['visibility']])
                .T.astype(float)
                .reshape(-1, kpts_shape[0], kpts_shape[1])
            )
        else:
            kpts = np.zeros(
                kpts_shape[0] * boxes.shape[0],
                dtype=np.dtype([('x', np.int32), ('y', np.int32), ('visibility', np.float32)]),
            )

        if scores := data.get('scores', b''):
            scores = np.frombuffer(scores, dtype=np.float32)
        else:
            scores = np.ones(boxes.shape[0], dtype=np.float32)
        kpts[0, :, 2] = scores

        return cls(_keypoints=[np.array(x) for x in kpts.tolist()], _boxes=[], _scores=scores)

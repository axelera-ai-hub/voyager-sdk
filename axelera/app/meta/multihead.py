# Copyright Axelera AI, 2025
# Metadata for multihead task
from __future__ import annotations

from dataclasses import dataclass, field
import struct
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np

from axelera import types
from axelera.app.meta.segmentation import SegmentationMask

from .. import logging_utils, plot_utils
from ..model_utils import box as box_utils

# from ..torch_utils import torch
from .base import AxTaskMeta, draw_bounding_boxes
from .gst_decode_utils import decode_bbox

if TYPE_CHECKING:
    from .. import display

LOG = logging_utils.getLogger(__name__)


_red = (255, 0, 0, 255)

Wh = tuple[int, int]
XyXy = tuple[int, int, int, int]


def _translate_image_space_rect(bbox: XyXy, input_roi: XyXy, mask_size=(160, 160)) -> XyXy:
    x0, y0, x1, y1 = bbox
    input_w, input_h = input_roi[2] - input_roi[0], input_roi[3] - input_roi[1]
    scale_factor_x = input_w / mask_size[0]
    scale_factor_y = input_h / mask_size[1]
    x0 = int(x0 * scale_factor_x + input_roi[0])
    y0 = int(y0 * scale_factor_y + input_roi[1])
    x1 = int(x1 * scale_factor_x + input_roi[0])
    y1 = int(y1 * scale_factor_y + input_roi[1])
    return x0, y0, x1, y1


@dataclass(frozen=True)
class PoseInsSegMeta(AxTaskMeta):
    """Metadata for multihead task"""

    _masks: list = field(default_factory=list, init=False)
    _boxes: list = field(default_factory=list, init=False)
    _kpts: list = field(default_factory=list, init=False)
    _class_ids: list = field(default_factory=list, init=False)
    _scores: list = field(default_factory=list, init=False)
    seg_shape: Optional[Wh] = None
    labels: Optional[tuple] = None
    extra_info: Dict[str, Any] = field(default_factory=dict)

    keypoints_shape = [12, 3]  # TODO: Get this shape from yaml

    def add_result(
        self,
        mask_data: SegmentationMask,
        box: np.ndarray,
        kpts: np.ndarray,
        class_id: int,
        score: float,
    ):
        if not isinstance(mask_data, SegmentationMask):
            raise ValueError("mask must be a SegmentationMask tuple")
        if not isinstance(box, np.ndarray):
            raise ValueError("box must be a numpy array")
        mask = mask_data[-1]
        if mask.shape[0] == 0:
            return  # no detection
        if mask.ndim != 2:
            raise ValueError("mask must be a 2D numpy array")
        if box.shape != (4,):
            raise ValueError("box must be a 1D numpy array with shape (4,)")
        if not isinstance(class_id, int):
            raise ValueError("class_id must be an integer")
        if not np.isscalar(score):
            raise ValueError("score must be a single scalar value")
        if not isinstance(kpts, np.nparray):
            raise ValueError("keypoints must be numpy array")

        self._masks.append(mask_data)
        self._boxes.append(box)
        self._kpts.append(kpts)
        self._class_ids.append(class_id)
        self._scores.append(score)

    def get_result(self, index: int = 0):
        if index >= len(self._masks):
            raise IndexError(f"Index {index} out of range for masks")
        return (
            self.get_mask(index),
            self.boxes[index],
            self.kpts[index],
            self.class_ids[index],
            self.scores[index],
        )

    def transfer_data(self, other: PoseInsSegMeta):
        if not isinstance(other, PoseInsSegMeta):
            raise TypeError("other must be an instance of PoseInsSegMeta")
        object.__setattr__(self, 'seg_shape', other.seg_shape)
        self._masks.extend(other._masks)
        self._boxes.extend(other._boxes)
        self._kpts.extend(other._kpts)
        self._class_ids.extend(other._class_ids)
        self._scores.extend(other._scores)

    def add_results(
        self,
        masks_data: list[SegmentationMask],
        boxes: np.ndarray,
        kpts: np.ndarray,
        class_ids: np.ndarray,
        scores: np.ndarray,
    ):

        if not isinstance(masks_data, list):
            raise ValueError("mask must be a list of SegmentationMask tuple")

        if not isinstance(boxes, np.ndarray):
            raise ValueError("box must be a numpy array")
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("boxes must be a 2D numpy array with shape (N, 4)")

        if class_ids.ndim != 1:
            raise ValueError("class_ids must be a 1D numpy array")
        if scores.ndim != 1:
            raise ValueError("scores must be a 1D numpy array")
        if (
            class_ids.size != boxes.shape[0]
            or class_ids.size != len(masks_data)
            or class_ids.size != scores.size
        ):
            raise ValueError(
                f"Inconsistend data: class_ids={class_ids.size} scores={scores.size} boxes={boxes.shape[0]} masks={len(masks_data)}"
            )

        self._masks.extend(masks_data)
        self._boxes.extend(boxes)
        self._kpts.extend(kpts)
        self._class_ids.extend(class_ids)
        self._scores.extend(scores)

    @property
    def masks(self) -> list:
        return self._masks

    def get_mask(self, index):
        if index >= len(self._masks):
            raise IndexError(f"Index {index} out of range for masks")
        return self._masks[index]

    @property
    def boxes(self) -> np.ndarray:
        return np.array(self._boxes)

    @property
    def class_ids(self) -> np.ndarray:
        return np.array(self._class_ids)

    @property
    def scores(self) -> np.ndarray:
        return np.array(self._scores)

    @property
    def kpts(self) -> np.ndarray:
        return np.array(self._kpts)

    def xyxy(self):
        return self.boxes

    def xywh(self):
        return box_utils.convert(self.boxes, types.BoxFormat.XYXY, types.BoxFormat.XYWH)

    def ltwh(self):
        return box_utils.convert(self.boxes, types.BoxFormat.XYXY, types.BoxFormat.LTWH)

    def to_evaluation(self):
        from ..eval_interfaces import PoseInsSegGroundTruthSample

        if not (ground_truth := self.access_ground_truth()):
            raise ValueError("Ground truth is not set")
        if isinstance(ground_truth, PoseInsSegGroundTruthSample):
            masks = np.zeros((len(self.masks), *self.seg_shape), dtype=np.uint8)
            for i, mask_data in enumerate(self.masks):
                mask, bbox = mask_data[-1], mask_data[:4]
                x0, y0, x1, y1 = bbox
                masks[i, y0:y1, x0:x1] = mask

            prediction = PoseInsSegGroundTruthSample.from_numpy(
                boxes=self.xyxy(),
                scores=self.scores,
                keypoints=self.kpts,
                masks=masks,
                labels=self.class_ids,
            )
            return prediction
        else:
            raise NotImplementedError(
                f"Ground truth is {type(ground_truth)} which is not supported yet"
            )

    def draw(self, draw: display.Draw):
        if len(self.masks) == 0 or not self.task_render_config.show_annotations:
            return

        if draw.options.show_bounding_boxes:
            draw_bounding_boxes(
                self,
                draw,
                self.task_render_config.show_labels,
                self.task_render_config.show_annotations,
            )
        for i, cls in enumerate(self.class_ids):
            color = plot_utils.get_color(int(cls), alpha=125)
            if draw.options.show_keypoints:
                for x, y, v in self.kpts[i, :, :]:
                    if v > 0.5:
                        draw.keypoint((x, y), _red)

            if draw.options.show_segmentation:
                draw.segmentation_mask(self.get_mask(i), color)

    @classmethod
    def decode(cls, data: Dict[str, Union[bytes, bytearray]]) -> PoseInsSegMeta:
        boxes = decode_bbox(data)
        segment_shape = np.frombuffer(data.get('segment_shape'), dtype=np.uint64)
        segment_count = segment_shape[0]
        segment_shape = segment_shape[2:0:-1]
        if scores := data.get('scores', b''):
            scores = np.frombuffer(scores, dtype=np.float32)
        else:
            scores = np.ones(boxes.shape[0], dtype=np.float32)
        if classes := data.get('classes', b''):
            classes = np.frombuffer(classes, dtype=np.int32)
        else:
            classes = np.zeros(boxes.shape[0], dtype=np.int32)

        masks_base_box = (
            np.frombuffer(data['base_box'], dtype=np.int32).reshape(-1, 4).squeeze()
            if 'base_box' in data
            else None
        )
        if 'segment_maps' in data and 'segment_bboxs' in data:
            segment_maps = data.get('segment_maps')
            segment_bboxs = data.get('segment_bboxs')

            offset = 0
            segments = []
            for idx, bbox in enumerate(struct.iter_unpack('4i', segment_bboxs)):
                if idx == segment_count:
                    break
                x0, y0, x1, y1 = bbox
                width = x1 - x0
                height = y1 - y0
                size = width * height
                image_coords = _translate_image_space_rect(bbox, masks_base_box, segment_shape)

                segments.append(
                    (
                        *bbox,
                        *image_coords,
                        np.frombuffer(
                            segment_maps, dtype=np.uint8, count=size, offset=offset
                        ).reshape(height, width),
                    )
                )

                # Update the offset
                offset += size

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
        meta = cls(seg_shape=segment_shape)
        meta.add_results(segments, boxes, kpts, classes, scores)

        return meta

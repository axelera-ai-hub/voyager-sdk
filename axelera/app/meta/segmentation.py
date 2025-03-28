# Copyright Axelera AI, 2024
# Metadata for semantic segmentation task and instance segmentation task
from __future__ import annotations

from dataclasses import dataclass, field
import struct
from typing import TYPE_CHECKING, Any, Dict, Optional, Union

import numpy as np

from .. import logging_utils
from ..eval_interfaces import InstSegEvalSample, InstSegGroundTruthSample
from ..model_utils import box as box_utils
from ..plot_utils import colormap, coolwarm, get_rgba_cmap
from ..torch_utils import torch
from .base import AxTaskMeta, class_as_color, draw_bounding_boxes
from .gst_decode_utils import decode_bbox

if TYPE_CHECKING:
    from .. import display

LOG = logging_utils.getLogger(__name__)

XyXy = tuple[int, int, int, int]
SegmentationMask = tuple[int, int, int, int, int, int, int, int, np.ndarray]


def _translate_image_space_rect(bbox: XyXy, input_roi: XyXy, mask_size=(160, 160)) -> XyXy:
    x0, y0, x1, y1 = bbox
    input_w, input_h = input_roi[2] - input_roi[0], input_roi[3] - input_roi[1]
    longest_edge = max(input_w, input_h)
    scale_factor = longest_edge / mask_size[0]
    xoffset = int((longest_edge - input_w) / 2)
    yoffset = int((longest_edge - input_h) / 2)
    x0 = int(x0 * scale_factor - xoffset + input_roi[0])
    y0 = int(y0 * scale_factor - yoffset + input_roi[1])
    x1 = int(x1 * scale_factor - xoffset + input_roi[0])
    y1 = int(y1 * scale_factor - yoffset + input_roi[1])
    return x0, y0, x1, y1


@dataclass(frozen=True)
class SemanticSegmentationMeta(AxTaskMeta):
    """Metadata for semantic segmentation task"""

    shape: list[int, int, int]  # height, width, num_classes
    class_map: np.ndarray = field(default_factory=list)
    probabilities: np.ndarray = field(default_factory=list)
    # The raw (non-normalized) predicted result: for mmsegmentation data structure
    seg_logits: Optional[Union[torch.Tensor, np.ndarray]] = None
    labels: Optional[tuple] = None
    palette: Optional[list] = None
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not isinstance(self.shape, list):
            raise ValueError("shape must be a list")
        if len(self.shape) != 3:
            raise ValueError("shape must be a list with length 3")

    def to_evaluation(self):
        raise NotImplementedError("Haven't implemented in-house evaluator yet")

    def draw(self, draw: display.Draw):
        # TODO only class_map drawing is supported for now
        draw.class_map_mask(self.class_map, get_rgba_cmap(colormap, 125, True))

    @classmethod
    def decode(cls, data: Dict[str, Union[bytes, bytearray]]) -> SemanticSegmentationMeta:

        if 'data_shape' not in data:
            raise ValueError("Missing required 'data_shape' in input data")

        try:
            sizes = struct.iter_unpack('3i', data.get('data_shape'))
            shape = list(next(sizes))
        except ValueError as e:
            raise ValueError(f"Failed to parse data_shape: {e}")
        if 'segment_classes' in data:
            try:
                class_map = np.frombuffer(data.get('segment_classes'), dtype=np.int32).reshape(
                    shape[:2]
                )

                return cls(shape=shape, class_map=class_map, probabilities=np.empty((0,)))
            except ValueError as e:
                raise ValueError(f"Failed to parse segment_classes: {e}")
        elif 'segment_probabilities' in data:
            try:
                probs = np.frombuffer(data.get('segment_probabilities'), dtype=np.float).reshape(
                    shape
                )
                return cls(shape=shape, class_map=np.empty((0,)), probabilities=probs)
            except ValueError as e:
                raise ValueError(f"Failed to parse segment_probabilities: {e}")
        else:
            raise ValueError(f"Missing segment data")


@dataclass(frozen=True)
class InstanceSegmentationMeta(AxTaskMeta):
    """Metadata for instance segmentation task"""

    _masks: list = field(default_factory=list, init=False)
    _boxes: list = field(default_factory=list, init=False)
    _class_ids: list = field(default_factory=list, init=False)
    _scores: list = field(default_factory=list, init=False)
    labels: Optional[tuple] = None
    # offsets (x, y) for the masks
    _masks_base_box: list = field(default_factory=list, init=False)
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def add_result(
        self,
        mask: np.ndarray,
        box: np.ndarray,
        class_id: int,
        score: float,
        masks_base_box: XyXy | None = None,
    ):
        if not isinstance(mask, np.ndarray):
            raise ValueError("mask must be a numpy array")
        if not isinstance(box, np.ndarray):
            raise ValueError("box must be a numpy array")
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

        self._masks.append(mask)
        self._boxes.append(box)
        self._class_ids.append(class_id)
        self._scores.append(score)
        if masks_base_box is not None:
            self._masks_base_box.append(masks_base_box)

    def get_result(self, index: int = 0):
        if index >= len(self._masks):
            raise IndexError(f"Index {index} out of range for masks")
        return self.get_mask(index), self.boxes[index], self.class_ids[index], self.scores[index]

    def transfer_data(self, other: InstanceSegmentationMeta):
        if not isinstance(other, InstanceSegmentationMeta):
            raise TypeError("other must be an instance of InstanceSegmentationMeta")
        self._masks.extend(other._masks)
        self._boxes.extend(other._boxes)
        self._class_ids.extend(other._class_ids)
        self._scores.extend(other._scores)
        self._masks_base_box.extend(other._masks_base_box)

    def add_results(
        self,
        masks: list[np.ndarray],
        boxes: np.ndarray,
        class_ids: np.ndarray,
        scores: np.ndarray,
        masks_base_box: tuple[int, int, int, int] | None = None,
    ):
        if boxes.ndim != 2 or boxes.shape[1] != 4:
            raise ValueError("boxes must be a 2D numpy array with shape (N, 4)")
        if class_ids.ndim != 1:
            raise ValueError("class_ids must be a 1D numpy array")
        if scores.ndim != 1:
            raise ValueError("scores must be a 1D numpy array")

        self._masks.extend(masks)
        self._boxes.extend(boxes)
        self._class_ids.extend(class_ids)
        self._scores.extend(scores)
        if masks_base_box is not None:
            self._masks_base_box.extend(masks_base_box)

    @property
    def masks(self) -> list:
        return self._masks

    def get_mask(self, index: int = 0):
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

    def xyxy(self):
        return self.boxes

    def xywh(self):
        return box_utils.convert(self.boxes, 'xyxy', 'xywh')

    def ltwh(self):
        return box_utils.convert(self.boxes, 'xyxy', 'ltwh')

    def to_evaluation(self):
        if not (ground_truth := self.access_ground_truth()):
            raise ValueError("Ground truth is not set")
        if isinstance(ground_truth, InstSegGroundTruthSample):
            if len(self.masks) > 0:

                masks = np.zeros((len(self.masks), 160, 160), dtype=np.uint8)
                for i, mask_data in enumerate(self.masks):
                    mask, bbox = mask_data[-1], mask_data[:4]
                    x0, y0, x1, y1 = bbox
                    masks[i, y0:y1, x0:x1] = mask

                prediction = InstSegEvalSample.from_numpy(
                    boxes=self.xyxy(),
                    labels=self.class_ids,
                    scores=self.scores,
                    masks=masks,
                )
            else:
                prediction = InstSegEvalSample.empty()

            return prediction
        else:
            raise NotImplementedError(
                f"Ground truth is {type(ground_truth)} which is not supported yet"
            )

    def draw(self, draw: display.Draw):
        if len(self.masks) == 0:
            return
        for i, cls in enumerate(self.class_ids):
            color = class_as_color(self, draw, int(cls), alpha=125)
            draw.segmentation_mask(self.get_mask(i), color)
        draw_bounding_boxes(self, draw)

    @classmethod
    def decode(cls, data: Dict[str, Union[bytes, bytearray]]) -> InstanceSegmentationMeta:
        boxes = decode_bbox(data)
        segment_shape = np.frombuffer(data.get('segment_shape'), dtype=np.uint64)
        if scores := data.get('scores', b''):
            scores = np.frombuffer(scores, dtype=np.float32)
        else:
            scores = np.ones(boxes.shape[0], dtype=np.float32)
        if classes := data.get('classes', b''):
            classes = np.frombuffer(classes, dtype=np.int32)
        else:
            classes = np.zeros(boxes.shape[0], dtype=np.int32)

        if 'base_box' in data:
            masks_base_box = (
                np.frombuffer(data.get('base_box'), dtype=np.int32).reshape(-1, 4).squeeze()
            )
        else:
            masks_base_box = None
        if 'segment_maps' in data and 'segment_bboxs' in data:
            segment_maps = data.get('segment_maps')
            segment_bboxs = data.get('segment_bboxs')

            offset = 0
            segments = []
            for idx, bbox in enumerate(struct.iter_unpack('4i', segment_bboxs)):
                if idx == segment_shape[0]:
                    break

                x0, y0, x1, y1 = bbox
                width = x1 - x0
                height = y1 - y0
                size = width * height

                image_coords = _translate_image_space_rect(
                    bbox, masks_base_box, segment_shape[2:0:-1]
                )

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

        else:
            raise ValueError("Missing mask data")

        meta = cls()
        meta.add_results(segments, boxes, classes, scores)
        return meta

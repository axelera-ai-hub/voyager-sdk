# Copyright Axelera AI, 2024
# Evaluation Data Classes: These classes are interfaces to transform evaluation
# data from an AxTaskMeta and a ground truth in a custom dataset into a format
# suitable for evaluators. Thus, the design of evaluation data classes is closely
# aligned with both task metadata and evaluator architectures.
from __future__ import annotations

import abc
import dataclasses
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union

import numpy as np
from typing_extensions import Self

from axelera import types

from .torch_utils import torch


@dataclasses.dataclass
class GeneralSample(types.BaseEvalSample):
    annotations: Any

    @property
    def data(self) -> Union[Any, Dict[str, Any]]:
        return self.annotations


@dataclasses.dataclass
class ClassificationGroundTruthSample(types.BaseEvalSample):
    class_id: int

    @property
    def data(self) -> Union[Any, Dict[str, Any]]:
        return self.class_id


@dataclasses.dataclass
class ClassificationEvalSample(types.BaseEvalSample):
    """
    Data element for classification evaluations.
    """

    num_classes: int
    class_ids: List[int] = dataclasses.field(default_factory=list)
    scores: List[float] = dataclasses.field(default_factory=list)

    @property
    def data(self) -> Union[Any, Dict[str, Any]]:
        if len(self.class_ids) > 0:
            return dict(zip(self.class_ids, self.scores))
        else:
            empty_tensor = torch.as_tensor([])
            return empty_tensor


@dataclasses.dataclass
class ObjDetGroundTruthSample(types.BaseEvalSample):
    boxes: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    labels: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    img_id: Union[int, str] = ''

    @classmethod
    def from_numpy(cls, boxes: np.ndarray, labels: np.ndarray, img_id: Union[int, str]) -> Self:
        return cls(boxes, labels, img_id)

    @classmethod
    def from_torch(
        cls, boxes: torch.Tensor, labels: torch.Tensor, img_id: Union[int, str] = ''
    ) -> Self:
        return cls.from_numpy(boxes.cpu().numpy(), labels.cpu().numpy(), img_id)

    @classmethod
    def from_list(
        cls, boxes: List[List[float]], labels: List[int], img_id: Union[int, str] = ''
    ) -> Self:
        return cls.from_numpy(
            np.array(boxes, dtype=np.float32), np.array(labels, dtype=np.int32), img_id
        )

    @property
    def data(self) -> Union[Any, Dict[str, Any]]:
        return {
            'boxes': self.boxes,
            'labels': self.labels,
            'img_id': self.img_id,
        }


@dataclasses.dataclass
class ObjDetEvalSample(types.BaseEvalSample):
    """
    Data element for object detection evaluations.

    Attributes:
        boxes (np.ndarray): The bounding boxes in xyxy format.
        labels (np.ndarray): Class labels for each bounding box.
        scores (np.ndarray): Confidence scores for each bounding box.

    The lengths of boxes, scores, and labels must be the same.
    """

    boxes: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    labels: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    scores: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))

    @classmethod
    def from_numpy(cls, boxes: np.ndarray, labels: np.ndarray, scores: np.ndarray) -> Self:
        return cls(boxes, labels, scores)

    @classmethod
    def from_torch(cls, boxes: torch.Tensor, labels: torch.Tensor, scores: torch.Tensor) -> Self:
        return cls.from_numpy(boxes.cpu().numpy(), labels.cpu().numpy(), scores.cpu().numpy())

    @classmethod
    def from_list(cls, boxes: List[List[float]], labels: List[int], scores: List[float]) -> Self:
        return cls.from_numpy(
            np.array(boxes, dtype=np.float32),
            np.array(labels, dtype=np.int32),
            np.array(scores, dtype=np.float32),
        )

    @property
    def data(self) -> Union[Any, Dict[str, Any]]:
        return {
            'boxes': self.boxes,
            'scores': self.scores,
            'labels': self.labels,
        }


@dataclasses.dataclass
class KptDetGroundTruthSample(types.BaseEvalSample):
    boxes: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    keypoints: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    img_id: Union[int, str] = ''

    @classmethod
    def from_torch(
        cls, boxes: torch.Tensor, keypoints: torch.Tensor, img_id: Union[int, str] = ''
    ) -> Self:
        return cls.from_numpy(
            boxes.cpu().numpy(),
            keypoints.cpu().numpy(),
            img_id,
        )

    @classmethod
    def from_numpy(
        cls, boxes: np.ndarray, keypoints: np.ndarray, img_id: Union[int, str] = ''
    ) -> Self:
        return cls(boxes, keypoints, img_id)

    @classmethod
    def from_list(
        cls, boxes: List[List[float]], keypoints: List[List[float]], img_id: Union[int, str] = ''
    ) -> Self:
        return cls.from_numpy(
            np.array(boxes, dtype=np.float32),
            np.array(keypoints, dtype=np.float32),
            img_id,
        )

    @property
    def data(self) -> Union[Any, Dict[str, Any]]:
        return {
            'boxes': self.boxes,
            'keypoints': self.keypoints,
            'img_id': self.img_id,
        }


@dataclasses.dataclass
class KptDetEvalSample(types.BaseEvalSample):
    """
    Data element for keypoint detection evaluations.

    Attributes:
        boxes (np.ndarray): The bounding boxes in xyxy format.
        keypoints (np.ndarray): keypoints for each instance.
        scores (np.ndarray): Confidence scores for each bounding box.

    The lengths of boxes, scores, and keypoints must be the same.
    """

    boxes: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    keypoints: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))
    scores: np.ndarray = dataclasses.field(default_factory=lambda: np.array([]))

    @classmethod
    def from_numpy(cls, boxes: np.ndarray, keypoints: np.ndarray, scores: np.ndarray) -> Self:
        """
        Create an KptDetEvalSample instance from numpy arrays.

        Args:
            boxes (np.ndarray): The bounding boxes in numpy array format.
            keypoints (np.ndarray): Keypoints in numpy array format.
            scores (np.ndarray): Confidence scores in numpy array format.

        Returns:
            KptDetEvalSample: An instance with the data as numpy arrays.
        """
        assert boxes.shape[1] == 4, f"boxes should have 4 columns, got {boxes.shape[1]}"
        assert (
            boxes.shape[0] == scores.shape[0] == keypoints.shape[0]
        ), f"boxes, scores, and keypoints should have the same shape at dimension 0"
        return cls(boxes, keypoints, scores)

    @classmethod
    def from_list(
        cls, boxes: List[List[float]], keypoints: List[List[float]], scores: List[float]
    ) -> Self:
        """
        Create an KptDetEvalSample instance from lists.
        """
        assert (
            len(boxes) == len(scores) == len(keypoints)
        ), "boxes, scores, and keypoints should have the same length"
        return cls.from_numpy(
            np.array(boxes, dtype=np.float32),
            np.array(keypoints, dtype=np.float32),
            np.array(scores, dtype=np.float32),
        )

    @classmethod
    def from_torch(
        cls, boxes: torch.Tensor, keypoints: torch.Tensor, scores: torch.Tensor
    ) -> Self:
        """
        Create an KptDetEvalSample instance from torch tensors.
        """
        return cls.from_numpy(
            boxes.cpu().numpy(),
            keypoints.cpu().numpy(),
            scores.cpu().numpy(),
        )

    @classmethod
    def empty(cls) -> Self:
        return cls(
            boxes=np.empty((0, 4), dtype=np.float32),
            keypoints=np.empty((0, 0), dtype=np.float32),
            scores=np.empty(0, dtype=np.float32),
        )

    @property
    def data(self) -> Union[Any, Dict[str, Any]]:
        return {
            'boxes': self.boxes,
            'scores': self.scores,
            'keypoints': self.keypoints,
        }


@dataclasses.dataclass
class InstSegGroundTruthSample(types.BaseEvalSample):
    raw_image_size: Tuple[int, int]
    boxes: np.ndarray
    labels: np.ndarray
    polygons: list
    img_id: Union[int, str] = dataclasses.field(default='')

    def __post_init__(self):
        assert (
            self.raw_image_size[0] > 0 and self.raw_image_size[1] > 0
        ), "image_size should be greater than 0"
        assert isinstance(self.boxes, np.ndarray), "boxes must be a numpy array"
        assert isinstance(self.labels, np.ndarray), "labels must be a numpy array"
        assert isinstance(self.polygons, list), "polygons must be a list"
        self._mask_helper = None

    def set_mask_parameters(
        self,
        mask_size: Tuple[int, int] = [160, 160],
        is_mask_overlap: bool = True,
        eval_with_letterbox: bool = True,
    ):
        from .model_utils import segment

        self._mask_helper = segment.MaskHelper(
            self.raw_image_size,
            mask_size,
            is_mask_overlap,
            eval_with_letterbox,
        )

    @classmethod
    def from_torch(
        cls,
        raw_image_size: Tuple[int, int],
        boxes: torch.Tensor,
        labels: torch.Tensor,
        polygons: list,
        img_id: Union[int, str] = '',
    ) -> Self:
        return cls.from_numpy(
            raw_image_size,
            boxes.cpu().numpy(),
            labels.cpu().numpy(),
            polygons,
            img_id,
        )

    @classmethod
    def from_numpy(
        cls,
        raw_image_size: Tuple[int, int],
        boxes: np.ndarray,
        labels: np.ndarray,
        polygons: list,
        img_id: Union[int, str] = '',
    ) -> Self:
        return cls(raw_image_size, boxes, labels, polygons, img_id)

    @classmethod
    def from_list(
        cls,
        raw_image_size: Tuple[int, int],
        boxes: List[List[float]],
        labels: List[int],
        polygons: List[List[float]],
        img_id: Union[int, str] = '',
    ) -> Self:
        return cls.from_numpy(
            raw_image_size,
            np.array(boxes, dtype=np.float32),
            np.array(labels, dtype=np.float32),
            polygons,
            img_id,
        )

    @property
    def data(self) -> Union[Any, Dict[str, Any]]:
        if self._mask_helper is None:
            raise ValueError("MaskHelper is not set, please set mask parameters first.")
        if len(self.polygons) > 0:
            mask, idx = self._mask_helper.polygons2masks(self.polygons)
        else:
            mask, idx = [], np.zeros(0)
        return {
            'boxes': self.boxes,
            'labels': self.labels,
            'masks': (mask, idx),
            'img_id': self.img_id,
        }


@dataclasses.dataclass
class InstSegEvalSample(types.BaseEvalSample):
    """
    Data element for instance segmentation evaluations.

    Attributes:
        boxes (np.ndarray): The bounding boxes in xyxy format.
        labels (np.ndarray): class labels of boxes
        masks (np.ndarray): masks for each instance.
        scores (np.ndarray): Confidence scores for each bounding box.
        masks_coords (np.ndarray): coordinates of masks

    The lengths of boxes, scores, and masks must be the same.
    """

    boxes: np.ndarray
    labels: np.ndarray
    scores: np.ndarray
    masks: np.ndarray

    @classmethod
    def from_numpy(
        cls,
        boxes: np.ndarray,
        labels: np.ndarray,
        scores: np.ndarray,
        masks: np.ndarray,
    ) -> Self:
        """
        Create an InstSegEvalSample instance from numpy arrays.

        Args:
            boxes (np.ndarray): The bounding boxes in numpy array format.
            masks (np.ndarray): masks in numpy array format.
            labels (np.ndarray): class labels in numpy array format.
            scores (np.ndarray): Confidence scores in numpy array format.
        Returns:
            InstSegEvalSample: An instance with the data as numpy arrays.
        """
        if boxes.size != 0:
            assert boxes.shape[1] == 4, f"boxes should have 4 columns, got {boxes.shape[1]}"
        assert (
            boxes.shape[0] == scores.shape[0] == scores.shape[0]
        ), f"predicted boxes, scores should have the same shape at dimension 0"
        return cls(boxes, labels, scores, masks)

    @classmethod
    def from_list(
        cls,
        boxes: List[List[float]],
        labels: List[int],
        scores: List[float],
        masks: List[List[float]],
    ) -> Self:
        """
        Create an InstSegEvalSample instance from lists.
        """
        assert len(boxes) == len(scores), "boxes and scores should have the same length"
        return cls.from_numpy(
            np.array(boxes, dtype=np.float32),
            np.array(labels, dtype=np.float32),
            np.array(scores, dtype=np.float32),
            np.array(masks, dtype=np.float32),
        )

    @classmethod
    def from_torch(
        cls,
        boxes: torch.Tensor,
        labels: torch.Tensor,
        scores: torch.Tensor,
        masks: torch.Tensor,
    ) -> Self:
        """
        Create an InstSegEvalSample instance from torch tensors.
        """
        return cls.from_numpy(
            boxes.cpu().numpy(),
            labels.cpu().numpy(),
            scores.cpu().numpy(),
            masks.cpu().numpy(),
        )

    @classmethod
    def empty(cls) -> Self:
        """
        Create an empty InstSegEvalSample instance with empty arrays.
        """
        return cls(
            boxes=np.empty((0, 4), dtype=np.float32),
            labels=np.empty(0, dtype=np.int32),
            scores=np.empty(0, dtype=np.float32),
            masks=np.empty((0), dtype=np.uint8),
        )

    @property
    def data(self) -> Union[Any, Dict[str, Any]]:
        return {
            'boxes': self.boxes,
            'labels': self.labels,
            'scores': self.scores,
            'masks': self.masks,
        }


@dataclasses.dataclass
class PairValidationEvalSample(types.BaseEvalSample):
    embedding_1: np.ndarray
    embedding_2: np.ndarray

    @classmethod
    def from_torch(
        cls, embedding_1: torch.Tensor, embedding_2: torch.Tensor
    ) -> PairValidationEvalSample:
        return cls.from_numpy(embedding_1.cpu().numpy(), embedding_2.cpu().numpy())

    @classmethod
    def from_numpy(
        cls, embedding_1: np.ndarray, embedding_2: np.ndarray
    ) -> PairValidationEvalSample:
        return cls(embedding_1, embedding_2)

    @property
    def data(self) -> Union[Any, Dict[str, Any]]:
        return {
            'embedding_1': self.embedding_1,
            'embedding_2': self.embedding_2,
        }


@dataclasses.dataclass
class PairValidationGroundTruthSample(types.BaseEvalSample):
    # Labels associated to each pair of images. The two label
    # values being different targets or the same targets.
    the_same: bool

    @property
    def data(self) -> Union[Any, Dict[str, Any]]:
        return self.the_same

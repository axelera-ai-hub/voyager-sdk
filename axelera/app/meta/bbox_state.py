# Copyright Axelera AI, 2024
# Utils for building task meta

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from axelera import types

from .. import logging_utils, utils
from ..model_utils import segment as segment_utils
from ..model_utils.box import convert
from ..torch_utils import torch

LOG = logging_utils.getLogger(__name__)


def _to_numpy(data, dtype=None):
    """Convert potential tensor or nested list/tuple structures to numpy."""
    if hasattr(data, 'cpu'):
        return data.cpu().numpy()
    elif isinstance(data, (list, tuple)):
        return np.array(data, dtype=dtype)
    return data


def crop_mask(masks, boxes):
    """
    It takes a mask and a bounding box, and returns a mask that is cropped to the bounding box.

    Args:
        masks (np.ndarray): [n, h, w] array of masks
        boxes (np.ndarray): [n, 4] array of bbox coordinates in relative point form

    Returns:
        (np.ndarray): The masks are being cropped to the bounding box.
    """
    n, h, w = masks.shape
    x1, y1, x2, y2 = np.split(boxes[:, :, None], 4, axis=1)  # x1 shape(n,1,1)
    r = np.arange(w)[None, None, :]  # rows shape(1,1,w)
    c = np.arange(h)[None, :, None]  # cols shape(1,h,1)

    return masks * ((r >= x1) & (r < x2) & (c >= y1) & (c < y2))


def process_mask(protos, mask_coef, bboxes, shape):
    """
    Apply masks to bounding boxes using the output of the mask head. This produces high quality masks.

    Args:
        protos (np.ndarray): An array of shape [mask_dim, mask_h, mask_w].
        mask_coef (np.ndarray): An array of shape [n, mask_dim]. This mask_coef is served as the weights.
        bboxes (np.ndarray): An array of shape [n, 4], where n is the number of masks after NMS.
        shape (tuple): A tuple of integers representing the size of the input image in the format (h, w).

    Returns:
        (np.ndarray): A binary mask array of shape [n, h, w], where n is the number of masks after NMS, and h and w
            are the height and width of the input image. The mask is applied to the bounding boxes.
    """
    c, mh, mw = protos.shape
    ih, iw = shape
    protos = np.array(protos)
    mask_coef = np.array(mask_coef)
    bboxes = np.array(bboxes)
    masks = (
        (mask_coef @ protos.reshape(c, -1)).astype(np.float32).reshape(-1, mh, mw)
    )  # size [c,h,w]
    w_ratio, h_ratio = mw / iw, mh / ih

    resize_bboxes = bboxes.copy()
    resize_bboxes[:, 0] *= w_ratio
    resize_bboxes[:, 2] *= w_ratio
    resize_bboxes[:, 3] *= h_ratio
    resize_bboxes[:, 1] *= h_ratio
    masks = crop_mask(masks, resize_bboxes)  # size [n,h,w]
    info = np.iinfo(np.uint8)
    masks = np.clip(masks * info.max, info.min, info.max).astype(np.uint8)

    cropped_masks = []
    for i, bbox in enumerate(resize_bboxes.astype(np.int32)):
        x0, y0, x1, y1 = bbox
        cropped_masks.append(masks[i, y0:y1, x0:x1])
    return resize_bboxes.astype(np.int32), cropped_masks


@dataclass
class BBoxState:
    """Utils for organizing detection results before assign into the task meta
    Args:
        model_width (int): The input width of the model.
        model_height (int): The input height of the model.
        src_image_width (int|float): The width of the source image.
        src_image_height (int|float): The height of the source image.
        box_format (types.BoxFormat): The format of the box.
        normalized_coord (bool): Whether the coordinates are normalized or not.
        scaled (types.ResizeMode): The scale mode of the input boxes.
        nms_max_boxes (int, optional): The maximum number of boxes allowed in NMS. Defaults to 5000.
        nms_iou_threshold (float, optional): The IOU threshold for NMS. Defaults to 0.0.
        nms_class_agnostic (bool, optional): Whether to use class-based NMS or not. Defaults to False.
        output_top_k (int, optional): The number of top K boxes to output according to scores. Defaults to 300.
    """

    # parameters
    model_width: int
    model_height: int
    src_image_width: int | float
    src_image_height: int | float
    box_format: types.BoxFormat
    normalized_coord: bool
    scaled: types.ResizeMode
    nms_max_boxes: int = field(default=5000)
    nms_iou_threshold: float = field(default=0.0)
    nms_class_agnostic: bool = field(default=False)
    output_top_k: int = field(default=300)
    labels: Optional[List[str]] = field(default=None)
    label_filter: Optional[List[str]] = field(default=None)

    # outputs
    _boxes: np.ndarray = field(default_factory=lambda: np.empty((0, 4), float), init=False)
    _scores: np.ndarray = field(default_factory=lambda: np.empty([0], float), init=False)
    _class_ids: np.ndarray = field(default_factory=lambda: np.empty([0], int), init=False)
    # keypoints 51 = 17*3 for COCO body with visibility
    _kpts: np.ndarray = field(default_factory=lambda: np.empty((0, 51), float), init=False)
    _masks: np.ndarray = field(default_factory=lambda: np.empty((0, 160, 160), float), init=False)

    def __post_init__(self) -> None:
        if self.nms_iou_threshold == 0.0:
            LOG.debug("No NMS will be applied!")

        self.label_filter_ids = None
        if self.labels and self.label_filter:  # get id from label
            if isinstance(self.labels, utils.FrozenIntEnumMeta):
                self.label_filter_ids = [getattr(self.labels, l) for l in self.label_filter]
            else:
                self.label_filter_ids = [self.labels.index(l) for l in self.label_filter]

        if isinstance(self.box_format, str):
            self.box_format = types.BoxFormat.parse(self.box_format)

        if self.scaled not in [
            types.ResizeMode.ORIGINAL,
            types.ResizeMode.LETTERBOX_FIT,
            types.ResizeMode.STRETCH,
            types.ResizeMode.SQUISH,
        ]:
            raise ValueError(f"Resize mode: {self.scaled} is not supported yet")

        if type(self.src_image_width) != int:
            self.src_image_width = int(self.src_image_width + 0.5)
            self.src_image_height = int(self.src_image_height + 0.5)

    def organize_bboxes(
        self, boxes: List[List[float]], scores: List[float], class_ids: List[int]
    ) -> Tuple[List[List[float]], List[float], List[int]]:
        """
        Organize the detected boxes, scores, and class IDs.
        Args:
        boxes: A list of 4-dimensional representative coordinates of the detected boxes.
        scores: A list of scores associated with each detected box.
        class_ids: A list of class IDs corresponding to each detected box.

        Returns:
        A tuple of 3 numpy arrays representing the filtered boxes, scores and class IDs respectively.
        """
        assert (
            len(boxes) == len(scores) == len(class_ids)
        ), f"Shapes do not match: boxes {len(boxes)}, scores {len(scores)}, class_ids {len(class_ids)}"

        boxes = _to_numpy(boxes, dtype=np.float64)
        scores = _to_numpy(scores)
        class_ids = _to_numpy(class_ids)

        # filter out invalid labels
        if self.label_filter_ids:
            valid = np.isin(class_ids, self.label_filter_ids)
            boxes = boxes[valid]
            scores = scores[valid]
            class_ids = class_ids[valid]

        if len(boxes) > 0:
            if boxes.shape[1] != 4:
                raise ValueError(
                    "The input box must be a 4-dimensional representative of coordinates"
                )

            # descending sort by score and remove excess boxes
            max_boxes = self.nms_max_boxes if self.nms_iou_threshold > 0 else self.output_top_k
            descending_idx = (-scores).argsort()[:max_boxes]
            self._boxes = boxes[descending_idx]
            self._scores = scores[descending_idx]
            self._class_ids = class_ids[descending_idx]
            self.formatting()
            if self.nms_iou_threshold > 0:
                self.nms()
        return self._boxes, self._scores, self._class_ids

    def organize_bboxes_and_kpts(
        self, boxes: List[List[float]], scores: List[float], kpts: List[List[float]]
    ) -> Tuple[List[List[float]], List[float], List[List[float]]]:
        """
        This method organizes detected keypoints, which are expected to be in a list of
        N-dimensional keypoints. Each keypoint is represented as (x, y) or (x, y, v),
        making N a multiple of 2 or 3. This implementation specifically supports the COCO
        format for keypoints. If keypoints are in a different format, please convert them
        to COCO format or implement a custom organizer function tailored to that format.

        Args:
        boxes: A list of 4-dimensional representative coordinates of the detected boxes.
        scores: A list of scores associated with each detected box.
        kpts: A list of N-dimensional representative coordinates of the detected keypoints.
        """
        num_samples = len(boxes)
        if num_samples == 0:
            return boxes, scores, kpts
        assert (
            num_samples == len(scores) == len(kpts)
        ), f"Shapes do not match: boxes {num_samples}, scores {len(scores)}, kpts {len(kpts)}"

        boxes = _to_numpy(boxes, dtype=np.float64)
        scores = _to_numpy(scores)
        kpts = _to_numpy(kpts, dtype=np.float64)
        assert kpts.ndim == 2, "The input keypoints must be a 2-dimensional tensor"
        # should check %3 or %2 before reshape
        if kpts.shape[1] % 3 == 0:
            kpts = kpts.reshape(num_samples, -1, 3)
        elif kpts.shape[1] % 2 == 0:
            kpts = kpts.reshape(num_samples, -1, 2)
        else:
            raise ValueError("Please confirm the number of keypoints")

        # filter out invalid labels
        if self.label_filter_ids:
            raise NotImplementedError("Label filtering is not supported for keypoints yet.")

        if len(boxes) > 0:
            if boxes.shape[1] != 4:
                raise ValueError(
                    "The input box must be a 4-dimensional representative of coordinates"
                )

            # descending sort by score and remove excess boxes
            max_boxes = self.nms_max_boxes if self.nms_iou_threshold > 0 else self.output_top_k
            descending_idx = (-scores).argsort()[:max_boxes]
            self._boxes = boxes[descending_idx]
            self._scores = scores[descending_idx]
            self._kpts = kpts[descending_idx]
            self.formatting_kpts()
            if self.nms_iou_threshold > 0:
                self.nms(is_kpts=True)
        return self._boxes, self._scores, self._kpts

    def organize_bboxes_and_instance_seg(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        class_ids: np.ndarray,
        mask_coef: np.ndarray,  # The coefficients or embeddings used to generate the final masks from the prototype masks.
        protos: np.ndarray,  # The prototype masks.
        unpad: bool,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        assert (
            len(boxes) == len(scores) == len(class_ids)
        ), f"Shapes do not match: boxes {len(boxes)}, scores {len(scores)}, class_ids {len(class_ids)}"

        boxes = _to_numpy(boxes, dtype=np.float32)
        scores = _to_numpy(scores)
        class_ids = _to_numpy(class_ids)
        mask_coef = _to_numpy(mask_coef, dtype=np.float32)

        # filter out invalid labels
        if self.label_filter_ids:
            valid = np.isin(class_ids, self.label_filter_ids)
            boxes = boxes[valid]
            scores = scores[valid]
            class_ids = class_ids[valid]
            mask_coef = mask_coef[valid]

        if len(boxes) > 0:
            if boxes.shape[1] != 4:
                raise ValueError(
                    "The input box must be a 4-dimensional representative of coordinates"
                )

            # descending sort by score and remove excess boxes
            max_boxes = self.nms_max_boxes if self.nms_iou_threshold > 0 else self.output_top_k
            descending_idx = (-scores).argsort()[:max_boxes]
            self._boxes = boxes[descending_idx]
            self._scores = scores[descending_idx]
            self._class_ids = class_ids[descending_idx]
            mask_coef = mask_coef[descending_idx]

            if self.box_format != types.BoxFormat.XYXY:
                self._boxes = self.xyxy()
                self.box_format = types.BoxFormat.XYXY
            if self.normalized_coord:
                self._boxes[:, [0, 2]] *= self.src_image_width
                self._boxes[:, [1, 3]] *= self.src_image_height
                self.normalized_coord = False
            if self.nms_iou_threshold > 0:
                self.nms(mask_coef=mask_coef)

            self._masks = process_mask(
                protos, mask_coef, self._boxes, (self.model_height, self.model_width)
            )
            self.rescale_boxes(unpad)

        return self._boxes, self._scores, self._class_ids, self._masks

    def empty(self):
        return len(self._class_ids) == 0

    def xyxy(self):
        return convert(self._boxes, self.box_format, types.BoxFormat.XYXY)

    def xywh(self):
        return convert(self._boxes, self.box_format, types.BoxFormat.XYWH)

    def ltwh(self):
        return convert(self._boxes, self.box_format, types.BoxFormat.LTWH)

    def formatting(self) -> None:
        """Rescale + denormalized + boxes as xyxy format"""
        if self.box_format != types.BoxFormat.XYXY:
            self._boxes = self.xyxy()
            self.box_format = types.BoxFormat.XYXY

        if self.normalized_coord:
            if self.scaled == types.ResizeMode.STRETCH:
                width, height = self.src_image_width, self.src_image_height
                self.scaled = types.ResizeMode.ORIGINAL
            elif self.scaled == types.ResizeMode.LETTERBOX_FIT:
                width, height = self.model_width, self.model_height
            else:
                raise ValueError(f"Unsupported resize mode: {self.scaled}")
            self._boxes[:, [0, 2]] *= width
            self._boxes[:, [1, 3]] *= height
            self.normalized_coord = False
        elif self.scaled != types.ResizeMode.ORIGINAL:
            self._boxes, _, _ = self.rescale(
                self._boxes,
                types.BoxFormat.XYXY,
                (self.model_height, self.model_width),
                (self.src_image_height, self.src_image_width),
                resize_mode=self.scaled,
            )
            self.scaled = types.ResizeMode.ORIGINAL
        self._boxes[:, [0, 2]] = np.clip(self._boxes[:, [0, 2]], 0, self.src_image_width)
        self._boxes[:, [1, 3]] = np.clip(self._boxes[:, [1, 3]], 0, self.src_image_height)

    def formatting_kpts(self) -> None:
        if self.box_format != types.BoxFormat.XYXY:
            self._boxes = self.xyxy()
            self.box_format = types.BoxFormat.XYXY

        if self.normalized_coord:
            if self.scaled == types.ResizeMode.STRETCH:
                width, height = self.src_image_width, self.src_image_height
                self.scaled = types.ResizeMode.ORIGINAL
            elif self.scaled == types.ResizeMode.LETTERBOX_FIT:
                width, height = self.model_width, self.model_height
            else:
                raise ValueError(f"Unsupported resize mode: {self.scaled}")
            self._boxes[:, [0, 2]] *= width
            self._boxes[:, [1, 3]] *= height
            self._kpts[:, :, 0] *= width
            self._kpts[:, :, 1] *= height
            self.normalized_coord = False

        if self.scaled != types.ResizeMode.ORIGINAL:
            self._boxes, self._kpts, _ = self.rescale(
                self._boxes,
                types.BoxFormat.XYXY,
                (self.model_height, self.model_width),
                (self.src_image_height, self.src_image_width),
                resize_mode=self.scaled,
                kpts=self._kpts,
            )
            self.scaled = types.ResizeMode.ORIGINAL
        self._boxes[:, [0, 2]] = np.clip(self._boxes[:, [0, 2]], 0, self.src_image_width)
        self._boxes[:, [1, 3]] = np.clip(self._boxes[:, [1, 3]], 0, self.src_image_height)

    def rescale_boxes(self, unpad: bool = True) -> None:
        if self.scaled != types.ResizeMode.ORIGINAL:
            self._boxes, _, self._masks = self.rescale(
                self._boxes,
                types.BoxFormat.XYXY,
                (self.model_height, self.model_width),
                (self.src_image_height, self.src_image_width),
                resize_mode=self.scaled,
                masks=self._masks,
                unpad=unpad,
            )
            self.scaled = types.ResizeMode.ORIGINAL
        self._boxes[:, [0, 2]] = np.clip(self._boxes[:, [0, 2]], 0, self.src_image_width)
        self._boxes[:, [1, 3]] = np.clip(self._boxes[:, [1, 3]], 0, self.src_image_height)

    @staticmethod
    def rescale(
        boxes: np.ndarray,
        box_format: types.BoxFormat,
        ori_shape: Tuple[int, int],
        target_shape: Tuple[int, int],
        ratio_pad: Optional[Tuple[float, Tuple[float, float]]] = None,
        resize_mode: types.ResizeMode = types.ResizeMode.LETTERBOX_FIT,
        kpts: Optional[np.ndarray] = None,
        masks: Optional[np.ndarray] = None,
        unpad: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        '''Rescale the bounding boxes and keypoints to the original image size.

        Args:
            boxes (np.ndarray): The input bounding boxes. Shape (n, 4) where n is the number of boxes.
            box_format (types.BoxFormat): The format of the bounding boxes. Either 'xyxy', 'xywh', or 'ltwh'.
            ori_shape (Tuple[int, int]): The original image size in the format (height, width).
            target_shape (Tuple[int, int]): The target image size in the format (height, width).
            ratio_pad (Optional[Tuple[float, Tuple[float, float]]]): The padding ratio and padding amounts when letterbox scaleup is true.
            resize_mode (types.ResizeMode): The resize mode.
            kpts (Optional[np.ndarray]): The input keypoints. Shape (n, k, 2 or 3) where n is the number of boxes and k is the number of keypoints.
            masks (Optional[np.ndarray]): The input masks. Shape (n, h, w) where n is the number of masks.
            unpad (bool): Whether to unpad the mask. Default is True.

        Returns:
            Tuple[np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]: The rescaled bounding boxes and keypointscaled masks (if any).
        '''
        rescaled_kpts = None
        rescaled_masks = None

        if resize_mode == types.ResizeMode.STRETCH:
            rescaled_boxes = np.copy(boxes)
            ratio_x = ori_shape[1] / target_shape[1]
            rescaled_boxes[:, [0, 2]] /= ratio_x
            ratio_y = ori_shape[0] / target_shape[0]
            rescaled_boxes[:, [1, 3]] /= ratio_y
            if kpts is not None:
                rescaled_kpts = np.copy(kpts)
                rescaled_kpts[:, :, 1] /= ratio_y
                rescaled_kpts[:, :, 0] /= ratio_x
            return rescaled_boxes, rescaled_kpts, rescaled_masks

        if ratio_pad is None:
            # same logic of letterbox
            ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
            new_unpad = (int(round(target_shape[0] * ratio)), int(round(target_shape[1] * ratio)))
            dh, dw = ori_shape[0] - new_unpad[0], ori_shape[1] - new_unpad[1]
            dh /= 2
            dw /= 2
            top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
            left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
            padding = (left, top, right, bottom)
        else:  # if letterbox scaleup is false, we will need ratio_pad for each image
            ratio = ratio_pad[0]
            padding = ratio_pad[1]

        rescaled_boxes = np.copy(boxes)
        rescaled_boxes[:, 0] -= padding[0]  # x1
        rescaled_boxes[:, 1] -= padding[1]  # y1
        if box_format == types.BoxFormat.XYXY:
            rescaled_boxes[:, 2] -= padding[0]  # x2
            rescaled_boxes[:, 3] -= padding[1]  # y2
        rescaled_boxes[:, :4] /= ratio
        if kpts is not None:
            rescaled_kpts = np.copy(kpts)
            rescaled_kpts[:, :, 0] -= padding[0]
            rescaled_kpts[:, :, 1] -= padding[1]
            rescaled_kpts[:, :, :2] /= ratio

        rescaled_boxes[:, [0, 2]] = rescaled_boxes[:, [0, 2]].clip(
            0, target_shape[1]
        )  # x1,w  or x1x2
        rescaled_boxes[:, [1, 3]] = rescaled_boxes[:, [1, 3]].clip(
            0, target_shape[0]
        )  # y1,h  or y1y2
        if kpts is not None:
            rescaled_kpts[:, :, 0] = rescaled_kpts[:, :, 0].clip(0, target_shape[1])
            rescaled_kpts[:, :, 1] = rescaled_kpts[:, :, 1].clip(0, target_shape[0])
        if masks is not None:
            rescaled_masks = masks
            if False:  # For debugging
                import cv2

                combined_mask = segment_utils.combine_masks(rescaled_masks)
                normalized = cv2.normalize(combined_mask, None, 0, 255, cv2.NORM_MINMAX)
                cv2.imwrite('rescaled_masks_combined.png', normalized)

        return rescaled_boxes, rescaled_kpts, rescaled_masks

    def nms(self, is_kpts: bool = False, mask_coef: Optional[np.ndarray] = None):
        """
        Perform Non-Maximum Suppression (NMS) to filter overlapping bounding boxes.
        If `mask_coef` is provided, it indicates a segmentation task, and the mask coefficients
        will be updated in-place.

        Args:
            is_kpts (bool): If True, keypoints are considered in the NMS process.
            mask_coef (Optional[np.ndarray]): Mask coefficients for segmentation, updated in-place if provided.
        """
        if not self._boxes.size:
            return

        from torchvision import ops

        boxes = torch.tensor(self._boxes).float()
        scores = torch.tensor(self._scores).float()

        final_indices = []

        if self.nms_class_agnostic:
            # Apply NMS directly without considering classes
            keep_indices = ops.nms(boxes, scores, self.nms_iou_threshold)
            final_indices.extend(keep_indices.tolist())
        else:
            # Perform NMS per class
            classes = torch.tensor(self._class_ids)
            for cls_id in torch.unique(classes):
                cls_mask = classes == cls_id
                cls_boxes = boxes[cls_mask]
                cls_scores = scores[cls_mask]
                if cls_boxes.shape[0] > 0:
                    keep_indices = ops.nms(cls_boxes, cls_scores, self.nms_iou_threshold)
                    # Map indices back to the original set of boxes
                    original_indices = cls_mask.nonzero().squeeze(-1)
                    mapped_indices = original_indices[keep_indices]
                    final_indices.extend(mapped_indices.tolist())

        final_indices = torch.unique(torch.tensor(final_indices))  # Remove duplicates

        if final_indices.numel() > 0:
            final_indices = final_indices[: self.output_top_k]  # Limit to top K results
            final_indices = final_indices.numpy()  # Convert back to NumPy array
            self._boxes = self._boxes[final_indices]
            self._scores = self._scores[final_indices]
            if is_kpts:
                self._kpts = self._kpts[final_indices]
            elif mask_coef is not None:
                new_mask_coef = mask_coef[final_indices]
                new_size = len(final_indices)
                mask_coef.resize((new_size, mask_coef.shape[1]), refcheck=False)
                mask_coef[:] = new_mask_coef
                self._class_ids = self._class_ids[final_indices]
            else:
                self._class_ids = self._class_ids[final_indices]

# Copyright Axelera AI, 2024
# General functions for handling segment related tasks
from __future__ import annotations

from copy import deepcopy
import math
from typing import TYPE_CHECKING, List, Optional, Tuple
import warnings

import cv2
import numpy as np

from axelera import types

if TYPE_CHECKING:
    from axelera.app import operators


def combine_masks(masks: List[np.ndarray]) -> np.ndarray:
    """
    Combine a list of masks into a single mask.
    """
    combined_mask = np.zeros(masks[0].shape, dtype=np.uint32)
    for i, mask in enumerate(masks):
        # Apply the mask with its label and clip in one step to optimize processing
        combined_mask = np.clip(combined_mask + mask * (i + 1), a_min=0, a_max=i + 1)
    return combined_mask.astype(np.uint8)


def simple_resize_masks(masks, target_shape, align_corners=False):
    """
    NumPy implementation to resize masks to the target shape using bilinear interpolation,
    aiming to align with PyTorch's F.interpolate(mode='bilinear', align_corners=False).

    Args:
        masks (np.ndarray): [n, h, w] array of masks
        target_shape (tuple): (height, width) of the target shape
        align_corners (bool): If True, use center-aligned interpolation. Defaults to False.

    Returns:
        (np.ndarray): Resized masks
    """
    n, h, w = masks.shape
    target_h, target_w = target_shape

    # Calculate scaling factors
    if align_corners:
        scale_h, scale_w = (target_h - 1) / (h - 1), (target_w - 1) / (w - 1)
    else:
        scale_h, scale_w = target_h / h, target_w / w

    # Create coordinate matrices for the target shape
    y_coords, x_coords = np.meshgrid(np.arange(target_h), np.arange(target_w), indexing='ij')

    # Calculate the corresponding coordinates in the original masks
    if align_corners:
        orig_y = y_coords / scale_h
        orig_x = x_coords / scale_w
    else:
        orig_y = (y_coords + 0.5) / scale_h - 0.5
        orig_x = (x_coords + 0.5) / scale_w - 0.5

    # Calculate the four nearest neighbor points
    y0 = np.floor(orig_y).astype(int)
    x0 = np.floor(orig_x).astype(int)
    y1 = np.minimum(y0 + 1, h - 1)
    x1 = np.minimum(x0 + 1, w - 1)
    # Clip coordinates to valid range
    y0 = np.clip(y0, 0, h - 1)
    x0 = np.clip(x0, 0, w - 1)

    # Calculate interpolation weights
    wy = orig_y - y0
    wx = orig_x - x0

    # Reshape weights for broadcasting
    wy = wy.reshape(1, target_h, target_w)
    wx = wx.reshape(1, target_h, target_w)

    # Perform bilinear interpolation for all masks at once
    top_left = masks[:, y0, x0]
    top_right = masks[:, y0, x1]
    bottom_left = masks[:, y1, x0]
    bottom_right = masks[:, y1, x1]

    resized_masks = (
        top_left * (1 - wy) * (1 - wx)
        + bottom_left * wy * (1 - wx)
        + top_right * (1 - wy) * wx
        + bottom_right * wy * wx
    )

    return resized_masks


class MaskHelper:
    def __init__(
        self,
        raw_img_size: Tuple[int, int],
        mask_size: Tuple[int, int],
        is_mask_overlap: bool = True,
        eval_with_letterbox: bool = True,
        color: int = 1,
    ):
        """
        raw_img_size: (height, width), this is the size of the raw image
        mask_size: (height, width), this is the size of the mask
        is_mask_overlap: bool, this is whether the mask is overlapped
        eval_with_letterbox: bool, this is whether to use letterbox to resize the mask
        color: int, this is the color of the mask, 1 as white
        """
        self.raw_img_size = raw_img_size
        self.is_mask_overlap = is_mask_overlap
        self.mask_size = mask_size
        self.dtype = np.uint8
        self.color = color

        # This approach assumes the use of a letterbox for mask resizing with specific parameters to avoid
        # dependencies that could hinder ML dev flexibility.
        self.eval_with_letterbox = eval_with_letterbox
        if self.eval_with_letterbox:
            from axelera.app import operators as op

            self.letterbox_or_resize_op = op.Letterbox(
                *self.mask_size, scaleup=True, half_pixel_centers=True, pad_val=0
            )

    def initialize_mask(self, image_size: Tuple[int, int]) -> np.ndarray:
        """Initialize a mask with zeros based on the image size and data type."""
        if self.mask_size is None:
            calculated_mask_size = (
                image_size[0] // self.downsample_ratio,
                image_size[1] // self.downsample_ratio,
            )
        else:
            calculated_mask_size = self.mask_size
        return np.zeros(calculated_mask_size, dtype=self.dtype)

    def convert_polygons_to_cv_format(self, polygons: List[np.ndarray]) -> List[np.ndarray]:
        """Convert polygons to a format suitable for OpenCV processing."""
        if not all(isinstance(polygon, np.ndarray) for polygon in polygons):
            raise ValueError("All polygons must be numpy arrays.")
        return [np.asarray(polygon, dtype=np.int32).reshape((-1, 2)) for polygon in polygons]

    def interpolate_segments(self, segments, n: int = 1000):
        """
        Interpolates each segment in the list to have n points.

        Args:
            segments (list of np.ndarray): List of segments where each segment is an array of shape (m, 2).
            n (int): The number of points to interpolate to.

        Returns:
            list of np.ndarray: List of interpolated segments with each segment having n points.
        """
        interpolated_segments = []

        for i, segment in enumerate(segments):
            # Ensure the segment is closed by appending the first point to the end
            closed_segment = np.concatenate((segment, segment[0:1, :]), axis=0)

            interp_points = np.linspace(0, len(closed_segment) - 1, n)
            original_points = np.arange(len(closed_segment))

            # Interpolate x and y coordinates
            interpolated_x = np.interp(interp_points, original_points, closed_segment[:, 0])
            interpolated_y = np.interp(interp_points, original_points, closed_segment[:, 1])
            interpolated_segment = np.vstack((interpolated_x, interpolated_y)).T

            # Convert to float32 and store the result
            interpolated_segments.append(interpolated_segment.astype(np.float32))

        return interpolated_segments

    def polygon2mask(self, polygons: List[np.ndarray]) -> np.ndarray:
        """
        Convert a list of polygons to a binary mask of the specified image size.
        """
        mask = np.zeros(self.raw_img_size, dtype=self.dtype)
        scaled_polygons = np.round(polygons).astype(np.int32)
        cv2.fillPoly(mask, [scaled_polygons], color=self.color)
        if self.eval_with_letterbox:
            mask = self.letterbox_or_resize_op.exec_torch(
                types.Image.fromarray(mask, types.ColorFormat.GRAY)
            )
            return mask.asarray()
        else:
            return mask

    def polygons2masks(self, polygons_list: List[List[np.ndarray]]) -> np.ndarray:
        """
        Convert a list of lists of polygons to a set of binary masks of the specified image size.
        Optionally, handle overlapping by sorting masks by area and layering them if is_mask_overlap is True.

        image_size: (height, width)
        """

        polygons_list = self.interpolate_segments(polygons_list)
        masks = np.array([self.polygon2mask(p) for p in polygons_list])

        if self.is_mask_overlap:
            areas = [-1 * (mask.sum().astype(np.float32)) for mask in masks]
            sorted_indices = np.argsort(np.array(areas))
            masks = [masks[i] for i in sorted_indices]
            masks = np.array(masks)
            combined_mask = combine_masks(masks)
        if False:  # visualize the combined mask for debug
            normalized = cv2.normalize(combined_mask, None, 0, 255, cv2.NORM_MINMAX)
            cv2.imwrite("combined_mask2.png", normalized)
        return combined_mask, sorted_indices

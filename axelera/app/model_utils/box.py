# Copyright Axelera AI, 2023
# General functions for handling boxes with different coordinate representations

import sys
from typing import List

import cv2
import numpy as np

from axelera import types


def convert(
    boxes: List[float], input_format: types.BoxFormat, output_format: types.BoxFormat
) -> List[float]:
    '''Converts the represented format of box

    Args:
    boxes(List[float]): The list of boxes to convert
    input_format(types.BoxFormat): The format of the input boxes
    output_format(types.BoxFormat): The format to convert the boxes to

    Returns:
    List[float]: The list of boxes in the output format
    '''
    if input_format != output_format:
        input_name = input_format.name.lower()
        output_name = output_format.name.lower()
        return getattr(sys.modules[__name__], f'{input_name}2{output_name}')(boxes)
    return boxes


def xyxy2xywh(xyxy):
    """Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h]

    Args:
        xyxy (np.narray or torch.Tensor): (x1,y1)=top-left, (x2,y2)=bottom-right

    Returns:
        np.narray or torch.Tensor: xywh, (x,y)=box center, (w,h)=box width & height
    """
    xywh = xyxy.clone() if hasattr(xyxy, 'clone') else np.copy(xyxy)

    xywh[..., 0] = (xyxy[..., 0] + xyxy[..., 2]) / 2  # x center
    xywh[..., 1] = (xyxy[..., 1] + xyxy[..., 3]) / 2  # y center
    xywh[..., 2] = xyxy[..., 2] - xyxy[..., 0]  # width
    xywh[..., 3] = xyxy[..., 3] - xyxy[..., 1]  # height
    return xywh


def xywh2xyxy(xywh):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]

    Args:
        xywh (np.narray or torch.Tensor): (x,y)=box center, (w,h)=box width & height

    Returns:
        np.narray or torch.Tensor: xyxy, (x1,y1)=top-left, (x2,y2)=bottom-right
    """

    xyxy = xywh.clone() if hasattr(xywh, 'clone') else np.copy(xywh)
    xyxy[..., 0] = xywh[..., 0] - xywh[..., 2] / 2
    xyxy[..., 1] = xywh[..., 1] - xywh[..., 3] / 2
    xyxy[..., 2] = xywh[..., 0] + xywh[..., 2] / 2
    xyxy[..., 3] = xywh[..., 1] + xywh[..., 3] / 2
    return xyxy


def xyxy2ltwh(xyxy):
    """Convert nx4 boxes from [x, y, x, y] to [x1, y1, w, h]

    Args:
        xyxy (np.narray or torch.Tensor): (x1,y1)=top-left, (x2,y2)=bottom-right

    Returns:
        np.narray or torch.Tensor: x1y1wh, (x1,y1)=top-left, (w,h)=box width & height
    """
    ltwh = xyxy.clone() if hasattr(xyxy, 'clone') else np.copy(xyxy)
    ltwh[..., 2] = xyxy[..., 2] - xyxy[..., 0]
    ltwh[..., 3] = xyxy[..., 3] - xyxy[..., 1]
    return ltwh


def xywh2ltwh(xywh):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, w, h]

    Args:
        xywh (np.narray or torch.Tensor): (x,y)=box center, (w,h)=box width & height

    Returns:
        np.narray or torch.Tensor: x1y1wh, (x1,y1)=top-left, (w,h)=box width & height
    """
    ltwh = xywh.clone() if hasattr(xywh, 'clone') else np.copy(xywh)
    ltwh[..., 0] = xywh[..., 0] - xywh[..., 2] / 2
    ltwh[..., 1] = xywh[..., 1] - xywh[..., 3] / 2
    return ltwh


def ltwh2xyxy(ltwh):
    """Convert nx4 boxes from [x1, y1, w, h] to [x1, y1, x2, y2]

    Args:
        ltwh (np.narray or torch.Tensor): lt=top-left, (w,h)=box width & height

    Returns:
        np.narray or torch.Tensor: xyxy, (x1,y1)=top-left, (x2,y2)=bottom-right
    """
    xyxy = ltwh.clone() if hasattr(ltwh, 'clone') else np.copy(ltwh)
    xyxy[..., 2] = ltwh[..., 0] + ltwh[..., 2]
    xyxy[..., 3] = ltwh[..., 1] + ltwh[..., 3]
    return xyxy


def ltwh2xywh(ltwh):
    """Convert nx4 boxes from [x1, y1, w, h] to [x, y, w, h]

    Args:
        ltwh (np.narray or torch.Tensor): lt=top-left, (w,h)=box width & height

    Returns:
        np.narray or torch.Tensor: xywh, (x,y)=box center, (w,h)=box width & height
    """
    xywh = ltwh.clone() if hasattr(ltwh, 'clone') else np.copy(ltwh)
    xywh[..., 0] = ltwh[..., 0] + ltwh[..., 2] / 2
    xywh[..., 1] = ltwh[..., 1] + ltwh[..., 3] / 2
    return xywh


def xywhr2xyxyxyxy(x):
    """
    Convert OBBs from [cx, cy, w, h, theta] -> [xy1, xy2, xy3, xy4] (counterclockwise order)

    Args:
        x (array-like): shape (N, 5) or (B, N, 5) or (5,)
            [cx, cy, w, h, theta], where theta is in radians (uses minAreaRect angles converted to radians, ~(-pi/2, 0]).

    Returns:
        np.ndarray: shape (N, 4, 2) or (B, N, 4, 2) (same leading dims as input).
    """
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        if arr.size != 5:
            raise ValueError(f"Flat input must have 5 numbers, got {arr.size}.")
        arr = arr.reshape(1, 5)

    if arr.shape[-1] != 5:
        raise ValueError(f"Expected last dim=5, got shape {arr.shape}.")

    ctr = arr[..., :2]  # (..., 2)
    w = arr[..., 2:3]  # (..., 1)
    h = arr[..., 3:4]  # (..., 1)
    ang = arr[..., 4:5]  # (..., 1)

    cos_a = np.cos(ang)  # (..., 1)
    sin_a = np.sin(ang)  # (..., 1)

    # Basis vectors along width (theta) and height (theta + 90°)
    vec1 = np.concatenate([(w * 0.5) * cos_a, (w * 0.5) * sin_a], axis=-1)  # (..., 2)
    vec2 = np.concatenate([-(h * 0.5) * sin_a, (h * 0.5) * cos_a], axis=-1)  # (..., 2)

    pt1 = ctr + vec1 + vec2
    pt2 = ctr + vec1 - vec2
    pt3 = ctr - vec1 - vec2
    pt4 = ctr - vec1 + vec2

    corners = np.stack([pt1, pt2, pt3, pt4], axis=-2)  # (..., 4, 2)
    return corners


def xyxyxyxy2xywhr(x):
    """
    Convert OBB corners [x1,y1,x2,y2,x3,y3,x4,y4] -> [cx, cy, w, h, theta] using cv2.minAreaRect.

    Args:
        x (array-like): shape (N, 8) or (8,), corners in any order.

    Returns:
        np.ndarray: shape (N, 5), dtype float32. Theta is radians converted from
        minAreaRect's angle (typically in (-pi/2, 0]).
    """
    arr = np.asarray(x)
    if arr.ndim == 1:
        if arr.size != 8:
            raise ValueError(f"Flat input must have 8 numbers, got {arr.size}.")
        arr = arr.reshape(1, 8)
    if arr.ndim != 2 or arr.shape[1] != 8:
        raise ValueError(f"Expected shape (N, 8), got {arr.shape}.")

    # (N, 4, 2) float32 for OpenCV
    pts = arr.reshape(-1, 4, 2).astype(np.float32, copy=False)

    out = np.empty((pts.shape[0], 5), dtype=np.float32)
    for i, p in enumerate(pts):
        (cx, cy), (w, h), angle_deg = cv2.minAreaRect(p)  # angle in degrees (-90, 0]
        out[i, 0] = cx
        out[i, 1] = cy
        out[i, 2] = w
        out[i, 3] = h
        out[i, 4] = angle_deg * np.pi / 180.0  # radians

    return out


def box_iou_1_to_many(box1: np.array, bboxes2: np.array, c: int = 1):
    """Calculate the intersected surface from bboxes over box1

    Args:
        box1 (xyxy): the base box with (x1,y1,x2,y2)
        bboxes2 (N x xyxy): targets to compare
        c: 1 if working with pixel/screen coordinates or 0 for point coordinates;
           see reason from https://github.com/AlexeyAB/darknet/issues/3995#issuecomment-535697357

    Returns:
        float: IoU in [0, 1]
    """
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = np.split(bboxes2, 4, axis=1)
    xA = np.maximum(x11, x21.T)
    yA = np.maximum(y11, y21.T)
    xB = np.minimum(x12, x22.T)
    yB = np.minimum(y12, y22.T)
    inter_area = np.maximum((xB - xA + c), 0) * np.maximum((yB - yA + c), 0)
    box1_area = (x12 - x11 + c) * (y12 - y11 + c)
    box2_area = (x22 - x21 + c) * (y22 - y21 + c)
    return inter_area / (box1_area + box2_area.T - inter_area)


def get_covariance_matrix(boxes_xywhr: np.ndarray):
    boxes = boxes_xywhr.astype(np.float64)
    w = boxes[:, 2]
    h = boxes[:, 3]
    r = boxes[:, 4]
    a = (w * w) / 12.0
    b = (h * h) / 12.0
    cos = np.cos(r)
    sin = np.sin(r)
    cos2 = cos * cos
    sin2 = sin * sin
    A = a * cos2 + b * sin2
    B = a * sin2 + b * cos2
    C = (a - b) * cos * sin
    return A, B, C


# Probabilistic IoU
def batch_probiou(obb1: np.ndarray, obb2: np.ndarray, eps: float = 1e-7) -> np.ndarray:
    obb1 = obb1.astype(np.float64)
    obb2 = obb2.astype(np.float64)
    x1 = obb1[:, 0:1]
    y1 = obb1[:, 1:2]
    x2 = obb2[None, :, 0]
    y2 = obb2[None, :, 1]
    a1, b1, c1 = get_covariance_matrix(obb1)
    a2, b2, c2 = get_covariance_matrix(obb2)
    a1 = a1[:, None]
    b1 = b1[:, None]
    c1 = c1[:, None]
    a2 = a2[None, :]
    b2 = b2[None, :]
    c2 = c2[None, :]
    denom = (a1 + a2) * (b1 + b2) - (c1 + c2) ** 2 + eps
    t1 = (((a1 + a2) * (y1 - y2) ** 2 + (b1 + b2) * (x1 - x2) ** 2) / denom) * 0.25
    t2 = (((c1 + c2) * (x2 - x1) * (y1 - y2)) / denom) * 0.5
    num3 = (a1 + a2) * (b1 + b2) - (c1 + c2) ** 2
    term1 = np.maximum(0.0, a1 * b1 - c1 * c1)
    term2 = np.maximum(0.0, a2 * b2 - c2 * c2)
    den3 = 4.0 * np.sqrt(term1 * term2) + eps
    t3 = 0.5 * np.log(num3 / den3 + eps)
    bd = np.clip(t1 + t2 + t3, eps, 100.0)
    hd = np.sqrt(1.0 - np.exp(-bd) + eps)
    return 1.0 - hd


def faster_rcnn_box_decode(locations, priors, center_variance, size_variance):
    """
    Decode predicted offsets to bounding box using Faster R-CNN box coder.

    Args:
    locations (ndarray): Predicted offsets for each prior box.
    priors (ndarray): Prior boxes in [cx, cy, w, h] format.
    center_variance, size_variance: Scaling factors.

    Returns:
    Decoded bounding boxes in [xmin, ymin, xmax, ymax] format.
    """
    boxes = np.concatenate(
        [
            locations[:, :2] * center_variance * priors[:, 2:] + priors[:, :2],
            np.exp(locations[:, 2:] * size_variance) * priors[:, 2:],
        ],
        axis=1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

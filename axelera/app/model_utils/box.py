# Copyright Axelera AI, 2024
# General functions for handling boxes with different coordinate representations

import sys
from typing import List

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

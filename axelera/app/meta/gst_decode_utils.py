# Copyright Axelera AI, 2024
import struct

import numpy as np


def decode_bbox(data):
    """
    Bbox holds a 1D array of integers which is of size num_entries * 4 (x1,y1,x2,y2)
    """
    bbox_size = 4
    bbox = data.get("bbox", b"")
    boxes1d = np.frombuffer(bbox, dtype=np.int32)
    boxes2d = np.reshape(boxes1d, (-1, bbox_size))
    return boxes2d


def decode_bbox_obb_xyxyxyxy(data):
    bbox_size = 8
    bbox = data.get("bbox_obb", b"")
    boxes1d = np.frombuffer(bbox, dtype=np.int32)
    boxes2d = np.reshape(boxes1d, (-1, bbox_size))

    return boxes2d


def decode_bbox_obb_xywhr(data):
    format_string = '<4if'
    group_size = struct.calcsize(format_string)
    bbox = data.get("bbox_obb", b"")
    boxes1d = np.array(
        [
            struct.unpack(format_string, bbox[i : i + group_size])
            for i in range(0, len(bbox), group_size)
            if i + group_size <= len(bbox)
        ]
    )
    return np.reshape(boxes1d, (-1, 5))


def decode_bbox_obb(data):
    is_xywhr = struct.unpack('?', data.get("is_xywhr", b'\x00'))[0]
    if is_xywhr:
        return decode_bbox_obb_xywhr(data)
    else:
        return decode_bbox_obb_xyxyxyxy(data)

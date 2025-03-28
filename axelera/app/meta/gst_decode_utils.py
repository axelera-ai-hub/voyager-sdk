# Copyright Axelera AI, 2023
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

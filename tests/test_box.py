# Copyright Axelera AI, 2023
import numpy as np
import pytest

from axelera import types
from axelera.app.model_utils.box import box_iou_1_to_many, convert


@pytest.mark.parametrize(
    "from_fmt,to_fmt,in_boxes,out_boxes",
    [
        (
            types.BoxFormat.XYXY,
            types.BoxFormat.XYWH,
            np.array([[0, 0, 10, 10], [20, 20, 30, 30]]),
            np.array([[5, 5, 10, 10], [25, 25, 10, 10]]),
        ),
        (
            types.BoxFormat.XYXY,
            types.BoxFormat.LTWH,
            np.array([[0, 0, 10, 10], [20, 20, 30, 30]]),
            np.array([[0, 0, 10, 10], [20, 20, 10, 10]]),
        ),
        (
            types.BoxFormat.XYWH,
            types.BoxFormat.LTWH,
            np.array([[5, 5, 10, 10], [25, 25, 10, 10]]),
            np.array([[0, 0, 10, 10], [20, 20, 10, 10]]),
        ),
    ],
)
def test_convert(from_fmt, to_fmt, in_boxes, out_boxes):
    input_boxes_copy = np.copy(in_boxes)
    output_boxes = convert(in_boxes, from_fmt, to_fmt)
    assert np.array_equal(output_boxes, out_boxes)
    assert np.array_equal(in_boxes, input_boxes_copy)

    # Reverse direction
    input_boxes_copy = np.copy(out_boxes)
    reverse_boxes = convert(out_boxes, to_fmt, from_fmt)
    assert np.array_equal(reverse_boxes, in_boxes)
    assert np.array_equal(out_boxes, input_boxes_copy)


# Tests that convert function raises an AssertionError when input_format or output_format is not valid.
def test_convert_with_invalid_input_output_formats():
    boxes = np.array([[100, 100, 200, 200], [300, 300, 400, 400]])
    with pytest.raises(AttributeError):
        convert(boxes, 'invalid_format', types.BoxFormat.XYXY)
    with pytest.raises(AttributeError):
        convert(boxes, types.BoxFormat.XYXY, 'invalid_format')


@pytest.mark.parametrize(
    "box1, bboxes2, expected_iou",
    [
        # Test case 1: box1 and bboxes2 have same coordinates
        (
            np.array([0, 0, 2, 2]),
            np.array([[0, 0, 2, 2], [0, 0, 2, 2], [0, 0, 2, 2]]),
            np.array([1.0, 1.0, 1.0]),
        ),
        # Test case 2: box1 and bboxes2 have no overlap
        (
            np.array([0, 0, 2, 2]),
            np.array([[5, 5, 7, 7], [10, 10, 12, 12], [15, 15, 17, 17]]),
            np.array([0.0, 0.0, 0.0]),
        ),
        # Test case 3: box1 and bboxes2 have partial overlap
        (
            np.array([0, 0, 3, 3]),
            np.array([[2, 2, 4, 4], [3, 3, 5, 5], [4, 4, 6, 6]]),
            np.array([0.19047619, 0.04166667, 0.0]),
        ),
    ],
)
def test_box_iou_1_to_many(box1, bboxes2, expected_iou):
    iou = box_iou_1_to_many(box1, bboxes2)
    assert np.allclose(iou, expected_iou, rtol=1e-5)

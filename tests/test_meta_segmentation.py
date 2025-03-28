# Copyright Axelera AI, 2024
import re
from unittest.mock import Mock, patch

from PIL import Image
import numpy as np
import pytest

from axelera.app import display_cv, plot_utils
from axelera.app.meta import ClassificationMeta, InstanceSegmentationMeta
from axelera.app.meta.segmentation import _translate_image_space_rect


def mock_draw(width, height):
    image = Image.fromarray(np.zeros((height, width, 3), np.uint8))
    draw = Mock()
    return draw, image


def mock_get_color(index):
    colors = [
        (255, 0, 0, 100),  # Red with alpha 100
        (0, 255, 0, 150),  # Green with alpha 150
        (0, 0, 255, 200),  # Blue with alpha 200
    ]
    return colors[index % len(colors)]


def test_draw_no_masks():
    draw, image = mock_draw(2, 2)
    meta = InstanceSegmentationMeta()

    with patch("axelera.app.meta.segmentation.LOG.warning") as mock_warning:
        with patch("axelera.app.display_cv.CVDraw", return_value=draw):
            display_draw = display_cv.CVDraw(image)
            meta.draw(display_draw)
            display_draw.draw()

            draw.segmentation_mask.assert_not_called()


def test_add_results():
    meta = InstanceSegmentationMeta()
    masks = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=np.uint8)
    boxes = np.array([[0, 0, 1, 1], [1, 1, 2, 2]], dtype=np.float32)
    class_ids = np.array([0, 1], dtype=np.int32)
    scores = np.array([0.9, 0.8], dtype=np.float32)

    meta.add_results(masks, boxes, class_ids, scores)

    assert len(meta._masks) == 2
    assert len(meta._boxes) == 2
    assert len(meta._class_ids) == 2
    assert len(meta._scores) == 2

    np.testing.assert_array_equal(meta.masks, masks)
    np.testing.assert_array_equal(meta.boxes, boxes)
    np.testing.assert_array_equal(meta.class_ids, class_ids)
    np.testing.assert_array_equal(meta.scores, scores)


def test_add_result():
    meta = InstanceSegmentationMeta()
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    box = np.array([0, 0, 1, 1], dtype=np.float32)
    class_id = 0
    score = 0.9

    meta.add_result(mask, box, class_id, score)

    assert len(meta._masks) == 1
    assert len(meta._boxes) == 1
    assert len(meta._class_ids) == 1
    assert len(meta._scores) == 1

    np.testing.assert_array_equal(meta.masks[0], mask)
    np.testing.assert_array_equal(meta.boxes[0], box)
    assert meta.class_ids[0] == class_id
    assert meta.scores[0] == score


def test_get_result():
    meta = InstanceSegmentationMeta()
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    box = np.array([0, 0, 1, 1], dtype=np.float32)
    class_id = 0
    score = 0.9

    meta.add_result(mask, box, class_id, score)

    result_mask, result_box, result_class_id, result_score = meta.get_result()

    np.testing.assert_array_equal(result_mask, mask)
    np.testing.assert_array_equal(result_box, box)
    assert result_class_id == class_id
    assert result_score == score


def test_get_result_out_of_range():
    meta = InstanceSegmentationMeta()
    with pytest.raises(IndexError):
        meta.get_result(1)


def test_transfer_data():
    meta1 = InstanceSegmentationMeta()
    meta2 = InstanceSegmentationMeta()

    mask1 = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    box1 = np.array([0, 0, 1, 1], dtype=np.float32)
    meta1.add_result(mask1, box1, 0, 0.9)

    mask2 = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    box2 = np.array([1, 1, 2, 2], dtype=np.float32)
    meta2.add_result(mask2, box2, 1, 0.8)

    meta1.transfer_data(meta2)

    assert len(meta1._masks) == 2
    assert len(meta1._boxes) == 2
    assert len(meta1._class_ids) == 2
    assert len(meta1._scores) == 2

    np.testing.assert_array_equal(meta1.masks[1], mask2)
    np.testing.assert_array_equal(meta1.boxes[1], box2)
    assert meta1.class_ids[1] == 1
    assert meta1.scores[1] == 0.8


def test_transfer_data_wrong_type():
    meta1 = InstanceSegmentationMeta()
    meta2 = ClassificationMeta()

    with pytest.raises(TypeError):
        meta1.transfer_data(meta2)


@pytest.mark.parametrize(
    "method,invalid_input,error_message",
    [
        (
            "add_result",
            (np.array([1, 2, 3]), np.array([0, 0, 1, 1]), 0, 0.9),
            "mask must be a 2D numpy array",
        ),
        (
            "add_result",
            (np.array([[1, 0], [0, 1]]), np.array([0, 0, 1]), 0, 0.9),
            r"box must be a 1D numpy array with shape (4,)",
        ),
        (
            "add_result",
            (np.array([[1, 0], [0, 1]]), np.array([0, 0, 1, 1]), 0.5, 0.9),
            "class_id must be an integer",
        ),
        (
            "add_result",
            (np.array([[1, 0], [0, 1]]), np.array([0, 0, 1, 1]), 0, np.array([0.9])),
            "score must be a single scalar value",
        ),
        (
            "add_results",
            (np.array([[[1, 0], [0, 1]]]), np.array([0, 0, 1, 1]), np.array([0]), np.array([0.9])),
            r"boxes must be a 2D numpy array with shape (N, 4)",
        ),
        (
            "add_results",
            (
                np.array([[[1, 0], [0, 1]]]),
                np.array([[0, 0, 1, 1]]),
                np.array([[0]]),
                np.array([0.9]),
            ),
            "class_ids must be a 1D numpy array",
        ),
        (
            "add_results",
            (
                np.array([[[1, 0], [0, 1]]]),
                np.array([[0, 0, 1, 1]]),
                np.array([0]),
                np.array([[0.9]]),
            ),
            "scores must be a 1D numpy array",
        ),
    ],
)
def test_invalid_inputs(method, invalid_input, error_message):
    meta = InstanceSegmentationMeta()
    with pytest.raises(ValueError, match=re.escape(error_message)):
        getattr(meta, method)(*invalid_input)


def test_transfer_data_empty():
    meta1 = InstanceSegmentationMeta()
    meta2 = InstanceSegmentationMeta()
    meta1.transfer_data(meta2)
    assert len(meta1._masks) == 0
    assert len(meta1._boxes) == 0
    assert len(meta1._class_ids) == 0
    assert len(meta1._scores) == 0


def test_properties():
    meta = InstanceSegmentationMeta()
    meta.add_result(np.array([[1, 0], [0, 1]]), np.array([0, 0, 1, 1]), 0, 0.9)
    meta.add_result(np.array([[0, 1], [1, 0]]), np.array([1, 1, 2, 2]), 1, 0.8)

    properties = ['masks', 'boxes', 'class_ids', 'scores']
    expected_shapes = [(2, 2), (2, 4), (2,), (2,)]

    for prop, shape in zip(properties, expected_shapes):
        value = getattr(meta, prop)
        if prop != 'masks':
            assert isinstance(value, np.ndarray)
            assert value.shape == shape
        else:
            assert isinstance(value, list)
            assert len(value) == 2
            assert isinstance(value[-1], np.ndarray)
            assert value[-1].shape == shape


@pytest.mark.parametrize(
    "method,expected",
    [
        ("xyxy", np.array([[0, 0, 1, 1], [1, 1, 2, 2]])),
        ("xywh", "xywh"),
        ("ltwh", "ltwh"),
    ],
)
def test_box_conversions(method, expected):
    meta = InstanceSegmentationMeta()
    meta.add_result(np.array([[1, 0], [0, 1]]), np.array([0, 0, 1, 1]), 0, 0.9)
    meta.add_result(np.array([[0, 1], [1, 0]]), np.array([1, 1, 2, 2]), 1, 0.8)

    if method == "xyxy":
        assert np.array_equal(getattr(meta, method)(), expected)
    else:
        with patch("axelera.app.meta.segmentation.box_utils.convert") as mock_convert:
            getattr(meta, method)()
            assert mock_convert.call_count == 1
            call_args = mock_convert.call_args[0]
            np.testing.assert_array_equal(call_args[0], meta.boxes)
            assert call_args[1] == 'xyxy'
            assert call_args[2] == method


def test_draw_empty():
    meta = InstanceSegmentationMeta()
    draw_mock = Mock()
    with patch("axelera.app.meta.segmentation.LOG.warning") as mock_warning:
        meta.draw(draw_mock)
        draw_mock.segmentation_mask.assert_not_called()


def test_translate_image_space_rect():
    input_roi = 958, 457, 958 + 165, 457 + 252
    bbox = 59, 43, 149, 77
    got = _translate_image_space_rect(bbox, input_roi, (160, 160))
    assert got == (1007, 524, 1149, 578)
    bbox = 66, 102, 88, 120
    got = _translate_image_space_rect(bbox, input_roi, (160, 160))
    assert got == (1018, 617, 1053, 646)

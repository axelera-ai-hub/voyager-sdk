# Copyright Axelera AI, 2024
import re
from unittest.mock import Mock, patch

from PIL import Image
import numpy as np
import pytest

from axelera import types
from axelera.app import plot_utils
from axelera.app.meta import ClassificationMeta, InstanceSegmentationMeta
from axelera.app.meta.segmentation import _translate_image_space_rect


class MockTaskRenderConfig:
    def __init__(self, show_labels=True, show_annotations=True):
        self.show_labels = show_labels
        self.show_annotations = show_annotations


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


def test_draw_no_masks(display_cv):
    draw, image = mock_draw(2, 2)
    meta = InstanceSegmentationMeta()
    container_meta = create_container_with_render_config("segmentation_meta")
    meta.set_container_meta(container_meta)
    object.__setattr__(meta, 'meta_name', "segmentation_meta")

    with patch("axelera.app.meta.segmentation.LOG.warning") as mock_warning:
        with patch("axelera.app.display_cv.CVDraw", return_value=draw):
            composite = types.Image.fromarray(np.zeros((1080, 1920, 3), dtype=np.uint8))
            display_draw = display_cv.CVDraw(0, 1, composite, image, [])
            meta.draw(display_draw)
            display_draw.draw()

            draw.segmentation_mask.assert_not_called()


def test_add_results():
    meta = InstanceSegmentationMeta()
    masks_3d = np.array([[[1, 0], [0, 1]], [[0, 1], [1, 0]]], dtype=np.uint8)
    masks = [(0, 0, 1, 1, 0, 0, 1, 1, masks_3d[0]), (0, 0, 1, 1, 0, 0, 1, 1, masks_3d[1])]
    boxes = np.array([[0, 0, 1, 1], [1, 1, 2, 2]], dtype=np.float32)
    class_ids = np.array([0, 1], dtype=np.int32)
    scores = np.array([0.9, 0.8], dtype=np.float32)

    meta.add_results(masks, boxes, class_ids, scores)

    assert len(meta._masks) == 2
    assert len(meta._boxes) == 2
    assert len(meta._class_ids) == 2
    assert len(meta._scores) == 2
    assert meta.masks[0][:8] == masks[0][:8]
    assert meta.masks[1][:8] == masks[1][:8]
    np.testing.assert_array_equal(meta.masks[0][-1], masks[0][-1])
    np.testing.assert_array_equal(meta.masks[1][-1], masks[1][-1])
    np.testing.assert_array_equal(meta.boxes, boxes)
    np.testing.assert_array_equal(meta.class_ids, class_ids)
    np.testing.assert_array_equal(meta.scores, scores)


def test_add_result():
    meta = InstanceSegmentationMeta()
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    box = np.array([0, 0, 1, 1], dtype=np.float32)
    class_id = 0
    score = 0.9

    meta.add_result((0, 0, 1, 1, 0, 0, 1, 1, mask), box, class_id, score)

    assert len(meta._masks) == 1
    assert len(meta._boxes) == 1
    assert len(meta._class_ids) == 1
    assert len(meta._scores) == 1

    np.testing.assert_array_equal(meta.masks[0][-1], mask)
    np.testing.assert_array_equal(meta.boxes[0], box)
    assert meta.class_ids[0] == class_id
    assert meta.scores[0] == score


def test_get_result():
    meta = InstanceSegmentationMeta()
    mask = np.array([[1, 0], [0, 1]], dtype=np.uint8)
    box = np.array([0, 0, 1, 1], dtype=np.float32)
    class_id = 0
    score = 0.9

    meta.add_result((0, 0, 1, 1, 0, 0, 1, 1, mask), box, class_id, score)

    result_mask, result_box, result_class_id, result_score = meta.get_result()

    np.testing.assert_array_equal(result_mask[-1], mask)
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
    meta1.add_result((0, 0, 1, 1, 0, 0, 1, 1, mask1), box1, 0, 0.9)

    mask2 = np.array([[0, 1], [1, 0]], dtype=np.uint8)
    box2 = np.array([1, 1, 2, 2], dtype=np.float32)
    meta2.add_result((0, 0, 1, 1, 1, 1, 2, 2, mask2), box2, 1, 0.8)

    meta1.transfer_data(meta2)

    assert len(meta1._masks) == 2
    assert len(meta1._boxes) == 2
    assert len(meta1._class_ids) == 2
    assert len(meta1._scores) == 2

    np.testing.assert_array_equal(meta1.masks[1][-1], mask2)
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
            ((np.array([1, 2, 3]),), np.array([0, 0, 1, 1]), 0, 0.9),
            "mask must be a SegmentationMask tuple",
        ),
        (
            "add_result",
            ((0, 0, 1, 1, 0, 0, 1, 1, np.array([3, 4, 5])), np.array([0, 0, 1, 1]), 0, 0.9),
            "mask must be a 2D numpy array",
        ),
        (
            "add_result",
            ((0, 0, 1, 1, 0, 0, 1, 1, np.array([[1, 0], [0, 1]])), np.array([0, 0, 1]), 0, 0.9),
            r"box must be a 1D numpy array with shape (4,)",
        ),
        (
            "add_result",
            (
                (0, 0, 1, 1, 0, 0, 1, 1, np.array([[1, 0], [0, 1]])),
                np.array([0, 0, 1, 1]),
                0.5,
                0.9,
            ),
            "class_id must be an integer",
        ),
        (
            "add_result",
            (
                (0, 0, 1, 1, 0, 0, 1, 1, np.array([[1, 0], [0, 1]])),
                np.array([0, 0, 1, 1]),
                0,
                np.array([0.9]),
            ),
            "score must be a single scalar value",
        ),
        (
            "add_results",
            (
                [(0, 0, 1, 1, 0, 0, 1, 1, np.array([[1, 0], [0, 1]]))],
                np.array([0, 0, 1, 1]),
                np.array([0]),
                np.array([0.9]),
            ),
            r"boxes must be a 2D numpy array with shape (N, 4)",
        ),
        (
            "add_results",
            (
                [(0, 0, 1, 1, 0, 0, 1, 1, np.array([[1, 0], [0, 1]]))],
                np.array([[0, 0, 1, 1]]),
                np.array([[0]]),
                np.array([0.9]),
            ),
            "class_ids must be a 1D numpy array",
        ),
        (
            "add_results",
            (
                [(0, 0, 1, 1, 0, 0, 1, 1, np.array([[1, 0], [0, 1]]))],
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
    meta.add_result(
        (0, 0, 1, 1, 0, 0, 1, 1, np.array([[1, 0], [0, 1]])), np.array([0, 0, 1, 1]), 0, 0.9
    )
    meta.add_result(
        (0, 0, 1, 1, 1, 1, 2, 2, np.array([[0, 1], [1, 0]])), np.array([1, 1, 2, 2]), 1, 0.8
    )

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
            assert isinstance(value[0][-1], np.ndarray)
            assert value[0][-1].shape == shape


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
    meta.add_result(
        (0, 0, 1, 1, 0, 0, 1, 1, np.array([[1, 0], [0, 1]])), np.array([0, 0, 1, 1]), 0, 0.9
    )
    meta.add_result(
        (0, 0, 1, 1, 1, 1, 2, 2, np.array([[0, 1], [1, 0]])), np.array([1, 1, 2, 2]), 1, 0.8
    )

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

    from axelera.app.config import RenderConfig
    from axelera.app.meta import AxMeta

    container_meta = AxMeta("test_image")
    render_config = RenderConfig()
    render_config.set_task(
        "segmentation_meta", show_labels=True, show_annotations=True, force_register=True
    )
    container_meta.set_render_config(render_config)

    # Set container_meta and meta_name properly
    meta.set_container_meta(container_meta)
    object.__setattr__(meta, 'meta_name', "segmentation_meta")

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


def create_container_with_render_config(meta_name, show_labels=True, show_annotations=True):
    """Create a container meta with proper render config for testing.

    Args:
        meta_name: The name to register for the meta in the render config
        show_labels: Whether to show labels in rendering
        show_annotations: Whether to show annotations in rendering

    Returns:
        container_meta: The configured AxMeta container
    """
    from axelera.app.config import RenderConfig
    from axelera.app.meta import AxMeta

    container_meta = AxMeta("test_image")
    render_config = RenderConfig()
    render_config.set_task(
        meta_name, show_labels=show_labels, show_annotations=show_annotations, force_register=True
    )
    container_meta.set_render_config(render_config)
    return container_meta


def test_decode_places_each_mask_on_its_detection_box():
    """The gst mask rect is the detection box, and that is what the renderer wants.

    A letterboxed model reaches the same rect either way, which is the point: the
    crop the decoder emits is the box mapped into mask space, so mapping it back
    can only reproduce the box or disagree with it. Numbers here are a 640x640
    model on a 1280x720 source, so the letterbox inverse carries a real 280 pixel
    offset rather than a degenerate one.
    """
    base_box = 0, 0, 1280, 720
    seg_shape = 160, 160
    mask_rect = 50, 60, 70, 85
    box = 400, 200, 560, 400
    mask = np.arange(20 * 25, dtype=np.uint8)

    data = {
        'bbox': np.array([box], dtype=np.int32).tobytes(),
        'segment_shape': np.array([1, seg_shape[1], seg_shape[0]], dtype=np.uint64).tobytes(),
        'segment_bboxs': np.array(mask_rect, dtype=np.int32).tobytes(),
        'base_boxes': np.array(base_box, dtype=np.int32).tobytes(),
        'segment_maps': mask.tobytes(),
        'scores': np.array([0.9], dtype=np.float32).tobytes(),
        'classes': np.array([3], dtype=np.int32).tobytes(),
    }

    meta = InstanceSegmentationMeta.decode(data)
    assert len(meta.masks) == 1
    entry = meta.get_mask(0)
    assert entry[:4] == mask_rect
    assert tuple(entry[4:8]) == box
    assert entry[-1].shape == (25, 20)
    assert _translate_image_space_rect(mask_rect, base_box, seg_shape) == box

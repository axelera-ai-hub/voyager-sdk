# Copyright Axelera AI, 2023
from dataclasses import FrozenInstanceError
import itertools
from unittest.mock import ANY, Mock, call, patch

from PIL import ImageDraw, ImageFont
import numpy as np
import pytest

from axelera.app import utils
from axelera.app.eval_interfaces import ObjDetEvalSample, ObjDetGroundTruthSample
from axelera.app.meta import DetectedObject, ObjectDetectionMeta

LABEL_BACK_COLOR = (0, 0, 0, 255)
LABEL_FORE_COLOR = (244, 190, 24, 255)

CLS0_COLOR = (255, 255, 000, 255)
CLS1_COLOR = (255, 178, 125, 255)
CLS2_COLOR = (255, 255, 255, 255)


class MockTaskRenderConfig:
    def __init__(self, show_labels=True, show_annotations=True):
        self.show_labels = show_labels
        self.show_annotations = show_annotations


def mock_draw(rt2_img, width, height, monkeypatch):
    mock = Mock(ImageDraw.ImageDraw)
    monkeypatch.setattr(ImageFont.FreeTypeFont, "getbbox", lambda *args, **kwargs: (0, 0, 40, 8))
    image = rt2_img.Image.from_array(np.zeros((height, width, 3), np.uint8), 'RGB')
    return mock, image


def mock_image_draw(draw):
    m = Mock()
    m.return_value = draw
    return m


@pytest.mark.parametrize(
    "boxes, scores, classes, expected_rectangles, expected_text, message",
    [
        pytest.param(
            np.array([[10.0, 10.0, 20.0, 20.0]]),
            np.array([0.3]),
            np.array([1]),
            [
                call(((10, 10), (20, 20)), None, CLS1_COLOR, 2),
                call(((10, 18), (50, 26)), LABEL_BACK_COLOR, None, 1),
            ],
            [call((10, 11), 'cls:1 30%', LABEL_FORE_COLOR, ANY)],
            "test box with no label names",
        ),
        pytest.param(
            np.array([[0.0, 0.0, 20.0, 20.0]]),
            np.array([0.3]),
            np.array([0]),
            [
                call(((0, 0), (20, 20)), None, CLS0_COLOR, 2),
                call(((0, 8), (40, 16)), LABEL_BACK_COLOR, None, 1),
            ],
            [call((0, 1), 'cls:0 30%', LABEL_FORE_COLOR, ANY)],
            "test box with no label names",
        ),
        pytest.param(
            np.array([[0.0, 0.0, 20.0, 20.0], [10.0, 10.0, 20.0, 20.0]]),
            np.array([0.3, 0.4]),
            np.array([0, 2]),
            [
                call(((0, 0), (20, 20)), None, CLS0_COLOR, 2),
                call(((0, 8), (40, 16)), LABEL_BACK_COLOR, None, 1),
                call(((10, 10), (20, 20)), None, CLS2_COLOR, 2),
                call(((10, 18), (50, 26)), LABEL_BACK_COLOR, None, 1),
            ],
            [
                call((0, 1), 'cls:0 30%', LABEL_FORE_COLOR, ANY),
                call((10, 11), 'cls:2 40%', LABEL_FORE_COLOR, ANY),
            ],
            "test multiple boxes with no label names",
        ),
    ],
)
def test_box_with_no_labels(
    display_cv,
    rt2_img,
    boxes,
    scores,
    classes,
    expected_rectangles,
    expected_text,
    message,
    monkeypatch,
):
    output_width = 1280
    output_height = 1280
    draw, image = mock_draw(rt2_img, output_width, output_height, monkeypatch)
    meta = ObjectDetectionMeta(boxes, scores, classes)
    container_meta = create_container_with_render_config("detection_meta")
    meta.set_container_meta(container_meta)
    object.__setattr__(meta, 'meta_name', "detection_meta")

    with patch("PIL.ImageDraw.Draw", mock_image_draw(draw)):
        composite = rt2_img.Image.from_array(
            np.zeros((output_height, output_width, 3), dtype=np.uint8), 'RGB'
        )
        display_draw = display_cv.CVDraw(0, 1, composite, image, [])
        meta.draw(display_draw)
        display_draw.draw()

        assert draw.rectangle.call_args_list == expected_rectangles, message
        assert draw.text.call_args_list == expected_text, message


@pytest.mark.parametrize(
    "boxes, scores, classes, expected_rectangles, expected_text, message",
    [
        pytest.param(
            np.array([[20.0, 20.0, 40.0, 40.0]]),
            np.array([0.3]),
            np.array([1]),
            [
                call(((20, 20), (40, 40)), None, CLS1_COLOR, 2),
                call(((20, 17), (60, 25)), LABEL_BACK_COLOR, None, 1),
            ],
            [call((20, 10), "car 30%", LABEL_FORE_COLOR, ANY)],
            "box with label names, name fits outside box",
        ),
        pytest.param(
            np.array([[20.0, 10.0, 40.0, 40.0]]),
            np.array([0.3]),
            np.array([1]),
            [
                call(((20, 10), (40, 40)), None, CLS1_COLOR, 2),
                call(((20, 18), (60, 26)), LABEL_BACK_COLOR, None, 1),
            ],
            [call((20, 11), "car 30%", LABEL_FORE_COLOR, ANY)],
            "box with label names, name does not fit outside box",
        ),
        pytest.param(
            np.array([[20.0, 10.0, 40.0, 40.0], [20.0, 20.0, 40.0, 40.0]]),
            np.array([0.3, 0.3]),
            np.array([1, 0]),
            [
                call(((20, 10), (40, 40)), None, CLS1_COLOR, 2),
                call(((20, 18), (60, 26)), LABEL_BACK_COLOR, None, 1),
                call(((20, 20), (40, 40)), None, CLS0_COLOR, 2),
                call(((20, 17), (60, 25)), LABEL_BACK_COLOR, None, 1),
            ],
            [
                call((20, 11), "car 30%", ANY, ANY),
                call((20, 10), "person 30%", ANY, ANY),
            ],
            "box with label names, multiple boxes both inside and outside",
        ),
    ],
)
@pytest.mark.parametrize(
    "labels",
    [
        ["person", "car", "bus"],
        utils.FrozenIntEnum("TestDataset", zip(["person", "car", "bus"], itertools.count())),
    ],
)
def test_box_with_labels(
    display_cv,
    rt2_img,
    boxes,
    scores,
    classes,
    expected_rectangles,
    expected_text,
    message,
    labels,
    monkeypatch,
):
    output_width = 1280
    output_height = 1280
    draw, image = mock_draw(rt2_img, output_width, output_height, monkeypatch)
    meta = ObjectDetectionMeta(boxes, scores, classes, labels=labels)

    container_meta = create_container_with_render_config("detection_meta")
    meta.set_container_meta(container_meta)
    object.__setattr__(meta, 'meta_name', "detection_meta")

    with patch("PIL.ImageDraw.Draw", mock_image_draw(draw)):
        composite = rt2_img.Image.from_array(
            np.zeros((output_height, output_width, 3), dtype=np.uint8), 'RGB'
        )
        display_draw = display_cv.CVDraw(0, 1, composite, image, [])
        meta.draw(display_draw)
        display_draw.draw()

        assert draw.rectangle.call_args_list == expected_rectangles, message
        assert draw.text.call_args_list == expected_text


def test_meta_immutable():
    boxes = np.array([[20.0, 10.0, 40.0, 40.0]])
    scores = np.array([0.3])
    classes = np.array([1])
    labels = ["person", "car", "bus"]
    extra_info = {"test": 123}
    meta = ObjectDetectionMeta.create_immutable_meta(
        boxes, scores, classes, labels=labels, extra_info=extra_info
    )

    with pytest.raises(FrozenInstanceError):
        meta.boxes = np.array([[20.0, 10.0, 40.0, 40.0]])
    with pytest.raises(ValueError):
        meta.boxes[0][0] = 0.0
    with pytest.raises(FrozenInstanceError):
        meta.scores = np.array([0.3])
    with pytest.raises(FrozenInstanceError):
        meta.classes = np.array([1])
    with pytest.raises(FrozenInstanceError):
        meta.labels = ["cat", "dog", "rabbit"]
    with pytest.raises(TypeError):
        meta.labels[0] = 'cat'
    with pytest.raises(FrozenInstanceError):
        meta.extra_info = {"test": 12, "test2": 123}
    with pytest.raises(TypeError):
        meta.extra_info["test"] = 12
    with pytest.raises(TypeError):
        meta.extra_info["test2"] = 123

    labels_enum = utils.FrozenIntEnum("TestDataset", zip(labels, itertools.count()))
    meta_labels_enum = ObjectDetectionMeta.create_immutable_meta(
        boxes, scores, classes, labels=labels_enum, extra_info=extra_info
    )
    with pytest.raises(AttributeError):
        meta_labels_enum.labels.cat = 1
    with pytest.raises(AttributeError):
        meta_labels_enum.labels.car = 3


def test_to_evaluation_correct_ground_truth():
    torch = pytest.importorskip("torch")
    meta = ObjectDetectionMeta(
        boxes=np.array([[10, 10, 50, 50], [2, 10, 30, 40]]),
        scores=np.array([0.9, 0.8]),
        class_ids=np.array([0, 1]),
    )
    ground_truth = ObjDetGroundTruthSample(
        boxes=np.array([[10, 10, 50, 50], [2, 10, 30, 40]]), labels=np.array([0, 1])
    )
    object.__setattr__(meta, 'access_ground_truth', lambda: ground_truth)
    eval_sample = meta.to_evaluation()
    assert isinstance(eval_sample, ObjDetEvalSample)
    assert np.array_equal(eval_sample.labels, np.array([0, 1]))
    assert np.array_equal(eval_sample.scores, np.array([0.9, 0.8]))
    assert np.array_equal(eval_sample.boxes, np.array([[10, 10, 50, 50], [2, 10, 30, 40]]))


def test_to_evaluation_ground_truth_not_accessable():
    torch = pytest.importorskip("torch")
    meta = ObjectDetectionMeta(
        boxes=np.array([[10, 10, 50, 50], [2, 10, 30, 40]]),
        scores=np.array([0.9, 0.8]),
        class_ids=np.array([0, 1]),
    )
    object.__setattr__(meta, 'container_meta', None)
    with pytest.raises(ValueError):
        meta.to_evaluation()


def test_to_evaluation_not_implemented():
    container_meta = Mock()
    torch = pytest.importorskip("torch")
    meta = ObjectDetectionMeta(
        boxes=np.array([[10, 10, 50, 50], [2, 10, 30, 40]]),
        scores=np.array([0.9, 0.8]),
        class_ids=np.array([0, 1]),
    )
    object.__setattr__(meta, 'container_meta', container_meta)
    with pytest.raises(NotImplementedError):
        meta.to_evaluation()


def test_object_detection_meta_objects():
    boxes = np.array([[10.0, 10.0, 20.0, 20.0], [30.0, 30.0, 40.0, 40.0]])
    scores = np.array([0.3, 0.4])
    classes = np.array([1, 2])
    meta = ObjectDetectionMeta(boxes, scores, classes)
    detections = meta.objects
    assert len(detections) == 2
    assert isinstance(detections[0], DetectedObject)
    assert isinstance(detections[1], DetectedObject)
    assert np.array_equal(detections[0].box, [10.0, 10.0, 20.0, 20.0])
    assert detections[0].score == 0.3
    assert detections[0].class_id == 1
    assert np.array_equal(detections[1].box, [30.0, 30.0, 40.0, 40.0])
    assert detections[1].score == 0.4
    assert detections[1].class_id == 2


def test_object_detection_meta_objects_tuple_labels():
    boxes = np.array([[10.0, 10.0, 20.0, 20.0]])
    scores = np.array([0.3])
    classes = np.array([1])
    labels = ["person", "car", "bus"]
    meta = ObjectDetectionMeta(boxes, scores, classes, labels=labels)
    detections = meta.objects
    with pytest.raises(NotImplementedError) as e:
        detections[0].label
        assert str(e.value) == "DetectedObject.label is not available for non-enum labels"
    with pytest.raises(NotImplementedError) as e:
        detections[0].is_car
        assert str(e.value) == "DetectedObject.is_car is not available for non-enum labels"


def test_object_detection_meta_objects_enum_labels():
    boxes = np.array([[10.0, 10.0, 20.0, 20.0]])
    scores = np.array([0.3])
    classes = np.array([1])
    labels = utils.FrozenIntEnum("TestDataset", zip(["person", "car", "bus"], itertools.count()))
    meta = ObjectDetectionMeta(boxes, scores, classes, labels=labels)
    detections = meta.objects
    assert detections[0].label.name == "car"
    assert detections[0].label == labels.car
    assert detections[0].is_car
    assert not detections[0].is_person
    assert not detections[0].is_bus


def test_object_detection_meta_objects_enum_labels_is_a():
    boxes = np.array([[10.0, 10.0, 20.0, 20.0]])
    scores = np.array([0.3])
    classes = np.array([1])
    labels = utils.FrozenIntEnum("TestDataset", zip(["person", "car", "bus"], itertools.count()))
    meta = ObjectDetectionMeta(boxes, scores, classes, labels=labels)
    detections = meta.objects
    assert detections[0].is_a("car")
    assert not detections[0].is_a("person")
    assert not detections[0].is_a("bus")


def test_object_detection_meta_objects_enum_labels_is_a_tuple():
    boxes = np.array(
        [[10.0, 10.0, 20.0, 20.0], [10.0, 10.0, 20.0, 20.0], [10.0, 10.0, 20.0, 20.0]]
    )
    scores = np.array([0.3, 0.3, 0.3])
    classes = np.array([0, 1, 2])
    labels = utils.FrozenIntEnum("TestDataset", zip(["person", "car", "bus"], itertools.count()))
    meta = ObjectDetectionMeta(boxes, scores, classes, labels=labels)
    detections = meta.objects
    assert not detections[0].is_a(("car", "bus"))
    assert detections[1].is_a(("car", "bus"))
    assert detections[2].is_a(("car", "bus"))


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

# Copyright Axelera AI, 2024
from dataclasses import FrozenInstanceError
import itertools
from unittest.mock import ANY, Mock, call, patch

from PIL import ImageDraw, ImageFont
import numpy as np
import pytest

from axelera import types
from axelera.app import display_cv, utils
from axelera.app.eval_interfaces import ObjDetEvalSample, ObjDetGroundTruthSample
from axelera.app.meta import DetectedObject, ObjectDetectionMeta

LABEL_BACK_COLOR = (0, 0, 0, 255)
LABEL_FORE_COLOR = (244, 190, 24, 255)

CLS0_COLOR = (255, 0, 0, 255)
CLS1_COLOR = (255, 178, 125, 255)
CLS2_COLOR = (255, 255, 255, 255)


def mock_draw(width, height, monkeypatch):
    mock = Mock(ImageDraw.ImageDraw)
    monkeypatch.setattr(ImageFont.FreeTypeFont, "getbbox", lambda *args, **kwargs: (0, 0, 40, 8))
    image = types.Image.fromarray(np.zeros((height, width, 3), np.uint8))
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
                call(((10, 11), (50, 19)), LABEL_BACK_COLOR, None, 1),
            ],
            [call((10, 11), 'cls:1 0.30', LABEL_FORE_COLOR, ANY)],
            "test box with no label names",
        ),
        pytest.param(
            np.array([[0.0, 0.0, 20.0, 20.0]]),
            np.array([0.3]),
            np.array([0]),
            [
                call(((0, 0), (20, 20)), None, CLS0_COLOR, 2),
                call(((0, 1), (40, 9)), LABEL_BACK_COLOR, None, 1),
            ],
            [call((0, 1), 'cls:0 0.30', LABEL_FORE_COLOR, ANY)],
            "test box with no label names",
        ),
        pytest.param(
            np.array([[0.0, 0.0, 20.0, 20.0], [10.0, 10.0, 20.0, 20.0]]),
            np.array([0.3, 0.4]),
            np.array([0, 2]),
            [
                call(((0, 0), (20, 20)), None, CLS0_COLOR, 2),
                call(((0, 1), (40, 9)), LABEL_BACK_COLOR, None, 1),
                call(((10, 10), (20, 20)), None, CLS2_COLOR, 2),
                call(((10, 11), (50, 19)), LABEL_BACK_COLOR, None, 1),
            ],
            [
                call((0, 1), 'cls:0 0.30', LABEL_FORE_COLOR, ANY),
                call((10, 11), 'cls:2 0.40', LABEL_FORE_COLOR, ANY),
            ],
            "test multiple boxes with no label names",
        ),
    ],
)
def test_box_with_no_labels(
    boxes, scores, classes, expected_rectangles, expected_text, message, monkeypatch
):
    output_width = 1280
    output_height = 1280
    draw, image = mock_draw(output_width, output_height, monkeypatch)
    meta = ObjectDetectionMeta(boxes, scores, classes)
    with patch("PIL.ImageDraw.Draw", mock_image_draw(draw)):
        display_draw = display_cv.CVDraw(image)
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
                call(((20, 10), (60, 18)), LABEL_BACK_COLOR, None, 1),
            ],
            [call((20, 10), "car 0.30", LABEL_FORE_COLOR, ANY)],
            "box with label names, name fits outside box",
        ),
        pytest.param(
            np.array([[20.0, 10.0, 40.0, 40.0]]),
            np.array([0.3]),
            np.array([1]),
            [
                call(((20, 10), (40, 40)), None, CLS1_COLOR, 2),
                call(((20, 11), (60, 19)), LABEL_BACK_COLOR, None, 1),
            ],
            [call((20, 11), "car 0.30", LABEL_FORE_COLOR, ANY)],
            "box with label names, name does not fit outside box",
        ),
        pytest.param(
            np.array([[20.0, 10.0, 40.0, 40.0], [20.0, 20.0, 40.0, 40.0]]),
            np.array([0.3, 0.3]),
            np.array([1, 0]),
            [
                call(((20, 10), (40, 40)), None, CLS1_COLOR, 2),
                call(((20, 11), (60, 19)), LABEL_BACK_COLOR, None, 1),
                call(((20, 20), (40, 40)), None, CLS0_COLOR, 2),
                call(((20, 10), (60, 18)), LABEL_BACK_COLOR, None, 1),
            ],
            [
                call((20, 11), "car 0.30", ANY, ANY),
                call((20, 10), "person 0.30", ANY, ANY),
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
    boxes, scores, classes, expected_rectangles, expected_text, message, labels, monkeypatch
):
    output_width = 1280
    output_height = 1280
    draw, image = mock_draw(output_width, output_height, monkeypatch)
    meta = ObjectDetectionMeta(boxes, scores, classes, labels=labels)
    with patch("PIL.ImageDraw.Draw", mock_image_draw(draw)):
        display_draw = display_cv.CVDraw(image)
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

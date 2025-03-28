# Copyright Axelera AI, 2024
from unittest.mock import ANY, Mock, call, patch

from PIL import ImageDraw, ImageFont
import numpy as np
import pytest

from axelera import types
from axelera.app import display_cv
from axelera.app.eval_interfaces import ClassificationEvalSample, ClassificationGroundTruthSample
from axelera.app.meta import ClassificationMeta, ClassifiedObject

LABEL_FORE_COLOR = (244, 190, 24, 255)
LABEL_BACK_COLOR = (0, 0, 0, 255)

CLS0_COLOR = (255, 0, 0, 255)
CLS1_COLOR = (255, 178, 125, 255)
CLS2_COLOR = (255, 255, 255, 255)


def mock_draw(width, height, boxes, scores, class_ids, labels, monkeypatch):
    mock = Mock(ImageDraw.ImageDraw)
    monkeypatch.setattr(ImageFont.FreeTypeFont, "getbbox", lambda *args, **kwargs: (0, 0, 40, 8))
    meta = ClassificationMeta(labels=labels)
    for score, class_id in zip(scores, class_ids):
        meta.add_result(class_id, score)

    mock_master_meta = Mock()
    mock_master_meta.boxes = [boxes]

    object.__setattr__(meta, 'master_meta_name', "mock_master_meta")
    object.__setattr__(meta, 'get_master_meta', lambda: mock_master_meta)
    object.__setattr__(meta, 'subframe_index', 0)

    image = types.Image.fromarray(np.zeros((height, width, 3), np.uint8))
    return mock, meta, image


def mock_image_draw(draw):
    m = Mock()
    m.return_value = draw
    return m


@pytest.mark.parametrize(
    "class_ids, labels, map_color, texts",
    [
        pytest.param([0, 1], None, lambda x: CLS0_COLOR, ["0", "1"]),
        pytest.param([0, 1], ["car", "person"], lambda x: x, ["car", "person"]),
    ],
)
def test_colors_text(class_ids, labels, map_color, texts, monkeypatch):
    output_width = 1280
    output_height = 1280
    draw, meta, image = mock_draw(
        output_width,
        output_height,
        [[10.0, 10.0, 20.0, 20.0]] * len(class_ids),
        [[0.34]] * len(class_ids),
        [[x] for x in class_ids],
        labels,
        monkeypatch,
    )
    with patch("PIL.ImageDraw.Draw", mock_image_draw(draw)):
        display_draw = display_cv.CVDraw(image)
        meta.draw(display_draw)
        display_draw.draw()

        assert draw.rectangle.call_args_list == [
            call(ANY, ANY, CLS0_COLOR, ANY),
            call(ANY, ANY, None, ANY),
            call(ANY, ANY, CLS1_COLOR, ANY),
            call(ANY, ANY, None, ANY),
        ]

        assert draw.text.call_args_list == [call(ANY, f'{text} 0.34', ANY, ANY) for text in texts]


@pytest.mark.parametrize(
    "class_ids, labels, boxes, scores, expected_rectangles, expected_text, message",
    [
        pytest.param(
            [0, 1],
            None,
            np.array([[10.0, 10.0, 20.0, 20.0]]),
            [np.array([0.3])],
            [
                call(((10, 10), (20, 20)), None, CLS0_COLOR, 2),
                call(((10, 11), (50, 19)), LABEL_BACK_COLOR, None, 1),
            ],
            [call((10, 11), "0 0.30", LABEL_FORE_COLOR, ANY)],
            "test box - not outside - class string",
        ),
        pytest.param(
            [0, 1],
            ["car", "person"],
            np.array([[10.0, 10.0, 20.0, 20.0]]),
            [np.array([0.3])],
            [
                call(((10, 10), (20, 20)), None, CLS0_COLOR, 2),
                call(((10, 11), (50, 19)), LABEL_BACK_COLOR, None, 1),
            ],
            [call((10, 11), "car 0.30", LABEL_FORE_COLOR, ANY)],
            "test box - not outside - label string",
        ),
        pytest.param(
            [0, 1],
            None,
            np.array([[10.0, 20.0, 20.0, 20.0]]),
            [np.array([0.3])],
            [
                call(((10, 20), (20, 20)), None, CLS0_COLOR, 2),
                call(((10, 10), (50, 18)), LABEL_BACK_COLOR, None, 1),
            ],
            [call((10, 10), "0 0.30", LABEL_FORE_COLOR, ANY)],
            "test box - outside - class string",
        ),
        pytest.param(
            [0, 1],
            ["car", "person"],
            np.array([[10.0, 20.0, 20.0, 20.0]]),
            [np.array([0.3])],
            [
                call(((10, 20), (20, 20)), None, CLS0_COLOR, 2),
                call(((10, 10), (50, 18)), LABEL_BACK_COLOR, None, 1),
            ],
            [call((10, 10), "car 0.30", LABEL_FORE_COLOR, ANY)],
            "test box - outside - label string",
        ),
        pytest.param(
            [0, 1],
            None,
            np.array([[10.0, 10.0, 20.0, 20.0], [10.0, 20.0, 20.0, 20.0]]),
            [np.array([0.3]), np.array([0.4])],
            [
                call(((10, 10), (20, 20)), None, CLS0_COLOR, 2),
                call(((10, 11), (50, 19)), LABEL_BACK_COLOR, None, 1),
                call(((10, 20), (20, 20)), None, ANY, 2),
                call(((10, 10), (50, 18)), LABEL_BACK_COLOR, None, 1),
            ],
            [
                call((10, 11), "0 0.30", LABEL_FORE_COLOR, ANY),
                call((10, 10), "1 0.40", LABEL_FORE_COLOR, ANY),
            ],
            "test boxes - inside and outside - class string",
        ),
        pytest.param(
            [0, 1],
            ["car", "person"],
            np.array([[10.0, 10.0, 20.0, 20.0], [10.0, 20.0, 20.0, 20.0]]),
            [np.array([0.3]), np.array([0.4])],
            [
                call(((10, 10), (20, 20)), None, CLS0_COLOR, 2),
                call(((10, 11), (50, 19)), LABEL_BACK_COLOR, None, 1),
                call(((10, 20), (20, 20)), None, ANY, 2),
                call(((10, 10), (50, 18)), LABEL_BACK_COLOR, None, 1),
            ],
            [
                call((10, 11), "car 0.30", LABEL_FORE_COLOR, ANY),
                call((10, 10), "person 0.40", LABEL_FORE_COLOR, ANY),
            ],
            "test boxes - inside and outside - label string",
        ),
    ],
)
def test_classification_box_cls_known(
    class_ids,
    labels,
    boxes,
    scores,
    expected_rectangles,
    expected_text,
    message,
    monkeypatch,
):
    output_width = 1280
    output_height = 1280
    draw, meta, image = mock_draw(
        output_width,
        output_height,
        boxes,
        scores,
        [[x] for x in class_ids],
        labels,
        monkeypatch,
    )
    with patch("PIL.ImageDraw.Draw", mock_image_draw(draw)):
        display_draw = display_cv.CVDraw(image)
        meta.draw(display_draw)
        display_draw.draw()

        assert draw.rectangle.call_args_list == expected_rectangles, message
        assert draw.text.call_args_list == expected_text, message


def test_get_result_method():
    meta = ClassificationMeta()
    meta.add_result([1], [0.9])
    meta.add_result([2], [0.8])
    meta.add_result([3], [0.7])

    class_ids, scores = meta.get_result(0)
    assert class_ids == [1]
    assert scores == [0.9]
    class_ids, scores = meta.get_result(1)
    assert class_ids == [2]
    assert scores == [0.8]
    class_ids, scores = meta.get_result(2)
    assert class_ids == [3]
    assert scores == [0.7]

    # Test getting an out-of-range instance
    with pytest.raises(AssertionError):
        meta.get_result(3)


def test_len_method():
    meta = ClassificationMeta()
    assert len(meta) == 0

    meta.add_result([1], [0.9])
    assert len(meta) == 1

    meta.add_result([2], [0.8])
    assert len(meta) == 2


def test_to_evaluation_correct_ground_truth():
    meta = ClassificationMeta(num_classes=2)
    meta.add_result([0], [0.9])
    ground_truth = ClassificationGroundTruthSample(class_id=0)
    # Use a workaround to set the lambda for a frozen dataclass
    object.__setattr__(meta, 'access_ground_truth', lambda: ground_truth)
    eval_sample = meta.to_evaluation()
    assert isinstance(eval_sample, ClassificationEvalSample)
    assert eval_sample.class_ids == [0]
    assert eval_sample.scores == [0.9]


def test_to_evaluation_incorrect_ground_truth():
    meta = ClassificationMeta(num_classes=2)
    meta.add_result([0], [0.9])
    # Set parent_meta to None explicitly
    object.__setattr__(meta, 'parent_meta', None)
    with pytest.raises(ValueError):
        meta.to_evaluation()


def test_to_evaluation_unsupported_ground_truth():
    meta = ClassificationMeta(num_classes=2)
    meta.add_result([0], [0.9])
    ground_truth = Mock()
    object.__setattr__(meta, 'access_ground_truth', lambda: ground_truth)
    with pytest.raises(NotImplementedError):
        meta.to_evaluation()


def test_to_evaluation_unexpected_class_ids():
    meta = ClassificationMeta(num_classes=2)
    meta.add_result([0], [0.9])
    meta.add_result([1], [0.8])
    ground_truth = ClassificationGroundTruthSample(class_id=0)
    object.__setattr__(meta, 'access_ground_truth', lambda: ground_truth)
    with pytest.raises(ValueError):
        meta.to_evaluation()


@pytest.mark.parametrize(
    "class_ids, scores, expected_class_ids, expected_scores",
    [
        (
            [1],
            [0.9],
            [[1], [1], [2], [1]],
            [[0.9], [0.9], [0.8], [0.9]],
        ),
        (
            1,
            0.9,
            [[1], [1], [2], [1]],
            [[0.9], [0.9], [0.8], [0.9]],
        ),
        (
            np.array(2),
            np.array(0.8),
            [[1], [1], [2], [2]],
            [[0.9], [0.9], [0.8], [0.8]],
        ),
        (
            np.array([3]),
            np.array([0.7]),
            [[1], [1], [2], [3]],
            [[0.9], [0.9], [0.8], [0.7]],
        ),
    ],
)
def test_add_result_method(class_ids, scores, expected_class_ids, expected_scores):
    meta = ClassificationMeta()

    meta.add_result([1], [0.9])
    meta.add_result(1, 0.9)
    meta.add_result([2], [0.8])

    meta.add_result(class_ids, scores)

    assert meta._class_ids == expected_class_ids
    assert meta._scores == expected_scores


@pytest.mark.parametrize(
    "class_ids, scores, error_type",
    [
        (
            [1, 2],
            [0.9],
            AssertionError,
        ),
        (
            [1],
            [0.9, 0.8],
            AssertionError,
        ),
        (
            [[1]],
            [0.9],
            TypeError,
        ),
        (
            ['1'],
            [0.9],
            TypeError,
        ),
        (
            [1],
            ['0.9'],
            TypeError,
        ),
    ],
)
def test_add_result_error_cases(class_ids, scores, error_type):
    meta = ClassificationMeta()

    with pytest.raises(error_type):
        meta.add_result(class_ids, scores)


@pytest.mark.parametrize(
    "class_ids, scores",
    [
        (
            {"class_id": 1},
            [0.9],
        ),
        (
            [1],
            {"score": 0.9},
        ),
    ],
)
def test_add_result_invalid_types(class_ids, scores):
    meta = ClassificationMeta()

    with pytest.raises(ValueError):
        meta.add_result(class_ids, scores)


def test_transfer_data():
    source = ClassificationMeta()
    source.add_result([1], [0.9])
    source.add_result([2], [0.8])
    source.add_result([3], [0.7])

    destination = ClassificationMeta()
    destination.add_result([4], [0.6])

    # Store memory addresses before transfer
    source_class_ids_id = id(source._class_ids)
    source_scores_id = id(source._scores)

    destination_class_ids_id = id(destination._class_ids)
    destination_scores_id = id(destination._scores)

    destination.transfer_data(source)

    # Verify data in destination
    assert len(destination) == 4
    assert destination._class_ids == [[4], [1], [2], [3]]
    assert destination._scores == [[0.6], [0.9], [0.8], [0.7]]

    # Check that source is unchanged
    assert len(source) == 3
    assert source._class_ids == [[1], [2], [3]]
    assert source._scores == [[0.9], [0.8], [0.7]]

    # Verify that memory addresses haven't changed
    assert id(source._class_ids) == source_class_ids_id
    assert id(source._scores) == source_scores_id

    assert id(destination._class_ids) == destination_class_ids_id
    assert id(destination._scores) == destination_scores_id

    # Verify that the transferred data in destination is not a copy
    assert id(destination._class_ids[1]) == id(source._class_ids[0])
    assert id(destination._scores[1]) == id(source._scores[0])


def test_transfer_data_empty_source():
    source = ClassificationMeta()
    destination = ClassificationMeta()
    destination.add_result([1], [0.9])

    destination.transfer_data(source)

    assert len(destination) == 1
    assert destination._class_ids == [[1]]
    assert destination._scores == [[0.9]]


def test_transfer_data_empty_destination():
    source = ClassificationMeta()
    source.add_result([1], [0.9])
    source.add_result([2], [0.8])

    destination = ClassificationMeta()

    destination.transfer_data(source)

    assert len(destination) == 2
    assert destination._class_ids == [[1], [2]]
    assert destination._scores == [[0.9], [0.8]]


def test_classification_meta_objects():
    meta = ClassificationMeta()
    meta.add_result([1], [0.9])
    meta.add_result([2], [0.8])

    classifications = meta.objects
    assert len(classifications) == 2
    assert isinstance(classifications[0], ClassifiedObject)
    assert classifications[0].class_id == [1]
    assert classifications[0].score == [0.9]
    assert isinstance(classifications[1], ClassifiedObject)
    assert classifications[1].class_id == [2]
    assert classifications[1].score == [0.8]

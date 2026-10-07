# Copyright Axelera AI, 2024
from unittest.mock import ANY, Mock, call, patch

from PIL import ImageDraw, ImageFont
import numpy as np
import pytest

from axelera.app.eval_interfaces import ClassificationEvalSample, ClassificationGroundTruthSample
from axelera.app.meta import AxMeta, ClassificationMeta, ClassifiedObject

LABEL_FORE_COLOR = (244, 190, 24, 255)
LABEL_BACK_COLOR = (0, 0, 0, 255)

CLS0_COLOR = (255, 255, 000, 255)
CLS1_COLOR = (255, 178, 125, 255)
CLS2_COLOR = (255, 255, 255, 255)


def mock_draw(
    rt2_img, width, height, boxes, scores, class_ids, labels, monkeypatch, softmax=False
):
    mock = Mock(ImageDraw.ImageDraw)
    monkeypatch.setattr(ImageFont.FreeTypeFont, "getbbox", lambda *args, **kwargs: (0, 0, 40, 8))
    meta = ClassificationMeta(labels=labels, extra_info={"softmax": softmax})
    for score, class_id in zip(scores, class_ids):
        meta.add_result(class_id, score)

    mock_master_meta = Mock()
    mock_master_meta.boxes = [boxes]

    container_meta = create_container_with_render_config(
        "mock_meta", show_labels=True, show_annotations=True
    )
    meta.set_container_meta(container_meta)
    object.__setattr__(meta, 'master_meta_name', "mock_master_meta")
    object.__setattr__(meta, 'get_master_meta', lambda: mock_master_meta)
    object.__setattr__(meta, 'subframe_index', 0)
    object.__setattr__(meta, 'meta_name', "mock_meta")

    image = rt2_img.Image.from_array(np.zeros((height, width, 3), np.uint8), 'RGB')
    return mock, meta, image


def mock_image_draw(draw):
    m = Mock()
    m.return_value = draw
    return m


class MockTaskRenderConfig:
    def __init__(self, show_labels=True, show_annotations=True):
        self.show_labels = show_labels
        self.show_annotations = show_annotations


@pytest.mark.parametrize(
    "class_ids, labels, map_color, texts",
    [
        pytest.param([0, 1], None, lambda x: CLS0_COLOR, ["0", "1"]),
        pytest.param([0, 1], ["car", "person"], lambda x: x, ["car", "person"]),
    ],
)
def test_colors_text(display_cv, rt2_img, class_ids, labels, map_color, texts, monkeypatch):
    output_width = 1280
    output_height = 1280
    draw, meta, image = mock_draw(
        rt2_img,
        output_width,
        output_height,
        [[10.0, 10.0, 20.0, 20.0]] * len(class_ids),
        [[3.4]] * len(class_ids),
        [[x] for x in class_ids],
        labels,
        monkeypatch,
    )

    with patch("PIL.ImageDraw.Draw", mock_image_draw(draw)):
        composite = rt2_img.Image.from_array(
            np.zeros((output_height, output_width, 3), dtype=np.uint8), 'RGB'
        )
        display_draw = display_cv.CVDraw(0, 1, composite, image, [])
        meta.draw(display_draw)
        display_draw.draw()

        assert draw.rectangle.call_args_list == [
            call(ANY, ANY, CLS0_COLOR, ANY),
            call(ANY, ANY, None, ANY),
            call(ANY, ANY, CLS1_COLOR, ANY),
            call(ANY, ANY, None, ANY),
        ]

        # When labels are provided, the format is "label score", otherwise it's "cls:label score"
        expected_texts = []
        for text in texts:
            if labels:
                expected_texts.append(call(ANY, f'{text} 3', ANY, ANY))
            else:
                expected_texts.append(call(ANY, f'cls:{text} 3', ANY, ANY))
        assert draw.text.call_args_list == expected_texts


@pytest.mark.parametrize(
    "class_ids, labels, texts",
    [
        pytest.param([0, 1], None, ["0", "1"]),
        pytest.param([0, 1], ["car", "person"], ["car", "person"]),
    ],
)
def test_label_softmax(display_cv, rt2_img, class_ids, labels, texts, monkeypatch):
    output_width = 1280
    output_height = 1280
    draw, meta, image = mock_draw(
        rt2_img,
        output_width,
        output_height,
        [[10.0, 10.0, 20.0, 20.0]] * len(class_ids),
        [[0.34]] * len(class_ids),
        [[x] for x in class_ids],
        labels,
        monkeypatch,
        softmax=True,
    )

    with patch("PIL.ImageDraw.Draw", mock_image_draw(draw)):
        composite = rt2_img.Image.from_array(
            np.zeros((output_height, output_width, 3), dtype=np.uint8), 'RGB'
        )
        display_draw = display_cv.CVDraw(0, 1, composite, image, [])
        meta.draw(display_draw)
        display_draw.draw()

        # When labels are provided, the format is "label score", otherwise it's "cls:label score"
        expected_texts = []
        for text in texts:
            if labels:
                expected_texts.append(call(ANY, f'{text} 34%', ANY, ANY))
            else:
                expected_texts.append(call(ANY, f'cls:{text} 34%', ANY, ANY))
        assert draw.text.call_args_list == expected_texts


@pytest.mark.parametrize(
    "class_ids, labels, boxes, scores, expected_rectangles, expected_text, message",
    [
        pytest.param(
            [0, 1],
            None,
            np.array([[10.0, 10.0, 20.0, 20.0]]),
            [np.array([3])],
            [
                call(((10, 10), (20, 20)), None, CLS0_COLOR, 2),
                call(((10, 18), (50, 26)), LABEL_BACK_COLOR, None, 1),
            ],
            [call((10, 11), "cls:0 3", LABEL_FORE_COLOR, ANY)],
            "test box - not outside - class string",
        ),
        pytest.param(
            [0, 1],
            ["car", "person"],
            np.array([[10.0, 10.0, 20.0, 20.0]]),
            [np.array([3])],
            [
                call(((10, 10), (20, 20)), None, CLS0_COLOR, 2),
                call(((10, 18), (50, 26)), LABEL_BACK_COLOR, None, 1),
            ],
            [call((10, 11), "car 3", LABEL_FORE_COLOR, ANY)],
            "test box - not outside - label string",
        ),
        pytest.param(
            [0, 1],
            None,
            np.array([[10.0, 20.0, 20.0, 20.0]]),
            [np.array([3])],
            [
                call(((10, 20), (20, 20)), None, CLS0_COLOR, 2),
                call(((10, 17), (50, 25)), LABEL_BACK_COLOR, None, 1),
            ],
            [call((10, 10), "cls:0 3", LABEL_FORE_COLOR, ANY)],
            "test box - outside - class string",
        ),
        pytest.param(
            [0, 1],
            ["car", "person"],
            np.array([[10.0, 20.0, 20.0, 20.0]]),
            [np.array([3])],
            [
                call(((10, 20), (20, 20)), None, CLS0_COLOR, 2),
                call(((10, 17), (50, 25)), LABEL_BACK_COLOR, None, 1),
            ],
            [call((10, 10), "car 3", LABEL_FORE_COLOR, ANY)],
            "test box - outside - label string",
        ),
        pytest.param(
            [0, 1],
            None,
            np.array([[10.0, 10.0, 20.0, 20.0], [10.0, 20.0, 20.0, 20.0]]),
            [np.array([3]), np.array([4])],
            [
                call(((10, 10), (20, 20)), None, CLS0_COLOR, 2),
                call(((10, 18), (50, 26)), LABEL_BACK_COLOR, None, 1),
                call(((10, 20), (20, 20)), None, ANY, 2),
                call(((10, 17), (50, 25)), LABEL_BACK_COLOR, None, 1),
            ],
            [
                call((10, 11), "cls:0 3", LABEL_FORE_COLOR, ANY),
                call((10, 10), "cls:1 4", LABEL_FORE_COLOR, ANY),
            ],
            "test boxes - inside and outside - class string",
        ),
        pytest.param(
            [0, 1],
            ["car", "person"],
            np.array([[10.0, 10.0, 20.0, 20.0], [10.0, 20.0, 20.0, 20.0]]),
            [np.array([3]), np.array([4])],
            [
                call(((10, 10), (20, 20)), None, CLS0_COLOR, 2),
                call(((10, 18), (50, 26)), LABEL_BACK_COLOR, None, 1),
                call(((10, 20), (20, 20)), None, ANY, 2),
                call(((10, 17), (50, 25)), LABEL_BACK_COLOR, None, 1),
            ],
            [
                call((10, 11), "car 3", LABEL_FORE_COLOR, ANY),
                call((10, 10), "person 4", LABEL_FORE_COLOR, ANY),
            ],
            "test boxes - inside and outside - label string",
        ),
    ],
)
def test_classification_box_cls_known(
    display_cv,
    rt2_img,
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
        rt2_img,
        output_width,
        output_height,
        boxes,
        scores,
        [[x] for x in class_ids],
        labels,
        monkeypatch,
    )

    with patch("PIL.ImageDraw.Draw", mock_image_draw(draw)):
        composite = rt2_img.Image.from_array(
            np.zeros((output_height, output_width, 3), dtype=np.uint8), 'RGB'
        )
        display_draw = display_cv.CVDraw(0, 1, composite, image, [])
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
    assert classifications[0].class_id == 1
    assert classifications[0].score == 0.9
    assert isinstance(classifications[1], ClassifiedObject)
    assert classifications[1].class_id == 2
    assert classifications[1].score == 0.8


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


def test_classified_object_properties():
    """Test ClassifiedObject property accessors with normal data."""
    meta = ClassificationMeta()
    meta.add_result([1], [0.9])

    obj = ClassifiedObject(meta, 0)
    assert obj.class_id == 1
    assert obj.score == 0.9
    assert obj.box is None
    assert [x.class_id for x in obj.topk] == [1]
    assert [x.score for x in obj.topk] == [0.9]
    assert obj.box is None


def test_classified_object_multiple_results():
    """Test ClassifiedObject with multiple classification results."""
    meta = ClassificationMeta()
    meta.add_result([1, 2], [0.9, 0.8])  # Top-2 results for first ROI
    meta.add_result([3], [0.7])  # Single result for second ROI

    # Test first object (top-2 results)
    obj1 = ClassifiedObject(meta, 0)
    assert obj1.class_id == 1
    assert obj1.score == 0.9
    assert obj1.box is None
    assert [o.class_id for o in obj1.topk] == [1, 2]
    assert [o.score for o in obj1.topk] == [0.9, 0.8]
    assert [o.box for o in obj1.topk] == [None, None]

    # Test second object (single result)
    obj2 = ClassifiedObject(meta, 1)
    assert obj2.class_id == 3
    assert obj2.score == 0.7
    assert obj2.box is None
    assert [o.class_id for o in obj2.topk] == [3]
    assert [o.score for o in obj2.topk] == [0.7]
    assert [o.box for o in obj2.topk] == [None]


def test_classified_object_edge_cases():
    """Test ClassifiedObject edge cases and bounds checking."""
    meta = ClassificationMeta()

    # Test with empty meta
    assert len(meta) == 0

    # Add some data
    meta.add_result([0], [0.5])

    obj = ClassifiedObject(meta, 0)
    # Test the helper method directly
    assert obj._get_meta_result_element(0) == 0  # class_ids
    assert obj._get_meta_result_element(1) == 0.5  # scores
    assert obj._get_meta_result_element(2) is None  # no box data
    assert obj._get_meta_result_element(10) is None  # way out of bounds


def test_classified_object_consistency_behavior():
    """Test that all properties behave consistently."""
    meta = ClassificationMeta()
    meta.add_result([42], [0.95])

    obj = ClassifiedObject(meta, 0)

    # All properties should access the same underlying data consistently
    result = meta.get_result(0)
    assert obj.class_id == result[0][0]  # Should be [42]
    assert obj.score == result[1][0]  # Should be [0.95]
    assert obj.box is None  # No box in standard result

    # Test that repeated access is consistent
    assert obj.class_id == obj.class_id
    assert obj.score == obj.score
    assert obj.box == obj.box


def test_classified_object_helper_method_safety():
    """Test the safety of the helper method _get_meta_result_element."""
    meta = ClassificationMeta()
    meta.add_result([1], [0.8])

    obj = ClassifiedObject(meta, 0)

    # Test accessing valid indices
    assert obj._get_meta_result_element(0) is not None  # class_ids exist
    assert obj._get_meta_result_element(1) is not None  # scores exist

    # Test accessing invalid indices safely returns None
    assert obj._get_meta_result_element(2) is None  # box doesn't exist
    assert obj._get_meta_result_element(5) is None  # way out of bounds

    # Note: negative indices are valid in Python and access from the end
    # -1 accesses the last element (scores in this case)
    assert obj._get_meta_result_element(-1) == 0.8  # Last element (scores)
    assert obj._get_meta_result_element(-2) == 1  # Second to last (class_ids)


def test_classified_object_with_single_values():
    """Test ClassifiedObject behavior with single int/float values."""
    meta = ClassificationMeta()
    # add_result converts single values to lists internally
    meta.add_result(5, 0.75)  # Single int and float

    obj = ClassifiedObject(meta, 0)
    assert obj.class_id == 5  # Should be converted to list
    assert obj.score == 0.75  # Should be converted to list
    assert obj.box is None


def test_classified_object_numpy_arrays():
    """Test ClassifiedObject behavior with numpy array inputs."""
    import numpy as np

    meta = ClassificationMeta()
    meta.add_result(np.array([7]), np.array([0.85]))

    obj = ClassifiedObject(meta, 0)
    assert obj.class_id == 7  # Should be converted from numpy
    assert obj.score == 0.85  # Should be converted from numpy
    assert obj.box is None


def test_classified_object_subindex_access():
    meta = ClassificationMeta()
    meta.add_result([10, 20, 30], [0.9, 0.8, 0.7])
    obj = ClassifiedObject(meta, 0, 0)
    assert obj.class_id == 10
    assert obj.score == 0.9
    obj = ClassifiedObject(meta, 0, 1)
    assert obj.class_id == 20
    assert obj.score == 0.8
    obj = ClassifiedObject(meta, 0, 2)
    assert obj.class_id == 30
    assert obj.score == 0.7


def test_classified_object_out_of_range_subindex():
    meta = ClassificationMeta()
    meta.add_result([10, 20], [0.9, 0.8])
    obj = ClassifiedObject(meta, 0, 0)
    assert obj.class_id == 10
    obj_invalid = ClassifiedObject(meta, 0, 5)
    with pytest.raises(IndexError):
        _ = obj_invalid.class_id

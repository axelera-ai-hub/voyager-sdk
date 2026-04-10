# Copyright Axelera AI, 2024
import pytest

from ax_models.decoders import yolo


@pytest.mark.parametrize(
    "test_name,shapes,num_classes,expected_model,expected_extra",
    [
        (
            "YOLOv5 (COCO-80)",
            [(1, 40, 40, 255), (1, 20, 20, 255), (1, 80, 80, 255)],
            80,
            yolo.YoloFamily.YOLOv5,
            {},
        ),
        (
            "YOLOv5 (20-classes)",
            [(1, 40, 40, 75), (1, 20, 20, 75), (1, 80, 80, 75)],
            20,
            yolo.YoloFamily.YOLOv5,
            {},
        ),
        (
            "YOLOX (COCO-80)",
            [
                (1, 80, 80, 80),
                (1, 40, 40, 1),
                (1, 40, 40, 80),
                (1, 20, 20, 1),
                (1, 20, 20, 80),
                (1, 80, 80, 1),
                (1, 80, 80, 4),
                (1, 40, 40, 4),
                (1, 20, 20, 4),
            ],
            80,
            yolo.YoloFamily.YOLOX,
            {},
        ),
        (
            "YOLOX (20-classes)",
            [
                (1, 80, 80, 20),
                (1, 40, 40, 1),
                (1, 40, 40, 20),
                (1, 20, 20, 1),
                (1, 20, 20, 20),
                (1, 80, 80, 1),
                (1, 80, 80, 4),
                (1, 40, 40, 4),
                (1, 20, 20, 4),
            ],
            20,
            yolo.YoloFamily.YOLOX,
            {},
        ),
        (
            "YOLOv8 (COCO-80)",
            [
                (1, 80, 80, 64),
                (1, 40, 40, 64),
                (1, 20, 20, 64),
                (1, 80, 80, 80),
                (1, 40, 40, 80),
                (1, 20, 20, 80),
            ],
            80,
            yolo.YoloFamily.YOLOv8,
            {},
        ),
        (
            "YOLOv8 (20-classes)",
            [
                (1, 80, 80, 64),
                (1, 40, 40, 64),
                (1, 20, 20, 64),
                (1, 80, 80, 20),
                (1, 40, 40, 20),
                (1, 20, 20, 20),
            ],
            20,
            yolo.YoloFamily.YOLOv8,
            {},
        ),
        (
            "YOLO-NAS (COCO-80, dfl_bins=17)",
            [
                (1, 80, 80, 80),
                (1, 80, 80, 68),
                (1, 40, 40, 80),
                (1, 40, 40, 68),
                (1, 20, 20, 80),
                (1, 20, 20, 68),
            ],
            80,
            yolo.YoloFamily.YOLO_NAS,
            {'dfl_bins': 17},
        ),
        (
            "YOLO-NAS (10-classes, dfl_bins=17)",
            [
                (1, 80, 80, 10),
                (1, 80, 80, 68),
                (1, 40, 40, 10),
                (1, 40, 40, 68),
                (1, 20, 20, 10),
                (1, 20, 20, 68),
            ],
            10,
            yolo.YoloFamily.YOLO_NAS,
            {'dfl_bins': 17},
        ),
        (
            "YOLO-NAS (5-classes, dfl_bins=17)",
            [
                (1, 80, 80, 5),
                (1, 80, 80, 68),
                (1, 40, 40, 5),
                (1, 40, 40, 68),
                (1, 20, 20, 5),
                (1, 20, 20, 68),
            ],
            5,
            yolo.YoloFamily.YOLO_NAS,
            {'dfl_bins': 17},
        ),
        (
            "YOLO-NAS (50-classes, dfl_bins=17)",
            [
                (1, 80, 80, 50),
                (1, 80, 80, 68),
                (1, 40, 40, 50),
                (1, 40, 40, 68),
                (1, 20, 20, 50),
                (1, 20, 20, 68),
            ],
            50,
            yolo.YoloFamily.YOLO_NAS,
            {'dfl_bins': 17},
        ),
        (
            "Unknown model",
            [(1, 40, 40, 100), (1, 20, 20, 100)],
            80,
            yolo.YoloFamily.Unknown,
            {},
        ),
    ],
)
def test_guess_yolo_model(test_name, shapes, num_classes, expected_model, expected_extra):
    """Test the YOLO model type detection logic."""
    model_type, explanation, extra = yolo._guess_yolo_model(shapes, num_classes)

    assert model_type == expected_model, (
        f"Failed for {test_name}: "
        f"Expected {expected_model.name}, but got {model_type.name}. "
        f"Explanation: {explanation}"
    )
    assert isinstance(explanation, str)
    assert len(explanation) > 0
    assert extra == expected_extra, (
        f"Failed for {test_name}: " f"Expected extra={expected_extra}, but got extra={extra}"
    )


@pytest.mark.parametrize(
    'input, expected',
    [
        (None, []),
        ('', []),
        ('person', ['person']),
        ('person,car', ['person', 'car']),
        ('person, car', ['person', 'car']),
        (' person ; car ; bus ', ['person', 'car', 'bus']),
        ([], []),
        (['person'], ['person']),
        (['person', 'car'], ['person', 'car']),
        (['person', ' car'], ['person', 'car']),
        ([' person ', ' car ', ' bus '], ['person', 'car', 'bus']),
        ("$$Variable", "$$Variable"),
    ],
)
def test_label_filter_formats(input, expected):
    """Test the label filter parsing logic."""
    decoder = yolo.DecodeYolo(
        box_format="xywh",
        normalized_coord=True,
        label_filter=input,
        label_exclude=input,
    )
    assert decoder.label_filter == expected
    assert decoder.label_exclude == expected

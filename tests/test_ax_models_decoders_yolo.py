# Copyright Axelera AI, 2024
import pytest

from ax_models.decoders import yolo


@pytest.mark.parametrize(
    "test_name,shapes,num_classes,expected_model",
    [
        (
            "YOLOv5 (COCO-80)",
            [(1, 40, 40, 255), (1, 20, 20, 255), (1, 80, 80, 255)],
            80,
            yolo.YoloFamily.YOLOv5,
        ),
        (
            "YOLOv5 (20-classes)",
            [(1, 40, 40, 75), (1, 20, 20, 75), (1, 80, 80, 75)],
            20,
            yolo.YoloFamily.YOLOv5,
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
        ),
        (
            "Unknown model",
            [(1, 40, 40, 100), (1, 20, 20, 100)],
            80,
            yolo.YoloFamily.Unknown,
        ),
    ],
)
def test_guess_yolo_model(test_name, shapes, num_classes, expected_model):
    """Test the YOLO model type detection logic."""
    model_type, explanation = yolo._guess_yolo_model(shapes, num_classes)

    assert model_type == expected_model, (
        f"Failed for {test_name}: "
        f"Expected {expected_model.name}, but got {model_type.name}. "
        f"Explanation: {explanation}"
    )
    assert isinstance(explanation, str)
    assert len(explanation) > 0

# Copyright Axelera AI, 2024

import os
from unittest.mock import patch

from axelera.app.pipe import manager


def test_create_manager_from_model_name():
    with patch.dict(os.environ, {'AXELERA_FRAMEWORK': "."}):
        network_name = "mc-yolov5s-v7-coco"
        expected_path = "ax_models/model_cards/yolo/object_detection/yolov5s-v7-coco.yaml"
        output = manager._get_real_path_if_path_is_model_name(network_name)
        assert output == expected_path

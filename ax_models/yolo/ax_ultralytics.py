# Axelera class for Ultralytics YOLO models
# Copyright Axelera AI, 2025
from __future__ import annotations

from pathlib import Path
import typing

from ultralytics import YOLO

from ax_models import base_torch
from axelera import types
from axelera.app import logging_utils, utils
from axelera.app.torch_utils import torch

LOG = logging_utils.getLogger(__name__)


def get_simplified_model(weights):
    yolo = YOLO(weights)
    model_type = yolo.task
    torch_model = yolo.model
    original_forward = torch_model.forward

    if model_type == 'segment':

        def new_forward(self, x):
            outputs = original_forward(x)
            if isinstance(outputs, (list, tuple)) and len(outputs) > 1:
                if isinstance(outputs[1], (list, tuple)) and len(outputs[1]) > 2:
                    # # Return exactly [detection_output, prototype_masks]
                    return [outputs[0], outputs[1][2]]
            return outputs

    else:

        def new_forward(self, x):
            outputs = original_forward(x)
            # Return only the first output (processed detections)
            return outputs[0] if isinstance(outputs, (list, tuple)) else outputs

    # Replace the forward method
    import types

    torch_model.forward = types.MethodType(new_forward, torch_model)
    return torch_model, model_type


class AxUltralyticsYOLO(base_torch.TorchModel):
    def __init__(self):
        super().__init__()

    def init_model_deploy(self, model_info: types.ModelInfo):
        weights = Path(model_info.weight_path)
        if not (weights.exists() and utils.md5_validates(weights, model_info.weight_md5)):
            utils.download(model_info.weight_url, weights, model_info.weight_md5)

        self.torch_model, self.model_type = get_simplified_model(weights)
        self.to("cpu")
        self.eval()

    def to_device(self, device: typing.Optional[torch.device] = None):
        self.to(device)

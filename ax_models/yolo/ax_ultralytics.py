# Axelera class for Ultralytics YOLO models
# Copyright Axelera AI, 2025
from __future__ import annotations

from pathlib import Path
import typing

from ultralytics import YOLO

from ax_models import base_torch
from axelera import types
from axelera.app import logging_utils, utils
from axelera.app.torch_utils import safe_torch_load, torch

LOG = logging_utils.getLogger(__name__)


def get_simplified_model(weights):
    # Use safe_torch_load if weights is a file path string or Path
    if isinstance(weights, (str, Path)) and str(weights).endswith(('.pt', '.pth')):
        try:
            # YOLO() constructor internally calls torch.load, so we can't directly intercept it
            # If it fails, we'll need a different approach
            yolo = YOLO(weights)
        except Exception as e:
            if "weights_only" in str(e) or "was not an allowed global" in str(e):
                LOG.warning("Standard YOLO loading failed, trying with compatibility mode")
                # Custom loading for PyTorch 2.6+
                torch_version = tuple(map(int, torch.__version__.split('.')[:2]))
                if torch_version >= (2, 6):
                    # This is a simplified approach - actual implementation may need more complexity
                    import os

                    from ultralytics.utils.downloads import attempt_download_asset

                    weights = (
                        attempt_download_asset(weights) if not os.path.isfile(weights) else weights
                    )
                    yolo = YOLO(None)  # initialize with no model
                    yolo.model = safe_torch_load(weights)  # load model with our safe loader
                    yolo.task = 'detect' if hasattr(yolo.model, 'task') else yolo.model.task
            else:
                raise
    else:
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


def flatten_ultralytics_outputs(outputs):
    # If dict, return as-is (already handled by types module)
    if isinstance(outputs, dict):
        return outputs

    flattened = []

    def visit(obj):
        if isinstance(obj, (list, tuple)):
            for item in obj:
                visit(item)
        elif isinstance(obj, dict):
            pass  # skip these because ultralytics return dicts of information we do not need.
        else:  # Do not walk into other types
            flattened.append(obj)

    visit(outputs)
    return flattened


class AxUltralyticsYOLO(base_torch.TorchModel):
    def __init__(self):
        super().__init__()

    def init_model_deploy(self, model_info: types.ModelInfo, dataset_config: dict, **kwargs):
        weights = Path(model_info.weight_path)
        if not (weights.exists() and utils.md5_validates(weights, model_info.weight_md5)):
            utils.download(model_info.weight_url, weights, model_info.weight_md5)

        self.torch_model, self.model_type = get_simplified_model(weights)
        self.to("cpu")
        self.eval()

    def forward(self, x):
        return flatten_ultralytics_outputs(super().forward(x))

    def to_device(self, device: typing.Optional[torch.device] = None):
        self.to(device)

# Copyright Axelera AI, 2025
# The simplest types.Model from ONNX

import PIL
import numpy as np
import torch
import torchvision.transforms as transforms

from ax_models import base_onnx
from axelera import types
from axelera.app import logging_utils

LOG = logging_utils.getLogger(__name__)


# Example of the simplest ONNX model; you should implement your override_preprocess in this class
class CustomONNXModel(base_onnx.AxONNXModel):
    def override_preprocess(self, img: PIL.Image.Image | np.ndarray) -> torch.Tensor:
        from ax_models.tutorials.resnet34_fruit360 import TRANSFORM

        # this equals to transforms.Compose([
        #     transforms.Resize((100, 100)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # ])(img)
        return TRANSFORM['val'](img)

# Copyright Axelera AI, 2025
# The simplest types.Model from Pytorch

import PIL
import numpy as np
import torch
import torchvision.transforms as transforms

from ax_models import base_torch
from axelera import types
from axelera.app import logging_utils

LOG = logging_utils.getLogger(__name__)


# Example of the simplest Pytorch model
class CustomPytorchModel(base_torch.TorchModel):

    # the key is to initialize your model and load the weights to assign to self.torch_model
    def init_model_deploy(self, model_info: types.ModelInfo):
        from ax_models.tutorials.resnet34_fruit360 import ResNet34Model

        # In this case, load_state_dict(torch.load(model_info.weight_path, weights_only=True))
        # will be performed in the ResNet34Model constructor. You should load the weights right
        # after the model is initialized if your model doesn't load the weights in the constructor.
        self.torch_model = ResNet34Model(
            exists_weights=model_info.weight_path, fixed_feature_extractor=False
        ).eval()


# Inherit from CustomPytorchModel, or you can implement init_model_deploy again
class CustomAxPytorchModelWithPreprocess(CustomPytorchModel):

    # copy the transforms from your preprocessing or dataset implementation and paste them here
    def override_preprocess(self, img: PIL.Image.Image | np.ndarray) -> torch.Tensor:
        from ax_models.tutorials.resnet34_fruit360 import TRANSFORM

        # this equals to transforms.Compose([
        #     transforms.Resize((100, 100)),
        #     transforms.ToTensor(),
        #     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        # ])(img)
        return TRANSFORM['val'](img)

# Axelera base class for PyTorch Image Models (Timm)
# Copyright Axelera AI, 2024

from pathlib import Path

import PIL
import timm
import torch

from ax_models import base_torch
from axelera import types
from axelera.app import logging_utils, utils
import axelera.app.yaml as YAML

LOG = logging_utils.getLogger(__name__)


def _disable_ceil_mode_in_avgpool(model):
    for module in model.modules():
        if isinstance(module, torch.nn.AvgPool2d):
            module.ceil_mode = False
    return model


def _convert_first_node_to_1_channel(model):
    """Convert the first conv layer to grayscale, by averaging the 3 channels."""
    import torch.nn as nn

    original_conv = model.conv1[0].weight.data
    new_conv = nn.Conv2d(
        1,
        out_channels=original_conv.size(0),
        kernel_size=model.conv1[0].kernel_size,
        stride=model.conv1[0].stride,
        padding=model.conv1[0].padding,
        bias=False if model.conv1[0].bias is None else True,
    )
    new_conv.weight.data = original_conv.mean(dim=1, keepdim=True)
    model.conv1[0] = new_conv
    return model


class AxTimmModel(base_torch.TorchModel):
    """Model methods for Timm models"""

    def init_model_deploy(self, model_info: types.ModelInfo, timm_model_args=''):
        model_name = YAML.attribute(timm_model_args, 'name')
        if model_info.weight_path:
            if not Path(model_info.weight_path).exists():
                if model_info.weight_url:
                    utils.download(
                        model_info.weight_url, Path(model_info.weight_path), model_info.weight_md5
                    )
                else:
                    raise FileNotFoundError(f"weight_path: {model_info.weight_path} not found")
            self.torch_model = timm.create_model(model_name, pretrained=False)
            weights = torch.load(model_info.weight_path, map_location=torch.device('cpu'))
            self.torch_model.load_state_dict(weights)
        else:  # use pretrained weights from timm
            self.torch_model = timm.create_model(model_name, pretrained=True)

        if model_info.extra_kwargs.get('disable_ceil_mode_in_avgpool', False):
            self.torch_model = _disable_ceil_mode_in_avgpool(self.torch_model)
        if model_info.extra_kwargs.get('convert_first_node_to_1_channel', False):
            self.torch_model = _convert_first_node_to_1_channel(self.torch_model)
        self.torch_model.eval()

        # # verify the mode can work
        # from torchvision import transforms
        # rgb_weights = torch.tensor([0.2989, 0.5870, 0.1140])
        # gray_mean = (0.485 * 0.2989 + 0.456 * 0.5870 + 0.406 * 0.1140)
        # gray_std = (0.229 * 0.2989 + 0.224 * 0.5870 + 0.225 * 0.1140)
        # transform = transforms.Compose([
        #     transforms.Resize(256),
        #     transforms.CenterCrop(224),
        #     transforms.Grayscale(num_output_channels=1),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[gray_mean], std=[gray_std])
        # ])
        # import PIL
        # img = PIL.Image.open('/home/ubuntu/software-platform/host/application/framework/data/ImageNet/val/n01818515/ILSVRC2012_val_00009091.JPEG')

        # input_tensor = transform(img)
        # input_batch = input_tensor.unsqueeze(0)
        # output = self.torch_model(input_batch)
        # from pdb import set_trace
        # set_trace()

    # TODO: use this as an example of how to deploy with override preprocess
    # def override_preprocess(self, img: PIL.Image.Image) -> torch.Tensor:
    #     from timm.data import resolve_data_config
    #     from timm.data.transforms_factory import create_transform

    #     config = resolve_data_config({}, model=self.torch_model, use_test_size=True)
    #     transform = create_transform(**config)
    #     return transform(img)

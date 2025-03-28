# Axelera PyTorch Torchvision SqueezeNet
# Copyright Axelera AI, 2023

from classifiers import AxTorchvisionClassifierModel
from torchvision.models import ResNet

from axelera.app import utils
import axelera.app.yaml as YAML


class AxTorchvisionResNet(ResNet, AxTorchvisionClassifierModel):
    def __init__(self, **kwargs):
        torchvision_args = YAML.attribute(kwargs, 'torchvision_args')
        keys_of_interest = ['groups', 'width_per_group']
        model_kwargs = {
            key: value
            for key, value in torchvision_args.items()
            if key in keys_of_interest and value != "None"
        }

        super().__init__(
            block=utils.import_from_module(
                "torchvision.models.resnet", YAML.attribute(torchvision_args, 'block')
            ),
            layers=YAML.attribute(torchvision_args, 'layers'),
            num_classes=YAML.attribute(kwargs, 'num_classes'),
            **model_kwargs,
        )

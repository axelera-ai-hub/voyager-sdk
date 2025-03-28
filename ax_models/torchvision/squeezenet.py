# Axelera PyTorch Torchvision SqueezeNet
# Copyright Axelera AI, 2023

from classifiers import AxTorchvisionClassifierModel
from torchvision.models import SqueezeNet

import axelera.app.yaml as YAML


class AxTorchvisionSqueezeNet1_0(SqueezeNet, AxTorchvisionClassifierModel):
    def __init__(self, **kwargs):
        super().__init__(
            version='1_0',
            num_classes=YAML.attribute(kwargs, 'num_classes'),
        )


class AxTorchvisionSqueezeNet1_1(SqueezeNet, AxTorchvisionClassifierModel):
    def __init__(self, **kwargs):
        super().__init__(
            version='1_1',
            num_classes=YAML.attribute(kwargs, 'num_classes'),
        )

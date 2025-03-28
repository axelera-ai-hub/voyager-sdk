from functools import partial

from classifiers import AxTorchvisionClassifierModel
from torch import nn
from torchvision import models
from torchvision.models import regnet
from torchvision.models._utils import _ovewrite_named_param

import axelera.app.yaml as YAML


class AxTorchvisionRegNet_Y_400MF(models.RegNet, AxTorchvisionClassifierModel):
    def __init__(self, **kwargs):
        model_kwargs = {}
        torchvision_args = YAML.attribute(kwargs, 'torchvision_args')
        weight_args = YAML.attribute(torchvision_args, 'torchvision_weights_args')
        weights = getattr(getattr(models, weight_args['object']), weight_args['name'])

        block_params = regnet.BlockParams.from_init_params(
            depth=16, w_0=48, w_a=27.89, w_m=2.09, group_width=8, se_ratio=0.25, **model_kwargs
        )
        _ovewrite_named_param(model_kwargs, "num_classes", len(weights.meta["categories"]))
        norm_layer = partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1)
        super().__init__(block_params, norm_layer=norm_layer, **model_kwargs)


class AxTorchvisionRegNet_X_400MF(models.RegNet, AxTorchvisionClassifierModel):
    def __init__(self, **kwargs):
        model_kwargs = {}
        torchvision_args = YAML.attribute(kwargs, 'torchvision_args')
        weight_args = YAML.attribute(torchvision_args, 'torchvision_weights_args')
        weights = getattr(getattr(models, weight_args['object']), weight_args['name'])

        block_params = regnet.BlockParams.from_init_params(
            depth=22, w_0=24, w_a=24.48, w_m=2.54, group_width=16, **model_kwargs
        )
        _ovewrite_named_param(model_kwargs, "num_classes", len(weights.meta["categories"]))
        norm_layer = partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1)
        super().__init__(block_params, norm_layer=norm_layer, **model_kwargs)


class AxTorchvisionRegNet_X_1_6GF(models.RegNet, AxTorchvisionClassifierModel):
    def __init__(self, **kwargs):
        model_kwargs = {}
        torchvision_args = YAML.attribute(kwargs, 'torchvision_args')
        weight_args = YAML.attribute(torchvision_args, 'torchvision_weights_args')
        weights = getattr(getattr(models, weight_args['object']), weight_args['name'])

        block_params = regnet.BlockParams.from_init_params(
            depth=18, w_0=80, w_a=34.01, w_m=2.25, group_width=24, **model_kwargs
        )
        _ovewrite_named_param(model_kwargs, "num_classes", len(weights.meta["categories"]))
        norm_layer = partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1)
        super().__init__(block_params, norm_layer=norm_layer, **model_kwargs)


class AxTorchvisionRegNet_Y_1_6GF(models.RegNet, AxTorchvisionClassifierModel):
    def __init__(self, **kwargs):
        model_kwargs = {}
        torchvision_args = YAML.attribute(kwargs, 'torchvision_args')
        weight_args = YAML.attribute(torchvision_args, 'torchvision_weights_args')
        weights = getattr(getattr(models, weight_args['object']), weight_args['name'])

        block_params = regnet.BlockParams.from_init_params(
            depth=27, w_0=48, w_a=20.71, w_m=2.65, group_width=24, se_ratio=0.25, **model_kwargs
        )
        _ovewrite_named_param(model_kwargs, "num_classes", len(weights.meta["categories"]))
        norm_layer = partial(nn.BatchNorm2d, eps=1e-05, momentum=0.1)
        super().__init__(block_params, norm_layer=norm_layer, **model_kwargs)

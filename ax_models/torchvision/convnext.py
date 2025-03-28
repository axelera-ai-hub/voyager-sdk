from classifiers import AxTorchvisionClassifierModel
from torchvision import models
from torchvision.models import convnext
from torchvision.models._utils import _ovewrite_named_param

import axelera.app.yaml as YAML


class AxTorchvisionConvNextTiny(models.ConvNeXt, AxTorchvisionClassifierModel):
    def __init__(self, **kwargs):
        model_kwargs = {}
        torchvision_args = YAML.attribute(kwargs, 'torchvision_args')
        weight_args = YAML.attribute(torchvision_args, 'torchvision_weights_args')
        weights = getattr(getattr(models, weight_args['object']), weight_args['name'])

        block_setting = [
            convnext.CNBlockConfig(96, 192, 3),
            convnext.CNBlockConfig(192, 384, 3),
            convnext.CNBlockConfig(384, 768, 9),
            convnext.CNBlockConfig(768, None, 3),
        ]
        stochastic_depth_prob = 0.1
        _ovewrite_named_param(model_kwargs, "num_classes", len(weights.meta["categories"]))

        super().__init__(
            block_setting, stochastic_depth_prob=stochastic_depth_prob, **model_kwargs
        )


class AxTorchvisionConvNextSmall(models.ConvNeXt, AxTorchvisionClassifierModel):
    def __init__(self, **kwargs):
        model_kwargs = {}
        torchvision_args = YAML.attribute(kwargs, 'torchvision_args')
        weight_args = YAML.attribute(torchvision_args, 'torchvision_weights_args')
        weights = getattr(getattr(models, weight_args['object']), weight_args['name'])

        block_setting = [
            convnext.CNBlockConfig(96, 192, 3),
            convnext.CNBlockConfig(192, 384, 3),
            convnext.CNBlockConfig(384, 768, 27),
            convnext.CNBlockConfig(768, None, 3),
        ]

        stochastic_depth_prob = 0.4
        _ovewrite_named_param(model_kwargs, "num_classes", len(weights.meta["categories"]))

        super().__init__(
            block_setting, stochastic_depth_prob=stochastic_depth_prob, **model_kwargs
        )


class AxTorchvisionConvNextBase(models.ConvNeXt, AxTorchvisionClassifierModel):
    def __init__(self, **kwargs):
        model_kwargs = {}
        torchvision_args = YAML.attribute(kwargs, 'torchvision_args')
        weight_args = YAML.attribute(torchvision_args, 'torchvision_weights_args')
        weights = getattr(getattr(models, weight_args['object']), weight_args['name'])

        block_setting = [
            convnext.CNBlockConfig(128, 256, 3),
            convnext.CNBlockConfig(256, 512, 3),
            convnext.CNBlockConfig(512, 1024, 27),
            convnext.CNBlockConfig(1024, None, 3),
        ]
        stochastic_depth_prob = 0.5
        _ovewrite_named_param(model_kwargs, "num_classes", len(weights.meta["categories"]))

        super().__init__(
            block_setting, stochastic_depth_prob=stochastic_depth_prob, **model_kwargs
        )


class AxTorchvisionConvNextLarge(models.ConvNeXt, AxTorchvisionClassifierModel):
    def __init__(self, **kwargs):
        model_kwargs = {}
        torchvision_args = YAML.attribute(kwargs, 'torchvision_args')
        weight_args = YAML.attribute(torchvision_args, 'torchvision_weights_args')
        weights = getattr(getattr(models, weight_args['object']), weight_args['name'])

        block_setting = [
            convnext.CNBlockConfig(192, 384, 3),
            convnext.CNBlockConfig(384, 768, 3),
            convnext.CNBlockConfig(768, 1536, 27),
            convnext.CNBlockConfig(1536, None, 3),
        ]
        stochastic_depth_prob = 0.5
        _ovewrite_named_param(model_kwargs, "num_classes", len(weights.meta["categories"]))

        super().__init__(
            block_setting, stochastic_depth_prob=stochastic_depth_prob, **model_kwargs
        )

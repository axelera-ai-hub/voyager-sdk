import collections
import re

from classifiers import AxTorchvisionClassifierModel
from torchvision import models
from torchvision.models import densenet
from torchvision.models._utils import _ovewrite_named_param

import axelera.app.yaml as YAML


class _BaseDenseNet(densenet.DenseNet):
    def _load_state_dict(self, weights: collections.OrderedDict) -> None:
        """
        Load state dict from weights object. It handles the DenseNet state dict
        from torchvision.models.densenet.

        Args:
            weights: state dict from weights object.
        """

        # '.'s are no longer allowed in module names, but previous _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
        )

        for key in list(weights.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                weights[new_key] = weights[key]
                del weights[key]
        self.load_state_dict(weights)


class AxTorchvisionDenseNet121(_BaseDenseNet, AxTorchvisionClassifierModel):
    def __init__(self, **kwargs):
        model_kwargs = {}
        torchvision_args = YAML.attribute(kwargs, 'torchvision_args')
        weight_args = YAML.attribute(torchvision_args, 'torchvision_weights_args')
        weights = getattr(getattr(models, weight_args['object']), weight_args['name'])
        _ovewrite_named_param(model_kwargs, "num_classes", len(weights.meta["categories"]))

        super().__init__(
            growth_rate=32, block_config=(6, 12, 24, 16), num_init_features=64, **model_kwargs
        )


class AxTorchvisionDenseNet161(_BaseDenseNet, AxTorchvisionClassifierModel):
    def __init__(self, **kwargs):
        model_kwargs = {}
        torchvision_args = YAML.attribute(kwargs, 'torchvision_args')
        weight_args = YAML.attribute(torchvision_args, 'torchvision_weights_args')
        weights = getattr(getattr(models, weight_args['object']), weight_args['name'])
        _ovewrite_named_param(model_kwargs, "num_classes", len(weights.meta["categories"]))

        super().__init__(
            growth_rate=48, block_config=(6, 12, 36, 24), num_init_features=96, **model_kwargs
        )

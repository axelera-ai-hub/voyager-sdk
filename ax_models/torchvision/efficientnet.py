from classifiers import AxTorchvisionClassifierModel
from torchvision import models
from torchvision.models import efficientnet
from torchvision.models._utils import _ovewrite_named_param

import axelera.app.yaml as YAML


class AxTorchvisionEfficientNet(models.EfficientNet, AxTorchvisionClassifierModel):
    def __init__(self, **kwargs):
        model_kwargs = {}
        torchvision_args = YAML.attribute(kwargs, 'torchvision_args')
        weight_args = YAML.attribute(torchvision_args, 'torchvision_weights_args')
        weights_class = getattr(models, weight_args['object'])
        weights = getattr(weights_class, weight_args['name'])

        # Determine the variant from the weights class name
        variant = weight_args['object'].lower().replace('_weights', '')

        # Define width_mult and depth_mult for each variant
        variant_config = {
            'efficientnet_b0': (1.0, 1.0, 0.2),
            'efficientnet_b1': (1.0, 1.1, 0.2),
            'efficientnet_b2': (1.1, 1.2, 0.3),
            'efficientnet_b3': (1.2, 1.4, 0.3),
            'efficientnet_b4': (1.4, 1.8, 0.4),
            'efficientnet_b5': (1.6, 2.2, 0.4),
            'efficientnet_b6': (1.8, 2.6, 0.5),
            'efficientnet_b7': (2.0, 3.1, 0.5),
            'efficientnet_v2_s': (None, None, 0.2),
            'efficientnet_v2_m': (None, None, 0.3),
            'efficientnet_v2_l': (None, None, 0.4),
        }

        width_mult, depth_mult, dropout = variant_config[variant]

        if width_mult and depth_mult:
            inverted_residual_setting, last_channel = efficientnet._efficientnet_conf(
                variant, width_mult=width_mult, depth_mult=depth_mult
            )
        else:
            inverted_residual_setting, last_channel = efficientnet._efficientnet_conf(variant)

        _ovewrite_named_param(model_kwargs, "num_classes", len(weights.meta["categories"]))

        super().__init__(
            inverted_residual_setting, dropout=dropout, last_channel=last_channel, **model_kwargs
        )

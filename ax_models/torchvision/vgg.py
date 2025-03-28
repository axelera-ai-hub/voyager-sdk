from classifiers import AxTorchvisionClassifierModel
from torchvision import models
from torchvision.models import vgg
from torchvision.models._utils import _ovewrite_named_param

import axelera.app.yaml as YAML


class AxTorchvisionVGG16(models.VGG, AxTorchvisionClassifierModel):
    def __init__(self, **kwargs):
        model_kwargs = {}
        torchvision_args = YAML.attribute(kwargs, 'torchvision_args')
        weight_args = YAML.attribute(torchvision_args, 'torchvision_weights_args')
        weights = getattr(getattr(models, weight_args['object']), weight_args['name'])
        _ovewrite_named_param(model_kwargs, "num_classes", len(weights.meta["categories"]))

        super().__init__(vgg.make_layers(vgg.cfgs['D'], batch_norm=False), **model_kwargs)

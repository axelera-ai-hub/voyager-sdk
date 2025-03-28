from classifiers import AxTorchvisionClassifierModel
from torchvision import models
from torchvision.models._utils import _ovewrite_named_param

import axelera.app.yaml as YAML


class AxTorchvisionInception3(models.Inception3, AxTorchvisionClassifierModel):
    def __init__(self, **kwargs):
        model_kwargs = {}
        torchvision_args = YAML.attribute(kwargs, 'torchvision_args')
        weight_args = YAML.attribute(torchvision_args, 'torchvision_weights_args')
        weights = getattr(getattr(models, weight_args['object']), weight_args['name'])
        _ovewrite_named_param(model_kwargs, "num_classes", len(weights.meta["categories"]))
        super().__init__(**model_kwargs)

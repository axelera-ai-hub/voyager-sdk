# Axelera base class for PyTorch Torchvision classifiers
# Copyright Axelera AI, 2023

import collections
from pathlib import Path

import torch

from axelera import types
from axelera.app import logging_utils, utils
import axelera.app.yaml as YAML

LOG = logging_utils.getLogger(__name__)


class AxTorchvisionClassifierModel(types.Model):
    """Model methods for Torchvision classifier models"""

    def init_model_deploy(self, model_info: types.ModelInfo, torchvision_args=''):
        if model_info.weight_path:
            if not Path(model_info.weight_path).exists():
                if model_info.weight_url:
                    utils.download(
                        model_info.weight_url, Path(model_info.weight_path), model_info.weight_md5
                    )
                else:
                    raise FileNotFoundError(f"weight_path: {model_info.weight_path} not found")
            weights = torch.load(model_info.weight_path, map_location=torch.device('cpu'))
        elif torchvision_args:  # should be a torchvision default model
            torchvision_weights_args = YAML.attribute(torchvision_args, 'torchvision_weights_args')
            weights_obj = utils.import_from_module(
                "torchvision.models", YAML.attribute(torchvision_weights_args, 'object')
            )
            weights_url = getattr(
                weights_obj, YAML.attribute(torchvision_weights_args, 'name')
            ).url
            weights = torch.hub.load_state_dict_from_url(weights_url)
        else:
            raise ValueError(
                f"Model {model_info.name} has no weights; please specify a "
                "weight_path or torchvision_weights_args in extra_kwargs"
            )

        self._load_state_dict(weights)

    def _load_state_dict(self, weights: collections.OrderedDict) -> None:
        """Load state dict from weights object.

        Args:
            weights: state dict from weights object.
        """
        self.load_state_dict(weights)

    def to_device(self, device):
        self.to(device)

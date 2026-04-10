# Axelera base class for PyTorch Image Models (Timm)
# Copyright Axelera AI, 2024

from pathlib import Path

import PIL
import timm
import torch

# local imports
from utils import convert_first_node_to_1_channel

from ax_models import base_torch
from axelera import types
from axelera.app import logging_utils, utils
from axelera.app.torch_utils import safe_torch_load
import axelera.app.yaml as YAML

LOG = logging_utils.getLogger(__name__)


def _disable_ceil_mode_in_avgpool(model):
    for module in model.modules():
        if isinstance(module, torch.nn.AvgPool2d):
            module.ceil_mode = False
    return model


class AxTimmModel(base_torch.TorchModel):
    """Model methods for Timm models"""

    def init_model_deploy(self, model_info: types.ModelInfo, dataset_config: dict, **kwargs):
        timm_model_args = YAML.attribute(kwargs, 'timm_model_args')
        model_name = YAML.attribute(timm_model_args, 'name')
        conversion_method = timm_model_args.get('grayscale_conversion_method', 'sum')

        # Extract input shape information
        input_tensor_shape = (
            model_info.input_tensor_shape if model_info.input_tensor_shape else None
        )
        input_channels = (
            input_tensor_shape[1] if input_tensor_shape and len(input_tensor_shape) >= 4 else 3
        )

        if model_info.weight_path:
            weights = Path(model_info.weight_path)
            utils.download_model_artifacts(
                weights,
                model_info.weight_url,
                model_info.weight_md5,
                model_name=model_info.name,
            )

            # Always load the model with its default configuration first
            self.torch_model = timm.create_model(model_name, pretrained=False)

            # Modify final layer to match the number of classes before loading weights
            num_classes = model_info.num_classes
            if num_classes:
                LOG.info(f"Adjusting output layer to {num_classes} classes")
                if hasattr(self.torch_model, 'fc'):
                    in_features = self.torch_model.fc.in_features
                    self.torch_model.fc = torch.nn.Linear(in_features, num_classes)
                elif hasattr(self.torch_model, 'classifier'):
                    in_features = self.torch_model.classifier.in_features
                    self.torch_model.classifier = torch.nn.Linear(in_features, num_classes)
                else:
                    LOG.warning("Could not find classification layer to modify")

            # If grayscale input is required, modify the model
            if input_channels == 1:
                self.torch_model = convert_first_node_to_1_channel(
                    self.torch_model, conversion_method
                )

            # Now load the weights after model structure has been modified
            weights = safe_torch_load(model_info.weight_path, map_location=torch.device('cpu'))
            self.torch_model.load_state_dict(weights)
        else:  # use pretrained weights from timm
            if input_channels != 3:
                try:
                    # Try to use timm's built-in support for different channel counts
                    # This may not work for all models, so we need to handle errors
                    self.torch_model = timm.create_model(
                        model_name, pretrained=True, in_chans=input_channels
                    )
                    LOG.info(
                        f"Created model with {input_channels} input channels using timm's native support"
                    )
                except Exception as e:
                    LOG.warning(f"Failed to create model with in_chans={input_channels}: {e}")
                    # Fall back to loading standard model and modifying
                    self.torch_model = timm.create_model(model_name, pretrained=True)
                    if input_channels == 1:
                        LOG.info(f"Converting model from 3-channel to 1-channel input...")
                        self.torch_model = convert_first_node_to_1_channel(
                            self.torch_model, conversion_method
                        )
            else:
                self.torch_model = timm.create_model(model_name, pretrained=True)

            # If custom num_classes is specified, adjust the output layer
            if model_info.num_classes and model_info.num_classes != 1000:
                if hasattr(self.torch_model, 'fc'):
                    in_features = self.torch_model.fc.in_features
                    self.torch_model.fc = torch.nn.Linear(in_features, model_info.num_classes)
                elif hasattr(self.torch_model, 'classifier'):
                    in_features = self.torch_model.classifier.in_features
                    self.torch_model.classifier = torch.nn.Linear(
                        in_features, model_info.num_classes
                    )

        if model_info.extra_kwargs.get('disable_ceil_mode_in_avgpool', False):
            self.torch_model = _disable_ceil_mode_in_avgpool(self.torch_model)
        self.torch_model.eval()


class AxTimmModelWithPreprocess(AxTimmModel):
    def override_preprocess(self, img: PIL.Image.Image) -> torch.Tensor:
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform

        config = resolve_data_config({}, model=self.torch_model, use_test_size=True)
        transform = create_transform(**config)
        return transform(img)

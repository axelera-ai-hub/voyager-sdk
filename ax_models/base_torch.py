# Copyright Axelera AI, 2024
# Helper class for pytorch models
from __future__ import annotations

import abc
import typing

import torch

from axelera import types


class TorchModel(types.Model, torch.nn.Module):
    """A wrapper class that adapts a PyTorch model to the Voyager SDK Model interface.

    This class provides a way to use existing PyTorch models within the Voyager SDK.
    There are two main approaches to using PyTorch models in the Voyager SDK:

    1. Directly inheriting from both Model and the PyTorch model (recommended):
       This approach is straightforward and exposes all model APIs directly.

       Example:
       class MyTorchModel(torch.nn.Module):
           def forward(self, x):
               return x * 2

       class AxMyTorchModel(Model, MyTorchModel):
           def init_model_deploy(self, model_info, dataset_config, **kwargs):
                # load model weights
                self.load_state_dict(torch.load(model_info.weight_path))

    2. Using TorchModel to simplify implementation:
       This approach can be useful when direct inheritance is not possible or when you want to
       keep the original model implementation separate.

       Example:
       class AxMyTorchModel(TorchModel):
           def init_model_deploy(self, model_info, dataset_config, **kwargs):
               self.torch_model = MyTorchModel()
               self.torch_model.load_state_dict(torch.load(model_info.weight_path))

    For custom models or significant modifications to existing models, you can use either approach:

    class AxMyTorchResnet(Model, torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.resnet = torchvision.models.resnet18()

        def forward(self, x):
            return self.resnet(x)

        def init_model_deploy(self, model_info, dataset_config, **kwargs):
            # Implementation specific to this model, e.g., loading weights
            self.resnet.load_state_dict(torch.load(model_info.weight_path))

    We recommend the first approach (direct inheritance) as it's simpler and exposes all model
    APIs directly. The TorchModel class is most useful when you need to wrap an existing PyTorch
    model without modifying its original implementation, and when direct inheritance is not
    feasible.
    """

    def __init__(self):
        super().__init__()

    @abc.abstractmethod
    def init_model_deploy(
        self, model_info: types.ModelInfo, dataset_config: dict, **kwargs
    ) -> None:
        ...

    @property
    def torch_model(self) -> typing.Optional[torch.nn.Module]:
        return getattr(self, "_internal_torch_model", None)

    def __setattr__(self, name, value):
        # If the attribute being set is 'torch_model', we need to handle it differently
        # because torch.nn.Module has its own special __setattr__ method
        if name == "torch_model":
            if isinstance(value, torch.nn.Module):
                self._internal_torch_model = value
                self.add_module("_internal_torch_model", self._internal_torch_model)
            else:
                raise ValueError("torch_model must be a torch.nn.Module")
        else:
            super().__setattr__(name, value)

    @torch.no_grad()
    def forward(self, *args, **kwargs):
        """
        Forward pass through the encapsulated model.
        """
        if self._internal_torch_model is None:
            raise ValueError("No torch model has been loaded or set.")
        return self._internal_torch_model(*args, **kwargs)

    def eval(self):
        super().eval()
        if self._internal_torch_model:
            self._internal_torch_model.eval()

    def to_device(self, device: torch.device) -> None:
        """Move the model to the specified device.

        Args:
            device: A torch.device object to move the model to.
        """
        super().to(device)
        if self._internal_torch_model:
            self._internal_torch_model = self._internal_torch_model.to(device)

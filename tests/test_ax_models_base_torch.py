# Copyright Axelera AI, 2024

from unittest.mock import patch

import pytest

torch = pytest.importorskip("torch")

from ax_models import base_torch
from axelera import types


@pytest.fixture
def mock_model_info():
    return types.ModelInfo(
        name="test_model",
        task_category=types.TaskCategory.Classification,
        input_tensor_shape=[1, 3, 224, 224],
        weight_path="/path/to/weights.onnx",
    )


@pytest.fixture
def mock_dataset_config():
    return {"dataset": "imagenet"}


def test_torch_model_instances_and_inheritance():
    # Test 1: Multiple instances
    class ConcreteTorchModel(base_torch.TorchModel):
        def init_model_deploy(self, model_info, dataset_config, **kwargs):
            pass

    # Test instance-level assignments
    model1, model2 = ConcreteTorchModel(), ConcreteTorchModel()
    mock_model1 = torch.nn.Linear(10, 1)
    mock_model2 = torch.nn.Linear(10, 1)

    model1.torch_model = mock_model1
    model2.torch_model = mock_model2

    assert model1.torch_model is not model2.torch_model
    assert model1.torch_model.in_features == model2.torch_model.in_features
    assert model1.torch_model.out_features == model2.torch_model.out_features

    # Test class-level assignments
    model3, model4 = ConcreteTorchModel(), ConcreteTorchModel()
    ConcreteTorchModel.torch_model = mock_model1

    assert model3.torch_model is model4.torch_model == mock_model1

    # Instance assignment should still work
    model4.torch_model = mock_model2
    assert model4.torch_model.in_features == mock_model2.in_features

    # Cleanup
    ConcreteTorchModel.torch_model = None

    # Test 2: Model inheritance and state dict loading
    class MyTorchModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(10, 1)

        def forward(self, x):
            return self.layer(x)

    # Test both inheritance approaches
    class CustomTorchModel(types.Model, MyTorchModel):
        def __init__(self):
            super().__init__()
            self.layer = torch.nn.Linear(10, 1)

        def init_model_deploy(self, model_info, dataset_config, **kwargs):
            state_dict = {"layer.weight": torch.ones(1, 10), "layer.bias": torch.zeros(1)}
            self.load_state_dict(state_dict)

    custom_model = CustomTorchModel()
    assert isinstance(custom_model, (types.Model, torch.nn.Module, MyTorchModel))

    class AxMyTorchModel(base_torch.TorchModel):
        def init_model_deploy(self, model_info, dataset_config, **kwargs):
            self.torch_model = MyTorchModel()
            state_dict = {"layer.weight": torch.ones(1, 10), "layer.bias": torch.zeros(1)}
            self.torch_model.load_state_dict(state_dict)

    ax_model = AxMyTorchModel()
    mock_info = types.ModelInfo(
        name="test_model",
        task_category=types.TaskCategory.Classification,
        input_tensor_shape=[1, 3, 224, 224],
        weight_path="/path/to/weights.pth",
    )
    ax_model.init_model_deploy(mock_info, {"dataset": "imagenet"})

    assert isinstance(ax_model, types.Model)
    assert isinstance(ax_model.torch_model, (torch.nn.Module, MyTorchModel))

    # Verify state dict loading
    expected_state = {"layer.weight": torch.ones(1, 10), "layer.bias": torch.zeros(1)}
    assert all(
        torch.equal(param, expected_param)
        for param, expected_param in zip(
            ax_model.torch_model.state_dict().values(), expected_state.values()
        )
    )


@patch("torch.load")
def test_torch_model_init_deploy(mock_torch_load, mock_model_info, mock_dataset_config):
    mock_torch_load.return_value = {
        "weight": torch.ones(1, 10),
        "bias": torch.zeros(1),
    }

    class ConcreteTorchModel(base_torch.TorchModel):
        def init_model_deploy(self, model_info, dataset_config, **kwargs):
            self.torch_model = torch.nn.Linear(10, 1)  # Example of a concrete model
            state_dict = torch.load(model_info.weight_path)
            self.torch_model.load_state_dict(state_dict)
            self.torch_model.eval()

    the_model = ConcreteTorchModel()
    the_model.init_model_deploy(mock_model_info, mock_dataset_config)
    mock_torch_load.assert_called_once_with(mock_model_info.weight_path)
    assert the_model.torch_model is not None
    assert isinstance(the_model.torch_model, torch.nn.Module)

    # Additional assertions to verify the state_dict was loaded correctly
    assert torch.equal(the_model.torch_model.weight, torch.ones(1, 10))
    assert torch.equal(the_model.torch_model.bias, torch.zeros(1))


def test_torch_model_forward():
    class ConcreteTorchModel(base_torch.TorchModel):
        def init_model_deploy(self, model_info, dataset_config, **kwargs):
            pass

    the_model = ConcreteTorchModel()
    mock_torch_model = torch.nn.Linear(10, 1)
    the_model.torch_model = mock_torch_model

    input_tensor = torch.randn(1, 10)
    output = the_model.forward(input_tensor)
    expected_output = mock_torch_model(input_tensor)
    assert torch.equal(output, expected_output)


def test_torch_model_forward_no_model():
    class ConcreteTorchModel(base_torch.TorchModel):
        def __init__(self):
            super().__init__()
            self._internal_torch_model = None

        def init_model_deploy(self, model_info, dataset_config, **kwargs):
            pass

    the_model = ConcreteTorchModel()
    with pytest.raises(ValueError, match="No torch model has been loaded or set."):
        the_model.forward(torch.randn(1, 3, 224, 224))


def test_torch_model_eval():
    class ConcreteTorchModel(base_torch.TorchModel):
        def init_model_deploy(self, model_info, dataset_config, **kwargs):
            pass

    the_model = ConcreteTorchModel()
    mock_torch_model = torch.nn.Linear(10, 1)
    the_model.torch_model = mock_torch_model

    the_model.eval()
    assert not mock_torch_model.training


def test_torch_model_to_device():
    class ConcreteTorchModel(base_torch.TorchModel):
        def init_model_deploy(self, model_info, dataset_config, **kwargs):
            pass

    the_model = ConcreteTorchModel()
    mock_torch_model = torch.nn.Linear(10, 1)
    the_model.torch_model = mock_torch_model

    device = torch.device("cpu")
    the_model.to_device(device)
    assert the_model.torch_model.weight.device == device
    assert the_model.torch_model.bias.device == device

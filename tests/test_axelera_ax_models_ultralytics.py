import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("ultralytics")

from ax_models.yolo import ax_ultralytics


def test_flatten_single_tensor_output():
    """Test model returning a single tensor."""
    original = torch.randn(1, 3, 224, 224)
    output = ax_ultralytics.flatten_ultralytics_outputs(original)
    assert isinstance(output, list)
    assert len(output) == 1
    assert isinstance(output[0], torch.Tensor)


def test_flatten_list_of_tensors_output():
    """Test model returning a list of tensors."""
    original = [torch.randn(1, 10), torch.randn(1, 20)]
    output = ax_ultralytics.flatten_ultralytics_outputs(original)
    assert isinstance(output, list)
    assert len(output) == 2
    assert all(isinstance(t, torch.Tensor) for t in output)


def test_flatten_nested_tuple_output():
    """Test model returning nested tuples (like ultralytics 8.4.0+)."""
    original = (
        (torch.randn(1, 10), torch.randn(1, 20)),
        torch.randn(1, 30),
    )
    output = ax_ultralytics.flatten_ultralytics_outputs(original)
    assert isinstance(output, list)
    assert len(output) == 3
    assert all(isinstance(t, torch.Tensor) for t in output)


def test_flatten_deeply_nested_structure():
    """Test model with deeply nested list/tuple structures."""
    original = [
        [torch.randn(1, 5), torch.randn(1, 10)],
        (torch.randn(1, 15), [torch.randn(1, 20)]),
    ]
    output = ax_ultralytics.flatten_ultralytics_outputs(original)
    assert isinstance(output, list)
    assert len(output) == 4
    assert all(isinstance(t, torch.Tensor) for t in output)


def test_flatten_dict_output_preserved():
    """Test model returning dict (should be preserved as-is)."""
    original = {
        'logits': torch.randn(1, 10),
        'features': torch.randn(1, 256),
    }
    output = ax_ultralytics.flatten_ultralytics_outputs(original)
    assert isinstance(output, dict)
    assert 'logits' in output
    assert 'features' in output


def test_flatten_mixed_nested_with_non_tensors():
    """Test model with mixed nested structures containing non-tensor values."""
    original = (
        torch.randn(1, 10),
        None,
        torch.randn(1, 20),
        {'key': 'value'},
        [torch.randn(1, 30)],
    )
    output = ax_ultralytics.flatten_ultralytics_outputs(original)

    assert isinstance(output, list)
    assert len(output) == 4
    assert original[0] is output[0]
    assert original[1] is output[1]
    assert original[2] is output[2]
    # note we lose the dict structure here
    assert original[4][0] is output[3]

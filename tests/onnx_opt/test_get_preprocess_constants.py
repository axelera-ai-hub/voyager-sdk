# Copyright Axelera AI, 2026
from __future__ import annotations

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from ax_models.onnx_optimizations import (
    _extract_preprocess_constants,
    get_preprocess_constants,
    validate_preprocess_pattern,
)


def _make_model_with_sub_div(sub_values, div_values):
    """Build an ONNX model with Sub and Div preprocessing operations."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

    # Create Sub constant
    sub_const = numpy_helper.from_array(np.array(sub_values, dtype=np.float32), "sub_const")

    # Create Div constant
    div_const = numpy_helper.from_array(np.array(div_values, dtype=np.float32), "div_const")

    # Create nodes
    sub_node = helper.make_node("Sub", ["input", "sub_const"], ["after_sub"])
    div_node = helper.make_node("Div", ["after_sub", "div_const"], ["output"])

    graph = helper.make_graph(
        [sub_node, div_node],
        "preprocess_graph",
        [X],
        [Y],
        initializer=[sub_const, div_const],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


def _make_model_with_sub_mul(sub_values, mul_values):
    """Build an ONNX model with Sub and Mul preprocessing operations (Mul is inverse of Div)."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

    # Create Sub constant
    sub_const = numpy_helper.from_array(np.array(sub_values, dtype=np.float32), "sub_const")

    # Create Mul constant
    mul_const = numpy_helper.from_array(np.array(mul_values, dtype=np.float32), "mul_const")

    # Create nodes
    sub_node = helper.make_node("Sub", ["input", "sub_const"], ["after_sub"])
    mul_node = helper.make_node("Mul", ["after_sub", "mul_const"], ["output"])

    graph = helper.make_graph(
        [sub_node, mul_node],
        "preprocess_graph",
        [X],
        [Y],
        initializer=[sub_const, mul_const],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


def _make_model_with_constant_node(sub_values, div_values):
    """Build an ONNX model using Constant nodes instead of initializers."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

    # Create Constant nodes
    sub_tensor = numpy_helper.from_array(np.array(sub_values, dtype=np.float32), "sub_const")
    div_tensor = numpy_helper.from_array(np.array(div_values, dtype=np.float32), "div_const")

    const_sub = helper.make_node("Constant", [], ["sub_const"], value=sub_tensor)
    const_div = helper.make_node("Constant", [], ["div_const"], value=div_tensor)

    # Create nodes
    sub_node = helper.make_node("Sub", ["input", "sub_const"], ["after_sub"])
    div_node = helper.make_node("Div", ["after_sub", "div_const"], ["output"])

    graph = helper.make_graph(
        [const_sub, const_div, sub_node, div_node],
        "preprocess_graph",
        [X],
        [Y],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


def _make_model_only_sub(sub_values):
    """Build an ONNX model with only Sub operation."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

    sub_const = numpy_helper.from_array(np.array(sub_values, dtype=np.float32), "sub_const")
    sub_node = helper.make_node("Sub", ["input", "sub_const"], ["output"])

    graph = helper.make_graph(
        [sub_node],
        "preprocess_graph",
        [X],
        [Y],
        initializer=[sub_const],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


def _make_model_only_div(div_values):
    """Build an ONNX model with only Div operation."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

    div_const = numpy_helper.from_array(np.array(div_values, dtype=np.float32), "div_const")
    div_node = helper.make_node("Div", ["input", "div_const"], ["output"])

    graph = helper.make_graph(
        [div_node],
        "preprocess_graph",
        [X],
        [Y],
        initializer=[div_const],
    )

    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


class TestGetPreprocessConstants:
    """Test suite for get_preprocess_constants function."""

    def test_extracts_sub_and_div_constants(self, tmp_path):
        """Test extraction of Sub and Div constants from a standard preprocessing graph."""
        sub_values = [123.675, 116.28, 103.53]
        div_values = [58.395, 57.12, 57.375]

        model = _make_model_with_sub_div(sub_values, div_values)
        model_path = tmp_path / "preprocess.onnx"
        onnx.save(model, str(model_path))

        sub_result, div_result = get_preprocess_constants(str(model_path))

        np.testing.assert_array_almost_equal(sub_result, sub_values, decimal=5)
        np.testing.assert_array_almost_equal(div_result, div_values, decimal=5)

    def test_constant_nodes_trigger_warning(self, tmp_path):
        """Test that Constant nodes in the graph are handled correctly.

        Note: The current implementation collects constants from both initializers and Constant nodes.
        """
        sub_values = [0.485, 0.456, 0.406]
        div_values = [0.229, 0.224, 0.225]

        model = _make_model_with_constant_node(sub_values, div_values)
        model_path = tmp_path / "preprocess.onnx"
        onnx.save(model, str(model_path))

        sub_result, div_result = get_preprocess_constants(str(model_path))

        # Constant nodes are properly handled and actual values are returned
        np.testing.assert_array_almost_equal(sub_result, sub_values, decimal=5)
        np.testing.assert_array_almost_equal(div_result, div_values, decimal=5)

    def test_handles_mul_as_inverse_div(self, tmp_path):
        """Test that Mul operation is correctly converted to Div (1/mul_const).

        Note: When all values in an array are identical, the implementation reduces
        it to a single-element array.
        """
        sub_values = [0.0, 0.0, 0.0]
        mul_values = [0.5, 0.5, 0.5]  # Mul by 0.5 is equivalent to Div by 2.0
        expected_sub = [0.0]  # Reduced to single element since all values are same
        expected_div = [2.0]  # Reduced to single element since all values are same

        model = _make_model_with_sub_mul(sub_values, mul_values)
        model_path = tmp_path / "preprocess.onnx"
        onnx.save(model, str(model_path))

        sub_result, div_result = get_preprocess_constants(str(model_path))

        np.testing.assert_array_almost_equal(sub_result, expected_sub, decimal=5)
        np.testing.assert_array_almost_equal(div_result, expected_div, decimal=5)

    def test_only_sub_operation(self, tmp_path):
        """Test model with only Sub operation (no Div).

        Note: When all values are identical, the implementation reduces to a single element.
        When no Div operation exists, default value [1.0] is returned.
        """
        sub_values = [128.0, 128.0, 128.0]
        expected_sub = [128.0]  # Reduced to single element since all values are same

        model = _make_model_only_sub(sub_values)
        model_path = tmp_path / "preprocess.onnx"
        onnx.save(model, str(model_path))

        sub_result, div_result = get_preprocess_constants(str(model_path))

        np.testing.assert_array_almost_equal(sub_result, expected_sub, decimal=5)
        # No div operation, so default value is returned
        assert div_result == [1.0]

    def test_only_div_operation(self, tmp_path):
        """Test model with only Div operation (no Sub).

        Note: When no Sub operation exists, default value [0] is returned.
        When all div values are identical, the implementation reduces to a single element.
        """
        div_values = [255.0, 255.0, 255.0]
        expected_div = [255.0]  # Reduced to single element since all values are same

        model = _make_model_only_div(div_values)
        model_path = tmp_path / "preprocess.onnx"
        onnx.save(model, str(model_path))

        sub_result, div_result = get_preprocess_constants(str(model_path))

        # No sub operation, so default value is returned
        assert sub_result == [0]
        np.testing.assert_array_almost_equal(div_result, expected_div, decimal=5)

    def test_single_channel_values(self, tmp_path):
        """Test extraction with single-channel (grayscale) preprocessing."""
        sub_values = [127.5]
        div_values = [127.5]

        model = _make_model_with_sub_div(sub_values, div_values)
        model_path = tmp_path / "preprocess.onnx"
        onnx.save(model, str(model_path))

        sub_result, div_result = get_preprocess_constants(str(model_path))

        # Results should be squeezed to scalar or 1-element list
        assert np.isclose(sub_result, sub_values[0], rtol=1e-5)
        assert np.isclose(div_result, div_values[0], rtol=1e-5)

    def test_imagenet_normalization(self, tmp_path):
        """Test with typical ImageNet normalization values."""
        # ImageNet mean and std (RGB)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

        # In preprocessing: (x - mean) / std
        # Which is implemented as: (x - mean*255) / (std*255) for uint8 inputs
        sub_values = [m * 255 for m in mean]
        div_values = [s * 255 for s in std]

        model = _make_model_with_sub_div(sub_values, div_values)
        model_path = tmp_path / "preprocess.onnx"
        onnx.save(model, str(model_path))

        sub_result, div_result = get_preprocess_constants(str(model_path))

        np.testing.assert_array_almost_equal(sub_result, sub_values, decimal=2)
        np.testing.assert_array_almost_equal(div_result, div_values, decimal=2)

    def test_scalar_broadcasting(self, tmp_path):
        """Test handling of scalar constants that broadcast to all channels."""
        # Scalar values that will broadcast
        sub_values = 128.0
        div_values = 255.0

        model = _make_model_with_sub_div([sub_values], [div_values])
        model_path = tmp_path / "preprocess.onnx"
        onnx.save(model, str(model_path))

        sub_result, div_result = get_preprocess_constants(str(model_path))

        # Should return scalar or single value
        if isinstance(sub_result, list):
            assert len(sub_result) == 1
            assert np.isclose(sub_result[0], sub_values, rtol=1e-5)
        else:
            assert np.isclose(sub_result, sub_values, rtol=1e-5)

        if isinstance(div_result, list):
            assert len(div_result) == 1
            assert np.isclose(div_result[0], div_values, rtol=1e-5)
        else:
            assert np.isclose(div_result, div_values, rtol=1e-5)

    def test_unexpected_node_type_raises_error(self, tmp_path):
        """Test that unexpected node types (unsupported preamble pattern) raise NotImplementedError."""
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
        Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

        # Add an unexpected node type (e.g., Conv) - not a normalization pattern
        weight = numpy_helper.from_array(np.random.randn(3, 3, 3, 3).astype(np.float32), "weight")
        conv_node = helper.make_node("Conv", ["input", "weight"], ["output"])

        graph = helper.make_graph(
            [conv_node],
            "preprocess_graph",
            [X],
            [Y],
            initializer=[weight],
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model_path = tmp_path / "preprocess.onnx"
        onnx.save(model, str(model_path))

        # Should raise NotImplementedError because Conv is not a supported normalization pattern
        with pytest.raises(NotImplementedError, match="UNSUPPORTED PREAMBLE PATTERN"):
            validate_preprocess_pattern(str(model_path))

    def test_validation_accepts_normalization_pattern(self, tmp_path):
        """Test that validate_preprocess_pattern accepts valid Sub+Div normalization."""
        sub_values = [123.675, 116.28, 103.53]
        div_values = [58.395, 57.12, 57.375]

        model = _make_model_with_sub_div(sub_values, div_values)
        model_path = tmp_path / "preprocess.onnx"
        onnx.save(model, str(model_path))

        # Should not raise
        result = validate_preprocess_pattern(str(model_path))
        assert result is True

    def test_validation_accepts_sub_only(self, tmp_path):
        """Test that validate_preprocess_pattern accepts Sub-only pattern."""
        sub_values = [128.0, 128.0, 128.0]

        model = _make_model_only_sub(sub_values)
        model_path = tmp_path / "preprocess.onnx"
        onnx.save(model, str(model_path))

        # Should not raise
        result = validate_preprocess_pattern(str(model_path))
        assert result is True

    def test_validation_accepts_div_only(self, tmp_path):
        """Test that validate_preprocess_pattern accepts Div-only pattern."""
        div_values = [255.0, 255.0, 255.0]

        model = _make_model_only_div(div_values)
        model_path = tmp_path / "preprocess.onnx"
        onnx.save(model, str(model_path))

        # Should not raise
        result = validate_preprocess_pattern(str(model_path))
        assert result is True

    def test_validation_rejects_empty_model(self, tmp_path):
        """Test that validate_preprocess_pattern rejects model with no operations."""
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
        Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

        # Empty graph with no nodes
        graph = helper.make_graph(
            [],
            "empty_graph",
            [X],
            [Y],
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model_path = tmp_path / "preprocess.onnx"
        onnx.save(model, str(model_path))

        with pytest.raises(NotImplementedError, match="UNSUPPORTED PREAMBLE PATTERN"):
            validate_preprocess_pattern(str(model_path))

    def test_validation_rejects_mixed_normalization_and_unsupported(self, tmp_path):
        """Test that validate_preprocess_pattern rejects a preamble mixing Sub/Div with unsupported ops."""
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
        Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

        sub_const = numpy_helper.from_array(
            np.array([123.675, 116.28, 103.53], dtype=np.float32), "sub_const"
        )
        reshape_shape = numpy_helper.from_array(
            np.array([1, 3, 224, 224], dtype=np.int64), "shape"
        )

        sub_node = helper.make_node("Sub", ["input", "sub_const"], ["after_sub"])
        reshape_node = helper.make_node("Reshape", ["after_sub", "shape"], ["output"])

        graph = helper.make_graph(
            [sub_node, reshape_node],
            "mixed_graph",
            [X],
            [Y],
            initializer=[sub_const, reshape_shape],
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        model_path = tmp_path / "preprocess.onnx"
        onnx.save(model, str(model_path))

        with pytest.raises(NotImplementedError, match="UNSUPPORTED PREAMBLE PATTERN"):
            validate_preprocess_pattern(str(model_path))


def _make_model_chained(ops):
    """Build an ONNX model with chained Sub/Div/Mul ops.

    ops is a list of (op_type, values) tuples, e.g.:
        [("Sub", [10.0]), ("Div", [2.0]), ("Sub", [5.0]), ("Div", [3.0])]
    """
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

    nodes = []
    initializers = []
    prev_output = "input"

    for i, (op_type, values) in enumerate(ops):
        const_name = f"const_{i}"
        out_name = "output" if i == len(ops) - 1 else f"intermediate_{i}"

        const_tensor = numpy_helper.from_array(np.array(values, dtype=np.float32), const_name)
        initializers.append(const_tensor)

        node = helper.make_node(op_type, [prev_output, const_name], [out_name])
        nodes.append(node)
        prev_output = out_name

    graph = helper.make_graph(nodes, "chained_graph", [X], [Y], initializer=initializers)
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_model_constant_first(op_type, const_values):
    """Build an ONNX model where the constant is the FIRST input (invalid orientation)."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

    const_tensor = numpy_helper.from_array(np.array(const_values, dtype=np.float32), "const_val")
    node = helper.make_node(op_type, ["const_val", "input"], ["output"])
    graph = helper.make_graph([node], "bad_graph", [X], [Y], initializer=[const_tensor])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_model_two_constants(op_type):
    """Build an ONNX model where both inputs are constants (no data flow)."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

    c1 = numpy_helper.from_array(np.array([1.0, 2.0, 3.0], dtype=np.float32), "const_a")
    c2 = numpy_helper.from_array(np.array([4.0, 5.0, 6.0], dtype=np.float32), "const_b")
    node = helper.make_node(op_type, ["const_a", "const_b"], ["output"])
    graph = helper.make_graph([node], "two_const_graph", [X], [Y], initializer=[c1, c2])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


def _make_model_two_data_inputs(op_type):
    """Build an ONNX model where both inputs come from the data path (no constants)."""
    X1 = helper.make_tensor_value_info("input1", TensorProto.FLOAT, [1, 3, 224, 224])
    X2 = helper.make_tensor_value_info("input2", TensorProto.FLOAT, [1, 3, 224, 224])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

    node = helper.make_node(op_type, ["input1", "input2"], ["output"])
    graph = helper.make_graph([node], "two_data_graph", [X1, X2], [Y])
    return helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])


class TestChainedNormalization:
    """Tests for sequential composition of normalization nodes."""

    def test_chained_sub_div_sub_div(self):
        """Sub(m1) -> Div(s1) -> Sub(m2) -> Div(s2) composes to (m1+m2*s1, s1*s2)."""
        m1, s1, m2, s2 = 10.0, 2.0, 5.0, 3.0
        model = _make_model_chained([("Sub", [m1]), ("Div", [s1]), ("Sub", [m2]), ("Div", [s2])])
        mean_result, std_result = _extract_preprocess_constants(model.graph)

        expected_mean = m1 + m2 * s1  # 10 + 5*2 = 20
        expected_std = s1 * s2  # 2*3 = 6
        assert np.isclose(mean_result[0], expected_mean)
        assert np.isclose(std_result[0], expected_std)

    def test_div_before_sub(self):
        """Div(s1) -> Sub(m2) composes to (m2*s1, s1)."""
        s1, m2 = 2.0, 5.0
        model = _make_model_chained([("Div", [s1]), ("Sub", [m2])])
        mean_result, std_result = _extract_preprocess_constants(model.graph)

        assert np.isclose(mean_result[0], m2 * s1)  # 10
        assert np.isclose(std_result[0], s1)  # 2

    def test_chained_mul_nodes(self):
        """Mul(c1) -> Mul(c2) composes to std=1/(c1*c2), mean=0."""
        c1, c2 = 2.0, 4.0
        model = _make_model_chained([("Mul", [c1]), ("Mul", [c2])])
        mean_result, std_result = _extract_preprocess_constants(model.graph)

        assert np.isclose(mean_result[0], 0.0)
        assert np.isclose(std_result[0], 1.0 / (c1 * c2))  # 0.125

    def test_per_channel_chained(self):
        """Per-channel chained normalizations compose element-wise."""
        m1 = [100.0, 110.0, 120.0]
        s1 = [50.0, 55.0, 60.0]
        m2 = [0.5, 0.5, 0.5]
        s2 = [2.0, 2.0, 2.0]
        model = _make_model_chained([("Sub", m1), ("Div", s1), ("Sub", m2), ("Div", s2)])
        mean_result, std_result = _extract_preprocess_constants(model.graph)

        expected_mean = [m1[i] + m2[i] * s1[i] for i in range(3)]
        expected_std = [s1[i] * s2[i] for i in range(3)]
        np.testing.assert_array_almost_equal(mean_result, expected_mean, decimal=4)
        np.testing.assert_array_almost_equal(std_result, expected_std, decimal=4)


class TestTopologyValidation:
    """Tests that invalid graph topologies raise NotImplementedError with clear messages."""

    def test_constant_as_first_input_of_sub_raises(self):
        """Sub node with constant as input[0] (constant - data) raises NotImplementedError."""
        model = _make_model_constant_first("Sub", [128.0])
        with pytest.raises(NotImplementedError, match="UNSUPPORTED PREAMBLE PATTERN"):
            _extract_preprocess_constants(model.graph)

    def test_constant_as_first_input_of_div_raises(self):
        """Div node with constant as input[0] (constant / data) raises NotImplementedError."""
        model = _make_model_constant_first("Div", [255.0])
        with pytest.raises(NotImplementedError, match="UNSUPPORTED PREAMBLE PATTERN"):
            _extract_preprocess_constants(model.graph)

    def test_both_inputs_constant_raises(self):
        """Sub/Div/Mul node with both inputs as constants raises NotImplementedError."""
        for op_type in ("Sub", "Div", "Mul"):
            model = _make_model_two_constants(op_type)
            with pytest.raises(NotImplementedError, match="UNSUPPORTED PREAMBLE PATTERN"):
                _extract_preprocess_constants(model.graph)

    def test_neither_input_constant_raises(self):
        """Sub/Div/Mul node with no constant inputs raises NotImplementedError."""
        for op_type in ("Sub", "Div", "Mul"):
            model = _make_model_two_data_inputs(op_type)
            with pytest.raises(NotImplementedError, match="UNSUPPORTED PREAMBLE PATTERN"):
                _extract_preprocess_constants(model.graph)

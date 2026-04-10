# Copyright Axelera AI, 2026
# Tests for ONNX Focus layer replacement functionality
#
# NOTE: This test is excluded from py310-runtime tox environment because:
# - base_onnx.py is only used during deployment/compilation phase
# - ONNX dependencies are not needed at runtime
# - See tox.ini [testenv:py{310,312}-runtime] for exclusion

from pathlib import Path

import numpy as np
import onnx
import pytest

from ax_models.base_onnx import apply_onnx_graph_optimizations
from ax_models.onnx_optimizations import (
    FocusConvPattern,
    detect_focus_conv_pattern,
    replace_focus_layer,
    transform_focus_conv_weights,
)
from axelera import types

ASSETS_DIR = Path(__file__).parent.parent / "assets"


class TestDetectFocusConvPattern:
    """Tests for detect_focus_conv_pattern() function."""

    def test_detect_focus_conv_pattern_valid(self):
        """Detects direct Focus+Conv pattern (4 slices) in focus_conv_test.onnx."""
        model = onnx.load(str(ASSETS_DIR / "focus_conv_test.onnx"))
        pattern = detect_focus_conv_pattern(model)

        assert pattern is not None
        assert isinstance(pattern, FocusConvPattern)
        assert len(pattern.slice_nodes) == 4
        assert pattern.concat_node.op_type == "Concat"
        assert pattern.conv_node.op_type == "Conv"
        assert pattern.pattern_input_name == "images"

    def test_detect_focus_conv_pattern_nested(self):
        """Detects nested Focus+Conv pattern (6 slices) in focus_conv_nested_test.onnx."""
        model = onnx.load(str(ASSETS_DIR / "focus_conv_nested_test.onnx"))
        pattern = detect_focus_conv_pattern(model)

        assert pattern is not None
        assert isinstance(pattern, FocusConvPattern)
        assert len(pattern.slice_nodes) == 6  # 2 H-slices + 4 W-slices
        assert pattern.concat_node.op_type == "Concat"
        assert pattern.conv_node.op_type == "Conv"
        assert pattern.pattern_input_name == "images"

    def test_detect_focus_conv_pattern_no_conv(self):
        """Returns None for Focus-only (no Conv) in focus_preprocess_graph.onnx."""
        model = onnx.load(str(ASSETS_DIR / "focus_preprocess_graph.onnx"))
        pattern = detect_focus_conv_pattern(model)

        # Has Focus pattern but no Conv after it
        assert pattern is None

    def test_detect_focus_conv_pattern_empty_model(self):
        """Returns None for empty model."""
        # Create minimal empty model
        graph = onnx.helper.make_graph([], "empty", [], [])
        model = onnx.helper.make_model(graph)
        pattern = detect_focus_conv_pattern(model)

        assert pattern is None


class TestTransformFocusConvWeights:
    """Tests for transform_focus_conv_weights() function."""

    def test_weight_transformation_shape(self):
        """Kernel [32, 12, 3, 3] -> [32, 3, 6, 6]."""
        np.random.seed(42)
        old_weight = np.random.randn(32, 12, 3, 3).astype(np.float32)
        new_weight = transform_focus_conv_weights(
            old_weight, in_channels=3, concat_order=[0, 1, 2, 3]
        )

        assert new_weight.shape == (32, 3, 6, 6)
        assert new_weight.dtype == np.float32

    def test_weight_transformation_shape_1x1(self):
        """Kernel [64, 12, 1, 1] -> [64, 3, 2, 2]."""
        old_weight = np.random.randn(64, 12, 1, 1).astype(np.float32)
        new_weight = transform_focus_conv_weights(
            old_weight, in_channels=3, concat_order=[0, 1, 2, 3]
        )

        assert new_weight.shape == (64, 3, 2, 2)

    def test_weight_transformation_values(self):
        """Weight values placed correctly in quadrants."""
        # Create test weight with known values
        old_weight = np.zeros((1, 12, 1, 1), dtype=np.float32)
        old_weight[0, 0:3, 0, 0] = [1, 2, 3]  # quadrant 0 (top-left)
        old_weight[0, 3:6, 0, 0] = [4, 5, 6]  # quadrant 1 (bottom-left)
        old_weight[0, 6:9, 0, 0] = [7, 8, 9]  # quadrant 2 (top-right)
        old_weight[0, 9:12, 0, 0] = [10, 11, 12]  # quadrant 3 (bottom-right)

        new_weight = transform_focus_conv_weights(
            old_weight, in_channels=3, concat_order=[0, 1, 2, 3]
        )

        # new_weight shape: [1, 3, 2, 2]
        assert new_weight.shape == (1, 3, 2, 2)
        assert np.allclose(new_weight[0, :, 0, 0], [1, 2, 3])  # top-left
        assert np.allclose(new_weight[0, :, 0, 1], [7, 8, 9])  # top-right
        assert np.allclose(new_weight[0, :, 1, 0], [4, 5, 6])  # bottom-left
        assert np.allclose(new_weight[0, :, 1, 1], [10, 11, 12])  # bottom-right

    def test_weight_transformation_grayscale(self):
        """Handles single-channel (grayscale) input."""
        old_weight = np.random.randn(16, 4, 3, 3).astype(np.float32)
        new_weight = transform_focus_conv_weights(
            old_weight, in_channels=1, concat_order=[0, 1, 2, 3]
        )

        assert new_weight.shape == (16, 1, 6, 6)


class TestReplaceFocusLayerOnnx:
    """Tests for replace_focus_layer() function."""

    def test_replace_focus_layer_graph_structure(self):
        """After replacement: 1 Conv node, no Slice/Concat."""
        model = onnx.load(str(ASSETS_DIR / "focus_conv_test.onnx"))
        replaced = replace_focus_layer(model)

        node_types = [n.op_type for n in replaced.graph.node]
        assert "Slice" not in node_types
        assert "Concat" not in node_types
        assert node_types.count("Conv") == 1

        # Check new Conv attributes
        conv = replaced.graph.node[0]
        assert conv.op_type == "Conv"

        kernel_shape = None
        strides = None
        for attr in conv.attribute:
            if attr.name == "kernel_shape":
                kernel_shape = list(attr.ints)
            elif attr.name == "strides":
                strides = list(attr.ints)

        # Original was 3x3 kernel, stride 1 -> should be 6x6 kernel, stride 2
        assert kernel_shape == [6, 6]
        assert strides == [2, 2]

    def test_replace_focus_layer_model_valid(self):
        """Replaced model passes ONNX validation."""
        model = onnx.load(str(ASSETS_DIR / "focus_conv_test.onnx"))
        replaced = replace_focus_layer(model)

        # Should not raise
        onnx.checker.check_model(replaced)

    def test_replace_focus_layer_no_pattern(self):
        """Model without Focus pattern returned unchanged."""
        model = onnx.load(str(ASSETS_DIR / "focus_preprocess_graph.onnx"))
        result = replace_focus_layer(model)

        # No Conv after Focus, so pattern not detected, model unchanged
        assert result is model

    def test_replace_focus_layer_numerical_equivalence(self):
        """Original and replaced model produce same output."""
        onnxruntime = pytest.importorskip("onnxruntime")

        model = onnx.load(str(ASSETS_DIR / "focus_conv_test.onnx"))
        replaced = replace_focus_layer(model)

        # Create test input
        np.random.seed(123)
        test_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

        # Run original
        sess_orig = onnxruntime.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        output_orig = sess_orig.run(None, {"images": test_input})[0]

        # Run replaced
        sess_replaced = onnxruntime.InferenceSession(
            replaced.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        output_replaced = sess_replaced.run(None, {"images": test_input})[0]

        # Compare (should be identical within float precision)
        assert output_orig.shape == output_replaced.shape
        assert np.allclose(output_orig, output_replaced, rtol=1e-5, atol=1e-6)

    def test_replace_focus_layer_nested_graph_structure(self):
        """After nested replacement: no Slice/Concat nodes, reduced node count."""
        model = onnx.load(str(ASSETS_DIR / "focus_conv_nested_test.onnx"))
        replaced = replace_focus_layer(model)

        # Should have no Slice or Concat nodes
        node_types = [n.op_type for n in replaced.graph.node]
        assert "Slice" not in node_types
        assert "Concat" not in node_types

        # Should have removed 6 Slices + 1 Concat = 7 nodes
        assert len(replaced.graph.node) == len(model.graph.node) - 7

    def test_replace_focus_layer_nested_numerical_equivalence(self):
        """Nested pattern: original and replaced model produce same output."""
        onnxruntime = pytest.importorskip("onnxruntime")

        model = onnx.load(str(ASSETS_DIR / "focus_conv_nested_test.onnx"))
        replaced = replace_focus_layer(model)

        np.random.seed(456)
        test_input = np.random.randn(1, 3, 640, 640).astype(np.float32)

        sess_orig = onnxruntime.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        output_orig = sess_orig.run(None, {"images": test_input})[0]

        sess_replaced = onnxruntime.InferenceSession(
            replaced.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        output_replaced = sess_replaced.run(None, {"images": test_input})[0]

        assert output_orig.shape == output_replaced.shape
        assert np.allclose(output_orig, output_replaced, rtol=1e-5, atol=1e-6)

    def test_replace_focus_layer_preserves_input_output_names(self):
        """Input and output names preserved after replacement."""
        model = onnx.load(str(ASSETS_DIR / "focus_conv_test.onnx"))

        orig_input_names = [i.name for i in model.graph.input]
        orig_output_names = [o.name for o in model.graph.output]

        replaced = replace_focus_layer(model)

        new_input_names = [i.name for i in replaced.graph.input]
        new_output_names = [o.name for o in replaced.graph.output]

        assert orig_input_names == new_input_names
        assert orig_output_names == new_output_names


class TestApplyOnnxGraphOptimizations:
    """Tests for apply_onnx_graph_optimizations() wrapper function."""

    def test_optimizations_enabled_by_default(self):
        """Focus layer replacement enabled by default when no config."""
        model = onnx.load(str(ASSETS_DIR / "focus_conv_test.onnx"))

        # Create mock ModelInfo with minimal fields
        from unittest.mock import Mock

        model_info = Mock()
        model_info.extra_kwargs = {}

        optimized = apply_onnx_graph_optimizations(model, model_info)

        # Should have replaced Focus pattern (default is True)
        node_types = [n.op_type for n in optimized.graph.node]
        assert "Slice" not in node_types
        assert "Concat" not in node_types
        assert node_types.count("Conv") == 1

    def test_optimizations_enabled_with_yolo_config_unspecified(self):
        """Focus layer replacement enabled when YOLO config exists but flag not specified."""
        model = onnx.load(str(ASSETS_DIR / "focus_conv_test.onnx"))

        from unittest.mock import Mock

        model_info = Mock()
        model_info.extra_kwargs = {"YOLO": {}}  # No focus_layer_replacement key

        optimized = apply_onnx_graph_optimizations(model, model_info)

        # Should have replaced Focus pattern (default is True)
        node_types = [n.op_type for n in optimized.graph.node]
        assert "Slice" not in node_types
        assert "Concat" not in node_types
        assert node_types.count("Conv") == 1

    def test_optimizations_explicitly_enabled(self):
        """Focus layer replacement works when explicitly enabled."""
        model = onnx.load(str(ASSETS_DIR / "focus_conv_test.onnx"))

        from unittest.mock import Mock

        model_info = Mock()
        model_info.extra_kwargs = {"YOLO": {"focus_layer_replacement": True}}

        optimized = apply_onnx_graph_optimizations(model, model_info)

        # Should have replaced Focus pattern
        node_types = [n.op_type for n in optimized.graph.node]
        assert "Slice" not in node_types
        assert "Concat" not in node_types
        assert node_types.count("Conv") == 1

    def test_optimizations_explicitly_disabled(self):
        """Focus layer replacement disabled when explicitly set to False."""
        model = onnx.load(str(ASSETS_DIR / "focus_conv_test.onnx"))

        from unittest.mock import Mock

        model_info = Mock()
        model_info.extra_kwargs = {"YOLO": {"focus_layer_replacement": False}}

        optimized = apply_onnx_graph_optimizations(model, model_info)

        # Should NOT have replaced Focus pattern
        node_types = [n.op_type for n in optimized.graph.node]
        assert "Slice" in node_types
        assert "Concat" in node_types
        assert node_types.count("Conv") == 1


class TestAxONNXModelIntegration:
    """Integration tests verifying modified model reaches compiler."""

    def test_init_model_deploy_applies_focus_replacement(self, tmp_path):
        """Verify AxONNXModel.init_model_deploy() applies Focus replacement."""
        import shutil
        from unittest.mock import Mock

        from ax_models.base_onnx import AxONNXModel

        # Copy test ONNX to temp location
        test_onnx = tmp_path / "focus_conv_test.onnx"
        shutil.copy(ASSETS_DIR / "focus_conv_test.onnx", test_onnx)

        # Create model info with default config (Focus replacement enabled)
        model_info = Mock()
        model_info.name = "test-model"
        model_info.weight_path = str(test_onnx)
        model_info.weight_md5 = None
        model_info.extra_kwargs = {}

        # Create and initialize model
        model = AxONNXModel()
        model.init_model_deploy(model_info, {})

        # Verify the onnx_model property (what compiler receives) has Focus replaced
        node_types = [n.op_type for n in model.onnx_model.graph.node]
        assert "Slice" not in node_types
        assert "Concat" not in node_types
        assert node_types.count("Conv") == 1

    def test_init_model_deploy_respects_disabled_flag(self, tmp_path):
        """Verify Focus replacement can be disabled via config."""
        import shutil
        from unittest.mock import Mock

        from ax_models.base_onnx import AxONNXModel

        # Copy test ONNX to temp location
        test_onnx = tmp_path / "focus_conv_test.onnx"
        shutil.copy(ASSETS_DIR / "focus_conv_test.onnx", test_onnx)

        # Create model info with Focus replacement disabled
        model_info = Mock()
        model_info.name = "test-model"
        model_info.weight_path = str(test_onnx)
        model_info.weight_md5 = None
        model_info.extra_kwargs = {"YOLO": {"focus_layer_replacement": False}}

        # Create and initialize model
        model = AxONNXModel()
        model.init_model_deploy(model_info, {})

        # Verify the onnx_model property still has Focus pattern
        node_types = [n.op_type for n in model.onnx_model.graph.node]
        assert "Slice" in node_types
        assert "Concat" in node_types

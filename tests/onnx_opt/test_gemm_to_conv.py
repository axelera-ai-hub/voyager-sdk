# Copyright Axelera AI, 2026
# Tests for Gemm to Conv ONNX optimization

import numpy as np
import pytest

onnx = pytest.importorskip("onnx")
onnxruntime = pytest.importorskip("onnxruntime")

from onnx import helper, numpy_helper, TensorProto

from ax_models.onnx_optimizations import (
    GEMM_TO_CONV_REDUCTION_THRESHOLD,
    detect_gemm_to_conv_pattern,
    replace_gemm_with_conv,
    validate_gemm_conv_equivalence,
)


def create_gemm_model(
    batch_size: int,
    in_channels: int,
    spatial_h: int,
    spatial_w: int,
    out_features: int,
    with_bias: bool = True,
    trans_b: bool = True,
    use_matmul: bool = False,
    add_batchnorm_before: bool = False,
):
    """
    Create a simple ONNX model with Reshape -> Gemm pattern.

    Args:
        batch_size: Batch size (0 for dynamic)
        in_channels: Number of input channels
        spatial_h: Spatial height
        spatial_w: Spatial width
        out_features: Output features from Gemm
        with_bias: Whether to include bias in Gemm
        trans_b: Whether to transpose B in Gemm
        use_matmul: Use MatMul instead of Gemm
        add_batchnorm_before: Add BatchNorm before Reshape
    """
    in_features = in_channels * spatial_h * spatial_w

    # Input
    input_shape = [batch_size, in_channels, spatial_h, spatial_w]
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape)

    # Output
    output_shape = [batch_size, out_features]
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)

    nodes = []
    initializers = []
    current_input = "input"

    # Optional BatchNorm before Reshape
    if add_batchnorm_before:
        bn_scale = np.ones(in_channels, dtype=np.float32)
        bn_bias = np.zeros(in_channels, dtype=np.float32)
        bn_mean = np.zeros(in_channels, dtype=np.float32)
        bn_var = np.ones(in_channels, dtype=np.float32)

        initializers.extend(
            [
                numpy_helper.from_array(bn_scale, "bn_scale"),
                numpy_helper.from_array(bn_bias, "bn_bias"),
                numpy_helper.from_array(bn_mean, "bn_mean"),
                numpy_helper.from_array(bn_var, "bn_var"),
            ]
        )

        bn_node = helper.make_node(
            "BatchNormalization",
            inputs=[current_input, "bn_scale", "bn_bias", "bn_mean", "bn_var"],
            outputs=["bn_output"],
            epsilon=1e-5,
        )
        nodes.append(bn_node)
        current_input = "bn_output"

    # Shape for Reshape: [batch, in_features]
    # Use Concat to build shape dynamically for batch dimension
    shape_node = helper.make_node(
        "Shape",
        inputs=[current_input],
        outputs=["input_shape"],
    )
    nodes.append(shape_node)

    gather_node = helper.make_node(
        "Gather",
        inputs=["input_shape", "zero_idx"],
        outputs=["batch_dim"],
        axis=0,
    )
    nodes.append(gather_node)

    unsqueeze_node = helper.make_node(
        "Unsqueeze",
        inputs=["batch_dim", "zero_axes"],
        outputs=["batch_dim_unsq"],
    )
    nodes.append(unsqueeze_node)

    concat_node = helper.make_node(
        "Concat",
        inputs=["batch_dim_unsq", "flatten_dim"],
        outputs=["reshape_shape"],
        axis=0,
    )
    nodes.append(concat_node)

    # Initializers for shape computation
    initializers.extend(
        [
            numpy_helper.from_array(np.array(0, dtype=np.int64), "zero_idx"),
            numpy_helper.from_array(np.array([0], dtype=np.int64), "zero_axes"),
            numpy_helper.from_array(np.array([in_features], dtype=np.int64), "flatten_dim"),
        ]
    )

    # Reshape node
    reshape_node = helper.make_node(
        "Reshape",
        inputs=[current_input, "reshape_shape"],
        outputs=["reshaped"],
    )
    nodes.append(reshape_node)

    # Gemm or MatMul
    if trans_b:
        weight_shape = (out_features, in_features)
    else:
        weight_shape = (in_features, out_features)

    weight = np.random.randn(*weight_shape).astype(np.float32) * 0.01
    initializers.append(numpy_helper.from_array(weight, "weight"))

    if use_matmul:
        # MatMul: Y = X @ W (no transpose, no bias built-in)
        if trans_b:
            # Need to transpose weight for MatMul
            weight_t = weight.T
            initializers[-1] = numpy_helper.from_array(weight_t, "weight")

        matmul_node = helper.make_node(
            "MatMul",
            inputs=["reshaped", "weight"],
            outputs=["matmul_out" if with_bias else "output"],
        )
        nodes.append(matmul_node)

        if with_bias:
            bias = np.random.randn(out_features).astype(np.float32) * 0.01
            initializers.append(numpy_helper.from_array(bias, "bias"))

            add_node = helper.make_node(
                "Add",
                inputs=["matmul_out", "bias"],
                outputs=["output"],
            )
            nodes.append(add_node)
    else:
        gemm_inputs = ["reshaped", "weight"]
        if with_bias:
            bias = np.random.randn(out_features).astype(np.float32) * 0.01
            initializers.append(numpy_helper.from_array(bias, "bias"))
            gemm_inputs.append("bias")

        gemm_node = helper.make_node(
            "Gemm",
            inputs=gemm_inputs,
            outputs=["output"],
            transB=1 if trans_b else 0,
        )
        nodes.append(gemm_node)

    # Create graph and model
    graph = helper.make_graph(nodes, "test_gemm", [X], [Y], initializers)
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])
    model.ir_version = 8

    return model


class TestGemmToConvDetection:
    """Tests for pattern detection."""

    def test_detect_basic_pattern(self):
        """Test detection of basic Reshape -> Gemm pattern."""
        model = create_gemm_model(
            batch_size=0,
            in_channels=512,
            spatial_h=7,
            spatial_w=7,
            out_features=2048,
        )

        pattern = detect_gemm_to_conv_pattern(model)

        assert pattern is not None
        assert pattern.spatial_h == 7
        assert pattern.spatial_w == 7
        assert pattern.in_channels == 512
        assert pattern.out_features == 2048

    def test_detect_14x14_pattern(self):
        """Test detection with 14x14 spatial size."""
        model = create_gemm_model(
            batch_size=0,
            in_channels=256,
            spatial_h=14,
            spatial_w=14,
            out_features=1024,
        )

        pattern = detect_gemm_to_conv_pattern(model)

        assert pattern is not None
        assert pattern.spatial_h == 14
        assert pattern.spatial_w == 14
        assert pattern.in_channels == 256

    def test_detect_with_batchnorm(self):
        """Test detection when BatchNorm precedes Reshape."""
        model = create_gemm_model(
            batch_size=0,
            in_channels=512,
            spatial_h=7,
            spatial_w=7,
            out_features=2048,
            add_batchnorm_before=True,
        )

        pattern = detect_gemm_to_conv_pattern(model)

        assert pattern is not None
        assert pattern.in_channels == 512

    def test_skip_small_gemm(self):
        """Test that small Gemm (below threshold) is not detected."""
        # 64 * 1 * 1 = 64 features, reduction_blocks = 1
        model = create_gemm_model(
            batch_size=0,
            in_channels=64,
            spatial_h=1,
            spatial_w=1,
            out_features=128,
        )

        pattern = detect_gemm_to_conv_pattern(model)

        # Should be None because reduction blocks (1) <= threshold
        assert pattern is None

    def test_detect_without_bias(self):
        """Test detection of Gemm without bias."""
        model = create_gemm_model(
            batch_size=0,
            in_channels=512,
            spatial_h=7,
            spatial_w=7,
            out_features=2048,
            with_bias=False,
        )

        pattern = detect_gemm_to_conv_pattern(model)

        assert pattern is not None


class TestGemmToConvConversion:
    """Tests for the actual conversion."""

    @pytest.mark.parametrize(
        "in_channels,spatial_h,spatial_w,out_features",
        [
            (512, 7, 7, 2048),  # Face recognition typical
            (256, 14, 14, 1024),  # Larger spatial
            (2048, 7, 7, 512),  # High channel count
            (128, 14, 14, 512),  # Medium size
        ],
    )
    def test_conversion_various_sizes(self, in_channels, spatial_h, spatial_w, out_features):
        """Test conversion with various tensor sizes."""
        model = create_gemm_model(
            batch_size=0,
            in_channels=in_channels,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
            out_features=out_features,
        )

        converted = replace_gemm_with_conv(model)

        # Check that Gemm is replaced with Conv
        op_types = [n.op_type for n in converted.graph.node]
        assert "Gemm" not in op_types
        assert "Conv" in op_types

    def test_conversion_with_bias(self):
        """Test conversion preserves bias."""
        model = create_gemm_model(
            batch_size=0,
            in_channels=512,
            spatial_h=7,
            spatial_w=7,
            out_features=2048,
            with_bias=True,
        )

        converted = replace_gemm_with_conv(model)

        # Find Conv node and check it has bias
        conv_node = next(n for n in converted.graph.node if n.op_type == "Conv")
        assert len(conv_node.input) == 3  # input, weight, bias

    def test_conversion_without_bias(self):
        """Test conversion without bias."""
        model = create_gemm_model(
            batch_size=0,
            in_channels=512,
            spatial_h=7,
            spatial_w=7,
            out_features=2048,
            with_bias=False,
        )

        converted = replace_gemm_with_conv(model)

        # Find Conv node and check it has no bias
        conv_node = next(n for n in converted.graph.node if n.op_type == "Conv")
        assert len(conv_node.input) == 2  # input, weight only

    def test_conversion_kernel_shape(self):
        """Test that Conv has correct kernel shape."""
        model = create_gemm_model(
            batch_size=0,
            in_channels=512,
            spatial_h=7,
            spatial_w=7,
            out_features=2048,
        )

        converted = replace_gemm_with_conv(model)

        conv_node = next(n for n in converted.graph.node if n.op_type == "Conv")
        kernel_shape = next(a.ints for a in conv_node.attribute if a.name == "kernel_shape")
        assert list(kernel_shape) == [7, 7]


class TestGemmToConvEquivalence:
    """Tests for numerical equivalence."""

    @pytest.mark.parametrize(
        "in_channels,spatial_h,spatial_w,out_features,with_bias",
        [
            (512, 7, 7, 2048, True),
            (512, 7, 7, 2048, False),
            (256, 14, 14, 1024, True),
            (2048, 7, 7, 512, True),
        ],
    )
    def test_numerical_equivalence(
        self, in_channels, spatial_h, spatial_w, out_features, with_bias
    ):
        """Test that converted model produces identical outputs."""
        # Use batch_size=1 for numerical testing (batch_size=0 causes ONNX runtime issues)
        model = create_gemm_model(
            batch_size=1,
            in_channels=in_channels,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
            out_features=out_features,
            with_bias=with_bias,
        )

        converted = replace_gemm_with_conv(model)

        # Validate equivalence (use relaxed tolerance for float32 precision)
        assert validate_gemm_conv_equivalence(model, converted, rtol=1e-4, atol=1e-5)

    def test_equivalence_multiple_inputs(self):
        """Test equivalence with multiple random inputs."""
        model = create_gemm_model(
            batch_size=1,
            in_channels=512,
            spatial_h=7,
            spatial_w=7,
            out_features=2048,
        )

        converted = replace_gemm_with_conv(model)

        # Create sessions
        orig_session = onnxruntime.InferenceSession(
            model.SerializeToString(), providers=["CPUExecutionProvider"]
        )
        conv_session = onnxruntime.InferenceSession(
            converted.SerializeToString(), providers=["CPUExecutionProvider"]
        )

        # Test with multiple random inputs
        for _ in range(5):
            test_input = np.random.randn(1, 512, 7, 7).astype(np.float32)

            orig_out = orig_session.run(None, {"input": test_input})[0]
            conv_out = conv_session.run(None, {"input": test_input})[0]

            assert np.allclose(orig_out, conv_out, rtol=1e-4, atol=1e-5)

    def test_equivalence_different_batch_sizes(self):
        """Test equivalence with different batch sizes."""
        # Test each batch size separately since ONNX models have fixed input shapes
        for batch_size in [1, 2, 4]:
            model = create_gemm_model(
                batch_size=batch_size,
                in_channels=512,
                spatial_h=7,
                spatial_w=7,
                out_features=2048,
            )

            converted = replace_gemm_with_conv(model)

            orig_session = onnxruntime.InferenceSession(
                model.SerializeToString(), providers=["CPUExecutionProvider"]
            )
            conv_session = onnxruntime.InferenceSession(
                converted.SerializeToString(), providers=["CPUExecutionProvider"]
            )

            test_input = np.random.randn(batch_size, 512, 7, 7).astype(np.float32)

            orig_out = orig_session.run(None, {"input": test_input})[0]
            conv_out = conv_session.run(None, {"input": test_input})[0]

            assert np.allclose(orig_out, conv_out, rtol=1e-4, atol=1e-5)


class TestGemmToConvEdgeCases:
    """Tests for edge cases and error handling."""

    def test_no_gemm_model(self):
        """Test that model without Gemm is returned unchanged."""
        # Create a simple Conv-only model
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
        Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 64, 112, 112])

        weight = np.random.randn(64, 3, 7, 7).astype(np.float32)
        weight_init = numpy_helper.from_array(weight, "weight")

        conv_node = helper.make_node(
            "Conv",
            inputs=["input", "weight"],
            outputs=["output"],
            kernel_shape=[7, 7],
            strides=[2, 2],
            pads=[3, 3, 3, 3],
        )

        graph = helper.make_graph([conv_node], "test", [X], [Y], [weight_init])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        result = replace_gemm_with_conv(model)

        # Model should be unchanged
        assert result is model

    def test_gemm_not_from_reshape(self):
        """Test that Gemm not preceded by Reshape is skipped."""
        # Create model with Gemm directly on 2D input
        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 512])
        Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 256])

        weight = np.random.randn(256, 512).astype(np.float32)
        weight_init = numpy_helper.from_array(weight, "weight")

        gemm_node = helper.make_node(
            "Gemm",
            inputs=["input", "weight"],
            outputs=["output"],
            transB=1,
        )

        graph = helper.make_graph([gemm_node], "test", [X], [Y], [weight_init])
        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)])

        result = replace_gemm_with_conv(model)

        # Model should be unchanged (no Reshape before Gemm)
        assert result is model


class TestMatMulToConv:
    """Tests for MatMul to Conv conversion."""

    def test_detect_matmul_pattern(self):
        """Test detection of Reshape -> MatMul pattern."""
        model = create_gemm_model(
            batch_size=0,
            in_channels=512,
            spatial_h=7,
            spatial_w=7,
            out_features=2048,
            use_matmul=True,
            with_bias=True,
        )

        pattern = detect_gemm_to_conv_pattern(model)

        assert pattern is not None
        assert pattern.is_matmul is True
        assert pattern.spatial_h == 7
        assert pattern.spatial_w == 7
        assert pattern.in_channels == 512
        assert pattern.out_features == 2048
        assert pattern.bias_node is not None

    def test_detect_matmul_without_bias(self):
        """Test detection of MatMul without bias (Add node)."""
        model = create_gemm_model(
            batch_size=0,
            in_channels=512,
            spatial_h=7,
            spatial_w=7,
            out_features=2048,
            use_matmul=True,
            with_bias=False,
        )

        pattern = detect_gemm_to_conv_pattern(model)

        assert pattern is not None
        assert pattern.is_matmul is True
        assert pattern.bias_node is None

    def test_matmul_conversion_with_bias(self):
        """Test MatMul+Add conversion produces Conv with bias."""
        model = create_gemm_model(
            batch_size=0,
            in_channels=512,
            spatial_h=7,
            spatial_w=7,
            out_features=2048,
            use_matmul=True,
            with_bias=True,
        )

        converted = replace_gemm_with_conv(model)

        # Check that MatMul and Add are replaced with Conv
        op_types = [n.op_type for n in converted.graph.node]
        assert "MatMul" not in op_types
        assert "Add" not in op_types or op_types.count("Add") == 0  # The bias Add should be gone
        assert "Conv" in op_types

        # Conv should have bias
        conv_node = next(n for n in converted.graph.node if n.op_type == "Conv")
        assert len(conv_node.input) == 3

    def test_matmul_conversion_without_bias(self):
        """Test MatMul conversion without bias."""
        model = create_gemm_model(
            batch_size=0,
            in_channels=512,
            spatial_h=7,
            spatial_w=7,
            out_features=2048,
            use_matmul=True,
            with_bias=False,
        )

        converted = replace_gemm_with_conv(model)

        op_types = [n.op_type for n in converted.graph.node]
        assert "MatMul" not in op_types
        assert "Conv" in op_types

        # Conv should not have bias
        conv_node = next(n for n in converted.graph.node if n.op_type == "Conv")
        assert len(conv_node.input) == 2

    @pytest.mark.parametrize(
        "in_channels,spatial_h,spatial_w,out_features,with_bias",
        [
            (512, 7, 7, 2048, True),
            (512, 7, 7, 2048, False),
            (256, 14, 14, 1024, True),
            (2048, 7, 7, 512, True),
        ],
    )
    def test_matmul_numerical_equivalence(
        self, in_channels, spatial_h, spatial_w, out_features, with_bias
    ):
        """Test that MatMul converted model produces identical outputs."""
        # Use batch_size=1 for numerical testing
        model = create_gemm_model(
            batch_size=1,
            in_channels=in_channels,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
            out_features=out_features,
            with_bias=with_bias,
            use_matmul=True,
        )

        converted = replace_gemm_with_conv(model)

        assert validate_gemm_conv_equivalence(model, converted, rtol=1e-4, atol=1e-5)


class TestRealModelPattern:
    """Test with pattern similar to real face recognition model."""

    def test_arcface_like_pattern(self):
        """Test pattern similar to ArcFace face recognition model."""
        # ArcFace typically has: features (512, 7, 7) -> flatten -> FC (25088 -> 512)
        # Use batch_size=1 for numerical testing
        model = create_gemm_model(
            batch_size=1,
            in_channels=512,
            spatial_h=7,
            spatial_w=7,
            out_features=512,
            with_bias=True,
            add_batchnorm_before=True,
        )

        # Detect pattern
        pattern = detect_gemm_to_conv_pattern(model)
        assert pattern is not None

        # Convert
        converted = replace_gemm_with_conv(model)

        # Validate
        assert validate_gemm_conv_equivalence(model, converted)

        # Check structure
        op_types = [n.op_type for n in converted.graph.node]
        assert "Conv" in op_types
        assert "Gemm" not in op_types

    def test_large_embedding_model(self):
        """Test with large embedding dimension (like the recognizer.onnx)."""
        # Similar to recognizer.onnx: (512, 14, 14) -> flatten -> FC (100352 -> 2048)
        # Use batch_size=1 for numerical testing
        model = create_gemm_model(
            batch_size=1,
            in_channels=512,
            spatial_h=14,
            spatial_w=14,
            out_features=2048,
            with_bias=True,
            add_batchnorm_before=True,
        )

        pattern = detect_gemm_to_conv_pattern(model)
        assert pattern is not None
        assert pattern.spatial_h == 14
        assert pattern.spatial_w == 14
        assert pattern.in_channels == 512

        converted = replace_gemm_with_conv(model)
        assert validate_gemm_conv_equivalence(model, converted)

        # Verify Conv kernel shape is 14x14
        conv_node = next(n for n in converted.graph.node if n.op_type == "Conv")
        kernel_shape = next(a.ints for a in conv_node.attribute if a.name == "kernel_shape")
        assert list(kernel_shape) == [14, 14]

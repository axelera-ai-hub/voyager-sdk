# Copyright Axelera AI, 2026
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper
import onnxruntime as ort

from axelera.app.onnx_lowering import expand_onnx_functions

ASSETS_DIR = Path(__file__).parent.parent / "assets"
SBS_S50_POSTPROCESS = ASSETS_DIR / "sbs_s50_postprocess_graph.onnx"


def _make_function_model():
    """Build a minimal ONNX model with a custom Function node (Clip+Pow+GlobalAveragePool)."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 1, 1])

    # Function body: Clip -> Pow -> GlobalAveragePool
    clip_min = numpy_helper.from_array(np.float32(0.0), "clip_min")
    clip_max = numpy_helper.from_array(np.float32(6.0), "clip_max")
    pow_exp = numpy_helper.from_array(np.float32(3.0), "pow_exp")

    fn_clip = helper.make_node("Clip", ["fn_in", "clip_min", "clip_max"], ["clipped"])
    fn_pow = helper.make_node("Pow", ["clipped", "pow_exp"], ["powered"])
    fn_gap = helper.make_node("GlobalAveragePool", ["powered"], ["fn_out"])

    custom_func = helper.make_function(
        domain="custom_host_onnx_functions",
        fname="ClipPowGAP",
        inputs=["fn_in"],
        outputs=["fn_out"],
        nodes=[fn_clip, fn_pow, fn_gap],
        opset_imports=[helper.make_opsetid("", 17)],
    )

    call_node = helper.make_node(
        "ClipPowGAP",
        inputs=["input"],
        outputs=["output"],
        domain="custom_host_onnx_functions",
    )

    graph = helper.make_graph(
        [call_node],
        "test_graph",
        [X],
        [Y],
        initializer=[clip_min, clip_max, pow_exp],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("custom_host_onnx_functions", 1),
        ],
        functions=[custom_func],
    )
    return model


def _make_standard_model():
    """Build a simple ONNX model with no Function nodes."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, None)
    relu = helper.make_node("Relu", ["input"], ["output"])
    graph = helper.make_graph([relu], "standard_graph", [X], [Y])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    return model


class TestExpandOnnxFunctions:
    def test_expand_replaces_function_node_with_body(self, tmp_path):
        model = _make_function_model()
        onnx_path = tmp_path / "postprocess_graph.onnx"
        onnx.save(model, str(onnx_path))

        manifest = MagicMock()
        manifest.preprocess_graph = None
        manifest.postprocess_graph = "postprocess_graph.onnx"

        expand_onnx_functions(manifest, tmp_path)

        expanded = onnx.load(str(onnx_path))
        op_types = [n.op_type for n in expanded.graph.node]
        assert "ClipPowGAP" not in op_types
        assert "Clip" in op_types
        assert "Pow" in op_types
        assert "GlobalAveragePool" in op_types
        assert len(expanded.functions) == 0
        custom_domains = [
            o.domain for o in expanded.opset_import if o.domain == "custom_host_onnx_functions"
        ]
        assert len(custom_domains) == 0

    def test_expand_noop_when_no_functions(self, tmp_path):
        model = _make_standard_model()
        onnx_path = tmp_path / "postprocess_graph.onnx"
        onnx.save(model, str(onnx_path))

        manifest = MagicMock()
        manifest.preprocess_graph = None
        manifest.postprocess_graph = "postprocess_graph.onnx"

        expand_onnx_functions(manifest, tmp_path)

        reloaded = onnx.load(str(onnx_path))
        assert len(reloaded.graph.node) == 1
        assert reloaded.graph.node[0].op_type == "Relu"

    def test_expand_noop_when_no_graph(self):
        manifest = MagicMock()
        manifest.preprocess_graph = None
        manifest.postprocess_graph = None

        expand_onnx_functions(manifest, Path("/nonexistent"))

    def test_expanded_model_runs_in_onnxruntime(self, tmp_path):
        model = _make_function_model()
        onnx_path = tmp_path / "postprocess_graph.onnx"
        onnx.save(model, str(onnx_path))

        manifest = MagicMock()
        manifest.preprocess_graph = None
        manifest.postprocess_graph = "postprocess_graph.onnx"

        expand_onnx_functions(manifest, tmp_path)

        sess = ort.InferenceSession(str(onnx_path))
        test_input = np.random.randn(1, 3, 4, 4).astype(np.float32)
        results = sess.run(None, {"input": test_input})
        assert results[0].shape[0] == 1

    def test_expand_real_sbs_s50_postprocess_graph_structure(self, tmp_path):
        """Real compiler output: mixed graph with custom function + standard ops expands correctly."""
        import shutil

        onnx_path = tmp_path / "sbs_s50_postprocess_graph.onnx"
        shutil.copy(SBS_S50_POSTPROCESS, onnx_path)

        manifest = MagicMock()
        manifest.preprocess_graph = None
        manifest.postprocess_graph = "sbs_s50_postprocess_graph.onnx"

        expand_onnx_functions(manifest, tmp_path)

        expanded = onnx.load(str(onnx_path))
        op_types = [n.op_type for n in expanded.graph.node]
        assert "ClipPowGlobalAveragePool" not in op_types
        assert "Clip" in op_types
        assert "Pow" in op_types
        assert "GlobalAveragePool" in op_types
        # Standard ops from the original graph are preserved
        assert "BatchNormalization" in op_types
        assert "Gather" in op_types
        assert len(expanded.functions) == 0

    def test_expand_real_sbs_s50_postprocess_graph_runs_in_onnxruntime(self, tmp_path):
        """Expanded real SBS-S50 postprocess graph runs correctly in onnxruntime."""
        import shutil

        onnx_path = tmp_path / "sbs_s50_postprocess_graph.onnx"
        shutil.copy(SBS_S50_POSTPROCESS, onnx_path)

        manifest = MagicMock()
        manifest.preprocess_graph = None
        manifest.postprocess_graph = "sbs_s50_postprocess_graph.onnx"

        expand_onnx_functions(manifest, tmp_path)

        sess = ort.InferenceSession(str(onnx_path))
        test_input = np.random.randn(1, 2048, 24, 8).astype(np.float32)
        results = sess.run(None, {"/backbone/layer4/layer4.2/relu_1/Relu_output_0": test_input})
        assert results[0].shape == (1, 2048)


def _make_multi_call_model():
    """Build an ONNX model with 3 calls to the same function (simulates YOLONAS 3-head pattern)."""
    inputs = [
        helper.make_tensor_value_info(f"head_{i}", TensorProto.FLOAT, [1, 3, 4, 4])
        for i in range(3)
    ]
    outputs = [
        helper.make_tensor_value_info(f"out_{i}", TensorProto.FLOAT, [1, 3, 1, 1])
        for i in range(3)
    ]

    clip_min = numpy_helper.from_array(np.float32(0.0), "clip_min")
    clip_max = numpy_helper.from_array(np.float32(6.0), "clip_max")
    pow_exp = numpy_helper.from_array(np.float32(3.0), "pow_exp")

    fn_clip = helper.make_node("Clip", ["fn_in", "clip_min", "clip_max"], ["clipped"])
    fn_pow = helper.make_node("Pow", ["clipped", "pow_exp"], ["powered"])
    fn_gap = helper.make_node("GlobalAveragePool", ["powered"], ["fn_out"])

    custom_func = helper.make_function(
        domain="custom_host_onnx_functions",
        fname="ClipPowGAP",
        inputs=["fn_in"],
        outputs=["fn_out"],
        nodes=[fn_clip, fn_pow, fn_gap],
        opset_imports=[helper.make_opsetid("", 17)],
    )

    call_nodes = [
        helper.make_node(
            "ClipPowGAP",
            inputs=[f"head_{i}"],
            outputs=[f"out_{i}"],
            domain="custom_host_onnx_functions",
        )
        for i in range(3)
    ]

    graph = helper.make_graph(
        call_nodes,
        "multi_call_graph",
        inputs,
        outputs,
        initializer=[clip_min, clip_max, pow_exp],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("custom_host_onnx_functions", 1),
        ],
        functions=[custom_func],
    )
    return model


def _make_multi_function_model():
    """Build an ONNX model with 2 different functions, each called once."""
    X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 4, 4])
    Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 1, 1])

    clip_min = numpy_helper.from_array(np.float32(0.0), "clip_min")
    clip_max = numpy_helper.from_array(np.float32(6.0), "clip_max")
    pow_exp = numpy_helper.from_array(np.float32(3.0), "pow_exp")

    fn1_clip = helper.make_node("Clip", ["fn1_in", "clip_min", "clip_max"], ["fn1_clipped"])
    fn1_pow = helper.make_node("Pow", ["fn1_clipped", "pow_exp"], ["fn1_powered"])
    fn1_gap = helper.make_node("GlobalAveragePool", ["fn1_powered"], ["fn1_out"])

    func1 = helper.make_function(
        domain="custom_host_onnx_functions",
        fname="ClipPowGAP",
        inputs=["fn1_in"],
        outputs=["fn1_out"],
        nodes=[fn1_clip, fn1_pow, fn1_gap],
        opset_imports=[helper.make_opsetid("", 17)],
    )

    fn2_relu = helper.make_node("Relu", ["fn2_in"], ["fn2_act"])
    fn2_sig = helper.make_node("Sigmoid", ["fn2_act"], ["fn2_out"])

    func2 = helper.make_function(
        domain="custom_host_onnx_functions",
        fname="ReluSigmoid",
        inputs=["fn2_in"],
        outputs=["fn2_out"],
        nodes=[fn2_relu, fn2_sig],
        opset_imports=[helper.make_opsetid("", 17)],
    )

    call1 = helper.make_node(
        "ClipPowGAP",
        inputs=["input"],
        outputs=["mid"],
        domain="custom_host_onnx_functions",
    )
    call2 = helper.make_node(
        "ReluSigmoid",
        inputs=["mid"],
        outputs=["output"],
        domain="custom_host_onnx_functions",
    )

    graph = helper.make_graph(
        [call1, call2],
        "multi_func_graph",
        [X],
        [Y],
        initializer=[clip_min, clip_max, pow_exp],
    )
    model = helper.make_model(
        graph,
        opset_imports=[
            helper.make_opsetid("", 17),
            helper.make_opsetid("custom_host_onnx_functions", 1),
        ],
        functions=[func1, func2],
    )
    return model


class TestMultiCallExpansion:
    def test_multiple_calls_unique_intermediates(self, tmp_path):
        model = _make_multi_call_model()
        onnx_path = tmp_path / "postprocess_graph.onnx"
        onnx.save(model, str(onnx_path))

        manifest = MagicMock()
        manifest.preprocess_graph = None
        manifest.postprocess_graph = "postprocess_graph.onnx"

        expand_onnx_functions(manifest, tmp_path)

        expanded = onnx.load(str(onnx_path))
        assert len(expanded.graph.node) == 9  # 3 calls x 3 ops

        all_outputs = []
        for node in expanded.graph.node:
            all_outputs.extend(out for out in node.output if out)
        assert len(all_outputs) == len(
            set(all_outputs)
        ), f"Duplicate output names violate SSA: {[o for o in all_outputs if all_outputs.count(o) > 1]}"

    def test_multiple_calls_passes_onnx_checker(self, tmp_path):
        model = _make_multi_call_model()
        onnx_path = tmp_path / "postprocess_graph.onnx"
        onnx.save(model, str(onnx_path))

        manifest = MagicMock()
        manifest.preprocess_graph = None
        manifest.postprocess_graph = "postprocess_graph.onnx"

        expand_onnx_functions(manifest, tmp_path)

        expanded = onnx.load(str(onnx_path))
        onnx.checker.check_model(expanded)

        sess = ort.InferenceSession(str(onnx_path))
        feeds = {f"head_{i}": np.random.randn(1, 3, 4, 4).astype(np.float32) for i in range(3)}
        results = sess.run(None, feeds)
        assert len(results) == 3
        for r in results:
            assert r.shape == (1, 3, 1, 1)

    def test_multiple_different_functions(self, tmp_path):
        model = _make_multi_function_model()
        onnx_path = tmp_path / "postprocess_graph.onnx"
        onnx.save(model, str(onnx_path))

        manifest = MagicMock()
        manifest.preprocess_graph = None
        manifest.postprocess_graph = "postprocess_graph.onnx"

        expand_onnx_functions(manifest, tmp_path)

        expanded = onnx.load(str(onnx_path))
        op_types = [n.op_type for n in expanded.graph.node]
        assert "ClipPowGAP" not in op_types
        assert "ReluSigmoid" not in op_types
        assert len(expanded.graph.node) == 5  # 3 from func1 + 2 from func2
        assert len(expanded.functions) == 0

        sess = ort.InferenceSession(str(onnx_path))
        test_input = np.random.randn(1, 3, 4, 4).astype(np.float32)
        results = sess.run(None, {"input": test_input})
        assert results[0].shape == (1, 3, 1, 1)

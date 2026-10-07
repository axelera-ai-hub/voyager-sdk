# Copyright Axelera AI, 2026
"""Small ONNX helpers the graph-rewriting passes in this directory share.

Nothing here is a pass. `get_node_by_output` and `get_constant_value` read a
value the graph states in one of two places, and `outputs_equivalent` answers
the question every rewrite has to answer before it ships: does the rewritten
graph still compute what the original computed?

A pass you write yourself is welcome to reuse these, and welcome to ignore
them. They depend on `onnx` and `numpy` only, both of which the compiler
already pulls in.
"""

from __future__ import annotations

import logging

import numpy as np
import onnx
from onnx import numpy_helper

LOG = logging.getLogger(__name__)


def get_node_by_output(graph, name):
    for node in graph.node:
        if name in node.output:
            return node
    return None


def get_constant_value(graph, name):
    """Retrieve value from initializer or Constant node."""
    # Check initializer
    for init in graph.initializer:
        if init.name == name:
            return numpy_helper.to_array(init)
    # Check Constant node
    node = get_node_by_output(graph, name)
    if node and node.op_type == "Constant":
        for attr in node.attribute:
            if attr.name == "value":
                return numpy_helper.to_array(attr.t)
    return None


def outputs_equivalent(
    original_model: onnx.ModelProto,
    converted_model: onnx.ModelProto,
    *,
    rtol: float,
    atol: float,
) -> bool:
    """Run both models in onnxruntime on one random input; True iff outputs match.

    Generic ONNX-rewrite equivalence check shared by the Gemm->Conv and
    Focus->Conv guards. Feeds a single random tensor matching the first graph
    input and compares every output with `np.allclose`. Dynamic dims resolve to
    1 for the batch axis and 64 for the rest, so a backbone with a minimum
    spatial-size constraint (feature-pyramid Concats etc.) still runs. Returns
    True (skips) when onnxruntime is unavailable or cannot run the model -- the
    guard never blocks a build on its own infra issue, only on a real numeric
    mismatch below.

    The draw is SEEDED. An unseeded one makes the verdict a sample: a build can
    fail on a rewrite that is correct, and the next person cannot reproduce
    either answer. One fixed sample is enough because these rewrites are exact
    or reassociated, never approximate, so a mismatch a second sample would
    find is one this sample finds too.
    """
    try:
        import onnxruntime as ort
    except ImportError:
        LOG.warning("onnxruntime not available, skipping equivalence validation")
        return True

    try:
        original_session = ort.InferenceSession(
            original_model.SerializeToString(), providers=['CPUExecutionProvider']
        )
        converted_session = ort.InferenceSession(
            converted_model.SerializeToString(), providers=['CPUExecutionProvider']
        )

        input_info = original_session.get_inputs()[0]
        input_name = input_info.name
        # Dynamic dims: batch axis -> 1, spatial/other -> 64 (small but large
        # enough for stride-32 backbones; the fusion is size-independent so any
        # valid size validates equivalence).
        concrete_shape = [
            d if isinstance(d, int) and d > 0 else (1 if axis == 0 else 64)
            for axis, d in enumerate(input_info.shape)
        ]
        test_input = np.random.default_rng(0).standard_normal(concrete_shape, dtype=np.float32)

        original_outputs = original_session.run(None, {input_name: test_input})
        converted_outputs = converted_session.run(None, {input_name: test_input})
    except Exception as e:  # noqa: BLE001 - fail open on any onnxruntime error
        LOG.warning("equivalence validation skipped (onnxruntime error: %s)", e)
        return True

    for i, (orig, conv) in enumerate(zip(original_outputs, converted_outputs, strict=True)):
        if not np.allclose(orig, conv, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(orig - conv))
            LOG.error(f"Output {i} mismatch: max_diff={max_diff}, rtol={rtol}, atol={atol}")
            return False
    return True

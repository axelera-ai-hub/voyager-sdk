# Copyright Axelera AI, 2026
from __future__ import annotations

import copy
import typing
from collections import defaultdict
from pathlib import Path

from . import logging_utils

try:
    import onnx
except ImportError:
    if typing.TYPE_CHECKING:
        import onnx

if typing.TYPE_CHECKING:
    from axelera import types

LOG = logging_utils.getLogger(__name__)


def _get_function_intermediates(func: onnx.FunctionProto) -> frozenset[str]:
    """Return node output names that are internal to the function (not formal outputs)."""
    formal_outputs = set(func.output)
    intermediates = set()
    for node in func.node:
        for out in node.output:
            if out and out not in formal_outputs:
                intermediates.add(out)
    return frozenset(intermediates)


def _expand_single_function_call(
    func: onnx.FunctionProto,
    node: onnx.NodeProto,
    call_idx: int,
    intermediate_names: frozenset[str],
) -> list[onnx.NodeProto]:
    """Expand a single function call node into its inlined body nodes.

    Intermediate names are suffixed with the function name and call index
    to preserve ONNX SSA (each name produced exactly once).
    """
    input_map = dict(zip(func.input, node.input))
    output_map = dict(zip(func.output, node.output))

    intermediate_map = {name: f"{name}_{func.name}_exp{call_idx}" for name in intermediate_names}

    # Attribute resolution: function defaults, then call-site overrides
    attr_map: dict[str, onnx.AttributeProto] = {}
    for attr in func.attribute_proto:
        attr_map[attr.name] = attr
    for attr in node.attribute:
        attr_map[attr.name] = attr

    expanded_nodes: list[onnx.NodeProto] = []
    for fn_node in func.node:
        expanded = copy.deepcopy(fn_node)

        for i, inp in enumerate(expanded.input):
            if inp in input_map:
                expanded.input[i] = input_map[inp]
            elif inp in intermediate_map:
                expanded.input[i] = intermediate_map[inp]

        for i, out in enumerate(expanded.output):
            if out in output_map:
                expanded.output[i] = output_map[out]
            elif out in intermediate_map:
                expanded.output[i] = intermediate_map[out]

        # Resolve ref_attr_name references to concrete attribute values
        new_attrs = []
        for attr in expanded.attribute:
            if attr.ref_attr_name and attr.ref_attr_name in attr_map:
                resolved = copy.deepcopy(attr_map[attr.ref_attr_name])
                resolved.name = attr.name
                resolved.ClearField('ref_attr_name')
                new_attrs.append(resolved)
            else:
                new_attrs.append(attr)
        del expanded.attribute[:]
        expanded.attribute.extend(new_attrs)

        expanded.domain = ''
        expanded_nodes.append(expanded)

    return expanded_nodes


def expand_onnx_functions(manifest: types.Manifest, output_path: Path) -> None:
    """Expand custom ONNX Function nodes so onnxruntime can execute the graphs."""
    for graph_relpath in (manifest.preprocess_graph, manifest.postprocess_graph):
        if not graph_relpath:
            continue
        onnx_path = output_path / graph_relpath
        if not onnx_path.exists():
            continue

        model = onnx.load(onnx_path)
        if not model.functions:
            continue

        func_map = {(f.domain, f.name): f for f in model.functions}

        intermediates_map = {
            key: _get_function_intermediates(func) for key, func in func_map.items()
        }

        new_nodes = []
        call_count: dict[tuple[str, str], int] = defaultdict(int)
        for node in model.graph.node:
            key = (node.domain, node.op_type)
            if key in func_map:
                func = func_map[key]
                call_idx = call_count[key]
                call_count[key] += 1
                expanded = _expand_single_function_call(
                    func,
                    node,
                    call_idx,
                    intermediates_map[key],
                )
                new_nodes.extend(expanded)
            else:
                new_nodes.append(node)

        del model.graph.node[:]
        model.graph.node.extend(new_nodes)
        del model.functions[:]
        custom_domains = {d for d, _ in func_map}
        opsets = [o for o in model.opset_import if o.domain not in custom_domains]
        del model.opset_import[:]
        model.opset_import.extend(opsets)

        onnx.checker.check_model(model)
        LOG.info("Expanded ONNX functions in %s", onnx_path.name)
        onnx.save(model, onnx_path)

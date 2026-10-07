# Copyright Axelera AI, 2026
"""Fold a YOLO Focus stem into the convolution that follows it.

WHAT IT LOOKS LIKE WITHOUT THE PASS
    The model starts by cutting the image into four half-resolution copies,
    stacking them on the channel axis, and running a convolution over the
    stack. In the exported graph that is six `Slice` nodes feeding a
    four-input `Concat` feeding a `Conv`. The compiler has no kernel for that
    shape of stem, so the build stops on the `Concat` or on a `SpaceToDepth`
    it cannot lower.

WHAT THE PASS DOES
    Slicing then convolving is the same arithmetic as one convolution with a
    kernel twice as wide and twice as tall, taking a stride of two. The pass
    detects the stem, rearranges the convolution weights into that larger
    kernel, and replaces all eight nodes with the single `Conv`. The model is
    unchanged numerically; it is written in a form the compiler already
    lowers.

WHICH MODELS NEED IT
    YOLOv5 (v5.0-era) and YOLOX backbones exported with the Focus stem in the
    graph. YOLOv8 and later do not: they start with a plain `Conv`. If you are
    unsure, run the model through `detect_focus_conv_pattern` below; it
    returns `None` when there is no Focus stem, and the pass is then a no-op.

WHY IT CAN BE TRUSTED
    `run` compares the original and the rewritten graph on a random input and
    raises when they disagree. A stem the detector matched by accident fails
    the build instead of quietly compiling into a wrong network. Any pass that
    rewrites weights should carry the same check.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ._onnx import LOG, get_constant_value, outputs_equivalent

# What happens if you do not run this pass. Free text, not a category: a build
# that stops is the easy case, and the ones that quietly cost you something are
# the reason this line exists. The README table repeats it verbatim, and a test
# keeps the two equal.
WITHOUT_IT = 'the build stops: the compiler cannot lower the Focus stem'

# Compiler backends this rewrite is valid for.
SUPPORTS = ('axrelay',)

# Weight formats it operates on. A build in another format skips it.
SOURCE_KINDS = ('.onnx',)

# One line, printed by `apply` when the pass runs.
DESCRIPTION = 'YOLO Focus (space-to-depth) + Conv stem folded into one Conv'

# What retires this pass. Every pass here is a workaround with an end date.
RETIREMENT_CONDITION = 'the compiler lowers a Focus / space-to-depth stem natively'

# Set to the SDK version that absorbed the fix to start the one-release grace
# period: the pass is skipped with a warning instead of running.
DEPRECATED_IN = None


def run(model: onnx.ModelProto) -> onnx.ModelProto | None:
    """Fold the Focus stem, or return `None` when the graph has none.

    `None` is the runner's "nothing to rewrite here" answer, and it is reported.
    Returning the model instead would be indistinguishable from a rewrite that
    happened in place, so a build could claim a fusion that never ran.
    """
    rewritten = replace_focus_layer(model)
    if rewritten is model:
        return None
    if not validate_focus_conv_equivalence(model, rewritten):
        raise RuntimeError(
            'replace_focus_layer produced a non-equivalent ONNX rewrite; drop the pass '
            'from PRE_COMPILE_PASSES, or supply a model whose stem it recognizes.'
        )
    return rewritten


def validate_focus_conv_equivalence(
    original_model: onnx.ModelProto,
    converted_model: onnx.ModelProto,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> bool:
    """True when the fused Conv computes what the Focus stem computed."""
    ok = outputs_equivalent(original_model, converted_model, rtol=rtol, atol=atol)
    if ok:
        LOG.info('Focus->Conv equivalence validation passed')
    return ok


@dataclass
class FocusConvPattern:
    """Container for detected Focus+Conv pattern nodes."""

    slice_nodes: list[onnx.NodeProto]
    concat_node: onnx.NodeProto
    conv_node: onnx.NodeProto
    pattern_input_name: str
    concat_order: list[int]  # Maps concat index to quadrant index (0:TL, 1:BL, 2:TR, 3:BR)


def detect_slice_quadrant(graph, slice_node, output_to_node):
    """Determine which quadrant (TL, BL, TR, BR) a Slice chain represents.

    Returns:
        0: Top-Left  (H=0::2, W=0::2)
        1: Bot-Left  (H=1::2, W=0::2)
        2: Top-Right (H=0::2, W=1::2)
        3: Bot-Right (H=1::2, W=1::2)
        None: If undetermined

    """
    # Track accumulated start/step for H (axis 2) and W (axis 3)
    # Assuming NCHW format

    # Defaults
    h_start = 0
    w_start = 0

    # Trace up the slice chain
    curr_node = slice_node
    while True:
        if curr_node.op_type != "Slice":
            break

        data_input = curr_node.input[0]
        # Parse inputs: data, starts, ends, axes, steps
        inputs = curr_node.input

        # We need at least starts (index 1)
        if len(inputs) < 2:
            return None

        starts = get_constant_value(graph, inputs[1])
        axes = get_constant_value(graph, inputs[3]) if len(inputs) > 3 else None

        # If axes is missing, it defaults to [0, 1, ...], but standard ONNX usually
        # provides it or it's implicitly all axes. For robustness, we mostly care
        # about explicit slicing on 2 and 3.

        if starts is None or axes is None:
            # Cannot determine statically
            return None

        for i, axis in enumerate(axes):
            if axis == 2:  # H
                h_start += starts[i]
            elif axis == 3:  # W
                w_start += starts[i]

        if data_input not in output_to_node or output_to_node[data_input].op_type != "Slice":
            break
        curr_node = output_to_node[data_input]

    # Map accumulated offsets to quadrants
    # Check parity of start index
    h_odd = h_start % 2
    w_odd = w_start % 2

    if h_odd == 0 and w_odd == 0:
        return 0  # TL
    if h_odd == 1 and w_odd == 0:
        return 1  # BL
    if h_odd == 0 and w_odd == 1:
        return 2  # TR
    if h_odd == 1 and w_odd == 1:
        return 3  # BR

    return None


def detect_focus_conv_pattern(model: onnx.ModelProto) -> FocusConvPattern | None:
    """Detect Focus layer pattern followed by Conv in an ONNX model.

    Robustly handles various YOLOX versions (static, dynamic, nested slices).
    """
    graph = model.graph
    output_to_node = {out: node for node in graph.node for out in node.output}

    # Find all potential Concat nodes (axis 1 or 3, 4 inputs)
    for node in graph.node:
        if node.op_type != "Concat" or len(node.input) != 4:
            continue

        axis = next((attr.i for attr in node.attribute if attr.name == "axis"), None)
        if axis not in (1, 3):
            continue

        # Trace back from Concat to check if it's a Focus pattern
        all_slice_nodes = []
        pattern_roots = set()
        quadrants = []

        def collect_slice_chain(tensor_name):
            if tensor_name not in output_to_node:
                return tensor_name  # Graph input or initializer

            prod_node = output_to_node[tensor_name]
            if prod_node.op_type != "Slice":
                return tensor_name  # Preprocessing node or other

            if prod_node not in all_slice_nodes:  # noqa: B023
                all_slice_nodes.append(prod_node)  # noqa: B023
            return collect_slice_chain(prod_node.input[0])

        valid_pattern = True
        for inp in node.input:
            if inp in output_to_node and output_to_node[inp].op_type == "Slice":
                q = detect_slice_quadrant(graph, output_to_node[inp], output_to_node)
                if q is None:
                    # Could not determine quadrant, assume pattern is invalid or non-standard
                    valid_pattern = False
                    break
                quadrants.append(q)

                root = collect_slice_chain(inp)
                pattern_roots.add(root)
            else:
                valid_pattern = False
                break

        if not valid_pattern or len(pattern_roots) != 1:
            continue

        # Ensure we have all 4 unique quadrants (TL, BL, TR, BR)
        if set(quadrants) != {0, 1, 2, 3}:
            continue

        root_name = next(iter(pattern_roots))

        # Verify a Conv follows the Concat (to absorb the space-to-depth)
        concat_output = node.output[0]
        consumers = [n for n in graph.node if concat_output in n.input]

        if not consumers:
            LOG.info(
                f"Detected Focus pattern at {node.name}, but no consumer found. "
                "Skipping replacement."
            )
            continue

        conv_node = next((n for n in consumers if n.op_type == "Conv"), None)
        if not conv_node:
            LOG.info(
                f"Detected Focus pattern at {node.name}, but consumer is "
                f"{consumers[0].op_type}, not Conv. Skipping."
            )
            continue

        return FocusConvPattern(
            slice_nodes=all_slice_nodes,
            concat_node=node,
            conv_node=conv_node,
            pattern_input_name=root_name,
            concat_order=quadrants,
        )
    return None


def transform_focus_conv_weights(
    old_weight: np.ndarray, in_channels: int, concat_order: list[int]
) -> np.ndarray:
    """Rearrange Conv weight tensor to account for space-to-depth.

    Uses detected concat_order to map weights correctly.
    """
    out_ch, _, kH, kW = old_weight.shape
    c1 = in_channels

    # New kernel size will be doubled (typically 3x3 -> 6x6)
    new_weight = np.zeros((out_ch, c1, kH * 2, kW * 2), dtype=old_weight.dtype)

    # concat_order is a list of 4 ints, where index i corresponds to the i-th input of Concat
    # and the value is the quadrant index (0:TL, 1:BL, 2:TR, 3:BR).

    # We iterate through the 4 chunks of the old weight (one per Concat input)
    for i, quadrant in enumerate(concat_order):
        # Determine target grid position based on quadrant
        # 0: TL (r=0, c=0), 1: BL (r=1, c=0), 2: TR (r=0, c=1), 3: BR (r=1, c=1)
        r_off = 1 if quadrant in (1, 3) else 0
        c_off = 1 if quadrant in (2, 3) else 0

        # Extract the corresponding chunk from old_weight
        # old_weight has shape [out, in*4, kH, kW]
        # The chunks are concatenated along dim 1.
        w_slice = old_weight[:, i * c1 : (i + 1) * c1, :, :]

        # Place into the new strided locations
        new_weight[:, :, r_off::2, c_off::2] = w_slice

    return new_weight


def replace_focus_layer(model: onnx.ModelProto) -> onnx.ModelProto:
    """Transform Focus+Conv into a single equivalent Conv."""
    pattern = detect_focus_conv_pattern(model)
    if pattern is None:
        LOG.debug("Focus+Conv pattern not found or already replaced.")
        return model

    LOG.info(
        f"Replacing Focus (root: {pattern.pattern_input_name}) and Conv ({pattern.conv_node.name})"
    )
    LOG.info(f"Detected Quadrant Order: {pattern.concat_order} (0:TL, 1:BL, 2:TR, 3:BR)")

    graph = model.graph
    conv_node = pattern.conv_node

    # Extract original Conv attributes
    attrs = {a.name: a for a in conv_node.attribute}
    old_ks = list(attrs['kernel_shape'].ints) if 'kernel_shape' in attrs else [3, 3]
    old_strides = list(attrs['strides'].ints) if 'strides' in attrs else [1, 1]
    old_pads = list(attrs['pads'].ints) if 'pads' in attrs else [0, 0, 0, 0]
    old_dilations = list(attrs['dilations'].ints) if 'dilations' in attrs else [1, 1]
    old_group = attrs['group'].i if 'group' in attrs else 1

    # Get weight tensor
    weight_name = conv_node.input[1]
    weight_init = next((i for i in graph.initializer if i.name == weight_name), None)
    if weight_init is None:
        LOG.error(f"Could not find weight initializer {weight_name}")
        return model

    old_weight = numpy_helper.to_array(weight_init)
    in_channels = old_weight.shape[1] // 4

    # Transform weights using detected quadrant order
    new_weight = transform_focus_conv_weights(old_weight, in_channels, pattern.concat_order)

    # New Conv attributes
    new_ks = [k * 2 for k in old_ks]
    new_strides = [s * 2 for s in old_strides]
    new_pads = [p * 2 for p in old_pads]

    # Create new initializer
    new_weight_name = f"{weight_name}_fused"
    new_weight_init = numpy_helper.from_array(new_weight, name=new_weight_name)

    # Build new Conv node
    new_conv_inputs = [pattern.pattern_input_name, new_weight_name]
    if len(conv_node.input) > 2:  # Has bias
        new_conv_inputs.append(conv_node.input[2])

    new_conv_node = helper.make_node(
        "Conv",
        inputs=new_conv_inputs,
        outputs=conv_node.output,
        name=f"{conv_node.name}_fused",
        kernel_shape=new_ks,
        strides=new_strides,
        pads=new_pads,
        dilations=old_dilations,
        group=old_group,
    )

    # Reconstruct graph
    nodes_to_remove = {id(n) for n in pattern.slice_nodes}
    nodes_to_remove.add(id(pattern.concat_node))
    nodes_to_remove.add(id(pattern.conv_node))

    new_nodes = []
    for node in graph.node:
        if id(node) == id(pattern.conv_node):
            new_nodes.append(new_conv_node)
        elif id(node) not in nodes_to_remove:
            new_nodes.append(node)

    # We ONLY remove the old weight initializer.
    # Slice parameters are often shared and should be preserved.
    new_initializers = [new_weight_init] + [i for i in graph.initializer if i.name != weight_name]

    new_graph = helper.make_graph(
        new_nodes, graph.name, graph.input, graph.output, new_initializers
    )

    # Preserve value_info
    new_graph.value_info.extend(graph.value_info)

    new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
    new_model.ir_version = model.ir_version

    try:
        onnx.checker.check_model(new_model)
        LOG.info("Success: Optimized model validated.")
    except Exception as e:
        LOG.warning(f"Validation Warning: {e}")

    return new_model

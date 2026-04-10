# Copyright Axelera AI, 2026
# ONNX graph optimization transformations

from __future__ import annotations

from dataclasses import dataclass
import math
import typing

import numpy as np

try:
    import onnx
    from onnx import helper, numpy_helper

    _HAVE_ONNX = True
except ImportError:
    _HAVE_ONNX = False
    if typing.TYPE_CHECKING:
        import onnx
        from onnx import helper, numpy_helper  # type: ignore

from axelera.app import logging_utils

LOG = logging_utils.getLogger(__name__)


# Threshold for Gemm->Conv conversion: only convert if reduction blocks exceed this
# MVM unit is 4x8x8 = 256, so we convert when workload exceeds capacity
GEMM_TO_CONV_REDUCTION_THRESHOLD = 256


@dataclass
class FocusConvPattern:
    """Container for detected Focus+Conv pattern nodes."""

    slice_nodes: typing.List[onnx.NodeProto]
    concat_node: onnx.NodeProto
    conv_node: onnx.NodeProto
    pattern_input_name: str
    concat_order: typing.List[int]  # Maps concat index to quadrant index (0:TL, 1:BL, 2:TR, 3:BR)


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


def detect_slice_quadrant(graph, slice_node, output_to_node):
    """
    Determine which quadrant (TL, BL, TR, BR) a Slice chain represents.
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

        # If axes is missing, it defaults to [0, 1, ...], but standard ONNX usually provides it or it's implicitly all axes.
        # For robustness, we mostly care about explicit slicing on 2 and 3.

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


def detect_focus_conv_pattern(model: onnx.ModelProto) -> typing.Optional[FocusConvPattern]:  # type: ignore[name-defined]
    """
    Detect Focus layer pattern followed by Conv in an ONNX model.
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

            if prod_node not in all_slice_nodes:
                all_slice_nodes.append(prod_node)
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

        root_name = list(pattern_roots)[0]

        # Verify a Conv follows the Concat (to absorb the space-to-depth)
        concat_output = node.output[0]
        consumers = [n for n in graph.node if concat_output in n.input]

        if not consumers:
            LOG.info(
                f"Detected Focus pattern at {node.name}, but no consumer found. Skipping replacement."
            )
            continue

        conv_node = next((n for n in consumers if n.op_type == "Conv"), None)
        if not conv_node:
            LOG.info(
                f"Detected Focus pattern at {node.name}, but consumer is {consumers[0].op_type}, not Conv. Skipping."
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
    old_weight: np.ndarray, in_channels: int, concat_order: typing.List[int]
) -> np.ndarray:
    """
    Rearrange Conv weight tensor to account for space-to-depth.
    Uses detected concat_order to map weights correctly.
    """
    out_ch, _, kH, kW = old_weight.shape
    c1 = in_channels

    # New kernel size will be doubled (typically 3x3 -> 6x6)
    new_weight = np.zeros((out_ch, c1, kH * 2, kW * 2), dtype=old_weight.dtype)

    # concat_order is a list of 4 ints, where index i corresponds to the i-th input of Concat
    # and the value is the quadrant index (0:TL, 1:BL, 2:TR, 3:BR).

    # We iterate through the 4 chunks of the old weight (which correspond to the 4 inputs of Concat)
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


def replace_focus_layer(model: onnx.ModelProto) -> onnx.ModelProto:  # type: ignore[name-defined]
    """
    Transforms Focus+Conv into a single equivalent Conv.
    """
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
    nodes_to_remove = set(id(n) for n in pattern.slice_nodes)
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


@dataclass
class GemmToConvPattern:
    """Container for detected Gemm/MatMul pattern that can be converted to Conv."""

    reshape_node: onnx.NodeProto
    fc_node: onnx.NodeProto  # Gemm or MatMul node
    bias_node: typing.Optional[onnx.NodeProto]  # Add node for MatMul bias (if any)
    feature_map_name: str  # Input to Reshape (4D tensor)
    spatial_h: int
    spatial_w: int
    in_channels: int
    out_features: int
    is_matmul: bool  # True if MatMul, False if Gemm


def _infer_spatial_dims_from_shape_inference(
    model: onnx.ModelProto,  # type: ignore[name-defined]
    tensor_name: str,
) -> typing.Optional[typing.Tuple[int, int, int]]:
    """
    Try to infer spatial dimensions using ONNX shape inference.

    Returns:
        Tuple of (channels, height, width) or None if inference fails.
    """
    try:
        from onnx import shape_inference

        inferred = shape_inference.infer_shapes(model)
        for vi in inferred.graph.value_info:
            if vi.name == tensor_name:
                dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
                if len(dims) == 4 and all(d > 0 for d in dims[1:]):
                    return (dims[1], dims[2], dims[3])  # C, H, W
        # Also check graph inputs
        for inp in inferred.graph.input:
            if inp.name == tensor_name:
                dims = [d.dim_value for d in inp.type.tensor_type.shape.dim]
                if len(dims) == 4 and all(d > 0 for d in dims[1:]):
                    return (dims[1], dims[2], dims[3])
    except Exception as e:
        LOG.debug(f"Shape inference failed for {tensor_name}: {e}")
    return None


def _infer_spatial_dims_from_in_features(
    in_features: int,
) -> typing.Optional[typing.Tuple[int, int, int]]:
    """
    Infer spatial dimensions from flattened feature count.

    Common patterns in face recognition and classification models:
    - 7x7xC (ResNet-style after 32x downsampling from 224)
    - 14x14xC (after 16x downsampling)
    - 4x4xC, 5x5xC, 6x6xC (various architectures)

    Returns:
        Tuple of (channels, height, width) or None if cannot infer.
    """
    # Spatial candidates ordered by preference (larger spatial = better tiling)
    spatial_candidates = [
        (14, 14),
        (7, 7),
        (8, 8),
        (6, 6),
        (5, 5),
        (4, 4),
        (3, 3),
        (2, 2),
        (1, 1),
    ]

    # Common channel counts in neural networks
    valid_channels = {32, 64, 128, 256, 512, 1024, 2048, 4096}

    found_spatial = None
    found_channels = None

    for h, w in spatial_candidates:
        spatial_size = h * w
        if in_features % spatial_size == 0:
            c = in_features // spatial_size
            # Check if channel count is reasonable
            is_power_of_2 = c > 0 and (c & (c - 1) == 0)
            is_common = c in valid_channels
            if is_power_of_2 or is_common:
                # Prefer larger spatial dimensions for tiling benefit
                if found_spatial is None or spatial_size > found_spatial[0] * found_spatial[1]:
                    found_spatial = (h, w)
                    found_channels = c

    if found_spatial is not None:
        return (found_channels, found_spatial[0], found_spatial[1])
    return None


def _get_fc_weight_and_dims(
    graph, fc_node: onnx.NodeProto
) -> typing.Optional[typing.Tuple[np.ndarray, int, int]]:
    """
    Get weight tensor and determine in/out features for Gemm or MatMul.

    Returns:
        Tuple of (weight_array, in_features, out_features) or None if failed.
    """
    is_gemm = fc_node.op_type == 'Gemm'
    weight_name = fc_node.input[1]

    weight = get_constant_value(graph, weight_name)
    if weight is None:
        return None

    if is_gemm:
        trans_b = 0
        trans_b_found = False
        for attr in fc_node.attribute:
            if attr.name == 'transB':
                trans_b = attr.i
                trans_b_found = True
                break

        if trans_b:
            out_features, in_features = weight.shape
        else:
            in_features, out_features = weight.shape
            # Heuristic ONLY if transB attribute is missing (ambiguous case)
            # Don't apply heuristic if transB was explicitly set to 0
            if not trans_b_found and weight.shape[0] > weight.shape[1]:
                # Likely transposed based on shape
                out_features, in_features = weight.shape
    else:
        # MatMul: Y = X @ W, so W is (in_features, out_features)
        in_features, out_features = weight.shape

    return weight, in_features, out_features


def detect_gemm_to_conv_pattern(
    model: onnx.ModelProto,  # type: ignore[name-defined]
) -> typing.Optional[GemmToConvPattern]:
    """
    Detect Gemm or MatMul layer preceded by Reshape that flattens a spatial feature map.

    Patterns detected:
    1. [N, C, H, W] -> Reshape -> [N, C*H*W] -> Gemm -> [N, out_features]
    2. [N, C, H, W] -> Reshape -> [N, C*H*W] -> MatMul -> [N, out_features]
    3. [N, C, H, W] -> Reshape -> [N, C*H*W] -> MatMul -> Add(bias) -> [N, out_features]

    This pattern is common in face recognition models (ArcFace, etc.) where
    the final FC layer operates on flattened features instead of using GAP.

    Converting this to Conv allows the compiler to tile the computation spatially.
    """
    graph = model.graph
    output_to_node = {out: node for node in graph.node for out in node.output}

    for node in graph.node:
        # Support both Gemm and MatMul
        if node.op_type not in ('Gemm', 'MatMul'):
            continue

        fc_node = node
        fc_input = fc_node.input[0]

        # Check if input comes from Reshape
        if fc_input not in output_to_node:
            continue

        reshape_node = output_to_node[fc_input]
        if reshape_node.op_type != 'Reshape':
            continue

        # Get weight and determine dimensions
        result = _get_fc_weight_and_dims(graph, fc_node)
        if result is None:
            continue

        weight, in_features, out_features = result

        # First try shape inference for accurate dimensions
        feature_map_name = reshape_node.input[0]
        spatial_info = _infer_spatial_dims_from_shape_inference(model, feature_map_name)

        if spatial_info is not None:
            in_channels, spatial_h, spatial_w = spatial_info
            # Verify dimensions match
            if in_channels * spatial_h * spatial_w != in_features:
                LOG.debug(
                    f"Shape inference mismatch: {in_channels}*{spatial_h}*{spatial_w} != {in_features}"
                )
                spatial_info = None

        # Fall back to heuristic inference
        if spatial_info is None:
            spatial_info = _infer_spatial_dims_from_in_features(in_features)

        if spatial_info is None:
            LOG.debug(
                f"FC node {fc_node.name}: could not infer spatial dims from in_features={in_features}"
            )
            continue

        in_channels, spatial_h, spatial_w = spatial_info

        # Check if this conversion would help (reduction blocks > threshold)
        reduction_blocks = in_features // 64  # Assuming 64-element blocks
        if reduction_blocks <= GEMM_TO_CONV_REDUCTION_THRESHOLD:
            LOG.debug(
                f"FC node {fc_node.name}: reduction_blocks={reduction_blocks} <= threshold, skipping"
            )
            continue

        # Check if the resulting Conv would be tileable by the compiler
        # The compiler tiles Convs when reduction_blocks is very large (max_parallel_blocks == 0),
        # but tiling fails if input_channels / num_tiles < PWORD_SIZE (64).
        # This happens when Gemm->Conv creates large-kernel Convs (e.g., 14x14) with moderate
        # channel counts (e.g., 512), resulting in tile_size = 0.
        #
        # Instead of skipping the optimization entirely, we can split the Gemm into multiple
        # smaller Convs that ARE tileable, then sum their outputs.
        PWORD_SIZE = 64
        N_REDUC_FULL_IMG_UTIL = (
            2048  # From compiler constants (MVM_N_ROWS * MVM_N_WEIGHTSETS * PWORD_SIZE)
        )
        opt_tiles_num = math.ceil(in_features / N_REDUC_FULL_IMG_UTIL)

        # Check if tiling would produce tile_size = 0
        # The compiler calculates: tile_size = ((in_channels // opt_tiles_num) // PWORD_SIZE) * PWORD_SIZE
        channels_per_tile = in_channels // opt_tiles_num
        if channels_per_tile < PWORD_SIZE:
            # Single Conv would be untileable, but we can split into multiple Convs
            # Calculate how many channel groups we need (each with >= PWORD_SIZE channels)
            num_channel_groups = max(1, (in_channels + PWORD_SIZE - 1) // PWORD_SIZE)

            LOG.info(
                f"FC node {fc_node.name}: Single Conv would be untileable "
                f"(in_channels={in_channels}, channels_per_tile={channels_per_tile} < {PWORD_SIZE}). "
                f"Will use multi-Conv replacement with {num_channel_groups} groups."
            )

        # For MatMul, check for Add node (bias)
        bias_node = None
        is_matmul = fc_node.op_type == 'MatMul'
        if is_matmul:
            # Look for Add node that uses MatMul output
            matmul_output = fc_node.output[0]
            for n in graph.node:
                if n.op_type == 'Add' and matmul_output in n.input:
                    # Check if other input is a bias (1D tensor)
                    other_input = n.input[0] if n.input[1] == matmul_output else n.input[1]
                    # Use get_constant_value() to check for bias (handles both initializers and Constant nodes)
                    bias_arr = get_constant_value(graph, other_input)
                    if (
                        bias_arr is not None
                        and len(bias_arr.shape) == 1
                        and bias_arr.shape[0] == out_features
                    ):
                        bias_node = n
                        break

        LOG.info(
            f"Detected {'MatMul' if is_matmul else 'Gemm'}->Conv candidate: {fc_node.name or 'unnamed'}, "
            f"in_features={in_features} ({in_channels}x{spatial_h}x{spatial_w}), "
            f"out_features={out_features}, reduction_blocks={reduction_blocks}"
        )

        return GemmToConvPattern(
            reshape_node=reshape_node,
            fc_node=fc_node,
            bias_node=bias_node,
            feature_map_name=feature_map_name,
            spatial_h=spatial_h,
            spatial_w=spatial_w,
            in_channels=in_channels,
            out_features=out_features,
            is_matmul=is_matmul,
        )

    return None


def replace_gemm_with_conv(model: onnx.ModelProto) -> onnx.ModelProto:  # type: ignore[name-defined]
    """
    Replace Gemm/MatMul (fully connected) layer with equivalent Conv layer.

    WHY THIS OPTIMIZATION EXISTS:
    ------------------------------
    The Axelera AIPU's MVM (Matrix-Vector Multiply) unit has limited capacity:
    - 4 weight sets × 8 rows × 8 cols × 64 elements = 256 parallel blocks
    - Large FC layers (e.g., 100352 → 2048) exceed this capacity

    By converting Gemm → Conv with spatial dimensions, we enable:
    - Spatial tiling by the compiler (process image patches sequentially)
    - Better memory locality and reduced intermediate buffer sizes
    - Hardware can process the same computation in smaller chunks

    TEMPORAL WORKAROUND (MULTI-CONV FALLBACK):
    -------------------------------------------
    This is a FRAMEWORK-LEVEL workaround for a COMPILER limitation. Ideally:
    - The compiler should detect untileable Convs and split them internally
    - This graph rewriting should happen in the compiler's optimization passes

    We implement it here because:
    1. Immediate fix needed for customer models
    2. Compiler modification requires extensive testing across all models
    3. Framework can iterate faster for customer-facing issues

    FUTURE: Once compiler implements proper multi-Conv tiling, this code
    should be removed or made conditional (only activate if compiler lacks support).

    MULTI-CONV FALLBACK STRATEGY:
    ------------------------------
    When a single Conv would be untileable (channels_per_tile < PWORD_SIZE=64):
    - Large kernels (e.g., 14×14) with moderate channels (e.g., 512) create
      untileable patterns where compiler needs 49 tiles but only 512 channels
    - Split input along channel dimension into groups (each ≥64 channels)
    - Create separate Conv for each group with corresponding weight slice
    - Sum all Conv outputs and flatten to match original shape

    This maintains optimization benefits while avoiding compiler crash.

    SUPPORTED PATTERNS:
    -------------------
    1. Reshape(NxCxHxW -> Nx(C*H*W)) -> Gemm((C*H*W) -> out)
    2. Reshape(NxCxHxW -> Nx(C*H*W)) -> MatMul -> Add(bias)

    OUTPUT:
    -------
    Single Conv path:  Conv(NxCxHxW, kernel=HxW) -> Flatten -> (Nxout)
    Multi-Conv path:   Split -> [Conv1, Conv2, ...] -> Sum -> Flatten -> (Nxout)

    The computation is mathematically equivalent but Conv operations can be
    tiled spatially by the compiler for efficient hardware execution.
    """
    pattern = detect_gemm_to_conv_pattern(model)
    if pattern is None:
        LOG.debug("Gemm/MatMul->Conv pattern not found or not beneficial.")
        return model

    # Check if we need multi-Conv tiling due to compiler limitations
    PWORD_SIZE = 64
    N_REDUC_FULL_IMG_UTIL = 2048
    in_features = pattern.in_channels * pattern.spatial_h * pattern.spatial_w
    opt_tiles_num = math.ceil(in_features / N_REDUC_FULL_IMG_UTIL)
    channels_per_tile = pattern.in_channels // opt_tiles_num

    use_multi_conv = channels_per_tile < PWORD_SIZE
    num_channel_groups = (
        (pattern.in_channels + PWORD_SIZE - 1) // PWORD_SIZE if use_multi_conv else 1
    )

    fc_type = "MatMul" if pattern.is_matmul else "Gemm"
    if use_multi_conv:
        LOG.info(
            f"Replacing {fc_type} with Multi-Conv: spatial={pattern.spatial_h}x{pattern.spatial_w}, "
            f"in_ch={pattern.in_channels}, out={pattern.out_features}, "
            f"num_groups={num_channel_groups} (channels_per_group~={pattern.in_channels // num_channel_groups})"
        )
    else:
        LOG.info(
            f"Replacing {fc_type} with Conv: spatial={pattern.spatial_h}x{pattern.spatial_w}, "
            f"in_ch={pattern.in_channels}, out={pattern.out_features}"
        )

    graph = model.graph
    fc_node = pattern.fc_node
    reshape_node = pattern.reshape_node

    # Get opset version for ONNX compatibility
    opset_version = model.opset_import[0].version if model.opset_import else 11

    # Get FC weight and bias
    weight_name = fc_node.input[1]

    # For Gemm, bias is 3rd input; for MatMul, it comes from Add node
    if pattern.is_matmul:
        bias_name = None
        if pattern.bias_node is not None:
            # Get bias from the Add node
            bias_input = (
                pattern.bias_node.input[0]
                if pattern.bias_node.input[1] == fc_node.output[0]
                else pattern.bias_node.input[1]
            )
            bias_name = bias_input
    else:
        bias_name = fc_node.input[2] if len(fc_node.input) > 2 else None

    # Use get_constant_value() to retrieve weight and bias (handles both initializers and Constant nodes)
    old_weight = get_constant_value(graph, weight_name)
    if old_weight is None:
        LOG.error(f"Could not find weight constant {weight_name}")
        return model

    # Handle weight transformation based on node type
    if pattern.is_matmul:
        # MatMul: Y = X @ W, weight is (in_features, out_features)
        # Conv expects (out_ch, in_ch, kH, kW), so transpose first
        conv_weight = old_weight.T.reshape(
            pattern.out_features, pattern.in_channels, pattern.spatial_h, pattern.spatial_w
        )
    else:
        # Gemm: check transB attribute
        trans_b = 0
        for attr in fc_node.attribute:
            if attr.name == 'transB':
                trans_b = attr.i
                break

        if trans_b or old_weight.shape[0] == pattern.out_features:
            # Weight is (out_features, in_features) - transposed
            conv_weight = old_weight.reshape(
                pattern.out_features, pattern.in_channels, pattern.spatial_h, pattern.spatial_w
            )
        else:
            # Weight is (in_features, out_features) - need to transpose first
            conv_weight = old_weight.T.reshape(
                pattern.out_features, pattern.in_channels, pattern.spatial_h, pattern.spatial_w
            )

    # Determine final output name (from Gemm, or from Add if MatMul+bias)
    if pattern.is_matmul and pattern.bias_node is not None:
        final_output_name = pattern.bias_node.output[0]
    else:
        final_output_name = fc_node.output[0]

    # Retrieve bias if present
    old_bias = None
    if bias_name:
        old_bias = get_constant_value(graph, bias_name)

    # Branch: Multi-Conv or Single-Conv
    if use_multi_conv:
        # Multi-Conv path: Split into channel groups
        # Calculate channel splits (try to make them equal-sized)
        channels_per_group = pattern.in_channels // num_channel_groups
        channel_splits = [channels_per_group] * num_channel_groups
        # Handle remainder
        remainder = pattern.in_channels - (channels_per_group * num_channel_groups)
        for i in range(remainder):
            channel_splits[i] += 1

        # Split weights along input channel dimension (axis=1 for Conv weight)
        conv_weight_splits = []
        start_ch = 0
        for ch_count in channel_splits:
            weight_slice = conv_weight[:, start_ch : start_ch + ch_count, :, :]
            conv_weight_splits.append(weight_slice)
            start_ch += ch_count

        # Create Conv nodes for each channel group
        conv_output_names = []
        new_initializers = []
        new_nodes = []

        # Split input along channel dimension
        split_output_names = []
        for i in range(num_channel_groups):
            split_output_names.append(f"{pattern.feature_map_name}_split_{i}")

        # ONNX Split operator: split format varies by opset
        # - Opset < 13: split is an attribute (list of ints)
        # - Opset >= 13: split is an input tensor
        if opset_version >= 13:
            # Create split as input tensor for opset >= 13
            split_tensor_name = f"{final_output_name}_split_sizes"
            split_tensor = numpy_helper.from_array(
                np.array(channel_splits, dtype=np.int64), split_tensor_name
            )
            new_initializers.append(split_tensor)
            split_node = helper.make_node(
                'Split',
                inputs=[pattern.feature_map_name, split_tensor_name],
                outputs=split_output_names,
                name=f"{final_output_name}_split",
                axis=1,
            )
        else:
            # Use split as attribute for opset < 13
            split_node = helper.make_node(
                'Split',
                inputs=[pattern.feature_map_name],
                outputs=split_output_names,
                name=f"{final_output_name}_split",
                axis=1,
                split=channel_splits,
            )
        new_nodes.append(split_node)

        # Create Conv for each split
        for i, (weight_slice, input_split_name) in enumerate(
            zip(conv_weight_splits, split_output_names)
        ):
            # Create weight initializer for this group
            weight_slice_name = f"{weight_name}_conv_group_{i}"
            weight_slice_init = numpy_helper.from_array(
                weight_slice.astype(np.float32), weight_slice_name
            )
            new_initializers.append(weight_slice_init)

            # Create Conv node (output will be [N, out_features, 1, 1])
            conv_out_name = f"{final_output_name}_conv_{i}"
            conv_inputs = [input_split_name, weight_slice_name]

            # Only first Conv gets bias (if any), others add to zero
            if old_bias is not None and i == 0:
                bias_slice_name = f"{bias_name}_conv"
                bias_slice_init = numpy_helper.from_array(
                    old_bias.astype(np.float32), bias_slice_name
                )
                new_initializers.append(bias_slice_init)
                conv_inputs.append(bias_slice_name)

            conv_node = helper.make_node(
                'Conv',
                inputs=conv_inputs,
                outputs=[conv_out_name],
                name=f"{fc_node.name or fc_type.lower()}_to_conv_group_{i}",
                kernel_shape=[pattern.spatial_h, pattern.spatial_w],
                strides=[1, 1],
                pads=[0, 0, 0, 0],
            )
            new_nodes.append(conv_node)
            conv_output_names.append(conv_out_name)

        # Sum all Conv outputs
        sum_output_name = f"{final_output_name}_sum"
        sum_node = helper.make_node(
            'Sum',
            inputs=conv_output_names,
            outputs=[sum_output_name],
            name=f"{final_output_name}_sum_multi_conv",
        )
        new_nodes.append(sum_node)

        # Flatten to match original output shape
        flatten_node = helper.make_node(
            'Flatten',
            inputs=[sum_output_name],
            outputs=[final_output_name],
            name=f"{fc_node.name or fc_type.lower()}_flatten",
            axis=1,
        )
        new_nodes.append(flatten_node)

    else:
        # Single-Conv path (existing logic)
        conv_weight_name = f"{weight_name}_conv"
        conv_weight_init = numpy_helper.from_array(
            conv_weight.astype(np.float32), conv_weight_name
        )

        conv_bias_name = None
        conv_bias_init = None
        if old_bias is not None:
            conv_bias_name = f"{bias_name}_conv"
            conv_bias_init = numpy_helper.from_array(old_bias.astype(np.float32), conv_bias_name)

        # Create Conv node
        conv_output_name = f"{final_output_name}_conv"
        conv_inputs = [pattern.feature_map_name, conv_weight_name]
        if conv_bias_name:
            conv_inputs.append(conv_bias_name)

        conv_node = helper.make_node(
            'Conv',
            inputs=conv_inputs,
            outputs=[conv_output_name],
            name=f"{fc_node.name or fc_type.lower()}_to_conv",
            kernel_shape=[pattern.spatial_h, pattern.spatial_w],
            strides=[1, 1],
            pads=[0, 0, 0, 0],
        )

        # Create Flatten node to convert (N, out, 1, 1) -> (N, out)
        # Using Flatten instead of Squeeze for better compatibility
        flatten_node = helper.make_node(
            'Flatten',
            inputs=[conv_output_name],
            outputs=[final_output_name],  # Use original output name
            name=f"{fc_node.name or fc_type.lower()}_flatten",
            axis=1,
        )

        new_initializers = [conv_weight_init]
        if conv_bias_init:
            new_initializers.append(conv_bias_init)
        new_nodes = [conv_node, flatten_node]

    # Find nodes to remove: Reshape, FC node, and optional Add (for MatMul)
    nodes_to_remove = {id(reshape_node), id(fc_node)}
    if pattern.is_matmul and pattern.bias_node is not None:
        nodes_to_remove.add(id(pattern.bias_node))

    # Find the shape input to Reshape and its producer nodes
    if len(reshape_node.input) > 1:
        shape_input = reshape_node.input[1]
        output_to_node = {out: node for node in graph.node for out in node.output}

        def mark_for_removal(tensor_name, depth=0):
            if depth > 5:  # Prevent infinite recursion
                return
            if tensor_name in output_to_node:
                node = output_to_node[tensor_name]
                # Only remove shape-computation nodes (Concat, Shape, Unsqueeze, Constant)
                if node.op_type in ['Concat', 'Shape', 'Unsqueeze', 'Constant', 'Gather']:
                    nodes_to_remove.add(id(node))
                    for inp in node.input:
                        mark_for_removal(inp, depth + 1)

        mark_for_removal(shape_input)

    # Build new node list (insert replacement nodes at FC node's position)
    replacement_nodes = new_nodes  # Nodes created in multi-Conv or single-Conv branch
    final_nodes = []
    fc_replaced = False
    for node in graph.node:
        if id(node) in nodes_to_remove:
            if id(node) == id(fc_node) and not fc_replaced:
                # Insert replacement nodes at FC node's position
                final_nodes.extend(replacement_nodes)
                fc_replaced = True
        else:
            final_nodes.append(node)

    # Build new initializer list
    init_names_to_remove = {weight_name}
    if bias_name:
        init_names_to_remove.add(bias_name)

    final_initializers = [
        init for init in graph.initializer if init.name not in init_names_to_remove
    ]
    final_initializers.extend(
        new_initializers
    )  # Add initializers from multi-Conv or single-Conv branch

    # Create new graph
    new_graph = helper.make_graph(
        final_nodes, graph.name, graph.input, graph.output, final_initializers
    )

    # Preserve value_info
    new_graph.value_info.extend(graph.value_info)

    new_model = helper.make_model(new_graph, opset_imports=model.opset_import)
    new_model.ir_version = model.ir_version

    try:
        onnx.checker.check_model(new_model)
        LOG.info("Success: Gemm->Conv optimized model validated.")
    except Exception as e:
        LOG.warning(f"Validation Warning after Gemm->Conv: {e}")

    return new_model


def validate_gemm_conv_equivalence(
    original_model: onnx.ModelProto,  # type: ignore[name-defined]
    converted_model: onnx.ModelProto,  # type: ignore[name-defined]
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> bool:
    """
    Validate that original and converted models produce equivalent outputs.

    Note: Uses relaxed tolerance (1e-4) to account for floating point accumulation
    in the multi-Conv path, which sums outputs from multiple Conv operations.
    Single-Conv path typically has errors <1e-5, but multi-Conv can reach ~1e-5
    due to FP rounding in 8+ separate operations being summed.

    Args:
        original_model: Original ONNX model with Gemm
        converted_model: Converted ONNX model with Conv
        rtol: Relative tolerance for comparison (default 1e-4)
        atol: Absolute tolerance for comparison (default 1e-4)

    Returns:
        True if outputs are equivalent within tolerance
    """
    try:
        import onnxruntime as ort
    except ImportError:
        LOG.warning("onnxruntime not available, skipping equivalence validation")
        return True

    # Create sessions
    original_session = ort.InferenceSession(
        original_model.SerializeToString(), providers=['CPUExecutionProvider']
    )
    converted_session = ort.InferenceSession(
        converted_model.SerializeToString(), providers=['CPUExecutionProvider']
    )

    # Get input info
    input_info = original_session.get_inputs()[0]
    input_name = input_info.name
    input_shape = input_info.shape

    # Replace dynamic dimensions with concrete values
    concrete_shape = []
    for dim in input_shape:
        if isinstance(dim, int) and dim > 0:
            concrete_shape.append(dim)
        else:
            concrete_shape.append(1)  # Use batch size 1 for testing

    # Generate random test input
    test_input = np.random.randn(*concrete_shape).astype(np.float32)

    # Run both models
    original_outputs = original_session.run(None, {input_name: test_input})
    converted_outputs = converted_session.run(None, {input_name: test_input})

    # Compare outputs
    for i, (orig, conv) in enumerate(zip(original_outputs, converted_outputs)):
        if not np.allclose(orig, conv, rtol=rtol, atol=atol):
            max_diff = np.max(np.abs(orig - conv))
            LOG.error(f"Output {i} mismatch: max_diff={max_diff}, rtol={rtol}, atol={atol}")
            return False

    LOG.info("Gemm->Conv equivalence validation passed")
    return True


@dataclass
class Yolo26SegProtoPattern:
    """Container for detected YOLO26 seg proto modification sites.

    YOLO26 segmentation models have a "proto" section that generates segmentation
    mask prototypes. This section contains Add operations that create problematic
    dependency chains for the compiler:

    1. External connections: Mul/Add operations from outside proto feeding into proto
    2. Chained Add operations: Direct Add→Add connections within proto

    This pattern holds the locations where MaxPool identity operations should be
    inserted to break these dependency chains while preserving correctness.
    """

    external_connections: typing.List[typing.Dict[str, typing.Any]]
    proto_chains: typing.List[typing.Dict[str, typing.Any]]


def insert_maxpool_identity(
    connection_name: str, node_counter: int
) -> typing.Tuple[onnx.NodeProto, str]:  # type: ignore[name-defined]
    """
    Create a MaxPool identity operation (1x1 kernel, 1x1 stride).

    MaxPool with 1x1 kernel and 1x1 stride is mathematically an identity operation
    (output equals input), but it breaks direct dependency chains for the compiler,
    enabling better optimization of the graph.

    Args:
        connection_name: Name of the tensor connection to insert MaxPool on
        node_counter: Counter for unique node naming

    Returns:
        Tuple of (maxpool_node, new_output_name)
    """
    new_output_name = f"{connection_name}_maxpool_identity_{node_counter}"

    maxpool_node = helper.make_node(
        'MaxPool',
        inputs=[connection_name],
        outputs=[new_output_name],
        name=f'identity_maxpool_{node_counter}',
        kernel_shape=[1, 1],
        strides=[1, 1],
        pads=[0, 0, 0, 0],
    )

    return maxpool_node, new_output_name


def detect_yolo26_seg_proto_pattern(
    model: onnx.ModelProto,  # type: ignore[name-defined]
) -> typing.Optional[Yolo26SegProtoPattern]:
    """
    Detect problematic proto patterns in YOLO26 segmentation models.

    Searches for the "proto" section (segmentation mask prototypes) and identifies:
    1. External connections: Add/Mul operations from outside proto feeding into proto Add nodes
    2. Proto chains: Direct Add→Add connections within the proto section

    These patterns cause compilation failures on Axelera hardware because they create
    dependency chains the compiler cannot optimize. Inserting MaxPool identity operations
    breaks these chains while preserving model correctness.

    Args:
        model: ONNX model to analyze

    Returns:
        Yolo26SegProtoPattern if problematic patterns found, None otherwise
    """
    graph = model.graph

    # Build lookup tables
    node_by_output = {}
    for node in graph.node:
        for out in node.output:
            node_by_output[out] = node

    # Find the proto section: nodes with "proto" in their name
    LOG.debug("Searching for proto section...")
    proto_nodes = [n for n in graph.node if 'proto' in n.name]

    if not proto_nodes:
        LOG.debug("No proto nodes found in model")
        return None

    proto_add_nodes = [n for n in proto_nodes if n.op_type == "Add"]
    LOG.debug(f"Found {len(proto_add_nodes)} Add nodes in proto section")

    if not proto_add_nodes:
        LOG.debug("No Add nodes found in proto section")
        return None

    # Find external connections into proto section
    external_connections = []
    LOG.debug("Searching for external connections into proto section...")
    for proto_add in proto_add_nodes:
        for input_name in proto_add.input:
            input_producer = node_by_output.get(input_name)
            if input_producer and 'proto' not in input_producer.name:
                # External connection into proto section
                if input_producer.op_type in ["Add", "Mul"]:
                    LOG.debug(
                        f"  Found external connection: {input_producer.name} -> {proto_add.name}"
                    )
                    external_connections.append(
                        {
                            'producer': input_producer,
                            'consumer': proto_add,
                            'connection': input_name,
                        }
                    )

    # Find Add→Add chains within proto section
    proto_chains = []
    LOG.debug("Searching for Add chains in proto section...")
    for node in proto_add_nodes:
        for input_name in node.input:
            input_producer = node_by_output.get(input_name)
            if (
                input_producer
                and input_producer.op_type == "Add"
                and 'proto' in input_producer.name
            ):
                LOG.debug(f"  Found proto chain: {input_producer.name} -> {node.name}")
                proto_chains.append(
                    {
                        'producer': input_producer,
                        'consumer': node,
                        'connection': input_name,
                    }
                )

    if not external_connections and not proto_chains:
        LOG.debug("No problematic proto patterns found")
        return None

    LOG.info(
        f"Detected YOLO26 seg proto pattern: "
        f"{len(external_connections)} external connections, "
        f"{len(proto_chains)} proto chains"
    )

    return Yolo26SegProtoPattern(
        external_connections=external_connections, proto_chains=proto_chains
    )


def insert_yolo26_seg_proto_identity_ops(
    model: onnx.ModelProto,  # type: ignore[name-defined]
) -> onnx.ModelProto:  # type: ignore[name-defined]
    """
    Insert MaxPool identity operations in YOLO26 seg proto section.

    YOLO26 segmentation models fail to compile on Axelera hardware due to problematic
    dependency chains in the "proto" section (segmentation mask prototypes). This
    function inserts MaxPool identity operations (1x1 kernel, 1x1 stride) to break
    these chains while preserving model correctness.

    Patterns fixed:
    1. External connections: Add/Mul from outside proto → proto Add
    2. Chained Add operations: proto Add → proto Add

    The MaxPool operations are mathematical identities (output = input) but break
    direct dependency paths, enabling the compiler to optimize the graph.

    Args:
        model: ONNX model to transform

    Returns:
        Transformed model with MaxPool identity operations inserted

    Raises:
        RuntimeError: If no proto patterns found (indicates misconfiguration)
    """
    pattern = detect_yolo26_seg_proto_pattern(model)

    if pattern is None or (not pattern.external_connections and not pattern.proto_chains):
        raise RuntimeError(
            "YOLO26 seg proto identity insertion enabled but no proto patterns found. "
            "Model may already have identity operations inserted, be incorrectly configured, "
            "or the yolo26_seg_proto_fix flag should be set to false."
        )

    graph = model.graph

    # Create modifications in order: external connections first, then proto chains
    modifications = []
    node_counter = 0

    for conn in pattern.external_connections:
        maxpool_node, new_output = insert_maxpool_identity(conn['connection'], node_counter)
        modifications.append(
            {
                'identity': maxpool_node,
                'producer': conn['producer'],
                'consumer': conn['consumer'],
                'old_connection': conn['connection'],
                'new_connection': new_output,
            }
        )
        node_counter += 1

    for chain in pattern.proto_chains:
        maxpool_node, new_output = insert_maxpool_identity(chain['connection'], node_counter)
        modifications.append(
            {
                'identity': maxpool_node,
                'producer': chain['producer'],
                'consumer': chain['consumer'],
                'old_connection': chain['connection'],
                'new_connection': new_output,
            }
        )
        node_counter += 1

    LOG.info(f"Inserting {len(modifications)} MaxPool identity operations into proto section")

    # Apply modifications: update consumer nodes to use new outputs
    for mod in modifications:
        consumer_node = mod['consumer']
        old_conn = mod['old_connection']
        new_conn = mod['new_connection']

        for i, inp in enumerate(consumer_node.input):
            if inp == old_conn:
                consumer_node.input[i] = new_conn
                LOG.debug(f"  Updated {consumer_node.name} input")
                break

    # Insert MaxPool nodes in topological order (right after each producer)
    new_nodes = []
    for node in graph.node:
        new_nodes.append(node)

        # Check if this node is a producer that needs MaxPool after it
        for mod in modifications:
            if node.name == mod['producer'].name:
                new_nodes.append(mod['identity'])
                LOG.debug(f"  Inserted {mod['identity'].name} after {node.name}")

    # Replace nodes
    del graph.node[:]
    graph.node.extend(new_nodes)

    LOG.info(
        f"Successfully inserted {len(modifications)} MaxPool identity operations "
        f"({pattern.external_connections.__len__()} external, {pattern.proto_chains.__len__()} chains)"
    )

    # Validate
    try:
        onnx.checker.check_model(model)
        LOG.debug("Model validation passed after proto identity insertion")
    except Exception as e:
        LOG.warning(f"Model validation warning after proto identity insertion: {e}")

    return model


def _load_preamble(model_path):
    """Load a preamble ONNX model, raising RuntimeError on failure."""
    if not _HAVE_ONNX:
        raise ImportError("onnx is required to load preamble models but is not installed")
    try:
        return onnx.load(model_path)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load preamble ONNX model from '{model_path}': {e}\n"
            "Please ensure the preamble file is a valid ONNX model."
        )


def _validate_preprocess_graph(graph, model_path):
    """Validate that a preamble graph contains a supported preprocessing pattern.

    Returns True if normalization ops found, False if no normalization ops,
    raises NotImplementedError for unsupported patterns.
    """
    has_sub = False
    has_div_or_mul = False
    unsupported_ops = []

    for node in graph.node:
        if node.op_type == "Sub":
            has_sub = True
        elif node.op_type in ["Div", "Mul"]:
            has_div_or_mul = True
        elif node.op_type not in ["Constant", "Identity"]:
            unsupported_ops.append(node.op_type)

    op_summary = f"Found operations: {sorted(set(n.op_type for n in graph.node))}"

    if not (has_sub or has_div_or_mul):
        if not graph.node:
            raise NotImplementedError(
                f"UNSUPPORTED PREAMBLE PATTERN: The preamble model is empty (no operations found).\n\n"
                "A preamble is the preprocessing subgraph (before the first convolutional layer) "
                "that is split into a separate ONNX file during model compilation.\n\n"
                "Possible causes:\n"
                f"1. The preamble file path is incorrect -- verify the 'preprocess_graph' entry "
                f"in manifest.json in your compiled model directory.\n"
                "2. The file is corrupt or not a valid ONNX model.\n"
                "3. A build error during compilation produced an empty preamble -- "
                "re-run compilation to verify.\n"
                "4. If the model has no preprocessing subgraph, remove the 'preprocess_graph' "
                "entry from manifest.json in your compiled model directory.\n"
                f"Preamble file: {model_path}"
            )

        if unsupported_ops:
            raise NotImplementedError(
                "UNSUPPORTED PREAMBLE PATTERN: The preamble contains only non-normalization "
                "operations -- automatic pipeline integration is not supported for this pattern.\n\n"
                "Automatic integration is supported for normalization-only preambles (Sub, Div, Mul). "
                "Known patterns that fall outside this include SpaceToDepth (used in Focus layers).\n\n"
                f"{op_summary}\n\n"
                "To use this model, express the preprocessing as YAML preprocess operators "
                "(Resize, LetterboxResize, Normalize, etc.) in your pipeline configuration. "
                "If the preprocessing cannot be expressed using the available preprocess operators, "
                "contact Axelera support.\n\n"
                f"Preamble file: {model_path}"
            )

        LOG.info(
            f"Preamble does not contain normalization operations for auto-integration. "
            f"{op_summary}. File: {model_path}"
        )
        return False

    if unsupported_ops:
        raise NotImplementedError(
            "UNSUPPORTED PREAMBLE PATTERN: The preamble contains a mix of normalization and "
            "non-normalization operations -- automatic pipeline integration is not supported "
            "for this pattern.\n\n"
            "Automatic integration is supported for normalization-only preambles (Sub, Div, Mul). "
            "Known patterns that fall outside this include SpaceToDepth (used in Focus layers).\n\n"
            f"{op_summary}\n\n"
            "To use this model, express the preprocessing as YAML preprocess operators "
            "(Resize, LetterboxResize, Normalize, etc.) in your pipeline configuration. "
            "If the preprocessing cannot be expressed using the available preprocess operators, "
            "contact Axelera support.\n\n"
            f"Preamble file: {model_path}"
        )

    return True


def _extract_preprocess_constants(graph):
    """Extract and compose normalization constants from a preamble ONNX graph.

    Walks the graph in topological order (ONNX guarantees this) and composes
    Sub/Div/Mul nodes incrementally into a single affine (mean, std) pair using
    the formula: given state y=(x-m)/s, applying Sub(c) gives new_m=m+c*s; Div(c)
    gives new_s=s*c; Mul(c) gives new_s=s/c.

    Returns:
        Tuple of (mean_list, std_list) as plain Python lists.

    Raises:
        NotImplementedError: If the graph cannot be reduced to a single affine
            transform (non-constant operands, constant on wrong operand side,
            or non-linear data flow topology).
    """
    data_tensors = {inp.name for inp in graph.input}
    mean = np.array([0.0], dtype=np.float64)
    std = np.array([1.0], dtype=np.float64)

    for node in graph.node:
        if node.op_type == "Identity":
            if node.input and node.input[0] in data_tensors:
                data_tensors.add(node.output[0])
            continue
        if node.op_type == "Constant":
            continue
        if node.op_type not in ("Sub", "Div", "Mul"):
            continue

        inp0, inp1 = node.input[0], node.input[1]
        # Short-circuit: data path tensors are never constants
        c0 = None if inp0 in data_tensors else get_constant_value(graph, inp0)
        c1 = None if inp1 in data_tensors else get_constant_value(graph, inp1)

        if c0 is not None and c1 is not None:
            raise NotImplementedError(
                f"UNSUPPORTED PREAMBLE PATTERN: {node.op_type} node has both inputs as "
                f"constants ('{inp0}', '{inp1}'). Expected one data input and one constant."
            )

        if c0 is None and c1 is None:
            raise NotImplementedError(
                f"UNSUPPORTED PREAMBLE PATTERN: {node.op_type} node has no constant inputs "
                f"('{inp0}', '{inp1}'). Normalization parameters must be compile-time constants."
            )

        if c0 is not None:
            if node.op_type in ("Sub", "Div"):
                raise NotImplementedError(
                    f"UNSUPPORTED PREAMBLE PATTERN: {node.op_type} node has the constant as "
                    f"the first operand ('{inp0}'). Standard normalization requires the data "
                    f"as the first operand: (data {node.op_type} constant)."
                )
            const_val = c0
            data_inp = inp1
        else:
            const_val = c1
            data_inp = inp0

        if data_inp not in data_tensors:
            raise NotImplementedError(
                f"UNSUPPORTED PREAMBLE PATTERN: {node.op_type} node's data input '{data_inp}' "
                f"is not in the known data path. This preamble has a non-linear topology "
                f"(branching or out-of-order data flow) that cannot be automatically integrated."
            )

        c = np.atleast_1d(np.squeeze(np.asarray(const_val, dtype=np.float64)))

        if node.op_type == "Sub":
            mean = mean + c * std
        elif node.op_type == "Div":
            std = std * c
        else:  # Mul
            std = std / c

        data_tensors.add(node.output[0])

    if np.all(mean == mean.flat[0]):
        mean = np.array([mean.flat[0]])
    if np.all(std == std.flat[0]):
        std = np.array([std.flat[0]])

    return mean.tolist(), std.tolist()


def validate_preprocess_pattern(model_path):
    """Validate that the preamble ONNX model contains a supported preprocessing pattern.

    Returns:
        True if the preamble contains normalization operations that can be extracted.
        False if the preamble exists but doesn't contain normalization operations.

    Raises:
        RuntimeError: If the preamble file cannot be loaded.
        NotImplementedError: If the preamble contains unsupported operation patterns.
    """
    model = _load_preamble(model_path)
    return _validate_preprocess_graph(model.graph, model_path)


def get_preprocess_constants(model_path):
    """Extract preprocessing constants (Sub and Div) from an ONNX model.

    Args:
        model_path (str): Path to the ONNX model file.

    Returns:
        Tuple of two lists: (sub_constants, div_constants).
    """
    model = onnx.load(model_path)
    return _extract_preprocess_constants(model.graph)


def get_preamble_normalization(model_path):
    """Validate and extract normalization constants from a preamble in a single load.

    Returns:
        Tuple of (sub_constants, div_constants) if normalization is present, None otherwise.

    Raises:
        RuntimeError: If the preamble file cannot be loaded.
        NotImplementedError: If the preamble contains unsupported operation patterns.
    """
    model = _load_preamble(model_path)
    if not _validate_preprocess_graph(model.graph, model_path):
        return None
    return _extract_preprocess_constants(model.graph)

# Copyright Axelera AI, 2026
"""Rewrite a classifier's large trailing fully-connected layer as convolutions.

THIS ONE IS ABOUT SPEED, NOT ABOUT COMPILING
    The other passes in this directory exist because a build stops without
    them. This one does not: a model with a `Gemm` head compiles either way.
    It is here because of how the arithmetic lands on the chip.

WHAT IT LOOKS LIKE WITHOUT THE PASS
    The last layer is one matrix multiply: a `Gemm`, or a `MatMul` plus an
    `Add`, over a flattened feature vector. Written as a single large
    reduction it occupies a fraction of the matrix array, and the layer takes
    longer than its arithmetic requires. Nothing fails, and nothing warns you;
    you see it only in a benchmark, as a classifier spending its time in the
    last layer.

WHAT THE PASS DOES
    A fully-connected layer over a 1x1 feature map is a 1x1 convolution with
    the same weights, reshaped. The pass detects the trailing `Gemm`, reshapes
    the weight, and emits `Conv` in its place. Where the reduction is larger
    than the array can take in one pass it splits the layer into several
    `Conv` ops the compiler can tile and sums them, which is why the tolerance
    in the guard below is looser than an exact match: the split changes the
    order the products are accumulated in.

    `GEMM_TO_CONV_REDUCTION_THRESHOLD` is the point below which it declines to
    act. A small head is already fine, and rewriting it would buy nothing, so
    naming this pass on such a model is a no-op rather than a mistake.

WHICH MODELS NEED IT
    Classifiers with a large head that survived export as `Gemm`; ONNX out of
    timm is the common case. Benchmark before and after rather than assuming:
    no model in the Axelera Model Zoo declares this pass today, so it ships
    with no published before/after number behind it.

WHY IT CAN BE TRUSTED
    `run` compares the original and the rewritten graph on a random input and
    raises when they disagree. That check is what makes a speed rewrite safe
    to apply: it changes how the work is scheduled, never the answer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import onnx
from onnx import helper, numpy_helper

from ._onnx import LOG, get_constant_value, outputs_equivalent

# What happens if you do not run this pass. Free text, not a category: a build
# that stops is the easy case, and the ones that quietly cost you something are
# the reason this line exists. The README table repeats it verbatim, and a test
# keeps the two equal.
WITHOUT_IT = 'the model builds and runs, with the last layer using a fraction of the matrix array'

# Compiler backends this rewrite is valid for.
SUPPORTS = ('axrelay',)

# Weight formats it operates on. A build in another format skips it.
SOURCE_KINDS = ('.onnx',)

# One line, printed by `apply` when the pass runs.
DESCRIPTION = 'large trailing Gemm rewritten as tileable Convs, for matrix-array use'

# What retires this pass. Every pass here is a workaround with an end date.
RETIREMENT_CONDITION = 'the compiler maps a large Gemm onto the matrix array without this rewrite'

# Set to the SDK version that absorbed the fix to start the one-release grace
# period: the pass is skipped with a warning instead of running.
DEPRECATED_IN = None

# Threshold for Gemm->Conv conversion: only convert if reduction blocks exceed this
# MVM unit is 4x8x8 = 256, so we convert when workload exceeds capacity
GEMM_TO_CONV_REDUCTION_THRESHOLD = 256


def run(model: onnx.ModelProto) -> onnx.ModelProto | None:
    """Rewrite the trailing Gemm, or return `None` when there is nothing worth doing.

    `None` covers both declining cases: no `Gemm`/`MatMul` head at all, and a
    head below `GEMM_TO_CONV_REDUCTION_THRESHOLD`. It is the runner's "nothing
    to rewrite" answer and is reported, so a build cannot claim a rewrite this
    pass declined to make.
    """
    rewritten = replace_gemm_with_conv(model)
    if rewritten is model:
        return None
    if not validate_gemm_conv_equivalence(model, rewritten):
        raise RuntimeError(
            'gemm_to_conv produced a non-equivalent ONNX rewrite; drop the pass from '
            'PRE_COMPILE_PASSES, or supply a model whose head it recognizes.'
        )
    return rewritten


def validate_gemm_conv_equivalence(
    original_model: onnx.ModelProto,
    converted_model: onnx.ModelProto,
    rtol: float = 1e-4,
    atol: float = 1e-4,
) -> bool:
    """True when the Conv head computes what the Gemm head computed.

    The tolerance is 1e-4 rather than exact: a reduction split across several
    Conv ops accumulates the products in a different order from one Gemm.
    """
    ok = outputs_equivalent(original_model, converted_model, rtol=rtol, atol=atol)
    if ok:
        LOG.info('Gemm->Conv equivalence validation passed')
    return ok


@dataclass
class GemmToConvPattern:
    """Container for detected Gemm/MatMul pattern that can be converted to Conv."""

    reshape_node: onnx.NodeProto
    fc_node: onnx.NodeProto  # Gemm or MatMul node
    bias_node: onnx.NodeProto | None  # Add node for MatMul bias (if any)
    feature_map_name: str  # Input to Reshape (4D tensor)
    spatial_h: int
    spatial_w: int
    in_channels: int
    out_features: int
    is_matmul: bool  # True if MatMul, False if Gemm


def _infer_spatial_dims_from_shape_inference(
    model: onnx.ModelProto,
    tensor_name: str,
) -> tuple[int, int, int] | None:
    """Try to infer spatial dimensions using ONNX shape inference.

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
) -> tuple[int, int, int] | None:
    """Infer spatial dimensions from flattened feature count.

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
            # Prefer larger spatial dimensions for tiling benefit
            if (is_power_of_2 or is_common) and (
                found_spatial is None or spatial_size > found_spatial[0] * found_spatial[1]
            ):
                found_spatial = (h, w)
                found_channels = c

    if found_spatial is not None:
        return (found_channels, found_spatial[0], found_spatial[1])
    return None


def _get_fc_weight_and_dims(graph, fc_node: onnx.NodeProto) -> tuple[np.ndarray, int, int] | None:
    """Get weight tensor and determine in/out features for Gemm or MatMul.

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
    model: onnx.ModelProto,
) -> GemmToConvPattern | None:
    """Detect Gemm or MatMul layer preceded by Reshape that flattens a spatial feature map.

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
                    f"Shape inference mismatch: "
                    f"{in_channels}*{spatial_h}*{spatial_w} != {in_features}"
                )
                spatial_info = None

        # Fall back to heuristic inference
        if spatial_info is None:
            spatial_info = _infer_spatial_dims_from_in_features(in_features)

        if spatial_info is None:
            LOG.debug(
                f"FC node {fc_node.name}: could not infer spatial dims "
                f"from in_features={in_features}"
            )
            continue

        in_channels, spatial_h, spatial_w = spatial_info

        # Check if this conversion would help (reduction blocks > threshold)
        reduction_blocks = in_features // 64  # Assuming 64-element blocks
        if reduction_blocks <= GEMM_TO_CONV_REDUCTION_THRESHOLD:
            LOG.debug(
                f"FC node {fc_node.name}: reduction_blocks={reduction_blocks} "
                "<= threshold, skipping"
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

        # Check if tiling would produce tile_size = 0. The compiler calculates:
        # tile_size = ((in_channels // opt_tiles_num) // PWORD_SIZE) * PWORD_SIZE
        channels_per_tile = in_channels // opt_tiles_num
        if channels_per_tile < PWORD_SIZE:
            # Single Conv would be untileable, but we can split into multiple Convs
            # Calculate how many channel groups we need (each with >= PWORD_SIZE channels)
            num_channel_groups = max(1, (in_channels + PWORD_SIZE - 1) // PWORD_SIZE)

            LOG.info(
                f"FC node {fc_node.name}: Single Conv would be untileable "
                f"(in_channels={in_channels}, channels_per_tile={channels_per_tile} "
                f"< {PWORD_SIZE}). "
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
                    # get_constant_value handles both initializers and Constant nodes
                    bias_arr = get_constant_value(graph, other_input)
                    if (
                        bias_arr is not None
                        and len(bias_arr.shape) == 1
                        and bias_arr.shape[0] == out_features
                    ):
                        bias_node = n
                        break

        LOG.info(
            f"Detected {'MatMul' if is_matmul else 'Gemm'}->Conv candidate: "
            f"{fc_node.name or 'unnamed'}, "
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


def replace_gemm_with_conv(model: onnx.ModelProto) -> onnx.ModelProto:
    """Replace Gemm/MatMul (fully connected) layer with equivalent Conv layer.

    WHY THIS OPTIMIZATION EXISTS:
    ------------------------------
    The Axelera AIPU's MVM (Matrix-Vector Multiply) unit has limited capacity:
    - 4 weight sets x 8 rows x 8 cols x 64 elements = 256 parallel blocks
    - Large FC layers (e.g., 100352 -> 2048) exceed this capacity

    By converting Gemm -> Conv with spatial dimensions, we enable:
    - Spatial tiling by the compiler (process image patches sequentially)
    - Better memory locality and reduced intermediate buffer sizes
    - Hardware can process the same computation in smaller chunks

    MULTI-CONV FALLBACK STRATEGY:
    ------------------------------
    When a single Conv would be untileable (channels_per_tile < PWORD_SIZE=64):
    - Large kernels (e.g., 14x14) with moderate channels (e.g., 512) create
      untileable patterns where compiler needs 49 tiles but only 512 channels
    - Split input along channel dimension into groups (each >=64 channels)
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
            f"Replacing {fc_type} with Multi-Conv: "
            f"spatial={pattern.spatial_h}x{pattern.spatial_w}, "
            f"in_ch={pattern.in_channels}, out={pattern.out_features}, "
            f"num_groups={num_channel_groups} "
            f"(channels_per_group~={pattern.in_channels // num_channel_groups})"
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

    # get_constant_value handles both initializers and Constant nodes
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
        # Gemm: trust transB when present. The shape heuristic is only a tiebreak
        # for the attribute-absent case -- it is wrong for a square weight, whose
        # shape[0] == out_features regardless of layout, so an explicit transB=0
        # (weight is [in, out], must be transposed) would otherwise be skipped.
        trans_b = 0
        trans_b_found = False
        for attr in fc_node.attribute:
            if attr.name == 'transB':
                trans_b = attr.i
                trans_b_found = True
                break

        if trans_b:
            transpose = False  # weight already (out_features, in_features)
        elif trans_b_found:
            transpose = True  # explicit transB=0: weight is (in_features, out_features)
        else:
            transpose = old_weight.shape[0] != pattern.out_features  # ambiguous: infer from shape
        weight = old_weight.T if transpose else old_weight
        conv_weight = weight.reshape(
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
            zip(conv_weight_splits, split_output_names, strict=True)
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

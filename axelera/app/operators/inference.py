# Copyright Axelera AI, 2023
# Inference Operator Implementation
from __future__ import annotations

import dataclasses
import zipfile
import logging
from pathlib import Path
import re
import sys
from typing import TYPE_CHECKING, Any, Optional, Union

import numpy as np

from axelera import types

from .. import compile, config, constants, gst_builder, logging_utils, torch_utils
from ..torch_utils import torch
from .base import AxOperator, builtin
from .custom_preprocessing import PermuteChannels

if TYPE_CHECKING:
    from axelera import runtime

    from . import PipelineContext
    from .. import device_manager, gst_builder
    from ..pipe import graph

LOG = logging_utils.getLogger(__name__)
logging_utils.getLogger('axelera.runtime').setLevel(logging.WARNING)


def _build_mock_options() -> str:
    options = {}
    if mock := config.env.inference_mock:
        m = re.match(r'(load|save)-([^@\n]+)(?:@(\d+))?', mock)
        if not m:
            raise ValueError(f"Invalid mock option: {mock}, see AXELERA_HELP=1")
        if config.env.UseDmaBuf.OUTPUTS in config.env.use_dmabuf:
            raise ValueError(
                "Cannot use mock with output dmabufs, please disable with AXELERA_USE_DMABUF=1"
            )
        path = Path(m.group(2))
        if m.group(1) == 'load':
            if not path.is_dir():
                raise ValueError(f"Mock path {path} must exist and be a directory")
            if not (path / "shapes.txt").is_file():
                raise ValueError(f"Mock path {path} must exist and be a directory")
            options['mock-load'] = str(path)
            options['mock-fps'] = m.group(3) or '500'
        else:
            if not path.is_dir():
                LOG.info("Creating mock directory %s", path)
                path.mkdir(parents=True)
            options['mock-save'] = str(path)
    return ';'.join(f"{k}:{v}" for k, v in options.items())


def dequantize_single(np_array, dequant_params):
    """Dequantize a single np array."""
    scale, zero_point = dequant_params
    dequantized_array = (np_array - zero_point) * scale
    return dequantized_array


def dequantize(np_arrays, dequantize_params):
    """Dequantize a list of np arrays."""
    return [
        dequantize_single(np_array, params)
        for np_array, params in zip(np_arrays, dequantize_params)
    ]


def pad_and_quantize(np_array, quant_params, n_padded_ch, tensor_layout):
    """Pad and quantize a single numpy array. Quantization must be
    done after padding to have the same zeropoint for the padded pixels.
    """
    scale, zero_point = quant_params[0]
    quantized = np.round(np_array / scale + zero_point).clip(-128, 127).astype(np.int8)

    if n_padded_ch:
        n_low, n_high = compile.get_padded_low_high(n_padded_ch, tensor_layout, 'N')[0]
        top, bottom = compile.get_padded_low_high(n_padded_ch, tensor_layout, 'H')[0]
        left, right = compile.get_padded_low_high(n_padded_ch, tensor_layout, 'W')[0]
        c_low, c_high = compile.get_padded_low_high(n_padded_ch, tensor_layout, 'C')[0]

        pad_width = (
            (n_low, n_high),
            (top, bottom),
            (left, right),
            (c_low, c_high),
        )  # Assuming the order is batch, height, width, channels (NHWC)
        quantized = np.pad(quantized, pad_width, mode='constant', constant_values=zero_point)

    return quantized


def _convert_output_arrays(output_arrays, n_padded_ch, current_layout, expected_layout):
    if not all(isinstance(array, np.ndarray) for array in output_arrays):
        raise TypeError("All output arrays must be NumPy arrays")
    if len(output_arrays) != len(n_padded_ch):
        raise ValueError("Length of output_arrays and n_padded_ch must be the same")

    # Get the original shapes using get_original_shape
    output_shapes = tuple(array.shape for array in output_arrays)
    original_shapes = compile.get_original_shape(
        output_shapes, n_padded_ch, current_layout, expected_layout
    )

    # Convert the arrays according to the new shapes and layout
    converted_arrays = []
    for array, original_shape in zip(output_arrays, original_shapes):
        # Assuming the padding is only applied to the channels
        # and the rest of the dimensions are the same
        if current_layout == 'NHWC' and expected_layout == 'NCHW':
            array = np.moveaxis(array, -1, 1)
        elif current_layout == 'NCHW' and expected_layout == 'NHWC':
            array = np.moveaxis(array, 1, -1)

        # Slicing the array to remove padding if necessary
        converted_array = array[tuple(slice(dim) for dim in original_shape)]
        converted_arrays.append(converted_array)

    return converted_arrays


def _convert_to_tensors(data_input) -> torch.Tensor:
    """
    Convert a list containing torch.Tensors or numpy.ndarrays to a single stacked tensor.
    If a single tensor is provided, it is cloned and detached.

    Parameters:
    - data_input (torch Tensor or list): A tensor or a list of torch Tensors or numpy.ndarrays

    Returns:
    - torch Tensor: Resulting tensor
    """

    if isinstance(data_input, torch.Tensor):
        return data_input.clone().detach()
    if isinstance(data_input, list):
        if not data_input:
            return torch.empty(0)
        else:
            tensor_list = []
            for item in data_input:
                if isinstance(item, torch.Tensor):
                    tensor_list.append(item.clone().detach())
                elif isinstance(item, np.ndarray):
                    tensor_list.append(torch.from_numpy(item))
                else:
                    raise ValueError(f"Unsupported data type {type(item)} in the list")
            return tensor_list
    elif isinstance(data_input, np.ndarray):
        if data_input.size == 0:
            return torch.empty(0)
        else:
            return torch.from_numpy(data_input)
    else:
        raise ValueError(f"Unsupported data type {type(data_input)} in the list")


def _match_arrays_to_shapes(arrays, target_shapes, target_names=None):
    """
    Match arrays with target shapes based on element count.
    Simplified version without zero-dimension handling.
    """
    if not isinstance(arrays, list):
        arrays = [arrays]

    np_arrays = []
    for arr in arrays:
        if not isinstance(arr, np.ndarray):
            arr = np.array(arr)
        np_arrays.append(arr)

    # Create dictionaries for efficient matching by element count
    arrays_by_size = {}
    for i, arr in enumerate(np_arrays):
        size = arr.size
        if size not in arrays_by_size:
            arrays_by_size[size] = []
        arrays_by_size[size].append((i, arr))

    shapes_by_size = {}
    for i, shape in enumerate(target_shapes):
        size = np.prod(shape)
        if size not in shapes_by_size:
            shapes_by_size[size] = []
        shapes_by_size[size].append((i, shape))

    # Match arrays to shapes by element count
    matched_arrays = []
    used_array_indices = set()
    used_shape_indices = set()

    # Process each size that has both arrays and shapes
    for size in set(arrays_by_size.keys()) & set(shapes_by_size.keys()):
        available_arrays = arrays_by_size[size]
        available_shapes = shapes_by_size[size]

        # Match arrays to shapes until we run out of either
        for (array_idx, array), (shape_idx, shape) in zip(available_arrays, available_shapes):
            if array_idx not in used_array_indices and shape_idx not in used_shape_indices:
                matched_arrays.append((array, shape, shape_idx))
                used_array_indices.add(array_idx)
                used_shape_indices.add(shape_idx)

    unmatched_arrays = [arrays[i] for i in range(len(arrays)) if i not in used_array_indices]
    unmatched_shape_indices = [i for i in range(len(target_shapes)) if i not in used_shape_indices]

    if unmatched_shape_indices:
        unmatched_shapes = [target_shapes[i] for i in unmatched_shape_indices]
        LOG.warning(
            f"Some target shapes have no matching array by element count: {unmatched_shapes}"
        )
    elif unmatched_arrays:
        LOG.warning(f"Could not find matching target shapes for {len(unmatched_arrays)} arrays")

    return matched_arrays, unmatched_arrays, unmatched_shape_indices


def _reshape_to_target_shapes(arrays, target_shapes):
    """
    Reshape arrays to target shapes using element count matching.
    Simplified version without zero-dimension handling.
    """
    if not isinstance(arrays, list):
        arrays = [arrays]

    matched, unmatched, unmatched_idx = _match_arrays_to_shapes(arrays, target_shapes)

    if unmatched:
        unmatched_shapes = [arr.shape for arr in unmatched]
        raise ValueError(
            f"Arrays with shapes {unmatched_shapes} couldn't be matched to any target shape "
            f"in {target_shapes}. Element counts don't match."
        )

    reshaped_arrays = []
    for arr, shape, _ in matched:
        reshaped_arr = arr.reshape(shape)
        reshaped_arrays.append(reshaped_arr)

    return reshaped_arrays


def convert_to_rgba(
    tensor: torch.Tensor, input_layout: types.TensorLayout = types.TensorLayout.NCHW
) -> torch.Tensor:
    """
    Converts an RGB tensor to RGBA with alpha set to 0 based on the layout.

    :param tensor: The input tensor in RGB format.
    :param input_layout: One of 'NCHW', 'NHWC', or 'CHWN'.
    :return: A tensor in RGBA format with the same layout.
    """
    if input_layout == types.TensorLayout.NCHW:
        if tensor.shape[1] != 3:
            raise ValueError(f"Expected tensor with 3 channels but got {tensor.shape[1]}")
        alpha_channel = torch.zeros(
            (tensor.shape[0], 1, tensor.shape[2], tensor.shape[3]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor, alpha_channel], dim=1)
    elif input_layout == types.TensorLayout.NHWC:
        if tensor.shape[3] != 3:
            raise ValueError(f"Expected tensor with 3 channels but got {tensor.shape[3]}")
        alpha_channel = torch.zeros(
            (tensor.shape[0], tensor.shape[1], tensor.shape[2], 1),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor, alpha_channel], dim=3)
    elif input_layout == types.TensorLayout.CHWN:
        if tensor.shape[0] != 3:
            raise ValueError(f"Expected tensor with 3 channels but got {tensor.shape[0]}")
        alpha_channel = torch.zeros(
            (1, tensor.shape[1], tensor.shape[2], tensor.shape[3]),
            dtype=tensor.dtype,
            device=tensor.device,
        )
        return torch.cat([tensor, alpha_channel], dim=0)
    else:
        raise ValueError("Invalid layout. Expected one of 'NCHW', 'NHWC', or 'CHWN'.")


def _add_batch_channel(
    tensor: torch.Tensor, input_tensor_layout: types.TensorLayout
) -> torch.Tensor:
    """
    Checks the dimensions of a tensor and adds a batch channel if necessary.

    If the tensor has shape (height, width, channels), this function will add a
    new dimension of size 1 to the beginning of the tensor, effectively adding
    a batch channel. If the tensor already has a batch channel (i.e., has shape
    (batch_size, height, width, channels)), this function does nothing.

    Args:
        tensor: The torch.Tensor to modify.

    Returns:
        The modified tensor.
    """

    if tensor.dim() == 3:
        if input_tensor_layout == types.TensorLayout.CHWN:
            tensor = tensor.unsqueeze(-1)
        else:
            tensor = tensor.unsqueeze(0)
    return tensor


def _determine_device(model):
    return 'aipu' if isinstance(model, types.Manifest) else torch_utils.device_name()


def build_onnx_inferencer(
    model: str, axm_path: str = ''
) -> tuple[Any, list[str], list[str], list[list[int]]]:
    import onnxruntime as rt

    preferred_providers = [
        # onnxruntime-gpu
        'CUDAExecutionProvider',
        # onnxruntime-gpu
        'MPSExecutionProvider' if sys.platform == 'darwin' else None,
        # onnxruntime-openvino
        'OpenVINOExecutionProvider',
        'CPUExecutionProvider',
    ]

    # Filter out unavailable providers
    available_providers = rt.get_available_providers()
    providers = [provider for provider in preferred_providers if provider in available_providers]
    LOG.debug(f"Available ONNX runtime providers: {available_providers}")

    onnx_input: str | bytes = model
    if axm_path:
        with zipfile.ZipFile(axm_path, 'r') as axm:
            onnx_input = axm.read(model)

    session = rt.InferenceSession(onnx_input, providers=providers)

    post_input_names = [node.name for node in session.get_inputs()]
    post_output_names = [node.name for node in session.get_outputs()]
    post_input_shapes = [node.shape for node in session.get_inputs()]
    return session, post_input_names, post_output_names, post_input_shapes


def _run_onnx_session(session, input_names, output_names, inputs):
    """Run ONNX inference session with proper input handling.

    Args:
        session: ONNX inference session
        input_names: List of input names expected by the session
        output_names: List of output names to retrieve
        inputs: Single tensor/array or list of tensors/arrays

    Returns:
        Output from ONNX session run
    """
    # Convert to list if single input
    inputs_list = inputs if isinstance(inputs, (list, tuple)) else [inputs]

    # Convert all inputs to numpy arrays
    numpy_inputs = []
    for inp in inputs_list:
        if isinstance(inp, torch.Tensor):
            numpy_inputs.append(inp.cpu().numpy())
        elif isinstance(inp, np.ndarray):
            numpy_inputs.append(inp)
        else:
            raise ValueError(f"Unsupported input type: {type(inp)}")

    # Create input dictionary and run session
    input_dict = dict(zip(input_names, numpy_inputs))
    return session.run(output_names, input_dict)


def _get_model_path(model_root: Path, model_file) -> Path | None:
    '''Return a checked path to the model.'''
    json_path = model_root / model_file
    if not json_path.exists():
        raise RuntimeError(f"{json_path!s} does not exist, cannot proceed")
    return json_path


@dataclasses.dataclass
class InferenceOpConfig:
    """
    Configuration settings related to inference operator.
    Initialized from model info and potentially updated by a postprocess operator.

    The 4 handle flags affect C++ pipeline only; for torch-aipu, the inference operator always
    handles dequantization, depadding, transpose, and pre-/post-amble processing.

    handle_all: Optional convenience parameter to set all 4 handle flags at once.
            - If None (default): individual flags are used as specified
            - If True: all 4 handle flags are set to True and the input/output will totally follow the source ONNX input/output nodes.
              This simplifies integration but might not be optimal for performance in all cases.
            - If False: all 4 handle flags are set to False

    Performance considerations:
        - For small models: Setting handle_all=False and moving operations to postprocessing may improve performance
          through fusion optimizations
        - For large models: Using handle_all=True typically doesn't impact performance significantly and simplifies integration
    """

    handle_all: Optional[bool] = None

    # Individual handle flags - only used when handle_all is None
    handle_dequantization_and_depadding: Optional[bool] = None
    handle_transpose: Optional[bool] = None
    handle_postamble: Optional[bool] = None
    handle_preamble: Optional[bool] = None

    # Other configuration options
    dequantize_using_lut: bool = True
    postamble_onnxruntime_intra_op_num_threads: int = 4
    postamble_onnxruntime_inter_op_num_threads: int = 4
    postamble_onnx: str = ''  # optional manual-cut postamble ONNX model path

    def __post_init__(self):
        """Validates the flags and sets the internal state.
        If handle_all is set, individual flags must not be explicitly set.
        If postamble is True, depadding and transpose must also be True (will warn and force).
        """
        # Check for conflicting configuration
        individual_flags_set = any(
            [
                self.handle_dequantization_and_depadding is not None,
                self.handle_transpose is not None,
                self.handle_postamble is not None,
                self.handle_preamble is not None,
            ]
        )

        if self.handle_all is not None and individual_flags_set:
            raise ValueError(
                "Cannot set both 'handle_all' and individual handle flags. "
                "Use either 'handle_all' for convenience or set individual flags explicitly."
            )

        # Configure flags based on handle_all or set defaults for individual flags
        if self.handle_all is not None:
            self._configure_all_cpp_flags(self.handle_all)
        else:
            # Set defaults for individual flags if not specified
            if self.handle_dequantization_and_depadding is None:
                self.handle_dequantization_and_depadding = True
            if self.handle_transpose is None:
                self.handle_transpose = True
            if self.handle_postamble is None:
                self.handle_postamble = True
            if self.handle_preamble is None:
                self.handle_preamble = True

        # Validate dependencies between flags
        if self.handle_postamble:
            if not self.handle_dequantization_and_depadding:
                LOG.warning(
                    "Configuration conflict: 'handle_postamble' is True, "
                    "so 'handle_dequantization_and_depadding' must also be True. Forcing it to True."
                )
                self.handle_dequantization_and_depadding = True
            if not self.handle_transpose:
                LOG.warning(
                    "Configuration conflict: 'handle_postamble' is True, "
                    "so 'handle_transpose' must also be True. Forcing it to True."
                )
                self.handle_transpose = True

        # Check postamble_onnx exists if specified
        if self.postamble_onnx:
            postamble_onnx = Path(self.postamble_onnx).resolve()
            if not postamble_onnx.is_file():
                raise ValueError(
                    f"Postamble ONNX model file {self.postamble_onnx} does not exist. "
                    "Please provide a valid path to the postamble ONNX model."
                )
            self.postamble_onnx = str(postamble_onnx)

        # Check if threads are reasonable
        if self.postamble_onnxruntime_intra_op_num_threads < 1:
            raise ValueError(
                f"postamble_onnxruntime_intra_op_num_threads must be >= 1, got {self.postamble_onnxruntime_intra_op_num_threads}"
            )
        if self.postamble_onnxruntime_inter_op_num_threads < 1:
            raise ValueError(
                f"postamble_onnxruntime_inter_op_num_threads must be >= 1, got {self.postamble_onnxruntime_inter_op_num_threads}"
            )

        self._transpose_aipu_output = None

    @property
    def transpose_aipu_output(self):
        if self._transpose_aipu_output is None:
            raise ValueError(
                "transpose_aipu_output is not set. Please call reconcile_manifest() "
                "after configuring the InferenceOpConfig with a manifest."
            )
        return self._transpose_aipu_output

    @staticmethod
    def _is_scalar_output(output_shape: list) -> bool:
        """
        Determine if an output shape represents a scalar/classifier output.

        Scalar outputs (1x1 spatial dimensions) typically come from:
        - Classification models (e.g., ResNet, EfficientNet)
        - Feature extraction models
        - Global pooling outputs

        These outputs generally don't require transpose operations.

        Args:
            output_shape: List representing tensor shape [N, H, W, C] or similar

        Returns:
            bool: True if this is a scalar output (1x1 spatial dimensions)
        """
        if len(output_shape) < 3:
            return False
        h, w = output_shape[1], output_shape[2]
        return h == w == 1

    def _configure_all_cpp_flags(self, cpp_handles_all: bool):
        """Configure all C++ flags based on the cpp_handles_all flag."""
        self.handle_dequantization_and_depadding = cpp_handles_all
        self.handle_transpose = cpp_handles_all
        self.handle_postamble = cpp_handles_all
        self.handle_preamble = cpp_handles_all

    def _determine_transpose_from_output_info(
        self,
        output_info: types.OutputInfo,
        manifest_output_shape: list,
        model_info: types.ModelInfo,
        output_index: int,
    ) -> bool:
        """
        Determine if transpose is needed based on OutputInfo and manifest output shape.

        Args:
            output_info: The OutputInfo containing original model output shape and metadata
            manifest_output_shape: The actual output shape from the compiled manifest
            model_info: Model information for context

        Returns:
            bool: True if transpose is needed, False otherwise
        """
        from .. import compile

        manifest = model_info.manifest
        if not manifest or not manifest.n_padded_ch_outputs:
            # Fallback to simple heuristic if no padding info available
            return not self._is_scalar_output(manifest_output_shape)

        if output_index >= len(manifest.n_padded_ch_outputs):
            # Fallback to simple heuristic
            return not self._is_scalar_output(manifest_output_shape)

        try:
            depadded_shapes = compile.get_original_shape(
                output_shapes=(tuple(manifest_output_shape),),
                n_padded_ch=(manifest.n_padded_ch_outputs[output_index],),
                current_layout="NHWC",  # Manifest output is in NHWC format
                expected_layout="NHWC",
            )
            depadded_shape = depadded_shapes[0]

            # Compare depadded shape with original output shape (excluding batch dimension)
            output_info_shape = output_info.shape
            if len(output_info_shape) == len(depadded_shape):
                # Same number of dimensions, compare all except batch (index 0)
                return depadded_shape[1:] != output_info_shape[1:]
            else:
                # Different structure, fallback to heuristic
                return not self._is_scalar_output(manifest_output_shape)

        except Exception as e:
            LOG.warning(
                f"Failed to determine transpose from output info: {e}, falling back to heuristic"
            )
            return not self._is_scalar_output(manifest_output_shape)

    def reconcile_manifest(self, model_info: types.ModelInfo):
        """Reconcile the configuration with the compiled manifest."""
        manifest = model_info.manifest
        if self.handle_postamble and self.postamble_onnx:
            if manifest.postprocess_graph:
                LOG.warning(
                    f"Duplicated postamble configuration detected. The ONNX postamble model "
                    f"'{self.postamble_onnx}' will be used, overriding the one specified in "
                    f"the manifest '{manifest.postprocess_graph}'."
                )
            else:
                if Path(self.postamble_onnx).is_file():
                    LOG.info(f"Found custom postamble ONNX model '{self.postamble_onnx}'.")
                else:
                    raise ValueError(
                        f"Custom postamble ONNX model '{self.postamble_onnx}' does not exist."
                    )
            manifest.postprocess_graph = self.postamble_onnx

        if self.handle_transpose and manifest.output_shapes:
            self._transpose_aipu_output = []

            if model_info.output_info and len(model_info.output_info) == len(
                manifest.output_shapes
            ):
                for i, (output_info, manifest_output_shape) in enumerate(
                    zip(model_info.output_info, manifest.output_shapes)
                ):
                    needs_transpose = self._determine_transpose_from_output_info(
                        output_info, manifest_output_shape, model_info, i
                    )
                    self._transpose_aipu_output.append(needs_transpose)
            else:
                # DEPRECATED: Fallback to legacy heuristic method
                # TODO: Remove this fallback after 2 releases - all models should use explicit transpose metadata
                LOG.warning(
                    "Using deprecated heuristic transpose detection. Redeploy the model to include explicit output info"
                )

                for output_shape in manifest.output_shapes:
                    # Legacy heuristic logic:
                    # - Scalar outputs (1x1): typically classifiers/feature extractors, no transpose needed
                    # - Full-size outputs (matching input dims): image processing tasks, no transpose needed
                    is_scalar_output = self._is_scalar_output(output_shape)
                    is_fullsize_output = (
                        output_shape[1] == model_info.input_height
                        and output_shape[2] == model_info.input_width
                    )

                    needs_transpose = not is_scalar_output and not is_fullsize_output
                    self._transpose_aipu_output.append(needs_transpose)
        else:
            self._transpose_aipu_output = [False] * len(manifest.output_shapes)

    @staticmethod
    def from_yaml_dict(
        phases: dict[str, Any],
        template: dict[str, Any],
    ):
        """Create InferenceOpConfig from YAML dictionaries with priority order.

        Priority order (highest to lowest):
        1. phases - highest priority, overrides everything
        2. template - overrides defaults
        3. defaults - class default values

        Configuration logic:
        - If phases has 'handle_all', it takes precedence and individual flags are ignored
        - If phases doesn't have 'handle_all' but has individual flags, they override template's 'handle_all'
        - Otherwise, template's 'handle_all' is used if present

        Args:
            phases: Phase-specific configuration (highest priority)
            template: Template configuration (medium priority)

        Returns:
            InferenceOpConfig instance with merged configuration
        """
        config_kwargs = {}

        # Individual flag keys
        individual_flag_keys = {
            'handle_dequantization_and_depadding',
            'handle_transpose',
            'handle_postamble',
            'handle_preamble',
        }

        # Step 1: Apply template configuration
        template = template or {}
        phases = phases or {}

        # Check if phases has handle_all or individual flags
        phases_has_handle_all = "handle_all" in phases
        phases_has_individual_flags = any(key in phases for key in individual_flag_keys)

        # Determine configuration strategy
        if phases_has_handle_all:
            # Phases handle_all takes precedence, ignore all individual flags
            config_kwargs["handle_all"] = phases["handle_all"]

            # Apply non-flag settings from template first, then phases
            for key, value in template.items():
                if key != "handle_all" and key not in individual_flag_keys:
                    config_kwargs[key] = value

            for key, value in phases.items():
                if key != "handle_all" and key not in individual_flag_keys:
                    config_kwargs[key] = value

        elif phases_has_individual_flags:
            # Phases has individual flags, don't use handle_all from anywhere
            # Apply all template settings except handle_all
            for key, value in template.items():
                if key != "handle_all":
                    config_kwargs[key] = value

            # Apply phases settings (overrides template)
            for key, value in phases.items():
                config_kwargs[key] = value

        else:
            # No handle_all or individual flags in phases, use template as-is
            for key, value in template.items():
                config_kwargs[key] = value

            # Apply phases settings (overrides template)
            for key, value in phases.items():
                config_kwargs[key] = value

        return InferenceOpConfig(**config_kwargs)

    @property
    def cpp_decoder_does_dequantization_and_depadding(self) -> bool:
        return self._cpp_decoder_does_dequantization_and_depadding


class Inference:
    # assume that the input tensor is well prepared according to its required order
    # but still lack of the batch channel

    # device can be AUTO, AIPU, CPU, CUDA
    # model is the model instance in Python and model manifest for AIPU
    # if device is AIPU, model must be a Manifest instance

    def __init__(
        self,
        device_man: device_manager.DeviceManager,
        compiled_model_dir: Path | None,
        model_name: str,
        model: Union[types.Manifest, types.Model],
        model_info: types.ModelInfo,
        inference_op_config: InferenceOpConfig,
        low_latency: bool,
    ):
        self.compiled_model_dir = compiled_model_dir
        self.model_name = model_name
        self.model = model
        self.model_info = model_info
        self._output_shape0 = []
        self._permute_op = None
        self._device_man = device_man
        self._axr_conn: runtime.Connection = None
        self._axr_model: runtime.Model = None
        self._axr_modeli: runtime.ModelInstance = None
        self._inf_config = inference_op_config
        self.pre_ort_sess, self.post_ort_sess = None, None
        self._core_model_output_shapes_cached = None
        self._low_latency = low_latency
        self.devices = []

        self.device = _determine_device(self.model)
        input_tensor_layout = model_info.input_tensor_layout
        if model := self._try_load_quantized_model(self.model):
            self.model = model
        elif self.device == 'aipu':
            self.devices = self._device_man.devices
            if not isinstance(self.model, types.Manifest):
                raise ValueError('AIPU device requires model to be a Manifest instance')

        elif isinstance(self.model, torch.nn.Module):
            self.model.eval()
            self.model.to_device(torch.device(self.device))

            if input_tensor_layout and input_tensor_layout != types.TensorLayout.NCHW:
                self._permute_op = PermuteChannels(
                    input_layout=types.TensorLayout.NCHW,
                    output_layout=input_tensor_layout,
                )
        elif isinstance(self.model, types.ONNXModel):
            self.ort_sess, self.input_names, self.output_names, _ = build_onnx_inferencer(
                self.model.onnx_model.SerializeToString()
            )

            if input_tensor_layout and input_tensor_layout != types.TensorLayout.NCHW:
                self._permute_op = PermuteChannels(
                    input_layout=types.TensorLayout.NCHW,
                    output_layout=input_tensor_layout,
                )
        else:
            raise ValueError(f'Unsupported model type {type(self.model)}')
        super().__init__()

    def release(self):
        if self._axr_modeli:
            self._axr_modeli.release()
        if self._axr_conn:
            self._axr_conn.release()

    def _try_load_quantized_model(
        self, model: Union[types.Manifest, types.Model]
    ) -> torch.fx.GraphModule | None:
        if isinstance(model, types.Manifest):
            if model.model_lib_file is None:
                LOG.trace(
                    f"Manifest {self.model_name} does not have model_lib_file, assume quantized model"
                )
                try:
                    from qtoolsv2.utils.graph.graph_save import load_qtools_graphmodule

                    model_path = self.compiled_model_dir / constants.K_MODEL_QUANTIZED_FOR_DEBUG
                    if not model_path.is_file():
                        raise ValueError(
                            f"Please run ./deploy.py {self.model_name} --mode=quantize_debug to generate the quantized model file"
                        )
                    model = load_qtools_graphmodule(model_path)
                    model.eval()
                except Exception as e:
                    raise ValueError(
                        f"Failed to load quantized model from manifest {self.model_name}: {e}"
                    )
                self.device = torch_utils.device_name()
                self._init_pre_and_post()
                return model
        return None

    def _get_core_model_output_shapes(self):
        if self._core_model_output_shapes_cached:
            return self._core_model_output_shapes_cached
        elif self.model.manifest_version == '1.1':
            self._core_model_output_shapes_cached = self.model.output_shapes_original
        else:
            # backward compatibility for manifest version 1.0; default to using output_info shapes
            LOG.warning(
                "We highly recommend redeploying the model as the manifest version will be deprecated soon. Run `make NN=<model_name> clean` to clean the build directory and redeploy the model."
            )
            output_shapes = [output_info.shape for output_info in self.model_info.output_info]
            self._core_model_output_shapes_cached = output_shapes
        return self._core_model_output_shapes_cached

    def _init_pre_and_post(self):
        self.pre_ort_sess, self.post_ort_sess = None, None

        # torch preprocessing gives NCHW - always permute to NHWC for AIPU if torch-aipu
        # (we will address TF2 NHWC input in SDK-2649)
        if self.device == 'aipu':
            self._permute_op = PermuteChannels(
                input_layout=types.TensorLayout.NCHW,
                output_layout=types.TensorLayout.NHWC,
            )

        if self.model.postprocess_graph:
            (
                self.post_ort_sess,
                self.post_input_names,
                self.post_output_names,
                self.post_input_shapes,
            ) = build_onnx_inferencer(self.compiled_model_dir / self.model.postprocess_graph)

            LOG.trace("Expected input for postprocess graph:")
            LOG.trace(
                [f"{input.name}: {input.shape}" for input in self.post_ort_sess.get_inputs()]
            )
        if self.model.preprocess_graph:
            (
                self.pre_ort_sess,
                self.pre_input_names,
                self.pre_output_names,
                self.pre_input_shapes,
            ) = build_onnx_inferencer(self.compiled_model_dir / self.model.preprocess_graph)
            LOG.trace("Expected input for preprocess graph:")
            LOG.trace([f"{input.name}: {input.shape}" for input in self.pre_ort_sess.get_inputs()])

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        compiled_model_dir: Path | None,
        task_graph: graph.DependencyGraph,
    ):
        self.task_name = task_name
        self.model_category = model_info.task_category
        if self.device == 'aipu':
            self._model_cores = model_info.manifest.input_shapes[0][0]
            self._output_shapes = model_info.manifest.output_shapes
        self._taskn = taskn
        if model_info.manifest and model_info.manifest.is_compiled():
            self._quant = model_info.manifest.quantize_params

        if context.color_format != model_info.input_color_format:
            raise ValueError(
                f"Input color format mismatch in {task_name}. Expected {model_info.input_color_format}, but got {context.color_format}"
            )

    def check_focus_layer_on_host(self):
        if hasattr(self.model, 'preprocess_graph') and self.model.preprocess_graph:
            preprocess_path = Path(self.model.preprocess_graph)
            if not preprocess_path.is_absolute():
                preprocess_path = self.compiled_model_dir / preprocess_path
            return has_focus_layer_onnx(preprocess_path)

    @property
    def config(self):
        return self._inf_config

    def _do_pads_or_preproc(self, gst: gst_builder.Builder):
        from ..pipe.gst import generate_padding

        padding = generate_padding(self.model)
        if self.config.handle_preamble and self.check_focus_layer_on_host():
            # Currently, we only handle a single special case involving 'preamble.onnx'.
            # TODO: Refactor and generalize preamble handling, similar to our approach for postamble
            gst.axtransform(lib='libtransform_yolopreproc.so', options=f'padding:{padding}')
        else:
            _, zero = zip(*self._quant)
            gst.axtransform(
                lib='libtransform_padding.so', options=f'padding:{padding};fill:{zero[0]}'
            )

    def build_inference_gst(self, gst: gst_builder.Builder, num_cores: int):
        self._do_pads_or_preproc(gst)

        num_children = 0
        if self._model_cores < num_cores:
            if num_cores % self._model_cores == 0:
                num_children = num_cores // self._model_cores
                nd = len(self.devices)
                LOG.debug(
                    f"Enabled {num_children}x{nd} inference queues for {self.model_name} because "
                    f"model_cores={self._model_cores} and num_cores={num_cores}"
                )
            else:
                LOG.info(
                    f"This model is restricted to run on up to {self._model_cores} cores (not {num_cores})"
                )

        model = _get_model_path(self.compiled_model_dir, self.model.model_lib_file)

        name = f'inference-task{self._taskn}'
        options = _build_mock_options()
        inf = dict(
            name=name,
            model=str(model),
            devices=','.join(d.name for d in self.devices),
            double_buffer=config.env.use_double_buffer and not self._low_latency,
            dmabuf_inputs=config.env.UseDmaBuf.INPUTS in config.env.use_dmabuf,
            dmabuf_outputs=config.env.UseDmaBuf.OUTPUTS in config.env.use_dmabuf,
            num_children=num_children,
        )
        if gst.tiling and gst.add_tiles:
            inf['meta'] = 'axelera-tiles-internal'
        if options:
            inf['options'] = options
        sopts = ' '.join(f"{k}={v}" for k, v in inf.items())
        LOG.debug(f"Using inferencenet {sopts}")
        gst.axinference(**inf)

    def exec_torch(self, image, result, meta):
        # result is the input tensor which changes in-place
        result = _add_batch_channel(result, self.model_info.input_tensor_layout)
        if isinstance(self.model, torch.nn.Module):
            if self._permute_op:
                result = self._permute_op.exec_torch(result)
            if self.pre_ort_sess:
                result = _run_onnx_session(
                    self.pre_ort_sess, self.pre_input_names, self.pre_output_names, result
                )
            result = result.to(self.device)
            with torch.no_grad():
                result = self.model(result)
            result = self._process_model_outputs(result)
        elif isinstance(self.model, types.ONNXModel):
            if self._permute_op:
                input_array = self._permute_op.exec_torch(result).numpy()
            else:
                input_array = result.numpy()
            result = _run_onnx_session(
                self.ort_sess, self.input_names, self.output_names, input_array
            )
            result = _convert_to_tensors(result if len(result) > 1 else result[0])
        elif isinstance(self.model, types.Manifest):
            if self._axr_conn is None:

                model_path = _get_model_path(self.compiled_model_dir, self.model.model_lib_file)
                c = self._device_man.context
                self._batch_size = config.env.torch_aipu_batch_size
                self._axr_model = c.load_model(model_path)
                self._axr_conn = c.device_connect(None, self._batch_size)
                self._axr_modeli = self._axr_conn.load_model_instance(
                    self._axr_model,
                    num_sub_devices=self._batch_size,
                    aipu_cores=self._batch_size,
                )

                LOG.debug(f"Loaded model : {model_path}")
                self._init_pre_and_post()

            if self.pre_ort_sess:
                input_dict = dict(zip(self.pre_input_names, [result.numpy()]))
                input_array = self.pre_ort_sess.run(self.pre_output_names, input_dict)
                assert len(input_array) == 1, (
                    f"Support only one input tensor as AIPU input for "
                    f"now, got {len(input_array)}"
                )
                input_array = torch.from_numpy(input_array[0])
            else:
                input_array = result

            # suppose the model has only one input, pad and quantize it
            # TODO: support multiple inputs
            # input_array = convert_to_rgba(input_array, self.input_tensor_layout)

            input_array = self._permute_op.exec_torch(input_array).numpy()
            input_array = pad_and_quantize(
                input_array,
                self.model.quantize_params,
                self.model.n_padded_ch_inputs,
                'NHWC',
            )

            if self._batch_size > 1:
                input_array = np.repeat(input_array, self._batch_size, axis=0)
                outputs = [
                    np.empty((self._batch_size,) + t.shape[1:], np.int8)
                    for t in self._axr_model.outputs()
                ]
            else:
                outputs = [np.empty(t.shape, np.int8) for t in self._axr_model.outputs()]
            inputs = [input_array]

            self._axr_modeli.run(inputs, outputs)

            if self._batch_size > 1:
                outputs = [o[0:1] for o in outputs]  # take only the first batch element

            if len(outputs) != len(self.model.dequantize_params):
                raise ValueError(
                    f"Got number of output arrays {len(outputs)} != number of dequantize params "
                    f"{len(self.model.dequantize_params)}"
                )

            outputs = [o.astype(np.float32) for o in outputs]
            if True:  # was self.need_padding_and_layout_transform_of_inputs_outputs
                # this should be factored out into a different fn
                if self.model.n_padded_ch_outputs:
                    # Convert from padded NHWC to NCHW with padding removed
                    outputs = _convert_output_arrays(
                        outputs,
                        self.model.n_padded_ch_outputs,
                        self.model.input_tensor_layout,
                        "NCHW",
                    )

                # Dequantize before reshaping to work with float values
                outputs = dequantize(outputs, self.model.dequantize_params)

                # Process outputs - either with post-processing or just reshaping
                result = self._process_model_outputs(outputs)

        else:
            raise ValueError(f"Unsupported model type: {type(self.model)}")
        return image, result, meta

    def _process_model_outputs(self, outputs):
        """
        Process model outputs by applying post-processing or reshaping.

        This method handles two main cases:
        1. If post-processing is needed, it matches outputs with the right shapes
           and runs them through the post-processing ONNX model
        2. If only reshaping is needed, it uses _reshape_to_target_shapes

        Parameters:
        - outputs: List of output arrays from model inference

        Returns:
        - Processed outputs, converted to tensors
        """
        if not isinstance(outputs, list):
            outputs = [outputs]

        if self.post_ort_sess:
            matched_outputs, unmatched_outputs, _ = _match_arrays_to_shapes(
                outputs, self.post_input_shapes, self.post_input_names
            )

            post_inputs = [None] * len(self.post_input_shapes)
            for array, shape, idx in matched_outputs:
                post_inputs[idx] = array.reshape(shape)

            for i in range(len(post_inputs)):
                if post_inputs[i] is None and unmatched_outputs:
                    array = unmatched_outputs.pop(0)
                    if array.size == np.prod(self.post_input_shapes[i]):
                        post_inputs[i] = array.reshape(self.post_input_shapes[i])

            post_inputs = [inp for inp in post_inputs if inp is not None]

            result = _run_onnx_session(
                self.post_ort_sess, self.post_input_names, self.post_output_names, post_inputs
            )

            if unmatched_outputs:
                result = result if isinstance(result, list) else [result]
                result.extend(unmatched_outputs)
        elif isinstance(self.model, types.Manifest) and (
            out_shapes := self._get_core_model_output_shapes()
        ):
            result = _reshape_to_target_shapes(outputs, out_shapes)
        else:
            result = outputs

        return _convert_to_tensors(
            result if isinstance(result, list) and len(result) > 1 else result[0]
        )


def _calculate_tensor_selection_plan(model, postprocess_graph_path, compiled_model_dir=None):
    """
    Calculate tensor selection plan for postamble ONNX model.

    This determines which tensors from model output should be used
    as inputs to the postamble model.

    Args:
        model: Model manifest containing information about expected input shapes
        postprocess_graph_path: Path to the ONNX postprocessing graph
        compiled_model_dir: Optional directory where compiled models are stored

    Returns:
        List of tensor indices to use for ONNX postamble input
    """
    # Default to empty plan if no postprocess graph
    if not postprocess_graph_path:
        return []

    try:
        import onnx

        if compiled_model_dir and not Path(postprocess_graph_path).is_absolute():
            model_path = Path(compiled_model_dir) / postprocess_graph_path
        else:
            model_path = Path(postprocess_graph_path)

        # Check if the file exists
        if not model_path.exists():
            LOG.warning(f"Postprocess graph not found at: {model_path}")
            return []

        onnx_model = onnx.load(str(model_path))

        postamble_inputs = []
        for input_info in onnx_model.graph.input:
            postamble_inputs.append(input_info.name)

        # Get original output shapes from the model manifest
        if not model or not hasattr(model, 'output_format'):
            return list(range(len(postamble_inputs)))  # Default to sequential indices

        # Match shapes between model outputs and postamble inputs
        indices = []
        model_shapes = []
        if hasattr(model, 'get_original_shape'):
            model_shapes = model.get_original_shape() or []

        # If we have shape information, use it to match tensors
        if model_shapes:
            for i in range(len(postamble_inputs)):
                # For simplicity, just use sequential mapping if counts match
                if i < len(model_shapes):
                    indices.append(i)
                else:
                    # If postamble expects more inputs than model provides,
                    # just use what we have (will likely fail at runtime)
                    break
        else:
            # Fallback: use sequential indices
            output_layers = getattr(model, 'output_layers', [])
            indices = list(range(min(len(postamble_inputs), len(output_layers))))

        return indices
    except Exception as e:
        LOG.error(f"Error calculating tensor selection plan: {e}")
        return []  # Return empty plan on error


def _generate_depadding(manifest: types.Manifest) -> str:
    return '|'.join(
        ','.join(str(-num) for num in sublist) for sublist in manifest.n_padded_ch_outputs
    )


@builtin
class AxeleraDequantize(AxOperator):
    model: Union[types.Manifest, types.Model]
    inference_op_config: InferenceOpConfig
    num_classes: int
    task_category: types.TaskCategory
    assigned_model_name: str
    manifest_dir: Path
    taskn: int = 0
    transpose: bool = False

    def _post_init(self):
        self._model_name = self.assigned_model_name

    def exec_torch(self, *args):
        raise NotImplementedError()

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        connections = dict(src=f'decoder_task{self.taskn}{stream_idx}.sink_1')

        # Check if any post-processing is needed in this operator
        all_handled_by_decoder = not any(
            [
                self.inference_op_config.handle_dequantization_and_depadding,
                self.inference_op_config.handle_transpose,
                self.inference_op_config.handle_postamble,
            ]
        )

        if all_handled_by_decoder:
            return

        deq_scales, deq_zeropoints = zip(*self.model.dequantize_params)
        scales = ','.join(str(s) for s in deq_scales)
        zeros = ','.join(str(s) for s in deq_zeropoints)

        if self.inference_op_config.handle_transpose:
            transpose_str = ','.join(
                str(int(t)) for t in self.inference_op_config.transpose_aipu_output
            )
        else:
            transpose_str = ','.join('0' for _ in deq_scales)

        # If postamble is not handled by the decoder, but a postprocess graph exists,
        # we must add all transforms: depadding, dequantize+transpose, and postamble.
        if self.inference_op_config.handle_postamble and self.model.postprocess_graph:
            if stream_idx:
                gst.queue(name=f'stream_queue{stream_idx}')

            tensor_selection_plan = _calculate_tensor_selection_plan(
                self.model, self.model.postprocess_graph, self.manifest_dir
            )

            options = (
                f'onnx_path:{self.manifest_dir / self.model.postprocess_graph};'
                f'dequant_scale:{scales};dequant_zeropoint:{zeros};transpose:{transpose_str};'
                f'padding:{_generate_depadding(self.model)};'
                f'dequant_lut:{int(self.inference_op_config.dequantize_using_lut)}'
            )

            if tensor_selection_plan:
                options += f';tensor_selection_plan:{",".join(map(str, tensor_selection_plan))}'
            LOG.debug(f"Using tensor selection plan for postamble: {tensor_selection_plan}")
            gst.axtransform(
                lib='libtransform_postamble.so',
                options=options,
            )
            return

        # If postamble is handled by the decoder, but not all transforms are,
        # add only the transforms not handled by the decoder.
        if stream_idx:
            gst.queue(name=f'stream_queue{stream_idx}')
        deq_scales, deq_zeropoints = zip(*self.model.dequantize_params)
        scales = ','.join(str(s) for s in deq_scales)
        zeros = ','.join(str(s) for s in deq_zeropoints)

        # Only add dequantize if not handled by decoder
        if self.inference_op_config.handle_dequantization_and_depadding:
            if self.model.n_padded_ch_outputs and any(self.model.n_padded_ch_outputs):

                gst.axtransform(
                    lib='libtransform_paddingdequantize.so',
                    options=f'padding:{_generate_depadding(self.model)};'
                    f'dequant_scale:{scales};dequant_zeropoint:{zeros};transpose:{transpose_str};'
                    f'dequant_lut:{int(self.inference_op_config.dequantize_using_lut)}',
                    connections=connections,
                )
            else:
                gst.axtransform(
                    lib='libtransform_dequantize.so',
                    options=f'dequant_scale:{scales};dequant_zeropoint:{zeros};transpose:{transpose_str};'
                    f'dequant_lut:{int(self.inference_op_config.dequantize_using_lut)}',
                    connections=connections,
                )


def has_focus_layer_onnx(onnx_model_path):
    """
    Detects a focus-layer pattern in an ONNX model by looking for a group of Slice nodes
    operating on the input, followed by a Concat along the channel axis.
    """
    try:
        import onnx

        model = onnx.load(str(onnx_model_path))
        graph = model.graph
        if not graph.input:
            return False
        input_name = graph.input[0].name
        # Find all nodes that take the input tensor as input
        first_nodes = [node for node in graph.node if input_name in node.input]
        # Look for a group of Slice nodes that all take the input tensor
        slice_nodes = [node for node in first_nodes if node.op_type == "Slice"]
        if not slice_nodes:
            return False
        # Find concat nodes that take outputs of these slice nodes
        for node in graph.node:
            if node.op_type == "Concat":
                # Check if all inputs to this concat come from our slice nodes
                if all(any(inp in s.output for s in slice_nodes) for inp in node.input):
                    # Optionally, check axis attribute (should be channel axis, e.g., 1 for NCHW, 3 for NHWC)
                    axis = None
                    for attr in node.attribute:
                        if attr.name == "axis":
                            axis = attr.i
                    if axis in (1, 3):
                        return True
        return False
    except Exception:
        return False

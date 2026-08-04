# Copyright Axelera AI, 2023
# Pre-processing operators following TorchVision
# TODO: Add all of https://pytorch.org/vision/stable/transforms.html
from __future__ import annotations

import enum
from fractions import Fraction
from pathlib import Path
from typing import TYPE_CHECKING, List, Union

import cv2

from axelera import types
import numpy as np
from .. import gst_builder, logging_utils
from ..torch_utils import torch
from .base import PreprocessOperator, builtin
from .custom_preprocessing import PermuteChannels

LOG = logging_utils.getLogger(__name__)

if TYPE_CHECKING:
    from .. import gst_builder
    from ..pipe import graph
    from .context import PipelineContext


def _parse_multichannel_values(
    source: str, values: Union[str, int, float, Fraction]
) -> List[Fraction]:
    def float_expr(s):
        try:
            return Fraction(s)
        except ValueError:
            raise ValueError(f"Cannot convert '{s}' to float in {source}") from None

    if isinstance(values, str):
        values = [x.strip().replace("'", "") for x in values.split(',')]
    elif not isinstance(values, (list, tuple)):
        values = [values]
    if len(values) not in (1, 3, 4):
        raise ValueError(f'{source} expects 1, 3 or 4 float/fraction expressions (got {values!r})')

    values = [float_expr(t) for t in values]
    if all(values[0] == x for x in values[1:]):
        return values[:1]
    return values


@builtin
class Crop(PreprocessOperator):
    left: int
    top: int
    width: int
    height: int

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        opencl = gst.getconfig() is not None and gst.getconfig().opencl
        lib = "libtransform_roicrop_cl.so" if opencl else "libtransform_roicrop.so"
        gst.axtransform(
            lib=lib,
            options=f'left:{self.left};top:{self.top};width:{self.width};height:{self.height}',
        )

    def exec_torch(self, image: types.Image) -> types.Image:
        import torchvision.transforms.functional as TF

        i = image.aspil()
        i = TF.crop(i, self.top, self.left, self.height, self.width)
        return types.Image.frompil(i, image.color_format)


@builtin
class CenterCrop(PreprocessOperator):
    width: int
    height: int

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        if self.width <= 0 or self.height <= 0:
            raise ValueError(f"Invalid dimensions for CenterCrop: {self.width}x{self.height}")
        gst.axtransform(
            lib='transform_centrecropextra.so',
            options=f'crop_width:{self.width},crop_height:{self.height}',
        )

    def exec_torch(self, image: types.Image) -> types.Image:
        import torchvision.transforms.functional as TF

        i = image.aspil()
        i = TF.center_crop(i, (self.height, self.width))
        return types.Image.frompil(i, image.color_format)


def _compose_normalizations(mean1, std1, mean2, std2):
    """Compose two normalizations: z = (y - m2)/s2 where y = (x - m1)/s1.

    Result: z = (x - (m1 + m2*s1)) / (s1*s2).
    Returns (combined_mean, combined_std) as plain Python lists.
    """
    m1 = np.atleast_1d(np.array(mean1, dtype=np.float64))
    s1 = np.atleast_1d(np.array(std1, dtype=np.float64))
    m2 = np.atleast_1d(np.array(mean2, dtype=np.float64))
    s2 = np.atleast_1d(np.array(std2, dtype=np.float64))
    return (m1 + m2 * s1).tolist(), (s1 * s2).tolist()


@builtin
class Normalize(PreprocessOperator):
    mean: Union[List[float], str] = '0'
    std: Union[List[float], str] = '1'
    tensor_layout: types.TensorLayout = types.TensorLayout.NCHW
    format: str = 'RGB'

    def _post_init(self):
        self._enforce_member_type('tensor_layout')
        self._mean = _parse_multichannel_values(self.__class__.__name__, self.mean)
        self._std = _parse_multichannel_values(self.__class__.__name__, self.std)

    @property
    def mean_values(self) -> List[Fraction]:
        '''The mean values as a list of Fraction.

        If the mean values are the same for all channels, the list is length 1.
        '''
        return self._mean

    @property
    def std_values(self) -> List[Fraction]:
        '''The standard deviation values as a list of Fraction.

        If the std values are the same for all channels, the list is length 1.
        '''
        return self._std

    def combine_normalizations(self, mean, std):
        '''Combine this normalization with another normalization specified by mean and std.

        The resulting normalization is equivalent to applying this normalization followed by
        the other normalization. If y = (x - mean1) / std1 and z = (y - mean2) / std2, then
        z = (x - (mean1 + mean2 * std1)) / (std1 * std2)
        '''
        self._mean, self._std = _compose_normalizations(
            self.mean_values, self.std_values, mean, std
        )

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        compiled_model_dir: Path | None,
        task_graph,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, compiled_model_dir, task_graph
        )
        self._scale = [1.0]
        self._zero = [0.0]
        if model_info and model_info.manifest and model_info.manifest.is_compiled():
            q = model_info.manifest.quantize_params
            if q:
                self._scale, self._zero = zip(*q)

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        # Standalone fallback when no megaop fuser matched. Emits libtransform_normalize_cl.so
        # with mean/std/quant_scale/quant_zeropoint and to_tensor:0 (upstream already produced
        # a tensor via standalone ToTensor). The plugin operates on uint8 input and applies
        # /255 internally; the mean/std passed here represent the user-space normalization
        # (in [0, 1] domain), matching the convention used by OpenCLToTensorAndNormalize.
        _ensure_len3 = lambda seq: list(seq) + [seq[0]] * (3 - len(seq))
        mean = _ensure_len3([float(x) for x in self._mean])
        std = _ensure_len3([float(x) for x in self._std])
        scale = getattr(self, '_scale', [1.0]) or [1.0]
        zero = getattr(self, '_zero', [0.0]) or [0.0]
        m = ",".join(f'{x:.6f}'.rstrip('0') for x in mean)
        s = ",".join(f'{x:.6f}'.rstrip('0') for x in std)
        gst.axtransform(
            lib='libtransform_normalize_cl.so',
            options=(
                f'to_tensor:0;mean:{m};std:{s};'
                f'quant_scale:{float(scale[0])};quant_zeropoint:{float(zero[0])}'
            ),
        )

    def exec_torch(self, tensor: torch.Tensor):
        import torchvision.transforms.functional as TF

        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"Normalize input must be of type Tensor (got {type(tensor).__name__})"
            )
        return TF.normalize(
            tensor, [float(x) for x in self._mean], [float(x) for x in self._std], inplace=True
        )


@builtin
class LinearScaling(PreprocessOperator):
    '''linear scaling of the input tensor by a scale factor and an optional bias.
    Typically transforms the pixel values from a range of [0, 255] to a range of [-1, 1].'''

    mean: str = '1'  # Default value is 1 to avoid division by zero
    shift: str = '0'
    tensor_layout: types.TensorLayout = types.TensorLayout.NCHW

    def _post_init(self):
        self._enforce_member_type('tensor_layout')
        self._mean = _parse_multichannel_values(self.__class__.__name__, self.mean)
        self._shift = _parse_multichannel_values(self.__class__.__name__, self.shift)

    @property
    def mean_values(self) -> List[Fraction]:
        '''The mean values as a list of Fraction.'''
        return self._mean

    @property
    def shift_values(self) -> List[Fraction]:
        '''The shift values as a list of Fraction.'''
        return self._shift

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        div, add = self._mean, self._shift
        opts = []
        if len(div) == 1:
            if div[0] != 1.0:
                opts.append(f'div:{float(div[0])}')
        else:
            opts.extend(f'div:{float(x)}@{i}' for i, x in enumerate(div) if x != 1.0)

        if len(add) == 1:
            if add[0] != 0.0:
                opts.append(f'add:{float(add[0])}')
        else:
            opts.extend(f'add:{float(x)}@{i}' for i, x in enumerate(add) if x != 0.0)

        if opts and (len(div) > 1 or len(add) > 1):
            channel_pos = len(self.tensor_layout.name) - 1 - self.tensor_layout.name.index('C')
            opts.insert(0, f'per-channel:true@{channel_pos}')
        if opts:
            raise NotImplementedError('None fused LinearScaling not implemented in gst pipeline')

    def exec_torch(self, tensor: torch.Tensor):
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                f"LinearScaling input must be of type Tensor (got {type(tensor).__name__})"
            )

        mean_tensor = torch.tensor(self._mean, dtype=tensor.dtype, device=tensor.device)
        shift_tensor = torch.tensor(self._shift, dtype=tensor.dtype, device=tensor.device)

        # Reshape mean and shift tensors to match the input tensor's dimensions
        for _ in range(len(tensor.shape) - len(mean_tensor.shape)):
            mean_tensor = mean_tensor.unsqueeze(-1)
            shift_tensor = shift_tensor.unsqueeze(-1)

        return tensor / mean_tensor + shift_tensor


class InterpolationMode(enum.Enum):
    nearest = enum.auto()
    bilinear = enum.auto()
    bicubic = enum.auto()
    lanczos = enum.auto()
    pillow_bilinear = enum.auto()


_open_cv_interpolation_modes = {
    InterpolationMode.nearest: cv2.INTER_NEAREST,
    InterpolationMode.bilinear: cv2.INTER_LINEAR,
    InterpolationMode.bicubic: cv2.INTER_CUBIC,
    InterpolationMode.lanczos: cv2.INTER_LANCZOS4,
    InterpolationMode.pillow_bilinear: cv2.INTER_AREA,  # OpenCV doesn't have a separate pillow bilinear mode
}


@builtin
class Resize(PreprocessOperator):
    '''If both width and height are specified, the image is resized to width x height. If size is specified, the
    smaller edge is scaled to size, and the other edge is scaled to preserve the aspect ratio.   Specify either
    width/height or size, but not both.

    If half_pixel_centers is True, the image is resized using half-pixel centers, which is currently provided by the
    opencv backend only. If half_pixel_centers is False, the image is resized using the default behavior of the backend.

    interpolation can be one of nearest, bilinear, bicubic, lanczos, or pillow_bilinear.  e.g. in the yaml : `interpolation: nearest`.
    From python `operators.Resize(interpolation=operators.InterpolationMode.nearest)` or
    `Resize(interpolation='nearest').

    Not all interpolation modes are supported by all backends, so the backend may choose a different interpolation
    mode, and a warning will be logged.
    '''

    width: int = 0
    height: int = 0
    size: int = 0
    half_pixel_centers: bool = False

    interpolation: InterpolationMode = InterpolationMode.bilinear

    def _post_init(self):
        for member in ['width', 'height', 'size']:
            self._enforce_member_type(member)
            value = getattr(self, member)
            if value < 0:
                raise ValueError(f"Invalid unsigned int value for {member}: {value}")
        self._enforce_member_type('half_pixel_centers')
        self._enforce_member_type('interpolation')
        _sz = bool(self.size)
        _w = bool(self.width)
        _h = bool(self.height)
        _wh = _w and _h

        if _sz == _wh or _w != _h:
            raise ValueError(
                f"Only size (given: {self.size}) or both width/height (given: {self.width}/{self.height}) can be specified (i.e. non-zero)"
            )

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
        context.resize_status = types.ResizeMode.STRETCH

        # TODO: it's weird that the use of size follows the smallest dimension? If this is a real case, we need to add a resize mode

    @property
    def effective_width(self):
        return self.size or self.width

    @property
    def effective_height(self):
        return self.size or self.height

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        # preserve previous behaviour until we have a full gst solution
        w, h = (self.size, self.size) if self.size else (self.width, self.height)
        if gst.getconfig() is not None and gst.getconfig().opencl:
            lib = 'libtransform_resize_cl.so'
        else:
            lib = 'libtransform_resize.so'
        gst.axtransform(lib=lib, options=f'width:{w};height:{h};letterbox:0')

    def exec_torch(self, image: types.Image) -> types.Image:
        if self.half_pixel_centers:
            # OpenCV Resize defaults to half-pixel correction; tensorflow abd ONNX resize has a parameter to enable
            if self.size:  # but ONNX doesn't have smallest size feature
                return types.Image.fromarray(
                    self._aspect_preserving_resize(image.asarray(), self.size), image.color_format
                )
            else:
                sz = (self.effective_width, self.effective_height)
                im = _open_cv_interpolation_modes[self.interpolation]
                return types.Image.fromarray(
                    cv2.resize(image.asarray(), sz, interpolation=im), image.color_format
                )
        else:  # torchvision does not support half-pixel correction, but it has
            import torchvision.transforms as T
            import torchvision.transforms.functional as TF

            _torchvision_interpolation_modes = {
                InterpolationMode.nearest: T.InterpolationMode.NEAREST,
                InterpolationMode.bilinear: T.InterpolationMode.BILINEAR,
                InterpolationMode.bicubic: T.InterpolationMode.BICUBIC,
                InterpolationMode.lanczos: T.InterpolationMode.LANCZOS,
            }
            im = _torchvision_interpolation_modes[self.interpolation]
            sz = self.size if self.size else (self.height, self.width)
            return types.Image.frompil(
                TF.resize(
                    image.aspil(),
                    sz,
                    im,
                    max_size=None,
                    antialias=True,
                ),
                image.color_format,
            )

    def _aspect_preserving_resize(self, image, resize_min):
        """
        Resize an image while preserving the aspect ratio using NumPy and OpenCV.

        Args:
            image: A NumPy array representing the image.
            resize_min: The size of the smallest side after resize.

        Returns:
            Resized image as a NumPy array.
        """
        height, width = image.shape[:2]
        scale = resize_min / min(height, width)
        new_height = int(height * scale)
        new_width = int(width * scale)

        im = _open_cv_interpolation_modes[self.interpolation]
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=im)
        return resized_image


@builtin
class TypeCast(PreprocessOperator):
    '''Cast the tensor to given datatype.'''

    datatype: str = 'float32'

    def _post_init(self):
        super()._post_init()
        if self.datatype not in ('float32', 'uint8'):
            raise ValueError(
                f"Only float32 and uint8 are supported for datatype not '{self.datatype}'"
            )

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        pass

    def exec_torch(self, t: torch.Tensor) -> torch.Tensor:
        if not isinstance(t, torch.Tensor) or t.dtype != torch.uint8:
            got = str(t.dtype) if isinstance(t, torch.Tensor) else type(t).__name__
            raise TypeError(f'Input must be a torch tensor of uint8 not {got}')
        return t.type(getattr(torch, self.datatype))


@builtin
class ToTensor(PreprocessOperator):
    '''Converts from image domain to tensor domain.

    No other pre-processing is done. For an operator that also permutes and
    converts data type, use `TorchToTensor` instead.
    '''

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        # This will either pass the data straight through if the video has no extra stride
        # or it will copy the data to a new buffer with the correct stride
        gst.axtransform(
            lib='libtransform_resize.so',
            options='to_tensor:1',
        )

    def exec_torch(self, image: types.Image) -> torch.Tensor:
        return torch.from_numpy(image.asarray().copy())


def _resolve_effective_normalization(norm, preamble_path):
    """Compute effective (mean, std) float lists, folding preamble constants if present.

    Dispatches on norm type to extract base mean/std, then folds preamble constants
    via the composition formula: combined = (x - (m1 + m2*s1)) / (s1*s2).

    Args:
        norm: A Normalize, LinearScaling, or other operator (uses 0/1/255 defaults).
        preamble_path: Path to preamble ONNX file, or None.

    Returns:
        Tuple of (mean, std) as plain float lists.
    """
    if isinstance(norm, Normalize):
        base_mean = [float(x) for x in norm.mean_values]
        base_std = [float(x) for x in norm.std_values]
    elif isinstance(norm, LinearScaling):
        # LinearScaling formula: y = x/div + shift (operating on raw [0,255] uint8 pixels).
        # Rearranging to the (x - mean)/std form expected by the GStreamer plugin:
        #   y = (x - (-shift*div)) / div
        # With /255 to convert from [0,255] to [0,1] range:
        #   std  = div / 255
        #   mean = -shift * div / 255
        # _parse_multichannel_values collapses equal values to 1 element, so we broadcast
        # single-element lists to match the other's channel count before combining.
        divs = norm.mean_values
        shifts = norm.shift_values
        n = max(len(divs), len(shifts))
        divs = divs * n if len(divs) == 1 else divs
        shifts = shifts * n if len(shifts) == 1 else shifts
        base_std = [float(x / 255.0) for x in divs]
        base_mean = [-float(s * m / 255.0) for s, m in zip(shifts, divs)]
    else:
        LOG.warning(
            "Unknown normalization operator type %s, using default mean=0 std=1/255",
            type(norm).__name__,
        )
        base_mean = [0.0]
        base_std = [1.0 / 255.0]

    if preamble_path:
        try:
            from ax_models.onnx_optimizations import get_preamble_normalization

            onnx_norm = get_preamble_normalization(str(preamble_path))
            if onnx_norm is not None:
                base_mean, base_std = _compose_normalizations(
                    base_mean, base_std, onnx_norm[0], onnx_norm[1]
                )
        except ImportError:
            LOG.debug("onnx not available, skipping preamble normalization from %s", preamble_path)
        except NotImplementedError as e:
            LOG.warning("Preamble normalization skipped for %s: %s", preamble_path, e)

    return base_mean, base_std


class CompositePreprocess(PreprocessOperator):
    _norm = None

    def _set_operators(self, operators):
        self._operators = operators

    def set_stream_match(self, sm):
        self._stream_match = sm

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
        self._scale = []
        self._zero = []
        self._out_shape = []
        self._effective_mean = None
        self._effective_std = None
        for op in self._operators:
            op.configure_model_and_context_info(
                model_info, context, task_name, taskn, compiled_model_dir, task_graph
            )
        preamble_path = None
        if model_info and model_info.manifest and model_info.manifest.is_compiled():
            q = model_info.manifest.quantize_params
            if model_info.manifest.input_shapes:
                self._out_shape = model_info.manifest.input_shapes[0]
            self._scale, self._zero = zip(*q)
            if model_info.manifest.preprocess_graph and compiled_model_dir:
                preamble_path = Path(compiled_model_dir) / model_info.manifest.preprocess_graph
        if self._norm is not None:
            self._effective_mean, self._effective_std = _resolve_effective_normalization(
                self._norm, preamble_path
            )

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        for op in self._operators:
            op.build_gst(gst, stream_idx)

    def exec_torch(self, image: Union[torch.Tensor, types.Image]) -> torch.Tensor:
        x = image
        for op in self._operators:
            x = op.exec_torch(x)
        return x


@builtin
class TorchToTensor(CompositePreprocess):
    '''Converts from image to tensor domain, permutes to given layout, and casts type.

    This functionality is similar to the torchvision.transforms.ToTensor() operator.
    '''

    input_layout: str = 'NHWC'
    output_layout: str = 'NCHW'
    datatype: str = 'float32'
    scale: bool = True

    def _post_init(self):
        super()._post_init()
        ops = [
            ToTensor(),
            PermuteChannels(input_layout=self.input_layout, output_layout=self.output_layout),
            TypeCast(datatype=self.datatype),
        ]
        if self.scale:
            ops.append(Normalize(std='255.0'))
        self._set_operators(ops)

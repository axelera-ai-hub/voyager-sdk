# Copyright Axelera AI, 2024
# Pre-processing operators following TorchVision
# TODO: Add all of https://pytorch.org/vision/stable/transforms.html
from __future__ import annotations

import enum
from fractions import Fraction
from typing import TYPE_CHECKING, List, Union

import cv2

from axelera import types

from .. import gst_builder
from ..torch_utils import torch
from .base import PreprocessOperator, builtin
from .custom_preprocessing import PermuteChannels
from .utils import inspect_resize_status

if TYPE_CHECKING:
    from pathlib import Path

    from .. import gst_builder
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
class CenterCrop(PreprocessOperator):
    width: int
    height: int

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        gst.videobox(autocrop=True)
        gst.capsfilter(caps=f'video/x-raw,width={self.width},height={self.height}')

    def exec_torch(self, image: types.Image) -> types.Image:
        import torchvision.transforms.functional as TF

        i = image.aspil()
        i = TF.center_crop(i, (self.height, self.width))
        return types.Image.frompil(i, image.color_format)


@builtin
class Normalize(PreprocessOperator):
    mean: str = '0'
    std: str = '1'
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

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        add, div = [-x for x in self._mean], self._std
        opts = []
        if len(add) == 1:
            if add[0] != 0.0:
                opts.append(f'add:{float(add[0])}')
        else:
            opts.extend(f'add:{float(x)}@{i}' for i, x in enumerate(add) if x != 0.0)

        if len(div) == 1:
            if div[0] != 1.0:
                opts.append(f'div:{float(div[0])}')
        else:
            opts.extend(f'div:{float(x)}@{i}' for i, x in enumerate(div) if x != 1.0)

        if opts and (len(div) > 1 or len(add) > 1):
            channel_pos = len(self.tensor_layout.name) - 1 - self.tensor_layout.name.index('C')
            opts.insert(0, f'per-channel:true@{channel_pos}')  # Does this need to go at the front?
        if opts:
            raise NotImplementedError('None fused Normalize not implemented in gst pipeline')

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


_open_cv_interpolation_modes = {
    InterpolationMode.nearest: cv2.INTER_NEAREST,
    InterpolationMode.bilinear: cv2.INTER_LINEAR,
    InterpolationMode.bicubic: cv2.INTER_CUBIC,
    InterpolationMode.lanczos: cv2.INTER_LANCZOS4,
}


@builtin
class Resize(PreprocessOperator):
    '''If both width and height are specified, the image is resized to width x height. If size is specified, the
    smaller edge is scaled to size, and the other edge is scaled to preserve the aspect ratio.   Specify either
    width/height or size, but not both.

    If half_pixel_centers is True, the image is resized using half-pixel centers, which is currently provided by the
    opencv backend only. If half_pixel_centers is False, the image is resized using the default behavior of the backend.

    interpolation can be one of nearest, bilinear, bicubic, or lanczos.  e.g. in the yaml : `interpolation: nearest`.
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
    input_color_format: str = 'rgb'

    def _post_init(self):
        for member in ['width', 'height', 'size']:
            self._enforce_member_type(member)
            value = getattr(self, member)
            if value < 0:
                raise ValueError(f"Invalid unsigned int value for {member}: {value}")
        self._enforce_member_type('half_pixel_centers')
        self._enforce_member_type('interpolation')
        self._enforce_member_type('input_color_format')
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
        where: str,
        compiled_model_dir: Path,
    ):
        self.task_name = task_name
        inspect_resize_status(context)
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
        if self.datatype == 'float32':
            raise NotImplementedError('float32 is not supported in gst')

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
            options=f'to_tensor:1',
        )

    def exec_torch(self, image: types.Image) -> torch.Tensor:
        return torch.from_numpy(image.asarray().copy())


class CompositePreprocess(PreprocessOperator):
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
        where: str,
        compiled_model_dir: Path,
    ):
        self.task_name = task_name
        for op in self._operators:
            op.configure_model_and_context_info(
                model_info, context, task_name, taskn, where, compiled_model_dir
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

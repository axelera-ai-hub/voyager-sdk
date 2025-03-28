# Copyright Axelera AI, 2023
from pathlib import Path
import platform
from typing import Union

from axelera import types

from . import custom_preprocessing, preprocessing
from .. import gst_builder
from .context import PipelineContext
from .utils import inspect_resize_status


def _get_input_color_format(format: Union[str, types.ColorFormat]) -> str:
    format_str = format.name.lower() if isinstance(format, types.ColorFormat) else format
    return format_str[format_str.find('2') + 1 :]


class ResizeAndConvert(preprocessing.CompositePreprocess):
    width: int = 0
    height: int = 0
    size: int = 0
    format: str = 'rgb2bgr'

    def _post_init(self):
        self._set_operators(
            [
                preprocessing.Resize(
                    width=self.width,
                    height=self.height,
                    size=self.size,
                    input_color_format=_get_input_color_format(self.format),
                ),
                custom_preprocessing.ConvertColor(self.format),
            ]
        )
        return super()._post_init()


class OpenCLPerspectiveTransform(preprocessing.CompositePreprocess):
    camera_matrix: list[float] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    format: str = 'rgb'

    def _post_init(self) -> None:
        self._set_operators(
            [
                custom_preprocessing.Perspective(self.camera_matrix),
                custom_preprocessing.ConvertColorInput(self.format),
            ]
        )
        return super()._post_init()

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        if self.format != 'rgb' and self.format != 'bgr':
            raise ValueError(f'Invalid format on Perspective {self.format}')
        bgra_out = self.format == 'bgr'
        gst.axtransform(
            lib='libtransform_perspective_cl.so',
            options=f'matrix:{self.camera_matrix};bgra_out:{int(bgra_out)}',
        )


class OpenCLBarrelDistortionCorrection(preprocessing.CompositePreprocess):
    fx: float = 1.0
    fy: float = 1.0
    cx: float = 0.5
    cy: float = 0.5
    distort_coefs: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
    format: str = 'rgb'

    def _post_init(self) -> None:
        self._set_operators(
            [
                custom_preprocessing.CameraUndistort(
                    self.fx, self.fy, self.cx, self.cy, self.distort_coefs
                ),
                custom_preprocessing.ConvertColorInput(self.format),
            ]
        )
        return super()._post_init()

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        if self.format != 'rgb' and self.format != 'bgr':
            raise ValueError(f'Invalid format on CamerUndistort {self.format}')
        bgra_out = self.format == 'bgr'
        gst.axtransform(
            lib='libtransform_barrelcorrect_cl.so',
            options=f'camera_props:{self.fx},{self.fy},{self.cx},{self.cy};distort_coefs:{self.distort_coefs};bgra_out:{int(bgra_out)}',
        )


class CroppedResizeWithExtraCrop(preprocessing.CompositePreprocess):
    width: int = 0
    height: int = 0
    size: int = 0
    hcrop: int = 0
    vcrop: int = 0

    def _post_init(self) -> None:
        self._w, self._h = (self.width, self.height) if self.size == 0 else (self.size, self.size)
        self._set_operators(
            [
                preprocessing.Resize(width=self.width, height=self.height, size=self.size),
                preprocessing.CenterCrop(self._w - self.hcrop, self._h - self.vcrop),
            ]
        )
        return super()._post_init()

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        if self.size == 0:
            raise ValueError('CroppedResizeWithExtraCrop works on square images only.')
        if self.hcrop != self.vcrop:
            raise ValueError('CroppedResizeWithExtraCrop works on equal hcrop and vcrop only.')
        if platform.processor() == 'x86_64':
            gst.axtransform(
                lib='libtransform_resizeratiocropexcess.so',
                options=f'resize_size:{self.size};final_size_after_crop:{self.size - self.hcrop}',
            )
        else:
            cs = self._w - self.hcrop
            ss = self._w
            gst.axtransform(
                lib='libtransform_centrecropextra.so',
                options=f'cropsize:{cs};scalesize:{ss}',
            )
            gst.axtransform(
                lib='libtransform_resize.so',
                options=(f'size:{cs}'),
            )


class VAAPIResize(preprocessing.CompositePreprocess):
    width: int = 0
    height: int = 0
    size: int = 0
    input_color_format: str = 'rgb'

    def _post_init(self) -> None:
        self._w, self._h = (self.width, self.height) if self.size == 0 else (self.size, self.size)
        self._set_operators(
            [
                preprocessing.Resize(
                    width=self.width,
                    height=self.height,
                    size=self.size,
                    input_color_format=self.input_color_format,
                ),
            ]
        )
        return super()._post_init()

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        gst.vaapipostproc(
            {
                'width': self._w,
                'height': self._h,
                'format': f'{_get_input_color_format(self.input_color_format)}a',
                'scale-method': 0,
            }
        )
        gst.axinplace()


class OpenCLResize(preprocessing.CompositePreprocess):
    width: int = 0
    height: int = 0
    size: int = 0
    input_color_format: str = 'rgb'

    def _post_init(self) -> None:
        self._w, self._h = (self.width, self.height) if self.size == 0 else (self.size, self.size)
        self._set_operators(
            [
                preprocessing.Resize(
                    width=self.width,
                    height=self.height,
                    size=self.size,
                    input_color_format=self.input_color_format,
                ),
            ]
        )
        return super()._post_init()

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        options = f'size:{self.size}' if self.size else f'width:{self.width};height:{self.height}'
        options += f';format:{self.input_color_format}a'
        gst.axtransform(lib="libtransform_resize_cl.so", options=options)


class VAAPICroppedResizeWithExtraCrop(preprocessing.CompositePreprocess):
    width: int = 0
    height: int = 0
    size: int = 0
    hcrop: int = 0
    vcrop: int = 0

    def _post_init(self) -> None:
        self._w, self._h = (self.width, self.height) if self.size == 0 else (self.size, self.size)
        self._set_operators(
            [
                preprocessing.Resize(width=self.width, height=self.height, size=self.size),
                preprocessing.CenterCrop(self._w - self.hcrop, self._h - self.vcrop),
            ]
        )
        return super()._post_init()

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        cs = self._w - self.hcrop
        ss = self._w
        if gst.new_inference:
            # we don't support vaapi resize. yet
            if platform.processor() == 'x86_64':
                gst.axtransform(
                    lib='libtransform_resizeratiocropexcess.so',
                    options=f'resize_size:{self.size};final_size_after_crop:{self.size - self.hcrop}',
                )
            else:
                gst.axtransform(
                    lib='libtransform_centrecropextra.so',
                    options=f'cropsize:{cs};scalesize:{ss}',
                )
                gst.axtransform(
                    lib='libtransform_resize.so',
                    options=(f'size:{cs}'),
                )
            return
        gst.axtransform(
            lib='libtransform_centrecropextra.so',
            options=f'cropsize:{cs};scalesize:{ss}',
        )
        gst.vaapipostproc(
            {
                'width': self._w - self.hcrop,
                'height': self._h - self.vcrop,
                'format': 'rgba',
                'scale-method': 0,
            }
        )
        gst.axinplace()


class OpenCLCroppedResizeWithExtraCrop(preprocessing.CompositePreprocess):
    width: int = 0
    height: int = 0
    size: int = 0
    hcrop: int = 0
    vcrop: int = 0

    def _post_init(self) -> None:
        self._w, self._h = (self.width, self.height) if self.size == 0 else (self.size, self.size)
        self._set_operators(
            [
                preprocessing.Resize(width=self.width, height=self.height, size=self.size),
                preprocessing.CenterCrop(self._w - self.hcrop, self._h - self.vcrop),
            ]
        )
        return super()._post_init()

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        cs = self._w - self.hcrop
        ss = self._w
        gst.axtransform(
            lib='libtransform_centrecropextra.so',
            options=f'cropsize:{cs};scalesize:{ss}',
        )
        gst.axtransform(
            lib='libtransform_resize_cl.so',
            options=(f'size:{cs}'),
        )


class OpenCLCroppedResizeWithExtraCropWithColor(preprocessing.CompositePreprocess):
    width: int = 0
    height: int = 0
    size: int = 0
    hcrop: int = 0
    vcrop: int = 0
    format: str = 'rgb'

    def _post_init(self) -> None:
        self._w, self._h = (self.width, self.height) if self.size == 0 else (self.size, self.size)
        self._set_operators(
            [
                custom_preprocessing.ConvertColorInput(self.format),
                preprocessing.Resize(width=self.width, height=self.height, size=self.size),
                preprocessing.CenterCrop(self._w - self.hcrop, self._h - self.vcrop),
            ]
        )
        return super()._post_init()

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        cs = self._w - self.hcrop
        ss = self._w
        input_color_format = f'{_get_input_color_format(self.format)}a'
        gst.axtransform(
            lib='libtransform_colorconvert.so',
            options=(f'format:{input_color_format}'),
        )
        gst.axtransform(
            lib='libtransform_centrecropextra.so',
            options=f'cropsize:{cs};scalesize:{ss}',
        )
        gst.axtransform(
            lib='libtransform_resize_cl.so',
            options=(f'size:{cs};format:{input_color_format}'),
        )


class OpenCLCroppedResizeWithExtraCropAndNormalize(preprocessing.CompositePreprocess):
    width: int = 0
    height: int = 0
    size: int = 0
    hcrop: int = 0
    vcrop: int = 0
    mean: str = '0'
    std: str = '1'

    def _post_init(self) -> None:
        self._w, self._h = (self.width, self.height) if self.size == 0 else (self.size, self.size)
        self._set_operators(
            [
                preprocessing.Resize(width=self.width, height=self.height, size=self.size),
                preprocessing.CenterCrop(self._w - self.hcrop, self._h - self.vcrop),
                preprocessing.ToTensor(),
                preprocessing.PermuteChannels(input_layout='NHWC', output_layout='NCHW'),
                preprocessing.TypeCast(datatype='float32'),
                preprocessing.Normalize(std='255.0'),
                preprocessing.Normalize(mean=self.mean, std=self.std),
            ]
        )
        self._norm = self._operators[-1]
        self._out_shape = []
        return super()._post_init()

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
        if model_info.manifest and model_info.manifest.is_compiled():
            q = model_info.manifest.quantize_params
            if model_info.manifest.input_shapes:
                self._out_shape = model_info.manifest.input_shapes[0]  # TODO multiple inputs
            # TODO be a bit more helpful if the params are wrongly formatted
            self._scale, self._zero = zip(*q)

        context.resize_status = types.ResizeMode.STRETCH

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        cs = self._w - self.hcrop
        ss = self._w
        _ensure_len3 = lambda seq: list(seq) + [seq[0]] * (3 - len(seq))
        mean = _ensure_len3(self._norm.mean_values)
        std = _ensure_len3(self._norm.std_values)
        scale = _ensure_len3(self._scale)
        zero = _ensure_len3(self._zero)
        m = [f"{float(x):0.3f}".rstrip('0') for x in mean]
        s = [f"{float(x):0.3f}".rstrip('0') for x in std]
        mean = ",".join(map(str, m))
        std = ",".join(map(str, s))
        gst.axtransform(
            lib='libtransform_centrecropextra.so',
            options=f'cropsize:{cs};scalesize:{ss}',
        )
        gst.axtransform(
            lib='libtransform_resize_cl.so',
            options=f'size:{cs};to_tensor:1;mean:{mean};std:{std};quant_scale:{float(scale[0])};quant_zeropoint:{float(zero[0])}',
        )


class OpenCLetterBoxColorConvert(preprocessing.CompositePreprocess):
    width: int = 0
    height: int = 0
    format: str = 'rgb'
    scaleup: bool = True
    pad_val: int = 114

    def _post_init(self) -> None:
        self._set_operators(
            [
                custom_preprocessing.ConvertColorInput(self.format),
                custom_preprocessing.Letterbox(
                    width=self.width,
                    height=self.height,
                    scaleup=self.scaleup,
                    pad_val=self.pad_val,
                ),
            ]
        )
        return super()._post_init()

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        format = f"{_get_input_color_format(self.format)}a"
        gst.axtransform(
            lib='libtransform_resize_cl.so',
            options=(
                f'width:{self.width};height:{self.height};format:{format};padding:{self.pad_val};letterbox:1'
            ),
        )


class OpenCLResizeToTensorAndNormalize(preprocessing.CompositePreprocess):
    width: int = 0
    height: int = 0
    size: int = 0
    mean: str = '0'
    std: str = '1'
    datatype: str = 'float32'
    scaleup: int = 0

    def _post_init(self) -> None:
        self._set_operators(
            [
                preprocessing.Resize(width=self.width, height=self.height, size=self.size),
                preprocessing.ToTensor(),
                preprocessing.PermuteChannels(input_layout='NHWC', output_layout='NCHW'),
                preprocessing.TypeCast(datatype='float32'),
                preprocessing.Normalize(std='255.0'),
                preprocessing.Normalize(mean=self.mean, std=self.std),
            ]
        )
        self._norm = self._operators[-1]
        self._scale, self._zero = [], []
        self._out_shape = []
        return super()._post_init()

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
        if model_info.manifest and model_info.manifest.is_compiled():
            q = model_info.manifest.quantize_params
            if model_info.manifest.input_shapes:
                self._out_shape = model_info.manifest.input_shapes[0]  # TODO multiple inputs
            # TODO be a bit more helpful if the params are wrongly formatted
            self._scale, self._zero = zip(*q)

        context.resize_status = types.ResizeMode.STRETCH

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        _ensure_len3 = lambda seq: list(seq) + [seq[0]] * (3 - len(seq))
        mean = _ensure_len3(self._norm.mean_values)
        std = _ensure_len3(self._norm.std_values)
        scale = _ensure_len3(self._scale)
        zero = _ensure_len3(self._zero)
        m = [f"{float(x):0.3f}".rstrip('0') for x in mean]
        s = [f"{float(x):0.3f}".rstrip('0') for x in std]
        mean = ",".join(map(str, m))
        std = ",".join(map(str, s))
        gst.axtransform(
            lib='libtransform_resize_cl.so',
            options=(
                f'width:{self.width};height:{self.height};'
                + f'to_tensor:1;mean:{mean};std:{std};quant_scale:{float(scale[0])};quant_zeropoint:{float(zero[0])}'
            ),
        )


class OpenCLetterBoxToTensorAndNormalize(preprocessing.CompositePreprocess):
    width: int = 0
    height: int = 0
    scaleup: bool = True
    pad_val: int = 114
    mean: str = '0'
    std: str = '1'
    datatype: str = 'float32'

    def _post_init(self) -> None:
        self._set_operators(
            [
                custom_preprocessing.Letterbox(
                    width=self.width,
                    height=self.height,
                    scaleup=self.scaleup,
                    pad_val=self.pad_val,
                ),
                preprocessing.ToTensor(),
                preprocessing.PermuteChannels(input_layout='NHWC', output_layout='NCHW'),
                preprocessing.TypeCast(datatype='float32'),
                preprocessing.Normalize(std='255.0'),
                preprocessing.Normalize(mean=self.mean, std=self.std),
            ]
        )
        self._norm = self._operators[-1]
        self._scale, self._zero = [], []
        self._out_shape = []
        return super()._post_init()

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
        if model_info.manifest and model_info.manifest.is_compiled():
            q = model_info.manifest.quantize_params
            if model_info.manifest.input_shapes:
                self._out_shape = model_info.manifest.input_shapes[0]  # TODO multiple inputs
            # TODO be a bit more helpful if the params are wrongly formatted
            self._scale, self._zero = zip(*q)

        if self.scaleup:
            context.resize_status = types.ResizeMode.LETTERBOX_FIT
        else:
            context.resize_status = types.ResizeMode.LETTERBOX_CONTAIN

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        _ensure_len3 = lambda seq: list(seq) + [seq[0]] * (3 - len(seq))
        mean = _ensure_len3(self._norm.mean_values)
        std = _ensure_len3(self._norm.std_values)
        scale = _ensure_len3(self._scale)
        zero = _ensure_len3(self._zero)
        m = [f"{float(x):0.3f}".rstrip('0') for x in mean]
        s = [f"{float(x):0.3f}".rstrip('0') for x in std]
        mean = ",".join(map(str, m))
        std = ",".join(map(str, s))
        gst.axtransform(
            lib='libtransform_resize_cl.so',
            options=(
                f'width:{self.width};height:{self.height};padding:{self.pad_val};letterbox:1;scale_up:{int(self.scaleup)};'
                + f'to_tensor:1;mean:{mean};std:{std};quant_scale:{float(scale[0])};quant_zeropoint:{float(zero[0])}'
            ),
        )


class TypeCastAndNormalize(preprocessing.CompositePreprocess):
    datatype: str = 'float32'
    mean: str = '0'
    std: str = '1'
    tensor_layout: types.TensorLayout = types.TensorLayout.NCHW

    def _post_init(self):
        self._enforce_member_type('tensor_layout')
        self._set_operators(
            [
                preprocessing.TypeCast(self.datatype),
                preprocessing.Normalize(self.mean, self.std, self.tensor_layout),
            ]
        )
        return super()._post_init()

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        cast, norm = gst_builder.Builder(), gst_builder.Builder()
        self._operators[0].build_gst(cast, stream_idx)
        self._operators[1].build_gst(norm, stream_idx)
        if norm:
            cast, norm = cast[0], norm[0]
            norm['option'] = f"{cast['option']};{norm['option']}"
            gst.append(norm)
        else:
            gst.extend(cast)


def pad(x, size, fill):
    if len(x) == 1:
        x *= 3
    if len(x) < size:
        x += [fill] * (size - len(x))
    return x


class ToTensorAndLinearScaling(preprocessing.CompositePreprocess):
    datatype: str = 'float32'
    mean: str = '1'
    shift: str = '0'
    in_tensor_layout: types.TensorLayout = types.TensorLayout.NHWC
    out_tensor_layout: types.TensorLayout = types.TensorLayout.NCHW

    def _post_init(self):
        self._enforce_member_type('in_tensor_layout')
        self._enforce_member_type('out_tensor_layout')
        self._set_operators(
            [
                preprocessing.ToTensor(),
                preprocessing.PermuteChannels(self.in_tensor_layout, self.out_tensor_layout),
                preprocessing.TypeCast(self.datatype),
                preprocessing.LinearScaling(self.mean, self.shift, self.out_tensor_layout),
            ]
        )
        self._norm = self._operators[-1]
        self._out_shape = []
        # These values are defaults, I do not believe they will change but or be passed
        # in as args, but I am leaving them here for now
        self._scale = [1.0 / 255]
        self._zero = [0]

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
        if model_info.manifest and model_info.manifest.is_compiled():
            q = model_info.manifest.quantize_params
            if model_info.manifest.input_shapes:
                self._out_shape = model_info.manifest.input_shapes[0]  # TODO multiple inputs
            self._scale, self._zero = zip(*q)

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        s = [f"{float(x/255.0):0.6f}".rstrip('0') for x in self._norm.mean_values]
        m = [
            f"{-float(s * m /255.0):0.6f}".rstrip('0')
            for s, m in zip(self._norm.shift_values, self._norm.mean_values)
        ]
        quant = ''
        if (self._scale and self._scale[0] != 1) or (self._zero and self._zero[0] != 0):
            quant = f';quant_scale:{self._scale[0]};quant_zeropoint:{self._zero[0]}'

        gst.axtransform(
            lib='libtransform_totensor.so',
            options='type:int8',
        )
        gst.axinplace(
            lib='libinplace_normalize.so',
            mode='write',
            options=f'mean:{",".join(m)};std:{",".join(s)};simd:avx2{quant}',
        )


class ToTensorAndNormalise(preprocessing.CompositePreprocess):
    mean: str = '0'
    std: str = '1'
    datatype: str = 'float32'

    def _post_init(self):
        self._set_operators(
            [
                preprocessing.ToTensor(),
                preprocessing.PermuteChannels(input_layout='NHWC', output_layout='NCHW'),
                preprocessing.TypeCast(
                    datatype='float32'
                ),  # ignore datatype here, as this is quant datatype
                preprocessing.Normalize(std='255.0'),
                preprocessing.Normalize(mean=self.mean, std=self.std),
            ]
        )
        self._norm = self._operators[-1]
        self._scale, self._zero = [], []
        self._out_shape = []

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
        if model_info.manifest and model_info.manifest.is_compiled():
            q = model_info.manifest.quantize_params
            if model_info.manifest.input_shapes:
                self._out_shape = model_info.manifest.input_shapes[0]  # TODO multiple inputs
            # TODO be a bit more helpful if the params are wrongly formatted
            self._scale, self._zero = zip(*q)
            if (self._scale and any(x != self._scale[0] for x in self._scale[1:])) or (
                self._zero and any(x != self._zero[0] for x in self._zero[1:])
            ):
                raise ValueError(
                    "axinplace_normalize.write only supports uniform quantization params"
                )

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        m = [f"{float(x):0.3f}".rstrip('0') for x in self._norm.mean_values]
        s = [f"{float(x):0.3f}".rstrip('0') for x in self._norm.std_values]
        quant = ''
        if (self._scale and self._scale[0] != 1) or (self._zero and self._zero[0] != 0):
            quant = f';quant_scale:{self._scale[0]};quant_zeropoint:{self._zero[0]}'

        gst.axtransform(
            lib='libtransform_totensor.so',
            options='type:int8',
        )
        gst.axinplace(
            lib='libinplace_normalize.so',
            mode='write',
            options=f'mean:{",".join(m)};std:{",".join(s)};simd:avx2{quant}',
        )


class LetterboxToTensorAndNormalise(preprocessing.CompositePreprocess):
    height: int = '0'
    width: int = '0'
    mean: str = '0'
    std: str = '1'
    datatype: str = 'float32'
    scaleup: bool = True

    def _post_init(self):
        self._set_operators(
            [
                custom_preprocessing.Letterbox(height=self.height, width=self.width),
                preprocessing.ToTensor(),
                preprocessing.PermuteChannels(input_layout='NHWC', output_layout='NCHW'),
                preprocessing.TypeCast(
                    datatype='float32'
                ),  # ignore datatype here, as this is quant datatype
                preprocessing.Normalize(std='255.0'),
                preprocessing.Normalize(mean=self.mean, std=self.std),
            ]
        )
        self._norm = self._operators[-1]
        self._scale, self._zero = [], []
        self._out_shape = []

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
        if model_info.manifest and model_info.manifest.is_compiled():
            q = model_info.manifest.quantize_params
            if model_info.manifest.input_shapes:
                self._out_shape = model_info.manifest.input_shapes[0]  # TODO multiple inputs
            # TODO be a bit more helpful if the params are wrongly formatted
            self._scale, self._zero = zip(*q)
            if (self._scale and any(x != self._scale[0] for x in self._scale[1:])) or (
                self._zero and any(x != self._zero[0] for x in self._zero[1:])
            ):
                raise ValueError(
                    "axinplace_normalize.write only supports uniform quantization params"
                )

        inspect_resize_status(context)
        if self.scaleup:
            context.resize_status = types.ResizeMode.LETTERBOX_FIT
        else:
            context.resize_status = types.ResizeMode.LETTERBOX_CONTAIN

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        m = [f"{float(x):0.3f}".rstrip('0') for x in self._norm.mean_values]
        s = [f"{float(x):0.3f}".rstrip('0') for x in self._norm.std_values]
        quant = ''
        if (self._scale and self._scale[0] != 1) or (self._zero and self._zero[0] != 0):
            quant = f';quant_scale:{self._scale[0]};quant_zeropoint:{self._zero[0]}'

        gst.axtransform(
            lib='libtransform_resize.so',
            options=f'width:{self.width};height:{self.height};padding:114;to_tensor:1;letterbox:1;scale_up:{int(self.scaleup)}',
        )
        gst.axinplace(
            lib='libinplace_normalize.so',
            mode='write',
            options=f'mean:{",".join(m)};std:{",".join(s)};simd:avx2{quant}',
        )


class OpenCLToTensorAndNormalize(preprocessing.CompositePreprocess):
    mean: str = '0'
    std: str = '1'

    def _post_init(self):
        self._set_operators(
            [
                preprocessing.ToTensor(),
                preprocessing.PermuteChannels(input_layout='NHWC', output_layout='NCHW'),
                preprocessing.TypeCast(datatype='float32'),
                preprocessing.Normalize(std='255.0'),
                preprocessing.Normalize(mean=self.mean, std=self.std),
            ]
        )
        self._norm = self._operators[-1]
        self._out_shape = []

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
        if model_info.manifest and model_info.manifest.is_compiled():
            q = model_info.manifest.quantize_params
            if model_info.manifest.input_shapes:
                self._out_shape = model_info.manifest.input_shapes[0]  # TODO multiple inputs
            # TODO be a bit more helpful if the params are wrongly formatted
            self._scale, self._zero = zip(*q)

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        _ensure_len3 = lambda seq: list(seq) + [seq[0]] * (3 - len(seq))
        mean = _ensure_len3(self._norm.mean_values)
        std = _ensure_len3(self._norm.std_values)
        scale = _ensure_len3(self._scale)
        zero = _ensure_len3(self._zero)
        m = [f"{float(x):0.3f}".rstrip('0') for x in mean]
        s = [f"{float(x):0.3f}".rstrip('0') for x in std]
        mean = ",".join(map(str, m))
        std = ",".join(map(str, s))
        gst.axtransform(
            lib='libtransform_normalize_cl.so',
            options=f'to_tensor:1;mean:{mean};std:{std};quant_scale:{float(scale[0])};quant_zeropoint:{float(zero[0])}',
        )

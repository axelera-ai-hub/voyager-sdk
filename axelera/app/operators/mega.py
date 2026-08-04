# Copyright Axelera AI, 2023
from pathlib import Path
import platform
from typing import Union

import numpy as np

from axelera import types

from . import custom_preprocessing, preprocessing
from .. import config, gst_builder
from .context import PipelineContext
from .custom_preprocessing import get_output_format_spec
from .utils import inspect_resize_status, add_alpha_channel


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
                ),
                custom_preprocessing.ConvertColor(self.format),
            ]
        )
        return super()._post_init()


class OpenCLPerspectiveTransform(preprocessing.CompositePreprocess):
    camera_matrix: list[float] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    invert: bool = False
    format: types.ColorFormat = None

    def _post_init(self) -> None:
        self._enforce_member_type('format')
        self._set_operators(
            [
                custom_preprocessing.Perspective(self.camera_matrix, self.invert),
                custom_preprocessing.ConvertColorInput(self.format),
            ]
        )
        return super()._post_init()

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        matrix = np.array(self.camera_matrix).reshape(3, 3)
        if self.invert:
            matrix = np.linalg.inv(matrix)
        matrix = ','.join(f'{x:.6g}' for x in matrix.flatten())
        out = get_output_format_spec(self.format)
        gst.axtransform(
            lib='libtransform_perspective_cl.so',
            options=f'matrix:{matrix}{out}',
        )


class OpenCLBarrelDistortionCorrection(preprocessing.CompositePreprocess):
    fx: float = 1.0
    fy: float = 1.0
    cx: float = 0.5
    cy: float = 0.5
    distort_coefs: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
    normalized: bool = True
    format: types.ColorFormat = None

    def _post_init(self) -> None:
        self._enforce_member_type('format')
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
        out = get_output_format_spec(self.format)
        gst.axtransform(
            lib='libtransform_barrelcorrect_cl.so',
            options=f'camera_props:{self.fx},{self.fy},{self.cx},{self.cy};'
            f'normalized_properties:{int(self.normalized)};'
            f'distort_coefs:{",".join(str(coef) for coef in self.distort_coefs)}{out}',
        )


class OpenCLBarrelDistortionCorrectionResize(preprocessing.CompositePreprocess):
    fx: float = 1.0
    fy: float = 1.0
    cx: float = 0.5
    cy: float = 0.5
    distort_coefs: list[float] = [0.0, 0.0, 0.0, 0.0, 0.0]
    normalized: bool = True
    format: types.ColorFormat = None
    width: int = 0
    height: int = 0
    size: int = 0

    def _post_init(self) -> None:
        self._enforce_member_type('format')
        self._set_operators(
            [
                custom_preprocessing.CameraUndistort(
                    self.fx, self.fy, self.cx, self.cy, self.distort_coefs
                ),
                preprocessing.Resize(width=self.width, height=self.height, size=self.size),
                custom_preprocessing.ConvertColorInput(self.format),
            ]
        )
        return super()._post_init()

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        if self.size:
            ss = f'size:{self.size};'
        elif self.width and self.height:
            ss = f'width:{self.width};height:{self.height};'
        else:
            ss = ''

        out = get_output_format_spec(self.format)
        gst.axtransform(
            lib='libtransform_barrelcorrect_cl.so',
            options=f'camera_props:{self.fx},{self.fy},{self.cx},{self.cy};'
            f'normalized_properties:{int(self.normalized)};{ss}'
            f'distort_coefs:{",".join(str(coef) for coef in self.distort_coefs)}{out}',
        )


class OpenCLPolar(preprocessing.CompositePreprocess):
    width: int = 2510
    height: int = 800
    size: int = 0
    rotate180: bool = False
    center_x: float = 0.5
    center_y: float = 0.5
    max_radius: int = 800
    inverse: bool = False
    linear_polar: bool = True
    rotate180: bool = True
    format: types.ColorFormat = None

    def _post_init(self) -> None:
        self._enforce_member_type('format')
        self._set_operators(
            [
                custom_preprocessing.Polar(
                    width=self.width,
                    height=self.height,
                    max_radius=self.max_radius,
                    center_x=self.center_x,
                    center_y=self.center_y,
                    inverse=self.inverse,
                    linear_polar=self.linear_polar,
                    rotate180=self.rotate180,
                    format=self.format,
                ),
                custom_preprocessing.ConvertColorInput(self.format),
            ]
        )
        return super()._post_init()

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        out = get_output_format_spec(self.format)
        gst.axtransform(
            lib='libtransform_polar_cl.so',
            options=f'width:{self.width};height:{self.height};max_radius:{self.max_radius};'
            f'center_x:{self.center_x};center_y:{self.center_y};inverse:{int(self.inverse)};'
            f'linear_polar:{int(self.linear_polar)};rotate180:{int(self.rotate180)}{out}',
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
        cw = self._w - self.hcrop
        ch = self._h - self.vcrop
        sw = self._w
        sh = self._h

        if platform.processor() == 'x86_64' and cw == ch and self.size:
            gst.axtransform(
                lib='libtransform_resizeratiocropexcess.so',
                options=f'resize_size:{self.size};final_size_after_crop:{cw}',
            )
        else:
            if self.size:
                ss = f'scalesize:{self.size};'
            elif sw and sh:
                ss = f'scale_width:{sw};scale_height:{sh};'
            else:
                ss = ''
            gst.axtransform(
                lib='libtransform_centrecropextra.so',
                options=ss + f'crop_width:{cw};crop_height:{ch}',
            )
            gst.axtransform(
                lib='libtransform_resize.so',
                options=(f'width:{cw};height:{ch}'),
            )


class OpenCLResize(preprocessing.CompositePreprocess):
    width: int = 0
    height: int = 0
    size: int = 0
    input_color_format: str = 'rgb'

    def _post_init(self) -> None:
        self._w, self._h = (self.width, self.height) if self.size == 0 else (self.size, self.size)
        self._set_operators(
            [
                custom_preprocessing.ConvertColorInput(self.input_color_format),
                preprocessing.Resize(
                    width=self.width,
                    height=self.height,
                    size=self.size,
                ),
            ]
        )
        return super()._post_init()

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        options = f'size:{self.size}' if self.size else f'width:{self.width};height:{self.height}'
        options += f';format:{add_alpha_channel(self.input_color_format)}'
        gst.axtransform(lib="libtransform_resize_cl.so", options=options)


class OpenCLFaceAlign(preprocessing.CompositePreprocess):
    keypoints_key: str = None  # Only used for torch pipeline
    width: int = 0
    height: int = 0
    padding: float = 0.0
    template_keypoints_x: str = None
    template_keypoints_y: str = None
    use_self_normalizing: bool = False
    save_aligned_images: bool = True  # for debugging purposes
    format: str = 'rgb'

    def _post_init(self) -> None:
        self._set_operators(
            [
                custom_preprocessing.ConvertColorInput(self.format),
                custom_preprocessing.FaceAlign(
                    keypoints_key=self.keypoints_key,
                    width=self.width,
                    height=self.height,
                    padding=self.padding,
                    template_keypoints_x=self.template_keypoints_x,
                    template_keypoints_y=self.template_keypoints_y,
                    use_self_normalizing=self.use_self_normalizing,
                    save_aligned_images=self.save_aligned_images,
                ),
            ]
        )
        return super()._post_init()

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        compiled_model_dir: Path | None,
        task_graph,
    ):
        self._where = task_graph.get_master(task_name)
        self._association = context.association or None

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        master_key = f'master_meta:{self._where};' if self._where else str()
        association_key = f'association_meta:{self._association};' if self._association else str()
        fmt = add_alpha_channel(self.format)
        gst.axtransform(
            lib='libtransform_facealign_cl.so',
            options=f'{master_key}'
            f'{association_key}'
            f'width:{self.width};'
            f'height:{self.height};'
            f'padding:{self.padding};'
            f'template_keypoints_x:{",".join(map(str, self.template_keypoints_x))};'
            f'template_keypoints_y:{",".join(map(str, self.template_keypoints_y))};'
            f'use_self_normalizing:{int(self.use_self_normalizing)};'
            f'format:{fmt}',
        )


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
        cw = self._w - self.hcrop
        ch = self._h - self.vcrop
        sw = self._w
        sh = self._h

        if self.size:
            ss = f'scalesize:{self.size};'
        elif sw and sh:
            ss = f'scale_width:{sw};scale_height:{sh};'
        else:
            ss = ''
        gst.axtransform(
            lib='libtransform_centrecropextra.so',
            options=ss + f'crop_width:{cw};crop_height:{ch}',
        )
        gst.axtransform(
            lib='libtransform_resize_cl.so',
            options=(f'width:{cw};height:{ch};interpolation:2'),
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
        cw = self._w - self.hcrop
        ch = self._h - self.vcrop
        sw = self._w
        sh = self._h
        input_color_format = f'{_get_input_color_format(self.format)}'
        if self.size:
            ss = f'scalesize:{self.size};'
        elif sw and sh:
            ss = f'scale_width:{sw};scale_height:{sh};'
        else:
            ss = ''
        gst.axtransform(
            lib='libtransform_colorconvert_cl.so',
            options=(f'format:{input_color_format}'),
        )
        gst.axtransform(
            lib='libtransform_centrecropextra.so',
            options=ss + f'crop_width:{cw};crop_height:{ch}',
        )
        gst.axtransform(
            lib='libtransform_resize_cl.so',
            options=(f'width:{cw};height:{ch};format:{input_color_format};interpolation:2'),
        )


class OpenCLColorConvertCroppedResizeWithExtraCropAndNormalize(preprocessing.CompositePreprocess):
    width: int = 0
    height: int = 0
    size: int = 0
    hcrop: int = 0
    vcrop: int = 0
    mean: str = '0'
    std: str = '1'
    format: str = 'rgb'

    def _post_init(self) -> None:
        self._w, self._h = (self.width, self.height) if self.size == 0 else (self.size, self.size)
        self._set_operators(
            [
                custom_preprocessing.ConvertColorInput(self.format),
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
        return super()._post_init()

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
        context.resize_status = types.ResizeMode.STRETCH

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        cw = self._w - self.hcrop
        ch = self._h - self.vcrop
        sw = self._w
        sh = self._h
        if self.size:
            ss = f'scalesize:{self.size};'
        elif sw and sh:
            ss = f'scale_width:{sw};scale_height:{sh};'
        else:
            ss = ''
        _ensure_len = lambda seq, channels: list(seq) + [seq[0]] * (channels - len(seq))
        mean = _ensure_len(self._effective_mean, self._out_shape[-1])
        std = _ensure_len(self._effective_std, self._out_shape[-1])
        scale = _ensure_len(self._scale, self._out_shape[-1])
        zero = _ensure_len(self._zero, self._out_shape[-1])
        m = [f'{float(x):.6f}'.rstrip('0') for x in mean]
        s = [f'{float(x):.6f}'.rstrip('0') for x in std]
        mean = ",".join(map(str, m))
        std = ",".join(map(str, s))
        gst.axtransform(
            lib='libtransform_centrecropextra.so',
            options=ss + f'crop_width:{cw};crop_height:{ch}',
        )
        fmt = add_alpha_channel(self.format)
        gst.axtransform(
            lib='libtransform_resize_cl.so',
            options=f'width:{cw};height:{ch};to_tensor:1;mean:{mean};std:{std};quant_scale:{float(scale[0])};quant_zeropoint:{float(zero[0])};format:{fmt};interpolation:2',
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
        return super()._post_init()

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
        context.resize_status = types.ResizeMode.STRETCH

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        cw = self._w - self.hcrop
        ch = self._h - self.vcrop
        sw = self._w
        sh = self._h
        if self.size:
            ss = f'scalesize:{self.size};'
        elif sw and sh:
            ss = f'scale_width:{sw};scale_height:{sh};'
        else:
            ss = ''
        _ensure_len = lambda seq, channels: list(seq) + [seq[0]] * (channels - len(seq))
        mean = _ensure_len(self._effective_mean, self._out_shape[-1])
        std = _ensure_len(self._effective_std, self._out_shape[-1])
        scale = _ensure_len(self._scale, self._out_shape[-1])
        zero = _ensure_len(self._zero, self._out_shape[-1])
        m = [f'{float(x):.6f}'.rstrip('0') for x in mean]
        s = [f'{float(x):.6f}'.rstrip('0') for x in std]
        mean = ",".join(map(str, m))
        std = ",".join(map(str, s))
        gst.axtransform(
            lib='libtransform_centrecropextra.so',
            options=ss + f'crop_width:{cw};crop_height:{ch}',
        )
        gst.axtransform(
            lib='libtransform_resize_cl.so',
            options=f'width:{cw};height:{ch};to_tensor:1;mean:{mean};std:{std};quant_scale:{float(scale[0])};quant_zeropoint:{float(zero[0])};interpolation:2',
        )


class OpenCLetterBoxColorConvert(preprocessing.CompositePreprocess):
    width: int = 0
    height: int = 0
    format: str = 'rgb'
    scaleup: bool = True
    half_pixel_centers: bool = False
    pad_val: int = 114

    def _post_init(self) -> None:
        self._set_operators(
            [
                custom_preprocessing.ConvertColorInput(self.format),
                custom_preprocessing.Letterbox(
                    width=self.width,
                    height=self.height,
                    scaleup=self.scaleup,
                    half_pixel_centers=self.half_pixel_centers,
                    pad_val=self.pad_val,
                ),
            ]
        )
        return super()._post_init()

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        format = f"{_get_input_color_format(self.format)}"
        gst.axtransform(
            lib='libtransform_resize_cl.so',
            options=(
                f'width:{self.width};height:{self.height};format:{format};padding:{self.pad_val};letterbox:1'
            ),
        )


class OpenCLResizeColorConverToTensorAndNormalize(preprocessing.CompositePreprocess):
    width: int = 0
    height: int = 0
    size: int = 0
    mean: str = '0'
    std: str = '1'
    format: str = 'rgba'
    datatype: str = 'float32'
    scaleup: int = 0

    def _post_init(self) -> None:
        ops = [
            preprocessing.Resize(width=self.width, height=self.height, size=self.size),
            preprocessing.ToTensor(),
            preprocessing.PermuteChannels(input_layout='NHWC', output_layout='NCHW'),
            preprocessing.TypeCast(datatype='float32'),
            preprocessing.Normalize(std='255.0'),
            preprocessing.Normalize(mean=self.mean, std=self.std),
        ]
        if self.format:
            ops.insert(0, custom_preprocessing.ConvertColorInput(self.format))

        self._set_operators(ops)
        self._norm = self._operators[-1]
        return super()._post_init()

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
        context.resize_status = types.ResizeMode.STRETCH

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):

        _ensure_len3 = lambda seq: list(seq) + [seq[0]] * (3 - len(seq))
        mean = _ensure_len3(self._effective_mean)

        std = _ensure_len3(self._effective_std)
        scale = _ensure_len3(self._scale)
        zero = _ensure_len3(self._zero)
        m = [f'{float(x):.6f}'.rstrip('0') for x in mean]
        s = [f'{float(x):.6f}'.rstrip('0') for x in std]
        mean = ",".join(map(str, m))
        std = ",".join(map(str, s))
        out = ''
        if self.format:
            out = f';format:{add_alpha_channel(self.format)}'
        gst.axtransform(
            lib='libtransform_resize_cl.so',
            options=(
                f'width:{self.width};height:{self.height};'
                + f'to_tensor:1;mean:{mean};std:{std};quant_scale:{float(scale[0])};quant_zeropoint:{float(zero[0])}'
                + out
            ),
        )


class OpenCLResizeToTensorAndLinearScaling(preprocessing.CompositePreprocess):
    width: int = 0
    height: int = 0
    size: int = 0
    mean: str = '0'
    shift: str = '1'
    datatype: str = 'float32'
    scaleup: int = 0

    def _post_init(self) -> None:
        self._set_operators(
            [
                preprocessing.Resize(width=self.width, height=self.height, size=self.size),
                preprocessing.ToTensor(),
                preprocessing.PermuteChannels(input_layout='NHWC', output_layout='NCHW'),
                preprocessing.TypeCast(datatype='float32'),
                preprocessing.LinearScaling(mean=self.mean, shift=self.shift),
            ]
        )
        self._norm = self._operators[-1]
        return super()._post_init()

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
        context.resize_status = types.ResizeMode.STRETCH

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        _ensure_len3 = lambda seq: list(seq) + [seq[0]] * (3 - len(seq))
        mean = _ensure_len3(self._effective_mean)
        std = _ensure_len3(self._effective_std)
        s = [f'{float(x):.6f}'.rstrip('0') for x in std]
        m = [f'{float(x):.6f}'.rstrip('0') for x in mean]
        quant = ''
        if (self._scale and self._scale[0] != 1) or (self._zero and self._zero[0] != 0):
            quant = f';quant_scale:{self._scale[0]};quant_zeropoint:{self._zero[0]}'

        mean = ",".join(map(str, m))
        std = ",".join(map(str, s))
        gst.axtransform(
            lib='libtransform_resize_cl.so',
            options=(
                f'width:{self.width};height:{self.height};'
                + f'to_tensor:1;mean:{mean};std:{std}{quant}'
            ),
        )


class OpenCLColorConvertResizeToTensorAndLinearScaling(preprocessing.CompositePreprocess):
    width: int = 0
    height: int = 0
    size: int = 0
    mean: str = '0'
    shift: str = '1'
    format: str = 'rgb'
    datatype: str = 'float32'
    scaleup: int = 0

    def _post_init(self) -> None:
        self._set_operators(
            [
                custom_preprocessing.ConvertColorInput(self.format),
                preprocessing.Resize(width=self.width, height=self.height, size=self.size),
                preprocessing.ToTensor(),
                preprocessing.PermuteChannels(input_layout='NHWC', output_layout='NCHW'),
                preprocessing.TypeCast(datatype='float32'),
                preprocessing.LinearScaling(mean=self.mean, shift=self.shift),
            ]
        )
        self._norm = self._operators[-1]
        return super()._post_init()

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
        context.resize_status = types.ResizeMode.STRETCH

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        _ensure_len3 = lambda seq: list(seq) + [seq[0]] * (3 - len(seq))
        mean = _ensure_len3(self._effective_mean)
        std = _ensure_len3(self._effective_std)
        s = [f'{float(x):.6f}'.rstrip('0') for x in std]
        m = [f'{float(x):.6f}'.rstrip('0') for x in mean]
        quant = ''
        if (self._scale and self._scale[0] != 1) or (self._zero and self._zero[0] != 0):
            quant = f';quant_scale:{self._scale[0]};quant_zeropoint:{self._zero[0]}'

        mean = ",".join(map(str, m))
        std = ",".join(map(str, s))
        fmt = add_alpha_channel(self.format)
        gst.axtransform(
            lib='libtransform_resize_cl.so',
            options=(
                f'width:{self.width};height:{self.height};'
                + f'to_tensor:1;mean:{mean};std:{std}{quant};format:{fmt}'
            ),
        )


class OpenCLetterBoxColorConvertToTensorAndNormalize(preprocessing.CompositePreprocess):
    width: int = 0
    height: int = 0
    scaleup: bool = True
    half_pixel_centers: bool = False
    pad_val: int = 114
    mean: str = '0'
    std: str = '1'
    format: str = 'rgba'
    datatype: str = 'float32'

    def _post_init(self) -> None:
        self._set_operators(
            [
                custom_preprocessing.ConvertColorInput(self.format),
                custom_preprocessing.Letterbox(
                    width=self.width,
                    height=self.height,
                    scaleup=self.scaleup,
                    half_pixel_centers=self.half_pixel_centers,
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
        return super()._post_init()

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
        if self.scaleup:
            context.resize_status = types.ResizeMode.LETTERBOX_FIT
        else:
            context.resize_status = types.ResizeMode.LETTERBOX_CONTAIN

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        _ensure_len3 = lambda seq: list(seq) + [seq[0]] * (3 - len(seq))
        mean = _ensure_len3(self._effective_mean)
        std = _ensure_len3(self._effective_std)
        scale = _ensure_len3(self._scale)
        zero = _ensure_len3(self._zero)
        m = [f'{float(x):.6f}'.rstrip('0') for x in mean]
        s = [f'{float(x):.6f}'.rstrip('0') for x in std]
        mean = ",".join(map(str, m))
        std = ",".join(map(str, s))
        fmt = add_alpha_channel(self.format)
        gst.axtransform(
            lib='libtransform_resize_cl.so',
            options=(
                f'width:{self.width};height:{self.height};padding:{self.pad_val};letterbox:1;scale_up:{int(self.scaleup)};'
                + f'to_tensor:1;mean:{mean};std:{std};quant_scale:{float(scale[0])};quant_zeropoint:{float(zero[0])};format:{fmt}'
            ),
        )


class OpenCLetterBoxToTensorAndNormalize(preprocessing.CompositePreprocess):
    width: int = 0
    height: int = 0
    scaleup: bool = True
    half_pixel_centers: bool = False
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
                    half_pixel_centers=self.half_pixel_centers,
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
        return super()._post_init()

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
        if self.scaleup:
            context.resize_status = types.ResizeMode.LETTERBOX_FIT
        else:
            context.resize_status = types.ResizeMode.LETTERBOX_CONTAIN

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        _ensure_len3 = lambda seq: list(seq) + [seq[0]] * (3 - len(seq))
        mean = _ensure_len3(self._effective_mean)
        std = _ensure_len3(self._effective_std)
        scale = _ensure_len3(self._scale)
        zero = _ensure_len3(self._zero)
        m = [f'{float(x):.6f}'.rstrip('0') for x in mean]
        s = [f'{float(x):.6f}'.rstrip('0') for x in std]
        mean = ",".join(map(str, m))
        std = ",".join(map(str, s))
        gst.axtransform(
            lib='libtransform_resize_cl.so',
            options=(
                f'width:{self.width};height:{self.height};padding:{self.pad_val};letterbox:1;scale_up:{int(self.scaleup)};'
                + f'to_tensor:1;mean:{mean};std:{std};quant_scale:{float(scale[0])};quant_zeropoint:{float(zero[0])}'
            ),
        )


class OpenCLetterBoxColorConvertToTensorAndLinearScaling(preprocessing.CompositePreprocess):
    width: int = 0
    height: int = 0
    scaleup: bool = True
    half_pixel_centers: bool = False
    pad_val: int = 114
    mean: str = '0'
    shift: str = '1'
    format: str = 'rgba'
    datatype: str = 'float32'

    def _post_init(self) -> None:
        ops = [
            custom_preprocessing.Letterbox(
                width=self.width,
                height=self.height,
                scaleup=self.scaleup,
                half_pixel_centers=self.half_pixel_centers,
                pad_val=self.pad_val,
            ),
            preprocessing.ToTensor(),
            preprocessing.PermuteChannels(input_layout='NHWC', output_layout='NCHW'),
            preprocessing.TypeCast(datatype='float32'),
            preprocessing.LinearScaling(mean=self.mean, shift=self.shift),
        ]
        if format:
            ops.insert(0, custom_preprocessing.ConvertColorInput(self.format))

        self._set_operators(ops)
        self._norm = self._operators[-1]
        # These values are defaults, I do not believe they will change but or be passed
        # in as args, but I am leaving them here for now
        self._scale = [1.0 / 255]
        self._zero = [0]
        return super()._post_init()

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
        if self.scaleup:
            context.resize_status = types.ResizeMode.LETTERBOX_FIT
        else:
            context.resize_status = types.ResizeMode.LETTERBOX_CONTAIN

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        _ensure_len3 = lambda seq: list(seq) + [seq[0]] * (3 - len(seq))
        mean = _ensure_len3(self._effective_mean)
        shift = _ensure_len3(self._effective_std)
        s = [f'{float(x):.6f}'.rstrip('0') for x in shift]
        m = [f'{float(x):.6f}'.rstrip('0') for x in mean]
        quant = ''
        if (self._scale and self._scale[0] != 1) or (self._zero and self._zero[0] != 0):
            quant = f';quant_scale:{self._scale[0]};quant_zeropoint:{self._zero[0]}'

        out = ''
        if self.format:
            out = f';format:{add_alpha_channel(self.format)}'
        gst.axtransform(
            lib='libtransform_resize_cl.so',
            options=(
                f'width:{self.width};height:{self.height};padding:{self.pad_val};letterbox:1;scale_up:{int(self.scaleup)};'
                + f'to_tensor:1;mean:{",".join(m)};std:{",".join(s)}{quant}'
                + out
            ),
        )


class OpenCLetterBoxColorConvertToTensor(preprocessing.CompositePreprocess):
    width: int = 0
    height: int = 0
    scaleup: bool = True
    half_pixel_centers: bool = False
    pad_val: int = 114
    mean: str = '0'
    std: str = '255'
    format: str = 'rgba'
    datatype: str = 'float32'

    def _post_init(self) -> None:
        ops = [
            custom_preprocessing.Letterbox(
                width=self.width,
                height=self.height,
                scaleup=self.scaleup,
                half_pixel_centers=self.half_pixel_centers,
                pad_val=self.pad_val,
            ),
            preprocessing.ToTensor(),
            preprocessing.PermuteChannels(input_layout='NHWC', output_layout='NCHW'),
            preprocessing.TypeCast(datatype='float32'),
        ]

        if self.format:
            ops.insert(0, custom_preprocessing.ConvertColorInput(self.format))

        self._set_operators(ops)
        self._norm = preprocessing.Normalize(mean=[0.0], std=[1.0 / 255.0])
        return super()._post_init()

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
        if self.scaleup:
            context.resize_status = types.ResizeMode.LETTERBOX_FIT
        else:
            context.resize_status = types.ResizeMode.LETTERBOX_CONTAIN

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        _ensure_len3 = lambda seq: list(seq) + [seq[0]] * (3 - len(seq))
        scale = _ensure_len3(self._scale)
        zero = _ensure_len3(self._zero)
        m = [f'{float(x):.6f}'.rstrip('0') for x in _ensure_len3(self._effective_mean)]
        s = [f'{float(x):.6f}'.rstrip('0') for x in _ensure_len3(self._effective_std)]
        mean = ",".join(map(str, m))
        std = ",".join(map(str, s))
        out = ''
        if self.format:
            fmt = add_alpha_channel(self.format)
            out = f';format:{fmt}'
        gst.axtransform(
            lib='libtransform_resize_cl.so',
            options=(
                f'width:{self.width};height:{self.height};padding:{self.pad_val};letterbox:1;scale_up:{int(self.scaleup)};'
                + f'to_tensor:1;mean:{mean};std:{std};quant_scale:{float(scale[0])};quant_zeropoint:{float(zero[0])};format:{fmt}'
                + out
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
        cast = gst_builder.Builder(
            gst.hw_config, gst.tiling, gst.default_queue_max_size_buffers, gst.which_cl
        )
        norm = gst_builder.Builder(
            gst.hw_config, gst.tiling, gst.default_queue_max_size_buffers, gst.which_cl
        )
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


def which_simd():
    return 'avx2' if platform.processor() == 'x86_64' else 'neon'


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
        compiled_model_dir: Path | None,
        task_graph,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, compiled_model_dir, task_graph
        )
        if not self._scale:
            self._scale = [1.0 / 255]
            self._zero = [0]

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        quant = ''
        if (self._scale and self._scale[0] != 1) or (self._zero and self._zero[0] != 0):
            quant = f';quant_scale:{self._scale[0]};quant_zeropoint:{self._zero[0]}'

        gst.axtransform(
            lib='libtransform_totensor.so',
            options='type:int8',
        )
        m = [f'{float(x):.6f}'.rstrip('0') for x in self._effective_mean]
        s = [f'{float(x):.6f}'.rstrip('0') for x in self._effective_std]
        gst.axinplace(
            lib='libinplace_normalize.so',
            mode='write',
            options=f'mean:{",".join(m)};std:{",".join(s)};simd:{which_simd()}{quant}',
        )


class ToTensorAndNoNormalise(preprocessing.CompositePreprocess):
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
            ]
        )
        self._norm = preprocessing.Normalize(mean=[0.0], std=[1.0 / 255.0])

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
        if (self._scale and any(x != self._scale[0] for x in self._scale[1:])) or (
            self._zero and any(x != self._zero[0] for x in self._zero[1:])
        ):
            raise ValueError("axinplace_normalize.write only supports uniform quantization params")

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        _ensure_len3 = lambda seq: list(seq) + [seq[0]] * (3 - len(seq))
        quant = ''
        if (self._scale and self._scale[0] != 1) or (self._zero and self._zero[0] != 0):
            quant = f';quant_scale:{self._scale[0]};quant_zeropoint:{self._zero[0]}'

        gst.axtransform(
            lib='libtransform_totensor.so',
            options='type:int8',
        )

        m = [f'{float(x):.6f}'.rstrip('0') for x in _ensure_len3(self._effective_mean)]
        s = [f'{float(x):.6f}'.rstrip('0') for x in _ensure_len3(self._effective_std)]
        gst.axinplace(
            lib='libinplace_normalize.so',
            mode='write',
            options=f'mean:{",".join(m)};std:{",".join(s)};simd:{which_simd()}{quant}',
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
        if (self._scale and any(x != self._scale[0] for x in self._scale[1:])) or (
            self._zero and any(x != self._zero[0] for x in self._zero[1:])
        ):
            raise ValueError("axinplace_normalize.write only supports uniform quantization params")

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        quant = ''
        if (self._scale and self._scale[0] != 1) or (self._zero and self._zero[0] != 0):
            quant = f';quant_scale:{self._scale[0]};quant_zeropoint:{self._zero[0]}'

        gst.axtransform(
            lib='libtransform_totensor.so',
            options='type:int8',
        )
        m = [f'{float(x):.6f}'.rstrip('0') for x in self._effective_mean]
        s = [f'{float(x):.6f}'.rstrip('0') for x in self._effective_std]
        gst.axinplace(
            lib='libinplace_normalize.so',
            mode='write',
            options=f'mean:{",".join(m)};std:{",".join(s)};simd:{which_simd()}{quant}',
        )


class LetterboxToTensorAndNormalise(preprocessing.CompositePreprocess):
    height: int = '0'
    width: int = '0'
    mean: str = '0'
    std: str = '1'
    datatype: str = 'float32'
    scaleup: bool = True
    half_pixel_centers: bool = False

    def _post_init(self):
        self._set_operators(
            [
                custom_preprocessing.Letterbox(
                    height=self.height,
                    width=self.width,
                    scaleup=self.scaleup,
                    half_pixel_centers=self.half_pixel_centers,
                ),
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
        if (self._scale and any(x != self._scale[0] for x in self._scale[1:])) or (
            self._zero and any(x != self._zero[0] for x in self._zero[1:])
        ):
            raise ValueError("axinplace_normalize.write only supports uniform quantization params")

        inspect_resize_status(context)
        if self.scaleup:
            context.resize_status = types.ResizeMode.LETTERBOX_FIT
        else:
            context.resize_status = types.ResizeMode.LETTERBOX_CONTAIN

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        quant = ''
        if (self._scale and self._scale[0] != 1) or (self._zero and self._zero[0] != 0):
            quant = f';quant_scale:{self._scale[0]};quant_zeropoint:{self._zero[0]}'

        gst.axtransform(
            lib='libtransform_resize.so',
            options=f'width:{self.width};height:{self.height};padding:114;to_tensor:1;letterbox:1;scale_up:{int(self.scaleup)}',
        )
        m = [f'{float(x):.6f}'.rstrip('0') for x in self._effective_mean]
        s = [f'{float(x):.6f}'.rstrip('0') for x in self._effective_std]
        gst.axinplace(
            lib='libinplace_normalize.so',
            mode='write',
            options=f'mean:{",".join(m)};std:{",".join(s)};simd:{which_simd()}{quant}',
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

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        _ensure_len3 = lambda seq: list(seq) + [seq[0]] * (3 - len(seq))
        mean = _ensure_len3(self._effective_mean)
        std = _ensure_len3(self._effective_std)
        scale = _ensure_len3(self._scale)
        zero = _ensure_len3(self._zero)
        m = [f'{float(x):.6f}'.rstrip('0') for x in mean]
        s = [f'{float(x):.6f}'.rstrip('0') for x in std]
        mean = ",".join(map(str, m))
        std = ",".join(map(str, s))
        gst.axtransform(
            lib='libtransform_normalize_cl.so',
            options=f'to_tensor:1;mean:{mean};std:{std};quant_scale:{float(scale[0])};quant_zeropoint:{float(zero[0])}',
        )


class OpenCLVideoFlipAndColorConvert(preprocessing.CompositePreprocess):
    method: config.VideoFlipMethod = config.VideoFlipMethod.clockwise
    format: str = 'rgb'

    def _post_init(self):
        self._enforce_member_type('method')
        self._set_operators(
            [
                custom_preprocessing.ConvertColorInput(self.format),
                custom_preprocessing.VideoFlip(method=self.method),
            ]
        )

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        fmt = add_alpha_channel(self.format)
        gst.axtransform(
            lib="libtransform_colorconvert_cl.so",
            options=f"format:{add_alpha_channel(fmt)};flip_method:{self.method.name.replace('_', '-')}",
        )

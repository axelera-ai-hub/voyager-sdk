# Copyright Axelera AI, 2024
# Custom pre-processing operators
from __future__ import annotations

from pathlib import Path
from typing import List, Tuple, Union

from PIL import Image, ImageOps
import cv2
import numpy as np

from axelera import types

from .. import gst_builder, logging_utils
from ..torch_utils import torch
from .base import PreprocessOperator, builtin
from .context import PipelineContext
from .utils import insert_color_convert, inspect_resize_status

if not hasattr(Image, "Resampling"):  # if Pillow<9.0
    Image.Resampling = Image

LOG = logging_utils.getLogger(__name__)


@builtin
class PermuteChannels(PreprocessOperator):
    input_layout: types.TensorLayout
    output_layout: types.TensorLayout

    def _post_init(self):
        super()._post_init()
        self._enforce_member_type('input_layout')
        self._enforce_member_type('output_layout')
        if self.input_layout != self.output_layout:
            self._dimchg = _get_dimchg(self.input_layout, self.output_layout)

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        if self.input_layout == self.output_layout:
            return
        raise NotImplementedError("PermuteChannels is not implemented for gst pipeline")

    def exec_torch(self, image: torch.Tensor) -> torch.Tensor:
        if not isinstance(image, torch.Tensor):
            raise TypeError(f"Unsupported input type: {type(image)}")
        if image.ndim < 3:
            # all supported layouts are HW, so all we can actually do here is
            # add a channel dimension (should this be in ToTensor?)
            image = image.unsqueeze(0)
        elif self.input_layout != self.output_layout:
            _in, _out = self.input_layout.name, self.output_layout.name
            if image.ndim < 4:
                _in, _out = _in.replace('N', ''), _out.replace('N', '')
            axes = [_in.index(x) for x in _out]
            image = image.permute(*axes).contiguous()
        return image


_dimchgs = {
    "NCHW:NHWC": '2:0',
    "NHWC:NCHW": '0:2',
    "CHWN:NCHW": '0:3',
    "NCHW:CHWN": '3:0',
}


def _get_dimchg(input_layout: types.TensorLayout, output_layout: types.TensorLayout) -> str:
    try:
        return _dimchgs[f'{input_layout.name}:{output_layout.name}']
    except KeyError:
        raise ValueError(
            f"Unsupported input/output layouts: {input_layout.name}/{output_layout.name}"
        ) from None


@builtin
class Letterbox(PreprocessOperator):
    height: int
    width: int
    scaleup: bool = True
    half_pixel_centers: bool = False
    pad_val: int = 114
    # Never crop, always end up in scaleup==True if no image_width or height specified
    image_width: int = 1000000
    image_height: int = 1000000

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        where: str,
        compiled_model_dir: Path,
    ):
        # TODO: implement SQUISH and LETTERBOX_CONTAIN mode in torch and verify accuracy on YOLOs
        self.task_name = task_name
        inspect_resize_status(context)
        if self.scaleup:
            context.resize_status = types.ResizeMode.LETTERBOX_FIT
        else:
            context.resize_status = types.ResizeMode.LETTERBOX_CONTAIN

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        if gst.getconfig().opencl:
            gst.axtransform(
                lib='libtransform_resize_cl.so',
                options=f'width:{self.width};height:{self.height};padding:{self.pad_val};letterbox:1;scale_up:{int(self.scaleup)}',
            )
        else:
            gst.axtransform(
                lib='libtransform_resize.so',
                options=f'width:{self.width};height:{self.height};padding:{self.pad_val};letterbox:1;scale_up:{int(self.scaleup)}',
            )

    def exec_torch(self, image: types.Image) -> types.Image:
        result = self._letterbox(
            image.aspil(),
            (self.height, self.width),
            scaleup=self.scaleup,
            color=(self.pad_val, self.pad_val, self.pad_val),
        )[0]
        return types.Image.frompil(result, image.color_format)

    def _letterbox(
        self,
        im: Image,
        new_shape: Tuple[int, int],
        color: Tuple[int, int, int],
        rect: bool = False,
        scaleup: bool = True,
        stride: int = 32,
    ):
        """Resize and pad image while meeting stride-multiple constraints.
        new_shape: (height, width)
        """

        shape = im.size[::-1]  # shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        if not scaleup:  # only scale down, do not scale up (for better val mAP)
            r = min(r, 1.0)

        # Compute padding
        new_unpad = int(round(shape[0] * r)), int(round(shape[1] * r))  # h, w
        dh, dw = (
            new_shape[0] - new_unpad[0],
            new_shape[1] - new_unpad[1],
        )  # hw padding

        # minimum rectangle
        # We don't support this mode due to the requirement of dynamic input dim
        # TODO: we should support detection with a fixed rectangle size which may increase performance
        if rect:
            dh, dw = np.mod(dh, stride), np.mod(dw, stride)  # wh padding

        dh /= 2  # divide padding into 2 sides
        dw /= 2

        if shape != new_unpad:  # resize
            if self.half_pixel_centers:
                im_resized = cv2.resize(
                    np.array(im), new_unpad[::-1], interpolation=cv2.INTER_LINEAR
                )
                im = Image.fromarray(im_resized)
            else:
                im = im.resize(new_unpad[::-1], Image.Resampling.BILINEAR)

        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
        if im.mode == 'RGB':
            im = ImageOps.expand(im, border=(left, top, right, bottom), fill=color)
        elif im.mode == 'L':
            if isinstance(color, tuple):
                color = color[0]  # Use the first value of the tuple for grayscale
            im = ImageOps.expand(im, border=(left, top, right, bottom), fill=color)
        else:
            raise ValueError(f"Unsupported image mode: {im.mode}")

        return im, r, (dw, dh)


_supported_formats = [
    'RGB2GRAY',
    'GRAY2RGB',
    'RGB2BGR',
    'BGR2RGB',
    'BGR2GRAY',
    'GRAY2BGR',
]


@builtin
class ConvertColor(PreprocessOperator):
    """TODO: merge ConvertChannel and ConvertColor, format follow
    OpenCV cvtColor like RGB2BGR, YUV2RGB, this means developer must know
    exact input and output formats."""

    format: str

    def _post_init(self):
        super()._post_init()
        _, output_format = self.format.split('2')
        self.output_format = output_format.upper()
        if self.format.upper() not in _supported_formats:
            raise ValueError(f"Unsupported conversion: {self.format}")

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
        input_format, output_format = self.format.split('2')
        input_format = input_format.upper()
        output_format = output_format.upper()

        format_map = {
            'RGB': types.ColorFormat.RGB,
            'BGR': types.ColorFormat.BGR,
            'GRAY': types.ColorFormat.GRAY,
        }

        if context.color_format != format_map[input_format]:
            raise ValueError(
                f"Input color format mismatch. Expected {input_format}, but got {context.color_format}"
            )

        self.input_format = format_map[input_format]
        context.color_format = format_map[output_format]

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        vaapi = gst.getconfig() is not None and gst.getconfig().vaapi
        opencl = gst.getconfig() is not None and gst.getconfig().opencl
        format = f'{self.format.split("2")[1].lower()}a'
        insert_color_convert(gst, vaapi, opencl, format)

    def exec_torch(
        self, image: Union[torch.Tensor, types.Image]
    ) -> Union[torch.Tensor, types.Image]:
        if isinstance(image, (types.Image, Image.Image)):
            result = image
        elif isinstance(image, np.ndarray):
            result = types.Image.fromarray(image, self.input_format)
        elif isinstance(image, torch.Tensor):
            return _convert_color_torch(image, self.format)
        else:
            raise TypeError(f"Unsupported input type: {type(image)}")

        new_array = result.asarray(self.output_format)
        result.update(new_array, color_format=self.output_format)
        return result


@builtin
class ConvertColorInput(PreprocessOperator):
    """
    This is the color convert from the Input operator. It has been moved into
    a separate operator to allow for more flexibility in the pipeline.
    """

    format: types.ColorFormat = types.ColorFormat.RGB

    def _post_init(self):
        self._enforce_member_type('format')

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        vaapi = gst.getconfig() is not None and gst.getconfig().vaapi
        opencl = gst.getconfig() is not None and gst.getconfig().opencl

        insert_color_convert(gst, vaapi, opencl, f'{self.format.name.lower()}a')

    def exec_torch(
        self, image: Union[torch.Tensor, types.Image]
    ) -> Union[torch.Tensor, types.Image]:
        return image


@builtin
class FaceAlign(PreprocessOperator):
    keypoints_submeta_key: str
    width: int = 0
    height: int = 0
    padding: float = 0.0
    template_keypoints_x: List[float] = []
    template_keypoints_y: List[float] = []

    def _post_init(self):
        if len(self.template_keypoints_x) != len(self.template_keypoints_y):
            raise ValueError("Number of template keypoints x and y must be equal")

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        where: str,
        compiled_model_dir: Path,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, where, compiled_model_dir
        )

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        gst.axtransform(
            lib='libtransform_facealign.so',
            options=f'master_meta:{self._where};'
            f'keypoints_submeta_key:{self.keypoints_submeta_key};'
            f'width:{self.width};'
            f'height:{self.height};'
            f'padding:{self.padding};'
            f'template_keypoints_x:{",".join(map(str, self.template_keypoints_x))};'
            f'template_keypoints_y:{",".join(map(str, self.template_keypoints_y))};',
        )

    def exec_torch(self, image):
        # TODO: SDK-4880
        raise NotImplementedError("FaceAlign is not yet implemented for torch pipeline")


@builtin
class Perspective(PreprocessOperator):
    camera_matrix: list[float] | str
    format: str = 'rgb'

    def _post_init(self):
        if self.camera_matrix is None:
            raise ValueError("You must specify camera matrix")
        if isinstance(self.camera_matrix, str):
            if len(self.camera_matrix.split(",")) != 9:
                raise ValueError("Number of camera matrix values must be 9")
            self.camera_matrix = [float(item) for item in self.camera_matrix.split(',')]
        elif isinstance(self.camera_matrix, list):
            if len(self.camera_matrix) != 9:
                raise ValueError("Number of camera matrix values must be 9")
        else:
            raise ValueError(
                "Camera matrix must be either a comma-separated string or a list of floats"
            )

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        opencl = gst.getconfig() is not None and gst.getconfig().opencl
        matrix = np.array(self.camera_matrix).reshape(3, 3)
        matrix = np.linalg.inv(matrix)
        matrix = ','.join(map(str, matrix.flatten()))
        if opencl:
            gst.axtransform(
                lib='libtransform_perspective_cl.so',
                options=f'matrix:{matrix}',
            )
        else:
            vaapi = False  # Gst perspective element doesn't respect from offsets and strides in VAsurfaces (GstVideoMeta)
            insert_color_convert(gst, vaapi, opencl, f"{self.format.lower()}a")
            gst.perspective(matrix=matrix)

    def exec_torch(self, image):
        matrix = np.array(self.camera_matrix).reshape(3, 3)
        transformed_image = cv2.warpPerspective(image.asarray(), matrix, image.size)
        return types.Image.fromarray(transformed_image)


@builtin
class CameraUndistort(PreprocessOperator):
    fx: float
    fy: float
    cx: float
    cy: float
    distort_coefs: List[float]
    bgra_out: bool = False

    def _post_init(self):
        if self.distort_coefs is None:
            raise ValueError("You must specify camera matrix")
        if len(self.distort_coefs) != 5:
            raise ValueError("Number of distort coeficients must be 5")

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        if gst.getconfig().opencl:
            gst.axtransform(
                lib='libtransform_barrelcorrect_cl.so',
                options=f'camera_props:{self.fx},{self.fy},{self.cx},{self.cy};distort_coefs:{",".join(str(coef) for coef in self.distort_coefs)};bgra_out:{int(self.bgra_out)}',
            )

        else:
            # for non OpenCL path camera matrix need to be denormalized in yaml file
            gst.videoconvert()
            gst.capsfilter(caps=f"video/x-raw,format={'BGRA' if self.bgra_out == 1 else 'RGBA'}")
            config = f'<?xml version=\"1.0\"?><opencv_storage><cameraMatrix type_id=\"opencv-matrix\"><rows>3</rows><cols>3</cols><dt>f</dt><data>{self.fx} 0. {self.cx} 0. {self.fy} {self.cy} 0. 0. 1.</data></cameraMatrix><distCoeffs type_id=\"opencv-matrix\"><rows>5</rows><cols>1</cols><dt>f</dt><data>{" ".join(str(coef) for coef in self.distort_coefs)}</data></distCoeffs></opencv_storage>'
            gst.cameraundistort(settings=config)

    def exec_torch(self, image):
        camera_matrix = np.array(
            [
                [self.fx * image.size[0], 0.0, self.cx * image.size[0]],
                [0.0, self.fy * image.size[1], self.cy * image.size[1]],
                [0.0, 0.0, 1.0],
            ]
        )
        camera_dist = np.array(self.distort_coefs, dtype=np.float64)
        new_camera, _ = cv2.getOptimalNewCameraMatrix(
            camera_matrix, camera_dist, image.size, 1, image.size
        )
        mapx, mapy = cv2.initUndistortRectifyMap(
            camera_matrix, camera_dist, None, new_camera, image.size, 5
        )
        new_image = cv2.remap(image.asarray(), mapx, mapy, cv2.INTER_LINEAR)

        return types.Image.fromarray(new_image)


def _convert_color_torch(img, format):
    format = getattr(cv2, f"COLOR_{format.upper()}")
    img = img.cpu().numpy().transpose(1, 2, 0)
    img = cv2.cvtColor(img, format)
    if len(img.shape) == 2:  # Grayscale output has 2 dimensions (H,W)
        img = img[np.newaxis, :, :]
    else:
        img = img.transpose(2, 0, 1)
    return torch.from_numpy(img)


@builtin
class ContrastNormalize(PreprocessOperator):
    """Also known as contrast stretching"""

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        gst.axtransform(
            lib='$AX_SUBPLUGIN_PATH/libtransform_contrastnormalize.so',
        )

    def exec_torch(
        self, image: Union[torch.Tensor, types.Image]
    ) -> Union[torch.Tensor, types.Image]:
        import torchvision.transforms.functional as F

        if isinstance(image, types.Image):
            # If the input is a PIL image, convert it to a tensor and normalize it
            tensor_image = F.to_tensor(image.aspil())
            min_value = torch.min(tensor_image)
            max_value = torch.max(tensor_image)
            normalized_tensor = (tensor_image - min_value) / (max_value - min_value)
            # to_pil_image scale the values back to 0-255 automaticlly
            normalized_image = F.to_pil_image(normalized_tensor)
            normalized_image = types.Image.frompil(normalized_image, image.color_format)
        elif isinstance(image, torch.Tensor):
            # If the input is a tensor, normalize it directly
            min_value = torch.min(image)
            max_value = torch.max(image)
            normalized_image = (image - min_value) / (max_value - min_value)
        else:
            raise TypeError("Input should be a PIL image or a tensor.")
        return normalized_image

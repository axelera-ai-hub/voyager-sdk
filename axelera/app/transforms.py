# Copyright Axelera AI, 2023
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction
from typing import TYPE_CHECKING, List, Type, get_type_hints

from axelera import types

from . import config, logging_utils, operators

if TYPE_CHECKING:
    from . import config

builtin_transformers = []
LOG = logging_utils.getLogger(__name__)


@dataclass
class Transformer:
    func: callable
    classes: List[Type[operators.AxOperator]]
    priority: int
    hardware_caps: List[str]
    '''List of hardware capabilities required to run this transformer.

    e.g. 'vaapi', 'opencl'.
    '''

    def __call__(self, ops):
        n = 0
        matched = False
        trace = LOG.trace

        def out(ops):
            return '\n'.join(f'  {op}' for op in ops)

        while n < len(ops) - len(self.classes) + 1:
            if all(isinstance(o, cls) for o, cls in zip(ops[n:], self.classes)):
                m = n + len(self.classes)
                matched = True
                got = self.func(*ops[n:m])
                if got != ops[n:m]:
                    trace(f"{self.func.__name__} transformed:\n{out(ops[n:m])}\n to:\n{out(got)}")
                else:
                    trace(f"{self.func.__name__} skipped:\n{out(ops[n:m])}")
                ops[n:m] = got
            n += 1


def builtin(transformer: Transformer):
    '''Mark a transformer as builtin.

    builtin transformers are automatically run on all preprocess operators.
    '''
    builtin_transformers.append(transformer)
    return transformer


def transformer(*, priority=0, hardware_caps=[]):
    '''Mark a function as a transformer.

    The function should take as many arguments as operators it intends to manipulate,
    the type annotations are used to do matching.

    For example:

    @tranformer(priority=1)
    def match_to_tensor_and_cast(type_cast: operators.TypeCast, to_tensor: operators.ToTensor):
        return [operators.mega.TypeCastAndNormalize(type_cast.datatype, to_tensor.mean,
                to_tensor.std, to_tensor.tensor_layout)]

    The return value of the transformer replaces the operators that it matched.  Thus to cancel a
    transformer, return the original operators.  (e.g. in the above example return [type_cast, to_tensor])

    `priority` is used to determine the order that transforms are performed.  Transforms are ordered
    lowest to highest in numerical order.

    `hardware_caps` is a list of hardware capabilities required to run this transformer.
    e.g. 'vaapi', 'opencl'.

    '''
    return lambda fn: Transformer(fn, list(get_type_hints(fn).values()), priority, hardware_caps)


def run_all_transformers(
    ops: List[operators.AxOperator],
    transformers: List[Transformer] = builtin_transformers,
    hardware_caps: config.HardwareCaps = config.HardwareCaps.NONE,
):
    '''Run all transformers on the list of operators.

    By default all transformers marked with `@builtin` are run, but a custom list of transformers
    can be provided.

    Only transformers that have all their hardware capabilities available will be run.
    e.g. if a transformer has t.hardware_caps=['vaapi'] then it will only be run if
    hardware_caps.vaapi is True.

    '''
    available = [t for t in transformers if all(hardware_caps.enabled(x) for x in t.hardware_caps)]
    # prefer low priority transformers, and for those with equal priority choose those with
    # hardware_caps over those that don't (i.e. prefer accelerated versions)
    transformers = sorted(available, key=lambda t: (t.priority, -len(t.hardware_caps)))
    for t in transformers:
        t(ops)


@builtin
@transformer(priority=5)
def composite_expansion(composite: operators.preprocessing.CompositePreprocess):
    '''Break up composite operators so that they can be optimised by later transformers.'''
    return composite._operators


@builtin
@transformer(priority=30)
def resize_and_convert_transform(resize: operators.Resize, convert: operators.ConvertColor):
    '''Example mega operator to merge a resize and color convert. This should be used
    as the basis for a VAAPI mega operator.
    '''
    if resize.size != 0:
        width, height = resize.size, resize.size
    else:
        width, height = resize.width, resize.height
    return [operators.mega.ResizeAndConvert(width=width, height=height, format=convert.format)]


@builtin
@transformer(priority=60)
def cropped_resize_with_extra_crop(resize: operators.Resize, center_crop: operators.CenterCrop):
    '''Mega operator that replaces a resize and center crop.'''
    return [
        operators.mega.CroppedResizeWithExtraCrop(
            width=resize.width,
            height=resize.height,
            size=resize.size,
            hcrop=resize.effective_width - center_crop.width,
            vcrop=resize.effective_height - center_crop.height,
        )
    ]


@builtin
@transformer(priority=60, hardware_caps=['opencl'])
def opencl_colorconvert_with_perspective(
    perspective: operators.custom_preprocessing.Perspective, convert: operators.ConvertColorInput
):
    '''Mega operators for opencl perspective transformation'''
    return [
        operators.mega.OpenCLPerspectiveTransform(
            camera_matrix=perspective.camera_matrix, format=convert.format.name.lower()
        )
    ]


@builtin
@transformer(priority=60, hardware_caps=['opencl'])
def opencl_colorconvert_with_cameraundistort(
    barrel: operators.custom_preprocessing.CameraUndistort,
    convert: operators.custom_preprocessing.ConvertColorInput,
):
    '''Mega operator for OpenCL barrel distortion correction'''
    # Fusing posibble only if output format is rgb or bgr
    if convert.format == types.ColorFormat.RGB or convert.format == types.ColorFormat.BGR:
        return [
            operators.mega.OpenCLBarrelDistortionCorrection(
                fx=barrel.fx,
                fy=barrel.fy,
                cx=barrel.cx,
                cy=barrel.cy,
                distort_coefs=barrel.distort_coefs,
                format=convert.format.name.lower(),
            )
        ]
    else:
        return [barrel, convert]


@builtin
@transformer(priority=50, hardware_caps=['opencl'])
def opencl_cropped_resize_with_extra_crop(
    resize: operators.Resize, center_crop: operators.CenterCrop
):
    '''Mega operator that replaces a resize and center crop with an OpenCL one.'''
    return [
        operators.mega.OpenCLCroppedResizeWithExtraCrop(
            width=resize.width,
            height=resize.height,
            size=resize.size,
            hcrop=resize.effective_width - center_crop.width,
            vcrop=resize.effective_height - center_crop.height,
        )
    ]


@builtin
@transformer(priority=20, hardware_caps=['opencl'])
def opencl_cropped_resize_with_extra_crop_and_2norms(
    resize: operators.Resize,
    center_crop: operators.CenterCrop,
    totensor: operators.ToTensor,
    permute: operators.PermuteChannels,
    type_cast: operators.TypeCast,
    norm1: operators.Normalize,
    norm2: operators.Normalize,
):
    '''Mega operator that replaces a resize and center crop with an OpenCL one.'''
    if (
        permute.input_layout == types.TensorLayout.NHWC
        and permute.output_layout == types.TensorLayout.NCHW
        and type_cast.datatype == 'float32'
        and norm1.std_values == [Fraction(255, 1)]
        and norm1.mean_values == [0]
    ):
        return [
            operators.mega.OpenCLCroppedResizeWithExtraCropAndNormalize(
                width=resize.width,
                height=resize.height,
                size=resize.size,
                hcrop=resize.effective_width - center_crop.width,
                vcrop=resize.effective_height - center_crop.height,
                mean=norm2.mean,
                std=norm2.std,
            )
        ]
    return [resize, center_crop, totensor, permute, type_cast, norm1, norm2]


@builtin
@transformer(priority=40, hardware_caps=['opencl'])
def opencl_cropped_resize_with_extra_crop_and_color_convert(
    convert: operators.custom_preprocessing.ConvertColorInput,
    resize: operators.Resize,
    center_crop: operators.CenterCrop,
):
    '''Mega operator that replaces a resize and center crop with a OpenCL one.'''
    return [
        operators.mega.OpenCLCroppedResizeWithExtraCropWithColor(
            format=convert.format.name.lower(),
            width=resize.width,
            height=resize.height,
            size=resize.size,
            hcrop=resize.effective_width - center_crop.width,
            vcrop=resize.effective_height - center_crop.height,
        )
    ]


@builtin
@transformer(priority=50, hardware_caps=['opencl'])
def opencl_resize(
    color: operators.custom_preprocessing.ConvertColorInput, resize: operators.Resize
):
    '''Mega operator that replaces a resize with an OpenCL one.'''
    return [
        operators.mega.OpenCLResize(
            width=resize.width,
            height=resize.height,
            size=resize.size,
            input_color_format=color.format.name.lower(),
        )
    ]


@builtin
@transformer(priority=30, hardware_caps=['opencl'])
def opencl_resize_with_normalize(
    resize: operators.Resize,
    totensor: operators.ToTensor,
    permute: operators.PermuteChannels,
    type_cast: operators.TypeCast,
    norm: operators.Normalize,
):
    '''Mega operator that replaces a resize with an OpenCL one.'''
    if (
        permute.input_layout == types.TensorLayout.NHWC
        and permute.output_layout == types.TensorLayout.NCHW
        and type_cast.datatype == 'float32'
    ):
        mean = [x / 255.0 for x in norm.mean_values]
        std = [x / 255.0 for x in norm.std_values]
        return [
            operators.mega.OpenCLResizeToTensorAndNormalize(
                width=resize.width,
                height=resize.height,
                size=resize.size,
                mean=mean,
                std=std,
            )
        ]
    return [resize, totensor, permute, type_cast, norm]


@builtin
@transformer(priority=20, hardware_caps=['opencl'])
def opencl_resize_with_2norms(
    resize: operators.Resize,
    totensor: operators.ToTensor,
    permute: operators.PermuteChannels,
    type_cast: operators.TypeCast,
    norm1: operators.Normalize,
    norm2: operators.Normalize,
):
    '''Mega operator that replaces a resize with an OpenCL one.'''
    if (
        permute.input_layout == types.TensorLayout.NHWC
        and permute.output_layout == types.TensorLayout.NCHW
        and type_cast.datatype == 'float32'
        and norm1.std_values == [Fraction(255, 1)]
        and norm1.mean_values == [0]
    ):
        return [
            operators.mega.OpenCLResizeToTensorAndNormalize(
                width=resize.width,
                height=resize.height,
                size=resize.size,
                mean=norm2.mean,
                std=norm2.std,
            )
        ]
    return [resize, totensor, permute, type_cast, norm]


@builtin
@transformer(priority=30, hardware_caps=['opencl'])
def opencl_letterbox(
    color: operators.custom_preprocessing.ConvertColorInput, resize: operators.Letterbox
):
    '''Mega operator that replaces a resize with an OpenCL one.'''
    return [
        operators.mega.OpenCLetterBoxColorConvert(
            width=resize.width,
            height=resize.height,
            input_color_format=color.format.name.lower(),
        )
    ]


@builtin
@transformer(priority=30, hardware_caps=['opencl'])
def opencl_letterbox_with_normalize(
    resize: operators.Letterbox,
    totensor: operators.ToTensor,
    permute: operators.PermuteChannels,
    type_cast: operators.TypeCast,
    norm: operators.Normalize,
):
    '''Mega operator that replaces a resize with an OpenCL one.'''
    if (
        permute.input_layout == types.TensorLayout.NHWC
        and permute.output_layout == types.TensorLayout.NCHW
        and type_cast.datatype == 'float32'
    ):
        mean = [x / 255.0 for x in norm.mean_values]
        std = [x / 255.0 for x in norm.std_values]
        return [
            operators.mega.OpenCLetterBoxToTensorAndNormalize(
                width=resize.width,
                height=resize.height,
                scaleup=resize.scaleup,
                mean=mean,
                std=std,
            )
        ]
    return [resize, totensor, permute, type_cast, norm]


@builtin
@transformer(priority=20, hardware_caps=['opencl'])
def opencl_letterbox_with_2normalize(
    resize: operators.Letterbox,
    totensor: operators.ToTensor,
    permute: operators.PermuteChannels,
    type_cast: operators.TypeCast,
    norm1: operators.Normalize,
    norm2: operators.Normalize,
):
    '''Mega operator that replaces a resize with an OpenCL one.'''
    if (
        permute.input_layout == types.TensorLayout.NHWC
        and permute.output_layout == types.TensorLayout.NCHW
        and type_cast.datatype == 'float32'
        and norm1.std_values == [Fraction(255, 1)]
        and norm1.mean_values == [0]
    ):
        return [
            operators.mega.OpenCLetterBoxToTensorAndNormalize(
                width=resize.width,
                height=resize.height,
                scaleup=resize.scaleup,
                mean=norm2.mean,
                std=norm2.std,
            )
        ]
    return [resize, totensor, permute, type_cast, norm1, norm2]


@builtin
@transformer(priority=40)
def ax_letterbox_to_tensor_and_inplace(
    letterbox: operators.Letterbox,
    totensor: operators.ToTensor,
    permute: operators.PermuteChannels,
    type_cast: operators.TypeCast,
    norm: operators.Normalize,
):
    '''Replace a sequence of preprocessing operators with gst elements.'''
    if (
        permute.input_layout == types.TensorLayout.NHWC
        and permute.output_layout == types.TensorLayout.NCHW
        and type_cast.datatype == 'float32'
    ):
        mean = [x / 255.0 for x in norm.mean_values]
        std = [x / 255.0 for x in norm.std_values]
        return [
            operators.mega.LetterboxToTensorAndNormalise(
                height=letterbox.height,
                width=letterbox.width,
                scaleup=letterbox.scaleup,
                std=std,
                mean=mean,
            )
        ]
    return [letterbox, totensor, permute, type_cast, norm]


@builtin
@transformer(priority=39)
def ax_letterbox_with_2normalize(
    resize: operators.Letterbox,
    totensor: operators.ToTensor,
    permute: operators.PermuteChannels,
    type_cast: operators.TypeCast,
    norm1: operators.Normalize,
    norm2: operators.Normalize,
):
    '''Mega operator that replaces a resize with an OpenCL one.'''
    if (
        permute.input_layout == types.TensorLayout.NHWC
        and permute.output_layout == types.TensorLayout.NCHW
        and type_cast.datatype == 'float32'
        and norm1.std_values == [Fraction(255, 1)]
        and norm1.mean_values == [0]
    ):
        return [
            operators.mega.LetterboxToTensorAndNormalise(
                width=resize.width,
                height=resize.height,
                scaleup=resize.scaleup,
                mean=norm2.mean,
                std=norm2.std,
            )
        ]
    return [resize, totensor, permute, type_cast, norm1, norm2]


@builtin
@transformer(priority=40)
def ax_to_tensor_and_inplace_2norms(
    totensor: operators.ToTensor,
    permute: operators.PermuteChannels,
    type_cast: operators.TypeCast,
    norm1: operators.Normalize,
    norm2: operators.Normalize,
):
    '''Replace a sequence of preprocessing operators with gst elements.'''
    if (
        permute.input_layout == types.TensorLayout.NHWC
        and permute.output_layout == types.TensorLayout.NCHW
        and type_cast.datatype == 'float32'
        and norm1.std_values == [Fraction(255, 1)]
        and norm1.mean_values == [0]
    ):
        return [operators.mega.ToTensorAndNormalise(std=norm2.std, mean=norm2.mean)]
    return [totensor, permute, type_cast, norm1, norm2]


@builtin
@transformer(priority=50)
def ax_to_tensor_and_inplace(
    totensor: operators.ToTensor,
    permute: operators.PermuteChannels,
    type_cast: operators.TypeCast,
    norm: operators.Normalize,
):
    '''Replace a sequence of preprocessing operators with gst elements.'''
    if (
        permute.input_layout == types.TensorLayout.NHWC
        and permute.output_layout == types.TensorLayout.NCHW
        and type_cast.datatype == 'float32'
    ):
        mean = [x / 255.0 for x in norm.mean_values]
        std = [x / 255.0 for x in norm.std_values]
        return [operators.mega.ToTensorAndNormalise(std=std, mean=mean)]
    return [totensor, permute, type_cast, norm]


@builtin
@transformer(priority=40)
def ax_to_tensor_and_inplace_with_quant(
    totensor: operators.ToTensor,
    permute: operators.PermuteChannels,
    type_cast: operators.TypeCast,
    norm1: operators.Normalize,
    norm2: operators.Normalize,
):
    '''Replace a sequence of preprocessing operators with gst elements.'''
    if (
        permute.input_layout == types.TensorLayout.NHWC
        and permute.output_layout == types.TensorLayout.NCHW
        and type_cast.datatype == 'float32'
        and norm1.std_values == [Fraction(255, 1)]
        and norm1.mean_values == [0]
    ):
        return [
            operators.mega.ToTensorAndNormalise(std=norm2.std, mean=norm2.mean, datatype='int8')
        ]
    return [totensor, permute, type_cast, norm1, norm2]


@builtin
@transformer(priority=30, hardware_caps=['opencl'])
def opencl_totensor_and_normalize_2norms(
    totensor: operators.ToTensor,
    permute: operators.PermuteChannels,
    type_cast: operators.TypeCast,
    norm1: operators.Normalize,
    norm2: operators.Normalize,
):
    '''Replace a std gstreamer with an opencl implementation if opencl is available.'''
    if (
        permute.input_layout == types.TensorLayout.NHWC
        and permute.output_layout == types.TensorLayout.NCHW
        and type_cast.datatype == 'float32'
        and norm1.std_values == [Fraction(255, 1)]
        and norm1.mean_values == [0]
    ):
        return [operators.mega.OpenCLToTensorAndNormalize(mean=norm2.mean, std=norm2.std)]
    return [totensor, permute, type_cast, norm1, norm2]


@builtin
@transformer(priority=30, hardware_caps=['opencl'])
def opencl_totensor_and_normalize(
    totensor: operators.ToTensor,
    permute: operators.PermuteChannels,
    type_cast: operators.TypeCast,
    norm: operators.Normalize,
):
    '''Replace a std gstreamer with an opencl implementation if opencl is available.'''
    if (
        permute.input_layout == types.TensorLayout.NHWC
        and permute.output_layout == types.TensorLayout.NCHW
        and type_cast.datatype == 'float32'
    ):
        mean = [x / 255.0 for x in norm.mean_values]
        std = [x / 255.0 for x in norm.std_values]
        return [operators.mega.OpenCLToTensorAndNormalize(mean=mean, std=std)]
    return [totensor, permute, type_cast, norm]


@builtin
@transformer(priority=30)
def adjacent_normalize(norm1: operators.Normalize, norm2: operators.Normalize):
    '''Merge adjacent normalize operators that have the same tensor layout and are
    normalizing to/from 0-255.
    '''
    if norm1.tensor_layout != norm2.tensor_layout or not (
        norm1.mean_values == [0]
        and norm1.std_values == [Fraction(255, 1)]
        and norm2.std_values == [Fraction(1, 255)]
    ):
        return [norm1, norm2]

    return [
        operators.Normalize(mean=[x * norm1.std_values[0].numerator for x in norm2.mean_values])
    ]


@builtin
@transformer(priority=50)
def adjacent_type_cast_and_normalise(type_cast: operators.TypeCast, norm: operators.Normalize):
    '''Merge adjacent type cast and normalize operators so that a single gstreamer element is produced.'''
    return [
        operators.mega.TypeCastAndNormalize(
            type_cast.datatype, norm.mean, norm.std, norm.tensor_layout
        )
    ]


@builtin
@transformer(priority=50)
def ax_to_tensor_and_linear_scale(
    totensor: operators.ToTensor,
    permute: operators.PermuteChannels,
    type_cast: operators.TypeCast,
    norm: operators.LinearScaling,
):
    '''Replace a sequence of preprocessing operators with gst elements.'''
    if (
        permute.input_layout == types.TensorLayout.NHWC
        and permute.output_layout == types.TensorLayout.NCHW
        and type_cast.datatype == 'float32'
    ):
        return [
            operators.mega.ToTensorAndLinearScaling(
                type_cast.datatype, norm.mean, norm.shift, permute.input_layout, norm.tensor_layout
            )
        ]
    return [totensor, permute, type_cast, norm]

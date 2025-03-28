# Copyright Axelera AI, 2023

from fractions import Fraction

import pytest

from axelera import types
from axelera.app import config, operators, transforms


def face_preprocess():
    return [
        operators.Resize(width=640, height=480),
        operators.ConvertColor(format='RGB2BGR'),
        operators.TorchToTensor(
            input_layout='NHWC', output_layout='NCHW', datatype='float32', scale=True
        ),
        operators.Normalize(
            mean='104/255, 117/255, 123/255',
            std='1/255, 1/255, 1/255',
            tensor_layout=types.TensorLayout.NCHW,
        ),
    ]


def test_torch_to_tensor_expansion():
    ops = face_preprocess()
    transforms.composite_expansion(ops)
    assert ops == [
        operators.Resize(width=640, height=480),
        operators.ConvertColor(format='RGB2BGR'),
        operators.ToTensor(),
        operators.PermuteChannels(input_layout='NHWC', output_layout='NCHW'),
        operators.TypeCast(datatype='float32'),
        operators.Normalize(std='255.0'),
        operators.Normalize(
            mean='104/255, 117/255, 123/255',
            std='1/255, 1/255, 1/255',
            tensor_layout=types.TensorLayout.NCHW,
        ),
    ]


def test_barrel_and_convert():
    ops = [
        operators.custom_preprocessing.CameraUndistort(
            fx=1.0, fy=1.0, cx=0.5, cy=0.5, distort_coefs=[1.0, 1.0, 1.0, 0.0, 0.0]
        ),
        operators.custom_preprocessing.ConvertColorInput(format='rgb'),
    ]

    got = ops.copy()
    transforms.opencl_colorconvert_with_cameraundistort(ops)
    assert got != ops
    assert ops == [
        operators.mega.OpenCLBarrelDistortionCorrection(
            distort_coefs=[1.0, 1.0, 1.0, 0.0, 0.0], fx=1.0, fy=1.0, cx=0.5, cy=0.5, format='rgb'
        )
    ]


def test_perspective_and_convert():
    ops = [
        operators.custom_preprocessing.Perspective(
            camera_matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
        ),
        operators.custom_preprocessing.ConvertColorInput(format='rgb'),
    ]

    got = ops.copy()
    transforms.opencl_colorconvert_with_perspective(ops)
    assert got != ops
    assert ops == [
        operators.mega.OpenCLPerspectiveTransform(
            camera_matrix=[1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0], format='rgb'
        )
    ]


def test_do_not_expand_resize_at_end():
    ops = [operators.Resize(width=640, height=480)]
    got = ops.copy()
    transforms.resize_and_convert_transform(got)
    assert got == ops


def test_resize_and_convert():
    got = face_preprocess()
    transforms.resize_and_convert_transform(got)
    assert got == [
        operators.mega.ResizeAndConvert(width=640, height=480, format='RGB2BGR'),
        operators.TorchToTensor(
            input_layout='NHWC', output_layout='NCHW', datatype='float32', scale=True
        ),
        operators.Normalize(
            mean='104/255, 117/255, 123/255',
            std='1/255, 1/255, 1/255',
            tensor_layout=types.TensorLayout.NCHW,
        ),
    ]


def test_adjacent_normalize_incompatible():
    got = [
        operators.TypeCast(datatype='float32'),
        operators.Normalize(mean='0', std='255.0', tensor_layout=types.TensorLayout.NCHW),
        operators.Normalize(
            mean='104/255, 117/255, 123/255',
            std='2/255, 2/255, 2/255',
            tensor_layout=types.TensorLayout.NCHW,
        ),
    ]
    transforms.adjacent_normalize(got)
    assert got == [
        operators.TypeCast(datatype='float32'),
        operators.Normalize(mean='0', std='255.0', tensor_layout=types.TensorLayout.NCHW),
        operators.Normalize(
            mean='104/255, 117/255, 123/255',
            std='2/255, 2/255, 2/255',
            tensor_layout=types.TensorLayout.NCHW,
        ),
    ]


def test_adjacent_normalize():
    got = [
        operators.TypeCast(datatype='float32'),
        operators.Normalize(mean='0', std='255.0', tensor_layout=types.TensorLayout.NCHW),
        operators.Normalize(
            mean='104/255, 117/255, 123/255',
            std='1/255, 1/255, 1/255',
            tensor_layout=types.TensorLayout.NCHW,
        ),
    ]
    transforms.adjacent_normalize(got)
    assert got == [
        operators.TypeCast(datatype='float32'),
        operators.Normalize(
            mean=[Fraction(104, 1), Fraction(117, 1), Fraction(123, 1)],
            std='1',
            tensor_layout=types.TensorLayout.NCHW,
        ),
    ]


def test_adjacent_type_cast_and_normalise():
    got = [
        operators.TypeCast(datatype='float32'),
        operators.Normalize(mean='0', std='255.0', tensor_layout=types.TensorLayout.NCHW),
        operators.Normalize(
            mean='104/255, 117/255, 123/255',
            std='1/255, 1/255, 1/255',
            tensor_layout=types.TensorLayout.NCHW,
        ),
    ]
    transforms.adjacent_type_cast_and_normalise(got)
    assert got == [
        operators.mega.TypeCastAndNormalize(
            datatype='float32', mean='0', std='255.0', tensor_layout=types.TensorLayout.NCHW
        ),
        operators.Normalize(
            mean='104/255, 117/255, 123/255',
            std='1/255, 1/255, 1/255',
            tensor_layout=types.TensorLayout.NCHW,
        ),
    ]


def temp_skip_test_all_face_transforms():  # TODO needs fixing, but facerecog is off the table for EAP0.5
    got = face_preprocess()
    transforms.run_all_transformers(got, hardware_caps=config.HardwareCaps.NONE)
    assert got == [
        operators.mega.ResizeAndConvert(width=640, height=480, format='RGB2BGR'),
        operators.mega.ToTensorAndNormalise(
            mean='104/255, 117/255, 123/255', std='1/255, 1/255, 1/255'
        ),
    ]


def test_all_face_transforms_with_incompatible_option():
    # note input/output layout of TorchToTensor is incompatible with ToTensorAndNormalise
    got = [
        operators.Resize(width=640, height=480),
        operators.ConvertColor(format='RGB2BGR'),
        operators.TorchToTensor(
            input_layout='NCHW', output_layout='NHWC', datatype='float32', scale=True
        ),
        operators.Normalize(
            mean='104/255, 117/255, 123/255',
            std='1/255, 1/255, 1/255',
            tensor_layout=types.TensorLayout.NCHW,
        ),
    ]
    transforms.run_all_transformers(got, hardware_caps=config.HardwareCaps.NONE)
    assert got == [
        operators.mega.ResizeAndConvert(width=640, height=480, format='RGB2BGR'),
        operators.ToTensor(),
        operators.PermuteChannels(input_layout='NCHW', output_layout='NHWC'),
        operators.mega.TypeCastAndNormalize(
            datatype='float32',
            mean=[Fraction(104, 1), Fraction(117, 1), Fraction(123, 1)],
            std='1',
            tensor_layout=types.TensorLayout.NCHW,
        ),
    ]


def test_adjacent_type_cast_and_normalise():
    mean = '1.0'
    shift = '0.406'
    got = [
        operators.ToTensor(),
        operators.PermuteChannels('NHWC', 'NCHW'),
        operators.TypeCast(datatype='float32'),
        operators.LinearScaling(mean, shift, 'NCHW'),
    ]
    transforms.ax_to_tensor_and_linear_scale(got)
    assert got == [
        operators.mega.ToTensorAndLinearScaling('float32', mean, shift, 'NHWC', 'NCHW'),
    ]


@pytest.mark.parametrize(
    'inp, outp, dtype',
    [
        ('NCHW', 'NCHW', 'float32'),
        ('NCHW', 'NHWC', 'float32'),
        ('NHWC', 'NHWC', 'float32'),
    ],
)
def test_adjacent_type_cast_and_normalise_non_matching(inp, outp, dtype):
    mean = [1.0]
    shift = [0.406]
    ops = [
        operators.ToTensor(),
        operators.PermuteChannels(inp, outp),
        operators.TypeCast(datatype=dtype),
        operators.LinearScaling(mean, shift, inp),
    ]
    got = ops[:]
    transforms.ax_to_tensor_and_linear_scale(got)
    assert got == ops


def test_adjacent_letterbox_to_tensor_normalise():
    got = [
        operators.Letterbox(width=640, height=640),
        operators.ToTensor(),
        operators.PermuteChannels('NHWC', 'NCHW'),
        operators.TypeCast(datatype='float32'),
        operators.Normalize(
            mean='0',
            std='255',
            tensor_layout=types.TensorLayout.NCHW,
        ),
    ]
    transforms.run_all_transformers(got, hardware_caps=config.HardwareCaps.OPENCL)
    assert got == [
        operators.mega.OpenCLetterBoxToTensorAndNormalize(
            width=640,
            height=640,
            mean=[0.0],
            std=[1.0],
        )
    ]


def test_adjacent_letterbox_to_tensor_2normalise():
    got = [
        operators.Letterbox(width=640, height=640),
        operators.ToTensor(),
        operators.PermuteChannels('NHWC', 'NCHW'),
        operators.TypeCast(datatype='float32'),
        operators.Normalize(
            mean='0',
            std='255',
            tensor_layout=types.TensorLayout.NCHW,
        ),
        operators.Normalize(
            mean='104/255, 117/255, 123/255',
            std='2/255, 2/255, 2/255',
            tensor_layout=types.TensorLayout.NCHW,
        ),
    ]
    transforms.run_all_transformers(got, hardware_caps=config.HardwareCaps.OPENCL)
    assert got == [
        operators.mega.OpenCLetterBoxToTensorAndNormalize(
            width=640, height=640, mean='104/255, 117/255, 123/255', std='2/255, 2/255, 2/255'
        ),
    ]


def test_adjacent_letterbox_to_tensor_2normalise_no_opencl():
    got = [
        operators.Letterbox(width=640, height=640),
        operators.ToTensor(),
        operators.PermuteChannels('NHWC', 'NCHW'),
        operators.TypeCast(datatype='float32'),
        operators.Normalize(
            mean='0',
            std='255',
            tensor_layout=types.TensorLayout.NCHW,
        ),
        operators.Normalize(
            mean='104/255, 117/255, 123/255',
            std='2/255, 2/255, 2/255',
            tensor_layout=types.TensorLayout.NCHW,
        ),
    ]
    transforms.run_all_transformers(got)
    assert got == [
        operators.mega.LetterboxToTensorAndNormalise(
            width=640, height=640, mean='104/255, 117/255, 123/255', std='2/255, 2/255, 2/255'
        ),
    ]


def test_adjacent_resize_to_tensor_2normalise():
    got = [
        operators.Resize(width=640, height=640),
        operators.ToTensor(),
        operators.PermuteChannels('NHWC', 'NCHW'),
        operators.TypeCast(datatype='float32'),
        operators.Normalize(
            mean='0',
            std='255',
            tensor_layout=types.TensorLayout.NCHW,
        ),
        operators.Normalize(
            mean='104/255, 117/255, 123/255',
            std='2/255, 2/255, 2/255',
            tensor_layout=types.TensorLayout.NCHW,
        ),
    ]
    transforms.run_all_transformers(got, hardware_caps=config.HardwareCaps.OPENCL)
    assert got == [
        operators.mega.OpenCLResizeToTensorAndNormalize(
            width=640,
            height=640,
            size=0,
            mean='104/255, 117/255, 123/255',
            std='2/255, 2/255, 2/255',
            datatype='float32',
            scaleup=0,
        ),
    ]

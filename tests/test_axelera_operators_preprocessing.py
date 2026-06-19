# Copyright Axelera AI, 2023
import inspect
from pathlib import Path
import re
from unittest.mock import ANY, MagicMock, patch

from axelera.types import img
import cv2
import numpy as np
import pytest

from axelera import types
from axelera.app import gst_builder, operators

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
import torchvision.transforms as T
import torchvision.transforms.functional as TF

packed_rgb = np.array(
    [
        [[1, 2, 3], [4, 5, 6]],
        [[11, 12, 13], [14, 15, 16]],
        [[21, 22, 23], [24, 25, 26]],
        [[31, 32, 33], [34, 35, 36]],
    ],
    dtype=np.uint8,
)
planar_rgb = np.ascontiguousarray(packed_rgb.transpose(2, 0, 1))


convert = {'instance': 'axtransform', 'lib': 'libtransform_resize.so', 'options': 'to_tensor:1'}


def _gen_gst(op, stream_idx=''):
    # note we use the old builder so we can test the gst output, the new builder
    # consumes the gst output in readiness for an axinferencenet. A forthcoming
    # PR will tidy this up  by having explicit begin/end axinferencenet
    gst = gst_builder._OldBuilder(None, None, 16)
    op.build_gst(gst, stream_idx)
    return list(gst)


def arithmetic(*parts):
    options = ','.join(parts)
    return {'instance': 'tensor_transform', 'mode': 'arithmetic', 'option': options}


def dimchg(option):
    return {'instance': 'tensor_transform', 'mode': 'dimchg', 'option': option}


@pytest.mark.parametrize(
    'params, exp_cls, exp_err',
    [
        (
            {'input_layout': 'wibble', 'output_layout': 'NCHW'},
            ValueError,
            r"Invalid value for input_layout: wibble \(expected one of NCHW, NHWC, CHWN\)",
        ),
        (
            {'input_layout': 'NCHW', 'output_layout': 'wobble'},
            ValueError,
            r"Invalid value for output_layout: wobble \(expected",
        ),
        (
            {'input_layout': 'NHWC', 'output_layout': 'CHWN'},
            ValueError,
            r"Unsupported input/output layouts: NHWC/CHWN",
        ),
        (
            {'datatype': 'bfloat16'},
            ValueError,
            r"Only float32 and uint8 are supported for datatype",
        ),
    ],
)
def test_torch_to_tensor_bad_values(params, exp_cls, exp_err):
    with pytest.raises(exp_cls, match=exp_err):
        operators.TorchToTensor(**params)


@pytest.mark.parametrize(
    'input, output, datatype, scale, exp_gst',
    [
        (
            'NCHW',
            'NHWC',
            'float32',
            False,
            [convert, dimchg('2:0'), arithmetic('typecast:float32')],
        ),
        ('NHWC', 'NCHW', 'uint8', False, [convert, dimchg('0:2')]),
        (
            'NCHW',
            'NHWC',
            'float32',
            True,
            [convert, dimchg('2:0'), arithmetic('typecast:float32'), arithmetic('div:255.0')],
        ),
    ],
)
def test_torch_to_tensor_defaults(input, output, datatype, scale, exp_gst):
    op = operators.TorchToTensor(
        input_layout=input, output_layout=output, datatype=datatype, scale=scale
    )
    i = img.fromarray(planar_rgb if input == 'NCHW' else packed_rgb)
    out = op.exec_torch(i)
    axes = [input[1:].index(x) for x in output[1:]]
    exp = i.asarray().transpose(*axes).astype(datatype)
    if scale:
        exp = exp / 255.0
    np.testing.assert_equal(out.numpy(), exp)
    assert out.numpy().flags['C_CONTIGUOUS']
    if input != 'NHWC':
        with pytest.raises(
            ValueError, match=r"PermuteChannels is only supported for NHWC layout in GStreamer"
        ):
            _gen_gst(op)
    else:
        _gen_gst(op)  # Should not raise for NHWC input


def test_torch_to_tensor_greyscale():
    op = operators.TorchToTensor()
    data = np.arange(24 * 32, dtype=np.uint8).reshape(24, 32)
    i = img.fromany(data.copy(), types.ColorFormat.GRAY)
    out = op.exec_torch(i)
    exp = data.astype(np.float32).reshape(1, 24, 32)
    exp = exp / 255.0
    np.testing.assert_equal(out.numpy(), exp)
    assert out.numpy().flags['C_CONTIGUOUS']


def test_normalise_expects_tensor():
    op = operators.Normalize()
    with pytest.raises(TypeError, match=r'Normalize input must be of type Tensor \(got Image\)'):
        op.exec_torch(img.fromarray(planar_rgb))


@pytest.mark.parametrize(
    'params, exp_cls, exp_err',
    [
        ({'mean': 'notanumber'}, ValueError, r"Cannot convert 'notanumber' to float in Normalize"),
        (
            {'std': '0, notanumber, 1'},
            ValueError,
            r"Cannot convert 'notanumber' to float in Normalize",
        ),
        (
            {'std': '1,2'},
            ValueError,
            r'Normalize expects 1, 3 or 4 float/fraction expressions',
        ),
    ],
)
def test_normalise_bad_values(params, exp_cls, exp_err):
    with pytest.raises(exp_cls, match=exp_err):
        operators.Normalize(**params)


def test_normalise():
    op = operators.Normalize(mean='0.5', std='0.8', tensor_layout='NCHW')
    assert _gen_gst(op) == [
        {
            'instance': 'axtransform',
            'lib': 'libtransform_normalize_cl.so',
            'options': (
                'to_tensor:0;mean:0.5,0.5,0.5;std:0.8,0.8,0.8;'
                'quant_scale:1.0;quant_zeropoint:0.0'
            ),
        },
    ]


def test_normalise_standalone_with_manifest_quant():
    mi = types.ModelInfo(
        'modelname',
        types.TaskCategory.Classification,
        [3, 224, 224],
    )
    mi.manifest = types.Manifest(
        'modellib',
        input_shapes=[(1, 3, 224, 224)],
        input_dtypes=['uint8'],
        output_shapes=[(1, 1000)],
        output_dtypes=['float32'],
        quantize_params=[(0.1, -14)],
        dequantize_params=[(0.3, 0.4)],
        model_lib_file='model.json',
    )
    op = operators.Normalize(mean='0.485, 0.456, 0.406', std='0.229, 0.224, 0.225')
    mock_task_graph = MagicMock()
    mock_task_graph.get_master.return_value = "mocked_master_value"
    op.configure_model_and_context_info(
        mi, operators.PipelineContext(), "task_name", 0, Path('.'), task_graph=mock_task_graph
    )
    assert _gen_gst(op) == [
        {
            'instance': 'axtransform',
            'lib': 'libtransform_normalize_cl.so',
            'options': (
                'to_tensor:0;mean:0.485,0.456,0.406;std:0.229,0.224,0.225;'
                'quant_scale:0.1;quant_zeropoint:-14.0'
            ),
        },
    ]


def test_normalise_standalone_scale_only():
    op = operators.Normalize(std='255')
    assert _gen_gst(op) == [
        {
            'instance': 'axtransform',
            'lib': 'libtransform_normalize_cl.so',
            'options': (
                'to_tensor:0;mean:0.,0.,0.;std:255.,255.,255.;'
                'quant_scale:1.0;quant_zeropoint:0.0'
            ),
        },
    ]


def test_linearscaling_expects_tensor():
    op = operators.LinearScaling()
    with pytest.raises(
        TypeError, match=r'LinearScaling input must be of type Tensor \(got Image\)'
    ):
        op.exec_torch(img.fromarray(planar_rgb))


@pytest.mark.parametrize(
    'params, exp_cls, exp_err',
    [
        (
            {'mean': 'notanumber'},
            ValueError,
            r"Cannot convert 'notanumber' to float in LinearScaling",
        ),
        (
            {'shift': '0, notanumber, 1'},
            ValueError,
            r"Cannot convert 'notanumber' to float in LinearScaling",
        ),
        (
            {'mean': '1,2'},
            ValueError,
            r'LinearScaling expects 1, 3 or 4 float/fraction expressions',
        ),
    ],
)
def test_linearscaling_bad_values(params, exp_cls, exp_err):
    with pytest.raises(exp_cls, match=exp_err):
        operators.LinearScaling(**params)


@pytest.mark.parametrize(
    'name, params, exp_gst',
    [
        (
            'all redundant (defaults)',
            {},
            [],
        ),
        (
            'all redundant',
            {'mean': '1', 'shift': '0'},
            [],
        ),
    ],
)
def test_linearscaling(name, params, exp_gst):
    data = np.arange((3 * 1 * 4)).reshape(3, 1, 4).astype(np.float32) / 100.0
    op = operators.LinearScaling(**params, tensor_layout='NCHW')
    got = op.exec_torch(torch.from_numpy(data.copy()))
    assert exp_gst == _gen_gst(op), f'in test {name}'
    m = params.get('mean', '1.0')
    divs = [m] if isinstance(m, float) else [float(eval(x.strip())) for x in m.split(',')]
    s = params.get('shift', '0.0')
    adds = [s] if isinstance(s, float) else [float(eval(x.strip())) for x in s.split(',')]
    divs = np.array(divs, dtype=np.float32).reshape(-1, 1, 1)
    adds = np.array(adds, dtype=np.float32).reshape(-1, 1, 1)
    exp = data / divs + adds
    np.testing.assert_array_almost_equal(exp, got.numpy(), decimal=4)


@pytest.mark.parametrize(
    'params, exp_cls, exp_err',
    [
        (
            {'input_layout': 'wibble', 'output_layout': 'NCHW'},
            ValueError,
            r"Invalid value for input_layout: wibble \(expected",
        ),
        (
            {'input_layout': 'NCHW', 'output_layout': 'wobble'},
            ValueError,
            r"Invalid value for output_layout: wobble \(expected",
        ),
    ],
)
def test_permute_channels_bad_values(params, exp_cls, exp_err):
    with pytest.raises(exp_cls, match=exp_err):
        operators.PermuteChannels(**params)


@pytest.mark.parametrize(
    'input, output, exp_gst',
    [
        ('NCHW', 'NHWC', [dimchg('2:0')]),
        ('NHWC', 'NCHW', [dimchg('0:2')]),
        ('CHWN', 'NCHW', [dimchg('0:3')]),
        ('NCHW', 'CHWN', [dimchg('3:0')]),
    ],
)
def test_permute_channels(input, output, exp_gst):
    op = operators.PermuteChannels(input_layout=input, output_layout=output)
    ndim = len(input)
    dims = range(2, 2 + ndim)
    i = np.arange(np.prod(dims), dtype=np.uint8).reshape(*dims)
    if input == output:
        exp = i.copy()
    else:
        axes = [input.index(x) for x in output]
        exp = i.transpose(*axes).copy()
    out = op.exec_torch(torch.from_numpy(i)).numpy()
    np.testing.assert_equal(out, exp)
    assert out.flags['C_CONTIGUOUS']
    if input != 'NHWC':
        with pytest.raises(
            ValueError, match="PermuteChannels is only supported for NHWC layout in GStreamer"
        ):
            _gen_gst(op)
    else:
        _gen_gst(op)  # Should not raise for NHWC input


@pytest.mark.parametrize(
    'params, exp_err',
    [
        ({'width': 'notanumber'}, r"Invalid value for width: notanumber"),
        (
            {'height': 'notanumber'},
            r"Invalid value for height: notanumber (expected convertible to int)",
        ),
        ({'size': 'notanumber'}, r"Invalid value for size: notanumber"),
        ({'width': -1}, r"Invalid unsigned int value for width: -1"),
        ({'height': "-1"}, r"Invalid unsigned int value for height: -1"),
        ({'size': -100}, r"Invalid unsigned int value for size: -1"),
        (
            {'width': 0, 'height': 0, 'size': 0},
            r"Only size (given: 0) or both width/height (given: 0/0) can be specified (i.e. non-zero)",
        ),
        (
            {'width': 0, 'height': 0},
            r"Only size (given: 0) or both width/height (given: 0/0) can be specified (i.e. non-zero)",
        ),
        (
            {'size': 0},
            r"Only size (given: 0) or both width/height (given: 0/0) can be specified (i.e. non-zero)",
        ),
        (
            {'width': 0, 'height': 3},
            r"Only size (given: 0) or both width/height (given: 0/3) can be specified (i.e. non-zero)",
        ),
        (
            {'width': 999, 'height': 0},
            r"Only size (given: 0) or both width/height (given: 999/0) can be specified (i.e. non-zero)",
        ),
        (
            {'size': 100, 'width': 0, 'height': 70},
            r"Only size (given: 100) or both width/height (given: 0/70) can be specified (i.e. non-zero)",
        ),
        (
            {'size': 100, 'width': 2, 'height': 0},
            r"Only size (given: 100) or both width/height (given: 2/0) can be specified (i.e. non-zero)",
        ),
        (
            {'size': 1, 'width': 2, 'height': 3},
            r"Only size (given: 1) or both width/height (given: 2/3) can be specified (i.e. non-zero)",
        ),
        (
            {'size': 1, 'interpolation': 'big'},
            r"Invalid value for interpolation: big (expected one of nearest, bilinear, bicubic, lanczos)",
        ),
    ],
)
def test_resize_bad_values(params, exp_err):
    with pytest.raises(ValueError, match=re.escape(exp_err)):
        operators.Resize(**params)


def gst_resize_exp_out(params):
    if 'size' in params:
        w, h = params['size'], params['size']
    else:
        w, h = params['width'], params['height']
    return [
        {
            'instance': "axtransform",
            'lib': 'libtransform_resize.so',
            'options': f'width:{w};height:{h};letterbox:0',
        },
    ]


@pytest.mark.parametrize('half_pixel_centers', [True, False])
@pytest.mark.parametrize(
    'name, img_shape, params, exp_passed_in_size, exp_shape, exp_interp',
    [
        (
            'unchanged w h',
            (20, 40, 3),
            {"width": 40, "height": 20},
            (20, 40),
            (20, 40, 3),
            T.InterpolationMode.BILINEAR,
        ),
        (
            'unchanged size h',
            (20, 40, 3),
            {"size": 20},
            20,
            (20, 40, 3),
            T.InterpolationMode.BILINEAR,
        ),
        (
            'unchanged size w',
            (40, 20, 3),
            {"size": 20},
            20,
            (40, 20, 3),
            T.InterpolationMode.BILINEAR,
        ),
        (
            'changed w h',
            (20, 40, 3),
            {"width": 20, "height": 10},
            (10, 20),
            (10, 20, 3),
            T.InterpolationMode.BILINEAR,
        ),
        (
            'changed size h',
            (20, 40, 3),
            {"size": 10},
            10,
            (10, 20, 3),
            T.InterpolationMode.BILINEAR,
        ),
        (
            'changed size w',
            (40, 20, 3),
            {"size": 10},
            10,
            (20, 10, 3),
            T.InterpolationMode.BILINEAR,
        ),
        (
            'interp: nearest',
            (40, 20, 3),
            {"size": 10, "interpolation": "nearest"},
            10,
            (20, 10, 3),
            T.InterpolationMode.NEAREST,
        ),
        (
            'interp: bilinear',
            (40, 20, 3),
            {"size": 10, "interpolation": "bilinear"},
            10,
            (20, 10, 3),
            T.InterpolationMode.BILINEAR,
        ),
        (
            'interp: bicubic',
            (40, 20, 3),
            {"size": 10, "interpolation": "bicubic"},
            10,
            (20, 10, 3),
            T.InterpolationMode.BICUBIC,
        ),
        (
            'interp: lanczos',
            (40, 20, 3),
            {"size": 10, "interpolation": "lanczos"},
            10,
            (20, 10, 3),
            T.InterpolationMode.LANCZOS,
        ),
        (
            'interp: lanczos via enum',
            (40, 20, 3),
            {"size": 10, "interpolation": operators.InterpolationMode.lanczos},
            10,
            (20, 10, 3),
            T.InterpolationMode.LANCZOS,
        ),
        (
            'preserve aspect ratio - portrait',
            (100, 50, 3),  # height > width
            {"size": 25},  # should resize width to 25 and height to 50
            25,
            (50, 25, 3),
            T.InterpolationMode.BILINEAR,
        ),
        (
            'preserve aspect ratio - landscape',
            (50, 100, 3),  # width > height
            {"size": 25},  # should resize height to 25 and width to 50
            25,
            (25, 50, 3),
            T.InterpolationMode.BILINEAR,
        ),
        (
            'unchanged w h with grayscale',
            (20, 40),
            {"width": 40, "height": 20},
            (20, 40),
            (20, 40),
            T.InterpolationMode.BILINEAR,
        ),
    ],
)
def test_resize(
    name, img_shape, params, exp_passed_in_size, exp_shape, exp_interp, half_pixel_centers
):
    def _get_image():
        if len(img_shape) == 2:
            return types.Image.fromarray(
                np.zeros(img_shape, dtype=np.uint8), types.ColorFormat.GRAY
            )
        else:
            return types.Image.fromarray(
                np.zeros(img_shape, dtype=np.uint8), types.ColorFormat.RGB
            )

    params['half_pixel_centers'] = half_pixel_centers
    if half_pixel_centers:
        with patch.object(cv2, 'resize', wraps=cv2.resize) as resize:
            op = operators.Resize(**params)
            i = _get_image()
            torch_out = op.exec_torch(i)
            if 'size' in params:
                assert resize.call_count == 1
            else:
                resize.assert_called_once_with(
                    ANY, (params['width'], params['height']), interpolation=ANY
                )
    else:
        with patch.object(TF, 'resize', wraps=TF.resize) as resize:
            op = operators.Resize(**params)
            torch_out = op.exec_torch(_get_image())
            resize.assert_called_once_with(
                ANY, exp_passed_in_size, exp_interp, max_size=ANY, antialias=ANY
            )

    output_shape = torch_out.asarray().shape
    assert output_shape == exp_shape
    assert _gen_gst(op) == gst_resize_exp_out(params)


@pytest.mark.parametrize(
    'name, img_shape, params, exp_passed_in_size, exp_shape, exp_interp',
    [
        (
            'unchanged w h',
            (20, 40),
            {"width": 40, "height": 20},
            (40, 20),
            (20, 40),
            cv2.INTER_LINEAR,
        ),
        (
            'changed w h',
            (20, 40),
            {"width": 20, "height": 10},
            (20, 10),
            (10, 20),
            cv2.INTER_LINEAR,
        ),
        (
            'interp: nearest',
            (40, 20),
            {"width": 20, "height": 10, "interpolation": "nearest"},
            (20, 10),
            (10, 20),
            cv2.INTER_NEAREST,
        ),
        (
            'interp: bilinear',
            (40, 20),
            {"width": 20, "height": 10, "interpolation": "bilinear"},
            (20, 10),
            (10, 20),
            cv2.INTER_LINEAR,
        ),
        (
            'interp: bicubic',
            (40, 20),
            {"width": 20, "height": 10, "interpolation": "bicubic"},
            (20, 10),
            (10, 20),
            cv2.INTER_CUBIC,
        ),
        (
            'interp: lanczos',
            (40, 20),
            {"width": 20, "height": 10, "interpolation": "lanczos"},
            (20, 10),
            (10, 20),
            cv2.INTER_LANCZOS4,
        ),
        (
            'interp: lanczos via enum',
            (40, 20),
            {"width": 20, "height": 10, "interpolation": operators.InterpolationMode.lanczos},
            (20, 10),
            (10, 20),
            cv2.INTER_LANCZOS4,
        ),
        (
            'smaller at w',
            (40, 20),
            {"size": 10},
            (10, 20),
            (20, 10),
            cv2.INTER_LINEAR,
        ),
        (
            'smaller at h',
            (20, 40),
            {"size": 10},
            (20, 10),
            (10, 20),
            cv2.INTER_LINEAR,
        ),
    ],
)
def test_resize_half_pixel(name, img_shape, params, exp_passed_in_size, exp_shape, exp_interp):
    params['half_pixel_centers'] = True
    with patch.object(cv2, 'resize', wraps=cv2.resize) as resize:
        op = operators.Resize(**params)
        i = types.Image.fromarray(np.zeros(img_shape, dtype=np.uint8), types.ColorFormat.GRAY)
        torch_out = op.exec_torch(i)
    resize.assert_called_once_with(ANY, exp_passed_in_size, interpolation=exp_interp)
    assert torch_out.asarray().shape == exp_shape
    print(_gen_gst(op))
    assert _gen_gst(op) == gst_resize_exp_out(params)


def test_to_tensor():
    op = operators.ToTensor()
    data = np.arange(5 * 4 * 3, dtype=np.uint8).reshape(5, 4, 3)
    got = op.exec_torch(img.fromany(data))
    np.testing.assert_equal(got.numpy(), data)
    assert got.numpy().flags['C_CONTIGUOUS']
    assert _gen_gst(op) == [convert]


def test_type_cast_bad_type():
    with pytest.raises(
        ValueError, match=r"Only float32 and uint8 are supported for datatype (not 'oops')"
    ):
        operators.TypeCast(datatype='oops')


def test_type_cast():
    op = operators.TypeCast(datatype='float32')
    data = np.arange(5 * 4 * 3, dtype=np.uint8).reshape(5, 4, 3)
    got = op.exec_torch(torch.from_numpy(data))
    np.testing.assert_equal(got.numpy(), data.astype('float32'))
    assert got.numpy().flags['C_CONTIGUOUS']
    assert _gen_gst(op) == []


def test_type_cast_uint8():
    op = operators.TypeCast(datatype='uint8')
    data = np.arange(5 * 4 * 3, dtype=np.uint8).reshape(5, 4, 3)
    got = op.exec_torch(torch.from_numpy(data))
    np.testing.assert_equal(got.numpy(), data)
    assert got.numpy().flags['C_CONTIGUOUS']
    assert _gen_gst(op) == []


def test_type_cast_invalid_input_type_torch():
    op = operators.TypeCast(datatype='uint8')
    data = np.arange(5 * 4 * 3, dtype=np.int16).reshape(5, 4, 3)
    with pytest.raises(TypeError, match=r"Input must be a torch tensor of uint8 not torch.int16"):
        op.exec_torch(torch.from_numpy(data))


def test_type_cast_invalid_input_type_numpy():
    op = operators.TypeCast(datatype='uint8')
    data = np.arange(5 * 4 * 3, dtype=np.uint16).reshape(5, 4, 3)
    with pytest.raises(TypeError, match=r"Input must be a torch tensor of uint8 not ndarray"):
        op.exec_torch(data)


def test_center_crop():
    data = np.arange(10 * 8 * 3, dtype=np.uint8).reshape(10, 8, 3)
    op = operators.CenterCrop(width=6, height=8)
    got = op.exec_torch(img.fromany(data))
    np.testing.assert_equal(got.asarray(), data[1:9, 1:7, :])
    assert _gen_gst(op) == [
        {
            'instance': 'axtransform',
            'lib': 'transform_centrecropextra.so',
            'options': 'crop_width:6,crop_height:8',
        },
    ]


def test_resize_interp_modes():
    assert (
        operators.Resize(size=178, interpolation='bilinear').interpolation
        == operators.preprocessing.InterpolationMode.bilinear
    )


@pytest.mark.parametrize(
    'stream_id, stream_match, expected',
    [
        (0, None, True),
        (1, None, True),
        (2, '.*', True),
        (0, 0, True),
        (1, 0, False),
        (0, 1, False),
        (0, [0, 1], True),
        (1, [0, 1], True),
        (2, [0, 1], False),
        (3, [0, 1], False),
        (0, {'include': [0, 1]}, True),
        (0, {'include': [0]}, True),
        (1, {'include': [0, 1]}, True),
        (2, {'include': [0, 1]}, False),
        (3, {'include': [0, 1]}, False),
        (3, {'include': [2, 3]}, True),
        (3, {'exclude': [2, 3]}, False),
        (0, {'exclude': [2, 3]}, True),
        (0, {'exclude': [0, 1]}, False),
    ],
)
def test_stream_match_modes(stream_id, stream_match, expected):

    op = operators.ConvertColorInput(stream_match=stream_match)
    assert op.stream_check_match(stream_id) == expected


@pytest.mark.parametrize(
    'camera_matrix',
    [
        '1.0,0,0,0,1.0,0,0,0,1.0',
        [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0],
    ],
)
def test_preprocess_operators_input_output_type(camera_matrix):
    sample_data = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
    sample_image = img.fromarray(sample_data)
    sample_tensor = torch.from_numpy(sample_data.transpose((2, 0, 1)).astype(np.float32) / 255.0)

    # Default values for common parameter types and unannotated parameters
    default_values = {
        int: 1,
        float: 1.0,
        str: '1.',
        bool: True,
        np.ndarray: np.array([]),
        torch.Tensor: torch.tensor([]),
        inspect._empty: None,  # Generic fallback for unannotated parameters
    }
    specific_default_for_op = {
        operators.Resize: {'size': 100},
        operators.PermuteChannels: {'input_layout': "NCHW", 'output_layout': "NCHW"},
        operators.ConvertColor: {
            'format': 'RGB2BGR',
        },
        operators.CameraUndistort: {
            'fx': 1920,
            'fy': 1080,
            'cx': 860,
            'cy': 540,
            'distort_coefs': [0, 0, 0, 0, 0],
        },
        operators.Perspective: {
            'camera_matrix': camera_matrix,
            'out_format': '3',
        },
        operators.ConvertColorInput: {
            'format': 'rgb',
        },
    }

    # Loop over all members in the operators module
    for name, obj in inspect.getmembers(operators):
        if (
            inspect.isclass(obj)
            and issubclass(obj, operators.PreprocessOperator)
            and obj is not operators.PreprocessOperator
        ):
            # Get the constructor parameters
            params = inspect.signature(obj.__init__).parameters
            kwargs = {}
            for param_name, param in params.items():
                if param_name == 'self':
                    continue
                # Use specific defaults if available for this operator
                if obj in specific_default_for_op and param_name in specific_default_for_op[obj]:
                    kwargs[param_name] = specific_default_for_op[obj][param_name]
                elif param.default is not inspect.Parameter.empty:
                    kwargs[param_name] = param.default
                elif param.annotation in default_values:
                    kwargs[param_name] = default_values[param.annotation]
                else:
                    raise ValueError(f"No default value found for {param_name} in {name}")
            # Instantiate the operator with default parameters
            op = obj(**kwargs)

            if hasattr(op, 'exec_torch'):
                # Test with both sample_image and sample_tensor
                for input_data in [sample_image, sample_tensor]:
                    try:
                        # Execute the operator
                        output_data = op.exec_torch(input_data)

                        # Check if input and output types are the same
                        assert type(input_data) == type(
                            output_data
                        ), f"Mismatch in {name}: {type(input_data)} != {type(output_data)}"
                    except Exception as e:
                        # Skip the test if the input type is not supported
                        # print(f"Skipped {name} with input {type(input_data).__name__}: {str(e)}")
                        pass


@pytest.mark.parametrize(
    'input_type, format, input_data, expected_output',
    [
        # Test RGB to BGR conversion with torch tensor
        (
            'tensor',
            'RGB2BGR',
            torch.tensor(
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=torch.uint8
            ),
            torch.tensor(
                [[[9, 10], [11, 12]], [[5, 6], [7, 8]], [[1, 2], [3, 4]]], dtype=torch.uint8
            ),
        ),
        # Test BGR to RGB conversion with torch tensor
        (
            'tensor',
            'BGR2RGB',
            torch.tensor(
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=torch.uint8
            ),
            torch.tensor(
                [[[9, 10], [11, 12]], [[5, 6], [7, 8]], [[1, 2], [3, 4]]], dtype=torch.uint8
            ),
        ),
        # Test RGB to BGR conversion with Image
        (
            'image',
            'RGB2BGR',
            types.Image.fromarray(
                np.array([[[1, 5, 9], [2, 6, 10]], [[3, 7, 11], [4, 8, 12]]], dtype=np.uint8),
                types.ColorFormat.RGB,
            ),
            np.array([[[9, 5, 1], [10, 6, 2]], [[11, 7, 3], [12, 8, 4]]], dtype=np.uint8),
        ),
        # Test BGR to RGB conversion with Image
        (
            'image',
            'BGR2RGB',
            types.Image.fromarray(
                np.array([[[1, 5, 9], [2, 6, 10]], [[3, 7, 11], [4, 8, 12]]], dtype=np.uint8),
                types.ColorFormat.BGR,
            ),
            np.array([[[9, 5, 1], [10, 6, 2]], [[11, 7, 3], [12, 8, 4]]], dtype=np.uint8),
        ),
        # Test RGB to GRAY conversion with torch tensor
        (
            'tensor',
            'RGB2GRAY',
            torch.tensor(
                [[[1, 2], [3, 4]], [[5, 6], [7, 8]], [[9, 10], [11, 12]]], dtype=torch.uint8
            ),
            torch.tensor([[[4, 5], [6, 7]]], dtype=torch.uint8),
        ),
    ],
)
def test_convert_color_exec_torch_without_context_info_populated(
    input_type, format, input_data, expected_output
):
    op = operators.ConvertColor(format=format)

    result = op.exec_torch(input_data)

    if input_type == 'tensor':
        assert isinstance(result, torch.Tensor)
        np.testing.assert_array_equal(result.numpy(), expected_output.numpy())
    else:
        assert isinstance(result, types.Image)
        np.testing.assert_array_equal(result.asarray(), expected_output)


def test_convert_color_invalid_input():
    op = operators.ConvertColor(format='RGB2BGR')
    with pytest.raises(TypeError, match="Unsupported input type: <class 'str'>"):
        op.exec_torch("invalid input")


def test_convert_color_invalid_format():
    with pytest.raises(ValueError, match="Unsupported conversion: INVALID2FORMAT"):
        operators.ConvertColor(format='INVALID2FORMAT')


class TestNormalizeCombineNormalizations:
    """Test suite for the combine_normalizations method of the Normalize operator."""

    def test_combine_identity_normalizations(self):
        """Test combining with identity normalization (mean=0, std=1)."""
        # First normalization: (x - 128) / 255
        norm1 = operators.Normalize(mean=[128.0, 128.0, 128.0], std=[255.0, 255.0, 255.0])

        # Second normalization: identity (x - 0) / 1
        mean2 = [0.0, 0.0, 0.0]
        std2 = [1.0, 1.0, 1.0]

        norm1.combine_normalizations(mean2, std2)

        # Result should be the same as original
        np.testing.assert_array_almost_equal(norm1.mean_values, [128.0, 128.0, 128.0])
        np.testing.assert_array_almost_equal(norm1.std_values, [255.0, 255.0, 255.0])

    def test_combine_two_normalizations(self):
        """Test combining two non-trivial normalizations."""
        # First normalization: (x - 0) / 255  (scale to [0, 1])
        norm1 = operators.Normalize(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0])

        # Second normalization: ImageNet normalization on [0, 1] input
        # (x - mean) / std
        mean2 = [0.485, 0.456, 0.406]
        std2 = [0.229, 0.224, 0.225]

        norm1.combine_normalizations(mean2, std2)

        # Combined: ((x - 0) / 255 - mean2) / std2
        # = (x - mean2 * 255) / (255 * std2)
        expected_mean = [m * 255 for m in mean2]
        expected_std = [s * 255 for s in std2]

        np.testing.assert_array_almost_equal(norm1.mean_values, expected_mean, decimal=2)
        np.testing.assert_array_almost_equal(norm1.std_values, expected_std, decimal=2)

    def test_combine_with_different_means(self):
        """Test combining normalizations with different mean values."""
        # First normalization: (x - 100) / 50
        norm1 = operators.Normalize(mean=[100.0, 100.0, 100.0], std=[50.0, 50.0, 50.0])

        # Second normalization: (x - 0.5) / 0.5
        mean2 = [0.5, 0.5, 0.5]
        std2 = [0.5, 0.5, 0.5]

        norm1.combine_normalizations(mean2, std2)

        # Combined: ((x - 100) / 50 - 0.5) / 0.5
        # = (x - 100 - 0.5 * 50) / (50 * 0.5)
        # = (x - 125) / 25
        expected_mean = [125.0, 125.0, 125.0]
        expected_std = [25.0, 25.0, 25.0]

        np.testing.assert_array_almost_equal(norm1.mean_values, expected_mean)
        np.testing.assert_array_almost_equal(norm1.std_values, expected_std)

    def test_combine_per_channel_values(self):
        """Test combining normalizations with different values per channel."""
        # RGB channels with different normalizations
        norm1 = operators.Normalize(mean=[100.0, 110.0, 120.0], std=[50.0, 60.0, 70.0])

        mean2 = [0.4, 0.5, 0.6]
        std2 = [0.2, 0.3, 0.4]

        norm1.combine_normalizations(mean2, std2)

        # Combined per channel: (x - mean1 - mean2 * std1) / (std1 * std2)
        expected_mean = [
            100.0 + 0.4 * 50.0,  # 120.0
            110.0 + 0.5 * 60.0,  # 140.0
            120.0 + 0.6 * 70.0,  # 162.0
        ]
        expected_std = [
            50.0 * 0.2,  # 10.0
            60.0 * 0.3,  # 18.0
            70.0 * 0.4,  # 28.0
        ]

        np.testing.assert_array_almost_equal(norm1.mean_values, expected_mean)
        np.testing.assert_array_almost_equal(norm1.std_values, expected_std)

    def test_combine_scalar_values(self):
        """Test combining normalizations with scalar values."""
        norm1 = operators.Normalize(mean=128.0, std=255.0)

        mean2 = 0.5
        std2 = 2.0

        norm1.combine_normalizations(mean2, std2)

        # Combined: (x - 128 - 0.5 * 255) / (255 * 2.0)
        # = (x - 255.5) / 510.0
        expected_mean = 128.0 + 0.5 * 255.0  # 255.5
        expected_std = 255.0 * 2.0  # 510.0

        # Check if values are close (accounting for scalar or array)
        if isinstance(norm1.mean_values, (list, np.ndarray)):
            np.testing.assert_array_almost_equal(norm1.mean_values, [expected_mean])
            np.testing.assert_array_almost_equal(norm1.std_values, [expected_std])
        else:
            assert np.isclose(norm1.mean_values, expected_mean)
            assert np.isclose(norm1.std_values, expected_std)

    def test_combine_zero_std_second_normalization(self):
        """Test combining when second normalization has zero std (edge case)."""
        norm1 = operators.Normalize(mean=[100.0, 100.0, 100.0], std=[50.0, 50.0, 50.0])

        # Second normalization with zero std
        mean2 = [0.0, 0.0, 0.0]
        std2 = [0.0, 0.0, 0.0]

        norm1.combine_normalizations(mean2, std2)

        # Result: std becomes 0 (division by zero case)
        np.testing.assert_array_almost_equal(norm1.mean_values, [100.0, 100.0, 100.0])
        np.testing.assert_array_almost_equal(norm1.std_values, [0.0, 0.0, 0.0])

    def test_combine_negative_values(self):
        """Test combining normalizations with negative values."""
        norm1 = operators.Normalize(mean=[-10.0, -20.0, -30.0], std=[5.0, 10.0, 15.0])

        mean2 = [-0.5, -0.3, -0.1]
        std2 = [0.5, 0.7, 0.9]

        norm1.combine_normalizations(mean2, std2)

        # Combined: (x - mean1 - mean2 * std1) / (std1 * std2)
        expected_mean = [
            -10.0 + (-0.5) * 5.0,  # -12.5
            -20.0 + (-0.3) * 10.0,  # -23.0
            -30.0 + (-0.1) * 15.0,  # -31.5
        ]
        expected_std = [
            5.0 * 0.5,  # 2.5
            10.0 * 0.7,  # 7.0
            15.0 * 0.9,  # 13.5
        ]

        np.testing.assert_array_almost_equal(norm1.mean_values, expected_mean)
        np.testing.assert_array_almost_equal(norm1.std_values, expected_std)

    def test_combine_grayscale_single_channel(self):
        """Test combining normalizations for grayscale (single channel)."""
        norm1 = operators.Normalize(mean=127.5, std=127.5)

        mean2 = 0.5
        std2 = 0.5

        norm1.combine_normalizations(mean2, std2)

        # Combined: (x - 127.5 - 0.5 * 127.5) / (127.5 * 0.5)
        # = (x - 191.25) / 63.75
        expected_mean = 127.5 + 0.5 * 127.5  # 191.25
        expected_std = 127.5 * 0.5  # 63.75

        if isinstance(norm1.mean_values, (list, np.ndarray)):
            assert len(norm1.mean_values) == 1
            assert np.isclose(norm1.mean_values[0], expected_mean)
            assert np.isclose(norm1.std_values[0], expected_std)
        else:
            assert np.isclose(norm1.mean_values, expected_mean)
            assert np.isclose(norm1.std_values, expected_std)

    def test_combine_modifies_in_place(self):
        """Test that combine_normalizations modifies the operator in place."""
        original_mean = [100.0, 100.0, 100.0]
        original_std = [50.0, 50.0, 50.0]
        norm1 = operators.Normalize(mean=original_mean.copy(), std=original_std.copy())

        mean2 = [0.5, 0.5, 0.5]
        std2 = [2.0, 2.0, 2.0]

        # Store original reference
        norm1_ref = norm1

        # Combine normalizations
        result = norm1.combine_normalizations(mean2, std2)

        # Should modify in place (return None or self)
        assert norm1_ref is norm1

        # Check expected values
        expected_mean = [m1 + m2 * s1 for m1, m2, s1 in zip(original_mean, mean2, original_std)]
        expected_std = [s1 * s2 for s1, s2 in zip(original_std, std2)]

        # Convert to numpy arrays for comparison
        result_mean = np.asarray(norm1.mean_values, dtype=float)
        result_std = np.asarray(norm1.std_values, dtype=float)

        # Values should be updated
        assert not np.allclose(result_mean, original_mean)
        assert not np.allclose(result_std, original_std)

        np.testing.assert_array_almost_equal(result_mean, expected_mean)
        np.testing.assert_array_almost_equal(result_std, expected_std)

    def test_combine_imagenet_preprocessing(self):
        """Test realistic ImageNet preprocessing scenario."""
        # First: scale from [0, 255] to [0, 1]
        norm1 = operators.Normalize(mean=[0.0, 0.0, 0.0], std=[255.0, 255.0, 255.0])

        # Second: apply ImageNet normalization
        imagenet_mean = [0.485, 0.456, 0.406]
        imagenet_std = [0.229, 0.224, 0.225]

        norm1.combine_normalizations(imagenet_mean, imagenet_std)

        # Test with sample values
        test_input = np.array([123.675, 116.28, 103.53])  # typical mean RGB values

        # Apply combined normalization manually
        result_combined = (test_input - norm1.mean_values) / norm1.std_values

        # Apply two-step normalization
        step1 = (test_input - 0.0) / 255.0
        result_twostep = (step1 - imagenet_mean) / imagenet_std

        # Should be equivalent
        np.testing.assert_array_almost_equal(result_combined, result_twostep, decimal=5)

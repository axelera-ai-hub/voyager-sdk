# Copyright Axelera AI, 2024
from pathlib import Path
from unittest.mock import ANY, patch

import cv2
import numpy as np
import pytest
from test_axelera_operators_preprocessing import arithmetic

torch = pytest.importorskip("torch")
import torchvision.transforms.functional as TF

from axelera import types
from axelera.app import gst_builder, operators


def _gen_gst(op, stream_idx=''):
    gst = gst_builder.Builder()
    op.build_gst(gst, stream_idx)
    return list(gst)


def test_type_cast():
    op = operators.mega.TypeCastAndNormalize(datatype='float32')
    data = np.arange(5 * 4 * 3, dtype=np.uint8).reshape(5, 4, 3)
    got = op.exec_torch(torch.from_numpy(data))
    np.testing.assert_equal(got.numpy(), data.astype('float32'))
    assert got.numpy().flags['C_CONTIGUOUS']
    with pytest.raises(NotImplementedError, match=r"float32 is not supported in gst"):
        _gen_gst(op)


def test_type_cast_and_norm_invalid_input_type_torch():
    op = operators.mega.TypeCastAndNormalize(datatype='uint8')
    data = np.arange(5 * 4 * 3, dtype=np.int16).reshape(5, 4, 3)
    with pytest.raises(TypeError, match=r"Input must be a torch tensor of uint8 not torch.int16"):
        op.exec_torch(torch.from_numpy(data))


def test_resize_and_convert():
    with patch.object(TF, 'resize', wraps=TF.resize) as resize:
        op = operators.mega.ResizeAndConvert(width=20, height=10, format='bgr2rgb')
        op.configure_model_and_context_info(
            types.ModelInfo('modelname', types.TaskCategory.Classification, [3, 20, 40]),
            operators.PipelineContext(color_format='BGR'),
            'task_name',
            0,
            '',
            Path('.'),
        )
        i = types.Image.fromarray(np.zeros((20, 40, 3), dtype=np.uint8), types.ColorFormat.BGR)
        torch_out = op.exec_torch(i)
        w, h = (20, 10)
        resize.assert_called_once_with(ANY, (h, w), ANY, max_size=ANY, antialias=ANY)
        assert torch_out.asarray().shape == (10, 20, 3)
        assert torch_out.color_format == types.ColorFormat.RGB

        gst_exp_out = [
            {
                'instance': 'axtransform',
                'lib': 'libtransform_resize.so',
                'options': 'width:20;height:10;letterbox:0',
            },
            {
                'instance': 'videoconvert',
            },
            {
                'caps': 'video/x-raw,format=RGBA',
                'instance': 'capsfilter',
            },
        ]
        assert _gen_gst(op) == gst_exp_out


def test_ax_letterbox_to_tensor_and_in_place_3_channels():
    op = operators.mega.LetterboxToTensorAndNormalise(
        height=480, width=640, mean='104/255, 117/255, 123/255', std='1/255, 1/255, 1/255'
    )
    assert _gen_gst(op) == [
        {
            'instance': 'axtransform',
            'lib': 'libtransform_resize.so',
            'options': 'width:640;height:480;padding:114;to_tensor:1;letterbox:1;scale_up:1',
        },
        {
            'instance': 'axinplace',
            'mode': 'write',
            'options': f'mean:0.408,0.459,0.482;std:0.004;simd:avx2',
            'lib': 'libinplace_normalize.so',
        },
    ]


def test_ax_to_tensor_and_in_place_3_channels():
    op = operators.mega.ToTensorAndNormalise(
        mean='104/255, 117/255, 123/255', std='1/255, 1/255, 1/255'
    )
    assert _gen_gst(op) == [
        {'instance': 'axtransform', 'lib': 'libtransform_totensor.so', 'options': 'type:int8'},
        {
            'instance': 'axinplace',
            'mode': 'write',
            'options': f'mean:0.408,0.459,0.482;std:0.004;simd:avx2',
            'lib': 'libinplace_normalize.so',
        },
    ]


def test_ax_to_tensor_and_in_place_1_channels():
    op = operators.mega.ToTensorAndNormalise(mean='104/255', std='1/255')
    assert _gen_gst(op) == [
        {'instance': 'axtransform', 'lib': 'libtransform_totensor.so', 'options': 'type:int8'},
        {
            'instance': 'axinplace',
            'mode': 'write',
            'options': f'mean:0.408;std:0.004;simd:avx2',
            'lib': 'libinplace_normalize.so',
        },
    ]


def test_ax_to_tensor_and_linear_scale_1_channels():
    op = operators.mega.ToTensorAndLinearScaling(shift='108', mean='1')
    assert _gen_gst(op) == [
        {'instance': 'axtransform', 'lib': 'libtransform_totensor.so', 'options': 'type:int8'},
        {
            'instance': 'axinplace',
            'lib': 'libinplace_normalize.so',
            'mode': 'write',
            'options': 'mean:-0.423529;std:0.003922;simd:avx2;quant_scale:0.00392156862745098;quant_zeropoint:0',
        },
    ]


def test_ax_to_tensor_and_linear_scale_3_channels():
    op = operators.mega.ToTensorAndLinearScaling(shift='108, 110, 114', mean='1, 1.1, 1.2')
    assert _gen_gst(op) == [
        {'instance': 'axtransform', 'lib': 'libtransform_totensor.so', 'options': 'type:int8'},
        {
            'instance': 'axinplace',
            'lib': 'libinplace_normalize.so',
            'mode': 'write',
            'options': 'mean:-0.423529,-0.47451,-0.536471;std:0.003922,0.004314,0.004706;simd:avx2;quant_scale:0.00392156862745098;quant_zeropoint:0',
        },
    ]


def test_ax_to_tensor_and_in_place_3_channels_with_pads_and_quant():
    mi = types.ModelInfo(
        'modelname',
        types.TaskCategory.Classification,
        [3, 224, 244],
    )
    mi.manifest = types.Manifest(
        'modellib',
        quantize_params=[(0.1, 0.2)],
        dequantize_params=[(0.3, 0.4)],
        input_shapes=[(1, 3, 224, 224)],
        model_lib_file='model.json',
    )
    op = operators.mega.ToTensorAndNormalise(
        mean='104/255, 117/255, 123/255', std='1/255, 1/255, 1/255'
    )
    op.configure_model_and_context_info(
        mi, operators.PipelineContext(), "task_name", 0, '', Path('.')
    )
    assert _gen_gst(op) == [
        {'instance': 'axtransform', 'lib': 'libtransform_totensor.so', 'options': 'type:int8'},
        {
            'instance': 'axinplace',
            'mode': 'write',
            'options': f'mean:0.408,0.459,0.482;std:0.004;simd:avx2;quant_scale:0.1;quant_zeropoint:0.2',
            'lib': 'libinplace_normalize.so',
        },
    ]


def test_ax_opencl_to_tensor_normalize():
    mi = types.ModelInfo(
        'modelname',
        types.TaskCategory.Classification,
        [3, 224, 244],
    )
    mi.manifest = types.Manifest(
        'modellib',
        quantize_params=[(0.1, -14)],
        dequantize_params=[(0.3, 0.4)],
        input_shapes=[(1, 3, 224, 224)],
        model_lib_file='model.json',
    )
    op = operators.mega.OpenCLToTensorAndNormalize(
        mean='104/255, 117/255, 123/255', std='1/255, 1/255, 1/255'
    )
    op.configure_model_and_context_info(
        mi, operators.PipelineContext(), "task_name", 0, '', Path('.')
    )
    print(_gen_gst(op))
    assert _gen_gst(op) == [
        {
            'instance': 'axtransform',
            'lib': 'libtransform_normalize_cl.so',
            'options': 'to_tensor:1;mean:0.408,0.459,0.482;std:0.004,0.004,0.004;quant_scale:0.1;quant_zeropoint:-14.0',
        },
    ]

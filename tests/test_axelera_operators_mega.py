# Copyright Axelera AI, 2023
from pathlib import Path
from unittest.mock import ANY, MagicMock, patch

import numpy as np
import pytest
from test_axelera_operators_preprocessing import arithmetic

torch = pytest.importorskip("torch")
import torchvision.transforms.functional as TF

from axelera import types
from axelera.app import config, gst_builder, operators


def _gen_gst(op, stream_idx=''):
    # note we use the old builder so we can test the gst output, the new builder
    # consumes the gst output in readiness for an axinferencenet. A forthcoming
    # PR will tidy this up  by having explicit begin/end axinferencenet
    gst = gst_builder._OldBuilder(None, None, 16)
    op.build_gst(gst, stream_idx)
    return list(gst)


def test_type_cast():
    op = operators.mega.TypeCastAndNormalize(datatype='float32')
    data = np.arange(5 * 4 * 3, dtype=np.uint8).reshape(5, 4, 3)
    got = op.exec_torch(torch.from_numpy(data))
    np.testing.assert_equal(got.numpy(), data.astype('float32'))
    assert got.numpy().flags['C_CONTIGUOUS']


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
            Path('.'),
            task_graph=None,
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
                'instance': 'axtransform',
                'lib': 'libtransform_colorconvert.so',
                'options': 'format:rgba',
            },
        ]
        assert _gen_gst(op) == gst_exp_out


def test_ax_letterbox_to_tensor_and_in_place_3_channels():
    op = operators.mega.LetterboxToTensorAndNormalise(
        height=480, width=640, mean='104/255, 117/255, 123/255', std='1/255, 1/255, 1/255'
    )
    _tg = MagicMock()
    _tg.get_master.return_value = ''
    op.configure_model_and_context_info(None, operators.PipelineContext(), 'task', 0, None, _tg)
    assert _gen_gst(op) == [
        {
            'instance': 'axtransform',
            'lib': 'libtransform_resize.so',
            'options': 'width:640;height:480;padding:114;to_tensor:1;letterbox:1;scale_up:1',
        },
        {
            'instance': 'axinplace',
            'mode': 'write',
            'options': f'mean:0.407843,0.458824,0.482353;std:0.003922;simd:{operators.mega.which_simd()}',
            'lib': 'libinplace_normalize.so',
        },
    ]


def test_ax_to_tensor_and_in_place_3_channels():
    op = operators.mega.ToTensorAndNormalise(
        mean='104/255, 117/255, 123/255', std='1/255, 1/255, 1/255'
    )
    _tg = MagicMock()
    _tg.get_master.return_value = ''
    op.configure_model_and_context_info(None, operators.PipelineContext(), 'task', 0, None, _tg)
    assert _gen_gst(op) == [
        {'instance': 'axtransform', 'lib': 'libtransform_totensor.so', 'options': 'type:int8'},
        {
            'instance': 'axinplace',
            'mode': 'write',
            'options': f'mean:0.407843,0.458824,0.482353;std:0.003922;simd:{operators.mega.which_simd()}',
            'lib': 'libinplace_normalize.so',
        },
    ]


def test_ax_to_tensor_and_no_normalize():
    op = operators.mega.ToTensorAndNormalise()
    _tg = MagicMock()
    _tg.get_master.return_value = ''
    op.configure_model_and_context_info(None, operators.PipelineContext(), 'task', 0, None, _tg)
    assert _gen_gst(op) == [
        {'instance': 'axtransform', 'lib': 'libtransform_totensor.so', 'options': 'type:int8'},
        {
            'instance': 'axinplace',
            'mode': 'write',
            'options': f'mean:0.;std:1.;simd:{operators.mega.which_simd()}',
            'lib': 'libinplace_normalize.so',
        },
    ]


def test_ax_to_tensor_and_in_place_1_channels():
    op = operators.mega.ToTensorAndNormalise(mean='104/255', std='1/255')
    _tg = MagicMock()
    _tg.get_master.return_value = ''
    op.configure_model_and_context_info(None, operators.PipelineContext(), 'task', 0, None, _tg)
    assert _gen_gst(op) == [
        {'instance': 'axtransform', 'lib': 'libtransform_totensor.so', 'options': 'type:int8'},
        {
            'instance': 'axinplace',
            'mode': 'write',
            'options': f'mean:0.407843;std:0.003922;simd:{operators.mega.which_simd()}',
            'lib': 'libinplace_normalize.so',
        },
    ]


def test_ax_to_tensor_and_linear_scale_1_channels():
    op = operators.mega.ToTensorAndLinearScaling(shift='108', mean='1')
    _tg = MagicMock()
    _tg.get_master.return_value = ''
    op.configure_model_and_context_info(None, operators.PipelineContext(), 'task', 0, None, _tg)
    assert _gen_gst(op) == [
        {'instance': 'axtransform', 'lib': 'libtransform_totensor.so', 'options': 'type:int8'},
        {
            'instance': 'axinplace',
            'lib': 'libinplace_normalize.so',
            'mode': 'write',
            'options': f'mean:-0.423529;std:0.003922;simd:{operators.mega.which_simd()};quant_scale:0.00392156862745098;quant_zeropoint:0',
        },
    ]


def test_ax_to_tensor_and_linear_scale_3_channels():
    op = operators.mega.ToTensorAndLinearScaling(shift='108, 110, 114', mean='1, 1.1, 1.2')
    _tg = MagicMock()
    _tg.get_master.return_value = ''
    op.configure_model_and_context_info(None, operators.PipelineContext(), 'task', 0, None, _tg)
    assert _gen_gst(op) == [
        {'instance': 'axtransform', 'lib': 'libtransform_totensor.so', 'options': 'type:int8'},
        {
            'instance': 'axinplace',
            'lib': 'libinplace_normalize.so',
            'mode': 'write',
            'options': f'mean:-0.423529,-0.47451,-0.536471;std:0.003922,0.004314,0.004706;simd:{operators.mega.which_simd()};quant_scale:0.00392156862745098;quant_zeropoint:0',
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
        input_shapes=[(1, 3, 224, 224)],
        input_dtypes=['uint8'],
        output_shapes=[(1, 1000)],
        output_dtypes=['float32'],
        quantize_params=[(0.1, 0.2)],
        dequantize_params=[(0.3, 0.4)],
        model_lib_file='model.json',
    )
    op = operators.mega.ToTensorAndNormalise(
        mean='104/255, 117/255, 123/255', std='1/255, 1/255, 1/255'
    )
    mock_task_graph = MagicMock()
    mock_task_graph.get_master.return_value = "mocked_master_value"
    op.configure_model_and_context_info(
        mi, operators.PipelineContext(), "task_name", 0, Path('.'), task_graph=mock_task_graph
    )
    assert _gen_gst(op) == [
        {'instance': 'axtransform', 'lib': 'libtransform_totensor.so', 'options': 'type:int8'},
        {
            'instance': 'axinplace',
            'mode': 'write',
            'options': f'mean:0.407843,0.458824,0.482353;std:0.003922;simd:{operators.mega.which_simd()};quant_scale:0.1;quant_zeropoint:0.2',
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
        input_shapes=[(1, 3, 224, 224)],
        input_dtypes=['uint8'],
        output_shapes=[(1, 1000)],
        output_dtypes=['float32'],
        quantize_params=[(0.1, -14)],
        dequantize_params=[(0.3, 0.4)],
        model_lib_file='model.json',
    )
    op = operators.mega.OpenCLToTensorAndNormalize(
        mean='104/255, 117/255, 123/255', std='1/255, 1/255, 1/255'
    )
    mock_task_graph = MagicMock()
    mock_task_graph.get_master.return_value = "mocked_master_value"
    op.configure_model_and_context_info(
        mi, operators.PipelineContext(), "task_name", 0, Path('.'), task_graph=mock_task_graph
    )
    print(_gen_gst(op))
    assert _gen_gst(op) == [
        {
            'instance': 'axtransform',
            'lib': 'libtransform_normalize_cl.so',
            'options': 'to_tensor:1;mean:0.407843,0.458824,0.482353;std:0.003922,0.003922,0.003922;quant_scale:0.1;quant_zeropoint:-14.0',
        },
    ]


@pytest.mark.parametrize(
    'format,afmethod,colorconvertmethod',
    [
        ('rgb', config.VideoFlipMethod.clockwise, 'clockwise'),
        ('rgb', config.VideoFlipMethod.rotate_180, 'rotate-180'),
        ('rgb', config.VideoFlipMethod.counterclockwise, 'counterclockwise'),
        ('rgb', config.VideoFlipMethod.horizontal_flip, 'horizontal-flip'),
        ('rgb', config.VideoFlipMethod.vertical_flip, 'vertical-flip'),
        ('rgb', config.VideoFlipMethod.upper_left_diagonal, 'upper-left-diagonal'),
        ('bgr', config.VideoFlipMethod.upper_right_diagonal, 'upper-right-diagonal'),
    ],
)
def test_opencl_videoflip_and_color(format, afmethod, colorconvertmethod):
    op = operators.mega.OpenCLVideoFlipAndColorConvert(format=format, method=afmethod)
    gst_exp_out = [
        {
            'instance': 'axtransform',
            'lib': 'libtransform_colorconvert_cl.so',
            'options': f'format:{format}a;flip_method:{colorconvertmethod}',
        },
    ]
    assert _gen_gst(op) == gst_exp_out


class TestPreambleIntegration:
    """Test suite for preamble ONNX preprocessing graph integration in mega operators."""

    def _create_test_onnx_preamble(self, tmp_path, sub_values, div_values):
        """Helper to create a test ONNX preamble file."""
        import onnx
        from onnx import TensorProto, helper, numpy_helper

        X = helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 3, 224, 224])
        Y = helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 3, 224, 224])

        sub_const = numpy_helper.from_array(np.array(sub_values, dtype=np.float32), "sub_const")
        div_const = numpy_helper.from_array(np.array(div_values, dtype=np.float32), "div_const")

        sub_node = helper.make_node("Sub", ["input", "sub_const"], ["after_sub"])
        div_node = helper.make_node("Div", ["after_sub", "div_const"], ["output"])

        graph = helper.make_graph(
            [sub_node, div_node],
            "preprocess_graph",
            [X],
            [Y],
            initializer=[sub_const, div_const],
        )

        model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
        preamble_path = tmp_path / "preprocess_graph.onnx"
        onnx.save(model, str(preamble_path))
        return preamble_path

    def test_effective_norm_not_set_before_configure(self):
        """Test that _effective_mean/_effective_std are not set before configure is called."""
        op = operators.mega.ToTensorAndNormalise()
        assert not hasattr(op, '_effective_mean')
        assert not hasattr(op, '_effective_std')

    def test_effective_norm_set_when_preprocess_graph_exists(self, tmp_path):
        """Test that _effective_mean/_effective_std are set when manifest has preprocess_graph."""
        self._create_test_onnx_preamble(
            tmp_path, [123.675, 116.28, 103.53], [58.395, 57.12, 57.375]
        )

        mi = types.ModelInfo('modelname', types.TaskCategory.Classification, [3, 224, 224])
        mi.manifest = types.Manifest(
            'modellib',
            input_shapes=[(1, 3, 224, 224)],
            input_dtypes=['uint8'],
            output_shapes=[(1, 1000)],
            output_dtypes=['float32'],
            quantize_params=[(1.0, 0)],
            dequantize_params=[(1.0, 0)],
            model_lib_file='model.json',
            preprocess_graph='preprocess_graph.onnx',
        )

        op = operators.mega.ToTensorAndNormalise()
        mock_task_graph = MagicMock()
        mock_task_graph.get_master.return_value = "mocked_master_value"
        op.configure_model_and_context_info(
            mi, operators.PipelineContext(), "task_name", 0, tmp_path, task_graph=mock_task_graph
        )

        assert op._effective_mean is not None
        assert op._effective_std is not None

    def test_effective_norm_set_to_base_values_when_no_preprocess_graph(self, tmp_path):
        """Test that _effective_mean/_effective_std reflect base norm when no preamble."""
        mi = types.ModelInfo('modelname', types.TaskCategory.Classification, [3, 224, 224])
        mi.manifest = types.Manifest(
            'modellib',
            input_shapes=[(1, 3, 224, 224)],
            input_dtypes=['uint8'],
            output_shapes=[(1, 1000)],
            output_dtypes=['float32'],
            quantize_params=[(1.0, 0)],
            dequantize_params=[(1.0, 0)],
            model_lib_file='model.json',
            preprocess_graph=None,
        )

        op = operators.mega.ToTensorAndNormalise(mean='0.5', std='0.25')
        mock_task_graph = MagicMock()
        mock_task_graph.get_master.return_value = "mocked_master_value"
        op.configure_model_and_context_info(
            mi, operators.PipelineContext(), "task_name", 0, tmp_path, task_graph=mock_task_graph
        )

        assert op._effective_mean == [0.5]
        assert op._effective_std == [0.25]

    def test_combine_normalization_applied_when_preamble_exists(self, tmp_path):
        """Test that normalization is combined with preamble constants when file exists."""
        # Create preamble with ImageNet normalization on [0,1] input
        # (x - mean) / std where mean and std are for [0,1] range
        imagenet_mean_01 = [0.485, 0.456, 0.406]
        imagenet_std_01 = [0.229, 0.224, 0.225]

        self._create_test_onnx_preamble(tmp_path, imagenet_mean_01, imagenet_std_01)

        mi = types.ModelInfo('modelname', types.TaskCategory.Classification, [3, 224, 224])
        mi.manifest = types.Manifest(
            'modellib',
            input_shapes=[(1, 3, 224, 224)],
            input_dtypes=['uint8'],
            output_shapes=[(1, 1000)],
            output_dtypes=['float32'],
            quantize_params=[(1.0, 0)],
            dequantize_params=[(1.0, 0)],
            model_lib_file='model.json',
            preprocess_graph='preprocess_graph.onnx',
        )

        # Start with scale-only normalization (0 mean, 1/255 std)
        op = operators.mega.ToTensorAndNormalise(mean='0, 0, 0', std='1/255, 1/255, 1/255')
        mock_task_graph = MagicMock()
        mock_task_graph.get_master.return_value = "mocked_master_value"
        op.configure_model_and_context_info(
            mi, operators.PipelineContext(), "task_name", 0, tmp_path, task_graph=mock_task_graph
        )

        # Build GST pipeline which should apply the combined normalization
        gst_output = _gen_gst(op)

        # Extract mean and std from the GST options
        normalize_step = [
            step for step in gst_output if step.get('lib') == 'libinplace_normalize.so'
        ][0]
        options = normalize_step['options']

        # Parse mean values from options string
        mean_match = [opt for opt in options.split(';') if opt.startswith('mean:')]
        assert len(mean_match) == 1

        mean_str = mean_match[0].split(':')[1]
        mean_values = [float(x) for x in mean_str.split(',')]

        # Expected: (x - 0) / (1/255) then (y - imagenet_mean_01) / imagenet_std_01
        # where y = x / (1/255) = x * 255 is the output of the first normalization
        # Combined: (x - (0 + imagenet_mean_01 * (1/255))) / ((1/255) * imagenet_std_01)
        # = (x - imagenet_mean_01 / 255) / (imagenet_std_01 / 255)
        expected_mean = [m / 255 for m in imagenet_mean_01]
        expected_std = [s / 255 for s in imagenet_std_01]

        # Check that normalization was combined (values should be close to expected)
        np.testing.assert_array_almost_equal(mean_values, expected_mean, decimal=5)

        # Also check std values
        std_match = [opt for opt in options.split(';') if opt.startswith('std:')]
        std_str = std_match[0].split(':')[1]
        std_values = [float(x) for x in std_str.split(',')]
        np.testing.assert_array_almost_equal(std_values, expected_std, decimal=5)

    def test_preamble_integration_opencl_resize_operator(self, tmp_path):
        """Test preamble integration for OpenCLResizeColorConverToTensorAndNormalize."""
        self._create_test_onnx_preamble(tmp_path, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        mi = types.ModelInfo('modelname', types.TaskCategory.Classification, [3, 224, 224])
        mi.manifest = types.Manifest(
            'modellib',
            input_shapes=[(1, 3, 224, 224)],
            input_dtypes=['uint8'],
            output_shapes=[(1, 1000)],
            output_dtypes=['float32'],
            quantize_params=[(1.0, 0)],
            dequantize_params=[(1.0, 0)],
            model_lib_file='model.json',
            preprocess_graph='preprocess_graph.onnx',
        )

        op = operators.mega.OpenCLResizeColorConverToTensorAndNormalize(
            width=224, height=224, mean='0, 0, 0', std='1/255, 1/255, 1/255'
        )
        mock_task_graph = MagicMock()
        mock_task_graph.get_master.return_value = "mocked_master_value"
        op.configure_model_and_context_info(
            mi, operators.PipelineContext(), "task_name", 0, tmp_path, task_graph=mock_task_graph
        )

        assert op._effective_mean is not None
        assert op._effective_std is not None

        # Build GST and check normalization was applied
        gst_output = _gen_gst(op)
        assert len(gst_output) > 0

        # Find the transform step with resize and normalization
        # OpenCLResizeColorConverToTensorAndNormalize uses libtransform_resize_cl.so
        # which combines resize, color conversion, to-tensor, and normalization
        transform_steps = [
            step for step in gst_output if step.get('lib') == 'libtransform_resize_cl.so'
        ]
        assert len(transform_steps) == 1

        options = transform_steps[0]['options']
        assert 'mean:' in options
        assert 'std:' in options

    def test_preamble_integration_letterbox_operator(self, tmp_path):
        """Test preamble integration for LetterboxToTensorAndNormalise."""
        self._create_test_onnx_preamble(tmp_path, [128.0, 128.0, 128.0], [128.0, 128.0, 128.0])

        mi = types.ModelInfo('modelname', types.TaskCategory.ObjectDetection, [3, 640, 640])
        mi.manifest = types.Manifest(
            'modellib',
            input_shapes=[(1, 3, 640, 640)],
            input_dtypes=['uint8'],
            output_shapes=[(1, 25200, 85)],
            output_dtypes=['float32'],
            quantize_params=[(1.0, 0)],
            dequantize_params=[(1.0, 0)],
            model_lib_file='model.json',
            preprocess_graph='preprocess_graph.onnx',
        )

        op = operators.mega.LetterboxToTensorAndNormalise(
            width=640, height=640, mean='0, 0, 0', std='1/255, 1/255, 1/255'
        )
        mock_task_graph = MagicMock()
        mock_task_graph.get_master.return_value = "mocked_master_value"
        op.configure_model_and_context_info(
            mi, operators.PipelineContext(), "task_name", 0, tmp_path, task_graph=mock_task_graph
        )

        assert op._effective_mean is not None
        assert op._effective_std is not None

        # Build GST
        gst_output = _gen_gst(op)

        # Find normalization step
        normalize_steps = [
            step for step in gst_output if step.get('lib') == 'libinplace_normalize.so'
        ]
        assert len(normalize_steps) == 1

    def test_format_function_used_instead_of_fstring(self):
        """Test that format() function is used instead of f-string for floating point formatting."""
        op = operators.mega.ToTensorAndNormalise(mean='104/255, 117/255, 123/255', std='1/255')

        mi = types.ModelInfo('modelname', types.TaskCategory.Classification, [3, 224, 224])
        mi.manifest = types.Manifest(
            'modellib',
            input_shapes=[(1, 3, 224, 224)],
            input_dtypes=['uint8'],
            output_shapes=[(1, 1000)],
            output_dtypes=['float32'],
            quantize_params=[(1.0, 0)],
            dequantize_params=[(1.0, 0)],
            model_lib_file='model.json',
        )

        mock_task_graph = MagicMock()
        mock_task_graph.get_master.return_value = "mocked_master_value"
        op.configure_model_and_context_info(
            mi, operators.PipelineContext(), "task_name", 0, Path('.'), task_graph=mock_task_graph
        )

        gst_output = _gen_gst(op)

        # Extract options string
        normalize_step = [
            step for step in gst_output if step.get('lib') == 'libinplace_normalize.so'
        ][0]
        options = normalize_step['options']

        # Check that values are properly formatted (no trailing zeros except after decimal point)
        # The format should be clean, e.g., "0.407843" not "0.407843000000"
        assert 'mean:' in options
        mean_part = [opt for opt in options.split(';') if opt.startswith('mean:')][0]
        mean_values = mean_part.split(':')[1].split(',')

        for val in mean_values:
            # Should not have excessive trailing zeros
            assert not val.endswith('00000'), f"Value {val} has excessive trailing zeros"
            # Should be a valid float
            float(val)

    def test_build_gst_twice_with_preamble_is_idempotent(self, tmp_path):
        """Call configure+build_gst twice; output must be identical and _norm must not mutate."""
        self._create_test_onnx_preamble(tmp_path, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

        mi = types.ModelInfo('modelname', types.TaskCategory.Classification, [3, 224, 224])
        mi.manifest = types.Manifest(
            'modellib',
            input_shapes=[(1, 3, 224, 224)],
            input_dtypes=['uint8'],
            output_shapes=[(1, 1000)],
            output_dtypes=['float32'],
            quantize_params=[(1.0, 0)],
            dequantize_params=[(1.0, 0)],
            model_lib_file='model.json',
            preprocess_graph='preprocess_graph.onnx',
        )

        op = operators.mega.ToTensorAndNormalise(mean='0, 0, 0', std='1/255, 1/255, 1/255')
        original_mean = list(op._norm.mean_values)
        original_std = list(op._norm.std_values)

        mock_task_graph = MagicMock()
        mock_task_graph.get_master.return_value = "mocked_master_value"

        op.configure_model_and_context_info(
            mi, operators.PipelineContext(), "task_name", 0, tmp_path, task_graph=mock_task_graph
        )
        gst_output_1 = _gen_gst(op)

        op.configure_model_and_context_info(
            mi, operators.PipelineContext(), "task_name", 0, tmp_path, task_graph=mock_task_graph
        )
        gst_output_2 = _gen_gst(op)

        assert gst_output_1 == gst_output_2
        assert list(op._norm.mean_values) == original_mean
        assert list(op._norm.std_values) == original_std

    def test_missing_super_fails_on_scale_access(self, tmp_path):
        """Mega operator skipping super().configure_model_and_context_info() fails at build_gst."""

        class _SkipSuper(operators.mega.ToTensorAndNormalise):
            mean: str = '0, 0, 0'
            std: str = '1/255, 1/255, 1/255'

            def configure_model_and_context_info(self, *args, **kwargs):
                pass  # intentionally skips super()

        mi = types.ModelInfo('modelname', types.TaskCategory.Classification, [3, 224, 224])
        mi.manifest = types.Manifest(
            'modellib',
            input_shapes=[(1, 3, 224, 224)],
            input_dtypes=['uint8'],
            output_shapes=[(1, 1000)],
            output_dtypes=['float32'],
            quantize_params=[(1.0, 0)],
            dequantize_params=[(1.0, 0)],
            model_lib_file='model.json',
        )

        op = _SkipSuper()
        mock_task_graph = MagicMock()
        mock_task_graph.get_master.return_value = "mocked_master_value"
        op.configure_model_and_context_info(
            mi, operators.PipelineContext(), "task_name", 0, tmp_path, task_graph=mock_task_graph
        )

        gst = gst_builder._OldBuilder(None, None, 16)
        with pytest.raises(AttributeError):
            op.build_gst(gst, '')

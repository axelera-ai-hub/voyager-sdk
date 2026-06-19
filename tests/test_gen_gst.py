# Copyright Axelera AI, 2023
import contextlib
import dataclasses
import io
import itertools
import os
from pathlib import Path
import platform
import re
import tempfile
from unittest.mock import Mock, mock_open, patch

import cv2
import pytest

onnxruntime = pytest.importorskip(
    'onnxruntime'
)  # TODO we should not need onnxruntime in the runtime tests
import yaml
from yaml_clean import yaml_clean

from axelera import types
from axelera.app import config, network, operators, pipe
from axelera.app.pipe import manager


def _ncore(manifest: types.Manifest, ncores: int) -> types.Manifest:
    inp = (ncores,) + manifest.input_shapes[0][1:]
    out = (ncores,) + manifest.output_shapes[0][1:]
    return dataclasses.replace(manifest, input_shapes=[inp], output_shapes=[out])


SQ_IN_YAML = 'ax_models/model_cards/torchvision/classification/squeezenet1.0-imagenet-onnx.yaml'
SQ_MANIFEST = types.Manifest(
    'sq',
    input_shapes=[(1, 224, 224, 3)],
    input_dtypes=['uint8'],
    output_shapes=[(1, 1, 1, 1000)],
    output_dtypes=['float32'],
    quantize_params=((0.01863, -14), (0.01863, -14), (0.01863, -14)),
    dequantize_params=((0.9, 0),),
    n_padded_ch_inputs=[(1, 2, 3, 4)],
    model_lib_file='model.json',
)
SQ_MANIFEST4 = _ncore(SQ_MANIFEST, 4)

RN34_IN_YAML = 'ax_models/model_cards/torchvision/classification/resnet34-imagenet-onnx.yaml'
RN50_IN_YAML = 'ax_models/model_cards/torchvision/classification/resnet50-imagenet-onnx.yaml'
RN_MANIFEST = types.Manifest(
    'sq',
    input_shapes=[(1, 224, 224, 3)],
    input_dtypes=['uint8'],
    output_shapes=[(1, 1, 1, 1000)],
    output_dtypes=['float32'],
    quantize_params=((0.01863, -14), (0.01863, -14), (0.01863, -14)),
    dequantize_params=((0.9, 0),),  # dequant?
    n_padded_ch_inputs=[(1, 2, 3, 4)],
    model_lib_file='lib_export/model.json',
)
RN_MANIFEST4 = _ncore(RN_MANIFEST, 4)

# Grayscale model configuration
GRAY_RES2NET_IN_YAML = 'ax_models/tutorials/grayscale/res2net50d-grayscale-beans.yaml'
GRAY_RES2NET_MANIFEST = types.Manifest(
    'res2net50d-grayscale-beans',
    input_shapes=[(1, 224, 224, 1)],
    input_dtypes=['uint8'],
    output_shapes=[(1, 1, 1, 3)],
    output_dtypes=['float32'],
    quantize_params=((0.017639562487602234, -12),),
    dequantize_params=((0.026342090219259262, -30),),
    n_padded_ch_inputs=[(0, 0, 1, 1, 1, 31, 0, 0)],
    model_lib_file='model.json',
)

YOLOV5S_V5_IN_YAML = 'ax_models/model_cards/yolo/object_detection/yolov5s-relu-coco-onnx.yaml'
YOLOV5S_V5_MANIFEST = types.Manifest(
    'yolo',
    input_shapes=[(1, 320, 320, 64)],
    input_dtypes=['uint8'],
    output_shapes=[[1, 20, 20, 256], [1, 40, 40, 256], [1, 80, 80, 256]],
    output_dtypes=['float32', 'float32', 'float32'],
    quantize_params=[(0.003919653594493866, -128)],
    dequantize_params=[
        [0.08142165094614029, 70],
        [0.09499982744455338, 82],
        [0.09290479868650436, 66],
    ],
    n_padded_ch_inputs=[(0, 0, 0, 0, 0, 0, 0, 52)],
    n_padded_ch_outputs=[
        (0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 1),
    ],
    model_lib_file='lib_export/model.json',
    postprocess_graph='lib_export/post_process.onnx',
    preprocess_graph='tests/assets/focus_preprocess_graph.onnx',
)

YOLOV5S_V7_IN_YAML = 'ax_models/model_cards/yolo/object_detection/yolov5s-v7-coco-onnx.yaml'
YOLOV5S_V7_MANIFEST = types.Manifest(
    'yolo',
    input_shapes=[(1, 640, 640, 64)],
    input_dtypes=['uint8'],
    output_shapes=[[1, 20, 20, 256], [1, 40, 40, 256], [1, 80, 80, 256]],
    output_dtypes=['float32', 'float32', 'float32'],
    quantize_params=[(0.003919653594493866, -128)],
    dequantize_params=[
        [0.08142165094614029, 70],
        [0.09499982744455338, 82],
        [0.09290479868650436, 66],
    ],
    n_padded_ch_inputs=[(0, 0, 0, 0, 0, 0, 0, 61)],
    n_padded_ch_outputs=[
        (0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 1),
    ],
    model_lib_file='lib_export/model.json',
    postprocess_graph='lib_export/post_process.onnx',
)

YOLOV5S_V7_IN_YAML = 'ax_models/model_cards/yolo/object_detection/yolov5m-v7-coco-onnx.yaml'
YOLO_TRACKER_RN_IN_YAML = 'ax_models/reference/cascade/with_tracker/yolov5m-tracker-resnet50.yaml'

RETINAFACE_IN_YAML = 'ax_models/model_cards/torch/retinaface-resnet50-widerface-onnx.yaml'
FACENET_IN_YAML = 'ax_models/model_cards/torch/facenet-lfw.yaml'
RETINAFACE_TRACKER_FACENET_IN_YAML = (
    'ax_models/reference/cascade/with_tracker/face-recognition-with-vote.yaml'
)
YOLOV5M_V7_MANIFEST = types.Manifest(
    'yolo',
    input_shapes=[(1, 644, 656, 4)],
    input_dtypes=['uint8'],
    output_shapes=[[1, 20, 20, 256], [1, 40, 40, 256], [1, 80, 80, 256]],
    output_dtypes=['float32', 'float32', 'float32'],
    quantize_params=[(0.003919653594493866, -128)],
    dequantize_params=[
        [0.0038571979384869337, -128],
        [0.0038748111110180616, -128],
        [0.0038069516886025667, -128],
    ],
    n_padded_ch_inputs=[(0, 0, 2, 2, 2, 14, 0, 1)],
    n_padded_ch_outputs=[
        (0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 1),
    ],
    model_lib_file='lib_export/lib.so',
    postprocess_graph='',
)

RETINAFCE_MANIFEST = types.Manifest(
    'rf',
    input_shapes=[(1, 846, 848, 4)],
    input_dtypes=['uint8'],
    output_shapes=[
        [1, 105, 105, 64],
        [1, 53, 53, 64],
        [1, 27, 27, 64],
        [1, 105, 105, 64],
        [1, 53, 53, 64],
        [1, 27, 27, 64],
        [1, 105, 105, 64],
        [1, 53, 53, 64],
        [1, 27, 27, 64],
    ],
    output_dtypes=["int8", "int8", "int8", "int8", "int8", "int8", "int8", "int8", "int8"],
    quantize_params=((0.01863, -14),),
    dequantize_params=(
        (0.9, 0),
        (0.9, 0),
        (0.9, 0),
        (0.9, 0),
        (0.9, 0),
        (0.9, 0),
        (0.9, 0),
        (0.9, 0),
    ),
    n_padded_ch_inputs=[[0, 0, 3, 3, 3, 5, 0, 1]],
    n_padded_ch_outputs=[
        [0, 0, 0, 0, 0, 0, 0, 56],
        [0, 0, 0, 0, 0, 0, 0, 56],
        [0, 0, 0, 0, 0, 0, 0, 56],
        [0, 0, 0, 0, 0, 0, 0, 60],
        [0, 0, 0, 0, 0, 0, 0, 60],
        [0, 0, 0, 0, 0, 0, 0, 60],
        [0, 0, 0, 0, 0, 0, 0, 44],
        [0, 0, 0, 0, 0, 0, 0, 44],
        [0, 0, 0, 0, 0, 0, 0, 44],
    ],
    model_lib_file='lib_export/lib.so',
    postprocess_graph='',
)

FACENET_MANIFEST = types.Manifest(
    'facenet',
    input_shapes=[[1, 160, 160, 4]],
    input_dtypes=["int8"],
    output_shapes=[[1, 1, 1, 512]],
    output_dtypes=["int8"],
    quantize_params=[[0.00787353515625, -1]],
    dequantize_params=[[0.01970456913113594, 3]],
    n_padded_ch_inputs=[[0, 0, 0, 0, 0, 0, 0, 1]],
    n_padded_ch_outputs=[[0, 0, 0, 0, 0, 0, 0, 0]],
    model_lib_file='lib_export/lib.so',
    postprocess_graph='',
)

YOLOV8POSE_YOLOV8N_IN_YAML = 'ax_models/reference/cascade/yolov8spose-yolov8n.yaml'
YOLOV8POSE_MANIFEST = types.Manifest(
    'yolov8spose-coco-onnx',
    input_shapes=[(1, 642, 656, 4)],
    input_dtypes=['uint8'],
    output_shapes=[
        [1, 80, 80, 64],
        [1, 40, 40, 64],
        [1, 20, 20, 64],
        [1, 80, 80, 64],
        [1, 40, 40, 64],
        [1, 20, 20, 64],
        [1, 80, 80, 64],
        [1, 40, 40, 64],
        [1, 20, 20, 64],
    ],
    output_dtypes=['float32'] * 9,
    quantize_params=[(0.003919653594493866, -128)],
    dequantize_params=[
        [0.055154770612716675, -65],
        [0.05989416316151619, -65],
        [0.06476129591464996, -56],
        [0.10392487794160843, 109],
        [0.17826798558235168, 109],
        [0.16040770709514618, 107],
        [0.04365239664912224, 12],
        [0.057816002517938614, 7],
        [0.066075898706913, 15],
    ],
    n_padded_ch_inputs=[(0, 0, 1, 1, 1, 15, 0, 1)],
    n_padded_ch_outputs=[
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 63),
        (0, 0, 0, 0, 0, 0, 0, 63),
        (0, 0, 0, 0, 0, 0, 0, 63),
        (0, 0, 0, 0, 0, 0, 0, 13),
        (0, 0, 0, 0, 0, 0, 0, 13),
        (0, 0, 0, 0, 0, 0, 0, 13),
    ],
    model_lib_file='lib_export/yolov8pose/model.json',
    postprocess_graph='lib_export/yolov8pose/post_process.onnx',
)
YOLOV8N_MANIFEST = types.Manifest(
    'yolov8n-coco-onnx',
    input_shapes=[(1, 642, 656, 4)],
    input_dtypes=['uint8'],
    output_shapes=[
        [1, 80, 80, 64],
        [1, 40, 40, 64],
        [1, 20, 20, 64],
        [1, 80, 80, 128],
        [1, 40, 40, 128],
        [1, 20, 20, 128],
    ],
    output_dtypes=['float32'] * 6,
    quantize_params=[(0.003919653594493866, -128)],
    dequantize_params=[
        [0.08838965743780136, -60],
        [0.07353860884904861, -57],
        [0.07168316841125488, -44],
        [0.10592737793922424, 127],
        [0.15443256497383118, 117],
        [0.18016019463539124, 104],
    ],
    n_padded_ch_inputs=[(0, 0, 1, 1, 1, 15, 0, 1)],
    n_padded_ch_outputs=[
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 0),
        (0, 0, 0, 0, 0, 0, 0, 48),
        (0, 0, 0, 0, 0, 0, 0, 48),
        (0, 0, 0, 0, 0, 0, 0, 48),
    ],
    model_lib_file='lib_export/yolov8n/model.json',
    postprocess_graph='lib_export/post_process.onnx',
)

SOMETEMPFILE = '/path/to/sometempfile.txt'


def mock_temp(*args, **kwargs):
    c = io.StringIO()
    c.name = SOMETEMPFILE
    return c


def _generate_pipeline(nn, input, hardware_caps, low_latency=False, jetson=False):
    '''Construct gst E2E pipeline'''

    manager.compile_pipelines(nn, input.sources, hardware_caps)

    nn.model_infos = network.ModelInfos()
    for task in nn.tasks:
        mi = task.model_info
        nn.model_infos.add_model(mi, manifest_path=Path('./mock_manifest_path'))

    dm = _mock_device_manager()
    assert ['metis-0:1:0'] == [d.name for d in dm.devices]
    task_graph = pipe.graph.DependencyGraph(nn.tasks)
    pipeline_config = config.PipelineConfig(pipe_type='gst', low_latency=low_latency)
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(tempfile, 'NamedTemporaryFile', mock_temp))
        # prevent ./gst_pipeline.yaml being written by tests
        stack.enter_context(patch.object(Path, 'write_text', return_value=None))
        stack.enter_context(patch.object(Path, 'exists', return_value=True))
        env = {'JETSON_MODEL': 'nanoultraplusplus'} if jetson else {}
        stack.enter_context(patch.dict(os.environ, env, clear=True))
        manager._propagate_model_and_context_info(nn, task_graph)
        p = pipe.create_pipe(
            dm,
            pipeline_config,
            nn,
            Path('./'),
            hardware_caps,
            task_graph,
            None,
            input,
        )

    return p.pipeline


class MockCapture:
    def __init__(self, path):
        pass

    def get(self, attr):
        attrs = {
            cv2.CAP_PROP_FRAME_WIDTH: 1600,
            cv2.CAP_PROP_FRAME_HEIGHT: 1200,
            cv2.CAP_PROP_FRAME_COUNT: 100,
            cv2.CAP_PROP_FPS: 30,
        }
        return float(attrs[attr])

    def isOpened(self):
        return True

    def release(self):
        pass


def _create_pipein(srcs, system_config, pipeline_config, tiling: bool = False):
    paths = [f'/path/to/src{i}.mp4' for i in range(srcs)] if isinstance(srcs, int) else srcs
    cfgs = [config.TilingConfig(640 * (n + 1)) if tiling else None for n in range(len(paths))]
    kwargss = [
        {'preprocessing': [config.ImagePreproc.from_tile_config(t)]} if t else {} for t in cfgs
    ]
    assert len(paths) >= 1
    alloc = pipe.SourceIdAllocator()
    with patch.object(Path, 'exists', return_value=True):
        with patch.object(Path, 'is_file', return_value=True):
            with patch.object(os, 'access', return_value=True):
                with patch.object(cv2, 'VideoCapture', new=MockCapture):
                    srcs = [
                        config.Source(p, **kw) for p, kw in itertools.zip_longest(paths, kwargss)
                    ]
                    return pipe.io.MultiplexPipeInput(srcs, system_config, pipeline_config, alloc)


def _video_path(out_name: str) -> Path:
    return Path(__file__).parent / 'golden-gst' / out_name


def _actual_path(out_name: str) -> Path:
    return Path(__file__).parent / 'out' / out_name


def _expected_path(out_name: str) -> Path:
    return Path(__file__).parent / 'exp' / out_name


def _prepare_expected(out_name, manifests, tasks, hardware_caps, gold_path=_video_path):
    exp = gold_path(out_name).read_text()
    extra_args = _expansion_params(manifests, tasks, hardware_caps)

    def do_subs(m):
        try:
            return str(extra_args[m.group(1)])
        except KeyError:
            return m.group(2) or m.group(0)

    exp = re.sub(r"{\{([^:\}]+)\s*(?::\s*([^}\n]*))?\}\}", do_subs, exp, flags=re.MULTILINE)
    return yaml_clean(exp)


def _write_file(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or content != path.read_text():
        path.write_text(content)


def _compare_yaml(expected, actual, out_name, gold_path=_video_path):
    exp_path, actual_path = _expected_path(out_name), _actual_path(out_name)
    if expected != actual:
        _write_file(exp_path, expected)
        _write_file(actual_path, actual)
        a = actual_path.relative_to(Path.cwd())
        e = exp_path.relative_to(Path.cwd())
        g = gold_path(out_name).relative_to(Path.cwd())
        assert (
            expected == actual
        ), f'{a} {e} : generated yaml does not match gold, gold was generated from {g} '


def _mock_device_manager():
    m = Mock()
    m.name = 'metis-0:1:0'
    return Mock(devices=[m])


def _create_output_info_from_manifest(manifest, task_info):
    """Create realistic mock OutputInfo objects based on model type and manifest data."""
    if not manifest or not manifest.output_shapes:
        return []

    output_infos = []

    # For classifiers, the original output is typically (batch, num_classes)
    if len(manifest.output_shapes) == 1 and manifest.output_shapes[0][-1] == 1000:
        # This looks like a classifier with 1000 classes (ImageNet)
        output_infos.append(
            types.OutputInfo(
                shape=(1, 1000), name="0", dtype="float32"  # Original classifier output shape
            )
        )
    else:
        # For YOLO and other models, convert from NHWC (manifest) to NCHW (original)
        for i, output_shape in enumerate(manifest.output_shapes):
            if len(output_shape) == 4:
                # Convert NHWC (manifest) to NCHW (original model output)
                # manifest: [N, H, W, C] -> original: [N, C, H, W]
                n, h, w, c = output_shape
                # Remove padding from C dimension if present
                if manifest.n_padded_ch_outputs and i < len(manifest.n_padded_ch_outputs):
                    padding = manifest.n_padded_ch_outputs[i]
                    if len(padding) >= 8:
                        c_padding = padding[7]  # Channel padding is at index 7
                        c = c - c_padding
                original_shape = (n, c, h, w)
                output_infos.append(
                    types.OutputInfo(shape=original_shape, name=str(i), dtype="float32")
                )
            else:
                raise ValueError(
                    f"Unexpected output shape: expected 4D tensor, got {output_shape}. "
                    f"This case is not currently handled. Please verify the real manifest, "
                    f"or support this case."
                )

    return output_infos


def _load_highlevel(requested_cores: int, path: str, *manifests, low_latency=False):
    if len(manifests) == 1 and isinstance(manifests[0], (list, tuple)):
        manifests = manifests[0]
    nn = network.parse_network_from_path(path)
    pconfig = config.PipelineConfig(
        pipe_type='gst', aipu_cores=requested_cores, low_latency=low_latency
    )
    network.restrict_cores(nn, pconfig, config.Metis.pcie)
    device_man = _mock_device_manager()
    for manifest, task in itertools.zip_longest(manifests, nn.tasks):
        task.model_info.manifest = manifest

        # Mock the output_info that would normally be registered by _register_model_output_info
        if manifest:
            task.model_info.output_info = _create_output_info_from_manifest(
                manifest, task.model_info
            )

        if 'YOLO' in task.model_info.extra_kwargs:
            task.model_info.extra_kwargs['YOLO']['anchors'] = [
                [1.25, 1.625, 2.0, 3.75, 4.125, 2.875],
                [1.875, 3.8125, 3.875, 2.8125, 3.6875, 7.4375],
                [3.625, 2.8125, 4.875, 6.1875, 11.65625, 10.1875],
            ]

        config_content = {
            "remove_quantization_of_inputs_outputs_from_graph": True,
            "apply_pword_padding": True,
            "remove_padding_and_layout_transform_of_inputs_outputs": True,
            "host_arch": "x86_64",
            "target": "axelera",
            "aipu_cores": 1,
            "io_location": "L2",
            "wgt_location": "L2",
            "quantize_only": False,
        }

        if task.is_dl_task:
            with patch('builtins.open', mock_open()):
                with patch('json.load', return_value=config_content):
                    task.inference = operators.Inference(
                        device_man=device_man,
                        compiled_model_dir=Path('build/manifest.json').parent,
                        model_name=task.model_info.name,
                        model=manifest,
                        model_info=task.model_info,
                        inference_op_config=task.inference_op_config,
                        low_latency=low_latency,
                    )
                    # Only patch for the focus preprocess_graph test asset
                    if (
                        getattr(manifest, 'preprocess_graph', None)
                        == 'tests/assets/focus_preprocess_graph.onnx'
                    ):
                        task.inference.compiled_model_dir = Path('build')
                        manifest.model_lib_file = 'lib_export/model.json'
                    if 'YOLO' not in task.model_info.extra_kwargs:
                        task.inference._icdf_params = object()  # not None!
    return nn


def _pads(manifest):
    pads = list(manifest.n_padded_ch_inputs[0]) if manifest.n_padded_ch_inputs else [0] * 4
    if len(pads) == 4:
        t, l, b, r = pads
        pads = [0, 0, t, b, l, r, 0, 0]
    pads[7] = pads[7] - 1 if pads[7] in (1, 61) else pads[7]
    return ','.join(str(x) for x in pads)


def _expansion_params(manifests, tasks, hardware_caps):
    base = {f'input_video{n}': f'/path/to/src{n}.mp4' for n in range(8)}

    def add(name, values, replace_dot=False):
        _replace = lambda x: x.replace('.', '_') if replace_dot else x
        base.update({f'{name}{n}': _replace(v) for n, v in enumerate(values)})
        base[name] = values[0]

    add('model_lib', [f'build/{m.model_lib_file}' if m else '' for m in manifests])
    add('model_name', [t.model_info.name for t in tasks], replace_dot=True)
    add('task_name', [t.name for t in tasks], replace_dot=True)
    add('label_file', ['/path/to/sometempfile.txt' for _ in tasks])
    add('tracker_params_json', ['/path/to/sometempfile.txt' for _ in tasks])
    add('pads', [_pads(m) if m else '0,0,0,0,0,0,0,0' for m in manifests])
    add('quant_scale', [m.quantize_params[0][0] if m else 0 for m in manifests])
    add('quant_zeropoint', [m.quantize_params[0][1] if m else 0 for m in manifests])
    add('dequant_scale', [m.dequantize_params[0][0] if m else 0 for m in manifests])
    add('dequant_zeropoint', [m.dequantize_params[0][1] if m else 0 for m in manifests])
    post_proc = 'lib_cpu_post_processing.so'
    add(
        'post_model_lib',
        [Path(m.model_lib_file).parent / post_proc if m else '' for m in manifests],
    )
    add('input_w', [m.input_shapes[0][2] if m else 0 for m in manifests])
    add('input_h', [m.input_shapes[0][1] if m else 0 for m in manifests])
    add('embeddings_file', [config.env.framework / 'famous_embeddings.json' for _ in tasks])
    return dict(
        base,
        force_sw_decoders=not hardware_caps.vaapi,
        prefix='',
        confidence_threshold=0.3,
        nms_threshold=0.5,
        class_agnostic=1,
        max_boxes=30000,
        nms_top_k=200 if 'ssd' in tasks[0].model_info.name.lower() else 300,
        sigmoid_in_postprocess=0,
    )


NONE = config.HardwareCaps.NONE
OPENCL = config.HardwareCaps.OPENCL

gen_gst_marker = pytest.mark.parametrize(
    'caps, cores, src, manifest, golden_template, sources, proc, limit_fps',
    [
        (
            NONE,
            1,
            YOLOV8POSE_YOLOV8N_IN_YAML,
            [YOLOV8POSE_MANIFEST, YOLOV8N_MANIFEST],
            'yolov8pose-yolov8n.yaml',
            1,
            'x86_64',
            0,
        ),
        (
            OPENCL,
            4,
            SQ_IN_YAML,
            SQ_MANIFEST,
            'opencl/classifier-imagenet.yaml',
            1,
            'x86_64',
            0,
        ),
        (
            OPENCL,
            1,
            YOLO_TRACKER_RN_IN_YAML,
            [YOLOV5M_V7_MANIFEST, None, RN_MANIFEST],
            'opencl/yolov5m-tracker-resnet50.yaml',
            1,
            'x86_64',
            0,
        ),
        (
            OPENCL,
            1,
            GRAY_RES2NET_IN_YAML,
            GRAY_RES2NET_MANIFEST,
            'opencl/res2net50d-grayscale-beans.yaml',
            1,
            'x86_64',
            0,
        ),
        (
            NONE,
            4,
            SQ_IN_YAML,
            SQ_MANIFEST4,
            'classifier-imagenet-4core-3streams.yaml',
            3,
            'x86_64',
            0,
        ),
        (
            NONE,
            4,
            RN50_IN_YAML,
            RN_MANIFEST4,
            'classifier-imagenet.yaml',
            1,
            'x86_64',
            0,
        ),
        (
            NONE,
            4,
            RN50_IN_YAML,
            RN_MANIFEST4,
            'classifier-imagenet-limit-fps.yaml',
            1,
            'x86_64',
            15,
        ),
        (
            NONE,
            4,
            RN50_IN_YAML,
            RN_MANIFEST4,
            'classifier-imagenet-arm.yaml',
            1,
            'arm',
            0,
        ),
        (
            NONE,
            1,
            YOLOV5S_V5_IN_YAML,
            YOLOV5S_V5_MANIFEST,
            'yolov5s-axelera-coco-1stream.yaml',
            1,
            'x86_64',
            0,
        ),
        (
            NONE,
            1,
            YOLOV5S_V5_IN_YAML,
            YOLOV5S_V5_MANIFEST,
            'yolov5s-axelera-coco-1stream-arm.yaml',
            1,
            'arm',
            0,
        ),
        (
            NONE,
            1,
            YOLOV5S_V5_IN_YAML,
            YOLOV5S_V5_MANIFEST,
            'yolov5s-axelera-coco-4streams.yaml',
            4,
            'x86_64',
            0,
        ),
        (
            NONE,
            1,
            'ax_models/reference/image_preprocess/yolov5s-v7-perspective-onnx.yaml',
            YOLOV5S_V5_MANIFEST,
            'yolov5s-v7-perspective-4streams.yaml',
            4,
            'x86_64',
            0,
        ),
        (
            OPENCL,
            1,
            'ax_models/reference/image_preprocess/yolov5s-v7-perspective-onnx.yaml',
            YOLOV5S_V5_MANIFEST,
            'opencl/yolov5s-v7-perspective-4streams.yaml',
            4,
            'x86_64',
            0,
        ),
        (
            OPENCL,
            1,
            'ax_models/reference/image_preprocess/yolov5s-v7-perspective-barrel-onnx.yaml',
            YOLOV5S_V5_MANIFEST,
            'opencl/yolov5s-v7-perspective-barrel-4streams.yaml',
            4,
            'x86_64',
            0,
        ),
        (
            # same output as above, but with image preproc in sources, not in yaml
            OPENCL,
            1,
            YOLOV5S_V5_IN_YAML,
            YOLOV5S_V5_MANIFEST,
            'opencl/yolov5s-v7-perspective-barrel-4streams.yaml',
            [
                'perspective[[0.6715333509316848,0.34390796884598407,-67.26359830365045,-0.4529519589678815,0.5027865426887486,493.6304064972456,0.0,0.0,1.0]]:/path/to/src0.mp4',
                'camera_undistort[0.614,1.091,0.488,0.482,[-0.37793616, 0.11966818, -0.00067655, 0, -0.00115868]]:/path/to/src1.mp4',
                '/path/to/src2.mp4',
                '/path/to/src3.mp4',
            ],
            'x86_64',
            0,
        ),
        (
            NONE,
            1,
            YOLOV5S_V5_IN_YAML,
            YOLOV5S_V5_MANIFEST,
            'yolov5s-v7-rotate90-2streams.yaml',
            [
                'rotate90:/path/to/src0.mp4',
                '/path/to/src1.mp4',
            ],
            'x86_64',
            0,
        ),
        (
            OPENCL,
            1,
            YOLOV5S_V5_IN_YAML,
            YOLOV5S_V5_MANIFEST,
            'opencl/yolov5s-v7-rotate90-2streams.yaml',
            [
                'rotate90:/path/to/src0.mp4',
                '/path/to/src1.mp4',
            ],
            'x86_64',
            0,
        ),
        (
            OPENCL,
            1,
            'ax_models/reference/image_preprocess/yolov5s-v7-polar-onnx.yaml',
            YOLOV5S_V5_MANIFEST,
            'opencl/yolov5s-v7-polar.yaml',
            1,
            'x86_64',
            0,
        ),
        (
            OPENCL,
            1,
            YOLOV5S_V5_IN_YAML,
            YOLOV5S_V5_MANIFEST,
            'opencl/yolov5s-v7-polar.yaml',
            [
                'polar[2510,800,0,true,0.5,0.495,800,false,true]:/path/to/src0.mp4',
            ],
            'x86_64',
            0,
        ),
        (
            OPENCL,
            1,
            'ax_models/reference/image_preprocess/yolov5s-v7-crop-onnx.yaml',
            YOLOV5S_V5_MANIFEST,
            'opencl/yolov5s-v7-crop-4streams.yaml',
            4,
            'x86_64',
            0,
        ),
        (
            # same output as above, but with image preproc in sources, not in yaml
            OPENCL,
            1,
            YOLOV5S_V5_IN_YAML,
            YOLOV5S_V5_MANIFEST,
            'opencl/yolov5s-v7-crop-4streams.yaml',
            [
                'crop[0,0,640,540]:/path/to/src0.mp4',
                'crop[100,240,600,640]:/path/to/src1.mp4',
                'crop[0,240,400,440]:/path/to/src2.mp4',
                'crop[240,0,440,400]:/path/to/src3.mp4',
            ],
            'x86_64',
            0,
        ),
        (
            OPENCL,
            1,
            RETINAFACE_IN_YAML,
            RETINAFCE_MANIFEST,
            'opencl/retinaface-resnet.yaml',
            1,
            'x86_64',
            0,
        ),
        (
            OPENCL,
            1,
            RETINAFACE_TRACKER_FACENET_IN_YAML,
            [RETINAFCE_MANIFEST, None, FACENET_MANIFEST],
            'opencl/face-recog.yaml',
            1,
            'x86_64',
            0,
        ),
        (
            NONE,
            1,
            RETINAFACE_TRACKER_FACENET_IN_YAML,
            [RETINAFCE_MANIFEST, None, FACENET_MANIFEST],
            'face-recog.yaml',
            1,
            'x86_64',
            0,
        ),
    ],
)


@gen_gst_marker
def test_lowlevel_output_new_inference(
    caps, cores, src, manifest, golden_template, sources, proc, limit_fps
):
    nn = _load_highlevel(cores, src, *([manifest] if not isinstance(manifest, list) else manifest))
    pipein = _create_pipein(
        sources,
        config.SystemConfig(hardware_caps=caps, allow_hardware_codec=False),
        config.PipelineConfig(specified_frame_rate=limit_fps, pipe_type='gst'),
    )
    with patch.object(platform, 'processor', return_value=proc):
        pipeline = _generate_pipeline(nn, pipein, hardware_caps=caps)
    actual = yaml.dump([{'pipeline': pipeline}], sort_keys=False)
    manifests = manifest if isinstance(manifest, list) else [manifest]
    exp = _prepare_expected(golden_template, manifests, nn.tasks, hardware_caps=caps)
    _compare_yaml(exp, actual, golden_template)


@pytest.mark.parametrize(
    'caps, cores, src, manifest, golden_template, sources, proc, limit_fps',
    [
        (
            NONE,
            1,
            YOLOV5S_V5_IN_YAML,
            YOLOV5S_V5_MANIFEST,
            'yolov5s-axelera-coco-tiled.yaml',
            1,
            'x86_64',
            0,
        ),
        (
            NONE,
            1,
            YOLOV5S_V5_IN_YAML,
            YOLOV5S_V5_MANIFEST,
            'yolov5s-axelera-coco-tiled-2sources.yaml',
            2,
            'x86_64',
            0,
        ),
    ],
)
def test_tiling(caps, cores, src, manifest, golden_template, sources, proc, limit_fps):
    nn = _load_highlevel(cores, src, *([manifest] if not isinstance(manifest, list) else manifest))
    pipein = _create_pipein(
        sources,
        config.SystemConfig(hardware_caps=caps, allow_hardware_codec=False),
        config.PipelineConfig(specified_frame_rate=limit_fps, pipe_type='gst'),
        tiling=True,
    )
    with patch.object(platform, 'processor', return_value=proc):
        pipeline = _generate_pipeline(nn, pipein, hardware_caps=caps)
    actual = yaml.dump([{'pipeline': pipeline}], sort_keys=False)
    manifests = manifest if isinstance(manifest, list) else [manifest]
    exp = _prepare_expected(golden_template, manifests, nn.tasks, hardware_caps=caps)
    _compare_yaml(exp, actual, golden_template)


@pytest.mark.parametrize(
    'caps, cores, src, manifest, golden_template, sources, proc, limit_fps',
    [
        (
            NONE,
            1,
            YOLOV5S_V5_IN_YAML,
            YOLOV5S_V5_MANIFEST,
            'yolov5s-axelera-coco-low-latency.yaml',
            1,
            'x86_64',
            0,
        ),
    ],
)
def test_low_latency(caps, cores, src, manifest, golden_template, sources, proc, limit_fps):
    manifests = [[manifest] if not isinstance(manifest, list) else manifest]
    nn = _load_highlevel(cores, src, *manifests, low_latency=True)
    pipein = _create_pipein(
        sources,
        config.SystemConfig(hardware_caps=caps, allow_hardware_codec=False),
        config.PipelineConfig(specified_frame_rate=limit_fps, pipe_type='gst', low_latency=True),
    )
    with patch.object(platform, 'processor', return_value=proc):
        pipeline = _generate_pipeline(nn, pipein, hardware_caps=caps, low_latency=True)
    actual = yaml.dump([{'pipeline': pipeline}], sort_keys=False)
    manifests = manifest if isinstance(manifest, list) else [manifest]
    exp = _prepare_expected(golden_template, manifests, nn.tasks, hardware_caps=caps)
    _compare_yaml(exp, actual, golden_template)


@pytest.mark.parametrize(
    'caps, cores, src, manifest, golden_template, sources, proc, limit_fps',
    [
        (
            NONE,
            1,
            YOLOV5S_V5_IN_YAML,
            YOLOV5S_V5_MANIFEST,
            'yolov5s-axelera-coco-jetson.yaml',
            1,
            'x86_64',
            0,
        ),
    ],
)
def test_jetson(caps, cores, src, manifest, golden_template, sources, proc, limit_fps):
    manifests = [[manifest] if not isinstance(manifest, list) else manifest]
    nn = _load_highlevel(cores, src, *manifests, low_latency=True)
    pipein = _create_pipein(
        sources,
        config.SystemConfig(hardware_caps=caps, allow_hardware_codec=False),
        config.PipelineConfig(specified_frame_rate=limit_fps, pipe_type='gst', low_latency=True),
    )
    with patch.object(platform, 'processor', return_value=proc):
        pipeline = _generate_pipeline(
            nn, pipein, hardware_caps=caps, low_latency=True, jetson=True
        )
    actual = yaml.dump([{'pipeline': pipeline}], sort_keys=False)
    manifests = manifest if isinstance(manifest, list) else [manifest]
    exp = _prepare_expected(golden_template, manifests, nn.tasks, hardware_caps=caps)
    _compare_yaml(exp, actual, golden_template)


@gen_gst_marker
def test_gst_pipeline_builder(
    caps, cores, src, manifest, golden_template, sources, proc, limit_fps
):
    # For the sake of better code coverage and to ensure that what we created was maybe
    # approximately sensible lowlevel yaml, also build the pipeline, hweever this does
    # not work in tox env, so allow it to skip if it fails to create a gst element
    del golden_template
    nn = _load_highlevel(cores, src, *([manifest] if not isinstance(manifest, list) else manifest))
    pipein = _create_pipein(
        sources,
        config.SystemConfig(hardware_caps=caps, allow_hardware_codec=False),
        config.PipelineConfig(specified_frame_rate=limit_fps, pipe_type='gst'),
    )
    with patch.object(platform, 'processor', return_value=proc):
        pipeline = _generate_pipeline(nn, pipein, hardware_caps=caps)
    # I had to disable this again because axinferencenet now proactively tries to
    # load the model and this fails because the model does not exist in the test env.
    # try:
    #     gst_helper.build_pipeline(pipeline)
    # except Exception as e:
    #     if 'Failed to create element of type' in str(e):
    #         pytest.skip('axstreamer plugins not installed')
    #     raise

# Copyright Axelera AI, 2023
import contextlib
import importlib
import logging
import os
import pathlib
import re
import sys
import tempfile
from unittest.mock import ANY, Mock, call, patch, MagicMock

import PIL
import pytest
from strictyaml.exceptions import YAMLValidationError

from axelera import types
from axelera.app import (
    config,
    constants,
    logging_utils,
    network,
    operators,
    pipe,
    utils,
    yaml_parser,
)
from axelera.app.network import AxNetwork
from axelera.app.operators import InferenceOpConfig, Input, PipelineContext, Resize
from axelera.app.pipeline import AxTask

IMAGENET_TEMPLATE = '''
preprocess:
    - resize:
        width: 1024
        height: 768
'''

SQUEEZENET_NAME = 'squeezenet1.0-imagenet-onnx'

SQUEEZENET_PIPELINE = '''
pipeline:
  - squeezenet1.0-imagenet-onnx:
      template_path: doesnotexist/imagenet.yaml
      input:
        type: image
      preprocess:
      postprocess:
'''

SQUEEZENET_MINIMAL_MODEL = '''\
  squeezenet1.0-imagenet-onnx:
    task_category: Classification
    class: AxTorchvisionSqueezeNet
    class_path: doesnotexist/squeezenet.py
    input_tensor_layout: NCHW
    input_tensor_shape: [1, 3, 224, 224]
    input_color_format: RGB  # RGB, BGR, Gray
'''

SQUEEZENET_NETWORK = f'''
name: squeezenet1.0-imagenet-onnx
{SQUEEZENET_PIPELINE}
models:
{SQUEEZENET_MINIMAL_MODEL}
'''

SQUEEZENET_NETWORK_WITH_IMAGENET_DATASET = f'''
{SQUEEZENET_NETWORK}
    dataset: ImageNet-1K
datasets:
  ImageNet-1K:
    class: AxImagenetDataAdapter
    class_path: $AXELERA_FRAMEWORK/ax_datasets/imagenet.py
    data_dir_name: ImageNet
    val: val
    test: test
    labels_path: imagenet1000_clsidx_to_labels.txt
'''


def parse_net(main_yaml, files):
    files = dict(files, **{'test.yaml': main_yaml})

    def isfile(path: pathlib.Path):
        return path.name in files

    def readtext(path, *args):
        try:
            return files[path.name]
        except KeyError:
            raise FileNotFoundError(path) from None

    with contextlib.ExitStack() as stack:
        stack.enter_context(patch('pathlib.Path.is_file', new=isfile))
        stack.enter_context(patch('pathlib.Path.read_text', new=readtext))
        return network.parse_network_from_path('test.yaml')


def test_parse_network():
    input = f'''
name: squeezenet1.0-imagenet-onnx
description: SqueezeNet 1.0 (ImageNet)
{SQUEEZENET_PIPELINE}
models:
  squeezenet1.0-imagenet-onnx:
    class: AxTorchvisionSqueezeNet
    class_path: doesnotexist/squeezenet.py
    version: "1_0"
    task_category: Classification
    input_tensor_layout: NCHW
    input_tensor_shape: [1, 3, 224, 224]
    input_color_format: RGB  # RGB, BGR, Gray
    dataset: ImageNet-1K
    num_classes: 1000
    extra_kwargs:
      torchvision_args:
        torchvision_weights_args:
          object: SqueezeNet1_0_Weights
          name: IMAGENET1K_V1
'''
    net = parse_net(input, {'imagenet.yaml': IMAGENET_TEMPLATE})
    assert net.path == 'test.yaml'
    assert len(net.hash_str) >= 8
    assert all(c in '0123456789abcdef' for c in net.hash_str)
    assert net.name == 'squeezenet1.0-imagenet-onnx'
    assert net.description == 'SqueezeNet 1.0 (ImageNet)'
    MODEL_INFO = types.ModelInfo(
        name=SQUEEZENET_NAME,
        task_category='Classification',
        input_tensor_shape=[3, 224, 224],
        input_color_format='RGB',
        dataset='ImageNet-1K',
        class_name='AxTorchvisionSqueezeNet',
        class_path=f'{os.getcwd()}/doesnotexist/squeezenet.py',
        version='1_0',
        num_classes=1000,
        extra_kwargs={
            'torchvision_args': {
                'torchvision_weights_args': {
                    'object': 'SqueezeNet1_0_Weights',
                    'name': 'IMAGENET1K_V1',
                },
            },
        },
    )

    mis = network.ModelInfos()
    mis.add_model(MODEL_INFO, None)
    exp = AxNetwork(
        path='test.yaml',
        name='squeezenet1.0-imagenet-onnx',
        description='SqueezeNet 1.0 (ImageNet)',
        tasks=[
            AxTask(
                SQUEEZENET_NAME,
                input=Input(),
                preprocess=[Resize(width=1024, height=768)],
                inference_op_config=InferenceOpConfig(),
                model_info=MODEL_INFO,
                context=PipelineContext(),
            )
        ],
        custom_operators={},
        model_infos=mis,
    )
    net.tasks[0].model_info.labels = []
    net.tasks[0].model_info.label_filter = []
    pipe.manager._update_pending_expansions(net.tasks[0])
    assert exp == net


def test_parse_network_with_extra_kwargs_inference_config_and_cpp_decoder_does_all(tmpdir):
    # Create a temporary ok_ops.py with the required Op class
    ok_ops_path = tmpdir.join("ok_ops.py")
    ok_ops_path.write_text(
        '''
from axelera.app import operators

class Op(operators.AxOperator):
    def _post_init(self):
        # This property is deprecated, but we keep it for backward compatibility testing
        self.cpp_decoder_does_all = True
    def exec_torch(self, img, result, meta):
        return img, result, meta
    def build_gst(self, gst, stream_idx):
        pass
''',
        encoding="utf-8",
    )

    input = f'''
name: yolov5s-v5
description: YOLOv5s-v5 (COCO)
pipeline:
  - detections:
      model_name: yolov5s-v5
      input:
        type: image
      preprocess:
      postprocess:
        - op:
models:
  yolov5s-v5:
    class: AxYolo
    class_path: doesnotexist/yolo.py
    task_category: ObjectDetection
    input_tensor_layout: NCHW
    input_tensor_shape: [1, 3, 640, 640]
    input_color_format: RGB
    dataset: COCO
    num_classes: 80

operators:
  op:
    class: Op
    class_path: {ok_ops_path}
'''

    net = parse_net(input, {})
    # Test that the network was parsed successfully
    assert net.path == 'test.yaml'
    assert net.name == 'yolov5s-v5'
    assert len(net.tasks) == 1
    assert len(net.tasks[0].postprocess) == 1

    # Test that the inference op config has default values
    inf_config = net.tasks[0].inference_op_config
    assert inf_config.handle_dequantization_and_depadding is True
    assert inf_config.handle_transpose is True
    assert inf_config.handle_postamble is True
    assert inf_config.handle_preamble is True


def test_parse_network_with_extra_kwargs_in_inference_config():
    input = f'''
name: yolov5s-v5
description: YOLOv5s-v5 (COCO)
pipeline:
  - detections:
      model_name: yolov5s-v5
      input:
        type: image
      preprocess:
      postprocess:

models:
  yolov5s-v5:
    class: AxYolo
    class_path: doesnotexist/yolo.py
    task_category: ObjectDetection
    input_tensor_layout: NCHW
    input_tensor_shape: [1, 3, 640, 640]
    input_color_format: RGB
    dataset: COCO
    num_classes: 80
    extra_kwargs:
      YOLO:
        focus_layer_replacement: False

'''
    net = parse_net(input, {})
    assert net.path == 'test.yaml'
    MODEL_INFO = types.ModelInfo(
        name='yolov5s-v5',
        task_category='ObjectDetection',
        input_tensor_shape=[3, 640, 640],
        input_color_format='RGB',
        dataset='COCO',
        class_name='AxYolo',
        class_path=f'{os.getcwd()}/doesnotexist/yolo.py',
        num_classes=80,
        extra_kwargs={
            'YOLO': {'focus_layer_replacement': False},
        },
    )

    mis = network.ModelInfos()
    mis.add_model(MODEL_INFO, None)
    exp = AxNetwork(
        path='test.yaml',
        name='yolov5s-v5',
        description='YOLOv5s-v5 (COCO)',
        tasks=[
            AxTask(
                'detections',
                input=Input(),
                preprocess=[],
                inference_op_config=InferenceOpConfig(),
                model_info=MODEL_INFO,
                context=PipelineContext(),
            )
        ],
        custom_operators={},
        model_infos=mis,
    )

    net.tasks[0].model_info.labels = []
    net.tasks[0].model_info.label_filter = []
    pipe.manager._update_pending_expansions(net.tasks[0])
    assert exp == net

    # Test that the inference op config has default values
    assert exp.tasks[0].inference_op_config.handle_dequantization_and_depadding is True
    assert exp.tasks[0].inference_op_config.handle_transpose is True
    assert exp.tasks[0].inference_op_config.handle_postamble is True
    assert exp.tasks[0].inference_op_config.handle_preamble is True


def test_parse_network_pipeline_assets():
    input = f'''
{SQUEEZENET_NETWORK}
pipeline_assets:
   http://someurl:
     md5: a1234567890
     path: /tmp/somefile
'''
    net = parse_net(input, {'imagenet.yaml': IMAGENET_TEMPLATE})
    assert net.assets == [network.Asset('http://someurl', 'a1234567890', '/tmp/somefile')]


def test_parse_network_find_model():
    net = parse_net(SQUEEZENET_NETWORK, {'imagenet.yaml': IMAGENET_TEMPLATE})
    assert net.find_model(SQUEEZENET_NAME) == net.tasks[0].model_info
    with pytest.raises(ValueError, match='Model Oops not found in models'):
        assert net.find_model('Oops')


def test_parse_network_model_names():
    net = parse_net(SQUEEZENET_NETWORK, {'imagenet.yaml': IMAGENET_TEMPLATE})
    assert net.model_names == [SQUEEZENET_NAME]


def test_parse_network_find_task():
    net = parse_net(SQUEEZENET_NETWORK, {'imagenet.yaml': IMAGENET_TEMPLATE})
    assert net.find_task(SQUEEZENET_NAME).name == net.tasks[0].name
    with pytest.raises(ValueError, match=r'Cannot find Oops in pipeline.*squeezenet'):
        assert net.find_task('Oops')


@pytest.mark.parametrize(
    'py, cls, err',
    [
        (None, FileNotFoundError, r'Failed to import.*squeezenet\.py'),
        ('x::y', SyntaxError, r'invalid syntax \(squeezenet.py, line 1\)'),
    ],
)
def test_parse_network_model_class_failures(tmpdir, py, cls, err):
    pyfile = pathlib.Path(f"{tmpdir}/squeezenet.py")
    if py is not None:
        pyfile.write_text(py)
    net = SQUEEZENET_NETWORK.replace('doesnotexist/squeezenet.py', str(pyfile))
    net = parse_net(net, {'imagenet.yaml': IMAGENET_TEMPLATE})
    with pytest.raises(cls, match=err):
        net.model_class(SQUEEZENET_NAME)


@pytest.mark.skip(reason='This fails with unmarshallable object')
def test_parse_network_model_class_success(tmpdir):
    pyfile = pathlib.Path(f"{tmpdir}/squeezenet3.py")
    pyfile.write_text(
        '''\
from axelera.app.types import Model

class AxTorchvisionSqueezeNet(Model):
    pass
'''
    )
    net = SQUEEZENET_NETWORK.replace('doesnotexist/squeezenet.py', str(pyfile))
    net = parse_net(net, {'imagenet.yaml': IMAGENET_TEMPLATE})
    with net.from_model_dir(SQUEEZENET_NAME):
        assert net.model_class(SQUEEZENET_NAME).__name__ == 'AxTorchvisionSqueezeNet'


@pytest.mark.parametrize(
    'input,error',
    [
        (
            f'''
pipeline_assets:
   - somelistitem
        ''',
            r"when expecting a mapping(?s:.)*?found a sequence(?s:.)*?somelistitem",
        ),
        (
            f'''
pipeline_assets:
   133
        ''',
            r"when expecting a mapping(?s:.)*?found an arbitrary integer(?s:.)*?pipeline_assets",
        ),
        (
            f'''
pipeline_assets:
   url:
      - md5
      - path
        ''',
            r"when expecting a mapping(?s:.)*?url(?s:.)*?found a sequence",
        ),
    ],
)
def test_parse_network_pipeline_assets_invalid(input, error):
    input = (
        f'''
name: squeezenet1.0-imagenet-onnx
{SQUEEZENET_PIPELINE}
models:
{SQUEEZENET_MINIMAL_MODEL}
'''
        + input
    )
    with pytest.raises(YAMLValidationError, match=error):
        parse_net(input, {'imagenet.yaml': IMAGENET_TEMPLATE})


def test_parse_network_with_operators():
    input = f'''
name: squeezenet1.0-imagenet-onnx
{SQUEEZENET_PIPELINE}
models:
{SQUEEZENET_MINIMAL_MODEL}
operators:
  op:
    class: Op
    class_path: tests/ok_ops.py
'''
    net = parse_net(
        input,
        {
            'imagenet.yaml': IMAGENET_TEMPLATE,
        },
    )
    assert 'op' in net.custom_operators
    assert issubclass(net.custom_operators['op'], operators.AxOperator)


def test_parse_network_with_operator_overrides_permute(caplog):
    input = f'''
name: squeezenet1.0-imagenet-onnx
{SQUEEZENET_PIPELINE}
models:
{SQUEEZENET_MINIMAL_MODEL}
operators:
  permutechannels:
    class: PermuteChannels
    class_path: tests/ok_ops.py
'''
    net = parse_net(
        input,
        {
            'imagenet.yaml': IMAGENET_TEMPLATE,
        },
    )
    assert 'permutechannels' in net.custom_operators
    assert issubclass(net.custom_operators['permutechannels'], operators.AxOperator)
    assert caplog.records[0].levelname == 'WARNING'
    assert 'permutechannels already in builtin-operator list' in caplog.records[0].message


def test_parse_network_with_operator_overrides_custom_operator(caplog):
    input = f'''
name: squeezenet1.0-imagenet-onnx
{SQUEEZENET_PIPELINE}
models:
{SQUEEZENET_MINIMAL_MODEL}
operators:
  op:
    class: Op
    class_path: tests/ok_ops.py
'''

    template = f'''
{IMAGENET_TEMPLATE}
operators:
  op:
    class: Op
    class_path: ../tests/ok_ops.py
'''
    net = parse_net(input, {'imagenet.yaml': template})
    assert 'op' in net.custom_operators
    assert issubclass(net.custom_operators['op'], operators.AxOperator)
    assert caplog.records[0].levelname == 'WARNING'
    assert 'op already in operator list' in caplog.records[0].message


def test_parse_network_with_operator_non_existent_import():
    input = f'''
name: squeezenet1.0-imagenet-onnx
description: SqueezeNet 1.0 (ImageNet)
{SQUEEZENET_PIPELINE}
models:
{SQUEEZENET_MINIMAL_MODEL}
operators:
  op:
    class: Op
    class_path: tests/doesnotexist.py
'''
    with pytest.raises(FileNotFoundError, match=r'No such file.*doesnotexist\.py'):
        parse_net(input, {'imagenet.yaml': IMAGENET_TEMPLATE})


def test_parse_network_with_operator_bad_import():
    input = f'''
name: squeezenet1.0-imagenet-onnx
{SQUEEZENET_PIPELINE}
models:
{SQUEEZENET_MINIMAL_MODEL}
operators:
  op:
    class: Op
    class_path: tests/bad_ops.py
'''
    msg = 'This file should not be imported'
    with pytest.raises(RuntimeError, match=msg):
        parse_net(input, {'imagenet.yaml': IMAGENET_TEMPLATE})


@pytest.mark.parametrize(
    'input, error_or_warning',
    [
        (
            '''
name: squeezenet1.0-imagenet-onnx
pipeline:
  - squeezenet1.0-imagenet-onnx:
      template_path: doesnotexist/imagenet.yaml
''',
            ValueError(r'No models defined in network'),
        ),
        (
            '''
name: squeezenet1.0-imagenet-onnx
pipeline:
  - squeezenet1.0-imagenet-onnx:
      template_path: doesnotexist/imagenet.yaml]
models: {}
''',
            ValueError(r'No models defined in network'),
        ),
        (
            '''
name: squeezenet1.0-imagenet-onnx
pipeline:
  - squeezenet1.0-imagenet-onnx:
      template_path: doesnotexist/oops.yaml
models:
  squeezenet1.0-imagenet-onnx:
    task_category: Classification
    input_tensor_layout: NCHW
    input_tensor_shape: [1, 3, 224, 224]
    input_color_format: RGB  # RGB, BGR, Gray
''',
            ValueError(
                r'squeezenet1.0-imagenet-onnx: template .*doesnotexist/oops.yaml not found'
            ),
        ),
        (
            '''
name: squeezenet1.0-imagenet-onnx
pipeline:
  - squeezenet1.0-imagenet-onnx:
      template_path: doesnotexist/imagenet.yaml

models:
  SomeOther:
    task_category: Classification
    input_tensor_layout: NCHW
    input_tensor_shape: [1, 3, 224, 224]
    input_color_format: RGB  # RGB, BGR, Gray
''',
            ValueError(r'Model squeezenet1.0-imagenet-onnx not found in models'),
        ),
        (
            f'''
name: squeezenet1.0-imagenet-onnx
pipeline:
  - squeezenet1.0-imagenet-onnx:
      template_path: doesnotexist/imagenet.yaml

models:
{SQUEEZENET_MINIMAL_MODEL}
  SomeOther:
    task_category: Classification
    input_tensor_layout: NCHW
    input_tensor_shape: [1, 3, 224, 224]
    input_color_format: RGB  # RGB, BGR, Gray
''',
            r'Model SomeOther defined but not referenced in any task',
        ),
    ],
)
def test_parse_network_model_not_found(input, error_or_warning, caplog):
    caplog.set_level(logging.WARNING)

    if isinstance(error_or_warning, Exception):
        with pytest.raises(type(error_or_warning), match=str(error_or_warning)):
            parse_net(input, {'imagenet.yaml': IMAGENET_TEMPLATE})
    else:
        parse_net(input, {'imagenet.yaml': IMAGENET_TEMPLATE})
        assert any(error_or_warning in record.message for record in caplog.records)


@pytest.mark.parametrize(
    'input, error',
    [
        (
            f'''
name: squeezenet1.0-imagenet-onnx
pipeline:
  - squeezenet1.0-imagenet-onnx:
      template_path: doesnotexist/imagenet.yaml

models:
{SQUEEZENET_MINIMAL_MODEL}
    input_tensor_format: NCHW
''',
            r"unexpected key not in schema 'input_tensor_format'",
        ),
    ],
)
def test_parse_network_model_invalid_key(input, error):
    with pytest.raises(YAMLValidationError, match=error):
        parse_net(input, {'imagenet.yaml': IMAGENET_TEMPLATE})


def test_network_model_dataset():
    net = parse_net(SQUEEZENET_NETWORK_WITH_IMAGENET_DATASET, {'imagenet.yaml': IMAGENET_TEMPLATE})
    assert net.model_dataset_from_model(SQUEEZENET_NAME)['class'] == 'AxImagenetDataAdapter'


def test_network_model_dataset_no_datasets_but_dataset_not_given():
    net = parse_net(SQUEEZENET_NETWORK, {'imagenet.yaml': IMAGENET_TEMPLATE})
    assert net.model_dataset_from_model(SQUEEZENET_NAME) == {}


def test_network_model_dataset_with_dataset_given_but_no_datasets():
    input = (
        SQUEEZENET_NETWORK
        + '''\
    dataset: ImageNet-1K
'''
    )
    net = parse_net(input, {'imagenet.yaml': IMAGENET_TEMPLATE})
    with pytest.raises(ValueError, match=r'Missing definition of top-level datasets section'):
        net.model_dataset_from_model(SQUEEZENET_NAME)


def test_bad_datasets_type():
    with pytest.raises(ValueError, match=r"datasets is not a dictionary"):
        AxNetwork(datasets=['animals'])


def test_network_model_dataset_given_not_dataset_not_found():
    input = SQUEEZENET_NETWORK_WITH_IMAGENET_DATASET.replace('ImageNet-1K:', 'oops:')
    net = parse_net(input, {'imagenet.yaml': IMAGENET_TEMPLATE})
    with pytest.raises(ValueError, match=r"Missing definition of dataset ImageNet-1K"):
        net.model_dataset_from_model(SQUEEZENET_NAME)


def test_network_model_dataset_with_class_given_but_no_class_path():
    input = SQUEEZENET_NETWORK_WITH_IMAGENET_DATASET.replace(
        "    class_path: $AXELERA_FRAMEWORK/ax_datasets/imagenet.py\n", ""
    )
    with pytest.raises(
        ValueError,
        match=r"squeezenet1.0-imagenet-onnx: Missing class_path for dataset ImageNet-1K",
    ):
        net = parse_net(input, {'imagenet.yaml': IMAGENET_TEMPLATE})


def test_network_datasets_list():
    input = SQUEEZENET_NETWORK_WITH_IMAGENET_DATASET.replace("  ImageNet-1K:", " - ImageNet-1K:")
    with pytest.raises(
        YAMLValidationError,
        match=r"when expecting a mapping(?s:.)*?datasets(?s:.)*?found a sequence",
    ):
        net = parse_net(input, {'imagenet.yaml': IMAGENET_TEMPLATE})


def test_network_dataset_is_a_list():
    input = f'''
{SQUEEZENET_NETWORK}
    dataset: ImageNet-1K
datasets:
  ImageNet-1K:
    - class: AxImagenetDataAdapter
    - class_path: $AXELERA_FRAMEWORK/ax_datasets/imagenet.py
    - data_dir_name: ImageNet
    - val: val
    - test: test
    - labels_path: imagenet1000_clsidx_to_labels.txt
'''
    with pytest.raises(
        YAMLValidationError,
        match=r"when expecting a mapping(?s:.)*?ImageNet-1K(?s:.)*?found a sequence",
    ):
        net = parse_net(input, {'imagenet.yaml': IMAGENET_TEMPLATE})


def test_network_model_dataset_with_class_path_given_but_no_class():
    input = SQUEEZENET_NETWORK_WITH_IMAGENET_DATASET.replace(
        '    class: AxImagenetDataAdapter\n', ''
    )
    with pytest.raises(
        ValueError, match=r"squeezenet1.0-imagenet-onnx: Missing class for dataset ImageNet-1K"
    ):
        net = parse_net(input, {'imagenet.yaml': IMAGENET_TEMPLATE})


def test_network_model_dataset_with_target_split_given():
    input = f'''
{SQUEEZENET_NETWORK_WITH_IMAGENET_DATASET}
    target_split: val
'''
    with pytest.raises(
        ValueError,
        match=r"squeezenet1.0-imagenet-onnx: target_split is a reserved keyword for dataset ImageNet-1K, please use a different name for this attribute",
    ):
        parse_net(input, {'imagenet.yaml': IMAGENET_TEMPLATE})


def vision_builtin_networks():
    info = yaml_parser.network_yaml_info(llm_in_model_cards=False)

    def tutorial(nn):
        return 'ax_models/tutorials/' in nn.yaml_path

    def llm(nn):
        return info.has_llm(nn.yaml_path)

    vision_networks = [n.yaml_path for n in info.get_all_info() if not tutorial(n) and not llm(n)]
    if not importlib.util.find_spec("axelera.compiler"):  # Check if we are in runtime only env
        return [vision_networks[0], vision_networks[-1]]
    return vision_networks


@pytest.fixture
def af_dir_setter():
    old = os.getcwd()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    yield
    os.chdir(old)


@pytest.mark.parametrize('path', vision_builtin_networks())
def test_parse_all_vision_builtin_networks(path, af_dir_setter):
    """Test parsing all vision builtin networks.

    Gracefully skips networks that have postamble as there is no postamble ONNX file in CI environments.
    """
    from axelera.app import data_utils, format_converters

    # Mock dataset downloading to avoid S3 access issues in CI
    # Mock Ultralytics data YAML processing to skip when dataset files don't exist
    with patch.object(
        data_utils, 'download_custom_dataset', return_value=None
    ) as mock_download, patch.object(
        format_converters, 'process_ultralytics_data_yaml', return_value=None
    ) as mock_ultralytics:
        try:
            network.parse_network_from_path(path)
        except (ValueError, Exception) as e:
            if "Postamble ONNX model file" in str(e) and "does not exist" in str(e):
                pytest.skip(f"Skipping {path} due to missing postamble ONNX file: {e}")
            else:
                raise  # Re-raise other exceptions


class ClassWithInit(types.Model):
    def __init__(self, arg1, arg2):
        self.arg1 = arg1
        self.arg2 = arg2

    def init_model_deploy(self, model_info, dataset_config, **kwargs):
        pass


class ClassWithInitAndExtraArgs(types.Model):
    def __init__(self, arg1, arg3, **kwargs):
        self.arg1 = arg1
        self.arg3 = arg3
        self.extra_args = kwargs

    def init_model_deploy(self, model_info, dataset_config, **kwargs):
        pass


class ClassNoInit(types.Model):
    def init_model_deploy(self, model_info, dataset_config, **kwargs):
        pass


class ClassInitNoArgs(types.Model):
    def __init__(self) -> None:
        super().__init__()

    def init_model_deploy(self, model_info, dataset_config, **kwargs):
        pass


class ClassWithKwargsOnly(types.Model):
    def __init__(self, **kwargs):
        self.extra_args = kwargs

    def init_model_deploy(self, model_info, dataset_config, **kwargs):
        pass


class ClassWithArgsKwargs(types.Model):
    def __init__(self, *args, **kwargs):
        self.extra_args = kwargs

    def init_model_deploy(self, model_info, dataset_config, **kwargs):
        pass


class CustomDataAdapter(types.DataAdapter):
    def __init__(self, dataset_config: dict, model_info: types.ModelInfo):
        self.dataset_config = dataset_config

    def create_calibration_data_loader(self, transform, root, batch_size, **kwargs):
        return "DataAdapter's calibration loader"

    def create_validation_data_loader(self, root, target_split=None, **kwargs):
        return "DataAdapter's validation loader"

    def reformat_batched_data(self, is_calibration, batched_data):
        return batched_data  # Implement a basic version


class ModelWithMethods(types.Model):
    def init_model_deploy(self, model_info, dataset_config, **kwargs):
        pass

    def create_calibration_data_loader(self, transform, root, batch_size, **kwargs):
        return "Model's calibration loader"


@pytest.mark.parametrize(
    "model_class, arguments, expected_attributes, expected_data_adapter",
    [
        (
            ClassWithInit,
            {'arg1': 1, 'arg2': 2, 'arg3': 3, 'extra_arg': 'extra'},
            {'arg1': 1, 'arg2': 2},
            None,
        ),
        (
            ClassWithInitAndExtraArgs,
            {'arg1': 1, 'arg2': 2, 'arg3': 3, 'extra_arg': 'extra'},
            {'arg1': 1, 'arg3': 3, 'extra_args': {'arg2': 2, 'extra_arg': 'extra'}},
            None,
        ),
        (
            ClassWithKwargsOnly,
            {'arg1': 1, 'arg2': 2, 'arg3': 3, 'extra_arg': 'extra'},
            {'extra_args': {'arg1': 1, 'arg2': 2, 'arg3': 3, 'extra_arg': 'extra'}},
            None,
        ),
        (
            ClassWithArgsKwargs,
            {'arg1': 1, 'arg2': 2, 'arg3': 3, 'extra_arg': 'extra'},
            {'extra_args': {'arg1': 1, 'arg2': 2, 'arg3': 3, 'extra_arg': 'extra'}},
            None,
        ),
        (ClassNoInit, {'arg1': 1, 'arg2': 2, 'arg3': 3, 'extra_arg': 'extra'}, {}, None),
        (ClassInitNoArgs, {'arg1': 1, 'arg2': 2, 'arg3': 3, 'extra_arg': 'extra'}, {}, None),
        (
            ClassWithInit,
            {
                'arg1': 1,
                'arg2': 2,
                'dataset_config': {
                    'class': 'CustomDataAdapter',
                    'class_path': 'path/to/custom_data_adapter.py',
                    'some_config': 'value',
                },
            },
            {'arg1': 1, 'arg2': 2},
            {
                'dataset_config': {
                    'class': 'CustomDataAdapter',
                    'class_path': 'path/to/custom_data_adapter.py',
                    'some_config': 'value',
                }
            },
        ),
    ],
)
def test_initialize_model(model_class, arguments, expected_attributes, expected_data_adapter):
    with patch('axelera.app.network.utils.import_class_from_file') as mock_import:
        if expected_data_adapter:
            mock_import.return_value = CustomDataAdapter
        else:
            mock_import.side_effect = ImportError("No DataAdapter")

        instance = network.initialize_model(model_class, arguments)

        assert isinstance(instance, model_class)
        for attribute, expected_value in expected_attributes.items():
            assert getattr(instance, attribute) == expected_value

        if expected_data_adapter:
            assert isinstance(instance, CustomDataAdapter)
            assert instance.dataset_config == expected_data_adapter['dataset_config']
        else:
            assert not isinstance(instance, types.DataAdapter)


def test_initialize_model_with_nonexistent_data_adapter():
    with patch(
        'axelera.app.network.utils.import_class_from_file',
        side_effect=ImportError("No module named 'nonexistent_adapter'"),
    ):
        with pytest.raises(
            ValueError,
            match=r"Can't find NonexistentDataAdapter in path/to/nonexistent_adapter\.py\. Please check YAML datasets section\.",
        ):
            network.initialize_model(
                ClassWithInit,
                {
                    'arg1': 1,
                    'arg2': 2,
                    'dataset_config': {
                        'class': 'NonexistentDataAdapter',
                        'class_path': 'path/to/nonexistent_adapter.py',
                    },
                },
            )


@pytest.mark.parametrize(
    "model_class, expected_calibration, expected_validation",
    [
        (ClassNoInit, "DataAdapter's calibration loader", "DataAdapter's validation loader"),
        (ModelWithMethods, "Model's calibration loader", "DataAdapter's validation loader"),
    ],
)
def test_initialize_model_with_data_adapter(
    model_class, expected_calibration, expected_validation
):
    with patch('axelera.app.network.utils.import_class_from_file', return_value=CustomDataAdapter):
        instance = network.initialize_model(
            model_class,
            {
                'dataset_config': {
                    'class': 'CustomDataAdapter',
                    'class_path': 'path/to/data_adapter.py',
                }
            },
        )

        assert instance.create_calibration_data_loader(None, 'root', 4) == expected_calibration
        assert instance.create_validation_data_loader('root') == expected_validation
        assert instance.reformat_batched_data(False, "test") == "test"


def _mini_network(dataset=None, custom_postprocess=None):
    mi = types.ModelInfo(
        'n',
        'classification',
        [3, 20, 10],
        dataset='animals' if dataset else '',
        base_dir='somebasedir',
    )
    mi.manifest = types.Manifest(
        quantized_model_file=constants.K_MODEL_QUANTIZED_FILE_NAME,
        input_shapes=[(3, 20, 10)],
        input_dtypes=['int8'],
        output_shapes=[(1, 5)],
        output_dtypes=['float32'],
        quantize_params=[(0.1, 0.2)],
        dequantize_params=[(0.3, 0.4)],
        model_lib_file=constants.K_MODEL_FILE_NAME,
    )
    mis = network.ModelInfos()
    mis.add_model(mi, pathlib.Path('/path'))
    if custom_postprocess:
        tasks = [AxTask('n', input=Input(), model_info=mi, postprocess=[custom_postprocess])]
    else:
        tasks = [AxTask('n', input=Input(), model_info=mi)]
    return AxNetwork(
        tasks=tasks,
        model_infos=mis,
        datasets={'animals': dataset} if dataset else {},
    )


def test_instantiate_model_not_model_subclass():
    n = _mini_network()
    with patch.object(n, 'model_class', return_value=int):
        with pytest.raises(TypeError, match=r"<class 'int'> is not a subclass of types.Model"):
            n.instantiate_model('n')


def test_instantiate_model_calls_init():
    n = _mini_network()
    calls = []

    class ClassWithInit(types.Model):
        def init_model_deploy(
            self, model_info: types.ModelInfo, dataset_config: dict, **kwargs
        ) -> None:
            assert isinstance(model_info, types.ModelInfo)
            assert isinstance(dataset_config, dict)
            assert 'name' in kwargs
            calls.append('init_model_deploy')

    with patch.object(n, 'model_class', return_value=ClassWithInit):
        m = n.instantiate_model('n')
    assert isinstance(m, ClassWithInit)
    assert calls == ['init_model_deploy']


def test_instantiate_model_for_deployment_creates_preprocess_from_method():
    torchvision = pytest.importorskip("torchvision")
    import torchvision.transforms.functional as TF

    n = _mini_network()
    calls = []

    class ClassWithPreprocess(types.Model):
        def init_model_deploy(self, model_info, dataset_config, **kwargs):
            pass

        def override_preprocess(self, img):
            calls.append('preprocess')
            return TF.to_tensor(img)

    task = n.tasks[0]
    task.context = operators.PipelineContext()
    with patch.object(n, 'model_class', return_value=ClassWithPreprocess):
        m = n.instantiate_model_for_deployment(task)
    assert isinstance(m, ClassWithPreprocess)
    assert task.preprocess != []
    x = types.Image.fromany(PIL.Image.new('RGB', (10, 10)))
    for p in task.preprocess:
        x = p.exec_torch(x)
    assert calls == ['preprocess']


def test_instantiate_model_for_deployment_raises_error_if_no_preprocess():
    n = _mini_network()

    class ClassWithNoPreprocess(types.Model):
        def init_model_deploy(self, model_info, dataset_config, **kwargs):
            pass

    with pytest.raises(RuntimeError):
        task = n.tasks[0]
        with patch.object(n, 'model_class', return_value=ClassWithNoPreprocess):
            m = n.instantiate_model_for_deployment(task)


@pytest.mark.parametrize(
    "input_data,expected_output",
    [
        ('', []),
        (None, []),
        ('dog,cat,bird,sheep', ['dog', 'cat', 'bird', 'sheep']),
        ('dog;cat;bird;sheep', ['dog', 'cat', 'bird', 'sheep']),
        ('dog;cat,bird,sheep', ['dog', 'cat', 'bird', 'sheep']),
        ('   dog  ,  cat  ;  bird  ,  sheep  ', ['dog', 'cat', 'bird', 'sheep']),
    ],
)
def test_load_labels_and_filters(input_data, expected_output):
    n = _mini_network(dataset={'labels_path': 'whocares', 'label_filter': input_data})
    valid = 'dog\ncat\nbird\nsheep\ngiraffe\n'
    with patch.object(pathlib.Path, 'is_file', return_value=True):
        with patch.object(pathlib.Path, 'read_text', return_value=valid):
            with patch.object(n, 'model_class', return_value=ClassNoInit):
                n.instantiate_model('n')
    mi = n.find_model('n')
    assert mi.labels == ['dog', 'cat', 'bird', 'sheep', 'giraffe']
    assert mi.label_filter == expected_output


def test_load_labels_and_filters_invalid_filter():
    n = _mini_network(dataset={'labels_path': 'somepath', 'label_filter': ' banana'})
    valid = 'dog\ncat\nbird\nsheep\ngiraffe\n'
    with patch.object(pathlib.Path, 'is_file', return_value=True):
        with patch.object(pathlib.Path, 'read_text', return_value=valid):
            with patch.object(n, 'model_class', return_value=ClassNoInit):
                with pytest.raises(Exception, match='label_filter contains invalid.*banana'):
                    n.instantiate_model('n')


def test_load_labels_and_filters_no_labels():
    n = _mini_network(dataset={'label_filter': ' banana'})
    with pytest.raises(Exception, match='label_filter cannot be used if there are no labels'):
        with patch.object(n, 'model_class', return_value=ClassNoInit):
            n.instantiate_model('n')


def test_load_labels_and_filters_empty_labels():
    n = _mini_network(dataset={'labels_path': 'somepath', 'label_filter': ' banana'})
    with patch.object(pathlib.Path, 'is_file', return_value=True):
        with patch.object(pathlib.Path, 'read_text', return_value=''):
            with patch.object(n, 'model_class', return_value=ClassNoInit):
                with pytest.raises(
                    Exception, match='label_filter cannot be used if there are no labels'
                ):
                    n.instantiate_model('n')


def test_load_labels_and_filters_bad_label_path():
    n = _mini_network(dataset={'labels_path': 'somepath'})
    with pytest.raises(FileNotFoundError, match=r"Labels file somepath not found"):
        with patch.object(n, 'model_class', return_value=ClassNoInit):
            n.instantiate_model('n')


def test_from_model_dir():
    n = _mini_network()
    with patch.object(os, 'getcwd', return_value='oldpwd'):
        with patch.object(os, 'chdir') as mock_chdir:
            with n.from_model_dir('n'):
                assert 'somebasedir' in sys.path
                assert mock_chdir.call_args_list == [call('somebasedir')]
            assert mock_chdir.call_args_list == [
                call('somebasedir'),
                call('oldpwd'),
            ]
            assert 'somebasedir' not in sys.path


def test_compiler_overrides_no_extra():
    mis = network.ModelInfos()
    mis.add_compiler_overrides('mymodel', {})
    assert {} == mis.model_compiler_overrides('mymodel', config.Metis.pcie)
    assert 1 == mis.determine_deploy_cores('mymodel', 1, config.Metis.none, False)
    # still 1 because max_compiler_cores is not set
    assert 1 == mis.determine_deploy_cores('mymodel', 4, config.Metis.none, False)
    assert 800 == mis.clock_profile('mymodel', config.Metis.none)
    assert 800 == mis.clock_profile('mymodel', config.Metis.pcie)
    assert 4 == mis.determine_execution_cores('mymodel', 4, config.Metis.none)
    assert 3 == mis.determine_execution_cores('mymodel', 3, config.Metis.none)
    assert 1 == mis.determine_execution_cores('mymodel', 1, config.Metis.none)


def test_compiler_overrides_cascade_no_cores_specified():
    mis = network.ModelInfos()
    mis.add_compiler_overrides('m0', {'max_compiler_cores': 3})
    mis.add_compiler_overrides('m1', {})
    with pytest.raises(logging_utils.UserError, match='The pipeline has multiple models but'):
        mis.determine_deploy_cores('m0', 4, config.Metis.none, False)
    with pytest.raises(logging_utils.UserError, match='model m1 does not specify aipu_cores'):
        mis.determine_deploy_cores('m1', 4, config.Metis.none, False)


def test_compiler_overrides_cascade():
    mis = network.ModelInfos()
    mis.add_compiler_overrides('m0', {'aipu_cores': 3, 'max_compiler_cores': 3})
    mis.add_compiler_overrides('m1', {'aipu_cores': 1, 'clock_profile': 400})
    assert 3 == mis.determine_deploy_cores('m0', 4, config.Metis.none, False)
    assert 1 == mis.determine_deploy_cores('m0', 4, config.Metis.none, True)
    assert 1 == mis.determine_deploy_cores('m1', 4, config.Metis.none, False)
    assert 1 == mis.determine_deploy_cores('m1', 4, config.Metis.none, True)
    assert 800 == mis.clock_profile('m0', config.Metis.none)
    assert 800 == mis.clock_profile('m0', config.Metis.pcie)
    assert 800 == mis.clock_profile('m0', config.Metis.m2)
    assert 400 == mis.clock_profile('m1', config.Metis.none)
    assert 400 == mis.clock_profile('m1', config.Metis.pcie)
    assert 400 == mis.clock_profile('m1', config.Metis.m2)


def test_compiler_overrides_with_overrides():
    mis = network.ModelInfos()
    extra = {
        'max_compiler_cores': 3,
    }
    mis.add_compiler_overrides('mymodel', extra)
    assert {
        'max_compiler_cores': 3,
    } == mis.model_compiler_overrides('mymodel', config.Metis.pcie)
    assert 1 == mis.determine_deploy_cores('mymodel', 1, config.Metis.none, False)
    assert 3 == mis.determine_deploy_cores('mymodel', 4, config.Metis.none, False)
    assert 1 == mis.determine_deploy_cores('mymodel', 4, config.Metis.none, True)
    assert 800 == mis.clock_profile('mymodel', config.Metis.none)
    assert 800 == mis.clock_profile('mymodel', config.Metis.pcie)
    assert 800 == mis.clock_profile('mymodel', config.Metis.m2)


def test_compiler_overrides_with_execution_overrides():
    mis = network.ModelInfos()
    extra = {
        'max_execution_cores': 3,
    }
    mis.add_compiler_overrides('mymodel', extra)
    assert {
        'max_execution_cores': 3,
    } == mis.model_compiler_overrides('mymodel', config.Metis.pcie)
    assert 1 == mis.determine_deploy_cores('mymodel', 1, config.Metis.none, False)
    assert 1 == mis.determine_deploy_cores('mymodel', 4, config.Metis.none, False)
    assert 3 == mis.determine_execution_cores('mymodel', 3, config.Metis.none)
    assert 1 == mis.determine_execution_cores('mymodel', 1, config.Metis.none)


def test_compiler_overrides_with_m2_overrides():
    mis = network.ModelInfos()
    extra = {
        'max_compiler_cores': 3,
        'm2': {
            'max_compiler_cores': 2,
            'clock_profile': 400,
        },
    }
    mis.add_compiler_overrides('mymodel', extra)
    assert {
        'max_compiler_cores': 2,
        'clock_profile': 400,
    } == mis.model_compiler_overrides('mymodel', config.Metis.m2)
    assert 1 == mis.determine_deploy_cores('mymodel', 1, config.Metis.m2, False)
    assert 2 == mis.determine_deploy_cores('mymodel', 4, config.Metis.m2, False)
    assert 400 == mis.clock_profile('mymodel', config.Metis.m2)
    assert {
        'max_compiler_cores': 3,
    } == mis.model_compiler_overrides('mymodel', config.Metis.pcie)
    assert 1 == mis.determine_deploy_cores('mymodel', 1, config.Metis.pcie, False)
    assert 3 == mis.determine_deploy_cores('mymodel', 4, config.Metis.pcie, False)
    assert 800 == mis.clock_profile('mymodel', config.Metis.pcie)


def test_compiler_overrides_with_m2_override_of_aipu_cores():
    mis = network.ModelInfos()
    extra = {
        'aipu_cores': 3,
        'max_compiler_cores': 3,
        'm2': {
            'aipu_cores': 2,
            'max_compiler_cores': 2,
        },
    }
    mis.add_compiler_overrides('mymodel', extra)
    assert {
        'aipu_cores': 2,
        'max_compiler_cores': 2,
    } == mis.model_compiler_overrides('mymodel', config.Metis.m2)
    assert 2 == mis.determine_deploy_cores('mymodel', 2, config.Metis.m2, False)
    assert 2 == mis.determine_deploy_cores('mymodel', 4, config.Metis.m2, False)
    assert {
        'aipu_cores': 3,
        'max_compiler_cores': 3,
    } == mis.model_compiler_overrides('mymodel', config.Metis.pcie)
    assert 3 == mis.determine_deploy_cores('mymodel', 3, config.Metis.pcie, False)
    assert 3 == mis.determine_deploy_cores('mymodel', 4, config.Metis.pcie, False)


def test_compiler_overrides_with_m2_override_of_execution_cores():
    mis = network.ModelInfos()
    extra = {'m2': {'max_execution_cores': 3}}
    mis.add_compiler_overrides('mymodel', extra)
    assert 1 == mis.determine_deploy_cores('mymodel', 2, config.Metis.m2, False)
    assert 1 == mis.determine_deploy_cores('mymodel', 4, config.Metis.m2, False)
    assert 2 == mis.determine_execution_cores('mymodel', 2, config.Metis.pcie)
    assert 4 == mis.determine_execution_cores('mymodel', 4, config.Metis.pcie)
    assert 1 == mis.determine_deploy_cores('mymodel', 3, config.Metis.pcie, False)
    assert 1 == mis.determine_deploy_cores('mymodel', 4, config.Metis.pcie, False)
    assert 2 == mis.determine_execution_cores('mymodel', 2, config.Metis.pcie)
    assert 4 == mis.determine_execution_cores('mymodel', 4, config.Metis.pcie)


def test_compiler_overrides_with_quantization_config():
    mis = network.ModelInfos()
    extra = {
        'compilation_config': {
            'quantization_debug': False,
            'quantizer_version': 1,
        },
    }
    mis.add_compiler_overrides('mymodel', extra)
    assert {
        'compilation_config': {
            'quantization_debug': False,
            'quantizer_version': 1,
        },
    } == mis.model_compiler_overrides('mymodel', config.Metis.pcie)


def test_network_cleanup_trigger_operator_pipeline_stopped():
    class MyOp(operators.AxOperator):
        def pipeline_stopped(self):
            pass

        def exec_torch(self, img, result, meta):
            return img, result, meta

        def build_gst(self, gst, stream_idx):
            pass

    nn = _mini_network(custom_postprocess=MyOp())

    spy_pipeline_stopped = Mock(wraps=nn.tasks[0].postprocess[0].pipeline_stopped)
    nn.tasks[0].postprocess[0].pipeline_stopped = spy_pipeline_stopped

    nn.cleanup()
    spy_pipeline_stopped.assert_called_once()
    assert spy_pipeline_stopped.mock_calls == [call()]


SQUEEZENET_NETWORK_WITH_MODEL_CARD = f'''
{SQUEEZENET_NETWORK}
internal-model-card:
    model_card: MC-000
'''


@patch('subprocess.run')
def test_model_dependencies_no_model_card(mock_pip):
    input = f'''
{SQUEEZENET_NETWORK}
'''
    mock_pip.return_value.stdout = mock_pip.return_value.stderr = ''
    net = parse_net(input, {'imagenet.yaml': IMAGENET_TEMPLATE})
    utils.ensure_dependencies_are_installed(net.dependencies)
    assert not mock_pip.mock_calls


@patch('subprocess.run')
def test_model_dependencies_no_deps(mock_pip):
    input = f'''
{SQUEEZENET_NETWORK_WITH_MODEL_CARD}
'''
    mock_pip.return_value.stdout = mock_pip.return_value.stderr = ''
    net = parse_net(input, {'imagenet.yaml': IMAGENET_TEMPLATE})
    utils.ensure_dependencies_are_installed(net.dependencies)
    assert not mock_pip.mock_calls


@patch('subprocess.run')
def test_model_dependencies_empty_deps(mock_pip):
    input = f'''
{SQUEEZENET_NETWORK_WITH_MODEL_CARD}
    dependencies: []
'''
    mock_pip.return_value.stdout = mock_pip.return_value.stderr = ''
    net = parse_net(input, {'imagenet.yaml': IMAGENET_TEMPLATE})
    utils.ensure_dependencies_are_installed(net.dependencies)
    assert not mock_pip.mock_calls


@patch('subprocess.run')
@patch('subprocess.Popen')
def test_model_dependencies(mock_popen, mock_run):
    input = f'''
{SQUEEZENET_NETWORK_WITH_MODEL_CARD}
    dependencies:
        - parp
        - toot >= 2.0, < 3.0
        - honk>1.0,<=2.0
        - -r file1.txt
        - -r    file2.txt
        - -rfile3.txt
        - -r $MY_FOLDER/requirements.txt
'''
    # Mock the dry-run to indicate some packages need installing
    mock_run.return_value = Mock(
        stdout=(
            "Collecting parp\n"
            "  Using cached parp-1.0.0.tar.gz\n"
            "Collecting toot\n"
            "  Using cached toot-2.1.0.tar.gz\n"
        ),
        stderr="",
    )

    # Mock the Popen process
    process_mock = Mock()
    process_mock.stdout.readline.side_effect = ['Installing...', '', None]
    process_mock.poll.return_value = 0
    process_mock.returncode = 0
    mock_popen.return_value = process_mock

    net = parse_net(input, {'imagenet.yaml': IMAGENET_TEMPLATE})
    with patch.dict(os.environ, MY_FOLDER='./myfolder'):
        utils.ensure_dependencies_are_installed(net.dependencies)

    # Assert the dry run call
    mock_run.assert_called_once_with(
        [
            'pip',
            'install',
            '--dry-run',
            'parp',
            'toot >= 2.0, < 3.0',
            'honk>1.0,<=2.0',
            '-rfile1.txt',
            '-rfile2.txt',
            '-rfile3.txt',
            '-r./myfolder/requirements.txt',
        ],
        encoding='utf8',
        check=True,
        capture_output=True,
    )

    # Assert the actual installation call
    import subprocess

    mock_popen.assert_called_once_with(
        [
            'pip',
            'install',
            'parp',
            'toot >= 2.0, < 3.0',
            'honk>1.0,<=2.0',
            '-rfile1.txt',
            '-rfile2.txt',
            '-rfile3.txt',
            '-r./myfolder/requirements.txt',
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        universal_newlines=True,
        env=ANY,  # Use ANY for environment comparison
    )


@patch('subprocess.run')
@patch('subprocess.Popen')
def test_model_dependencies_with_args(mock_popen, mock_run):
    input = f'''
{SQUEEZENET_NETWORK_WITH_MODEL_CARD}
    dependencies:
        - parp --no-cache-dir
        - toot >= 2.0, < 3.0
        - honk>1.0,<=2.0
        - -r file1.txt
        - -r    file2.txt
        - -rfile3.txt
        - -r $MY_FOLDER/requirements.txt
'''
    # Mock the dry-run to indicate some packages need installing
    mock_run.return_value = Mock(
        stdout=(
            "Collecting parp\n"
            "  Using cached parp-1.0.0.tar.gz\n"
            "Collecting toot\n"
            "  Using cached toot-2.1.0.tar.gz\n"
        ),
        stderr="",
    )

    # Mock the Popen process
    process_mock = Mock()
    process_mock.stdout.readline.side_effect = ['Installing...', '', None]
    process_mock.poll.return_value = 0
    process_mock.returncode = 0
    mock_popen.return_value = process_mock

    net = parse_net(input, {'imagenet.yaml': IMAGENET_TEMPLATE})
    with patch.dict(os.environ, MY_FOLDER='./myfolder'):
        utils.ensure_dependencies_are_installed(net.dependencies)

    # Assert the dry run call
    mock_run.assert_called_once_with(
        [
            'pip',
            'install',
            '--dry-run',
            'parp',
            'toot >= 2.0, < 3.0',
            'honk>1.0,<=2.0',
            '-rfile1.txt',
            '-rfile2.txt',
            '-rfile3.txt',
            '-r./myfolder/requirements.txt',
        ],
        encoding='utf8',
        check=True,
        capture_output=True,
    )

    # Assert the actual installation call
    import subprocess

    mock_popen.assert_has_calls(
        [
            call(
                [
                    'pip',
                    'install',
                    'toot >= 2.0, < 3.0',
                    'honk>1.0,<=2.0',
                    '-rfile1.txt',
                    '-rfile2.txt',
                    '-rfile3.txt',
                    '-r./myfolder/requirements.txt',
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                env=ANY,  # Use ANY for environment comparison
            ),
            call(
                [
                    'pip',
                    'install',
                    'parp',
                    '--no-cache-dir',
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                bufsize=1,
                universal_newlines=True,
                env=ANY,  # Use ANY for environment comparison
            ),
        ],
        any_order=True,
    )


@patch('subprocess.run')
def test_model_dependencies_dry_run(mock_pip):
    input = f'''
{SQUEEZENET_NETWORK_WITH_MODEL_CARD}
    dependencies:
        - parp
        - toot >= 2.0, < 3.0
        - -r file1.txt
        - -r    file2.txt
        - -rfile3.txt
        - -r $MY_FOLDER/requirements.txt
'''
    mock_pip.return_value.stdout = mock_pip.return_value.stderr = ''
    net = parse_net(input, {'imagenet.yaml': IMAGENET_TEMPLATE})
    with patch.dict(os.environ, MY_FOLDER='./myfolder'):
        utils.ensure_dependencies_are_installed(net.dependencies, dry_run=True)
    mock_pip.assert_called_once_with(
        [
            'pip',
            'install',
            '--dry-run',
            'parp',
            'toot >= 2.0, < 3.0',
            '-rfile1.txt',
            '-rfile2.txt',
            '-rfile3.txt',
            '-r./myfolder/requirements.txt',
        ],
        encoding=ANY,
        check=ANY,
        capture_output=ANY,
    )


@pytest.mark.parametrize(
    "input, expected, user, expect_debug",
    [
        ("/path/to/file.txt", "/path/to/file.txt", "whoever", True),
        ("/home/ubuntu/file.txt", "/home/ubuntu/file.txt", "ubuntu", True),
        ("/home/ubuntu/file.txt", "/home/ubuntu/file.txt", "fluffy", True),
        (
            "/home/ubuntu/.cache/axelera/file.txt",
            "/home/ubuntu/.cache/axelera/file.txt",
            "ubuntu",
            False,
        ),
        (
            "/home/ubuntu/.cache/axelera/file.txt",
            "/home/fluffy/.cache/axelera/file.txt",
            "fluffy",
            False,
        ),
        (
            "/home/parp/.cache/axelera/file.txt",
            "/home/toot/.cache/axelera/file.txt",
            "toot",
            False,
        ),
        (
            "/home/parp/.cache/axelerant/file.txt",
            "/home/parp/.cache/axelerant/file.txt",
            "toot",
            True,
        ),
        (
            "/nested/home/ubuntu/.cache/axelera/file.txt",
            "/nested/home/ubuntu/.cache/axelera/file.txt",
            "fluffy",
            True,
        ),
    ],
)
def test_localise_path(input, expected, user, expect_debug):
    with patch('os.path.expanduser', return_value=f'/home/{user}'):
        with patch('axelera.app.network.LOG') as mock_log:
            assert network.localise_path(input) == expected
            if expect_debug:
                mock_log.debug.assert_called_once()
                debug_call = mock_log.debug.call_args[0][0]
                assert ".cache/axelera/ not found" in debug_call
            else:
                mock_log.debug.assert_not_called()


# Tests for postamble ONNX validation feature
def test_postamble_onnx_validation_success(tmpdir):
    """Test that postamble ONNX validation passes when file exists."""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as temp_file:
        temp_file.write(b"dummy onnx content")
        temp_file.flush()

        config = InferenceOpConfig(postamble_onnx=temp_file.name)
        assert temp_file.name in config.postamble_onnx  # Path gets resolved


def test_postamble_onnx_validation_failure():
    """Test that postamble ONNX validation fails when file doesn't exist."""
    nonexistent_file = "/path/to/nonexistent/postamble.onnx"

    with pytest.raises(ValueError, match=r"Postamble ONNX model file .* does not exist"):
        InferenceOpConfig(postamble_onnx=nonexistent_file)


def test_postamble_onnx_validation_with_existing_file(tmpdir):
    """Test that postamble ONNX validation works when file exists."""
    with tempfile.NamedTemporaryFile(suffix=".onnx", delete=True) as temp_file:
        temp_file.write(b"dummy onnx content")
        temp_file.flush()

        config = InferenceOpConfig(postamble_onnx=temp_file.name)
        assert temp_file.name in config.postamble_onnx  # Path gets resolved


def test_postamble_onnx_validation_empty_path():
    """Test that postamble ONNX validation is skipped for empty path."""
    config = InferenceOpConfig(postamble_onnx="")
    assert config.postamble_onnx == ""

    config = InferenceOpConfig(postamble_onnx=None)
    assert config.postamble_onnx is None


def test_postamble_onnx_validation_relative_path(tmpdir):
    """Test that postamble ONNX validation works with relative paths."""
    onnx_file_path = tmpdir.join("postamble.onnx")
    onnx_file_path.write_binary(b"dummy onnx content")

    old_cwd = os.getcwd()
    try:
        os.chdir(str(tmpdir))
        config = InferenceOpConfig(postamble_onnx="postamble.onnx")
        assert "postamble.onnx" in config.postamble_onnx
    finally:
        os.chdir(old_cwd)
        os.remove(onnx_file_path)


def test_postamble_onnx_in_network_parsing():
    """Test that network parsing properly validates postamble ONNX files."""
    # We don't need to test the full parsing with real files here
    # The key test is that validation works correctly

    input_yaml = f'''
name: test-network
pipeline:
  - test-model:
      input:
        type: image
      preprocess:
      postprocess:
      inference:
        postamble_onnx: /nonexistent/postamble.onnx

models:
  test-model:
    class: AxTorchvisionSqueezeNet
    class_path: doesnotexist/squeezenet.py
    task_category: Classification
    input_tensor_layout: NCHW
    input_tensor_shape: [1, 3, 224, 224]
    input_color_format: RGB
'''

    # Should fail with postamble ONNX validation error
    with pytest.raises(ValueError, match=r"Postamble ONNX model file .* does not exist"):
        parse_net(input_yaml, {})


def test_postamble_onnx_in_network_parsing_missing_file():
    """Test that network parsing fails when postamble ONNX file is missing."""
    input_yaml = f'''
name: test-network
pipeline:
  - test-model:
      input:
        type: image
      preprocess:
      postprocess:
      inference:
        postamble_onnx: /nonexistent/postamble.onnx

models:
  test-model:
    class: AxTorchvisionSqueezeNet
    class_path: doesnotexist/squeezenet.py
    task_category: Classification
    input_tensor_layout: NCHW
    input_tensor_shape: [1, 3, 224, 224]
    input_color_format: RGB
'''

    with pytest.raises(ValueError, match=r"Postamble ONNX model file .* does not exist"):
        parse_net(input_yaml, {})


@pytest.mark.parametrize(
    ("module, klass"),
    [
        ("ax_models/base_onnx.py", "AxONNXModel"),
        ("ax_models/yolo/ax_ultralytics.py", "AxUltralyticsYOLO"),
        ("ax_models/yolo/ax_yolo.py", "AxYolo"),
        ("ax_models/yolo/ax_darknet.py", "AxYoloDarknet"),
        ("ax_models/mmlab/mmseg/ax_mmseg.py", "AxMMSegmentationPytorch"),
        ("ax_models/mmlab/mmseg/ax_mmseg.py", "AxMMSegmentationOnnx"),
        ("customers/qtec/model_semantic_segmentation.py", "QTec_UNet"),
        ("ax_models/torchvision/classifiers.py", "AxTorchvisionClassifierModel"),
        ("ax_models/torch/LPRNet.py", "LPRNetTorchModel"),
        ("ax_models/torch/facenet.py", "AxFaceNet"),
        ("ax_models/torch/ax_timm.py", "AxTimmModel"),
    ],
)
@pytest.mark.parametrize("bad_class", [True, False])
def test_network_weights_and_anchors_download_messages_and_class_fails(
    bad_class, module, klass, tmp_path
):
    path = 'whatever.thingy'
    url = 'http://whatever/whatever.thingy'

    n = _mini_network()
    mi = n.find_model('n')
    mi.weight_path = path
    mi.weight_url = url
    if klass == "AxTimmModel":
        mi.extra_kwargs = {
            'timm_model_args': {
                'name': 'tiny_tim',
            },
        }

    class DummyModel(types.Model):
        def __init__(self, *args, **kwargs):
            super().__init__()

    with patch.dict(
        sys.modules,
        onnx=MagicMock(),
        ultralytics=MagicMock(YOLO=Mock()),
        ax_models=MagicMock(base_torch=MagicMock(TorchModel=types.Model)),
        models=MagicMock(
            yolo=MagicMock(Model=DummyModel),
            common=MagicMock(),
            darknet=MagicMock(Darknet=DummyModel),
        ),
        torch=MagicMock(nn=MagicMock(Module=DummyModel)),
        numpy=MagicMock(),
        PIL=MagicMock(),
        timm=MagicMock(),
        utils=MagicMock(),
    ), patch.dict(
        sys.modules,
        {
            "mmseg.apis": MagicMock(),
            "mmengine.config": MagicMock(),
            "mmengine.registry": MagicMock(),
            "torch.utils.data": MagicMock(),
            "ax_models.base_torch": MagicMock(TorchModel=types.Model),
            "torch.nn": MagicMock(Module=DummyModel),
        },
    ):
        ModelClass = utils.import_class_from_file(klass, module)
        ModelClass.__init__ = lambda self, *args, **kwargs: None
        if bad_class:
            # force the class to fail here to provoke the appropriate error message
            ModelClass.init_model_deploy = lambda self, *args, **kwargs: 1 / 0
    with patch.object(n, 'model_class', return_value=ModelClass):
        # ensure weights error message has all the parts we expect with quotes around path and url
        with patch.object(utils, 'download', side_effect=RuntimeError("download failed")):
            # n.instantiate_model('n')
            expected = re.compile(
                rf'Failed to initialize model.+ failed to download weights at "{path}" from "{url}".+ check settings for "weight_url" and "weight_path"',
            )
            with pytest.raises(RuntimeError) as e:
                n.instantiate_model('n')
            msg = str(e.value).replace('\n', ' ')
            if bad_class:
                assert 'appropriate class' in msg, msg
            else:
                assert expected.match(msg), msg
                assert 'appropriate class' not in msg, msg

            mi.weight_url = None
            expected = re.compile(
                rf'Failed to initialize model.+ no suitable weights found at "{path}" and no "weight_url" specified',
            )

            with pytest.raises(RuntimeError) as e:
                n.instantiate_model('n')
            msg = str(e.value).replace('\n', ' ')
            if bad_class:
                assert 'appropriate class' in msg, msg
            else:
                assert expected.match(msg), msg
                assert 'appropriate class' not in msg, msg

        if klass == "AxONNXModel" and not bad_class:
            # check anchor messages similarly
            file = tmp_path / path
            file.write_text("dummy weights")
            mi.weight_path = str(file)

            anchors_path = 'anchors.thingy'
            anchors_url = 'http://whatever/anchors.thingy'
            mi.extra_kwargs = {
                'YOLO': {
                    'anchors_path': anchors_path,
                    'anchors_url': anchors_url,
                }
            }
            with patch.object(utils, 'download', side_effect=RuntimeError("download failed")):
                expected = re.compile(
                    rf'Failed to initialize model.+ failed to download anchors at "{anchors_path}" from "{anchors_url}".+ check settings for "anchors_url" and "anchors_path"',
                )
                with pytest.raises(RuntimeError) as e:
                    n.instantiate_model('n')
                assert expected.match(str(e.value).replace('\n', ' ')), str(e.value)

                mi.extra_kwargs = {
                    'YOLO': {
                        'anchors_path': anchors_path,
                    }
                }
                expected = re.compile(
                    rf'Failed to initialize model.+ no "anchors" specified, no suitable anchors found at "{anchors_path}" and no "anchors_url" specified',
                )
                with pytest.raises(RuntimeError) as e:
                    n.instantiate_model('n')
                assert expected.match(str(e.value).replace('\n', ' ')), str(e.value)

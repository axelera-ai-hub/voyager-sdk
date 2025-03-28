# Copyright Axelera AI, 2024
import contextlib
import logging
import os
import pathlib
import sys
from unittest.mock import ANY, Mock, call, patch

import PIL
import pytest

from axelera import types
from axelera.app import (
    config,
    constants,
    logging_utils,
    network,
    operators,
    pipeline,
    utils,
    yaml_parser,
)
from axelera.app.network import AxNetwork, initialize_model
from axelera.app.operators import InferenceConfig, Input, PipelineContext, Resize
from axelera.app.operators.custom_preprocessing import ConvertColorInput
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
                inference_config=InferenceConfig.from_dict({}, True),
                model_info=MODEL_INFO,
                context=PipelineContext(),
            )
        ],
        custom_operators={},
        model_infos=mis,
    )
    net.tasks[0].model_info.labels = []
    net.tasks[0].model_info.label_filter = []
    pipeline.update_pending_expansions(net.tasks[0])
    assert exp == net


def test_parse_network_pipeline_assets():
    input = f'''
{SQUEEZENET_NETWORK}
pipeline-assets:
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
pipeline-assets:
   - somelistitem
        ''',
            r'pipeline-assets must be dict \(found type list\)',
        ),
        (
            f'''
pipeline-assets:
   133
        ''',
            r'pipeline-assets must be dict \(found type int\)',
        ),
        (
            f'''
pipeline-assets:
   url:
      - md5
      - path
        ''',
            r'pipeline-assets element for url must be dict \(found type list\)',
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
    with pytest.raises(ValueError, match=error):
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
    with pytest.raises(FileNotFoundError, match=r'to import.*doesnotexist\.py'):
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
    msg = 'Failed to import module bad_ops : This file should not be imported'
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
models:
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
    input_tesnor_format: NCHW
''',
            r'input_tesnor_format is not a valid key for model squeezenet1.0-imagenet-onnx',
        ),
        (
            f'''
name: squeezenet1.0-imagenet-onnx
pipeline:
  - squeezenet1.0-imagenet-onnx:
      template_path: doesnotexist/imagenet.yaml

models:
{SQUEEZENET_MINIMAL_MODEL}
    input_tesnor_format: NCHW
    wibble: 1
''',
            r'input_tesnor_format and wibble are not valid keys for model squeezenet1.0-imagenet-onnx',
        ),
        (
            f'''
name: squeezenet1.0-imagenet-onnx
pipeline:
  - squeezenet1.0-imagenet-onnx:
      template_path: doesnotexist/imagenet.yaml

models:
{SQUEEZENET_MINIMAL_MODEL}
    input_tesnor_format: NCHW
    wibble: 1
    wobble: 2
''',
            r'input_tesnor_format, wibble and wobble are not valid keys for model squeezenet1.0-imagenet-onnx',
        ),
    ],
)
def test_parse_network_model_invalid_key(input, error):
    with pytest.raises(ValueError, match=error):
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
        net = parse_net(input, {'imagenet.yaml': IMAGENET_TEMPLATE})


def test_parse_all_builtin_networks():
    old = os.getcwd()
    os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    try:
        network_yaml_info = yaml_parser.get_network_yaml_info()
        for nn in network_yaml_info.get_all_info():
            if 'ax_models/tutorials/' in nn.yaml_path:
                continue
            try:
                network.parse_network_from_path(nn.yaml_path)
            except Exception as e:
                raise ValueError(
                    f"Error parsing network '{nn.name}' from {nn.yaml_path}: {str(e)}"
                ) from e
    finally:
        os.chdir(old)


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
            n.instantiate_model('n', with_data_adapter=True)


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
        m = n.instantiate_model('n', with_data_adapter=True)
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
                n.instantiate_model('n', with_data_adapter=True)
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
                    n.instantiate_model('n', with_data_adapter=True)


def test_load_labels_and_filters_no_labels():
    n = _mini_network(dataset={'label_filter': ' banana'})
    with pytest.raises(Exception, match='label_filter cannot be used if there are no labels'):
        with patch.object(n, 'model_class', return_value=ClassNoInit):
            n.instantiate_model('n', with_data_adapter=True)


def test_load_labels_and_filters_empty_labels():
    n = _mini_network(dataset={'labels_path': 'somepath', 'label_filter': ' banana'})
    with patch.object(pathlib.Path, 'is_file', return_value=True):
        with patch.object(pathlib.Path, 'read_text', return_value=''):
            with patch.object(n, 'model_class', return_value=ClassNoInit):
                with pytest.raises(
                    Exception, match='label_filter cannot be used if there are no labels'
                ):
                    n.instantiate_model('n', with_data_adapter=True)


def test_load_labels_and_filters_bad_label_path():
    n = _mini_network(dataset={'labels_path': 'somepath'})
    with pytest.raises(FileNotFoundError, match=r"Labels file somepath not found"):
        with patch.object(n, 'model_class', return_value=ClassNoInit):
            n.instantiate_model('n', with_data_adapter=True)


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
    assert 1 == mis.determine_deploy_cores('mymodel', 1, config.Metis.none)
    # still 1 because max_compiler_cores is not set
    assert 1 == mis.determine_deploy_cores('mymodel', 4, config.Metis.none)
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
        mis.determine_deploy_cores('m0', 4, config.Metis.none)
    with pytest.raises(logging_utils.UserError, match='model m1 does not specify aipu_cores'):
        mis.determine_deploy_cores('m1', 4, config.Metis.none)


def test_compiler_overrides_cascade():
    mis = network.ModelInfos()
    mis.add_compiler_overrides('m0', {'aipu_cores': 3, 'max_compiler_cores': 3})
    mis.add_compiler_overrides('m1', {'aipu_cores': 1, 'clock_profile': 400})
    assert 3 == mis.determine_deploy_cores('m0', 4, config.Metis.none)
    assert 1 == mis.determine_deploy_cores('m1', 4, config.Metis.none)
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
        'compilation_config': {'backend_config': {'double_buffer': True}},
    }
    mis.add_compiler_overrides('mymodel', extra)
    assert {
        'max_compiler_cores': 3,
        'compilation_config': {'backend_config': {'double_buffer': True}},
    } == mis.model_compiler_overrides('mymodel', config.Metis.pcie)
    assert 1 == mis.determine_deploy_cores('mymodel', 1, config.Metis.none)
    assert 3 == mis.determine_deploy_cores('mymodel', 4, config.Metis.none)
    assert 800 == mis.clock_profile('mymodel', config.Metis.none)
    assert 800 == mis.clock_profile('mymodel', config.Metis.pcie)
    assert 800 == mis.clock_profile('mymodel', config.Metis.m2)


def test_compiler_overrides_with_execution_overrides():
    mis = network.ModelInfos()
    extra = {
        'max_execution_cores': 3,
        'compilation_config': {'backend_config': {'double_buffer': True}},
    }
    mis.add_compiler_overrides('mymodel', extra)
    assert {
        'max_execution_cores': 3,
        'compilation_config': {'backend_config': {'double_buffer': True}},
    } == mis.model_compiler_overrides('mymodel', config.Metis.pcie)
    assert 1 == mis.determine_deploy_cores('mymodel', 1, config.Metis.none)
    assert 1 == mis.determine_deploy_cores('mymodel', 4, config.Metis.none)
    assert 3 == mis.determine_execution_cores('mymodel', 3, config.Metis.none)
    assert 1 == mis.determine_execution_cores('mymodel', 1, config.Metis.none)


def test_compiler_overrides_with_m2_overrides():
    mis = network.ModelInfos()
    extra = {
        'max_compiler_cores': 3,
        'compilation_config': {'backend_config': {'mvm_utilization_limit': 0.75}},
        'm2': {
            'max_compiler_cores': 2,
            'clock_profile': 400,
            'compilation_config': {'backend_config': {'mvm_utilization_limit': 0.5}},
        },
    }
    mis.add_compiler_overrides('mymodel', extra)
    assert {
        'max_compiler_cores': 2,
        'clock_profile': 400,
        'compilation_config': {'backend_config': {'mvm_utilization_limit': 0.5}},
    } == mis.model_compiler_overrides('mymodel', config.Metis.m2)
    assert 1 == mis.determine_deploy_cores('mymodel', 1, config.Metis.m2)
    assert 2 == mis.determine_deploy_cores('mymodel', 4, config.Metis.m2)
    assert 400 == mis.clock_profile('mymodel', config.Metis.m2)
    assert {
        'max_compiler_cores': 3,
        'compilation_config': {'backend_config': {'mvm_utilization_limit': 0.75}},
    } == mis.model_compiler_overrides('mymodel', config.Metis.pcie)
    assert 1 == mis.determine_deploy_cores('mymodel', 1, config.Metis.pcie)
    assert 3 == mis.determine_deploy_cores('mymodel', 4, config.Metis.pcie)
    assert 800 == mis.clock_profile('mymodel', config.Metis.pcie)


def test_compiler_overrides_with_m2_override_of_aipu_cores():
    mis = network.ModelInfos()
    extra = {
        'aipu_cores': 3,
        'max_compiler_cores': 3,
        'compilation_config': {'backend_config': {'mvm_utilization_limit': 0.75}},
        'm2': {
            'aipu_cores': 2,
            'max_compiler_cores': 2,
            'compilation_config': {'backend_config': {'mvm_utilization_limit': 0.5}},
        },
    }
    mis.add_compiler_overrides('mymodel', extra)
    assert {
        'aipu_cores': 2,
        'max_compiler_cores': 2,
        'compilation_config': {'backend_config': {'mvm_utilization_limit': 0.5}},
    } == mis.model_compiler_overrides('mymodel', config.Metis.m2)
    assert 2 == mis.determine_deploy_cores('mymodel', 2, config.Metis.m2)
    assert 2 == mis.determine_deploy_cores('mymodel', 4, config.Metis.m2)
    assert {
        'aipu_cores': 3,
        'max_compiler_cores': 3,
        'compilation_config': {'backend_config': {'mvm_utilization_limit': 0.75}},
    } == mis.model_compiler_overrides('mymodel', config.Metis.pcie)
    assert 3 == mis.determine_deploy_cores('mymodel', 3, config.Metis.pcie)
    assert 3 == mis.determine_deploy_cores('mymodel', 4, config.Metis.pcie)


def test_compiler_overrides_with_m2_override_of_execution_cores():
    mis = network.ModelInfos()
    extra = {'m2': {'max_execution_cores': 3}}
    mis.add_compiler_overrides('mymodel', extra)
    assert 1 == mis.determine_deploy_cores('mymodel', 2, config.Metis.m2)
    assert 1 == mis.determine_deploy_cores('mymodel', 4, config.Metis.m2)
    assert 2 == mis.determine_execution_cores('mymodel', 2, config.Metis.pcie)
    assert 4 == mis.determine_execution_cores('mymodel', 4, config.Metis.pcie)
    assert 1 == mis.determine_deploy_cores('mymodel', 3, config.Metis.pcie)
    assert 1 == mis.determine_deploy_cores('mymodel', 4, config.Metis.pcie)
    assert 2 == mis.determine_execution_cores('mymodel', 2, config.Metis.pcie)
    assert 4 == mis.determine_execution_cores('mymodel', 4, config.Metis.pcie)


def test_compiler_overrides_with_quantization_config():
    mis = network.ModelInfos()
    extra = {
        'compilation_config': {
            'quantization_config': {
                'quantization_debug': False,
                'quantizer_version': 1,
            }
        },
    }
    mis.add_compiler_overrides('mymodel', extra)
    assert {
        'compilation_config': {
            'quantization_config': {
                'quantization_debug': False,
                'quantizer_version': 1,
            }
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
    model-card: MC-000
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
    dependencies:
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

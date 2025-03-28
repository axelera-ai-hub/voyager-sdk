# Copyright Axelera AI, 2024
import argparse
import builtins
import contextlib
import os
from pathlib import Path
import shlex
import sys
from unittest.mock import patch

import pytest

from axelera.app import config, utils, yaml_parser

schema = {
    "$id": "data-path-commands-schema.json",
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "description": "A representation of a person, company, organization, or place",
    "title": "Axelera compiler configuration schema",
    "type": "object",
    "additionalProperties": False,
    "$defs": {
        "frontend_config": {
            "properties": {
                "apply_pword_padding": {"type": "boolean"},
            },
        },
        "backend_config": {
            "properties": {
                "host_arch": {"enum": ["x86_64", "aarch64"]},
                "target": {"enum": ["x86-emu", "axelera", "axelera-qemu", "aicore-simulator"]},
            },
        },
        "control_flow_config": {
            "properties": {
                "quantize_only": {"type": "boolean"},
            },
        },
    },
    "properties": {
        "frontend_config": {"$ref": "#/$defs/frontend_config"},
        "backend_config": {"$ref": "#/$defs/backend_config"},
        "control_flow_config": {"$ref": "#/$defs/control_flow_config"},
    },
}


@pytest.fixture
def mock_load_schema():
    with patch('axelera.app.config._load_schema', return_value=schema) as mock:
        yield mock


@pytest.fixture
def mock_log():
    with patch('axelera.app.config.LOG') as mock:
        yield mock


def test_valid_config(mock_load_schema, mock_log):
    valid_config = {
        'control_flow_config': {'quantize_only': False},
        'frontend_config': {'apply_pword_padding': False},
    }
    assert config._get_validated_config(valid_config) == valid_config
    mock_log.warning.assert_not_called()


def test_remove_undefined_section(mock_load_schema, mock_log):
    valid_config = {
        'control_flow_config': {'quantize_only': False},
    }
    config_with_invalid_section = {
        'invalid_section': {'key': 'value'},
        'control_flow_config': {'quantize_only': False},
    }
    assert config._get_validated_config(config_with_invalid_section) == valid_config
    mock_log.warning.assert_called()


def test_remove_undefined_key(mock_load_schema, mock_log):
    valid_config = {
        'control_flow_config': {'quantize_only': False},
    }
    config_with_invalid_key = {
        'control_flow_config': {'quantize_only': False, 'invalid_key': 'value'},
    }
    assert config._get_validated_config(config_with_invalid_key) == valid_config
    mock_log.warning.assert_called()


def test_schema_validation_error(mock_load_schema):
    config_with_validation_error = {
        'control_flow_config': {'quantize_only': 'invalid_type'},
    }
    with pytest.raises(ValueError) as exc_info:
        config._get_validated_config(config_with_validation_error)


@pytest.mark.parametrize(
    "input_schema",
    [
        ({'properties': {"name": {"type": "str"}}}),
    ],
)
def test_invalid_schema(mock_load_schema, input_schema):
    mock_load_schema.return_value = input_schema
    valid_config = {
        'control_flow_config': {'quantize_only': False},
        'frontend_config': {'apply_pword_padding': False},
    }
    with pytest.raises(ValueError) as exc_info:
        config._get_validated_config(valid_config)


def _caps(vaapi, opencl, opengl, aipu_cores):
    return config.HardwareCaps(
        getattr(config.HardwareEnable, vaapi),
        getattr(config.HardwareEnable, opencl),
        getattr(config.HardwareEnable, opengl),
        aipu_cores,
    )


@pytest.mark.parametrize(
    "args, exp_vaapi, exp_opencl, exp_opengl, exp_aipu_cores",
    [
        ('', 'disable', 'disable', 'detect', 4),
        ('--auto-vaapi', 'detect', 'disable', 'detect', 4),
        ('--enable-vaapi', 'enable', 'disable', 'detect', 4),
        ('--disable-vaapi', 'disable', 'disable', 'detect', 4),
        ('--auto-opencl', 'disable', 'detect', 'detect', 4),
        ('--enable-opencl', 'disable', 'enable', 'detect', 4),
        ('--disable-opencl', 'disable', 'disable', 'detect', 4),
        ('--auto-opengl', 'disable', 'disable', 'detect', 4),
        ('--enable-opengl', 'disable', 'disable', 'enable', 4),
        ('--disable-opengl', 'disable', 'disable', 'disable', 4),
        ('--aipu-cores=1', 'disable', 'disable', 'detect', 1),
        ('--aipu-cores=4', 'disable', 'disable', 'detect', 4),
    ],
)
def test_hardware_caps_argparser(args, exp_vaapi, exp_opencl, exp_opengl, exp_aipu_cores):
    parser = argparse.ArgumentParser()
    defaults = config.HardwareCaps(
        vaapi=config.HardwareEnable.disable,
        opencl=config.HardwareEnable.disable,
        opengl=config.HardwareEnable.detect,
    )
    config.HardwareCaps.add_to_argparser(parser, defaults)
    args = parser.parse_args(shlex.split(args))
    caps = config.HardwareCaps.from_parsed_args(args)
    exp = _caps(exp_vaapi, exp_opencl, exp_opengl, exp_aipu_cores)
    assert caps == exp


@pytest.mark.parametrize(
    "vaapi, opencl, opengl, aipu_cores, exp_args",
    [
        ('disable', 'disable', 'detect', 4, ''),
        ('detect', 'disable', 'detect', 4, '--auto-vaapi'),
        ('enable', 'disable', 'detect', 4, '--enable-vaapi'),
        ('disable', 'disable', 'detect', 4, ''),
        ('disable', 'detect', 'detect', 4, '--auto-opencl'),
        ('disable', 'enable', 'detect', 4, '--enable-opencl'),
        ('disable', 'disable', 'detect', 4, ''),
        ('disable', 'disable', 'detect', 1, '--aipu-cores=1'),
        ('disable', 'disable', 'detect', 4, ''),
        ('disable', 'disable', 'enable', 4, '--enable-opengl'),
        ('disable', 'disable', 'disable', 4, '--disable-opengl'),
    ],
)
def test_hardware_caps_as_argv(vaapi, opencl, opengl, aipu_cores, exp_args):
    got = _caps(vaapi, opencl, opengl, aipu_cores).as_argv()
    assert got == exp_args


@pytest.mark.parametrize(
    "vaapi, opencl, opengl, avails, exp_vaapi, exp_opencl, exp_opengl",
    [
        (
            'detect',
            'detect',
            'detect',
            'vaapi|opencl|opengl',
            'enable',
            'enable',
            'enable',
        ),
        (
            'detect',
            'detect',
            'detect',
            'vaapi',
            'enable',
            'disable',
            'disable',
        ),
        (
            'detect',
            'detect',
            'detect',
            'opencl',
            'disable',
            'enable',
            'disable',
        ),
        (
            'detect',
            'detect',
            'detect',
            'aipu',
            'disable',
            'disable',
            'disable',
        ),
        (
            'detect',
            'detect',
            'detect',
            'vaapi',
            'enable',
            'disable',
            'disable',
        ),
        (
            'disable',
            'detect',
            'detect',
            'vaapi|opencl',
            'disable',
            'enable',
            'disable',
        ),
        (
            'detect',
            'disable',
            'detect',
            'vaapi',
            'enable',
            'disable',
            'disable',
        ),
        (
            'detect',
            'detect',
            'disable',
            'opencl',
            'disable',
            'enable',
            'disable',
        ),
        ('enable', 'enable', 'detect', 'opengl', 'enable', 'enable', 'enable'),
    ],
)
def test_hardware_caps_detect(vaapi, opencl, opengl, avails, exp_vaapi, exp_opencl, exp_opengl):
    with patch.object(utils, 'is_vaapi_available', return_value='vaapi' in avails):
        with patch.object(utils, 'is_opencl_available', return_value='opencl' in avails):
            with patch.object(utils, 'is_opengl_available', return_value='opengl' in avails):
                got = _caps(vaapi, opencl, opengl, 4).detect_caps()

    exp = _caps(exp_vaapi, exp_opencl, exp_opengl, 4)
    assert got == exp


NETWORK_YAML = yaml_parser.NetworkYamlInfo()
NETWORK_YAML.add_info('yolo', 'yolo.yaml', {'models': {'YOLO-COCO': None}}, '', 'here')
NETWORK_YAML.add_info('resnet', 'resnet.yaml', {'models': {'RESNET-IMAGENET': None}}, '', 'there')
NETWORK_YAML.add_info(
    'facerecog',
    'facerecog.yaml',
    {'models': {'FACE-DETECT': None, 'FACE-RECOG': None}},
    '',
    'there',
)
NETWORK_YAML.add_info(
    'facerecog2',
    'facerecog2.yaml',
    {'models': {'FACE-DETECT': None, 'FACE-RECOG': None}},
    '',
    'everywhere',
)

DEFAULT_BUILD_ROOT = Path('/some_build_root')


@pytest.mark.parametrize(
    'args, exp',
    [
        (
            'yolo file.mp4',
            dict(network='yolo.yaml', sources=['file.mp4']),
        ),
        (
            'yolo rtsp://summit/',
            dict(network='yolo.yaml', sources=['rtsp://summit/']),
        ),
        (
            'yolo dataset',
            dict(
                network='yolo.yaml',
                sources=['dataset'],
                build_root=DEFAULT_BUILD_ROOT,
            ),
        ),
        ('yolo dataset', dict(network='yolo.yaml', sources=['dataset'])),
        (
            'yolo dataset:foo',
            dict(network='yolo.yaml', sources=['dataset:foo']),
        ),
        (
            'yolo.yaml file.mp4',
            dict(network='yolo.yaml', sources=['file.mp4']),
        ),
        (
            'yolo.yaml file1.mp4 file2.mp4',
            dict(
                network='yolo.yaml',
                sources=['file1.mp4', 'file2.mp4'],
            ),
        ),
        (
            'yolo.yaml file.mp4',
            dict(network='yolo.yaml', sources=['file.mp4']),
        ),
        (
            'yolo.yaml f1.mp4 f2.mp4 f3.mp4 f4.mp4',
            dict(
                network='yolo.yaml',
                sources=['f1.mp4', 'f2.mp4', 'f3.mp4', 'f4.mp4'],
            ),
        ),
        (
            'yolo.yaml f1.mp4 f2.mp4 f3.mp4 f4.mp4 f5.mp4 f6.mp4 f7.mp4 f8.mp4',
            dict(
                network='yolo.yaml',
                sources=[
                    'f1.mp4',
                    'f2.mp4',
                    'f3.mp4',
                    'f4.mp4',
                    'f5.mp4',
                    'f6.mp4',
                    'f7.mp4',
                    'f8.mp4',
                ],
            ),
        ),
        (
            'yolo.yaml file.mp4',
            dict(network='yolo.yaml', sources=['file.mp4']),
        ),
        (
            'yolo.yaml dataset_file1.mp4 dataset_file2.mp4 --pipe=torch',
            dict(
                network='yolo.yaml',
                sources=['dataset_file1.mp4', 'dataset_file2.mp4'],
            ),
        ),
        (
            'yolo ~/file.mp4',
            dict(network='yolo.yaml', sources=['/homer/file.mp4']),
        ),
        ('yolo dataset', dict(data_root=Path('/pwd/data'), build_root=DEFAULT_BUILD_ROOT)),
        ('yolo.yaml dataset', dict(data_root=Path('/pwd/data'), build_root=DEFAULT_BUILD_ROOT)),
        ('resnet dataset', dict(data_root=Path('/pwd/data'), build_root=DEFAULT_BUILD_ROOT)),
        (
            'resnet.yaml dataset',
            dict(data_root=Path('/pwd/data'), build_root=DEFAULT_BUILD_ROOT),
        ),
        ('yolo dataset --data-root=there', dict(data_root=Path('/pwd/there'))),
        ('yolo dataset --data-root=~/there', dict(data_root=Path('/homer/there'))),
        ('yolo dataset --build-root=temp', dict(build_root=Path('/pwd/temp'))),
        ('yolo dataset --build-root=/temp', dict(build_root=Path('/temp'))),
        ('yolo dataset --build-root=~/temp', dict(build_root=Path('/homer/temp'))),
        ('yolo dataset', dict(show_stats=False, aipu_cores=4, show_system_fps=True)),
        ('yolo dataset --no-show-system-fps', dict(show_system_fps=False, display=True)),
        (
            'yolo dataset --show-system-fps --no-display',
            dict(show_system_fps=True, display=False),
        ),
        ('yolo dataset --show-stats', dict(show_stats=True)),
        ('yolo dataset --show-stats --pipe=torch', dict(show_stats=False)),
        ('yolo dataset --aipu-cores=1', dict(aipu_cores=1)),
        ('yolo dataset --pipe=torch-aipu', dict(aipu_cores=1)),
        ('yolo dataset  --aipu-cores=4 --pipe=torch-aipu', dict(aipu_cores=1)),
        ('yolo dataset', dict(frames=0)),
        ('yolo dataset --frames=100', dict(frames=100)),
        ('yolo dataset', dict(pipe='gst')),
        ('yolo dataset --pipe=torch', dict(pipe='torch')),
        ('yolo dataset --pipe=Torch', dict(pipe='torch')),
        ('yolo dataset --pipe=torch-aipu', dict(pipe='torch-aipu')),
    ],
)
def test_inference_parser_torch_installed(args, exp):
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.dict(sys.modules, torch='torch'))
        stack.enter_context(
            patch.dict(
                os.environ,
                {'AXELERA_FRAMEWORK': '/pwd', 'AXELERA_BUILD_ROOT': str(DEFAULT_BUILD_ROOT)},
                clear=True,
            )
        )
        stack.enter_context(
            patch.object(Path, 'absolute', lambda p: p if p.is_absolute() else Path('/pwd') / p)
        )
        stack.enter_context(
            patch.object(Path, 'expanduser', lambda p: Path(str(p).replace('~', '/homer')))
        )
        p = config.create_inference_argparser(NETWORK_YAML)
        args = p.parse_args(shlex.split(args))
        actual = {k: getattr(args, k) for k in exp.keys()}
        assert actual == exp


# errors are output to stderr and SystemExit: 2 is raised, so convert to ArgumentError
def _mock_exit(status, msg):
    raise argparse.ArgumentError(None, msg)


@pytest.mark.parametrize(
    'args, error',
    [
        ('yolo', r'No source provided'),
        (
            'yolo dataset dataset:val --pipe=torch',
            r'Dataset sources cannot be used with multistream',
        ),
        (
            'yolo file1.mp4 dataset:val --pipe=torch',
            r'Dataset sources cannot be used with multistream',
        ),
        (
            'yolo dataset:val file1.mp4 --pipe=torch',
            r'Dataset sources cannot be used with multistream',
        ),
        ('yolo --pipe=torch', r'No source provided'),
        (
            '',
            r'the following arguments are required: network',
        ),
        ('yool dataset', r"Invalid network 'yool', did you mean yolo\?"),
        ('yool.yaml dataset', r"Invalid network 'yool.yaml', did you mean yolo.yaml\?"),
        (
            'facerecoo dataset',
            r"Invalid network 'facerecoo', did you mean one of: facerecog, facerecog2\?",
        ),
        (
            'zzzzz dataset',
            r"Invalid network 'zzzzz', no close match found. Please `make help` to see all available models.",
        ),
        ('yolo dataset --frames=-1', r'argument --frames: cannot be negative: -1'),
    ],
)
def test_inference_parser_errors(args, error):
    p = config.create_inference_argparser(NETWORK_YAML)
    p.exit = _mock_exit
    with patch.dict(sys.modules, torch='torch'):
        with pytest.raises(argparse.ArgumentError, match=error):
            p.parse_args(shlex.split(args))


@pytest.mark.parametrize(
    'args, msg',
    [
        (
            'yolo.yaml dataset',
            'Dataset source requires torch to be installed : no torch today',
        ),
        (
            'yolo.yaml somefile.mp4 --pipe=torch',
            'torch pipeline requires torch to be installed : no torch today',
        ),
        (
            'yolo.yaml somefile.mp4 --pipe=torch-aipu',
            'torch-aipu pipeline requires torch to be installed : no torch today',
        ),
        (
            'yolo.yaml dataset --pipe=torch',
            'Dataset source and torch pipeline require torch to be installed : no torch today',
        ),
    ],
)
def test_inference_parser_no_torch_installed_parser_errors(args, msg):
    orig_import = builtins.__import__

    def new_import(name, *args, **kwargs):
        if name == 'torch':
            raise ImportError('no torch today')
        return orig_import(name, *args, **kwargs)

    p = config.create_inference_argparser(NETWORK_YAML)
    p.exit = _mock_exit
    with patch.object(builtins, '__import__', new_import):
        with pytest.raises(argparse.ArgumentError, match=msg):
            p.parse_args(shlex.split(args))


@pytest.mark.parametrize(
    'args, error',
    [
        ('facerecog dataset', r'cascaded models are not supported'),
    ],
)
def test_inference_parser_errors_from_cascaded(args, error):
    def _unsupported_yaml_condition(info):
        if isinstance(info, yaml_parser.NetworkYamlBase):
            return info.cascaded
        else:
            raise ValueError("info must be an instance of config.NetworkYamlBase")

    p = config.create_inference_argparser(
        NETWORK_YAML,
        unsupported_yaml_cond=_unsupported_yaml_condition,
        unsupported_reason='cascaded models are not supported',
    )

    # errors are output to stderr and SystemExit: 2 is raised, so convert to ArgumentError
    def exit(status, msg):
        raise argparse.ArgumentError(None, msg)

    p.exit = exit
    with pytest.raises(argparse.ArgumentError, match=error):
        p.parse_args(shlex.split(args))


@pytest.mark.parametrize(
    'args, exp',
    [
        ('yolo', dict(network='yolo.yaml')),
        ('yolo.yaml', dict(network='yolo.yaml')),
        ('resnet', dict(data_root=Path('/pwd/data'), build_root=DEFAULT_BUILD_ROOT)),
        ('resnet.yaml', dict(data_root=Path('/pwd/data'), build_root=DEFAULT_BUILD_ROOT)),
        ('yolo --data-root=there', dict(data_root=Path('/pwd/there'))),
        ('yolo --data-root=~/there', dict(data_root=Path('/homer/there'))),
        ('yolo --build-root=temp', dict(build_root=Path('/pwd/temp'))),
        ('yolo --build-root=/temp', dict(build_root=Path('/temp'))),
        ('yolo --build-root=~/temp', dict(build_root=Path('/homer/temp'))),
        ('yolo --aipu-cores=1', dict(aipu_cores=1)),
        ('yolo --pipe=torch-aipu', dict(aipu_cores=1)),
        ('yolo  --aipu-cores=4 --pipe=torch-aipu', dict(aipu_cores=1)),
        ('yolo', dict(pipe='gst')),
        ('yolo --pipe=torch', dict(pipe='torch')),
        ('yolo --pipe=Torch', dict(pipe='torch')),
        ('yolo --pipe=torch-aipu', dict(pipe='torch-aipu')),
        ('yolo', dict(mode=config.DeployMode.PREQUANTIZED)),
        ('yolo --mode=quantize', dict(mode=config.DeployMode.QUANTIZE)),
        ('yolo --mode=quantcompile', dict(mode=config.DeployMode.QUANTCOMPILE)),
        ('yolo --mode=prequantized', dict(mode=config.DeployMode.PREQUANTIZED)),
        ('yolo --mode=PREQUANTIZED', dict(mode=config.DeployMode.PREQUANTIZED)),
        ('yolo', dict(metis=config.Metis.none)),
        ('yolo --metis=m2', dict(metis=config.Metis.m2)),
        ('yolo --metis=pcie', dict(metis=config.Metis.pcie)),
        ('yolo --metis=auto', dict(metis=config.Metis.none)),
        ('yolo --metis=none', dict(metis=config.Metis.none)),
    ],
)
def test_deploy_parser(args, exp):
    with contextlib.ExitStack() as stack:
        stack.enter_context(
            patch.dict(
                os.environ,
                {'AXELERA_FRAMEWORK': '/pwd', 'AXELERA_BUILD_ROOT': str(DEFAULT_BUILD_ROOT)},
                clear=True,
            )
        )
        stack.enter_context(
            patch.object(Path, 'absolute', lambda p: p if p.is_absolute() else Path('/pwd') / p)
        )
        stack.enter_context(
            patch.object(Path, 'expanduser', lambda p: Path(str(p).replace('~', '/homer')))
        )
        p = config.create_deploy_argparser(NETWORK_YAML)
        args = p.parse_args(shlex.split(args))
        actual = {k: getattr(args, k) for k in exp.keys()}
        assert actual == exp

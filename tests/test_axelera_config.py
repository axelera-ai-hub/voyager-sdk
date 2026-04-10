# Copyright Axelera AI, 2023
import argparse
import builtins
import contextlib
import os
from pathlib import Path
import shlex
import sys
from unittest.mock import patch

import numpy as np
import pytest

from axelera import types
from axelera.app import config, utils, yaml_parser
from axelera.app.config import Source, SourceType, TilingConfig


@pytest.fixture
def mock_log():
    with patch('axelera.app.config.LOG') as mock:
        yield mock


def _caps(vaapi, opencl, opengl):
    return config.HardwareCaps(
        getattr(config.HardwareEnable, vaapi),
        getattr(config.HardwareEnable, opencl),
        getattr(config.HardwareEnable, opengl),
    )


@pytest.mark.parametrize(
    'creator', [config.create_inference_argparser, config.create_deploy_argparser]
)
def test_help_does_not_error(capsys, creator):
    p = creator(NETWORK_YAML)
    with pytest.raises(SystemExit):
        p.parse_args(['--help'])
    out = capsys.readouterr().out
    assert '--help-network' in out
    if creator == config.create_inference_argparser:
        assert '--help-sources' in out
    else:
        assert '--help-sources' not in out


@pytest.mark.parametrize(
    'creator', [config.create_inference_argparser, config.create_deploy_argparser]
)
def test_help_network(capsys, creator):
    p = creator(NETWORK_YAML)
    with pytest.raises(SystemExit):
        p.parse_args(['--help-network'])
    out = capsys.readouterr().out
    assert 'facerecog2' in out


def test_deploy_help_does_not_error(capsys):
    p = config.create_deploy_argparser(NETWORK_YAML)
    with pytest.raises(SystemExit):
        p.parse_args(['--help'])  # Should not raise an error

    out = capsys.readouterr().out
    assert '--help-sources' not in out
    assert '--help-network' in out
    with pytest.raises(SystemExit):
        p.parse_args(['--help-network'])
    with pytest.raises(SystemExit):
        p.parse_args(['--help-sources'])


@pytest.mark.parametrize(
    "args, exp_vaapi, exp_opencl, exp_opengl",
    [
        ('', 'disable', 'disable', 'detect'),
        ('--auto-vaapi', 'detect', 'disable', 'detect'),
        ('--enable-vaapi', 'enable', 'disable', 'detect'),
        ('--disable-vaapi', 'disable', 'disable', 'detect'),
        ('--auto-opencl', 'disable', 'detect', 'detect'),
        ('--enable-opencl', 'disable', 'enable', 'detect'),
        ('--disable-opencl', 'disable', 'disable', 'detect'),
        ('--auto-opengl', 'disable', 'disable', 'detect'),
        ('--enable-opengl', 'disable', 'disable', 'enable'),
        ('--disable-opengl', 'disable', 'disable', 'disable'),
    ],
)
def test_hardware_caps_argparser(args, exp_vaapi, exp_opencl, exp_opengl):
    parser = argparse.ArgumentParser()
    defaults = config.HardwareCaps(
        vaapi=config.HardwareEnable.disable,
        opencl=config.HardwareEnable.disable,
        opengl=config.HardwareEnable.detect,
    )
    config.HardwareCaps.add_to_argparser(parser, defaults)
    args = parser.parse_args(shlex.split(args))
    caps = config.HardwareCaps.from_parsed_args(args)
    exp = _caps(exp_vaapi, exp_opencl, exp_opengl)
    assert caps == exp


@pytest.mark.parametrize(
    "vaapi, opencl, opengl, exp_args",
    [
        ('disable', 'detect', 'detect', ''),
        ('detect', 'detect', 'detect', '--auto-vaapi'),
        ('enable', 'detect', 'detect', '--enable-vaapi'),
        ('disable', 'disable', 'detect', '--disable-opencl'),
        ('disable', 'enable', 'detect', '--enable-opencl'),
        ('disable', 'detect', 'enable', '--enable-opengl'),
        ('disable', 'detect', 'disable', '--disable-opengl'),
    ],
)
def test_hardware_caps_as_argv(vaapi, opencl, opengl, exp_args):
    got = _caps(vaapi, opencl, opengl).as_argv()
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
                got = _caps(vaapi, opencl, opengl).detect_caps()

    exp = _caps(exp_vaapi, exp_opencl, exp_opengl)
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
        ('yolo dataset --no-show-system-fps', dict(show_system_fps=False, display='auto')),
        (
            'yolo dataset --show-system-fps --no-display',
            dict(show_system_fps=True, display=False),
        ),
        ('yolo dataset --no-display', dict(enable_opengl=config.HardwareEnable.disable)),
        ('yolo dataset --display=opengl', dict(enable_opengl=config.HardwareEnable.enable)),
        (
            'yolo dataset --display=auto --enable-opengl',
            dict(display='auto', enable_opengl=config.HardwareEnable.enable),
        ),
        (
            'yolo dataset --display=auto',
            dict(display='auto', enable_opengl=config.HardwareEnable.disable),
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


def test_task_render_config_dataclass():
    """Test that TaskRenderConfig is properly configured as a dataclass."""
    settings = config.TaskRenderConfig()
    assert settings.show_annotations is True
    assert settings.show_labels is True

    settings = config.TaskRenderConfig(show_annotations=False, show_labels=False)
    assert settings.show_annotations is False
    assert settings.show_labels is False

    # Test direct attribute access instead of to_dict
    assert hasattr(settings, 'show_annotations')
    assert hasattr(settings, 'show_labels')


def test_render_config_creation():
    """Test creating a RenderConfig with and without tasks."""
    # Test empty config
    render_config = config.RenderConfig()
    assert len(render_config._config) == 0

    # Test with task list using keyword-based initialization
    render_config = config.RenderConfig(
        detections=config.TaskRenderConfig(),
        segmentation=config.TaskRenderConfig(),
    )
    assert len(render_config._config) == 2
    assert 'detections' in render_config._config
    assert 'segmentation' in render_config._config

    # Default settings should be True for both options
    assert render_config._config['detections'].show_annotations is True
    assert render_config._config['detections'].show_labels is True

    # Test keyword-based initialization with custom settings
    render_config2 = config.RenderConfig(
        detections=config.TaskRenderConfig(show_annotations=False, show_labels=False),
        tracker=config.TaskRenderConfig(show_annotations=True, show_labels=False),
    )
    assert 'detections' in render_config2._config
    assert 'tracker' in render_config2._config
    assert render_config2['detections'].show_annotations is False
    assert render_config2['tracker'].show_labels is False


def test_render_config_set_task():
    """Test setting task configuration in RenderConfig."""
    render_config = config.RenderConfig(detections=config.TaskRenderConfig())

    # Test method chaining
    result = render_config.set_task('detections', False, False)
    assert result is render_config  # Should return self for chaining

    # Test settings were applied
    assert render_config._config['detections'].show_annotations is False
    assert render_config._config['detections'].show_labels is False

    # Test updating existing task
    render_config.set_task('detections', True, False)
    assert render_config._config['detections'].show_annotations is True
    assert render_config._config['detections'].show_labels is False

    # Test KeyError for non-existent task without force_register
    with pytest.raises(KeyError):
        render_config.set_task('new_task', False, False)

    # Test registering new task with force_register
    render_config.set_task('new_task', False, False, force_register=True)
    assert render_config._config['new_task'].show_annotations is False
    assert render_config._config['new_task'].show_labels is False


def test_render_config_get_item():
    """Test __getitem__ functionality of RenderConfig."""
    render_config = config.RenderConfig(existing=config.TaskRenderConfig())

    settings = render_config['existing']
    assert settings.show_annotations is True
    assert settings.show_labels is True

    with pytest.raises(KeyError):
        _ = render_config['new_task']

    # Register and access
    render_config.set_task('new_task', False, False, force_register=True)
    settings = render_config['new_task']
    assert settings.show_annotations is False
    assert settings.show_labels is False


def test_render_config_to_dict():
    """Test converting RenderConfig to dictionary format directly."""
    render_config = config.RenderConfig(
        task1=config.TaskRenderConfig(),
        task2=config.TaskRenderConfig(),
    )
    render_config.set_task('task1', True, True)
    render_config.set_task('task2', False, False)

    assert len(render_config._config) == 2
    assert 'task1' in render_config._config
    assert 'task2' in render_config._config

    assert render_config._config['task1'].show_annotations is True
    assert render_config._config['task1'].show_labels is True
    assert render_config._config['task2'].show_annotations is False
    assert render_config._config['task2'].show_labels is False


def test_task_render_config_from_dict():
    d = {'show_annotations': False, 'show_labels': True}
    settings = config.TaskRenderConfig.from_dict(d)
    assert isinstance(settings, config.TaskRenderConfig)
    assert settings.show_annotations is False
    assert settings.show_labels is True

    d_unknown = {'show_annotations': True, 'show_labels': True, 'extra': 1}
    with pytest.raises(ValueError, match='Unknown keys'):
        config.TaskRenderConfig.from_dict(d_unknown)


@pytest.mark.parametrize(
    'source_str, exp',
    [
        ('path with spaces.mp4', Source.VIDEO_FILE('path with spaces.mp4')),
        ('cwd/cwd-based.mp4', Source.VIDEO_FILE('cwd/cwd-based.mp4')),
        ('cwd/cwd-based.mp4@30', Source.VIDEO_FILE('cwd/cwd-based.mp4', fps=30)),
        ('cwd/cwd-based.mp4@auto', Source.VIDEO_FILE('cwd/cwd-based.mp4', fps=-1)),
        (
            '~/videos/special_chars-123!@#$.mp4',
            Source.VIDEO_FILE('/homer/videos/special_chars-123!@#$.mp4'),
        ),
        (
            '~/videos/special_chars-123!@#$.mp4@24',
            Source.VIDEO_FILE('/homer/videos/special_chars-123!@#$.mp4', fps=24),
        ),
        (
            'http://example.com/video.mp4?param=value',
            Source.HLS('http://example.com/video.mp4?param=value'),
        ),
        (
            'https://example.com/video.mp4#fragment',
            Source.HLS('https://example.com/video.mp4#fragment'),
        ),
        (
            'rtsp://example.com:8554/stream?auth=token',
            Source.RTSP('rtsp://example.com:8554/stream?auth=token'),
        ),
        (
            'http://user:pass@example.com/video.mp4',
            Source.HLS('http://user:pass@example.com/video.mp4'),
        ),
        (
            'rtsp://admin:password@192.168.1.100:554/stream',
            Source.RTSP('rtsp://admin:password@192.168.1.100:554/stream'),
        ),
        ('http://example.com/path/', Source.HLS('http://example.com/path/')),
        ('rtsp://example.com:8554/stream/', Source.RTSP('rtsp://example.com:8554/stream/')),
        ('usb', Source.USB('0')),
        ('usb:1', Source.USB('1')),
        ('usb:1:640x480', Source.USB('1', width=640, height=480)),
        ('usb@22', Source.USB('0', fps=22)),
        ('usb:0:640x480@21', Source.USB('0', width=640, height=480, fps=21)),
        ('usb:/dev/video0', Source.USB('/dev/video0')),
        ('usb:/dev/video1', Source.USB('/dev/video1')),
        ('usb:/dev/video1:640x480', Source.USB('/dev/video1', width=640, height=480)),
        ('usb:/dev/video0@22', Source.USB('/dev/video0', fps=22)),
        ('usb:/dev/video0:640x480@24', Source.USB('/dev/video0', width=640, height=480, fps=24)),
        (
            'usb:0:640x480@22/image/jpeg',
            Source.USB('0', width=640, height=480, fps=22, codec='image/jpeg'),
        ),
        ('usb:/dev/video7:/yuyv', Source.USB('/dev/video7', codec='yuyv')),
        ('fakevideo', Source.FAKE_VIDEO(width=1280, height=720, fps=30)),
        ('fakevideo:1024x768', Source.FAKE_VIDEO(width=1024, height=768, fps=30)),
        ('fakevideo:1024x768@22', Source.FAKE_VIDEO(width=1024, height=768, fps=22)),
        ('fakevideo@30', Source.FAKE_VIDEO(width=1280, height=720, fps=30)),
        ('/image/a.jpg', Source.IMAGE_FILES('/image/a.jpg', images=[Path('/image/a.jpg')])),
        ('/dir/', Source.IMAGE_FILES('/dir', images=[Path('/dir/a.jpg'), Path('/dir/b.jpg')])),
        ('dataset', Source.DATASET('val')),
        ('dataset:wibble', Source.DATASET('wibble')),
        ('loop:cwd/cwd-based.mp4', Source.VIDEO_FILE('cwd/cwd-based.mp4', loop=True)),
        ('loop:cwd/cwd-based.mp4@30', Source.VIDEO_FILE('cwd/cwd-based.mp4', fps=30, loop=True)),
        ('loop:cwd/cwd-based.mp4@auto', Source.VIDEO_FILE('cwd/cwd-based.mp4', fps=-1, loop=True)),
        (
            'loop:~/videos/special_chars-123!@#$.mp4',
            Source.VIDEO_FILE('/homer/videos/special_chars-123!@#$.mp4', loop=True),
        ),
        (
            'loop:~/videos/special_chars-123!@#$.mp4@24',
            Source.VIDEO_FILE('/homer/videos/special_chars-123!@#$.mp4', fps=24, loop=True),
        ),
        (
            'loop:/dir/',
            Source.IMAGE_FILES('/dir', images=[Path('/dir/a.jpg'), Path('/dir/b.jpg')], loop=True),
        ),
        (
            '/dir/@15',
            Source.IMAGE_FILES('/dir', images=[Path('/dir/a.jpg'), Path('/dir/b.jpg')], fps=15),
        ),
        (
            'loop:/dir/@10',
            Source.IMAGE_FILES(
                '/dir', images=[Path('/dir/a.jpg'), Path('/dir/b.jpg')], fps=10, loop=True
            ),
        ),
    ],
)
def test_source_from_str(source_str, exp):
    with contextlib.ExitStack() as stack:
        enter = stack.enter_context
        enter(patch.object(Path, 'resolve', lambda p: Path(str(p).replace('cwd/', '/cwd/'))))
        enter(patch.object(Path, 'expanduser', lambda p: Path(str(p).replace('~', '/homer'))))
        isfile = exp.type == SourceType.VIDEO_FILE or (
            exp.type == SourceType.IMAGE_FILES and len(exp.images) == 1
        )
        isdir = exp.type == SourceType.IMAGE_FILES and not isfile
        files = [Path('/dir/a.jpg'), Path('/dir/b.jpg')] if isdir else []
        enter(patch.object(Path, 'is_file', return_value=isfile))
        enter(patch.object(Path, 'is_dir', return_value=isdir))
        enter(patch.object(utils, 'list_images_recursive', return_value=files))
        assert Source(source_str) == exp


def test_source_allows_string_with_kwargs():
    with patch.object(Path, 'is_file', return_value=True):
        source = Source('video1.mp4', preprocessing=config.rotate90())
        assert source == Source(
            config.SourceType.VIDEO_FILE, location='video1.mp4', preprocessing=config.rotate90()
        )


def test_source_ctor_from_source():
    source = Source('usb:/dev/video0:640x480@30')
    assert source == Source(source)
    with pytest.raises(TypeError, match=r'Source\(\) takes 1 string'):
        Source(source, 'invalid_type', 'bob')
    source2 = Source('usb:/dev/video1:640x480@30')
    assert source2 == Source(source, location='/dev/video1')


def test_source_ctor_invalid():
    with pytest.raises(
        TypeError, match=r'When using a string parameter you must use kwargs, not args'
    ):
        Source('invalid_source', 'invalid_type')
    with pytest.raises(TypeError, match=r'Source\(\) takes 1 string'):
        Source(0, 'invalid_type')
    with pytest.raises(TypeError, match=r'Source\(\) takes 1 string'):
        Source(0, 'invalid_type')
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(Path, 'is_file', return_value=False))
        stack.enter_context(patch.object(Path, 'is_dir', return_value=True))
        stack.enter_context(patch.object(Path, 'rglob', return_value=[Path('a.txt')]))
        with pytest.raises(RuntimeError, match=r'Failed to locate any images in /dir'):
            Source('/dir/')
    with contextlib.ExitStack() as stack:
        stack.enter_context(patch.object(Path, 'is_file', return_value=False))
        stack.enter_context(patch.object(Path, 'is_dir', return_value=False))
        with pytest.raises(
            FileNotFoundError, match=r'No such file or directory: /doesnotexist.mp4'
        ):
            Source('/doesnotexist.mp4')

    with pytest.raises(ValueError, match=r'Unrecognized source:'):
        Source('notavalid:type')


def test_source_typo_in_usb():
    with pytest.raises(ValueError, match=r'Badly formatted stream properties:'):
        Source('usb:/dev/video0:640x480x30')


def test_pipeline_config_dataset():

    with pytest.raises(ValueError, match=r'Dataset source cannot be used with multiple sources'):
        config.PipelineConfig('yolov5s-v7-coco', ['dataset:test', 'rtsp://hello/'])

    with pytest.raises(ValueError, match=r'Dataset source cannot be used with multiple sources'):
        config.PipelineConfig('yolov5s-v7-coco', ['rtsp://hello/', 'dataset:test'])

    with pytest.raises(ValueError, match=r'Dataset source cannot be used with multiple sources'):
        config.PipelineConfig('yolov5s-v7-coco', [config.Source.DATASET('test'), 'rtsp://hello/'])


def test_calibration_batch_raises_warning(caplog):
    caplog.set_level('WARNING')
    parser = config.create_deploy_argparser(NETWORK_YAML)
    args = parser.parse_args(['yolo', '--calibration-batch', '4'])
    r = caplog.records[0]
    assert r.levelname == 'WARNING'
    assert r.message.startswith('Please note --calibration-batch is not supported')
    assert args.calibration_batch == 1
    parser = config.create_inference_argparser(NETWORK_YAML)
    args = parser.parse_args(['yolo', 'fakevideo', '--calibration-batch', '4'])
    r = caplog.records[0]
    assert r.levelname == 'WARNING'
    assert r.message.startswith('Please note --calibration-batch is not supported')
    assert args.calibration_batch == 1


@pytest.mark.parametrize(
    's, args',
    [
        ('', []),
        ('1', ['1']),
        ('1,2', ['1', '2']),
        ('1,[2,3]', ['1', '[2,3]']),
        ('1,[2,3],4', ['1', '[2,3]', '4']),
        ('1,format=2,3, [4,5]', ['1', 'format=2', '3', '[4,5]']),
        ('1, [ 2,3,4,5,6 ]', ['1', '[ 2,3,4,5,6 ]']),
        ('[1],2,3,4,5,6,[7]', ['[1]', '2', '3', '4', '5', '6', '[7]']),
        ('format=[1],2,3,4,5,6,[7]', ['format=[1]', '2', '3', '4', '5', '6', '[7]']),
    ],
)
def test_split_args(s, args):
    """Test the _split_args function."""
    assert config._split_args(s) == args


# some common image preproc operators for testing
_rotate90 = config.ImagePreproc('videoflip', (config.VideoFlipMethod.clockwise,), {})
_rotate180 = config.ImagePreproc('videoflip', (config.VideoFlipMethod.rotate_180,), {})
_rotate270 = config.ImagePreproc('videoflip', (config.VideoFlipMethod.counterclockwise,), {})
_perspective = config.ImagePreproc('perspective', ([0.1, 0.2, 0.3], ''), {})
_perspective_rgb = config.ImagePreproc('perspective', ([0.1, 0.2, 0.3], 'rgb'), {})
_camera_undistort = config.ImagePreproc(
    'camera_undistort',
    (
        0.614,
        1.091,
        0.488,
        0.482,
        [-0.37793616, 0.11966818, -0.00067655, 0.0, -0.00115868],
        True,
        types.ColorFormat.RGB,
    ),
    {},
)
_vflip = config.ImagePreproc('videoflip', (config.VideoFlipMethod.vertical_flip,), {})


@pytest.mark.parametrize(
    'src, remaining, ops',
    [
        ('videoflip:rtsp://localhost', 'rtsp://localhost', [_rotate90]),
        ('videoflip:v.mp4', 'v.mp4', [_rotate90]),
        ('videoflip[]:v.mp4', 'v.mp4', [_rotate90]),
        ('rotate90:v.mp4', 'v.mp4', [_rotate90]),
        ('rotate180:v.mp4', 'v.mp4', [_rotate180]),
        ('rotate270:v.mp4', 'v.mp4', [_rotate270]),
        ('videoflip[counterclockwise]:v.mp4', 'v.mp4', [_rotate270]),
        ('videoflip[method=vertical_flip]:v.mp4', 'v.mp4', [_vflip]),
        ('perspective[[0.1,0.2,0.3]]:v.mp4', 'v.mp4', [_perspective]),
        ('perspective[[0.1,0.2,0.3],rgb]:v.mp4', 'v.mp4', [_perspective_rgb]),
        (
            'camera_undistort[0.614,1.091,0.488,0.482,[-0.37793616, 0.11966818, -0.00067655, 0, -0.00115868],normalized=True]:v.mp4',
            'v.mp4',
            [_camera_undistort],
        ),
    ],
)
def test_parse_image_preprocs(src, remaining, ops):
    gotremaining, gotops = config._parse_image_preprocs(src)
    assert gotremaining == remaining
    assert gotops == ops


@pytest.mark.parametrize(
    'src,cfg_contents,ops',
    [
        ('source_config[myconfig.txt]:v.mp4', 'rotate90', [_rotate90]),
        ('source_config[myconfig.txt]:v.mp4', 'rotate90:', [_rotate90]),
        (
            'source_config[myconfig.txt]:v.mp4',
            'rotate90:perspective[[0.1,0.2,0.3]]',
            [_rotate90, _perspective],
        ),
        (
            'source_config[myconfig.txt]:v.mp4',
            'rotate90\nperspective[[0.1,0.2,0.3]]',
            [_rotate90, _perspective],
        ),
        (
            'source_config[myconfig.txt]:v.mp4',
            'rotate90:\nperspective[[0.1,0.2,0.3]]',
            [_rotate90, _perspective],
        ),
        ('source_config[myconfig.txt]:rotate270:v.mp4', 'rotate90', [_rotate90, _rotate270]),
    ],
)
def test_parse_image_preprocs_load_source(src, cfg_contents, ops):
    with patch.object(Path, 'is_file', return_value=True):
        with patch.object(Path, 'read_text', return_value=cfg_contents):
            got = config.Source(src)
    assert Path(got.location).name == 'v.mp4'
    assert got.type == config.SourceType.VIDEO_FILE
    assert got.preprocessing == ops


@pytest.mark.parametrize(
    'src, err',
    [
        (
            'perspective:v.mp4',
            "Invalid arguments for perspective: missing a required argument: 'camera_matrix'",
        ),
        (
            'perspective[]:v.mp4',
            "Invalid arguments for perspective: missing a required argument: 'camera_matrix'",
        ),
        (
            'videoflip[reverse=False]:v.mp4',
            "Invalid arguments for videoflip: got an unexpected keyword argument 'reverse'",
        ),
        ('videoflip[1]:v.mp4', "Invalid value '1' for enum VideoFlipMethod in param 'method'"),
        (
            'videoflip[klokwize]:v.mp4',
            "Invalid value 'klokwize' for enum VideoFlipMethod in param 'method'",
        ),
        (
            'videoflip[too,many,args]:v.mp4',
            "Invalid arguments for videoflip: too many positional arguments",
        ),
    ],
)
def test_parse_image_preprocs_errors(src, err):
    with pytest.raises(Exception, match=err):
        remaining, ops = config._parse_image_preprocs(src)
        assert False, f"did not raise, but returned {remaining=} {ops=}"


def image_generator():
    for i in range(3):
        yield np.zeros((480, 640, 3), dtype=np.uint8)


def types_image_generator():
    for img in image_generator():
        yield types.Image.fromarray(img)


def test_source_data_source_from_np_generator():
    gen = image_generator()
    source = Source(gen)

    assert source.type == SourceType.DATA_SOURCE
    assert source.reader is gen
    assert source.preprocessing == []


def test_source_data_source_in_pipeline_config():
    source = Source(image_generator())
    conf = config.PipelineConfig("yolo.yaml", source)
    assert conf.sources[0].type == SourceType.DATA_SOURCE


def test_source_data_source_multiple_sources():
    source = Source(image_generator())
    cfg = config.PipelineConfig("yolo.yaml", [source, "rtsp://example.com/stream"])
    assert len(cfg.sources) == 2


def test_source_data_source_with_types_image():
    gen = types_image_generator()
    source = Source(gen)
    assert source.type == SourceType.DATA_SOURCE
    assert source.reader is gen


def test_source_data_source_invalid():
    with pytest.raises(TypeError):
        Source([1, 2, 3])


def test_source_data_source_copy_with_reader():
    gen1 = image_generator()
    source1 = Source(gen1)

    gen2 = image_generator()
    source2 = Source(source1, reader=gen2)
    assert source2.reader is gen2


@pytest.mark.parametrize(
    'source_str, exp_tiling_kwargs, exp_preprocessing_names',
    [
        (
            'tile[size=640,overlap=50,position=left,show=1]:usb',
            dict(size=640, overlap=50, position='left', show=True, file=''),
            [],
        ),
        (
            'tile[size=320]:usb',
            dict(size=320, overlap=0, position='none', show=False, file=''),
            [],
        ),
        (
            'tile[file=tiles.json]:usb',
            dict(size=0, overlap=0, position='none', show=False, file='tiles.json'),
            [],
        ),
        (
            'tile[file=tiles.json,show=True]:usb',
            dict(size=0, overlap=0, position='none', show=True, file='tiles.json'),
            [],
        ),
        (
            'tile[size=512,overlap=25]:usb',
            dict(size=512, overlap=25, position='none', show=False, file=''),
            [],
        ),
        (
            'tile[position=right]:usb',
            dict(size=0, overlap=0, position='right', show=False, file=''),
            [],
        ),
        (
            'tile[size=640]:rotate90:usb',
            dict(size=640, overlap=0, position='none', show=False, file=''),
            ['videoflip'],
        ),
        (
            'rotate90:tile[size=640]:usb',
            dict(size=640, overlap=0, position='none', show=False, file=''),
            ['videoflip'],
        ),
        (
            'tile[size=640,show=0]:usb',
            dict(size=640, overlap=0, position='none', show=False, file=''),
            [],
        ),
    ],
)
def test_tiling_config_from_source_string(source_str, exp_tiling_kwargs, exp_preprocessing_names):
    source = Source(source_str)
    tiling = [o.args[0] for o in source.preprocessing if o.is_tiling][0]
    assert isinstance(tiling, TilingConfig)
    assert [op.name for op in source.preprocessing if not op.is_tiling] == exp_preprocessing_names
    for field_name, expected_value in exp_tiling_kwargs.items():
        assert getattr(tiling, field_name) == expected_value, (
            f"TilingConfig.{field_name}: expected {expected_value!r}, "
            f"got {getattr(tiling, field_name)!r}"
        )


@pytest.mark.parametrize(
    'source_str, error_pattern',
    [
        (
            'tile[file=tiles.json,size=640]:usb',
            r'Cannot specify both file and other parameters for til',
        ),
        (
            'tile[file=tiles.json,position=left]:usb',
            r'Cannot specify both file and other parameters for til',
        ),
        (
            'tile[position=meh]:usb',
            r'Invalid tile position: meh',
        ),
        (
            'tile[file=tiles.json,overlap=50]:usb',
            r'Cannot specify both file and other parameters for til',
        ),
        (
            'tile[size=640]:tile[size=320]:usb',
            r'Only one tile preproc is allowed per source',
        ),
    ],
)
def test_tiling_config_from_source_string_errors(source_str, error_pattern):
    with pytest.raises(ValueError, match=error_pattern):
        Source(source_str)

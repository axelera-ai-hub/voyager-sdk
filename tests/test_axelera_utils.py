# Copyright Axelera AI, 2024
from __future__ import annotations

import builtins
import logging
import pathlib
import shutil
import subprocess
import sys
import tarfile
import time
from unittest.mock import MagicMock, call, patch
import zipfile

import numpy as np
import pyopencl
import pytest

from axelera.app import utils, yaml


def test_load_yaml_no_file():
    with pytest.raises(FileNotFoundError):
        utils.load_yaml_by_reference("no_file.yaml", {})


def test_load_yaml_no_substitutions():
    with patch("pathlib.Path.read_text", return_value="not yaml"):
        assert "not yaml" == utils.load_yaml_by_reference("no_yaml.yaml", {})


@pytest.mark.skip(reason="this is invalid yaml syntax due to flow nodes")
def test_load_yaml_unterminated_substitutions():
    with patch("pathlib.Path.read_text", return_value="x: {{"):
        assert {"x": "{{"} == utils.load_yaml_by_reference("no_yaml.yaml", {})


@pytest.mark.parametrize(
    "test_name,input,refs,expected",
    [
        (
            "as_taken_from_template",
            "letterbox:\n  width: {{input_width}}\n  height: {{input_height}}",
            dict(input_width=100, input_height=200),
            {"letterbox": {"width": 100, "height": 200}},
        ),
        (
            "special_values",
            "values: {{i}} {{x}} {{y}} {{z}}",
            dict(i="s", x=None, y=True, z=False),
            {"values": "s null true false"},
        ),
        (
            "expansion works when the value is a list entry",
            "values:\n - {{i}}\n - {{x}}\n - {{y}}\n - {{z}}",
            dict(i="s", x=None, y=True, z=False),
            {"values": ["s", None, True, False]},
        ),
        (
            "expansion works in the key",  # which may or may not be desired?
            "value{{i}}: {{i}} {{x}} {{y}} {{z}}",
            dict(i="s", x=None, y=True, z=False),
            {"values": "s null true false"},
        ),
        (
            "expr evaluation",
            'values: ${str(42//3) + "x"}',
            dict(),
            {"values": "14x"},
        ),
        (
            "expr evaluation referencing a refs key",
            'values: ${str(i//3) + "x"}',
            dict(i=18),
            {"values": "6x"},
        ),
        (
            "expr evaluation in a list",
            "values:\n - ${42//3}",
            dict(),
            {"values": [14]},
        ),
        (
            "default value",
            "values: {{i:default_i}} {{j:default_j}}",
            dict(i="actual_i"),
            {"values": "actual_i default_j"},
        ),
        (
            "default value integer and specials",
            "values:\n"
            " - {{i:nope}}\n"
            " - {{j:14}}\n"
            " - {{k:null}}\n"
            " - {{l:true}}\n"
            " - {{m:false}}",
            dict(i="actual_i"),
            {"values": ["actual_i", 14, None, True, False]},
        ),
    ],
)
def test_load_yaml_substitutions(test_name, input, refs, expected, caplog):
    with patch("pathlib.Path.read_text", return_value=input):
        assert expected == utils.load_yaml_by_reference("inp.yaml", refs)
    assert "" == caplog.text


def test_load_yaml_unmatched_substitutions_raise_err_log(caplog):
    input = "values: ${{i}}"
    with patch("pathlib.Path.read_text", return_value=input):
        assert {"values": "${{i}}"} == utils.load_yaml_by_reference("inp.yaml", {"I": ""})
    assert "inp.yaml: Variable i missing argument from {'I'" in caplog.text


def test_load_yaml_unmatched_substitutions_in_comments_ignored(caplog):
    input = "values:\n - ok\n# - ${{also_ok}}"
    with patch("pathlib.Path.read_text", return_value=input):
        assert {"values": ["ok"]} == utils.load_yaml_by_reference("inp", {})
    assert "" == caplog.text


@pytest.mark.skip(reason="TODO filter unsafe evals")
@pytest.mark.parametrize(
    "test_name,input,refs,expected",
    [
        (
            "naughty eval (ought not eval)",
            "values: ${exit(42//3)}",
            dict(),
            {"values": "${exit(42//3)}"},
        ),
    ],
)
def test_load_yaml_unsafe_evals(test_name, input, refs, expected):
    with patch("pathlib.Path.read_text", return_value=input):
        assert expected == utils.load_yaml_by_reference("inp.yaml", refs)


def test_load_labels_empty_path():
    assert utils.load_labels("") == []


def test_load_labels_file_not_found():
    with patch.object(pathlib.Path, 'exists', return_value=False):
        with pytest.raises(FileNotFoundError):
            utils.load_labels("nonexistent_file.names")


def test_load_labels_plain_text_file(tmp_path):
    # Using pytest's tmp_path fixture for better temporary file handling
    temp_file = tmp_path / "labels.names"
    temp_file.write_text("person\nbicycle\ncar\n\n")  # Include empty line to test filtering

    assert utils.load_labels(str(temp_file)) == ["person", "bicycle", "car"]


def test_load_labels_yaml_dict_format(tmp_path):
    temp_file = tmp_path / "labels.yaml"
    temp_file.write_text(
        """
names:
  0: person
  1: bicycle
  2: car
  3: motorcycle
    """
    )

    assert utils.load_labels(str(temp_file)) == ["person", "bicycle", "car", "motorcycle"]


def test_load_labels_yaml_list_format(tmp_path):
    temp_file = tmp_path / "labels.yml"
    temp_file.write_text(
        """
names: ['guns', 'knife', 'sword']
    """
    )

    assert utils.load_labels(str(temp_file)) == ['guns', 'knife', 'sword']


def test_load_labels_yaml_missing_names(tmp_path):
    temp_file = tmp_path / "labels.yaml"
    temp_file.write_text(
        """
other_key: value
    """
    )

    with pytest.raises(ValueError) as excinfo:
        utils.load_labels(str(temp_file))
    assert "must contain a 'names' key" in str(excinfo.value)


def test_load_labels_yaml_invalid_names_type(tmp_path):
    temp_file = tmp_path / "labels.yaml"
    temp_file.write_text(
        """
names: "invalid string type"
    """
    )

    with pytest.raises(ValueError) as excinfo:
        utils.load_labels(str(temp_file))
    assert "must be either a list or dictionary" in str(excinfo.value)


def test_load_labels_yaml_unordered_dict_keys(tmp_path):
    temp_file = tmp_path / "labels.yaml"
    temp_file.write_text(
        """
names:
  2: car
  0: person
  3: motorcycle
  1: bicycle
    """
    )

    # Should be ordered by key, not by appearance in file
    assert utils.load_labels(str(temp_file)) == ["person", "bicycle", "car", "motorcycle"]


def test_is_method_overriden():
    class BaseClass:
        def some_method(self):
            pass

    class DerivedClass(BaseClass):
        def some_method(self):
            pass

    class AnotherDerivedClass(BaseClass):
        pass

    class OtherBase:
        pass

    class MultipleInheritence(OtherBase, DerivedClass):
        pass

    class NegativeMultipleInheritence(OtherBase, AnotherDerivedClass):
        pass

    base = BaseClass()
    derived = DerivedClass()
    another_derived = AnotherDerivedClass()
    multiple_inheritence = MultipleInheritence()
    negative_multiple_inheritence = NegativeMultipleInheritence()

    assert utils.is_method_overridden(base, 'some_method') == False
    assert utils.is_method_overridden(derived, 'some_method') == True
    assert utils.is_method_overridden(another_derived, 'some_method') == False
    assert utils.is_method_overridden(multiple_inheritence, 'some_method') == True
    assert utils.is_method_overridden(negative_multiple_inheritence, 'some_method') == False


def mock_time(*times):
    times = list(times)

    def mock_time():
        return times.pop(0)

    return mock_time


def test_catchtime_no_logger():
    with patch.object(time, 'perf_counter', mock_time(0.1, 0.5, 0.7, 1.0)):
        with utils.catchtime() as t:
            interim = t.time
        assert interim == 0.4
        assert t.time == 0.6


def test_catchtime():
    messages = []
    with patch.object(time, 'perf_counter', mock_time(0.1, 0.5, 0.7, 1.0)):
        with utils.catchtime('test', messages.append) as t:
            interim = t.time
        assert messages == ['test took 600.000 mseconds']
        assert interim == 0.4
        assert t.time == 0.6


def test_catchtime_reset():
    messages = []
    with patch.object(time, 'perf_counter', mock_time(0.1, 0.5, 0.7, 1.0)):
        with utils.catchtime('test', messages.append) as t:
            t.reset()
            interim = t.time
        assert messages == ['test took 500.000 mseconds']
        np.testing.assert_allclose([interim], [0.2])
        assert t.time == 0.5


def test_catchtime_early_stop():
    messages = []
    with patch.object(time, 'perf_counter', mock_time(0.1, 0.5, 0.7, 1.0)) as mock:
        with utils.catchtime('test', messages.append) as t:
            interim = t.time
            t.stop()
            assert mock() == 1.0  # advance time a bit
        assert messages == ['test took 600.000 mseconds']
        assert interim == 0.4
        assert t.time == 0.6


@pytest.mark.parametrize(
    'res, times, output',
    [
        ('s', [0.1, 0.5], 'test took 0.400 seconds'),
        ('m', [0.1, 0.5], 'test took 400.000 mseconds'),
        ('u', [0.1, 0.5], 'test took 400000.000 useconds'),
        ('', [0.1, 1.5], 'test took 1.400 seconds'),
        ('', [0.1, 0.5], 'test took 400.000 mseconds'),
        ('', [0.1, 0.10001], 'test took 10.000 useconds'),
    ],
)
def test_catchtime_resolution(res, times, output):
    messages = []
    with patch.object(time, 'perf_counter', mock_time(*times)):
        with utils.catchtime('test', messages.append, res) as t:
            pass
        assert messages == [output]


def test_catchtime_invalid_resolution():
    with pytest.raises(ValueError, match='Invalid resolution oops for timer, valid are '):
        messages = []
        with utils.catchtime('test', messages.append, 'oops'):
            pass

    with pytest.raises(ValueError, match='Invalid resolution oops for timer, valid are '):
        messages = []
        with utils.catchtime('test', messages.append, 'oops'):
            pass


@patch('psutil.Process')
def test_catch_resources_current_process(mock_process):
    mock_process.return_value.memory_info.return_value.rss = 1024 * 1024 * 10  # 10 MB
    mock_process.return_value.cpu_percent.return_value = 50.0

    with utils.catch_resources(interval=0.01) as monitor:
        mock_process.return_value.memory_info.return_value.rss = 1024 * 1024 * 20  # 20 MB
        mock_process.return_value.cpu_percent.return_value = 60.0
        time.sleep(0.1)
        mock_process.return_value.memory_info.return_value.rss = 1024 * 1024 * 15  # 15 MB
        mock_process.return_value.cpu_percent.return_value = 55.0
        time.sleep(0.1)

    assert monitor.peak_memory == 20.0
    assert monitor.peak_cpu == 60.0


@patch('axelera.app.utils.SystemResourceMonitor')
def test_catch_resources_with_logger(mock_monitor_class):
    mock_logger = MagicMock()

    # Create a mock instance of SystemResourceMonitor
    mock_monitor = MagicMock()
    mock_monitor.peak_memory = 20.0
    mock_monitor.peak_cpu = 60.0
    mock_monitor.average_memory = 17.5
    mock_monitor.average_cpu = 57.5
    mock_monitor.memory_snapshots = [20.0, 15.0]
    mock_monitor.cpu_snapshots = [60.0, 55.0]

    # Make the mock class return our mock instance
    mock_monitor_class.return_value = mock_monitor

    with utils.catch_resources(pid=1234, task_name='Test Task', logger=mock_logger, interval=0.1):
        pass

    # Check if the peak memory usage was logged correctly
    mock_logger.assert_any_call('[Test Task] Peak Memory Usage: 20.00 MB')
    mock_logger.assert_any_call('[Test Task] Average CPU Usage: 57.50%')
    mock_logger.assert_any_call('[Test Task] Average Memory Usage: 17.50 MB')
    mock_logger.assert_any_call('[Test Task] Peak CPU Usage: 60.00%')

    # Verify that the monitor methods were called
    mock_monitor.start.assert_called_once()
    mock_monitor.stop.assert_called_once()


@patch('threading.Thread')
def test_catch_resources_current_process_averages_with_loggrt(mock_thread):
    # This prevents the thread from happening at all, so we manually inject the values
    mock_logger = MagicMock()
    with utils.catch_resources(
        task_name='Test Task', logger=mock_logger, interval=0.01
    ) as monitor:
        monitor.memory_snapshots.extend([20.0, 15.0])
        monitor.cpu_snapshots.extend([60.0, 55.0])

    assert call('[Test Task] Peak Memory Usage: 20.00 MB') in mock_logger.mock_calls
    assert call('[Test Task] Average Memory Usage: 17.50 MB') in mock_logger.mock_calls
    assert call('[Test Task] Peak CPU Usage: 60.00%') in mock_logger.mock_calls
    assert call('[Test Task] Average CPU Usage: 57.50%') in mock_logger.mock_calls


def create_tar_gz_archive(file_path):
    tmp_path = file_path.parent
    nested_dir1 = tmp_path / "dir1"
    nested_dir2 = nested_dir1 / "dir2"
    nested_dir2.mkdir(parents=True, exist_ok=True)
    sample_txt_file = nested_dir2 / 'sample.txt'

    with open(sample_txt_file, 'w') as f:
        f.write('This is a sample file.')

    with tarfile.open(file_path, 'w:gz') as tar:
        tar.add(nested_dir1, arcname='dir1')

    shutil.rmtree(nested_dir1)


def create_zip_archive(file_path):
    tmp_path = file_path.parent
    nested_dir1 = tmp_path / "dir1"
    nested_dir2 = nested_dir1 / "dir2"
    nested_dir2.mkdir(parents=True, exist_ok=True)
    sample_txt_file = nested_dir2 / 'sample.txt'

    with open(sample_txt_file, 'w') as f:
        f.write('This is a sample file.')

    with zipfile.ZipFile(file_path, 'w') as zipf:
        zipf.write(sample_txt_file, arcname='dir1/dir2/sample.txt')

    shutil.rmtree(nested_dir1)


def test_extract_unsupported_format(tmp_path):
    unsupported_file = tmp_path / 'unsupported.7z'
    unsupported_file.write_text('This is an unsupported file format.')


def test_extract_invalid_file(tmp_path):
    non_existent_file = pathlib.Path(tmp_path) / 'non_existent.tar.gz'
    with pytest.raises(RuntimeError):
        utils.extract(non_existent_file, drop_dirs=1)


def test_extract_supported_formats(tmp_path):
    # Test .tar.gz format
    tar_gz_file = pathlib.Path(tmp_path) / 'sample.tar.gz'
    create_tar_gz_archive(tar_gz_file)
    utils.extract(tar_gz_file, drop_dirs=0)
    assert (tar_gz_file.parent / 'dir1' / 'dir2' / 'sample.txt').exists()

    # Test .tgz format
    tgz_file = pathlib.Path(tmp_path) / 'sample.tgz'
    create_tar_gz_archive(tgz_file)
    utils.extract(tgz_file, drop_dirs=0)
    assert (tgz_file.parent / 'dir1' / 'dir2' / 'sample.txt').exists()

    # Test .zip format
    zip_file = pathlib.Path(tmp_path) / 'sample.zip'
    create_zip_archive(zip_file)
    utils.extract(zip_file, drop_dirs=0)
    assert (zip_file.parent / 'dir1' / 'dir2' / 'sample.txt').exists()


def test_extract_dest(tmp_path):
    # Test .tar.gz format
    tar_gz_file = pathlib.Path(tmp_path) / 'sample.tar.gz'
    create_tar_gz_archive(tar_gz_file)
    dest = pathlib.Path(tmp_path) / 'dest_dir0'
    utils.extract(tar_gz_file, drop_dirs=0, dest=dest)
    assert (dest / 'dir1' / 'dir2' / 'sample.txt').exists()

    # Test .tgz format
    tgz_file = pathlib.Path(tmp_path) / 'sample.tgz'
    create_tar_gz_archive(tgz_file)
    dest = pathlib.Path(tmp_path) / 'dest_dir1'
    utils.extract(tgz_file, drop_dirs=0, dest=dest)
    assert (dest / 'dir1' / 'dir2' / 'sample.txt').exists()

    # Test .zip format
    zip_file = pathlib.Path(tmp_path) / 'sample.zip'
    create_zip_archive(zip_file)
    dest = pathlib.Path(tmp_path) / 'dest_dir2'
    utils.extract(zip_file, drop_dirs=0, dest=dest)
    assert (dest / 'dir1' / 'dir2' / 'sample.txt').exists()


def test_extract_drop_dirs(tmp_path):
    # Test .tar.gz format
    tar_gz_file = pathlib.Path(tmp_path) / 'sample.tar.gz'
    create_tar_gz_archive(tar_gz_file)

    utils.extract(tar_gz_file, drop_dirs=1)
    assert not (tar_gz_file.parent / 'dir1' / 'sample.txt').exists()
    assert (tar_gz_file.parent / 'dir2' / 'sample.txt').exists()

    utils.extract(tar_gz_file, drop_dirs=2)
    assert (tar_gz_file.parent / 'sample.txt').exists()

    # Test .zip format
    zip_file = pathlib.Path(tmp_path) / 'sample.zip'
    create_zip_archive(zip_file)

    utils.extract(zip_file, drop_dirs=1)
    assert not (zip_file.parent / 'dir1' / 'sample.txt').exists()
    assert (zip_file.parent / 'dir2' / 'sample.txt').exists()

    utils.extract(zip_file, drop_dirs=2)
    assert (zip_file.parent / 'sample.txt').exists()


@pytest.mark.parametrize(
    'out, expected, expected_level, expected_log',
    [
        (subprocess.CalledProcessError, False, logging.INFO, 'Could not exec vainfo'),
        ('nothing', False, logging.INFO, 'Did not find VA-API in vainfo output'),
        (
            'blurb\nvleading vainfo: Supported profile and entrypoints,\nmore',
            True,
            logging.INFO,
            'Did not find VA-API in vainfo output',
        ),
    ],
)
def test_is_vaapi_available_x86(caplog, out, expected, expected_level, expected_log):
    caplog.set_level(logging.INFO)

    class Out:
        stdout = out

    def run(cmd, **kwargs):
        assert cmd == ['vainfo']
        if out is subprocess.CalledProcessError:
            raise out(1, 'cmd')
        return Out()

    class Uname:
        machine = 'x86_64'

    with patch('platform.uname', return_value=Uname):
        with patch.object(subprocess, 'run', run) as m:
            assert utils.is_vaapi_available() == expected
    if expected:
        assert caplog.text == ''
    else:
        assert expected_log in caplog.text
        assert expected_level in [x.levelno for x in caplog.get_records('call')]


def test_is_vaapi_available_aarch64(caplog):
    caplog.set_level(logging.INFO)

    def run(cmd, **kwargs):
        assert False, "On arm we should not try to use vaapi"

    class Uname:
        machine = 'aarch64'

    with patch('platform.uname', return_value=Uname):
        with patch.object(subprocess, 'run', run) as m:
            assert utils.is_vaapi_available() is False
    assert caplog.text == '', 'We do not expect any logged output on arm'


def test_is_vaapi_available_cmd_failure():
    with patch.object(subprocess, 'run') as m:
        m.side_effect = subprocess.CalledProcessError(1, 'cmd')
        assert not utils.is_vaapi_available()


CPU = pyopencl.device_type.CPU
GPU = pyopencl.device_type.GPU
ACC = pyopencl.device_type.ACCELERATOR


@pytest.mark.parametrize(
    'platforms, expected, expected_msgs',
    [
        ([], False, ['No OpenCL platforms found']),
        ([[CPU]], False, ['No OpenCL GPU devices found']),
        ([[], [CPU]], False, ['No OpenCL GPU devices found']),
        ([[CPU], [ACC]], False, ['No OpenCL GPU devices found']),
        ([[ACC]], False, ['No OpenCL GPU devices found']),
        ([[GPU]], True, []),
        ([[CPU], [ACC, GPU], []], True, []),
        ([[CPU], [GPU], [ACC]], True, []),
        (RuntimeError('spam'), False, ['Failed to get OpenCL platforms', 'spam']),
        ([RuntimeError('spam')], False, ['Failed to get OpenCL devices for platform ', 'spam']),
    ],
)
def test_is_opencl_available(caplog, platforms, expected, expected_msgs):
    caplog.set_level(logging.WARNING)  # just test warning messages

    class Mod:
        device_type = pyopencl.device_type

        def get_platforms(self):
            if isinstance(platforms, Exception):
                raise platforms
            return [Platform(f'plat{n}', devices) for n, devices in enumerate(platforms)]

    class Platform:
        def __init__(self, name, devices):
            self.name = name
            self.devices = devices

        def get_devices(self, dtype):
            if isinstance(self.devices, Exception):
                raise self.devices
            return [Device(f'dev{n}/{d}') for n, d in enumerate(self.devices) if (d & dtype) != 0]

    class Device:
        def __init__(self, name):
            self.name = name

    with patch.dict(sys.modules, {'pyopencl': Mod()}):
        assert utils.is_opencl_available() == expected
    for expected_msg in expected_msgs:
        assert expected_msg in caplog.text


def test_is_opencl_available_import_error():
    with patch.object(builtins, '__import__') as m:
        m.side_effect = ImportError('ham')
        assert not utils.is_opencl_available()


@pytest.mark.parametrize(
    "version, expected",
    [
        ("gl,4,5", ("gl", 4, 5)),
        ("gl,3,1", ("gl", 3, 1)),
        ("gl,3,3", ("gl", 3, 3)),
        ("gles,3,1", ("gles", 3, 1)),
        ("gles,3,2", ("gles", 3, 2)),
        ("gl,1,0", ("gl", 1, 0)),
        ("GLES,3,1", ("gles", 3, 1)),
        ("GL,3,3", ("gl", 3, 3)),
        ("gLeS,3,1", ("gles", 3, 1)),
    ],
)
def test_get_backend_opengl_version(version, expected):
    default = "gl,3,3"
    assert utils.get_backend_opengl_version(version, default) == expected


@pytest.mark.parametrize(
    "version", ["gl,3,1,1", "gl,3", "gl,3,", "gl", "gles", "cv,3,3", "gl,4.5", "g33"]
)
def test_get_backend_opengl_version_error(version):
    default = "gl,3,3"
    expected = ("gl", 3, 3)
    with patch('axelera.app.utils.LOG') as log:
        output = utils.get_backend_opengl_version(version, default)
    assert output == expected
    log.error.assert_any_call(f"Invalid OpenGL backend {version}, using default ({default})")


def test_is_opengl_available_import_error():
    with patch.object(builtins, '__import__') as m:
        m.side_effect = ImportError('spam')
        assert not utils.is_opengl_available("gl,3,3")


@pytest.mark.parametrize('fail_on', ['', 'get_display', 'get_default_screen', 'get_best_config'])
def test_is_opengl_available(fail_on):
    class Mod:
        def get_display(self):
            if fail_on == 'get_display':
                raise RuntimeError('spam')
            return self

        def get_default_screen(self):
            if fail_on == 'get_default_screen':
                raise RuntimeError('spam')
            return self

        def get_best_config(self):
            if fail_on == 'get_best_config':
                raise RuntimeError('spam')
            return self

    mod = Mod()
    mod.display = mod
    expected = not fail_on
    with patch.dict(sys.modules, {'pyglet': mod}):
        assert utils.is_opengl_available("gl,3,3") == expected


def test_is_compiler_available_import_error():
    with patch.object(builtins, '__import__') as m:
        m.side_effect = ImportError('spam')
        assert not utils.is_compiler_available()


def test_is_compiler_available_import_ok():
    with patch.dict(sys.modules, {'axelera.compiler': object()}):
        assert utils.is_compiler_available()


def create_test_files(tmp_path):
    (tmp_path / 'file1.txt').write_text("This is file 1.")
    (tmp_path / 'file2.txt').write_text("This is file 2.")
    (tmp_path / 'subfolder').mkdir()
    (tmp_path / 'subfolder/file3.txt').write_text("This is file 3 in a subfolder.")


def test_zipdir(tmp_path):
    create_test_files(tmp_path)
    zip_path = tmp_path / 'test_archive.zip'
    utils.zipdir(tmp_path, zip_path)

    assert zip_path.exists()
    with zipfile.ZipFile(zip_path, 'r') as zipf:
        namelist = zipf.namelist()
        assert 'file1.txt' in namelist
        assert 'file2.txt' in namelist
        assert 'subfolder/file3.txt' in namelist


def test_spinner():
    pytest.importorskip('alive_progress')
    with utils.spinner(1) as bar:
        bar.title('hello')
        bar()
    with utils.spinner() as bar:
        bar.title('world')
        bar()


def test_spinner_fallback(caplog):
    caplog.set_level(logging.INFO)

    with patch.object(builtins, '__import__') as m:
        m.side_effect = ImportError('spam')
        with utils.spinner(1) as bar:
            bar.title('hello')
            bar()
        with utils.spinner() as bar:
            bar.title('world')
            bar()
    assert 'hello' in caplog.text
    assert 'world' in caplog.text


@pytest.mark.parametrize(
    "labels, expected, exp_log",
    [
        (
            ["dog", "cat", "rabbit"],
            [("dog", 0), ("cat", 1), ("rabbit", 2)],
            None,
        ),  # basic case, no aliases
        (
            ["dog, spaniel", "cat, persian", "rabbit"],
            [("dog", 0), ("spaniel", 0), ("cat", 1), ("persian", 1), ("rabbit", 2)],
            None,
        ),  # aliases
        (
            ["dog", "dog, spaniel", "cat", "rabbit"],
            [("dog", 0), ("spaniel", 1), ("cat", 2), ("rabbit", 3)],
            "Dataset enumeration: Replacing class 1 label 'dog' with alias 'spaniel', as 'dog' already a base name for class 0",
        ),  # duplicate base name with alias substitute
        (
            ["dog", "dog, spaniel", "dog, spaniel, springer spaniel", "cat", "rabbit"],
            [("dog", 0), ("spaniel", 1), ("springer_spaniel", 2), ("cat", 3), ("rabbit", 4)],
            "Dataset enumeration: Replacing class 2 label 'dog' with alias 'springer_spaniel', as 'dog' already a base name for class 0",
        ),  # duplicate base name, with first choice subsitute already used
        (
            ["dog, spaniel", "spaniel, springer spaniel", "cat", "rabbit"],
            [("dog", 0), ("spaniel", 1), ("springer_spaniel", 1), ("cat", 2), ("rabbit", 3)],
            "Dataset enumeration: Skipping class 0 alias 'spaniel', as it is a base name for class 1",
        ),  # base name exists as alias, alias available
        (
            ["dog, spaniel", "spaniel", "cat", "rabbit"],
            [("dog", 0), ("spaniel", 1), ("cat", 2), ("rabbit", 3)],
            "Dataset enumeration: Skipping class 0 alias 'spaniel', as it is a base name for class 1",
        ),  # base name exists as alias, alias not available
        (
            ["dog, spaniel", "dog", "cat", "rabbit"],
            [("spaniel", 0), ("dog", 1), ("cat", 2), ("rabbit", 3)],
            "Dataset enumeration: Replacing class 0 label 'dog' with alias 'spaniel', as 'dog' already a base name for class 1",
        ),  # duplicate base name, current class has no alias, but existing does.
        (
            ["spaniel, dog", "labrador, dog", "cat", "rabbit"],
            [("dog", 0), ("spaniel", 0), ("labrador", 1), ("cat", 2), ("rabbit", 3)],
            "",
        ),  # duplicate aliases
        ([], [], None),  # empty labels
    ],
)
def test_create_enumerators(labels, expected, exp_log):
    with patch('axelera.app.utils.LOG') as log:
        output = utils.create_enumerators(labels)
    unique = set(output)
    assert len(unique) == len(output)  # check no duplicates
    print(output)
    print(expected)
    assert unique == set(expected)  # order doesn't matter
    if exp_log:
        log.trace.assert_any_call(exp_log)


@pytest.mark.parametrize(
    "labels",
    [
        (["dog", "dog", "cat", "rabbit"]),  # Duplicate base names
        (["dog", "dog, spaniel", "spaniel, dog"]),  # Aliases available, but cannot resolve
        (
            ["spaniel, springer spaniel", "dog, spaniel", "dog"]
        ),  # Resolution possible, but too complex for current implementation
    ],
)
def test_create_enumerators_error(labels):
    with pytest.raises(ValueError):
        utils.create_enumerators(labels)


def test_yaml_bool():
    assert yaml.yamlBool("True") == 1
    assert yaml.yamlBool("true") == 1
    assert yaml.yamlBool("TRUE") == 1
    assert yaml.yamlBool(1) == 1
    assert yaml.yamlBool(yaml.yamlBool("true")) == yaml.yamlBool("true")

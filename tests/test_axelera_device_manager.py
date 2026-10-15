# Copyright Axelera AI, 2024
from __future__ import annotations

import contextlib
import functools
import logging
import os
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest

from axelera.app import config, device_manager


@functools.lru_cache(maxsize=1)
def _has_aipu() -> bool:
    return any(Path('/dev').glob('metis*'))


def _has_error_reporter_api() -> bool:
    from axelera import runtime

    return hasattr(runtime, 'activate_error_reporter') and hasattr(
        runtime, 'trigger_error_reporter'
    )


_needs_aipu = pytest.mark.skipif(not _has_aipu(), reason='needs an AIPU')
_needs_error_reporter_api = pytest.mark.skipif(
    not _has_error_reporter_api(), reason='needs error reporter runtime API'
)

_REPORTER_SCRIPT = """
import os
import sys

from axelera import runtime
from axelera.app import device_manager

os.environ['AXELERA_ENABLE_AUTOMATIC_REPORTS'] = '1'
with device_manager.create_device_manager('gst'):
    if sys.argv[1] == 'trigger':
        if not runtime.trigger_error_reporter():
            raise RuntimeError('failed to trigger error reporter')
    elif sys.argv[1] == 'crash':
        os._exit(86)
"""


def _wait_for_error_report(tmp_path: Path) -> Path:
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        reports = list(tmp_path.glob('report-*.zip'))
        assert len(reports) <= 1
        if reports:
            with contextlib.suppress(zipfile.BadZipFile):
                with zipfile.ZipFile(reports[0]) as archive:
                    if archive.testzip() is None:
                        return reports[0]
        time.sleep(0.1)
    pytest.fail('error reporter did not produce a valid report archive')


def _run_error_reporter_process(tmp_path: Path, termination: str) -> subprocess.CompletedProcess:
    framework_root = Path(__file__).resolve().parents[1]
    pythonpath = os.environ.get('PYTHONPATH')
    environment = os.environ | {
        'PYTHONPATH': (
            f'{framework_root}{os.pathsep}{pythonpath}' if pythonpath else str(framework_root)
        )
    }
    return subprocess.run(
        [sys.executable, '-c', _REPORTER_SCRIPT, termination],
        cwd=tmp_path,
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
    )


def _run_error_reporter(tmp_path: Path, termination: str, expected_returncode: int) -> None:
    result = _run_error_reporter_process(tmp_path, termination)
    assert result.returncode == expected_returncode, result.stderr

    report = _wait_for_error_report(tmp_path)
    assert report.stat().st_size > 0
    with zipfile.ZipFile(report) as archive:
        assert any(Path(name).name.startswith('report-') for name in archive.namelist())


@_needs_aipu
@_needs_error_reporter_api
@pytest.mark.parametrize('termination, expected_returncode', [('trigger', 0), ('crash', 86)])
def test_automatic_error_reporter_creates_report(tmp_path, termination, expected_returncode):
    _run_error_reporter(tmp_path, termination, expected_returncode)


@_needs_aipu
@_needs_error_reporter_api
def test_automatic_error_reporter_does_not_create_report_after_clean_exit(tmp_path):
    result = _run_error_reporter_process(tmp_path, 'clean')
    assert result.returncode == 0, result.stderr
    assert list(tmp_path.glob('report-*.zip')) == []


@pytest.fixture(scope="function", autouse=True)
def _reset_list_devices_cache():
    device_manager._list_devices._cache = None


def _make_device_info(board_type: int = 0, devicen: int = 0):
    from axelera import runtime

    name = f'metis-0:{devicen+1}:0'.encode('utf-8')
    device = runtime.axruntime.DeviceInfo(name, 4, 0, 0, b'', board_type, b'', 0)
    return runtime.DeviceInfo(device)


def mock_runtime(board_types: list[str], read_device_configuration: dict | None = None):
    from axelera import runtime

    board_types = [getattr(runtime.axruntime.BoardType, b) for b in board_types]
    ctx = MagicMock()
    ctx.__enter__.return_value = ctx
    ctx.list_devices.return_value = [_make_device_info(b, n) for n, b in enumerate(board_types)]
    ctx.read_device_configuration.return_value = read_device_configuration
    return patch.object(runtime, 'Context', return_value=ctx)


def test_no_devices_found():
    with mock_runtime([]):
        with pytest.raises(RuntimeError, match='No devices found'):
            device_manager.create_device_manager('gst')


def test_no_devices_found_override_board():
    with mock_runtime([]):
        with device_manager.create_device_manager('gst', override_board=config.Board.pcie) as dm:
            assert dm.get_board_type() == config.Board.pcie


@pytest.mark.parametrize('value, expected_calls', [(None, 0), ('0', 0), ('1', 1)])
def test_automatic_error_reporter_activation(monkeypatch, value, expected_calls):
    from axelera import runtime

    if value is None:
        monkeypatch.delenv('AXELERA_ENABLE_AUTOMATIC_REPORTS', raising=False)
    else:
        monkeypatch.setenv('AXELERA_ENABLE_AUTOMATIC_REPORTS', value)
    with patch.object(
        runtime, 'activate_error_reporter', create=True, return_value=True
    ) as activate:
        with mock_runtime(['OMEGA_PCIE']):
            with device_manager.create_device_manager('gst'):
                pass

    assert activate.call_count == expected_calls


@pytest.mark.parametrize(
    'result',
    [False, AttributeError('missing Python API'), OSError('missing runtime symbol')],
)
def test_automatic_error_reporter_activation_failure_is_nonfatal(monkeypatch, caplog, result):
    from axelera import runtime

    monkeypatch.setenv('AXELERA_ENABLE_AUTOMATIC_REPORTS', '1')
    kwargs = {'side_effect': result} if isinstance(result, Exception) else {'return_value': result}
    with patch.object(runtime, 'activate_error_reporter', create=True, **kwargs):
        with mock_runtime(['OMEGA_PCIE']):
            with device_manager.create_device_manager('gst') as dm:
                assert dm.get_board_type() == config.Board.pcie

    assert 'automatic error report' in caplog.text.lower()


@pytest.mark.parametrize('board_type', ['ALPHA_PCIE', 'ALPHA_M2'])
def test_alpha_devices_found(board_type):
    with mock_runtime([board_type]):
        with pytest.raises(RuntimeError, match='Failed to detect metis-0:1:0'):
            device_manager.create_device_manager('gst')


@pytest.mark.parametrize(
    'selector,got',
    [
        ('', ['metis-0:1:0', 'metis-0:2:0']),
        ('1', ['metis-0:2:0']),
        ('metis-0:1:0,1', ['metis-0:1:0', 'metis-0:2:0']),
        ('0,1', ['metis-0:1:0', 'metis-0:2:0']),
        ('0,1', ['metis-0:1:0', 'metis-0:2:0']),
        ('1,0', ['metis-0:2:0', 'metis-0:1:0']),
    ],
)
def test_two_devices_found(selector, got):
    with mock_runtime(['OMEGA_PCIE', 'OMEGA_PCIE']):
        with device_manager.create_device_manager('gst', device_selector=selector) as dm:
            assert dm.get_board_type() == config.Board.pcie
            names = [d.name for d in dm.devices]
    assert names == got


DEVICES = [[_make_device_info(devicen=n) for n in range(n)] for n in range(5)]


@pytest.mark.parametrize(
    'available, level, selector, expected, messages',
    [
        (DEVICES[1], logging.DEBUG, '', ['metis-0:1:0'], 'Using device metis-0:1:0'),
        (DEVICES[1], logging.DEBUG, 'metis-0:1:0', ['metis-0:1:0'], 'Selected device metis-0:1:0'),
        (
            DEVICES[2],
            logging.INFO,
            'metis-0:1:0',
            ['metis-0:1:0'],
            'Selected device metis-0:1:0 (deselected 1 other device)',
        ),
        (
            DEVICES[3],
            logging.INFO,
            'metis-0:1:0',
            ['metis-0:1:0'],
            'Selected device metis-0:1:0 (deselected 2 other devices)',
        ),
        (
            DEVICES[2],
            logging.INFO,
            '1',
            ['metis-0:2:0'],
            'Selected device metis-0:2:0 (deselected 1 other device)',
        ),
        (
            DEVICES[2],
            logging.DEBUG,
            '0,1',
            ['metis-0:1:0', 'metis-0:2:0'],
            'Selected devices metis-0:1:0, metis-0:2:0',
        ),
        (
            DEVICES[2],
            logging.DEBUG,
            '',
            ['metis-0:1:0', 'metis-0:2:0'],
            'Using devices metis-0:1:0, metis-0:2:0',
        ),
    ],
)
def test_device_selector_messages(caplog, level, available, selector, expected, messages):
    caplog.set_level(level)
    got = device_manager._select_devices(available, selector)
    assert [x.name for x in got] == expected
    assert [rec.message for rec in caplog.records] == messages.splitlines()


def test_torch_pipe_no_error():
    with mock_runtime([]):
        with device_manager.create_device_manager('torch') as dm:
            assert dm.get_board_type() == config.Board.none
            assert [] == dm.configure_boards_and_tracers(Mock(), [])


@pytest.mark.parametrize(
    'board_type, override, expected, warnings',
    [
        ('OMEGA_PCIE', config.Board.none, config.Board.pcie, ''),
        ('OMEGA_M2', config.Board.none, config.Board.m2, ''),
        ('OMEGA_DEVBOARD', config.Board.none, config.Board.pcie, 'Unknown board type'),
        ('OMEGA_SBC', config.Board.none, config.Board.pcie, ''),
        ('OMEGA_PCIE', config.Board.pcie, config.Board.pcie, ''),
        ('OMEGA_M2', config.Board.pcie, config.Board.pcie, ''),
        ('OMEGA_PCIE', config.Board.m2, config.Board.m2, ''),
        ('OMEGA_M2', config.Board.m2, config.Board.m2, ''),
    ],
)
def test_get_board_type(caplog, board_type, override, expected, warnings):
    with mock_runtime([board_type]):
        dm = device_manager.create_device_manager('gst', override_board=override)
        assert dm.get_board_type() == expected
    if warnings:
        assert warnings in caplog.text
    else:
        assert caplog.text == ''


@pytest.mark.parametrize(
    'board_type, expected',
    [
        ('', config.Board.none),
        ('ALPHA_PCIE', config.Board.none),
        ('OMEGA_PCIE', config.Board.pcie),
        ('OMEGA_M2', config.Board.m2),
        ('OMEGA_DEVBOARD', config.Board.pcie),
        ('OMEGA_SBC', config.Board.pcie),
    ],
)
def test_detect_board_type(board_type, expected):
    with mock_runtime([board_type] if board_type else []):
        mt = device_manager.detect_board_type()
        assert mt == expected


@pytest.mark.parametrize(
    'clock, expected, warnings',
    [
        (None, 800, 'No clock profile found'),
        ('', 800, 'Unparseable'),
        ('100', 100, ''),
        ('200', 200, ''),
        ('400', 400, ''),
        ('500', 500, ''),
        ('600', 600, ''),
        ('700', 700, ''),
        ('800', 800, ''),
        ('999', 999, ''),
        ('goo', 800, 'Unparseable'),
    ],
)
def test_tracer_clock_correct(caplog, clock, expected, warnings):
    ncores = 3
    cfg = {}
    if clock is not None:
        cfg = {f'clock_profile_core_{n}': clock for n in range(ncores)}
    mock_tracer = Mock()
    mock_nn = Mock()
    mock_nn.tasks = [Mock()]
    mock_nn.tasks[0].aipu_cores = ncores
    mock_nn.tasks[0].model_info.name = 'mymodel'
    mock_nn.tasks[0].is_dl_task = True
    mock_nn.model_infos.clock_profile.return_value = clock
    mock_nn.model_infos.mvm_limitation.return_value = 100
    with mock_runtime(['OMEGA_PCIE'], read_device_configuration=cfg):
        with device_manager.create_device_manager('gst') as dm:
            got = dm.configure_boards_and_tracers(mock_nn, [mock_tracer])
    assert got == [mock_tracer]
    exp = {n: expected for n in range(ncores)}
    devices = ['metis-0:1:0']
    mock_tracer.initialize_models.assert_called_once_with(
        mock_nn.model_infos, config.Board.pcie, exp, devices
    )
    if warnings:
        assert warnings in caplog.text
    else:
        assert caplog.text == ''


@pytest.mark.parametrize(
    'in_env,in_clock,in_board,in_mvm,clock,mvm',
    [
        (None, 600, 'OMEGA_PCIE', 88, 600, 88),
        ('', 600, 'OMEGA_PCIE', 88, 600, 88),
        ('', 600, 'OMEGA_SBC', 88, 600, 88),
        ('0', 600, 'OMEGA_PCIE', 88, None, None),
        ('1', 600, 'OMEGA_PCIE', 88, 600, 88),
        ('400', 600, 'OMEGA_PCIE', 88, 400, 88),
        ('400', 600, 'OMEGA_SBC', 88, 400, 88),
        ('400', 600, 'OMEGA_M2', 88, 400, 88),
        ('400,99', 600, 'OMEGA_M2', 88, 400, 99),
    ],
)
def test_configure_board(in_env, in_clock, in_board, in_mvm, clock, mvm):
    with contextlib.ExitStack() as c:
        newenv = {'AXELERA_CONFIGURE_BOARD': in_env} if in_env is not None else {}
        c.enter_context(patch.dict(os.environ, newenv))
        cfg = {f'clock_profile_core_{n}': in_clock for n in range(4)}
        Ctx = c.enter_context(mock_runtime([in_board], read_device_configuration=cfg))
        ctx = Ctx()
        nn = Mock()
        nn.tasks = [Mock()]
        nn.tasks[0].aipu_cores = 1
        nn.model_infos.clock_profile.return_value = in_clock
        nn.model_infos.mvm_limitation.return_value = in_mvm
        with device_manager.create_device_manager('gst') as dm:
            dm.configure_boards_and_tracers(nn, [])
        if clock is not None:
            ctx.configure_device.assert_called_once_with(
                ctx.list_devices()[0],
                device_firmware='1',
                clock_profile_core_0=clock,
                mvm_utilisation_core_0=mvm,
            )
        else:
            ctx.configure_device.assert_not_called()


@pytest.mark.parametrize(
    'in_env, err',
    [
        ('spam,100', 'Invalid literal for int'),
        ('spam', 'Invalid literal for int'),
        ('400,abc', "Invalid literal"),
    ],
)
def test_configure_board_bad_input(in_env, err):
    with contextlib.ExitStack() as c:
        c.enter_context(patch.dict(os.environ, {'AXELERA_CONFIGURE_BOARD': in_env}))
        ctx = c.enter_context(mock_runtime(['OMEGA_PCIE']))
        nn = Mock()
        nn.tasks = [Mock()]
        nn.tasks[0].aipu_cores = 1
        nn.tasks[0].is_dl_task = True
        nn.model_infos.clock_profile.return_value = 800
        nn.model_infos.mvm_limitation.return_value = 100
        with device_manager.create_device_manager('gst') as dm:
            with pytest.raises(
                ValueError, match=f'Badly formatted AXELERA_CONFIGURE_BOARD : {err}'
            ):
                dm.configure_boards_and_tracers(nn, [])
            ctx.configure_devices.assert_not_called()


@pytest.mark.parametrize(
    "key, value, expected",
    [
        ('mvm_utilisation_core_0', 100, '100%'),
        ('mvm_utilisation_core_1', 50, '50%'),
        ('mvm_utilisation_core_2', 37.5, '37.5%'),
        ('mvm_utilisation_core_3', 0, '0%'),
        ('clock_profile_core_0', 200, '200MHz'),
        ('clock_profile_core_1', 400, '400MHz'),
        ('clock_profile_core_2', 800, '800MHz'),
        ('clock_profile_core_3', 1000, '1.0GHz'),
        ('clock_profile_core_4', 1600, '1.6GHz'),
    ],
)
def test_human_readable(key, value, expected):
    assert device_manager._human_readable(key, value) == expected


def test_null_device_manager_devices_property():
    dm = device_manager._NullDeviceManager()
    assert dm.devices == []


def test_device_index_name_not_found():
    devices = [_make_device_info(devicen=n) for n in range(2)]
    with pytest.raises(ValueError, match='No device found matching nonexistent'):
        device_manager._device_index(devices, 'nonexistent')


def test_select_devices_invalid_selector_logs_and_raises():
    """Out-of-range numeric index falls through to name match and raises."""
    devices = [_make_device_info(devicen=n) for n in range(2)]
    # '5' is a valid int but out of range; the ValueError is caught and re-raised
    # as "No device found matching 5" because no device name equals "5".
    with pytest.raises(ValueError, match='No device found matching 5'):
        device_manager._select_devices(devices, '5')


def _has_europa_board_types():
    from axelera import runtime

    return hasattr(runtime.axruntime.BoardType, 'EUROPA_PCIE')


@pytest.mark.skipif(
    not _has_europa_board_types(), reason='axruntime wheel predates EUROPA board types'
)
@pytest.mark.parametrize(
    'board_type, expected',
    [
        ('EUROPA_PCIE', config.Board.europa),
        ('EUROPA_DEVBOARD', config.Board.europa),
    ],
)
def test_get_board_type_europa(board_type, expected):
    with mock_runtime([board_type]):
        dm = device_manager.create_device_manager('gst')
        assert dm.get_board_type() == expected


@pytest.mark.skipif(
    not _has_europa_board_types(), reason='axruntime wheel predates EUROPA board types'
)
@pytest.mark.parametrize(
    'board_type',
    ['EUROPA_PCIE', 'EUROPA_DEVBOARD'],
)
def test_configure_boards_europa_returns_empty(board_type):
    """Europa devices skip per-core configuration; _configure_boards returns {}."""
    with mock_runtime([board_type]) as ctx_patch:
        with device_manager.create_device_manager('gst') as dm:
            result = dm._configure_boards(Mock())
    assert result == {}
    ctx_patch().configure_device.assert_not_called()


def test_configure_boards_non_dl_task_skipped():
    """Tasks where is_dl_task is False should be skipped."""
    nn = Mock()
    dl_task = Mock()
    dl_task.is_dl_task = True
    dl_task.aipu_cores = 1
    dl_task.model_info.name = 'model1'

    non_dl_task = Mock()
    non_dl_task.is_dl_task = False

    nn.tasks = [non_dl_task, dl_task]
    nn.model_infos.clock_profile.return_value = 800
    nn.model_infos.mvm_limitation.return_value = 100

    cfg = {'clock_profile_core_0': 800}
    with mock_runtime(['OMEGA_PCIE'], read_device_configuration=cfg):
        with device_manager.create_device_manager('gst') as dm:
            tracers = dm.configure_boards_and_tracers(nn, [])
    assert tracers == []


def test_configure_boards_waits_when_device_not_ready():
    """When configure_device returns False, polls until device_ready is True."""
    nn = Mock()
    nn.tasks = [Mock()]
    nn.tasks[0].is_dl_task = True
    nn.tasks[0].aipu_cores = 1
    nn.tasks[0].model_info.name = 'model'
    nn.model_infos.clock_profile.return_value = 800
    nn.model_infos.mvm_limitation.return_value = 100

    cfg = {'clock_profile_core_0': 800}
    with mock_runtime(['OMEGA_PCIE'], read_device_configuration=cfg) as ctx_patch:
        ctx = ctx_patch()
        ctx.configure_device.return_value = False
        # Return False first (enters loop body / sleep), then True to exit
        ctx.device_ready.side_effect = [False, True]
        with patch('axelera.app.utils.spinner'):
            with patch('time.sleep') as mock_sleep:
                with device_manager.create_device_manager('gst') as dm:
                    dm.configure_boards_and_tracers(nn, [])
    ctx.configure_device.assert_called_once()
    mock_sleep.assert_called_once_with(0.3)


def test_configure_boards_and_tracers_tracer_exception_skipped(caplog):
    """Tracers that raise on initialize_models are skipped with a warning."""
    nn = Mock()
    nn.tasks = []
    nn.model_infos.clock_profile.return_value = 800

    bad_tracer = Mock()
    bad_tracer.initialize_models.side_effect = RuntimeError("tracer exploded")
    good_tracer = Mock()

    cfg = {}
    with mock_runtime(['OMEGA_PCIE'], read_device_configuration=cfg):
        with device_manager.create_device_manager('gst') as dm:
            result = dm.configure_boards_and_tracers(nn, [bad_tracer, good_tracer])

    assert result == [good_tracer]
    assert 'skip trace' in caplog.text


def test_create_device_manager_oserror_linux(caplog):
    """OSError from _AipuDeviceManager is logged as error on linux."""
    with patch('axelera.app.device_manager._AipuDeviceManager', side_effect=OSError("no device")):
        with patch('sys.platform', 'linux'):
            dm = device_manager.create_device_manager('gst')
    assert isinstance(dm, device_manager._NullDeviceManager)
    assert 'Failed to create device manager' in caplog.text


def test_create_device_manager_oserror_darwin(caplog):
    """OSError from _AipuDeviceManager is logged as info on darwin."""
    with caplog.at_level(logging.INFO):
        with patch(
            'axelera.app.device_manager._AipuDeviceManager', side_effect=OSError("no device")
        ):
            with patch('sys.platform', 'darwin'):
                dm = device_manager.create_device_manager('gst')
    assert isinstance(dm, device_manager._NullDeviceManager)
    assert 'Failed to create device manager' in caplog.text


def test_is_europa_device_via_hw_generation():
    """_is_europa_device returns True when hw_generation.name == 'europa'."""
    device = Mock()
    hw_gen = Mock()
    hw_gen.name = 'europa'
    device.hw_generation = hw_gen
    assert device_manager._is_europa_device(device) is True


def test_is_europa_device_via_hw_generation_not_europa():
    """_is_europa_device returns False when hw_generation.name != 'europa'."""
    device = Mock()
    hw_gen = Mock()
    hw_gen.name = 'omega'
    device.hw_generation = hw_gen
    device.board_type = Mock()
    device.board_type.name = 'omega_pcie'
    assert device_manager._is_europa_device(device) is False


@pytest.mark.parametrize(
    'global_clock, expected, warning',
    [
        ('800', 800, ''),
        ('not-a-number', 800, 'Unparseable global clock_profile'),
    ],
)
def test_get_core_clocks_global_clock_profile(caplog, global_clock, expected, warning):
    """When per-core clocks are missing, fall back to global clock_profile."""
    ctx = MagicMock()
    ctx.read_device_configuration.return_value = {'clock_profile': global_clock}
    device = _make_device_info()  # OMEGA board, non-europa

    result = device_manager._get_core_clocks(ctx, device, 0, 2)

    if warning:
        assert warning in caplog.text
        assert result == {0: 800, 1: 800}
    else:
        assert result == {0: expected, 1: expected}


@pytest.mark.parametrize(
    'env_value, expected',
    [
        (None, 'err'),
        ('dbg', 'dbg'),
        ('wrn', 'wrn'),
        ('inf:collector', 'inf:collector'),
    ],
)
def test_slog_level_applied_when_device_is_opened(monkeypatch, env_value, expected):
    """AXELERA_SLOG_LEVEL (default err) is pushed to sysctrl when the device is opened."""
    if env_value is None:
        monkeypatch.delenv('AXELERA_SLOG_LEVEL', raising=False)
    else:
        monkeypatch.setenv('AXELERA_SLOG_LEVEL', env_value)
    with mock_runtime(['OMEGA_PCIE']) as ctx_patch:
        ctx = ctx_patch()
        with device_manager.create_device_manager('gst'):
            pass
    ctx.device_set_log_level.assert_called_once_with(ctx.list_devices()[0], 'sysctrl', expected)


def test_slog_level_empty_leaves_the_firmware_level_alone(monkeypatch):
    """Empty is the documented opt-out, not a request to set the level to ''."""
    monkeypatch.setenv('AXELERA_SLOG_LEVEL', '')
    with mock_runtime(['OMEGA_PCIE']) as ctx_patch:
        ctx = ctx_patch()
        with device_manager.create_device_manager('gst'):
            pass
    ctx.device_set_log_level.assert_not_called()


def test_slog_level_applied_to_every_device(monkeypatch):
    monkeypatch.delenv('AXELERA_SLOG_LEVEL', raising=False)
    with mock_runtime(['OMEGA_PCIE', 'OMEGA_PCIE']) as ctx_patch:
        ctx = ctx_patch()
        with device_manager.create_device_manager('gst'):
            pass
    assert [c.args for c in ctx.device_set_log_level.call_args_list] == [
        (d, 'sysctrl', 'err') for d in ctx.list_devices()
    ]


def test_slog_level_failure_is_nonfatal(monkeypatch, caplog):
    """Older firmware may reject the call; a noisy log beats refusing to run."""
    monkeypatch.delenv('AXELERA_SLOG_LEVEL', raising=False)
    with mock_runtime(['OMEGA_PCIE']) as ctx_patch:
        ctx = ctx_patch()
        ctx.device_set_log_level.side_effect = RuntimeError('unsupported')
        with caplog.at_level(logging.WARNING):
            with device_manager.create_device_manager('gst') as dm:
                assert dm.devices
    assert 'sysctrl log level' in caplog.text
    assert 'unsupported' in caplog.text


def test_slog_level_reapplied_after_board_configuration(monkeypatch):
    """configure_device passes device_firmware, which can reload the firmware and reset
    a level set at open."""
    monkeypatch.delenv('AXELERA_SLOG_LEVEL', raising=False)
    nn = Mock()
    nn.tasks = [Mock()]
    nn.tasks[0].is_dl_task = True
    nn.tasks[0].aipu_cores = 1
    nn.tasks[0].model_info.name = 'model'
    nn.model_infos.clock_profile.return_value = 800
    nn.model_infos.mvm_limitation.return_value = 100
    cfg = {'clock_profile_core_0': 800}
    with mock_runtime(['OMEGA_PCIE'], read_device_configuration=cfg) as ctx_patch:
        ctx = ctx_patch()
        with device_manager.create_device_manager('gst') as dm:
            assert ctx.device_set_log_level.call_count == 1, 'set once when opened'
            dm.configure_boards_and_tracers(nn, [])
    ctx.configure_device.assert_called_once()
    assert ctx.device_set_log_level.call_count == 2, 're-applied after configuration'


_NO_RUNTIME_SCRIPT = """
import sys


class _BlockRuntime:
    '''Make `axelera.runtime` unimportable, as on a compile-only install.'''

    def find_spec(self, name, path=None, target=None):
        if name == 'axelera.runtime' or name.startswith('axelera.runtime.'):
            raise ModuleNotFoundError("No module named '%s'" % name, name=name)
        return None


sys.meta_path.insert(0, _BlockRuntime())
sys.modules.pop('axelera.runtime', None)

import axelera

if hasattr(axelera, 'runtime'):
    del axelera.runtime

import axelera.app.device_manager  # noqa: F401

print('imported')
"""


def test_device_manager_imports_without_a_runtime():
    """
    A compile-only install - `axelera-devkit` without `axelera-rt` - has no
    `axelera.runtime`, and `deploy.py` reaches this module through
    `axelera.app.pipeline`. A module-level `from axelera import runtime` here
    therefore made the whole deploy flow unimportable on exactly the machine the
    install guide describes.

    Run in a subprocess: blocking a module cannot be undone cleanly enough to
    leave the rest of the suite alone.
    """
    result = subprocess.run(
        [sys.executable, '-c', _NO_RUNTIME_SCRIPT], capture_output=True, text=True
    )

    assert result.returncode == 0, result.stderr
    assert 'imported' in result.stdout


def test_create_device_manager_without_a_runtime_yields_the_null_manager(monkeypatch, caplog):
    """
    Deploying needs no device, so a missing runtime has to fall through to the
    null manager rather than propagate.

    `except OSError` did not catch this: an absent module raises ImportError,
    which skipped the fallback and failed the run instead.
    """
    import axelera

    monkeypatch.delattr(axelera, 'runtime', raising=False)
    monkeypatch.setitem(sys.modules, 'axelera.runtime', None)

    with caplog.at_level(logging.INFO):
        with device_manager.create_device_manager('gst') as dm:
            assert isinstance(dm, device_manager._NullDeviceManager)

    assert 'no runtime available' in caplog.text.lower()


def test_an_import_error_from_inside_the_runtime_is_not_swallowed():
    """
    A runtime that is installed but cannot import - a dependency of its own
    missing, a C extension failing to load - is a broken install, not a
    compile-only one.

    The fallback is scoped to `axelera.runtime` itself by name, because the
    `try` covers the whole constructor: without that check any ImportError
    raised while opening a device would be reported as "no runtime" and the
    deploy would quietly continue with no device.
    """
    broken = ModuleNotFoundError("No module named 'a_dependency'", name='a_dependency')

    with patch.object(device_manager, '_AipuDeviceManager', side_effect=broken):
        with pytest.raises(ModuleNotFoundError) as exc_info:
            device_manager.create_device_manager('gst')

    assert exc_info.value.name == 'a_dependency'

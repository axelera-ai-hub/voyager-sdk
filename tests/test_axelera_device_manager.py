# Copyright Axelera AI, 2024
from __future__ import annotations

import contextlib
import logging
import os
from unittest.mock import MagicMock, Mock, patch

import pytest

from axelera.app import config, device_manager


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


def test_no_devices_found_emulate():
    with mock_runtime([]):
        with device_manager.create_device_manager('gst', emulate='CPU') as dm:
            assert dm.get_metis_type() == config.Metis.none


def test_no_devices_found_override_metis():
    with mock_runtime([]):
        with device_manager.create_device_manager('gst', override_metis=config.Metis.pcie) as dm:
            assert dm.get_metis_type() == config.Metis.pcie


@pytest.mark.parametrize('board_type', ['ALPHA_PCIE', 'ALPHA_M2'])
def test_alpha_devices_found(board_type):
    with mock_runtime([board_type]):
        with pytest.raises(RuntimeError, match='Metis Alpha not supported'):
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
            assert dm.get_metis_type() == config.Metis.pcie
            names = [d.name for d in dm.devices]
    assert names == got


DEVICES = [[_make_device_info(devicen=n) for n in range(n)] for n in range(5)]


@pytest.mark.parametrize(
    'available, selector, expected, messages',
    [
        (DEVICES[1], '', ['metis-0:1:0'], 'Using device metis-0:1:0'),
        (DEVICES[1], 'metis-0:1:0', ['metis-0:1:0'], 'Selected device metis-0:1:0'),
        (
            DEVICES[2],
            'metis-0:1:0',
            ['metis-0:1:0'],
            'Selected device metis-0:1:0 (deselected 1 other device)',
        ),
        (
            DEVICES[3],
            'metis-0:1:0',
            ['metis-0:1:0'],
            'Selected device metis-0:1:0 (deselected 2 other devices)',
        ),
        (
            DEVICES[2],
            '1',
            ['metis-0:2:0'],
            'Selected device metis-0:2:0 (deselected 1 other device)',
        ),
        (
            DEVICES[2],
            '0,1',
            ['metis-0:1:0', 'metis-0:2:0'],
            'Selected devices metis-0:1:0, metis-0:2:0',
        ),
        (DEVICES[2], '', ['metis-0:1:0', 'metis-0:2:0'], 'Using devices metis-0:1:0, metis-0:2:0'),
    ],
)
def test_device_selector_messages(caplog, available, selector, expected, messages):
    caplog.set_level(logging.INFO)
    got = device_manager._select_devices(available, selector)
    assert [x.name for x in got] == expected
    assert [rec.message for rec in caplog.records] == messages.splitlines()


def test_torch_pipe_no_error():
    with mock_runtime([]):
        with device_manager.create_device_manager('torch') as dm:
            assert dm.get_metis_type() == config.Metis.none
            assert [] == dm.configure_boards_and_tracers(Mock(), [])


@pytest.mark.parametrize(
    'board_type, override, expected, warnings',
    [
        ('OMEGA_PCIE', config.Metis.none, config.Metis.pcie, ''),
        ('OMEGA_M2', config.Metis.none, config.Metis.m2, ''),
        ('OMEGA_DEVBOARD', config.Metis.none, config.Metis.pcie, 'Unknown board type'),
        ('OMEGA_SBC', config.Metis.none, config.Metis.pcie, ''),
        ('OMEGA_PCIE', config.Metis.pcie, config.Metis.pcie, ''),
        ('OMEGA_M2', config.Metis.pcie, config.Metis.pcie, ''),
        ('OMEGA_PCIE', config.Metis.m2, config.Metis.m2, ''),
        ('OMEGA_M2', config.Metis.m2, config.Metis.m2, ''),
    ],
)
def test_get_metis_type(caplog, board_type, override, expected, warnings):
    with mock_runtime([board_type]):
        dm = device_manager.create_device_manager('gst', override)
        assert dm.get_metis_type() == expected
    if warnings:
        assert warnings in caplog.text
    else:
        assert caplog.text == ''


@pytest.mark.parametrize(
    'board_type, expected',
    [
        ('', config.Metis.none),
        ('ALPHA_PCIE', config.Metis.none),
        ('OMEGA_PCIE', config.Metis.pcie),
        ('OMEGA_M2', config.Metis.m2),
        ('OMEGA_DEVBOARD', config.Metis.pcie),
        ('OMEGA_SBC', config.Metis.pcie),
    ],
)
def test_detect_metis_type(board_type, expected):
    with mock_runtime([board_type] if board_type else []):
        mt = device_manager.detect_metis_type()
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
    with mock_runtime(['OMEGA_PCIE'], read_device_configuration=cfg):
        with device_manager.create_device_manager('gst') as dm:
            got = dm.configure_boards_and_tracers(mock_nn, [mock_tracer])
    assert got == [mock_tracer]
    exp = {n: expected for n in range(ncores)}
    mock_tracer.initialize_models.assert_called_once_with(
        mock_nn.model_infos, config.Metis.pcie, exp
    )
    if warnings:
        assert warnings in caplog.text
    else:
        assert caplog.text == ''


@pytest.mark.parametrize(
    'in_env,in_clock,in_metis,in_mvm,clock,size,mvm',
    [
        (None, 600, 'OMEGA_PCIE', 88, 600, 0x80000000, 88),
        ('', 600, 'OMEGA_PCIE', 88, 600, 0x80000000, 88),
        ('', 600, 'OMEGA_SBC', 88, 600, 0x80000000, 88),
        ('0', 600, 'OMEGA_PCIE', 88, None, None, None),
        ('1', 600, 'OMEGA_PCIE', 88, 600, 0x80000000, 88),
        ('400', 600, 'OMEGA_PCIE', 88, 400, 0x80000000, 88),
        ('400,0x10000000', 600, 'OMEGA_PCIE', 88, 400, 0x10000000, 88),
        ('400', 600, 'OMEGA_PCIE', 88, 400, 0x80000000, 88),
        ('400', 600, 'OMEGA_SBC', 88, 400, 0x80000000, 88),
        ('400,0x10000000', 600, 'OMEGA_M2', 88, 400, 0x10000000, 88),
        ('400,,99', 600, 'OMEGA_M2', 88, 400, 0x40000000, 99),
        ('400,0x10000000,99', 600, 'OMEGA_M2', 88, 400, 0x10000000, 99),
    ],
)
def test_configure_board(in_env, in_clock, in_metis, in_mvm, clock, size, mvm):
    with contextlib.ExitStack() as c:
        newenv = {'AXELERA_CONFIGURE_BOARD': in_env} if in_env is not None else {}
        c.enter_context(patch.dict(os.environ, newenv))
        cfg = {f'clock_profile_core_{n}': in_clock for n in range(4)}
        Ctx = c.enter_context(mock_runtime([in_metis], read_device_configuration=cfg))
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
                ddr_size=size,
                mvm_utilisation_core_0=mvm,
            )
        else:
            ctx.configure_device.assert_not_called()


@pytest.mark.parametrize(
    'in_env, err',
    [
        ('spam,100', 'Invalid literal for int'),
        ('spam', 'Invalid literal for int'),
        ('100,spam', 'Invalid literal for int'),
        ('100,0xspam', 'Invalid literal for int'),
        ('100,0x10000000,spam', 'Invalid literal for int'),
        ('100,abcd', 'Invalid literal for int'),
        ('400,,abc', "Invalid literal"),
    ],
)
def test_configure_board_bad_input(in_env, err):
    with contextlib.ExitStack() as c:
        c.enter_context(patch.dict(os.environ, {'AXELERA_CONFIGURE_BOARD': in_env}))
        ctx = c.enter_context(mock_runtime(['OMEGA_PCIE']))
        nn = Mock()
        nn.tasks = [Mock()]
        nn.tasks[0].aipu_cores = 1
        with device_manager.create_device_manager('gst') as dm:
            with pytest.raises(
                ValueError, match=f'Badly formatted AXELERA_CONFIGURE_BOARD : {err}'
            ):
                dm.configure_boards_and_tracers(nn, [])
            ctx.configure_devices.assert_not_called()


@pytest.mark.parametrize(
    "key, value, expected",
    [
        ('ddr_size', 0x80000000, '2.0GB'),
        ('ddr_size', 0x60000000, '1.5GB'),
        ('ddr_size', 0x40000000, '1.0GB'),
        ('ddr_size', 0x20000000, '512.0MB'),
        ('ddr_size', 0x100000, '1.0MB'),
        ('ddr_size', 0x80000, '512.0KB'),
        ('ddr_size', 0x400, '1.0KB'),
        ('ddr_size', 0x200, '512B'),
        ('ddr_size', 0x1, '1B'),
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

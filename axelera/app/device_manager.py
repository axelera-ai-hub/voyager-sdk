# Copyright Axelera AI, 2024
# Utility functions used by the Voyager SDK
from __future__ import annotations

import abc
import os
import sys
import time
from typing import TYPE_CHECKING, Any

from . import config, logging_utils, network, utils

if TYPE_CHECKING:
    from axelera import runtime

    from . import inf_tracers, logging_utils, network, utils

    Tracers = list[inf_tracers.Tracer]

LOG = logging_utils.getLogger(__name__)


class DeviceManager(abc.ABC):
    """Handle aipu device configuration and own the device.

    Create a device manager using :func:`create_device_manager` and use it as a context manager
    or ensure you release the device using :meth:`release` when you are done with it.
    """

    @abc.abstractmethod
    def configure_boards_and_tracers(self, nn: network.AxNetwork, tracers: Tracers) -> Tracers:
        '''Configure the boards and tracers based on the model_infos board type'''

    @abc.abstractmethod
    def release(self) -> None:
        '''Release any resources held by the device manager'''

    @abc.abstractmethod
    def get_board_type(self) -> config.Board:
        '''Get the board type of the (first) device.'''

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.release()


class _NullDeviceManager(DeviceManager):
    def configure_boards_and_tracers(self, nn, tracers):
        return tracers

    def release(self) -> None:
        pass

    def get_board_type(self) -> config.Board:
        return config.Board.none

    @property
    def devices(self):
        return []


def _device_index(devices, sel):
    try:
        x = int(sel)
        if x < 0 or x >= len(devices):
            raise ValueError(f"Device selector {sel} out of range")
        return x
    except ValueError:
        matching = [idx for idx, d in enumerate(devices) if d.name == sel]
        if not matching:
            raise ValueError(f"No device found matching {sel}")
        return matching[0]


def _select_devices(available, selector):
    if selector:
        indexes = [_device_index(available, s) for s in selector.split(',')]
        devices = [available[i] for i in indexes]
        if not devices:
            raise RuntimeError(f"No devices found matching {selector}")
    else:
        devices = available
    verb = 'Selected' if selector else 'Using'
    names = ', '.join(d.name for d in devices)
    devices_s = 's' if len(devices) > 1 else ''
    deselected = len(available) - len(devices)
    deselected_s = '' if deselected == 1 else 's'
    deselected = f" (deselected {deselected} other device{deselected_s})" if deselected else ''
    log = LOG.info if deselected else LOG.debug
    log(f"{verb} device{devices_s} {names}{deselected}")
    return devices


def _human_readable(key, value):
    if key.startswith('mvm_utilisation_core_') and type(value) in (int, float):
        return f"{value}%"
    if key.startswith('clock_profile_core_') and type(value) is int:
        if value >= 1000:
            return f"{value/1000}GHz"
        return f"{value}MHz"
    return value


def _list_devices(ctx):
    if _list_devices._cache is None:
        _list_devices._cache = ctx.list_devices()
    return _list_devices._cache


_list_devices._cache = None


class _AipuDeviceManager(DeviceManager):
    def __init__(self, override_board=config.Board.none, device_selector: str = ''):
        # Imported here rather than at module scope so that a machine with no
        # runtime installed - a compile-only install, axelera-devkit without
        # axelera-rt - can still import this module and compile. Everything
        # below this point needs a device anyway.
        from axelera import runtime

        self.context = runtime.Context()
        if os.environ.get('AXELERA_ENABLE_AUTOMATIC_REPORTS') == '1':
            try:
                if not runtime.activate_error_reporter():
                    LOG.warning('Failed to activate automatic error reporting')
            except (AttributeError, OSError) as e:
                LOG.warning(f'Automatic error reporting is unavailable: {e}')
        try:
            devices = _list_devices(self.context)
            if not devices:
                raise RuntimeError("No devices found")
        except RuntimeError as e:
            if override_board == config.Board.none:
                raise
            LOG.info(f"Failed to detect device: {e}")
            self.devices = []
            self.board = override_board
        else:
            self.devices = _select_devices(devices, device_selector)
            b = _board_type_as_board(self.devices[0].board_type, self.devices[0].name)
            self.board = b if override_board == config.Board.none else override_board
        self._set_device_log_level()

    def _set_device_log_level(self) -> None:
        '''Quieten the device sysctrl log (see AXELERA_SLOG_LEVEL, default err).

        The firmware default is verbose, and shipping that log off the device costs enough
        host CPU to starve the host-side pipeline: the device keeps reporting full fps while
        end-to-end throughput collapses.
        '''
        if not (level := config.env.slog_level):
            return
        for device in self.devices:
            try:
                self.context.device_set_log_level(device, 'sysctrl', level)
            except Exception as e:
                # Broad: the runtime maps each result code onto its own exception type.
                LOG.warning(f"Could not set {device.name} sysctrl log level to {level}: {e}")

    def _configure_boards(self, nn) -> dict[int, int]:
        core_index = 0
        last = 0
        configures = {}
        if bool(self.devices) and _is_europa_device(self.devices[0]):
            # Europa clock and MVM configuration is handled by the firmware power limiter.
            # No per-core or per-model configuration is needed from the host side.
            return {}
        for task in nn.tasks:
            if not task.is_dl_task:
                continue
            cores = task.aipu_cores
            last = core_index + cores
            clock = nn.model_infos.clock_profile(task.model_info.name, self.board)
            mvm = nn.model_infos.mvm_limitation(task.model_info.name, self.board)
            configures.update(_get_configures(core_index, last, clock, mvm))
            core_index = last

        if configures:
            conf = ', '.join(f"{k}={_human_readable(k, v)}" for k, v in configures.items())
            LOG.debug("Reconfiguring devices with %s", conf)
            ready = [self.context.configure_device(d, **configures) for d in self.devices]
            if not all(ready):
                with utils.spinner():
                    while not all(self.context.device_ready(d) for d in self.devices):
                        time.sleep(0.3)
            # _get_configures passes device_firmware, so this may have reloaded the
            # firmware and reset the log level set at open.
            self._set_device_log_level()
        return _get_core_clocks(self.context, self.devices[0], 0, last)

    def configure_boards_and_tracers(self, nn, tracers):
        core_clocks = self._configure_boards(nn)
        new_tracers = []
        devices = [d.name for d in self.devices]
        for tracer in tracers:
            try:
                tracer.initialize_models(nn.model_infos, self.board, core_clocks, devices)
            except Exception as e:
                LOG.warning(f"{e}, skip trace")
            else:
                new_tracers.append(tracer)
        return new_tracers

    def get_board_type(self) -> config.Board:
        return self.board

    def release(self) -> None:
        self.context.release()


def create_device_manager(
    pipe_type: str,
    override_board: config.Board = config.Board.none,
    deploy_mode: config.DeployMode = config.DeployMode.QUANTCOMPILE,
    device_selector: str = '',
) -> DeviceManager:
    '''Create a device manager for the given pipe type.

    Use it as a context manager or ensure you release the device using :meth:`release` when you are
    done with it.

    `override_board` can be used to force the board type to a specific value.  This is used to
    override the detected type and only impacts limitations normally applied to core on M.2 to
    control power restrictions.

    `device_selector` can be used to select a specific device by name or index.  If empty, all
    available devices will be used.  Alternatively it can be a commma seperated list of device
    indexes or device names as returned by axdevice.
    '''
    if pipe_type not in ['torch', 'quantized'] and deploy_mode not in (
        config.DeployMode.QUANTIZE,
        config.DeployMode.QUANTIZE_DEBUG,
    ):
        try:
            return _AipuDeviceManager(override_board, device_selector)
        except ImportError as e:
            if e.name != 'axelera.runtime':
                # The runtime is there but could not import - a dependency of
                # its own is missing, or a C extension failed to load. That is
                # a broken install, not a compile-only one, and hiding it here
                # would turn it into a silent device-less compile.
                raise
            # No runtime installed, which is a compile-only install rather than
            # a broken one. Deployment does not need a device, so fall through
            # to the null manager the same way a missing device does.
            LOG.info(f"No runtime available, continuing without a device: {e}")
        except OSError as e:
            info_or_error = LOG.info if sys.platform == 'darwin' else LOG.error
            info_or_error(f"Failed to create device manager: {e}")

    return _NullDeviceManager()


def detect_board_type() -> config.Board:
    '''Quickly detect board type without creating a device manager.

    On failure returns config.Board.none so this can be used as a light touch
    detection.
    '''
    try:
        dm = create_device_manager('gst')
    except (OSError, RuntimeError) as e:
        LOG.debug(f"Failed to create device manager: {e}")
        return config.Board.none
    with dm:
        return dm.get_board_type()


def _board_type_as_board(board_type: runtime.BoardType, name: str) -> config.Board:
    if board_type.name == 'm2':
        return config.Board.m2
    if board_type.name.startswith('alpha'):
        raise RuntimeError(f"Failed to detect {name}")
    if board_type.name in ('europa_pcie', 'europa_devboard'):
        return config.Board.europa
    if board_type.name in ('pcie', 'sbc'):
        return config.Board.pcie  # Metis PCIe / SBC boards
    LOG.warning(f"Unknown board type {board_type.name} from device {name}, assuming pcie")
    return config.Board.pcie


def _is_europa_device(device: runtime.DeviceInfo) -> bool:
    hw_generation = getattr(device, 'hw_generation', None)
    if hw_generation is not None and getattr(hw_generation, 'name', '') == 'europa':
        return True
    return getattr(getattr(device, 'board_type', None), 'name', '').startswith('europa')


def _get_core_clocks(
    ctx: runtime.Context, d: runtime.DeviceInfo, first: int, last: int
) -> dict[int, int]:
    '''Get the current clock frequencies of the cores AIPU device in MHz'''
    cfg = ctx.read_device_configuration(d)
    clocks = {n: cfg.get(f'clock_profile_core_{n}', None) for n in range(first, last)}
    default_clock = (
        config.DEFAULT_CORE_CLOCK_EUROPA if _is_europa_device(d) else config.DEFAULT_CORE_CLOCK
    )
    default = {n: default_clock for n in range(first, last)}
    if any(v is None for v in clocks.values()):
        if global_clock := cfg.get('clock_profile', None):
            try:
                clock = int(global_clock)
            except ValueError:
                LOG.warning(
                    "Unparseable global clock_profile %r, assuming %s", global_clock, default
                )
                return default
            return {n: clock for n in range(first, last)}
        LOG.warning("No clock profile found in device configuration")
        return default
    try:
        return {n: int(v) for n, v in clocks.items()}
    except ValueError as e:
        LOG.warning(
            "Unparseable frequency from read_device_configuration: %s, assuming %s",
            e,
            default,
        )
        return default


def _get_configures(
    first_core: int, last_core: int, core_clock: int, mvm_limitation: int
) -> dict[str, Any]:
    '''Configure the AIPU board based on the given configuration and env overrides.'''
    val = config.env.configure_board
    if val == '0':
        LOG.debug("Skipping board configuration")
        return {}

    if val != '1':
        parts = val.split(',', 2)
        try:
            if clock := parts[0]:
                core_clock = int(clock)
            if mvm := len(parts) > 1 and parts[1]:
                mvm_limitation = int(mvm)
        except ValueError as e:
            msg = str(e).capitalize()  # for consistency, 'invalid literal...'
            raise ValueError(f"Badly formatted AXELERA_CONFIGURE_BOARD : {msg}") from None

    configures = {}
    if first_core == 0:
        configures['device_firmware'] = '1'
    for core in range(first_core, last_core):
        configures[f'mvm_utilisation_core_{core}'] = mvm_limitation
        configures[f'clock_profile_core_{core}'] = core_clock
    return configures

# Copyright Axelera AI, 2023
from __future__ import annotations

import argparse
import collections
import dataclasses
import difflib
import enum
import inspect
import logging
import os
from pathlib import Path
import re
import sys
import typing
from typing import Any, Callable, Dict, Optional, Tuple, TYPE_CHECKING

import numpy as np

from axelera import types

from . import environ, logging_utils, utils, yaml_parser

# Import TOML library (Python 3.11+ has tomllib built-in)
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        tomllib = None  # type: ignore

if TYPE_CHECKING:
    from axelera.compiler.config import CompilerConfig

LOG = logging_utils.getLogger(__name__)
MODEL_FOR_HELP_EXAMPLE = 'yolov8n-coco'


class DeployMode(enum.Enum):
    QUANTIZE = enum.auto()
    # this will be invoked automatically when inference is called with --pipe quantized
    QUANTIZE_DEBUG = enum.auto()
    QUANTCOMPILE = enum.auto()
    PREQUANTIZED = enum.auto()


class HWGeneration(enum.Enum):
    '''Hardware generation, independent of board form factor.'''

    none = enum.auto()
    metis = enum.auto()
    europa = enum.auto()


class Metis(enum.Enum):
    none = enum.auto()
    pcie = enum.auto()
    m2 = enum.auto()
    europa = enum.auto()

    @property
    def hw_generation(self) -> HWGeneration:
        if self == Metis.europa:
            return HWGeneration.europa
        if self == Metis.none:
            return HWGeneration.none
        return HWGeneration.metis


class HardwareEnable(enum.Enum):
    detect = enum.auto()
    enable = enum.auto()
    disable = enum.auto()

    def __bool__(self) -> bool:
        return self == HardwareEnable.enable


env = environ.env
'''Access environment variable configuration switches in a consistent way.'''

_DETECTABLE_CAPS = ('vaapi', 'opencl', 'opengl')
_DETECTABLE_CAPS_AVAILABLE_ARGS = collections.defaultdict(list)
_DETECTABLE_CAPS_AVAILABLE_ARGS['opengl'] = [env.opengl_backend]

DEFAULT_MAX_EXECUTION_CORES = 4
'''The default number of AI cores to execute on (Metis/Omega).'''

DEFAULT_MAX_EXECUTION_CORES_EUROPA = 8
'''The default number of AI cores to execute on for Europa devices.'''

DEFAULT_CORE_CLOCK = 800
'''The default core clock frequency in MHz (Metis/Omega).'''

DEFAULT_CORE_CLOCK_EUROPA = 1200
'''The default core clock frequency in MHz for Europa devices.'''

DEFAULT_WINDOW_SIZE = (900, 600)
'''The default window size for display windows.'''


class _HardwareEnableAction(argparse.Action):
    def __init__(
        self,
        option_strings,
        dest,
        default=None,
        type=None,
        choices=None,
        required=False,
        help=None,
        metavar=None,
    ):
        _option_strings = []
        for option_string in option_strings:
            _option_strings.append(option_string)
            if option_string.startswith('--enable-'):
                _option_strings.append('--disable-' + option_string[9:])
                _option_strings.append('--auto-' + option_string[9:])

        if help is not None and default is not None:
            help += f" (default: {default.name})"

        super().__init__(
            option_strings=_option_strings,
            dest=dest,
            nargs=0,
            default=default,
            type=type,
            choices=choices,
            required=required,
            help=help,
            metavar=metavar,
        )

    def __call__(self, parser, namespace, values, option_string=None):
        if option_string in self.option_strings:
            val = HardwareEnable.enable
            if option_string.startswith('--disable-'):
                val = HardwareEnable.disable
            elif option_string.startswith('--auto-'):
                val = HardwareEnable.detect
            setattr(namespace, self.dest, val)

    def format_usage(self):
        return ' | '.join(self.option_strings)


@dataclasses.dataclass
class HardwareCaps:
    vaapi: HardwareEnable = HardwareEnable.disable
    opencl: HardwareEnable = HardwareEnable.detect
    opengl: HardwareCaps = HardwareEnable.detect

    def detect_caps(self) -> HardwareCaps:
        '''Return a new HardwareCaps with any 'detect' value resolved.'''
        vals = [
            (n, getattr(self, n), getattr(utils, f'is_{n}_available')) for n in _DETECTABLE_CAPS
        ]
        conv = {True: HardwareEnable.enable, False: HardwareEnable.disable}
        new = {
            n: conv[detect(*_DETECTABLE_CAPS_AVAILABLE_ARGS[n])]
            for n, v, detect in vals
            if v == HardwareEnable.detect
        }
        return dataclasses.replace(self, **new)

    def as_argv(self) -> str:
        '''Convert the HardwareCaps to a string of command line arguments.'''
        values = [(n, getattr(self, n)) for n in _DETECTABLE_CAPS]
        defaults = HardwareCaps()
        conv = dict(detect='auto', disable='disable', enable='enable')
        enables = [f"--{conv[v.name]}-{n}" for n, v in values if v != getattr(defaults, n)]
        return ' '.join(enables)

    @classmethod
    def from_parsed_args(cls, args: argparse.Namespace) -> HardwareCaps:
        '''Construct HardwareCaps from the parsed command line arguments.'''
        return cls(
            args.enable_vaapi,
            args.enable_opencl,
            args.enable_opengl,
        )

    @classmethod
    def add_to_argparser(
        cls, parser: argparse.ArgumentParser, defaults: Optional[HardwareCaps] = None
    ) -> None:
        '''Add hardware caps arguments to the given argparse parser.'''
        defaults = defaults or cls.DETECT_ALL
        for cap in _DETECTABLE_CAPS:
            parser.add_argument(
                f'--enable-{cap}',
                dest=f'enable_{cap}',
                action=_HardwareEnableAction,
                default=getattr(defaults, cap),
                help=f'enable/disable/detect {cap} acceleration',
            )

    def enabled(self, cap: str) -> bool:
        '''Return True if the given cap is enabled.'''
        return getattr(self, cap) == HardwareEnable.enable


def add_aipu_cores(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--aipu-cores',
        type=int,
        choices=range(0, 9),
        default=8,
        help='number of AIPU cores to use; supported options are %(choices)s; '
        'by default uses all available cores on the device '
        '(Metis: 4, Europa: 8)',
    )


def range_check(min: int, max: int):
    def _range_check(value: str) -> int:
        value = int(value)
        if value < min or value > max:
            raise argparse.ArgumentTypeError(f"must be in range {min} to {max}")
        return value

    return _range_check


def add_compile_extras(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--num-cal-images',
        type=range_check(2, 2000),
        default=200,
        help='Specify the required number of images for model quantization. '
        'This value is rounded up to a multiple of the batch size if necessary. '
        'Minimum is 2 images, and the default is %(default)s.',
    )
    parser.add_argument(
        '--calibration-batch',
        type=int,
        default=1,
        help=argparse.SUPPRESS,  #'specify batch size for model quantization (default: %(default)s)',
    )
    parser.add_argument(
        '--cal-seed',
        type=int,
        default=None,
        help='Seed for torch.manual_seed; passing one enables shuffled calibration order so '
        'different seeds can be swept to study the impact on accuracy. '
        'If not set, calibration images are consumed in deterministic sorted order.',
    )
    parser.add_argument(
        '--default-representative-images',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Use the default representative images for model quantization. If not set, the user needs to provide the representative images or calibration data.',
    )
    parser.add_argument(
        '--dump-core-model',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Dump the "core model" that will really run on the AIPU for debugging.',
    )


HardwareCaps.ALL = HardwareCaps(
    HardwareEnable.enable, HardwareEnable.enable, HardwareEnable.enable
)
HardwareCaps.DEFAULT = HardwareCaps()
HardwareCaps.NONE = HardwareCaps(
    HardwareEnable.disable, HardwareEnable.disable, HardwareEnable.disable
)
HardwareCaps.DETECT_ALL = HardwareCaps(
    HardwareEnable.detect, HardwareEnable.detect, HardwareEnable.detect
)
HardwareCaps.OPENCL = HardwareCaps(
    HardwareEnable.disable, HardwareEnable.enable, HardwareEnable.enable
)


def resolve_toml_path(toml_ref: str, yaml_dir: Optional[Path] = None) -> Dict[str, Any]:
    """Resolve TOML configuration file path and determine its source.

    Supports three path types:
    1. Absolute paths: Used as-is (local file)
    2. Relative paths with directory: Relative to yaml_dir (local file)
    3. Filename only: Check yaml_dir first (local), then compiler configs (wheel)

    Args:
        toml_ref: TOML file reference (absolute path, relative path, or filename)
        yaml_dir: Directory of the YAML file (for relative path resolution)

    Returns:
        Dictionary with:
            - 'path': Resolved Path object
            - 'source': 'local' or 'compiler_wheel'
            - 'original_ref': Original reference string

    Raises:
        FileNotFoundError: If TOML file cannot be found
    """
    toml_path = Path(toml_ref)

    # Absolute path - use as-is (local file)
    if toml_path.is_absolute():
        if toml_path.exists():
            return {'path': toml_path, 'source': 'local', 'original_ref': toml_ref}
        raise FileNotFoundError(f"TOML config not found: {toml_path}")

    # Relative path with directory - relative to YAML dir (local file)
    if len(toml_path.parts) > 1:
        if yaml_dir is None:
            yaml_dir = Path.cwd()
        resolved = yaml_dir / toml_path
        if resolved.exists():
            return {'path': resolved, 'source': 'local', 'original_ref': toml_ref}
        raise FileNotFoundError(f"TOML config not found: {resolved}")

    # Filename only - check yaml_dir first (local takes precedence)
    if yaml_dir is not None:
        yaml_relative = yaml_dir / toml_ref
        if yaml_relative.exists():
            return {'path': yaml_relative, 'source': 'local', 'original_ref': toml_ref}

    # Then check compiler configs (from axelera.compiler.config package)
    try:
        import axelera.compiler.config as compiler_config_pkg

        compiler_models_dir = Path(compiler_config_pkg.__file__).parent / "models"
        compiler_path = compiler_models_dir / toml_ref
        if compiler_path.exists():
            return {
                'path': compiler_path,
                'source': 'compiler_wheel',
                'original_ref': toml_ref,
            }
    except ImportError:
        LOG.debug("axelera.compiler.config not available (compiler wheel not installed)")
        compiler_models_dir = None

    # Not found anywhere - collect available TOML files for suggestions
    if yaml_dir is None:
        yaml_dir = Path.cwd()
    yaml_relative = yaml_dir / toml_ref

    available_tomls = sorted({f.name for f in yaml_dir.glob('*.toml')})
    if compiler_models_dir is not None:
        available_tomls = sorted(
            set(available_tomls) | {f.name for f in compiler_models_dir.glob('*.toml')}
        )

    error_msg = f"TOML config '{toml_ref}' not found in:\n  - {yaml_relative} (local)"
    if compiler_models_dir is not None:
        error_msg += f"\n  - {compiler_models_dir / toml_ref} (compiler wheel)"
    else:
        error_msg += "\n  - compiler wheel (not installed)"

    if available_tomls:
        close_matches = difflib.get_close_matches(toml_ref, available_tomls, n=3, cutoff=0.4)
        if close_matches:
            error_msg += f"\n  Did you mean: {', '.join(close_matches)}?"
        else:
            error_msg += f"\n  Available configs: {', '.join(available_tomls)}"

    error_msg += "\n  To use default compiler settings, comment out the 'compiler_config_file' line in your YAML."

    raise FileNotFoundError(error_msg)


def load_toml_config(toml_path: Path) -> Dict[str, Any]:
    """Load TOML configuration file.

    Args:
        toml_path: Path to TOML file

    Returns:
        Dictionary of config key-value pairs

    Raises:
        RuntimeError: If TOML library not available
        ValueError: If TOML file has syntax errors
    """
    if tomllib is None:
        raise RuntimeError("TOML support not available. Please install tomli: pip install tomli")

    try:
        with open(toml_path, "rb") as f:
            config = tomllib.load(f)
        LOG.debug(f"Loaded TOML config: {toml_path}")
        return config
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"Invalid TOML syntax in {toml_path}: {e}") from e


def gen_compilation_config(
    deploy_cores, user_cfg, deploy_mode, yaml_dir: Optional[Path] = None
) -> Tuple["CompilerConfig", Dict[str, Any]]:
    """Generate the compilation configuration based on the user configuration.

    NOTE: CompilerConfig is a Pydantic model that will validate the configuration
    when it is created or when a field is set. If the configuration is invalid,
    a ValueError or ValidationError will be raised.

    Args:
        deploy_cores: Number of AIPU cores to deploy on
        user_cfg: User configuration dictionary from YAML extra_kwargs
        deploy_mode: Deployment mode (QUANTIZE, QUANTCOMPILE, PREQUANTIZED, etc.)
        yaml_dir: Directory of the YAML file (for TOML path resolution)

    Returns:
        Tuple of (CompilerConfig, metadata_dict) where metadata contains:
            - toml_info: TOML source information (if used)
            - yaml_overrides: List of fields overridden by YAML
            - toml_fields: List of fields set by TOML
    """

    try:
        from axelera.compiler.config import CompilerConfig, HostArch, HostOS, MulticoreMode
    except ImportError as e:
        raise RuntimeError(
            "axelera.compiler is required for model compilation. "
            "Please install the compiler wheel or set up the development environment. "
            f"Import error: {e}"
        ) from e

    # Deploy cores may be 0 for classical cv models. This is unsupported by the
    # CompilerConfig. This is a workaround since the CompilerConfig is discarded
    # ultimately for these models, and this avoids a larger refactor.
    deploy_cores = max(1, deploy_cores)

    # Initialize metadata tracking
    metadata = {'toml_info': None, 'yaml_overrides': [], 'toml_fields': []}

    # 1. Start with default CompilerConfig
    compiler_config = CompilerConfig(
        multicore_mode=MulticoreMode.BATCH,  # Default to batch mode
        aipu_cores_used=deploy_cores,
        resources_used=0.25 * deploy_cores,
        host_processes_used=1,  # Will disable compiler-internal resource validation
        host_arch=HostArch.auto_detect(),
        host_os=HostOS.auto_detect(),
        model_name="model",
    )

    # 2. Load TOML config if specified (middle priority)
    compiler_config_file = user_cfg.get('compiler_config_file')
    toml_settings = {}
    if compiler_config_file:
        try:
            toml_info = resolve_toml_path(compiler_config_file, yaml_dir)
            toml_path = toml_info['path']
            toml_settings = load_toml_config(toml_path)

            # Store TOML info for manifest
            metadata['toml_info'] = toml_info

            # Log TOML source
            source_desc = (
                "compiler wheel" if toml_info['source'] == 'compiler_wheel' else "local file"
            )
            LOG.info(f"Loading compiler config from TOML: {toml_path}")
            LOG.info(f"TOML source: {source_desc}")

            # Apply TOML settings, warn about unknown fields
            for key, value in toml_settings.items():
                if hasattr(compiler_config, key):
                    setattr(compiler_config, key, value)
                    metadata['toml_fields'].append(key)
                else:
                    LOG.warning(
                        f"Unknown compiler config field '{key}' in {toml_path}. "
                        f"This field will be ignored (forward compatibility)."
                    )
        except FileNotFoundError as e:
            LOG.warning(f"Compiler config file not found, using default settings.\n{e}")
        except Exception as e:
            LOG.error(f"Failed to load TOML config '{compiler_config_file}': {e}")
            raise

    # 3. Apply inline YAML overrides (highest priority)
    user_overrides = user_cfg.get('compilation_config', {})
    if user_overrides:
        if compiler_config_file:
            LOG.warning(
                "Both compiler_config_file and inline compilation_config are specified. "
                "Inline settings will override TOML settings."
            )

        # Track overrides and log them
        for key, value in user_overrides.items():
            old_value = getattr(compiler_config, key, None)
            setattr(compiler_config, key, value)
            metadata['yaml_overrides'].append(key)

            # Log overrides with before/after values
            if key in toml_settings:
                LOG.info(f"YAML override: {key}: {toml_settings[key]} (TOML) -> {value} (YAML)")
            else:
                LOG.info(f"YAML override: {key}: {old_value} (default) -> {value} (YAML)")

    # Log summary
    if metadata['toml_fields'] or metadata['yaml_overrides']:
        LOG.info(
            f"Compilation config sources: "
            f"TOML fields: {len(metadata['toml_fields'])}, "
            f"YAML overrides: {len(metadata['yaml_overrides'])}"
        )

    if deploy_mode == DeployMode.QUANTIZE:
        return compiler_config, metadata

    elif deploy_mode == DeployMode.QUANTIZE_DEBUG:
        if not compiler_config.quantization_debug:
            compiler_config.quantization_debug = True
        return compiler_config, metadata

    else:
        return compiler_config, metadata


def positive_int_for_argparse(value: str) -> int:
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError(f"cannot be negative: {value}")
    return ivalue


def _window_size(value: str) -> Tuple[int, int]:
    if value == 'fullscreen':
        from . import display  # lazy import to avoid circular import

        return display.FULL_SCREEN
    m = re.match(r'(\d+)(?:[,x](\d+))?', value)
    if not m:
        raise argparse.ArgumentTypeError(f"cannot parse {value} as window size")
    w = int(m.group(1))
    h = int(m.group(2)) if m.group(2) else (w * 10 // 16)
    return max(100, w), max(100, h)


def default_build_root() -> Path:
    return env.build_root


def default_data_root() -> Path:
    return env.data_root


def default_exported_root() -> Path:
    return env.exported_root


def add_nn_and_network_arguments(
    parser: _ExtendedHelpParser,
    network_yaml_info: yaml_parser.NetworkYamlInfo,
    default_network: str | None = None,
) -> None:
    # Format the list of available networks without breaking names across lines
    all_yaml_names = sorted(network_yaml_info.get_all_yaml_names())
    all_yaml_names = [x for x in all_yaml_names if not x.startswith(('ax-', 'mc-'))]

    all_valid_nets = '\n    '.join(all_yaml_names)
    extended_nnhelp = f"""The full set of available networks is:
    {all_valid_nets}
"""
    nnhelp = 'network to run, this can be a path to a pipeline file or the name of a network.'
    netopts = {'default': '', 'nargs': '?'} if default_network else {}
    parser.add_argument('network', help=nnhelp, extended_help=extended_nnhelp, **netopts)


def add_system_config_arguments(parser):
    parser.add_argument(
        '--build-root',
        default=default_build_root(),
        type=str,
        metavar='PATH',
        help='specify build directory',
    )
    parser.add_argument(
        '--data-root',
        type=str,
        default=default_data_root(),
        metavar='PATH',
        help='specify dataset download directory, or point to your existing dataset directory',
    )


def add_device_arguments_vision(parser):
    """Add device selection arguments for vision models (supports multiple devices)."""
    parser.add_argument(
        "-d",
        "--devices",
        type=str,
        default='',
        help="comma separated list of devices to run on. e.g. -d0,1,2. Default is all devices",
        extended_help="""Identifiers can be zero-based
index, e.g. -d0,1 or by name, e.g. -dmetis-0:1:0.  Use the tool axdevice to enumerate available devices.

$ axdevice
Device 0: metis-0:1:0 board_type=pcie fwver=v1.0.0-a6-15-g9d681b7bcfe9 clock=800MHz
Device 1: metis-0:3:0 board_type=pcie fwver=v1.0.0-a6-15-g9d681b7bcfe9 clock=800MHz
$ %(prog)s yolov5s-v7-coco media/traffic3_480p.mp4 -d0,1
$ %(prog)s yolov5s-v7-coco media/traffic3_480p.mp4 -dmetis-0:3:0

By default all devices available will be used.""",
    )


def add_device_arguments_llm(parser):
    """Add device selection arguments for LLM models (single device per process)."""
    parser.add_argument(
        "-d",
        "--devices",
        type=str,
        default='',
        help="device to run on. e.g. -d0 or -d1. Default is first available device",
        extended_help="""Identifiers can be zero-based
index, e.g. -d0 or by name, e.g. -dmetis-0:1:0.  Use the tool axdevice to enumerate available devices.

$ axdevice
Device 0: metis-0:1:0 board_type=pcie fwver=v1.0.0-a6-15-g9d681b7bcfe9 clock=800MHz
Device 1: metis-0:3:0 board_type=pcie fwver=v1.0.0-a6-15-g9d681b7bcfe9 clock=800MHz
$ %(prog)s llama-3-2-1b-1024-4core-static -d0
$ %(prog)s llama-3-2-1b-1024-4core-static -dmetis-0:3:0

Each LLM process uses all 4 cores of a single device. For multi-process usage,
process 1 gets device 0, process 2 gets device 1, etc.""",
    )


class _MetisAction(argparse.Action):
    AUTO = 'auto'
    CHOICES = (
        [AUTO]
        + [m.name for m in Metis]
        + [m.name.replace('_', '-') for m in Metis if '_' in m.name]
    )

    def __call__(self, parser, namespace, value, option_string=None):
        v = Metis.none if value == self.AUTO else Metis[value.lower().replace('-', '_')]
        setattr(namespace, self.dest, v)


def add_metis_arg(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        '--metis',
        default=Metis.none,
        action=_MetisAction,
        choices=_MetisAction.CHOICES,
        help='specify metis target for deployment (default: detect)',
    )


class _ExtendedHelpParser(argparse.ArgumentParser):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._hp = argparse.ArgumentParser(add_help=False)
        # prevent --help from matching --help-somethingelse due to prefix matching
        self._hp.add_argument('--help', action='store_true')
        self._hp.add_argument('--help-all', action='store_true', help=argparse.SUPPRESS)
        self._hp.set_defaults(extended_help='')
        self._all = {}

    def add_argument(self, *args, extended_help: str = '', **kwargs) -> None:
        """Add extra help text to the parser for a topic."""
        if extended_help:
            help = kwargs['help']
            longest = max(args, key=len)
            topic = longest.lstrip(self.prefix_chars)
            opt = f'--help-{topic}'
            self._all[opt] = extended = f"{longest}: {help}\n\n{extended_help}"
            self._hp.add_argument(opt, action='store_const', dest='extended_help', const=extended)
            kwargs['help'] = f"{help}\nFor more information see {opt}"
        super().add_argument(*args, **kwargs)
        if extended_help:
            self.add_argument(opt, action='store_true', help=f'show extended help for {topic}')

    def parse_args(self, args=None, namespace=None) -> argparse.Namespace:
        '''Check for any --help-something topics, and then proceed as normal.'''
        # prevent --help from matching --help-somethingelse due to prefix matching
        hpargs, _ = self._hp.parse_known_args(args, namespace)
        if hpargs.help_all:
            print(self.format_help())
            print()
            for k, v in self._all.items():
                print(f"$ ./{self.prog} {k}\n{v}" % {'prog': self.prog})
                print()
            raise SystemExit(0)

        if hpargs.extended_help:
            print(self.format_usage())
            print(hpargs.extended_help % {'prog': self.prog})
            raise SystemExit(0)
        return super().parse_args(args, namespace)


class _LlmArgumentParser(_ExtendedHelpParser):
    def __init__(
        self,
        network_yaml_info: yaml_parser.NetworkYamlInfo,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._network_yaml_info = network_yaml_info

    def parse_args(self, args=None, namespace=None) -> argparse.Namespace:
        ns = super().parse_args(args, namespace)
        _resolve_network(
            self,
            ns,
            self._network_yaml_info,
        )
        ns.build_root = Path(ns.build_root).expanduser().absolute()
        return ns


GENAI_SYSTEM_PROMPT = "Be concise. You are a chatbot for Axelera AI, a start-up building a state-of-the-art AI accelerator chip, called Metis. Answer user questions accurately and to the best of your ability. You run on the Metis chip."


def create_llm_argparser(
    network_yaml_info: yaml_parser.NetworkYamlInfo, **kwargs
) -> argparse.ArgumentParser:
    """Create an argument parser for LLM tasks."""
    parser = _LlmArgumentParser(network_yaml_info, **kwargs)
    add_nn_and_network_arguments(parser, network_yaml_info)
    add_system_config_arguments(parser)

    # --- Simplified CLI modes and prompt handling ---
    parser.add_argument(
        '--ui',
        nargs='?',
        const='share',
        default=None,
        choices=[None, 'local', 'share', 'local_simple', 'share_simple'],
        help='Enable Gradio web UI mode. Use --ui local or --ui share for classic UI, --ui local_simple or --ui share_simple for the simplified native UI. If not set, runs in CLI mode.',
    )
    parser.add_argument(
        '--prompt',
        type=str,
        nargs='?',
        default=None,
        help='Prompt for single-prompt CLI mode. If not provided, runs in interactive CLI mode.',
    )
    parser.add_argument(
        '--history',
        action=argparse.BooleanOptionalAction,
        default=True,
        help='Enable/disable conversation history (stateless mode). Each message will be processed independently if disabled (default: %(default)s)',
    )
    parser.add_argument(
        '--pipeline',
        type=str.lower,
        default='transformers-aipu',
        choices=['transformers', 'transformers-aipu'],
        help=argparse.SUPPRESS,  # we hide this for now as it requires a different version of pytorch
        # help='Specify pipeline backend:\n'
        # '  - transformers: Model runs on CPU/GPU based on the Hugging Face Transformers library\n'
        # '  - transformers-aipu: Model runs on Axelera AIPU with Transformer\'s tokenizer on the host (default)\n',
    )
    parser.add_argument(
        '--system-prompt',
        type=str,
        default=GENAI_SYSTEM_PROMPT,
        metavar='STR',
        help='Specify system prompt',
    )
    parser.add_argument(
        '--temperature', type=float, default=0, metavar='INT', help='Logits temperature'
    )
    parser.add_argument(
        '--show-stats',
        action='store_true',
        default=False,
        help='show performance statistics',
    )
    # Add CPU and temperature monitoring flags (implicitly enabled when --show-stats is used with transformers-aipu)
    parser.add_argument(
        '--show-cpu-usage',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Enable/disable CPU usage monitoring (default: on when --show-stats is used)',
    )
    parser.add_argument(
        '--show-temp',
        action=argparse.BooleanOptionalAction,
        default=False,
        help='Enable/disable temperature monitoring (default: on when --show-stats is used)',
    )
    parser.add_argument(
        '--rich-cli',
        action='store_true',
        default=False,
        help='Enable beautiful Rich-based CLI chat experience (colors, panels, markdown).',
    )
    parser.add_argument(
        '--tokenizer-dir',
        type=str,
        default=None,
        help='Path to a local directory containing tokenizer files (overrides tokenizer_url and HuggingFace)',
    )
    parser.add_argument(
        '--port',
        type=int,
        default=7860,
        help='Port to connect to the server (default: %(default)s)',
    )
    add_device_arguments_llm(parser)
    logging_utils.add_logging_args(parser)
    return parser


def add_display_arguments(
    parser: argparse.ArgumentParser,
):
    parser.add_argument(
        '--display',
        choices=['none', 'opengl', 'opencv', 'console', 'iterm2', 'auto'],
        default='auto',
        help='display the results of the inference in a window. The window can be opengl, opencv,\n'
        'console (using ANSI control codes) or none. If auto then if DISPLAY is set then OpenGL\n'
        'is preferred over OpenCV, and if DISPLAY is not set then a console display is used.'
        '(iterm2 is experimental, using the iTerm2 terminal and some other terminals to render images).\n',
    )
    parser.add_argument(
        '--no-display',
        dest='display',
        action='store_const',
        const='none',
        help='This is an alias for --display=none',
    )
    wsize = parser.add_mutually_exclusive_group()
    wsize.add_argument(
        '--window-size',
        type=_window_size,
        metavar='WxH | W | fullscreen',
        default=DEFAULT_WINDOW_SIZE,
        help=(
            'If --display sets the size of the window. Default is {}x{}.\n'
            'Size can be given as 800x600, just a width or fullscreen.'
        ).format(*DEFAULT_WINDOW_SIZE),
    )
    wsize.add_argument(
        '--fullscreen',
        action='store_const',
        const=_window_size('fullscreen'),
        dest='window_size',
        help='Alias for --window-size=fullscreen.',
    )
    parser.add_argument(
        '--multi-res-layout',
        action='store_true',
        default=False,
        help=(
            'Enable multi-resolution display layout with primary/secondary panel split, '
            'per-stream downscaling, stream ID labels, and interactive stream swapping '
            '(click a primary pane to select it, then click a secondary pane to swap). '
            'Disabled by default.'
        ),
    )


def create_inference_argparser(
    network_yaml_info: yaml_parser.NetworkYamlInfo | None = None,
    default_caps: HardwareCaps | None = None,
    default_network: str | None = None,
    default_show_stats=False,
    default_show_system_fps=True,
    default_show_inference_rate=False,
    default_show_device_fps=False,
    default_show_host_fps=False,
    default_show_cpu_usage=True,
    default_show_temp=True,
    default_show_latency=True,
    default_speedometer_smoothing=True,
    unsupported_yaml_cond: Optional[Callable[[Any], bool]] = None,
    unsupported_reason: str = '',
    port=None,
    **kwargs,
) -> argparse.ArgumentParser:
    network_yaml_info = network_yaml_info or yaml_parser.get_network_yaml_info()
    parser = _InferenceArgumentParser(
        network_yaml_info,
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f'\nExample: ./%(prog)s {MODEL_FOR_HELP_EXAMPLE} usb',
        unsupported_yaml_cond=unsupported_yaml_cond,
        unsupported_reason=unsupported_reason,
        default_network=default_network,
        **kwargs,
    )
    add_nn_and_network_arguments(parser, network_yaml_info, default_network=default_network)
    preproc_help = '\n'.join(f'  - {f.help()}' for f in _image_preproc_ops.values())
    source_help = f'''{_source_help()}

For fakevideo and video file source, you can specify the frame rate using @N where N
is the desired frame rate in frames per second. For video file, you can use '@auto' to
use the source's native frame rate.

For RTSP source, you can specify the username and password as normal like
`rtsp://id:pwd@10.40.130.221/stream0/media.amp?videocodec=jpeg&resolution=1280x960`.
Username and password can be encoded using the URI encoding protocol like P%%40ssword.

Sources can also be prefixed with one or more image preprocessing steps, separated by colons:
    rotate90:horizontalflip:input.mp4
'''
    parser.add_argument(
        'sources',
        default=[],
        nargs='*',
        help="source input(s); for example input.mp4, rotate90:usb, or rtsp:://host/path.",
        extended_help=f'''Each source can be one of the following:
{source_help}

Sources can also be prefixed with one or more image preprocessing steps, separated by colons:
The available preprocessing steps are:

{preproc_help}''',
    )
    add_system_config_arguments(parser)

    parser.add_argument(
        '--pipe',
        type=str.lower,
        default='gst',
        choices=['gst', 'torch', 'torch-aipu', 'quantized'],
        help='specify pipeline type:\n'
        '  - gst: C++ pipeline based on GStreamer (uses AIPU)\n'
        '  - torch-aipu: PyTorch pre/post-processing with model on AIPU\n'
        '  - torch: PyTorch pipeline in FP32 for accuracy baseline (uses ONNXRuntime for ONNX models)\n'
        '  - quantized: PyTorch pipeline using Axelera mixed-precision quantized model on host',
    )
    parser.add_argument(
        '--frames',
        type=positive_int_for_argparse,
        default=0,
        help='Specify number of frames to process (0 for all frames).\n'
        'When using multiple sources, this is the total number of '
        'frames to process across all sources combined.',
    ),
    add_display_arguments(parser)
    parser.add_argument('--save-output', default='', help=argparse.SUPPRESS)
    parser.add_argument(
        '-o',
        '--output',
        default='',
        dest='save_output',
        metavar='PATH',
        help='save inference results with annotations to an output video or directory of images',
        extended_help='''\
Single source:
• 'result.jpg' - saves exactly as result.jpg (single image only)
• 'output/' - saves as output/output_00000.jpg, output_00001.jpg, etc.
• 'img_%%03d.png' - saves as img_000.png, img_001.png, etc.
• 'result.mp4' - saves as video file

Multiple sources:
• 'stream_%%d.mp4' - saves as stream_0.mp4, stream_1.mp4, etc.
• 'out_%%d_img_%%03d.jpg' - saves as out_0_img_000.jpg, out_1_img_000.jpg, etc.

Stream indices (%%d) are auto-assigned: source1=0, source2=1, source3=2, etc.
Frame indices (%%03d, %%05d) increment for each frame within a stream.

Note: Use %%%% to escape %% in shell commands (single %% in config files).
''',
    )
    parser.add_argument(
        '--timeout',
        default=5,
        type=float,
        metavar="SEC",
        help="specify timeout to wait for next inference in seconds",
    )
    HardwareCaps.add_to_argparser(parser, default_caps)
    add_aipu_cores(parser)
    add_compile_extras(parser)
    parser.add_argument(
        '--enable-hardware-codec',
        action='store_true',
        default=False,
        help='enable hardware video codec in an optimized GST pipeline',
    )
    parser.add_argument(
        '--ax-precompiled-gst',
        type=str,
        help='''\
Load a precompiled GStreamer pipeline from a file. This is useful for debugging or
to use a precompiled pipeline without the need for compilation.  To create a
precompiled file you can use the --save-compiled-gst option.''',
    )
    parser.add_argument(
        '--save-compiled-gst',
        type=str,
        help='''\
Save the compiled GStreamer pipeline to a file. This is useful for debugging or
the file can also be used with `--ax-precompiled-gst` to load the pipeline
without compilation.

NOTE: If an rtsp source is used and the URL contains a username and password,
then the saved file will include the username and password in the saved file.''',
    )
    parser.add_argument(
        '--show-stats',
        action='store_true',
        default=default_show_stats,
        help='show performance statistics',
    )
    on_off = lambda x: 'on' if x else 'off'
    parser.add_argument(
        '--show-system-fps',
        action=argparse.BooleanOptionalAction,
        default=default_show_system_fps,
        help=f'show system FPS (default {on_off(default_show_system_fps)})',
    )
    parser.add_argument(
        '--show-inference-rate',
        action=argparse.BooleanOptionalAction,
        default=default_show_inference_rate,
        help=f'show inference rate (default {on_off(default_show_inference_rate)})',
    )
    parser.add_argument(
        '--show-device-fps',
        action=argparse.BooleanOptionalAction,
        default=default_show_device_fps,
        help=f'show device FPS (default {on_off(default_show_device_fps)})',
    )
    parser.add_argument(
        '--show-host-fps',
        action=argparse.BooleanOptionalAction,
        default=default_show_host_fps,
        help=f'show host FPS (default {on_off(default_show_host_fps)})',
    )
    parser.add_argument(
        '--show-temp',
        action=argparse.BooleanOptionalAction,
        default=default_show_temp,
        help=f'show AI Core temperatures (default {on_off(default_show_temp)})',
    )
    parser.add_argument(
        '--show-cpu-usage',
        action=argparse.BooleanOptionalAction,
        default=default_show_cpu_usage,
        help=f'show CPU usage (default {on_off(default_show_cpu_usage)})',
    )
    parser.add_argument(
        '--show-stream-timing',
        action=argparse.BooleanOptionalAction,
        default=False,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        '--show-latency',
        action=argparse.BooleanOptionalAction,
        default=default_show_latency,
        help=f'show stream latency (default {on_off(default_show_latency)})',
    )
    parser.add_argument(
        '--speedometer-smoothing',
        action=argparse.BooleanOptionalAction,
        default=default_speedometer_smoothing,
        help=f'''\
Enable realistic movement of the metric speedometers. Speedometers will start at 0 and
smoothly ramp up to the desired values with subtle wobbling at higher values, instead
of jumping immediately to the target value on each frame. Note that smoothed
speedometers may not reflect instantaneous metric changes - disable this option when
you need immediate visual feedback for debugging performance drops or demonstrating
real-time metric responses. (default {on_off(default_speedometer_smoothing)})
''',
    )
    parser.add_argument(
        '--rtsp-latency',
        default=500,
        type=int,
        metavar="MSEC",
        help="specify latency for rtsp input in milliseconds",
    )
    parser.add_argument(
        '--low-latency',
        default=False,
        action=argparse.BooleanOptionalAction,
        help='''enable optimisations for lower latency.

        Enabling this option causes the application framework and inference to favor low latency
        over throughput.

         * disables double buffering in Metis.
         * disables batch mode in Metis.
         * reduces the size of all queues in the pipeline to 1.
         * disables all render buffering.
        ''',
    )
    parser.add_argument(
        '--frame-rate',
        default=0,
        type=int,
        metavar="FPS",
        help="""\
for gst-pipe only. Specify the frame rate for all of the input streams. If the input source is
unable to provide the video at the specified frame rate, the pipeline will drop or duplicate frames
from the input source to produce a stream of frames at the specified frame rate. If the frame rate
is set to 0, the pipeline will use the frame rate of each individual input sources.
""",
    )
    add_device_arguments_vision(parser)

    parser.add_argument(
        "--tiled",
        default=0,
        type=int,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--tile-overlap",
        default=0,
        type=int,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--tile-position",
        default='none',
        type=str,
        choices=['none', 'left', 'right', 'bottom', 'top'],
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--show-tiles",
        action='store_true',
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        "--custom-tiles",
        default="",
        type=str,
        help=argparse.SUPPRESS,
    )

    parser.add_argument(
        '--cl-platform',
        type=str,
        choices=['auto', 'intel', 'nvidia', 'arm', 'gpu', 'cpu'],
        default='auto',
        help="specify OpenCL platform to use (default: auto)",
    )
    add_metis_arg(parser)

    if port is not None:
        parser.add_argument(
            '--port',
            default=port,
            help=f'port to connect to the server (default: {port})',
        )

    logging_utils.add_logging_args(parser)
    return parser


def _is_dataset(x: str) -> bool:
    return x == 'dataset' or x.startswith('dataset:') or x == 'server' or x.startswith('server:')


def _is_network(x: str, known_networks: list[str]) -> bool:
    return x in known_networks or (x.endswith('.yaml') and os.path.exists(x))


def _expand_source_path(x: str) -> str:
    if _is_dataset(x) or re.match(r'^\w+://', x):
        return x
    return str(Path(x).expanduser())


def _resolve_network(
    parser: argparse.ArgumentParser,
    ns: argparse.Namespace,
    network_yaml_info: yaml_parser.NetworkYamlInfo,
    unsupported_yaml_cond: Optional[Callable[[Any], bool]] = None,
    unsupported_reason: str = '',
    default_network: str | None = None,
) -> None:
    valid_nets = sorted(network_yaml_info.get_all_yaml_names())
    if not ns.network:
        if default_network:
            ns.network = default_network
        else:
            parser.error(
                f"The network argument is required, consider one of: {', '.join(valid_nets)}"
            )
    elif default_network and not _is_network(ns.network, valid_nets):
        ns.sources.insert(0, ns.network)
        ns.network = default_network

    try:
        yaml_info = network_yaml_info.get_info(ns.network)
    except KeyError as e:
        parser.error(str(e))
    if unsupported_yaml_cond and unsupported_yaml_cond(yaml_info):
        parser.error(
            f"Unsupported network '{ns.network}'{': ' + unsupported_reason if unsupported_reason else ''}"
        )
    ns.network = yaml_info.yaml_path


class _InferenceArgumentParser(_ExtendedHelpParser):
    '''A subclass of argparse.ArgumentParser that does some extra transforms
    after parsing inference arguments.
    '''

    def __init__(
        self,
        network_yaml_info: yaml_parser.NetworkYamlInfo,
        unsupported_yaml_cond: Optional[Callable[[Any], bool]] = None,
        unsupported_reason: str = '',
        default_network: str | None = None,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._network_yaml_info = network_yaml_info
        self._unsupported_yaml_cond = unsupported_yaml_cond
        self._unsupported_reason = unsupported_reason
        self._default_network = default_network

    def parse_args(self, args=None, namespace=None) -> argparse.Namespace:
        '''Parse the given args, and do some extra validation.'''
        ns = super().parse_args(args, namespace)
        if ns.pipe == 'torch' and ns.show_stats:
            LOG.info('Do not support torch pipeline for show-stats')
            ns.show_stats = False
        if ns.pipe == 'torch-aipu' and ns.aipu_cores > 1:
            LOG.info('torch-aipu pipeline supports aipu-cores=1 only')
            ns.aipu_cores = 1
        if ns.timeout:
            if ns.pipe == 'torch':
                ns.timeout *= 2
            elif ns.pipe == 'torch-aipu':
                ns.timeout *= 1.5

        _resolve_network(
            self,
            ns,
            self._network_yaml_info,
            self._unsupported_yaml_cond,
            self._unsupported_reason,
            self._default_network,
        )
        ns.sources = [_expand_source_path(s) for s in ns.sources]
        ns.data_root = Path(ns.data_root).expanduser().absolute()
        ns.build_root = Path(ns.build_root).expanduser().absolute()

        if not ns.sources:
            self.error('No source provided')

        if len(ns.sources) > 1 and ns.save_output and '%' not in ns.save_output:
            self.error(
                f'--save-output requires a pattern for multistream input e.g. "output_%02d.mp4" (got: {ns.save_output})'
            )
        if len(ns.sources) > 1 and any(_is_dataset(x) for x in ns.sources):
            self.error("Dataset sources cannot be used with multistream")
        if _is_dataset(ns.sources[0]) or ns.pipe in ('torch', 'torch-aipu', 'quantized'):
            try:
                import torch  # noqa: F401 just ensure torch is available

            except ImportError as e:
                if _is_dataset(ns.sources[0]) and ns.pipe.startswith('torch'):
                    msg = f'Dataset source and {ns.pipe} pipeline require torch to be installed'
                elif _is_dataset(ns.sources[0]):
                    msg = 'Dataset source requires torch to be installed'
                else:
                    msg = f'{ns.pipe} pipeline requires torch to be installed'
                self.error(f"{msg} : {e}")

        if ns.display == 'none':
            ns.display = False
            ns.enable_opengl = HardwareEnable.disable
        elif ns.display == 'opengl':
            ns.enable_opengl = HardwareEnable.enable
        elif not os.environ.get('DISPLAY') and ns.enable_opengl == HardwareEnable.detect:
            LOG.debug('DISPLAY not set, disabling OpenGL detection')
            ns.enable_opengl = HardwareEnable.disable

        _ignore_calibration_batch(ns)
        ns.show_latency = ns.show_latency or ns.show_stream_timing
        delattr(ns, 'show_stream_timing')
        return ns


class CompileAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        values = DeployMode[values]
        setattr(namespace, 'mode', values)

        if values in (DeployMode.QUANTIZE, DeployMode.QUANTIZE_DEBUG):
            if not namespace.model and not namespace.models_only:
                setattr(namespace, 'models_only', True)
            setattr(namespace, 'pipeline_only', False)


def create_deploy_argparser(
    network_yaml_info: yaml_parser.NetworkYamlInfo,
    **kwargs,
) -> argparse.ArgumentParser:
    parser = _DeployArgumentParser(
        network_yaml_info,
        description='Deploy a model to Axelera platforms',
        formatter_class=argparse.RawTextHelpFormatter,
        epilog=f'\nExample: ./%(prog)s {MODEL_FOR_HELP_EXAMPLE}',
        **kwargs,
    )
    add_nn_and_network_arguments(parser, network_yaml_info)
    add_system_config_arguments(parser)

    deploy_group = parser.add_mutually_exclusive_group()
    deploy_group.add_argument(
        '--model',
        type=str,
        default='',
        help='compile a specific model in the given network YAML without deploying pipeline',
    )
    deploy_group.add_argument(
        '--models-only',
        action='store_true',
        default=False,
        help='compile all models in the given network YAML without deploying pipeline',
    )
    deploy_group.add_argument(
        '--pipeline-only',
        action='store_true',
        default=False,
        help='compile pipeline in network YAML with pre-compiled models',
    )
    parser.add_argument(
        "--mode",
        type=str.upper,
        default=DeployMode.PREQUANTIZED,
        choices=DeployMode.__members__,
        action=CompileAction,
        help="Specify the model deployment mode:\n"
        " - QUANTIZE: Quantize the model. (Will NOT deploy pipeline)\n"
        " - QUANTCOMPILE: Quantize and compile the model.\n"
        " - PREQUANTIZED: Compile from a pre-quantized model (default).\n",
    )
    add_metis_arg(parser)
    parser.add_argument(
        '--pipe',
        type=str.lower,
        required=False,
        default='gst',
        choices=['gst', 'torch', 'torch-aipu'],
        help='specify pipeline type; gst is always with AIPU',
    )
    parser.add_argument(
        "--export",
        action='store_true',
        default=False,
        help="Export quantized/compiled model to exported/<model_name>.zip",
    )
    logging_utils.add_logging_args(parser)
    HardwareCaps.add_to_argparser(parser)
    add_aipu_cores(parser)
    add_compile_extras(parser)
    return parser


def _ignore_calibration_batch(ns: argparse.Namespace):
    if ns.calibration_batch != 1:
        LOG.warning('Please note --calibration-batch is not supported, a batch of 1 will be used')
        ns.calibration_batch = 1


class _DeployArgumentParser(_ExtendedHelpParser):
    '''A subclass of argparse.ArgumentParser that does some extra transforms
    after parsing deploy arguments.
    '''

    def __init__(
        self,
        network_yaml_info: yaml_parser.NetworkYamlInfo,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self._network_yaml_info = network_yaml_info

    def parse_args(self, args=None, namespace=None) -> argparse.Namespace:
        ns = super().parse_args(args, namespace)

        if ns.pipe == 'torch-aipu' and ns.aipu_cores > 1:
            LOG.info('torch-aipu pipeline supports aipu-cores=1 only')
            ns.aipu_cores = 1

        _resolve_network(self, ns, self._network_yaml_info)

        ns.data_root = Path(ns.data_root).expanduser().absolute()
        ns.build_root = Path(ns.build_root).expanduser().absolute()
        _ignore_calibration_batch(ns)
        return ns


######## Rendering Configuration ########
@dataclasses.dataclass
class TaskRenderConfig:
    """Settings for rendering metadata for a specific task.
    - show_annotations: Whether to draw visual elements like bounding boxes, keypoints, etc.
    - show_labels: Whether to draw class labels and score text.
    """

    show_annotations: bool = True
    show_labels: bool = True

    @classmethod
    def from_dict(cls, settings_dict: dict[str, Any]) -> TaskRenderConfig:
        """Create TaskRenderConfig from a dictionary with strict validation.

        Args:
            settings_dict: Dictionary containing render settings

        Returns:
            TaskRenderConfig instance

        Raises:
            TypeError: If any values are not the correct type
            ValueError: If dictionary contains unknown keys
        """
        field_info = {field.name: field.type for field in dataclasses.fields(cls)}

        unknown_keys = set(settings_dict.keys()) - set(field_info.keys())
        if unknown_keys:
            raise ValueError(f"Unknown keys in settings_dict: {unknown_keys}")

        return cls(**settings_dict)


DEFAULT_RENDER_CONFIG = TaskRenderConfig()


class RenderConfig:
    """Configuration for rendering metadata from different tasks."""

    def __init__(self, **kwargs):
        """Initialize render configuration with task-specific settings.

        Args:
            **kwargs: Keyword arguments where each key is a task name and
                     each value is a TaskRenderConfig object.

        Example:
            render_config = RenderConfig(
                detections=TaskRenderConfig(show_annotations=False, show_labels=False),
                tracker=TaskRenderConfig(show_annotations=True, show_labels=False),
            )
        """
        self._config: dict[str, TaskRenderConfig] = {}

        # Validate that all values are TaskRenderConfig instances
        for task_name, settings in kwargs.items():
            if not isinstance(settings, TaskRenderConfig):
                raise TypeError(
                    f"Value for task '{task_name}' must be a TaskRenderConfig instance, "
                    f"got {type(settings)}"
                )
            self._config[task_name] = settings

    @classmethod
    def from_tasks(cls, tasks: list[str] | None = None):
        """Initialize render configuration with a list of task names using default settings.

        Args:
            tasks: Optional list of task names to initialize with default settings

        Returns:
            RenderConfig instance with default settings for each task
        """
        instance = cls()
        if tasks:
            for task in tasks:
                instance.set_task(task)
        return instance

    def __repr__(self):
        """String representation of the render configuration."""
        return f"RenderConfig({self._config})"

    def set_task(
        self,
        task_name: str,
        show_annotations: bool = True,
        show_labels: bool = True,
        force_register: bool = False,
    ):
        """Set rendering configuration for a specific task.

        Args:
            task_name: Name of the task to configure
            show_annotations: Whether to draw visual elements like bounding boxes
            show_labels: Whether to draw class labels and score text
            force_register: If True, register the task if it doesn't exist.
                          If False, raise error for non-existent tasks.

        Returns:
            Self for method chaining

        Raises:
            KeyError: If task doesn't exist and force_register is False
        """
        if task_name not in self._config:
            if force_register:
                self._config[task_name] = TaskRenderConfig(show_annotations, show_labels)
            else:
                raise KeyError(
                    f"Task '{task_name}' not found in configuration. "
                    f"Use force_register=True to register new tasks or add it during initialization."
                )
        else:
            self._config[task_name].show_annotations = show_annotations
            self._config[task_name].show_labels = show_labels
        return self

    def get(self, task_name: str, default=None):
        """Get settings for a task or return default if not found.

        This mimics the dictionary get() method behavior.

        Args:
            task_name: Name of the task to retrieve settings for
            default: Value to return if task_name is not found

        Returns:
            The task's settings or the default value
        """
        return self._config.get(task_name, default)

    def __getitem__(self, task_name: str) -> TaskRenderConfig:
        """Get settings for a specific task."""
        if task_name not in self._config:
            self.set_task(task_name)
        return self._config[task_name]

    def keys(self):
        """Get all task names in the configuration."""
        return self._config.keys()


@dataclasses.dataclass
class BaseConfig:
    @classmethod
    def valid_kwargs(cls) -> set[str]:
        """Return the valid kwargs for this config."""
        return {f.name for f in dataclasses.fields(cls)}

    @classmethod
    def from_kwargs(cls, kwargs) -> BaseConfig:
        """Create config from kwargs.

        Used kwargs are removed from the dictionary.
        """
        x = cls()
        x.update_from_kwargs(kwargs)
        return x

    def update_from_kwargs(self, kwargs: dict[str, Any]) -> None:
        """Update config from kwargs.

        Used kwargs are removed from the dictionary.
        """
        for f in self.valid_kwargs():
            try:
                setattr(self, f, kwargs.pop(f))
            except KeyError:
                pass
        self.__post_init__()

    def __post_init__(self):
        """Post-initialization hook for additional setup."""
        # This can be overridden in subclasses for custom initialization logic
        pass


@dataclasses.dataclass
class LoggingConfig:
    console_level: int = logging.INFO
    """Logging level for the console output."""
    file_level: int = logging.NOTSET
    """Logging level for the file output, only relevant if `path` is set to a valid path."""
    path: str = ""
    """If set then log message of level `file_level` will be appended to `file_path`."""
    timestamp: bool = False
    """If True then the timestamp will be shown in the console and file output."""
    brief: bool = False
    """If True then suppress debug and warning logs, show others without decoration"""

    compiler_level: int = logging.WARNING
    """Logging level for the compiler output, this is set to WARNING by default,
    and `-v` or `-q` on the command line will always the level one level quieter
    than the console level.
    """

    @classmethod
    def from_parsed_args(cls, args: argparse.Namespace) -> LoggingConfig:
        '''Create a LoggingConfig from parsed arguments.'''
        from .logging_utils import get_config_from_args

        return get_config_from_args(args)


@dataclasses.dataclass
class SystemConfig(BaseConfig):
    """Configuration about the system."""

    data_root: Path | None = None
    build_root: Path | None = None
    hardware_caps: HardwareCaps = dataclasses.field(
        default_factory=lambda: HardwareCaps(
            HardwareEnable.detect,
            HardwareEnable.detect,
            HardwareEnable.detect,
        )
    )
    allow_hardware_codec: bool = False
    metis: Metis = Metis.none

    def __post_init__(self):
        self.data_root = (self.data_root or default_data_root()).resolve()
        self.build_root = self.build_root or default_build_root()

    @classmethod
    def from_parsed_args(cls, args: argparse.Namespace) -> SystemConfig:
        """Create a SystemConfig from parsed arguments.

        See stream.create_inference_stream() for an example of how to use this.
        """
        return cls(
            data_root=args.data_root,
            build_root=args.build_root,
            hardware_caps=HardwareCaps.from_parsed_args(args),
            allow_hardware_codec=getattr(args, 'enable_hardware_codec', False),
        )


@dataclasses.dataclass
class DeployConfig(BaseConfig):
    """Configuration about how models are deployed."""

    num_cal_images: int = 200
    '''Specify the required number of images for model quantization. This value is rounded up to
    a multiple of the batch size if necessary. Minimum is 2 images, and the default is 200.'''
    cal_seed: int | None = None
    '''Specify the seed for the torch.manual_seed which will affect the dataset shuffling.
    We use it to experiment with different seeds to see the impact on the accuracy.'''
    default_representative_images: bool = True
    '''Specify whether to use default representative images for model quantization.'''
    dump_core_model: bool = False
    '''Specify whether to dump the "core model" that will really run on the AIPU for debugging.'''

    def __post_init__(self):
        if self.cal_seed is not None:
            from axelera.app import torch_utils

            torch_utils.set_random_seed(self.cal_seed)

    @classmethod
    def from_parsed_args(cls, args: argparse.Namespace) -> DeployConfig:
        '''Create a DeployConfig from parsed arguments.'''
        return cls(
            num_cal_images=args.num_cal_images,
            cal_seed=args.cal_seed,
            default_representative_images=args.default_representative_images,
            dump_core_model=args.dump_core_model,
        )


@dataclasses.dataclass
class InferenceStreamConfig(BaseConfig):
    """Configuration that affects all pipelines."""

    timeout: int = 5
    """Timeout in seconds to wait for the next frame in the stream.

    This is used to control how long the stream will wait for a new frame
    before timing out. If set to 0, it will wait indefinitely for the next frame.
    """

    frames: int = 0
    """Number of frames to process in the stream.

    If set to 0, it will process all frames in the stream.
    When using multiple sources, this is the total number of frames to process.
    """

    @classmethod
    def from_parsed_args(cls, args: argparse.Namespace) -> InferenceStreamConfig:
        '''Create an InferenceStreamConfig from parsed arguments.'''
        return cls(timeout=args.timeout, frames=args.frames)


class SourceType(enum.Enum):
    USB = enum.auto()
    RTSP = enum.auto()
    HLS = enum.auto()
    VIDEO_FILE = enum.auto()
    IMAGE_FILES = enum.auto()
    FAKE_VIDEO = enum.auto()
    DATA_SOURCE = enum.auto()
    DATASET = enum.auto()


def _source_ctor(type):
    return classmethod(lambda cls, *args, **kwargs: cls(type, *args, **kwargs))


def _update_dataclass_from_args(datacls, *args, **kwargs):
    """Update a dataclass as if __init__ were called with (*args, **kwargs)."""
    P = inspect.Parameter

    parameters = []
    factory = object()
    for f in dataclasses.fields(datacls):
        if f.default is dataclasses.MISSING and f.default_factory is dataclasses.MISSING:
            default = P.empty
        elif f.default is not dataclasses.MISSING:
            default = f.default
        else:
            default = factory
        parameters.append(P(f.name, P.POSITIONAL_OR_KEYWORD, default=default))

    sig = inspect.Signature(parameters)
    bound = sig.bind(*args, **kwargs)
    bound.apply_defaults()
    for f in dataclasses.fields(datacls):
        v = bound.arguments.get(f.name, None)
        if v is factory:
            v = f.default_factory()
        setattr(datacls, f.name, v)


def _setattr_dataclass_from_args(datacls, **kwargs):
    sentinel = object()

    for f in dataclasses.fields(datacls):
        val = kwargs.pop(f.name, sentinel)
        if val is not sentinel:
            setattr(datacls, f.name, val)
    if kwargs:
        raise TypeError(f"Unknown arguments: {', '.join(kwargs.keys())}")


_source_types = []


def _source_match(type: SourceType, regex: str | None = None):
    """Register a function as a source type handler.

    If regex is provided then the func will only be called if the regex matches
    and the function will be passed the result of the match.

    If regex is not provided then the func will be called if all previous
    handlers failed to match, and it will be passed the source string.

    Once a handler is matched, the source type is set to the type of the handler
    and no more handlers are called. (Thus the order of handlers is important.)
    """

    def wrapper(func):
        def regex_wrapper(src: Source, s: str) -> bool:
            return (m := re.match(regex, s)) and func(src, m)

        f = regex_wrapper if regex else func

        def handler(src: Source, s: str) -> bool:
            if m := f(src, s):
                src.type = type
            return m

        _source_types.append((type, handler, func.__doc__))

    return wrapper


def _parse_stream_props(
    src: Source, s: str, default_width=1280, default_height=720, default_fps=30
):
    '''Parse stream properties from a string.'''
    m = re.match(r'(?::(\d+)x(\d+))?(?:@(\d+))?(?::?/([a-zA-Z0-9_/]+))?$', s)
    if not m:
        raise ValueError(f"Badly formatted stream properties: {s}")
    src.width = int(m.group(1)) if m.group(1) else default_width
    src.height = int(m.group(2)) if m.group(2) else default_height
    src.fps = int(m.group(3)) if m.group(3) else default_fps
    src.codec = m.group(4) or ''


@_source_match(SourceType.DATASET, r'dataset(?::(\w+))?(.*)$')
def _dataset(src: Source, m: re.Match) -> bool:
    '''Measure accuracy using the models dataset. Optional split (dataset:val)'''
    src.location = m.group(1) or 'val'
    return True


@_source_match(SourceType.USB, r'usb(?::(\d+)|:(/dev/\w+))?(.*)$')
def _usb(src: Source, m: re.Match) -> bool:
    '''USB camera (usb, usb:0, usb:/dev/video1)(:WxH)(@fps)(:codec)'''
    src.location = m.group(2) or m.group(1) or '0'
    _parse_stream_props(src, m.group(3) or '', 0, 0, 0)
    return True


@_source_match(SourceType.FAKE_VIDEO, r'fakevideo(?:(.*))?$')
def _fakevideo(src: Source, m: re.Match) -> bool:
    '''Test video source (fakevideo)(:WxH)(@fps)(:codec)'''
    _parse_stream_props(src, m.group(1))
    return True


@_source_match(SourceType.HLS, r'https?://.*')
def _hls(src: Source, m: re.Match):
    '''HLS source, e.g. (https://ipaddress/sourcename.m3u8)'''
    if not m.group(0).endswith('.m3u8'):
        LOG.warning(f"Unrecognised http/https format, assuming the format is HLS: {m.group(0)}")
    src.location = m.group(0)
    return True


@_source_match(SourceType.RTSP, r'rtsp://.*')
def _rtsp(src: Source, m: re.Match):
    '''RTSP source (rtsp://user:password@ipaddress:port/stream)'''
    src.location = m.group(0)
    return True


def _is_image(p: Path):
    return p.suffix.lower() in utils.IMAGE_EXTENSIONS


@_source_match(SourceType.VIDEO_FILE, r'(loop:)?(.*?)(?:@(\d+|auto))?$')
def _video_file(src: Source, m: re.Match) -> bool:
    '''Video file (filename.mp4)(@fps|@auto)'''
    path = Path(m.group(2))
    if m2 := (path.suffix.lower() in utils.VIDEO_EXTENSIONS):
        if not path.is_file():
            raise FileNotFoundError(f"No such file or directory: {m.group(0)}")
        src.location = str(path.expanduser())
        src.loop = bool(m.group(1))
        fps = m.group(3)
        src.fps = -1 if fps == 'auto' else int(fps or '0')
    return m2


@_source_match(SourceType.IMAGE_FILES, r'(loop:)?(.*?)(?:@(\d+))?$')
def _image_files(src: Source, m: re.Match) -> bool:
    '''Directory of images (loop:)(path/to/images)(@fps) (images located recursively)'''
    path = Path(m.group(2))
    if m2 := path.is_dir():
        src.location = str(path.expanduser())
        src.images = utils.list_images_recursive(path)
        if not src.images:
            raise RuntimeError(f"Failed to locate any images in {src.location}")
        src.loop = bool(m.group(1))
        fps = m.group(3)
        src.fps = -1 if fps == 'auto' else int(fps or '0')
    return m2


@_source_match(SourceType.IMAGE_FILES)
def _image_file(src: Source, s: str) -> bool:
    '''Single Image file (image.jpg)'''
    path = Path(s)
    if m := _is_image(path):
        if not path.is_file():
            raise FileNotFoundError(f"No such file or directory : {s}")
        src.location = str(path.expanduser().resolve())
        src.images = [path.expanduser().resolve()]
    return m


def _source_help():
    return '\n'.join(f'  - {h}' for _, _, h in _source_types if h)


class VideoFlipMethod(enum.Enum):
    """Enum for video flip methods."""

    none = 0
    '''Identity (no rotation)'''
    clockwise = enum.auto()
    '''Rotate clockwise 90 degrees'''
    rotate_180 = enum.auto()
    '''Rotate 180 degrees'''
    counterclockwise = enum.auto()
    '''Rotate counter-clockwise 90 degrees'''
    horizontal_flip = enum.auto()
    '''Flip horizontally'''
    vertical_flip = enum.auto()
    '''Flip vertically'''
    upper_left_diagonal = enum.auto()
    '''Flip across upper left/lower right diagonal'''
    upper_right_diagonal = enum.auto()
    '''Flip across upper right/lower left diagonal'''
    automatic = enum.auto()
    '''Select flip method based on image-orientation tag'''


_image_preproc_ops = {}


def _arg_from_str(param: str, s: str | Any, t: type | None) -> Any:
    sequences = (list, tuple)
    if t is None or not isinstance(s, str):
        return s  # no type hint, or not a string (came from a default arg)
    # unwrap typing.Optional[t]
    origin = typing.get_origin(t)
    if origin is typing.Union:
        args = [a for a in typing.get_args(t) if a is not type(None)]
        if len(args) == 1:
            t = args[0]
    s = s.strip()
    if t is str:
        return s
    elif t is bool:
        return s.lower() in ('true', '1', 'yes', 'on')
    elif t in sequences or (container := typing.get_origin(t)) in sequences:
        if t in sequences:
            container, converter = t, lambda n, x: x
        else:
            subt = typing.get_args(t)[0]  # todo mixed tuple like [int, float]
            converter = lambda n, x: _arg_from_str(param, x.strip(), subt)
        if not s.startswith('[') or not s.endswith(']'):
            raise ValueError(f"Expected a list in {param}, got: {s}")
        s = s.removeprefix('[').removesuffix(']')
        return container(converter(n, x.strip()) for n, x in enumerate(s.strip().split(',')))
    elif issubclass(t, enum.Enum):
        try:
            return t[s]
        except KeyError:
            raise ValueError(
                f"Invalid value '{s}' for enum {t.__name__} in param '{param}'"
            ) from None
    else:
        try:
            return t(s)
        except Exception as e:
            raise ValueError(f"Cannot convert {s} to {t.__name__} in param {param}: {e}") from None


def _image_preproc(func):
    """Decorator to match preprocessing steps for image processing functions."""

    sig = inspect.signature(func)

    def wrapper(args, kwargs):
        hints = typing.get_type_hints(func)
        # convert positional or kwargs to the param list
        try:
            bound = sig.bind(*args, **kwargs)  # may raise TypeError if args/kwargs do not match
            bound.apply_defaults()
        except TypeError as e:
            raise ValueError(f"Invalid arguments for {func.__name__}: {e}") from None
        # now we can do any type conversion from str if appropriate
        for n, s in list(bound.arguments.items()):
            bound.arguments[n] = _arg_from_str(n, s, hints.get(n, None))
        # now allow the function to do any final conversions
        return func(*bound.args, **bound.kwargs)

    def help():
        args = ', '.join(p.name for p in sig.parameters.values())
        args = f"[{args}]" if args else ''
        begin, end = f"{func.__name__}{args}:", func.__doc__
        return f"{begin}\n    {end}" if len(begin + end) > 80 else f"{begin} {end}"

    wrapper.help = help
    _image_preproc_ops[func.__name__] = wrapper
    return func


def _split_args(s: str) -> list[str]:
    '''Split a list of comma separated strings, handling embedded parameters.'''
    args = []
    depth = 0
    current_arg = ''
    for c in s:
        if c == ',' and depth == 0:
            args.append(current_arg.strip())
            current_arg = ''
        else:
            if c in '[{(':
                depth += 1
            elif c in ')]}':
                depth -= 1
            current_arg += c
    if current_arg:
        args.append(current_arg.strip())
    return args


def _parse_image_preprocs(s: str) -> tuple[str, list[ImagePreproc]]:
    """Parse image preprocessing steps from a string."""
    reop = re.compile(r'^(\w+)([^:]*):')
    rekwarg = re.compile(r'(\w+)\s*=\s*(.*)\s*')
    ops = []
    while s and (m := reop.match(s)):
        name, sargs = m.group(1), (m.group(2) or '')
        try:
            op = _image_preproc_ops[name]
        except KeyError:
            break  # let the input parser report the error
        args, kwargs = [], {}
        sargs = sargs.removeprefix('[').removesuffix(']')
        for sarg in _split_args(sargs):
            if kw := rekwarg.match(sarg):
                kwargs[kw.group(1)] = kw.group(2)
            else:
                args.append(sarg)
        ops.extend(op(args, kwargs))
        s = s[m.end() :]
    return s, ops


@_image_preproc
def videoflip(method: VideoFlipMethod = VideoFlipMethod.clockwise) -> list[ImagePreproc]:
    '''Rotate or transform the image using the specified method.'''
    return [ImagePreproc('videoflip', (method,), {})]


@_image_preproc
def rotate90() -> list[ImagePreproc]:
    '''Rotate the image by 90 degrees clockwise.'''
    return [ImagePreproc('videoflip', (VideoFlipMethod.clockwise,), {})]


@_image_preproc
def rotate180() -> list[ImagePreproc]:
    '''Rotate the image by 180 degrees clockwise.'''
    return [ImagePreproc('videoflip', (VideoFlipMethod.rotate_180,), {})]


@_image_preproc
def rotate270() -> list[ImagePreproc]:
    '''Rotate the image by 270 degrees clockwise.'''
    return [ImagePreproc('videoflip', (VideoFlipMethod.counterclockwise,), {})]


@_image_preproc
def verticalflip() -> list[ImagePreproc]:
    '''Flip the image vertically.'''
    return [ImagePreproc('videoflip', (VideoFlipMethod.vertical_flip,), {})]


@_image_preproc
def horizontalflip() -> list[ImagePreproc]:
    '''Flip the image horizontally.'''
    return [ImagePreproc('videoflip', (VideoFlipMethod.horizontal_flip,), {})]


@_image_preproc
def crop(left: int, top: int, width: int, height: int) -> list[ImagePreproc]:
    '''Crop the image to the given region.'''
    return [ImagePreproc('crop', (left, top, width, height), {})]


@_image_preproc
def perspective(camera_matrix: list[float], format: str = '') -> list[ImagePreproc]:
    '''Perform perspective transformation.'''
    return [ImagePreproc('perspective', (camera_matrix, format), {})]


@_image_preproc
def camera_undistort(
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    distort_coefs: list[float],
    normalized: bool = True,
    out_format: types.ColorFormat = types.ColorFormat.RGB,
) -> list[ImagePreproc]:
    '''Undistort the image using camera parameters and distortion coefficients.'''
    return [
        ImagePreproc(
            'camera_undistort', (fx, fy, cx, cy, distort_coefs, normalized, out_format), {}
        )
    ]


@_image_preproc
def polar(
    width: int,
    height: int,
    size: int,
    rotate180: bool,
    center_x: float,
    center_y: float,
    max_radius: int,
    inverse: bool,
    linear_polar: bool,
    format: types.ColorFormat = types.ColorFormat.RGB,
) -> list[ImagePreproc]:
    '''Apply polar transformation to the image.'''
    return [
        ImagePreproc(
            'polar',
            (
                width,
                height,
                size,
                rotate180,
                center_x,
                center_y,
                max_radius,
                inverse,
                linear_polar,
                format,
            ),
            {},
        )
    ]


@_image_preproc
def source_config(path: Path) -> list[ImagePreproc]:
    '''Read preproc operators from a file, the format is the same as the command line format.

    Operators may be colon separated on one line (identical to the command line) or for
    ease of reading they may also be on separate lines, e.g.:
        rotate90:perspective[[1.019,-0.697,412.602,0.918,1.361,-610.083,0.0,0.0,1.0]]
    OR
        rotate90
        perspective[[1.019,-0.697,412.602,0.918,1.361,-610.083,0.0,0.0,1.0]]
    '''
    cfg = ':'.join(x.strip(':') for x in path.read_text().splitlines() if x)
    if not cfg.endswith(':'):
        cfg += ':'
    ops: list[ImagePreproc] = []
    while cfg:
        cfg, new_ops = _parse_image_preprocs(cfg)
        if not new_ops:
            raise ValueError(f"Failed to parse preprocessing operators from {path}: {cfg}")
        ops.extend(new_ops)
    return ops


@_image_preproc
def tile(
    *,
    size: int = 0,
    overlap: int = 0,
    position: str = 'none',
    show: bool = False,
    file: str = '',
):
    '''Split input into overlapping tiles for inference (all keyword args are optional).
    size=N    - tile size in pixels; enables tiling (default: 0 = disabled)
    overlap=N - overlap between adjacent tiles in pixels (default: 0)
    show=true - draw tile boundaries in the output (default: false)
    file=PATH - load tile layout from a JSON file (cannot combine with size/overlap)
    E.g.: tile[size=640]:input.mp4  or  tile[size=640,overlap=32]:usb'''
    return [ImagePreproc.from_tile_config(TilingConfig(size, overlap, position, show, file))]


@dataclasses.dataclass
class ImagePreproc:
    name: str
    """The name of the preprocessing operator, e.g. 'videoflip', 'barrel', etc."""
    args: tuple[Any, ...] = tuple()
    """The arguments for the preprocessing operator."""
    kwargs: dict[str, Any] = dataclasses.field(default_factory=dict)
    """The keyword arguments for the preprocessing operator."""

    @property
    def is_tiling(self) -> bool:
        """Whether this preprocessing step is a tiling operation."""
        return self.name == '_add_tiles'

    @classmethod
    def from_tile_config(cls, cfg: TilingConfig) -> ImagePreproc:
        """Create an ImagePreproc from a TilingConfig."""
        return cls('_add_tiles', (cfg,), {})


Image = typing.Union[np.ndarray, types.Image]
ImageReader = typing.Generator[Image, None, None]


@dataclasses.dataclass(init=False)
class Source:
    type: SourceType
    """The type of the source, Source.USB, RTSP, VIDEO_FILE, etc."""
    location: str = ''
    """Path to file, usb device, or stream URL."""
    width: int = 0
    '''Used to attempt to select the input resolution, only relevant for USB and fakevideo.'''
    height: int = 0
    '''Used to attempt to select the input resolution, only relevant for USB and fakevideo.'''
    fps: int = 0
    '''Used to attempt to select the input frame rate, only relevant for USB and fakevideo.'''
    codec: str = ''
    '''Used to attempt to select the input codec, only relevant for USB.'''
    images: list[Path] | None = None
    '''For SourceType.IMAGE_FILES, the list of images to use.'''
    reader: Optional[ImageReader] = None
    '''For SourceType.DATA_SOURCE, a generator of numpy arrays or axelera.types.Image to use.'''
    preprocessing: list[ImagePreproc] | None = None
    '''Preprocessing steps to apply to the input.'''
    loop: bool = False
    '''For SourceType.VIDEO_FILE, whether to loop the video file.'''

    def __init__(self, source_or_type: str | Path | SourceType, *args, **kwargs):
        """Create a Source from a string, n existing Source, or from (keyword) arguments.

        Overload 1
            Construct from a string or Path, e.g. "usb:0", "video.mp4", "rtsp://...", etc.
            You can also provide kwargs to override the fields of the source, e.g.
                Source("usb:0", preprocessing=config.rotate90())
            Note that preprocessing constructors return a list of ImagePreproc, so you can add them
            together:
                Source("usb:0", preprocessing=config.rotate90()+config.horizontalflip())

            Note that kwargs take precedence over options specified in the string, e.g.
                Source("usb:0:640x480@30", fps=15)  # will use fps=15, not 30

        Overload 2
            Construct from an existing Source, e.g. Source(src). This is used to ensure all
            sources are Source objects. Additionally keyword arguments can be passed to override
            the fields of the existing source.

        Overload 3
            Construct from all the parts of a Source. Normally this is not used directly, but
            instead from one of the class method constructors, e.g. Source.USB('/dev/video1')
        """
        takes = 'Source() takes 1 string argument, or a Source or SourceType'
        if isinstance(source_or_type, Source):
            if args:
                raise TypeError(f"{takes} and other keyword arguments to override the fields")
            try:
                all_kwargs = dict(dataclasses.asdict(source_or_type), **kwargs)
            # dataclasses.asdict will fail with generators as they are not pickleable, so fall back here
            except TypeError:
                all_kwargs = {}
                for field in dataclasses.fields(source_or_type):
                    all_kwargs[field.name] = getattr(source_or_type, field.name)
                all_kwargs.update(kwargs)
            _update_dataclass_from_args(self, **all_kwargs)
            return self.__post_init__()
        if isinstance(source_or_type, Path):
            source_or_type = str(source_or_type)
        if isinstance(source_or_type, SourceType):
            _update_dataclass_from_args(self, source_or_type, *args, **kwargs)
            return self.__post_init__()
        if isinstance(source_or_type, collections.abc.Iterator):
            _update_dataclass_from_args(
                self, SourceType.DATA_SOURCE, *args, **kwargs, reader=source_or_type
            )
            return self.__post_init__()
        if not isinstance(source_or_type, str):
            raise TypeError(
                f"{takes}, but you cannot use args and kwargs arguments for the fields)"
            )
        if args:
            raise TypeError("When using a string parameter you must use kwargs, not args")
        super().__init__()
        source, self.preprocessing = _parse_image_preprocs(source_or_type)
        if len([op for op in self.preprocessing if op.is_tiling]) > 1:
            raise ValueError("Only one tile preproc is allowed per source")
        for type, handler, _ in _source_types:
            if handler(self, source):
                _setattr_dataclass_from_args(self, **kwargs)
                return self.__post_init__()
        helps = _source_help()
        raise ValueError(f"Unrecognized source: {source}. Valid formats are:\n{helps}")

    @property
    def has_tiling(self) -> bool:
        """Whether this source has tiling enabled in its preprocessing steps."""
        return any(p.is_tiling for p in self.preprocessing or [])

    @property
    def has_image_preproc(self) -> bool:
        """Whether this source has any non-tiling ops in its preprocessing steps."""
        return any(not p.is_tiling for p in self.preprocessing or [])

    def __post_init__(self):
        self.images = self.images or []
        self.preprocessing = self.preprocessing or []

    def __str__(self):
        """Return a string representation of the source suitable for display."""
        if self.type == SourceType.VIDEO_FILE:
            return os.path.relpath(self.location)
        return self.location

    USB = _source_ctor(SourceType.USB)
    HLS = _source_ctor(SourceType.HLS)
    RTSP = _source_ctor(SourceType.RTSP)
    FAKE_VIDEO = _source_ctor(SourceType.FAKE_VIDEO)
    VIDEO_FILE = _source_ctor(SourceType.VIDEO_FILE)
    IMAGE_FILES = _source_ctor(SourceType.IMAGE_FILES)
    DATASET = _source_ctor(SourceType.DATASET)


@dataclasses.dataclass
class TilingConfig:
    """Configuration for tiled inference."""

    size: int = 0
    """Size of the tile in pixels. If 0, tiled inference is disabled."""
    overlap: int = 0
    """Overlap between tiles in pixels. If 0, no overlap is used."""
    position: str = 'none'
    """Position of the tile, one of 'none', 'left', 'right', 'bottom', 'top'."""
    show: bool = False
    """Whether to show the tiles in the output."""
    file: str = ''
    """If set, load tile configuration from the specified file."""

    def __post_init__(self):
        if self.file and any((self.size, self.overlap, self.position != 'none')):
            raise ValueError("Cannot specify both file and other parameters for tiling config")
        if self.position not in ('none', 'left', 'right', 'bottom', 'top'):
            raise ValueError(f"Invalid tile position: {self.position}")

    def __bool__(self):
        """Return True if tiled inference is enabled."""
        return self.size > 0 or bool(self.file)

    @classmethod
    def from_parsed_args(cls, args: argparse.Namespace) -> TilingConfig:
        """Create a TilingConfig from parsed arguments."""
        return cls(
            args.tiled, args.tile_overlap, args.tile_position, args.show_tiles, args.custom_tiles
        )


@dataclasses.dataclass
class PipelineConfig(BaseConfig):
    """Configuration for an individual pipeline."""

    network: str = dataclasses.field(default='')
    sources: list[Source] = dataclasses.field(default_factory=list)
    pipe_type: str = dataclasses.field(default='gst')
    device_selector: str = ''
    aipu_cores: int = 1
    ax_precompiled_gst: str | Path = ''
    save_compiled_gst: Path | None = None
    specified_frame_rate: int = 0
    low_latency: bool = False
    """If True then use low latency settings for the pipeline."""
    rtsp_latency: int = 500
    save_output: str = ''
    tiling: TilingConfig = dataclasses.field(default_factory=TilingConfig)
    """Legacy tiling configuration, will be applied to all sources that do not have tiling configured."""
    which_cl: str = 'auto'
    """Configuration for tiled inference."""
    render_config: RenderConfig | None = None
    """Initial render configuration for the pipeline."""

    @property
    def eval_mode(self) -> bool:
        return self.sources and self.sources[0].type == SourceType.DATASET

    def __post_init__(self):
        if self.sources is not None and not isinstance(self.sources, list):
            self.sources = [self.sources]
        if not self.sources:
            # we allow no sources for the default pipeline config
            return  # raise ValueError("No sources provided")

        self.sources = [s if isinstance(s, Source) else Source(s) for s in self.sources]
        single_source_types = (SourceType.DATASET,)
        if len(self.sources) > 1 and any(s.type in single_source_types for s in self.sources):
            which = [s.type for s in self.sources if s.type in single_source_types][0]
            raise ValueError(
                f"{which.name.title().replace('_', ' ')} source cannot be used with multiple sources"
            )
        if self.tiling:
            for s in self.sources:
                if not s.has_tiling:
                    s.preprocessing.append(ImagePreproc.from_tile_config(self.tiling))
            # clear legacy tiling config as it's now applied to the sources
            self.tiling = TilingConfig()

        if self.sources[0].type == SourceType.DATASET:
            LOG.info("Using dataset %s", self.sources[0].location)

    @classmethod
    def from_parsed_args(cls, args: argparse.Namespace) -> PipelineConfig:
        """Create a PipelineConfig from parsed arguments.

        See stream.create_inference_stream() for an example of how to use this.
        """
        return cls(
            network=args.network,
            sources=args.sources,
            pipe_type=args.pipe,
            device_selector=args.devices,
            aipu_cores=args.aipu_cores,
            ax_precompiled_gst=Path(args.ax_precompiled_gst) if args.ax_precompiled_gst else '',
            save_compiled_gst=Path(args.save_compiled_gst) if args.save_compiled_gst else None,
            specified_frame_rate=args.frame_rate,
            rtsp_latency=args.rtsp_latency,
            low_latency=args.low_latency,
            save_output=args.save_output,
            tiling=TilingConfig.from_parsed_args(args),
            which_cl=args.cl_platform,
        )

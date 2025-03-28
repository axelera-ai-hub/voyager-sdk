# Copyright Axelera AI, 2025
# Logging functions

import argparse
from dataclasses import dataclass
import logging
import re
import sys
import time
import traceback
from typing import Union
import warnings

import tqdm

_logging_configured = False
TRACE = 5
"""A logging level more verbose than debug."""

NOTSET = logging.NOTSET
"""The unset/undefined logging level (0)."""

DEBUG = logging.DEBUG
"""Debug-level messages (10)."""

INFO = logging.INFO
"""Informational messages (20)."""

LEVEL_NAMES = [
    "TRACE",
    "DEBUG",
    "INFO",
    "WARNING",
    "ERROR",
    "CRITICAL",
    "FATAL",
]

_BRGREEN = "\x1b[01;32m"
_BOLD_WHITE = "\x1b[1m"
_BOLD_WHITE_UNDERLINE = "\x1b[01;21m"
_WHITE_W_GREEN_BG = "\x1b[42;20m"
_YELLOW = "\x1b[33;20m"
_RED = "\x1b[31;20m"
_BOLD_RED = "\x1b[31;1m"
_ITALIC = "\x1b[3m"
_REVERSE = "\x1b[7m"
_RESET = "\x1b[0m"


_FORMATS = {
    TRACE: [_WHITE_W_GREEN_BG, _RESET],
    logging.DEBUG: [_BRGREEN, _RESET],
    logging.INFO: ["", ""],
    logging.WARNING: [_YELLOW, _RESET],
    logging.ERROR: [_RED, _RESET],
    logging.CRITICAL: [_BOLD_RED, _RESET],
    logging.FATAL: [_BOLD_RED, _RESET],
}

_RE_EMPHASIS = re.compile(r"(?<!\\)\*\*(.*?)\*\*")
_RE_HEADING = re.compile(r"^##+ (.*)")


class UserError(RuntimeError):
    '''An error that should be reported to the user without traceback or log formatting.'''

    def format(self):
        if sys.stdin.isatty():
            pre, post = _BOLD_RED, _RESET
        else:
            pre, post = '', ''
        return f'\n{pre}ERROR{post}: {self}'


@dataclass
class Config:
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


class _TqdmStream:
    @classmethod
    def write(_, msg):
        tqdm.tqdm.write(msg, end="")


class LogWithTrace(logging.Logger):
    def trace(self, msg, *args, **kwargs):
        if self.isEnabledFor(TRACE):
            self._log(TRACE, msg, args, **kwargs)

    def trace_exception(self):
        """Send a traceback of the current exception to the logger at ERROR level,
        but only if DEBUG is enabled."""
        if self.isEnabledFor(DEBUG):
            self.error(traceback.format_exc())

    def report_recoverable_exception(self, e: Exception):
        """Send a traceback of the current exception to the logger at ERROR level,
        if TRACE is enabled then a traceback is included."""
        if self.isEnabledFor(TRACE):
            self.error(traceback.format_exc())
        else:
            self.error(f'Exception: {e}')

    def exit_with_error_log(self, e: Union[str, Exception] = None):
        """Exit the program with an optional error message."""
        if e is not None:
            self.error(e)
        self.trace_exception()
        sys.exit(1)


def getLogger(name: str) -> LogWithTrace:
    """To be used internally by the axelera package. Use logging.getLogger instead."""
    oldClass = logging.getLoggerClass()
    # redefine LogWithTrace but with the current logger class as base
    cls = type("LogWithTrace", (oldClass,), dict(LogWithTrace.__dict__))
    logging.addLevelName(TRACE, "TRACE")
    try:
        logging.setLoggerClass(cls)
        return logging.getLogger(name)
    finally:
        logging.setLoggerClass(oldClass)


def _level_to_int(loglevel: str):
    log_level = getattr(logging, loglevel.upper(), globals().get(loglevel.upper(), None))
    if log_level is None:
        valid = ", ".join(LEVEL_NAMES)
        raise ValueError(f"Unknown logging level. {valid} are valid but not {loglevel}")
    return log_level


def _markdown(s):
    """Ridiculously basic markdown support. Only **bold** and ## Heading are supported."""
    pre, post, reset = "", "", _RESET
    if m := _RE_HEADING.match(s):
        s = m.group(1)
        pre, post = _BOLD_WHITE_UNDERLINE, _RESET
        reset = _RESET + _BOLD_WHITE_UNDERLINE
    return pre + _RE_EMPHASIS.sub(f"{_ITALIC}\\1{reset}", s) + post


class _Formatter(logging.Formatter):
    def __init__(self, show_timestamp, show_module=True, ansi_colors=True, brief=False):
        super().__init__()
        self._show_timestamp = show_timestamp
        self._show_module = show_module
        self._ansi_colors = ansi_colors and not brief
        self._brief = brief

    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        if self._brief:
            return "\n".join(msg.splitlines())
        msg = _markdown(msg) if self._ansi_colors else msg
        levelname = self._format_category(record)
        time_string = self._format_time(record) if self._show_timestamp else ""
        hide = not (self._show_module and record.name not in ("root", "__main__"))
        name = "" if hide else f"{record.name}:"
        return "\n".join(f"{time_string}{levelname}:{name} {l}" for l in msg.splitlines())

    def _format_category(self, record):
        if self._ansi_colors:
            pre, post = _FORMATS.get(record.levelno, ("", ""))
            return f"{pre}{record.levelname}{post}" + " " * (8 - len(record.levelname))
        else:
            return record.levelname.ljust(8)

    def _format_time(self, record):
        asctime = time.strftime("%H:%M:%S", time.localtime(record.created))
        return f"[{asctime}.{int(record.msecs):03}] "


def add_logging_args(parser: argparse.ArgumentParser):
    """Add logging arguments to an argparse parser suitable for logging from
    the axelera package.

    Usage example:

        from axelera.app import logging_utils
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument(...)  # any user arguments
        logging_utils.add_logging_args(parser)

        args = parser.parse_args()
        logging_utils.configure_logging(logging_utils.get_config_from_args(args))

    """
    valid = ", ".join(LEVEL_NAMES)
    parser.add_argument(
        "--loglevel",
        help=f"set logging level for --logfile; one of {valid}\n(default: %(default)s)",
        default="",
        choices=LEVEL_NAMES,
        type=str.upper,
        metavar="LEVEL",
    )
    parser.add_argument(
        "--logfile", default="", metavar="PATH", help="log output to specified file"
    )
    parser.add_argument(
        "--logtimestamp", action="store_true", help="show timestamp in log messages"
    )
    parser.add_argument(
        "--brief-logging",
        dest='brief',
        action="store_true",
        help="suppress debug and warning logs; show others without decoration",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "-q",
        "--quiet",
        default=0,
        action="count",
        help="be less verbose; use repeatedly for less info",
    )
    group.add_argument(
        "-v",
        "--verbose",
        default=0,
        action="count",
        help="be more verbose; use repeatedly for more info",
    )


def _args_to_level(args: argparse.Namespace, quiet_offset: int) -> int:
    levels = {
        quiet_offset - 2: logging.ERROR,
        quiet_offset - 1: logging.WARNING,
        quiet_offset + 0: logging.INFO,
        quiet_offset + 1: logging.DEBUG,
        quiet_offset + 2: TRACE,
    }
    if args.quiet:
        return levels.get(-args.quiet, logging.ERROR)
    else:
        return levels.get(args.verbose, TRACE)


def get_config_from_args(args: argparse.Namespace) -> Config:
    """Extract logging configuration from argparse arguments.

    See add_logging_args for a usage example.
    """
    console_level = _args_to_level(args, 0)
    file_level = logging.NOTSET
    if args.logfile:
        file_level = console_level if args.loglevel == "" else _level_to_int(args.loglevel)
    return Config(console_level, file_level, args.logfile, args.logtimestamp, args.brief)


def configure_compiler_level(args: argparse.Namespace):
    compiler_level = _args_to_level(args, 1)
    compiler_modules = '''\
        te_compiler
        onnx2torch
        python_jsonschema_objects
        axelera.aipu
        axelera.compiler
        axelera.miraculix
        axelera.miraculix_plugins
    '''.split()
    for m in compiler_modules:
        getLogger(m).setLevel(compiler_level)
    warnings.filterwarnings("ignore", module='pydantic', message="Pydantic serializer warnings:")


class _BriefFilter(logging.Filter):
    def filter(self, record):
        return record.levelno not in [logging.DEBUG, logging.WARNING, TRACE]


def configure_logging(config: Config = Config()):
    """Configure axelera logging and formatters.

    See add_logging_args for a usage example with argparse, or use the Config
    class to configure logging explicitly.
    """
    logger = getLogger("")

    global _logging_configured
    if _logging_configured:
        for handler in logger.handlers[:]:
            logger.removeHandler(handler)

    formatter = _Formatter(config.timestamp, True, ansi_colors=False, brief=config.brief)
    console_module = config.console_level <= logging.DEBUG
    istty = sys.stdout.isatty()
    console_f = _Formatter(config.timestamp, console_module, ansi_colors=istty, brief=config.brief)

    if config.file_level == logging.NOTSET:
        logger.setLevel(config.console_level)
    else:
        logger.setLevel(min(config.file_level, config.console_level))

    console_handler = logging.StreamHandler(stream=_TqdmStream)
    console_handler.setFormatter(console_f)
    console_handler.setLevel(config.console_level)
    if config.brief:
        console_handler.addFilter(_BriefFilter())
    logger.addHandler(console_handler)

    if config.path:
        logger.info(f"Logging debug messages to {config.path}")
        file_handler = logging.FileHandler(config.path)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(config.file_level)
        if config.brief:
            file_handler.addFilter(_BriefFilter())
        logger.addHandler(file_handler)

    # exclude debug messages from various third party libraries
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)
    logging.getLogger("boto3").setLevel(logging.WARNING)
    logging.getLogger("botocore").setLevel(logging.WARNING)
    logging.getLogger("s3transfer").setLevel(logging.WARNING)
    _logging_configured = True

# Copyright Axelera AI, 2023

import argparse
from contextlib import contextmanager
import logging
from logging import DEBUG, ERROR, INFO, WARNING
import shlex
import time
import unittest.mock

import pytest

from axelera.app import logging_utils
from axelera.app.logging_utils import TRACE

IT = logging_utils._ITALIC
BU = logging_utils._BOLD_WHITE_UNDERLINE
RE = logging_utils._RESET
TR = logging_utils._WHITE_W_GREEN_BG
DE = logging_utils._BRGREEN
WA = logging_utils._YELLOW
ER = logging_utils._RED
MOCK_NOW = 123456722.123


def test_markdown():
    assert f"This is a {IT}trace{RE} message" == logging_utils._markdown(
        "This is a **trace** message"
    )
    assert f"{BU}Heading{RE}" == logging_utils._markdown("## Heading")
    assert f"{BU}Heading {IT}emp{RE}{BU} normal{RE}" == logging_utils._markdown(
        "## Heading **emp** normal"
    )


class MockHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, msg):
        self.records.append(msg)


@contextmanager
def create_test_logger(level):
    log = logging_utils.getLogger("test_logger")
    log.setLevel(level)
    handler = MockHandler()
    log.addHandler(handler)
    try:
        yield handler.records, log
    finally:
        logging.root.manager.loggerDict.pop("test_logger", None)


def test_get_logger_has_trace_method():
    with create_test_logger(TRACE) as (records, log):
        log.trace("This is a trace message")
        assert records[0].getMessage() == "This is a trace message"
        assert records[0].levelno == TRACE


def assert_exception_record(record, exc_type, exc_msg, level):
    message = record.getMessage()
    assert exc_msg in message
    assert exc_type.__name__ in message
    assert "most recent call last" in message
    assert record.levelno == level


def test_trace_exception():
    with create_test_logger(TRACE) as (records, log):
        try:
            raise RuntimeError("oopsie")
        except Exception as e:
            log.trace_exception()
        assert_exception_record(records[0], RuntimeError, "oopsie", ERROR)


def test_trace_exception_with_non_verbose_level():
    with create_test_logger(WARNING) as (records, log):
        try:
            raise RuntimeError("oopsie")
        except Exception as e:
            log.trace_exception()
        assert records == []


def test_exit_with_error_log_no_msg():
    with create_test_logger(WARNING) as (records, log):
        try:
            raise RuntimeError("oopsie")
        except Exception as e:
            with pytest.raises(SystemExit):
                log.exit_with_error_log()
        assert records == []


def test_exit_with_error_log_str_msg():
    with create_test_logger(WARNING) as (records, log):
        try:
            raise RuntimeError("oopsie")
        except Exception as e:
            with pytest.raises(SystemExit):
                log.exit_with_error_log("hello")
        assert records[0].getMessage() == "hello"


def test_exit_with_error_log_exc_msg():
    with create_test_logger(WARNING) as (records, log):
        try:
            raise RuntimeError("oopsie")
        except Exception as e:
            with pytest.raises(SystemExit):
                log.exit_with_error_log(e)
        assert "oopsie" in records[0].getMessage()


@pytest.mark.parametrize(
    "cmdline, expected",
    [
        ("-qq", (ERROR, logging.NOTSET, "", False)),
        ("-q", (WARNING, logging.NOTSET, "", False)),
        ("", (INFO, logging.NOTSET, "", False)),
        ("--logtimestamp", (INFO, logging.NOTSET, "", True)),
        ("-v", (DEBUG, logging.NOTSET, "", False)),
        ("-vv", (TRACE, logging.NOTSET, "", False)),
        ("-vvv", (TRACE, logging.NOTSET, "", False)),
        ("-qq --logfile=x", (ERROR, ERROR, "x", False)),
        ("-q --logfile=x", (WARNING, WARNING, "x", False)),
        ("--logfile=x", (INFO, INFO, "x", False)),
        ("-v --logfile=x", (DEBUG, DEBUG, "x", False)),
        ("-vv --logfile=x", (TRACE, TRACE, "x", False)),
        ("-vvv --logfile=x", (TRACE, TRACE, "x", False)),
        ("-qq --logfile=x --loglevel=debug", (ERROR, DEBUG, "x", False)),
        ("-q --logfile=x --loglevel=debug", (WARNING, DEBUG, "x", False)),
        ("--logfile=x --loglevel=debug", (INFO, DEBUG, "x", False)),
        ("-v --logfile=x --loglevel=info", (DEBUG, INFO, "x", False)),
        ("-vv --logfile=x --loglevel=info", (TRACE, INFO, "x", False)),
        ("-vvv --logfile=x --loglevel=error", (TRACE, ERROR, "x", False)),
    ],
)
def test_get_config_from_args(cmdline, expected):
    parser = argparse.ArgumentParser()
    logging_utils.add_logging_args(parser)
    args = parser.parse_args(shlex.split(cmdline))
    assert logging_utils.Config(*expected) == logging_utils.get_config_from_args(args)


def test_get_config_from_args_with_invalid_loglevel():
    e = r"Unknown logging level.*FATAL are valid but not invalid"
    args = argparse.Namespace(
        verbose=0, quiet=False, logfile="x", loglevel="invalid", logtimestamp=""
    )
    with pytest.raises(ValueError, match=e):
        logging_utils.get_config_from_args(args)


def test_quiet_and_verbose_are_mutually_exclusive(capsys):
    parser = argparse.ArgumentParser()
    logging_utils.add_logging_args(parser)
    with pytest.raises(SystemExit, match="2"):
        parser.parse_args(shlex.split("-q -v"))
    out, err = capsys.readouterr()
    assert "" == out
    assert "-v/--verbose: not allowed with argument -q/--quiet" in err


def emit_messages(
    console_level=TRACE,
    file_level=logging.NOTSET,
    logfile="",
    timestamp=False,
):
    with unittest.mock.patch.object(logging.time, "time") as mock_time:
        mock_time.return_value = MOCK_NOW
        logging_utils.configure_logging(
            logging_utils.Config(console_level, file_level, logfile, timestamp)
        )
        log = logging_utils.getLogger("test_axelera_logging")
        log.trace("This is a trace message")
        log.debug("This is a **debug** message")
        log.info("This is an info message")
        log.warning("This is a warning")
        log.error("## This is an error")


@pytest.mark.parametrize("timestamp", [False, True])
@pytest.mark.parametrize(
    "console_level, file_level, expected_console, expected_file",
    [
        (
            TRACE,
            WARNING,
            """\
{ts}INFO    : Logging debug messages to {logfile}
{ts}TRACE   :test_axelera_logging: This is a trace message
{ts}DEBUG   :test_axelera_logging: This is a **debug** message
{ts}INFO    :test_axelera_logging: This is an info message
{ts}WARNING :test_axelera_logging: This is a warning
{ts}ERROR   :test_axelera_logging: ## This is an error
""",
            """\
{ts}WARNING :test_axelera_logging: This is a warning
{ts}ERROR   :test_axelera_logging: ## This is an error
""",
        ),
        (
            DEBUG,
            ERROR,
            """\
{ts}INFO    : Logging debug messages to {logfile}
{ts}DEBUG   :test_axelera_logging: This is a **debug** message
{ts}INFO    :test_axelera_logging: This is an info message
{ts}WARNING :test_axelera_logging: This is a warning
{ts}ERROR   :test_axelera_logging: ## This is an error
""",
            """\
{ts}ERROR   :test_axelera_logging: ## This is an error
""",
        ),
        (
            INFO,
            ERROR,
            """\
{ts}INFO    : Logging debug messages to {logfile}
{ts}INFO    : This is an info message
{ts}WARNING : This is a warning
{ts}ERROR   : ## This is an error
""",
            """\
{ts}ERROR   :test_axelera_logging: ## This is an error
""",
        ),
        (
            WARNING,
            TRACE,
            """\
{ts}WARNING : This is a warning
{ts}ERROR   : ## This is an error
""",
            """\
{ts}TRACE   :test_axelera_logging: This is a trace message
{ts}DEBUG   :test_axelera_logging: This is a **debug** message
{ts}INFO    :test_axelera_logging: This is an info message
{ts}WARNING :test_axelera_logging: This is a warning
{ts}ERROR   :test_axelera_logging: ## This is an error
""",
        ),
    ],
)
def test_set_logging(
    tmp_path,
    capsys,
    console_level,
    file_level,
    timestamp,
    expected_console,
    expected_file,
):
    try:
        logfile = tmp_path / "x"
        emit_messages(console_level, file_level, logfile, timestamp)
        out, _ = capsys.readouterr()
        now = time.strftime("%H:%M:%S", time.localtime(MOCK_NOW))
        ts = f"[{now}.122] " if timestamp else ""
        assert expected_console.format(**locals()) == out
        file_contents = logfile.read_text()
        assert expected_file.format(**locals()) == file_contents
    finally:
        logging.root.handlers.clear()


def test_ansi_formatting_goes_to_console_only(tmp_path, capsys):
    logfile = tmp_path / "x"
    ts = ""
    expected_console = f"""\
{ts}INFO    : Logging debug messages to {logfile}
{ts}{TR}TRACE{RE}   :test_axelera_logging: This is a trace message
{ts}{DE}DEBUG{RE}   :test_axelera_logging: This is a {IT}debug{RE} message
{ts}INFO    :test_axelera_logging: This is an info message
{ts}{WA}WARNING{RE} :test_axelera_logging: This is a warning
{ts}{ER}ERROR{RE}   :test_axelera_logging: {BU}This is an error{RE}
"""
    expected_file = f"""\
{ts}WARNING :test_axelera_logging: This is a warning
{ts}ERROR   :test_axelera_logging: ## This is an error
"""
    try:
        with unittest.mock.patch.object(logging_utils.sys.stdout, "isatty") as mock_isatty:
            mock_isatty.return_value = True
            emit_messages(logfile=logfile, file_level=WARNING)
        out, _ = capsys.readouterr()
        assert expected_console == out
        file_contents = logfile.read_text()
        assert expected_file == file_contents
    finally:
        logging.root.handlers.clear()


def test_multiline_log_with_ansi(tmp_path, capsys):
    logfile = tmp_path / "x"
    ts = ""
    expected_console = f"""\
{ts}{WA}WARNING{RE} : This is a multi
{ts}{WA}WARNING{RE} : line message
{ts}{WA}WARNING{RE} : and a third line
"""
    expected_file = f"""\
{ts}WARNING :test_axelera_logging: This is a multi
{ts}WARNING :test_axelera_logging: line message
{ts}WARNING :test_axelera_logging: and a third line
"""
    try:
        with unittest.mock.patch.object(logging_utils.sys.stdout, "isatty") as mock_isatty:
            mock_isatty.return_value = True
            logging_utils.configure_logging(logging_utils.Config(WARNING, WARNING, logfile))
            log = logging_utils.getLogger("test_axelera_logging")
            log.warning("This is a multi\nline message\nand a third line")
        out, _ = capsys.readouterr()
        assert expected_console == out
        file_contents = logfile.read_text()
        assert expected_file == file_contents
    finally:
        logging.root.handlers.clear()

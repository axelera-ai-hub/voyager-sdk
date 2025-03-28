#!/usr/bin/env python3
# Copyright Axelera AI, 2025
from __future__ import annotations

from argparse import Namespace
import importlib.metadata
import os
from pathlib import Path
from unittest.mock import Mock, patch

import download_prebuilt
import pytest

from axelera.app import config


@pytest.mark.parametrize(
    'input, exp',
    [
        ('0.9.0', '0.9.0'),
        ('0.9.0-rc1', '0.9.0-rc1'),
        ('0.9.0rc1.post4+git.14156ade.dirty', '0.9.0-rc1'),
    ],
)
def test_get_version_cfg_contains(input, exp):
    mok = Mock()
    mok.read_text.return_value = f'nelly the elephant\n AX_VERSION: {input}'
    mbad = Mock()
    mbad.read_text.return_value = f'nelly the elephant\n AX_VERSN: {input}'
    with patch.object(Path, 'glob', return_value=[mok, mbad]):
        assert download_prebuilt.get_version() == exp
    with patch.object(Path, 'glob', return_value=[mbad, mok]):
        assert download_prebuilt.get_version() == exp


@pytest.mark.parametrize(
    'input, exp',
    [
        ('1.2.0rc1.post4+git.14156ade.dirty', '1.2.0-rc1'),
        ('1.2.0rc1.post4+git.14156ade', '1.2.0-rc1'),
        ('1.2.0rc1.post4+git', '1.2.0-rc1'),
        ('1.2.0rc1', '1.2.0-rc1'),
        ('1.3.0', '1.3.0'),
    ],
)
def test_get_version_cfg_fails_importlib_ok(input, exp):
    with patch.object(Path, 'glob', return_value=[]):
        with patch.object(importlib.metadata, 'version', return_value=input):
            assert download_prebuilt.get_version() == exp


def test_get_version_cfg_fails_importlib_fails():
    def package_not_found(*args, **kwargs):
        raise importlib.metadata.PackageNotFoundError("asdasdasd")

    def os_error(*args, **kwargs):
        raise OSError("some OS Error")

    with patch.object(Path, 'glob', return_value=[]):
        with patch.object(importlib.metadata, 'version', package_not_found):
            with pytest.raises(
                RuntimeError,
                match=r'''Failed to find config file with AX_VERSION in cfg\.
Failed to find python package axelera-runtime: No package metadata was found for asdasdasd\.
Please specify version with --version''',
            ):
                download_prebuilt.get_version()

    with patch.object(Path, 'glob', os_error):
        with patch.object(importlib.metadata, 'version', package_not_found):
            with pytest.raises(
                RuntimeError,
                match=r'''Failed to detect version from config: some OS Error\.
Failed to find python package axelera-runtime: No package metadata was found for asdasdasd\.
Please specify version with --version''',
            ):
                download_prebuilt.get_version()

    with patch.object(Path, 'glob', return_value=[]):
        with patch.object(importlib.metadata, 'version', return_value='unparseable'):
            with pytest.raises(
                RuntimeError,
                match=r'''Failed to find config file with AX_VERSION in cfg\.
Failed to parse version \(unparseable\) of python package axelera-runtime\.
Please specify version with --version''',
            ):
                download_prebuilt.get_version()

        # assert download_prebuilt.get_version() == '1.1.0-rc1'


def test_toplevel_no_framework():
    nofmwk = os.environ.copy()
    nofmwk.pop('AXELERA_FRAMEWORK', None)
    with patch.object(config, 'env', Namespace(framework='')):
        with pytest.raises(RuntimeError, match=r'Environment variable AXELERA_FRAMEWORK not set'):
            download_prebuilt.download(Namespace())


def test_toplevel_incorrect_cwd():
    with patch.object(Path, 'cwd', return_value=Path('somewhereelse')):
        with pytest.raises(RuntimeError, match=r'This script must be run from'):
            download_prebuilt.download(Namespace())


def test_toplevel_show_version(capsys):
    with patch.object(Path, 'cwd', return_value=config.env.framework):
        with patch.object(download_prebuilt, 'get_version', return_value='1.1.0-rc99'):
            download_prebuilt.download(Namespace(version='', show_version=True))
    assert capsys.readouterr().out == '1.1.0-rc99\n'


def test_toplevel_show_version_leading_v(capsys):
    with patch.object(Path, 'cwd', lambda: config.env.framework):
        with patch.object(download_prebuilt, 'get_version', return_value='v1.1.0-rc99'):
            download_prebuilt.download(Namespace(version='', show_version=True))
    assert capsys.readouterr().out == '1.1.0-rc99\n'


def test_toplevel_list_models(capsys):
    with patch.object(Path, 'cwd', lambda: config.env.framework):
        with patch.object(download_prebuilt, 'get_models', return_value=['alice.zip', 'bob.zip']):
            download_prebuilt.download(Namespace(version='', show_version=False, list=True))
    assert capsys.readouterr().out == 'alice\nbob\n'

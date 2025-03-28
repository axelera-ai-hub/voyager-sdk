#!/usr/bin/env python3
# Copyright Axelera AI, 2024
from __future__ import annotations

import argparse
import importlib.metadata
import pathlib
import re
import subprocess
import sys
import tempfile

import requests

from axelera.app import config, utils

MODELS_FILE = 'models.json'
BASE_URL = 'https://media.axelera.ai/artifacts/prebuilt/voyager-sdk-v'

parser = argparse.ArgumentParser(description='Download models from Axelera cloud service.')
parser.add_argument(
    'model',
    type=str,
    nargs='?',
    help="Model name to download, e.g. yolov5s-v7-coco",
)
parser.add_argument('-v', '--verbose', action='store_true', help='Print stack trace on error')
parser.add_argument('--version', default='', help='Override version of SDK, e.g. 1.2.0-rc3')
parser.add_argument('--show-version', action='store_true', help='Print version of SDK and exit')
parser.add_argument(
    '--list', action='store_true', help='List all available models for the current version'
)
parser.add_argument(
    '--all', action='store_true', help='Download all available models for the current version'
)


def get_version():
    def with_rc(v, rc):
        return f'{v}-{rc}' if rc else v

    try:
        for item in pathlib.Path('cfg').glob('config-*.yaml'):
            contents = item.read_text()
            if m := re.search(
                r'^\s*AX_VERSION:\s*(\d+\.\d+\.\d+)-?(rc\d+)?', contents, re.MULTILINE
            ):
                return with_rc(m.group(1), m.group(2))
        err = 'Failed to find config file with AX_VERSION in cfg.'
    except Exception as e:
        err = f"Failed to detect version from config: {e}."
    please = '\nPlease specify version with --version'
    try:
        v = importlib.metadata.version('axelera-runtime')
    except importlib.metadata.PackageNotFoundError as e:
        err += f"\nFailed to find python package axelera-runtime: {e}.{please}"
        raise RuntimeError(err)
    m = re.match(r'^(\d+\.\d+\.\d+)-?(rc\d+)?', v)
    if not m:
        err += f"\nFailed to parse version ({v}) of python package axelera-runtime.{please}"
        raise RuntimeError(err)
    return with_rc(m.group(1), m.group(2))


def get_models(version):
    with requests.get(f"{BASE_URL}{version}/{MODELS_FILE}") as request:
        request.raise_for_status()
        return request.json()


def download(args):
    '''Download prebuilt models from the cloud.'''
    fmwk = config.env.framework
    if not fmwk:
        raise RuntimeError(
            'Environment variable AXELERA_FRAMEWORK not set\n'
            'Please activate the framework environment with\n'
            '$ . venv/bin/activate'
        )
    if pathlib.Path.cwd() != pathlib.Path(fmwk):
        raise RuntimeError(f'This script must be run from the framework directory : {fmwk}')

    version = args.version or get_version()
    version = version.lstrip('v')  # remove leading 'v' if present - a common mistake
    if args.show_version:
        print(version)
        return

    models = get_models(version)
    if args.list:
        indent = ''
        if sys.stdout.isatty():
            print(f'Available models: (version {version})')
            indent = '  '
        for model in models:
            print(f'{indent}{model.removesuffix(".zip")}')
        return

    if args.all:
        to_download = models.keys()
    elif not args.model:
        raise ValueError('Please specify a model to download, or --all or --list')
    else:
        to_download = [args.model.removeprefix('mc-')]

    to_download = [m.removesuffix(".zip") for m in to_download]
    for mname in to_download:
        f = f"{mname}.zip"
        if not (checksum := models.get(f)):
            raise ValueError(f'Prebuilt model not available for {args.model} in v{version}')

        with tempfile.TemporaryDirectory() as unpack_dir:
            url = f"{BASE_URL}{version}/{f}"
            utils.download(url, pathlib.Path(f"{unpack_dir}/{f}"), checksum)
            try:
                subprocess.run(f'unzip {unpack_dir}/{f}', check=True, shell=True)
            finally:
                (pathlib.Path(unpack_dir) / f).unlink()


if __name__ == '__main__':
    args = parser.parse_args()
    try:
        download(args)
    except Exception as e:
        if args.verbose:
            raise
        sys.exit(f'Error: {e}')

# Copyright Axelera AI, 2026
from __future__ import annotations

import types
import urllib.error
from pathlib import Path
from unittest import mock

import pytest

from model_recipes.shared import assets


def test_fetch_reports_unpublished_http_asset(tmp_path: Path):
    error = urllib.error.HTTPError(
        'https://media.example/missing.pt2', 403, 'Forbidden', {}, None
    )

    with mock.patch.object(assets.urllib.request, 'urlopen', side_effect=error):
        try:
            assets.fetch(error.url, tmp_path / 'missing.pt2')
        except FileNotFoundError as caught:
            message = str(caught)
        else:
            raise AssertionError('expected FileNotFoundError')

    assert 'HTTP 403' in message
    assert 'not be published yet' in message
    assert 'Export the model locally' in message


def test_fetch_reports_missing_http_asset(tmp_path: Path):
    error = urllib.error.HTTPError(
        'https://media.example/missing.pt2', 404, 'Not Found', {}, None
    )

    with mock.patch.object(assets.urllib.request, 'urlopen', side_effect=error):
        try:
            assets.fetch(error.url, tmp_path / 'missing.pt2')
        except FileNotFoundError as caught:
            message = str(caught)
        else:
            raise AssertionError('expected FileNotFoundError')

    assert 'HTTP 404' in message
    assert 'not be published yet' in message


def _fake_boto3(download_file: mock.Mock) -> types.ModuleType:
    """A stand-in `boto3` module -- neither `boto3` nor `botocore` is a `tox -e py310`
    dependency, so a real import would fail there regardless of what it returns."""
    module = types.ModuleType('boto3')
    module.client = mock.Mock(return_value=mock.Mock(download_file=download_file))
    return module


class _FakeClientError(Exception):
    """Stand-in for `botocore.exceptions.ClientError`: same `.response` shape."""

    def __init__(self, code: str):
        super().__init__(code)
        self.response = {'Error': {'Code': code}}


def _fake_botocore() -> dict[str, types.ModuleType]:
    exceptions = types.ModuleType('botocore.exceptions')
    exceptions.ClientError = _FakeClientError
    botocore = types.ModuleType('botocore')
    botocore.exceptions = exceptions
    return {'botocore': botocore, 'botocore.exceptions': exceptions}


def test_fetch_downloads_an_s3_asset(tmp_path: Path):
    dest = tmp_path / 'weight.pt2'

    def fake_download_file(bucket, key, filename):
        assert bucket == 'axelera-model-card'
        assert key == 'timm-vit-small/vit_small.pt2'
        Path(filename).write_bytes(b'weight-bytes')

    fake_boto3 = _fake_boto3(mock.Mock(side_effect=fake_download_file))
    with mock.patch.dict('sys.modules', {'boto3': fake_boto3}):
        result = assets.fetch('s3://axelera-model-card/timm-vit-small/vit_small.pt2', dest)

    assert result == dest
    assert dest.read_bytes() == b'weight-bytes'


def test_fetch_reports_missing_s3_asset(tmp_path: Path):
    fake_boto3 = _fake_boto3(mock.Mock(side_effect=_FakeClientError('NoSuchKey')))
    with mock.patch.dict('sys.modules', {'boto3': fake_boto3, **_fake_botocore()}), pytest.raises(
        FileNotFoundError
    ) as caught:
        assets.fetch('s3://axelera-model-card/missing.pt2', tmp_path / 'missing.pt2')

    message = str(caught.value)
    assert 's3://axelera-model-card/missing.pt2' in message
    assert 'not be published yet' in message
    assert 'axzoo export <config> --publish' in message


def test_fetch_reports_missing_boto3_for_s3_asset(tmp_path: Path):
    with mock.patch.dict('sys.modules', {'boto3': None}), pytest.raises(RuntimeError) as caught:
        assets.fetch('s3://axelera-model-card/weight.pt2', tmp_path / 'weight.pt2')

    assert 'boto3 is required for S3 downloads' in str(caught.value)



@pytest.mark.parametrize('status', [403, 404])
def test_calibration_images_reports_missing_download(tmp_path: Path, status: int):
    error = urllib.error.HTTPError(
        f'https://media.example/calibration-{status}.zip', status, 'Forbidden', {}, None
    )

    with mock.patch.object(assets.urllib.request, 'urlopen', side_effect=error), pytest.raises(
        FileNotFoundError
    ) as caught:
        assets.calibration_images(
            directory='',
            url=error.url,
            expected_md5='deadbeef',
            cache_dir=tmp_path / 'cache',
            limit=10,
        )

    message = str(caught.value)
    assert 'Calibration images not available at' in message
    assert 'CALIBRATION_DIR' in message

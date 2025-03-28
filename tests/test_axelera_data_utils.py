# Copyright Axelera AI, 2024

import itertools
import os
from pathlib import Path
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import yaml

from axelera import types
from axelera.app import data_utils
from axelera.app.data_utils import DatasetStatus, DatasetYamlFile, check_and_download_dataset


class DLIterator:
    def __init__(self, data, batch_size):
        self.data = data
        self.batch_size = batch_size
        self._iter = iter(data)

    def __iter__(self):
        return self

    def __next__(self):
        while batch := list(itertools.islice(self._iter, self.batch_size)):
            return batch
        raise StopIteration

    def __len__(self):
        return len(self.data) // self.batch_size


class MockDataLoader(types.DataLoader):
    def __init__(self, batch_size=1, size=12, repeat_batch=False):
        self.dataset = (
            itertools.repeat((np.full((3, 9), 0), 0, f'image0'), size)
            if repeat_batch
            else [(np.full((3, 9), i), i, f'image{i}') for i in range(size)]
        )
        self.batch_size = batch_size
        self.repeat_batch = repeat_batch

    def __len__(self):
        return len(self.dataset) // self.batch_size

    def __iter__(self):
        return DLIterator(self.dataset, self.batch_size)


def formatter_for_calibration(batch_data):
    return [data[0] for data in batch_data]


def formatter_for_validation(batch_data):
    return [
        {'img': data[0], 'groundtruth': data[1], 'kwargs': {'img_id': data[2]}}
        for data in batch_data
    ]


@pytest.mark.parametrize('batch_size, expected_len', [(1, 12), (4, 3)])
@pytest.mark.parametrize('is_calibration', [True, False])
def test_normalized_data_loader(batch_size, expected_len, is_calibration):
    formatter = formatter_for_calibration if is_calibration else formatter_for_validation
    normalized = data_utils.NormalizedDataLoaderImpl(
        MockDataLoader(batch_size=batch_size), formatter, is_calibration=is_calibration
    )
    assert len(normalized) == expected_len
    for ix, batch in enumerate(normalized):
        assert len(batch) == batch_size
        value = ix * batch_size
        if is_calibration:
            assert batch[0].shape == (3, 9)
            assert batch[0][0, 0] == value
        else:
            assert batch[0]['img'].shape == (3, 9)
            assert batch[0]['groundtruth'] == value
            assert batch[0]['kwargs']['img_id'] == f'image{value}'
            assert batch[0]['img'][0, 0] == value


@pytest.mark.parametrize('batch_size', [1, 4])
def test_batch_validate(batch_size):
    normalized = data_utils.NormalizedDataLoaderImpl(
        MockDataLoader(batch_size=batch_size, repeat_batch=True),
        formatter_for_calibration,
        is_calibration=True,
    )
    with pytest.raises(ValueError, match="batch is the same as the previous batch"):
        for ix, batch in enumerate(normalized):
            if ix == 2:
                break


@pytest.mark.parametrize('batch_size, num_images, expected', [(1, 10, 10), (4, 10, 3), (2, 10, 5)])
def test_num_images(batch_size, num_images, expected):
    normalized = data_utils.NormalizedDataLoaderImpl(
        MockDataLoader(batch_size=batch_size, size=200),
        formatter_for_calibration,
        is_calibration=True,
        num_batches=data_utils.NormalizedDataLoaderImpl.required_batches(num_images, batch_size),
    )
    assert len(normalized) == expected
    assert len([batch for batch in normalized]) == expected
    assert sum([len(batch) for batch in normalized]) >= num_images


def test_iters():
    dl = MockDataLoader()
    normalized = data_utils.NormalizedDataLoaderImpl(
        dl, formatter_for_calibration, is_calibration=True
    )
    iter1 = iter(normalized)
    iter2 = iter(normalized)
    assert np.array_equal(next(iter1), next(iter2))
    assert np.array_equal(next(iter1), next(iter2))
    remain1 = len([batch for batch in iter1])
    remain2 = len([batch for batch in iter2])
    assert remain1 == remain2
    assert remain1 == len(dl) - 2


@pytest.fixture
def mock_dataset_private_yaml():
    return {
        'ImageNet': {
            'description': 'ImageNet dataset',
            'download_hint': 'Please download from image-net.org {split_files}',
            'splits': {
                'val': [
                    {
                        'url': 's3://bucket/ILSVRC2012_devkit_t12.tar.gz',
                        'md5': 'fa75699e90414af021442c21a62c3abf',
                        'drop_dirs': 0,
                        'check_files': ['val/n01641577'],
                    }
                ],
                'subset': [
                    {
                        'url': 's3://bucket/imagenet_subset.zip',
                        'md5': '4c48c075ec5d3e93e197f37cf3f67a7b',
                        'drop_dirs': 0,
                        'check_files': ['readme_imagenet_subset.txt'],
                    }
                ],
            },
        },
        'Customer.QNAP.FaceRecognition': {
            'description': 'QNAP face recognition dataset',
            'download_hint': 'Please download QNAP dataset',
            'splits': {
                'val': [
                    {
                        'url': 's3://bucket/qnap.zip',
                        'md5': 'md5hash',
                        'drop_dirs': 0,
                        'check_files': ['val/data.txt'],
                    }
                ]
            },
        },
    }


@pytest.fixture
def mock_dataset_public_yaml():
    return {
        'ImageNet': {
            'description': 'ImageNet dataset',
            'splits': {
                'val': [
                    {
                        'url': 's3://public-bucket/ILSVRC2012_devkit_t12.tar.gz',
                        'md5': 'fa75699e90414af021442c21a62c3abf',
                        'drop_dirs': 0,
                        'check_files': ['val/n01641577'],
                    }
                ]
            },
        }
    }


@pytest.fixture
def mock_dataset_prompt_yaml():
    return {
        'ImageNet': {
            'description': 'ImageNet dataset',
            'download_hint': 'Please download from image-net.org {split_files}',
            'splits': {'val': [{'check_files': ['val/n01641577']}]},
        },
        'Customer.QNAP.FaceRecognition': {
            'description': 'QNAP face recognition dataset',
            'download_hint': 'Please download QNAP dataset',
            'splits': {'val': [{'check_files': ['val/data.txt']}]},
        },
    }


@pytest.fixture
def mock_dataset_prompt_yaml_with_files():
    return {
        'ImageNet': {
            'description': 'ImageNet dataset',
            'download_hint': 'Please download from image-net.org {split_files}',
            'splits': {
                'val': [
                    {
                        'file': 'ILSVRC2012_devkit_t12.tar.gz',
                        'drop_dirs': 0,
                        'check_files': ['val/n01641577'],
                    },
                    {
                        'file': 'ILSVRC2012_img_val.tar',
                        'drop_dirs': 0,
                        'sub_dir': 'val',
                        'check_files': ['val/n01440764/ILSVRC2012_val_00003014.JPEG'],
                    },
                ]
            },
        },
        'Customer.QNAP.FaceRecognition': {
            'description': 'QNAP face recognition dataset',
            'download_hint': 'Please download QNAP dataset {split_files}',
            'splits': {
                'val': [{'file': 'qnap.zip', 'drop_dirs': 0, 'check_files': ['val/data.txt']}]
            },
        },
    }


@pytest.fixture
def mock_filesystem(
    tmp_path, mock_dataset_private_yaml, mock_dataset_public_yaml, mock_dataset_prompt_yaml
):
    framework_path = tmp_path / "framework"
    framework_path.mkdir(parents=True)
    datasets_path = framework_path / "ax_datasets"
    datasets_path.mkdir()

    # Create dataset YAML files
    with open(datasets_path / DatasetYamlFile.DATASET_PRIVATE.value, 'w') as f:
        yaml.dump(mock_dataset_private_yaml, f)

    with open(datasets_path / DatasetYamlFile.DATASET_PUBLIC.value, 'w') as f:
        yaml.dump(mock_dataset_public_yaml, f)

    with open(datasets_path / DatasetYamlFile.DATASET_PROMPT.value, 'w') as f:
        yaml.dump(mock_dataset_prompt_yaml, f)

    return framework_path


@pytest.mark.parametrize("s3_available", ['0', '1'])
@pytest.mark.parametrize("is_private", [True, False])
@pytest.mark.parametrize(
    "dataset_name,expected_error",
    [
        ('ImageNet', None),
        ('Customer.QNAP.FaceRecognition', ValueError),  # Should fail for public (is_private=False)
    ],
)
def test_dataset_download_scenarios(
    mock_filesystem, tmp_path, s3_available, is_private, dataset_name, expected_error
):
    with patch.dict(
        os.environ,
        {'AXELERA_FRAMEWORK': str(mock_filesystem), 'AXELERA_S3_AVAILABLE': s3_available},
    ), patch('axelera.app.data_utils._check_dataset_status') as mock_check, patch(
        'axelera.app.data_utils._download_dataset'
    ) as mock_download, patch(
        'axelera.app.data_utils._create_completion_stamp'
    ) as mock_stamp, patch(
        'axelera.app.data_utils._print_hint'
    ) as mock_hint, patch(
        'axelera.app.data_utils.utils.download_and_extract_asset'
    ) as mock_extract:

        mock_check.return_value = (DatasetStatus.INCOMPLETE, "Dataset is incomplete")

        if not is_private and dataset_name.startswith('Customer.'):
            with pytest.raises(ValueError) as exc_info:
                check_and_download_dataset(dataset_name, tmp_path, is_private=is_private)
            assert "not found in configuration" in str(exc_info.value)
            return

        if s3_available == '0':
            with pytest.raises(RuntimeError) as exc_info:
                check_and_download_dataset(dataset_name, tmp_path, is_private=is_private)
            assert "Please follow the hint to download the dataset." in str(exc_info.value)
            mock_hint.assert_called_once()
            mock_download.assert_not_called()
        else:
            check_and_download_dataset(dataset_name, tmp_path, is_private=is_private)
            mock_download.assert_called_once()
            mock_stamp.assert_called_once()
            mock_hint.assert_not_called()


def test_fallback_to_prompt_yaml(mock_filesystem, tmp_path):
    # Remove private YAML to test fallback
    private_yaml = mock_filesystem / "ax_datasets" / DatasetYamlFile.DATASET_PRIVATE.value
    private_yaml.unlink()

    with patch.dict(
        os.environ, {'AXELERA_FRAMEWORK': str(mock_filesystem), 'AXELERA_S3_AVAILABLE': '0'}
    ), patch('axelera.app.data_utils._check_dataset_status') as mock_check, patch(
        'axelera.app.data_utils._print_hint'
    ) as mock_hint, patch(
        'axelera.app.data_utils._download_dataset', MagicMock()
    ) as mock_download:

        mock_check.return_value = (DatasetStatus.INCOMPLETE, "Dataset is incomplete")

        # Should fall back to prompt YAML and show hint
        with pytest.raises(RuntimeError) as exc_info:
            check_and_download_dataset('ImageNet', tmp_path, is_private=True)

        assert "Please follow the hint to download the dataset." in str(exc_info.value)
        mock_hint.assert_called_once()
        mock_download.assert_not_called()


def test_subset_override_for_ci(mock_filesystem, tmp_path):
    with patch.dict(
        os.environ, {'AXELERA_FRAMEWORK': str(mock_filesystem), 'AXELERA_S3_AVAILABLE': 'subset'}
    ), patch('axelera.app.data_utils._check_dataset_status') as mock_check, patch(
        'axelera.app.data_utils._download_dataset'
    ) as mock_download:

        mock_check.return_value = (DatasetStatus.INCOMPLETE, "Dataset is incomplete")

        check_and_download_dataset('ImageNet', tmp_path, split='val')
        # Verify that 'subset' was used instead of 'val'
        assert mock_check.call_args[0][2] == 'subset'
        mock_download.assert_called_once()


def test_corrupted_dataset_redownload(mock_filesystem, tmp_path):
    with patch.dict(
        os.environ, {'AXELERA_FRAMEWORK': str(mock_filesystem), 'AXELERA_S3_AVAILABLE': '1'}
    ), patch('axelera.app.data_utils._check_dataset_status') as mock_check, patch(
        'axelera.app.data_utils._download_dataset'
    ) as mock_download, patch(
        'axelera.app.data_utils._create_completion_stamp'
    ) as mock_stamp:

        mock_check.return_value = (DatasetStatus.CORRUPTED, "Dataset is corrupted")

        check_and_download_dataset('ImageNet', tmp_path)
        mock_download.assert_called_once()
        mock_stamp.assert_called_once()


def test_missing_required_files(mock_filesystem, tmp_path):
    with patch.dict(
        os.environ, {'AXELERA_FRAMEWORK': str(mock_filesystem), 'AXELERA_S3_AVAILABLE': '1'}
    ), patch('axelera.app.data_utils._check_dataset_status') as mock_check, patch(
        'axelera.app.data_utils.DatasetConfig.verify_required_files'
    ) as mock_verify, patch(
        'axelera.app.data_utils._download_dataset'
    ) as mock_download:

        mock_check.return_value = (DatasetStatus.COMPLETE, "Dataset is complete")
        mock_verify.return_value = ['missing_file.txt']

        check_and_download_dataset('ImageNet', tmp_path)
        mock_verify.assert_called_once()
        mock_download.assert_called_once()


def test_download_hint_with_file_extraction(
    mock_filesystem, tmp_path, mock_dataset_prompt_yaml_with_files
):
    # Replace the default prompt YAML with our updated version
    prompt_yaml_path = mock_filesystem / "ax_datasets" / DatasetYamlFile.DATASET_PROMPT.value
    with open(prompt_yaml_path, 'w') as f:
        yaml.dump(mock_dataset_prompt_yaml_with_files, f)

    # Simulate private YAML being unavailable to force using prompt YAML
    private_yaml = mock_filesystem / "ax_datasets" / DatasetYamlFile.DATASET_PRIVATE.value
    private_yaml.unlink()

    with patch.dict(
        os.environ, {'AXELERA_FRAMEWORK': str(mock_filesystem), 'AXELERA_S3_AVAILABLE': '0'}
    ), patch('axelera.app.data_utils._check_dataset_status') as mock_check, patch(
        'axelera.app.data_utils._print_hint'
    ) as mock_hint:

        mock_check.return_value = (DatasetStatus.INCOMPLETE, "Dataset is incomplete")

        # Run the function which should generate a hint with file information
        with pytest.raises(RuntimeError):
            check_and_download_dataset('ImageNet', tmp_path, is_private=True)

        # Verify the hint contains the extracted filenames
        mock_hint.assert_called_once()
        hint_message = mock_hint.call_args[0][0]

        # Check that the hint includes both filenames from the val split
        assert 'ILSVRC2012_devkit_t12.tar.gz' in hint_message
        assert 'ILSVRC2012_img_val.tar' in hint_message


def test_integration_with_release_model_cards(mock_filesystem, tmp_path):
    """Test that the release_model_cards.py creates compatible prompt YAML that works with data_utils."""
    from internal_tools.release_model_cards import create_dataset_prompt_yaml

    # Get the paths
    private_yaml_path = mock_filesystem / "ax_datasets" / DatasetYamlFile.DATASET_PRIVATE.value
    prompt_yaml_path = mock_filesystem / "ax_datasets" / DatasetYamlFile.DATASET_PROMPT.value

    # Generate the prompt YAML using our updated function
    create_dataset_prompt_yaml(private_yaml_path, prompt_yaml_path)

    # Now unlink the private YAML to force using the prompt YAML
    private_yaml_path.unlink()

    with patch.dict(
        os.environ, {'AXELERA_FRAMEWORK': str(mock_filesystem), 'AXELERA_S3_AVAILABLE': '0'}
    ), patch('axelera.app.data_utils._check_dataset_status') as mock_check, patch(
        'axelera.app.data_utils._print_hint'
    ) as mock_hint:

        mock_check.return_value = (DatasetStatus.INCOMPLETE, "Dataset is incomplete")

        # Test with ImageNet dataset - default split is 'val'
        with pytest.raises(RuntimeError):
            check_and_download_dataset('ImageNet', tmp_path, is_private=True)

        # Verify hint contains the filename from val split
        mock_hint.assert_called_once()
        hint_message = mock_hint.call_args[0][0]
        assert 'ILSVRC2012_devkit_t12.tar.gz' in hint_message

        # Reset mock to test the subset split
        mock_hint.reset_mock()

        # Test with the 'subset' split specifically
        with pytest.raises(RuntimeError):
            check_and_download_dataset('ImageNet', tmp_path, split='subset', is_private=True)

        # Verify hint contains the filename from subset split
        mock_hint.assert_called_once()
        subset_hint_message = mock_hint.call_args[0][0]
        assert 'imagenet_subset.zip' in subset_hint_message


def test_customer_filtering():
    """Test that the create_dataset_prompt_yaml correctly filters customers."""
    from internal_tools.release_model_cards import create_dataset_prompt_yaml

    # Create a test YAML with multiple customers
    multi_customer_yaml = {
        'Customer.Apple.Dataset': {
            'description': 'Apple dataset',
            'splits': {'val': [{'url': 's3://bucket/apple.zip', 'md5': 'hash1'}]},
        },
        'Customer.Google.Dataset': {
            'description': 'Google dataset',
            'splits': {'val': [{'url': 's3://bucket/google.zip', 'md5': 'hash2'}]},
        },
        'PublicDataset': {
            'description': 'Public dataset',
            'splits': {'val': [{'url': 's3://bucket/public.zip', 'md5': 'hash3'}]},
        },
    }

    with tempfile.TemporaryDirectory() as tmpdirname:
        tmp_dir = Path(tmpdirname)
        tmp_dir.mkdir(exist_ok=True)
        source_path = tmp_dir / 'source.yaml'

        with open(source_path, 'w') as f:
            yaml.dump(multi_customer_yaml, f)

        # Test with specific customer filter
        apple_dest = tmp_dir / 'apple.yaml'
        create_dataset_prompt_yaml(source_path, apple_dest, customer_name='Apple')

        with open(apple_dest, 'r') as f:
            apple_result = yaml.safe_load(f)

        # Should include Apple and PublicDataset, but not Google
        assert 'Customer.Apple.Dataset' in apple_result
        assert 'PublicDataset' in apple_result
        assert 'Customer.Google.Dataset' not in apple_result

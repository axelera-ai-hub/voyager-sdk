# Copyright Axelera AI, 2025
#
# This file is used to define the data adapter for the model.
# The data adapter wraps the original dataset object and provides a common interface
# for the types.Model to access the dataset. The data adapter is also responsible for
# the formatting of the data for measurement pipeline.
# With the data adapter, we can decouple the Voyager SDK from dataset specifics,
# enabling the reuse of existing datasets as plugins and focusing on the model itself.
from __future__ import annotations

import datetime
import enum
import json
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple, Union
from urllib.parse import urlparse
import zipfile

import numpy as np
import yaml

from axelera import types

from . import logging_utils, utils
from .config import env
from .torch_utils import torch

if TYPE_CHECKING:
    from PIL import Image

LOG = logging_utils.getLogger(__name__)


class NormalizedDataIterator:
    def __init__(self, user_dataloader, formatter, process_batch):
        self._user_dataloader = user_dataloader
        self._process_batch = process_batch or (lambda x: x)
        self._validated = None if process_batch else True
        self._formatter = formatter
        self._iter_dataloader = iter(user_dataloader)

    def __iter__(self):
        return self

    def __next__(self):
        batch_data = next(self._iter_dataloader)
        axelera_batch_data = self._formatter(batch_data)
        axelera_batch_data = self._process_batch(axelera_batch_data)
        if not self._validated:
            if self._validated is None:
                self._validated = False
                self._first = axelera_batch_data
            else:
                self._validated = True
                for lhs, rhs in zip(self._first, axelera_batch_data):
                    if np.array_equal(lhs, rhs):
                        raise ValueError(
                            "The second batch is the same as the previous batch. Please ensure batches are different for quantization"
                        )
        return axelera_batch_data


class NormalizedDataLoaderImpl(types.NormalizedDataLoader):
    def __init__(self, user_dataloader, formatter, is_calibration, num_batches=100):
        self.user_dataloader = utils.LimitedLengthDataLoader(user_dataloader, num_batches)
        self.formatter = formatter
        self.is_calibration = is_calibration
        # preprocess onnxruntime session built from build_preprocess_ort()
        self.pp_ort_session = None
        self._validated = None

    @staticmethod
    def required_batches(num_images, batch_size):
        return (num_images + batch_size - 1) // batch_size

    def __iter__(self):
        return NormalizedDataIterator(
            self.user_dataloader,
            self.formatter,
            self._process_batch if self.is_calibration else None,
        )

    def __len__(self):
        return len(self.user_dataloader)

    def _process_batch(self, image_data):
        if self.pp_ort_session:
            return self.preprocess(image_data)

        return image_data

    def preprocess(self, image_batch):
        all_outputs = [[] for _ in range(len(self.pp_output_names))]

        for image in image_batch:
            image_4d = image.unsqueeze(0).numpy()  # The shape becomes (1, C, H, W)
            results = self.pp_ort_session.run(
                self.pp_output_names, {self.pp_input_names[0]: image_4d}
            )

            for i, output in enumerate(results):
                torch_tensor = torch.tensor(output).squeeze(0)
                all_outputs[i].append(torch_tensor)

        batched_outputs = [torch.stack(output_list, dim=0) for output_list in all_outputs]
        return batched_outputs[0] if len(batched_outputs) == 1 else tuple(batched_outputs)

    def build_preprocess_ort(self, preprocess_graph):
        # this function is called by axelera-compiler before passing into quantizer
        import onnxruntime

        self.pp_ort_session = onnxruntime.InferenceSession(
            preprocess_graph, providers=['CPUExecutionProvider']
        )
        self.pp_input_names = [node.name for node in self.pp_ort_session.get_inputs()]
        self.pp_output_names = [node.name for node in self.pp_ort_session.get_outputs()]


def download_repr_dataset_impl(repr_imgs_path: Path, repr_imgs_url: str, repr_imgs_md5: str = ''):
    """If no license concern, download the representative dataset for calibration
    Return True if the representative dataset is downloaded and extracted, False otherwise.
    """
    url_filename = urlparse(repr_imgs_url).path.split('/')[-1]
    target_file = repr_imgs_path / url_filename

    if not repr_imgs_path.is_dir() or not any(
        file for file in repr_imgs_path.iterdir() if file != target_file
    ):
        repr_imgs_path.mkdir(parents=True, exist_ok=True)
        utils.download(repr_imgs_url, target_file, repr_imgs_md5)
        if target_file.suffix == '.zip':
            drop_dirs = _determine_drop_dirs(target_file)
        else:
            raise ValueError(f"Unsupported file type: {target_file}")

        if drop_dirs is None:
            LOG.warning(
                f"Unable to determine the correct directory structure. Please manually organize the representative dataset in {repr_imgs_path}"
            )
        else:
            utils.extract(target_file, drop_dirs=drop_dirs)
        LOG.debug(f"{url_filename} uncompressed. You may now safely delete this file")
        return True
    return False


def _determine_drop_dirs(zip_file: Path) -> Optional[int]:
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        file_list = zip_ref.namelist()

        # Check if all files are in the root
        if all('/' not in file for file in file_list):
            return 0

        # Check if all files are in a single directory
        top_level_dirs = set(file.split('/')[0] for file in file_list if '/' in file)
        if len(top_level_dirs) == 1 and all(file.count('/') <= 1 for file in file_list):
            return 1

        # If structure is more complex, return None
        return None


class DatasetYamlFile(enum.Enum):
    DATASET_PRIVATE = 'dataset_private.yaml'
    DATASET_PROMPT = 'dataset_prompt.yaml'
    DATASET_PUBLIC = 'dataset_public.yaml'


class DatasetConfig:
    def __init__(self, yaml_path: str, dataset_yaml_name: DatasetYamlFile):
        self.config = self._load_yaml(yaml_path)
        self._validate_config(dataset_yaml_name)

    def _load_yaml(self, yaml_path: str) -> Dict:
        with open(yaml_path, 'r') as file:
            return yaml.safe_load(file)

    def _validate_config(self, dataset_yaml_name: DatasetYamlFile):
        for dataset, data in self.config.items():
            assert 'description' in data, f"Missing 'description' for {dataset}"
            if dataset_yaml_name != DatasetYamlFile.DATASET_PUBLIC:
                assert 'download_hint' in data, f"Missing 'download_hint' for {dataset}"
            if dataset_yaml_name != DatasetYamlFile.DATASET_PROMPT:
                assert 'splits' in data, f"Missing 'splits' for {dataset}"

                for split, split_data in data['splits'].items():
                    assert split in [
                        'test',
                        'val',
                        'train',
                        'subset',
                    ], f"Invalid split '{split}' in {dataset}"
                    for item in split_data:
                        assert 'url' in item, f"Missing 'url' in {dataset}.{split}"
                        assert item['url'].startswith(
                            's3://'
                        ), f"Invalid URL format in {dataset}.{split}"
                        assert 'drop_dirs' in item, f"Missing 'drop_dirs' in {dataset}.{split}"
                        if 'md5' in item:
                            assert isinstance(
                                item['md5'], str
                            ), f"Invalid md5 format in {dataset}.{split}"

    def get_datasets(self) -> List[str]:
        return list(self.config.keys())

    def get_dataset_info(self, dataset: str) -> Optional[Dict]:
        if not (dataset_info := self.config.get(dataset)):
            err_msg = f"Dataset '{dataset}' not found in configuration. "
            err_msg += f"Did you mean one of these: {self.get_datasets()}?"
            raise ValueError(err_msg)
        return dataset_info

    def get_description(self, dataset: str) -> Optional[str]:
        dataset_info = self.get_dataset_info(dataset)
        return dataset_info['description'] if dataset_info else None

    def get_split_files_str(self, dataset: str, split: str) -> str:
        files = self.get_files(dataset, split)
        if files:
            return ', '.join(f"'{item['name']}'" for item in files if 'name' in item)
        return ''

    def get_download_hint(self, dataset: str, split: str = 'val') -> Optional[str]:
        dataset_info = self.get_dataset_info(dataset)
        if dataset_info and 'download_hint' in dataset_info:
            hint_template = dataset_info['download_hint']
            split_files = self.get_split_files_str(dataset, split)
            hint = hint_template.format(split_files=split_files)
            return hint.strip()
        return None

    def get_splits(self, dataset: str) -> Optional[List[str]]:
        dataset_info = self.get_dataset_info(dataset)
        return list(dataset_info['splits'].keys()) if dataset_info else None

    def get_files(self, dataset: str, split: str) -> Optional[List[Dict]]:
        dataset_info = self.get_dataset_info(dataset)
        if dataset_info and split in dataset_info.get('splits', {}):
            file_infos = dataset_info['splits'][split]
            # parse file name from url and add to file_info
            for file_info in file_infos:
                # For DATASET_PROMPT format, file field already exists
                if 'file' in file_info:
                    file_name = file_info['file']
                # For private/public YAML, parse from URL
                elif 'url' in file_info:
                    file_name = file_info['url'].split('/')[-1]
                else:
                    # Use first check file as name or a default name
                    check_files = file_info.get('check_files', [])
                    file_name = check_files[0] if check_files else 'dataset_file'
                file_info['name'] = file_name
            return file_infos
        return None

    def get_urls(self, dataset: str, split: str) -> Optional[List[Dict]]:
        dataset_info = self.get_dataset_info(dataset)
        if dataset_info and split in dataset_info.get('splits', {}):
            return dataset_info['splits'][split]
        return None

    def verify_required_files(self, dataset_root: Path, dataset: str, split: str) -> List[str]:
        """Verify the existence of required files and directories for a given dataset and split."""
        dataset_info = self.get_dataset_info(dataset)
        if dataset_info and split in dataset_info.get('splits', {}):
            split_data = dataset_info['splits'][split]
            missing_files = []

            for item in split_data:
                check_files = item.get('check_files', [])
                for file in check_files:
                    file_path = dataset_root / file
                    if not file_path.exists():
                        missing_files.append(str(file_path))

            return missing_files
        return []


class DatasetStatus(enum.Enum):
    COMPLETE = enum.auto()
    INCOMPLETE = enum.auto()
    CORRUPTED = enum.auto()
    NOT_FOUND = enum.auto()
    IGNORE_CHECK = enum.auto()


def _create_completion_stamp(
    dataset_root: Path, dataset_name: str, split: str, config: DatasetConfig
):
    complete_file = dataset_root / ".ax_dataset_complete"
    md5_checksums = {}
    files = config.get_files(dataset_name, split)
    for file_info in files:
        md5_checksums[file_info['name']] = file_info.get('md5', '')

    existing_data = {}
    if complete_file.exists():
        with open(complete_file, 'r') as f:
            existing_data = json.load(f)

    timestamp = datetime.datetime.now().isoformat()
    dataset_key = f"{dataset_name}_{split}"

    if dataset_key not in existing_data:
        existing_data[dataset_key] = {"history": []}

    update_entry = {"timestamp": timestamp, "md5_checksums": md5_checksums}

    existing_data[dataset_key]["history"].append(update_entry)
    existing_data[dataset_key]["last_updated"] = timestamp

    with open(complete_file, 'w') as f:
        json.dump(existing_data, f, indent=2)


def _check_dataset_status(
    dataset_root: Path, dataset_name: str, split: str, config: DatasetConfig
) -> Tuple[DatasetStatus, str]:
    if not dataset_root.exists():
        return DatasetStatus.NOT_FOUND, f"Dataset directory {dataset_root} does not exist."

    # As we already have Imagenet locally, we don't want to redownload it.
    # If there is no local ImageNet, we will still download it due to lack of verified files.
    if dataset_name == 'ImageNet':
        return DatasetStatus.IGNORE_CHECK, "Ignore the check for ImageNet."
    elif dataset_name.startswith('Customer.'):
        return DatasetStatus.IGNORE_CHECK, "Ignore the check for Customer dataset."

    complete_file = dataset_root / ".ax_dataset_complete"
    if complete_file.exists():
        with open(complete_file, 'r') as f:
            stored_data = json.load(f)

        dataset_key = f"{dataset_name}_{split}"
        dataset_info = stored_data.get(dataset_key)
        if not dataset_info:
            return DatasetStatus.INCOMPLETE, f"No completion data found for {dataset_key}."

        history = dataset_info.get("history", [])
        if not history:
            return DatasetStatus.INCOMPLETE, f"No history found for {dataset_key}."

        latest_entry = history[-1]
        stored_md5s = latest_entry.get("md5_checksums", {})
        last_updated = dataset_info.get("last_updated", "Unknown")

        files = config.get_files(dataset_name, split)
        for file_info in files:
            config_md5 = file_info.get('md5')
            stored_md5 = stored_md5s.get(file_info['name'])

            if config_md5 and stored_md5 and config_md5 != stored_md5:
                return (
                    DatasetStatus.CORRUPTED,
                    f"MD5 mismatch for {file_info['name']}. File may have been modified or config updated.",
                )

        update_count = len(history)
        return (
            DatasetStatus.COMPLETE,
            f"All files present and MD5 checksums match the configuration. Last updated: {last_updated}. Total updates: {update_count}",
        )

    contains = [p.name for p in dataset_root.iterdir() if p.name not in ('.DS_Store', '__MACOSX')]
    if not contains:
        return DatasetStatus.INCOMPLETE, f"Dataset directory is empty."
    else:
        return (
            DatasetStatus.INCOMPLETE,
            f"Dataset directory contains: {', '.join(contains)}. No completion stamp found.",
        )


def _download_dataset(dataset_dir: Path, dataset_name: str, split: str, config: DatasetConfig):
    files = config.get_files(dataset_name, split)
    if not files:
        raise ValueError(f"No files found for dataset '{dataset_name}' split '{split}'.")

    dataset_dir.mkdir(parents=True, exist_ok=True)

    for file_info in files:
        url = file_info['url']
        drop_dirs = file_info['drop_dirs']
        md5 = file_info.get('md5')
        sub_dir = file_info.get('sub_dir', '')
        dir = dataset_dir / sub_dir if sub_dir else dataset_dir
        file_path = dir / file_info['name']
        if dataset_name == 'ImageNet':
            # special case for ImageNet, as we let torchvision do the organization
            if file_info['name'] in ['ILSVRC2012_img_val.tar', 'ILSVRC2012_img_train.tar']:
                file_path = dir.parent / file_info['name']
            utils.download(url, file_path, md5)
            if split == 'subset' and file_info['name'] == 'imagenet_subset.zip':
                utils.extract(file_path, drop_dirs=drop_dirs)
                file_path.unlink()
        else:
            utils.download_and_extract_asset(
                url, file_path, md5, drop_dirs=drop_dirs, delete_archive=True
            )


def _print_hint(message: str):
    from rich.console import Console
    from rich.panel import Panel
    from rich.text import Text

    lines = message.split('\n')
    formatted_text = Text()
    for line in lines:
        if line.strip():
            formatted_text.append(line.strip() + "\n", style="bold red")
        else:
            formatted_text.append("\n")

    panel = Panel(formatted_text, title="[bold red]HINT", expand=False, border_style="red")
    Console().print(panel)


def check_and_download_dataset(
    dataset_name: str, data_root_dir: Path, split: str = 'val', is_private: bool = True
):
    '''
    Datasets here are with license concern. We provide ability to download and verify the dataset for internal development; for external users, we provide download hint to guide them to download the dataset.
    '''
    dataset_yaml_name = (
        DatasetYamlFile.DATASET_PRIVATE if is_private else DatasetYamlFile.DATASET_PUBLIC
    )
    url_config = env.framework / f'ax_datasets/{dataset_yaml_name.value}'
    if not url_config.exists():
        if is_private:
            prompt = env.framework / f'ax_datasets/{DatasetYamlFile.DATASET_PROMPT.value}'
            if prompt.exists():
                url_config = prompt
                dataset_yaml_name = DatasetYamlFile.DATASET_PROMPT
                LOG.trace(f"Using {DatasetYamlFile.DATASET_PROMPT.value}")
            else:
                raise FileNotFoundError(
                    f"Neither {dataset_yaml_name.value} nor {DatasetYamlFile.DATASET_PROMPT.value} found in {url_config.parent}"
                )
        else:
            raise FileNotFoundError(f"{dataset_yaml_name.value} not found in {url_config.parent}")

    config = DatasetConfig(url_config, dataset_yaml_name)
    dataset_info = config.get_dataset_info(dataset_name)

    if not dataset_info:
        LOG.error(f"Dataset '{dataset_name}' not found in configuration.")
        return

    if env.s3_available.lower() == 'subset' and 'subset' in dataset_info.get('splits', {}):
        split = 'subset'  # This is for CI and platform validation
    # TODO: when release, we check if assets of public dataset are available in S3 public bucket

    dataset_status, status_message = _check_dataset_status(
        data_root_dir, dataset_name, split, config
    )

    if dataset_status == DatasetStatus.COMPLETE:
        missing_files = config.verify_required_files(data_root_dir, dataset_name, split)
        if missing_files:
            LOG.error(
                f"Dataset '{dataset_name}' is complete but has missing files: "
                f"{', '.join(missing_files)}. Try to download the dataset again."
            )
            dataset_status = DatasetStatus.INCOMPLETE
        else:
            LOG.trace(
                f"Dataset '{dataset_name}' is already complete and verified. Skipping download."
            )
            return
    else:
        LOG.debug(f"Dataset '{dataset_name}' status: {dataset_status.value}\n{status_message}")

    if dataset_status in [DatasetStatus.INCOMPLETE, DatasetStatus.NOT_FOUND]:
        if env.s3_available == '0':
            # dataset with license concern in a customer environment
            hint = config.get_download_hint(dataset_name, split)
            full_message = f"{status_message}\n\n{hint}"
            _print_hint(full_message)
            raise RuntimeError("Please follow the hint to download the dataset.")
        else:
            _download_dataset(data_root_dir, dataset_name, split, config)
    elif dataset_status == DatasetStatus.CORRUPTED:
        LOG.warning(f"Dataset '{dataset_name}' appears to be corrupted. Redownloading...")
        _download_dataset(data_root_dir, dataset_name, split, config)

    _create_completion_stamp(data_root_dir, dataset_name, split, config)
    LOG.info(
        f"Dataset '{dataset_name}' split '{split}' downloaded successfully to {data_root_dir}"
    )


def check_dataset_directory(dataset_root: Path) -> Tuple[bool, str]:
    '''
    Basic check if the dataset directory exists and is a valid dataset directory. This is used for 3rd party dataset validation.
    '''
    text = []
    text.append(
        f'''
        Note: {dataset_root} :'''
    )

    if dataset_root.is_symlink():
        resolved_path = dataset_root.resolve()
        broken = 'broken ' if not resolved_path.exists() else ''
        text.append(f"\tis a {broken}symlink to {resolved_path}")
        dataset_root = resolved_path  # Use resolved path for further checks

    if not dataset_root.exists():
        text.append(f"\tdoes not exist")
        return False, '\n'.join(text)

    if not dataset_root.is_dir():
        text.append(f"\tis not a directory")
        return False, '\n'.join(text)
    else:
        if not any(dataset_root.iterdir()):
            text.append(f"\tis an empty directory")
            return False, '\n'.join(text)
    return True, '\n'.join(text)


def get_image_size(image: Union[np.ndarray, Image.Image]) -> Tuple[int, int]:
    '''
    Get the image size in (height, width) format.
    '''
    if isinstance(image, np.ndarray):
        return image.shape[:2]
    else:
        return image.size[::-1]

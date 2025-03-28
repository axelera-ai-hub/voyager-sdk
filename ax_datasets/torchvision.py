# Axelera Torchvision datasets
# Copyright Axelera AI, 2025
from __future__ import annotations

import os
from pathlib import Path
import pickle
import textwrap
from typing import Any, Tuple

import torch
import torchvision

from axelera import types
from axelera.app import config, data_utils, eval_interfaces, logging_utils
import axelera.app.yaml as YAML
from axelera.app.yaml import MapYAMLtoFunction

LOG = logging_utils.getLogger(__name__)


def _safe_unlink(split: str, root: Path):
    files = [root / 'ILSVRC2012_devkit_t12.tar.gz']
    if split == 'train':
        files.append(root / 'ILSVRC2012_img_train.tar')
    elif split == 'val':
        files.append(root / 'ILSVRC2012_img_val.tar')

    if config.env.s3_available != '0':
        for file in files:
            if file.exists():
                file.unlink()
    else:  # we don't proactively delete files for customer env
        delete_files = [file for file in files if file.exists()]
        if delete_files:
            LOG.warning(f"You are safe to delete: {delete_files}")


class ImageNet(torchvision.datasets.ImageNet):
    def __init__(self, transform, root, args):
        yargs = MapYAMLtoFunction(
            supported=['split'],
            required=[],
            defaults={'split': 'train'},
            named_args=['split'],
            attribs=args,
        )
        split = yargs.get_arg('split')
        self.is_subset = config.env.s3_available.lower() == 'subset'
        if self.is_subset:
            root = root / 'subset'
        data_utils.check_and_download_dataset('ImageNet', root, split)

        kwargs = yargs.get_kwargs()
        super().__init__(root, transform=transform, split=split, **kwargs)
        _safe_unlink(split, root)

    def parse_archives(self) -> None:
        """Overwrite to handle subset split"""
        from torchvision.datasets.imagenet import (
            META_FILE,
            parse_devkit_archive,
            parse_train_archive,
            parse_val_archive,
        )
        from torchvision.datasets.utils import check_integrity

        if not check_integrity(os.path.join(self.root, META_FILE)):
            parse_devkit_archive(self.root)

        if not self.is_subset:
            if not os.path.isdir(self.split_folder):
                if self.split == "train":
                    parse_train_archive(self.root)
                elif self.split == "val":
                    parse_val_archive(self.root)

    def __getitem__(self, index: int) -> Tuple[Any, Any, str]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target, file_name) where target is class_index of the target class.
        """
        img_path, target = self.imgs[index]
        img = self.loader(img_path)
        target = self.targets[index]

        # Apply transformations if any
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # Get the file name
        file_name = Path(img_path).stem
        return img, target, file_name


class MNIST(torchvision.datasets.MNIST):
    def __init__(self, transform, root, args):
        yargs = MapYAMLtoFunction(
            supported=['train', 'download'],
            required=[],
            defaults={'train': True, 'download': True},
            named_args=['train'],
            attribs=args,
        )
        train = yargs.get_arg('train')
        download = yargs.get_arg('download')
        super().__init__(root, train, transform, target_transform=None, download=download)


class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, transform, root, args):
        yargs = MapYAMLtoFunction(
            supported=['train', 'download'],
            required=[],
            defaults={'train': True, 'download': True},
            named_args=['train'],
            attribs=args,
        )
        train = yargs.get_arg('train')
        download = yargs.get_arg('download')
        super().__init__(root, train, transform, target_transform=None, download=download)


class ImageFolder(torchvision.datasets.DatasetFolder):
    def __init__(self, transform, root, args):
        supported_args = {
            'cal': ['cal_data', 'cal_index_pkl', 'is_one_indexed'],
            'val': ['val_data', 'val_index_pkl', 'is_one_indexed'],
        }
        required_args = {'cal': ['cal_data'], 'val': ['val_data']}

        split = args['split']
        yargs = MapYAMLtoFunction(
            supported=supported_args[split] + ['split'],
            required=required_args[split] + ['split'],
            defaults={'split': 'train'},
            named_args=['split'],
            attribs=args,
        )

        data_dir = root / yargs.get_arg(f'{split}_data')
        index_pkl_path = None
        if split == 'cal' and 'cal_index_pkl' in args:
            index_pkl_path = args['cal_index_pkl']
        elif split == 'val' and 'val_index_pkl' in args:
            index_pkl_path = args['val_index_pkl']

        base_dataset = torchvision.datasets.ImageFolder(data_dir, transform=None)
        if index_pkl_path:
            index = self._load_index_pkl(root / index_pkl_path)
            self.subset = torch.utils.data.Subset(base_dataset, index) if index else base_dataset
        else:
            self.subset = base_dataset

        self.transform = transform
        self.is_one_indexed = args.get('is_one_indexed', False)

    def __getitem__(self, index):
        x, y = self.subset[index]
        if x.mode == 'L':
            x = x.convert('RGB')
        if self.transform:
            x = self.transform(x)
        if self.is_one_indexed:
            y -= 1
        return x, y

    def __len__(self):
        return len(self.subset)

    def _load_index_pkl(self, index_pkl_path):
        if index_pkl_path.exists():
            with open(index_pkl_path, 'rb') as f:
                indices = pickle.load(f)
                return indices
        raise FileNotFoundError(f"No index_pkl from {index_pkl_path}")


class VOCDetection(torchvision.datasets.VOCDetection):
    def __init__(self, transform, root, args):
        yargs = MapYAMLtoFunction(
            supported=['year', 'image_set', 'download'],
            required=[],
            defaults={'year': '2011', 'image_set': 'train', 'download': True},
            named_args=['year', 'image_set', 'download'],
            attribs=args,
        )
        year = yargs.get_arg('year')
        image_set = yargs.get_arg('image_set')
        download = yargs.get_arg('download')
        super().__init__(
            root, year=year, image_set=image_set, download=download, transform=transform
        )


class LFWPairs(torchvision.datasets.LFWPairs):
    def __init__(self, transform, root, args):
        yargs = MapYAMLtoFunction(
            supported=['image_set', 'download', 'split'],
            required=[],
            defaults={'image_set': 'funneled', 'download': True, 'split': 'test'},
            named_args=['split'],
            attribs=args,
        )
        image_set = yargs.get_arg('image_set')
        download = yargs.get_arg('download')
        super().__init__(root, transform=transform, download=download, image_set=image_set)

    def __getitem__(self, index: int) -> Tuple[Any, Any, int, str]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image1, image2, target, img_id) where target is `0` for different indentities and `1` for same identities.
            img_id is the name of the first image.
        """
        img_id1, img_id2 = self.data[index]
        img1, img2 = self._loader(img_id1), self._loader(img_id2)
        target = self.targets[index]
        img_id = Path(img_id1).parent.name

        if self.transform is not None:
            img1, img2 = self.transform(img1), self.transform(img2)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img1, img2, target, img_id


class LFWPeople(torchvision.datasets.LFWPeople):
    def __init__(self, transform, root, args):
        yargs = MapYAMLtoFunction(
            supported=['image_set', 'download', 'split'],
            required=[],
            defaults={'image_set': 'funneled', 'download': True, 'split': 'test'},
            named_args=['split'],
            attribs=args,
        )
        image_set = yargs.get_arg('image_set')
        download = yargs.get_arg('download')
        super().__init__(root, transform=transform, download=download, image_set=image_set)


class Caltech101(torchvision.datasets.Caltech101):
    def __init__(self, transform, root, args):
        yargs = MapYAMLtoFunction(
            supported=['download'],
            required=[],
            defaults={'download': False},
            named_args=[],
            attribs=args,
        )
        download = yargs.get_arg('download')
        super().__init__(root, transform=transform, download=download)


def _add_download_info(
    dataset_root: Path,
    existing_text: str,
    dataset_split: types.DatasetSplit = types.DatasetSplit.VAL,
):
    text = [existing_text.strip().replace('. ', '.\n')]
    if 'caltech101' in str(dataset_root).lower():
        text.append(
            textwrap.dedent(
                f'''
                To download the dataset please execute the following command:

                    python ax_models/tutorials/resnet34_caltech101/tutorial_resnet34_caltech101.py --download

                (See the SDK Quick Start Guide for further information on using the Caltech101 dataset)'''
            )
        )
    else:
        text.append(
            textwrap.dedent(
                f'''
                Failed to prepare the dataset from the split {dataset_split}.'''
            )
        )
    return '\n'.join(text)


class TorchvisionDataAdapter(types.DataAdapter):
    """Data adapter for classification task using the Imagenet dataset.

    Args:
        root (str): Root directory of the dataset.
        task (str): Task of the dataset, either 'val' or 'test'.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        batch_size (int, optional): How many samples per batch to load.

    examples:
        ImageNet-1K:
            class: TorchvisionDataAdapter
            class_path: $AXELERA_FRAMEWORK/ax_datasets/torchvision.py
            data_dir_name: ImageNet
            labels_path: $AXELERA_FRAMEWORK/ax_datasets/labels/imagenet1000_clsidx_to_labels.txt
        LFWTorchvision:
            class: TorchvisionDataAdapter
            class_path: $AXELERA_FRAMEWORK/ax_datasets/torchvision.py
            dataset_name: LFWPeople
            data_dir_name: LFW
            image_set: deepfunneled
    """

    def __init__(self, dataset_config, model_info):
        YAML.require_attribute(dataset_config, 'data_dir_name')
        data_dir_name = dataset_config['data_dir_name']
        dataset_name = dataset_config.get('dataset_name', data_dir_name)

        if dataset_name not in globals():
            LOG.debug(f"Dataset class '{dataset_name}' is not defined in this module.")
            LOG.info("Assuming it's a custom dataset with ImageFolder format.")
            dataset_class = "ImageFolder"
        else:
            dataset_class = dataset_name
        self.DatasetClass = globals()[dataset_class]

    def create_calibration_data_loader(self, transform, root, batch_size, **kwargs):
        if repr_dataloader := self.check_representative_images(transform, batch_size, **kwargs):
            return repr_dataloader

        if self.DatasetClass == ImageFolder:
            kwargs['split'] = 'cal'
        else:
            kwargs['split'] = 'train'
        return torch.utils.data.DataLoader(
            self._get_dataset_class(transform, root, kwargs),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
            num_workers=0,
        )

    def create_validation_data_loader(self, root, target_split, **kwargs):
        kwargs['split'] = 'val'
        return torch.utils.data.DataLoader(
            self._get_dataset_class(None, root, kwargs),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=0,
        )

    def reformat_for_calibration(self, batched_data: Any):
        return (
            batched_data
            if self.use_repr_imgs
            else torch.stack([data[0] for data in batched_data], 0)
        )

    def reformat_for_validation(self, batched_data: Any):
        return self._format_measurement_data(batched_data)

    def _get_dataset_class(self, transform, root, kwargs):
        try:
            return self.DatasetClass(transform, root, kwargs)
        except Exception as e:
            raise logging_utils.UserError(
                _add_download_info(
                    root,
                    f"Dataset class '{self.DatasetClass}' failed to initialize: {e}",
                    kwargs.get('split'),
                )
            ) from None

    def _format_measurement_data(self, batched_data: Any) -> list[types.FrameInput]:
        return [
            types.FrameInput.from_image(
                img=img,
                color_format=types.ColorFormat.RGB,
                ground_truth=eval_interfaces.ClassificationGroundTruthSample(class_id=target),
                img_id=optional[0] if optional else '',
            )
            for img, target, *optional in batched_data
        ]

    def evaluator(
        self, dataset_root, dataset_config, model_info, custom_config, pair_validation=False
    ):
        if pair_validation:
            raise ValueError("Unexpected pair_validation flag")
        from ax_evaluators.classification import ClassificationEvaluator

        return ClassificationEvaluator()


class TorchvisionPairValidationDataAdapter(TorchvisionDataAdapter):
    """Data adapter for pair validation task using Torchvision dataset APIs.

    example:
      LFWTorchvision:
        class: TorchvisionPairValidationDataAdapter
        class_path: $AXELERA_FRAMEWORK/ax_datasets/torchvision.py
        dataset_name: LFWPairs
        data_dir_name: LFW
        image_set: deepfunneled
    """

    def _format_measurement_data(self, batched_data: Any) -> list[types.FrameInput]:
        return [
            types.FrameInput.from_image_pair(
                img1=image1,
                img2=image2,
                color_format=types.ColorFormat.RGB,
                ground_truth=eval_interfaces.PairValidationGroundTruthSample(the_same=the_same),
                img_id=img_id,
            )
            for image1, image2, the_same, img_id in batched_data
        ]

    def evaluator(
        self, dataset_root, dataset_config, model_info, custom_config, pair_validation=False
    ):
        if pair_validation:
            from ax_evaluators.pair_validation import PairValidationEvaluator

            return PairValidationEvaluator(**custom_config)
        else:
            raise NotImplementedError(
                "Pair validation data loader expects pair_validation as True"
            )

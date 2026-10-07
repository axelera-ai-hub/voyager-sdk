# Copyright Axelera AI, 2025
from __future__ import annotations

import os

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from axelera import types
from axelera.app import data_utils, eval_interfaces, logging_utils


def h5_loader(path):
    h5f = h5py.File(path, "r")
    rgb = np.array(h5f['rgb'])
    rgb = np.transpose(rgb, (1, 2, 0))
    depth = np.array(h5f['depth'])
    return rgb, depth


class NYUDepthV2(Dataset):
    def __init__(self, path_to_data, transform=None):
        self.h5_files = [os.path.join(path_to_data, fname) for fname in os.listdir(path_to_data)]
        self.transform = transform

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):

        rgb, depth = h5_loader(self.h5_files[idx])

        if self.transform is not None:
            rgb = self.transform(rgb)
            depth = cv2.resize(depth, dsize=(rgb.size()[1], rgb.size()[2]))

        return rgb, depth


class DepthEstimationDataAdapter(types.DataAdapter):
    def __init__(self, dataset_config, model_info):
        self.dataset_config = dataset_config
        self.model_info = model_info

    def reformat_for_calibration(self, batched_data: Any):
        return (
            batched_data
            if self.use_repr_imgs
            else torch.stack([data[0] for data in batched_data], 0)
        )

    def create_calibration_data_loader(self, transform, root, batch_size, **kwargs):
        data_utils.check_and_download_dataset(
            dataset_name='NYUDepthV2',
            data_root_dir=root,
            split='train',
            is_private=False,
        )

        assert 'cal_data' in kwargs, "cal_data is required in the YAML dataset config"

        return torch.utils.data.DataLoader(
            NYUDepthV2(os.path.join(root, kwargs['cal_data']), transform),
            batch_size=batch_size,
            shuffle=True,
            generator=kwargs.get('generator'),
            collate_fn=lambda x: x,
            num_workers=0,
        )

    def create_validation_data_loader(self, root, target_split, **kwargs):
        data_utils.check_and_download_dataset(
            dataset_name='NYUDepthV2',
            data_root_dir=root,
            split='val',
            is_private=False,
        )

        assert 'val_data' in kwargs, "val_data is required in the YAML dataset config"

        return torch.utils.data.DataLoader(
            NYUDepthV2(os.path.join(root, kwargs['val_data'])),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=0,
        )

    def reformat_for_validation(self, batched_data):
        return [
            types.FrameInput.from_image(
                img=rgb,
                color_format=types.ColorFormat.RGB,
                ground_truth=eval_interfaces.ImageSample(img=depth),
                img_id='',
            )
            for rgb, depth in batched_data
        ]

    def evaluator(
        self, dataset_root, dataset_config, model_info, custom_config, pair_validation=False
    ):

        from ax_evaluators.depth_estimation import DepthEstimationEvaluator

        return DepthEstimationEvaluator(
            align=dataset_config.get('eval_align', 'none'),
            min_depth=float(dataset_config.get('eval_min_depth', 1e-3)),
            max_depth=float(dataset_config.get('eval_max_depth', 10.0)),
        )

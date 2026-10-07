# Copyright Axelera AI, 2025
import os

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from ax_models.tutorials.grayscale.beans_dataset import BeansDataset
from axelera import types
from axelera.app import eval_interfaces


class CustomDataAdapter(types.DataAdapter):
    def __init__(self, dataset_config, model_info):
        # self.is_grayscale = model_info.input_color_format == types.ColorFormat.GRAYSCALE
        pass

    def create_calibration_data_loader(self, transform, root, batch_size, **kwargs):
        return DataLoader(
            BeansDataset(
                load_dataset("AI-Lab-Makerere/beans", cache_dir=root)["train"],
                transform=transform,
            ),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
            num_workers=4,
        )

    def reformat_for_calibration(self, batched_data):
        images = []
        for data, _ in batched_data:
            images.append(data)

        return torch.stack(images, 0)

    def create_validation_data_loader(self, root, target_split, **kwargs):
        if target_split == 'train':
            split = 'train'
        elif target_split == 'val':
            split = 'validation'
        elif target_split == 'test':
            split = 'test'

        return DataLoader(
            BeansDataset(
                load_dataset("AI-Lab-Makerere/beans", cache_dir=root)[split],
            ),
            batch_size=1,
            shuffle=True,
            collate_fn=lambda x: x,
            num_workers=4,
        )

    def reformat_for_validation(self, batched_data):
        # Create FrameInput objects for the evaluator
        return [
            types.FrameInput.from_image(
                img=img,
                color_format=types.ColorFormat.GRAY,
                ground_truth=eval_interfaces.ClassificationGroundTruthSample(class_id=target),
                img_id=f'img_{i}',
            )
            for i, (img, target) in enumerate(batched_data)
        ]

    def evaluator(
        self, dataset_root, dataset_config, model_info, custom_config, pair_validation=False
    ):
        from ax_evaluators.classification import ClassificationEvaluator

        return ClassificationEvaluator(top_k=1)

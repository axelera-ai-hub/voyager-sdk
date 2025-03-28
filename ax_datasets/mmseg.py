# Copyright Axelera AI, 2024
from __future__ import annotations

import dataclasses
import importlib
from pathlib import Path
import subprocess
import typing

import PIL.Image

from axelera import types
from axelera.app import data_utils, logging_utils, utils
from axelera.app.torch_utils import torch

if typing.TYPE_CHECKING:
    from mmengine.structures import BaseDataElement

LOG = logging_utils.getLogger(__name__)


class MMSegSample(types.BaseEvalSample):
    """
    Data element for semantic segmentation evaluations using MMSegmentation evaluators.
    """

    data_sample: BaseDataElement

    def __init__(self, data_sample: BaseDataElement):
        self.data_sample = data_sample

    @property
    def data(self) -> typing.Any:
        return self.data_sample


class MMSegDataAdapter(types.DataAdapter):
    """Data adapter for semantic segmentation task using the MMSeg dataset."""

    def __init__(self, dataset_config: dict, model_info: types.ModelInfo):
        self._evaluator = None
        # HACK of "from mmseg.datasets import *" and "from mmseg.evaluation import *"
        # TODO: find a better way to import all the datasets and transforms
        for module_name in ['mmseg.datasets', 'mmseg.evaluation']:
            module = importlib.import_module(module_name)
            all_attributes = dir(module)
            imported_objects = {}
            for attribute in all_attributes:
                if not attribute.startswith('__'):
                    imported_objects[attribute] = getattr(module, attribute)

    def create_calibration_data_loader(self, transform, root, batch_size, **kwargs):
        if repr_dataloader := self.check_representative_images(transform, batch_size, **kwargs):
            self.transform = transform
            return repr_dataloader

        return torch.utils.data.DataLoader(
            self._get_dataset_class(transform, root, 'val', kwargs),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
            num_workers=0,
        )

    def create_validation_data_loader(self, root, target_split='test', **kwargs):
        return torch.utils.data.DataLoader(
            # following MMSeg's logic, we use val set for test
            # TODO: find a better way to handle this
            self._get_dataset_class(None, root, 'val', kwargs),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=0,
        )

    def reformat_for_calibration(self, batched_data: typing.Any):
        return self._format_calibration_data(batched_data)

    def reformat_for_validation(self, batched_data: typing.Any):
        return self._format_measurement_data(batched_data)

    def _get_dataset_class(self, transform, data_root, split, kwargs) -> torch.utils.data.Dataset:
        data_utils.check_and_download_dataset('MMLab-Cityscapes', data_root, split)
        # check if MMLab's cityscapes dataset is converted
        # check labelTrainIds.png exists
        if split == 'test':
            check_file = "gtFine/test/bonn/bonn_000019_000019_gtFine_labelTrainIds.png"
        else:
            check_file = "gtFine/val/frankfurt/frankfurt_000000_000294_gtFine_labelTrainIds.png"
        if (data_root / check_file).exists():
            LOG.trace("MMLab's cityscapes dataset is already converted")
        else:
            LOG.info("MMLab's cityscapes dataset is not converted, converting...")
            import site
            import sys

            site_packages_path = next(
                (path for path in site.getsitepackages() if path.endswith('site-packages')), None
            )
            if site_packages_path is None:
                raise ValueError("Failed to find site-packages path")
            mmseg_converter_path = (
                Path(site_packages_path) / 'mmseg/.mim/tools/dataset_converters/cityscapes.py'
            )
            try:
                subprocess.run(
                    [sys.executable, mmseg_converter_path, data_root, "--nproc", "8"],
                    encoding='utf8',
                    check=True,
                    capture_output=True,
                )
            except subprocess.CalledProcessError as e:
                # Check if the error is due to missing directories
                if "No such file or directory" in str(e.stderr):
                    if (
                        split in ['train', 'val', 'test']
                        and (data_root / "gtFine" / split).exists()
                    ):
                        LOG.trace("Directory exists, continuing...")
                else:
                    LOG.error("Command output: %s", e.stdout)
                    raise RuntimeError(f"Failed to convert Cityscapes dataset: {e.stderr}") from e
        cfg = self.get_shared_param('cfg')
        from mmengine.evaluator import Evaluator

        from mmseg.registry import DATASETS

        self.transform = transform
        evaluator = None
        # mmseg has a weird dependency on the TRANSFORMS registry, so we need to build it after init_model
        # TODO: seems like we can update dataset.pipeline by transform, it's a Compose
        # but it can break the logic; maybe we only need LoadImageFromFile, LoadAnnotations;
        # not sure if the "resize" in the pipeline can affect accuracy
        if split == 'train':
            cfg.train_dataloader.dataset.data_root = data_root
            dataset = DATASETS.build(cfg.train_dataloader.dataset)
        elif split == 'val':
            cfg.val_dataloader.dataset.data_root = data_root
            dataset = DATASETS.build(cfg.val_dataloader.dataset)
            # Bind evaluator to dataset because this evaluator works with the
            # specific format generated by this dataset
            evaluator = Evaluator(cfg.val_evaluator)
        elif split == 'test':
            cfg.test_dataloader.dataset.data_root = data_root
            dataset = DATASETS.build(cfg.test_dataloader.dataset)
            # TODO: find a better way to pass this evaluator to AxEvaluator
            evaluator = Evaluator(cfg.test_evaluator)
        else:
            raise ValueError(f"Unsupported split: {split}")

        if evaluator:
            if hasattr(dataset, 'metainfo'):
                # follow https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py#L328
                # the meta has 'classes', 'palette'
                evaluator.dataset_meta = dataset.metainfo

            EvaluatorClass = utils.import_from_module('ax_evaluators.mmlab', 'MMSegEvaluator')
            self._evaluator_instance = EvaluatorClass(evaluator, dataset_type=type(dataset))
        return dataset

    def _convert_color_format(self, img_tensor):
        if self.get_shared_param('color_format') == 'RGB':  # BGR to RGB
            return img_tensor[[2, 1, 0], :, :]
        return img_tensor

    def _format_calibration_data(self, batched_data: typing.typing.Any) -> list:
        if self.transform is None:
            raise ValueError("Transform should not be None for calibration")

        new_batched_data = []
        # TODO: consider moving the following 2 lines into Input Operator and
        # have input operator in the transform
        for data in batched_data:
            if isinstance(data, dict):  # mmseg returns a dict
                image = self._convert_color_format(data['inputs']).numpy().transpose(1, 2, 0)
                image = PIL.Image.fromarray(image)
                new_batched_data.append(self.transform(image))
            elif isinstance(data, torch.Tensor):
                # already preprocessed
                new_batched_data.append(data)
            else:
                raise ValueError(f"Unsupported data type: {type(data)}")

        return torch.stack(new_batched_data, 0)

    def _format_measurement_data(self, batched_data: typing.Any) -> list[types.FrameInput]:
        def as_ground_truth(d):
            return MMSegSample(data_sample=d['data_samples']) if 'data_samples' in d else None

        def as_frame_input(d):
            return types.FrameInput.from_image(
                img=d['inputs'].numpy().transpose(1, 2, 0),
                color_format=types.ColorFormat.BGR,
                ground_truth=as_ground_truth(d),
                img_id=Path(d['data_samples'].img_path).stem,
            )

        return [as_frame_input(d) for d in batched_data]

    def evaluator(
        self, dataset_root, dataset_config, model_info, custom_config, pair_validation=False
    ):
        if self._evaluator_instance:
            return self._evaluator_instance
        raise ValueError("The adapter does not have an assigned evaluator.")

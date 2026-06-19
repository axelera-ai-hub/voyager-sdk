# Copyright Axelera AI, 2026
"""In-house Cityscapes semantic-segmentation data adapter.

No dependency on mmsegmentation / mmengine / mmcv / mmdeploy /
cityscapesscripts. The labelId-to-trainId mapping is the canonical
Cityscapes 19-class evaluation mapping from Cordts et al. "The
Cityscapes Dataset for Semantic Urban Scene Understanding", CVPR
2016, section 3.2. The same mapping appears in the BSD-licensed
cityscapesscripts/helpers/labels.py reference implementation; we
reimplement it inline so the framework has no third-party label-helper
dependency.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, List, Tuple

import numpy as np
import PIL.Image

from axelera import types
from axelera.app import data_utils, logging_utils
from axelera.app.eval_interfaces import SemanticSegmentationGroundTruthSample
from axelera.app.torch_utils import torch

LOG = logging_utils.getLogger(__name__)


# Cityscapes labelId -> trainId, per Cordts et al. 2016 sec. 3.2.
# 33 raw labelIds collapse to 19 evaluation trainIds + ignore (255).
LABEL_ID_TO_TRAIN_ID = {
    0: 255,
    1: 255,
    2: 255,
    3: 255,
    4: 255,
    5: 255,
    6: 255,
    7: 0,  # road
    8: 1,  # sidewalk
    9: 255,
    10: 255,
    11: 2,  # building
    12: 3,  # wall
    13: 4,  # fence
    14: 255,
    15: 255,
    16: 255,
    17: 5,  # pole
    18: 255,
    19: 6,  # traffic light
    20: 7,  # traffic sign
    21: 8,  # vegetation
    22: 9,  # terrain
    23: 10,  # sky
    24: 11,  # person
    25: 12,  # rider
    26: 13,  # car
    27: 14,  # truck
    28: 15,  # bus
    29: 255,
    30: 255,
    31: 16,  # train
    32: 17,  # motorcycle
    33: 18,  # bicycle
}

NUM_CLASSES = 19
IGNORE_INDEX = 255


def _build_remap_lut() -> np.ndarray:
    """Build a 256-entry lookup table mapping raw labelIds to trainIds.

    Any labelId not in LABEL_ID_TO_TRAIN_ID defaults to IGNORE_INDEX so
    unknown / future labels do not silently pollute the metric.
    """
    lut = np.full(256, IGNORE_INDEX, dtype=np.uint8)
    for label_id, train_id in LABEL_ID_TO_TRAIN_ID.items():
        lut[label_id] = train_id
    return lut


_REMAP_LUT = _build_remap_lut()


def labelid_to_trainid(label_mask: np.ndarray) -> np.ndarray:
    """Vectorized labelId -> trainId conversion via lookup table.

    Args:
        label_mask: uint8 array of raw labelIds.

    Returns:
        uint8 array of same shape with trainIds in [0, 19) and
        IGNORE_INDEX (255) elsewhere.
    """
    return _REMAP_LUT[label_mask]


def _list_val_pairs(data_root: Path) -> List[Tuple[Path, Path, str]]:
    """List (leftImg8bit_path, gtFine_labelIds_path, image_id) for val."""
    images_root = data_root / 'leftImg8bit' / 'val'
    labels_root = data_root / 'gtFine' / 'val'
    pairs: List[Tuple[Path, Path, str]] = []
    if not images_root.is_dir():
        return pairs
    for city_dir in sorted(images_root.iterdir()):
        if not city_dir.is_dir():
            continue
        for img_path in sorted(city_dir.glob('*_leftImg8bit.png')):
            stem = img_path.name[: -len('_leftImg8bit.png')]
            gt_path = labels_root / city_dir.name / f'{stem}_gtFine_labelIds.png'
            if gt_path.exists():
                pairs.append((img_path, gt_path, stem))
    return pairs


try:
    _DatasetBase = torch.utils.data.Dataset
except ImportError:
    _DatasetBase = object


class _CityscapesValDataset(_DatasetBase):
    """Yields (preprocessed_image_tensor, ground_truth_sample) tuples."""

    def __init__(self, pairs: List[Tuple[Path, Path, str]], transform=None):
        self._pairs = pairs
        self._transform = transform

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, idx: int):
        img_path, gt_path, image_id = self._pairs[idx]
        with PIL.Image.open(img_path) as img:
            img = img.convert('RGB')
            raw_image_size = (img.height, img.width)
            if self._transform is not None:
                image_tensor = self._transform(img)
            else:
                image_tensor = np.array(img)
        with PIL.Image.open(gt_path) as gt:
            gt_mask = labelid_to_trainid(np.array(gt, dtype=np.uint8))
        return image_tensor, SemanticSegmentationGroundTruthSample(
            gt_mask=gt_mask,
            img_id=image_id,
            raw_image_size=raw_image_size,
        )


class _CityscapesCalibDataset(_DatasetBase):
    """Iterates calibration images from a folder (e.g. coco2017_repr400)."""

    _SUFFIXES = ('.jpg', '.jpeg', '.png')

    def __init__(self, root: Path, transform=None):
        self._files = sorted(p for p in root.iterdir() if p.suffix.lower() in self._SUFFIXES)
        self._transform = transform
        if not self._files:
            raise ValueError(f'No calibration images found under {root}')

    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int):
        img_path = self._files[idx]
        with PIL.Image.open(img_path) as img:
            img = img.convert('RGB')
            if self._transform is not None:
                return self._transform(img)
            return np.array(img)


class CityscapesSemSegDataAdapter(types.DataAdapter):
    """Data adapter for Cityscapes semantic segmentation, no mmlab deps."""

    def __init__(self, dataset_config: dict, model_info: types.ModelInfo):
        self.dataset_config = dataset_config
        self.model_info = model_info
        self._evaluator_instance = None

    def create_calibration_data_loader(self, transform, root, batch_size, **kwargs):
        repr_imgs_path = self.dataset_config.get('repr_imgs_dir_path') or kwargs.get(
            'repr_imgs_dir_path'
        )
        if repr_imgs_path is None:
            raise ValueError(
                "CityscapesSemSegDataAdapter requires 'repr_imgs_dir_path' in the dataset "
                "config (typically pointing at data/coco2017_repr400)"
            )
        repr_imgs_path = Path(repr_imgs_path)
        if not repr_imgs_path.is_dir():
            raise FileNotFoundError(
                f'repr_imgs_dir_path {repr_imgs_path} does not exist or is not a directory'
            )
        dataset = _CityscapesCalibDataset(repr_imgs_path, transform=transform)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=kwargs.get('generator'),
            collate_fn=lambda x: torch.stack([torch.as_tensor(t) for t in x], 0),
            num_workers=0,
        )

    def create_validation_data_loader(self, root, target_split='val', **kwargs):
        data_root = Path(root) / self.dataset_config.get('data_dir_name', 'cityscapes')
        data_utils.check_and_download_dataset(
            dataset_name='Cityscapes',
            data_root_dir=data_root,
            split='val',
            is_private=True,
        )
        pairs = _list_val_pairs(data_root)
        if not pairs:
            raise RuntimeError(
                f'Found no leftImg8bit/val/*/_leftImg8bit.png pairs under {data_root}'
            )
        LOG.info(f'Cityscapes val: {len(pairs)} image/mask pairs from {data_root}')
        dataset = _CityscapesValDataset(pairs, transform=None)
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=0,
        )

    def reformat_for_calibration(self, batched_data: Any):
        return batched_data

    def reformat_for_validation(self, batched_data: Any):
        return [
            types.FrameInput(
                img=types.Image.fromany(image_array),
                ground_truth=gt_sample,
                img_id=gt_sample.img_id,
            )
            for image_array, gt_sample in batched_data
        ]

    def evaluator(
        self, dataset_root, dataset_config, model_info, custom_config, pair_validation=False
    ):
        if self._evaluator_instance is None:
            from ax_evaluators.semantic_segmentation import SemanticSegmentationEvaluator

            self._evaluator_instance = SemanticSegmentationEvaluator(
                num_classes=NUM_CLASSES,
                ignore_index=IGNORE_INDEX,
                labels=list(self.model_info.labels or []),
            )
        return self._evaluator_instance

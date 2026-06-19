# Copyright Axelera AI, 2026
"""In-house semantic-segmentation evaluator.

Pure-numpy confusion-matrix accumulator. No dependency on
mmsegmentation / mmengine / mmcv.

Metrics produced by ``collect_metrics``:
  - mIoU        : mean of per-class IoU over classes that appear in
                  ground truth (rowsum > 0).
  - pixAcc      : fraction of pixels classified correctly.
  - mAcc        : mean per-class recall.
  - IoU_<label> : per-class IoU (one metric per class label; classes
                  absent from ground truth are omitted).
  - num_samples : count of samples accumulated.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from axelera import types
from axelera.app import logging_utils
from axelera.app.eval_interfaces import SemanticSegmentationGroundTruthSample

LOG = logging_utils.getLogger(__name__)

METRIC_MIOU = 'mIoU'
METRIC_PIX_ACC = 'pixAcc'
METRIC_MACC = 'mAcc'
METRIC_NAMES = (METRIC_MIOU, METRIC_PIX_ACC, METRIC_MACC)


def _resize_class_map_nearest(class_map: np.ndarray, target_hw: tuple) -> np.ndarray:
    """Nearest-neighbour resize for a 2D label map without bringing in cv2.

    Used when the predicted class_map and ground-truth mask differ in
    resolution (model output is letterboxed; ground truth is original).
    """
    th, tw = target_hw
    sh, sw = class_map.shape
    if (sh, sw) == (th, tw):
        return class_map
    # vectorised nearest-neighbour via integer index gather
    y_idx = (np.arange(th, dtype=np.int64) * sh // th).clip(0, sh - 1)
    x_idx = (np.arange(tw, dtype=np.int64) * sw // tw).clip(0, sw - 1)
    return class_map[y_idx][:, x_idx]


class SemanticSegmentationEvaluator(types.Evaluator):
    """Pixel-wise confusion-matrix evaluator for semantic segmentation."""

    def __init__(
        self,
        num_classes: int,
        ignore_index: int = 255,
        labels: Optional[List[str]] = None,
    ):
        super().__init__()
        if num_classes <= 0:
            raise ValueError('num_classes must be > 0')
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.labels = labels or []
        self._matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        self._num_samples = 0

    def reset(self) -> None:
        self._matrix.fill(0)
        self._num_samples = 0

    def add(self, pred: np.ndarray, gt: np.ndarray) -> None:
        """Accumulate a single prediction / ground-truth pair.

        Args:
            pred: 2D uint8 class-id array, shape (H, W).
            gt:   2D uint8 ground-truth label array, shape (H, W).
                  Pixels equal to ``self.ignore_index`` are excluded.
        """
        if pred.shape != gt.shape:
            pred = _resize_class_map_nearest(pred, gt.shape)
        pred_flat = np.ascontiguousarray(pred).reshape(-1)
        gt_flat = np.ascontiguousarray(gt).reshape(-1)
        # Drop ignore_index and any GT label outside [0, num_classes).
        # Future label maps may emit sentinels other than self.ignore_index,
        # so the upper-bound check stays even though _REMAP_LUT only emits
        # the canonical 19 trainIds plus 255.
        valid = (gt_flat != self.ignore_index) & (gt_flat < self.num_classes)
        if not valid.any():
            return
        pred_v = pred_flat[valid].astype(np.int64).clip(0, self.num_classes - 1)
        gt_v = gt_flat[valid].astype(np.int64)
        idx = pred_v * self.num_classes + gt_v
        bincount = np.bincount(idx, minlength=self.num_classes * self.num_classes)
        self._matrix += bincount.reshape(self.num_classes, self.num_classes)
        self._num_samples += 1

    def process_meta(self, meta) -> None:
        eval_sample = meta.to_evaluation()
        pred = np.asarray(eval_sample.data)
        while pred.ndim > 2 and pred.shape[0] == 1:
            pred = pred[0]
        ground_truth = meta.access_ground_truth()
        if ground_truth is None:
            raise ValueError(
                'SemanticSegmentationEvaluator.process_meta: ground truth is not set on meta'
            )
        if isinstance(ground_truth, SemanticSegmentationGroundTruthSample):
            gt_mask = ground_truth.gt_mask
        else:
            gt_mask = np.asarray(ground_truth.data)
        self.add(pred, np.asarray(gt_mask))

    def collect_metrics(self):
        matrix = self._matrix
        diag = np.diag(matrix).astype(np.float64)
        # matrix[pred, gt]: rows = predictions, cols = ground truth.
        row_sum = matrix.sum(axis=1).astype(np.float64)
        col_sum = matrix.sum(axis=0).astype(np.float64)
        union = row_sum + col_sum - diag
        with np.errstate(divide='ignore', invalid='ignore'):
            per_class_iou = np.where(union > 0, diag / np.maximum(union, 1), np.nan)
            per_class_acc = np.where(col_sum > 0, diag / np.maximum(col_sum, 1), np.nan)
        present = col_sum > 0
        if present.any():
            mIoU = float(np.nanmean(per_class_iou[present]))
            mAcc = float(np.nanmean(per_class_acc[present]))
        else:
            mIoU = 0.0
            mAcc = 0.0
        total = matrix.sum()
        pixAcc = float(diag.sum() / total) if total > 0 else 0.0

        per_class_keys = [
            f'IoU_{self.labels[i]}' if i < len(self.labels) else f'IoU_class_{i}'
            for i in range(self.num_classes)
        ]
        result = types.EvalResult(
            metric_names=list(METRIC_NAMES) + per_class_keys + ['num_samples'],
            aggregators=None,
            key_metric=METRIC_MIOU,
        )
        for name, value in (
            (METRIC_MIOU, mIoU),
            (METRIC_PIX_ACC, pixAcc),
            (METRIC_MACC, mAcc),
        ):
            result.set_metric_result(name, value, is_percentage=True)
        for key, v in zip(per_class_keys, per_class_iou):
            if np.isnan(v):
                continue
            result.set_metric_result(key, float(v), is_percentage=True)
        result.set_metric_result('num_samples', self._num_samples, is_percentage=False)
        return result

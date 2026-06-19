# Copyright Axelera AI, 2026
"""Unit tests for the in-house SemanticSegmentationEvaluator."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

EVALUATOR_PATH = Path(__file__).parent.parent / 'ax_evaluators' / 'semantic_segmentation.py'


def test_evaluator_has_no_mmlab_imports():
    text = EVALUATOR_PATH.read_text()
    for banned in ('mmseg', 'mmengine', 'mmcv', 'mmdeploy'):
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                assert (
                    banned not in line
                ), f'{EVALUATOR_PATH.name}: forbidden import of {banned!r}: {line!r}'


def test_evaluator_imports_cleanly():
    from ax_evaluators.semantic_segmentation import SemanticSegmentationEvaluator

    assert callable(SemanticSegmentationEvaluator)


def test_perfect_prediction_yields_mIoU_1():
    from ax_evaluators.semantic_segmentation import SemanticSegmentationEvaluator

    ev = SemanticSegmentationEvaluator(num_classes=4)
    gt = np.array([[0, 1, 2, 3], [3, 2, 1, 0]], dtype=np.uint8)
    pred = gt.copy()
    ev.add(pred, gt)
    m = ev.collect_metrics()
    assert m.metrics_result['mIoU'] == pytest.approx(1.0)
    assert m.metrics_result['pixAcc'] == pytest.approx(1.0)
    assert m.metrics_result['mAcc'] == pytest.approx(1.0)


def test_ignore_index_excluded():
    from ax_evaluators.semantic_segmentation import SemanticSegmentationEvaluator

    ev = SemanticSegmentationEvaluator(num_classes=2, ignore_index=255)
    gt = np.array([[0, 0, 255], [1, 255, 1]], dtype=np.uint8)
    # Predict wrong on the ignored pixels; metric must not penalize.
    pred = np.array([[0, 0, 1], [1, 0, 1]], dtype=np.uint8)
    ev.add(pred, gt)
    m = ev.collect_metrics()
    assert m.metrics_result['mIoU'] == pytest.approx(1.0)
    # pixAcc counts only the 4 valid pixels, all correct
    assert m.metrics_result['pixAcc'] == pytest.approx(1.0)


def test_partial_overlap_per_class_iou():
    from ax_evaluators.semantic_segmentation import SemanticSegmentationEvaluator

    ev = SemanticSegmentationEvaluator(num_classes=2)
    # GT: half class-0, half class-1.
    gt = np.array([[0, 0], [1, 1]], dtype=np.uint8)
    # Pred: confused one class-1 pixel with class-0.
    pred = np.array([[0, 0], [0, 1]], dtype=np.uint8)
    ev.add(pred, gt)
    m = ev.collect_metrics()
    # Class 0: TP=2, FP=1, FN=0  -> IoU = 2/3
    # Class 1: TP=1, FP=0, FN=1  -> IoU = 1/2
    expected_mIoU = (2 / 3 + 1 / 2) / 2
    assert m.metrics_result['mIoU'] == pytest.approx(expected_mIoU)
    assert m.metrics_result['pixAcc'] == pytest.approx(3 / 4)


def test_class_not_in_gt_excluded_from_mIoU():
    from ax_evaluators.semantic_segmentation import SemanticSegmentationEvaluator

    ev = SemanticSegmentationEvaluator(num_classes=3)
    gt = np.array([[0, 1], [1, 0]], dtype=np.uint8)  # only classes 0 and 1
    pred = gt.copy()
    ev.add(pred, gt)
    m = ev.collect_metrics()
    # Class 2 absent from GT -> excluded; mIoU averages over classes 0,1 only.
    assert m.metrics_result['mIoU'] == pytest.approx(1.0)
    # Per-class IoU is now emitted as IoU_<label> entries. Classes with NaN
    # IoU (absent from GT) are surfaced as None so the metrics dict still
    # exposes the full label list to downstream consumers.
    assert m.metrics_result.get('IoU_class_2') is None


def test_resize_when_pred_smaller_than_gt():
    from ax_evaluators.semantic_segmentation import SemanticSegmentationEvaluator

    ev = SemanticSegmentationEvaluator(num_classes=2)
    pred = np.array([[0, 1]], dtype=np.uint8)  # (1, 2)
    gt = np.array([[0, 0, 1, 1]], dtype=np.uint8)  # (1, 4)
    ev.add(pred, gt)
    m = ev.collect_metrics()
    # nearest-neighbour upsample of pred -> [0, 0, 1, 1] (matches gt)
    assert m.metrics_result['mIoU'] == pytest.approx(1.0)


def test_accumulation_across_samples():
    from ax_evaluators.semantic_segmentation import SemanticSegmentationEvaluator

    ev = SemanticSegmentationEvaluator(num_classes=2)
    for _ in range(3):
        gt = np.array([[0, 1]], dtype=np.uint8)
        pred = np.array([[0, 0]], dtype=np.uint8)
        ev.add(pred, gt)
    m = ev.collect_metrics()
    # Each call: class 0 IoU = 1/2 (TP=1, FP=1, FN=0), class 1 IoU = 0 (TP=0, FN=1)
    # Accumulated still IoU 0.5 / 0.0 -> mIoU = 0.25
    assert m.metrics_result['mIoU'] == pytest.approx(0.25)
    assert m.metrics_result.get('num_samples', 0) == 3


def test_eval_interfaces_groundtruth_sample():
    from axelera.app.eval_interfaces import (
        SemanticSegmentationEvalSample,
        SemanticSegmentationGroundTruthSample,
    )

    gt = SemanticSegmentationGroundTruthSample(
        gt_mask=np.zeros((10, 10), dtype=np.uint8),
        img_id='frankfurt_000000_001016',
        raw_image_size=(1024, 2048),
    )
    assert gt.data.shape == (10, 10)
    assert gt.img_id == 'frankfurt_000000_001016'
    pred = SemanticSegmentationEvalSample(class_map=np.zeros((5, 5), dtype=np.uint8))
    assert pred.data.shape == (5, 5)

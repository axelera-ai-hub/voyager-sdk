# Copyright Axelera AI, 2024
from unittest.mock import Mock, patch

from ax_evaluators.yolo_eval import (
    EvaluationMetrics,
    YoloEvalmAPCalculator,
    YoloEvaluator,
    ap_per_class,
    box_iou,
    compute_ap,
    kpt_iou,
    mask_iou,
    match_predictions,
    smooth,
)
from axelera.types import EvalResult
import numpy as np
import pytest

from axelera.app.meta import InstanceSegmentationMeta, KeypointDetectionMeta, ObjectDetectionMeta


def test_box_iou():
    box1 = np.array([[0, 0, 2, 2], [1, 1, 3, 3]])
    box2 = np.array([[1, 1, 3, 3], [2, 2, 4, 4]])
    expected_iou = np.array([[0.14285714, 0.0], [1.0, 0.14285714]])
    iou = box_iou(box1, box2)
    np.testing.assert_almost_equal(iou, expected_iou, decimal=6)


def test_kpt_iou():
    gt_kpts = np.array([[[0, 0, 1], [1, 1, 1]], [[1, 1, 1], [2, 2, 1]]])
    pred_kpts = np.array([[[0, 0], [1, 1]], [[1, 1], [2, 2]]])
    area = np.array([1, 1])
    sigma = [0.1, 0.1]
    expected_oks = np.array([[1.0, 0.0], [0.0, 1.0]])
    oks = kpt_iou(gt_kpts, pred_kpts, area, sigma)
    np.testing.assert_almost_equal(oks, expected_oks, decimal=6)


def test_mask_iou():
    mask1 = np.array([[1, 0, 1], [0, 1, 0], [1, 0, 1]])
    mask2 = np.array([[1, 1, 0], [0, 0, 1], [1, 1, 0]])
    expected_iou = np.array(
        [[0.33333334, 0.5, 0.33333334], [0.5, 0.0, 0.5], [0.33333334, 0.5, 0.33333334]]
    )
    iou = mask_iou(mask1, mask2)
    np.testing.assert_almost_equal(iou, expected_iou, decimal=6)


def test_match_predictions():
    pred_classes = np.array([0, 1])
    true_classes = np.array([0, 1])
    iou = np.array([[0.6, 0.4], [0.4, 0.6]])
    expected_correct = np.array(
        [
            [True, True, True, False, False, False, False, False, False, False],
            [True, True, True, False, False, False, False, False, False, False],
        ]
    )
    correct = match_predictions(pred_classes, true_classes, iou)
    np.testing.assert_array_equal(correct, expected_correct)


def test_compute_ap():
    recall = np.array([0.0, 0.5, 1.0])
    precision = np.array([1.0, 0.5, 0.0])
    expected_ap = 0.5
    ap, mpre, mrec = compute_ap(recall, precision)
    assert ap == pytest.approx(expected_ap, 0.01)


def test_ap_per_class():
    tp = np.array([[1, 0], [0, 1]])
    conf = np.array([0.9, 0.8])
    pred_cls = np.array([0, 1])
    target_cls = np.array([0, 1])
    metrics = ap_per_class(tp, conf, pred_cls, target_cls)
    assert isinstance(metrics, EvaluationMetrics)
    assert metrics.tp.shape == (2,)
    assert metrics.fp.shape == (2,)
    assert metrics.precision.shape == (2,)
    assert metrics.recall.shape == (2,)
    assert metrics.f1.shape == (2,)
    assert metrics.ap.shape == (2, 2)
    assert metrics.unique_classes.shape == (2,)
    assert metrics.p_curve.shape == (2, 1000)
    assert metrics.r_curve.shape == (2, 1000)
    assert metrics.f1_curve.shape == (2, 1000)
    assert metrics.x.shape == (1000,)


def test_smooth():
    y = np.array([1, 2, 3, 4, 5])
    expected_smoothed_y = np.array([1.0, 1.5, 2.5, 3.5, 4.5])
    smoothed_y = smooth(y, f=0.2)
    np.testing.assert_almost_equal(smoothed_y[:-1], expected_smoothed_y, decimal=6)


def test_yolo_eval_kpt_map_calculator():
    calculator = YoloEvalmAPCalculator(is_pose=True, is_lazy=False)
    pred_all = [
        {
            'boxes': np.array([[411.0, 156.2, 465.2, 299.0]]),
            'keypoints': np.array(
                [
                    [
                        [430.3, 168.9],
                        [426.0, 165.8],
                        [431.1, 165.5],
                        [435.9, 166.3],
                        [441.2, 165.1],
                        [439.1, 180.1],
                        [449.5, 179.1],
                        [435.0, 202.2],
                        [444.6, 201.8],
                        [420.3, 216.7],
                        [426.9, 213.5],
                        [446.7, 222.9],
                        [454.2, 222.6],
                        [445.7, 254.9],
                        [454.8, 255.4],
                        [449.1, 288.7],
                        [455.8, 286.6],
                    ]
                ]
            ),
            'scores': np.array([0.9]),
        }
    ]
    label_all = [
        {
            'boxes': np.array([[412.0, 157.0, 465.0, 295.0]]),
            'keypoints': np.array(
                [
                    [
                        [427.0, 170.0, 1.0],
                        [426.0, 169.0, 2.0],
                        [0.0, -107.0, 0.0],
                        [434.0, 167.9, 2.0],
                        [0.0, -107.0, 0.0],
                        [441.0, 177.0, 2.0],
                        [446.0, 177.0, 2.0],
                        [437.0, 200.0, 2.0],
                        [430.0, 206.0, 2.0],
                        [430.0, 220.0, 2.0],
                        [420.0, 215.0, 2.0],
                        [444.9, 225.9, 2.0],
                        [452.0, 222.9, 2.0],
                        [447.0, 260.0, 2.0],
                        [454.0, 260.0, 2.0],
                        [450.0, 290.0, 2.0],
                        [455.0, 290.0, 2.0],
                    ]
                ]
            ),
        }
    ]
    result = calculator(pred_all, label_all)
    assert isinstance(result, EvalResult)
    metric_names = [metric for metric, _, _ in result]
    aggregators = [aggregator for _, aggregator, _ in result]
    values = [value for _, _, value in result]
    assert metric_names == [
        'mAP',
        'mAP',
        'mAP50',
        'mAP50',
        'precision',
        'precision',
        'recall',
        'recall',
    ]
    assert aggregators == ['box', 'pose', 'box', 'pose', 'box', 'pose', 'box', 'pose']
    assert values == [
        '89.55%',
        '59.70%',
        '99.50%',
        '99.50%',
        '100.00%',
        '100.00%',
        '100.00%',
        '100.00%',
    ]


@pytest.fixture
def mock_evaluator():
    evaluator = Mock(spec=YoloEvalmAPCalculator)
    evaluator.is_pose = False
    evaluator.is_seg = False
    return evaluator


@pytest.fixture
def yolo_evaluator(mock_evaluator):
    return YoloEvaluator(evaluator=mock_evaluator)


def test_process_meta_instance_segmentation(yolo_evaluator, mock_evaluator):
    meta = Mock(spec=InstanceSegmentationMeta)
    meta.to_evaluation.return_value = Mock(data='eval_data')
    meta.access_ground_truth.return_value = Mock(data='gt_data')

    yolo_evaluator.process_meta(meta)

    mock_evaluator.process.assert_called_once_with('eval_data', 'gt_data')


def test_process_meta_keypoint_detection(yolo_evaluator, mock_evaluator):
    meta = Mock(spec=KeypointDetectionMeta)
    meta.to_evaluation.return_value = Mock(data='eval_data')
    meta.access_ground_truth.return_value = Mock(data='gt_data')

    yolo_evaluator.process_meta(meta)

    mock_evaluator.process.assert_called_once_with('eval_data', 'gt_data')


def test_process_meta_object_detection(yolo_evaluator, mock_evaluator):
    meta = Mock(spec=ObjectDetectionMeta)
    meta.to_evaluation.return_value = Mock(data='eval_data')
    meta.access_ground_truth.return_value = Mock(data='gt_data')

    yolo_evaluator.process_meta(meta)

    mock_evaluator.process.assert_called_once_with('eval_data', 'gt_data')


def test_collect_metrics(yolo_evaluator, mock_evaluator):
    mock_evaluator.summary.return_value = EvalResult(
        metric_names=["mAP", "mAP50", "precision", "recall"],
        aggregators=["box", "box", "box", "box"],
    )

    result = yolo_evaluator.collect_metrics()

    mock_evaluator.summary.assert_called_once()
    assert isinstance(result, EvalResult)

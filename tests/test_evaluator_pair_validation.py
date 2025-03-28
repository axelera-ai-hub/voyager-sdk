# Copyright Axelera AI, 2024
from unittest.mock import MagicMock

from ax_evaluators.pair_validation import PairValidationEvaluator, _calculate_roc
import numpy as np
import pytest


@pytest.mark.parametrize(
    "thresholds, distances, actual_issame, nrof_folds, expected_accuracy, expected_best_threshold, expected_tprs, expected_fprs, expected_auc, expected_eer",
    [
        (
            np.array([0.5, 1.0, 1.5]),
            np.array([0.3, 0.7, 1.2, 1.8]),
            np.array([True, False, True, False]),
            2,
            np.array([0.5, 0.5]),
            1.0,
            np.array([[1.0, 1.0, 1.0], [0.0, 0.0, 1.0]]),
            np.array([[0.0, 1.0, 1.0], [0.0, 0.0, 0.0]]),
            0.5,
            0.0,
        ),
        (
            np.array([0.4, 0.8, 1.2]),
            np.array([0.2, 0.6, 1.0, 1.4]),
            np.array([True, True, False, False]),
            2,
            np.array([0.5, 1.0]),
            0.6,
            np.array([[0.5, 1.0, 1.0], [0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.5]]),
            0.0,
            0.25,
        ),
    ],
)
def test_calculate_roc(
    thresholds,
    distances,
    actual_issame,
    nrof_folds,
    expected_accuracy,
    expected_best_threshold,
    expected_tprs,
    expected_fprs,
    expected_auc,
    expected_eer,
):
    accuracy, best_threshold, tprs, fprs, auc, eer = _calculate_roc(
        thresholds, distances, actual_issame, np.less, nrof_folds
    )

    assert accuracy.shape == (nrof_folds,)
    assert isinstance(best_threshold, float)
    assert tprs.shape == (nrof_folds, len(thresholds))
    assert fprs.shape == (nrof_folds, len(thresholds))

    np.testing.assert_almost_equal(accuracy, expected_accuracy, decimal=5)
    assert best_threshold == pytest.approx(expected_best_threshold, rel=1e-5)
    np.testing.assert_almost_equal(tprs, expected_tprs, decimal=5)
    np.testing.assert_almost_equal(fprs, expected_fprs, decimal=5)
    assert auc == pytest.approx(expected_auc, rel=1e-5)
    assert eer == pytest.approx(expected_eer, rel=1e-5)


def generate_synthetic_roc_data(n_samples=1000, pos_mean=0.3, neg_mean=0.7, std_dev=0.1, seed=42):
    np.random.seed(seed)
    n_pos = n_samples // 2
    n_neg = n_samples - n_pos

    pos_distances = np.random.normal(loc=pos_mean, scale=std_dev, size=n_pos)
    neg_distances = np.random.normal(loc=neg_mean, scale=std_dev, size=n_neg)

    distances = np.concatenate([pos_distances, neg_distances])
    actual_issame = np.array([True] * n_pos + [False] * n_neg)

    # Shuffle the data to mix positive and negative pairs
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    distances = distances[indices]
    actual_issame = actual_issame[indices]

    return distances, actual_issame


@pytest.mark.parametrize(
    "thresholds, nrof_folds, expected_accuracy_range, expected_auc_range, expected_eer_range",
    [
        (
            np.linspace(0, 1, 100),  # 100 thresholds between 0 and 1
            5,  # 5-fold cross-validation
            (0.7, 1.0),  # Expected accuracy range
            (0.7, 1.0),  # Expected AUC range
            (0.0, 0.4),  # Expected EER range
        ),
    ],
)
def test_calculate_roc_with_synthetic_roc_data(
    thresholds,
    nrof_folds,
    expected_accuracy_range,
    expected_auc_range,
    expected_eer_range,
):
    distances, actual_issame = generate_synthetic_roc_data(n_samples=1000)
    accuracy, best_threshold, tprs, fprs, auc, eer = _calculate_roc(
        thresholds, distances, actual_issame, np.less, nrof_folds
    )

    assert accuracy.shape == (nrof_folds,)
    assert isinstance(best_threshold, float)
    assert tprs.shape == (nrof_folds, len(thresholds))
    assert fprs.shape == (nrof_folds, len(thresholds))

    # Check if the values are within the expected ranges
    assert expected_accuracy_range[0] <= np.mean(accuracy) <= expected_accuracy_range[1]
    assert expected_auc_range[0] <= auc <= expected_auc_range[1]
    assert expected_eer_range[0] <= eer <= expected_eer_range[1]


@pytest.fixture
def mock_ax_task_meta():
    mock_meta = MagicMock()
    mock_meta.to_evaluation.return_value.ground_truth.data = True
    mock_meta.to_evaluation.return_value.prediction.data = {
        'embedding_1': np.random.rand(1, 128),
        'embedding_2': np.random.rand(1, 128),
    }
    return mock_meta


def test_pair_validation_evaluator_process_meta(mock_ax_task_meta):
    evaluator = PairValidationEvaluator(k_fold=5, distance_metric='euclidean_distance')
    distances, actual_issame = generate_synthetic_roc_data(n_samples=1000)

    for i in range(len(actual_issame)):
        mock_sample = MagicMock()
        mock_sample.data = {
            'embedding_1': np.random.rand(1, 128),
            'embedding_2': np.random.rand(1, 128),
        }
        mock_ax_task_meta.to_evaluation.return_value = mock_sample
        mock_ax_task_meta.access_ground_truth.return_value.data = actual_issame[i]
        evaluator.process_meta(mock_ax_task_meta)

    assert len(evaluator.is_same) == 1000
    assert len(evaluator.distances) == 1000


def test_pair_validation_evaluator_collect_metrics():
    evaluator = PairValidationEvaluator(k_fold=5, distance_metric='euclidean_distance')
    distances, actual_issame = generate_synthetic_roc_data(n_samples=1000)

    for i in range(len(actual_issame)):
        evaluator.is_same.append(actual_issame[i])
        evaluator.distances.append(distances[i])

    evaluator.distances = np.array(evaluator.distances).reshape(1, -1)
    evaluator.is_same = np.array(evaluator.is_same)

    result = evaluator.collect_metrics()

    assert 'accuracy' in result.metrics_result
    assert 'validation_rate' in result.metrics_result
    assert 'auc' in result.metrics_result
    assert 'eer' in result.metrics_result

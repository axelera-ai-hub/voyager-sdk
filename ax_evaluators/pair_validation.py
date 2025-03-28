# Copyright Axelera AI, 2024
from __future__ import annotations

import typing

import numpy as np

from axelera import types
from axelera.app.model_utils import embeddings
from axelera.app.utils import logging_utils

if typing.TYPE_CHECKING:
    from axelera.app import meta

LOG = logging_utils.getLogger(__name__)


def calculate_accuracy(
    threshold: float,
    distances: np.ndarray,
    actual_issame: np.ndarray,
    compare_func: typing.Callable = np.less,
) -> float:
    """Calculate the accuracy given a threshold, distances, and ground truth labels."""
    predict_issame = compare_func(distances, threshold)

    true_positive = np.sum(np.logical_and(predict_issame, actual_issame))
    true_negative = np.sum(
        np.logical_and(np.logical_not(predict_issame), np.logical_not(actual_issame))
    )

    accuracy = (true_positive + true_negative) / len(actual_issame)
    # print(
    #     f"predict_issame: {predict_issame}, actual_issame: {actual_issame}, accuracy: {accuracy}, distances: {distances}"
    # )
    return accuracy


def _calculate_roc(
    thresholds: np.ndarray,
    distances: np.ndarray,
    actual_issame: np.ndarray,
    compare_func: typing.Callable,
    nrof_folds: int = 10,
) -> typing.Tuple[np.ndarray, float, np.ndarray, np.ndarray, float, float]:
    """Calculate the receiver-operator characteristics (ROC) and related metrics.

    Args:
        thresholds (np.ndarray): Array of thresholds to determine whether a pair is similar.
        distances (np.ndarray): Precomputed distances between pairs.
        actual_issame (np.ndarray): Ground truth array indicating whether pairs are the same.
        nrof_folds (int, optional): Number of folds to use for k-fold ROC calculation. Default is 10.

    Returns:
        Tuple[np.ndarray, float, np.ndarray, np.ndarray, float, float]:
            - accuracy (np.ndarray): Array of accuracies for each fold.
            - best_threshold (float): The best threshold determined from the ROC calculation.
            - tprs (np.ndarray): True positive rates for each threshold.
            - fprs (np.ndarray): False positive rates for each threshold.
            - auc (float): Area under the ROC curve.
            - eer (float): Equal error rate.
    """
    nrof_pairs = min(len(actual_issame), len(distances))
    nrof_thresholds = len(thresholds)
    accuracy = np.zeros(nrof_folds)
    indices = np.arange(nrof_pairs)

    # Custom K-Fold implementation
    fold_sizes = np.full(nrof_folds, nrof_pairs // nrof_folds, dtype=int)
    fold_sizes[: nrof_pairs % nrof_folds] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        folds.append(indices[start:stop])
        current = stop

    best_thresholds = []
    tprs = []
    fprs = []
    aucs = []
    eers = []

    for fold_idx in range(nrof_folds):
        test_set = folds[fold_idx]
        train_set = np.concatenate([folds[i] for i in range(nrof_folds) if i != fold_idx])

        # Find the best threshold for the fold
        acc_train = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            acc_train[threshold_idx] = calculate_accuracy(
                threshold,
                distances[train_set],
                actual_issame[train_set],
                compare_func=compare_func,
            )
        best_threshold_index = np.argmax(acc_train)
        best_thresholds.append(thresholds[best_threshold_index])

        accuracy[fold_idx] = calculate_accuracy(
            thresholds[best_threshold_index],
            distances[test_set],
            actual_issame[test_set],
            compare_func=compare_func,
        )

        tpr = np.zeros(nrof_thresholds)
        fpr = np.zeros(nrof_thresholds)
        for threshold_idx, threshold in enumerate(thresholds):
            predict_issame = distances[test_set] < threshold
            tp = np.sum(np.logical_and(predict_issame, actual_issame[test_set]))
            fp = np.sum(np.logical_and(predict_issame, np.logical_not(actual_issame[test_set])))
            fn = np.sum(np.logical_and(np.logical_not(predict_issame), actual_issame[test_set]))
            tn = np.sum(
                np.logical_and(
                    np.logical_not(predict_issame), np.logical_not(actual_issame[test_set])
                )
            )

            tpr[threshold_idx] = tp / (tp + fn) if (tp + fn) > 0 else 0
            fpr[threshold_idx] = fp / (fp + tn) if (fp + tn) > 0 else 0
        tprs.append(tpr)
        fprs.append(fpr)

        auc = np.trapz(tpr, fpr)
        aucs.append(auc)

        eer = fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))]
        eers.append(eer)

    best_threshold = np.mean(best_thresholds)
    mean_auc = np.mean(aucs)
    mean_eer = np.mean(eers)
    return accuracy, best_threshold, np.array(tprs), np.array(fprs), mean_auc, mean_eer


def plot_roc(tprs: np.ndarray, fprs: np.ndarray, roc_file_path: str = "roc_curve.png") -> None:
    """Plot the ROC curve
    Params:
        tprs (np.ndarray): true positive rates
        fprs (np.ndarray): false positive rates
        roc_file_path (str): file path to save the ROC curve plot
    """
    import matplotlib.pyplot as plt

    mean_tpr = np.mean(tprs, axis=0)
    mean_fpr = np.mean(fprs, axis=0)
    plt.figure()
    plt.plot(mean_fpr, mean_tpr, label='ROC curve')
    plt.scatter(mean_fpr, mean_tpr, color='red', s=10)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='best')
    plt.savefig(roc_file_path)
    plt.close()


class PairValidationEvaluator(types.Evaluator):
    """
    Evaluator for Pair Validation
    """

    def __init__(
        self,
        k_fold: int = 10,
        distance_metric: typing.Union[
            str, embeddings.DistanceMetric
        ] = embeddings.DistanceMetric.euclidean_distance,
        distance_threshold: float = 0,
        plot_roc: bool = False,
    ):
        # k_fold is the number of folds to use for k-fold roc calculation
        # When k_fold=1, skip ROC calculation but directly calculate accuracy
        super().__init__()
        if k_fold > 1:
            self.thresholds = np.arange(0, 4, 0.01)
        else:
            self.thresholds = np.array([distance_threshold])
        self.is_same: typing.List[bool] = []
        self.distances: typing.List[float] = []
        if isinstance(distance_metric, str):
            try:
                distance_metric = embeddings.DistanceMetric[distance_metric]
            except KeyError:
                raise ValueError(f"Distance metric {distance_metric} is not supported")
        self.distance_metric = distance_metric
        self.k_fold = k_fold
        self.plot_roc = plot_roc
        self.compare_func = (
            np.greater
            if self.distance_metric == embeddings.DistanceMetric.cosine_similarity
            else np.less
        )

    def process_meta(self, ax_task_meta: meta.AxTaskMeta) -> None:
        sample = ax_task_meta.to_evaluation()
        self.is_same.append(ax_task_meta.access_ground_truth().data)
        embedding_1 = sample.data['embedding_1']
        embedding_2 = sample.data['embedding_2']
        if self.distance_metric == embeddings.DistanceMetric.euclidean_distance:
            dist_or_sim = embeddings.euclidean_distance(embedding_1, embedding_2)
        elif self.distance_metric == embeddings.DistanceMetric.squared_euclidean_distance:
            dist_or_sim = embeddings.squared_euclidean_distance(embedding_1, embedding_2)
        elif self.distance_metric == embeddings.DistanceMetric.cosine_distance:
            dist_or_sim = embeddings.cosine_distance(embedding_1, embedding_2)
        else:
            dist_or_sim = embeddings.cosine_similarity(embedding_1, embedding_2)
        self.distances.append(dist_or_sim)

    def collect_metrics(self) -> types.EvalResult:
        distances = np.concatenate(self.distances)
        is_same = np.asarray(self.is_same)
        if self.k_fold == 1:
            accuracy = calculate_accuracy(
                threshold=self.thresholds[0],
                distances=distances,
                actual_issame=is_same,
                compare_func=self.compare_func,
            )
            accuracy = np.array([accuracy])  # Wrap in array for consistency
            auc = 1.0  # AUC is 1.0 when k_fold is 1
            eer = 0.0  # EER is 0.0 when k_fold is 1
            tprs = None
            fprs = None
            validation_rate = None
            std_validation_rate = None
        else:
            accuracy, suggested_threshold, tprs, fprs, auc, eer = _calculate_roc(
                self.thresholds, distances, is_same, self.compare_func, nrof_folds=self.k_fold
            )
            LOG.info("Suggested threshold based on ROC: %.4f", suggested_threshold)
            if self.plot_roc:
                plot_roc(tprs, fprs, roc_file_path="roc_curve.png")
            validation_rate = np.mean(tprs[:, np.argmin(np.abs(self.thresholds - 0.001))])
            std_validation_rate = np.std(tprs[:, np.argmin(np.abs(self.thresholds - 0.001))])

        mean_accuracy = np.mean(accuracy, axis=0)
        std_accuracy = np.std(accuracy, axis=0)

        if self.k_fold == 1:
            result = types.EvalResult(
                ['accuracy'],
                {
                    'accuracy': ['mean', 'std'],
                },
                key_metric='accuracy',
                key_aggregator='mean',
            )
            result.set_metric_result('accuracy', mean_accuracy, 'mean')
            result.set_metric_result('accuracy', std_accuracy, 'std')
        else:
            result = types.EvalResult(
                ['accuracy', 'validation_rate', 'auc', 'eer'],
                {
                    'accuracy': ['mean', 'std'],
                    'validation_rate': ['mean', 'std'],
                    'auc': ['mean'],
                    'eer': ['mean'],
                },
                key_metric='accuracy',
                key_aggregator='mean',
            )
            result.set_metric_result('accuracy', mean_accuracy, 'mean', True)
            result.set_metric_result('accuracy', std_accuracy, 'std', True)
            result.set_metric_result('validation_rate', validation_rate, 'mean', True)
            result.set_metric_result('validation_rate', std_validation_rate, 'std', True)
            result.set_metric_result('auc', auc, 'mean', True)
            result.set_metric_result('eer', eer, 'mean', True)
        return result

# Copyright Axelera AI, 2025
from __future__ import annotations

import numpy as np
import torch

from ax_models.tutorials.resnet34_fruit360 import build_dataset
from axelera import types
from axelera.app import eval_interfaces, meta


class CustomDataAdapter(types.DataAdapter):
    def __init__(self, dataset_config, model_info):
        self.dataset_config = dataset_config
        self.model_info = model_info

    def create_validation_data_loader(self, root, target_split, **kwargs):
        assert 'val_data' in kwargs, "val_data is required in the YAML dataset config"
        return torch.utils.data.DataLoader(
            build_dataset(root / kwargs['val_data'], transform=None),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=0,
        )

    def reformat_for_validation(self, batched_data):
        return [
            types.FrameInput.from_image(
                img=img,
                color_format=types.ColorFormat.RGB,
                ground_truth=eval_interfaces.ClassificationGroundTruthSample(class_id=target),
                img_id='',
            )
            for img, target in batched_data
        ]

    def evaluator(
        self, dataset_root, dataset_config, model_info, custom_config, pair_validation=False
    ):
        from ax_evaluators.classification import ClassificationEvaluator

        return ClassificationEvaluator()


# Tutorial-8: Custom Data Adapter with Custom Evaluator
def organize_results(top_k, accuracy, top_k_accuracy, class_report=None):
    """
    Organize evaluation metrics into a structured result object.

    Args:
        top_k (int): K value for top-K accuracy
        accuracy (float): Top-1 accuracy value
        top_k_accuracy (float): Top-K accuracy value
        class_report (dict, optional): Classification report dictionary from sklearn
    Returns:
        types.EvalResult: Structured evaluation results
    """
    metric_names = ['Top-1 Accuracy', f'Top-{top_k} Accuracy']
    aggregator_dict = {'Top-1 Accuracy': ['average'], f'Top-{top_k} Accuracy': ['average']}

    if class_report is not None:
        metric_names.extend(['Precision', 'Recall', 'F1-score'])
        # Using both macro and weighted averages
        aggregator_dict.update(
            {
                'Precision': ['macro_avg', 'weighted_avg'],
                'Recall': ['macro_avg', 'weighted_avg'],
                'F1-score': ['macro_avg', 'weighted_avg'],
            }
        )

    eval_result = types.EvalResult(
        metric_names=metric_names,
        aggregators=aggregator_dict,
        key_metric='Top-1 Accuracy',
        key_aggregator='average',
    )

    # Set base metrics
    eval_result.set_metric_result('Top-1 Accuracy', accuracy, 'average', is_percentage=True)
    eval_result.set_metric_result(
        f'Top-{top_k} Accuracy', top_k_accuracy, 'average', is_percentage=True
    )

    if class_report is not None:
        try:
            # Define metric mappings
            metric_mappings = [
                ('precision', 'Precision'),
                ('recall', 'Recall'),
                ('f1-score', 'F1-score'),
            ]

            # Set macro average metrics
            if 'macro avg' in class_report:
                for metric_key, metric_name in metric_mappings:
                    if metric_key in class_report['macro avg']:
                        eval_result.set_metric_result(
                            metric_name,
                            class_report['macro avg'][metric_key],
                            'macro_avg',
                            is_percentage=True,
                        )

            # Set weighted average metrics
            if 'weighted avg' in class_report:
                for metric_key, metric_name in metric_mappings:
                    if metric_key in class_report['weighted avg']:
                        eval_result.set_metric_result(
                            metric_name,
                            class_report['weighted avg'][metric_key],
                            'weighted_avg',
                            is_percentage=True,
                        )

        except KeyError as e:
            print(f"Warning: Classification report is missing expected keys: {e}")
            raise
    return eval_result


class OfflineEvaluator(types.Evaluator):
    def __init__(self, top_k: int = 1, **kwargs):
        """
        Initialize the evaluator.
        :param top_k: The value of k for top-k accuracy (default is 1, i.e., top-1 accuracy).
        """
        self.top_k = top_k  # Store the top-k value
        self.labels = []  # Predicted top-1 labels
        self.gt_labels = []  # Ground truth labels
        self.top_k_correct = []  # Track whether ground truth is in top-k predictions

    def process_meta(self, meta) -> None:
        """
        Process metadata for a single sample.
        - meta.class_ids: List of top-k predicted class IDs (sorted by score).
        - meta.ground_truth.class_id: Ground truth class ID for the sample.
        """
        # Append the top-1 predicted label (assuming class_ids is sorted by score)
        class_ids = meta.to_evaluation().data
        self.labels.append(class_ids[0])  # Top-1 prediction
        self.gt_labels.append(meta.access_ground_truth().data)

        # Check if ground truth is in the top-k predictions
        is_top_k_correct = meta.access_ground_truth().data in class_ids[: self.top_k]
        self.top_k_correct.append(is_top_k_correct)

    def collect_metrics(self, key_metric_aggregator: str | None = None):
        """
        Collect and compute evaluation metrics after processing all samples by using sklearn
        """
        from sklearn.metrics import accuracy_score, classification_report

        labels = np.array(self.labels)  # Predicted top-1 labels
        gt_labels = np.array(self.gt_labels)  # Ground truth labels

        accuracy = accuracy_score(gt_labels, labels)
        top_k_accuracy = np.mean(self.top_k_correct)

        class_report = classification_report(gt_labels, labels, output_dict=True)

        return organize_results(self.top_k, accuracy, top_k_accuracy, class_report)


class OnlineEvaluator(types.Evaluator):
    def __init__(self, top_k: int = 1, num_classes: int = None, **kwargs):
        """
        Initialize the evaluator.
        :param top_k: The value of k for top-k accuracy (default is 1, i.e., top-1 accuracy).
        :param num_classes: The number of classes in the dataset (required for confusion matrix).
        """
        self.top_k = top_k  # Store the top-k value
        self.num_classes = num_classes  # Number of classes (for confusion matrix)
        self.total_samples = 0  # Total number of samples processed
        self.correct_top1 = 0  # Count of correct top-1 predictions
        self.correct_topk = 0  # Count of correct top-k predictions

        # Initialize confusion matrix
        if num_classes is not None:
            self.confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
        else:
            self.confusion_matrix = None

    def process_meta(self, meta) -> None:
        """
        Process metadata for a single sample and update metrics.
        - meta.class_ids: List of top-k predicted class IDs (sorted by score).
        - meta.ground_truth.class_id: Ground truth class ID for the sample.
        """
        self.total_samples += 1
        gt_label = meta.access_ground_truth().data
        top1_pred = meta.to_evaluation().data[0]
        if top1_pred == gt_label:
            self.correct_top1 += 1
        if gt_label in meta.to_evaluation().data[: self.top_k]:
            self.correct_topk += 1

        if self.confusion_matrix is not None:
            self.confusion_matrix[gt_label, top1_pred] += 1

    def collect_metrics(self):
        """
        Collect and return the final metrics.
        """
        from sklearn.metrics import classification_report

        top1_accuracy = self.correct_top1 / self.total_samples if self.total_samples > 0 else 0.0
        topk_accuracy = self.correct_topk / self.total_samples if self.total_samples > 0 else 0.0

        if self.confusion_matrix is not None:
            gt_labels = []
            pred_labels = []
            for i in range(self.num_classes):
                for j in range(self.num_classes):
                    count = self.confusion_matrix[i, j]
                    gt_labels.extend([i] * count)
                    pred_labels.extend([j] * count)

            class_report = classification_report(gt_labels, pred_labels, output_dict=True)
        else:
            class_report = None

        return organize_results(self.top_k, top1_accuracy, topk_accuracy, class_report)


class CustomDataAdapterWithOfflineEvaluator(CustomDataAdapter):
    def evaluator(
        self, dataset_root, dataset_config, model_info, custom_config, pair_validation=False
    ):
        return OfflineEvaluator(top_k=custom_config.get('top_k', 1))


class CustomDataAdapterWithOnlineEvaluator(CustomDataAdapter):
    def evaluator(
        self, dataset_root, dataset_config, model_info, custom_config, pair_validation=False
    ):
        return OnlineEvaluator(
            top_k=custom_config.get('top_k', 1), num_classes=model_info.num_classes
        )

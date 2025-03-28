# Copyright Axelera AI, 2024
# This file is inspired by Ultralytics eval logic and reimplemented here
# in order to get similar metric results when using Ultralytics models
from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple, Union

import numpy as np

from axelera import types
from axelera.app.model_utils.box import xyxy2xywh
from axelera.app.model_utils.segment import simple_resize_masks
from axelera.app.utils import logging_utils

if TYPE_CHECKING:
    from axelera.app.meta import (
        InstanceSegmentationMeta,
        KeypointDetectionMeta,
        ObjectDetectionMeta,
    )

LOG = logging_utils.getLogger(__name__)


@dataclasses.dataclass
class EvaluationMetrics:
    """
    Dataclass to store evaluation metrics.

    Attributes:
        tp (np.ndarray): True positive counts at threshold given by max F1 metric.
        fp (np.ndarray): False positive counts at threshold given by max F1 metric.
        precision (np.ndarray): Precision values at threshold given by max F1 metric.
        recall (np.ndarray): Recall values at threshold given by max F1 metric.
        f1 (np.ndarray): F1-score values at threshold given by max F1 metric.
        ap (np.ndarray): Average precision for each class at different IoU thresholds.
        unique_classes (np.ndarray): An array of unique classes that have data.
        p_curve (np.ndarray): Precision curves for each class. Shape: (nc, 1000).
        r_curve (np.ndarray): Recall curves for each class. Shape: (nc, 1000).
        f1_curve (np.ndarray): F1-score curves for each class. Shape: (nc, 1000).
        x (np.ndarray): X-axis values for the curves. Shape: (1000,).
    """

    tp: np.ndarray
    fp: np.ndarray
    precision: np.ndarray
    recall: np.ndarray
    f1: np.ndarray
    ap: np.ndarray
    unique_classes: np.ndarray
    p_curve: np.ndarray
    r_curve: np.ndarray
    f1_curve: np.ndarray
    x: np.ndarray


def box_iou(box1, box2, eps=1e-7):
    """
    Calculate intersection-over-union (IoU) of boxes. Both box1 and box2 use format (x1, y1, x2, y2).

    Args:
        box1 (np.ndarray): Array of bounding boxes [N, 4].
        box2 (np.ndarray): Array of bounding boxes [M, 4].
        eps (float, optional): Small value to avoid division by zero.

    Returns:
        np.ndarray: An NxM array containing the pairwise IoU values for every element in box1 and box2.
    """
    inter_x1 = np.maximum(box1[:, None, 0], box2[:, 0])
    inter_y1 = np.maximum(box1[:, None, 1], box2[:, 1])
    inter_x2 = np.minimum(box1[:, None, 2], box2[:, 2])
    inter_y2 = np.minimum(box1[:, None, 3], box2[:, 3])

    inter_area = np.clip(inter_x2 - inter_x1, a_min=0, a_max=None) * np.clip(
        inter_y2 - inter_y1, a_min=0, a_max=None
    )
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    return inter_area / (box1_area[:, None] + box2_area - inter_area + eps)


def kpt_iou(ground_truth_kpts, predicted_kpts, area, sigma, eps=1e-7):
    """
    Calculate Object Keypoint Similarity (OKS) using NumPy.

    Args:
        ground_truth_kpts (np.ndarray): Ground truth keypoints [N, 17, 3].
        predicted_kpts (np.ndarray): Predicted keypoints [N, 17, 2].
        area (np.ndarray): Areas of ground truth [N].
        sigma (list): Scales for keypoints [17].
        eps (float, optional): Small value to avoid division by zero.

    Returns:
        np.ndarray: OKS values [N, N].
    """
    # Ensure only x, y coordinates are used from both ground truth and predicted keypoints
    gt_kpts_xy = ground_truth_kpts[..., :2]  # Shape: [N, 17, 2]
    pred_kpts_xy = predicted_kpts[..., :2]  # Shape: [N, 17, 2]

    # Calculate squared differences for x, y coordinates
    d = (gt_kpts_xy[:, None, :, :] - pred_kpts_xy[None, :, :, :]) ** 2
    squared_distances = d.sum(axis=-1)  # Shape: [N, N, 17]

    # Convert sigma to array and prepare for broadcasting
    sigma_array = np.array(sigma)  # Shape: [17]
    sigma_squared = (2 * sigma_array) ** 2

    # Check visibility of each keypoint in the ground truth
    visibility_mask = ground_truth_kpts[..., 2] > 0  # Shape: [N, 17]

    # Calculate the exponential term for OKS
    e = squared_distances / (sigma_squared[None, None, :] * (area[:, None, None] + eps))
    oks_exp = np.exp(-e / 2)

    # Apply visibility mask and sum over keypoints
    visible_oks = (oks_exp * visibility_mask[:, None, :]).sum(axis=-1)  # Shape: [N, N]

    # Normalize by the number of visible keypoints
    normalization = visibility_mask.sum(axis=-1, keepdims=True)  # Shape: [N, 1]
    normalized_oks = visible_oks / (normalization + eps)  # Shape: [N, N]

    return normalized_oks


def mask_iou(mask1, mask2, eps=1e-7):
    """
    Calculate masks IoU.

    Args:
        mask1 (np.ndarray): An array with shape (N, n).
        mask2 (np.ndarray): An array with shape (M, n).
        eps (float, optional): A small constant, defaults to 1e-7.

    Returns:
        (np.ndarray): An array of shape (N, M) representing masks IoU.
    """
    mask1 = mask1.astype(np.float32)
    mask2 = mask2.astype(np.float32)

    intersection = np.dot(mask1, mask2.T).clip(0)
    union = (mask1.sum(axis=1)[:, None] + mask2.sum(axis=1)[None, :]) - intersection
    return intersection / (union + eps)


def _get_matches(iou, threshold):
    """
    Get matches for a given IoU threshold.

    Args:
        iou (np.ndarray): IoU matrix.
        threshold (float): IoU threshold.

    Returns:
        np.ndarray: Array of matches.
    """
    return np.argwhere(iou >= threshold)


def _sort_and_filter_matches(matches, iou):
    """
    Sort matches by IoU and filter to ensure unique ground truth and predictions.

    Args:
        matches (np.ndarray): Array of matches.
        iou (np.ndarray): IoU matrix.

    Returns:
        np.ndarray: Filtered matches.
    """
    if matches.shape[0] > 1:
        # Sort matches by IoU in descending order
        sorted_indices = np.argsort(-iou[matches[:, 0], matches[:, 1]])
        matches = matches[sorted_indices]

        # Filter to ensure unique ground truth and predictions
        unique_gt_indices = np.unique(matches[:, 1], return_index=True)[1]
        matches = matches[unique_gt_indices]
        unique_pred_indices = np.unique(matches[:, 0], return_index=True)[1]
        matches = matches[unique_pred_indices]

    return matches


def match_predictions(pred_classes, true_classes, iou, use_gt_indices=True):
    """
    Matches predictions to ground truth objects (pred_classes, true_classes) using IoU.

    Args:
        pred_classes (np.ndarray): Predicted class indices of shape (N,).
        true_classes (np.ndarray): Target class indices of shape (M,).
        iou (np.ndarray): An NxM array containing the pairwise IoU values for
        predictions and ground truth.
        use_gt_indices (bool, optional): Whether to use ground truth indices to mark correct.

    Returns:
        (np.ndarray): Correct tensor of shape (N, 10) for 10 IoU thresholds.
    """
    iouv = np.linspace(0.5, 0.95, 10)
    correct = np.zeros((pred_classes.shape[0], iouv.shape[0]), dtype=bool)

    # Apply class-based filtering
    correct_class = true_classes[:, None] == pred_classes
    iou_filtered = iou * correct_class

    for i, threshold in enumerate(iouv):
        matches = _get_matches(iou_filtered, threshold)
        if matches.size > 0:
            matches = _sort_and_filter_matches(matches, iou_filtered)
            if matches.size > 0:
                if use_gt_indices:
                    correct[matches[:, 1], i] = True
                else:
                    # use predicted indices to mark correct
                    valid_indices = (
                        matches[:, 0] < pred_classes.shape[0]
                    )  # Ensure indices are within bounds
                    correct[matches[valid_indices, 0], i] = True

    return correct


def compute_ap(recall, precision):
    """
    Compute the average precision (AP) given the recall and precision curves.

    Returns:
        (float): Average precision.
        (np.ndarray): Precision envelope curve.
        (np.ndarray): Modified recall curve with sentinel values added at the beginning and end.
    """
    # Append sentinel values to beginning and end
    mrec = np.r_[0.0, recall, 1.0]
    mpre = np.r_[1.0, precision, 0.0]

    # Compute the precision envelope
    mpre = np.maximum.accumulate(mpre[::-1])[::-1]

    # 101-point interpolation (COCO)
    x = np.linspace(0, 1, 101)
    ap = np.trapz(np.interp(x, mrec, mpre), x)  # integrate

    return ap, mpre, mrec


def ap_per_class(
    tp, conf, pred_cls, target_cls, names={0: "person"}, eps=1e-16
) -> EvaluationMetrics:
    """
    Computes the average precision per class

    Args:
        tp (np.ndarray): Binary array indicating whether the detection is correct or not.
        conf (np.ndarray): Array of confidence scores of the detections.
        pred_cls (np.ndarray): Array of predicted classes of the detections.
        target_cls (np.ndarray): Array of true classes of the detections.
        names (dict, optional): Dictionary of class names.
        eps (float, optional): A small value to avoid division by zero. Defaults to 1e-16.

    Returns:
        EvaluationMetrics: A dataclass containing evaluation metrics.
    """
    # Sort by confidence
    i = np.argsort(-conf)
    tp, conf, pred_cls = tp[i], conf[i], pred_cls[i]

    # Find unique classes
    unique_classes = np.unique(target_cls)
    nc = unique_classes.shape[0]
    x = np.linspace(0, 1, 1000)

    ap = np.zeros((nc, tp.shape[1]))
    p_curve = np.zeros((nc, 1000))
    r_curve = np.zeros((nc, 1000))

    for ci, c in enumerate(unique_classes):
        class_mask = pred_cls == c
        n_l = (target_cls == c).sum()
        n_p = class_mask.sum()
        if n_p == 0 or n_l == 0:
            continue

        tp_class = tp[class_mask]
        conf_class = conf[class_mask]

        fpc = (1 - tp_class).cumsum(0)
        tpc = tp_class.cumsum(0)

        # Recall
        recall = tpc / (n_l + eps)
        r_curve[ci] = np.interp(-x, -conf_class, recall[:, 0], left=0)

        # Precision
        precision = tpc / (tpc + fpc)
        p_curve[ci] = np.interp(-x, -conf_class, precision[:, 0], left=1)

        # AP from recall-precision curve
        for j in range(tp.shape[1]):
            ap[ci, j], _, _ = compute_ap(recall[:, j], precision[:, j])

    # Compute F1 (harmonic mean of precision and recall)
    f1_curve = 2 * p_curve * r_curve / (p_curve + r_curve + eps)
    names = [v for k, v in names.items() if k in unique_classes]
    names = dict(enumerate(names))

    i = smooth(f1_curve.mean(0), 0.1).argmax()
    p, r, f1 = p_curve[:, i], r_curve[:, i], f1_curve[:, i]
    tp = (r * (target_cls == unique_classes[:, None]).sum(1)).round()
    fp = (tp / (p + eps) - tp).round()

    return EvaluationMetrics(
        tp, fp, p, r, f1, ap, unique_classes.astype(int), p_curve, r_curve, f1_curve, x
    )


def smooth(y, f=0.05):
    """
    Smooths a sequence using a box filter.

    Args:
        y (np.ndarray): The input sequence to be smoothed.
        f (float): Fraction of the data length to use for the smoothing window, defaults to 0.05.

    Returns:
        np.ndarray: The smoothed sequence.
    """
    if len(y) < 2:
        return y  # No smoothing needed if data is too short
    # Calculate the window size as a fraction of the data length, ensuring it's at least 1
    window_size = max(1, int(round(len(y) * f * 2) // 2 + 1))
    # Extend the sequence at both ends to handle boundary effects
    extended_y = np.pad(y, (window_size // 2, window_size // 2), mode='edge')
    # Apply the box filter using convolution
    smoothed_y = np.convolve(extended_y, np.ones(window_size) / window_size, mode='valid')
    return smoothed_y


class YoloEvalmAPCalculator:
    def __init__(
        self, num_classes=1, is_pose=False, is_seg=False, eval_seg_overlap=True, is_lazy=True
    ) -> None:
        if is_pose:
            assert num_classes == 1, "num_classes should be 1 for pose model"

        self.num_classes = num_classes
        self.is_pose = is_pose
        self.is_seg = is_seg
        self.eval_seg_overlap = eval_seg_overlap
        self.is_lazy = is_lazy

        self.stats = {
            'correct_bboxes': [],
            'scores': [],
            'pred_classes': [],
            'true_classes': [],
        }
        if self.is_pose:
            self.stats['correct_kpts'] = []
        if self.is_seg:
            self.stats['correct_mask'] = []

    def _append_stat(
        self, correct_bboxes, correct_kpts, correct_mask, scores, pred_classes, true_classes
    ):
        self.stats['correct_bboxes'].append(correct_bboxes)
        if self.is_pose:
            self.stats['correct_kpts'].append(correct_kpts)
        if self.is_seg:
            self.stats['correct_mask'].append(correct_mask)
        self.stats['scores'].append(scores)
        self.stats['pred_classes'].append(pred_classes)
        self.stats['true_classes'].append(true_classes)

    def _append_empty_stat(self, num_preds):
        self.stats['correct_bboxes'].append(np.zeros((num_preds, 10), dtype=bool))
        if self.is_pose:
            self.stats['correct_kpts'].append(np.zeros((num_preds, 10), dtype=bool))
        if self.is_seg:
            self.stats['correct_mask'].append(np.zeros((num_preds, 10), dtype=bool))
        self.stats['scores'].append(np.zeros(0))
        self.stats['pred_classes'].append(np.zeros(0))
        self.stats['true_classes'].append(np.zeros(0))

    def _concatenated_stats(self):
        concat_stats = {
            k: np.concatenate(self.stats[k]) if len(self.stats[k]) > 1 else self.stats[k][0]
            for k in self.stats.keys()
        }
        return concat_stats

    def _compute_pred_accuracy(
        self,
        pred_boxes,
        gt_boxes,
        pred_classes,
        true_classes,
        pred_kpts=None,
        gt_kpts=None,
        pred_masks=None,
        gt_masks=None,
    ):
        # fmt: off
        SIGMA = np.array([0.26, 0.25, 0.25, 0.35, 0.35, 0.79, 0.79, 0.72, 0.72, 0.62, 0.62, 1.07, 1.07, 0.87, 0.87, 0.89, 0.89]) / 10.0
        # fmt: on

        """
        Return correct prediction matrix.

        Args:
            pred_boxes (np.ndarray): Array of detection, format x1, y1, x2, y2,
            gt_boxes (np.ndarray): Array of label, format x1, y1, x2, y2
            pred_kpts (np.ndarray, optional): Array of predicted keypoints.
            gt_kpts (np.ndarray, optional): Array of ground truth keypoints.
            pred_masks (np.ndarray, optional): Array of predicted masks.
            gt_masks (np.ndarray, optional): Array of ground truth masks.

        Returns:
            np.ndarray: Correct prediction matrix of shape [N, 10] for 10 IoU levels.
        """
        if pred_kpts is not None and gt_kpts is not None:
            # from https://github.com/jin-s13/xtcocoapi/blob/master/xtcocotools/cocoeval.py#L384
            area = xyxy2xywh(gt_boxes)[:, 2:].prod(1) * 0.53
            iou = kpt_iou(gt_kpts, pred_kpts, sigma=SIGMA, area=area)
        elif pred_masks is not None and gt_masks is not None:
            iou = mask_iou(gt_masks, pred_masks)
        else:  # boxes
            iou = box_iou(gt_boxes, pred_boxes)
        return match_predictions(pred_classes, true_classes, iou)

    def _compute_ap_per_sample(self, preds, targets):
        if (npr := len(preds['boxes'])) == 0:
            self._append_empty_stat(npr)
            return

        if (ntr := len(targets['boxes'])) == 0:
            return

        correct_kpts, correct_mask = None, None
        if self.is_pose:
            pred_classes = np.zeros(len(preds['boxes']), dtype=int)
            true_classes = np.zeros(len(targets['boxes']), dtype=int)
            true_boxes = targets['boxes']
            correct_kpts = self._compute_pred_accuracy(
                preds['boxes'],
                true_boxes,
                pred_classes,
                true_classes,
                pred_kpts=preds['keypoints'],
                gt_kpts=targets['keypoints'],
            )
        elif self.is_seg:
            gt_masks, idx = targets['masks']
            gt_masks = np.expand_dims(gt_masks, axis=0)
            pred_masks = preds['masks']

            pred_classes = preds['labels']
            # reorder the labels and boxes according to the idx
            true_classes = targets['labels'][idx]
            true_boxes = targets['boxes'][idx]

            if self.eval_seg_overlap:
                nl = len(true_classes)
                index = np.arange(nl).reshape(nl, 1, 1) + 1
                gt_masks = np.repeat(gt_masks, nl, axis=0)  # shape(1,640,640) -> (n,640,640)
                gt_masks = np.where(gt_masks == index, 1.0, 0.0)
            if gt_masks.shape[1:] != pred_masks.shape[1:]:
                scale_y = pred_masks.shape[1] / gt_masks.shape[1]
                scale_x = pred_masks.shape[2] / gt_masks.shape[2]

                # Align the size of the ground truth masks to the prediction masks
                new_shape = (pred_masks.shape[1], pred_masks.shape[2])
                # enable align_corners as a trick; it's not really necessary for measurement
                gt_masks = simple_resize_masks(gt_masks, new_shape, align_corners=False)

            gt_masks = gt_masks.reshape(gt_masks.shape[0], -1)
            pred_masks = pred_masks.reshape(pred_masks.shape[0], -1)

            correct_mask = self._compute_pred_accuracy(
                preds['boxes'],
                true_boxes,
                pred_classes,
                true_classes,
                pred_masks=pred_masks,
                gt_masks=gt_masks,
            )
        else:
            pred_classes = preds['labels']
            true_classes = targets['labels']
            true_boxes = targets['boxes']

        correct_bboxes = self._compute_pred_accuracy(
            preds['boxes'], true_boxes, pred_classes, true_classes
        )
        self._append_stat(
            correct_bboxes, correct_kpts, correct_mask, preds['scores'], pred_classes, true_classes
        )

    def summary(self, with_box=True) -> types.EvalResult:
        '''
        Calculate the summary of the evaluation.
        Args:
            with_box: Whether to calculate the summary for the box task.
        Returns:
            A list of tuples, each containing a metric name, an aggregator, and its value.
        '''
        if not with_box:
            assert self.is_pose or self.is_seg, "Only pose or seg can be without box"
        concatenated_stats = self._concatenated_stats()

        mp_box, mr_box, map50_box, map_box = 0.0, 0.0, 0.0, 0.0
        mp_task, mr_task, map50_task, map_task = 0.0, 0.0, 0.0, 0.0
        if with_box:
            if concatenated_stats['correct_bboxes'].any():
                results_box = ap_per_class(
                    concatenated_stats['correct_bboxes'],
                    concatenated_stats['scores'],
                    concatenated_stats['pred_classes'],
                    concatenated_stats['true_classes'],
                )
                p_b, r_b, f1_b, ap_b, ap_class_b = (
                    results_box.precision,
                    results_box.recall,
                    results_box.f1,
                    results_box.ap,
                    results_box.unique_classes,
                )
                ap50, ap = ap_b[:, 0], ap_b.mean(1)
                mp_box, mr_box, map50_box, map_box = p_b.mean(), r_b.mean(), ap50.mean(), ap.mean()

        if self.is_pose:
            if concatenated_stats['correct_kpts'].any():
                results_pose = ap_per_class(
                    concatenated_stats['correct_kpts'],
                    concatenated_stats['scores'],
                    concatenated_stats['pred_classes'],
                    concatenated_stats['true_classes'],
                )
                p_k, r_k, f1_k, ap_k, ap_class_k = (
                    results_pose.precision,
                    results_pose.recall,
                    results_pose.f1,
                    results_pose.ap,
                    results_pose.unique_classes,
                )
                ap50, ap = ap_k[:, 0], ap_k.mean(1)
                mp_task, mr_task, map50_task, map_task = (
                    p_k.mean(),
                    r_k.mean(),
                    ap50.mean(),
                    ap.mean(),
                )
        elif self.is_seg:
            if concatenated_stats['correct_mask'].any():
                results_seg = ap_per_class(
                    concatenated_stats['correct_mask'],
                    concatenated_stats['scores'],
                    concatenated_stats['pred_classes'],
                    concatenated_stats['true_classes'],
                )
                p_s, r_s, f1_s, ap_s, ap_class_s = (
                    results_seg.precision,
                    results_seg.recall,
                    results_seg.f1,
                    results_seg.ap,
                    results_seg.unique_classes,
                )
                ap50, ap = ap_s[:, 0], ap_s.mean(1)
                mp_task, mr_task, map50_task, map_task = (
                    p_s.mean(),
                    r_s.mean(),
                    ap50.mean(),
                    ap.mean(),
                )

        task_name = "mask" if self.is_seg else "pose" if self.is_pose else None
        supported_metric_names = ["mAP", "mAP50", "precision", "recall"]
        aggregators = {"mAP": ["box"], "mAP50": ["box"], "precision": ["box"], "recall": ["box"]}
        if task_name:
            aggregators["mAP"].append(task_name)
            aggregators["mAP50"].append(task_name)
            aggregators["precision"].append(task_name)
            aggregators["recall"].append(task_name)
        key_aggregator = task_name if task_name else "box"
        eval_result = types.EvalResult(supported_metric_names, aggregators, "mAP", key_aggregator)
        results = [
            # metric, aggregator, value
            ("mAP", "box", map_box),
            ("mAP50", "box", map50_box),
            ("precision", "box", mp_box),
            ("recall", "box", mr_box),
        ]
        if task_name:
            task_results = [
                ("mAP", task_name, map_task),
                ("mAP50", task_name, map50_task),
                ("precision", task_name, mp_task),
                ("recall", task_name, mr_task),
            ]
            results.extend(task_results)
        for metric, aggregator, value in results:
            eval_result.set_metric_result(metric, value, aggregator, is_percentage=True)
        return eval_result

    def process(self, predicted, actual) -> None:
        """Processes a single pair of predicted and actual data to compute average precision per sample.

        Args:
            predicted: KptDetEvalSample representing the predicted data.
            actual: KptDetGroundTruthSample representing the actual ground truth data.
        """
        self._compute_ap_per_sample(predicted, actual)

    def __call__(self, predicted, actual) -> List[Tuple[str, float]]:
        """Calculates the mean average precision for each prediction against the actual data.

        Args:
            predicted: List of KptDetEvalSample representing the predicted data.
            actual: List of KptDetGroundTruthSample representing the actual ground truth data.

        Returns:
            List[Tuple[str, float]]: A list of tuples containing metric names and their corresponding values.
        """
        if self.is_lazy:
            raise ValueError(
                "Lazy evaluation is enabled. Please use the process and collect_metrics methods instead."
            )

        for prediction, ground_truth in zip(predicted, actual):
            self._compute_ap_per_sample(prediction, ground_truth)

        return self.summary()


class YoloEvaluator(types.Evaluator):
    """
    Evaluator for YOLO, including object detection, pose estimation and instance segmentation.
    """

    def __init__(self, evaluator: YoloEvalmAPCalculator):
        super().__init__()
        self.evaluator = evaluator

    def process_meta(
        self, meta: Union[InstanceSegmentationMeta, KeypointDetectionMeta, ObjectDetectionMeta]
    ):
        self.evaluator.process(meta.to_evaluation().data, meta.access_ground_truth().data)

    def collect_metrics(self):
        return self.evaluator.summary()

    """
    # here we showed the offline evaluation process as reference
    def __init__(self, evaluator: YoloEvalmAPCalculator):
        super().__init__()
        self.evaluator = evaluator
        self.preds = []
        self.ground_truths = []

    def process_meta(
        self, meta: Union[InstanceSegmentationMeta, KeypointDetectionMeta, ObjectDetectionMeta]
    ):
        if not (ground_truth := meta.access_ground_truth()):
            raise ValueError("Ground truth is not set")

        if isinstance(
            meta, (InstanceSegmentationMeta, KeypointDetectionMeta, ObjectDetectionMeta)
        ):
            self.preds.append(meta.to_evaluation().data)
            self.ground_truths.append(ground_truth.data)
        else:
            raise NotImplementedError(
                f"Ground truth is {type(ground_truth)} which is not supported yet"
            )

    def collect_metrics(self):
        if not self.preds or not self.ground_truths:
            raise ValueError("No valid samples for evaluation")

        if len(self.preds) != len(self.ground_truths):
            LOG.warning(
                f"Mismatch in number of predicted ({len(self.preds)}) and actual ({len(self.ground_truths)}) samples"
            )

        result_box, result_task = self.evaluator(self.preds, self.ground_truths)
        if self.evaluator.is_pose or self.evaluator.is_seg:
            return result_task
        else:
            return result_box
    """

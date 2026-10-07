# Copyright Axelera AI, 2025

import math

import cv2
import numpy as np

from axelera import types
from axelera.app import logging_utils

LOG = logging_utils.getLogger(__name__)


def organize_results(rmse, delta1):
    metric_names = ['RMSE', 'delta1']
    aggregator_dict = {'RMSE': ['average'], 'delta1': ['average']}

    eval_result = types.EvalResult(
        metric_names=metric_names,
        aggregators=aggregator_dict,
        key_metric='RMSE',
        key_aggregator='average',
    )

    eval_result.set_metric_result('RMSE', rmse, 'average', is_percentage=False)
    eval_result.set_metric_result('delta1', delta1, 'average', is_percentage=False)

    return eval_result


def evaluate(output, target):
    valid_mask = (target > 0) & (output > 0)

    output = output[valid_mask]
    target = target[valid_mask]

    abs_diff = np.abs(output - target)

    mse = np.mean(np.square(abs_diff))
    rmse = np.sqrt(mse)

    max_ratio = np.maximum(output / target, target / output)
    delta1 = np.mean(max_ratio < 1.25)

    return rmse, delta1


# Aligning the median over a handful of pixels is meaningless, so an image with
# fewer valid ground-truth pixels than this contributes nothing.
MIN_VALID_PIXELS = 10


def evaluate_aligned(output, target, min_depth, max_depth):
    """Score one image after rescaling the prediction to the target's median.

    A head whose output is determined only up to a scale factor cannot be
    compared against metric ground truth directly: a constant factor alone
    drives delta1 to zero. Rescaling by `median(target) / median(output)`
    removes that factor, which is how Ultralytics reports this model.

    Returns `None` when the image has too few valid pixels to score.
    """
    valid_mask = (target > min_depth) & (target < max_depth)
    if int(valid_mask.sum()) < MIN_VALID_PIXELS:
        return None

    output = output[valid_mask].astype(np.float64)
    target = target[valid_mask].astype(np.float64)

    finite = np.isfinite(output)
    if finite.any():
        denominator = np.median(np.maximum(output[finite], min_depth))
        if denominator > 0:
            output = output * (np.median(target[finite]) / denominator)

    output = np.nan_to_num(output, nan=max_depth, posinf=max_depth, neginf=min_depth)
    output = np.clip(output, min_depth, max_depth)

    rmse = np.sqrt(np.mean(np.square(output - target)))
    delta1 = np.mean(np.maximum(output / target, target / output) < 1.25)

    return rmse, delta1


class DepthEstimationEvaluator(types.Evaluator):
    """Average per-image RMSE and delta1 over the validation split.

    `align='median'` rescales each prediction to the ground truth's median
    before scoring and restricts scoring to ground-truth depths inside
    `(min_depth, max_depth)`. Use it for a model whose published metric is
    reported that way; a model that emits true metric depth and was trained on
    this dataset is scored with the default `align='none'`.
    """

    def __init__(self, align='none', min_depth=1e-3, max_depth=10.0, **kwargs):
        if align not in ('none', 'median'):
            raise ValueError(f"align must be 'none' or 'median', got {align!r}")
        self.align = align
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.total_samples = 0
        self.seen_samples = 0
        self.rmse_sum = 0
        self.delta1_sum = 0

    def process_meta(self, meta) -> None:
        self.seen_samples += 1
        depth = np.array(meta.to_evaluation().data)
        depth = np.squeeze(depth)

        target = np.array(meta.access_ground_truth().data)

        if self.align == 'median':
            # Ultralytics scores the prediction at model resolution and brings
            # the depth map up to it with nearest resampling
            # (`DepthFormat.apply_depth`), so reproduce that rather than moving
            # the prediction.
            target = cv2.resize(
                target, (depth.shape[1], depth.shape[0]), interpolation=cv2.INTER_NEAREST
            )
            scored = evaluate_aligned(depth, target, self.min_depth, self.max_depth)
            if scored is None:
                return
            rmse, delta1 = scored
        else:
            target = cv2.resize(target, (depth.shape[1], depth.shape[0]))
            rmse, delta1 = evaluate(depth, target)

        self.total_samples += 1
        self.rmse_sum += rmse
        self.delta1_sum += delta1

    def collect_metrics(self):
        # RMSE is the key metric and 0.0 is its best value, so averaging nothing
        # would publish a perfect score for a split that was never scored.
        if self.total_samples == 0:
            raise ValueError(
                f'No sample had at least {MIN_VALID_PIXELS} ground-truth pixels inside '
                f'({self.min_depth}, {self.max_depth}) m, so nothing was scored. Check that '
                f'the depth maps are in metres.'
            )
        if self.total_samples < self.seen_samples:
            LOG.warning(
                'Scored %d of %d frames; %d had too few ground-truth pixels inside '
                '(%s, %s) m. The reported numbers are over the scored subset only.',
                self.total_samples,
                self.seen_samples,
                self.seen_samples - self.total_samples,
                self.min_depth,
                self.max_depth,
            )
        avg_rmse = self.rmse_sum / self.total_samples
        avg_delta1 = self.delta1_sum / self.total_samples

        return organize_results(avg_rmse, avg_delta1)

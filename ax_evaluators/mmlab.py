# Copyright Axelera AI, 2024
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple, Union

from mmengine.structures import BaseDataElement, PixelData
import numpy as np
import torch
import torch.nn.functional as TF

from axelera import types

if TYPE_CHECKING:
    from axelera.app.meta import SemanticSegmentationMeta


ONLINE_EVAL = True


def _rescale_seg_logits(
    seg_logits: torch.Tensor, target_size: Tuple[int, int], mode: str = "bilinear"
):
    # Add batch dimension if input is [C, H, W]
    needs_squeeze = seg_logits.dim() == 3
    if needs_squeeze:
        seg_logits = seg_logits.unsqueeze(0)
    resized = TF.interpolate(
        seg_logits,
        size=target_size,
        mode=mode,
        align_corners=False if mode == "bilinear" else None,
    )
    return resized.squeeze(0) if needs_squeeze else resized


def _merge_prediction_with_mmseg_sample(
    data_sample: BaseDataElement,
    class_map: Optional[Union[torch.Tensor, np.ndarray]] = None,
    seg_logits: Optional[Union[torch.Tensor, np.ndarray]] = None,
):
    if class_map is not None:
        if isinstance(class_map, np.ndarray):
            seg_pred = torch.from_numpy(class_map)
        else:
            seg_pred = class_map
        # Ensure class_map is 3D (batch_size, height, width)
        if seg_pred.dim() == 2:
            seg_pred = seg_pred.unsqueeze(0)
        if hasattr(data_sample, 'gt_sem_seg'):
            target_size = data_sample.gt_sem_seg.shape[-2:]  # Get (H, W) from ground truth
            if seg_pred.shape[-2:] != target_size:
                seg_pred = _rescale_seg_logits(seg_pred.float(), target_size, mode="nearest")
            if seg_logits is not None and seg_logits.shape[-2:] != target_size:
                seg_logits = _rescale_seg_logits(seg_logits, target_size, mode="bilinear")

        data_dict = {
            'pred_sem_seg': PixelData(data=seg_pred),
        }

        if seg_logits is not None:
            if isinstance(seg_logits, np.ndarray):
                seg_logits = torch.from_numpy(seg_logits)
            data_dict['seg_logits'] = PixelData(data=seg_logits)

        data_sample.set_data(data_dict)


def _reformat_mmlab_results(dataset_type, results: Dict) -> types.EvalResult:
    """Reformat the results to the standard format.

    Args:
        dataset_type: the type of the dataset
        results (Dict[str, Any]): the results to reformat.

    Returns:
        EvalResult: the reformatted results.
    """
    framework = None
    try:
        from mmseg.datasets import BaseSegDataset

        if issubclass(dataset_type, BaseSegDataset):
            framework = 'mmseg'
    except ImportError:
        pass
    try:
        from mmpose.datasets import BasePoseDataset

        if issubclass(dataset_type, BasePoseDataset):
            framework = 'mmpose'
    except ImportError:
        pass
    try:
        from mmdet.datasets import BaseDetDataset

        if issubclass(dataset_type, BaseDetDataset):
            framework = 'mmdet'
    except ImportError:
        pass
    if framework is None:
        raise ValueError(f"Unsupported datatype: {type(dataset_type)}")

    # Normalize float values by converting percentages to proportions
    for k, v in results.items():
        if isinstance(v, float):
            results[k] = v / 100

    if framework == 'mmseg':
        # TODO: see if we can get class_wise mIoU and nIoU
        # reformatted_results["classes_mIoU"] = results.get("averageScoreClasses", None)
        # reformatted_results["classes_nIoU"] = results.get("averageScoreInstClasses", None)
        # reformatted_results["categories_mIoU"] = results.get("averageScoreCategories", None)
        # reformatted_results["categories_nIoU"] = results.get("averageScoreInstCategories", None)

        # mIoU is now a general term, so no need to specify aggregator
        supported_metric_names = ["mIoU", "Accuracy", "F1_score", "Fscore", "Precision", "Recall"]
        aggregators = {
            "Accuracy": ["mean", "avg"],
            "F1_score": ["mean"],
            "Fscore": ["mean"],
            "Precision": ["mean"],
            "Recall": ["mean"],
        }

        eval_result = types.EvalResult(supported_metric_names, aggregators, "Accuracy", "mean")

        # Explicitly map results to metrics and aggregators
        metric_mapping = {
            "mIoU": "mIoU",
            "mAcc": "Accuracy",
            "aAcc": "Accuracy",
            "mDice": "F1_score",
            "mFscore": "Fscore",
            "mPrecision": "Precision",
            "mRecall": "Recall",
        }

        for result_key, metric_name in metric_mapping.items():
            if result_key in results:
                aggregator = "mean" if result_key != "aAcc" else "avg"
                if metric_name not in aggregators:
                    aggregator = None
                eval_result.set_metric_result(
                    metric_name, results[result_key], aggregator, is_percentage=True
                )

        # Remove metrics from eval_result if they were not found in results
        for metric in supported_metric_names:
            if metric not in [metric_mapping[res] for res in results if res in metric_mapping]:
                eval_result.metrics_result.pop(metric, None)
    elif framework == 'mmpose':
        raise NotImplementedError("Please implement reformating logic for mmpose")
    elif framework == 'mmdet':
        raise NotImplementedError("Please implement reformating logic for mmdet")

    return eval_result


class MMSegEvalSample(types.BaseEvalSample):
    """
    Data element for semantic segmentation evaluations using MMSegmentation evaluators.
    """

    def __init__(
        self,
        data_sample: BaseDataElement,
        class_map: Optional[Union[torch.Tensor, np.ndarray]] = None,
        seg_logits: Optional[Union[torch.Tensor, np.ndarray]] = None,
    ):
        self.data_sample = data_sample.clone()
        _merge_prediction_with_mmseg_sample(self.data_sample, class_map, seg_logits)

    @property
    def data(self) -> Union[Any, Dict[str, Any]]:
        return self.data_sample


class MMSegEvaluator(types.Evaluator):
    """
    Evaluator for MMSeg, which is a popular semantic segmentation framework.
    """

    def __init__(self, evaluator: Any, dataset_type: Any):
        super().__init__()
        self.num_eval_samples = 0
        self.dataset_type = dataset_type
        self.evaluator = evaluator
        if not ONLINE_EVAL:
            self.results = []
            from ax_datasets.mmseg import MMSegEvalSample

    def process_meta(self, meta: SemanticSegmentationMeta):
        """
        Directly build data_sample from meta and process.
        We can also build data_sample from meta and process in one go, e.g.,
          sample = self.to_eval_sample(meta)
          self.evaluator.process(data_samples=[sample.data])
        """
        if ONLINE_EVAL:
            self.num_eval_samples += 1
            if not (ground_truth := meta.access_ground_truth()):
                raise ValueError("Ground truth is not set")
            data_sample = ground_truth.data
            _merge_prediction_with_mmseg_sample(data_sample, meta.class_map, meta.seg_logits)
            self.evaluator.process(data_samples=[data_sample])
        else:
            if not (ground_truth := meta.access_ground_truth()):
                raise ValueError("Ground truth is not set")

            if isinstance(ground_truth.data, BaseDataElement):
                self.results.append(
                    MMSegEvalSample(
                        data_sample=ground_truth.data,
                        seg_logits=meta.seg_logits,
                        class_map=meta.class_map,
                    ).data
                )
            else:
                raise NotImplementedError(
                    f"Ground truth is {type(ground_truth.data)} which is not supported yet"
                )

    def collect_metrics(self):
        if ONLINE_EVAL:
            result = self.evaluator.evaluate(self.num_eval_samples)
            return _reformat_mmlab_results(self.dataset_type, result)
        else:
            result = self.evaluator.offline_evaluate(data_samples=self.results)
            return _reformat_mmlab_results(self.dataset_type, result)

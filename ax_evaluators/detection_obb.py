# Copyright Axelera AI, 2025

import cv2
import numpy as np

if np.lib.NumpyVersion(np.__version__) >= '2.0.0':
    from numpy import trapezoid as trapz
else:
    from numpy import trapz as trapz
from tqdm import tqdm

from axelera import types
from axelera.app import logging_utils
from axelera.app.model_utils import box as box_utils
from axelera.app.model_utils.box import batch_probiou

LOG = logging_utils.getLogger(__name__)


def ap_from_pr(precision: np.ndarray, recall: np.ndarray) -> float:
    """
    Compute AP from precision-recall arrays using Ultralytics/COCO-compatible interpolation.
    - Adds sentinel values
    - Applies precision envelope
    - Integrates interpolated precision over recall at 101 points
    """
    precision = np.asarray(precision, dtype=np.float64)
    recall = np.asarray(recall, dtype=np.float64)
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([1.0], precision, [0.0]))
    # precision envelope
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    # 101-point interpolation over recall
    x = np.linspace(0.0, 1.0, 101)
    ap = trapz(np.interp(x, mrec, mpre), x)
    return float(ap)


# Local AP computation compatible with Ultralytics ap_per_class (minimal output)
# tp: (Npred, T) bool/float, conf: (Npred,), pred_cls: (Npred,), target_cls: (Ngt,)
# Returns: ap (nc_present, T), unique_classes (nc_present,)
def ap_per_class(
    tp: np.ndarray,
    conf: np.ndarray,
    pred_cls: np.ndarray,
    target_cls: np.ndarray,
    iou_thresholds: np.ndarray,
    eps: float = 1e-16,
):
    # Sort by descending confidence
    order = np.argsort(-conf)
    tp = tp[order].astype(float)
    conf = conf[order]
    pred_cls = pred_cls[order]
    # Unique classes present in targets
    unique_classes, nt = (
        np.unique(target_cls, return_counts=True)
        if target_cls.size
        else (np.array([], dtype=int), np.array([], dtype=int))
    )
    T = iou_thresholds.size
    nc = unique_classes.shape[0]
    ap = np.zeros((nc, T), dtype=float)
    for ci, c in enumerate(unique_classes):
        i = pred_cls == c
        n_l = int(nt[ci])
        n_p = int(i.sum())
        if n_p == 0 or n_l == 0:
            continue
        tpc = tp[i].cumsum(0)
        fpc = (1.0 - tp[i]).cumsum(0)
        recall = tpc / (n_l + eps)
        precision = tpc / (tpc + fpc + eps)
        # AP per IoU threshold via 101-point interpolation
        for j in range(T):
            ap[ci, j] = ap_from_pr(precision[:, j], recall[:, j])
    return ap, unique_classes


def evaluate_map_obb(preds, gts, iou_thresholds=None, nc=15):
    """
    Ultralytics-style evaluation using probabilistic IoU (probiou) and greedy matching.
    Vectorized per-image IoU + greedy assignment per IoU threshold.
    """
    if iou_thresholds is None:
        iou_thresholds = np.linspace(0.50, 0.95, 10)

    imgs = list(gts.keys())
    preds_by_img = {img: [] for img in imgs}
    for p in preds:
        preds_by_img.setdefault(p['img'], []).append(p)

    tps_all, conf_all, predc_all, targc_all = [], [], [], []

    for img in tqdm(imgs, total=len(imgs), desc="IoU/match"):
        pi = preds_by_img.get(img, [])
        if len(pi):
            pred_boxes = np.stack([p['box'] for p in pi], axis=0).astype(np.float64)
            pred_conf = np.array([p['conf'] for p in pi], dtype=np.float64)
            pred_cls = np.array([p['cls'] for p in pi], dtype=np.int64)
        else:
            pred_boxes = np.zeros((0, 5), dtype=np.float64)
            pred_conf = np.zeros((0,), dtype=np.float64)
            pred_cls = np.zeros((0,), dtype=np.int64)

        gi = gts[img]
        gt_boxes = []
        gt_cls = []
        for c, boxes in gi.items():
            for b in boxes:
                gt_boxes.append(b)
                gt_cls.append(c)
        if len(gt_boxes):
            gt_boxes = np.stack(gt_boxes, axis=0).astype(np.float64)
            gt_cls = np.array(gt_cls, dtype=np.int64)
        else:
            gt_boxes = np.zeros((0, 5), dtype=np.float64)
            gt_cls = np.zeros((0,), dtype=np.int64)

        Nd = pred_boxes.shape[0]
        Ng = gt_boxes.shape[0]
        targc_all.append(gt_cls)
        conf_all.append(pred_conf)
        predc_all.append(pred_cls)

        if Nd == 0:
            tps_all.append(np.zeros((0, iou_thresholds.size), dtype=bool))
            continue

        if Ng:
            gt_eval = gt_boxes.copy()
            iou = batch_probiou(gt_eval, pred_boxes)  # (Ng, Nd)
        else:
            iou = np.zeros((0, Nd), dtype=np.float64)

        correct = np.zeros((Nd, iou_thresholds.size), dtype=bool)
        if Ng:
            correct_class = gt_cls[:, None] == pred_cls[None, :]
            iou = iou * correct_class.astype(iou.dtype)

            for ti, thr in enumerate(iou_thresholds):
                rows, cols = np.nonzero(iou >= thr)
                if rows.size:
                    matches = np.stack([rows, cols], axis=1)
                    order = np.argsort(iou[rows, cols])[::-1]
                    matches = matches[order]
                    _, ui = np.unique(matches[:, 1], return_index=True)
                    matches = matches[ui]
                    _, ui = np.unique(matches[:, 0], return_index=True)
                    matches = matches[ui]
                    if matches.size:
                        correct[matches[:, 1].astype(int), ti] = True

        tps_all.append(correct)

    target_cls_cat = (
        np.concatenate(targc_all, axis=0) if targc_all else np.zeros((0,), dtype=np.int64)
    )
    conf_cat = np.concatenate(conf_all, axis=0) if conf_all else np.zeros((0,), dtype=np.float64)
    pred_cls_cat = (
        np.concatenate(predc_all, axis=0) if predc_all else np.zeros((0,), dtype=np.int64)
    )
    correct_cat = (
        np.concatenate(tps_all, axis=0)
        if tps_all
        else np.zeros((0, iou_thresholds.size), dtype=bool)
    )

    ap_local, _ = ap_per_class(correct_cat, conf_cat, pred_cls_cat, target_cls_cat, iou_thresholds)
    mAP5095 = float(ap_local.mean()) if ap_local.size else 0.0
    mAP50 = float(ap_local[:, 0].mean()) if ap_local.size else 0.0

    return {"mAP50-95": mAP5095, "mAP50": mAP50}


class DetectionOBBEvaluator(types.Evaluator):
    def __init__(self, **kwargs):
        self.preds = []
        self.gts = {}
        self.num_of_classes = -1

    def process_meta(self, meta) -> None:
        prediction = meta.to_evaluation().data

        target = meta.access_ground_truth().data
        name = target["img_id"]
        for b, conf, kls in zip(prediction['boxes'], prediction['scores'], prediction['labels']):
            obb = np.array(b, dtype=np.float32)
            self.preds.append({"img": name, "cls": int(kls), "conf": float(conf), "box": obb})

        gt_boxes, gt_cls = target['boxes'], target['labels']

        if len(gt_boxes):
            # gt_boxes: list of (4,2) corner arrays
            gt_corners = np.stack(gt_boxes, axis=0).astype(np.float32)  # (N,4,2)
            gt_cls = np.asarray(gt_cls, dtype=np.int64)
            gt_flat = gt_corners.reshape(-1, 8)  # (N,8)
            gt_xywhr = box_utils.convert(
                gt_flat, types.BoxFormat.XYXYXYXY, types.BoxFormat.XYWHR
            )  # (N,5)
            gt_xywhr = np.ascontiguousarray(gt_xywhr, dtype=np.float32)
        else:
            gt_cls = np.zeros((0,), dtype=np.int64)
            gt_xywhr = np.zeros((0, 5), dtype=np.float32)

        if self.num_of_classes < 0:
            self.num_of_classes = len(meta.labels)

        # Per-class dictionary: list of (5,) arrays
        per = {
            c: [gt_xywhr[i] for i in range(len(gt_cls)) if int(gt_cls[i]) == c]
            for c in range(self.num_of_classes)
        }
        self.gts[name] = per

    def collect_metrics(self):
        assert self.num_of_classes > 0, "num_of_classes must be set."

        metrics = evaluate_map_obb(
            self.preds,
            self.gts,
            iou_thresholds=np.arange(0.50, 0.96, 0.05),
            nc=self.num_of_classes,
        )

        metric_names = ['mAP50-95', 'mAP50']
        aggregator_dict = {'mAP50-95': ['average'], 'mAP50': ['average']}

        eval_result = types.EvalResult(
            metric_names=metric_names,
            aggregators=aggregator_dict,
            key_metric='mAP50-95',
            key_aggregator='average',
        )

        eval_result.set_metric_result(
            'mAP50-95', metrics['mAP50-95'], 'average', is_percentage=True
        )
        eval_result.set_metric_result('mAP50', metrics['mAP50'], 'average', is_percentage=True)

        return eval_result

# Copyright Axelera AI, 2024
# Wrapper of types.Evaluator for existing WiderFace evaluator
from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import tqdm

from axelera import types
from axelera.app import logging_utils
from axelera.app.meta.keypoint import FaceLandmarkLocalizationMeta

LOG = logging_utils.getLogger(__name__)


def get_bbox_overlaps():
    try:
        from .bbox import bbox_overlaps

        return bbox_overlaps
    except ImportError:
        LOG.trace("Cython bbox_overlaps not available. Compiling on the fly.")
        import logging

        import numpy as np
        import pyximport

        # https://github.com/cython/cython/issues/5380
        # pyximport affects logging level; we should get rid of it after
        # having a proper cython build system
        original_level = logging.root.level
        try:
            pyximport.install(setup_args={"include_dirs": np.get_include()})
            from .bbox import bbox_overlaps

            return bbox_overlaps
        finally:
            logging.root.setLevel(original_level)


bbox_overlaps = get_bbox_overlaps()


def load_mat_file(file_path: Path):
    # not able to use h5py to load; the mat file probalby older than MATLAB 7.3
    # with h5py.File(file_path, 'r') as file:
    #     return {key: file[key][:] for key in file.keys()}

    try:
        import scipy.io
    except ImportError:
        raise ImportError("scipy not available. Please install it by `pip install scipy`")

    try:
        mat = scipy.io.loadmat(file_path)  # struct_as_record=False, squeeze_me=True)
    except Exception as e:
        # LOG.error(f"Failed to load mat file {file_path}: {e}")
        raise e
    dict_output = dict()
    for key, value in mat.items():
        if not key.startswith('_'):
            dict_output[key] = value
    return dict_output


def _get_gt_boxes(gt_dir: Path):
    gt_mat = load_mat_file(gt_dir / 'wider_face_val.mat')
    hard_mat = load_mat_file(gt_dir / 'wider_hard_val.mat')
    medium_mat = load_mat_file(gt_dir / 'wider_medium_val.mat')
    easy_mat = load_mat_file(gt_dir / 'wider_easy_val.mat')

    return (
        gt_mat['face_bbx_list'],
        gt_mat['event_list'],
        gt_mat['file_list'],
        hard_mat['gt_list'],
        medium_mat['gt_list'],
        easy_mat['gt_list'],
    )


def get_gt_boxes_from_txt(gt_path: Path, cache_dir: Path):
    cache_file = cache_dir / 'gt_cache.pkl'
    if cache_file.exists():
        with cache_file.open('rb') as f:
            return pickle.load(f)

    with gt_path.open('r') as f:
        lines = f.read().strip().split('\n')
    print(len(lines))

    boxes = {}
    current_name = None
    current_boxes = []
    for line in lines:
        if '--' in line:
            if current_name is not None:
                boxes[current_name] = np.array(current_boxes, dtype='float32')
            current_name = line.strip()
            current_boxes = []
        else:
            box = list(map(float, line.split()[:4]))
            current_boxes.append(box)
    boxes[current_name] = np.array(current_boxes, dtype='float32')

    with cache_file.open('wb') as f:
        pickle.dump(boxes, f)
    return boxes


def _norm_score(pred):
    # pred {key: [[x1,y1,x2,y2,s]]}
    max_score = max(max(np.max(v[:, -1]) for v in k.values() if len(v) > 0) for k in pred.values())
    min_score = min(min(np.min(v[:, -1]) for v in k.values() if len(v) > 0) for k in pred.values())
    diff = max_score - min_score
    for k in pred.values():
        for v in k.values():
            if len(v) > 0:
                v[:, -1] = (v[:, -1] - min_score) / diff


def _image_eval(pred, gt, ignore, iou_thresh):
    """single image evaluation
    pred: Nx5
    gt: Nx4
    ignore:
    """
    _pred = pred.copy()
    if _pred.dtype == np.float64:
        _pred = _pred.astype(np.float32)
    _gt = gt.copy()
    pred_recall = np.zeros(_pred.shape[0])
    recall_list = np.zeros(_gt.shape[0])
    proposal_list = np.ones(_pred.shape[0])

    # _pred[:, 2] = _pred[:, 2] + _pred[:, 0]
    # _pred[:, 3] = _pred[:, 3] + _pred[:, 1]
    _gt[:, 2] = _gt[:, 2] + _gt[:, 0]
    _gt[:, 3] = _gt[:, 3] + _gt[:, 1]

    overlaps = bbox_overlaps(_pred[:, :4], _gt)

    for h in range(_pred.shape[0]):
        gt_overlap = overlaps[h]
        max_overlap, max_idx = gt_overlap.max(), gt_overlap.argmax()
        if max_overlap >= iou_thresh:
            if ignore[max_idx] == 0:
                recall_list[max_idx] = -1
                proposal_list[h] = -1
            elif recall_list[max_idx] == 0:
                recall_list[max_idx] = 1

        r_keep_index = np.where(recall_list == 1)[0]
        pred_recall[h] = len(r_keep_index)
    return pred_recall, proposal_list


def _img_pr_info(thresh_num, pred_info, proposal_list, pred_recall):
    pr_info = np.zeros((thresh_num, 2)).astype('float')
    for t in range(thresh_num):

        thresh = 1 - (t + 1) / thresh_num
        r_index = np.where(pred_info[:, 4] >= thresh)[0]
        if len(r_index) == 0:
            pr_info[t, 0] = 0
            pr_info[t, 1] = 0
        else:
            r_index = r_index[-1]
            p_index = np.where(proposal_list[: r_index + 1] == 1)[0]
            pr_info[t, 0] = len(p_index)
            pr_info[t, 1] = pred_recall[r_index]
    return pr_info


def _dataset_pr_info(thresh_num, pr_curve, count_face):
    _pr_curve = np.zeros((thresh_num, 2)).astype('float')
    for i in range(thresh_num):
        _pr_curve[i, 0] = pr_curve[i, 1] / pr_curve[i, 0]
        _pr_curve[i, 1] = pr_curve[i, 1] / count_face
    return _pr_curve


def _voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))

    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    i = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def _read_pred_file(filepath: Path):
    with filepath.open('r') as f:
        lines = f.readlines()
    img_file = lines[0].rstrip('\n\r')
    boxes = []
    for line in lines[2:]:
        if line.strip():
            boxes.append(list(map(float, line.split()[:5])))
    return img_file.split('/')[-1], np.array(boxes, dtype='float32')


def _get_preds_from_dir(pred_dir: Path):
    boxes = {}
    for event_dir in tqdm.tqdm(pred_dir.iterdir(), desc='Reading Predictions'):
        boxes[event_dir.name] = {}
        for img_txt in event_dir.iterdir():
            img_name, _boxes = _read_pred_file(img_txt)
            boxes[event_dir.name][img_name.rstrip('.jpg')] = _boxes
    return boxes


def _form_pred_sample_online(bboxes, scores, image_path):
    event_name = image_path.parent.name
    img_name = image_path.stem
    return event_name, img_name, np.column_stack((bboxes, scores))


def evaluation(pred, gt_dir: Path, iou_thresh=0.5):
    _norm_score(pred)
    (
        facebox_list,
        event_list,
        file_list,
        hard_gt_list,
        medium_gt_list,
        easy_gt_list,
    ) = _get_gt_boxes(gt_dir)
    settings = ['easy', 'medium', 'hard']
    setting_gts = [easy_gt_list, medium_gt_list, hard_gt_list]
    aps = []
    thresh_num = 1000

    total_count_face = 0
    total_pr_curve = np.zeros((thresh_num, 2)).astype('float')

    for setting_id in range(3):
        gt_list = setting_gts[setting_id]
        count_face = 0
        pr_curve = np.zeros((thresh_num, 2))

        for i in tqdm.tqdm(range(len(event_list)), desc=f'Processing {settings[setting_id]}'):
            event_name = str(event_list[i][0][0])

            if event_name not in pred:
                continue

            img_list = file_list[i][0]
            pred_list = pred[event_name]
            sub_gt_list = gt_list[i][0]
            gt_bbx_list = facebox_list[i][0]

            for j in range(len(img_list)):
                file_name = str(img_list[j][0][0])
                if file_name not in pred_list:
                    continue
                pred_info = pred_list[file_name]

                gt_boxes = gt_bbx_list[j][0].astype(np.float32)
                keep_index = sub_gt_list[j][0]
                count_face += len(keep_index)

                if len(gt_boxes) == 0 or len(pred_info) == 0:
                    continue
                ignore = np.zeros(gt_boxes.shape[0])
                if len(keep_index) != 0:
                    ignore[keep_index - 1] = 1
                pred_recall, proposal_list = _image_eval(pred_info, gt_boxes, ignore, iou_thresh)
                pr_curve += _img_pr_info(thresh_num, pred_info, proposal_list, pred_recall)

        total_count_face += count_face
        total_pr_curve += pr_curve

        pr_curve = _dataset_pr_info(thresh_num, pr_curve, count_face)
        propose = pr_curve[:, 0]
        recall = pr_curve[:, 1]
        ap = _voc_ap(recall, propose)
        aps.append(ap)

    total_pr_curve = _dataset_pr_info(thresh_num, total_pr_curve, total_count_face)
    total_propose = total_pr_curve[:, 0]
    total_recall = total_pr_curve[:, 1]
    total_ap = _voc_ap(total_recall, total_propose)
    return total_ap, aps


class WiderFaceEvaluator(types.Evaluator):
    """
    Evaluator specific for WiderFace dataset. This is not a general purpose evaluator because it uses ground truth from WideFace matlab files. It can be customized if you follow the same structure to
    build your own matlab ground truth.
    """

    def __init__(self, dataset_root):
        super().__init__()
        self.detections = dict()
        self.ground_truths = []
        self.dataset_root = dataset_root

    def process_meta(self, meta: FaceLandmarkLocalizationMeta):
        if not (ground_truth := meta.access_ground_truth()):
            raise ValueError("Ground truth is not set")

        file_path = Path(meta.access_image_id())
        event_name, img_name, det = _form_pred_sample_online(meta.boxes, meta.scores, file_path)
        if event_name not in self.detections:
            self.detections[event_name] = {}
        self.detections[event_name][img_name] = det
        self.ground_truths.append(ground_truth.data[:, :4])

    def collect_metrics(self) -> types.EvalResult:
        total_ap, aps = evaluation(self.detections, gt_dir=self.dataset_root / 'wider_face_split')

        supported_metric_names = ["mAP"]
        aggregators = {"mAP": ["Total", "Easy", "Medium", "Hard"]}
        eval_result = types.EvalResult(supported_metric_names, aggregators, "mAP", "Easy")
        results = [
            # metric, aggregator, value
            (supported_metric_names[0], "Total", total_ap),
            (supported_metric_names[0], "Easy", aps[0]),
            (supported_metric_names[0], "Medium", aps[1]),
            (supported_metric_names[0], "Hard", aps[2]),
        ]
        for metric, aggregator, value in results:
            eval_result.set_metric_result(metric, value, aggregator, is_percentage=True)
        return eval_result

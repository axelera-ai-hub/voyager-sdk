# Copyright Axelera AI, 2024
# Operators that convert retinaface-specific tensor output to
# generalized metadata representation

from itertools import product
from math import ceil
from pathlib import Path
import re
from typing import List, Optional

import numpy as np

from axelera import types
from axelera.app import gst_builder, logging_utils
from axelera.app.meta import BBoxState, FaceLandmarkLocalizationMeta
from axelera.app.operators import AxOperator, PipelineContext

LOG = logging_utils.getLogger(__name__)

NUM_LOC_COORDS = 4
NUM_CONF_CLASSES = 2  # background and face
NUM_LANDMARK_PAIRS = 5


def _filter_samples(threshold, *arrays):
    """
    Filters samples based on the first array (scores) with threshold.

    Args:
        threshold (float): The threshold value for filtering.
        *arrays: Variable number of numpy arrays to filter.

    Returns:
        tuple: Filtered arrays.
    """
    if not arrays:
        return tuple()

    scores = arrays[0]
    valid_indices = scores > threshold

    return tuple(arr[valid_indices] for arr in arrays)


def decode_loc(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (numpy.ndarray): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (numpy.ndarray): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    if len(loc) == 0 or len(priors) == 0:
        return np.empty((0, NUM_LOC_COORDS))

    try:
        boxes = np.concatenate(
            (
                priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
                priors[:, 2:] * np.exp(loc[:, 2:] * variances[1]),
            ),
            axis=1,
        )
    except Exception as e:
        from pdb import set_trace

        set_trace()
    return boxes


def decode_landm(pre, priors, variances):
    """Decode landm from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        pre (numpy.ndarray): landm predictions for loc layers,
            Shape: [num_priors,10]
        priors (numpy.ndarray): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded landm predictions
    """
    if len(pre) == 0 or len(priors) == 0:
        return np.empty((0, NUM_LANDMARK_PAIRS * 2))

    landms = np.concatenate(
        (
            priors[:, :2] + pre[:, :2] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 2:4] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 4:6] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 6:8] * variances[0] * priors[:, 2:],
            priors[:, :2] + pre[:, 8:10] * variances[0] * priors[:, 2:],
        ),
        axis=1,
    )
    return landms


def generate_priors(cfg, image_size):
    """Generate prior boxes based on the configuration.
    Args:
        cfg (dict): Configuration dictionary.
        image_size (tuple): Image size (height, width).
    Returns:
        numpy.ndarray: Prior boxes in center-offset form.
    """
    min_sizes = cfg['min_sizes']
    steps = cfg['steps']
    clip = cfg['clip']
    feature_maps = [[ceil(image_size[0] / step), ceil(image_size[1] / step)] for step in steps]

    anchors = []
    for k, f in enumerate(feature_maps):
        min_sizes_k = min_sizes[k]
        for i, j in product(range(f[0]), range(f[1])):
            for min_size in min_sizes_k:
                s_kx = min_size / image_size[1]
                s_ky = min_size / image_size[0]
                dense_cx = [x * steps[k] / image_size[1] for x in [j + 0.5]]
                dense_cy = [y * steps[k] / image_size[0] for y in [i + 0.5]]
                for cy, cx in product(dense_cy, dense_cx):
                    anchors.append([cx, cy, s_kx, s_ky])

    output = np.array(anchors)
    if clip:
        output = np.clip(output, 0, 1)
    return output


class DecodeRetinaface(AxOperator):
    """
    Decoding bounding boxes and add model info into Axelera metadata

    Input:
        predict: batched predictions
        kwargs: model info
    Output:
        list of BboxesMeta mapping to each image
    """

    box_format: str
    normalized_coord: bool
    labels: Optional[List[str]] = None
    label_filter: Optional[List[str]] = None
    conf_threshold: float = 0.25
    max_nms_boxes: int = 30000
    num_classes: int = 80
    nms_iou_threshold: float = 0.45
    nms_class_agnostic: bool = False
    nms_top_k: int = 300

    def _post_init(self):
        if isinstance(self.label_filter, str) and not self.label_filter.startswith('$$'):
            stripped = (self.label_filter or '').strip()
            self.label_filter = [x for x in re.split(r'\s*[,;]\s*', stripped) if x]
        else:
            self.label_filter = []
        self._tmp_labels: Optional[Path] = None
        if self.box_format not in ["xyxy", "xywh", "ltwh"]:
            raise ValueError(f"Unknown box format {self.box_format}")
        self.use_multi_label = False
        self.gst_decoder_does_dequantization_and_depadding = True
        super()._post_init()

    def __del__(self):
        if self._tmp_labels is not None and self._tmp_labels.exists():
            self._tmp_labels.unlink()

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        where: str,
        compiled_model_dir: Path,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, where, compiled_model_dir
        )
        if model_info.manifest and model_info.manifest.is_compiled():
            self._deq_scales, self._deq_zeropoints = zip(*model_info.manifest.dequantize_params)
            self._postprocess_graph = model_info.manifest.postprocess_graph
            self._n_padded_ch_outputs = model_info.manifest.n_padded_ch_outputs
        self.scaled = context.resize_status
        self.model_width = model_info.input_width
        self.model_height = model_info.input_height
        self.cfg = model_info.extra_kwargs['RetinaFace']['cfg']

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):

        if not gst.new_inference:
            conns = {'src': f'decoder_task{self._taskn}{stream_idx}.sink_0'}
            gst.queue(name=f'queue_decoder_task{self._taskn}{stream_idx}', connections=conns)

        scales = ','.join(str(s) for s in self._deq_scales)
        zeros = ','.join(str(s) for s in self._deq_zeropoints)
        padding = '|'.join(
            ','.join(str(num) for num in sublist) for sublist in self._n_padded_ch_outputs
        )
        steps = ','.join(str(s) for s in self.cfg['steps'])
        min_sizes = '|'.join(
            ','.join(str(num) for num in sublist) for sublist in self.cfg['min_sizes']
        )
        variances = ','.join(str(s) for s in self.cfg['variance'])
        clip = int(self.cfg['clip'])

        gst.decode_muxer(
            name=f'decoder_task{self._taskn}{stream_idx}',
            lib='libdecode_retinaface.so',
            mode='read',
            options=f'meta_key:{str(self.task_name)};'
            f'width:{self.model_width};'
            f'height:{self.model_height};'
            f'padding:{padding};'
            f'scales:{scales};'
            f'zero_points:{zeros};'
            f'transpose:1;'
            f'confidence_threshold:{self.conf_threshold};'
            f'steps:{steps};'
            f'min_sizes:{min_sizes};'
            f'variances:{variances};'
            f'clip:{clip};'
            f'scale_up:{int(self.scaled==types.ResizeMode.LETTERBOX_FIT)};'
            f'decoder_name:FaceLandmarkLocalizationMeta;',
        )
        gst.axinplace(
            lib='libinplace_nms.so',
            options=f'meta_key:{str(self.task_name)};'
            f'max_boxes:{self.nms_top_k};'
            f'nms_threshold:{self.nms_iou_threshold};'
            f'class_agnostic:{int(self.nms_class_agnostic)};'
            f'location:CPU',
        )

    def exec_torch(self, image, predict, meta):
        loc = next(p for p in predict if p.shape[-1] == NUM_LOC_COORDS)
        conf = next(p for p in predict if p.shape[-1] == NUM_CONF_CLASSES)
        landmarks = next(p for p in predict if p.shape[-1] == NUM_LANDMARK_PAIRS * 2)

        loc = loc.cpu().detach().numpy()
        conf = conf.cpu().detach().numpy()
        landmarks = landmarks.cpu().detach().numpy()

        priors = generate_priors(self.cfg, (self.model_height, self.model_width))

        boxes = decode_loc(loc[0], priors, self.cfg['variance'])
        scores = conf[0, :, 1]
        landms = decode_landm(landmarks[0], priors, self.cfg['variance'])

        # Filter samples by object confidence threshold
        scores, landms, boxes = _filter_samples(self.conf_threshold, scores, landms, boxes)

        if self._where:
            master_meta = meta[self._where]
            # get boxes of the last secondary frame index
            base_box = master_meta.boxes[
                master_meta.get_next_secondary_frame_index(self.task_name)
            ]
            src_img_width = base_box[2] - base_box[0]
            src_img_height = base_box[3] - base_box[1]
        else:
            src_img_width = image.size[0]
            src_img_height = image.size[1]

        state = BBoxState(
            self.model_width,
            self.model_height,
            src_img_width,
            src_img_height,
            self.box_format,
            self.normalized_coord,
            self.scaled,
            self.max_nms_boxes,
            self.nms_iou_threshold,
            self.nms_class_agnostic,
            self.nms_top_k,
        )

        boxes, scores, landms = state.organize_bboxes_and_kpts(boxes, scores, landms)

        if self._where:
            boxes[:, [0, 2]] += base_box[0]
            boxes[:, [1, 3]] += base_box[1]

        model_meta = FaceLandmarkLocalizationMeta(
            keypoints=landms.astype(np.float32),
            boxes=boxes.astype(np.float32),
            scores=scores.astype(np.float32),
        )
        meta.add_instance(self.task_name, model_meta, self._where)

        return image, predict, meta

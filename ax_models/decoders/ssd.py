# Copyright Axelera AI, 2023
# Operators that convert model-specific tensor output to
# generalized metadata representation
from __future__ import annotations

from pathlib import Path
import re
from typing import Optional, Union

import numpy as np

from axelera import types
from axelera.app import gst_builder, logging_utils
from axelera.app.meta import BBoxState, ObjectDetectionMeta
from axelera.app.operators import AxOperator, PipelineContext, utils
from axelera.app.torch_utils import torch

LOG = logging_utils.getLogger(__name__)


def _output_swap_column(x):
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., [0, 1, 2, 3]] = y[..., [1, 0, 3, 2]]
    return y


class DecodeSsdMobilenet(AxOperator):
    """
    Decoding bounding boxes and add model info into Axelera metadata

    Output:
        Metadata list
    """

    box_format: str
    normalized_coord: bool
    label_filter: Union[list, str] = []
    conf_threshold: float = 0.25
    max_nms_boxes: int = 30000
    nms_iou_threshold: float = 0.45
    nms_class_agnostic: bool = False
    nms_top_k: int = 300
    overwrite_labels: bool = False

    def _post_init(self):
        if self.box_format not in ["xyxy", "xywh", "ltwh"]:
            raise ValueError(f"Unknown box format {self.box_format}")
        if isinstance(self.overwrite_labels, int):
            self.overwrite_labels = bool(self.overwrite_labels)
        if isinstance(self.label_filter, str) and not self.label_filter.startswith('$$'):
            stripped = (self.label_filter or '').strip()
            self.label_filter = [x for x in re.split(r'\s*[,;]\s*', stripped) if x]
        else:
            self.label_filter = []
        self._tmp_labels: Optional[Path] = None
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
            self._n_padded_ch_outputs = model_info.manifest.n_padded_ch_outputs
        self.scaled = context.resize_status
        self.model_width = model_info.input_width
        self.model_height = model_info.input_height
        self.labels = model_info.labels
        self.num_classes = model_info.num_classes

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        if self._tmp_labels is None:
            self._tmp_labels = utils.create_tmp_labels(self.labels)

        if not gst.new_inference:
            conns = {'src': f'decoder_task{self._taskn}{stream_idx}.sink_0'}
            gst.queue(name=f'queue_decoder_task{self._taskn}{stream_idx}', connections=conns)

        scales = ','.join(str(s) for s in self._deq_scales)
        zeros = ','.join(str(s) for s in self._deq_zeropoints)
        if self._n_padded_ch_outputs:
            paddings = '|'.join(
                ','.join(str(num) for num in sublist) for sublist in self._n_padded_ch_outputs
            )
        sieve = utils.build_class_sieve(self.label_filter, self._tmp_labels)

        gst.decode_muxer(
            name=f'decoder_task{self._taskn}{stream_idx}',
            lib='libdecode_ssd2.so',
            mode='read',
            options=f'meta_key:{str(self.task_name)};'
            f'confidence_threshold:{self.conf_threshold};'
            f'classes:{self.num_classes};'
            f'classlabels_file:{self._tmp_labels};'
            f'max_boxes:{self.max_nms_boxes};'
            f'scales:{scales};'
            f'zero_points:{zeros};'
            f'transpose:1;'
            f'class_agnostic:{int(self.nms_class_agnostic)}'
            f'model_width:{self.model_width};'
            f'model_height:{self.model_height};'
            f'scale_up:{int(self.scaled==types.ResizeMode.LETTERBOX_FIT)};'
            f'letterbox:{int(self.scaled in [types.ResizeMode.LETTERBOX_FIT, types.ResizeMode.LETTERBOX_CONTAIN])}'
            + (f';padding:{paddings}' if self._n_padded_ch_outputs else '')
            + (f';label_filter:{",".join(sieve)}' if sieve else ''),
        )
        gst.axinplace(
            lib='libinplace_nms.so',
            options=f'meta_key:{str(self.task_name)};nms_threshold:{self.nms_iou_threshold};class_agnostic:{int(self.nms_class_agnostic)};max_boxes:{self.nms_top_k}',
        )

    def exec_torch(self, image, predict, meta):
        if len(predict) == 12:
            bboxes_xyxy, pred_scores = _multibox_layer(predict, len(self.labels))
        elif len(predict) == 2:
            if type(predict[0]) == torch.Tensor:
                predict[0] = predict[0].cpu().detach().numpy()
                predict[1] = predict[1].cpu().detach().numpy()
            pred_boxes, pred_scores = predict
            # tensorflow bboxes are in [ymin, xmin, ymax, xmax] format
            # convert to [xmin, ymin, xmax, ymax] to follow the xyxy format
            bboxes_xyxy = _output_swap_column(pred_boxes)

        if len(self.labels) == 1:  # single class
            max_scores = pred_scores
            labels_out = np.zeros_like(max_scores, dtype=np.int32)
        else:
            classes = pred_scores.argmax(axis=1)
            meshgrid = np.ogrid[: pred_scores.shape[0]]
            max_scores = pred_scores[meshgrid, classes]
            labels_out = np.full_like(max_scores, fill_value=classes, dtype=np.int32)

        # filter by confidence
        indices = max_scores > self.conf_threshold
        max_scores = max_scores[indices]
        bboxes_xyxy = bboxes_xyxy[indices]
        labels_out = labels_out[indices]

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
            labels=self.labels,
            label_filter=self.label_filter,
        )
        boxes, scores, classes = state.organize_bboxes(bboxes_xyxy, max_scores, labels_out)

        if self._where:
            # Adjust both x and y coordinates
            boxes[:, [0, 2]] += base_box[0]
            boxes[:, [1, 3]] += base_box[1]

        model_meta = ObjectDetectionMeta.create_immutable_meta(
            boxes=boxes,
            scores=scores,
            class_ids=classes,
            labels=self.labels,
        )
        meta.add_instance(self.task_name, model_meta, self._where)
        return image, predict, meta


def _organize_ssd_outputs(predict, num_classes):
    """
    Organize the SSD model outputs into two groups: confidences and locations.

    Args:
    predict: List of 12 unordered tensors (outputs from the SSD model).
    num_classes: The total number of classes (including background).

    Returns:
    confidences: Ordered list of tensors representing confidence scores.
    locations: Ordered list of tensors representing location coordinates.
    feature_map_info: List of parsed information (feature_map_height, feature_map_width, num_priors).
    """
    confidences = []
    locations = []

    # Separating and calculating num_priors for each tensor
    for tensor in predict:
        batch_size, channels, height, width = tensor.shape

        if channels % 4 == 0:  # num_anchors*4
            # Likely a location tensor
            num_priors = channels // 4
            locations.append((tensor, height, width, num_priors))
        elif channels % num_classes == 0:  # num_anchors*num_classes
            # Likely a confidence tensor
            num_priors = channels // num_classes
            confidences.append((tensor, height, width, num_priors))
        else:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")

    # Ensure that the number of location and confidence tensors are the same
    if len(locations) != len(confidences):
        raise ValueError("Mismatched number of location and confidence tensors")

    # Sorting based on feature map sizes (height, width)
    confidences.sort(key=lambda x: (x[1], x[2]), reverse=True)
    locations.sort(key=lambda x: (x[1], x[2]), reverse=True)

    # Extracting tensors in order and compiling feature map information
    ordered_confidences = [tensor for tensor, _, _, _ in confidences]
    ordered_locations = [tensor for tensor, h, w, p in locations]
    feature_map_info = [
        (h, w, p) for _, h, w, p in locations
    ]  # Assuming location and confidence maps match

    return ordered_confidences, ordered_locations, feature_map_info


def faster_rcnn_box_decode(locations, priors, center_variance, size_variance):
    """
    Decode predicted offsets to bounding box using Faster R-CNN box coder.

    Args:
    locations (ndarray): Predicted offsets for each prior box.
    priors (ndarray): Prior boxes in [cx, cy, w, h] format.
    center_variance, size_variance: Scaling factors.

    Returns:
    Decoded bounding boxes in [xmin, ymin, xmax, ymax] format.
    """
    boxes = np.concatenate(
        [
            locations[:, :2] * center_variance * priors[:, 2:] + priors[:, :2],
            np.exp(locations[:, 2:] * size_variance) * priors[:, 2:],
        ],
        axis=1,
    )
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def _generate_anchors(
    feature_map_info,
    num_layers=6,
    min_scale=0.2,
    max_scale=0.95,
    aspect_ratios=(1.0, 2.0, 1 / 2.0, 3.0, 1 / 3.0),
    reduce_boxes_in_lowest_layer=True,
    interpolated_scale_aspect_ratio=1.0,
    base_anchor_size=[1.0, 1.0],
):
    """Generate anchor boxes for all feature maps by following tf2 MultiscaleGridAnchorGenerator.
    Each feature map has a different scale and aspect ratio. For SSD-Mobilenet, size of anchor is
    1917 = 1083+600+150+54+24+6.
    """
    all_anchors = []
    num_layers = len(feature_map_info)
    base_anchor_height, base_anchor_width = base_anchor_size
    scales = np.linspace(min_scale, max_scale, num_layers)

    for idx, (grid_height, grid_width, _) in enumerate(feature_map_info):
        anchors = []
        scale = scales[idx]
        next_scale = scales[idx + 1] if idx + 1 < num_layers else 1.0

        for y in range(grid_height):
            for x in range(grid_width):
                cy, cx = (y + 0.5) / grid_height, (x + 0.5) / grid_width

                if idx == 0 and reduce_boxes_in_lowest_layer:
                    for ratio, special_scale in zip([1.0, 2.0, 0.5], [0.1, scale, scale]):
                        anchor_height = base_anchor_height * special_scale / np.sqrt(ratio)
                        anchor_width = base_anchor_width * special_scale * np.sqrt(ratio)
                        anchors.append([cx, cy, anchor_width, anchor_height])
                else:
                    for ratio in aspect_ratios:
                        anchor_height = base_anchor_height * scale / np.sqrt(ratio)
                        anchor_width = base_anchor_width * scale * np.sqrt(ratio)
                        anchors.append([cx, cy, anchor_width, anchor_height])

                # Interpolated scale anchor
                if idx > 0 or not reduce_boxes_in_lowest_layer:
                    if interpolated_scale_aspect_ratio > 0.0:
                        interpolated_scale = np.sqrt(scale * next_scale)
                        interpolated_anchor_height = base_anchor_height * interpolated_scale
                        interpolated_anchor_width = base_anchor_width * interpolated_scale
                        anchors.append(
                            [cx, cy, interpolated_anchor_width, interpolated_anchor_height]
                        )

        all_anchors.append(np.array(anchors))

    return all_anchors


def _multibox_layer(predict, num_classes):
    num_classes = num_classes + 1  # Account for the background class

    if not hasattr(_multibox_layer, '_cached_anchors'):
        _multibox_layer._cached_anchors = None

    # Following https://github.com/tensorflow/models/blob/master/research/object_detection/samples/configs/ssd_mobilenet_v2_coco.config
    # Pleae modify if using a different model config.
    min_scale = 0.2
    max_scale = 0.95
    num_layers = 6
    aspect_ratios = [1, 2, 0.5, 3, 0.3333]
    x_scale = 10.0
    y_scale = 10.0
    height_scale = 5.0
    width_scale = 5.0
    assert x_scale == y_scale, "x_scale and y_scale must be the same"
    assert height_scale == width_scale, "height_scale and width_scale must be the same"

    center_variance = 1 / x_scale
    size_variance = 1 / height_scale

    all_decoded_boxes = []
    all_conf_scores = []

    confidences, locations, feature_map_info = _organize_ssd_outputs(predict, num_classes)

    if _multibox_layer._cached_anchors is None:
        _multibox_layer._cached_anchors = _generate_anchors(
            feature_map_info, num_layers, min_scale, max_scale, aspect_ratios
        )
    anchors = _multibox_layer._cached_anchors

    for loc, conf, feature_map_anchors in zip(locations, confidences, anchors):
        # permute to (N, H, W, C) before reshaping
        loc_np = loc.permute(0, 2, 3, 1).cpu().detach().numpy().reshape(-1, 4)
        conf_np = conf.permute(0, 2, 3, 1).cpu().detach().numpy().reshape(-1, num_classes)

        assert (
            loc_np.shape[0] == feature_map_anchors.shape[0]
        ), f"Mismatch in number of anchors {feature_map_anchors.shape[0]} and location predictions {loc_np.shape[0]}"

        # reformat yxhw to xywh
        loc_np = _output_swap_column(loc_np)

        decoded_boxes = faster_rcnn_box_decode(
            loc_np, feature_map_anchors, center_variance, size_variance
        )
        all_decoded_boxes.append(decoded_boxes)

        # sigmoid to get the confidence scores
        conf_np = 1 / (1 + np.exp(-conf_np))
        all_conf_scores.append(conf_np[:, 1:])  # remove the background class

    pred_boxes = np.concatenate(all_decoded_boxes, axis=0)
    pred_scores = np.concatenate(all_conf_scores, axis=0)

    if pred_boxes.shape[0] != pred_scores.shape[0]:
        raise ValueError("Mismatch in number of predicted boxes and confidence scores.")

    return pred_boxes, pred_scores

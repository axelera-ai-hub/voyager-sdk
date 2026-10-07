# Copyright Axelera AI, 2026
# Operators that convert YOLACT-specific tensor output to
# generalized metadata representation

from __future__ import annotations

from itertools import product
from math import sqrt
from pathlib import Path

import numpy as np

from axelera import types
from axelera.app import gst_builder
from axelera.app.meta import InstanceSegmentationMeta
from axelera.app.model_utils.box import faster_rcnn_box_decode
from axelera.app.operators import AxOperator, PipelineContext, utils
from axelera.app.torch_utils import torch

# Box regression variances and the anchor layout of the upstream YOLACT ResNet
# configurations, which use square anchors scaled in input pixels.
_CENTER_VARIANCE = 0.1
_SIZE_VARIANCE = 0.2
_ASPECT_RATIOS = (1.0, 0.5, 2.0)
_ANCHOR_SCALES = (24, 48, 96, 192, 384)
_CROP_PADDING = 1
_NUM_MASK_COEFFICIENTS = 32


def _make_priors(width: int, height: int) -> np.ndarray:
    """Anchor priors as normalized [cx, cy, w, h], one row per anchor.

    Port of the upstream PredictionModule.make_priors. The first pyramid level
    is the stride-8 backbone output and each further level halves it rounding
    up; the per-cell iteration order (rows, then columns, then aspect ratio)
    has to match the flattened order of the prediction heads.
    """
    conv_width, conv_height = width, height
    for _ in range(3):
        conv_width, conv_height = -(-conv_width // 2), -(-conv_height // 2)
    longest_side = max(width, height)
    priors = []
    for scale in _ANCHOR_SCALES:
        for row, column in product(range(conv_height), range(conv_width)):
            x = (column + 0.5) / conv_width
            y = (row + 0.5) / conv_height
            for ratio in _ASPECT_RATIOS:
                side = scale * sqrt(ratio) / longest_side
                priors.append((x, y, side, side))
        conv_width, conv_height = -(-conv_width // 2), -(-conv_height // 2)
    return np.array(priors, dtype=np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    exponentials = np.exp(logits - logits.max(axis=-1, keepdims=True))
    return exponentials / exponentials.sum(axis=-1, keepdims=True)


def _to_pixels(low: np.ndarray, high: np.ndarray, size: int, padding: int = 0):
    """Upstream sanitize_coordinates: order the pair, scale to pixels, clamp."""
    low, high = low * size, high * size
    return (
        np.maximum(np.minimum(low, high) - padding, 0.0),
        np.minimum(np.maximum(low, high) + padding, float(size)),
    )


def _pairwise_iou(boxes: np.ndarray) -> np.ndarray:
    """IoU of every box pair within each row of an [n, m, 4] batch.

    The overlap is taken one axis at a time. Broadcasting both coordinate pairs
    together would allocate three [n, m, m, 2] temporaries, which for a per-class
    matrix over 200 candidates dominates the cost of the whole decode.
    """

    def overlap(low, high):
        top = np.maximum(low[:, :, None], low[:, None, :])
        bottom = np.minimum(high[:, :, None], high[:, None, :])
        return np.clip(bottom - top, 0.0, None)

    intersection = overlap(boxes[:, :, 0], boxes[:, :, 2]) * overlap(
        boxes[:, :, 1], boxes[:, :, 3]
    )
    areas = (boxes[:, :, 2] - boxes[:, :, 0]) * (boxes[:, :, 3] - boxes[:, :, 1])
    with np.errstate(invalid='ignore'):
        return intersection / (areas[:, :, None] + areas[:, None, :] - intersection)


def _fast_nms(boxes, coefficients, scores, iou_threshold, max_nms_boxes, top_k, score_threshold):
    """Upstream Detect.fast_nms: a per-class IoU matrix, then a global score cap.

    scores is [num_classes, num_candidates]. A candidate can survive under more
    than one class, so its class is read back from the row it survived in.

    Rows and depth are trimmed to the entries that clear score_threshold first.
    That is exact rather than an approximation: suppression only ever comes from a
    higher-scoring box of the same class, so an entry at or below the threshold
    cannot change the fate of one above it, and it would be discarded afterwards
    anyway. Without the trim the IoU matrix is num_classes x max_nms_boxes square
    whatever the image contains, which dominates the whole decode.
    """
    above = scores > score_threshold
    class_ids = np.flatnonzero(above.any(axis=1))
    depth = min(max_nms_boxes, int(above.sum(axis=1).max(initial=0)))
    if depth == 0:
        empty_coefficients = np.zeros((0, coefficients.shape[1]), coefficients.dtype)
        return np.zeros((0, 4)), empty_coefficients, np.zeros(0, int), np.zeros(0)
    scores = scores[class_ids]
    order = np.argsort(-scores, axis=1, kind='stable')[:, :depth]
    scores = np.take_along_axis(scores, order, axis=1)
    num_classes, num_candidates = order.shape
    boxes = boxes[order.reshape(-1)].reshape(num_classes, num_candidates, 4)
    coefficients = coefficients[order.reshape(-1)].reshape(num_classes, num_candidates, -1)
    # A box survives if no higher scoring box of the same class overlaps it.
    highest_iou = np.triu(_pairwise_iou(boxes), k=1).max(axis=1)
    keep = highest_iou <= iou_threshold
    classes = np.broadcast_to(class_ids[:, None], keep.shape)[keep]
    boxes, coefficients, scores = boxes[keep], coefficients[keep], scores[keep]
    order = np.argsort(-scores, kind='stable')[:top_k]
    return boxes[order], coefficients[order], classes[order], scores[order]


class DecodeYolact(AxOperator):
    """
    Decoding YOLACT into Axelera metadata

    Input:
        image: the source image
        predict: the four prediction heads (box regression, class logits, mask
            coefficients and mask prototypes), matched by shape
    Output:
        image, predict, meta
    """

    conf_threshold: float = 0.05
    nms_iou_threshold: float = 0.5
    max_nms_boxes: int = 200
    nms_top_k: int = 100

    def _post_init(self):
        self._priors = None
        self._tmp_labels: Path | None = None
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
        compiled_model_dir: Path | None,
        task_graph,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, compiled_model_dir, task_graph
        )
        if self._where:
            raise NotImplementedError(
                "DecodeYolact decodes whole frames and cannot run on a cropped region"
            )
        if context.resize_status != types.ResizeMode.STRETCH:
            raise ValueError(
                f"DecodeYolact needs a stretching resize, got {context.resize_status.name}; "
                "YOLACT is trained on stretched input and decodes its masks in that space"
            )
        self.meta_type_name = "InstanceSegmentationMeta"
        self.model_width = model_info.input_width
        self.model_height = model_info.input_height
        self.labels = model_info.labels
        self.num_classes = model_info.num_classes
        self._priors = _make_priors(self.model_width, self.model_height)
        self._association = context.association or None
        if model_info.manifest and model_info.manifest.is_compiled():
            self._deq_scales, self._deq_zeropoints = zip(*model_info.manifest.dequantize_params)
            self._n_padded_ch_outputs = model_info.manifest.n_padded_ch_outputs

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        """Configure libdecode_yolact, which also owns NMS.

        The element runs YOLACT's fast NMS itself rather than deferring to
        libinplace_nms.so, because fast NMS judges each candidate against the
        higher scoring boxes of its own class in one pass instead of sweeping
        transitively, and the two do not give the same detections. That leaves
        no stage in which per-tile results could be merged, so tiling is
        refused rather than silently dropping tiles.
        """
        if gst.tiling and gst.add_tiles:
            raise NotImplementedError(
                "DecodeYolact runs NMS inside the decoder and has no stage that could merge "
                "tiles; run this model without tiling"
            )
        if self._tmp_labels is None:
            self._tmp_labels = utils.create_tmp_labels(self.labels)
        scales = ','.join(str(s) for s in self._deq_scales)
        zeros = ','.join(str(z) for z in self._deq_zeropoints)
        paddings = '|'.join(
            ','.join(str(num) for num in sublist) for sublist in self._n_padded_ch_outputs
        )
        master_key, association_key = gst_builder.TileInfo(self, gst).get_decode_keys()

        gst.decode_muxer(
            name=f'decoder_task{self._taskn}{stream_idx}',
            lib='libdecode_yolact.so',
            mode='read',
            options=f'meta_key:{str(self.task_name)};'
            f'decoder_name:{self.meta_type_name};'
            f'{master_key}'
            f'{association_key}'
            f'classes:{self.num_classes};'
            f'num_seg_masks:{_NUM_MASK_COEFFICIENTS};'
            f'confidence_threshold:{self.conf_threshold};'
            f'nms_iou_threshold:{self.nms_iou_threshold};'
            f'max_nms_boxes:{self.max_nms_boxes};'
            f'topk:{self.nms_top_k};'
            f'scales:{scales};'
            f'zero_points:{zeros};'
            f'padding:{paddings};'
            f'anchor_scales:{",".join(str(s) for s in _ANCHOR_SCALES)};'
            f'aspect_ratios:{",".join(str(r) for r in _ASPECT_RATIOS)};'
            f'center_variance:{_CENTER_VARIANCE};'
            f'size_variance:{_SIZE_VARIANCE};'
            f'crop_padding:{_CROP_PADDING};'
            f'classlabels_file:{self._tmp_labels};'
            f'model_width:{self.model_width};'
            f'model_height:{self.model_height};',
        )

    def exec_torch(self, image, predict, meta):
        loc, conf, coefficients, prototypes = self._match_outputs(predict)
        boxes, coefficients, classes, scores = self._detect(loc, conf, coefficients)
        source_width, source_height = image.size
        left, right = _to_pixels(boxes[:, 0], boxes[:, 2], source_width)
        top, bottom = _to_pixels(boxes[:, 1], boxes[:, 3], source_height)
        pixel_boxes = np.stack([left, top, right, bottom], axis=1)
        model_meta = InstanceSegmentationMeta(
            seg_shape=prototypes.shape[1::-1], labels=self.labels
        )
        model_meta.add_results(
            self._crop_masks(boxes, coefficients, prototypes, pixel_boxes),
            pixel_boxes,
            classes,
            scores,
        )
        meta.add_instance(self.task_name, model_meta, self._where)
        return image, predict, meta

    def _match_outputs(self, predict):
        """Pick the four heads out of predict by shape, as their order is not fixed."""
        if len(predict) != 4:
            raise ValueError(f"Expected 4 YOLACT output tensors, got {len(predict)}")
        tensors = [
            p.cpu().detach().numpy() if isinstance(p, torch.Tensor) else np.asarray(p)
            for p in predict
        ]
        if any(t.shape[0] != 1 for t in tensors):
            raise ValueError("Batch size >1 is not supported for the torch pipelines")
        matched = {}
        for tensor in (t[0] for t in tensors):
            if tensor.ndim == 3:
                matched['prototypes'] = tensor
            elif tensor.shape[1] == 4:
                matched['loc'] = tensor
            elif tensor.shape[1] == self.num_classes + 1:
                matched['conf'] = tensor
            else:
                matched['coefficients'] = tensor
        missing = {'loc', 'conf', 'coefficients', 'prototypes'} - matched.keys()
        if missing:
            raise ValueError(f"Could not match YOLACT outputs {sorted(missing)} by shape")
        return matched['loc'], matched['conf'], matched['coefficients'], matched['prototypes']

    def _detect(self, loc, conf, coefficients):
        """Decode the boxes, drop the background class, then run per-class fast NMS."""
        if len(self._priors) != len(loc):
            raise ValueError(
                f"The model emits {len(loc)} anchors but {self.model_width}x"
                f"{self.model_height} priors give {len(self._priors)}"
            )
        scores = _softmax(conf)[:, 1:].T
        candidates = scores.max(axis=0) > self.conf_threshold
        scores = scores[:, candidates]
        if scores.shape[1] == 0:
            empty_coefficients = np.zeros((0, coefficients.shape[1]), coefficients.dtype)
            return np.zeros((0, 4)), empty_coefficients, np.zeros(0, int), np.zeros(0)
        boxes, coefficients, classes, scores = _fast_nms(
            faster_rcnn_box_decode(
                loc[candidates], self._priors[candidates], _CENTER_VARIANCE, _SIZE_VARIANCE
            ),
            coefficients[candidates],
            scores,
            self.nms_iou_threshold,
            self.max_nms_boxes,
            self.nms_top_k,
            self.conf_threshold,
        )
        # A candidate is kept for every class it survived, so the per-class
        # score still has to clear the threshold.
        above = scores > self.conf_threshold
        return boxes[above], coefficients[above], classes[above], scores[above]

    def _crop_masks(self, boxes, coefficients, prototypes, pixel_boxes):
        """Mask entries of prototype-space rect, image-space rect and cropped mask.

        The linear combination is thresholded rather than activated: upstream
        binarizes sigmoid(x) at 0.5, which is exactly x > 0. Cropping to the box
        is expressed as the integer rect that the upstream float crop bounds
        cover, so the stored mask holds only what falls inside the box.

        The image-space rect is the detection's own pixel box, because the crop is
        that box mapped into prototype space and the renderer scales the mask back
        onto it.
        """
        proto_height, proto_width = prototypes.shape[:2]
        logits = prototypes @ coefficients.T
        lefts, rights = _to_pixels(boxes[:, 0], boxes[:, 2], proto_width, _CROP_PADDING)
        tops, bottoms = _to_pixels(boxes[:, 1], boxes[:, 3], proto_height, _CROP_PADDING)
        entries = []
        for index in range(len(boxes)):
            x0, x1 = int(np.ceil(lefts[index])), int(np.ceil(rights[index]))
            y0, y1 = int(np.ceil(tops[index])), int(np.ceil(bottoms[index]))
            if x1 <= x0 or y1 <= y0:
                # The box misses the prototype grid, so the crop leaves nothing.
                x0, y0, x1, y1 = 0, 0, 1, 1
                mask = np.zeros((1, 1), dtype=np.uint8)
            else:
                mask = (logits[y0:y1, x0:x1, index] > 0).astype(np.uint8) * 255
            entries.append((x0, y0, x1, y1, *pixel_boxes[index].astype(int).tolist(), mask))
        return entries

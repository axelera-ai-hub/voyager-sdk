# Copyright Axelera AI, 2024
# Operators that convert YOLO-SEG-specific tensor output to
# generalized metadata representation

from pathlib import Path
import re
from typing import List, Optional

import numpy as np

from axelera import types
from axelera.app import gst_builder, logging_utils
from axelera.app.meta import BBoxState, InstanceSegmentationMeta
from axelera.app.meta.segmentation import _translate_image_space_rect
from axelera.app.operators import AxOperator, PipelineContext, utils
from axelera.app.torch_utils import torch

LOG = logging_utils.getLogger(__name__)


class DecodeYoloSeg(AxOperator):
    """
    Decoding YOLO-SEG into Axelera metadata

    Input:
        predict: batched predictions
        kwargs: model info
    Output:
        image, predict, meta
    """

    box_format: str
    normalized_coord: bool
    label_filter: Optional[List[str]] = None
    label_exclude: Optional[List[str]] = None
    conf_threshold: float = 0.25
    max_nms_boxes: int = 30000
    use_multi_label: bool = False
    nms_class_agnostic: bool = True
    cpp_opt_heatmap: bool = False
    nms_iou_threshold: float = 0.7
    nms_top_k: int = 30
    unpad: bool = True

    def _post_init(self):
        if isinstance(self.label_filter, str) and not self.label_filter.startswith('$$'):
            stripped = (self.label_filter or '').strip()
            self.label_filter = [x for x in re.split(r'\s*[,;]\s*', stripped) if x]
        else:
            self.label_filter = []
        if isinstance(self.label_exclude, str) and not self.label_exclude.startswith('$$'):
            stripped = (self.label_exclude or '').strip()
            self.label_exclude = [x for x in re.split(r'\s*[,;]\s*', stripped) if x]
        else:
            self.label_exclude = []
        self._tmp_labels: Optional[Path] = None
        if self.box_format not in ["xyxy", "xywh", "ltwh"]:
            raise ValueError(f"Unknown box format {self.box_format}")
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
        if self.label_filter and self.label_exclude:
            self.label_filter = [x for x in self.labels if x not in self.label_exclude]
        elif not self.label_filter and self.label_exclude:
            self.label_filter = [x for x in model_info.labels if x not in self.label_exclude]
        self.meta_type_name = "InstanceSegmentationMeta"
        if model_info.manifest and model_info.manifest.is_compiled():
            self._deq_scales, self._deq_zeropoints = zip(*model_info.manifest.dequantize_params)
            self._postprocess_graph = model_info.manifest.postprocess_graph
            self._n_padded_ch_outputs = model_info.manifest.n_padded_ch_outputs

        self.scaled = context.resize_status
        self.model_width = model_info.input_width
        self.model_height = model_info.input_height
        self.labels = model_info.labels
        self.num_classes = model_info.num_classes

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        # Supports only yolov8seg for now
        if self._n_padded_ch_outputs:
            paddings = '|'.join(
                ','.join(str(num) for num in sublist) for sublist in self._n_padded_ch_outputs
            )
        if self._tmp_labels is None:
            self._tmp_labels = utils.create_tmp_labels(self.labels)
        if not gst.new_inference:
            conns = {'src': f'decoder_task{self._taskn}{stream_idx}.sink_0'}
            gst.queue(name=f'queue_decoder_task{self._taskn}{stream_idx}', connections=conns)
        scales = ','.join(str(s) for s in self._deq_scales)
        zeros = ','.join(str(s) for s in self._deq_zeropoints)
        sieve = utils.build_class_sieve(self.label_filter, self._tmp_labels)
        master_key = ''
        if self._where:
            master_key = f'master_meta:{self._where};'
        gst.decode_muxer(
            name=f'decoder_task{self._taskn}{stream_idx}',
            lib='libdecode_yolov8seg.so',
            mode='read',
            options=f'meta_key:{str(self.task_name)};'
            f'decoder_name:{self.meta_type_name};'
            f'{master_key}'
            f'classes:{self.num_classes};'
            f'confidence_threshold:{self.conf_threshold};'
            f'scales:{scales};'
            f'zero_points:{zeros};'
            f'padding:{paddings};'
            f'multiclass:{int(self.use_multi_label)};'
            f'classlabels_file:{self._tmp_labels};'
            f'model_width:{self.model_width};'
            f'model_height:{self.model_height};'
            + (f';label_filter:{",".join(sieve)}' if sieve else ''),
        )
        gst.axinplace(
            lib='libinplace_nms.so',
            options=f'meta_key:{str(self.task_name)};'
            f'{master_key}'
            f'max_boxes:{self.nms_top_k};'
            f'nms_threshold:{self.nms_iou_threshold};'
            f'class_agnostic:0;'
            f'location:CPU;',
        )

    def exec_torch(self, image, predict, meta):
        if not isinstance(predict, list) or len(predict) != 2:
            raise ValueError("Expected a list of two tensors, got {}".format(predict))

        # Determine which tensor is for bounding boxes and which is for prototypes
        pred_bboxes = next(p for p in predict if p.dim() == 3)
        protos = next(p for p in predict if p.dim() == 4)

        if isinstance(pred_bboxes, torch.Tensor):
            pred_bboxes = pred_bboxes.cpu().detach().numpy()
            protos = protos.cpu().detach().numpy()

        if pred_bboxes.shape[0] > 1:
            raise ValueError(
                f"Batch size >1 not supported for torch and torch-aipu pipelines, output tensor={predict[0].shape}"
            )

        if pred_bboxes.shape[1] < pred_bboxes.shape[2]:
            pred_bboxes = pred_bboxes.transpose([0, 2, 1])

        for pred, proto in zip(pred_bboxes, protos):
            boxes, scores, classes, mask_coef = self._process_predictions(pred)
            if self._where:
                master_meta = meta[self._where]
                # get boxes of the last secondary frame index
                base_box = master_meta.boxes[
                    master_meta.get_next_secondary_frame_index(self.task_name)
                ]
                src_img_width = base_box[2] - base_box[0]
                src_img_height = base_box[3] - base_box[1]
                bbox = np.array(
                    [
                        max(0, int(base_box[0])),
                        max(0, int(base_box[1])),
                        min(image.size[0], int(base_box[2])),
                        min(image.size[1], int(base_box[3])),
                    ],
                    dtype=int,
                )
            else:
                src_img_width = image.size[0]
                src_img_height = image.size[1]
                bbox = np.array([0, 0, src_img_width, src_img_height], dtype=int)

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
                nms_class_agnostic=self.nms_class_agnostic,
                output_top_k=self.nms_top_k,
                labels=self.labels,
                label_filter=self.label_filter,
            )

            (boxes, scores, classes, masks,) = state.organize_bboxes_and_instance_seg(
                boxes, scores, classes, mask_coef, proto, False
            )

            if self._where:
                # Adjust both x and y coordinates
                boxes[:, [0, 2]] += base_box[0]
                boxes[:, [1, 3]] += base_box[1]

            model_meta = InstanceSegmentationMeta(
                labels=self.labels,
            )
            aranged_masks = []
            if len(masks) != 0:
                for i, sbox in enumerate(masks[0]):
                    mbox = _translate_image_space_rect(sbox, bbox)
                    aranged_masks.append((*sbox, *mbox, masks[1][i]))

            model_meta.add_results(aranged_masks, boxes, classes, scores)
            meta.add_instance(self.task_name, model_meta, self._where)
        return image, predict, meta

    def _process_predictions(self, pred):
        box_coordinates, box_labels, mask_coef = np.split(pred, [4, 4 + self.num_classes], axis=1)

        if self.use_multi_label:
            boxes, scores, classes, mask_coef = self._process_multi_label(
                box_coordinates, box_labels, mask_coef
            )
        else:
            boxes, scores, classes, mask_coef = self._process_single_label(
                box_coordinates, box_labels, mask_coef
            )

        return boxes, scores, classes, mask_coef

    def _process_multi_label(self, box_coordinates, box_labels, mask_coef):
        i, j = np.where(box_labels > self.conf_threshold)
        boxes = box_coordinates[i]
        scores = box_labels[i, j]
        classes = j.astype(np.float32)
        mask_coef = mask_coef[i]
        return boxes, scores, classes, mask_coef

    def _process_single_label(self, box_coordinates, box_labels, mask_coef):
        classes = np.argmax(box_labels, axis=1)
        scores = np.max(box_labels, axis=1)
        mask = scores > self.conf_threshold
        boxes = box_coordinates[mask]
        scores = scores[mask]
        classes = classes[mask]
        mask_coef = mask_coef[mask]
        return boxes, scores, classes, mask_coef

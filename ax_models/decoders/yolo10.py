# Copyright Axelera AI, 2025
# Operators that convert YOLO10-specific tensor output to
# generalized metadata representation

import enum
import itertools
from pathlib import Path
from typing import List, Optional

import numpy as np

from axelera import types
from axelera.app import compile, gst_builder, logging_utils
from axelera.app.meta import BBoxState, ObjectDetectionMeta
from axelera.app.operators import AxOperator, PipelineContext, utils
from axelera.app.torch_utils import torch

LOG = logging_utils.getLogger(__name__)


class DecodeYolo10(AxOperator):
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
    label_filter: Optional[List[str] | str] = None
    label_exclude: Optional[List[str] | str] = None
    conf_threshold: float = 0.25
    use_multi_label: bool = False
    top_k: int = 300
    dfl_size: int = 16

    def _post_init(self):
        self.label_filter = utils.parse_labels_filter(self.label_filter)
        self.label_exclude = utils.parse_labels_filter(self.label_exclude)

        self._tmp_labels: Optional[Path] = None
        if self.box_format not in ["xyxy", "xywh", "ltwh"]:
            raise ValueError(f"Unknown box format {self.box_format}")
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
        if self.label_filter and self.label_exclude:
            self.label_filter = utils.label_exclude_to_label_filter(
                self.label_filter, self.label_exclude
            )
        elif not self.label_filter and self.label_exclude:
            self.label_filter = utils.label_exclude_to_label_filter(
                model_info.labels, self.label_exclude
            )
        if model_info.manifest and model_info.manifest.is_compiled():
            self._deq_scales, self._deq_zeropoints = zip(*model_info.manifest.dequantize_params)
            self._postprocess_graph = compiled_model_dir / model_info.manifest.postprocess_graph
            self._n_padded_ch_outputs = model_info.manifest.n_padded_ch_outputs

            output_shapes = compile.get_original_shape(
                model_info.manifest.output_shapes,
                model_info.manifest.n_padded_ch_outputs,
                'NHWC',
                'NHWC',
            )
        self.scaled = context.resize_status
        self.model_width = model_info.input_width
        self.model_height = model_info.input_height
        self.labels = model_info.labels
        self.num_classes = model_info.num_classes
        self.use_multi_label &= self.num_classes > 1
        self._association = context.association or None

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        if self._tmp_labels is None:
            self._tmp_labels = utils.create_tmp_labels(self.labels)

        sieve = utils.build_class_sieve(self.label_filter, self.labels)
        tiling = gst_builder.TileInfo(self, gst)
        master_key, association_key = tiling.get_decode_keys()
        if self._n_padded_ch_outputs:
            paddings = '|'.join(
                ','.join(str(num) for num in sublist) for sublist in self._n_padded_ch_outputs
            )
        else:
            raise ValueError(f"Missing n_padded_ch_outputs for {self.model_name}")

        if False:  # this is meta assignment only
            gst.decode_muxer(
                name=f'decoder_task{self._taskn}{stream_idx}',
                lib='libdecode_yolov10_simple.so',
                mode='read',
                options=f'meta_key:{str(self.task_name)};'
                f'{master_key}'
                f'{association_key}'
                f'classes:{self.num_classes};'
                f'confidence_threshold:{self.conf_threshold};'
                f'topk:{self.top_k};'
                f'multiclass:{int(self.use_multi_label)};'
                f'classlabels_file:{self._tmp_labels};'
                f'model_width:{self.model_width};'
                f'model_height:{self.model_height};'
                f'scale_up:{int(self.scaled==types.ResizeMode.LETTERBOX_FIT)};'
                f'letterbox:{int(self.scaled in [types.ResizeMode.LETTERBOX_FIT, types.ResizeMode.LETTERBOX_CONTAIN])}'
                + (f';label_filter:{",".join(sieve)}' if sieve else ''),
            )
        else:
            scales = ','.join(str(s) for s in self._deq_scales)
            zeros = ','.join(str(s) for s in self._deq_zeropoints)
            gst.decode_muxer(
                name=f'decoder_task{self._taskn}{stream_idx}',
                lib='libdecode_yolov8.so',
                mode='read',
                options=f'meta_key:{str(self.task_name)};'
                f'{master_key}'
                f'{association_key}'
                f'classes:{self.num_classes};'
                f'confidence_threshold:{self.conf_threshold};'
                f'scales:{scales};'
                f'padding:{paddings};'
                f'zero_points:{zeros};'
                f'topk:{self.top_k};'
                f'multiclass:{int(self.use_multi_label)};'
                f'dfl_size:{self.dfl_size};'
                f'classlabels_file:{self._tmp_labels};'
                f'model_width:{self.model_width};'
                f'model_height:{self.model_height};'
                f'scale_up:{int(self.scaled==types.ResizeMode.LETTERBOX_FIT)};'
                f'letterbox:{int(self.scaled in [types.ResizeMode.LETTERBOX_FIT, types.ResizeMode.LETTERBOX_CONTAIN])}'
                + (f';label_filter:{",".join(sieve)}' if sieve else ''),
            )
        if gst.tiling:
            master_key = tiling.get_nms_keys()
            gst.axinplace(
                lib='libinplace_nms.so',
                options=f'meta_key:{str(self.task_name)};'
                f'{master_key}'
                f'location:CPU;'
                f'max_boxes:{tiling.get_max_boxes(self.nms_top_k)}',
            )

    def exec_torch(self, image, predict, meta):
        if type(predict) == torch.Tensor:
            predict = predict.cpu().detach().numpy()

        if len(predict) == 1 and predict.shape[0] > 1:
            raise ValueError(
                f"Batch size >1 not supported for torch and torch-aipu pipelines, output tensor={predict[0].shape}"
            )
        elif len(predict) > 1:  # Handling multiple predictions, possibly yolo-nas
            raise ValueError(
                f"Unexpected output shapes, {predict[0].shape} and {predict[1].shape}"
            )

        bboxes = predict[0]
        # Ensure bboxes are transposed to format (number of samples, info per sample) if needed
        if bboxes.shape[0] < bboxes.shape[1]:
            bboxes = bboxes.transpose()

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

        mask = bboxes[:, 4] >= self.conf_threshold
        boxes = bboxes[mask, :4]
        scores = bboxes[mask, 4]
        classes = bboxes[mask, 5].astype(int)
        if self._where:
            boxes[:, [0, 2]] += base_box[0]
            boxes[:, [1, 3]] += base_box[1]

        state = BBoxState(
            self.model_width,
            self.model_height,
            src_img_width,
            src_img_height,
            self.box_format,
            self.normalized_coord,
            self.scaled,
            nms_iou_threshold=0.0,  # Disable NMS
            labels=self.labels,
            label_filter=self.label_filter,
        )
        boxes, scores, classes = state.organize_bboxes(boxes, scores, classes)

        model_meta = ObjectDetectionMeta.create_immutable_meta(
            boxes=boxes,
            scores=scores,
            class_ids=classes,
            labels=self.labels,
        )

        meta.add_instance(self.task_name, model_meta, self._where)
        return image, predict, meta

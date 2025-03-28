# Copyright Axelera AI, 2024
# Operators that convert model-specific tensor output to
# generalized metadata representation
from __future__ import annotations

from pathlib import Path

import numpy as np

from axelera import types
from axelera.app import gst_builder
from axelera.app.meta import BBoxState, ObjectDetectionMeta
from axelera.app.operators import AxOperator, PipelineContext, utils
from axelera.app.torch_utils import torch


class DecodeRTMDet(AxOperator):
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
    conf_threshold: float = 0.25
    max_nms_boxes: int = 30000
    nms_iou_threshold: float = 0.45
    nms_class_agnostic: bool = False
    nms_top_k: int = 300
    # num_classes: int = 80

    def _post_init(self):
        super()._post_init()
        self.gst_decoder_does_dequantization_and_depadding = True

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
        self.scaled = context.resize_status
        self.model_width = model_info.input_width
        self.model_height = model_info.input_height

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        conns = {'src': f'decoder_task{self._taskn}{stream_idx}.sink_0'}
        gst.queue(name=f'queue_decoder_task{self._taskn}{stream_idx}', connections=conns)

        scales = ','.join(str(s) for s in self._deq_scales)
        zeros = ','.join(str(s) for s in self._deq_zeropoints)
        gst.decode_muxer(
            name=f'decoder_task{self._taskn}{stream_idx}',
            lib='libdecode_rtmdet.so',
            mode='read',
            options=f'meta_key:{str(self.task_name)};'
            f'confidence_threshold:{self.conf_threshold};'
            f'scales:{scales};'
            f'zero_points:{zeros};',
        )
        gst.axinplace(
            lib='libinplace_nms.so',
            options=f'meta_key:{str(self.task_name)};'
            f'max_boxes:{self.nms_top_k};'
            f'nms_threshold:{self.nms_iou_threshold};'
            f'class_agnostic:0;'
            f'location:CPU',
        )

    def _adjust_letterbox(self, boxes, sizes):
        for idx, _ in enumerate(boxes):
            if sizes[0] > sizes[1]:  # Landscape
                boxes[idx][1] = (0.5 + boxes[idx][1]) * boxes[idx][1]
                boxes[idx][3] = (0.5 + boxes[idx][3]) * boxes[idx][3]
            elif sizes[0] < sizes[1]:
                boxes[idx][0] = (0.5 + boxes[idx][0]) * boxes[idx][0]
                boxes[idx][2] = (0.5 + boxes[idx][2]) * boxes[idx][2]
        return boxes

    def _aipu_post_process(self, image, predict, meta):
        tensors = list()
        if isinstance(predict, torch.Tensor):
            tensors = predict.cpu().detach().numpy()
        elif isinstance(predict, list):
            for tensor in predict:
                reshaped_tensor = np.transpose(tensor.cpu().detach().numpy(), (0, 2, 3, 1))
                tensors.append(np.squeeze(reshaped_tensor, axis=0))
        else:
            tensors = predict
        boxes = list()
        scores = list()
        classes = list()
        for i in range(3):
            scale = self.model_width / tensors[i].shape[0]
            for xy in np.ndindex(tensors[i].shape[:2]):
                conf = 1.0 / (1.0 + np.exp(-tensors[i + 3][xy][0]))
                if conf > self.conf_threshold:
                    gridcell = scale * tensors[i][xy][:]
                    x1 = (scale * xy[1] - gridcell[0]) / self.model_width
                    y1 = (scale * xy[0] - gridcell[1]) / self.model_height
                    x2 = (scale * xy[1] + gridcell[2]) / self.model_width
                    y2 = (scale * xy[0] + gridcell[3]) / self.model_height
                    box = [x1, y1, x2, y2]
                    boxes.append(box)
                    scores.append(conf)
                    classes.append(0)
        if len(scores) == 0:
            return image, predict, meta

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
        boxes, scores, classes = state.organize_bboxes(boxes, scores, classes)

        if self._where:
            # Adjust both x and y coordinates
            boxes[:, [0, 2]] += base_box[0]
            boxes[:, [1, 3]] += base_box[1]

        return ObjectDetectionMeta.create_immutable_meta(
            boxes=boxes,
            scores=scores,
            class_ids=classes,
            labels=['hand'],
        )

    def _generic_post_process(self, image, predict, meta):
        bounding_boxes = predict[0].cpu().detach().numpy().squeeze(0)[:, :4]
        confidance_scores = predict[0].cpu().detach().numpy().squeeze(0)[:, -1]

        indecies = np.argwhere(confidance_scores > self.conf_threshold)
        if len(indecies) == 0:
            return image, predict, meta

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

        boxes = bounding_boxes[indecies].squeeze(1)
        scores = confidance_scores[indecies].squeeze(1)
        classes = np.zeros(len(scores), dtype=int)
        scaled_boxes = list()
        for box in boxes:
            x1 = (src_img_width * box[0]) / self.model_width
            y1 = (src_img_height * box[1]) / self.model_height
            x2 = (src_img_width * box[2]) / self.model_width
            y2 = (src_img_height * box[3]) / self.model_height
            scaled_boxes.append([x1, y1, x2, y2])

        if self._where:
            scaled_boxes[:, [0, 2]] += base_box[0]
            scaled_boxes[:, [1, 3]] += base_box[1]

        return ObjectDetectionMeta.create_immutable_meta(
            boxes=scaled_boxes,
            scores=scores,
            class_ids=classes,
            labels=['hand'],
        )

    def exec_torch(self, image, predict, meta):
        if len(predict) == 6:
            model_meta = self._aipu_post_process(image, predict, meta)
        elif len(predict) == 2:
            model_meta = self._generic_post_process(image, predict, meta)
        else:
            raise ValueError(f"Invalid count of predict tensors: {len(predict)}")

        meta.add_instance(self.task_name, model_meta, self._where)
        return image, predict, meta

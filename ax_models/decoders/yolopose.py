# Copyright Axelera AI, 2024
# Operators that convert YOLO-POSE-specific tensor output to
# generalized metadata representation
from __future__ import annotations

from pathlib import Path
from typing import List, Optional

from axelera import types
from axelera.app import gst_builder, logging_utils
from axelera.app.meta import BBoxState, CocoBodyKeypointsMeta
from axelera.app.operators import AxOperator, PipelineContext
from axelera.app.torch_utils import torch

LOG = logging_utils.getLogger(__name__)


class DecodeYoloPose(AxOperator):
    """
    Decoding bounding boxes and add model info into Axelera metadata

    Supports both YOLO8-POSE and YOLO26-POSE models.

    Input:
        predict: batched predictions
        kwargs: model info
    Output:
        image, predict, meta
    """

    box_format: str
    normalized_coord: bool
    conf_threshold: float = 0.25
    max_nms_boxes: int = 30000
    nms_iou_threshold: float = 0.65
    nms_top_k: int = 300
    nms_free: bool = False
    dfl_size: int = 16

    def _post_init(self):
        self._tmp_labels: Optional[Path] = None
        if self.box_format not in ["xyxy", "xywh", "ltwh"]:
            raise ValueError(f"Unknown box format {self.box_format}")
        self._nms_class_agnostic = True

        # Set keypoint decoding parameters based on model type
        if self.nms_free:
            # YOLO26: (x + offset + 1.0*raw) instead of (x + 2.0*raw)
            self._kpt_multiplier = 1.0
            self._kpt_offset = 0.5
            self._kpts_start_idx = 6  # Skip class_id at index 5
        else:
            # YOLO8
            self._kpt_multiplier = 2.0
            self._kpt_offset = 0.0
            self._kpts_start_idx = 5

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
        self.meta_type_name = "CocoBodyKeypointsMeta"
        if model_info.manifest and model_info.manifest.is_compiled():
            self._deq_scales, self._deq_zeropoints = zip(*model_info.manifest.dequantize_params)
            self._postprocess_graph = model_info.manifest.postprocess_graph
            self._n_padded_ch_outputs = model_info.manifest.n_padded_ch_outputs

            self._kpts_shape = CocoBodyKeypointsMeta.keypoints_shape

        self.scaled = context.resize_status
        self.model_width = model_info.input_width
        self.model_height = model_info.input_height
        self._association = context.association or None

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        if self._n_padded_ch_outputs:
            paddings = '|'.join(
                ','.join(str(num) for num in sublist) for sublist in self._n_padded_ch_outputs
            )
        else:
            raise ValueError(f"Missing n_padded_ch_outputs for {self.model_name}")

        scales = ','.join(str(s) for s in self._deq_scales)
        zeros = ','.join(str(s) for s in self._deq_zeropoints)
        kpt_shape = ','.join(str(s) for s in self._kpts_shape)

        tiling = gst_builder.TileInfo(self, gst)
        master_key, association_key = tiling.get_decode_keys()

        kpt_options = (
            f'kpt_multiplier:{self._kpt_multiplier};kpt_offset:{self._kpt_offset};'
            if self.nms_free
            else ''
        )

        gst.decode_muxer(
            name=f'decoder_task{self._taskn}{stream_idx}',
            lib='libdecode_yolov8.so',
            mode='read',
            options=f'meta_key:{str(self.task_name)};'
            f'{master_key}'
            f'{association_key}'
            f'confidence_threshold:{self.conf_threshold};'
            f'scales:{scales};'
            f'padding:{paddings};'
            f'zero_points:{zeros};'
            f'classes:1;'
            f'kpts_shape:{kpt_shape};'
            f'{kpt_options}'
            f'dfl_size:{self.dfl_size};'
            f'model_width:{self.model_width};'
            f'model_height:{self.model_height};'
            f'scale_up:{int(self.scaled==types.ResizeMode.LETTERBOX_FIT)};'
            f'decoder_name:{self.meta_type_name};',
        )

        # NMS: skip for nms_free models unless tiling is enabled
        if not self.nms_free or gst.tiling:
            master_key = tiling.get_nms_keys()
            gst.axinplace(
                lib='libinplace_nms.so',
                options=f'meta_key:{str(self.task_name)};'
                f'{master_key}'
                f'max_boxes:{tiling.get_max_boxes(self.nms_top_k)};'
                f'nms_threshold:{self.nms_iou_threshold};'
                f'class_agnostic:{int(self._nms_class_agnostic)};'
                f'location:CPU;',
            )

    def exec_torch(self, image, predict, meta):
        if type(predict) == torch.Tensor:  # torch.Size([1, 56, 8400])
            predict = predict.cpu().detach().numpy()

        if len(predict) == 1 and predict.shape[0] > 1:
            raise ValueError(
                f"Batch size >1 not supported for torch and torch-aipu pipelines, output tensor={predict[0].shape}"
            )

        bboxes = predict[0]

        # YOLO26 output format: [batch, num_detections, 6+keypoints]
        # where 6 = [x1, y1, x2, y2, confidence, class_id]
        if bboxes.shape[0] < bboxes.shape[1]:
            bboxes = bboxes.transpose()

        # Extract keypoints starting from the correct index
        kpts = bboxes[:, self._kpts_start_idx :]
        box_confidence = bboxes[:, 4]
        box_coordinates = bboxes[:, :4]

        # filter out low confidence boxes
        indices = box_confidence > self.conf_threshold
        box_confidence = box_confidence[indices]
        box_coordinates = box_coordinates[indices]
        kpts = kpts[indices]

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
            nms_iou_threshold=(
                0.0 if self.nms_free else self.nms_iou_threshold
            ),  # Disable NMS for YOLO26
            nms_class_agnostic=self._nms_class_agnostic,  # for keypoint
            output_top_k=self.nms_top_k,
        )

        if self._where:
            # Adjust both x and y coordinates
            box_coordinates[:, [0, 2]] += base_box[0]
            box_coordinates[:, [1, 3]] += base_box[1]

        boxes, scores, kpts = state.organize_bboxes_and_kpts(box_coordinates, box_confidence, kpts)

        model_meta = CocoBodyKeypointsMeta(
            keypoints=kpts,
            boxes=boxes,
            scores=scores,
        )
        meta.add_instance(self.task_name, model_meta, self._where)
        return image, predict, meta

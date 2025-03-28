# Copyright Axelera AI, 2024
# Operators that convert YOLO-specific tensor output to
# generalized metadata representation

import enum
import itertools
from pathlib import Path
import re
from typing import List, Optional

import numpy as np

from axelera import types
from axelera.app import compile, gst_builder, logging_utils
from axelera.app.meta import BBoxState, ObjectDetectionMeta
from axelera.app.operators import AxOperator, PipelineContext, utils
from axelera.app.torch_utils import torch

LOG = logging_utils.getLogger(__name__)


class YoloFamily(enum.Enum):
    YOLOv5 = enum.auto()  # all anchor-based using yolov5 like head
    YOLOv8 = enum.auto()  # all anchor-free using yolov8 like head
    YOLOX = enum.auto()  # anchor-free using yolox like head
    # we don't know which family the model belongs to according to the output shape
    Unknown = enum.auto()


def _filter_samples(scores, class_confidences, box_coordinates, threshold):
    """
    Filters samples based on scores with threshold.

    Args:
        scores (np.ndarray): The score for thresholding.
        class_confidences (np.ndarray): The confidence scores for each class in the bounding boxes.
        box_coordinates (np.ndarray): The coordinates of the bounding boxes.

    Returns:
        tuple: Filtered class confidences, object confidences, and box coordinates.
    """
    valid_indices = scores > threshold
    return (
        scores[valid_indices],
        class_confidences[valid_indices],
        box_coordinates[valid_indices],
    )


class DecodeYolo(AxOperator):
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
    label_filter: Optional[List[str]] = None
    label_exclude: Optional[List[str]] = None
    conf_threshold: float = 0.25
    max_nms_boxes: int = 30000
    use_multi_label: bool = False
    nms_iou_threshold: float = 0.45
    nms_class_agnostic: bool = False
    nms_top_k: int = 300
    generic_gst_decoder: bool = False

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
        # TODO: check config to determine the value of sigmoid_in_postprocess
        self.sigmoid_in_postprocess = False
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
            self.model_type, model_type_explanation = _guess_yolo_model(
                output_shapes, model_info.num_classes
            )
            if self.model_type == YoloFamily.Unknown:
                LOG.warning(f"Unknown model type for {model_info.name}, using generic GST decoder")
                self.generic_gst_decoder = True
            else:
                LOG.debug(f"Model Type: {self.model_type} ({model_type_explanation})")

            if self.model_type == YoloFamily.YOLOv5:
                try:
                    self._anchors = model_info.extra_kwargs['YOLO']['anchors']
                except (TypeError, KeyError):
                    raise ValueError(
                        f"Missing YOLO/anchors in extra_kwargs for {model_info.name}"
                    ) from None
                if not isinstance(self._anchors, (tuple, list)) or not self._anchors:
                    raise ValueError(
                        f"Invalid anchors in extra_kwargs for {model_info.name}:"
                        f" should be list of N lists of 6 elements, got {self._anchors!r}"
                    )
        self.scaled = context.resize_status
        self.model_width = model_info.input_width
        self.model_height = model_info.input_height
        self.labels = model_info.labels
        self.num_classes = model_info.num_classes
        self.use_multi_label &= self.num_classes > 1

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
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
        if self._n_padded_ch_outputs:
            paddings = '|'.join(
                ','.join(str(num) for num in sublist) for sublist in self._n_padded_ch_outputs
            )
        else:
            raise ValueError(f"Missing n_padded_ch_outputs for {self.model_name}")

        if self.model_type == YoloFamily.YOLOv8:
            gst.decode_muxer(
                name=f'decoder_task{self._taskn}{stream_idx}',
                lib='libdecode_yolov8.so',
                mode='read',
                options=f'meta_key:{str(self.task_name)};'
                f'{master_key}'
                f'classes:{self.num_classes};'
                f'confidence_threshold:{self.conf_threshold};'
                f'scales:{scales};'
                f'padding:{paddings};'
                f'zero_points:{zeros};'
                f'topk:{self.max_nms_boxes};'
                f'multiclass:{int(self.use_multi_label)};'
                f'classlabels_file:{self._tmp_labels};'
                f'model_width:{self.model_width};'
                f'model_height:{self.model_height};'
                f'scale_up:{int(self.scaled==types.ResizeMode.LETTERBOX_FIT)};'
                f'letterbox:{int(self.scaled in [types.ResizeMode.LETTERBOX_FIT, types.ResizeMode.LETTERBOX_CONTAIN])}'
                + (f';label_filter:{",".join(sieve)}' if sieve else ''),
            )
        elif self.model_type == YoloFamily.YOLOX:
            gst.decode_muxer(
                name=f'decoder_task{self._taskn}{stream_idx}',
                lib='libdecode_yolox.so',
                mode='read',
                options=f'meta_key:{str(self.task_name)};'
                f'classes:{self.num_classes};'
                f'confidence_threshold:{self.conf_threshold};'
                f'scales:{scales};'
                f'zero_points:{zeros};'
                f'padding:{paddings};'
                f'topk:{self.max_nms_boxes};'
                f'multiclass:{int(self.use_multi_label)};'
                f'classlabels_file:{self._tmp_labels};'
                f'model_width:{self.model_width};'
                f'model_height:{self.model_height};'
                f'scale_up:{int(self.scaled==types.ResizeMode.LETTERBOX_FIT)};'
                f'letterbox:{int(self.scaled in [types.ResizeMode.LETTERBOX_FIT, types.ResizeMode.LETTERBOX_CONTAIN])}'
                + (f';label_filter:{",".join(sieve)}' if sieve else ''),
            )
        elif self.model_type == YoloFamily.YOLOv5:
            anchors = ','.join(str(s) for s in itertools.chain.from_iterable(self._anchors))
            gst.decode_muxer(
                name=f'decoder_task{self._taskn}{stream_idx}',
                lib='libdecode_yolov5.so',
                mode='read',
                options=f'meta_key:{str(self.task_name)};'
                f'{master_key}'
                f'anchors:{anchors};'
                f'classes:{self.num_classes};'
                f'confidence_threshold:{self.conf_threshold};'
                f'scales:{scales};'
                f'zero_points:{zeros};'
                f'topk:{self.max_nms_boxes};'
                f'multiclass:{int(self.use_multi_label)};'
                f'sigmoid_in_postprocess:{int(self.sigmoid_in_postprocess)};'
                f'transpose:1;'
                f'classlabels_file:{self._tmp_labels};'
                f'model_width:{self.model_width};'
                f'model_height:{self.model_height};'
                f'scale_up:{int(self.scaled==types.ResizeMode.LETTERBOX_FIT)};'
                f'letterbox:{int(self.scaled in [types.ResizeMode.LETTERBOX_FIT, types.ResizeMode.LETTERBOX_CONTAIN])}'
                + (f';label_filter:{",".join(sieve)}' if sieve else ''),
            )
        elif self.generic_gst_decoder:  # YoloFamily.Unknown or YAML config
            gst.decode_muxer(
                name=f'decoder_task{self._taskn}{stream_idx}',
                lib='libdecode_yolo.so',
                mode='read',
                options=f'meta_key:{str(self.task_name)};'
                f'{master_key}'
                f'classes:{self.num_classes};'
                f'confidence_threshold:{self.conf_threshold};'
                f'scales:{scales};'
                f'zero_points:{zeros};'
                f'paddings:{paddings};'
                f'model_width:{self.model_width};'
                f'model_height:{self.model_height};'
                f'normalized_coord:{int(self.normalized_coord)};'
                f'topk:{self.max_nms_boxes};'
                f'multiclass:{int(self.use_multi_label)};'
                f'transpose:1;'  # AIPU always NHWC
                f'feature_decoder_onnx:{self._postprocess_graph};'
                f'classlabels_file:{self._tmp_labels};'
                f'scale_up:{int(self.scaled==types.ResizeMode.LETTERBOX_FIT)}'
                + (f';label_filter:{",".join(sieve)}' if sieve else ''),
            )
        else:
            raise ValueError(
                f"Unsupported model type {self.model_type}. Please try to enable generic_gst_decoder in YAML config."
            )

        gst.axinplace(
            lib='libinplace_nms.so',
            options=f'meta_key:{str(self.task_name)};'
            f'{master_key}'
            f'max_boxes:{self.nms_top_k};'
            f'nms_threshold:{self.nms_iou_threshold};'
            f'class_agnostic:{int(self.nms_class_agnostic)};'
            f'location:CPU',
        )

    def exec_torch(self, image, predict, meta):
        if type(predict) == torch.Tensor:
            predict = predict.cpu().detach().numpy()

        if len(predict) == 1 and predict.shape[0] > 1:
            raise ValueError(
                f"Batch size >1 not supported for torch and torch-aipu pipelines, output tensor={predict[0].shape}"
            )
        elif len(predict) > 1:  # Handling multiple predictions, possibly yolo-nas
            # Determine the dimension with consistent size across predictions
            info_at_dim = 1 if predict[0].shape[1] < predict[0].shape[2] else 2
            # Validate if the dimension sizes match to ensure compatibility
            if len(predict) > 2:
                raise ValueError(f"Unexpected number of predictions ({len(predict)}) encountered.")
            elif (
                predict[0].shape[info_at_dim] == 4
                and predict[1].shape[info_at_dim] == self.num_classes
            ):
                # Merge predictions if exactly two are present
                predict = np.concatenate(predict, axis=2)
            else:
                raise ValueError(
                    f"Unexpected output shapes, {predict[0].shape} and {predict[1].shape}"
                )

        bboxes = predict[0]
        # Ensure bboxes are transposed to format (number of samples, info per sample) if needed
        if bboxes.shape[0] < bboxes.shape[1]:
            bboxes = bboxes.transpose()
        # Calculate the number of output channels excluding class predictions
        output_channels_excluding_classes = bboxes.shape[1] - self.num_classes
        if output_channels_excluding_classes == 5:
            # Anchor-based result
            class_confidences = bboxes[:, 5:]
            object_confidence = bboxes[:, 4]
            box_coordinates = bboxes[:, :4]
            has_object_confidence = True
        elif output_channels_excluding_classes == 4:
            # Anchor-free result which has no object confidences
            # Use the max class confidence for the initial candidate filtering
            class_confidences = bboxes[:, 4:]
            object_confidence = class_confidences.max(axis=1)
            box_coordinates = bboxes[:, :4]
            has_object_confidence = False
        else:
            raise ValueError(f"Unknown number of output channels {bboxes.shape}")

        # Filter samples by object confidence threshold
        object_confidence, class_confidences, box_coordinates = _filter_samples(
            object_confidence, class_confidences, box_coordinates, self.conf_threshold
        )

        if not self.use_multi_label:
            best_class = class_confidences.argmax(axis=1)
            meshgrid = np.ogrid[: class_confidences.shape[0]]
            best_class_conf = class_confidences[meshgrid, best_class]
            if has_object_confidence:
                scores = object_confidence * best_class_conf
            else:
                scores = best_class_conf
            # filter samples by "score", and get boxes, scores, and classes
            scores, classes, boxes = _filter_samples(
                scores, best_class, box_coordinates, self.conf_threshold
            )
        else:  # typically for evaluation
            # Compute initial scores based on object confidence and class confidences
            if has_object_confidence:
                scores = object_confidence[:, None] * class_confidences
            else:
                # When there's no object confidence, use class confidences as scores
                scores = class_confidences

            # Find all scores above the confidence threshold
            valid_scores_indices = np.argwhere(scores > self.conf_threshold)

            if valid_scores_indices.size > 0:  # Check if there are any scores above the threshold
                scores = scores[valid_scores_indices[:, 0], valid_scores_indices[:, 1]]
                classes = valid_scores_indices[:, 1]
                boxes = box_coordinates[valid_scores_indices[:, 0]]
            else:  # No scores above the threshold, initialize empty arrays
                scores = np.array([])
                classes = np.array([])
                boxes = np.array([]).reshape(0, box_coordinates.shape[1])

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
        boxes, scores, classes = state.organize_bboxes(boxes, scores, classes)

        if self._where:
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


def _guess_yolo_model(depadded_shapes, num_classes):
    """
    Guess the YOLO model variant based on depadded shapes and number of classes.

    Args:
        depadded_shapes: List of lists containing actual shape information [[batch, height, width, channels], ...]
        num_classes: Number of classes the model was trained on

    Returns:
        tuple: (YoloFamily enum, str explanation)
    """

    num_outputs = len(depadded_shapes)
    channels = [shape[3] for shape in depadded_shapes]

    def analyze_yolov5():
        # YOLOv5: 3 outputs, each with 3(anchors) * (4 + 1 + num_classes) channels
        expected_channels = 3 * (4 + 1 + num_classes)
        return all(c == expected_channels for c in channels)

    def analyze_yolox():
        # YOLOX: 9 outputs with specific pattern
        if len(channels) != 9:
            return False

        expected_pattern = [
            num_classes,  # cls
            1,  # obj
            num_classes,  # cls
            1,  # obj
            num_classes,  # cls
            1,  # obj
            4,  # box
            4,  # box
            4,  # box
        ]
        return channels == expected_pattern

    def analyze_yolov8():
        # YOLOv8: 6 outputs, split between regression and classification
        if len(channels) != 6:
            return False

        reg_channels = channels[:3]
        cls_channels = channels[3:]

        return all(c == 64 for c in reg_channels) and all(c == num_classes for c in cls_channels)

    if num_outputs == 3 and analyze_yolov5():
        explanation = (
            "YOLOv5 pattern:\n"
            "- 3 output tensors (anchor-based)\n"
            f"- Each output has {channels[0]} channels\n"
            f"  = 3 anchors × (4 box + 1 obj + {num_classes} classes)\n"
            f"  = 3 × ({4} + {1} + {num_classes}) = {3 * (4 + 1 + num_classes)}\n"
            f"- Shapes: {[list(shape) for shape in depadded_shapes]}"
        )
        return YoloFamily.YOLOv5, explanation

    elif num_outputs == 9 and analyze_yolox():
        explanation = (
            "YOLOX pattern:\n"
            "- 9 output tensors (anchor-free)\n"
            "- Separate outputs for cls/obj/box predictions\n"
            f"- Classification branches: {num_classes} channels\n"
            "- Objectness branches: 1 channel\n"
            "- Box branches: 4 channels\n"
            f"- Channel pattern: {channels}\n"
            f"- Shapes: {[list(shape) for shape in depadded_shapes]}"
        )
        return YoloFamily.YOLOX, explanation

    elif num_outputs == 6 and analyze_yolov8():
        explanation = (
            "YOLOv8 pattern:\n"
            "- 6 output tensors (anchor-free)\n"
            "- 3 regression branches (64 channels)\n"
            f"- 3 classification branches ({num_classes} channels)\n"
            f"- Channel pattern: {channels}\n"
            f"- Shapes: {[list(shape) for shape in depadded_shapes]}"
        )
        return YoloFamily.YOLOv8, explanation

    else:
        explanation = (
            "Unknown pattern:\n"
            f"- {num_outputs} output tensors\n"
            f"- Channel dimensions: {channels}\n"
            f"- Shapes: {[list(shape) for shape in depadded_shapes]}"
        )
        return YoloFamily.Unknown, explanation

# Copyright Axelera AI, 2023
# Operators that convert YOLO-specific tensor output to
# generalized metadata representation

import enum
import itertools
from pathlib import Path
from typing import List, Optional

import numpy as np

from axelera import types
from axelera.app import compile, gst_builder, logging_utils
from axelera.app.meta import BBoxState, ObjectDetectionMeta, ObjectDetectionMetaOBB
from axelera.app.model_utils.box import batch_probiou
from axelera.app.operators import AxOperator, PipelineContext, utils
from axelera.app.torch_utils import torch

LOG = logging_utils.getLogger(__name__)


class YoloFamily(enum.Enum):
    YOLOv5 = enum.auto()  # all anchor-based using yolov5 like head
    YOLOv8 = enum.auto()  # all anchor-free using yolov8 like head
    YOLOX = enum.auto()  # anchor-free using yolox like head
    YOLO_OBB = enum.auto()  # oriented bounding box
    YOLO_NAS = (
        enum.auto()
    )  # anchor-free, same grid structure as YOLOv8 but different DFL bin count
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


def _merge_dual_tensor_predictions(predict, num_classes):
    """Merge two separate prediction tensors (e.g. YOLO-NAS) into one [boxes, classes] tensor.

    Some models output boxes and class scores as separate tensors in either order.
    Identifies which tensor is boxes (4 channels) vs classes (num_classes channels)
    and concatenates them in [boxes, classes] order for downstream decoding.
    """
    if len(predict) > 2:
        raise ValueError(f"Unexpected number of predictions ({len(predict)}) encountered.")
    # The channel dimension is whichever of dims[1],dims[2] is smaller
    info_at_dim = 1 if predict[0].shape[1] < predict[0].shape[2] else 2
    if predict[0].shape[info_at_dim] == 4 and predict[1].shape[info_at_dim] == num_classes:
        return np.concatenate(predict, axis=info_at_dim)
    if predict[0].shape[info_at_dim] == num_classes and predict[1].shape[info_at_dim] == 4:
        return np.concatenate([predict[1], predict[0]], axis=info_at_dim)
    raise ValueError(f"Unexpected output shapes, {predict[0].shape} and {predict[1].shape}")


def _decode_yolo_grid_boxes(boxes, model_width, model_height, strides=[8, 16, 32]):
    """
    Decode YOLO-family boxes from grid-space to pixel-space coordinates.

    Some YOLO ONNX exports output raw grid predictions that need decoding:
    - boxes[:, :2] = (offset + grid_xy) * stride  # center coordinates
    - boxes[:, 2:4] = exp(log_wh) * stride        # width, height

    This is the standard YOLO grid decoding used by YOLOX, YOLOv5, YOLOv8, etc.
    when exported with decode_in_inference=False or similar settings.

    Args:
        boxes (np.ndarray): Box coordinates in grid-space format (N, 4) as [cx_offset, cy_offset, w_log, h_log]
        model_width (int): Model input width
        model_height (int): Model input height
        strides (list): Stride values for each feature pyramid level (default: [8, 16, 32])

    Returns:
        np.ndarray: Decoded boxes in pixel-space as [cx, cy, w, h]
    """
    # Calculate grid dimensions for each stride level
    grid_dims = []
    for stride in strides:
        h = model_height // stride
        w = model_width // stride
        grid_dims.append((h, w))

    # Generate grid coordinates and stride arrays
    all_grids = []
    all_strides = []

    for stride, (h, w) in zip(strides, grid_dims):
        # Create grid for this level
        grid_y, grid_x = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
        grid = np.stack([grid_x, grid_y], axis=-1).reshape(-1, 2)  # (h*w, 2)
        all_grids.append(grid)

        # Create stride array for this level
        stride_array = np.full((h * w, 1), stride, dtype=np.float32)
        all_strides.append(stride_array)

    # Concatenate all grids and strides
    all_grids = np.concatenate(all_grids, axis=0)  # (total_anchors, 2)
    all_strides = np.concatenate(all_strides, axis=0)  # (total_anchors, 1)

    # Decode boxes using standard YOLO grid decoding logic
    boxes_decoded = boxes.copy()

    # Decode center coordinates: (offset + grid_xy) * stride
    boxes_decoded[:, 0:2] = (boxes[:, 0:2] + all_grids[: len(boxes)]) * all_strides[: len(boxes)]

    # Decode width/height: exp(log_wh) * stride
    boxes_decoded[:, 2:4] = np.exp(boxes[:, 2:4]) * all_strides[: len(boxes)]

    return boxes_decoded


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
    label_filter: Optional[List[str] | str] = None
    label_exclude: Optional[List[str] | str] = None
    conf_threshold: float = 0.25
    max_nms_boxes: int = 30000
    use_multi_label: bool = False
    nms_iou_threshold: float = 0.45
    nms_class_agnostic: bool = False
    nms_top_k: int = 300
    generic_gst_decoder: bool = False
    nms_free: bool = False  # Currently only used for YOLO26-OBB without focal loss
    # Per-feature-map xy/wh decode params. When None, values are read from
    # `model_info.extra_kwargs["YOLO"]` (Darknet path auto-populates from .cfg);
    # otherwise the YAML values override. Set explicitly when the model was
    # trained with a custom decoder convention not derivable from the loader.
    scale_x_y: Optional[List[float]] = None
    new_coords: Optional[List[bool]] = None

    def _post_init(self):
        self.label_filter = utils.parse_labels_filter(self.label_filter)
        self.label_exclude = utils.parse_labels_filter(self.label_exclude)

        self._tmp_labels: Optional[Path] = None
        if self.box_format not in ["xyxy", "xywh", "ltwh", "xywhr", "xyxyxyxy"]:
            raise ValueError(f"Unknown box format {self.box_format}")

        self.model_type = YoloFamily.Unknown
        self._dfl_bins = None
        if self.box_format == "xyxyxyxy":
            self.x_indexes = [0, 2, 4, 6]
            self.y_indexes = [1, 3, 5, 7]
            self.model_type = YoloFamily.YOLO_OBB
        elif self.box_format == "xywhr":
            self.x_indexes = [0]
            self.y_indexes = [1]
            self.model_type = YoloFamily.YOLO_OBB
        else:
            self.x_indexes = [0, 2]  # x1, x2
            self.y_indexes = [1, 3]  # y1, y2

        # When True the C++ decoder's int8->float LUT bakes in sigmoid. The
        # default False matches models that fuse sigmoid into the chip kernel
        # (e.g. Ultralytics v5/v7/v8 ONNX exports). The Darknet loader sets
        # this True via extra_kwargs since YOLOLayer's sigmoid stays in the
        # postamble graph and the gst pipe bypasses the postamble.
        self.sigmoid_in_postprocess = False
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

            model_type_explanation = "Determined from box format"
            if self.model_type == YoloFamily.Unknown:
                self.model_type, model_type_explanation, extra = _guess_yolo_model(
                    output_shapes, model_info.num_classes
                )
                self._dfl_bins = extra.get('dfl_bins')

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
                # Per-feature-map xy/wh activation params. YAML override beats
                # loader-derived (extra_kwargs). When neither is set we leave
                # `_scale_x_y` / `_new_coords` as None and the gst decoder falls
                # back to its hard-coded Ultralytics v5 defaults -- this keeps
                # the gst option string byte-identical for v5/v7/v8 models that
                # do not need explicit values.
                yolo_kw = model_info.extra_kwargs['YOLO']
                self._scale_x_y = self.scale_x_y or yolo_kw.get('scale_x_y')
                self._new_coords = (
                    self.new_coords if self.new_coords is not None else yolo_kw.get('new_coords')
                )
                if 'sigmoid_in_postprocess' in yolo_kw:
                    self.sigmoid_in_postprocess = bool(yolo_kw['sigmoid_in_postprocess'])
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

        scales = ','.join(str(s) for s in self._deq_scales)
        zeros = ','.join(str(s) for s in self._deq_zeropoints)
        sieve = utils.build_class_sieve(self.label_filter, self.labels)
        tiling = gst_builder.TileInfo(self, gst)
        master_key, association_key = tiling.get_decode_keys()
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
                f'{association_key}'
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
        elif self.model_type == YoloFamily.YOLO_NAS:
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
                f'topk:{self.max_nms_boxes};'
                f'multiclass:{int(self.use_multi_label)};'
                f'classlabels_file:{self._tmp_labels};'
                f'model_width:{self.model_width};'
                f'model_height:{self.model_height};'
                f'dfl_size:{self._dfl_bins};'
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
                f'{master_key}'
                f'{association_key}'
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
            activation_opts = ''
            if self._scale_x_y is not None:
                activation_opts += f'scale_x_y:{",".join(str(s) for s in self._scale_x_y)};'
            if self._new_coords is not None:
                activation_opts += (
                    f'new_coords:{",".join(str(int(bool(s))) for s in self._new_coords)};'
                )
            gst.decode_muxer(
                name=f'decoder_task{self._taskn}{stream_idx}',
                lib='libdecode_yolov5.so',
                mode='read',
                options=f'meta_key:{str(self.task_name)};'
                f'{master_key}'
                f'{association_key}'
                f'anchors:{anchors};'
                f'classes:{self.num_classes};'
                f'confidence_threshold:{self.conf_threshold};'
                f'scales:{scales};'
                f'zero_points:{zeros};'
                f'topk:{self.max_nms_boxes};'
                f'multiclass:{int(self.use_multi_label)};'
                f'sigmoid_in_postprocess:{int(self.sigmoid_in_postprocess)};'
                f'{activation_opts}'
                f'transpose:1;'
                f'classlabels_file:{self._tmp_labels};'
                f'model_width:{self.model_width};'
                f'model_height:{self.model_height};'
                f'scale_up:{int(self.scaled==types.ResizeMode.LETTERBOX_FIT)};'
                f'letterbox:{int(self.scaled in [types.ResizeMode.LETTERBOX_FIT, types.ResizeMode.LETTERBOX_CONTAIN])}'
                + (f';label_filter:{",".join(sieve)}' if sieve else ''),
            )
        elif self.model_type == YoloFamily.YOLO_OBB:
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
                f'topk:{self.max_nms_boxes};'
                f'multiclass:{int(self.use_multi_label)};'
                f'has_angle:1;'
                + (f'raw_radians:1;dfl_size:1;' if self.nms_free else '')
                + f'classlabels_file:{self._tmp_labels};'
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
                f'{association_key}'
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

        # NMS: skip for nms_free models unless tiling is enabled
        if not self.nms_free or gst.tiling:
            master_key = tiling.get_nms_keys()
            gst.axinplace(
                lib='libinplace_nms.so',
                options=f'meta_key:{str(self.task_name)};'
                f'{master_key}'
                f'max_boxes:{tiling.get_max_boxes(self.nms_top_k)};'
                f'nms_threshold:{self.nms_iou_threshold};'
                f'class_agnostic:{int(self.nms_class_agnostic)};'
                f'location:CPU',
            )

    def exec_torch(self, image, predict, meta):
        if type(predict) == torch.Tensor:
            predict = predict.cpu().detach().numpy()

        # Handle OBB (Oriented Bounding Box) models
        if self.model_type == YoloFamily.YOLO_OBB:
            boxes, scores, classes = decode_raw_obb(
                predict,
                self.conf_threshold,
                self.nms_iou_threshold,
                self.nms_top_k,
                self.max_nms_boxes,
                self.nms_class_agnostic,
                self.use_multi_label,
                self.nms_free,
            )
        else:
            # Standard YOLO detection path
            predict_info = (
                [p.shape for p in predict] if isinstance(predict, (list, tuple)) else predict.shape
            )
            LOG.debug(f"Prediction tensor info: len={len(predict)}, shapes={predict_info}")

            if len(predict) == 1 and predict.shape[0] > 1:
                raise ValueError(
                    f"Batch size >1 not supported for torch and torch-aipu pipelines, output tensor={predict[0].shape}"
                )
            elif len(predict) > 1:
                predict = _merge_dual_tensor_predictions(predict, self.num_classes)

            bboxes = predict[0]
            # Ensure bboxes are transposed to format (number of samples, info per sample) if needed
            if bboxes.shape[0] < bboxes.shape[1]:
                bboxes = bboxes.transpose()
            # Calculate the number of output channels excluding class predictions
            # YOLOX format: [box(4), obj(1), cls(num_classes)]
            total_channels = bboxes.shape[1]
            expected_yolox_channels = 4 + 1 + self.num_classes  # box + obj + class

            if total_channels == expected_yolox_channels:
                # YOLOX anchor-free: [box(4), obj(1), cls(N)]
                box_coordinates = bboxes[:, :4]
                object_confidence = bboxes[:, 4]
                class_confidences = bboxes[:, 5:]
                has_object_confidence = True  # YOLOX has objectness
            elif total_channels == 4 + self.num_classes:
                # YOLOv8-style anchor-free without objectness: [box(4), cls(N)]
                box_coordinates = bboxes[:, :4]
                class_confidences = bboxes[:, 4:]
                object_confidence = class_confidences.max(axis=1)
                has_object_confidence = False
            else:
                raise ValueError(
                    f"Unknown number of output channels: {bboxes.shape}, expected {expected_yolox_channels} for YOLOX or {4+self.num_classes} for YOLOv8"
                )

            # Check if boxes need YOLO grid decoding
            # Some YOLO ONNX exports output raw grid predictions that need decoding:
            #   centers = (offset + grid_xy) * stride, sizes = exp(log_wh) * stride
            # This applies to YOLOX, YOLOv5, YOLOv8, etc. when exported with decode_in_inference=False
            # Auto-detect by checking if box values are in grid-space range vs pixel-space
            needs_grid_decode = False
            if len(box_coordinates) > 0 and not self.normalized_coord:
                sample_boxes = box_coordinates[: min(100, len(box_coordinates))]
                max_center = np.max(np.abs(sample_boxes[:, :2]))
                max_size = np.max(sample_boxes[:, 2:4])
                # Grid offsets are typically in [-10, 10], log(wh) in [0, 5]
                # If boxes are already in pixel space, centers would be 100s-1000s
                if max_center < 50 and max_size < 10:
                    needs_grid_decode = True
                    LOG.debug(f"Detected grid-space boxes, applying YOLO grid decoding")

            if needs_grid_decode:
                box_coordinates = _decode_yolo_grid_boxes(
                    box_coordinates, self.model_width, self.model_height
                )

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

                if (
                    valid_scores_indices.size > 0
                ):  # Check if there are any scores above the threshold
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
            boxes[:, self.x_indexes] += base_box[0]
            boxes[:, self.y_indexes] += base_box[1]

        MetaCls = (
            ObjectDetectionMetaOBB
            if self.model_type == YoloFamily.YOLO_OBB
            else ObjectDetectionMeta
        )
        model_meta = MetaCls.create_immutable_meta(
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
        tuple: (YoloFamily enum, str explanation, dict extra)
            extra may contain 'dfl_bins' for YOLO-NAS models
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
        # YOLOv8: Handle standard, P6, and P2 versions
        if len(channels) not in [6, 8, 10]:
            return False

        if len(channels) == 6:  # Standard version
            reg_channels = channels[:3]
            cls_channels = channels[3:]
        elif len(channels) == 8:  # P6 version (1280x1280)
            reg_channels = channels[:4]
            cls_channels = channels[4:]
        elif len(channels) == 10:  # P2 version (320x320)
            reg_channels = channels[:5]
            cls_channels = channels[5:]

        return all(c == 64 for c in reg_channels) and all(c == num_classes for c in cls_channels)

    def analyze_yolonas():
        """Detect YOLO-NAS: 6 tensors at 3 scales, paired cls/box per scale.
        Box channels are divisible by 4 (DFL bins * 4) but != 64 (that's YOLOv8).
        Works for any num_classes as long as num_classes != box_channels."""
        if len(channels) != 6:
            return False, 0
        cls_channels = []
        box_channels = []
        for c in channels:
            if c == num_classes:
                cls_channels.append(c)
            elif c % 4 == 0 and c // 4 > 1 and c != 64:
                box_channels.append(c)
            else:
                return False, 0
        if len(cls_channels) != 3 or len(box_channels) != 3:
            return False, 0
        if len(set(box_channels)) != 1:
            return False, 0
        if box_channels[0] % 4 != 0:
            return False, 0
        dfl_bins = box_channels[0] // 4
        return True, dfl_bins

    if num_outputs == 3 and analyze_yolov5():
        explanation = (
            "YOLOv5 pattern:\n"
            "- 3 output tensors (anchor-based)\n"
            f"- Each output has {channels[0]} channels\n"
            f"  = 3 anchors × (4 box + 1 obj + {num_classes} classes)\n"
            f"  = 3 × ({4} + {1} + {num_classes}) = {3 * (4 + 1 + num_classes)}\n"
            f"- Shapes: {[list(shape) for shape in depadded_shapes]}"
        )
        return YoloFamily.YOLOv5, explanation, {}

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
        return YoloFamily.YOLOX, explanation, {}

    elif num_outputs == 6 and analyze_yolov8():
        explanation = (
            "YOLOv8 pattern:\n"
            "- 6 output tensors (anchor-free)\n"
            "- 3 regression branches (64 channels)\n"
            f"- 3 classification branches ({num_classes} channels)\n"
            f"- Channel pattern: {channels}\n"
            f"- Shapes: {[list(shape) for shape in depadded_shapes]}"
        )
        return YoloFamily.YOLOv8, explanation, {}

    else:
        is_nas, dfl_bins = analyze_yolonas()
        if is_nas:
            explanation = (
                "YOLO-NAS pattern:\n"
                "- 6 output tensors (anchor-free, same grid structure as YOLOv8)\n"
                f"- 3 classification branches ({num_classes} channels)\n"
                f"- 3 regression branches ({dfl_bins * 4} channels = {dfl_bins} DFL bins x 4)\n"
                f"- Channel pattern: {channels}\n"
                f"- Shapes: {[list(shape) for shape in depadded_shapes]}"
            )
            return YoloFamily.YOLO_NAS, explanation, {'dfl_bins': dfl_bins}

        if num_outputs == 6 and num_classes % 4 == 0 and num_classes // 4 > 1:
            LOG.error(
                f"Cannot distinguish YOLO-NAS class tensors from box tensors: "
                f"num_classes={num_classes} is divisible by 4, which collides with "
                f"the DFL box channel count. All 6 tensors appear identical. "
                f"Falling back to generic decoder (slower). "
                f"Channel pattern: {channels}"
            )

        explanation = (
            "Unknown pattern:\n"
            f"- {num_outputs} output tensors\n"
            f"- Channel dimensions: {channels}\n"
            f"- Shapes: {[list(shape) for shape in depadded_shapes]}"
        )
        return YoloFamily.Unknown, explanation, {}


def nms_obb(boxes_xywhr, scores, classes=None, iou_thr=0.5, max_det=300, agnostic=False):
    """
    NumPy fast-NMS using probabilistic IoU.

    Args:
        boxes_xywhr: (N, 5) array of [x, y, w, h, r] boxes.
        scores:      (N,) scores.
        classes:     (N,) class ids (ints or floats). If provided and agnostic=False,
                     only boxes of the same class can suppress each other.
        iou_thr:     IoU threshold for suppression.
        max_det:     Max number of detections to keep.
        agnostic:    If True, ignore classes (all can suppress each other).

    Returns:
        keep: (M,) int64 indices of kept boxes (original indexing).
    """
    if boxes_xywhr.size == 0:
        return np.zeros((0,), dtype=np.int64)

    order = np.argsort(-scores.astype(np.float64))
    b = boxes_xywhr[order].astype(np.float64)

    # Pairwise probabilistic IoU
    ious = batch_probiou(b, b).astype(np.float64)

    # Class-aware masking (only same-class overlaps can suppress)
    if classes is not None and not agnostic:
        cls = classes[order].reshape(-1, 1)
        same_class = cls == cls.T
        # Zero out IoUs between different classes so they never suppress
        ious = ious * same_class.astype(np.float64)

    # Only consider higher-scored boxes suppressing lower-scored ones
    ious = np.triu(ious, k=1)

    # Keep boxes that are not overlapped >= thr by any higher-scored box
    keep_mask = (ious >= float(iou_thr)).sum(axis=0) == 0
    pick = np.nonzero(keep_mask)[0]

    if pick.size > max_det:
        pick = pick[:max_det]

    keep = order[pick]
    return keep.astype(np.int64)


def decode_raw_obb(
    raw_tensor,
    conf_thres=0.25,
    iou_thres=0.50,
    max_det=300,
    max_nms=30000,
    agnostic=False,
    multi_label=True,
    nms_free=False,
):
    """
    Pure NumPy decode for YOLO OBB raw outputs.

    Args:
        raw_tensor: Model output tensor
        conf_thres: Confidence threshold for filtering
        iou_thres: IoU threshold for NMS (ignored if nms_free=True)
        max_det: Maximum detections after NMS
        max_nms: Maximum boxes before NMS
        agnostic: Class-agnostic NMS (ignored if nms_free=True)
        multi_label: Multi-label detection (ignored if nms_free=True)
        nms_free: If True, expects YOLO26/YOLO10 format [batch, num_det, 7] with built-in NMS
                  If False, expects YOLO8 format [batch, num_anchors, 4+nc+1] requiring NMS
    """
    # Unwrap single output
    if isinstance(raw_tensor, (tuple, list)):
        raw_tensor = raw_tensor[0]

    # Convert to numpy and ensure 3D [B, N, C]
    if isinstance(raw_tensor, torch.Tensor):
        arr = raw_tensor.detach().cpu().numpy()
    elif isinstance(raw_tensor, np.ndarray):
        arr = raw_tensor
    else:
        raise RuntimeError(f"Unsupported raw type: {type(raw_tensor)}")

    if arr.ndim == 2:  # [N, C]
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise RuntimeError(f"Unexpected RAW ndim: {arr.ndim}, shape={arr.shape}")

    # Handle NMS-free models (YOLO26/YOLO10)
    if nms_free:
        # Format: [batch, num_detections, 7] where 7 = [x, y, w, h, conf, class_id, angle]
        bboxes = arr[0]  # Take first batch

        # Transpose if needed (smaller dim should be features)
        if bboxes.shape[0] < bboxes.shape[1]:
            bboxes = bboxes.transpose()

        # Filter by confidence threshold
        mask = bboxes[:, 4] >= conf_thres
        bboxes = bboxes[mask]

        # Extract components
        box_xywh = bboxes[:, :4]
        scores = bboxes[:, 4].astype(np.float32)
        classes = bboxes[:, 5].astype(np.int64)
        box_angles = bboxes[:, 6:7]

        # Combine to xywhr format
        boxes = np.column_stack([box_xywh, box_angles]).astype(np.float32)

        return boxes, scores, classes

    # Standard YOLO8 OBB path requiring NMS
    pred = np.transpose(arr, (0, 2, 1)).copy()

    # Split fields according to Ultralytics rotated layout: [xywh | nc class probs | angle]
    C = pred.shape[2]
    nc = C - 5
    boxes_xywh = pred[..., :4]  # normalized xywh
    cls_probs = pred[..., 4 : 4 + nc]  # class probabilities (already sigmoid-ed by head)
    theta = pred[..., 4 + nc : 4 + nc + 1]  # angle radians (shape [B,N,1])

    # Use only first image in batch for this helper
    boxes0_xywh_all = boxes_xywh[0]
    theta0_all = theta[0]
    cls_probs0_all = cls_probs[0]

    # Anchor-level keep: any class score above threshold
    keep_anchor = (cls_probs0_all > float(conf_thres)).any(axis=1)
    if not np.any(keep_anchor):
        return (
            np.empty((0, 5), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    boxes0_xywh = boxes0_xywh_all[keep_anchor]
    theta0 = theta0_all[keep_anchor]
    cls_probs0 = cls_probs0_all[keep_anchor]

    if multi_label:
        ai, aj = np.nonzero(cls_probs0 > float(conf_thres))
        if ai.size == 0:
            return (
                np.empty((0, 5), dtype=np.float32),
                np.empty((0,), dtype=np.float32),
                np.empty((0,), dtype=np.int64),
            )
        conf0 = cls_probs0[ai, aj]
        cls0 = aj.astype(np.int64)
        boxes0_xywh = boxes0_xywh[ai]
        theta0 = theta0[ai]
        if max_nms is not None and conf0.shape[0] > int(max_nms):
            idx = np.argpartition(-conf0, int(max_nms) - 1)[: int(max_nms)]
            boxes0_xywh = boxes0_xywh[idx]
            theta0 = theta0[idx]
            conf0 = conf0[idx]
            cls0 = cls0[idx]
    else:
        conf0 = cls_probs0.max(axis=1)
        cls0 = cls_probs0.argmax(axis=1).astype(np.int64)
        if max_nms is not None and boxes0_xywh.shape[0] > int(max_nms):
            idx = np.argpartition(-conf0, int(max_nms) - 1)[: int(max_nms)]
            boxes0_xywh = boxes0_xywh[idx]
            theta0 = theta0[idx]
            conf0 = conf0[idx]
            cls0 = cls0[idx]

    # Boxes are already in canvas pixels (head multiplies by stride); keep radians angle
    boxes_xywhr = np.concatenate([boxes0_xywh, theta0], axis=1).astype(np.float64)

    # Rotated NMS per class (unless agnostic=True)
    keep_idx = nms_obb(
        boxes_xywhr,
        conf0.astype(np.float64),
        classes=None if agnostic else cls0,
        iou_thr=float(iou_thres),
        max_det=int(max_det),
        agnostic=bool(agnostic),
    )
    if keep_idx.size == 0:
        return (
            np.empty((0, 5), dtype=np.float32),
            np.empty((0,), dtype=np.float32),
            np.empty((0,), dtype=np.int64),
        )

    boxes_out = boxes_xywhr[keep_idx].astype(np.float32)
    conf_out = conf0[keep_idx].astype(np.float32)
    cls_out = cls0[keep_idx].astype(np.int64)
    return boxes_out, conf_out, cls_out

# Copyright Axelera AI, 2025
# A model pipeline must start from an Input Operator
# which return a list of Axelera Image and meta.
from __future__ import annotations

import inspect
from typing import TYPE_CHECKING

import numpy as np

from axelera import types

from . import utils
from .. import gst_builder, logging_utils, meta
from .base import AxOperator, PreprocessOperator, builtin

if TYPE_CHECKING:
    from pathlib import Path

    from ..pipe import graph
    from .context import PipelineContext

LOG = logging_utils.getLogger(__name__)


def get_input_operator(source: str):
    if source in ["default", "full"]:
        return Input
    elif source == "roi":
        return InputFromROI
    elif source == "image_processing":
        return InputWithImageProcessing
    else:
        raise ValueError(f"Unsupported source: {source}")


def _convert_image_to_types_image_for_deploy(
    image, color_format: types.ColorFormat = types.ColorFormat.RGB
):
    if isinstance(image, types.img.PILImage):
        return types.Image.frompil(image, color_format=color_format)
    elif isinstance(image, np.ndarray):
        return types.Image.fromarray(image, color_format=types.ColorFormat.BGR)
    elif isinstance(image, types.Image):
        return image
    else:
        raise ValueError(f"Unsupported image type: {type(image)}")


@builtin
class Input(AxOperator):
    """
    We now support type==image only.

    imreader_backend is the image loader used in your dataset/dataloader
    color_format is the color format of the loaded image; typically,
    it should be RGB when using PIL, and BGR when using OpenCV
    """

    type: str = 'image'
    color_format: types.ColorFormat = types.ColorFormat.RGB
    imreader_backend: types.ImageReader = types.ImageReader.PIL

    def _post_init(self):
        self._enforce_member_type('color_format')
        self._enforce_member_type('imreader_backend')

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        compiled_model_dir: Path | None,
        task_graph: graph.DependencyGraph,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, compiled_model_dir, task_graph
        )
        context.color_format = self.color_format
        context.imreader_backend = self.imreader_backend
        context.pipeline_input_color_format = self.color_format

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        pass

    def exec_torch(self, image, result, meta, stream_id=0):
        if result is None and meta is None:
            image = _convert_image_to_types_image_for_deploy(image, self.color_format)

        if isinstance(image, types.Image):
            result = [image]
        elif isinstance(image, list):
            for im in image:
                if not isinstance(im, types.Image):
                    raise ValueError("Input must be a list of types.Image")
            result = image
        else:
            raise ValueError("Input must be an types.Image or a list of types.Image")

        if self.color_format != image.color_format:
            for im in result:
                new_im = im.asarray(self.color_format)
                im.update(new_im, color_format=self.color_format)
        return image, result, meta


def _deploy_cascade_model(image: types.Image, color_format: types.ColorFormat):
    assert isinstance(image, types.Image), "Input image must be a types.Image"
    if color_format != image.color_format:
        image.update(image.asarray(color_format))
    return image, None, None


@builtin
class InputFromROI(AxOperator):
    '''ROI extraction from a ObjectDetectionMeta'''

    type: str = 'image'
    where: str
    min_width: int = 0
    min_height: int = 0
    top_k: int = 0
    which: str = 'NONE'
    label_filter: list[str] = []
    image_processing_on_roi: list[PreprocessOperator] = []
    color_format: types.ColorFormat = types.ColorFormat.RGB

    def _post_init(self):
        super()._post_init()
        self._enforce_member_type('color_format')
        SUPPORTED_WHICH = ['AREA', 'SCORE', 'CENTER', 'NONE']
        if self.which.upper() not in SUPPORTED_WHICH:
            raise ValueError(f"which is not in support list: {SUPPORTED_WHICH}")
        self.label_filter = utils.parse_labels_filter(self.label_filter)
        self.where = str(
            self.where
        )  # TODO SAM this is a hack to ensure it's a string not a yamlString

        if self.min_width > 0 and self.min_height == 0:
            raise ValueError("min_height must be set if min_width is set")
        if self.min_width == 0 and self.min_height > 0:
            raise ValueError("min_width must be set if min_height is set")

        self._need_filter = (
            self.which != 'NONE' or self.label_filter or self.min_width > 0 or self.min_height > 0
        )

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        compiled_model_dir: Path | None,
        task_graph: graph.DependencyGraph,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, compiled_model_dir, task_graph
        )
        # for cascade models, we know the input color format passed from the upstream model
        self._need_color_convert = context.pipeline_input_color_format != self.color_format
        context.color_format = self.color_format
        if self.where != self._where:
            raise ValueError(f"where is not equal to _where: {self.where} vs {self._where}")
        self._association = self._where
        self._input_meta_key = self._where
        master_is_tracker = (
            task_graph.get_task(self._where).model_info.task_category
            == types.TaskCategory.ObjectTracking
        )
        if master_is_tracker:
            self._input_meta_key = "boxes_created_by_tracker_task_" + self._where
            if self.task_name in context.submodels_with_boxes_from_tracker:
                self._input_meta_key = self._where + "_adapted_as_input_for_" + self.task_name
            self._association = self._input_meta_key
            context.association = self._association
        if self._need_filter:
            self._association = self._where + "_adapted_as_input_for_" + self.task_name
            context.association = self._association
        self._label_filter_ids = None
        if self.label_filter:
            master_task_with_labels = task_graph.get_task(self._where)
            if master_is_tracker:
                master_of_master = task_graph.get_master(self._where)
                if master_of_master:
                    master_task_with_labels = task_graph.get_task(master_of_master)
            master_labels = master_task_with_labels.model_info.labels
            self._label_filter_ids = utils.build_class_sieve(self.label_filter, master_labels)

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        if self._need_filter:
            classes_to_keep_str = (
                str()
                if self._label_filter_ids is None
                else f';classes_to_keep:{",".join(self._label_filter_ids)}'
            )
            gst.axinplace(
                lib='libinplace_filterdetections.so',
                options=f'input_meta_key:{self._input_meta_key};'
                f'output_meta_key:{self._association};'
                f'hide_output_meta:1;'
                f'which:{self.which};'
                f'top_k:{self.top_k};'
                f'min_width:{self.min_width};'
                f'min_height:{self.min_height}'
                f'{classes_to_keep_str}',
            )

        if self._need_color_convert:
            vaapi = gst.getconfig() is not None and gst.getconfig().vaapi
            opencl = gst.getconfig() is not None and gst.getconfig().opencl
            utils.insert_color_convert(gst, self.color_format, vaapi, opencl)

        gst.start_axinference()
        gst.distributor(meta=str(self._association))
        gst.axtransform(
            lib='libtransform_roicrop.so',
            options=f'meta_key:{self._association}',
        )

        for op in self.image_processing_on_roi:
            op.build_gst(gst, stream_idx)

    def exec_torch(self, image, result, axmeta, stream_id=0):
        if result is None and axmeta is None:
            image = _convert_image_to_types_image_for_deploy(image, self.color_format)
            return _deploy_cascade_model(image, self.color_format)

        assert self._need_color_convert == (
            self.color_format != image.color_format
        ), f"color format is not consistent: {self.color_format} vs {image.color_format}"
        if self._need_color_convert:
            image.update(image.asarray(self.color_format))

        src_meta = axmeta[self.where]
        frame_width, frame_height = image.size

        # Get the first K bbox (the highest score ones)
        if isinstance(
            src_meta,
            (
                meta.ObjectDetectionMeta,
                meta.BottomUpKeypointDetectionMeta,
                meta.InstanceSegmentationMeta,
                meta.PoseInsSegMeta,
            ),
        ):
            boxes = src_meta.boxes.copy()
            indices = np.arange(len(boxes))
            if len(boxes) == 0:
                return image, result, axmeta

            if hasattr(src_meta, 'class_ids') and self._label_filter_ids:
                filtered = np.isin(src_meta.class_ids, [int(i) for i in self._label_filter_ids])
                boxes = boxes[filtered]
                indices = indices[filtered]

            if self.which == "CENTER":
                boxes, indices = _sort_boxes_by_distance(
                    boxes, indices, (frame_height, frame_width)
                )
            elif self.which == "AREA":
                boxes, indices = _sort_boxes_by_area(boxes, indices)
            elif self.which == "SCORE":
                if not hasattr(src_meta, 'scores'):
                    raise ValueError(f"Cannot find scores in {src_meta}")
                boxes, indices = _sort_boxes_by_score(boxes, indices, src_meta.scores)

            if self.top_k > 0:
                boxes = boxes[: self.top_k]
                indices = indices[: self.top_k]
        else:
            raise RuntimeError(f"{src_meta.__class__.__name__ } is not an ObjectDetectionMeta")

        boxes = boxes.astype(int)
        boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, frame_width)
        boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, frame_height)

        result = []
        for box, idx in zip(boxes, indices):
            x1, y1, x2, y2 = box
            # TODO: consider to filter out these boxes
            if x2 == x1 or y2 == y1:
                continue
            cropped_image = image.asarray()[y1:y2, x1:x2]
            roi_image = types.Image.fromarray(cropped_image, image.color_format)

            if self.image_processing_on_roi:
                for op in self.image_processing_on_roi:
                    try:
                        match = op.stream_check_match(stream_id)
                        if match:
                            # Check if the operator's exec_torch accepts metadata
                            sig = inspect.signature(op.exec_torch)
                            if len(sig.parameters) > 1 and 'meta' in sig.parameters:
                                roi_image = op.exec_torch(roi_image, axmeta)
                            else:
                                roi_image = op.exec_torch(roi_image)
                    except Exception as e:
                        raise ValueError(
                            f"Operator {op.__class__.__name__} failed to process ROI due to: {str(e)}"
                        )

            result.append(roi_image)
            axmeta[self.where].add_secondary_frame_index(self.task_name, idx)
        return image, result, axmeta


def _sort_boxes_by_distance(boxes, indices, img_shape):
    img_center = np.array(img_shape[:2][::-1]) / 2.0
    box_centers = (boxes[:, :2] + boxes[:, 2:]) / 2.0
    distances = np.linalg.norm(box_centers - img_center, axis=1)
    sorted_order = np.argsort(distances)
    sorted_boxes = boxes[sorted_order]
    sorted_indices = indices[sorted_order]
    return sorted_boxes, sorted_indices


def _sort_boxes_by_area(boxes, indices):
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    sorted_order = np.argsort(-areas)
    sorted_boxes = boxes[sorted_order]
    sorted_indices = indices[sorted_order]
    return sorted_boxes, sorted_indices


def _sort_boxes_by_score(boxes, indices, scores):
    sorted_order = np.argsort(-scores)
    sorted_boxes = boxes[sorted_order]
    sorted_indices = indices[sorted_order]
    return sorted_boxes, sorted_indices


@builtin
class InputWithImageProcessing(AxOperator):
    """Input operator with image processing. The selected operators should
    be able to accept types.Image and return types.Image for torch pipeline.

    Example:
        input:
            source: image_processing
            type: image
            color_format: RGB
            image_processing:
                - resize:
                    width: 1280
                    height: 720
                - other-operator:
    """

    type: str = 'image'
    color_format: types.ColorFormat = types.ColorFormat.RGB
    image_processing: list[PreprocessOperator]

    def _post_init(self):
        super()._post_init()
        self._enforce_member_type('color_format')

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        compiled_model_dir: Path | None,
        task_graph: graph.DependencyGraph,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, compiled_model_dir, task_graph
        )
        context.color_format = self.color_format

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        pass

    def _process_image(self, image: types.Image, stream_id: int, meta=None) -> types.Image:
        for op in self.image_processing:
            try:
                match = op.stream_check_match(stream_id)
                if match:
                    # Check if the operator's exec_torch accepts metadata
                    import inspect

                    sig = inspect.signature(op.exec_torch)
                    if len(sig.parameters) > 1 and 'meta' in sig.parameters and meta is not None:
                        image = op.exec_torch(image, meta)
                    else:
                        image = op.exec_torch(image)
            except Exception as e:
                raise ValueError(
                    f"Operator {op.__class__.__name__} failed to process types.Image due to: {str(e)}"
                )
        return image

    def exec_torch(self, image, result, meta, stream_id=0):
        if result is None and meta is None:
            image = _convert_image_to_types_image_for_deploy(image, self.color_format)
            return _deploy_cascade_model(image, self.color_format)

        if isinstance(image, types.Image):
            result = [self._process_image(image, stream_id, meta)]
        elif isinstance(image, list):
            new_images = []
            for im in image:
                if not isinstance(im, types.Image):
                    raise ValueError("Input must be a list of types.Image")
                new_images.append(self._process_image(im, stream_id, meta))
            result = new_images
        else:
            raise ValueError("Input must be an types.Image or a list of types.Image")

        if self.color_format != image.color_format:
            for im in result:
                new_im = im.asarray(self.color_format)
                im.update(new_im, color_format=self.color_format)
        if result is None and meta is None:
            LOG.trace("Return for deploying the cascade model")
            return result[0], None, None
        return result[0], result, meta

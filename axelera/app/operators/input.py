# Copyright Axelera AI, 2025
# A model pipeline must start from an Input Operator
# which return a list of Axelera Image and meta.
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from axelera import types

from .. import gst_builder, logging_utils, meta
from .base import AxOperator, PreprocessOperator, builtin
from .utils import insert_color_convert

if TYPE_CHECKING:
    from pathlib import Path

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
        where: str,
        compiled_model_dir: Path,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, where, compiled_model_dir
        )
        context.color_format = self.color_format
        context.imreader_backend = self.imreader_backend
        context.pipeline_input_color_format = self.color_format

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        if gst.new_inference:
            return
        vaapi = gst.getconfig() is not None and gst.getconfig().vaapi
        opencl = gst.getconfig() is not None and gst.getconfig().opencl
        insert_color_convert(gst, vaapi, opencl, f'{self.color_format.name.lower()}a')

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
    expand_margin: float = 0.0
    top_k: int = 1
    which: str = 'NONE'
    classes: list = []
    image_processing_on_roi: list[PreprocessOperator] = []
    color_format: types.ColorFormat = types.ColorFormat.RGB

    def _post_init(self):
        super()._post_init()
        self._enforce_member_type('color_format')
        SUPPORTED_WHICH = ['AREA', 'SCORE', 'CENTER', 'NONE']
        if self.which.upper() not in SUPPORTED_WHICH:
            raise ValueError(f"which is not in support list: {SUPPORTED_WHICH}")
        self.classes = self._parse_classes(self.classes)
        self.where = str(
            self.where
        )  # TODO SAM this is a hack to ensure it's a string not a yamlString

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
        # for cascade models, we know the input color format passed from the upstream model
        self._need_color_convert = context.color_format != self.color_format
        context.color_format = self.color_format
        assert self.where == where, f"where is not consistent: {self.where} vs {where}"

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        if self.which != 'NONE':
            gst.axinplace(
                lib='libinplace_filterdetections.so',
                options=f'meta_key:{self.where};which:{self.which};top_k:{self.top_k}',
            )

        gst.distributor(meta=str(self.where))
        gst.axtransform(
            lib='libtransform_roicrop.so',
            options=f'meta_key:{self.where}',
        )

        for op in self.image_processing_on_roi:
            op.build_gst(gst, stream_idx)

    def _parse_classes(self, class_value):
        """Parse the class value and return a list of class IDs.
        If the class value is a wildcard character, return an empty list."""
        if class_value == '*':
            return []

        if isinstance(class_value, str) and class_value.isdigit():
            return [int(class_value)]
        if isinstance(class_value, str):
            return [class_value]

        # Check if class value is a string with multiple elements separated by commas
        if isinstance(class_value, str) and ',' in class_value:
            classes = class_value.split(',')
            return [int(c) for c in classes]

        # Check if class value is a list of integers or strings
        if isinstance(class_value, list):
            if all(isinstance(c, int) for c in class_value):
                return class_value
            else:
                return [str(c) for c in class_value]
        raise ValueError('Unsupported format for class value')

    def exec_torch(self, image, result, axmeta, stream_id=0):
        if result is None and axmeta is None:
            image = _convert_image_to_types_image_for_deploy(image, self.color_format)
            return _deploy_cascade_model(image, self.color_format)

        if self.image_processing_on_roi:
            raise NotImplementedError("Please implement image_processing_on_roi in exec_torch")

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
            ),
        ):
            boxes = src_meta.boxes.copy()
            indices = np.arange(len(boxes))
            if len(boxes) == 0:
                return image, result, axmeta

            if hasattr(src_meta, 'class_ids') and self.classes:
                class_ids = src_meta.class_ids
                if isinstance(self.classes[0], str):
                    if src_meta.labels:
                        label_ids = [src_meta.labels.index(label) for label in self.classes]
                    else:
                        raise ValueError(f"Cannot find labels in {src_meta}")
                else:
                    label_ids = self.classes
                boxes, indices = _filter_boxes_by_label_id(boxes, class_ids, label_ids)

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
            top_det = boxes[: self.top_k]
            indices = indices[: self.top_k]
        else:
            raise RuntimeError(f"{src_meta.__class__.__name__ } is not an ObjectDetectionMeta")

        if self.expand_margin > 0:
            # Calculate the width and height of the bounding boxes
            box_width = top_det[:, 2] - top_det[:, 0]
            box_height = top_det[:, 3] - top_det[:, 1]
            # Expand the bounding box by expand_margin
            box_width *= 1 + self.expand_margin
            box_height *= 1 + self.expand_margin
            # Update the coordinates
            top_det[:, 0] -= box_width / 2
            top_det[:, 2] += box_width / 2
            top_det[:, 1] -= box_height / 2
            top_det[:, 3] += box_height / 2
        top_det = top_det.astype(int)
        top_det[:, [0, 2]] = np.clip(top_det[:, [0, 2]], 0, frame_width)
        top_det[:, [1, 3]] = np.clip(top_det[:, [1, 3]], 0, frame_height)

        result = []
        for box, idx in zip(top_det, indices):
            x1, y1, x2, y2 = box
            # TODO: consider to filter out these boxes
            if x2 == x1 or y2 == y1:
                continue
            cropped_image = image.asarray()[y1:y2, x1:x2]
            result.append(types.Image.fromarray(cropped_image, image.color_format))
            axmeta[self.where].add_secondary_frame_index(self.task_name, idx)
        return image, result, axmeta


def _filter_boxes_by_label_id(boxes, class_ids, label_ids):
    filtered_indices = np.isin(class_ids, label_ids)
    filtered_boxes = boxes[filtered_indices]
    return filtered_boxes, filtered_indices


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
        where: str,
        compiled_model_dir: Path,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, where, compiled_model_dir
        )

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        for op in self.image_processing:
            match = op.stream_check_match(int(stream_idx))
            if match:
                op.build_gst(gst, stream_idx)

    def _process_image(self, image: types.Image, stream_id: int) -> types.Image:
        for op in self.image_processing:
            try:
                match = op.stream_check_match(stream_id)
                if match:
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
            result = [self._process_image(image, stream_id)]
        elif isinstance(image, list):
            new_images = []
            for im in image:
                if not isinstance(im, types.Image):
                    raise ValueError("Input must be a list of types.Image")
                new_images.append(self._process_image(im, stream_id))
            result = new_images
        else:
            raise ValueError("Input must be an types.Image or a list of types.Image")

        if self.color_format != image.color_format:
            for im in result:
                new_im = im.asarray(self.color_format)
                im.update(new_im, color_format=self.color_format)
        if result is None and meta is None:
            LOG.trace(f"Return for deploying the cascade model")
            return result[0], None, None
        return result[0], result, meta

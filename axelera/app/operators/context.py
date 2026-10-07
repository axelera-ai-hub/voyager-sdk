# Copyright Axelera AI, 2024
# Contextual information for the pipeline operations

import dataclasses

from axelera import types


@dataclasses.dataclass
class PipelineContext:
    """
    Contextual information for the pipeline operations along with additional properties for
    measurement.

    color_format: the current color format; it starts from color format of input operator
      and will be changed by color format conversion operators
    resize_status: the current resize status; it starts from original status of input operator
      and will be changed by resize or letterbox operators. We use this status to determine how
      to decode the inference results to the original image size
    submodels_with_boxes_from_tracker: a set of submodels for which the tracker created new
      hidden metadata with boxes, which has been defined via the callback mechanism.
    association: the name of the metadata used to create the ROIs for the secondary model of a
      cascaded pipeline. Not equal to the master meta when applying a filter or a tracker.
      This association metadata contains boxes with ids which either represent the indices or
      the track_ids of those boxes in the master meta.

    """

    color_format: types.ColorFormat = types.ColorFormat.RGB
    resize_status: types.ResizeMode = types.ResizeMode.ORIGINAL
    submodels_with_boxes_from_tracker: set = dataclasses.field(default_factory=set)
    association: str = str()

    def __post_init__(self):
        self.color_format = types.ColorFormat.parse(self.color_format)
        self.resize_status = types.ResizeMode.parse(self.resize_status)
        self._pipeline_input_color_format = types.ColorFormat.RGB
        self._imreader_backend = types.ImageReader.PIL

    @property
    def pipeline_input_color_format(self) -> types.ColorFormat:
        return self._pipeline_input_color_format

    @pipeline_input_color_format.setter
    def pipeline_input_color_format(self, color_format: types.ColorFormat):
        self._pipeline_input_color_format = color_format

    @property
    def imreader_backend(self) -> types.ImageReader:
        # preferred image reader for measurement
        return self._imreader_backend

    @imreader_backend.setter
    def imreader_backend(self, backend: types.ImageReader):
        self._imreader_backend = backend

    def propagate(self) -> 'PipelineContext':
        """Create a deep copy of the PipelineContext object.
        We don't want to propagate the resize status and the color format to the next task, as the input image is the original image.
        """
        new_context = PipelineContext(
            color_format=self._pipeline_input_color_format,
            resize_status=types.ResizeMode.ORIGINAL,
        )
        new_context._pipeline_input_color_format = self._pipeline_input_color_format
        new_context._imreader_backend = self._imreader_backend
        new_context.submodels_with_boxes_from_tracker = self.submodels_with_boxes_from_tracker
        return new_context

    def update(self, other: 'PipelineContext') -> None:
        """Update this PipelineContext with values from another PipelineContext."""
        self.color_format = other.color_format
        self.resize_status = other.resize_status
        self._pipeline_input_color_format = other._pipeline_input_color_format
        self._imreader_backend = other._imreader_backend
        self.submodels_with_boxes_from_tracker = other.submodels_with_boxes_from_tracker

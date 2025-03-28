# Copyright 2024 Axelera AI, Inc.
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

    """

    color_format: types.ColorFormat = types.ColorFormat.RGB
    resize_status: types.ResizeMode = types.ResizeMode.ORIGINAL

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
        return new_context

    def update(self, other: 'PipelineContext') -> None:
        """Update this PipelineContext with values from another PipelineContext."""
        self.color_format = other.color_format
        self.resize_status = other.resize_status
        self._pipeline_input_color_format = other._pipeline_input_color_format
        self._imreader_backend = other._imreader_backend

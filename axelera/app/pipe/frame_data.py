# Copyright Axelera AI, 2024
# Basic data structure for pipeline output data
from __future__ import annotations

import dataclasses
import enum
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    import numpy as np

    from axelera import types

    from ..meta import AxMeta


@dataclasses.dataclass
class FrameResult:
    image: Optional[types.Image] = None
    tensor: Optional[np.ndarray] = None
    meta: Optional[AxMeta] = None
    stream_id: int = 0
    src_timestamp: int = 0
    sink_timestamp: int = 0
    inferences: int = 0
    render_timestamp: int = 0

    @property
    def source_id(self) -> int:
        # an an alias for stream_id
        return self.stream_id

    def __getattr__(self, attr):
        try:
            return self.meta[attr].objects
        except KeyError:
            raise AttributeError(f"'FrameResult' object has no attribute '{attr}'") from None


class FrameEventType(enum.Enum):
    result = enum.auto()
    '''Occurs when a new result is available from the pipeline.'''

    source_error = enum.auto()
    '''Occurs when there is an error with a source, such as when an RTSP source connection fails.'''

    end_of_source = enum.auto()
    '''Occurs when a source has finished sending frames.'''

    end_of_pipeline = enum.auto()
    '''Occurs when a pipeline has finished processing all frames.'''


@dataclasses.dataclass
class FrameEvent:
    type: FrameEventType = FrameEventType.result

    source_id: int = 0
    '''The source for which this event applies.'''

    message: str = ''

    result: FrameResult | None = None
    '''An optional FrameResult associated with this event, for example in the case of a result event.'''

    @classmethod
    def from_result(cls, result: FrameResult) -> FrameEvent:
        return cls(FrameEventType.result, result.source_id, result=result)

    @classmethod
    def from_end_of_source(cls, source_id: int, message: str) -> FrameEvent:
        return cls(FrameEventType.end_of_source, source_id, message=message)

    @classmethod
    def from_source_error(cls, source_id: int, message: str) -> FrameEvent:
        return cls(FrameEventType.source_error, source_id, message=message)

    @classmethod
    def from_end_of_pipeline(cls, source_id: int, message: str) -> FrameEvent:
        return cls(FrameEventType.end_of_pipeline, source_id, message=message)

    def __repr__(self):
        if self.type == FrameEventType.result:
            return f"FrameEvent({self.type.name}, source_id={self.source_id}, result=FrameResult(...))"
        else:
            return f"FrameEvent({self.type.name}, source_id={self.source_id}, message='{self.message}')"

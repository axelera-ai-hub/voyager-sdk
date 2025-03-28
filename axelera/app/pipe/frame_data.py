# Copyright Axelera AI, 2024
# Basic data structure for pipeline output data
from __future__ import annotations

import dataclasses
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
    render_timestamp: int = 0

    def __getattr__(self, attr):
        try:
            return self.meta[attr].objects
        except KeyError:
            raise AttributeError(f"'FrameResult' object has no attribute '{attr}'")

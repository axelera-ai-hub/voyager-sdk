# Copyright Axelera AI, 2025

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Union

import numpy as np

from axelera.app import eval_interfaces

from .. import display
from .base import AxTaskMeta


@dataclass(frozen=True)
class ImageMeta(AxTaskMeta):
    img: np.ndarray

    def to_evaluation(self):
        return eval_interfaces.ImageSample(img=self.img)

    def draw(self, draw: display.Draw):
        draw.draw_image(self.img)

    @classmethod
    def decode(cls, data: Dict[str, Union[bytes, bytearray]]) -> ImageMeta:
        is_float = bool(data.get('float_datatype', b'\x00')[0])
        width = int.from_bytes(data.get('width'), byteorder='little')
        height = int.from_bytes(data.get('height'), byteorder='little')
        channels = int.from_bytes(data.get('channels'), byteorder='little')
        dtype = np.float32 if is_float else np.uint8
        # Layout is CHW (matching the NCHW tensor byte order from C++ decoder)
        depth = np.frombuffer(data.get('data'), dtype=dtype).reshape(channels, height, width)
        return cls(img=depth)

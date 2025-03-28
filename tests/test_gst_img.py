# Copyright Axelera AI, 2024
import ctypes

import PIL.Image
import gi
import numpy as np

from axelera import types

gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
gi.require_version('GstApp', '1.0')  # for try_pull_sample

from gi.repository import Gst, GstVideo
import pytest


def create_image_from_array(width, height, format, data=None):
    if data:
        return types.Image.fromarray(
            np.array(data, dtype=np.uint8).reshape((height, width, len(format)))
        )
    return types.Image.fromarray(np.zeros((height, width, len(format)), dtype=np.uint8))


def create_image_from_PIL(width, height, format, data=None):
    image = create_image_from_array(width, height, format, data)
    pil = PIL.Image.fromarray(image.asarray())
    return types.Image.frompil(pil)


def create_image_from_gst(width, height, format, data=None):
    image = create_image_from_array(width, height, format, data)
    w, h = image.size
    frame = image.asarray().tobytes()
    num_channels = len(format)
    in_info = GstVideo.VideoInfo()
    fmt = GstVideo.VideoFormat.RGBA if format == "RGBA" else GstVideo.VideoFormat.RGB
    in_info.set_format(fmt, w, h)
    in_info.fps_n = 120
    in_info.fps_d = 1
    caps = in_info.to_caps()
    if in_info.stride[0] == w * num_channels:
        buffer = Gst.Buffer.new_wrapped(frame)
    else:
        buffer = Gst.Buffer.new_allocate(None, in_info.size, None)
        for i in range(h):
            frame_offset = i * w * num_channels
            buffer_offset = in_info.offset[0] + i * in_info.stride[0]
            buffer.fill(buffer_offset, frame[frame_offset : frame_offset + w * num_channels])

    return types.Image.fromgst(Gst.Sample.new(buffer, caps))


def create_image_from_type(type, width, height, format, data=None):
    if type == "array":
        return create_image_from_array(width, height, format, data)
    elif type == "PIL":
        return create_image_from_PIL(width, height, format, data)
    elif type == "gst":
        return create_image_from_gst(width, height, format, data)
    else:
        raise ValueError(f"Unknown type: {type}")


@pytest.mark.parametrize(
    'width, height, format, type, expected_stride, expected_pixel_stride',
    [
        (10, 10, "RGB", "array", 30, 3),
        (10, 10, "RGBA", "array", 40, 4),
        (10, 10, "RGB", "PIL", 30, 3),
        (10, 10, "RGBA", "PIL", 40, 4),
        (10, 10, "RGB", "gst", 32, 3),
        (10, 10, "RGBA", "gst", 40, 4),
    ],
)
def test_strides(width, height, format, type, expected_stride, expected_pixel_stride):
    img = create_image_from_type(type, width, height, format)
    assert img.pitch == expected_stride
    assert img.pixel_stride == expected_pixel_stride
    assert img.size == (width, height)


@pytest.mark.parametrize(
    'type, expected_size',
    [
        ('array', 16),
        ('PIL', 16),
        ('gst', 16),
    ],
)
def test_as_c_void_p(type, expected_size):
    array = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]
    img = create_image_from_type(type, 2, 2, "RGBA", data=array)
    size = img.pitch * img.height
    assert img.size == (2, 2)
    assert size == expected_size
    with img.as_c_void_p() as ptr:
        b = ctypes.cast(ptr, ctypes.POINTER(ctypes.c_ubyte * size)).contents
        assert bytes(b) == bytes(array)

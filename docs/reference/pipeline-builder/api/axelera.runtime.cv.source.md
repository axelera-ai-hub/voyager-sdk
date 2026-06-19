---
title: "axelera.runtime.cv.source"
---
# `axelera.runtime.cv.source`


Video streaming implementation for zero-copy frame decoding.

## Summary

| Name | Description |
|------|-------------|
| [Image](#image) | A single image frame backed by a numpy array, PIL image, GStreamer sample, or VideoBuffer. |
| [Source](#source) | Abstract base for frame sources: iterable of `Image` frames with `fps` and `frame_count` metadata. |
| [VideoSource](#videosource) | Video source stream wrapper that provides both frame iteration and metadata access. |
| [create_source](#create_source) | Create a video source that yields decoded frames with metadata access. |

---

### Image

A single image frame backed by a numpy array, PIL image, GStreamer sample, or VideoBuffer.

Construct one with the `from_array`, `from_pil`, `from_gst`, `from_any`, or
`from_videobuffer` class methods rather than calling the constructor directly.

**Properties:**

- **`source`** (`str`) — Source identifier for the image, e.g. filename or stream name.
- **`shape`** (`tuple[int, int, int]`) — Shape of the image in pixels as (height, width, channels).
- **`presentation_timestamp`** (`int | None`) — Presentation timestamp of the image in us.
- **`ndim`** (`int`) — Number of dimensions (always 3: height, width, channels).
- **`nbytes`** (`int`) — Total size of the image buffer in bytes.
- **`width`** (`int`) — Width of the image in pixels.
- **`height`** (`int`) — Height of the image in pixels.
- **`pitch`** (`int`) — Stride of the image in bytes.
- **`pixel_stride`** (`int`) — Pixel stride of the image in bytes.
- **`strides`** (`tuple[int, int, int]`) — Strides of the image array in bytes.
- **`offsets`** (`tuple[int, ...]`) — Offsets of the image buffer in bytes.
- **`color_format`** (`ColorFormat`) — Color format of the image.
- **`has_pil`** (`bool`) — True if the image contains a PIL image.
- **`has_array`** (`bool`) — True if the image contains a numpy array.
- **`has_gst`** (`bool`) — True if the image contains a Gst.Sample.
- **`has_videobuffer`** (`bool`) — True if the image contains a VideoBuffer.

**Methods:**

#### classmethod from_videobuffer

```python
from_videobuffer(buffer: _videodecoder.VideoBuffer)
```

Construct an Image from a VideoBuffer.

#### classmethod from_array

```python
from_array(array: ImageArray, fallback_color_format: ColorFormat | str | None = None, source: str = '', like: Image | None = None)
```

Construct an Image from a numpy array or torch tensor.

If `like` is provided, the new Image will inherit metadata (excluding color_format) from
it. An explicit `source` parameter takes precedence over `like.source`.

The color_format is inferred from the shape of the incoming array. For example (H, W, 1)
or (H, W) is interpreted as ColorFormat.GRAY. If the shape is (H, W, 3|4)
then the color format cannot be precisely inferred and color format is ColorFormat.PACKED
unless fallback_color_format is provided.

If no format can be inferred and fallback_color_format is not provided, a ValueError is
raised.

PACKED represents some packed pixel format (i.e. not planar) but the interpretation is not
exactly known (for example BGR vs RGB). In general it does not prevent most properties
such as width/height being used or many operations such as transpose etc, but if
a request is made that requires knowledge of the exact format then a ValueError is raised.
For example a conversion to RGB cannot be performed if the source format is not known.

#### classmethod from_pil

```python
from_pil(pil: PILImage, source: str = '', like: Image | None = None)
```

Construct an Image from a PIL Image.

If `like` is provided, the new Image will inherit metadata from it. An explicit
`source` parameter takes precedence over `like.source`.

#### classmethod from_gst

```python
from_gst(sample: Gst.Sample, source: str = '', like: Image | None = None)
```

Construct an Image from a GST video buffer.

If `like` is provided, the new Image will inherit metadata from it. An explicit
`source` parameter takes precedence over `like.source`.

#### classmethod from_any

```python
from_any(i: Any, fallback_color_format: ColorFormat | str | None = None, source: str = '', like: Image | None = None)
```

Construct an Image from a PIL Image, a numpy or torch tensor, Gst buffer, or an Image.

If the passed image is not one of the types above a TypeError is raised.

If `like` is provided, the new Image will inherit metadata from it. An explicit
`source` parameter takes precedence over `like.source`.

`fallback_color_format` is used only when passing a numpy or torch tensor, when the input
color format cannot be inferred automatically.  See `from_array` for more details.

#### to_numpy

```python
to_numpy() -> ImageArray
```

Return a read-only zero-copy numpy view of the image.

For GStreamer-backed images, maps the buffer via ctypes without copying;
the mapping is released when the returned array (and anything referencing
its memory) is garbage collected, via a `weakref.finalize` callback.

For VideoBuffer-backed images, calls buffer.to_numpy() for zero-copy access.

For numpy-backed or PIL-backed images, returns the existing array directly.

#### convert

```python
convert(color: ColorFormat | str) -> Image
```

Return a new Image whose backing array is in the requested color format.

If the image is already in `color`, returns `self` (no copy).

Supported conversions (rows = source, columns = destination, `Y` =
supported, `-` = not supported). Use `is_convert_available` to query
this programmatically; it is the source of truth as availability
ultimately depends on the installed OpenCV build::

            RGB BGR RGBA BGRA GRAY I420 NV12 NV16 YUY2 PACKED
    RGB      Y   Y   Y    Y    Y    -    -    -    -     -
    BGR      Y   Y   Y    Y    Y    -    -    -    -     -
    RGBA     Y   Y   Y    Y    Y    -    -    -    -     -
    BGRA     Y   Y   Y    Y    Y    -    -    -    -     -
    GRAY     Y   Y   Y    Y    Y    -    -    -    -     -
    I420     Y   Y   Y    Y    Y    Y    -    -    -     -
    NV12     Y   Y   Y    Y    Y    -    Y    -    -     -
    NV16     Y   Y   Y    Y    Y    -    -    Y    Y     -
    YUY2     Y   Y   Y    Y    Y    -    -    -    Y     -
    PACKED   -   -   -    -    -    -    -    -    -     Y

In words:

- The diagonal (same format) is always a no-op that returns `self`.
- RGB, BGR, RGBA, BGRA and GRAY are fully interconvertible.
- YUV formats (I420, NV12, NV16, YUY2) decode to the RGB family or GRAY,
  but encoding *to* a YUV format, and converting between different YUV
  formats, is not supported (except NV16 -> YUY2, used internally).
- PACKED means the exact packed layout (e.g. RGB vs BGR) is unknown, so it
  cannot be converted to or from any explicit format. Construct the image
  with the real format first (e.g. `fallback_color_format='BGR'`).

**Args:**

- **color**: Target color format.

**Returns:**

`Image` -- An `Image` in the requested color format. Use `to_numpy()` to obtain
`Image` -- the underlying numpy array.

**Raises:**

- **ValueError**: if the source format is PACKED, or the conversion is not supported (see the table above).

---

### Source

Abstract base for frame sources: iterable of `Image` frames with `fps` and `frame_count` metadata.

**Properties:**

- **`fps`** (`float`)
- **`frame_count`** (`int`)

---

### VideoSource

Video source stream wrapper that provides both frame iteration and metadata access.

This class wraps a video stream generator and provides access to video metadata
like FPS, frame count, and timestamps while also allowing iteration over frames.

**Properties:**

- **`fps`** (`float`) — Get the video's native frame rate (FPS).
- **`frame_count`** (`int`) — Get the total number of frames in the video.

**Methods:**

#### close

```python
close()
```

Close the video source and clean up resources.

This method closes the underlying generator, which triggers cleanup
of the decoder and frame queue resources, and waits for cleanup to complete.
Safe to call multiple times.

---

### create_source

```python
create_source(input_path: str, event_callback: Callable[[int, str], None] | None = None, buffer_size: int = 30, backend: Literal['ffmpeg', 'opencv'] = 'ffmpeg', live_source: bool | None = None) -> VideoSource
```

Create a video source that yields decoded frames with metadata access.

This function uses FFMpegVideoDecoder by default, or can use the opencv based
OpenCVVideoDecoder for video decoding. Uses zero-copy VideoBuffer callbacks
for efficient frame transfer.

**Args:**

- **input_path**: Path to the video file or stream URL
- **event_callback**: Optional callback function for handling decoder events.             Receives (error_code: int, error_message: str).             Only used with FFmpeg backend.
- **buffer_size**: Size of the internal frame buffer queue (default: 30).          Larger values prevent frame drops but use more memory.
- **backend**: Video decoder backend to use - "ffmpeg" (default) or "opencv".      "ffmpeg" uses FFMpegVideoDecoder for better performance and format support,      while "opencv" uses OpenCVVideoDecoder for maximum compatibility.
- **live_source**: Override input-kind classification. By default,          `input_path` is auto-classified as a live source if it          uses an RTSP/RTP/RTMP/SRT/UDP/MJPEG URI or starts with          `/dev/video`; everything else (incl. http(s)://) is          treated as a file. Live sources drop frames on full          queue to stay real-time; files block. Pass          `live_source=True` for HTTP live streams (HLS,          MJPEG-over-HTTP) or `live_source=False` to force          blocking semantics.

**Returns:** `VideoSource` -- A video source wrapper that yields decoded frames

**Raises:**

- **RuntimeError**: If video decoder fails to start or decode

**Examples:**

```python
>>> # Use as context manager for automatic cleanup
>>> with create_source("video.mp4") as source:
...     for frame in source:
...         print(frame.to_numpy().shape)
```
```python
>>> # Get frames as Image objects
>>> source = create_source("video.mp4")
>>> for frame in source:
...     print(type(frame))  # <class 'axelera.runtime2.img.Image'>
...     print(frame.shape)  # (H, W, 3)
```

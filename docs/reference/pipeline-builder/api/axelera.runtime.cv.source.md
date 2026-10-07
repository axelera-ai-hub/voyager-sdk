---
title: "axelera.runtime.cv.source"
---
# `axelera.runtime.cv.source`


Video streaming implementation for zero-copy frame decoding.

## Summary

| Name | Description |
|------|-------------|
| [Image](#image) | A single image frame backed by a numpy array, PIL image, GStreamer sample, or VideoBuffer. |
| [Source](#source) | ABC for all sources: iterable of `Image` frames with `fps` and `frame_count` metadata. |
| [VideoSource](#videosource) | Video source stream wrapper that provides both frame iteration and metadata access. |
| [create_source](#create_source) | Create a video source that yields decoded frames with metadata access. |
| [frame_count](#frame_count) | How many frames `source` holds, or None when it cannot say. |
| [get_media_files](#get_media_files) | Get all media files (video/image) recursively from a directory, checking if images-only. |
| [is_single_image](#is_single_image) | Whether `source` is a single still image, best-effort (see `frame_count`). |
| [iter_results](#iter_results) | Yield `(image, objects)` for each frame of `source`. |

---

### Image

A single image frame backed by a numpy array, PIL image, GStreamer sample, or VideoBuffer.

Construct one with the `from_array`, `from_pil`, `from_gst`, `from_any`, or
`from_videobuffer` class methods rather than calling the constructor directly.

**Properties:**

- **`source`** (`str`) — Source identifier for the image, e.g. filename or stream name.
- **`shape`** (`tuple[int, int, int]`) — Shape of the image in pixels as (height, width, channels).
- **`presentation_timestamp`** (`int | None`) — Presentation timestamp of the image in us.
- **`decoded_timestamp`** (`int | None`) — Wall-clock time decoding of this frame completed.
- **`color_range`** (`int`) — Color range from the source stream (GstRange int): 0=unknown, 1=limited, 2=full.
- **`color_matrix`** (`int`) — Color matrix from the source stream (GstMatrix int): 0=unknown, 3=BT.601, 4=BT.709, etc.
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
from_videobuffer(buffer: _videodecoder.VideoBuffer, like: Image | None = None)
```

Construct an Image from a VideoBuffer.

If *like* is provided its source, presentation_timestamp, and decoded_timestamp
are copied to the new Image (useful when the buffer is a processed version of
an existing frame and the caller wants to preserve frame provenance).

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

#### classmethod is_supported

```python
is_supported(i: Any) -> bool
```

Return True if `from_any` accepts `i`, without constructing an Image.

For a caller that discards the result unless some condition holds, this checks the
type up front without paying for a conversion that is then thrown away.

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

ABC for all sources: iterable of `Image` frames with `fps` and `frame_count` metadata.

**Properties:**

- **`fps`** (`float`)
- **`frame_count`** (`int`)

---

### VideoSource

Video source stream wrapper that provides both frame iteration and metadata access.

This class wraps a video stream generator and provides access to video metadata
like FPS, frame count, and timestamps while also allowing iteration over frames.
`create_source` builds it; you do not construct one directly.

**Constructor:**

```python
__init__(generator: Generator[Image, None, None], decoder=None, cleanup_done: threading.Event | None = None, stop_requested: threading.Event | None = None, decode_finished: threading.Event | None = None, frame_queue: queue.Queue | None = None, frame_count: int | None = None, dropped: list[int] | None = None, live_source: bool = False, classified_live: bool = False)
```

**Properties:**

- **`fps`** (`float`) — Get the video's native frame rate (FPS).
- **`frame_count`** (`int`) — Get the total number of frames in the video.
- **`dropped`** (`int`) — Frames dropped unseen because the consumer could not keep up.
- **`is_live`** (`bool`) — Whether this source serves the freshest frame on a full queue.
- **`may_drop`** (`bool`) — Whether this source can lose frames at all.
- **`buffer_size`** (`int`) — Queue depth this source was given, or 0 if it has no queue of its own.

**Methods:**

#### queue_depth

```python
queue_depth() -> int | None
```

Return decoded frames currently buffered, or None if this source has no queue.

A live snapshot of the decode queue's `qsize()`, for callers that want to
sample how full the decode buffer is (e.g. a consumer profiling whether it
is source-starved or back-pressured). Directory sources decode nothing and
have no queue.

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
create_source(input_path: str, event_callback: Callable[[int, str], None] | None = None, buffer_size: int | None = None, backend: Literal['ffmpeg', 'opencv'] = 'ffmpeg', live_source: bool | None = None) -> VideoSource
```

Create a video source that yields decoded frames with metadata access.

This function uses FFMpegVideoDecoder by default, or can use the opencv based
OpenCVVideoDecoder for video decoding. Uses zero-copy VideoBuffer callbacks
for efficient frame transfer.

**Args:**

- **input_path**: Path to the video file or stream URL. A trailing `@fps` or         `@auto` paces a file as if it were a live camera of that frame         rate (`@auto` uses the source's native rate), so end-to-end         latency reflects real-time arrival rather than decoder read-ahead.         Example: `"clip.mp4@auto"` or `"clip.mp4@30"`.
- **event_callback**: Optional callback function for handling decoder events.             Receives (error_code: int, error_message: str).             Only used with FFmpeg backend.
- **buffer_size**: Depth of the internal frame buffer queue. Defaults to 30 for a          file (read-ahead, no frame lost) and 2 for a live source, where          the depth is a latency floor rather than a safety margin: a          consumer that falls a frame behind a camera trails it by the          whole depth until the run ends. Raise it for a live source only          to trade latency for tolerance of consumer jitter.
- **backend**: Video decoder backend to use - "ffmpeg" (default) or "opencv".      "ffmpeg" uses FFMpegVideoDecoder for better performance and format support,      while "opencv" uses OpenCVVideoDecoder for maximum compatibility.
- **live_source**: Override input-kind classification, which selects the queue          policy. By default, `input_path` is auto-classified as a live          source if it uses an RTSP/RTP/RTMP/SRT/UDP/MJPEG URI or starts          with `/dev/video`; everything else (incl. http(s)://) is          treated as a file. A live source is freshest-wins -- an          arriving frame evicts the stale ones rather than being          dropped itself, so the consumer never accumulates a backlog          and trails the newest decoded frame by at most          `buffer_size - 1` (one frame at the default depth of 2) --          while a file blocks the decoder (back-pressure) to deliver          every frame. Pass `live_source=True` for HTTP live streams          (HLS, MJPEG-over-HTTP), or `live_source=False` to force          blocking semantics: every frame is then delivered in order,          but on an input that auto-classifies as live the decoder's          wait for room is capped at one second (after which the frame          is dropped and counted in `VideoSource.dropped`), because a          camera cannot be asked to slow down and a parked reader only          moves the same loss into the socket buffer, uncounted.

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
...     print(type(frame).__name__)  # Image
...     print(frame.shape)  # (H, W, 3)
```

---

### frame_count

```python
frame_count(source: Source) -> int | None
```

How many frames `source` holds, or None when it cannot say.

Best-effort: a live source (camera, RTSP) has no count, and a decoder may not
expose one for a given container. Callers get None rather than a fabricated
number, so a progress bar shows elapsed-only instead of counting to a wrong
total. A non-positive or non-integer count is normalized to None.

---

### get_media_files

```python
get_media_files(directory: str | pathlib.Path) -> tuple[list[str], bool]
```

Get all media files (video/image) recursively from a directory, checking if images-only.

**Args:**

- **directory**: Path to directory containing media files

**Returns:** `tuple[list[str], bool]` -- A tuple containing: - List of absolute paths to media files, sorted alphabetically - True if all discovered media files are images, otherwise False

---

### is_single_image

```python
is_single_image(source: Source) -> bool
```

Whether `source` is a single still image, best-effort (see `frame_count`).

---

### iter_results

```python
iter_results(pipeline, source: str | Source, *, max_frames: int | None = None) -> Iterator[tuple[Image, object]]
```

Yield `(image, objects)` for each frame of `source`.

Streams `source` -- a path (single image, video, or stream URL) or an already
open :class:`Source` -- through the built `pipeline` (pipelined across cores) and
yields each frame's image alongside its result objects, so a caller can render
both (each object draws its own overlay via `.draw()`). The image is whatever
the pipeline emits for that frame -- typically the decoder-native buffer, so its
colour format follows the source rather than being normalized here. `max_frames`,
when set, caps how many frames are yielded, and stops before decoding the one
after.

A path is opened here and closed deterministically on exit; an open `Source` is
left to its owner, so a caller that needs the source's metadata (frame count,
fps) can open it once and share it rather than probing it a second time.
The stream is drained on early exit either way, so in-flight futures unwind now
rather than at GC.

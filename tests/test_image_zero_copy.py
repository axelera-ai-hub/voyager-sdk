# Copyright Axelera AI, 2026
"""Integration tests for Image.as_ndarray_view() with real GStreamer.

These tests validate zero-copy buffer access works correctly with real
GStreamer buffers (not mocks). They are skipped when GStreamer is not
available (e.g., in tox CI without containerless).

Run locally: source containerless.sh && python3 -m pytest tests/test_image_zero_copy.py -v
"""
import time

import numpy as np
import pytest

try:
    import gi

    gi.require_version('Gst', '1.0')
    gi.require_version('GstVideo', '1.0')
    from gi.repository import Gst

    Gst.init(None)
    HAS_GST = True
except (ImportError, ValueError):
    HAS_GST = False

pytestmark = pytest.mark.skipif(not HAS_GST, reason='GStreamer not available')


def _has_as_ndarray_view():
    from axelera.types import Image

    return hasattr(Image, 'as_ndarray_view')


skip_no_method = pytest.mark.skipif(
    not _has_as_ndarray_view() if HAS_GST else True,
    reason='Image.as_ndarray_view() not available in installed axelera-types',
)


def _make_sample(width, height, fmt='RGB'):
    """Create a real GStreamer sample with a deterministic pixel pattern.

    Pattern: pixel[row, col] = (row % 256, col % 256, 42[, 255])
    """
    channels = 4 if 'A' in fmt else 3
    frame = np.zeros((height, width, channels), dtype=np.uint8)
    for r in range(height):
        frame[r, :, 0] = r % 256
    for c in range(width):
        frame[:, c, 1] = c % 256
    frame[:, :, 2] = 42
    if channels == 4:
        frame[:, :, 3] = 255

    data = frame.tobytes()
    buf = Gst.Buffer.new_allocate(None, len(data), None)
    buf.fill(0, data)
    caps = Gst.Caps.from_string(
        f'video/x-raw,format={fmt},width={width},height={height},framerate=30/1'
    )
    return Gst.Sample.new(buf, caps, None, None), frame


@skip_no_method
class TestAsNdarrayViewRealGst:
    def test_rgb_correctness(self):
        """as_ndarray_view returns correct pixel values for RGB."""
        from axelera.types import Image

        sample, expected = _make_sample(200, 100, 'RGB')
        img = Image.fromgst(sample)
        with img.as_ndarray_view() as view:
            assert view.shape == (100, 200, 3)
            np.testing.assert_array_equal(view, expected)

    def test_rgba_correctness(self):
        """as_ndarray_view returns correct pixel values for RGBA."""
        from axelera.types import Image

        sample, expected = _make_sample(100, 50, 'RGBA')
        img = Image.fromgst(sample)
        with img.as_ndarray_view() as view:
            assert view.shape == (50, 100, 4)
            np.testing.assert_array_equal(view, expected)

    def test_view_is_readonly(self):
        """GStreamer-backed view must be read-only."""
        from axelera.types import Image

        sample, _ = _make_sample(100, 50, 'RGB')
        img = Image.fromgst(sample)
        with img.as_ndarray_view() as view:
            assert not view.flags.writeable
            with pytest.raises((ValueError, TypeError)):
                view[0, 0, 0] = 255

    def test_roi_crop_correctness(self):
        """Slicing a small ROI returns correct data."""
        from axelera.types import Image

        sample, expected = _make_sample(1920, 1080, 'RGB')
        img = Image.fromgst(sample)
        with img.as_ndarray_view() as view:
            roi = view[100:200, 300:400]
            assert roi.shape == (100, 100, 3)
            np.testing.assert_array_equal(roi, expected[100:200, 300:400])
            roi_copy = roi.copy()
            assert roi_copy.flags.writeable

    def test_matches_asarray(self):
        """as_ndarray_view produces identical pixels to asarray."""
        from axelera.types import Image

        sample1, _ = _make_sample(640, 480, 'RGB')
        img1 = Image.fromgst(sample1)
        with img1.as_ndarray_view() as view:
            view_copy = view.copy()

        sample2, _ = _make_sample(640, 480, 'RGB')
        img2 = Image.fromgst(sample2)
        arr = img2.asarray()

        np.testing.assert_array_equal(view_copy, arr)

    def test_zero_copy_faster_than_asarray(self):
        """as_ndarray_view + ROI is significantly faster than asarray + ROI.

        At 1080p, zero-copy should be at least 2x faster (typically 4-10x).
        This confirms the ctypes path avoids the full-buffer copy.
        """
        from axelera.types import Image

        N = 30
        w, h = 1920, 1080

        samples_zc = [_make_sample(w, h, 'RGB')[0] for _ in range(N)]
        t0 = time.perf_counter()
        for s in samples_zc:
            img = Image.fromgst(s)
            with img.as_ndarray_view() as v:
                v[100:300, 200:400].copy()
        t_zc = time.perf_counter() - t0

        samples_aa = [_make_sample(w, h, 'RGB')[0] for _ in range(N)]
        t0 = time.perf_counter()
        for s in samples_aa:
            img = Image.fromgst(s)
            img.asarray()[100:300, 200:400].copy()
        t_aa = time.perf_counter() - t0

        speedup = t_aa / t_zc
        assert speedup > 2.0, (
            f'Expected at least 2x speedup, got {speedup:.1f}x '
            f'(zero-copy={t_zc / N * 1000:.2f}ms, asarray={t_aa / N * 1000:.2f}ms)'
        )


@skip_no_method
class TestInputFromROIZeroCopy:
    """End-to-end test: InputFromROI.exec_torch() with as_ndarray_view()."""

    def _make_image_and_meta(self, use_gst):
        from axelera.app.meta import AxMeta, ObjectDetectionMeta
        from axelera.types import Image, ColorFormat

        img_array = np.zeros((480, 640, 3), dtype=np.uint8)
        for r in range(480):
            img_array[r, :, 0] = r % 256
        for c in range(640):
            img_array[:, c, 1] = c % 256
        img_array[:, :, 2] = 42

        if use_gst:
            data = img_array.tobytes()
            buf = Gst.Buffer.new_allocate(None, len(data), None)
            buf.fill(0, data)
            caps = Gst.Caps.from_string(
                'video/x-raw,format=RGB,width=640,height=480,framerate=30/1'
            )
            img = Image.fromgst(Gst.Sample.new(buf, caps, None, None))
        else:
            img = Image.fromarray(img_array, ColorFormat.RGB)

        boxes = np.array([[100, 50, 200, 150], [300, 200, 450, 350]], dtype=np.float32)
        scores = np.array([0.9, 0.85], dtype=np.float32)
        class_ids = np.array([0, 1], dtype=np.int64)
        det_meta = ObjectDetectionMeta(boxes=boxes, scores=scores, class_ids=class_ids)

        axmeta = AxMeta(image_id=0)
        axmeta.add_instance('ObjectDetection', det_meta)
        return img, img_array, boxes, axmeta

    def _run_input_from_roi(self, img, axmeta):
        from axelera.app.operators.input import InputFromROI

        op = InputFromROI(where='ObjectDetection')
        op._need_color_convert = False
        op._need_filter = False
        op._label_filter_ids = None
        op.task_name = 'test_task'
        return op.exec_torch(img, [], axmeta, stream_id='0')

    @pytest.mark.parametrize('use_gst', [False, True], ids=['numpy', 'gstreamer'])
    def test_roi_pixels_match(self, use_gst):
        """InputFromROI extracts ROIs with correct pixel values."""
        img, img_array, boxes, axmeta = self._make_image_and_meta(use_gst)
        _, rois, _ = self._run_input_from_roi(img, axmeta)

        assert len(rois) == 2
        for i, roi in enumerate(rois):
            box = boxes[i].astype(int)
            expected = img_array[box[1] : box[3], box[0] : box[2]]
            np.testing.assert_array_equal(roi.asarray(), expected)


@skip_no_method
class TestAsCVoidPRealGst:
    """Tests for as_c_void_p() using _GstBufferMap (zero-copy)."""

    def test_returns_valid_pointer(self):
        """as_c_void_p returns a non-null ctypes.c_void_p for GStreamer images."""
        import ctypes

        from axelera.types import Image

        sample, _ = _make_sample(200, 100, 'RGB')
        img = Image.fromgst(sample)
        with img.as_c_void_p() as ptr:
            assert ptr is not None
            assert isinstance(ptr, ctypes.c_void_p)
            assert ptr.value is not None

    def test_pointer_reads_correct_data(self):
        """Reading through the pointer returns the same pixel data as asarray."""
        import ctypes

        from axelera.types import Image

        sample1, expected = _make_sample(200, 100, 'RGB')
        img = Image.fromgst(sample1)
        with img.as_c_void_p() as ptr:
            # Read first 6 bytes (2 pixels) via the pointer
            arr = (ctypes.c_uint8 * 6).from_address(ptr.value)
            first_two_pixels = np.frombuffer(arr, dtype=np.uint8)
            np.testing.assert_array_equal(first_two_pixels[:3], expected[0, 0])
            np.testing.assert_array_equal(first_two_pixels[3:6], expected[0, 1])


class TestPyGObjectMapBehavior:
    """Document and verify PyGObject buf.map() behavior.

    These tests confirm that PyGObject's buf.map() returns Python bytes
    (a copy), which is why the ctypes approach is needed for zero-copy.
    """

    def test_buf_map_data_is_bytes(self):
        """PyGObject's buf.map() info.data is bytes, not a zero-copy view."""
        sample, _ = _make_sample(100, 50, 'RGB')
        buf = sample.get_buffer()
        success, info = buf.map(Gst.MapFlags.READ)
        assert success
        assert isinstance(info.data, bytes), (
            f'Expected bytes, got {type(info.data).__name__}. '
            'If this fails, PyGObject may now support zero-copy and '
            'the ctypes approach in _GstBufferMap may no longer be needed.'
        )
        buf.unmap(info)

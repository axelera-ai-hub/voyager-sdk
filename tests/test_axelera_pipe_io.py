# Copyright Axelera AI, 2024

import contextlib
import itertools
import logging
from pathlib import Path
from unittest.mock import ANY, Mock, patch

import cv2
import numpy as np
import pytest

from axelera import types
from axelera.app import display_cv, gst_builder, meta, operators, pipe, pipeline, utils
from axelera.app.config import HardwareCaps

bgr_img = types.Image.fromarray(
    np.full((4, 6, 3), (1, 2, 3), dtype=np.uint8), color_format=types.ColorFormat.BGR
)
rgb_img = cv2.cvtColor(bgr_img.asarray(), cv2.COLOR_BGR2RGB)

MP4V = cv2.VideoWriter_fourcc(*"mp4v")

common_res = [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)]
low_res = [(640, 480), (1280, 720)]


def _gen_input_gst(pipein):
    gst = gst_builder.Builder()
    pipein.build_input_gst(gst, '')
    return list(gst)


def _streamid(idx: int = 0):
    return {
        'name': f'axinplace-addstreamid{idx}',
        'instance': 'axinplace',
        'name': 'axinplace-addstreamid0',
        'lib': 'libinplace_addstreamid.so',
        'mode': 'meta',
        'options': f'stream_id:{idx}',
    }


class MockCapture:
    def __init__(self, supported=common_res, count=100, fps=30, is_opened=True, fail_caps=False):
        self.supported = supported
        w, h = supported[0] if supported else (0, 0)
        self.props = {
            cv2.CAP_PROP_FRAME_WIDTH: float(w),
            cv2.CAP_PROP_FRAME_HEIGHT: float(h),
            cv2.CAP_PROP_FRAME_COUNT: float(count),
            cv2.CAP_PROP_FPS: float(fps),
        }
        self._index = 0
        self._isOpened = is_opened
        self._fail_caps = fail_caps

    def __call__(self, source):
        # mocked constructor
        self.source = source
        return self

    def isOpened(self):
        return self._isOpened

    def get(self, attr):
        if self._fail_caps:
            raise RuntimeError(f'Failed to get property {attr}')
        return self.props[attr]

    def set(self, attr, value):
        if attr == cv2.CAP_PROP_FRAME_WIDTH:
            self.props[attr] = value if int(value) in [w for w, _ in self.supported] else 0
        elif attr == cv2.CAP_PROP_FRAME_HEIGHT:
            self.props[attr] = value if int(value) in [h for _, h in self.supported] else 0
        else:
            self.props[attr] = value

    def read(self):
        if (
            not self._isOpened
            or self.props[cv2.CAP_PROP_FRAME_WIDTH] == 0
            or self.props[cv2.CAP_PROP_FRAME_HEIGHT] == 0
        ):
            return False, None

        num_frames = self.props[cv2.CAP_PROP_FRAME_COUNT]
        if num_frames and self._index >= num_frames:
            return False, None

        bgr = bgr_img.asarray()
        x = bgr if self._index == 0 else np.full_like(bgr, self._index)
        self._index += 1
        return True, x

    def release(self):
        pass


@pytest.mark.parametrize('resolutions', [common_res, low_res])
def test_input_torch_usb0(resolutions):
    with patch.object(cv2, 'VideoCapture', new=MockCapture(resolutions, 100, 30)) as capture:
        pipein = pipe.PipeInput('Torch', source='usb')
    assert pipein.format == 'usb'
    assert pipein.location == '/dev/video0'
    assert pipein.number_of_frames == 100
    assert pipein.fps == 30
    assert capture.source == 0


def test_input_torch_usb_no_res():
    with patch.object(cv2, 'VideoCapture', new=MockCapture([], 100, 30)):
        with pytest.raises(RuntimeError, match='Unable to find a supported resolution'):
            pipe.PipeInput('Torch', source='usb')


@contextlib.contextmanager
def mock_pathlib(*, is_file: bool = None, is_dir: bool = None, resolve: str = None):
    with contextlib.ExitStack() as stack:
        if is_file is not None:
            stack.enter_context(patch.object(Path, 'is_file', return_value=is_file))
        if is_dir is not None:
            stack.enter_context(patch.object(Path, 'is_dir', return_value=is_dir))
        if resolve is not None:

            def resolver(x: Path):
                return Path(resolve) / x

            stack.enter_context(patch.object(Path, 'resolve', resolver))
        yield


@pytest.mark.parametrize(
    'source, format, location, cap_source, width, height',
    [
        ('usb', 'usb', '/dev/video0', 0, 0, 0),
        ('usb:0', 'usb', '/dev/video0', 0, 0, 0),
        ('usb:1', 'usb', '/dev/video1', 1, 0, 0),
        ('usb:1:1280x720', 'usb', '/dev/video1', 1, 1280, 720),
        ('usb:/dev/summit', 'usb', '/dev/summit', '/dev/summit', 0, 0),
        ('usb:/dev/summit:1280x720', 'usb', '/dev/summit', '/dev/summit', 1280, 720),
        ('csi:/dev/video0', 'csi', '/dev/video0', '/dev/video0', 0, 0),
        ('http://localhost', 'hls', 'http://localhost', 'http://localhost', 0, 0),
        ('https://localhost', 'hls', 'https://localhost', 'https://localhost', 0, 0),
        ('rtsp://localhost', 'rtsp', 'rtsp://localhost', 'rtsp://localhost', 0, 0),
        ("hello.flv", 'video', "/abs/hello.flv", "/abs/hello.flv", 0, 0),
        ("hello.avi", 'video', "/abs/hello.avi", "/abs/hello.avi", 0, 0),
        ("hello.mp4", 'video', "/abs/hello.mp4", "/abs/hello.mp4", 0, 0),
        ("hello.3gp", 'video', "/abs/hello.3gp", "/abs/hello.3gp", 0, 0),
        ("hello.mov", 'video', "/abs/hello.mov", "/abs/hello.mov", 0, 0),
        ("hello.webm", 'video', "/abs/hello.webm", "/abs/hello.webm", 0, 0),
        ("hello.ogg", 'video', "/abs/hello.ogg", "/abs/hello.ogg", 0, 0),
        ("hello.qt", 'video', "/abs/hello.qt", "/abs/hello.qt", 0, 0),
        ("hello.avchd", 'video', "/abs/hello.avchd", "/abs/hello.avchd", 0, 0),
    ],
)
def test_input_torch_video_sources(source, format, location, cap_source, width, height):
    with patch.object(cv2, 'VideoCapture', MockCapture()) as capture:
        with mock_pathlib(is_file=True, resolve='/abs'):
            pipein = pipe.PipeInput('Torch', source)
    assert pipein.format == format
    assert pipein.location == location
    assert capture.source == cap_source
    print(f"pipein.width: {pipein.width}, pipein.height: {pipein.height}")


@pytest.mark.parametrize(
    'source, format, location, width, height',
    [
        ('fakevideo', 'video', 'fake', 1280, 720),
        ('fakevideo:800x480', 'video', 'fake', 800, 480),
    ],
)
def test_input_torch_fakevideo_sources(source, format, location, width, height):
    pipein = pipe.PipeInput('Torch', source)
    assert pipein.format == format
    assert pipein.location == location
    assert pipein.width == width
    assert pipein.height == height

    images = list(itertools.islice(pipein.create_generator(), 2))
    assert len(images) == 2
    assert images[0].img.size == (width, height)
    np.testing.assert_equal(images[0].img.asarray(), images[1].img.asarray())


@pytest.mark.parametrize('extension', utils.IMAGE_EXTENSIONS)
def test_input_torch_image(extension):
    with mock_pathlib(is_file=True, resolve='/abs/somedir'):
        with patch.object(cv2, 'imread', return_value=bgr_img.asarray()):
            pipein = pipe.PipeInput('Torch', f'file{extension}')

    assert pipein.format == 'image'
    assert pipein.location == f'/abs/somedir/file{extension}'
    assert pipein.number_of_frames == 1


real_iterdir = Path.iterdir


def mock_iterdir(*files):
    def iterdir(path):
        if path.name == 'somedir':
            for f in files:
                yield Path(f)
        else:
            yield from real_iterdir(path)

    return iterdir


empty_iterdir = mock_iterdir('a.txt')
a_b_iterdir = mock_iterdir('a.jpg', 'somethingelse.txt', 'b.jpg')


def test_input_torch_images_from_dir():
    with mock_pathlib(is_file=False, is_dir=True, resolve='/abs'):
        with patch.object(Path, 'iterdir', new=a_b_iterdir):
            with patch.object(cv2, 'imread', return_value=bgr_img.asarray()):
                pipein = pipe.PipeInput('Torch', 'somedir')

    assert pipein.format == 'images'
    assert pipein.location == '/abs/somedir'
    assert pipein.images == [Path('a.jpg'), Path('b.jpg')]
    assert pipein.number_of_frames == 2


def test_input_torch_images_from_empty_dir():
    with mock_pathlib(is_file=False, is_dir=True, resolve='/abs'):
        with patch.object(Path, 'iterdir', new=empty_iterdir):
            with pytest.raises(RuntimeError, match='Failed to locate any images in /abs/somedir'):
                pipe.PipeInput('Torch', 'somedir')


def test_input_torch_images_from_bad_image():
    with mock_pathlib(is_file=False, is_dir=True, resolve='/abs'):
        with patch.object(Path, 'iterdir', new=a_b_iterdir):
            with patch.object(cv2, 'imread', return_value=None):
                with pytest.raises(RuntimeError, match='Failed to read image: a.jpg'):
                    pipe.PipeInput('Torch', 'somedir')


def test_input_torch_nonexistent_path():
    with mock_pathlib(is_file=False, is_dir=False, resolve='/abs'):
        with pytest.raises(FileNotFoundError, match="No such file or directory: 'somefile'"):
            pipe.PipeInput('Torch', 'somefile')


def test_input_video_is_opened_false():
    with mock_pathlib(is_file=True, resolve='/file'):
        with patch.object(cv2, 'VideoCapture', new=MockCapture(is_opened=False)):
            with pytest.raises(RuntimeError, match='Failed to open video device'):
                pipe.PipeInput('Torch', source='/file/video.mp4')


def test_input_video_fail_to_get_caps():
    with mock_pathlib(is_file=True, resolve='/file'):
        with patch.object(cv2, 'VideoCapture', new=MockCapture(fail_caps=True)):
            with pytest.raises(RuntimeError, match='Failed to get video capabilities'):
                pipe.PipeInput('Torch', source='/file/video.mp4')


@pytest.mark.parametrize(
    'allow_hardware_codec, vaapi_enabled, expected',
    [
        (True, True, False),
        (True, False, False),
        (False, True, True),
        (False, False, True),
    ],
)
def test_gen_decodebin(allow_hardware_codec, vaapi_enabled, expected):
    gst = gst_builder.Builder()
    pipe.io.build_decodebin(gst, allow_hardware_codec, vaapi_enabled, '')
    assert list(gst) == [
        {
            'instance': 'decodebin',
            'force-sw-decoders': expected,
            'caps': 'video/x-raw(ANY)',
            'expose-all-streams': False,
            'connections': {'src_%u': 'queue_after_decodebin.sink'},
        },
        {
            'instance': 'queue',
            'name': 'queue_after_decodebin',
            'max-size-buffers': 16,
            'max-size-bytes': 0,
            'max-size-time': 0,
        },
    ]
    gst = gst_builder.Builder()
    pipe.io.build_decodebin(gst, allow_hardware_codec, vaapi_enabled, '0')
    assert list(gst) == [
        {
            'instance': 'decodebin',
            'force-sw-decoders': expected,
            'caps': 'video/x-raw(ANY)',
            'expose-all-streams': False,
            'connections': {'src_%u': 'queue_after_decodebin0.sink'},
        },
        {
            'instance': 'queue',
            'name': 'queue_after_decodebin0',
            'max-size-buffers': 16,
            'max-size-bytes': 0,
            'max-size-time': 0,
        },
    ]


@pytest.mark.parametrize(
    'source, device, codec, width, height, expected_caps, hardware_caps',
    [
        ('usb', '/dev/video0', 'mjpg', 0, 0, 'image/jpeg', HardwareCaps.OPENCL),
        ('usb', '/dev/video0', 'mjpg', 0, 0, 'image/jpeg', HardwareCaps.ALL),
        ('usb', '/dev/video0', 'yuyv', 0, 0, 'video/x-raw,format=YUY2', HardwareCaps.OPENCL),
        (
            'usb',
            '/dev/video0',
            'yuyv',
            640,
            480,
            'video/x-raw,format=YUY2,width=640,height=480',
            HardwareCaps.NONE,
        ),
        (
            'usb@45',
            '/dev/video0',
            'yuyv',
            640,
            480,
            'video/x-raw,format=YUY2,width=640,height=480,framerate=45/1',
            HardwareCaps.ALL,
        ),
        ('usb:1', '/dev/video1', 'mjpg', 0, 0, 'image/jpeg', HardwareCaps.ALL),
        ('usb:1@45', '/dev/video1', 'mjpg', 0, 0, 'image/jpeg,framerate=45/1', HardwareCaps.ALL),
        (
            'usb:/dev/summit',
            '/dev/summit',
            'yuyv',
            0,
            0,
            'video/x-raw,format=YUY2',
            HardwareCaps.ALL,
        ),
    ],
)
def test_input_gst_usb(source, device, codec, width, height, expected_caps, hardware_caps):
    with patch.object(cv2, 'VideoCapture', new=MockCapture()):
        pipein = pipe.PipeInput('gst', source)
        pipein.width = width
        pipein.height = height
        pipein.codec = codec
        pipein.hardware_caps = hardware_caps
    gst_repr = [
        {'instance': 'v4l2src', 'device': device},
        {'instance': 'capsfilter', 'caps': expected_caps},
    ]
    if codec == 'mjpg':
        gst_repr.extend(
            [
                {
                    'instance': 'decodebin',
                    'force-sw-decoders': False,
                    'caps': 'video/x-raw(ANY)',
                    'expose-all-streams': False,
                    'connections': {'src_%u': 'queue_after_decodebin.sink'},
                },
                {
                    'instance': 'queue',
                    'name': 'queue_after_decodebin',
                    'max-size-buffers': 16,
                    'max-size-bytes': 0,
                    'max-size-time': 0,
                },
            ]
        )
    gst_repr.append(_streamid())
    assert _gen_input_gst(pipein) == gst_repr


@pytest.mark.parametrize(
    'source, location, username, password, codec',
    [
        ('rtsp://somehost/', 'rtsp://somehost/', '', '', None),
        ('rtsp://user@somehost/path?param=1', 'rtsp://somehost/path?param=1', 'user', '', None),
        ('rtsp://user:pass@somehost/', 'rtsp://somehost/', 'user', 'pass', 'mjpg'),
        ('rtsp://user:pass@somehost/', 'rtsp://somehost/', 'user', 'pass', 'h264'),
        ('rtsp://user:pass@somehost/', 'rtsp://somehost/', 'user', 'pass', 'h265'),
    ],
)
def test_input_gst_rtsp(source, location, username, password, codec):
    with patch.object(cv2, 'VideoCapture', new=MockCapture()):
        pipein = pipe.PipeInput('gst', source)
        if codec:
            pipein.codec = codec
    assert _gen_input_gst(pipein) == [
        {
            'instance': 'rtspsrc',
            'location': location,
            'user-id': username,
            'user-pw': password,
            'latency': 500,
            'connections': {'stream_%u': 'rtspcapsfilter.sink'},
        },
        {
            'instance': 'capsfilter',
            'caps': 'application/x-rtp,media=video',
            'name': 'rtspcapsfilter',
        },
        {
            'instance': 'queue',
            'name': 'queue_in',
            'max-size-buffers': 16,
            'max-size-bytes': 0,
            'max-size-time': 0,
        },
        {
            'instance': 'decodebin',
            'force-sw-decoders': False,
            'caps': 'video/x-raw(ANY)',
            'expose-all-streams': False,
            'connections': {'src_%u': 'queue_after_decodebin.sink'},
        },
        {
            'instance': 'queue',
            'name': 'queue_after_decodebin',
            'max-size-buffers': 16,
            'max-size-bytes': 0,
            'max-size-time': 0,
        },
        _streamid(),
    ]


def test_input_gst_video():
    with patch.object(cv2, 'VideoCapture', new=MockCapture()):
        with mock_pathlib(is_dir=False, is_file=True, resolve='/abs'):
            pipein = pipe.PipeInput('gst', 'path/to/video.mp4')
    assert _gen_input_gst(pipein) == [
        {'instance': 'filesrc', 'location': '/abs/path/to/video.mp4'},
        {
            'instance': 'queue',
            'name': 'queue_in',
            'max-size-buffers': 16,
            'max-size-bytes': 0,
            'max-size-time': 0,
        },
        {
            'instance': 'decodebin',
            'force-sw-decoders': False,
            'caps': 'video/x-raw(ANY)',
            'expose-all-streams': False,
            'connections': {'src_%u': 'queue_after_decodebin.sink'},
        },
        {
            'instance': 'queue',
            'name': 'queue_after_decodebin',
            'max-size-buffers': 16,
            'max-size-bytes': 0,
            'max-size-time': 0,
        },
        _streamid(),
    ]


def test_input_gst_image():
    with mock_pathlib(is_file=True, resolve='/abs/somedir'):
        with patch.object(cv2, 'imread', return_value=bgr_img.asarray()):
            pipein = pipe.PipeInput('gst', 'file.jpg')
    assert _gen_input_gst(pipein) == [
        {'instance': 'filesrc', 'location': '/abs/somedir/file.jpg'},
        {
            'instance': 'decodebin',
            'force-sw-decoders': True,
            'caps': 'video/x-raw(ANY)',
            'expose-all-streams': False,
            'connections': {'src_%u': 'queue_after_decodebin.sink'},
        },
        {
            'instance': 'queue',
            'name': 'queue_after_decodebin',
            'max-size-buffers': 16,
            'max-size-bytes': 0,
            'max-size-time': 0,
        },
        _streamid(),
    ]


def test_input_gst_images():
    with patch.object(Path, 'iterdir', new=a_b_iterdir):
        with mock_pathlib(is_dir=True, is_file=False, resolve='/abs/somedir'):
            with patch.object(cv2, 'imread', return_value=bgr_img.asarray()):
                with pytest.raises(
                    NotImplementedError, match="images format not supported in gst pipe"
                ):
                    pipein = pipe.PipeInput('gst', 'somedir')
                    _gen_input_gst(pipein)


def test_input_create_generator_video():
    with mock_pathlib(is_file=True, resolve='/'):
        with patch.object(cv2, 'VideoCapture', new=MockCapture(count=5)):
            pipein = pipe.PipeInput('Torch', source='/file/video.mp4')

    got = list(pipein.create_generator())
    assert len(got) == 5
    np.testing.assert_equal(got[0].img.asarray(), bgr_img.asarray())


def test_input_create_generator_images():
    with patch.object(cv2, 'imread', return_value=bgr_img.asarray()):
        with patch.object(Path, 'iterdir', new=a_b_iterdir):
            with mock_pathlib(is_dir=True, is_file=False, resolve='/'):
                pipein = pipe.PipeInput('Torch', source='/abs/somedir')

        got = list(pipein.create_generator())
    assert len(got) == 2
    np.testing.assert_equal(got[0].img.asarray(), bgr_img.asarray())


def new_capture_5(source):
    return MockCapture(count=5)


def test_input_create_generator_multiplex():
    with mock_pathlib(is_file=True, resolve='/file'):
        with patch.object(cv2, 'VideoCapture', new_capture_5):
            pipein = pipe.io.MultiplexPipeInput('Torch', ['video.mp4', 'video.mp4'])

    got = list(pipein.create_generator())
    assert len(got) == 10
    np.testing.assert_equal(got[0].img.asarray(), bgr_img.asarray())


def test_input_create_generator_multiplex_mismatch_codec():
    with contextlib.ExitStack() as stack:
        enter = stack.enter_context
        enter(mock_pathlib(is_file=True, resolve='/'))
        enter(patch.object(cv2, 'VideoCapture', new_capture_5))
        enter(pytest.raises(ValueError, match='All codecs must be the same'))
        pipe.io.MultiplexPipeInput(
            'Torch', ['/file/video.mp4', '/file/video.mp4'], ['jpeg', 'mpeg']
        )


def test_input_create_generator_multiplex_mismatch_format():
    with contextlib.ExitStack() as stack:
        enter = stack.enter_context
        enter(mock_pathlib(is_file=True, resolve='/file'))
        enter(mock_pathlib(is_file=True, resolve='/file'))
        enter(patch.object(cv2, 'imread', return_value=bgr_img.asarray()))
        enter(patch.object(cv2, 'VideoCapture', new_capture_5))
        mock_warning = enter(patch.object(logging.Logger, 'warning'))
        pipe.io.MultiplexPipeInput('Torch', ['video.mp4', 'image.jpg'])
        mock_warning.assert_called_once_with(
            'Not all input sources have the same format: [\'video\', \'image\']'
        )


def create_pipein(fps):
    m = Mock()
    m.fps = fps
    return m


def do_writes(pipein, pipeout, *writes):
    pipeout.initialize_writer(pipein)
    for data, name in writes:
        result = pipe.FrameResult(data, meta=meta.AxMeta(name))
        pipeout.sink(result)
    pipeout.close_writer()


def np_assert_called_with(mock, calls):
    if len(mock.call_args_list) != len(calls):
        times = 'once' if len(calls) == 1 else f'{len(calls)} times'
        wth = ['\n'.join(f'({args}, {kwargs}' for args, kwargs in calls)]
        msg = "Expected '%s' to be called %s with %s. Called %s times.%s" % (
            mock._mock_name or 'mock',
            times,
            wth,
            mock.call_count,
            mock._calls_repr(),
        )
        raise AssertionError(msg)
    np.testing.assert_equal(mock.call_args_list, calls)


def np_assert_called_once_with(mock, *args, **kwargs):
    np_assert_called_with(mock, [(args, kwargs)])


def test_output_no_save():
    pipein = create_pipein(30)
    pipeout = pipe.PipeOutput()
    with patch.object(cv2, 'imwrite') as mock_imwrite:
        do_writes(pipein, pipeout, (bgr_img, 'unused'))
        assert mock_imwrite.called is False
    assert pipeout.result.image is bgr_img


def test_output_save_video_valid_input():
    pipein = create_pipein(30)
    pipeout = pipe.PipeOutput(save_output='somefile.mp4')
    with patch.object(cv2, 'VideoWriter') as mock_writer:
        mock_cvwriter = mock_writer.return_value
        do_writes(pipein, pipeout, (bgr_img, 'unused'))
        mock_writer.assert_called_once_with('somefile.mp4', MP4V, 30, (6, 4))
        np.testing.assert_array_equal(
            mock_cvwriter.write.call_args_list[0][0][0], bgr_img.asarray('BGR')
        )
        mock_cvwriter.release.assert_called_once()


def test_output_save_video_valid_input_with_meta_does_draw():
    draw_called = []

    class MockTaskMeta(meta.AxTaskMeta):
        def draw(self, d):
            draw_called.append(d)

    pipein = create_pipein(30)
    pipeout = pipe.PipeOutput(save_output='somefile.mp4')
    with patch.object(display_cv, 'CVDraw'):
        with patch.object(cv2, 'VideoWriter'):
            m = meta.AxMeta('whocares')
            m.get_instance('whocares', MockTaskMeta)
            pipeout.initialize_writer(pipein)
            pipeout.sink(pipe.FrameResult(bgr_img, None, m, 0))
    assert len(draw_called) == 1


def test_output_no_save_video_expected_gst_is_empty():
    # TODO this will need updating when we add app sink
    pipeout = pipe.PipeOutput(save_output='')
    gst = gst_builder.Builder()
    pipeout.build_output_gst(gst, 0)
    assert list(gst) == [
        {
            'drop': False,
            'instance': 'appsink',
            'max-buffers': 16,
            'sync': False,
        }
    ]
    gst = gst_builder.Builder()
    pipeout.build_output_gst(gst, 1)
    assert list(gst) == [
        {
            'drop': False,
            'instance': 'appsink',
            'max-buffers': 16,
            'sync': False,
        }
    ]


def test_output_save_video_invalid_input():
    # e.g. input is images not video, fps/width/height cannot be retrieved from source
    # so we use the image size and assume fps is 30
    pipein = create_pipein(0)
    pipeout = pipe.PipeOutput(save_output='somefile.mp4')
    with patch.object(cv2, 'VideoWriter') as mock_writer:
        mock_cvwriter = mock_writer.return_value
        do_writes(pipein, pipeout, (bgr_img, 'unused'))
        mock_writer.assert_called_once_with('somefile.mp4', MP4V, 30, (6, 4))
        assert mock_cvwriter.write.call_count == 1
        exp_img = bgr_img.asarray('BGR')
        np.testing.assert_array_equal(mock_cvwriter.write.call_args[0][0], exp_img)
        mock_cvwriter.release.assert_called_once()


def test_output_save_images_img_id_specified():
    pipein = create_pipein(30)
    pipeout = pipe.PipeOutput(save_output='/path/to/images/')
    with patch.object(cv2, 'imwrite') as mock_write:
        with patch.object(Path, 'mkdir') as mock_mkdir:
            do_writes(pipein, pipeout, (bgr_img, 'someinput.jpg'))
            np_assert_called_once_with(
                mock_write, '/path/to/images/output_someinput.jpg', bgr_img.asarray()
            )
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_output_save_images_img_id_not_specified():
    pipein = create_pipein(30)
    pipeout = pipe.PipeOutput(save_output='/path/to/images/')
    with patch.object(cv2, 'imwrite') as mock_write:
        with patch.object(Path, 'mkdir') as mock_mkdir:
            do_writes(pipein, pipeout, (bgr_img, ''), (bgr_img, ''))
            np_assert_called_with(
                mock_write,
                [
                    (('/path/to/images/output_00000.jpg', bgr_img.asarray()), {}),
                    (('/path/to/images/output_00001.jpg', bgr_img.asarray()), {}),
                ],
            )
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_output_save_images_custom_pattern():
    pipein = create_pipein(30)
    pipeout = pipe.PipeOutput(save_output='/path/to/images/img%02d.jpg')
    with patch.object(cv2, 'imwrite') as mock_write:
        with patch.object(Path, 'mkdir') as mock_mkdir:
            do_writes(pipein, pipeout, (bgr_img, ''), (bgr_img, ''))
            np_assert_called_with(
                mock_write,
                [
                    (('/path/to/images/img00.jpg', bgr_img.asarray()), {}),
                    (('/path/to/images/img01.jpg', bgr_img.asarray()), {}),
                ],
            )
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_output_save_images_no_pattern_one_image():
    pipein = create_pipein(30)
    pipeout = pipe.PipeOutput(save_output='/path/to/images/img.jpg')
    with patch.object(cv2, 'imwrite') as mock_write:
        with patch.object(Path, 'mkdir') as mock_mkdir:
            do_writes(pipein, pipeout, (bgr_img, ''))
            np_assert_called_once_with(mock_write, '/path/to/images/img.jpg', bgr_img.asarray())
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_output_save_images_no_pattern_more_than_once_image():
    pipein = create_pipein(30)
    pipeout = pipe.PipeOutput(save_output='/path/to/images/img.jpg')
    with patch.object(cv2, 'imwrite'):
        with patch.object(Path, 'mkdir'):
            with pytest.raises(ValueError, match="containing '%d', then the input"):
                do_writes(pipein, pipeout, (bgr_img, ''), (bgr_img, ''))

# Copyright Axelera AI, 2023

import contextlib
import itertools
import logging
import os
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import pytest

from axelera import types
from axelera.app import config, display_cv, gst_builder, inf_tracers, meta, pipe, utils
from axelera.app.config import HardwareCaps
from axelera.app.pipe import io

bgr_img = types.Image.fromarray(
    np.full((4, 6, 3), (1, 2, 3), dtype=np.uint8), color_format=types.ColorFormat.BGR
)
rgb_img = cv2.cvtColor(bgr_img.asarray(), cv2.COLOR_BGR2RGB)

MP4V = cv2.VideoWriter_fourcc(*"mp4v")

common_res = [(640, 480), (1280, 720), (1920, 1080), (3840, 240)]
low_res = [(640, 480), (1280, 720)]


def _gen_input_gst(pipein):
    gst = gst_builder.Builder(None, None, 4, 'auto')
    pipein.build_input_gst(gst, '0')
    return list(gst)


def _streamid(idx: int = 0, fps_limit='', decodebin_required: bool = False):
    fps = f';fps_limit:{fps_limit}' if fps_limit else ''
    name = {'name': f'decodebin-link{idx}'} if decodebin_required else {}
    return {
        **name,
        'instance': 'axinplace',
        'lib': 'libinplace_addstreamid.so',
        'mode': 'meta',
        'options': f'stream_id:{idx}{fps}',
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
        pipein = io.SinglePipeInput('torch', config.Source('usb'))
    assert pipein.sources[0].type == config.SourceType.USB
    assert pipein.number_of_frames == 100
    assert pipein.fps == 30
    assert capture.source == 0


def test_input_torch_usb_no_res():
    with patch.object(cv2, 'VideoCapture', new=MockCapture([], 100, 30)):
        with pytest.raises(RuntimeError, match='Unable to find a supported resolution'):
            io.SinglePipeInput('torch', config.Source('usb:0:0x0'))


@contextlib.contextmanager
def mock_pathlib(
    *,
    is_file: bool | None = None,
    is_dir: bool | None = None,
    resolve: str | None = None,
    exists: bool | None = True,
    rglob: list[str] | None = None,
):
    with contextlib.ExitStack() as stack:
        if is_file is not None:
            stack.enter_context(patch.object(Path, 'is_file', return_value=is_file))
        if is_dir is not None:
            stack.enter_context(patch.object(Path, 'is_dir', return_value=is_dir))
        if resolve is not None:

            def resolver(x: Path):
                return Path(resolve) / x

            stack.enter_context(patch.object(Path, 'resolve', resolver))

        stack.enter_context(patch.object(Path, 'exists', return_value=exists))
        if rglob is not None:
            stack.enter_context(patch.object(Path, 'rglob', return_value=rglob))
        yield


@pytest.mark.parametrize(
    'source, cap_source',
    [
        ('usb', 0),
        ('usb:0', 0),
        ('usb:1', 1),
        ('usb:/dev/summit', '/dev/summit'),
    ],
)
def test_input_torch_video_usb_capture(source, cap_source):
    with patch.object(cv2, 'VideoCapture', MockCapture()) as capture:
        with mock_pathlib(is_file=True, resolve='/abs'):
            io.SinglePipeInput('torch', config.Source(source))
    assert capture.source == cap_source


@pytest.mark.parametrize(
    'source, width, height',
    [
        ('fakevideo', 1280, 720),
        ('fakevideo:800x480', 800, 480),
    ],
)
def test_input_torch_fakevideo_sources(source, width, height):
    pipein = io.SinglePipeInput('torch', config.Source(source))
    assert pipein.sources[0].type == config.SourceType.FAKE_VIDEO
    images = list(itertools.islice(pipein.frame_generator(), 2))
    assert len(images) == 2
    assert images[0].img.size == (width, height)
    np.testing.assert_equal(images[0].img.asarray(), images[1].img.asarray())


@pytest.mark.parametrize('extension', utils.IMAGE_EXTENSIONS)
def test_input_torch_image(extension):
    with mock_pathlib(is_file=True, resolve='/abs/somedir'):
        with patch.object(cv2, 'imread', return_value=bgr_img.asarray()):
            pipein = io.SinglePipeInput('Torch', config.Source(f'file{extension}'))

    assert pipein.sources[0].type == config.SourceType.IMAGE_FILES
    assert pipein.sources[0].images == [Path(f'/abs/somedir/file{extension}')]
    assert pipein.number_of_frames == 1


real_iterdir = Path.iterdir


def mock_iterdir(*files):
    def iterdir(path):
        if path.name == 'somedir':
            for f in files:
                yield Path(f'somedir/{f}')
        else:
            yield from real_iterdir(path)

    return iterdir


empty_iterdir = mock_iterdir('a.txt')
a_b_iterdir = mock_iterdir('a.jpg', 'somethingelse.txt', 'b.jpg')


def test_input_torch_images_from_dir():
    """Test images from directory - mock utils.list_images_recursive"""
    with mock_pathlib(is_file=False, is_dir=True, resolve='/abs'):
        with patch(
            'axelera.app.utils.list_images_recursive', return_value=[Path('a.jpg'), Path('b.jpg')]
        ):
            with patch.object(cv2, 'imread', return_value=bgr_img.asarray()):
                pipein = io.SinglePipeInput('torch', config.Source('somedir'))
                assert pipein.number_of_frames == 2

                # Check that img_id is set correctly - INSIDE the mock context
                images = list(itertools.islice(pipein.frame_generator(), 2))
                assert len(images) == 2
                assert images[0].img_id == os.path.relpath(str(Path('a.jpg')), os.getcwd())
                assert images[1].img_id == os.path.relpath(str(Path('b.jpg')), os.getcwd())


def test_input_torch_images_from_empty_dir():
    """Test images from empty directory"""
    with mock_pathlib(is_file=False, is_dir=True, resolve='/abs'):
        with patch('axelera.app.utils.list_images_recursive', return_value=[]):
            with pytest.raises(RuntimeError, match='Failed to locate any images in somedir'):
                io.SinglePipeInput('Torch', config.Source('somedir'))


def test_input_torch_images_from_bad_image():
    """Test handling of bad image files"""
    with mock_pathlib(is_file=False, is_dir=True, resolve='/abs'):
        with patch(
            'axelera.app.utils.list_images_recursive', return_value=[Path('a.jpg'), Path('b.jpg')]
        ):
            with patch.object(cv2, 'imread', return_value=None):
                with pytest.raises(RuntimeError, match='Failed to read image: a.jpg'):
                    pipein = io.SinglePipeInput('Torch', config.Source('somedir'))
                    next(pipein.frame_generator())


def test_input_torch_nonexistent_path():
    with mock_pathlib(is_file=False, is_dir=False, resolve='/abs'):
        with pytest.raises(ValueError, match="Unrecognized source: somefile"):
            io.SinglePipeInput('Torch', config.Source('somefile'))


def test_input_video_is_opened_false():
    with mock_pathlib(is_file=True, resolve='/file'):
        with patch.object(cv2, 'VideoCapture', new=MockCapture(is_opened=False)):
            with pytest.raises(RuntimeError, match='Failed to open video device'):
                io.SinglePipeInput('torch', config.Source('/file/video.mp4'))


def test_input_video_fail_to_get_caps():
    with mock_pathlib(is_file=True, resolve='/file'):
        with patch.object(cv2, 'VideoCapture', new=MockCapture(fail_caps=True)):
            with pytest.raises(RuntimeError, match='Failed to get video capabilities'):
                io.SinglePipeInput('torch', config.Source('/file/video.mp4'))


def test_input_video_fail_to_get_caps_gst():
    with mock_pathlib(is_file=True, resolve='/file'):
        with patch.object(os, 'access', return_value=True):
            with patch.object(cv2, 'VideoCapture', new=MockCapture(fail_caps=True)):
                with pytest.raises(RuntimeError, match='Failed to get video capabilities'):
                    io.SinglePipeInput('gst', config.Source('/file/video.mp4'))


@pytest.mark.parametrize(
    'allow_hardware_codec, expected',
    [
        (True, False),
        (False, True),
    ],
)
def test_gen_decodebin(allow_hardware_codec, expected):
    gst = gst_builder.Builder(None, None, 4, 'auto')
    pipe.io.build_decodebin(gst, allow_hardware_codec, '')
    assert list(gst) == [
        {
            'instance': 'decodebin',
            'force-sw-decoders': expected,
            'caps': 'video/x-raw(ANY)',
            'expose-all-streams': False,
            'connections': {'src_%u': 'decodebin-link0.sink'},
        },
    ]
    gst = gst_builder.Builder(None, None, 4, 'auto')
    pipe.io.build_decodebin(gst, allow_hardware_codec, '0')
    assert list(gst) == [
        {
            'instance': 'decodebin',
            'force-sw-decoders': expected,
            'caps': 'video/x-raw(ANY)',
            'expose-all-streams': False,
            'connections': {'src_%u': 'decodebin-link0.sink'},
        },
    ]


@pytest.mark.parametrize(
    'source, device, expected_caps, hardware_caps',
    [
        ('usb:0:/mjpg', '/dev/video0', 'image/jpeg', HardwareCaps.OPENCL),
        ('usb:0:/mjpg', '/dev/video0', 'image/jpeg', HardwareCaps.ALL),
        ('usb:0:/yuyv', '/dev/video0', 'video/x-raw,format=YUY2', HardwareCaps.OPENCL),
        (
            'usb:0:640x480:/yuyv',
            '/dev/video0',
            'video/x-raw,format=YUY2,width=640,height=480',
            HardwareCaps.NONE,
        ),
        (
            'usb:0:640x480@45/yuyv',
            '/dev/video0',
            'video/x-raw,format=YUY2,width=640,height=480,framerate=45/1',
            HardwareCaps.ALL,
        ),
        ('usb:1:/mjpg', '/dev/video1', 'image/jpeg', HardwareCaps.ALL),
        ('usb:1@45:/mjpg', '/dev/video1', 'image/jpeg,framerate=45/1', HardwareCaps.ALL),
        (
            'usb:/dev/summit:/yuyv',
            '/dev/summit',
            'video/x-raw,format=YUY2',
            HardwareCaps.ALL,
        ),
    ],
)
def test_input_gst_usb(source, device, expected_caps, hardware_caps):
    with patch.object(os, 'access', return_value=True) as maccess:
        pipein = io.SinglePipeInput('gst', config.Source(source), hardware_caps)

    maccess.assert_called_once_with(device, os.F_OK | os.R_OK | os.W_OK)
    gst_repr = [
        {'instance': 'v4l2src', 'device': device},
        {'instance': 'capsfilter', 'caps': expected_caps},
    ]
    if 'jpeg' in expected_caps:
        gst_repr.extend(
            [
                {
                    'instance': 'decodebin',
                    'force-sw-decoders': False,
                    'caps': 'video/x-raw(ANY)',
                    'expose-all-streams': False,
                    'connections': {'src_%u': 'decodebin-link0.sink'},
                },
            ]
        )
    gst_repr.append(_streamid(decodebin_required='jpeg' in expected_caps))
    with patch.object(os, 'access', return_value=1):
        assert _gen_input_gst(pipein) == gst_repr


def test_input_gst_bad_usb():
    with patch.object(os, 'access', return_value=0):
        with pytest.raises(RuntimeError, match='Cannot access device at /dev/nottoday'):
            io.SinglePipeInput('gst', config.Source('usb:/dev/nottoday'))


@pytest.mark.parametrize(
    'source, location, username, password, protocol, exp_protocol',
    [
        ('rtsp://somehost/', 'rtsp://somehost/', '', '', '', None),
        ('rtsp://somehost/', 'rtsp://somehost/', '', '', 'tcp', 4),
        ('rtsp://somehost/', 'rtsp://somehost/', '', '', 'udp', 1),
        ('rtsp://somehost/', 'rtsp://somehost/', '', '', 'all', None),
        (
            'rtsp://user@somehost/path?param=1',
            'rtsp://somehost/path?param=1',
            'user',
            '',
            '',
            None,
        ),
        ('rtsp://user:pass@somehost/', 'rtsp://somehost/', 'user', 'pass', 'all', None),
        ('rtsp://user:pass@somehost/', 'rtsp://somehost/', 'user', 'pass', 'all', None),
        ('rtsp://user:pass@somehost/', 'rtsp://somehost/', 'user', 'pass', 'all', None),
        (
            'rtsp://admin:P@ssw0rd123@172.20.80.248/',
            'rtsp://172.20.80.248/',
            'admin',
            'P@ssw0rd123',
            'all',
            None,
        ),
        (
            'rtsp://admin:P%40ssw0rd123@172.20.80.248/',
            'rtsp://172.20.80.248/',
            'admin',
            'P@ssw0rd123',
            'all',
            None,
        ),
        (
            'rtsp://admin:Passw0rd123@@172.20.80.248/',
            'rtsp://172.20.80.248/',
            'admin',
            'Passw0rd123@',
            'all',
            None,
        ),
        (
            'rtsp://admin:Passw0rd123%40@172.20.80.248/',
            'rtsp://172.20.80.248/',
            'admin',
            'Passw0rd123@',
            'all',
            None,
        ),
        ('rtsp://user:Pass:word@somehost/', 'rtsp://somehost/', 'user', 'Pass:word', '', None),
        ('rtsp://user:Pass%3Aword@somehost/', 'rtsp://somehost/', 'user', 'Pass:word', '', None),
        ('rtsp://user:Pass%2Fword@somehost/', 'rtsp://somehost/', 'user', 'Pass/word', '', None),
        ('rtsp://user:Pass%3Fword@somehost/', 'rtsp://somehost/', 'user', 'Pass?word', '', None),
        ('rtsp://user:Pass%26word@somehost/', 'rtsp://somehost/', 'user', 'Pass&word', '', None),
        ('rtsp://user:Pass%3Dword@somehost/', 'rtsp://somehost/', 'user', 'Pass=word', '', None),
        ('rtsp://user:Pass%23word@somehost/', 'rtsp://somehost/', 'user', 'Pass#word', '', None),
        ('rtsp://user:Pass%25word@somehost/', 'rtsp://somehost/', 'user', 'Pass%word', '', None),
        ('rtsp://user:Pass%20word@somehost/', 'rtsp://somehost/', 'user', 'Pass word', '', None),
        ('rtsp://user:Pass%2Bword@somehost/', 'rtsp://somehost/', 'user', 'Pass+word', '', None),
        ('rtsp://user:Pass%21word@somehost/', 'rtsp://somehost/', 'user', 'Pass!word', '', None),
        ('rtsp://user:Pass%24word@somehost/', 'rtsp://somehost/', 'user', 'Pass$word', '', None),
        ('rtsp://user:Pass%2Aword@somehost/', 'rtsp://somehost/', 'user', 'Pass*word', '', None),
        (
            'rtsp://user:Pass%28word%29@somehost/',
            'rtsp://somehost/',
            'user',
            'Pass(word)',
            '',
            None,
        ),
        (
            'rtsp://admin:P%40ss%23w0rd%21%26%24@192.168.1.1/',
            'rtsp://192.168.1.1/',
            'admin',
            'P@ss#w0rd!&$',
            'tcp',
            4,
        ),
        (
            'rtsp://root:Abc%21%40%23%24%25%5E%26%2A%28%29123@10.0.0.1/stream',
            'rtsp://10.0.0.1/stream',
            'root',
            'Abc!@#$%^&*()123',
            'udp',
            1,
        ),
    ],
)
def test_input_gst_rtsp(source, location, username, password, protocol, exp_protocol):
    with contextlib.ExitStack() as s:
        s.enter_context(patch.object(os, 'access', return_value=1))
        s.enter_context(patch.object(config, 'env', Mock(rtsp_protocol=protocol)))
        pipein = io.SinglePipeInput('gst', config.Source(source))
        got = _gen_input_gst(pipein)
    prot = {} if exp_protocol is None else {'protocols': exp_protocol}
    assert got == [
        {
            'instance': 'rtspsrc',
            'location': location,
            'user-id': username,
            'user-pw': password,
            'latency': 500,
            'connections': {'stream_%u': 'rtspcapsfilter0.sink'},
            **prot,
        },
        {
            'instance': 'capsfilter',
            'caps': 'application/x-rtp,media=video',
            'name': 'rtspcapsfilter0',
        },
        {
            'instance': 'decodebin',
            'force-sw-decoders': False,
            'caps': 'video/x-raw(ANY)',
            'expose-all-streams': False,
            'connections': {'src_%u': 'decodebin-link0.sink'},
        },
        _streamid(decodebin_required=True),
    ]


@pytest.mark.parametrize(
    'source,fps_limit',
    [('path/to/video.mp4', ''), ('path/to/video.mp4@15', '15'), ('path/to/video.mp4@auto', '30')],
)
def test_input_gst_video(source, fps_limit):
    with patch.object(os, 'access', return_value=True) as maccess:
        with mock_pathlib(is_dir=False, is_file=True):
            with patch.object(cv2, 'VideoCapture', new=MockCapture()):
                pipein = io.SinglePipeInput('gst', config.Source(source))
    assert 30 == pipein.sources[0].fps
    assert 100 == pipein.number_of_frames
    maccess.assert_called_once_with('path/to/video.mp4', os.F_OK | os.R_OK | os.W_OK)
    assert _gen_input_gst(pipein) == [
        {'instance': 'filesrc', 'location': 'path/to/video.mp4'},
        {
            'instance': 'decodebin',
            'force-sw-decoders': False,
            'caps': 'video/x-raw(ANY)',
            'expose-all-streams': False,
            'connections': {'src_%u': 'decodebin-link0.sink'},
        },
        _streamid(fps_limit=fps_limit, decodebin_required=True),
    ]


def test_input_gst_images():
    """Test GST images input"""
    with patch(
        'axelera.app.utils.list_images_recursive', return_value=[Path('a.jpg'), Path('b.jpg')]
    ):
        with mock_pathlib(is_dir=True, is_file=False, resolve='/abs/somedir'):
            with patch.object(cv2, 'imread', return_value=bgr_img.asarray()):
                pipein = io.SinglePipeInput('gst', config.Source('somedir'))
                gst_result = _gen_input_gst(pipein)
                expected_start = [
                    {
                        'instance': 'appsrc',
                        'is-live': True,
                        'do-timestamp': True,
                        'format': 3,
                    },
                ]
                expected_end = [
                    {
                        'instance': 'axinplace',
                        'lib': 'libinplace_addstreamid.so',
                        'mode': 'meta',
                        'options': 'stream_id:0',
                    },
                ]
            assert gst_result[0] == expected_start[0]
            assert gst_result[-len(expected_end) :] == expected_end


def test_input_frame_generator_video():
    with mock_pathlib(is_file=True, resolve='/'):
        with patch.object(cv2, 'VideoCapture', new=MockCapture(count=5)):
            pipein = io.SinglePipeInput('torch', config.Source('/file/video.mp4'))

    got = list(pipein.frame_generator())
    assert len(got) == 5
    np.testing.assert_equal(got[0].img.asarray(), bgr_img.asarray())


def test_input_frame_generator_images():
    """Test frame generator for images"""
    with patch.object(cv2, 'imread', return_value=bgr_img.asarray()):
        with patch(
            'axelera.app.utils.list_images_recursive', return_value=[Path('a.jpg'), Path('b.jpg')]
        ):
            with mock_pathlib(is_dir=True, is_file=False, resolve='/abs'):
                pipein = io.SinglePipeInput('Torch', config.Source('/abs/somedir'))
                got = list(pipein.frame_generator())
                assert len(got) == 2
                np.testing.assert_equal(got[0].img.asarray(), bgr_img.asarray())


def new_capture_5(source):
    return MockCapture(count=5)


def test_input_frame_generator_multiplex():
    with mock_pathlib(is_file=True, resolve='/file'):
        with patch.object(cv2, 'VideoCapture', new_capture_5):
            pipein = pipe.io.MultiplexPipeInput(
                [config.Source('video.mp4'), config.Source('video.mp4')],
                config.SystemConfig(),
                config.PipelineConfig(pipe_type='torch'),
                pipe.SourceIdAllocator(),
            )

    got = list(pipein.frame_generator())
    assert len(got) == 10
    np.testing.assert_equal(got[0].img.asarray(), bgr_img.asarray())
    assert got[0].stream_id == 0
    assert got[1].stream_id == 1


def test_input_frame_generator_multiplex_mismatch_format():
    """Test multiplex input with mismatched formats"""
    with contextlib.ExitStack() as stack:
        enter = stack.enter_context
        enter(mock_pathlib(is_file=True, resolve='/file'))
        enter(mock_pathlib(is_file=True, resolve='/file'))
        enter(patch.object(cv2, 'imread', return_value=bgr_img.asarray()))
        enter(patch.object(cv2, 'VideoCapture', new_capture_5))
        mock_warning = enter(patch.object(logging.Logger, 'warning'))
        pipe.io.MultiplexPipeInput(
            [config.Source('video.mp4'), config.Source('image.jpg')],
            config.SystemConfig(),
            config.PipelineConfig(pipe_type='torch'),
            pipe.SourceIdAllocator(),
        )
        mock_warning.assert_called_once_with(
            'Not all input sources have the same format: VIDEO_FILE, IMAGE_FILES'
        )


def image_generator():
    for i in range(3):
        img = np.full((480, 640, 3), i, dtype=np.uint8)
        yield img


def types_image_generator():
    for i in range(3):
        img_data = np.full((480, 640, 3), i, dtype=np.uint8)
        yield types.Image.fromarray(img_data, types.ColorFormat.BGR)


def test_input_torch_data_source_np_generator():
    gen = image_generator()
    source = config.Source(gen)
    pipein = io.SinglePipeInput('torch', source)

    assert pipein.sources[0].type == config.SourceType.DATA_SOURCE
    assert pipein.sources[0].reader is gen
    assert pipein.number_of_frames == 0

    frames = list(itertools.islice(pipein.frame_generator(), 3))
    assert len(frames) == 3

    for i, frame in enumerate(frames):
        assert isinstance(frame.img, types.Image)
        np.testing.assert_equal(frame.img.asarray(), np.full((480, 640, 3), i, dtype=np.uint8))


def test_input_torch_data_source_types_image_generator():
    gen = types_image_generator()
    pipein = io.SinglePipeInput('torch', config.Source(gen))

    frames = list(itertools.islice(pipein.frame_generator(), 3))
    assert len(frames) == 3

    for i, frame in enumerate(frames):
        assert isinstance(frame.img, types.Image)
        np.testing.assert_equal(frame.img.asarray(), np.full((480, 640, 3), i, dtype=np.uint8))


def test_input_gst_data_source():
    gen = image_generator()
    pipein = io.SinglePipeInput('gst', config.Source(gen))

    gst_elements = _gen_input_gst(pipein)

    # Data source doesn't do anything special in GST, just checking it creates
    # successfully and the pipeline looks normal
    assert gst_elements[0]['instance'] == 'appsrc'
    assert 'name' not in gst_elements[-1]
    assert gst_elements[-1]['instance'] == 'axinplace'


def test_input_data_source_bad_generator():
    def bad_generator():
        yield np.zeros((480, 640, 3), dtype=np.uint8)
        yield None

    gen = bad_generator()
    pipein = io.SinglePipeInput('torch', config.Source(gen))

    first_frame = next(pipein.frame_generator())
    assert isinstance(first_frame.img, types.Image)

    with pytest.raises(
        RuntimeError, match="Failed to convert data source output None to axelera.types.Image"
    ):
        next(pipein.frame_generator())


def create_pipein(fps):
    m = Mock()
    m.stream_count.return_value = 1
    m.fps = fps
    return m


def do_writes(pipeout, *writes):
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
    with patch.object(cv2, 'imwrite') as mock_imwrite:
        pipeout = pipe.PipeOutput('', pipein)
        do_writes(pipeout, (bgr_img, 'unused'))
        assert mock_imwrite.called is False


def test_output_save_video_valid_input():
    pipein = create_pipein(30)
    pipeout = pipe.PipeOutput('somefile.mp4', pipein)
    with patch.object(cv2, 'VideoWriter') as mock_writer:
        mock_cvwriter = mock_writer.return_value
        do_writes(pipeout, (bgr_img, 'unused'))
        mock_writer.assert_called_once_with('somefile.mp4', MP4V, 30, (6, 4))
        np.testing.assert_array_equal(
            mock_cvwriter.write.call_args_list[0][0][0], bgr_img.asarray('BGR')
        )
        mock_cvwriter.release.assert_called_once()


@pytest.mark.parametrize(
    "speedometers_enabled,env_var_value,expected_registered_calls,expected_metric_calls,test_description",
    [
        (False, "0", 1, 0, "when speedometers are disabled with '0' (only registered tasks)"),
        (
            False,
            "false",
            1,
            0,
            "when speedometers are disabled with 'false' (only registered tasks)",
        ),
        (True, "1", 1, 1, "when speedometers are enabled with '1' (all tasks and metrics)"),
        (True, "true", 1, 1, "when speedometers are enabled with 'true' (all tasks and metrics)"),
        (True, None, 1, 1, "when env var not set (default: all tasks and metrics)"),
    ],
)
def test_output_save_video_speedometer_behavior(
    speedometers_enabled,
    env_var_value,
    expected_registered_calls,
    expected_metric_calls,
    test_description,
):
    """
    Test that the AXELERA_RENDER_SPEEDOMETERS_ON_SAVED_OUTPUTS environment variable controls
    which metadata elements have their draw method called:
    - "0", "false": Only registered tasks draw (AxTaskMeta instances)
    - "1", "true", or unset (default): All tasks and metrics draw (AxTaskMeta + TraceMetric instances)
    """
    task_draw_called = []
    metric_draw_called = []

    class MockTaskMeta(meta.AxTaskMeta):
        def draw(self, d):
            task_draw_called.append(d)

        def __len__(self):
            return 0

        def visit(self, func):
            func(self)

    class MockTraceMetric(inf_tracers.TraceMetric):
        def __init__(self, key="test_metric"):
            self.key = key
            self.title = "Test Metric"
            self.value = 42.0
            self.max_scale_value = 100.0
            self.unit = "fps"

        def draw(self, d):
            metric_draw_called.append(d)

        def visit(self, func):
            func(self)

        def __len__(self):
            return 0  # Return a valid length to avoid any iterations

    pipein = create_pipein(30)
    render_config = config.RenderConfig()
    registered_task_name = 'registered_task'
    metric_key = 'performance_metric'

    render_config.set_task(registered_task_name, True, True, True)
    pipeout = pipe.PipeOutput('somefile.mp4', pipein, render_config=render_config)

    original_env_value = os.environ.get('AXELERA_RENDER_SPEEDOMETERS_ON_SAVED_OUTPUTS', None)

    if env_var_value is None:
        # Remove the env var to test default behavior
        if 'AXELERA_RENDER_SPEEDOMETERS_ON_SAVED_OUTPUTS' in os.environ:
            del os.environ['AXELERA_RENDER_SPEEDOMETERS_ON_SAVED_OUTPUTS']
    else:
        # Set the env var to the specified value
        os.environ['AXELERA_RENDER_SPEEDOMETERS_ON_SAVED_OUTPUTS'] = env_var_value

    try:
        with patch.object(display_cv, 'CVDraw') as mock_cv_draw:
            mock_cv_draw.return_value = Mock()
            with patch.object(cv2, 'VideoWriter'):
                m = meta.AxMeta('test_meta')

                m.get_instance(registered_task_name, MockTaskMeta)

                # Create unregistered metric instance (this is what gets filtered out when speedometers are disabled)
                metric_instance = MockTraceMetric(metric_key)
                m.add_instance(metric_key, metric_instance)

                # Verify both instances were created and accessible
                assert (
                    registered_task_name in m
                ), f"Registered task {registered_task_name} not found in metadata"
                assert metric_key in m, f"Metric {metric_key} not found in metadata"

                # Verify the configuration is working as expected
                assert (
                    config.env.render_speedometers_on_saved_outputs == speedometers_enabled
                ), f"Expected config.env.render_speedometers_on_saved_outputs to be {speedometers_enabled} {test_description}"

                pipeout.sink(pipe.FrameResult(bgr_img, None, m, 0))
                pipeout.close_writer()

        # Verify the registered task's draw method was called the expected number of times
        assert len(task_draw_called) == expected_registered_calls, (
            f"Expected registered task's draw method to be called {expected_registered_calls} times {test_description}, "
            f"but was called {len(task_draw_called)} times"
        )

        # Verify the metric's draw method was called the expected number of times
        assert len(metric_draw_called) == expected_metric_calls, (
            f"Expected metric's draw method to be called {expected_metric_calls} times {test_description}, "
            f"but was called {len(metric_draw_called)} times"
        )

    finally:
        if original_env_value is None:
            if 'AXELERA_RENDER_SPEEDOMETERS_ON_SAVED_OUTPUTS' in os.environ:
                del os.environ['AXELERA_RENDER_SPEEDOMETERS_ON_SAVED_OUTPUTS']
        else:
            os.environ['AXELERA_RENDER_SPEEDOMETERS_ON_SAVED_OUTPUTS'] = original_env_value


def test_output_save_video_invalid_input():
    # e.g. input is images not video, fps/width/height cannot be retrieved from source
    # so we use the image size and assume fps is 30
    pipein = create_pipein(0)
    with patch.object(cv2, 'VideoWriter') as mock_writer:
        pipeout = pipe.PipeOutput('somefile.mp4', pipein)
        mock_cvwriter = mock_writer.return_value
        do_writes(pipeout, (bgr_img, 'unused'))
        mock_writer.assert_called_once_with('somefile.mp4', MP4V, 30, (6, 4))
        assert mock_cvwriter.write.call_count == 1
        exp_img = bgr_img.asarray('BGR')
        np.testing.assert_array_equal(mock_cvwriter.write.call_args[0][0], exp_img)
        mock_cvwriter.release.assert_called_once()


def test_output_save_images_img_id_specified():
    pipein = create_pipein(30)
    with patch.object(cv2, 'imwrite') as mock_write:
        with patch.object(Path, 'mkdir') as mock_mkdir:
            pipeout = pipe.PipeOutput('/path/to/images/', pipein)
            do_writes(pipeout, (bgr_img, 'someinput.jpg'))
            np_assert_called_once_with(
                mock_write, '/path/to/images/output_someinput.jpg', bgr_img.asarray()
            )
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_output_save_images_img_id_not_specified():
    pipein = create_pipein(30)
    with patch.object(cv2, 'imwrite') as mock_write:
        with patch.object(Path, 'mkdir') as mock_mkdir:
            pipeout = pipe.PipeOutput('/path/to/images/', pipein)
            do_writes(pipeout, (bgr_img, ''), (bgr_img, ''))
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
    with patch.object(cv2, 'imwrite') as mock_write:
        with patch.object(Path, 'mkdir') as mock_mkdir:
            pipeout = pipe.PipeOutput('/path/to/images/img%02d.jpg', pipein)
            do_writes(pipeout, (bgr_img, ''), (bgr_img, ''))
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
    with patch.object(cv2, 'imwrite') as mock_write:
        with patch.object(Path, 'mkdir') as mock_mkdir:
            pipeout = pipe.PipeOutput('/path/to/images/img.jpg', pipein)
            do_writes(pipeout, (bgr_img, ''))
            np_assert_called_once_with(mock_write, '/path/to/images/img.jpg', bgr_img.asarray())
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)


def test_output_save_images_no_pattern_more_than_once_image():
    pipein = create_pipein(30)
    with patch.object(cv2, 'imwrite'):
        with patch.object(Path, 'mkdir'):
            with pytest.raises(ValueError, match="containing '%d', then the input"):
                pipeout = pipe.PipeOutput('/path/to/images/img.jpg', pipein)
                do_writes(pipeout, (bgr_img, ''), (bgr_img, ''))


def test_output_save_single_image_exact_filename():
    """Test that single image output saves with exact specified filename."""
    pipein = create_pipein(30)
    with patch.object(cv2, 'imwrite') as mock_write:
        with patch.object(Path, 'mkdir') as mock_mkdir:
            pipeout = pipe.PipeOutput('myresult.jpg', pipein)
            do_writes(pipeout, (bgr_img, 'input.jpg'))
            # Should save as exact filename, not based on input filename
            np_assert_called_once_with(mock_write, 'myresult.jpg', bgr_img.asarray())
            mock_mkdir.assert_called_once_with(parents=True, exist_ok=True)

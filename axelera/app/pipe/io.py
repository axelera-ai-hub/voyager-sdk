# Copyright Axelera AI, 2024
# Construct application pipeline
from __future__ import annotations

import abc
import dataclasses
import enum
import errno
import os
from pathlib import Path
import re
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple
import urllib

import cv2

from axelera import types

from .. import config, display, display_cv, gst_builder, logging_utils, utils
from ..operators.utils import insert_color_convert

if TYPE_CHECKING:
    from . import frame_data
    from .. import inf_tracers, network

LOG = logging_utils.getLogger(__name__)

UnifyDataFormat = Callable[[Any], List[Dict[str, Any]]]


class BaseInput(abc.ABC):
    def __init__(self) -> None:
        self.location: str = ''
        '''Source of input. One of (device path|image path|video path|url|url|"measurement")'''

        self.format: str = ''
        '''Type of input.  One of (usb|csi|image|images|video|rtsp|hls|"dataset")'''

        self.number_of_frames: int = 0
        '''Number of frames in the input, or 0 if unbounded.'''

        self.batched_data_reformatter: Optional[UnifyDataFormat] = None
        '''For a dataset input this determines how to unify the data format.'''

    @abc.abstractmethod
    def create_generator(self):
        """frame generator for a BaseInput subclass"""

    @abc.abstractmethod
    def build_input_gst(self, gst: gst_builder.Builder, stream_idx: str):
        """generate gst pipeline representation"""

    @abc.abstractmethod
    def stream_count(self) -> int:
        """Number of input streams, 1 for most, N for MultiplexPipeInput."""


@dataclasses.dataclass
class ValidationComponents:
    dataloader: types.DataLoader
    reformatter: Callable
    evaluator: Optional[types.Evaluator] = None


def get_validation_components(
    network: Optional[network.AxNetwork],
    model_info: Optional[types.ModelInfo],
    task_name: str,
    data_root: str,
    dataset_split: str,
    batch_size: int,
    validation_settings: Optional[dict] = None,
) -> ValidationComponents:
    if network is None or model_info is None:
        raise RuntimeError("Either model_obj or both network and model_info must be provided")

    dataset_cfg = network.model_dataset_from_task(task_name)

    if 'data_dir_name' not in dataset_cfg:
        raise ValueError(
            f"To measure the accuracy of a model, you must provide a dataset. Please add 'data_dir_name' to the dataset section in your YAML file."
        )
    dataset_root = data_root / dataset_cfg['data_dir_name']

    def create_dataloader(obj):
        return obj.create_validation_data_loader(
            root=dataset_root, target_split=dataset_split, **dataset_cfg
        )

    try:
        with network.from_model_dir(model_info.name):
            the_obj = network.instantiate_data_adapter(model_info.name)
            dataloader = create_dataloader(the_obj)
    except types.SharedParameterError:
        LOG.info(
            "This data adapter requires parameters shared from the model. Attempting to construct the entire model."
        )
        try:
            with network.from_model_dir(model_info.name):
                the_obj = network.instantiate_model_for_deployment(network.tasks[0])
                dataloader = create_dataloader(the_obj)
        except Exception as e:
            LOG.error(f"Failed to evaluate this model in the current environment: {e}")
            raise

    reformatter = the_obj.reformat_for_validation

    try:
        if callable(the_obj.evaluator):
            evaluator = the_obj.evaluator(
                dataset_root=dataset_root,
                dataset_config=dataset_cfg,
                model_info=model_info,
                pair_validation=validation_settings.pop('pair_validation', False),
                custom_config=validation_settings,
            )
            if not isinstance(evaluator, types.Evaluator):
                raise TypeError(
                    f"Expected 'evaluator' to be a subclass of types.Evaluator, but got {type(evaluator).__name__}"
                )
            LOG.trace(f"Use evaluator with settings: {validation_settings}, {dataset_cfg}")
        else:
            raise TypeError(
                f"Expected 'evaluator' to be callable, but got {type(the_obj.evaluator).__name__}"
            )
    except NotImplementedError:
        evaluator = None
    except Exception as e:
        LOG.error(f"Failed to build evaluator {the_obj.evaluator}: {e}")
        raise

    return ValidationComponents(dataloader, reformatter, evaluator)


class DatasetInput(BaseInput):
    def __init__(
        self,
        data_loader: types.DataLoader,
        reformatter: Callable[[Any], List[types.FrameInput]],
        color_format: types.ColorFormat = types.ColorFormat.RGB,
        limit_frames: int = 0,
        hardware_caps: Optional[dict] = None,
    ):
        super().__init__()
        self.location = 'measurement'
        self.format = 'dataset'
        self.hardware_caps = hardware_caps or {}
        self.batched_data_reformatter = reformatter
        # TODO: handle batching: len(self.dataloader) * batch_size
        self.dataloader = (
            utils.LimitedLengthDataLoader(data_loader, limit_frames)
            if limit_frames
            else data_loader
        )
        self.number_of_frames = len(self.dataloader)
        self.color_format = color_format

    def create_generator(self):
        return self.dataloader

    def stream_count(self):
        return 1

    def build_input_gst(self, gst: gst_builder.Builder, stream_idx: str):
        assert stream_idx == '0'
        gst.appsrc(
            {
                'name': 'axelera_dataset_src',
                'is-live': True,
                'do-timestamp': True,
                'format': 3,  # GST_FORMAT_TIME
            }
        )
        gst.queue(name='queue_axelera_dataset')
        gst.axinplace(
            name=f'axinplace-addstreamid{stream_idx or 0}',
            lib='libinplace_addstreamid.so',
            mode='meta',
            options=f'stream_id:{stream_idx or 0}',
        )
        vaapi = gst.getconfig() is not None and gst.getconfig().vaapi
        opencl = gst.getconfig() is not None and gst.getconfig().opencl
        insert_color_convert(gst, vaapi, opencl, format=f'{self.color_format.name.lower()}a')


def server_reformatter(data):
    return [data]


class ServerInput(BaseInput):
    def __init__(
        self,
        limit_frames: int = 0,
        hardware_caps: Optional[dict] = None,
        server_loader: Optional[types.ServerLoader] = None,
    ):
        super().__init__()
        self.format = 'server'
        self.hardware_caps = hardware_caps or {}
        self.batched_data_reformatter = server_reformatter
        self.dataloader = server_loader
        self.number_of_frames = limit_frames

    def create_generator(self):
        return self.dataloader

    def stream_count(self):
        return 1

    def build_input_gst(self, gst: gst_builder.Builder, stream_idx: str):
        assert stream_idx == '0'
        gst.appsrc(
            {
                'name': 'axelera_server_src',
                'is-live': True,
                'do-timestamp': True,
                'format': 3,  # GST_FORMAT_TIME
            }
        )
        gst.queue(name='queue_axelera_dataset')
        gst.axinplace(
            name=f'axinplace-addstreamid{stream_idx or 0}',
            lib='libinplace_addstreamid.so',
            mode='meta',
            options=f'stream_id:{stream_idx or 0}',
        )


def _parse_livestream_location(location: str) -> Tuple[str, str, str]:
    # example: location=rtsp://id:pwd@10.40.130.221/axis-media/media.amp?audio=0&videocodec=jpeg&resolution=1280x960
    res = urllib.parse.urlparse(location)
    username = res.username or ''
    password = res.password or ''
    # urllib won't let you replace user/pass and leaves it in netloc, so instead:
    if '@' in res.netloc:
        res = res._replace(netloc=res.netloc.split('@', 1)[1])
    return username, password, urllib.parse.urlunparse(res)


def _is_youtube_url(url):
    """Check if the URL is a YouTube URL."""
    youtube_domains = ['youtube.com', 'youtu.be']
    return any(domain in url for domain in youtube_domains)


def _is_hls_url(url):
    """Check if the URL is an HLS URL."""
    return url.endswith('.m3u8')


def _get_youtube_stream_url(youtube_url):
    """Get the direct stream URL from a YouTube video URL."""
    import subprocess

    try:
        cmd = ["your_parser", youtube_url]
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode == 0:
            import json

            stream_data = json.loads(result.stdout)
            return stream_data.get("url", youtube_url)
        else:
            LOG.warning(
                f"Unable to parse {youtube_url} as a YouTube URL; proceeding as a standard URL."
            )
            return youtube_url
    except FileNotFoundError as e:
        raise logging_utils.UserError(
            f"""\
            Youtube url is not directly supported. You have to implement your parser
            to get the direct stream url if want to use youtube url as input.
            """
            + f'\n{e}'
        ) from None


def _determine_max_camera_resolution(cap: cv2.VideoCapture):
    # Get the resolution of the video stream to the maximum supported by the camera
    common_resolutions = [(640, 480), (1280, 720), (1920, 1080), (3840, 2160)]
    max_resolution = None
    for resolution in common_resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
        if cap.read()[0]:
            max_resolution = resolution
        elif max_resolution is None:
            raise RuntimeError("Unable to find a supported resolution")
    return max_resolution


def build_decodebin(
    gst: gst_builder.Builder, allow_hardware_codec, vaapi_enabled, stream_idx, downstream=None
):
    props = {}
    hw_decoder = allow_hardware_codec
    props = {
        'force-sw-decoders': not hw_decoder,
        'caps': 'video/x-raw(ANY)',
        'expose-all-streams': False,
    }
    downstream_name = downstream or f'queue_after_decodebin{stream_idx}'
    if gst.new_inference:
        props['name'] = f'queue_in{stream_idx}'
        props['connections'] = {'src_%u': f'axinplace-addstreamid{stream_idx or 0}.sink'}
    else:
        props['connections'] = {'src_%u': f'{downstream_name}.sink'}
    gst.decodebin(props)
    if downstream is None and not gst.new_inference:
        gst.queue(name=downstream_name)


def build_gst_repr_by_usb_cam(
    gst: gst_builder.Builder,
    location,
    codec,
    allow_hardware_codec,
    vaapi_enabled,
    stream_idx,
    width=0,
    height=0,
    fps=0,
):
    if codec not in ('yuyv', 'mjpg'):
        raise NotImplementedError(f"codec {codec} not supported in usb cam")

    gst.v4l2src(
        device=location,
    )
    dimensions = f'width={width},height={height}' if width and height else ''
    framerate = f'framerate={fps}/1' if fps else ''
    extras = ''.join(f',{x}' for x in [dimensions, framerate] if x)
    if codec == 'yuyv':
        gst.capsfilter(caps=f'video/x-raw,format=YUY2{extras}')
    else:  # codec == 'mjpg':
        gst.capsfilter(caps=f'image/jpeg{extras}')
        build_decodebin(gst, allow_hardware_codec, vaapi_enabled, stream_idx)


def build_gst_repr_by_rtsp(gst: gst_builder.Builder, location: str, stream_idx: str, latency=500):
    username, password, location = _parse_livestream_location(location)
    gst.rtspsrc(
        {
            'location': f"{location}",
            'user-id': f"{username}",
            'user-pw': f"{password}",
            'latency': latency,
            # decodebin to dynamically select the appropriate decoder
            # for h264, h265, mjpg, jpeg
            'connections': {'stream_%u': f'rtspcapsfilter{stream_idx}.sink'},
        }
    )
    gst.capsfilter(caps='application/x-rtp,media=video', name=f'rtspcapsfilter{stream_idx}')
    if not gst.new_inference:
        gst.queue(name=f'queue_in{stream_idx}')


@dataclasses.dataclass
class SourceInfo:
    format: str
    location: str
    width: int = 0
    height: int = 0
    images: Optional[List[Path]] = None
    usb_device: Optional[int] = None
    fps: int = 0


def parse_source(source: str) -> SourceInfo:
    if m_usb := re.match(r'usb(?::(\d+))?(?::(/dev/\w+))?(?::(\d+)x(\d+))?(?:@(\d+))?$', source):
        if m_usb.group(2):
            usb_device = None
            location = m_usb.group(2)
        else:
            usb_device = int(m_usb.group(1) or 0)
            location = f'/dev/video{usb_device}'

        width = int(m_usb.group(3)) if m_usb.group(3) else 0
        height = int(m_usb.group(4)) if m_usb.group(4) else 0
        fps = int(m_usb.group(5)) if m_usb.group(5) else 0
        return SourceInfo('usb', location, width, height, usb_device=usb_device, fps=fps)
    elif source.startswith("csi:") and len(source) > 4:
        return SourceInfo('csi', source[4:])
    elif source.startswith(('http://', 'https://')):
        if _is_youtube_url(source):
            stream_url = _get_youtube_stream_url(source)
        else:
            if not _is_hls_url(source):
                LOG.warning(
                    f"Unrecognised http/https format, assuming the format is HLS: {source}"
                )
            stream_url = source
        return SourceInfo('hls', stream_url)
    elif source.startswith(('rtsp://')):
        return SourceInfo('rtsp', source)
    elif m_fakevideo := re.match(r'fakevideo(?::(\d+)x(\d+))?$', source):
        if m_fakevideo.group(1) and m_fakevideo.group(2):
            width = int(m_fakevideo.group(1))
            height = int(m_fakevideo.group(2))
        else:  # default with 720P
            width = 1280
            height = 720
        return SourceInfo('video', 'fake', width, height)
    else:
        path = Path(source).resolve()
        images = None
        if path.is_file():
            format = utils.get_media_type(path)  # image or video
            if format == 'image':
                images = [str(path)]
        elif path.is_dir():  # search images under dir
            format = 'images'
            images = [x for x in path.iterdir() if x.suffix in utils.IMAGE_EXTENSIONS]
            if not images:
                raise RuntimeError(f"Failed to locate any images in {path}")
        else:
            raise FileNotFoundError(errno.ENOENT, os.strerror(errno.ENOENT), source)
        return SourceInfo(format, str(path), images=images)


def is_placeholder(format: str, location: str) -> bool:
    return format == 'rtsp' and location == 'rtsp://placeholder'


class PipeInput(BaseInput):
    def __init__(
        self,
        pipe_type: str,
        source: str,
        codec: str = 'mjpg',
        hardware_caps: config.HardwareCaps = config.HardwareCaps.NONE,
        allow_hardware_codec=True,
        color_format: types.ColorFormat = types.ColorFormat.RGB,
        rtsp_latency=500,
        specified_frame_rate=0,
    ):
        """
        Input arguments:
        pipe_type:    pipe route  (gst|torch|torch-aipu)
        source:  input argument      (usb|csi|file|rtsp|hls)

        Member variables:
        fps:      int  (frames per second)
        codec:    str  (h264|h265|mjpg)
        """
        super().__init__()
        self.hardware_caps = hardware_caps
        self.fps = 0
        self.codec = codec.lower()
        self.allow_hardware_codec = allow_hardware_codec
        self.color_format = color_format
        self.rtsp_latency = rtsp_latency
        info = parse_source(source)
        self.format = info.format
        self.location = info.location
        self.width, self.height = 0, 0
        if (self.format == 'video' and self.location == 'fake') or self.format == 'usb':
            self.width, self.height = info.width, info.height
        elif self.format in ('image', 'images'):
            self.images = info.images
            self.number_of_frames = len(self.images)
        self.specified_frame_rate = specified_frame_rate
        # Get video properties
        # For GST path, we need source w/h and fps to construct the pipeline
        if pipe_type == 'gst' and self.format == 'usb':
            self.fps = info.fps
        elif pipe_type == 'gst' and self.format in ('rtsp', 'hls'):
            pass  # don't interrogate the stream
        elif self.format == 'video' and self.location == 'fake':
            # a super high one, may not achieve due to different resolutions and patterns
            self.fps = 3000
        elif self.format not in ('image', 'images'):
            if (
                pipe_type == 'gst'
                and self.format == 'usb'
                and self.width != 0
                and self.height != 0
            ):
                LOG.info(
                    f"Use user specified resolution: {self.width}x{self.height} for usbcam {self.location}"
                )
            elif not is_placeholder(self.format, self.location):
                cap = cv2.VideoCapture(
                    self.location if info.usb_device is None else info.usb_device
                )
                if not cap.isOpened():
                    raise RuntimeError(
                        f"Failed to open video device: {self.format}, {self.location}"
                    )

                if self.format == 'usb':
                    if self.width == 0 and self.height == 0:
                        max_resolution = _determine_max_camera_resolution(cap)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_resolution[0])
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_resolution[1])
                        self.width, self.height = max_resolution

                self._get_video_attributes(cap)

                if pipe_type == 'gst':
                    cap.release()
                else:  # torch
                    self.cap = cap
        else:
            # this is here merely to give a nice early error for a bad path
            i = cv2.imread(str(self.images[0]))
            if i is None:
                raise RuntimeError(f"Failed to read image: {self.images[0]}")

    def stream_count(self):
        return 1

    def __del__(self):
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()

    def _get_video_attributes(self, cap: cv2.VideoCapture):
        try:
            self.number_of_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
            self.fps = int(cap.get(cv2.CAP_PROP_FPS))
            LOG.debug(f"FPS of {self.location}: {self.fps}")
        except Exception as e:
            LOG.error(f"Failed to get video capabilities: {e}")
            raise RuntimeError(f"Failed to get video capabilities: {e}") from None

    def create_generator(self):
        if hasattr(self, 'images'):
            LOG.debug("Create image generator from a series of images")
            for image in self.images:
                frame = cv2.imread(str(image))
                yield types.FrameInput(
                    img=types.Image.fromarray(frame, types.ColorFormat.BGR),
                    img_id=Path(image).name,
                    ground_truth=None,
                )
        elif self.format == 'video' and self.location == 'fake':
            x = cv2.imread(display.ICONS[192])
            x = cv2.resize(x, (self.width, self.height))
            while 1:
                yield types.FrameInput(
                    img=types.Image.fromarray(x, types.ColorFormat.BGR),
                    img_id='',
                    ground_truth=None,
                )
        else:
            LOG.debug("Create image generator from VideoCapture")
            try:
                while True:
                    ret, frame = self.cap.read()
                    if not ret:
                        break
                    yield types.FrameInput(
                        img=types.Image.fromarray(frame, types.ColorFormat.BGR),
                        img_id='',
                        ground_truth=None,
                    )
            finally:
                self.cap.release()

    def build_input_gst(self, gst: gst_builder.Builder, stream_idx: str):
        if self.format == 'usb':
            build_gst_repr_by_usb_cam(
                gst,
                self.location,
                self.codec,
                self.allow_hardware_codec,
                bool(self.hardware_caps.vaapi),
                stream_idx,
                width=self.width,
                height=self.height,
                fps=self.fps,
            )

        elif self.format == 'rtsp':
            build_gst_repr_by_rtsp(
                gst,
                self.location,
                stream_idx,
                latency=self.rtsp_latency,
            )

        elif self.format == 'hls':
            gst.playbin(uri=self.location)

        elif self.format in ('image', 'video'):
            if self.location == 'fake':
                gst.videotestsrc(is_live=True, pattern=0)
                gst.capsfilter(
                    caps=f'video/x-raw,width={self.width},height={self.height},format=NV12,framerate={self.fps}/1',
                )
            else:
                gst.filesrc(location=self.location)
        else:
            raise NotImplementedError(
                f"{self.format} format not supported in gst pipe ({self.location})"
            )

        if self.format == 'image':
            build_decodebin(gst, None, None, stream_idx)
            gst.axinplace(
                name=f'axinplace-addstreamid{stream_idx or 0}',
                lib='libinplace_addstreamid.so',
                mode='meta',
                options=f'stream_id:{stream_idx or 0}',
            )
            if gst.new_inference:
                insert_color_convert(
                    gst,
                    self.hardware_caps.vaapi,
                    self.hardware_caps.opencl,
                    f'{self.color_format.name.lower()}a',
                )
            return
        if self.format == 'video' and self.location != 'fake':
            if not gst.new_inference:
                gst.queue(name=f'queue_in{stream_idx}')

        if self.format not in ['usb', 'hls']:
            build_decodebin(
                gst, self.allow_hardware_codec, bool(self.hardware_caps.vaapi), stream_idx
            )

        gst.axinplace(
            name=f'axinplace-addstreamid{stream_idx or 0}',
            lib='libinplace_addstreamid.so',
            mode='meta',
            options=f'stream_id:{stream_idx or 0}',
        )
        if self.specified_frame_rate:
            gst.videorate()
            gst.capsfilter(caps=f'video/x-raw,framerate={self.specified_frame_rate}/1')

        if method := config.env.videoflip:
            gst.videoflip(method=method)

        if gst.new_inference:
            insert_color_convert(
                gst,
                self.hardware_caps.vaapi,
                self.hardware_caps.opencl,
                f'{self.color_format.name.lower()}a',
            )


def _all_same(things):
    return len(things) == 1 or all(x == things[0] for x in things[1:])


class MultiplexPipeInput(BaseInput):
    def __init__(
        self,
        pipe_type: str,
        sources: List[str],
        codecs: List[str] = [],
        hardware_caps: config.HardwareCaps = config.HardwareCaps(),
        allow_hardware_codec=True,
        color_format: types.ColorFormat = types.ColorFormat.RGB,
        rtsp_latency=500,
        specified_frame_rate=0,
    ):
        super().__init__()
        if len(codecs) < len(sources):
            codecs = codecs + ['mjpg'] * (len(sources) - len(codecs))
        self.inputs = [
            PipeInput(
                pipe_type,
                source,
                codec,
                hardware_caps,
                allow_hardware_codec,
                color_format=color_format,
                rtsp_latency=rtsp_latency,
                specified_frame_rate=specified_frame_rate,
            )
            for source, codec in zip(sources, codecs)
        ]

        formats = [input.format for input in self.inputs]
        if not _all_same(formats):
            LOG.warning(f'Not all input sources have the same format: {formats}')
        codecs = [input.codec for input in self.inputs]
        if not _all_same(codecs):
            raise ValueError(f'All codecs must be the same : {codecs}')
        self.format = formats[0]
        self.codec = codecs[0]
        # TODO we currently and maybe always restrict to the shortest stream,
        num_frames = [i.number_of_frames for i in self.inputs]
        self.number_of_frames = 0 if any(n == 0 for n in num_frames) else sum(num_frames)
        self.batched_data_reformatter = None  # we do not support multiplex dataset.

    def build_input_gst(self, gst: gst_builder.Builder, stream_idx: str):
        idx = int(stream_idx) if stream_idx else 0
        self.inputs[idx].build_input_gst(gst, stream_idx)

    def stream_count(self):
        return len(self.inputs)

    def create_generator(self):
        active = {sid: input.create_generator() for sid, input in enumerate(self.inputs)}
        while 1:
            dead = []
            for stream_id, gen in active.items():
                try:
                    inp = next(gen)
                except StopIteration:
                    dead.append(stream_id)
                else:
                    inp.stream_id = stream_id
                    yield inp
            for stream_id in dead:
                del active[stream_id]
            if not active:
                return


class _OutputMode(enum.Enum):
    NONE = enum.auto()
    VIDEO = enum.auto()
    IMAGES = enum.auto()


def _resolve_output_index(pattern: Path, index: int):
    return str(pattern) % (index,)


def _determine_output_mode(location: str):
    if not location:
        return _OutputMode.NONE, location

    path = Path(location)
    if location.endswith('/') or path.is_dir():
        return _OutputMode.IMAGES, (path / 'output_%05d.jpg')

    if '%' in path.name:
        _resolve_output_index(path, 0)  # ensure it can be expanded
        if path.name.endswith('.mp4'):
            return _OutputMode.VIDEO, path
        return _OutputMode.IMAGES, path

    suffix_type = utils.get_media_type(path.name)
    if suffix_type == 'image':
        return _OutputMode.IMAGES, path

    return _OutputMode.VIDEO, path


class _NullWriter:
    def write(self, image: types.Image, input_filename: str, stream_id: int):
        del input_filename
        del image
        del stream_id

    def release(self):
        pass


class _ImageWriter(_NullWriter):
    def __init__(self, location: Path):
        self._location = location
        self._index = 0

    def write(self, image: types.Image, input_filename: str, stream_id: int):
        del stream_id
        if input_filename:
            filename = Path(input_filename)
            stem, suffix = filename.stem, filename.suffix
            name = str(self._location.parent.joinpath(f'output_{stem}{suffix}'))
        elif '%' in self._location.name:
            name = _resolve_output_index(self._location, self._index)
        elif self._index == 0:
            name = str(self._location)
        else:
            raise ValueError(
                "If output is not a directory or a path pattern containing '%d', "
                "then the input must be a single image"
            )

        LOG.info(f"Save the result image to {name}")
        bgr = image.asarray(types.ColorFormat.BGR)
        cv2.imwrite(name, bgr)
        self._index += 1


class _VideoWriter:
    def __init__(self, location, input):
        if isinstance(input, MultiplexPipeInput):
            self._location = [_resolve_output_index(location, i) for i in range(len(input.inputs))]
            self._fps = [i.fps for i in input.inputs]
        else:
            self._location = [location]
            self._fps = [input.fps]
        self._writer = None

    def write(self, image: types.Image, input_filename: str, stream_id: int):
        del input_filename
        if not self._writer:
            self._writer = [
                cv2.VideoWriter(
                    str(location),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps or 30,
                    image.size,
                )
                for location, fps in zip(self._location, self._fps)
            ]
        bgr = image.asarray(types.ColorFormat.BGR)
        self._writer[stream_id].write(bgr)

    def release(self):
        for w in self._writer:
            w.release()


@dataclasses.dataclass
class PipeOutput:
    save_output: str = ''
    '''If given then the output is saved to the specified location.

    Location may be:

        1. An existing directory, in which case the output is saved as images of the form
        'image_00000.jpg', 'image_00001.jpg', etc.

        2. A string ending in `/` in which case the path is created if necessary. The
        behaviour is otherwise the same as 1.

        3. A string containing a %format specifier, e.g. `output/img_%05d.jpg` in which
        case the images are output as `output/img_00000.jpg`, `output/img_00001.jpg`, etc.

        4. A string ending in `.mp4` (or another video format extension) in which case the output
        is saved as a video file. The containing directory will be created if necessary.

        5. A string ending in `.jpg` or `.png` in which case the output is saved as a
        single image file. The containing directory will be created if necessary. In this
        case the input must also be a single image and an error will be raised if the input stream
        includes more than once image.

    Output images/video will include the inference results overlaid on the original input image.
    '''

    tracers: List[inf_tracers.Tracer] = dataclasses.field(default_factory=list, repr=False)

    def __post_init__(self):
        self._mode, self._parsed_location = _determine_output_mode(self.save_output)
        self._writer = _NullWriter()
        self._results = []

    def initialize_writer(self, input: PipeInput):
        if self._mode == _OutputMode.IMAGES:
            self._parsed_location.parent.mkdir(parents=True, exist_ok=True)
            self._writer = _ImageWriter(self._parsed_location)

        elif self._mode == _OutputMode.VIDEO:
            self._parsed_location.parent.mkdir(parents=True, exist_ok=True)
            self._writer = _VideoWriter(self._parsed_location, input)

    def close_writer(self):
        self._writer.release()

    @property
    def result(self):
        '''Return the most recent result.'''
        return self._results[-1]

    def sink(self, frame_result: frame_data.FrameResult):
        # At present we only keep the current frame's results
        frame_result.sink_timestamp = time.time()
        self._results[:] = [frame_result]
        meta = frame_result.meta
        if meta and self.tracers:
            for tracer in self.tracers:
                tracer.update(frame_result)
            if frame_result.stream_id == 0:
                for tracer in self.tracers:
                    for m in tracer.get_metrics():
                        meta.add_instance(m.key, m)

        image = frame_result.image
        if meta and self._mode != _OutputMode.NONE:
            image = image.copy()
            draw = display_cv.CVDraw(image)
            for m in meta.values():
                m.visit(lambda m: m.draw(draw))
            draw.draw()
        self._writer.write(image, str(meta.image_id), frame_result.stream_id)

    def build_output_gst(self, gst: gst_builder.Builder, stream_count: int):
        qsize = gst.default_queue_max_size_buffers
        gst.appsink({'max-buffers': qsize, 'drop': False, 'sync': False})

# Copyright Axelera AI, 2024
# Construct application pipeline
from __future__ import annotations

import abc
import dataclasses
import enum
import os
from pathlib import Path
import shutil
import subprocess
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Optional, Tuple
import urllib

import cv2
import numpy as np

from axelera import types

from .. import config, display, display_cv, gst_builder, logging_utils, utils

if TYPE_CHECKING:
    from . import base, frame_data
    from .. import network

LOG = logging_utils.getLogger(__name__)

UnifyDataFormat = Callable[[Any], List[Dict[str, Any]]]
FrameInputGenerator = Generator[types.FrameInput, None, None]


class PipeInput(abc.ABC):
    def __init__(self) -> None:
        self.number_of_frames: int = 0
        '''Number of frames in the input, or 0 if unbounded.'''

        self.batched_data_reformatter: Optional[UnifyDataFormat] = None
        '''For a dataset input this determines how to unify the data format.'''

    @property
    def sources(self) -> list[config.Source]:
        '''Sources of input.'''
        raise NotImplementedError()

    @abc.abstractmethod
    def frame_generator(self) -> FrameInputGenerator:
        """Generates input data to a torch/torch-aipu pipe or dataset pipes in gst.

        This function should yield FrameInputs with the stream_id set.
        """

    @abc.abstractmethod
    def build_input_gst(self, gst: gst_builder.Builder, stream_idx: str):
        """Build the gst pipeline representation for the input element."""

    @abc.abstractmethod
    def stream_count(self) -> int:
        """Number of input streams, 1 for most, N for MultiplexPipeInput."""

    def add_source(
        self,
        source: config.Source,
        source_id: int,
        rtsp_latency: int | None = None,
        specified_frame_rate: int | None = None,
    ) -> PipeInput:
        """Add a new source to the input.  This is used for multiplex input."""
        del source
        del source_id
        del rtsp_latency
        del specified_frame_rate
        raise NotImplementedError(f"{type(self).__name__} does not support add_source()")

    @property
    def inputs(self) -> dict[int, PipeInput]:
        """Iterate over the inputs.  This is used for multiplex input."""
        return {self._sid: self}


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


def _build_data_loader(gst):
    gst.appsrc(
        {
            'name': 'axelera_dataset_src',
            'is-live': True,
            'do-timestamp': True,
            'format': 3,  # GST_FORMAT_TIME
        }
    )
    gst.queue(name='queue_axelera_dataset')


class DatasetInput(PipeInput):
    def __init__(
        self,
        src: config.Source,
        data_loader: types.DataLoader,
        reformatter: Callable[[Any], List[types.FrameInput]],
        limit_frames: int,
        hardware_caps: Optional[dict],
        stream_id: int,
    ):
        super().__init__()
        self._hwcaps = hardware_caps or {}
        self._src = src
        self.batched_data_reformatter = reformatter
        self.dataloader = (
            utils.LimitedLengthDataLoader(data_loader, limit_frames)
            if limit_frames
            else data_loader
        )
        self.number_of_frames = len(self.dataloader)
        self._sid = stream_id

    @property
    def sources(self) -> list[config.Source]:
        '''Sources of input'''
        return [self._src]

    def frame_generator(self) -> FrameInputGenerator:
        for frame in self.dataloader:
            yield frame

    def stream_count(self):
        return 1

    def build_input_gst(self, gst: gst_builder.Builder, stream_idx: str):
        assert stream_idx == '0', repr(stream_idx)
        _build_data_loader(gst)
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


def _set_max_camera_resolution(cap: cv2.VideoCapture):
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
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, max_resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, max_resolution[1])
    return max_resolution


def build_decodebin(gst: gst_builder.Builder, allow_hardware_codec, stream_idx):
    props = {}
    hw_decoder = allow_hardware_codec
    props = {
        'force-sw-decoders': not hw_decoder,
        # Keep caps aligned with existing golden pipelines/tests
        'caps': 'video/x-raw(ANY)',
        'expose-all-streams': False,
    }
    props['connections'] = {'src_%u': f'decodebin-link{stream_idx or 0}.sink'}
    gst.decodebin(props)


def build_jetson_video_decode(gst: gst_builder.Builder, stream_idx):
    """Build a Jetson-optimized video decode pipeline for MP4/MOV files.
    
    NVIDIA nvv4l2decoder requires byte-stream format, but MP4 files contain
    AVC1/HVC1 format. We need to explicitly insert h264parse/h265parse to
    convert the stream format, which decodebin doesn't do automatically.
    """
    # qtdemux extracts video stream from MP4/MOV container
    # video_%u is a dynamic pad - the SDK will use pad-added callback to link it
    gst.qtdemux(connections={'video_%u': f'videoparser{stream_idx or 0}.sink'})
    
    # h264parse converts from AVC1 to byte-stream format required by nvv4l2decoder
    # Note: For H.265 videos, this would need h265parse instead
    gst.h264parse(name=f'videoparser{stream_idx or 0}')
    
    # NVIDIA hardware decoder
    gst.nvv4l2decoder(connections={'src': f'decodebin-link{stream_idx or 0}.sink'})


def _build_gst_usb(gst: gst_builder.Builder, src: config.Source, allow_hardware_codec, stream_idx):
    codec = src.codec or 'mjpg'  #
    codecs = {
        'mjpg': 'image/jpeg',
        'image/jpeg': 'image/jpeg',
        'yuyv': 'video/x-raw,format=YUY2',
    }
    if codec not in codecs:
        raise NotImplementedError(f"codec {codec!r} not supported in usb cam")
    caps = codecs[codec]

    loc = f"/dev/video{src.location!s}" if re.match(r'\d+$', src.location) else src.location
    gst.v4l2src(device=loc)
    dimensions = f'width={src.width},height={src.height}' if src.width and src.height else ''
    framerate = f'framerate={src.fps}/1' if src.fps else ''
    extras = ''.join(f',{x}' for x in [dimensions, framerate] if x)
    LOG.info("Using v2l caps=%s", f'{caps}{extras}')
    gst.capsfilter(caps=f'{caps}{extras}')
    if 'jpeg' in caps:
        build_decodebin(gst, allow_hardware_codec, stream_idx)


def _build_gst_rtsp(gst: gst_builder.Builder, location: str, stream_idx: str, latency=500):
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


def _usb_device_path(src: config.Source) -> str:
    # TODO Win32
    return f"/dev/video{src.location}" if re.match(r'\d+$', src.location) else src.location


def _open_cap(src: config.Source) -> cv2.VideoCapture:
    c = cv2.VideoCapture(int(src.location) if re.match(r'\d+$', src.location) else src.location)
    if not c.isOpened():
        raise RuntimeError(f"Failed to open video device: {src.location}")
    return c


class SinglePipeInput(PipeInput):
    def __init__(
        self,
        pipe_type: str,
        src: config.Source,
        hardware_caps: config.HardwareCaps = config.HardwareCaps.NONE,
        allow_hardware_codec=True,
        rtsp_latency=500,
        specified_frame_rate=0,
        source_id: int = 0,
    ):
        super().__init__()
        self._cap = None
        self._hwcaps = hardware_caps
        self._allow_hardware_codec = allow_hardware_codec
        self._rtsp_latency = rtsp_latency
        self._specified_frame_rate = specified_frame_rate
        self._src = src
        self._sid = source_id
        t = self._src.type
        if t == config.SourceType.IMAGE_FILES:
            self.number_of_frames = len(self._src.images)
            self.batched_data_reformatter = lambda data: [data]
            i = cv2.imread(str(self._src.images[0]))
            if i is None:
                raise RuntimeError(f"Failed to read image: {self._src.images[0]}")
        elif t == config.SourceType.DATA_SOURCE:
            self.number_of_frames = 0
            self.batched_data_reformatter = lambda data: [data]
        elif pipe_type == 'gst' and t == config.SourceType.VIDEO_FILE:
            if not os.access(self._src.location, os.F_OK | os.R_OK | os.W_OK):
                raise RuntimeError(f"Cannot access video file: {self._src.location}")
        elif pipe_type == 'gst' and t == config.SourceType.USB:
            # with gst we want to do a very light touch, just check the device is there
            path = _usb_device_path(self._src)
            if path and not os.access(path, os.F_OK | os.R_OK | os.W_OK):
                raise RuntimeError(f"Cannot access device at {self._src.location}")
        elif pipe_type != 'gst' and t != config.SourceType.FAKE_VIDEO:
            # not gst and not fake, we will need to open the device
            self._cap = _open_cap(self._src)
            if t == config.SourceType.USB and (self._src.width == 0 and self._src.height == 0):
                self._src.width, self._src.height = _set_max_camera_resolution(self._cap)
            self._get_video_attributes(self._cap)

    @property
    def source_id(self) -> int:
        """The source ID of this input."""
        return self._sid

    @property
    def sources(self) -> list[config.Source]:
        '''Sources of input'''
        return [self._src]

    def stream_count(self):
        return 1

    def __del__(self):
        if c := getattr(self, '_cap', None):
            c.release()

    @property
    def fps(self) -> int:
        return self._src.fps

    def _get_video_attributes(self, cap: cv2.VideoCapture):
        try:
            self.number_of_frames = max(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)), 0)
            self._src.fps = int(cap.get(cv2.CAP_PROP_FPS))
            LOG.debug(f"FPS of {self._src.location}: {self._src.fps}")
        except Exception as e:
            LOG.error(f"Failed to get video capabilities: {e}")
            raise RuntimeError(f"Failed to get video capabilities: {e}") from None

    def frame_generator(self) -> FrameInputGenerator:
        if self._src.type == config.SourceType.IMAGE_FILES:
            LOG.debug("Create image generator from a series of images")
            for image in self._src.images:
                if (frame := cv2.imread(str(image))) is None:
                    raise RuntimeError(f"Failed to read image: {image}")
                img = types.Image.fromarray(frame, types.ColorFormat.BGR)
                img_id = os.path.relpath(str(image), os.getcwd())
                yield types.FrameInput(img=img, img_id=img_id, stream_id=self._sid)

        elif self._src.type == config.SourceType.DATA_SOURCE:
            LOG.debug("Create image generator from a python generator of images")
            for frame in self._src.reader:
                try:
                    img = types.Image.fromany(frame, types.ColorFormat.BGR)
                except TypeError:
                    raise RuntimeError(
                        f"Failed to convert data source output {frame} to axelera.types.Image"
                    )
                yield types.FrameInput(img=img, stream_id=self._sid)

        elif self._src.type == config.SourceType.FAKE_VIDEO:
            x = cv2.imread(display.ICONS[192])
            x = cv2.resize(x, (self._src.width, self._src.height))
            while 1:
                img = types.Image.fromarray(x, types.ColorFormat.BGR)
                yield types.FrameInput(img=img, stream_id=self._sid)

        else:
            LOG.debug("Create image generator from VideoCapture")
            try:
                while True:
                    ret, frame = self._cap.read()
                    if not ret:
                        break
                    img = types.Image.fromarray(frame, types.ColorFormat.BGR)
                    yield types.FrameInput(img=img, stream_id=self._sid)
            finally:
                self._cap.release()

    def build_input_gst(self, gst: gst_builder.Builder, stream_idx: str):
        requires_decodebin = False
        if self._src.type == config.SourceType.USB:
            _build_gst_usb(gst, self._src, self._allow_hardware_codec, stream_idx)
        elif self._src.type == config.SourceType.RTSP:
            _build_gst_rtsp(gst, self._src.location, stream_idx, latency=self._rtsp_latency)
            requires_decodebin = True
        elif self._src.type == config.SourceType.HLS:
            gst.playbin(uri=self._src.location)
        elif self._src.type == config.SourceType.FAKE_VIDEO:
            gst.videotestsrc(is_live=True, pattern=0)
            dims = f"width={self._src.width},height={self._src.height}"
            gst.capsfilter(caps=f'video/x-raw,{dims},format=NV12,framerate={self._src.fps}/1')
            requires_decodebin = True  # but does, it really?
        elif self._src.type in (config.SourceType.IMAGE_FILES, config.SourceType.DATA_SOURCE):
            _build_data_loader(gst)
            gst.images = [os.path.relpath(str(i), os.getcwd()) for i in self._src.images]
    elif self._src.type == config.SourceType.VIDEO_FILE:
            # Optionally build an explicit demux/parse/decode chain for file sources to avoid
            # decodebin negotiation issues seen on Jetson/NVIDIA pipelines.
            # Enable with AXELERA_GST_EXPLICIT_PARSE=1
            use_explicit_parse = str(os.getenv('AXELERA_GST_EXPLICIT_PARSE', '0')).lower() in (
                '1',
                'true',
                'yes',
                'on',
            )
            gst.filesrc(location=self._src.location)
            if use_explicit_parse:
                # Detect codec to decide whether to use explicit parse (h264/h265) or fall back to decodebin
                codec = ''
                if shutil.which('ffprobe'):
                    try:
                        out = subprocess.run(
                            [
                                'ffprobe',
                                '-v',
                                'error',
                                '-select_streams',
                                'v:0',
                                '-show_entries',
                                'stream=codec_name',
                                '-of',
                                'default=nw=1:nk=1',
                                self._src.location,
                            ],
                            check=True,
                            capture_output=True,
                            text=True,
                        )
                        codec = (out.stdout or '').strip().lower()
                    except Exception:
                        codec = ''

                is_h264 = codec in ('h264', 'avc1')
                is_h265 = codec in ('hevc', 'h265', 'hev1', 'hvc1')

                if not (is_h264 or is_h265):
                    # Unknown/unsupported for explicit path – fall back to legacy decodebin
                    requires_decodebin = True
                else:
                    # filesrc → qtdemux (video_%u) → h26x parse → decoder → decodebin-linkX
                    gst.qtdemux(connections={'video_%u': f'videoparser{stream_idx or 0}.sink'})
                    if is_h265:
                        gst.h265parse(
                            name=f'videoparser{stream_idx or 0}',
                            connections={'src': f'videodecoder{stream_idx or 0}.sink'},
                        )
                    else:
                        gst.h264parse(
                            name=f'videoparser{stream_idx or 0}',
                            connections={'src': f'videodecoder{stream_idx or 0}.sink'},
                        )

                    # Choose decoder: prefer hardware if enabled, otherwise software
                    if self._allow_hardware_codec:
                        gst.nvv4l2decoder(name=f'videodecoder{stream_idx or 0}')
                        gst.nvvidconv()
                        gst.capsfilter(
                            caps='video/x-raw,format=NV12',
                            connections={'src': f'decodebin-link{stream_idx or 0}.sink'},
                        )
                    else:
                        if is_h265:
                            gst.avdec_h265(
                                name=f'videodecoder{stream_idx or 0}',
                                connections={'src': f'decodebin-link{stream_idx or 0}.sink'},
                            )
                        else:
                            gst.avdec_h264(
                                name=f'videodecoder{stream_idx or 0}',
                                connections={'src': f'decodebin-link{stream_idx or 0}.sink'},
                            )
                    requires_decodebin = False
            else:
                # Legacy/default behaviour: rely on decodebin auto-plugging
                requires_decodebin = True
        else:
            raise NotImplementedError(f"{self._src.type} format not supported in gst pipe")

        if requires_decodebin:
            build_decodebin(gst, self._allow_hardware_codec, stream_idx)

        # Ensure elements are named so that the upstream connects with the correct one
        decodebin_link = f'decodebin-link{stream_idx or 0}'
        axinplace_name = decodebin_link
        if self._specified_frame_rate:
            gst.videorate(name=decodebin_link)
            gst.capsfilter(caps=f'video/x-raw,framerate={self._specified_frame_rate}/1')
            axinplace_name = f'axinplace-addstreamid{stream_idx or 0}'

        gst.axinplace(
            name=axinplace_name,
            lib='libinplace_addstreamid.so',
            mode='meta',
            options=f'stream_id:{stream_idx or 0}',
        )


def _all_same(things):
    return len(things) == 1 or all(x == things[0] for x in things[1:])


class MultiplexPipeInput(PipeInput):
    def __init__(
        self,
        sources: list[config.Source],
        system_config: config.SystemConfig,
        pipeline_config: config.PipelineConfig,
        id_allocator: base.SourceIdAllocator,
    ):
        super().__init__()
        self._pipe_type = pipeline_config.pipe_type
        self._hwcaps = system_config.hardware_caps
        self._allow_hardware_codec = system_config.allow_hardware_codec
        self._rtsp_latency = pipeline_config.rtsp_latency
        self._specified_frame_rate = pipeline_config.specified_frame_rate
        self._inputs = [
            SinglePipeInput(
                self._pipe_type,
                source,
                self._hwcaps,
                self._allow_hardware_codec,
                self._rtsp_latency,
                self._specified_frame_rate,
                id_allocator.allocate(),
            )
            for source in sources
        ]

        types = [input.sources[0].type for input in self._inputs]
        if not _all_same(types):
            stypes = ', '.join(t.name for t in types)
            LOG.warning(f'Not all input sources have the same format: {stypes}')
        # TODO we currently and maybe always restrict to the shortest stream,
        num_frames = [i.number_of_frames for i in self._inputs]
        self.number_of_frames = 0 if any(n == 0 for n in num_frames) else sum(num_frames)
        self.batched_data_reformatter = self._inputs[0].batched_data_reformatter

    @property
    def sources(self) -> list[config.Source]:
        '''Sources of input'''
        return [i.sources[0] for i in self._inputs]

    @property
    def inputs(self) -> dict[int, PipeInput]:
        """Return dict of source_id: PipeInput for all the inputs"""
        return {i.source_id: i for i in self._inputs}

    def add_source(
        self,
        source: config.Source,
        source_id: int,
        rtsp_latency: int | None = None,
        specified_frame_rate: int | None = None,
    ) -> PipeInput:
        if rtsp_latency is None:
            rtsp_latency = self._rtsp_latency
        if specified_frame_rate is None:
            specified_frame_rate = self._specified_frame_rate
        new_pipe = SinglePipeInput(
            self._pipe_type,
            source,
            self._hwcaps,
            self._allow_hardware_codec,
            rtsp_latency,
            specified_frame_rate,
            source_id,
        )
        self._inputs.append(new_pipe)
        return new_pipe

    def remove_source(self, source: str) -> None:
        idx = -1
        for i, input in enumerate(self._inputs):
            if input.location == source:
                idx = i
                break
        if idx != -1:
            del self._inputs[idx]

    def build_input_gst(self, gst: gst_builder.Builder, stream_idx: str):
        raise NotImplementedError(f"MultiplexPipeInput does not support build_input_gst()")

    def stream_count(self):
        return len(self._inputs)

    @property
    def fps(self):
        fps = [input.fps for input in self._inputs]
        if not _all_same(fps):
            LOG.warning(f'Not all input sources have the same fps: {fps}')
        return min(fps)

    def frame_generator(self) -> FrameInputGenerator:
        active = {i.source_id: i.frame_generator() for i in self._inputs}
        while 1:
            dead = []
            for sid, gen in active.items():
                try:
                    inp = next(gen)
                except StopIteration:
                    dead.append(sid)
                else:
                    yield inp
            for sid in dead:
                del active[sid]
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

        is_directory_pattern = (
            '%' in self._location.name
            and self._location.name.startswith('output_')
            and self._location.name.endswith('.jpg')
        )

        if is_directory_pattern and input_filename:
            # Directory output with input filename: use input filename
            filename = Path(input_filename)
            stem, suffix = filename.stem, filename.suffix
            name = str(self._location.parent.joinpath(f'output_{stem}{suffix}'))
        elif '%' in self._location.name:
            # User pattern or directory pattern without input filename: use pattern
            name = _resolve_output_index(self._location, self._index)
        elif self._index == 0:
            # Single image output: use exact specified filename
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
        if input.stream_count() > 1:
            self._location = [
                _resolve_output_index(location, i) for i in range(input.stream_count())
            ]
            self._fps = [i.fps for _, i in sorted(input.inputs.items())]
        else:
            self._location = [location]
            self._fps = [input.fps]
        self._writer: list[cv2.VideoWriter] = []

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


class PipeOutput:
    '''Used to save the rendered output of a pipe.

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

    def __init__(
        self,
        save_output: str,
        input: PipeInput,
        render_config: config.RenderConfig = config.RenderConfig(),
    ):
        self._mode, parsed_location = _determine_output_mode(save_output)
        self._render_config = render_config
        if self._mode == _OutputMode.IMAGES:
            parsed_location.parent.mkdir(parents=True, exist_ok=True)
            self._writer = _ImageWriter(parsed_location)

        elif self._mode == _OutputMode.VIDEO:
            parsed_location.parent.mkdir(parents=True, exist_ok=True)
            self._writer = _VideoWriter(parsed_location, input)
        else:
            self._writer = _NullWriter()

    def close_writer(self):
        self._writer.release()

    def sink(self, frame_result: frame_data.FrameResult):
        image = frame_result.image
        meta = frame_result.meta
        if frame_result.meta and self._mode != _OutputMode.NONE:
            image = image.copy()
            draw = display_cv.CVDraw(image, [])
            if config.env.render_speedometers_on_saved_outputs:
                for m in meta.values():
                    m.visit(lambda m: m.draw(draw))
            else:
                for key in self._render_config.keys():
                    if key in meta:
                        meta[key].visit(lambda m: m.draw(draw))
            draw.draw()
        self._writer.write(image, str(meta.image_id), frame_result.stream_id)

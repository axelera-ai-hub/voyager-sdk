# Copyright Axelera AI, 2025
from __future__ import annotations

import re

from . import config


def element(f):
    def wrapper(self, props={}, **kwargs):
        allprops = dict(instance=f.__name__, **kwargs)
        allprops.update(props)
        self.append(allprops)

    return wrapper


class Builder(list):
    def __init__(self, hw_config=None, default_queue_max_size_buffers=16):
        self.hw_config = hw_config
        self.default_queue_max_size_buffers = default_queue_max_size_buffers

    @property
    def new_inference(self) -> bool:
        return False

    def getconfig(self):
        return self.hw_config

    @element
    def identity(self, props={}, **kwargs) -> None:
        ...

    @element
    def appsink(self, props={}, **kwargs) -> None:
        ...

    @element
    def appsrc(self, props={}, **kwargs) -> None:
        ...

    @element
    def aspectratiocrop(self, props={}, **kwargs) -> None:
        ...

    @element
    def axinplace(self, props={}, **kwargs) -> None:
        ...

    @element
    def axtransform(self, props={}, **kwargs) -> None:
        ...

    @element
    def capsfilter(self, props={}, **kwargs) -> None:
        ...

    @element
    def clpreproc(self, props={}, **kwargs) -> None:
        ...

    @element
    def compositor(self, props={}, **kwargs) -> None:
        ...

    @element
    def decode_muxer(self, props={}, **kwargs) -> None:
        ...

    @element
    def decodebin(self, props={}, **kwargs) -> None:
        ...

    @element
    def distributor(self, props={}, **kwargs) -> None:
        ...

    @element
    def fakesink(self, props={}, **kwargs) -> None:
        ...

    @element
    def filesink(self, props={}, **kwargs) -> None:
        ...

    @element
    def filesrc(self, props={}, **kwargs) -> None:
        ...

    @element
    def h264parse(self, props={}, **kwargs) -> None:
        ...

    @element
    def h265parse(self, props={}, **kwargs) -> None:
        ...

    @element
    def jpegdec(self, props={}, **kwargs) -> None:
        ...

    @element
    def mp4mux(self, props={}, **kwargs) -> None:
        ...

    @element
    def multifilesink(self, props={}, **kwargs) -> None:
        ...

    @element
    def multifilesrc(self, props={}, **kwargs) -> None:
        ...

    @element
    def nvv4l2decoder(self, props={}, **kwargs) -> None:
        ...

    @element
    def nvv4l2h264enc(self, props={}, **kwargs) -> None:
        ...

    @element
    def nvvideoconvert(self, props={}, **kwargs) -> None:
        ...

    @element
    def playbin(self, props={}, **kwargs) -> None:
        ...

    @element
    def progressreport(self, props={}, **kwargs) -> None:
        ...

    @element
    def qtdemux(self, props={}, **kwargs) -> None:
        ...

    @element
    def qtmux(self, props={}, **kwargs) -> None:
        ...

    def queue(self, props={}, **kwargs) -> None:
        allprops = dict(instance='queue', **kwargs)
        allprops.update(props)
        allprops.setdefault('max-size-buffers', self.default_queue_max_size_buffers)
        allprops.setdefault('max-size-time', 0)
        allprops.setdefault('max-size-bytes', 0)
        self.append(allprops)

    @element
    def rawvideoparse(self, props={}, **kwargs) -> None:
        ...

    @element
    def rtph264depay(self, props={}, **kwargs) -> None:
        ...

    @element
    def rtspclientsink(self, props={}, **kwargs) -> None:
        ...

    @element
    def rtspsrc(self, props={}, **kwargs) -> None:
        ...

    @element
    def tee(self, props={}, **kwargs) -> None:
        ...

    @element
    def funnel(self, props={}, **kwargs) -> None:
        ...

    @element
    def axfunnel(self, props={}, **kwargs) -> None:
        ...

    @element
    def textoverlay(self, props={}, **kwargs) -> None:
        ...

    @element
    def vaapih264dec(self, props={}, **kwargs) -> None:
        ...

    @element
    def vaapih264enc(self, props={}, **kwargs) -> None:
        ...

    @element
    def vaapih265dec(self, props={}, **kwargs) -> None:
        ...

    @element
    def vaapipostproc(self, props={}, **kwargs) -> None:
        ...

    @element
    def v4l2src(self, props={}, **kwargs) -> None:
        ...

    @element
    def videobox(self, props={}, **kwargs) -> None:
        ...

    @element
    def videoconvert(self, props={}, **kwargs) -> None:
        ...

    @element
    def videocrop(self, props={}, **kwargs) -> None:
        ...

    @element
    def videoflip(self, props={}, **kwargs) -> None:
        ...

    @element
    def videorate(self, props={}, **kwargs) -> None:
        ...

    @element
    def videoscale(self, props={}, **kwargs) -> None:
        ...

    @element
    def videotestsrc(self, props={}, **kwargs) -> None:
        ...

    @element
    def x264enc(self, props={}, **kwargs) -> None:
        ...

    @element
    def axinference(self, props={}, **kwargs) -> None:
        ...

    @element
    def cameraundistort(self, props={}, **kwargs) -> None:
        ...

    @element
    def perspective(self, props={}, **kwargs) -> None:
        ...


_NOT_IN_AXINFERENCENET = {
    'addstreamid',
    'barrelcorrect',
    'colorconvert',
    'contrastnormalize',
    'filterdetections',
    'perspective',
}


def _belongs_in_axinferencenet(lib):
    m = re.match(r'lib(?:transform|inplace|decoder)_(.+)\.(?:so|dylib|dll)', lib)
    return m.group(1).removesuffix('_cl') not in _NOT_IN_AXINFERENCENET


class BuilderNewInference(Builder):
    def __init__(self, hw_config=None, default_queue_max_size_buffers=4):
        super().__init__(
            hw_config=hw_config, default_queue_max_size_buffers=default_queue_max_size_buffers
        )
        self.axinf_preops = []
        self.axinf_props = {}
        self.axinf_postops = []
        self.where = None

    @property
    def new_inference(self) -> bool:
        return True

    def build_axinference(self, props) -> None:
        inf = dict(instance='axinferencenet', **self.axinf_props)

        def ops(phase, ops):
            for n, (lib, opts, mode, batch) in enumerate(ops):
                inf[f'{phase}process{n}_lib'] = lib
                inf[f'{phase}process{n}_options'] = opts
                if mode:
                    inf[f'{phase}process{n}_mode'] = mode
                if batch:
                    inf[f'{phase}process{n}_batch'] = batch

        if self.axinf_preops or self.axinf_postops:
            ops('pre', self.axinf_preops)
            ops('post', self.axinf_postops)
            self.append(inf)
        self.axinf_preops = []
        self.axinf_props = {}
        self.axinf_postops = []

    def axtransform(self, props={}, **kwargs) -> None:
        if not props and not kwargs:
            props = {**props, **kwargs}
            self.build_axinference(props)
        elif not _belongs_in_axinferencenet(kwargs.get('lib', '')):
            super().axtransform(props, **kwargs)
        else:
            ops = self.axinf_postops if self.axinf_props else self.axinf_preops
            props = {**props, **kwargs}
            ops.append(
                (
                    props['lib'],
                    props.get('options', ''),
                    props.get('mode', ''),
                    props.get('batch', ''),
                )
            )

    def axinplace(self, props={}, **kwargs) -> None:
        lib = kwargs.get('lib', '')
        if lib == '' or not _belongs_in_axinferencenet(lib):
            super().axinplace(props, **kwargs)
        else:
            ops = self.axinf_postops if self.axinf_props else self.axinf_preops
            props = {**props, **kwargs}
            ops.append(
                (
                    props['lib'],
                    props.get('options', ''),
                    props.get('mode', ''),
                    props.get('batch', ''),
                )
            )

    def decode_muxer(self, props={}, **kwargs) -> None:
        props = {**props, **kwargs}
        self.axinf_postops.append(
            (props['lib'], props['options'], props.get('mode', ''), props.get('batch', ''))
        )

    def axinference(self, **kwargs) -> None:
        props = kwargs
        if self.where:
            props |= self.where
        self.axinf_props = props
        self.where = None

    def appsink(self, props={}, **kwargs) -> None:
        self.build_axinference({})
        return super().appsink(props, **kwargs)

    def distributor(self, props={}, **kwargs) -> None:
        props = {**props, **kwargs}
        self.build_axinference(props)
        self.where = props

    def tee(self, props={}, **kwargs) -> None:
        pass


def builder(*args, **kwargs):
    new_inference = config.env.axinferencenet
    cls = BuilderNewInference if new_inference else Builder
    return cls(*args, **kwargs)

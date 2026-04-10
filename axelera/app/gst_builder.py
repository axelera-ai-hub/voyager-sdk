# Copyright Axelera AI, 2023
from __future__ import annotations

from . import config


def element(f):
    def wrapper(self, props={}, **kwargs):
        allprops = dict(instance=f.__name__, **kwargs)
        allprops.update(props)
        if allprops.get('name', 'not empty') == '':
            del allprops['name']
        self.append(allprops)

    return wrapper


class TileInfo:
    def __init__(self, op, gst):
        self.tile_meta = "axelera-tiles-internal"
        self.gst = gst
        self.op = op
        self.tiling = gst.tiling and gst.add_tiles
        if op._where:
            self.master_key = f'master_meta:{op._where};'
        elif self.tiling:
            self.master_key = f'master_meta:{self.tile_meta};'
        else:
            self.master_key = str()
        self.association_key = str()
        if op._association:
            self.association_key = f'association_meta:{op._association};'

    def get_decode_keys(self):
        return self.master_key, self.association_key

    def get_nms_keys(self, flatten_meta: int = 1):
        if self.tiling:
            return f'flatten_meta:{flatten_meta};master_meta:{self.tile_meta};'
        return self.master_key

    def get_max_boxes(self, max_boxes: int):
        if self.tiling:
            return 2 * max_boxes
        return max_boxes


class _OldBuilder(list):
    def __init__(
        self,
        hw_config=None,
        tiling: config.TilingConfig | None = None,
        default_queue_max_size_buffers=16,
        which_cl: str = 'auto',
    ):
        self.hw_config = hw_config
        self.default_queue_max_size_buffers = default_queue_max_size_buffers
        self.tiling = tiling or config.TilingConfig()
        self.images = None
        self.which_cl = which_cl

    def getconfig(self):
        return self.hw_config

    @element
    def identity(self, props={}, **kwargs) -> None: ...

    @element
    def appsink(self, props={}, **kwargs) -> None: ...

    @element
    def appsrc(self, props={}, **kwargs) -> None: ...

    @element
    def aspectratiocrop(self, props={}, **kwargs) -> None: ...

    @element
    def axinplace(self, props={}, **kwargs) -> None: ...

    def axtransform(self, props={}, **kwargs) -> None:
        allprops = dict(instance='axtransform', **kwargs)
        allprops.update(props)
        if self.getconfig() and self.getconfig().opencl:
            if self.which_cl and self.which_cl != 'auto':
                allprops.setdefault('cl-platform', self.which_cl)
        self.append(allprops)

    @element
    def capsfilter(self, props={}, **kwargs) -> None: ...

    @element
    def clpreproc(self, props={}, **kwargs) -> None: ...

    @element
    def compositor(self, props={}, **kwargs) -> None: ...

    @element
    def decode_muxer(self, props={}, **kwargs) -> None: ...

    @element
    def decodebin(self, props={}, **kwargs) -> None: ...

    @element
    def distributor(self, props={}, **kwargs) -> None: ...

    @element
    def fakesink(self, props={}, **kwargs) -> None: ...

    @element
    def filesink(self, props={}, **kwargs) -> None: ...

    @element
    def filesrc(self, props={}, **kwargs) -> None: ...

    @element
    def h264parse(self, props={}, **kwargs) -> None: ...

    @element
    def h265parse(self, props={}, **kwargs) -> None: ...

    @element
    def nvh264dec(self, props={}, **kwargs) -> None: ...

    @element
    def jpegdec(self, props={}, **kwargs) -> None: ...

    @element
    def mp4mux(self, props={}, **kwargs) -> None: ...

    @element
    def multifilesink(self, props={}, **kwargs) -> None: ...

    @element
    def multifilesrc(self, props={}, **kwargs) -> None: ...

    @element
    def nvv4l2decoder(self, props={}, **kwargs) -> None: ...

    @element
    def nvv4l2h264enc(self, props={}, **kwargs) -> None: ...

    @element
    def nvvideoconvert(self, props={}, **kwargs) -> None: ...

    @element
    def playbin(self, props={}, **kwargs) -> None: ...

    @element
    def progressreport(self, props={}, **kwargs) -> None: ...

    @element
    def qtdemux(self, props={}, **kwargs) -> None: ...

    @element
    def qtmux(self, props={}, **kwargs) -> None: ...

    def queue(self, props={}, **kwargs) -> None:
        allprops = dict(instance='queue', **kwargs)
        allprops.update(props)
        allprops.setdefault('max-size-buffers', self.default_queue_max_size_buffers)
        allprops.setdefault('max-size-time', 0)
        allprops.setdefault('max-size-bytes', 0)
        self.append(allprops)

    @element
    def rawvideoparse(self, props={}, **kwargs) -> None: ...

    @element
    def rtph264depay(self, props={}, **kwargs) -> None: ...

    @element
    def rtspclientsink(self, props={}, **kwargs) -> None: ...

    @element
    def rtspsrc(self, props={}, **kwargs) -> None: ...

    @element
    def tee(self, props={}, **kwargs) -> None: ...

    @element
    def funnel(self, props={}, **kwargs) -> None: ...

    @element
    def axfunnel(self, props={}, **kwargs) -> None: ...

    @element
    def textoverlay(self, props={}, **kwargs) -> None: ...

    @element
    def vaapih264dec(self, props={}, **kwargs) -> None: ...

    @element
    def vaapih264enc(self, props={}, **kwargs) -> None: ...

    @element
    def vaapih265dec(self, props={}, **kwargs) -> None: ...

    @element
    def vaapipostproc(self, props={}, **kwargs) -> None: ...

    @element
    def v4l2src(self, props={}, **kwargs) -> None: ...

    @element
    def videobox(self, props={}, **kwargs) -> None: ...

    @element
    def videoconvert(self, props={}, **kwargs) -> None: ...

    @element
    def videocrop(self, props={}, **kwargs) -> None: ...

    @element
    def videoflip(self, props={}, **kwargs) -> None: ...

    @element
    def videorate(self, props={}, **kwargs) -> None: ...

    @element
    def videoscale(self, props={}, **kwargs) -> None: ...

    @element
    def videotestsrc(self, props={}, **kwargs) -> None: ...

    @element
    def x264enc(self, props={}, **kwargs) -> None: ...

    @element
    def axinference(self, props={}, **kwargs) -> None: ...

    @element
    def cameraundistort(self, props={}, **kwargs) -> None: ...

    @element
    def perspective(self, props={}, **kwargs) -> None: ...

    @element
    def nvvidconv(self, props={}, **kwargs) -> None: ...


class Builder(_OldBuilder):
    def __init__(
        self,
        hw_config,
        tiling: config.TilingConfig | None,
        default_queue_max_size_buffers,
        which_cl: str,
    ):
        super().__init__(hw_config, tiling, default_queue_max_size_buffers, which_cl)
        self.axinf_preops = []
        self.axinf_props = {}
        self.axinf_postops = []
        self.where = None
        self.building_axinference = False
        self.add_tiles = True  # Will be set to False when input is from ROI
        self.margin = None

    def start_axinference(self, props={}) -> None:
        self.building_axinference = True

    def finish_axinference(self, props={}) -> None:
        if not self.building_axinference:
            return
        inf = dict(instance='axinferencenet', **self.axinf_props)

        def ops(phase, ops):
            for n, (lib, opts, mode, batch) in enumerate(ops):
                inf[f'{phase}process{n}_lib'] = lib
                inf[f'{phase}process{n}_options'] = opts
                if mode:
                    inf[f'{phase}process{n}_mode'] = mode

        if self.axinf_preops or self.axinf_postops:
            ops('pre', self.axinf_preops)
            ops('post', self.axinf_postops)
            self.append(inf)
        self.axinf_preops = []
        self.axinf_props = {}
        self.axinf_postops = []
        self.building_axinference = False

    def axtransform(self, props={}, **kwargs) -> None:
        if not self.building_axinference:
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
        if not self.building_axinference:
            super().axinplace(props, **kwargs)
        elif not {**props, **kwargs}.get('lib'):
            # if no lib, this follows a gstreamer element so move out of axinferencenet
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
        if not self.building_axinference:
            raise ValueError('decode_muxer not allowed outside of axinferencenet building')
        props = {**props, **kwargs}
        self.axinf_postops.append(
            (props['lib'], props['options'], props.get('mode', ''), props.get('batch', ''))
        )

    def axinference(self, **kwargs) -> None:
        if not self.building_axinference:
            raise ValueError('axinference not allowed outside of axinferencenet building')
        props = kwargs
        if self.where:
            props |= self.where
        if self.margin:
            props['margin'] = self.margin
        if self.getconfig() and self.getconfig().opencl:
            if self.which_cl and self.which_cl != 'auto':
                props.setdefault('cl-platform', self.which_cl)
        self.axinf_props = props
        self.where = None

    def appsink(self, props={}, **kwargs) -> None:
        self.finish_axinference()
        return super().appsink(props, **kwargs)

    def distributor(self, props={}, **kwargs) -> None:
        if not self.building_axinference:
            raise ValueError('distributor not allowed outside of axinferencenet building')
        props = {**props, **kwargs}
        self.where = {'meta': props.get('meta')}
        margin = props.get('margin')
        if margin:
            self.margin = margin

    def tee(self, props={}, **kwargs) -> None:
        if not self.building_axinference:
            super().tee(props, **kwargs)


def builder(*args, **kwargs):
    return Builder(*args, **kwargs)

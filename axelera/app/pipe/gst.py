# Copyright Axelera AI, 2023
# Construct GStreamer application pipeline
from __future__ import annotations

import collections
import ctypes
import functools
import os
from pathlib import Path
import pprint
import queue
import re
import threading
import time
import traceback
from typing import TYPE_CHECKING, Any, Callable, Iterable, List
import warnings

try:
    from gi.repository import GObject, Gst, GstApp, GstVideo  # noqa: F401

    GST_AVAILABLE = True
except ModuleNotFoundError:
    (GObject, Gst, GstApp, GstVideo) = (None, None, None, None)
    GST_AVAILABLE = False

import yaml

from axelera import types

from . import base, graph, gst_helper, io
from .. import config, gst_builder, logging_utils, meta, operators, utils
from .frame_data import FrameEvent, FrameResult
from .gst_helper import AGGREGATE_NAME

if TYPE_CHECKING:
    from .. import config, network, pipeline

    GstTaskMeta = dict[str, Any]

LOG = logging_utils.getLogger(__name__)


class GstStream:
    '''
    GstStream is a wrapper around a GStreamer pipeline that
    extracts inference metadata from the pipeline and yields it,
    along with the buffer, to the caller.
    '''

    def __init__(
        self,
        pipeline: Gst.Pipeline,
        logging_dir: Path,
        hardware_caps: config.HardwareCaps,
        frame_generators: dict[int, io.FrameInputGenerator],
    ):
        self.pipeline = pipeline
        self.agg_pads: dict[int, str] = {}
        self.logging_dir = logging_dir
        self._stop_event = threading.Event()
        self._handlers: dict[int, _AppSrcHandler] = {}

        for appsrc in gst_helper.list_all_by_element_factory_name(pipeline, 'appsrc'):
            LOG.debug(f"Found appsrc element {appsrc.name} in pipeline")
            sid = gst_helper.get_stream_id_from_appsrc(appsrc)
            loader = iter(frame_generators[sid])
            self._handlers[sid] = _AppSrcHandler(appsrc, loader, self._stop_event)

        self.stream_meta_key = meta.GstMetaInfo('stream_id', 'stream_meta')
        self.agg_pads = gst_helper.get_agg_pads(self.pipeline)
        self.hardware_caps = hardware_caps
        self._appsinks = gst_helper.list_all_by_element_factory_name(pipeline, 'appsink')
        self.decoder = meta.GstDecoder()
        self._removed_sources: set[str] = set()
        self._pending_events: collections.deque[FrameEvent] = collections.deque()
        gst_helper.set_state_and_wait(self.pipeline, Gst.State.READY)

        try:
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                raise RuntimeError("Unable to set the pipeline to the playing state")

            for t in self._handlers.values():
                t.start()

            self._pre_sample = self._pre_roll_pipeline()
        except BaseException as e:
            # BaseException because we want to include KeyboardException here. You see this happen
            # when you press Ctrl-C during pipeline setup if we have got the pipeline into READY
            # or PLAYING we need to ensure a clean shutdown or else you get Gst-CRITICAL msgs
            LOG.warning(f"An error occurred whilst in pre-roll pipeline: {e!r}, stopping pipeline")
            self.stop()
            raise

    def is_pair_validation(self, sid: int) -> bool:
        handler = self._handlers.get(sid)
        return bool(handler) and handler.is_pair_validation

    def _at_eos(self):
        if not self.pipeline:
            return True
        bus = self.pipeline.get_bus()
        continue_stream = True
        while continue_stream:
            if (msg := bus.pop()) is None:
                return False
            continue_stream = self._on_bus_message(msg)
        return not continue_stream

    def _on_bus_message(self, message: Gst.Message) -> bool:
        '''Callback function for watching the GST pipeline. Dump the pipeline graph to a .dot file
        and convert it to a .svg file once the stream starts playing.'''
        STOP, CONTINUE = False, True
        mtype, src = message.type, message.src
        src_name = src.get_name() if src else 'unknown'
        if mtype == Gst.MessageType.EOS:
            LOG.debug(f"End of stream ({src_name})")
            return STOP
        elif mtype == Gst.MessageType.ERROR:
            err, debug = message.parse_error()

            LOG.error('%s: BUS: %s\n -> %s', src_name, err.message, debug)
            if src_name in self._removed_sources:
                LOG.debug(f"Error source {src_name}, has already been removed")
                return CONTINUE
            sid = gst_helper.find_source_id_by_src_elem(self.pipeline, src)
            if sid >= 0:
                message = f"Error occurred in source {src_name}: {err.message}"
                self._pending_events.append(FrameEvent.from_source_error(sid, message))
                if len(self.agg_pads) > 1:
                    LOG.debug(f"Error source is {sid=}, removing source from pipeline")
                    if (agg_pad_name := self.agg_pads.pop(sid, None)) is not None:
                        gst_helper.remove_source(self.pipeline, agg_pad_name)
                    self._removed_sources.add(src_name)
                    return CONTINUE
                LOG.warning(f"Error source is {sid=}, all streams exhausted")
                message = f"Error occurred in source {sid} {src_name} {err.message}"
                self._pending_events.append(FrameEvent.from_end_of_pipeline(sid, message))
            elif self._removed_sources:
                # we started, and at least one source failed and is being/has been removed
                # so probably this is vestigial error from the removed source, so keep going
                LOG.warning("Unable to identify error source, but recently removed source")
                return CONTINUE
            else:
                loc = gst_helper.src_info(src)
                raise RuntimeError(
                    f"Error in GStreamer pipeline from an unlinked source : {loc}):\n{err.message}\nDebug info: {debug}"
                )
            try:
                gst_helper.set_state_and_wait(self.pipeline, Gst.State.NULL)
            except Exception as e:
                # we really don't want to propagate any other error here
                LOG.error("Failed to set pipeline NULL state: %s", str(e))
            return STOP
        elif mtype == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            LOG.warning('%s: BUS: %s\n (%s)', src_name, err.message, debug)
            return CONTINUE
        else:
            gst_helper.log_state_changes(message, self.pipeline, self.logging_dir)
            return CONTINUE

    def _pre_roll_pipeline(self):
        # we send the first frame through the pipeline to ensure all elements are
        # fully constructed, this means errors in the pipeline are detected early
        while not self._at_eos():
            for stream_id, sink in enumerate(self._appsinks):
                sample = sink.try_pull_sample(Gst.MSECOND)
                if sample:
                    LOG.debug("Received first frame from gstreamer")
                    return stream_id, sample
        raise RuntimeError('Failed to pre-roll pipeline, aborting pipeline build')

    def _frame_from_sample(self, sample, stream_id) -> tuple[FrameEvent, GstTaskMeta]:
        now = time.time()
        buf = sample.get_buffer()
        image = types.Image.fromgst(sample)

        gst_task_meta = self.decoder.extract_all_meta(buf)
        # pop here so that the stream_id is not treated as a normal meta element
        stream_data = gst_task_meta.pop(self.stream_meta_key, (stream_id, 0, 0))
        stream_id, ts, inferences = (
            stream_data
            if isinstance(stream_data, tuple)
            else (
                stream_data.get('stream_id', stream_id),
                stream_data.get('timestamp', 0),
                stream_data.get('inferences', 0),
            )
        )

        tensor = None  # TODO where do we get tensor from?
        mq = self._handlers.get(stream_id)
        img_id, gt = mq.get() if mq is not None else ('', None)
        ax_meta = meta.AxMeta(str(img_id), ground_truth=gt)
        result = FrameResult(image, tensor, ax_meta, stream_id, ts, now, inferences)
        return (FrameEvent.from_result(result), gst_task_meta)

    def __iter__(self) -> Iterable[tuple[FrameEvent, dict[str, Any] | None]]:
        try:
            if self._pre_sample is not None:
                stream_id, sample = self._pre_sample
                self._pre_sample = None
                if sample:
                    yield self._frame_from_sample(sample, stream_id)

            while not self._at_eos():
                while self._pending_events:
                    yield (self._pending_events.popleft(), None)
                for stream_id, sink in enumerate(self._appsinks):
                    sample = sink.try_pull_sample(Gst.MSECOND)
                    if sample:
                        yield self._frame_from_sample(sample, stream_id)
        except Exception as e:
            LOG.error(f"Error in GstStream iteration: {e!r}")
            self.stop()
            raise
        else:
            LOG.debug("Finished iterating frames from GStreamer pipeline")
            self.stop()

    def stop(self):
        if self._handlers:
            self._stop_event.set()
            for t in self._handlers.values():
                t.join()
            self._stop_event.clear()
        if pipeline := self.pipeline:
            self.pipeline = None
            self.agg_pads = {}
            self._appsinks = None
            LOG.trace("GstStream.stop: pipeline sending event")
            # pipeline.send_event(Gst.Event.new_eos())
            LOG.trace("GstStream.stop: pipeline event sent")
            gst_helper.set_state_and_wait(pipeline, Gst.State.NULL)
            del pipeline


class _AppSrcHandler:
    def __init__(self, appsrc, frame_generator, stop_event):
        self._mq = queue.Queue()
        self._enough_data = False
        appsrc.connect("need-data", self._on_need_data)
        appsrc.connect("enough-data", self._on_enough_data)
        self._mq = queue.Queue()
        self._thr = utils.ExceptionThread(target=self._feed_data, args=(appsrc, frame_generator))
        self._stop_event = stop_event
        self._pair_validation = None  # until we know otherwise

    def get(self):
        return self._mq.get()

    def start(self):
        self._thr.start()

    def join(self):
        self._thr.join()

    @property
    def is_pair_validation(self) -> bool:
        return bool(self._pair_validation)

    def _on_need_data(self, appsrc, length):
        self._enough_data = False

    def _on_enough_data(self, appsrc):
        self._enough_data = True

    def _feed_data(self, appsrc, frame_generator):
        '''
        This is called in a thread, and waits for the appsrc to need more data
        It calls the data src to get the next image and pushes it to the appsrc
        If the loader raises StopIteration, it will emit an end-of-stream signal
        '''
        # we need this limitation for VAAPI+GPU pipeline which runs too fast at preprocessing
        delay = 0.001
        try:
            while not self._stop_event.is_set():
                if not self._enough_data:
                    delay = self._feed_data_once(appsrc, frame_generator)
                    delay = 0.001 if delay is None else 0.01
                else:
                    time.sleep(delay)
        except StopIteration:
            LOG.debug("Frame generator raised StopIteration, stopping feeding thread")
        except Exception as e:
            LOG.error(f"Frame generator raised an error: {e}, stopping feeding thread")
            LOG.error(traceback.format_exc())
        appsrc.end_of_stream()
        self._stop_event.set()
        LOG.trace("Feeding thread stopped")

    def _feed_data_once(self, appsrc, frame_generator):
        data = next(frame_generator)
        assert isinstance(data, types.FrameInput), f"Expected FrameInput, got {type(data)}"
        image_source = [data.img] if data.img else data.imgs
        if image_source is None:
            LOG.warning("No image data found in the batch")
            return None

        if self._pair_validation is None:
            self._pair_validation = data.imgs is not None
        if self._pair_validation and len(image_source) != 2:
            LOG.warning(f"Pair validation requires 2 images, got {len(image_source)}")
        for image in image_source:
            w, h = image.size
            num_channels = 4
            if image.color_format == types.ColorFormat.RGB:
                format = types.ColorFormat.RGBA
            elif image.color_format == types.ColorFormat.BGR:
                format = types.ColorFormat.BGRA
            elif image.color_format == types.ColorFormat.GRAY:
                num_channels = 1
                format = types.ColorFormat.GRAY
            else:
                raise NotImplementedError(f"Unsupported color format: {image.color_format}")
            frame = image.tobytes(format)
            format_int = image.get_gst_format(format)
            in_info = GstVideo.VideoInfo()
            in_info.set_format(format_int, w, h)
            in_info.fps_n = 120
            in_info.fps_d = 1
            caps = in_info.to_caps()
            appsrc.set_caps(caps)
            if in_info.stride[0] == w * num_channels:
                buffer = Gst.Buffer.new_wrapped(frame)
            else:
                buffer = Gst.Buffer.new_allocate(None, in_info.size, None)
                for i in range(h):
                    frame_offset = i * w * num_channels
                    buffer_offset = in_info.offset[0] + i * in_info.stride[0]
                    buffer.fill(
                        buffer_offset, frame[frame_offset : frame_offset + w * num_channels]
                    )
            appsrc.push_buffer(buffer)
            self._mq.put((data.img_id, data.ground_truth))
        return data


def generate_padding(manifest: types.Manifest) -> str:
    padding = manifest.n_padded_ch_inputs[0] if manifest.n_padded_ch_inputs else []
    if len(padding) == 4:  # legacy remove soon
        top, left, bottom, right = padding
        padding = [0, 0, top, bottom, left, right, 0, 0]
    else:
        padding = list(padding[:8]) + [0] * (8 - len(padding))
    if padding[-1] in (1, 61):
        padding[-1] -= 1  # 1 byte of padding is due to using RGB[Ax], don't pad it further
    return ','.join(str(x) for x in padding)


def _labels(app_fmwk: Path, model_name: str):
    if 'ssd-mobilenet' in model_name.lower():
        return app_fmwk / "ax_datasets/labels/coco90.names"
    elif 'yolo' in model_name.lower():
        return app_fmwk / "ax_datasets/labels/coco.names"
    else:
        return app_fmwk / "ax_datasets/labels/imagenet1000_clsidx_to_labels.txt"


def _parse_low_level_pipeline(
    tasks: list[pipeline.AxTask],
    mi: List,
    input_sources: list[config.Source],
    hardware_caps: config.HardwareCaps,
    ax_precompiled_gst: str,
):
    manifests = [model.manifest for model in mi]
    manifest = manifests[0]
    model_names = [model.name for model in mi]

    fmwk = config.env.framework
    is_measure = input_sources[0].type == config.SourceType.DATASET
    hardware_tag = "gpu" if hardware_caps.vaapi and hardware_caps.opencl else "."
    LOG.debug(f"Reference lowlevel yaml: {ax_precompiled_gst}")
    quant_scale, quant_zeropoint = manifest.quantize_params[0]
    dequant_scale, dequant_zeropoint = manifest.dequantize_params[0]
    pp_file = Path(manifest.model_lib_file).parent / "lib_cpu_post_processing.so"
    ref = {
        'class_agnostic': 1,
        'confidence_threshold': 0.0016 if is_measure else 0.3,
        'dequant_scale': dequant_scale,
        'dequant_zeropoint': dequant_zeropoint,
        'force_sw_decoders': hardware_tag != 'gpu',
        'input_h': manifest.input_shapes[0][1],
        'input_video': input_sources[0].location,
        'input_w': manifest.input_shapes[0][2],
        'label_file': _labels(fmwk, model_names[0]),
        'max_boxes': 30000,
        'nms_top_k': 200,
        'model_lib': manifest.model_lib_file,
        'model_name': model_names[0],
        'nms_threshold': 0.5,
        'pads': generate_padding(manifest),
        'post_model_lib': pp_file,
        'prefix': '',
        'quant_scale': quant_scale,
        'quant_zeropoint': quant_zeropoint,
    }
    ref.update({f'input_video{n}': p.location for n, p in enumerate(input_sources)})
    ref.update({f'model_lib{n}': m.model_lib_file for n, m in enumerate(manifests)})
    ref.update({f'model_name{n}': t.model_info.name for n, t in enumerate(tasks)})
    ref.update({f'label_file{n}': _labels(fmwk, t.model_info.name) for n, t in enumerate(tasks)})
    LOG.debug(f"ref: {ref}")
    return utils.load_yaml_by_reference(ax_precompiled_gst, ref)


def _read_low_level_pipeline(
    ax_precompiled_gst: str | Path,
    nn,
    sources: list[config.Source],
    hardware_caps: config.HardwareCaps,
):
    if sources[0].type == config.SourceType.IMAGE_FILES:
        raise ValueError(
            "Precompiled GST pipeline is not supported for directory of images as input"
        )
    if isinstance(ax_precompiled_gst, str) and not os.path.exists(ax_precompiled_gst):
        _pipelines = yaml.safe_load(ax_precompiled_gst)
    elif isinstance(ax_precompiled_gst, (str, Path)):
        _pipelines = _parse_low_level_pipeline(
            nn.tasks,
            list(iter(nn.model_infos.models())),
            sources,
            hardware_caps,
            ax_precompiled_gst,
        )
    else:
        raise ValueError(
            f"Invalid precompiled GST pipeline {ax_precompiled_gst}, expected a Path"
            f" to a YAML file, or a string containing the yaml content.",
        )
    try:
        return _pipelines[0]['pipeline']
    except (ValueError, IndexError, KeyError):
        raise ValueError(
            f"Invalid precompiled YAML file {ax_precompiled_gst}: expected `- pipeline:`"
        ) from None


def _collect_gst_plugins(pipeline: list[dict[str, Any]]) -> list[str]:
    """Collect all GStreamer element factories required by the pipeline.

    Returns a sorted list of unique element names.
    """
    return sorted({e['instance'] for e in pipeline if 'instance' in e})


def _collect_ax_plugins(pipeline: list[dict[str, Any]]) -> list[str]:
    """Collect axelera plugin names used in the pipeline.

    Extracts the 'lib' key from axinplace/axtransform elements and
    'preprocess{N}_lib' / 'postprocess{N}_lib' keys from axinferencenet elements.
    """
    plugin_wrappers = {'axinplace', 'axtransform', 'axdecode'}
    plugins = [elem.get('lib', '') for elem in pipeline if elem.get('instance') in plugin_wrappers]
    axnets = [elem for elem in pipeline if elem.get('instance') == 'axinferencenet']
    is_op = re.compile(r'(pre|post)process\d+_lib$')
    plugins += [v for axnet in axnets for k, v in axnet.items() if is_op.match(k) and v]
    return sorted({p for p in plugins if p})


def _resolve_lib(lib: str, search_dirs: list[Path]) -> Path | None:
    """Resolve a shared library filename to a full path."""
    p = Path(lib)
    if not p.is_absolute():
        for d in search_dirs:
            p = d / lib
            if p.exists():
                break
        else:
            search = ':'.join(str(d) for d in search_dirs)
            LOG.debug("Failed to find library %s in any of: %s", lib, search)
            return None
    try:
        handle = ctypes.CDLL(str(p))
        del handle
    except OSError as e:
        LOG.debug("Failed to load library %s at %s: %s", lib, p, e)
        return None
    else:
        LOG.trace("Found library %s at %s", lib, p)
        return p


def _find_missing_gst_plugins(gst_plugins: list[str]) -> list[str]:
    """Find any GStreamer elements that do not have a valid Gst factory."""
    if not Gst.is_initialized():
        Gst.init(None)
    return [name for name in gst_plugins if Gst.ElementFactory.find(name) is None]


@functools.lru_cache(maxsize=1)
def _get_search_dirs_for_ax_plugins() -> list[Path]:
    """Get the list of directories to search for axelera plugin shared libraries."""

    def _split_path(env) -> list[Path]:
        return [Path(d) for d in os.environ.get(env, '').split(os.pathsep) if d]

    gstaxstreamer_name = 'libgstaxstreamer.so'
    if axstreamer := _resolve_lib(gstaxstreamer_name, _split_path('GST_PLUGIN_PATH')):
        LOG.debug(
            "Found %s in GST_PLUGIN_PATH, using %s for axelera plugins",
            gstaxstreamer_name,
            axstreamer.parent,
        )
        search_dirs = [axstreamer.parent]
    else:
        LOG.warning(
            "Failed to find %s in GST_PLUGIN_PATH, falling back to searching LD_LIBRARY_PATH for axelera plugins",
            gstaxstreamer_name,
        )
        search_dirs: list[Path] = []

    for env in ('AX_SUBPLUGIN_PATH', 'LD_LIBRARY_PATH'):
        search_dirs.extend(_split_path(env))
    return search_dirs


def _find_missing_ax_plugins(ax_plugins: list[str]) -> list[str]:
    """Verify that all required gst elements and ax plugins can be loaded."""
    # Build a search path matching what dlopen / the C++ load_plugin will use
    search_dirs = _get_search_dirs_for_ax_plugins()
    return [p for p in ax_plugins if _resolve_lib(p, search_dirs) is None]


def _report_missing_plugins(gst_missing: list[str], ax_missing: list[str]) -> None:
    ax_gst = [name for name in gst_missing if name.startswith('ax')]
    std_gst = [name for name in gst_missing if not name.startswith('ax')]
    if std_gst:
        s = '' if len(std_gst) == 1 else 's'
        LOG.error("The following GStreamer element%s are missing:\n  %s", s, '\n  '.join(std_gst))
        LOG.error("Please ensure the required GStreamer plugins are installed")
    if ax_gst:
        s = '' if len(ax_gst) == 1 else 's'
        LOG.error(
            "The following Axelera GStreamer element%s are missing:\n  %s", s, '\n  '.join(ax_gst)
        )

    plugin_path = os.environ.get('GST_PLUGIN_PATH', '')
    if gst_missing:
        if plugin_path:
            LOG.error(f"GST_PLUGIN_PATH is set to: {plugin_path}")
        else:
            LOG.error(
                "GST_PLUGIN_PATH is not set\n"
                "[ This should have be configured by the Axelera framework at runtime]"
            )

    if ax_missing:
        search_dirs = _get_search_dirs_for_ax_plugins()
        searched = '\n  '.join(str(d) for d in search_dirs) if search_dirs else '<none>'
        formatted = '\n  '.join(ax_missing)
        s = '' if len(ax_missing) == 1 else 's'
        LOG.error(
            f"The following axstreamer plugin{s} could not be loaded:\n  {formatted}\n"
            f"Search directories:\n  {searched}\n"
        )
    LOG.error(
        "Please ensure all GStreamer elements and operators are built by running:\n"
        "  make clobber-libs operators"
    )


def _verify_pipeline_plugins(pipeline: list[dict[str, Any]]) -> None:
    '''Verify that all required gst elements and ax plugins can be loaded and emit a report.'''
    gst_plugins = _collect_gst_plugins(pipeline)
    ax_plugins = _collect_ax_plugins(pipeline)
    LOG.debug(
        "Verifying required plugins for gst pipeline, GST_PLUGIN_PATH=%s",
        os.environ.get('GST_PLUGIN_PATH', '<notset>'),
    )
    LOG.debug("Required GStreamer plugins: %s", ', '.join(gst_plugins))
    LOG.debug("Required Axelera plugin libraries: %s", ', '.join(ax_plugins))
    gst_missing = _find_missing_gst_plugins(gst_plugins)
    ax_missing = _find_missing_ax_plugins(ax_plugins)
    if gst_missing or ax_missing:
        _report_missing_plugins(gst_missing, ax_missing)


def _add_element_name(name_counter, e):
    if 'instance' in e and 'name' not in e:
        prefix = e['instance']
        if 'lib' in e:
            prefix += '-' + e['lib'].split('_', 1)[1][:-3]
        n = name_counter[prefix]
        name_counter[prefix] = n + 1
        e['name'] = f"{prefix.replace('_', '-')}{n}"
    # NOTE we rebuild the dict to ensure order is instance, name, ... for testing gold output
    return dict(
        instance=e['instance'],
        name=e['name'],
        **{k: v for k, v in e.items() if k not in ['instance', 'name']},
    )


def _add_element_names(elements):
    counter = collections.defaultdict(int)
    elements = [_add_element_name(counter, e) for e in elements]
    return elements


def _build_input_pipeline(
    gst: gst_builder.Builder,
    task: pipeline.AxTask,
    multiplex_input: io.MultiplexPipeInput,
):
    for sourcen, (source_id, input) in enumerate(multiplex_input.inputs.items()):
        input.build_input_gst(gst, str(source_id))
        preproc = task.image_preproc_ops.get(sourcen, [])
        task.input.build_gst(gst, source_id)  # now a nop, left here to remind to remove it
        for op in preproc:
            op.build_gst(gst, source_id)
        gst.queue(connections={'src': f'{AGGREGATE_NAME}.sink_%u'})


def _build_pipeline(
    nn: network.AxNetwork,
    pipein: io.PipeInput,
    hw_caps: config.HardwareCaps,
    which_cl: str,
    low_latency: bool,
) -> list[dict[str, Any]]:
    qsize = 1 if low_latency else 4
    tiling = any(s.has_tiling for s in pipein.sources)
    gst = gst_builder.builder(hw_caps, tiling, qsize, which_cl)
    _build_input_pipeline(gst, nn.tasks[0], pipein)

    for taskn, task in enumerate(nn.tasks):
        if task.model_info.model_type == types.ModelType.CLASSICAL_CV:
            # TODO: use Input operator to convert color space (if needed) before cv_process
            for op in task.cv_process:
                op.build_gst(gst, '')
            continue
        if isinstance(task.input, operators.Input):
            gst.add_tiles = bool(gst.tiling)
        if isinstance(task.input, operators.InputNoTiles):
            gst.add_tiles = False
        if isinstance(task.input, operators.InputFromROI):
            task.input.build_gst(gst, '')
            gst.add_tiles = False
        else:
            gst.start_axinference()
        # TODO: We probably need a better way to ensure we only tile the network we actually want to tile
        if gst.tiling and gst.add_tiles:
            gst.axtransform(
                lib='libtransform_roicrop.so',
                options='meta_key:axelera-tiles-internal',
            )

        for op in task.preprocess:
            op.build_gst(gst, '')

        task.inference.build_inference_gst(gst, task.aipu_cores)
        dq = []
        if task.inference.device == 'aipu':
            task.inference.config.reconcile_manifest(task.model_info)

            dq = [
                operators.AxeleraDequantize(
                    model=task.inference.model,
                    inference_op_config=task.inference.config,
                    num_classes=task.model_info.num_classes,
                    task_category=task.model_info.task_category,
                    assigned_model_name=task.model_info.name,
                    manifest_dir=task.inference.compiled_model_dir,
                    taskn=taskn,
                )
            ]
        for op in dq + task.postprocess:
            op.build_gst(gst, '')

        gst.finish_axinference()
    qsize = gst.default_queue_max_size_buffers
    gst.appsink({'max-buffers': qsize, 'drop': False, 'sync': False})
    return list(gst)


def _format_axnet_prop(k, v):
    if isinstance(v, str) and 'options' in k:
        subopts = v.split(';')
        v = ';'.join(x for x in subopts if x and not x.startswith('classlabels_file:'))
    if k == 'model':
        rel = os.path.relpath(v)
        if len(rel) < len(v):
            v = rel
    return f"{k}={v}"


def _save_axnet_files(gst: list[dict[str, str]], task_names: list[str], logging_dir: Path):
    axnets = [x for x in gst if x['instance'] == 'axinferencenet']
    IGNORE = ('instance', 'name')
    for axnet, task_name in zip(axnets, task_names):
        src = '\n'.join(_format_axnet_prop(k, v) for k, v in axnet.items() if k not in IGNORE)
        (logging_dir / f"../{task_name}.axnet").write_text(src)


def _softmax(nn, task_name):
    task = [t for t in nn.tasks if t.name == task_name]
    if not task:
        return False
    return any(getattr(op, 'softmax', False) for op in task[0].postprocess)


class GstPipe(base.Pipe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_ax_meta = None  # Cache the full ax_meta instance for pair validation
        self._stream: GstStream | None = None
        self._meta_assembler: meta.GstMetaAssembler | None = None
        self._gen_end2end_pipe()

    def _get_source_elements(self):
        """Find and return all source elements in the given pipeline."""
        assert self._stream, "Pipeline not yet created, call init_loop() first"
        sources = []
        for element in gst_helper.list_elements(self._stream.pipeline):
            # Check if the element is a source (it has only source pads)
            if not gst_helper.list_sink_pads(element):
                sources.append(element)
        return sources

    def _get_pausable_elements(self):
        if self._stream is None:
            raise RuntimeError("Pipeline not yet created")
        # if we have any live srcs we will pause them first, and then if we have aby non-live srcs
        # we also pause/play the whole pipeline
        # [for now we assume that only rtsp srcs are live, we should proably check
        #  the pipeline for other live srcs such as v4l]
        srcs = [src for src in self._get_source_elements()]
        rtsp = [src for src in srcs if src.get_name().startswith("rtsp")]
        if len(rtsp) != len(srcs):
            srcs.append(self._stream.pipeline)
        return srcs

    def pause(self):
        for elem in self._get_pausable_elements():
            gst_helper.set_state_and_wait(elem, Gst.State.PAUSED)

    def play(self):
        for elem in self._get_pausable_elements():
            gst_helper.set_state_and_wait(elem, Gst.State.PLAYING)

    def add_source(self, pipe_newinput: io.PipeInput):
        qsize = 1 if self.config.low_latency else 4
        tiling = any(s.has_tiling for s in pipe_newinput.sources)
        gst = gst_builder.builder(self.hardware_caps, tiling, qsize, self.config.which_cl)
        _build_input_pipeline(gst, self.nn.tasks[0], pipe_newinput)
        self._stream.agg_pads[pipe_newinput.source_id] = gst_helper.add_input(
            gst, self._stream.pipeline
        )

    def remove_source(self, source_id: int) -> None:
        if self._stream is None:
            raise RuntimeError("Pipeline not yet created")
        if len(self._stream.agg_pads) == 1:
            LOG.warning("Can't remove last stream, stop pipeline instead")
            return
        try:
            agg_pad_name = self._stream.agg_pads.pop(source_id)
        except KeyError:
            LOG.warning(f"Source slot {source_id} not occupied")
            return
        else:
            gst_helper.remove_source(self._stream.pipeline, agg_pad_name)

    def _get_aggregator(self):
        return self._stream.pipeline.get_by_name(AGGREGATE_NAME)

    def stream_select(self, streams: Iterable[int] | str) -> None:
        if isinstance(streams, str):
            warnings.warn(
                "Passing a comma-separated string to stream_select is deprecated. "
                "Please pass an iterable of integers instead.",
                DeprecationWarning,
            )
            if not re.match(r'^\d+(,\d+)*$', streams):
                raise ValueError(f"Invalid stream select format: {streams}")
        else:
            streams = ','.join(str(s) for s in sorted(streams))
        self._get_aggregator().set_property("stream_select", streams)

    def get_stream_select(self) -> list[int]:
        streams = self._get_aggregator().get_property("stream_select")
        return [int(s) for s in streams.split(',') if s]

    def _gen_end2end_pipe(self):
        self._frame_generators = self._pipein.frame_generators()

        pipeline = None
        if self.config.ax_precompiled_gst:
            pipeline = _read_low_level_pipeline(
                self.config.ax_precompiled_gst, self.nn, self._pipein.sources, self.hardware_caps
            )
        else:
            pipeline = _build_pipeline(
                self.nn,
                self._pipein,
                self.hardware_caps,
                self.config.which_cl,
                self.config.low_latency,
            )
            task_names = [t.model_info.name for t in self.nn.tasks]
            _save_axnet_files(pipeline, task_names, self.logging_dir)

        plugin_path = os.environ.get('GST_PLUGIN_PATH', '')
        if 'operators' not in plugin_path:
            extra = f"{config.env.framework}/operators/lib"
            os.environ['GST_PLUGIN_PATH'] = f"{extra}:{plugin_path}" if plugin_path else extra

        if self.config.ax_precompiled_gst and self.config.save_compiled_gst:
            LOG.debug("Not writing GST representation because it was passed in")
        elif out_yaml := self.config.save_compiled_gst:
            pipelines = [{'pipeline': pipeline}]
            out_yaml.write_text(yaml.dump(pipelines, sort_keys=False))
            LOG.debug(f"GST representation written to {out_yaml!s}")

        self.pipeline = _add_element_names(pipeline)

        self.model_info_labels_dict = {}
        self.model_info_num_classes_dict = {}
        for t in self.nn.tasks:
            self.model_info_labels_dict[t.name] = t.model_info.labels
            self.model_info_num_classes_dict[t.name] = t.model_info.num_classes

            # pass labels from detections to tracker
            if t.model_info.task_category == types.TaskCategory.ObjectTracking:
                bbox_task_name = self.task_graph.get_master(t.name)
                self.model_info_labels_dict[t.name] = self.model_info_labels_dict[bbox_task_name]
                self.model_info_num_classes_dict[t.name] = self.model_info_num_classes_dict[
                    bbox_task_name
                ]

        self._ensure_meta_assembler()

    def _ensure_meta_assembler(self) -> None:
        if self._meta_assembler is not None:
            return
        model_info_provider = meta.ModelInfoProvider(
            labels_dict=getattr(self, 'model_info_labels_dict', {}),
            num_classes_dict=getattr(self, 'model_info_num_classes_dict', {}),
            softmax_lookup=lambda task_name: _softmax(self.nn, task_name),
        )
        self._meta_assembler = meta.GstMetaAssembler(model_info_provider)

    def init_loop(self) -> Callable[[], None]:
        if not GST_AVAILABLE:
            raise RuntimeError(
                "Python bindings for GStreamer are not available, please run\n"
                " ./install-dependencies.sh\n"
                " pip install -r requirements.application.txt"
            )
        if LOG.isEnabledFor(logging_utils.TRACE):
            env = {k: v for k, v in sorted(os.environ.items()) if k.startswith('AX')}
            senv = pprint.pformat(env, width=1, compact=True, depth=1)
            LOG.trace("environment at gst pipeline construction:\n%s", senv)

        start = time.time()
        _verify_pipeline_plugins(self.pipeline)
        LOG.debug("Started building gst pipeline")
        loop = any(s.loop for s in self.config.sources)
        gst = gst_helper.build_pipeline(self.pipeline, loop)
        self._stream = GstStream(gst, self.logging_dir, self.hardware_caps, self._frame_generators)
        LOG.debug("Finished building gst pipeline - build time = %.3f", time.time() - start)
        return self._loop

    def _handle_pair_validation(self, ax_meta, decoded_meta):
        gst_meta_key, task_meta = next(iter(decoded_meta.items()))
        embeddings = task_meta.results
        pair_validation_meta = ax_meta.get_instance(
            gst_meta_key.task_name,
            meta.PairValidationMeta,  # gst_meta_key.meta_type is 'PairValidationMeta'
        )
        if pair_validation_meta.add_result(embeddings[0]):
            return None
        return ax_meta

    def _loop(self):
        try:
            for event, decoded_meta in self._stream:
                if self._stop_event.is_set():
                    break

                if event.result is None:
                    self._on_event(event)
                    continue

                fr = event.result
                if self._stream.is_pair_validation(fr.stream_id):
                    ax_meta = self._cached_ax_meta if self._cached_ax_meta else fr.meta
                    self._cached_ax_meta = self._handle_pair_validation(ax_meta, decoded_meta)
                    if self._cached_ax_meta is not None:
                        continue
                    fr.meta = ax_meta
                else:
                    ax_meta = fr.meta
                    self._ensure_meta_assembler()
                    self._meta_assembler.process(
                        ax_meta,
                        decoded_meta or {},
                        task_graph=self.task_graph,
                        result_view=graph.EdgeType.RESULT,
                    )

                if self._on_event(event) is False:
                    LOG.debug("_on_event requested to stop the pipeline")
                    break
        except Exception as e:
            LOG.error(f"Pipeline error occurred: {str(e)}\n{traceback.format_exc()}")
            self._stream.stop()
            self._stream = None
            self._on_event(FrameEvent.from_end_of_pipeline(0, str(e)))
            raise

        else:
            self._stream.stop()
            self._stream = None
            self._on_event(FrameEvent.from_end_of_pipeline(0, 'Normal termination'))

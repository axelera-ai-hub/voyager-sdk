# Copyright Axelera AI, 2025
# Construct GStreamer application pipeline
from __future__ import annotations

import collections
import os
from pathlib import Path
import pprint
import queue
import re
import threading
import time
from typing import TYPE_CHECKING, Any, Callable, Dict, List

import gi

from ..meta import (
    CocoBodyKeypointsMeta,
    FaceLandmarkTopDownMeta,
    InstanceSegmentationMeta,
    SemanticSegmentationMeta,
)
from ..operators import Input, InputWithImageProcessing

gi.require_version('Gst', '1.0')
gi.require_version('GstVideo', '1.0')
gi.require_version('GstApp', '1.0')  # for try_pull_sample

from gi.repository import GObject, Gst, GstApp, GstVideo
import yaml

from axelera import types

from . import base, frame_data, gst_helper
from .. import config, gst_builder, logging_utils, meta, operators, utils

if TYPE_CHECKING:
    from . import io
    from .. import config, network

LOG = logging_utils.getLogger(__name__)


class GstStream:
    '''
    GstStream is a wrapper around a GStreamer pipeline that
    extracts inference metadata from the pipeline and yields it,
    along with the buffer, to the caller.
    '''

    def __init__(
        self,
        pipeline,
        logging_dir,
        hardware_caps,
        progress: gst_helper.InitProgress,
        src=None,
        loader=None,
        data_formatter=None,
        is_pair_validation=False,
    ):
        '''
        pipeline - GStreamer pipeline to run
        logging_dir - directory to write pipeline debug files (dot/png/yaml) to
        src - name of the app source to feed the pipeline
        loader - callable that produces the input

        If sink is None, the pipeline runs without querying for inference
        metadata.
        if src is not None, the pipeline is fed from the buffer returned
        by the loader callable.
        if src is not None then loader must also be not None
        '''
        self.pipeline = pipeline
        self.logging_dir = logging_dir
        self.mqueue = None  # exist only if with measurement
        self._feeding_thread = None
        self.is_pair_validation = is_pair_validation
        if src is not None:
            loader_iterator = iter(loader)
            self.initialize_src(self.pipeline, src, loader_iterator, data_formatter)
            # container for measurement data
            self.mqueue = queue.Queue()
        self.pushed_frames, self.received_frames = 0, 0
        self.stream_meta_key = meta.GstMetaInfo('stream_id', 'stream_meta')

        self.hardware_caps = hardware_caps
        self._appsinks = gst_helper._gst_iterate(
            pipeline.iterate_all_by_element_factory_name('appsink')
        )
        self.decoder = meta.GstDecoder()
        ret = self.pipeline.set_state(Gst.State.READY)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Unable to set the pipeline to the ready state")
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            raise RuntimeError("Unable to set the pipeline to the playing state")

        # Uncomment to debug if a queue overrun/underrun occurs
        # self.connect_queue_overrun_signals(pipeline)
        if self._feeding_thread:
            self._feeding_thread.start()

        self._pre_sample = None
        while not self.at_eos(progress.on_message):
            for stream_id, sink in enumerate(self._appsinks):
                sample = sink.try_pull_sample(Gst.MSECOND)
                if sample:
                    progress.set_state(gst_helper.InitState.first_frame_received)
                    self._pre_sample = stream_id, sample
                    return

    def connect_queue_overrun_signals(self, pipeline):
        # Function to be called recursively to search for queue elements
        def find_queues(element):
            if isinstance(element, Gst.Bin):  # If the element is a container, look inside
                for child in element.iterate_elements():
                    find_queues(child)
            elif element.get_factory() and element.get_factory().get_name() == 'queue':
                # If the element is a queue, connect to its 'overrun' and 'underrun' signal
                element.connect("overrun", self.on_queue_overrun)
                element.connect("underrun", self.on_queue_underrun)

        find_queues(pipeline)

    def on_queue_underrun(self, queue):
        LOG.trace("Queue '{}' has underrun (it's nearly empty)!".format(queue.get_name()))

    def on_queue_overrun(self, queue):
        LOG.warning("Queue '{}' has overrun!".format(queue.get_name()))

    def at_eos(self, on_message=gst_helper.gst_on_message):
        if self.pipeline:
            bus = self.pipeline.get_bus()
            continue_stream = True
            while continue_stream:
                if self._appsinks:
                    msg = bus.pop()
                else:
                    msg = bus.timed_pop_filtered(Gst.MSECOND, Gst.MessageType.EOS)

                if msg is None:
                    return False
                continue_stream = on_message(msg, self.pipeline, self.logging_dir)

        return True

    def frame_from_sample(self, sample, stream_id):
        now = time.time()
        buf = sample.get_buffer()
        image = types.Image.fromgst(sample)

        gst_task_meta = self.decoder.extract_all_meta(buf)
        # pop here so that the stream_id is not treated as a normal meta element
        stream_data = gst_task_meta.pop(self.stream_meta_key, (stream_id, 0))
        stream_id, ts = (
            stream_data
            if isinstance(stream_data, tuple)
            else (stream_data.get('stream_id', stream_id), stream_data.get('timestamp', 0))
        )

        tensor = None  # TODO where do we get tensor from?
        self.received_frames += 1
        img_id, gt = self.mqueue.get() if self.mqueue else (self.received_frames, None)
        ax_meta = meta.AxMeta(str(img_id), ground_truth=gt)
        return frame_data.FrameResult(image, tensor, ax_meta, stream_id, ts, now), gst_task_meta

    def __iter__(self):
        try:
            if self._pre_sample is not None:
                stream_id, sample = self._pre_sample
                self._pre_sample = None
                if sample:
                    yield self.frame_from_sample(sample, stream_id)

            while not self.at_eos():
                for stream_id, sink in enumerate(self._appsinks):
                    sample = sink.try_pull_sample(Gst.MSECOND)
                    if sample:
                        yield self.frame_from_sample(sample, stream_id)
        finally:
            self.stop()

    def stop(self):
        if self.mqueue:
            self._stop_event.set()
            self._feeding_thread.join()
            self._stop_event.clear()
        if self.pipeline:
            self.pipeline.send_event(Gst.Event.new_eos())
            self.pipeline.set_state(Gst.State.NULL)
            self.pipeline = None

            if (
                (not self.mqueue)
                and self.pushed_frames
                and (self.pushed_frames != self.received_frames)
            ):
                LOG.error(
                    f"Pushed {self.pushed_frames} frames, but received {self.received_frames} frames."
                )
            elif self._appsinks:
                LOG.trace(f"Finished processing {self.received_frames} frames")

    def initialize_src(self, pipeline, name, loader, data_formatter):
        '''
        If the pipeline contains an appsrc element with the given name,
        connect the need-data signal to the loader callable.
        '''
        appsrc = pipeline.get_by_name(name)
        if not appsrc:
            raise ValueError(f"Unable to find appsrc element {name} in pipeline")
        if loader is None:
            raise ValueError("loader must be specified if src is specified")

        appsrc.connect("need-data", self._need_data)
        appsrc.connect("enough-data", self._enough_data)
        self.enough_data_signal = False

        queue = pipeline.get_by_name("queue_axelera_dataset")
        max_size_buffers = queue.get_property("max-size-buffers")
        max_size_bytes = queue.get_property("max-size-bytes")
        max_size_time = queue.get_property("max-size-time")
        LOG.trace(
            f"queue_axelera_dataset max size: {max_size_buffers} buffers, {max_size_bytes} bytes, {max_size_time} time"
        )

        self._stop_event = threading.Event()
        self._feeding_thread = utils.ExceptionThread(
            target=self._feed_data, args=(appsrc, loader, data_formatter)
        )

    def check_appsrc_queue_level(self, queue, show_info):
        '''Check the queue level and return delay seconds if it is nearing/over its capacity'''
        current_level_buffers = queue.get_property("current-level-buffers")
        current_level_bytes = queue.get_property("current-level-bytes")
        current_level_time = queue.get_property("current-level-time")

        max_size_buffers = queue.get_property("max-size-buffers")
        max_size_bytes = queue.get_property("max-size-bytes")
        max_size_time = queue.get_property("max-size-time")

        loading_buffer = current_level_buffers / max_size_buffers if max_size_buffers else 0
        loading_bytes = current_level_bytes / max_size_bytes if max_size_bytes else 0
        loading_time = current_level_time / max_size_time if max_size_time else 0
        max_loading = max(loading_buffer, loading_bytes, loading_time)
        if show_info:
            LOG.debug(
                f"Queue overrun, loading {loading_buffer*100:.2f}% buffers, "
                f"{loading_bytes*100:.2f}% bytes, {loading_time*100:.2f}% time"
            )
        return max_loading

    def _enough_data(self, appsrc):
        self.enough_data_signal = True

    def _need_data(self, appsrc, length):
        self.enough_data_signal = False

    def _feed_data(self, appsrc, loader, data_formatter):
        '''
        This is called when the appsrc needs more data
        It calls the data loader to get the next image
        and pushes it to the appsrc
        If the loader returns None, None, None, it will
        emit an end-of-stream signal
        '''
        # we need this limitation for VAAPI+GPU pipeline which runs too fast at preprocessing
        delay = 0.001
        while not self._stop_event.is_set():
            if not self.enough_data_signal:
                delay = self._feed_data_once(appsrc, loader, data_formatter)
                delay = 0.001 if delay is None else 0.01
            else:
                time.sleep(delay)
        LOG.trace("Feeding thread stopped")

    def _feed_data_once(self, appsrc, loader, data_formatter):
        try:
            data = next(loader)
        except StopIteration:
            appsrc.end_of_stream()
            return None
        batched_data = data_formatter(data)
        assert len(batched_data) == 1, "batch_size > 1 is not supported"
        data = batched_data[0]
        image_source = [data.img] if data.img else data.imgs
        assert self.is_pair_validation == (len(image_source) == 2), "pair validation mismatch"
        for idx, image in enumerate(image_source):
            w, h = image.size
            if image.color_format == types.ColorFormat.RGB:
                formata = types.ColorFormat.RGBA
            elif image.color_format == types.ColorFormat.BGR:
                formata = types.ColorFormat.BGRA
            else:
                raise NotImplementedError(f"Unsupported color format: {image.color_format}")
            frame = image.tobytes(formata)
            format_int = image.get_gst_format(formata)
            num_channels = 4
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
            self.pushed_frames += 1
            self.mqueue.put((data.img_id, data.ground_truth))
            batched_data, buffer, frame = None, None, None
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
    tasks: List[network.AxTask],
    mi: List,
    input_sources: List[str],
    hardware_caps: config.HardwareCaps,
    ax_precompiled_gst: str,
):
    manifests = [model.manifest for model in mi]
    manifest = manifests[0]
    model_names = [model.name for model in mi]

    fmwk = config.env.framework
    is_measure = input_sources == ['measurement']
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
        'input_video': input_sources[0],
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
    ref.update({f'input_video{n}': p for n, p in enumerate(input_sources)})
    ref.update({f'model_lib{n}': m.model_lib_file for n, m in enumerate(manifests)})
    ref.update({f'model_name{n}': t.model_info.name for n, t in enumerate(tasks)})
    ref.update({f'label_file{n}': _labels(fmwk, t.model_info.name) for n, t in enumerate(tasks)})
    LOG.debug(f"ref: {ref}")
    return utils.load_yaml_by_reference(ax_precompiled_gst, ref)


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


def _add_element_names(pipeline):
    counter = collections.defaultdict(int)
    res = []
    for pipe in pipeline:
        pipe_name, elements = next(iter(pipe.items()))
        elements = [_add_element_name(counter, e) for e in elements]
        res.append({pipe_name: elements})
    return res


def _build_input_pipeline(
    gst: gst_builder.Builder, tasks: list[network.AxTask], input: io.PipeInput
):
    stream_count = input.stream_count()
    for n in range(stream_count):
        idx = str(n)
        input.build_input_gst(gst, idx)
        for task in tasks:
            if isinstance(task.input, InputWithImageProcessing):
                task.input.build_gst(gst, idx)
        if gst.new_inference:
            gst.queue(connections={'src': 'inference-task0.sink_%u'})
        else:
            gst.identity(connections={'src': 'inference-funnel.sink_%u'})
    if not gst.new_inference:
        gst.axfunnel(name='inference-funnel')


def _build_pipeline(
    nn: network.AxNetwork, input: io.PipeInput, output: io.PipeOutput, hw_caps: config.HardwareCaps
) -> List[Dict[str, Any]]:
    gst = gst_builder.builder(hw_caps)
    _build_input_pipeline(gst, nn.tasks, input)

    for task in nn.tasks:
        if isinstance(task.input, Input):
            task.input.build_gst(gst, '')

    for taskn, task in enumerate(nn.tasks):
        if not gst.new_inference:
            conns = {
                'src_%u': [
                    f'queue_task{taskn}.sink',
                    f'queue_decoder_task{taskn}.sink',
                ]
            }
            gst.axtransform()
            gst.tee(name=f'input_tee{taskn}', connections=conns)
            gst.queue(name=f'queue_task{taskn}')
        else:
            # This is a marker for axinferencenet, it signals that the current instance
            # is complete and the next instance should begin
            gst.axtransform()
        if not isinstance(task.input, (InputWithImageProcessing, Input)):
            task.input.build_gst(gst, '')
        for op in task.preprocess:
            op.build_gst(gst, '')

        task.inference.build_inference_gst(gst, task.aipu_cores)
        dq = []
        if task.inference.device == 'aipu':
            dq = [
                operators.AxeleraDequantize(
                    model=task.inference.model,
                    inference_op_config=task.inference.config,
                    num_classes=task.model_info.num_classes,
                    task_category=task.model_info.task_category,
                    assigned_model_name=task.model_info.name,
                    taskn=taskn,
                )
            ]
        for op in dq + task.postprocess:
            op.build_gst(gst, '')
    output.build_output_gst(gst, input.stream_count())
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


def _samefile(a: str | Path, b: str | Path) -> bool:
    '''Like os.path.samefile, but does not fail if either path doesn't exist.

    Not perfect because it doesn't resolve symlinks, but good enough.
    '''
    a, b = os.path.abspath(a), os.path.abspath(b)
    return a == b


class GstPipe(base.Pipe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pipeline_cache = None
        self.agg_pads = None

    def get_pipe(self):
        return self.pipeline_cache

    def _get_source_elements(self, pipeline):
        """Find and return all source elements in the given pipeline."""
        sources = []
        for element in gst_helper._gst_iterate(pipeline.iterate_elements()):
            # Check if the element is a source (it has only source pads)
            pads = gst_helper._gst_iterate(element.iterate_pads())
            if not any(p for p in pads if p.get_direction() == Gst.PadDirection.SINK):
                sources.append(element)
        return sources

    def pause(self):
        for src in self._get_source_elements(self.pipeline_cache):
            if src.get_name().startswith("rtsp"):
                src.set_state(Gst.State.PAUSED)
            else:
                try:
                    ret = self.pipeline_cache.set_state(Gst.State.PAUSED)
                except Exception as e:
                    raise RuntimeError(f"Unable to pause stream: {e}")

                if ret == Gst.StateChangeReturn.FAILURE:
                    raise RuntimeError("Unable to set the pipeline to the pause state")
                return

    def play(self):
        for src in self._get_source_elements(self.pipeline_cache):
            if src.get_name().startswith("rtsp"):
                src.set_state(Gst.State.PLAYING)
            else:

                try:
                    ret = self.pipeline_cache.set_state(Gst.State.PLAYING)
                except Exception as e:
                    raise RuntimeError(f"Unable to play stream: {e}")
                if ret == Gst.StateChangeReturn.FAILURE:
                    raise RuntimeError("Unable to set the pipeline to the playing state")
                return

    def remove_source(self, agg_pad_name):
        if self.pipeline_cache == None:
            raise RuntimeError("Pipeline not yet created")

        self.pipeline_cache = gst_helper.gst_remove_source(agg_pad_name, self.pipeline_cache)

    def stream_select(self, streams):
        agg_name = 'inference-task0' if config.env.axinferencenet else 'inference-funnel'
        agg = self.pipeline_cache.get_by_name(agg_name)
        agg.set_property("stream_select", streams)

    def gen_newinput_gst(self, input, idx):
        gst = gst_builder.builder(self.hardware_caps)
        input.build_input_gst(gst, idx)

        # Process tasks for InputWithImageProcessing
        for task in self.nn.tasks:
            for op in filter(
                lambda op: isinstance(op, InputWithImageProcessing), [task.input, *task.preprocess]
            ):
                op.build_gst(gst, idx)

        # Set up the aggregator name and optionally add an identity element
        agg_name = 'inference-task0' if gst.new_inference else 'inference-funnel'
        if not gst.new_inference:
            gst.identity(connections={'src': f'{agg_name}.sink_%u'})

        # Ensure the pipeline cache exists and add the GStreamer input
        if self.pipeline_cache is None:
            raise RuntimeError("Pipeline not yet created")

        self.pipeline_cache, agg_pad_name = gst_helper.add_gst_input(gst, self.pipeline_cache)
        return agg_pad_name

    def gen_end2end_pipe(self, input, output):
        '''Construct gst E2E pipeline'''
        self.batched_data_reformatter = input.batched_data_reformatter
        if self.batched_data_reformatter:
            self.input_generator = input.create_generator()
        self.format = input.format
        # from generic Input, get the first model in the pipeline
        inputs = getattr(input, 'inputs', [input])

        sources = [i.location for i in inputs]

        pipeline = None
        if self.ax_precompiled_gst:
            mi = list(iter(self.model_infos.models()))
            pipeline = _parse_low_level_pipeline(
                self.nn.tasks, mi, sources, self.hardware_caps, self.ax_precompiled_gst
            )
        plugin_path = os.environ.get('GST_PLUGIN_PATH', '')
        if 'operators' not in plugin_path:
            extra = f"{config.env.framework}/operators/lib"
            os.environ['GST_PLUGIN_PATH'] = f"{extra}:{plugin_path}" if plugin_path else extra
        if pipeline is None:
            gst = _build_pipeline(self.nn, input, output, self.hardware_caps)
            pipeline = [{'pipeline': gst}]
            task_names = [t.model_info.name for t in self.nn.tasks]
            _save_axnet_files(gst, task_names, self.logging_dir)

        self.pipeout = output
        self.pipeout.initialize_writer(input)

        out_yaml = self.logging_dir / "gst_pipeline.yaml"
        if self.ax_precompiled_gst and _samefile(self.ax_precompiled_gst, out_yaml):
            LOG.debug(f"Not writing GST representation because it was passed in")
        else:
            out_yaml.write_text(yaml.dump(pipeline, sort_keys=False))
            LOG.debug(f"GST representation written to {os.path.relpath(out_yaml)}")

        # TODO: take off the following hard assignment of self.pipeline, it does
        # not need to be stored.
        self.pipeline = _add_element_names(pipeline)
        self.model_info_labels_dict = {t.name: t.model_info.labels for t in self.nn.tasks}
        self.model_info_num_classes_dict = {
            t.name: t.model_info.num_classes for t in self.nn.tasks
        }

    def get_agg_pads(self):
        return self.agg_pads

    def get_agg_pad(self, idx):
        if self.agg_pads is None:
            # Pipeline not yet created
            LOG.warning('Pipeline not yet started to add new source')
            return None

        if idx not in self.agg_pads:
            LOG.warning('Pipeline slot does not exist')
            return None
        return self.agg_pads[idx]

    def init_loop(self) -> Callable[[], None]:
        pipeline_names = [t.model_info.name for t in self.nn.tasks]
        with gst_helper.InitProgress() as progress:
            if LOG.isEnabledFor(logging_utils.TRACE):
                env = {k: v for k, v in sorted(os.environ.items()) if k.startswith('AX')}
                env = pprint.pformat(env, width=1, compact=True, depth=1)
                LOG.trace(f"environment at gst pipeline construction:\n%s", env)

            if self.pipeline_cache is None:
                self.pipeline_cache = gst_helper.build_gst_pipelines(
                    self.pipeline, pipeline_names, progress
                )[-1]
                new_inference = config.env.axinferencenet
                agg_pads = gst_helper.get_agg_pads(self.pipeline_cache, new_inference)

                self.agg_pads = {n: pad for n, pad in enumerate(agg_pads)}
            self.is_pair_validation = False

            args = ()
            if self.batched_data_reformatter:
                src_name = (
                    'axelera_dataset_src' if self.format == 'dataset' else 'axelera_server_src'
                )
                args = (src_name, self.input_generator, self.batched_data_reformatter)
                # check if it is pair validation
                data = next(iter(self.input_generator))
                if self.batched_data_reformatter:
                    data = self.batched_data_reformatter(data)[0]
                    self.is_pair_validation = data.imgs is not None
                else:
                    self.is_pair_validation = False
            stream = GstStream(
                self.pipeline_cache,
                self.logging_dir,
                self.hardware_caps,
                progress,
                *args,
            )
        return lambda: self._loop(stream)

    def _create_meta_instances_from_gst_decoded_meta(
        self, ax_meta, gst_meta_info, task_meta, master_meta
    ):
        gst_meta_key = gst_meta_info.task_name
        if type(task_meta) == meta.ClassificationMeta:
            model_meta = meta.ClassificationMeta(
                num_classes=self.model_info_num_classes_dict[gst_meta_key],
                # label is not important for measurement
                labels=self.model_info_labels_dict[gst_meta_key],
            )
            model_meta.transfer_data(task_meta)
        elif type(task_meta) == meta.ObjectDetectionMeta:
            model_meta = meta.ObjectDetectionMeta.create_immutable_meta(
                boxes=task_meta.boxes,
                scores=task_meta.scores,
                class_ids=task_meta.class_ids,
                labels=self.model_info_labels_dict[gst_meta_key],
                make_extra_info_mutable=True,
            )
        elif type(task_meta) == meta.TrackerMeta:
            model_meta = meta.TrackerMeta(
                tracking_history=task_meta.tracking_history,
                class_ids=task_meta.class_ids,
                object_meta=task_meta.object_meta,
                frame_object_meta=task_meta.frame_object_meta,
                labels=self.model_info_labels_dict[gst_meta_key],
                labels_dict=self.model_info_labels_dict,
            )
        elif type(task_meta) == meta.CocoBodyKeypointsMeta:
            model_meta = CocoBodyKeypointsMeta(
                keypoints=task_meta.keypoints,
                boxes=task_meta.boxes,
                scores=task_meta.scores,
            )
        elif type(task_meta) == meta.FaceLandmarkLocalizationMeta:
            model_meta = meta.FaceLandmarkLocalizationMeta(
                keypoints=task_meta.keypoints,
                boxes=task_meta.boxes,
                scores=task_meta.scores,
            )
        elif type(task_meta) == meta.InstanceSegmentationMeta:
            model_meta = meta.InstanceSegmentationMeta(self.model_info_labels_dict[gst_meta_key])
            model_meta.transfer_data(task_meta)
        elif type(task_meta) == meta.FaceLandmarkTopDownMeta:
            model_meta = FaceLandmarkTopDownMeta(
                _keypoints=task_meta._keypoints, _boxes=[], _scores=[]
            )
        elif type(task_meta) == meta.SemanticSegmentationMeta:
            model_meta = SemanticSegmentationMeta(
                shape=task_meta.shape,
                class_map=task_meta.class_map,
                probabilities=task_meta.probabilities,
                extra_info=task_meta.extra_info,
            )
        else:
            LOG.debug(f"Directly registering {type(task_meta)} into ax_meta")
            ax_meta.add_instance(gst_meta_info.task_name, task_meta, master_meta_name=master_meta)
            return

        ax_meta.add_instance(gst_meta_info.task_name, model_meta, master_meta_name=master_meta)

    def _handle_pair_validation(self, ax_meta, decoded_meta):
        gst_meta_key, task_meta = next(iter(decoded_meta.items()))
        embeddings = task_meta.embeddings
        model_meta = ax_meta.get_instance(
            gst_meta_key,
            meta.PairValidationMeta,
        )
        return model_meta.add_result(embeddings)

    def _loop(self, stream):
        try:
            for fr, decoded_meta in stream:
                if self._stop_event.is_set():
                    break

                ax_meta = fr.meta
                if self.is_pair_validation:
                    if not self._handle_pair_validation(ax_meta, decoded_meta):
                        continue
                else:
                    for gst_meta_info, task_meta in (decoded_meta or {}).items():
                        master_meta = self.task_graph.get_master(gst_meta_info.task_name)
                        has_master = master_meta and gst_meta_info.master
                        if has_master and gst_meta_info.master != master_meta:
                            raise ValueError(
                                f"From the YAML, the master of {gst_meta_info.task_name} is expected to be {master_meta}, but GST is sending it as {gst_meta_info.master}."
                            )
                        only_yaml_has_master = master_meta and not gst_meta_info.master
                        if only_yaml_has_master:
                            LOG.warning(
                                f"GST is treating {gst_meta_info.task_name} as a master meta, but from the YAML it is expected to be a submeta of {master_meta}."
                            )
                        only_gst_has_master = not master_meta and gst_meta_info.master
                        if only_gst_has_master:
                            LOG.warning(
                                f"GST is treating {gst_meta_info.task_name} as a submeta of {gst_meta_info.master}, but from the YAML it is expected to be a master meta."
                            )
                        self._create_meta_instances_from_gst_decoded_meta(
                            ax_meta, gst_meta_info, task_meta, gst_meta_info.master
                        )

                self.pipeout.sink(fr)
                self._callback(self.pipeout.result)
        except Exception as e:
            import traceback

            LOG.error(f"Pipeline error occurred: {str(e)}\n{traceback.format_exc()}")
            raise
        finally:
            stream.stop()
            self._callback(None)  # no further messages
            self.pipeout.close_writer()

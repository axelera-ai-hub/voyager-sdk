# Copyright Axelera AI, 2023
# Inference stream class
from __future__ import annotations

import collections
import copy
import dataclasses
import io
import itertools
import queue
import signal
import sys
import threading
import time
import traceback
from typing import TYPE_CHECKING, Iterable

from . import config, logging_utils, pipe, utils
from .pipe import FrameEventType

LOG = logging_utils.getLogger(__name__)


class _Sentinel:
    def __init__(self, name):
        self.name = name

    def __repr__(self):
        return self.name


_INTERRUPT_RAISED = _Sentinel('INTERRUPT_RAISED')
_PIPELINE_REMOVED = _Sentinel('PIPELINE_REMOVED')
_FRAMES_COMPLETED = _Sentinel('FRAMES_COMPLETED')


if TYPE_CHECKING:
    from axelera import types

    from .inf_tracers import TraceMetric, Tracer


class InterruptHandler:
    def __init__(self, stream=None):
        self.stream = stream
        self._interrupted = threading.Event()

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self)
            signal.signal(signal.SIGTERM, self)

    def __call__(self, *args):
        already_interrupted = self._interrupted.is_set()
        self._interrupted.set()
        if self.stream is not None:
            if already_interrupted:
                LOG.info("Interrupting the inference stream again, active threads:")
                try:
                    for t in list(threading.enumerate()):
                        LOG.info(f"Thread {t.name} {t.daemon=} {t.is_alive()=}")
                except Exception as e:
                    LOG.info(f"Failed to list active threads: {e}")
            else:
                LOG.info("Interrupting the inference stream")
            return self.stream.stop()
        else:
            LOG.error('Unable to stop stream')
            sys.exit(1)

    def is_interrupted(self):
        return self._interrupted.is_set()


def _create_manager(
    system: config.SystemConfig,
    stream: config.InferenceStreamConfig,
    pipeline: config.PipelineConfig,
    deploy: config.DeployConfig,
    tracers: list[Tracer],
    render_config: config.RenderConfig,
    id_allocator: pipe.SourceIdAllocator,
) -> pipe.PipeManager:
    if not pipeline.network:
        raise ValueError("network must be specified")
    if not pipeline.sources:
        raise ValueError("sources must be specified")
    if pipeline.pipe_type not in ['gst', 'torch', 'torch-aipu', 'quantized']:
        raise ValueError(
            f"Invalid pipe type: {pipeline.pipe_type}, valid types are gst, torch, torch-aipu, quantized"
        )
    try:
        return pipe.PipeManager(
            system,
            stream,
            pipeline,
            deploy,
            tracers,
            render_config,
            id_allocator,
        )
    except TypeError as e:
        raise TypeError(f"Invalid PipeManager configuration: {str(e)}") from e


def _calculate_num_frames(pipelines, requested_frames):
    nframes = [p.number_of_frames for p in pipelines]
    combined = [x for x in itertools.chain(nframes, [requested_frames]) if x > 0]
    return min(combined) if combined else 0


class _IterateWithLen:
    def __init__(self, iter: Iterable, len: int):
        self._iter = iter
        self._len = len

    def __iter__(self):
        return self._iter

    def __len__(self):
        return self._len


def _debug_log_sentinel(pipeline, msg, text):
    if isinstance(msg, _Sentinel):
        LOG.debug(text, f"{pipeline.name if pipeline else 'None'}: {msg!r}")


class InferenceStream:
    """An iterator that launches the inference pipeline locally
    and yields inference results for each frame"""

    def __init__(
        self,
        system_config: config.SystemConfig,
        stream_config: config.InferenceStreamConfig,
        deploy_config: config.DeployConfig,
        default_pipeline_config: config.PipelineConfig,
        pipeline_configs: list[config.PipelineConfig] | None = None,
        tracers: list[Tracer] | None = None,
    ):
        self._system_config = system_config
        self._stream_config = stream_config
        self._deploy_config = deploy_config
        self._default_pipeline_config = default_pipeline_config
        any_eval_mode = any(p.eval_mode for p in pipeline_configs or [])
        self.timeout = (
            stream_config.timeout if stream_config.timeout > 0 and not any_eval_mode else None
        )
        self._queue = queue.Queue(maxsize=10)
        self._interrupt_raised = False
        self._pipelines: list[pipe.PipeManager] = []
        self._removed_pipelines: set[pipe.PipeManager] = set()
        self._pending_evaluators: list[types.Evaluator] = []
        self._timer = utils.Timer()
        self._frames_requested = stream_config.frames
        self._frames_lock = threading.Lock()
        self._frames_executed = 0
        self._stream_lock = threading.Lock()
        self._interrupt_handler = InterruptHandler(self)
        self._id_allocator = pipe.SourceIdAllocator()
        self.hardware_caps = system_config.hardware_caps.detect_caps()
        self._queue_blocked = False
        self._tracers: list[Tracer] = tracers or []
        self._started = False
        for pc in pipeline_configs:
            self.add_pipeline(pc)
        self._frames = _calculate_num_frames(self._pipelines, stream_config.frames)

    def pipeline_by_stream_id(self, stream_id: int) -> pipe.PipeManager:
        """Find a pipeline by its stream ID.

        If not found, raises ValueError.
        """
        for pipeline in self._pipelines:
            if stream_id in pipeline.sources:
                return pipeline
        raise ValueError(f"Stream ID {stream_id} not found in any pipeline")

    def add_pipeline(
        self, pipeline_config: config.PipelineConfig | None = None, **kwargs
    ) -> pipe.PipeManager:
        '''Add a new pipeline to the stream.

        This pipeline will execute in parallel with any existing pipelines, and
        results will be yielded as they become available in the same inference
        loop.

        The configuration can either be provided as a PipelineConfig object, or
        as keyword arguments that will be used to create a PipelineConfig object.
        '''
        pipeline_config = pipeline_config or copy.deepcopy(self._default_pipeline_config)
        pipeline_config.update_from_kwargs(kwargs)
        other_pipelines = self._pipelines[:]
        if self._started:
            for p in other_pipelines:
                p.pause_pipe()
        try:
            manager = _create_manager(
                self._system_config,
                self._stream_config,
                pipeline_config,
                self._deploy_config,
                self._tracers,
                pipeline_config.render_config,
                self._id_allocator,
            )
            manager.init_pipe()
            manager.setup_callback(lambda result: self._feed_result(manager, result))
            self._pipelines.append(manager)
            if self._started:
                manager.run_pipe()
        finally:
            if self._started:
                for p in other_pipelines:
                    p.play_pipe()
        return manager

    def remove_pipeline(self, pipeline: pipe.PipeManager | int):
        """Remove a pipeline from the stream.

        Note that the pipeline will be stopped, and removed immediately. No more
        results will be received.

        Args:
            pipeline: The pipeline to remove, either a PipeManager instance or an index.
        """
        if isinstance(pipeline, int):
            if pipeline < 0 or pipeline >= len(self._pipelines):
                raise IndexError(
                    f"Pipeline index {pipeline} out of range [0, {len(self._pipelines)})"
                )
            pipeline = self._pipelines[pipeline]
        if pipeline not in self._pipelines:
            LOG.warning(f"Pipeline {pipeline.network.name} not found in the stream")
            return

        name = f'Pipeline {self._pipelines.index(pipeline)} ({pipeline.network.name})'
        pipeline.stop_pipe()
        LOG.debug(f"Stopped {name}, awaiting removal")
        while pipeline in self._pipelines:
            LOG.debug(f"{name} was not yet removed")
            time.sleep(0.1)
        for src_id in pipeline.sources.keys():
            self._id_allocator.deallocate(src_id)
        LOG.debug(f"{name} has been removed")

    @property
    def manager(self):
        if len(self._pipelines) > 1:
            raise RuntimeError(
                "More than one pipeline in the stream, please use pipelines property with multiple pipelines"
            )
        return self._pipelines[0]

    @property
    def pipelines(self):
        return self._pipelines[:]

    @property
    def sources(self) -> dict[int, config.Source]:
        '''Return the list of input sources'''
        return collections.ChainMap(*(p.sources for p in self._pipelines))

    @property
    def frames_executed(self) -> int:
        '''Return the number of frames that have been processed so far.'''
        with self._frames_lock:
            return self._frames_executed

    @property
    def frames(self) -> int:
        '''Return the number of frames to process, or 0 if there is no bounding condition.

        If all inputs are unbounded and the stream is not configured with a frame limit, this will
        be 0.

        If any input is bounded (eg a filesrc), or if the stream was configured to stop after a
        number this will return the minimum of all the non-zero boundaries.

        Typically used to implement a progress bar, it shold not be considered a reliable indicator
        of when the stream will finish, as the stream may be interrupted by a signal or other
        event.
        '''
        return self._frames_requested

    def __len__(self):
        return self._frames

    def _feed_result(self, pipeline: pipe.PipeManager, event: pipe.FrameEvent | None) -> bool:
        should_continue = True
        if self.is_interrupted:
            should_continue = False
            LOG.debug(f"Stream stopping, ignoring new results {event!r}")
            return should_continue

        msgs = []
        if event is None and pipeline is not None:
            LOG.debug(f"Received None from {pipeline.name}, removing pipeline")
        if event.type == FrameEventType.end_of_pipeline:
            LOG.debug(f"Pipeline {pipeline.name} implicit _PIPELINE_REMOVED")
            if pipeline not in self._removed_pipelines:
                msgs.append((pipeline, _PIPELINE_REMOVED))
        else:
            msgs.append((pipeline, event))

        if event is not None and event.result:
            with self._frames_lock:
                self._frames_executed += 1
                if self._frames_requested and self._frames_executed >= self._frames_requested:
                    LOG.debug(f"Reached frame limit ({self._frames_requested}), stopping stream")
                    for p in self._pipelines:
                        msgs.append((p, _PIPELINE_REMOVED))
                    msgs.append((None, _FRAMES_COMPLETED))
                    should_continue = False

        for p, msg in msgs:
            while not self.is_interrupted:
                try:
                    self._queue.put((p, msg), timeout=0.2)
                    qsize = self._queue.qsize()
                    if self._queue_blocked and qsize < self._queue.maxsize // 2:
                        LOG.info(f"InferenceStream queue is unblocked ({qsize=})")
                        self._queue_blocked = False
                    break
                except queue.Full:
                    if not self._queue_blocked:
                        qsize = self._queue.qsize()
                        LOG.warning(f"InferenceStream queue is full ({qsize=})")
                    self._queue_blocked = True
        with self._stream_lock:
            for p, msg in msgs:
                _debug_log_sentinel(p, msg, "Emitting %s")
                if msg is _PIPELINE_REMOVED and p in self._pipelines:
                    self._pipelines.remove(p)
                    self._removed_pipelines.add(p)
                    if p.evaluator:
                        self._pending_evaluators.append(p.evaluator)
        return should_continue

    def _iter(self) -> Iterable[pipe.FrameEvent]:
        if not self._pipelines and self._queue.qsize() == 0:
            raise ValueError(
                "No pipeline configs provided, please add a pipeline using add_pipeline()"
            )
        for pipeline in self._pipelines:
            pipeline.run_pipe()
        self._started = True
        self._timer.reset()
        try:
            while self._pipelines or self._queue.qsize() > 0:
                if self.is_interrupted:
                    LOG.debug("Stream interrupted, stopping...")
                    break
                try:
                    p, msg = self._queue.get(timeout=self.timeout)
                    _debug_log_sentinel(p, msg, "Got %s")
                    if self._interrupt_raised:
                        LOG.debug("Stream interrupted, stopping...")
                        break
                    if msg is _PIPELINE_REMOVED:
                        continue
                    if msg in (_INTERRUPT_RAISED, _FRAMES_COMPLETED):
                        break
                    if p.evaluator:
                        result = msg.result
                        # TODO consider if this should happen in the _feed_result thread
                        p.evaluator.append_new_sample(result.meta)

                except queue.Empty:  # timeout
                    LOG.warning("Timeout for querying an inference")
                    raise RuntimeError('Timeout for querying an inference') from None
                # Reset the timer after the first two frames which are usually very slow
                if self.frames_executed == 2:
                    self._timer.reset()
                yield msg
        except Exception as e:
            LOG.error(f'InferenceStream terminated due to {str(e)}')
            LOG.debug(f'{traceback.format_exc()}')
            raise
        finally:
            LOG.trace("InferenceStream._iter() is in finally")
            self._timer.stop()
            with self._stream_lock:
                for evaluator in self._pending_evaluators:
                    self._report_summary(evaluator)
            LOG.trace("InferenceStream._iter() has completed")

    def without_events(self) -> Iterable[pipe.FrameResult]:
        '''Iterate over the stream, yielding only FrameResult objects.

        Any FrameEvents will be ignored, with a warning logged.
        '''
        for x in self._iter():
            if x.result:
                yield x.result
            else:
                LOG.warning(f"Ignoring event {x!r}")
                LOG.warning_once("Use stream.with_events() to receive events")

    def with_events(self) -> Iterable[pipe.FrameEvent]:
        '''Iterate over the stream, yielding FrameEvent objects.

        FrameEvent objects are used to notify the application of events such as end-of-stream,
        errors, or other conditions. if the event has a result property then it is of
        type FrameResult, which is the same object returned by the normal iterator of the stream.

        The advantage of using `with_events` is that it allows the application to
        respond to non inference events, for example to handle end-of-stream or errors.

        stream = create_inference_stream(network="resnet18-imagenet", sources=['usb:0'])
        for event in stream.with_events():
            if fr := event.result:
                window.show(fr.image, fr.meta, stream_id=fr.stream_id)
            elif event.type == FrameEventType.stream_error:
                print(f"Stream {event.stream_id} had an error and has been closed: {event.message}")
            else:
                print(f"Stream {event.stream_id} had an event: {event.type} {event.message}")

        '''
        return _IterateWithLen(self._iter(), len(self))

    __iter__ = without_events

    def stream_select(self, streams: Iterable[int] | str) -> None:
        '''Configure streams to be in paused or resumed state.

        Args:
            streams: A list of stream IDs that should be in the playing state.

        NOTE: This is only supported by GStreamer pipelines.
        NOTE: For compatiblity reasons a string of the form '0,2,3' is also supported, but
              this is deprecated and will raise a DeprecationWarning.
        '''

        used = set()
        all = set(streams)
        with self._stream_lock:
            for pipeline in self._pipelines:
                this_pipeline = pipeline.sources.keys() & all
                pipeline.stream_select(list(this_pipeline))
                used |= this_pipeline
        if used != all:
            LOG.warning("Did not find the correct pipeline for the following stream_id ")

    def get_stream_select(self) -> list[int]:
        '''Get the list of currently playing streams, as configured by stream_select().

        NOTE: This is only supported by GStreamer pipelines.
        '''

        sids = set()
        with self._stream_lock:
            for p in self._pipelines:
                sids |= set(p.get_stream_select())
        valid_ids = set(self.sources.keys())
        if paused := valid_ids - sids:
            spaused = ' '.join(str(x) for x in paused)
            LOG.debug(f"stream ids in source but not in stream_select (paused): {spaused}")
        if unknown := sids - valid_ids:
            sunknown = ' '.join(str(x) for x in unknown)
            LOG.error(f"stream ids in stream_select but not in sources (unknown): {sunknown}")
        return sorted(sids)

    def add_source(
        self,
        source: str | config.Source,
        source_id: int = -1,
        pipeline: pipe.PipeManager | int = 0,
    ) -> int:
        '''Add a new source to the pipeline.

        Args:
            source: The source to add.
            source_id: The source id of the source to add. If -1, a new source_id will be assigned.
            pipeline: Reference to the pipeline to add, or the index of the pipeline to add to.
        Returns:
            The source id of the new source.
        NOTE: This is only supported by GStreamer pipelines.
        '''
        source = config.Source(source)
        pipeline = pipeline or self._pipelines[pipeline]
        with self._stream_lock:
            pipeline.pause_pipe()
            source_id = pipeline.add_source(source, source_id)
            pipeline.play_pipe()
        LOG.info(f"Added new source: {source} as {source_id=}")
        return source_id

    def remove_source(self, source_id: int):
        '''Remove a source from the pipeline.

        Args:
            source_id: The source id of the source to remove.
        NOTE: This is only supported by GStreamer pipelines.
        '''
        pipelines = {sid: p for p in self._pipelines for sid in p.sources.keys()}
        pipeline = pipelines[source_id]
        with self._stream_lock:
            pipeline.pause_pipe()
            pipeline.remove_source(source_id)
            pipeline.play_pipe()

    @property
    def is_interrupted(self) -> bool:
        '''Returns True if the stream has been interrupted by a signal.'''
        return self._interrupt_raised or self._interrupt_handler.is_interrupted()

    def stop(self):
        self._interrupt_raised = True
        # put a None, but if the queue is full flush it, do this in a loop until we succeed
        # otherwise the queue may be being fed by the producer
        for p in self._pipelines:
            p.stop_pipe()
        while 1:
            try:
                self._queue.put((None, _INTERRUPT_RAISED), timeout=0)  # unblock the queue
                LOG.debug("Stream stop requested")
                break
            except queue.Full:
                pass
            while not self._queue.empty():
                try:
                    self._queue.get(timeout=0)
                except queue.Empty:
                    break

    def is_single_image(self) -> bool:
        '''True if any input stream is a single image.'''
        return any(p.is_single_image() for p in self._pipelines)

    def _report_summary(self, evaluator: types.Evaluator) -> None:
        '''Report the summary of the evaluator.'''
        duration_s = self._timer.time
        evaluator.evaluate_metrics(duration_s)
        output = io.StringIO()
        evaluator.write_metrics(output)
        LOG.info(output.getvalue().strip())

    def get_all_metrics(self) -> dict[str, TraceMetric]:
        '''Return all tracer metrics.

        The available tracer metrics will depend on those that were passed to the PipeManager
        (or create_inference_stream) at construction.

        See examples/application.py for an example of how to use this method.
        '''
        metrics = {}
        for p in self._pipelines:
            # TODO should we have tracers per pipeline or per InferenceStream?
            for t in p.tracers:
                metrics.update({m.key.strip('_'): m for m in t.get_metrics()})
        return metrics


def create_inference_stream(
    system_config: config.SystemConfig | None = None,
    stream_config: config.InferenceStreamConfig | None = None,
    pipeline_configs: config.PipelineConfig | list[config.PipelineConfig] | None = None,
    logging_config: config.LoggingConfig | None = None,
    deploy_config: config.DeployConfig | None = None,
    *,
    log_level: int | None = None,
    tracers: list[Tracer] | None = None,
    **kwargs,
) -> InferenceStream:
    """Factory function to create appropriate stream type.

    Args:
        system_config: Optional SystemConfig object, if not provided, a default will be created.
        stream_config: Optional InferenceConfig object, if not provided, a default will be created.
        pipeline_configs: Optional PipelineConfig objects, if not provided, and suitable kwargs are found
        logging_config: Optional LoggingConfig object, if not provided, a default will be created.
        deploy_config: Optional DeployConfig object, if not provided, a default will be created.
        then a PipelineConfig will be created.

    Additional keyword only arguments:
        log_level: Optional log level, if provided this will override that set in logging_config.
        tracers: Optional list of Tracer objects, if not provided no tracers will be configured.

    Additional keyword arguments:
        All other keyword arguments are passed to the SystemConfig, InferenceStreamConfig and
        PipelineConfig as appropriate this allows you to override any of the default values in the
        configs.  For example `allow_hardware_codec=True` will override the value of
        `allow_hardware_codec` in the SystemConfig (whether SystemConfig was passed in or a default
        created).

        **kwargs: these kwargs will override any of the settings in the above configs
    Returns:
        InferenceStream: Configured inference stream

    TODO        Blah de blah
    For example:

            parser = config.create_inference_argparser()
            args = parser.parse_args()
            stream = stream.create_inference_stream(
                config.SystemConfig.from_parsed_args(args),
                config.InferenceStreamConfig.from_parsed_args(args),
                config.PipelineConfig.from_parsed_args(args),
            )


    """
    if logging_config is None:
        logging_config = config.LoggingConfig()
    if log_level is not None:
        logging_config.console_level = log_level
    logging_utils.configure_logging(logging_config)

    system_config = system_config or config.SystemConfig()
    system_config.update_from_kwargs(kwargs)
    system_config.hardware_caps = system_config.hardware_caps.detect_caps()
    stream_config = stream_config or config.InferenceStreamConfig()
    stream_config.update_from_kwargs(kwargs)
    deploy_config = deploy_config or config.DeployConfig()
    deploy_config.update_from_kwargs(kwargs)
    default = config.PipelineConfig.from_kwargs(kwargs)
    if isinstance(pipeline_configs, config.PipelineConfig):
        # if a single pipeline config is passed, convert it to a list
        pipeline_configs = [pipeline_configs]
    pipeline_configs = pipeline_configs or []
    if default.sources or default.ax_precompiled_gst:
        # create a pipeline config from kwargs
        pipeline_configs.append(default)
        # then reset the defaults for the active fields
        d = config.PipelineConfig()
        default = dataclasses.replace(
            default, sources=d.sources, ax_precompiled_gst=d.ax_precompiled_gst
        )

    if kwargs:
        unexp = ', '.join(kwargs.keys())
        all_valid = (
            config.SystemConfig.valid_kwargs()
            | config.InferenceStreamConfig.valid_kwargs()
            | config.PipelineConfig.valid_kwargs()
            | config.DeployConfig.valid_kwargs()
        )
        valid = ', '.join(f"{k}" for k in all_valid)
        raise ValueError(f"Unexpected keyword arguments: {unexp}, valid kwargs are {valid}")
    return InferenceStream(
        system_config, stream_config, deploy_config, default, pipeline_configs, tracers
    )

# Copyright Axelera AI, 2025
# Inference stream class
from __future__ import annotations

import dataclasses
import io
import queue
import signal
import sys
import threading
from typing import TYPE_CHECKING

from . import config, logging_utils, pipe, utils

LOG = logging_utils.getLogger(__name__)

if TYPE_CHECKING:
    from .inf_tracers import TraceMetric


class InterruptHandler:
    def __init__(self, stream=None):
        self.stream = stream
        self._interrupted = threading.Event()

        if threading.current_thread() is threading.main_thread():
            signal.signal(signal.SIGINT, self)
            signal.signal(signal.SIGTERM, self)

    def __call__(self, *args):
        self._interrupted.set()
        if self.stream is not None:
            LOG.info("Interrupting the inference stream")
            return self.stream.stop()
        else:
            LOG.error('Unable to stop stream')
            sys.exit(1)

    def is_interrupted(self):
        return self._interrupted.is_set()


@dataclasses.dataclass
class BaseInferenceConfig:
    """Base class for inference config"""

    timeout: int = 0
    log_level: int = logging_utils.INFO  # INFO, DEBUG, TRACE

    def __post_init__(self):
        logging_utils.configure_logging(logging_utils.Config(self.log_level))


@dataclasses.dataclass
class InferenceConfig(BaseInferenceConfig):
    """Configuration for local inference mode"""

    network: str = dataclasses.field(default='')
    sources: list[str] = dataclasses.field(default_factory=list)
    pipe_type: str = dataclasses.field(default='gst')
    hardware_caps: config.HardwareCaps = dataclasses.field(
        default_factory=lambda: config.HardwareCaps(
            config.HardwareEnable.detect,
            config.HardwareEnable.detect,
            config.HardwareEnable.detect,
        )
    )
    frames: int = 0
    # Store additional configuration parameters
    extra_params: dict = dataclasses.field(default_factory=dict)

    @classmethod
    def from_kwargs(cls, network: str, sources: list[str], **kwargs) -> 'InferenceConfig':
        """Create config from kwargs, with required network and sources parameters"""
        known_fields = cls.__dataclass_fields__.keys()
        config_kwargs = {k: v for k, v in kwargs.items() if k in known_fields}
        extra_kwargs = {k: v for k, v in kwargs.items() if k not in known_fields}
        config_kwargs['extra_params'] = extra_kwargs

        instance = cls(network=network, sources=sources, **config_kwargs)
        return instance

    def __post_init__(self):
        super().__post_init__()
        if not self.network:
            raise ValueError("network must be specified")
        if not self.sources:
            raise ValueError("sources must be specified")
        if self.pipe_type not in ['gst', 'torch', 'torch-aipu']:
            raise ValueError(f"Invalid pipe type: {self.pipe_type}")

        # Pass both standard args and extra params to PipeManager
        pipe_args = {
            'network_path': self.network,
            'sources': self.sources,
            'pipe_type': self.pipe_type,
            'hardware_caps': self.hardware_caps,
            **self.extra_params,
        }
        try:
            self._pipe_mgr = pipe.PipeManager(**pipe_args)
        except TypeError as e:
            raise TypeError(f"Invalid PipeManager configuration: {str(e)}") from e

    @property
    def pipe_mgr(self):
        return self._pipe_mgr


class InferenceStream:
    """An iterator that launches the inference pipeline locally
    and yields inference results for each frame"""

    def __init__(self, pipe_mgr: pipe.PipeManager, frames: int = 0, timeout: int = 0):
        if pipe_mgr is None:
            raise ValueError("pipe_mgr must be provided")
        self.timeout = timeout if timeout > 0 and not pipe_mgr.eval_mode else None
        self._queue = queue.Queue()
        self.pipe_mgr = pipe_mgr
        self.pipe_mgr.setup_callback(self.feed_result)
        _pipe_frames = self.pipe_mgr.number_of_frames
        if _pipe_frames > 0:
            self.frames = min(_pipe_frames, frames) if frames > 0 else _pipe_frames
        else:
            self.frames = frames
        self._interrupt_raised = False
        self._timer = utils.Timer()
        self._frames_executed = 0
        self.pipe_mgr.init_pipe()
        self.lock = threading.Lock()
        self._interrupt_handler = InterruptHandler(self)

    @classmethod
    def from_config(cls, config: InferenceConfig) -> 'InferenceStream':
        """Alternative constructor using InferenceConfig"""
        return cls(
            pipe_mgr=config.pipe_mgr,
            frames=config.frames,
            timeout=config.timeout,
        )

    @property
    def manager(self):
        return self.pipe_mgr

    @property
    def sources(self) -> list[str]:
        '''Return the list of input sources'''
        return self.pipe_mgr.sources

    def __len__(self):
        return self.frames

    def feed_result(self, result: pipe.FrameResult | None):
        self._queue.put(result)

    def __iter__(self):
        self.pipe_mgr.run_pipe()
        self._timer.reset()
        try:
            n = 0
            while not self.frames or n < self.frames:
                if self._interrupt_raised or self._interrupt_handler.is_interrupted():
                    print("Interrupted")
                    break
                try:
                    result = self._queue.get(block=True, timeout=self.timeout)
                    if result is None or self._interrupt_raised:
                        break
                    if self.pipe_mgr.evaluator:
                        self.pipe_mgr.evaluator.append_new_sample(result.meta)

                except queue.Empty:  # timeout
                    LOG.warning("Timeout for querying an inference")
                    raise RuntimeError('timeout for querying an inference') from None
                if n == 1:
                    # Reset the timer after the first two frames which are usually very slow
                    self._timer.reset()
                yield result
                n += 1
        finally:
            self._frames_executed = max(n - 2, 0)
            self._timer.stop()
            self.pipe_mgr.stop_pipe()
            self.report_summary()

    def stream_select(self, streams):
        with self.lock:
            self.pipe_mgr.stream_select(streams)
            LOG.info(f"Active streams: {streams}")

    def add_source(self, source, idx=-1):
        with self.lock:
            LOG.info(f"Adding new source: {source}")
            self.pipe_mgr.pause_pipe()
            if idx == -1:
                idx = self.pipe_mgr.add_source(source)
            else:
                self.pipe_mgr.add_source_with_id(source, idx)

            self.pipe_mgr.play_pipe()
            LOG.info(f"New source: {source} added {idx}")
            return idx

    def remove_source(self, idx):
        with self.lock:
            LOG.info(f"Removing slot: {idx}")
            self.pipe_mgr.pause_pipe()
            self.pipe_mgr.remove_source(idx)
            self.pipe_mgr.play_pipe()
            LOG.info(f"Slot: {idx} removed")

    def stop(self):
        self._interrupt_raised = True
        self._queue.put(None)  # unblock the queue

    def is_single_image(self) -> bool:
        '''True if the input stream is a single image.'''
        return self.pipe_mgr.is_single_image()

    def report_summary(self):
        '''When evaluating, report the summary of the evaluation.'''
        if self.pipe_mgr.evaluator:
            duration_s = self._timer.time
            self.pipe_mgr.evaluator.evaluate_metrics(duration_s)
            output = io.StringIO()
            self.pipe_mgr.evaluator.write_metrics(output)
            LOG.info(output.getvalue().strip())

    def get_all_metrics(self) -> dict[str, TraceMetric]:
        '''Return all tracer metrics.

        The available tracer metrics will depend on those that were passed to the PipeManager
        (or create_inference_stream) at construction.

        See examples/application.py for an example of how to use this method.
        '''
        metrics = {}
        for t in self.pipe_mgr.pipeout.tracers:
            metrics.update({m.key.strip('_'): m for m in t.get_metrics()})
        return metrics


def create_inference_stream(config: InferenceConfig | None = None, **kwargs) -> InferenceStream:
    """Factory function to create appropriate stream type
    Args:
        config: Optional InferenceConfig object
        **kwargs: If config is None, these kwargs will be used to construct an InferenceConfig
    Returns:
        InferenceStream: Configured inference stream
    """
    if config is None:
        if 'network' not in kwargs or 'sources' not in kwargs:
            raise ValueError(
                "When config is not provided, 'network' and 'sources' must be specified in kwargs"
            )
        config = InferenceConfig.from_kwargs(**kwargs)
    elif kwargs:
        raise ValueError("Cannot specify both config object and kwargs")

    return InferenceStream.from_config(config)

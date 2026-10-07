# Copyright Axelera AI, 2023
# Construct application pipeline
from __future__ import annotations

import abc
import collections
import itertools
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Callable, Iterable

from .. import logging_utils, network, utils

if TYPE_CHECKING:
    from . import frame_data, graph, io
    from .. import config, device_manager

    EventCallback = Callable[[frame_data.FrameEvent], bool]

LOG = logging_utils.getLogger(__name__)


class SourceIdAllocator:
    '''A simple allocator for source IDs, which can be reused after deallocation.'''

    def __init__(self):
        self._next = iter(itertools.count())
        self._free = collections.deque()

    def allocate(self) -> int:
        '''Return the next available source ID, and reserve it.'''
        if self._free:
            return self._free.popleft()
        return next(self._next)

    def deallocate(self, sid: int) -> None:
        '''Make the given source ID available for reuse.'''
        self._free.append(sid)


class Pipe(abc.ABC):
    device_man: device_manager.DeviceManager
    pipeline = None
    nn: network.AxNetwork = None
    build_root: Path
    logging_dir: Path
    model_infos: network.ModelInfos
    task_graph: graph.DependencyGraph
    output: io.PipeOutput = None
    _on_event: EventCallback | None = None

    def __init__(
        self,
        device_man,
        network,
        logging_dir,
        hardware_caps,
        pipeline_config: config.PipelineConfig,
        task_graph,
        on_event: EventCallback | None,
        pipein: io.PipeInput,
    ) -> None:
        self.device_man = device_man
        self.nn = network
        self.logging_dir = logging_dir
        self.model_infos = self.nn.model_infos  # TODO remove this
        self.hardware_caps = hardware_caps or {}
        self.config = pipeline_config
        self.task_graph = task_graph
        self._on_event = on_event
        self._pipein = pipein
        self._stop_event = None
        self._thread = None
        self._loopfn = None

    def init(self):
        self._stop_event = threading.Event()
        self._loopfn = self.init_loop()

    def run(self):
        self._thread = utils.ExceptionThread(target=self._loopfn, name="PipeThread")
        self._thread.start()

    def stop(self):
        if self._stop_event is not None:
            self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
        for task in self.nn.tasks:
            if task.inference:
                task.inference.release()

    @abc.abstractmethod
    def init_loop(self) -> Callable[[], None]:
        pass

    def stream_select(self, streams: Iterable[int] | str) -> None:
        '''Configure streams to be in paused or resumed state.

        Args:
            streams: A list of stream IDs that should be in the playing state.

        NOTE: This is only supported by GStreamer pipelines.
        NOTE: For compatiblity reasons a string of the form '0,2,3' is also supported, but
              this is deprecated and will raise a DeprecationWarning.
        '''
        del streams
        p = type(self).__name__
        raise NotImplementedError(f"Streams cannot be paused or resumed in {p} pipeline")

    def get_stream_select(self) -> list[int]:
        '''Get the list of currently playing streams, as configured by stream_select().

        NOTE: This is only supported by GStreamer pipelines.
        '''
        p = type(self).__name__
        raise NotImplementedError(f"Streams cannot be paused or resumed in {p} pipeline")


def create_pipe(
    device_man: device_manager.DeviceManager,
    pipeline_config: config.PipelineConfig,
    nn: network.AxNetwork,
    logging_dir: Path,
    hardware_caps: config.HardwareCaps,
    task_graph: graph.DependencyGraph,
    result_ready_callback: EventCallback | None,
    pipein: io.PipeInput,
) -> Pipe:
    '''Factory function for AxPipe.'''
    if pipeline_config.pipe_type == 'gst':
        from .gst import GstPipe as Pipe
    elif pipeline_config.pipe_type == 'torch':
        from .torch import TorchPipe as Pipe
    elif pipeline_config.pipe_type == 'torch-aipu':
        from .torch import TorchAipuPipe as Pipe
    elif pipeline_config.pipe_type == 'quantized':
        from .torch import QuantizedPipe as Pipe
    else:
        raise RuntimeError(f"Not supported for {pipeline_config.pipe_type}")
    return Pipe(
        device_man,
        nn,
        logging_dir,
        hardware_caps,
        pipeline_config,
        task_graph,
        result_ready_callback,
        pipein,
    )

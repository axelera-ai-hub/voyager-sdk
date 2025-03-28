# Copyright Axelera AI, 2024
# Construct application pipeline
from __future__ import annotations

import abc
from pathlib import Path
import threading
from typing import TYPE_CHECKING, Any, Callable, Dict, Optional, Union

from axelera import types

from .. import logging_utils, network, operators, pipeline, utils

if TYPE_CHECKING:
    from . import frame_data, graph, io
    from .. import config, device_manager

    ResultCallback = Callable[[frame_data.FrameResult], None]

LOG = logging_utils.getLogger(__name__)


class Pipe(abc.ABC):
    device_man: device_manager.DeviceManager
    pipeline = None
    nn: network.AxNetwork = None
    build_root: Path
    logging_dir: Path
    model_infos: network.ModelInfos
    task_graph: graph.DependencyGraph
    output: io.PipeOutput = None
    _callback: Optional[ResultCallback] = None

    def __init__(
        self,
        device_man,
        network,
        logging_dir,
        hardware_caps,
        ax_precompiled_gst,
        task_graph,
    ) -> None:
        self.device_man = device_man
        self.nn = network
        self.logging_dir = logging_dir
        self.model_infos = self.nn.model_infos  # TODO remove this
        self.batched_data_reformatter = None
        self.hardware_caps = hardware_caps or {}
        self.ax_precompiled_gst = ax_precompiled_gst
        self.task_graph = task_graph

    @abc.abstractmethod
    def gen_end2end_pipe(self, input: io.BaseInput, output: io.PipeOutput):
        ...

    def get_pipe(self):
        return None

    def setup_callback(self, callback: ResultCallback):
        self._callback = callback

    def init(self):
        self._stop_event = threading.Event()
        self._loopfn = self.init_loop()

    def run(self):
        self._thread = utils.ExceptionThread(target=self._loopfn, name="PipeThread")
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        self._thread.join()
        for task in self.nn.tasks:
            task.inference.release()

    @abc.abstractmethod
    def init_loop(self) -> Callable[[], None]:
        pass

    def gen_network_pipe(self):
        '''
        Construct pytorch or gst pipeline. For pytorch path, models are constructed
        from YAML model config.
        '''
        for task in self.nn.tasks:
            compiled_model_dir = self.model_infos.manifest_path(task.model_info.name).parent
            task.model_info = self.model_infos.model(task.model_info.name)
            the_manifest = self._get_manifest(task.model_info)
            model_or_manifest = (
                self._get_model(task.model_info) if the_manifest is None else the_manifest
            )
            if len(task.preprocess) == 0:
                # take preprocess from types.Model::override_preprocess
                the_model = (
                    self._get_model(task.model_info)
                    if the_manifest is not None
                    else model_or_manifest
                )
                self.nn.attach_model_specific_preprocess(the_model, task)
            input_tensor_layout = task.model_info.input_tensor_layout
            pipeline.update_pending_expansions(task)
            task.inference = operators.Inference(
                self.device_man,
                compiled_model_dir,
                task.model_info.name,
                model_or_manifest,
                input_tensor_layout,
                task.inference_config,
            )

    def propagate_model_and_context_info(self):
        previous_context = None
        for taskn, task in enumerate(self.nn.tasks):
            compiled_model_dir = self.model_infos.manifest_path(task.model_info.name).parent
            where = getattr(task.input, 'where', '')
            if previous_context:
                task.context.update(previous_context)
            op_list = [task.input]
            if isinstance(task.input, operators.InputWithImageProcessing):
                op_list += task.input.image_processing
            elif isinstance(task.input, operators.InputFromROI):
                op_list += task.input.image_processing_on_roi
            op_list += task.preprocess + [task.inference] + task.postprocess
            for op in op_list:
                op.configure_model_and_context_info(
                    task.model_info, task.context, task.name, taskn, where, compiled_model_dir
                )
            previous_context = task.context.propagate()

    def _get_model(self, model_info: types.ModelInfo) -> types.Model:
        with self.nn.from_model_dir(model_info.name):
            model = self.nn.instantiate_model(model_info.name)
        return model

    def _get_manifest(self, model_info: types.ModelInfo) -> types.Manifest:
        return model_info.manifest


def create_pipe(
    device_man: device_manager.DeviceManager,
    pipe_type: str,
    nn: network.AxNetwork,
    logging_dir: Path,
    hardware_caps: config.HardwareCaps,
    ax_precompiled_gst: str,
    task_graph: graph.DependencyGraph,
) -> Pipe:
    '''Factory function for AxPipe.'''
    if pipe_type == 'gst':
        from .gst import GstPipe as Pipe
    elif pipe_type == 'torch':
        from .torch import TorchPipe as Pipe
    elif pipe_type == 'torch-aipu':
        from .torch import TorchAipuPipe as Pipe
    elif pipe_type == 'quantized':
        from .torch import QuantizedPipe as Pipe
    else:
        raise RuntimeError(f"Not supported for {pipe_type}")
    return Pipe(
        device_man,
        nn,
        logging_dir,
        hardware_caps,
        ax_precompiled_gst,
        task_graph,
    )

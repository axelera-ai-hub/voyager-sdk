# Copyright Axelera AI, 2024
# functions for deploying pipeline and base object for building pipeline
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time
from typing import TYPE_CHECKING, Optional

from . import base, graph, io
from .. import (
    config,
    device_manager,
    logging_utils,
    network,
    pipeline,
    transforms,
    utils,
    yaml_parser,
)

if TYPE_CHECKING:
    from . import inf_tracers, network


LOG = logging_utils.getLogger(__name__)


def _deploy_one_model_via_subprocess(
    model: str,
    nn_name: str,
    data_root: str,
    build_root: Path,
    pipe_type: str,
    num_cal_images: int,
    batch: int,
    hardware_caps: config.HardwareCaps,
    aipu_cores: int,
    metis: config.Metis,
):
    cap_argv = hardware_caps.as_argv()
    if cap_argv:
        cap_argv = ' ' + cap_argv
    if pipe_type == 'torch':
        cores = ''
        mode = ''
    else:
        s = 's' if aipu_cores > 1 else ''
        cores = f' for {aipu_cores} core{s}. This may take a while...'
        mode = ' --mode quantize_debug' if pipe_type == 'quantized' else ''
    LOG.info(f"Deploying model {model}{cores}")
    run_dir = config.env.framework
    try:
        with utils.spinner():
            run(
                f'{run_dir}/deploy.py --model {model} --num-cal-images {num_cal_images} '
                f'--calibration-batch {batch}{cap_argv} '
                f'--data-root {data_root} --pipe {pipe_type} --build-root {build_root} {nn_name} '
                f'--aipu-cores {aipu_cores} --metis {metis.name}{mode}',
            )

    except subprocess.CalledProcessError as e:
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        LOG.error(f"Failed to deploy model {model}")
        raise


def _get_or_deploy_models(
    user_specifed_nn_name_or_path: str,
    nn: network.AxNetwork,
    data_root: str,
    build_root: Path,
    pipe_type: str,
    num_cal_images: int,
    batch: int,
    hardware_caps: config.HardwareCaps,
    metis: config.Metis,
) -> network.ModelInfos:
    requested_aipu_cores = hardware_caps.aipu_cores
    nn_dir = build_root / nn.name
    model_infos = network.read_deployed_model_infos(
        nn_dir, nn, pipe_type, requested_aipu_cores, metis
    )
    model_infos.add_label_enums(nn.datasets)
    if not model_infos.ready:
        for model in model_infos.missing():
            exec_cores = model_infos.determine_execution_cores(model, requested_aipu_cores, metis)
            deploy_cores = model_infos.determine_deploy_cores(model, exec_cores, metis)
            if deploy_cores < exec_cores and deploy_cores > 1:
                LOG.info(f"{model} is restricted to deploy and run for up to {deploy_cores} cores")
            elif deploy_cores < exec_cores and deploy_cores == 1:
                LOG.warning(
                    "This model is restricted to deploy for single-core (but can be run using multiple cores)."
                )
            else:
                LOG.debug(f"Model deploy cores is {deploy_cores}")
            _deploy_one_model_via_subprocess(
                model,
                user_specifed_nn_name_or_path,
                data_root,
                build_root,
                pipe_type,
                num_cal_images,
                batch,
                hardware_caps,
                deploy_cores,
                metis,
            )

        model_infos = network.read_deployed_model_infos(
            nn_dir, nn, pipe_type, requested_aipu_cores, metis
        )
        model_infos.check_ready()
    return model_infos


def run(cmd, shell=True, check=True, verbose=False, capture_output=True):
    if verbose:
        print(cmd)
    try:
        result = subprocess.run(
            cmd, shell=shell, check=check, capture_output=capture_output, text=True
        )
        if verbose:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        raise


def _parse_dataset_source(source: str) -> Optional[str]:
    dataset_split = None
    if source.lower() == "dataset":
        dataset_split = "val"
        LOG.info("Using default val dataset")
    elif source.lower().startswith('dataset:'):
        dataset_split = source[8:].lower()
        if not dataset_split:
            raise ValueError("Dataset input requires a set name")
        LOG.info(f"Using dataset {dataset_split}")
    return dataset_split


def _get_real_path_if_path_is_model_name(network_path):
    if not os.path.isfile(network_path) and not network_path.endswith('.yaml'):
        # maybe it is a model name?
        network_yaml_info = yaml_parser.get_network_yaml_info()
        network_path = network_yaml_info.get_info(network_path).yaml_path
    return network_path


class PipeManager:
    """Parse input, output, and model options, and then create a low-level pipeline
    by using PipeInput, PipeOutput, and Pipe and its subclasses. Provide interfaces
    for updating pipeline by different input and output options. Create evaluator if input
    is a dataset.

    Input options:
      source:   str  (usb|csi|file|livestream|dataset) with :<location>
    where device location is from /dev/video, uri or file path

    """

    def __init__(
        self,
        network_path,
        sources,
        pipe_type,
        ax_precompiled_gst='',
        frames: int = 0,
        save_output: str = '',
        num_cal_images: int = 100,
        batch: int = 1,
        data_root: Path | None = None,
        build_root: Path | None = None,
        hardware_caps: config.HardwareCaps = config.HardwareCaps.NONE,
        allow_hardware_codec=True,
        tracers: list[inf_tracers.Tracer] = [],
        server_loader=None,
        metis: config.Metis = config.Metis.none,
        rtsp_latency: int = 500,
        device_selector: str = '',
        specified_frame_rate: int = 0,
    ):
        network_path = _get_real_path_if_path_is_model_name(network_path)
        self.pipe_type = pipe_type
        self._dataset_split = _parse_dataset_source(sources[0])
        self.server_mode = sources[0].lower() == "server"
        if not server_loader and self.server_mode:
            raise ValueError("Server mode requires a loader")
        self.server_loader = server_loader

        self.agg_pads = None
        self.specified_frame_rate = specified_frame_rate
        data_root = data_root or config.default_data_root()
        build_root = build_root or config.default_build_root()
        self.allow_hardware_codec = allow_hardware_codec
        self.rtsp_latency = rtsp_latency
        # TODO: (SDK-782) if without deployment, we don't know how many aipu cores
        # are needed from inference.py; torch pipeline doesn't support aipu_cores=4
        self._device_man = device_manager.create_device_manager(
            pipe_type, metis, device_selector=device_selector
        )
        metis = self._device_man.get_metis_type()

        data_root = Path(data_root).resolve()
        nn = network.parse_network_from_path(network_path, self.eval_mode, data_root)
        if pipe_type in ('torch', 'torch-aipu') or self.eval_mode:
            utils.ensure_dependencies_are_installed(nn.dependencies)
        network.restrict_cores(nn, pipe_type, hardware_caps.aipu_cores, metis)

        nn.model_infos = _get_or_deploy_models(
            network_path,
            nn,
            data_root,
            build_root,
            pipe_type,
            num_cal_images,
            batch,
            hardware_caps,
            metis,
        )

        self.tracers = self._device_man.configure_boards_and_tracers(nn, tracers)

        self.hardware_caps = hardware_caps.detect_caps()
        task_graph = self.build_dependency_graph(nn.tasks)
        self._pipeline = pipeline.get_pipeline(
            self._device_man,
            pipe_type,
            nn,
            build_root,
            self.hardware_caps,
            ax_precompiled_gst,
            task_graph,
        )
        for task in nn.tasks:
            transforms.run_all_transformers(task.preprocess, hardware_caps=self.hardware_caps)
            if hasattr(task.input, 'image_processing'):
                ops = []
                for idx in range(len(sources)):
                    ops.insert(idx, [])
                    for op in task.input.image_processing:
                        match = op.stream_check_match(idx)
                        if match:
                            ops[idx].append(op)
                    if len(ops[idx]) > 1:
                        transforms.run_all_transformers(ops[idx], hardware_caps=self.hardware_caps)
                    for op in ops[idx]:
                        if hasattr(op, 'set_stream_match'):
                            op.set_stream_match(str(idx))

                task.input.image_processing = [item for sublist in ops for item in sublist]
        # propagate model and context info after all transformers have been applied and before invoking get_validation_components, which utilizes the context info.
        self._pipeline.propagate_model_and_context_info()

        LOG.trace("Parse input and output options")
        if self.server_mode:
            self.pipein = io.ServerInput(
                frames,
                hardware_caps,
                server_loader,
            )
            self._evaluator = None
        elif self.eval_mode:
            if task_graph.network_type in {
                graph.NetworkType.SINGLE_MODEL,
                graph.NetworkType.CASCADE_NETWORK,
            }:
                root_task, leaf_task = task_graph.get_root_and_leaf_tasks()
                if task_graph.network_type == graph.NetworkType.CASCADE_NETWORK:
                    LOG.info(
                        f"Cascade network detected, measuring the applicable accuracy of the last task: {leaf_task}"
                    )
                mi = nn.find_model_info_from_task(leaf_task)
                validation_settings = nn.find_task(leaf_task).validation_settings

                # if cascade network, we should build model name as <model1>_<model2> to report in the Evaluator
                if task_graph.network_type == graph.NetworkType.CASCADE_NETWORK:
                    model_name = " -> ".join(
                        [
                            nn.find_model_info_from_task(task).name
                            for task in task_graph.task_map.keys()
                        ]
                    )
                else:
                    model_name = mi.name
            else:
                raise ValueError(
                    f"For accuracy measurement, only single model and cascade networks are supported"
                )

            val_components = io.get_validation_components(
                nn, mi, leaf_task, data_root, self._dataset_split, batch, validation_settings
            )
            self.pipein = io.DatasetInput(
                val_components.dataloader,
                val_components.reformatter,
                task.input.color_format,
                frames,
                hardware_caps,
            )
            # TODO: consider to add dataset and data_root to types.DataLoader,
            # so that evaluator has a chance to access the dataset
            dataset = getattr(val_components.dataloader, 'dataset', None)
            from .. import evaluation

            self._evaluator = evaluation.AxEvaluator(
                model_name,
                nn.find_model(mi.name).dataset,
                mi.task_category,
                leaf_task,
                nn.datasets[nn.find_model(mi.name).dataset],
                dataset,
                master_task=root_task if root_task != leaf_task else None,
                evaluator=val_components.evaluator,
            )
        else:
            if len(sources) > 1:
                self.pipein = io.MultiplexPipeInput(
                    pipe_type,
                    sources,
                    hardware_caps=self.hardware_caps,
                    allow_hardware_codec=self.allow_hardware_codec,
                    color_format=task.input.color_format,
                    rtsp_latency=self.rtsp_latency,
                    specified_frame_rate=self.specified_frame_rate,
                )
            else:
                self.pipein = io.PipeInput(
                    pipe_type,
                    sources[0],
                    hardware_caps=self.hardware_caps,
                    allow_hardware_codec=self.allow_hardware_codec,
                    color_format=task.input.color_format,
                    rtsp_latency=self.rtsp_latency,
                    specified_frame_rate=self.specified_frame_rate,
                )
            self._evaluator = None
        self.sources = {n: src for n, src in enumerate(sources)}
        self.pipeout = io.PipeOutput(save_output=save_output, tracers=tracers)
        self._pipeline.gen_end2end_pipe(self.pipein, self.pipeout)

    def add_source_with_id(self, source, next_id):
        if next_id in self.sources:
            LOG.warning(f"Unable to add source on slot {next_id} already taken")
            return

        if self.agg_pads == None:
            self.agg_pads = self._pipeline.get_agg_pads()
        pipe_newinput = io.PipeInput(
            self.pipe_type,
            source,
            hardware_caps=self.hardware_caps,
            allow_hardware_codec=self.allow_hardware_codec,
        )
        self.agg_pads[next_id] = self._pipeline.gen_newinput_gst(pipe_newinput, next_id)
        self.sources[next_id] = source

    def stream_select(self, streams):
        self._pipeline.stream_select(streams)

    def add_source(self, source):
        if self.agg_pads == None:
            self.agg_pads = self._pipeline.get_agg_pads()
        next_id = max(self.sources.keys()) + 1
        self.add_source_with_id(source, next_id)
        return next_id

    def remove_source(self, idx):
        if self.agg_pads == None:
            self.agg_pads = self._pipeline.get_agg_pads()
        if idx not in self.agg_pads:
            LOG.warning(f"Source slot {idx} not occupied")
            return
        if len(self.agg_pads) == 1:
            LOG.warning(f"Can't remove last stream, stop pipeline insted")
            return
        self._pipeline.remove_source(self.agg_pads[idx])
        del self.agg_pads[idx]
        del self.sources[idx]

    @property
    def pipe(self):
        return self._pipeline

    def __getattr__(self, task_name):
        for t in self._pipeline.nn.tasks:
            if task_name == t.name:
                return t
        raise AttributeError(f"{self.__class__.__name__} object has no attribute '{task_name}'")

    def __dir__(self):
        return sorted(set(super().__dir__() + [t.name for t in self._pipeline.nn.tasks]))

    def init_pipe(self):
        self._pipeline.init()

    def get_pipe(self):
        return self._pipeline.get_pipe()

    def run_pipe(self):
        for tracer in self.tracers:
            tracer.start_monitoring()
        if self.tracers:
            time.sleep(1)  # aipu and temp tracers need some time to start
        self._pipeline.run()

    def stop_pipe(self):
        for tracer in self.tracers:
            tracer.stop_monitoring()
        self._pipeline.stop()
        self._device_man.release()

    def pause_pipe(self):
        self._pipeline.pause()

    def play_pipe(self):
        self._pipeline.play()

    def setup_callback(self, callback: base.ResultCallback):
        self._pipeline.setup_callback(callback)

    @property
    def number_of_frames(self):
        return self.pipein.number_of_frames

    def is_single_image(self):
        return self.pipein.format == 'image'

    @property
    def evaluator(self):
        return self._evaluator

    @property
    def eval_mode(self):
        return bool(self._dataset_split)

    def build_dependency_graph(self, nn_tasks):
        import io

        task_graph = graph.DependencyGraph(nn_tasks)
        output = io.StringIO()
        task_graph.print_graph(output)
        LOG.info(f"Network type: {task_graph.network_type}\n{output.getvalue()}")
        return task_graph

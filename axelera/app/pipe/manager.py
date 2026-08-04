# Copyright Axelera AI, 2023
# functions for deploying pipeline and base object for building pipeline
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import time
from typing import TYPE_CHECKING, Iterable

from axelera import types

from . import base, graph, io
from .. import (
    config,
    device_manager,
    logging_utils,
    network,
    operators,
    pipeline,
    transforms,
    utils,
    yaml_parser,
)
from .frame_data import FrameEvent, FrameEventType

if TYPE_CHECKING:
    from .. import inf_tracers, network


LOG = logging_utils.getLogger(__name__)


def _deploy_one_model_via_subprocess(
    model: str,
    nn_name: str,
    data_root: str,
    build_root: Path,
    pipe_type: str,
    deploy_config: config.DeployConfig,
    aipu_cores: int,
    metis: config.Metis,
):
    num_cal_images = deploy_config.num_cal_images
    default_representative_images = deploy_config.default_representative_images
    cal_seed = deploy_config.cal_seed
    cores_argv = f' --aipu-cores {aipu_cores}'
    cores = ''
    pipe = ''
    mode = ''
    if pipe_type == 'quantized':
        mode = ' --mode QUANTIZE_DEBUG'
    else:
        pipe = f' --pipe {pipe_type}'
        if pipe_type != 'torch':
            s = 's' if aipu_cores > 1 else ''
            cores = f' for {aipu_cores} core{s}. This may take a while...'
    LOG.info(f"Deploying model {model}{cores}")
    run_dir = config.env.framework
    try:
        cal_argv = f'--cal-seed {cal_seed}' if cal_seed else ''
        cal_argv += (
            ' --no-default-representative-images' if not default_representative_images else ''
        )
        dump_core_model_argv = ' --dump-core-model' if deploy_config.dump_core_model else ''
        with utils.spinner():
            run(
                f'{run_dir}/deploy.py --model {model} --num-cal-images {num_cal_images} '
                f'{cores_argv} '
                f'--data-root {data_root}{pipe}{mode} --build-root {build_root} {nn_name} '
                f'--aipu-cores {aipu_cores} --metis {metis.name} {cal_argv} {dump_core_model_argv}',
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
    system_config: config.SystemConfig,
    pipeline_config: config.PipelineConfig,
    deploy_config: config.DeployConfig,
    metis: config.Metis,
) -> network.ModelInfos:
    nn_dir = system_config.build_root / nn.name
    model_infos = network.read_deployed_model_infos(nn_dir, nn, pipeline_config, metis)
    model_infos.add_label_enums(nn.datasets)
    if not model_infos.ready:
        for model in model_infos.missing():
            exec_cores = model_infos.determine_execution_cores(
                model, pipeline_config.aipu_cores, metis
            )
            deploy_cores = model_infos.determine_deploy_cores(
                model, exec_cores, metis, pipeline_config.low_latency
            )
            if deploy_cores < exec_cores:
                _for = f'up to {deploy_cores} cores' if deploy_cores > 1 else 'single-core'
                but = f'(but can be run using {exec_cores} cores)'
                LOG.info(f"{model} is being compiled for {_for} cores {but}.")
            else:
                LOG.debug(f"Model deploy cores is {deploy_cores}")
            _deploy_one_model_via_subprocess(
                model,
                user_specifed_nn_name_or_path,
                system_config.data_root,
                system_config.build_root,
                pipeline_config.pipe_type,
                deploy_config,
                deploy_cores,
                metis,
            )

        model_infos = network.read_deployed_model_infos(nn_dir, nn, pipeline_config, metis)
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
        raise e


def _get_real_path_if_path_is_model_name(network_path):
    if not os.path.isfile(network_path) and not network_path.endswith('.yaml'):
        # maybe it is a model name?
        network_yaml_info = yaml_parser.get_network_yaml_info()
        network_path = network_yaml_info.get_info(network_path).yaml_path
    return network_path


def _assign_image_preproc_ops(
    task: pipeline.AxTask,
    sources: list[config.Source],
    custom_operators: dict[str, type[operators.AxOperator]],
) -> None:
    if preprocs := getattr(task.input, 'image_processing', []):
        for idx, src in enumerate(sources):
            for op in preprocs:
                match = op.stream_check_match(idx)
                if match:
                    src.preprocessing.append(op.as_image_preproc())
    task.image_preproc_ops = {
        n: pipeline.create_transforms_from_config(src.preprocessing, custom_operators)
        for n, src in enumerate(sources)
    }


def compile_pipelines(
    nn: network.AxNetwork,
    srcs: list[config.Source],
    hwcaps: config.HardwareCaps,
) -> None:
    '''Compile the pipeline for the given network and sources.

    Converts source image preprocessing operations to AxOperator instances in the first task,
    and runs all transformers for the image preproc and task preprocs.
    Tiling is applied if any task in the pipeline has Input operator (not InputNoTiles).
    '''
    _assign_image_preproc_ops(nn.tasks[0], srcs, nn.custom_operators)
    if len(nn.tasks) > 1 and any(
        isinstance(t.input, operators.InputWithImageProcessing) for t in nn.tasks[1:]
    ):
        LOG.warning(
            "Multiple tasks with InputWithImageProcessing are not supported, "
            "only the first task will be used."
        )
    from ..operators.custom_preprocessing import ConvertColorInput, _AddTiles

    any_task_needs_tiling = any(isinstance(task.input, operators.Input) for task in nn.tasks)
    for preproc, src in zip(nn.tasks[0].image_preproc_ops.values(), srcs):
        if src.has_image_preproc or not hwcaps.opencl:
            # We only need to convert color if there are image processing ops
            preproc.insert(0, ConvertColorInput(nn.tasks[0].input.color_format))
        if not any_task_needs_tiling:
            # If no task needs tiling, we need to remove any tiling preproc
            preproc[:] = [op for op in preproc if not isinstance(op, _AddTiles)]
    for preproc in nn.tasks[0].image_preproc_ops.values():
        transforms.run_all_transformers(preproc, hardware_caps=hwcaps)
    for task in nn.tasks:
        if isinstance(task.input, operators.InputFromROI):
            # Add image processing to preprocessing ops to enable fusion
            if task.input.image_processing_on_roi:
                task.preprocess[:0] = task.input.image_processing_on_roi
                task.input.image_processing_on_roi = []

        if task.preprocess and hwcaps.opencl:
            task.preprocess.insert(0, ConvertColorInput(task.input.color_format))
        transforms.run_all_transformers(task.preprocess, hardware_caps=hwcaps)


def _update_pending_expansions(task):
    from .. import schema

    for ops in [task.preprocess, task.postprocess]:
        for op in ops:
            for field in op.supported:
                x = getattr(op, field)
                if x == schema.SENTINELS['labels']:
                    setattr(op, field, task.model_info.labels)
                elif x == schema.SENTINELS['label_filter']:
                    setattr(op, field, task.model_info.label_filter)
                elif x == schema.SENTINELS['num_classes']:
                    setattr(op, field, task.model_info.num_classes)


def _create_inference_operators(
    device_man: device_manager.DeviceManager,
    nn: network.AxNetwork,
    low_latency: bool,
    async_mode: bool = False,
):
    def _instantiate_model(model_name: str) -> types.Model:
        with nn.from_model_dir(model_name):
            return nn.instantiate_model(model_name)

    for task in nn.tasks:
        if not task.is_dl_task:
            continue
        model_name = task.model_info.name
        compiled_model_dir = nn.model_infos.manifest_path(model_name).parent
        task.model_info = nn.model_infos.model(model_name)
        model_or_manifest = task.model_info.manifest or _instantiate_model(model_name)
        if len(task.preprocess) == 0:
            # take preprocess from types.Model::override_preprocess
            the_model = (
                _instantiate_model(model_name)
                if task.model_info.manifest is not None
                else model_or_manifest
            )
            nn.attach_model_specific_preprocess(the_model, task)
        _update_pending_expansions(task)
        task.inference = operators.Inference(
            device_man,
            compiled_model_dir,
            model_name,
            model_or_manifest,
            task.model_info,
            task.inference_op_config,
            low_latency,
            async_mode,
        )


def _propagate_model_and_context_info(nn: network.AxNetwork, task_graph: graph.DependencyGraph):
    # Track context per task name
    task_contexts = {}

    for taskn, task in enumerate(nn.tasks):

        # master_task is "where" defined in the Input operator
        if master_task := task_graph.get_master(task.name):
            if master_task in task_contexts:
                task.context.update(task_contexts[master_task])

        def configure_ops(ops, compiled_model_dir):
            for op in ops:
                op.configure_model_and_context_info(
                    task.model_info,
                    task.context,
                    task.name,
                    taskn,
                    compiled_model_dir,
                    task_graph,
                )

        if not task.is_dl_task:
            op_list = [task.input] + task.cv_process
            compiled_model_dir = None
            for image_preproc_ops in task.image_preproc_ops.values():
                configure_ops(image_preproc_ops, compiled_model_dir=compiled_model_dir)
            configure_ops(op_list, compiled_model_dir=compiled_model_dir)

            for op in op_list:
                # pass labels from detections to tracker
                if isinstance(op, operators.Tracker):
                    model_name = nn.find_model_info_from_task(op.bbox_task_name).name
                    op.labels = nn.model_infos.model(model_name).labels
        else:
            compiled_model_dir = nn.model_infos.manifest_path(task.model_info.name).parent

            op_list = [task.input]
            if isinstance(task.input, operators.InputWithImageProcessing):
                op_list += task.input.image_processing
            elif isinstance(task.input, operators.InputFromROI):
                op_list += task.input.image_processing_on_roi
            op_list += task.preprocess + [task.inference] + task.postprocess
            for image_preproc_ops in task.image_preproc_ops.values():
                configure_ops(image_preproc_ops, compiled_model_dir=compiled_model_dir)
            configure_ops(op_list, compiled_model_dir=compiled_model_dir)

        # Store this task's propagated context for its children to use
        task_contexts[task.name] = task.context.propagate()


def _verify_evaluation_model_and_leaf(nn: network.AxNetwork, task_graph: graph.DependencyGraph):
    eval_tracking_task = False
    net_type = task_graph.network_type
    task_names = task_graph.task_names

    root_task, leaf_task = task_graph.get_root_and_leaf_tasks()
    if net_type == graph.NetworkType.CASCADE_NETWORK:
        LOG.info(
            f"Cascade network detected, measuring the applicable accuracy of the last task: {leaf_task}"
        )

    # if cascade network, we should build model name as <model1>_<model2> to report in the Evaluator
    if task_graph.network_type == graph.NetworkType.CASCADE_NETWORK:
        model_name = " -> ".join(nn.find_model_info_from_task(t).name for t in task_names)
    else:
        model_name = nn.find_model_info_from_task(leaf_task).name

    task_categories = {task_graph.get_task(t).model_info.task_category for t in task_names}
    if types.TaskCategory.ObjectTracking in task_categories:
        eval_tracking_task = True
        LOG.info(
            "Tracker task detected for accuracy measurement. To evaluate object detection accuracy"
            " instead, please remove the tracker task from the pipeline"
        )
    return model_name, root_task, leaf_task, eval_tracking_task


def _create_pipein_and_evaluator(
    nn: network.AxNetwork,
    task_graph: graph.DependencyGraph,
    system_config: config.SystemConfig,
    stream_config: config.StreamConfig,
    pipeline_config: config.PipelineConfig,
    src: config.Source,
    id_allocator: base.SourceIdAllocator,
):
    if pipeline_config.tiling:
        raise ValueError("dataset evaluation cannot be performed with tiling enabled")
    ok = (graph.NetworkType.SINGLE_MODEL, graph.NetworkType.CASCADE_NETWORK)
    if task_graph.network_type not in ok:
        raise ValueError(
            f"dataset evaluation can only be performed with single model or cascade networks.\n"
            f"This network is a {task_graph.network_type.name}"
        )

    model_name, root_task, leaf_task, eval_tracking_task = _verify_evaluation_model_and_leaf(
        nn, task_graph
    )

    split = src.location
    mi = nn.find_model_info_from_task(leaf_task)
    validation_settings = nn.find_task(leaf_task).validation_settings
    val_components = io.get_validation_components(
        nn,
        mi,
        leaf_task,
        system_config.data_root,
        split,
        validation_settings,
    )
    pipein = io.DatasetInput(
        src,
        val_components.dataloader,
        val_components.reformatter,
        stream_config.frames,
        system_config.hardware_caps,
        id_allocator.allocate(),
    )
    # TODO: consider to add dataset and data_root to types.DataLoader,
    # so that evaluator has a chance to access the dataset
    dataset = getattr(val_components.dataloader, 'dataset', None)
    from .. import evaluation

    evaluator = evaluation.AxEvaluator(
        model_name,
        nn.find_model(mi.name).dataset,
        mi.task_category,
        leaf_task,
        nn.datasets[nn.find_model(mi.name).dataset],
        dataset,
        master_task=root_task if root_task != leaf_task and not eval_tracking_task else None,
        evaluator=val_components.evaluator,
    )
    return pipein, evaluator


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
        system_config: config.SystemConfig,
        stream_config: config.InferenceStreamConfig,
        pipeline_config: config.PipelineConfig,
        deploy_config: config.DeployConfig,
        tracers: list[inf_tracers.Tracer],
        render_config: config.RenderConfig,
        id_allocator: base.SourceIdAllocator,
    ):
        network_path = _get_real_path_if_path_is_model_name(pipeline_config.network)
        self._sources = sources = pipeline_config.sources
        self._parent_callback = None
        self._device_man = device_manager.create_device_manager(
            pipeline_config.pipe_type,
            system_config.metis,
            device_selector=pipeline_config.device_selector,
        )
        metis = self._device_man.get_metis_type()

        self._network = nn = network.parse_network_from_path(
            network_path, deploy_config, pipeline_config.eval_mode, system_config.data_root
        )
        if pipeline_config.pipe_type in ('torch', 'torch-aipu') or pipeline_config.eval_mode:
            utils.ensure_dependencies_are_installed(nn.dependencies)
        network.restrict_cores(nn, pipeline_config, metis)

        nn.model_infos = _get_or_deploy_models(
            network_path,
            nn,
            system_config,
            pipeline_config,
            deploy_config,
            metis,
        )
        self.tracers = self._device_man.configure_boards_and_tracers(nn, tracers)
        self.hardware_caps = system_config.hardware_caps
        task_graph = self.build_dependency_graph(nn.tasks)
        logging_dir = pipeline.download_nn_assets(system_config, nn)

        # this is kind of the core of the pipeline builder. But note it is still dependent on the device manager
        compile_pipelines(nn, sources, self.hardware_caps)
        _create_inference_operators(
            self._device_man,
            nn,
            low_latency=pipeline_config.low_latency,
            async_mode=pipeline_config.async_mode,
        )
        nn.model_infos.add_label_enums(nn.datasets)
        _propagate_model_and_context_info(nn, task_graph)

        if pipeline_config.eval_mode:
            self._pipein, self._evaluator = _create_pipein_and_evaluator(
                nn,
                task_graph,
                system_config,
                stream_config,
                pipeline_config,
                self._sources[0],
                id_allocator,
            )
        else:
            self._pipein = io.MultiplexPipeInput(
                pipeline_config.sources,
                system_config,
                pipeline_config,
                id_allocator,
            )
            self._evaluator = None
        self.sources = {sid: p.sources[0] for sid, p in self._pipein.inputs.items()}
        rc = _set_render_config(nn, render_config)
        self._pipeout = io.PipeOutput(pipeline_config.save_output, self._pipein, rc)
        self._pipeline = base.create_pipe(
            self._device_man,
            pipeline_config,
            nn,
            logging_dir,
            system_config.hardware_caps,
            task_graph,
            self._event_callback,
            self._pipein,
        )
        self._id_allocator = id_allocator

    @property
    def name(self):
        '''Return the name of the network associated with this PipeManager.'''
        return self._network.name

    @property
    def network(self) -> network.AxNetwork:
        '''Return the network associated with this PipeManager.'''
        return self._network

    def add_source(self, source: config.Source, source_id: int = -1):
        if source_id == -1:
            source_id = self._id_allocator.allocate()
        if source_id in self.sources:
            LOG.warning(f"Unable to add source on slot {source_id} already taken")
            return

        pipe_newinput = self._pipein.add_source(source, source_id)
        self._pipeline.add_source(pipe_newinput)
        self.sources[source_id] = source
        return source_id

    def remove_source(self, source_id):
        self._pipein.remove_source(self.sources[source_id])
        self._pipeline.remove_source(source_id)
        self._id_allocator.deallocate(source_id)
        del self.sources[source_id]

    def stream_select(self, streams: Iterable[int] | str) -> None:
        self._pipeline.stream_select(streams)

    def get_stream_select(self) -> list[int]:
        return self._pipeline.get_stream_select()

    @property
    def pipe(self):
        return self._pipeline

    @property
    def pipeout(self):
        return self._pipeout

    def set_render(self, show_labels=False, show_annotations=False):
        """Set render mode for the pipeline for all tasks."""
        for task in self._pipeline.nn.tasks:
            self._pipeout.set_task_render(task.name, show_annotations, show_labels)
        return self

    def __getattr__(self, task_name):
        for t in self._pipeline.nn.tasks:
            if task_name == t.name:
                return TaskProxy(self, task_name)
        raise AttributeError(f"{self.__class__.__name__} object has no attribute '{task_name}'")

    def __dir__(self):
        return sorted(set(super().__dir__() + [t.name for t in self._pipeline.nn.tasks]))

    def init_pipe(self):
        self._pipeline.init()

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

    def _event_callback(self, event: FrameEvent) -> bool:
        """Callback for the pipe output to handle events/results."""
        if event.result is not None:
            result = event.result
            result.sink_timestamp = time.time()
            if result.meta:
                result.meta.set_render_config(self._pipeout.get_render_config())
            if result.meta and self.tracers:
                for tracer in self.tracers:
                    tracer.update(result)
                if result.meta and result.stream_id == 0:
                    for tracer in self.tracers:
                        for m in tracer.get_metrics():
                            result.meta.add_instance(m.key, m)
            self._pipeout.sink(result)
        elif event.type == FrameEventType.end_of_pipeline:
            self._pipeout.close_writer()
        if self._parent_callback:
            return self._parent_callback(event)
        return True

    def setup_callback(self, callback: base.ResultCallback):
        self._parent_callback = callback

    @property
    def number_of_frames(self):
        return self._pipein.number_of_frames

    def is_single_image(self):
        return (
            len(self._sources) == 1
            and self._sources[0].type == config.SourceType.IMAGE_FILES
            and len(self._sources[0].images) == 1
        )

    @property
    def evaluator(self):
        return self._evaluator

    @property
    def eval_mode(self):
        return bool(self._sources[0].type == config.SourceType.DATASET)

    def build_dependency_graph(self, nn_tasks):
        task_graph = graph.DependencyGraph(nn_tasks)
        if LOG.isEnabledFor(logging_utils.DEBUG):
            task_graph.print_all_views(LOG.debug)
            LOG.debug(f"Network type: {task_graph.network_type}")
        return task_graph


def _set_render_config(
    nn, render_config_from_api: config.RenderConfig | None
) -> config.RenderConfig:
    """Set render configuration for the pipeline based on the tasks in the neural network."""
    task_names = {task.name for task in nn.tasks}

    if render_config_from_api is not None:
        config_keys = set(render_config_from_api.keys())
        extra_keys = config_keys - task_names
        if extra_keys:
            raise ValueError(f"render_config_from_api contains keys not in nn.tasks: {extra_keys}")

    final_config = config.RenderConfig()
    api_config_keys = (
        set(render_config_from_api.keys()) if render_config_from_api is not None else set()
    )
    for task in nn.tasks:
        if render_config_from_api is not None and task.name in api_config_keys:
            # Priority 1: API config overrides everything
            task_render_config = render_config_from_api[task.name]
        elif task.task_render_config is not None:
            # Priority 2: Task's own task_render_config
            task_render_config = task.task_render_config
        else:
            # Priority 3: Default TaskRenderConfig
            task_render_config = config.TaskRenderConfig()

        final_config.set_task(
            task.name,
            show_annotations=task_render_config.show_annotations,
            show_labels=task_render_config.show_labels,
            force_register=True,
        )
    return final_config


class TaskProxy:
    """Proxy class for tasks to allow setting render options per task while preserving access to the original task attributes."""

    def __init__(self, pipe_manager, task_name):
        self._pipe_manager = pipe_manager
        self.task_name = task_name

        for t in pipe_manager._pipeline.nn.tasks:
            if task_name == t.name:
                self._original_task = t
                break

    def set_render(self, show_annotations=True, show_labels=True):
        """Set render options for this specific task.

        Args:
            show_annotations: Whether to draw visual elements like bounding boxes
            show_labels: Whether to draw class labels and score text

        Returns:
            Self for method chaining
        """
        self._pipe_manager.pipeout.set_task_render(self.task_name, show_annotations, show_labels)
        return self

    def __getattr__(self, name):
        """Forward attribute access to the original task object.

        This allows accessing attributes like 'classes' directly from the proxy.
        """
        # Delegate to the original task for any attributes not found on the proxy
        return getattr(self._original_task, name)

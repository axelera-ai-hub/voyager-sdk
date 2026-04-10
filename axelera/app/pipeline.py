# Copyright Axelera AI, 2023
# Translate YAML pipelines to code
from __future__ import annotations

import builtins
import collections
import dataclasses
import enum
import io
import itertools
import logging
import os
from pathlib import Path
import re
import subprocess
import sys
from typing import Callable, Dict, List, Optional, Tuple
from unittest.mock import patch

from axelera import types

from . import (
    compile,
    config,
    constants,
    data_utils,
    device_manager,
    exceptions,
    logging_utils,
    network,
    operators,
    schema,
    utils,
)
from .operators import AxOperator, compose_preprocess_transforms, get_input_operator

LOG = logging_utils.getLogger(__name__)
COMPILER_LOG = logging_utils.getLogger('compiler')


TASK_PROPERTIES = {
    'input',
    'preprocess',
    'inference',
    'postprocess',
    'template_path',
    'operators',
    'aipu_cores',
    'meta_key',
    'model_name',
}


@dataclasses.dataclass
class AxTask:
    name: str
    input: AxOperator
    preprocess: List[AxOperator] = dataclasses.field(default_factory=list)
    image_preproc_ops: dict[int, list[AxOperator]] = dataclasses.field(default_factory=dict)
    model_info: types.ModelInfo = None
    context: operators.PipelineContext = None
    inference_op_config: operators.InferenceOpConfig = None
    inference: Optional[operators.Inference] = None
    postprocess: List[AxOperator] = dataclasses.field(default_factory=list)
    aipu_cores: Optional[int] = None
    validation_settings: dict = dataclasses.field(default_factory=dict)
    data_adapter: types.DataAdapter = None
    # cv_process will be assigned for classical CV operators only
    cv_process: AxOperator = None
    # default render config from YAML; can be overridden by APIs
    task_render_config: Optional[config.RenderConfig] = None

    def __repr__(self):
        return f"""AxTask('{self.name}',
    input={self.input},
    preprocess={self.preprocess},
    inference_op_config={self.inference_op_config},
    inference={self.inference},
    postprocess={self.postprocess},
    model_info={self.model_info},
    context={self.context},
    aipu_cores={self.aipu_cores},
    validation_settings={self.validation_settings},
    data_adapter={self.data_adapter},
    cv_process={self.cv_process},
    task_render_config={self.task_render_config},
)"""

    @property
    def classes(self):
        if hasattr(self.model_info, 'labels') and isinstance(
            self.model_info.labels, utils.FrozenIntEnumMeta
        ):
            return self.model_info.labels
        raise NotImplementedError("Class enum is only available with enumerated labels")

    @property
    def is_dl_task(self):
        return self.model_info.model_type == types.ModelType.DEEP_LEARNING


def _parse_task_render_config(phases: dict) -> Optional[config.TaskRenderConfig]:
    task_render_config = None
    if task_render_config_dict := phases.pop('render', {}):
        task_render_config = config.TaskRenderConfig.from_dict(task_render_config_dict)
    return task_render_config


def _parse_deep_learning_task(
    phases: dict,
    task_name: str,
    model_infos: network.ModelInfos,
    custom_operators: Dict[str, AxOperator],
    eval_mode: bool = False,
) -> AxTask:
    model_name = phases.get('model_name', task_name)
    model_info = model_infos.model(model_name)
    phases = utils.substitute_vars_and_expr(phases, model_info_as_kwargs(model_info), model_name)
    task_render_config = _parse_task_render_config(phases)
    template = _get_template_processing_steps(phases, model_info)
    for d in [template, phases]:
        d.setdefault('input', {})
        d.setdefault('preprocess', [])
        d.setdefault('inference', {})
        d.setdefault('postprocess', [])
        if unknown := [k for k in d.keys() if k not in TASK_PROPERTIES]:
            msg = 'is not a valid property' if len(unknown) == 1 else 'are not valid properties'
            raise ValueError(f"{', '.join(unknown)} {msg} of a Task")

    inp = _gen_input_transforms(phases['input'], template['input'], custom_operators)
    inf_op_config = operators.InferenceOpConfig.from_yaml_dict(
        phases['inference'], template['inference']
    )
    preprocess = _gen_processing_transforms(
        phases, template, custom_operators, 'preprocess', task_name, eval_mode
    )
    postprocess = _gen_processing_transforms(
        phases, template, custom_operators, 'postprocess', task_name, eval_mode
    )
    task = AxTask(
        task_name,
        input=inp,
        preprocess=preprocess,
        model_info=model_info,
        context=operators.PipelineContext(),
        inference_op_config=inf_op_config,
        inference=None,
        postprocess=postprocess,
        aipu_cores=model_info.extra_kwargs.get('aipu_cores'),
        task_render_config=task_render_config,
    )

    validation_settings = {}
    if task.postprocess:
        for op in task.postprocess:
            if any(key != 'pair_validation' for key in op.validation_settings):
                if common_keys := set(
                    key for key in op.validation_settings.keys() if key != 'pair_validation'
                ) & set(validation_settings.keys()):
                    raise ValueError(
                        f"Operator {op.name} has validation settings {common_keys} that are already registered."
                    )
                validation_settings.update(op.validation_settings)
    task.validation_settings = validation_settings
    return task


def _parse_classical_cv_task(
    phases: dict,
    task_name: str,
    model_infos: network.ModelInfos,
    custom_operators: Dict[str, AxOperator],
    eval_mode: bool = False,
) -> AxTask:
    model_name = phases.get('model_name', task_name)
    model_info = model_infos.model(model_name)

    input = _gen_input_transforms(phases['input'], None, custom_operators)
    cv_transforms = _get_dict_of_operator_list(phases['cv_process'])
    all_ops = {
        k: v for k, v in custom_operators.items() if isinstance(v, operators.BaseClassicalCV)
    }
    all_ops.update(operators.builtins_classical_cv)

    cv_process = []
    for el in phases['cv_process']:
        opname, attribs = _get_op(el)
        try:
            operator = all_ops[opname]
        except KeyError:
            raise ValueError(f"{opname}: Not a valid classical CV operator")
        attribs = dict(attribs or {}, **(cv_transforms.get(opname) or {}))
        cv_process.append(operator(**attribs))
    task_render_config = _parse_task_render_config(phases)
    return AxTask(
        task_name,
        input=input,
        cv_process=cv_process,
        context=operators.PipelineContext(),
        model_info=model_info,
        task_render_config=task_render_config,
    )


def parse_task(
    model: dict,
    custom_operators: Dict[str, AxOperator],
    model_infos: network.ModelInfos,
    eval_mode: bool = False,
) -> AxTask:
    """Parse YAML pipeline to AxTask object.

    model: a dict containing exactly one item: {task_name: processing_steps}
      Where processing_steps is a dict containing one or more of:
         model_name: the name of the model
         input: the input transform
         preprocess: the preprocessing transform
         postprocess: the postprocessing transform
         template_path: path to the template file

    """
    _check_type(model, 'Task properties', dict)
    assert len(model) == 1
    task_name, phases = _get_op(model)
    if phases is None:
        raise ValueError(f"No pipeline config for {task_name}")
    _check_type(phases, 'Task properties', dict)
    if not phases:
        raise ValueError(f"No pipeline config for {task_name}")
    task_model_info = model_infos.model(model[task_name].get('model_name', task_name))
    if task_model_info.model_type == types.ModelType.DEEP_LEARNING:
        return _parse_deep_learning_task(
            phases, task_name, model_infos, custom_operators, eval_mode
        )
    elif task_model_info.model_type == types.ModelType.CLASSICAL_CV:
        return _parse_classical_cv_task(
            phases, task_name, model_infos, custom_operators, eval_mode
        )
    else:
        raise ValueError(f"Invalid model type: {task_model_info.model_type}")


def _check_type(element, element_name, required_type):
    if not isinstance(element, required_type):
        # hide the yaml types from the error message
        actual = 'None' if element is None else type(element).__name__
        for maybe in [str, dict, list, int, bool]:
            if isinstance(element, maybe):
                actual = maybe.__name__
                break
        if isinstance(required_type, tuple):
            required_type = '|'.join(
                'None' if t is type(None) else t.__name__ for t in required_type
            )
        else:
            required_type = required_type.__name__
        raise ValueError(f"{element_name} must be {required_type} (found type {actual})")


def _get_op(el: Dict[str, dict]) -> Tuple[str, dict]:
    return next(iter(el.items()))


def _get_dict_of_operator_list(list_of_dicts):
    """convert list of dict to a dict"""
    return {k: v for d in list_of_dicts for k, v in d.items()} if list_of_dicts else {}


def _normalize_operator_name(name: str) -> str:
    return name.replace('-', '').replace('_', '').lower()


def _convert_yaml_operator_name(operation_steps: list, target: str) -> None:
    '''Remove dash and underscore from input operator names in-place'''
    if not operation_steps:
        return []
    for operation in operation_steps:
        key, value = operation.popitem()  # should have one element only
        operation[_normalize_operator_name(key)] = value


def _gen_processing_transforms(
    processing_steps, template, custom_operators, target, model_name, eval_mode: bool = False
):
    assert target in ['preprocess', 'postprocess']
    _convert_yaml_operator_name(processing_steps[target], target)
    _convert_yaml_operator_name(template[target], target)
    return _gen_process_list(
        processing_steps[target], custom_operators, template[target], model_name, target, eval_mode
    )


def model_info_as_kwargs(model_info: types.ModelInfo):
    # note that using dataclasses.asdict causes massive recursion due to all
    # the `.parent` attributes in yaml derived values, these are then deep
    # copied and causes issues with MockOpen.
    vals = {f.name: getattr(model_info, f.name) for f in dataclasses.fields(types.ModelInfo)}
    vals = {k: v.name if isinstance(v, enum.Enum) else v for k, v in vals.items()}

    if model_info.model_type == types.ModelType.DEEP_LEARNING:
        return dict(
            vals,
            input_width=model_info.input_width,
            input_height=model_info.input_height,
            input_channel=model_info.input_channel,
            **model_info.extra_kwargs,
        )
    else:
        return dict(vals, **model_info.extra_kwargs)


def _get_template_processing_steps(processing_steps, model_info):
    template = {'input': None, 'preprocess': None, 'postprocess': None}
    if template_path := (processing_steps and processing_steps.get('template_path')):
        template_path = config.env.expandvars(template_path)
        if not Path(template_path).is_file():
            raise FileNotFoundError(
                f"Template file not found: {template_path}\n"
                f"Check that the template_path in your model card points to a valid file."
            )

        refs = model_info_as_kwargs(model_info)
        compiled_schema = schema.load_task(template_path, False)
        template.update(utils.load_yaml_by_reference(template_path, refs, compiled_schema))
    return template


def _batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def _trace_model_info(model_info: types.ModelInfo, out: Callable[[str], None]):
    def col(left, right):
        return out(f'{left:>20} {right}'.rstrip())

    col('Field', 'Value')
    for name, val in model_info_as_kwargs(model_info).items():
        if val and isinstance(val, list):
            for i, items in enumerate(_batched(val, 5)):
                col(name if i == 0 else '', ', '.join(str(x) for x in items))
        else:
            col(name, val)


def _check_calibration_data_loader(model: types.Model, data_loader: types.DataLoader):
    is_suitable = model.check_calibration_data_loader(data_loader)
    if is_suitable is None:
        if data_loader and (_sampler := getattr(data_loader, 'sampler', None)):
            from torch.utils.data import sampler

            is_suitable = isinstance(_sampler, sampler.RandomSampler)

    if is_suitable is False:
        LOG.warning(
            "The calibration dataloader does not appear to have a suitably representative dataset.\n"
            "This may give poor calibration results."
        )
    elif is_suitable is None:
        LOG.warning(
            "Unable to determine if the calibration dataloader/sampler has a suitably representative dataset.\n"
            "This may give poor calibration results.\n"
            "(You can check/confirm the dataloader by implementing `check_calibration_dataloader` model methods.)"
        )


def _register_model_output_info(model_obj: types.Model, model_info: types.ModelInfo):
    if isinstance(model_obj, types.ONNXModel):
        model_info.register_outputs_from_onnx(model_info.weight_path)
    else:
        model_info.register_outputs_from_pytorch(model_obj)


builtin_print = builtins.print
matcher = re.compile(r'(error|warning|info|debug|trace|fatal|critical|exception)', re.IGNORECASE)


def print_as_logger(*args, **kwargs):
    f = kwargs['file'] = io.StringIO()
    builtin_print(*args, **kwargs)
    s = f.getvalue()
    m = matcher.search(s)
    level = logging.INFO
    if m:
        level = getattr(logging, m.group(1).upper(), logging.INFO)
    COMPILER_LOG.log(level, s)


def _deploy_model(
    task: AxTask,
    model_name,
    nn,
    compilation_cfg,
    num_cal_images,
    model_dir,
    is_export: bool,
    compile_object: bool,
    deploy_mode: config.DeployMode,
    metis: config.Metis,
    model_infos: network.ModelInfos,
    cal_seed: int | None,
    is_default_representative_images: bool = True,
    dump_core_model: bool = False,
    toml_metadata: dict = None,
):
    # Compile specified model
    batch = 1
    try:
        dataset_cfg = nn.model_dataset_from_task(task.name) or {}

        if model_infos.model(model_name).model_type == types.ModelType.DEEP_LEARNING:
            with nn.from_model_dir(model_name):
                model_obj = nn.instantiate_model_for_deployment(task)
                LOG.debug("Compose dataset calibration transforms")
                preprocess = compose_preprocess_transforms(task.preprocess, task.input)

                if compile_object:
                    data_root_path = Path(dataset_cfg['data_dir_path'])

                    generator = None
                    if cal_seed is not None:
                        import torch

                        generator = torch.Generator()
                        generator.manual_seed(cal_seed)

                    if "repr_imgs_dir_path" in dataset_cfg:
                        value = dataset_cfg["repr_imgs_dir_path"]
                        if not isinstance(value, Path):
                            dataset_cfg["repr_imgs_dir_path"] = Path(value)
                        calibration_data_loader = model_obj.check_representative_images(
                            preprocess,
                            batch,
                            generator=generator,
                            **dataset_cfg,
                        )
                    else:  # this need to change to when users saying --cal_seed == None
                        calibration_data_loader = model_obj.create_calibration_data_loader(
                            preprocess,
                            data_root_path,
                            batch,
                            generator=generator,
                            **dataset_cfg,
                        )

                    _check_calibration_data_loader(model_obj, calibration_data_loader)
                    num_cal_images_from_dataset = len(calibration_data_loader) * batch
                    if num_cal_images > num_cal_images_from_dataset:
                        raise ValueError(
                            f"Cannot use {num_cal_images} calibration images when dataset only contains "
                            f"{num_cal_images_from_dataset} images. Please either:\n"
                            f"  1. Reduce --num-cal-images to {num_cal_images_from_dataset} or less\n"
                            f"  2. Add more images to the calibration dataset"
                        )
                    if is_default_representative_images:
                        reformat_for_calibration = lambda x: x
                    else:
                        reformat_for_calibration = model_obj.reformat_for_calibration

                    batch_loader = data_utils.NormalizedDataLoaderImpl(
                        calibration_data_loader,
                        reformat_for_calibration,
                        is_calibration=True,
                        num_batches=(num_cal_images + batch - 1) // batch,
                    )
                    model_obj.set_calibration_normalized_loader(batch_loader)
                    if deploy_mode in {
                        config.DeployMode.QUANTIZE,
                        config.DeployMode.QUANTIZE_DEBUG,
                    }:
                        decoration_flags = ''
                    else:
                        ncores = compilation_cfg.aipu_cores_used
                        decoration_flags = model_infos.determine_deploy_decoration(
                            model_name, ncores, metis
                        )
                        _register_model_output_info(model_obj, task.model_info)

                    with patch.object(builtins, 'print', print_as_logger):
                        compile.compile(
                            model_obj,
                            task.model_info,
                            compilation_cfg,
                            model_dir,
                            is_export,
                            deploy_mode,
                            metis,
                            decoration_flags,
                            dump_core_model,
                            toml_metadata,
                        )

        _trace_model_info(task.model_info, LOG.trace)
        out_json = model_dir / constants.K_MODEL_INFO_FILE_NAME
        LOG.trace(f'Write {task.model_info.name} model info to {out_json} file')
        out_json.parents[0].mkdir(parents=True, exist_ok=True)
        axelera_framework = config.env.framework
        if task.model_info.class_path:
            task.model_info.class_path = os.path.relpath(
                task.model_info.class_path, axelera_framework
            )
        out_json.write_text(task.model_info.to_json())
    except logging_utils.UserError:
        raise
    except exceptions.PrequantizedModelRequired:
        # pass the exception out to trigger prequantization
        raise
    except Exception as e:
        LOG.error(e)
        LOG.trace_exception()
        return False
    return True


def deploy_from_yaml(
    nn_name: str,
    pipeline_only,
    models_only,
    model,
    system_config: config.SystemConfig,
    pipeline_config: config.PipelineConfig,
    deploy_config: config.DeployConfig,
    deploy_mode: config.DeployMode,
    is_export,
    metis: config.Metis,
):
    with device_manager.create_device_manager(pipeline_config.pipe_type, metis, deploy_mode) as dm:
        return _deploy_from_yaml(
            dm,
            nn_name,
            pipeline_only,
            models_only,
            model,
            system_config,
            pipeline_config,
            deploy_config,
            deploy_mode,
            is_export,
            metis,
        )


def run(cmd, shell=True, check=True, verbose=False, capture_output=True):
    if verbose:
        print(cmd)
        capture_output = False  # for debugging
    try:
        result = subprocess.run(
            cmd,
            shell=shell,
            check=check,
            capture_output=capture_output,
            text=True,
        )
        if verbose and not capture_output:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print(result.stderr)
        return result
    except subprocess.CalledProcessError as e:
        print(f"Command failed: {e.cmd}")
        print(f"Return code: {e.returncode}")
        if e.stdout:
            print(f"STDOUT: {e.stdout}")
        if e.stderr:
            print(f"STDERR: {e.stderr}")
        raise


def _build_deploy_command_and_run(
    nn_name: str,
    model_name: str,
    num_cal_images: int,
    calibration_batch: int,
    data_root: str,
    pipe_type: str,
    build_root,
    metis: config.Metis,
    cal_seed: int | None = None,
    hardware_caps: config.HardwareCaps | None = None,
    aipu_cores: int | None = None,
    mode: str | None = None,
    debug: bool = False,
    trace: str = '',
    default_representative_images: bool = True,
    capture_output: bool = True,
    verbose_run: bool = False,
    use_spinner: bool = False,
) -> None:
    """
    Build and execute a deploy.py command with all the consolidated logic.

    This function consolidates all deploy.py subprocess calling patterns across
    the codebase, including run_dir handling, command construction, and execution.
    """
    run_dir = config.env.framework

    # Build base command
    cmd_parts = [
        f'{run_dir}/deploy.py',
        f'--model {model_name}',
        f'--num-cal-images {num_cal_images}',
        f'--calibration-batch {calibration_batch}',
    ]

    # Add hardware capabilities if provided
    if hardware_caps:
        cap_argv = hardware_caps.as_argv()
        if cap_argv:
            cmd_parts.append(cap_argv)

    # Add default representative images flag
    if default_representative_images:
        cmd_parts.append('--default-representative-images')

    if pipe_type == 'quantized':
        mode = 'QUANTIZE'
        debug = True
    else:
        cmd_parts.append(f'--pipe {pipe_type}')

    # Add other standard parameters
    cmd_parts.extend(
        [
            f'--data-root {data_root}',
            f'--build-root {build_root}',
            nn_name,
        ]
    )

    # Add AIPU cores if specified
    if aipu_cores is not None:
        cmd_parts.append(f'--aipu-cores {aipu_cores}')

    # Add metis configuration
    cmd_parts.append(f'--metis {metis.name}')

    # Add mode if specified (for quantization)
    if mode:
        cmd_parts.append(f'--mode {mode}{"_DEBUG" if debug else ""}')

    # Add trace flag if provided
    if trace:
        cmd_parts.append(trace)

    # Add calibration seed if provided
    if cal_seed is not None:
        cmd_parts.append(f'--cal-seed {cal_seed}')

    cmd = ' '.join(cmd_parts)

    # Execute the command with appropriate context
    def _execute():
        run(cmd, capture_output=capture_output, verbose=verbose_run)

    if use_spinner:
        with utils.spinner():
            _execute()
    else:
        _execute()


def _quantize_single_model(
    nn_name: str,
    model_name: str,
    system_config: config.SystemConfig,
    deploy_config: config.DeployConfig,
    pipe_type: str,
    metis: config.Metis,
    debug: bool = False,
):
    """quantize a model in a separate process because of quantizer OOM"""
    deploy_info = f'{nn_name}: {model_name}'
    LOG.info(f"Prequantizing {deploy_info}")
    run_dir = config.env.framework
    try:
        trace = '-v ' if LOG.isEnabledFor(logging_utils.TRACE) else ''
        # default_representative_images
        cal_argv = f'--cal-seed {deploy_config.cal_seed}' if deploy_config.cal_seed else ''
        cal_argv += (
            ' --no-default-representative-images'
            if not deploy_config.default_representative_images
            else ''
        )
        dump_core_model_argv = ' --dump-core-model' if deploy_config.dump_core_model else ''
        run(
            f'{run_dir}/deploy.py --num-cal-images {deploy_config.num_cal_images} --model {model_name} '
            f'{system_config.hardware_caps.as_argv()} '
            # Q? do we need hw_caps here? and do we need to also pass aipu_cores for quant?
            f'--data-root {system_config.data_root} --pipe {pipe_type} --build-root {system_config.build_root} {nn_name} '
            f'--mode QUANTIZE{"_DEBUG" if debug else ""} {trace}'
            f'--metis {metis.name} {cal_argv} {dump_core_model_argv}',
            capture_output=False,
            verbose=LOG.isEnabledFor(logging.DEBUG),
        )
        LOG.info(f"Successfully prequantized {deploy_info}")
    except subprocess.CalledProcessError as e:
        if e.stdout:
            print(e.stdout)
        if e.stderr:
            print(e.stderr)
        LOG.error(f"Failed to prequantize {deploy_info}")
        sys.exit(1)


def _deploy_from_yaml(
    device_man: device_manager.DeviceManager,
    nn_name: str,
    pipeline_only,
    models_only,
    model,
    system_config: config.SystemConfig,
    pipeline_config: config.PipelineConfig,
    deploy_config: config.DeployConfig,
    deploy_mode: config.DeployMode,
    is_export,
    metis: config.Metis,
):
    path = pipeline_config.network
    ok = True
    compile_obj = pipeline_config.pipe_type in ['gst', 'torch-aipu']

    nn = network.parse_network_from_path(
        pipeline_config.network,
        deploy_config,
        data_root=system_config.data_root,
        from_deploy=True,
    )
    utils.ensure_dependencies_are_installed(nn.dependencies)
    nnname = f"network {nn.name}"
    nn_dir = system_config.build_root / nn.name

    if compile_obj and metis == config.Metis.none:
        metis = device_man.get_metis_type()
        LOG.info("Detected Metis type as %s", metis.name)

    network.restrict_cores(nn, pipeline_config, metis, deploy=True)

    model_infos = network.read_deployed_model_infos(nn_dir, nn, pipeline_config, metis)

    if len(nn.tasks) < 1:
        raise ValueError(f"No tasks found in {path}")
    elif nn.tasks[0].preprocess is None:
        # only if pure model deployment and not through our YAML pipeline
        pipeline_only = False
        models_only = True

    if not pipeline_only:
        verb = (
            'Quantizing'
            if deploy_mode in {config.DeployMode.QUANTIZE, config.DeployMode.QUANTIZE_DEBUG}
            else 'Compiling'
        )
        LOG.info(f"## {verb} {nnname} **{path}**{f' {model}' if model else ''}")
        found = False
        for task in nn.tasks:
            model_name = task.model_info.name
            if model:
                if model_name != model:
                    continue
                else:
                    found = True

            compiler_overrides = model_infos.model_compiler_overrides(model_name, metis)
            deploy_cores = model_infos.determine_deploy_cores(
                model_name, pipeline_config.aipu_cores, metis, pipeline_config.low_latency
            )
            try:
                # Extract yaml_dir from compiler_overrides (added by ModelInfos)
                yaml_dir = compiler_overrides.pop('_yaml_dir', None)
                compilation_cfg, toml_metadata = config.gen_compilation_config(
                    deploy_cores,
                    compiler_overrides,
                    deploy_mode,
                    yaml_dir=yaml_dir,
                )
            except (ImportError, ModuleNotFoundError, OSError):
                if compile_obj:
                    raise
                else:
                    LOG.warning(
                        "Failed to import axelera.compiler, trying empty "
                        "compilation config, as we are not compiling"
                    )
                    compilation_cfg = dict()
                    toml_metadata = None

            LOG.info(f"Compile model: {model_name}")
            model_dir = nn_dir / model_name
            retried_quantize = False

            if (
                deploy_mode in {config.DeployMode.QUANTIZE, config.DeployMode.QUANTIZE_DEBUG}
                and len(nn.tasks) > 1
                and not model
            ):
                # if a multi-model network, we need to quantize each model in a separate process
                _quantize_single_model(
                    nn_name,
                    model_name,
                    system_config,
                    deploy_config,
                    pipeline_config.pipe_type,
                    metis,
                    debug=(deploy_mode == config.DeployMode.QUANTIZE_DEBUG),
                )
                ok = True
            else:
                this_deploy_mode = deploy_mode
                while True:
                    try:
                        if this_deploy_mode == config.DeployMode.QUANTCOMPILE:
                            # force prequantization in a separate process
                            this_deploy_mode = config.DeployMode.PREQUANTIZED
                            raise exceptions.PrequantizedModelRequired(
                                model_name, model_dir.joinpath(constants.K_MODEL_QUANTIZED_DIR)
                            )

                        ok = ok and _deploy_model(
                            task,
                            model_name,
                            nn,
                            compilation_cfg,
                            deploy_config.num_cal_images,
                            model_dir,
                            is_export,
                            compile_obj,
                            this_deploy_mode,
                            metis,
                            model_infos,
                            deploy_config.cal_seed,
                            deploy_config.default_representative_images,
                            deploy_config.dump_core_model,
                            toml_metadata,
                        )
                    except exceptions.PrequantizedModelRequired as e:
                        if not retried_quantize:
                            retried_quantize = True
                            _quantize_single_model(
                                nn_name,
                                e.model_name,
                                system_config,
                                deploy_config,
                                pipeline_config.pipe_type,
                                metis,
                                deploy_config.cal_seed,
                            )
                            continue
                        ok = False
                    break

            if model and ok:
                # Deploy a single model
                LOG.info(f"## Finished {verb.lower()} {nnname}: model '{model_name}'")
                return ok

        if model and not found:
            LOG.info(f"## Deploy {nnname}: model '{model}' not found in {Path(path).name}")
            return False
    else:
        LOG.info(f"## Deploy {nnname} pipeline only")

    # redetect ready state, or else `check_ready` fails in _deploy_pipeline
    model_infos = network.read_deployed_model_infos(nn_dir, nn, pipeline_config, metis)
    if models_only or model:
        return ok
    elif ok:
        LOG.info(f"Compile {Path(path).name}:pipeline")
        return _deploy_pipeline(nn, system_config, model_infos)
    else:
        return False


def _deploy_pipeline(
    nn: network.AxNetwork,
    system_config: config.SystemConfig,
    model_infos: network.ModelInfos,
):
    try:
        model_infos.check_ready()
        nn.model_infos = model_infos
        download_nn_assets(system_config, nn)

    except Exception as e:
        LOG.error(e)
        LOG.trace_exception()
        return False
    return True


def download_nn_assets(system_config: config.SystemConfig, nn: network.AxNetwork):
    logging_dir = system_config.build_root / nn.name / 'logs'
    logging_dir.mkdir(parents=True, exist_ok=True)
    for asset in nn.assets:
        utils.download_and_extract_asset(asset.url, Path(asset.path), asset.md5)
    return logging_dir


def _gen_process_list(
    custom_pipeline, custom_operators, template, model_name, target, eval_mode: bool = False
):
    if custom_pipeline is None:
        custom_pipeline = []
    custom_configs = _get_dict_of_operator_list(custom_pipeline)
    pipeline = template if template else custom_pipeline

    if target == 'preprocess' and template and custom_pipeline:
        template_ops = list(_get_dict_of_operator_list(template).keys())
        custom_ops = list(_get_dict_of_operator_list(custom_pipeline).keys())
        # operators in custom_ops is a subset of template_ops and in the same order
        it = iter(template_ops)
        assert all(
            elem in it for elem in custom_ops
        ), f"{custom_ops} is not a subset of {template_ops}"

    if target == 'postprocess' and template and custom_pipeline:
        # check if the custom pipeline contains extra operators after the template
        # if so, add them to the end of the pipeline; but not allow to have extra
        # operators before operators in the template. If operators in template are
        # not defined in custom pipeline but there are extra operators in custom
        # pipeline, directly append them to the end of the template pipeline.
        template_ops = list(_get_dict_of_operator_list(template).keys())
        custom_ops = list(_get_dict_of_operator_list(custom_pipeline).keys())
        custom_pipeline_has_ops_in_template = any(op in template_ops for op in custom_ops)
        for op in custom_pipeline:
            opname, attribs = _get_op(op)
            if custom_pipeline_has_ops_in_template and template_ops:
                target_op = template_ops.pop(0)
                assert opname == target_op, f"{target_op} missing from {custom_pipeline}"
            else:
                pipeline.append(op)
    template_dict = _get_dict_of_operator_list(template)

    all_ops = collections.ChainMap(custom_operators, operators.builtins)

    transforms = []
    for el in pipeline:
        opname, attribs = _get_op(el)
        if template and opname not in template_dict:
            raise ValueError(f"{opname}: Not in the template")
        try:
            operator = all_ops[opname]
        except KeyError:
            # check if the operator is a classical CV operator
            if opname in operators.builtins_classical_cv:
                raise ValueError(f"{opname} is a classical CV operator, not allowed in {target}")
            else:
                raise ValueError(f"{opname}: Unsupported {target} operator") from None
        attribs = dict(attribs or {}, **(custom_configs.get(opname) or {}))
        if eval_mode:
            attribs['__eval_mode'] = True
            if 'eval' in attribs or 'pair_eval' in attribs:
                LOG.debug(
                    f"{opname}: 'eval' or 'pair_eval' should be declared in the YAML pipeline"
                )
                if 'eval' in attribs and 'pair_eval' in attribs:
                    LOG.error(
                        f"{opname}: Both 'eval' and 'pair_eval' are present. Consider commenting out one of them."
                    )
                    raise ValueError(
                        f"{opname}: Both 'eval' and 'pair_eval' cannot be present simultaneously."
                    )
                if 'eval' in attribs and not isinstance(attribs['eval'], dict):
                    raise TypeError(f"{opname}: 'eval' must be a dictionary")
                if 'pair_eval' in attribs and not isinstance(attribs['pair_eval'], dict):
                    raise TypeError(f"{opname}: 'pair_eval' must be a dictionary")
        else:
            if 'eval' in attribs:
                del attribs['eval']
            if 'pair_eval' in attribs:
                del attribs['pair_eval']
        transforms.append(operator(**attribs))
    return transforms


def create_transforms_from_config(
    ops: list[config.ImagePreproc], custom_operators: dict[str, type[AxOperator]]
) -> list[operators.Operator]:
    """Create image preprocessing ops from ops specified in a config.Source.

    Args:
        list[config.ImagePreproc]: List of image preprocessing operator configurations.
        custom_operators (dict[str, type[AxOperator]]): Custom operators to be used in the pipeline.

    Returns:
        List[Operator]: List of instantiated image preprocessing operators.
    """
    all_ops = collections.ChainMap(operators.builtins, custom_operators)
    transforms = []

    for op in ops:
        norm = _normalize_operator_name(op.name)
        try:
            operator = all_ops[norm]
        except KeyError:
            raise ValueError(f"Unknown image processing operator: {norm}") from None

        transforms.append(operator(*op.args, **op.kwargs))

    return transforms


def _create_transforms(transform_list, transform_name, all_ops):
    """Helper function to create transforms from config

    Args:
        transform_list (List[Dict]): List of transform configurations
        transform_name (str): Name of the transform type for error messages
        all_ops (ChainMap): Combined map of custom and builtin operators

    Returns:
        List[Operator]: List of instantiated transform operators

    Raises:
        ValueError: If operator name is not found in all_ops
    """
    if not transform_list:
        return []

    _convert_yaml_operator_name(transform_list, transform_name)
    transforms = []

    for op in transform_list:
        opname, attribs = _get_op(op)
        if opname not in all_ops:
            raise ValueError(f"Unknown {transform_name} operator: {opname}")

        operator = all_ops[opname]
        transforms.append(operator(**(attribs or {})))

    return transforms


def _gen_input_transforms(custom_config, template, custom_operators):
    if custom_config is None:
        custom_config = {}
    config = template if template else custom_config
    # overwrite by custom config
    config = dict(config, **custom_config)
    source = config.pop('source', None)
    if not source:
        LOG.trace("The source is not clearly declared, default as full frame")
        source = "full"
    elif source == "image_processing":
        assert 'image_processing' in config, "Please specify the image processing operator"
        all_ops = collections.ChainMap(custom_operators, operators.builtins)
        config['image_processing'] = _create_transforms(
            config['image_processing'], 'image_processing', all_ops
        )
    elif source == "roi":
        all_ops = collections.ChainMap(custom_operators, operators.builtins)
        if 'image_processing_on_roi' in config:
            config['image_processing_on_roi'] = _create_transforms(
                config['image_processing_on_roi'], 'image_processing_on_roi', all_ops
            )
    operator = get_input_operator(source)

    return operator(**config)

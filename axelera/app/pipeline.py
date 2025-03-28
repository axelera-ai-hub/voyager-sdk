# Copyright Axelera AI, 2025
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
    pipe,
    utils,
)
from .operators import (
    AxOperator,
    Inference,
    InferenceConfig,
    compose_preprocess_transforms,
    get_input_operator,
)

LOG = logging_utils.getLogger(__name__)
COMPILER_LOG = logging_utils.getLogger('compiler')

LABELS_SENTINEL = "$$labels$$"
LABEL_FILTER_SENTINEL = "$$label_filter$$"
NUM_CLASSES_SENTINEL = '$$num_classes$$'

TASK_PROPERTIES = {
    'input',
    'preprocess',
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
    model_info: types.ModelInfo = None
    context: operators.PipelineContext = None
    inference_config: InferenceConfig = None
    inference: Optional[Inference] = None
    postprocess: List[AxOperator] = dataclasses.field(default_factory=list)
    aipu_cores: Optional[int] = None
    validation_settings: dict = dataclasses.field(default_factory=dict)
    data_adapter: types.DataAdapter = None

    def __repr__(self):
        return f"""AxTask('{self.name}',
    input={self.input},
    preprocess={self.preprocess},
    inference_config={self.inference_config},
    inference={self.inference},
    postprocess={self.postprocess},
    model_info={self.model_info},
    context={self.context},
    aipu_cores={self.aipu_cores},
    validation_settings={self.validation_settings},
    data_adapter={self.data_adapter},
)"""

    @property
    def classes(self):
        if hasattr(self.model_info, 'labels') and isinstance(
            self.model_info.labels, utils.FrozenIntEnumMeta
        ):
            return self.model_info.labels
        raise NotImplementedError("Class enum is only available with enumerated labels")


def update_pending_expansions(task):
    for ops in [task.preprocess, task.postprocess]:
        for op in ops:
            for field in op.supported:
                x = getattr(op, field)
                if x == LABELS_SENTINEL:
                    setattr(op, field, task.model_info.labels)
                elif x == LABEL_FILTER_SENTINEL:
                    setattr(op, field, task.model_info.label_filter)
                elif x == NUM_CLASSES_SENTINEL:
                    setattr(op, field, task.model_info.num_classes)


def parse_task(
    model: dict,
    custom_operators: Dict[str, AxOperator],
    model_infos: network.ModelInfos,
    eval_mode: bool = False,
) -> AxTask:
    """Parse YAML pipeline to AxTask object.

    model: a dict containing exactly one item: {model_name: processing_steps}
      Where processing_steps is a dict containing one or more of:
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
    model_name = phases.get('model_name', task_name)
    model_info = model_infos.model(model_name)
    template = _get_template_processing_steps(phases, model_info)
    for d in [template, phases]:
        d.setdefault('input', {})
        d.setdefault('preprocess', [])
        d.setdefault('postprocess', [])
        if unknown := [k for k in d.keys() if k not in TASK_PROPERTIES]:
            msg = 'is not a valid property' if len(unknown) == 1 else 'are not valid properties'
            raise ValueError(f"{', '.join(unknown)} {msg} of a Task")

    inp = _gen_input_transforms(phases['input'], template['input'], custom_operators)
    preprocess = _gen_processing_transforms(
        phases, template, custom_operators, 'preprocess', task_name, eval_mode
    )
    postprocess = _gen_processing_transforms(
        phases, template, custom_operators, 'postprocess', task_name, eval_mode
    )
    first_postprocess_op = postprocess[0] if postprocess else None
    inf_config = _gen_inference_config(first_postprocess_op, model_info)

    task = AxTask(
        task_name,
        input=inp,
        preprocess=preprocess,
        model_info=model_info,
        context=operators.PipelineContext(),
        inference_config=inf_config,
        inference=None,
        postprocess=postprocess,
        aipu_cores=model_info.extra_kwargs.get('aipu_cores'),
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


def _convert_yaml_operator_name(operation_steps: list, target: str) -> None:
    '''Remove dash and underscore from input operator names in-place'''
    if not operation_steps:
        return []
    _check_type(operation_steps, f'{target} operations', list)
    for operation in operation_steps:
        _check_type(operation, f'{target} operator properties', dict)
        key, value = operation.popitem()  # should have one element only
        _check_type(value, f'{target} operator properties', (dict, type(None)))
        operation[key.replace('-', '').replace('_', '')] = value


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
    return dict(
        vals,
        input_width=model_info.input_width,
        input_height=model_info.input_height,
        input_channel=model_info.input_channel,
        **model_info.extra_kwargs,
    )


def _get_template_processing_steps(processing_steps, model_info):
    template = {'input': None, 'preprocess': None, 'postprocess': None}
    if template_path := (processing_steps and processing_steps.get('template_path')):
        _check_type(template_path, 'template_path', str)
        template_path = os.path.expandvars(template_path)
        refs = model_info_as_kwargs(model_info)
        template.update(utils.load_yaml_by_reference(template_path, refs))
    return template


def _batched(iterable, n):
    "Batch data into tuples of length n. The last batch may be shorter."
    # batched('ABCDEFG', 3) --> ABC DEF G
    it = iter(iterable)
    while batch := tuple(itertools.islice(it, n)):
        yield batch


def _trace_model_info(model_info: types.ModelInfo, out: Callable[[str], None]):
    def col(l, r):
        return out(f'{l:>20} {r}'.rstrip())

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
    batch,
    model_dir,
    is_export: bool,
    compile_object: bool,
    deploy_mode: config.DeployMode,
    metis: config.Metis,
    model_infos: network.ModelInfos,
):
    # Compile specified model
    try:
        dataset_cfg = nn.model_dataset_from_task(task.name) or {}

        with nn.from_model_dir(model_name):
            model_obj = nn.instantiate_model_for_deployment(task)
            LOG.debug(f"Compose dataset calibration transforms")
            preprocess = compose_preprocess_transforms(task.preprocess, task.input)

            if compile_object:
                data_root_path = Path(dataset_cfg['data_dir_path'])
                if "repr_imgs_dir_path" in dataset_cfg:
                    value = dataset_cfg["repr_imgs_dir_path"]
                    if not isinstance(value, Path):
                        dataset_cfg["repr_imgs_dir_path"] = Path(value)
                calibration_data_loader = model_obj.create_calibration_data_loader(
                    preprocess,
                    data_root_path,
                    batch,
                    **dataset_cfg,
                )

                _check_calibration_data_loader(model_obj, calibration_data_loader)
                if num_cal_images > (len(calibration_data_loader) * batch):
                    raise ValueError(
                        f"Cannot use {num_cal_images} calibration images when dataset only contains "
                        f"{len(calibration_data_loader)*batch} images. Please either:\n"
                        f"  1. Reduce --num-cal-images to {len(calibration_data_loader)*batch} or less\n"
                        f"  2. Add more images to the calibration dataset"
                    )
                batch_loader = data_utils.NormalizedDataLoaderImpl(
                    calibration_data_loader,
                    model_obj.reformat_for_calibration,
                    is_calibration=True,
                    num_batches=(num_cal_images + batch - 1) // batch,
                )
                model_obj.set_calibration_normalized_loader(batch_loader)
                if deploy_mode in {config.DeployMode.QUANTIZE, config.DeployMode.QUANTIZE_DEBUG}:
                    decoration_flags = ''
                else:
                    ncores = compilation_cfg.backend_config.aipu_cores
                    decoration_flags = model_infos.determine_deploy_decoration(
                        model_name, ncores, metis
                    )

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
                    )

        LOG.trace(f'Write {task.model_info.name} model info to JSON file')
        _trace_model_info(task.model_info, LOG.trace)

        out_json = model_dir / constants.K_MODEL_INFO_FILE_NAME
        out_json.parents[0].mkdir(parents=True, exist_ok=True)
        axelera_framework = config.env.framework
        task.model_info.class_path = os.path.relpath(task.model_info.class_path, axelera_framework)

        out_json.write_text(task.model_info.to_json())
    except logging_utils.UserError:
        raise
    except exceptions.PrequantizedModelRequired as e:
        # pass the exception out to trigger prequantization
        raise
    except Exception as e:
        LOG.error(e)
        LOG.trace_exception()
        return False
    return True


def deploy_from_yaml(
    nn_name: str,
    path,
    pipeline_only,
    models_only,
    model,
    pipe_type,
    deploy_mode: config.DeployMode,
    num_cal_images,
    calibration_batch,
    data_root,
    build_root,
    is_export,
    hardware_caps: config.HardwareCaps,
    emulate,
    metis: config.Metis,
):
    with device_manager.create_device_manager(pipe_type, metis, emulate) as dm:
        return _deploy_from_yaml(
            dm,
            nn_name,
            path,
            pipeline_only,
            models_only,
            model,
            pipe_type,
            deploy_mode,
            num_cal_images,
            calibration_batch,
            data_root,
            build_root,
            is_export,
            hardware_caps,
            emulate,
            metis,
        )


def run(cmd, shell=True, check=True, verbose=False, capture_output=True):
    if verbose:
        print(cmd)
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
        raise


def _quantize_single_model(
    nn_name: str,
    model_name: str,
    num_cal_images: int,
    calibration_batch: int,
    hardware_caps: config.HardwareCaps,
    data_root: str,
    pipe_type: str,
    build_root,
    emulate,
    metis: config.Metis,
    debug: bool = False,
):
    """quantize a model in a separate process because of quantizer OOM"""
    deploy_info = f'{nn_name}: {model_name}'
    LOG.info(f"Prequantizing {deploy_info}")
    run_dir = os.environ.get('AXELERA_FRAMEWORK', '.')
    try:
        trace = '-v ' if LOG.isEnabledFor(logging_utils.TRACE) else ''
        run(
            f'{run_dir}/deploy.py --num-cal-images {num_cal_images} --model {model_name} '
            f'--calibration-batch {calibration_batch} {hardware_caps.as_argv()} '
            f'--data-root {data_root} --pipe {pipe_type} --build-root {build_root} {nn_name} '
            f'--mode QUANTIZE{"_DEBUG" if debug else ""} {trace}'
            f'--emulate {emulate} --metis {metis.name} ',
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
    path,
    pipeline_only,
    models_only,
    model,
    pipe_type,
    deploy_mode: config.DeployMode,
    num_cal_images,
    calibration_batch,
    data_root,
    build_root,
    is_export,
    hardware_caps: config.HardwareCaps,
    emulate,
    metis: config.Metis,
):
    ok = True
    compile_obj = pipe_type in ['gst', 'torch-aipu']

    nn = network.parse_network_from_path(path, data_root=data_root, from_deploy=True)
    utils.ensure_dependencies_are_installed(nn.dependencies)
    nnname = f"network {nn.name}"
    nn_dir = build_root / nn.name

    if compile_obj and metis == config.Metis.none:
        metis = device_man.get_metis_type()
        LOG.info("Detected Metis type as %s", metis.name)

    network.restrict_cores(nn, pipe_type, hardware_caps.aipu_cores, metis, deploy=True)

    requested_exec_cores = hardware_caps.aipu_cores
    model_infos = network.read_deployed_model_infos(
        nn_dir, nn, pipe_type, requested_exec_cores, metis
    )

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
                model_name, requested_exec_cores, metis
            )
            try:
                compilation_cfg = config.gen_compilation_config(
                    deploy_cores,
                    compiler_overrides,
                    deploy_mode,
                    emulate,
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
                    num_cal_images,
                    calibration_batch,
                    hardware_caps,
                    data_root,
                    pipe_type,
                    build_root,
                    emulate,
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
                            num_cal_images,
                            calibration_batch,
                            model_dir,
                            is_export,
                            compile_obj,
                            this_deploy_mode,
                            metis,
                            model_infos,
                        )
                    except exceptions.PrequantizedModelRequired as e:
                        if not retried_quantize:
                            retried_quantize = True
                            _quantize_single_model(
                                nn_name,
                                e.model_name,
                                num_cal_images,
                                calibration_batch,
                                hardware_caps,
                                data_root,
                                pipe_type,
                                build_root,
                                emulate,
                                metis,
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
    model_infos = network.read_deployed_model_infos(
        nn_dir, nn, pipe_type, requested_exec_cores, metis
    )
    if models_only or model:
        return ok
    elif ok:
        ax_precompiled_gst = ''  # TODO make this configurable on the command line
        LOG.info(f"Compile {Path(path).name}:pipeline")
        return _deploy_pipeline(
            device_man, nn, pipe_type, build_root, hardware_caps, ax_precompiled_gst, model_infos
        )
    else:
        return False


def _deploy_pipeline(
    device_man: device_manager.DeviceManager,
    nn: network.AxNetwork,
    pipe_type: str,
    build_root: Path,
    hardware_caps,
    ax_precompiled_gst,
    model_infos,
):
    # Build low-level network pipeline representation
    # and compile pre-/post-processing components
    try:
        model_infos.check_ready()
        nn.model_infos = model_infos
        get_pipeline(device_man, pipe_type, nn, build_root, hardware_caps, ax_precompiled_gst)
    except Exception as e:
        LOG.error(e)
        LOG.trace_exception()
        return False
    return True


def get_pipeline(
    device_man: device_manager.DeviceManager,
    pipe_type,
    nn,
    build_root,
    hardware_caps,
    ax_precompiled_gst,
    task_graph=None,
):
    logging_dir = build_root / nn.name / 'logs'
    logging_dir.mkdir(parents=True, exist_ok=True)
    p = pipe.create_pipe(
        device_man,
        pipe_type,
        nn,
        logging_dir,
        hardware_caps,
        ax_precompiled_gst,
        task_graph,
    )

    for asset in nn.assets:
        utils.download_and_extract_asset(asset.url, Path(asset.path), asset.md5)

    p.gen_network_pipe()
    nn.model_infos.add_label_enums(nn.datasets)
    return p


def _gen_process_list(
    custom_pipeline, custom_operators, template, model_name, target, eval_mode: bool = False
):
    if custom_pipeline is None:
        custom_pipeline = []
    _check_type(custom_pipeline, f'Task {target}', list)
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
        _check_type(el, f"{el}: {target} pipeline element", dict)
        opname, attribs = _get_op(el)
        if template and opname not in template_dict:
            raise ValueError(f"{opname}: Not in the template")
        try:
            operator = all_ops[opname]
        except KeyError:
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
    _check_type(custom_config, 'Task input', dict)
    config = template if template else custom_config
    # overwrite by custom config
    config = dict(config, **custom_config)
    source = config.pop('source', None)
    if not source:
        LOG.trace("The source is not clearly declared, default as full frame")
        source = "full"
    elif source == "image_processing":
        assert 'image_processing' in config, f"Please specify the image processing operator"
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


def _gen_inference_config(first_postprocess_op: AxOperator, model_info: types.ModelInfo):
    gst_decoder_does_dequantization_and_depadding = (
        first_postprocess_op and first_postprocess_op.gst_decoder_does_dequantization_and_depadding
    )
    # config from extra_kwargs
    config = {
        'gst_focus_layer_on_host': bool(
            model_info.extra_kwargs.get('YOLO', {}).get('focus_layer_replacement', False)
        )
    }
    return InferenceConfig.from_dict(config, gst_decoder_does_dequantization_and_depadding)

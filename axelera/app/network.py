# Copyright Axelera AI, 2025
# Define network object containing YAML pipeline, models and datasets
from __future__ import annotations

from contextlib import contextmanager
import copy
import dataclasses
import enum
import functools
import hashlib
import inspect
import itertools
import os
from pathlib import Path
import re
import sys
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

from axelera import types

from . import compile, config, constants, data_utils, logging_utils, operators, pipeline, utils
from . import yaml as YAML
from .torch_utils import torch

LOG = logging_utils.getLogger(__name__)

TOP_LEVEL_MODEL_OPTIONS = [
    'aipu_cores',
    'max_compiler_cores',
    'max_execution_cores',
    'clock_profile',
    'mvm_limitation',
]

ENUM_LABEL_CATEGORIES = ['ObjectDetection', 'Classification', 'InstanceSegmentation']


def _rel(p: str | Path) -> str:
    '''Return a relative path string for log and error messages.'''
    return os.path.relpath(str(p))


def _call_method(
    classfn: Callable,
    callable: Callable,
    arguments: Dict[str, Any],
    has_data_adapter: bool = False,
) -> Any:
    """
    Call a method with filtered arguments based on its signature.

    Args:
        classfn: The class method being called.
        callable: The actual callable object.
        arguments: Dictionary of arguments to pass to the callable.
        has_data_adapter: Whether the callable expects a dataset_config argument.

    Returns:
        The result of calling the callable with filtered arguments.

    Raises:
        TypeError: If the arguments don't match the callable's signature.
    """
    if classfn is object.__init__:
        return (
            callable(
                dataset_config=arguments.get('dataset_config'),
                model_info=arguments.get('model_info'),
            )
            if has_data_adapter
            else callable()
        )

    sig = inspect.signature(classfn)
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())

    filter_args = (
        arguments if has_kwargs else {k: v for k, v in arguments.items() if k in sig.parameters}
    )

    if has_data_adapter and 'dataset_config' not in filter_args:
        filter_args['dataset_config'] = arguments.get('dataset_config')

    try:
        return callable(**filter_args)
    except TypeError as e:
        LOG.error(f"Error calling {classfn.__name__}: {e}")
        LOG.debug(f"Arguments passed: {filter_args}")
        LOG.debug(f"Function signature: {sig}")
        raise TypeError(f"Error calling {classfn.__name__}: {e}") from None


def _create_model_class(
    ModelClass: Type[types.Model], DataAdapterClass: Optional[Type[types.DataAdapter]] = None
) -> Type[types.Model]:
    """
    Create a dynamic model class that integrates a DataAdapter with the base Model class.
    If no DataAdapter is specified, the base Model class is returned.

    Args:
        ModelClass: The base Model class.
        DataAdapterClass: Optional DataAdapter class to integrate.

    Returns:
        A new Model class with integrated DataAdapter functionality.

    Raises:
        ValueError: If DataAdapterClass is not a subclass of DataAdapter or
        if DataAdapterClass does not have a qualified __init__ method.
    """
    if DataAdapterClass:
        if not issubclass(DataAdapterClass, types.DataAdapter):
            raise ValueError(f"{DataAdapterClass} is not a subclass of DataAdapter")
    else:
        return ModelClass

    bases = (DataAdapterClass, ModelClass)

    def wrap_method(method_name):
        @functools.wraps(getattr(DataAdapterClass, method_name, None))
        def wrapper(self, *args, **kwargs):
            model_method = getattr(ModelClass, method_name, None)
            if callable(model_method):
                try:
                    return model_method(self, *args, **kwargs)
                except NotImplementedError:
                    # LOG.trace(
                    #     f"{method_name} not implemented in {ModelClass.__name__}, using {DataAdapterClass.__name__}"
                    # )
                    pass
            return getattr(DataAdapterClass, method_name)(self, *args, **kwargs)

        return wrapper

    class_dict = {
        '__init__': lambda self, *args, **kwargs: (
            (
                DataAdapterClass.__init__(
                    self,
                    dataset_config=kwargs.pop('dataset_config', None),
                    model_info=kwargs.pop('model_info', None),
                )
                if DataAdapterClass
                else None
            ),
            ModelClass.__init__(self, **kwargs),
        )[-1]
    }

    if DataAdapterClass:
        for method_name in [
            'create_calibration_data_loader',
            'create_validation_data_loader',
            'reformat_for_calibration',
            'reformat_for_validation',
            'evaluator',
        ]:
            if hasattr(DataAdapterClass, method_name):
                class_dict[method_name] = wrap_method(method_name)
    return type(f"Dynamic{ModelClass.__name__}", bases, class_dict)


def initialize_model(class_obj: Type[types.Model], arguments: Dict[str, Any]) -> types.Model:
    """
        Initialize a model instance with optional DataAdapter integration.

    Args:
        class_obj: The model class to instantiate.
        arguments: Arguments collected from YAML configs for model initialization.

    Returns:
        An instance of the types.Model

    Raises:
        ValueError: If there's an issue with DataAdapter import or initialization.
    """
    dataset_config = arguments.get('dataset_config', {})
    dataset_class = dataset_config.get('class', '')
    dataset_class_path = dataset_config.get('class_path', '')

    DataAdapter = None
    if dataset_class == 'DataAdapter':
        DataAdapter = types.DataAdapter
    elif dataset_class and dataset_class_path:
        try:
            DataAdapter = utils.import_class_from_file(dataset_class, dataset_class_path)
            if not issubclass(DataAdapter, types.DataAdapter):
                raise TypeError(f"{dataset_class} is not a subclass of DataAdapter")
            LOG.info(f"Imported DataAdapter {dataset_class} from {dataset_class_path}")
        except ImportError as e:
            LOG.warning(f"Failed to import DataAdapter: {e}")
            if dataset_class_path != arguments.get('class_path'):
                raise ValueError(
                    f"Can't find {dataset_class} in {dataset_class_path}. Please check YAML datasets section."
                )

    DynamicModelClass = _create_model_class(class_obj, DataAdapter)
    model = _call_method(class_obj.__init__, DynamicModelClass, arguments, bool(DataAdapter))
    if isinstance(model, types.DataAdapter):
        model.set_dataset_download_callback(data_utils.download_repr_dataset_impl)
    return model


def _call_init_model_deploy(model: types.Model, arguments):
    return _call_method(model.__class__.init_model_deploy, model.init_model_deploy, arguments)


def _load_labels_and_filter(model_info: types.ModelInfo, dataset: dict):
    model_info.labels = utils.load_labels(dataset.get('labels_path', ''))
    if label_filter := dataset.get('label_filter', ''):
        stripped = label_filter.strip()
        model_info.label_filter = [x for x in re.split(r'\s*[,;]\s*', stripped) if x]
    else:
        model_info.label_filter = []
    if model_info.label_filter:
        if not model_info.labels:
            key = YAML.key(dataset, 'label_filter')
            raise YAML.AxYAMLError("label_filter cannot be used if there are no labels", key)
        if invalid := sorted(set(model_info.label_filter) - set(model_info.labels)):
            invalid = ', '.join(invalid)
            key = YAML.key(dataset, 'label_filter')
            raise YAML.AxYAMLError(f"label_filter contains invalid labels: {invalid}", key)


@dataclasses.dataclass
class Asset:
    url: str = ""
    md5: str = ""
    path: str = ""


class _ModelDefinedPreprocess(operators.AxOperator):
    supported = []

    def __init__(self, model, pipeline_input_color_format: Optional[types.ColorFormat]):
        """
        When pipeline_input_color_format is None, it means users should take care of the color format themselves from their dataloader to the preprocess transform, to ensure the input image is in the correct format.
        """
        self.model = model
        self.pipeline_input_color_format = pipeline_input_color_format
        assert self.pipeline_input_color_format in [
            types.ColorFormat.RGB,
            types.ColorFormat.BGR,
        ], f"Unsupported pipeline_input_color_format: {self.pipeline_input_color_format.name}"

    def build_gst(self, gst, stream_idx: str):
        raise NotImplementedError("Please implement YAML pipeline to enable GStreamer pipeline")

    def exec_torch(self, image):
        if image.has_pil and self.pipeline_input_color_format != types.ColorFormat.RGB:
            raise ValueError(
                f"Report an issue if you really want to use {self.pipeline_input_color_format.name} with a PIL image"
            )

        # For torch, it's always PIL in and torch.Tensor out
        # TODO: for tensorflow, input can be numpy or PIL, and tf.Tensor out
        # TODO: for onnx, it's often numpy in and numpy out
        result = self.model.override_preprocess(image.aspil())
        assert isinstance(result, torch.Tensor)
        return result


class _InputPassthrough(operators.AxOperator):
    """Simple passthrough. This is used for pure model deployment"""

    def build_gst(self, gst, stream_idx):
        raise NotImplementedError("Please implement YAML pipeline to enable GStreamer pipeline")

    def exec_torch(self, image, result, meta, stream_id=0):
        result = [image]
        return image, result, meta


class ModelInfos:
    def __init__(self):
        self._models: Dict[str, types.ModelInfo] = {}
        self._manifest_paths: Dict[str, Path] = {}
        self._errors: Dict[str, str] = {}
        self._compiler_overrides: Dict[str, Dict[str, Any]] = {}
        self._enum_cache = {}

    def __eq__(self, rhs):
        return (
            isinstance(rhs, ModelInfos)
            and self._models == rhs._models
            and self._manifest_paths == rhs._manifest_paths
            and self._errors == rhs._errors
        )

    def __repr__(self):
        return f"ModelInfos(models={self._models}, manifest_paths={self._manifest_paths}, errors={self._errors})"

    def models(self) -> Iterable[types.ModelInfo]:
        return self._models.values()

    def missing(self) -> Iterable[str]:
        return self._errors.keys()

    def add_compiler_overrides(self, name: str, extra_kwargs: Dict[str, Any]):
        self._compiler_overrides[name] = _deep_copy_extra_kwargs(extra_kwargs)

    def add_error(self, idx: int, name: str, error: str, level=LOG.trace):
        level(f"%d. %s:%s", idx, name, error)
        self._errors[name] = error

    def add_model(self, model_info: types.ModelInfo, manifest_path: Optional[Path]):
        self._errors.pop(model_info.name, None)
        self._models[model_info.name] = model_info
        if manifest_path is not None:
            self._manifest_paths[model_info.name] = manifest_path

    def singular_model(self, error: str) -> types.ModelInfo:
        '''Return the only model if there is only one, otherwise raise error.'''
        if len(self._models) != 1:
            raise logging_utils.UserError(error)
        return next(iter(self._models.values()))

    def model(self, model_name: str) -> types.ModelInfo:
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not found in models")
        return self._models[model_name]

    def model_compiler_overrides(self, model_name: str, metis: config.Metis) -> Dict[str, Any]:
        if model_name not in self._compiler_overrides:
            raise ValueError(f"Model {model_name} not found in models")
        return _collapse_compiler_overrides(self._compiler_overrides[model_name], metis)

    def _hw_option(
        self,
        model: str,
        metis: config.Metis,
        option: str,
        default: Any,
        converter: Callable[[Any], Any],
    ) -> Any:
        overrides = self._compiler_overrides[model]
        opts = _collapse_compiler_overrides(overrides, metis)
        if option in opts:
            try:
                return converter(opts[option])
            except Exception:
                LOG.warning(f"Invalid {option}: {opts[option]}")
        return default

    def clock_profile(self, model: str, metis: config.Metis) -> int:
        return self._hw_option(model, metis, 'clock_profile', config.DEFAULT_CORE_CLOCK, int)

    def mvm_limitation(self, model: str, metis: config.Metis) -> int:
        return self._hw_option(model, metis, 'mvm_limitation', 100, int)

    def determine_deploy_cores(self, name: str, execution_cores: int, metis: config.Metis) -> int:
        overrides = _collapse_compiler_overrides(self._compiler_overrides.get(name, {}), metis)
        if len(self._compiler_overrides) > 1:
            if 'aipu_cores' not in overrides:
                raise logging_utils.UserError(
                    f"The pipeline has multiple models but model {name} does not specify aipu_cores"
                )

        desired = overrides.get('aipu_cores', execution_cores)
        max_compiler = overrides.get('max_compiler_cores', config.env.max_compiler_cores)
        max_exec = overrides.get('max_execution_cores', config.DEFAULT_MAX_EXECUTION_CORES)
        ncores = min(desired, max_compiler)
        if max_exec < ncores:
            due_to = f'max_execution_cores setting (for metis: {metis.name})'
            LOG.info(f"Deploying for {max_exec} cores instead of {ncores} due to {due_to}")
            max_compiler = max_exec
        return ncores

    def determine_deploy_decoration(
        self, name: str, deploy_cores: int, metis: config.Metis
    ) -> str:
        deploy_flags = [deploy_cores]
        return '-'.join(str(f) for f in deploy_flags)

    def determine_execution_cores(self, name: str, requested_cores: int, metis: config.Metis):
        overrides = _collapse_compiler_overrides(self._compiler_overrides.get(name, {}), metis)
        if len(self._compiler_overrides) > 1:
            if 'aipu_cores' not in overrides:
                raise logging_utils.UserError(
                    f"The pipeline has multiple models but model {name} does not specify aipu_cores"
                )
        desired = overrides.get('aipu_cores', requested_cores)
        max_exec = overrides.get('max_execution_cores', config.DEFAULT_MAX_EXECUTION_CORES)
        if max_exec < requested_cores:
            # TODO if we could distinguish between the user setting a value and the default then this
            # would be a warning
            due_to = f'max_execution_cores setting (for metis: {metis.name})'
            LOG.info(f"Executing on {max_exec} cores instead of {requested_cores} due to {due_to}")
        return min(desired, max_exec)

    def manifest_path(self, model_name: str) -> Path:
        if model_name not in self._models:
            raise ValueError(f"Model {model_name} not found in models")
        if model_name not in self._manifest_paths:
            raise ValueError(f"Manifest for {model_name} not found")
        return self._manifest_paths[model_name]

    @property
    def ready(self):
        return not self._errors

    def check_ready(self) -> None:
        if not self.ready:
            raise RuntimeError(f"Model(s): {', '.join(self.missing())} not deployed")

    def any_emulated_model(self):
        '''Return True if any model is emulated.'''
        for model in self._models.values():
            if model.manifest.model_lib_file != constants.K_MODEL_TVM_AIPU_LIB:
                return True
        return False

    def add_label_enums(self, datasets):
        for mi in self._models.values():
            dataset = datasets.get(mi.dataset) if datasets else None
            if dataset and not dataset.get('disable_enumeration'):
                if mi.dataset not in self._enum_cache:
                    if mi.task_category.name not in ENUM_LABEL_CATEGORIES:
                        continue
                    try:
                        dataset = datasets[mi.dataset]
                    except (KeyError, TypeError) as e:
                        LOG.warning(f"Failed to create enum for dataset {mi.dataset}: {e}")
                        continue

                    try:
                        labels = utils.load_labels(dataset.get('labels_path', ''))
                    except FileNotFoundError as e:
                        LOG.warning(f"Failed to create enum for dataset {mi.dataset}: {e}")
                        continue

                    try:
                        e = utils.FrozenIntEnum(
                            f"{utils.ident(mi.dataset)}", utils.create_enumerators(labels)
                        )

                    except ValueError as e:
                        LOG.warning(f"Failed to create enum for dataset {mi.dataset}: {e}")
                        continue
                    self._enum_cache[mi.dataset] = e
                else:
                    e = self._enum_cache[mi.dataset]
                mi.labels = e


@dataclasses.dataclass
class AxNetwork:
    """Network class containing objects to access top-level
    network configuration file and instantiate models"""

    path: str = ""
    name: str = ""
    description: str = ""
    tasks: List[pipeline.AxTask] = dataclasses.field(default_factory=list)
    custom_operators: Dict[str, operators.AxOperator] = dataclasses.field(default_factory=dict)
    model_infos: ModelInfos = dataclasses.field(default_factory=ModelInfos)
    assets: List[Asset] = dataclasses.field(default_factory=list)
    datasets: Optional[dict] = None
    dependencies: List[str] = dataclasses.field(default_factory=list)
    _instantiated_models: Dict[str, types.Model] = dataclasses.field(default_factory=dict)
    '''Note this is not parsed at parse_network time yet.'''

    _enum_cache = {}

    def __post_init__(self):
        if self.datasets is not None and not isinstance(self.datasets, dict):
            raise ValueError(f"datasets is not a dictionary: {self.datasets}")

    @property
    def hash_str(self) -> int:
        return hashlib.sha256(str(Path(self.path).resolve()).encode('utf-8')).hexdigest()[:8]

    def find_model(self, name) -> types.ModelInfo:
        return self.model_infos.model(name)

    def model_dataset_from_model(self, model_name):
        dataset_name = self.find_model(model_name).dataset
        return self._find_dataset(dataset_name) if dataset_name else {}

    def model_dataset_from_task(self, task_name):
        dataset_name = self.find_model_info_from_task(task_name).dataset
        return self._find_dataset(dataset_name) if dataset_name else {}

    def _find_dataset(self, dataset_name):
        '''Return dataset config by dataset_name.'''
        if self.datasets is None:
            raise ValueError("Missing definition of top-level datasets section")
        try:
            return self.datasets[dataset_name]
        except KeyError as e:
            raise ValueError(f"Missing definition of dataset {dataset_name}") from None

    @property
    def task_names(self) -> List[str]:
        '''Return list of task names from pipeline.'''
        return [t.name for t in self.tasks]

    @property
    def model_names(self) -> List[str]:
        '''Return list of model names from pipeline.'''
        return [t.model_info.name for t in self.tasks]

    def find_task(self, name) -> pipeline.AxTask:
        '''Find task by name from pipeline'''
        for t in self.tasks:
            if t.name == name:
                return t
        raise ValueError(f"Cannot find {name} in pipeline section: {self.task_names}")

    def find_model_info_from_task(self, task_name: str) -> types.ModelInfo:
        '''Find model info by task name from pipeline'''
        for t in self.tasks:
            if t.name == task_name:
                return t.model_info
        raise ValueError(f"Cannot find {task_name} in pipeline section: {self.task_names}")

    @contextmanager
    def from_model_dir(self, model_name: str) -> None:
        # Execute code from specified model base directory
        root_dir = os.getcwd()
        model = self.find_model(model_name)
        base_dir = model.base_dir or str(Path(model.class_path).parents[0])
        try:
            # Change working directory the model project
            sys.path.insert(1, base_dir)
            os.chdir(base_dir)
            yield
        finally:
            # Remove project from path and change back to root dir
            sys.path.remove(base_dir)
            os.chdir(root_dir)

    def model_class(self, model_name: str) -> types.Model:
        model = self.find_model(model_name)
        LOG.debug(f"Import model of type {model.class_name} from {model.class_path}")
        return utils.import_class_from_file(model.class_name, model.class_path)

    def instantiate_model(self, model_name: str, with_data_adapter: bool = False) -> types.Model:
        mi = self.find_model(model_name)
        ModelClass = self.model_class(model_name)
        if not issubclass(ModelClass, types.Model):
            raise TypeError(f"Class {ModelClass} is not a subclass of types.Model")
        LOG.debug(f"Instantiate model: {model_name}")
        dataset = self.model_dataset_from_model(model_name)
        _load_labels_and_filter(mi, dataset)
        kwargs = pipeline.model_info_as_kwargs(mi)
        kwargs['model_info'] = mi
        if with_data_adapter:
            kwargs['dataset_config'] = dataset
        model = initialize_model(ModelClass, kwargs)
        model.task_category = mi.task_category
        model.input_tensor_layout = mi.input_tensor_layout
        model.input_tensor_shape = mi.input_tensor_shape
        model.input_color_format = mi.input_color_format
        _call_init_model_deploy(model, kwargs)
        return model

    def instantiate_model_for_deployment(self, task: pipeline.AxTask) -> types.Model:
        """To deploy a model, we need to instantiate it with a data adapter, and prepare the preprocess method"""
        model_name = task.model_info.name
        model = self._instantiated_models.get(model_name)
        if not model:
            model = self.instantiate_model(model_name, with_data_adapter=True)
            self._instantiated_models[model_name] = model
        self.attach_model_specific_preprocess(model, task)
        return model

    def attach_model_specific_preprocess(self, model: types.Model, task: pipeline.AxTask):
        """Attach model-specific preprocess from types.Model to task.preprocess"""
        if not task.preprocess:
            # if not, check if preprocess method has been implemented in user model file
            LOG.debug(
                f"No pipeline specified in YAML file. Default to user-provided 'preprocess' method instead"
            )
            if not utils.is_method_overridden(model, 'override_preprocess'):
                raise RuntimeError(
                    f"Missing section preprocess in pipeline. Add to YAML or implement {model.__class__.__name__}.override_preprocess",
                )
            task.preprocess = [
                _ModelDefinedPreprocess(model, task.context.pipeline_input_color_format)
            ]

    def instantiate_data_adapter(self, model_name: str) -> types.DataAdapter:
        mi = self.find_model(model_name)
        dataset_config = self.model_dataset_from_model(model_name)
        dataset_class = dataset_config.get('class', '')
        dataset_class_path = dataset_config.get('class_path', '')

        if dataset_class and dataset_class_path:
            try:
                DataAdapterClass = utils.import_class_from_file(dataset_class, dataset_class_path)
                if not issubclass(DataAdapterClass, types.DataAdapter):
                    raise TypeError(f"{dataset_class} is not a subclass of DataAdapter")
                LOG.info(f"Imported DataAdapter {dataset_class} from {dataset_class_path}")
            except ImportError as e:
                LOG.warning(f"Failed to import DataAdapter: {e}")
            if dataset_class_path != dataset_config.get('class_path'):
                raise ValueError(
                    f"Can't find {dataset_class} in {dataset_class_path}. Please check YAML datasets section."
                )
            try:
                data_adapter = DataAdapterClass(dataset_config=dataset_config, model_info=mi)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to instantiate data adapter for model {model_name}: {e}"
                ) from e
        else:
            raise ValueError(
                f"Both class and class_path must be specified for YAML dataset {mi.dataset}"
            )

        data_adapter.set_dataset_download_callback(data_utils.download_repr_dataset_impl)
        return data_adapter

    def cleanup(self):
        '''Send stop signal to all operators in the network.'''
        for task in self.tasks:
            task.input.pipeline_stopped()
            for op in task.preprocess:
                op.pipeline_stopped()
            for op in task.postprocess:
                op.pipeline_stopped()


def _register_custom_operators(
    yaml_path: Path, custom_operators: Dict[str, operators.AxOperator], yaml: dict
):
    '''Add any custom operators referenced in the yaml to custom_operators.'''
    for name, attribs in yaml.get('operators', {}).items():
        name = name.replace('-', '').replace('_', '')
        cls = YAML.attribute(attribs, 'class')
        path = YAML.attribute(attribs, 'class_path')
        LOG.debug(f"Register custom operator '{name}' with class {cls} from {path}")

        OperatorClass = utils.import_class_from_file(cls, path)
        if name in operators.builtins:
            LOG.warning(f"{name} already in builtin-operator list; will be overwritten")
        if name in custom_operators:
            LOG.warning(f"{name} already in operator list; will be overwritten")
        custom_operators[name] = OperatorClass


def _find_template_custom_operators(custom_operators: Dict[str, operators.AxOperator], yaml):
    '''Find all templates from pipeline, and register any custom operators referenced'''
    for el in yaml.get('pipeline', []):
        model_name = next(iter(el.keys()), '')
        paths = utils.find_values_in_dict('template_path', el)
        for path in paths:
            if Path(path).is_file():
                template_yaml = utils.load_yaml_ignore_braces(path)
                # TODO rel paths in the template are treated as relative to the parent yaml
                # but I think this would be better to be relative to the template itself with:
                template_base_dir = Path(path).parent.resolve()
                utils.make_paths_in_dict_absolute(template_base_dir, template_yaml)
                _register_custom_operators(path, custom_operators, template_yaml)
            else:
                raise ValueError(f"{model_name}: template {path} not found")


def _provisional_model_info_from_yaml(name, yaml_props) -> types.ModelInfo:
    '''Read minimal data from the yaml file, for bootstrapping the pipeline load.'''
    # can probably remove this shortly
    props = dict(yaml_props)
    obj = types.ModelInfo(
        name,
        props.pop('task_category'),
        props.pop('input_tensor_shape'),
        props.pop('input_color_format'),
        props.pop('input_tensor_layout'),
        labels=pipeline.LABELS_SENTINEL,
        label_filter=pipeline.LABEL_FILTER_SENTINEL,
        weight_path=props.pop('weight_path', ''),
        weight_url=props.pop('weight_url', ''),
        weight_md5=props.pop('weight_md5', ''),
        prequantized_url=props.pop('prequantized_url', ''),
        prequantized_md5=props.pop('prequantized_md5', ''),
        dataset=props.pop('dataset', ''),
        base_dir=props.pop('base_dir', ''),
        class_name=props.pop('class', ''),
        class_path=props.pop('class_path', ''),
        version=props.pop('version', ''),
        num_classes=props.pop('num_classes', 1),
        extra_kwargs=props.pop('extra_kwargs', {}),
    )
    if len(props) == 1:
        key, _ = props.popitem()
        raise ValueError(f"{key} is not a valid key for model {name}")
    elif len(props) > 1:
        keys = sorted(props.keys())
        keys, lastkey = ', '.join(keys[:-1]), keys[-1]
        raise ValueError(f"{keys} and {lastkey} are not valid keys for model {name}")

    return obj


def parse_and_validate_datasets(datasets: Optional[dict], data_root: Optional[Path]) -> None:
    """
    This function is used to parse and validate the datasets section in the yaml file.
    It will also set the data_dir_path according to the data_dir_name or repr_imgs_dir_path.
    """
    if datasets is not None:
        pipeline._check_type(datasets, 'datasets', dict)
        for dataset_name, dataset in datasets.items():
            pipeline._check_type(dataset, f'dataset {dataset_name}', dict)
            if 'data_dir_path' in dataset:
                raise ValueError(f"data_dir_path is a reserved keyword for YAML dataset section")
            if 'class' in dataset:
                if dataset['class'] == 'DataAdapter':
                    LOG.debug(f"Using the default DataAdapter")
                    dataset['data_dir_path'] = data_root
                    if 'repr_imgs_dir_path' in dataset:
                        dataset['repr_imgs_dir_path'] = Path(dataset['repr_imgs_dir_path'])
                        if 'repr_imgs_dataloader_color_format' in dataset:
                            if dataset['repr_imgs_dataloader_color_format'] not in [
                                'RGB',
                                'BGR',
                                'GRAY',
                            ]:
                                raise ValueError(
                                    f"repr_imgs_dataloader_color_format must be RGB, BGR, or GRAY"
                                )
                    elif 'data_dir_name' not in dataset or 'repr_imgs_dir_name' not in dataset:
                        raise ValueError(
                            f"data_dir_name and repr_imgs_dir_name are both required for dataset {dataset_name} if you don't specify repr_imgs_dir_path"
                        )
                    else:
                        if data_root is not None:
                            dataset['data_dir_path'] = data_root / dataset['data_dir_name']
                            dataset['repr_imgs_dir_path'] = (
                                dataset['data_dir_path'] / dataset['repr_imgs_dir_name']
                            )
                    continue  # this is a special case, we don't need to check the rest of the dataset
                elif 'class_path' not in dataset:
                    raise ValueError(f"Missing class_path for dataset {dataset_name}")
            elif 'class_path' in dataset:
                raise ValueError(f"Missing class for dataset {dataset_name}")
            # Both 'class' and 'class_path' can be absent, so we don't raise an error in that case
            if 'data_dir_name' not in dataset:
                raise ValueError(f"Missing data_dir_name for dataset {dataset_name}")
            elif data_root is not None:
                dataset['data_dir_path'] = data_root / dataset['data_dir_name']
            if 'metrics' in dataset:
                pipeline._check_type(
                    dataset['metrics'], f"metrics for dataset {dataset_name}", list
                )
                for metric in dataset['metrics']:
                    pipeline._check_type(metric, f"metric for dataset {dataset_name}", str)

            if 'labels' in dataset and 'labels_path' in dataset:
                raise ValueError(
                    f"labels and labels_path cannot be specified at the same time for dataset {dataset_name}"
                )
            elif 'labels_path' in dataset:
                pipeline._check_type(
                    dataset['labels_path'], f"labels_path for dataset {dataset_name}", str
                )
                # labels_path is an absolute path mistakenly associated with a yaml file;
                # correct it as an absolute path but associated with data_root
                if data_root is not None:
                    if not Path(dataset['labels_path']).is_file():
                        original_labels_path = Path(dataset['labels_path'])
                        correct_labels_path = (
                            data_root / dataset['data_dir_name'] / original_labels_path.name
                        )
                        dataset['labels_path'] = str(correct_labels_path)
                        LOG.trace(
                            f"Correcting labels_path from {original_labels_path} to {correct_labels_path}"
                        )
            elif 'labels' in dataset:
                dataset['labels_path'] = data_root / dataset['data_dir_name'] / dataset['labels']

            if 'label_filter' in dataset:
                # label_filter may be a list or a comma-separated list of labels
                if isinstance(dataset['label_filter'], str):
                    dataset['label_filter'] = dataset['label_filter'].split(',')
                pipeline._check_type(
                    dataset['label_filter'], f"label_filter for dataset {dataset_name}", list
                )
                for label in dataset['label_filter']:
                    pipeline._check_type(label, f"label for dataset {dataset_name}", str)
            if 'repr_imgs_dir_name' in dataset and 'repr_imgs_dir_path' in dataset:
                raise ValueError(
                    f"repr_imgs_dir_name and repr_imgs_dir_path cannot be specified at the same time for dataset {dataset_name}"
                )
            elif 'repr_imgs_dir_name' in dataset:
                if data_root is not None:
                    dataset['repr_imgs_dir_path'] = (
                        data_root / dataset['data_dir_name'] / dataset['repr_imgs_dir_name']
                    )
            elif 'repr_imgs_dir_path' in dataset:
                dataset['repr_imgs_dir_path'] = Path(dataset['repr_imgs_dir_path'])
            reserved_keywords = ['target_split', 'imreader_backend', 'pipeline_input_color_format']
            for keyword in reserved_keywords:
                if keyword in dataset:
                    raise ValueError(
                        f"{keyword} is a reserved keyword for dataset {dataset_name},"
                        f" please use a different name for this attribute"
                    )


def _parse_network(
    path, yaml, eval_mode: bool, data_root: Optional[Path], from_deploy: bool
) -> AxNetwork:
    # Get pipeline tasks and models, defaulting pipeline to empty list if None
    if not from_deploy and yaml.get('pipeline') is None:
        raise ValueError(
            f"This YAML doesn't support inference as it doesn't define a pipeline: {path}"
        )
    pipeline_tasks = yaml.get('pipeline', []) or []
    if not pipeline_tasks:
        yaml['pipeline'] = []
    models = yaml.get('models', {}) or {}

    # Initialize empty extra_kwargs for models if needed
    for model in models.values():
        if model.get('extra_kwargs') is None:
            model['extra_kwargs'] = {}

    # Check for duplicate task names and missing models
    if pipeline_tasks and len(pipeline_tasks) != len(
        set(next(iter(t.keys())) for t in pipeline_tasks)
    ):
        raise ValueError(f"Duplicate task names in pipeline {path}")
    if not models:
        raise ValueError(f'No models defined in network {path}')

    model_infos = ModelInfos()
    for name, v in models.items():
        manifest_path = None  # Note we do not know this yet
        model_info_from_yaml = _provisional_model_info_from_yaml(name, v)
        model_infos.add_compiler_overrides(name, model_info_from_yaml.extra_kwargs)
        model_infos.add_model(model_info_from_yaml, manifest_path)

    custom_operators = {}
    _find_template_custom_operators(custom_operators, yaml)
    # register custom operators from customer setting; will overwrite template operators
    # if repeated declarations
    _register_custom_operators(path, custom_operators, yaml)

    if pipeline_tasks:
        tasks = [
            pipeline.parse_task(model, custom_operators, model_infos, eval_mode)
            for model in pipeline_tasks
        ]
    else:
        LOG.info(
            'No pipeline section found in the yaml file, assuming simple model deployment only'
        )
        tasks = []
        for mi in model_infos.models():
            tasks.append(
                pipeline.AxTask(
                    name=f'model_deployment_{mi.name}',
                    model_info=mi,
                    input=_InputPassthrough(),
                    preprocess=None,
                    context=operators.PipelineContext(),
                )
            )

    network = AxNetwork(
        path,
        yaml.get('name', ''),
        yaml.get('description', ''),
        tasks,
        custom_operators=custom_operators,
        model_infos=model_infos,
    )

    task_model_names = {t.model_info.name for t in tasks}
    model_names = {m.name for m in model_infos.models()}
    if not task_model_names.issubset(model_names):
        undefined = task_model_names - model_names
        s = 's' if len(undefined) > 1 else ''
        undefined = ', '.join(undefined)
        raise ValueError(
            f"Model{s} {undefined} referenced in tasks but not defined in models section"
        )

    if task_model_names != model_names:
        unreferenced = model_names - task_model_names
        if unreferenced:
            s = 's' if len(unreferenced) > 1 else ''
            unreferenced = ', '.join(unreferenced)
            LOG.warning(f"Model{s} {unreferenced} defined but not referenced in any task")

    if assets := yaml.get('pipeline-assets'):
        pipeline._check_type(assets, 'pipeline-assets', dict)
        for url, d in assets.items():
            pipeline._check_type(d, f'pipeline-assets element for {url}', dict)
            network.assets.append(Asset(url, d.get('md5', ''), d.get('path', '')))

    network.datasets = yaml.get('datasets', None)
    try:
        parse_and_validate_datasets(network.datasets, data_root)
    except Exception as e:
        raise type(e)(f"{network.name}: {str(e)}") from None

    internal_model_card_deps = yaml.get('internal-model-card', {}).get('dependencies', [])
    external_model_card_deps = yaml.get('model-env', {}).get('dependencies', [])
    if internal_model_card_deps and external_model_card_deps:
        LOG.warning(
            f"Dependencies found in both 'internal-model-card' and 'model-card' sections. "
            f"Using 'internal-model-card' dependencies: {internal_model_card_deps}"
        )
    network.dependencies = internal_model_card_deps or external_model_card_deps
    network.model_infos.add_label_enums(network.datasets)

    # adjust the pipeline_input_color_format for the pure model deployment case
    if len(pipeline_tasks) == 0:
        for task in network.tasks:
            dataset = network.model_dataset_from_model(task.model_info.name)
            if color_format := dataset.get('repr_imgs_dataloader_color_format'):
                LOG.debug(
                    f"Adjusting pipeline_input_color_format as {color_format} for {task.model_info.name}"
                )
                task.context.pipeline_input_color_format = types.ColorFormat.parse(color_format)
            else:
                raise ValueError(
                    f"Please specify repr_imgs_dataloader_color_format in the dataset section for {task.model_info.name}"
                )
    return network


def restrict_cores(
    network: AxNetwork,
    pipe_type: str,
    requested_cores: int,
    metis: config.Metis,
    deploy: bool = False,
):
    for task in network.tasks:
        if pipe_type in ('torch-aipu', 'torch'):
            ncores = 1
        elif deploy:
            ncores = network.model_infos.determine_deploy_cores(
                task.model_info.name, requested_cores, metis
            )
        else:
            ncores = network.model_infos.determine_execution_cores(
                task.model_info.name, requested_cores, metis
            )
        task.aipu_cores = ncores
        task.model_info.extra_kwargs['aipu_cores'] = ncores
        backend_config = task.model_info.extra_kwargs.get('compilation_config', {}).get(
            'backend_config', {}
        )
        for flag in ['single_op_network_aicore_start_idx', 'subdevices']:
            backend_config.pop(flag, None)


def parse_network_from_path(
    path: str, eval_mode: bool = False, data_root: Optional[Path] = None, from_deploy: bool = False
) -> AxNetwork:
    """Parse YAML pipeline to Network object, returns the network and the yaml."""
    # First create network and ensure all models are deployed
    LOG.debug(f"Create network from {path}")
    yaml = utils.load_yamlfile(path)
    base = Path(path).parent.resolve()
    # Make all paths in model config absolute so it can be processed from a different directory
    weight_base = Path(os.path.expandvars("${HOME}/.cache/axelera/weights"))
    yaml_from_cache = copy.deepcopy(yaml)
    utils.make_weight_paths_in_dict_absolute(weight_base, yaml_from_cache)
    yaml_from_base = copy.deepcopy(yaml)
    utils.make_paths_in_dict_absolute(base, yaml_from_base)
    if models := yaml_from_base.get('models', {}):
        for model_name, model_info in models.items():
            if Path(model_info.get('weight_path', '')).is_file():
                yaml['models'][model_name] = model_info
            else:
                yaml['models'][model_name] = yaml_from_cache['models'][model_name]
    utils.make_paths_in_dict_absolute(base, yaml)
    return _parse_network(path, yaml, eval_mode, data_root, from_deploy)


def _collapse_compiler_overrides(
    extra_kwargs: Dict[str, Any], metis_type: config.Metis
) -> Dict[str, Any]:
    '''Parse compiler option overrides from extra_kwargs (via model info) in the yaml.'''
    obj = {}
    read_from = [extra_kwargs, extra_kwargs.get(metis_type.name, {})]
    for extra_kwargs in read_from:
        for top in TOP_LEVEL_MODEL_OPTIONS:
            if top in extra_kwargs:
                obj[top] = extra_kwargs[top]

        if "compilation_config" in extra_kwargs:
            obj['compilation_config'] = {}
            for section_name in ['backend_config', 'quantization_config', 'frontend_config']:
                if section_name in extra_kwargs['compilation_config']:
                    obj['compilation_config'][section_name] = {}
                    # TODO validate boolean values
                    for key, value in extra_kwargs['compilation_config'][section_name].items():
                        obj['compilation_config'][section_name][key] = value

    return obj


def _deep_copy_extra_kwargs(extra_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    # deep copy it manually but not with deepcopy because the YAML library does weird stuff
    def _visit(v):
        if isinstance(v, (bool, bytes, int, str, float, complex)):
            return v
        if isinstance(v, dict):
            return {k: _visit(v) for k, v in v.items()}
        if isinstance(v, list):
            return [_visit(v) for v in v]
        if isinstance(v, tuple):
            return tuple(_visit(v) for v in v)
        raise TypeError(f"Unexpected type in extra_kwargs: {type(v)} : {v!r}")

    return _visit(extra_kwargs)


def _find_manifest(i, root):
    if (p := root / constants.K_MANIFEST_FILE_NAME).is_file():
        return p
    raise RuntimeError(f"No manifest file found in {root}")


def _load_manifest_and_check_batch(manifest_file, needed_cores):
    m = compile.load_manifest_from_file(manifest_file)
    batch = (m.input_shapes and m.input_shapes[0] and m.input_shapes[0][0]) or 0
    if batch != needed_cores:
        raise RuntimeError(
            f"{manifest_file}: Has wrong batch size ({batch}) for aipu-cores {needed_cores}"
        )
    return m


def read_deployed_model_infos(
    nn_dir: Path,
    nn: AxNetwork,
    pipe_type: str,
    requested_cores: int,
    metis: config.Metis,
) -> ModelInfos:
    '''
    For all models, check if both binary and JSON are available.

    If all required files exist and the deployed info aligns with aipu_cores then
    returned model_infoss.ready will be True.

    `execution_cores` is the value given to `--aipu-cores` and is the number of cores
    to execute upon, unless overridden by the model's `max_execution_cores` setting
    for the metis board type.
    '''
    model_infos = ModelInfos()
    for i, model_info_from_yaml in enumerate(nn.model_infos.models(), 1):
        name = model_info_from_yaml.name
        model_json = nn_dir / f'{name}/{constants.K_MODEL_INFO_FILE_NAME}'
        model_infos.add_compiler_overrides(name, model_info_from_yaml.extra_kwargs)
        if not model_json.is_file():
            model_infos.add_error(i, name, f"{_rel(model_json)}: File not found")
            continue

        LOG.trace(f"{i}. {_rel(model_json)}: Available")
        try:
            model_info = types.ModelInfo.from_json(model_json.read_text())
            model_info.class_path = os.path.join(config.env.framework, model_info.class_path)
            # # fill manifest with dummy values for non-aipu pipelines, it
            # # will be overwritten by the manifest file if it exists
            # model_info.manifest = types.Manifest("", (), ())
        except Exception as e:
            msg = f"Failed to load model info from {_rel(model_json)}: {e}"
            model_infos.add_error(i, name, msg, LOG.exception)
            continue

        if pipe_type == 'torch':
            model_infos.add_model(model_info, model_json)
            continue
        elif pipe_type == 'quantized':
            try:
                quantized_dir = nn_dir / f'{name}/{constants.K_MODEL_QUANTIZED_DIR}'
                manifest_file = _find_manifest(i, quantized_dir)
                model_info.manifest = compile.load_prequant_manifest(quantized_dir)
            except RuntimeError as e:
                model_infos.add_error(i, name, str(e))
            else:
                LOG.trace(f"{i}. {manifest_file}: quantized manifest available")
                model_infos.add_model(model_info, manifest_file)
            continue

        # elif pipe_type in ['torch-aipu', 'gst']:
        deploy_cores = model_infos.determine_deploy_cores(name, requested_cores, metis)
        root = model_json.parent / model_infos.determine_deploy_decoration(
            name, deploy_cores, metis
        )
        try:
            manifest_file = _find_manifest(i, root)
        except RuntimeError as e:
            model_infos.add_model(model_info_from_yaml, {})
            model_infos.add_error(i, name, str(e))
            continue

        try:
            model_info.manifest = _load_manifest_and_check_batch(manifest_file, deploy_cores)
        except RuntimeError as e:
            model_infos.add_error(i, name, str(e))
        else:
            LOG.trace(f"{i}. {_rel(manifest_file)}: Available")
            model_infos.add_model(model_info, manifest_file)

    return model_infos

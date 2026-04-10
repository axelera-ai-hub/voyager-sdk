# Copyright Axelera AI, 2024
import enum
import functools
import importlib
import inspect
from pathlib import Path
import re
import sys
import typing

from .. import logging_utils, utils
from .types import (
    Any,
    Bool,
    Enum,
    Float,
    Int,
    IntEnum,
    List,
    MapPattern,
    OperatorMap,
    Optional,
    Regex,
    Required,
    Sentinel,
    Str,
    Union,
    Variable,
)

LOG = logging_utils.getLogger(__name__)

NESTED_OPERATORS = ("inputfromroi", "inputwithimagepreprocessing")

# Fields in generated operators which support $$Sentinel$$, to be replaced by model
# info in the pipeline at runtime
SENTINELS = {
    'labels': '$$labels$$',
    'label_filter': '$$label_filter$$',
    'num_classes': '$$num_classes$$',
}


def _load_op(class_path, class_name):
    MODULE = 'schema_op_location'
    spec = importlib.util.spec_from_file_location(MODULE, class_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[MODULE] = module
    spec.loader.exec_module(module)
    return getattr(module, class_name)


def _find_template_operators(network):
    '''Find any customer operators from templates in the yaml'''
    operators = {}
    yaml_pipeline = network.get('pipeline', [])
    if yaml_pipeline is not None:
        for el in yaml_pipeline:
            paths = utils.find_values_in_dict('template_path', el)
            for path in paths:
                if Path(path).is_file():
                    template_yaml = utils.load_yaml_ignore_braces(path)
                    template_base_dir = Path(path).parent.resolve()
                    utils.make_paths_in_dict_absolute(template_base_dir, template_yaml)
                    operators.update(template_yaml.get('operators', {}))
                else:
                    raise FileNotFoundError(
                        f"Template file not found: {path}\n"
                        f"Check that the template_path in your model card points to a valid file."
                    )
    return operators


def _parse_configs(op: object) -> typing.List[typing.Tuple[str, typing.Any, bool]]:
    sig = inspect.signature(op)
    hints = typing.get_type_hints(op)
    configs = []
    for name, typ in hints.items():
        typ = hints.get(name, str)
        if typing.get_origin(typ) is typing.Union:
            # NoneType in Union means typing.Optional, which is irrelevant to the schema,
            # hence remove it
            args = tuple(a for a in typing.get_args(typ) if a is not type(None))
            typ = args[0] if len(args) == 1 else typing.Union[args]

        req = sig.parameters[name].default == inspect.Parameter.empty
        configs.append((name, typ, req))
    return configs


def _to_type(t, allow_variable=True, allow_sentinel=False, existing=None, use_enum_values=False):
    from ..operators import PreprocessOperator

    # From YAML POV there is no difference between a str and a Path
    if t is str or t is Path:
        cast = Str
    elif t is int:
        cast = Int
    elif t is float:
        cast = Float
    elif t is bool:
        cast = Bool
    elif inspect.isclass(t) and issubclass(t, enum.Enum):
        if use_enum_values and all(isinstance(e.value, int) for e in t):
            cast = IntEnum[[e.value for e in t]]
        else:
            cast = Enum[[e.value for e in t]] if use_enum_values else Enum[[e.name for e in t]]
    elif t is PreprocessOperator and existing is not None:
        cast = existing
    elif typing.get_origin(t) is typing.List or typing.get_origin(t) is list:
        if typing.get_args(t):
            cast = List[
                _to_type(
                    typing.get_args(t)[0],
                    allow_variable=False,
                    existing=existing,
                    use_enum_values=use_enum_values,
                )
            ]
        else:
            cast = List[Any]
    elif t is list:
        cast = List[Any]
    # This may need refining in the future, only needed by `algo_params` in tracker as of writing,
    # which in turn is only used to write to json again, so type casting is not necessary.
    elif typing.get_origin(t) is typing.Dict or typing.get_origin(t) is dict:
        if typing.get_args(t):
            cast = MapPattern[
                _to_type(
                    typing.get_args(t)[1],
                    allow_variable=False,
                    existing=existing,
                    use_enum_values=use_enum_values,
                )
            ]
        else:
            cast = MapPattern[Any]
    elif t is dict:
        cast = MapPattern[Any]
    elif typing.get_origin(t) is typing.Union:
        args = typing.get_args(t)
        # The second case is for `typing.Optional``, which is really `typing.Union[..., None]`.
        # We ignore it for StrictYAML.
        if len(args) == 1 or (len(args) == 2 and args[1] is type(None)):
            cast = _to_type(
                args[0],
                allow_variable=False,
                existing=existing,
                use_enum_values=use_enum_values,
            )
        else:
            cast = Union[
                [
                    _to_type(
                        a,
                        allow_variable=False,
                        existing=existing,
                        use_enum_values=use_enum_values,
                    )
                    for a in args
                ]
            ]
    else:
        cast = Any
    types = []
    if allow_variable:
        types.append(Variable)
    if allow_sentinel:
        types.append(Sentinel)
    types.append(cast)
    return Union[types] if len(types) > 1 else cast


def _add_special_cases(configs, eval, op):
    from ..operators import PreprocessOperator

    special_cases = [
        c[1]
        for c in [
            (
                issubclass(op, PreprocessOperator),
                {
                    Optional["stream_match"]: Union[
                        {
                            Optional["include"]: List[Int],
                            Optional["exclude"]: List[Int],
                        },
                        Int,
                        List[Int],
                        Regex[r'\.\*'],
                    ],
                },
            ),
        ]
        if c[0]
    ]
    for case in special_cases:
        for key, typ in case.items():
            configs[key] = typ
            eval[key] = typ


def _generate_operator(op, existing=None):
    configs = {}
    _eval = {}
    for key, typ, req in _parse_configs(op):
        typ = _to_type(typ, allow_sentinel=key in SENTINELS, existing=existing)
        configs[Required[key] if req else Optional[key]] = typ
        _eval[Optional[key]] = typ

    _add_special_cases(configs, _eval, op)

    configs[Optional["eval"]] = _eval
    configs[Optional["pair_eval"]] = _eval
    return configs


@functools.lru_cache
def _generate_builtins():
    '''Generate the list of built-in operators supported by the schema.'''
    from ..operators import builtins, builtins_classical_cv

    combined_dict = {**builtins, **builtins_classical_cv}
    return {
        name: _generate_operator(op)
        for name, op in combined_dict.items()
        if name not in NESTED_OPERATORS
    }


def _generate_nested_builtins(existing):
    from ..operators import builtins, builtins_classical_cv

    existing = {Optional[k]: v for k, v in existing.items()}
    existing["_allow_key_dashes"] = True
    combined_dict = {**builtins, **builtins_classical_cv}
    return {
        name: _generate_operator(op, existing)
        for name, op in combined_dict.items()
        if name in NESTED_OPERATORS
    }


def generate_operators(path: str, base: str) -> None:
    if path and base:
        network = utils.load_yaml_ignore_braces(path)
        utils.make_paths_in_dict_absolute(base, network)
        operators = network.get('operators', {})
        template_operators = _find_template_operators(network)
        operators = dict(template_operators, **operators)
    else:
        operators = {}
    custom_schemas = {}
    for op_name, attribs in operators.items():
        if not ('class' in attribs and 'class_path' in attribs):
            raise ValueError(
                f"Operator {op_name} must have a 'class' and 'class_path' attribute defined in the yaml"
            )
        class_name = attribs['class']
        class_path = attribs['class_path']
        op = _load_op(class_path, class_name)
        schema_name = re.sub(r"[-_]", "", op_name)
        custom_schemas[schema_name] = _generate_operator(op)
    builtin_schemas = _generate_builtins()
    schemas = dict(builtin_schemas, **custom_schemas)
    schemas = dict(schemas, **_generate_nested_builtins(schemas))
    schemas = {Optional[k]: v for k, v in schemas.items()}
    schemas["_type"] = OperatorMap
    schemas["_allow_key_dashes"] = True
    return schemas


def _get_compiler_config():
    try:
        from axelera.compiler.config import CompilerConfig
    except ImportError:
        LOG.info(
            "axelera.compiler could not be found. YAML compilation_config will not be validated."
        )
        return None
    except Exception as e:
        LOG.info(
            f"axelera.compiler found, but the following error occured during import: {e}. YAML compilation_config will not be validated."
        )
        return None
    return CompilerConfig()


@functools.lru_cache
def generate_compilation_configs(load_compiler_config):
    if load_compiler_config and (conf := _get_compiler_config()):
        schema = {}
        for f, t in conf.model_fields.items():
            schema_type = _to_type(t.annotation, allow_variable=False, use_enum_values=True)
            schema[Optional[f]] = schema_type
            if alias := t.alias:
                schema[Optional[alias]] = schema_type
        return schema
    return MapPattern[Any]

# Copyright Axelera AI, 2024
# Base operator class
from __future__ import annotations

import abc
import copy
import enum
import functools
import typing
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Type, Union, final

from axelera import types

from .. import gst_builder, logging_utils
from ..torch_utils import torch

if TYPE_CHECKING:
    ImageToTensorTransform = Callable[[types.Image], torch.Tensor]
    from pathlib import Path

    from . import PipelineContext

LOG = logging_utils.getLogger(__name__)


class EvalMode(enum.Enum):
    NONE = enum.auto()
    EVAL = enum.auto()
    PAIR_EVAL = enum.auto()

    def __bool__(self):
        return self != EvalMode.NONE


def _compile_fn(code, name):
    ns = {}
    exec(code, globals(), ns)
    return ns[name]


def _default_repr(v):
    if isinstance(v, enum.Enum):
        return f"'{v.name}'"
    return repr(v)


def _init_fn(cls):
    params = cls.required + [f'{k}={_default_repr(v)}' for k, v in cls.defaults.items()]
    param_names = [p.split('=')[0] for p in params]

    # Different reserved keywords for different operator types
    base_reserved_keywords = {'eval', 'pair_eval', '_override_params_for_eval'}
    preprocess_reserved_keywords = {'stream_match'}

    reserved_keywords = base_reserved_keywords
    if cls._PREPROCESS_OPERATOR:
        reserved_keywords = base_reserved_keywords | preprocess_reserved_keywords

    used_reserved_keyword = reserved_keywords.intersection(param_names)
    if used_reserved_keyword:
        raise ValueError(
            f"{', '.join(used_reserved_keyword)} is a reserved keyword and cannot be used as parameters"
        )

    params_str = ', '.join(params)
    if params_str:
        params_str += ', '
    params_str += 'eval=None, pair_eval=None, __eval_mode=None'
    if cls._PREPROCESS_OPERATOR:
        params_str += ', stream_match=r".*"'

    init_body = '\n    '.join(f'self.{k} = {k}' for k in cls.required + list(cls.defaults.keys()))

    # Add PreprocessOperator specific initialization
    if cls._PREPROCESS_OPERATOR:
        init_body += '''
    try:
        self._stream_match = stream_match
        if isinstance(self._stream_match, (int, list, dict)):
            if isinstance(self._stream_match, dict):
                if not ('include' in self._stream_match or 'exclude' in self._stream_match):
                    raise ValueError("Dict stream_match must contain either 'include' or 'exclude' key")
                if 'include' in self._stream_match and 'exclude' in self._stream_match:
                    raise ValueError("Cannot specify both 'include' and 'exclude' in stream_match")
        elif self._stream_match != r'.*' and self._stream_match is not None:
            raise ValueError("stream_match must be an int, list, dict, '.*', or None")
    except Exception as e:
        raise ValueError(f"Invalid stream_match parameter for {self.__class__.__name__}: {str(e)}") from e
    '''

    init_body += '''
    self._eval_params = None
    self._eval_mode = EvalMode.NONE
    if __eval_mode:
        if eval:
            self._eval_mode = EvalMode.EVAL
            self._eval_params = eval
        elif pair_eval:
            self._eval_mode = EvalMode.PAIR_EVAL
            self._eval_params = pair_eval
        else: # no override and pair_eval is not set, default as EVAL
            self._eval_mode = EvalMode.EVAL
    else:
        if eval is not None or pair_eval is not None:
            raise ValueError("'eval' and 'pair_eval' are reserved keywords and cannot be used as parameters")
    '''
    return _compile_fn(
        f'''\
def {cls.__name__}(self, /, {params_str}):
    {init_body}
    AxOperator._post_init(self)
    self._post_init()
''',
        cls.__name__,
    )


def _all_param_names(cls):
    params = cls.required + list(cls.defaults.keys())
    return params


def _repr_fn(cls):
    params = [f"{x}={{self.{x}!r}}" for x in _all_param_names(cls)]
    return _compile_fn(
        f'''\
def __repr__(self) -> str:
    parts = [{', '.join(f"f'{p}'" for p in params)}]
    if self._model_name is not None:
        parts.append(f"model_name={{self._model_name!r}}")
    return f'{cls.__name__}({{", ".join(parts)}})'
''',
        '__repr__',
    )


def _eq_fn(cls):
    params = ' and '.join(f"self.{x} == other.{x}" for x in _all_param_names(cls))
    if not params:
        params = 'True'
    return _compile_fn(
        f'''\
def __eq__(self, other) -> bool:
    if not isinstance(other, self.__class__):
        return NotImplemented
    return {params} and self._model_name == other._model_name
''',
        '__eq__',
    )


def _is_normal_prop(k, v):
    return not callable(v) and not k.startswith('_') and not isinstance(v, property)


class AxOperator(abc.ABC):
    """Base Axelera Operator Class"""

    _PREPROCESS_OPERATOR = False
    _model_name = None
    _compiled_model_dir = None
    required = []
    defaults = {}
    supported = []
    property_types = {}

    @classmethod
    def name(cls):
        return cls.__name__.replace('-', '').replace('_', '').lower()

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # get existing user vars before we add any more
        members = [(k, getattr(cls, k)) for k in cls.__dict__]
        user_vars = [k for k, v in members if _is_normal_prop(k, v)]
        cls.required = []
        cls.defaults = {}
        cls.property_types = {}
        annotations = getattr(cls, '__annotations__', {})

        for member, mtype in annotations.items():
            cls.property_types[member] = mtype
            try:
                cls.defaults[member] = getattr(cls, member)
            except AttributeError:
                # note we are not using the type annotation here
                cls.required.append(member)
        for member in user_vars:
            if member not in cls.property_types:
                cls.property_types[member] = type(getattr(cls, member))
                cls.defaults[member] = getattr(cls, member)
        for var in cls.defaults:
            delattr(cls, var)
        cls.supported = cls.required + list(cls.defaults.keys())

        if 'validation_settings' in cls.__dict__:
            raise TypeError(
                f"{cls.__name__} tries to override final property 'validation_settings'; please use register_validation_params instead"
            )

        if '__init__' not in cls.__dict__:
            cls.__init__ = _init_fn(cls)
        if '__repr__' not in cls.__dict__:
            cls.__repr__ = _repr_fn(cls)
        cls.__eq__ = _eq_fn(cls)

    @classmethod
    def add_defaults(cls, attribs):
        resolved = copy.deepcopy(cls.defaults)
        resolved.update(attribs)
        return resolved

    def _enforce_member_type(self, member: str):
        '''Check that the member value is of the correct type for the annotation.

        If the value is not of the correct type, try to convert it to the correct
        type.
        '''
        # NOTE that we do not support typing based types here, and for this reason
        # it is not enabled for all members by default. It is opt-in.
        mtype = typing.get_type_hints(self.__class__)[member]
        value = getattr(self, member)
        if isinstance(mtype, type) and issubclass(mtype, enum.Enum) and isinstance(value, str):
            if hasattr(mtype, 'parse'):
                try:
                    value = mtype.parse(value)
                except ValueError:
                    raise ValueError(
                        f"Invalid value for {member}: {value} (expected one of {', '.join(mtype.__members__)})"
                    ) from None
            else:
                try:
                    value = getattr(mtype, value)
                except AttributeError:
                    raise ValueError(
                        f"Invalid value for {member}: {value} (expected one of {', '.join(mtype.__members__)})"
                    ) from None
        elif not isinstance(value, mtype):
            try:
                value = mtype(value)
            except ValueError:
                raise ValueError(
                    f"Invalid value for {member}: {value} (expected convertible to {mtype.__name__})"
                ) from None
        setattr(self, member, value)

    def _post_init(self) -> None:
        """
        `eval` and `pair_eval` are reserved keyword that overrides the default operator configs for
        evaluation. It is not allowed to be used as a parameter in the operator's constructor;
        instead, it is set by the YAML pipeline along with the `__eval_mode` flag.
        """

        if self._eval_params:
            for key, value in self._eval_params.items():
                if hasattr(self, key):
                    setattr(self, key, value)
                else:
                    raise AttributeError(
                        f"Attribute '{key}' not found in {self.__class__.__name__}"
                    )

        self._validation_settings = {'pair_validation': False}
        if self._eval_mode == EvalMode.PAIR_EVAL:
            self._validation_settings['pair_validation'] = True

    @abc.abstractmethod
    def exec_torch(self, *args):
        pass

    @abc.abstractmethod
    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        pass

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        where: str,
        compiled_model_dir: Path,
    ):
        '''Perform any initialisation that requires access to model info or task num,
        and also update model_info properties dynamically. Here we set frequent properties
        that are used in the exec_torch and build_gst methods.'''
        self._task_category = model_info.task_category
        self._taskn = taskn
        self._where = where
        self._model_name = model_info.name
        self.task_name = task_name
        self._compiled_model_dir = compiled_model_dir

    def pipeline_stopped(self):
        '''Pipeline stop signal. It can be overridden by subclasses to perform any cleanup.'''

    @property
    @final
    def model_name(self):
        return self._model_name

    @property
    def compiled_model_dir(self):
        return self._compiled_model_dir

    @property
    @final
    def validation_settings(self):
        """
        Parameters to pass to constructing the Evaluator. Default to
        {'pair_validation': False}. If pair validation is enabled, this should
        be a dictionary of parameters built from the pair_validation_params.
        """
        return self._validation_settings

    @final
    def register_validation_params(self, params: Dict[str, Union[str, Any]]):
        """
        Register custom parameters into the _validation_settings dictionary.

        Args:
            params (Dict[str, Union[str, Any]]): A dictionary of parameters to register.

        Raises:
            AttributeError: If any key in params is already present in _validation_settings.
        """
        for key, value in params.items():
            if key in self._validation_settings:
                raise AttributeError(
                    f"Parameter '{key}' is already registered in validation settings."
                )
            self._validation_settings[key] = value

    @property
    @final
    def eval_mode(self):
        '''Eval mode is default as NONE and triggered by the pipeline'''
        return self._eval_mode

    @property
    @final
    def gst_decoder_does_dequantization_and_depadding(self):
        '''A special flag to tell inference operator that the decoder does dequantization
        and depadding so inference operator can skip these steps.
        '''
        return getattr(self, '_gst_decoder_does_dequantization_and_depadding', False)

    @gst_decoder_does_dequantization_and_depadding.setter
    def gst_decoder_does_dequantization_and_depadding(self, value: bool):
        '''Set the flag in your decoder's _post_init() method if your decoder has optimized
        with dequantization and depadding.
        '''
        self._gst_decoder_does_dequantization_and_depadding = value


class PreprocessOperator(AxOperator):
    """
    Base class for preprocessing operators in the Axelera framework.

    This class is tasked with transforming the input image into a format that is optimized for the model,
    typically involving tasks such as normalization, resizing, and formatting adjustments. It is specifically
    designed to manage a single input and output image, focusing primarily on image preprocessing operations.
    """

    _PREPROCESS_OPERATOR = True

    def stream_check_match(self, stream_id):
        match = self._stream_match

        if match == r'.*' or match == None:
            return True

        if isinstance(match, (int, list)) and stream_id in (
            match if isinstance(match, list) else [match]
        ):
            return True

        if isinstance(match, dict):
            if 'include' in match and 'exclude' in match:
                raise ValueError("Specify only one of include or exclude.")

            include, exclude = match.get('include'), match.get('exclude')
            if not (include or exclude):
                raise ValueError("Specify either include or exclude list.")

            if include is not None and stream_id in include:
                return True
            if exclude is not None and stream_id not in exclude:
                return True

        return False

    @property
    def stream_match(self):
        return self._stream_match

    @abc.abstractmethod
    def exec_torch(
        self, image: Union[torch.Tensor, types.Image, types.img.PILImage]
    ) -> Union[torch.Tensor, types.Image, types.img.PILImage]:
        """
        Abstract method to process an image. The output format generally mirrors the input format.
        Operators with the name 'ToTensor' are expected to convert types.Image to torch.Tensor. There
        are no operators that perform the reverse conversion.

        Args:
            image (Union[torch.Tensor, types.Image, img.PILImage]): The image to be processed.

        Returns:
            Union[torch.Tensor, types.Image]: The processed image, prepared for input into the model.

        Raises:
            TypeError: If the input type is neither types.Image nor torch.Tensor.
        """
        if not isinstance(image, (types.Image, torch.Tensor)):
            raise TypeError(
                f"Expected input type types.Image or torch.Tensor, received {type(image).__name__} instead."
            )


def builtin(cls: Type[AxOperator]):
    '''Class decorator to mark an operator as builtin.'''
    cls.builtin = True
    name = cls.name()
    if name in builtins:
        LOG.warning(f"{name} already in Operator list; will be overwritten")
    builtins[name] = cls
    return cls


builtins: Dict[str, Type[AxOperator]] = {}
'''Mapping of lower case operator names to operator classes.'''


def compose_preprocess_transforms(
    transform_list: List[AxOperator], input_op: AxOperator
) -> ImageToTensorTransform:
    """Compose a list of AxOperator into a callable transform.
    Always include a image conversion transform at the beginning of the list.
    If input_op is provided, it will be executed first and only the image part of its output will be used."""

    def compose_two_funcs(f, g):
        return lambda x: g(f(x))

    def wrap_input_op(input_op):
        # Wrap input operator to only return the image part
        def wrapped(x):
            image, _, _ = input_op.exec_torch(x, None, None)
            return image

        return wrapped

    transforms = [wrap_input_op(input_op)]
    transforms.extend([t.exec_torch for t in transform_list])

    return functools.reduce(compose_two_funcs, transforms)

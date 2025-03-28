# Copyright Axelera AI, 2024
# Pipeline operators

from . import custom_preprocessing, inference, mega, postprocessing, preprocessing
from .base import AxOperator, EvalMode, PreprocessOperator, builtins, compose_preprocess_transforms
from .context import PipelineContext
from .inference import AxeleraDequantize, Inference, InferenceConfig
from .input import Input, InputFromROI, InputWithImageProcessing, get_input_operator
from .preprocessing import InterpolationMode

for _op in builtins.values():
    globals()[_op.__name__] = _op

__all__ = [
    "AxOperator",
    "PreprocessOperator",
    "EvalMode",
    "compose_preprocess_transforms",
    "Input",
    "InputFromROI",
    "InputWithImageProcessing",
    "get_input_operator",
    "InterpolationMode",
] + [x.__name__ for x in builtins.values() if not x.__name__.startswith("_")]

# Copyright Axelera AI, 2023
# Pipeline operators

from . import classical_cv, custom_preprocessing, inference, mega, postprocessing, preprocessing
from .base import (
    AxOperator,
    BaseClassicalCV,
    EvalMode,
    PreprocessOperator,
    builtins,
    builtins_classical_cv,
    compose_preprocess_transforms,
)
from .context import PipelineContext
from .inference import AxeleraDequantize, Inference, InferenceOpConfig
from .input import Input, InputNoTiles, InputFromROI, InputWithImageProcessing, get_input_operator
from .preprocessing import InterpolationMode

for _op in builtins.values():
    globals()[_op.__name__] = _op

for _op in builtins_classical_cv.values():
    globals()[_op.__name__] = _op

__all__ = (
    [
        "AxeleraDequantize",
        "AxOperator",
        "BaseClassicalCV",
        "builtins_classical_cv",
        "builtins",
        "classical_cv",
        "compose_preprocess_transforms",
        "custom_preprocessing",
        "EvalMode",
        "get_input_operator",
        "inference",
        "Inference",
        "InferenceOpConfig",
        "Input",
        "InputNoTiles",
        "InputFromROI",
        "InputWithImageProcessing",
        "InterpolationMode",
        "mega",
        "PipelineContext",
        "postprocessing",
        "preprocessing",
        "PreprocessOperator",
    ]
    + [x.__name__ for x in builtins.values() if not x.__name__.startswith("_")]
    + [x.__name__ for x in builtins_classical_cv.values() if not x.__name__.startswith("_")]
)

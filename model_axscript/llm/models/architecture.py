# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Architecture interface + registry.

An LLMArchitecture defines the quantizable torch reference and produces
a quantized_model.QuantizedModel; it knows nothing about kernels or
compilation. Each model module registers one via register_architecture
and the pipeline looks it up with get_architecture.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # annotation only: this module itself computes nothing
    import torch


class LLMArchitecture(ABC):
    """The per-model contract: produce a quantized, exportable model."""

    #: registry key
    name: str

    @abstractmethod
    def load_pretrained(self, repo_id: str, **kwargs) -> torch.nn.Module:
        """Build the float reference model from a checkpoint id."""

    @abstractmethod
    def quantize(self, model, calibration, *, smooth_alpha: float, qk_alpha: float) -> None:
        """Calibrate + fake-quantize model in place (W8A8 + SmoothQuant)."""

    @abstractmethod
    def record_lut_ranges(self, model, calibration) -> None:
        """Record each nonlinearity's input range for downstream LUT fitting."""

    def set_calibration_recording(self, model, on: bool) -> None:
        """Enable/disable the LUT-range (and any side-channel) recorders so they capture during the
        quantize calibration passes instead of in a separate pass -- used by --parallel to
        fold LUT-range recording into calibration. The captured ranges are on the fp, OBSERVE-mode
        calibration activations; for GEMM-downstream LUTs that's smoothing-invariant and differs from
        the post-quantize ranges only by activation quant noise (negligible for a min/max envelope).
        Default toggles the generic RangeObservers; architectures with extra recorders override."""
        from ..quant import set_lut_recording

        set_lut_recording(model, on)

    @abstractmethod
    def num_layers(self, model) -> int:
        """Number of decoder layers."""

    @abstractmethod
    def config_to_dict(self, model_config) -> dict:
        """Serialise the model's config object to a JSON-able dict (for the export)."""

    @abstractmethod
    def config_from_dict(self, data: dict):
        """Rebuild the model's config object from config_to_dict output."""

    @abstractmethod
    def export_host_io(self, model) -> dict:
        """Host-side params for the embedding/final-norm/RoPE boundary.

        A {name: np.ndarray} dict with at least embed_weight,
        final_norm_weight, rms_eps, head_dim, num_q_heads,
        rope_theta.
        """


_REGISTRY: dict[str, LLMArchitecture] = {}


def register_architecture(arch: LLMArchitecture) -> LLMArchitecture:
    """Register an architecture instance under its name (idempotent)."""
    _REGISTRY[arch.name] = arch
    return arch


_BUNDLED = False


#: The architectures this package registers. Imported by NAME rather than with
#: `from . import ...`, and that is the whole point: `sync_axscript_exposure.py` walks imports
#: statically, and a function-level `from . import` is still an edge, so every architecture
#: became part of the mirror closure of any ONE model that reached this module. Promoting a
#: single Qwen3-VL card then asked approval for `models/lfm2`, `models/llama` and
#: `models/qwen3_5`, implementations belonging to cards that had not been promoted at all.
#: A dotted string is not an edge, so the closure narrows to what a model actually uses.
#:
#: `models/qwen3` is NOT in that accidental half and must stay: `qwen3vl/qwen3vl.py` imports
#: `..qwen3.qwen3`, and `qwen3vl/text_impl.py` imports `..qwen3.multicore_impl` and
#: `..qwen3.optimized_impl`, so a VLM's decoder tower IS a Qwen3 decoder.
#:
#: Each name is both the package and the reference module inside it (`lfm2/lfm2.py`), and the
#: MODULE is what gets imported: the package registers nothing, so a caller that wants a
#: device kernel calls `implementation.register_kernels` and never reaches torch.
_BUNDLED_ARCHITECTURES = (
    "domyn",
    "lfm2",
    "lfm2vl",
    "llama",
    "qwen3",
    "qwen3_5",
    "qwen3vl",
    "smolvlm2",
)


def register_architectures() -> None:
    """Import the bundled torch reference architectures once, so their registrations run.

    Deferred rather than done at package import: importing one architecture must not import
    the others, and the import is resolved against ``__package__`` so it still names whatever
    package this tree was vendored under.

    Registration happens as an import SIDE EFFECT: each reference module calls `register` at
    module scope, so nothing here needs the module objects and dropping the name bindings
    costs nothing. `tests/axscript/llm/test_architecture_registry.py` holds that: the registry
    must carry every name in `_BUNDLED_ARCHITECTURES` after this runs.
    """
    global _BUNDLED
    if _BUNDLED:
        return
    _BUNDLED = True
    import importlib

    for name in _BUNDLED_ARCHITECTURES:
        importlib.import_module(f"{__package__}.{name}.{name}")


def get_architecture(name: str) -> LLMArchitecture:
    register_architectures()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(f"unknown architecture {name!r}; registered: {sorted(_REGISTRY)}") from None

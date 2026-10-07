# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Serializable exported quantized model: the architecture-neutral bundle of
torch-namespace params, LUT ranges, and host params.

On disk it is a directory: tensors.safetensors (tensors under
group/path/field keys; 0-d tensors widened to (1,) and noted in the
manifest) and quantized_model.json (the manifest).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path

import numpy as np
import torch

# safetensors is imported inside `save` / `load` below rather than here: it is needed to MOVE
# a checkpoint, never to describe one, and at module scope it made the whole LLM toolkit
# unimportable without it -- this module sits on the
# `generate -> implementation -> quantized_model` path that every LLM test walks.

FORMAT_VERSION = 1

_MANIFEST = "quantized_model.json"
_TENSORS = "tensors.safetensors"


@dataclass
class QuantizedModel:
    """An exported quantized model: the torch-namespace params + what realises them."""

    architecture: str
    config: dict
    num_layers: int
    linear: dict  # {module_path: {field: tensor | scalar}}
    sdpa: dict  # {module_path: {field: tensor | scalar}}
    norm: dict  # {module_path: tensor}
    lut_ranges: dict  # {module_path: {"function", "min", "max"}}
    host_tensors: dict  # {name: tensor}  e.g. embed_weight, final_norm_weight
    host_scalars: dict  # {name: number}  e.g. rms_eps, head_dim, num_q_heads, rope_theta
    # {module_path: {field: value}} depthwise conv params: full-precision convs hold
    # {"weight": taps}; W8A8 QuantConv1d hold {w, qs, qz, ds, dz, ...}.
    conv: dict = field(default_factory=dict)
    # {module_path: {field: tensor}}  DeltaNet per-block params not covered by the
    # generic linear/sdpa/norm/conv exports: dt_bias, A_log, and the gated output
    # norm's scale (a GatedRMSNorm, not the shared RMSNorm).
    deltanet: dict = field(default_factory=dict)

    def save(self, path) -> Path:
        """Serialize to a directory (tensors.safetensors + quantized_model.json)."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        tensors: dict[str, torch.Tensor] = {}
        scalars: dict[str, object] = {}
        zero_dim: list[str] = []

        def put(key: str, value) -> None:
            if isinstance(value, torch.Tensor):
                t = value.detach().cpu().contiguous()
                if t.ndim == 0:  # safetensors needs >=1-d; widen + remember to undo on load
                    t = t.reshape(1)
                    zero_dim.append(key)
                tensors[key] = t
            else:
                scalars[key] = _to_py(value)

        for module_path, fields in self.linear.items():
            for field_name, value in fields.items():
                put(f"linear/{module_path}/{field_name}", value)
        for module_path, fields in self.sdpa.items():
            for field_name, value in fields.items():
                put(f"sdpa/{module_path}/{field_name}", value)
        for module_path, value in self.norm.items():
            put(f"norm/{module_path}", value)
        for module_path, fields in self.conv.items():
            for field_name, value in fields.items():
                put(f"conv/{module_path}/{field_name}", value)
        for module_path, fields in self.deltanet.items():
            for field_name, value in fields.items():
                put(f"deltanet/{module_path}/{field_name}", value)
        for name, value in self.host_tensors.items():
            put(f"host/{name}", _as_tensor(value))

        from safetensors.torch import save_file

        save_file(tensors, str(path / _TENSORS))
        manifest = {
            "format_version": FORMAT_VERSION,
            "architecture": self.architecture,
            "config": self.config,
            "num_layers": self.num_layers,
            "lut_ranges": self.lut_ranges,
            "host_scalars": {k: _to_py(v) for k, v in self.host_scalars.items()},
            "scalars": scalars,
            "zero_dim": zero_dim,
        }
        (path / _MANIFEST).write_text(json.dumps(manifest, indent=2))
        return path

    @classmethod
    def load(cls, path) -> "QuantizedModel":
        """Reload a bundle written by save."""
        path = Path(path)
        manifest = json.loads((path / _MANIFEST).read_text())
        version = manifest.get("format_version")
        if version != FORMAT_VERSION:
            raise ValueError(f"unsupported export format_version {version!r}")

        zero_dim = set(manifest.get("zero_dim", []))
        linear: dict = {}
        sdpa: dict = {}
        norm: dict = {}
        conv: dict = {}
        deltanet: dict = {}
        host_tensors: dict = {}

        def place(key: str, value) -> None:
            group, _, rest = key.partition("/")
            if group == "linear":
                module_path, _, field_name = rest.rpartition("/")
                linear.setdefault(module_path, {})[field_name] = value
            elif group == "sdpa":
                module_path, _, field_name = rest.rpartition("/")
                sdpa.setdefault(module_path, {})[field_name] = value
            elif group == "norm":
                norm[rest] = value
            elif group == "conv":
                module_path, _, field_name = rest.rpartition("/")
                conv.setdefault(module_path, {})[field_name] = value
            elif group == "deltanet":
                module_path, _, field_name = rest.rpartition("/")
                deltanet.setdefault(module_path, {})[field_name] = value
            elif group == "host":
                host_tensors[rest] = value
            else:
                raise ValueError(f"unknown tensor group in key {key!r}")

        from safetensors.torch import load_file

        for key, tensor in load_file(str(path / _TENSORS)).items():
            place(key, tensor.reshape(()) if key in zero_dim else tensor)
        for key, value in manifest.get("scalars", {}).items():
            place(key, value)

        return cls(
            architecture=manifest["architecture"],
            config=manifest["config"],
            num_layers=manifest["num_layers"],
            linear=linear,
            sdpa=sdpa,
            norm=norm,
            lut_ranges=manifest["lut_ranges"],
            host_tensors=host_tensors,
            host_scalars=manifest["host_scalars"],
            conv=conv,
            deltanet=deltanet,
        )


def architecture_conflict(bundle_dir: Path | str, requested: str) -> str | None:
    """Checks the pre-quantized bundle directory for the architecture key and returns an error if it does not match"""
    manifest = Path(bundle_dir) / _MANIFEST
    try:
        recorded = json.loads(manifest.read_text()).get("architecture")
    except (OSError, ValueError):
        return (
            f"{manifest} is not readable; point --quantized-model at a prior "
            f"build's `quantized_model/` directory."
        )
    if recorded is None:
        return f"{manifest} does not record an `architecture`."
    if recorded == requested:
        return None
    return (
        f"{bundle_dir} was quantized for {recorded!r}, not {requested!r}; it cannot be "
        f"loaded into this compile."
    )


def _to_py(value):
    """Coerce numpy scalars to native python for JSON."""
    return value.item() if isinstance(value, np.generic) else value


def _as_tensor(value) -> torch.Tensor:
    if isinstance(value, torch.Tensor):
        return value
    return torch.from_numpy(np.ascontiguousarray(value))

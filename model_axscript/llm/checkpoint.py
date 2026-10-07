# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Architecture-agnostic HuggingFace checkpoint loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

import torch


def repo_file(repo_id: str, filename: str, *, optional: bool = False) -> str | None:
    """`filename` from a local checkpoint DIRECTORY or from a hub repo.

    Everything in HuggingFace takes either spelling -- ``from_pretrained("Qwen/…")`` and
    ``from_pretrained("./my-finetune")`` are the same call -- so custom weights are a
    directory the caller points at, not a second code path. ``optional`` returns None
    instead of raising, for the files a checkpoint may legitimately not have (the shard
    index of a single-file checkpoint).
    """
    local = Path(repo_id)
    if local.is_dir():
        path = local / filename
        if path.exists():
            return str(path)
        if optional:
            return None
        raise FileNotFoundError(f"{filename} not found in {local}")

    from huggingface_hub import hf_hub_download
    from huggingface_hub.utils import EntryNotFoundError

    try:
        return hf_hub_download(repo_id, filename)
    except EntryNotFoundError:
        if optional:
            return None
        raise


def shard_files(repo_id: str) -> list[str]:
    """The checkpoint's safetensors shard filenames, in a stable order.

    A sharded checkpoint carries an index mapping each tensor to its shard; a
    single-file one (e.g. Qwen3-0.6B) has no index and is just ``model.safetensors``.
    Shared by both loaders below so the two agree on what a checkpoint contains.
    """
    index_path = repo_file(repo_id, "model.safetensors.index.json", optional=True)
    if index_path is None:
        return ["model.safetensors"]
    with open(index_path) as f:
        return sorted(set(json.load(f)["weight_map"].values()))


def load_hf_safetensors(
    repo_id: str,
    *,
    device: str | torch.device = "cpu",
    dtype: torch.dtype | None = torch.float32,
    key_filter: Callable[[str], bool] | None = None,
) -> dict[str, torch.Tensor]:
    """Merges the safetensors shards of a hub repo OR a local directory into one dict.

    ``dtype=None`` keeps each tensor as the checkpoint stores it (bf16 for the Qwen
    releases) instead of upcasting on the way in. That matters from 8B up: the caller
    already holds an fp32 model, so an fp32 staging dict doubles the peak to twice the
    model, which is what an ordinary host gets OOM-killed for. Copying bf16 into an fp32
    parameter widens it on write, and bf16 -> fp32 is exact, so the loaded weights are
    identical either way.

    key_filter, when given, is applied per-tensor as each shard loads: a name it
    rejects is never added to the returned dict. Filter here rather than on the
    result -- a caller that discards keys from the return value still pays for
    materializing every one of them first (e.g. a VLM checkpoint's vision/MTP
    towers alongside its text tower, which is what made this worth adding).

    This holds the WHOLE checkpoint in memory, so a caller that then loads it into an
    already-allocated model has two full copies resident. Past ~8B parameters use
    stream_hf_safetensors_into instead; see its docstring.
    """
    # Imports deferred so this module is importable without the optional deps.
    from safetensors.torch import load_file

    state_dict: dict[str, torch.Tensor] = {}
    for shard in shard_files(repo_id):
        shard_path = repo_file(repo_id, shard)
        for name, tensor in load_file(shard_path, device=str(device)).items():
            if key_filter is not None and not key_filter(name):
                continue
            state_dict[name] = tensor if dtype is None else tensor.to(dtype)
    return state_dict


@torch.no_grad()
def stream_hf_safetensors_into(
    model: torch.nn.Module,
    repo_id: str,
    *,
    dtype: torch.dtype = torch.float32,
) -> tuple[list[str], list[str]]:
    """Copies a checkpoint into an ALREADY-ALLOCATED model, one shard at a time.

    Returns ``(missing, unexpected)`` with the same meaning as
    ``load_state_dict(strict=False)``: model tensors no shard supplied, and checkpoint
    tensors the model has no slot for.

    The point is peak memory. ``load_hf_safetensors`` merges every shard into one dict
    and ``load_state_dict`` then copies that into the model, so both are resident at
    once -- 2x the parameters. At 9.8B in float32 that is ~79 GB, which does not fit in
    a 62 GB host; streaming caps the peak at the model plus ONE shard (~43 GB for the
    same model). Each tensor is copied straight into its destination with
    ``Tensor.copy_``, which does the dtype conversion in place, so no full-precision
    copy of the shard is materialised either.

    The model must be constructed normally (not on ``torch.device("meta")`` +
    ``to_empty()``): non-persistent buffers absent from the checkpoint -- the RoPE
    inverse frequencies, the observers' running min/max, the SmoothQuant scales -- get
    their values from ``__init__``, and ``to_empty`` would leave those uninitialised.
    """
    from safetensors.torch import load_file

    # state_dict (not named_parameters + named_buffers) so "missing" means exactly what
    # load_state_dict(strict=False) means: NON-PERSISTENT buffers are excluded, since a
    # checkpoint is never expected to carry them (the RoPE inverse frequencies, the
    # SmoothQuant scales, the RangeObservers' running min/max). keep_vars leaves the
    # entries as the live tensors, so copy_ writes through to the model.
    slots = dict(model.state_dict(keep_vars=True))
    seen: set[str] = set()
    unexpected: list[str] = []

    for shard in shard_files(repo_id):
        shard_path = repo_file(repo_id, shard)
        # device="cpu": the destination copy_ moves it wherever the model lives, and
        # staging the shard on an accelerator would defeat the point of streaming.
        shard_tensors = load_file(shard_path, device="cpu")
        for name, tensor in shard_tensors.items():
            target = slots.get(name)
            if target is None:
                unexpected.append(name)
                continue
            if tuple(target.shape) != tuple(tensor.shape):
                raise RuntimeError(
                    f"{name}: checkpoint shape {tuple(tensor.shape)} != model shape "
                    f"{tuple(target.shape)}"
                )
            target.copy_(tensor.to(dtype))
            seen.add(name)
        del shard_tensors  # free the shard before the next one is read

    missing = [name for name in slots if name not in seen]
    return missing, unexpected

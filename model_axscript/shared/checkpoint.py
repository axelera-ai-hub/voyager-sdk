# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Architecture-agnostic HuggingFace checkpoint loading helpers."""

from __future__ import annotations

import contextlib

import torch


def load_prefixed_state_dict(module, ckpt_path, prefix: str, *, strict: bool = False) -> bool:
    """Overlay a checkpoint's ``<prefix>*`` weights onto ``module`` (the prefix stripped).

    Pulls the ``prefix``-scoped keys out of a checkpoint (its ``state_dict`` sub-key or the dict
    itself) and loads them. Returns True when applied; False (a no-op) when ``ckpt_path`` is None
    or missing -- the pluggable-weights contract. Raises KeyError when the ckpt carries no
    ``<prefix>*`` keys at all.

    ``weights_only=True`` is pinned rather than left to the torch default: a bring-up checkpoint
    is an untrusted third-party file, and the default flipped across torch versions, so the safe
    behaviour must not depend on which torch the env resolved.
    """
    from pathlib import Path

    if ckpt_path is None or not Path(ckpt_path).is_file():
        return False
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt.get("state_dict", ckpt) if isinstance(ckpt, dict) else ckpt
    sub = {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
    if not sub:
        raise KeyError(f"no '{prefix}*' keys in {ckpt_path}")
    module.load_state_dict(sub, strict=strict)
    return True


@contextlib.contextmanager
def hf_access(repo_id: str):
    """Turn a gated / 401-403 Hugging Face load into an actionable message.

    Gated repos (e.g. google/gemma-3-1b-it) fail before any Axelera code runs; the
    raw error does not say what to do. Re-raise it as a login instruction. Wrap any
    HF ``from_pretrained`` / download so every family (LLM + vision) gives the same
    guidance for a user's own gated checkpoint.
    """
    from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

    try:
        yield
    except (GatedRepoError, HfHubHTTPError) as e:
        status = getattr(getattr(e, "response", None), "status_code", None)
        if isinstance(e, GatedRepoError) or status in (401, 403):
            raise RuntimeError(
                f"{repo_id} is a gated Hugging Face repo. Run `huggingface-cli login` with a "
                f"token from an account that has accepted the model's license on "
                f"https://huggingface.co/{repo_id}, then retry."
            ) from e
        raise

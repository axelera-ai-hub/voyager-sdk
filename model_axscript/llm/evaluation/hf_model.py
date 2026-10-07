# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""The eval-only glue over the HuggingFace adapter, plus the fp32 reference in HF shape.

`AxeleraForCausalLM` here is not a second adapter: it IS the class `llm.huggingface.model_cls`
builds, re-exported under the name this module has always used. A model a customer loads and a
model this suite scores must be the same presentation, or the number reported here belongs to
something nobody runs. So the two paths and their limits are documented there, and what is left
in this file is the two things only an eval needs:

  * `make_hflm`, which wraps the adapter in lm_eval's `HFLM` and pins `max_length` to the
    artifact's compiled `context_len`.
  * `ReferenceForCausalLM`, the fp32 torch reference presented in the same shape, so the
    reference leg of an accuracy pair can be scored by the same harness.

Two limits worth repeating, because they bound what a harness may ask for: one sequence at a
time on the device (batches are looped, so `batch_size=1`), and a sequence longer than the
compiled `context_len` is left-truncated. Everything else about generation is transformers'
own, including which sampling parameters are honoured and which are refused by name.

Usage (Python):

    from model_axscript.llm.evaluation.hf_model import AxeleraForCausalLM, make_hflm
    model = AxeleraForCausalLM.from_artifact("build/qwen3-0.6b", repo_id="Qwen/Qwen3-0.6B")
    import lm_eval
    lm_eval.simple_evaluate(model=make_hflm("build/qwen3-0.6b", "Qwen/Qwen3-0.6B"),
                            tasks=["hellaswag"], limit=200)
"""

from __future__ import annotations

import logging

import torch
from torch import nn

# One copy of the config resolution, in the customer-facing HuggingFace module: an
# artifact that presents itself through the Auto classes needs exactly what this adapter
# needs, and two copies is how the two would come to disagree. Imported under the private
# name this module already used, so its call sites read unchanged.
from ..huggingface import text_config

logger = logging.getLogger(__name__)


# The adapter itself lives in `llm/huggingface.py`, the customer-facing module: an artifact
# presented to a customer and an artifact scored by this suite must be the SAME presentation,
# or the number this file reports belongs to a model the customer never runs. What stays here
# is the eval-only glue below.
def _model_cls():
    from ..huggingface import model_cls

    return model_cls()


def make_hflm(artifact_dir, repo_id: str | None = None, *, batch_size: int = 1, **kwargs):
    """Wrap the shim in lm_eval's HFLM so the full `--model hf` task path drives the device.

    max_length is pinned to the artifact's compiled context_len: HFLM otherwise reads it from the
    HF config (max_position_embeddings, e.g. 40960 for Qwen3) and would build rolling/truncation
    windows far larger than the compiled graph -- forward() would then silently left-truncate them
    to context_len, corrupting loglikelihood_rolling (wikitext) perplexity.
    """
    from lm_eval.models.huggingface import HFLM

    model = AxeleraForCausalLM.from_artifact(artifact_dir, repo_id, **kwargs)
    return HFLM(
        pretrained=model,
        tokenizer=model.tokenizer,
        batch_size=batch_size,
        max_length=model._context_len,
    )


def _build_reference_cls():
    """Define the fp-reference shim (imported lazily, like `_build_model_cls`)."""
    from transformers import PretrainedConfig, PreTrainedModel
    from transformers.modeling_outputs import CausalLMOutputWithPast

    class ReferenceForCausalLM(PreTrainedModel):
        """The fp32 reference in HF shape, for an architecture transformers cannot load.

        `AxeleraForCausalLM` presents the COMPILED artifact; this presents the torch
        reference the build calibrates against, so `axzoo eval --pipe torch` can score a
        model whose `model_type` the pinned transformers does not know. It wraps the tree's
        own `nn.Module` rather than reimplementing it: same weights, same numerics as the
        build's reference, which is what makes the fp column comparable with the device one.

        Log-likelihood only. `generate` is deliberately absent: the eval adapter refuses a
        generative task by name, and a half-working decode loop here would be a second
        implementation of one that already exists on the compiled path.
        """

        config_class = PretrainedConfig
        main_input_name = "input_ids"
        _supports_cache_class = False
        supports_gradient_checkpointing = False

        @classmethod
        def from_architecture(cls, architecture: str, repo_id: str, *, device: str = "cpu"):
            from ..models.architecture import get_architecture

            # fp32 and no fake-quant: this is the REFERENCE leg. `quantize_attention` is
            # what the build passes to observe attention for calibration, and turning it on
            # here would make the "fp32" number a fake-quant number.
            inner = get_architecture(architecture).load_pretrained(
                repo_id, dtype=torch.float32, device=device, quantize_attention=False
            )
            inner.eval()
            # The inner model's own config is the fallback source, so this works for an
            # architecture whose HF config transformers cannot parse either.
            model = cls(text_config(repo_id, fallback=inner.config))
            model._inner = inner
            model.eval()
            return model

        def forward(
            self, input_ids=None, attention_mask=None, labels=None, return_dict=None, **kwargs
        ):
            """Teacher-forced logits [batch, seq, vocab] from the reference module."""
            ids = input_ids if torch.is_tensor(input_ids) else torch.as_tensor(input_ids)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            logits = self._inner(ids)
            loss = None
            if labels is not None:
                vocab = logits.shape[-1]
                loss = nn.functional.cross_entropy(
                    logits[:, :-1, :].reshape(-1, vocab),
                    labels[:, 1:].reshape(-1).to(torch.long),
                    ignore_index=-100,
                )
            return CausalLMOutputWithPast(loss=loss, logits=logits, past_key_values=None)

        @property
        def device(self):  # the real parameters live on the wrapped module
            return next(self._inner.parameters()).device

    return ReferenceForCausalLM


# Built at import (None where transformers is unavailable, so the rest of the suite imports).
try:
    AxeleraForCausalLM = _model_cls()
    ReferenceForCausalLM = _build_reference_cls()
except ImportError:
    AxeleraForCausalLM = None
    ReferenceForCausalLM = None

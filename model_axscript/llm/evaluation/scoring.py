# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Accuracy-evaluation helpers for a decoder LLM: perplexity and fidelity to a
reference model (KL divergence, top-1 agreement, NLL gap)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def token_nll(model, input_ids: torch.Tensor, window: int = 1024) -> tuple[float, int]:
    """Summed teacher-forced NLL over input_ids ([1, T]).

    Processed in non-overlapping windows of at most window tokens. Returns
    (total_nll, num_predicted_tokens) so callers can pool across sequences
    before exponentiating.
    """
    total_nll, n_tokens = 0.0, 0
    seq_len = input_ids.shape[1]
    device = next(model.parameters()).device  # score where the model lives (cpu / cuda)
    for start in range(0, seq_len, window):
        chunk = input_ids[:, start : start + window].to(device)
        if chunk.shape[1] < 2:
            continue
        logits = model(chunk)
        shift_logits = logits[:, :-1, :]
        shift_labels = chunk[:, 1:]
        nll = F.cross_entropy(
            shift_logits.reshape(-1, shift_logits.shape[-1]),
            shift_labels.reshape(-1),
            reduction="sum",
        )
        total_nll += nll.item()
        n_tokens += shift_labels.numel()
    return total_nll, n_tokens


def perplexity(model, input_ids: torch.Tensor, window: int = 1024) -> float:
    """Teacher-forced perplexity exp(mean NLL per token) over input_ids."""
    total_nll, n_tokens = token_nll(model, input_ids, window=window)
    if n_tokens == 0:
        raise ValueError("need at least 2 tokens to measure perplexity")
    return math.exp(total_nll / n_tokens)


@torch.no_grad()
def fidelity(reference_logits: torch.Tensor, test_logits: torch.Tensor) -> dict:
    """Distributional fidelity of test logits to reference logits.

    Both tensors are [..., vocab]. Returns mean KL(reference || test) (nats),
    top-1 agreement rate, and the mean next-token NLL gap on the reference's argmax.
    """
    ref_logp = F.log_softmax(reference_logits, dim=-1)
    test_logp = F.log_softmax(test_logits, dim=-1)
    ref_p = ref_logp.exp()

    kl = (ref_p * (ref_logp - test_logp)).sum(-1).mean().item()
    top1 = (reference_logits.argmax(-1) == test_logits.argmax(-1)).float().mean().item()

    ref_argmax = reference_logits.argmax(-1, keepdim=True)
    nll_gap = (
        ((-test_logp.gather(-1, ref_argmax)) - (-ref_logp.gather(-1, ref_argmax))).mean().item()
    )

    return {"kl": kl, "top1_agreement": top1, "nll_gap": nll_gap}

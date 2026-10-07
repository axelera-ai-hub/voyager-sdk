# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""What a decoder's DDR footprint does as `context_len` grows.

`context_len` is not a free knob: two device buffers scale with it, and one of them scales
QUADRATICALLY, which is why raising it fails at prefill rather than degrading. Both come
straight off the layer kernel's buffer declarations, so this is arithmetic rather than a
guess, and it can be evaluated before the multi-minute compile instead of discovered as a
device hang minutes later.

The two terms, per the kernel, and they do NOT scale the same way with layer count:

    k_cache / v_cache : (1, num_kv_heads, context_len, head_dim_pw) int8x64
                        -> num_kv_heads * context_len * padded_head_dim bytes, x2, PER LAYER.
                           LINEAR in context_len, and lives for the whole generation.
                           Allocated once per layer, so the model total is x num_layers.

    attn_mask         : (1, num_q_heads, seq_len, context_len // 64) float32x64
                        -> num_q_heads * seq_len * context_len * 4 bytes, ONCE PER MODEL.
                           seq_len == context_len during PREFILL (it is 1 when decoding),
                           so this is QUADRATIC in context_len on the prefill graph.
                           It is a `shared_input`: every layer's KernelCall binds the same
                           tensor name, so the compiler allocates ONE buffer for the whole
                           model, not one per layer. Multiplying this term by num_layers
                           over-counts by that factor.

Which term leads therefore depends on the size, and the crossover is real. Setting the two
equal gives it in closed form:

    num_q_heads * ctx^2 * 4  ==  num_layers * 2 * num_kv_heads * ctx * padded_head_dim

For Qwen2.5-0.5B (24 layers, 2 KV heads, 14 Q heads, padded head dim 128) that is ctx 219.4.
Below it the CACHE total leads (at 128: mask 0.9 MiB against 1.5 MiB of cache across all
layers, 0.58x); above it the mask does, and pulls away quadratically (2.33x at 512, 4.67x at
1024, 9.33x at 2048, where 224 MiB of mask sits against 24 MiB of cache). So the mask wins at
every size that matters even though it does not win at the smallest. That is why a too-large
context dies at prefill specifically, and why `phases=("decode",)` exists - it skips that
buffer entirely, trading prefill throughput for the headroom.

Comparing the mask against ONE layer's cache instead of the model's total is what makes it
look like a fixed 14x-to-224x dominance. Those are per-layer ratios; the whole-model ones
are 0.58x and 9.3x.

These are the context-dependent terms ONLY. Weights, activations and allocator overhead are
not modelled, so a total here would be a fiction - compare the terms across context lengths,
do not read the sum as a device footprint.
"""

from __future__ import annotations

from dataclasses import dataclass

PWORD = 64
_MIB = 1024 * 1024


def padded_head_dim(head_dim: int) -> int:
    """Physical head dim after the design-P padding the kernels apply.

    Mirrors `_design_p_dims`: the half-dim rope freqs occupy whole PWORD blocks and the head
    is padded to twice that. A no-op when head_dim is already a multiple of 2*PWORD.
    """
    rope_dim_pw = (head_dim // 2 + PWORD - 1) // PWORD
    return 2 * rope_dim_pw * PWORD


@dataclass(frozen=True)
class ContextMemory:
    """DDR bytes that depend on `context_len`, split by term.

    The terms are NOT in the same unit, so read the field docs before summing them:
    the caches are per layer and the masks are per model. `summary()` does the scaling.
    """

    context_len: int
    num_layers: int
    kv_cache_bytes: int
    """K and V for ONE layer. Model total is this x `num_layers`."""
    prefill_mask_bytes: int
    """The whole model's prefill mask. A `shared_input`, allocated once; do NOT x num_layers."""
    decode_mask_bytes: int
    """The whole model's decode mask. Shared like `prefill_mask_bytes`."""

    @property
    def kv_cache_mib(self) -> float:
        return self.kv_cache_bytes / _MIB

    @property
    def prefill_mask_mib(self) -> float:
        return self.prefill_mask_bytes / _MIB

    def summary(self) -> str:
        """One line for a build log or a report row.

        Takes no layer count: it is already fixed by the estimate, and passing it again let a
        caller report a 24-layer model's cache as a 48-layer total.
        """
        return (
            f"context_len={self.context_len}: "
            f"kv cache {self.kv_cache_mib * self.num_layers:.1f} MiB over "
            f"{self.num_layers} layers, "
            f"prefill attn mask {self.prefill_mask_mib:.1f} MiB shared by all layers "
            f"(quadratic; {self.decode_mask_bytes / 1024:.1f} KiB when decoding)"
        )


def context_memory(
    context_len: int,
    *,
    num_layers: int,
    num_kv_heads: int,
    num_q_heads: int,
    head_dim: int,
) -> ContextMemory:
    """The context-dependent DDR terms at `context_len`.

    Caches come out per layer, masks per model; see `ContextMemory`.
    """
    if context_len < 1:
        raise ValueError(f"context_len must be >= 1 (got {context_len})")
    phys_head_dim = padded_head_dim(head_dim)
    kv = 2 * num_kv_heads * context_len * phys_head_dim  # K and V
    return ContextMemory(
        context_len=context_len,
        num_layers=num_layers,
        kv_cache_bytes=kv,
        # float32 mask, seq_len == context_len while prefilling.
        prefill_mask_bytes=num_q_heads * context_len * context_len * 4,
        decode_mask_bytes=num_q_heads * 1 * context_len * 4,
    )

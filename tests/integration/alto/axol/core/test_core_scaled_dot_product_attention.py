# Copyright Axelera AI, 2026
"""Vendored numpy references from the ALTO core AXOL SDPA test.

Source: axelera-ai/host.compiler.alto,
axelera_compiler/tests/integration/alto/axol/core/test_core_scaled_dot_product_attention.py,
revision ca88e4d224b63a7b00b074715810d0eea0c54a2a.

That file tests the core (non-LLM) `scaled_dot_product_attention` AXOL op and stays in
ALTO -- it was not part of this repo's LLM-op migration. test_sdpa_flash.py only reuses
its causal mask builder and fakequant reference, so only those two functions are vendored
here (no SDPAConfig / make_sdpa_module / device-parametrized tests).
"""

from axelera.compiler.alto.compiler.device import PWORD
import numpy as np
import numpy.typing as npt


def _default_causal_mask(
    batch_size: int,
    num_heads: int,
    n_queries: int,
    n_keys: int,
) -> npt.NDArray[np.int8]:
    """Causal mask with PWORD padding, shape ``(batch, heads, n_queries, ctx_blocks, PWORD)``.

    int8 additive mask: 0 (visible) / -127 (masked). -127 is exp-negligible, and int8
    halves the mask's DDR/L1 footprint vs float16. The kernel casts it to float18.
    """
    n_keys_padded = ((n_keys + PWORD - 1) // PWORD) * PWORD
    query_pos = np.arange(n_keys - n_queries, n_keys)[:, None]
    key_pos = np.arange(n_keys_padded)[None, :]
    causal = key_pos > query_pos
    padding = key_pos >= n_keys
    m = causal | padding
    return (
        np.where(m, -127, 0)
        .astype("int8")
        .reshape(1, 1, n_queries, n_keys_padded // PWORD, PWORD)
        .repeat(num_heads, axis=1)
        .repeat(batch_size, axis=0)
    )


def quantized_sdpa_numpy_reference(
    query: npt.NDArray[np.int8],
    key: npt.NDArray[np.int8],
    value: npt.NDArray[np.int8],
    mask: npt.NDArray[np.float16],
    scores_dq_scale: float,
    scores_q_scale: float,
    scale: float = 1.0,
    attention_output_dequantizer_scale: float = 1.0,
) -> npt.NDArray[np.float32]:
    """Fakequant numpy SDPA reference (exact math, quantised weights)."""
    batch_size, num_heads, seq_len_q, head_dim_blocks, pword = query.shape
    _, num_kv_heads, context_len, _, _ = key.shape
    _, _, _, value_dim_blocks, _ = value.shape
    kv_groups = num_heads // num_kv_heads

    query_flat = query.reshape(batch_size, num_heads, seq_len_q, -1)
    key_flat = key.reshape(batch_size, num_kv_heads, context_len, -1)
    value_flat = value.reshape(batch_size, num_kv_heads, context_len, -1)
    mask_flat = mask.reshape(batch_size, num_heads, seq_len_q, context_len).astype(np.float32)

    key_flat = np.repeat(key_flat, kv_groups, axis=1)
    value_flat = np.repeat(value_flat, kv_groups, axis=1)

    qk_int32 = query_flat.astype(np.int32) @ key_flat.astype(np.int32).transpose(0, 1, 3, 2)
    scores = qk_int32.astype(np.float64) * np.float64(scores_dq_scale) * np.float64(scale)
    scores = scores + mask_flat

    max_scores = np.max(scores, axis=-1, keepdims=True)
    exp_scores = np.exp(scores - max_scores)
    sum_exp = np.sum(exp_scores, axis=-1, keepdims=True)

    scaled_exp = exp_scores * np.float64(scores_q_scale)
    quantized_weights = np.clip(np.round(scaled_exp), -128, 127).astype(np.int8)

    output = quantized_weights.astype(np.int32) @ value_flat.astype(np.int32)
    output = output.astype(np.float64) * np.float64(attention_output_dequantizer_scale) / sum_exp

    return output.reshape(batch_size, num_heads, seq_len_q, value_dim_blocks, pword).astype(
        np.float32
    )

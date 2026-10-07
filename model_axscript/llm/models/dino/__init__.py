# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""DINOv2-style vision-transformer encoder block (POC vision path).

A bidirectional, single-pass ViT block that reuses the LLM device op stack
(int8 matmul + dequant, bidirectional SDPA, LayerNorm, GELU LUT) without any of
the autoregressive-decoder machinery (no KV cache, no causal mask, no RoPE, no
token embedding / lm_head). Scoped to a single encoder block validated
single-shot on AICORE_SIM against a torch W8A8 fake-quant reference.

reference.py      -- the torch fake-quant reference block + config + encoder stack.
optimized_impl.py -- the fused alto-script block kernel + per-buffer param binding.
hf.py             -- load real DINOv2 weights (one layer) from a HF checkpoint.
calibration.py    -- real-image calibration set (the vision analog of llm/calibration.py).
model.py          -- full-model inference: the WHOLE model (patch-embed conv + N blocks +
                     final LayerNorm) authored as ONE graph (LLM graph mechanism) and run
                     in a single device dispatch: pixels -> last_hidden_state. Host does
                     only image normalization, an int8 pixel quantize+pad, and an output slice.
"""

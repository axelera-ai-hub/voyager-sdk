# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5: the PyTorch reference architecture (text tower) plus its device impls.

The kernels are the hybrid gated-DeltaNet / gated full-attention stack in
`.deltanet_impl` and `.europa_multicore_impl`.

Registration is NOT an import side effect of this package; see the note in
`models.qwen3`. Import the module that owns what you want:
`from ...models.qwen3_5.qwen3_5 import Qwen3_5ForCausalLM`.
"""

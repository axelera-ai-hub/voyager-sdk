# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""LFM2: the PyTorch reference architecture plus its Metis implementation.

The short-conv linear-mixer blocks are still full precision by default; see lfm2.py.

Registration is NOT an import side effect of this package; see the note in
`models.qwen3`. Import the module that owns what you want:
`from ...models.lfm2.lfm2 import Lfm2ForCausalLM`.
"""

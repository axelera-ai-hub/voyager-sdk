# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Domyn Small (Nemotron): the PyTorch reference architecture plus its Metis kernel.

Registration is NOT an import side effect of this package; see the note in
`models.qwen3`. The reference model (`.domyn`, which needs torch) and the device
kernel (`.metis_impl`, which does not) are registered by
`models.architecture.register_architectures` and `implementation.register_kernels`.
Import the module that owns what you want:
`from ...models.domyn.domyn import DomynForCausalLM`.
"""

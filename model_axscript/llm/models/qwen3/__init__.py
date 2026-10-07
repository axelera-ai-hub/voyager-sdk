# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Qwen3: the PyTorch reference architecture plus its Metis implementations.

Registration is NOT an import side effect of this package. The reference model
(`.qwen3`, which needs torch) and the device kernels (`.naive_impl`,
`.optimized_impl`, `.multicore_impl`, which do not) are registered by
`models.architecture.register_architectures` and
`implementation.register_kernels`, so resolving a kernel by name on a machine with
no torch installed does not drag the reference in. Import the module that owns what
you want: `from ...models.qwen3.qwen3 import Qwen3ForCausalLM`.
"""

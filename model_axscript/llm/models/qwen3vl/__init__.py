# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL: the PyTorch reference architecture plus its Metis implementations.

Everything is imported from the module that owns it: `.qwen3vl` for the reference,
`.pipeline` for the host path, `.text_impl` for the decoder kernels, `.vision_impl` for
the tower. The vision tower is a graph of its own, driven by
pipeline.Qwen3VLPipeline, not an LLM layer implementation.

Registration is NOT an import side effect of this package; see the note in
`models.qwen3`.
"""

# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""LFM2.5-VL: the PyTorch reference architecture plus its Metis implementation.

Everything is imported from the module that owns it: `.lfm2vl` for the reference,
`.pipeline` for the host path, `.text_impl` for the decoder kernel, `.vision_impl` for
the SigLIP2 tower. The vision tower is a graph of its own, driven by
pipeline.Lfm2VlPipeline, not an LLM layer implementation.

Registration is NOT an import side effect of this package; see the note in
`models.qwen3`.
"""

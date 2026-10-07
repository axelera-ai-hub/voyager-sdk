# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Concrete decoder-LLM architectures + their Metis implementations.

Registration is NOT an import side effect of this package. It used to be, and the cost was
paid by anyone importing a sibling: a vision card reaches `models.dino`, which executes this
file, which pulled in every decoder LLM and its implementation. Call `architecture.register_architectures()` for the torch references or
`implementation.register_kernels()` for the device kernels; the two registry lookups
already do, and only the first one needs torch.
"""

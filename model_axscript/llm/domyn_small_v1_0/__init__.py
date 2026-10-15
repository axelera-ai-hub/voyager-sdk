# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Domyn-Small-v1.0 on the AxScript path: the build+run config for this model.

The model code lives in ``model_axscript/llm/models/domyn/`` and the toolkit in
``model_axscript/shared/``; this package holds only the per-model configuration
that ``__main__.py`` drives.

`domyn/Domyn-Small-v1.0` is a 9.82B `NemotronForCausalLM`, and the three traits that
make it its own architecture rather than a Llama re-skin are LayerNorm1P (mean
subtracted, bias added, gain stored as `weight + 1`), a squared-ReLU MLP with no gate,
and partial RoPE over the leading 64 of each head's 128 channels. See
``models/domyn/metis_impl.py`` for how each is realised without a new compiler
primitive.

The checkpoint is ungated but LARGE: 9.815B parameters, and the reference is built in
float32, so a build needs ~45 GB of host RAM to load it and ~52 GB through calibration.
The load streams shard by shard (``checkpoint.stream_hf_safetensors_into``) rather than
merging the checkpoint into one state dict, which would need twice that.
"""

from pathlib import Path

MODEL_TAG = "domyn-small-v1.0"


def make_config(artifact_dir, cal_device: str = ""):
    """Per-model build+run config (implementation, checkpoint, context length).

    Used by ``__main__.py`` for both build and run. ``cal_device`` is a compile-only
    knob (the torch device used for calibration); the runtime path ignores it.
    """
    from ..shared.config import GenerationConfig

    return GenerationConfig(
        implementation="domyn-metis",
        repo_id="domyn/Domyn-Small-v1.0",
        artifact_dir=Path(artifact_dir),
        # The configuration this architecture has been built and run at: a 2048-token
        # context, prefill ingested in 512-token chunks, sharded over 4 AI cores. The
        # artifact is 7.72 GiB of device memory pools (160 MiB of it KV cache) plus 4.0 GB
        # of host params -- the fp16 embedding and the untied lm_head -- so it needs a 16GB
        # part, and a 4-core build reserves 4 sub-devices.
        context_len=2048,
        # Each size here is a claim it was RUN on a Metis part in this tree, and each was:
        # measured on metis-0:1:0 (16 GiB, 4 cores) and carried by a row in
        # `axelera_zoo/reports/llm/domyn_small_v1_0.yaml`. 2048 was measured first, from the
        # published ctx1024 artifact's `quantized_model/` recompiled at each size with
        # `--reuse-quant` -- calibration is what needs ~45 GB of host RAM, and reusing the
        # bundle skips it, so a size costs ~6 minutes of codegen rather than a host nobody
        # has. 1024 and 4096 were added the same way: declared, built, run on chip, and kept
        # only because the chip took them. Widening this set is still an on-chip measurement
        # and never an edit to make a build pass -- a size that fails comes back out.
        #
        # 128, 256 and 512 are declared too, so a build may be targeted at them
        # (`--seq-len 128`); they are the cheap end of the range, strictly below the smallest
        # size measured here, and the compiled footprint falls with context length. NO on-chip
        # row exists for them yet -- `axzoo report` has not swept them -- so they are carried in
        # the pending-measurement list in `axelera_zoo/tests/build/test_model_reports.py` and
        # owe a row, exactly as the three above earned theirs.
        supported_context_lens=(128, 256, 512, 1024, 2048, 4096),
        # One chunk per declared size, positionally aligned with the set above. A prefill graph
        # is compiled for `chunk_len` query positions, so the 512-wide chunk the three measured
        # sizes run at is not merely slow at a 128-token context, it is wider than the whole KV
        # cache -- which is why the width follows the size rather than being one number. The
        # small sizes take half their context, which is what keeps the CHUNKED path exercised
        # rather than degenerating to a single full-width prefill; 64 at ctx128 is also the
        # width this architecture's tier-0 entry `domyn-small-sim-smoke` already declares.
        # 1024/2048/4096 KEEP 512: that is what the committed report rows were measured at, and
        # the gates compare `chunk_len` field by field.
        chunk_lens=(64, 128, 256, 512, 512, 512),
        # 512, not the 1024 the other 8B-class cards use: at hidden 4096 / intermediate
        # 16384 the prefill kernel already sits at 3.85 MiB of the 4 MiB L1 at chunk 512.
        chunk_len=512,
        # Decode is DDR-bandwidth bound -- the stack reads 7.72 GiB of int8 weights per
        # token however many cores read it -- so the cores pay for prefill (a clean 2.0x)
        # and add only ~27% to decode.
        num_cores=4,
        cal_device=cal_device,
    )

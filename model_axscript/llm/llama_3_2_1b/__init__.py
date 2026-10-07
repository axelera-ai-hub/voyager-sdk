# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Llama-3.2-1B on the AxScript path: the build+run config for this model.

The model code lives in ``model_axscript/llm/models/llama/`` and the toolkit in
``model_axscript/shared/``; this package holds only the per-model configuration
that ``__main__.py`` drives.

The checkpoint is the ungated `unsloth/` re-upload rather than `meta-llama/`. The
weights are byte-identical (same safetensors size on every shard); unsloth fills in
metadata Meta leaves unset, notably `pad_token_id`, and needs no licence acceptance,
so CI and the calibration host can fetch it without a gated-repo token. It is also
the checkpoint the committed golden was measured from.
"""

from pathlib import Path

MODEL_TAG = "llama-3.2-1b"


def make_config(artifact_dir, cal_device: str = ""):
    """Per-model build+run config (implementation, checkpoint, context length).

    Used by ``__main__.py`` for both build and run. ``cal_device`` is a compile-only
    knob (the torch device used for calibration); the runtime path ignores it.
    """
    from ..shared.config import GenerationConfig

    return GenerationConfig(
        implementation="llama-metis-multicore",
        # NOTE: the committed golden llama3-1b was produced from the BASE model,
        # while this model's axgraph sibling uses Llama-3.2-1B-Instruct and the 3B/8B
        # AxScript cards are both Instruct. Matching the golden keeps the tolerance
        # comparison valid; switching to Instruct would void it.
        repo_id="unsloth/Llama-3.2-1B",
        artifact_dir=Path(artifact_dir),
        # The default build: a 2048-token context, prefill ingested in 1024-token chunks,
        # sharded over 4 AI cores. The committed golden was measured at 4096 rather than this
        # size -- see the note under `context_len` below. A 4-core build reserves 4
        # sub-devices, so it needs a part that has them. Calibration deliberately differs: the gate
        # uses a 64-sample budget to keep a nightly run short, where a shipped build takes the full
        # mix.
        context_len=2048,
        # 1024 and 2048 are declared so `axzoo report` measures this model at those three
        # sizes: the verb sweeps the DECLARED set and never widens it, so a size absent from
        # here is a size no report row can exist for. `context_len` is 2048: the size a
        # build gets when nobody passes `--seq-len`, chosen as the useful middle of the range
        # rather than its top, since the prefill mask grows with the SQUARE of it.
        #
        # THE GOLDEN GATE STAYS AT 4096, deliberately and in writing. The committed golden
        # was measured there, so this model's entry in `llm/regression/manifest.toml` PINS
        # `[entry.overrides] context_len = 4096`: a `model_tag` entry otherwise resolves the
        # size from this card on every run, and the gate would build 2048 and diff it against
        # a 4096 golden -- a run that cannot pass and costs a device. `[entry.measured]` was
        # NOT edited to say 2048; that would claim a re-measurement nobody performed.
        #
        # So the nightly DRIFT baseline sits at 4096 while the default a user gets is 2048.
        # That is the one thing conceded, and it is not a gap in evidence: 2048 carries its own
        # successful on-chip row with an accuracy pair in `axelera_zoo/reports/llm/`, which
        # `test_shipped_config_matches_the_golden` re-checks before it allows the two to
        # differ. Establishing a 2048 golden retires the pin; that test fails if the golden
        # catches up and the pin is left behind.
        #
        # MEASURED ON CHIP at all three sizes on 2026-09-06, on metis-0:1:0
        # over 4 cores: `axzoo report llama-3.2-1b --backend axscript`. One row per
        # size, each carrying its own accuracy pair, in
        # `axelera_zoo/reports/llm/llama_3_2_1b.yaml`. The declared set and the measured
        # evidence agree at those three sizes, which is what
        # `test_declared_run_sizes_match_the_measured_evidence` checks.
        #
        # 128, 256 and 512 are declared too, so a build may be targeted at them
        # (`--seq-len 128`); they are the cheap end of the range, strictly below the smallest
        # size measured here. NO on-chip row exists for them yet -- `axzoo report` has not
        # swept them -- so they are carried in the pending-measurement list in
        # `axelera_zoo/tests/build/test_model_reports.py` and owe a row, exactly as the three
        # above earned theirs.
        supported_context_lens=(128, 256, 512, 1024, 2048, 4096),
        # One chunk per declared size, positionally aligned with the set above. A prefill
        # graph is compiled for `chunk_len` query positions, so the 1024-wide chunk the three
        # measured sizes run at is not merely slow at a 128-token context, it is wider than
        # the whole KV cache -- which is why the width has to follow the size rather than be
        # one number. 1024/2048/4096 KEEP 1024: that is what the committed golden and the
        # report rows were measured at, and both gates compare `chunk_len` field by field.
        chunk_lens=(64, 128, 256, 1024, 1024, 1024),
        num_cores=4,
        cal_device=cal_device,
    )

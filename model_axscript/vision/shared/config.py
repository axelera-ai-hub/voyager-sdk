# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Build + run config for one vision-encoder artifact - the vision analogue of the LLM
GenerationConfig. A single-graph feature extractor has no autoregressive-decoder knobs
(context_len / phases / max_new_tokens / temperature / stop_token_ids / tokenizer), so the
config carries only the encoder + calibration parameters."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from model_axscript.shared.hardware import default_cal_device

from .source import EncoderSource

# 30k diverse natural images at ~454px median, downscaled (not upscaled) to the input res --
# the COCO calibration default (see calibration.py for why tiny-imagenet under-covers @high-res).
_DEFAULT_CALIBRATION_DATASET = "sayakpaul/coco-30-val-2014"


@dataclass
class EncoderConfig:
    """Build + run config for one DINOv2-style encoder artifact (defaults: dinov2-vits14)."""

    # Which module builds and loads the device graph, and the label written into
    # pipeline.json. The prefix names the model family, the suffix names who implements the
    # graph: "-metis" is the in-tree DINOv2 recipe, "-axten" is the compiler's own ViT offload.
    # `shared/builder.py` holds the enumeration and resolves it.
    implementation: str = "dinov2-metis"
    repo_id: str = "facebook/dinov2-small"  # identity for naming / meta.json (== source.name)
    # Where the fp weights come from. None => the per-model default (HFSource(repo_id));
    # set it to any EncoderSource (torchvision / local checkpoint / SoT) to decouple from HF.
    source: EncoderSource | None = None
    target: str = "device"  # "device" (real Metis) or "sim" (AICORE_SIM)
    # Torch device for the reference-model load + PTQ CALIBRATION (build time only, not the
    # Metis run device). "" is resolved in __post_init__ ("cuda" when a GPU is present, else
    # "cpu"), so every consumer sees a concrete device. The ALTO compile + export always run
    # on CPU.
    cal_device: str = ""
    # Reference-load dtype. "float32" is the default (ViT-small/base/large fit easily in fp32).
    load_dtype: str = "float32"
    artifact_dir: Path = Path("build/dinov2-vits14")  # holds model.json + meta.json + blobs
    # Square input side; None keeps the HF processor default (224 crop). 518 is native DINOv2.
    image_size: int | None = None
    # The input sides this model has been measured at on chip. The EFFECTIVE size (after None
    # resolves through the processor) must be one of them - see `shared.artifact_contract.validate_run_size`.
    # 518 is not here yet: at native resolution the LayerNorm inv_sqrt LUT saturates.
    supported_image_sizes: tuple[int, ...] = (224,)
    # AI cores ONE inference is split across (a latency shard, not a batch or a replica
    # count). This is a build-time decision: the runtime reserves whatever the build spans,
    # derived from `call_graphs` (`shared.runtime.required_cores`), so it is never re-picked
    # at run time. 4 is the default because it is the shipped shard and what the eval
    # harnesses measure - leaving it at 1 here made `__main__.py --benchmark` and `knn.py`
    # report different numbers for the same model. Drop to 1 to measure the single-core
    # baseline.
    shard: int = 4
    n_calib: int = 32  # real images for min/max + SmoothQuant observers (~32-256 is plenty)
    calibration_dataset: str = _DEFAULT_CALIBRATION_DATASET
    # Apply the encoder's final LayerNorm (default). False bypasses it (nn.Identity on the
    # reference; the final_norm kernel is dropped from the device graph) -- for downstream heads
    # (e.g. DINO-Mix MixVPR) that consume pre-final-norm patch features.
    final_norm: bool = True
    # The output head BAKED into the device graph, and the ONE field that says a card has one:
    # a backbone size the publisher ships a head for ("vits14" | "vitb14" | "vitl14"; see
    # `shared.seg_head.SIZES` / `shared.dinov2_depth.SIZES`), or a path to your own head file,
    # which is how a customer runs their own label set on the same backbone. Which of the two it
    # is, is decided by the filesystem. The class count is read from whichever file arrives and
    # is never declared here. None keeps the encoder headless, so the graph emits features, and
    # it is also what makes `--head` refuse: a card with no head has nothing for it to replace.
    # With a head the graph emits per-token class logits, so it is the same encoder with a
    # different output, not a different model. A source that ASSEMBLES the head with the
    # backbone keeps its own copy and `EncoderSource.with_head` moves the two together.
    head: str | None = None
    smooth_alpha: float | None = 0.5  # SmoothQuant migration strength; None disables it
    calib_method: str = "minmax"  # "minmax" | "percentile"
    percentile: float = 99.99  # only used when calib_method == "percentile"

    def __post_init__(self) -> None:
        if not self.cal_device:
            self.cal_device = default_cal_device()


#: Configuration fields the artifact key leaves out: where the artifact goes, and which torch
#: device ran the calibration. The key is built by EXCLUSION so a field added to `EncoderConfig`
#: later is keyed by DEFAULT rather than silently left out, which is the defect that let one
#: build be reported as another.
#:
#: `artifact_dir` cannot change what is compiled. `cal_device` is an accepted approximation
#: rather than the same claim: a CUDA forward and a CPU forward differ in the last bits, and
#: calibration fits the int8 scales from what it observed, so the two can produce artifacts
#: that are not byte-identical. They are excluded together because a cached artifact is reused
#: across both, which is the prior behaviour and a deliberate trade, and an accuracy difference
#: between them is a rebuild away rather than a defect.
_NOT_IN_THE_ARTIFACT = ("artifact_dir", "cal_device")


def artifact_fields(config) -> list:
    """Everything about `config` that decides what gets compiled, as comparable text.

    Both places that key an artifact directory read this: the front door, which keeps a
    customer's build off the shipped build's path, and `imported_dino.keyed_artifact_dir`,
    which caches a build the eval path made. Two lists would be two answers to one question,
    and the weaker of the two decides whether a build is reported as another.
    """
    return sorted(
        (field.name, repr(getattr(config, field.name)))
        for field in dataclasses.fields(config)
        if field.name not in _NOT_IN_THE_ARTIFACT
    )

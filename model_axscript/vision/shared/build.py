# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""The pieces every vision build and run shares: the source seam, the artifact files it
writes beside the compiled graph, the host preprocessing, and the timing loop.

The build itself lives in `imported_dino.py`, because what authors the device graph is the
DINOv2 implementation under `model_axscript/llm/models/dino/`. What is here is everything
around that graph: `meta.json` and the runtime-loadable contract artifact (`pipeline.json` +
`host_params.npz`), which is what lets a build be reloaded and run without re-calibrating or
re-compiling.
"""

from __future__ import annotations

import json
from pathlib import Path
from time import perf_counter

import numpy as np


def _source(config):
    """The config's EncoderSource. A configuration that names none is an error, not a default.

    It used to fall back to `HFSource(config.repo_id)`, which made every card that forgot to
    declare a source silently a HuggingFace card, and left the upstream invisible to anyone
    reading the card instead of the recipe.
    """
    if config.source is None:
        raise ValueError(
            f"{config.repo_id or 'this configuration'} names no EncoderSource. Set "
            f"`source=` in the model's `make_config` (HFSource, TimmViTSource, or your own "
            f"implementation of the seam in `vision/shared/source.py`)."
        )
    return config.source


def _write_meta(
    model, block_config, config, specs, artifact_dir, seq_real, *, seg_num_classes=None
) -> None:
    """Persist the scalars the host front-end needs to run the artifact (pixel int8 quant scale,
    padding-mask shape, geometry) so ``DinoV2Device.load`` can run it without re-calibrating or
    re-compiling. Mirrors ``DinoV2Device._write_meta`` but stays device-free at build time.

    Two configs, and the difference matters: the shapes come from the BLOCK config the source
    derived, while `repo_id` and `final_norm` are build intent and live on the ENCODER config.
    Reading either from the other is what made this raise, so both are named parameters.

    `seg_num_classes` is the class count of a segmentation head baked into the graph, and it
    is what the loader reads to know the output is per-token LOGITS rather than features. It
    comes from the head that was actually baked, not from the config, because the config names
    a backbone size and the class count is read from the head file.
    """
    block_spec = specs["block"]
    seq_pad = int(block_config.seq_len)
    hidden = int(block_config.hidden_size)
    grid = int(round((seq_real - 1) ** 0.5))
    pix_scale = float(model.patch_embed.proj.act_observer.qparams()[0].reshape(-1)[0])
    meta = {
        "repo_id": config.repo_id,
        "image_size": int(block_config.image_size),
        "patch_size": int(block_config.patch_size),
        "seq_real": int(seq_real),
        "seq_pad": seq_pad,
        "hidden": hidden,
        "grid": grid,
        "pix_scale": pix_scale,
        "mask_shape": list(block_spec["attn_mask"][0]),
        "seg_head": seg_num_classes is not None,
        "seg_num_classes": seg_num_classes,
        "final_norm": bool(config.final_norm),
        **build_provenance(config),
    }
    (Path(artifact_dir) / "meta.json").write_text(json.dumps(meta, indent=2))


def write_host_params(artifact_dir, base: dict, extra: dict | None) -> None:
    """Save `host_params.npz`, refusing a source stage that would overwrite the build's own key.

    An `.npz` raises nothing when an array is unused, so a silent overwrite here surfaces much
    later as a wrong number rather than as an error. Both builders write the blob through this,
    so the refusal cannot hold on one path and not the other.
    """
    clash = set(extra or {}) & set(base)
    if clash:
        raise ValueError(
            f"source host_params would overwrite the build's own keys: {sorted(clash)}"
        )
    merged = dict(base)
    merged.update(extra or {})
    np.savez(Path(artifact_dir) / "host_params.npz", **merged)


def write_pipeline_json(
    artifact_dir, *, implementation: str, device_model_json: str, image_size: int, graphs
) -> None:
    """Save `pipeline.json`: the manifest `shared.artifact_contract` gates and the runtime loads.

    Each device graph becomes one device stage, so the contract's stages-within-graphs check
    holds. `image_size` is the COMPILED run size, recorded so the runtime can state what a
    number is comparable at without re-reading the build configuration.
    """
    pipeline = {
        "implementation": implementation,
        "host_params": "host_params.npz",
        "device_model_json": device_model_json,
        "image_size": int(image_size),
        "stages": [{"name": g, "placement": "device", "graphs": [g]} for g in graphs],
    }
    (Path(artifact_dir) / "pipeline.json").write_text(json.dumps(pipeline, indent=2))


def build_provenance(config) -> dict:
    """What this build was GIVEN, so an artifact says whose weights it holds.

    `repo_id` already names the checkpoint; these are the other two inputs a caller can
    override, and without them a customer's build is indistinguishable from the shipped one.
    Both builders write it from here, so a fourth overridable input is one edit and not one
    per builder.
    """
    return {
        "head_source": head_source(config),
        "calibration_dataset": config.calibration_dataset,
        "n_calib": int(config.n_calib),
    }


def head_source(config):
    """What the card's head was loaded from, or None when it compiles no head of its own.

    Read off the ONE field that declares a head, whichever builder bakes it and whether or not
    the source keeps a copy of its own. The artifact records the string, so a reader can tell a
    customer's head from the published one without opening the weights.
    """
    return config.head


def source_host_params(config, ref) -> dict:
    """Host-side arrays the upstream contributes to the artifact, keyed by array name.

    A source that owns a HOST stage (an output head's trained weights) implements
    `host_params(ref)`, and the arrays land in `host_params.npz` beside the pixel scale and the
    padding mask, which is where the bring-up contract puts a host stage's weights. A source
    without the hook contributes nothing, so a plain backbone is unaffected. Keeping the head in
    the artifact is what lets a run reload it without the upstream checkpoint.
    """
    hook = getattr(_source(config), "host_params", None)
    return dict(hook(ref)) if hook is not None else {}


def _write_contract_artifact(
    model, block_config, config, specs, artifact_dir, seq_real, extra_host_params=None
) -> None:
    """Emit the runtime-loadable contract artifact (pipeline.json + host_params.npz) alongside
    model.json, the same contract the LLM build writes. The vision host front-end runs in torch,
    so host_params carries what bridges host->chip: the pixel int8 scale, the geometry, the
    precomputed padding mask the `encode` graph consumes as its `attn_mask` input, and any host
    stage the source contributes (`source_host_params`). Each device graph in model.json becomes
    one device stage so validate_artifact's stages<=graphs holds."""
    artifact_dir = Path(artifact_dir)
    graphs = list(json.loads((artifact_dir / "model.json").read_text()).get("graphs", {}))
    pix_scale = float(model.patch_embed.proj.act_observer.qparams()[0].reshape(-1)[0])
    mask_shape = tuple(specs["block"]["attn_mask"][0])  # (1,1,seq_pad,ctx_blocks,64) int8
    keys = np.arange(mask_shape[-2] * 64).reshape(mask_shape[-2], 64)
    attn_mask = np.broadcast_to(
        np.where(keys < seq_real, 0, -127).astype(np.int8), mask_shape
    ).copy()
    host_params = {
        "pix_scale": np.float32(pix_scale),
        "attn_mask": attn_mask,
        "seq_real": np.int32(seq_real),
        "seq_pad": np.int32(block_config.seq_len),
        "image_size": np.int32(block_config.image_size),
        "patch_size": np.int32(block_config.patch_size),
    }
    write_host_params(artifact_dir, host_params, extra_host_params)
    write_pipeline_json(
        artifact_dir,
        implementation=config.implementation,
        device_model_json="model.json",
        image_size=block_config.image_size,
        graphs=graphs,
    )


def preprocess_image(image, processor) -> np.ndarray:
    """(image path | PIL image) + a HF image processor -> (1, C, H, W) pixel_values."""
    if isinstance(image, (str, Path)):
        from PIL import Image

        image = Image.open(image).convert("RGB")
    return processor(images=image, return_tensors="pt").pixel_values


def preprocess(image, config):
    """(image path | PIL image) -> preprocessed pixel_values via the source's own processor."""
    return preprocess_image(image, _source(config).make_processor(config.image_size))


def timeit(fn, *, iters: int = 20, warmup: int = 3) -> dict:
    """Time `fn()`: `warmup` untimed calls, then a timed loop -> median latency (ms) + fps.

    The MEDIAN, not the mean over a total: one scheduling stall in a twenty-call loop moves a
    mean by more than any change worth reporting, and a latency figure is quoted as what a
    caller usually waits. `fps` is derived from the same median so the two describe one
    measurement rather than two.
    """
    from statistics import median

    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(iters):
        start = perf_counter()
        fn()
        samples.append(perf_counter() - start)
    seconds = median(samples)
    return {"ms": 1000.0 * seconds, "fps": (1.0 / seconds) if seconds > 0 else 0.0}

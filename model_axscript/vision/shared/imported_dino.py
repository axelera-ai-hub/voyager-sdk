# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Build a vision card on the IMPORTED DINOv2 implementation, sourced through EncoderSource.

This is the vision family's build. `model_axscript/llm/models/dino/` authors the device graph
and everything after calibration is its code path unchanged; what this module adds is the
SEAM, plus the artifact files that go beside the compiled graph.

The seam is why the module exists. The imported `load_dinov2` hardcodes
`Dinov2Model.from_pretrained`, so a card cannot bring its own upstream, and a segmentation
card cannot come from Hugging Face at all because the ADE20K head is published as a file
rather than as a model card. Here every upstream fact comes from the card's `EncoderSource`
instead of from a repository id, and the imported tree is not edited, which matters because
the ALTO team keeps porting into it.

Building on the imported implementation rather than on a copy of it was a MEASUREMENT. This
tree used to carry a fork of the same encoder, and on the same checkpoint, calibration set and
device its CLS features agreed with the fp reference at cosine 0.917 where the imported
implementation reached 0.984. The two differed only in the toolkit under them. The fork is
gone; `axelera_zoo/docs/internal/axscript_zoo_integration.md` keeps the measurement.

A source hands over a block config, which is copied field by field rather than passed across,
so a source that derives a parameter the device graph has nowhere to put fails by name instead
of having it silently dropped somewhere inside the compile.
"""

from __future__ import annotations

import contextlib
import dataclasses
from typing import Any

import numpy as np
import torch

_ALIGN = 64  # the sequence is padded to a multiple of this; the padded keys are masked out


def _round_up(n: int, m: int) -> int:
    return ((n + m - 1) // m) * m


def _imported_block_config(block_config) -> Any:
    """The imported `DinoBlockConfig` carrying the fields the source derived."""
    from model_axscript.llm.models.dino.reference import DinoBlockConfig as Imported

    fields = {field.name for field in dataclasses.fields(Imported)}
    supplied = {
        name: value for name, value in dataclasses.asdict(block_config).items() if name in fields
    }
    dropped = sorted(set(dataclasses.asdict(block_config)) - fields)
    if dropped:
        raise ValueError(
            f"the source derived block-config fields the imported implementation does not "
            f"have: {dropped}. The two dataclasses have diverged, so copying between them "
            f"would silently drop a parameter the device graph needs."
        )
    config = Imported(**supplied)
    config.validate()
    return config


def calibration_pixels(config, source) -> list:
    """Preprocessed calibration images, through the card's own source and dataset.

    The imported `image_calibration` defaults to a 64px dataset, which under-covers the
    LayerNorm ranges at high resolution (see `shared/calibration.py`), so the images come from
    the card's declared set and are preprocessed by the source's own processor.
    """
    from .calibration import load_calibration_images

    processor = source.make_processor(config.image_size)
    images = load_calibration_images(config.n_calib, dataset=config.calibration_dataset)
    return [processor(images=image, return_tensors="pt").pixel_values for image in images]


def load_calibrated(config, *, seg_head_dim: int | None = None):
    """Load the upstream, build the imported reference, and calibrate it to int8.

    Returns `(reference, model, block_config, seq_real)`, matching the imported `load_dinov2`,
    so everything downstream of calibration is the imported code path unchanged.

    `seg_head_dim` is the width a baked segmentation head expects. When it is given, the
    symmetric int8 scale of the final-norm output is calibrated too, because that output
    becomes the head's activation input on device.
    """
    from model_axscript.llm.models.dino.reference import DinoV2Model, smoothquant_groups
    from model_axscript.llm.quant import (
        enable_w8a8,
        quantize_model,
        record_lut_ranges,
        set_calibration_method,
    )

    from .build import _source

    source = _source(config)
    reference = source.load_reference()
    pixels = calibration_pixels(config, source)
    if not pixels:
        raise ValueError(
            f"no calibration images loaded from {config.calibration_dataset!r}; int8 scales "
            f"cannot be observed without them"
        )
    image_size = int(pixels[0].shape[-1])

    seq_real = int(source.seq_len(reference, pixels[0]))
    block_config = _imported_block_config(
        source.block_config(reference, _round_up(seq_real, _ALIGN), image_size)
    )
    if seg_head_dim is not None and seg_head_dim != block_config.hidden_size:
        raise ValueError(
            f"the segmentation head expects a {seg_head_dim}-wide backbone and this one is "
            f"{block_config.hidden_size} wide"
        )

    model = DinoV2Model(
        block_config,
        source.num_layers(reference),
        block_config.num_channels,
        block_config.patch_size,
        seq_real,
    )
    source.load_weights(model, reference, image_size)
    model.eval()
    check_reference_against_upstream(source, reference, model, pixels[0])

    model.to(config.cal_device)
    calibration = [px.to(config.cal_device) for px in pixels]
    if config.calib_method != "minmax":
        observers = set_calibration_method(
            model, config.calib_method, percentile=config.percentile
        )
        print(
            f"[dino calib] {config.calib_method} (p={config.percentile}) on {observers} observers"
        )
    quantize_model(
        model,
        calibration,
        groups=smoothquant_groups(model),
        smooth_alpha=config.smooth_alpha,
        qk_alpha=None,
    )
    enable_w8a8(model)
    record_lut_ranges(model, calibration)
    if seg_head_dim is not None:
        with torch.no_grad():
            amax = max(float(model(px).abs().max()) for px in calibration)
        model.seg_feat_scale = amax / 127.0
    model.to("cpu")
    return reference, model, block_config, seq_real


@torch.no_grad()
def check_reference_against_upstream(source, reference, model, pixels, *, floor: float = 0.99):
    """Check OUR fp reference against the upstream's own forward, before quantisation.

    A weight-mapping fault (a permuted projection, a missing bias, the wrong patch grid) is
    invisible until the chip disagrees with torch, hours into a build. The upstream computes
    the same features in seconds, and both sides here are fp32 on the same pixels, so the whole
    class of faults is ruled out or named now. Returns the CLS cosine.

    The two sides have to be read at the SAME point. `source.final_norm` (an optional
    attribute, True when a source does not declare one, matching every source before
    DinoMixSource) says whether `source.encode` already applies the encoder's final
    LayerNorm; `model` is read at that same point, through `encoder_output` when it does not.
    Comparing a post-norm side against a pre-norm one fails however correct the weight mapping
    is, since the two tensors are then different functions of the same weights, not the same
    function computed twice.

    The floor is 0.99 rather than 1.0 because the two paths differ in arithmetic order: ours
    pads the sequence and slices it back. A mapping fault does not land at 0.99; it lands far
    below.
    """
    from .similarity import mean_cosine

    final_norm = getattr(source, "final_norm", True)
    ours_features = model(pixels) if final_norm else model.encoder_output(pixels)
    ours = np.asarray(ours_features.detach().float())[0, 0].astype("float32")
    upstream = np.asarray(source.encode(reference, pixels).detach().float())[0, 0].astype(
        "float32"
    )
    if ours.shape != upstream.shape:
        raise ValueError(
            f"the fp reference emits a {ours.shape} CLS feature and the upstream emits "
            f"{upstream.shape}; the two are not the same model"
        )
    cosine = float(mean_cosine(ours[None], upstream[None]))
    if cosine < floor:
        raise ValueError(
            f"the fp reference disagrees with the upstream on the same pixels: CLS cosine "
            f"{cosine:.5f}, floor {floor}. Fix the weight mapping before compiling."
        )
    print(f"[dino fp] reference agrees with the upstream, CLS cosine {cosine:.5f}")
    return cosine


def build_artifact(config):
    """Calibrate and compile one encoder artifact, for every vision card.

    The one vision build path. This tree used to carry a second author of the device graph
    beside the imported one, and consolidating on the import left this the only one; the
    artifact is the same SHAPE of directory whatever the card (model.json, pipeline.json,
    host_params.npz, meta.json) and one contract gates it.

    `config.head` bakes a linear segmentation head into the final kernel, so the graph
    emits per-token class logits instead of features, in the same single dispatch. The head is
    a property of the artifact rather than of the run: an artifact built with one cannot emit
    features, which is why the class count is written into meta.json.
    """
    from pathlib import Path

    from model_axscript.shared.artifact_contract import validate_artifact, validate_run_size
    from model_axscript.shared.compiler import require_alto_dtypes
    from model_axscript.shared.runtime import _NP_DTYPE, probe_runtime_load

    # Check the compiler BEFORE the download and the calibration: a missing dtype surfaces in
    # seconds, and finding it after a multi-minute calibration costs that time for nothing.
    require_alto_dtypes()

    from model_axscript.llm.models.dino.model import build_encoder_graph

    from .build import _write_contract_artifact, _write_meta, source_host_params
    from .seg_head import check_backbone, folded_head, load_seg_head

    head = load_seg_head(config.head) if config.head else None

    artifact_dir = Path(config.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    reference, model, block_config, seq_real = load_calibrated(
        config, seg_head_dim=head.dim if head else None
    )
    if head is not None:
        check_backbone(head, head_spec=config.head, hidden_size=block_config.hidden_size)
    # The effective size, not the requested one: config.image_size may be None and resolve
    # through the processor. Checked before the compile, like the other build path.
    validate_run_size(block_config.image_size, config.supported_image_sizes, "image_size")

    seg_w, seg_b = folded_head(head) if head is not None else (None, None)
    _, specs = build_encoder_graph(
        reference,
        model,
        block_config,
        artifact_dir,
        cores=config.shard,
        final_norm=config.final_norm,
        seg_head_w=seg_w,
        seg_head_b=seg_b,
        seg_feat_scale=float(model.seg_feat_scale) if head is not None else None,
    )
    _write_meta(
        model,
        block_config,
        config,
        specs,
        artifact_dir,
        seq_real,
        seg_num_classes=head.num_classes if head is not None else None,
    )
    _write_contract_artifact(
        model,
        block_config,
        config,
        specs,
        artifact_dir,
        seq_real,
        # A source that owns a HOST stage (a trained output head) contributes its weights to
        # the artifact, which is what lets a run reload them without the upstream checkpoint.
        # Dropping this does not fail the build: it fails the RUN, on a card whose head is
        # then missing from the file it is read out of.
        extra_host_params=source_host_params(config, reference),
    )
    validate_artifact(artifact_dir, known_dtypes=_NP_DTYPE.keys())
    probe_runtime_load(artifact_dir)
    return artifact_dir


def load_device(config):
    """The loaded device for an artifact this module built, with no torch model and no compile.

    The imported `DinoV2Device` reads the same meta.json `build_artifact` writes, and it owns
    the host halves the graph does not: the int8 pixel quantize, the padding mask, and the
    output slice back to the real token count (and to the class count, with a head).
    """
    from pathlib import Path

    from model_axscript.llm.models.dino.model import DinoV2Device

    # the shard must match the COMPILE: reserving fewer sub-devices than the build spans is
    # a DEVICE_LOST (see shared/runtime.required_cores).
    return DinoV2Device.load(Path(config.artifact_dir), cores=config.shard)


def keyed_artifact_dir(config):
    """Where a build with THESE inputs lives, so identical inputs reuse one artifact.

    Calibration is deterministic (a fixed slice of a fixed dataset, deterministic PTQ), so
    identical inputs produce an identical artifact and reusing one is safe. What is not safe
    is a fixed directory: a run at a different resolution, shard or calibration budget
    would load the previous build and report its numbers as its own.

    The key comes from `config.artifact_fields`, which the front door also uses, and it is built
    by EXCLUSION: it names the two fields that cannot change what is compiled (where the output
    goes, which torch device calibrated it) rather than the fields that can. A key that misses
    any input is worse than no key at all, and an enumerated list here missed the SOURCE, which
    is the object that loads the weights and therefore the one a customer changes.
    """
    import hashlib
    from pathlib import Path

    from model_axscript.llm.models.dino.model import default_cache_root

    from .config import artifact_fields

    fields = artifact_fields(config)
    digest = hashlib.sha1(repr(fields).encode()).hexdigest()[:12]
    return Path(default_cache_root()) / f"{str(config.repo_id).split('/')[-1]}-{digest}"


def load_or_build(config, *, cache_dir=None):
    """The loaded device for `config`, building it once into a keyed cache when it is missing.

    `cache_dir` names a directory to use instead of the keyed one, which is what a caller
    measuring a specific artifact (a report row, a golden gate) passes so that the score and
    the timing describe the same build.
    """
    import dataclasses
    from pathlib import Path

    artifact_dir = Path(cache_dir) if cache_dir else keyed_artifact_dir(config)
    config = dataclasses.replace(config, artifact_dir=artifact_dir)
    if not (artifact_dir / "model.json").is_file():
        print(f"[dino cache] building artifact -> {artifact_dir}")
        build_artifact(config)
    else:
        print(f"[dino cache] reusing artifact {artifact_dir}")
    return load_device(config)


def release(device) -> None:
    """Give the part back, whatever shape the card's driver is. Idempotent.

    A card that runs everything on device holds its `RuntimeModel` directly; one with a host
    stage holds it behind the backbone it wraps. A measurement that keeps its reservation makes
    the NEXT one fail with `requested num_sub_devices(4) > available (4)`, which reads as a part
    too small for the build rather than as a model still holding the part.
    """
    from model_axscript.shared.runtime import release_runtime

    runtime = getattr(device, "runtime", None) or getattr(
        getattr(device, "backbone", None), "runtime", None
    )
    if runtime is not None:
        release_runtime(runtime)


def device_name(device) -> str | None:
    """The device the loaded artifact opened, as the runtime names it. Read back, not echoed."""
    runtime = getattr(device, "runtime", None) or getattr(
        getattr(device, "backbone", None), "runtime", None
    )
    return getattr(getattr(runtime, "_device", None), "name", None)


@contextlib.contextmanager
def loaded_device(config, *, cache_dir=None):
    """The loaded device for `config`, released when the caller is done with it.

    Every accuracy harness needs this. A harness that returns while still holding the part is
    invisible on its own, and breaks the NEXT measurement in the same process: scoring one run
    size and then timing another is exactly what `axzoo report` does.
    """
    device = load_or_build(config, cache_dir=cache_dir)
    try:
        yield device
    finally:
        release(device)

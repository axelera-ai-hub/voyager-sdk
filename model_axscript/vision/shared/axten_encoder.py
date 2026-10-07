# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Build a vision card on axten, the compiler's own ViT offload, sourced through EncoderSource.

The alternative to `imported_dino`. That module authors the device graph in this tree, on top of
`model_axscript/llm/models/dino/`; this one hands the upstream module to
`axelera.compiler.axten` and lets the compiler match the ViT encoder and lower it. Both take
every upstream fact from the card's `EncoderSource`, so a card selects a builder and nothing
else changes.

Why axten owns more than the graph. It takes the REAL token count, pads to the 64 boundary
inside the kernel builder, and generates the int8 key-padding mask itself
(`axten/models/vit/optimized_impl.py::_padding_mask`). So this module never rounds a sequence
up and never builds a mask, and the artifact it writes carries neither.

The two builders cannot run in ONE process: `model_axscript/llm/ops/matmul.py` and
`axten/ops/transformers/matmul.py` both register the alto op `llm.matmul`, and the second
import raises. A caller that wants to compare them runs each in its own process.

Layout of what is written. axten renders its graph model into `<output_dir>/graph`, and
`CompiledModel.load` resolves `model.json` beside the artifact it is given, so the whole
axten artifact stays in `<artifact_dir>/graph`. The runtime-facing `pipeline.json` names it
through its own `device_model_json` field rather than the files being moved.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

import numpy as np
import torch

GRAPH_SUBDIR = "graph"
"""Where axten's own artifact lives under a card's `artifact_dir` (its `AxtenConfig.graph_dir`)."""


@torch.no_grad()
def _retarget_image_size(reference, image_size: int):
    """Resize a HuggingFace position embedding once, offline, so the model is NATIVELY `image_size`.

    HF interpolates its position embedding on every forward pass unless the input grid already
    matches the stored one. That interpolation is a large subgraph the axten embed pattern does
    not match, so the encoder would fall back to the host. Doing the same bicubic resize once at
    load time leaves a graph with no interpolation in it and the same pretrained weights.

    A reference with no `position_embeddings` (a timm ViT stores its own for the native size) is
    returned untouched.
    """
    embeddings = getattr(reference, "embeddings", None)
    position = getattr(embeddings, "position_embeddings", None)
    if position is None:
        return reference

    patch_size = int(reference.config.patch_size)
    if image_size % patch_size:
        raise ValueError(f"image_size {image_size} must be a multiple of patch_size {patch_size}")
    if image_size == int(reference.config.image_size):
        return reference

    hidden = position.shape[-1]
    old_grid = int(round((position.shape[1] - 1) ** 0.5))
    new_grid = image_size // patch_size
    cls_position, patch_position = position[:, :1], position[:, 1:]
    patch_position = patch_position.reshape(1, old_grid, old_grid, hidden).permute(0, 3, 1, 2)
    patch_position = torch.nn.functional.interpolate(
        patch_position.float(), size=(new_grid, new_grid), mode="bicubic", align_corners=False
    )
    patch_position = patch_position.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, hidden)
    embeddings.position_embeddings = torch.nn.Parameter(
        torch.cat([cls_position, patch_position.to(cls_position.dtype)], dim=1),
        requires_grad=False,
    )
    reference.config.image_size = image_size
    patch_embeddings = getattr(embeddings, "patch_embeddings", None)
    if patch_embeddings is not None:
        patch_embeddings.image_size = [image_size, image_size]
    return reference


class _Features(torch.nn.Module):
    """The upstream's features as a plain tensor, which is what `torch.export` can take.

    A HuggingFace encoder returns a dataclass. The card's `EncoderSource.encode` already knows
    how to get features out of its own upstream, so this defers to it rather than reaching for
    an attribute name that only HuggingFace has.
    """

    def __init__(self, source, reference: torch.nn.Module):
        super().__init__()
        self._source = source
        self.reference = reference

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self._source.encode(self.reference, pixel_values)


class _Headed(torch.nn.Module):
    """A backbone's output run through one more module, so both lower into ONE device graph.

    The head is `nn.Linear`-shaped and matches `LinearOffload`, so the compiled graph goes from
    pixels to whatever the head emits with no host round trip. `imported_dino` reaches the same
    place by baking the head into its final kernel; here the head is simply part of the module
    that gets exported, which is why axten needs no seam of its own for it.
    """

    def __init__(self, backbone: torch.nn.Module, head: torch.nn.Module):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(pixel_values))


def _head_this_builder_bakes(config) -> str | None:
    """The head spec THIS builder must bake, or None when it must bake nothing.

    The head field says the card compiles a head. It does NOT say who assembles it, and the two
    are different questions. A source that returns the composite from `load_reference`, a depth
    card being backbone and head in one module, has already put the head inside the reference,
    so baking one here attaches a second head to a module that already has one: the ADE20K
    segmentation projection onto a depth map, which fails as
    `mat1 and mat2 shapes cannot be multiplied` inside the compile.

    A source owning a `head` field is the signal, and it is the same one `with_head` is written
    around (`source.Overridable.with_head`, overridden by `dinov2_depth.Dinov2DepthSource`), so
    the two cannot answer differently. Every caller reads this rather than the field directly, or
    the next head kind re-opens the same hole in whichever site was missed.
    """
    if getattr(config.source, "head", None) is not None:
        return None
    return config.head or None


def _seg_head_module(config) -> torch.nn.Module | None:
    """The card's segmentation head as one `nn.Linear`, or None when the card is headless.

    The head is a BatchNorm plus a 1x1 conv; `seg_head.folded_head` collapses the pair
    algebraically, which is the same fold `imported_dino` bakes into its final kernel. So the two
    builders emit the same logits from the same weights, and a card moves between them by
    changing one string.

    The `nn.Linear` is shaped from the head that arrived, so a customer's head with a different
    class count is a different device graph and reaches the compile as one.
    """
    head_spec = _head_this_builder_bakes(config)
    if not head_spec:
        return None
    from .seg_head import folded_head, load_seg_head

    weight, bias = folded_head(load_seg_head(head_spec))
    head = torch.nn.Linear(weight.shape[1], weight.shape[0])
    head.weight = torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(weight)).float(), False)
    head.bias = torch.nn.Parameter(torch.from_numpy(np.ascontiguousarray(bias)).float(), False)
    return head.eval()


def _seg_num_classes(config) -> int | None:
    """How many classes the card's baked head predicts, or None when the card is headless.

    Read from the head FILE, not from the config: the config names a backbone size or a path,
    and the class count is a property of the head that arrived under it.
    """
    head_spec = _head_this_builder_bakes(config)
    if not head_spec:
        return None
    from .seg_head import load_seg_head

    return int(load_seg_head(head_spec).num_classes)


def exportable_module(config, source, reference) -> torch.nn.Module:
    """The module axten exports: the upstream at `config.image_size`, plus the card's head.

    A source may substitute its OWN module through an `export_module` hook, because the module
    the fp leg runs is not always the module that can be exported: a timm ViT fuses q, k and v
    into one projection, which matches no block pattern, while the identical weights in
    HuggingFace's layout do. The hook returns a module that already emits a bare tensor, so it
    replaces `_Features` rather than being wrapped by it.
    """
    hook = getattr(source, "export_module", None)
    if hook is not None:
        backbone = hook(reference, config.image_size)
    else:
        backbone = _Features(source, _retarget_image_size(reference, config.image_size))
    head = _seg_head_module(config)
    module = backbone if head is None else _Headed(backbone, head)
    return module.eval()


def check_head_fits(config, source, reference, pixels, image_size: int) -> None:
    """Refuse a head paired with a backbone it was not trained on, BEFORE the compile.

    `imported_dino.build_artifact` runs this check on its own path and this builder had none,
    so a mismatched head reached the first matmul inside the compiler and came back as
    `a and b must have same reduction dim, but got [1370, 384] X [768, 12]`, which names
    neither the head nor the backbone. Every shipped segmentation card builds through here,
    so that was the path a customer actually took.
    """
    head_spec = _head_this_builder_bakes(config)
    if not head_spec:
        return
    from .seg_head import check_backbone, load_seg_head

    hidden = _geometry(source, reference, pixels, image_size)[1]
    check_backbone(load_seg_head(head_spec), head_spec=head_spec, hidden_size=hidden)


def _model_name(config) -> str:
    """The name axten gives the compiled graph and its kernels, from the checkpoint the card names.

    NOT from the artifact directory. A card's directory is `build/<tag>/axscript`, whose leaf is
    the BACKEND, so every card would compile to a graph called "axscript" and their kernels would
    be indistinguishable in a profile or a log.

    A checkpoint the customer keeps on disk is named by its directory's leaf rather than by the
    whole path, or every kernel in a profile would carry the absolute path to it as a prefix.
    """
    checkpoint = Path(str(config.repo_id))
    name = checkpoint.name if checkpoint.is_dir() else str(config.repo_id)
    return re.sub(r"[^0-9A-Za-z]+", "_", name).strip("_")


def axten_config(config, artifact_dir: Path) -> Any:
    """The `AxtenConfig` this card compiles under.

    `num_sub_devices` is PINNED to the shard rather than left to derive: the derived value can
    come back below `num_cores`, and reserving fewer sub-devices than the build spans is a
    DEVICE_LOST at run time.
    """
    from axelera.compiler import axten

    return axten.AxtenConfig(
        name=_model_name(config),
        output_dir=artifact_dir,
        num_cores=config.shard,
        num_sub_devices=None if config.shard == 1 else config.shard,
    )


#: The sequence granularity `DinoBlockConfig.validate` demands, because the IN-TREE recipe packs
#: its key-padding mask at that width. axten has none: it compiles for the real token count.
_RECIPE_SEQ_GRANULARITY = 64


def _geometry(source, reference, pixels, image_size: int) -> tuple:
    """`(seq_real, hidden, grid, patch_size)` of the BACKBONE, from the source that owns it.

    Read off the encoder rather than off the compiled output shape, because the output shape is
    the CARD's, not the backbone's: a segmentation card emits one class row per token and a depth
    card emits a dense map with no token axis at all, so a geometry derived from it would record
    the head's width as the encoder's, or fail to derive anything.

    `seq_real` is the REAL token count, and it is what the graph runs. The block config is asked
    for with that count ROUNDED UP, and only its `hidden_size` and `patch_size` are read: those
    two are properties of the checkpoint and independent of any sequence, while `validate` refuses
    a sequence that is not a multiple of 64. That rule belongs to the in-tree recipe's mask
    packing, and DINOv2 at 518px runs 1370 tokens, so asking with the real count would refuse a
    build that is already correct. Nothing here reads the padded number back.
    """
    seq_real = int(source.seq_len(reference, pixels))
    granularity = _RECIPE_SEQ_GRANULARITY
    padded = -(-seq_real // granularity) * granularity
    block = source.block_config(reference, padded, image_size)
    patch_size = int(block.patch_size)
    if image_size % patch_size:
        raise ValueError(f"image size {image_size} is not divisible by patch size {patch_size}")
    grid = image_size // patch_size
    if grid * grid + 1 != seq_real:
        raise ValueError(
            f"a {grid}x{grid} patch grid plus one prefix token is {grid * grid + 1} tokens and "
            f"the source reports {seq_real}; this encoder's geometry is not one CLS token over a "
            f"square grid, so it cannot be recorded this way"
        )
    return seq_real, int(block.hidden_size), grid, patch_size


def _pixel_scale(compiled) -> float:
    """The int8 quantization scale for the input pixels, off the embed site's calibration.

    The current path reads the same number from its torch fake-quant observer; here the compiler
    owns the calibration, so it comes from the site it calibrated.
    """
    for site in compiled.sites.values():
        if "pix" in site.params:
            return float(site.params["pix"])
    raise ValueError(
        "the compiled model has no calibrated pixel scale: no site carries a 'pix' param, so the "
        "patch-embedding pattern did not match and the encoder is not fully offloaded"
    )


#: The distributions that decide what this builder compiles. Recorded per artifact because a
#: card's accuracy is now a function of the compiler it was built with, so a number that moves
#: after a wheel bump has to be attributable to it.
_COMPILER_PACKAGES = ("axelera-compiler", "axelera-tvm", "axelera-codegen", "axelera-config")


def _compiler_versions() -> dict:
    """Version of each alto distribution present, for `meta.json`.

    All of them, not one summary string. They can disagree: an editable dev shadow reports its
    source version (`0.0.1`) rather than a revision, and the native and pure-Python halves can
    sit at different revisions. A single field would hide exactly the skew a reader chasing a
    changed number needs to see.
    """
    from importlib.metadata import PackageNotFoundError, version

    versions = {}
    for package in _COMPILER_PACKAGES:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            continue
    return versions


def write_runtime_artifact(config, compiled, reference, pixels, image_size: int) -> None:
    """Write `meta.json`, `pipeline.json` and `host_params.npz` beside axten's graph.

    The same contract `shared/artifact_contract.validate_artifact` gates for every card, so
    `axzoo` and the runtime see no difference between a card built here and one built by
    `imported_dino`.

    THREE FIELDS ARE DELIBERATELY ABSENT, and restoring them is a bug. `imported_dino` writes
    `attn_mask` and `seq_pad` into `host_params.npz` and `seq_pad` and `mask_shape` into
    `meta.json`, because its device graph takes a padded sequence and a host-built key-padding
    mask. axten takes the real token count, pads to the 64 boundary inside its own kernel
    builder, and generates the `{0, -127}` mask there
    (`axten/models/vit/optimized_impl.py::_padding_mask`, which says it "owns the WHOLE map").
    Carrying the old fields over would ship a second mask that contradicts the one in the kernel,
    and an unused array in an `.npz` raises nothing, so the contradiction would be silent.
    """
    from .build import (
        _source,
        build_provenance,
        source_host_params,
        write_host_params,
        write_pipeline_json,
    )

    artifact_dir = Path(config.artifact_dir)
    source = _source(config)
    seq_real, hidden, grid, patch_size = _geometry(source, reference, pixels, image_size)
    pix_scale = _pixel_scale(compiled)
    seg_num_classes = _seg_num_classes(config)

    meta = {
        "repo_id": config.repo_id,
        "image_size": image_size,
        "patch_size": patch_size,
        "seq_real": seq_real,
        "hidden": hidden,
        "grid": grid,
        "pix_scale": pix_scale,
        "seg_head": seg_num_classes is not None,
        "seg_num_classes": seg_num_classes,
        "final_norm": bool(config.final_norm),
        **build_provenance(config),
        "compiler": _compiler_versions(),
    }
    (artifact_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    host_params = {
        "pix_scale": np.float32(pix_scale),
        "seq_real": np.int32(seq_real),
        "image_size": np.int32(image_size),
        "patch_size": np.int32(patch_size),
    }
    write_host_params(artifact_dir, host_params, source_host_params(config, reference))

    model_json = Path(GRAPH_SUBDIR) / "model.json"
    graphs = list(json.loads((artifact_dir / model_json).read_text()).get("graphs", {}))
    write_pipeline_json(
        artifact_dir,
        implementation=config.implementation,
        device_model_json=str(model_json),
        image_size=image_size,
        graphs=graphs,
    )


#: Calibration knobs `EncoderConfig` declares that this builder does NOT act on. axten runs its
#: own observers inside the block pattern, and `patterns=["all"]` constructs that pattern with
#: the compiler's defaults; nothing here can reach them without hand-building a pattern list,
#: which the compiler's own suite says not to do (a bare instance prepended to "all" claims the
#: internals of the block patterns behind it). The DEFAULTS happen to agree today, so a card
#: leaving them alone is unaffected; a card that changed one would be ignored without a word.
_COMPILER_OWNED_CALIBRATION = ("smooth_alpha", "calib_method", "percentile")


def _refuse_unhonoured_knobs(config) -> None:
    """Refuse a card whose calibration knobs this builder cannot act on, by name.

    Compared against the field DEFAULT rather than a fixed value, so the check keeps meaning if
    a default moves. Refusing costs a build; not refusing costs an accuracy investigation into a
    knob that was never applied.
    """
    import dataclasses

    defaults = {
        field.name: field.default
        for field in dataclasses.fields(config)
        if field.name in _COMPILER_OWNED_CALIBRATION
    }
    ignored = {
        name: getattr(config, name)
        for name, default in defaults.items()
        if getattr(config, name) != default
    }
    if ignored:
        raise NotImplementedError(
            f"the axten builder does not act on {sorted(ignored)}: the compiler runs its own "
            f"observers inside the block pattern and this build cannot reach them, so "
            f"{ignored} would be ignored in silence. Build this card with an in-tree "
            f"implementation, or leave the knobs at their defaults and let the compiler "
            f"calibrate."
        )


def build_artifact(config):
    """Calibrate and compile one encoder artifact through axten, into `config.artifact_dir`.

    Same signature and same job as `imported_dino.build_artifact`. This lands the compile only:
    the runtime-facing `meta.json`, `pipeline.json` and `host_params.npz` are written beside it
    by `write_runtime_artifact`.
    """
    from model_axscript.shared.artifact_contract import validate_artifact, validate_run_size
    from model_axscript.shared.compiler import require_axten
    from model_axscript.shared.runtime import _NP_DTYPE

    # Checked BEFORE the download and the calibration, so a compiler without axten costs
    # seconds rather than surfacing after a multi-minute calibration.
    require_axten()
    _refuse_unhonoured_knobs(config)

    from axelera.compiler import axten

    from .build import _source
    from .calibration import image_calibration

    artifact_dir = Path(config.artifact_dir)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    source = _source(config)
    reference = source.load_reference()
    calibration = image_calibration(
        source,
        n=config.n_calib,
        dataset=config.calibration_dataset,
        image_size=config.image_size,
    )
    if not calibration:
        raise ValueError(
            f"no calibration images loaded from {config.calibration_dataset!r}; int8 scales "
            f"cannot be observed without them"
        )
    # The effective size, not the requested one: config.image_size may be None and resolve
    # through the processor.
    image_size = int(calibration[0].shape[-1])
    validate_run_size(image_size, config.supported_image_sizes, "image_size")

    # Before the compile, so a head that does not fit the backbone costs seconds rather than
    # surfacing minutes in as a shape error naming neither side.
    check_head_fits(config, source, reference, calibration[0], image_size)
    compiled = axten.compile(
        exportable_module(config, source, reference),
        calibration[0],
        patterns=["all"],
        calib_inputs=calibration,
        config=axten_config(config, artifact_dir),
    )
    print(f"[axten] {compiled.summary()}")
    print(f"[axten] coverage:\n{compiled.coverage().format()}")
    # Before save(), which finalizes the graph and drops the calibration the scales come from.
    write_runtime_artifact(config, compiled, reference, calibration[0], image_size)
    compiled.save()
    validate_artifact(artifact_dir, _NP_DTYPE)
    return compiled


class AxtenEncoder:
    """A built axten artifact, presented as the `encode(prepared)` surface a card's `infer` calls.

    The whole host stage is inside the reloaded graph: the pixel quantize, the alignment, the
    device dispatch and the dequantize are all nodes in it, so there is nothing here to do but
    call it. That is the difference from `DinoV2Device`, which owns those host halves itself.
    """

    def __init__(self, compiled):
        self._compiled = compiled

    def encode(self, prepared):
        """Last-hidden-state features for one preprocessed `(1, 3, H, W)` input."""
        return self._compiled(prepared)


def release(device) -> None:
    """Give the part back. Idempotent, and safe to call on a device that never loaded one.

    Not a method on the loaded object: axten keeps its loaded graph models in a module-level
    cache (`passes/graph_compile._LOADED`), because the device op is a torch custom op with no
    handle to thread one through. A loaded model holds its sub-devices for as long as it lives,
    so a harness that scores one artifact and then loads another in the same process needs this
    between them, or the second load fails with `requested num_sub_devices(N) > available`.
    """
    from axelera.compiler.axten.passes.graph_compile import release_graph_models

    release_graph_models()


def device_name(device) -> str | None:
    """The device the loaded graph actually opened, as the runtime names it. Read back, not echoed.

    Not read off `device` for the same reason `release` is not a method on it: axten keeps its
    loaded graph model in a module-level cache, because the device op is a torch custom op with
    no handle to thread one through, so the object holding the reservation is not the one the
    caller has. `loaded_device_name` is the compiler's own accessor for that cache.
    """
    from axelera.compiler.axten.passes.graph_compile import loaded_device_name

    return loaded_device_name()


def load_device(config) -> AxtenEncoder:
    """The loaded device for an artifact this module built, with no torch model and no compile.

    `CompiledModel.load` reads the graph, the baked calibration and the sub-device count from the
    artifact, so a run neither recompiles nor recalibrates, and the shard it reserves is the one
    the build spanned.
    """
    from axelera.compiler import axten

    return AxtenEncoder(axten.CompiledModel.load(Path(config.artifact_dir) / GRAPH_SUBDIR))

# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""The five driver seams shared by every monocular-depth card.

The depth counterpart of `ade20k_seg.py`. A depth card's output is a dense `(1, H, W)` map with no
token axis and no label space, and the head that produces it is part of the module the compiler
exports, so the seams are the same five for every such card and only the card's `EncoderSource`
differs: `dinov2_depth.Dinov2DepthSource` composes a DINOv2 backbone with Meta's NYU linear head,
`depth_anything.DepthAnythingSource` takes HuggingFace's already-joined DPT model.

Both legs of the accuracy run the SAME head, because the head is inside the module on the device
leg and inside the reference on the fp leg. So a device-versus-fp difference is encoder and head
quantization, and never two implementations of the head disagreeing.
"""

from __future__ import annotations


def build(config) -> None:
    """Compile the artifact `config` describes into its `artifact_dir`."""
    from .builder import build_artifact

    build_artifact(config)


def load_device(config):
    """The loaded card, compiling first only when nothing this builder wrote is there."""
    from .builder import is_built
    from .builder import load_device as load

    if not is_built(config):
        print(f"[build] {config.repo_id} -> {config.artifact_dir}")
        build(config)
    return load(config)


def load_reference(config):
    """The fp32 counterpart of `load_device`: the same module, unquantized, on the host."""
    from .build import _source
    from .reference import ReferenceEncoder

    source = _source(config)
    return ReferenceEncoder(
        source,
        source.load_reference(),
        device=config.cal_device,
        image_size=config.image_size or 0,
    )


def prepare(image, config):
    """The preprocessed pixels for one image: the source's own resize, crop and normalize."""
    from .build import preprocess

    return preprocess(image, config)


def infer(device, prepared):
    """The `(1, H, W)` depth map for one prepared input, on the loaded device."""
    return device.encode(prepared)


def report_run(device, images, config) -> str:
    """What this card predicts, and how far the device is from the fp reference predicting it.

    Three numbers, because they answer different questions. The depth RANGE is the prediction
    itself, in the unit the head emits, and is the line a reader judges first: a map that is flat
    or negative is wrong however well it correlates. `AbsRel` is the per-pixel relative error
    against the fp leg, which is the quantity the depth literature reports and the one a lowering
    error shows up in. The cosine is a smoke test: a smooth depth map reads ~0.999 on it even when
    individual pixels are off, so it is quoted to be dismissed rather than relied on.

    Both legs read the SAME prepared pixels, so what is measured is the device against the
    reference and not two preprocessing runs against each other. This is a FIDELITY line, not the
    card's accuracy: the scored number comes from `axzoo eval`, against ground truth.
    """
    import numpy as np

    reference = load_reference(config)
    lines = []
    for image in images:
        prepared = prepare(image, config)
        device_depth = np.asarray(infer(device, prepared), dtype=np.float32).reshape(-1)
        fp_depth = np.asarray(reference.encode(prepared), dtype=np.float32).reshape(-1)
        lines.append(f"{image}: {_range(device_depth)}")
        lines.append(f"  {_agreement(device_depth, fp_depth)}")
    return "\n".join(lines)


def _range(depth) -> str:
    """The predicted depth map as the three numbers that say whether it is a depth map at all."""
    import numpy as np

    return (
        f"depth min {float(depth.min()):.3f}, median {float(np.median(depth)):.3f}, "
        f"max {float(depth.max()):.3f}"
    )


def _agreement(device_depth, fp_depth) -> str:
    """How the device's depth compares with the fp reference's, in one line."""
    import numpy as np

    from .similarity import mean_cosine

    # Only where the reference is positive: AbsRel divides by it, and a head emitting a zero or a
    # negative there is reporting something other than a distance.
    valid = fp_depth > 0
    abs_rel = (
        float(np.mean(np.abs(device_depth[valid] - fp_depth[valid]) / fp_depth[valid]))
        if valid.any()
        else float("nan")
    )
    return (
        f"vs the fp reference: AbsRel {abs_rel:.4f} over {int(valid.sum())} pixels, "
        f"cosine {mean_cosine(device_depth[None], fp_depth[None]):.4f}"
    )

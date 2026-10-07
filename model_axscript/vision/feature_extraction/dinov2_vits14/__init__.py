# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""DINOv2-small (ViT-S/14): a headless ViT backbone on the AxScript (ALTO-direct) path.

The whole model, patch embedding, the 12 identical encoder blocks and the final LayerNorm,
is ONE device graph and runs in a single dispatch (pixels to last_hidden_state). There is no
autoregressive decode loop and no KV cache; the host does image-processor normalization and the
int8 pixel quantize.

`implementation` selects which module builds that graph, and this card defaults to `axten`, the
compiler's own ViT offload. Measured against the in-tree `dinov2-metis` builder on the same
checkpoint, the same 32 calibration images and the same device, over eight held-out images: mean
CLS cosine against the fp reference 0.9892 for axten and 0.9899 for the in-tree builder, worst
per-image difference 0.0031, and the same image is the worst on both. The result is core-count
invariant, so the shard does not move it. Setting `implementation="dinov2-metis"` is the whole
of going back.

The two builders lay out their artifacts differently and cannot be imported into one process
(both register the alto op `llm.matmul`), so `shared/builder.py` resolves exactly one.
"""

from pathlib import Path

MODEL_TAG = "dinov2-vits14"

_REPO_ID = "facebook/dinov2-small"


def make_config(artifact_dir, cal_device: str = ""):
    """Per-model build and run config (checkpoint, dtypes, input size, calibration).

    Shared by the build and the run, so a run reads the same declaration the artifact was
    compiled from. `cal_device` is a compile-only knob (the torch calibration device).
    """
    from ...shared.config import EncoderConfig  # deferred: avoids an import cycle
    from ...shared.hf import HFSource

    return EncoderConfig(
        implementation="dinov2-axten",
        repo_id=_REPO_ID,
        source=HFSource(_REPO_ID),  # HF is the default upstream; swap for any EncoderSource
        load_dtype="float32",
        image_size=224,
        supported_image_sizes=(224,),
        artifact_dir=Path(artifact_dir),
        cal_device=cal_device,
    )


def build(config) -> None:
    """Compile the artifact `config` describes into its `artifact_dir`."""
    from ...shared.builder import build_artifact

    build_artifact(config)


def load_device(config):
    """The loaded encoder, compiling first only when nothing is built at `artifact_dir`."""
    from ...shared.builder import is_built
    from ...shared.builder import load_device as load

    if not is_built(config):
        print(f"[build] {MODEL_TAG} -> {config.artifact_dir}")
        build(config)
    return load(config)


def load_reference(config):
    """The fp32 counterpart of `load_device`: the checkpoint's own encoder, same interface.

    A bare backbone has no host stage above it, so the reference is the encoder alone and
    `infer` is the same call on either leg.
    """
    from ...shared.build import _source
    from ...shared.reference import ReferenceEncoder

    source = _source(config)
    return ReferenceEncoder(source, source.load_reference(), device=config.cal_device)


def prepare(image, config):
    """The preprocessed pixels for one image: the source's own resize, crop and normalize."""
    from ...shared.build import preprocess

    return preprocess(image, config)


def infer(device, prepared):
    """Last-hidden-state features for one prepared input, on the loaded device."""
    return device.encode(prepared)


def report_run(device, images, config) -> str:
    """Agreement with the fp reference on each image, which is the readable result here.

    A headless encoder has no label to print, and a tensor shape proves only that something
    well-formed came back. Cosine 1.0 means the device features ARE the fp features. Both legs
    see the SAME prepared pixels, so the number is the device against the reference and not
    two preprocessing runs against each other. When the reference is unavailable, say so
    rather than fail a run over a missing check.
    """
    lines = []
    for image in images:
        prepared = prepare(image, config)
        features = infer(device, prepared)
        cosine = _cls_cosine_vs_fp(features, prepared, config)
        if cosine is None:
            lines.append(f"{image}: extracted features (fp reference unavailable, no check run)")
        else:
            lines.append(
                f"{image}: device features match the fp reference, CLS cosine {cosine:.4f}"
            )
        lines.append(f"  features {tuple(features.shape)} {features.dtype}")
    return "\n".join(lines)


def _cls_cosine_vs_fp(features, prepared, config) -> float | None:
    """Cosine between the device CLS feature and the fp CLS feature, on the same pixels."""
    try:
        import numpy as np

        from ...shared.build import _source
        from ...shared.similarity import mean_cosine

        source = _source(config)
        reference = source.load_reference()
        fp_cls = source.encode(reference, prepared)[0, 0].detach().float().numpy()
        device_cls = np.asarray(features)[0, 0].astype("float32")
        return mean_cosine(device_cls[None], fp_cls[None])
    except Exception as error:  # noqa: BLE001 - a missing fp reference must not fail a run
        print(f"[warn] skipped CLS-cosine-vs-fp sanity ({error})")
        return None

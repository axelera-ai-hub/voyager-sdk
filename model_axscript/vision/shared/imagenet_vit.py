# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""The five driver seams shared by every plain-ViT ImageNet classification card.

The device side is the DINOv2 implementation every vision card shares (patch embed + N fused
blocks + final LayerNorm, one dispatch), authored from the CARD's shapes and the CARD's weights
(`timm_vit.TimmViTSource`). No new ALTO kernel: the block kernel is shape-driven, and a plain
ViT is the LayerScale-free case of the same block. So ViT-S/16, ViT-B/16 and ViT-L/16 differ
only in the checkpoint their `EncoderConfig` names, and the seams live here once rather than
once per package.

The host side is one linear projection of the CLS token (`imagenet_head.ImageNetHead`). Because
the head is identical on both legs, a device-versus-fp comparison of the logits reflects only
encoder quantization.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # the annotations are strings, so a card imports this module without torch
    import numpy as np

    from model_axscript.llm.models.dino.model import DinoV2Device

    from .imagenet_head import ImageNetHead


class ViTClassifierDevice:
    """Encoder on the AIPU + ImageNet classifier on the host.

    `top_k(pixel_values)` is the readable result for the classification task, and `logits` is
    the device leg the accuracy harness compares against the fp one. The fp leg is NOT a
    method here: a loaded artifact carries no torch model, so the reference comes from the
    card's `EncoderSource`, and a method pretending otherwise raised on exactly the path a
    run takes.

    `source` travels with the device because the harness needs the upstream's own
    preprocessing to score both legs on the same pixels.

    There is no `load` here on purpose. Reserving the part is one decision, and it belongs to
    the one loader every card goes through (`shared/builder.load_device`), which reads
    the shard the artifact was COMPILED at. A second `load` on this class reserved a
    default of one sub-device for a four-core build, which is a `DEVICE_LOST` at the first
    dispatch and reads as a part too small for the model.
    """

    def __init__(self, backbone: DinoV2Device, head: ImageNetHead, source=None) -> None:
        self.backbone = backbone
        self.head = head
        self.source = source

    def logits(self, pixel_values) -> np.ndarray:
        """(1, C, H, W) pixels -> (classes,) logits, encoder on device."""
        return self.head.logits(self.backbone.encode(pixel_values))

    def top_k(self, pixel_values, k: int = 5) -> list[tuple[int, str, float]]:
        """The k best classes as (index, label, probability), encoder on device."""
        return self.head.top_k(self.backbone.encode(pixel_values), k=k)


def build(config) -> None:
    """Compile the artifact `config` describes into its `artifact_dir`."""
    from .builder import build_artifact

    build_artifact(config)


def load_device(config):
    """The loaded classifier: the shared encoder loader, plus the head the artifact carries."""
    from pathlib import Path

    from .builder import is_built
    from .builder import load_device as load_backbone
    from .imagenet_head import ImageNetHead

    if not is_built(config):
        print(f"[build] {config.repo_id} -> {config.artifact_dir}")
        build(config)
    return ViTClassifierDevice(
        load_backbone(config),
        ImageNetHead.from_host_params(Path(config.artifact_dir)),
        source=config.source,
    )


def load_reference(config):
    """The fp32 counterpart of `load_device`: the checkpoint's encoder under the SAME head.

    The head is read from the artifact both legs share, so a device-versus-fp comparison
    reflects encoder quantization and nothing about the host classifier.
    """
    from pathlib import Path

    from .imagenet_head import ImageNetHead
    from .reference import ReferenceEncoder
    from .timm_vit import TimmViTSource

    source = config.source or TimmViTSource(config.repo_id)
    return ViTClassifierDevice(
        ReferenceEncoder(source, source.load_reference(), device=config.cal_device),
        ImageNetHead.from_host_params(Path(config.artifact_dir)),
        source=source,
    )


def prepare(image, config):
    """The preprocessed pixels for one image: the source's own resize, crop and normalize."""
    from .build import preprocess

    return preprocess(image, config)


def infer(device, prepared):
    """ImageNet logits for one prepared input: the encode dispatch plus the host classifier."""
    return device.logits(prepared)


def report_run(device, images, config, top_k: int = 5) -> str:
    """The classes this model predicts, and whether the fp encoder predicts the same.

    The labels are the task result. The fp line beside them is the fidelity signal: the SAME
    host head runs on both legs, so any disagreement there is encoder quantization and nothing
    else. Loading the upstream is optional, because the artifact already carries the classifier
    and an offline run must still produce its labels.
    """
    lines = []
    for image in images:
        prepared = prepare(image, config)
        # Not "ImageNet classes" unconditionally: a fine-tune onto another label set predicts
        # the customer's classes, and the line that introduces them must not claim otherwise.
        count = device.head.num_classes
        named = "ImageNet classes" if device.head.is_imagenet else f"of {count} classes"
        lines.append(f"{image}: top-{top_k} {named}")
        for index, label, probability in device.top_k(prepared, k=top_k):
            lines.append(f"  {probability:6.2%}  {label} ({index})")
        agreement = _fp_agreement(device, prepared, config)
        if agreement is not None:
            lines.append(agreement)
    return "\n".join(lines)


def _fp_agreement(device, prepared, config) -> str | None:
    """Does the fp encoder call this image the same thing? A printable line, or None."""
    try:
        import numpy as np

        from .similarity import mean_cosine
        from .timm_vit import TimmViTSource

        source = config.source or TimmViTSource(config.repo_id)
        reference = source.load_reference()
        fp_logits = device.head.logits(source.encode(reference, prepared).detach().float().numpy())
        device_logits = infer(device, prepared)
        agree = int(np.argmax(device_logits)) == int(np.argmax(fp_logits))
        return (
            f"  fp reference: top-1 {'agrees' if agree else 'DISAGREES'}, "
            f"logit cosine {mean_cosine(device_logits[None], fp_logits[None]):.4f}"
        )
    except Exception as error:  # noqa: BLE001 - a missing fp reference must not fail a run
        print(f"[warn] skipped the fp-reference check ({error})")
        return None

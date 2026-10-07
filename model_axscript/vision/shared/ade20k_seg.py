# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""The five driver seams shared by every DINOv2 + ADE20K linear segmentation card.

The three published sizes (`vits14`, `vitb14`, `vitl14`) differ only in which backbone
checkpoint they load and which head file goes with it, and both are declared by the card's
`EncoderConfig`. So the seams live here once and each model package declares its
configuration, rather than three packages carrying the same body three times.

The head is baked into the device graph, so `infer` returns per-token class LOGITS rather
than features: one dispatch takes pixels to segmentation. Token 0 is the CLS row, which the
head was never trained to classify; the patch rows are `[1:]` and reshape to the `grid x grid`
class map. `report_run` reads the patch rows only, for that reason.
"""

from __future__ import annotations


def build(config) -> None:
    """Compile the artifact `config` describes into its `artifact_dir`."""
    from .builder import build_artifact

    build_artifact(config)


def load_device(config):
    """The loaded card, compiling first only when nothing is built at `artifact_dir`.

    `is_built` and not a file check: the two builders lay an artifact out differently, and it
    also refuses one built by the OTHER builder, so a card whose `implementation` changed
    rebuilds rather than handing one builder's directory to the other's loader.
    """
    from .builder import is_built
    from .builder import load_device as load

    if not is_built(config):
        print(f"[build] {config.head} seg -> {config.artifact_dir}")
        build(config)
    return load(config)


def load_reference(config):
    """The fp32 counterpart of `load_device`: the checkpoint's encoder plus the SAME head.

    The device graph bakes the head, so the reference applies the head this card declares to
    the same token rows and returns per-token class logits, which is what `infer` reads on
    either leg.
    """
    from .build import _source
    from .reference import ReferenceEncoder
    from .seg_head import load_seg_head

    source = _source(config)
    return ReferenceEncoder(
        source,
        source.load_reference(),
        seg_head=load_seg_head(config.head),
        device=config.cal_device,
    )


def prepare(image, config):
    """The preprocessed pixels for one image: the source's own resize, crop and normalize."""
    from .build import preprocess

    return preprocess(image, config)


def infer(device, prepared):
    """Per-token ADE20K class logits for one prepared input, on the loaded device."""
    return device.encode(prepared)


def report_run(device, images, config) -> str:
    """What this card predicts, and how far the device is from the fp reference saying it.

    Two numbers, because they answer different questions. The mean patch cosine says how close
    the int8 logits are to the fp ones, and the label agreement says how often that difference
    changes the predicted class, which is the only difference a segmentation consumer sees. A
    tensor shape would prove neither. The class histogram is the prediction itself, in the
    label vocabulary the head was trained on.

    Both legs read the SAME prepared pixels, so the comparison is the device against the
    reference and not two preprocessing runs against each other.
    """
    from .build import _source
    from .seg_head import class_names, load_seg_head

    head = load_seg_head(config.head)
    # A customer's head predicts their own label set, so the vocabulary comes from what the
    # card declared rather than from ADE20K unconditionally.
    names = class_names(config.head, head)
    source = _source(config)
    reference = source.load_reference()

    lines = []
    for image in images:
        device_logits, fp_logits = patch_logits(device, image, config, head, source, reference)
        lines.append(f"{image}: {_agreement(device_logits, fp_logits)}")
        lines.append(f"  {_histogram(device_logits, names)}")
    return "\n".join(lines)


def patch_logits(device, image, config, head, source, reference):
    """The device's per-patch class logits and the fp reference's, for ONE image.

    One function because two callers need the pair and they must agree about it: `report_run`
    turns it into a line a reader judges, and the mIoU harness turns it into a confusion
    matrix. If the CLS row were dropped on one side and not the other, the report and the
    score would describe the same model differently, and neither number would look wrong.

    Token 0 is the CLS row on both legs, and the head was never trained to classify it, so it
    is dropped from both. Both legs read the SAME prepared pixels, so what is measured is the
    device against the reference rather than two preprocessing runs against each other.
    """
    import numpy as np

    from .seg_head import seg_logits

    prepared = prepare(image, config)
    device_logits = np.asarray(infer(device, prepared)).astype("float32")[0][1:]
    features = source.encode(reference, prepared)[0].detach().float().numpy()
    return device_logits, seg_logits(features[1:], head)


def _agreement(device_logits, fp_logits) -> str:
    """How the device's per-patch logits compare with the fp reference's, in one line."""
    import numpy as np

    from .similarity import mean_cosine

    agree = float(np.mean(device_logits.argmax(-1) == fp_logits.argmax(-1)))
    return (
        f"{100 * agree:.1f}% of {len(device_logits)} patches take the fp reference's label, "
        f"mean patch cosine {mean_cosine(device_logits, fp_logits):.4f}"
    )


def _histogram(logits, names, top: int = 5) -> str:
    """The predicted class mix, as the share of patches each of the top classes takes."""
    import numpy as np

    predicted = logits.argmax(-1)
    counts = np.bincount(predicted, minlength=len(names))
    order = np.argsort(counts)[::-1][:top]
    parts = [f"{names[i]} {100 * counts[i] / len(predicted):.0f}%" for i in order if counts[i] > 0]
    return "predicted: " + ", ".join(parts)

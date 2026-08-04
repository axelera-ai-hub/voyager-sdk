#!/usr/bin/env python
# Copyright Axelera AI, 2026
"""Example script: Evaluate ViT-backbone + linear-head seg models on ADE20K.

Three inference modes:

    --mode torch       Run FP32 baseline (assembled from MODEL_REGISTRY)
    --mode quantized   Run AxMO-quantized FX graph on CPU (INT8 reference)
    --mode torch-aipu  Run on Metis hardware

Metric: ADE20K mean IoU (plus pixel/mean accuracy and dice) via
intersect/union accumulation, matching the reference DINOv2 seg
evaluator. ``ADE20KSegDataset`` preprocesses the image (resize shorter
side -> center-crop -> pad to a fixed square) and puts the label through
the SAME geometry, so predictions and references share one grid and the
model is never scored on pixels it didn't see.
ADE20K annotations are 1-indexed (pixel 0 = background); the FB linear
head was trained with mmseg's ``reduce_zero_label=True`` so channel c
maps to label c+1, which the metric undoes by shifting labels down one
and folding the original 0 (and the 255 pad) into the ignore index.

Compare FP32 vs quantized accuracy on a local ADE20K set::

    python inference_seg.py ./build/dinov2_seg_vits14_ade20k_linear \\
        --mode torch --val-data /path/to/ADEChallengeData2016 --frames 50
    python inference_seg.py ./build/dinov2_seg_vits14_ade20k_linear \\
        --mode quantized --val-data /path/to/ADEChallengeData2016 --frames 50

Chip accuracy on Metis::

    python inference_seg.py ./build/dinov2_seg_vits14_ade20k_linear \\
        --mode torch-aipu --val-data /path/to/ADEChallengeData2016 \\
        --dump-logits /tmp/chip_seg_logits.npz

Notes:
  - The exported model returns patch-resolution logits
    ``[B, num_classes, H/patch, W/patch]``. The script bilinear-upsamples
    on the host to the (square) label resolution before argmax. This
    matches the deploy-side decision in ``deploy_seg.py``.
  - ``--mode torch-aipu`` runs the chip through ``axelera.runtime2``
    ``op.load``, which handles input quant/pack and output
    depad/dequant/align internally (``--ncores`` sets the core allocation).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterator

try:
    from axelera.app import config, logging_utils
except ImportError:
    sys.exit("Please activate the Axelera environment with source venv/bin/activate and run again")

import numpy as np
import torch
import torch.nn.functional as F
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from axelera.model_optimizer.utils.fx_graphmodule_extensions import load_fx_graphmodule
from axelera.runtime2 import op as rt2_op

from deploy_seg import (
    ADE20KSegDataset,
    MODEL_REGISTRY,
    ModelSpec,
    build_seg_model,
    download_ade20k_subset,
)

LOG = logging_utils.getLogger(__name__)


# ---------------------------------------------------------------------------
# Device + config helpers
# ---------------------------------------------------------------------------


def get_device(specified_device: str | None = None) -> str:
    if specified_device is not None:
        LOG.info(f"Using specified device: {specified_device}")
        return specified_device
    if torch.cuda.is_available():
        device = "cuda"
        LOG.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        LOG.info("No GPU detected, using CPU")
    return device


def load_config(model_dir: Path | None) -> dict | None:
    if model_dir is None:
        return None
    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_spec(model_config: dict | None, override_model_name: str | None) -> ModelSpec:
    name = override_model_name
    if name is None and model_config is not None:
        name = model_config.get("model_name")
    if name is None:
        raise SystemExit(
            "Cannot determine model name. Provide --model-name or ensure "
            "config.yaml in the model directory has a `model_name` field."
        )
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise SystemExit(f"Unknown model '{name}'. Available models: {available}")
    return MODEL_REGISTRY[name]


def _log_model_spec(model_name: str, spec: ModelSpec) -> None:
    LOG.info("Resolved model spec:")
    LOG.info(f"  name        : {model_name}")
    LOG.info(f"  backbone    : {spec.backbone_hub_repo} / {spec.backbone_hub_name}")
    LOG.info(f"  head url    : {spec.head_url}")
    LOG.info(f"  input size  : {spec.input_size}x{spec.input_size}")
    LOG.info(f"  num classes : {spec.num_classes}")
    LOG.info(f"  dataset     : {spec.dataset}")


# ---------------------------------------------------------------------------
# Validation data
#
# ADE20KSegDataset preprocesses the image (resize shorter side -> center-crop
# -> pad to a fixed square, ImageNet-normalised in 0-255 space) and runs the
# label through the SAME geometry (nearest-neighbor + 255 pad). Prediction and
# reference therefore live on one square grid, so the model is never scored on
# pixels it didn't see -- this square-aligned eval is what makes the reported
# mIoU correct.
# ---------------------------------------------------------------------------


def get_hf_ade20k_validation_loader(
    spec: ModelSpec,
    batch_size: int,
    frames: int | None,
    data_root: Path,
) -> DataLoader:
    """Download the canonical HuggingFace ADE20K subset and evaluate on its
    validation split via the same local seg loader used by ``--val-data``."""
    ade20k_root = download_ade20k_subset(data_root.expanduser().absolute())
    return get_local_ade20k_validation_loader(ade20k_root, spec, batch_size, frames)


def get_local_ade20k_validation_loader(
    val_data_path: Path,
    spec: ModelSpec,
    batch_size: int,
    frames: int | None,
) -> DataLoader:
    # ADE20KSegDataset owns its seg transform and yields (image, aligned_label,
    # name); its .collate keeps only image + label. max_samples=0 means "all".
    dataset = ADE20KSegDataset(
        root=str(val_data_path),
        spec=spec,
        split="validation",
        max_samples=frames if frames else 0,
    )
    LOG.info(f"Loaded local ADE20K validation set with {len(dataset)} images")
    # ADE20KSegDataset.collate stacks images into (B, C, S, S) and keeps labels
    # as a list of (S, S) tensors (dropping the name column).
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=ADE20KSegDataset.collate,
        pin_memory=True,
    )


def get_validation_loader(args: argparse.Namespace, spec: ModelSpec) -> DataLoader:
    if args.val_data:
        return get_local_ade20k_validation_loader(
            Path(args.val_data).expanduser().absolute(),
            spec,
            args.batch_size,
            args.frames,
        )
    if args.use_hf_subset:
        return get_hf_ade20k_validation_loader(
            spec,
            args.batch_size,
            args.frames,
            Path(args.data_root).expanduser().absolute(),
        )
    raise SystemExit(
        "Pass --use-hf-subset (download the canonical HuggingFace ADE20K subset) "
        "or --val-data PATH (local ADE20K layout)."
    )


# ---------------------------------------------------------------------------
# Post-process + mIoU
# ---------------------------------------------------------------------------


def _upsample_argmax_to_label_size(
    logits_nchw: torch.Tensor, label_hw: tuple[int, int]
) -> torch.Tensor:
    """Bilinear-upsample logits to the label resolution and argmax.

    Returns an int64 ``(H, W)`` LongTensor of per-pixel class ids.
    """
    if logits_nchw.dim() == 3:
        logits_nchw = logits_nchw.unsqueeze(0)
    up = F.interpolate(
        logits_nchw.float(),
        size=tuple(int(x) for x in label_hw),
        mode="bilinear",
        align_corners=False,
    )
    return up.argmax(dim=1).squeeze(0).long()


# ---------------------------------------------------------------------------
# Metric: ADE20K mIoU via intersect/union accumulation.
#
# ADE20K labels are 1-indexed (0 = background/ignore). The FB linear head was
# trained with mmseg's ``reduce_zero_label=True``, so model channel c maps to
# label c+1. ``_preprocess_label`` shifts labels down by one and folds the
# original 0 (and the 255 pad) into the ignore index, aligning references to
# predictions. mIoU/mean-acc/dice are averaged over classes present in the
# references. This mirrors the reference DINOv2 seg evaluator exactly.
# ---------------------------------------------------------------------------


def _preprocess_label(label: torch.Tensor, ignore_index: int = 255) -> torch.Tensor:
    label = label.clone()
    label[label == ignore_index] += 1
    label -= 1
    label[label == -1] = ignore_index
    return label


def _intersect_and_union(
    pred: torch.Tensor,
    label: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255,
    reduce_zero_label: bool = True,
) -> torch.Tensor:
    if reduce_zero_label:
        label = _preprocess_label(label, ignore_index)
    mask = label != ignore_index
    pred = pred[mask].float()
    label = label[mask].float()
    intersect = pred[pred == label]
    area_intersect = torch.histc(intersect, bins=num_classes, min=0, max=num_classes - 1)
    area_pred = torch.histc(pred, bins=num_classes, min=0, max=num_classes - 1)
    area_label = torch.histc(label, bins=num_classes, min=0, max=num_classes - 1)
    area_union = area_pred + area_label - area_intersect
    return torch.stack([area_intersect, area_union, area_pred, area_label])


def _aggregate_metrics(all_results: list[torch.Tensor]) -> dict:
    results = torch.stack(all_results)
    total_i = results[:, 0].sum(0)
    total_u = results[:, 1].sum(0)
    total_pred = results[:, 2].sum(0)
    total_label = results[:, 3].sum(0)

    iou = total_i / (total_u + 1e-10)
    valid = total_label > 0
    class_acc = total_i / (total_label + 1e-10)
    dice = 2 * total_i / (total_pred + total_label + 1e-10)
    return {
        "mIoU": iou[valid].mean().item() * 100,
        "pixel_acc": (total_i.sum() / (total_label.sum() + 1e-10)).item() * 100,
        "mean_acc": class_acc[valid].mean().item() * 100,
        "mean_dice": dice[valid].mean().item() * 100,
        "iou_per_class": iou.cpu().numpy(),
        "num_valid_classes": int(valid.sum().item()),
    }


# ---------------------------------------------------------------------------
# Inference loops
# ---------------------------------------------------------------------------


def _iter_torch_logits(
    model: torch.nn.Module,
    dataloader: DataLoader,
    device: str,
) -> Iterator[tuple[torch.Tensor, list[torch.Tensor]]]:
    """Yield ``(logits_NCHW_on_cpu, list_of_label_tensors)`` per batch."""
    model.eval()
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", unit="batch"):
            images = images.to(device)
            outputs = model(images)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            yield outputs.detach().to(torch.float32).cpu(), labels


def _iter_rt2_logits(
    axm_path: str,
    spec: ModelSpec,
    dataloader: DataLoader,
    ncores: int | None = None,
) -> Iterator[tuple[torch.Tensor, list[torch.Tensor]]]:
    """Yield ``(logits_NCHW_on_cpu, [label])`` per frame via runtime2 ``op.load``.

    ``op.load`` handles input quant/pack and output depad/dequant/align
    internally, so no host-side packing or dequant is needed here:
    ``out`` already arrives as the dequantized, ONNX-shaped seg logits
    ``(1, num_classes, H/p, W/p)``. Frames are streamed (pipelined across the
    allocated cores) and yielded in source order, so labels stay in sync.
    """
    patch = spec.input_size // spec.patch_size
    inference = rt2_op.load(axm_path, core_allocation=ncores)

    # Materialize (image, label) pairs so a plain list feeds the stream source
    # while labels stay aligned with the in-order results.
    pairs: list[tuple[np.ndarray, np.ndarray]] = []
    for images, labels in dataloader:
        for i in range(images.shape[0]):
            pairs.append((images[i : i + 1].numpy(), labels[i]))
    images_source = [img for img, _ in pairs]
    labels_iter = (lbl for _, lbl in pairs)

    stream = inference.stream(images_source, max_in_flight=ncores)
    for (_, out), label in tqdm(
        zip(stream, labels_iter), total=len(pairs), desc="Evaluating (AIPU)", unit="frame"
    ):
        arr = (
            out.detach().cpu().numpy().astype(np.float32)
            if isinstance(out, torch.Tensor)
            else np.asarray(out, dtype=np.float32)
        )
        if arr.ndim == 4:
            arr = arr[0]
        if arr.ndim != 3:
            raise RuntimeError(f"Unexpected runtime2 seg output shape {arr.shape}")
        logits = np.ascontiguousarray(arr[: spec.num_classes, :patch, :patch])
        yield torch.from_numpy(logits).unsqueeze(0), [label]


def evaluate(
    logits_iter: Iterator[tuple[torch.Tensor, list[torch.Tensor]]],
    spec: ModelSpec,
    dump_logits: Path | None,
) -> dict:
    all_results: list[torch.Tensor] = []
    pred_dump: list[np.ndarray] = []
    label_dump: list[np.ndarray] = []
    for batch_logits, labels in logits_iter:
        for i, label in enumerate(labels):
            label = label.cpu()
            pred = _upsample_argmax_to_label_size(batch_logits[i], label.shape[-2:]).cpu()
            all_results.append(_intersect_and_union(pred, label, num_classes=spec.num_classes))
            if dump_logits is not None:
                pred_dump.append(pred.numpy().astype(np.int16))
                label_dump.append(label.numpy().astype(np.int16))
    if not all_results:
        raise SystemExit("No samples were evaluated (empty dataloader).")
    result = _aggregate_metrics(all_results)
    if dump_logits is not None:
        _save_predictions_dump(dump_logits, pred_dump, label_dump)
    return result


def _save_predictions_dump(
    dump_path: Path,
    preds: list[np.ndarray],
    labels: list[np.ndarray],
) -> None:
    if not preds:
        LOG.warning(f"No predictions to dump to {dump_path}")
        return
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    # Predictions/labels may have heterogeneous shapes; store as object array.
    np.savez(
        dump_path,
        preds=np.array(preds, dtype=object),
        labels=np.array(labels, dtype=object),
    )
    LOG.info(f"Saved per-sample dump: {dump_path} ({len(preds)} samples)")


# ---------------------------------------------------------------------------
# Mode entry points
# ---------------------------------------------------------------------------


def _run_torch(
    model_dir: Path | None,
    spec: ModelSpec,
    device: str,
    args: argparse.Namespace,
    dataloader: DataLoader,
) -> dict:
    LOG.info("Building FP32 seg model from registry...")
    model = build_seg_model(spec, head_weights_path=args.head_weights_path)
    model.to(device)
    return evaluate(
        _iter_torch_logits(model, dataloader, device),
        spec,
        Path(args.dump_logits).expanduser().absolute() if args.dump_logits else None,
    )


def _run_quantized(
    model_dir: Path,
    spec: ModelSpec,
    device: str,
    args: argparse.Namespace,
    dataloader: DataLoader,
) -> dict:
    model_path = model_dir / "model.ptgraph"
    if not model_path.exists():
        raise SystemExit(
            f"Model file not found: {model_path}\n"
            f"Run deploy_seg.py first to produce model.ptgraph"
        )
    LOG.info(f"Loading quantized model from {model_path}")
    # Use load_fx_graphmodule directly (not axmo's public `load_model`)
    # so we can force `map_location='cpu'` and skip the AxMO version-tag
    # check. This lets us load both our own deploy_seg.py output AND
    # untagged ptgraphs from upstream artifact stashes
    # (e.g. ptq_dinov2_seg.axm on s3 has no version tag).
    # AxMO fake-quant FX graphs evaluate correctly only on CPU; pin it even
    # when a GPU is present.
    model = load_fx_graphmodule(str(model_path), map_location="cpu")
    model.to("cpu")
    return evaluate(
        _iter_torch_logits(model, dataloader, "cpu"),
        spec,
        Path(args.dump_logits).expanduser().absolute() if args.dump_logits else None,
    )


def _run_aipu(
    model_dir: Path | None,
    spec: ModelSpec,
    device: str,
    args: argparse.Namespace,
    dataloader: DataLoader,
) -> dict:
    if args.axm:
        axm_path = Path(args.axm).expanduser().absolute()
    elif model_dir is not None:
        axm_path = model_dir / "model.axm"
    else:
        raise SystemExit("Provide --axm or a model_dir for torch-aipu mode")
    if not axm_path.exists():
        raise SystemExit(f"Compiled model not found: {axm_path}")
    return evaluate(
        _iter_rt2_logits(str(axm_path), spec, dataloader, ncores=args.ncores),
        spec,
        Path(args.dump_logits).expanduser().absolute() if args.dump_logits else None,
    )


def _print_results(mode: str, device: str, result: dict, spec: ModelSpec) -> None:
    mode_display = {
        "torch": "FP32 (torch)",
        "quantized": "Quantized (INT8, AxMO CPU)",
        "torch-aipu": "Metis",
    }
    print("\n" + "=" * 60)
    print("Evaluation Results")
    print("=" * 60)
    print(f"Mode      : {mode_display.get(mode, mode)}")
    print(f"Device    : {device}")
    print(f"Dataset   : {spec.dataset}")
    print(f"Classes   : {spec.num_classes} (reduce_zero_label=True; label 0 ignored)")
    print("-" * 60)
    print(f"mIoU      : {float(result['mIoU']):.2f}%")
    print(f"Pixel acc : {float(result['pixel_acc']):.2f}%")
    print(f"Mean acc  : {float(result['mean_acc']):.2f}%")
    print(f"Mean dice : {float(result['mean_dice']):.2f}%")
    print(f"Valid cls : {result['num_valid_classes']}/{spec.num_classes}")
    print("=" * 60)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate accuracy of seg models on ADE20K",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "model_dir",
        nargs="?",
        type=str,
        default=None,
        help="Path to model folder containing model.ptgraph and config.yaml",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for inference (default: auto-detect GPU/CPU)",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="quantized",
        choices=["torch", "quantized", "torch-aipu"],
        metavar="MODE",
        help=(
            "Inference mode (default: quantized). One of: "
            "torch (FP32 baseline) | quantized (AxMO INT8 reference) | "
            "torch-aipu (Metis hardware)."
        ),
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Number of frames to evaluate (default: all)",
    )
    parser.add_argument(
        "--use-hf-subset",
        action="store_true",
        help=(
            "Download the canonical HuggingFace ADE20K subset and evaluate on "
            "its validation split (used when --val-data is not given)."
        ),
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to local ADE20K root with images/validation + annotations/validation",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=config.default_data_root(),
        metavar="PATH",
        help="Download/extract dir for --use-hf-subset (default: framework data root)",
    )
    parser.add_argument(
        "--axm",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to compiled .axm (torch-aipu mode; overrides model_dir/model.axm)",
    )
    parser.add_argument(
        "--ncores",
        type=int,
        default=None,
        metavar="N",
        help="Number of AIPU cores for runtime2 op.load (torch-aipu mode). "
        "Default: None (equal share across all cores).",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        metavar="NAME",
        help="Override the model name from config.yaml (must match MODEL_REGISTRY)",
    )
    parser.add_argument(
        "--head-weights-path",
        type=str,
        default=None,
        metavar="FILE",
        help="Override FB seg head .pth URL (only used in --mode torch)",
    )
    parser.add_argument(
        "--dump-logits",
        type=str,
        default=None,
        metavar="PATH",
        help="Save per-sample predictions + labels as np.savez(PATH, preds, labels)",
    )
    args = parser.parse_args()

    logging_utils.configure_logging(config.LoggingConfig())

    device = get_device(args.device)

    if args.mode == "quantized" and args.model_dir is None:
        raise SystemExit("Provide a model_dir for --mode quantized (must contain model.ptgraph)")
    if args.mode == "torch-aipu" and args.model_dir is None and args.axm is None:
        raise SystemExit("Provide model_dir or --axm for --mode torch-aipu")

    model_dir = Path(args.model_dir).expanduser().absolute() if args.model_dir else None
    if model_dir is not None and not model_dir.is_dir():
        raise SystemExit(f"Model directory not found: {model_dir}")

    model_config = load_config(model_dir)
    spec = resolve_spec(model_config, args.model_name)
    resolved_name = args.model_name or (model_config or {}).get("model_name")
    LOG.info(f"Mode: {args.mode}")
    _log_model_spec(resolved_name, spec)

    LOG.info(
        f"Preprocessing: square resize to {spec.input_size}x{spec.input_size} "
        f"(built into ADE20KSegDataset)"
    )

    dataloader = get_validation_loader(args, spec)

    if args.mode == "torch":
        result = _run_torch(model_dir, spec, device, args, dataloader)
    elif args.mode == "quantized":
        result = _run_quantized(model_dir, spec, device, args, dataloader)
    else:
        result = _run_aipu(model_dir, spec, device, args, dataloader)

    _print_results(args.mode, device, result, spec)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.exit_with_error_log()
    except logging_utils.UserError as e:
        LOG.exit_with_error_log(e.format())
    except Exception as e:
        LOG.exit_with_error_log(e)

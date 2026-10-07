#!/usr/bin/env python
# Copyright Axelera AI, 2026
"""Example script: Evaluate ViT classification models on ImageNet.

This script validates the accuracy of models built with `axzoo build`
(the compiled model.axm). It supports three inference modes:

  --mode torch       Run FP32 baseline (original HuggingFace model)
  --mode quantized   Run quantized INT8 model (default)
  --mode torch-aipu  Run on Metis hardware

GPU is auto-detected (CUDA when available, CPU otherwise). Use --device
to override (e.g., --device cpu to force CPU even with a GPU).

Preprocessing is read from config.yaml in the model folder so the same
transforms used during calibration are applied at inference time.

Compare FP32 vs quantized accuracy:
    # FP32 baseline
    python inference_huggingface_classifier.py ./build/timm_vit_small_patch16_224 \\
        --mode torch --use-hf-subset

    # Quantized model
    python inference_huggingface_classifier.py ./build/timm_vit_small_patch16_224 \\
        --mode quantized --use-hf-subset

Local ImageNet:
    python inference_huggingface_classifier.py \\
        ./build/timm_vit_small_patch16_224 \\
        --data-root /path/to/datasets

Quick validation with limited samples:
    python inference_huggingface_classifier.py \\
        ./build/timm_vit_small_patch16_224 \\
        --use-hf-subset --frames 100

Debug aid -- dump per-sample logits for offline comparison:
    python inference_huggingface_classifier.py ./build/timm_vit_small_patch16_224 \\
        --mode quantized --use-hf-subset --frames 50 --dump-logits /tmp/q.npz
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

try:
    from axelera.app import config, logging_utils
except ImportError:
    sys.exit("Please activate the Axelera environment with source venv/bin/activate and run again")

import numpy as np
import timm
import time
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from axelera.app.eval_interfaces import ClassificationGroundTruthSample
from axelera.app.meta import ClassificationMeta
from axelera.app.meta.base import AxMeta
from axelera.model_optimizer.api import load_model as axmo_load_model
from ax_evaluators.classification import ClassificationEvaluator

from axelera.runtime2 import op as rt2_op
from axelera.runtime2.profiler import OperatorProfiler

LOG = logging_utils.getLogger(__name__)


# HuggingFace eval-data subset. Lives here (not deploy_huggingface_classifier.py,
# which is removed -- deploy now goes through `axzoo build`) so this inference
# script stays self-contained.
HF_SUBSETS = {
    "imagenet": (
        "piupiuisland/imagenet_subset_100",
        "imagenet_subset.zip",
        "imagenet_subset",
    ),
}


def download_hf_subset(dataset_name: str, data_root: Path) -> Path:
    """Download + extract a HuggingFace dataset subset; return its directory."""
    import zipfile

    from huggingface_hub import hf_hub_download

    if dataset_name not in HF_SUBSETS:
        available = list(HF_SUBSETS.keys())
        raise ValueError(
            f"No HuggingFace subset available for '{dataset_name}'. "
            f"Available HF subsets: {available}. "
            f"Use --data-root without --use-hf-subset to use local dataset."
        )
    repo_id, filename, extract_dir = HF_SUBSETS[dataset_name]
    extract_path = data_root / extract_dir
    if extract_path.exists():
        LOG.info(f"HF subset already exists at {extract_path}")
        return extract_path
    LOG.info(f"Downloading HF subset from {repo_id}...")
    data_root.mkdir(parents=True, exist_ok=True)
    zip_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    LOG.info(f"Extracting to {extract_path}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_root)
    return extract_path


def get_device(specified_device: str | None = None) -> str:
    """Auto-detect best available device or use specified device.

    Args:
        specified_device: If provided, use this device and log it as specified.
                         If None, auto-detect CUDA availability.

    Returns:
        Device string ('cuda' or 'cpu')
    """
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
    """Load preprocessing config from model folder, or None if not found."""
    if model_dir is None:
        return None
    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        return None
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(
    model_dir: Path | None,
    device: str,
    mode: str,
    model_name: str,
    axm_override: str | None = None,
):
    """Load model based on inference mode.

    Args:
        model_dir: Path to the model folder containing model.ptgraph
        device: Device to move model to
        mode: Inference mode ('torch', 'quantized', 'torch-aipu')
        model_name: HuggingFace model name (from config.yaml)

    Returns:
        The loaded model
    """
    if mode == "torch":
        LOG.info(f"Loading original HuggingFace model: {model_name}")
        model = timm.create_model(model_name, pretrained=True)
        model.eval()
        model.to(device)
        return model

    elif mode == "quantized":
        model_path = model_dir / "model.ptgraph"
        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}\n"
                "Model folder must contain model.ptgraph (a quantized "
                "ExportedProgram). axzoo build emits only model.axm; use "
                "--mode torch-aipu with --axm for the compiled model."
            )

        try:
            LOG.info(f"Loading quantized model from {model_path}")
            model = axmo_load_model(str(model_path))
            model.to(device)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}") from e

    elif mode == "torch-aipu":
        if axm_override:
            axm_path = Path(axm_override).expanduser().absolute()
        else:
            if model_dir is None:
                raise ValueError("Provide --axm or a model_dir for torch-aipu mode")
            axm_path = model_dir / "model.axm"
        if not axm_path.exists():
            raise FileNotFoundError(
                f"Compiled model not found: {axm_path}\n"
                "Build it with `axzoo build <config>` or pass --axm directly."
            )
        return str(axm_path)

    else:
        raise ValueError(f"Unknown mode: {mode}")


def create_transform_from_config(preprocessing: dict | None, model_name: str | None = None):
    """Build the timm preprocessing transform from a saved or default config.

    Returns:
        (transform, input_size) where input_size is the (C, H, W) the model
        expects. Use timm's own resolver when no preprocessing dict is saved.
    """
    if preprocessing is None:
        if model_name is None:
            raise ValueError(
                "No config.yaml found and no --model-name given; cannot determine preprocessing"
            )
        LOG.info(f"No config.yaml found; using timm default preprocessing for {model_name}")
        m = timm.create_model(model_name, pretrained=False)
        preprocessing = timm.data.resolve_model_data_config(m)

    transform = timm.data.create_transform(**preprocessing, is_training=False)
    input_size = preprocessing["input_size"]
    LOG.info(f"Model input size: {input_size[1]}x{input_size[2]}")
    LOG.info(
        f"Preprocessing: mean={preprocessing['mean']}, std={preprocessing['std']}, "
        f"crop_pct={preprocessing.get('crop_pct')}"
    )
    return transform, input_size


def get_hf_imagenet_validation_loader(
    data_root: Path,
    transform,
    batch_size: int,
    frames: int | None,
) -> DataLoader:
    """Get validation dataloader from HuggingFace ImageNet subset with labels."""
    subset_path = download_hf_subset("imagenet", data_root)

    train_path = subset_path / "train"
    if not train_path.exists():
        train_path = subset_path

    dataset = ImageFolder(root=str(train_path), transform=transform)
    LOG.info(f"Loaded HF ImageNet subset with {len(dataset)} images")

    if frames is not None and frames < len(dataset):
        dataset = Subset(dataset, list(range(frames)))
        LOG.info(f"Using first {frames} samples for evaluation")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


def get_imagenet_validation_loader(
    data_root: Path,
    transform,
    batch_size: int,
    frames: int | None,
) -> DataLoader:
    """Get validation dataloader for local ImageNet with labels."""
    from ax_datasets.torchvision import ImageNet

    dataset_root = data_root / "ImageNet"
    dataset = ImageNet(
        transform=transform,
        root=dataset_root,
        args={"split": "val"},
    )
    LOG.info(f"Loaded ImageNet validation set with {len(dataset)} images")

    if frames is not None and frames < len(dataset):
        dataset = Subset(dataset, list(range(frames)))
        LOG.info(f"Using first {frames} samples for evaluation")

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )


def get_validation_loader(
    data_root: Path,
    transform,
    batch_size: int,
    frames: int | None,
    use_hf_subset: bool,
) -> DataLoader:
    """Get validation dataloader for ImageNet.

    `--use-hf-subset` selects the HuggingFace imagenet_subset_100 (sized by
    `--frames`); the default path uses local/S3 ImageNet and honors the
    `AXELERA_S3_AVAILABLE=subset` env. The two subset levers do not compose.
    """
    if use_hf_subset:
        import os

        if os.environ.get("AXELERA_S3_AVAILABLE") == "subset":
            LOG.warning(
                "AXELERA_S3_AVAILABLE=subset has no effect with --use-hf-subset; "
                "the HF subset path is sized by --frames only."
            )
        return get_hf_imagenet_validation_loader(data_root, transform, batch_size, frames)
    return get_imagenet_validation_loader(data_root, transform, batch_size, frames)


def get_custom_validation_loader(
    val_data_path: Path,
    transform,
    batch_size: int,
    frames: int | None,
    label_map: dict[str, int] | None = None,
    num_classes_hint: int | None = None,
) -> tuple[DataLoader, int]:
    """Get validation dataloader from custom ImageFolder dataset.

    Args:
        val_data_path: Root directory in ImageFolder format
            (one subdir per class).
        transform: timm preprocessing transform.
        batch_size: Inference batch size.
        frames: Optional cap on total samples to evaluate.
        label_map: Optional mapping from folder name to model class index.
            When provided, overrides ImageFolder's default alphabetic
            ``class_to_idx`` so the targets emitted by the loader align
            with the model's training-time class indexing (e.g. timm
            ImageNet-21k uses WNID-sorted order, not alphabetic).
        num_classes_hint: Optional total class count (e.g. 21843 for
            ImageNet-21k). Defaults to the number of discovered classes
            when omitted. Use when only a subset of classes is present
            in ``val_data_path`` but you want top-k computed against
            the full label space.

    Returns:
        Tuple of (dataloader, num_classes).
    """
    if label_map is None:
        dataset = ImageFolder(root=str(val_data_path), transform=transform)
    else:
        # ImageFolder normally assigns class_to_idx alphabetically. Override.
        dataset = ImageFolder(root=str(val_data_path), transform=transform)
        unknown = [c for c in dataset.classes if c not in label_map]
        if unknown:
            raise ValueError(
                f"--label-map missing entries for {len(unknown)} folders, e.g. {unknown[:5]}"
            )
        old_class_to_idx = dict(dataset.class_to_idx)
        new_class_to_idx = {c: int(label_map[c]) for c in dataset.classes}
        dataset.class_to_idx = new_class_to_idx
        # ImageFolder.samples is List[Tuple[path, idx_old]] built at __init__;
        # rewrite to use the new indices.
        old_to_new = {old_class_to_idx[c]: new_class_to_idx[c] for c in dataset.classes}
        dataset.samples = [(p, old_to_new[i]) for (p, i) in dataset.samples]
        dataset.targets = [old_to_new[i] for i in dataset.targets]
        LOG.info(
            f"Applied label_map: {len(new_class_to_idx)} folders remapped "
            f"(e.g. {next(iter(new_class_to_idx.items()))})"
        )

    num_classes = int(num_classes_hint) if num_classes_hint is not None else len(dataset.classes)
    LOG.info(
        f"Loaded custom validation dataset: {len(dataset)} images, "
        f"{len(dataset.classes)} folder classes, top-k over {num_classes} model classes"
    )
    if len(dataset.classes) <= 32:
        LOG.info(f"Folders: {dataset.classes}")

    if frames is not None and frames < len(dataset):
        dataset = Subset(dataset, list(range(frames)))
        LOG.info(f"Using first {frames} samples for evaluation")

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    return dataloader, num_classes


def create_classification_meta(
    predictions: torch.Tensor,
    label: int,
    num_classes: int,
    image_id: str,
    top_k: int = 5,
) -> ClassificationMeta:
    """Create ClassificationMeta with predictions and ground truth."""
    top_k_scores, top_k_indices = torch.topk(predictions, k=min(top_k, predictions.shape[0]))

    ground_truth = ClassificationGroundTruthSample(class_id=label)
    container = AxMeta(image_id=image_id, ground_truth=ground_truth)

    meta = ClassificationMeta(num_classes=num_classes)
    meta.add_result(
        top_k_indices.cpu().tolist(),
        top_k_scores.cpu().tolist(),
    )
    meta.set_container_meta(container)

    return meta


def evaluate(
    model,
    dataloader: DataLoader,
    device: str,
    num_classes: int,
    mode: str,
    dump_logits: Path | None = None,
    perf_dump: Path | None = None,
    ncores: int | None = None,
) -> None:
    """Run evaluation and print accuracy metrics."""
    evaluator = ClassificationEvaluator(top_k=5)

    total_samples = len(dataloader.dataset)
    LOG.info(f"Starting evaluation on {total_samples} samples...")

    if mode == "torch-aipu":
        _evaluate_aipu(
            model,
            dataloader,
            num_classes,
            evaluator,
            dump_logits,
            perf_dump=perf_dump,
            ncores=ncores,
        )
    else:
        _evaluate_torch(model, dataloader, device, num_classes, evaluator, dump_logits)

    _print_results(mode, device, evaluator)


def _save_logits_dump(
    dump_path: Path,
    logits_list: list[np.ndarray],
    labels_list: list[int],
) -> None:
    if not logits_list:
        LOG.warning(f"No logits to dump to {dump_path}")
        return
    logits = np.stack(logits_list, axis=0).astype(np.float32)
    labels = np.asarray(labels_list, dtype=np.int64)
    dump_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(dump_path, logits=logits, labels=labels)
    LOG.info(f"Saved logits dump: {dump_path} (logits={logits.shape}, labels={labels.shape})")


def _evaluate_torch(
    model,
    dataloader: DataLoader,
    device: str,
    num_classes: int,
    evaluator,
    dump_logits: Path | None = None,
) -> None:
    """Evaluate a torch / AxMO-loaded model."""
    sample_idx = 0
    logits_list: list[np.ndarray] = []
    labels_list: list[int] = []
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            images, labels = batch[0], batch[1]
            inputs = images.to(device)
            outputs = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            for i in range(outputs.shape[0]):
                label = labels[i].item() if isinstance(labels, torch.Tensor) else labels[i]
                meta = create_classification_meta(
                    outputs[i],
                    label,
                    num_classes,
                    image_id=f"sample_{sample_idx}",
                    top_k=5,
                )
                evaluator.process_meta(meta)
                if dump_logits is not None:
                    logits_list.append(
                        outputs[i].detach().cpu().numpy().astype(np.float32)[:num_classes]
                    )
                    labels_list.append(int(label))
                sample_idx += 1
    if dump_logits is not None:
        _save_logits_dump(dump_logits, logits_list, labels_list)


def _evaluate_aipu(
    axm_path: str,
    dataloader: DataLoader,
    num_classes: int,
    evaluator,
    dump_logits: Path | None,
    perf_dump: Path | None = None,
    ncores: int | None = None,
) -> None:
    """Drive an .axm via op.load using the scheduler's stream API.

    When ncores=1 the scheduler runs single-threaded (profiler captures per-stage
    timings). When ncores>1 (or None = all cores) frames are pipelined across cores
    for throughput; per-stage profiling is unavailable but end-to-end FPS is shown.
    """
    inference = rt2_op.load(axm_path, core_allocation=ncores)

    # Collect (image_array, label) pairs from the dataloader upfront so we can
    # feed a plain list as the stream source while keeping labels in sync.
    t_mat0 = time.perf_counter()
    pairs: list[tuple[np.ndarray, int]] = []
    for batch in dataloader:
        images, labels = batch[0], batch[1]
        for i in range(images.shape[0]):
            img = images[i : i + 1].numpy()
            lbl = int(labels[i].item() if isinstance(labels, torch.Tensor) else labels[i])
            pairs.append((img, lbl))

    images_source = [img for img, _ in pairs]
    labels_iter = iter(lbl for _, lbl in pairs)
    materialize_s = time.perf_counter() - t_mat0

    # results holds (logits_array, label) collected during the tight stream loop;
    # accuracy evaluation runs afterwards so it doesn't stall the scheduler deque.
    results: list[tuple[np.ndarray, int]] = []
    frame_times: list[float] = []
    profiler = OperatorProfiler()
    t_prev = time.perf_counter()
    t_stream0 = t_prev
    try:
        with profiler:
            stream = inference.stream(images_source, max_in_flight=ncores)
            for (img, out), label in tqdm(
                zip(stream, labels_iter),
                total=len(pairs),
                desc="Evaluating",
                unit="frame",
            ):
                t_now = time.perf_counter()
                frame_times.append(t_now - t_prev)
                t_prev = t_now
                arr = (
                    out.detach().cpu().numpy()
                    if isinstance(out, torch.Tensor)
                    else np.asarray(out, dtype=np.float32)
                )
                results.append((arr.reshape(-1)[:num_classes], label))
    finally:
        close = getattr(inference, "close", None)
        if close is not None:
            close()
    stream_s = time.perf_counter() - t_stream0

    n_frames = len(pairs)
    stream_fps = n_frames / stream_s if stream_s > 0 else 0.0
    per_frame = f" ({materialize_s / n_frames * 1e3:.2f} ms/frame)" if n_frames else ""
    print(f"  [timing] dataloader materialize: {materialize_s:.3f}s{per_frame}")
    print(
        f"  [timing] stream inference:      {stream_s:.3f}s for {n_frames} frames "
        f"-> {stream_fps:.1f} FPS (ncores={ncores})"
    )

    _print_op_load_perf(profiler, frame_times, axm_path, perf_dump)

    # Post-processing: accuracy evaluation and optional logits dump (outside timing)
    logits_list: list[np.ndarray] = []
    labels_list_out: list[int] = []
    for sample_idx, (logits, label) in enumerate(results):
        meta = create_classification_meta(
            torch.from_numpy(logits),
            label,
            num_classes,
            image_id=f"sample_{sample_idx}",
            top_k=5,
        )
        evaluator.process_meta(meta)
        if dump_logits is not None:
            logits_list.append(logits.copy())
            labels_list_out.append(label)
    if dump_logits is not None:
        _save_logits_dump(dump_logits, logits_list, labels_list_out)


def _aggregate_profiler_records(profiler: OperatorProfiler) -> dict[str, dict[str, float]]:
    """Group profiler records by qualified name; report mean / stddev / min / max / count.

    The qualified name, not the operator name, is what identifies a stage: a cascade runs
    an `AxRuntimeModel` and a `Dequant` per model, and a model's preamble and postamble
    are both an `OnnxModel`, so grouping on the operator name alone sums unrelated stages
    into one row (and into one `--perf-dump` key).
    """
    grouped: dict[str, list[float]] = {}
    for rec in profiler.records:
        grouped.setdefault(rec.qualified_name, []).append(rec.elapsed_seconds * 1000.0)
    stats: dict[str, dict[str, float]] = {}
    for name, ts in grouped.items():
        stats[name] = {
            "count": len(ts),
            "mean_ms": statistics.fmean(ts),
            "stddev_ms": statistics.pstdev(ts) if len(ts) > 1 else 0.0,
            "min_ms": min(ts),
            "max_ms": max(ts),
            "total_ms": sum(ts),
        }
    return dict(sorted(stats.items(), key=lambda kv: kv[1]["total_ms"], reverse=True))


def _print_op_load_perf(
    profiler: OperatorProfiler,
    frame_times: list[float],
    axm_path: str,
    perf_dump: Path | None,
) -> None:
    """Print per-stage timing + end-to-end FPS for the op.load path."""
    stats = _aggregate_profiler_records(profiler)
    if not stats and not frame_times:
        return
    n_frames = len(frame_times)
    total_s = sum(frame_times)
    mean_frame_ms = (total_s / n_frames) * 1000.0 if n_frames else 0.0
    stddev_frame_ms = statistics.pstdev([t * 1000.0 for t in frame_times]) if n_frames > 1 else 0.0
    fps = (n_frames / total_s) if total_s > 0 else 0.0

    print("\n" + "=" * 64)
    print(f"Per-stage latency (op.load path, {n_frames} frames)")
    print(f"axm: {axm_path}")
    print("=" * 64)
    print(f"{'stage':<32} {'count':>6} {'mean_ms':>10} {'sigma':>8} {'min':>8} {'max':>8}")
    for name, s in stats.items():
        print(
            f"{name:<32} {int(s['count']):>6d} {s['mean_ms']:>10.3f} "
            f"{s['stddev_ms']:>8.3f} {s['min_ms']:>8.3f} {s['max_ms']:>8.3f}"
        )
    print("-" * 64)
    print(
        f"end-to-end:                      {n_frames:>6d} {mean_frame_ms:>10.3f} "
        f"{stddev_frame_ms:>8.3f}"
    )
    print(f"FPS: {fps:.2f}")
    print("=" * 64)

    if perf_dump is not None:
        perf_dump.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "axm_path": str(axm_path),
            "n_frames": n_frames,
            "mean_frame_ms": mean_frame_ms,
            "stddev_frame_ms": stddev_frame_ms,
            "fps": fps,
            "stages": stats,
        }
        perf_dump.write_text(json.dumps(payload, indent=2))
        LOG.info(f"Wrote per-stage perf dump: {perf_dump}")


def _print_results(mode: str, device: str, evaluator) -> None:
    result = evaluator.collect_metrics()
    mode_display = {
        "torch": "FP32 (torch)",
        "quantized": "Quantized (INT8)",
        "torch-aipu": "Metis",
    }
    print("\n" + "=" * 50)
    print("Evaluation Results")
    print("=" * 50)
    print(f"Mode:   {mode_display.get(mode, mode)}")
    print(f"Device: {device}")
    print("-" * 50)

    top1 = result.metrics_result["accuracy-top-1"]["average"]
    print(f"Top-1 Accuracy: {top1 * 100:.2f}%")

    try:
        top5 = result.metrics_result["accuracy-top-5"]["average"]
        print(f"Top-5 Accuracy: {top5 * 100:.2f}%")
    except KeyError:
        pass

    print("=" * 50)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate accuracy of quantized classification models (example script)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "model_dir",
        nargs="?",
        type=str,
        default=None,
        help=(
            "Path to model folder containing model.ptgraph and config.yaml "
            "(optional when --axm is given)"
        ),
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for inference (default: 1, must match export batch size)",
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
        help="Inference mode (default: quantized). One of: "
        "torch (FP32 baseline) | quantized (INT8 reference) | "
        "torch-aipu (Metis hardware).",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Number of frames to evaluate (default: all)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=config.default_data_root(),
        metavar="PATH",
        help="Dataset directory (default: ./data)",
    )
    parser.add_argument(
        "--use-hf-subset",
        action="store_true",
        help="Download and use HuggingFace dataset subset for evaluation",
    )
    parser.add_argument(
        "--val-data",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Path to custom validation data in ImageFolder format "
            "(overrides --use-hf-subset and ImageNet)"
        ),
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of classes (auto-detected from ImageFolder if not specified)",
    )
    parser.add_argument(
        "--axm",
        type=str,
        default=None,
        metavar="FILE",
        help="Path to compiled .axm model file (torch-aipu mode; overrides model_dir/model.axm)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=None,
        metavar="NAME",
        help="timm model name for preprocessing defaults when no config.yaml is available",
    )
    parser.add_argument(
        "--dump-logits",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Save per-sample logits and labels as np.savez(PATH, logits, labels) "
            "for offline bisection"
        ),
    )
    parser.add_argument(
        "--perf-dump",
        type=str,
        default=None,
        metavar="PATH",
        help="Write per-stage timing JSON to PATH (torch-aipu / op.load path only). "
        "Captures pre_amble / NchwToNhwc / Quant / Pad / chip / Depad / Dequant / "
        "post_amble means + variances and end-to-end FPS.",
    )
    parser.add_argument(
        "--label-map",
        type=str,
        default=None,
        metavar="JSON",
        help=(
            "Path to a JSON file mapping ImageFolder subfolder names to model "
            "class indices (e.g. WNID -> timm-21k index). Use with --val-data "
            "when the val set's folder ordering does not match the model's "
            "trained class indexing. Required for accurate top-k on "
            "ImageNet-21k val (timm uses WNID-sorted order, not the alphabetic "
            "order that ImageFolder assigns by default)."
        ),
    )
    parser.add_argument(
        "--ncores",
        type=int,
        default=None,
        help="Number of AIPU cores to use for inference (torch-aipu mode). "
        "If unset, uses all cores available on the device.",
    )
    args = parser.parse_args()

    logging_utils.configure_logging(config.LoggingConfig())

    # Auto-detect device or use specified
    device = get_device(args.device)

    # Validate args
    if args.mode == "quantized" and args.model_dir is None:
        LOG.exit_with_error_log(
            "Provide a model_dir for --mode quantized (must contain model.ptgraph)"
        )
    if args.mode == "torch-aipu" and args.model_dir is None and args.axm is None:
        LOG.exit_with_error_log("Provide model_dir or --axm for --mode torch-aipu")

    model_dir = Path(args.model_dir).expanduser().absolute() if args.model_dir else None
    if model_dir is not None and not model_dir.is_dir():
        LOG.exit_with_error_log(f"Model directory not found: {model_dir}")

    model_config = load_config(model_dir)
    model_name = (model_config or {}).get("model_name") or args.model_name
    LOG.info(f"Model: {model_name or '(unknown)'}")
    LOG.info(f"Mode: {args.mode}")

    model = load_model(model_dir, device, args.mode, model_name or "", axm_override=args.axm)

    # Create transform from saved preprocessing config (or timm defaults)
    transform, input_size = create_transform_from_config(
        (model_config or {}).get("preprocessing"), model_name
    )

    LOG.info(f"Expected input shape: {input_size}")

    # Get validation data
    data_root = Path(args.data_root).expanduser().absolute()

    if args.val_data:
        # Custom ImageFolder dataset
        val_data_path = Path(args.val_data).expanduser().absolute()
        LOG.info(f"Loading custom validation data from {val_data_path}...")
        label_map: dict[str, int] | None = None
        if args.label_map:
            import json

            with open(args.label_map) as f:
                label_map = json.load(f)
            if not isinstance(label_map, dict):
                LOG.exit_with_error_log(
                    f"--label-map must be a JSON object, got {type(label_map)}"
                )
            LOG.info(
                f"Loaded --label-map from {args.label_map}: {len(label_map)} folder->index entries"
            )
        dataloader, detected_num_classes = get_custom_validation_loader(
            val_data_path,
            transform,
            args.batch_size,
            args.frames,
            label_map=label_map,
            num_classes_hint=args.num_classes,
        )
        num_classes = args.num_classes if args.num_classes else detected_num_classes
        if args.num_classes and args.num_classes != detected_num_classes:
            LOG.info(
                f"Using specified num_classes={num_classes} "
                f"(detected {detected_num_classes} from ImageFolder)"
            )
    else:
        # ImageNet (HuggingFace subset or local)
        num_classes = 1000  # ImageNet
        source = "HuggingFace subset" if args.use_hf_subset else "local"
        LOG.info(f"Loading validation data from ImageNet ({source}, data_root: {data_root})")
        dataloader = get_validation_loader(
            data_root,
            transform,
            args.batch_size,
            args.frames,
            args.use_hf_subset,
        )

    dump_logits = Path(args.dump_logits).expanduser().absolute() if args.dump_logits else None
    perf_dump = Path(args.perf_dump).expanduser().absolute() if args.perf_dump else None

    evaluate(
        model,
        dataloader,
        device,
        num_classes,
        args.mode,
        dump_logits=dump_logits,
        perf_dump=perf_dump,
        ncores=args.ncores,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.exit_with_error_log()
    except logging_utils.UserError as e:
        LOG.exit_with_error_log(e.format())
    except Exception as e:
        LOG.exit_with_error_log(e)

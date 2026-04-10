#!/usr/bin/env python
# Copyright Axelera AI, 2026
"""Example script: Evaluate classification models on ImageNet.

This script demonstrates how to validate accuracy of models created by
deploy_huggingface_classifier.py. It supports multiple inference modes:

  --mode torch      : Run FP32 baseline (original HuggingFace model)
  --mode quantized  : Run quantized INT8 model (default)
  --mode torch-aipu : Run on Metis hardware (future)

GPU is auto-detected - uses CUDA when available, falls back to CPU.
Use --device to override (e.g., --device cpu to force CPU even with GPU).

The script reads preprocessing config from config.yaml in the model folder,
ensuring the same transforms are used as during calibration.

Compare FP32 vs quantized accuracy:
    # First run FP32 baseline
    python inference_huggingface_classifier.py ./build/timm_vit_small_patch16_224 \\
        --mode torch --use-hf-subset

    # Then run quantized model
    python inference_huggingface_classifier.py ./build/timm_vit_small_patch16_224 \\
        --mode quantized --use-hf-subset

Usage with local ImageNet:
    python inference_huggingface_classifier.py \\
        ./build/timm_vit_small_patch16_224 \\
        --data-root /path/to/datasets

Quick validation with limited samples:
    python inference_huggingface_classifier.py \\
        ./build/timm_vit_small_patch16_224 \\
        --use-hf-subset \\
        --frames 100
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    from axelera.app import config, logging_utils
except ImportError:
    sys.exit("Please activate the Axelera environment with source venv/bin/activate and run again")

import timm
import torch
import yaml
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.datasets import ImageFolder
from tqdm import tqdm

from axelera.app.eval_interfaces import ClassificationGroundTruthSample
from axelera.app.meta import ClassificationMeta
from axelera.app.meta.base import AxMeta
from axelera.model_optimizer.api import load_model as axmo_load_model
from ax_evaluators.classification import ClassificationEvaluator

from deploy_huggingface_classifier import download_hf_subset

LOG = logging_utils.getLogger(__name__)


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


def load_config(model_dir: Path) -> dict:
    """Load preprocessing config from model folder.

    Args:
        model_dir: Path to the model folder containing config.yaml

    Returns:
        Config dict with model_name, preprocessing, and quantization settings
    """
    config_path = model_dir / "config.yaml"
    if not config_path.exists():
        raise FileNotFoundError(
            f"Config file not found: {config_path}\n"
            f"Model folder must contain config.yaml created by deploy_huggingface_classifier.py"
        )

    with open(config_path) as f:
        return yaml.safe_load(f)


def load_model(model_dir: Path, device: str, mode: str, model_name: str):
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
                f"Model folder must contain model.ptgraph created by deploy_huggingface_classifier.py"
            )

        try:
            LOG.info(f"Loading quantized model from {model_path}")
            model = axmo_load_model(str(model_path))
            model.to(device)
            return model
        except Exception as e:
            raise RuntimeError(f"Failed to load model from {model_path}: {e}") from e

    elif mode == "torch-aipu":
        raise NotImplementedError(
            "torch-aipu mode coming in future release.\n"
            "This mode will run compiled models on Metis hardware."
        )

    else:
        raise ValueError(f"Unknown mode: {mode}")


def create_transform_from_config(preprocessing: dict):
    """Create preprocessing transform from config.yaml settings.

    Args:
        preprocessing: Dict with mean, std, input_size, interpolation, crop_pct

    Returns:
        Tuple of (transform, input_size)
    """
    mean = preprocessing['mean']
    std = preprocessing['std']
    input_size = preprocessing['input_size']
    crop_pct = preprocessing.get('crop_pct', 0.875)
    interpolation = preprocessing.get('interpolation', 'bicubic')

    img_size = input_size[1]  # (C, H, W) -> H
    crop_size = int(img_size / crop_pct)

    # Map interpolation string to PIL constant
    interp_map = {
        'bilinear': transforms.InterpolationMode.BILINEAR,
        'bicubic': transforms.InterpolationMode.BICUBIC,
        'nearest': transforms.InterpolationMode.NEAREST,
    }
    interp_mode = interp_map.get(interpolation, transforms.InterpolationMode.BICUBIC)

    transform = transforms.Compose(
        [
            transforms.Resize(crop_size, interpolation=interp_mode),
            transforms.CenterCrop(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    LOG.info(f"Model input size: {img_size}x{img_size}")
    LOG.info(f"Preprocessing: mean={mean}, std={std}, crop_pct={crop_pct}")

    return transform, input_size


def get_hf_imagenet_validation_loader(
    data_root: Path,
    transform,
    batch_size: int,
    frames: int | None,
) -> DataLoader:
    """Get validation dataloader from HuggingFace ImageNet subset with labels."""
    subset_path = download_hf_subset('imagenet', data_root)

    train_path = subset_path / 'train'
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

    dataset_root = data_root / 'ImageNet'
    dataset = ImageNet(
        transform=transform,
        root=dataset_root,
        args={'split': 'val'},
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
    """Get validation dataloader for ImageNet."""
    if use_hf_subset:
        return get_hf_imagenet_validation_loader(data_root, transform, batch_size, frames)
    return get_imagenet_validation_loader(data_root, transform, batch_size, frames)


def get_custom_validation_loader(
    val_data_path: Path,
    transform,
    batch_size: int,
    frames: int | None,
) -> tuple[DataLoader, int]:
    """Get validation dataloader from custom ImageFolder dataset.

    Returns:
        Tuple of (dataloader, num_classes)
    """
    dataset = ImageFolder(root=str(val_data_path), transform=transform)
    num_classes = len(dataset.classes)
    LOG.info(f"Loaded custom validation dataset: {len(dataset)} images, {num_classes} classes")
    LOG.info(f"Classes: {dataset.classes}")

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
) -> None:
    """Run evaluation and print accuracy metrics."""
    evaluator = ClassificationEvaluator(top_k=5)

    total_samples = len(dataloader.dataset)
    LOG.info(f"Starting evaluation on {total_samples} samples...")

    sample_idx = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
            # Handle different dataset formats
            if len(batch) == 2:
                images, labels = batch
            else:
                images, labels = batch[0], batch[1]

            images = images.to(device)

            outputs = model(images)
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
                sample_idx += 1

    result = evaluator.collect_metrics()

    # Show mode and device in results for easy comparison
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

    top1 = result.get_metric_result('accuracy-top-1', 'average')
    print(f"Top-1 Accuracy: {top1 * 100:.2f}%")

    try:
        top5 = result.get_metric_result('accuracy-top-5', 'average')
        print(f"Top-5 Accuracy: {top5 * 100:.2f}%")
    except (KeyError, ValueError):
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
        type=str,
        help="Path to model folder containing model.ptgraph and config.yaml",
    )
    parser.add_argument(
        "--batch_size",
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
        help="Inference mode: torch (FP32 baseline), quantized (default), torch-aipu (Metis, future)",
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
        help="Path to custom validation data in ImageFolder format (overrides --use-hf-subset and ImageNet)",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=None,
        help="Number of classes (auto-detected from ImageFolder if not specified)",
    )
    args = parser.parse_args()

    logging_utils.configure_logging(config.LoggingConfig())

    # Auto-detect device or use specified
    device = get_device(args.device)

    # Load config and model from folder
    model_dir = Path(args.model_dir).expanduser().absolute()
    if not model_dir.is_dir():
        LOG.exit_with_error_log(f"Model directory not found: {model_dir}")

    model_config = load_config(model_dir)
    LOG.info(f"Model: {model_config['model_name']}")
    LOG.info(f"Mode: {args.mode}")

    model = load_model(model_dir, device, args.mode, model_config['model_name'])

    # Create transform from saved preprocessing config
    transform, input_size = create_transform_from_config(model_config['preprocessing'])

    LOG.info(f"Expected input shape: {input_size}")

    # Get validation data
    data_root = Path(args.data_root).expanduser().absolute()

    if args.val_data:
        # Custom ImageFolder dataset
        val_data_path = Path(args.val_data).expanduser().absolute()
        LOG.info(f"Loading custom validation data from {val_data_path}...")
        dataloader, detected_num_classes = get_custom_validation_loader(
            val_data_path,
            transform,
            args.batch_size,
            args.frames,
        )
        num_classes = args.num_classes if args.num_classes else detected_num_classes
        if args.num_classes and args.num_classes != detected_num_classes:
            LOG.info(
                f"Using specified num_classes={num_classes} (detected {detected_num_classes} from ImageFolder)"
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

    # Run evaluation
    evaluate(model, dataloader, device, num_classes, args.mode)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.exit_with_error_log()
    except logging_utils.UserError as e:
        LOG.exit_with_error_log(e.format())
    except Exception as e:
        LOG.exit_with_error_log(e)

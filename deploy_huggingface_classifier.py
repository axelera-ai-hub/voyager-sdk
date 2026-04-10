#!/usr/bin/env python
# Copyright Axelera AI, 2026
"""Example script: Deploy HuggingFace classification models using AxMO.

This script demonstrates how to use AxMO (Axelera Model Optimizer) for
post-training quantization. It uses HuggingFace/timm models as examples,
but AxMO works with any PyTorch model - see the documentation for how to
adapt this for your own models.

Workflow:
1. Load a model from HuggingFace (using timm for vision models)
2. Export to torch ExportedProgram
3. Load calibration data (ImageNet)
4. Run AxMO PTQ quantization with SmoothQuant
5. Save model.ptgraph and config.yaml to build folder

Output structure:
    build/<model_name>/
        model.ptgraph       # Quantized ExportedProgram
        config.yaml     # Preprocessing config for inference

Usage with HuggingFace subset (easy, auto-downloads):
    python deploy_huggingface_classifier.py \\
        --model_name timm/vit_small_patch16_224.augreg_in21k_ft_in1k \\
        --use-hf-subset

Usage with local ImageNet:
    python deploy_huggingface_classifier.py \\
        --model_name timm/vit_small_patch16_224.augreg_in21k_ft_in1k \\
        --data-root /path/to/datasets

Custom build output:
    python deploy_huggingface_classifier.py \\
        --model_name timm/resnet50.a1_in1k \\
        --build-root ./my_models
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
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from huggingface_hub import hf_hub_download
import zipfile

import yaml

import axelera.model_optimizer
from axelera.model_optimizer.api import (
    finalize_optimized_model,
    prepare_model_for_optimization,
    save_model,
)
from axelera.model_optimizer.trainer.calibration import calibrate_model

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


# HuggingFace subset configurations
# Maps dataset name to (repo_id, filename, extract_dir_name)
HF_SUBSETS = {
    'imagenet': (
        'piupiuisland/imagenet_subset_100',
        'imagenet_subset.zip',
        'imagenet_subset',
    ),
}


def download_hf_subset(dataset_name: str, data_root: Path) -> Path:
    """Download and extract HuggingFace dataset subset.

    Args:
        dataset_name: Name of the dataset ('imagenet', etc.)
        data_root: Root directory to extract dataset into

    Returns:
        Path to the extracted dataset directory
    """
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

    zip_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type='dataset',
    )

    LOG.info(f"Extracting to {extract_path}...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(data_root)

    return extract_path


def get_hf_imagenet_calibration_loader(
    data_root: Path,
    transform: transforms.Compose,
    batch_size: int,
    num_samples: int,
) -> DataLoader:
    """Get calibration dataloader from HuggingFace ImageNet subset.

    Args:
        data_root: Root directory where HF subset is extracted
        transform: Preprocessing transform from the model's config
        batch_size: Batch size for calibration
        num_samples: Number of calibration samples
    """
    from torchvision.datasets import ImageFolder

    subset_path = download_hf_subset('imagenet', data_root)

    # The HF subset has structure: imagenet_subset/train/class_folders/
    train_path = subset_path / 'train'
    if not train_path.exists():
        # Some subsets might have images directly in the folder
        train_path = subset_path

    dataset = ImageFolder(root=str(train_path), transform=transform)
    LOG.info(f"Loaded HF ImageNet subset with {len(dataset)} images")

    subset = Subset(dataset, list(range(min(num_samples, len(dataset)))))

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        return images

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def get_custom_calibration_loader(
    cal_data_path: Path,
    transform: transforms.Compose,
    batch_size: int,
    num_samples: int,
) -> DataLoader:
    """Get calibration dataloader from custom ImageFolder dataset.

    Args:
        cal_data_path: Path to ImageFolder directory with class subdirectories
        transform: Preprocessing transform from the model's config
        batch_size: Batch size for calibration
        num_samples: Number of calibration samples
    """
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(root=str(cal_data_path), transform=transform)
    LOG.info(f"Loaded custom calibration dataset with {len(dataset)} images from {cal_data_path}")

    subset = Subset(dataset, list(range(min(num_samples, len(dataset)))))

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        return images

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def get_imagenet_calibration_loader(
    data_root: Path,
    transform: transforms.Compose,
    batch_size: int,
    num_samples: int,
) -> DataLoader:
    """Get calibration dataloader for ImageNet.

    Uses the framework's ImageNet dataset wrapper which handles auto-download
    and dataset management.

    Args:
        data_root: Root directory for datasets
        transform: Preprocessing transform from the model's config
        batch_size: Batch size for calibration
        num_samples: Number of calibration samples
    """
    from ax_datasets.torchvision import ImageNet

    dataset_root = data_root / 'ImageNet'
    dataset = ImageNet(
        transform=transform,
        root=dataset_root,
        args={'split': 'train'},
    )

    subset = Subset(dataset, list(range(min(num_samples, len(dataset)))))

    def collate_fn(batch):
        images = torch.stack([item[0] for item in batch])
        return images

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )


def get_calibration_loader(
    dataset_name: str,
    data_root: Path,
    transform: transforms.Compose,
    batch_size: int,
    num_samples: int,
    use_hf_subset: bool = False,
) -> DataLoader:
    """Get calibration dataloader for the specified dataset.

    Args:
        dataset_name: Name of the dataset ('imagenet', etc.)
        data_root: Root directory for datasets (same as --data-root in deploy.py)
        transform: Preprocessing transform from the model's config
        batch_size: Batch size for calibration
        num_samples: Number of calibration samples
        use_hf_subset: If True, download and use HuggingFace subset

    Returns:
        DataLoader with calibration data
    """
    if use_hf_subset:
        if dataset_name == 'imagenet':
            return get_hf_imagenet_calibration_loader(
                data_root, transform, batch_size, num_samples
            )
        else:
            # Will raise ValueError with available HF subsets
            download_hf_subset(dataset_name, data_root)

    if dataset_name == 'imagenet':
        return get_imagenet_calibration_loader(data_root, transform, batch_size, num_samples)
    else:
        raise ValueError(
            f"Dataset '{dataset_name}' not yet supported. " f"Supported datasets: imagenet"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Deploy HuggingFace models to Metis hardware",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="HuggingFace model name (e.g., timm/vit_small_patch16_224.augreg_in21k_ft_in1k)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for calibration (default: 1)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device for calibration (default: auto-detect GPU/CPU)",
    )
    parser.add_argument(
        "--smooth_quant_alpha",
        type=float,
        default=0.5,
        help="SmoothQuant alpha parameter (default: 0.5)",
    )
    parser.add_argument(
        "--num_calibration_samples",
        type=int,
        default=100,
        help="Number of calibration samples (default: 100)",
    )
    parser.add_argument(
        "--build-root",
        type=str,
        default="./build",
        metavar="PATH",
        help="Output directory for quantized model (default: ./build)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=config.default_data_root(),
        metavar="PATH",
        help="Dataset download directory (default: ./data)",
    )
    parser.add_argument(
        "--use-hf-subset",
        action="store_true",
        help="Download and use HuggingFace dataset subset for calibration (available for: imagenet)",
    )
    parser.add_argument(
        "--cal-data",
        type=str,
        default=None,
        metavar="PATH",
        help="Path to custom calibration data in ImageFolder format (overrides --use-hf-subset and ImageNet)",
    )
    args = parser.parse_args()

    logging_utils.configure_logging(config.LoggingConfig())

    # Auto-detect device or use specified
    device = get_device(args.device)

    LOG.info(f"Loading model: {args.model_name}")
    model = timm.create_model(args.model_name, pretrained=True)
    model.eval()

    # Get model's preprocessing config and create transform
    data_config = timm.data.resolve_model_data_config(model)
    img_size = data_config['input_size'][1]  # (C, H, W) -> H
    LOG.info(f"Model input size: {img_size}x{img_size}")
    LOG.info(
        f"Model preprocessing: mean={data_config['mean']}, std={data_config['std']}, crop_pct={data_config.get('crop_pct', 'N/A')}"
    )

    # Create preprocessing transform using the model's config
    transform = timm.data.create_transform(**data_config, is_training=False)
    LOG.info(f"Transform: {transform}")

    # Export to ExportedProgram
    LOG.info("Exporting model to ExportedProgram...")
    dummy_input = torch.ones([args.batch_size, 3, img_size, img_size])
    exported_program = torch.export.export(model, args=(dummy_input,))

    # Get calibration data
    data_root = Path(args.data_root).expanduser().absolute()

    if args.cal_data:
        # Custom ImageFolder dataset
        cal_data_path = Path(args.cal_data).expanduser().absolute()
        LOG.info(f"Loading custom calibration data from {cal_data_path}...")
        cal_dataloader = get_custom_calibration_loader(
            cal_data_path,
            transform,
            args.batch_size,
            args.num_calibration_samples,
        )
    else:
        # ImageNet (HuggingFace subset or local)
        source = "HuggingFace subset" if args.use_hf_subset else "local"
        LOG.info(f"Loading calibration data from ImageNet ({source}, data_root: {data_root})...")
        cal_dataloader = get_calibration_loader(
            'imagenet',
            data_root,
            transform,
            args.batch_size,
            args.num_calibration_samples,
            use_hf_subset=args.use_hf_subset,
        )

    # Run AxMO quantization
    LOG.info("Preparing model for quantization...")
    axmo_config = axelera.model_optimizer.get_default_config(
        generation=axelera.model_optimizer.HardwareGeneration.METIS,
        smooth_quant_alpha=args.smooth_quant_alpha,
    )
    fx_graph_model = axelera.model_optimizer.import_exported_program(exported_program)
    fx_graph_model.to(device)
    prepare_model_for_optimization(fx_graph_model, axmo_config)

    LOG.info("Running calibration...")
    calibrate_model(
        fx_graph_model,
        cal_dataloader,
        progress_bar=True,
        device=torch.device(device),
    )

    LOG.info("Finalizing quantized model...")
    finalize_optimized_model(fx_graph_model, axmo_config)

    # Create output folder structure: build_root/model_name/
    build_root = Path(args.build_root).expanduser().absolute()
    safe_name = args.model_name.replace('/', '_')
    model_dir = build_root / safe_name
    model_dir.mkdir(parents=True, exist_ok=True)

    # Save model.ptgraph
    model_path = model_dir / "model.ptgraph"
    LOG.info(f"Saving quantized model to {model_path}")
    save_model(fx_graph_model, str(model_path))

    # Save config.yaml with preprocessing params for inference
    config_data = {
        'model_name': args.model_name,
        'preprocessing': {
            'mean': list(data_config['mean']),
            'std': list(data_config['std']),
            'input_size': list(data_config['input_size']),
            'interpolation': data_config.get('interpolation', 'bicubic'),
            'crop_pct': data_config.get('crop_pct', 0.875),
        },
        'quantization': {
            'smooth_quant_alpha': args.smooth_quant_alpha,
            'num_calibration_samples': args.num_calibration_samples,
            'batch_size': args.batch_size,
        },
    }
    config_path = model_dir / "config.yaml"
    LOG.info(f"Saving config to {config_path}")
    with open(config_path, 'w') as f:
        yaml.dump(config_data, f, default_flow_style=False)

    LOG.info("Quantization complete!")
    LOG.info(f"Output saved to: {model_dir}/")
    LOG.info("  - model.ptgraph: Quantized ExportedProgram")
    LOG.info("  - config.yaml: Preprocessing config for inference")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.exit_with_error_log()
    except logging_utils.UserError as e:
        LOG.exit_with_error_log(e.format())
    except Exception as e:
        LOG.exit_with_error_log(e)

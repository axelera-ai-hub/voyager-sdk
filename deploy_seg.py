#!/usr/bin/env python
# Copyright Axelera AI, 2026
"""Example script: Deploy ViT-backbone + linear-conv-head semantic
segmentation models to Metis.

This script mirrors ``deploy_huggingface_classifier.py`` for the
segmentation family. The first registered model is DINOv2-S/14 with the
official ADE20K linear head; the same script extends to other DINOv2
sizes, VOC2012 heads, and DINOv3 (linear head, same shape) by adding a
row to ``MODEL_REGISTRY``.

Pipeline:

  facebookresearch/dinov2 (torch.hub) -> backbone
  dl.fbaipublicfiles.com .pth          -> linear seg head
                                       -> assembled nn.Module
                                       -> torch.export
                                       -> AxMO PTQ (SmoothQuant)
                                       -> compile_single_graph -> .axm

The exported model returns patch-resolution logits ``[B, num_classes,
H/patch, W/patch]`` (e.g. ``[1, 150, 37, 37]`` for DINOv2-S/14 at 518).
Bilinear upsample to ``input_size`` happens on the host inside
``inference_seg.py``, NOT inside the compiled ``.axm``.

Output:

    build/<model_name>/
        model.ptgraph   # Quantized ExportedProgram
        config.yaml     # Preprocessing config for inference
        model.axm       # Symlink to the compiled Metis artifact

Usage:

    # Calibrate on the canonical HuggingFace ADE20K subset (downloads
    # piupiuisland/ade20k_subset_200 and calibrates on its training split).
    python deploy_seg.py dinov2_seg_vits14_ade20k_linear --use-hf-subset

    # Calibrate on a local ADE20K set. Pass the ADE20K root containing
    # images/training + annotations/training.
    python deploy_seg.py dinov2_seg_vits14_ade20k_linear \\
        --cal-data /path/to/ADEChallengeData2016

    # Custom calibration directory (plain ImageFolder)
    python deploy_seg.py dinov2_seg_vits14_ade20k_linear \\
        --cal-data /path/to/my-images
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from axelera.app import config, logging_utils
except ImportError:
    sys.exit("Please activate the Axelera environment with source venv/bin/activate and run again")

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from huggingface_hub import hf_hub_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms

from axelera.compiler.alto.compiler.config import HardwareGeneration, Target
from axelera.graph_compiler.api import compile_single_graph
from axelera.graph_compiler.config import CompilerConfig
from axelera.graph_compiler.types import DeviceQuantBoundarySetting, DType, Pipeline, TensorType

import axelera.model_optimizer
from axelera.model_optimizer.api import (
    finalize_optimized_model,
    prepare_model_for_optimization,
    save_model,
)
from axelera.model_optimizer.trainer.calibration import calibrate_model

LOG = logging_utils.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelSpec:
    """One row in ``MODEL_REGISTRY``.

    The backbone must expose ``get_intermediate_layers(x, n=1,
    reshape=True)[0]`` returning a feature map of shape ``(B, embed_dim,
    H/patch, W/patch)`` (DINOv2 / DINOv3 family convention). The head
    ``.pth`` must have keys
    ``decode_head.bn.{weight,bias,running_mean,running_var}`` and
    ``decode_head.conv_seg.{weight,bias}`` (FB linear-head schema).
    """

    backbone_hub_repo: str
    backbone_hub_name: str
    head_url: str
    embed_dim: int
    input_size: int
    patch_size: int
    num_classes: int
    dataset: str
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    interpolation: str = "bilinear"
    backbone_loader_kwargs: dict[str, Any] = field(default_factory=dict)


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "dinov2_seg_vits14_ade20k_linear": ModelSpec(
        backbone_hub_repo="facebookresearch/dinov2",
        backbone_hub_name="dinov2_vits14",
        head_url=(
            "https://dl.fbaipublicfiles.com/dinov2/dinov2_vits14/"
            "dinov2_vits14_ade20k_linear_head.pth"
        ),
        embed_dim=384,
        input_size=518,
        patch_size=14,
        num_classes=150,
        dataset="ade20k",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation="bilinear",
    ),
    "dinov2_seg_vitb14_ade20k_linear": ModelSpec(
        backbone_hub_repo="facebookresearch/dinov2",
        backbone_hub_name="dinov2_vitb14",
        head_url=(
            "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitb14/"
            "dinov2_vitb14_ade20k_linear_head.pth"
        ),
        embed_dim=768,
        input_size=518,
        patch_size=14,
        num_classes=150,
        dataset="ade20k",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation="bilinear",
    ),
    "dinov2_seg_vitl14_ade20k_linear": ModelSpec(
        backbone_hub_repo="facebookresearch/dinov2",
        backbone_hub_name="dinov2_vitl14",
        head_url=(
            "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/"
            "dinov2_vitl14_ade20k_linear_head.pth"
        ),
        embed_dim=1024,
        input_size=518,
        patch_size=14,
        num_classes=150,
        dataset="ade20k",
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
        interpolation="bilinear",
    ),
}


def _resolve_model_spec(model_name: str) -> ModelSpec:
    if model_name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise SystemExit(f"Unknown model '{model_name}'. Available models: {available}")
    return MODEL_REGISTRY[model_name]


# ---------------------------------------------------------------------------
# Model assembly
# ---------------------------------------------------------------------------


def _load_dinov2_backbone(spec: ModelSpec) -> nn.Module:
    """Load a DINOv2-family backbone via torch.hub."""
    LOG.info(
        f"Loading backbone from torch.hub: {spec.backbone_hub_repo} / " f"{spec.backbone_hub_name}"
    )
    backbone = torch.hub.load(
        spec.backbone_hub_repo,
        spec.backbone_hub_name,
        source="github",
        **spec.backbone_loader_kwargs,
    )
    backbone.eval()
    return backbone


def _load_seg_head_state_dict(
    spec: ModelSpec, override_path: str | None = None
) -> dict[str, torch.Tensor]:
    """Fetch the linear seg head .pth state-dict.

    The FB releases are saved in mmsegmentation format -- the actual
    tensors live under a top-level ``state_dict`` key alongside ``meta``
    and ``optimizer``. We unwrap and return the inner dict, whose keys
    are ``decode_head.{bn,conv_seg}.*``.
    """
    if override_path is not None:
        LOG.info(f"Loading seg head weights from override path: {override_path}")
        raw = torch.load(override_path, map_location="cpu", weights_only=False)
    else:
        LOG.info(f"Downloading seg head weights from {spec.head_url}")
        raw = torch.hub.load_state_dict_from_url(
            spec.head_url, map_location="cpu", check_hash=False
        )
    if isinstance(raw, dict) and "state_dict" in raw and isinstance(raw["state_dict"], dict):
        return raw["state_dict"]
    return raw


class _LinearSegHead(nn.Module):
    """Single 1x1 Conv2d. The FB ``BNHead`` (BN + Conv2d in eval mode)
    is folded into one conv at load time, so the exported graph has no
    BN node. This avoids depending on an AxMO MetisAnnotator factory for
    ``aten._native_batch_norm_legit_no_training.default`` (not all AxMO
    builds register it), and is mathematically identical to the original
    BN -> Conv1x1 in eval mode.
    """

    def __init__(self, embed_dim: int, num_classes: int):
        super().__init__()
        self.conv_seg = nn.Conv2d(embed_dim, num_classes, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv_seg(x)

    def load_fb_state_dict(self, fb_state_dict: dict[str, torch.Tensor]) -> None:
        """Port the FB ``decode_head.*`` keys and fold BN into the 1x1 conv.

        In eval mode BN is an affine per-channel transform
        ``y_i = a_i * x_i + b_i`` with
        ``a_i = gamma_i / sqrt(var_i + eps)`` and
        ``b_i = beta_i - a_i * mu_i``. Composing with the 1x1 conv
        ``z_j = sum_i w_ji * y_i + c_j`` gives the fused conv
        ``w'_ji = w_ji * a_i`` and
        ``c'_j = sum_i w_ji * b_i + c_j``.
        """
        bn_keys = (
            "decode_head.bn.weight",
            "decode_head.bn.bias",
            "decode_head.bn.running_mean",
            "decode_head.bn.running_var",
        )
        conv_keys = (
            "decode_head.conv_seg.weight",
            "decode_head.conv_seg.bias",
        )
        required = bn_keys + conv_keys
        missing = [k for k in required if k not in fb_state_dict]
        if missing:
            raise KeyError(
                f"FB seg head .pth missing required keys: {missing}. "
                f"Got keys: {sorted(fb_state_dict.keys())}"
            )

        eps = 1e-5  # mmsegmentation's default BatchNorm2d eps
        gamma = fb_state_dict["decode_head.bn.weight"].float()
        beta = fb_state_dict["decode_head.bn.bias"].float()
        mean = fb_state_dict["decode_head.bn.running_mean"].float()
        var = fb_state_dict["decode_head.bn.running_var"].float()
        w_in = fb_state_dict["decode_head.conv_seg.weight"].float()  # (C_out, C_in, 1, 1)
        b_in = fb_state_dict["decode_head.conv_seg.bias"].float()  # (C_out,)

        a = gamma / torch.sqrt(var + eps)  # (C_in,)
        b = beta - a * mean  # (C_in,)

        fused_w = w_in * a.view(1, -1, 1, 1)
        per_out_bias_term = (w_in.squeeze(-1).squeeze(-1) @ b).view(-1)
        fused_b = b_in + per_out_bias_term

        if fused_w.shape != self.conv_seg.weight.shape:
            raise RuntimeError(
                f"Fused conv weight shape {tuple(fused_w.shape)} does not match "
                f"target {tuple(self.conv_seg.weight.shape)}"
            )
        with torch.no_grad():
            self.conv_seg.weight.copy_(fused_w.to(self.conv_seg.weight.dtype))
            self.conv_seg.bias.copy_(fused_b.to(self.conv_seg.bias.dtype))


class SegModel(nn.Module):
    """Backbone + linear seg head.

    The exported forward returns ``[B, num_classes, H/patch, W/patch]``;
    upsample to input resolution is done on the host post-process inside
    ``inference_seg.py``.
    """

    def __init__(self, backbone: nn.Module, head: _LinearSegHead):
        super().__init__()
        self.backbone = backbone
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # get_intermediate_layers(n=1, reshape=True) -> tuple of one
        # tensor shaped (B, embed_dim, H/patch, W/patch).
        feats = self.backbone.get_intermediate_layers(x, n=1, reshape=True)[0]
        return self.head(feats)


def build_seg_model(spec: ModelSpec, head_weights_path: str | None = None) -> SegModel:
    backbone = _load_dinov2_backbone(spec)
    head = _LinearSegHead(spec.embed_dim, spec.num_classes)
    fb_sd = _load_seg_head_state_dict(spec, override_path=head_weights_path)
    head.load_fb_state_dict(fb_sd)
    model = SegModel(backbone, head)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Device + auxiliaries
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


def _build_seg_transform(spec: ModelSpec) -> transforms.Compose:
    """Resize-then-center-crop transform matching DINOv2 conventions."""
    interp = (
        transforms.InterpolationMode.BILINEAR
        if spec.interpolation == "bilinear"
        else transforms.InterpolationMode.BICUBIC
    )
    return transforms.Compose(
        [
            transforms.Resize(spec.input_size, interpolation=interp),
            transforms.CenterCrop(spec.input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=list(spec.mean), std=list(spec.std)),
        ]
    )


# ---------------------------------------------------------------------------
# ADE20K dataset (with built-in segmentation preprocessing)
#
# Resize-shorter-side -> center-crop -> pad to a fixed square, normalising in
# 0-255 space.
# ---------------------------------------------------------------------------


class ADE20KSegDataset(Dataset):
    """ADE20K ``images/<split>`` + ``annotations/<split>`` layout.

    Yields ``(image_tensor, label_tensor, name)``. Preprocessing is built in --
    there is no separate transform object:

      * image: resize shorter side to ``spec.input_size``, center-crop + pad to
        a square ``(input_size, input_size)``, then ImageNet-normalise in 0-255
        pixel space (identical to torchvision ToTensor + Normalize).
      * label: the same geometry with nearest-neighbor interpolation, padding
        with ``IGNORE_INDEX`` so padded regions are excluded from the metric.

    This is the only preprocessing the model uses (calibration + eval).
    """

    IGNORE_INDEX = 255

    def __init__(
        self,
        root: str,
        spec: ModelSpec,
        split: str = "validation",
        max_samples: int = 0,
    ):
        self.root = Path(root)
        self.resize_size = spec.input_size
        self.mean = torch.tensor([m * 255 for m in spec.mean]).view(3, 1, 1)
        self.std = torch.tensor([s * 255 for s in spec.std]).view(3, 1, 1)
        self.images_dir = self.root / "images" / split
        self.annotations_dir = self.root / "annotations" / split
        if not self.images_dir.is_dir() or not self.annotations_dir.is_dir():
            raise SystemExit(
                f"Expected ADE20K layout {self.images_dir} and {self.annotations_dir}"
            )
        self.image_files = sorted(self.images_dir.glob("*.jpg"))
        if max_samples > 0:
            self.image_files = self.image_files[:max_samples]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, idx: int):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        ann_path = self.annotations_dir / img_path.name.replace(".jpg", ".png")
        annotation = torch.from_numpy(np.array(Image.open(ann_path))).long()
        return self._transform_image(image), self._transform_label(annotation), img_path.name

    # ------------------------------------------------------------------
    # Segmentation preprocessing (shared geometry for image + label)
    # ------------------------------------------------------------------
    def _resize_shorter_side(self, h: int, w: int) -> tuple[int, int]:
        if h < w:
            return self.resize_size, int(w * self.resize_size / h + 0.5)
        return int(h * self.resize_size / w + 0.5), self.resize_size

    def _transform_image(self, image: Image.Image) -> torch.Tensor:
        w, h = image.size
        new_h, new_w = self._resize_shorter_side(h, w)
        image = image.resize((new_w, new_h), Image.BILINEAR)
        t = torch.from_numpy(np.array(image)).permute(2, 0, 1).float()
        t = (t - self.mean) / self.std
        s = self.resize_size
        _, H, W = t.shape
        top = max((H - s) // 2, 0)
        left = max((W - s) // 2, 0)
        t = t[:, top : top + s, left : left + s]
        pad_h = max(s - t.shape[-2], 0)
        pad_w = max(s - t.shape[-1], 0)
        if pad_h or pad_w:
            t = F.pad(t, (0, pad_w, 0, pad_h))
        return t

    def _transform_label(self, label: torch.Tensor) -> torch.Tensor:
        H, W = label.shape
        new_h, new_w = self._resize_shorter_side(H, W)
        label = (
            F.interpolate(label[None, None].float(), size=(new_h, new_w), mode="nearest")
            .squeeze(0)
            .squeeze(0)
            .long()
        )
        s = self.resize_size
        H, W = label.shape
        top = max((H - s) // 2, 0)
        left = max((W - s) // 2, 0)
        label = label[top : top + s, left : left + s]
        pad_h = max(s - label.shape[-2], 0)
        pad_w = max(s - label.shape[-1], 0)
        if pad_h or pad_w:
            label = F.pad(label, (0, pad_w, 0, pad_h), value=self.IGNORE_INDEX)
        return label

    @staticmethod
    def collate(
        batch: list[tuple[torch.Tensor, torch.Tensor, str]],
    ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        """Collate ``(image, label, name)`` items into ``(images_BCHW, [labels])``.

        Images are stacked into a batch tensor; labels are kept as a list (one
        ``(S, S)`` tensor each) so the eval loop can score them per-sample, and
        the name column is dropped.
        """
        images = torch.stack([b[0] for b in batch])
        labels = [b[1] for b in batch]
        return images, labels


def is_ade20k_layout(path: Path, split: str) -> bool:
    return (path / "images" / split).is_dir() and (path / "annotations" / split).is_dir()


# ---------------------------------------------------------------------------
# Calibration loaders
# ---------------------------------------------------------------------------


# Canonical 200-image ADE20K subset on HuggingFace: a single zip with the
# standard ADE20K layout (images/<split> + annotations/<split>), 100 train +
# 100 val. Downloading it makes both deploy (calibration) and inference (eval)
# fully reproducible without the full 20k-image dataset.
HF_ADE20K_SUBSET = (
    "piupiuisland/ade20k_subset_200",  # repo_id
    "ade20k_subset_200.zip",  # filename in the repo
    "ade20k_subset_200",  # top-level dir inside the zip
)


def download_ade20k_subset(data_root: Path) -> Path:
    """Download + extract the canonical ADE20K subset from HuggingFace.

    Returns the extracted ADE20K root (``images/<split>`` +
    ``annotations/<split>``), suitable for the local seg loaders used by both
    ``deploy_seg.py`` (calibration) and ``inference_seg.py`` (eval).
    """
    repo_id, filename, extract_dir = HF_ADE20K_SUBSET
    root = data_root / extract_dir
    if is_ade20k_layout(root, "validation") and is_ade20k_layout(root, "training"):
        LOG.info(f"ADE20K subset already present at {root}")
        return root
    data_root.mkdir(parents=True, exist_ok=True)
    LOG.info(f"Downloading ADE20K subset from {repo_id}...")
    zip_path = hf_hub_download(repo_id=repo_id, filename=filename, repo_type="dataset")
    LOG.info(f"Extracting to {data_root}...")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(data_root)
    return root


def _get_custom_calibration_loader(
    cal_data_path: Path,
    transform: transforms.Compose,
    batch_size: int,
    num_samples: int,
) -> DataLoader:
    """Local ImageFolder calibration loader."""
    from torchvision.datasets import ImageFolder

    dataset = ImageFolder(root=str(cal_data_path), transform=transform)
    LOG.info(f"Loaded custom calibration set with {len(dataset)} images from {cal_data_path}")
    subset = Subset(dataset, list(range(min(num_samples, len(dataset)))))

    def collate_fn(batch: list[tuple[torch.Tensor, int]]) -> torch.Tensor:
        return torch.stack([item[0] for item in batch])

    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True,
    )


class _ImagesOnly(Dataset):
    """Drop (label, name) so ``calibrate_model`` gets a plain image tensor."""

    def __init__(self, base: Dataset):
        self._base = base

    def __len__(self) -> int:
        return len(self._base)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self._base[idx][0]


# Calibration draws from the training split: PTQ ranges must be collected from
# data disjoint from the validation set we report accuracy on, otherwise eval
# images leak into calibration.
CALIBRATION_SPLIT = "training"


def _get_ade20k_seg_calibration_loader(
    ade20k_root: Path,
    spec: ModelSpec,
    batch_size: int,
    num_samples: int,
) -> DataLoader:
    """Calibrate on local ADE20K images using the square seg preprocessing.

    Reuses ``ADE20KSegDataset`` (same built-in geometry as ``inference_seg.py``),
    so the quant ranges are collected from exactly the input distribution the
    model is evaluated on -- this is what makes the chip accuracy match. Only
    the images are read (labels are unused for calibration); they are pulled
    from the ``training`` split so we never calibrate on the validation images
    that accuracy is reported on.
    """
    dataset = ADE20KSegDataset(
        root=str(ade20k_root),
        spec=spec,
        split=CALIBRATION_SPLIT,
        max_samples=num_samples,
    )
    LOG.info(f"Calibration: {len(dataset)} ADE20K {CALIBRATION_SPLIT} images from {ade20k_root}")
    return DataLoader(
        _ImagesOnly(dataset),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=lambda batch: torch.stack(batch),
        pin_memory=True,
    )


def get_calibration_loader(
    spec: ModelSpec,
    args: argparse.Namespace,
    transform: transforms.Compose,
) -> DataLoader:
    """Pick calibration source.

    Priority: ``--cal-data`` (local ADE20K layout or ImageFolder) > the
    canonical HuggingFace ADE20K subset (``--use-hf-subset``). The HF path
    downloads data, so it is gated behind ``--use-hf-subset``; without
    ``--cal-data`` and without that flag there is no source and we error rather
    than download silently. Either way calibration runs on the ``training``
    split with the square seg transform.
    """
    if args.cal_data:
        cal_data_path = Path(args.cal_data).expanduser().absolute()
        if is_ade20k_layout(cal_data_path, CALIBRATION_SPLIT):
            LOG.info(f"Calibrating on local ADE20K (seg transform) from {cal_data_path}...")
            return _get_ade20k_seg_calibration_loader(
                cal_data_path, spec, args.batch_size, args.num_calibration_samples
            )
        LOG.info(f"Loading custom (ImageFolder) calibration data from {cal_data_path}...")
        return _get_custom_calibration_loader(
            cal_data_path, transform, args.batch_size, args.num_calibration_samples
        )
    if not args.use_hf_subset:
        raise SystemExit(
            "No calibration source. Pass --cal-data PATH for a local ADE20K set, "
            "or --use-hf-subset to download the canonical HuggingFace ADE20K subset "
            f"({HF_ADE20K_SUBSET[0]})."
        )
    # Download the canonical ADE20K subset and calibrate on its training split.
    data_root = Path(args.data_root).expanduser().absolute()
    ade20k_root = download_ade20k_subset(data_root)
    return _get_ade20k_seg_calibration_loader(
        ade20k_root, spec, args.batch_size, args.num_calibration_samples
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _log_model_spec(model_name: str, spec: ModelSpec) -> None:
    LOG.info("Resolved model spec:")
    LOG.info(f"  name        : {model_name}")
    LOG.info(f"  backbone    : {spec.backbone_hub_repo} / {spec.backbone_hub_name}")
    LOG.info(f"  head url    : {spec.head_url}")
    LOG.info(f"  input size  : {spec.input_size}x{spec.input_size}")
    LOG.info(f"  patch size  : {spec.patch_size}")
    LOG.info(f"  num classes : {spec.num_classes}")
    LOG.info(f"  dataset     : {spec.dataset}")
    LOG.info(f"  mean / std  : {spec.mean} / {spec.std}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Deploy ViT-backbone + linear-head seg models to Metis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "model_name",
        type=str,
        help=("Registered model name. Available: " + ", ".join(sorted(MODEL_REGISTRY.keys()))),
    )
    parser.add_argument(
        "--batch-size",
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
        "--smooth-quant-alpha",
        type=float,
        default=0.5,
        help="SmoothQuant alpha for the main quant path (default: 0.5)",
    )
    parser.add_argument(
        "--smooth-qk-alpha",
        type=float,
        default=None,
        help=(
            "SmoothQuant alpha for the SDPA QK matmul. None=disabled "
            "(default). Try 0.5 if attention-heavy ViTs lose accuracy."
        ),
    )
    parser.add_argument(
        "--num-calibration-samples",
        type=int,
        default=100,
        help="Number of calibration samples (default: 100)",
    )
    parser.add_argument(
        "--build-root",
        type=str,
        default="./build",
        metavar="PATH",
        help="Output directory for quantized + compiled model (default: ./build)",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=config.default_data_root(),
        metavar="PATH",
        help="Dataset download directory (used for ImageNet fallback; default: ./data)",
    )
    parser.add_argument(
        "--use-hf-subset",
        action="store_true",
        help=(
            "Download the canonical HuggingFace ADE20K subset "
            f"({HF_ADE20K_SUBSET[0]}) and calibrate on its training split. "
            "Required when --cal-data is not given."
        ),
    )
    parser.add_argument(
        "--cal-data",
        type=str,
        default=None,
        metavar="PATH",
        help=(
            "Local calibration source. Either an ADE20K root (images/training "
            "+ annotations/training), calibrated with the square seg "
            "transform, or a plain ImageFolder directory."
        ),
    )
    parser.add_argument(
        "--head-weights-path",
        type=str,
        default=None,
        metavar="FILE",
        help="Override the FB seg head .pth URL with a local file",
    )
    args = parser.parse_args()

    logging_utils.configure_logging(config.LoggingConfig())

    spec = _resolve_model_spec(args.model_name)
    _log_model_spec(args.model_name, spec)

    device = get_device(args.device)

    LOG.info("Building FP32 segmentation model (backbone + linear head)...")
    seg_model = build_seg_model(spec, head_weights_path=args.head_weights_path)

    transform = _build_seg_transform(spec)
    LOG.info(f"Calibration transform: {transform}")

    LOG.info("Exporting model to ExportedProgram...")
    dummy_input = torch.ones([args.batch_size, 3, spec.input_size, spec.input_size])
    exported_program = torch.export.export(seg_model, args=(dummy_input,))

    LOG.info("Loading calibration data...")
    cal_dataloader = get_calibration_loader(spec, args, transform)

    build_root = Path(args.build_root).expanduser().absolute()
    safe_name = args.model_name.replace("/", "_")
    model_dir = build_root / safe_name
    model_dir.mkdir(parents=True, exist_ok=True)

    LOG.info("Preparing model for quantization...")
    axmo_config = axelera.model_optimizer.get_default_config(
        generation=axelera.model_optimizer.HardwareGeneration.METIS,
        smooth_quant_alpha=args.smooth_quant_alpha,
        smooth_qk_alpha=args.smooth_qk_alpha,
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

    config_data = {
        "model_name": args.model_name,
        "preprocessing": {
            "mean": list(spec.mean),
            "std": list(spec.std),
            "input_size": [3, spec.input_size, spec.input_size],
            "interpolation": spec.interpolation,
            "crop_pct": 1.0,
        },
        "model": {
            "num_classes": spec.num_classes,
            "patch_size": spec.patch_size,
            "embed_dim": spec.embed_dim,
            "dataset": spec.dataset,
            "head_url": spec.head_url,
            "backbone_hub_repo": spec.backbone_hub_repo,
            "backbone_hub_name": spec.backbone_hub_name,
        },
        "quantization": {
            "smooth_quant_alpha": args.smooth_quant_alpha,
            "smooth_qk_alpha": args.smooth_qk_alpha,
            "num_calibration_samples": args.num_calibration_samples,
            "batch_size": args.batch_size,
        },
    }

    compiler_config = CompilerConfig(
        generation=HardwareGeneration.OMEGA,
        target=Target.DEVICE,
        output_dir=model_dir,
        emit_alto_config=True,
        enable_buffer_promotion=False,
        emit_atex_artifacts=True,
        assign_unique_input_pools=True,
        pipeline=Pipeline.GENERIC,
        save_mlir_module=True,
        enable_fold_constant_ops=False,
        device_quant_boundary_setting=DeviceQuantBoundarySetting.HOST_ONLY,
    )
    input_types = (
        TensorType(
            shape=(args.batch_size, 3, spec.input_size, spec.input_size),
            dtype=DType.FLOAT32,
            name="input_image",
        ),
    )

    # Stash model.ptgraph + config.yaml OUTSIDE output_dir before compile,
    # because alto clobbers output_dir at compile-start. We restore them
    # after compile so the artifacts survive whether compile succeeds or
    # fails -- a failed compile still leaves a usable quantized reference
    # for ``inference_seg.py --mode quantized``.
    stash_dir = model_dir.parent / f"{model_dir.name}.ptgraph_stash"
    stash_dir.mkdir(parents=True, exist_ok=True)
    stash_ptgraph = stash_dir / "model.ptgraph"
    stash_config = stash_dir / "config.yaml"
    LOG.info(f"Stashing quantized model to {stash_ptgraph}")
    save_model(fx_graph_model, str(stash_ptgraph))
    with open(stash_config, "w") as f:
        yaml.dump(config_data, f, default_flow_style=False)

    LOG.info("Compile target: %s", compiler_config.target)
    LOG.info("Device quant boundary: %s", compiler_config.device_quant_boundary_setting)
    LOG.info("Compiling to hardware artifacts...")
    compile_failed: Exception | None = None
    axm_path = None
    try:
        axm_path = compile_single_graph(fx_graph_model, compiler_config, input_types)
        LOG.info("Compiled kernel: main -> %s", axm_path)
    except Exception as exc:  # noqa: BLE001 -- preserve stash on any compile failure
        compile_failed = exc
        LOG.error("compile_single_graph raised: %r", exc)

    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "model.ptgraph"
    config_path = model_dir / "config.yaml"
    LOG.info(f"Restoring quantized model to {model_path}")
    stash_ptgraph.replace(model_path)
    stash_config.replace(config_path)
    stash_dir.rmdir()

    if axm_path is not None:
        link = model_dir / "model.axm"
        if link.is_symlink() or link.exists():
            link.unlink()
        rel_target = Path(os.path.relpath(axm_path.resolve(), start=model_dir))
        link.symlink_to(rel_target)
        LOG.info("Deployment complete!")
        LOG.info(f"Output saved to: {model_dir}/")
        LOG.info("  - model.ptgraph: Quantized ExportedProgram")
        LOG.info("  - config.yaml: Preprocessing + model spec for inference_seg.py")
        LOG.info(f"  - model.axm -> {rel_target}")
    else:
        LOG.warning("Compile failed; model.axm NOT created.")
        LOG.warning(f"Quantized reference saved at {model_path}")
        LOG.warning(f"Config saved at {config_path}")
        LOG.warning(
            "Use `inference_seg.py --mode quantized` or `--mode torch` to "
            "measure baselines; rerun deploy after compiler fix to produce .axm."
        )
        if compile_failed is not None:
            raise compile_failed


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.exit_with_error_log()
    except logging_utils.UserError as e:
        LOG.exit_with_error_log(e.format())
    except Exception as e:
        LOG.exit_with_error_log(e)

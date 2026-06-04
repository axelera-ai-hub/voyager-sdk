# Copyright Axelera AI, 2026
"""Preprocessing transforms for axcompile calibration.

Usage:
    axcompile --input model.onnx --transform tools/axcompile_transforms.py:tf_ssd ...

Available presets:
    tf_ssd            - TF SSD models (resize 300x300, tf-mode normalize to [-1,1])
    imagenet_tf       - TF-mode ImageNet (resize 224x224, normalize to [-1,1])
    imagenet_caffe    - Caffe-mode ImageNet (resize 224x224, BGR mean subtraction)
    imagenet_torch    - Torchvision ImageNet (resize 224x224, per-channel normalize)

Each preset is a top-level function: (PIL.Image | np.ndarray) -> torch.Tensor.
Write your own for models with non-standard preprocessing.
"""

import numpy as np
import torch
from PIL import Image

# Standard ImageNet statistics (RGB order)
_TORCH_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_TORCH_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Caffe ImageNet mean (BGR order, pixel values)
_CAFFE_BGR_MEAN = np.array([103.939, 116.779, 123.68], dtype=np.float32)


def _to_chw_float(image, size):
    """Resize image and convert to CHW float32 tensor."""
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    image = image.resize(size, Image.BILINEAR)
    arr = np.array(image, dtype=np.float32)
    return torch.from_numpy(arr).permute(2, 0, 1)


def tf_ssd(image):
    """TF SSD-MobileNetV2: resize 300x300, tf-mode normalize to [-1, 1]."""
    tensor = _to_chw_float(image, (300, 300))
    return (tensor - 127.5) / 127.5


def imagenet_tf(image):
    """TF-mode ImageNet (MobileNet, EfficientNet): resize 224x224, normalize to [-1, 1]."""
    tensor = _to_chw_float(image, (224, 224))
    return (tensor - 127.5) / 127.5


def imagenet_caffe(image):
    """Caffe-mode ImageNet (VGG, ResNet-caffe): resize 224x224, subtract BGR mean."""
    tensor = _to_chw_float(image, (224, 224))
    # tensor is CHW RGB; convert to BGR and subtract per-channel mean
    tensor = tensor.flip(0)  # RGB -> BGR
    mean = torch.from_numpy(_CAFFE_BGR_MEAN).view(3, 1, 1)
    return tensor - mean


def imagenet_torch(image):
    """Torchvision ImageNet (ResNet, DenseNet): resize 224x224, scale+normalize."""
    tensor = _to_chw_float(image, (224, 224))
    tensor = tensor / 255.0
    mean = torch.from_numpy(_TORCH_MEAN).view(3, 1, 1)
    std = torch.from_numpy(_TORCH_STD).view(3, 1, 1)
    return (tensor - mean) / std

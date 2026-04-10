![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Deploying Models with AxMO [Experimental]

## Contents
- [Introduction](#introduction)
  - [What is AxMO?](#what-is-axmo)
  - [What These Scripts Are For](#what-these-scripts-are-for)
  - [When to Use AxMO vs deploy.py](#when-to-use-axmo-vs-deploypy)
- [Prerequisites](#prerequisites)
- [Deploying Models](#deploying-models)
  - [Deploy Use Case 1: Reproduce Published Numbers](#deploy-use-case-1-reproduce-published-numbers)
  - [Deploy Use Case 2: Deploy Your HuggingFace Model](#deploy-use-case-2-deploy-your-huggingface-model)
  - [Deploy Use Case 3: Build Your Own Deploy Script](#deploy-use-case-3-build-your-own-deploy-script)
- [Evaluating Models](#evaluating-models)
  - [Inference Use Case 1: Validate Published Numbers](#inference-use-case-1-validate-published-numbers)
  - [Inference Use Case 2: Evaluate Your Model](#inference-use-case-2-evaluate-your-model)
  - [Inference Use Case 3: Build Your Own Inference Script](#inference-use-case-3-build-your-own-inference-script)
- [Customization Guide](#customization-guide)
- [Troubleshooting](#troubleshooting)
- [Command Reference](#command-reference)

## Introduction

### What is AxMO?

**AxMO (Axelera Model Optimizer)** is a post-training quantization tool for PyTorch models. It converts FP32 models to INT8 for efficient inference on Metis hardware.

### What These Scripts Are For

The example scripts (`deploy_huggingface_classifier.py` and `inference_huggingface_classifier.py`) serve three purposes:

| Use Case | When to Use | Jump To |
|----------|-------------|---------|
| **Reproduce published numbers** | Verify Axelera's accuracy benchmarks for HuggingFace models | [Deploy](#deploy-use-case-1-reproduce-published-numbers) + [Evaluate](#inference-use-case-1-validate-published-numbers) |
| **Deploy your HuggingFace model** | You trained a model on your custom dataset and pushed it to HuggingFace | [Deploy](#deploy-use-case-2-deploy-your-huggingface-model) + [Evaluate](#inference-use-case-2-evaluate-your-model) |
| **Reference code** | Copy code snippets to build your own scripts for local models | [Deploy](#deploy-use-case-3-build-your-own-deploy-script) + [Evaluate](#inference-use-case-3-build-your-own-inference-script) |

> [!TIP]
> Start with Use Case 1 to understand the workflow, then adapt for your needs.

### When to Use AxMO vs deploy.py

| Approach | Best For |
|----------|----------|
| **AxMO (this tutorial)** | Custom PyTorch models, research experiments, full control over quantization |
| **deploy.py with YAML** | Pre-configured models from model zoo, simpler workflow |

Currently, the AxMO pathway provides limited support, exclusively for ViT models. What we are presenting today is our strategic roadmap, as we envision a complete transition to AxMO in the near future. Once fully integrated, AxMO will grant you direct, seamless Python-level control over the entire workflow.

## Prerequisites

### Environment Setup

```bash
# Activate the SDK environment
source venv/bin/activate
```

### GPU Requirements

**GPU is optional.** The scripts auto-detect hardware:
- With CUDA GPU: Uses GPU for faster calibration/inference
- Without GPU: Falls back to CPU automatically

You'll see a log message indicating which device is used:
```
INFO: Using GPU: NVIDIA GeForce RTX 3090
# or
INFO: No GPU detected, using CPU
```

Use `--device cpu` to force CPU even when GPU is available.

### Dependencies

The SDK includes most required dependencies. For reference:
- PyTorch 2.x with torch.export support
- torchvision (for preprocessing transforms)

But the user will need to pip install:
- timm (for HuggingFace model loading)

## Deploying Models

### Deploy Use Case 1: Reproduce Published Numbers

**Purpose:** Verify Axelera's published accuracy numbers for HuggingFace models on ImageNet.

Quantize a HuggingFace model using the built-in ImageNet calibration subset:

```bash
python deploy_huggingface_classifier.py \
    --model_name timm/vit_small_patch16_224.augreg_in21k_ft_in1k \
    --use-hf-subset
```

**What this does:**
- `--model_name`: Downloads and loads the ViT model from HuggingFace/timm
- `--use-hf-subset`: Downloads a small ImageNet subset (~100 images) for calibration

#### Expected Output

```
INFO: Using GPU: NVIDIA GeForce RTX 3090
INFO: Loading model: timm/vit_small_patch16_224.augreg_in21k_ft_in1k
INFO: Model input size: 224x224
INFO: Model preprocessing: mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
INFO: Exporting model to ExportedProgram...
INFO: Downloading HF subset from piupiuisland/imagenet_subset_100...
INFO: Loaded HF ImageNet subset with 100 images
INFO: Preparing model for quantization...
INFO: Running calibration...
100%|----------------------------------------| 100/100 [00:15<00:00]
INFO: Finalizing quantized model...
INFO: Saving quantized model to ./build/timm_vit_small_patch16_224.augreg_in21k_ft_in1k/model.ptgraph
INFO: Saving config to ./build/timm_vit_small_patch16_224.augreg_in21k_ft_in1k/config.yaml
INFO: Quantization complete!
```

#### Files Created

```
build/
  timm_vit_small_patch16_224.augreg_in21k_ft_in1k/
    model.ptgraph   # Quantized ExportedProgram (INT8)
    config.yaml     # Preprocessing config for inference
```

The `config.yaml` stores preprocessing parameters so inference uses identical transforms.

### Deploy Use Case 2: Deploy Your HuggingFace Model

**Purpose:** You trained a classification model on your own dataset and pushed it to HuggingFace.

#### Prepare Your Dataset

Organize your calibration data in ImageFolder format:

```
my_dataset/
  train/           # For calibration (~100-200 images)
    class_a/
      img001.jpg
      img002.jpg
    class_b/
      img003.jpg
```

#### Deploy with Your Calibration Data

```bash
python deploy_huggingface_classifier.py \
    --model_name your-username/your-model \
    --cal-data ./my_dataset/train
```

> [!TIP]
> ImageFolder assigns class indices alphabetically. Ensure your model's output ordering matches.

### Deploy Use Case 3: Build Your Own Deploy Script

**Purpose:** You have a local PyTorch model (not on HuggingFace) and want full control over your workflow.

#### What You Need

The only requirement is a calibration DataLoader that yields input tensors. You can:
- Use the `get_custom_calibration_loader` pattern below (ImageFolder wrapper)
- Use your existing dataset/dataloader
- Generate synthetic calibration data

#### 1. Export Your PyTorch Model

```python
import torch

# Your model (any PyTorch model)
model = your_model
model.eval()

# Export to ExportedProgram
dummy_input = torch.ones([1, 3, 224, 224])  # Match your input shape
exported_program = torch.export.export(model, args=(dummy_input,))
```

#### 2. Run AxMO Quantization

This is the core API to copy into your own script:

```python
import torch
import axelera.model_optimizer
from axelera.model_optimizer.api import (
    prepare_model_for_optimization,
    finalize_optimized_model,
)
from axelera.model_optimizer.trainer.calibration import calibrate_model

# Auto-detect device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Configure for Metis platform
config = axelera.model_optimizer.get_default_config(
    generation=axelera.model_optimizer.HardwareGeneration.METIS,
    smooth_quant_alpha=0.5,
)

# Import and prepare
fx_graph_model = axelera.model_optimizer.import_exported_program(exported_program)
fx_graph_model.to(device)
prepare_model_for_optimization(fx_graph_model, config)

# Calibrate with representative data
calibrate_model(
    fx_graph_model,
    your_dataloader,
    progress_bar=True,
    device=torch.device(device),
)

# Finalize quantization
finalize_optimized_model(fx_graph_model)

# Save
torch.export.save(exported_program, 'model.ptgraph')
```

#### 3. Create a Calibration DataLoader (Optional Pattern)

If you need to load from ImageFolder, here's a simple pattern:

```python
def collate_fn(batch):
    # Return only images, not labels
    images = torch.stack([item[0] for item in batch])
    return images

cal_dataloader = DataLoader(
    dataset,
    batch_size=1,
    collate_fn=collate_fn,
)
```

Or use your own existing dataloader - just ensure it returns batches of input tensors.

See `deploy_huggingface_classifier.py` for a complete working example.

## Evaluating Models

### Inference Use Case 1: Validate Published Numbers

**Purpose:** Compare FP32 baseline accuracy against quantized accuracy to measure any degradation.

#### Run FP32 Baseline

```bash
python inference_huggingface_classifier.py \
    ./build/timm_vit_small_patch16_224.augreg_in21k_ft_in1k \
    --mode torch \
    --use-hf-subset
```

**Expected Output:**

```
INFO: Using GPU: NVIDIA GeForce RTX 3090
INFO: Model: timm/vit_small_patch16_224.augreg_in21k_ft_in1k
INFO: Mode: torch
INFO: Loading original HuggingFace model: timm/vit_small_patch16_224.augreg_in21k_ft_in1k
INFO: Starting evaluation on 100 samples...
Evaluating: 100%|----------------------------------------| 100/100 [00:08<00:00]

==================================================
Evaluation Results
==================================================
Mode:   FP32 (torch)
Device: cuda
--------------------------------------------------
Top-1 Accuracy: 82.00%
Top-5 Accuracy: 96.00%
==================================================
```

#### Run Quantized Model

```bash
python inference_huggingface_classifier.py \
    ./build/timm_vit_small_patch16_224.augreg_in21k_ft_in1k \
    --mode quantized \
    --use-hf-subset
```

**Expected Output:**

```
INFO: Using GPU: NVIDIA GeForce RTX 3090
INFO: Model: timm/vit_small_patch16_224.augreg_in21k_ft_in1k
INFO: Mode: quantized
INFO: Loading quantized model from ./build/.../model.ptgraph
INFO: Starting evaluation on 100 samples...
Evaluating: 100%|----------------------------------------| 100/100 [00:05<00:00]

==================================================
Evaluation Results
==================================================
Mode:   Quantized (INT8)
Device: cuda
--------------------------------------------------
Top-1 Accuracy: 81.00%
Top-5 Accuracy: 95.00%
==================================================
```

#### Compare Results

| Mode | Top-1 | Top-5 |
|------|-------|-------|
| FP32 (torch) | 82.00% | 96.00% |
| Quantized (INT8) | 81.00% | 95.00% |
| **Degradation** | -1.00% | -1.00% |

A 1-2% accuracy drop is typical and acceptable for INT8 quantization.

#### Inference Modes

| Mode | Description |
|------|-------------|
| `torch` | Load original FP32 model from HuggingFace - use for baseline |
| `quantized` | Load quantized INT8 model.ptgraph (default) |
| `torch-aipu` | Run on Metis hardware (coming in future release) |

### Inference Use Case 2: Evaluate Your Model

**Purpose:** Validate accuracy on your custom validation dataset.

```bash
# FP32 baseline
python inference_huggingface_classifier.py ./build/your-username_your-model \
    --mode torch \
    --val-data ./my_dataset/val

# Quantized model
python inference_huggingface_classifier.py ./build/your-username_your-model \
    --mode quantized \
    --val-data ./my_dataset/val
```

The number of classes is auto-detected from ImageFolder subdirectories.

### Inference Use Case 3: Build Your Own Inference Script

**Purpose:** You want full control over evaluation. Use your own dataloader, metrics, and workflow.

#### What You Need

- Your quantized model (loaded with `torch.export.load()`)
- Your validation dataloader (any format)
- Your evaluation logic

#### Example Pattern (Optional)

Here's a simple top-k accuracy pattern you can adapt:

```python
import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

def evaluate_model(model, dataloader, device, num_classes):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, pred_top1 = outputs.topk(1, dim=1)
            _, pred_top5 = outputs.topk(min(5, num_classes), dim=1)

            correct_top1 += (pred_top1.squeeze() == labels).sum().item()
            correct_top5 += (labels.unsqueeze(1) == pred_top5).any(dim=1).sum().item()
            total += labels.size(0)

    return {
        'top1': correct_top1 / total * 100,
        'top5': correct_top5 / total * 100,
    }
```

Or use your own evaluation logic entirely - the quantized model works like any PyTorch model.

See `inference_huggingface_classifier.py` for a complete working example.

## Customization Guide

### Custom Preprocessing

> [!WARNING]
> **Preprocessing MUST match your model's training configuration.** Using incorrect mean/std/crop values will produce garbage accuracy.

For HuggingFace/timm models, use built-in config:
```python
data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)
```

For your own models:
```python
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # YOUR training mean
        std=[0.229, 0.224, 0.225],   # YOUR training std
    ),
])
```

### Custom PyTorch Models

AxMO works with any PyTorch model that supports `torch.export.export()`:

```python
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3)
        self.fc = nn.Linear(64, 1000)

    def forward(self, x):
        x = self.conv(x)
        x = x.flatten(1)
        return self.fc(x)

model = MyModel()
model.load_state_dict(torch.load('my_weights.pth'))
model.eval()

# Then use the AxMO workflow above
```

## Troubleshooting

### No GPU Available

**Symptom:** `INFO: No GPU detected, using CPU`

**Solution:** This is normal - calibration and inference work on CPU, just slower. No action needed.

To verify GPU detection:
```python
import torch
print(torch.cuda.is_available())  # Should be True if GPU is available
print(torch.cuda.get_device_name(0))  # Shows GPU name
```

### Out of Memory

**Symptom:** `CUDA out of memory` error during calibration

**Solutions:**
1. Reduce batch size: `--batch_size 1`
2. Reduce calibration samples: `--num_calibration_samples 50`
3. Use CPU: `--device cpu` (slower but works)

### Accuracy Too Low

**Symptom:** Quantized accuracy is much lower than FP32 (>5% drop)

**Solutions:**
1. **Check preprocessing:** Ensure calibration uses exact same transforms as training
2. **Adjust SmoothQuant alpha:** Try `--smooth_quant_alpha 0.7` or `0.3`
3. **Use more calibration samples:** `--num_calibration_samples 200`
4. **Check data quality:** Calibration images should be representative of real inputs

### Model Export Fails

**Symptom:** `torch.export.export()` raises an error

**Solutions:**
1. Ensure model is in eval mode: `model.eval()`
2. Check for dynamic shapes - torch.export requires static shapes
3. Check for unsupported operations - some custom ops may need adjustment

## Command Reference

### deploy_huggingface_classifier.py

```
python deploy_huggingface_classifier.py --model_name MODEL [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--model_name` | (required) | HuggingFace model name |
| `--build-root` | `./build` | Output directory for model folder |
| `--batch_size` | `1` | Batch size for calibration |
| `--device` | auto-detect | Device for calibration (auto-detects GPU/CPU) |
| `--smooth_quant_alpha` | `0.5` | SmoothQuant alpha (0.0-1.0) |
| `--num_calibration_samples` | `100` | Number of calibration samples |
| `--data-root` | `./data` | Dataset directory |
| `--use-hf-subset` | `false` | Download HuggingFace calibration subset |
| `--cal-data` | `None` | Path to custom calibration data in ImageFolder format |

### inference_huggingface_classifier.py

```
python inference_huggingface_classifier.py MODEL_DIR [options]
```

| Argument | Default | Description |
|----------|---------|-------------|
| `model_dir` | (required) | Path to model folder |
| `--mode` | `quantized` | Inference mode: `torch`, `quantized`, `torch-aipu` |
| `--batch_size` | `1` | Batch size for inference |
| `--device` | auto-detect | Device for inference (auto-detects GPU/CPU) |
| `--frames` | all | Number of frames to evaluate |
| `--data-root` | `./data` | Dataset directory |
| `--use-hf-subset` | `false` | Use HuggingFace dataset subset |
| `--val-data` | `None` | Path to custom validation data in ImageFolder format |
| `--num-classes` | auto-detect | Number of classes (auto-detected from ImageFolder) |

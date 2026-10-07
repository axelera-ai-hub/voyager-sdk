#!/usr/bin/env python3
# Copyright Axelera AI, 2025
"""
Grayscale CNN Transfer Learning Script

This script demonstrates transfer learning with grayscale images using three approaches:
1. Modifying model architecture - Converting the first layer to accept grayscale input
2. Standard RGB approach - For comparison baseline
3. Grayscale-to-RGB preprocessing - Duplicating grayscale channel to 3 channels

The script fine-tunes pretrained CNN models on the Beans dataset to compare
which approach works best for transfer learning with grayscale images.
"""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from PIL import Image

# Use relative imports for local modules
from beans_dataset import (
    BeansDataset,
    GrayscaleToRGBBeansDataset,
    GrayscaleToRGBDataset,
    create_dataloaders,
    create_gray2rgb_dataloaders,
    download_beans_dataset,
    get_grayscale_to_rgb_transforms,
)
import matplotlib.pyplot as plt
import numpy as np
from pytorch_accelerated.trainer import Trainer
import timm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

try:
    from ax_models.torch.utils import convert_first_node_to_1_channel
except Exception as e:
    raise ImportError("Please run the script from the Axelera Framework root directory.") from e


# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Default configuration
CONFIG = {
    "model_name": "res2net50d.in1k",  # Model from timm
    "batch_size": 32,
    "num_epochs": 20,
    "learning_rate": 0.001,
    "weight_decay": 2e-5,
    "grayscale_conversion_method": "sum",  # "sum", "weighted", "average"
    "data_dir": "data/beans_dataset",  # Download dataset to this directory
    "results_dir": Path(__file__).parent,  # Use this file's directory for results
    "mixup_alpha": 0.2,
    "cutmix_alpha": 1.0,
    "label_smoothing": 0.1,
}


class GrayscaleTrainer(Trainer):
    def __init__(self, model, optimizer, loss_fn):
        super().__init__(model=model, optimizer=optimizer, loss_func=loss_fn)

    def training_step(self, batch):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.loss_func(outputs, labels)
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        total = labels.size(0)
        return {"loss": loss, "correct": correct, "total": total}

    def validation_step(self, batch):
        inputs, labels = batch
        outputs = self.model(inputs)
        loss = self.loss_func(outputs, labels)
        _, predicted = outputs.max(1)
        correct = predicted.eq(labels).sum().item()
        total = labels.size(0)
        return {"loss": loss, "correct": correct, "total": total}

    def calculate_metrics(self, outputs):
        loss = torch.stack([output["loss"] for output in outputs]).mean().item()
        correct = sum(output["correct"] for output in outputs)
        total = sum(output["total"] for output in outputs)
        accuracy = 100.0 * correct / total
        return {"loss": loss, "accuracy": accuracy}


def plot_history(grayscale_history, rgb_history=None, gray2rgb_history=None, save_path=None):
    """Plot training and validation metrics."""
    plt.figure(figsize=(15, 10))

    # Plot training & validation accuracy
    plt.subplot(2, 1, 1)
    plt.plot(grayscale_history['train_acc'], label='Grayscale Model Train')
    plt.plot(grayscale_history['val_acc'], label='Grayscale Model Val')

    if rgb_history:
        plt.plot(rgb_history['train_acc'], label='RGB Model Train')
        plt.plot(rgb_history['val_acc'], label='RGB Model Val')

    if gray2rgb_history:
        plt.plot(gray2rgb_history['train_acc'], label='Gray-to-RGB Train')
        plt.plot(gray2rgb_history['val_acc'], label='Gray-to-RGB Val')

    plt.title('Model Accuracy')
    plt.ylabel('Accuracy (%)')
    plt.xlabel('Epoch')
    plt.legend(loc='lower right')

    # Plot training & validation loss
    plt.subplot(2, 1, 2)
    plt.plot(grayscale_history['train_loss'], label='Grayscale Model Train')
    plt.plot(grayscale_history['val_loss'], label='Grayscale Model Val')

    if rgb_history:
        plt.plot(rgb_history['train_loss'], label='RGB Model Train')
        plt.plot(rgb_history['val_loss'], label='RGB Model Val')

    if gray2rgb_history:
        plt.plot(gray2rgb_history['train_loss'], label='Gray-to-RGB Train')
        plt.plot(gray2rgb_history['val_loss'], label='Gray-to-RGB Val')

    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(loc='upper right')

    plt.tight_layout()

    if save_path:
        os.makedirs(
            os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True
        )
        plt.savefig(save_path)
    try:
        plt.show()
    except Exception as e:
        print(f"Warning: Could not display plot: {e}")
    finally:
        plt.close()


def evaluate_model(model, test_loader, device='cuda'):
    """Evaluate model on test dataset and return accuracy."""
    model.to(device)
    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc="Evaluating"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    accuracy = 100.0 * correct / total
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy


def train_with_accelerated(
    model,
    train_loader,
    val_loader,
    num_epochs=10,
    lr=0.001,
    weight_decay=1e-5,
    device='cuda',
    model_name="model",
):
    """Train the model using pytorch_accelerated.Trainer and return training history."""
    model = model.to(device)
    train_dataset = train_loader.dataset
    eval_dataset = val_loader.dataset
    batch_size = train_loader.batch_size
    criterion = nn.CrossEntropyLoss(label_smoothing=CONFIG["label_smoothing"])
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None
    best_val_acc = 0.0

    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    # MixUp implementation
    def mixup_data(x, y, alpha=CONFIG["mixup_alpha"]):
        """Apply mixup transformation to the batch."""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(device)

        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    # CutMix implementation
    def cutmix_data(x, y, alpha=CONFIG["cutmix_alpha"]):
        """Apply cutmix transformation to the batch."""
        if alpha <= 0:
            return x, y, y, 1.0

        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(device)

        y_a, y_b = y, y[index]

        # Generate random box dimensions
        lam = np.random.beta(alpha, alpha)

        # Get image dimensions (assuming square images)
        image_h, image_w = x.size(2), x.size(3)

        # Calculate cut size
        cut_ratio = np.sqrt(1.0 - lam)
        cut_h = int(image_h * cut_ratio)
        cut_w = int(image_w * cut_ratio)

        # Get random center point
        cx = np.random.randint(image_w)
        cy = np.random.randint(image_h)

        # Calculate box coordinates
        bbx1 = np.clip(cx - cut_w // 2, 0, image_w)
        bby1 = np.clip(cy - cut_h // 2, 0, image_h)
        bbx2 = np.clip(cx + cut_w // 2, 0, image_w)
        bby2 = np.clip(cy + cut_h // 2, 0, image_h)

        # Create cutmix image
        x_clone = x.clone()
        x_clone[:, :, bby1:bby2, bbx1:bbx2] = x[index, :, bby1:bby2, bbx1:bbx2]

        # Adjust lambda based on actual box size
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image_h * image_w))

        return x_clone, y_a, y_b, lam

    class AdvancedTrainer(GrayscaleTrainer):
        def __init__(self, model, optimizer, loss_fn):
            super().__init__(model=model, optimizer=optimizer, loss_fn=loss_fn)

        def training_step(self, batch):
            inputs, labels = batch
            inputs = inputs.to(device)
            labels = labels.to(device)

            # Randomly apply mixup or cutmix with 50% probability each
            if np.random.random() < 0.5 and CONFIG["mixup_alpha"] > 0:
                inputs, labels_a, labels_b, lam = mixup_data(inputs, labels)
                outputs = self.model(inputs)
                loss = lam * self.loss_func(outputs, labels_a) + (1 - lam) * self.loss_func(
                    outputs, labels_b
                )
            elif CONFIG["cutmix_alpha"] > 0:
                inputs, labels_a, labels_b, lam = cutmix_data(inputs, labels)
                outputs = self.model(inputs)
                loss = lam * self.loss_func(outputs, labels_a) + (1 - lam) * self.loss_func(
                    outputs, labels_b
                )
            else:
                outputs = self.model(inputs)
                loss = self.loss_func(outputs, labels)

            _, predicted = outputs.max(1)
            correct = predicted.eq(labels).sum().item()
            total = labels.size(0)

            return {"loss": loss, "correct": correct, "total": total}

        def on_train_epoch_end(self, metrics):
            train_losses.append(metrics['loss'])
            train_accs.append(metrics['accuracy'])

        def on_eval_epoch_end(self, metrics):
            nonlocal best_val_loss, patience_counter, best_model_state, best_val_acc
            val_loss = metrics['loss']
            val_acc = metrics['accuracy']

            val_losses.append(val_loss)
            val_accs.append(val_acc)

            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                print(
                    f"New best model with validation accuracy: {val_acc:.2f}% (loss: {val_loss:.4f})"
                )
            elif val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_state = self.model.state_dict().copy()
                print(f"New best model with validation loss: {val_loss:.4f}")
            else:
                patience_counter += 1
                if patience_counter >= 2:
                    for param_group in self.optimizer.param_groups:
                        param_group['lr'] *= 0.1
                    patience_counter = 0
                    print(f"Reduced learning rate to {self.optimizer.param_groups[0]['lr']}")

    trainer = AdvancedTrainer(model=model, optimizer=optimizer, loss_fn=criterion)
    trainer.train(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        num_epochs=num_epochs,
        per_device_batch_size=batch_size,
    )

    save_path = f"{model_name}_beans.pth"
    model_save_path = str(CONFIG["results_dir"] / save_path)

    if best_model_state is not None:
        torch.save(best_model_state, model_save_path)
        print(f"Best model saved to: {model_save_path}")
    else:
        torch.save(trainer.model.state_dict(), model_save_path)
        print(f"Final model saved to: {model_save_path}")

    history = {
        'train_loss': train_losses,
        'train_acc': train_accs,
        'val_loss': val_losses,
        'val_acc': val_accs,
    }

    return history, save_path


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results_dir = Path(CONFIG["results_dir"])
    results_dir.mkdir(exist_ok=True)

    data_dir = download_beans_dataset(CONFIG["data_dir"])

    print("Creating grayscale dataloaders...")
    gray_train_loader, gray_val_loader, gray_test_loader = create_dataloaders(
        data_dir, CONFIG["batch_size"], grayscale=True
    )

    print("Creating RGB dataloaders...")
    rgb_train_loader, rgb_val_loader, rgb_test_loader = create_dataloaders(
        data_dir, CONFIG["batch_size"], grayscale=False
    )

    num_classes = len(gray_train_loader.dataset.classes)
    print(f"Number of classes: {num_classes}")
    model_arch = CONFIG["model_name"].split(".")[0]
    model_weights_paths = {}

    # ========== Train Modified Grayscale Model ==========
    print("\n======= Training Grayscale Model with Modified Architecture =======")
    grayscale_model = timm.create_model(CONFIG["model_name"], pretrained=True)

    grayscale_model = convert_first_node_to_1_channel(
        grayscale_model, conversion_method=CONFIG["grayscale_conversion_method"]
    )

    if hasattr(grayscale_model, 'fc'):
        in_features = grayscale_model.fc.in_features
        grayscale_model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(grayscale_model, 'classifier'):
        in_features = grayscale_model.classifier.in_features
        grayscale_model.classifier = nn.Linear(in_features, num_classes)
    else:
        raise ValueError("Could not find classification layer to modify")

    grayscale_history, grayscale_weight_path = train_with_accelerated(
        grayscale_model,
        gray_train_loader,
        gray_val_loader,
        num_epochs=CONFIG["num_epochs"],
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        device=device,
        model_name=f"grayscale_{model_arch}",
    )
    model_weights_paths['grayscale'] = grayscale_weight_path

    grayscale_model.load_state_dict(torch.load(results_dir / grayscale_weight_path))
    grayscale_acc = evaluate_model(grayscale_model, gray_test_loader, device=device)

    # ========== Train RGB Model for Comparison ==========
    print("\n======= Training RGB Model for Comparison =======")
    rgb_model = timm.create_model(CONFIG["model_name"], pretrained=True)

    if hasattr(rgb_model, 'fc'):
        in_features = rgb_model.fc.in_features
        rgb_model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(rgb_model, 'classifier'):
        in_features = rgb_model.classifier.in_features
        rgb_model.classifier = nn.Linear(in_features, num_classes)

    rgb_history, rgb_weight_path = train_with_accelerated(
        rgb_model,
        rgb_train_loader,
        rgb_val_loader,
        num_epochs=CONFIG["num_epochs"],
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        device=device,
        model_name=f"rgb_{model_arch}",
    )
    model_weights_paths['rgb'] = rgb_weight_path

    rgb_model.load_state_dict(torch.load(results_dir / rgb_weight_path))
    rgb_acc = evaluate_model(rgb_model, rgb_test_loader, device=device)

    # ========== Train Grayscale-to-RGB Model ==========
    print("\n======= Training Model with Grayscale-to-RGB Preprocessing =======")
    gray2rgb_model = timm.create_model(CONFIG["model_name"], pretrained=True)

    if hasattr(gray2rgb_model, 'fc'):
        in_features = gray2rgb_model.fc.in_features
        gray2rgb_model.fc = nn.Linear(in_features, num_classes)
    elif hasattr(gray2rgb_model, 'classifier'):
        in_features = gray2rgb_model.classifier.in_features
        gray2rgb_model.classifier = nn.Linear(in_features, num_classes)

    gray2rgb_train_loader, gray2rgb_val_loader, gray2rgb_test_loader = create_gray2rgb_dataloaders(
        data_dir, CONFIG["batch_size"]
    )

    gray2rgb_history, gray2rgb_weight_path = train_with_accelerated(
        gray2rgb_model,
        gray2rgb_train_loader,
        gray2rgb_val_loader,
        num_epochs=CONFIG["num_epochs"],
        lr=CONFIG["learning_rate"],
        weight_decay=CONFIG["weight_decay"],
        device=device,
        model_name=f"gray2rgb_{model_arch}",
    )
    model_weights_paths['gray2rgb'] = gray2rgb_weight_path

    gray2rgb_model.load_state_dict(torch.load(results_dir / gray2rgb_weight_path))
    gray2rgb_acc = evaluate_model(gray2rgb_model, gray2rgb_test_loader, device=device)

    plot_history(
        grayscale_history, rgb_history, gray2rgb_history, save_path=results_dir / "comparison.png"
    )
    print(f"Training history plot saved to: {results_dir / 'comparison.png'}")

    print("\n========== Results ==========")
    print(f"Grayscale Model (Modified Architecture): {grayscale_acc:.2f}%")
    print(f"RGB Model: {rgb_acc:.2f}%")
    print(f"Grayscale-to-RGB Model (Preprocessing Approach): {gray2rgb_acc:.2f}%")

    with open(results_dir / "results.txt", "w") as f:
        f.write(f"Experiment Configuration:\n{CONFIG}\n\n")
        f.write("Results:\n")
        f.write(f"Grayscale Model (Modified Architecture): {grayscale_acc:.2f}%\n")
        f.write(f"RGB Model: {rgb_acc:.2f}%\n")
        f.write(f"Grayscale-to-RGB Model (Preprocessing Approach): {gray2rgb_acc:.2f}%\n")
        f.write("\nSaved Model Weights:\n")
        for model_type, weight_path in model_weights_paths.items():
            f.write(f"{model_type}: {results_dir / weight_path}\n")

    print(f"Results saved to: {results_dir / 'results.txt'}")


if __name__ == "__main__":
    main()

# Copyright Axelera AI, 2025
"""
Beans Dataset Module

This module contains the dataset class and utility functions for the Beans dataset
from Hugging Face, used for training and evaluation of models.
"""

from datasets import load_dataset
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


def download_beans_dataset(root_dir):
    """Download the Beans dataset from Hugging Face."""
    print("Downloading and preparing Beans dataset from Hugging Face...")

    # Load dataset from Hugging Face
    beans_dataset = load_dataset("AI-Lab-Makerere/beans", cache_dir=root_dir)

    print(
        f"Dataset downloaded and prepared. Train size: {len(beans_dataset['train'])}, "
        f"Validation size: {len(beans_dataset['validation'])}, "
        f"Test size: {len(beans_dataset['test'])}"
    )

    return beans_dataset


class BeansDataset(Dataset):
    """Custom dataset for the Beans dataset from Hugging Face."""

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.classes = ['angular_leaf_spot', 'bean_rust', 'healthy']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['labels']

        if self.transform:
            image = self.transform(image)

        return image, label


class GrayscaleToRGBDataset(Dataset):
    """Dataset that converts grayscale images to 3-channel by repeating."""

    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform
        self.classes = ['angular_leaf_spot', 'bean_rust', 'healthy']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item['image']
        label = item['labels']

        if self.transform:
            image = self.transform(image)

        # If the image is grayscale (1 channel), repeat it to 3 channels
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        return image, label


class GrayscaleToRGBBeansDataset(BeansDataset):
    """Dataset that extends BeansDataset to convert grayscale to RGB."""

    def __getitem__(self, idx):
        # Get the image and label using the parent class method
        image, label = super().__getitem__(idx)

        # If the image is grayscale (1 channel), repeat it to 3 channels
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)

        return image, label


def create_dataloaders(hf_dataset, batch_size=32, grayscale=False):
    """Create train, validation, and test dataloaders for the Beans dataset."""
    # Define image transformations
    if grayscale:
        # Enhanced grayscale transformations with more augmentations
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),  # Wider scale range
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(p=0.2),  # Add vertical flipping
                transforms.RandomRotation(15),  # Add rotation
                transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Add translation
                transforms.ColorJitter(brightness=0.2),  # Add brightness adjustment
                transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
                transforms.ToTensor(),
                transforms.Normalize([0.456], [0.224]),  # Grayscale normalization
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
                transforms.ToTensor(),
                transforms.Normalize([0.456], [0.224]),  # Grayscale normalization
            ]
        )
    else:
        # RGB transformations
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    # Create custom datasets
    train_dataset = BeansDataset(hf_dataset['train'], transform=train_transform)

    val_dataset = BeansDataset(hf_dataset['validation'], transform=test_transform)

    test_dataset = BeansDataset(hf_dataset['test'], transform=test_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader


def get_grayscale_to_rgb_transforms():
    """Returns transforms for grayscale to RGB conversion."""
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.ToTensor(),
            transforms.Normalize([0.456], [0.224]),  # Grayscale normalization
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.Grayscale(num_output_channels=1),  # Convert to grayscale
            transforms.ToTensor(),
            transforms.Normalize([0.456], [0.224]),  # Grayscale normalization
        ]
    )

    return train_transform, test_transform


def create_gray2rgb_dataloaders(hf_dataset, batch_size=32):
    """Create grayscale-to-RGB dataloaders for the Beans dataset."""
    train_transform, test_transform = get_grayscale_to_rgb_transforms()

    # Create datasets
    train_dataset = GrayscaleToRGBBeansDataset(hf_dataset['train'], transform=train_transform)

    val_dataset = GrayscaleToRGBBeansDataset(hf_dataset['validation'], transform=test_transform)

    test_dataset = GrayscaleToRGBBeansDataset(hf_dataset['test'], transform=test_transform)

    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

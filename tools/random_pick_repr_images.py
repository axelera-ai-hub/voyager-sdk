#!/usr/bin/env python
# Copyright Axelera AI, 2025
#
import argparse
import os
from pathlib import Path
import random
import shutil


def select_random_images(source_dir, target_dir, num_images):
    """
    Search for images in the source directory (including subdirectories),
    randomly select N images, and copy them to the target directory.

    Args:
        source_dir (str): Path to the source directory.
        target_dir (str): Path to the target directory.
        num_images (int): Number of images to select and copy.
    """
    image_extensions = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".webp"}

    source_dir = os.path.expanduser(source_dir)
    target_dir = os.path.expanduser(target_dir)

    if not os.path.exists(source_dir):
        print(f"Source directory '{source_dir}' does not exist.")
        return

    os.makedirs(target_dir, exist_ok=True)

    image_files = []
    for root, _, files in os.walk(source_dir):
        for file in files:
            if Path(file).suffix.lower() in image_extensions:
                image_files.append(os.path.join(root, file))

    if len(image_files) < num_images:
        print(
            f"Not enough images found. Found {len(image_files)} images, but {num_images} were requested."
        )
        return
    selected_images = random.sample(image_files, num_images)
    for image in selected_images:
        shutil.copy(image, target_dir)

    print(f"Successfully copied {num_images} images to '{target_dir}'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Randomly select N images from a source directory and copy them to a target directory."
    )
    parser.add_argument(
        "source_dir", type=str, help="Path to the source directory (can include subdirectories)."
    )
    parser.add_argument(
        "target_dir", type=str, help="Path to the target directory (no subdirectories)."
    )
    parser.add_argument(
        "--N", type=int, required=True, help="Number of images to randomly select and copy."
    )

    args = parser.parse_args()
    select_random_images(args.source_dir, args.target_dir, args.N)

#!/usr/bin/env python3
# Copyright Axelera AI, 2025
"""
Convert AFLink PyTorch model to ONNX format for C++ inference.

This script loads the pre-trained AFLink model and exports it to ONNX format
for use with ONNX Runtime in the C++ TrackTrack implementation.
"""

# ==============================================================================
# IMPORTANT: This script requires external dependencies not included in this repo
# ==============================================================================
#
# Required directory structure:
#   trackers/algorithms/TrackTrack_python/AFLink/
#
# To use this script:
#   1. Obtain the TrackTrack Python reference implementation
#   2. Place it at: trackers/algorithms/TrackTrack_python/
#   3. Install dependencies: pip install torch onnx
#   4. Run: python convert_aflink_to_onnx.py --input path/to/model.pth --output path/to/output.onnx
#
# ==============================================================================

import argparse
import os
from pathlib import Path
import sys

import torch
import torch.onnx

# Add parent directory to path to import AFLink
sys.path.append(str(Path(__file__).parent.parent.parent / "TrackTrack_python"))

try:
    from AFLink.model import PostLinker
except ImportError:
    print("Error: Could not import AFLink.model.PostLinker")
    print("Make sure you're running this script from the tracktrack directory")
    print("and that the TrackTrack_python directory exists in algorithms/")
    sys.exit(1)


def convert_model(model_path, output_path, opset_version=11):
    """
    Convert AFLink PyTorch model to ONNX format.

    Args:
        model_path: Path to the PyTorch model (.pth file)
        output_path: Path for the output ONNX model
        opset_version: ONNX opset version to use
    """
    print(f"Loading PyTorch model from: {model_path}")

    # Create model instance
    model = PostLinker()

    # Load model weights
    if not os.path.exists(model_path):
        print(f"Error: Model file not found: {model_path}")
        sys.exit(1)

    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    print("Model loaded successfully")

    # Create dummy inputs
    # Based on AFLink architecture, it takes two track sequences
    # Each track is represented as [frames, features]
    # Features: [x, y, w, h] for bounding box
    batch_size = 1
    max_frames = 30  # Maximum track length
    feature_dim = 4  # x, y, w, h

    dummy_track1 = torch.randn(batch_size, max_frames, feature_dim)
    dummy_track2 = torch.randn(batch_size, max_frames, feature_dim)

    print(f"Dummy input shapes: track1={dummy_track1.shape}, track2={dummy_track2.shape}")

    # Export to ONNX
    print(f"Exporting to ONNX: {output_path}")

    torch.onnx.export(
        model,
        (dummy_track1, dummy_track2),
        output_path,
        export_params=True,
        opset_version=opset_version,
        do_constant_folding=True,
        input_names=["track1", "track2"],
        output_names=["similarity_score"],
        dynamic_axes={
            "track1": {1: "seq_len"},  # Variable sequence length
            "track2": {1: "seq_len"},  # Variable sequence length
        },
        verbose=False,
    )

    print("Export completed successfully!")

    # Verify the exported model
    try:
        import onnx

        onnx_model = onnx.load(output_path)
        onnx.checker.check_model(onnx_model)
        print("ONNX model validation passed!")

        # Print model info
        print("\nModel information:")
        print(f"- Input names: {[i.name for i in onnx_model.graph.input]}")
        print(f"- Output names: {[o.name for o in onnx_model.graph.output]}")
        print(f"- Model size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")

    except ImportError:
        print("\nWarning: onnx package not installed. Skipping model validation.")
        print("Install with: pip install onnx")
    except Exception as e:
        print(f"\nWarning: Model validation failed: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert AFLink PyTorch model to ONNX")
    parser.add_argument(
        "--input",
        "-i",
        default="../../TrackTrack_python/AFLink/AFLink_epoch20.pth",
        help="Path to input PyTorch model (default: ../../TrackTrack_python/AFLink/AFLink_epoch20.pth)",
    )
    parser.add_argument(
        "--output",
        "-o",
        default="../models/aflink_epoch20.onnx",
        help="Path for output ONNX model (default: ../models/aflink_epoch20.onnx)",
    )
    parser.add_argument("--opset", type=int, default=11, help="ONNX opset version (default: 11)")

    args = parser.parse_args()

    # Create output directory if needed
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        print(f"Created output directory: {output_dir}")

    # Convert the model
    convert_model(args.input, args.output, args.opset)


if __name__ == "__main__":
    main()

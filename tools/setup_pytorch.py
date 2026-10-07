#!/usr/bin/env python3
# Copyright Axelera AI, 2025
"""
PyTorch Setup Utilities

This module provides functions for detecting CUDA availability and installing
PyTorch with appropriate CUDA support.
"""

import argparse
import platform
import re
import subprocess
import sys
from typing import List, Optional, Tuple

# Default PyTorch versions - keep in sync with containerless.sh
DEFAULT_TORCH_VERSION = "2.7.0"

# Known compatible PyTorch-TorchVision version pairs
# These are the officially supported combinations
TORCH_TORCHVISION_PAIRS = {
    "2.7.0": "0.22.0",
    "2.6.0": "0.21.0",
    "2.5.0": "0.20.0",
    "2.1.0": "0.16.0",
    "2.0.0": "0.15.0",
    "1.13.0": "0.14.0",
    "1.13.1": "0.14.1",
    "1.12.0": "0.13.0",
    "1.11.0": "0.12.0",
}

# Known CUDA version requirements by PyTorch version
# These are the supported CUDA versions for each PyTorch version
# The list should be ordered from highest to lowest version for each PyTorch release
TORCH_CUDA_REQUIREMENTS = {
    "2.7.0": ["12.8", "12.1", "11.8"],
    "2.6.0": ["12.4", "12.1", "11.8"],
    "2.5.0": ["12.1", "11.8"],
    "2.1.0": ["12.1", "11.8"],
    "2.0.0": ["11.8", "11.7"],
    "1.13.0": ["11.7", "11.6"],
    "1.13.1": ["11.7", "11.6"],
    "1.12.0": ["11.6", "11.3"],
    "1.11.0": ["11.3"],
}

# Special cases for CUDA wheel suffixes that don't follow the standard pattern
CUDA_SUFFIX_EXCEPTIONS = {
    "12.8": "cu128",
    "12.4": "cu126",
    "12.1": "cu126",
    "12.0": "cu120",
    "11.8": "cu118",
    "11.7": "cu117",
    "11.6": "cu116",
    "11.3": "cu113",
}


def get_cuda_suffix(cuda_version: str) -> str:
    """Convert a CUDA version string to a PyTorch wheel suffix.

    Args:
        cuda_version: CUDA version string (e.g., "12.1")

    Returns:
        The corresponding PyTorch wheel suffix (e.g., "cu121")
    """
    # Check if this is a special case
    if cuda_version in CUDA_SUFFIX_EXCEPTIONS:
        return CUDA_SUFFIX_EXCEPTIONS[cuda_version]

    # Standard case: Remove dots and prepend "cu"
    return f"cu{cuda_version.replace('.', '')}"


def get_compatible_torchvision_version(torch_version: str) -> str:
    """Get the compatible TorchVision version for a given PyTorch version.

    Args:
        torch_version: PyTorch version string

    Returns:
        Compatible TorchVision version string
    """
    if torch_version in TORCH_TORCHVISION_PAIRS:
        return TORCH_TORCHVISION_PAIRS[torch_version]

    # If not found, try to find a close match or warn and use a default
    print(
        f"Warning: No known compatible TorchVision version for PyTorch {torch_version}",
        file=sys.stderr,
    )
    print("Using heuristic to determine TorchVision version...", file=sys.stderr)

    # Extract the major and minor version
    version_parts = torch_version.split('.')
    if len(version_parts) >= 2:
        major = version_parts[0]
        minor = version_parts[1]

        # Try with patch version 0
        test_version = f"{major}.{minor}.0"
        if test_version in TORCH_TORCHVISION_PAIRS:
            print(f"Using TorchVision version for PyTorch {test_version}", file=sys.stderr)
            return TORCH_TORCHVISION_PAIRS[test_version]

    print("Warning: Could not determine compatible TorchVision version.", file=sys.stderr)
    print(
        "Please specify TorchVision version manually or update the TORCH_TORCHVISION_PAIRS dictionary.",
        file=sys.stderr,
    )

    # Default to the latest known version as a fallback
    latest_torch = max(
        TORCH_TORCHVISION_PAIRS.keys(), key=lambda v: [int(x) for x in v.split('.')]
    )
    fallback_version = TORCH_TORCHVISION_PAIRS[latest_torch]
    print(
        f"Falling back to TorchVision {fallback_version}, which may not be compatible.",
        file=sys.stderr,
    )

    return fallback_version


def select_cuda_version_for_pytorch(
    detected_cuda_version: float, torch_version: str
) -> Optional[str]:
    """Select the appropriate CUDA version for the given PyTorch version.

    Args:
        detected_cuda_version: Detected CUDA version as float (e.g., 11.8)
        torch_version: PyTorch version string

    Returns:
        The selected CUDA version as string, or None if no compatible version
    """
    if torch_version not in TORCH_CUDA_REQUIREMENTS:
        return None

    # Get the CUDA version thresholds for this PyTorch version
    cuda_versions = TORCH_CUDA_REQUIREMENTS[torch_version]
    detected_str = f"{detected_cuda_version}"

    # Check for exact version match first
    if detected_str in cuda_versions:
        return detected_str

    # Find the appropriate CUDA version based on thresholds
    detected_major = int(detected_cuda_version)
    detected_minor = int((detected_cuda_version - detected_major) * 10)

    for version in cuda_versions:
        major, minor = map(int, version.split('.'))

        # If detected version is at least this threshold, use this version
        if (detected_major > major) or (detected_major == major and detected_minor >= minor):
            return version

    # If detected version is lower than all supported versions, return None
    return None


def detect_cuda_and_get_torch_install_args(
    torch_version: str,
    torchvision_version: Optional[str] = None,
    force_cpu: bool = False,
    quiet: bool = False,
) -> Tuple[List[str], Optional[str]]:
    """Detect CUDA version and return appropriate PyTorch install arguments.

    Args:
        torch_version: Version of PyTorch to install
        torchvision_version: Version of TorchVision to install (optional, will auto-detect if not specified)
        force_cpu: Whether to force CPU-only installation
        quiet: Suppress informational output

    Returns:
        Tuple of (requirements list, index_url)
    """
    # If torchvision_version is not specified, determine it based on torch_version
    if torchvision_version is None:
        torchvision_version = get_compatible_torchvision_version(torch_version)
        if not quiet:
            print(
                f"Auto-selected TorchVision version {torchvision_version} for PyTorch {torch_version}",
                file=sys.stderr,
            )

    os_platform = platform.system()
    torch_requirements = []
    index_url = None

    if os_platform == "Darwin" or force_cpu:
        # For macOS or when CPU is forced, use standard packages
        torch_requirements = [f"torch=={torch_version}", f"torchvision=={torchvision_version}"]
        return torch_requirements, index_url

    # Try to detect CUDA using nvidia-smi
    try:
        nvidia_smi_output = subprocess.check_output(["nvidia-smi"], text=True)
        # Extract CUDA version
        cuda_match = re.search(r"CUDA Version: (\d+\.\d+)", nvidia_smi_output)
        if cuda_match:
            detected_cuda_version = float(cuda_match.group(1))
            if not quiet:
                print(f"Detected CUDA version: {detected_cuda_version}", file=sys.stderr)

            # Select appropriate CUDA version based on thresholds
            selected_cuda_version = select_cuda_version_for_pytorch(
                detected_cuda_version, torch_version
            )

            if selected_cuda_version:
                # Get the wheel suffix for the selected CUDA version
                cuda_suffix = get_cuda_suffix(selected_cuda_version)

                if not quiet:
                    print(
                        f"Using PyTorch with CUDA {selected_cuda_version} (package suffix: {cuda_suffix})",
                        file=sys.stderr,
                    )

                index_url = f"https://download.pytorch.org/whl/{cuda_suffix}"
                torch_requirements = [
                    f"torch=={torch_version}+{cuda_suffix}",
                    f"torchvision=={torchvision_version}+{cuda_suffix}",
                ]
            else:
                if not quiet:
                    print(
                        f"CUDA version {detected_cuda_version} is too old for PyTorch {torch_version}.",
                        file=sys.stderr,
                    )
                    print(f"Falling back to CPU version.", file=sys.stderr)
                index_url = "https://download.pytorch.org/whl/cpu"
                torch_requirements = [
                    f"torch=={torch_version}+cpu",
                    f"torchvision=={torchvision_version}+cpu",
                ]
        else:
            if not quiet:
                print(
                    "Could not determine CUDA version. Falling back to CPU version.",
                    file=sys.stderr,
                )
            index_url = "https://download.pytorch.org/whl/cpu"
            torch_requirements = [
                f"torch=={torch_version}+cpu",
                f"torchvision=={torchvision_version}+cpu",
            ]
    except (subprocess.SubprocessError, FileNotFoundError):
        if not quiet:
            print(
                "nvidia-smi not found or failed. Installing CPU version of PyTorch.",
                file=sys.stderr,
            )
        index_url = "https://download.pytorch.org/whl/cpu"
        torch_requirements = [
            f"torch=={torch_version}+cpu",
            f"torchvision=={torchvision_version}+cpu",
        ]

    return torch_requirements, index_url


def get_install_command(
    torch_version: str, torchvision_version: Optional[str] = None, force_cpu: bool = False
) -> str:
    """Get the pip install command string to install PyTorch.

    Args:
        torch_version: Version of PyTorch to install
        torchvision_version: Version of TorchVision to install (optional)
        force_cpu: Whether to force CPU-only installation

    Returns:
        String containing the pip install command
    """
    torch_requirements, index_url = detect_cuda_and_get_torch_install_args(
        torch_version, torchvision_version, force_cpu, quiet=True
    )

    if index_url:
        return f"--index-url {index_url} {' '.join(torch_requirements)}"
    else:
        return f"{' '.join(torch_requirements)}"


def check_pytorch_cuda(python_path: str) -> Tuple[str, int]:
    """Check if PyTorch can see CUDA and print version information.

    Args:
        python_path: Path to Python executable to use

    Returns:
        Tuple of (command output, return code)
    """
    print("\nChecking PyTorch installation...", file=sys.stderr)
    pytorch_check = """
import torch
print('PyTorch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA version:', torch.version.cuda)
    print('GPU device count:', torch.cuda.device_count())
    for i in range(torch.cuda.device_count()):
        print(f'GPU {i} name:', torch.cuda.get_device_name(i))
else:
    print('CUDA version: N/A')
    print('GPU device count: 0')
"""
    try:
        result = subprocess.run(
            [python_path, "-c", pytorch_check],
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
        )
        print(result.stdout, file=sys.stderr)
        return result.stdout, result.returncode
    except subprocess.CalledProcessError as e:
        print(f"Command failed with error code {e.returncode}", file=sys.stderr)
        print(e.output, file=sys.stderr)
        return e.output, e.returncode


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="PyTorch CUDA detection and installation utility")
    parser.add_argument(
        "--torch-version",
        type=str,
        default=DEFAULT_TORCH_VERSION,
        help=f"PyTorch version to install (default: {DEFAULT_TORCH_VERSION})",
    )
    parser.add_argument(
        "--torchvision-version",
        type=str,
        help="TorchVision version to install (will auto-detect if not specified)",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU-only installation even if CUDA is available",
    )
    parser.add_argument(
        "--get-install-cmd",
        action="store_true",
        help="Output only the pip install command for use in scripts",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.get_install_cmd:
        # When using --get-install-cmd, only output the command string
        cmd = get_install_command(args.torch_version, args.torchvision_version, args.force_cpu)
        print(cmd)
    else:
        # Normal interactive mode
        torch_requirements, index_url = detect_cuda_and_get_torch_install_args(
            args.torch_version, args.torchvision_version, args.force_cpu
        )

        print("\nDetected configuration:")
        print(f"PyTorch version: {args.torch_version}")
        if args.torchvision_version:
            print(f"TorchVision version: {args.torchvision_version}")
        else:
            torchvision_version = get_compatible_torchvision_version(args.torch_version)
            print(f"TorchVision version (auto): {torchvision_version}")

        print("\nInstallation command:")
        if index_url:
            print(f"pip install --index-url {index_url} {' '.join(torch_requirements)}")
        else:
            print(f"pip install {' '.join(torch_requirements)}")

        print(
            "\nYou can run this command yourself, or set up your virtual environment using the tools/setup_training_env.py script."
        )

#!/usr/bin/env python3
# Copyright Axelera AI, 2025
"""
ML Training Environment Setup Script

This script creates a dedicated Python virtual environment for training
machine learning models, installing all required dependencies including PyTorch
with appropriate CUDA support.

Usage:
    # Create a training environment:
    python tools/setup_training_env.py

    # Activate the environment:
    source training_env/bin/activate

    # Update an existing environment with new packages:
    python setup_training_env.py --force-reinstall
"""

import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import venv

from setup_pytorch import (
    DEFAULT_TORCH_VERSION,
    check_pytorch_cuda,
    detect_cuda_and_get_torch_install_args,
)

# Common ML dependencies; consider to move to a separate requirements file
ML_REQUIREMENTS = [
    "timm==0.9.8",
    "matplotlib==3.10.8",
    "tqdm==4.66.1",
    "datasets==2.15.0",
    "pytorch-accelerated==0.1.52",
    "ipykernel",
    "jupyter",
    "tensorboard",
]


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Create a Python virtual environment for ML model training."
    )
    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Force CPU-only PyTorch installation (not usually needed as CUDA detection is automatic)",
    )
    parser.add_argument(
        "--env-dir",
        type=str,
        default="training_env",
        help="Directory to create virtual environment in (default: training_env)",
    )
    parser.add_argument(
        "--torch-version",
        type=str,
        default=DEFAULT_TORCH_VERSION,
        help=f"PyTorch version to install (default: {DEFAULT_TORCH_VERSION})",
    )
    parser.add_argument(
        "--force-reinstall",
        action="store_true",
        help="Force reinstall packages even if environment already exists",
    )
    parser.add_argument(
        "--no-activation-script",
        action="store_true",
        dest='existing_venv',
        help="Don't generate a separate activation script (used by containerless.sh)",
    )
    return parser.parse_args()


def print_colored(text, color="green"):
    """Print colored text to the console."""
    colors = {
        "red": "\033[31m",
        "green": "\033[32m",
        "yellow": "\033[33m",
        "blue": "\033[34m",
        "magenta": "\033[35m",
        "cyan": "\033[36m",
        "reset": "\033[0m",
    }
    print(f"{colors.get(color, '')}{text}{colors['reset']}")


def check_python_version():
    """Check that the Python version is sufficient."""
    min_version = (3, 8)
    current_version = sys.version_info[:2]

    if current_version < min_version:
        print_colored(
            f"Error: Python {min_version[0]}.{min_version[1]} or higher is required.", "red"
        )
        print_colored(f"Current version: {current_version[0]}.{current_version[1]}", "red")
        return False
    return True


def run_command(cmd, env=None, check=True, shell=False):
    """Run a command and return its output and return code."""
    print(f"Running: {' '.join(cmd) if isinstance(cmd, list) else cmd}")
    try:
        if shell:
            result = subprocess.run(
                cmd,
                env=env,
                check=check,
                shell=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        else:
            result = subprocess.run(
                cmd,
                env=env,
                check=check,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
            )
        return result.stdout, result.returncode
    except subprocess.CalledProcessError as e:
        print_colored(f"Command failed with error code {e.returncode}", "red")
        print(e.output)
        if check:
            raise
        return e.output, e.returncode


def create_venv(env_dir, force_reinstall=False):
    """Create a virtual environment at the specified path."""
    print(f"Creating virtual environment at: {env_dir}")
    if os.path.exists(env_dir):
        if force_reinstall:
            print(f"Force reinstall requested, removing existing environment at {env_dir}")
            shutil.rmtree(env_dir)
        else:
            print(f"Using existing environment at {env_dir}")
            return False  # to indicate environment not created

    venv.create(env_dir, with_pip=True)
    print_colored(f"Virtual environment created successfully at {env_dir}", "green")
    return True  # to indicate new environment was created


def get_pip_path(env_dir):
    """Get the path to pip in the virtual environment."""
    if platform.system() == "Windows":
        pip_path = os.path.join(env_dir, "Scripts", "pip")
    else:
        pip_path = os.path.join(env_dir, "bin", "pip")
    return pip_path


def install_dependencies(env_dir, torch_version, use_cuda=True):
    """Install dependencies in the virtual environment."""
    pip_path = get_pip_path(env_dir)
    run_command([pip_path, "install", "--upgrade", "pip"])
    torch_requirements, index_url = detect_cuda_and_get_torch_install_args(
        torch_version, force_cpu=not use_cuda
    )

    if index_url:
        print(f"Installing PyTorch from {index_url}...")
        torch_install_cmd = [pip_path, "install", "--index-url", index_url] + torch_requirements
    else:
        print("Installing PyTorch from PyPI...")
        torch_install_cmd = [pip_path, "install"] + torch_requirements

    run_command(torch_install_cmd)
    print("Installing common ML libraries...")
    for req in ML_REQUIREMENTS:
        run_command([pip_path, "install", req])


def register_with_jupyter(env_dir):
    """Register the environment with Jupyter if available."""
    if platform.system() == "Windows":
        python_path = os.path.join(env_dir, "Scripts", "python")
    else:
        python_path = os.path.join(env_dir, "bin", "python")

    try:
        run_command(
            [
                python_path,
                "-m",
                "ipykernel",
                "install",
                "--user",
                "--name=training_env",
                "--display-name=ML Training Environment",
            ],
            check=False,
        )
        print("Environment registered with Jupyter successfully")
    except Exception as e:
        print_colored(f"Failed to register with Jupyter: {e}", "yellow")


def get_python_path(env_dir):
    """Get path to Python executable in virtual environment."""
    if platform.system() == "Windows":
        return os.path.join(env_dir, "Scripts", "python")
    else:
        return os.path.join(env_dir, "bin", "python")


def print_activation_instructions(env_dir):
    """Print instructions for activating the virtual environment."""
    print_colored("\nEnvironment setup completed successfully!", "green")

    if platform.system() == "Windows":
        activate_cmd = os.path.join(env_dir, "Scripts", "activate")
        print("\nTo activate the environment in cmd.exe:")
        print_colored(f"    {activate_cmd}", "cyan")
        print("\nTo activate the environment in PowerShell:")
        print_colored(f"    & {activate_cmd}", "cyan")
    else:
        activate_cmd = os.path.join(env_dir, "bin", "activate")
        print("\nTo activate the environment:")
        print_colored(f"    source {activate_cmd}", "cyan")

    print("\nTo deactivate the environment:")
    print_colored("    deactivate", "cyan")


def main():
    if not check_python_version():
        sys.exit(1)

    args = parse_args()
    env_dir = args.env_dir

    if not args.existing_venv and os.environ.get('VIRTUAL_ENV'):
        print_colored("\nWARNING: You are currently in a Python virtual environment.", "yellow")
        print("It's recommended to run this script outside any virtual environment.")
        response = input("Do you want to continue anyway? (y/n): ")
        if response.lower() != 'y':
            print("Exiting...")
            sys.exit(0)

    need_install = args.existing_venv or create_venv(env_dir, args.force_reinstall)
    if need_install or args.force_reinstall:
        install_dependencies(env_dir, args.torch_version, not args.no_cuda)
        register_with_jupyter(env_dir)
        python_path = get_python_path(env_dir)
        check_pytorch_cuda(python_path)
    else:
        print_colored("Using existing environment - skipping package installation", "green")
        print("Use --force-reinstall flag to force package reinstallation")

    if not args.existing_venv:
        print_activation_instructions(env_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
# Copyright Axelera AI, 2026
"""A temporary script to package a compiled build directory into an .axm file.
This file will be deleted once we transition to Pythonic API.

Usage:
    python tools/make_axm.py build/yolo26n-coco-onnx/yolo26n-coco-onnx
    python tools/make_axm.py build/yolo26n-coco-onnx/yolo26n-coco-onnx -o my_model.axm
"""
import argparse
import sys
import zipfile
from pathlib import Path


# Files expected inside the numbered version directory
VERSION_DIR_FILES = {'manifest.json', 'model.json'}


def make_axm(build_dir: Path, output: Path) -> None:
    version_dir = build_dir / '1'
    if not version_dir.is_dir():
        sys.exit(f'Error: version directory {version_dir} not found')

    version_files = [
        f for f in version_dir.iterdir() if f.is_file() and f.name in VERSION_DIR_FILES
    ]

    missing_version = VERSION_DIR_FILES - {f.name for f in version_files}
    if missing_version:
        print(f'Warning: missing version files: {missing_version}', file=sys.stderr)

    if not version_files:
        sys.exit('Error: no files to package')

    with zipfile.ZipFile(output, 'w', zipfile.ZIP_STORED) as zf:
        for f in version_files:
            zf.write(f, f.name)

    total_mb = output.stat().st_size / (1024 * 1024)
    print(
        f'Created {output} ({total_mb:.1f} MB, {len(version_files)} files from version {version_dir.name}/)'
    )


def main():
    parser = argparse.ArgumentParser(description='Package a build directory into an .axm file')
    parser.add_argument(
        'build_dir',
        type=Path,
        help='Path to the build directory (e.g. build/yolo26n-coco-onnx/yolo26n-coco-onnx)',
    )
    parser.add_argument(
        '-o',
        '--output',
        type=Path,
        default=None,
        help='Output .axm path (default: <model_name>.axm in current directory)',
    )
    args = parser.parse_args()

    build_dir = args.build_dir.resolve()
    if not build_dir.is_dir():
        sys.exit(f'Error: {build_dir} is not a directory')

    output = args.output or Path(f'{build_dir.name}.axm')
    make_axm(build_dir, output)


if __name__ == '__main__':
    main()

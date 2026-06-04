#!/usr/bin/env python3
# Copyright Axelera AI, 2026
"""Package a compiled model directory into an .axm archive.

An .axm file is a ZIP archive containing all artifacts needed for on-device
inference: the compiled model, manifest, and optional pre/post-processing
ONNX graphs.

Supported input layouts (auto-detected):
    deploy.py:  build/<name>/<name>/1/manifest.json
    axcompile:  build/<name>/compiled_model/manifest.json
    direct:     <dir>/manifest.json  (when pointing at the model dir itself)

Manifest paths are rewritten to basenames so the flat .axm layout works
regardless of where the model was originally compiled.  Compiler-only
artifacts (benchmark reports, dependency graphs) are excluded.

Usage:
    python tools/make_axm.py build/yolo26n-coco-onnx/yolo26n-coco-onnx
    python tools/make_axm.py build/ssd-mobilenetv2-preamble
    python tools/make_axm.py build/ssd-mobilenetv2-preamble -o my_model.axm
"""
import argparse
import json
import sys
import zipfile
from pathlib import Path


_REQUIRED_FILES = {'manifest.json', 'model.json'}

# Compiler-only artifacts that are not needed at runtime
_SKIP_FILES = {'report.json', 'pass_benchmark_report.json', 'pass_dependency_graph.json'}

# Manifest fields that may contain absolute paths and need to be rewritten
# to basenames for the flat .axm layout.  Matches the compiler's own
# _make_manifest_relative() in top_level_utils.py.
_MANIFEST_PATH_FIELDS = (
    'model_lib_file',
    'model_params_file',
    'quantized_model_file',
    'preprocess_graph',
    'postprocess_graph',
)


def _find_model_dir(build_dir: Path) -> Path:
    """Find the directory containing manifest.json and model artifacts."""
    # deploy.py layout: build/<name>/<name>/1/
    version_dir = build_dir / '1'
    if version_dir.is_dir() and (version_dir / 'manifest.json').exists():
        return version_dir
    # axcompile layout: build/<name>/compiled_model/
    compiled_dir = build_dir / 'compiled_model'
    if compiled_dir.is_dir() and (compiled_dir / 'manifest.json').exists():
        return compiled_dir
    # Direct: build_dir itself contains manifest.json
    if (build_dir / 'manifest.json').exists():
        return build_dir
    sys.exit(
        f'Error: cannot find manifest.json in {build_dir}.\n'
        f'Looked in: {version_dir}, {compiled_dir}, {build_dir}'
    )


def _check_batch_size(model_dir: Path) -> None:
    """Verify all input shapes have batch=1. .axm only supports single-batch inference."""
    manifest_path = model_dir / 'manifest.json'
    if not manifest_path.exists():
        return
    manifest = json.loads(manifest_path.read_text())
    for shape in manifest.get('input_shapes', []):
        if shape and shape[0] != 1:
            sys.exit(
                f'Error: .axm only supports batch=1, but manifest has input shape {shape}. '
                f'Re-compile the model with batch_size=1.'
            )


def _patch_manifest(model_dir: Path) -> bytes | None:
    """Rewrite absolute paths in manifest to basenames for the flat .axm layout.

    Returns patched JSON bytes if any field was rewritten, None otherwise.
    """
    manifest_path = model_dir / 'manifest.json'
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text())
    changed = False
    for key in _MANIFEST_PATH_FIELDS:
        val = manifest.get(key)
        if val and '/' in str(val):
            manifest[key] = Path(val).name
            changed = True
    if changed:
        return json.dumps(manifest, indent=2).encode()
    return None


def make_axm(build_dir: Path, output: Path) -> None:
    model_dir = _find_model_dir(build_dir)
    _check_batch_size(model_dir)
    model_files = [f for f in model_dir.iterdir() if f.is_file() and f.name not in _SKIP_FILES]

    missing = _REQUIRED_FILES - {f.name for f in model_files}
    if missing:
        print(f'Warning: missing required files: {missing}', file=sys.stderr)

    if not model_files:
        sys.exit('Error: no files to package')

    manifest_patched = _patch_manifest(model_dir)

    with zipfile.ZipFile(output, 'w', zipfile.ZIP_STORED) as zf:
        for f in model_files:
            if f.name == 'manifest.json' and manifest_patched:
                zf.writestr('manifest.json', manifest_patched)
            else:
                zf.write(f, f.name)

    total_mb = output.stat().st_size / (1024 * 1024)
    print(f'Created {output} ({total_mb:.1f} MB, {len(model_files)} files from {model_dir.name}/)')


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

#!/usr/bin/env python3
# Copyright Axelera AI, 2026
"""Convert compiler configuration from JSON format to TOML format.

This tool helps users convert existing JSON compiler configurations to the
new TOML format. It can convert:
1. Standalone JSON files (e.g., compile_config.json from build directories)
2. Inline JSON from YAML model cards (compilation_config field)

The TOML format is the preferred way to specify compiler configurations as it:
- Reduces duplication across model cards
- Provides a single source of truth
- Is easier to read and maintain
"""

import argparse
import json
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from axelera.app import utils


def json_to_toml(json_data: dict, exclude_defaults: bool = True) -> str:
    """Convert JSON compiler config to TOML format.

    Args:
        json_data: Dictionary with compiler configuration
        exclude_defaults: If True, only include non-default values

    Returns:
        TOML formatted string
    """
    # CompilerConfig default values (from axelera.compiler.config.CompilerConfig)
    # These are typical defaults - actual defaults may vary by SDK version
    COMMON_DEFAULTS = {
        'quantization_scheme': 'per_tensor_min_max',
        'ignore_weight_buffers': False,
        'tiling_depth': 7,
        'split_buffer_promotion': False,
        'dfs_search_constraint': 3,
        'compiler_mode': 'compile',
        'resources_used': 1,
    }

    lines = []
    for key, value in sorted(json_data.items()):
        # Skip null values
        if value is None:
            continue

        # Skip defaults if requested
        if exclude_defaults and key in COMMON_DEFAULTS:
            if COMMON_DEFAULTS[key] == value:
                continue

        # Skip internal fields that shouldn't be in TOML
        if key in {
            'compiler_mode',
            'quantized_model_debug_save_dir',
            'graph_cleaner_dump_core_onnx',
            'output_dir',
            'model_name',
            'aipu_cores',
            'model_metadata',
        }:
            continue

        # Format value for TOML
        if isinstance(value, bool):
            toml_value = 'true' if value else 'false'
        elif isinstance(value, str):
            toml_value = f'"{value}"'
        elif isinstance(value, (int, float)):
            toml_value = str(value)
        elif isinstance(value, list):
            # TOML array format
            toml_value = (
                '[' + ', '.join(f'"{v}"' if isinstance(v, str) else str(v) for v in value) + ']'
            )
        else:
            # Skip complex types
            continue

        lines.append(f'{key} = {toml_value}')

    return '\n'.join(lines)


def convert_json_file(
    json_path: Path, output_path: Path = None, exclude_defaults: bool = True, force: bool = False
):
    """Convert a JSON file to TOML format.

    Args:
        json_path: Path to JSON file
        output_path: Output TOML path (default: same name with .toml extension)
        exclude_defaults: Only include non-default values
        force: Overwrite existing output file
    """
    if not json_path.exists():
        print(f"ERROR: JSON file not found: {json_path}")
        return False

    # Determine output path
    if output_path is None:
        output_path = json_path.with_suffix('.toml')

    if output_path.exists() and not force:
        print(f"ERROR: Output file already exists: {output_path}")
        print("Use --force to overwrite")
        return False

    # Load JSON
    try:
        with open(json_path) as f:
            json_data = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in {json_path}: {e}")
        return False

    # Convert to TOML
    toml_content = json_to_toml(json_data, exclude_defaults)

    if not toml_content.strip():
        print(f"WARNING: No non-default settings found in {json_path}")
        if not exclude_defaults:
            print("All settings appear to be defaults or unsupported in TOML format")
        return False

    # Write TOML
    output_path.write_text(toml_content + '\n')
    print(f"✓ Converted {json_path.name} → {output_path.name}")
    print(f"  Output: {output_path}")
    return True


def convert_yaml_inline(yaml_path: Path, output_dir: Path = None, exclude_defaults: bool = True):
    """Extract compilation_config from YAML and convert to TOML.

    Args:
        yaml_path: Path to YAML model card
        output_dir: Directory for output TOML (default: same as YAML)
        exclude_defaults: Only include non-default values
    """
    if not yaml_path.exists():
        print(f"ERROR: YAML file not found: {yaml_path}")
        return False

    # Load YAML
    try:
        yaml_data = utils.load_yamlfile(str(yaml_path))
    except Exception as e:
        print(f"ERROR: Failed to load YAML {yaml_path}: {e}")
        return False

    # Find compilation_config in models
    models = yaml_data.get('models', {})
    if not models:
        print(f"ERROR: No models found in {yaml_path}")
        return False

    converted_count = 0
    for model_name, model_config in models.items():
        extra_kwargs = model_config.get('extra_kwargs') or {}

        if 'compilation_config' not in extra_kwargs:
            continue

        compilation_config = extra_kwargs['compilation_config']
        if not compilation_config:
            continue

        # Determine output path
        if output_dir is None:
            output_dir = yaml_path.parent

        output_path = output_dir / f"{model_name}.toml"

        # Convert to TOML
        toml_content = json_to_toml(compilation_config, exclude_defaults)

        if not toml_content.strip():
            print(f"  {model_name}: No non-default settings found")
            continue

        # Write TOML
        output_path.write_text(toml_content + '\n')
        print(f"✓ {model_name}")
        print(f"  Output: {output_path}")
        converted_count += 1

    if converted_count == 0:
        print(f"No compilation_config found in {yaml_path}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(
        description="Convert compiler configurations from JSON to TOML format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert a compile_config.json file
  %(prog)s build/yolov8n/yolov8n/compile_config.json

  # Convert with custom output name
  %(prog)s compile_config.json -o my-model_mc-100.toml

  # Extract compilation_config from YAML model card
  %(prog)s --from-yaml ax_models/model_cards/yolo/yolov8n-coco.yaml

  # Include all settings (not just non-defaults)
  %(prog)s compile_config.json --no-exclude-defaults

  # Overwrite existing output file
  %(prog)s compile_config.json --force
        """,
    )

    parser.add_argument(
        'input', nargs='?', help='JSON file to convert (or YAML if using --from-yaml)'
    )
    parser.add_argument('-o', '--output', type=Path, help='Output TOML file path')
    parser.add_argument(
        '--from-yaml', action='store_true', help='Extract compilation_config from YAML model card'
    )
    parser.add_argument(
        '--output-dir', type=Path, help='Output directory for TOML files (with --from-yaml)'
    )
    parser.add_argument(
        '--no-exclude-defaults', action='store_true', help='Include default values in TOML output'
    )
    parser.add_argument('--force', action='store_true', help='Overwrite existing output files')

    args = parser.parse_args()

    if not args.input:
        parser.print_help()
        return 1

    input_path = Path(args.input)
    exclude_defaults = not args.no_exclude_defaults

    if args.from_yaml:
        success = convert_yaml_inline(
            input_path, output_dir=args.output_dir, exclude_defaults=exclude_defaults
        )
    else:
        success = convert_json_file(
            input_path,
            output_path=args.output,
            exclude_defaults=exclude_defaults,
            force=args.force,
        )

    return 0 if success else 1


if __name__ == '__main__':
    sys.exit(main())

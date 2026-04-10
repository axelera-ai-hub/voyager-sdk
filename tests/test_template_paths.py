# Copyright Axelera AI, 2026
"""Test to validate all template_path references in model cards point to existing files"""

import os
from pathlib import Path
import pytest
import yaml


def get_framework_root():
    return Path(__file__).parent.parent


def find_all_model_cards():
    """Find all YAML files in ax_models/model_cards/ excluding template directories"""
    framework_root = get_framework_root()
    model_cards_dir = framework_root / 'ax_models' / 'model_cards'

    if not model_cards_dir.exists():
        pytest.skip(f"Model cards directory not found: {model_cards_dir}")

    yaml_files = []
    for yaml_file in model_cards_dir.rglob('*.yaml'):
        # Exclude template directories
        if 'template' not in yaml_file.parts:
            yaml_files.append(yaml_file)

    for yaml_file in model_cards_dir.rglob('*.yml'):
        if 'template' not in yaml_file.parts:
            yaml_files.append(yaml_file)

    return yaml_files


def extract_template_path(yaml_content):
    """Extract template_path from parsed YAML content"""
    if 'pipeline' in yaml_content and isinstance(yaml_content['pipeline'], list):
        for stage in yaml_content['pipeline']:
            if isinstance(stage, dict):
                for key, value in stage.items():
                    if isinstance(value, dict) and 'template_path' in value:
                        return value['template_path']
    return None


def resolve_path(template_path):
    template_path = template_path.replace('$AXELERA_FRAMEWORK', str(get_framework_root()))
    return os.path.expandvars(template_path)


def test_template_paths_exist():
    """Verify all template_path references in model cards point to existing files"""
    framework_root = get_framework_root()
    model_cards = find_all_model_cards()

    if not model_cards:
        pytest.skip("No model card YAML files found")

    errors = []
    checked_count = 0

    for yaml_file in model_cards:
        with open(yaml_file, 'r') as f:
            content = yaml.safe_load(f)

        if not content:
            continue

        template_path = extract_template_path(content)

        if template_path:
            checked_count += 1
            resolved_path = resolve_path(template_path)

            if not os.path.exists(resolved_path):
                rel_yaml_path = yaml_file.relative_to(framework_root)
                errors.append(
                    f"{rel_yaml_path}: template_path '{template_path}' "
                    f"resolves to '{resolved_path}' which does not exist"
                )

    if errors:
        error_msg = f"Found {len(errors)} model card(s) with invalid template_path references:\n\n"
        error_msg += "\n".join(f"  - {err}" for err in errors)
        error_msg += f"\n\nChecked {checked_count} model card(s) with template_path references."
        pytest.fail(error_msg)

    assert checked_count > 0, (
        f"No model cards with template_path found. "
        f"Checked {len(model_cards)} YAML files in ax_models/model_cards/"
    )


def test_missing_template_error_message():
    """Test that missing template files produce clear error messages"""
    from axelera.app.schema import generated

    framework_root = get_framework_root()

    test_network = {
        'pipeline': [
            {
                'test_task': {
                    'template_path': str(
                        framework_root / 'pipeline-template' / 'NONEXISTENT_TEMPLATE.yaml'
                    )
                }
            }
        ]
    }

    with pytest.raises(FileNotFoundError) as exc_info:
        generated._find_template_operators(test_network)

    error_msg = str(exc_info.value)

    assert (
        "Template file not found" in error_msg
    ), "Error message should mention 'Template file not found'"
    assert (
        "NONEXISTENT_TEMPLATE.yaml" in error_msg
    ), "Error message should include the actual template path"
    assert (
        "Check that the template_path" in error_msg
    ), "Error message should provide helpful guidance"

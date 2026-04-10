# Copyright Axelera AI, 2026
"""Tests for compiler TOML config integration."""

import tempfile
from pathlib import Path
import pytest

from axelera.app import config

# Check if compiler is available for tests that need it
COMPILER_AVAILABLE = False
try:
    from axelera.compiler.config import CompilerConfig  # noqa: F401

    COMPILER_AVAILABLE = True
except ImportError:
    pass

# Create pytest marker for compiler-dependent tests
requires_compiler = pytest.mark.skipif(
    not COMPILER_AVAILABLE, reason="axelera.compiler not available (wheel not installed)"
)


def test_resolve_toml_path_absolute():
    """Test resolving absolute TOML path."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write('[test]\nvalue = 1\n')
        toml_path = Path(f.name)

    try:
        result = config.resolve_toml_path(str(toml_path))
        assert result['path'] == toml_path
        assert result['path'].exists()
        assert result['source'] == 'local'
        assert result['original_ref'] == str(toml_path)
    finally:
        toml_path.unlink()


def test_resolve_toml_path_relative():
    """Test resolving relative TOML path."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)
        subdir = tmpdir / "configs"
        subdir.mkdir()

        toml_file = subdir / "test.toml"
        toml_file.write_text('[test]\nvalue = 1\n')

        # Resolve relative path from parent directory
        result = config.resolve_toml_path("configs/test.toml", yaml_dir=tmpdir)
        assert result['path'] == toml_file
        assert result['path'].exists()
        assert result['source'] == 'local'


def test_resolve_toml_path_filename_only_compiler_configs():
    """Test resolving filename-only path from compiler configs directory."""
    try:
        import axelera.compiler.config as compiler_config_pkg

        configs_dir = Path(compiler_config_pkg.__file__).parent / "models"

        if configs_dir.exists():
            toml_files = list(configs_dir.glob("*.toml"))
            if toml_files:
                toml_file = toml_files[0]
                result = config.resolve_toml_path(toml_file.name)
                assert result['path'] == toml_file
                assert result['path'].exists()
                assert result['source'] == 'compiler_wheel'
            else:
                pytest.skip("No TOML files found in compiler configs directory")
        else:
            pytest.skip("Compiler configs models directory not found")
    except ImportError:
        pytest.skip("axelera.compiler not available (wheel not installed)")


def test_resolve_toml_path_not_found():
    """Test that FileNotFoundError is raised for non-existent file."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        with pytest.raises(FileNotFoundError) as exc_info:
            config.resolve_toml_path("nonexistent.toml", yaml_dir=tmpdir)

        # Check error message includes attempted paths and default suggestion
        assert "nonexistent.toml" in str(exc_info.value)
        assert "comment out" in str(exc_info.value)


def test_resolve_toml_path_not_found_suggests_close_match():
    """Test that FileNotFoundError suggests closest matching TOML files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create some TOML files to match against
        (tmpdir / "non-existent11n.toml").write_text("[test]\n")
        (tmpdir / "non-existent11s.toml").write_text("[test]\n")
        (tmpdir / "non-existentv8s.toml").write_text("[test]\n")

        with pytest.raises(FileNotFoundError) as exc_info:
            config.resolve_toml_path("non-existent11.toml", yaml_dir=tmpdir)

        error_msg = str(exc_info.value)
        assert "Did you mean" in error_msg
        assert "non-existent11n.toml" in error_msg or "non-existent11s.toml" in error_msg


def test_load_toml_config():
    """Test loading TOML config file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write(
            '''
[compiler]
quantization_scheme = "per_tensor_min_max"
aipu_cores_used = 4
'''
        )
        toml_path = Path(f.name)

    try:
        result = config.load_toml_config(toml_path)
        assert isinstance(result, dict)
        # Note: TOML structure may vary, so check for either flat or nested
        if 'compiler' in result:
            assert result['compiler']['quantization_scheme'] == 'per_tensor_min_max'
            assert result['compiler']['aipu_cores_used'] == 4
        else:
            assert result.get('quantization_scheme') == 'per_tensor_min_max'
            assert result.get('aipu_cores_used') == 4
    finally:
        toml_path.unlink()


def test_load_toml_config_invalid_syntax():
    """Test that ValueError is raised for invalid TOML syntax."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.toml', delete=False) as f:
        f.write('this is not valid TOML [')
        toml_path = Path(f.name)

    try:
        with pytest.raises(ValueError) as exc_info:
            config.load_toml_config(toml_path)

        assert "Invalid TOML syntax" in str(exc_info.value)
    finally:
        toml_path.unlink()


@requires_compiler
def test_gen_compilation_config_with_toml():
    """Test gen_compilation_config with TOML file reference."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a test TOML file
        toml_file = tmpdir / "test_config.toml"
        toml_file.write_text(
            '''
quantization_scheme = "per_tensor_min_max"
'''
        )

        # Create user config with compiler_config_file
        user_cfg = {
            'compiler_config_file': 'test_config.toml',
        }

        # Call gen_compilation_config
        result, metadata = config.gen_compilation_config(
            deploy_cores=4,
            user_cfg=user_cfg,
            deploy_mode=config.DeployMode.QUANTCOMPILE,
            yaml_dir=tmpdir,
        )

        # Check that TOML settings were applied
        assert result.quantization_scheme.value == "per_tensor_min_max"

        # Check metadata
        assert metadata['toml_info'] is not None
        assert metadata['toml_info']['source'] == 'local'
        assert 'quantization_scheme' in metadata['toml_fields']


@requires_compiler
def test_gen_compilation_config_yaml_overrides_toml():
    """Test that inline YAML config overrides TOML config."""
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir = Path(tmpdir)

        # Create a test TOML file
        toml_file = tmpdir / "test_config.toml"
        toml_file.write_text(
            '''
quantization_scheme = "per_tensor_min_max"
'''
        )

        # Create user config with both TOML and inline overrides
        user_cfg = {
            'compiler_config_file': 'test_config.toml',
            'compilation_config': {
                'quantization_scheme': 'per_tensor_histogram',  # Should override TOML
            },
        }

        # Call gen_compilation_config
        result, metadata = config.gen_compilation_config(
            deploy_cores=4,
            user_cfg=user_cfg,
            deploy_mode=config.DeployMode.QUANTCOMPILE,
            yaml_dir=tmpdir,
        )

        # Check that YAML override took precedence
        assert result.quantization_scheme.value == "per_tensor_histogram"

        # Check that override was tracked
        assert 'quantization_scheme' in metadata['yaml_overrides']
        assert 'quantization_scheme' in metadata['toml_fields']


@requires_compiler
def test_gen_compilation_config_backward_compat():
    """Test backward compatibility - works without compiler_config_file."""
    user_cfg = {
        'compilation_config': {
            'quantization_scheme': 'per_tensor_histogram',
        }
    }

    # Call gen_compilation_config without yaml_dir
    result, metadata = config.gen_compilation_config(
        deploy_cores=4, user_cfg=user_cfg, deploy_mode=config.DeployMode.QUANTCOMPILE
    )

    # Check that inline config still works
    assert result.quantization_scheme.value == "per_tensor_histogram"

    # Check that no TOML was used
    assert metadata['toml_info'] is None
    assert 'quantization_scheme' in metadata['yaml_overrides']


@requires_compiler
def test_config_available():
    """Test that compiler config module is available."""
    try:
        # this import will fail until axelera-ai/software-platform#5089 merges
        from axelera.compiler import config as compiler_config

        assert compiler_config is not None
        assert hasattr(compiler_config, 'CompilerConfig')
    except ImportError:
        pytest.fail("compiler config module should be available but is not")

# Copyright Axelera AI, 2025
from pathlib import Path
from unittest.mock import Mock, patch

from axelera.types import Manifest, ModelInfo, OutputInfo
import numpy as np
import pytest

from axelera.app.operators.inference import (
    InferenceOpConfig,
    _match_arrays_to_shapes,
    _reshape_to_target_shapes,
    _supports_async_executor,
)


class TestArrayShapeMatching:
    """Tests for the _match_arrays_to_shapes and _reshape_to_target_shapes utility functions."""

    def test_match_arrays_perfect_match(self):
        """Test matching arrays to shapes when there's a perfect one-to-one match."""
        # Create test arrays with different shapes
        arrays = [
            np.zeros((1, 12, 19, 19)),  # 4,332 elements
            np.zeros((1, 24, 10, 10)),  # 2,400 elements
            np.zeros((1, 546, 1, 1)),  # 546 elements
        ]

        # Create target shapes with the same number of elements
        target_shapes = [
            (1, 546, 1, 1),  # 546 elements
            (1, 24, 10, 10),  # 2,400 elements
            (1, 12, 19, 19),  # 4,332 elements
        ]

        # Match arrays to shapes
        matched, unmatched, unmatched_idx = _match_arrays_to_shapes(arrays, target_shapes)

        # Verify all arrays were matched
        assert len(matched) == 3
        assert len(unmatched) == 0
        assert len(unmatched_idx) == 0

        # Verify matches are correct (based on element count)
        sizes = [(arr.size, shape, idx) for arr, shape, idx in matched]
        assert (4332, (1, 12, 19, 19), 2) in sizes
        assert (2400, (1, 24, 10, 10), 1) in sizes
        assert (546, (1, 546, 1, 1), 0) in sizes

    def test_match_arrays_some_matches(self):
        """Test matching arrays to shapes when only some arrays match available shapes."""
        # Create test arrays
        arrays = [
            np.zeros((1, 12, 19, 19)),  # 4,332 elements
            np.zeros((1, 24, 10, 10)),  # 2,400 elements
            np.zeros((1, 546, 1, 1)),  # 546 elements
            np.zeros((1, 273, 19, 19)),  # 98,439 elements - no match
        ]

        # Create target shapes (missing one)
        target_shapes = [
            (1, 24, 10, 10),  # 2,400 elements
            (1, 12, 19, 19),  # 4,332 elements
            (1, 100, 1, 1),  # 100 elements - no match
        ]

        # Match arrays to shapes
        matched, unmatched, unmatched_idx = _match_arrays_to_shapes(arrays, target_shapes)

        # Verify correct number of matches and unmatched items
        assert len(matched) == 2
        assert len(unmatched) == 2
        assert len(unmatched_idx) == 1

        # Verify specific matches
        assert any(arr.shape == (1, 12, 19, 19) for arr, _, _ in matched)
        assert any(arr.shape == (1, 24, 10, 10) for arr, _, _ in matched)

        # Verify unmatched arrays
        unmatched_shapes = [arr.shape for arr in unmatched]
        assert (1, 546, 1, 1) in unmatched_shapes
        assert (1, 273, 19, 19) in unmatched_shapes

        # Verify unmatched target shape index
        assert 2 in unmatched_idx  # The 100-element shape wasn't matched

    def test_match_arrays_with_duplicates(self):
        """Test matching when there are duplicate sizes."""
        # Create arrays with some duplicate sizes
        arrays = [
            np.zeros((1, 10, 10, 10)),  # 1,000 elements
            np.zeros((1, 10, 10, 10)),  # 1,000 elements (duplicate)
            np.zeros((1, 20, 5, 10)),  # 1,000 elements (different shape, same size)
        ]

        # Create target shapes with some duplicate sizes
        target_shapes = [
            (1, 1000, 1, 1),  # 1,000 elements
            (1, 10, 10, 10),  # 1,000 elements (duplicate)
            (1, 25, 40, 1),  # 1,000 elements (different shape, same size)
        ]

        # Match arrays to shapes
        matched, unmatched, unmatched_idx = _match_arrays_to_shapes(arrays, target_shapes)

        # Verify all arrays matched (since we have enough target shapes of the right size)
        assert len(matched) == 3
        assert len(unmatched) == 0
        assert len(unmatched_idx) == 0

    def test_reshape_to_target_shapes(self):
        """Test reshaping arrays to target shapes."""
        # Create test arrays
        arrays = [
            np.ones((1, 12, 19, 19)),  # 4,332 elements
            np.ones((1, 24, 10, 10)) * 2,  # 2,400 elements
            np.ones((1, 546, 1, 1)) * 3,  # 546 elements
        ]

        # Create target shapes in different order
        target_shapes = [
            (1, 546, 1, 1),  # 546 elements
            (1, 24, 10, 10),  # 2,400 elements
            (1, 12, 19, 19),  # 4,332 elements
        ]

        # Reshape arrays
        reshaped = _reshape_to_target_shapes(arrays, target_shapes)

        # Verify correct shapes
        assert len(reshaped) == 3

        # Check that shapes match expected and values are preserved
        shapes = [arr.shape for arr in reshaped]
        assert (1, 12, 19, 19) in shapes
        assert (1, 24, 10, 10) in shapes
        assert (1, 546, 1, 1) in shapes

        # Find array with value 3 (should be the 546-element one)
        for arr in reshaped:
            if arr.shape == (1, 546, 1, 1):
                assert arr[0, 0, 0, 0] == 3


class TestScalarOutputHelper:
    """Tests for the _is_scalar_output helper method."""

    def test_scalar_output_1x1(self):
        """Test that 1x1 outputs are correctly identified as scalar."""
        config = InferenceOpConfig()

        # Test various 1x1 configurations
        assert config._is_scalar_output([1, 1, 1, 1000])  # Standard classifier
        assert config._is_scalar_output([1, 1, 1, 10])  # Small classifier
        assert config._is_scalar_output([5, 1, 1, 256])  # Batch size > 1

    def test_non_scalar_outputs(self):
        """Test that non-1x1 outputs are not identified as scalar."""
        config = InferenceOpConfig()

        # Test various non-scalar configurations
        assert not config._is_scalar_output([1, 224, 224, 3])  # Typical image
        assert not config._is_scalar_output([1, 112, 112, 64])  # Downsampled feature map
        assert not config._is_scalar_output([1, 28, 28, 256])  # Small feature map
        assert not config._is_scalar_output([1, 7, 7, 512])  # Even smaller feature map
        assert not config._is_scalar_output([1, 2, 1, 100])  # Only width is 1
        assert not config._is_scalar_output([1, 1, 2, 100])  # Only height is 1

    def test_edge_cases(self):
        """Test edge cases for shape validation."""
        config = InferenceOpConfig()

        # Test insufficient dimensions
        assert not config._is_scalar_output([1])  # 1D
        assert not config._is_scalar_output([1, 1])  # 2D
        assert not config._is_scalar_output([])  # Empty

    def test_different_tensor_formats(self):
        """Test with different tensor dimension layouts."""
        config = InferenceOpConfig()

        # Test 4D tensors (typical case)
        assert config._is_scalar_output([1, 1, 1, 1000])  # NHWC classifier
        assert not config._is_scalar_output([1, 224, 224, 3])  # NHWC image

        # Test 5D tensors (e.g., video or sequence data)
        assert config._is_scalar_output([1, 1, 1, 1000, 1])  # 5D scalar-like
        assert not config._is_scalar_output([1, 16, 224, 224, 3])  # 5D video


class TestTransposeLogic:
    """Tests for the improved transpose logic using OutputInfo."""

    def setup_method(self):
        """Set up test fixtures."""
        self.config = InferenceOpConfig(handle_transpose=True)

    def create_mock_model_info(self, output_infos, input_height=224, input_width=224):
        """Create a mock ModelInfo with specified output infos."""
        model_info = Mock(spec=ModelInfo)
        model_info.output_info = output_infos
        model_info.input_height = input_height
        model_info.input_width = input_width

        # Create mock manifest
        manifest = Mock(spec=Manifest)
        model_info.manifest = manifest

        return model_info

    def test_transpose_not_needed_for_matching_shapes(self):
        """Test that transpose is not needed when depadded shape matches OutputInfo shape."""
        # Create OutputInfo for a segmentation model (same spatial dims as input)
        output_info = OutputInfo(shape=(1, 224, 224, 19), name="output", dtype="float32")

        # Mock manifest with padding
        manifest = Mock()
        manifest.n_padded_ch_outputs = [[0, 0, 0, 0, 0, 0, 0, 45]]  # padding on channels only

        model_info = self.create_mock_model_info([output_info])
        model_info.manifest = manifest

        # Manifest output shape (padded)
        manifest_output_shape = [1, 224, 224, 64]  # 19 + 45 = 64

        with patch('axelera.app.compile.get_original_shape') as mock_get_original_shape:
            # Mock get_original_shape to return the depadded shape
            mock_get_original_shape.return_value = [(1, 224, 224, 19)]

            needs_transpose = self.config._determine_transpose_from_output_info(
                output_info, manifest_output_shape, model_info, 0
            )

            # Should not need transpose since shapes match
            assert not needs_transpose

    def test_transpose_needed_for_different_shapes(self):
        """Test that transpose is needed when depadded shape differs from OutputInfo shape."""
        # Create OutputInfo with different spatial dimensions
        output_info = OutputInfo(shape=(1, 112, 112, 19), name="output", dtype="float32")

        # Mock manifest with padding
        manifest = Mock()
        manifest.n_padded_ch_outputs = [[0, 0, 0, 0, 0, 0, 0, 45]]  # padding on channels only

        model_info = self.create_mock_model_info([output_info])
        model_info.manifest = manifest

        # Manifest output shape (padded) with different spatial dims
        manifest_output_shape = [1, 224, 224, 64]  # Different H,W than OutputInfo

        with patch('axelera.app.compile.get_original_shape') as mock_get_original_shape:
            # Mock get_original_shape to return the depadded shape
            mock_get_original_shape.return_value = [(1, 224, 224, 19)]

            needs_transpose = self.config._determine_transpose_from_output_info(
                output_info, manifest_output_shape, model_info, 0
            )

            # Should need transpose since shapes differ
            assert needs_transpose

    def test_transpose_not_needed_for_classifier_output(self):
        """Test that transpose is not needed for 1x1 classifier outputs."""
        # Create OutputInfo for a classifier (1x1 spatial dims)
        output_info = OutputInfo(shape=(1, 1, 1, 1000), name="output", dtype="float32")

        # Mock manifest with no padding
        manifest = Mock()
        manifest.n_padded_ch_outputs = [[0, 0, 0, 0, 0, 0, 0, 0]]  # no padding

        model_info = self.create_mock_model_info([output_info])
        model_info.manifest = manifest

        # Manifest output shape for classifier
        manifest_output_shape = [1, 1, 1, 1000]

        with patch('axelera.app.compile.get_original_shape') as mock_get_original_shape:
            # Mock get_original_shape to return the same shape
            mock_get_original_shape.return_value = [(1, 1, 1, 1000)]

            needs_transpose = self.config._determine_transpose_from_output_info(
                output_info, manifest_output_shape, model_info, 0
            )

            # Should not need transpose for 1x1 outputs
            assert not needs_transpose

    def test_fallback_to_heuristic_when_no_manifest(self):
        """Test fallback to heuristic when manifest is not available."""
        output_info = OutputInfo(shape=(1, 224, 224, 19), name="output", dtype="float32")

        model_info = self.create_mock_model_info([output_info])
        model_info.manifest = None  # No manifest

        manifest_output_shape = [1, 224, 224, 19]

        needs_transpose = self.config._determine_transpose_from_output_info(
            output_info, manifest_output_shape, model_info, 0
        )

        # Should use heuristic: transpose for non-1x1 outputs (since 224x224 is not 1x1)
        assert needs_transpose

    def test_fallback_to_heuristic_when_no_padding_info(self):
        """Test fallback to heuristic when padding info is not available."""
        output_info = OutputInfo(shape=(1, 224, 224, 19), name="output", dtype="float32")

        manifest = Mock()
        manifest.n_padded_ch_outputs = None  # No padding info

        model_info = self.create_mock_model_info([output_info])
        model_info.manifest = manifest

        manifest_output_shape = [1, 1, 1, 1000]  # 1x1 classifier-like output

        needs_transpose = self.config._determine_transpose_from_output_info(
            output_info, manifest_output_shape, model_info, 0
        )

        # Should use heuristic: no transpose for 1x1 outputs
        assert not needs_transpose

    def test_reconcile_manifest_with_output_info(self):
        """Test the complete reconcile_manifest method with OutputInfo."""
        output_infos = [
            OutputInfo(shape=(1, 224, 224, 19), name="segmentation", dtype="float32"),
            OutputInfo(shape=(1, 1, 1, 1000), name="classification", dtype="float32"),
        ]

        manifest = Mock()
        manifest.output_shapes = [
            [1, 224, 224, 64],  # Segmentation output (padded)
            [1, 1, 1, 1000],  # Classification output (no padding)
        ]
        manifest.n_padded_ch_outputs = [
            [0, 0, 0, 0, 0, 0, 0, 45],  # padding for segmentation
            [0, 0, 0, 0, 0, 0, 0, 0],  # no padding for classification
        ]
        manifest.postprocess_graph = None

        model_info = self.create_mock_model_info(output_infos)
        model_info.manifest = manifest

        with patch('axelera.app.compile.get_original_shape') as mock_get_original_shape:
            # Mock get_original_shape calls
            mock_get_original_shape.side_effect = [
                [(1, 224, 224, 19)],  # First call for segmentation
                [(1, 1, 1, 1000)],  # Second call for classification
            ]

            self.config.reconcile_manifest(model_info)

            # Check transpose decisions
            assert len(self.config._transpose_aipu_output) == 2
            assert not self.config._transpose_aipu_output[
                0
            ]  # No transpose for matching segmentation
            assert not self.config._transpose_aipu_output[1]  # No transpose for 1x1 classification

    def test_exception_handling_in_transpose_logic(self):
        """Test that exceptions in get_original_shape are handled gracefully."""
        output_info = OutputInfo(shape=(1, 224, 224, 19), name="output", dtype="float32")

        manifest = Mock()
        manifest.n_padded_ch_outputs = [[0, 0, 0, 0, 0, 0, 0, 45]]

        model_info = self.create_mock_model_info([output_info])
        model_info.manifest = manifest

        manifest_output_shape = [1, 1, 1, 1000]  # 1x1 output for fallback test

        with patch('axelera.app.compile.get_original_shape') as mock_get_original_shape:
            # Mock get_original_shape to raise an exception
            mock_get_original_shape.side_effect = Exception("Test exception")

            with patch('axelera.app.operators.inference.LOG') as mock_log:
                needs_transpose = self.config._determine_transpose_from_output_info(
                    output_info, manifest_output_shape, model_info, 0
                )

                # Should fall back to heuristic and log warning
                mock_log.warning.assert_called_once()
                assert not needs_transpose  # 1x1 output, no transpose

    def test_helper_method_integration_with_legacy_logic(self):
        """Test that the helper method works correctly in the legacy reconcile path."""
        manifest = Mock()
        manifest.output_shapes = [
            [1, 1, 1, 1000],  # Classifier output
            [1, 224, 224, 19],  # Segmentation output
            [1, 56, 56, 256],  # Feature map output
        ]
        manifest.n_padded_ch_outputs = None  # Force legacy path
        manifest.postprocess_graph = None

        model_info = Mock(spec=ModelInfo)
        model_info.output_info = None  # Force legacy path
        model_info.input_height = 224
        model_info.input_width = 224
        model_info.manifest = manifest

        with patch('axelera.app.operators.inference.LOG'):
            self.config.reconcile_manifest(model_info)

            # Check that legacy heuristic logic using helper method produces expected results
            assert len(self.config._transpose_aipu_output) == 3
            # First output: 1x1 classifier
            # Second output: 224x224 segmentation
            # Third output: 56x56 feature map
            assert [False, False, True] == self.config._transpose_aipu_output


class TestPostambleValidation:
    """Tests for the postamble validation logic in reconcile_manifest."""

    def setup_method(self):
        """Set up test fixtures."""
        # Don't set postamble_onnx in constructor to avoid validation
        self.config = InferenceOpConfig(handle_postamble=True)

    def create_mock_model_info(self):
        """Create a mock ModelInfo with manifest."""
        model_info = Mock(spec=ModelInfo)
        manifest = Mock(spec=Manifest)
        manifest.postprocess_graph = None
        manifest.output_shapes = []  # Add this to avoid AttributeError
        model_info.manifest = manifest
        return model_info

    def test_postamble_validation_file_exists(self):
        """Test that validation passes when postamble file exists."""
        # Set postamble_onnx after creation to avoid constructor validation
        self.config.postamble_onnx = "/path/to/postamble.onnx"
        model_info = self.create_mock_model_info()

        with patch('pathlib.Path.is_file') as mock_is_file:
            mock_is_file.return_value = True

            with patch('axelera.app.operators.inference.LOG') as mock_log:
                self.config.reconcile_manifest(model_info)

                # Verify info log was called
                mock_log.info.assert_called_once_with(
                    "Found custom postamble ONNX model '/path/to/postamble.onnx'."
                )

                # Verify manifest was updated
                assert model_info.manifest.postprocess_graph == "/path/to/postamble.onnx"

    def test_postamble_validation_file_not_exists(self):
        """Test that ValueError is raised when postamble file doesn't exist."""
        # Set postamble_onnx after creation to avoid constructor validation
        self.config.postamble_onnx = "/path/to/postamble.onnx"
        model_info = self.create_mock_model_info()

        with patch('pathlib.Path.is_file') as mock_is_file:
            mock_is_file.return_value = False

            with pytest.raises(
                ValueError,
                match="Custom postamble ONNX model '/path/to/postamble.onnx' does not exist.",
            ):
                self.config.reconcile_manifest(model_info)

    def test_postamble_validation_with_existing_manifest_postprocess(self):
        """Test that warning is logged when manifest already has postprocess_graph."""
        # Set postamble_onnx after creation to avoid constructor validation
        self.config.postamble_onnx = "/path/to/postamble.onnx"
        model_info = self.create_mock_model_info()
        model_info.manifest.postprocess_graph = "/existing/postamble.onnx"

        with patch('pathlib.Path.is_file') as mock_is_file:
            mock_is_file.return_value = True

            with patch('axelera.app.operators.inference.LOG') as mock_log:
                self.config.reconcile_manifest(model_info)

                # Verify warning was logged
                mock_log.warning.assert_called_once_with(
                    "Duplicated postamble configuration detected. The ONNX postamble model "
                    "'/path/to/postamble.onnx' will be used, overriding the one specified in "
                    "the manifest '/existing/postamble.onnx'."
                )

                # Verify manifest was updated
                assert model_info.manifest.postprocess_graph == "/path/to/postamble.onnx"

    def test_postamble_validation_disabled(self):
        """Test that validation is skipped when handle_postamble is False."""
        config = InferenceOpConfig(handle_postamble=False)
        config.postamble_onnx = "/path/to/postamble.onnx"
        model_info = self.create_mock_model_info()

        with patch('pathlib.Path.is_file') as mock_is_file:
            config.reconcile_manifest(model_info)

            # Verify is_file was not called since validation is disabled
            mock_is_file.assert_not_called()

    def test_postamble_validation_no_postamble_onnx(self):
        """Test that validation is skipped when postamble_onnx is None."""
        config = InferenceOpConfig(handle_postamble=True)
        config.postamble_onnx = None
        model_info = self.create_mock_model_info()

        with patch('pathlib.Path.is_file') as mock_is_file:
            config.reconcile_manifest(model_info)

            # Verify is_file was not called since postamble_onnx is None
            mock_is_file.assert_not_called()

    def test_postamble_validation_path_object(self):
        """Test that validation works with Path objects."""
        config = InferenceOpConfig(handle_postamble=True)
        config.postamble_onnx = Path("/path/to/postamble.onnx")
        model_info = self.create_mock_model_info()

        with patch('pathlib.Path.is_file') as mock_is_file:
            mock_is_file.return_value = True

            with patch('axelera.app.operators.inference.LOG') as mock_log:
                config.reconcile_manifest(model_info)

                # Verify info log was called
                mock_log.info.assert_called_once_with(
                    "Found custom postamble ONNX model '/path/to/postamble.onnx'."
                )


class TestSupportsAsyncExecutor:
    """Tests for _supports_async_executor, which gates async_mode on model batching."""

    def test_single_core_model_supports_async(self):
        assert _supports_async_executor(1) is True

    def test_batched_model_does_not_support_async(self):
        assert _supports_async_executor(2) is False
        assert _supports_async_executor(4) is False

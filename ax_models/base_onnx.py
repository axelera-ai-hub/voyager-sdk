# Copyright Axelera AI, 2024
# General axelera.types.Model with ONNX model for object detection

from __future__ import annotations

from pathlib import Path
import typing

import numpy as np

try:
    import onnx
except ImportError:
    if typing.TYPE_CHECKING:
        import onnx

from axelera import types
from axelera.app import logging_utils, utils
import axelera.app.yaml as YAML

LOG = logging_utils.getLogger(__name__)


def apply_onnx_graph_optimizations(
    model: onnx.ModelProto, model_info: types.ModelInfo  # type: ignore[name-defined]
) -> onnx.ModelProto:  # type: ignore[name-defined]
    """
    Apply graph-level optimizations to an ONNX model.

    This function serves as a central point for applying various ONNX graph
    transformations that improve performance. New optimizations can be added
    here in the future.

    Current optimizations:
    - Focus layer replacement: Fuses Focus (Space-to-Depth) + Conv patterns
    - Gemm to Conv conversion: Converts large FC layers to Conv for better AIPU tiling
    - YOLO26 seg proto identity insertion: Inserts MaxPool identity ops to break dependency chains

    Args:
        model: ONNX model to optimize
        model_info: Model information containing configuration

    Returns:
        Optimized model
    """
    # Focus layer replacement (enabled by default, disable via YOLO.focus_layer_replacement=False)
    yolo_config = model_info.extra_kwargs.get('YOLO', {})
    if yolo_config.get('focus_layer_replacement', True):
        # Lazy import to avoid issues when onnx is not installed (e.g., py310-runtime test environment)
        from ax_models.onnx_optimizations import replace_focus_layer

        model = replace_focus_layer(model)

    # Gemm to Conv conversion (enabled by default for large FC layers)
    # This helps AIPU handle large fully-connected layers by allowing spatial tiling
    # Enable via extra_kwargs.gemm_to_conv_replacement=True
    if model_info.extra_kwargs.get('gemm_to_conv_replacement', False):
        from ax_models.onnx_optimizations import (
            replace_gemm_with_conv,
            validate_gemm_conv_equivalence,
        )

        original_model = model
        model = replace_gemm_with_conv(model)

        # Validate equivalence if model was modified
        if model is not original_model:
            if not validate_gemm_conv_equivalence(original_model, model):
                LOG.warning("Gemm->Conv conversion validation failed, reverting to original model")
                model = original_model

    # YOLO26 seg proto identity insertion (opt-in via model card YAML)
    # Inserts MaxPool identity operations to break problematic dependency chains in the
    # proto (segmentation mask prototype) section that cause compilation failures.
    # This transformation is required for YOLO26 seg models to compile on Axelera hardware.
    if model_info.extra_kwargs.get('yolo26_seg_insert_proto_identity', False):
        from ax_models.onnx_optimizations import insert_yolo26_seg_proto_identity_ops

        LOG.info(f"Applying YOLO26 seg proto identity insertion to {model_info.name}")
        model = insert_yolo26_seg_proto_identity_ops(model)

    return model


def update_model_specific_config(model_info: types.ModelInfo):
    """
    Load and update model specific configuration from the model info extra_kwargs.
    """
    YOLO_kwargs = model_info.extra_kwargs.get('YOLO', {})
    if YOLO_kwargs:
        # scale anchors by strides
        anchors = YOLO_kwargs.get('anchors', [])
        anchors_path = YOLO_kwargs.get('anchors_path', None)
        anchors_path = Path(anchors_path) if anchors_path else None
        anchors_url = YOLO_kwargs.get('anchors_url', None)
        anchors_md5 = YOLO_kwargs.get('anchors_md5', None)
        if anchors and anchors_path:
            LOG.warning(
                f'anchors and anchors_path have both been specified for {model_info.name} - ignoring anchors_path'
            )
            anchors_path = None
        if not anchors and anchors_path:
            utils.download_model_artifacts(
                anchors_path,
                anchors_url,
                anchors_md5,
                model_name=model_info.name,
                artifact='anchors',
            )
            LOG.debug(f'Load ONNX model anchors from anchors_path {anchors_path}')
            try:
                anchors = utils.load_yamlfile(anchors_path).get('anchors', [])
            except Exception as e:
                raise RuntimeError(f"Failed to find anchors in {anchors_path}")
        if anchors:
            strides = YOLO_kwargs.get('strides', [])
            if len(strides) == 0:  # default P3, P4, P5, P6, P7 strides
                strides = [8, 16, 32, 64, 128][: len(anchors)]
            else:
                assert len(strides) == len(
                    anchors
                ), 'strides and anchors must have the same length'
        # rewrite anchors by anchors/strides
        for i, anchor in enumerate(anchors):
            anchors[i] = [a / strides[i] for a in anchor]
        model_info.extra_kwargs['YOLO']['anchors'] = anchors


class AxONNXModel(types.ONNXModel):
    """Create an axelera.types.ONNXModel instance with auto download"""

    def init_model_deploy(self, model_info: types.ModelInfo, dataset_config: dict, **kwargs):
        weights = Path(model_info.weight_path)
        utils.download_model_artifacts(
            weights,
            model_info.weight_url,
            model_info.weight_md5,
            model_name=model_info.name,
        )
        LOG.debug(f'Load ONNX model with weights {weights}')
        self.onnx_model = onnx.load(weights)

        # Apply graph optimizations (always enabled, can be disabled via config)
        self.onnx_model = apply_onnx_graph_optimizations(self.onnx_model, model_info)

        update_model_specific_config(model_info)

# Copyright Axelera AI, 2024
# General axelera.types.Model with ONNX model for object detection

from pathlib import Path
import typing

import onnx

from axelera import types
from axelera.app import logging_utils, utils
import axelera.app.yaml as YAML

LOG = logging_utils.getLogger(__name__)


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
        if not anchors and anchors_path:
            if not anchors_path.exists() or (
                anchors_md5 and not utils.md5_validates(anchors_path, anchors_md5)
            ):
                if not anchors_url:
                    raise ValueError(
                        f'"anchors" not specified, no suitable anchors found at {anchors_path} and no anchors_url specified for {model_info.name}'
                    )
                try:
                    utils.download(anchors_url, anchors_path, anchors_md5)
                except Exception as e:
                    raise RuntimeError(
                        f'Failed to download {anchors_path} from {anchors_url}\n\t{e}'
                    ) from None
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

    def init_model_deploy(self, model_info: types.ModelInfo):
        weights = Path(model_info.weight_path)
        if not weights.exists() or (
            model_info.weight_md5 and not utils.md5_validates(weights, model_info.weight_md5)
        ):
            if not model_info.weight_url:
                raise ValueError(
                    f'No suitable weights found for {model_info.name} at {weights} and no weight_url specified'
                )
            try:
                utils.download(model_info.weight_url, weights, model_info.weight_md5)
            except Exception as e:
                raise RuntimeError(
                    f'Failed to download {weights} from {model_info.weight_url}\n\t{e}'
                ) from None
        LOG.debug(f'Load ONNX model with weights {weights}')
        self.onnx_model = onnx.load(weights)
        update_model_specific_config(model_info)

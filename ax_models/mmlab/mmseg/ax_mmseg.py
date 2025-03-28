# Axelera class for MMSegmentation
# Copyright Axelera AI, 2024

import os
from pathlib import Path
import typing

from mmengine.config import Config
from mmengine.registry import init_default_scope
import onnx
import torch

from ax_models import base_torch
from axelera import types
from axelera.app import logging_utils, utils
from axelera.app import yaml as YAML
from mmseg.apis import init_model

LOG = logging_utils.getLogger(__name__)
package_name = 'mmseg'
config_directory = '.mim/configs'


def find_config_file(filename):
    """
    Recursively search for a file within a directory.
    :param filename: The name of the file to search for.
    :return: The full path to the file if found, otherwise None.
    """
    import mmseg

    global config_directory

    mmseg_dir = os.path.dirname(mmseg.__file__)
    config_dir_path = os.path.join(mmseg_dir, config_directory)

    for root, _, files in os.walk(config_dir_path):
        if filename in files:
            return os.path.join(root, filename)
    return None


def replace_sync_bn(cfg):
    """
    Recursively search through the mmengine.config.config.Config object and replace 'SyncBN' with 'BN'.
    :param cfg: The Config object or a part of it.

    See https://github.com/open-mmlab/mmsegmentation/issues/292 and
        https://github.com/open-mmlab/mmdeploy/pull/1631
    """

    def handle_dict(d):
        for key, value in d.items():
            if key == 'norm_cfg' and value.get('type') == 'SyncBN':
                value['type'] = 'BN'
            elif isinstance(value, dict):
                handle_dict(value)
            elif isinstance(value, list):
                handle_list(value)

    def handle_list(l):
        for item in l:
            if isinstance(item, dict):
                handle_dict(item)
            elif isinstance(item, list):
                handle_list(item)

    if hasattr(cfg, 'items'):
        handle_dict(cfg)
    elif isinstance(cfg, list):
        handle_list(cfg)


class AxMMSegmentationBase:
    def __init__(self, **kwargs):
        types.Model.__init__(self)
        self.working_dir = str(Path.cwd())
        LOG.debug(f'Current working directory is {self.working_dir}')
        mmlab_args = YAML.attribute(kwargs, 'mmseg')
        config_file = YAML.attribute(mmlab_args, 'config_file')
        self.model_config_path = find_config_file(config_file)
        if self.model_config_path is None:
            raise ValueError(f'Config file {config_file} not found')
        self.cfg = Config.fromfile(self.model_config_path)
        self.set_shared_param('cfg', self.cfg)
        init_default_scope(self.cfg.get('default_scope', 'mmseg'))

        try:
            if self.cfg.data_preprocessor.bgr_to_rgb and self.input_color_format == 'BGR':
                raise ValueError("data_preprocessor is not set in the config file")
            self.set_shared_param(
                'color_format', 'RGB' if self.cfg.data_preprocessor.bgr_to_rgb else 'BGR'
            )
        except Exception as e:
            raise ValueError(f'Failed to determine color format from data_preprocessor: {e}')

        # TODO: support imreader_backend in YAML
        self.imreader_backend = 'opencv'
        # TODO: pipeline_input_color_format should come from Input Operator but not always the same as input_color_format
        self.pipeline_input_color_format = self.input_color_format
        # TODO: see if we can download the config file from MIM
        # from mmengine.hub import get_config
        # self.cfg = get_config(f'mmseg::{config_file}', pretrained=True)


class AxMMSegmentationPytorch(AxMMSegmentationBase, base_torch.TorchModel):
    def __init__(self, **kwargs):
        AxMMSegmentationBase.__init__(self, **kwargs)
        torch.nn.Module.__init__(self)
        replace_sync_bn(self.cfg)
        self.torch_model = init_model(self.cfg, device=torch.device('cpu'))

    def init_model_deploy(self, model_info: types.ModelInfo):
        checkpoint = Path(model_info.weight_path)
        if not (checkpoint.exists() and utils.md5_validates(checkpoint, model_info.weight_md5)):
            utils.download(model_info.weight_url, checkpoint, model_info.weight_md5)
        self.cfg.data_preprocessor = None
        self.cfg.model.data_preprocessor = None
        self.torch_model = init_model(self.cfg, str(checkpoint), device=torch.device('cpu'))
        self.eval()
        if hasattr(self.torch_model, 'data_preprocessor'):
            del self.torch_model.data_preprocessor


class AxMMSegmentationOnnx(AxMMSegmentationBase, types.ONNXModel):
    def __init__(self, **kwargs):
        AxMMSegmentationBase.__init__(self, **kwargs)

    def init_model_deploy(self, model_info: types.ModelInfo):
        checkpoint = Path(model_info.weight_path)
        if not (checkpoint.exists() and utils.md5_validates(checkpoint, model_info.weight_md5)):
            utils.download(model_info.weight_url, checkpoint, model_info.weight_md5)

        self.cfg.data_preprocessor = None
        self.cfg.model.data_preprocessor = None
        # # here we have to load transforms from the config file before loading the dataset
        # from mmseg.registry import DATASETS
        # from mmseg.datasets import transforms

        LOG.debug(f'Load ONNX model with weights {checkpoint}')
        self.onnx_model = onnx.load(checkpoint)

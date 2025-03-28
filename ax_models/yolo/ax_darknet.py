# Axelera class for PyTorch Darknet
# Copyright Axelera AI, 2023
from __future__ import annotations

from pathlib import Path
import typing

from axelera import types
from axelera.app import logging_utils, utils
from axelera.app import yaml as YAML
from axelera.app.torch_utils import torch
from models import darknet

LOG = logging_utils.getLogger(__name__)


def _find_layers_of_type(module, layer_type):
    """
    Recursively find all layers of a specific type in a given module.

    Args:
    - module (nn.Module): The parent module to search in.
    - layer_type (type): The layer type to search for.

    Returns:
    - List[nn.Module]: List of found layers of the specified type.
    """
    layers = []

    for name, sub_module in module.named_children():
        if isinstance(sub_module, layer_type):
            layers.append(sub_module)
        layers += _find_layers_of_type(sub_module, layer_type)

    return layers


# Support models trained from
#  - https://github.com/WongKinYiu/yolor
#  - https://github.com/WongKinYiu/PyTorch_YOLOv4
#  - https://github.com/AlexeyAB/darknet
class AxYoloDarknet(darknet.Darknet, types.Model):
    MODEL_INPUT_HW = None

    def __init__(self, **kwargs):
        self.working_dir = str(Path.cwd())
        LOG.debug(f'Current working directory is {self.working_dir}')

        cfg = YAML.attribute(kwargs, 'darknet_cfg_path')
        shape = YAML.attribute(kwargs, 'input_tensor_shape')
        if YAML.attribute(kwargs, 'input_tensor_layout') == 'NCHW':
            imgsz = shape[2:]
        else:  # NHWC / CHWN
            imgsz = shape[1:3]
        self.MODEL_INPUT_HW = imgsz

        # setup Darknet here
        super().__init__(cfg, imgsz)

    def init_model_deploy(self, model_info: types.ModelInfo):
        weights = Path(model_info.weight_path)
        if not (weights.exists() and utils.md5_validates(weights, model_info.weight_md5)):
            utils.download(model_info.weight_url, weights, model_info.weight_md5)

        self.device = "cpu"
        self.number_of_classes = self.module_list[-1].nc
        LOG.debug(f'Load weights {weights}')
        try:  # model with .pth/.pt format
            self.load_state_dict(torch.load(weights)['model'])
        except:  # model with .weights format
            darknet.load_darknet_weights(self, weights)

        yolo_layers = _find_layers_of_type(self.module_list, darknet.YOLOLayer)
        # write back stride and anchors to model_info to build the decoder in GST pipeline
        model_info.extra_kwargs["YOLO"] = {}
        model_info.extra_kwargs["YOLO"]["stride"] = []
        model_info.extra_kwargs["YOLO"]["anchors"] = []
        for yolo_layer in yolo_layers:
            model_info.extra_kwargs["YOLO"]["stride"].append(yolo_layer.stride)
            # clean up the anchors
            anchor_wh = yolo_layer.anchor_wh.squeeze().tolist()
            # flatten the list
            anchors = [item for sublist in anchor_wh for item in sublist]
            model_info.extra_kwargs["YOLO"]["anchors"].append(anchors)

        # self.model.fuse() # TODO leave it for TVM/QTools processing
        self.to(self.device)  # make sure weights are on CPU

    def to_device(self, device: typing.Optional[torch.device] = None) -> None:
        device = device or torch.device()
        self.module_list[-1].anchor_wh = self.module_list[-1].anchor_wh.to(device)
        self.module_list[-1].grid = self.module_list[-1].grid.to(device)
        self.to(device)
        self.device = device

# Axelera class for PyTorch Darknet
# Copyright Axelera AI, 2023
from __future__ import annotations

from pathlib import Path
import types as _types
import typing

from axelera import types
from axelera.app import logging_utils, utils
from axelera.app.torch_utils import safe_torch_load, torch
from models import darknet

LOG = logging_utils.getLogger(__name__)


def _find_layers_of_type(module, layer_type):
    """Recursively find all layers of a specific type in a given module."""
    layers = []
    for _name, sub_module in module.named_children():
        if isinstance(sub_module, layer_type):
            layers.append(sub_module)
        layers += _find_layers_of_type(sub_module, layer_type)
    return layers


# `torch.fx.wrap` triggers fx import at decoration time, which fails in
# runtime-only test environments that provide only a lazy `torch` shim. Fall
# back to a no-op decorator so the module still imports there.
try:
    _fx_wrap = torch.fx.wrap
except (ImportError, AttributeError):

    def _fx_wrap(fn):
        return fn


# Upstream `YOLOLayer.forward` in models/darknet.py hardcodes the Ultralytics
# v5 decode formula. Swap it per-instance for the formula declared by the
# .cfg the model was trained on (read from the parsed `[yolo]` block).
@_fx_wrap
def _ax_yolo_activate(x, grid, anchor_wh, stride, scale_x_y, new_coords):
    # bxy = scale_x_y * sigmoid(t) - 0.5 * (scale_x_y - 1) + grid
    # bwh = (2 * sigmoid(t)) ** 2 * anchor    when new_coords  (Ultralytics v5)
    #     = exp(t_raw) * anchor               otherwise        (classic Darknet)
    offset = 0.5 * (scale_x_y - 1.0)
    if new_coords:
        io = x.sigmoid()
        io[..., 2:4] = (io[..., 2:4] * 2) ** 2 * anchor_wh
    else:
        io = x.clone()
        io[..., :2] = x[..., :2].sigmoid()
        io[..., 4:] = x[..., 4:].sigmoid()
        io[..., 2:4] = torch.exp(x[..., 2:4]) * anchor_wh
    io[..., :2] = io[..., :2] * scale_x_y - offset + grid
    io[..., :4] *= stride
    return io


def _ax_yolo_forward(self, x):
    bs, _, ny, nx = x.shape
    x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
    io = _ax_yolo_activate(
        x, self.grid, self.anchor_wh, self.stride, self._ax_scale_x_y, self._ax_new_coords
    )
    return io.view(bs, -1, self.no)


def _patch_yolo_layers(model):
    """Read scale_x_y / new_coords from each [yolo] cfg block and bind them onto
    the matching YOLOLayer instance, then swap in `_ax_yolo_forward`. Defaults
    follow classic Darknet semantics (scale_x_y=1.0, new_coords=False).
    """
    yolo_cfg_blocks = [m for m in model.module_defs if m.get('type') == 'yolo']
    yolo_layers = _find_layers_of_type(model.module_list, darknet.YOLOLayer)
    if len(yolo_layers) != len(yolo_cfg_blocks):
        LOG.warning(
            f"YOLOLayer count ({len(yolo_layers)}) does not match cfg yolo-block "
            f"count ({len(yolo_cfg_blocks)}); skipping activation-param override"
        )
        return
    for layer, mdef in zip(yolo_layers, yolo_cfg_blocks):
        layer._ax_scale_x_y = float(mdef.get('scale_x_y', 1.0))
        layer._ax_new_coords = bool(mdef.get('new_coords', False))
        layer.forward = _types.MethodType(_ax_yolo_forward, layer)


# Support models trained from
#  - https://github.com/WongKinYiu/yolor
#  - https://github.com/WongKinYiu/PyTorch_YOLOv4
#  - https://github.com/AlexeyAB/darknet
class AxYoloDarknet(darknet.Darknet, types.Model):
    MODEL_INPUT_HW = None

    def __init__(self, **kwargs):
        self.working_dir = str(Path.cwd())
        LOG.debug(f'Current working directory is {self.working_dir}')
        if missing := [
            k
            for k in ['darknet_cfg_path', 'input_tensor_shape', 'input_tensor_layout']
            if k not in kwargs
        ]:
            raise ValueError(f'Missing required arguments: {missing}')
        cfg = kwargs['darknet_cfg_path']
        shape = kwargs['input_tensor_shape']
        if kwargs['input_tensor_layout'] == 'NCHW':
            imgsz = shape[2:]
        else:  # NHWC / CHWN
            imgsz = shape[1:3]
        self.MODEL_INPUT_HW = imgsz

        # setup Darknet here
        super().__init__(cfg, imgsz)
        _patch_yolo_layers(self)

    def init_model_deploy(self, model_info: types.ModelInfo, dataset_config: dict, **kwargs):
        weights = Path(model_info.weight_path)
        utils.download_model_artifacts(
            weights,
            model_info.weight_url,
            model_info.weight_md5,
            model_name=model_info.name,
        )

        self.device = "cpu"
        self.number_of_classes = self.module_list[-1].nc
        LOG.debug(f'Load weights {weights}')
        try:  # model with .pth/.pt format
            self.load_state_dict(safe_torch_load(weights)['model'])
        except:  # model with .weights format
            darknet.load_darknet_weights(self, weights)

        yolo_layers = _find_layers_of_type(self.module_list, darknet.YOLOLayer)
        # Stride, anchors, and per-layer scale_x_y / new_coords flow to the GST
        # decoder via extra_kwargs["YOLO"]. Source-of-truth is the .cfg.
        # sigmoid_in_postprocess=True tells the GST decoder to bake sigmoid
        # into its int8->float LUT: YOLOLayer's sigmoid stays in the postamble
        # graph (the chip kernel does not fuse it), and the gst pipe bypasses
        # the postamble (handle_all: False), so the decoder must sigmoid the
        # raw chip output itself.
        yolo_cfg_blocks = [m for m in self.module_defs if m.get('type') == 'yolo']
        model_info.extra_kwargs["YOLO"] = {}
        model_info.extra_kwargs["YOLO"]["stride"] = []
        model_info.extra_kwargs["YOLO"]["anchors"] = []
        model_info.extra_kwargs["YOLO"]["scale_x_y"] = []
        model_info.extra_kwargs["YOLO"]["new_coords"] = []
        model_info.extra_kwargs["YOLO"]["sigmoid_in_postprocess"] = True
        for yolo_layer, mdef in zip(yolo_layers, yolo_cfg_blocks):
            model_info.extra_kwargs["YOLO"]["stride"].append(yolo_layer.stride)
            # clean up the anchors
            anchor_wh = yolo_layer.anchor_wh.squeeze().tolist()
            # flatten the list
            anchors = [item for sublist in anchor_wh for item in sublist]
            model_info.extra_kwargs["YOLO"]["anchors"].append(anchors)
            model_info.extra_kwargs["YOLO"]["scale_x_y"].append(float(mdef.get('scale_x_y', 1.0)))
            model_info.extra_kwargs["YOLO"]["new_coords"].append(
                bool(mdef.get('new_coords', False))
            )

        # self.model.fuse() # TODO leave it for TVM/QTools processing
        self.to(self.device)  # make sure weights are on CPU

    def to_device(self, device: typing.Optional[torch.device] = None) -> None:
        device = device or torch.device()
        self.module_list[-1].anchor_wh = self.module_list[-1].anchor_wh.to(device)
        self.module_list[-1].grid = self.module_list[-1].grid.to(device)
        self.to(device)
        self.device = device

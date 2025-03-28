# Axelera class for PyTorch YOLOv5 and YOLOv7
# Copyright Axelera AI, 2023
from __future__ import annotations

from pathlib import Path
import typing

from axelera import types
from axelera.app import logging_utils, utils
from axelera.app.torch_utils import torch
from models import common, yolo

LOG = logging_utils.getLogger(__name__)


def _visit_to_device(
    thing: typing.Union[typing.List[torch.Tensor], torch.Tensor], device: torch.device
):
    if isinstance(thing, torch.Tensor):
        return thing.to(device)
    elif isinstance(thing, list):
        return [_visit_to_device(x, device) for x in thing]
    else:
        raise TypeError(f'Unsupported type {type(thing)}')


def _replace_focus(module: torch.nn.Module):
    for name, child in module.named_children():
        if isinstance(child, common.Focus):
            c1 = child.conv.conv.in_channels // 4  # original number of input channels
            c2 = child.conv.conv.out_channels  # number of output channels
            # doubling the kernel size, stride and padding
            k = (child.conv.conv.kernel_size[0] * 2, child.conv.conv.kernel_size[1] * 2)
            s = (child.conv.conv.stride[0] * 2, child.conv.conv.stride[1] * 2)
            p = (child.conv.conv.padding[0] * 2, child.conv.conv.padding[1] * 2)

            # Creating the equivalent Conv layer
            conv = common.Conv(c1, c2, k, s, p)

            # Transferring weights
            conv.conv.weight.data[:, :, ::2, ::2] = child.conv.conv.weight.data[:, :3]
            conv.conv.weight.data[:, :, 1::2, ::2] = child.conv.conv.weight.data[:, 3:6]
            conv.conv.weight.data[:, :, ::2, 1::2] = child.conv.conv.weight.data[:, 6:9]
            conv.conv.weight.data[:, :, 1::2, 1::2] = child.conv.conv.weight.data[:, 9:12]
            # Transferring batch normalization, activator and the rest of the parameters
            conv.bn = child.conv.bn
            conv.act = child.conv.act
            for attr in ['f', 'i', 'training']:  # YOLOv5 specific attributes
                if hasattr(child, attr):
                    setattr(conv, attr, getattr(child, attr))
            # Replacing the original layer with the new one
            setattr(module, name, conv)
        else:
            _replace_focus(child)


def _replace_conv_with_focus(model, replaced=False):
    class SimpleFocus(torch.nn.Module):
        def __init__(self, c1, c2, k=1, s=1, p=None, g=1):
            super().__init__()
            # Using nn.Conv2d directly
            self.conv = torch.nn.Conv2d(
                c1 * 4, c2, k, s, common.autopad(k, p), groups=g, bias=False
            )

        def forward(self, x):
            x = torch.cat(
                [x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1
            )
            return self.conv(x)

    # Check if the model already has a Focus layer at the beginning
    for _, module in model.named_children():
        if isinstance(module, common.Focus):
            return False  # Focus layer found, no need to replace

    for name, module in model.named_children():
        if replaced:
            break
        if isinstance(module, torch.nn.Conv2d):
            c1 = module.in_channels
            c2 = module.out_channels
            # Halve the kernel size, stride, and padding for the Focus operation
            k = module.kernel_size[0] // 2
            s = module.stride[0] // 2
            p = module.padding[0] // 2
            g = module.groups

            focus_like = SimpleFocus(c1, c2, k, s, p, g)

            # Reorganizing and assigning weights from the Conv layer to the SimpleFocus's convolution
            new_weight = torch.zeros((c2, 4 * c1, k, k))
            # Assigning weights from the Conv layer to the Focus layer's convolution
            original_weight = module.weight.data
            new_weight[:, :3] = original_weight[:, :, ::2, ::2]
            new_weight[:, 3:6] = original_weight[:, :, 1::2, ::2]
            new_weight[:, 6:9] = original_weight[:, :, ::2, 1::2]
            new_weight[:, 9:12] = original_weight[:, :, 1::2, 1::2]

            focus_like.conv.weight.data = new_weight
            if module.bias is not None and focus_like.conv.bias is not None:
                focus_like.conv.bias.data = module.bias.data.clone()

            for attr in ['f', 'i', 'training']:
                if hasattr(module, attr):
                    setattr(focus_like, attr, getattr(module, attr))

            # Replace the original Conv2d layer with SimpleFocus
            setattr(model, name, focus_like)
            return True  # Only replace the first Conv2d
        else:
            replaced = _replace_conv_with_focus(module, replaced)
    return replaced


# Support models trained from
#  - https://github.com/ultralytics/yolov5
#  - https://github.com/WongKinYiu/yolov7


class AxYolo(yolo.Model, types.Model):
    MODEL_INPUT_HW = None

    def __init__(self, input_width: int, input_height: int, input_channel: int, yolo_cfg_path=''):
        self.working_dir = str(Path.cwd())
        LOG.debug(f'Current working directory is {self.working_dir}')
        self.MODEL_INPUT_HW = (input_height, input_width)
        # Maybe setup fake YOLO model; hacky but helps usability
        cfg = yolo_cfg_path or 'cfg/fake_cfg.yaml'
        super().__init__(cfg, ch=input_channel)

    def init_model_deploy(self, model_info: types.ModelInfo):
        weights = Path(model_info.weight_path)
        if not (weights.exists() and utils.md5_validates(weights, model_info.weight_md5)):
            utils.download(model_info.weight_url, weights, model_info.weight_md5)

        self.device = "cpu"
        LOG.debug(f'Load weights {weights}')
        ckpt = torch.load(weights, map_location=self.device)

        # we currently don't support ensembled model
        model = ckpt['ema' if ckpt.get('ema') else 'model']

        yolo_config = model_info.extra_kwargs.get('YOLO', {})
        if yolo_config.get("focus_layer_replacement", None) and yolo_config.get(
            "first_conv_layer_replacement", None
        ):
            raise ValueError(
                f"Both focus_layer_replacement and first_conv_layer_replacement are enabled. "
                f"Please enable only one of them. If your model is trained with YOLOv5 before v5.0, "
                f"please enable first_conv_layer_replacement. Otherwise, enable focus_layer_replacement."
            )

        # TODO: enable focus layer replacement if we see performance improvement
        # if not declared explicitly, always enable focus layer replacement
        if yolo_config.get("focus_layer_replacement", True):
            LOG.debug("Replace Focus layers with equivalent Conv layers")
            _replace_focus(model)

        if yolo_config.get("first_conv_layer_replacement", False):
            if _replace_conv_with_focus(model):
                LOG.debug("Replace first Conv layer with equivalent Focus layer")

        # force update all parameters but not weights only
        self.__dict__.update(model.__dict__)

        # self.model.model[-1].include_nms
        self.number_of_classes = self.nc
        self.model.float()

        LOG.debug(
            f'Preprocess anchor grids with the input size {self.MODEL_INPUT_HW[1]}x{self.MODEL_INPUT_HW[0]}'
        )
        self.model[-1].grid[:], self.model[-1].anchor_grid[:] = self.__make_grid(
            self.MODEL_INPUT_HW[1],
            self.MODEL_INPUT_HW[0],
            self.model[-1].na,
            self.model[-1].anchors,
            self.model[-1].stride,
            device=self.device,
        )

        # Compatibility updates
        for m in self.modules():
            if type(m) is torch.nn.Upsample:
                m.recompute_scale_factor = None  # torch 1.11.0

        # write back stride and anchors to model_info to build the decoder in GST pipeline
        YOLO = model_info.extra_kwargs.setdefault('YOLO', {})
        YOLO["stride"] = self.model[-1].stride.int().tolist()
        anchors = self.model[-1].anchors.tolist()
        anchors = [
            [value for sublist in outer_list for value in sublist] for outer_list in anchors
        ]
        model_info.extra_kwargs["YOLO"]["anchors"] = anchors

        # self.model.fuse() # should we fuse or leave it for TVM/QTools processing
        self.to(self.device)  # make sure weights are on CPU

    def __make_grid(self, input_w, input_h, na, anchors, stride, device):
        # TODO: experiment for rectangle input
        grids, anchor_grids = [], []
        for i in range(na):
            nx, ny = int(input_w / stride[i]), int(input_h / stride[i])
            shape = 1, na, ny, nx, 2  # grid shape
            yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
            grid = torch.stack((xv, yv), 2).expand(shape).float() - 0.5  # grid pre-offset
            anchor_grid = (anchors[i] * stride[i]).view((1, na, 1, 1, 2))
            grids.append(grid.type(anchors.dtype).to(device))
            anchor_grids.append(anchor_grid.type(anchors.dtype).to(device))
        return grids, torch.stack(anchor_grids)

    def to_device(self, device: typing.Optional[torch.device] = None):
        device = device or torch.device()
        detect = self.model[-1]
        detect.anchor_grid = _visit_to_device(detect.anchor_grid, device)
        detect.grid = _visit_to_device(detect.grid, device)
        self.to(device)
        self.device = device

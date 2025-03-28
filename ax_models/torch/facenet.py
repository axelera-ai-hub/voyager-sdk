from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import torch

from axelera import types
from axelera.app import logging_utils, utils
import axelera.app.yaml as YAML

LOG = logging_utils.getLogger(__name__)


class BasicConv2d(torch.nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding=0):
        super().__init__()
        self.conv = torch.nn.Conv2d(
            in_planes,
            out_planes,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )  # verify bias false
        self.bn = torch.nn.BatchNorm2d(
            out_planes,
            eps=0.001,  # value found in tensorflow
            momentum=0.1,  # default pytorch value
            affine=True,
        )
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class Block35(torch.nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(256, 32, kernel_size=1, stride=1)

        self.branch1 = torch.nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )

        self.branch2 = torch.nn.Sequential(
            BasicConv2d(256, 32, kernel_size=1, stride=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
            BasicConv2d(32, 32, kernel_size=3, stride=1, padding=1),
        )

        self.conv2d = torch.nn.Conv2d(96, 256, kernel_size=1, stride=1)
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block17(torch.nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()

        self.scale = scale

        self.branch0 = BasicConv2d(896, 128, kernel_size=1, stride=1)

        self.branch1 = torch.nn.Sequential(
            BasicConv2d(896, 128, kernel_size=1, stride=1),
            BasicConv2d(128, 128, kernel_size=(1, 7), stride=1, padding=(0, 3)),
            BasicConv2d(128, 128, kernel_size=(7, 1), stride=1, padding=(3, 0)),
        )

        self.conv2d = torch.nn.Conv2d(256, 896, kernel_size=1, stride=1)
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        out = self.relu(out)
        return out


class Block8(torch.nn.Module):
    def __init__(self, scale=1.0, noReLU=False):
        super().__init__()

        self.scale = scale
        self.noReLU = noReLU

        self.branch0 = BasicConv2d(1792, 192, kernel_size=1, stride=1)

        self.branch1 = torch.nn.Sequential(
            BasicConv2d(1792, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=(1, 3), stride=1, padding=(0, 1)),
            BasicConv2d(192, 192, kernel_size=(3, 1), stride=1, padding=(1, 0)),
        )

        self.conv2d = torch.nn.Conv2d(384, 1792, kernel_size=1, stride=1)
        if not self.noReLU:
            self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        out = torch.cat((x0, x1), 1)
        out = self.conv2d(out)
        out = out * self.scale + x
        if not self.noReLU:
            out = self.relu(out)
        return out


class Mixed_6a(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.branch0 = BasicConv2d(256, 384, kernel_size=3, stride=2)

        self.branch1 = torch.nn.Sequential(
            BasicConv2d(256, 192, kernel_size=1, stride=1),
            BasicConv2d(192, 192, kernel_size=3, stride=1, padding=1),
            BasicConv2d(192, 256, kernel_size=3, stride=2),
        )

        self.branch2 = torch.nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        out = torch.cat((x0, x1, x2), 1)
        return out


class Mixed_7a(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.branch0 = torch.nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 384, kernel_size=3, stride=2),
        )

        self.branch1 = torch.nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2),
        )

        self.branch2 = torch.nn.Sequential(
            BasicConv2d(896, 256, kernel_size=1, stride=1),
            BasicConv2d(256, 256, kernel_size=3, stride=1, padding=1),
            BasicConv2d(256, 256, kernel_size=3, stride=2),
        )

        self.branch3 = torch.nn.MaxPool2d(3, stride=2)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)
        x3 = self.branch3(x)
        out = torch.cat((x0, x1, x2, x3), 1)
        return out


class InceptionResnetV1(torch.nn.Module):
    def __init__(self, pretrained='vggface2', classify=False, num_classes=None, dropout_prob=0.6):
        super().__init__()

        self.classify = classify
        self._dropout_prob = dropout_prob
        if pretrained == "vggface2":
            tmp_classes = 8631
        elif pretrained == "casia-webface":
            tmp_classes = 10575
        elif pretrained is None and self.classify and self.num_classes is None:
            raise Exception(
                'If "pretrained" is not specified and "classify" is True, "num_classes" '
                + "must be specified"
            )

        # Define layers
        self.conv2d_1a = BasicConv2d(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = BasicConv2d(32, 32, kernel_size=3, stride=1)
        self.conv2d_2b = BasicConv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.maxpool_3a = torch.nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = BasicConv2d(64, 80, kernel_size=1, stride=1)
        self.conv2d_4a = BasicConv2d(80, 192, kernel_size=3, stride=1)
        self.conv2d_4b = BasicConv2d(192, 256, kernel_size=3, stride=2)
        self.repeat_1 = torch.nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
        )
        self.mixed_6a = Mixed_6a()
        self.repeat_2 = torch.nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
        )
        self.mixed_7a = Mixed_7a()
        self.repeat_3 = torch.nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
        )
        self.block8 = Block8(noReLU=True)
        self.avgpool_1a = torch.nn.AdaptiveAvgPool2d(1)
        self.dropout = torch.nn.Dropout(self._dropout_prob)
        self.last_linear = torch.nn.Linear(1792, 512, bias=False)
        self.last_bn = torch.nn.BatchNorm1d(512, eps=0.001, momentum=0.1, affine=True)
        self.flatten = torch.nn.Flatten()

        if pretrained is not None:
            self.logits = torch.nn.Linear(512, tmp_classes)
        else:  # self.num_classes is not None:
            self.logits = torch.nn.Linear(512, self.num_classes)

    def forward(self, x):
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        x = self.avgpool_1a(x)
        x = self.dropout(x)
        x = self.flatten(x)
        x = self.last_linear(x)
        x = self.last_bn(x)
        if self.classify:
            x = self.logits(x)
        else:
            # x = F.normalize(x, p=2, dim=1)
            return x


class AxFaceNet(InceptionResnetV1, types.Model):
    """This class creates an axelera.types.Model instance for the FaceNet model"""

    def __init__(self):
        super().__init__()

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
        LOG.debug(f'Load model with weights {weights}')

        device = next(self.parameters()).device
        weights_tensor = torch.load(weights, map_location=device)
        self.load_state_dict(weights_tensor)

    def to_device(self, device: Optional[torch.device] = None):
        self.device = device or torch.device()
        self.to(self.device)

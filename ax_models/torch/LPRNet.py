# Copyright Axelera AI, 2025

from pathlib import Path

import torch
import torch.nn as nn

from ax_models.base_torch import TorchModel
from axelera import types
from axelera.app import logging_utils, utils

LOG = logging_utils.getLogger(__name__)


class small_basic_block(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(small_basic_block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(ch_in, ch_out // 4, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(3, 1), padding=(1, 0)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out // 4, kernel_size=(1, 3), padding=(0, 1)),
            nn.ReLU(),
            nn.Conv2d(ch_out // 4, ch_out, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


class LPRNet(nn.Module):
    def __init__(self, lpr_max_len, class_num, dropout_rate):
        super(LPRNet, self).__init__()
        self.lpr_max_len = lpr_max_len
        self.class_num = class_num
        self.backbone = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1),  # 0
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),  # 2
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 1, 1)),
            small_basic_block(ch_in=64, ch_out=128),  # *** 4 ***
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),  # 6
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(2, 1, 2)),
            small_basic_block(ch_in=64, ch_out=256),  # 8
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 10
            small_basic_block(ch_in=256, ch_out=256),  # *** 11 ***
            nn.BatchNorm2d(num_features=256),  # 12
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(4, 1, 2)),  # 14
            nn.Dropout(dropout_rate),
            nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(1, 4), stride=1),  # 16
            nn.BatchNorm2d(num_features=256),
            nn.ReLU(),  # 18
            nn.Dropout(dropout_rate),
            nn.Conv2d(
                in_channels=256, out_channels=class_num, kernel_size=(13, 1), stride=1
            ),  # 20
            nn.BatchNorm2d(num_features=class_num),
            nn.ReLU(),  # *** 22 ***
        )
        self.container = nn.Sequential(
            nn.Conv2d(
                in_channels=448 + self.class_num,
                out_channels=self.class_num,
                kernel_size=(1, 1),
                stride=(1, 1),
            ),
            # nn.BatchNorm2d(num_features=self.class_num),
            # nn.ReLU(),
            # nn.Conv2d(in_channels=self.class_num, out_channels=self.lpr_max_len+1, kernel_size=3, stride=2),
            # nn.ReLU(),
        )

    def forward(self, x):
        keep_features = list()
        for i, layer in enumerate(self.backbone.children()):
            x = layer(x)
            if i in [2, 6, 13, 22]:  # [2, 4, 8, 11, 22]
                keep_features.append(x)

        global_context = list()
        for i, f in enumerate(keep_features):
            if i in [0, 1]:
                f = nn.AvgPool2d(kernel_size=5, stride=5)(f)
            if i in [2]:
                f = nn.AvgPool2d(kernel_size=(4, 10), stride=(4, 2))(f)
            f_pow = torch.pow(f, 2)
            f_mean = torch.mean(f_pow)
            f = torch.div(f, f_mean)
            global_context.append(f)

        x = torch.cat(global_context, 1)
        x = self.container(x)
        logits = torch.mean(x, dim=2)

        return logits


def build_lprnet(lpr_max_len, class_num, dropout_rate=0.5):
    net = LPRNet(lpr_max_len, class_num, dropout_rate)
    return net.eval()


class LPRNetTorchModel(TorchModel):
    def init_model_deploy(self, model_info: types.ModelInfo, dataset_config: dict, **kwargs):
        weights = Path(model_info.weight_path)
        utils.download_model_artifacts(
            weights,
            model_info.weight_url,
            model_info.weight_md5,
            model_name=model_info.name,
        )
        LOG.debug(f'Load model with weights {weights}')

        if 'LPRNet' not in kwargs:
            raise ValueError("Missing required parameter: 'LPRNet'")

        lpr_params = kwargs['LPRNet']
        if 'lpr_max_len' not in lpr_params:
            raise ValueError("Missing required parameter: 'lpr_max_len' in LPRNet config")

        lpr_max_len = int(lpr_params['lpr_max_len'])

        class_num = model_info.num_classes
        if class_num != len(model_info.labels):
            raise ValueError(
                f'Number of classes {class_num} does not match number of labels {len(model_info.labels)}'
            )

        self.torch_model = build_lprnet(lpr_max_len, class_num)
        self.torch_model.load_state_dict(
            torch.load(model_info.weight_path, map_location=torch.device('cpu'))
        )

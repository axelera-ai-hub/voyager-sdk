# Copyright Axelera AI, 2023
# Utility functions to building YOLO models
#
# Ultralytics YOLO Models and Software License
# The Ultralytics YOLO models, along with their supporting files and documentation
# (collectively, the "Software"), are developed and distributed by Ultralytics
# under the GNU Affero General Public License version 3.0 (AGPL-3.0), subject to
# the following modification for Axelera AI:
#
# Evaluation Use Only:
# This Software is solely for testing and evaluation on Axelera AI hardware. Any
# other use requires a commercial license from Ultralytics.
#
# Commercial Use Prohibited Without License:
# Commercial use of this Software, including deployment, reproduction, or
# distribution in products or services internal or external to a commercial
# entity requires an Ultralytics Enterprise License. Visit
# https://www.ultralytics.com/license for licensing details. By using this
# Software, you acknowledge and agree to comply with the above terms.
from copy import deepcopy
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from axelera.app import logging_utils

LOG = logging_utils.getLogger(__name__)


def fuse_conv_and_bn(conv, bn):
    # Fuse convolution and batchnorm layers https://tehnokv.com/posts/fusing-batchnorm-and-conv/
    fusedconv = (
        nn.Conv2d(
            conv.in_channels,
            conv.out_channels,
            kernel_size=conv.kernel_size,
            stride=conv.stride,
            padding=conv.padding,
            groups=conv.groups,
            bias=True,
        )
        .requires_grad_(False)
        .to(conv.weight.device)
    )

    # prepare filters
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fusedconv.weight.copy_(torch.mm(w_bn, w_conv).view(fusedconv.weight.shape))

    # prepare spatial bias
    b_conv = (
        torch.zeros(conv.weight.size(0), device=conv.weight.device)
        if conv.bias is None
        else conv.bias
    )
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fusedconv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fusedconv


def model_info(model, verbose=False, img_size=640):
    # Model information. img_size may be int or list, i.e. img_size=640 or img_size=[640, 320]
    n_p = sum(x.numel() for x in model.parameters())  # number parameters
    n_g = sum(x.numel() for x in model.parameters() if x.requires_grad)  # number gradients
    if verbose:
        LOG.debug(
            '%5s %40s %9s %12s %20s %10s %10s'
            % ('layer', 'name', 'gradient', 'parameters', 'shape', 'mu', 'sigma')
        )
        for i, (name, p) in enumerate(model.named_parameters()):
            name = name.replace('module_list.', '')
            LOG.debug(
                '%5g %40s %9s %12g %20s %10.3g %10.3g'
                % (i, name, p.requires_grad, p.numel(), list(p.shape), p.mean(), p.std())
            )

    try:  # FLOPS
        from thop import profile

        stride = max(int(model.stride.max()), 32) if hasattr(model, 'stride') else 32
        img = torch.zeros(
            (1, model.yaml.get('ch', 3), stride, stride), device=next(model.parameters()).device
        )  # input
        flops = (
            profile(deepcopy(model), inputs=(img,), verbose=False)[0] / 1e9 * 2
        )  # stride GFLOPS
        img_size = (
            img_size if isinstance(img_size, list) else [img_size, img_size]
        )  # expand if int/float
        fs = ', %.1f GFLOPS' % (
            flops * img_size[0] / stride * img_size[1] / stride
        )  # 640x640 GFLOPS
    except (ImportError, Exception):
        fs = ''

    LOG.debug(
        f"Model summary: {len(list(model.modules()))} layers, {n_p} parameters, {n_g} gradients{fs}"
    )


def check_anchor_order(m):
    # Check anchor order against stride order for YOLO Detect() module m, and correct if necessary
    a = m.anchor_grid.prod(-1).view(-1)  # anchor area
    da = a[-1] - a[0]  # delta a
    ds = m.stride[-1] - m.stride[0]  # delta s
    if da.sign() != ds.sign():  # same order
        LOG.debug('Reversing anchor order')
        m.anchors[:] = m.anchors.flip(0)
        m.anchor_grid[:] = m.anchor_grid.flip(0)


def make_divisible(x, divisor):
    # Returns x evenly divisible by divisor
    return math.ceil(x / divisor) * divisor


def initialize_weights(model):
    for m in model.modules():
        t = type(m)
        if t is nn.Conv2d:
            pass  # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        elif t is nn.BatchNorm2d:
            m.eps = 1e-3
            m.momentum = 0.03
        elif t in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6]:
            m.inplace = True


def scale_img(img, ratio=1.0, same_shape=False, gs=32):  # img(16,3,256,416)
    # scales img(bs,3,y,x) by ratio constrained to gs-multiple
    if ratio == 1.0:
        return img
    else:
        h, w = img.shape[2:]
        s = (int(h * ratio), int(w * ratio))  # new size
        img = F.interpolate(img, size=s, mode='bilinear', align_corners=False)  # resize
        if not same_shape:  # pad/crop img
            h, w = [math.ceil(x * ratio / gs) * gs for x in (h, w)]
        return F.pad(img, [0, w - s[1], 0, h - s[0]], value=0.447)  # value = imagenet mean


def copy_attr(a, b, include=(), exclude=()):
    # Copy attributes from b to a, options to only include [...] and to exclude [...]
    for k, v in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, v)


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

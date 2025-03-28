# Copyright Axelera AI, 2023
# Darknet model

import os
from pathlib import Path

import numpy as np

from axelera.app import logging_utils
from axelera.app.torch_utils import torch

from .activations import *
from .common import *
from .yolo_utils import fuse_conv_and_bn, model_info

LOG = logging_utils.getLogger(__name__)


def parse_model_cfg(path):
    # Parse the yolo *.cfg file and return module definitions path may be 'cfg/yolov3.cfg', 'yolov3.cfg', or 'yolov3'
    if not path.endswith('.cfg'):  # add .cfg suffix if omitted
        path += '.cfg'
    if not os.path.exists(path) and os.path.exists(
        'cfg' + os.sep + path
    ):  # add cfg/ prefix if omitted
        path = 'cfg' + os.sep + path

    with open(path, 'r') as f:
        lines = f.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]  # get rid of fringe whitespaces
    mdefs = []  # module definitions
    for line in lines:
        if line.startswith('['):  # This marks the start of a new block
            mdefs.append({})
            mdefs[-1]['type'] = line[1:-1].rstrip()
            if mdefs[-1]['type'] == 'convolutional':
                mdefs[-1][
                    'batch_normalize'
                ] = 0  # pre-populate with zeros (may be overwritten later)

        else:
            key, val = line.split("=")
            key = key.rstrip()

            if key == 'anchors':  # return nparray
                mdefs[-1][key] = np.array([float(x) for x in val.split(',')]).reshape(
                    (-1, 2)
                )  # np anchors
            elif (key in ['from', 'layers', 'mask']) or (
                key == 'size' and ',' in val
            ):  # return array
                mdefs[-1][key] = [int(x) for x in val.split(',')]
            else:
                val = val.strip()
                if val.isnumeric():  # return int or float
                    mdefs[-1][key] = int(val) if (int(val) - float(val)) == 0 else float(val)
                else:
                    mdefs[-1][key] = val  # return string

    # Check all fields are supported
    supported = [
        'type',
        'batch_normalize',
        'filters',
        'size',
        'stride',
        'pad',
        'activation',
        'layers',
        'groups',
        'from',
        'mask',
        'anchors',
        'classes',
        'num',
        'jitter',
        'ignore_thresh',
        'truth_thresh',
        'random',
        'stride_x',
        'stride_y',
        'weights_type',
        'weights_normalization',
        'scale_x_y',
        'beta_nms',
        'nms_kind',
        'iou_loss',
        'iou_normalizer',
        'cls_normalizer',
        'iou_thresh',
        'atoms',
        'na',
        'nc',
    ]

    f = []  # fields
    for x in mdefs[1:]:
        [f.append(k) for k in x if k not in f]
    u = [x for x in f if x not in supported]  # unsupported fields
    assert not any(
        u
    ), "Unsupported fields %s in %s. See https://github.com/ultralytics/yolov3/issues/631" % (
        u,
        path,
    )

    return mdefs


def create_modules(module_defs, img_size, cfg, device):
    # Constructs module list of layer blocks from module configuration in module_defs

    img_size = [img_size] * 2 if isinstance(img_size, int) else img_size  # expand if necessary
    _ = module_defs.pop(0)  # cfg training hyperparams (unused)
    output_filters = [3]  # input channels
    module_list = nn.ModuleList()
    routs = []  # list of layers which rout to deeper layers
    yolo_index = -1

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            k = mdef['size']  # kernel size
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if isinstance(k, int):  # single-size conv
                modules.add_module(
                    'Conv2d',
                    nn.Conv2d(
                        in_channels=output_filters[-1],
                        out_channels=filters,
                        kernel_size=k,
                        stride=stride,
                        padding=k // 2 if mdef['pad'] else 0,
                        groups=mdef['groups'] if 'groups' in mdef else 1,
                        bias=not bn,
                    ),
                )
            else:  # multiple-size conv
                modules.add_module(
                    'MixConv2d',
                    MixConv2d(
                        in_ch=output_filters[-1], out_ch=filters, k=k, stride=stride, bias=not bn
                    ),
                )

            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=1e-4))
            else:
                routs.append(i)  # detection output (goes into yolo layer)

            if (
                mdef['activation'] == 'leaky'
            ):  # activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mdef['activation'] == 'mish':
                modules.add_module('activation', Mish())
            elif mdef['activation'] == 'emb':
                modules.add_module('activation', F.normalize())
            elif mdef['activation'] == 'logistic':
                modules.add_module('activation', nn.Sigmoid())
            elif mdef['activation'] == 'silu':
                modules.add_module('activation', nn.SiLU())

        elif mdef['type'] == 'deformableconvolutional':
            bn = mdef['batch_normalize']
            filters = mdef['filters']
            k = mdef['size']  # kernel size
            stride = mdef['stride'] if 'stride' in mdef else (mdef['stride_y'], mdef['stride_x'])
            if isinstance(k, int):  # single-size conv
                modules.add_module(
                    'DeformConv2d',
                    DeformConv2d(
                        output_filters[-1],
                        filters,
                        kernel_size=k,
                        padding=k // 2 if mdef['pad'] else 0,
                        stride=stride,
                        bias=not bn,
                        modulation=True,
                    ),
                )
            else:  # multiple-size conv
                modules.add_module(
                    'MixConv2d',
                    MixConv2d(
                        in_ch=output_filters[-1], out_ch=filters, k=k, stride=stride, bias=not bn
                    ),
                )

            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.03, eps=1e-4))
            else:
                routs.append(i)  # detection output (goes into yolo layer)

            if (
                mdef['activation'] == 'leaky'
            ):  # activation study https://github.com/ultralytics/yolov3/issues/441
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif mdef['activation'] == 'swish':
                modules.add_module('activation', Swish())
            elif mdef['activation'] == 'mish':
                modules.add_module('activation', Mish())
            elif mdef['activation'] == 'silu':
                modules.add_module('activation', nn.SiLU())

        elif mdef['type'] == 'dropout':
            p = mdef['probability']
            modules = nn.Dropout(p)

        elif mdef['type'] == 'avgpool':
            modules = GAP()

        elif mdef['type'] == 'silence':
            filters = output_filters[-1]
            modules = Silence()

        elif mdef['type'] == 'scale_channels':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = ScaleChannel(layers=layers)

        elif mdef['type'] == 'shift_channels':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = ShiftChannel(layers=layers)

        elif (
            mdef['type'] == 'shift_channels_2d'
        ):  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = ShiftChannel2D(layers=layers)

        elif (
            mdef['type'] == 'control_channels'
        ):  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = ControlChannel(layers=layers)

        elif (
            mdef['type'] == 'control_channels_2d'
        ):  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = ControlChannel2D(layers=layers)

        elif (
            mdef['type'] == 'alternate_channels'
        ):  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1] * 2
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = AlternateChannel(layers=layers)

        elif (
            mdef['type'] == 'alternate_channels_2d'
        ):  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1] * 2
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = AlternateChannel2D(layers=layers)

        elif mdef['type'] == 'select_channels':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = SelectChannel(layers=layers)

        elif (
            mdef['type'] == 'select_channels_2d'
        ):  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = SelectChannel2D(layers=layers)

        elif mdef['type'] == 'sam':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = ScaleSpatial(layers=layers)

        elif mdef['type'] == 'BatchNorm2d':
            filters = output_filters[-1]
            modules = nn.BatchNorm2d(filters, momentum=0.03, eps=1e-4)
            if i == 0 and filters == 3:  # normalize RGB image
                # imagenet mean and var https://pytorch.org/docs/stable/torchvision/models.html#classification
                modules.running_mean = torch.tensor([0.485, 0.456, 0.406])
                modules.running_var = torch.tensor([0.0524, 0.0502, 0.0506])

        elif mdef['type'] == 'maxpool':
            k = mdef['size']  # kernel size
            stride = mdef['stride']
            maxpool = nn.MaxPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
            if k == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('MaxPool2d', maxpool)
            else:
                modules = maxpool

        elif mdef['type'] == 'local_avgpool':
            k = mdef['size']  # kernel size
            stride = mdef['stride']
            avgpool = nn.AvgPool2d(kernel_size=k, stride=stride, padding=(k - 1) // 2)
            if k == 2 and stride == 1:  # yolov3-tiny
                modules.add_module('ZeroPad2d', nn.ZeroPad2d((0, 1, 0, 1)))
                modules.add_module('AvgPool2d', avgpool)
            else:
                modules = avgpool

        elif mdef['type'] == 'upsample':
            modules = nn.Upsample(scale_factor=mdef['stride'])

        elif mdef['type'] == 'route':  # nn.Sequential() placeholder for 'route' layer
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat(layers=layers)

        elif mdef['type'] == 'route2':  # nn.Sequential() placeholder for 'route' layer
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat2(layers=layers)

        elif mdef['type'] == 'route3':  # nn.Sequential() placeholder for 'route' layer
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers])
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat3(layers=layers)

        elif mdef['type'] == 'route_lhalf':  # nn.Sequential() placeholder for 'route' layer
            layers = mdef['layers']
            filters = sum([output_filters[l + 1 if l > 0 else l] for l in layers]) // 2
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = FeatureConcat_l(layers=layers)

        elif mdef['type'] == 'shortcut':  # nn.Sequential() placeholder for 'shortcut' layer
            layers = mdef['from']
            filters = output_filters[-1]
            routs.extend([i + l if l < 0 else l for l in layers])
            modules = WeightedFeatureFusion(layers=layers, weight='weights_type' in mdef)

        elif mdef['type'] == 'reorg3d':  # yolov3-spp-pan-scale
            pass

        elif mdef['type'] == 'reorg':  # yolov3-spp-pan-scale
            filters = 4 * output_filters[-1]
            modules.add_module('Reorg', Reorg())

        elif mdef['type'] == 'dwt':  # yolov3-spp-pan-scale
            filters = 4 * output_filters[-1]
            modules.add_module('DWT', DWT())

        elif mdef['type'] == 'implicit_add':  # yolov3-spp-pan-scale
            filters = mdef['filters']
            modules = ImplicitA(channel=filters)

        elif mdef['type'] == 'implicit_mul':  # yolov3-spp-pan-scale
            filters = mdef['filters']
            modules = ImplicitM(channel=filters)

        elif mdef['type'] == 'implicit_cat':  # yolov3-spp-pan-scale
            filters = mdef['filters']
            modules = ImplicitC(channel=filters)

        elif mdef['type'] == 'implicit_add_2d':  # yolov3-spp-pan-scale
            channels = mdef['filters']
            filters = mdef['atoms']
            modules = Implicit2DA(atom=filters, channel=channels)

        elif mdef['type'] == 'implicit_mul_2d':  # yolov3-spp-pan-scale
            channels = mdef['filters']
            filters = mdef['atoms']
            modules = Implicit2DM(atom=filters, channel=channels)

        elif mdef['type'] == 'implicit_cat_2d':  # yolov3-spp-pan-scale
            channels = mdef['filters']
            filters = mdef['atoms']
            modules = Implicit2DC(atom=filters, channel=channels)

        elif mdef['type'] == 'yolo':
            yolo_index += 1
            stride = [8, 16, 32, 64, 128]  # P3, P4, P5, P6, P7 strides
            if any(x in cfg for x in ['yolov4-tiny', 'fpn', 'yolov3']):  # P5, P4, P3 strides
                stride = [32, 16, 8]
            layers = mdef['from'] if 'from' in mdef else []
            modules = YOLOLayer(
                anchors=mdef['anchors'][mdef['mask']],  # anchor list
                nc=mdef['classes'],  # number of classes
                img_size=img_size,  # (416, 416)
                yolo_index=yolo_index,  # 0, 1, 2...
                layers=layers,  # output layers
                stride=stride[yolo_index],
                device=device,
            )

            # Initialize preceding Conv2d() bias (https://arxiv.org/pdf/1708.02002.pdf section 3.3)
            try:
                j = layers[yolo_index] if 'from' in mdef else -2
                bias_ = module_list[j][0].bias  # shape(255,)
                bias = bias_[: modules.no * modules.na].view(modules.na, -1)  # shape(3,85)
                # bias[:, 4] += -4.5  # obj
                bias.data[:, 4] += math.log(
                    8 / (640 / stride[yolo_index]) ** 2
                )  # obj (8 objects per 640 image)
                bias.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))  # cls (sigmoid(p) = 1/nc)
                module_list[j][0].bias = torch.nn.Parameter(
                    bias_, requires_grad=bias_.requires_grad
                )

                # j = [-2, -5, -8]
                # for sj in j:
                #    bias_ = module_list[sj][0].bias
                #    bias = bias_[:modules.no * 1].view(1, -1)
                #    bias.data[:, 4] += math.log(8 / (640 / stride[yolo_index]) ** 2)
                #    bias.data[:, 5:] += math.log(0.6 / (modules.nc - 0.99))
                #    module_list[sj][0].bias = torch.nn.Parameter(bias_, requires_grad=bias_.requires_grad)
            except:
                LOG.debug('WARNING: smart bias initialization failure')

        else:
            raise RuntimeError("Unrecognized layer type: {mdef['type']}")

        # Register module list and number of output filters
        module_list.append(modules)
        output_filters.append(filters)

    routs_binary = [False] * (i + 1)
    for i in routs:
        routs_binary[i] = True
    return module_list, routs_binary


# workaround for symbolic tracing with FX
@torch.fx.wrap
def activate_xywh(io, grid, anchor_wh, stride):
    io[..., :2] = io[..., :2] * 2.0 - 0.5 + grid
    io[..., 2:4] = (io[..., 2:4] * 2) ** 2 * anchor_wh
    io[..., :4] *= stride


class YOLOLayer(nn.Module):
    def __init__(self, anchors, nc, img_size, yolo_index, layers, stride, device):
        super(YOLOLayer, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.index = yolo_index  # index of this layer in layers
        self.layers = layers  # model output layer indices
        self.stride = stride  # layer stride
        self.nl = len(layers)  # number of output layers (3)
        self.na = len(anchors)  # number of anchors (3)
        self.nc = nc  # number of classes (80)
        self.no = nc + 5  # number of outputs (85)
        self.nx, self.ny, self.ng = 0, 0, 0  # initialize number of x, y gridpoints
        self.anchor_vec = self.anchors / self.stride
        anchor_wh = self.anchor_vec.view(1, self.na, 1, 1, 2).to(device)
        self.register_buffer('anchor_wh', anchor_wh)

        self.img_size = img_size
        self.num_anchors = len(anchors)

        nxny = [int(x / stride) for x in img_size]
        grid = self._make_grid(*nxny).to(device)
        self.register_buffer('grid', grid)

    @staticmethod
    def _make_grid(nx=20, ny=20):
        yv, xv = torch.meshgrid([torch.arange(ny), torch.arange(nx)], indexing="ij")
        return torch.stack((xv, yv), 2).view((1, 1, ny, nx, 2)).float()

    def forward(self, x):
        bs, _, ny, nx = x.shape  # x(bs,255,20,20) to x(bs,3,20,20,85)
        x = x.view(bs, self.num_anchors, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

        io = x.sigmoid()
        activate_xywh(io, self.grid, self.anchor_wh, self.stride)
        return io.view(bs, -1, self.no)


class Darknet(nn.Module):
    # YOLOv3 object detection model

    def __init__(self, cfg, img_size=(416, 416), device='cpu'):
        super(Darknet, self).__init__()

        self.module_defs = parse_model_cfg(cfg)
        self.module_list, self.routs = create_modules(self.module_defs, img_size, cfg, device)
        self.yolo_layers = get_yolo_layers(self)

        # Darknet Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.array(
            [0, 2, 5], dtype=np.int32
        )  # (int32) version info: major, minor, revision
        self.seen = np.array([0], dtype=np.int64)  # (int64) number of images seen during training
        self.info()

    def forward(self, x):
        img_size = x.shape[-2:]  # height, width
        yolo_out, out = [], []

        for i, module in enumerate(self.module_list):
            name = module.__class__.__name__
            if name in [
                'WeightedFeatureFusion',
                'FeatureConcat',
                'FeatureConcat2',
                'FeatureConcat3',
                'FeatureConcat_l',
                'ScaleChannel',
                'ShiftChannel',
                'ShiftChannel2D',
                'ControlChannel',
                'ControlChannel2D',
                'AlternateChannel',
                'AlternateChannel2D',
                'SelectChannel',
                'SelectChannel2D',
                'ScaleSpatial',
            ]:  # sum, concat
                x = module(x, out)  # WeightedFeatureFusion(), FeatureConcat()
            elif name in [
                'ImplicitA',
                'ImplicitM',
                'ImplicitC',
                'Implicit2DA',
                'Implicit2DM',
                'Implicit2DC',
            ]:
                x = module()
            elif name == 'YOLOLayer':
                yolo_out.append(module(x))
            else:  # run module directly, i.e. mtype = 'convolutional', 'upsample', 'maxpool', 'batchnorm2d' etc.
                x = module(x)

            out.append(x if self.routs[i] else [])

        return torch.cat(yolo_out, 1)
        # x, p = zip(*yolo_out)  # inference output, training output
        # x = torch.cat(x, 1)  # cat yolo outputs
        # return x, p

    def fuse(self):
        # Fuse Conv2d + BatchNorm2d layers throughout model
        LOG.debug('Fuse layers')
        fused_list = nn.ModuleList()
        for a in list(self.children())[0]:
            if isinstance(a, nn.Sequential):
                for i, b in enumerate(a):
                    if isinstance(b, nn.modules.batchnorm.BatchNorm2d):
                        # fuse this bn layer with the previous conv2d layer
                        conv = a[i - 1]
                        fused = fuse_conv_and_bn(conv, b)
                        a = nn.Sequential(fused, *list(a.children())[i + 1 :])
                        break
            fused_list.append(a)
        self.module_list = fused_list
        self.info()  # yolov3-spp reduced from 225 to 152 layers

    def info(self, verbose=False, img_size=640):
        model_info(self, verbose, img_size)


def get_yolo_layers(model):
    return [
        i
        for i, m in enumerate(model.module_list)
        if m.__class__.__name__ in ['YOLOLayer', 'JDELayer']
    ]  # [89, 101, 113]


def load_darknet_weights(self, weights, cutoff=-1):
    # Parses and loads the weights stored in 'weights'

    # Establish cutoffs (load layers between 0 and cutoff. if cutoff = -1 all are loaded)
    file = Path(weights).name
    if file == 'darknet53.conv.74':
        cutoff = 75
    elif file == 'yolov3-tiny.conv.15':
        cutoff = 15

    # Read weights file
    with open(weights, 'rb') as f:
        # Read Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version = np.fromfile(
            f, dtype=np.int32, count=3
        )  # (int32) version info: major, minor, revision
        self.seen = np.fromfile(
            f, dtype=np.int64, count=1
        )  # (int64) number of images seen during training

        weights = np.fromfile(f, dtype=np.float32)  # the rest are weights

    ptr = 0
    for i, (mdef, module) in enumerate(zip(self.module_defs[:cutoff], self.module_list[:cutoff])):
        if mdef['type'] == 'convolutional':
            conv = module[0]
            if mdef['batch_normalize']:
                # Load BN bias, weights, running mean and running variance
                bn = module[1]
                nb = bn.bias.numel()  # number of biases
                # Bias
                bn.bias.data.copy_(torch.from_numpy(weights[ptr : ptr + nb]).view_as(bn.bias))
                ptr += nb
                # Weight
                bn.weight.data.copy_(torch.from_numpy(weights[ptr : ptr + nb]).view_as(bn.weight))
                ptr += nb
                # Running Mean
                bn.running_mean.data.copy_(
                    torch.from_numpy(weights[ptr : ptr + nb]).view_as(bn.running_mean)
                )
                ptr += nb
                # Running Var
                bn.running_var.data.copy_(
                    torch.from_numpy(weights[ptr : ptr + nb]).view_as(bn.running_var)
                )
                ptr += nb
            else:
                # Load conv. bias
                nb = conv.bias.numel()
                conv_b = torch.from_numpy(weights[ptr : ptr + nb]).view_as(conv.bias)
                conv.bias.data.copy_(conv_b)
                ptr += nb
            # Load conv. weights
            nw = conv.weight.numel()  # number of weights
            conv.weight.data.copy_(torch.from_numpy(weights[ptr : ptr + nw]).view_as(conv.weight))
            ptr += nw


def save_weights(self, path='model.weights', cutoff=-1):
    # Converts a PyTorch model to Darket format (*.pt to *.weights)
    # Note: Does not work if model.fuse() is applied
    with open(path, 'wb') as f:
        # Write Header https://github.com/AlexeyAB/darknet/issues/2914#issuecomment-496675346
        self.version.tofile(f)  # (int32) version info: major, minor, revision
        self.seen.tofile(f)  # (int64) number of images seen during training

        # Iterate through layers
        for i, (mdef, module) in enumerate(
            zip(self.module_defs[:cutoff], self.module_list[:cutoff])
        ):
            if mdef['type'] == 'convolutional':
                conv_layer = module[0]
                # If batch norm, load bn first
                if mdef['batch_normalize']:
                    bn_layer = module[1]
                    bn_layer.bias.data.cpu().numpy().tofile(f)
                    bn_layer.weight.data.cpu().numpy().tofile(f)
                    bn_layer.running_mean.data.cpu().numpy().tofile(f)
                    bn_layer.running_var.data.cpu().numpy().tofile(f)
                # Load conv bias
                else:
                    conv_layer.bias.data.cpu().numpy().tofile(f)
                # Load conv weights
                conv_layer.weight.data.cpu().numpy().tofile(f)


def convert(
    cfg='cfg/yolov3-spp.cfg', weights='weights/yolov3-spp.weights', saveto='converted.weights'
):
    # Converts between PyTorch and Darknet format per extension (i.e. *.weights convert to *.pt and vice versa)
    # convert('cfg/yolov3-spp.cfg', 'weights/yolov3-spp.weights')

    # Initialize model
    model = Darknet(cfg)
    ckpt = torch.load(weights)  # load checkpoint
    ckpt['model'] = {
        k: v for k, v in ckpt['model'].items() if model.state_dict()[k].numel() == v.numel()
    }
    model.load_state_dict(ckpt['model'], strict=False)
    save_weights(model, path=saveto, cutoff=-1)

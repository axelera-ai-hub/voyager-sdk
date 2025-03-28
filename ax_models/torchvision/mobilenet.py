from classifiers import AxTorchvisionClassifierModel
from torchvision import models
from torchvision.models import mobilenetv3


class AxTorchvisionMobileNetv2(models.MobileNetV2, AxTorchvisionClassifierModel):
    def __init__(self, **kwargs):
        super().__init__()


class AxTorchvisionMobileNetV3_Large(models.MobileNetV3, AxTorchvisionClassifierModel):
    def __init__(self, **kwargs):
        inverted_residual_setting, last_channel = mobilenetv3._mobilenet_v3_conf(
            "mobilenet_v3_large"
        )

        super().__init__(inverted_residual_setting, last_channel, **kwargs)


class AxTorchvisionMobileNetV3_Small(models.MobileNetV3, AxTorchvisionClassifierModel):
    def __init__(self, **kwargs):
        inverted_residual_setting, last_channel = mobilenetv3._mobilenet_v3_conf(
            "mobilenet_v3_small"
        )

        super().__init__(inverted_residual_setting, last_channel, **kwargs)

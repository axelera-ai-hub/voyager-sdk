import os

import yaml

from axelera.app import yaml_parser

# fmt: off
model_name_to_card_name_dict = {
'ax-yolov5s-axelera'                      : 'NO CARD NAME',
'convnext_base-imagenet'                  : 'ConvNeXT base',
'convnext_base-imagenet-onnx'             : 'ConvNeXT base',
'convnext_large-imagenet'                 : 'ConvNeXT large',
'convnext_large-imagenet-onnx'            : 'ConvNeXT large',
'convnext_small-imagenet'                 : 'ConvNeXT small',
'convnext_small-imagenet-onnx'            : 'ConvNeXT small',
'convnext_tiny-imagenet'                  : 'ConvNeXT tiny',
'convnext_tiny-imagenet-onnx'             : 'ConvNeXT tiny',
'crnn-mobilenetv3-large-onnx'             : 'CRNN-MobilenetV3-Large',
'deep-oc-sort-sbs50-onnx'                 : 'Deep-OC-Sort SBS50',
'densenet121-imagenet'                    : 'DenseNet-121',
'densenet121-imagenet-onnx'               : 'DenseNet-121',
'densenet161-imagenet'                    : 'DenseNet-161',
'densenet161-imagenet-onnx'               : 'DenseNet-161',
'efficientnet_b0-imagenet'                : 'EfficientNet-B0',
'efficientnet_b0-imagenet-onnx'           : 'EfficientNet-B0',
'efficientnet_b1-imagenet'                : 'EfficientNet-B1',
'efficientnet_b1-imagenet-onnx'           : 'EfficientNet-B1',
'efficientnet_b2-imagenet'                : 'EfficientNet-B2',
'efficientnet_b2-imagenet-onnx'           : 'EfficientNet-B2',
'efficientnet_b3-imagenet'                : 'EfficientNet-B3',
'efficientnet_b3-imagenet-onnx'           : 'EfficientNet-B3',
'efficientnet_b4-imagenet'                : 'EfficientNet-B4',
'efficientnet_b4-imagenet-onnx'           : 'EfficientNet-B4',
'facenet-lfw'                             : 'FaceNet - InceptionResnetV1',
'fake_cfg'                                : 'NO CARD NAME',
'inception_v3-imagenet'                   : 'Inception V3',
'inception_v3-imagenet-onnx'              : 'Inception V3',
'lprnet-onnx'                             : 'LPRNet',
'mnasnet0_5-imagenet'                     : 'MnasNet0_5',
'mnasnet0_5-imagenet-onnx'                : 'MnasNet0_5',
'mnasnet0_75-imagenet'                    : 'MnasNet0_75',
'mnasnet0_75-imagenet-onnx'               : 'MnasNet0_75',
'mnasnet1_0-imagenet'                     : 'MnasNet1_0',
'mnasnet1_0-imagenet-onnx'                : 'MnasNet1_0',
'mnasnet1_3-imagenet'                     : 'MnasNet1_3',
'mnasnet1_3-imagenet-onnx'                : 'MnasNet1_3',
'mobilenetv2-imagenet'                    : 'MobileNetV2',
'mobilenetv2-imagenet-onnx'               : 'MobileNetV2',
'mobilenetv3_large-imagenet'              : 'MobileNetV3-large',
'mobilenetv3_large-imagenet-onnx'         : 'MobileNetV3-large',
'mobilenetv3_small-imagenet'              : 'MobileNetV3-small',
'mobilenetv3_small-imagenet-onnx'         : 'MobileNetV3-small',
'mobilenetv4_small-imagenet'              : 'MobileNetV4-small',
'mobilenetv4_small-imagenet-onnx'         : 'MobileNetV4-small',
'mobilenetv4_medium-imagenet'             : 'MobileNetV4-medium',
'mobilenetv4_medium-imagenet-onnx'        : 'MobileNetV4-medium',
'mobilenetv4_large-imagenet'              : 'MobileNetV4-large',
'mobilenetv4_large-imagenet-onnx'         : 'MobileNetV4-large',
'mobilenetv4_aa_large-imagenet'           : 'MobileNetV4-aa_large',
'mobilenetv4_aa_large-imagenet-onnx'      : 'MobileNetV4-aa_large',
'parallel-yolov8spose-retinaface'         : 'NO CARD NAME',
'real-esrgan-x4plus-onnx'                 : 'Real-ESRGAN-x4plus',
'regnet_x_1_6gf-imagenet'                 : 'RegNetX-1_6GF',
'regnet_x_1_6gf-imagenet-onnx'            : 'RegNetX-1_6GF',
'regnet_x_400mf-imagenet'                 : 'RegNetX-400MF',
'regnet_x_400mf-imagenet-onnx'            : 'RegNetX-400MF',
'regnet_y_1_6gf-imagenet'                 : 'RegNetY-1_6GF',
'regnet_y_1_6gf-imagenet-onnx'            : 'RegNetY-1_6GF',
'regnet_y_400mf-imagenet'                 : 'RegNetY-400MF',
'regnet_y_400mf-imagenet-onnx'            : 'RegNetY-400MF',
'resnet101-imagenet'                      : 'ResNet-101',
'resnet101-imagenet-onnx'                 : 'ResNet-101',
'resnet10t-grayscale-imagenet'            : 'ResNet-10t 1*2456*2058',
'resnet10t-imagenet'                      : 'ResNet-10t',
'resnet10t-imagenet-onnx'                 : 'ResNet-10t',
'resnet152-imagenet'                      : 'ResNet-152',
'resnet152-imagenet-onnx'                 : 'ResNet-152',
'resnet18-imagenet'                       : 'ResNet-18',
'resnet18-imagenet-onnx'                  : 'ResNet-18',
'resnet34-imagenet'                       : 'ResNet-34',
'resnet34-imagenet-onnx'                  : 'ResNet-34',
'resnet50-imagenet'                       : 'ResNet-50 v1.5',
'resnet50-imagenet-onnx'                  : 'ResNet-50 v1.5',
'resnet50-imagenet-tf2-onnx'              : 'ResNet-50 v1.0',
'resnext50_32x4d-imagenet'                : 'ResNeXt50_32x4d',
'resnext50_32x4d-imagenet-onnx'           : 'ResNeXt50_32x4d',
'retinaface-mobilenet0.25-widerface-onnx' : 'RetinaFace - mb0.25',
'retinaface-resnet50-widerface-onnx'      : 'RetinaFace - Resnet50',
'squeezenet1.0-imagenet'                  : 'SqueezeNet 1.0',
'squeezenet1.0-imagenet-onnx'             : 'SqueezeNet 1.0',
'squeezenet1.1-imagenet'                  : 'SqueezeNet 1.1',
'squeezenet1.1-imagenet-onnx'             : 'SqueezeNet 1.1',
'ssd-mobilenetv1-coco-poc-onnx'           : 'SSD-MobileNetV1',
'ssd-mobilenetv1-coco-tracker'            : 'NO CARD NAME',
'ssd-mobilenetv1-resnet50'                : 'NO CARD NAME',
'ssd-mobilenetv2-coco-onnx'               : 'SSD-MobileNetV2',
'ssd-mobilenetv2-coco-poc-onnx'           : 'SSD-MobileNetV2',
'tf2-mobilenetv2'                         : 'NO CARD NAME',
'tf2-resnet50v2'                          : 'NO CARD NAME',
'tutorial_resnet34_caltech101'            : 'NO CARD NAME',
'unet_fcn-cityscapes'                     : 'U-Net FCN',
'unet_fcn-cityscapes-onnx'                : 'U-Net FCN',
'unet_fcn_256-cityscapes-onnx'            : 'U-Net FCN 256',
'unet_fcn_512-cityscapes'                 : 'U-Net FCN 512',
'vgg16-imagenet'                          : 'VGG16',
'vgg16-imagenet-onnx'                     : 'VGG16',
'wide_resnet50-imagenet'                  : 'Wide ResNet-50',
'wide_resnet50-imagenet-onnx'             : 'Wide ResNet-50',
'yolonas-l-coco-onnx'                     : 'YOLO-NAS L',
'yolonas-m-coco-onnx'                     : 'YOLO-NAS M',
'yolonas-s-coco-onnx'                     : 'YOLO-NAS S',
'yolov3'                                  : 'NO CARD NAME',
'yolov3-coco-onnx'                        : 'YOLOv3',
'yolov3-mask'                             : 'NO CARD NAME',
'yolov3tiny-coco-onnx'                    : 'YOLOv3-tiny',
'yolov4'                                  : 'NO CARD NAME',
'yolov4-csp-leaky'                        : 'NO CARD NAME',
'yolov4-csp-mish'                         : 'NO CARD NAME',
'yolov5l-v7-coco'                         : 'YOLOv5l',
'yolov5l-v7-coco-onnx'                    : 'YOLOv5l',
'yolov5m-tracker-resnet50'                : 'NO CARD NAME',
'yolov5m-v7-coco'                         : 'YOLOv5m',
'yolov5m-v7-coco-onnx'                    : 'YOLOv5m',
'yolov5m-v7-coco-tracker'                 : 'NO CARD NAME',
'yolov5n-v7-coco'                         : 'YOLOv5n',
'yolov5n-v7-coco-onnx'                    : 'YOLOv5n',
'yolov5s-relu-coco'                       : 'YOLOv5s-Relu',
'yolov5s-relu-coco-onnx'                  : 'YOLOv5s-Relu',
'yolov5s-v5-coco'                         : 'YOLOv5s-v5',
'yolov5s-v5-coco-onnx'                    : 'YOLOv5s-v5',
'yolov5s-v7-barrel-onnx'                  : 'NO CARD NAME',
'yolov5s-v7-coco'                         : 'YOLOv5s',
'yolov5s-v7-coco-onnx'                    : 'YOLOv5s',
'yolov5s-v7-perspective-barrel-onnx'      : 'NO CARD NAME',
'yolov5s-v7-perspective-onnx'             : 'NO CARD NAME',
'yolov6m-coco-onnx'                       : 'YOLOv6m',
'yolov7-640x480-coco'                     : 'YOLOv7 640x480',
'yolov7-640x480-coco-onnx'                : 'YOLOv7 640x480',
'yolov7-coco'                             : 'YOLOv7',
'yolov7-coco-onnx'                        : 'YOLOv7',
'yolov7-tiny-coco'                        : 'YOLOv7-tiny',
'yolov7-tiny-coco-onnx'                   : 'YOLOv7-tiny',
'yolov7-w6-coco'                          : 'YOLOv7-W6',
'yolov7-w6-coco-onnx'                     : 'YOLOv7-W6',
'yolov8l-coco-onnx'                       : 'YOLOv8l',
'yolov8l-coco'                            : 'YOLOv8l',
'yolov8lpose-coco-onnx'                   : 'YOLOv8l-pose',
'yolov8lpose-coco'                        : 'YOLOv8l-pose',
'yolov8lseg-coco-onnx'                    : 'YOLOv8l-seg',
'yolov8lseg-coco'                         : 'YOLOv8l-seg',
'yolov8m-coco-onnx'                       : 'YOLOv8m',
'yolov8m-coco'                            : 'YOLOv8m',
'yolov8mpose-coco-onnx'                   : 'YOLOv8m-pose',
'yolov8mpose-coco'                        : 'YOLOv8m-pose',
'yolov8mseg-coco-onnx'                    : 'YOLOv8m-seg',
'yolov8mseg-coco'                         : 'YOLOv8m-seg',
'yolov8n-coco-onnx'                       : 'YOLOv8n',
'yolov8n-coco'                            : 'YOLOv8n',
'yolov8n-license-plate'                   : 'NO CARD NAME',
'yolov8n-weapons-and-knives'              : 'NO CARD NAME',
'yolov8n-yolov8s'                         : 'NO CARD NAME',
'yolov8npose-coco-onnx'                   : 'YOLOv8n-pose',
'yolov8npose-coco'                        : 'YOLOv8n-pose',
'yolov8nseg-coco-onnx'                    : 'YOLOv8n-seg',
'yolov8nseg-coco'                         : 'YOLOv8n-seg',
'yolov8s-coco-onnx'                       : 'YOLOv8s',
'yolov8s-coco'                            : 'YOLOv8s',
'yolov8spose-coco-onnx'                   : 'YOLOv8s-pose',
'yolov8spose-coco'                        : 'YOLOv8s-pose',
'yolov8spose-yolov8n'                     : 'NO CARD NAME',
'yolov8spose-yolov8n-weapons'             : 'NO CARD NAME',
'yolov8sseg-coco-onnx'                    : 'YOLOv8s-seg',
'yolov8sseg-coco'                         : 'YOLOv8s-seg',
'yolov9c-coco-onnx'                       : 'YOLOv9c',
'yolov9m-coco-onnx'                       : 'YOLOv9m',
'yolov9s-coco-onnx'                       : 'YOLOv9s',
'yolov9t-coco-onnx'                       : 'YOLOv9t',
'yolo11n-coco-onnx'                       : 'YOLO11n',
'yolo11n-coco'                            : 'YOLO11n',
'yolo11s-coco-onnx'                       : 'YOLO11s',
'yolo11s-coco'                            : 'YOLO11s',
'yolo11m-coco-onnx'                       : 'YOLO11m',
'yolo11m-coco'                            : 'YOLO11m',
'yolo11l-coco-onnx'                       : 'YOLO11l',
'yolo11l-coco'                            : 'YOLO11l',
'yolo11x-coco-onnx'                       : 'YOLO11x',
'yolo11x-coco'                            : 'YOLO11x',
'yolo11npose-coco-onnx'                   : 'YOLO11n-pose',
'yolo11npose-coco'                        : 'YOLO11n-pose',
'yolo11lpose-coco-onnx'                   : 'YOLO11l-pose',
'yolo11lpose-coco'                        : 'YOLO11l-pose',
'yolo11nseg-coco-onnx'                    : 'YOLO11n-seg',
'yolo11nseg-coco'                         : 'YOLO11n-seg',
'yolo11lseg-coco-onnx'                    : 'YOLO11l-seg',
'yolo11lseg-coco'                         : 'YOLO11l-seg',
'yolox-m-coco-onnx'                       : 'YOLOX-m',
'yolox-s-coco-onnx'                       : 'YOLOX-s',
'ces2025-ls'                              : 'NO CARD NAME',
'ces2025-ln'                              : 'NO CARD NAME',
}
# fmt: on


def _read_yaml_file(filename):
    with open(filename, "r") as f:
        return yaml.safe_load(f)


def _is_there_a_dictionary_entry(model_name, expected_card_name):
    assert expected_card_name is not None, (
        f"Please add '{model_name}' with expected card_name to the"
        " model_name_to_card_name_dict in tests/test_yaml_mc_card_name.py"
    )


def _cannot_have_a_no_card_name_for_release_model(
    model_name, release_models, expected_card_name, full_path
):
    if model_name in release_models:
        assert expected_card_name != "NO CARD NAME", (
            f"The release model '{model_name}' cannot have an expected card_name 'NO CARD NAME'"
            f". Please set the expected card_name in the model_name_to_card_name_dict"
            f" in tests/test_yaml_mc_card_name.py and add the card_name to the model yaml file"
            f" {full_path}"
        )


def _is_expected_value(model_name, card_name, expected_card_name, full_path):
    assert card_name == expected_card_name, (
        f"The card name '{card_name}' for '{model_name}' does not match the"
        f" expected '{expected_card_name}' in the model_name_to_card_name_dict"
        f" in test_yaml_mc_card_name.py. Please check the card_name in {full_path} or"
        " set the expected in the dictionary correctly."
    )


def test_card_names():
    file_dir_path = os.path.dirname(os.path.realpath(__file__))
    network_yaml_info = yaml_parser.get_network_yaml_info(model_cards_only=True)
    release_models_path = os.path.join(
        file_dir_path, '..', 'internal_tools', 'model_release_candidates.yaml'
    )
    rel_models_dict = _read_yaml_file(release_models_path)
    release_models = rel_models_dict['VALIDATION']

    for nn in network_yaml_info.get_all_info():
        model_name = nn.name
        full_path = nn.yaml_path
        yamldict = _read_yaml_file(full_path)
        try:
            card_name = yamldict['internal-model-card']['card_name']
        except KeyError:
            card_name = "NO CARD NAME"
        expected_card_name = model_name_to_card_name_dict.get(model_name, None)

        _is_there_a_dictionary_entry(model_name, expected_card_name)
        _cannot_have_a_no_card_name_for_release_model(
            model_name, release_models, expected_card_name, full_path
        )
        _is_expected_value(model_name, card_name, expected_card_name, full_path)


if __name__ == '__main__':
    test_card_names()

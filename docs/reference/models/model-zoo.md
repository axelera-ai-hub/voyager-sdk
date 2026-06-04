---
title: "Model Zoo"
---
# Model Zoo — Pre-trained Models

The Voyager Model Zoo is a collection of pre-trained AI models ready to run on Axelera hardware. When you run `inference.py yolov5s-v7-coco usb:0`, the model name (`yolov5s-v7-coco`) comes from the Model Zoo.

## Listing available models

From the SDK root directory (with environment activated):

```bash
make
```

This lists three categories:

| Category | What it contains |
|----------|-----------------|
| **ZOO** | Individual models — one model, one task |
| **REFERENCE APPLICATION PIPELINES** | Multi-model pipelines (e.g., detection cascaded into pose estimation) |
| **TUTORIALS** | Example models used by the tutorial documentation |

## How model names work

Model names follow a pattern:

```
<architecture>-<dataset>[-<variant>]
```

Examples:

| Name | Architecture | Dataset | Notes |
|------|-------------|---------|-------|
| `yolov5s-v7-coco` | YOLOv5 small | [COCO](../../glossary.md#coco) | v7 release of YOLOv5 |
| `yolov8s-coco-onnx` | YOLOv8 small | COCO | ONNX format |
| `resnet50-imagenet` | ResNet-50 | ImageNet | Classification model |

## Task types

| Task | What it does | Example model |
|------|-------------|---------------|
| Object detection | Finds and labels objects with bounding boxes | `yolov5s-v7-coco` |
| Classification | Identifies what's in an image (single label) | `resnet50-imagenet` |
| Semantic segmentation | Labels every pixel by category | `yolov8sseg-coco-onnx` |
| Instance segmentation | Labels every pixel AND distinguishes individual objects | `yolov8sseg-coco-onnx` |
| Keypoint detection | Finds body joints and pose landmarks | `yolov8lpose-coco-onnx` |
| Depth estimation | Estimates distance of each pixel from camera | `fastdepth-nyuv2` |
| License plate recognition | Reads license plates | Available in Model Zoo |
| Face recognition | Identifies or verifies faces | Available in Model Zoo |

## Running a model

```bash
# Object detection with USB camera
./inference.py yolov5s-v7-coco usb:0

# Classification with a video file
./inference.py resnet50-imagenet media/traffic1_1080p.mp4

# Headless benchmarking (no display)
./inference.py yolov8s-coco-onnx usb:0 --no-display --frames 1000
```

The first time you run a model, the SDK:

1. Downloads the pre-trained weights (if not cached)
2. Compiles the model for the [AIPU](../../glossary.md#aipu)
3. Caches the compiled model for subsequent runs
4. Runs [inference](../../glossary.md#inference)

Subsequent runs skip steps 1-3 and start immediately.

## Datasets

Models are trained on specific datasets. The dataset name in the model identifier tells you what the model can recognize:

| Dataset | What it contains | Typical use |
|---------|-----------------|-------------|
| [COCO](../../glossary.md#coco) | 80 object categories (person, car, dog, etc.) | General object detection |
| ImageNet | 1000 image categories | Image classification |
| VOC | 20 object categories | Object detection (smaller set) |

## Non-redistributable datasets

Most datasets download automatically when you first run a model. A few datasets require manual registration and download due to licensing restrictions. These must be downloaded by hand from the links below, then placed in the specified directory within your SDK installation. The SDK raises an error with the expected path if a required dataset is missing.

| Dataset | Archive | Download location |
|---------|---------|-------------------|
| [Cityscapes (val)](https://www.cityscapes-dataset.com/) | `gtFine_val.zip` | `data/cityscapes` |
| [Cityscapes (val)](https://www.cityscapes-dataset.com/) | `leftImg8bit_val.zip` | `data/cityscapes` |
| [Cityscapes (test)](https://www.cityscapes-dataset.com/) | `gtFine_test.zip` | `data/cityscapes` |
| [Cityscapes (test)](https://www.cityscapes-dataset.com/) | `leftImg8bit_test.zip` | `data/cityscapes` |
| [ImageNet (train)](https://www.image-net.org/download.php) | `ILSVRC2012_devkit_t12.tar.gz` | `data/ImageNet` |
| [ImageNet (train)](https://www.image-net.org/download.php) | `ILSVRC2012_img_train.tar` | `data/ImageNet` |
| [ImageNet (val)](https://www.image-net.org/download.php) | `ILSVRC2012_devkit_t12.tar.gz` | `data/ImageNet` |
| [ImageNet (val)](https://www.image-net.org/download.php) | `ILSVRC2012_img_val.tar` | `data/ImageNet` |
| WiderFace (train) | `widerface_train.zip` | `data/widerface` |
| WiderFace (val) | `widerface_val.zip` | `data/widerface` |

You are responsible for adhering to the terms and conditions of each dataset's license.

---

## Performance characteristics

The tables below list all Model Zoo models for this SDK release. Columns:

- **Ref FP32** — accuracy of the original floating-point model
- **Accuracy loss** — FP32 accuracy minus quantized int8 accuracy (lower is better)
- **Ref PCIe FPS** — host throughput on Intel Core i9-13900K + Metis 1× PCIe card
- **Ref M.2 FPS** — host throughput on Intel Core i5-1145G7E + Metis 1× M.2 card

Accuracy is measured using:

```bash
./inference.py <model> dataset --pipe=torch-aipu --no-display
```

Throughput is measured using a 720p h.264 video file:

```bash
./inference.py <model> media/traffic2_720p.mp4 --pipe=gst --no-display
```

### Image Classification

| Model                                                                                      | ONNX                                                                                      | Repo                                                             | Resolution | Dataset     | Ref FP32 Top1 | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :----------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------- | :--------------------------------------------------------------- | :--------- | :---------- | ------------: | ------------: | -----------: | ----------: | ------------: |
| [DenseNet-121](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/densenet121-imagenet.yaml)        | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/densenet121-imagenet-onnx.yaml)     | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 74.44         | 0.86          | 281          | 156         | BSD-3-Clause  |
| [EfficientNet-B0](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/efficientnet_b0-imagenet.yaml) | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/efficientnet_b0-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 77.67         | 1.12          | 1429         | 1450        | BSD-3-Clause  |
| [EfficientNet-B1](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/efficientnet_b1-imagenet.yaml) | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/efficientnet_b1-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 77.6          | 0.47          | 972          | 960         | BSD-3-Clause  |
| [EfficientNet-B2](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/efficientnet_b2-imagenet.yaml) | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/efficientnet_b2-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 77.79         | 0.46          | 903          | 863         | BSD-3-Clause  |
| [EfficientNet-B3](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/efficientnet_b3-imagenet.yaml) | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/efficientnet_b3-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 78.54         | 0.50          | 787          | 721         | BSD-3-Clause  |
| [EfficientNet-B4](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/efficientnet_b4-imagenet.yaml) | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/efficientnet_b4-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 79.27         | 0.71          | 576          | 436         | BSD-3-Clause  |
| [MobileNetV2](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/mobilenetv2-imagenet.yaml)         | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/mobilenetv2-imagenet-onnx.yaml)     | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 71.87         | 1.50          | 3670         | 3638        | BSD-3-Clause  |
| [MobileNetV4-small](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/timm/mobilenetv4_small-imagenet.yaml)                   | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/timm/mobilenetv4_small-imagenet-onnx.yaml)                     | [&#x1F517;](https://github.com/huggingface/pytorch-image-models) | 224x224    | ImageNet-1K | 73.74         | 5.07          | 4937         | 4807        | Apache 2.0    |
| [MobileNetV4-medium](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/timm/mobilenetv4_medium-imagenet.yaml)                 | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/timm/mobilenetv4_medium-imagenet-onnx.yaml)                    | [&#x1F517;](https://github.com/huggingface/pytorch-image-models) | 224x224    | ImageNet-1K | 79.04         | 0.90          | 2517         | 2395        | Apache 2.0    |
| [MobileNetV4-large](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/timm/mobilenetv4_large-imagenet.yaml)                   | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/timm/mobilenetv4_large-imagenet-onnx.yaml)                     | [&#x1F517;](https://github.com/huggingface/pytorch-image-models) | 384x384    | ImageNet-1K | 82.92         | 0.95          | 761          | 460         | Apache 2.0    |
| [MobileNetV4-aa_large](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/timm/mobilenetv4_aa_large-imagenet.yaml)             | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/timm/mobilenetv4_aa_large-imagenet-onnx.yaml)                  | [&#x1F517;](https://github.com/huggingface/pytorch-image-models) | 384x384    | ImageNet-1K | 83.22         | 1.96          | 667          | 391         | Apache 2.0    |
| [SqueezeNet 1.0](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/squeezenet1.0-imagenet.yaml)    | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/squeezenet1.0-imagenet-onnx.yaml)   | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 58.1          | 2.80          | 953          | 811         | BSD-3-Clause  |
| [SqueezeNet 1.1](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/squeezenet1.1-imagenet.yaml)    | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/squeezenet1.1-imagenet-onnx.yaml)   | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 58.19         | 1.86          | 7298         | 7264        | BSD-3-Clause  |
| [Inception V3](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/inception_v3-imagenet.yaml)       | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/inception_v3-imagenet-onnx.yaml)    | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 69.85         | 0.25          | 1136         | 636         | BSD-3-Clause  |
| [RegNetX-1_6GF](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/regnet_x_1_6gf-imagenet.yaml)    | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/regnet_x_1_6gf-imagenet-onnx.yaml)  | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 79.33         | 0.22          | 695          | 369         | BSD-3-Clause  |
| [RegNetX-400MF](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/regnet_x_400mf-imagenet.yaml)    | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/regnet_x_400mf-imagenet-onnx.yaml)  | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 74.48         | 0.36          | 1199         | 636         | BSD-3-Clause  |
| [RegNetY-1_6GF](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/regnet_y_1_6gf-imagenet.yaml)    | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/regnet_y_1_6gf-imagenet-onnx.yaml)  | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 80.73         | 0.24          | 595          | 322         | BSD-3-Clause  |
| [RegNetY-400MF](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/regnet_y_400mf-imagenet.yaml)    | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/regnet_y_400mf-imagenet-onnx.yaml)  | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 75.63         | 0.13          | 1642         | 975         | BSD-3-Clause  |
| [ResNet-18](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/resnet18-imagenet.yaml)              | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/resnet18-imagenet-onnx.yaml)        | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 69.76         | 0.36          | 3904         | 3749        | BSD-3-Clause  |
| [ResNet-34](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/resnet34-imagenet.yaml)              | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/resnet34-imagenet-onnx.yaml)        | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 73.3          | 0.12          | 2282         | 2075        | BSD-3-Clause  |
| [ResNet-50 v1.5](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/resnet50-imagenet.yaml)         | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/resnet50-imagenet-onnx.yaml)        | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 76.15         | 0.18          | 1946         | 1756        | BSD-3-Clause  |
| [ResNet-101](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/resnet101-imagenet.yaml)            | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/resnet101-imagenet-onnx.yaml)       | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 77.37         | 0.79          | 1049         | 673         | BSD-3-Clause  |
| [ResNet-152](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/resnet152-imagenet.yaml)            | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/resnet152-imagenet-onnx.yaml)       | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 78.31         | 0.23          | 493          | 261         | BSD-3-Clause  |
| [ResNet-10t](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/timm/resnet10t-imagenet.yaml)                                  | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/timm/resnet10t-imagenet-onnx.yaml)                             | [&#x1F517;](https://huggingface.co/timm/resnet10t.c3_in1k)       | 224x224    | ImageNet-1K | 68.22         | 1.06          | 5212         | 5015        | Apache 2.0    |
| [ResNeXt50_32x4d](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/resnext50_32x4d-imagenet.yaml) | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/resnext50_32x4d-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 77.61         | 0.08          | 437          | 236         | BSD-3-Clause  |
| [Wide ResNet-50](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/wide_resnet50-imagenet.yaml)    | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torchvision/classification/wide_resnet50-imagenet-onnx.yaml)   | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 78.48         | 0.36          | 436          | 236         | BSD-3-Clause  |

### Object Detection

| Model                                                                           | ONNX                                                                                       | Repo                                                                                                        | Resolution | Dataset                   | Ref FP32 mAP | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------- | :--------- | :------------------------ | -----------: | ------------: | -----------: | ----------: | ------------: |
| GELAN-s                                                                         | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/gelan-s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/WongKinYiu/yolov9)                                                           | 640x640    | COCO2017                  | 46.41        | 2.99          | 376          | 237         | GPL-3.0       |
| GELAN-m                                                                         | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/gelan-m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/WongKinYiu/yolov9)                                                           | 640x640    | COCO2017                  | 50.86        | 1.06          | 203          | 148         | GPL-3.0       |
| GELAN-c                                                                         | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/gelan-c-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/WongKinYiu/yolov9)                                                           | 640x640    | COCO2017                  | 52.3         | 0.49          | 199          | 144         | GPL-3.0       |
| RetinaFace - Resnet50                                                           | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torch/retinaface-resnet50-widerface-onnx.yaml)                  | [&#x1F517;](https://github.com/biubug6/Pytorch_Retinaface/tree/master)                                      | 840x840    | WiderFace                 | 95.25        | 0.25          | 90           | 51          | MIT           |
| RetinaFace - mb0.25                                                             | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torch/retinaface-mobilenet0.25-widerface-onnx.yaml)             | [&#x1F517;](https://github.com/biubug6/Pytorch_Retinaface/tree/master)                                      | 640x640    | WiderFace                 | 89.44        | 1.36          | 1020         | 774         | MIT           |
| SSD-MobileNetV1                                                                 | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/tensorflow/object_detection/ssd-mobilenetv1-coco-poc-onnx.yaml) | [&#x1F517;](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) | 300x300    | COCO2017                  | 24.77        | -0.05         | 3356         | 3019        | Apache 2.0    |
| SSD-MobileNetV2                                                                 | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/tensorflow/object_detection/ssd-mobilenetv2-coco-poc-onnx.yaml) | [&#x1F517;](https://github.com/tensorflow/models)                                                           | 300x300    | COCO2017                  | 19.25        | 0.87          | 2261         | 2195        | Apache 2.0    |
| YOLOv3                                                                          | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov3-coco-onnx.yaml)                    | [&#x1F517;](https://github.com/ultralytics/yolov3)                                                          | 640x640    | COCO2017                  | 46.61        | 0.79          | 163          | 96          | AGPL-3.0      |
| [YOLOv5s-Relu](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov5s-relu-coco.yaml)     | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov5s-relu-coco-onnx.yaml)              | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017                  | 35.09        | 0.52          | 785          | 536         | AGPL-3.0      |
| [YOLOv5s-v5](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov5s-v5-coco.yaml)         | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov5s-v5-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017                  | 36.18        | 0.37          | 790          | 526         | AGPL-3.0      |
| [YOLOv5n](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov5n-v7-coco.yaml)            | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov5n-v7-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017                  | 27.72        | 0.87          | 1028         | 656         | AGPL-3.0      |
| [YOLOv5s](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov5s-v7-coco.yaml)            | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov5s-v7-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017                  | 37.25        | 0.80          | 865          | 824         | AGPL-3.0      |
| [YOLOv5m](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov5m-v7-coco.yaml)            | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov5m-v7-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017                  | 44.94        | 0.85          | 455          | 322         | AGPL-3.0      |
| [YOLOv5l](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov5l-v7-coco.yaml)            | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov5l-v7-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017                  | 48.67        | 0.84          | 299          | 204         | AGPL-3.0      |
| [YOLOv7](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov7-coco.yaml)                 | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov7-coco-onnx.yaml)                    | [&#x1F517;](https://github.com/WongKinYiu/yolov7)                                                           | 640x640    | COCO2017                  | 51.02        | 0.58          | 212          | 173         | GPL-3.0       |
| [YOLOv7-tiny](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov7-tiny-coco.yaml)       | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov7-tiny-coco-onnx.yaml)               | [&#x1F517;](https://github.com/WongKinYiu/yolov7)                                                           | 416x416    | COCO2017                  | 33.12        | 0.49          | 1441         | 1110        | GPL-3.0       |
| [YOLOv7 640x480](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov7-640x480-coco.yaml) | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov7-640x480-coco-onnx.yaml)            | [&#x1F517;](https://github.com/WongKinYiu/yolov7)                                                           | 640x480    | COCO2017                  | 50.78        | 0.52          | 242          | 164         | GPL-3.0       |
| [YOLOv8n](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov8n-coco.yaml)               | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov8n-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 37.12        | 1.18          | 834          | 764         | AGPL-3.0      |
| [YOLOv8s](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov8s-coco.yaml)               | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov8s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 44.8         | 0.93          | 643          | 524         | AGPL-3.0      |
| [YOLOv8m](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov8m-coco.yaml)               | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov8m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 50.16        | 1.32          | 242          | 177         | AGPL-3.0      |
| [YOLOv8l](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov8l-coco.yaml)               | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov8l-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 52.83        | 2.06          | 181          | 142         | AGPL-3.0      |
| YOLOv8n-obb                                                                     | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/obb_detection/yolov8n-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset | 48.73        | 5.68          | 269          | 162         | AGPL-3.0      |
| YOLOv8l-obb                                                                     | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/obb_detection/yolov8l-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset | 56.06        | 4.41          | 36           | 19          | AGPL-3.0      |
| YOLOX-s                                                                         | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolox-s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/Megvii-BaseDetection/YOLOX)                                                  | 640x640    | COCO2017                  | 39.24        | -0.81         | 642          | 423         | Apache-2.0    |
| YOLOX-m                                                                         | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolox-m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/Megvii-BaseDetection/YOLOX)                                                  | 640x640    | COCO2017                  | 46.26        | -0.37         | 349          | 268         | Apache-2.0    |
| YOLOX-x Human                                                                   | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolox-x-crowdhuman-onnx.yaml)             | [&#x1F517;](https://github.com/FoundationVision/ByteTrack)                                                  | 1440x800   | COCO2017                  | 57.66        | 3.38          | 21           | -           | MIT           |
| YOLOv9t                                                                         | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov9t-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 37.81        | 1.25          | 415          | 247         | AGPL-3.0      |
| YOLOv9s                                                                         | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov9s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 46.28        | 1.12          | 374          | 237         | AGPL-3.0      |
| YOLOv9m                                                                         | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov9m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 51.24        | 2.29          | 203          | 148         | AGPL-3.0      |
| YOLOv9c                                                                         | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov9c-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 52.67        | 2.35          | 194          | 150         | AGPL-3.0      |
| YOLOv10n                                                                        | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov10n-coco-onnx.yaml)                  | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 38.08        | 0.74          | 738          | 561         | AGPL-3.0      |
| YOLOv10s                                                                        | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov10s-coco-onnx.yaml)                  | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 45.74        | 0.45          | 580          | 461         | AGPL-3.0      |
| YOLOv10b                                                                        | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolov10b-coco-onnx.yaml)                  | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 51.79        | 0.45          | 251          | 217         | AGPL-3.0      |
| YOLO11n                                                                         | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolo11n-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 39.17        | 0.71          | 759          | 574         | AGPL-3.0      |
| YOLO11s                                                                         | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolo11s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 46.54        | 0.55          | 565          | 426         | AGPL-3.0      |
| YOLO11m                                                                         | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolo11m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 51.31        | 0.55          | 269          | 196         | AGPL-3.0      |
| YOLO11l                                                                         | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolo11l-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 53.23        | 0.49          | 183          | 125         | AGPL-3.0      |
| YOLO11x                                                                         | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolo11x-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 54.67        | 0.58          | 53           | 31          | AGPL-3.0      |
| YOLO11n-obb                                                                     | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/obb_detection/yolo11n-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset | 50.01        | 1.07          | 250          | 172         | AGPL-3.0      |
| YOLO11l-obb                                                                     | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/obb_detection/yolo11l-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset | 56.41        | 1.08          | 36           | 20          | AGPL-3.0      |
| YOLO26n                                                                         | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolo26n-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 40.18        | 1.95          | 662          | 487         | AGPL-3.0      |
| YOLO26s                                                                         | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolo26s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 47.66        | 2.05          | 498          | 396         | AGPL-3.0      |
| YOLO26m                                                                         | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolo26m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 52.45        | 2.14          | 258          | 192         | AGPL-3.0      |
| YOLO26l                                                                         | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolo26l-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 54.11        | 2.03          | 179          | 122         | AGPL-3.0      |
| YOLO26x                                                                         | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolo26x-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 56.92        | 2.43          | 53           | 31          | AGPL-3.0      |
| YOLO26n-obb                                                                     | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/obb_detection/yolo26n-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset | 49.41        | 3.12          | 206          | 139         | AGPL-3.0      |
| YOLO26s-obb                                                                     | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/obb_detection/yolo26s-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset | 54.02        | 2.01          | 167          | 114         | AGPL-3.0      |
| YOLO26m-obb                                                                     | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/obb_detection/yolo26m-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset | 56.66        | 1.72          | 58           | 33          | AGPL-3.0      |
| YOLO26l-obb                                                                     | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/obb_detection/yolo26l-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset | 57.35        | 1.05          | 34           | 19          | AGPL-3.0      |
| YOLO26x-obb                                                                     | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/obb_detection/yolo26x-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset | 58.4         | 5.34          | 15           | -           | AGPL-3.0      |
| YOLO-NAS S                                                                      | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolonas-s-coco-onnx.yaml)                 | [&#x1F517;](https://github.com/Deci-AI/super-gradients)                                                     | 640x640    | COCO2017                  | 47.06        |               | 450          | 318         | Apache-2.0    |
| YOLO-NAS M                                                                      | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolonas-m-coco-onnx.yaml)                 | [&#x1F517;](https://github.com/Deci-AI/super-gradients)                                                     | 640x640    | COCO2017                  | 51.0         |               | 285          | 221         | Apache-2.0    |
| YOLO-NAS L                                                                      | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/object_detection/yolonas-l-coco-onnx.yaml)                 | [&#x1F517;](https://github.com/Deci-AI/super-gradients)                                                     | 640x640    | COCO2017                  | 51.79        |               | 157          | 96          | Apache-2.0    |

### Semantic Segmentation

| Model                                                                    | ONNX                                                                      | Repo                                                                             | Resolution | Dataset    | Ref FP32 mIoU | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :----------------------------------------------------------------------- | :------------------------------------------------------------------------ | :------------------------------------------------------------------------------- | :--------- | :--------- | ------------: | ------------: | -----------: | ----------: | ------------: |
| U-Net FCN 256                                                            | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/mmlab/mmseg/unet_fcn_256-cityscapes-onnx.yaml) | [&#x1F517;](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/unet) | 256x256    | Cityscapes | 57.75         | 0.34          | 249          | 198         | Apache 2.0    |
| [U-Net FCN 512](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/mmlab/mmseg/unet_fcn_512-cityscapes.yaml) |                                                                           | [&#x1F517;](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/unet) | 512x512    | Cityscapes | 66.62         | 0.01          | 34           | 19          | Apache 2.0    |

### Instance Segmentation

| Model                                                                         | ONNX                                                                             | Repo                                                    | Resolution | Dataset  | Ref FP32 mAP | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :---------------------------------------------------------------------------- | :------------------------------------------------------------------------------- | :------------------------------------------------------ | :--------- | :------- | -----------: | ------------: | -----------: | ----------: | ------------: |
| [YOLOv8n-seg](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/instance_segmentation/yolov8nseg-coco.yaml) | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/instance_segmentation/yolov8nseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 29.98        | 0.92          | 639          | 433         | AGPL-3.0      |
| [YOLOv8s-seg](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/instance_segmentation/yolov8sseg-coco.yaml) | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/instance_segmentation/yolov8sseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 36.32        | 0.57          | 482          | 345         | AGPL-3.0      |
| [YOLOv8m-seg](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/instance_segmentation/yolov8mseg-coco.yaml) | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/instance_segmentation/yolov8mseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 40.39        | 0.65          | 198          | 156         | AGPL-3.0      |
| [YOLOv8l-seg](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/instance_segmentation/yolov8lseg-coco.yaml) | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/instance_segmentation/yolov8lseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 42.27        | 1.11          | 167          | 134         | AGPL-3.0      |
| YOLO11n-seg                                                                   | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/instance_segmentation/yolo11nseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 31.84        | 1.11          | 598          | 406         | AGPL-3.0      |
| YOLO11l-seg                                                                   | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/instance_segmentation/yolo11lseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 43.26        | 0.13          | 156          | 107         | AGPL-3.0      |
| YOLO26n-seg                                                                   | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/instance_segmentation/yolo26nseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 32.95        | 2.64          | 516          | 352         | AGPL-3.0      |
| YOLO26s-seg                                                                   | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/instance_segmentation/yolo26sseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 39.28        | 3.28          | 385          | 292         | AGPL-3.0      |
| YOLO26m-seg                                                                   | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/instance_segmentation/yolo26mseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 43.34        | 1.40          | 201          | 156         | AGPL-3.0      |
| YOLO26l-seg                                                                   | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/instance_segmentation/yolo26lseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 45.09        | 1.87          | 141          | 96          | AGPL-3.0      |
| YOLO26x-seg                                                                   | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/instance_segmentation/yolo26xseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 46.54        | 2.00          | 46           | 28          | AGPL-3.0      |

### Keypoint Detection

| Model                                                                        | ONNX                                                                           | Repo                                                    | Resolution | Dataset  | Ref FP32 mAP | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :--------------------------------------------------------------------------- | :----------------------------------------------------------------------------- | :------------------------------------------------------ | :--------- | :------- | -----------: | ------------: | -----------: | ----------: | ------------: |
| [YOLOv8n-pose](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/keypoint_detection/yolov8npose-coco.yaml) | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/keypoint_detection/yolov8npose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 51.11        | 1.75          | 822          | 723         | AGPL-3.0      |
| [YOLOv8s-pose](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/keypoint_detection/yolov8spose-coco.yaml) | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/keypoint_detection/yolov8spose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 60.65        | 2.98          | 592          | 471         | AGPL-3.0      |
| [YOLOv8m-pose](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/keypoint_detection/yolov8mpose-coco.yaml) | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/keypoint_detection/yolov8mpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 65.58        | 1.91          | 231          | 168         | AGPL-3.0      |
| [YOLOv8l-pose](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/keypoint_detection/yolov8lpose-coco.yaml) | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/keypoint_detection/yolov8lpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 68.39        | 1.47          | 186          | 145         | AGPL-3.0      |
| YOLO11n-pose                                                                 | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/keypoint_detection/yolo11npose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 51.15        | 3.23          | 759          | 532         | AGPL-3.0      |
| YOLO11l-pose                                                                 | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/keypoint_detection/yolo11lpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 67.44        | 3.14          | 179          | 122         | AGPL-3.0      |
| YOLO26n-pose                                                                 | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/keypoint_detection/yolo26npose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 57.66        | 6.54          | 658          | 450         | AGPL-3.0      |
| YOLO26s-pose                                                                 | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/keypoint_detection/yolo26spose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 63.61        | 5.12          | 467          | 359         | AGPL-3.0      |
| YOLO26m-pose                                                                 | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/keypoint_detection/yolo26mpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 69.54        | 4.83          | 235          | 166         | AGPL-3.0      |
| YOLO26l-pose                                                                 | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/keypoint_detection/yolo26lpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 71.05        | 3.02          | 174          | 120         | AGPL-3.0      |
| YOLO26x-pose                                                                 | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/yolo/keypoint_detection/yolo26xpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 72.75        | 16.62         | 51           | 30          | AGPL-3.0      |

### Depth Estimation

| Model     | ONNX                                                             | Repo                                                                              | Resolution | Dataset    | Ref FP32 RMSE | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :-------- | :--------------------------------------------------------------- | :-------------------------------------------------------------------------------- | :--------- | :--------- | ------------: | ------------: | -----------: | ----------: | ------------: |
| FastDepth | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torch/fastdepth-nyudepthv2-onnx.yaml) | [&#x1F517;](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/146_FastDepth) | 224x224    | NYUDepthV2 | 0.6574        | -0.0065       | 974          | 855         | MIT           |

### License Plate Recognition

| Model                                      | ONNX | Repo                                                     | Resolution | Dataset       | Ref FP32 WLA | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :----------------------------------------- | :--- | :------------------------------------------------------- | :--------- | :------------ | -----------: | ------------: | -----------: | ----------: | ------------: |
| [LPRNet](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torch/lprnet.yaml) |      | [&#x1F517;](https://github.com/sirius-ai/LPRNet_Pytorch) | 94x24      | LPRNetDataset | 89.4         | 1.90          | 10268        | 9335        | Apache-2.0    |

### Image Enhancement (Super Resolution)

| Model              | ONNX                                                           | Repo                                                | Resolution | Dataset                         | Ref FP32 PSNR | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :----------------- | :------------------------------------------------------------- | :-------------------------------------------------- | :--------- | :------------------------------ | ------------: | ------------: | -----------: | ----------: | ------------: |
| Real-ESRGAN-x4plus | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torch/real-esrgan-x4plus-onnx.yaml) | [&#x1F517;](https://github.com/xinntao/Real-ESRGAN) | 128x128    | SuperResolutionCustomSet128x128 | 24.77         |               | -            | -           | BSD-3-Clause  |

### Face Recognition

| Model                                                                | ONNX                                                    | Repo                                                     | Resolution | Dataset            | Ref FP32 top1_avg | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :------------------------------------------------------------------- | :------------------------------------------------------ | :------------------------------------------------------- | :--------- | :----------------- | ----------------: | ------------: | -----------: | ----------: | ------------: |
| [FaceNet - InceptionResnetV1](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torch/facenet-lfw.yaml) | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torch/facenet-lfw-onnx.yaml) | [&#x1F517;](https://github.com/timesler/facenet-pytorch) | 160x160    | LFWTorchvisionPair | 98.35             | 0.00          | 1321         | 720         | MIT           |

### Re-Identification

| Model      | ONNX                                                              | Repo                                                          | Resolution | Dataset               | Ref FP32 mAP | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :--------- | :---------------------------------------------------------------- | :------------------------------------------------------------ | :--------- | :-------------------- | -----------: | ------------: | -----------: | ----------: | ------------: |
| OSNet x1_0 | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torch/osnet-x1-0-market1501-onnx.yaml) | [&#x1F517;](https://github.com/KaiyangZhou/deep-person-reid)  | 256x128    | Market1501ReIdDataset | 82.55        | 0.93          | 1732         | 1770        | Apache-2.0    |
| SBS50      | [&#x1F517;](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/torch/sbs-s50-market1501-onnx.yaml)    | [&#x1F517;](https://github.com/JDAI-CV/fast-reid/tree/master) | 384x128    | Market1501ReIdDataset | 89.02        | -0.16         | 666          | 405         | Apache-2.0    |

### Large Language Models

For usage details see the [LLM Inference guide](../../user-guides/llm.md).

| Model | Max context (tokens) | Required PCIe card RAM |
|-------|---------------------:|----------------------:|
| [microsoft/Phi-3-mini-4k-instruct](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/llm/phi3-mini-512-static.yaml) | 512 | 4 GB |
| [microsoft/Phi-3-mini-4k-instruct](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/llm/phi3-mini-1024-4core-static.yaml) | 1024 | 16 GB |
| [microsoft/Phi-3-mini-4k-instruct](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/llm/phi3-mini-2048-4core-static.yaml) | 2048 | 16 GB |
| [meta-llama/Llama-3.2-1B-Instruct](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/llm/llama-3-2-1b-1024-4core-static.yaml) | 1024 | 4 GB |
| [meta-llama/Llama-3.2-3B-Instruct](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/llm/llama-3-2-3b-1024-4core-static.yaml) | 1024 | 4 GB |
| [meta-llama/Llama-3.1-8B-Instruct](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/llm/llama-3-1-8b-1024-4core-static.yaml) | 1024 | 16 GB |
| [Almawave/Velvet-2B](https://github.com/axelera-ai-hub/voyager-sdk/blob/release/v1.6/ax_models/zoo/llm/velvet-2b-1024-4core-static.yaml) | 1024 | 4 GB |

---

## Experimenting with optimized input shapes

Most models are trained on square inputs (640×640), but real-world video is often rectangular (16:9). Standard pipelines pad the input ("letterboxing"), forcing the model to process empty pixels.

By switching to a rectangular input shape that matches your video's aspect ratio, you can often achieve significant speedups with minimal accuracy impact. This is especially effective for fixed-camera applications like surveillance or traffic monitoring.

### How to test

Export models with dynamic input shapes, then compare:

```bash
# Standard 640×640
./inference.py yolox-m-coco-onnx dataset --pipe=torch-aipu --no-display

# Rectangular 640×480
./inference.py yolox-m-coco-onnx-rect dataset --pipe=torch-aipu --no-display
```

### Expected results

| Configuration | Input shape | Speedup | mAP impact | Best for |
|---------------|-------------|---------|------------|----------|
| Standard | 640×640 | Baseline | Baseline | General purpose, diverse content |
| Optimized | 640×480 | +24% | −0.3% | Near-square content, balanced performance |
| Optimized | 640×384 | +47% | −2.0% | Landscape video (16:9), maximum throughput |

---

## Custom weights

You can use your own trained weights with any model architecture. This involves updating the model's YAML configuration to point to your custom weight file. See [Deploy Custom Weights](../../tutorials/custom-weights.md) for the full walkthrough.

## See also

- [First Inference](../../user-guides/first-inference.md) — run your first model
- [Measure Accuracy](../../tutorials/measure-accuracy.md) — benchmark model performance
- [inference.py](../tools/inference-py.md) — full command reference
- [Glossary](../../glossary.md) — definitions of COCO, YOLO, mAP, and other terms

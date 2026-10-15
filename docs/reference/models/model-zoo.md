---
title: "Model Zoo"
---
# Model Zoo — Pre-trained Models

The Voyager Model Zoo is a collection of pre-trained AI models ready to run on Axelera hardware. When you run `inference.py yolov5s-v7-coco usb:0`, the model name (`yolov5s-v7-coco`) comes from the Model Zoo.

- [Model Zoo — Pre-trained Models](#model-zoo--pre-trained-models)
  - [Listing available models](#listing-available-models)
  - [How model names work](#how-model-names-work)
  - [Convolutional Neural Networks (CNNs)](#convolutional-neural-networks-cnns)
    - [Task types](#task-types)
    - [Running a CNN model](#running-a-cnn-model)
    - [Datasets](#datasets)
    - [Non-redistributable datasets](#non-redistributable-datasets)
    - [CNN Performance characteristics](#cnn-performance-characteristics)
    - [Image Classification](#image-classification)
    - [Object Detection](#object-detection)
    - [Semantic Segmentation](#semantic-segmentation)
    - [Instance Segmentation](#instance-segmentation)
    - [Keypoint Detection](#keypoint-detection)
    - [Depth Estimation](#depth-estimation)
    - [License Plate Recognition](#license-plate-recognition)
    - [Face Recognition](#face-recognition)
    - [Re-identification](#re-identification)
  - [Vision Transformers (ViTs)](#vision-transformers-vits)
    - [ViT Performance characteristics](#vit-performance-characteristics)
    - [ViT Reference Platform](#vit-reference-platform)
  - [Vision Language Models (VLMs)](#vision-language-models-vlms)
    - [VLM Performance characteristics](#vlm-performance-characteristics)
    - [VLM Reference platforms](#vlm-reference-platforms)
  - [Large Language Models (LLMs)](#large-language-models-llms)
    - [LLM Performance characteristics](#llm-performance-characteristics)
    - [LLM Reference platform](#llm-reference-platform)
  - [Experimenting with optimized input shapes](#experimenting-with-optimized-input-shapes)
    - [How to test](#how-to-test)
    - [Expected results](#expected-results)
  - [Custom weights](#custom-weights)
  - [See also](#see-also)
## Listing available models

From the SDK root directory (with environment activated):

```bash
make
```

This lists three categories:

| Category                            | What it contains                                                      |
| :---------------------------------- | :-------------------------------------------------------------------- |
| **ZOO**                             | Individual models — one model, one task                               |
| **REFERENCE APPLICATION PIPELINES** | Multi-model pipelines (e.g., detection cascaded into pose estimation) |
| **TUTORIALS**                       | Example models used by the tutorial documentation                     |

## How model names work

Model names follow a pattern:

```
<architecture>-<dataset>[-<variant>]
```

Examples:

| Name                | Architecture  | Dataset                        | Notes                |
| :------------------ | :------------ | :----------------------------- | :------------------- |
| `yolov5s-v7-coco`   | YOLOv5 small  | [COCO](../../glossary.md#coco) | v7 release of YOLOv5 |
| `yolov8s-coco-onnx` | YOLOv8 small  | COCO                           | ONNX format          |
| `resnet50-imagenet` | ResNet-50     | ImageNet                       | Classification model |

## Convolutional Neural Networks (CNNs)

### Task types

| Task                      | What it does                                            | Example model           |
| :------------------------ | :------------------------------------------------------ | :---------------------- |
| Object detection          | Finds and labels objects with bounding boxes            | `yolov5s-v7-coco`       |
| Classification            | Identifies what's in an image (single label)            | `resnet50-imagenet`     |
| Semantic segmentation     | Labels every pixel by category                          | `yolov8sseg-coco-onnx`  |
| Instance segmentation     | Labels every pixel AND distinguishes individual objects | `yolov8sseg-coco-onnx`  |
| Keypoint detection        | Finds body joints and pose landmarks                    | `yolov8lpose-coco-onnx` |
| Depth estimation          | Estimates distance of each pixel from camera            | `fastdepth-nyuv2`       |
| License plate recognition | Reads license plates                                    | Available in Model      |
| Face recognition          | Identifies or verifies faces                            | Available in Model Zoo  |

### Running a CNN model

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

### Datasets

Models are trained on specific datasets. The dataset name in the model identifier tells you what the model can recognize:

| Dataset                        | What it contains                                      | Typical use                    |
| :----------------------------- | :---------------------------------------------------- | :-------------                 |
| [COCO](../../glossary.md#coco) | 80 object categories (person, car, dog, etc.)         | General object detection       |
| ImageNet                       | 1000 image categories                                 | Image classification           |
| VOC                            | 20 object categories                                  | Object detection (smaller set) |

### Non-redistributable datasets

Most datasets download automatically when you first run a model. A few datasets require manual registration and download due to licensing restrictions. These must be downloaded by hand from the links below, then placed in the specified directory within your SDK installation. The SDK raises an error with the expected path if a required dataset is missing.

| Dataset                                                      | Archive                        | Download location |
| :----------------------------------------------------------- | :----------------------------- | :---------------- |
| [Cityscapes (val)](https://www.cityscapes-dataset.com/)      | `gtFine_val.zip`               | `data/cityscapes` |
| [Cityscapes (val)](https://www.cityscapes-dataset.com/)      | `leftImg8bit_val.zip`          | `data/cityscapes` |
| [Cityscapes (test)](https://www.cityscapes-dataset.com/)     | `gtFine_test.zip`              | `data/cityscapes` |
| [Cityscapes (test)](https://www.cityscapes-dataset.com/)     | `leftImg8bit_test.zip`         | `data/cityscapes` |
| [ImageNet (train)](https://www.image-net.org/download.php)   | `ILSVRC2012_devkit_t12.tar.gz` | `data/ImageNet`   |
| [ImageNet (train)](https://www.image-net.org/download.php)   | `ILSVRC2012_img_train.tar`     | `data/ImageNet`   |
| [ImageNet (val)](https://www.image-net.org/download.php)     | `ILSVRC2012_devkit_t12.tar.gz` | `data/ImageNet`   |
| [ImageNet (val)](https://www.image-net.org/download.php)     | `ILSVRC2012_img_val.tar`       | `data/ImageNet`   |
| WiderFace (train)                                            | `widerface_train.zip`          | `data/widerface`  |
| WiderFace (val)                                              | `widerface_val.zip`            | `data/widerface`  |

You are responsible for adhering to the terms and conditions of each dataset's license.

---

### CNN Performance characteristics

The tables below list all Model Zoo models for this SDK release. Columns:

- **Ref FP32** — accuracy of the original floating-point model
- **Accuracy loss** — FP32 accuracy minus quantized int8 accuracy (lower is better)
- **Ref PCIe FPS** — host throughput on Intel Core Ultra 7 265 + Axelera Edge 130p (1× PCIe) card
- **Ref M.2 FPS** — host throughput on Intel Core i5-1145G7E + Axelera Embedded 113m (1× M.2) card with the default power limit of 8.5W

Accuracy is measured using:

```bash
./inference.py <model> dataset --pipe=torch-aipu --no-display
```

Throughput is measured using a 720p h.264 video file:

```bash
./inference.py <model> media/traffic2_720p.mp4 --pipe=gst --no-display
```

### Image Classification
| Model                                                                                                  | ONNX                                                                                                | Repo                                                             | Resolution | Dataset     | Ref FP32 Top1 | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :----------------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------- | :--------- | :---------- | ------------: | ------------: | -----------: | ----------: | ------------: |
| [DenseNet-121](../../../ax_models/zoo/torchvision/classification/densenet121-imagenet.yaml)            | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/densenet121-imagenet-onnx.yaml)       | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 74.44         | 0.94          | 274          | 281         | BSD-3-Clause  |
| [EfficientNet-B0](../../../ax_models/zoo/torchvision/classification/efficientnet_b0-imagenet.yaml)     | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/efficientnet_b0-imagenet-onnx.yaml)   | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 77.67         | 0.82          | 1386         | 1375        | BSD-3-Clause  |
| [EfficientNet-B1](../../../ax_models/zoo/torchvision/classification/efficientnet_b1-imagenet.yaml)     | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/efficientnet_b1-imagenet-onnx.yaml)   | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 77.6          | 0.42          | 950          | 927         | BSD-3-Clause  |
| [EfficientNet-B2](../../../ax_models/zoo/torchvision/classification/efficientnet_b2-imagenet.yaml)     | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/efficientnet_b2-imagenet-onnx.yaml)   | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 77.79         | 0.43          | 883          | 862         | BSD-3-Clause  |
| [EfficientNet-B3](../../../ax_models/zoo/torchvision/classification/efficientnet_b3-imagenet.yaml)     | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/efficientnet_b3-imagenet-onnx.yaml)   | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 78.54         | 0.38          | 776          | 760         | BSD-3-Clause  |
| [EfficientNet-B4](../../../ax_models/zoo/torchvision/classification/efficientnet_b4-imagenet.yaml)     | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/efficientnet_b4-imagenet-onnx.yaml)   | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 79.27         | 0.70          | 560          | 565         | BSD-3-Clause  |
| [MobileNetV2](../../../ax_models/zoo/torchvision/classification/mobilenetv2-imagenet.yaml)             | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/mobilenetv2-imagenet-onnx.yaml)       | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 71.87         | 1.42          | 3471         | 3305        | BSD-3-Clause  |
| [MobileNetV3-large](../../../ax_models/zoo/torchvision/classification/mobilenetv3_large-imagenet.yaml) | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/mobilenetv3_large-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 74.05         | 0.97          | 2665         | 2724        | BSD-3-Clause  |
| [MobileNetV3-small](../../../ax_models/zoo/torchvision/classification/mobilenetv3_small-imagenet.yaml) | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/mobilenetv3_small-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 67.67         | 2.04          | 4138         | 3928        | BSD-3-Clause  |
| [MobileNetV4-small](../../../ax_models/zoo/timm/mobilenetv4_small-imagenet.yaml)                       | [&#x1F517;](../../../ax_models/zoo/timm/mobilenetv4_small-imagenet-onnx.yaml)                       | [&#x1F517;](https://github.com/huggingface/pytorch-image-models) | 224x224    | ImageNet-1K | 73.74         | 1.20          | 4591         | 4257        | Apache 2.0    |
| [MobileNetV4-medium](../../../ax_models/zoo/timm/mobilenetv4_medium-imagenet.yaml)                     | [&#x1F517;](../../../ax_models/zoo/timm/mobilenetv4_medium-imagenet-onnx.yaml)                      | [&#x1F517;](https://github.com/huggingface/pytorch-image-models) | 224x224    | ImageNet-1K | 79.04         | 1.37          | 2393         | 2442        | Apache 2.0    |
| [MobileNetV4-large](../../../ax_models/zoo/timm/mobilenetv4_large-imagenet.yaml)                       | [&#x1F517;](../../../ax_models/zoo/timm/mobilenetv4_large-imagenet-onnx.yaml)                       | [&#x1F517;](https://github.com/huggingface/pytorch-image-models) | 384x384    | ImageNet-1K | 82.92         | 0.92          | 736          | 765         | Apache 2.0    |
| [MobileNetV4-aa_large](../../../ax_models/zoo/timm/mobilenetv4_aa_large-imagenet.yaml)                 | [&#x1F517;](../../../ax_models/zoo/timm/mobilenetv4_aa_large-imagenet-onnx.yaml)                    | [&#x1F517;](https://github.com/huggingface/pytorch-image-models) | 384x384    | ImageNet-1K | 83.22         | 0.98          | 611          | 639         | Apache 2.0    |
| [SqueezeNet 1.0](../../../ax_models/zoo/torchvision/classification/squeezenet1.0-imagenet.yaml)        | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/squeezenet1.0-imagenet-onnx.yaml)     | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 58.1          | 3.38          | 920          | 886         | BSD-3-Clause  |
| [SqueezeNet 1.1](../../../ax_models/zoo/torchvision/classification/squeezenet1.1-imagenet.yaml)        | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/squeezenet1.1-imagenet-onnx.yaml)     | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 58.19         | 1.85          | 6477         | 5881        | BSD-3-Clause  |
| [Inception V3](../../../ax_models/zoo/torchvision/classification/inception_v3-imagenet.yaml)           | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/inception_v3-imagenet-onnx.yaml)      | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 69.85         | 0.21          | 1073         | 1112        | BSD-3-Clause  |
| [RegNetX-1_6GF](../../../ax_models/zoo/torchvision/classification/regnet_x_1_6gf-imagenet.yaml)        | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/regnet_x_1_6gf-imagenet-onnx.yaml)    | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 79.33         | 0.13          | 668          | 700         | BSD-3-Clause  |
| [RegNetX-400MF](../../../ax_models/zoo/torchvision/classification/regnet_x_400mf-imagenet.yaml)        | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/regnet_x_400mf-imagenet-onnx.yaml)    | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 74.48         | 0.30          | 1128         | 1182        | BSD-3-Clause  |
| [RegNetY-1_6GF](../../../ax_models/zoo/torchvision/classification/regnet_y_1_6gf-imagenet.yaml)        | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/regnet_y_1_6gf-imagenet-onnx.yaml)    | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 80.73         | 0.09          | 577          | 595         | BSD-3-Clause  |
| [RegNetY-400MF](../../../ax_models/zoo/torchvision/classification/regnet_y_400mf-imagenet.yaml)        | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/regnet_y_400mf-imagenet-onnx.yaml)    | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 75.63         | 0.17          | 1533         | 1577        | BSD-3-Clause  |
| [ResNet-18](../../../ax_models/zoo/torchvision/classification/resnet18-imagenet.yaml)                  | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/resnet18-imagenet-onnx.yaml)          | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 69.76         | 0.75          | 3598         | 3374        | BSD-3-Clause  |
| [ResNet-34](../../../ax_models/zoo/torchvision/classification/resnet34-imagenet.yaml)                  | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/resnet34-imagenet-onnx.yaml)          | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 73.3          | 0.24          | 2109         | 2151        | BSD-3-Clause  |
| [ResNet-50 v1.5](../../../ax_models/zoo/torchvision/classification/resnet50-imagenet.yaml)             | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/resnet50-imagenet-onnx.yaml)          | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 76.15         | 0.26          | 1825         | 1849        | BSD-3-Clause  |
| [ResNet-101](../../../ax_models/zoo/torchvision/classification/resnet101-imagenet.yaml)                | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/resnet101-imagenet-onnx.yaml)         | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 77.37         | 0.38          | 986          | 1007        | BSD-3-Clause  |
| [ResNet-152](../../../ax_models/zoo/torchvision/classification/resnet152-imagenet.yaml)                | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/resnet152-imagenet-onnx.yaml)         | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 78.31         | 0.37          | 468          | 495         | BSD-3-Clause  |
| [ResNet-10t](../../../ax_models/zoo/timm/resnet10t-imagenet.yaml)                                      | [&#x1F517;](../../../ax_models/zoo/timm/resnet10t-imagenet-onnx.yaml)                               | [&#x1F517;](https://huggingface.co/timm/resnet10t.c3_in1k)       | 224x224    | ImageNet-1K | 68.22         | 1.13          | 4740         | 4412        | Apache 2.0    |
| [ResNeXt50_32x4d](../../../ax_models/zoo/torchvision/classification/resnext50_32x4d-imagenet.yaml)     | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/resnext50_32x4d-imagenet-onnx.yaml)   | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 77.61         | 0.11          | 421          | 441         | BSD-3-Clause  |
| [Wide ResNet-50](../../../ax_models/zoo/torchvision/classification/wide_resnet50-imagenet.yaml)        | [&#x1F517;](../../../ax_models/zoo/torchvision/classification/wide_resnet50-imagenet-onnx.yaml)     | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 78.48         | 0.21          | 422          | 442         | BSD-3-Clause  |

### Object Detection
| Model                                                                                                             | ONNX                                                                                               | Repo                                                                                                        | Resolution | Dataset                   | Ref FP32 mAP | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :---------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------- | :--------- | :------------------------ | -----------: | ------------: | -----------: | ----------: | ------------: |
| [GELAN-s](https://github.com/WongKinYiu/yolov9)                                                                   | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/gelan-s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/WongKinYiu/yolov9)                                                           | 640x640    | COCO2017                  | 46.41        | 2.54          | 369          | 295         | GPL-3.0       |
| [GELAN-m](https://github.com/WongKinYiu/yolov9)                                                                   | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/gelan-m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/WongKinYiu/yolov9)                                                           | 640x640    | COCO2017                  | 50.86        | 1.09          | 200          | 144         | GPL-3.0       |
| [GELAN-c](https://github.com/WongKinYiu/yolov9)                                                                   | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/gelan-c-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/WongKinYiu/yolov9)                                                           | 640x640    | COCO2017                  | 52.3         | 0.64          | 183          | 130         | GPL-3.0       |
| [RetinaFace - Resnet50](https://github.com/biubug6/Pytorch_Retinaface/tree/master)                                | [&#x1F517;](../../../ax_models/zoo/torch/retinaface-resnet50-widerface-onnx.yaml)                  | [&#x1F517;](https://github.com/biubug6/Pytorch_Retinaface/tree/master)                                      | 840x840    | WiderFace                 | 95.25        | 0.19          | 89           | 91          | MIT           |
| [RetinaFace - mb0.25](https://github.com/biubug6/Pytorch_Retinaface/tree/master)                                  | [&#x1F517;](../../../ax_models/zoo/torch/retinaface-mobilenet0.25-widerface-onnx.yaml)             | [&#x1F517;](https://github.com/biubug6/Pytorch_Retinaface/tree/master)                                      | 640x640    | WiderFace                 | 89.44        | 1.27          | 996          | 962         | MIT           |
| [SSD-MobileNetV1](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) | [&#x1F517;](../../../ax_models/zoo/tensorflow/object_detection/ssd-mobilenetv1-coco-poc-onnx.yaml) | [&#x1F517;](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) | 300x300    | COCO2017                  | 24.77        | -0.09         | 2894         | 3004        | Apache 2.0    |
| [SSD-MobileNetV2](https://github.com/tensorflow/models)                                                           | [&#x1F517;](../../../ax_models/zoo/tensorflow/object_detection/ssd-mobilenetv2-coco-poc-onnx.yaml) | [&#x1F517;](https://github.com/tensorflow/models)                                                           | 300x300    | COCO2017                  | 19.25        | 0.58          | 2030         | 2017        | Apache 2.0    |
| [YOLOv3](https://github.com/ultralytics/yolov3)                                                                   | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolov3-coco-onnx.yaml)                    | [&#x1F517;](https://github.com/ultralytics/yolov3)                                                          | 640x640    | COCO2017                  | 46.61        | 0.80          | 157          | 98          | AGPL-3.0      |
| [YOLOv4](../../../ax_models/zoo/yolo/object_detection/yolov4-416-coco.yaml)                                       |                                                                                                    | [&#x1F517;](https://github.com/AlexeyAB/darknet)                                                            | 416x416    | COCO2017                  | 21.75        | -0.12         | 273          | 192         | Unlicense     |
| [YOLOv4-CSP-Leaky](../../../ax_models/zoo/yolo/object_detection/yolov4-csp-leaky-coco.yaml)                       |                                                                                                    | [&#x1F517;](https://github.com/WongKinYiu/CrossStagePartialNetworks)                                        | 640x640    | COCO2017                  | 26.49        | 1.42          | 162          | 98          | GPL-3.0       |
| [YOLOv5s-Relu](../../../ax_models/zoo/yolo/object_detection/yolov5s-relu-coco.yaml)                               | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolov5s-relu-coco-onnx.yaml)              | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017                  | 35.09        | 0.47          | 761          | 619         | AGPL-3.0      |
| [YOLOv5s-v5](../../../ax_models/zoo/yolo/object_detection/yolov5s-v5-coco.yaml)                                   | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolov5s-v5-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017                  | 36.18        | 0.39          | 764          | 526         | AGPL-3.0      |
| [YOLOv5n](../../../ax_models/zoo/yolo/object_detection/yolov5n-v7-coco.yaml)                                      | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolov5n-v7-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017                  | 27.72        | 0.82          | 1000         | 824         | AGPL-3.0      |
| [YOLOv5s](../../../ax_models/zoo/yolo/object_detection/yolov5s-v7-coco.yaml)                                      | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolov5s-v7-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017                  | 37.25        | 0.72          | 812          | 645         | AGPL-3.0      |
| [YOLOv5m](../../../ax_models/zoo/yolo/object_detection/yolov5m-v7-coco.yaml)                                      | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolov5m-v7-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017                  | 44.94        | 0.76          | 447          | 240         | AGPL-3.0      |
| [YOLOv5l](../../../ax_models/zoo/yolo/object_detection/yolov5l-v7-coco.yaml)                                      | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolov5l-v7-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017                  | 48.67        | 0.93          | 292          | 139         | AGPL-3.0      |
| [YOLOv7](../../../ax_models/zoo/yolo/object_detection/yolov7-coco.yaml)                                           | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolov7-coco-onnx.yaml)                    | [&#x1F517;](https://github.com/WongKinYiu/yolov7)                                                           | 640x640    | COCO2017                  | 51.02        | 0.56          | 210          | 134         | GPL-3.0       |
| [YOLOv7-tiny](../../../ax_models/zoo/yolo/object_detection/yolov7-tiny-coco.yaml)                                 | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolov7-tiny-coco-onnx.yaml)               | [&#x1F517;](https://github.com/WongKinYiu/yolov7)                                                           | 416x416    | COCO2017                  | 33.12        | 0.50          | 1374         | 1337        | GPL-3.0       |
| [YOLOv7 640x480](../../../ax_models/zoo/yolo/object_detection/yolov7-640x480-coco.yaml)                           | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolov7-640x480-coco-onnx.yaml)            | [&#x1F517;](https://github.com/WongKinYiu/yolov7)                                                           | 640x480    | COCO2017                  | 50.78        | 0.56          | 242          | 171         | GPL-3.0       |
| [YOLOv8n](../../../ax_models/zoo/yolo/object_detection/yolov8n-coco.yaml)                                         | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolov8n-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 37.12        | 0.77          | 783          | 774         | AGPL-3.0      |
| [YOLOv8s](../../../ax_models/zoo/yolo/object_detection/yolov8s-coco.yaml)                                         | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolov8s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 44.8         | 0.84          | 594          | 442         | AGPL-3.0      |
| [YOLOv8m](../../../ax_models/zoo/yolo/object_detection/yolov8m-coco.yaml)                                         | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolov8m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 50.16        | 1.36          | 232          | 145         | AGPL-3.0      |
| [YOLOv8l](../../../ax_models/zoo/yolo/object_detection/yolov8l-coco.yaml)                                         | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolov8l-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 52.83        | 1.58          | 174          | 96          | AGPL-3.0      |
| [YOLOv8n-obb](https://github.com/ultralytics/ultralytics)                                                         | [&#x1F517;](../../../ax_models/zoo/yolo/obb_detection/yolov8n-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset | 48.73        | 3.82          | 261          | 220         | AGPL-3.0      |
| [YOLOv8l-obb](https://github.com/ultralytics/ultralytics)                                                         | [&#x1F517;](../../../ax_models/zoo/yolo/obb_detection/yolov8l-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset | 56.06        | 4.10          | 34           | 29          | AGPL-3.0      |
| [YOLOX-s](https://github.com/Megvii-BaseDetection/YOLOX)                                                          | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolox-s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/Megvii-BaseDetection/YOLOX)                                                  | 640x640    | COCO2017                  | 40.38        | 0.41          | 634          | 384         | Apache-2.0    |
| [YOLOX-m](https://github.com/Megvii-BaseDetection/YOLOX)                                                          | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolox-m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/Megvii-BaseDetection/YOLOX)                                                  | 640x640    | COCO2017                  | 46.92        | 0.42          | 342          | 182         | Apache-2.0    |
| [YOLOX-x Human](https://github.com/FoundationVision/ByteTrack)                                                    | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolox-x-crowdhuman-onnx.yaml)             | [&#x1F517;](https://github.com/FoundationVision/ByteTrack)                                                  | 1440x800   | COCO2017                  | 57.66        | 3.50          | 22           | 17          | MIT           |
| [YOLOv9t](https://github.com/ultralytics/ultralytics)                                                             | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolov9t-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 37.81        | 0.85          | 402          | 399         | AGPL-3.0      |
| [YOLOv9s](https://github.com/ultralytics/ultralytics)                                                             | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolov9s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 46.28        | 1.14          | 369          | 272         | AGPL-3.0      |
| [YOLOv9m](https://github.com/ultralytics/ultralytics)                                                             | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolov9m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 51.24        | 1.18          | 199          | 139         | AGPL-3.0      |
| [YOLOv9c](https://github.com/ultralytics/ultralytics)                                                             | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolov9c-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 52.67        | 1.78          | 198          | 132         | AGPL-3.0      |
| [YOLOv10n](https://github.com/ultralytics/ultralytics)                                                            | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolov10n-coco-onnx.yaml)                  | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 38.08        | 0.74          | 727          | 703         | AGPL-3.0      |
| [YOLOv10s](https://github.com/ultralytics/ultralytics)                                                            | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolov10s-coco-onnx.yaml)                  | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 45.74        | 0.39          | 568          | 463         | AGPL-3.0      |
| [YOLOv10b](https://github.com/ultralytics/ultralytics)                                                            | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolov10b-coco-onnx.yaml)                  | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 51.79        | 0.33          | 253          | 165         | AGPL-3.0      |
| [YOLO11n](https://github.com/ultralytics/ultralytics)                                                             | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolo11n-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 39.17        | 0.67          | 744          | 724         | AGPL-3.0      |
| [YOLO11s](https://github.com/ultralytics/ultralytics)                                                             | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolo11s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 46.54        | 0.52          | 554          | 446         | AGPL-3.0      |
| [YOLO11m](https://github.com/ultralytics/ultralytics)                                                             | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolo11m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 51.31        | 0.45          | 264          | 190         | AGPL-3.0      |
| [YOLO11l](https://github.com/ultralytics/ultralytics)                                                             | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolo11l-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 53.23        | 0.38          | 180          | 137         | AGPL-3.0      |
| [YOLO11x](https://github.com/ultralytics/ultralytics)                                                             | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolo11x-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 54.67        | 0.42          | 52           | 53          | AGPL-3.0      |
| [YOLO11n-obb](https://github.com/ultralytics/ultralytics)                                                         | [&#x1F517;](../../../ax_models/zoo/yolo/obb_detection/yolo11n-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset | 50.01        | 1.58          | 248          | 252         | AGPL-3.0      |
| [YOLO11l-obb](https://github.com/ultralytics/ultralytics)                                                         | [&#x1F517;](../../../ax_models/zoo/yolo/obb_detection/yolo11l-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset | 56.41        | 1.42          | 33           | 35          | AGPL-3.0      |
| [YOLO26n](https://github.com/ultralytics/ultralytics)                                                             | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolo26n-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 40.18        | 1.89          | 650          | 628         | AGPL-3.0      |
| [YOLO26s](https://github.com/ultralytics/ultralytics)                                                             | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolo26s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 47.66        | 1.82          | 495          | 447         | AGPL-3.0      |
| [YOLO26m](https://github.com/ultralytics/ultralytics)                                                             | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolo26m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 52.45        | 2.08          | 254          | 191         | AGPL-3.0      |
| [YOLO26l](https://github.com/ultralytics/ultralytics)                                                             | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolo26l-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 54.11        | 2.08          | 175          | 137         | AGPL-3.0      |
| [YOLO26x](https://github.com/ultralytics/ultralytics)                                                             | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolo26x-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 56.92        | 2.19          | 54           | 54          | AGPL-3.0      |
| [YOLO26n-obb](https://github.com/ultralytics/ultralytics)                                                         | [&#x1F517;](../../../ax_models/zoo/yolo/obb_detection/yolo26n-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset | 49.41        | 3.19          | 202          | 208         | AGPL-3.0      |
| [YOLO26s-obb](https://github.com/ultralytics/ultralytics)                                                         | [&#x1F517;](../../../ax_models/zoo/yolo/obb_detection/yolo26s-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset | 54.02        | 2.01          | 166          | 159         | AGPL-3.0      |
| [YOLO26m-obb](https://github.com/ultralytics/ultralytics)                                                         | [&#x1F517;](../../../ax_models/zoo/yolo/obb_detection/yolo26m-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset | 56.66        | 1.67          | 61           | 62          | AGPL-3.0      |
| [YOLO26l-obb](https://github.com/ultralytics/ultralytics)                                                         | [&#x1F517;](../../../ax_models/zoo/yolo/obb_detection/yolo26l-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset | 57.35        | 1.24          | 33           | 35          | AGPL-3.0      |
| [YOLO26x-obb](https://github.com/ultralytics/ultralytics)                                                         | [&#x1F517;](../../../ax_models/zoo/yolo/obb_detection/yolo26x-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset | 58.4         | 5.34          | 15           | -           | AGPL-3.0      |
| [YOLO-NAS S](https://github.com/Deci-AI/super-gradients)                                                          | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolonas-s-coco-onnx.yaml)                 | [&#x1F517;](https://github.com/Deci-AI/super-gradients)                                                     | 640x640    | COCO2017                  | 47.06        | 0.21          | 445          | 376         | Apache-2.0    |
| [YOLO-NAS M](https://github.com/Deci-AI/super-gradients)                                                          | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolonas-m-coco-onnx.yaml)                 | [&#x1F517;](https://github.com/Deci-AI/super-gradients)                                                     | 640x640    | COCO2017                  | 51           | 0.27          | 285          | 203         | Apache-2.0    |
| [YOLO-NAS L](https://github.com/Deci-AI/super-gradients)                                                          | [&#x1F517;](../../../ax_models/zoo/yolo/object_detection/yolonas-l-coco-onnx.yaml)                 | [&#x1F517;](https://github.com/Deci-AI/super-gradients)                                                     | 640x640    | COCO2017                  | 51.79        | 0.38          | 157          | 128         | Apache-2.0    |

### Semantic Segmentation
| Model                                                                                | ONNX                                                                                           | Repo                                                                             | Resolution | Dataset    | Ref FP32 mIoU | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :----------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------- | :--------- | :--------- | ------------: | ------------: | -----------: | ----------: | ------------: |
| [U-Net FCN 256](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/unet) | [&#x1F517;](../../../ax_models/zoo/mmlab/mmseg/unet_fcn_256-cityscapes-onnx.yaml)              | [&#x1F517;](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/unet) | 256x256    | Cityscapes | 57.75         | 0.21          | 242          | 206         | Apache 2.0    |
| [U-Net FCN 512](../../../ax_models/zoo/mmlab/mmseg/unet_fcn_512-cityscapes.yaml)     |                                                                                                | [&#x1F517;](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/unet) | 512x512    | Cityscapes | 66.62         | 0.20          | 32           | 34          | Apache 2.0    |
| [YOLO26n-sem](https://github.com/ultralytics/ultralytics)                            | [&#x1F517;](../../../ax_models/zoo/yolo/semantic_segmentation/yolo26nsem-cityscapes-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics)                          | 1024x1024  | Cityscapes | 71.42         | 1.18          | 280          | 294         | AGPL-3.0      |
| [YOLO26s-sem](https://github.com/ultralytics/ultralytics)                            | [&#x1F517;](../../../ax_models/zoo/yolo/semantic_segmentation/yolo26ssem-cityscapes-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics)                          | 1024x1024  | Cityscapes | 76.5          | 0.36          | 198          | 189         | AGPL-3.0      |
| [YOLO26m-sem](https://github.com/ultralytics/ultralytics)                            | [&#x1F517;](../../../ax_models/zoo/yolo/semantic_segmentation/yolo26msem-cityscapes-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics)                          | 1024x1024  | Cityscapes | 79.23         | 0.34          | 76           | 76          | AGPL-3.0      |
| [YOLO26l-sem](https://github.com/ultralytics/ultralytics)                            | [&#x1F517;](../../../ax_models/zoo/yolo/semantic_segmentation/yolo26lsem-cityscapes-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics)                          | 1024x1024  | Cityscapes | 79.73         | 0.02          | 52           | 55          | AGPL-3.0      |
| [YOLO26x-sem](https://github.com/ultralytics/ultralytics)                            | [&#x1F517;](../../../ax_models/zoo/yolo/semantic_segmentation/yolo26xsem-cityscapes-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics)                          | 1024x1024  | Cityscapes | 80.76         | 0.38          | 19           | 19          | AGPL-3.0      |

### Instance Segmentation
| Model                                                                                 | ONNX                                                                                     | Repo                                                    | Resolution | Dataset  | Ref FP32 mAP | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :------------------------------------------------------------------------------------ | :--------------------------------------------------------------------------------------- | :------------------------------------------------------ | :--------- | :------- | -----------: | ------------: | -----------: | ----------: | ------------: |
| [YOLACT-ResNet50](https://github.com/dbolya/yolact)                                   | [&#x1F517;](../../../ax_models/zoo/torch/yolact-resnet50-coco-onnx.yaml)                 | [&#x1F517;](https://github.com/dbolya/yolact)           | 550x550    | COCO2017 | 28.39        | 0.53          | 287          | 186         | MIT           |
| [YOLOv8n-seg](../../../ax_models/zoo/yolo/instance_segmentation/yolov8nseg-coco.yaml) | [&#x1F517;](../../../ax_models/zoo/yolo/instance_segmentation/yolov8nseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 29.95        | 0.43          | 487          | 520         | AGPL-3.0      |
| [YOLOv8s-seg](../../../ax_models/zoo/yolo/instance_segmentation/yolov8sseg-coco.yaml) | [&#x1F517;](../../../ax_models/zoo/yolo/instance_segmentation/yolov8sseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 36.3         | 0.56          | 442          | 294         | AGPL-3.0      |
| [YOLOv8m-seg](../../../ax_models/zoo/yolo/instance_segmentation/yolov8mseg-coco.yaml) | [&#x1F517;](../../../ax_models/zoo/yolo/instance_segmentation/yolov8mseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 40.36        | 0.31          | 197          | 115         | AGPL-3.0      |
| [YOLOv8l-seg](../../../ax_models/zoo/yolo/instance_segmentation/yolov8lseg-coco.yaml) | [&#x1F517;](../../../ax_models/zoo/yolo/instance_segmentation/yolov8lseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 42.24        | 0.62          | 155          | 73          | AGPL-3.0      |
| [YOLO11n-seg](https://github.com/ultralytics/ultralytics)                             | [&#x1F517;](../../../ax_models/zoo/yolo/instance_segmentation/yolo11nseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 31.82        | 0.95          | 574          | 536         | AGPL-3.0      |
| [YOLO11l-seg](https://github.com/ultralytics/ultralytics)                             | [&#x1F517;](../../../ax_models/zoo/yolo/instance_segmentation/yolo11lseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 43.23        | 0.14          | 153          | 101         | AGPL-3.0      |
| [YOLO26n-seg](https://github.com/ultralytics/ultralytics)                             | [&#x1F517;](../../../ax_models/zoo/yolo/instance_segmentation/yolo26nseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 32.92        | 2.63          | 507          | 494         | AGPL-3.0      |
| [YOLO26s-seg](https://github.com/ultralytics/ultralytics)                             | [&#x1F517;](../../../ax_models/zoo/yolo/instance_segmentation/yolo26sseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 39.26        | 2.49          | 383          | 289         | AGPL-3.0      |
| [YOLO26m-seg](https://github.com/ultralytics/ultralytics)                             | [&#x1F517;](../../../ax_models/zoo/yolo/instance_segmentation/yolo26mseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 43.31        | 1.13          | 196          | 118         | AGPL-3.0      |
| [YOLO26l-seg](https://github.com/ultralytics/ultralytics)                             | [&#x1F517;](../../../ax_models/zoo/yolo/instance_segmentation/yolo26lseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 45.05        | 1.47          | 146          | 96          | AGPL-3.0      |
| [YOLO26x-seg](https://github.com/ultralytics/ultralytics)                             | [&#x1F517;](../../../ax_models/zoo/yolo/instance_segmentation/yolo26xseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 46.51        | 1.43          | 45           | 45          | AGPL-3.0      |

### Keypoint Detection
| Model                                                                                | ONNX                                                                                   | Repo                                                    | Resolution | Dataset  | Ref FP32 mAP | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :----------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------- | :------------------------------------------------------ | :--------- | :------- | -----------: | ------------: | -----------: | ----------: | ------------: |
| [YOLOv8n-pose](../../../ax_models/zoo/yolo/keypoint_detection/yolov8npose-coco.yaml) | [&#x1F517;](../../../ax_models/zoo/yolo/keypoint_detection/yolov8npose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 51.11        | 1.96          | 762          | 746         | AGPL-3.0      |
| [YOLOv8s-pose](../../../ax_models/zoo/yolo/keypoint_detection/yolov8spose-coco.yaml) | [&#x1F517;](../../../ax_models/zoo/yolo/keypoint_detection/yolov8spose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 60.65        | 2.77          | 563          | 423         | AGPL-3.0      |
| [YOLOv8m-pose](../../../ax_models/zoo/yolo/keypoint_detection/yolov8mpose-coco.yaml) | [&#x1F517;](../../../ax_models/zoo/yolo/keypoint_detection/yolov8mpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 65.58        | 2.13          | 225          | 140         | AGPL-3.0      |
| [YOLOv8l-pose](../../../ax_models/zoo/yolo/keypoint_detection/yolov8lpose-coco.yaml) | [&#x1F517;](../../../ax_models/zoo/yolo/keypoint_detection/yolov8lpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 68.39        | 1.45          | 177          | 90          | AGPL-3.0      |
| [YOLO11n-pose](https://github.com/ultralytics/ultralytics)                           | [&#x1F517;](../../../ax_models/zoo/yolo/keypoint_detection/yolo11npose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 51.15        | 3.25          | 742          | 718         | AGPL-3.0      |
| [YOLO11l-pose](https://github.com/ultralytics/ultralytics)                           | [&#x1F517;](../../../ax_models/zoo/yolo/keypoint_detection/yolo11lpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 67.44        | 3.12          | 177          | 138         | AGPL-3.0      |
| [YOLO26n-pose](https://github.com/ultralytics/ultralytics)                           | [&#x1F517;](../../../ax_models/zoo/yolo/keypoint_detection/yolo26npose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 57.66        | 6.33          | 640          | 618         | AGPL-3.0      |
| [YOLO26s-pose](https://github.com/ultralytics/ultralytics)                           | [&#x1F517;](../../../ax_models/zoo/yolo/keypoint_detection/yolo26spose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 63.61        | 5.11          | 461          | 400         | AGPL-3.0      |
| [YOLO26m-pose](https://github.com/ultralytics/ultralytics)                           | [&#x1F517;](../../../ax_models/zoo/yolo/keypoint_detection/yolo26mpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 69.54        | 3.89          | 244          | 181         | AGPL-3.0      |
| [YOLO26l-pose](https://github.com/ultralytics/ultralytics)                           | [&#x1F517;](../../../ax_models/zoo/yolo/keypoint_detection/yolo26lpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 71.05        | 2.88          | 173          | 133         | AGPL-3.0      |
| [YOLO26x-pose](https://github.com/ultralytics/ultralytics)                           | [&#x1F517;](../../../ax_models/zoo/yolo/keypoint_detection/yolo26xpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 72.75        | 8.19          | 52           | 52          | AGPL-3.0      |

### Depth Estimation
| Model                                                                             | ONNX                                                                                         | Repo                                                                              | Resolution | Dataset    | Ref FP32 RMSE      | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :-------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------- | :-------------------------------------------------------------------------------- | :--------- | :--------- | -----------------: | ------------: | -----------: | ----------: | ------------: |
| [FastDepth](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/146_FastDepth) | [&#x1F517;](../../../ax_models/zoo/torch/fastdepth-nyudepthv2-onnx.yaml)                     | [&#x1F517;](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/146_FastDepth) | 224x224    | NYUDepthV2 | 0.6574181318283081 | -0.0045       | 946          | 942         | MIT           |
| [YOLO26n-Depth](https://github.com/ultralytics/ultralytics)                       | [&#x1F517;](../../../ax_models/zoo/yolo/depth_estimation/yolo26n-depth-nyudepthv2-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics)                           | 768x768    | NYUDepthV2 | 0.5763             | -0.0051       | 90           | 96          | AGPL-3.0      |
| [YOLO26s-Depth](https://github.com/ultralytics/ultralytics)                       | [&#x1F517;](../../../ax_models/zoo/yolo/depth_estimation/yolo26s-depth-nyudepthv2-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics)                           | 768x768    | NYUDepthV2 | 0.5681             | -0.0310       | 71           | 77          | AGPL-3.0      |
| [YOLO26m-Depth](https://github.com/ultralytics/ultralytics)                       | [&#x1F517;](../../../ax_models/zoo/yolo/depth_estimation/yolo26m-depth-nyudepthv2-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics)                           | 768x768    | NYUDepthV2 | 0.544              | -0.0311       | 44           | 46          | AGPL-3.0      |
| [YOLO26l-Depth](https://github.com/ultralytics/ultralytics)                       | [&#x1F517;](../../../ax_models/zoo/yolo/depth_estimation/yolo26l-depth-nyudepthv2-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics)                           | 768x768    | NYUDepthV2 | 0.5353             | -0.0219       | 33           | 36          | AGPL-3.0      |
| [YOLO26x-Depth](https://github.com/ultralytics/ultralytics)                       | [&#x1F517;](../../../ax_models/zoo/yolo/depth_estimation/yolo26x-depth-nyudepthv2-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics)                           | 768x768    | NYUDepthV2 | 0.5244             |               | 19           | 20          | AGPL-3.0      |

### License Plate Recognition
| Model                                              | ONNX | Repo                                                     | Resolution | Dataset       | Ref FP32 WLA | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :------------------------------------------------- | :--- | :------------------------------------------------------- | :--------- | :------------ | -----------: | ------------: | -----------: | ----------: | ------------: |
| [LPRNet](../../../ax_models/zoo/torch/lprnet.yaml) |      | [&#x1F517;](https://github.com/sirius-ai/LPRNet_Pytorch) | 94x24      | LPRNetDataset | 89.4         | 1.90          | 7943         | 7750        | Apache-2.0    |

### Face Recognition
| Model                                                                        | ONNX                                                            | Repo                                                     | Resolution | Dataset            | Ref FP32 top1_avg | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :--------------------------------------------------------------------------- | :-------------------------------------------------------------- | :------------------------------------------------------- | :--------- | :----------------- | ----------------: | ------------: | -----------: | ----------: | ------------: |
| [FaceNet - InceptionResnetV1](../../../ax_models/zoo/torch/facenet-lfw.yaml) | [&#x1F517;](../../../ax_models/zoo/torch/facenet-lfw-onnx.yaml) | [&#x1F517;](https://github.com/timesler/facenet-pytorch) | 160x160    | LFWTorchvisionPair | 98.35             | 0.02          | 1226         | 1285        | MIT           |

### Re-identification
| Model                                                         | ONNX                                                                      | Repo                                                          | Resolution | Dataset               | Ref FP32 mAP | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :------------------------------------------------------------ | :------------------------------------------------------------------------ | :------------------------------------------------------------ | :--------- | :-------------------- | -----------: | ------------: | -----------: | ----------: | ------------: |
| [OSNet x1_0](https://github.com/KaiyangZhou/deep-person-reid) | [&#x1F517;](../../../ax_models/zoo/torch/osnet-x1-0-market1501-onnx.yaml) | [&#x1F517;](https://github.com/KaiyangZhou/deep-person-reid)  | 256x128    | Market1501ReIdDataset | 82.55        | 0.78          | 1686         | 1673        | Apache-2.0    |
| [SBS50](https://github.com/JDAI-CV/fast-reid/tree/master)     | [&#x1F517;](../../../ax_models/zoo/torch/sbs-s50-market1501-onnx.yaml)    | [&#x1F517;](https://github.com/JDAI-CV/fast-reid/tree/master) | 384x128    | Market1501ReIdDataset | 89.02        | -0.19         | 652          | 409         | Apache-2.0    |

## Vision Transformers (ViTs)

### ViT Performance characteristics
- **Resolution** — input resolution of the model.
- **benchmark ms / fps** — latency per frame and throughput.
- **License** — model license.
  
Latency and throughput are measured with:
 
```bash
python -m model_axscript.vision benchmark <model> <image>
```

### ViT Reference Platform
- **PCIe** — Intel Core i7-13700 32GB RAM running Ubuntu 24.04.5 with 1x Axelera Edge 130p PCIe card.


| Model                                                                                                                                       | Resolution | benchmark ms / fps | License    |
| :-------------------------------------------------------------------------------------------------------------------------------------------| ---------: | -----------------: | ---------: |
| [depth_anything_v2_small](../../../axelera_zoo/python/axelera/zoo/configs/depth_estimation/depth_anything_v2_small.py)                      | 224x224    | 42.91 / 23.3       | Apache-2.0 |
| [dinov2_depth_vits14_nyu_linear](../../../axelera_zoo/python/axelera/zoo/configs/depth_estimation/dinov2_depth_vits14_nyu_linear.py)        | 224x224    | 48.25 / 20.7       | Apache-2.0 |
| [dinov2_seg_vitl14_ade20k_linear](../../../axelera_zoo/python/axelera/zoo/configs/semantic_segmentation/dinov2_seg_vitl14_ade20k_linear.py) | 518x518    | 163.28 / 6.1       | Apache-2.0 |
| [dinov2_vits14](../../../axelera_zoo/python/axelera/zoo/configs/feature_extraction/dinov2_vits14.py)                                        | 224x224    |  7.63 / 131.1      | Apache-2.0 |
| [vit_large_patch16_224](../../../axelera_zoo/python/axelera/zoo/configs/classification/vit_large_patch16_224_imagenet.py)                   | 224x224    | 113.00 / 8.85      | Apache-2.0 |

## Vision Language Models (VLMs)

### VLM Performance characteristics
- **Resolution** — input image resolution of the vision encoder.
- **Ctx** — maximum context window, in tokens.
- **Prompt tokens** — prompt length used for the measurement
- **PCIe / M.2 TTFT (s)** — time to first token, in seconds (lower is better).
- **PCIe / M.2 Decode (tok/s)** — token generation rate after the first token (higher is better).
- **PCIe / M.2 Vision (fps)** — vision encoder throughput, frames per second.
- **License** — model license.

TTFT, decode rate and vision-encoder throughput are measured with the model's default prompt:
 
```bash
python -m model_axscript.llm benchmark <model>
```
 
To run a VLM, see [Serve an AxScript LLM through HuggingFace](../../tutorials/axscript/llm-huggingface.md#a-vision-language-model).

### VLM Reference platforms  
- **PCIe** — Intel Core Ultra 9 285 128GB RAM running Ubuntu 24.04 with 1x Axelera Edge 16GB 130p PCIe card.
- **M.2** — Intel Core Ultra 5 125H 32GB RAM running Ubuntu 24.04 with 1x Axelera Embedded 113m M.2 card.
- **—** — Model not supported on this form factor.

| Model                                                                                                               | Resolution | Ctx  | Prompt tokens | PCIe TTFT (s) | PCIe Decode (tok/s) | PCIe Vision (fps) | M.2 TTFT (s) | M.2 Decode (tok/s) | M.2 Vision (fps) | License               |
|:--------------------------------------------------------------------------------------------------------------------| :--------- | ---: | ------------: | ------------: | ------------------: | ----------------: | -----------: | -----------------: | ---------------: | :-------------------- |
| [lfm2_5_vl_450m](../../../axelera_zoo/python/axelera/zoo/configs/llm/lfm2_5_vl_450m.py)                             | 512×512    | 2048 |           274 |           0.5 |                37.3 |              20.5 |         0.52 |               37.2 |             19.8 | LFM Open License v1.0 |
| [lfm2_5_vl_3b](../../../axelera_zoo/python/axelera/zoo/configs/llm/lfm2_5_vl_3b.py)                                 | 512×512    | 2048 |           273 |           2.0 |                7.17 |               5.9 |         2.06 |               7.16 |             5.82 | LFM Open License v1.0 |
| [qwen3_vl_2b_instruct](../../../axelera_zoo/python/axelera/zoo/configs/llm/qwen3_vl_2b_instruct.py)                 | 288×512    | 2048 |           160 |         1.470 |                10.8 |              7.41 |        1.482 |              10.86 |             8.55 | Apache-2.0            |
| [qwen3_vl_8b_instruct](../../../axelera_zoo/python/axelera/zoo/configs/llm/qwen3_vl_8b_instruct.py)                 | 288×512    | 2048 |           160 |         4.846 |                 2.8 |              5.38 |            — |                  — |                — | Apache-2.0            |
| [smolvlm2_256m_video_instruct](../../../axelera_zoo/python/axelera/zoo/configs/llm/smolvlm2_256m_video_instruct.py) | 512×512    | 2048 |          1141 |         0.508 |                42.3 |              13.7 |        0.525 |               39.9 |             18.7 | Apache-2.0            |
| [smolvlm2_500m_video_instruct](../../../axelera_zoo/python/axelera/zoo/configs/llm/smolvlm2_500m_video_instruct.py) | 512×512    | 2048 |          1141 |         0.929 |                  24 |              17.9 |        0.954 |                 23 |             17.6 | Apache-2.0            |
| [smolvlm2_2_2b_instruct](../../../axelera_zoo/python/axelera/zoo/configs/llm/smolvlm2_2_2b_instruct.py)             | 384×384    | 2048 |          1430 |         2.024 |                7.08 |              6.45 |        2.044 |               7.13 |             7.36 | Apache-2.0            |

## Large Language Models (LLMs)
### LLM Performance characteristics
- **Ctx** — maximum context window, in tokens.
- **Prompt tokens** — prompt length used for the measurement.
- **PCIe TTFT (s)** — time to first token, in seconds (lower is better).
- **PCIe Decode (tok/s)** — token generation rate after the first token (higher is better).
- **License** — model license.

TTFT and decode rate are measured with the model's default prompt:
 
```bash
python -m model_axscript.llm benchmark <model>
```
 
To run an LLM, see [Serve an AxScript LLM through HuggingFace](../../tutorials/axscript/llm-huggingface.md).

### LLM Reference platform
**PCIe** — Intel Core Ultra 9 285 (24 cores) 128GB RAM running Ubuntu 24.04 with 1x Axelera Edge 16GB 130p PCIe card.


| Model                                                                              | Ctx  | Prompt tokens | PCIe TTFT (s) | PCIe Decode (tok/s) | License                     |
|:---------------------------------------------------------------------------------- | ---: | ------------: | ------------: | ------------------: | :-------------------------- |
| [lfm2.5-350m](../../../axelera_zoo/python/axelera/zoo/configs/llm/lfm2_5_350m.py)  | 4096 |          4096 |          0.31 |                33.8 | LFM Open License v1.0       |
| [lfm2-700m](../../../axelera_zoo/python/axelera/zoo/configs/llm/lfm2_700m.py)      | 2048 |          2048 |          0.46 |                21.2 | LFM Open License v1.0       |
| [lfm2.5-1.2b](../../../axelera_zoo/python/axelera/zoo/configs/llm/lfm2_5_1_2b.py)  |    — |             — |          0.64 |                13.7 | LFM Open License v1.0       |
| [llama3.2-1b](../../../axelera_zoo/python/axelera/zoo/configs/llm/llama_3_2_1b.py) | 4096 |             — |          0.84 |                11.9 | Llama 3.2 Community License |
| [llama3.2-3b](../../../axelera_zoo/python/axelera/zoo/configs/llm/llama_3_2_3b.py) | 4096 |          4096 |          1.50 |                 5.5 | Llama 3.2 Community License |
| [qwen3-0.6b](../../../axelera_zoo/python/axelera/zoo/configs/llm/qwen3_0_6b.py)    | 4096 |          4096 |          0.71 |                19.4 | Apache-2.0                  |
| [qwen3-1.7b](../../../axelera_zoo/python/axelera/zoo/configs/llm/qwen3_1_7b.py)    | 4096 |          4096 |          0.92 |                10.0 | Apache-2.0                  |
| [qwen3-4b](../../../axelera_zoo/python/axelera/zoo/configs/llm/qwen3_4b.py)        | 4096 |          4096 |          2.04 |                 4.7 | Apache-2.0                  |
| [qwen3-8b](../../../axelera_zoo/python/axelera/zoo/configs/llm/qwen3_8b.py)        | 4096 |          4096 |          2.92 |                 2.7 | Apache-2.0                  |

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

| Configuration | Input shape | Speedup  | mAP impact | Best for                                   |
| :------------ | ----------: | -------: | ---------: | :----------------------------------------- |
| Standard      | 640×640     | Baseline | Baseline   | General purpose, diverse content           |
| Optimized     | 640×480     | +24%     | −0.3%      | Near-square content, balanced performance  |
| Optimized     | 640×384     | +47%     | −2.0%      | Landscape video (16:9), maximum throughput |

---

## Custom weights

You can use your own trained weights with any model architecture. This involves updating the model's YAML configuration to point to your custom weight file. See [Deploy Custom Weights](../../tutorials/custom-weights.md) for the full walkthrough.

## See also

- [First Inference](../../user-guides/first-inference.md) — run your first model
- [Measure Accuracy](../../tutorials/measure-accuracy.md) — benchmark model performance
- [inference.py](../tools/inference-py.md) — full command reference
- [Glossary](../../glossary.md) — definitions of COCO, YOLO, mAP, and other terms

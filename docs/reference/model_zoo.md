![](/docs/images/Ax_Page_Banner_2500x168_01.png)

# Voyager model zoo

- [Voyager model zoo](#voyager-model-zoo)
  - [Querying the supported models and pipelines](#querying-the-supported-models-and-pipelines)
  - [Working with models trained on non-redistributable datasets](#working-with-models-trained-on-non-redistributable-datasets)
  - [Supported models and performance characteristics](#supported-models-and-performance-characteristics)
    - [Image Classification](#image-classification)
    - [Object Detection](#object-detection)
    - [Semantic Segmentation](#semantic-segmentation)
    - [Instance Segmentation](#instance-segmentation)
    - [Keypoint Detection](#keypoint-detection)
  - [Next Steps](#next-steps)

The Voyager model zoo provides a comprehensive set of industry-standard models for common tasks
such as classification, object detection, segmentation and keypoint detection. It also provides
examples of pipelines that utilize these models in different ways.

The Voyager SDK makes it easy to
[deploy](/docs/reference/deploy.md) and [evaluate](/docs/reference/inference.md)
any model or pipeline on the command-line. Furthermore, most model YAML files can be modified to
replace the default weights with your own [pretrained weights](/docs/tutorials/custom_weights.md).
Pipeline YAML files can be modified to replace any model with any other model with the same task
type.

## Querying the supported models and pipelines

To view a list of all models and pipelines supported by the current release of the Voyager SDK,
type the following command from the root of the Voyager SDK repository:

```bash
make
```

The Voyager SDK outputs information similar to the example fragment below.


```yaml
ZOO
  yolov8n-coco-onnx                yolov8n ultralytics v8.1.0, 640x640 (COCO), anchor free model
  ...
REFERENCE APPLICATION PIPELINES
  yolov8sseg-yolov8lpose           Cascade example - yolov8sseg cascaded into yolov8lpose
  ...
TUTORIALS
  t1-simplest-onnx                 ONNX Tutorial-1 - An example demonstrating how to deploy an ONNX
                                   model with minimal effort. The compiled model, located at
                                   build/t1-simplest-onnx/model1/1/model.json, can be utilized in
                                   AxRuntime to create your own pipeline.
  ...

```

The `MODELS` section lists all the basic models supported from the model zoo.

The `REFERENCE APPLICATION PIPELINES` section includes examples of more complex pipelines such as
[model cascading](/docs/tutorials/cascaded_model.md) and object tracking.

The `TUTORIALS` section provides examples referred to by the
[model deployment tutorials](/ax_models/tutorials/general/tutorials.md),
which covers many aspects of model deployment and evaluation.

You can build and run most models with a single command, for example:

```bash
./inference.py yolov8n-coco-onnx usb:0
```

This command first downloads and compiles the yolov8n-coco-onnx PyTorch model from the model zoo,
if necessary, and then runs the compiled model on an available Metis device using a USB camera as
input.

Axelera also provides precompiled versions of many models, which helps reduce deployment time on
many systems
with limited performance and memory. To use a precompiled model, first download it with a command
such as:

```bash
./download_prebuilt.py yolov8n-coco-onnx
```

Further introductory information on how to run and evaluate models on Metis hardware can be
found in the [quick start guide](/docs/tutorials/quick_start_guide.md).


## Working with models trained on non-redistributable datasets

Axelera provides pre-compiled binaries for most models, which you can use directly in inferencing
applications. Access to the dataset used to train or validate the model is required only when
compiling an ML model from source or validating and verifying the accuracy of a compiled model.

In most cases, running either [`deploy.py`](/deploy.py) or
[`inference.py`](/inference.py) with the `dataset` input option will download the
required dataset to your system, if it is not already present.
The compiler uses the dataset's validation images or representative images to calibrate
quantization, while the evaluation abilities use the dataset's test images to calculate model
accuracy.

Not all industry-standard models are trained using datasets that are publicly
redistributable. In these cases, you may need to register directly with the dataset provider
and download the dataset manually. The Voyager SDK raises an error if the dataset is
missing when needed, providing you with the expected location on your system and any
data preparation steps required. The table below summarises the datasets that require manual
download.

| Dataset  | Archive | Download location |
| :------- | :------ | :---- |
| [Cityscapes (val)](https://www.cityscapes-dataset.com/) | `gtFine_val.zip` | `data/cityscapes` |
| [Cityscapes (val)](https://www.cityscapes-dataset.com/) | `leftImg8bit_val.zip` | `data/cityscapes` |
| [Cityscapes (test)](https://www.cityscapes-dataset.com/) | `gtFine_test.zip` | `data/cityscapes` |
| [Cityscapes (test)](https://www.cityscapes-dataset.com/) | `leftImg8bit_test.zip` | `data/cityscapes` |
| [ImageNet (train)](https://www.image-net.org/download.php) | `ILSVRC2012_devkit_t12.tar.gz`  | `data/ImageNet` |
| [ImageNet (train)](https://www.image-net.org/download.php) | `ILSVRC2012_img_train.tar`  | `data/ImageNet` |
| [ImageNet (val)](https://www.image-net.org/download.php) | `ILSVRC2012_devkit_t12.tar.gz`  | `data/ImageNet` |
| [ImageNet (val)](https://www.image-net.org/download.php) | `ILSVRC2012_img_val.tar`  | `data/ImageNet` |
| WiderFace (train) | `widerface_train.zip` | `data/widerface` |
| WiderFace (val) | `widerface_val.zip` | `data/widerface` |

You are responsible for adhering to all terms and conditions of the dataset licenses.

## Supported models and performance characteristics

The tables below list all model zoo models supported by this release of the Voyager SDK. The models
are categorised by task type (such as classification or object detection) and the tables provide 
information including the accuracy of the original FP32 model, the accuracy loss following
compilation and quantization (FP32 accuracy minus Quantized model accuracy), and the host
throughput in frames per second (FPS) which is measured from the host side when running inference
on the following reference platform:

* Intel Core i9-13900K CPU with Metis 1x PCIe card
* Intel Core i5-1145G7E CPU with Metis 1x M.2 card

The [benchmarking and performance evaluation guide](/docs/tutorials/benchmarking.md) explains how
to verify these results and how to perform many other evaluation tasks on all supported platforms.

### Image Classification
| Model                                                                                      | ONNX                                                                                      | Repo                                                                                       | Resolution | Dataset     | Ref FP32 accuracy | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :----------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------- |:--------- | :---------- | ----------------: | ------------: | -----------: | ----------: | ------------: |
| [EfficientNet-B0](/ax_models/zoo/torchvision/classification/efficientnet_b0-imagenet.yaml) | [&#x1F517;](/ax_models/zoo/torchvision/classification/efficientnet_b0-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                                             | 224x224    | ImageNet-1K | 77.67             | 0.87          | 1307         | 1309        | BSD-3-Clause  |
| [EfficientNet-B1](/ax_models/zoo/torchvision/classification/efficientnet_b1-imagenet.yaml) | [&#x1F517;](/ax_models/zoo/torchvision/classification/efficientnet_b1-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                                             | 224x224    | ImageNet-1K | 77.60             | 0.40          | 885          | 882         | BSD-3-Clause  |
| [EfficientNet-B2](/ax_models/zoo/torchvision/classification/efficientnet_b2-imagenet.yaml) | [&#x1F517;](/ax_models/zoo/torchvision/classification/efficientnet_b2-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                                             | 224x224    | ImageNet-1K | 77.79             | 0.49          | 818          | 795         | BSD-3-Clause  |
| [EfficientNet-B3](/ax_models/zoo/torchvision/classification/efficientnet_b3-imagenet.yaml) | [&#x1F517;](/ax_models/zoo/torchvision/classification/efficientnet_b3-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                                             | 224x224    | ImageNet-1K | 78.54             | 0.50          | 679          | 620         | BSD-3-Clause  |
| [EfficientNet-B4](/ax_models/zoo/torchvision/classification/efficientnet_b4-imagenet.yaml) | [&#x1F517;](/ax_models/zoo/torchvision/classification/efficientnet_b4-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                                             | 224x224    | ImageNet-1K | 79.27             | 0.72          | 493          | 420         | BSD-3-Clause  |
| [MobileNetV2](/ax_models/zoo/torchvision/classification/mobilenetv2-imagenet.yaml)         | [&#x1F517;](/ax_models/zoo/torchvision/classification/mobilenetv2-imagenet-onnx.yaml)     | [&#x1F517;](https://github.com/pytorch/vision)                                             | 224x224    | ImageNet-1K | 71.87             | 1.61          | 3147         | 3114        | BSD-3-Clause  |
| [MobileNetV4-small](/ax_models/zoo/timm/mobilenetv4_small-imagenet.yaml)                   | [&#x1F517;](/ax_models/zoo/timm/mobilenetv4_small-imagenet-onnx.yaml)                     | [&#x1F517;](https://github.com/huggingface/pytorch-image-models)                           | 224x224    | ImageNet-1K | 73.74             | 2.40          | 4346         | 4199        | Apache 2.0    |
| [MobileNetV4-medium](/ax_models/zoo/timm/mobilenetv4_medium-imagenet.yaml)                 | [&#x1F517;](/ax_models/zoo/timm/mobilenetv4_medium-imagenet-onnx.yaml)                    | [&#x1F517;](https://github.com/huggingface/pytorch-image-models)                           | 224x224    | ImageNet-1K | 79.04             | 2.05          | 2138         | 2059        | Apache 2.0    |
| [MobileNetV4-large](/ax_models/zoo/timm/mobilenetv4_large-imagenet.yaml)                   | [&#x1F517;](/ax_models/zoo/timm/mobilenetv4_large-imagenet-onnx.yaml)                     | [&#x1F517;](https://github.com/huggingface/pytorch-image-models)                           | 384x384    | ImageNet-1K | 82.92             | 0.51          | 624          | 448         | Apache 2.0    |
| [MobileNetV4-aa_large](/ax_models/zoo/timm/mobilenetv4_aa_large-imagenet.yaml)             | [&#x1F517;](/ax_models/zoo/timm/mobilenetv4_aa_large-imagenet-onnx.yaml)                  | [&#x1F517;](https://github.com/huggingface/pytorch-image-models)                           | 384x384    | ImageNet-1K | 83.22             | 1.87          | 527          | 388         | Apache 2.0    |
| [SqueezeNet 1.0](/ax_models/zoo/torchvision/classification/squeezenet1.0-imagenet.yaml)    | [&#x1F517;](/ax_models/zoo/torchvision/classification/squeezenet1.0-imagenet-onnx.yaml)   | [&#x1F517;](https://github.com/pytorch/vision)                                             | 224x224    | ImageNet-1K | 58.10             | 1.30          | 1010         | 857         | BSD-3-Clause  |
| [SqueezeNet 1.1](/ax_models/zoo/torchvision/classification/squeezenet1.1-imagenet.yaml)    | [&#x1F517;](/ax_models/zoo/torchvision/classification/squeezenet1.1-imagenet-onnx.yaml)   | [&#x1F517;](https://github.com/pytorch/vision)                                             | 224x224    | ImageNet-1K | 58.19             | 2.67          | 7124         | 6855        | BSD-3-Clause  |
| [ResNet-18](/ax_models/zoo/torchvision/classification/resnet18-imagenet.yaml)              | [&#x1F517;](/ax_models/zoo/torchvision/classification/resnet18-imagenet-onnx.yaml)        | [&#x1F517;](https://github.com/pytorch/vision)                                             | 224x224    | ImageNet-1K | 69.76             | 0.46          | 3726         | 3590        | BSD-3-Clause  |
| [ResNet-34](/ax_models/zoo/torchvision/classification/resnet34-imagenet.yaml)              | [&#x1F517;](/ax_models/zoo/torchvision/classification/resnet34-imagenet-onnx.yaml)        | [&#x1F517;](https://github.com/pytorch/vision)                                             | 224x224    | ImageNet-1K | 73.30             | 0.53          | 2290         | 2096        | BSD-3-Clause  |
| [ResNet-50 v1.5](/ax_models/zoo/torchvision/classification/resnet50-imagenet.yaml)         | [&#x1F517;](/ax_models/zoo/torchvision/classification/resnet50-imagenet-onnx.yaml)        | [&#x1F517;](https://github.com/pytorch/vision)                                             | 224x224    | ImageNet-1K | 76.15             | 0.20          | 1877         | 1768        | BSD-3-Clause  |
| ResNet-50 v1.0                                                                             | [&#x1F517;](/ax_models/zoo/tensorflow/classification/resnet50-imagenet-tf2-onnx.yaml)     | [&#x1F517;](https://github.com/keras-team/keras/blob/v2.15.0/keras/applications/resnet.py) | 224x224    | ImageNet-1K | 74.72             | 0.06          | 1903         | 1770        | Apache 2.0    |
| [ResNet-101](/ax_models/zoo/torchvision/classification/resnet101-imagenet.yaml)            | [&#x1F517;](/ax_models/zoo/torchvision/classification/resnet101-imagenet-onnx.yaml)       | [&#x1F517;](https://github.com/pytorch/vision)                                             | 224x224    | ImageNet-1K | 77.37             | 1.60          | 999          | 630         | BSD-3-Clause  |
| [ResNet-152](/ax_models/zoo/torchvision/classification/resnet152-imagenet.yaml)            | [&#x1F517;](/ax_models/zoo/torchvision/classification/resnet152-imagenet-onnx.yaml)       | [&#x1F517;](https://github.com/pytorch/vision)                                             | 224x224    | ImageNet-1K | 78.31             | 0.25          | 425          | 264         | BSD-3-Clause  |
| [ResNet-10t](/ax_models/zoo/timm/resnet10t-imagenet.yaml)                                  | [&#x1F517;](/ax_models/zoo/timm/resnet10t-imagenet-onnx.yaml)                             | [&#x1F517;](https://huggingface.co/timm/resnet10t.c3_in1k)                                 | 224x224    | ImageNet-1K | 68.22             | 1.21          | 5397         | 5170        | Apache 2.0    |


### Object Detection
| Model                                                                           | ONNX                                                                                       |  Repo                                                                                                       | Resolution | Dataset   | Ref FP32 accuracy | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------- | :--------- | :-------- | ----------------: | ------------: | -----------: | ----------: | ------------: |
| RetinaFace - Resnet50                                                           | [&#x1F517;](/ax_models/zoo/torch/retinaface-resnet50-widerface-onnx.yaml)                  | [&#x1F517;](https://github.com/biubug6/Pytorch_Retinaface/tree/master)                                      | 840x840    | WiderFace | 95.25             | 0.16          | 103          | 68          | MIT           |
| RetinaFace - mb0.25                                                             | [&#x1F517;](/ax_models/zoo/torch/retinaface-mobilenet0.25-widerface-onnx.yaml)             | [&#x1F517;](https://github.com/biubug6/Pytorch_Retinaface/tree/master)                                      | 640x640    | WiderFace | 89.44             | 1.02          | 914          | 786         | MIT           |
| SSD-MobileNetV1                                                                 | [&#x1F517;](/ax_models/zoo/tensorflow/object_detection/ssd-mobilenetv1-coco-poc-onnx.yaml) | [&#x1F517;](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) | 300x300    | COCO2017  | 25.70             | 0.90          | 3075         | 2903        | Apache 2.0    |
| SSD-MobileNetV2                                                                 | [&#x1F517;](/ax_models/zoo/tensorflow/object_detection/ssd-mobilenetv2-coco-poc-onnx.yaml) | [&#x1F517;](https://github.com/tensorflow/models)                                                           | 300x300    | COCO2017  | 19.21             | 0.77          | 1935         | 1739        | Apache 2.0    |
| YOLOv3                                                                          | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov3-coco-onnx.yaml)                    | [&#x1F517;](https://github.com/ultralytics/yolov3)                                                          | 640x640    | COCO2017  | 46.61             | 0.83          | 121          | 84          | AGPL-3.0      |
| [YOLOv5s-Relu](/ax_models/zoo/yolo/object_detection/yolov5s-relu-coco.yaml)     | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov5s-relu-coco-onnx.yaml)              | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017  | 35.09             | 0.54          | 753          | 545         | AGPL-3.0      |
| [YOLOv5s-v5](/ax_models/zoo/yolo/object_detection/yolov5s-v5-coco.yaml)         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov5s-v5-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017  | 36.18             | 0.40          | 737          | 537         | AGPL-3.0      |
| [YOLOv5n](/ax_models/zoo/yolo/object_detection/yolov5n-v7-coco.yaml)            | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov5n-v7-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017  | 27.72             | 0.92          | 915          | 680         | AGPL-3.0      |
| [YOLOv5s](/ax_models/zoo/yolo/object_detection/yolov5s-v7-coco.yaml)            | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov5s-v7-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017  | 37.25             | 0.73          | 789          | 558         | AGPL-3.0      |
| [YOLOv5m](/ax_models/zoo/yolo/object_detection/yolov5m-v7-coco.yaml)            | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov5m-v7-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017  | 44.94             | 0.86          | 411          | 314         | AGPL-3.0      |
| [YOLOv5l](/ax_models/zoo/yolo/object_detection/yolov5l-v7-coco.yaml)            | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov5l-v7-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017  | 48.67             | 0.97          | 263          | 185         | AGPL-3.0      |
| [YOLOv7](/ax_models/zoo/yolo/object_detection/yolov7-coco.yaml)                 | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov7-coco-onnx.yaml)                    | [&#x1F517;](https://github.com/WongKinYiu/yolov7)                                                           | 640x640    | COCO2017  | 51.02             | 0.61          | 184          | 139         | GPL-3.0       |
| [YOLOv7-tiny](/ax_models/zoo/yolo/object_detection/yolov7-tiny-coco.yaml)       | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov7-tiny-coco-onnx.yaml)               | [&#x1F517;](https://github.com/WongKinYiu/yolov7)                                                           | 416x416    | COCO2017  | 33.12             | 0.45          | 1226         | 939         | GPL-3.0       |
| [YOLOv7 640x480](/ax_models/zoo/yolo/object_detection/yolov7-640x480-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov7-640x480-coco-onnx.yaml)            | [&#x1F517;](https://github.com/WongKinYiu/yolov7)                                                           | 640x480    | COCO2017  | 50.78             | 0.58          | 227          | 167         | GPL-3.0       |
| [YOLOv8n](/ax_models/zoo/yolo/object_detection/yolov8n-coco.yaml)               | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov8n-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017  | 37.12             | 1.18          | 714          | 526         | AGPL-3.0      |
| [YOLOv8s](/ax_models/zoo/yolo/object_detection/yolov8s-coco.yaml)               | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov8s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017  | 44.80             | 0.75          | 556          | 390         | AGPL-3.0      |
| [YOLOv8m](/ax_models/zoo/yolo/object_detection/yolov8m-coco.yaml)               | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov8m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017  | 50.16             | 0.95          | 220          | 173         | AGPL-3.0      |
| [YOLOv8l](/ax_models/zoo/yolo/object_detection/yolov8l-coco.yaml)               | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov8l-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017  | 52.83             | 0.92          | 170          | 129         | AGPL-3.0      |
| YOLOX-s                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolox-s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/Megvii-BaseDetection/YOLOX)                                                  | 640x640    | COCO2017  | 39.24             | 0.21          | 607          | 427         | Apache-2.0    |
| YOLOX-m                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolox-m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/Megvii-BaseDetection/YOLOX)                                                  | 640x640    | COCO2017  | 46.26             | 0.34          | 329          | 261         | Apache-2.0    |
| YOLOv9t                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov9t-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017  | 37.81             | 1.50          | 381          | 243         | AGPL-3.0      |
| YOLOv9s                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov9s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017  | 46.28             | 1.24          | 333          | 231         | AGPL-3.0      |
| YOLOv9m                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov9m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017  | 51.24             | 2.38          | 174          | 133         | AGPL-3.0      |
| YOLOv9c                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov9c-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017  | 52.67             | 2.57          | 172          | 130         | AGPL-3.0      |

### Semantic Segmentation
| Model                                                                    | ONNX                                                                       | Repo                                                                             | Resolution | Dataset    | Ref FP32 accuracy | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :----------------------------------------------------------------------- | :------------------------------------------------------------------------- | :------------------------------------------------------------------------------- | :--------- | :--------- | ----------------: | ------------: | -----------: | ----------: | ------------: |
| U-Net FCN 256                                                            | [&#x1F517;](/ax_models/zoo/mmlab/mmseg/unet_fcn_256-cityscapes-onnx.yaml)  | [&#x1F517;](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/unet) | 256x256    | Cityscapes | 57.75             | 0.25          | 223          | 181         | Apache 2.0    |
| [U-Net FCN 512](/ax_models/zoo/mmlab/mmseg/unet_fcn_512-cityscapes.yaml) |                                                                            | [&#x1F517;](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/unet) | 512x512    | Cityscapes | 66.62             | 0.06          | 30           | 19          | Apache 2.0    |

### Instance Segmentation
| Model                                                                         | ONNX                                                                             | Repo                                                    | Resolution | Dataset  | Ref FP32 accuracy | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :---------------------------------------------------------------------------- | :------------------------------------------------------------------------------- | :------------------------------------------------------ | :--------- | :------- | ----------------: | ------------: | -----------: | ----------: | ------------: |
| [YOLOv8n-seg](/ax_models/zoo/yolo/instance_segmentation/yolov8nseg-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolov8nseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 30.08             | 0.76          | 521          | 357         | AGPL-3.0      |
| [YOLOv8s-seg](/ax_models/zoo/yolo/instance_segmentation/yolov8sseg-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolov8sseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 36.65             | 0.47          | 428          | 310         | AGPL-3.0      |
| [YOLOv8l-seg](/ax_models/zoo/yolo/instance_segmentation/yolov8lseg-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolov8lseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 42.76             | 0.83          | 145          | 122         | AGPL-3.0      |

### Keypoint Detection
| Model                                                                        | ONNX                                                                           | Repo                                                    | Resolution | Dataset  | Ref FP32 accuracy | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :--------------------------------------------------------------------------- | :----------------------------------------------------------------------------- | :------------------------------------------------------ | :--------- | :------- | ----------------: | ------------: | -----------: | ----------: | ------------: |
| [YOLOv8n-pose](/ax_models/zoo/yolo/keypoint_detection/yolov8npose-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolov8npose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 51.11             | 1.96          | 700          | 493         | AGPL-3.0      |
| [YOLOv8s-pose](/ax_models/zoo/yolo/keypoint_detection/yolov8spose-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolov8spose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 60.65             | 2.67          | 533          | 390         | AGPL-3.0      |
| [YOLOv8l-pose](/ax_models/zoo/yolo/keypoint_detection/yolov8lpose-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolov8lpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 68.39             | 1.63          | 164          | 133         | AGPL-3.0      |


## Next Steps

You can quickly experiment with any of the above models following the
[quick start guide](/docs/tutorials/quick_start_guide.md), and replacing the name of the model in
the example commands given.

You can also evaluate your own pretrained weights for most model zoo models by following the
[custom weights tutorial](/docs/tutorials/custom_weights.md).

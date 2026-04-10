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
    - [Depth Estimation](#depth-estimation)
    - [License Plate Recognition](#license-plate-recognition)
    - [Image Enhancement Super Resolution](#image-enhancement-super-resolution)
    - [Face Recognition](#face-recognition)
    - [Re Identification](#re-identification)
    - [Large Language Model (LLM)](#large-language-model-llm)
  - [Next Steps](#next-steps)
    - [Experimenting with Optimized Input Shapes](#experimenting-with-optimized-input-shapes)
      - [How to Run the Experiment](#how-to-run-the-experiment)
  - [Further support](#further-support)

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
axdownloadmodel yolov8n-coco-onnx
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

The accuracy for each model on Metis is determined using a pipeline where the pre-processing and post-processing elements are implemented using PyTorch/torchvision:

`inference.py <model> dataset --pipe=torch-aipu --no-display`

Because most models are originally trained using pre-processing and post-processing code implemented in PyTorch, this pipeline configuration most accurately isolates the quantization loss introduced by Metis, independent of the host, thereby enabling like-for-like comparison with other AI accelerators.

`inference.py <model> media/traffic2_720p.mp4 --pipe=gst --no-display`

This command measures both the host frame rate and end-to-end frame rate. The input video is h.264-encoded 720p consistent with many real-world deployments. The tables below report the host frame rate, thereby enabling like-for-like comparison with other AI accelerators. You can also modify the above command with different video sources to measure the end-to-end performance for your specific use case.

Additionally, you can modify the accuracy measurement command with the flag --pipe=gst to measure the end-to-end accuracy on your target platform. To the best of our knowledge, we are the only provider offering this comprehensive end-to-end accuracy measurement. We will be publishing a dedicated blog post explaining the significance of this approach and how it differs from standard industry practices.

The [benchmarking and performance evaluation guide](/docs/tutorials/benchmarking.md) explains how
to verify these results and how to perform many other evaluation tasks on all supported platforms.

> [!NOTE]
> Some of the FP32 comparison values in the tables below may be missing or produced by earlier versions of Voyager SDK. These will be updated with the latest values soon.

### Image Classification
| Model                                                                                      | ONNX                                                                                      | Repo                                                             | Resolution | Dataset     | Ref FP32 Top1 | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :----------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------- | :--------------------------------------------------------------- | :--------- | :---------- | ------------: | ------------: | -----------: | ----------: | ------------: |
| [DenseNet-121](/ax_models/zoo/torchvision/classification/densenet121-imagenet.yaml)        | [&#x1F517;](/ax_models/zoo/torchvision/classification/densenet121-imagenet-onnx.yaml)     | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 74.44         |               | 281          | 156         | BSD-3-Clause  |
| [EfficientNet-B0](/ax_models/zoo/torchvision/classification/efficientnet_b0-imagenet.yaml) | [&#x1F517;](/ax_models/zoo/torchvision/classification/efficientnet_b0-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 77.67         |               | 1429         | 1450        | BSD-3-Clause  |
| [EfficientNet-B1](/ax_models/zoo/torchvision/classification/efficientnet_b1-imagenet.yaml) | [&#x1F517;](/ax_models/zoo/torchvision/classification/efficientnet_b1-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 77.6          |               | 972          | 960         | BSD-3-Clause  |
| [EfficientNet-B2](/ax_models/zoo/torchvision/classification/efficientnet_b2-imagenet.yaml) | [&#x1F517;](/ax_models/zoo/torchvision/classification/efficientnet_b2-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 77.79         |               | 903          | 863         | BSD-3-Clause  |
| [EfficientNet-B3](/ax_models/zoo/torchvision/classification/efficientnet_b3-imagenet.yaml) | [&#x1F517;](/ax_models/zoo/torchvision/classification/efficientnet_b3-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 78.54         |               | 787          | 721         | BSD-3-Clause  |
| [EfficientNet-B4](/ax_models/zoo/torchvision/classification/efficientnet_b4-imagenet.yaml) | [&#x1F517;](/ax_models/zoo/torchvision/classification/efficientnet_b4-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 79.27         |               | 576          | 436         | BSD-3-Clause  |
| [MobileNetV2](/ax_models/zoo/torchvision/classification/mobilenetv2-imagenet.yaml)         | [&#x1F517;](/ax_models/zoo/torchvision/classification/mobilenetv2-imagenet-onnx.yaml)     | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 71.87         |               | 3670         | 3638        | BSD-3-Clause  |
| [MobileNetV4-small](/ax_models/zoo/timm/mobilenetv4_small-imagenet.yaml)                   | [&#x1F517;](/ax_models/zoo/timm/mobilenetv4_small-imagenet-onnx.yaml)                     | [&#x1F517;](https://github.com/huggingface/pytorch-image-models) | 224x224    | ImageNet-1K | 73.74         |               | 4937         | 4807        | Apache 2.0    |
| [MobileNetV4-medium](/ax_models/zoo/timm/mobilenetv4_medium-imagenet.yaml)                 | [&#x1F517;](/ax_models/zoo/timm/mobilenetv4_medium-imagenet-onnx.yaml)                    | [&#x1F517;](https://github.com/huggingface/pytorch-image-models) | 224x224    | ImageNet-1K | 79.04         |               | 2517         | 2395        | Apache 2.0    |
| [MobileNetV4-large](/ax_models/zoo/timm/mobilenetv4_large-imagenet.yaml)                   | [&#x1F517;](/ax_models/zoo/timm/mobilenetv4_large-imagenet-onnx.yaml)                     | [&#x1F517;](https://github.com/huggingface/pytorch-image-models) | 384x384    | ImageNet-1K | 82.92         |               | 761          | 460         | Apache 2.0    |
| [MobileNetV4-aa_large](/ax_models/zoo/timm/mobilenetv4_aa_large-imagenet.yaml)             | [&#x1F517;](/ax_models/zoo/timm/mobilenetv4_aa_large-imagenet-onnx.yaml)                  | [&#x1F517;](https://github.com/huggingface/pytorch-image-models) | 384x384    | ImageNet-1K | 83.22         |               | 667          | 391         | Apache 2.0    |
| [SqueezeNet 1.0](/ax_models/zoo/torchvision/classification/squeezenet1.0-imagenet.yaml)    | [&#x1F517;](/ax_models/zoo/torchvision/classification/squeezenet1.0-imagenet-onnx.yaml)   | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 58.1          |               | 953          | 811         | BSD-3-Clause  |
| [SqueezeNet 1.1](/ax_models/zoo/torchvision/classification/squeezenet1.1-imagenet.yaml)    | [&#x1F517;](/ax_models/zoo/torchvision/classification/squeezenet1.1-imagenet-onnx.yaml)   | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 58.19         |               | 7298         | 7264        | BSD-3-Clause  |
| [Inception V3](/ax_models/zoo/torchvision/classification/inception_v3-imagenet.yaml)       | [&#x1F517;](/ax_models/zoo/torchvision/classification/inception_v3-imagenet-onnx.yaml)    | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K |               |               | 1136         | 636         | BSD-3-Clause  |
| [RegNetX-1_6GF](/ax_models/zoo/torchvision/classification/regnet_x_1_6gf-imagenet.yaml)    | [&#x1F517;](/ax_models/zoo/torchvision/classification/regnet_x_1_6gf-imagenet-onnx.yaml)  | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K |               |               | 695          | 369         | BSD-3-Clause  |
| [RegNetX-400MF](/ax_models/zoo/torchvision/classification/regnet_x_400mf-imagenet.yaml)    | [&#x1F517;](/ax_models/zoo/torchvision/classification/regnet_x_400mf-imagenet-onnx.yaml)  | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K |               |               | 1199         | 636         | BSD-3-Clause  |
| [RegNetY-1_6GF](/ax_models/zoo/torchvision/classification/regnet_y_1_6gf-imagenet.yaml)    | [&#x1F517;](/ax_models/zoo/torchvision/classification/regnet_y_1_6gf-imagenet-onnx.yaml)  | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K |               |               | 595          | 322         | BSD-3-Clause  |
| [RegNetY-400MF](/ax_models/zoo/torchvision/classification/regnet_y_400mf-imagenet.yaml)    | [&#x1F517;](/ax_models/zoo/torchvision/classification/regnet_y_400mf-imagenet-onnx.yaml)  | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K |               |               | 1642         | 975         | BSD-3-Clause  |
| [ResNet-18](/ax_models/zoo/torchvision/classification/resnet18-imagenet.yaml)              | [&#x1F517;](/ax_models/zoo/torchvision/classification/resnet18-imagenet-onnx.yaml)        | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 69.76         |               | 3904         | 3749        | BSD-3-Clause  |
| [ResNet-34](/ax_models/zoo/torchvision/classification/resnet34-imagenet.yaml)              | [&#x1F517;](/ax_models/zoo/torchvision/classification/resnet34-imagenet-onnx.yaml)        | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 73.3          |               | 2282         | 2075        | BSD-3-Clause  |
| [ResNet-50 v1.5](/ax_models/zoo/torchvision/classification/resnet50-imagenet.yaml)         | [&#x1F517;](/ax_models/zoo/torchvision/classification/resnet50-imagenet-onnx.yaml)        | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 76.15         |               | 1946         | 1756        | BSD-3-Clause  |
| [ResNet-101](/ax_models/zoo/torchvision/classification/resnet101-imagenet.yaml)            | [&#x1F517;](/ax_models/zoo/torchvision/classification/resnet101-imagenet-onnx.yaml)       | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 77.37         |               | 1049         | 673         | BSD-3-Clause  |
| [ResNet-152](/ax_models/zoo/torchvision/classification/resnet152-imagenet.yaml)            | [&#x1F517;](/ax_models/zoo/torchvision/classification/resnet152-imagenet-onnx.yaml)       | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 78.31         |               | 493          | 261         | BSD-3-Clause  |
| [ResNet-10t](/ax_models/zoo/timm/resnet10t-imagenet.yaml)                                  | [&#x1F517;](/ax_models/zoo/timm/resnet10t-imagenet-onnx.yaml)                             | [&#x1F517;](https://huggingface.co/timm/resnet10t.c3_in1k)       | 224x224    | ImageNet-1K | 68.22         |               | 5212         | 5015        | Apache 2.0    |
| [ResNeXt50_32x4d](/ax_models/zoo/torchvision/classification/resnext50_32x4d-imagenet.yaml) | [&#x1F517;](/ax_models/zoo/torchvision/classification/resnext50_32x4d-imagenet-onnx.yaml) | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K |               |               | 437          | 236         | BSD-3-Clause  |
| [Wide ResNet-50](/ax_models/zoo/torchvision/classification/wide_resnet50-imagenet.yaml)    | [&#x1F517;](/ax_models/zoo/torchvision/classification/wide_resnet50-imagenet-onnx.yaml)   | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K |               |               | 436          | 236         | BSD-3-Clause  |

### Object Detection
| Model                                                                           | ONNX                                                                                       | Repo                                                                                                        | Resolution | Dataset                   | Ref FP32 mAP | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------- | :--------- | :------------------------ | -----------: | ------------: | -----------: | ----------: | ------------: |
| GELAN-s                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/gelan-s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/WongKinYiu/yolov9)                                                           | 640x640    | COCO2017                  | 46.41        |               | 376          | 237         | GPL-3.0       |
| GELAN-m                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/gelan-m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/WongKinYiu/yolov9)                                                           | 640x640    | COCO2017                  | 50.86        |               | 203          | 148         | GPL-3.0       |
| GELAN-c                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/gelan-c-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/WongKinYiu/yolov9)                                                           | 640x640    | COCO2017                  |              |               | 199          | 144         | GPL-3.0       |
| RetinaFace - Resnet50                                                           | [&#x1F517;](/ax_models/zoo/torch/retinaface-resnet50-widerface-onnx.yaml)                  | [&#x1F517;](https://github.com/biubug6/Pytorch_Retinaface/tree/master)                                      | 840x840    | WiderFace                 | 95.25        |               | 90           | 51          | MIT           |
| RetinaFace - mb0.25                                                             | [&#x1F517;](/ax_models/zoo/torch/retinaface-mobilenet0.25-widerface-onnx.yaml)             | [&#x1F517;](https://github.com/biubug6/Pytorch_Retinaface/tree/master)                                      | 640x640    | WiderFace                 | 89.44        |               | 1020         | 774         | MIT           |
| SSD-MobileNetV1                                                                 | [&#x1F517;](/ax_models/zoo/tensorflow/object_detection/ssd-mobilenetv1-coco-poc-onnx.yaml) | [&#x1F517;](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) | 300x300    | COCO2017                  | 24.77        |               | 3356         | 3019        | Apache 2.0    |
| SSD-MobileNetV2                                                                 | [&#x1F517;](/ax_models/zoo/tensorflow/object_detection/ssd-mobilenetv2-coco-poc-onnx.yaml) | [&#x1F517;](https://github.com/tensorflow/models)                                                           | 300x300    | COCO2017                  | 19.25        |               | 2261         | 2195        | Apache 2.0    |
| YOLOv3                                                                          | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov3-coco-onnx.yaml)                    | [&#x1F517;](https://github.com/ultralytics/yolov3)                                                          | 640x640    | COCO2017                  | 46.61        |               | 163          | 96          | AGPL-3.0      |
| [YOLOv5s-Relu](/ax_models/zoo/yolo/object_detection/yolov5s-relu-coco.yaml)     | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov5s-relu-coco-onnx.yaml)              | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017                  | 35.09        |               | 785          | 536         | AGPL-3.0      |
| [YOLOv5s-v5](/ax_models/zoo/yolo/object_detection/yolov5s-v5-coco.yaml)         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov5s-v5-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017                  | 36.18        |               | 790          | 526         | AGPL-3.0      |
| [YOLOv5n](/ax_models/zoo/yolo/object_detection/yolov5n-v7-coco.yaml)            | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov5n-v7-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017                  | 27.72        |               | 1028         | 656         | AGPL-3.0      |
| [YOLOv5s](/ax_models/zoo/yolo/object_detection/yolov5s-v7-coco.yaml)            | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov5s-v7-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017                  | 37.25        |               | 865          | 824         | AGPL-3.0      |
| [YOLOv5m](/ax_models/zoo/yolo/object_detection/yolov5m-v7-coco.yaml)            | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov5m-v7-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017                  | 44.94        |               | 455          | 322         | AGPL-3.0      |
| [YOLOv5l](/ax_models/zoo/yolo/object_detection/yolov5l-v7-coco.yaml)            | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov5l-v7-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017                  | 48.67        |               | 299          | 204         | AGPL-3.0      |
| [YOLOv7](/ax_models/zoo/yolo/object_detection/yolov7-coco.yaml)                 | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov7-coco-onnx.yaml)                    | [&#x1F517;](https://github.com/WongKinYiu/yolov7)                                                           | 640x640    | COCO2017                  | 51.02        |               | 212          | 173         | GPL-3.0       |
| [YOLOv7-tiny](/ax_models/zoo/yolo/object_detection/yolov7-tiny-coco.yaml)       | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov7-tiny-coco-onnx.yaml)               | [&#x1F517;](https://github.com/WongKinYiu/yolov7)                                                           | 416x416    | COCO2017                  | 33.12        |               | 1441         | 1110        | GPL-3.0       |
| [YOLOv7 640x480](/ax_models/zoo/yolo/object_detection/yolov7-640x480-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov7-640x480-coco-onnx.yaml)            | [&#x1F517;](https://github.com/WongKinYiu/yolov7)                                                           | 640x480    | COCO2017                  | 50.78        |               | 242          | 164         | GPL-3.0       |
| [YOLOv8n](/ax_models/zoo/yolo/object_detection/yolov8n-coco.yaml)               | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov8n-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 37.12        |               | 834          | 764         | AGPL-3.0      |
| [YOLOv8s](/ax_models/zoo/yolo/object_detection/yolov8s-coco.yaml)               | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov8s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 44.8         |               | 643          | 524         | AGPL-3.0      |
| [YOLOv8m](/ax_models/zoo/yolo/object_detection/yolov8m-coco.yaml)               | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov8m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 50.16        |               | 242          | 177         | AGPL-3.0      |
| [YOLOv8l](/ax_models/zoo/yolo/object_detection/yolov8l-coco.yaml)               | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov8l-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 52.83        |               | 181          | 142         | AGPL-3.0      |
| YOLOv8n-obb                                                                     | [&#x1F517;](/ax_models/zoo/yolo/obb_detection/yolov8n-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset |              |               | 269          | 162         | AGPL-3.0      |
| YOLOv8l-obb                                                                     | [&#x1F517;](/ax_models/zoo/yolo/obb_detection/yolov8l-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset |              |               | 36           | 19          | AGPL-3.0      |
| YOLOX-s                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolox-s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/Megvii-BaseDetection/YOLOX)                                                  | 640x640    | COCO2017                  | 39.24        |               | 642          | 423         | Apache-2.0    |
| YOLOX-m                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolox-m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/Megvii-BaseDetection/YOLOX)                                                  | 640x640    | COCO2017                  | 46.26        |               | 349          | 268         | Apache-2.0    |
| YOLOX-x Human                                                                   | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolox-x-crowdhuman-onnx.yaml)             | [&#x1F517;](https://github.com/FoundationVision/ByteTrack)                                                  | 1440x800   | COCO2017                  |              |               | 21           | -           | MIT           |
| YOLOv9t                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov9t-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 37.81        |               | 415          | 247         | AGPL-3.0      |
| YOLOv9s                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov9s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 46.28        |               | 374          | 237         | AGPL-3.0      |
| YOLOv9m                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov9m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 51.24        |               | 203          | 148         | AGPL-3.0      |
| YOLOv9c                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov9c-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 52.67        |               | 194          | 150         | AGPL-3.0      |
| YOLOv10n                                                                        | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov10n-coco-onnx.yaml)                  | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 38.08        |               | 738          | 561         | AGPL-3.0      |
| YOLOv10s                                                                        | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov10s-coco-onnx.yaml)                  | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 45.74        |               | 580          | 461         | AGPL-3.0      |
| YOLOv10b                                                                        | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov10b-coco-onnx.yaml)                  | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 51.79        |               | 251          | 217         | AGPL-3.0      |
| YOLO11n                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolo11n-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 39.17        |               | 759          | 574         | AGPL-3.0      |
| YOLO11s                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolo11s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 46.54        |               | 565          | 426         | AGPL-3.0      |
| YOLO11m                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolo11m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 51.31        |               | 269          | 196         | AGPL-3.0      |
| YOLO11l                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolo11l-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 53.23        |               | 183          | 125         | AGPL-3.0      |
| YOLO11x                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolo11x-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 54.67        |               | 53           | 31          | AGPL-3.0      |
| YOLO11n-obb                                                                     | [&#x1F517;](/ax_models/zoo/yolo/obb_detection/yolo11n-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset |              |               | 250          | 172         | AGPL-3.0      |
| YOLO11l-obb                                                                     | [&#x1F517;](/ax_models/zoo/yolo/obb_detection/yolo11l-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset |              |               | 36           | 20          | AGPL-3.0      |
| YOLO26n                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolo26n-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  |              |               | 662          | 487         | AGPL-3.0      |
| YOLO26s                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolo26s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  |              |               | 498          | 396         | AGPL-3.0      |
| YOLO26m                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolo26m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  |              |               | 258          | 192         | AGPL-3.0      |
| YOLO26l                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolo26l-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  |              |               | 179          | 122         | AGPL-3.0      |
| YOLO26x                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolo26x-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  |              |               | 53           | 31          | AGPL-3.0      |
| YOLO26n-obb                                                                     | [&#x1F517;](/ax_models/zoo/yolo/obb_detection/yolo26n-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset |              |               | 206          | 139         | AGPL-3.0      |
| YOLO26s-obb                                                                     | [&#x1F517;](/ax_models/zoo/yolo/obb_detection/yolo26s-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset |              |               | 167          | 114         | AGPL-3.0      |
| YOLO26m-obb                                                                     | [&#x1F517;](/ax_models/zoo/yolo/obb_detection/yolo26m-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset |              |               | 58           | 33          | AGPL-3.0      |
| YOLO26l-obb                                                                     | [&#x1F517;](/ax_models/zoo/yolo/obb_detection/yolo26l-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset |              |               | 34           | 19          | AGPL-3.0      |
| YOLO26x-obb                                                                     | [&#x1F517;](/ax_models/zoo/yolo/obb_detection/yolo26x-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset |              |               | 15           | -           | AGPL-3.0      |
| YOLO-NAS S                                                                      | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolonas-s-coco-onnx.yaml)                 | [&#x1F517;](https://github.com/Deci-AI/super-gradients)                                                     | 640x640    | COCO2017                  |              |               | 450          | 318         | Apache-2.0    |
| YOLO-NAS M                                                                      | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolonas-m-coco-onnx.yaml)                 | [&#x1F517;](https://github.com/Deci-AI/super-gradients)                                                     | 640x640    | COCO2017                  |              |               | 285          | 221         | Apache-2.0    |
| YOLO-NAS L                                                                      | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolonas-l-coco-onnx.yaml)                 | [&#x1F517;](https://github.com/Deci-AI/super-gradients)                                                     | 640x640    | COCO2017                  |              |               | 157          | 96          | Apache-2.0    |

### Semantic Segmentation
| Model                                                                    | ONNX                                                                      | Repo                                                                             | Resolution | Dataset    | Ref FP32 mIoU | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :----------------------------------------------------------------------- | :------------------------------------------------------------------------ | :------------------------------------------------------------------------------- | :--------- | :--------- | ------------: | ------------: | -----------: | ----------: | ------------: |
| U-Net FCN 256                                                            | [&#x1F517;](/ax_models/zoo/mmlab/mmseg/unet_fcn_256-cityscapes-onnx.yaml) | [&#x1F517;](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/unet) | 256x256    | Cityscapes | 57.75         |               | 249          | 198         | Apache 2.0    |
| [U-Net FCN 512](/ax_models/zoo/mmlab/mmseg/unet_fcn_512-cityscapes.yaml) |                                                                           | [&#x1F517;](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/unet) | 512x512    | Cityscapes | 66.62         |               | 34           | 19          | Apache 2.0    |

### Instance Segmentation
| Model                                                                         | ONNX                                                                             | Repo                                                    | Resolution | Dataset  | Ref FP32 mAP | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :---------------------------------------------------------------------------- | :------------------------------------------------------------------------------- | :------------------------------------------------------ | :--------- | :------- | -----------: | ------------: | -----------: | ----------: | ------------: |
| [YOLOv8n-seg](/ax_models/zoo/yolo/instance_segmentation/yolov8nseg-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolov8nseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 29.98        |               | 639          | 433         | AGPL-3.0      |
| [YOLOv8s-seg](/ax_models/zoo/yolo/instance_segmentation/yolov8sseg-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolov8sseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 36.32        |               | 482          | 345         | AGPL-3.0      |
| [YOLOv8m-seg](/ax_models/zoo/yolo/instance_segmentation/yolov8mseg-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolov8mseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 40.39        |               | 198          | 156         | AGPL-3.0      |
| [YOLOv8l-seg](/ax_models/zoo/yolo/instance_segmentation/yolov8lseg-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolov8lseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 42.27        |               | 167          | 134         | AGPL-3.0      |
| YOLO11n-seg                                                                   | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolo11nseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 |              |               | 598          | 406         | AGPL-3.0      |
| YOLO11l-seg                                                                   | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolo11lseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 |              |               | 156          | 107         | AGPL-3.0      |
| YOLO26n-seg                                                                   | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolo26nseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 |              |               | 516          | 352         | AGPL-3.0      |
| YOLO26s-seg                                                                   | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolo26sseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 |              |               | 385          | 292         | AGPL-3.0      |
| YOLO26m-seg                                                                   | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolo26mseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 |              |               | 201          | 156         | AGPL-3.0      |
| YOLO26l-seg                                                                   | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolo26lseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 |              |               | 141          | 96          | AGPL-3.0      |
| YOLO26x-seg                                                                   | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolo26xseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 |              |               | 46           | 28          | AGPL-3.0      |

### Keypoint Detection
| Model                                                                        | ONNX                                                                           | Repo                                                    | Resolution | Dataset  | Ref FP32 mAP | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :--------------------------------------------------------------------------- | :----------------------------------------------------------------------------- | :------------------------------------------------------ | :--------- | :------- | -----------: | ------------: | -----------: | ----------: | ------------: |
| [YOLOv8n-pose](/ax_models/zoo/yolo/keypoint_detection/yolov8npose-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolov8npose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 51.11        |               | 822          | 723         | AGPL-3.0      |
| [YOLOv8s-pose](/ax_models/zoo/yolo/keypoint_detection/yolov8spose-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolov8spose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 60.65        |               | 592          | 471         | AGPL-3.0      |
| [YOLOv8m-pose](/ax_models/zoo/yolo/keypoint_detection/yolov8mpose-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolov8mpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 |              |               | 231          | 168         | AGPL-3.0      |
| [YOLOv8l-pose](/ax_models/zoo/yolo/keypoint_detection/yolov8lpose-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolov8lpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 68.39        |               | 186          | 145         | AGPL-3.0      |
| YOLO11n-pose                                                                 | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolo11npose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 51.15        |               | 759          | 532         | AGPL-3.0      |
| YOLO11l-pose                                                                 | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolo11lpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 67.44        |               | 179          | 122         | AGPL-3.0      |
| YOLO26n-pose                                                                 | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolo26npose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 |              |               | 658          | 450         | AGPL-3.0      |
| YOLO26s-pose                                                                 | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolo26spose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 |              |               | 467          | 359         | AGPL-3.0      |
| YOLO26m-pose                                                                 | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolo26mpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 |              |               | 235          | 166         | AGPL-3.0      |
| YOLO26l-pose                                                                 | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolo26lpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 |              |               | 174          | 120         | AGPL-3.0      |
| YOLO26x-pose                                                                 | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolo26xpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 |              |               | 51           | 30          | AGPL-3.0      |

### Depth Estimation
| Model     | ONNX                                                             | Repo                                                                              | Resolution | Dataset    | Ref FP32 RMSE | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :-------- | :--------------------------------------------------------------- | :-------------------------------------------------------------------------------- | :--------- | :--------- | ------------: | ------------: | -----------: | ----------: | ------------: |
| FastDepth | [&#x1F517;](/ax_models/zoo/torch/fastdepth-nyudepthv2-onnx.yaml) | [&#x1F517;](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/146_FastDepth) | 224x224    | NYUDepthV2 | 0.6574        |               | 974          | 855         | MIT           |

### License Plate Recognition
| Model                                      | ONNX | Repo                                                     | Resolution | Dataset       | Ref FP32 WLA | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :----------------------------------------- | :--- | :------------------------------------------------------- | :--------- | :------------ | -----------: | ------------: | -----------: | ----------: | ------------: |
| [LPRNet](/ax_models/zoo/torch/lprnet.yaml) |      | [&#x1F517;](https://github.com/sirius-ai/LPRNet_Pytorch) | 94x24      | LPRNetDataset | 89.4         |               | 10268        | 9335        | Apache-2.0    |

### Image Enhancement Super Resolution
| Model              | ONNX                                                           | Repo                                                | Resolution | Dataset                         | Ref FP32 PSNR | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :----------------- | :------------------------------------------------------------- | :-------------------------------------------------- | :--------- | :------------------------------ | ------------: | ------------: | -----------: | ----------: | ------------: |
| Real-ESRGAN-x4plus | [&#x1F517;](/ax_models/zoo/torch/real-esrgan-x4plus-onnx.yaml) | [&#x1F517;](https://github.com/xinntao/Real-ESRGAN) | 128x128    | SuperResolutionCustomSet128x128 | 24.77         |               | -            | -           | BSD-3-Clause  |

### Face Recognition
| Model                                                                | ONNX                                                    | Repo                                                     | Resolution | Dataset            | Ref FP32 top1_avg | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :------------------------------------------------------------------- | :------------------------------------------------------ | :------------------------------------------------------- | :--------- | :----------------- | ----------------: | ------------: | -----------: | ----------: | ------------: |
| [FaceNet - InceptionResnetV1](/ax_models/zoo/torch/facenet-lfw.yaml) | [&#x1F517;](/ax_models/zoo/torch/facenet-lfw-onnx.yaml) | [&#x1F517;](https://github.com/timesler/facenet-pytorch) | 160x160    | LFWTorchvisionPair | 98.35             |               | 1321         | 720         | MIT           |

### Re Identification
| Model      | ONNX                                                              | Repo                                                          | Resolution | Dataset               | Ref FP32 mAP | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :--------- | :---------------------------------------------------------------- | :------------------------------------------------------------ | :--------- | :-------------------- | -----------: | ------------: | -----------: | ----------: | ------------: |
| OSNet x1_0 | [&#x1F517;](/ax_models/zoo/torch/osnet-x1-0-market1501-onnx.yaml) | [&#x1F517;](https://github.com/KaiyangZhou/deep-person-reid)  | 256x128    | Market1501ReIdDataset | 82.55        |               | 1732         | 1770        | Apache-2.0    |
| SBS50      | [&#x1F517;](/ax_models/zoo/torch/sbs-s50-market1501-onnx.yaml)    | [&#x1F517;](https://github.com/JDAI-CV/fast-reid/tree/master) | 384x128    | Market1501ReIdDataset |              |               | 666          | 405         | Apache-2.0    |


### Large Language Model (LLM)
For details of usage please see [SLM Inference on Axelera AI Platform](/docs/tutorials/llm.md).

| Model                                                                                      | Max Context Window (tokens) | Required PCIe Card RAM |
| :----------------------------------------------------------------------------------------- | --------------------------: | ---------------------: |
| [microsoft/Phi-3-mini-4k-instruct](/ax_models/zoo/llm/phi3-mini-512-static.yaml)           | 512                         | 4 GB                   |
| [microsoft/Phi-3-mini-4k-instruct](/ax_models/zoo/llm/phi3-mini-1024-4core-static.yaml)    | 1024                        | 16 GB                  |
| [microsoft/Phi-3-mini-4k-instruct](/ax_models/zoo/llm/phi3-mini-2048-4core-static.yaml)    | 2048                        | 16 GB                  |
| [meta-llama/Llama-3.2-1B-Instruct](/ax_models/zoo/llm/llama-3-2-1b-1024-4core-static.yaml) | 1024                        | 4 GB                   |
| [meta-llama/Llama-3.2-3B-Instruct](/ax_models/zoo/llm/llama-3-2-3b-1024-4core-static.yaml) | 1024                        | 4 GB                   |
| [meta-llama/Llama-3.1-8B-Instruct](/ax_models/zoo/llm/llama-3-1-8b-1024-4core-static.yaml) | 1024                        | 16 GB                  |
| [Almawave/Velvet-2B](/ax_models/zoo/llm/velvet-2b-1024-4core-static.yaml)                  | 1024                        | 4 GB                   |

## Next Steps

You can quickly experiment with any of the above models following the
[quick start guide](/docs/tutorials/quick_start_guide.md), and replacing the name of the model in
the example commands given.

You can also evaluate your own pretrained weights for most model zoo models by following the
[custom weights tutorial](/docs/tutorials/custom_weights.md).

### Experimenting with Optimized Input Shapes

After verifying your model's baseline performance, a powerful next step is to customize the input resolution to match your actual video data.

Most models are trained on square inputs (640×640), but real-world video is often rectangular (16:9). Standard pipelines handle this mismatch by adding padding ("letterboxing"), forcing the model to process empty pixels and reducing overall efficiency.

The Voyager Model Zoo makes it easy to reclaim this performance. By switching to an optimized rectangular input shape that better matches your video's aspect ratio, you can often achieve significant speedups with minimal impact on accuracy. For applications with fixed camera angles—like surveillance or traffic monitoring—this provides one of the simplest ways to boost throughput without complex model modifications.

#### How to Run the Experiment

You don't need to retrain a model to test this. We recommend exporting models with dynamic input shapes to facilitate easy experimentation with different resolutions using a single model file.

You can validate performance trade-offs for your specific use case. The [`yolox-m-coco-onnx-rect`](/ax_models/reference/others/yolox-m-coco-onnx-rect.yaml) configuration demonstrates how simple input shape optimization can provide measurable performance improvements with minimal accuracy impact.

```bash
# Test standard configuration accuracy
inference.py yolox-m-coco-onnx dataset --pipe=torch-aipu --no-display

# Test rectangular configuration accuracy
inference.py yolox-m-coco-onnx-rect dataset --pipe=torch-aipu --no-display
```

Replace `dataset` with your test video and use `--pipe=gst` for high-performance pipeline inference benchmarking.

The following table shows measured results from our reference platform:

| Configuration | Input Shape | Speedup | mAP Impact | Use Case |
|---------------|-------------|---------|------------|----------|
| Standard | 640×640 | Baseline | Baseline | General purpose, diverse content |
| Optimized | 640×480 | +24% | -0.3% | Near-square content, balanced performance |
| Optimized | 640×384 | +47% | -2.0% | Landscape video (16:9), maximum throughput |

Compare the reported FPS and accuracy metrics to determine if the speed improvement justifies any minor accuracy trade-off for your application.

## Further support

For blog posts, projects and technical support please visit [Axelera AI Community](https://community.axelera.ai/).

For technical documents and guides please visit [Customer Portal](https://support.axelera.ai/).

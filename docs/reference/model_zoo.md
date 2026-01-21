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

### Image Classification
| Model                                                                                          | ONNX                                                                                        | Repo                                                             | Resolution | Dataset     | Ref FP32 Top1 | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :--------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------ | :--------------------------------------------------------------- | :--------- | :---------- | ------------: | ------------: | -----------: | ----------: | ------------: |
| [DenseNet-121](/ax_models/zoo/torchvision/classification/densenet121-imagenet.yaml)            | [&#x1F517;](/ax_models/zoo/torchvision/classification/densenet121-imagenet-onnx.yaml)       | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 74.44         | 1.11          | 277          | 156         | BSD-3-Clause  |
| [EfficientNet-B0](/ax_models/zoo/torchvision/classification/efficientnet_b0-imagenet.yaml)     | [&#x1F517;](/ax_models/zoo/torchvision/classification/efficientnet_b0-imagenet-onnx.yaml)   | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 77.67         | 0.91          | 1439         | 1467        | BSD-3-Clause  |
| [EfficientNet-B1](/ax_models/zoo/torchvision/classification/efficientnet_b1-imagenet.yaml)     | [&#x1F517;](/ax_models/zoo/torchvision/classification/efficientnet_b1-imagenet-onnx.yaml)   | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 77.60         | 0.48          | 974          | 987         | BSD-3-Clause  |
| [EfficientNet-B2](/ax_models/zoo/torchvision/classification/efficientnet_b2-imagenet.yaml)     | [&#x1F517;](/ax_models/zoo/torchvision/classification/efficientnet_b2-imagenet-onnx.yaml)   | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 77.79         | 0.45          | 903          | 880         | BSD-3-Clause  |
| [EfficientNet-B3](/ax_models/zoo/torchvision/classification/efficientnet_b3-imagenet.yaml)     | [&#x1F517;](/ax_models/zoo/torchvision/classification/efficientnet_b3-imagenet-onnx.yaml)   | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 78.54         | 0.56          | 786          | 740         | BSD-3-Clause  |
| [EfficientNet-B4](/ax_models/zoo/torchvision/classification/efficientnet_b4-imagenet.yaml)     | [&#x1F517;](/ax_models/zoo/torchvision/classification/efficientnet_b4-imagenet-onnx.yaml)   | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 79.27         | 0.78          | 571          | 443         | BSD-3-Clause  |
| [MobileNetV2](/ax_models/zoo/torchvision/classification/mobilenetv2-imagenet.yaml)             | [&#x1F517;](/ax_models/zoo/torchvision/classification/mobilenetv2-imagenet-onnx.yaml)       | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 71.87         | 1.48          | 3709         | 3701        | BSD-3-Clause  |
| [MobileNetV4-small](/ax_models/zoo/timm/mobilenetv4_small-imagenet.yaml)                       | [&#x1F517;](/ax_models/zoo/timm/mobilenetv4_small-imagenet-onnx.yaml)                       | [&#x1F517;](https://github.com/huggingface/pytorch-image-models) | 224x224    | ImageNet-1K | 73.74         | 1.94          | 4990         | 4923        | Apache 2.0    |
| [MobileNetV4-medium](/ax_models/zoo/timm/mobilenetv4_medium-imagenet.yaml)                     | [&#x1F517;](/ax_models/zoo/timm/mobilenetv4_medium-imagenet-onnx.yaml)                      | [&#x1F517;](https://github.com/huggingface/pytorch-image-models) | 224x224    | ImageNet-1K | 79.04         | 0.73          | 2548         | 2445        | Apache 2.0    |
| [MobileNetV4-large](/ax_models/zoo/timm/mobilenetv4_large-imagenet.yaml)                       | [&#x1F517;](/ax_models/zoo/timm/mobilenetv4_large-imagenet-onnx.yaml)                       | [&#x1F517;](https://github.com/huggingface/pytorch-image-models) | 384x384    | ImageNet-1K | 82.92         | 1.02          | 753          | 468         | Apache 2.0    |
| [MobileNetV4-aa_large](/ax_models/zoo/timm/mobilenetv4_aa_large-imagenet.yaml)                 | [&#x1F517;](/ax_models/zoo/timm/mobilenetv4_aa_large-imagenet-onnx.yaml)                    | [&#x1F517;](https://github.com/huggingface/pytorch-image-models) | 384x384    | ImageNet-1K | 83.22         | 1.41          | 643          | 399         | Apache 2.0    |
| [SqueezeNet 1.0](/ax_models/zoo/torchvision/classification/squeezenet1.0-imagenet.yaml)        | [&#x1F517;](/ax_models/zoo/torchvision/classification/squeezenet1.0-imagenet-onnx.yaml)     | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 58.1          | 3.51          | 943          | 852         | BSD-3-Clause  |
| [SqueezeNet 1.1](/ax_models/zoo/torchvision/classification/squeezenet1.1-imagenet.yaml)        | [&#x1F517;](/ax_models/zoo/torchvision/classification/squeezenet1.1-imagenet-onnx.yaml)     | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 58.19         | 1.79          | 7404         | 7242        | BSD-3-Clause  |
| [Inception V3](/ax_models/zoo/torchvision/classification/inception_v3-imagenet.yaml)           | [&#x1F517;](/ax_models/zoo/torchvision/classification/inception_v3-imagenet-onnx.yaml)      | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 69.85         | 0.21          | 1106         | 633         | BSD-3-Clause  |
| [RegNetX-1_6GF](/ax_models/zoo/torchvision/classification/regnet_x_1_6gf-imagenet.yaml)        | [&#x1F517;](/ax_models/zoo/torchvision/classification/regnet_x_1_6gf-imagenet-onnx.yaml)    | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 79.33         | 0.30          | 685          | 369         | BSD-3-Clause  |
| [RegNetX-400MF](/ax_models/zoo/torchvision/classification/regnet_x_400mf-imagenet.yaml)        | [&#x1F517;](/ax_models/zoo/torchvision/classification/regnet_x_400mf-imagenet-onnx.yaml)    | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 74.48         | 0.35          | 1179         | 627         | BSD-3-Clause  |
| [RegNetY-1_6GF](/ax_models/zoo/torchvision/classification/regnet_y_1_6gf-imagenet.yaml)        | [&#x1F517;](/ax_models/zoo/torchvision/classification/regnet_y_1_6gf-imagenet-onnx.yaml)    | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 80.73         | 0.17          | 585          | 322         | BSD-3-Clause  |
| [RegNetY-400MF](/ax_models/zoo/torchvision/classification/regnet_y_400mf-imagenet.yaml)        | [&#x1F517;](/ax_models/zoo/torchvision/classification/regnet_y_400mf-imagenet-onnx.yaml)    | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 75.63         | 0.20          | 1611         | 966         | BSD-3-Clause  |
| [ResNet-18](/ax_models/zoo/torchvision/classification/resnet18-imagenet.yaml)                  | [&#x1F517;](/ax_models/zoo/torchvision/classification/resnet18-imagenet-onnx.yaml)          | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 69.76         | 0.39          | 3951         | 3812        | BSD-3-Clause  |
| [ResNet-34](/ax_models/zoo/torchvision/classification/resnet34-imagenet.yaml)                  | [&#x1F517;](/ax_models/zoo/torchvision/classification/resnet34-imagenet-onnx.yaml)          | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 73.3          | 0.20          | 2267         | 2352        | BSD-3-Clause  |
| [ResNet-50 v1.5](/ax_models/zoo/torchvision/classification/resnet50-imagenet.yaml)             | [&#x1F517;](/ax_models/zoo/torchvision/classification/resnet50-imagenet-onnx.yaml)          | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 76.15         | 0.25          | 1938         | 1958        | BSD-3-Clause  |
| [ResNet-101](/ax_models/zoo/torchvision/classification/resnet101-imagenet.yaml)                | [&#x1F517;](/ax_models/zoo/torchvision/classification/resnet101-imagenet-onnx.yaml)         | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 77.37         | 0.62          | 1057         | 698         | BSD-3-Clause  |
| [ResNet-152](/ax_models/zoo/torchvision/classification/resnet152-imagenet.yaml)                | [&#x1F517;](/ax_models/zoo/torchvision/classification/resnet152-imagenet-onnx.yaml)         | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 78.31         | 0.18          | 484          | 262         | BSD-3-Clause  |
| [ResNet-10t](/ax_models/zoo/timm/resnet10t-imagenet.yaml)                                      | [&#x1F517;](/ax_models/zoo/timm/resnet10t-imagenet-onnx.yaml)                               | [&#x1F517;](https://huggingface.co/timm/resnet10t.c3_in1k)       | 224x224    | ImageNet-1K | 68.22         | 1.05          | 5232         | 5105        | Apache 2.0    |
| [ResNeXt50_32x4d](/ax_models/zoo/torchvision/classification/resnext50_32x4d-imagenet.yaml)     | [&#x1F517;](/ax_models/zoo/torchvision/classification/resnext50_32x4d-imagenet-onnx.yaml)   | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 77.61         | 0.09          | 431          | 236         | BSD-3-Clause  |
| [Wide ResNet-50](/ax_models/zoo/torchvision/classification/wide_resnet50-imagenet.yaml)        | [&#x1F517;](/ax_models/zoo/torchvision/classification/wide_resnet50-imagenet-onnx.yaml)     | [&#x1F517;](https://github.com/pytorch/vision)                   | 224x224    | ImageNet-1K | 78.48         | 0.37          | 430          | 236         | BSD-3-Clause  |

### Object Detection
| Model                                                                           | ONNX                                                                                       | Repo                                                                                                        | Resolution | Dataset                   | Ref FP32 mAP | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :------------------------------------------------------------------------------ | :----------------------------------------------------------------------------------------- | :---------------------------------------------------------------------------------------------------------- | :--------- | :------------------------ | -----------: | ------------: | -----------: | ----------: | ------------: |
| RetinaFace - Resnet50                                                           | [&#x1F517;](/ax_models/zoo/torch/retinaface-resnet50-widerface-onnx.yaml)                  | [&#x1F517;](https://github.com/biubug6/Pytorch_Retinaface/tree/master)                                      | 840x840    | WiderFace                 | 95.25        | 0.18          | 88           | 50          | MIT           |
| RetinaFace - mb0.25                                                             | [&#x1F517;](/ax_models/zoo/torch/retinaface-mobilenet0.25-widerface-onnx.yaml)             | [&#x1F517;](https://github.com/biubug6/Pytorch_Retinaface/tree/master)                                      | 640x640    | WiderFace                 | 89.44        | 1.20          | 1009         | 763         | MIT           |
| SSD-MobileNetV1                                                                 | [&#x1F517;](/ax_models/zoo/tensorflow/object_detection/ssd-mobilenetv1-coco-poc-onnx.yaml) | [&#x1F517;](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2018_01_28.tar.gz) | 300x300    | COCO2017                  | 24.77        | 0.01          | 3393         | 3125        | Apache 2.0    |
| SSD-MobileNetV2                                                                 | [&#x1F517;](/ax_models/zoo/tensorflow/object_detection/ssd-mobilenetv2-coco-poc-onnx.yaml) | [&#x1F517;](https://github.com/tensorflow/models)                                                           | 300x300    | COCO2017                  | 19.25        | 0.60          | 2315         | 2221        | Apache 2.0    |
| [YOLOv5s-Relu](/ax_models/zoo/yolo/object_detection/yolov5s-relu-coco.yaml)     | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov5s-relu-coco-onnx.yaml)              | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017                  | 35.09        | 0.49          | 792          | 538         | AGPL-3.0      |
| [YOLOv5s-v5](/ax_models/zoo/yolo/object_detection/yolov5s-v5-coco.yaml)         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov5s-v5-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017                  | 36.18        | 0.38          | 787          | 533         | AGPL-3.0      |
| [YOLOv5n](/ax_models/zoo/yolo/object_detection/yolov5n-v7-coco.yaml)            | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov5n-v7-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017                  | 27.72        | 0.85          | 1009         | 665         | AGPL-3.0      |
| [YOLOv5s](/ax_models/zoo/yolo/object_detection/yolov5s-v7-coco.yaml)            | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov5s-v7-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017                  | 37.25        | 0.75          | 868          | 832         | AGPL-3.0      |
| [YOLOv5m](/ax_models/zoo/yolo/object_detection/yolov5m-v7-coco.yaml)            | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov5m-v7-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017                  | 44.94        | 0.86          | 454          | 328         | AGPL-3.0      |
| [YOLOv5l](/ax_models/zoo/yolo/object_detection/yolov5l-v7-coco.yaml)            | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov5l-v7-coco-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/yolov5)                                                          | 640x640    | COCO2017                  | 48.67        | 0.87          | 300          | 204         | AGPL-3.0      |
| [YOLOv7](/ax_models/zoo/yolo/object_detection/yolov7-coco.yaml)                 | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov7-coco-onnx.yaml)                    | [&#x1F517;](https://github.com/WongKinYiu/yolov7)                                                           | 640x640    | COCO2017                  | 51.02        | 0.61          | 212          | 173         | GPL-3.0       |
| [YOLOv7-tiny](/ax_models/zoo/yolo/object_detection/yolov7-tiny-coco.yaml)       | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov7-tiny-coco-onnx.yaml)               | [&#x1F517;](https://github.com/WongKinYiu/yolov7)                                                           | 416x416    | COCO2017                  | 33.12        | 0.48          | 1441         | 1117        | GPL-3.0       |
| [YOLOv7 640x480](/ax_models/zoo/yolo/object_detection/yolov7-640x480-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov7-640x480-coco-onnx.yaml)            | [&#x1F517;](https://github.com/WongKinYiu/yolov7)                                                           | 640x480    | COCO2017                  | 50.78        | 0.57          | 237          | 164         | GPL-3.0       |
| [YOLOv8n](/ax_models/zoo/yolo/object_detection/yolov8n-coco.yaml)               | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov8n-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 37.12        | 1.21          | 847          | 771         | AGPL-3.0      |
| [YOLOv8s](/ax_models/zoo/yolo/object_detection/yolov8s-coco.yaml)               | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov8s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 44.8         | 0.79          | 649          | 531         | AGPL-3.0      |
| [YOLOv8m](/ax_models/zoo/yolo/object_detection/yolov8m-coco.yaml)               | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov8m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 50.16        | 1.24          | 238          | 175         | AGPL-3.0      |
| [YOLOv8l](/ax_models/zoo/yolo/object_detection/yolov8l-coco.yaml)               | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov8l-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 52.83        | 1.93          | 182          | 142         | AGPL-3.0      |
| YOLOv8n-obb                                                                     | [&#x1F517;](/ax_models/zoo/yolo/obb_detection/yolov8n-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset | 48.73        | 4.44          | 260          | 162         | AGPL-3.0      |
| YOLOv8l-obb                                                                     | [&#x1F517;](/ax_models/zoo/yolo/obb_detection/yolov8l-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset | 56.06        | 5.10          | 35           | 19          | AGPL-3.0      |
| YOLOX-s                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolox-s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/Megvii-BaseDetection/YOLOX)                                                  | 640x640    | COCO2017                  | 39.24        | 0.24          | 642          | 408         | Apache-2.0    |
| YOLOX-m                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolox-m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/Megvii-BaseDetection/YOLOX)                                                  | 640x640    | COCO2017                  | 46.26        | 0.22          | 349          | 268         | Apache-2.0    |
| YOLOX-x Human                                                                   | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolox-x-crowdhuman-onnx.yaml)             | [&#x1F517;](https://github.com/FoundationVision/ByteTrack)                                                  | 1440x800   | COCO2017                  | 57.66        | 3.56          | 23           | 14          | MIT           |
| YOLOv9t                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov9t-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 37.81        | 1.27          | 413          | 247         | AGPL-3.0      |
| YOLOv9s                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov9s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 46.28        | 1.08          | 377          | 237         | AGPL-3.0      |
| YOLOv9m                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov9m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 51.24        | 2.40          | 202          | 147         | AGPL-3.0      |
| YOLOv9c                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov9c-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 52.67        | 2.40          | 198          | 151         | AGPL-3.0      |
| YOLOv10n                                                                        | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov10n-coco-onnx.yaml)                  | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 38.08        | 0.74          | 725          | 561         | AGPL-3.0      |
| YOLOv10s                                                                        | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov10s-coco-onnx.yaml)                  | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 45.74        | 0.44          | 576          | 458         | AGPL-3.0      |
| YOLOv10b                                                                        | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolov10b-coco-onnx.yaml)                  | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 51.79        | 0.41          | 253          | 215         | AGPL-3.0      |
| YOLO11n                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolo11n-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 39.17        | 0.73          | 746          | 574         | AGPL-3.0      |
| YOLO11s                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolo11s-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 46.54        | 0.58          | 562          | 426         | AGPL-3.0      |
| YOLO11m                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolo11m-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 51.31        | 0.50          | 268          | 195         | AGPL-3.0      |
| YOLO11l                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolo11l-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 53.23        | 0.53          | 182          | 122         | AGPL-3.0      |
| YOLO11x                                                                         | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolo11x-coco-onnx.yaml)                   | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 640x640    | COCO2017                  | 54.67        | 0.52          | 53           | 31          | AGPL-3.0      |
| YOLO11n-obb                                                                     | [&#x1F517;](/ax_models/zoo/yolo/obb_detection/yolo11n-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset | 50.01        | 1.19          | 244          | 170         | AGPL-3.0      |
| YOLO11l-obb                                                                     | [&#x1F517;](/ax_models/zoo/yolo/obb_detection/yolo11l-obb-dotav1-onnx.yaml)                | [&#x1F517;](https://github.com/ultralytics/ultralytics)                                                     | 1024x1024  | DOTAv1DetectionOBBDataset | 56.41        | 0.80          | 35           | 20          | AGPL-3.0      |
| YOLO26n | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolo26n-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics)  | 640x640  | COCO2017 | 40.18 | 2.00 | 648.6 | 484.9 | AGPL-3.0 |
| YOLO26s | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolo26s-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics)  | 640x640  | COCO2017 | 47.66 | 2.16 | 492.1 | 394.2 | AGPL-3.0 |
| YOLO26m | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolo26m-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics)  | 640x640  | COCO2017 | 52.45 | 2.11 | 256.0 | 191.4 | AGPL-3.0 |
| YOLO26l | [&#x1F517;](/ax_models/zoo/yolo/object_detection/yolo26l-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics)  | 640x640  | COCO2017 | 54.11 | 1.99 | 177.9 | 122.0 | AGPL-3.0 |

### Semantic Segmentation
| Model                                                                    | ONNX                                                                      | Repo                                                                             | Resolution | Dataset    | Ref FP32 mIoU | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :----------------------------------------------------------------------- | :------------------------------------------------------------------------ | :------------------------------------------------------------------------------- | :--------- | :--------- | ------------: | ------------: | -----------: | ----------: | ------------: |
| U-Net FCN 256                                                            | [&#x1F517;](/ax_models/zoo/mmlab/mmseg/unet_fcn_256-cityscapes-onnx.yaml) | [&#x1F517;](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/unet) | 256x256    | Cityscapes | 57.75         | 0.18          | 248          | 198         | Apache 2.0    |
| [U-Net FCN 512](/ax_models/zoo/mmlab/mmseg/unet_fcn_512-cityscapes.yaml) |                                                                           | [&#x1F517;](https://github.com/open-mmlab/mmsegmentation/tree/main/configs/unet) | 512x512    | Cityscapes | 66.62         | 0.24          | 34           | 19          | Apache 2.0    |

### Instance Segmentation
| Model                                                                         | ONNX                                                                             | Repo                                                    | Resolution | Dataset  | Ref FP32 mAP | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :---------------------------------------------------------------------------- | :------------------------------------------------------------------------------- | :------------------------------------------------------ | :--------- | :------- | -----------: | ------------: | -----------: | ----------: | ------------: |
| [YOLOv8n-seg](/ax_models/zoo/yolo/instance_segmentation/yolov8nseg-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolov8nseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 54.12        | 1.69          | 642          | 417         | AGPL-3.0      |
| [YOLOv8s-seg](/ax_models/zoo/yolo/instance_segmentation/yolov8sseg-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolov8sseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 63.13        | 0.88          | 487          | 334         | AGPL-3.0      |
| [YOLOv8m-seg](/ax_models/zoo/yolo/instance_segmentation/yolov8mseg-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolov8mseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 68.08        | 1.02          | 197          | 148         | AGPL-3.0      |
| [YOLOv8l-seg](/ax_models/zoo/yolo/instance_segmentation/yolov8lseg-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolov8lseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 70.5         | 1.51          | 166          | 134         | AGPL-3.0      |
| YOLO11n-seg                                                                   | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolo11nseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 56.49        | 1.32          | 596          | 400         | AGPL-3.0      |
| YOLO11l-seg                                                                   | [&#x1F517;](/ax_models/zoo/yolo/instance_segmentation/yolo11lseg-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 71.76        | 0.38          | 154          | 106         | AGPL-3.0      |

### Keypoint Detection
| Model                                                                        | ONNX                                                                           | Repo                                                    | Resolution | Dataset  | Ref FP32 mAP | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :--------------------------------------------------------------------------- | :----------------------------------------------------------------------------- | :------------------------------------------------------ | :--------- | :------- | -----------: | ------------: | -----------: | ----------: | ------------: |
| [YOLOv8n-pose](/ax_models/zoo/yolo/keypoint_detection/yolov8npose-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolov8npose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 51.11        | 2.00          | 834          | 724         | AGPL-3.0      |
| [YOLOv8s-pose](/ax_models/zoo/yolo/keypoint_detection/yolov8spose-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolov8spose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 60.65        | 2.71          | 599          | 475         | AGPL-3.0      |
| [YOLOv8m-pose](/ax_models/zoo/yolo/keypoint_detection/yolov8mpose-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolov8mpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 65.58        | 2.07          | 228          | 167         | AGPL-3.0      |
| [YOLOv8l-pose](/ax_models/zoo/yolo/keypoint_detection/yolov8lpose-coco.yaml) | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolov8lpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 68.39        | 1.59          | 182          | 137         | AGPL-3.0      |
| YOLO11n-pose                                                                 | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolo11npose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 51.15        | 3.47          | 737          | 527         | AGPL-3.0      |
| YOLO11l-pose                                                                 | [&#x1F517;](/ax_models/zoo/yolo/keypoint_detection/yolo11lpose-coco-onnx.yaml) | [&#x1F517;](https://github.com/ultralytics/ultralytics) | 640x640    | COCO2017 | 67.44        | 3.08          | 177          | 122         | AGPL-3.0      |

### Depth Estimation
| Model     | ONNX                                                             | Repo                                                                              | Resolution | Dataset    | Ref FP32 RMSE | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :-------- | :--------------------------------------------------------------- | :-------------------------------------------------------------------------------- | :--------- | :--------- | ------------: | ------------: | -----------: | ----------: | ------------: |
| FastDepth | [&#x1F517;](/ax_models/zoo/torch/fastdepth-nyudepthv2-onnx.yaml) | [&#x1F517;](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/146_FastDepth) | 224x224    | NYUDepthV2 | 0.6574        | -0.0002       | 970          | 855         | MIT           |

### License Plate Recognition
| Model                                      | ONNX | Repo                                                     | Resolution | Dataset       | Ref FP32 WLA | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :----------------------------------------- | :--- | :------------------------------------------------------- | :--------- | :------------ | -----------: | ------------: | -----------: | ----------: | ------------: |
| [LPRNet](/ax_models/zoo/torch/lprnet.yaml) |      | [&#x1F517;](https://github.com/sirius-ai/LPRNet_Pytorch) | 94x24      | LPRNetDataset | 89.40        | 1.10          | 9926         | 9581        | Apache-2.0    |

### Image Enhancement Super Resolution
| Model              | ONNX                                                           | Repo                                                | Resolution | Dataset                         | Ref FP32 PSNR | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :----------------- | :------------------------------------------------------------- | :-------------------------------------------------- | :--------- | :------------------------------ | ------------: | ------------: | -----------: | ----------: | ------------: |
| Real-ESRGAN-x4plus | [&#x1F517;](/ax_models/zoo/torch/real-esrgan-x4plus-onnx.yaml) | [&#x1F517;](https://github.com/xinntao/Real-ESRGAN) | 128x128    | SuperResolutionCustomSet128x128 | 24.77         |               | 10.7         | 8.2         | BSD-3-Clause  |

### Face Recognition
| Model                                                                | ONNX                                                    | Repo                                                     | Resolution | Dataset            | Ref FP32 top1_avg | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :------------------------------------------------------------------- | :------------------------------------------------------ | :------------------------------------------------------- | :--------- | :----------------- | ----------------: | ------------: | -----------: | ----------: | ------------: |
| [FaceNet - InceptionResnetV1](/ax_models/zoo/torch/facenet-lfw.yaml) | [&#x1F517;](/ax_models/zoo/torch/facenet-lfw-onnx.yaml) | [&#x1F517;](https://github.com/timesler/facenet-pytorch) | 160x160    | LFWTorchvisionPair | 98.35             | -0.10         | 1286         | 712         | MIT           |

### Re Identification
| Model      | ONNX                                                              | Repo                                                         | Resolution | Dataset               | Ref FP32 mAP | Accuracy loss | Ref PCIe FPS | Ref M.2 FPS | Model license |
| :--------- | :---------------------------------------------------------------- | :----------------------------------------------------------- | :--------- | :-------------------- | -----------: | ------------: | -----------: | ----------: | ------------: |
| OSNet x1_0 | [&#x1F517;](/ax_models/zoo/torch/osnet-x1-0-market1501-onnx.yaml) | [&#x1F517;](https://github.com/KaiyangZhou/deep-person-reid) | 256x128    | Market1501ReIdDataset | 82.55        | 1.04          | 1745         | 1795        | Apache-2.0    |


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

## Further support
- For blog posts, projects and technical support please visit [Axelera AI Community](https://community.axelera.ai/).
- For technical documents and guides please visit [Customer Portal](https://support.axelera.ai/).

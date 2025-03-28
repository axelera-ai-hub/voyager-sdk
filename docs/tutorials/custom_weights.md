![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Model zoo custom weights deployment

- [Model zoo custom weights deployment](#model-zoo-custom-weights-deployment)
  - [Deploy a PyTorch object detector directly (YOLOv8n with custom weights)](#deploy-a-pytorch-object-detector-directly-yolov8n-with-custom-weights)
    - [Model definition](#model-definition)
    - [Data adapter definition](#data-adapter-definition)
    - [Use the Voyager SDK to deploy your new model](#use-the-voyager-sdk-to-deploy-your-new-model)
  - [Deploy an ONNX-exported model (YOLOv8n with custom weights)](#deploy-an-onnx-exported-model-yolov8n-with-custom-weights)
  - [Deploy a PyTorch classifier directly (ResNet50 with custom weights)](#deploy-a-pytorch-classifier-directly-resnet50-with-custom-weights)
  - [All supported dataset adapters](#all-supported-dataset-adapters)

Axelera model zoo models are provided with default weights based on industry-standard datasets.
This enables you to quickly and easily evaluate model accuracy on Metis compared to other
implementations. For most models you can also easily substitute the default weights with your
own pretrained weights, commonly referred to as *custom weights*.

During model deployment, the Axelera compiler inspects images from your custom training dataset.
This subset of images is referred to as the *calibration dataset*,
because the compiler uses these images to determine quantisation parameters automatically.
Post-deployment accuracy is then measured in the usual way using your validation dataset.

Deploying custom weights on Metis is usually as simple as modifying an existing YAML file and
replacing references to default weights and datasets with your own custom weights and datasets. 
In some cases, you may also need to tune the calibration process by providing
additional images.

The Axelera compiler can work with PyTorch models directly. This simplifies the deployment
process in many cases by removing the need to first export your model to ONNX.
The subsections below provide examples of how to deploy custom weights for
a [YOLOv8n](/ax_models/zoo/yolo/object_detection/yolov8n-coco.yaml) model, both
directly from PyTorch and using ONNX.

The example below is based on an [Ultralytics YOLOv8n model](https://docs.ultralytics.com/)
which has been [fine tuned](https://github.com/Muhammad-Zeerak-Khan/Automatic-License-Plate-Recognition-using-YOLOv8)
using a license plate dataset from [Roboflow](https://public.roboflow.com/object-detection/license-plates-us-eu).

> [!IMPORTANT]  
> Each model zoo model is based on a specific repository (see the *repo* column in the
> [model zoo tables](/docs/reference/model_zoo.md#supported-models-and-performance-characteristics)).
> You should ensure that your custom weights file is trained from the same repository to avoid
> any compatibility issues arising during model deployment.

## Deploy a PyTorch object detector directly (YOLOv8n with custom weights)

The first step to deploy your custom weights is to locate the model zoo YAML file
for your model (based on industry-standard weights) and copy this from a location
in `ax_models/zoo` to `customers`.

For this example, first we must install the ultralytics python module.

```bash
pip install --upgrade pip
pip install ultralytics
```

To create a new project based on YOLOv8n, run the following commands from the root
of the repository:

```bash
mkdir -p customers/mymodels
cp ax_models/zoo/yolo/object_detection/yolov8n-coco.yaml customers/mymodels/yolov8n-licenseplate.yaml
```

To download the pretrained weights, run the following command:

```bash
wget -P customers/mymodels https://d1o2y3tc25j7ge.cloudfront.net/artifacts/model_cards/weights/yolo/object_detection/yolov8n_licenseplate.pt
```

To download and unzip the dataset, run the following commands:

```bash
wget -P data https://d1o2y3tc25j7ge.cloudfront.net/artifacts/data/licenseplate_v4_resized640_aug3x-ACCURATE.zip
unzip -q data/licenseplate_v4_resized640_aug3x-ACCURATE.zip -d data/licenseplate_v4_resized640_aug3x-ACCURATE
```

Open the new file `yolov8n-licenseplate.yaml` in an editor and modify it so it looks similar to
the example below.
 
```yaml
name: yolov8n-licenseplate

description: A custom YOLOv8n model trained on a licenseplate dataset

pipeline:
  - YOLOv8n-licenseplate:
      template_path: $AXELERA_FRAMEWORK/pipeline-template/yolo-letterbox.yaml
      postprocess:
        - decodeyolo:                  # fine-tune decoder settings
            max_nms_boxes: 30000
            conf_threshold: 0.25
            nms_iou_threshold: 0.45
            nms_class_agnostic: False
            nms_top_k: 300
            eval:
               conf_threshold: 0.9     # overwrites above parameter during accuracy measurements

models:
  YOLOv8n-licenseplate:
    class: AxUltralyticsYOLO
    class_path: $AXELERA_FRAMEWORK/ax_models/yolo/ax_ultralytics.py
    weight_path: $AXELERA_FRAMEWORK/customers/mymodels/yolov8n_licenseplate.pt  # pretrained weights file
    task_category: ObjectDetection
    input_tensor_layout: NCHW
    input_tensor_shape: [1, 3, 640, 640]                                         # your model input size
    input_color_format: RGB
    num_classes: 1                                                               # the number of classes in your dataset
    dataset: licenseplate

datasets:
  licenseplate:
    class: ObjDataAdaptor
    class_path: $AXELERA_FRAMEWORK/ax_datasets/objdataadapter.py
    data_dir_name: licenseplate_v4_resized640_aug3x-ACCURATE
    label_type: YOLOv8
    labels: data.yaml
    cal_data: valid
    val_data: test
```

The YAML file contains five top-level sections:

| Field | Description |
| :---- | :---------- |
| `name` | A unique name. The Axelera model zoo convention is to specify the model name followed by a dash followed by the dataset name e.g. `yolov8n-licenseplate` |
| `description` | A user-friendly description |
| `pipeline` | An end-to-end pipeline description including image preprocessing, model and post-processing. The model names in this section must reference models declared in the `models` section |
| `models` | List of models used in the pipeline, in this case a single `YOLOv8n-licenseplate` model and its configuration parameters |
| `datasets` | List of datasets associated with the models, in this case a single `licenseplate` model (referenced in the `dataset` field of `YOLOv8n-licenseplate`) |

There is usually no need to modify `pipeline` settings when changing only the weights. The YOLO
decoder has a number of configuration properties that can be fine tuned later when running the
model using `inference.py` or the application-integration APIs.

### Model definition

In the YAML file, each model in the `models` section specifies a model class definition and its
configuration parameters.

| Field | Description |
| :---- | :---------- |
| `class` | The name of a PyTorch class used to instantiate an object of the YOLO model. For this example, set to `AxYolo`
| `class_path` | Absolute path to a Python file containing the above class definition. For this example, set to [ax_models/yolo/ax_yolo.py](/ax_models/yolo/ax_yolo.py)
| `weight_path` | Absolute path to your weights file. (Ensure that neither `weight_url` or `prequantized_url` are specified, as these fields take precedence.) |
| `task_category` | All YOLO object detectors have task category `ObjectDetection`. Keypoint detectors such as YOLOv8l-Pose have task category `KeypointDetection` and instance segmentation models such as YOLOv8m-seg have task category `InstanceSegmentation` |
| `input_tensor_layout` | The tensor layout. Ultralytics YOLO models are all `NCHW`
| `input_tensor_shape` | The size of your model input specified as [1, channels, width, height]. The batch size is always set to 1
| `input_color_format` | The input color format for your model. Ultralytics YOLO models are always `RGB`
| `num_classes` | The number of classes in the custom dataset used to train your model
| `dataset` | A reference to a dataset in the `datasets` section of the file. This is the dataset used to train your custom model

In most cases, you need only change the fields `weight_path`, `num_classes` and `dataset`, and you
should leave the other fields unchanged.

### Data adapter definition

In the YAML file, each dataset in the `datasets` section specifies a dataset adapter class and its
configuration parameters. A dataset adapter outputs images and ground truth metadata for the
configured dataset in the `AxTaskMeta` format used by the Voyager tools.


| Field | Description |
| :---- | :---------- |
| `class` | The name of a Voyager data adapter class. For this example, set to `ObjDataAdapter` |
| `class_path` | Absolute path to a Python file containing the above class definition. For this example, set to [`/ax_datasets/objdataadapter.py`](/ax_datasets/objdataadapter.py) |
| `data_dir_name` | The name of the dataset directory, which is specified relative to a *data root*. The default data root is the Voyager repository `data` directory, and this can be changed when running the Voyager tools by setting the command-line option `--data-root` |
| `label_type` | The label annotation format. Supported formats include `YOLOv8` and `COCO JSON` |
| `labels` | The name of the labels file, specified relative to `data_dir_name`. If your labels file is maintained elsewhere, use `labels_path` to provide an absolute path instead |
| `cal_data` | A text file, specified relative to `data_dir_name`, which contains a list of image paths used by the compiler during calibration. The list of images is also specified relative to `data_dir_name` |
| `val_data` | A text file, specified relative to `data_dir_name`, which contains a list of image paths used to measure end-to-end accuracy of the deployed model. The list of images is also specified relative to `data_dir_name` |
| `repr_imgs_dir_path` | Absolute path to a directory containing a set of representative images. Can be specified instead of `cal_data` |

The Axelera data adapter `ObjDataAdapter` is a flexible generic adapter that can be
configured with any dataset that uses YOLO/Darknet and COCO label formats (specified in the
field `label_type`). It provides methods to initialize calibration and validation dataloaders,
and to convert the ground-truth labels to Axelera bounding box metadata. The conversion to Axelera
metadata ensures that the dataset can be used with any model with the same task category (object detection)
and with any of the Axelera evaluation libraries for calculating related metrics, such as
mean average precision (mAP).

This example uses the YOLOv8 label format, in which a YAML file (`labels: data.yaml`) contains
an ordered list of class names, the first representing a detection with class zero, the second with
class one, and so on. If this data is not provided, detections will still be displayed but only
as class id integer numbers.

> [!CAUTION]
> Ensure the field `download_year` is removed from your YAML otherwise the data adapter
> will default to using the default COCO 2017 dataset instead.

The field `cal_data` points to the calibration dataset, and the field `val_data` points to the validation
dataset. In this example, the corresponding field values `valid` and `text` are defined in `data.yaml`
as text files each providing a list of images. In most cases you should set the calibration dataset to be
your training or validation dataset, and the compiler will select a randomly-shuffled subset of images
automatically during quantisation.

> [!TIP]
> As an alternative to specifying `cal_data` you can specify `repr_imgs_dir_path` as a path to a
> directory containing a set of representative images. The calibration dataset should contain
> between 200-400 images.

### Use the Voyager SDK to deploy your new model

To deploy your new model YAML file, run the following command in your
[activated development environment](/docs/tutorials/install.md#activate-the-development-environment):

```bash
./deploy.py customers/mymodels/yolov8n-licenseplate.yaml
```

By default the compiler uses up to 200 calibration images if provided. If the post-deployment
accuracy loss is higher than expected, you can deploy again using up to 400 calibration images
as follows:

```bash
./deploy.py <path/to/the/yaml> --num-cal-image=400
```

Refer to the [benchmarking guide](/docs/tutorials/benchmarking.md) for further information on how
to measure the accuracy of your deployed model.

## Deploy an ONNX-exported model (YOLOv8n with custom weights)

You can easily deploy YOLO models that have been
[exported from Ultralytics](https://docs.ultralytics.com/integrations/onnx/#usage).
It is easiest to use the Ultralytics command-line export tool, for example:

```bash
yolo export model=yolov8n_licenseplate.pt format=onnx opset=14

```

> [!NOTE]
> The Axelera compiler defaults to [ONNX opset14](/docs/reference/onnx-opset14-support.md).

default supports opset 14 (not all operators fully supports opset 14 and  model zoo YOLO model has been fully verified using only opsets 14-17.

To create a new project based on this ONNX model, copy the file `yolov8n-licenseplate.yaml`
(defined in the previous section) to a new file `yolov8n-licenseplate-onnx.yaml` and update
the the following sections.

```yaml
name: yolov8n-licenseplate-onnx

models:
  YOLOv8n-licenseplate:
    class: AxONNXModel
    class_path: $AXELERA_FRAMEWORK/ax_models/base_onnx.py
    weight_path: $AXELERA_FRAMEWORK/customers/mymodels/yolov8n_licenseplate.onnx
```

The model class `AxONNXModel` defined in `ax_models/base_onnx.py` is a generic class that
can be used to instantiate ONNX models. The field `weight_path` is set to the ONNX file that contains
both the model definition and weights.

You can deploy this YAML file in the same way as you deploy PyTorch models directly.

> [!NOTE]
> When deploying YOLO models earlier than version 8 from ONNX, you must specify the anchors
> in the model field `extra_kwargs`. Further information on model-specific fields can be
> found in the reference documentation for each model.

## Deploy a PyTorch classifier directly (ResNet50 with custom weights)

The example below shows how to adapt the Axelera model zoo
[torchvision ResNet-50](/ax_models/zoo/torchvision/classification/resnet50-imagenet.yaml)
model (specified with default weights based on ImageNet) to use your own custom weights and
dataset. To create a new project, run the following commands:

```bash
mkdir -p customers/mymodels
cp ax_models/zoo/torchvision/classification/resnet50-imagenet.yaml customers/mymodels/resnet50-mydataset.yaml
```

Open the new file `resnet50-mydataset.yaml` in an editor and modify it so it looks similar to
the example below.

```bash
name: resnet50-mydataset

description: A custom ResNet-50 model trained on mydataset

pipeline:
  - resnet50-imagenet:
      template_path: $AXELERA_FRAMEWORK/pipeline-template/torch-imagenet.yaml
      postprocess:
        - topk:
            k: 5

models:
  resnet50-imagenet:
    class: AxTorchvisionResNet
    class_path: $AXELERA_FRAMEWORK/ax_models/torchvision/resnet.py
    weight_path: weights/your_weights.pt
    task_category: Classification
    input_tensor_layout: NCHW
    input_tensor_shape: [1, 3, 224, 224]
    input_color_format: RGB
    num_classes: 1000
    extra_kwargs:
      torchvision-args:
        block: Bottleneck
        layers: [3, 4, 6, 3]
    dataset: mydataset

datasets:
  mydataset:
    class: TorchvisionDataAdapter
    class_path: $AXELERA_FRAMEWORK/ax_datasets/torchvision.py
    data_dir_name: mydataset
    labels: labels.names
    repr_imgs_dir_path: absolute/path/to/cal/images
    val_data: path/to/val/root
```

The configuration options are similar to the YOLO example.

The torchvision implementation of ResNet provides a number of parameters including `block` and
`layers` which specialize ResNet backbone to a specific architecture (in this example ResNet-50).
Model-specific parameterisation options are usually provided as `extra_kwargs` section.

In this case, the dataset
clss specified is `TorchvisionDataAdapter`. The Axelera data adapter
[`TorchvisionDataAdapter`](/ax_datasets/torchvision.py)
is a flexible generic adapter that can be configured with any dataset that uses the standard
ImageNet label format. This means that the calibration images are put into a single directory
(specified with `repr_images_dir_path`) and the validation dataset is specified as a root folder that
contains a set of subdirectories each with the label name (e.g. `val/person`, `val/cat`, etc.).

## All supported dataset adapters

The Axelera model zoo models are defined using generic data adapters. You can usually just
replace reference to the industry-standard weights and dataset implementation with one
of the following generic data loaders, assuming that your data uses an industry standard
labelling format.

| Data adapter class | Task category | Description | YAML fields |
| :----------------- | :------------ | :---------- | :---------- |
| [TorchvisionDataAdapter](/ax_datasets/torchvision.py) | `Classification` | Axelera generic data loader for classifier models based on torchvision. Provides built-in support for many torchvision datasets such as ImageNet, MNIST, LFWPairs, LFWPeople and CalTech101 | [reference](/docs/reference/adapters.md#torchvisiondataadapter) |
| [ObjDataAdapter](/ax_datasets/objdataadapter.py) | `ObjectDetection` | Axelera generic data loader with multi-format label support. Provides built-in support for [COCO 2014 and 2017 datasets](https://cocodataset.org) | [reference](/docs/reference/adapters.md#objdataadaptor) |
| [KptDataAdapter](/ax_datasets/objdataadapter.py) | `KeypointDetection` | Axelera data loader for YOLO keypoints. Provides built-in support for [COCO 2017 dataset](https://cocodataset.org) | [reference](/docs/reference/adapters.md#kptdataadapter) |
| [SegDataAdapter](/ax_datasets/objdataadapter.py) | `InstanceSegmentation` | Axelera data loader for YOLO segmentation. Provides built-in support for [COCO 2017 dataset](https://cocodataset.org) | [reference](/docs/reference/adapters.md#segdataadapter) |

If your cannot use any of these data adapters for your dataset, you can instead implement your own custom data adapter.

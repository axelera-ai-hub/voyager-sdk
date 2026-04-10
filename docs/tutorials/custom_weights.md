![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Model zoo custom weights deployment

## Contents
- [Model zoo custom weights deployment](#model-zoo-custom-weights-deployment)
  - [Contents](#contents)
  - [Prerequisites](#prerequisites)
  - [Level](#level)
  - [Overview](#overview)
  - [Deploy a PyTorch object detector directly (YOLOv8n with custom weights)](#deploy-a-pytorch-object-detector-directly-yolov8n-with-custom-weights)
    - [Model definition](#model-definition)
    - [Data adapter definition](#data-adapter-definition)
    - [Ultralytics Data YAML Support](#ultralytics-data-yaml-support)
    - [Traditional Class Name Configuration](#traditional-class-name-configuration)
    - [Model env definition](#model-env-definition)
    - [Use the Voyager SDK to deploy your new model](#use-the-voyager-sdk-to-deploy-your-new-model)
  - [Deploy an ONNX-exported model (YOLOv8n with custom weights)](#deploy-an-onnx-exported-model-yolov8n-with-custom-weights)
  - [Deploy a PyTorch classifier directly (ResNet50 with custom weights)](#deploy-a-pytorch-classifier-directly-resnet50-with-custom-weights)
  - [All supported dataset adapters](#all-supported-dataset-adapters)
  - [Automatic asset management with URLs and MD5 checksums](#automatic-asset-management-with-urls-and-md5-checksums)
    - [Model weight management](#model-weight-management)
    - [Dataset management](#dataset-management)
  - [Next Steps](#next-steps)
  - [Related Documentation](#related-documentation)
  - [Further support](#further-support)

## Prerequisites
- Complete [Quick Start Guide](quick_start_guide.md) - understand basic model deployment
- [Application Integration](application.md) or [AxInferenceNet Tutorial](axinferencenet.md) - know how to run inference
- Your own trained model weights (PyTorch .pth or ONNX format)
- Calibration dataset representative of your target domain
- Understanding of YAML configuration files

## Level
**Intermediate** - Requires understanding of model architectures and dataset preparation

## Overview
Axelera model zoo models are provided with default weights based on industry-standard datasets.
This enables you to quickly and easily evaluate model accuracy on Metis compared to other
implementations. For most models you can also easily substitute the default weights with your
own pretrained weights, commonly referred to as *custom weights*.

During model deployment, the Axelera compiler inspects images from your custom training dataset.
This subset of images is referred to as the *calibration dataset*,
because the compiler uses these images to determine quantization parameters automatically.
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

> [!NOTE]
> **About file organization**:
> - **YAML model configuration files**: Place these in the `customers/` directory to keep your custom model definitions separate from the SDK's model zoo in `ax_models/zoo/`.
> - **Weight files and datasets**: Can be stored anywhere on your filesystem using absolute paths. You may organize them in `customers/` alongside your YAMLs for convenience, or store them in shared locations, external storage, or any directory you prefer.

The first step to deploy your custom weights is to locate the model zoo YAML file
for your model (based on industry-standard weights) and copy this from a location
in `ax_models/zoo` to a location of your choice (e.g., `customers/` directory).

To create a new project based on YOLOv8n, run the following commands from the root
of the repository:

```bash
mkdir -p customers/mymodels
cp ax_models/zoo/yolo/object_detection/yolov8n-coco.yaml customers/mymodels/yolov8n-licenseplate.yaml
```

To download the pretrained weights, run the following command:

```bash
wget -P customers/mymodels https://media.axelera.ai/artifacts/model_cards/weights/yolo/object_detection/yolov8n_licenseplate.pt
```

To download and unzip the dataset, run the following commands:

```bash
wget -P data https://media.axelera.ai/artifacts/data/licenseplate_v4_resized640_aug3x-ACCURATE.zip
unzip -q data/licenseplate_v4_resized640_aug3x-ACCURATE.zip -d data/licenseplate_v4_resized640_aug3x-ACCURATE
```

Open the new file `yolov8n-licenseplate.yaml` in an editor and modify it so it looks similar to
the example below.
 
```yaml
axelera-model-format: 1.0.0

name: yolov8n-licenseplate

description: A custom YOLOv8n model trained on a licenseplate dataset

pipeline:
  - detections:
      model_name: yolov8n-licenseplate
      input:
        type: image
      preprocess:
        - letterbox:
            height: ${{input_height}}
            scaleup: true
            width: ${{input_width}}
        - torch-totensor:
      inference:
        handle_all: false
      postprocess:
        - decodeyolo:
            box_format: xywh
            conf_threshold: 0.25
            label_filter: ${{label_filter}}
            max_nms_boxes: 30000
            nms_class_agnostic: true
            nms_iou_threshold: 0.45
            nms_top_k: 300
            normalized_coord: false
            use_multi_label: false
            eval:
              conf_threshold: 0.001
              nms_class_agnostic: false
              nms_iou_threshold: 0.7
              use_multi_label: true

models:
  yolov8n-licenseplate:
    class: AxUltralyticsYOLO
    class_path: $AXELERA_FRAMEWORK/ax_models/yolo/ax_ultralytics.py
    weight_path: /home/user/mymodels/yolov8n_licenseplate.pt  # absolute path to your weights file (recommended)
    task_category: ObjectDetection
    input_tensor_layout: NCHW
    input_tensor_shape: [1, 3, 640, 640]                                 # your model input size
    input_color_format: RGB
    num_classes: 1                                                       # the number of classes in your dataset
    dataset: licenseplate                                                # dataset

datasets: # Python dataloader
  licenseplate:
    class: ObjDataAdapter
    class_path: $AXELERA_FRAMEWORK/ax_datasets/objdataadapter.py
    data_dir_name: licenseplate_v4_resized640_aug3x-ACCURATE
    ultralytics_data_yaml: data.yaml    # Recommended: Ultralytics format (replaces cal_data, val_data, labels)
    label_type: YOLOv8

model-env:
  dependencies: [ultralytics]

operators:
  decodeyolo:
    class: DecodeYolo
    class_path: $AXELERA_FRAMEWORK/ax_models/decoders/yolo.py
```

The YAML file contains five top-level sections:

| Field | Description |
| :---- | :---------- |
| `name` | A unique name. The Axelera model zoo convention is to specify the model name followed by a dash followed by the dataset name e.g. `yolov8n-licenseplate` |
| `description` | A user-friendly description |
| `model-env` | Optional section specifying model-specific dependencies that are auto-installed during deployment but isolated from SDK dependencies. |
| `pipeline` | An end-to-end pipeline description including image preprocessing, model and post-processing. The model names in this section must reference models declared in the `models` section |
| `models` | List of models used in the pipeline, in this case a single `YOLOv8n-licenseplate` model and its configuration parameters |
| `datasets` | List of datasets associated with the models, in this case a single `licenseplate` model (referenced in the `dataset` field of `YOLOv8n-licenseplate`) |

There is usually no need to modify `pipeline` settings when changing only the weights. The YOLO
decoder has a number of configuration properties that can be fine tuned later when running the
model using `inference.py` or the application-integration APIs.

Modifying parameters like `conf_threshold` can be useful for specific use cases. In the YAML pipeline, the top-level `conf_threshold` (e.g., 0.25) applies to general inference with `inference.py` on media input. However, the `eval` section’s `conf_threshold` (e.g., 0.001) takes over during evaluation (e.g., `inference.py <model> dataset`) to measure performance. Evaluation often uses a lower threshold, like `0.001` in YOLO, for broader detection, while real-world scenarios may require testing higher values (e.g., `0.25`) to optimize precision and recall. Here’s an example of how this might look in the YAML configuration:

```yaml
conf_threshold: 0.25
eval:
  conf_threshold: 0.001
```

### Model definition

In the YAML file, each model in the `models` section specifies a model class definition and its
configuration parameters.

| Field | Description |
| :---- | :---------- |
| `class` | The name of a PyTorch class used to instantiate an object of the YOLO model. For this example, set to `AxYolo`
| `class_path` | Absolute path to a Python file containing the above class definition. For this example, set to [ax_models/yolo/ax_yolo.py](/ax_models/yolo/ax_yolo.py)
| `weight_path` | **Path to your custom weights file**. For custom weights, use one of these options:<br><br>• **Absolute path (recommended)**: `/home/user/my_weights/model.pt` - Most straightforward, no ambiguity<br><br>• **Home directory with `~`**: `~/my_weights/model.pt` - Expands to your home directory<br><br>• **Relative path**: If your YAML is at `customers/mymodels/yolov8n-licenseplate.yaml` and model name is `yolov8n-licenseplate`, then `custom_weights/model.pt` resolves to `customers/mymodels/yolov8n-licenseplate/custom_weights/model.pt`<br><br>**Note**: The `weights/` prefix (e.g., `weights/model.pt`) is a special pattern used by the SDK's model zoo for auto-downloaded weights. It's removed during path resolution, which can cause confusion. For custom weights, use absolute paths or avoid the `weights/` prefix. Ensure that neither `weight_url` nor `prequantized_url` are specified, as these fields take precedence over `weight_path`. |
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
| `label_type` | The label annotation format. Supported formats include `YOLOv8` and `COCO JSON` for custom datasets, both widely recognized as industry-standard labeling formats, and `COCO2017` and `COCO2014` for the official COCO 2017 and 2014 datasets. |
| `labels` | The name of the labels file, specified relative to `data_dir_name`. If your labels file is maintained elsewhere, use `labels_path` to provide an absolute path instead |
| `ultralytics_data_yaml` | **Recommended**: Path to an Ultralytics data YAML file, relative to `data_dir_name`. Automatically generates `cal_data`, `val_data`, and `labels_path` from the Ultralytics configuration. Cannot be used with `cal_data`, `val_data`, `labels_path`, or `labels` |
| `cal_data` | Calibration data. For `label_type: YOLOv8`, a text file (e.g., `val.txt`) with image paths or a directory (e.g., `valid`) with labels and images subdirs, relative to `data_dir_name`. For `label_type: COCO JSON`, a COCO JSON file (e.g., `val.json`) relative to `data_dir_name`. |
| `val_data` | Validation data for end-to-end accuracy. For `label_type: YOLOv8`, a text file (e.g., `test.txt`) with image paths or a directory (e.g., `test`) with labels and images subdirs, relative to `data_dir_name`. For `label_type: COCO JSON`, a COCO JSON file (e.g., `test.json`) relative to `data_dir_name`. |
| `repr_imgs_dir_path` | Absolute path to a directory containing a set of representative images. Can be specified instead of `cal_data` |

The Axelera data adapter `ObjDataAdapter` is a flexible generic adapter that can be
configured with any dataset that uses YOLO/Darknet and COCO label formats (specified in the
field `label_type`). It provides methods to initialize calibration and validation dataloaders,
and to convert the ground-truth labels to Axelera bounding box metadata. The conversion to Axelera
metadata ensures that the dataset can be used with any model with the same task category (object detection)
and with any of the Axelera evaluation libraries for calculating related metrics, such as
mean average precision (mAP).

### Ultralytics Data YAML Support

The Axelera framework now supports using Ultralytics-format data YAML files directly, which simplifies dataset configuration for users migrating from Ultralytics. Instead of manually creating `cal_data`, `val_data`, and `labels_path` files, you can use the `ultralytics_data_yaml` parameter.

**Traditional Format:**
```yaml
datasets:
  licenseplate:
    class: ObjDataAdapter
    class_path: $AXELERA_FRAMEWORK/ax_datasets/objdataadapter.py
    data_dir_name: licenseplate_v4_resized640_aug3x-ACCURATE
    label_type: YOLOv8
    labels: data.yaml
    cal_data: val.txt     # Text file with image paths or directory like `valid`
    val_data: test.txt    # Text file with image paths or directory like `test`
```

**Ultralytics Format (Recommended):**
```yaml
datasets:
  licenseplate:
    class: ObjDataAdapter
    class_path: $AXELERA_FRAMEWORK/ax_datasets/objdataadapter.py
    data_dir_name: licenseplate_v4_resized640_aug3x-ACCURATE
    ultralytics_data_yaml: data.yaml  # Automatically handles cal_data, val_data, and labels
    label_type: YOLOv8
```

When using `ultralytics_data_yaml`, the framework automatically:
- Parses the Ultralytics data YAML file to extract class names and dataset paths
- Creates temporary calibration and validation files based on the `train` and `val` fields
- Handles both directory-based and file-based image lists
- Supports relative and absolute paths with intelligent path resolution

**Example Ultralytics data.yaml:**
```yaml
# licenseplate_v4_resized640_aug3x-ACCURATE/data.yaml
path: ../datasets/licenseplate_v4_resized640_aug3x-ACCURATE
train: images/train
val: images/val
test: images/test

nc: 1
names: ['License_Plate']
```

The framework intelligently resolves paths like `../train/images` to `train/images` when the actual dataset structure doesn't match, ensuring compatibility with various dataset formats.

> [!NOTE]
> When using `ultralytics_data_yaml`, you cannot simultaneously specify `cal_data`, `val_data`, `labels_path`, or `labels` as these are automatically generated from the Ultralytics configuration.

### Traditional Class Name Configuration

For traditional configurations or when not using Ultralytics format, class names can be defined in two ways:

1. **YAML File**: A YAML file (e.g., `labels: data.yaml`) specifies an ordered list of class names. Examples:

```yaml
names: ['License_Plate']          # List format: class 0 is "License_Plate"
```
Or:
```yaml
names: 
  0: License_Plate               # Dictionary format: class 0 is "License_Plate"
```

The first entry corresponds to class 0, the second to class 1, and so on.

2. **Names File**: A text file (e.g., `labels: labels.names`) lists class names, one per line. Example:

```text
License_Plate                    # Class 0
```

Each line maps to a class ID in order (first line = class 0, second = class 1, etc.).
If no class name data is provided, detections are displayed as integer class IDs (e.g., 0, 1, etc.) instead of names. If `labels` is not provided, detections will still be displayed but only
as class id integer numbers.


> [!CAUTION]
> Ensure the field `label_type` is set to `YOLOv8` or `COCO JSON` in your YAML otherwise the data adapter
> will default to using the default COCO 2017 dataset instead.

The field `cal_data` points to the calibration dataset, and the field `val_data` points to the validation
dataset. In this example, the corresponding field values `valid` and `text` are defined in `data.yaml`
as text files each providing a list of images. In most cases you should set the calibration dataset to be
your training or validation dataset, and the compiler will select a randomly-shuffled subset of images
automatically during quantization.

> [!TIP]
> As an alternative to specifying `cal_data` you can specify `repr_imgs_dir_path` as a path to a
> directory containing a set of representative images. The calibration dataset should contain
> between 200-400 images.

### Model env definition

The `model-env` section allows you to specify Python packages required specifically for your model implementation:

```yaml
model-env:
  dependencies:
    - ultralytics
```

These dependencies are automatically installed during deployment without requiring manual pip install commands. You can specify exact versions using standard pip syntax:

```yaml
dependencies:
  - ultralytics==8.0.12
```

This approach keeps model-specific dependencies separate from SDK dependencies, ensuring cleaner deployment. When needed, you can specify exact package versions (e.g., those used during training) to maintain consistency between environments, though this is optional for many cases.


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
yolo export model=yolov8n_licenseplate.pt format=onnx opset=17

```

> [!NOTE]
> The Axelera compiler defaults to [ONNX opset17](/docs/reference/onnx-opset17-support.md).
> Not all operators fully support opset 17, and the model zoo YOLO models have been fully verified using only opsets 14-17.

To create a new project based on this ONNX model, copy the file `yolov8n-licenseplate.yaml`
(defined in the previous section) to a new file `yolov8n-licenseplate-onnx.yaml` and update
the following sections.

```yaml
name: yolov8n-licenseplate-onnx

models:
  YOLOv8n-licenseplate:
    class: AxONNXModel
    class_path: $AXELERA_FRAMEWORK/ax_models/base_onnx.py
    weight_path: /home/user/mymodels/yolov8n_licenseplate.onnx  # absolute path (recommended)
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

```yaml
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
class specified is `TorchvisionDataAdapter`. The Axelera data adapter
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
| [ObjDataAdapter](/ax_datasets/objdataadapter.py) | `ObjectDetection` | Axelera generic data loader with multi-format label support. Provides built-in support for [COCO 2014 and 2017 datasets](https://cocodataset.org) and Ultralytics data YAML format | [reference](/docs/reference/adapters.md#objdataadapter) |
| [KptDataAdapter](/ax_datasets/objdataadapter.py) | `KeypointDetection` | Axelera data loader for YOLO keypoints. Provides built-in support for [COCO 2017 dataset](https://cocodataset.org) | [reference](/docs/reference/adapters.md#kptdataadapter) |
| [SegDataAdapter](/ax_datasets/objdataadapter.py) | `InstanceSegmentation` | Axelera data loader for YOLO segmentation. Provides built-in support for [COCO 2017 dataset](https://cocodataset.org) | [reference](/docs/reference/adapters.md#segdataadapter) |

If your cannot use any of these data adapters for your dataset, you can instead implement your own custom data adapter.

## Automatic asset management with URLs and MD5 checksums

The Axelera model zoo provides mechanisms to automatically download and verify model weights and datasets. This is particularly useful for ISVs who want to distribute models to end customers without requiring manual asset management.

### Model weight management

When defining a model in your YAML file, you can include `weight_url` and `weight_md5` fields:

```yaml
    models:
      YOLOv8n-custom:
        weight_path: weights/yolov8n_custom.onnx
        weight_url: https://example.com/path/to/yolov8n_custom.onnx
        weight_md5: 292190cdc6452001c1d1d26c46ecf88b
```

With this configuration:

1. The system will first check if the file exists at the specified `weight_path`
2. If the file exists, it verifies the MD5 checksum matches the provided `weight_md5`
3. If the file doesn't exist or the MD5 doesn't match, it downloads the file from `weight_url` to `~/.cache/axelera/weights/yolov8n_custom.onnx`

> [!IMPORTANT]
> **When to use `weight_url` and `weight_md5`:**
> - **For team sharing/production**: Include these fields to share weights across your team (via internal server, S3, etc.), ensuring everyone uses the exact same weights with automatic download and integrity verification.
> - **For local development**: Omit these fields and use only `weight_path` to iterate quickly with your local weights without triggering downloads or checksum verification.

### Dataset management

Similarly, datasets can be automatically downloaded and extracted using the following fields:

```yaml
    datasets:
      custom_dataset:
        data_dir_name: custom_dataset
        dataset_url: https://example.com/path/to/custom_dataset.zip
        dataset_md5: 92a3905a986b28a33bb66b0b17184d16  # optional
        dataset_drop_dirs: 0  # controls directory structure after extraction
```

When these fields are present:

1. The system will download the dataset archive from `dataset_url` if it doesn't exist locally
2. If `dataset_md5` is provided, it verifies the checksum
3. The archive is extracted to `data_root/custom_dataset/`
4. The `dataset_drop_dirs` parameter controls how many directory levels to remove during extraction:
   - `0` (default): Preserves the original directory structure
   - `1`: Removes one level of directories from the archive

For example, with a ZIP file containing `dataset/train/` and `dataset/val/`:
- With `dataset_drop_dirs: 0`: Files extract to `data_root/custom_dataset/dataset/train/` and `data_root/custom_dataset/dataset/val/`
- With `dataset_drop_dirs: 1`: Files extract to `data_root/custom_dataset/train/` and `data_root/custom_dataset/val/`

> [!TIP]
> The `weight_path` field supports absolute paths and paths with `~/` for home directory expansion, giving you flexibility in where you store your model weights.

## Next Steps
- **Deploy entirely new architectures**: [Custom Model Tutorial](custom_model.md)
- **Chain multiple models**: [Cascaded Models Tutorial](cascaded_model.md)
- **Validate accuracy**: [Benchmarking Tutorial](benchmarking.md)
- **Integrate into application**: [Application Integration](application.md)
- **Optimize performance**: [Compiler Configs Reference](../reference/compiler_configs.md)

## Related Documentation
**Tutorials:**
- [Custom Model](custom_model.md) - For entirely new architectures not in Model Zoo
- [Application Integration](application.md) - Run your custom model in applications
- [Benchmarking](benchmarking.md) - Validate accuracy of your custom weights

**References:**
- [Compiler CLI](../reference/compiler_cli.md) - Command-line compilation options
- [Compiler API](../reference/compiler_api.md) - Programmatic compilation
- [Compiler Configs](../reference/compiler_configs.md) - Advanced configuration options
- [Deploy Reference](../reference/deploy.md) - Deployment workflow details
- [Model Zoo](../reference/model_zoo.md) - Available base architectures
- [YAML Operators](../reference/yaml_operators.md) - Pipeline operator syntax
- [Pipeline Operators](../reference/pipeline_operators.md) - Pre/post-processing operators

**Examples:**
- Model deployment examples in SDK use custom weight patterns

## Further support

For blog posts, projects and technical support please visit [Axelera AI Community](https://community.axelera.ai/).

For technical documents and guides please visit [Customer Portal](https://support.axelera.ai/).

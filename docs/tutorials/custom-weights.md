---
title: "Deploy Custom Weights"
---
# Deploy Custom Weights

How to take a Model Zoo architecture and run it with your own trained weights.

The process is: copy a YAML file from `ax_models/zoo/`, update the paths to point at your weights and dataset, and run `./deploy.py`. No code required.

---

## Overview

Model Zoo models come with default weights trained on standard datasets (COCO, ImageNet, etc.). To use your own weights — trained on a custom dataset — you modify the YAML pipeline file to point at your files.

The YAML has five top-level sections:

| Section | What it does |
|---------|-------------|
| `name` | Unique model identifier |
| `description` | Human-readable description |
| `pipeline` | End-to-end inference steps (pre-process → model → post-process) |
| `models` | Model class, weight path, input shape, number of classes |
| `datasets` | Dataset adapter, calibration data, validation data |
| `model-env` | *(Optional)* Python package dependencies auto-installed during deployment |

In most cases you only need to change: `weight_path`, `num_classes`, and the dataset configuration.

---

## Example: YOLOv8n object detector (PyTorch)

### 1. Copy the base YAML

```bash
mkdir -p customers/mymodels
cp ax_models/zoo/yolo/object_detection/yolov8n-coco.yaml \
   customers/mymodels/yolov8n-licenseplate.yaml
```

### 2. Edit the YAML

Open `customers/mymodels/yolov8n-licenseplate.yaml` and update the `name`, `models`, and `datasets` sections:

```yaml
name: yolov8n-licenseplate

pipeline:
  - detections:
      model_name: yolov8n-licenseplate
      input:
        type: image
      preprocess:
        - letterbox:
            height: ${input_height}
            width: ${input_width}
            scaleup: true
        - torch-totensor:
      postprocess:
        - decodeyolo:
            conf_threshold: 0.25
            nms_iou_threshold: 0.45
            nms_top_k: 300
            eval:
              conf_threshold: 0.001

models:
  yolov8n-licenseplate:
    class: AxUltralyticsYOLO
    class_path: $AXELERA_FRAMEWORK/ax_models/yolo/ax_ultralytics.py
    weight_path: /absolute/path/to/yolov8n_licenseplate.pt   # use absolute paths
    task_category: ObjectDetection
    input_tensor_layout: NCHW
    input_tensor_shape: [1, 3, 640, 640]
    input_color_format: RGB
    num_classes: 1                # update to match your dataset
    dataset: licenseplate

datasets:
  licenseplate:
    class: ObjDataAdapter
    class_path: $AXELERA_FRAMEWORK/ax_datasets/objdataadapter.py
    data_dir_name: licenseplate_dataset
    ultralytics_data_yaml: data.yaml   # recommended: auto-resolves cal/val/labels
    label_type: YOLOv8

model-env:
  dependencies: [ultralytics]
```

> [!TIP]
> **`weight_path` resolution rules**
> The SDK resolves `weight_path` in this order:
>
> 1. **Absolute path** — used as-is: `/home/user/models/yolov8n.pt`
> 2. **Home expansion** — `~` is expanded: `~/models/yolov8n.pt`
> 3. **Relative path** — resolved relative to the **SDK root** (the directory containing `deploy.py`), not the YAML file's location
>
> **The `weights/` prefix trap**: Model Zoo YAML files use `weight_path: weights/modelname.pt`. This resolves to `<SDK_ROOT>/weights/modelname.pt` — a directory where the SDK auto-downloads zoo weights. If you copy a zoo YAML and keep this pattern, the SDK will look for your file in `<SDK_ROOT>/weights/`, which is probably not where you put it. Use an absolute path for custom weights.
>
> **`weight_url` / `prequantized_url` override**: If a zoo YAML you copied has `weight_url` or `prequantized_url` set, those values take precedence and the SDK will download (or use) those files instead of your `weight_path`. Remove or comment out these fields when using custom weights.


### 3. Deploy

```bash
./deploy.py customers/mymodels/yolov8n-licenseplate.yaml
```

If accuracy is lower than expected, increase the calibration image count (default: 200):

```bash
./deploy.py customers/mymodels/yolov8n-licenseplate.yaml --num-cal-images 400
```

### 4. Run inference

```bash
./inference.py yolov8n-licenseplate usb:0
```

---

## Dataset configuration

### Ultralytics format (recommended)

If your dataset uses an Ultralytics `data.yaml`, point to it with `ultralytics_data_yaml`. The SDK automatically extracts class names, calibration paths, and validation paths:

```yaml
datasets:
  licenseplate:
    class: ObjDataAdapter
    class_path: $AXELERA_FRAMEWORK/ax_datasets/objdataadapter.py
    data_dir_name: licenseplate_dataset
    ultralytics_data_yaml: data.yaml
    label_type: YOLOv8
```

### Traditional format

If you're not using Ultralytics format, specify calibration and validation data explicitly:

```yaml
datasets:
  licenseplate:
    class: ObjDataAdapter
    class_path: $AXELERA_FRAMEWORK/ax_datasets/objdataadapter.py
    data_dir_name: licenseplate_dataset
    label_type: YOLOv8
    labels: labels.names       # one class name per line
    cal_data: valid             # directory or text file with image paths (for calibration)
    val_data: test              # directory or text file with image paths (for validation)
```

The calibration dataset should contain 200–400 representative images. The SDK randomly selects from this set during quantization.

---

## ONNX export path

If you have already exported to ONNX, use `AxONNXModel` instead of the PyTorch class:

```yaml
models:
  yolov8n-licenseplate:
    class: AxONNXModel
    class_path: $AXELERA_FRAMEWORK/ax_models/base_onnx.py
    weight_path: /absolute/path/to/yolov8n_licenseplate.onnx
```

Export from Ultralytics:

```bash
yolo export model=yolov8n_licenseplate.pt format=onnx opset=17
```

---

## Example: ResNet50 classifier

The classifier workflow is the same — copy the YAML, change weights and dataset:

```yaml
name: resnet50-mydataset

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
    weight_path: /absolute/path/to/your_weights.pt
    task_category: Classification
    input_tensor_layout: NCHW
    input_tensor_shape: [1, 3, 224, 224]
    input_color_format: RGB
    num_classes: 10              # update to match your dataset
    dataset: mydataset
    extra_kwargs:
      torchvision-args:
        block: Bottleneck
        layers: [3, 4, 6, 3]

datasets:
  mydataset:
    class: TorchvisionDataAdapter
    class_path: $AXELERA_FRAMEWORK/ax_datasets/torchvision.py
    data_dir_name: mydataset
    labels: labels.names
    repr_imgs_dir_path: /absolute/path/to/calibration/images
    val_data: /absolute/path/to/val/root   # subdirs per class: val/cat, val/dog, etc.
```

---

## YAML fields reference

### models section

| Field | Description |
|-------|-------------|
| `class` | PyTorch/ONNX class to instantiate the model |
| `class_path` | Path to the Python file containing the class |
| `weight_path` | **Your weights file.** Use absolute paths. |
| `task_category` | `ObjectDetection`, `Classification`, `KeypointDetection`, `InstanceSegmentation` |
| `input_tensor_layout` | `NCHW` (most models) |
| `input_tensor_shape` | `[1, channels, height, width]` — must match your model |
| `input_color_format` | `RGB` or `BGR` |
| `num_classes` | Number of classes in your custom dataset |
| `dataset` | References a dataset name in the `datasets` section |

### datasets section

| Field | Description |
|-------|-------------|
| `class` | Data adapter class to use (see supported adapters below) |
| `class_path` | Path to the Python file containing the class |
| `data_dir_name` | Directory name under the SDK data root where the dataset is stored |
| `label_type` | Label format: `YOLOv8`, `COCO`, `VOC`, etc. |
| `labels` | Path to a file listing class names, one per line |
| `ultralytics_data_yaml` | Path to a `data.yaml` (Ultralytics format). Auto-resolves cal/val/labels. Relative paths resolved from the dataset directory. |
| `cal_data` | Directory or text file listing calibration image paths |
| `val_data` | Directory or text file listing validation image paths |
| `repr_imgs_dir_path` | Absolute path to a directory of representative calibration images (used by TorchvisionDataAdapter) |

### Supported data adapters

| Adapter class | Use for | Class path |
|---------------|---------|-----------|
| `ObjDataAdapter` | Object detection (YOLO, COCO formats) | `$AXELERA_FRAMEWORK/ax_datasets/objdataadapter.py` |
| `TorchvisionDataAdapter` | Classification (ImageNet-style folder structure) | `$AXELERA_FRAMEWORK/ax_datasets/torchvision.py` |
| `KptDataAdapter` | Keypoint detection | `$AXELERA_FRAMEWORK/ax_datasets/kptdataadapter.py` |
| `SegDataAdapter` | Semantic and instance segmentation | `$AXELERA_FRAMEWORK/ax_datasets/segdataadapter.py` |

### model-env section

Packages listed here are automatically installed during deployment, isolated from SDK dependencies:

```yaml
model-env:
  dependencies:
    - ultralytics==8.0.12   # pin to training version if needed
```

---

## Auto-download weights and datasets

> [!TIP]
> **When to use this**
> This is primarily useful for **reproducible deployments** — sharing models across teams, CI/CD pipelines, or deploying to multiple devices where you need consistent results without manually copying weight files. For local development, you can skip this and just set `weight_path` to your local file (comment out `weight_url` and `weight_md5`).


You can have the SDK automatically download and verify weights:

```yaml
models:
  my-model:
    weight_path: weights/model.pt           # local cache path
    weight_url: https://example.com/model.pt
    weight_md5: 292190cdc6452001c1d1d26c46ecf88b
```

The SDK checks `weight_path` first; if absent or checksum mismatches, it downloads from `weight_url`. Similarly for datasets:

```yaml
datasets:
  my-dataset:
    data_dir_name: my_dataset
    dataset_url: https://example.com/dataset.zip
    dataset_md5: 92a3905a986b28a33bb66b0b17184d16
    dataset_drop_dirs: 1   # strip one directory level during extraction
```

---

## See also

- [Dataset Adapters](../reference/models/adapters.md) — full reference for ObjDataAdapter, TorchvisionDataAdapter, etc.
- [Cascaded Pipelines](cascaded-pipelines.md) — chain two models together
- [Compiler Python API](../reference/compiler/compiler-api.md) — programmatic compilation workflow
- [Performance Metrics](../reference/measurement/performance.md) — validate accuracy after deployment

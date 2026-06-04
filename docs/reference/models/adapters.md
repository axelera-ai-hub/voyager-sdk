---
title: "Dataset Adapters"
---
# Dataset Adapters

Dataset adapters provide calibration and validation data to the compiler and accuracy measurement tools. Each adapter class corresponds to a task category and a dataset format.

See [Deploy Custom Weights](../../tutorials/custom-weights.md) for how adapters are used in pipeline YAML files.

---

## ObjDataAdapter

For object detection models. Supports COCO 2014/2017 and custom datasets in YOLO/COCO JSON formats.

**Definition:** `$AXELERA_FRAMEWORK/ax_datasets/objdataadapter.py`

| Field | Type | Description |
|-------|------|-------------|
| `data_dir_name` | string | Dataset directory name, relative to the data root (default: `data/`) |
| `label_type` | string | Label format: `YOLOv8`, `COCO JSON`, `COCO2017`, `COCO2014` |
| `ultralytics_data_yaml` | string | Path to an Ultralytics `data.yaml`, relative to `data_dir_name`. Auto-generates cal/val/labels. Cannot be used with `cal_data`, `val_data`, or `labels`. |
| `cal_data` | string | Calibration data: directory with images or text file listing image paths |
| `val_data` | string | Validation data: directory with images or text file listing image paths |
| `labels` | string | Labels file (YAML or `.names`), relative to `data_dir_name` |
| `repr_imgs_dir_path` | string | Absolute path to a directory of representative calibration images. Alternative to `cal_data`. |
| `download_year` | string | COCO dataset year: `"2014"` or `"2017"` (for built-in COCO support) |
| `format` | string | COCO class format: default COCO-80, or `"coco91"` / `"coco91-with-bg"` |
| `output_format` | string | Bounding box format: `"xyxy"` (default), `"xywh"`, `"ltwh"` |
| `is_label_image_same_dir` | bool | `True` if images and labels are in the same directory (default: `False`) |
| `[val\|cal]_img_dir_name` | string | Override image directory for val or cal, relative to `data_dir_name` |

### KptDataAdapter

Subclass of `ObjDataAdapter` for YOLO keypoint detection models. Uses COCO 2017 pose dataset.

Inherits all `ObjDataAdapter` fields. No additional fields.

### SegDataAdapter

Subclass of `ObjDataAdapter` for YOLO instance segmentation models. Uses COCO 2017 segmentation dataset.

Inherits all `ObjDataAdapter` fields, plus:

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `is_mask_overlap` | bool | `True` | Whether masks are overlapped during evaluation |
| `eval_with_letterbox` | bool | `True` | Whether to use letterbox resize during mask evaluation |
| `mask_size` | tuple | `(160, 160)` | Mask dimensions `(height, width)` during evaluation |

---

## TorchvisionDataAdapter

For classification models based on torchvision. Supports ImageNet-style datasets and standard torchvision datasets.

**Definition:** `$AXELERA_FRAMEWORK/ax_datasets/torchvision.py`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `dataset_name` | string | `"ImageFolder"` | torchvision dataset class to use |

Beyond `dataset_name`, fields map to the corresponding torchvision dataset class arguments. Supported dataset classes:

### ImageFolder

For custom datasets organized as `root/class_name/image.jpg`.

| Field | Type | Default |
|-------|------|---------|
| `split` | string | `"train"` |
| `[val\|cal]_data` | string (required) | — |
| `[val\|cal]_index_pkl` | string | None |
| `is_one_indexed` | bool | `False` |

### ImageNet

| Field | Type | Default |
|-------|------|---------|
| `split` | string | `"train"` |

### MNIST / CIFAR10

| Field | Type | Default |
|-------|------|---------|
| `train` | bool | `True` |
| `download` | bool | `True` |

### VOCDetection

| Field | Type | Default |
|-------|------|---------|
| `year` | string | `"2011"` |
| `image_set` | string | `"train"` |
| `download` | bool | `False` |

### LFWPairs / LFWPeople

| Field | Type | Default |
|-------|------|---------|
| `image_set` | string | `"funneled"` |
| `download` | bool | `True` |
| `split` | string | `"test"` |

### Caltech101

| Field | Type | Default |
|-------|------|---------|
| `download` | bool | `False` |

---

## See also

- [Deploy Custom Weights](../../tutorials/custom-weights.md) — how to configure adapters in a YAML file
- [Measure Accuracy](../../tutorials/measure-accuracy.md) — running validation with a dataset source
- [Model Zoo](model-zoo.md) — pre-built models and their dataset types

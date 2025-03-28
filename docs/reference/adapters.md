![](/docs/images/Ax_Page_Banner_2500x168_01.png)
# Dataset Adapters - YAML Configurations
 
- [Dataset Adapters - YAML Configurations](#dataset-adapters---yaml-configurations)
  - [ObjDataAdaptor](#objdataadaptor)
    - [KptDataAdapter](#kptdataadapter)
    - [SegDataAdapter](#segdataadapter)
  - [TorchvisionDataAdapter](#torchvisiondataadapter)
    - [ImageFolder](#imagefolder)
    - [ImageNet](#imagenet)
    - [MNIST](#mnist)
    - [CIFAR10](#cifar10)
    - [VOCDetection](#vocdetection)
    - [LFWPairs](#lfwpairs)
    - [LFWPeople](#lfwpeople)
    - [Caltech101](#caltech101)

## ObjDataAdaptor

For use with COCO 2014 and 2017 datasets.

Reference Documentation: [COCO](https://docs.ultralytics.com/datasets/detect/coco/)

Definition: [`/ax_datasets/objdataadapter.py`](/ax_datasets/objdataadapter.py)

| Field | Type | Description |
|---|---|---|
| download_year | str | Which year of the COCO dataset to use, "2014" or "2017" are supported. |
| format | str | COCO-80-classes is used by default with the COCO dataset. This can be overridden to COCO-91-classes with format "coco91" or "coco91-with-bg". |
| is_label_image_same_dir | bool | Whether the images and labels are stored in the same directory. Defaults to False, so assumes different directories. |
| [val\|cal]_img_dir_name | str (Path) | Specify a custom directory of images to validate or calibrate with instead of the original. Allows specifying a directory directly instead of going via a file like [val\|cal]_data. |
| output_format | str | Specify the output format for the dataset. Default is "xyxy". Can be changed to "xywh" and "ltwh" |

The following adapters are subclasses of of `ObjDataAdapter`, so can use the above configurations as
well as any additional configurations specified.

### KptDataAdapter

For use with YOLO keypoint detection models using the COCO 2017 dataset.

Reference Documentation: [Coco-Pose](https://docs.ultralytics.com/datasets/pose/coco/)

Definition: [`/ax_datasets/objdataadapter.py`](/ax_datasets/objdataadapter.py)

| Field | Type | Description |
|---|---|---|
|  |  |  |

### SegDataAdapter

For use with YOLO segmentation models using the COCO 2017 dataset.

Reference Documentation: [COCO-Seg](https://docs.ultralytics.com/datasets/segment/coco/)

Definition: [`/ax_datasets/objdataadapter.py`](/ax_datasets/objdataadapter.py)

| Field | Type | Description |
|---|---|---|
| is_mask_overlap | bool | Whether or not the mask is overlapped when evaluating. Defaults to True. |
| eval_with_letterbox | bool | Whether or not to use letterbox to resize the mask when evaluating. Defaults to True. |
| mask_size | Tuple[Int, Int] | The size of the mask in format height X width when evaluating. Defaults to 160 X 160. |

## TorchvisionDataAdapter

For use with classifier models based on torchvision

Reference Documentation: [PyTorch](https://pytorch.org/vision/stable/index.html)

Definition: [`/ax_datasets/torchvision.py`](/ax_datasets/torchvision.py)

| Field | Type | Description |
|---|---|---|
| dataset_name | str | The name of the dataset class to use. The `torchvision.datasets` classes defined in `torchvision.py` are available, and the default is `ImageFolder`  |

Beyond this, for torchvision, the args depend on torchvision dataset class used. And map to the original
documented configurations for the dataset. Not all configurations available in torchvision are supported.
Supported configurations are listed for each dataset below.

### ImageFolder

Reference documentation: [ImageFolder](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html#torchvision.datasets.ImageFolder)

| Field | Type | Default
|---|---|---|
| split | str | "train" |
| [val\|cal]_data | str (Path, Required) | None |
| [val\|cal]_index_pkl | str (Path) | None |
| is_one_indexed | bool | False |

> Note that the use of "val" or "cal" for the field depends on whether you are performing validation ("val") or calibration ("cal").

### ImageNet

Reference documentation: [ImageNet](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageNet.html#torchvision.datasets.ImageNet)

| Field | Type | Default
|---|---|---|
| split | str | "train" |

### MNIST

Reference documentation: [MNIST](https://pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html#torchvision.datasets.MNIST)

| Field | Type | Default
|---|---|---|
| train | bool | True |
| download | bool | True |

### CIFAR10

Reference documentation: [CIFAR10](https://pytorch.org/vision/stable/generated/torchvision.datasets.CIFAR10.html#torchvision.datasets.CIFAR10)

| Field | Type | Default
|---|---|---|
| train | bool | True |
| download | bool | True |

### VOCDetection

Reference documentation: [VOCDetection](https://pytorch.org/vision/stable/generated/torchvision.datasets.VOCDetection.html#torchvision.datasets.VOCDetection)

| Field | Type | Default
|---|---|---|
| year | str | "2011" |
| image_set | str | "train" |
| download | bool | False |

### LFWPairs

Reference documentation: [LFWPairs](https://pytorch.org/vision/stable/generated/torchvision.datasets.LFWPairs.html#torchvision.datasets.LFWPairs)

| Field | Type | Default
|---|---|---|
| image_set | str | "funneled" |
| download | bool | True |
| split | str | "test" |

### LFWPeople

Reference documentation: [LFWPeople](https://pytorch.org/vision/stable/generated/torchvision.datasets.LFWPeople.html#torchvision.datasets.LFWPeople)

| Field | Type | Default
|---|---|---|
| image_set | str | "funneled" |
| download | bool | True |
| split | str | "test" |

### Caltech101

Reference documentation: [Caltech101](https://pytorch.org/vision/stable/generated/torchvision.datasets.Caltech101.html#torchvision.datasets.Caltech101)

| Field | Type | Default
|---|---|---|
| download | bool | False |

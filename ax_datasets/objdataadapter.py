# Copyright Axelera AI, 2024
# Flexible dataset module for object detection models that use
# COCO, Darknet/YOLO and PascalVOC label formats
# Returns VOC, COCO or YOLO-style labels; default as VOC
#
# It also supports polygon to mask conversion for instance
# segmentation task and keypoints for human pose estimation task.
#
# VOC format
# xyxy: the upper-left coordinates of the bounding box and
#       the lower-right coordinates of the bounding box
#
# COCO format
# x,y: the upper-left coordinates of the bounding box
# w,h: the dimensions of the bounding box
#
# Darknet/YOLO format
# x & y are center of the bounding box
# xywh ranging: (0,1]; relative to width and height of image
#
# Output image format is following PIL (RGB). Note that
# image preprocessing should be provided by the 'transform'
# argument when initializing the dataloader

# nc: number of corrupt images
# nm: number of missing labels
# nf: number of found labels
# ne: number of empty labels

import enum
import hashlib
import json
from multiprocessing.pool import ThreadPool as Pool
import os
from pathlib import Path
import pickle
import tempfile
import time
from typing import Any, Dict, List

from PIL import ExifTags, Image, ImageOps
import numpy as np
from torch.utils.data import Dataset
from tqdm import tqdm

from axelera import types
from axelera.app import eval_interfaces, logging_utils, utils
from axelera.app.model_utils.box import xywh2ltwh, xywh2xyxy, xyxy2xywh
from axelera.app.torch_utils import torch
import axelera.app.yaml as YAML
from axelera.app.yaml import AxYAMLError

LOG = logging_utils.getLogger(__name__)

COCO = {
    "2014": {
        # When using COCO dataset, we use COCO-80-classes defaultly.
        # To return COCO-91-classes, please explicitly set 'format' to coco91
        # or coco91-with-bg in YAML dataset. Noted that coco80 and coco91 are
        # both zero-indexing. coco91-with-bg is one-indexing coco91 with background
        # class at index 0 which is used by COCO annotations by default. coco90-with-bg
        # is one-indexing coco90 with background class at index 0 which is used by
        # Tensorflow.
        "format": "coco80",
        "train": {
            "url": "https://d1o2y3tc25j7ge.cloudfront.net/artifacts/data/coco/train2014.zip",
            "filename": Path("images", "train2014.zip"),
            "md5": "",
            "base_dir": Path("images", "train2014"),
            "train": Path("trainvalno5k.part"),
        },
        "train_reference": {
            "url": "https://d1o2y3tc25j7ge.cloudfront.net/artifacts/data/coco/trainvalno5k.part",
            "filename": Path("trainvalno5k.part"),
            "md5": "914293b6d98f3339f1e2c5491918e7e9",
        },
        "val": {
            "url": "https://d1o2y3tc25j7ge.cloudfront.net/artifacts/data/coco/val2014.zip",
            "filename": Path("images", "val2014.zip"),
            "md5": "a3d79f5ed8d289b7a7554ce06a5782b3",
            "base_dir": Path("images", "val2014"),
            "val": Path("5k.part"),
        },
        "val_reference": {
            "url": "https://d1o2y3tc25j7ge.cloudfront.net/artifacts/data/coco/5k.part",
            "filename": Path("5k.part"),
            "md5": "853bde28e4a9eec2aa1ad6cd1f22ce2b",
        },
        "annotations": {
            "url": "https://d1o2y3tc25j7ge.cloudfront.net/artifacts/data/coco/instances_train-val2014.zip",
            "filename": Path("instances_train-val2014.zip"),
            "md5": "59582776b8dd745d649cd249ada5acf7",
            "base_dir": Path("annotations"),
            "check_files": [
                Path("coco", "annotations", "instances_val2014.json"),
            ],
        },
        "labels": {
            "url": "https://d1o2y3tc25j7ge.cloudfront.net/artifacts/data/coco/labels.tgz",
            "filename": Path("labels.tgz"),
            "md5": "d4adaab8f1174071b6eb20f85ce24016",
            "base_dir": Path("labels"),
            "check_files": [
                Path("labels", "train2014"),
                Path("labels", "val2014"),
            ],
        },
    },
    "2017": {
        "format": "coco80",
        "train": {
            "url": "https://d1o2y3tc25j7ge.cloudfront.net/artifacts/data/coco/train2017.zip",
            "filename": Path("images", "train2017.zip"),
            "md5": "cced6f7f71b7629ddf16f17bbcfab6b2",
            "base_dir": Path("images", "train2017"),
            "train": Path("train2017.txt"),
        },
        "val": {
            "url": "https://d1o2y3tc25j7ge.cloudfront.net/artifacts/data/coco/val2017.zip",
            "filename": Path("images", "val2017.zip"),
            "md5": "442b8da7639aecaf257c1dceb8ba8c80",
            "base_dir": Path("images", "val2017"),
            "val": Path("val2017.txt"),
        },
        "annotations": {
            "url": "https://d1o2y3tc25j7ge.cloudfront.net/artifacts/data/coco/annotations_trainval2017.zip",
            "filename": Path("annotations_trainval2017.zip"),
            "md5": "f4bbac642086de4f52a3fdda2de5fa2c",
            "base_dir": Path("annotations"),
            "check_files": [
                Path("annotations", "captions_train2017.json"),
                Path("annotations", "captions_val2017.json"),
                Path("annotations", "instances_train2017.json"),
                Path("annotations", "instances_val2017.json"),
                Path("annotations", "person_keypoints_train2017.json"),
                Path("annotations", "person_keypoints_val2017.json"),
            ],
        },
        "labels": {
            "url": "https://d1o2y3tc25j7ge.cloudfront.net/artifacts/data/coco/coco2017labels.zip",
            "filename": Path("coco2017labels.zip"),
            "md5": "db23085d5ebf5d1ca33f304eab7bdd87",
            "base_dir": Path("labels"),
            "check_files": [
                Path("labels", "train2017"),
                Path("labels", "val2017"),
            ],
            "drop_dirs": 1,
        },
        "kpts": {
            "url": "https://d1o2y3tc25j7ge.cloudfront.net/artifacts/data/coco/coco2017labels-kpts.zip",
            "filename": Path("coco2017labels-kpts.zip"),
            "md5": "6df7d8035de2ab34c11afb689c7fa827",
            "base_dir": Path("labels_kpts"),
            "check_files": [
                Path("labels_kpts", "train2017"),
                Path("labels_kpts", "val2017"),
                Path("labels_kpts", "train2017.txt"),
                Path("labels_kpts", "val2017.txt"),
            ],
        },
        "seg": {
            "url": "https://d1o2y3tc25j7ge.cloudfront.net/artifacts/data/coco/coco2017labels-seg.zip",
            "filename": Path("coco2017labels-seg.zip"),
            "md5": "92ed19be81b08680f5c56df48304d99c",
            "base_dir": Path("labels_seg"),
            "check_files": [
                Path("labels_seg", "train2017"),
                Path("labels_seg", "val2017"),
            ],
        },
    },
}

VOC = {
    "2012": {
        "trainval": {
            "url": "https://d1o2y3tc25j7ge.cloudfront.net/artifacts/data/pascalvoc/VOCtrainval_11-May-2012.tar",
            "filename": "VOCtrainval_11-May-2012.tar",
            "md5": "6cd6e144f989b92b3379bac3b3de84fd",
            "base_dir": Path("VOCdevkit", "VOC2012"),
            "val": Path("VOC2012", "ImageSets", "Layout", "val.txt"),
        },
    },
    "2011": {
        "trainval": {
            "url": "https://d1o2y3tc25j7ge.cloudfront.net/artifacts/data/pascalvoc/VOCtrainval_25-May-2011.tar",
            "filename": "VOCtrainval_25-May-2011.tar",
            "md5": "6c3384ef61512963050cb5d687e5bf1e",
            "base_dir": Path("TrainVal", "VOCdevkit", "VOC2011"),
        },
    },
    "2010": {
        "trainval": {
            "url": "https://d1o2y3tc25j7ge.cloudfront.net/artifacts/data/pascalvoc/VOCtrainval_03-May-2010.tar",
            "filename": "VOCtrainval_03-May-2010.tar",
            "md5": "da459979d0c395079b5c75ee67908abb",
            "base_dir": Path("VOCdevkit", "VOC2010"),
        },
    },
    "2009": {
        "trainval": {
            "url": "https://d1o2y3tc25j7ge.cloudfront.net/artifacts/data/pascalvoc/VOCtrainval_11-May-2009.tar",
            "filename": "VOCtrainval_11-May-2009.tar",
            "md5": "a3e00b113cfcfebf17e343f59da3caa1",
            "base_dir": Path("VOCdevkit", "VOC2009"),
        },
    },
    "2008": {
        "trainval": {
            "url": "https://d1o2y3tc25j7ge.cloudfront.net/artifacts/data/pascalvoc/VOCtrainval_14-Jul-2008.tar",
            "filename": "VOCtrainval_11-May-2012.tar",
            "md5": "2629fa636546599198acfcfbfcf1904a",
            "base_dir": Path("VOCdevkit", "VOC2008"),
        },
    },
    "2007": {
        "trainval": {
            "url": "https://d1o2y3tc25j7ge.cloudfront.net/artifacts/data/pascalvoc/VOCtrainval_06-Nov-2007.tar",
            "filename": "VOCtrainval_06-Nov-2007.tar",
            "md5": "c52e279531787c972589f7e41ab4ae64",
            "base_dir": Path("VOCdevkit", "VOC2007"),
            "train": Path("VOCdevkit", "VOC2007", "ImageSets", "Layout", "train.txt"),
        },
    },
    "2007-test": {
        "trainval": {
            "url": "http://host.robots.ox.ac.uk/pascal/VOC/voc2007/VOCtest_06-Nov-2007.tar/VOCtest_06-Nov-2007.tar",
            "filename": "VOCtest_06-Nov-2007.tar",
            "md5": "b6e924de25625d8de591ea690078ad9f",
            "base_dir": Path("VOCdevkit", "VOC2007"),
        },
    },
}

# Dictionary for mapping 'data_dir_name' to a set of dataset assets
DATASETS = {'coco': COCO, 'VOCdevkit': VOC, 'TrainVal': VOC}

# Parameters
IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])

# Get orientation exif tag
for k, v in ExifTags.TAGS.items():
    if v == "Orientation":
        ORIENTATION = k
        break


def usable_cpus():
    """Return number of cpus configured to be used by this process."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        # MacOS does not provide sched_getaffinity, but cpu_count is good enough
        return os.cpu_count()


def coco80_to_coco91_table():
    # coco has 91-index in their paper but takes 80-index in val2017 dataset
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
    coco91 = list(range(1, 92))
    missing_classes = {12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91}
    return [x for x in coco91 if x not in missing_classes]


def coco91_to_coco80_table():
    coco80 = coco80_to_coco91_table()
    # Initialize an array of -1's
    coco91_to_coco80 = np.full(91, -1)

    # Populate indices with their corresponding classes
    for i, cls in enumerate(coco80):
        coco91_to_coco80[cls - 1] = i

    return coco91_to_coco80


def _filter_valid_image_paths(data_root, annotation_file, img_dir_name, task_enum):
    """
    Filter image paths based on the annotations in the JSON or XML file.

    Args:
        data_root (Path): The root directory of the dataset.
        annotation_file (Path): The path to the annotation file (JSON or XML).
        img_dir_name (str): The name of the directory containing the images.
        task_enum (SupportedTaskCategory): The task category (e.g., ObjDet, Seg, Kpts).

    Returns:
        List[Path]: A list of valid image paths.
    """
    valid_filenames = set()

    if annotation_file.suffix == '.json':
        with open(annotation_file, 'r') as f:
            json_data = json.load(f)
        if task_enum == SupportedTaskCategory.Kpts:
            valid_filenames = {
                img['file_name']
                for img in json_data['images']
                if any(
                    ann['image_id'] == img['id'] and ann['num_keypoints'] > 0
                    for ann in json_data['annotations']
                )
            }
        else:
            valid_filenames = {img['file_name'] for img in json_data['images']}
    elif annotation_file.suffix == '.xml':
        import xml.etree.ElementTree as ET

        tree = ET.parse(annotation_file)
        root = tree.getroot()
        for image in root.findall('image'):
            file_name = image.get('file_name')
            valid_filenames.add(file_name)
    else:
        raise ValueError(f"Unsupported annotation file format: {annotation_file.suffix}")

    LOG.trace(f"Number of valid filenames from annotations: {len(valid_filenames)}")
    img_paths = [
        Path(data_root, img_dir_name, p)
        for p in os.listdir(Path(data_root, img_dir_name))
        if p in valid_filenames
    ]
    LOG.trace(f"Valid image number: {len(img_paths)}")
    return img_paths


def _segments2polygon(segments):
    """
    Convert segment labels to polygon format for COCO annotations.

    Args:
        segments (List[np.ndarray]): List of segments where each segment is an array of points (x, y).

    Returns:
        List[List[float]]: List of polygons where each polygon is a list of points in the format [x1, y1, x2, y2, ...].
    """
    polygons = []
    for segment in segments:
        polygon = segment.flatten().tolist()
        polygons.append(polygon)
    return polygons


def _segments2boxes(segments):
    # Convert segment labels to box labels, i.e. (cls, xy1, xy2, ...) to (cls, xywh)
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))  # cls, xywh


def download_custom_dataset(data_root, **kwargs):
    if 'dataset_url' in kwargs:
        utils.download_and_extract_asset(
            kwargs['dataset_url'],
            data_root / kwargs['dataset_url'].split('/')[-1],
            md5=kwargs.get('dataset_md5', None),
            drop_dirs=kwargs.get('dataset_drop_dirs', 0),
        )


def _create_image_list_file(input_path, subdir=None):
    """
    Create a temporary text file listing all images from its 'images' subdirectory.

    Args:
        input_path: Path to a directory or a file
        subdir: Optional subdirectory name to look for (default: "images")

    Returns:
        Path to the file to use (either the original file or a newly created temp file)
    """
    path = Path(input_path)
    if path.is_file():  # If it's a file, just return it
        return path

    if path.is_dir():  # If it's a directory, first check for images subdirectory
        if subdir is None:
            subdir = "images"

        images_dir = path / subdir

        # If the specified subdirectory doesn't exist, use the original path
        if not images_dir.is_dir():
            images_dir = path

        temp_file = tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.txt')
        temp_path = Path(temp_file.name)

        # Find all image files and write their paths to the temp file
        image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        file_count = 0

        with temp_file:
            for image_file in images_dir.glob('**/*'):
                if image_file.suffix.lower() in image_extensions:
                    rel_path = f"{image_file.absolute()}"
                    temp_file.write(f"{rel_path}\n")
                    file_count += 1

        if file_count == 0:
            temp_path.unlink()
            raise FileNotFoundError(f"No image files found in {images_dir}")

        return temp_path
    raise FileNotFoundError(f"Path {path} is neither a file nor a directory")


class SupportedTaskCategory(enum.Enum):
    ObjDet = enum.auto()  # object detection
    Seg = enum.auto()  # instance segmentation
    Kpts = enum.auto()  # keypoint detection


class SupportedLabelType(enum.Enum):
    NONE = enum.auto()  # None for default datasets like COCO2017
    YOLOv8 = enum.auto()
    YOLOv5 = enum.auto()
    PascalVOCXML = enum.auto()
    COCOJSON = enum.auto()

    @classmethod
    def from_string(cls, label_type_str: str) -> 'SupportedLabelType':
        mapping = {
            'YOLOv8': cls.YOLOv8,
            'YOLOv5': cls.YOLOv5,
            'Pascal VOC XML': cls.PascalVOCXML,
            'Pascal_VOC_XML': cls.PascalVOCXML,
            'COCO JSON': cls.COCOJSON,
            'COCO_JSON': cls.COCOJSON,
            'yolov8': cls.YOLOv8,
            'yolov5': cls.YOLOv5,
            'pascal voc xml': cls.PascalVOCXML,
            'pascal_voc_xml': cls.PascalVOCXML,
            'coco json': cls.COCOJSON,
            'coco_json': cls.COCOJSON,
        }

        if label_type_str not in mapping:
            raise ValueError(f"Unsupported label_type: {label_type_str}")

        return mapping[label_type_str]


class CocoAndYoloFormatDataset(Dataset):
    # labels caching version; bump up when changing labeling or caching method
    cache_version = 0.3

    def __init__(
        self,
        data_root: Path,
        split='val',  # 'val' returns (image, label), 'test' returns image only
        transform=None,
        output_format='xyxy',  # 'xyxy', 'xywh' or 'ltwh'
        labels_path: str = None,
        task=SupportedTaskCategory.ObjDet,
        label_type=SupportedLabelType.YOLOv8,
        **kwargs,
    ):
        t1 = time.time()
        assert split.lower() in ("train", "val", "test"), f"Unsupported split: {split}"
        assert output_format.lower() in (
            "xyxy",
            "xywh",
            "ltwh",
        ), f"Unsupported output format: {output_format}"
        self.output_format = output_format
        self.task_enum = task
        self.__dict__.update(locals())
        self.img_paths = []
        self.labels = []
        self.transform = transform

        # Configure dataset
        data_dir, reference_file, label_format = self._configure_data(data_root, **kwargs)

        # This is specific for pycocotools measurement
        self._groundtruth_json = None
        if reference_file.suffix == '.json':
            self._groundtruth_json = reference_file
        year = kwargs.get('download_year', None)
        if (label_format == 'coco80') and year:
            # find 'instances_<split><year>.json' in data_dir and its subdirectories
            if self.task_enum == SupportedTaskCategory.Kpts:
                self._groundtruth_json = next(
                    data_dir.glob(f'**/person_keypoints_{split}{year}.json'), None
                )
                # self._groundtruth_json = None
            else:  # SupportedTaskCategory.ObjDet or SupportedTaskCategory.Seg
                self._groundtruth_json = next(
                    data_dir.glob(f'**/instances_{split}{year}.json'), None
                )
        self.label_type = SupportedLabelType.NONE if year else label_type

        # Get label format
        if "format" in kwargs:
            if label_format and label_format != kwargs["format"]:
                LOG.warn(f"Overriding '{label_format}' format specified for downloaded dataset")
            label_format = kwargs["format"]

        # Get class names from labels
        class_names = utils.load_labels(labels_path) if labels_path else None

        # TODO: Implement 'test' path (assume working on 'val' for now)
        (
            self.img_paths,
            self.labels,
            self.segments,
            self.image_ids,
            gt_json,
        ) = self._get_imgs_labels(
            data_dir, reference_file, label_format, class_names, output_format.lower(), **kwargs
        )
        if not len(self.labels) and self.split == 'val':
            raise RuntimeError("Validation requested, but no labels available")

        # if self.task_enum == SupportedTaskCategory.Kpts:
        #     self._groundtruth_json = gt_json

        self.total_frames = len(self.img_paths)
        t2 = time.time()
        LOG.debug(f"Dataset initialization completed in %.1fs" % (t2 - t1))

    def __len__(self):
        return self.total_frames

    def __getitem__(self, idx):
        sample: Dict = {}
        path = self.img_paths[idx]
        img_id = self.image_ids[idx]

        try:
            label = self.labels(idx).name
        except TypeError:
            label = self.labels[idx]
        img = Image.open(path)
        if img is None:
            raise RuntimeError(f"Image not found {path}, workdir: {Path.cwd().as_posix()}")
        elif img.mode == "L":
            img = img.convert("RGB")
        elif img.mode != "RGB":
            raise ValueError(f"Unsupported PIL image mode: {img.mode}")

        raw_w, raw_h = img.size
        if self.transform:
            img = self.transform(img)
        sample["image"] = img
        sample["image_id"] = img_id

        if self.split in ["train", "val"]:
            label = np.array(label, dtype=np.float32)
            if len(label) > 0:
                sample['bboxes'] = torch.from_numpy(label[:, 1:5])
                sample["category_id"] = torch.from_numpy(label[:, 0])
                if self.task_enum == SupportedTaskCategory.Kpts:
                    label = label[:, 5:].reshape(label.shape[0], -1, 3)
                    sample['keypoints'] = torch.from_numpy(label)
                elif self.task_enum == SupportedTaskCategory.Seg:
                    # return unscaled polygons to save memory
                    # we will convert them to masks if evaluator asks for
                    sample["polygons"] = self.segments[idx]
            else:
                sample["bboxes"] = torch.as_tensor([])
                sample["category_id"] = torch.as_tensor([])
                if self.task_enum == SupportedTaskCategory.Kpts:
                    sample['keypoints'] = torch.as_tensor([])
                elif self.task_enum == SupportedTaskCategory.Seg:
                    # polygons is a list; it cannot convert as tensor because
                    # the segments have varying lengths
                    sample['polygons'] = []
            sample["raw_w"] = raw_w
            sample["raw_h"] = raw_h
        return sample

    def _configure_data(self, data_root, **kwargs):
        # Configure/download data needed for specified split
        reference_file = None
        data_dir = YAML.attribute(kwargs, "data_dir_name")
        label_format = None
        if data_dir in DATASETS:
            if "download_year" in kwargs:
                year = str(kwargs["download_year"])
                if year in DATASETS[data_dir]:
                    data = DATASETS[data_dir][year]
                    label_format = data["format"] if "format" in data else None
                    if self.split in data:
                        self.download_and_extract_asset(data_root, data[self.split])
                        if self.split + '_reference' in data:
                            self.download_and_extract_asset(
                                data_root, data[self.split + '_reference']
                            )
                    elif 'trainval' in data:
                        self.download_and_extract_asset(data_root, data['trainval'])
                    if self.split == 'train' or self.split == 'val':
                        if 'annotations' in data:
                            self.download_and_extract_asset(data_root, data['annotations'])
                        if 'labels' in data:
                            self.download_and_extract_asset(data_root, data['labels'])
                    else:
                        raise RuntimeError(f"Unsupported split '{self.split}'")
                    # Get settings from data dictionary unless specified by user
                    if not self.split in kwargs:
                        if self.split in data and self.split in data[self.split]:
                            reference_file = data[self.split][self.split]
                        elif "trainval" in data and self.split in data["trainval"]:
                            reference_file = data["trainval"][self.split]

                    if self.task_enum == SupportedTaskCategory.Kpts:
                        self.download_and_extract_asset(data_root, data['kpts'])
                        if year == '2014':
                            raise KeyError(f"Unsupported year: {year} for keypoint detection")
                        reference_file = Path('labels_kpts', f'{self.split}{year}.txt')
                    else:  # Seg or ObjDet
                        if self.task_enum == SupportedTaskCategory.Seg:
                            self.download_and_extract_asset(data_root, data['seg'])

                        # update the train.txt or val.txt according to the local data
                        from ax_datasets.tools.gen_image_path_list import list_images_to_a_file

                        list_images_to_a_file(
                            data_root / data[self.split]['base_dir'],
                            data_root / data[self.split][self.split],
                        )
                else:
                    raise AxYAMLError(
                        f'Dataset {data_dir} has no data available for year {year}',
                        YAML.key(kwargs, 'download_year'),
                    )
            else:
                LOG.debug(
                    f"No 'download_year' attribute specified for {data_dir} dataset (data will not be automatically downloaded"
                )

        if not reference_file:
            if self.split == 'train':
                if 'cal_data' not in kwargs:
                    raise AxYAMLError(
                        f"Please specify 'cal_data' for {data_dir} dataset if you want to calibrate with your custom data",
                        YAML.key(kwargs, 'cal_data'),
                    )
                if not (data_root / kwargs['cal_data']).is_file():
                    download_custom_dataset(data_root, **kwargs)
            elif self.split == 'val':
                if 'val_data' not in kwargs:
                    raise AxYAMLError(
                        f"Please specify 'val_data' for {data_dir} dataset if you want to evaluate with your custom data",
                        YAML.key(kwargs, 'val_data'),
                    )
                if not (data_root / kwargs['val_data']).is_file():
                    download_custom_dataset(data_root, **kwargs)
            reference_file = (
                kwargs.get('cal_data') if self.split == 'train' else kwargs.get('val_data')
            )

        reference_file = Path(data_root, reference_file)
        reference_file = _create_image_list_file(reference_file)
        if not reference_file.is_file():
            raise AxYAMLError(f"{reference_file}: File not found", YAML.key(kwargs, self.split))
        return Path(data_root), reference_file, label_format

    def _get_imgs_labels(
        self, data_root, reference_file, label_format, class_names, output_format, **kwargs
    ):
        # Return lists of image paths and labels in output_format (coco or yolo)
        img_paths = []

        # Images and labels are assumed to be stored in separate directories
        # unless specified otherwise
        is_label_image_same_dir = False
        if "is_label_image_same_dir" in kwargs:
            is_label_image_same_dir = kwargs["is_label_image_same_dir"]

        # Images can be taken from various sources including JSON
        # annotations (COCO format) or a reference file with .txt
        # or .part formats, or a directory
        # TODO check later integration with measurement code
        self.label_tag = None
        if reference_file.suffix == '.json':
            self.label_tag = f'{data_root.stem}_labels_{self.task_enum.name.lower()}'
            if self.split == 'val':
                if 'val_img_dir_name' not in kwargs:
                    raise AxYAMLError(
                        f"Please specify 'val_img_dir_name' for {data_dir} dataset if you want to evaluate with your custom data",
                        YAML.key(kwargs, 'val_img_dir_name'),
                    )
                img_paths = _filter_valid_image_paths(
                    data_root, reference_file, kwargs['val_img_dir_name'], self.task_enum
                )
            elif self.split == 'train':
                if 'cal_img_dir_name' not in kwargs:
                    raise AxYAMLError(
                        f"Please specify 'cal_img_dir_name' for {data_dir} dataset if you want to calibrate with your custom data",
                        YAML.key(kwargs, 'cal_img_dir_name'),
                    )
                img_paths = _filter_valid_image_paths(
                    data_root, reference_file, kwargs['cal_img_dir_name'], self.task_enum
                )

            first_file = CocoAndYoloFormatDataset.replace_last_match_dir(
                img_paths[0], 'images', self.label_tag
            )
            if not first_file.is_file():
                self.coco2yolo_format_labels(str(reference_file), first_file.parent)
            label_paths = self.image2label_paths(
                img_paths, is_label_image_same_dir, self.label_tag
            )
        elif reference_file.suffix in (".txt", ".part"):
            with open(reference_file, "r", encoding="UTF-8") as reader:
                while line := reader.readline().rstrip():
                    line = Path(line)
                    if reference_file.suffix == ".part" and line.is_absolute():
                        img_paths.append(Path(reference_file.parent, *line.parts[1:]))
                    elif line.is_absolute():
                        img_paths.append(line)
                    else:
                        img_paths.append(reference_file.parent / line)
            if reference_file.suffix == ".part":
                img_paths = [Path(data_root, s) for s in img_paths]

            if self.label_tag is None:
                self.label_tag = "labels"
                if self.label_type == SupportedLabelType.NONE:
                    if self.task_enum == SupportedTaskCategory.Kpts:
                        self.label_tag = "labels_kpts"
                    elif self.task_enum == SupportedTaskCategory.Seg:
                        self.label_tag = "labels_seg"

            # we sort the image paths to make sure the order of images and labels are the same,
            # this is important for CI test
            img_paths.sort()
            label_paths = self.image2label_paths(
                img_paths, is_label_image_same_dir, self.label_tag
            )

            if not label_paths[0].is_file() and label_paths[0].with_suffix('.xml').is_file():
                # Convert VOC to YOLO labels; TODO: verify the implementation
                if self.split == 'val':
                    img_paths = _filter_valid_image_paths(
                        data_root, reference_file, kwargs['val_img_dir_name'], self.task_enum
                    )
                elif self.split == 'train':
                    img_paths = _filter_valid_image_paths(
                        data_root, reference_file, kwargs['cal_img_dir_name'], self.task_enum
                    )
                xml_paths = [lp.with_suffix('.xml') for lp in label_paths]
                first_file = CocoAndYoloFormatDataset.replace_last_match_dir(
                    img_paths[0], 'images', self.label_tag
                )
                label_paths = self.voc2yolo_format_labels(
                    xml_paths, first_file.parent, class_names
                )
        elif reference_file.is_dir():
            img_paths = [path for path in reference_file.rglob("*")]
        else:
            raise ValueError(f"{reference_file}: Unsupported format")

        img_paths = [p for p in img_paths if p.is_file() and p.suffix[1:].lower() in IMG_FORMATS]
        if not img_paths:
            raise AxYAMLError(
                f'No supported images found in {reference_file}', YAML.key(kwargs, self.split)
            )

        cache_file = Path(
            label_paths[0].parent,
            f"{self.split}_{data_root.stem}_{self.task_enum.name.lower()}.cache",
        )
        img_paths, shapes, labels, segments, from_cache = self.__cache_and_verify_dataset(
            cache_file, img_paths, label_paths, output_format
        )
        image_ids = self._collect_image_ids([p.stem for p in img_paths])

        # Convert COCO-80 to COCO-91 classes if required
        if label_format in ["coco91", "coco91-with-bg"]:
            shift = 0 if label_format == "coco91-with-bg" else 1
            for sample_labels in labels:
                for label in sample_labels:
                    label[0] = coco80_to_coco91_table()[int(label[0])] - shift
            from_cache = False
        # Generate JSON of the corrected ground truth if not existing
        # or if the cache was regenerated (required for measurement)
        gt_json = Path(data_root, f"{data_root.stem}_{self.task_enum.name.lower()}_gt.json")
        if not gt_json.is_file() or not from_cache:
            coco_80_to_91 = label_format == "coco80"
            self.dump_coco_format_json(
                class_names,
                img_paths,
                image_ids,
                shapes,
                labels,
                gt_json,
                current_format=output_format,
                coco80_to_coco91=coco_80_to_91,
            )
        return img_paths, labels, segments, image_ids, gt_json

    def __load_cache(self, cache_path, hash):
        try:
            cache = pickle.load(open(cache_path, "rb"))
            if (cache["version"] != self.cache_version) or (cache["hash"] != hash):
                cache = {}
        except:
            cache = {}
        return cache

    def __cache_and_verify_dataset(self, cache_path, img_paths, label_paths, output_format):
        def extract_output(cache):
            # remove useless items before efficient parsing lists
            [cache.pop(k) for k in ("hash", "version", "status")]
            labels, shapes, segments = zip(*cache.values())
            labels = list(labels)
            segments = list(segments)
            shapes = np.array(shapes, np.int32)
            qualified_img_files = list(cache.keys())
            return qualified_img_files, shapes, labels, segments

        nm, nf, ne, nc = 0, 0, 0, 0
        data_status = (
            "{0} labels found, {1} corrupt images\n{2} background images: "
            "{3} no label files, {4} empty label files"
        )  # nf, nc, nm+ne, nm, ne

        hash = self._get_hash(img_paths + label_paths, output_format)

        # Load data from cache if present
        cache = self.__load_cache(cache_path, hash)
        if cache:
            LOG.debug(f"Load cache from {cache_path}:")
            nm, nf, ne, nc = cache["status"]
            status = data_status.format(nf, nc, nm + ne, nm, ne)
            for line in iter(str(status).splitlines()):
                LOG.debug('  ' + line)
            return *extract_output(cache), True

        # No cache present; first create
        nthreads = usable_cpus()
        LOG.debug(f"Create new label cache file {cache_path}")
        LOG.trace(f"Spawn {nthreads} threads to speedup cache creation")
        with Pool(nthreads) as pool:
            pbar = pool.imap(
                self.check_image_label,
                zip(img_paths, label_paths),
            )
            pbar = tqdm(
                pbar, total=len(label_paths), desc=f"Create label cache", unit='label', leave=False
            )
            for (
                img_path,
                labels_per_file,
                img_shape,
                segments_per_file,
                nc_per_file,
                nm_per_file,
                nf_per_file,
                ne_per_file,
            ) in pbar:
                if nc_per_file == 0:
                    cache[img_path] = [labels_per_file, img_shape, segments_per_file]
                nc += nc_per_file
                nm += nm_per_file
                nf += nf_per_file
                ne += ne_per_file
        pbar.close()
        status = data_status.format(nf, nc, nm + ne, nm, ne)
        for line in iter(str(status).splitlines()):
            LOG.debug("  " + line)
        if nf == 0:
            raise RuntimeError(f"No valid image/label pairs found in dataset")

        # Write cache file
        cache['hash'] = hash
        cache['version'] = self.cache_version
        cache['status'] = nm, nf, ne, nc
        try:
            pickle.dump(cache, open(cache_path, "wb"))
        except Exception as e:
            LOG.warning(f"Failed to write cache file {cache_path}: {e}")
        return *extract_output(cache), False

    def check_image_label(self, args):
        img_path, lb_path = args
        nc = 0  # number of corrupt image
        nm, nf, ne = 0, 0, 0  # number of labels (missing/found/empty)
        segments = []

        # Check images with PIL
        def exif_size(img):
            # Returns exif-corrected PIL size
            s = img.size  # (width, height)
            im_exif = img._getexif()
            if im_exif and ORIENTATION in im_exif:
                rotation = im_exif[ORIENTATION]
                if rotation in [6, 8]:  # rotation 270 or 90
                    s = (s[1], s[0])
            return s

        try:
            img = Image.open(img_path)
            img.verify()
            w, h = exif_size(img)
            assert (w > 9) and (h > 9), f"image w:{w} or h:{h} <10 pixels"
            assert img.format.lower() in IMG_FORMATS, f"invalid image format {img.format}"
            if img.format.lower() in ("jpg", "jpeg"):
                with open(img_path, "rb") as f:
                    f.seek(-2, 2)
                    if f.read() != b"\xff\xd9":  # corrupt JPEG
                        ImageOps.exif_transpose(Image.open(img_path)).save(
                            img_path, "JPEG", subsampling=0, quality=100
                        )
                        LOG.warn(f"{img_path}: corrupt JPEG restored and saved")
        except Exception as e:
            nc = 1
            LOG.warn(f"Ignoring image {img_path}: {e}")
            return img_path, None, None, None, nc, nm, nf, ne

        # Validate labels in label file
        try:
            if lb_path.is_file():
                nf = 1  # label found
                with open(lb_path, "r") as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if self.task_enum == SupportedTaskCategory.Seg and any(len(l) > 6 for l in lb):
                        classes = np.array([l[0] for l in lb], dtype=np.float32)
                        segments = [
                            np.array(l[1:], dtype=np.float32).reshape(-1, 2) for l in lb
                        ]  # (cls, xy1...)
                        lb = np.concatenate(
                            (
                                classes.reshape(-1, 1),
                                _segments2boxes(segments),
                            ),
                            1,
                        )  # (cls, xywh)
                    lb = np.array(lb, dtype=np.float32)
                if len(lb):
                    assert lb.shape[1] >= 5, f"each row required at least 5 values"
                    assert (lb >= 0).all(), f"all values in label file must > 0"
                    assert (lb[:, 1:5] <= 1).all(), f"found unnormalized coordinates"
                    _, idx = np.unique(lb, axis=0, return_index=True)
                    if len(idx) < len(lb):  # if duplicate row
                        lb = lb[idx]  # remove duplicates
                        LOG.warn(f"{lb_path}: {len(lb) - len(idx)} duplicate labels removed")
                    # lb should be with normalized xywh format now; here we always convert to absolute coordinates
                    if self.output_format == "ltwh":
                        lb[:, 1:5] = xywh2ltwh(lb[:, 1:5])
                        lb[:, [1, 3]] *= w
                        lb[:, [2, 4]] *= h
                    elif self.output_format == "xyxy":
                        lb[:, 1:5] = xywh2xyxy(lb[:, 1:5])
                        lb[:, [1, 3]] *= w
                        lb[:, [2, 4]] *= h
                    elif self.output_format == "xywh":
                        lb[:, [1, 3]] *= w
                        lb[:, [2, 4]] *= h

                    if self.task_enum == SupportedTaskCategory.Kpts:
                        lb[:, 5::3] *= w
                        lb[:, 6::3] *= h
                        # 7::3 is visibility, 0: not labeled, 1: labeled but not visible, and 2: labeled and visible.
                    if segments:
                        # scale segments
                        segments = [seg * [w, h] for seg in segments]
                    lb = lb.tolist()
                else:
                    ne = 1  # label empty
                    lb = []
            else:
                nm = 1  # label missing
                lb = []
            return img_path, lb, (w, h), segments, nc, nm, nf, ne
        except Exception as e:
            LOG.warn(f"{lb_path}: ignoring invalid labels: {e}")
            return img_path, None, None, None, nc, nm, nf, ne

    @staticmethod
    def replace_last_match_dir(path: Path, old, new):
        parts = path.parts[::-1]
        index = parts.index(old)
        index = len(parts) - index - 1
        return Path(*path.parts[0:index], new, *path.parts[index + 1 :])

    @staticmethod
    def image2label_paths(img_paths, is_label_image_same_dir, tag="labels"):
        # the order in the list of label paths must must correspond to image files
        if is_label_image_same_dir:
            LOG.info("Please ensure all images and labels are under the same folder")
            LOG.info("with the same naming rule (only differing in their file extension)")
            label_files = [p.with_suffix(".txt") for p in img_paths]
        else:
            # Replace /images/ with /labels/ (last instance)
            label_files = []
            for x in img_paths:
                label_files.append(
                    CocoAndYoloFormatDataset.replace_last_match_dir(x, 'images', tag).with_suffix(
                        '.txt'
                    )
                )
        return label_files

    @staticmethod
    def evaluate(gt_json, pred_json, img_ids=[], metrics=["mAP", "mAP50"]):
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval

        # implement this by using COCOeval
        # return a dict mapping each metric to a value
        # assert all(m in supported_metrics for m in metrics)
        results = dict.fromkeys(metrics, 0)
        try:
            gt = COCO(gt_json)
            pred = gt.loadRes(pred_json)
            eval = COCOeval(gt, pred, iouType="bbox")

            if len(img_ids) > 0:  # image IDs to evaluate
                eval.params.imgIds = img_ids
            eval.evaluate()
            eval.accumulate()
            eval.summarize()
            map, map50 = eval.stats[:2]
            if "mAP" in results:
                results["mAP"] = map
            if "mAP50" in results:
                results["mAP50"] = map50
        except Exception as e:
            print("Failed to evaulate by pycocotools")
        # TODO: print each class ap from pycocotool result
        # return dictionary by "metrics"
        # TODO: implement precision and recall of single class
        # tp, fp, p, r, f1, ap, ap_class = evaluate_per_class(*stats)
        # ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        # mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        return results

    @staticmethod
    def label_to_coco_bbox(format, label, img_w, img_h):
        # convert label to coco bbox format
        # input: label in "format" format
        # output: label in coco bbox format
        if format == "ltwh":
            c, x1, y1, w, h = label[:5]
        elif format == "xyxy":
            c, x1, y1, x2, y2 = label[:5]
            w = x2 - x1
            h = y2 - y1
        elif format == 'xywh':
            c, x, y, w, h = label[:5]
            x1 = (x - w / 2) * img_w
            y1 = (y - h / 2) * img_h
            w *= img_w
            h *= img_h
        else:
            raise Exception(f"Unsupported format: {format}")
        return int(c), x1, y1, w, h

    def dump_coco_format_json(
        self,
        class_names,
        img_paths,
        image_ids,
        shapes,
        images_labels,
        save_to,
        current_format,
        coco80_to_coco91=False,
    ):
        # Input is YOLO format
        assert len(img_paths) == len(shapes) == len(images_labels)
        ann_id = 0
        LOG.debug(f"Save labels to JSON file (for future measurements): {save_to}")
        dataset = {"categories": [], "annotations": [], "images": []}
        if class_names:
            for i, class_name in enumerate(class_names):
                dataset["categories"].append({"id": i, "name": class_name, "supercategory": ""})
        for img_path, img_id, shape, labels in tqdm(
            zip(img_paths, image_ids, shapes, images_labels), leave=False
        ):
            img_w, img_h = shape.tolist()
            dataset["images"].append(
                {
                    "file_name": img_path.name,
                    "id": img_id,
                    "width": img_w,
                    "height": img_h,
                }
            )
            for label in labels:
                if self.task_enum == SupportedTaskCategory.Kpts:
                    keypoints = label[5:]
                    label = label[:5]
                elif self.task_enum == SupportedTaskCategory.Seg:
                    segments = label[5:]
                    label = label[:5]
                c, x1, y1, w, h = CocoAndYoloFormatDataset.label_to_coco_bbox(
                    current_format, label, img_w, img_h
                )
                if coco80_to_coco91:
                    c = coco80_to_coco91_table()[c]

                anno = {
                    "area": h * w * 0.53,
                    "bbox": [x1, y1, w, h],
                    "category_id": c,
                    "id": ann_id,
                    "image_id": img_id,
                    "iscrowd": 0,
                    # mask
                    "segmentation": [],
                }
                if self.task_enum == SupportedTaskCategory.Kpts:
                    anno["keypoints"] = keypoints
                    anno["num_keypoints"] = sum(
                        1 for i in range(2, len(keypoints), 3) if keypoints[i] > 0
                    )
                elif self.task_enum == SupportedTaskCategory.Seg:
                    anno["segmentation"] = _segments2polygon(segments)
                dataset["annotations"].append(anno)
                ann_id += 1
        with open(save_to, "w") as f:
            json.dump(dataset, f)

    @staticmethod
    def voc2yolo_format_labels(xml_paths: List[str], save_base_path: Path, classes: List[str]):
        import xml.etree.ElementTree as ET

        def convert(size, box):
            dw = 1.0 / (size[0])
            dh = 1.0 / (size[1])
            x = (box[0] + box[1]) / 2.0 - 1
            y = (box[2] + box[3]) / 2.0 - 1
            w = box[1] - box[0]
            h = box[3] - box[2]
            return (x * dw, y * dh, w * dw, h * dh)

        LOG.debug(f"Label folder is {save_base_path}; write xmls")
        save_base_path.mkdir(parents=True, exist_ok=True)
        label_files = []
        for a_path in tqdm(xml_paths, leave=False):
            # Read annotation xml
            ann_tree = ET.parse(a_path)
            ann_root = ann_tree.getroot()
            size = ann_root.find("size")
            w = int(size.find("width").text)
            h = int(size.find("height").text)
            out_file = save_base_path / f"{Path(a_path).stem}.txt"
            with out_file.open(mode="w", encoding="utf-8") as f:
                for obj in ann_root.iter("object"):
                    difficult = obj.find("difficult").text
                    cls = obj.find("name").text
                    if classes:
                        if cls not in classes or int(difficult) == 1:
                            LOG.warning(
                                f'{cls} is not in label name file or" \
                                        " difficult={int(difficult)}!'
                            )
                            continue
                        cls_id = classes.index(cls)
                    else:
                        cls_id = cls
                    xmlbox = obj.find('bndbox')
                    b = (
                        float(xmlbox.find('xmin').text),
                        float(xmlbox.find('xmax').text),
                        float(xmlbox.find('ymin').text),
                        float(xmlbox.find('ymax').text),
                    )
                    bb = convert((w, h), b)
                    f.write(str(cls_id) + " " + " ".join([str(a) for a in bb]) + '\n')
            label_files.append(str(out_file))
        return label_files

    @staticmethod
    def coco2yolo_format_labels(annotation_path: str, save_base_path: Path):
        from pycocotools.coco import COCO

        LOG.debug(f"Label folder is {save_base_path}")
        save_base_path.mkdir(parents=True, exist_ok=True)
        LOG.debug(f"Load annotations from {annotation_path}")
        data_source = COCO(annotation_file=annotation_path)
        catIds = data_source.getCatIds()
        categories = data_source.loadCats(catIds)
        categories.sort(key=lambda x: x["id"])
        classes = {}
        coco_labels = {}
        coco_labels_inverse = {}
        for c in categories:
            coco_labels[len(classes)] = c["id"]
            coco_labels_inverse[c["id"]] = len(classes)
            classes[c["name"]] = len(classes)

        img_ids = data_source.getImgIds()
        for _, img_id in tqdm(
            enumerate(img_ids), desc='Convert .json file to .txt files', leave=False
        ):
            img_info = data_source.loadImgs(img_id)[0]
            file_name = img_info["file_name"].split(".")[0]
            height = img_info["height"]
            width = img_info["width"]

            save_path = save_base_path / f"{file_name}.txt"
            with save_path.open(mode="w", encoding="utf-8") as fp:
                annotation_id = data_source.getAnnIds(img_id)
                boxes = np.zeros((0, 5))
                if len(annotation_id) == 0:
                    fp.write("")
                    continue
                annotations = data_source.loadAnns(annotation_id)
                lines = ""
                for annotation in annotations:
                    box = annotation["bbox"]
                    # some annotations have basically no width / height, skip them
                    if box[2] < 1 or box[3] < 1:
                        continue
                    # top_x,top_y,width,height---->cen_x,cen_y,width,height
                    box[0] = round((box[0] + box[2] / 2) / width, 6)
                    box[1] = round((box[1] + box[3] / 2) / height, 6)
                    box[2] = round(box[2] / width, 6)
                    box[3] = round(box[3] / height, 6)
                    label = coco_labels_inverse[annotation["category_id"]]
                    lines = lines + str(label)
                    for i in box:
                        lines += " " + str(i)

                    # Add keypoints if available
                    if "keypoints" in annotation:
                        keypoints = np.array(annotation["keypoints"], dtype=np.float32).reshape(
                            -1, 3
                        )
                        keypoints[..., 0] /= width  # normalize x
                        keypoints[..., 1] /= height  # normalize y
                        keypoints = keypoints.reshape(-1).tolist()
                        lines += " " + " ".join(map(str, keypoints))
                    elif "segmentation" in annotation:
                        segments = annotation["segmentation"]
                        yolo_segments = [
                            f"{(x) / width:.5f} {(y) / height:.5f}"
                            for x, y in zip(segments[0][::2], segments[0][1::2])
                        ]
                        yolo_segments = ' '.join(yolo_segments)
                        lines += " " + yolo_segments
                    lines += "\n"
                fp.writelines(lines)

    def _get_hash(self, paths: List[Path], output_format: str) -> str:
        """Get the hash value of paths"""
        size = sum(x.stat().st_size for x in paths if x.exists())
        h = hashlib.md5(str(size).encode())  # hashing sizes
        h.update(str(sorted([p.as_posix() for p in paths])).encode())  # hashing sorted paths
        h.update(output_format.encode())  # hashing output format
        return h.hexdigest()

    def _collect_image_ids(self, filenames: List[str]) -> List[int]:
        image_ids = []
        for filename in filenames:
            if filename.isnumeric():
                image_ids.append(int(filename))
            else:
                image_id = int(hashlib.sha256(filename.encode("utf-8")).hexdigest(), 16) % 10**8
                if image_id in image_ids:
                    # If the image_id is a duplicate, generate a new unique id
                    new_id = max(image_ids) + 1
                    image_ids.append(new_id)
                else:
                    image_ids.append(image_id)
        return image_ids

    @staticmethod
    def download_and_extract_asset(data_root, asset):
        # Download file and, if archive, extract
        # Archives are represented with base_dir attributes
        filename = Path(data_root, asset['filename'])
        if not 'base_dir' in asset:
            if not filename.exists():
                utils.download(asset['url'], filename, asset['md5'])
        elif utils.dir_needed(
            Path(data_root, asset['base_dir'])
        ) or CocoAndYoloFormatDataset.missing_files(data_root, asset):
            # Download archive (implied by base_dir)
            drop_dirs = asset["drop_dirs"] if "drop_dirs" in asset else 0
            utils.download(asset["url"], filename, asset["md5"])
            utils.extract(filename, drop_dirs)
            filename.unlink()

    @staticmethod
    def missing_files(data_root, asset):
        # For shared directories, check if any
        # key files/subdirs are missing
        if "check_files" in asset:
            for file in asset["check_files"]:
                if not Path(data_root, file).exists():
                    return True
        return False

    @property
    def groundtruth_json(self):
        return str(self._groundtruth_json) if self._groundtruth_json else ''


class ObjDataAdaptor(types.DataAdapter):
    """Data adapter for tasks which follow YOLO format.

    Args:
        root (str): Root directory of the dataset.utils
        split (str): Split of the dataset, either 'val' or 'test', for validation
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        batch_size (int, optional): How many samples per batch to load.
    """

    def __init__(self, dataset_config, model_info):
        if 'download_year' not in dataset_config:
            if 'cal_data' not in dataset_config and 'repr_imgs_dir_name' not in dataset_config:
                raise ValueError(
                    f"Please specify either 'repr_imgs_dir_name' or 'cal_data' for {self.__class__.__name__} "
                    f"if you want to deploy with your custom data, or 'download_year' for COCO dataset"
                )
            if 'val_data' not in dataset_config:
                raise ValueError(
                    f"Please specify 'val_data' for {self.__class__.__name__} "
                    f"if you want to evaluate with your custom data, or 'download_year' for using COCO dataset"
                )
        self.label_type = self._check_supported_label_type(dataset_config)

    def _check_supported_label_type(self, dataset_config: dict) -> SupportedLabelType:
        label_type = SupportedLabelType.from_string(dataset_config.get('label_type', 'YOLOv8'))
        if label_type in [
            SupportedLabelType.COCOJSON,
            SupportedLabelType.YOLOv8,
            SupportedLabelType.YOLOv5,
            SupportedLabelType.PascalVOCXML,
        ]:
            LOG.debug(f"Label type is {label_type}")
            return label_type
        raise ValueError(f"Unsupported label_type: {label_type}")

    def create_calibration_data_loader(self, transform, root, batch_size, **kwargs):
        if repr_dataloader := self.check_representative_images(transform, batch_size, **kwargs):
            return repr_dataloader
        return torch.utils.data.DataLoader(
            self._get_dataset_class(transform, root, 'train', kwargs),
            batch_size=batch_size,
            shuffle=True,
            collate_fn=lambda x: x,
            num_workers=0,
        )

    def create_validation_data_loader(self, root, target_split, **kwargs):
        return torch.utils.data.DataLoader(
            self._get_dataset_class(None, root, 'val', kwargs),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=0,
        )

    def reformat_for_calibration(self, batched_data: Any):
        return (
            batched_data
            if self.use_repr_imgs
            else torch.stack([data['image'] for data in batched_data], 0)
        )

    def reformat_for_validation(self, batched_data: Any):
        return self._format_measurement_data(batched_data)

    def _get_dataset_class(self, transform, root, split, kwargs):
        return CocoAndYoloFormatDataset(transform=transform, data_root=root, split=split, **kwargs)

    def _format_measurement_data(self, batched_data: Any) -> List[types.FrameInput]:
        def as_ground_truth(d):
            if 'bboxes' in d:
                return eval_interfaces.ObjDetGroundTruthSample.from_torch(
                    d['bboxes'], d['category_id'], d['image_id']
                )
            return None

        def as_frame_input(d):
            return types.FrameInput(
                img=types.Image.fromany(d['image']),
                ground_truth=as_ground_truth(d),
                img_id=d['image_id'],
            )

        return [as_frame_input(d) for d in batched_data]

    def evaluator(
        self, dataset_root, dataset_config, model_info, custom_config, pair_validation=False
    ):
        from ax_evaluators import yolo_eval

        return yolo_eval.YoloEvaluator(yolo_eval.YoloEvalmAPCalculator(model_info.num_classes))


class KptDataAdapter(ObjDataAdaptor):
    def __init__(self, dataset_config, model_info):
        super().__init__(dataset_config, model_info)
        self.label_type = self._check_supported_label_type(dataset_config)

    def _get_dataset_class(self, transform, root, split, kwargs):
        return CocoAndYoloFormatDataset(
            transform=transform,
            data_root=root,
            split=split,
            task=SupportedTaskCategory.Kpts,
            **kwargs,
        )

    def _check_supported_label_type(self, dataset_config: dict) -> SupportedLabelType:
        # We now default with COCO JSON, and we target to support more label types in the future
        label_type = SupportedLabelType.from_string(dataset_config.get('label_type', 'COCO JSON'))
        if label_type == SupportedLabelType.COCOJSON:
            return label_type
        raise ValueError(f"Unsupported label_type: {label_type}")

    def _format_measurement_data(self, batched_data: Any) -> List[types.FrameInput]:
        def as_ground_truth(d):
            if 'bboxes' in d:
                return eval_interfaces.KptDetGroundTruthSample.from_torch(
                    d['bboxes'], d['keypoints'], d['image_id']
                )
            return None

        def as_frame_input(d):
            return types.FrameInput.from_image(
                img=d['image'],
                color_format=types.ColorFormat.RGB,
                ground_truth=as_ground_truth(d),
                img_id=d['image_id'],
            )

        return [as_frame_input(d) for d in batched_data]

    def evaluator(
        self, dataset_root, dataset_config, model_info, custom_config, pair_validation=False
    ):
        from ax_evaluators import yolo_eval

        return yolo_eval.YoloEvaluator(
            yolo_eval.YoloEvalmAPCalculator(model_info.num_classes, is_pose=True)
        )


class SegDataAdapter(ObjDataAdaptor):
    """
    Data adapter for YOLO-format instance segmentation models.
    """

    def __init__(self, dataset_config, model_info):
        super().__init__(dataset_config, model_info)
        self.label_type = self._check_supported_label_type(dataset_config)
        self.is_mask_overlap = dataset_config.get('is_mask_overlap', True)
        if mask_size := dataset_config.get('mask_size', None):
            if not (isinstance(mask_size, (tuple, list)) and len(mask_size) == 2):
                raise ValueError("mask_size must be a tuple or list of two integers")
            LOG.info(f"mask_size is set to {mask_size}")
            self.mask_size = tuple(mask_size)
        else:
            # Create mask with raw model size, but it will be resized to (160,160)
            # during evaluation. This follows a trick to aligns with YOLOv8 metrics.
            self.mask_size = (640, 640)

    def _check_supported_label_type(self, dataset_config: dict) -> SupportedLabelType:
        label_type = SupportedLabelType.from_string(dataset_config.get('label_type', 'COCO JSON'))
        if label_type == SupportedLabelType.COCOJSON:
            return label_type
        raise ValueError(f"Unsupported label_type: {label_type}")

    def _get_dataset_class(self, transform, root, split, kwargs):
        # We can pass pipeline into kwargs to inspect if letterbox was applied
        self.eval_with_letterbox = kwargs.get('eval_with_letterbox', True)
        return CocoAndYoloFormatDataset(
            transform=transform,
            data_root=root,
            split=split,
            task=SupportedTaskCategory.Seg,
            **kwargs,
        )

    def _format_measurement_data(self, batched_data: Any) -> List[types.FrameInput]:
        formatted_data = []
        for data in batched_data:
            ground_truth = None
            if 'bboxes' in data:
                ground_truth = eval_interfaces.InstSegGroundTruthSample.from_torch(
                    raw_image_size=(data['raw_h'], data['raw_w']),
                    boxes=data['bboxes'],
                    labels=data['category_id'],
                    polygons=data['polygons'],
                    img_id=data['image_id'],
                )
                ground_truth.set_mask_parameters(
                    mask_size=self.mask_size,
                    is_mask_overlap=self.is_mask_overlap,
                    eval_with_letterbox=self.eval_with_letterbox,
                )
            formatted_data.append(
                types.FrameInput.from_image(
                    img=data['image'],
                    color_format=types.ColorFormat.RGB,
                    ground_truth=ground_truth,
                    img_id=data['image_id'],
                )
            )

        return formatted_data

    def evaluator(
        self, dataset_root, dataset_config, model_info, custom_config, pair_validation=False
    ):
        from ax_evaluators import yolo_eval

        return yolo_eval.YoloEvaluator(
            yolo_eval.YoloEvalmAPCalculator(model_info.num_classes, is_seg=True)
        )

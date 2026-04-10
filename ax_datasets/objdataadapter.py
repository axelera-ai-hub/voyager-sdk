# Copyright Axelera AI, 2025
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
from typing import Any, Dict, List, Optional, Tuple, Union
import xml.etree.ElementTree as ET

from PIL import ExifTags, Image, ImageOps
import numpy as np
from tqdm import tqdm

from axelera import types
from axelera.app import eval_interfaces, logging_utils
from axelera.app.model_utils.box import xywh2ltwh, xywh2xyxy, xyxy2xywh
from axelera.app.torch_utils import data as torch_data
from axelera.app.torch_utils import torch

LOG = logging_utils.getLogger(__name__)


# Predictable calibration tracking file location
def _get_calibration_temp_file():
    """Get the predictable temporary file for tracking calibration images."""
    # Use a predictable filename based on process ID and current working directory
    import tempfile

    temp_dir = tempfile.gettempdir()
    # Create a deterministic filename that can be found by other processes
    cwd_hash = hashlib.md5(os.getcwd().encode()).hexdigest()[:8]
    temp_file = os.path.join(temp_dir, f"axelera_calibration_images_{cwd_hash}.txt")
    return temp_file


def _track_calibration_image(image_path: str):
    """Track a calibration image by appending to temporary file."""
    temp_file = _get_calibration_temp_file()
    try:
        # Use append mode and flush immediately to ensure persistence
        with open(temp_file, 'a') as f:
            f.write(f"{image_path}\n")
            f.flush()
            os.fsync(f.fileno())  # Force write to disk
    except Exception as e:
        LOG.warning(f"Failed to track calibration image {image_path}: {e}")


def save_calibration_images_to_file(output_path):
    """Save the list of calibration images used to a text file."""
    temp_file = _get_calibration_temp_file()

    if not os.path.exists(temp_file):
        LOG.warning("No calibration images were tracked")
        return

    try:
        # Read all tracked images and remove duplicates while preserving order
        unique_images = []
        seen = set()

        with open(temp_file, 'r') as f:
            for line in f:
                image_path = line.strip()
                if image_path and image_path not in seen:
                    unique_images.append(image_path)
                    seen.add(image_path)

        if unique_images:
            with open(output_path, 'w') as f:
                for img_path in sorted(unique_images):  # Sort for consistent output
                    f.write(f"{img_path}\n")
            LOG.info(f"Saved {len(unique_images)} calibration image paths to {output_path}")
        else:
            LOG.warning("No calibration images were tracked")

    except Exception as e:
        LOG.error(f"Failed to save calibration images to {output_path}: {e}")


def clear_calibration_images_tracking():
    """Clear the calibration images tracking."""
    temp_file = _get_calibration_temp_file()
    if os.path.exists(temp_file):
        try:
            os.unlink(temp_file)
            LOG.debug(f"Removed calibration tracking temp file: {temp_file}")
        except Exception as e:
            LOG.warning(f"Failed to remove calibration tracking temp file: {e}")


# Get orientation exif tag
for k, v in ExifTags.TAGS.items():
    if v == "Orientation":
        ORIENTATION = k
        break

# Parameters
IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])


# Exception classes
class InvalidConfigurationError(Exception):
    """Exception raised for invalid dataset configurations."""

    pass


class DataFormatError(Exception):
    """Exception raised for issues with data formats."""

    pass


class DataLoadingError(Exception):
    """Exception raised for issues with loading data."""

    pass


# Enum classes
class SupportedTaskCategory(enum.Enum):
    ObjDet = enum.auto()  # object detection
    Seg = enum.auto()  # instance segmentation
    Kpts = enum.auto()  # keypoint detection


class SupportedLabelType(enum.Enum):
    YOLOv8 = enum.auto()  # YOLO v8 format
    YOLOv5 = enum.auto()  # YOLO v5 format
    PascalVOCXML = enum.auto()  # Pascal VOC XML format
    COCOJSON = enum.auto()  # COCO JSON format

    # standard datasets
    COCO2017 = enum.auto()  # COCO 2017 dataset
    COCO2014 = enum.auto()  # COCO 2014 dataset

    @classmethod
    def from_string(cls, label_type_str: str) -> 'SupportedLabelType':
        """Convert a string to a SupportedLabelType enum value."""
        mapping = {
            'YOLOv8': cls.YOLOv8,
            'YOLOv5': cls.YOLOv5,
            'Pascal VOC XML': cls.PascalVOCXML,
            'Pascal_VOC_XML': cls.PascalVOCXML,
            'COCO JSON': cls.COCOJSON,
            'COCO_JSON': cls.COCOJSON,
            'COCOJSON': cls.COCOJSON,
            'COCO2017': cls.COCO2017,
            'COCO2014': cls.COCO2014,
            'yolov8': cls.YOLOv8,
            'yolov5': cls.YOLOv5,
            'pascal voc xml': cls.PascalVOCXML,
            'pascal_voc_xml': cls.PascalVOCXML,
            'coco json': cls.COCOJSON,
            'coco_json': cls.COCOJSON,
            'cocojson': cls.COCOJSON,
            'coco2017': cls.COCO2017,
            'coco2014': cls.COCO2014,
        }

        # First check for direct match
        if label_type_str in mapping:
            return mapping[label_type_str]

        # If not found, try case-insensitive matching
        label_type_lower = label_type_str.lower()
        for key, value in mapping.items():
            if key.lower() == label_type_lower:
                return value

        # Also check against enum member names directly
        # This handles cases like 'PascalVOCXML' that might not be in the mapping
        for enum_member in cls:
            if enum_member.name.lower() == label_type_lower:
                return enum_member

        raise ValueError(f"Unsupported label_type: {label_type_str}")

    @classmethod
    def parse(cls, label_type: Union[str, 'SupportedLabelType']) -> 'SupportedLabelType':
        """Ensure a value is a SupportedLabelType enum."""
        if isinstance(label_type, str):
            return cls.from_string(label_type)
        elif isinstance(label_type, SupportedLabelType):
            return label_type
        else:
            raise ValueError(f"Unsupported label_type: {label_type}")


# Utility functions
def usable_cpus():
    """Return number of cpus configured to be used by this process."""
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        # MacOS does not provide sched_getaffinity, but cpu_count is good enough
        return os.cpu_count()


def coco80_to_coco91_table():
    """Convert COCO-80 classes to COCO-91 classes."""
    coco91 = list(range(1, 92))
    missing_classes = {12, 26, 29, 30, 45, 66, 68, 69, 71, 83, 91}
    return [x for x in coco91 if x not in missing_classes]


def coco91_to_coco80_table():
    """Convert COCO-91 classes to COCO-80 classes."""
    coco80 = coco80_to_coco91_table()
    # Initialize an array of -1's
    coco91_to_coco80 = np.full(91, -1)

    # Populate indices with their corresponding classes
    for i, cls in enumerate(coco80):
        coco91_to_coco80[cls - 1] = i

    return coco91_to_coco80


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
    """
    Convert segment labels to box labels, i.e., (cls, xy1, xy2, ...) to (cls, xywh).

    Args:
        segments (List[np.ndarray]): List of segments where each segment is an array of points.

    Returns:
        np.ndarray: Array of boxes in xywh format.
    """
    boxes = []
    for s in segments:
        x, y = s.T  # segment xy
        boxes.append([x.min(), y.min(), x.max(), y.max()])  # cls, xyxy
    return xyxy2xywh(np.array(boxes))


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


# Configuration class
class DatasetConfig:
    """Configuration class for the UnifiedDataset."""

    def __init__(
        self,
        data_root: Union[str, Path],
        val_data: Optional[str] = None,
        cal_data: Optional[str] = None,
        task: SupportedTaskCategory = SupportedTaskCategory.ObjDet,
        label_type: Union[str, SupportedLabelType] = SupportedLabelType.YOLOv8,
        output_format: str = 'xyxy',
        use_cache: bool = True,
        mask_size: Optional[Tuple[int, int]] = None,
        **kwargs,
    ):
        """Initialize the dataset configuration.

        Args:
            data_root: Root directory of the dataset
            val_data: Path to validation data list file
            cal_data: Path to calibration data list file
            task: Task category (ObjDet, Seg, Kpts)
            label_type: Label format type
            output_format: Output bbox format ('xyxy', 'xywh', 'ltwh')
            use_cache: Whether to use cached labels
            mask_size: Size of masks for segmentation (width, height)
            **kwargs: Additional configuration parameters
        """
        self.data_root = Path(data_root)
        self.val_data = val_data
        self.cal_data = cal_data
        self.task = task
        self.label_type = SupportedLabelType.parse(label_type)
        self.output_format = output_format
        self.use_cache = use_cache
        self.mask_size = mask_size

        # Store additional kwargs
        for key, value in kwargs.items():
            setattr(self, key, value)

        # Validate configuration
        self._validate_config()

    def _validate_config(self):
        """Validate the configuration parameters."""
        valid_output_formats = ['xyxy', 'xywh', 'ltwh']
        if self.output_format not in valid_output_formats:
            raise InvalidConfigurationError(
                f"Invalid output_format: {self.output_format}. Must be one of {valid_output_formats}"
            )

        if self.task == SupportedTaskCategory.Seg and self.mask_size is not None:
            if not isinstance(self.mask_size, (tuple, list)) or len(self.mask_size) != 2:
                raise InvalidConfigurationError(
                    f"mask_size must be a tuple or list of two integers, got {self.mask_size}"
                )

    def to_dict(self) -> Dict:
        """Convert the configuration to a dictionary."""
        result = {
            'data_root': str(self.data_root),
            'val_data': self.val_data,
            'cal_data': self.cal_data,
            'task': self.task.value,
            'label_type': self.label_type.value,
            'output_format': self.output_format,
            'use_cache': self.use_cache,
            'mask_size': self.mask_size,
        }

        # Add any additional attributes
        for key, value in vars(self).items():
            if key not in result:
                result[key] = value

        return result

    @classmethod
    def from_dict(cls, config_dict: Dict) -> 'DatasetConfig':
        """Create a configuration from a dictionary."""
        # Convert enum values back to enums
        task_value = config_dict.pop('task', SupportedTaskCategory.ObjDet.value)
        if isinstance(task_value, int):
            for task in SupportedTaskCategory:
                if task.value == task_value:
                    config_dict['task'] = task
                    break

        label_type_value = config_dict.pop('label_type', SupportedLabelType.YOLOv8.value)
        if isinstance(label_type_value, int):
            for label_type in SupportedLabelType:
                if label_type.value == label_type_value:
                    config_dict['label_type'] = label_type
                    break

        return cls(**config_dict)


# Main dataset class
class UnifiedDataset(torch_data.Dataset):
    """Unified dataset class for object detection, segmentation, and keypoint detection."""

    # labels caching version; bump up when changing labeling or caching method
    cache_version = 0.5

    def __init__(
        self,
        data_root: Union[str, Path],
        split: str = 'val',
        task: SupportedTaskCategory = SupportedTaskCategory.ObjDet,
        label_type: Union[str, SupportedLabelType] = SupportedLabelType.YOLOv8,
        output_format: str = 'xyxy',
        transform=None,
        use_cache: bool = True,
        labels_path: str = None,
        **kwargs,
    ):
        """Initialize the UnifiedDataset.

        Args:
            data_root: Root directory of the dataset
            split: Dataset split ('train', 'val', 'test')
            task: Task category (ObjDet, Seg, Kpts)
            label_type: Label format type
            output_format: Output bbox format ('xyxy', 'xywh', 'ltwh')
            transform: Optional transform to be applied on images
            use_cache: Whether to use cached labels
            labels_path: Path to class label names
            **kwargs: Additional configuration parameters
        """
        t1 = time.time()

        # Validate inputs
        assert split.lower() in ("train", "val", "test"), f"Unsupported split: {split}"
        assert output_format.lower() in (
            "xyxy",
            "xywh",
            "ltwh",
        ), f"Unsupported output format: {output_format}"

        # Initialize attributes
        self.data_root = Path(data_root)
        self.split = split.lower()
        self.task_enum = task
        self.output_format = output_format.lower()
        self.transform = transform
        self.use_cache = use_cache
        self.label_type = SupportedLabelType.parse(label_type)
        self.labels_path = labels_path
        self.img_paths = []
        self.labels = []
        self.segments = []
        self.image_ids = []
        self._groundtruth_json = None
        self.data_path = None  # Will be set in _configure_data

        # Configure dataset
        data_dir, reference_file, label_format = self._configure_data(self.data_root, **kwargs)

        # Get class names from labels if provided
        class_names = None
        if self.labels_path:
            class_names = self._load_class_names(self.labels_path)

        # Get images, labels, and related info
        (
            self.img_paths,
            self.labels,
            self.segments,
            self.image_ids,
            gt_json,
        ) = self._get_imgs_labels(
            data_dir, reference_file, label_format, class_names, self.output_format, **kwargs
        )

        # Store ground truth JSON path if available (for COCO evaluation)
        self._groundtruth_json = gt_json

        if not len(self.labels) and self.split == 'val':
            raise DataLoadingError("Validation requested, but no labels available")

        self.total_frames = len(self.img_paths)
        t2 = time.time()
        LOG.debug(f"Dataset initialization completed in {t2 - t1:.1f}s")

    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.total_frames

    def __getitem__(self, idx):
        """Get a sample from the dataset at the given index."""
        sample = {}
        path = self.img_paths[idx]
        img_id = self.image_ids[idx]

        # Track calibration images used during training split (calibration)
        if self.split == 'train':
            _track_calibration_image(str(path))

        # Load image
        img = self._load_image(path)
        raw_w, raw_h = img.size

        if self.transform:
            img = self.transform(img)

        sample["image"] = img
        sample["image_id"] = img_id

        # Handle labels if in training or validation mode
        if self.split in ["train", "val"]:
            label = np.array(self.labels[idx], dtype=np.float32)
            if len(label) > 0:
                sample['bboxes'] = torch.from_numpy(label[:, 1:5])
                sample["category_id"] = torch.from_numpy(label[:, 0])

                if self.task_enum == SupportedTaskCategory.Kpts:
                    label_kpts = label[:, 5:].reshape(label.shape[0], -1, 3)
                    sample['keypoints'] = torch.from_numpy(label_kpts)
                elif self.task_enum == SupportedTaskCategory.Seg:
                    sample["polygons"] = self.segments[idx]
            else:
                sample["bboxes"] = torch.as_tensor([])
                sample["category_id"] = torch.as_tensor([])

                if self.task_enum == SupportedTaskCategory.Kpts:
                    sample['keypoints'] = torch.as_tensor([])
                elif self.task_enum == SupportedTaskCategory.Seg:
                    sample['polygons'] = []

            sample["raw_w"] = raw_w
            sample["raw_h"] = raw_h

        return sample

    def _load_image(self, path):
        """Load an image from the given path."""
        try:
            img = Image.open(path)
            if img is None:
                raise DataLoadingError(f"Image not found {path}")
            elif img.mode == "L":
                img = img.convert("RGB")
            elif img.mode != "RGB":
                raise DataFormatError(f"Unsupported PIL image mode: {img.mode}")
            return img
        except Exception as e:
            raise DataLoadingError(f"Error loading image {path}: {str(e)}")

    def _load_class_names(self, labels_path):
        """Load class names from a file."""
        try:
            with open(labels_path, 'r') as f:
                class_names = [line.strip() for line in f.readlines()]
            return class_names
        except Exception as e:
            print(f"Warning: Could not load class names from {labels_path}: {str(e)}")
            return None

    def _configure_data(self, data_root, **kwargs):
        """Configure and prepare data for the specified split."""
        from axelera.app import data_utils

        # Initialize label_format to None
        label_format = None

        # Special handling for standard datasets
        if self.label_type in (SupportedLabelType.COCO2017, SupportedLabelType.COCO2014):
            # Determine which dataset and split to download
            dataset_name = (
                "COCO2017" if self.label_type == SupportedLabelType.COCO2017 else "COCO2014"
            )
            year = "2017" if self.label_type == SupportedLabelType.COCO2017 else "2014"

            # Set label_format for COCO datasets
            label_format = kwargs.get("format", "coco80")

            # Default reference file names
            if self.split == 'val':
                reference_file_name = f"val{year}.txt"
                download_split = 'val'
            else:  # train
                reference_file_name = f"train{year}.txt"
                download_split = 'train'

            # Store original reference file name for later use
            self.data_path = reference_file_name

            # Download the dataset if not already available
            try:
                # Download the main split
                data_utils.check_and_download_dataset(
                    dataset_name=dataset_name,
                    data_root_dir=data_root,
                    split=download_split,
                    is_private=False,
                )

                # Download labels
                data_utils.check_and_download_dataset(
                    dataset_name=dataset_name,
                    data_root_dir=data_root,
                    split='labels',
                    is_private=False,
                )

                # Download annotations
                data_utils.check_and_download_dataset(
                    dataset_name=dataset_name,
                    data_root_dir=data_root,
                    split='annotations',
                    is_private=False,
                )

                # If needed for specific tasks, download additional data
                if self.task_enum == SupportedTaskCategory.Kpts:
                    data_utils.check_and_download_dataset(
                        dataset_name=dataset_name,
                        data_root_dir=data_root,
                        split='pose',
                        is_private=False,
                    )
                elif self.task_enum == SupportedTaskCategory.Seg:
                    data_utils.check_and_download_dataset(
                        dataset_name=dataset_name,
                        data_root_dir=data_root,
                        split='seg',
                        is_private=False,
                    )
            except Exception as e:
                LOG.warning(f"Error downloading dataset: {e}")

            # Try to find the reference file in various locations
            potential_locations = [
                # Try direct path first
                Path(data_root, reference_file_name),
                # Look in labels directory
                Path(data_root, "labels", reference_file_name),
                # Look in labels_kpts directory for keypoints task
                Path(data_root, "labels_kpts", reference_file_name),
                # Look in annotations directory
                Path(data_root, "annotations", reference_file_name),
                # Other possible paths based on directory structure
                Path(data_root, "images", f"{self.split}{year}", "..", reference_file_name),
            ]

            reference_file_path = None
            for potential_path in potential_locations:
                if potential_path.is_file():
                    reference_file_path = potential_path
                    break

            # If not found, try to create from images directory
            if reference_file_path is None:
                try:
                    # First try standard directory structure
                    img_dir = Path(data_root, "images", f"{self.split}{year}")
                    if img_dir.is_dir():
                        reference_file_path = _create_image_list_file(img_dir)
                        LOG.info(
                            f"Created reference file from images directory: {reference_file_path}"
                        )
                    else:
                        # Try alternative paths
                        img_dir = Path(data_root, f"images/{self.split}{year}")
                        if img_dir.is_dir():
                            reference_file_path = _create_image_list_file(img_dir)
                            LOG.info(
                                f"Created reference file from alternative directory: {reference_file_path}"
                            )
                except Exception as e:
                    LOG.warning(f"Error creating reference file: {e}")

            if reference_file_path is None:
                raise DataLoadingError(
                    f"Reference file not found for {self.label_type} dataset. Please ensure the dataset is downloaded correctly."
                )

            return Path(data_root), reference_file_path, label_format

        # Regular handling for custom datasets
        if self.split == 'val':
            if 'val_data' not in kwargs:
                raise ValueError("Please specify 'val_data' for validation split")
            reference_file = kwargs['val_data']
        else:  # train
            if 'cal_data' not in kwargs:
                raise ValueError("Please specify 'cal_data' for training split")
            reference_file = kwargs['cal_data']

        # Store data path for later use
        self.data_path = reference_file

        # Convert to Path and ensure it exists
        reference_file = Path(data_root, reference_file)

        # For COCO JSON format, check if the reference file is a JSON file
        if self.label_type == SupportedLabelType.COCOJSON and reference_file.suffix == '.json':
            if not reference_file.is_file():
                raise DataLoadingError(f"COCO JSON file not found: {reference_file}")
            return Path(data_root), reference_file, label_format

        # For other formats, create image list file
        reference_file = _create_image_list_file(reference_file)

        if not reference_file.is_file():
            raise DataLoadingError(f"Reference file not found: {reference_file}")

        return Path(data_root), reference_file, label_format

    def _get_imgs_labels(
        self, data_root, reference_file, label_format, class_names, output_format, **kwargs
    ):
        """Get images, labels, and related information."""
        img_paths = []

        # Handle COCO JSON format differently
        if self.label_type == SupportedLabelType.COCOJSON and reference_file.suffix == '.json':
            return self._load_from_coco_json(data_root, reference_file, output_format, **kwargs)

        # Get image paths from reference file
        if reference_file.suffix in (".txt", ".part"):
            with open(reference_file, "r", encoding="UTF-8") as reader:
                lines = [line.rstrip() for line in reader.readlines() if line.strip()]

            # Check if the reference file contains a path to an annotation file
            if len(lines) == 1:
                annotation_path = Path(lines[0])
                if not annotation_path.is_absolute():
                    annotation_path = reference_file.parent / annotation_path

                # Handle COCO JSON annotation files
                if (
                    annotation_path.suffix.lower() == '.json'
                    and self.label_type == SupportedLabelType.COCOJSON
                ):
                    return self._load_from_coco_json(
                        data_root, annotation_path, output_format, **kwargs
                    )

                # Handle PascalVOC XML annotation files
                # Note: PascalVOC XML format expects labels to be in separate XML files per image,
                # not a single XML file for all images. So we only handle single XML if it's
                # explicitly a dataset-level XML (which is rare for PascalVOC)

            for line in lines:
                line_path = Path(line)
                if reference_file.suffix == ".part" and line_path.is_absolute():
                    img_paths.append(Path(reference_file.parent, *line_path.parts[1:]))
                elif line_path.is_absolute():
                    img_paths.append(line_path)
                else:
                    img_paths.append(reference_file.parent / line_path)

        # Filter to include only valid image formats
        img_paths = [p for p in img_paths if p.is_file() and p.suffix[1:].lower() in IMG_FORMATS]
        if not img_paths:
            raise DataLoadingError(f"No supported images found in {reference_file}")

        # Sort the image paths to ensure consistent ordering
        img_paths.sort()

        # Get corresponding label paths
        is_label_image_same_dir = kwargs.get('is_label_image_same_dir', False)
        label_tag = kwargs.get('label_tag', 'labels')
        if self.task_enum == SupportedTaskCategory.Kpts:
            label_tag = 'labels_kpts'
        elif self.task_enum == SupportedTaskCategory.Seg:
            label_tag = 'labels_seg'

        label_paths = self.image2label_paths(img_paths, is_label_image_same_dir, label_tag)

        # Load and process labels from cache or directly
        cache_file = Path(
            label_paths[0].parent,
            f"{self.split}_{data_root.stem}_{self.task_enum.name.lower()}.cache",
        )

        img_paths, shapes, labels, segments, from_cache = self._cache_and_verify_dataset(
            cache_file, img_paths, label_paths, output_format
        )

        # Convert COCO-80 to COCO-91 classes if required
        if label_format in ["coco91", "coco91-with-bg"]:
            shift = 0 if label_format == "coco91-with-bg" else 1
            for sample_labels in labels:
                for label in sample_labels:
                    label[0] = coco80_to_coco91_table()[int(label[0])] - shift
            from_cache = False

        # Generate image IDs
        image_ids = self._collect_image_ids([p.stem for p in img_paths])

        # No JSON ground truth file by default
        gt_json = None

        return img_paths, labels, segments, image_ids, gt_json

    def _convert_coco_category_id(self, category_id, **kwargs):
        """Convert COCO category ID to internal format.

        Args:
            category_id: Category ID from COCO annotation
            **kwargs: May contain 'coco_91_to_80' or 'keep_1_based_category_ids'

        Returns:
            Converted category ID (typically 0-based)
        """
        if kwargs.get('coco_91_to_80', False):
            return coco91_to_coco80_table()[category_id - 1]
        elif not kwargs.get('keep_1_based_category_ids', False):
            return category_id - 1
        return category_id

    def _convert_coco_bbox_to_format(
        self, bbox, category_id, img_width, img_height, output_format
    ):
        """Convert COCO bbox to specified output format with normalization.

        Args:
            bbox: COCO bbox [x, y, w, h] in absolute pixel coordinates
            category_id: Category ID for the bbox
            img_width: Image width for normalization
            img_height: Image height for normalization
            output_format: Target format ('xyxy', 'xywh', 'ltwh')

        Returns:
            List of [category_id, ...bbox_coords] in normalized coordinates or None if invalid

        Note:
            Following YOLO format convention, coordinates are normalized to [0, 1] range first,
            then converted to the output format. The consumer (check_image_label) will scale
            them back to absolute pixels as needed.

            Bboxes are clipped to image boundaries to handle COCO datasets with occluded/cropped
            objects that extend beyond the image edges.
        """
        if len(bbox) != 4:
            return None

        x, y, w, h = bbox

        if w <= 0 or h <= 0:
            return None

        # Clip bbox to image boundaries (COCO allows bboxes to extend beyond image for occluded objects)
        # Calculate x2, y2 before clipping x, y
        x2 = x + w
        y2 = y + h

        # Clip all coordinates to [0, width/height]
        x = max(0, min(x, img_width))
        y = max(0, min(y, img_height))
        x2 = max(0, min(x2, img_width))
        y2 = max(0, min(y2, img_height))

        # Recalculate width and height after clipping
        w = x2 - x
        h = y2 - y

        # After clipping, bbox might become invalid
        if w <= 0 or h <= 0:
            return None

        # Normalize coordinates to [0, 1] range (YOLO format convention)
        x_norm = x / img_width
        y_norm = y / img_height
        w_norm = w / img_width
        h_norm = h / img_height

        # Convert to target format (still normalized)
        if output_format == 'xyxy':
            # Convert xywh to xyxy, keeping normalized
            return [category_id, x_norm, y_norm, x_norm + w_norm, y_norm + h_norm]
        elif output_format == 'ltwh':
            # ltwh is same as COCO format but normalized
            return [category_id, x_norm, y_norm, w_norm, h_norm]
        elif output_format == 'xywh':
            # Convert to center coordinates, keeping normalized
            return [category_id, x_norm + w_norm / 2, y_norm + h_norm / 2, w_norm, h_norm]
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

    def _process_coco_annotation(self, ann, img_width, img_height, output_format, **kwargs):
        """Process a single COCO annotation into internal format.

        Args:
            ann: COCO annotation dictionary
            img_width: Image width for normalization
            img_height: Image height for normalization
            output_format: Target bbox format
            **kwargs: Additional parameters

        Returns:
            Tuple of (bbox_with_label, segment) or (None, None) if invalid
        """
        # Convert category ID
        category_id = self._convert_coco_category_id(ann['category_id'], **kwargs)

        # Convert bbox with normalization
        bbox_out = self._convert_coco_bbox_to_format(
            ann['bbox'], category_id, img_width, img_height, output_format
        )
        if bbox_out is None:
            ann_id = ann.get('id', 'unknown')
            LOG.warning(f"Skipping invalid annotation {ann_id}: bbox={ann.get('bbox')}")
            return None, None

        # Handle keypoints if present
        if self.task_enum == SupportedTaskCategory.Kpts and 'keypoints' in ann:
            bbox_out.extend(ann['keypoints'])

        # Handle segmentation if present
        segment = None
        if self.task_enum == SupportedTaskCategory.Seg and 'segmentation' in ann:
            seg = ann['segmentation']
            if isinstance(seg, list) and len(seg) > 0:
                segment = np.array(seg[0]).reshape(-1, 2)

        return bbox_out, segment

    def _find_image_directory(self, data_root, json_file, **kwargs):
        """Find the directory containing images for a COCO JSON file.

        Args:
            data_root: Root directory of the dataset
            json_file: Path to the COCO JSON annotation file
            **kwargs: Additional parameters (may contain 'img_dir')

        Returns:
            Path: Directory containing the images
        """
        # Strategy 1: Check if img_dir is explicitly provided
        if 'img_dir' in kwargs and kwargs['img_dir']:
            candidate = Path(data_root, kwargs['img_dir'])
            if candidate.is_dir():
                LOG.debug(f"Using explicitly provided img_dir: {candidate}")
                return candidate
            else:
                LOG.warning(
                    f"Specified img_dir '{kwargs['img_dir']}' not found at {candidate}. "
                    f"Will search for images in other locations."
                )

        # Strategy 2: Look in the same directory as JSON file
        candidate = json_file.parent
        if any(candidate.glob('*.jpg')) or any(candidate.glob('*.png')):
            LOG.info(f"Found images in JSON file directory: {candidate}")
            return candidate

        # Strategy 3: Try standard COCO directory structure (images/<split>)
        split_name = json_file.stem
        candidate = data_root / 'images' / split_name
        if candidate.is_dir():
            LOG.info(f"Found images in standard COCO directory: {candidate}")
            return candidate

        # Strategy 4: Try 'images' directory at data_root
        candidate = data_root / 'images'
        if candidate.is_dir():
            LOG.info(f"Found images directory: {candidate}")
            return candidate

        # Strategy 5: Use data_root itself
        LOG.warning(
            f"Could not find images in expected locations. Using data_root as fallback: {data_root}\n"
            f"Searched locations:\n"
            f"  1. {json_file.parent} (JSON file directory)\n"
            f"  2. {data_root / 'images' / split_name} (standard COCO structure)\n"
            f"  3. {data_root / 'images'} (images directory)\n"
            f"If images are not found, please specify 'img_dir' in your dataset configuration."
        )
        return data_root

    def _load_and_validate_coco_json(self, json_file):
        """Load and validate COCO JSON file.

        Args:
            json_file: Path to COCO JSON file

        Returns:
            Dictionary containing parsed COCO data

        Raises:
            DataLoadingError: If file cannot be read or parsed
            DataFormatError: If required fields are missing
        """
        try:
            with open(json_file, 'r') as f:
                coco_data = json.load(f)
        except json.JSONDecodeError as e:
            raise DataLoadingError(f"Failed to parse COCO JSON file {json_file}: {e}")
        except Exception as e:
            raise DataLoadingError(f"Failed to read COCO JSON file {json_file}: {e}")

        # Validate required fields
        if 'images' not in coco_data:
            raise DataFormatError(
                f"COCO JSON file {json_file} missing required 'images' field.\n"
                f"A valid COCO JSON file must have 'images', 'annotations', and 'categories' fields.\n"
                f"Current fields: {list(coco_data.keys())}"
            )
        if 'annotations' not in coco_data:
            raise DataFormatError(
                f"COCO JSON file {json_file} missing required 'annotations' field.\n"
                f"A valid COCO JSON file must have 'images', 'annotations', and 'categories' fields.\n"
                f"Current fields: {list(coco_data.keys())}"
            )

        return coco_data

    def _build_coco_mappings(self, coco_data):
        """Build lookup tables for COCO data.

        Args:
            coco_data: Parsed COCO JSON dictionary

        Returns:
            Tuple of (img_id_to_info, img_id_to_anns)
        """
        img_id_to_info = {img['id']: img for img in coco_data['images']}

        img_id_to_anns = {}
        for ann in coco_data['annotations']:
            img_id = ann['image_id']
            if img_id not in img_id_to_anns:
                img_id_to_anns[img_id] = []
            img_id_to_anns[img_id].append(ann)

        return img_id_to_info, img_id_to_anns

    def _load_from_coco_json(self, data_root, json_file, output_format, **kwargs):
        """Load images and labels from COCO JSON format.

        Args:
            data_root: Root directory of the dataset
            json_file: Path to the COCO JSON annotation file
            output_format: Output bbox format ('xyxy', 'xywh', 'ltwh')
            **kwargs: Additional parameters

        Returns:
            Tuple of (img_paths, labels, segments, image_ids, gt_json)
        """
        LOG.info(f"Loading COCO JSON annotations from {json_file}")

        # Load and validate JSON
        coco_data = self._load_and_validate_coco_json(json_file)

        # Build lookup mappings
        img_id_to_info, img_id_to_anns = self._build_coco_mappings(coco_data)

        # Find image directory
        img_dir = self._find_image_directory(data_root, json_file, **kwargs)

        # Process images and annotations
        img_paths = []
        labels = []
        segments = []
        image_ids = []

        missing_count = 0
        for img_id, img_info in img_id_to_info.items():
            # Get and validate image path
            img_path = Path(img_dir, img_info['file_name'])
            if not img_path.is_file():
                if missing_count < 5:  # Only log first 5 missing images
                    LOG.warning(f"Image not found: {img_path}")
                missing_count += 1
                continue

            # Get image dimensions for normalization
            img_width = img_info.get('width')
            img_height = img_info.get('height')

            if img_width is None or img_height is None or img_width <= 0 or img_height <= 0:
                LOG.warning(
                    f"Image {img_path} has invalid dimensions (w={img_width}, h={img_height}), skipping"
                )
                continue

            # Process all annotations for this image
            img_labels = []
            img_segments = []

            for ann in img_id_to_anns.get(img_id, []):
                bbox_out, segment = self._process_coco_annotation(
                    ann, img_width, img_height, output_format, **kwargs
                )
                if bbox_out is not None:
                    # Scale normalized coordinates back to absolute pixels
                    # This matches the behavior of check_image_label() for YOLO format
                    # where coordinates are stored in absolute pixels, not normalized
                    bbox_scaled = [bbox_out[0]]  # Keep class_id as is
                    if output_format == 'xyxy':
                        # Scale x1, y1, x2, y2
                        bbox_scaled.extend(
                            [
                                bbox_out[1] * img_width,  # x1
                                bbox_out[2] * img_height,  # y1
                                bbox_out[3] * img_width,  # x2
                                bbox_out[4] * img_height,  # y2
                            ]
                        )
                    elif output_format == 'xywh':
                        # Scale cx, cy, w, h
                        bbox_scaled.extend(
                            [
                                bbox_out[1] * img_width,  # cx
                                bbox_out[2] * img_height,  # cy
                                bbox_out[3] * img_width,  # w
                                bbox_out[4] * img_height,  # h
                            ]
                        )
                    elif output_format == 'ltwh':
                        # Scale x, y, w, h
                        bbox_scaled.extend(
                            [
                                bbox_out[1] * img_width,  # x
                                bbox_out[2] * img_height,  # y
                                bbox_out[3] * img_width,  # w
                                bbox_out[4] * img_height,  # h
                            ]
                        )

                    # Add any additional data (e.g., keypoints for Kpts task)
                    if len(bbox_out) > 5:
                        bbox_scaled.extend(bbox_out[5:])

                    img_labels.append(bbox_scaled)
                    if segment is not None:
                        img_segments.append(segment)

            # Add to dataset
            img_paths.append(img_path)
            labels.append(img_labels)
            segments.append(img_segments)
            image_ids.append(img_id)

        if missing_count > 0:
            LOG.warning(
                f"Skipped {missing_count} images that were not found. "
                f"Successfully loaded {len(img_paths)}/{len(img_id_to_info)} images from COCO JSON.\n"
                f"If this is unexpected, check that:\n"
                f"  1. Image directory is correct: {img_dir}\n"
                f"  2. Image file names in JSON match actual files\n"
                f"  3. You may need to specify 'img_dir' parameter in dataset config"
            )
        else:
            LOG.info(f"Successfully loaded {len(img_paths)} images from COCO JSON")

        if len(img_paths) == 0:
            raise DataLoadingError(
                f"No images found from COCO JSON file {json_file}.\n"
                f"Searched in: {img_dir}\n"
                f"Total images in JSON: {len(img_id_to_info)}\n"
                f"Please ensure images are in the correct location or specify 'img_dir' in your dataset configuration."
            )

        # Return ground truth JSON path ONLY for standard COCO datasets
        # For custom datasets, returning None forces evaluation to use our converted labels
        # This avoids category ID mismatch (custom dataset may have category_id=1 for "person"
        # but we convert to 0-based, so predictions use class 0 while JSON still has class 1)
        gt_json = None
        if self.label_type in (SupportedLabelType.COCO2017, SupportedLabelType.COCO2014):
            gt_json = str(json_file)

        return img_paths, labels, segments, image_ids, gt_json

    def _cache_and_verify_dataset(self, cache_path, img_paths, label_paths, output_format):
        """Load labels from cache or create new cache."""

        def extract_output(cache):
            # Remove useless items before parsing lists
            [cache.pop(k) for k in ("hash", "version", "status")]
            labels, shapes, segments = zip(*cache.values())
            labels = list(labels)
            segments = list(segments)
            shapes = np.array(shapes, np.int32)
            qualified_img_files = list(cache.keys())
            return qualified_img_files, shapes, labels, segments

        nm, nf, ne, nc = 0, 0, 0, 0

        # Get hash of dataset files
        hash = self._get_hash(img_paths + label_paths, output_format)

        # Load data from cache if present and valid
        cache = self._load_cache(cache_path, hash)
        if cache:
            LOG.debug(f"Loading labels from cache: {cache_path}")
            nm, nf, ne, nc = cache["status"]
            return *extract_output(cache), True

        # No valid cache present; create new cache
        print(f"Creating new label cache: {cache_path}")
        cache = {}

        # Process images and labels in parallel
        nthreads = usable_cpus()
        with Pool(nthreads) as pool:
            pbar = pool.imap(
                self.check_image_label,
                zip(img_paths, label_paths),
            )

            pbar = tqdm(
                pbar,
                total=len(label_paths),
                desc="Creating label cache",
                unit='label',
                leave=False,
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

        # Report status
        print(f"Labels found: {nf}, corrupt images: {nc}")
        print(f"Background images: {nm+ne}, missing label files: {nm}, empty label files: {ne}")

        if nf == 0:
            raise DataLoadingError("No valid image/label pairs found in dataset")

        # Write cache file
        cache['hash'] = hash
        cache['version'] = self.cache_version
        cache['status'] = nm, nf, ne, nc
        try:
            with open(cache_path, 'wb') as f:
                pickle.dump(cache, f)
        except Exception as e:
            print(f"Warning: Failed to write cache file {cache_path}: {e}")

        return *extract_output(cache), False

    def _load_cache(self, cache_path, hash):
        """Load label cache if it exists and is valid."""
        try:
            with open(cache_path, 'rb') as f:
                cache = pickle.load(f)
            if (cache["version"] != self.cache_version) or (cache["hash"] != hash):
                return {}
            return cache
        except Exception:
            return {}

    def check_image_label(self, args):
        """Check and process an image and its label file."""
        img_path, lb_path = args
        nc = 0  # number of corrupt image
        nm, nf, ne = 0, 0, 0  # number of labels (missing/found/empty)
        segments = []

        # Check images with PIL
        def exif_size(img):
            # Returns exif-corrected PIL size
            s = img.size  # (width, height)
            try:
                im_exif = img._getexif()
                if im_exif and ORIENTATION in im_exif:
                    rotation = im_exif[ORIENTATION]
                    if rotation in [6, 8]:  # rotation 270 or 90
                        s = (s[1], s[0])
            except (AttributeError, TypeError):
                # Handle cases where _getexif() is not available or returns None
                pass
            return s

        try:
            # First: verify image integrity (must be called directly after open)
            img = Image.open(img_path)
            img.verify()

            # Second: get format and size info with fresh image object
            img = Image.open(img_path)
            img_format = img.format
            w, h = exif_size(img)

            assert (w > 9) and (h > 9), f"Image w:{w} or h:{h} <10 pixels"
            assert img_format.lower() in IMG_FORMATS, f"Invalid image format {img_format}"
            if img_format.lower() in ("jpg", "jpeg"):
                with open(img_path, "rb") as f:
                    f.seek(-2, 2)
                    if f.read() != b"\xff\xd9":  # corrupt JPEG
                        ImageOps.exif_transpose(Image.open(img_path)).save(
                            img_path, "JPEG", subsampling=0, quality=100
                        )
                        print(f"Warning: {img_path}: corrupt JPEG restored and saved")
        except Exception as e:
            nc = 1
            print(f"Warning: Ignoring image {img_path}: {e}")
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
                    assert lb.shape[1] >= 5, "Each row required at least 5 values"
                    assert (lb >= 0).all(), "All values in label file must > 0"
                    assert (lb[:, 1:5] <= 1).all(), "Found unnormalized coordinates"

                    # Remove duplicates
                    _, idx = np.unique(lb, axis=0, return_index=True)
                    if len(idx) < len(lb):  # if duplicate row
                        lb = lb[idx]  # remove duplicates
                        if len(lb) - len(idx) > 0:
                            LOG.warning(
                                f"{lb_path}: {len(lb) - len(idx)} duplicate labels removed"
                            )

                    # Convert to specified output format
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

                    # Scale keypoints if present
                    if self.task_enum == SupportedTaskCategory.Kpts:
                        lb[:, 5::3] *= w
                        lb[:, 6::3] *= h

                    # Scale segments if present
                    if segments:
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
            print(f"Warning: {lb_path}: ignoring invalid labels: {e}")
            return img_path, None, None, None, nc, nm, nf, ne

    def _get_hash(self, paths, output_format):
        """Calculate a hash for a set of paths and configuration."""
        size = sum(p.stat().st_size for p in paths if p.exists())
        h = hashlib.md5(str(size).encode())
        h.update(str(sorted([p.as_posix() for p in paths])).encode())
        h.update(output_format.encode())
        return h.hexdigest()

    def _get_image_paths(self):
        """Get image paths from the data path."""
        img_paths = []
        if self.data_path and Path(self.data_path).is_file():
            with open(self.data_path, 'r') as f:
                for line in f.readlines():
                    line = line.strip()
                    if line:
                        img_paths.append(Path(line))
        return img_paths

    def _collect_image_ids(self, filenames):
        """Collect or generate image IDs from filenames."""
        image_ids = []
        for filename in filenames:
            if filename.isnumeric():
                image_ids.append(int(filename))
            else:
                image_id = int(hashlib.sha256(filename.encode("utf-8")).hexdigest(), 16) % 10**8
                image_ids.append(image_id)
        return image_ids

    @staticmethod
    def image2label_paths(img_paths, is_same_dir=False, tag="labels"):
        """Convert image paths to corresponding label paths."""
        if is_same_dir:
            return [p.with_suffix(".txt") for p in img_paths]
        else:
            # Replace /images/ with /labels/ (last instance)
            return [
                UnifiedDataset.replace_last_match_dir(p, 'images', tag).with_suffix('.txt')
                for p in img_paths
            ]

    @staticmethod
    def replace_last_match_dir(path: Path, old, new):
        """Replace the last occurrence of a directory name in a path."""
        parts = path.parts[::-1]
        try:
            index = parts.index(old)
            index = len(parts) - index - 1
            return Path(*path.parts[0:index], new, *path.parts[index + 1 :])
        except ValueError:
            return path

    @property
    def groundtruth_json(self):
        """Get the path to the ground truth JSON file."""
        return str(self._groundtruth_json) if self._groundtruth_json else ''


# Data adapter implementations
class ObjDataAdapter(types.DataAdapter):
    """Data adapter for object detection tasks."""

    def __init__(self, dataset_config, model_info):
        """Initialize the object detection data adapter."""
        self.dataset_config = dataset_config
        self.model_info = model_info
        self.label_type = self._check_supported_label_type(dataset_config)

        # Additional validation for custom datasets
        if self.label_type not in (SupportedLabelType.COCO2017, SupportedLabelType.COCO2014):
            # Check if any of the required data sources are provided
            has_cal_data = 'cal_data' in dataset_config
            has_repr_imgs = 'repr_imgs_dir_name' in dataset_config
            has_ultralytics_yaml = 'ultralytics_data_yaml' in dataset_config

            if not (has_cal_data or has_repr_imgs or has_ultralytics_yaml):
                raise ValueError(
                    f"Please specify either 'repr_imgs_dir_name', 'cal_data', or 'ultralytics_data_yaml' for {self.__class__.__name__} "
                    f"if you want to deploy with your custom data, or use a standard dataset like COCO2017 by setting 'label_type: COCO2017'"
                )

            # For validation, we need val_data unless using ultralytics (which should provide it)
            if not has_ultralytics_yaml and 'val_data' not in dataset_config:
                raise ValueError(
                    f"Please specify 'val_data' or 'ultralytics_data_yaml' for {self.__class__.__name__} "
                    f"if you want to evaluate with your custom data, or use a standard dataset like COCO2017"
                )

    def _check_supported_label_type(self, dataset_config):
        """Check if the label type is supported for object detection."""
        label_type_str = dataset_config.get('label_type', 'YOLOv8')
        label_type = SupportedLabelType.from_string(label_type_str)

        supported_types = [
            SupportedLabelType.COCOJSON,
            SupportedLabelType.YOLOv8,
            SupportedLabelType.YOLOv5,
            SupportedLabelType.PascalVOCXML,
            SupportedLabelType.COCO2017,
            SupportedLabelType.COCO2014,
        ]

        if label_type in supported_types:
            LOG.debug(f"Label type is {label_type}")
            return label_type

        raise ValueError(f"Unsupported label_type: {label_type}")

    def create_calibration_data_loader(self, transform, root, batch_size, **kwargs):
        """Create a data loader for calibration."""
        return torch.utils.data.DataLoader(
            self._get_dataset_class(transform, root, 'train', kwargs),
            batch_size=batch_size,
            shuffle=True,
            generator=kwargs.get('generator'),
            collate_fn=lambda x: x,
            num_workers=0,
        )

    def create_validation_data_loader(self, root, target_split, **kwargs):
        """Create a data loader for validation."""
        return torch.utils.data.DataLoader(
            self._get_dataset_class(None, root, 'val', kwargs),
            batch_size=1,
            shuffle=False,
            collate_fn=lambda x: x,
            num_workers=0,
        )

    def _get_dataset_class(self, transform, root, split, kwargs):
        """Get the dataset class for the adapter."""
        # For test compatibility - if label_type is default, don't pass it
        params = {
            "transform": transform,
            "data_root": root,
            "split": split,
            "task": SupportedTaskCategory.ObjDet,
        }

        # Remove label_type from kwargs to avoid duplication
        kwargs.pop('label_type', None)

        # Only add label_type if it's not the default
        if self.label_type != SupportedLabelType.YOLOv8:
            params["label_type"] = self.label_type

        return UnifiedDataset(**params, **kwargs)

    def reformat_for_calibration(self, batched_data: Any):
        return (
            batched_data
            if self.use_repr_imgs
            else torch.stack([data['image'] for data in batched_data], 0)
        )

    def reformat_for_validation(self, batched_data: Any):
        return self._format_measurement_data(batched_data)

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
        from ax_evaluators import obj_eval

        return obj_eval.ObjEvaluator(obj_eval.YoloEvalmAPCalculator(model_info.num_classes))


class SegDataAdapter(ObjDataAdapter):
    """Data adapter for segmentation tasks."""

    def __init__(self, dataset_config, model_info):
        """Initialize the segmentation data adapter."""
        super().__init__(dataset_config, model_info)

        # Set segmentation-specific parameters
        self.is_mask_overlap = dataset_config.get('is_mask_overlap', True)
        mask_size = dataset_config.get('mask_size', (640, 640))
        if mask_size and not (isinstance(mask_size, (tuple, list)) and len(mask_size) == 2):
            raise ValueError("mask_size must be a tuple or list of two integers")

        self.mask_size = tuple(int(x) for x in mask_size)

    def _check_supported_label_type(self, dataset_config):
        """Check if the label type is supported for segmentation."""
        label_type_str = dataset_config.get('label_type', 'COCO JSON')
        label_type = SupportedLabelType.from_string(label_type_str)

        supported_types = [
            SupportedLabelType.COCOJSON,
            SupportedLabelType.COCO2017,
            SupportedLabelType.COCO2014,
        ]

        if label_type in supported_types:
            return label_type

        raise ValueError(f"Unsupported label_type: {label_type}")

    def _get_dataset_class(self, transform, root, split, kwargs):
        """Get the dataset class for segmentation."""
        self.eval_with_letterbox = kwargs.get('eval_with_letterbox', True)
        kwargs.pop('label_type', None)

        # Create dataset parameters
        dataset_params = {
            "transform": transform,
            "data_root": root,
            "split": split,
            "task": SupportedTaskCategory.Seg,
        }

        # Only include label_type if it's not the default for this adapter (COCOJSON)
        if self.label_type != SupportedLabelType.COCOJSON:
            dataset_params["label_type"] = self.label_type

        return UnifiedDataset(**dataset_params, **kwargs)

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
        from ax_evaluators import obj_eval

        return obj_eval.ObjEvaluator(
            obj_eval.YoloEvalmAPCalculator(model_info.num_classes, is_seg=True)
        )


class KptDataAdapter(ObjDataAdapter):
    """Data adapter for keypoint detection tasks."""

    def __init__(self, dataset_config, model_info):
        """Initialize the keypoint detection data adapter."""
        super().__init__(dataset_config, model_info)

    def _check_supported_label_type(self, dataset_config):
        """Check if the label type is supported for keypoint detection."""
        label_type_str = dataset_config.get('label_type', 'COCO JSON')
        label_type = SupportedLabelType.from_string(label_type_str)

        supported_types = [
            SupportedLabelType.COCOJSON,
            SupportedLabelType.COCO2017,
            SupportedLabelType.COCO2014,
        ]

        if label_type in supported_types:
            return label_type

        raise ValueError(f"Unsupported label_type: {label_type}")

    def _get_dataset_class(self, transform, root, split, kwargs):
        """Get the dataset class for keypoint detection."""
        kwargs.pop('label_type', None)

        # Create dataset parameters
        dataset_params = {
            "transform": transform,
            "data_root": root,
            "split": split,
            "task": SupportedTaskCategory.Kpts,
        }

        # Only include label_type if it's not the default for this adapter (COCOJSON)
        if self.label_type != SupportedLabelType.COCOJSON:
            dataset_params["label_type"] = self.label_type

        return UnifiedDataset(**dataset_params, **kwargs)

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
        from ax_evaluators import obj_eval

        return obj_eval.ObjEvaluator(
            obj_eval.YoloEvalmAPCalculator(model_info.num_classes, is_pose=True)
        )

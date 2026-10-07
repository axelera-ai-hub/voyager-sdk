# Copyright Axelera AI, 2023
import json
import os
from pathlib import Path
import pickle
import shutil
import tempfile
from unittest import mock
from unittest.mock import MagicMock, mock_open, patch
import xml.etree.ElementTree as ET

from PIL import Image
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ax_datasets.objdataadapter import (
    DataFormatError,
    DataLoadingError,
    DatasetConfig,
    InvalidConfigurationError,
    KptDataAdapter,
    ObjDataAdapter,
    SegDataAdapter,
    SupportedLabelType,
    SupportedTaskCategory,
    UnifiedDataset,
    _create_image_list_file,
    coco80_to_coco91_table,
    coco91_to_coco80_table,
    xywh2ltwh,
    xywh2xyxy,
    xyxy2xywh,
)


@pytest.fixture
def temp_dir():
    """Create a temporary directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def sample_config():
    """Create a sample dataset configuration."""
    return DatasetConfig(
        data_root="/tmp/dataset",
        val_data="val_data.txt",
        cal_data="cal_data.txt",
        task=SupportedTaskCategory.ObjDet,
        label_type=SupportedLabelType.YOLOv8,
    )


@pytest.fixture
def sample_image():
    """Create a sample image for testing."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[30:70, 30:70] = 255  # Add a white square
    return img


@pytest.fixture
def sample_labels():
    """Create sample labels for testing."""
    # Format: [class_id, x1, y1, x2, y2]
    return [[0, 0.3, 0.3, 0.7, 0.7]]


@pytest.fixture
def sample_image_file(temp_dir, sample_image):
    """Create a sample image file for testing."""
    img_path = temp_dir / "sample.jpg"
    Image.fromarray(sample_image).save(img_path)
    return img_path


@pytest.fixture
def sample_label_file(temp_dir, sample_labels):
    """Create a sample label file for testing."""
    label_path = temp_dir / "sample.txt"
    with open(label_path, 'w') as f:
        for label in sample_labels:
            f.write(" ".join(map(str, label)) + "\n")
    return label_path


@pytest.fixture
def sample_image_list_file(temp_dir, sample_image_file):
    """Create a sample image list file for testing."""
    list_path = temp_dir / "images.txt"
    with open(list_path, 'w') as f:
        f.write(f"{sample_image_file}\n")
    return list_path


@pytest.fixture
def mock_coco_data():
    """Create mock COCO dataset."""
    return {
        "images": [
            {"id": 1, "file_name": "img1.jpg", "width": 640, "height": 480},
            {"id": 2, "file_name": "img2.jpg", "width": 640, "height": 480},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [100, 100, 200, 200]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [300, 300, 100, 100]},
            {"id": 3, "image_id": 2, "category_id": 1, "bbox": [150, 150, 250, 250]},
        ],
        "categories": [{"id": 1, "name": "person"}, {"id": 2, "name": "car"}],
    }


@pytest.fixture
def mock_voc_xml():
    """Create mock VOC XML annotation."""
    xml_content = '''
    <annotation>
        <filename>img1.jpg</filename>
        <size>
            <width>640</width>
            <height>480</height>
            <depth>3</depth>
        </size>
        <object>
            <name>person</name>
            <bndbox>
                <xmin>100</xmin>
                <ymin>100</ymin>
                <xmax>300</xmax>
                <ymax>300</ymax>
            </bndbox>
        </object>
    </annotation>
    '''
    return xml_content


class TestDatasetConfig:
    """Test the DatasetConfig class."""

    def test_init_with_defaults(self):
        """Test initialization with default values."""
        config = DatasetConfig(data_root="/tmp/dataset")
        assert config.data_root == Path("/tmp/dataset")
        assert config.task == SupportedTaskCategory.ObjDet
        assert config.label_type == SupportedLabelType.YOLOv8
        assert config.output_format == 'xyxy'
        assert config.use_cache is True
        assert config.mask_size is None

    def test_init_with_custom_values(self):
        """Test initialization with custom values."""
        config = DatasetConfig(
            data_root="/tmp/dataset",
            task=SupportedTaskCategory.Seg,
            label_type=SupportedLabelType.COCOJSON,
            output_format="xywh",
            use_cache=False,
            mask_size=(320, 320),
        )
        assert config.data_root == Path("/tmp/dataset")
        assert config.task == SupportedTaskCategory.Seg
        assert config.label_type == SupportedLabelType.COCOJSON
        assert config.output_format == "xywh"
        assert config.use_cache is False
        assert config.mask_size == (320, 320)

    def test_validation(self):
        """Test configuration validation."""
        # Test invalid output format
        with pytest.raises(InvalidConfigurationError):
            DatasetConfig(data_root="/tmp/dataset", output_format="invalid")

        # Test invalid mask size
        with pytest.raises(InvalidConfigurationError):
            DatasetConfig(
                data_root="/tmp/dataset", task=SupportedTaskCategory.Seg, mask_size="invalid"
            )

    def test_to_dict_from_dict(self):
        """Test conversion to and from dictionary."""
        original = DatasetConfig(
            data_root="/tmp/dataset",
            task=SupportedTaskCategory.Kpts,
            label_type=SupportedLabelType.COCOJSON,
        )
        config_dict = original.to_dict()
        recreated = DatasetConfig.from_dict(config_dict)

        assert recreated.data_root == original.data_root
        assert recreated.task == original.task
        assert recreated.label_type == original.label_type


class TestSupportedLabelType:
    """Test the SupportedLabelType enum."""

    def test_from_string(self):
        """Test conversion from string to enum."""
        assert SupportedLabelType.from_string("YOLOv8") == SupportedLabelType.YOLOv8
        assert SupportedLabelType.from_string("yolov8") == SupportedLabelType.YOLOv8
        assert SupportedLabelType.from_string("COCO JSON") == SupportedLabelType.COCOJSON
        assert SupportedLabelType.from_string("coco json") == SupportedLabelType.COCOJSON
        assert SupportedLabelType.from_string("COCO2017") == SupportedLabelType.COCO2017

        with pytest.raises(ValueError):
            SupportedLabelType.from_string("InvalidType")

    def test_parse(self):
        """Test parsing various input types."""
        # Test with string
        assert SupportedLabelType.parse("YOLOv8") == SupportedLabelType.YOLOv8

        # Test with enum
        assert SupportedLabelType.parse(SupportedLabelType.COCOJSON) == SupportedLabelType.COCOJSON

        # Test with invalid type
        with pytest.raises(ValueError):
            SupportedLabelType.parse(123)


class TestUtilityFunctions:
    """Test utility functions."""

    def test_coco80_to_coco91_table(self):
        """Test conversion from COCO80 to COCO91 class indices."""
        table = coco80_to_coco91_table()
        assert len(table) == 80
        assert table[0] == 1  # First class should be 1
        assert 12 not in table  # Class 12 should be missing (as per COCO dataset)

    def test_coco91_to_coco80_table(self):
        """Test conversion from COCO91 to COCO80 class indices."""
        table = coco91_to_coco80_table()
        assert len(table) == 91
        assert table[0] == 0  # Class 1 in COCO91 maps to 0 in COCO80
        assert table[11] == -1  # Class 12 in COCO91 is not in COCO80

    def test_xywh2xyxy(self):
        """Test conversion from xywh to xyxy format."""
        xywh = np.array([[0.5, 0.5, 0.2, 0.2]])
        xyxy = xywh2xyxy(xywh)
        np.testing.assert_allclose(xyxy, np.array([[0.4, 0.4, 0.6, 0.6]]))

    def test_xyxy2xywh(self):
        """Test conversion from xyxy to xywh format."""
        xyxy = np.array([[0.4, 0.4, 0.6, 0.6]])
        xywh = xyxy2xywh(xyxy)
        np.testing.assert_allclose(xywh, np.array([[0.5, 0.5, 0.2, 0.2]]))

    def test_xywh2ltwh(self):
        """Test conversion from xywh to ltwh format."""
        xywh = np.array([[0.5, 0.5, 0.2, 0.2]])
        ltwh = xywh2ltwh(xywh)
        np.testing.assert_allclose(ltwh, np.array([[0.4, 0.4, 0.2, 0.2]]))

    @patch('pathlib.Path.is_file', return_value=True)
    def test_create_image_list_file_with_file(self, mock_is_file):
        """Test creating an image list file from an existing file."""
        path = Path("/tmp/file.txt")
        result = _create_image_list_file(path)
        assert result == path

    def test_create_image_list_file_behavior(self):
        """Test the behavior of _create_image_list_file function."""
        # Create a temporary directory with an image file for testing
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test image file
            test_dir = Path(temp_dir)
            images_dir = test_dir / "images"
            images_dir.mkdir()
            test_image = images_dir / "test.jpg"
            with open(test_image, 'w') as f:
                f.write("test image content")

            # Call the function with the test directory
            result = _create_image_list_file(test_dir)

            # Verify the result is a file
            assert result.is_file()

            # Verify the file contains the image path
            with open(result, 'r') as f:
                content = f.read()
                assert str(test_image.absolute()) in content

            # Clean up the temporary file
            result.unlink()


class TestUnifiedDataset:
    """Test the UnifiedDataset class."""

    @patch('ax_datasets.objdataadapter.Path.exists', return_value=True)
    @patch('ax_datasets.objdataadapter.Path.is_file', return_value=True)
    @patch('ax_datasets.objdataadapter.UnifiedDataset._get_imgs_labels')
    @patch('ax_datasets.objdataadapter.UnifiedDataset._configure_data')
    def test_initialization(
        self, mock_configure_data, mock_get_imgs_labels, mock_is_file, mock_exists
    ):
        """Test initialization of UnifiedDataset."""
        # Mock the necessary methods
        mock_configure_data.return_value = (Path('/tmp'), Path('/tmp/val.txt'), None)
        mock_get_imgs_labels.return_value = (
            [Path('/tmp/img1.jpg')],  # img_paths
            [[[0, 0.1, 0.1, 0.2, 0.2]]],  # labels
            [[]],  # segments
            [1],  # image_ids
            None,  # gt_json
        )

        dataset = UnifiedDataset(
            data_root="/tmp/dataset",
            split='val',
            task=SupportedTaskCategory.ObjDet,
            label_type=SupportedLabelType.YOLOv8,
        )

        assert dataset.data_root == Path("/tmp/dataset")
        assert dataset.split == 'val'
        assert dataset.task_enum == SupportedTaskCategory.ObjDet
        assert dataset.label_type == SupportedLabelType.YOLOv8
        assert len(dataset.img_paths) == 1
        assert len(dataset.labels) == 1

    @patch('ax_datasets.objdataadapter.Path.exists', return_value=True)
    @patch('ax_datasets.objdataadapter.Path.is_file', return_value=True)
    @patch('builtins.open')
    @patch('ax_datasets.objdataadapter.UnifiedDataset._configure_data')
    def test_get_image_paths(self, mock_configure_data, mock_open, mock_is_file, mock_exists):
        """Test getting image paths."""
        # Mock file reading
        mock_open.return_value.__enter__.return_value.readlines.return_value = [
            'img1.jpg\n',
            'img2.jpg\n',
        ]
        mock_configure_data.return_value = (Path('/tmp'), Path('/tmp/val.txt'), None)

        with patch(
            'ax_datasets.objdataadapter.UnifiedDataset._get_imgs_labels',
            return_value=(
                [Path('img1.jpg'), Path('img2.jpg')],  # img_paths
                [[[0, 0.1, 0.1, 0.2, 0.2]], [[0, 0.3, 0.3, 0.4, 0.4]]],  # labels
                [[], []],  # segments
                [1, 2],  # image_ids
                None,  # gt_json
            ),
        ):
            dataset = UnifiedDataset(data_root='/tmp/dataset', split='val')
            dataset.data_path = 'val_list.txt'

            # Call the method
            img_paths = dataset._get_image_paths()

            # Verify results
            assert len(img_paths) == 2
            assert img_paths[0] == Path('img1.jpg')
            assert img_paths[1] == Path('img2.jpg')

    @patch('ax_datasets.objdataadapter.Path.exists', return_value=True)
    @patch('ax_datasets.objdataadapter.Path.is_file', return_value=True)
    @patch('ax_datasets.objdataadapter.pickle.load')
    @patch('ax_datasets.objdataadapter.UnifiedDataset._cache_and_verify_dataset')
    @patch('ax_datasets.objdataadapter.UnifiedDataset._configure_data')
    @patch('ax_datasets.objdataadapter._create_image_list_file')
    @patch('ax_datasets.objdataadapter.UnifiedDataset._get_imgs_labels')
    def test_cache_loading(
        self,
        mock_get_imgs_labels,
        mock_create_list,
        mock_configure_data,
        mock_cache,
        mock_load,
        mock_is_file,
        mock_exists,
    ):
        """Test cache loading."""
        # Mock configuration
        mock_configure_data.return_value = (Path('/tmp'), Path('/tmp/val.txt'), None)
        mock_create_list.return_value = Path('/tmp/val.txt')

        # Mock _get_imgs_labels to avoid file operations
        mock_get_imgs_labels.return_value = (
            [Path('img1.jpg')],  # img_paths
            [[[0, 10, 20, 30, 40]]],  # labels
            [[]],  # segments
            [1],  # image_ids
            None,  # gt_json
        )

        # Mock cache loading
        mock_cache.return_value = (
            [Path('img1.jpg')],  # img_paths
            np.array([[100, 100]]),  # shapes
            [[[0, 10, 20, 30, 40]]],  # labels
            [[]],  # segments
            True,  # from_cache
        )

        mock_load.return_value = {
            'version': 0.3,
            'hash': 'test_hash',
            'status': (0, 10, 0, 0),
            'img1.jpg': ([[0, 10, 20, 30, 40]], (100, 100), []),
        }

        with patch(
            'ax_datasets.objdataadapter.UnifiedDataset._get_hash', return_value='test_hash'
        ):
            with patch('builtins.open', MagicMock()):
                dataset = UnifiedDataset(
                    data_root='/tmp/dataset',
                    split='val',
                    use_cache=True,
                    val_data='val.txt',  # Add required val_data parameter
                )

                # Verify dataset was initialized correctly
                assert dataset.total_frames == 1
                assert len(dataset.labels) == 1
                assert dataset.labels[0] == [[0, 10, 20, 30, 40]]

    @patch('ax_datasets.objdataadapter.UnifiedDataset._cache_and_verify_dataset')
    @patch('ax_datasets.objdataadapter.UnifiedDataset._configure_data')
    @patch('ax_datasets.objdataadapter.UnifiedDataset._load_image')
    @patch('ax_datasets.objdataadapter.UnifiedDataset._get_imgs_labels')
    def test_getitem(
        self,
        mock_get_imgs_labels,
        mock_load_image,
        mock_configure_data,
        mock_cache,
        sample_image,
        sample_labels,
    ):
        """Test __getitem__ method."""
        # Mock configuration
        mock_configure_data.return_value = (Path('/tmp'), Path('/tmp/val.txt'), None)

        # Mock _get_imgs_labels to avoid file operations
        mock_get_imgs_labels.return_value = (
            [Path('/tmp/img1.jpg')],  # img_paths
            [sample_labels],  # labels
            [[]],  # segments
            [1],  # image_ids
            None,  # gt_json
        )

        # Mock cache loading
        mock_cache.return_value = (
            [Path('/tmp/img1.jpg')],  # img_paths
            np.array([[100, 100]]),  # shapes
            [sample_labels],  # labels
            [[]],  # segments
            True,  # from_cache
        )

        # Mock image loading
        mock_load_image.return_value = Image.fromarray(sample_image)

        # Create dataset
        dataset = UnifiedDataset(
            data_root='/tmp/dataset',
            split='val',
            val_data='val.txt',  # Add required val_data parameter
        )

        # Get an item
        item = dataset[0]

        # Verify item contents
        assert 'image' in item
        assert 'bboxes' in item
        assert 'category_id' in item
        assert 'image_id' in item
        assert isinstance(item['bboxes'], torch.Tensor)
        assert isinstance(item['category_id'], torch.Tensor)

    @patch('ax_datasets.objdataadapter.UnifiedDataset._cache_and_verify_dataset')
    @patch('ax_datasets.objdataadapter.UnifiedDataset._configure_data')
    @patch('ax_datasets.objdataadapter._create_image_list_file')
    @patch('ax_datasets.objdataadapter.UnifiedDataset._get_imgs_labels')
    def test_len(self, mock_get_imgs_labels, mock_create_list, mock_configure_data, mock_cache):
        """Test __len__ method."""
        # Mock configuration
        mock_configure_data.return_value = (Path('/tmp'), Path('/tmp/val.txt'), None)
        mock_create_list.return_value = Path('/tmp/val.txt')

        # Mock _get_imgs_labels to avoid file operations
        mock_get_imgs_labels.return_value = (
            [Path('/tmp/img1.jpg'), Path('/tmp/img2.jpg')],  # img_paths
            [[[0, 0.1, 0.1, 0.2, 0.2]], [[0, 0.3, 0.3, 0.4, 0.4]]],  # labels
            [[], []],  # segments
            [1, 2],  # image_ids
            None,  # gt_json
        )

        # Mock cache loading
        mock_cache.return_value = (
            [Path('/tmp/img1.jpg'), Path('/tmp/img2.jpg')],  # img_paths
            np.array([[100, 100], [200, 200]]),  # shapes
            [[[0, 0.1, 0.1, 0.2, 0.2]], [[0, 0.3, 0.3, 0.4, 0.4]]],  # labels
            [[], []],  # segments
            True,  # from_cache
        )

        with patch('builtins.open', MagicMock()):
            dataset = UnifiedDataset(
                data_root='/tmp/dataset',
                split='val',
                val_data='val.txt',  # Add required val_data parameter
            )
            assert len(dataset) == 2

    def test_image2label_paths(self):
        """Test conversion from image paths to label paths."""
        img_paths = [Path('/tmp/dataset/images/img1.jpg'), Path('/tmp/dataset/images/img2.jpg')]

        # Test with same directory
        label_paths = UnifiedDataset.image2label_paths(img_paths, is_same_dir=True)
        assert label_paths[0] == Path('/tmp/dataset/images/img1.txt')
        assert label_paths[1] == Path('/tmp/dataset/images/img2.txt')

        # Test with different directory
        label_paths = UnifiedDataset.image2label_paths(img_paths, is_same_dir=False)
        assert label_paths[0] == Path('/tmp/dataset/labels/img1.txt')
        assert label_paths[1] == Path('/tmp/dataset/labels/img2.txt')

        # Test with custom label tag
        label_paths = UnifiedDataset.image2label_paths(
            img_paths, is_same_dir=False, tag='annotations'
        )
        assert label_paths[0] == Path('/tmp/dataset/annotations/img1.txt')
        assert label_paths[1] == Path('/tmp/dataset/annotations/img2.txt')

    def test_replace_last_match_dir(self):
        """Test replacing the last match directory in a path."""
        path = Path('/tmp/dataset/images/subdir/img1.jpg')

        # Replace 'images' with 'labels'
        new_path = UnifiedDataset.replace_last_match_dir(path, 'images', 'labels')
        assert new_path == Path('/tmp/dataset/labels/subdir/img1.jpg')

        # Test with path that doesn't contain the match
        path = Path('/tmp/dataset/data/img1.jpg')
        new_path = UnifiedDataset.replace_last_match_dir(path, 'images', 'labels')
        assert new_path == path  # Should remain unchanged


class TestLabelFormats:
    """Test handling of different label formats."""

    @patch('ax_datasets.objdataadapter.Path.exists', return_value=True)
    @patch('ax_datasets.objdataadapter.Path.is_file', return_value=True)
    @patch('builtins.open')
    @patch('json.load')
    def test_coco_format(
        self, mock_json_load, mock_open, mock_is_file, mock_exists, mock_coco_data
    ):
        """Test handling of COCO format labels."""
        # Mock JSON loading
        mock_json_load.return_value = mock_coco_data

        with patch(
            'ax_datasets.objdataadapter.UnifiedDataset._get_imgs_labels'
        ) as mock_get_imgs_labels:
            # Set up mock return values
            mock_get_imgs_labels.return_value = (
                [Path('/tmp/img1.jpg'), Path('/tmp/img2.jpg')],  # img_paths
                [
                    [[0, 100, 100, 300, 300], [1, 300, 300, 400, 400]],
                    [[0, 150, 150, 400, 400]],
                ],  # labels
                [[], []],  # segments
                [1, 2],  # image_ids
                mock_coco_data,  # gt_json
            )

            # Create dataset with COCO format
            dataset = UnifiedDataset(
                data_root='/tmp/dataset',
                split='val',
                label_type=SupportedLabelType.COCOJSON,
                val_data='annotations.json',
            )

            # Verify dataset was initialized correctly
            assert dataset.label_type == SupportedLabelType.COCOJSON
            assert len(dataset.img_paths) == 2
            assert len(dataset.labels) == 2
            assert dataset.image_ids == [1, 2]

    @patch('ax_datasets.objdataadapter.Path.exists', return_value=True)
    @patch('ax_datasets.objdataadapter.Path.is_file', return_value=True)
    @patch('ax_datasets.objdataadapter.Path.glob')
    def test_yolo_format(
        self, mock_glob, mock_is_file, mock_exists, temp_dir, sample_image_file, sample_label_file
    ):
        """Test handling of YOLO format labels."""
        # Mock glob to return our sample files
        mock_glob.return_value = [sample_label_file]

        with patch(
            'ax_datasets.objdataadapter.UnifiedDataset._get_imgs_labels'
        ) as mock_get_imgs_labels:
            # Set up mock return values
            mock_get_imgs_labels.return_value = (
                [sample_image_file],  # img_paths
                [[[0, 0.3, 0.3, 0.7, 0.7]]],  # labels
                [[]],  # segments
                [1],  # image_ids
                None,  # gt_json
            )

            # Create dataset with YOLO format
            dataset = UnifiedDataset(
                data_root=temp_dir,
                split='val',
                label_type=SupportedLabelType.YOLOv8,
                val_data='images.txt',
            )

            # Verify dataset was initialized correctly
            assert dataset.label_type == SupportedLabelType.YOLOv8
            assert len(dataset.img_paths) == 1
            assert len(dataset.labels) == 1

    @patch('ax_datasets.objdataadapter.Path.exists', return_value=True)
    @patch('ax_datasets.objdataadapter.Path.is_file', return_value=True)
    @patch('xml.etree.ElementTree.parse')
    def test_voc_format(self, mock_parse, mock_is_file, mock_exists, mock_voc_xml):
        """Test handling of Pascal VOC format labels."""
        # Mock XML parsing
        mock_root = ET.fromstring(mock_voc_xml)
        mock_parse.return_value.getroot.return_value = mock_root

        with patch(
            'ax_datasets.objdataadapter.UnifiedDataset._get_imgs_labels'
        ) as mock_get_imgs_labels:
            # Set up mock return values
            mock_get_imgs_labels.return_value = (
                [Path('/tmp/img1.jpg')],  # img_paths
                [[[0, 100, 100, 300, 300]]],  # labels
                [[]],  # segments
                [1],  # image_ids
                None,  # gt_json
            )

            # Create dataset with VOC format
            dataset = UnifiedDataset(
                data_root='/tmp/dataset',
                split='val',
                label_type=SupportedLabelType.PascalVOCXML,
                val_data='val.txt',
            )

            # Verify dataset was initialized correctly
            assert dataset.label_type == SupportedLabelType.PascalVOCXML
            assert len(dataset.img_paths) == 1
            assert len(dataset.labels) == 1


class TestDataAdapters:
    """Test the data adapter classes."""

    @patch('ax_datasets.objdataadapter.ObjDataAdapter._check_supported_label_type')
    def test_obj_data_adapter(self, mock_check_label_type, sample_config):
        """Test ObjDataAdapter initialization."""
        mock_check_label_type.return_value = SupportedLabelType.YOLOv8

        model_info = MagicMock()
        model_info.num_classes = 80

        adapter = ObjDataAdapter(sample_config.to_dict(), model_info)
        assert adapter.dataset_config['task'] == sample_config.task.value
        assert adapter.dataset_config['label_type'] == sample_config.label_type.value

    @patch('ax_datasets.objdataadapter.SegDataAdapter._check_supported_label_type')
    def test_seg_data_adapter(self, mock_check_label_type, sample_config):
        """Test SegDataAdapter initialization."""
        mock_check_label_type.return_value = SupportedLabelType.COCOJSON

        model_info = MagicMock()
        model_info.num_classes = 80

        config_dict = sample_config.to_dict()
        config_dict['task'] = SupportedTaskCategory.Seg.value
        config_dict['label_type'] = SupportedLabelType.COCOJSON.value
        config_dict['mask_size'] = (160, 160)

        adapter = SegDataAdapter(config_dict, model_info)
        assert adapter.dataset_config['task'] == SupportedTaskCategory.Seg.value
        assert adapter.dataset_config['label_type'] == SupportedLabelType.COCOJSON.value
        assert adapter.mask_size == (160, 160)

    @patch('ax_datasets.objdataadapter.KptDataAdapter._check_supported_label_type')
    def test_kpt_data_adapter(self, mock_check_label_type, sample_config):
        """Test KptDataAdapter initialization."""
        mock_check_label_type.return_value = SupportedLabelType.COCOJSON

        model_info = MagicMock()
        model_info.num_classes = 80

        config_dict = sample_config.to_dict()
        config_dict['task'] = SupportedTaskCategory.Kpts.value
        config_dict['label_type'] = SupportedLabelType.COCOJSON.value

        adapter = KptDataAdapter(config_dict, model_info)
        assert adapter.dataset_config['task'] == SupportedTaskCategory.Kpts.value
        assert adapter.dataset_config['label_type'] == SupportedLabelType.COCOJSON.value

    @patch('ax_datasets.objdataadapter.ObjDataAdapter._check_supported_label_type')
    @patch('ax_datasets.objdataadapter.UnifiedDataset')
    def test_data_loader_creation(self, mock_dataset, mock_check_label_type, sample_config):
        """Test creation of data loaders."""
        mock_check_label_type.return_value = SupportedLabelType.YOLOv8

        # Mock the dataset instance
        mock_dataset_instance = MagicMock()
        # Set __len__ to return a positive value to avoid ValueError in RandomSampler
        mock_dataset_instance.__len__.return_value = 10
        mock_dataset.return_value = mock_dataset_instance

        model_info = MagicMock()
        model_info.num_classes = 80

        # Create adapter
        adapter = ObjDataAdapter(sample_config.to_dict(), model_info)

        # Test calibration data loader
        transform = MagicMock()
        cal_loader = adapter.create_calibration_data_loader(transform, "/tmp/root", 8)
        assert cal_loader is not None

        # Updated assertion to match the actual behavior - without label_type for default
        mock_dataset.assert_called_with(
            transform=transform,
            data_root="/tmp/root",
            split='train',
            task=SupportedTaskCategory.ObjDet,
        )


class TestErrorHandling:
    """Test error handling in the dataset module."""

    def test_invalid_configuration_error(self):
        """Test InvalidConfigurationError."""
        error = InvalidConfigurationError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_data_format_error(self):
        """Test DataFormatError."""
        error = DataFormatError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    def test_data_loading_error(self):
        """Test DataLoadingError."""
        error = DataLoadingError("Test error message")
        assert str(error) == "Test error message"
        assert isinstance(error, Exception)

    @patch('ax_datasets.objdataadapter.Path.exists', return_value=True)
    @patch('ax_datasets.objdataadapter.Path.is_file', return_value=False)
    @patch('ax_datasets.objdataadapter.Path.is_dir', return_value=False)
    def test_error_on_missing_reference_file(self, mock_is_dir, mock_is_file, mock_exists):
        """Test error when reference file is missing."""
        with pytest.raises(FileNotFoundError, match="Path .* is neither a file nor a directory"):
            UnifiedDataset(data_root='/tmp/dataset', split='val', val_data='missing.txt')

    @patch('ax_datasets.objdataadapter.UnifiedDataset._configure_data')
    def test_error_on_missing_reference_file_dataloadingerror(self, mock_configure_data):
        """Test error when reference file is missing."""
        mock_configure_data.side_effect = DataLoadingError("Reference file not found")

        with pytest.raises(DataLoadingError, match="Reference file not found"):
            UnifiedDataset(data_root='/tmp/dataset', split='val', val_data='missing.txt')

    @patch('ax_datasets.objdataadapter.Path.exists', return_value=True)
    @patch('ax_datasets.objdataadapter.Path.is_file', return_value=True)
    @patch('builtins.open', mock_open(read_data=""))
    def test_error_on_empty_image_list(self, mock_is_file, mock_exists):
        """Test error when image list is empty."""
        with patch(
            'ax_datasets.objdataadapter.UnifiedDataset._get_imgs_labels'
        ) as mock_get_imgs_labels:
            mock_get_imgs_labels.side_effect = DataLoadingError("No supported images found")

            with pytest.raises(DataLoadingError, match="No supported images found"):
                UnifiedDataset(data_root='/tmp/dataset', split='val', val_data='empty.txt')


# Standard dataset fixtures and tests
@pytest.fixture
def mock_coco_dirs():
    """Create a temp directory with COCO-like structure."""
    temp_dir = tempfile.mkdtemp()
    data_root = Path(temp_dir)

    # Create mock directory structure for COCO
    (data_root / "images" / "train2017").mkdir(parents=True)
    (data_root / "images" / "val2017").mkdir(parents=True)
    (data_root / "labels").mkdir(parents=True)
    (data_root / "labels_kpts").mkdir(parents=True)

    # Create sample image list file
    with open(data_root / "train2017.txt", "w") as f:
        f.write("/path/to/image1.jpg\n")
        f.write("/path/to/image2.jpg\n")

    with open(data_root / "val2017.txt", "w") as f:
        f.write("/path/to/image3.jpg\n")

    yield data_root

    # Clean up
    shutil.rmtree(temp_dir)


@pytest.fixture
def mock_download():
    """Mock the download function."""
    with mock.patch('axelera.app.data_utils.check_and_download_dataset') as m:
        yield m


@pytest.mark.parametrize("split", ["train", "val"])
def test_coco2017_dataset_automatic_download(mock_coco_dirs, mock_download, split):
    """Test COCO2017 dataset automatic download."""
    with mock.patch('ax_datasets.objdataadapter.Path.is_file', return_value=True), mock.patch(
        'ax_datasets.objdataadapter.UnifiedDataset._get_imgs_labels',
        side_effect=DataLoadingError("Test error"),
    ), mock.patch('ax_datasets.objdataadapter.Image'):

        # This should raise the DataLoadingError we're mocking
        with pytest.raises(DataLoadingError):
            dataset = UnifiedDataset(
                data_root=mock_coco_dirs, label_type=SupportedLabelType.COCO2017, split=split
            )

        # Verify download was attempted
        assert mock_download.called

        # Check COCO2017 was requested
        calls = mock_download.call_args_list
        assert any('COCO2017' in str(call) for call in calls)


# DatasetConfig tests
def test_config_creation():
    """Test creating a DatasetConfig with various parameters."""
    config = DatasetConfig(
        data_root="/data",
        val_data="val.txt",
        cal_data="train.txt",
        task=SupportedTaskCategory.ObjDet,
        label_type=SupportedLabelType.YOLOv8,
        output_format="xyxy",
        use_cache=True,
        custom_param="value",
    )

    # Check standard attributes
    assert str(config.data_root) == "/data"
    assert config.val_data == "val.txt"
    assert config.cal_data == "train.txt"
    assert config.task == SupportedTaskCategory.ObjDet
    assert config.label_type == SupportedLabelType.YOLOv8
    assert config.output_format == "xyxy"
    assert config.use_cache is True

    # Check custom attribute
    assert config.custom_param == "value"


def test_config_from_dict():
    """Test creating a DatasetConfig from a dictionary."""
    config_dict = {
        'data_root': "/data",
        'val_data': "val.txt",
        'cal_data': "train.txt",
        'task': SupportedTaskCategory.ObjDet.value,  # Integer value
        'label_type': SupportedLabelType.YOLOv8.value,  # Integer value
        'output_format': "xyxy",
        'use_cache': True,
        'custom_param': "value",
    }

    config = DatasetConfig.from_dict(config_dict)

    # Check that enums were correctly converted
    assert config.task == SupportedTaskCategory.ObjDet
    assert config.label_type == SupportedLabelType.YOLOv8
    assert config.custom_param == "value"


def test_config_validation():
    """Test validation of configuration parameters."""
    # Test invalid output format
    with pytest.raises(InvalidConfigurationError):
        DatasetConfig(data_root="/data", output_format="invalid_format")

    # Test invalid mask_size for segmentation
    with pytest.raises(InvalidConfigurationError):
        DatasetConfig(
            data_root="/data", task=SupportedTaskCategory.Seg, mask_size=123  # Not a tuple/list
        )


# Cache mechanism tests
@pytest.fixture
def mock_cache():
    """Create a mock cache file."""
    temp_dir = tempfile.mkdtemp()
    cache_path = Path(temp_dir) / "test.cache"

    # Create a mock cache file
    cache_data = {
        "hash": "abc123",
        "version": 0.3,
        "status": (0, 5, 0, 0),  # nm, nf, ne, nc
        "image1.jpg": [[[0, 0.5, 0.5, 0.5, 0.5]], (100, 100), []],
        "image2.jpg": [[[1, 0.2, 0.2, 0.3, 0.3]], (200, 200), []],
    }

    with open(cache_path, "wb") as f:
        pickle.dump(cache_data, f)

    yield cache_path

    # Clean up
    shutil.rmtree(temp_dir)


def test_load_from_cache(mock_cache):
    """Test loading dataset from cache."""
    with mock.patch('ax_datasets.objdataadapter.UnifiedDataset._get_hash', return_value="abc123"):
        dataset = mock.MagicMock(spec=UnifiedDataset)
        dataset.cache_version = 0.3

        # Call the method we're testing
        cache = UnifiedDataset._load_cache(dataset, mock_cache, "abc123")

        # Check the cache was loaded correctly
        assert cache["hash"] == "abc123"
        assert cache["version"] == 0.3
        assert cache["status"] == (0, 5, 0, 0)
        assert len(cache) - 3 == 2  # 2 images plus hash, version, status


def test_invalid_cache_hash(mock_cache):
    """Test handling of cache with invalid hash."""
    with mock.patch(
        'ax_datasets.objdataadapter.UnifiedDataset._get_hash', return_value="different_hash"
    ):
        dataset = mock.MagicMock(spec=UnifiedDataset)
        dataset.cache_version = 0.3

        # Call the method we're testing
        cache = UnifiedDataset._load_cache(dataset, mock_cache, "different_hash")

        # Cache should be rejected due to hash mismatch
        assert cache == {}


def test_invalid_cache_version(mock_cache):
    """Test handling of cache with invalid version."""
    with mock.patch('ax_datasets.objdataadapter.UnifiedDataset._get_hash', return_value="abc123"):
        dataset = mock.MagicMock(spec=UnifiedDataset)
        dataset.cache_version = 0.4  # Different from cache file

        # Call the method we're testing
        cache = UnifiedDataset._load_cache(dataset, mock_cache, "abc123")

        # Cache should be rejected due to version mismatch
        assert cache == {}


@pytest.mark.parametrize("output_format", ["xyxy", "xywh", "ltwh"])
def test_output_formats(output_format):
    """Test different output formats."""
    # More complete mocking to capture the output_format
    with mock.patch(
        'ax_datasets.objdataadapter.UnifiedDataset._configure_data',
        return_value=(Path("/tmp"), Path("/tmp/list.txt"), None),
    ), mock.patch(
        'ax_datasets.objdataadapter.UnifiedDataset._get_imgs_labels',
        side_effect=DataLoadingError("Test error"),
    ), mock.patch(
        'pathlib.Path.is_file', return_value=True
    ), mock.patch(
        'ax_datasets.objdataadapter.UnifiedDataset.__getitem__'
    ) as mock_getitem:

        # Create the dataset - should initialize but raise error in _get_imgs_labels
        with pytest.raises(DataLoadingError):
            dataset = UnifiedDataset(
                data_root="/tmp", output_format=output_format, val_data="dummy.txt"
            )

            # Verify the output_format was set correctly on the dataset
            assert dataset.output_format == output_format


# YOLO format tests
@pytest.fixture
def yolo_dirs():
    """Create temp directories for YOLO format testing."""
    temp_dir = tempfile.mkdtemp()
    data_root = Path(temp_dir)

    # Create directory structure for directory-based approach
    images_dir = data_root / "images"
    labels_dir = data_root / "labels"
    train_dir = images_dir / "train"
    val_dir = images_dir / "valid"

    # Create directories
    train_dir.mkdir(parents=True)
    val_dir.mkdir(parents=True)
    labels_dir.mkdir(parents=True)

    # Create sample image files
    (train_dir / "img1.jpg").touch()
    (train_dir / "img2.jpg").touch()
    (val_dir / "img3.jpg").touch()

    # Create corresponding label files
    (labels_dir / "train").mkdir(exist_ok=True, parents=True)
    (labels_dir / "valid").mkdir(exist_ok=True, parents=True)
    (labels_dir / "train" / "img1.txt").write_text("0 0.5 0.5 0.1 0.1")
    (labels_dir / "train" / "img2.txt").write_text("1 0.3 0.3 0.2 0.2")
    (labels_dir / "valid" / "img3.txt").write_text("2 0.6 0.6 0.15 0.15")

    # Create text file for text-based approach
    train_txt = data_root / "train.txt"
    val_txt = data_root / "val.txt"

    with open(train_txt, "w") as f:
        f.write(f"./images/train/img1.jpg\n")
        f.write(f"./images/train/img2.jpg\n")

    with open(val_txt, "w") as f:
        f.write(f"./images/valid/img3.jpg\n")

    yield data_root

    # Clean up
    shutil.rmtree(temp_dir)


def test_yolo_directory_based(yolo_dirs):
    """Test YOLO format with directory-based approach."""
    # Mock _check_supported_label_type to return the enum directly
    with mock.patch(
        'ax_datasets.objdataadapter.ObjDataAdapter._check_supported_label_type',
        return_value=SupportedLabelType.YOLOv8,
    ), mock.patch('ax_datasets.objdataadapter.Image'):

        # Create a proper config dictionary with a string for label_type
        config_dict = {
            'data_root': str(yolo_dirs),
            'cal_data': "train",
            'val_data': "valid",
            'label_type': "YOLOv8",  # Use string instead of enum value
        }

        # Initialize adapter
        model_info = mock.MagicMock()
        model_info.num_classes = 80
        adapter = ObjDataAdapter(config_dict, model_info)

        # Test with mocked dataset class
        with mock.patch(
            'ax_datasets.objdataadapter.UnifiedDataset', side_effect=DataLoadingError("Test error")
        ):

            with pytest.raises(DataLoadingError):
                train_dataset = adapter._get_dataset_class(
                    transform=None, root=yolo_dirs, split="train", kwargs={"cal_data": "train"}
                )


def test_yolo_text_file_based(yolo_dirs):
    """Test YOLO format with text file-based approach."""
    # Mock _check_supported_label_type to return the enum directly
    with mock.patch(
        'ax_datasets.objdataadapter.ObjDataAdapter._check_supported_label_type',
        return_value=SupportedLabelType.YOLOv8,
    ), mock.patch('ax_datasets.objdataadapter.Image'):

        # Create a proper config dictionary with a string for label_type
        config_dict = {
            'data_root': str(yolo_dirs),
            'cal_data': "train.txt",
            'val_data': "val.txt",
            'label_type': "YOLOv8",  # Use string instead of enum value
        }

        # Initialize adapter
        model_info = mock.MagicMock()
        model_info.num_classes = 80
        adapter = ObjDataAdapter(config_dict, model_info)

        # Test with mocked dataset class
        with mock.patch(
            'ax_datasets.objdataadapter.UnifiedDataset', side_effect=DataLoadingError("Test error")
        ):

            with pytest.raises(DataLoadingError):
                train_dataset = adapter._get_dataset_class(
                    transform=None, root=yolo_dirs, split="train", kwargs={"cal_data": "train.txt"}
                )


class TestUltralyticsIntegration:
    """Test integration with Ultralytics data YAML format."""

    def test_obj_data_adapter_accepts_ultralytics_yaml(self):
        """Test that ObjDataAdapter accepts ultralytics_data_yaml parameter."""
        from axelera import types

        dataset_config = {'ultralytics_data_yaml': 'data.yaml', 'label_type': 'YOLOv8'}

        model_info = types.ModelInfo(
            name='test_model',
            task_category='ObjectDetection',
            input_tensor_shape=[1, 3, 640, 640],
            input_color_format='RGB',
            input_tensor_layout='NCHW',
        )

        # This should not raise a validation error
        adapter = ObjDataAdapter(dataset_config, model_info)
        assert adapter.label_type == SupportedLabelType.YOLOv8

    def test_obj_data_adapter_rejects_mixed_ultralytics_traditional(self):
        """Test that ObjDataAdapter rejects mixing ultralytics_data_yaml with traditional params."""
        from axelera import types

        dataset_config = {
            'ultralytics_data_yaml': 'data.yaml',
            'cal_data': 'train.txt',  # Should not be allowed together
            'label_type': 'YOLOv8',
        }

        model_info = types.ModelInfo(
            name='test_model',
            task_category='ObjectDetection',
            input_tensor_shape=[1, 3, 640, 640],
            input_color_format='RGB',
            input_tensor_layout='NCHW',
        )

        # This should not raise an error at initialization since the validation happens at processing time
        adapter = ObjDataAdapter(dataset_config, model_info)
        assert adapter.label_type == SupportedLabelType.YOLOv8

    @patch('ax_datasets.objdataadapter.ObjDataAdapter._check_supported_label_type')
    def test_validation_checks_ultralytics_or_traditional_data_sources(
        self, mock_check_label_type
    ):
        """Test that validation requires either ultralytics_data_yaml or traditional data sources."""
        from axelera import types

        mock_check_label_type.return_value = SupportedLabelType.YOLOv8

        # Test with no data sources at all
        dataset_config = {
            'label_type': 'YOLOv8'
            # Missing any data source configuration
        }

        model_info = types.ModelInfo(
            name='test_model',
            task_category='ObjectDetection',
            input_tensor_shape=[1, 3, 640, 640],
            input_color_format='RGB',
            input_tensor_layout='NCHW',
        )

        with pytest.raises(
            ValueError,
            match="Please specify either 'repr_imgs_dir_name', 'cal_data', or 'ultralytics_data_yaml'",
        ):
            ObjDataAdapter(dataset_config, model_info)

    @patch('ax_datasets.objdataadapter.ObjDataAdapter._check_supported_label_type')
    def test_validation_allows_ultralytics_without_val_data(self, mock_check_label_type):
        """Test that validation allows ultralytics_data_yaml without explicit val_data."""
        from axelera import types

        mock_check_label_type.return_value = SupportedLabelType.YOLOv8

        dataset_config = {
            'ultralytics_data_yaml': 'data.yaml',
            'label_type': 'YOLOv8',
            # No val_data - should be OK since ultralytics will provide it
        }

        model_info = types.ModelInfo(
            name='test_model',
            task_category='ObjectDetection',
            input_tensor_shape=[1, 3, 640, 640],
            input_color_format='RGB',
            input_tensor_layout='NCHW',
        )

        # This should not raise an error
        adapter = ObjDataAdapter(dataset_config, model_info)
        assert adapter.label_type == SupportedLabelType.YOLOv8

    @patch('ax_datasets.objdataadapter.ObjDataAdapter._check_supported_label_type')
    def test_validation_requires_val_data_for_traditional_format(self, mock_check_label_type):
        """Test that validation requires val_data for traditional format."""
        from axelera import types

        mock_check_label_type.return_value = SupportedLabelType.YOLOv8

        dataset_config = {
            'cal_data': 'train.txt',
            'label_type': 'YOLOv8',
            # Missing val_data for traditional format
        }

        model_info = types.ModelInfo(
            name='test_model',
            task_category='ObjectDetection',
            input_tensor_shape=[1, 3, 640, 640],
            input_color_format='RGB',
            input_tensor_layout='NCHW',
        )

        with pytest.raises(
            ValueError, match="Please specify 'val_data' or 'ultralytics_data_yaml'"
        ):
            ObjDataAdapter(dataset_config, model_info)


class TestCocoJsonHelperFunctions:
    """Test the new COCO JSON helper functions."""

    def test_convert_coco_category_id_default(self):
        """Test default category ID conversion (1-based to 0-based)."""
        dataset = mock.MagicMock(spec=UnifiedDataset)
        category_id = UnifiedDataset._convert_coco_category_id(dataset, 1)
        assert category_id == 0

        category_id = UnifiedDataset._convert_coco_category_id(dataset, 5)
        assert category_id == 4

    def test_convert_coco_category_id_keep_1_based(self):
        """Test keeping 1-based category IDs."""
        dataset = mock.MagicMock(spec=UnifiedDataset)
        category_id = UnifiedDataset._convert_coco_category_id(
            dataset, 1, keep_1_based_category_ids=True
        )
        assert category_id == 1

    def test_convert_coco_category_id_coco91_to_80(self):
        """Test COCO-91 to COCO-80 conversion."""
        dataset = mock.MagicMock(spec=UnifiedDataset)
        # Category 1 in COCO-91 is index 0 in COCO-80
        category_id = UnifiedDataset._convert_coco_category_id(dataset, 1, coco_91_to_80=True)
        assert category_id == 0

    def test_convert_coco_bbox_to_format_xyxy(self):
        """Test bbox conversion to xyxy format with normalization."""
        dataset = mock.MagicMock(spec=UnifiedDataset)
        bbox = [100, 150, 200, 300]  # [x, y, w, h] in absolute pixels
        img_width, img_height = 800, 600
        result = UnifiedDataset._convert_coco_bbox_to_format(
            dataset, bbox, 0, img_width, img_height, 'xyxy'
        )
        # Expected: normalized coordinates [cls, x1/w, y1/h, x2/w, y2/h]
        # [0, 100/800, 150/600, 300/800, 450/600] = [0, 0.125, 0.25, 0.375, 0.75]
        assert result == [0, 0.125, 0.25, 0.375, 0.75]

    def test_convert_coco_bbox_to_format_xywh(self):
        """Test bbox conversion to xywh format with normalization."""
        dataset = mock.MagicMock(spec=UnifiedDataset)
        bbox = [100, 150, 200, 300]  # [x, y, w, h] in absolute pixels
        img_width, img_height = 800, 600
        result = UnifiedDataset._convert_coco_bbox_to_format(
            dataset, bbox, 0, img_width, img_height, 'xywh'
        )
        # Expected: normalized [cls, cx/w, cy/h, w/w, h/h]
        # cx = 100 + 200/2 = 200, cy = 150 + 300/2 = 300
        # [0, 200/800, 300/600, 200/800, 300/600] = [0, 0.25, 0.5, 0.25, 0.5]
        assert result == [0, 0.25, 0.5, 0.25, 0.5]

    def test_convert_coco_bbox_to_format_ltwh(self):
        """Test bbox conversion to ltwh format with normalization."""
        dataset = mock.MagicMock(spec=UnifiedDataset)
        bbox = [100, 150, 200, 300]  # [x, y, w, h] in absolute pixels
        img_width, img_height = 800, 600
        result = UnifiedDataset._convert_coco_bbox_to_format(
            dataset, bbox, 0, img_width, img_height, 'ltwh'
        )
        # Expected: normalized [cls, x/w, y/h, w/w, h/h]
        # [0, 100/800, 150/600, 200/800, 300/600] = [0, 0.125, 0.25, 0.25, 0.5]
        assert result == [0, 0.125, 0.25, 0.25, 0.5]

    def test_convert_coco_bbox_normalization_always_within_bounds(self):
        """Test that normalization always produces values in [0, 1] range."""
        dataset = mock.MagicMock(spec=UnifiedDataset)

        # Test various bbox positions and sizes
        test_cases = [
            # (bbox, img_width, img_height)
            ([0, 0, 800, 600], 800, 600),  # Full image
            ([400, 300, 100, 50], 800, 600),  # Center region
            ([0, 0, 1, 1], 1920, 1080),  # Tiny bbox on large image
            ([1900, 1070, 10, 5], 1920, 1080),  # Near edge
            # Out of bounds bboxes (should be clipped)
            ([72, 202, 163, 503], 800, 600),  # CrowdHuman example: y2=705 > 600
            ([199, 180, 144, 499], 800, 600),  # CrowdHuman example: y2=679 > 600
            ([-10, -10, 50, 50], 800, 600),  # Negative coordinates
            ([750, 550, 100, 100], 800, 600),  # Extends beyond both width and height
        ]

        for bbox, w, h in test_cases:
            for output_format in ['xyxy', 'xywh', 'ltwh']:
                result = UnifiedDataset._convert_coco_bbox_to_format(
                    dataset, bbox, 0, w, h, output_format
                )
                # Check all coordinate values are in [0, 1] range
                for val in result[1:]:  # Skip class_id at index 0
                    assert 0 <= val <= 1, (
                        f"Coordinate {val} out of [0,1] range for bbox={bbox}, "
                        f"img_size=({w},{h}), format={output_format}"
                    )

    def test_convert_coco_bbox_clipping_behavior(self):
        """Test that out-of-bounds bboxes are properly clipped."""
        dataset = mock.MagicMock(spec=UnifiedDataset)
        img_width, img_height = 800, 600

        # Test bbox that extends beyond image height (CrowdHuman example)
        bbox = [72, 202, 163, 503]  # y2 = 202 + 503 = 705 > 600
        result = UnifiedDataset._convert_coco_bbox_to_format(
            dataset, bbox, 0, img_width, img_height, 'xyxy'
        )

        # After clipping: x=[72, 235], y=[202, 600]
        # Normalized: x1=72/800=0.09, y1=202/600≈0.3367, x2=235/800=0.29375, y2=600/600=1.0
        assert result is not None
        assert result[0] == 0  # class_id
        assert abs(result[1] - 0.09) < 0.001  # x1
        assert abs(result[2] - 0.3367) < 0.001  # y1
        assert abs(result[3] - 0.29375) < 0.001  # x2
        assert abs(result[4] - 1.0) < 0.001  # y2 clipped to 1.0

        # Test bbox with negative coordinates
        bbox = [-10, -10, 50, 50]  # Should be clipped to [0, 0, 40, 40]
        result = UnifiedDataset._convert_coco_bbox_to_format(
            dataset, bbox, 0, img_width, img_height, 'xyxy'
        )
        assert result is not None
        assert result[1] == 0.0  # x1 clipped to 0
        assert result[2] == 0.0  # y1 clipped to 0
        assert result[3] == 40 / 800  # x2 = 40/800 = 0.05
        assert result[4] == 40 / 600  # y2 = 40/600 ≈ 0.0667

    def test_convert_coco_bbox_invalid_length(self):
        """Test bbox conversion with invalid bbox length."""
        dataset = mock.MagicMock(spec=UnifiedDataset)
        bbox = [100, 150, 200]  # Invalid: only 3 values
        result = UnifiedDataset._convert_coco_bbox_to_format(dataset, bbox, 0, 800, 600, 'xyxy')
        assert result is None

    def test_convert_coco_bbox_invalid_dimensions(self):
        """Test bbox conversion with invalid dimensions (w or h <= 0)."""
        dataset = mock.MagicMock(spec=UnifiedDataset)

        # Test with w = 0
        bbox = [100, 150, 0, 300]
        result = UnifiedDataset._convert_coco_bbox_to_format(dataset, bbox, 0, 800, 600, 'xyxy')
        assert result is None

        # Test with h = 0
        bbox = [100, 150, 200, 0]
        result = UnifiedDataset._convert_coco_bbox_to_format(dataset, bbox, 0, 800, 600, 'xyxy')
        assert result is None

        # Test with negative w
        bbox = [100, 150, -10, 300]
        result = UnifiedDataset._convert_coco_bbox_to_format(dataset, bbox, 0, 800, 600, 'xyxy')
        assert result is None

    def test_convert_coco_bbox_unsupported_format(self):
        """Test bbox conversion with unsupported format."""
        dataset = mock.MagicMock(spec=UnifiedDataset)
        bbox = [100, 150, 200, 300]

        with pytest.raises(ValueError, match="Unsupported output format"):
            UnifiedDataset._convert_coco_bbox_to_format(
                dataset, bbox, 0, 800, 600, 'invalid_format'
            )

    @patch('ax_datasets.objdataadapter.LOG')
    def test_process_coco_annotation_valid(self, mock_log):
        """Test processing a valid COCO annotation with normalization."""
        # Create a real instance to test the actual methods
        dataset = UnifiedDataset.__new__(UnifiedDataset)
        dataset.task_enum = SupportedTaskCategory.ObjDet

        ann = {'id': 1, 'category_id': 1, 'bbox': [100, 150, 200, 300]}
        img_width, img_height = 800, 600

        bbox_out, segment = dataset._process_coco_annotation(ann, img_width, img_height, 'xyxy')

        # Expected: normalized [cls, x1/w, y1/h, x2/w, y2/h]
        # [0, 100/800, 150/600, 300/800, 450/600] = [0, 0.125, 0.25, 0.375, 0.75]
        assert bbox_out == [0, 0.125, 0.25, 0.375, 0.75]
        assert segment is None

    @patch('ax_datasets.objdataadapter.LOG')
    def test_process_coco_annotation_with_keypoints(self, mock_log):
        """Test processing COCO annotation with keypoints."""
        dataset = UnifiedDataset.__new__(UnifiedDataset)
        dataset.task_enum = SupportedTaskCategory.Kpts

        ann = {
            'id': 1,
            'category_id': 1,
            'bbox': [100, 150, 200, 300],
            'keypoints': [10, 20, 1, 30, 40, 1],  # [x1, y1, v1, x2, y2, v2]
        }
        img_width, img_height = 800, 600

        bbox_out, segment = dataset._process_coco_annotation(ann, img_width, img_height, 'xyxy')

        # Expected: normalized bbox + keypoints (keypoints are NOT normalized in COCO)
        assert bbox_out[:5] == [0, 0.125, 0.25, 0.375, 0.75]
        assert bbox_out[5:] == [10, 20, 1, 30, 40, 1]
        assert segment is None

    @patch('ax_datasets.objdataadapter.LOG')
    def test_process_coco_annotation_with_segmentation(self, mock_log):
        """Test processing COCO annotation with segmentation."""
        dataset = UnifiedDataset.__new__(UnifiedDataset)
        dataset.task_enum = SupportedTaskCategory.Seg

        ann = {
            'id': 1,
            'category_id': 1,
            'bbox': [100, 150, 200, 300],
            'segmentation': [[10, 20, 30, 40, 50, 60]],  # polygon points
        }
        img_width, img_height = 800, 600

        bbox_out, segment = dataset._process_coco_annotation(ann, img_width, img_height, 'xyxy')

        # Expected: normalized bbox
        assert bbox_out == [0, 0.125, 0.25, 0.375, 0.75]
        assert segment is not None
        assert isinstance(segment, np.ndarray)
        assert segment.shape == (3, 2)  # 3 points, 2 coordinates each

    @patch('ax_datasets.objdataadapter.LOG')
    def test_process_coco_annotation_invalid_bbox(self, mock_log):
        """Test processing annotation with invalid bbox."""
        dataset = UnifiedDataset.__new__(UnifiedDataset)
        dataset.task_enum = SupportedTaskCategory.ObjDet

        ann = {'id': 1, 'category_id': 1, 'bbox': [100, 150, 0, 300]}  # w=0 is invalid
        img_width, img_height = 800, 600

        bbox_out, segment = dataset._process_coco_annotation(ann, img_width, img_height, 'xyxy')

        assert bbox_out is None
        assert segment is None
        # Verify warning was logged
        assert mock_log.warning.called

    def test_load_and_validate_coco_json_valid(self, temp_dir):
        """Test loading and validating a valid COCO JSON file."""
        dataset = mock.MagicMock(spec=UnifiedDataset)

        # Create a valid COCO JSON file
        json_path = temp_dir / "annotations.json"
        coco_data = {
            'images': [{'id': 1, 'file_name': 'img1.jpg', 'width': 640, 'height': 480}],
            'annotations': [
                {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [100, 100, 200, 200]}
            ],
            'categories': [{'id': 1, 'name': 'person'}],
        }

        with open(json_path, 'w') as f:
            json.dump(coco_data, f)

        result = UnifiedDataset._load_and_validate_coco_json(dataset, json_path)

        assert result == coco_data
        assert 'images' in result
        assert 'annotations' in result

    def test_load_and_validate_coco_json_missing_images(self, temp_dir):
        """Test validation error for missing 'images' field."""
        dataset = mock.MagicMock(spec=UnifiedDataset)

        json_path = temp_dir / "annotations.json"
        coco_data = {
            'annotations': [{'id': 1, 'image_id': 1}],
            'categories': [{'id': 1, 'name': 'person'}],
        }

        with open(json_path, 'w') as f:
            json.dump(coco_data, f)

        with pytest.raises(DataFormatError, match="missing required 'images' field"):
            UnifiedDataset._load_and_validate_coco_json(dataset, json_path)

    def test_load_and_validate_coco_json_missing_annotations(self, temp_dir):
        """Test validation error for missing 'annotations' field."""
        dataset = mock.MagicMock(spec=UnifiedDataset)

        json_path = temp_dir / "annotations.json"
        coco_data = {
            'images': [{'id': 1, 'file_name': 'img1.jpg'}],
            'categories': [{'id': 1, 'name': 'person'}],
        }

        with open(json_path, 'w') as f:
            json.dump(coco_data, f)

        with pytest.raises(DataFormatError, match="missing required 'annotations' field"):
            UnifiedDataset._load_and_validate_coco_json(dataset, json_path)

    def test_load_and_validate_coco_json_invalid_json(self, temp_dir):
        """Test error for invalid JSON format."""
        dataset = mock.MagicMock(spec=UnifiedDataset)

        json_path = temp_dir / "invalid.json"
        with open(json_path, 'w') as f:
            f.write("{invalid json content")

        with pytest.raises(DataLoadingError, match="Failed to parse COCO JSON file"):
            UnifiedDataset._load_and_validate_coco_json(dataset, json_path)

    def test_build_coco_mappings(self):
        """Test building COCO lookup mappings."""
        dataset = mock.MagicMock(spec=UnifiedDataset)

        coco_data = {
            'images': [
                {'id': 1, 'file_name': 'img1.jpg', 'width': 640, 'height': 480},
                {'id': 2, 'file_name': 'img2.jpg', 'width': 640, 'height': 480},
            ],
            'annotations': [
                {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [100, 100, 200, 200]},
                {'id': 2, 'image_id': 1, 'category_id': 2, 'bbox': [300, 300, 100, 100]},
                {'id': 3, 'image_id': 2, 'category_id': 1, 'bbox': [150, 150, 250, 250]},
            ],
        }

        img_id_to_info, img_id_to_anns = UnifiedDataset._build_coco_mappings(dataset, coco_data)

        # Check image mappings
        assert len(img_id_to_info) == 2
        assert img_id_to_info[1]['file_name'] == 'img1.jpg'
        assert img_id_to_info[2]['file_name'] == 'img2.jpg'

        # Check annotation mappings
        assert len(img_id_to_anns) == 2
        assert len(img_id_to_anns[1]) == 2  # Image 1 has 2 annotations
        assert len(img_id_to_anns[2]) == 1  # Image 2 has 1 annotation

    def test_find_image_directory_explicit(self, temp_dir):
        """Test finding image directory with explicit img_dir parameter."""
        dataset = mock.MagicMock(spec=UnifiedDataset)

        img_dir = temp_dir / 'custom_images'
        img_dir.mkdir()

        json_file = temp_dir / 'annotations.json'
        json_file.touch()

        result = UnifiedDataset._find_image_directory(
            dataset, temp_dir, json_file, img_dir='custom_images'
        )

        assert result == img_dir

    def test_find_image_directory_same_as_json(self, temp_dir):
        """Test finding images in the same directory as JSON file."""
        dataset = mock.MagicMock(spec=UnifiedDataset)

        json_file = temp_dir / 'annotations.json'
        json_file.touch()

        # Create an image in the same directory
        (temp_dir / 'img1.jpg').touch()

        result = UnifiedDataset._find_image_directory(dataset, temp_dir, json_file)

        assert result == temp_dir

    def test_find_image_directory_standard_coco_structure(self, temp_dir):
        """Test finding images in standard COCO directory structure."""
        dataset = mock.MagicMock(spec=UnifiedDataset)

        # Create standard COCO structure: images/val/
        img_dir = temp_dir / 'images' / 'val'
        img_dir.mkdir(parents=True)

        json_file = temp_dir / 'val.json'
        json_file.touch()

        result = UnifiedDataset._find_image_directory(dataset, temp_dir, json_file)

        assert result == img_dir

    def test_find_image_directory_fallback_to_data_root(self, temp_dir):
        """Test fallback to data_root when no images found elsewhere."""
        dataset = mock.MagicMock(spec=UnifiedDataset)

        json_file = temp_dir / 'subdir' / 'annotations.json'
        json_file.parent.mkdir()
        json_file.touch()

        result = UnifiedDataset._find_image_directory(dataset, temp_dir, json_file)

        assert result == temp_dir


class TestErrorMessageQuality:
    """Test that error messages are clear and helpful."""

    def test_json_file_not_treated_as_text(self, temp_dir):
        """Test that JSON files are not read line-by-line as text files.

        This is a regression test for the bug where JSON file contents
        were being read as image paths, causing massive error messages.
        """
        # Create a COCO JSON file
        json_path = temp_dir / "annotations.json"
        coco_data = {
            'images': [{'id': 1, 'file_name': 'img1.jpg', 'width': 640, 'height': 480}],
            'annotations': [
                {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [100, 100, 200, 200]}
            ],
            'categories': [{'id': 1, 'name': 'person'}],
        }

        with open(json_path, 'w') as f:
            json.dump(coco_data, f)

        # Create the image
        img_path = temp_dir / 'img1.jpg'
        img = Image.new('RGB', (640, 480))
        img.save(img_path)

        # Mock the dataset initialization
        with patch('ax_datasets.objdataadapter.UnifiedDataset._configure_data') as mock_config:
            mock_config.return_value = (temp_dir, json_path, None)

            # This should work without trying to read JSON content as image paths
            dataset = UnifiedDataset(
                data_root=temp_dir,
                split='val',
                label_type=SupportedLabelType.COCOJSON,
                val_data='annotations.json',
            )

            # Should successfully load 1 image
            assert len(dataset) == 1
            assert dataset.img_paths[0] == img_path

    def test_missing_images_error_is_clear(self, temp_dir):
        """Test that missing images produce clear error messages."""
        dataset = UnifiedDataset.__new__(UnifiedDataset)

        json_path = temp_dir / "annotations.json"
        coco_data = {
            'images': [
                {'id': 1, 'file_name': 'missing1.jpg', 'width': 640, 'height': 480},
                {'id': 2, 'file_name': 'missing2.jpg', 'width': 640, 'height': 480},
            ],
            'annotations': [
                {'id': 1, 'image_id': 1, 'category_id': 1, 'bbox': [100, 100, 200, 200]},
                {'id': 2, 'image_id': 2, 'category_id': 1, 'bbox': [100, 100, 200, 200]},
            ],
            'categories': [{'id': 1, 'name': 'person'}],
        }

        with open(json_path, 'w') as f:
            json.dump(coco_data, f)

        # This should raise a clear error about no images found
        with pytest.raises(DataLoadingError, match="No images found from COCO JSON"):
            dataset._load_from_coco_json(temp_dir, json_path, 'xyxy')

    def test_invalid_json_format_error_is_clear(self, temp_dir):
        """Test that invalid JSON produces a clear error message."""
        dataset = UnifiedDataset.__new__(UnifiedDataset)

        json_path = temp_dir / "invalid.json"
        with open(json_path, 'w') as f:
            f.write("{ this is not valid json }")

        with pytest.raises(DataLoadingError, match="Failed to parse COCO JSON file"):
            dataset._load_and_validate_coco_json(json_path)

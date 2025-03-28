# Copyright Axelera AI, 2024
import json
from pathlib import Path
import tempfile
from unittest import mock
from unittest.mock import MagicMock, patch
import xml.etree.ElementTree as ET

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ax_datasets.objdataadapter import (
    CocoAndYoloFormatDataset,
    ObjDataAdaptor,
    coco80_to_coco91_table,
)


@pytest.fixture
def mock_dataset():
    with mock.patch.object(
        CocoAndYoloFormatDataset, '__init__', lambda self, *args, **kwargs: None
    ):
        yield CocoAndYoloFormatDataset()


def test_get_hash(mock_dataset, tmp_path):
    paths1 = [tmp_path / "image1.jpg", tmp_path / "image2.jpg"]
    paths2 = [tmp_path / "image3.jpg", tmp_path / "image4.jpg"]
    for path in paths1 + paths2:
        path.touch()

    output_format1 = "voc"
    output_format2 = "coco"

    # same_paths_different_format
    hash1 = mock_dataset._get_hash(paths1, output_format1)
    hash2 = mock_dataset._get_hash(paths1, output_format2)
    hash3 = mock_dataset._get_hash(paths2, output_format1)
    assert hash1 != hash2
    assert hash1 != hash3
    assert hash1 == mock_dataset._get_hash(list(reversed(paths1)), output_format1)


def test_collect_image_ids_all_numbers(mock_dataset):
    filenames = ['1', '2', '3']
    expected_ids = [1, 2, 3]
    assert mock_dataset._collect_image_ids(filenames) == expected_ids


def test_collect_image_ids_mix_of_numbers_and_names(mock_dataset):
    filenames = ['img1', '1', 'img3', '4', 'img5']
    expected_ids = [79467065, 1, 33612514, 4, 87772161]
    assert mock_dataset._collect_image_ids(filenames) == expected_ids


def test_collect_image_ids_duplicate_names(mock_dataset):
    filenames = ['img1', 'img2', 'img3', 'img1', 'img4']
    expected_ids = [79467065, 63918327, 33612514, 79467066, 29151243]
    assert mock_dataset._collect_image_ids(filenames) == expected_ids


def test_collect_image_ids_empty_list(mock_dataset):
    filenames = []
    expected_ids = []
    assert mock_dataset._collect_image_ids(filenames) == expected_ids


def test_collect_image_ids_single_name(mock_dataset):
    filenames = ['img1']
    expected_ids = [79467065]
    assert mock_dataset._collect_image_ids(filenames) == expected_ids


def test_replace_last_match_dir():
    path = Path('/path/to/images/image1.jpg')
    old = 'images'
    new = 'labels'
    expected_output = Path('/path/to/labels/image1.jpg')
    assert CocoAndYoloFormatDataset.replace_last_match_dir(path, old, new) == expected_output


def test_image2label_paths_same_dir():
    img_paths = [Path('/path/to/images/image1.jpg'), Path('/path/to/images/image2.jpg')]
    label_paths = CocoAndYoloFormatDataset.image2label_paths(img_paths, True)
    expected_paths = [Path('/path/to/images/image1.txt'), Path('/path/to/images/image2.txt')]
    assert label_paths == expected_paths


def test_image2label_paths_different_dir():
    img_paths = [Path('/path/to/images/image1.jpg'), Path('/path/to/images/image2.jpg')]
    label_paths = CocoAndYoloFormatDataset.image2label_paths(img_paths, False)
    expected_paths = [Path('/path/to/labels/image1.txt'), Path('/path/to/labels/image2.txt')]
    assert label_paths == expected_paths


def test_image2label_paths_different_dir_with_tag():
    img_paths = [
        Path('data/train/images/img1.jpg'),
        Path('data/train/images/img2.jpg'),
        Path('data/train/images/img3.jpg'),
    ]
    expected_output = [
        Path('data/train/annotations/img1.txt'),
        Path('data/train/annotations/img2.txt'),
        Path('data/train/annotations/img3.txt'),
    ]
    actual_output = CocoAndYoloFormatDataset.image2label_paths(
        img_paths, is_label_image_same_dir=False, tag='annotations'
    )
    assert expected_output == actual_output


def test_image2label_paths():
    img_paths = [Path('images/img1.jpg'), Path('images/img2.jpg'), Path('images/img3.jpg')]
    is_label_image_same_dir = False
    expected_label_paths = [
        Path('labels/img1.txt'),
        Path('labels/img2.txt'),
        Path('labels/img3.txt'),
    ]
    assert (
        CocoAndYoloFormatDataset.image2label_paths(img_paths, is_label_image_same_dir)
        == expected_label_paths
    )


def test_coco80_to_coco91_table():
    expected_table = [
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9,
        10,
        11,
        13,
        14,
        15,
        16,
        17,
        18,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        27,
        28,
        31,
        32,
        33,
        34,
        35,
        36,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        46,
        47,
        48,
        49,
        50,
        51,
        52,
        53,
        54,
        55,
        56,
        57,
        58,
        59,
        60,
        61,
        62,
        63,
        64,
        65,
        67,
        70,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        84,
        85,
        86,
        87,
        88,
        89,
        90,
    ]
    coco91 = coco80_to_coco91_table()
    assert len(coco91) == 80
    assert coco91 == expected_table


def create_sample_annotation(save_to: str):
    coco_data = {
        "images": [
            {"id": 1, "file_name": "image1.jpg", "height": 500, "width": 400},
            {"id": 2, "file_name": "image2.jpg", "height": 600, "width": 800},
        ],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [50, 125, 100, 250]},
            {"id": 2, "image_id": 1, "category_id": 2, "bbox": [290, 175, 60, 100]},
            {"id": 3, "image_id": 2, "category_id": 3, "bbox": [20, 30, 160, 120]},
        ],
        "categories": [
            {"id": 1, "name": "cat"},
            {"id": 2, "name": "dog"},
            {"id": 3, "name": "bird"},
        ],
    }
    with open(save_to, "w") as f:
        json.dump(coco_data, f)


def test_coco2yolo_format_labels():
    # Create a temporary directory to save the output files
    with tempfile.TemporaryDirectory() as tmpdirname:
        save_base_path = Path(tmpdirname)

        create_sample_annotation(save_base_path / 'sample_annotation.json')
        annotation_file = save_base_path / 'sample_annotation.json'
        CocoAndYoloFormatDataset.coco2yolo_format_labels(annotation_file, save_base_path)

        # Check if the output files are created
        expected_files = ['image1.txt', 'image2.txt']
        for file_name in expected_files:
            assert (save_base_path / file_name).exists()

        # Check the content of the output files
        with open(save_base_path / 'image1.txt', 'r') as f:
            lines = f.readlines()
            assert len(lines) == 2
            assert lines[0].strip() == '0 0.25 0.5 0.25 0.5'
            assert lines[1].strip() == '1 0.8 0.45 0.15 0.2'
        with open(save_base_path / 'image2.txt', 'r') as f:
            lines = f.readlines()
            assert len(lines) == 1
            assert lines[0].strip() == '2 0.125 0.15 0.2 0.2'


def test_voc2yolo_format_labels():
    # Create a temporary directory to store output files
    with tempfile.TemporaryDirectory() as tempdir:
        # Create a sample VOC annotation file
        xml_path = Path(tempdir) / 'sample.xml'
        ann_root = ET.Element('annotation')
        size_elem = ET.SubElement(ann_root, 'size')
        ET.SubElement(size_elem, 'width').text = '640'
        ET.SubElement(size_elem, 'height').text = '480'
        obj_elem = ET.SubElement(ann_root, 'object')
        ET.SubElement(obj_elem, 'name').text = 'cat'
        ET.SubElement(obj_elem, 'difficult').text = '0'
        bbox_elem = ET.SubElement(obj_elem, 'bndbox')
        ET.SubElement(bbox_elem, 'xmin').text = '100'
        ET.SubElement(bbox_elem, 'xmax').text = '200'
        ET.SubElement(bbox_elem, 'ymin').text = '150'
        ET.SubElement(bbox_elem, 'ymax').text = '250'
        ann_tree = ET.ElementTree(ann_root)
        ann_tree.write(str(xml_path))

        # Convert the VOC annotation file to yolo format
        save_base_path = Path(tempdir) / 'labels'
        classes = ['cat', 'dog']
        label_files = CocoAndYoloFormatDataset.voc2yolo_format_labels(
            [str(xml_path)], save_base_path, classes
        )

        # Check that the output file exists and contains the correct labels
        expected_output = '0 0.2328125 0.4145833333333333 0.15625 0.20833333333333334\n'
        output_path = save_base_path / f'{xml_path.stem}.txt'
        assert output_path.exists()
        with output_path.open(mode='r', encoding='utf-8') as f:
            assert f.read() == expected_output


def test_label_to_coco_bbox():
    image_width = 800
    image_height = 600
    coco_label = [0, 100, 150, 200, 300]
    voc_label = [0, 100, 150, 300, 450]
    yolo_label = [0, 0.25, 0.5, 0.25, 0.5]

    coco_output = CocoAndYoloFormatDataset.label_to_coco_bbox(
        'ltwh', coco_label, image_width, image_height
    )
    voc_output = CocoAndYoloFormatDataset.label_to_coco_bbox(
        'xyxy', voc_label, image_width, image_height
    )
    yolo_output = CocoAndYoloFormatDataset.label_to_coco_bbox(
        'xywh', yolo_label, image_width, image_height
    )

    assert list(coco_output) == coco_label
    assert list(voc_output) == coco_label
    assert list(yolo_output) == coco_label


@pytest.fixture
def yolo_format_data_adapter():
    dataset_config = {
        'val_data': '/path/to/val_data',
        'cal_data': '/path/to/cal_data',
    }
    with patch('ax_datasets.objdataadapter.CocoAndYoloFormatDataset', autospec=True) as mock_coco:
        mock_coco.return_value = MagicMock()
        adapter = ObjDataAdaptor(dataset_config, None)
    return adapter


def test_init_yolo_format_data_adapter(yolo_format_data_adapter):
    assert isinstance(yolo_format_data_adapter, ObjDataAdaptor)

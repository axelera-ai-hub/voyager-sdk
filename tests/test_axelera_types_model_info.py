# Copyright Axelera AI, 2024

import functools
import operator

import pytest

from axelera import types

Task = types.TaskCategory


@pytest.mark.parametrize(
    "input, expected_cats",
    [
        ("Classification", [Task.Classification]),
        ("ObjectDetection", [Task.ObjectDetection]),
        ("objectdetection", [Task.ObjectDetection]),
        ("KeypointDetection", [Task.KeypointDetection]),
        ("SemanticSegmentation", [Task.SemanticSegmentation]),
        ("InstanceSegmentation", [Task.InstanceSegmentation]),
        ("ImageEnhancement", [Task.ImageEnhancement]),
        ("ObjectDetection|KeypointDetection", [Task.ObjectDetection, Task.KeypointDetection]),
        (
            "InstanceSegmentation | SemanticSegmentation",
            [Task.InstanceSegmentation, Task.SemanticSegmentation],
        ),
        (["Classification", " KeypointDetection"], [Task.Classification, Task.KeypointDetection]),
        (
            Task.Classification | Task.KeypointDetection,
            [Task.Classification, Task.KeypointDetection],
        ),
        (Task.Classification, [Task.Classification]),
    ],
)
def test_parse_tasks(input, expected_cats):
    expected = functools.reduce(operator.or_, expected_cats)
    got = Task.parse(input)
    assert got == expected
    # check that we can parse it correctly in ModelInfo param list
    assert (
        types.ModelInfo("spam", input, [3, 10, 10], types.ColorFormat.RGB).task_category
        == expected
    )
    # __len__ and __iter__ are py 3.11+ only, so test our impl here
    assert len(got) == len(expected_cats)
    assert set(got) == set(expected_cats)


@pytest.mark.parametrize(
    "input, exc, match",
    [
        (
            "Clasification",
            ValueError,
            "Unknown TaskCategory Clasification, valid options are Classification Object",
        ),
        (1, TypeError, r"Invalid type for TaskCategory `int`, use str or list\[str\]"),
    ],
)
def test_parse_task_failures(input, exc, match):
    with pytest.raises(exc, match=match):
        Task.parse(input)


@pytest.mark.parametrize(
    "input, expected",
    [
        ("RGB", types.ColorFormat.RGB),
        ("BGR", types.ColorFormat.BGR),
        ("bgr", types.ColorFormat.BGR),
        ("GRAY", types.ColorFormat.GRAY),
    ],
)
def test_parse_color_format(input, expected):
    assert types.ColorFormat.parse(input) == expected
    assert (
        types.ModelInfo("spam", "Classification", [3, 10, 10], input).input_color_format
        == expected
    )


@pytest.mark.parametrize(
    "input, exc, match",
    [
        ("RGBX", ValueError, "Unknown ColorFormat RGBX, valid options are RGB BGR RGBA BGRA GRAY"),
        (1, TypeError, r"Invalid type for ColorFormat `int`, use str"),
    ],
)
def test_parse_color_format_failures(input, exc, match):
    with pytest.raises(exc, match=match):
        types.ColorFormat.parse(input)


@pytest.mark.parametrize(
    "input, expected",
    [
        ("NCHW", types.TensorLayout.NCHW),
        ("NHWC", types.TensorLayout.NHWC),
        ("CHWN", types.TensorLayout.CHWN),
    ],
)
def test_parse_tensor_layout(input, expected):
    assert types.TensorLayout.parse(input) == expected
    mi = types.ModelInfo("s", "Classification", [3, 10, 10], types.ColorFormat.RGB, input)
    assert expected == mi.input_tensor_layout


@pytest.mark.parametrize("seq_type", [list, tuple])
@pytest.mark.parametrize(
    "inp_shape, width, height, channels, format",
    [
        ((1, 3, 10, 16), 16, 10, 3, "nchw"),
        ((1, 10, 16, 3), 16, 10, 3, "nhwc"),
        ((3, 10, 16, 1), 16, 10, 3, "chwn"),
        ((3, 10, 16), 16, 10, 3, "nchw"),
        ((10, 16, 3), 16, 10, 3, "nhwc"),
        ((3, 10, 16), 16, 10, 3, "chwn"),  # ?
    ],
)
def test_model_info_input_tensor_shape(seq_type, inp_shape, width, height, channels, format):
    shape = seq_type(inp_shape)
    mi = types.ModelInfo("n", "classification", shape, input_tensor_layout=format)
    assert width == mi.input_width
    assert height == mi.input_height
    assert channels == mi.input_channel


def test_model_info_invalid_input_tensor_shape():
    with pytest.raises(ValueError, match="Invalid input tensor shape"):
        types.ModelInfo("n", "classification", [1])
    with pytest.raises(ValueError, match="Invalid input tensor shape"):
        types.ModelInfo("n", "classification", [1, 2])
    with pytest.raises(ValueError, match="Invalid input tensor shape"):
        types.ModelInfo("n", "classification", [1, 2, 3, 4, 5])


gold_json = """\
{
  "name": "s",
  "task_category": "Classification",
  "input_tensor_shape": [
    1,
    3,
    10,
    15
  ],
  "input_color_format": "RGB",
  "input_tensor_layout": "NCHW",
  "num_classes": 1,
  "labels": [],
  "label_filter": [],
  "weight_path": "",
  "weight_url": "",
  "weight_md5": "",
  "prequantized_url": "",
  "prequantized_md5": "",
  "dataset": "",
  "base_dir": "",
  "class_name": "",
  "class_path": "",
  "version": "",
  "extra_kwargs": {}
}"""


def test_model_info_to_json():
    mi = types.ModelInfo("s", "Classification", [3, 10, 15], "rgb", "nchw")
    assert gold_json == mi.to_json()


def test_model_info_from_json():
    mi = types.ModelInfo.from_json(gold_json)
    assert mi.name == "s"
    assert mi.task_category == Task.Classification
    assert mi.input_tensor_shape == [1, 3, 10, 15]
    assert mi.input_width == 15
    assert mi.input_height == 10
    assert mi.input_channel == 3
    assert mi.input_color_format == types.ColorFormat.RGB
    assert mi.input_tensor_layout == types.TensorLayout.NCHW
    assert mi.num_classes == 1
    assert mi.labels == []
    assert mi.label_filter == []
    assert mi.weight_path == ""
    assert mi.weight_url == ""
    assert mi.weight_md5 == ""
    assert mi.dataset == ""
    assert mi.base_dir == ""
    assert mi.class_name == ""
    assert mi.class_path == ""
    assert mi.version == ""
    assert mi.extra_kwargs == {}

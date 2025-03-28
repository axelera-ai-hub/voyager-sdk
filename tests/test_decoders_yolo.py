# Copyright Axelera AI, 2024
from pathlib import Path
from unittest.mock import patch

from PIL import Image as PILImage
import pytest

torch = pytest.importorskip("torch")

from ax_models.decoders import yolo
from axelera import types
from axelera.app import meta, operators

MANIFEST = types.Manifest(
    'yolo',
    [(0.003919653594493866, -128)],
    [[0.08142165094614029, 70], [0.09499982744455338, 82], [0.09290479868650436, 66]],
    input_shapes=[(1, 320, 320, 64)],
    output_shapes=[[1, 20, 20, 256], [1, 40, 40, 256], [1, 80, 80, 256]],
    n_padded_ch_inputs=[(0, 0, 0, 0, 0, 0, 0, 52)],
    n_padded_ch_outputs=[
        (0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 1),
        (0, 0, 0, 0, 0, 0, 0, 1),
    ],
    model_lib_file='lib_export/lib.so',
    postprocess_graph='lib_export/post_process.onnx',
)


@pytest.mark.parametrize(
    "use_multi_label, num_of_class, num_of_samples, expected_boxes_count",
    [(False, 10, 30, 27), (True, 10, 30, 270)],
)
@patch(
    'ax_models.decoders.yolo.BBoxState.organize_bboxes',
    side_effect=lambda boxes, scores, classes: (boxes, scores, classes),
)
def test_exec_torch_obj_confidence_filtering(
    mock_organize_bboxes, use_multi_label, num_of_class, num_of_samples, expected_boxes_count
):
    the_meta = meta.AxMeta(0)
    the_img = types.img.frompil(PILImage.new('RGB', (640, 640)))
    # Adjust the tensor shape based on whether multi-label is used
    input_tensor_shape = (1, num_of_samples, 5 + num_of_class)
    input_tensor = torch.ones(input_tensor_shape)
    # set the first 3 samples with confidence 0.4 and the others as 0.6
    input_tensor[0, 0:3, 4] = 0.4
    input_tensor[0, 3:, 4] = 0.6

    decoder = yolo.DecodeYolo(
        box_format='xywh',
        normalized_coord=True,
        conf_threshold=0.5,
        use_multi_label=use_multi_label,
    )
    model_info = types.ModelInfo(
        name="yolov5s",
        task_category=types.TaskCategory.ObjectDetection,
        input_tensor_shape=input_tensor_shape,
        num_classes=10,
    )
    model_info.manifest = MANIFEST
    decoder.configure_model_and_context_info(
        model_info=model_info,
        context=operators.PipelineContext(),
        task_name="task_name",
        taskn=0,
        where=None,
        compiled_model_dir=Path('.'),
    )
    _, _, o_meta = decoder.exec_torch(the_img, input_tensor, the_meta)
    assert 'task_name' in o_meta
    assert len(o_meta['task_name'].boxes) == expected_boxes_count


@pytest.mark.parametrize(
    "use_multi_label, num_of_class, num_of_samples, expected_boxes_count",
    [(False, 10, 30, 12), (True, 10, 30, 120)],  # 30/2 - 3 = 12  # 12*10
)
@patch(
    'ax_models.decoders.yolo.BBoxState.organize_bboxes',
    side_effect=lambda boxes, scores, classes: (boxes, scores, classes),
)
def test_exec_torch_score_filtering(
    mock_organize_bboxes, use_multi_label, num_of_class, num_of_samples, expected_boxes_count
):
    the_meta = meta.AxMeta(0)
    the_img = types.img.frompil(PILImage.new('RGB', (640, 640)))
    # Adjust the tensor shape based on whether multi-label is used
    input_tensor_shape = (1, num_of_samples, 5 + num_of_class)
    input_tensor = torch.full(input_tensor_shape, 0.9)
    half_samples = num_of_samples // 2
    # values set to 0.8, resulting in 0.8*0.6 < 0.5, will be filtered out
    input_tensor[0, half_samples:, :] = 0.8
    # set the first 3 samples with confidence 0.4 and the others as 0.6
    input_tensor[0, 0:3, 4] = 0.4
    input_tensor[0, 3:, 4] = 0.6

    decoder = yolo.DecodeYolo(
        box_format='xywh',
        normalized_coord=True,
        conf_threshold=0.5,
        use_multi_label=use_multi_label,
    )
    model_info = types.ModelInfo(
        name="yolov5s",
        task_category=types.TaskCategory.ObjectDetection,
        input_tensor_shape=input_tensor_shape,
        num_classes=10,
    )
    model_info.manifest = MANIFEST
    decoder.configure_model_and_context_info(
        model_info=model_info,
        context=operators.PipelineContext(),
        task_name="task_name",
        taskn=0,
        where=None,
        compiled_model_dir=Path('.'),
    )
    _, _, o_meta = decoder.exec_torch(the_img, input_tensor, the_meta)
    assert 'task_name' in o_meta
    assert len(o_meta['task_name'].boxes) == expected_boxes_count


@pytest.mark.parametrize(
    "use_multi_label, num_of_class, num_of_samples, expected_boxes_count",
    [(False, 10, 30, 28), (True, 10, 30, 280)],
)
@patch(
    'ax_models.decoders.yolo.BBoxState.organize_bboxes',
    side_effect=lambda boxes, scores, classes: (boxes, scores, classes),
)
def test_exec_torch_obj_confidence_filtering_anchor_free(
    mock_organize_bboxes, use_multi_label, num_of_class, num_of_samples, expected_boxes_count
):
    the_meta = meta.AxMeta(0)
    the_img = types.img.frompil(PILImage.new('RGB', (640, 640)))
    # Adjust the tensor shape based on whether multi-label is used
    input_tensor_shape = (1, num_of_samples, 4 + num_of_class)
    input_tensor = torch.full(input_tensor_shape, 0.9)
    input_tensor[0, 0:2, :] = 0.3

    decoder = yolo.DecodeYolo(
        box_format='xywh',
        normalized_coord=True,
        conf_threshold=0.5,
        use_multi_label=use_multi_label,
    )
    model_info = types.ModelInfo(
        name="yolov5s",
        task_category=types.TaskCategory.ObjectDetection,
        input_tensor_shape=input_tensor_shape,
        num_classes=10,
    )
    model_info.manifest = MANIFEST
    decoder.configure_model_and_context_info(
        model_info=model_info,
        context=operators.PipelineContext(),
        task_name="task_name",
        taskn=0,
        where=None,
        compiled_model_dir=Path('.'),
    )
    _, _, o_meta = decoder.exec_torch(the_img, input_tensor, the_meta)
    assert 'task_name' in o_meta
    assert len(o_meta['task_name'].boxes) == expected_boxes_count


@patch(
    'ax_models.decoders.yolo.BBoxState.organize_bboxes',
    side_effect=lambda boxes, scores, classes: (boxes, scores, classes),
)
def test_exec_torch_yolonas_merge_input(mock_organize_bboxes):
    the_meta = meta.AxMeta(0)
    the_img = types.img.frompil(PILImage.new('RGB', (640, 640)))
    # Adjust the tensor shape based on whether multi-label is used
    input_tensor = (torch.full((1, 30, 4), 0.6), torch.full((1, 30, 10), 0.6))

    decoder = yolo.DecodeYolo(
        box_format='xyxy',
        normalized_coord=True,
        conf_threshold=0.5,
        use_multi_label=False,
    )
    model_info = types.ModelInfo(
        name="yolo-nas-s",
        task_category=types.TaskCategory.ObjectDetection,
        input_tensor_shape=[1, 3, 640, 640],
        num_classes=10,
    )
    model_info.manifest = MANIFEST
    decoder.configure_model_and_context_info(
        model_info=model_info,
        context=operators.PipelineContext(),
        task_name="task_name",
        taskn=0,
        where=None,
        compiled_model_dir=Path('.'),
    )
    _, _, o_meta = decoder.exec_torch(the_img, input_tensor, the_meta)
    assert 'task_name' in o_meta
    assert len(o_meta['task_name'].boxes) == 30


@pytest.mark.parametrize(
    "input_tensor_shape, expected_error",
    [
        ((1, 30, 16), r"Unknown number of output channels \(30, 16\)"),
        (
            (1, 30, 5, 10),
            r"Unknown number of output channels \(30, 5, 10\)",
        ),
    ],
)
def test_exec_torch_tensor_shape_errors(input_tensor_shape, expected_error):
    the_meta = meta.AxMeta(0)
    the_img = types.img.frompil(PILImage.new('RGB', (640, 640)))
    if isinstance(input_tensor_shape, tuple) and isinstance(input_tensor_shape[0], tuple):
        input_tensor = (
            torch.full(input_tensor_shape[0], 0.6),
            torch.full(input_tensor_shape[1], 0.6),
        )
    else:
        input_tensor = torch.full(input_tensor_shape, 0.6)

    decoder = yolo.DecodeYolo(
        box_format='xywh',
        normalized_coord=True,
        conf_threshold=0.5,
        use_multi_label=False,
    )
    model_info = types.ModelInfo(
        name="yolov5s",
        task_category=types.TaskCategory.ObjectDetection,
        input_tensor_shape=input_tensor_shape,
        num_classes=10,
    )
    model_info.manifest = MANIFEST

    with pytest.raises(ValueError, match=expected_error):
        decoder.configure_model_and_context_info(
            model_info=model_info,
            context=operators.PipelineContext(),
            task_name="task_name",
            taskn=0,
            where=None,
            compiled_model_dir=Path('.'),
        )
        decoder.exec_torch(the_img, input_tensor, the_meta)

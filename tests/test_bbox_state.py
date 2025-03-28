# Copyright Axelera AI, 2024
from axelera.types import BoxFormat, ResizeMode
import numpy as np
import pytest

from axelera.app.meta import BBoxState


class TestBBoxState:
    # Tests the rescaling functionality of the BBoxState class for different input formats and rescaling modes.
    @pytest.mark.parametrize(
        "boxes, box_format, ori_shape, target_shape, ratio_pad, resize_mode, kpts, expected_rescaled_boxes, expected_rescaled_kpts",
        [
            # Test case 1: Rescale with xyxy format
            (
                np.array([[10, 20, 30, 40], [50, 60, 70, 80]], np.float64),
                BoxFormat.XYXY,
                (120, 100),
                (50, 50),
                None,
                ResizeMode.STRETCH,
                None,
                np.array([[5, 8.333333, 15, 16.666666], [25, 25, 35, 33.333333]], np.float64),
                None,
            ),
            # Test case 2: Rescale with xyxy format with letterbox
            (
                np.array([[10, 20, 30, 40], [50, 60, 70, 80]], np.float64),
                BoxFormat.XYXY,
                (120, 100),
                (50, 50),
                None,
                ResizeMode.LETTERBOX_FIT,
                None,
                np.array([[5, 5, 15, 15], [25, 25, 35, 35]], np.float64),
                None,
            ),
            # Test case 3: Rescale with ltwh format
            (
                np.array([[10, 20, 30, 40], [50, 60, 70, 80]], np.float64),
                BoxFormat.LTWH,
                (120, 100),
                (50, 50),
                None,
                ResizeMode.STRETCH,
                None,
                np.array([[5, 8.333333, 15, 16.666666], [25, 25, 35, 33.333333]], np.float64),
                None,
            ),
            # Test case 4: Rescale with ltwh with letterbox
            (
                np.array([[10, 20, 30, 40], [50, 60, 70, 80]], np.float64),
                BoxFormat.LTWH,
                (120, 100),
                (50, 50),
                None,
                ResizeMode.LETTERBOX_FIT,
                None,
                np.array([[5, 5, 15, 20], [25, 25, 35, 40]], np.float64),
                None,
            ),
            # Test case 5: Rescale with xywh format
            (
                np.array([[10, 20, 30, 40], [50, 60, 70, 80]], np.float64),
                BoxFormat.XYWH,
                (120, 100),
                (50, 50),
                None,
                ResizeMode.STRETCH,
                None,
                np.array([[5, 8.333333, 15, 16.666666], [25, 25, 35, 33.333333]], np.float64),
                None,
            ),
            # Test case 6: Rescale with xywh format with letterbox
            (
                np.array([[10, 20, 30, 40], [50, 60, 70, 80]], np.float64),
                BoxFormat.XYWH,
                (120, 100),
                (50, 50),
                None,
                ResizeMode.LETTERBOX_FIT,
                None,
                np.array([[5, 5, 15, 20], [25, 25, 35, 40]], np.float64),
                None,
            ),
            # Test case 7: Rescale with ratio_pad
            (
                np.array([[10, 20, 30, 40], [50, 60, 70, 80]], np.float64),
                BoxFormat.XYXY,
                (120, 100),
                (50, 50),
                (0.5, (10, 10)),
                ResizeMode.LETTERBOX_FIT,
                None,
                np.array([[0, 20, 40, 50], [50, 50, 50, 50]], np.float64),
                None,
            ),
            # Test case 8: Rescale with kpts
            (
                np.array([[10, 20, 30, 40], [50, 60, 70, 80]], np.float64),
                BoxFormat.XYXY,
                (120, 100),
                (50, 50),
                None,
                ResizeMode.LETTERBOX_FIT,
                np.array([[[15, 25], [35, 45]], [[75, 85], [95, 105]]], np.float64),
                np.array([[5, 5, 15, 15], [25, 25, 35, 35]], np.float64),
                np.array([[[7.5, 7.5], [17.5, 17.5]], [[37.5, 37.5], [47.5, 47.5]]], np.float64),
            ),
        ],
    )
    def test_rescale(
        self,
        boxes,
        box_format,
        ori_shape,
        target_shape,
        ratio_pad,
        resize_mode,
        kpts,
        expected_rescaled_boxes,
        expected_rescaled_kpts,
    ):
        rescaled_boxes, rescaled_kpts, _ = BBoxState.rescale(
            boxes,
            box_format,
            ori_shape,
            target_shape,
            ratio_pad=ratio_pad,
            resize_mode=resize_mode,
            kpts=kpts,
        )
        assert np.allclose(rescaled_boxes, expected_rescaled_boxes, atol=1e-6)
        if kpts is not None:
            assert np.allclose(rescaled_kpts, expected_rescaled_kpts, atol=1e-6)
        else:
            assert rescaled_kpts is None

    # Tests that the empty method returns True when there are no boxes in the BBoxState object.
    def test_empty(self):
        # happy path: empty BBoxState object
        bbox_state = BBoxState(10, 10, 20, 20, BoxFormat.XYXY, False, ResizeMode.STRETCH)
        assert bbox_state.empty()

        # general behavior: non-empty BBoxState object
        bbox_state = BBoxState(10, 10, 20, 20, BoxFormat.XYXY, False, ResizeMode.ORIGINAL)
        bbox_state.organize_bboxes([[0, 0, 10, 10]], [1.0], [0])
        assert not bbox_state.empty()

    def test_formatting(self):
        bbox_state = BBoxState(
            model_width=416,
            model_height=416,
            src_image_width=640,
            src_image_height=480,
            box_format=BoxFormat.XYWH,
            normalized_coord=True,
            scaled=ResizeMode.STRETCH,
        )
        bbox_state.organize_bboxes(
            [[0.3, 0.5, 0.1, 0.15], [0.7, 0.3, 0.2, 0.1]], [1.0, 0.8], [0, 1]
        )
        assert np.array_equal(bbox_state.xyxy(), bbox_state._boxes)
        expected_output = np.array([[160.0, 204.0, 224.0, 276.0], [384.0, 120.0, 512.0, 168.0]])
        assert np.allclose(bbox_state.xyxy(), expected_output)
        assert bbox_state.box_format == BoxFormat.XYXY
        assert bbox_state.normalized_coord == False
        assert bbox_state.scaled == ResizeMode.ORIGINAL

    # Tests that the nms method correctly performs class-based NMS and select the highest score
    def test_nms_with_class_based(self):
        pytest.importorskip('torchvision')
        boxes = np.array(
            [
                [10, 10, 50, 50],
                [10, 10, 50, 50],
                [15, 15, 55, 55],
                [100, 100, 150, 150],
                [100, 100, 150, 150],
                [105, 105, 155, 155],
            ]
        )
        scores = np.array([0.8, 0.9, 0.7, 0.95, 0.85, 0.75])
        class_ids = np.array([1, 1, 1, 2, 2, 2])

        bbox_state = BBoxState(
            model_width=300,
            model_height=300,
            src_image_width=300,
            src_image_height=300,
            box_format=BoxFormat.XYXY,
            normalized_coord=False,
            scaled=ResizeMode.ORIGINAL,
            nms_iou_threshold=0.5,
            nms_class_agnostic=False,
        )
        bbox_state.organize_bboxes(boxes, scores, class_ids)

        # Expected results after NMS; notice that the content is sorted by score
        expected_boxes = np.array([[100, 100, 150, 150], [10, 10, 50, 50]])
        expected_scores = np.array([0.95, 0.9])
        expected_class_ids = np.array([2, 1])

        np.testing.assert_array_equal(bbox_state._boxes, expected_boxes)
        np.testing.assert_array_equal(bbox_state._scores, expected_scores)
        np.testing.assert_array_equal(bbox_state._class_ids, expected_class_ids)
        np.testing.assert_array_equal(bbox_state._boxes, expected_boxes)
        np.testing.assert_array_equal(bbox_state._scores, expected_scores)
        np.testing.assert_array_equal(bbox_state._class_ids, expected_class_ids)

    # Tests that the nms with nms_class_agnostic as True
    def test_nms_with_class_agnostic(self):
        pytest.importorskip('torchvision')
        bbox_state = BBoxState(
            model_width=416,
            model_height=416,
            src_image_width=800,
            src_image_height=800,
            box_format=BoxFormat.XYXY,
            normalized_coord=False,
            scaled=ResizeMode.ORIGINAL,
            nms_max_boxes=5000,
            nms_iou_threshold=0.5,
            nms_class_agnostic=True,
            output_top_k=300,
        )

        boxes = np.array([[100, 100, 200, 200], [120, 110, 180, 210], [80, 110, 190, 200]])
        scores = np.array([0.9, 0.8, 0.7])
        class_ids = np.array([1, 2, 1])

        bbox_state.organize_bboxes(boxes, scores, class_ids)

        assert bbox_state._boxes.shape == (1, 4)
        assert bbox_state._scores.shape == (1,)
        assert bbox_state._class_ids.shape == (1,)
        assert np.allclose(bbox_state._boxes[0], [100, 100, 200, 200])
        assert np.allclose(bbox_state._scores[0], 0.9)
        assert np.allclose(bbox_state._class_ids[0], 1)

        # Tests that non-maximum suppression is correctly applied with the specified output_top_k value, keeping only the top k boxes with the highest scores.

    def test_nms_with_nms_topk(self):
        pytest.importorskip('torchvision')
        # Create a BBoxState object with some initial values
        bbox_state = BBoxState(
            model_width=416,
            model_height=416,
            src_image_width=800,
            src_image_height=800,
            box_format=BoxFormat.XYXY,
            normalized_coord=False,
            scaled=ResizeMode.ORIGINAL,
            nms_max_boxes=5000,
            nms_iou_threshold=0.5,
            nms_class_agnostic=False,
            output_top_k=2,
        )

        # Create some sample boxes, scores and class_ids
        boxes = [
            [100, 100, 200, 200],
            [150, 150, 250, 250],
            [300, 300, 400, 400],
            [200, 200, 300, 300],
            [250, 250, 350, 350],
        ]
        scores = [0.9, 0.8, 0.7, 0.95, 0.6]
        class_ids = [1, 2, 3, 4, 5]

        bbox_state.organize_bboxes(boxes, scores, class_ids)

        # Check that only the top 2 boxes are kept after NMS
        assert len(bbox_state._boxes) == 2
        assert len(bbox_state._scores) == 2
        assert len(bbox_state._class_ids) == 2
        assert np.allclose(bbox_state._boxes[0], [200, 200, 300, 300])
        assert np.allclose(bbox_state._boxes[1], [100, 100, 200, 200])
        assert np.allclose(bbox_state._scores[0], 0.95)
        assert np.allclose(bbox_state._scores[1], 0.9)
        assert np.allclose(bbox_state._class_ids[0], 4)
        assert np.allclose(bbox_state._class_ids[1], 1)

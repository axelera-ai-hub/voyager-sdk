# Copyright Axelera AI, 2024
import re
from unittest.mock import Mock, patch

import numpy as np
import pytest

from axelera.app import display
from axelera.app.eval_interfaces import KptDetEvalSample, KptDetGroundTruthSample
from axelera.app.meta.keypoint import (
    BottomUpKeypointDetectionMeta,
    CocoBodyKeypointsMeta,
    KeypointDetectionMeta,
    KeypointObjectWithBbox,
    TopDownKeypointDetectionMeta,
)


@pytest.fixture
def coco_body_keypoints_meta():
    keypoints = [
        [
            [427.0, 170.0, 1.0],
            [426.0, 169.0, 2.0],
            [0.0, -107.0, 0.0],
            [434.0, 167.9, 2.0],
            [0.0, -107.0, 0.0],
            [441.0, 177.0, 2.0],
            [446.0, 177.0, 2.0],
            [437.0, 200.0, 2.0],
            [430.0, 206.0, 2.0],
            [430.0, 220.0, 2.0],
            [420.0, 215.0, 2.0],
            [444.9, 225.9, 2.0],
            [452.0, 222.9, 2.0],
            [447.0, 260.0, 2.0],
            [454.0, 256.9, 2.0],
            [455.0, 289.9, 2.0],
            [458.9, 286.0, 2.0],
        ]
    ]
    boxes = [[412.0, 157.0, 465.0, 295.0]]
    scores = [0.9]
    return CocoBodyKeypointsMeta.from_list(keypoints=keypoints, boxes=boxes, scores=scores)


@pytest.fixture
def top_down_keypoints_meta():
    return TopDownKeypointDetectionMeta()


@pytest.fixture
def bottom_up_keypoints_meta():
    keypoints = np.array(
        [
            [
                [427.0, 170.0, 1.0],
                [426.0, 169.0, 2.0],
                [0.0, -107.0, 0.0],
                [434.0, 167.9, 2.0],
                [0.0, -107.0, 0.0],
                [441.0, 177.0, 2.0],
                [446.0, 177.0, 2.0],
                [437.0, 200.0, 2.0],
                [430.0, 206.0, 2.0],
                [430.0, 220.0, 2.0],
                [420.0, 215.0, 2.0],
                [444.9, 225.9, 2.0],
                [452.0, 222.9, 2.0],
                [447.0, 260.0, 2.0],
                [454.0, 256.9, 2.0],
                [455.0, 289.9, 2.0],
                [458.9, 286.0, 2.0],
            ]
        ]
    )
    boxes = np.array([[412.0, 157.0, 465.0, 295.0]])
    scores = np.array([0.9])
    return BottomUpKeypointDetectionMeta(keypoints=keypoints, boxes=boxes, scores=scores)


def test_meta_immutable(coco_body_keypoints_meta):
    with pytest.raises(AttributeError):
        coco_body_keypoints_meta.keypoints = [[[10, 20, 1], [30, 40, 1]]]


def test_xyxy(coco_body_keypoints_meta):
    assert np.allclose(coco_body_keypoints_meta.xyxy(), np.array([[412.0, 157.0, 465.0, 295.0]]))


def test_xywh(coco_body_keypoints_meta):
    expected_xywh = np.array([[438.5, 226.0, 53.0, 138.0]])
    assert np.allclose(coco_body_keypoints_meta.xywh(), expected_xywh)


def test_ltwh(coco_body_keypoints_meta):
    expected_ltwh = np.array([[412.0, 157.0, 53.0, 138.0]])
    assert np.allclose(coco_body_keypoints_meta.ltwh(), expected_ltwh)


def test_CocoBodyKeypointsMeta_to_evaluation(coco_body_keypoints_meta):
    pytest.importorskip("torch")
    ground_truth = KptDetGroundTruthSample.from_numpy(
        np.array([[412.0, 157.0, 465.0, 295.0]]),
        np.array([[427.0, 170.0, 1.0], [426.0, 169.0, 2.0]]),
    )
    with patch.object(KeypointDetectionMeta, 'access_ground_truth', return_value=ground_truth):
        eval_sample = coco_body_keypoints_meta.to_evaluation()
        assert isinstance(eval_sample, KptDetEvalSample)


def test_TopDownKeypointDetectionMeta_add_result(top_down_keypoints_meta):
    # Test adding keypoints as numpy array with boxes and scores
    keypoints = np.array([[10, 20, 1], [30, 40, 1]])
    boxes = np.array([1, 2, 3, 4])
    scores = 0.95

    top_down_keypoints_meta.add_result(keypoints, boxes, scores)

    assert len(top_down_keypoints_meta._keypoints) == 1
    assert len(top_down_keypoints_meta._boxes) == 1
    assert len(top_down_keypoints_meta._scores) == 1
    assert np.allclose(top_down_keypoints_meta._keypoints[0], keypoints)
    assert np.allclose(top_down_keypoints_meta._boxes[0], boxes)
    assert top_down_keypoints_meta._scores[0] == scores

    # Test adding keypoints without box and score
    keypoints_only = np.array([[70, 80, 0.9], [90, 100, 0.7]])
    top_down_keypoints_meta.add_result(keypoints_only)
    assert len(top_down_keypoints_meta._keypoints) == 2
    assert np.allclose(top_down_keypoints_meta._keypoints[1], keypoints_only)


def test_TopDownKeypointDetectionMeta_add_result_single_keypoint():
    # Create a new instance for this test
    meta = TopDownKeypointDetectionMeta()

    # Test adding a single keypoint as a tuple
    meta.add_result((50, 60, 0.8))
    assert len(meta._keypoints) == 1
    assert np.allclose(meta._keypoints[0], np.array([[50, 60, 0.8]]))

    # Test adding another single keypoint
    meta.add_result((70, 80, 0.9))
    assert len(meta._keypoints) == 2
    assert np.allclose(meta._keypoints[1], np.array([[70, 80, 0.9]]))


def test_TopDownKeypointDetectionMeta_add_result_different_shape():
    # Create a new instance for this test
    meta = TopDownKeypointDetectionMeta()

    # Add initial keypoints
    meta.add_result(np.array([[10, 20, 1], [30, 40, 1]]))

    # Test error when adding keypoints with different shape
    with pytest.raises(
        ValueError, match="keypoints must be of the same shape as the existing ones"
    ):
        meta.add_result(np.array([[10, 20]]))


@pytest.mark.parametrize(
    "keypoints, boxes, scores, expected_error",
    [
        (
            [[10, 20, 1], [30, 40, 1]],
            [1, 2, 3, 4],
            0.95,
            ValueError("keypoints must be either a tuple or a numpy array"),
        ),
        (
            np.array([[10, 20, 0.001, 1], [30, 40, 0.001, 1]]),
            np.array([1, 2, 3, 4]),
            0.95,
            ValueError(
                "keypoints must be a 2D numpy array with shape (N, 2) or (N, 3) containing int or float values"
            ),
        ),
        (
            np.array([[10, 20, 1], [30, 40, 1]]),
            [1, 2, 3, 4],
            0.95,
            ValueError("boxes must be a 1D numpy array with shape (4,)"),
        ),
        (
            np.array([[10, 20, 1], [30, 40, 1]]),
            np.array([[1, 2, 3, 4]]),
            0.95,
            ValueError("boxes must be a 1D numpy array with shape (4,)"),
        ),
        (
            np.array([[10, 20, 1], [30, 40, 1]]),
            np.array([1, 2, 3, 4]),
            [0.95],
            ValueError("scores must be a single scalar value"),
        ),
        (
            np.array([[10, 20, 1], [30, 40, 1]]),
            np.array([1, 2, 3, 4]),
            0.95,
            None,  # Valid input, no error expected
        ),
    ],
)
def test_TopDownKeypointDetectionMeta_add_result_invalid_input(
    top_down_keypoints_meta, keypoints, boxes, scores, expected_error
):
    if expected_error:
        with pytest.raises(type(expected_error), match=re.escape(str(expected_error))):
            top_down_keypoints_meta.add_result(keypoints, boxes, scores)
    else:
        top_down_keypoints_meta.add_result(keypoints, boxes, scores)
        assert len(top_down_keypoints_meta._keypoints) == 1
        assert len(top_down_keypoints_meta._boxes) == 1
        assert len(top_down_keypoints_meta._scores) == 1
        assert np.allclose(top_down_keypoints_meta._keypoints[0], keypoints)
        assert np.allclose(top_down_keypoints_meta._boxes[0], boxes)
        assert top_down_keypoints_meta._scores[0] == scores


@pytest.mark.parametrize(
    "keypoints, boxes, scores, expected_error",
    [
        (
            [[[10, 20, 1], [30, 40, 1]]],
            [[1, 2, 3, 4]],
            [0.95],
            TypeError("keypoints must be a numpy array, expected dimensions (N, K, 2 or 3)"),
        ),
        (
            np.array([[10, 20], [30, 40]]),
            np.array([[1, 2, 3, 4], [1, 2, 3, 4]]),
            np.array([[0.95], [0.95]]),
            ValueError("keypoints must be a 3D numpy array"),
        ),
        (
            np.array([[[10, 20, 1], [30, 40, 1]]]),
            [[1, 2, 3, 4]],
            [0.95],
            TypeError("boxes must be a numpy array, expected dimensions (N, 4)"),
        ),
        (
            np.array([[[10, 20, 1], [30, 40, 1]]]),
            np.array([[[1, 2, 3, 4]]]),
            np.array([[[0.95]]]),
            ValueError("boxes must be a 2D numpy array"),
        ),
        (
            np.array([[[10, 20, 1], [30, 40, 1]]]),
            np.array([[1, 2, 3, 4]]),
            [0.95],
            TypeError("scores must be a numpy array, expected dimensions (N,)"),
        ),
        (
            np.array([[[10, 20, 1], [30, 40, 1]]]),
            np.array([[1, 2, 3, 4]]),
            np.array([0.95]),
            None,  # Valid input, no error expected
        ),
        (
            np.array([]),
            np.array([]),
            np.array([]),
            None,  # Valid input, no error expected
        ),
    ],
)
def test_BottomUpKeypointDetectionMeta_init_invalid_input(
    keypoints, boxes, scores, expected_error
):
    if expected_error:
        with pytest.raises(type(expected_error), match=re.escape(str(expected_error))):
            BottomUpKeypointDetectionMeta(keypoints=keypoints, boxes=boxes, scores=scores)
    else:
        meta = BottomUpKeypointDetectionMeta(keypoints=keypoints, boxes=boxes, scores=scores)
        assert np.allclose(meta.keypoints, keypoints)
        assert np.allclose(meta.boxes, boxes)
        assert np.allclose(meta.scores, scores)


def test_BottomUpKeypointDetectionMeta_add_result_not_supported(bottom_up_keypoints_meta):
    keypoints = np.array([[[10, 20, 1], [30, 40, 1]]])
    boxes = np.array([[1, 2, 3, 4]])
    scores = np.array([0.95])

    with pytest.raises(
        ValueError,
        match=re.escape(
            f"{BottomUpKeypointDetectionMeta.__name__} does not support adding results object-by-object"
        ),
    ):
        bottom_up_keypoints_meta.add_result(keypoints, boxes, scores)


def test_TopDownKeypointDetectionMeta_to_evaluation(top_down_keypoints_meta):
    keypoints = np.array([[10, 20, 1], [30, 40, 1]])
    boxes = np.array([1, 2, 3, 4])
    scores = 0.95

    top_down_keypoints_meta.add_result(keypoints, boxes, scores)

    with pytest.raises(NotImplementedError):
        top_down_keypoints_meta.to_evaluation()


def test_BottomUpKeypointDetectionMeta_add_result_invalid_input(bottom_up_keypoints_meta):
    keypoints = np.array([[10, 20, 1], [30, 40, 1]])  # Invalid shape
    boxes = np.array([[1, 2, 3, 4]])
    scores = np.array([0.95])

    with pytest.raises(ValueError):
        bottom_up_keypoints_meta.add_result(keypoints, boxes, scores)


def test_BottomUpKeypointDetectionMeta_to_evaluation(bottom_up_keypoints_meta):
    container_meta = Mock()
    object.__setattr__(bottom_up_keypoints_meta, 'container_meta', container_meta)
    with pytest.raises(NotImplementedError):
        bottom_up_keypoints_meta.to_evaluation()


def test_coco_keypoint_detection_meta_draw(coco_body_keypoints_meta):
    draw = Mock(display.Draw)
    coco_body_keypoints_meta.draw(draw)
    expected_lines = [
        [np.array(p) for p in l]
        for l in [
            [
                [455.0, 289.9],
                [447.0, 260.0],
                [444.9, 225.9],
                [441.0, 177.0],
                [446.0, 177.0],
                [452.0, 222.9],
                [454.0, 256.9],
                [458.9, 286.0],
            ],  # body
            [[441.0, 177.0], [437.0, 200.0], [430.0, 220.0]],  # left arm
            [[446.0, 177.0], [430.0, 206.0], [420.0, 215.0]],  # right arm
            [[427.0, 170.0], [443.5, 177.0]],  # neck
        ]
    ]
    assert draw.polylines.call_count == 1
    assert all(
        [
            np.array_equal(got, exp)
            for got, exp in zip(
                [p for l in draw.polylines.call_args[0][0] for p in l],
                [p for l in expected_lines for p in l],
            )
        ]
    )
    assert draw.polylines.call_args[0][1:] == (False, (255, 255, 0, 255), 2)
    assert draw.keypoint.call_count == 15
    j = 0
    for i in range(len(coco_body_keypoints_meta.keypoints[0])):
        if coco_body_keypoints_meta.keypoints[0][i][2] > 0.5:
            assert np.array_equal(
                coco_body_keypoints_meta.keypoints[0][i][:2],
                draw.keypoint.call_args_list[j][0][0],
            )
            assert draw.keypoint.call_args_list[j][0][1:] == ((255, 0, 0, 255), 6)
            j += 1


def test_coco_body_keypoints_meta_objects():
    keypoints = [
        [
            [427.0, 170.0, 1.0],
            [426.0, 169.0, 2.0],
            [0.0, -107.0, 0.0],
            [434.0, 167.9, 2.0],
            [0.0, -107.0, 0.0],
            [441.0, 177.0, 2.0],
            [446.0, 177.0, 2.0],
            [437.0, 200.0, 2.0],
            [430.0, 206.0, 2.0],
            [430.0, 220.0, 2.0],
            [420.0, 215.0, 2.0],
            [444.9, 225.9, 2.0],
            [452.0, 222.9, 2.0],
            [447.0, 260.0, 2.0],
            [454.0, 256.9, 2.0],
            [455.0, 289.9, 2.0],
            [458.9, 286.0, 2.0],
        ]
    ]
    boxes = [[412.0, 157.0, 465.0, 295.0]]
    scores = [0.9]
    coco_body_keypoints_meta = CocoBodyKeypointsMeta.from_list(
        keypoints=keypoints, boxes=boxes, scores=scores
    )
    keypoint_detections = coco_body_keypoints_meta.objects
    assert len(keypoint_detections) == 1
    assert isinstance(keypoint_detections[0], KeypointObjectWithBbox)
    assert np.array_equal(keypoint_detections[0].keypoints, keypoints[0])
    assert np.array_equal(keypoint_detections[0].box, boxes[0])
    assert keypoint_detections[0].score == scores[0]

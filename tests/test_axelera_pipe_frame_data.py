# Copyright Axelera AI, 2024

import numpy as np
import pytest

from axelera.app.meta import (
    AxMeta,
    ClassificationMeta,
    CocoBodyKeypointsMeta,
    ObjectDetectionMeta,
    TrackerMeta,
)
from axelera.app.pipe import FrameResult


@pytest.mark.parametrize(
    "MetaClass, task_name, meta_args",
    [
        (ClassificationMeta, "classifications", []),
        (CocoBodyKeypointsMeta, "keypoint_detections", []),
        (
            ObjectDetectionMeta,
            "detections",
            [np.array([[10.0, 10.0, 20.0, 20.0]]), np.array([0.3]), np.array([1])],
        ),
        (TrackerMeta, "tracked_objects", []),
    ],
)
def test_frame_result_getattr(MetaClass, task_name, meta_args):
    meta = AxMeta("test")
    meta.add_instance(task_name, MetaClass(*meta_args))
    result = FrameResult(meta=meta)
    assert hasattr(result, task_name)


def test_frame_result_getattr_multiple_unique():
    meta = AxMeta("test")
    meta.add_instance("classifications", ClassificationMeta())
    meta.add_instance(
        "detections",
        ObjectDetectionMeta(np.array([[10.0, 10.0, 20.0, 20.0]]), np.array([0.3]), np.array([1])),
    )
    result = FrameResult(meta=meta)
    assert hasattr(result, "classifications")
    assert hasattr(result, "detections")
    assert not hasattr(result, "keypoint_detections")
    assert not hasattr(result, "pair_validations")
    assert not hasattr(result, "tracked_objects")

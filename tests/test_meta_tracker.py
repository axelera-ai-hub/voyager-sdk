# Copyright Axelera AI, 2024

import itertools

from axelera.app import utils
from axelera.app.meta import TrackedObject, TrackerMeta


def test_tracker_meta_objects():
    tracking_history = {1: [0, 1], 2: [2, 3]}
    class_ids = [1, 2]
    labels = utils.FrozenIntEnum("TestDataset", zip(["person", "car", "bus"], itertools.count()))

    meta = TrackerMeta(tracking_history, class_ids, labels=labels)
    tracked_objects = meta.objects
    assert len(tracked_objects) == 2
    assert isinstance(tracked_objects[0], TrackedObject)
    assert tracked_objects[0].track_id == 1
    assert tracked_objects[0].history == [0, 1]
    assert tracked_objects[0].class_id == 1
    assert tracked_objects[0].label == labels.car
    assert tracked_objects[0].label.name == "car"
    assert tracked_objects[0].is_car
    assert not tracked_objects[0].is_person
    assert not tracked_objects[0].is_bus
    assert isinstance(tracked_objects[1], TrackedObject)
    assert tracked_objects[1].track_id == 2
    assert tracked_objects[1].history == [2, 3]
    assert tracked_objects[1].class_id == 2
    assert tracked_objects[1].label == labels.bus
    assert tracked_objects[1].label.name == "bus"
    assert not tracked_objects[1].is_car
    assert not tracked_objects[1].is_person
    assert tracked_objects[1].is_bus

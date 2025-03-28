# Copyright Axelera AI, 2024
import dataclasses

import pytest

from axelera.app.meta.base import (
    AggregationNotRequiredForEvaluation,
    AxBaseTaskMeta,
    AxMeta,
    AxTaskMeta,
    MetaObject,
    NoMasterDetectionsError,
    RestrictedDict,
)


class Thingy(AxTaskMeta):
    pass


class Doodah(AxTaskMeta):
    pass


def test_restricted_dict():
    rd = RestrictedDict()
    rd["key1"] = "value1"
    assert rd["key1"] == "value1"

    with pytest.raises(Exception, match="An instance of str already exists for key 'key1'"):
        rd["key1"] = 123

    rd["key2"] = 123
    assert rd["key2"] == 123


def test_instance_mapping_methods():
    meta = AxMeta("image_id")
    with pytest.raises(
        AttributeError,
        match="Cannot set meta directly. Use add_instance or get_instance method to add or update",
    ):
        meta["thingy"] = Thingy()
    assert "thingy" not in meta

    thingy = meta.get_instance("thingy", Thingy)
    assert isinstance(thingy, Thingy)

    assert thingy.container_meta is meta
    assert "thingy" in meta
    assert meta["thingy"] is thingy
    assert meta.get("thingy") is thingy

    with pytest.raises(Exception, match="An instance of Thingy already exists for key 'thingy'"):
        meta.add_instance("thingy", Doodah)

    with pytest.raises(KeyError, match="unicorn not found in meta_map."):
        meta["unicorn"]
    assert 'unicorn' not in meta
    assert meta.get('unicorn') is None
    assert ["thingy"] == list(meta.keys())
    assert [thingy] == list(meta.values())
    assert [("thingy", thingy)] == list(meta.items())


def test_add_instance():
    meta = AxMeta("image_id")
    thingy = Thingy()
    meta.add_instance("thingy", thingy)
    assert "thingy" in meta
    assert meta["thingy"] is thingy
    assert thingy.container_meta is meta

    with pytest.raises(Exception, match="An instance of Thingy already exists for key 'thingy'"):
        meta.add_instance("thingy", Doodah())

    doodah = Doodah()
    meta.add_instance("doodah", doodah)
    assert "doodah" in meta
    assert meta["doodah"] is doodah


def test_add_instance_of_task_meta():
    meta = AxMeta("image_id")
    task_meta = AxTaskMeta()
    meta.add_instance("task_meta", task_meta)
    assert "task_meta" in meta
    assert meta["task_meta"] is task_meta
    assert task_meta.container_meta is meta


class MockAxMeta(AxMeta):
    def __init__(self, ground_truth=None):
        super().__init__(image_id="test_id")
        self.ground_truth = ground_truth


def test_access_ground_truth_with_container_meta_and_ground_truth():
    ground_truth = "some_ground_truth_data"
    container_meta = MockAxMeta(ground_truth=ground_truth)
    task_meta = AxTaskMeta()
    task_meta.set_container_meta(container_meta)
    assert (
        task_meta.access_ground_truth() == ground_truth
    ), "Should return the ground truth set in container_meta"


def test_access_ground_truth_without_container_meta():
    task_meta = AxTaskMeta()
    with pytest.raises(ValueError, match="AxMeta is not set"):
        task_meta.access_ground_truth()


def test_access_ground_truth_with_container_meta_no_ground_truth():
    container_meta = MockAxMeta()  # No ground truth provided
    task_meta = AxTaskMeta()
    task_meta.set_container_meta(container_meta)
    with pytest.raises(ValueError, match="Ground truth is not set"):
        task_meta.access_ground_truth()


def test_access_ground_truth_after_add_instance():
    ground_truth = "some_ground_truth_data"
    container_meta = AxMeta("image_id")
    container_meta.ground_truth = ground_truth

    task_meta = AxTaskMeta()
    container_meta.add_instance("task_meta", task_meta)

    assert (
        task_meta.access_ground_truth() == ground_truth
    ), "Should return the ground truth set in container_meta after add_instance"


def test_access_ground_truth_after_get_instance():
    ground_truth = "some_ground_truth_data"
    container_meta = AxMeta("image_id")
    container_meta.ground_truth = ground_truth

    task_meta = container_meta.get_instance("task_meta", AxTaskMeta)

    assert (
        task_meta.access_ground_truth() == ground_truth
    ), "Should return the ground truth set in container_meta after get_instance"


@dataclasses.dataclass(frozen=True)
class MockAxTaskMeta(AxTaskMeta):
    field1: int = 0
    field2: str = ""
    field3: float = 0.0


def test_keys():
    task_meta = MockAxTaskMeta()
    expected_keys = [
        'secondary_frame_indices',
        '_secondary_metas',
        'container_meta',
        'master_meta_name',
        'subframe_index',
        '_objects',
        'field1',
        'field2',
        'field3',
    ]
    assert (
        task_meta.members() == expected_keys
    ), "Should return all member variable names except 'container_meta'"


def test_keys_with_no_fields():
    @dataclasses.dataclass(frozen=True)
    class EmptyAxTaskMeta(AxTaskMeta):
        pass

    task_meta = EmptyAxTaskMeta()
    expected_keys = [
        'secondary_frame_indices',
        '_secondary_metas',
        'container_meta',
        'master_meta_name',
        'subframe_index',
        '_objects',
    ]

    assert task_meta.members() == expected_keys, "Should return all members from AxBaseTaskMeta"


def test_add_secondary_meta():
    meta = AxMeta("image_id")
    master_meta = Thingy()
    meta.add_instance("master", master_meta)

    secondary_meta = Doodah()
    meta.add_instance("secondary", secondary_meta, master_meta_name="master")

    assert master_meta.num_secondary_metas("secondary") == 1
    assert master_meta.get_secondary_meta("secondary", 0) is secondary_meta
    assert secondary_meta.container_meta is meta
    assert secondary_meta.master_meta_name == "master"
    assert secondary_meta.subframe_index == 0


def test_get_master_meta():
    meta = AxMeta("image_id")
    master_meta = Thingy()
    meta.add_instance("master", master_meta)

    secondary_meta = Doodah()
    meta.add_instance("secondary", secondary_meta, master_meta_name="master")

    assert secondary_meta.get_master_meta() is master_meta


class MockAggregateTaskMeta(AxTaskMeta):
    def __init__(self, value):
        super().__init__()
        self.value = value
        self.boxes = None

    @classmethod
    def aggregate(cls, meta_list):
        return cls(sum(meta.value for meta in meta_list))


def test_secondary_frame_indices():
    meta = AxMeta("image_id")
    task_meta = AxTaskMeta()
    meta.add_instance("master", task_meta)

    task_meta.add_secondary_frame_index("secondary1", 5)
    task_meta.add_secondary_frame_index("secondary1", 10)
    task_meta.add_secondary_frame_index("secondary1", 15)

    assert task_meta.secondary_frame_indices["secondary1"] == [5, 10, 15]
    assert task_meta.get_next_secondary_frame_index("secondary1") == 5
    assert task_meta.get_next_secondary_frame_index("secondary1") == 5

    secondary_meta1 = MockAggregateTaskMeta(2)
    meta.add_instance("secondary1", secondary_meta1, master_meta_name="master")
    assert secondary_meta1.subframe_index == 5

    assert task_meta.get_next_secondary_frame_index("secondary1") == 10

    secondary_meta2 = MockAggregateTaskMeta(3)
    meta.add_instance("secondary1", secondary_meta2, master_meta_name="master")
    assert secondary_meta2.subframe_index == 10
    assert task_meta.get_next_secondary_frame_index("secondary1") == 15

    secondary_meta3 = MockAggregateTaskMeta(4)
    meta.add_instance("secondary1", secondary_meta3, master_meta_name="master")
    assert secondary_meta3.subframe_index == 15

    with pytest.raises(IndexError, match="No more secondary frame indices available"):
        task_meta.get_next_secondary_frame_index("secondary1")


def test_secondary_frame_indices_master_with_two_secondary_tasks():
    meta = AxMeta("image_id")
    master_meta = AxTaskMeta()
    meta.add_instance("master", master_meta)

    # Add indices for two different secondary tasks
    master_meta.add_secondary_frame_index("secondary1", 5)
    master_meta.add_secondary_frame_index("secondary1", 10)
    master_meta.add_secondary_frame_index("secondary1", 15)

    master_meta.add_secondary_frame_index("secondary2", 20)
    master_meta.add_secondary_frame_index("secondary2", 25)
    master_meta.add_secondary_frame_index("secondary2", 30)

    # Check if indices are correctly stored
    assert master_meta.secondary_frame_indices["secondary1"] == [5, 10, 15]
    assert master_meta.secondary_frame_indices["secondary2"] == [20, 25, 30]

    # Test get_next_secondary_frame_index for both secondary tasks
    assert master_meta.get_next_secondary_frame_index("secondary1") == 5
    assert master_meta.get_next_secondary_frame_index("secondary2") == 20

    # Add instances for secondary1
    secondary1_meta1 = MockAggregateTaskMeta(2)
    meta.add_instance("secondary1", secondary1_meta1, master_meta_name="master")
    assert secondary1_meta1.subframe_index == 5

    assert master_meta.get_next_secondary_frame_index("secondary1") == 10

    secondary1_meta2 = MockAggregateTaskMeta(3)
    meta.add_instance("secondary1", secondary1_meta2, master_meta_name="master")
    assert secondary1_meta2.subframe_index == 10

    # Add instances for secondary2
    secondary2_meta1 = MockAggregateTaskMeta(4)
    meta.add_instance("secondary2", secondary2_meta1, master_meta_name="master")
    assert secondary2_meta1.subframe_index == 20

    assert master_meta.get_next_secondary_frame_index("secondary2") == 25

    # Exhaust all indices for secondary1
    secondary1_meta3 = MockAggregateTaskMeta(5)
    meta.add_instance("secondary1", secondary1_meta3, master_meta_name="master")
    assert secondary1_meta3.subframe_index == 15

    with pytest.raises(
        IndexError, match="No more secondary frame indices available for task: secondary1"
    ):
        master_meta.get_next_secondary_frame_index("secondary1")

    # secondary2 should still have available indices
    assert master_meta.get_next_secondary_frame_index("secondary2") == 25

    # Test for non-existent secondary task
    with pytest.raises(KeyError, match="No secondary frame indices found for task: non_existent"):
        master_meta.get_next_secondary_frame_index("non_existent")

    # Verify the number of secondary metas for each task
    assert master_meta.num_secondary_metas("secondary1") == 3
    assert master_meta.num_secondary_metas("secondary2") == 1


def test_inject_groundtruth():
    meta = AxMeta("image_id")
    ground_truth = object()  # Mock BaseEvalSample

    meta.inject_groundtruth(ground_truth)
    assert meta.ground_truth is ground_truth

    with pytest.raises(ValueError, match="Ground truth is already set"):
        meta.inject_groundtruth(object())


def test_aggregate_leaf_metas():
    meta = AxMeta("image_id")
    master_meta = MockAggregateTaskMeta(1)
    meta.add_instance("master", master_meta)

    meta.add_instance("secondary1", MockAggregateTaskMeta(2), master_meta_name="master")
    meta.add_instance("secondary1", MockAggregateTaskMeta(3), master_meta_name="master")

    aggregated = meta.aggregate_leaf_metas("master", "secondary1")
    assert len(aggregated) == 1
    assert isinstance(aggregated[0], MockAggregateTaskMeta)
    assert aggregated[0].value == 5  # Sum of all leaf meta values (2 + 3)

    # Test for non-existent master meta
    with pytest.raises(KeyError, match="non_existent not found in meta_map"):
        meta.aggregate_leaf_metas("non_existent", "secondary1")

    # Test for non-existent secondary meta
    with pytest.raises(NoMasterDetectionsError):
        meta.aggregate_leaf_metas("master", "non_existent_secondary")

    # Test for a meta with no secondary metas
    solo_meta = MockAggregateTaskMeta(value=10)
    meta.add_instance("solo", solo_meta)
    with pytest.raises(NoMasterDetectionsError):
        meta.aggregate_leaf_metas("solo", "secondary1")

    # Test for AggregationNotRequiredForEvaluation exception
    class NonAggregateTaskMeta(MockAggregateTaskMeta):
        @classmethod
        def aggregate(cls, meta_list):
            raise AggregationNotRequiredForEvaluation(cls)

    non_aggregate_meta = NonAggregateTaskMeta(value=1)
    meta.add_instance("non_aggregate", non_aggregate_meta)
    meta.add_instance(
        "non_aggregate_secondary", NonAggregateTaskMeta(value=2), master_meta_name="non_aggregate"
    )

    result = meta.aggregate_leaf_metas("non_aggregate", "non_aggregate_secondary")
    assert len(result) == 1
    assert isinstance(result[0], NonAggregateTaskMeta)
    assert result[0].value == 2

    # Test for multiple secondary tasks
    meta.add_instance(
        "non_aggregate_secondary2", NonAggregateTaskMeta(value=3), master_meta_name="non_aggregate"
    )

    with pytest.raises(ValueError, match="Multiple secondary tasks found for meta"):
        meta.aggregate_leaf_metas("non_aggregate", "non_aggregate_secondary")

    # Test for non-AxTaskMeta master
    non_task_meta = object()
    meta.add_instance("non_task", non_task_meta)
    with pytest.raises(ValueError, match="Master meta non_task is not an instance of AxTaskMeta"):
        meta.aggregate_leaf_metas("non_task", "secondary1")


def test_visit_with_lambda():
    calls = []

    class MockMeta(AxTaskMeta):
        def callee(self, *args):
            calls.append((self,) + args)

    meta = AxMeta("image_id")
    master_meta = MockMeta()
    meta.add_instance("master", master_meta)
    secondary_meta1 = MockMeta()
    meta.add_instance("secondary1", secondary_meta1, master_meta_name="master")
    secondary_meta2 = MockMeta()
    meta.add_instance("secondary2", secondary_meta2, master_meta_name="master")

    master_meta.visit(lambda m: m.callee(123))
    assert calls == [
        (master_meta, 123),
        (secondary_meta1, 123),
        (secondary_meta2, 123),
    ]


def test_visit_with_kwargs():
    calls = []

    class MockMeta(AxTaskMeta):
        def callee(self, arg, kwarg1=None):
            calls.append((self, arg, kwarg1))

    meta = AxMeta("image_id")
    master_meta = MockMeta()
    meta.add_instance("master", master_meta)
    secondary_meta1 = MockMeta()
    meta.add_instance("secondary1", secondary_meta1, master_meta_name="master")
    secondary_meta2 = MockMeta()
    meta.add_instance("secondary2", secondary_meta2, master_meta_name="master")

    master_meta.visit(MockMeta.callee, 123, kwarg1="value1")
    assert calls == [
        (master_meta, 123, "value1"),
        (secondary_meta1, 123, "value1"),
        (secondary_meta2, 123, "value1"),
    ]


def test_meta_object():
    class MockTaskMeta(AxTaskMeta):
        def __init__(self):
            super().__init__()
            object.__setattr__(self, '_secondary_metas', {"secondary": [AxTaskMeta()]})

        def get_secondary_meta(self, index):
            return self._secondary_metas["secondary"][index]

    meta = MockTaskMeta()
    meta_object = MetaObject(meta, 0)

    assert isinstance(meta_object.secondary_meta, AxTaskMeta)
    assert meta_object.secondary_meta is meta._secondary_metas["secondary"][0]

    class MockSecondaryTaskMeta(AxTaskMeta):
        Object = MetaObject

        def __init__(self):
            super().__init__()
            object.__setattr__(self, '_data', [1])  # Mock data for __len__

        def __len__(self):
            return len(self._data)

    secondary_meta = MockSecondaryTaskMeta()
    new_meta = MockTaskMeta()
    object.__setattr__(new_meta, '_secondary_metas', {"secondary": [secondary_meta]})
    new_meta_object = MetaObject(new_meta, 0)

    assert isinstance(new_meta_object.secondary_objects[0], MetaObject)
    assert len(new_meta_object.secondary_objects) == 1


def test_ax_base_task_meta():
    base_meta = AxBaseTaskMeta()

    assert base_meta.members() == [
        'secondary_frame_indices',
        '_secondary_metas',
        'container_meta',
        'master_meta_name',
        'subframe_index',
    ]

    container_meta = AxMeta("image_id")
    container_meta.ground_truth = "ground_truth"
    base_meta.set_container_meta(container_meta)

    assert base_meta.access_ground_truth() == "ground_truth"
    assert base_meta.access_image_id() == "image_id"

    base_meta.set_master_meta("master", 0)
    assert base_meta.master_meta_name == "master"
    assert base_meta.subframe_index == 0

    secondary_meta = AxBaseTaskMeta()
    base_meta.add_secondary_meta("secondary", secondary_meta)
    assert base_meta.get_secondary_meta("secondary", 0) == secondary_meta

    base_meta.add_secondary_frame_index("task", 1)
    assert base_meta.get_next_secondary_frame_index("task") == 1

    assert base_meta.num_secondary_metas("secondary") == 1
    assert base_meta.get_secondary_task_names() == ["secondary"]
    assert base_meta.has_secondary_metas() == True


def test_ax_task_meta():
    class MockTaskMeta(AxTaskMeta):
        def __init__(self):
            super().__init__()
            self._data = [1, 2, 3]  # Mock some data

        def __len__(self):
            return len(self._data)

        def draw(self, draw, **kwargs):
            pass

        def to_evaluation(self):
            return "evaluation"

        @classmethod
        def aggregate(cls, meta_list):
            return cls()

        @classmethod
        def decode(cls, data):
            return cls()

    task_meta = MockTaskMeta()

    with pytest.raises(NotImplementedError):
        task_meta.objects

    MockTaskMeta.Object = MetaObject
    assert isinstance(task_meta.objects[0], MetaObject)
    assert len(task_meta.objects) == 3  # Based on the mock data

    # Test that objects are cached
    assert task_meta.objects is task_meta.objects

    # Test that objects are created only once
    original_objects = task_meta.objects
    task_meta._data.append(4)  # This shouldn't affect the existing objects
    assert len(task_meta.objects) == 3
    assert task_meta.objects is original_objects


def test_ax_meta():
    meta = AxMeta("image_id")

    assert "key" not in meta

    with pytest.raises(KeyError):
        meta["key"]

    with pytest.raises(AttributeError):
        meta["key"] = "value"

    class MockTaskMeta(AxTaskMeta):
        pass

    meta.add_instance("key", MockTaskMeta())
    assert "key" in meta
    assert isinstance(meta["key"], MockTaskMeta)

    meta.delete_instance("key")
    assert "key" not in meta

    instance = meta.get_instance("new_key", MockTaskMeta)
    assert isinstance(instance, MockTaskMeta)
    assert "new_key" in meta

    with pytest.raises(ValueError):
        meta.inject_groundtruth("ground_truth")
        meta.inject_groundtruth("another_ground_truth")

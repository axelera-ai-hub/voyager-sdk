# Copyright Axelera AI, 2025
import os
from unittest.mock import Mock, call

import pytest

from axelera.app import config, display
from axelera.app.meta import AxMeta, AxTaskMeta, base

base.safe_label_format.cache_clear()

LABELS = ['', 'chicken', 'duck', 'goose', 'ostrich', 'badger', 'aardvark', 'mushroom']
DUCK = 2
GOOST = 3
BADGER = 5
BADGER_COLOR = (0, 0, 136, 255)
DUCK_COLOR = (255, 255, 255, 255)
GOOSE_COLOR = (132, 185, 255, 255)


def _make_meta(box, score, cls):
    meta = Mock()
    meta.boxes = [box]
    meta.scores = [score]
    meta.class_ids = [cls]
    meta.labels = LABELS
    return meta


def _add_meta(meta, box, score, cls):
    meta.boxes.append(box)
    meta.scores.append(score)
    meta.class_ids.append(cls)


JUST_A_BADGER = _make_meta((10, 20, 30, 40), 0.9, BADGER)

BIRDS = _make_meta((10, 20, 30, 40), 0.9, DUCK)
_add_meta(BIRDS, (50, 60, 70, 80), 0.8, GOOST)


def _make_draw_with_options(**kwargs):
    if not display.HAVE_RUNTIME2:
        pytest.skip('axelera.runtime2 is not installed')
    draw = Mock()
    draw.options = display.Options(**kwargs)
    return draw


def test_draw_bounding_box_bad_format(caplog):
    draw = _make_draw_with_options(bbox_label_format='{')
    base.draw_bounding_boxes(JUST_A_BADGER, draw)
    draw.labelled_box.assert_called_once_with((10, 20), (30, 40), 'badger 90%', BADGER_COLOR)
    assert "Error in bbox_label_format: { (Single '{' encountered" in caplog.text


def test_draw_bounding_box_bad_macro(caplog):
    draw = _make_draw_with_options(bbox_label_format='{sroce}')
    base.draw_bounding_boxes(JUST_A_BADGER, draw)
    draw.labelled_box.assert_called_once_with((10, 20), (30, 40), 'badger 90%', BADGER_COLOR)
    assert (
        "Unknown name 'sroce' in bbox_label_format '{sroce}', valid names are label, score, scorep"
        in caplog.text
    )


def test_draw_bounding_box_ok(caplog):
    draw = _make_draw_with_options(bbox_label_format='{scorep:.0f}%')
    base.draw_bounding_boxes(JUST_A_BADGER, draw)
    assert caplog.text == ''
    draw.labelled_box.assert_called_once_with((10, 20), (30, 40), '90%', BADGER_COLOR)


def test_draw_bounding_box_multiple(caplog):
    draw = _make_draw_with_options()
    base.draw_bounding_boxes(BIRDS, draw)
    assert caplog.text == ''
    draw.labelled_box.assert_has_calls(
        [
            call((10, 20), (30, 40), 'duck 90%', DUCK_COLOR),
            call((50, 60), (70, 80), 'goose 80%', GOOSE_COLOR),
        ]
    )


def test_draw_bounding_box_hide_class(caplog):
    """Test that class labels are hidden when AXELERA_RENDER_BBOX_CLASS is disabled."""
    original_show_class = os.environ.get("AXELERA_RENDER_BBOX_CLASS")
    original_show_score = os.environ.get("AXELERA_RENDER_BBOX_SCORE")

    try:
        os.environ["AXELERA_RENDER_BBOX_CLASS"] = "0"
        os.environ["AXELERA_RENDER_BBOX_SCORE"] = "1"

        draw = _make_draw_with_options()
        base.draw_bounding_boxes(BIRDS, draw)

        assert caplog.text == ''

        draw.labelled_box.assert_has_calls(
            [
                call((10, 20), (30, 40), '0.90', DUCK_COLOR),  # Only score, no "duck" class
                call((50, 60), (70, 80), '0.80', GOOSE_COLOR),  # Only score, no "goose" class
            ]
        )
    finally:
        if original_show_class is not None:
            os.environ["AXELERA_RENDER_BBOX_CLASS"] = original_show_class
        else:
            os.environ.pop("AXELERA_RENDER_BBOX_CLASS", None)

        if original_show_score is not None:
            os.environ["AXELERA_RENDER_BBOX_SCORE"] = original_show_score
        else:
            os.environ.pop("AXELERA_RENDER_BBOX_SCORE", None)


def test_draw_bounding_box_hide_class_no_label(caplog):
    draw = _make_draw_with_options()
    base.draw_bounding_boxes(BIRDS, draw, show_labels=False)
    assert caplog.text == ''
    draw.labelled_box.assert_has_calls(
        [
            call((10, 20), (30, 40), '', DUCK_COLOR),
            call((50, 60), (70, 80), '', GOOSE_COLOR),
        ]
    )


def test_draw_bounding_box_with_color_override(caplog):
    NEW_DUCK_COLOR = (0, 0, 255, 255)
    assert NEW_DUCK_COLOR != DUCK_COLOR
    draw = _make_draw_with_options(
        bbox_class_colors={
            DUCK: NEW_DUCK_COLOR,
        }
    )
    base.draw_bounding_boxes(BIRDS, draw)
    assert caplog.text == ''
    draw.labelled_box.assert_has_calls(
        [
            call((10, 20), (30, 40), 'duck 90%', NEW_DUCK_COLOR),
            call((50, 60), (70, 80), 'goose 80%', GOOSE_COLOR),
        ]
    )


def test_draw_bounding_box_with_color_override_by_name(caplog):
    NEW_DUCK_COLOR = (0, 0, 255, 255)
    assert NEW_DUCK_COLOR != DUCK_COLOR
    draw = _make_draw_with_options(
        bbox_class_colors={
            'duck': NEW_DUCK_COLOR,
        }
    )
    base.draw_bounding_boxes(BIRDS, draw)
    assert caplog.text == ''
    draw.labelled_box.assert_has_calls(
        [
            call((10, 20), (30, 40), 'duck 90%', NEW_DUCK_COLOR),
            call((50, 60), (70, 80), 'goose 80%', GOOSE_COLOR),
        ]
    )


def test_task_render_config_property():
    """Test the task_render_config property of AxBaseTaskMeta."""
    container = AxMeta("test_image")

    render_config = config.RenderConfig()
    render_config.set_task("task1", show_labels=True, show_annotations=False, force_register=True)
    render_config.set_task(
        "TaskClass", show_labels=False, show_annotations=True, force_register=True
    )

    container.set_render_config(render_config)

    class TestTaskMeta(AxTaskMeta):
        def draw(self, draw, **kwargs):
            pass

        def __len__(self):
            return 0

    task1 = TestTaskMeta()
    container.add_instance("task1", task1)
    assert task1.meta_name == "task1"

    assert task1.task_render_config is not None
    assert task1.task_render_config.show_labels is True
    assert task1.task_render_config.show_annotations is False

    task2 = TestTaskMeta()
    container.add_instance("task2", task2)
    assert task2.task_render_config == config.DEFAULT_RENDER_CONFIG

    task_no_container = TestTaskMeta()
    with pytest.raises(ValueError, match="not part of a container"):
        task_no_container.task_render_config

    container_no_config = AxMeta("test_image")
    task_no_config = TestTaskMeta()
    container_no_config.add_instance("task3", task_no_config)
    with pytest.raises(ValueError, match="not set in the container"):
        task_no_config.task_render_config

    class TaskClass(AxTaskMeta):
        def draw(self, draw, **kwargs):
            pass

        def __len__(self):
            return 0

    task_class = TaskClass()
    container.add_instance("different_name", task_class)
    assert task_class.task_render_config == config.DEFAULT_RENDER_CONFIG


def test_task_render_config_in_secondary_meta():
    """Test that task_render_config works properly for secondary metas."""
    import pytest

    from axelera.app import config

    container = AxMeta("test_image")
    render_config = config.RenderConfig()
    render_config.set_task(
        "master_task", show_labels=True, show_annotations=True, force_register=True
    )
    render_config.set_task(
        "secondary_task", show_labels=False, show_annotations=True, force_register=True
    )

    container.set_render_config(render_config)

    class MasterTaskMeta(AxTaskMeta):
        def draw(self, draw, **kwargs):
            pass

        def __len__(self):
            return 0

    class SecondaryTaskMeta(AxTaskMeta):
        def draw(self, draw, **kwargs):
            pass

        def __len__(self):
            return 0

    master_meta = MasterTaskMeta()
    container.add_instance("master_task", master_meta)
    secondary_meta = SecondaryTaskMeta()
    container.add_instance("secondary_task", secondary_meta, master_meta_name="master_task")

    assert master_meta.meta_name == "master_task"
    assert master_meta.task_render_config is not None
    assert master_meta.task_render_config.show_labels is True
    assert master_meta.task_render_config.show_annotations is True

    assert secondary_meta.meta_name == "secondary_task"
    assert secondary_meta.task_render_config is not None
    assert secondary_meta.task_render_config.show_labels is False
    assert secondary_meta.task_render_config.show_annotations is True

    assert secondary_meta.master_meta_name == "master_task"
    assert secondary_meta.container_meta is container
    assert secondary_meta.get_master_meta() is master_meta

# Copyright Axelera AI, 2025
from unittest.mock import Mock, call

from axelera.app import display
from axelera.app.meta import base

base._safe_label_format.cache_clear()

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
    draw = Mock()
    draw.options = display.Options(**kwargs)
    return draw


def test_draw_bounding_box_bad_format(caplog):
    draw = _make_draw_with_options(bbox_label_format='{')
    base.draw_bounding_boxes(JUST_A_BADGER, draw)
    draw.labelled_box.assert_called_once_with((10, 20), (30, 40), 'badger 0.90', BADGER_COLOR)
    assert "Error in bbox_label_format: { (Single '{' encountered" in caplog.text


def test_draw_bounding_box_bad_macro(caplog):
    draw = _make_draw_with_options(bbox_label_format='{sroce}')
    base.draw_bounding_boxes(JUST_A_BADGER, draw)
    draw.labelled_box.assert_called_once_with((10, 20), (30, 40), 'badger 0.90', BADGER_COLOR)
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
            call((10, 20), (30, 40), 'duck 0.90', DUCK_COLOR),
            call((50, 60), (70, 80), 'goose 0.80', GOOSE_COLOR),
        ]
    )


def test_draw_bounding_box_hide_class(caplog):
    draw = _make_draw_with_options()
    base.draw_bounding_boxes(BIRDS, draw, show_class=False)
    assert caplog.text == ''
    draw.labelled_box.assert_has_calls(
        [
            call((10, 20), (30, 40), '0.90', DUCK_COLOR),
            call((50, 60), (70, 80), '0.80', GOOSE_COLOR),
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
            call((10, 20), (30, 40), 'duck 0.90', NEW_DUCK_COLOR),
            call((50, 60), (70, 80), 'goose 0.80', GOOSE_COLOR),
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
            call((10, 20), (30, 40), 'duck 0.90', NEW_DUCK_COLOR),
            call((50, 60), (70, 80), 'goose 0.80', GOOSE_COLOR),
        ]
    )

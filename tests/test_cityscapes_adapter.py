# Copyright Axelera AI, 2026
"""Smoke + correctness tests for the in-house Cityscapes adapter.

These tests do NOT require Cityscapes data on disk; they exercise the
labelId->trainId map, the adapter's import surface, and the no-mmlab
dependency invariant via source grep.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ADAPTER_PATH = Path(__file__).parent.parent / 'ax_datasets' / 'cityscapes.py'


def test_adapter_has_no_mmlab_imports():
    """Source-level guarantee: no mmseg / mmengine / mmcv / mmdeploy / cityscapesscripts."""
    text = ADAPTER_PATH.read_text()
    for banned in ('mmseg', 'mmengine', 'mmcv', 'mmdeploy', 'cityscapesscripts'):
        # Allow the banned token inside comments / docstrings (it appears as
        # documentation in our adapter). Reject only on import statements.
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith('import ') or stripped.startswith('from '):
                assert (
                    banned not in line
                ), f'{ADAPTER_PATH.name}: forbidden import of {banned!r}: {line!r}'


def test_adapter_imports_cleanly():
    import ax_datasets.cityscapes  # noqa: F401
    from ax_datasets.cityscapes import (
        CityscapesSemSegDataAdapter,
        IGNORE_INDEX,
        LABEL_ID_TO_TRAIN_ID,
        NUM_CLASSES,
        labelid_to_trainid,
    )

    assert NUM_CLASSES == 19
    assert IGNORE_INDEX == 255
    assert callable(labelid_to_trainid)
    assert callable(CityscapesSemSegDataAdapter)


def test_label_map_19_classes():
    from ax_datasets.cityscapes import LABEL_ID_TO_TRAIN_ID, NUM_CLASSES

    assert NUM_CLASSES == 19
    train_ids = [v for v in LABEL_ID_TO_TRAIN_ID.values() if v != 255]
    assert sorted(set(train_ids)) == list(range(19))


def test_label_map_canonical_assignments():
    from ax_datasets.cityscapes import LABEL_ID_TO_TRAIN_ID

    # Spot-check Cordts 2016 sec. 3.2 canonical mapping
    assert LABEL_ID_TO_TRAIN_ID[7] == 0  # road
    assert LABEL_ID_TO_TRAIN_ID[8] == 1  # sidewalk
    assert LABEL_ID_TO_TRAIN_ID[11] == 2  # building
    assert LABEL_ID_TO_TRAIN_ID[26] == 13  # car
    assert LABEL_ID_TO_TRAIN_ID[33] == 18  # bicycle
    # Ignored labels
    assert LABEL_ID_TO_TRAIN_ID[0] == 255
    assert LABEL_ID_TO_TRAIN_ID[9] == 255  # parking
    assert LABEL_ID_TO_TRAIN_ID[14] == 255  # guard rail


def test_labelid_to_trainid_vectorized():
    from ax_datasets.cityscapes import IGNORE_INDEX, labelid_to_trainid

    raw = np.array([[7, 8, 11, 12], [26, 0, 33, 9]], dtype=np.uint8)
    expected = np.array([[0, 1, 2, 3], [13, IGNORE_INDEX, 18, IGNORE_INDEX]], dtype=np.uint8)
    np.testing.assert_array_equal(labelid_to_trainid(raw), expected)


def test_labelid_to_trainid_unknown_label_is_ignore():
    from ax_datasets.cityscapes import IGNORE_INDEX, labelid_to_trainid

    # Labels >33 are not in spec; map to ignore_index defensively.
    raw = np.array([[200, 250]], dtype=np.uint8)
    out = labelid_to_trainid(raw)
    assert (out == IGNORE_INDEX).all()


def test_cityscapes_names_file_has_19_classes():
    names_file = Path(__file__).parent.parent / 'ax_datasets' / 'labels' / 'cityscapes.names'
    lines = [line.strip() for line in names_file.read_text().splitlines() if line.strip()]
    assert len(lines) == 19
    assert lines[0] == 'road'
    assert lines[13] == 'car'
    assert lines[18] == 'bicycle'

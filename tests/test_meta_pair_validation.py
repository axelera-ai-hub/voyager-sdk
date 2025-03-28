# Copyright Axelera AI, 2024
import numpy as np
import pytest

from axelera.app.eval_interfaces import PairValidationGroundTruthSample
from axelera.app.meta.pair_validation import PairValidationMeta


def test_add_result_single_value():
    meta = PairValidationMeta()
    result = np.array([0.8])
    assert meta.add_result(result) == False
    assert np.array_equal(meta.result1, np.array([result]))
    assert np.array_equal(meta.result2, np.array([]))


def test_add_result_1d_array():
    meta = PairValidationMeta()
    result1 = np.array([0.8, 0.6])
    result2 = np.array([0.7, 0.9])
    assert meta.add_result(result1) == False
    assert meta.add_result(result2) == True
    assert np.array_equal(meta.result1, result1.reshape(1, -1))
    assert np.array_equal(meta.result2, result2.reshape(1, -1))


def test_add_result_2d_array():
    meta = PairValidationMeta()
    result1 = np.array([[0.8, 0.6], [0.7, 0.9]])
    result2 = np.array([[0.5, 0.4], [0.3, 0.2]])
    assert meta.add_result(result1) == False
    assert meta.add_result(result2) == True
    assert np.array_equal(meta.result1, result1)
    assert np.array_equal(meta.result2, result2)


def test_add_result_error_case():
    meta = PairValidationMeta()
    result1 = np.array([0.8])
    result2 = np.array([0.7])
    result3 = np.array([0.6])
    meta.add_result(result1)
    meta.add_result(result2)
    with pytest.raises(ValueError):
        meta.add_result(result3)


def test_to_evaluation():
    meta = PairValidationMeta()
    result1 = np.array([0.8, 0.6])
    result2 = np.array([0.7, 0.9])
    meta.add_result(result1)
    meta.add_result(result2)
    ground_truth = PairValidationGroundTruthSample(the_same=False)
    # Use a workaround to set the lambda for a frozen dataclass
    object.__setattr__(meta, 'access_ground_truth', lambda: ground_truth)
    eval_sample = meta.to_evaluation()
    assert np.array_equal(eval_sample.embedding_1, result1.reshape(1, -1))
    assert np.array_equal(eval_sample.embedding_2, result2.reshape(1, -1))


def test_draw():
    meta = PairValidationMeta()
    with pytest.raises(RuntimeError):
        meta.draw(None)

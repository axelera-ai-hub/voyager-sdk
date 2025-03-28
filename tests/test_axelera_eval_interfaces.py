# Copyright Axelera AI, 2024

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from axelera import types
from axelera.app import eval_interfaces


def test_classification_eval_sample():
    element = eval_interfaces.ClassificationEvalSample(
        num_classes=3, class_ids=[1, 2], scores=[0.9, 0.8]
    )
    data = element.data
    assert isinstance(data, dict)
    assert data[1] == 0.9
    assert data[2] == 0.8


def test_objdet_eval_sample():
    boxes = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    labels = torch.tensor([1, 2])
    scores = torch.tensor([0.9, 0.8])
    element = eval_interfaces.ObjDetEvalSample(boxes, labels, scores)
    data = element.data
    assert isinstance(data, dict)
    assert torch.allclose(data['boxes'], boxes)
    assert torch.allclose(data['labels'], labels)
    assert torch.allclose(data['scores'], scores)


def test_objdet_eval_sample_from_numpy():
    boxes = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    labels = np.array([1, 2])
    scores = np.array([0.9, 0.8])
    element = eval_interfaces.ObjDetEvalSample.from_numpy(boxes, labels, scores)
    assert np.allclose(element.boxes, boxes)
    assert np.allclose(element.labels, labels)
    assert np.allclose(element.scores, scores)


def test_objdet_eval_sample_from_list():
    boxes = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    labels = [1, 2]
    scores = [0.9, 0.8]
    element = eval_interfaces.ObjDetEvalSample.from_list(boxes, labels, scores)
    assert np.allclose(element.boxes, boxes)
    assert np.allclose(element.labels, labels)
    assert np.allclose(element.scores, scores)


def test_objdet_ground_truth_sample():
    boxes = torch.tensor([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    labels = torch.tensor([1, 2])
    img_id = 'image1'
    element = eval_interfaces.ObjDetGroundTruthSample(boxes, labels, img_id)
    data = element.data
    assert isinstance(data, dict)
    assert torch.allclose(data['boxes'], boxes)
    assert torch.allclose(data['labels'], labels)
    assert data['img_id'] == img_id


def test_objdet_ground_truth_sample_from_numpy():
    boxes = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    labels = np.array([1, 2])
    img_id = 'image1'
    element = eval_interfaces.ObjDetGroundTruthSample.from_numpy(boxes, labels, img_id)
    assert np.allclose(element.boxes, boxes)
    assert np.allclose(element.labels, labels)
    assert element.img_id == img_id


def test_objdet_ground_truth_sample_from_list():
    boxes = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    labels = [1, 2]
    img_id = 'image1'
    element = eval_interfaces.ObjDetGroundTruthSample.from_list(boxes, labels, img_id)
    assert np.allclose(element.boxes, boxes)
    assert np.allclose(element.labels, labels)
    assert element.img_id == img_id


def test_pair_validation_eval_sample():
    embedding_1 = np.array([0.1, 0.2, 0.3])
    embedding_2 = np.array([0.4, 0.5, 0.6])
    element = eval_interfaces.PairValidationEvalSample(embedding_1, embedding_2)

    assert np.array_equal(element.embedding_1, embedding_1)
    assert np.array_equal(element.embedding_2, embedding_2)
    assert element.data == {'embedding_1': embedding_1, 'embedding_2': embedding_2}

    # Test from_numpy method
    element_from_numpy = eval_interfaces.PairValidationEvalSample.from_numpy(
        embedding_1, embedding_2
    )
    assert element_from_numpy == element

    # Test from_torch method
    embedding_1_torch = torch.tensor([0.1, 0.2, 0.3])
    embedding_2_torch = torch.tensor([0.4, 0.5, 0.6])
    element_from_torch = eval_interfaces.PairValidationEvalSample.from_torch(
        embedding_1_torch, embedding_2_torch
    )
    assert np.array_equal(element_from_torch.embedding_1, embedding_1_torch.numpy())
    assert np.array_equal(element_from_torch.embedding_2, embedding_2_torch.numpy())


def test_pair_validation_ground_truth_sample():
    the_same = True
    element = eval_interfaces.PairValidationGroundTruthSample(the_same)

    assert element.the_same == the_same
    assert element.data == the_same

# Copyright Axelera AI, 2024
import json
from pathlib import Path
from unittest.mock import MagicMock, mock_open, patch

import numpy as np
import pytest

from axelera.app.model_utils.embeddings import (
    DistanceMetric,
    EmbeddingsFile,
    JSONEmbeddingsFile,
    NumpyEmbeddingsFile,
    cosine_distance,
    cosine_similarity,
    euclidean_distance,
    open_embeddings_file,
    squared_euclidean_distance,
)


@pytest.fixture
def temp_file(tmp_path):
    return tmp_path / "temp_file"


@pytest.mark.parametrize(
    "embeddings1, embeddings2, expected",
    [
        (np.array([[1, 2], [3, 4]]), np.array([[5, 6], [7, 8]]), np.array([32, 32])),
        (np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]), np.array([0, 0])),
        (np.array([[-1, -2], [-3, -4]]), np.array([[1, 2], [3, 4]]), np.array([20, 100])),
    ],
)
def test_squared_euclidean_distance(embeddings1, embeddings2, expected):
    result = squared_euclidean_distance(embeddings1, embeddings2)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "embeddings1, embeddings2, expected",
    [
        (
            np.array([[1, 2], [3, 4]]),
            np.array([[5, 6], [7, 8]]),
            np.array([5.65685425, 5.65685425]),
        ),
        (np.array([[0, 0], [0, 0]]), np.array([[0, 0], [0, 0]]), np.array([0, 0])),
        (np.array([[-1, -2], [-3, -4]]), np.array([[1, 2], [3, 4]]), np.array([4.47213595, 10.0])),
    ],
)
def test_euclidean_distance(embeddings1, embeddings2, expected):
    result = euclidean_distance(embeddings1, embeddings2)
    np.testing.assert_allclose(result, expected, rtol=1e-7, atol=1e-7)


@pytest.mark.parametrize(
    "embeddings1, embeddings2, expected",
    [
        (np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]), np.array([1, 1])),
        # orthogonal vectors
        (np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]]), np.array([0, 0])),
        # same direction
        (np.array([[0, 0], [0, 0]]), np.array([[1, 0], [0, 1]]), np.array([0, 0])),
        # opposite direction
        (np.array([[-1, 0], [0, -1]]), np.array([[1, 0], [0, 1]]), np.array([-1, -1])),
    ],
)
def test_cosine_similarity(embeddings1, embeddings2, expected):
    result = cosine_similarity(embeddings1, embeddings2)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "embeddings1, embeddings2, expected",
    [
        (np.array([[1, 0], [0, 1]]), np.array([[1, 0], [0, 1]]), np.array([0, 0])),
        (np.array([[1, 0], [0, 1]]), np.array([[0, 1], [1, 0]]), np.array([0.5, 0.5])),
        (np.array([[0, 0], [0, 0]]), np.array([[1, 0], [0, 1]]), np.array([0.5, 0.5])),
        (np.array([[-1, 0], [0, -1]]), np.array([[1, 0], [0, 1]]), np.array([1, 1])),
    ],
)
def test_cosine_distance(embeddings1, embeddings2, expected):
    result = cosine_distance(embeddings1, embeddings2)
    assert np.array_equal(result, expected)


@pytest.mark.parametrize(
    "embedding_shape, num_embeddings",
    [
        ((1, 3), 1),
        ((1, 3), 3),
        ((1, 512), 1),
        ((1, 512), 5),
        ((2, 256), 1),
        ((2, 256), 4),
    ],
)
def test_json_embeddings_file(temp_file, embedding_shape, num_embeddings):
    temp_file = temp_file.with_suffix('.json')
    json_file = JSONEmbeddingsFile(temp_file)

    embeddings = [np.random.rand(*embedding_shape) for _ in range(num_embeddings)]

    for i, emb in enumerate(embeddings):
        json_file.update(emb, i)

    assert json_file.dirty is True
    for i, emb in enumerate(embeddings):
        assert np.array_equal(np.array(json_file.embedding_dict[i]).flatten(), emb.flatten())

    json_file.commit()
    assert json_file.dirty is False

    with temp_file.open('r') as f:
        data = json.load(f)
    for i, emb in enumerate(embeddings):
        assert np.array_equal(np.array(data[str(i)]).flatten(), emb.flatten())

    labels = json_file.read_labels()
    assert labels == [str(i) for i in range(num_embeddings)]

    loaded_embeddings = json_file.load_embeddings()
    assert loaded_embeddings.shape == (num_embeddings, np.prod(embedding_shape))
    for i, emb in enumerate(embeddings):
        assert np.array_equal(loaded_embeddings[i], emb.flatten())


@pytest.mark.parametrize(
    "embedding_shape, num_embeddings",
    [
        ((1, 3), 1),
        ((1, 3), 3),
        ((1, 512), 1),
        ((1, 512), 5),
        ((2, 256), 1),
        ((2, 256), 4),
    ],
)
def test_numpy_embeddings_file(temp_file, embedding_shape, num_embeddings):
    temp_file = temp_file.with_suffix('.npy')
    npy_file = NumpyEmbeddingsFile(temp_file)

    embeddings = [np.random.rand(*embedding_shape) for _ in range(num_embeddings)]

    for i, emb in enumerate(embeddings):
        npy_file.update(emb, f"img_{i}")

    assert npy_file.dirty is True
    assert npy_file.embedding_array.shape == (num_embeddings, np.prod(embedding_shape))

    # Test updating an existing embedding (should not change)
    original_embedding = npy_file.embedding_array[0].copy()
    updated_embedding = np.random.rand(*embedding_shape)
    npy_file.update(updated_embedding, "img_0")

    assert np.array_equal(npy_file.embedding_array[0], original_embedding)

    # Test adding a new embedding
    new_embedding = np.random.rand(*embedding_shape)
    npy_file.update(new_embedding, f"img_{num_embeddings}")

    assert npy_file.embedding_array.shape == (num_embeddings + 1, np.prod(embedding_shape))
    assert np.array_equal(npy_file.embedding_array[-1], new_embedding.flatten())

    npy_file.commit()
    assert npy_file.dirty is False

    loaded_array = np.load(temp_file)
    assert loaded_array.shape == (num_embeddings + 1, np.prod(embedding_shape))
    for i in range(num_embeddings):
        assert np.array_equal(loaded_array[i], embeddings[i].flatten())
    assert np.array_equal(loaded_array[-1], new_embedding.flatten())

    labels = npy_file.read_labels()
    assert labels == [f"img_{i}" for i in range(num_embeddings + 1)]

    # Test label file
    label_file = temp_file.with_suffix('.txt')
    assert label_file.exists()
    with label_file.open('r') as f:
        assert f.read() == ''.join(f"img_{i}\n" for i in range(num_embeddings + 1))


@pytest.mark.parametrize(
    "file_class, file_suffix",
    [
        (JSONEmbeddingsFile, ".json"),
        (NumpyEmbeddingsFile, ".npy"),
    ],
)
def test_empty_embeddings_file(temp_file, file_class, file_suffix):
    embeddings_file = file_class(temp_file.with_suffix(file_suffix))

    assert embeddings_file.load_embeddings().shape == (0, 0)
    assert embeddings_file.read_labels() == []


@pytest.mark.parametrize(
    "file_class, file_suffix",
    [
        (JSONEmbeddingsFile, ".json"),
        (NumpyEmbeddingsFile, ".npy"),
    ],
)
def test_update_existing_embedding(temp_file, file_class, file_suffix):
    embeddings_file = file_class(temp_file.with_suffix(file_suffix))

    embedding1 = np.array([1.0, 2.0, 3.0])
    embedding2 = np.array([4.0, 5.0, 6.0])
    new_embedding = np.array([7.0, 8.0, 9.0])

    embeddings_file.update(embedding1, 1)
    embeddings_file.update(embedding2, 2)
    embeddings_file.update(new_embedding, 1)  # Try to update existing

    loaded_embeddings = embeddings_file.load_embeddings()
    assert np.array_equal(loaded_embeddings, np.array([embedding1, embedding2]))


@pytest.mark.parametrize(
    "file_suffix, expected_class",
    [
        (".json", JSONEmbeddingsFile),
        (".npy", NumpyEmbeddingsFile),
    ],
)
def test_open_embeddings_file(temp_file, file_suffix, expected_class):
    temp_file = temp_file.with_suffix(file_suffix)
    embeddings_file = open_embeddings_file(temp_file)
    assert isinstance(embeddings_file, expected_class)


@pytest.mark.parametrize(
    "metric, expected_name",
    [
        (DistanceMetric.euclidean_distance, 'euclidean_distance'),
        (DistanceMetric.squared_euclidean_distance, 'squared_euclidean_distance'),
        (DistanceMetric.cosine_distance, 'cosine_distance'),
        (DistanceMetric.cosine_similarity, 'cosine_similarity'),
    ],
)
def test_enum_values(metric, expected_name):
    assert metric.name == expected_name


@pytest.mark.parametrize(
    "metric, expected_value",
    [
        (DistanceMetric.euclidean_distance, 1),
        (DistanceMetric.squared_euclidean_distance, 2),
        (DistanceMetric.cosine_distance, 3),
        (DistanceMetric.cosine_similarity, 4),
    ],
)
def test_enum_auto_values(metric, expected_value):
    assert metric.value == expected_value


def test_euclidean_distance_raises_value_error_for_1d_arrays():
    embedding_1 = np.random.rand(128)  # 1-dimensional array
    embedding_2 = np.random.rand(128)  # 1-dimensional array

    with pytest.raises(ValueError, match="Input arrays must be 2-dimensional"):
        euclidean_distance(embedding_1, embedding_2)


def test_euclidean_distance_raises_value_error_for_unequal_columns():
    embedding_1 = np.random.rand(1, 128)  # 2-dimensional array with 128 columns
    embedding_2 = np.random.rand(1, 256)  # 2-dimensional array with 256 columns

    with pytest.raises(ValueError, match="Input arrays must have the same number of columns"):
        euclidean_distance(embedding_1, embedding_2)

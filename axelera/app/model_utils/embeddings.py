# Copyright Axelera AI, 2024
# General functions for recognition, classification, clustering, etc.
# which use embedding features

import abc
import enum
import json
import math
from pathlib import Path
from typing import List, Optional, Union

import numpy as np


class DistanceMetric(enum.Enum):
    euclidean_distance = enum.auto()
    squared_euclidean_distance = enum.auto()
    cosine_distance = enum.auto()
    cosine_similarity = enum.auto()


def _check_embeddings_shape(embeddings1: np.ndarray, embeddings2: np.ndarray):
    if embeddings1.ndim != 2 or embeddings2.ndim != 2:
        raise ValueError("Input arrays must be 2-dimensional")
    if embeddings1.shape[1] != embeddings2.shape[1]:
        raise ValueError("Input arrays must have the same number of columns")


def euclidean_distance(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """
    This is the standard Euclidean distance.
    """
    _check_embeddings_shape(embeddings1, embeddings2)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sqrt(np.sum(np.square(diff), 1))
    return dist


def squared_euclidean_distance(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    """
    Here we use the sum of squares of the differences between the embeddings, which is the squared Euclidean distance.
    This avoids the computational cost of taking the square root, making it faster.
    """
    _check_embeddings_shape(embeddings1, embeddings2)

    diff = np.subtract(embeddings1, embeddings2)
    dist = np.sum(np.square(diff), 1)
    return dist


def cosine_similarity(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    _check_embeddings_shape(embeddings1, embeddings2)

    dot = np.sum(np.multiply(embeddings1, embeddings2), axis=1).astype(float)
    norm = np.linalg.norm(embeddings1, axis=1) * np.linalg.norm(embeddings2, axis=1)
    similarity = np.divide(dot, norm, out=np.zeros_like(dot), where=norm != 0)
    return similarity


def cosine_distance(embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
    _check_embeddings_shape(embeddings1, embeddings2)

    similarity = cosine_similarity(embeddings1, embeddings2)
    dist = np.arccos(similarity) / math.pi
    return dist


class EmbeddingsFile(abc.ABC):
    def __init__(self, path: Union[str, Path]):
        self.path = Path(path).resolve()
        self._dirty = False

    @property
    def dirty(self) -> bool:
        return self._dirty

    @abc.abstractmethod
    def update(self, embedding: np.ndarray, image_id: str):
        pass

    @abc.abstractmethod
    def commit(self):
        pass

    @abc.abstractmethod
    def read_labels(self, labels: Optional[List[str]] = None) -> List[str]:
        pass

    @abc.abstractmethod
    def load_embeddings(self) -> np.ndarray:
        pass


class JSONEmbeddingsFile(EmbeddingsFile):
    def __init__(self, path: Union[str, Path]):
        super().__init__(path)
        if not self.path.exists():
            self.path.write_text('{}')
        with self.path.open('r') as f:
            try:
                self.embedding_dict = json.load(f)
            except json.JSONDecodeError:
                self.embedding_dict = {}

    def update(self, embedding: np.ndarray, image_id: str):
        if image_id not in self.embedding_dict:
            self.embedding_dict[image_id] = embedding.flatten().tolist()
            self._dirty = True

    def commit(self):
        if self._dirty:
            with self.path.open('w') as f:
                json.dump(self.embedding_dict, f)
            self._dirty = False

    def read_labels(self, labels: Optional[List[str]] = None) -> List[str]:
        return list(map(str, self.embedding_dict.keys()))

    def load_embeddings(self) -> np.ndarray:
        if not self.embedding_dict:
            return np.empty((0, 0))
        embeddings = [np.array(v).flatten() for v in self.embedding_dict.values()]
        return np.vstack(embeddings)


class NumpyEmbeddingsFile(EmbeddingsFile):
    def __init__(self, path: Union[str, Path]):
        super().__init__(path)
        self.label_file = self.path.with_suffix('.txt')
        if self.path.exists():
            self.embedding_array = np.load(self.path)
            self.labels = self._read_labels()
        else:
            self.embedding_array = np.empty((0, 0))
            self.labels = []

    def _read_labels(self):
        if self.label_file.exists():
            with self.label_file.open('r') as f:
                return [line.strip() for line in f]
        # no label file, generate by index
        return [str(i) for i in range(len(self.embedding_array))]

    def update(self, embedding: np.ndarray, image_id: str):
        embedding = embedding.reshape(1, -1)
        if image_id not in self.labels:
            if self.embedding_array.size == 0:
                self.embedding_array = embedding
            else:
                self.embedding_array = np.vstack((self.embedding_array, embedding))
            self.labels.append(image_id)
            with self.label_file.open('a') as f:
                f.write(f"{image_id}\n")
            self._dirty = True

    def commit(self):
        if self._dirty:
            np.save(self.path, self.embedding_array)
            with self.label_file.open('w') as f:
                f.write('\n'.join(self.labels) + '\n')
            self._dirty = False

    def read_labels(self, labels: Optional[List[str]] = None) -> List[str]:
        return labels if labels is not None else self.labels

    def load_embeddings(self) -> np.ndarray:
        return self.embedding_array


def open_embeddings_file(path: Union[str, Path]) -> EmbeddingsFile:
    path = Path(path)
    if path.suffix == '.json':
        return JSONEmbeddingsFile(path)
    elif path.suffix == '.npy':
        return NumpyEmbeddingsFile(path)
    else:
        raise ValueError(f"Unsupported file format: {path.suffix}")

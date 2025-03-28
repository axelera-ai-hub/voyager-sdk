# Copyright Axelera AI, 2024
import pathlib
import re
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

from axelera import types
from axelera.app import meta, network
from axelera.app.model_utils import embeddings
from axelera.app.operators import EvalMode, Recognition, TopK


class MockAxMeta:
    def __init__(self):
        self.instances = {}
        self.image_id = "test_image"

    def get_instance(self, name, cls, **kwargs):
        if name not in self.instances:
            self.instances[name] = cls(**kwargs)
        return self.instances[name]

    def add_instance(self, name, instance, master_meta_name=''):
        self.instances[name] = instance


FACE_RECOGNITION_MODEL_INFO = types.ModelInfo('FaceRecognition', 'Classification', [3, 160, 160])


def make_model_infos(the_model_info):
    model_infos = network.ModelInfos()
    model_infos.add_model(the_model_info, pathlib.Path('/path'))
    return model_infos


@pytest.fixture
def temp_embeddings_file(request):
    default_values = request.param
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as tmp:
        if default_values:
            tmp.write(
                '{"person1": [0.4, 0.5, 0.6], "person2": [0.1, 0.2, 0.3], "person3": [0.7, 0.8, 0.9]}'
            )
        else:
            tmp.write('{}')
    yield pathlib.Path(tmp.name)
    pathlib.Path(tmp.name).unlink()


def test_topk_operator_as_master_task_exec_torch():
    torch = pytest.importorskip("torch")
    topk = TopK(
        k=3,
        largest=True,
        sorted=True,
    )
    topk._model_name = 'FaceRecognition'
    topk._where = ''
    topk.task_name = "task_name"
    topk.labels = ['person1', 'person2', 'person3']
    topk.num_classes = 3

    image = MagicMock()
    image.size = (224, 224)
    predict = torch.tensor([[0.1, 0.3, 0.6]])
    axmeta = meta.AxMeta('id')

    image, predict, axmeta = topk.exec_torch(image, predict, axmeta)
    assert "task_name" in axmeta
    classification_meta = axmeta["task_name"]
    assert classification_meta._class_ids == [[2, 1, 0]]
    np.testing.assert_allclose(classification_meta._scores, np.array([[0.6, 0.3, 0.1]]))


def test_topk_operator_as_secondary_task_exec_torch():
    torch = pytest.importorskip("torch")
    topk = TopK(
        k=3,
        largest=True,
        sorted=True,
    )
    topk._model_name = 'FaceRecognition'
    topk._where = 'Detection'
    topk.task_name = "task_name"
    topk.labels = ['person1', 'person2', 'person3']  # assigned in configure_model_and_context_info
    topk.num_classes = 3  # assigned in configure_model_and_context_info

    image = MagicMock()
    image.size = (224, 224)
    predict = torch.tensor([[0.1, 0.3, 0.6]])

    # set master meta
    axmeta = meta.AxMeta('id')
    detection_meta = meta.ObjectDetectionMeta(
        boxes=np.array([[0, 10, 10, 20], [10, 20, 20, 30]]),
        scores=np.array([0.8, 0.7]),
        class_ids=np.array([0, 1]),
    )
    axmeta.add_instance('Detection', detection_meta)

    image, predict, axmeta = topk.exec_torch(image, predict, axmeta)
    assert "task_name" not in axmeta
    classification_meta0 = axmeta["Detection"].get_secondary_meta("task_name", 0)
    assert classification_meta0._class_ids == [[2, 1, 0]]
    np.testing.assert_allclose(classification_meta0._scores, np.array([[0.6, 0.3, 0.1]]))

    # add 2nd classification
    predict = torch.tensor([[0.4, 0.5, 0.6]])
    image, predict, axmeta = topk.exec_torch(image, predict, axmeta)
    classification_meta1 = axmeta["Detection"].get_secondary_meta("task_name", 1)
    assert classification_meta1._class_ids == [[2, 1, 0]]
    np.testing.assert_allclose(classification_meta1._scores, np.array([[0.6, 0.5, 0.4]]))


@pytest.mark.parametrize('temp_embeddings_file', [False], indirect=True)
def test_recognition_operator_exec_torch_no_embeddings(temp_embeddings_file):
    torch = pytest.importorskip("torch")
    recognition = Recognition(
        embeddings_file=temp_embeddings_file,
        distance_threshold=0.5,
        distance_metric=embeddings.DistanceMetric.cosine_similarity,
        k=2,
    )
    with pytest.raises(
        RuntimeError, match=f"No reference embedding found, please check .*{temp_embeddings_file}"
    ):
        recognition.exec_torch(MagicMock(), MagicMock(), MockAxMeta())


@pytest.mark.parametrize('temp_embeddings_file', [True], indirect=True)
def test_recognition_operator_as_master_task_exec_torch(temp_embeddings_file):
    torch = pytest.importorskip("torch")
    recognition = Recognition(
        embeddings_file=temp_embeddings_file,
        distance_threshold=0.5,
        distance_metric=embeddings.DistanceMetric.cosine_similarity,
        k=2,
    )
    recognition._model_name = 'FaceRecognition'
    recognition._where = ''
    recognition.task_name = 'task_name'
    recognition.labels = [
        'person1',
        'person2',
        'person3',
    ]  # assigned in configure_model_and_context_info

    image = MagicMock()
    image.size = (224, 224)
    predict = torch.tensor([[0.1, 0.2, 0.3]])
    axmeta = meta.AxMeta('id')
    image, predict, axmeta = recognition.exec_torch(image, predict, axmeta)

    assert "task_name" in axmeta
    classification_meta = axmeta["task_name"]
    assert classification_meta._class_ids == [[1, 0]]
    np.testing.assert_allclose(
        np.array(classification_meta._scores), np.array([[1.0, 0.974631876199521]])
    )


@pytest.mark.parametrize('temp_embeddings_file', [True], indirect=True)
def test_recognition_operator_as_secondary_task_exec_torch(temp_embeddings_file):
    torch = pytest.importorskip("torch")
    recognition = Recognition(
        embeddings_file=temp_embeddings_file,
        distance_threshold=0.5,
        distance_metric=embeddings.DistanceMetric.cosine_similarity,
        k=2,
    )
    recognition._model_name = 'FaceRecognition'
    recognition._where = 'Detection'
    recognition.task_name = 'task_name'
    recognition.labels = [
        'person1',
        'person2',
        'person3',
    ]  # assigned in configure_model_and_context_info

    # Test exec_torch method
    image = MagicMock()
    image.size = (224, 224)
    predict = torch.tensor([[0.1, 0.2, 0.3]])
    # axmeta = MockAxMeta()
    axmeta = meta.AxMeta('id')
    detection_meta = meta.ObjectDetectionMeta(
        boxes=np.array([[0, 10, 10, 20], [10, 20, 20, 30]]),
        scores=np.array([0.8, 0.7]),
        class_ids=np.array([0, 1]),
    )
    axmeta.add_instance('Detection', detection_meta)

    image, predict, axmeta = recognition.exec_torch(image, predict, axmeta)

    assert "task_name" not in axmeta
    classification_meta0 = axmeta["Detection"].get_secondary_meta("task_name", 0)
    assert classification_meta0._class_ids == [[1, 0]]
    np.testing.assert_allclose(
        np.array(classification_meta0._scores), np.array([[1.0, 0.974631876199521]])
    )

    # add the 2nd recognition
    predict = torch.tensor([[0.4, 0.5, 0.6]])
    image, predict, axmeta = recognition.exec_torch(image, predict, axmeta)
    assert detection_meta.num_secondary_metas("task_name") == 2
    classification_meta1 = axmeta["Detection"].get_secondary_meta("task_name", 1)
    assert classification_meta0._class_ids == [[1, 0]]
    assert classification_meta1._class_ids == [[0, 2]]
    np.testing.assert_allclose(
        np.array(classification_meta1._scores), np.array([[1.0, 0.9981909319700831]])
    )


@pytest.mark.parametrize('temp_embeddings_file', [True], indirect=True)
def test_recognition_operator_exec_torch_eval_mode(temp_embeddings_file):
    torch = pytest.importorskip("torch")
    model_infos = make_model_infos(FACE_RECOGNITION_MODEL_INFO)
    recognition = Recognition(
        embeddings_file=temp_embeddings_file,
        distance_threshold=0.5,
        distance_metric=embeddings.DistanceMetric.cosine_similarity,
        k=2,
        generate_embeddings=False,
    )
    recognition._model_name = 'FaceRecognition'
    recognition._where = ''
    recognition.task_name = 'task_name'
    recognition._eval_mode = EvalMode.EVAL  # force eval mode
    recognition.labels = [
        'person1',
        'person2',
        'person3',
    ]  # assigned in configure_model_and_context_info

    image = MagicMock(size=(224, 224))
    predict = torch.tensor([[0.1, 0.2, 0.3]])
    axmeta = MockAxMeta()

    image, predict, axmeta = recognition.exec_torch(image, predict, axmeta)

    assert "task_name" in axmeta.instances
    the_meta = axmeta.instances["task_name"]
    assert isinstance(the_meta, meta.ClassificationMeta)


@pytest.mark.parametrize('temp_embeddings_file', [True], indirect=True)
def test_recognition_operator_exec_torch_pair_eval_mode(temp_embeddings_file):
    torch = pytest.importorskip("torch")
    model_infos = make_model_infos(FACE_RECOGNITION_MODEL_INFO)
    recognition = Recognition(
        embeddings_file=temp_embeddings_file,
        distance_threshold=0.5,
        distance_metric=embeddings.DistanceMetric.cosine_similarity,
        k=2,
        generate_embeddings=True,
    )
    recognition._eval_mode = EvalMode.PAIR_EVAL  # force eval mode
    recognition._is_pair_validation = True  # force pair validation
    recognition._model_name = 'FaceRecognition'
    recognition.task_name = 'task_name'
    recognition._where = ''

    image = MagicMock(size=(224, 224))
    predict = torch.tensor([[0.1, 0.2, 0.3]])
    axmeta = MockAxMeta()

    image, predict, axmeta = recognition.exec_torch(image, predict, axmeta)

    assert "task_name" in axmeta.instances
    the_meta = axmeta.instances["task_name"]
    assert isinstance(the_meta, meta.PairValidationMeta)

    # Check if embedding was generated
    embeddings_file = embeddings.JSONEmbeddingsFile(temp_embeddings_file)
    loaded_embeddings = embeddings_file.load_embeddings()
    assert loaded_embeddings.shape == (3, 3)
    assert np.allclose(loaded_embeddings[0], [0.4, 0.5, 0.6])
    assert np.allclose(loaded_embeddings[1], [0.1, 0.2, 0.3])
    assert np.allclose(loaded_embeddings[2], [0.7, 0.8, 0.9])


def test_recognition_invalid_distance_metric():
    with pytest.raises(
        ValueError,
        match=re.escape(
            'Invalid value for distance_metric: invalid_metric (expected one of euclidean_distance, squared_euclidean_distance, cosine_distance, cosine_similarity)'
        ),
    ):
        recognition = Recognition(
            embeddings_file="path/to/embeddings",
            distance_threshold=0.5,
            distance_metric="invalid_metric",  # Invalid distance metric
            k=2,
        )


def test_recognition_no_reference_embedding():
    with pytest.raises(ValueError, match='Unsupported file format:'):
        recognition = Recognition(
            embeddings_file="path/to/embeddings",
            distance_threshold=0.5,
            distance_metric=embeddings.DistanceMetric.euclidean_distance,
            k=2,
        )

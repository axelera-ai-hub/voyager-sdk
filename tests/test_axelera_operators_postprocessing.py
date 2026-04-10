# Copyright Axelera AI, 2024

from pathlib import Path
import re
import tempfile
from unittest.mock import MagicMock

from axelera.types import ModelInfo
import numpy as np
import pytest

from axelera.app import meta, network
from axelera.app.model_utils import embeddings
from axelera.app.operators.base import EvalMode
from axelera.app.operators.postprocessing import Recognition, TopK, _reorder_embeddings_by_names


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
        # Set up the secondary meta relationship if needed
        if master_meta_name and isinstance(instance, meta.AxTaskMeta):
            master = self.instances.get(master_meta_name)
            if master and isinstance(master, meta.AxTaskMeta):
                # Add the task name as a secondary meta to the master
                if not hasattr(master, 'secondary_frame_indices'):
                    setattr(master, 'secondary_frame_indices', {})
                if name not in master.secondary_frame_indices:
                    master.secondary_frame_indices[name] = []
                master.secondary_frame_indices[name].append(
                    len(master.secondary_frame_indices.get(name, []))
                )

                # Make sure _secondary_metas is initialized
                if not hasattr(master, '_secondary_metas'):
                    setattr(master, '_secondary_metas', {})
                if name not in master._secondary_metas:
                    master._secondary_metas[name] = []
                master._secondary_metas[name].append(instance)


FACE_RECOGNITION_MODEL_INFO = ModelInfo('FaceRecognition', 'Classification', [3, 160, 160])


def make_model_infos(the_model_info):
    model_infos = network.ModelInfos()
    model_infos.add_model(the_model_info, Path('/path'))
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
    yield Path(tmp.name)
    Path(tmp.name).unlink()


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
        update_embeddings=False,  # Force RuntimeError when no embeddings exist
    )
    recognition.labels = [
        'person1',
        'person2',
        'person3',
    ]  # assigned in configure_model_and_context_info
    with pytest.raises(
        RuntimeError, match=f"No reference embedding found, please check .*{temp_embeddings_file}"
    ):
        image = MagicMock()
        predict = torch.tensor([[0.1, 0.2, 0.3]])
        axmeta = MockAxMeta()
        recognition.exec_torch(image, predict, axmeta)


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

    # Test the new MetaObject interface for secondary metadata access
    detection_objects = detection_meta.objects
    assert len(detection_objects) == 2

    # Test first detection's classification
    classifier_meta0 = detection_objects[0].get_secondary_meta("task_name")
    assert classifier_meta0 is classification_meta0

    # Test task names
    assert detection_objects[0].secondary_task_names == ["task_name"]
    assert detection_objects[1].secondary_task_names == ["task_name"]


@pytest.mark.parametrize('temp_embeddings_file', [True], indirect=True)
def test_recognition_operator_exec_torch_eval_mode(temp_embeddings_file):
    torch = pytest.importorskip("torch")
    model_infos = make_model_infos(FACE_RECOGNITION_MODEL_INFO)
    recognition = Recognition(
        embeddings_file=temp_embeddings_file,
        distance_threshold=0.5,
        distance_metric=embeddings.DistanceMetric.cosine_similarity,
        k=2,
        update_embeddings=False,
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
        update_embeddings=True,
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
    """Test handling when embeddings file path is just a string"""
    recognition = Recognition(
        embeddings_file="nonexistent_file.json",
        distance_threshold=0.5,
        distance_metric=embeddings.DistanceMetric.cosine_similarity,
        k=2,
    )
    # Test that the pipeline_stopped method doesn't crash when embeddings_file is a string
    recognition.pipeline_stopped()  # Should not raise AttributeError


class TestExtractPersonNameFromPath:
    """Comprehensive tests for _extract_person_name_from_path function."""

    def test_extract_from_folder_structure(self):
        """Test extraction from AJ_Cook/AJ_Cook_0001.jpg format"""
        torch = pytest.importorskip("torch")
        recognition = Recognition(
            embeddings_file="dummy.json",
            distance_threshold=0.5,
            distance_metric=embeddings.DistanceMetric.cosine_similarity,
            k=2,
        )

        # Test case where folder name matches part of filename
        result = recognition._extract_person_name_from_path("AJ_Cook/AJ_Cook_0001.jpg")
        assert result == "AJ_Cook"

        result = recognition._extract_person_name_from_path("John_Doe/John_Doe_portrait.jpg")
        assert result == "John_Doe"

        # Test case where folder name appears in filename
        result = recognition._extract_person_name_from_path(
            "celebrities/Brad_Pitt/Brad_Pitt_001.jpg"
        )
        assert result == "Brad_Pitt"

    def test_extract_from_filename_only(self):
        """Test extraction when folder name doesn't match filename"""
        torch = pytest.importorskip("torch")
        recognition = Recognition(
            embeddings_file="dummy.json",
            distance_threshold=0.5,
            distance_metric=embeddings.DistanceMetric.cosine_similarity,
            k=2,
        )

        # When folder name doesn't appear in filename, use filename stem
        result = recognition._extract_person_name_from_path("photos/celebrity_photo.jpg")
        assert result == "celebrity_photo"

        result = recognition._extract_person_name_from_path("images/portrait.png")
        assert result == "portrait"

        # Test with nested directories
        result = recognition._extract_person_name_from_path("data/faces/unknown_person.jpg")
        assert result == "unknown_person"

    def test_extract_with_empty_or_none_input(self):
        """Test edge cases with empty or None input"""
        torch = pytest.importorskip("torch")
        recognition = Recognition(
            embeddings_file="dummy.json",
            distance_threshold=0.5,
            distance_metric=embeddings.DistanceMetric.cosine_similarity,
            k=2,
        )

        # Empty string
        result = recognition._extract_person_name_from_path("")
        assert result is None

        # None input
        result = recognition._extract_person_name_from_path(None)
        assert result is None

    def test_extract_various_file_extensions(self):
        """Test with various file extensions"""
        torch = pytest.importorskip("torch")
        recognition = Recognition(
            embeddings_file="dummy.json",
            distance_threshold=0.5,
            distance_metric=embeddings.DistanceMetric.cosine_similarity,
            k=2,
        )

        test_cases = [
            ("person/person.jpg", "person"),
            ("person/person.png", "person"),
            ("person/person.jpeg", "person"),
            ("person/person.bmp", "person"),
            ("person/person.tiff", "person"),
        ]

        for path, expected in test_cases:
            result = recognition._extract_person_name_from_path(path)
            assert result == expected, f"Failed for {path}"

    def test_extract_with_underscores_and_special_chars(self):
        """Test with names containing underscores and special characters"""
        torch = pytest.importorskip("torch")
        recognition = Recognition(
            embeddings_file="dummy.json",
            distance_threshold=0.5,
            distance_metric=embeddings.DistanceMetric.cosine_similarity,
            k=2,
        )

        # Test with underscores (common in dataset naming)
        result = recognition._extract_person_name_from_path(
            "Mary_Jane_Watson/Mary_Jane_Watson_01.jpg"
        )
        assert result == "Mary_Jane_Watson"

        # Test with numbers in names
        result = recognition._extract_person_name_from_path("Person123/Person123_photo.jpg")
        assert result == "Person123"


class TestReorderEmbeddingsByNames:
    """Comprehensive tests for _reorder_embeddings_by_names function."""

    def test_no_reordering_needed(self):
        """Test when labels already match names file order"""
        labels = ["Alice", "Bob", "Charlie"]
        embeddings = np.array([[1, 2], [3, 4], [5, 6]])

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Alice\nBob\nCharlie\n")
            names_file = Path(f.name)

        try:
            result_labels, result_embeddings, was_reordered = _reorder_embeddings_by_names(
                labels, embeddings, names_file
            )

            assert result_labels == labels
            np.testing.assert_array_equal(result_embeddings, embeddings)
            assert was_reordered is False
        finally:
            names_file.unlink()

    def test_reordering_required(self):
        """Test reordering when names file has different order"""
        labels = ["Alice", "Bob", "Charlie"]
        embeddings = np.array([[1, 2], [3, 4], [5, 6]])

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Charlie\nAlice\nBob\n")
            names_file = Path(f.name)

        try:
            result_labels, result_embeddings, was_reordered = _reorder_embeddings_by_names(
                labels, embeddings, names_file
            )

            expected_labels = ["Charlie", "Alice", "Bob"]
            expected_embeddings = np.array([[5, 6], [1, 2], [3, 4]])  # Reordered

            assert result_labels == expected_labels
            np.testing.assert_array_equal(result_embeddings, expected_embeddings)
            assert was_reordered is True
        finally:
            names_file.unlink()

    def test_subset_reordering(self):
        """Test reordering when names file contains more names than current labels"""
        labels = ["Alice", "Charlie"]  # Missing Bob
        embeddings = np.array([[1, 2], [5, 6]])

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Alice\nBob\nCharlie\nDave\n")  # More names than we have
            names_file = Path(f.name)

        try:
            result_labels, result_embeddings, was_reordered = _reorder_embeddings_by_names(
                labels, embeddings, names_file
            )

            expected_labels = ["Alice", "Bob", "Charlie", "Dave"]
            expected_embeddings = np.array(
                [
                    [1, 2],  # Alice
                    [0, 0],  # Bob (missing, filled with zeros)
                    [5, 6],  # Charlie
                    [0, 0],  # Dave (missing, filled with zeros)
                ]
            )

            assert result_labels == expected_labels
            np.testing.assert_array_equal(result_embeddings, expected_embeddings)
            assert was_reordered is True
        finally:
            names_file.unlink()

    def test_lfw_names_format(self):
        """Test special handling for LFW names file format"""
        labels = ["Abel_Pacheco", "Akhmed_Zakayev"]
        embeddings = np.array([[1, 2, 3], [4, 5, 6]])

        # Create LFW-style names file directly with the correct name
        with tempfile.NamedTemporaryFile(mode='w', suffix='-lfw-names.txt', delete=False) as f:
            f.write("Akhmed_Zakayev\t31\n")  # LFW format: name\tcount
            f.write("Abel_Pacheco\t15\n")
            names_file = Path(f.name)

        try:
            result_labels, result_embeddings, was_reordered = _reorder_embeddings_by_names(
                labels, embeddings, names_file
            )

            expected_labels = ["Akhmed_Zakayev", "Abel_Pacheco"]  # Reordered per LFW file
            expected_embeddings = np.array([[4, 5, 6], [1, 2, 3]])  # Reordered

            assert result_labels == expected_labels
            np.testing.assert_array_equal(result_embeddings, expected_embeddings)
            assert was_reordered is True
        finally:
            if names_file.exists():
                names_file.unlink()

    def test_unexpected_names_assertion(self):
        """Test that assertion is raised when labels contain unexpected names"""
        labels = ["Alice", "Bob", "UnknownPerson"]  # UnknownPerson not in names file
        embeddings = np.array([[1, 2], [3, 4], [5, 6]])

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Alice\nBob\nCharlie\n")  # No UnknownPerson
            names_file = Path(f.name)

        try:
            with pytest.raises(AssertionError, match="labels: .* must be a subset of names"):
                _reorder_embeddings_by_names(labels, embeddings, names_file)
        finally:
            names_file.unlink()

    def test_zero_filling_for_missing_embeddings(self):
        """Test that missing embeddings are filled with zeros correctly"""
        labels = ["Alice"]  # Only one person
        embeddings = np.array([[1, 2, 3]])

        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("Alice\nBob\nCharlie\n")
            names_file = Path(f.name)

        try:
            result_labels, result_embeddings, was_reordered = _reorder_embeddings_by_names(
                labels, embeddings, names_file
            )

            # Check that zeros are properly filled for missing persons
            assert result_embeddings.shape == (3, 3)  # 3 persons, 3 dimensions
            np.testing.assert_array_equal(result_embeddings[0], [1, 2, 3])  # Alice
            np.testing.assert_array_equal(result_embeddings[1], [0, 0, 0])  # Bob (missing)
            np.testing.assert_array_equal(result_embeddings[2], [0, 0, 0])  # Charlie (missing)

            assert was_reordered is True
        finally:
            names_file.unlink()


class TestRecognitionWithNamesFile:
    """Test Recognition operator with names_file functionality"""

    @pytest.mark.parametrize('temp_embeddings_file', [True], indirect=True)
    def test_recognition_with_names_file_reordering(self, temp_embeddings_file):
        """Test that Recognition correctly uses names_file for reordering"""
        torch = pytest.importorskip("torch")

        # Create a names file with different ordering
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("person3\nperson1\nperson2\n")  # Different order from embeddings file
            names_file_path = f.name

        try:
            recognition = Recognition(
                embeddings_file=temp_embeddings_file,
                distance_threshold=0.5,
                distance_metric=embeddings.DistanceMetric.cosine_similarity,
                k=2,
                names_file=names_file_path,  # Use names file for reordering
            )
            recognition._model_name = 'FaceRecognition'
            recognition._where = ''
            recognition.task_name = 'task_name'

            # Load embeddings - this should trigger reordering
            recognition._load_reference_embeddings()

            # Check that labels are reordered according to names file
            expected_order = ["person3", "person1", "person2"]
            assert recognition.labels == expected_order

            # Check that embeddings are reordered accordingly
            assert recognition.ref_embedding.shape == (3, 3)
            # person3 should be first now (originally was [0.7, 0.8, 0.9])
            np.testing.assert_allclose(recognition.ref_embedding[0], [0.7, 0.8, 0.9])
            # person1 should be second now (originally was [0.4, 0.5, 0.6])
            np.testing.assert_allclose(recognition.ref_embedding[1], [0.4, 0.5, 0.6])
            # person2 should be third now (originally was [0.1, 0.2, 0.3])
            np.testing.assert_allclose(recognition.ref_embedding[2], [0.1, 0.2, 0.3])

        finally:
            Path(names_file_path).unlink()

    @pytest.mark.parametrize('temp_embeddings_file', [True], indirect=True)
    def test_recognition_without_names_file(self, temp_embeddings_file):
        """Test that Recognition works normally without names_file"""
        torch = pytest.importorskip("torch")

        recognition = Recognition(
            embeddings_file=temp_embeddings_file,
            distance_threshold=0.5,
            distance_metric=embeddings.DistanceMetric.cosine_similarity,
            k=2,
            names_file=None,  # No names file
        )
        recognition._model_name = 'FaceRecognition'
        recognition._where = ''
        recognition.task_name = 'task_name'

        # Load embeddings - should maintain original order
        recognition._load_reference_embeddings()

        # Check that labels maintain original order from embeddings file
        expected_order = ["person1", "person2", "person3"]
        assert recognition.labels == expected_order

        # Check that embeddings maintain original order
        assert recognition.ref_embedding.shape == (3, 3)
        np.testing.assert_allclose(recognition.ref_embedding[0], [0.4, 0.5, 0.6])  # person1
        np.testing.assert_allclose(recognition.ref_embedding[1], [0.1, 0.2, 0.3])  # person2
        np.testing.assert_allclose(recognition.ref_embedding[2], [0.7, 0.8, 0.9])  # person3


class TestRecognitionEmbeddingUpdates:
    """Test Recognition operator embedding update functionality"""

    @pytest.mark.parametrize('temp_embeddings_file', [True], indirect=True)
    def test_new_person_detection_and_addition(self, temp_embeddings_file):
        """Test adding a completely new person to the embeddings"""
        torch = pytest.importorskip("torch")

        recognition = Recognition(
            embeddings_file=temp_embeddings_file,
            distance_threshold=0.5,
            distance_metric=embeddings.DistanceMetric.cosine_similarity,
            k=2,
            update_embeddings=True,
        )
        recognition._model_name = 'FaceRecognition'
        recognition._where = ''
        recognition.task_name = 'task_name'

        # Load initial embeddings
        recognition._load_reference_embeddings()
        initial_num_classes = recognition.num_classes
        initial_labels = recognition.labels.copy()

        # Test with new person embedding
        image = MagicMock()
        predict = torch.tensor([[0.9, 0.8, 0.7]])  # New person embedding
        axmeta = MockAxMeta()
        axmeta.image_id = "new_person/new_person_001.jpg"  # New person name

        result = recognition.exec_torch(image, predict, axmeta)

        # Check that the new person was added
        assert recognition.num_classes == initial_num_classes + 1
        assert "new_person" in recognition.labels
        assert "new_person" not in initial_labels

        # Check that embeddings were updated
        assert recognition.ref_embedding.shape[0] == initial_num_classes + 1

        # Verify the new embedding was added correctly
        new_person_idx = recognition.labels.index("new_person")
        expected_normalized = np.array([0.9, 0.8, 0.7])
        expected_normalized = expected_normalized / np.linalg.norm(expected_normalized)
        np.testing.assert_allclose(
            recognition.ref_embedding[new_person_idx], expected_normalized, rtol=1e-5
        )

    @pytest.mark.parametrize('temp_embeddings_file', [False], indirect=True)
    def test_first_embedding_initialization(self, temp_embeddings_file):
        """Test initialization when no embeddings exist yet"""
        torch = pytest.importorskip("torch")

        recognition = Recognition(
            embeddings_file=temp_embeddings_file,
            distance_threshold=0.5,
            distance_metric=embeddings.DistanceMetric.cosine_similarity,
            k=2,
            update_embeddings=True,
        )
        recognition._model_name = 'FaceRecognition'
        recognition._where = ''
        recognition.task_name = 'task_name'

        # Test with first embedding
        image = MagicMock()
        predict = torch.tensor([[0.6, 0.8, 1.0]])  # First embedding
        axmeta = MockAxMeta()
        axmeta.image_id = "first_person/first_person_001.jpg"

        result = recognition.exec_torch(image, predict, axmeta)

        # Check that embeddings were initialized
        assert recognition.num_classes == 1
        assert recognition.labels == ["first_person"]
        assert recognition.ref_embedding.shape == (1, 3)

        # Check that the method returns early after adding first embedding
        assert result is not None


class TestRecognitionTensorHandling:
    """Test Recognition operator tensor handling edge cases"""

    @pytest.mark.parametrize('temp_embeddings_file', [True], indirect=True)
    def test_1d_tensor_normalization(self, temp_embeddings_file):
        """Test that 1D tensors are correctly reshaped before normalization"""
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

        # Test with 1D tensor (the original bug case)
        image = MagicMock()
        predict = torch.tensor([0.1, 0.2, 0.3])  # 1D tensor
        axmeta = meta.AxMeta('id')

        # This should not raise an AxisError anymore
        result = recognition.exec_torch(image, predict, axmeta)

        assert result is not None
        assert "task_name" in axmeta

    @pytest.mark.parametrize('temp_embeddings_file', [True], indirect=True)
    def test_2d_tensor_normalization(self, temp_embeddings_file):
        """Test that 2D tensors work correctly (normal case)"""
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

        # Test with 2D tensor (normal case)
        image = MagicMock()
        predict = torch.tensor([[0.1, 0.2, 0.3]])  # 2D tensor
        axmeta = meta.AxMeta('id')

        result = recognition.exec_torch(image, predict, axmeta)

        assert result is not None
        assert "task_name" in axmeta


class TestFaceAlign:
    """Test the FaceAlign preprocessing operator with configurable templates."""

    @pytest.fixture(autouse=True)
    def setup(self):
        """Set up required imports for all tests"""
        pytest.importorskip("cv2")  # Skip tests if cv2 is not available
        # Import here to avoid issues with imports not being resolved
        from axelera.app.operators.custom_preprocessing import FaceAlign

        self.FaceAlign = FaceAlign

    def test_face_align_with_default_template(self):
        """Test FaceAlign with default 5-point template"""
        import numpy as np

        # Create mock face alignment operator with explicit template values
        face_align = self.FaceAlign(
            keypoints_key="face_keypoints",
            width=112,
            height=112,
            template_keypoints_x=[
                30.2946 / 96,
                65.5318 / 96,
                48.0252 / 96,
                33.5493 / 96,
                62.7299 / 96,
            ],
            template_keypoints_y=[
                51.6963 / 96,
                51.5014 / 96,
                71.7366 / 96,
                92.3655 / 96,
                92.2041 / 96,
            ],
        )

        # Check the template values
        expected_x = [30.2946 / 96, 65.5318 / 96, 48.0252 / 96, 33.5493 / 96, 62.7299 / 96]
        expected_y = [51.6963 / 96, 51.5014 / 96, 71.7366 / 96, 92.3655 / 96, 92.2041 / 96]
        assert face_align.template_keypoints_x == expected_x
        assert face_align.template_keypoints_y == expected_y
        assert face_align.use_self_normalizing is False

        # The 5-point template should match the expected default
        template = np.array(
            list(zip(face_align.template_keypoints_x, face_align.template_keypoints_y))
        )
        # Expected template from OpenCV values
        expected_template = np.array(
            [
                [30.2946 / 96, 51.6963 / 96],  # Left eye
                [65.5318 / 96, 51.5014 / 96],  # Right eye
                [48.0252 / 96, 71.7366 / 96],  # Nose
                [33.5493 / 96, 92.3655 / 96],  # Left mouth corner
                [62.7299 / 96, 92.2041 / 96],  # Right mouth corner
            ]
        )
        np.testing.assert_array_almost_equal(template, expected_template)

    def test_face_align_with_custom_template(self):
        """Test FaceAlign with custom 5-point template"""
        import numpy as np

        # Custom template coordinates
        custom_x = [0.25, 0.75, 0.5, 0.3, 0.7]
        custom_y = [0.25, 0.25, 0.45, 0.75, 0.75]

        # Create face alignment operator with custom template
        face_align = self.FaceAlign(
            keypoints_key="face_keypoints",
            width=112,
            height=112,
            template_keypoints_x=custom_x,
            template_keypoints_y=custom_y,
        )

        # Check custom template values
        np.testing.assert_array_equal(face_align.template_keypoints_x, custom_x)
        np.testing.assert_array_equal(face_align.template_keypoints_y, custom_y)

    def test_face_align_with_standard_51_point(self):
        """Test FaceAlign with standard 51-point template"""
        import numpy as np

        # Create face alignment operator to test standard 51-point template
        face_align = self.FaceAlign(keypoints_key="face_keypoints", width=112, height=112)

        # Get the standard 51-point template
        x, y = face_align._get_standard_51_point_template()

        # Should have 51 points
        assert len(x) == 51
        assert len(y) == 51

    def test_face_align_self_normalizing_mode(self):
        """Test FaceAlign with self-normalizing mode enabled"""
        # Create face alignment operator with self-normalizing mode
        face_align = self.FaceAlign(
            keypoints_key="face_keypoints",
            width=112,
            height=112,
            use_self_normalizing=True,
        )

        # Check self-normalizing flag
        assert face_align.use_self_normalizing is True

    def test_face_align_invalid_template(self):
        """Test FaceAlign with invalid template (mismatched x/y lengths)"""
        # Try with mismatched template points
        with pytest.raises(ValueError, match="Number of template keypoints x and y must be equal"):
            self.FaceAlign(
                keypoints_key="face_keypoints",
                width=112,
                height=112,
                template_keypoints_x=[0.3, 0.7, 0.5, 0.3],  # 4 points
                template_keypoints_y=[0.3, 0.3, 0.5, 0.7, 0.7],  # 5 points
            )

    def test_face_align_transformation(self):
        """Test the transformation matrix calculation with mocked landmarks"""
        from unittest.mock import MagicMock, patch

        import numpy as np

        # Create face alignment operator
        face_align = self.FaceAlign(keypoints_key="face_keypoints", width=112, height=112)

        # Create mock landmarks (5 points)
        landmarks = np.array(
            [
                [30, 30],  # Left eye
                [70, 30],  # Right eye
                [50, 50],  # Nose
                [30, 70],  # Left mouth corner
                [70, 70],  # Right mouth corner
            ]
        )

        # Create target points with the same coordinates but scaled
        target = np.array(
            [
                [33.6, 33.6],  # Left eye
                [78.4, 33.6],  # Right eye
                [56, 56],  # Nose
                [33.6, 78.4],  # Left mouth corner
                [78.4, 78.4],  # Right mouth corner
            ]
        )

        # Calculate transformation matrix
        M = face_align._transformation_from_points(landmarks, target)

        # Check that M is a 3x3 matrix
        assert M.shape == (3, 3)

        # Verify the transformation by applying it to one point
        # The transformation should map landmarks[0] close to target[0]
        point = np.array([landmarks[0][0], landmarks[0][1], 1])
        transformed = np.dot(M, point)
        transformed = transformed[:2]  # Get x,y coordinates

        # Check that the transformed point is close to the target
        np.testing.assert_allclose(transformed, target[0], rtol=1e-3)

    def test_face_align_exec_torch(self):
        """Test the exec_torch method with mocked inputs"""
        from unittest.mock import MagicMock, patch

        import numpy as np

        cv2 = pytest.importorskip("cv2")

        # Create mock image
        mock_image = MagicMock()
        mock_image.size = (100, 100)
        mock_image.color_format = "RGB"
        mock_image.asarray.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        # Create mock metadata with keypoints
        mock_keypoints_meta = MagicMock()
        # Create 5-point facial landmarks: [[x1,y1], [x2,y2], ...]
        mock_keypoints_meta.keypoints = np.array(
            [
                [
                    [30, 30],  # Left eye
                    [70, 30],  # Right eye
                    [50, 50],  # Nose
                    [30, 70],  # Left mouth corner
                    [70, 70],  # Right mouth corner
                ]
            ]
        )

        # Create metadata dictionary
        mock_meta = {"face_keypoints": mock_keypoints_meta}

        # Create the face align operator with custom template
        face_align = self.FaceAlign(
            keypoints_key="face_keypoints",
            width=112,
            height=112,
            template_keypoints_x=[0.25, 0.75, 0.5, 0.3, 0.7],
            template_keypoints_y=[0.25, 0.25, 0.45, 0.75, 0.75],
        )

        # Mock cv2.warpAffine to verify it's called with the right parameters
        with patch('cv2.warpAffine') as mock_warp:
            # Mock the return value of warpAffine
            mock_warp.return_value = np.zeros((112, 112, 3), dtype=np.uint8)

            # Mock fromarray to return our mock image
            with patch('axelera.types.Image.fromarray', return_value=mock_image):
                # Call exec_torch
                result = face_align.exec_torch(mock_image, mock_meta)

                # Verify warpAffine was called once
                assert mock_warp.call_count == 1

                # The first arg should be the input image array
                assert mock_warp.call_args[0][0] is mock_image.asarray.return_value

                # The third arg should be the output dimensions (width, height)
                assert mock_warp.call_args[0][2] == (112, 112)

    def test_face_align_exec_torch_with_standard_51_point(self):
        """Test the exec_torch method with standard 51-point template"""
        from unittest.mock import MagicMock, patch

        import numpy as np

        cv2 = pytest.importorskip("cv2")

        # Create mock image
        mock_image = MagicMock()
        mock_image.size = (100, 100)
        mock_image.color_format = "RGB"
        mock_image.asarray.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        # Create mock metadata with keypoints
        mock_keypoints_meta = MagicMock()
        # Create 51-point facial landmarks to trigger standard template selection
        # We'll mock a 51-point detection by creating an array of the right shape
        mock_keypoints_meta.keypoints = np.zeros((1, 51, 2), dtype=np.float32)

        # Create metadata dictionary
        mock_meta = {"face_keypoints": mock_keypoints_meta}

        # Create the face align operator (will automatically use 51-point template for 51 keypoints)
        face_align = self.FaceAlign(keypoints_key="face_keypoints", width=112, height=112)

        # Mock the transformation_from_points method to prevent SVD errors with zero arrays
        with patch.object(face_align, '_transformation_from_points') as mock_transform:
            # Return a simple transformation matrix
            mock_transform.return_value = np.array(
                [[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float64
            )

            # Mock _get_standard_51_point_template to verify it's called
            with patch.object(
                face_align,
                '_get_standard_51_point_template',
                wraps=face_align._get_standard_51_point_template,
            ) as mock_get_template:
                # Mock cv2.warpAffine
                with patch('cv2.warpAffine') as mock_warp:
                    # Mock the return value of warpAffine
                    mock_warp.return_value = np.zeros((112, 112, 3), dtype=np.uint8)

                    # Mock fromarray to return our mock image
                    with patch('axelera.types.Image.fromarray', return_value=mock_image):
                        # Call exec_torch
                        result = face_align.exec_torch(mock_image, mock_meta)

                        # Verify _get_standard_51_point_template was called
                        mock_get_template.assert_called_once()

    def test_get_standard_51_point_template(self):
        """Test the _get_standard_51_point_template method returns the correct template coordinates"""
        import numpy as np

        # Create face align operator
        face_align = self.FaceAlign(keypoints_key="face_keypoints", width=112, height=112)

        # Get the standard 51-point template
        template_x, template_y = face_align._get_standard_51_point_template()

        # Convert to numpy arrays if they aren't already
        if not isinstance(template_x, np.ndarray):
            template_x = np.array(template_x)
        if not isinstance(template_y, np.ndarray):
            template_y = np.array(template_y)

        # Verify template length matches the 51 points we need
        assert len(template_x) == 51
        assert len(template_y) == 51

        # Verify values are in range [0, 1]
        assert np.all(template_x >= 0) and np.all(template_x <= 1)
        assert np.all(template_y >= 0) and np.all(template_y <= 1)

    def test_face_align_self_normalizing_exec(self):
        """Test self-normalizing mode in exec_torch"""
        from unittest.mock import MagicMock, patch

        import numpy as np

        cv2 = pytest.importorskip("cv2")

        # Create mock image
        mock_image = MagicMock()
        mock_image.size = (100, 100)
        mock_image.color_format = "RGB"
        mock_image.asarray.return_value = np.zeros((100, 100, 3), dtype=np.uint8)

        # Create mock metadata with keypoints
        mock_keypoints_meta = MagicMock()
        # Create 5-point facial landmarks: [[x1,y1], [x2,y2], ...]
        mock_keypoints_meta.keypoints = np.array(
            [
                [
                    [30, 30],  # Left eye
                    [70, 30],  # Right eye
                    [50, 50],  # Nose
                    [30, 70],  # Left mouth corner
                    [70, 70],  # Right mouth corner
                ]
            ]
        )

        # Create metadata dictionary
        mock_meta = {"face_keypoints": mock_keypoints_meta}

        # Create the face align operator with self-normalizing enabled
        face_align = self.FaceAlign(
            keypoints_key="face_keypoints",
            width=112,
            height=112,
            use_self_normalizing=True,  # Enable self-normalizing mode
        )

        # Add debug logging to see why self-normalizing mode isn't being triggered
        with patch('axelera.app.logging_utils.getLogger') as mock_logger:
            mock_log = MagicMock()
            mock_logger.return_value = mock_log

            # Mock cv2 functions
            with patch('cv2.getRotationMatrix2D') as mock_get_matrix:
                mock_get_matrix.return_value = np.array([[1, 0, 0], [0, 1, 0]])

                with patch('cv2.warpAffine') as mock_warp:
                    # Mock the return value of warpAffine
                    mock_warp.return_value = np.zeros((112, 112, 3), dtype=np.uint8)

                    # Mock fromarray to return our mock image
                    with patch('axelera.types.Image.fromarray', return_value=mock_image):
                        # Call exec_torch
                        result = face_align.exec_torch(mock_image, mock_meta)

                        # Log the info calls to see what path was taken
                        info_calls = [call[0][0] for call in mock_log.info.call_args_list]
                        print(f"Log info calls: {info_calls}")

                        # Verify getRotationMatrix2D was called (self-normalizing mode uses this)
                        assert mock_get_matrix.call_count == 1

                    # Verify warpAffine was called
                    assert mock_warp.call_count == 1


def test_object_access_for_cascaded_pipeline():
    """Test object access for cascaded pipelines."""
    detection_meta = meta.ObjectDetectionMeta(
        boxes=np.array([[0, 10, 10, 20], [10, 20, 20, 30]]),
        scores=np.array([0.8, 0.7]),
        class_ids=np.array([0, 1]),
    )

    # Add secondary metadata (using correct API signature)
    classification_meta1 = meta.ClassificationMeta(labels=['cat', 'dog'], num_classes=2)
    classification_meta1.add_result([0], [0.9])
    detection_meta.add_secondary_meta("classifier", classification_meta1)

    classification_meta2 = meta.ClassificationMeta(labels=['cat', 'dog'], num_classes=2)
    classification_meta2.add_result([1], [0.8])
    detection_meta.add_secondary_meta("classifier", classification_meta2)

    # Basic functionality test - ensure meta can be added and retrieved
    assert detection_meta.num_secondary_metas("classifier") == 2

    # Test getting secondary metas
    retrieved_meta1 = detection_meta.get_secondary_meta("classifier", 0)
    retrieved_meta2 = detection_meta.get_secondary_meta("classifier", 1)

    assert retrieved_meta1 is classification_meta1
    assert retrieved_meta2 is classification_meta2

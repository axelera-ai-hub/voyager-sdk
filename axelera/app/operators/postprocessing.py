# Copyright Axelera AI, 2023
# General post-processing operators
from __future__ import annotations

from pathlib import Path
from typing import Optional, Union

import numpy as np

from axelera import types
from axelera.app.operators import utils

from .. import gst_builder, logging_utils, meta
from ..model_utils import embeddings as embed_utils
from ..torch_utils import torch
from .base import AxOperator, EvalMode, builtin
from .context import PipelineContext

LOG = logging_utils.getLogger(__name__)


@builtin
class TopK(AxOperator):
    # TopK is actually a decode operator, takes tensor as input
    k: int = 1
    largest: bool = True
    sorted: bool = False
    softmax: bool = False

    def _post_init(self):
        self._tmp_labels: Optional[Path] = None
        self.sorted = bool(self.sorted)
        self.largest = bool(self.largest)

    def __del__(self):
        if self._tmp_labels is not None and self._tmp_labels.exists():
            self._tmp_labels.unlink()

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        compiled_model_dir: Path | None,
        task_graph,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, compiled_model_dir, task_graph
        )
        self.labels = model_info.labels
        self.num_classes = model_info.num_classes
        self._association = context.association or None

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        master_key = f'master_meta:{self._where};' if self._where else str()
        association_key = f'association_meta:{self._association};' if self._association else str()

        # TODO only k == 1, largest==True, and sorted==True is supported,
        if self._tmp_labels is None:
            self._tmp_labels = utils.create_tmp_labels(self.labels)

        gst.decode_muxer(
            name=f'decoder_task{self._taskn}{stream_idx}',
            lib='libdecode_classification.so',
            options=f'meta_key:{str(self.task_name)};'
            f'{master_key}'
            f'{association_key}'
            f'classlabels_file:{self._tmp_labels};'
            f'top_k:{self.k};'
            f'softmax:{int(self.softmax)}',
        )

    def exec_torch(self, image, predict, axmeta):
        model_meta = meta.ClassificationMeta(
            labels=self.labels,
            num_classes=self.num_classes,
            extra_info={"softmax": self.softmax},
        )
        import torch.nn.functional as TF

        if self.softmax:
            predict = TF.softmax(predict, dim=1)
        top_scores, top_ids = torch.topk(
            predict, k=self.k, largest=self.largest, sorted=self.sorted
        )

        model_meta.add_result(
            top_ids.cpu().detach().numpy()[0],
            top_scores.cpu().detach().numpy()[0],
        )
        axmeta.add_instance(self.task_name, model_meta, self._where)
        return image, predict, axmeta


def _check_single_reducemean_node(file_path):
    """
    Checks if an ONNX model file contains exactly one intermediate node
    and if that node is a ReduceMean operator. Prints the result.
    """
    import sys

    import onnx

    try:
        model = onnx.load(file_path)
        nodes = model.graph.node
        num_nodes = len(nodes)

        if num_nodes == 1 and nodes[0].op_type == 'ReduceMean':
            LOG.trace(
                f"✅ Verification PASSED for '{file_path}':\n   Exactly one intermediate node found, and it is 'ReduceMean'."
            )
            return True
        elif num_nodes == 1:
            LOG.trace(
                f"❌ Verification FAILED for '{file_path}':\n   Exactly one intermediate node found, but it is '{nodes[0].op_type}', not 'ReduceMean'."
            )
        else:
            LOG.trace(
                f"❌ Verification FAILED for '{file_path}':\n   Found {num_nodes} intermediate node(s), not exactly one."
            )
    except FileNotFoundError:
        LOG.error(
            f"Error: File not found at '{file_path}'. Please ensure the file name is correct.",
            file=sys.stderr,
        )
    except Exception as e:
        LOG.error(f"Error processing ONNX file '{file_path}': {e}")
    return False


@builtin
class CTCDecoder(AxOperator):
    """
    Decodes raw sequential output from a model using CTC greedy decoding.

    This operator is designed to post-process the output tensor from a
    neural network (e.g., RNN, LSTM, Transformer) that has been trained
    on sequence prediction tasks using a Connectionist Temporal Classification
    (CTC) loss function.

    It takes the tensor containing probabilities or logits for each character
    (including a 'blank' character) at each timestep/position in the sequence.
    It then applies the CTC greedy decoding algorithm, which involves:
    1. Finding the most likely character index at each timestep (`argmax`).
    2. Removing consecutive duplicate characters.
    3. Removing all 'blank' characters.

    The result is the final predicted character sequence as a string.

    **When to use:**
    Use this operator immediately after an inference operator whose underlying
    model was trained with CTC loss for tasks like:
    - Optical Character Recognition (OCR) of text lines.
    - Automatic Speech Recognition (ASR).
    - License Plate Recognition (LPR).
    - Any other task where a variable-length character sequence is predicted
      from sequential input data using CTC.
    """

    # Add blank_index as a configurable parameter, defaulting to -1 (auto-detect)
    blank_index: int = -1

    def _post_init(self):
        self._tmp_chars: Optional[Path] = None
        self._gst_decoder_do_reduce_mean = False  # Default value

    def __del__(self):
        if self._tmp_chars is not None and self._tmp_chars.exists():
            self._tmp_chars.unlink()

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        compiled_model_dir: Path | None,
        task_graph,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, compiled_model_dir, task_graph
        )
        if not model_info.labels:
            raise ValueError(f"Missing 'labels' in model_info for {self.task_name}")
        self.chars = model_info.labels  # List of characters (strings)

        LOG.debug(
            f"Read {len(self.chars)} labels for {self.task_name}. First label: '{self.chars[0] if self.chars else ''}'"
        )

        # --- Determine Blank Index ---
        # Use self.blank_index directly as it's now a class attribute set by config/default
        configured_blank_index = self.blank_index  # Get value from config or default -1

        if configured_blank_index >= 0:
            LOG.info(f"[{self.task_name}] Using configured blank_index: {configured_blank_index}")
            if configured_blank_index >= len(self.chars):
                raise ValueError(
                    f"[{self.task_name}] Configured blank_index {configured_blank_index} out of bounds for labels (size {len(self.chars)})"
                )
            # Use the configured index
            self.blank_index = configured_blank_index
        else:
            # If not configured (it's -1), detect by finding the empty string ""
            LOG.info(
                f"[{self.task_name}] blank_index not configured, attempting to detect empty string ''"
            )
            try:
                # Find the index of the empty string in the list loaded from model_info
                detected_blank_index = self.chars.index("")
                # if no empty string found, try to find "-"
                if detected_blank_index == -1:
                    detected_blank_index = self.chars.index("-")
                    LOG.info(
                        f"[{self.task_name}] Detected hyphen '-' blank token at index: {detected_blank_index}"
                    )
                else:
                    LOG.info(
                        f"[{self.task_name}] Detected empty string '' blank token at index: {detected_blank_index}"
                    )
                self.blank_index = detected_blank_index  # Store the detected index
            except ValueError:
                # Error if "" not found and not configured
                LOG.error(
                    f"[{self.task_name}] Blank token '' not found in model_info.labels and blank_index not configured."
                )
                first_few = [f"'{c}'(len={len(c)})" for c in self.chars[:5]]
                LOG.error(
                    f"[{self.task_name}] First few labels read: [{', '.join(first_few)}, ...]"
                )
                raise ValueError(
                    f"[{self.task_name}] Could not determine CTC blank index. Ensure labels list contains '' or set 'blank_index' parameter."
                )
        # --- End Blank Index Determination ---

        # GST Related logic (check if needed for your use case)
        if model_info.manifest and model_info.manifest.is_compiled():
            postprocess_graph = compiled_model_dir / model_info.manifest.postprocess_graph
            if postprocess_graph.exists() and _check_single_reducemean_node(postprocess_graph):
                self._gst_decoder_do_reduce_mean = True
            else:
                self._gst_decoder_do_reduce_mean = False
            LOG.debug(
                f'[{self.task_name}] _gst_decoder_do_reduce_mean: {self._gst_decoder_do_reduce_mean}'
            )

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        master_meta_option = str()
        if self._where:
            master_meta_option = f'master_meta:{self._where};'

        if self._tmp_chars is None:
            self._tmp_chars = utils.create_tmp_chars(self.chars)

        gst_options = (
            f'task_category:{self._task_category.name};'
            f'meta_key:{str(self.task_name)};'
            f'{master_meta_option}'
            f'chars_file:{self._tmp_chars};'
            f'blank_index:{self.blank_index};'
            f'do_reduce_mean:{int(self._gst_decoder_do_reduce_mean)};'
        )
        gst.decode_muxer(
            name=f'decoder_task{self._taskn}{stream_idx}',
            lib='libdecode_ctc.so',
            options=gst_options,
        )

    def exec_torch(self, image, predict, axmeta):
        # Ensure predict is a numpy array
        if torch.is_tensor(predict):
            tensor = predict.cpu().detach().numpy()
        elif isinstance(predict, np.ndarray):
            tensor = predict
        else:
            LOG.error(f"[{self.task_name}] Unexpected input type for predict: {type(predict)}")
            return image, predict, axmeta  # Or raise error

        # --- CTC Greedy Decode ---
        try:
            # Revert to previous logic: Assume axis=1 for argmax
            # This assumes tensor shape is [Time, Classes] or compatible
            # Log the shape for verification
            LOG.debug(
                f"[{self.task_name}] exec_torch tensor shape: {tensor.shape}. Using axis=1 for argmax."
            )
            max_indices = np.argmax(tensor, axis=1).flatten()

        except Exception as e:
            # More context in error
            LOG.error(f"[{self.task_name}] Error during np.argmax(axis=1): {e}")
            LOG.error(f"[{self.task_name}] Tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
            return image, predict, axmeta

        # Log the blank index being used for decoding
        blank_idx = self.blank_index
        LOG.debug(f"[{self.task_name}] Starting CTC decode loop with blank_index: {blank_idx}")

        prev = blank_idx
        plate = ''
        for c in max_indices:
            if c != blank_idx and c != prev:
                if 0 <= c < len(self.chars):
                    plate += self.chars[c]
                else:
                    LOG.warning(
                        f"[{self.task_name}] Decoded index {c} out of bounds (0-{len(self.chars)-1})"
                    )
            prev = c
        # --- End Decode ---

        final_plate = plate
        LOG.trace(f"[{self.task_name}] Decoded plate: '{final_plate}'")
        model_meta = meta.LicensePlateMeta()
        model_meta.add_result(final_plate)
        axmeta.add_instance(self.task_name, model_meta, self._where)
        return image, predict, axmeta


def _reorder_embeddings_by_names(
    labels: list, ref_embedding: np.ndarray, names_file: Path
) -> tuple[list, np.ndarray, bool]:
    """Reorders embeddings and labels according to a reference names file.

    Args:
        labels: Original list of label names
        ref_embedding: Original embeddings array (n_samples x embedding_dim)
        names_file: Path to file containing reference name ordering

    Returns:
        tuple: (reordered_labels, reordered_embeddings, was_reordered)
    """
    names = names_file.read_text().splitlines()

    # Special handling for LFW names file format
    if 'lfw-names.txt' in names_file.name:
        names = [name.split()[0] for name in names]

    # Check if reordering is needed
    if labels == names:
        return labels, ref_embedding, False

    unexpected_names = set(labels) - set(names)
    assert (
        not unexpected_names
    ), f"labels: {labels} must be a subset of names: {names}, unexpected names {unexpected_names}"

    embedding_dim = ref_embedding.shape[1]
    new_embeddings = np.zeros((len(names), embedding_dim))

    # Map each name to its embedding, using zeros for missing names
    for i, name in enumerate(names):
        try:
            idx = labels.index(name)
            new_embeddings[i] = ref_embedding[idx]
        except ValueError:
            # If name not found, leave as zeros
            pass

    return names, new_embeddings, True


@builtin
class DecodeEmbeddings(AxOperator):
    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        where: str,
        compiled_model_dir: Path | None,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, where, compiled_model_dir
        )
        self._association = context.association or None

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        master_meta_option = str()
        association_key = f'association_meta:{self._association};' if self._association else str()
        self.meta_type_name = "EmbeddingsMeta"
        if self._where:
            master_meta_option = f'master_meta:{self._where};'

        gst.decode_muxer(
            name=f'decoder_task{self._taskn}{stream_idx}',
            lib='libdecode_embeddings.so',
            options=f'meta_key:{str(self.task_name)};'
            f'decoder_name:{self.meta_type_name};'
            f'{association_key}'
            f'{master_meta_option}',
        )

    def exec_torch(self, image, predict, axmeta):
        if isinstance(predict[0], torch.Tensor):
            embedding = predict[0].cpu().detach().numpy()
        else:
            embedding = predict[0]
        model_meta = meta.EmbeddingsMeta()
        model_meta.add_results(embedding)
        axmeta.add_instance(self.task_name, model_meta, self._where)

        return image, predict, axmeta


@builtin
class Recognition(AxOperator):
    embeddings_file: Union[str, embed_utils.EmbeddingsFile]
    distance_threshold: float = 0.2
    distance_metric: embed_utils.DistanceMetric = embed_utils.DistanceMetric.euclidean_distance
    k: int = 1

    # Controls whether new embeddings can be added to the embeddings file.
    # Set to True to allow adding/updating embeddings from new images.
    update_embeddings: bool = False

    # Optional path to a file containing ordered labels for consistent embedding ordering.
    # If provided, embeddings will be reordered to match this ordering.
    # The file should contain one label per line.
    names_file: Optional[str] = None

    # pair validation params
    k_fold: int = 1
    plot_roc: bool = False

    def _post_init(self):
        self._enforce_member_type('distance_metric')
        self._tmp_labels: Optional[Path] = None
        self._tmp_embeddings: Optional[Path] = None
        self.embeddings_file = embed_utils.open_embeddings_file(self.embeddings_file)
        LOG.info(f'Take embeddings file from {self.embeddings_file.path}')

        self._is_pair_validation = self.eval_mode == EvalMode.PAIR_EVAL
        if self._is_pair_validation:
            LOG.debug("Pair Verification is enabled")
        else:
            if self.names_file is not None:
                self.embeddings_file_path = self._create_reordered_embeddings_file()
            else:
                self.embeddings_file_path = self.embeddings_file.path
            self._load_reference_embeddings()

        if self._is_pair_validation:
            self.register_validation_params(
                {
                    'distance_metric': self.distance_metric,
                    'distance_threshold': self.distance_threshold,
                    'k_fold': self.k_fold,
                    'plot_roc': self.plot_roc,
                }
            )

    def __del__(self):
        self.pipeline_stopped()

    def pipeline_stopped(self):
        if (
            hasattr(self, '_tmp_labels')
            and self._tmp_labels is not None
            and self._tmp_labels.exists()
        ):
            self._tmp_labels.unlink()

        if (
            hasattr(self, '_tmp_embeddings')
            and self._tmp_embeddings is not None
            and self._tmp_embeddings.exists()
        ):
            self._tmp_embeddings.unlink()

        if self.embeddings_file and hasattr(self.embeddings_file, 'commit'):
            if self.embeddings_file.commit():
                LOG.debug(f'Embeddings committed to {self.embeddings_file.path}')

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        compiled_model_dir: Path | None,
        task_graph,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, compiled_model_dir, task_graph
        )
        if not self._is_pair_validation:
            model_info.labels = self.labels
            model_info.num_classes = self.num_classes
        self._association = context.association or None

    def _create_reordered_embeddings_file(self):
        """Create a temporary embeddings file with names_file ordering applied."""
        import json
        import time
        import uuid

        original_embeddings = self.embeddings_file.load_embeddings()
        original_labels = self.embeddings_file.read_labels()

        if original_embeddings.size == 0:
            LOG.debug("No embeddings found to reorder")
            return self.embeddings_file.path

        names_file_path = Path(self.names_file)
        LOG.debug(f"Creating reordered embeddings file using names file: {names_file_path}")

        reordered_labels, reordered_embeddings, was_reordered = _reorder_embeddings_by_names(
            original_labels, original_embeddings, names_file_path
        )

        if not was_reordered:
            LOG.debug("No reordering needed, using original embeddings file")
            return self.embeddings_file.path

        reordered_dict = {
            label: embedding.tolist()
            for label, embedding in zip(reordered_labels, reordered_embeddings)
        }

        timestamp = int(time.time())
        unique_id = str(uuid.uuid4())[:8]
        temp_path = Path("/tmp") / f"reordered_embeddings_{timestamp}_{unique_id}.json"

        with open(temp_path, 'w') as f:
            json.dump(reordered_dict, f)

        self._tmp_embeddings = Path(temp_path)
        LOG.debug(f"Created reordered embeddings file: {self._tmp_embeddings}")
        return str(self._tmp_embeddings)

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        self.meta_type_name = (
            "PairValidationMeta" if self._is_pair_validation else "ClassificationMeta"
        )
        master_key = f'master_meta:{self._where};' if self._where else str()
        association_key = f'association_meta:{self._association};' if self._association else str()
        embeddings_file_option = (
            f'embeddings_file:{self.embeddings_file_path};'
            if not self._is_pair_validation
            else str()
        )
        gst.decode_muxer(
            name=f'decoder_task{self._taskn}{stream_idx}',
            lib='libdecode_facenet.so',
            options=f'meta_key:{str(self.task_name)};'
            f'decoder_name:{self.meta_type_name};'
            f'{master_key}'
            f'{association_key}'
            f'{embeddings_file_option}'
            f'distance_threshold:{self.distance_threshold};'
            f'metric_type:{self.distance_metric.name};'
            f'pair_validation:{int(self._is_pair_validation)};'
            f'top_k:{self.k};'
            f'update_embeddings:{int(self.update_embeddings)};'
            f'{self._join_labels_for_update(gst.images)}',
        )

    def _extract_person_name_from_path(self, image_id):
        """Extract person name from an image path.

        Handles paths like:
        - AJ_Cook/AJ_Cook_0001.jpg (folder/person_name_image.jpg)
        - DIR/AJ_Cook.jpg (any_folder/person_name.jpg)
        """
        if not image_id:
            LOG.warning("Cannot extract person name: empty image_id")
            return None

        path = Path(image_id)

        if path.parent.name and path.parent.name in path.stem:
            return path.parent.name
        else:
            return path.stem

    def _join_labels_for_update(self, images):
        if not self.update_embeddings:
            return str()
        if not images:
            raise ValueError(
                "Embeddings can only be updated if the input is an image or a folder of images."
            )
        labels_for_update = 'labels_for_update:'
        for img_id in images:
            person_name = self._extract_person_name_from_path(img_id)
            if not person_name:
                raise ValueError(
                    f"Cannot extract person name from image path: {img_id}. "
                    "Ensure the image path follows the expected format."
                )
            labels_for_update += f'{person_name},'
        if labels_for_update.endswith(','):
            labels_for_update = labels_for_update[:-1]
        labels_for_update += ';'
        return labels_for_update

    def _load_reference_embeddings(self):
        self.ref_embedding = self.embeddings_file.load_embeddings()
        if self.ref_embedding.size == 0:
            LOG.info(
                "No reference embeddings found in the file. Will create new if update_embeddings=True."
            )
            self.labels = []
            self.num_classes = 0
            return

        self.labels = self.embeddings_file.read_labels()
        self.num_classes = len(self.labels)

        LOG.info(
            f"Loaded {self.num_classes} reference embeddings, shape: {self.ref_embedding.shape}"
        )

        # Reorder embeddings only once if names_file is provided
        if self.names_file is not None:
            names_file_path = Path(self.names_file)
            LOG.info(f"Using names file for embedding ordering: {names_file_path}")

            (self.labels, self.ref_embedding, was_reordered) = _reorder_embeddings_by_names(
                self.labels, self.ref_embedding, names_file_path
            )
            self.num_classes = len(self.labels)
            LOG.info(
                f"After reordering: {len(self.labels)} labels, embedding shape: {self.ref_embedding.shape}"
            )

    def exec_torch(self, image, predict, axmeta):
        embedding = predict.cpu().detach().numpy()
        if embedding.ndim != 2:  # Handle both 1D and 2D embeddings
            embedding = embedding.reshape(1, -1)

        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / norm

        if self._is_pair_validation:
            # Face Verification mode
            model_meta = axmeta.get_instance(
                self.task_name,
                meta.PairValidationMeta,
            )

            if model_meta.add_result(embedding) and self._where:
                axmeta.add_instance(self.task_name, model_meta, self._where)
                axmeta.delete_instance(self.task_name)

            if self.update_embeddings and axmeta.image_id:
                person_name = self._extract_person_name_from_path(axmeta.image_id)
                if person_name:
                    self.embeddings_file.update(embedding, person_name)

        else:
            # Check if we have reference embeddings
            if self.ref_embedding is None or self.ref_embedding.size == 0:
                if self.update_embeddings and axmeta.image_id:
                    # No reference embeddings yet, but we can start building them
                    person_name = self._extract_person_name_from_path(axmeta.image_id)
                    if person_name:
                        self.embeddings_file.update(embedding, person_name)

                        # Initialize reference embeddings with this first one
                        self.ref_embedding = embedding
                        self.labels = [person_name]
                        self.num_classes = 1
                        self._embeddings_loaded = True
                else:
                    raise RuntimeError(
                        f"No reference embedding found, please check {self.embeddings_file.path}"
                    )
                # Skip classification if we just added the first embedding
                return image, predict, axmeta

            # Perform classification
            model_meta = meta.ClassificationMeta(
                labels=self.labels,
                num_classes=self.num_classes,
            )

            # Calculate distances/similarities
            if self.distance_metric == embed_utils.DistanceMetric.euclidean_distance:
                distances_or_similarities = embed_utils.euclidean_distance(
                    embedding, self.ref_embedding
                )
                top_ids = np.argsort(distances_or_similarities)
            elif self.distance_metric == embed_utils.DistanceMetric.cosine_distance:
                distances_or_similarities = embed_utils.cosine_distance(
                    embedding, self.ref_embedding
                )
                top_ids = np.argsort(distances_or_similarities)
            elif self.distance_metric == embed_utils.DistanceMetric.cosine_similarity:
                distances_or_similarities = embed_utils.cosine_similarity(
                    embedding, self.ref_embedding
                )
                top_ids = np.argsort(-distances_or_similarities)
            else:
                raise ValueError(f'Unsupported distance metric: {self.distance_metric}')

            top_ids = top_ids[: self.k]
            top_scores = distances_or_similarities[top_ids]

            # if unseen, set to -1
            if self.distance_metric == embed_utils.DistanceMetric.cosine_similarity:
                index = np.where(top_scores < self.distance_threshold)
            else:
                index = np.where(top_scores > self.distance_threshold)
            top_ids[index] = -1
            top_scores[index] = -1

            for i in range(len(top_ids)):
                if top_ids[i] >= 0:
                    try:
                        label = self.labels(int(top_ids[i])).name
                    except TypeError:
                        label = self.labels[int(top_ids[i])]
                    LOG.debug(
                        f'top_ids: {top_ids[i]}, top_scores: {top_scores[i]}, person: {label}'
                    )

            model_meta.add_result(top_ids, top_scores)
            axmeta.add_instance(self.task_name, model_meta, self._where)

            # Handle updating embeddings in classification mode
            if self.update_embeddings and axmeta.image_id:
                person_name = self._extract_person_name_from_path(axmeta.image_id)
                if person_name:
                    # Check if this is a new person or an update
                    is_new_person = person_name not in self.labels

                    LOG.debug(
                        f"[CLASSIFICATION] {'Adding new' if is_new_person else 'Updating'} embedding for: {person_name}"
                    )
                    self.embeddings_file.update(embedding, person_name)

                    # Only update in-memory references for truly new persons
                    if is_new_person:
                        LOG.info(f"New person detected, adding to reference: {person_name}")
                        # Append new embedding to existing ones
                        self.ref_embedding = np.vstack([self.ref_embedding, embedding])
                        self.labels.append(person_name)
                        self.num_classes = len(self.labels)

        return image, predict, axmeta


@builtin
class SemanticSegmentation(AxOperator):
    '''
    Semantic Segmentation operator.

    Args:
        width: width of the output image; if not set, the width of the input image will be used
        height: height of the output image; if not set, the height of the input image will be used
        palette: palette of the output image; if not set, the palette of the input image will be used
        labels: labels of the output image; if not set, the labels of the input image will be used
        binary_threshold: threshold to decide the class map for binary segmentation
        interpolate_before_argmax: if True, interpolate logits to original resolution before argmax (improves accuracy)
    '''

    width: int = 0
    height: int = 0
    palette: list = None
    # for binay segmentation, the threshold to decide the class map
    binary_threshold: float = 1.0
    # Interpolate logits to original resolution before argmax for better accuracy
    interpolate_before_argmax: bool = False

    def _post_init(self):
        self._tmp_labels: Optional[Path] = None
        self._deq_scales = None
        self._deq_zeropoints = None
        self._n_padded_ch_outputs = None
        if (self.width > 0) != (self.height > 0):
            raise ValueError('width and height must both be set, or both unset')

    def __del__(self):
        if self._tmp_labels is not None and self._tmp_labels.exists():
            self._tmp_labels.unlink()

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        compiled_model_dir: Path | None,
        task_graph,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, compiled_model_dir, task_graph
        )
        if self.width == 0 and self.height == 0:
            self.width = model_info.input_width
            self.height = model_info.input_height
        self.num_classes = model_info.num_classes
        self.labels = model_info.labels
        self.scaled = context.resize_status
        if model_info.manifest and model_info.manifest.is_compiled():
            self._deq_scales, self._deq_zeropoints = zip(*model_info.manifest.dequantize_params)
            self._n_padded_ch_outputs = model_info.manifest.n_padded_ch_outputs
        # TODO: get labels and palette from model_info for both gst and torch pipelines

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        self.meta_type_name = "SemanticSegmentationMeta"
        options = f'meta_key:{str(self.task_name)};decoder_name:{self.meta_type_name};'
        if self._deq_scales is not None:
            scales = ','.join(str(s) for s in self._deq_scales)
            zeros = ','.join(str(s) for s in self._deq_zeropoints)
            options += f'scales:{scales};zero_points:{zeros};'
        if self._n_padded_ch_outputs:
            paddings = '|'.join(
                ','.join(str(num) for num in sublist) for sublist in self._n_padded_ch_outputs
            )
            options += f'padding:{paddings};'
        # Add letterbox parameters for proper cropping
        options += f'model_width:{self.width};model_height:{self.height};'
        options += f'scale_up:{int(self.scaled==types.ResizeMode.LETTERBOX_FIT)};'
        options += f'letterbox:{int(self.scaled in [types.ResizeMode.LETTERBOX_FIT, types.ResizeMode.LETTERBOX_CONTAIN])};'
        # Add interpolation parameter
        options += f'interpolate_before_argmax:{int(self.interpolate_before_argmax)};'
        gst.decode_muxer(
            name=f'decoder_task{self._taskn}{stream_idx}',
            lib='libdecode_semantic_seg.so',
            mode='read',
            options=options,
        )

    def _rescale(self, target_height, target_width, seg_logits):
        import torch.nn.functional as TF

        if self.scaled in [
            types.ResizeMode.LETTERBOX_FIT,
            types.ResizeMode.LETTERBOX_CONTAIN,
            types.ResizeMode.SQUISH,
        ]:
            ratio = min(self.height / target_height, self.width / target_width)
            scaled_height = int(target_height * ratio)
            scaled_width = int(target_width * ratio)
            padding_top = (self.height - scaled_height) // 2
            padding_left = (self.width - scaled_width) // 2

            seg_logits = seg_logits[
                :,
                :,
                padding_top : padding_top + scaled_height,
                padding_left : padding_left + scaled_width,
            ]
            seg_logits = TF.interpolate(
                seg_logits,
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False,
            )
        elif self.scaled == types.ResizeMode.STRETCH:
            seg_logits = TF.interpolate(
                seg_logits,
                size=(target_height, target_width),
                mode='bilinear',
                align_corners=False,
            )
        elif self.scaled == types.ResizeMode.ORIGINAL:
            pass
        else:
            raise NotImplementedError(f"Resize mode {self.scaled} is not yet implemented")
        return seg_logits

    def exec_torch(self, image, predict, axmeta):
        batch_size, C, H, W = predict.shape
        shape = [H, W, self.num_classes]
        if C == 1 and predict.dtype in [torch.int64, torch.int32]:  # already in class map
            class_map = predict.cpu().detach().numpy()[0, 0]
            class_map_tensor = (
                torch.from_numpy(class_map).unsqueeze(0).unsqueeze(0).to(torch.float32)
            )
            class_map_resized = (
                torch.nn.functional.interpolate(
                    class_map_tensor, size=(1024, 2048), mode='nearest'
                )
                .squeeze(0)
                .squeeze(0)
            )
            class_map_resized = class_map_resized.numpy().astype(np.int32)
            prob_map = np.ones((batch_size, H, W), dtype=np.float32)
            for b in range(batch_size):
                model_meta = meta.SemanticSegmentationMeta(
                    shape=shape,
                    class_map=class_map_resized,
                    probabilities=prob_map[b],
                    seg_logits=None,
                    labels=[],  # Placeholder for future label integration
                    palette=[],  # Placeholder for future palette integration
                )
                axmeta.add_instance(self.task_name, model_meta, self._where)
            return image, predict, axmeta

        assert C == self.num_classes, f'C: {C} != num_classes: {self.num_classes}'
        # TODO: depadding
        # i_seg_logits = seg_logits[i:i + 1, :,
        #                           padding_top:H - padding_bottom,
        #                           padding_left:W - padding_right]
        # seg_logits shape is 1, C, H, W after remove padding
        if isinstance(predict, np.ndarray):
            seg_logits = torch.from_numpy(predict)
        elif torch.is_tensor(predict):
            seg_logits = predict.clone()

        if C > 1:
            # prob_map, class_map = [m.cpu().detach().numpy()[0] for m in predict.max(1)]
            # Move the tensor to CPU, detach it from the computation graph, and convert it to a numpy array
            seg_logits = self._rescale(image.height, image.width, seg_logits)
            predict_np = seg_logits.cpu().detach().numpy()
            # Compute the class map (indices of max values) along the class dimension
            # and the corresponding probability map (max values)
            class_map = np.argmax(predict_np, axis=1)  # Shape: [batch_size, H, W]
            prob_map = np.max(predict_np, axis=1)  # Shape: [batch_size, H, W]
        else:  # For binary segmentation, apply sigmoid and threshold to predict map
            seg_logits = seg_logits.sigmoid()
            seg_logits = self._rescale(image.height, image.width, seg_logits)
            prob_map = (
                seg_logits.cpu().detach().numpy()
            )  # Shape: [batch_size, H, W], as probability map
            class_map = (prob_map > self.binary_threshold).astype(
                np.int32
            )  # Shape: [batch_size, H, W]

        for b in range(batch_size):
            model_meta = meta.SemanticSegmentationMeta(
                shape=shape,
                class_map=class_map[b],
                probabilities=prob_map[b],
                seg_logits=seg_logits[b],
                labels=[],  # self.labels,
                palette=[],  # self.palette
            )
            axmeta.add_instance(self.task_name, model_meta, self._where)
        return image, predict, axmeta


@builtin
class GetRawTensor(AxOperator):
    """
    This operator is used to get the raw tensor output from the model.
    It is useful when the model output is not a valid tensor, such as a list of tensors.

    TODO:
    - consider using GetRawTensor automatically when --tensor
    - set default as True when we have a powerful SuperPostamble to handle all the transforms efficiently
    """

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        where: str,
        compiled_model_dir: Path | None,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, where, compiled_model_dir
        )

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        master_meta_option = str()
        if self._where:
            master_meta_option = f'master_meta:{self._where};'

        gst.decode_muxer(
            name=f'decoder_task{self._taskn}{stream_idx}',
            lib='libdecode_to_raw_tensor.so',
            options=f'meta_key:{str(self.task_name)};' f'{master_meta_option}',
        )

    def exec_torch(self, image, predict, axmeta):
        model_meta = meta.TensorMeta(
            tensors=[predict.cpu().detach().numpy()],
        )
        axmeta.add_instance(self.task_name, model_meta, self._where)
        return image, predict, axmeta

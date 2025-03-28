# Copyright Axelera AI, 2024
# General post-processing operators
from __future__ import annotations

import json
from pathlib import Path
import tempfile
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
        where: str,
        compiled_model_dir: Path,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, where, compiled_model_dir
        )
        self.labels = model_info.labels
        self.num_classes = model_info.num_classes

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        master_meta_option = str()
        if self._where:
            master_meta_option = f'master_meta:{self._where};'

        # TODO only k == 1, largest==True, and sorted==True is supported,
        if self._tmp_labels is None:
            self._tmp_labels = utils.create_tmp_labels(self.labels)
        if not gst.new_inference:
            conns = {'src': f'decoder_task{self._taskn}{stream_idx}.sink_0'}
            if self._where:
                gst.queue(
                    {'max-size-buffers': 0},
                    connections=conns,
                    name=f'queue_decoder_task{self._taskn}{stream_idx}',
                )
            else:
                gst.queue(
                    connections=conns,
                    name=f'queue_decoder_task{self._taskn}{stream_idx}',
                )

        gst.decode_muxer(
            name=f'decoder_task{self._taskn}{stream_idx}',
            lib='libdecode_classification.so',
            options=f'meta_key:{str(self.task_name)};'
            f'{master_meta_option}'
            f'classlabels_file:{self._tmp_labels};'
            f'top_k:{self.k};'
            f'softmax:{int(self.softmax)}',
        )

    def exec_torch(self, image, predict, axmeta):
        model_meta = meta.ClassificationMeta(
            labels=self.labels,
            num_classes=self.num_classes,
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


def _reorder_embeddings_by_names(
    labels: list, ref_embedding: np.ndarray, names_file: Path
) -> tuple[list, np.ndarray]:
    """Reorders embeddings and labels according to a reference names file.

    where embeddings built from LFWPair dataset have a different order compared to LFWPeople.
    This function ensures consistent ordering between different LFW variants.

    Args:
        labels: Original list of label names
        ref_embedding: Original embeddings array (n_samples x embedding_dim)
        names_file: Path to file containing reference name ordering

    Returns:
        tuple: (reordered_labels, reordered_embeddings)
    """
    names = names_file.read_text().splitlines()

    # Special handling for LFW names file format
    if names_file.name == 'lfw-names.txt':
        names = [name.split()[0] for name in names]

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

    return names, new_embeddings


@builtin
class Recognition(AxOperator):
    embeddings_file: Union[str, embed_utils.EmbeddingsFile]
    distance_threshold: float = 0.2
    distance_metric: embed_utils.DistanceMetric = embed_utils.DistanceMetric.euclidean_distance
    k: int = 1
    generate_embeddings: bool = False

    # a file containing the labels in a specific order
    names_file: Optional[str] = None

    # pair validation params
    k_fold: int = 1
    plot_roc: bool = False

    def _post_init(self):
        self._enforce_member_type('distance_metric')
        self._tmp_labels: Optional[Path] = None
        self.embeddings_file = embed_utils.open_embeddings_file(self.embeddings_file)
        LOG.info(f'Take embeddings file from {self.embeddings_file.path}')

        self._is_pair_validation = self.eval_mode == EvalMode.PAIR_EVAL
        if self._is_pair_validation:
            LOG.debug("Pair Verification is enabled")
        else:
            self.ref_embedding = None

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
            self.embeddings_file
            and hasattr(self.embeddings_file, 'dirty')
            and self.embeddings_file.dirty
        ):
            self.embeddings_file.commit()
            LOG.info(f'Embeddings committed to {self.embeddings_file.path}')

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        where: str,
        compiled_model_dir: Path,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, where, compiled_model_dir
        )
        self.labels = model_info.labels

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        raise NotImplementedError()

    def exec_torch(self, image, predict, axmeta):
        if not self._is_pair_validation:
            if self.ref_embedding is None:
                self.ref_embedding = self.embeddings_file.load_embeddings()
                if self.ref_embedding.size == 0:
                    raise RuntimeError(
                        f'No reference embedding found, please check {self.embeddings_file.path}'
                    )
                self.labels = self.embeddings_file.read_labels(self.labels)
                self.num_classes = len(self.labels)

                if self.names_file is not None:
                    self.labels, self.ref_embedding = _reorder_embeddings_by_names(
                        self.labels, self.ref_embedding, Path(self.names_file)
                    )

            model_meta = meta.ClassificationMeta(
                labels=self.labels,
                num_classes=self.num_classes,
            )

        # find closest embedding
        embedding = predict.cpu().detach().numpy()
        # L2 norm, equal to TF.normalize(predict, p=2, dim=1)
        norm = np.linalg.norm(embedding, axis=1, keepdims=True)
        embedding = embedding / norm

        if not self._is_pair_validation:
            embedding.shape, self.ref_embedding.shape
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
                try:
                    label = self.labels(int(top_ids[i])).name
                except TypeError:
                    label = self.labels[int(top_ids[i])]
                LOG.trace(f'top_ids: {top_ids[i]}, top_scores: {top_scores[i]}, person: {label}')

            model_meta.add_result(top_ids, top_scores)
            axmeta.add_instance(self.task_name, model_meta, self._where)
        else:
            # print(f"axmeta: {axmeta}")
            model_meta = axmeta.get_instance(
                self.task_name,
                meta.PairValidationMeta,
            )
            if model_meta.add_result(embedding) and self._where:
                # put the task meta into the secondary meta map
                axmeta.add_instance(self.task_name, model_meta, self._where)
                axmeta.delete_instance(self.task_name)
            if self.generate_embeddings:
                self.embeddings_file.update(embedding, axmeta.image_id)

        return image, predict, axmeta


@builtin
class Tracker(AxOperator):
    '''Configure tracker.'''

    algorithm: str = 'oc-sort'
    algo_params: dict = None
    # history_length is now maintained by gst; TODO: support this in Python
    history_length: int = 30
    num_subtask_runs: int = 10

    def _post_init(self):
        supported_algorithms = ['sort', 'scalarmot', 'oc-sort', 'bytetrack']
        self.algorithm = self.algorithm.lower()
        assert (
            self.algorithm in supported_algorithms
        ), f'Only {supported_algorithms} are supported for now'
        assert (
            isinstance(self.algo_params, dict) or self.algo_params is None
        ), f'algo_params must be a dict or None, got {type(self.algo_params)}'
        self._verify_params()
        self._algo_params_json: Optional[Path] = None

    def _verify_params(self):
        if self.algo_params is None:
            return
        if self.algorithm == 'oc-sort':
            supported_params = [
                'det_thresh',
                'max_age',
                'min_hits',
                'iou_threshold',
                'delta',
                'asso_func',
                'inertia',
                'use_byte',
                'max_id',
            ]
        elif self.algorithm == 'bytetrack':
            supported_params = [
                'frame_rate',
                'track_buffer',
            ]
        elif self.algorithm == 'sort':
            supported_params = [
                'det_thresh',
                'maxAge',
                'minHits',
                'iouThreshold',
            ]
        elif self.algorithm == 'scalarmot':
            supported_params = [
                'maxLostFrames',
            ]

        for k in self.algo_params:
            assert (
                k in supported_params
            ), f'Only {supported_params} are supported for {self.algorithm}'

    def __del__(self):
        if self._algo_params_json is not None and self._algo_params_json.exists():
            self._algo_params_json.unlink()

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        if self._algo_params_json is None:
            with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as t:
                t.write(json.dumps(self.algo_params))
            self._algo_params_json = Path(t.name)
        gst.axinplace(
            lib='libinplace_tracker.so',
            options=f'meta_key:{str(self.task_name)};'
            f'history_length:{self.history_length};'
            f'num_subtask_runs:{self.num_subtask_runs};'
            f'algorithm:{self.algorithm.lower()};'
            f'algo_params_json:{self._algo_params_json};',
        )

    def exec_torch(self, image, predict, axmeta):
        raise NotImplementedError("Tracker is not yet implemented for torch pipeline")


@builtin
class FilterDetections(AxOperator):
    min_width: int = 0
    min_height: int = 0

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        gst.axinplace(
            lib='libinplace_filterdetections.so',
            options=f'meta_key:{self.task_name};min_width:{self.min_width};min_height:{self.min_height}',
            mode='read',
        )

    def exec_torch(self, image, predict, axmeta):
        raise NotImplementedError("FilterDetections is not yet implemented for torch pipeline")


@builtin
class AddClassificationsToTracker(AxOperator):
    where: str

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        gst.axinplace(
            lib='libinplace_trackeraddclassifications.so',
            options=f'classification_meta_key:{str(self.task_name)};'
            f'tracking_meta_key:{str(self.where)};',
            mode='read',
        )

    def exec_torch(self, image, predict, axmeta):
        raise NotImplementedError(
            "AddClassificationsToTracker is not yet implemented for torch pipeline"
        )


@builtin
class AddKeypointsToTracker(AxOperator):
    where: str

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        gst.axinplace(
            lib='libinplace_trackeraddkeypoints.so',
            options=f'keypoints_submeta_key:{str(self.where)};'
            f'tracking_meta_key:{str(self.where)};',
            mode='read',
        )

    def exec_torch(self, image, predict, axmeta):
        raise NotImplementedError(
            "AddKeypointsToTracker is not yet implemented for torch pipeline"
        )


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
    '''

    width: int = 0
    height: int = 0
    palette: list = None
    # for binay segmentation, the threshold to decide the class map
    binary_threshold: float = 1.0

    def _post_init(self):
        self._tmp_labels: Optional[Path] = None
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
        where: str,
        compiled_model_dir: Path,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, where, compiled_model_dir
        )
        if self.width == 0 and self.height == 0:
            self.width = model_info.input_width
            self.height = model_info.input_height
        self.num_classes = model_info.num_classes
        self.labels = model_info.labels
        self.scaled = context.resize_status
        # TODO: get labels and palette from model_info for both gst and torch pipelines

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        self.meta_type_name = "SemanticSegmentationMeta"
        if not gst.new_inference:
            conns = {'src': f'decoder_task{self._taskn}{stream_idx}.sink_0'}
            gst.queue(name=f'queue_decoder_task{self._taskn}{stream_idx}', connections=conns)

        gst.decode_muxer(
            name=f'decoder_task{self._taskn}{stream_idx}',
            lib='libdecode_semantic_seg.so',
            mode='read',
            options=f'meta_key:{str(self.task_name)};' f'decoder_name:{self.meta_type_name};',
        )

    def _rescale(self, target_height, target_width, seg_logits):
        import torch.nn.functional as TF

        if self.scaled in [types.ResizeMode.LETTERBOX_FIT, types.ResizeMode.SQUISH]:
            ratio = min(self.height / target_height, self.width / target_width)
            scaled_height = int(target_height * ratio)
            scaled_width = int(target_width * ratio)
            padding_top = (self.height - scaled_height) // 2
            padding_left = (self.width - scaled_width) // 2

            # Correct slicing to remove padding
            seg_logits = seg_logits[
                :,
                :,
                padding_top : padding_top + scaled_height,
                padding_left : padding_left + scaled_width,
            ]
            # scale back to original size
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

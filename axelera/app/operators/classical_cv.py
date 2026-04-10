# Copyright Axelera AI, 2025
# classical cv operators
from __future__ import annotations

import json
from pathlib import Path
import tempfile
from typing import Any, Dict, List, Optional, Union

from axelera import types

from .. import gst_builder, logging_utils
from .base import BaseClassicalCV, builtin_classical_cv
from .context import PipelineContext

LOG = logging_utils.getLogger(__name__)


@builtin_classical_cv
class Tracker(BaseClassicalCV):
    '''Configure tracker.'''

    bbox_task_name: str  # a meta which contains bounding boxes
    embeddings_task_name: str | None = None  # a meta which contains embeddings
    algorithm: str = 'oc-sort'
    algo_params: Dict[str, Union[int, float, bool]] = None
    # history_length is now maintained by gst; TODO: support this in Python
    history_length: int = 30
    filter_callbacks: Dict[str, Dict] = None
    determine_object_attribute_callbacks: Dict[str, Dict] = None
    min_width: int = 0
    min_height: int = 0
    label_filter: List[str] = []

    def _post_init(self):
        supported_algorithms = [
            'sort',
            'scalarmot',
            'oc-sort',
            'bytetrack',
            'tracktrack',
        ]
        self.algorithm = self.algorithm.lower()
        assert (
            self.algorithm in supported_algorithms
        ), f'Only {supported_algorithms} are supported for now'
        assert (
            isinstance(self.algo_params, dict) or self.algo_params is None
        ), f'algo_params must be a dict or None, got {type(self.algo_params)}'
        self._verify_params()
        self._algo_params_json: Optional[Path] = None

        def _adapt_callback_dict(
            name: str, callback_dict: dict[str, dict[str, Any]] | None
        ) -> None:
            # Adapts the callback dictionary in-place to ensure all values are strings.
            if callback_dict is None:
                return
            assert isinstance(
                callback_dict, dict
            ), f'{name} must be a dict|None, got {type(callback_dict)}'
            res = {}
            for task_name, options in callback_dict.items():
                assert 'lib' in options, f'lib must be provided for {name}'
                res[task_name] = {k: str(v) for k, v in options.items()}
            callback_dict.update(res)

        _adapt_callback_dict('filter_callbacks', self.filter_callbacks)
        _adapt_callback_dict(
            'determine_object_attribute_callbacks', self.determine_object_attribute_callbacks
        )

        self._submodels_with_boxes_from_tracker = set(k for k in self.filter_callbacks or {})

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
        context.submodels_with_boxes_from_tracker = self._submodels_with_boxes_from_tracker

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
                'inertia',
                'w_assoc_emb',
                'alpha_fixed_emb',
                'max_id',
                'aw_enabled',
                'aw_param',
                'cmc_enabled',
                'enable_id_recovery',
                'rec_image_rect_margin',
                'rec_track_min_time_since_update_at_boundary',
                'rec_track_min_time_since_update_inside',
                'rec_track_min_age',
                'rec_track_merge_lap_thresh',
                'rec_track_memory_capacity',
                'rec_track_memory_max_age',
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
        elif self.algorithm == 'tracktrack':
            supported_params = [
                'det_thr',
                'init_thr',
                'match_thr',
                'tai_thr',
                'penalty_p',
                'penalty_q',
                'reduce_step',
                'max_time_lost',
                'min_len',
                'min_box_area',
                'alpha',
                'use_cmc',
                'dataset_type',
            ]

        for k in self.algo_params:
            if k not in supported_params:
                raise ValueError(f'Only {supported_params} are supported for {self.algorithm}')

        if self.algorithm == 'oc-sort' and self.algo_params.get('enable_id_recovery'):
            max_age = self.algo_params.get('max_age')
            boundary = self.algo_params.get('rec_track_min_time_since_update_at_boundary')
            inside = self.algo_params.get('rec_track_min_time_since_update_inside')
            if max_age is not None:
                if boundary is not None:
                    if max_age < boundary:
                        raise ValueError(
                            'max_age must be >= rec_track_min_time_since_update_at_boundary when '
                            'enable_id_recovery is true; this is necessary for the ID recovery feature '
                            '(memory bank) to work correctly; boundary reappearing trackers may never '
                            'be added when max_age is smaller.'
                        )
                if inside is not None:
                    if max_age < inside:
                        raise ValueError(
                            'max_age must be >= rec_track_min_time_since_update_inside when '
                            'enable_id_recovery is true; this is necessary for the ID recovery feature '
                            '(memory bank) to work correctly; inside reappearing trackers may never '
                            'be added when max_age is smaller.'
                        )

    def __del__(self):
        if self._algo_params_json is not None and self._algo_params_json.exists():
            self._algo_params_json.unlink()

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        if self._algo_params_json is None:
            with tempfile.NamedTemporaryFile(suffix='.json', mode='w', delete=False) as t:
                t.write(json.dumps(self.algo_params))
            self._algo_params_json = Path(t.name)

        embeddings_meta_key = (
            str()
            if self.embeddings_task_name is None
            else f'embeddings_meta_key:{self.embeddings_task_name};'
        )

        def _fill_callbacks_option(callbacks: Dict[str, Dict[str, Any]], name: str) -> str:
            if not callbacks:
                return str()
            option = f'{name}:'
            for key, value in callbacks.items():
                option += f'key={key}'
                for k, v in value.items():
                    option += f'&{k}={v}'
                option += '?'
            option = option[:-1]
            option += ';'
            return option

        filter_callbacks_option = _fill_callbacks_option(self.filter_callbacks, 'filter_callbacks')
        determine_object_attribute_callbacks_option = _fill_callbacks_option(
            self.determine_object_attribute_callbacks, 'determine_object_attribute_callbacks'
        )

        input_to_tracker_meta = self.bbox_task_name

        if (self.min_width > 0) or (self.min_height > 0):
            input_to_tracker_meta = f'{self.bbox_task_name}_adapted_as_input_for_{self.task_name}'
            gst.axinplace(
                lib='libinplace_filterdetections.so',
                options=f'input_meta_key:{self.bbox_task_name};'
                f'output_meta_key:{input_to_tracker_meta};'
                f'hide_output_meta:1;'
                f'min_width:{self.min_width};'
                f'min_height:{self.min_height}',
            )

        mode = ''
        if (
            self.algo_params
            and 'cmc_enabled' in self.algo_params
            and self.algo_params['cmc_enabled']
        ):
            mode = 'read'

        gst.axinplace(
            lib='libinplace_tracker.so',
            mode=mode,
            options=f'tracker_meta_key:{self.task_name};'
            f'input_meta_key:{input_to_tracker_meta};'
            f'output_meta_key:boxes_created_by_tracker_task_{self.task_name};'
            f'{embeddings_meta_key}'
            f'history_length:{self.history_length};'
            f'algorithm:{self.algorithm.lower()};'
            f'algo_params_json:{self._algo_params_json};'
            f'{filter_callbacks_option}'
            f'{determine_object_attribute_callbacks_option}',
        )

    def exec_torch(self, image, predict, axmeta):
        raise NotImplementedError("Tracker is not yet implemented for torch pipeline")

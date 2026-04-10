# Copyright Axelera AI, 2023
# Metadata for tracker
from __future__ import annotations

from dataclasses import dataclass, field
import importlib
from typing import Any, ClassVar

import numpy as np

from axelera.app.meta.classification import ClassificationMeta
from axelera.app.meta.keypoint import CocoBodyKeypointsMeta

from .. import display, eval_interfaces, plot_utils
from .base import AxTaskMeta, MetaObject, class_as_label
from .gst_decode_utils import decode_bbox


class TrackedObject(MetaObject):
    def __init__(self, meta, index, track_id):
        super().__init__(meta, index)
        self._track_id = track_id

    @property
    def track_id(self):
        return self._track_id

    @property
    def history(self):
        return self._meta.tracking_history[self.track_id]

    @property
    def class_id(self):
        return self._meta.class_ids[self._index]

    def get_object_meta(
        self, task_name: str, include_frame_meta: bool = True
    ) -> AxTaskMeta | None:
        """Return tracker-level object metadata for this task if available."""
        meta = self._meta
        containers = []
        if isinstance(getattr(meta, 'object_meta', None), dict):
            containers.append(meta.object_meta)
        if include_frame_meta and isinstance(getattr(meta, 'frame_object_meta', None), dict):
            containers.append(meta.frame_object_meta)

        for container in containers:
            per_task = container.get(task_name)
            if not isinstance(per_task, dict):
                continue
            if self.track_id in per_task:
                return per_task[self.track_id]
        return None


_red = (255, 0, 0, 255)
_yellow = (255, 255, 0, 255)


def _track_id_as_label(track_id: int, labels: list | None = None) -> str:
    if not labels:
        return f'id:{track_id}'
    return f"{track_id}"


# the dataclasses for each computer vision task
@dataclass(frozen=True)
class TrackerMeta(AxTaskMeta):
    """Metadata for tracker task"""

    Object: ClassVar[MetaObject] = TrackedObject
    META_TYPE: ClassVar[str] = 'tracking_meta'

    # key is the track id, value is the bbox history
    tracking_history: dict[int, np.ndarray] = field(default_factory=dict)
    class_ids: list[int] = field(default_factory=list)
    object_meta: dict[str, dict[int, AxTaskMeta]] = field(default_factory=dict)
    frame_object_meta: dict[str, dict[int, AxTaskMeta]] = field(default_factory=dict)
    labels: list | None = field(default_factory=lambda: None, repr=False)
    labels_dict: dict[str, list] = field(default_factory=dict)
    extra_info: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert len(self.tracking_history) == len(
            self.class_ids
        ), f"Number of track ids {len(self.tracking_history)} does not match number of class ids {len(self.class_ids)}"

    def draw(self, draw: display.Draw):
        for idx, (track_id, bboxes) in enumerate(self.tracking_history.items()):
            cmap = draw.options.bbox_class_colors
            color = cmap.get(track_id, plot_utils.get_color(track_id))

            tag = ''
            if self.task_render_config.show_labels:
                label = ''
                if (class_id := self.class_ids[idx]) != -1:
                    label = f'{class_as_label(self.labels, class_id)}'
                track_id_label = _track_id_as_label(track_id, self.labels)
                fmt = draw.options.tracker_label_format
                tag = fmt.format(
                    label=label,
                    track_id=track_id_label,
                )
                for submeta_key, values in self.object_meta.items():
                    value = values.get(track_id)
                    if value is None:
                        continue
                    if isinstance(value, ClassificationMeta):
                        class_id = value._class_ids[0][0]
                        score = value._scores[0][0]
                        submeta_fmt = draw.options.bbox_label_format

                        sublabel = submeta_fmt.format(
                            label=class_as_label(self.labels_dict[submeta_key], class_id),
                            score=score,
                            scorep=score * 100,
                            scoreunit='%',
                        )
                        if sublabel:
                            if tag:
                                tag += f'\n{submeta_key}: {sublabel}'
                            else:
                                tag = f'{submeta_key}: {sublabel}'

            bbox = bboxes[-1]
            if np.all(bbox == 0):
                continue
            if self.task_render_config.show_annotations and draw.options.show_bounding_boxes:
                # Show box with label at left-top
                draw.labelled_box((bbox[0], bbox[1]), (bbox[2], bbox[3]), tag, color)
            elif tag and draw.options.show_bounding_boxes:
                # Label only at left-top (zero-sized box)
                draw.labelled_box((bbox[0], bbox[1]), (bbox[0], bbox[1]), tag, color)
            elif tag and draw.options.tracker_label_format:
                # No box shown but tracking enabled: show ID at center-top (floating label)
                mid_x = (bbox[0] + bbox[2]) // 2
                draw.labelled_box((mid_x, bbox[1]), (mid_x, bbox[1]), tag, color)
            # Draw trajectory independently of bounding boxes
            if draw.options.show_trajectory:
                draw.trajectory(bboxes, color)

        for submeta_key, values in self.frame_object_meta.items():
            for track_id, value in values.items():
                if value is None:
                    continue
                if isinstance(value, CocoBodyKeypointsMeta):
                    value.draw(draw)

    @classmethod
    def decode(cls, data: dict[str, bytes | bytearray]) -> 'TrackerMeta':
        meta_module = importlib.import_module('axelera.app.meta')
        tracking_history: dict[int, np.ndarray] = {}
        class_ids: list[int] = []
        object_meta: dict[str, dict[int, AxTaskMeta]] = {}
        frame_object_meta: dict[str, dict[int, AxTaskMeta]] = {}
        objmeta_key_to_string: dict[int, str] = {}

        key_data = data.get('objmeta_keys', b'') or b''
        key_data_size = len(key_data)
        while key_data_size > 0:
            objmeta_key = int(np.frombuffer(key_data[:1], dtype=np.uint8)[0])
            objmeta_string_size = int(np.frombuffer(key_data[1:9], dtype=np.uint64)[0])
            start = 9
            end = start + objmeta_string_size
            objmeta_string = key_data[start:end].decode('utf-8')
            objmeta_key_to_string[objmeta_key] = objmeta_string
            key_data = key_data[end:]
            key_data_size = len(key_data)

        for key, track_data in data.items():
            if not key.startswith('track_'):
                continue
            track_id = int(key[6:])
            class_id = int(np.frombuffer(track_data[:4], dtype=np.int32)[0])
            class_ids.append(class_id)
            num_boxes = int(np.frombuffer(track_data[4:8], dtype=np.int32)[0])
            bbox_start = 8
            bbox_end = bbox_start + num_boxes * 16
            bbox_data = {'bbox': track_data[bbox_start:bbox_end]}
            tracking_history[track_id] = decode_bbox(bbox_data)
            offset = bbox_end
            offset = _process_object_metadata(
                track_id,
                meta_module,
                frame_object_meta,
                objmeta_key_to_string,
                track_data,
                offset,
            )
            offset = _process_object_metadata(
                track_id,
                meta_module,
                object_meta,
                objmeta_key_to_string,
                track_data,
                offset,
            )

        return cls(
            tracking_history=tracking_history,
            class_ids=class_ids,
            object_meta=object_meta,
            frame_object_meta=frame_object_meta,
        )

    @property
    def boxes(self) -> dict[int, np.ndarray]:
        """Returns the current boxes for all tracks as a dict mapping track_id to box."""
        return {track_id: history[-1] for track_id, history in self.tracking_history.items()}

    @property
    def objects(self) -> list[TrackedObject]:
        if not self._objects:
            self._objects.extend(
                self.Object(self, idx, track_id)
                for idx, track_id in enumerate(self.tracking_history.keys())
            )
        return self._objects

    def to_evaluation(self):
        if not self.access_ground_truth():
            raise ValueError("Ground truth is not set")

        bboxes = []
        track_ids = []

        for idx, (track_id, bbox_history) in enumerate(self.tracking_history.items()):
            last_bbox = bbox_history[-1]

            bboxes.append(last_bbox)
            track_ids.append(track_id)

        if len(self.class_ids) > 0:
            prediction = eval_interfaces.TrackerEvalSample(
                np.array(bboxes, dtype=np.float32),
                np.array(track_ids, dtype=np.int32),
                np.array(self.class_ids, dtype=np.int32),
            )
        else:
            prediction = eval_interfaces.TrackerEvalSample()

        return prediction


def _process_object_metadata(
    track_id: int,
    meta_module,
    destination: dict[str, dict[int, AxTaskMeta]],
    objmeta_key_to_string: dict[int, str],
    track_data: bytes | bytearray,
    offset: int,
) -> int:
    num_objmeta = int(np.frombuffer(track_data[offset : offset + 4], dtype=np.int32)[0])
    offset += 4
    if num_objmeta <= 0:
        return offset

    results_objmeta: dict[tuple[str, str], dict[str, bytes | bytearray]] = {}
    for _ in range(num_objmeta):
        objmeta_key = int(np.frombuffer(track_data[offset : offset + 1], dtype=np.uint8)[0])
        offset += 1
        objmeta_string = objmeta_key_to_string.get(objmeta_key, f"key_{objmeta_key}")

        metavec_size = int(np.frombuffer(track_data[offset : offset + 4], dtype=np.int32)[0])
        offset += 4

        for _ in range(metavec_size):
            objmeta_type_size = int(
                np.frombuffer(track_data[offset : offset + 4], dtype=np.int32)[0]
            )
            offset += 4
            objmeta_type = track_data[offset : offset + objmeta_type_size].decode('utf-8')
            offset += objmeta_type_size

            objmeta_subtype_size = int(
                np.frombuffer(track_data[offset : offset + 4], dtype=np.int32)[0]
            )
            offset += 4
            objmeta_subtype = track_data[offset : offset + objmeta_subtype_size].decode('utf-8')
            offset += objmeta_subtype_size

            objmeta_size = int(np.frombuffer(track_data[offset : offset + 4], dtype=np.int32)[0])
            offset += 4
            objmeta_data = track_data[offset : offset + objmeta_size]
            offset += objmeta_size
            if objmeta_size == 0:
                continue

            results_key = (objmeta_string, objmeta_type)
            entry = results_objmeta.setdefault(results_key, {})
            if objmeta_subtype in entry:
                objmeta_data = entry[objmeta_subtype] + objmeta_data
            entry[objmeta_subtype] = objmeta_data

    for (objmeta_name, meta_type), results_data in results_objmeta.items():
        meta_class = getattr(meta_module, meta_type)
        decoded_meta = meta_class.decode(results_data)
        destination.setdefault(objmeta_name, {})[track_id] = decoded_meta

    return offset

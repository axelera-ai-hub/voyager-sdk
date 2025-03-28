# Copyright Axelera AI, 2023
# Metadata for tracker
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, ClassVar, Dict, List, Optional

import numpy as np

from axelera.app.meta.classification import ClassificationMeta
from axelera.app.meta.keypoint import CocoBodyKeypointsMeta

from .. import display, plot_utils
from .base import AxTaskMeta, MetaObject, class_as_label


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


_red = (255, 0, 0, 255)
_yellow = (255, 255, 0, 255)


# the dataclasses for each computer vision task
@dataclass(frozen=True)
class TrackerMeta(AxTaskMeta):
    """Metadata for tracker task"""

    Object: ClassVar[MetaObject] = TrackedObject

    # key is the track id, value is the bbox history
    tracking_history: Dict[int, np.ndarray] = field(default_factory=dict)
    class_ids: List[int] = field(default_factory=list)
    object_meta: Dict[str, List[AxTaskMeta]] = field(default_factory=dict)
    frame_object_meta: Dict[str, List[AxTaskMeta]] = field(default_factory=dict)
    labels: Optional[list] = field(default_factory=lambda: None, repr=False)
    labels_dict: Dict[str, list] = field(default_factory=dict)
    extra_info: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        assert len(self.tracking_history) == len(
            self.class_ids
        ), f"Number of track ids {len(self.tracking_history)} does not match number of class ids {len(self.class_ids)}"

    def draw(self, draw: display.Draw):
        for idx, (track_id, bboxes) in enumerate(self.tracking_history.items()):
            color = plot_utils.get_color(track_id)
            label = ''
            if (class_id := self.class_ids[idx]) != -1:
                label = f'{class_as_label(self.labels, class_id)}-'
            tag = f'{label}{track_id}' if self.labels else f'{label}id {track_id}'
            for key, value_list in self.object_meta.items():
                value = value_list[idx]
                if isinstance(value, ClassificationMeta):
                    class_id = value._class_ids[0][0]
                    sublabel = (
                        class_as_label(self.labels_dict[key], class_id)
                        if self.labels_dict[key]
                        else str(class_id)
                    )
                    tag += f'\n{key}: {sublabel}'

            # draw the latest bbox
            bbox = bboxes[-1]
            draw.labelled_box((bbox[0], bbox[1]), (bbox[2], bbox[3]), tag, color)

            # draw the trajectory
            draw.trajectory(bboxes, color)

        for key, value_list in self.frame_object_meta.items():
            for idx, value in enumerate(value_list):
                if isinstance(value, CocoBodyKeypointsMeta):
                    value.draw(draw)

    @property
    def objects(self) -> List[TrackedObject]:
        if not self._objects:
            self._objects.extend(
                self.Object(self, idx, track_id)
                for idx, track_id in enumerate(self.tracking_history.keys())
            )
        return self._objects

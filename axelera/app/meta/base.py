# Copyright Axelera AI, 2023
# Base dataclasses used to represent metadata
from __future__ import annotations

import abc
import collections
import dataclasses
import functools
import itertools
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Type, TypeVar, final

from axelera import types

from .. import config, display, exceptions, logging_utils, plot_utils, utils
from ..model_utils.box import convert

if TYPE_CHECKING:
    from ..eval_interfaces import BaseEvalSample

LOG = logging_utils.getLogger(__name__)
_VALID_LABEL_FORMATS = {'label': 'label', 'score': 0.56, 'scorep': 56.7, 'scoreunit': '%'}
_DEFAULT_LABEL_FORMAT = display.Options().bbox_label_format


class NoMasterDetectionsError(Exception):
    """Exception raised when no master detections are found."""

    def __init__(self, master_key: str):
        self.message = f"No master detections were found for {master_key}"
        super().__init__(self.message)


class AggregationNotRequiredForEvaluation(Exception):
    """Exception raised when no aggregation is needed for evaluating a task."""

    def __init__(self, cls: Type[AxTaskMeta]):
        self.message = f"Aggregation is not needed for {cls.__name__}"
        super().__init__(self.message)


def class_as_label(labels, class_id):
    '''Labels may be enumerated, and therefore callable. Otherwise access via index.'''
    if not labels:
        return f"cls:{class_id}"
    if class_id == -1:
        return "Unknown"
    try:
        return labels(class_id).name
    except TypeError:
        return labels[class_id]
    except ValueError:
        pass
    return str(class_id)


@functools.lru_cache(maxsize=200)
def safe_label_format(fmt: str) -> str:
    try:
        fmt.format(**_VALID_LABEL_FORMATS)
        return fmt
    except ValueError as e:
        LOG.error("Error in bbox_label_format: %s (%s)", fmt, str(e))
    except KeyError as e:
        valid = ', '.join(f"{k}" for k in _VALID_LABEL_FORMATS)
        LOG.error(
            "Unknown name %s in bbox_label_format '%s', valid names are %s", str(e), fmt, valid
        )
    return _DEFAULT_LABEL_FORMAT


RGBAColor = tuple[int, int, int, int]
ColorMap = dict[str | int, RGBAColor]


def _class_as_color(
    label: str, cls: int, color_map: ColorMap, alpha: int | None = None
) -> RGBAColor:
    color = color_map.get(label, color_map.get((cls), plot_utils.get_color(int(cls))))
    if alpha is not None:
        color = color[:3] + (alpha,)
    return color


def class_as_color(meta: AxTaskMeta, draw: display.Draw, class_id: int, alpha: int | None = None):
    labels = getattr(meta, 'labels', None)
    '''Labels may be enumerated, and therefore callable. Otherwise access via index.'''
    label = class_as_label(labels, class_id)
    return _class_as_color(label, class_id, draw.options.bbox_class_colors, alpha=alpha)


def _draw_bounding_box(
    box, score, cls, labels, draw, bbox_label_format, color_map, override_color
):
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    label = class_as_label(labels, cls)
    color = override_color or _class_as_color(label, int(cls), color_map)
    # An id less than zero is a manufactured box, so do not label
    if cls < 0:
        txt = ''
    else:
        txt = bbox_label_format.format(label=label, score=score, scorep=score * 100, scoreunit='%')
    draw.labelled_box(p1, p2, txt, color)


def _draw_oriented_bounding_box(
    box, score, cls, labels, draw, bbox_label_format, color_map, override_color
):
    if len(box) != 8 and len(box) != 5:
        raise ValueError(
            f"Oriented bounding box must have 5 (xywhr) or 8 (xyxyxyxy) coordinates, got {len(box)}"
        )

    if len(box) == 5:
        box = convert(box.reshape(1, 5), types.BoxFormat.XYWHR, types.BoxFormat.XYXYXYXY)

    box = box.reshape(4, 2).astype(int)
    label = class_as_label(labels, cls)
    color = override_color or _class_as_color(label, int(cls), color_map)
    # An id less than zero is a manufactured box, so do not label
    if cls < 0:
        txt = ''
    else:
        txt = bbox_label_format.format(label=label, score=score, scorep=score * 100, scoreunit='%')
    draw.labelled_polygon(box, txt, color)


_SHOW_OPTIONS = {(False, False): '', (False, True): '{score:.2f}', (True, False): '{label}'}


def draw_bounding_boxes(meta, draw, show_labels=True, show_annotations=True, override_color=None):
    """
    Draw bounding boxes on an image.

    Args:
        meta: The metadata containing bounding box information
        draw: The drawing context to use
        show_labels: Whether to draw class labels and scores (may be affected by environment variables)
        show_annotations: Whether to draw the bounding boxes themselves
    """
    from .. import config

    show_class = show_labels and config.env.render_bbox_class
    show_score = show_labels and config.env.render_bbox_score

    fmt = _SHOW_OPTIONS.get((show_class, show_score), draw.options.bbox_label_format)
    fmt = safe_label_format(fmt)
    labels = getattr(meta, 'labels', None)
    class_ids = getattr(meta, 'class_ids', itertools.repeat(0))
    color_map = draw.options.bbox_class_colors

    if show_annotations:
        for box, score, cls in zip(meta.boxes, meta.scores, class_ids):
            if len(box) == 4:
                _draw_bounding_box(box, score, cls, labels, draw, fmt, color_map, override_color)
            else:
                _draw_oriented_bounding_box(
                    box, score, cls, labels, draw, fmt, color_map, override_color
                )


class RestrictedDict(dict):
    def check_type(self, key, cls):
        if key in self and not isinstance(self[key], cls):
            raise Exception(
                f"An instance of {type(self[key]).__name__} already exists for key '{key}'"
            )

    def __setitem__(self, key, value):
        self.check_type(key, type(value))
        super().__setitem__(key, value)


T = TypeVar('T', bound='AxTaskMeta')


class MetaObject(abc.ABC):
    """
    Base class for the object-based view of the metadata. Acts as a
    view of the metadata to avoid copying data. Subclasses should
    implement properties to expose each object's fields using the
    metadata and index provided.

    Args:
        meta: The metadata containing meta for at least this object
        index: The index of this object in the metadata's fields.
    """

    __slots__ = ('_meta', '_index')

    def __init__(self, meta, index: int):
        self._meta = meta
        self._index = index

    def get_secondary_meta(self, task_name):
        """Get secondary metadata for a specific task

        Args:
            task_name: The name of the secondary task (e.g., 'classifier')

        Returns:
            The secondary metadata for this object and the specified task, or None if not found
        """
        meta = self._meta
        if not meta._secondary_metas or task_name not in meta._secondary_metas:
            return None

        # Check if this object's index is in the secondary_frame_indices for this task
        if task_name in meta.secondary_frame_indices:
            indices = meta.secondary_frame_indices[task_name]
            if self._index in indices:
                # Get the position in the indices list
                position = indices.index(self._index)
                # Use that position to get the corresponding secondary meta
                if position < len(meta._secondary_metas[task_name]):
                    return meta._secondary_metas[task_name][position]
        elif task_name in meta._secondary_metas:
            # If there are no explicit indices, we assume natural ordering
            # Return the metadata at position corresponding to this object's index
            if self._index < len(meta._secondary_metas[task_name]):
                return meta._secondary_metas[task_name][self._index]
        return None

    @property
    def secondary_meta(self):
        """Get secondary metadata for the first available secondary task (backward compatibility)

        Returns:
            The secondary metadata for this object, or None if not found
        """
        meta = self._meta
        if not hasattr(meta, '_secondary_metas') or not meta._secondary_metas:
            return None

        # Use the first available secondary task
        task_names = self.secondary_task_names
        if task_names:
            return self.get_secondary_meta(task_names[0])

        return None

    def get_secondary_objects(self, task_name):
        """Get secondary objects for a specific task

        Args:
            task_name: The name of the secondary task (e.g., 'classifier')

        Returns:
            A list of secondary objects, or an empty list if none found
        """
        sec_meta = self.get_secondary_meta(task_name)
        if sec_meta is None:
            return []

        # Check if the secondary meta has an objects property that returns MetaObject instances
        if hasattr(sec_meta, 'Object') and sec_meta.Object is not None:
            # Create MetaObject instances for this secondary meta
            return [sec_meta.Object(sec_meta, i) for i in range(len(sec_meta))]
        return []

    @property
    def secondary_objects(self):
        """Get secondary objects for the first available secondary task

        This is a convenience property for backward compatibility.
        For pipelines with multiple secondary tasks, use get_secondary_objects(task_name) instead.

        Returns:
            A list of secondary objects, or an empty list if none found
        """
        # Use the first available secondary task
        task_names = self.secondary_task_names
        if not task_names:
            return []

        # Use the get_secondary_objects method to ensure consistency
        return self.get_secondary_objects(task_names[0])

    @property
    def secondary_task_names(self):
        """Get the names of all secondary tasks for this object

        Returns:
            A list of task names that have secondary metadata for this object
        """
        meta = self._meta
        if not hasattr(meta, '_secondary_metas') or not meta._secondary_metas:
            return []

        # Return all task names for which this object has secondary metadata
        result = []
        for task_name in meta._secondary_metas.keys():
            if (
                hasattr(meta, 'secondary_frame_indices')
                and task_name in meta.secondary_frame_indices
            ):
                # If there are explicit indices, check if this object's index is in the indices
                indices = meta.secondary_frame_indices[task_name]
                if self._index in indices:
                    result.append(task_name)
            else:
                # For naturally ordered secondary metas (without explicit indices)
                # Check if the index is within the range of available secondary metas
                if self._index < len(meta._secondary_metas[task_name]):
                    result.append(task_name)

        return result

    @property
    @final
    def label(self):
        try:
            return self._meta.labels(self.class_id)
        except TypeError:
            raise NotImplementedError(
                f"{self.__class__.__name__}.label is not available for non-enum labels"
            )
        except AttributeError:
            raise AttributeError(
                f"'{self.__class__.__name__}'.label not available, no labels provided in metadata"
            )

    def __getattr__(self, attr):
        if attr.startswith("is_"):
            label = attr[3:]
            try:
                if not isinstance(self._meta.labels, utils.FrozenIntEnumMeta):
                    raise NotImplementedError(
                        f"{type(self).__name__}.{attr} is not available for non-enum labels"
                    )
            except AttributeError:
                raise AttributeError(
                    f"'{type(self).__name__}'.{attr} not available, no labels provided in metadata"
                )
            try:
                return self.label == getattr(self._meta.labels, label)
            except AttributeError:
                pass
        raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{attr}'")

    def __dir__(self):
        if hasattr(self._meta, 'labels') and isinstance(
            self._meta.labels, utils.FrozenIntEnumMeta
        ):
            return sorted(
                set(super().__dir__() + [f"is_{i}" for i in self._meta.labels.__members__.keys()])
            )
        return super().__dir__()

    def is_a(self, label_or_labels: str | tuple[str]) -> bool:
        '''Return True if this enum value is a given label or any of the given labels.

        labels may be a single item or a tuple, or passed as separate arguments.

        >>> obj.is_a('car')
        True
        >>> obj.is_a(('car', 'motorbike'))
        True
        '''
        labels = label_or_labels if isinstance(label_or_labels, tuple) else (label_or_labels,)
        return any(getattr(self, f'is_{i}') for i in labels)

    def __init_subclass__(cls) -> None:
        MetaObject._subclasses[cls.__name__] = cls
        return super().__init_subclass__()


MetaObject._subclasses = {}


@dataclasses.dataclass(frozen=True)
class AxBaseTaskMeta:
    secondary_frame_indices: dict[str, list[int]] = dataclasses.field(
        default_factory=dict, init=False
    )
    _secondary_metas: dict[str, list[AxBaseTaskMeta]] = dataclasses.field(
        default_factory=dict, init=False
    )
    container_meta: AxMeta | None = dataclasses.field(default=None, init=False)
    master_meta_name: str = dataclasses.field(default='', init=False)
    subframe_index: int | None = dataclasses.field(default=None, init=False)
    meta_name: str = dataclasses.field(default='', init=False)

    def __repr__(self):
        fields = []
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.name == 'container_meta' and value is not None:
                fields.append(f"{field.name}=<AxMeta: {value.image_id}>")
            else:
                fields.append(f"{field.name}={value!r}")
        return f"{self.__class__.__name__}({', '.join(fields)})"

    def members(self):
        """Return all member variable names"""
        return [f.name for f in dataclasses.fields(type(self))]

    def access_ground_truth(self):
        """Method to access the ground truth from the parent AxMeta"""
        if self.container_meta is None:
            raise ValueError("AxMeta is not set")
        if self.container_meta.ground_truth is None:
            raise ValueError("Ground truth is not set")
        return self.container_meta.ground_truth

    def access_image_id(self):
        """Method to access the image id from the parent AxMeta"""
        if self.container_meta is None:
            raise ValueError("AxMeta is not set")
        return self.container_meta.image_id

    def set_container_meta(self, container_meta: AxMeta):
        object.__setattr__(self, 'container_meta', container_meta)

    def set_master_meta(self, master_meta_name: str, subframe_index: int | None = None):
        object.__setattr__(self, 'master_meta_name', master_meta_name)
        object.__setattr__(self, 'subframe_index', subframe_index)

    def get_master_meta(self) -> AxTaskMeta:
        if self.container_meta is None:
            raise ValueError("Container meta is not set")
        return self.container_meta[self.master_meta_name]

    def add_secondary_meta(self, secondary_task_name: str, meta: 'AxBaseTaskMeta') -> None:
        """
        Add a secondary meta for a given secondary task name.
        Handles index assignment and ordering logic robustly.
        """
        if secondary_task_name not in self._secondary_metas:
            object.__setattr__(
                self, '_secondary_metas', {**self._secondary_metas, secondary_task_name: []}
            )
        assigned_metas = self._secondary_metas[secondary_task_name]
        assigned_indices = self.secondary_frame_indices.get(secondary_task_name, [])
        naturally_ordered = len(assigned_indices) == 0
        indices_available = len(assigned_indices) > len(assigned_metas)
        if not naturally_ordered and not indices_available:
            raise IndexError(
                f"No available secondary frame indices for task '{secondary_task_name}'. "
                f"Assigned metas: {len(assigned_metas)}, indices: {assigned_indices}"
            )
        assigned_metas.append(meta)

    def get_secondary_meta(self, secondary_task_name: str, index: int) -> 'AxBaseTaskMeta':
        """
        Retrieve a secondary meta by task name and index.
        Handles both naturally ordered and indexed cases robustly.
        """
        if secondary_task_name not in self._secondary_metas:
            raise KeyError(f"No secondary metas found for task: {secondary_task_name}")
        if not isinstance(index, int):
            raise TypeError("Index must be an integer")
        metas = self._secondary_metas[secondary_task_name]
        indices = self.secondary_frame_indices.get(secondary_task_name, [])
        naturally_ordered = len(indices) == 0
        if naturally_ordered:
            if not (0 <= index < len(metas)):
                raise IndexError(
                    f"Secondary frame index {index} out of range for task: {secondary_task_name}"
                )
            return metas[index]
        # Indexed case: find the position of the requested index
        try:
            meta_pos = indices.index(index)
        except ValueError:
            raise IndexError(
                f"Secondary frame index {index} not found for task: {secondary_task_name}"
            )
        if not (0 <= meta_pos < len(metas)):
            raise IndexError(
                f"Secondary frame index {index} out of range for task: {secondary_task_name}"
            )
        return metas[meta_pos]

    def add_secondary_frame_index(self, task_name: str, index: int) -> None:
        """
        Add a frame index for a secondary task, initializing if needed.
        """
        if task_name not in self.secondary_frame_indices:
            object.__setattr__(
                self, 'secondary_frame_indices', {**self.secondary_frame_indices, task_name: []}
            )
        self.secondary_frame_indices[task_name].append(index)

    def get_next_secondary_frame_index(self, task_name: str) -> int:
        """
        Get the next available secondary frame index for a task.
        Raises if none are available.
        """
        if task_name not in self.secondary_frame_indices:
            raise KeyError(f"No secondary frame indices found for task: {task_name}")
        indices = self.secondary_frame_indices[task_name]
        used = self.num_secondary_metas(task_name)
        if used >= len(indices):
            raise IndexError(f"No more secondary frame indices available for task: {task_name}")
        return indices[used]

    def num_secondary_metas(self, task_name: str) -> int:
        return len(self._secondary_metas.get(task_name, []))

    def get_secondary_task_names(self) -> list[str]:
        return list(self._secondary_metas.keys())

    def has_secondary_metas(self):
        return bool(self._secondary_metas)

    def visit(self, callable, *args, **kwargs):
        '''Call the callable on the current meta and all secondary metas'''
        callable(self, *args, **kwargs)
        for metas in self._secondary_metas.values():
            for meta in metas:
                try:
                    meta.visit(callable, *args, **kwargs)
                except exceptions.NotSupportedForTask:
                    pass

    @property
    def task_render_config(self):
        """Access render settings from the parent AxMeta"""
        if self.container_meta is None:
            raise ValueError("This meta is not part of a container AxMeta")
        if self.container_meta.render_config is None:
            raise ValueError("Render configuration is not set in the container AxMeta")
        return self.container_meta.render_config.get(self.meta_name, config.DEFAULT_RENDER_CONFIG)


@dataclasses.dataclass(frozen=True)
class AxTaskMeta(AxBaseTaskMeta):
    """Base metadata of a computer vision task"""

    Object: ClassVar[MetaObject] = None

    _objects: list[MetaObject] = dataclasses.field(default_factory=list, init=False)

    def draw(self, draw: display.Draw):
        """
        Draw the task metadata on an image.

        Args:
            draw (display.Draw): The drawing context to use.
            **kwargs: Additional keyword arguments for drawing.

        Note:
            Rendering options (e.g., show_labels, show_annotations) are accessed via self.task_render_config.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(f"Implement draw() for {self.__class__.__name__}")

    def to_evaluation(self) -> BaseEvalSample:
        """
        Convert the task metadata to a format suitable for evaluation.

        Returns:
            BaseEvalSample: The evaluation sample.

        Raises:
            ValueError: If ground truth is not set.
            NotImplementedError: If the subclass does not implement this method.
        """
        if not self.access_ground_truth():
            raise ValueError("Ground truth is not set")
        raise NotImplementedError(f"Implement to_evaluation() for {self.__class__.__name__}")

    @classmethod
    def aggregate(cls, meta_list: list[AxTaskMeta]) -> AxTaskMeta:
        """Aggregate a list of task meta objects into a single meta object.

        This is used to aggregate the secondary metas into one meta for measuring applicable
        accuracy of the last task.

        Args:
            meta_list (list[AxTaskMeta]): The task meta objects to aggregate.

        Returns:
            AxTaskMeta: A new aggregated task meta object.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
            EvaluationAggregationNotNeeded: If the subclass does not need aggregation.
        """
        raise NotImplementedError(f"Implement aggregate() for {cls.__name__}")

    @classmethod
    def decode(cls: Type[T], data: dict[str, bytes | bytearray]) -> T:
        """
        Decode raw byte data into task-specific metadata.

        This method should be implemented by subclasses to parse raw byte data
        received from C++ into the appropriate AxTaskMeta subclass instance.

        Args:
            data (dict[str, bytes | bytearray]): A dictionary containing raw byte data
                with task-specific keys.

        Returns:
            T: An instance of the task-specific AxTaskMeta subclass.

        Raises:
            NotImplementedError: If the subclass does not implement this method.
        """
        raise NotImplementedError(f"Implement decode() for {cls.__name__}")

    @property
    def objects(self) -> list[MetaObject]:
        """
        Get a list of MetaObject instances representing the task metadata.

        Returns:
            list[MetaObject]: A list of MetaObject instances.

        Raises:
            NotImplementedError: If the Object class variable is not set.
        """
        if not self.Object:
            raise NotImplementedError(f"Specify {self.__class__.__name__}.Object ")
        if not self._objects:
            self._objects.extend(self.Object(self, i) for i in range(len(self)))
        return self._objects

    def __init_subclass__(cls, **kwargs) -> None:
        super().__init_subclass__(**kwargs)
        AxTaskMeta._subclasses[cls.__name__] = cls
        meta_type = getattr(cls, 'META_TYPE', None)
        if meta_type:
            AxTaskMeta._subclasses[meta_type] = cls


AxTaskMeta._subclasses = {}


@dataclasses.dataclass
class AxMeta(collections.abc.Mapping):
    image_id: str
    attribute_meta: object | None = dataclasses.field(default=None, init=False)
    _meta_map: RestrictedDict = dataclasses.field(default_factory=RestrictedDict)
    extra_info: dict[str, Any] = dataclasses.field(default_factory=dict)
    # a pipeline has a single dataloader as input, so there will be a single ground truth
    # for the whole meta even if it is a multi-model network
    ground_truth: BaseEvalSample | None = dataclasses.field(default=None)
    render_config: config.RenderConfig | None = dataclasses.field(default=None)

    def __getitem__(self, key):
        if (val := self._meta_map.get(key)) is None:
            raise KeyError(f"{key} not found in meta_map.")
        return val

    def __len__(self) -> int:
        return len(self._meta_map)

    def __iter__(self) -> Iterator[str]:
        return iter(self._meta_map)

    def __setitem__(self, key, meta: AxTaskMeta):
        raise AttributeError(
            "Cannot set meta directly. Use add_instance or get_instance method to add or update"
        )

    def set_render_config(self, render_config: config.RenderConfig):
        """Set render configuration for the meta"""
        self.render_config = render_config

    def _add_secondary_meta(
        self,
        master_key: str,
        secondary_meta: 'AxBaseTaskMeta',
        secondary_task_name: str,
        subframe_index: int = -1,
    ) -> None:
        """
        Add a secondary meta to the master meta, handling index logic robustly.
        """
        if master_key not in self._meta_map:
            raise KeyError(
                f"master_key {master_key} for secondary_meta {secondary_meta} not found in meta. "
                f"Available keys: {list(self._meta_map.keys())}"
            )
        master_meta = self._meta_map[master_key]
        if not isinstance(master_meta, AxBaseTaskMeta):
            raise TypeError(f"Master meta {master_key} is not an instance of AxBaseTaskMeta")
        # Decide subframe index
        if subframe_index == -1:
            if master_meta.secondary_frame_indices.get(secondary_task_name):
                subframe_index = master_meta.get_next_secondary_frame_index(secondary_task_name)
            else:
                subframe_index = master_meta.num_secondary_metas(secondary_task_name)
        else:
            master_meta.add_secondary_frame_index(secondary_task_name, subframe_index)
        secondary_meta.set_container_meta(self)
        secondary_meta.set_master_meta(master_key, subframe_index)
        master_meta.add_secondary_meta(secondary_task_name, secondary_meta)

    def add_instance(self, key, instance, master_meta_name='', subframe_index=-1):
        self._meta_map.check_type(key, type(instance))
        if isinstance(instance, AxTaskMeta):
            instance.set_container_meta(self)
            object.__setattr__(instance, 'meta_name', key)
            if not master_meta_name:
                if key in self._meta_map:
                    raise ValueError(f"Master meta {key} already exists")
                self._meta_map[key] = instance
            else:
                self._add_secondary_meta(master_meta_name, instance, key, subframe_index)
        else:
            self._meta_map[key] = instance

    def delete_instance(self, key):
        try:
            del self._meta_map[key]
        except KeyError:
            LOG.warning(f"Attempted to delete non-existent key '{key}' from meta_map")

    def get_instance(self, key, cls, *args, master_meta_name='', **kwargs):
        """Get an instance of a model from meta_map.

        If the key is not found, a new instance will be created by using the provided keyword
        arguments and stored in meta_map.
        """
        self._meta_map.check_type(key, cls)
        if key not in self._meta_map:
            if hasattr(cls, "create_immutable_meta"):
                instance = cls.create_immutable_meta(*args, **kwargs)
            else:
                instance = cls(*args, **kwargs)
            self.add_instance(key, instance, master_meta_name)
        return self._meta_map[key]

    def inject_groundtruth(self, ground_truth: BaseEvalSample):
        """Inject ground truth into the meta"""
        if self.ground_truth is not None:
            raise ValueError("Ground truth is already set")
        self.ground_truth = ground_truth

    def aggregate_leaf_metas(self, master_key: str, secondary_task_name: str) -> list[AxTaskMeta]:
        """Aggregate secondary metas into one meta.

        This is used to aggregate the secondary metas into one meta for measuring applicable
        accuracy of the last task.
        """
        if master_key not in self._meta_map:
            raise KeyError(f"{master_key} not found in meta_map.")

        def collect_leaf_metas(root_meta):
            stack = [root_meta]
            leaf_metas = []

            while stack:
                meta = stack.pop()
                if not isinstance(meta, AxTaskMeta) or not meta.has_secondary_metas():
                    leaf_metas.append(meta)
                else:
                    # Assume there's only one secondary task, so we can just get the first (and only) value
                    if len(meta._secondary_metas) > 1:
                        raise ValueError(
                            f"Multiple secondary tasks found for meta {meta}. Using the first one."
                        )
                    secondary_metas = next(iter(meta._secondary_metas.values()))
                    stack.extend(secondary_metas[::-1])  # Add secondary metas in reverse order

            return leaf_metas

        meta = self._meta_map[master_key]
        if isinstance(meta, AxTaskMeta):
            if meta.num_secondary_metas(secondary_task_name) > 0:
                if leaf_metas := collect_leaf_metas(meta):
                    try:
                        aggregated_meta = type(leaf_metas[0]).aggregate(leaf_metas)
                    except AggregationNotRequiredForEvaluation:
                        return leaf_metas
                    aggregated_meta.set_container_meta(self)
                    return [aggregated_meta]
            else:
                raise NoMasterDetectionsError(master_key)

        raise ValueError(
            f"Master meta {master_key} is not an instance of AxTaskMeta, but {type(meta)}"
        )

# Copyright Axelera AI, 2024
# Base dataclasses used to represent metadata
from __future__ import annotations

import abc
import collections
import dataclasses
import functools
import itertools
from typing import TYPE_CHECKING, Any, ClassVar, Iterator, Type, TypeVar, final

from .. import display, logging_utils, plot_utils, utils

if TYPE_CHECKING:
    from .. import display
    from ..eval_interfaces import BaseEvalSample

LOG = logging_utils.getLogger(__name__)
_VALID_LABEL_FORMATS = {'label': 'label', 'score': 0.56, 'scorep': 56.7}
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
    try:
        return labels(class_id).name
    except TypeError:
        return labels[class_id]
    except ValueError:
        pass
    return str(class_id)


@functools.lru_cache(maxsize=200)
def _safe_label_format(fmt: str) -> str:
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


def _draw_bounding_box(box, score, cls, labels, draw, bbox_label_format, color_map):
    p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
    label = class_as_label(labels, cls)
    color = _class_as_color(label, int(cls), color_map)
    txt = bbox_label_format.format(label=label, score=score, scorep=score * 100)
    draw.labelled_box(p1, p2, txt, color)


_SHOW_OPTIONS = {(False, False): '', (False, True): '{score:.2f}', (True, False): '{label}'}


def draw_bounding_boxes(meta, draw, show_class=True, show_score=True):
    fmt = _SHOW_OPTIONS.get((show_class, show_score), draw.options.bbox_label_format)
    fmt = _safe_label_format(fmt)
    labels = getattr(meta, 'labels', None)
    class_ids = getattr(meta, 'class_ids', itertools.repeat(0))
    color_map = draw.options.bbox_class_colors
    for box, score, cls in zip(meta.boxes, meta.scores, class_ids):
        _draw_bounding_box(box, score, cls, labels, draw, fmt, color_map)


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

    def __init__(self, meta, index):
        self._meta = meta
        self._index = index

    @property
    def secondary_meta(self):
        return self._meta.get_secondary_meta(self._index)

    @property
    def secondary_objects(self):
        return self.secondary_meta.objects

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
                set(super().__dir__() + [f"is_{l}" for l in self._meta.labels.__members__.keys()])
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
        return any(getattr(self, f'is_{l}') for l in labels)

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

    def add_secondary_meta(self, secondary_task_name: str, meta: AxBaseTaskMeta):
        if secondary_task_name not in self._secondary_metas:
            object.__setattr__(
                self, '_secondary_metas', {**self._secondary_metas, secondary_task_name: []}
            )
        self._secondary_metas[secondary_task_name].append(meta)

    def get_secondary_meta(self, secondary_task_name: str, index: int) -> AxBaseTaskMeta:
        if secondary_task_name not in self._secondary_metas:
            raise KeyError(f"No secondary metas found for task: {secondary_task_name}")
        if not isinstance(index, int):
            raise TypeError("Index must be an integer")
        if index < 0 or index >= len(self._secondary_metas[secondary_task_name]):
            raise IndexError("Submeta index out of range")
        return self._secondary_metas[secondary_task_name][index]

    def add_secondary_frame_index(self, task_name: str, index: int):
        if task_name not in self.secondary_frame_indices:
            object.__setattr__(
                self, 'secondary_frame_indices', {**self.secondary_frame_indices, task_name: []}
            )
        self.secondary_frame_indices[task_name].append(index)

    def get_next_secondary_frame_index(self, task_name: str) -> int:
        """
        Get the next secondary frame index to be used when adding a new secondary meta.

        Returns:
            int: The next secondary frame index.

        Raises:
            IndexError: If there are no more secondary frame indices available.
        """
        if task_name not in self.secondary_frame_indices:
            raise KeyError(f"No secondary frame indices found for task: {task_name}")

        task_indices = self.secondary_frame_indices[task_name]
        if len(task_indices) <= self.num_secondary_metas(task_name):
            raise IndexError(f"No more secondary frame indices available for task: {task_name}")
        return task_indices[self.num_secondary_metas(task_name)]

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
                meta.visit(callable, *args, **kwargs)


@dataclasses.dataclass(frozen=True)
class AxTaskMeta(AxBaseTaskMeta):
    """Base metadata of a computer vision task"""

    Object: ClassVar[MetaObject] = None

    _objects: list[MetaObject] = dataclasses.field(default_factory=list, init=False)

    def draw(self, draw: display.Draw, **kwargs):
        """
        Draw the task metadata on an image.

        Args:
            draw (display.Draw): The drawing context to use.
            **kwargs: Additional keyword arguments for drawing.

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
        if not (ground_truth := self.access_ground_truth()):
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

    def __init_subclass__(cls) -> None:
        AxTaskMeta._subclasses[cls.__name__] = cls
        return super().__init_subclass__()


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

    def _add_secondary_meta(
        self, master_key: str, secondary_meta: AxBaseTaskMeta, secondary_task_name: str
    ):
        if master_key not in self._meta_map:
            raise KeyError(
                f"master_key {master_key} for secondary_meta {secondary_meta} not found in meta"
                f"which contains the following keys: {self._meta_map.keys()}"
            )

        master_meta = self._meta_map[master_key]
        if not isinstance(master_meta, AxBaseTaskMeta):
            raise TypeError(f"Master meta {master_key} is not an instance of AxBaseTaskMeta")

        if len(master_meta.secondary_frame_indices) > 0:
            subframe_index = master_meta.get_next_secondary_frame_index(secondary_task_name)
        else:
            subframe_index = master_meta.num_secondary_metas(secondary_task_name)
        secondary_meta.set_container_meta(self)
        secondary_meta.set_master_meta(master_key, subframe_index)
        master_meta.add_secondary_meta(secondary_task_name, secondary_meta)

    def add_instance(self, key, instance, master_meta_name=''):
        self._meta_map.check_type(key, type(instance))
        if isinstance(instance, AxTaskMeta):
            instance.set_container_meta(self)
            if not master_meta_name:
                if key in self._meta_map:
                    raise ValueError(f"Master meta {key} already exists")
                self._meta_map[key] = instance
            else:
                self._add_secondary_meta(master_meta_name, instance, key)
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

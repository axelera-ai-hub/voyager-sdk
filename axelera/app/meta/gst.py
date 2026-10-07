# Copyright Axelera AI, 2023
from __future__ import annotations

import ctypes
import dataclasses
import importlib
from typing import Any, Callable, Optional
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=ImportWarning, module="importlib")


from .. import logging_utils  # noqa : E402 not at top of file. But we must ignore the warnings
from .base import AxTaskMeta  # noqa : E402 not at top of file. But we must ignore the warnings
from .gst_decode_utils import (  # noqa : E402 not at top of file. But we must ignore the warnings
    decode_bbox,
)

try:
    from gi.repository import Gst
except ModuleNotFoundError:
    pass

LOG = logging_utils.getLogger(__name__)


class ModelInfoProvider:
    """Lightweight access layer for model metadata required during GST assembly."""

    def __init__(
        self,
        labels_dict: dict[str, Any] | None = None,
        num_classes_dict: dict[str, int] | None = None,
        softmax_lookup: Callable[[str], bool] | None = None,
    ) -> None:
        self._labels_dict = labels_dict or {}
        self._num_classes_dict = num_classes_dict or {}
        self._softmax_lookup = softmax_lookup or (lambda _task: False)

    def labels_for(self, task_name: str) -> Any:
        return self._labels_dict.get(task_name)

    def num_classes_for(self, task_name: str) -> int:
        return self._num_classes_dict.get(task_name, 0)

    def has_softmax(self, task_name: str) -> bool:
        try:
            return bool(self._softmax_lookup(task_name))
        except Exception as exc:  # pragma: no cover - defensive logging path
            LOG.debug("Softmax lookup failed for %s: %s", task_name, exc)
            return False

    @property
    def labels_dict(self) -> dict[str, Any]:
        return self._labels_dict


class GstMetaAssembler:
    """Helper that builds AxMeta structures directly from GST decoded metadata."""

    def __init__(self, model_info: ModelInfoProvider) -> None:
        self._model_info = model_info
        self._meta_module = importlib.import_module('axelera.app.meta')
        self._seen_types: set[type] = set()
        self._normalisers: dict[type, Callable[[Any, str], AxTaskMeta]] = {}
        self._register_default_normalisers()

    def process(
        self,
        ax_meta,
        decoded_meta,
        task_graph=None,
        result_view=None,
    ) -> None:
        for gst_meta_info, task_meta in (decoded_meta or {}).items():
            expected_master = (
                task_graph.get_master(gst_meta_info.task_name, view=result_view)
                if task_graph is not None and result_view is not None
                else None
            )
            actual_master = gst_meta_info.master

            if expected_master and actual_master and actual_master != expected_master:
                raise ValueError(
                    "From the YAML, the master of %s is expected to be %s, "
                    "but GST is sending it as %s."
                    % (gst_meta_info.task_name, expected_master, actual_master)
                )
            if expected_master and not actual_master:
                LOG.warning(
                    "GST is treating %s as a master meta, but from the YAML it is expected to "
                    "be a submeta of %s.",
                    gst_meta_info.task_name,
                    expected_master,
                )
            if not expected_master and actual_master:
                LOG.warning(
                    "GST is treating %s as a submeta of %s, but from the YAML it is expected "
                    "to be a master meta.",
                    gst_meta_info.task_name,
                    actual_master,
                )

            self.add_meta(ax_meta, gst_meta_info, task_meta, actual_master)

    def add_meta(self, ax_meta, gst_meta_info, task_meta, master_meta_name: Optional[str]) -> None:
        instance = self._create_meta_instance(gst_meta_info.task_name, task_meta)
        subframe_index = (
            gst_meta_info.subframe_index if gst_meta_info.subframe_index is not None else -1
        )
        ax_meta.add_instance(
            gst_meta_info.task_name,
            instance,
            master_meta_name or '',
            subframe_index,
        )

        tracker_cls = getattr(self._meta_module, 'TrackerMeta', None)
        if tracker_cls and isinstance(instance, tracker_cls):
            self._attach_tracker_cascades(ax_meta, gst_meta_info, instance)

    def _create_meta_instance(self, task_name: str, task_meta):
        mm = self._meta_module

        if normaliser := self._normalisers.get(type(task_meta)):
            return normaliser(task_meta, task_name)

        if isinstance(
            task_meta,
            (
                mm.CocoBodyKeypointsMeta,
                mm.FaceLandmarkLocalizationMeta,
                mm.FaceLandmarkTopDownMeta,
                mm.SemanticSegmentationMeta,
                mm.LicensePlateMeta,
                mm.ImageMeta,
            ),
        ):
            return task_meta

        meta_type = type(task_meta)
        if meta_type not in self._seen_types:
            self._seen_types.add(meta_type)
            LOG.debug("Directly registering %s into ax_meta", meta_type)
        return task_meta

    def _register_default_normalisers(self) -> None:
        mm = self._meta_module
        self._normalisers = {
            mm.ClassificationMeta: self._normalise_classification,
            mm.ObjectDetectionMeta: self._normalise_detection,
            mm.ObjectDetectionMetaOBB: self._normalise_detection,
            mm.TrackerMeta: self._normalise_tracker,
            mm.InstanceSegmentationMeta: self._normalise_instance_segmentation,
            mm.PoseInsSegMeta: self._normalise_pose_instance_segmentation,
        }

    def _attach_tracker_cascades(self, ax_meta, tracker_info: GstMetaInfo, tracker_meta):
        # Merge cascades: object_meta (aggregated) takes precedence over frame_object_meta (current frame)
        cascades_by_task = {}
        for source in (
            getattr(tracker_meta, 'frame_object_meta', {}) or {},
            getattr(tracker_meta, 'object_meta', {}) or {},
        ):
            for task_name, track_map in source.items():
                if isinstance(track_map, dict):
                    cascades_by_task.setdefault(task_name, {}).update(track_map)

        for task_name, track_map in cascades_by_task.items():
            for track_id, raw_meta in track_map.items():
                if raw_meta is None:
                    continue
                try:
                    subframe_index = int(track_id)
                except (TypeError, ValueError):
                    subframe_index = track_id

                child_meta = self._create_meta_instance(task_name, raw_meta)
                tracker_meta.add_secondary_frame_index(task_name, subframe_index)

                child_meta.set_container_meta(ax_meta)
                child_meta.set_master_meta(tracker_info.task_name, subframe_index)
                tracker_meta.add_secondary_meta(task_name, child_meta)

    def _normalise_classification(self, meta_obj, task_name: str):
        labels = self._model_info.labels_for(task_name)
        num_classes_hint = self._model_info.num_classes_for(task_name)
        softmax = self._model_info.has_softmax(task_name)

        updated_labels = meta_obj.labels if labels is None else labels
        updated_num_classes = meta_obj.num_classes
        if num_classes_hint:
            updated_num_classes = num_classes_hint

        extra_info = meta_obj.extra_info
        if (
            extra_info.get('softmax') == softmax
            and labels is None
            and num_classes_hint in (None, meta_obj.num_classes)
        ):
            return meta_obj

        extra_info = meta_obj.extra_info
        if extra_info.get('softmax') != softmax:
            extra_info = dict(extra_info)
            extra_info['softmax'] = softmax

        mm = self._meta_module
        clone = mm.ClassificationMeta(
            labels=updated_labels,
            num_classes=updated_num_classes,
            extra_info=extra_info,
        )
        clone.transfer_data(meta_obj)
        return clone

    def _normalise_detection(self, meta_obj, task_name: str):
        labels = self._model_info.labels_for(task_name)
        if labels is None or getattr(meta_obj, 'labels', None) == labels:
            return meta_obj
        return dataclasses.replace(meta_obj, labels=labels)

    def _normalise_tracker(self, meta_obj, task_name: str):
        labels = self._model_info.labels_for(task_name)
        if (
            getattr(meta_obj, 'labels', None) == labels
            and meta_obj.labels_dict == self._model_info.labels_dict
        ):
            return meta_obj
        return dataclasses.replace(
            meta_obj,
            labels=labels,
            labels_dict=self._model_info.labels_dict,
        )

    def _normalise_instance_segmentation(self, meta_obj, task_name: str):
        mm = self._meta_module
        labels = self._model_info.labels_for(task_name)
        if labels is None or labels == getattr(meta_obj, 'labels', None):
            return meta_obj
        clone = mm.InstanceSegmentationMeta(labels=labels)
        clone.transfer_data(meta_obj)
        return clone

    def _normalise_pose_instance_segmentation(self, meta_obj, task_name: str):
        mm = self._meta_module
        labels = self._model_info.labels_for(task_name)
        if labels is None or labels == getattr(meta_obj, 'labels', None):
            return meta_obj
        clone = mm.PoseInsSegMeta(labels=labels)
        clone.transfer_data(meta_obj)
        return clone


def decode_landmarks(data):
    """
    Landmarks hold a dictionary with a single element 'facial_landmarks' which is a 1D array of floats
    The array is of size num_entries * num_landmark_points * 2 (x,y)
    """
    point_size = 2
    num_landmarks = 68
    landmarks = data.get("facial_landmarks", b"")
    boxes3d = np.frombuffer(landmarks, dtype=np.float32).reshape(-1, num_landmarks, point_size)
    return boxes3d


def _decode_single(data, field, dtype):
    bin = data.get(field)
    if bin is None:
        raise RuntimeError(f"Expecting {field} meta element")
    sz = dtype(0).itemsize
    if not isinstance(bin, bytes) or len(bin) != sz:
        raise RuntimeError(f"Expecting {field} to be a byte stream {sz} bytes long")
    return np.frombuffer(bin, dtype=dtype)[0]


def decode_stream_meta(data):
    stream_id = int(_decode_single(data, "stream_id", np.int32))
    ts = int(_decode_single(data, "timestamp", np.uint64)) / 1000000000
    inferences = int(_decode_single(data, "inferences", np.int32))
    return stream_id, ts, inferences


def load_meta_dll(meta_dll: str = "libgstaxstreamer.so"):
    return ctypes.CDLL(meta_dll)


class c_meta(ctypes.Structure):
    _fields_ = [
        ("type", ctypes.c_char_p),
        ("subtype", ctypes.c_char_p),
        ("size", ctypes.c_int),
        ("data", ctypes.POINTER(ctypes.c_char)),
    ]


class c_named_meta(ctypes.Structure):
    _fields_ = [("name", ctypes.c_char_p), ("meta", c_meta)]


class c_all_meta(ctypes.Structure):
    _fields_ = [("num_meta", ctypes.c_int), ("meta", ctypes.POINTER(c_named_meta))]


class c_extern_indexed_submeta(ctypes.Structure):
    _fields_ = [
        ("meta_vector", ctypes.POINTER(c_meta)),
        ("subframe_index", ctypes.c_int),
        ("num_extern_meta", ctypes.c_int),
    ]


class c_extern_named_submeta_collection(ctypes.Structure):
    _fields_ = [
        ("name_submodel", ctypes.c_char_p),
        ("meta_indices", ctypes.POINTER(c_extern_indexed_submeta)),
        ("subframe_number", ctypes.c_int),
    ]


class c_all_extern_submeta(ctypes.Structure):
    _fields_ = [
        ("name_model", ctypes.c_char_p),
        ("num_submeta", ctypes.c_int),
        ("meta_submodels", ctypes.POINTER(c_extern_named_submeta_collection)),
    ]


class c_all_extern_submeta_all_models(ctypes.Structure):
    _fields_ = [("num_meta", ctypes.c_int), ("meta_models", ctypes.POINTER(c_all_extern_submeta))]


@dataclasses.dataclass(frozen=True)
class GstMetaInfo:
    '''
    Hashable and subscriptable class to represent a GST meta type
    '''

    task_name: str
    meta_type: str
    subframe_index: Optional[int] = None
    master: Optional[str] = None

    def __hash__(self):
        return hash((self.task_name, self.meta_type, self.subframe_index, self.master))

    def __eq__(self, other):
        if not isinstance(other, GstMetaInfo):
            return NotImplemented
        return (
            self.task_name,
            self.meta_type,
            self.subframe_index,
            self.master,
        ) == (
            other.task_name,
            other.meta_type,
            other.subframe_index,
            other.master,
        )

    def __getitem__(self, index):
        return (self.task_name, self.meta_type, self.subframe_index)[index]


class GstDecoder:
    def __init__(self):
        self.libmeta = load_meta_dll()
        self.libmeta.get_meta_from_buffer.argtypes = [ctypes.c_void_p]
        self.libmeta.get_meta_from_buffer.restype = c_all_meta

        self.libmeta.free_meta.argtypes = [ctypes.POINTER(c_named_meta)]
        self.libmeta.free_meta.restype = None

        self.libmeta.get_submeta_from_buffer.argtypes = [ctypes.c_void_p]
        self.libmeta.get_submeta_from_buffer.restype = c_all_extern_submeta_all_models

        self.libmeta.free_submeta.argtypes = [ctypes.POINTER(c_all_extern_submeta_all_models)]
        self.libmeta.free_submeta.restype = None

        self.decoders = {
            "bbox": decode_bbox,
            "landmarks": decode_landmarks,
            "stream_meta": decode_stream_meta,
        }
        self.meta_module = importlib.import_module('axelera.app.meta')

    def register_decoder(self, name, decoder):
        """
        Register a decoder for a specific meta type
        """
        self.decoders[name] = decoder

    def extract_all_meta(self, buffer: Gst.Buffer):
        """
        Extract and decode all ax meta data from a Gst.Buffer
        """
        results: dict[GstMetaInfo, Any] = {}
        all_meta = self.libmeta.get_meta_from_buffer(hash(buffer))
        all_submeta = self.libmeta.get_submeta_from_buffer(hash(buffer))
        try:
            meta = all_meta.meta
            for i in range(all_meta.num_meta):
                name = str(meta[i].name, encoding="utf8")
                meta_type = str(meta[i].meta.type, encoding="utf8")
                key = GstMetaInfo(name, meta_type)
                subtype = str(meta[i].meta.subtype, encoding="utf8")
                val = ctypes.string_at(meta[i].meta.data, meta[i].meta.size)
                entry = results.get(key, {})
                if subtype in entry:
                    val = entry[subtype] + val
                entry[subtype] = val
                results[key] = entry
            for i in range(all_submeta.num_meta):
                master_task_name = str(all_submeta.meta_models[i].name_model, encoding="utf8")
                for j in range(all_submeta.meta_models[i].num_submeta):
                    submodel_coll = all_submeta.meta_models[i].meta_submodels[j]
                    subtask_name = str(submodel_coll.name_submodel, encoding="utf8")
                    for k in range(submodel_coll.subframe_number):
                        submeta = submodel_coll.meta_indices[k]
                        subframe_index = submeta.subframe_index
                        for m in range(submeta.num_extern_meta):
                            meta_entry = submeta.meta_vector[m]
                            meta_type = str(meta_entry.type, encoding="utf8")
                            subtype = str(meta_entry.subtype, encoding="utf8")
                            key = GstMetaInfo(
                                subtask_name, meta_type, subframe_index, master_task_name
                            )
                            val = ctypes.string_at(meta_entry.data, meta_entry.size)
                            entry = results.get(key, {})
                            if subtype in entry:
                                val = entry[subtype] + val
                            entry[subtype] = val
                            results[key] = entry
            return self.decode(results)
        finally:
            self.libmeta.free_meta(all_meta.meta)
            self.libmeta.free_submeta(all_submeta)

    def decode(self, meta):
        """
        Decode all meta data using the registered decoders
        If no decoder for a type exists the raw buffers are retained
        """
        decoded_meta = {}
        for key, data in meta.items():
            meta_type = key[1]
            if meta_class := AxTaskMeta._subclasses.get(meta_type):
                if issubclass(meta_class, AxTaskMeta):
                    decoded_meta[key] = meta_class.decode(data)
                else:
                    raise NotImplementedError
            else:
                # TODO: for general meta types we should always use the decode method in AxTaskMeta
                decoder = self.decoders.get(meta_type, None)
                if decoder is not None:
                    decoded_meta[key] = decoder(data)
                else:
                    import traceback

                    traceback.print_exc()
                    raise RuntimeError(f"No decoder for {meta_type}")
        return decoded_meta

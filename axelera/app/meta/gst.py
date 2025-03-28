# Copyright Axelera AI, 2023
import ctypes
import dataclasses
import importlib
from typing import Any, Optional
import warnings

import gi
import numpy as np

warnings.filterwarnings("ignore", category=ImportWarning, module="importlib")
gi.require_version("Gst", "1.0")
gi.require_version("GstApp", "1.0")
from gi.repository import Gst

from .. import logging_utils
from .base import AxTaskMeta
from .gst_decode_utils import decode_bbox
from .tracker import TrackerMeta

LOG = logging_utils.getLogger(__name__)


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


def decode_tracking(data):
    meta_module = importlib.import_module('axelera.app.meta')
    tracking_history = {}
    class_ids = []
    frame_object_meta: dict[str, list[AxTaskMeta]] = {}
    object_meta: dict[str, list[AxTaskMeta]] = {}
    objmeta_key_to_string = dict()
    for key in data.keys():
        if key == 'objmeta_keys':
            key_data = data[key]
            key_data_size = len(key_data)
            while key_data_size > 0:
                objmeta_key = np.frombuffer(key_data[:1], dtype=np.uint8)[0]
                objmeta_string_size = int(np.frombuffer(key_data[1:9], dtype=np.uint64)[0])
                objmeta_string = key_data[9 : 9 + objmeta_string_size].decode("utf-8")
                objmeta_key_to_string[objmeta_key] = objmeta_string
                key_data = key_data[9 + objmeta_string_size :]
                key_data_size = len(key_data)
    for key in data.keys():
        if key.startswith('track_'):
            track_id = int(key[6:])
            track_data = data[key]
            class_ids.append(np.frombuffer(track_data[:4], dtype=np.int32)[0])
            num_boxes = np.frombuffer(track_data[4:8], dtype=np.int32)[0]
            bbox_data = {"bbox": track_data[8 : num_boxes * 16 + 8]}
            tracking_history[track_id] = decode_bbox(bbox_data)
            ind_track_data = num_boxes * 16 + 8
            ind_track_data = process_object_metadata(
                meta_module, frame_object_meta, objmeta_key_to_string, track_data, ind_track_data
            )
            ind_track_data = process_object_metadata(
                meta_module, object_meta, objmeta_key_to_string, track_data, ind_track_data
            )

    return TrackerMeta(
        tracking_history=tracking_history,
        class_ids=class_ids,
        object_meta=object_meta,
        frame_object_meta=frame_object_meta,
    )


def process_object_metadata(
    meta_module, object_meta, objmeta_key_to_string, track_data, ind_track_data
):
    results_objmeta: dict[GstMetaInfo, Any] = {}
    num_objmeta = np.frombuffer(track_data[ind_track_data : ind_track_data + 4], dtype=np.int32)[0]
    ind_track_data += 4
    for i in range(num_objmeta):
        objmeta_key = np.frombuffer(
            track_data[ind_track_data : ind_track_data + 1], dtype=np.uint8
        )[0]
        ind_track_data += 1
        objmeta_string = objmeta_key_to_string.get(objmeta_key, f"key_{objmeta_key}")
        metavec_size = np.frombuffer(
            track_data[ind_track_data : ind_track_data + 4], dtype=np.int32
        )[0]
        ind_track_data += 4
        for j in range(metavec_size):
            objmeta_type_stringsize = np.frombuffer(
                track_data[ind_track_data : ind_track_data + 4], dtype=np.int32
            )[0]
            ind_track_data += 4
            objmeta_type_string = track_data[
                ind_track_data : ind_track_data + objmeta_type_stringsize
            ].decode("utf-8")
            ind_track_data += objmeta_type_stringsize
            objmeta_subtype_stringsize = np.frombuffer(
                track_data[ind_track_data : ind_track_data + 4], dtype=np.int32
            )[0]
            ind_track_data += 4
            objmeta_subtype_string = track_data[
                ind_track_data : ind_track_data + objmeta_subtype_stringsize
            ].decode("utf-8")
            ind_track_data += objmeta_subtype_stringsize
            objmeta_size = np.frombuffer(
                track_data[ind_track_data : ind_track_data + 4], dtype=np.int32
            )[0]
            ind_track_data += 4
            objmeta_data = track_data[ind_track_data : ind_track_data + objmeta_size]
            ind_track_data += objmeta_size
            if objmeta_size == 0:
                continue

            results_key = GstMetaInfo(objmeta_string, objmeta_type_string)
            entry = results_objmeta.get(results_key, {})
            if objmeta_subtype_string in entry:
                objmeta_data = entry[objmeta_subtype_string] + objmeta_data
            entry[objmeta_subtype_string] = objmeta_data
            results_objmeta[results_key] = entry

    for results_key, results_data in results_objmeta.items():
        meta_type = results_key[1]
        meta_class = getattr(meta_module, meta_type)
        object_meta.setdefault(results_key[0], []).append(meta_class.decode(results_data))

    return ind_track_data


def decode_embeddings(data):
    buffer = data.get("data", b"")
    num_embeddings = int.from_bytes(buffer[:4], byteorder='little')
    embeddings = np.frombuffer(buffer[4:], dtype=np.float32)
    # Reshape to (num_embeddings, M) where M is the vector length of each embedding
    return embeddings.reshape(num_embeddings, -1)


class timespec(ctypes.Structure):
    _fields_ = [
        ("tv_sec", ctypes.c_long),
        ("tv_nsec", ctypes.c_long),
    ]


def decode_stream_meta(data):
    stream_id_binary = data.get("stream_id")
    if stream_id_binary is None:
        raise RuntimeError("Expecting stream_id meta element")
    sz = np.int32(0).itemsize
    if not isinstance(stream_id_binary, bytes) or len(stream_id_binary) != sz:
        raise RuntimeError(f"Expecting stream_id to be a byte stream {sz} bytes long")
    stream_id = int(np.frombuffer(stream_id_binary, dtype=np.int32)[0])
    ts = data.get("timestamp")
    if ts is None:
        raise RuntimeError("Expecting timestamp meta element")
    sz = ctypes.sizeof(timespec)
    if not isinstance(ts, bytes) or len(ts) != sz:
        raise RuntimeError(f"Expecting timestamp to be a byte stream {sz} bytes long")
    ts = timespec.from_buffer_copy(ts)
    ts = ts.tv_sec + ts.tv_nsec / 1000000000
    return stream_id, ts


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
        return (self.task_name, self.meta_type, self.subframe_index, self.master,) == (
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
            "tracking_meta": decode_tracking,
            "stream_meta": decode_stream_meta,
            "embeddings": decode_embeddings,
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
                        for l in range(submeta.num_extern_meta):
                            meta_entry = submeta.meta_vector[l]
                            meta_type = str(meta_entry.type, encoding="utf8")
                            subtype = str(meta_entry.subtype, encoding="utf8")
                            key = GstMetaInfo(
                                subtask_name, meta_type, subframe_index, master_task_name
                            )
                            if master_task_name == subtask_name:
                                # The tracker stores detections as submeta
                                # But here they are not required any more
                                continue
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

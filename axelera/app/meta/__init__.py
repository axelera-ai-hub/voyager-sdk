# Copyright Axelera AI, 2025
# Metadata for streaming pipeline

# Import all meta modules so that subclasses are known about
from . import classification, keypoint, object_detection, pair_validation, segmentation, tracker
from .base import AxMeta, AxTaskMeta, MetaObject, NoMasterDetectionsError
from .bbox_state import BBoxState
from .gst import GstDecoder, GstMetaInfo
from .registry import MetaRegistry

__all__ = [
    'AxMeta',
    'AxTaskMeta',
    'BBoxState',
    'GstDecoder',
    'GstMetaInfo',
    'NoMasterDetectionsError',
    'MetaRegistry',
]
__all__.extend(AxTaskMeta._subclasses.keys())
__all__.extend(MetaObject._subclasses.keys())

# Add the imported classes to the module's global namespace
globals().update(AxTaskMeta._subclasses)
globals().update(MetaObject._subclasses)

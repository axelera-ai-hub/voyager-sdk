# Copyright Axelera AI, 2023
# Metadata for streaming pipeline

# Import all meta modules so that subclasses are known about
from . import (
    classification,
    image,
    keypoint,
    licenseplate,
    multihead,
    object_detection,
    object_detection_obb,
    pair_validation,
    segmentation,
    tensor,
    tracker,
)
from .base import AxMeta, AxTaskMeta, MetaObject, NoMasterDetectionsError
from .bbox_state import BBoxState
from .gst import GstDecoder, GstMetaAssembler, GstMetaInfo, ModelInfoProvider
from .registry import MetaRegistry

__all__ = [
    'AxMeta',
    'AxTaskMeta',
    'BBoxState',
    'GstDecoder',
    'GstMetaInfo',
    'GstMetaAssembler',
    'ModelInfoProvider',
    'NoMasterDetectionsError',
    'MetaRegistry',
    'classification',
    'image',
    'keypoint',
    'licenseplate',
    'multihead',
    'object_detection',
    'object_detection_obb',
    'pair_validation',
    'segmentation',
    'tensor',
    'tracker',
]
__all__.extend(AxTaskMeta._subclasses.keys())
__all__.extend(MetaObject._subclasses.keys())

# Add the imported classes to the module's global namespace
globals().update(AxTaskMeta._subclasses)
globals().update(MetaObject._subclasses)

# Copyright Axelera AI, 2025
from __future__ import annotations

from typing import Dict, Type

from .base import AxTaskMeta


class MetaRegistry:
    _meta_types: Dict[str, Type[AxTaskMeta]] = {}

    @classmethod
    def register(cls, meta_class: Type[AxTaskMeta]) -> Type[AxTaskMeta]:
        """Decorator to register a meta type using its class name"""
        if not issubclass(meta_class, AxTaskMeta):
            raise TypeError(f"{meta_class.__name__} must be a subclass of AxTaskMeta")
        cls._meta_types[meta_class.__name__] = meta_class
        return meta_class

    @classmethod
    def get_meta_class(cls, name: str) -> Type[AxTaskMeta]:
        """Get a registered meta class by name"""
        return cls._meta_types.get(name)

# Copyright Axelera AI, 2024
from abc import ABC

from axelera.types import enums
import strictyaml as sy

from .validators import (
    AllowDashValidator,
    CaseInsensitiveEnumValidator,
    IntEnumValidator,
    LabelTypeValidator,
    NullValidator,
    OperatorMapValidator,
)

# Fundamental types - not for direct use


class BaseType(ABC):
    @classmethod
    def _as_strictyaml(cls):
        raise NotImplementedError()

    def __new__(cls):
        return cls._as_strictyaml()


class CompoundType(ABC):
    @classmethod
    def _as_strictyaml(cls, check_required):
        raise NotImplementedError()

    def __new__(cls, check_required):
        return cls._as_strictyaml(check_required)


class MapType:
    def __new__(cls, schema, allow_key_dashes):
        return cls._as_strictyaml(schema, allow_key_dashes)

    @classmethod
    def _as_strictyaml(cls, schema):
        raise NotImplementedError()


# Custom Regex Type


class Regex(BaseType):
    def __class_getitem__(cls, pattern):
        return type(f"{cls.__name__}[{pattern}]", (cls,), {"pattern": pattern})

    @classmethod
    def _as_strictyaml(cls):
        return sy.Regex(cls.pattern)


# Custom Enum Types


class Enum(BaseType):
    def __class_getitem__(cls, values):
        split = values.split(",") if isinstance(values, str) else values
        return type(f"{cls.__name__}[{values}]", (cls,), {"values": split})

    @classmethod
    def _as_strictyaml(cls):
        return sy.Enum(cls.values)


class CaseInsensitiveEnum(Enum):
    @classmethod
    def _as_strictyaml(cls):
        return CaseInsensitiveEnumValidator(cls.values)


class IntEnum(Enum):
    @classmethod
    def _as_strictyaml(cls):
        return IntEnumValidator(cls.values)


# Special types
class Optional(
    CompoundType
):  # Optional doesn't really need to be CompoundType, but it makes compile logic easier.
    def __class_getitem__(cls, key):
        return type(f"Optional[{key}]", (cls,), {"key": key})

    @classmethod
    def _as_strictyaml(cls, _):
        return sy.Optional(cls.key)


class Required(CompoundType):
    def __class_getitem__(cls, key):
        return type(f"Required[{key}]", (cls,), {"key": key})

    @classmethod
    def _as_strictyaml(cls, check_required):
        if check_required:
            return cls.key
        return sy.Optional(cls.key)


class Map(MapType):
    @classmethod
    def _as_strictyaml(cls, schema, allow_key_dashes):
        key_validator = AllowDashValidator() if allow_key_dashes else None
        return sy.Map(schema, key_validator)


class OperatorMap(MapType):
    """A Map with better error messages for undeclared operators."""

    @classmethod
    def _as_strictyaml(cls, schema, allow_key_dashes):
        key_validator = AllowDashValidator() if allow_key_dashes else None
        return OperatorMapValidator(schema, key_validator)


class MapCombined(MapType):
    @classmethod
    def _as_strictyaml(cls, schema, allow_key_dashes):
        if allow_key_dashes:
            return sy.MapCombined(schema, key_validator=AllowDashValidator())
        return sy.MapCombined(schema, sy.Str(), sy.Any())


class Union(CompoundType):
    def __class_getitem__(cls, types):
        named_types = [t.__name__ if isinstance(t, type) else "dict" for t in types]
        return type(f"Union[{', '.join(named_types)}]", (cls,), {"types": types})

    @classmethod
    def _as_strictyaml(cls, check_required):
        schema = None
        for t in cls.types:
            if isinstance(t, dict):
                t_schema = compile_schema(t, check_required)
            elif issubclass(t, CompoundType):
                t_schema = t(check_required)
            else:
                t_schema = t()
            schema = t_schema if schema is None else schema | t_schema
        return schema


# Basic Types


class Any(BaseType):
    @classmethod
    def _as_strictyaml(cls):
        return sy.Any()


class Bool(BaseType):
    @classmethod
    def _as_strictyaml(cls):
        return sy.Bool()


class Float(BaseType):
    @classmethod
    def _as_strictyaml(cls):
        return sy.Float()


class Int(BaseType):
    @classmethod
    def _as_strictyaml(cls):
        return sy.Int()


class Str(BaseType):
    @classmethod
    def _as_strictyaml(cls):
        return sy.Str()


class List(CompoundType):
    Element = Any

    @classmethod
    def __class_getitem__(cls, item):
        if isinstance(item, dict):
            return type(f"{cls.__name__}[{dict}]", (cls,), {"Element": item})
        return type(f"{cls.__name__}[{item.__name__}]", (cls,), {"Element": item})

    @classmethod
    def _as_strictyaml(cls, check_required):
        if cls.Element in (Bool, Float, Int, Str):
            return sy.Seq(cls.Element()) | sy.CommaSeparated(cls.Element())
        if isinstance(cls.Element, dict):
            return sy.Seq(compile_schema(cls.Element, check_required))
        if issubclass(cls.Element, Union) and all(
            t in (Bool, Float, Int, Str) for t in cls.Element.types
        ):
            schema = sy.Seq(cls.Element(check_required))
            for t in cls.Element.types:
                schema = schema | sy.CommaSeparated(t())
            return schema
        if issubclass(cls.Element, CompoundType):
            return sy.Seq(cls.Element(check_required))
        return sy.Seq(cls.Element())


class MapPattern(CompoundType):
    Element = Any

    @classmethod
    def __class_getitem__(cls, item):
        if isinstance(item, dict):
            return type(f"{cls.__name__}[{dict}]", (cls,), {"Element": item})
        return type(f"{cls.__name__}[{item.__name__}]", (cls,), {"Element": item})

    @classmethod
    def _as_strictyaml(cls, check_required):
        if isinstance(cls.Element, dict):
            return sy.MapPattern(sy.Str(), compile_schema(cls.Element, check_required))
        elif issubclass(cls.Element, CompoundType):
            return sy.MapPattern(sy.Str(), cls.Element(check_required))
        return sy.MapPattern(sy.Str(), cls.Element())


class EmptyDict(BaseType):
    @classmethod
    def _as_strictyaml(cls):
        return sy.EmptyDict()


class EmptyList(BaseType):
    @classmethod
    def _as_strictyaml(cls):
        return sy.EmptyList()


class Null(BaseType):
    @classmethod
    def _as_strictyaml(cls):
        return sy.EmptyNone() | NullValidator()


# Enum types
class TaskCategory(Enum):
    values = [e.name for e in enums.TaskCategory]


class TensorLayout(Enum):
    values = [e.name for e in enums.TensorLayout]


class ColorFormat(CaseInsensitiveEnum):
    values = [e.name for e in enums.ColorFormat]


class ModelType(Enum):
    values = [e.name for e in enums.ModelType]


class ImageReaderBackend(CaseInsensitiveEnum):
    values = [e.name for e in enums.ImageReader]


class InterpolationMode(Enum):
    values = [
        'nearest',
        'bilinear',
        'bicubic',
        'lanczos',
    ]


class InputSource(Enum):
    values = [
        'full',
        'meta',
        'roi',
        'image_processing',
        'no_tiles',
    ]


class TopKRanking(Enum):
    values = [
        'AREA',
        'SCORE',
        'CENTER',
        'NONE',
    ]


class SupportedLabelType(BaseType):
    @classmethod
    def _as_strictyaml(cls):
        return LabelTypeValidator()


# Regex types
class Variable(Regex):
    pattern = r"\${{([^:}]+)(?::([^}]+))?}}"


class Sentinel(Regex):
    pattern = r"\$\$[^\$]+\$\$"


class Expression(Regex):
    pattern = r"\${([^{}][^}]*)}"


def compile_schema(definition, check_required):
    compiled = {}
    MapType = Map
    allow_key_dashes = False
    for key, value in definition.items():
        if key == "_type":
            MapType = value
        elif key == "_allow_key_dashes":
            allow_key_dashes = value
        elif isinstance(value, dict):
            compiled[key(check_required)] = sy.EmptyDict() | compile_schema(value, check_required)
        elif issubclass(value, CompoundType):
            compiled[key(check_required)] = value(check_required)
        else:
            compiled[key(check_required)] = value()
    return MapType(compiled, allow_key_dashes)

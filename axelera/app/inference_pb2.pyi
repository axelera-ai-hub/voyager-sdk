from typing import ClassVar as _ClassVar
from typing import Iterable as _Iterable
from typing import Mapping as _Mapping
from typing import Optional as _Optional
from typing import Union as _Union

from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf.internal import containers as _containers

DESCRIPTOR: _descriptor.FileDescriptor

class InitInferenceRequest(_message.Message):
    __slots__ = ("network",)
    NETWORK_FIELD_NUMBER: _ClassVar[int]
    network: str
    def __init__(self, network: _Optional[str] = ...) -> None: ...

class InitInferenceResponse(_message.Message):
    __slots__ = ("status",)
    STATUS_FIELD_NUMBER: _ClassVar[int]
    status: str
    def __init__(self, status: _Optional[str] = ...) -> None: ...

class StreamInferenceRequest(_message.Message):
    __slots__ = ("image",)
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    image: Image
    def __init__(self, image: _Optional[_Union[Image, _Mapping]] = ...) -> None: ...

class InferenceRequest(_message.Message):
    __slots__ = ("network", "input")
    NETWORK_FIELD_NUMBER: _ClassVar[int]
    INPUT_FIELD_NUMBER: _ClassVar[int]
    network: str
    input: str
    def __init__(self, network: _Optional[str] = ..., input: _Optional[str] = ...) -> None: ...

class ObjectMeta(_message.Message):
    __slots__ = ("boxes", "scores", "classes")
    BOXES_FIELD_NUMBER: _ClassVar[int]
    SCORES_FIELD_NUMBER: _ClassVar[int]
    CLASSES_FIELD_NUMBER: _ClassVar[int]
    boxes: bytes
    scores: bytes
    classes: bytes
    def __init__(
        self,
        boxes: _Optional[bytes] = ...,
        scores: _Optional[bytes] = ...,
        classes: _Optional[bytes] = ...,
    ) -> None: ...

class Scores(_message.Message):
    __slots__ = ("scores",)
    SCORES_FIELD_NUMBER: _ClassVar[int]
    scores: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, scores: _Optional[_Iterable[bytes]] = ...) -> None: ...

class Boxes(_message.Message):
    __slots__ = ("boxes",)
    BOXES_FIELD_NUMBER: _ClassVar[int]
    boxes: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, boxes: _Optional[_Iterable[bytes]] = ...) -> None: ...

class Classes(_message.Message):
    __slots__ = ("classes",)
    CLASSES_FIELD_NUMBER: _ClassVar[int]
    classes: _containers.RepeatedScalarFieldContainer[bytes]
    def __init__(self, classes: _Optional[_Iterable[bytes]] = ...) -> None: ...

class Classifier(_message.Message):
    __slots__ = ("scores", "boxes", "classes")
    SCORES_FIELD_NUMBER: _ClassVar[int]
    BOXES_FIELD_NUMBER: _ClassVar[int]
    CLASSES_FIELD_NUMBER: _ClassVar[int]
    scores: _containers.RepeatedCompositeFieldContainer[Scores]
    boxes: _containers.RepeatedCompositeFieldContainer[Boxes]
    classes: _containers.RepeatedCompositeFieldContainer[Classes]
    def __init__(
        self,
        scores: _Optional[_Iterable[_Union[Scores, _Mapping]]] = ...,
        boxes: _Optional[_Iterable[_Union[Boxes, _Mapping]]] = ...,
        classes: _Optional[_Iterable[_Union[Classes, _Mapping]]] = ...,
    ) -> None: ...

class Image(_message.Message):
    __slots__ = ("width", "height", "channels", "image", "color_format")
    WIDTH_FIELD_NUMBER: _ClassVar[int]
    HEIGHT_FIELD_NUMBER: _ClassVar[int]
    CHANNELS_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    COLOR_FORMAT_FIELD_NUMBER: _ClassVar[str]
    width: int
    height: int
    channels: int
    image: bytes
    color_format: str
    def __init__(
        self,
        width: _Optional[int] = ...,
        height: _Optional[int] = ...,
        channels: _Optional[int] = ...,
        image: _Optional[bytes] = ...,
        color_format: _Optional[str] = ...,
    ) -> None: ...

class Inferenceresult(_message.Message):
    __slots__ = ("obj", "classifier", "image")
    OBJ_FIELD_NUMBER: _ClassVar[int]
    CLASSIFIER_FIELD_NUMBER: _ClassVar[int]
    IMAGE_FIELD_NUMBER: _ClassVar[int]
    obj: ObjectMeta
    classifier: Classifier
    image: Image
    def __init__(
        self,
        obj: _Optional[_Union[ObjectMeta, _Mapping]] = ...,
        classifier: _Optional[_Union[Classifier, _Mapping]] = ...,
        image: _Optional[_Union[Image, _Mapping]] = ...,
    ) -> None: ...

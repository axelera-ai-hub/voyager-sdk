# Copyright Axelera AI, 2023
from __future__ import annotations

import abc
from collections import OrderedDict
from dataclasses import asdict, dataclass, field
import enum
import importlib
import logging
import math
import os
import queue
import random
import re
import threading
import time
import traceback
from types import UnionType
import typing
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Tuple
import uuid

import numpy as np
import psutil

from axelera import types

from . import config, logging_utils, utils

if TYPE_CHECKING:
    from . import inf_tracers
    from .meta import AxMeta, AxTaskMeta


LOG = logging_utils.getLogger(__name__)

FULL_SCREEN = (-1, -1)
ICONS = {
    sz: f'{os.path.dirname(__file__)}/render_assets/axelera-{sz}x{sz}.png' for sz in [32, 128, 192]
}


def _parse_coord(coord: str, format: str) -> tuple[float | int, str]:
    try:
        if format == '%':
            return float(coord) / 100, 'rel'
        if format == 'px':
            return int(float(coord)), 'px'
    except ValueError:
        pass
    raise ValueError(f"Invalid coordinate: {coord}. Must be in format 'c[%|px]'")


def _scale_visibility(
    now: float,
    visibility: float,
    fadeout_from: float,
    fadeout_for: float,
    fadein_by: float,
    fadein_for: float,
) -> float:
    fully_visible_from = fadein_by if fadein_by is not None else float('-inf')
    fully_visible_until = fadeout_from if fadeout_from is not None else float('inf')
    fully_invisible_from = fully_visible_until + (fadeout_for or 0.0)
    fully_invisible_before = fully_visible_from - (fadein_for or 0.0)

    if now < fully_invisible_before or now >= fully_invisible_from:
        return 0.0
    if fully_visible_from <= now < fully_visible_until:
        return 1.0
    # If not one of the above states, we must be in a partially faded state
    fadeout = fadein = None
    if fadeout_from and fadeout_for:
        fadeout = (fadeout_from - now + fadeout_for) / fadeout_for
    if fadein_by and fadein_for:
        fadein = (now - fadein_by + fadein_for) / fadein_for
    if fadeout is not None and fadein is not None:
        scale = min(fadein, fadeout)
    else:
        scale = fadein if fadein is not None else fadeout
    if scale is not None:
        visibility = max(0.0, min(visibility, scale * visibility))
    return visibility


class Coords:
    '''
    The Coords class is used to represent coordinates for layers in the window.

    The coordinates system is string based, and CSS-inspired (not like for like).

    The currently supported formats are:
      - Absolute pixels, denoted by 'px' (e.g. '100px', '200px')
      - Relative, denoted by '%' (e.g. '50%', '75%')

    The coordinates can be specified as a single string, or as an x and y argument
    .
    For example:
      - '100px', '200px'
      - '50%', '75%'
      - '100px, 200px'
      - '50%, 75%',
      - '100px,50%'

    are all valid coordinates.

    When interpreted by a layer, the coordinates will be normalized to the same as the layer.
    For example, if the layer belongs to a specific stream, the coordinates will be interpreted
    relative to the stream image canvas. If the layer belongs to the window, the coordinates will
    be interpreted relative to the canvas of the entire window.

    Coordinates are static after creation, but can be returned as any supported format using the
    provided methods.

    Args:
        x_or_xy: The x coordinate, or a string containing both x and y coordinates in the format
                 'x[%|px], y[%|px]'.
        y:       The y coordinate. This is only used if `x_or_xy` is a single coordinate string.
                 If `x_or_xy` is a pair of coordinates, this argument must be None.
    '''

    COORD = r'\s*([\d.]+)(px|%)\s*'

    @classmethod
    def rel(cls, x: float, y: float) -> Coords:
        '''
        Create a Coords object from relative coordinates.

        This is useful for creating coordinates from programatic values, and is not
        recommended for parsing user input.

        Args:
            x: The x coordinate as a float between 0.0 and 1.0.
            y: The y coordinate as a float between 0.0 and 1.0.
        '''
        if type(x) is not float or type(y) is not float:
            raise TypeError("x and y must be floats")
        return cls(f'{x * 100}%', f'{y * 100}%')

    @classmethod
    def px(cls, x: int, y: int) -> Coords:
        '''
        Create a Coords object from absolute pixel coordinates.

        This is useful for creating coordinates from programatic values, and is not
        recommended for parsing user input.

        Args:
            x: The x coordinate as an int.
            y: The y coordinate as an int.
        '''
        if type(x) is not int or type(y) is not int:
            raise TypeError("x and y must be ints")
        return cls(f'{x}px', f'{y}px')

    def __init__(self, x_or_xy: str, y: str = None):
        if not isinstance(x_or_xy, str):
            raise TypeError(
                "x_or_xy must be a coordinate string in format 'x[%|px]', or "
                "a pair of coordinates in format 'x[%|px], y[%|px]'. "
                f"Got {type(x_or_xy).__name__}"
            )
        xy_pair = re.match(rf'{self.COORD},{self.COORD}$', x_or_xy)
        if xy_pair:
            if y:
                raise ValueError(
                    "y must be None when coordinates are given as a pair in "
                    f"a single string. Got {y}"
                )
            self._x, self._x_format = _parse_coord(xy_pair.group(1), xy_pair.group(2))
            self._y, self._y_format = _parse_coord(xy_pair.group(3), xy_pair.group(4))
            return

        x = re.match(rf'{self.COORD}$', x_or_xy)
        if not x:
            raise ValueError(
                "x_or_xy must be a coordinate string in format 'x[%|px]', or "
                "a pair of coordinates in format 'x[%|px], y[%|px]'. "
                f"Got {x_or_xy}"
            )
        self._x, self._x_format = _parse_coord(x.group(1), x.group(2))
        if not isinstance(y, str):
            raise TypeError(
                "y must be a coordinate string in format 'y[%|px]'. "
                f"Got {type(x_or_xy).__name__}"
            )
        y = re.match(rf'{self.COORD}$', y)
        if not y:
            raise ValueError("y must be a coordinate string in format 'y[%|px]'. " f"Got {y}")
        self._y, self._y_format = _parse_coord(y.group(1), y.group(2))

    def __eq__(self, value):
        '''
        Coordinates may be tested for equality with other coordinates, or tuple/strings coordinates
        which follow the same format as the constructor.

        Note that coordinates are not considered equal to float or int tuples, as this is
        insufficient information to determine the format of the coordinates.
        '''
        if isinstance(value, Coords):
            x = value.x
            y = value.y
            x_format = value._x_format
            y_format = value._y_format
        elif isinstance(value, (tuple, list)) and len(value) == 2:
            if not isinstance(value[0], str) or not isinstance(value[1], str):
                return False
            try:
                x_match = re.match(self.COORD, value[0])
                y_match = re.match(self.COORD, value[1])
                if not x_match or not y_match:
                    return False
                x, x_format = _parse_coord(x_match.group(1), x_match.group(2))
                y, y_format = _parse_coord(y_match.group(1), y_match.group(2))
            except ValueError:
                return False
        elif isinstance(value, str):
            xy_matches = re.match(rf'{self.COORD},{self.COORD}$', value)
            if xy_matches:
                try:
                    x, x_format = _parse_coord(xy_matches.group(1), xy_matches.group(2))
                    y, y_format = _parse_coord(xy_matches.group(3), xy_matches.group(4))
                except ValueError:
                    return False
            else:
                return False
        else:
            return False
        return (
            self._x == x
            and self._y == y
            and self._x_format == x_format
            and self._y_format == y_format
        )

    @property
    def format(self) -> str:
        '''
        Return the format of the coordinates.

        The format is given as a string shorthand name of the format, such as 'px' or 'rel'.
        The coordinates will only be considered to be a given format if both the x and y
        coordinates are that same format. If the x and y coordinates are different formats,
        the format 'mix' will be returned.
        '''
        if self._x_format == self._y_format:
            return self._x_format
        return 'mix'

    @property
    def x(self) -> float | int:
        '''Return the x coordinate.'''
        return self._x

    @property
    def y(self) -> float | int:
        '''Return the y coordinate.'''
        return self._y

    def as_px(self, image_size: tuple[int, int] = None) -> tuple[int, int]:
        '''
        Convert the coordinates to absolute pixel coordinates.

        If the coordinates are already in absolute pixel format, they will be returned as is.

        Otherwise, the `image_size` argument must be provided, and the coordinates will
        be converted to absolute pixels, relative to this image size.

        Args:
            image_size: The size of the image as a tuple (width, height). This is only used
                        if the coordinates are not already in absolute pixel format.
        '''
        if self.format == 'px':
            return self._x, self._y
        if image_size is None:
            raise ValueError(f"image_size must be provided to convert {self.format} to px")
        x = self._x
        y = self._y
        if self._x_format == 'rel':
            x = int(x * image_size[0])
        if self._y_format == 'rel':
            y = int(y * image_size[1])
        return x, y

    def as_rel(self, image_size: tuple[int, int] = None) -> tuple[float, float]:
        '''
        Convert the coordinates to relative coordinates.

        If the coordinates are already in relative format, they will be returned as is.

        Otherwise, the `image_size` argument must be provided, and the coordinates will
        be converted to relative pixels, from this image size.

        Args:
            image_size: The size of the image as a tuple (width, height). This is only used
                        if the coordinates are not already in relative format.
        '''
        if self.format == 'rel':
            return self._x, self._y
        if image_size is None:
            raise ValueError(f"image_size must be provided to convert {self.format} to rel")
        x = self._x
        y = self._y
        if self._x_format == 'px':
            x = x / image_size[0]
        if self._y_format == 'px':
            y = y / image_size[1]
        return x, y


class FrameStyle(enum.Enum):
    NORMAL = 'normal'
    GRID = 'grid'
    SIDE_BY_SIDE = 'side_by_side'


class _Message:
    pass


@dataclass
class _StreamMessage(_Message):
    stream_id: int


@dataclass
class _CloseSource(_StreamMessage):
    reopen: bool


@dataclass
class _OpenSource(_StreamMessage):
    pass


@dataclass
class _SetOptions(_StreamMessage):
    options: dict[str, Any]


@dataclass
class _ClearState(_StreamMessage):
    pass


@dataclass
class _Frame(_StreamMessage):
    image: types.Image
    meta: AxMeta | None


@dataclass
class _BlockingFrame(_Frame):
    '''
    For use when rendering to a Surface, with blocking_render.

    Instructs the rendering thread to always render this frame (don't drop
    it), and add it to the blocking queue in the _FrameSink.
    '''

    pass


@dataclass
class _Shutdown(_Message):
    pass


@dataclass
class _ThreadCompleted(_Message):
    pass


@dataclass
class _Layer(_StreamMessage):
    id: uuid.UUID
    position: Coords
    anchor_x: str
    anchor_y: str
    fadeout_from: Optional[float]
    fadeout_for: Optional[float]
    fadein_by: Optional[float]
    fadein_for: Optional[float]

    def __post_init__(self):
        if self.fadeout_from and self.fadein_by and self.fadeout_from < self.fadein_by:
            raise ValueError(
                "fadeout_from must not be less than fadein_by if both are set."
                f" fadeout_from: {self.fadeout_from}, fadein_by: {self.fadein_by}"
            )

    @property
    def visibility(self) -> float:
        '''Return the visibility of the layer.'''
        return _scale_visibility(
            time.time(),
            1.0,  # TODO: this should be a layer parameter i.e. self._visibility
            self.fadeout_from,
            self.fadeout_for,
            self.fadein_by,
            self.fadein_for,
        )


@dataclass
class _Text(_Layer):
    text: str
    color: Color
    bgcolor: Color
    font_size: int


@dataclass
class _Image(_Layer):
    path: str
    scale: float | None


@dataclass
class _Rectangle(_Layer):
    bottom_right: Coords
    color: Color


class LayerHandle:
    '''
    The LayerHandle is used to create and control layers in the window.

    In general, LayerHandles should be created from the `Window` class, not
    directly.

    After a layer is created, it is assigned a unique id, and belongs to a
    given stream_id. If the `stream_id` is -1, the layer belongs to the window.

    Other fields of the layer can be updated using the `set` method, or by
    dict-style assignment. The layer is updated in place, and any changes will
    be sent to the renderer for it to interpret.

    In addition to updating fields, the LayerHandle provides various control
    methods to handle the state of the layer, and can return information about
    the layer.

    The following arguments may be passed to the constructor of the LayerHandle,
    but it is recommened to use the `Window` class to create layers instead.

    Args:
        message: The message to create the layer from. This must be a subclass of `_Layer`.
                 The message will be used to create the handle, and will be sent to the
                 renderer as part of the contruction.
        window:  The window the layer belongs to. This is used to send messages to the
                 renderer.
    '''

    def __init__(self, message: _Layer, window: Window):
        self._fields = asdict(message, dict_factory=OrderedDict)
        self._id = self._fields.pop('id')
        self._stream_id = self._fields.pop('stream_id')
        self._Message = type(message)
        self._window = window
        if self.visible:
            self._send_message(message)

    @property
    def id(self) -> uuid.UUID:
        '''The id of the layer.'''
        return self._id

    @property
    def stream_id(self) -> int:
        '''
        The id of the stream the layer belongs to.

        stream_id `-1` means the layer is not associated with a stream, and belongs
        to the window(s).
        '''
        return self._stream_id

    @property
    def visible(self) -> float:
        '''Returns a float between 0.0 and 1.0 representing the visibility of the layer.

        0.0 means the layer is fully invisible, 1.0 means the layer is fully visible.

        Values between 0.0 and 1.0 occur when the layer is fading in or out.

        It is the responsibility of the calling code to interpret the visibility
        how it sees fit.
        '''
        return _scale_visibility(
            time.time(),
            1.0,
            self["fadeout_from"],
            self["fadeout_for"],
            self["fadein_by"],
            self["fadein_for"],
        )

    @property
    def message(self) -> _Layer:
        '''
        Create and return a message from the current state of the LayerHandle.

        The message will be a _Layer subclass matching the LayerHandle, setup with
        the stream ID and ID of the LayerHandle, and the current values of all the
        fields.

        It is not normally necessary to use this method directly. It is useful if
        you need an instance of the message to send somewhere other than
        the window renderer. Such as an output sink to render layers to a file.
        '''
        return self._Message(
            self.stream_id,
            self.id,
            *self._fields.values(),
        )

    def _send_message(self, message: _Layer = None) -> None:
        message = message or self.message
        if self._window._queue is not None:
            self._window._queue.put(message)

    def _set_from_dict(self, new_fields: dict[str, Any], message=None) -> None:
        changed = False
        was_visible = self.visible
        for name, value in new_fields.items():
            if name in self._fields:
                if name == 'position':
                    if not isinstance(value, Coords):
                        if isinstance(value, str):
                            value = Coords(value)
                        else:
                            value = Coords(*value)
                if self._fields[name] == value:
                    continue
                self._fields[name] = value
                changed = True
            else:
                LOG.warning(
                    f"Cannot set attribute '{name}' for LayerHandle."
                    f" id={self.id}, message_type={self._Message.__name__}"
                )
        # Special case: if the layer is now invisible due to fadeout, we still
        # need to message the renderer
        should_send = self.visible or was_visible
        if should_send and changed:
            self._send_message(message)

    def _set_from_msg(self, message: _Layer) -> None:
        _Message = getattr(self, '_Message')
        if _Message and _Message is not type(message):
            raise TypeError(
                f"Cannot set LayerHandle from message with different type. "
                f"LayerHandle type: {_Message.__name__}, message type: {type(message).__name__}"
            )
        new_fields = asdict(message, dict_factory=OrderedDict)
        for _id in ('id', 'stream_id'):
            existing = getattr(self, _id)
            new = new_fields.pop(_id)
            if existing and existing != new:
                raise ValueError(
                    f"Cannot set LayerHandle from message with different {_id}. "
                    f"LayerHandle {_id}: {existing}, message {_id}: {new}"
                )
        self._set_from_dict(new_fields, message)

    def set(self, *args, **kwargs) -> None:
        '''
        Update fields in the layer.

        Only fields which are available in the layer, and are mutable, may be set here.
        `id` and `stream_id` are not mutable, and cannot be set. Fields may be either be
        set via field-name kwargs (recommended) or a single message arg.

        If field-name kwargs may be used such as `anchor_x='left'. Multiple fields may be
        updated at once. It is recommended to set multiple fields in the manner to reduce
        the number of messages sent to the renderer. After the fields are set, a message
        will be generated and sent to the renderer^.

        Alternatively, a single message may be passed a sole argument. The fields in the
        LayerHandle will be replaced with the fields in the message. The message must be
        of the same type, and have the same id and stream_id as the LayerHandle. After
        updating the fields, the message will be forwarded onto the renderer^. This
        approach can be useful programatically, it is generally recommended to use the
        kwarg approach in application code.

        ^ If the layer is completely invisible, the message will not be sent to the renderer,
          but the LayerHandle will still be updated. When the layer becomes visible again,
          any changes made whilst the layer was invisible will be sent to the renderer at
          this point. The exception to this rule is if the changes cause the layer to
          transition from invisible to visible, in which case the message will be sent.
        '''
        if args and kwargs:
            raise ValueError("Cannot set LayerHandle with both message and keyword arguments.")
        if args:
            if len(args) == 1 and isinstance(args[0], _Layer):
                return self._set_from_msg(args[0])
            raise ValueError("LayerHandle.set with args only accepts a single _Layer message.")
        if kwargs:
            return self._set_from_dict(kwargs)
        raise ValueError(f"Invalid arguments for LayerHandle.set: {args}, {kwargs}")

    def __setitem__(self, key, value):
        '''
        Update a single field in the layer, in the same manner as `set`.
        '''
        self.set(**{key: value})

    def __getitem__(self, name):
        '''
        Retrieve a field from the layer. This cannot be used to retrieve the id or
        stream_id of the layer, use the corresponding attribute instead.
        '''
        if name in self._fields:
            return self._fields[name]
        raise KeyError(
            f"Cannot get attribute '{name}' from LayerHandle."
            f" id={self.id}, message_type={self._Message.__name__}"
        )

    def hide(self, from_: Optional[float] = None, fadeout: Optional[float] = None):
        '''
        Hides the layer from the window.

        If called with no arguments the renderer will be instructed to hide the layer
        immediately.

        Note that if the layer is already in the process of being hidden, extra calls will
        be ignored and logged. In addition, calling the hide methiod will cancel any
        show calls which are in progress.

        Optional Args:
            from_:  The UNIX timestamp at which the layer should start fading out.
                    If this is None, the layer will be hidden immediately.

            fadeout: How long in seconds the layer will fade out for. This value is only
                     used if `from_` is not None. If this is None, there will be no fadeout,
                     and the layer will be hidden immediately at `from_`.

        '''
        from_ = from_ if from_ is not None else time.time()
        fading_out = (
            self["fadeout_from"] and self["fadeout_from"] + (self["fadeout_for"] or 0) > from_
        )
        if not fading_out:
            self.set(
                fadeout_from=from_,
                fadeout_for=fadeout,
                fadein_by=None,
                fadein_for=None,
            )
        else:
            LOG.trace(f'layer {self.id} already fading out')

    def show(self, from_: Optional[float] = None, fadein: Optional[float] = None):
        '''
        Shows the layer in the window.

        If called with no arguments the renderer will be instructed to show the layer
        immediately.

        Note that if the layer is already in the process of being shown, extra calls will
        be ignored and logged. In addition, calling the show method will cancel any
        hide calls which are in progress.

        Optional Args:
            from_:  The UNIX timestamp at which the layer should beging fading in.
                    If this is None, the layer will begin fading in immediately.

            fadein: How long in seconds the layer will fade in for. This value is only
                    used if `by` is not None. If this is None, the layer will be fully
                    visible immediately at `by`.
        '''
        by = (from_ if from_ is not None else (time.time())) + (fadein or 0.0)
        fading_in = self["fadein_by"] and self["fadein_by"] - (self["fadein_for"] or 0.0) < by
        if not fading_in:
            self.set(
                fadeout_from=None,
                fadeout_for=None,
                fadein_by=by,
                fadein_for=fadein,
            )
        else:
            LOG.trace(f'layer {self.id} already fading in')


SHUTDOWN = _Shutdown()
THREAD_COMPLETED = _ThreadCompleted()


def _typecheck(v, expected_type):
    if typing.get_origin(expected_type) is UnionType:
        return any(_typecheck(v, t) for t in typing.get_args(expected_type))
    if typing.get_origin(expected_type) is tuple:
        expected_type = typing.get_args(expected_type)
        if len(typing.get_args(expected_type)) == 1:
            return isinstance(v, tuple) and all(isinstance(x, expected_type[0]) for x in v)
        return (
            isinstance(v, tuple)
            and len(v) == len(expected_type)
            and all(isinstance(x, t) for x, t in zip(v, expected_type))
        )
    return isinstance(v, expected_type)


@dataclass
class Options:
    title: str = ''
    '''The title of the stream window, set to '' to hide the title bar.'''

    title_size: int = 20
    '''Size of the title text in points'''

    title_position: str | tuple[str, str] = '0%, 0%'
    '''Position of title. '0%, 0%' top left. '100%, 100%' bottom right.'''

    title_color: tuple[int, int, int, int] = (244, 190, 24, 255)
    '''Text color of title, RGBA (0-255)'''

    title_bgcolor: tuple[int, int, int, int] = (0, 0, 0, 192)
    '''Background color of title, RGBA (0-255)'''

    title_anchor_x: str = 'left'
    '''Anchor pos, one of left/center/right'''
    title_anchor_y: str = 'top'
    '''Anchor pos, one of top/center/bottom'''

    grayscale: float | int | bool = 0.0
    '''Render the inferenced image in grayscale. This can be effective when viewing segmentation.
    The value is a float between 0.0 and 1.0 where 0.0 is the original image and 1.0 is completely
    grayscale.
    '''

    bbox_label_format: str = "{label} {scorep:.0f}{scoreunit}"
    '''Control how labels on bounding boxes should be shown. Available keywords are:

    label: str Label if known, or "cls:%d" if no label for that class id
    score: float The score of the detection as a float 0-1
    scorep: float The score of the detection as a percentage 0-100
    scoreunit: str The unit of the score, such as '%'

    An empty string will result in no label.

    If there are formatting errors in the format string, an error is shown and the default is used
    instead.

    For example, assuming label is "apple" and score is 0.75:

        "{label} {score:.2f}" -> "apple 0.75"
        "{label} {scorep:.0f}%" -> "apple 75%"
        "{scorep}" -> "75"
        "{label}" -> "apple"
    '''

    tracker_label_format: str = "{label}-{track_id}"
    '''Control how labels on tracked bounding boxes should be shown.
    Available keywords are:

    label: str Label if known, or "cls:%d" if no label for that class id
    track_id: str The tracker id of the re-detection, or "id:%d" if there
                  is no label for the aforementioned class id

    An empty string will result in no label.

    If there are formatting errors in the format string, an error is shown and the default is used
    instead.

    For example, assuming label is "apple" (class id 47) and track_id is 3:

        "{label}-{track_id}" -> "apple-3"
        "{label}-{track_id}" -> "cls:47-id:3" (When no labels available)
        "{label}" -> "apple"
        "{track_id}" -> "3"
        "{track_id}" -> "id:3" (When no labels available)

    Note this format doesn't control submeta labels of the tag, they are controlled by
    bbox_label_format.
    '''

    bbox_class_colors: dict = field(default_factory=dict)
    '''A dictionary of class id to color.

    The key should be a class id (int) or a label (str) and the value should be a color tuple
    (r, g, b, a) where each value is an integer between 0 and 255.

    class ids and labels can be mixed. A matching label is given priority over a matching class id.
    '''

    speedometer_smoothing: bool = True
    '''
    Enable realistic movement of the metric speedometers.

    Speedometers will start at 0 and smoothly ramp up to the desired values with subtle
    wobbling at higher values, instead of jumping immediately to the target value on each frame.
    Note that smoothed speedometers may not reflect instantaneous metric changes - disable this
    option when you need immediate visual feedback for debugging performance drops or demonstrating
    real-time metric responses. (default on)
    '''

    show_tiles: bool = False
    '''Internal only: Show tile boundaries when rendering tiled streams.'''

    show_bounding_boxes: bool = True
    '''Show bounding boxes on detected objects.'''

    show_keypoints: bool = True
    '''Show keypoints on detected objects.'''

    show_segmentation: bool = True
    '''Show segmentation masks on detected objects.'''

    show_trajectory: bool = True
    '''Show trajectory lines for tracked objects.'''

    def update(self, **options: dict[str, Any]) -> None:
        '''Update the options with the given dictionary.

        Warns for unsupported options, and logs if the value is not the expected type, in both cases

        '''
        unsupported = []
        types = typing.get_type_hints(type(self))
        for k, v in options.items():
            try:
                expected_type = types[k]
            except KeyError:
                unsupported.append(k)
            else:
                if _typecheck(v, expected_type):
                    setattr(self, k, v)
                else:
                    exp = getattr(expected_type, '__name__', str(expected_type))
                    got = type(v).__name__
                    _for = f"{type(self).__name__}.{k}"
                    LOG.error(f"Expected {exp} for {_for}, but got {got}")

        if unsupported:
            s = 's' if len(unsupported) > 1 else ''
            LOG.info(f"Unsupported option{s} : {type(self).__name__}.{', '.join(unsupported)}")


def _get_visible_and_expired_layers(
    layers_by_id: dict[uuid.UUID, _StreamMessage], source_ids: tuple[int]
) -> tuple[list[_Message], list[uuid.UUID]]:
    now = time.time()
    layers = []
    expired = []
    for k, v in layers_by_id.items():
        if v.stream_id in source_ids:
            if v.fadeout_from:
                expiry_time = v.fadeout_from + (v.fadeout_for or 0)
                if expiry_time <= now:  # Finished fading out
                    expired.append(k)
                    continue
            if v.fadein_by:
                fadein_start = v.fadein_by - (v.fadein_for or 0)
                if fadein_start > now:  # Too early
                    continue
                if now >= v.fadein_by:  # Finished fading in
                    v.fadein_by = None
                    v.fadein_for = None
            layers.append(v)
    return layers, expired


default_anchor_x = 'left'
default_anchor_y = 'top'
default_fadeout_from = None
default_fadeout_for = None
default_fadein_by = None
default_fadein_for = None


class _FrameSink:
    '''
    Object for handling rendering output when application access to the image is
    required, instead of using a builtin Window.

    Supports three methods of accessing the rendered image:

    1. _FrameSink.latest - simple variable always containing the last rendered image, or None
    2. _FrameSink.pop_latest() - non-blocking method to return the last rendered image once. Afterwards
                             returns None until a new image is rendered. Returns None if no images have
                             been rendered yet.
    3. _FrameSink.pop_latest_blocking() - blocking method to return the output of the last `_BlockingFrame` to
                                      be sent to the renderer. Blocks until the rendering is complete.
    '''

    def __init__(self):
        self._latest = None
        self._one_shot_q = queue.Queue(maxsize=1)
        self._blocking_one_shot_q = queue.Queue(maxsize=1)

    def push(self, frame, block=False) -> None:
        '''
        Add a rendered image to the frame sink.

        _FrameSink.latest will be updated to this image.

        Remove any previous image if not consumed by _FrameSink.pop_latest(),
        and return this image on the next call instead.

        If block=True, also add this image to the blocking queue. The last blocking
        image, if any, must have been consumed via _FrameSink.pop_latest_blocking() or
        a RuntimeError will be raised.

        Args:
            frame: The image to add to the frame sink.
            block: Whether to also add this image to the blocking queue.

        Raises:
            RuntimeError: If block=True and the previous blocking image has not been consumed.

        '''
        self._latest = frame
        try:
            self._one_shot_q.put_nowait(frame)
        except queue.Full:
            try:
                self._one_shot_q.get_nowait()  # remove the previous frame if not consumed
            except queue.Empty:
                pass
            self._one_shot_q.put_nowait(frame)
        if block:
            try:
                self._blocking_one_shot_q.put_nowait(frame)
            except queue.Full:
                raise RuntimeError(
                    "Improper use of blocking frame sink, previous frame not consumed"
                )

    def pop_latest(self) -> types.Image | None:
        try:
            return self._one_shot_q.get_nowait()

        except queue.Empty:
            return None

    def pop_latest_blocking(self) -> types.Image:
        return self._blocking_one_shot_q.get()

    @property
    def latest(self) -> types.Image | None:
        return self._latest


def _validate_image(should_be_image: Any) -> None:
    if not isinstance(should_be_image, types.Image):
        raise TypeError(f"Expected axelera.types.Image, got {should_be_image}")


def _requires_frame_sink(func):
    def wrapper(self, *args, **kwargs):
        if self._frame_sink is None:
            raise RuntimeError(
                f"`{func.__name__}` requires `Surface` to be exposed at initialization"
            )
        return func(self, *args, **kwargs)

    return wrapper


def get_primary_screen_resolution():
    try:
        import subprocess

        output = subprocess.check_output(
            r'xrandr | grep "\*" | cut -d" " -f4', shell=True
        ).decode()
        split = output.split()[0].split('x')
        resolution = (int(split[0]), int(split[1]))
        LOG.debug(f"Determined fullscreen primary display resolution as {resolution} via xrandr")
        return resolution
    except Exception:
        LOG.warning(
            f"Could not determine screen resolution with xrandr, using {config.DEFAULT_WINDOW_SIZE}"
        )
        return config.DEFAULT_WINDOW_SIZE


class Surface:
    '''Created by `App.create_surface` to display inference results.'''

    def __init__(
        self,
        q: Optional[queue.Queue],
        size: tuple[int, int],
        frame_sink: Optional[_FrameSink] = None,
        check_running: Optional[callable] = None,
    ):
        self._queue = q
        self._frame_sink = frame_sink
        self._warned_full = False
        self._warned_full_at = 0.0
        self._self_proc = psutil.Process(os.getpid())
        self._last_print = time.time() - 2
        if size == FULL_SCREEN:
            size = get_primary_screen_resolution()
        self.width = size[0]
        self.height = size[1]

        self.__check_running = check_running
        self._checked_running = not self.__check_running

        self.__dummy = None

    def options(self, stream_id: int, **options: dict[str, Any]) -> None:
        '''Set options for the given stream.

        Valid options depends on the renderer being used. All renderers support title:str.
        '''
        if self._queue is not None:
            self._queue.put(_SetOptions(stream_id, options))

    def delete(
        self,
        layer: LayerHandle,
        from_: Optional[float] = None,
        fadeout: Optional[float] = None,
    ) -> None:
        '''Delete the given layer from the stream.'''
        layer.hide(from_, fadeout)

    def layer(
        self,
        Layer: type[_Layer],
        position: str | tuple[str, str] | Coords,
        anchor_x: str,
        anchor_y: str,
        fadeout_from: Optional[float],
        fadeout_for: Optional[float],
        fadein_by: Optional[float],
        fadein_for: Optional[float],
        *args,
        existing: Optional[LayerHandle] = None,
        stream_id: int = -1,
    ) -> LayerHandle:
        '''
        A Layer is a visual element in the surface, created and controlled in the application
        code instead of within the inference stream.

        It is not recommended to use this method directly, instead use the specific methods
        for each supported layer, such as `text` or `image`. This method lists each argument
        which is common to all layers (except for `Layer`), which are described below.

        By default, upon creating a layer, it will be visible from the next frame rendered,
        this can be avoided using the `fadeout_from` or `fadein_by` arguments. See below.

        Args:
            Layer:        The type of layer to create. This must be supported by the renderer being used
                          by the surface. All builtin layers are supported by the CV and GL renderers.

            position:     The position of the layer in the surface. This can either be a `Coords` object,
                          or a string/tuple in a format parseable by the `Coords` constructor. See
                          `Coords` for more details. If the layer belongs to a stream, the position
                          is relative to the stream image size. If the layer belongs to the surface,
                          the position is relative to the surface size.

            anchor_x:     The anchor point of the layer in the x direction. This can be one of
                          'left', 'center', 'right'. The the x coordinate of the position will be
                          located at this anchor.

            anchor_y:     The anchor point of the layer in the y direction. This can be one of
                          'top', 'center', 'bottom'. The the y coordinate of the position will be
                          located at this anchor.

            fadeout_from: The UNIX timestamp at which the layer should start fading out
                          (i.e. be deleted). There are different behaviours which can occur
                          based on this value. Firstly, if this is `None`, the layer will be
                          visible indefinitely. If this is set to a timestamp in the future,
                          the layer will be visible until that timestamp, after which it
                          will start fading out (based on the value of fadeout_for). Finally,
                          if this is set to a timestamp in the past, the layer will not be
                          visible, or will be immediately fading out (depending on the value of
                          fadeout_for). However, a `LayerHandle` will still be created, so the
                          layer can be shown again later.

            fadeout_for:  How long in seconds the layer will fade out for, after the `fadeout_from`
                          timestamp is reached. This value is only used if `fadeout_from` is not
                          `None`. If this is `None`, there will be no fadeout, and the layer will
                          be hidden as soon as the `fadeout_from` timestamp is reached.

            fadein_by:    The UNIX timestamp at which the layer should be fully visible. There are
                          different behaviours which can occur based on this value. Firstly, if this
                          is `None` or a timestamp in the past, the layer will be fully visible. If
                          this is set to a timestamp in the future, the layer will be fully invisible
                          until `fadein_for` seconds before that timestamp, becoming fully visible at
                          the given timestamp. A LayerHandle will still be created even if the layer
                          is not visible yet.

            fadein_for:   How long in seconds the layer will fade in for, before the `fadein_by`
                          timestamp is reached. This value is only used if `fadein_by` is in the
                          future. For example, if fadein_by is 10 seconds in the future and fadein_for
                          is 5 seconds, the layer will be fully invisible until 5 seconds before
                          the `fadein_by` timestamp, at which point it will start fading in. The layer
                          will be fully visible at the `fadein_by` timestamp. If this is `None`, the
                          layer will immediately become fully visible at the `fadein_by` timestamp.

        Optional Args:
            existing:     If this is not `None`, the layer will be updated instead of created. This can
                          be used to update the fields of an existing layer. The layer must be of the
                          same type as the one being created. The layer will be updated in place, and a
                          new LayerHandle will not be created. Default is `None`.

            stream_id:    The id of the stream the layer belongs to. This is used to determine if
                          the layer belongs to given stream, or to the surface.
                          If this is -1, the layer belongs to the surface. Default is -1.

        Returns:
            LayerHandle: A handle to the layer. This can be used to update the layer later.
        '''
        _id = existing.id if existing else uuid.uuid4()
        if not isinstance(position, Coords):
            if isinstance(position, str):
                position = Coords(position)
            else:
                position = Coords(*position)
        layer = Layer(
            stream_id,
            _id,
            position,
            anchor_x,
            anchor_y,
            fadeout_from,
            fadeout_for,
            fadein_by,
            fadein_for,
            *args,
        )
        if existing:
            existing.set(layer)
        return existing or LayerHandle(layer, self)

    def text(
        self,
        position: str | tuple[str, str] | Coords,
        text: str,
        anchor_x: float = default_anchor_x,
        anchor_y: float = default_anchor_y,
        fadeout_from: Optional[float] = default_fadeout_from,
        fadeout_for: Optional[float] = default_fadeout_for,
        fadein_by: Optional[float] = default_fadein_by,
        fadein_for: Optional[float] = default_fadein_for,
        color: Color = (244, 190, 24, 255),  # Orange
        bgcolor: Color = (0, 0, 0, 192),  # Smoked glass
        font_size: int = 32,
        existing: Optional[LayerHandle] = None,
        stream_id: int = -1,
    ) -> LayerHandle:
        '''
        Text is a layer which displays the given text on the Window. The font, text color
        and background color of the text may be configured.

        The following args are in addition to the common args described in `layer`.
        Args:
            text:        The text to display. This must be a string.

        Optional Args:
            color:       The color of the text. This is a tuple of 4 integers between 0 and 255,
                         representing the red, green, blue and alpha channels respectively. Defaults
                         to an orange color (244, 190, 24, 255).

            bgcolor:     The background color of the text. This is a tuple of 4 integers between 0
                         and 255, representing the red, green, blue and alpha channels respectively.
                         Defaults to a smoked glass color (0, 0, 0, 192). Can be set to (0, 0, 0, 255)
                         for a fully transparent background.

            font_size:   The size of the font in pixels. This is an integer. Defaults to 32.

        Returns:
            LayerHandle: A handle to the layer. This can be used to update the layer later.
        '''
        return self.layer(
            _Text,
            position,
            anchor_x,
            anchor_y,
            fadeout_from,
            fadeout_for,
            fadein_by,
            fadein_for,
            text,
            color,
            bgcolor,
            font_size,
            existing=existing,
            stream_id=stream_id,
        )

    def image(
        self,
        position: str | tuple[str, str] | Coords,
        path: str,
        anchor_x: float = default_anchor_x,
        anchor_y: float = default_anchor_y,
        fadeout_from: Optional[float] = default_fadeout_from,
        fadeout_for: Optional[float] = default_fadeout_for,
        fadein_by: Optional[float] = default_fadein_by,
        fadein_for: Optional[float] = default_fadein_for,
        scale: Optional[float] = None,
        existing: Optional[LayerHandle] = None,
        stream_id: int = -1,
    ) -> LayerHandle:
        '''
        Image is a layer which displays the given image on the Window. The image is loaded from
        the given path. Currently, only images loaded from paths are supported.

        The following args are in addition to the common args described in `layer`.
        Args:
            path:       The path to the image to display. This must be a string.

        Optional Args:
            scale:      The scale of the image. This is a float between 0.0 and 1.0, where 1.0 is
                        the original size of the image. Defaults to None, which means the image
                        will be displayed at its original size.

        Returns:
            LayerHandle: A handle to the layer. This can be used to update the layer later.
        '''
        return self.layer(
            _Image,
            position,
            anchor_x,
            anchor_y,
            fadeout_from,
            fadeout_for,
            fadein_by,
            fadein_for,
            path,
            scale,
            existing=existing,
            stream_id=stream_id,
        )

    def rectangle(
        self,
        position: str | tuple[str, str] | Coords,
        bottom_right: str | tuple[str, str] | Coords,
        fadeout_from: Optional[float] = default_fadeout_from,
        fadeout_for: Optional[float] = default_fadeout_for,
        fadein_by: Optional[float] = default_fadein_by,
        fadein_for: Optional[float] = default_fadein_for,
        color: Optional[Color] = (255, 0, 0, 255),  # Red
        existing: Optional[LayerHandle] = None,
        stream_id: int = -1,
    ) -> LayerHandle:
        '''
        Rectangle is a layer which displays a rectangle on the Window. The rectangle is defined
        by the top-left position and bottom-right position.

        The following args are in addition to the common args described in `layer`.
        Args:
            bottom_right:   The bottom-right position of the rectangle. This can either be a
                            `Coords` object, or a string/tuple in a format parseable by the
                            `Coords` constructor. See `Coords` for more details.

        Optional Args:
            color:      The color of the rectangle. This is a tuple representing an RGBA color.
                        Defaults to red.

        Returns:
            LayerHandle: A handle to the layer. This can be used to update the layer later.
        '''
        if not isinstance(bottom_right, Coords):
            if isinstance(bottom_right, str):
                bottom_right = Coords(bottom_right)
            else:
                bottom_right = Coords(*bottom_right)
        return self.layer(
            _Rectangle,
            position,
            default_anchor_x,
            default_anchor_y,
            fadeout_from,
            fadeout_for,
            fadein_by,
            fadein_for,
            bottom_right,
            color,
            existing=existing,
            stream_id=stream_id,
        )

    def close_source(self, source_id: int, reopen: bool = False):
        '''
        Close (stop rendering) the given source, without closing the surface.

        Args:
            source_id:  The id of the source to close.
            reopen: If True, the source will be automatically reopened when a new frame
                    belonging to the source is received. If False, the source will remain
                    closed, ignoring any new frames unless explictly reopened via `open_source`.
                    Default False.

        '''
        if self._queue is not None:
            self._queue.put(_CloseSource(source_id, reopen))
        else:
            LOG.warning(f"Cannot close source {source_id}, surface message queue not set.")

    def open_source(self, source_id: int):
        '''
        Reopen a source which was closed via `close_source`.

        After calling this method, the source may not reappear as soon as this method is
        called. It will only reappear when the next frame belonging to the source is
        received.

        Args:
            source_id:  The id of the source to reopen.
        '''
        if self._queue is not None:
            self._queue.put(_OpenSource(source_id))
        else:
            LOG.warning(f"Cannot open source {source_id}, surface message queue not set.")

    def clear_state(self, source_id: int):
        '''
        Clear any rendering state which persists between frames for the given source.

        Args:
            source_id:  The id of the source to clear the state of.
        '''
        if self._queue is not None:
            self._queue.put(_ClearState(source_id))
        else:
            LOG.warning(
                f"Cannot clear state for source {source_id}, surface message queue not set."
            )

    def _check_running(self):
        if self.__check_running is not None and not self._checked_running:
            self.__check_running()
            self._checked_running = True

    def resize(self, width: int, height: int) -> None:
        del width, height
        raise NotImplementedError("`resize` will be available in the future.")

    @property
    @_requires_frame_sink
    def latest(self) -> types.Image:
        '''
        View the latest rendered frame, even if already popped.

        Returns:
            1. The last rendered frame
            2. A black frame of surface size, if no frames have been pushed yet
        Raises:
            RuntimeError: If access to the contents of the surface was not exposed at initialization.
        '''
        self._check_running()
        if self._frame_sink.latest is None:
            # Return a black frame if no frames have been rendered yet
            if self.__dummy is None:
                self.__dummy = types.Image.fromarray(
                    np.zeros((self.height, self.width, 3), dtype=np.uint8)
                )
            return self.__dummy
        return self._frame_sink.latest

    @_requires_frame_sink
    def pop(self, timeout=None) -> types.Image:
        del timeout
        raise NotImplementedError(
            "`pop` will be available in the future. Use"
            " `pop_latest` to retrieve the latest rendered frame."
        )

    @_requires_frame_sink
    def pop_latest(self) -> types.Image | None:
        '''
        Returns the latest frame rendered.

        Note any intermediary frames are discarded. To access all frames
        without discarding any use pop() instead. (which may NotImplementedError)

        Returns:
            1. The latest rendered frame
            2. None, if no frames have been pushed since the last call
        Raises:
            RuntimeError: If access to the contents of the surface was not exposed at initialization.
        '''
        self._check_running()
        return self._frame_sink.pop_latest()

    def _print_mem_usage(self):
        if LOG.isEnabledFor(logging.DEBUG) and time.time() - self._last_print > 2:
            minfo = self._self_proc.memory_info()
            system = psutil.virtual_memory().used
            qsize = self._queue.qsize() if self._queue is not None else 0
            LOG.debug(
                f"System memory: {system / 1024 ** 2:.2f} MB\t"
                f"axelera: {minfo.rss / 1024 ** 2:.2f} MB, vms = {minfo.vms / 1024 ** 2:.2f} MB\t"
                f"display queue size: {qsize}"
            )
            self._last_print = time.time()

    def push(self, image: types.Image, meta: AxMeta | None = None, source_id: int = 0) -> None:
        '''
        Send the given image and corresponding metadata and source id to the rendering queue.

        The image may not be dropped and not rendered if the rendering thread cannot keep up.

        Args:
            image:     The image to render onto
            meta:      The AxMeta metadata corresponding to the image, or None if there is no metadata.
                       the draw methods of the metadata entries will be called by the rendering thread.
            source_id: The id of the source the image and metadata belong to. This is used to
                       determine which panel the image should be rendered in, and which layers
                       belong to the source. Default is 0.
        '''
        self._check_running()
        _validate_image(image)
        if self._queue is not None:
            try:
                self._queue.put_nowait(_Frame(source_id, image, meta))
                if self._warned_full:
                    LOG.trace("Display queue is no longer full")
                self._warned_full = False
            except queue.Full:
                now = time.time()
                level = LOG.trace
                if now - (self._warned_full_at) > 10:
                    level = LOG.warning
                    self._warned_full_at = now
                level("Display queue is full, dropping frame(s)")
                self._warned_full = True
                return
        self._print_mem_usage()

    @_requires_frame_sink
    def render(
        self, image: types.Image, meta: AxMeta | None = None, source_id: int = 0
    ) -> types.Image:
        '''
        Render the given frame result and return the rendered image.

        Args:
            image:     The image to render onto
            meta:      The AxMeta metadata corresponding to the image, or None if there is no metadata.
                       the draw methods of the metadata entries will be called by the rendering thread.
            source_id: The id of the source the image and metadata belong to. This is used to
                       determine which panel the image should be rendered in, and which layers
                       belong to the source. Default is 0.

        Returns:
            The image provided, with rendering added

        Raises:
            RuntimeError: If access to the contents of the surface was not exposed at initialization.
        '''
        self._check_running()
        _validate_image(image)
        LOG.trace("Adding blocking frame to display queue")
        self._queue.put(_BlockingFrame(source_id, image, meta))
        self._print_mem_usage()
        LOG.trace("Retrieving rendered frame from the _FrameSink")
        return self._frame_sink.pop_latest_blocking()


class Window(Surface):
    '''Created by `App.create_window` to display inference results.'''

    def __init__(
        self,
        q: Optional[queue.Queue],
        size: tuple[int, int],
        is_closed: threading.Event,
        frame_sink: Optional[_FrameSink] = None,
    ):
        super().__init__(q, size, frame_sink)
        self._is_closed = is_closed

    def show(self, image: types.Image, meta: AxMeta | None = None, stream_id: int = 0) -> None:
        '''Send a frame result to rendering queue for display in the window.'''
        self.push(image, meta, stream_id)

    @property
    def is_closed(self) -> bool:
        '''True if the window has been closed.'''
        return self._is_closed.is_set()

    def close(self):
        '''Close the window.'''
        if self._queue is not None:
            self._queue.put(SHUTDOWN)
        else:
            self._is_closed.set()

    def wait_for_close(self):
        LOG.info("stream has a single frame, close the window or press q to exit")
        if self._queue is not None and self._queue.empty():
            self._queue.put(THREAD_COMPLETED)
        while not self.is_closed:
            time.sleep(0.1)


def _safe_gl_import(package, app, display_env, error_on_fail=True):
    try:
        module = importlib.import_module(package, package="axelera.app")
        return getattr(module, app)
    except Exception as e:
        if display_env:
            msg = f"DISPLAY environment variable={display_env}"
        else:
            msg = "Please try exporting the environment variable DISPLAY=:0.0"
        msg = f"Failed to initialize OpenGL: {e!r}\n{msg}"
        if error_on_fail:
            raise RuntimeError(msg)
        LOG.warning(msg)


def _find_display_class(display: str | bool, opengl: config.HardwareEnable):
    from . import display_console, display_cv

    display_env = os.environ.get('DISPLAY')
    # take care not to import display_gl before checking for the backend availablity
    display = 'auto' if display is True else display
    display = 'none' if display is False else display

    if display in ('auto', 'opengl'):
        if display == 'auto' and opengl == config.HardwareEnable.detect:
            opengl = (
                config.HardwareEnable.enable
                if utils.is_opengl_available(config.env.opengl_backend)
                else config.HardwareEnable.disable
            )
        if display != 'auto' or opengl == config.HardwareEnable.enable:
            # if user explicilty requested opengl, we should not fallback to anything
            if gl_app := _safe_gl_import('.display_gl', 'GLApp', display_env, display == 'opengl'):
                return gl_app
        if display_env:
            return display_cv.CVApp
        if os.environ.get('LC_TERMINAL') == 'iTerm2':  # prefer iTerm2 if available
            return display_console.iTerm2App
        return display_console.ConsoleApp
    elif display == 'opencv':
        return display_cv.CVApp
    elif display == 'console':
        return display_console.ConsoleApp
    elif display == 'iterm2':
        return display_console.iTerm2App
    elif display == 'wx':  # Internal only - requires OpenGL
        return _safe_gl_import('._display_wx', 'WxApp', display_env)
    elif display != 'none':
        expect = "'auto', 'opengl', 'opencv', 'console', 'iterm2', 'none' or False"
        raise ValueError(f"Invalid display option: {display}, expect one of {expect}")
    return NullApp


class App:
    '''The App instance manages the windows and event queues.

    Because most UI frameworks require that windows be created in the main
    thread, any workload must be created in a sub thread using `start_thread`
    and then the event handling must be processed using `run`.

    For example:

        with display.App(renderer=args.display) as app:
            app.start_thread(main, (args, stream, app), name='InferenceThread')
            app.run(interval=1 / 10)
    '''

    Surface = Surface
    Window = Window
    SupportedOptions = Options

    def __init__(self, *args, **kwargs):
        self._wnds = []
        self._queues = []
        self._create_queue = queue.Queue()
        self._is_closed = threading.Event()
        self._thr = None
        self._run_thr = None
        self._running_in_main = False
        self._native_windows = False

    def __new__(
        cls,
        renderer: str | bool = False,
        opengl: config.HardwareEnable = config.HardwareEnable.detect,
        buffering=True,
        **kwargs,
    ):
        if visible := kwargs.pop('visible', None):
            txt = f'"{visible}"' if isinstance(visible, str) else str(visible)
            LOG.warning(
                f"display.App(visible={txt}) is deprecated, " f"please use renderer={txt} instead."
            )
            renderer = visible
        if cls is App:
            cls = _find_display_class(renderer, opengl)

        x = object.__new__(cls)
        x.__init__(buffering=buffering)
        return x

    def create_surface(self, size) -> Surface:
        '''Create a new Surface, with given size.

        A surface is used for rendering frames without a window. Frames can be
        retrieved from the surface instead of being displayed in a window.

        See display.Surface for documentation on rendering to the surface and
        accessing the rendered frame.

        This method can be called from any thread.

        Args:
            size: The size of the surface, as a tuple of (width, height) in pixels.
                   Use FULL_SCREEN for full screen.

        Returns:
            The created surface.
        '''
        # note that surfaces must be created in UI thread, so push to create Q
        self._queues.append(q := queue.Queue(maxsize=200))
        frame_sink = _FrameSink()
        self._create_queue.put_nowait((q, frame_sink, '', size))
        cls = type(self)
        return cls.Surface(q, size, frame_sink, self._run_background)

    def create_window(self, title: str, size: tuple[int, int], expose_surface=False) -> Window:
        '''Create a new Window, with given title and size.

        This method can be called from any thread.  The returned window may not
        be visible immediately.

        Args:
            title: The title of the window. (Not the title of the stream(s))
            size: The size of the window, as a tuple of (width, height) in pixels.
                   Use FULL_SCREEN for full screen.
            expose_surface: If True, the rendered frames will be retrievable in the
                            same way as using `create_surface`, in addition to being
                            displayed in the window.

        Returns:
            The created window.
        '''
        # note that windows must be created in UI thread, so push to create Q
        if self._run_thr is not None:
            raise RuntimeError(
                "Cannot create native Voyager windows when display is running in background."
            )
        self._native_windows = True
        self._queues.append(q := queue.Queue(maxsize=200))
        frame_sink = _FrameSink() if expose_surface else None
        self._create_queue.put_nowait((q, frame_sink, title, size))
        cls = type(self)
        return cls.Window(q, size, self._is_closed, frame_sink)

    def start_thread(self, target, args=(), kwargs={}, name=None):
        '''Start a worker thread.

        The thread starts immediately and is joined at the end of the `with`
        block.  Arguments are similar to the standard `thread.Thread`
        constructor.
        '''

        def _target():
            try:
                target(*args, **kwargs)
            except Exception as e:
                LOG.error('Exception in inference thread: %s', name, exc_info=True)
                if LOG.isEnabledFor(logging.DEBUG):
                    LOG.error(traceback.format_exc())
                else:
                    LOG.error(traceback.format_exception_only(type(e), e))
            finally:
                self._destroy_all_windows()

        thr = threading.Thread(target=_target, name=name)
        thr.start()
        self._thr = thr

    @property
    def has_thread_completed(self):
        '''True if the thread has completed.

        Specifically, this returns True if start_thread has been called and the
        worker thread has completed.

        Typically this is used from backends in the event handling code to
        shut down the app.
        '''
        return self._thr and not self._thr.is_alive()

    def run(self, interval=1 / 30):
        '''Start handling UI events.

        This function will not return until the thread has completed.
        '''
        if self._run_thr is not None:
            raise RuntimeError("Display already running in background")
        try:
            self._running_in_main = True
            self._run(interval)
        finally:
            self._is_closed.set()

    def _run_background(self, interval=1 / 30):
        '''Start handling UI events in a background thread.

        This function will return immediately. The UI thread will stop when
        the main thread exits, or when the thread started by `start_thread`
        completes.
        '''
        if self._running_in_main or self._run_thr is not None:
            return
        if self._native_windows:
            raise RuntimeError(
                "Cannot run display in background when using native Voyager windows, use app.run() instead."
            )
        self._run_thr = threading.Thread(target=self._run, args=(interval,))
        self._run_thr.start()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if self._thr is not None:
            self._thr.join()
        if self._run_thr is not None:
            self._run_thr.join()

    def _create_new_windows(self):
        while True:
            try:
                args = self._create_queue.get(block=False)
            except queue.Empty:
                break
            else:
                self._wnds.append(self._create_new_window(*args))


class FontFamily(enum.Enum):
    sans_serif = enum.auto()
    serif = enum.auto()


@dataclass
class Font:
    family: FontFamily = FontFamily.sans_serif
    size: int = 12
    bold: bool = False
    italic: bool = False

    @property
    def weight(self):
        return 'bold' if self.bold else 'normal'


Point = tuple[int, int]
Color = tuple[int, int, int, int]
OptionalColor = Color | None


def midpoint(a: Point, b: Point) -> Point:
    '''Return the midpoint between two points.'''
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


class Draw(abc.ABC):
    @property
    @abc.abstractmethod
    def options(self) -> Options:
        '''Return the options for the renderer.'''

    @property
    @abc.abstractmethod
    def canvas_size(self) -> Point:
        '''Return the (width, height) of the canvas in pixels.'''

    @property
    @abc.abstractmethod
    def image_size(self) -> Point:
        '''Return the unscaled (width, height) of the input image in pixels.'''

    @abc.abstractmethod
    def polylines(
        self,
        lines: Sequence[Sequence[Point]],
        closed: bool = False,
        color: Color = (255, 255, 255, 255),
        width: int = 1,
    ) -> None:
        '''Draw a series of lines, with the given color.

        Args:
            lines: a sequence of polylines to draw. Each polyline is a
              sequence of Points.
            closed: If True, then the first and last points of each polyline
              are connected.
            color: The color to use for the rendering of the lines.
            width: Width of the lines in pixels.
        '''

    @abc.abstractmethod
    def rectangle(
        self,
        p1: Point,
        p2: Point,
        fill: OptionalColor = None,
        outline: OptionalColor = None,
        width: int = 1,
    ) -> None:
        '''Draw a rectangle from p1 to p2, inclusive, with the given fill and outline.

        Args:
            p1: First point of the rectangle (typically top left)
            p2: Opposite point of the rectangle (typically bottom right)
            fill:  The color to use for the fill of the rectangle, or None for unfilled.
            outline:  The color to use for the border of the rectangle, or None for no border.
            width: The width of the border of rectangle (if outline is not None)
        '''

    @abc.abstractmethod
    def textsize(self, text: str, font: Font = Font()) -> tuple[int, int]:
        '''Return the size of the text in pixels, given the font.'''

    @abc.abstractmethod
    def text(
        self,
        p: Point,
        text: str,
        txt_color: Color,
        back_color: OptionalColor = None,
        font: Font = Font(),
    ) -> None:
        '''Draw the text at the given point, with the given colors and font.'''

    @abc.abstractmethod
    def keypoint(self, p: Point, color: Color = (255, 255, 255, 255), size=1) -> None:
        """Draw a point/dot/marker at given position of the given color.

        Args:
            p: Center coordinate of the point.
            color: The color to use.
            size: Size of point to display in pixels.
        """

    @abc.abstractmethod
    def draw(self) -> None:
        '''Render all of the items to the output, called by the UI system.

        This may be called multiple times by the display window, for example
        when an invalidation occurs or a resize event occurs.
        '''

    @abc.abstractmethod
    def heatmap(self, data: np.ndarray, color_map: np.ndarray) -> None:
        '''Overlay a heatmap mask on the image. `data` is expected to be
        an float32 np.ndarray of the same size as the image. With values for
        each pixel between 0-1. Float values will be mapped to the nearest color
        in the color map, with lower values choosing from the start, and higher
        values choosing from the end.
        '''

    @abc.abstractmethod
    def segmentation_mask(
        self, mask: np.ndarray, master_box: np.array, color: tuple[int, ...]
    ) -> None:
        '''Overlay a mask on the image.
        If resize_to_input is True, the mask is expected to be resized to the input size.
        '''

    @abc.abstractmethod
    def class_map_mask(self, class_map: np.ndarray, color_map: np.ndarray) -> None:
        '''Overlay a class map mask on the image.
        `class_map` is expected to be a numpy array where each pixel value is the class ID
        of what is detected in that pixel. The mask will be resized to the input size.
        '''

    @abc.abstractmethod
    def draw_image(self, image: np.ndarray) -> None:
        '''Draw an image on the canvas.'''

    def labelled_box(self, p1: Point, p2: Point, label: str, color: OptionalColor = None) -> None:
        """Draw a labelled bounding box in the best way for the renderer.

        The default implementation is to show an unfilled box with the label on
        top of the box if it fits, or else inside the box.  But the
        implementation may choose to render it completely differently or even
        make it an interactive UI element.
        """
        txt_color = 244, 190, 24, 255  # axelera orange
        label_back_color = 0, 0, 0, 255
        line_width = max(1, self.canvas_size[1] // 640)
        font = Font(size=max(self.canvas_size[1] // 50, 12))
        if color is not None:
            self.rectangle(p1, p2, outline=color, width=line_width)
        if label:
            _, text_height = self.textsize('Iy', font)
            text_box_height = text_height + line_width + 1
            label_outside = p1[1] - text_box_height >= 0
            top = p1[1] - text_box_height if label_outside else p1[1]
            self.text((p1[0], top + 1), label, txt_color, label_back_color, font)

    def labelled_polygon(
        self, points: Sequence[Point], label: str, color: OptionalColor = None
    ) -> None:
        """Draw a labelled polygon in the best way for the renderer.
        The default implementation is to show an unfilled polygon with the label on
        top of the polygon if it fits, or else inside the polygon.  But the
        implementation may choose to render it completely differently or even
        make it an interactive UI element.
        """

        txt_color = 244, 190, 24, 255  # axelera orange
        label_back_color = 0, 0, 0, 255
        line_width = max(1, self.canvas_size[1] // 240)
        font = Font(size=max(self.canvas_size[1] // 50, 12))
        if color is not None:
            self.polylines([points], closed=True, color=color, width=line_width)
        if label:
            _, text_height = self.textsize('Iy', font)
            text_box_height = text_height + line_width + 1
            xs, ys = zip(*points)
            min_x, max_x = min(xs), max(xs)
            min_y = min(ys)
            centroid_x = int(sum(xs) / len(xs))
            label_outside = min_y - text_box_height >= 0
            top = min_y - text_box_height if label_outside else min_y
            left = min(max(min_x, centroid_x - len(label) * font.size // 4), max_x)
            self.text((left, top + 1), label, txt_color, label_back_color, font)

    def trajectory(self, bboxes, color):
        """Draw the trajectory of an object based on its bounding boxes.

        Args:
            bboxes (2D np.array or list of lists): List of bounding boxes (x1, y1, x2, y2).
            color (tuple): Color of the trajectory line.
        """
        if isinstance(bboxes, list):
            bboxes = np.array(bboxes)

        if bboxes.shape[0] < 2:
            return

        tracked_bboxes = bboxes[np.any(bboxes != 0, axis=1)]

        # Calculate midpoints of the bottom edge of bounding boxes
        mid_x = (tracked_bboxes[:, 0] + tracked_bboxes[:, 2]) / 2
        bottom_y = tracked_bboxes[:, 3]

        footprints = np.stack((mid_x, bottom_y), axis=-1).astype(int).reshape(-1, 2)
        footprints = [footprints.tolist()]  # Convert to a list of lists format for polylines

        self.polylines(footprints, color=color, width=2)


class NullApp(App):
    def create_window(self, title, size) -> Window:
        del title  # unused
        del size  # unused
        return Window(None, (None, None), self._is_closed)

    def start_thread(self, target, args=(), kwargs={}, name=None):
        del name  # unused
        target(*args, **kwargs)

    def _run(self, interval=1 / 30):
        del interval  # unused

    def _destroy_all_windows(self):
        pass


class SpeedometerSmoothing:
    '''
    The SpeedometerSmoothing class is used smoothly change speedometer values of
    a metric, simulating a real speedometer.

    It maintains state for each metric, keyed based on the metric's key and title.
    The basic pattern of usage is to create an instance of this class, then whenever
    you need a new value, call `update`, followed by `value` to get it.

    The smoothing is time based.

    This class can also be used to add a wobble effect to the speedometer needle,
    at higher values.

    Default contruction doesn't require any arguments, but you can configure the
    speed/smoothness of motion with the constructor arguments.

    Args:
        dps - Refers to difference per second. It is the maximum percent (as float)
              of the different between the last value and the target value that
              the speedometer will change in one second. For example, if the last
              value stored is 60, the target value is 100, and the dps is 3.0, the
              maximum change in one second will be 120 (300% of 40). Or in other
              words, the speedometer will take about 1/3 of a second to reach the
              target value. This is usefuly to allow the speedometer to slow down
              as it approached the target value, but go faster if it is far away.
        acceleration - The maximum increase from the last speed in one second, as
              a percentage (float) of the last speed. For example, if the last speed
              is 1.0, and the acceleration is 0.75, the maximum speed which can be
              reached in one second is 1.75 (1.0 + 75% of 1.0). This is useful to
              prevent the speedometer from jumping too fast in response to a
              suddent change in the metric value.
        minimum_speed - The minimum speed in percent (float) of the maximum value
              that the speedometer will change in one second. This is useful to
              allow the speedometer to move at a reasonable pace after it as
              slowed down, preventing long accelerations due to very low last
              speed. For example, if the maximum value is 100, and the
              minimum_speed is 0.015, the speedometer will change at least 1.5
              per second in this case.
        wobble_thresh - The threshold percentage of the maximum value (float)
              at which the needle will start to wobble. For example, if the maximum
              is 100 and the wobble_thresh is 0.8, the wobble will start at 80.
        max_wobble - The maximum wobble in degrees (int) that the needle will wobble
              +/- between. When the value is above the wobble_thresh. The wobbling
              will increase linearly from 0 to max_wobble as the value goes
              from wobble_thresh to max value.
    '''

    def __init__(
        self,
        dps: float = 3.0,
        acceleration: float = 0.5,
        minimum_speed: float = 0.015,
        wobble_thresh: int = 0.8,
        max_wobble: int = 5,
    ):
        self._dps = dps
        self._acceleration = acceleration
        self._wobble_thresh = wobble_thresh
        self._max_wobble = max_wobble
        self._minimum_speed = minimum_speed
        self._interval = 1 / 1000  # Default update interval in seconds
        self._last = {}
        self._last_max = {}

    def _key(self, metric: inf_tracers.TraceMetric) -> str:
        return f"{metric.key}-{metric.title}"

    def _value(self, metric: inf_tracers.TraceMetric) -> tuple[float, float, float]:
        key = self._key(metric)
        if key not in self._last:
            self._last[key] = (0.0, None, self._minimum_speed * metric.max_scale_value)
        return self._last[key]

    def _max(self, metric: inf_tracers.TraceMetric) -> tuple[float, float, float]:
        key = self._key(metric)
        if key not in self._last_max:
            self._last_max[key] = (
                metric.max_scale_value,
                self._minimum_speed * metric.max_scale_value,
            )
        return self._last_max[key]

    def value(self, metric: inf_tracers.TraceMetric) -> float:
        '''Get the most recent calculated value for the metric.'''
        return self._value(metric)[0]

    def max(self, metric: inf_tracers.TraceMetric) -> float:
        '''Get the most recent calculated maximum value for the metric.'''
        return self._max(metric)[0]

    def wobble(self, value, max) -> float:
        '''Calculate the wobble (in degrees) for the given value and max value.'''
        if value > self._wobble_thresh * max:
            wobble = (
                (value - self._wobble_thresh * max) / (max - self._wobble_thresh * max)
            ) * self._max_wobble
            return random.uniform(-wobble, wobble)
        return 0.0

    def _integrate(self, value: float, target: float, metric_max: float, speed: float):
        delta = abs(target - value)
        min_speed = self._minimum_speed * self._interval * metric_max
        max_diff = self._dps * self._interval * delta
        speed += self._acceleration * self._interval
        speed = max(min(max_diff, speed), min_speed)
        change = min(delta, speed)
        if target > value:
            value += change
        else:
            value -= change
        return value, speed

    def update(self, metric: inf_tracers.TraceMetric, now: Optional[float] = None):
        '''
        Update the given metric value.

        Several possible differences will be calculated:
        1. The difference based on `dps`
        2. The difference based on the previous speed + acceleration since last update
        3. The minimum speed based on the metric's maximum value
        4. The difference between the target and last value

        Of these four values, the smallest will be used to update the value, to ensure
        smoothness.

        Args:
            metric: The metric to update.
            now: Optional time to use for the update, otherwise current time is used.
        '''
        now = now or time.time()
        key = self._key(metric)
        value, update, speed = self._value(metric)
        metric_max, m_speed = self._max(metric)
        if update is None:
            self._last[key] = (value, now, speed)
            return
        elapsed = now - update
        target_max = metric.max_scale_value
        while elapsed >= self._interval:
            metric_max, m_speed = self._integrate(metric_max, target_max, metric_max, m_speed)
            value, speed = self._integrate(value, metric.value, metric_max, speed)
            elapsed -= self._interval
        self._last[key] = (value, now, speed)
        self._last_max[key] = (metric_max, m_speed)


def calculate_speedometer_text(
    metric: inf_tracers.TraceMetric, smoothing: SpeedometerSmoothing = None
) -> str:
    value = metric.value if not smoothing else smoothing.value(metric)
    metric_max = metric.max_scale_value if not smoothing else smoothing.max(metric)
    if value == 0.0:
        text = '  --'
    else:
        text = f'{round(value, 1):-5}' if metric_max < 100.0 else f'{round(value):-4}'
    text += metric.unit
    return text


def calculate_speedometer_needle_pos(
    metric: inf_tracers.TraceMetric, smoothing: SpeedometerSmoothing = None
) -> int:
    limits = 90 - 45, 270 + 45
    dial_range = limits[1] - limits[0]
    value = metric.value if not smoothing else smoothing.value(metric)
    metric_max = metric.max_scale_value if not smoothing else smoothing.max(metric)
    angle = limits[0] + (value / max(1, metric_max)) * dial_range
    if smoothing:
        angle += smoothing.wobble(value, metric_max)
    # input is angle in degrees clockwise from 6 o'clock
    # mapped to ImageDraw angles starting from 3 o'clock
    return (angle + 90) % 360


class SpeedometerMetrics:
    needle_color = (255, 0, 0, 255)
    text_color = (255, 255, 255, 255)

    def __init__(self, canvas_size, index):
        _size = min(canvas_size)
        self.radius = _size // 10
        self.diameter = self.radius * 2
        xoffset = index * round(self.diameter * 1.1)
        self.bottom_left = (_size // 20 + xoffset, _size - _size // 20)
        self.needle_radius = self.radius - _size // 48
        self.text_offset = self.radius * 2 // 5
        self.text_size = _size // 35

    @property
    def center(self):
        return (self.bottom_left[0] + self.radius, self.bottom_left[1] - self.radius)

    @property
    def top_left(self):
        return (self.bottom_left[0], self.bottom_left[1] - self.diameter)


def gen_title_message(stream_id, options):
    if isinstance(options.title_position, str):
        position = Coords(options.title_position)
    else:
        position = Coords(*options.title_position)
    return _Text(
        stream_id,
        uuid.uuid4(),
        position,
        options.title_anchor_x,
        options.title_anchor_y,
        None,
        None,
        None,
        None,
        options.title,
        options.title_color,
        options.title_bgcolor,
        options.title_size,
    )


class MetaCache:
    '''When inference skip frames is enabled, meta will be None on those frames
    that are skipped.

    This cache stores the last non-None meta for each stream.
    '''

    def __init__(self):
        self._last = {}

    def get(
        self, stream_id, meta_map: Mapping[str, AxTaskMeta] | None
    ) -> tuple[bool, Mapping[str, AxTaskMeta]]:
        '''Return the meta for given stream as (cached, meta_map).

        If meta is valid then the cache is updated. Otherwise the cached value is returned.

        Correctly handles stream_id which includes `__fps__` and other special metas.
        '''
        cached = False
        if stream_id == 0 and meta_map:
            # stream_id 0 always contains fps/latency etc. so we need to separate that
            meta_metas = {k: v for k, v in meta_map.items() if k.startswith('__')}
            actual_metas = {k: v for k, v in meta_map.items() if not k.startswith('__')}
            if actual_metas:
                meta_map = self._last[stream_id] = actual_metas
            else:
                meta_map = self._last.get(stream_id, {})
                cached = True
            meta_map = {**meta_metas, **meta_map}

        elif meta_map:
            meta_map = self._last[stream_id] = meta_map
        else:
            meta_map = self._last.get(stream_id, {})
            cached = True
        return cached, meta_map


@dataclass
class Canvas:
    '''A logical canvas defines the relationship between image coordinates and the window
    coordinates. Information provided is used to scale and position the image from the
    application as it is compressed to fit the window.

    For example, the incoming frame may be 1024x768(1.33AR) pixels, but the display window may be
    600x500.  The image is scaled to fit the window whilst maintaining aspect ratio, and so the
    image is squashed to (600, 600/1.33=450) pixels. (With 25 pixels top and bottom). In this case
    Canvas will be Canvas(0, 25, 600, 450, 600/1024=0.5859, ).
    '''

    left: int
    '''Left of the canvas in the coordinate system.'''
    bottom: int
    '''Bottom of the canvas in the coordinate system.'''
    width: int
    '''Width of the canvas in coordinate pixels.'''
    height: int
    '''Height of the canvas in coordinate pixels.'''
    scale: float
    '''Conversion factor from logical (image) coordinates to window coordinate space.'''

    window_width: int
    '''Width of the window in coordinate pixels.'''
    window_height: int
    '''Height of the window in coordinate pixels.'''

    @property
    def size(self) -> Tuple[int, int]:
        '''Size of the canvas in coordinate pixels.'''
        return self.width, self.height

    @property
    def window_size(self) -> Tuple[int, int]:
        '''Size of the owning window in coordinate pixels'''
        return self.window_width, self.window_height


def _get_layout(
    stream_id: int, num_sources: int, aspect: float
) -> tuple[float, float, float, float]:
    layouts = {
        3: (0.5, 0.5, [(0.0, 0.0), (0.5, 0.25), (0.0, 0.5)]),
        5: (0.4, 0.4, [(0.0, 0.0), (0.6, 0.0), (0.3, 0.3), (0.0, 0.6), (0.6, 0.6)]),
    }
    try:
        w, h, positions = layouts[num_sources]
        return positions[stream_id] + (w, h)
    except KeyError:
        cols = math.ceil(math.sqrt(num_sources))
        rows = math.ceil(num_sources / cols)
        if aspect < 1.0:
            cols, rows = rows, cols
        x, y = (stream_id % cols) / cols, (stream_id // cols) / rows
        return x, y, 1 / cols, 1 / rows


def _fit_within_rect(image: tuple[int, int], bounding: tuple[int, int]):
    imgr = image[0] / image[1]
    wndr = bounding[0] / bounding[1]
    if imgr > wndr:
        new_width = bounding[0]
        new_height = int(new_width / imgr)
    else:
        new_height = bounding[1]
        new_width = int(new_height * imgr)
    return new_width, new_height


def pane_position(
    stream_id: int, num_streams: int, image: tuple[int, int], window: tuple[int, int]
) -> tuple[float, float, float, float]:
    x, y, w, h = _get_layout(stream_id, num_streams, window[0] / window[1])
    x *= window[0]
    y *= window[1]
    bounding_box = w * window[0], h * window[1]
    w, h = _fit_within_rect(image, bounding_box)
    x += (bounding_box[0] - w) / 2
    y += (bounding_box[1] - h) / 2
    return x, y, w, h


def get_layers(all_layers, stream_id):
    '''
    Delete any layers if necessary, and return the layers to be rendered by the
    requested window.

    Layers with a stream_id of -1 will be rendered relative to the window.
    Other layers will be rendered relative to the image in the window corresponding to
    that stream ID.

    Stream 0 will render window layers, as well as stream 0 layers.
    '''
    _stream_ids = (stream_id, -1) if stream_id == 0 else (stream_id,)
    layers, expired = _get_visible_and_expired_layers(all_layers, _stream_ids)
    for k in expired:
        all_layers.pop(k)
    return layers


def canvas_scale_to_img_scale(scale, image_size, canvas_size):
    '''
    Get the factor by which you need to scale an image to to be the same
    size as the (scale * canvas size)

    For example, a scale of 0.1 for a (2000x1000) image on a (800x400) canvas
    would return 0.04, resulting in a (80x40) image. Thus, the width is 0.1 of
    the canvas width.

    The scaling factor is obtained from the largest dimension of the image to
    maintain aspect ratio.
    '''
    if scale is not None:
        img_x, img_y = image_size
        canvas_x, canvas_y = canvas_size
        if img_x >= img_y:
            max_img_dim = img_x
            max_target_dim = canvas_x
        else:
            max_img_dim = img_y
            max_target_dim = canvas_y
        return scale * max_target_dim / max_img_dim
    return 1.0

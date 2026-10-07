# Copyright Axelera AI, 2023
"""Thin app-side layer over the ``axelera.runtime2`` display stack.

The authoritative renderer implementations - the base classes and the OpenCV, console
and Vulkan renderers - now live in ``axelera.runtime2.display`` and are re-exported here
for backwards compatibility. New code should prefer ``axelera.runtime.display``.

Some small app specific segments remain here to support display_gl.

``axelera.runtime2`` is an optional dependency: it is required to render, but not to build
or deploy a pipeline. When it is absent this module still imports and :data:`HAVE_RUNTIME2`
is False; every route into a renderer goes through :func:`_find_display_class`, which
raises ImportError naming the missing dependency.
"""
from __future__ import annotations

# The runtime2 re-export comes first, before this module's own imports and definitions, so
# that everything below always wins: a name added to axelera.runtime2.display can shadow
# nothing here.
try:
    from axelera.runtime2.display import *  # noqa: F403
    from axelera.runtime2.display import (  # underscore names are not covered by ``import *``
        _BlockingFrame,  # noqa: F401
        _ClearState,  # noqa: F401
        _CloseSource,  # noqa: F401
        _Frame,  # noqa: F401
        _Image,  # noqa: F401
        _Layer,  # noqa: F401
        _OpenSource,  # noqa: F401
        _Rectangle,  # noqa: F401
        _SetOptions,  # noqa: F401
        _StreamMessage,  # noqa: F401
        _Text,  # noqa: F401
        _scale_visibility,  # noqa: F401
    )
    import axelera.runtime2.display as _rt2_display
except ImportError as e:
    _rt2_display = None
    _rt2_import_error = e
else:
    _rt2_import_error = None

from collections.abc import Mapping
import enum
import importlib
import os
from typing import TYPE_CHECKING

from . import config, logging_utils, utils

if TYPE_CHECKING:
    from .meta import AxTaskMeta

LOG = logging_utils.getLogger(__name__)

HAVE_RUNTIME2 = _rt2_display is not None

RUNTIME2_REQUIRED = (
    'the display renderers require axelera.runtime2, which is not installed; it is an '
    'optional dependency, needed to render but not to build or deploy a pipeline'
)


def __getattr__(name):
    '''Explain the missing optional dependency when a re-exported name is not bound.

    Only reached without runtime2 - when it is installed the ``import *`` above has already
    bound every re-exported name. This raises AttributeError rather than ImportError so
    that attribute probing still behaves: ``hasattr`` returns False, ``getattr`` with a
    default returns the default, and the import machinery's dunder lookups work. The
    dependency is named in the message, and the routes that actually need a renderer -
    :func:`_find_display_class` and the renderer modules - raise ImportError themselves.
    '''
    msg = f"module {__name__!r} has no attribute {name!r}"
    if not HAVE_RUNTIME2 and not name.startswith('__'):
        msg = f"{msg}; note that {RUNTIME2_REQUIRED}"
    raise AttributeError(msg)


# Without runtime2 there is nothing to derive from; the app stays importable and every route
# into a renderer goes through _find_display_class, which reports the missing dependency.
_AppBase = _rt2_display.App if HAVE_RUNTIME2 else object


class FrameStyle(enum.Enum):
    NORMAL = 'normal'
    GRID = 'grid'
    SIDE_BY_SIDE = 'side_by_side'


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


_NO_DISPLAY_ENV = "No display found, please export the environment variable DISPLAY=:0.0"


def _safe_gl_import(package, app, display_env, error_on_fail=True):
    try:
        module = importlib.import_module(package, package="axelera.app")
        return getattr(module, app)
    except Exception as e:
        if HAVE_RUNTIME2:
            if display_env:
                msg = f"DISPLAY environment variable={display_env}"
            else:
                msg = "Please try exporting the environment variable DISPLAY=:0.0"
            msg = f"Failed to initialize OpenGL: {e!r}\n{msg}"
        else:
            # OpenGL itself lives here, but it draws an axelera.runtime2 image.
            msg = f"Cannot use the OpenGL renderer: {RUNTIME2_REQUIRED}"
        if error_on_fail:
            raise RuntimeError(msg)
        LOG.warning(msg)


def _renderer(module: str, name: str):
    '''Import a renderer class by name, reporting an absent runtime2 as the optional dep.

    Every renderer other than OpenGL lives in ``axelera.runtime2``, as does ``NullApp``, so
    this is the single place that turns its absence into an actionable message.
    '''
    try:
        return getattr(importlib.import_module(module, package="axelera.app"), name)
    except ImportError as e:
        if HAVE_RUNTIME2:
            raise
        raise ImportError(f"{name} is unavailable: {RUNTIME2_REQUIRED}") from e


def _find_display_class(display: str | bool, opengl: config.HardwareEnable):
    display_env = os.environ.get('DISPLAY')
    # take care not to import display_gl before checking for the backend availability
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
            # if user explicitly requested opengl, we should not fallback to anything
            if gl_app := _safe_gl_import('.display_gl', 'GLApp', display_env, display == 'opengl'):
                return gl_app
        if display_env:
            return _renderer('.display_cv', 'CVApp')
        LOG.warning(_NO_DISPLAY_ENV)
        if os.environ.get('LC_TERMINAL') == 'iTerm2':  # prefer iTerm2 if available
            return _renderer('.display_console', 'iTerm2App')
        return _renderer('.display_console', 'ConsoleApp')
    elif display == 'opencv':
        cv_app = _renderer('.display_cv', 'CVApp')
        # The OpenCV renderer needs a window, so say so here rather than let cv2 fail later.
        if not display_env:
            raise RuntimeError(_NO_DISPLAY_ENV)
        return cv_app
    elif display == 'console':
        return _renderer('.display_console', 'ConsoleApp')
    elif display == 'iterm2':
        return _renderer('.display_console', 'iTerm2App')
    elif display == 'wx':  # Internal only - requires OpenGL
        return _safe_gl_import('._display_wx', 'WxApp', display_env)
    elif display == 'vulkan':
        return _renderer('.display_vk', 'VKApp')
    elif display != 'none':
        expect = "'auto', 'opengl', 'opencv', 'console', 'iterm2', 'vulkan', 'none' or False"
        raise ValueError(f"Invalid display option: {display}, expect one of {expect}")
    # NullApp renders nothing but is still a runtime2 class, so it is resolved the same way.
    return _renderer('axelera.runtime2.display', 'NullApp')


class App(_AppBase):
    '''App factory that keeps OpenGL as the app default and delegates the other
    renderers (opencv/console/iterm2/vulkan/none) to ``axelera.runtime2``.

    See ``axelera.runtime2.display.App`` for the full instance API; this subclass only
    customises renderer selection via the app-local :func:`_find_display_class`.
    '''

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
                f"display.App(visible={txt}) is deprecated, please use renderer={txt} instead."
            )
            renderer = visible
        if cls is App:
            cls = _find_display_class(renderer, opengl)

        x = object.__new__(cls)
        x.__init__(buffering=buffering)
        return x

# Copyright Axelera AI, 2023
import contextlib
import importlib
import os
from unittest.mock import patch

import pytest

pytest.importorskip('axelera.runtime2.display', reason='axelera.runtime2 is not installed')

from axelera.app import config, display, display_console, display_cv, meta, utils


def test_meta_cache():
    mc = display.MetaCache()
    meta0_0 = {'a': 10, 'b': 20, '__fps__': 30}
    meta1_0 = {'a': 11, 'b': 21}
    meta0_1 = {'__fps__': 31}
    meta0_2 = {'a': 110, 'b': 120, '__fps__': 32}
    meta1_2 = {'a': 111, 'b': 121}
    meta0_3 = {'__fps__': 33}
    assert (False, meta0_0) == mc.get(0, meta0_0)
    assert (False, meta1_0) == mc.get(1, meta1_0)
    assert (True, {'a': 10, 'b': 20, '__fps__': 31}) == mc.get(0, meta0_1)
    assert (True, {'a': 11, 'b': 21}) == mc.get(1, None)
    assert (False, {'a': 110, 'b': 120, '__fps__': 32}) == mc.get(0, meta0_2)
    assert (False, {'a': 111, 'b': 121}) == mc.get(1, meta1_2)
    assert (True, {'a': 110, 'b': 120, '__fps__': 33}) == mc.get(0, meta0_3)
    assert (True, {'a': 111, 'b': 121}) == mc.get(1, None)


disable = config.HardwareEnable.disable

detect = config.HardwareEnable.detect

enable = config.HardwareEnable.enable


@pytest.mark.parametrize(
    'display_opt, exp_class',
    [
        (False, display.NullApp),
        ('none', display.NullApp),
        ('console', display_console.ConsoleApp),
        ('iterm2', display_console.iTerm2App),
    ],
)
@pytest.mark.parametrize('detect_state', [disable, enable, detect])
def test_find_display_class_non_opengl(display_opt, exp_class, detect_state):
    #  opengl enable state should not affect opencv/console/none, nor should they try to detect it
    with patch.object(utils, 'is_opengl_available', return_value=False) as is_opengl_available:
        with patch.dict(os.environ, {'DISPLAY': ''}):
            assert display._find_display_class(display_opt, detect_state) is exp_class
        with patch.dict(os.environ, {'DISPLAY': 'something'}):
            assert display._find_display_class(display_opt, detect_state) is exp_class
    is_opengl_available.assert_not_called()


@pytest.mark.parametrize('detect_state', [disable, enable, detect])
def test_find_display_class_opencv(detect_state):
    with patch.object(utils, 'is_opengl_available', return_value=False) as is_opengl_available:
        with patch.dict(os.environ, {'DISPLAY': 'something'}):
            assert display._find_display_class('opencv', detect_state) is display_cv.CVApp
    is_opengl_available.assert_not_called()


def test_find_display_class_opencv_without_display_env():
    # the OpenCV renderer needs a window; say so rather than let cv2 fail later
    with patch.dict(os.environ, {'DISPLAY': ''}):
        with pytest.raises(RuntimeError, match='No display found'):
            display._find_display_class('opencv', detect)


def test_find_display_class_auto_warns_without_display_env(caplog):
    with patch.object(utils, 'is_opengl_available', return_value=False):
        with patch.dict(os.environ, {'DISPLAY': '', 'LC_TERMINAL': ''}):
            assert display._find_display_class('auto', disable) is display_console.ConsoleApp
    assert any('No display found' in x for x in caplog.messages)


def test_find_display_class_bad_option():
    with pytest.raises(ValueError, match='Invalid display option'):
        display._find_display_class('bad_option', detect)


def test_find_display_class_auto_opengl_disabled():
    with patch.object(utils, 'is_opengl_available', return_value=False) as is_opengl_available:
        with patch.dict(os.environ, {'DISPLAY': '', 'LC_TERMINAL': ''}):
            assert display._find_display_class('auto', disable) is display_console.ConsoleApp
        with patch.dict(os.environ, {'DISPLAY': '', 'LC_TERMINAL': 'iTerm2'}):
            assert display._find_display_class('auto', disable) is display_console.iTerm2App
        with patch.dict(os.environ, {'DISPLAY': 'something', 'LC_TERMINAL': ''}):
            assert display._find_display_class('auto', disable) is display_cv.CVApp
        with patch.dict(os.environ, {'DISPLAY': '', 'LC_TERMINAL': ''}):
            assert display._find_display_class(True, disable) is display_console.ConsoleApp
        with patch.dict(os.environ, {'DISPLAY': '', 'LC_TERMINAL': 'iTerm2'}):
            assert display._find_display_class(True, disable) is display_console.iTerm2App
    is_opengl_available.assert_not_called()


@contextlib.contextmanager
def _mock_display_gl_import(succeeds=True):
    # importing display_gl is a little risky, so we mock it
    # (we should probably make it less risky to import display_gl!!!)

    class MockDisplayGLModule:
        class GLApp:
            pass

    real_import_module = importlib.import_module

    def mock_import_module(name, package=None):
        if name == '.display_gl':
            if succeeds:
                return MockDisplayGLModule
            raise ImportError('display_gl import failed')
        # the runtime2 renderer modules are resolved the same way, so let them through
        return real_import_module(name, package)

    with patch('axelera.app.display.importlib.import_module', side_effect=mock_import_module):
        yield


def test_find_display_class_auto_opengl_enabled():
    # we should not try to detect opengl if explicitly requested or --enable-opengl specifed
    with patch.object(utils, 'is_opengl_available') as m:
        with _mock_display_gl_import():
            assert display._find_display_class('auto', enable).__name__ == 'GLApp'
            assert display._find_display_class('opengl', enable).__name__ == 'GLApp'
            assert display._find_display_class('opengl', detect).__name__ == 'GLApp'

    m.assert_not_called()


def test_find_display_class_gl_import_succeeds():
    with patch.object(utils, 'is_opengl_available', return_value=True) as m:
        with _mock_display_gl_import():
            assert display._find_display_class('auto', detect).__name__ == 'GLApp'

    m.assert_called_once()


def test_find_display_class_opengl_import_fails():
    with patch.dict(os.environ, {'DISPLAY': ''}):
        with patch.object(utils, 'is_opengl_available', return_value=True) as m:
            with _mock_display_gl_import(succeeds=False):
                with pytest.raises(RuntimeError, match=r'Failed to init.*\n.*try exporting'):
                    display._find_display_class('opengl', detect)
    with patch.dict(os.environ, {'DISPLAY': 'something'}):
        with patch.object(utils, 'is_opengl_available', return_value=True) as m:
            with _mock_display_gl_import(succeeds=False):
                with pytest.raises(RuntimeError, match=r'Failed to init.*\n.*something'):
                    display._find_display_class('opengl', detect)


def test_find_display_class_auto_import_fails(caplog):
    with patch.dict(os.environ, {'DISPLAY': ''}):
        with patch.object(utils, 'is_opengl_available', return_value=True) as m:
            with _mock_display_gl_import(succeeds=False):
                display._find_display_class('auto', detect)
    assert any('Failed to init' in x for x in caplog.messages)


class _Drawable:
    '''Stands in for a task meta: records the Draw it was given, and visits secondaries.'''

    def __init__(self, name, secondaries=()):
        self.name = name
        self.drawn_by = []
        self._secondaries = list(secondaries)

    def visit(self, callable, *args, **kwargs):
        callable(self, *args, **kwargs)
        for secondary in self._secondaries:
            secondary.visit(callable, *args, **kwargs)

    def draw(self, draw):
        self.drawn_by.append(draw)


class _VisitOnly(_Drawable):
    '''A member which can be visited but cannot draw itself.'''

    draw = None


def _axmeta(**members):
    m = meta.AxMeta('image')
    for key, value in members.items():
        m.add_instance(key, value)
    return m


def test_axmeta_is_drawn_by_the_renderer_traversal():
    # the renderers traverse whatever they are given with draw_recursive; an AxMeta is a
    # drawable in its own right, so no renderer needs to know about the metadata model
    primary = _Drawable('primary')
    other = _Drawable('other')
    draw = object()
    display.draw_recursive(_axmeta(a=primary, b=other), draw)
    assert primary.drawn_by == [draw]
    assert other.drawn_by == [draw]


def test_axmeta_draw_visits_secondary_metas():
    secondary = _Drawable('secondary')
    primary = _Drawable('primary', secondaries=[secondary])
    draw = object()
    _axmeta(a=primary).draw(draw)
    assert primary.drawn_by == [draw]
    assert secondary.drawn_by == [draw]


def test_axmeta_draw_skips_members_that_cannot_draw():
    drawable = _Drawable('drawable')
    draw = object()
    # a visitable member with no draw, and a member which is neither visitable nor drawable
    _axmeta(a=_VisitOnly('novisit'), b=object(), c=drawable).draw(draw)
    assert drawable.drawn_by == [draw]


def test_axmeta_draw_propagates_a_failure_in_a_member():
    class Boom(_Drawable):
        def draw(self, draw):
            raise AttributeError('boom')

    with pytest.raises(AttributeError, match='boom'):
        _axmeta(a=Boom('boom')).draw(object())

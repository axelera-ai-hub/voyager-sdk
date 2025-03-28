# Copyright Axelera AI, 2024
import os
from unittest.mock import ANY, Mock, call, patch

import cv2
import numpy as np

from axelera import types
from axelera.app import config, display, display_cv

OPENGL = config.HardwareEnable.disable


def test_window_creation():
    with patch.dict(os.environ, DISPLAY='somewhere'):
        with patch.object(display_cv, "cv2") as mcv:
            mcv.imread.return_value = np.full((32, 32, 3), (42, 43, 44), np.uint8)
            mcv.copyMakeBorder.return_value = np.full((480, 640, 3), (42, 43, 44), np.uint8)
            with display.App(visible=True, opengl=OPENGL) as app:
                wnd = app.create_window("test", (640, 480))
                app.start_thread(wnd.close)
                app.run()
                assert wnd.is_closed == True
    assert mcv.namedWindow.call_args_list == [call("test", mcv.WINDOW_NORMAL)]
    assert mcv.imshow.call_args_list == [call("test", ANY)]
    assert mcv.imshow.call_args_list[0].args[1].shape == (480, 640, 3)
    assert mcv.setWindowProperty.call_args_list == [
        call("test", mcv.WND_PROP_ASPECT_RATIO, mcv.WINDOW_KEEPRATIO)
    ]


def test_window_creation_fullscreen():
    with patch.dict(os.environ, DISPLAY='somewhere'):
        with patch.object(display_cv, "cv2") as mcv:
            with display.App(visible=True, opengl=OPENGL) as app:
                wnd = app.create_window("test", display.FULL_SCREEN)
                app.start_thread(wnd.close)
                app.run()
                assert wnd.is_closed == True
    assert mcv.namedWindow.call_args_list == [call("test", mcv.WINDOW_FULLSCREEN)]
    assert mcv.setWindowProperty.call_args_list == [
        call("test", mcv.WND_PROP_ASPECT_RATIO, mcv.WINDOW_KEEPRATIO)
    ]


def test_window_creation_with_data():
    with patch.dict(os.environ, DISPLAY='somewhere'):
        with patch.object(display_cv, "cv2") as mcv:
            mcv.waitKey.return_value = ord("q")
            mcv.getWindowProperty.return_value = 0.0
            with display.App(visible=True, opengl=OPENGL) as app:
                wnd = app.create_window("test", (640, 480))
                array = np.full((480, 640, 4), (42, 43, 44, 0), np.uint8)
                i = types.Image.fromarray(array)
                meta_map = {}
                app.start_thread(wnd.show, args=(i, meta_map))
                app.run()
                assert wnd.is_closed == True
    # note RGB -> BGR swap for opencv:
    exp = np.full((480, 640, 4), (44, 43, 42, 0), np.uint8)
    assert mcv.imshow.call_args_list == [
        call("test", ANY),
        call("test", ANY),
    ]
    np.testing.assert_equal(mcv.imshow.call_args_list[1].args[1], exp)


def test_window_creation_no_display_env_visible_diagnostic(caplog):
    import sys

    sys.modules['climage'] = None
    with patch.dict(os.environ, DISPLAY=''):
        with patch.object(display_cv, "cv2") as mcv:
            with display.App(visible=True, opengl=OPENGL) as app:
                wnd = app.create_window("test", (640, 480))
                app.start_thread(wnd.close)
                app.run()
                assert wnd.is_closed == True
    assert mcv.namedWindow.call_args_list == []
    assert 'WARNING' in caplog.text
    assert '--no-display' in caplog.text


def test_window_creation_no_display_env_no_visible_no_diagnostic(caplog):
    with patch.dict(os.environ, DISPLAY=''):
        with patch.object(display_cv, "cv2") as mcv:
            with display.App(visible=False, opengl=OPENGL) as app:
                wnd = app.create_window("test", (640, 480))
                app.start_thread(wnd.close)
                app.run()
                assert wnd.is_closed == True
    assert mcv.namedWindow.call_args_list == []
    assert '' == caplog.text

# Copyright Axelera AI, 2025
import os
import queue
from unittest.mock import ANY, Mock, call, patch

import numpy as np

from axelera import types
from axelera.app import config, display, display_cv

OPENGL = config.HardwareEnable.disable


def test_window_creation():
    with patch.dict(os.environ, DISPLAY='somewhere'):
        with patch.object(display_cv, "cv2") as mcv:
            mcv.getWindowProperty.return_value = 1.0
            mcv.imread.return_value = np.full((32, 32, 3), (42, 43, 44), np.uint8)
            mcv.copyMakeBorder.return_value = np.full((480, 640, 3), (42, 43, 44), np.uint8)
            with display.App(renderer=True, opengl=OPENGL) as app:
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
        with patch.object(display_cv, "cv2") as mcv, patch.object(
            display, "get_primary_screen_resolution", return_value=(1920, 1080)
        ):
            mcv.getWindowProperty.return_value = 1.0
            with display.App(renderer=True, opengl=OPENGL) as app:
                wnd = app.create_window("test", display.FULL_SCREEN)
                app.start_thread(wnd.close)
                app.run()
                assert wnd.is_closed == True
    assert mcv.namedWindow.call_args_list == [call("test", mcv.WINDOW_NORMAL)]
    assert mcv.setWindowProperty.call_args_list == [
        call("test", mcv.WND_PROP_FULLSCREEN, mcv.WINDOW_FULLSCREEN)
    ]


def test_window_creation_with_data():
    with patch.dict(os.environ, DISPLAY='somewhere'):
        import cv2 as real_cv2

        with patch.object(display_cv, "cv2") as mcv:
            mcv.waitKey.return_value = ord("q")
            mcv.getWindowProperty.return_value = 0.0
            mcv.resize = real_cv2.resize
            with display.App(renderer=True, opengl=OPENGL) as app:
                wnd = app.create_window("test", (640, 480))
                array = np.full((480, 640, 3), (42, 43, 44), np.uint8)
                i = types.Image.fromarray(array)
                meta_map = {}
                app.start_thread(wnd.show, args=(i, meta_map))
                app.run()
                assert wnd.is_closed == True
    # note RGB -> BGR swap for opencv:
    exp = np.full((480, 640, 3), (44, 43, 42), np.uint8)
    assert mcv.imshow.call_args_list == [
        call("test", ANY),
        call("test", ANY),
    ]
    np.testing.assert_equal(mcv.imshow.call_args_list[1].args[1], exp)


def test_layered_draw_list():
    dlist = display_cv._LayeredDrawList()
    dlist[0].text((10, 10), "Layer 0", (255, 255, 255, 255))
    dlist[1].rectangle(((20, 20), (30, 30)), (255, 0, 0, 255))
    dlist[2].line([40, 40, 50, 50], (0, 255, 0, 255), 2)
    dlist[-1].ellipse(((60, 60), (70, 70)), (0, 0, 255, 255))
    dlist[-2].paste("mock_img", (80, 80), "mock_mask")
    assert len(dlist) == 5
    expected = ["text", "rectangle", "line", "paste", "ellipse"]
    actual = [op[0] for op in dlist]
    assert actual == expected
    assert sorted(dlist.keys()) == [-2, -1, 0, 1, 2]


def test_layered_draw_list_access():
    dlist = display_cv._LayeredDrawList()
    dlist.text((10, 10), "Default", (255, 255, 255, 255))
    dlist[2].rect((20, 20, 30, 30), (255, 0, 0, 255))
    assert len(dlist[1]) == 1
    assert len(dlist[2]) == 1
    assert dlist[1][0][0] == "text"
    assert dlist[2][0][0] == "rect"
    assert len(dlist) == 2


def test_layered_draw_list_empty():
    dlist = display_cv._LayeredDrawList()
    assert list(dlist) == []
    assert len(dlist) == 0


def test_layered_draw_list_order():
    dlist = display_cv._LayeredDrawList()
    dlist[-5].op1(1)
    dlist[3].op2(2)
    dlist[0].op3(3)
    dlist[-2].op4(4)
    dlist[1].op5(5)
    expected = ["op3", "op5", "op2", "op1", "op4"]
    actual = [op[0] for op in dlist]
    assert actual == expected


def test_layered_draw_list_multiple_ops():
    dlist = display_cv._LayeredDrawList()
    dlist[0].op1(1)
    dlist[0].op2(2)
    dlist[1].op3(3)
    dlist[1].op4(4)
    dlist[2].op5(5)
    dlist[-2].op6(6)
    dlist[-2].op7(7)
    dlist[-1].op8(8)
    dlist[-1].op9(9)
    expected = ["op1", "op2", "op3", "op4", "op5", "op6", "op7", "op8", "op9"]
    actual = [op[0] for op in dlist]
    assert actual == expected
    assert len(dlist) == 9
    assert len(dlist[0]) == 2
    assert len(dlist[1]) == 2
    assert len(dlist[-1]) == 2


def test_cv_window_open_close_source():
    mock_frame = (Mock(), Mock(), False)
    new_frame = (Mock(), Mock(), False)
    wnd = display_cv.CVWindow(None, None, '', (640, 480))
    wnd._current = {0: mock_frame, 1: mock_frame, 2: mock_frame}
    assert not wnd._closed_sources
    wnd._handle_message(display._CloseSource(1, False))
    assert wnd._closed_sources == {1}
    assert wnd._current == {0: mock_frame, 2: mock_frame}
    wnd._handle_message(display._Frame(1, *new_frame[:2]))
    # check new frames are ignored
    assert wnd._current == {0: mock_frame, 2: mock_frame}
    wnd._handle_message(display._OpenSource(1))
    assert not wnd._closed_sources
    # After open message received - no restart until next frame received
    assert wnd._current == {0: mock_frame, 2: mock_frame}
    wnd._handle_message(display._Frame(1, *new_frame[:2]))
    assert wnd._current == {0: mock_frame, 1: new_frame, 2: mock_frame}


def test_cv_window_close_source_reopen():
    mock_frame = (Mock(), Mock(), False)
    new_frame = (Mock(), Mock(), False)
    wnd = display_cv.CVWindow(None, None, '', (640, 480))
    wnd._current = {0: mock_frame, 1: mock_frame, 2: mock_frame}
    assert not wnd._closed_sources
    wnd._handle_message(display._CloseSource(1, True))
    assert not wnd._closed_sources
    assert wnd._current == {0: mock_frame, 2: mock_frame}
    wnd._handle_message(display._Frame(1, *new_frame[:2]))
    assert wnd._current == {0: mock_frame, 1: new_frame, 2: mock_frame}


def test_cvdraw_warns_when_stream_id_exceeds_num_streams():
    array = np.zeros((480, 640, 3), dtype=np.uint8)
    composite = types.Image.fromarray(array)
    image = types.Image.fromarray(array)

    with patch.object(display_cv, 'LOG') as mock_log, patch.object(display_cv, 'cv2'):
        display_cv.CVDraw(1, 1, composite, image, [])

    mock_log.warning.assert_called_once_with(
        "Source ID 1 is too high for the number of sources (1), expect rendering glitches."
    )

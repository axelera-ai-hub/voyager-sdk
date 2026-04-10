# Copyright Axelera AI, 2026

# _display_wx.py provides a wxPython Frame which integrates with pyglet and the Axelera display
# system. To use - subclass WxGLFrame and build the wxPython window from there. Run using the
# normal mechanism (display.App(display='wx')...).
# The contents of this file are considered internal and experimental and no guarantees are made
# about its API.

import wx

from . import display
from .display_gl import GLWindow, pyglet


class WxGLFrame(wx.Frame):
    def __init__(self, app):
        super().__init__(
            None,
            title=app.frame_title,
            size=app.frame_min_size,
            style=wx.STAY_ON_TOP | wx.DEFAULT_FRAME_STYLE,
        )
        self._wnds = app._wnds
        self.__timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self.__on_timer, self.__timer)
        self.__app = app
        self._start_timer()
        self.Bind(wx.EVT_CLOSE, self.__on_close)
        self.Bind(wx.EVT_CHAR_HOOK, self.__on_key)

    def __on_timer(self, event):
        del event
        self.__app._create_new_windows()
        pyglet.clock.tick()
        for wnd in self._wnds:
            wnd.switch_to()
            wnd.dispatch_events()
        wx.YieldIfNeeded()

    def __on_key(self, event):
        if event.GetKeyCode() == wx.WXK_ESCAPE:
            self.__handover()
        else:
            event.Skip()

    def __handover(self):
        if self.__timer.IsRunning():
            self.__timer.Stop()
        self.Hide()
        wx.CallAfter(wx.GetApp().ExitMainLoop)

    def _start_timer(self):
        if not self.__timer.IsRunning():
            self.__timer.Start(int(self.__app.interval * 1000))

    def __on_close(self, event):
        self.__app._close = True
        self.__handover()


class WxApp(display.App):
    def __init__(self, *args, **kwargs):
        self.buffering = kwargs.pop('buffering', True)
        self.Frame = kwargs.pop('Frame', WxGLFrame)
        self.frame_options = kwargs.pop(
            'frame_options',
            {
                "frame_min_size": (800, 600),
                "frame_title": "Axelera",
            },
        )
        self.frame_min_size = self.frame_options["frame_min_size"]
        self.frame_title = self.frame_options["frame_title"]
        self._close = False
        self._frame = None  # created at runtime
        super().__init__(*args, **kwargs)

    def _create_new_window(self, q, frame_sink, title, size):
        del frame_sink
        return GLWindow(q, title, size, self.buffering, None, hard_stop=False, borderless=True)

    def _start(self):
        self._frame.Show()
        self._frame._start_timer()
        self._wx_app.MainLoop()

    def _run(self, interval=1 / 30):
        self._wx_app = wx.App(False)
        self.interval = interval
        self._frame = self.Frame(self)
        while not self._close:
            self._start()
            if not self._close:
                pyglet.app.run(interval=None)
        self._destroy_all_windows()

    def _destroy_all_windows(self):
        pyglet.app.exit()
        if self._frame:
            wx.CallAfter(self._frame.Destroy)

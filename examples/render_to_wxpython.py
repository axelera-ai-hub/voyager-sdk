#!/usr/bin/env python
# Copyright Axelera AI, 2025
"""Render an inference stream and display frames in a wxPython window.

This demo shows how to:
  * Run the display.App event loop (for compositing / drawing) in a background thread
    * Poll the latest rendered frame via Surface
    * Present frames in a wxPython window at ~60 FPS
    * Add UI controls: greyscale toggle, network selection, source selection, image preprocessor selection, and dynamic add/remove pipelines

Requirements:
    sudo apt install libgtk-3-dev
    pip install wxpython
"""

from __future__ import annotations

import os
import sys
import re
import threading

import numpy as np

try:
    import wx
except ImportError:
    sys.exit(
        "ERROR: wxPython is required to run this example.\n"
        "Please install it via 'pip install wxpython'."
    )

from axelera import types
from axelera.app import display
from axelera.app.stream import create_inference_stream

W = 900
H = W * 1080 // 1920
SURFACE_SIZE = (W, H)
POLL_INTERVAL_MS = 1000 // 30

NETWORKS = [
    "yolov5m-v7-coco-tracker",
    "yolov8s-coco",
    "yolov8spose-coco",
    "yolov8sseg-coco",
]


def _enum_usb_video_devices() -> list[str]:
    return sorted(
        f'usb:{m.group(1)}' for m in (re.match(r'video(\d+)', f) for f in os.listdir('/dev')) if m
    )


SOURCES = _enum_usb_video_devices()
SOURCES += sorted([f'loop:media/{f}' for f in os.listdir('media') if f.endswith('.mp4')])

IMAGE_PREPROCESSORS = [
    '',
    'rotate90',
    'rotate180',
    'rotate270',
    'horizontalflip',
    'verticalflip',
    'perspective[[1.019,-0.697,412.602,0.918,1.361,-610.083,0.0,0.0,1.0]]',
]

# Initial stream with a single pipeline
stream = create_inference_stream(
    network=NETWORKS[0],
    sources=[SOURCES[0]],
    aipu_cores=1,
    timeout=0,
    low_latency=True,
)

MAX_PIPELINES = 4

pending_pipelines = []


class WxViewer(wx.Frame):
    def __init__(
        self, app: display.App, size, stop: threading.Event, surfaces: list[display.Surface]
    ):
        super().__init__(parent=None, title="Render to Image (wx)", size=size)
        self._app = app
        self._stop = stop
        self._surfaces = surfaces
        self._sizer = wx.BoxSizer(wx.HORIZONTAL)
        self.show_annotations = True
        self.show_labels = True

        self._grid_sizer = wx.GridSizer(2, 2, 0, 0)
        self._bmp_ctrls = []
        w, h = SURFACE_SIZE
        black = wx.Bitmap.FromBuffer(w, h, np.zeros((h, w, 3), dtype=np.uint8))
        for _ in range(MAX_PIPELINES):
            bmpctrl = wx.StaticBitmap(self, -1, size=SURFACE_SIZE)
            self._grid_sizer.Add(bmpctrl, 1, wx.EXPAND)
            bmpctrl.SetBitmap(black)
            self._bmp_ctrls.append(bmpctrl)

        self._sizer.Add(self._grid_sizer, 1, wx.EXPAND)

        controls = wx.BoxSizer(wx.VERTICAL)

        def add_choices(title, choices):
            choice = wx.Choice(self, -1, choices=choices)
            choice.SetSelection(0)
            controls.Add(wx.StaticText(self, -1, title), 0, wx.ALL, 5)
            controls.Add(choice, 0, wx.ALL, 5)
            return choice

        self._network_choice = add_choices("Network:", NETWORKS)
        self._source_choice = add_choices("Source:", SOURCES)
        self._preproc_choice = add_choices("Preproc:", IMAGE_PREPROCESSORS)

        self._btn_add = wx.Button(self, -1, "Add")
        self._btn_remove = wx.Button(self, -1, "Remove")
        self._btn_add.Bind(wx.EVT_BUTTON, self._on_add)
        self._btn_remove.Bind(wx.EVT_BUTTON, self._on_remove)
        controls.Add(self._btn_add, 0, wx.ALL, 5)
        controls.Add(self._btn_remove, 0, wx.ALL, 5)

        # Render configuration checkboxes
        self._cb_labels = wx.CheckBox(self, -1, "Show labels")
        self._cb_labels.SetValue(self.show_labels)
        self._cb_annotations = wx.CheckBox(self, -1, "Show annotations")
        self._cb_annotations.SetValue(self.show_annotations)
        controls.Add(self._cb_annotations, 0, wx.ALL, 5)
        controls.Add(self._cb_labels, 0, wx.ALL, 5)
        controls.AddStretchSpacer()
        controls.Add(wx.StaticLine(self, -1, style=wx.LI_HORIZONTAL), 0, wx.EXPAND | wx.ALL, 5)

        self._sizer.Add(controls, 0, wx.ALL, 0)
        self._update_button_states()
        self.SetSizer(self._sizer)
        self._timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_timer, self._timer)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self._timer.Start(POLL_INTERVAL_MS)

        self._cb_annotations.Bind(wx.EVT_CHECKBOX, self._on_render_config_change)
        self._cb_labels.Bind(wx.EVT_CHECKBOX, self._on_render_config_change)
        for pipeline in stream.pipelines:
            pipeline.set_render(self.show_labels, self.show_annotations)

        self.Fit()
        self.Show()

    def _on_timer(self, _):
        if self._stop.is_set():
            self._on_close(wx.CloseEvent())
            return

        for n, s in enumerate(self._surfaces):
            if new := s.pop_latest():
                np_img = new.asarray(types.ColorFormat.RGB)
                h, w, _ = np_img.shape
                bmp = wx.Bitmap.FromBuffer(w, h, np_img)
                self._bmp_ctrls[n].SetBitmap(bmp)
        # Periodically check if buttons need state updates (in case external changes happen)
        self._update_button_states()

    def _on_close(self, evt):
        try:
            if self._timer.IsRunning():
                self._timer.Stop()
        finally:
            self._stop.set()
            self.Destroy()

    # --- Pipeline management UI logic ---
    def _on_add(self, _evt):
        if len(stream.pipelines) >= MAX_PIPELINES:
            return
        sel = self._network_choice.GetSelection()
        network = NETWORKS[sel] if sel != wx.NOT_FOUND else NETWORKS[0]
        src_sel = self._source_choice.GetSelection()
        source = SOURCES[src_sel] if src_sel != wx.NOT_FOUND else SOURCES[0]
        pre_sel = self._preproc_choice.GetSelection()
        preproc = IMAGE_PREPROCESSORS[pre_sel] if pre_sel != wx.NOT_FOUND else ''
        source = f"{preproc}:{source}" if preproc else source
        pending_pipelines.append((network, source))
        self._update_button_states()

    def _on_remove(self, _evt):
        if len(stream.pipelines) <= 1:
            return
        # Remove first pipeline (index 0) per requirement
        stream.remove_pipeline(0)
        self._update_button_states()

    def _update_button_states(self):
        num = len(stream.pipelines)
        self._btn_remove.Enable(num > 1)
        self._btn_add.Enable(num < MAX_PIPELINES)

    def _on_render_config_change(self, _evt):
        self.show_annotations = self._cb_annotations.GetValue()
        self.show_labels = self._cb_labels.GetValue()
        for pipeline in stream.pipelines:
            pipeline.set_render(self.show_labels, self.show_annotations)


def main(stream, stop: threading.Event, surfaces: list[display.Surface], wx_wnd: WxViewer):
    try:
        for frame_result in stream:
            if stop.is_set():
                break
            surfaces[frame_result.stream_id].push(frame_result.image, frame_result.meta)

            if pending_pipelines:
                network, source = pending_pipelines.pop(0)
                pipeline = stream.add_pipeline(
                    network=network, sources=[source], aipu_cores=1, low_latency=True
                )
                pipeline.set_render(wx_wnd.show_labels, wx_wnd.show_annotations)
                wx.CallAfter(wx_wnd._update_button_states)
    finally:
        stop.set()


with display.App(renderer='opencv') as app:
    surfaces = [app.create_surface(SURFACE_SIZE) for _ in range(MAX_PIPELINES)]
    stop = threading.Event()
    wx_app = wx.App(False)
    wx_wnd = WxViewer(app, (1200, 600), stop, surfaces)
    app.start_thread(main, (stream, stop, surfaces, wx_wnd), name="InferenceThread")
    wx_app.MainLoop()
stream.stop()

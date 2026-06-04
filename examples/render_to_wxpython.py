#!/usr/bin/env python
# Copyright Axelera AI, 2025
"""Render an inference stream and display frames in a wxPython window.

This demo shows how to:
  * Dynamically add / remove pipelines at runtime
  * Display OpenCV images rendered by the framework in wxPython

Requirements:
    sudo apt install libgtk-3-dev
    pip install wxpython
"""

from __future__ import annotations

import dataclasses
import enum
import os
import queue
import re
import subprocess
import sys
import threading
import time

import numpy as np

try:
    import wx
except ImportError:
    sys.exit(
        "ERROR: wxPython is required to run this example.\n"
        "Please install it via 'pip install wxpython'."
    )

from axelera import types

if __name__ == '__main__':
    # Application Framework is not a package, so add it to the path to import it
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from axelera.app import display
from axelera.app.pipe import FrameEventType
from axelera.app.stream import create_inference_stream

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

GRID_W = 900
GRID_H = GRID_W * 1080 // 1920
SLOT_W = GRID_W // 2
SLOT_H = GRID_H // 2
SLOT_SIZE = (SLOT_W, SLOT_H)
SURFACE_SIZE = SLOT_SIZE  # render at slot resolution
POLL_INTERVAL_MS = 1000 // 30
MAX_SLOTS = 4
COMPLETION_TIMEOUT_S = 2.0
BORDER_PX = 3

NETWORKS = [
    'yolov5m-v7-coco-tracker',
    'yolov8s-coco',
    'yolov8spose-coco',
    'yolov8sseg-coco',
]

IMAGE_PREPROCESSORS = [
    '',
    'rotate90',
    'rotate180',
    'rotate270',
    'horizontalflip',
    'verticalflip',
    'perspective[[1.019,-0.697,412.602,0.918,1.361,-610.083,0.0,0.0,1.0]]',
]

# ---------------------------------------------------------------------------
# Source / model discovery helpers
# ---------------------------------------------------------------------------


def _enum_usb_video_devices() -> list[str]:
    try:
        return sorted(
            f'usb:{m.group(1)}'
            for m in (re.match(r'video(\d+)', f) for f in os.listdir('/dev'))
            if m
        )
    except FileNotFoundError:
        return []


def _enum_media_sources() -> list[tuple[str, str]]:
    """Return (display_name, source_string) pairs for mp4 files in media/."""
    results = []
    try:
        for f in sorted(os.listdir('media')):
            if f.endswith('.mp4'):
                results.append((f, f'media/{f}@auto'))
    except FileNotFoundError:
        pass
    return results


def _check_model_deployed(name: str) -> bool:
    return os.path.isdir(f'build/{name}')


def _build_network_list() -> list[tuple[str, str, bool]]:
    """Return (display_name, network_name, is_deployed) for each known network."""
    results = []
    for name in NETWORKS:
        deployed = _check_model_deployed(name)
        label = name if deployed else f'{name} (requires download)'
        results.append((label, name, deployed))
    return results


# ---------------------------------------------------------------------------
# Slot state
# ---------------------------------------------------------------------------


class SlotState(enum.Enum):
    IDLE = 'idle'
    STARTING = 'starting'
    RUNNING = 'running'
    STOPPING = 'stopping'


@dataclasses.dataclass
class SlotInfo:
    state: SlotState = SlotState.IDLE
    pipeline: object = None  # PipeManager or None
    source_id: int = -1
    network: str = ''
    source: str = ''
    frames_done: int = 0
    frames_total: int = 0
    last_frame_time: float = 0.0


# ---------------------------------------------------------------------------
# Shared state (thread-safe)
# ---------------------------------------------------------------------------


class _ModifyCounter:
    """Reference-counted modification guard.

    Unlike threading.Event, each set() increments a counter and each clear()
    decrements it, so concurrent modifications (e.g. _do_add_pipeline on the
    inference thread and _bg_remove on a daemon thread) cannot accidentally
    clear each other's in-progress flag.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._count = 0

    def set(self):
        with self._lock:
            self._count += 1

    def clear(self):
        with self._lock:
            if self._count > 0:
                self._count -= 1

    def reset(self):
        """Hard-reset to zero; used during stream teardown."""
        with self._lock:
            self._count = 0

    def is_set(self) -> bool:
        with self._lock:
            return self._count > 0


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.slots: list[SlotInfo] = [SlotInfo() for _ in range(MAX_SLOTS)]
        self.pending_add: queue.Queue = queue.Queue()
        self.pending_remove: queue.Queue = queue.Queue()
        self.stop = threading.Event()
        self.teardown_stream = threading.Event()
        self.is_modifying = _ModifyCounter()
        # Render config (read by inference thread when adding pipelines)
        self.show_labels = True
        self.show_annotations = True

    def get_slot(self, idx: int) -> SlotInfo:
        with self.lock:
            return dataclasses.replace(self.slots[idx])

    def set_slot_state(self, idx: int, state: SlotState):
        with self.lock:
            self.slots[idx].state = state

    def set_slot_field(self, idx: int, **kwargs):
        with self.lock:
            for k, v in kwargs.items():
                setattr(self.slots[idx], k, v)

    def reset_slot(self, idx: int):
        with self.lock:
            self.slots[idx] = SlotInfo()

    def find_slot_by_pipeline(self, pipeline) -> int | None:
        with self.lock:
            for i, s in enumerate(self.slots):
                if s.pipeline is pipeline and s.state == SlotState.RUNNING:
                    return i
        return None

    def find_slot_by_source_id(self, source_id: int) -> int | None:
        with self.lock:
            for i, s in enumerate(self.slots):
                if s.source_id == source_id and s.state == SlotState.RUNNING:
                    return i
        return None

    def running_slots(self) -> list[int]:
        with self.lock:
            return [i for i, s in enumerate(self.slots) if s.state == SlotState.RUNNING]


# ---------------------------------------------------------------------------
# Inference thread
# ---------------------------------------------------------------------------


def inference_main(
    shared: SharedState,
    surfaces: list[display.Surface],
):
    """Long-lived inference thread.  Outer loop waits for add requests, creates
    a stream, iterates until all pipelines are gone, then loops back."""

    while not shared.stop.is_set():
        # Wait for the first add request
        try:
            first_add = shared.pending_add.get(timeout=0.1)
        except queue.Empty:
            continue

        # Drain any pending removes that arrived while we had no stream
        _drain_queue(shared.pending_remove)

        slot_id, network, source, preproc = first_add
        source_str = f'{preproc}:{source}' if preproc else source

        shared.set_slot_state(slot_id, SlotState.STARTING)
        shared.is_modifying.set()

        try:
            stream = create_inference_stream(
                network=network,
                sources=[source_str],
                aipu_cores=1,
                timeout=0,
                low_latency=True,
            )
        except Exception as e:
            shared.reset_slot(slot_id)
            shared.is_modifying.clear()
            continue

        # Record pipeline info
        pipeline = stream.pipelines[0]
        pipeline.set_render(shared.show_labels, shared.show_annotations)
        _source_ids = list(pipeline.sources.keys())
        sid = _source_ids[0] if _source_ids else 0
        total = pipeline.number_of_frames
        shared.set_slot_field(
            slot_id,
            state=SlotState.RUNNING,
            pipeline=pipeline,
            source_id=sid,
            network=network,
            source=source,
            frames_done=0,
            frames_total=total,
            last_frame_time=time.monotonic(),
        )
        shared.is_modifying.clear()

        # Build a lookup: source_id -> slot_id
        sid_to_slot: dict[int, int] = {sid: slot_id}

        # --- Inner iteration loop ---
        try:
            for event in stream.with_events():
                if shared.stop.is_set() or shared.teardown_stream.is_set():
                    break

                # Handle frame results
                if event.result:
                    fr = event.result
                    slot = sid_to_slot.get(fr.source_id)
                    if slot is not None:
                        surfaces[slot].push(fr.image, fr.meta)
                        with shared.lock:
                            shared.slots[slot].frames_done += 1
                            shared.slots[slot].last_frame_time = time.monotonic()
                elif event.type == FrameEventType.source_error:
                    slot = sid_to_slot.get(event.source_id)
                    print(f'[InferenceThread] Source error on slot {slot}: {event.message}')

                # Process pending add requests
                while not shared.pending_add.empty():
                    try:
                        req = shared.pending_add.get_nowait()
                    except queue.Empty:
                        break
                    _do_add_pipeline(stream, shared, surfaces, sid_to_slot, req)

                # Process pending remove requests
                while not shared.pending_remove.empty():
                    try:
                        rem_slot = shared.pending_remove.get_nowait()
                    except queue.Empty:
                        break
                    _do_remove_pipeline(stream, shared, sid_to_slot, rem_slot)

                if shared.teardown_stream.is_set():
                    break

                # Detect pipelines that disappeared (source finished)
                _detect_finished_pipelines(stream, shared, sid_to_slot)

                if shared.teardown_stream.is_set():
                    break

        except Exception as e:
            print(f'[InferenceThread] Stream error: {e}')
        finally:
            # Stream ended -- tear down and reset all slots back to idle
            try:
                stream.stop()
            except Exception as ex:
                print(f'[InferenceThread] stream.stop() raised: {ex}')
            for i in range(MAX_SLOTS):
                info = shared.get_slot(i)
                if info.state in (SlotState.RUNNING, SlotState.STARTING, SlotState.STOPPING):
                    shared.reset_slot(i)
            sid_to_slot.clear()
            shared.is_modifying.reset()
            shared.teardown_stream.clear()
            _drain_queue(shared.pending_remove)


def _drain_queue(q: queue.Queue):
    while not q.empty():
        try:
            q.get_nowait()
        except queue.Empty:
            break


def _do_add_pipeline(stream, shared: SharedState, surfaces, sid_to_slot, req):
    slot_id, network, source, preproc = req
    source_str = f'{preproc}:{source}' if preproc else source

    shared.set_slot_state(slot_id, SlotState.STARTING)
    shared.is_modifying.set()
    try:
        pipeline = stream.add_pipeline(
            network=network, sources=[source_str], aipu_cores=1, low_latency=True
        )
        pipeline.set_render(shared.show_labels, shared.show_annotations)
        _source_ids = list(pipeline.sources.keys())
        sid = _source_ids[0] if _source_ids else 0
        total = pipeline.number_of_frames
        now = time.monotonic()
        shared.set_slot_field(
            slot_id,
            state=SlotState.RUNNING,
            pipeline=pipeline,
            source_id=sid,
            network=network,
            source=source,
            frames_done=0,
            frames_total=total,
            last_frame_time=now,
        )
        sid_to_slot[sid] = slot_id
        # Refresh timestamps for ALL running slots so the completion timeout
        # does not fire for pipelines that were paused during the add.
        _refresh_running_timestamps(shared, now)
    except Exception as e:
        print(f'[InferenceThread] Failed to add pipeline: {e}')
        shared.reset_slot(slot_id)
    finally:
        shared.is_modifying.clear()


def _do_remove_pipeline(stream, shared: SharedState, sid_to_slot, slot_id):
    """Remove a pipeline.  If this is the last running pipeline, signal a full
    stream teardown instead of calling remove_pipeline (which would deadlock).
    Otherwise, runs remove_pipeline in a background thread to avoid deadlock."""
    info = shared.get_slot(slot_id)
    if info.state != SlotState.RUNNING or info.pipeline is None:
        return

    # Remove from mapping immediately so no more frames are routed to this slot
    sid_to_slot.pop(info.source_id, None)

    # Check if this is the last running pipeline
    running = shared.running_slots()
    is_last = running == [slot_id]

    if is_last:
        # Last pipeline -- tear down the entire stream; the outer loop will
        # reset slots and restart when a new source is added.
        # Keep the slot in STOPPING so the UI shows "Stopping pipeline..."
        # until stream.stop() completes in the inference thread's finally block.
        shared.set_slot_state(slot_id, SlotState.STOPPING)
        shared.teardown_stream.set()
        return

    # Not the last -- remove normally via background thread
    shared.set_slot_state(slot_id, SlotState.STOPPING)
    shared.is_modifying.set()

    pipeline_ref = info.pipeline

    def _bg_remove():
        try:
            stream.remove_pipeline(pipeline_ref)
        except Exception as e:
            print(f'[InferenceThread] Failed to remove pipeline: {e}')
        finally:
            shared.reset_slot(slot_id)
            # Refresh timestamps so remaining slots don't get timed out
            _refresh_running_timestamps(shared, time.monotonic())
            shared.is_modifying.clear()

    threading.Thread(target=_bg_remove, daemon=True, name=f'RemoveSlot{slot_id}').start()


def _refresh_running_timestamps(shared: SharedState, now: float):
    """Reset last_frame_time for all running slots to prevent false timeouts
    after pipeline modifications (which pause all pipelines)."""
    with shared.lock:
        for slot in shared.slots:
            if slot.state == SlotState.RUNNING:
                slot.last_frame_time = now


def _detect_finished_pipelines(stream, shared: SharedState, sid_to_slot):
    """Check if any slot's pipeline has been removed from the stream by the
    framework (source finished).  Update slot state accordingly.
    If no running slots remain after cleanup, signal a full stream teardown."""
    current_pipelines = set(stream.pipelines)
    for i in range(MAX_SLOTS):
        info = shared.get_slot(i)
        if info.state == SlotState.RUNNING and info.pipeline is not None:
            if info.pipeline not in current_pipelines:
                sid_to_slot.pop(info.source_id, None)
                shared.reset_slot(i)
    # If no running slots remain, tear down the stream so the outer loop
    # can restart cleanly when a new source is added.
    running = shared.running_slots()
    if not running:
        shared.teardown_stream.set()


# ---------------------------------------------------------------------------
# Download helper
# ---------------------------------------------------------------------------


def _download_model_thread(model_name: str, complete_callback):
    """Run axdownloadmodel in a subprocess, then invoke complete_callback on the wx thread."""
    try:
        result = subprocess.run(
            ['axdownloadmodel', model_name],
            capture_output=True,
            text=True,
            timeout=600,
        )
        success = result.returncode == 0
        msg = result.stdout if success else result.stderr
    except Exception as e:
        success = False
        msg = str(e)
    wx.CallAfter(complete_callback, model_name, success, msg)


# ---------------------------------------------------------------------------
# wxPython UI
# ---------------------------------------------------------------------------

IDLE_BG = wx.Colour(40, 40, 40)
SELECTED_BORDER = wx.Colour(0, 120, 215)
UNSELECTED_BORDER = wx.Colour(80, 80, 80)
OVERLAY_FG = wx.Colour(255, 255, 255)
STATUS_FG = wx.Colour(200, 200, 100)


class _ConfirmDlg(wx.Dialog):
    """Non-modal Yes/No dialog that lets the parent frame keep updating while open."""

    def __init__(self, parent, title, message, on_yes, on_no=None):
        super().__init__(parent, title=title, style=wx.DEFAULT_DIALOG_STYLE | wx.STAY_ON_TOP)
        sizer = wx.BoxSizer(wx.VERTICAL)
        text = wx.StaticText(self, -1, message)
        text.Wrap(360)
        sizer.Add(text, 0, wx.ALL, 12)
        btn_sizer = wx.StdDialogButtonSizer()
        btn_yes = wx.Button(self, wx.ID_YES)
        btn_no = wx.Button(self, wx.ID_NO)
        btn_yes.SetDefault()
        btn_sizer.AddButton(btn_yes)
        btn_sizer.AddButton(btn_no)
        btn_sizer.Realize()
        sizer.Add(btn_sizer, 0, wx.ALL | wx.EXPAND, 8)
        self.SetSizerAndFit(sizer)
        self.CentreOnParent()
        btn_yes.Bind(wx.EVT_BUTTON, lambda e: self._respond(on_yes))
        btn_no.Bind(wx.EVT_BUTTON, lambda e: self._respond(on_no))
        self.Bind(wx.EVT_CLOSE, lambda e: self._respond(on_no))
        self.Show()

    def _respond(self, callback):
        self.Destroy()
        if callback:
            callback()


class SlotPanel(wx.Panel):
    """A single video slot with bitmap display and info bar below."""

    def __init__(self, parent, slot_id: int, size):
        super().__init__(parent, style=wx.BORDER_NONE)
        self.slot_id = slot_id
        self._size = size
        self.SetBackgroundColour(UNSELECTED_BORDER)

        # Inner panel holds actual content (dark background)
        self._inner = wx.Panel(self, style=wx.BORDER_NONE)
        self._inner.SetBackgroundColour(IDLE_BG)

        inner_sizer = wx.BoxSizer(wx.VERTICAL)

        self._bmp = wx.StaticBitmap(self._inner, -1, size=size)
        w, h = size
        self._black_bmp = wx.Bitmap.FromBuffer(w, h, np.zeros((h, w, 3), dtype=np.uint8))
        self._bmp.SetBitmap(self._black_bmp)
        inner_sizer.Add(self._bmp, 0, wx.EXPAND)

        # Info bar below the bitmap — not overlapping
        info_panel = wx.Panel(self._inner)
        info_panel.SetBackgroundColour(IDLE_BG)
        info_sizer = wx.BoxSizer(wx.VERTICAL)

        self._overlay = wx.StaticText(
            info_panel,
            -1,
            f'Slot {slot_id}: Not Running',
            style=wx.ALIGN_CENTRE_HORIZONTAL | wx.ST_NO_AUTORESIZE,
        )
        self._overlay.SetForegroundColour(OVERLAY_FG)
        self._overlay.SetBackgroundColour(IDLE_BG)
        font = self._overlay.GetFont()
        font.SetPointSize(10)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        self._overlay.SetFont(font)

        self._status = wx.StaticText(
            info_panel,
            -1,
            '',
            style=wx.ALIGN_CENTRE_HORIZONTAL | wx.ST_NO_AUTORESIZE,
        )
        self._status.SetForegroundColour(STATUS_FG)
        self._status.SetBackgroundColour(IDLE_BG)
        sfont = self._status.GetFont()
        sfont.SetPointSize(8)
        self._status.SetFont(sfont)

        info_sizer.Add(self._overlay, 0, wx.EXPAND | wx.TOP, 2)
        info_sizer.Add(self._status, 0, wx.EXPAND, 0)
        info_sizer.AddSpacer(4)
        info_panel.SetSizer(info_sizer)
        info_panel.SetMinSize((-1, 48))

        inner_sizer.Add(info_panel, 0, wx.EXPAND | wx.LEFT | wx.RIGHT | wx.BOTTOM, 2)
        self._inner.SetSizer(inner_sizer)

        outer_sizer = wx.BoxSizer(wx.VERTICAL)
        outer_sizer.Add(self._inner, 1, wx.EXPAND | wx.ALL, BORDER_PX)
        self.SetSizer(outer_sizer)

        # Dirty-state tracking — avoid redundant GTK calls
        self._last_overlay_label = f'Slot {slot_id}: Not Running'
        self._last_status_label = ''
        self._bmp_is_black = True  # True while showing _black_bmp

        # Forward clicks on all children to the panel
        for w in (self, self._inner, self._bmp, self._overlay, self._status, info_panel):
            w.Bind(wx.EVT_LEFT_DOWN, self._on_click)

    def _on_click(self, evt):
        wx.PostEvent(self.GetParent().GetParent(), SlotSelectedEvent(slot_id=self.slot_id))

    def set_selected(self, selected: bool):
        colour = SELECTED_BORDER if selected else UNSELECTED_BORDER
        self.SetBackgroundColour(colour)
        self.Refresh()

    def _set_overlay(self, text: str):
        if text != self._last_overlay_label:
            self._overlay.SetLabel(text)
            self._last_overlay_label = text

    def _set_status(self, text: str):
        if text != self._last_status_label:
            self._status.SetLabel(text)
            self._last_status_label = text

    def update_display(self, info: SlotInfo, frame_image=None):
        if info.state == SlotState.IDLE:
            self._set_overlay(f'Slot {self.slot_id}: Not Running')
            self._set_status('')
            if frame_image is None and not self._bmp_is_black:
                self._bmp.SetBitmap(self._black_bmp)
                self._bmp_is_black = True
        elif info.state == SlotState.STARTING:
            self._set_overlay(f'Slot {self.slot_id}: Starting pipeline...')
            self._set_status(f'{info.network}')
        elif info.state == SlotState.STOPPING:
            self._set_overlay(f'Slot {self.slot_id}: Stopping pipeline...')
            self._set_status('')
        elif info.state == SlotState.RUNNING:
            if info.frames_total > 0:
                remaining = max(0, info.frames_total - info.frames_done)
                self._set_overlay(
                    f'Frame {info.frames_done} / {info.frames_total}  ({remaining} left)'
                )
            else:
                self._set_overlay(f'Frame {info.frames_done}')
            self._set_status(f'{info.network} | {_display_source(info.source)}')

        if frame_image is not None:
            np_img = frame_image.asarray(types.ColorFormat.RGB)
            h, w, _ = np_img.shape
            bmp = wx.Bitmap.FromBuffer(w, h, np_img)
            self._bmp.SetBitmap(bmp)
            self._bmp_is_black = False


def _display_source(source: str) -> str:
    """Pretty-print a source string for display."""
    s = source
    if s.startswith('media/'):
        s = s[len('media/') :]
    if s.endswith('@auto'):
        s = s[: -len('@auto')]
    return s


# Custom event for slot selection
EVT_SLOT_SELECTED_TYPE = wx.NewEventType()
EVT_SLOT_SELECTED = wx.PyEventBinder(EVT_SLOT_SELECTED_TYPE, 1)


class SlotSelectedEvent(wx.PyCommandEvent):
    def __init__(self, slot_id=0):
        super().__init__(EVT_SLOT_SELECTED_TYPE)
        self.slot_id = slot_id


class WxViewer(wx.Frame):
    def __init__(
        self,
        app: display.App,
        size,
        shared: SharedState,
        surfaces: list[display.Surface],
    ):
        super().__init__(parent=None, title='Axelera Inference Demo (wxPython)', size=size)
        self._app = app
        self._shared = shared
        self._surfaces = surfaces
        self._selected_slot = 0
        self._downloading: set[str] = set()
        self._download_pending: dict[str, tuple] = {}  # model -> (slot_idx, source, preproc)
        self._pending_remove_slots: set[int] = set()  # track slots with queued removes
        self._last_controls_state: tuple | None = None  # (is_idle, is_running) cache

        self._network_list = _build_network_list()
        self._source_list = _enum_usb_video_devices()
        self._media_sources = _enum_media_sources()

        self._build_ui()
        self._update_controls()
        self._select_slot(0)

        self._timer = wx.Timer(self)
        self.Bind(wx.EVT_TIMER, self._on_timer, self._timer)
        self.Bind(wx.EVT_CLOSE, self._on_close)
        self._timer.Start(POLL_INTERVAL_MS)
        self.Show()
        wx.CallAfter(self.Fit)

    # ---- UI construction ----

    def _build_ui(self):
        main_sizer = wx.BoxSizer(wx.HORIZONTAL)

        # Left: 2x2 grid of slot panels
        grid_panel = wx.Panel(self)
        grid = wx.GridSizer(2, 2, 2, 2)
        self._slot_panels: list[SlotPanel] = []
        for i in range(MAX_SLOTS):
            panel = SlotPanel(grid_panel, i, SLOT_SIZE)
            grid.Add(panel, 1, wx.EXPAND)
            self._slot_panels.append(panel)
        grid_panel.SetSizer(grid)
        self.Bind(EVT_SLOT_SELECTED, self._on_slot_selected)
        main_sizer.Add(grid_panel, 1, wx.EXPAND | wx.ALL, 4)

        # Right: controls
        ctrl_panel = wx.Panel(self)
        controls = wx.BoxSizer(wx.VERTICAL)

        # Selected slot indicator
        self._slot_label = wx.StaticText(ctrl_panel, -1, 'Selected: Slot 0')
        font = self._slot_label.GetFont()
        font.SetPointSize(12)
        font.SetWeight(wx.FONTWEIGHT_BOLD)
        self._slot_label.SetFont(font)
        controls.Add(self._slot_label, 0, wx.ALL, 8)
        controls.Add(wx.StaticLine(ctrl_panel, style=wx.LI_HORIZONTAL), 0, wx.EXPAND | wx.ALL, 4)

        # Network choice
        controls.Add(wx.StaticText(ctrl_panel, -1, 'Network:'), 0, wx.LEFT | wx.TOP, 8)
        self._network_choice = wx.Choice(
            ctrl_panel, -1, choices=[n[0] for n in self._network_list]
        )
        self._network_choice.SetSelection(0)
        self._network_choice.Bind(wx.EVT_CHOICE, self._on_network_choice)
        controls.Add(self._network_choice, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

        # Source choice
        controls.Add(wx.StaticText(ctrl_panel, -1, 'Source:'), 0, wx.LEFT | wx.TOP, 8)
        source_labels = [s for s in self._source_list] + [m[0] for m in self._media_sources]
        self._source_choice = wx.Choice(ctrl_panel, -1, choices=source_labels)
        if source_labels:
            self._source_choice.SetSelection(0)
        controls.Add(self._source_choice, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

        # Preproc choice
        controls.Add(wx.StaticText(ctrl_panel, -1, 'Preprocessor:'), 0, wx.LEFT | wx.TOP, 8)
        self._preproc_choice = wx.Choice(
            ctrl_panel, -1, choices=[p if p else '(none)' for p in IMAGE_PREPROCESSORS]
        )
        self._preproc_choice.SetSelection(0)
        controls.Add(self._preproc_choice, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 8)

        controls.AddSpacer(12)

        # Buttons
        btn_sizer = wx.BoxSizer(wx.HORIZONTAL)
        self._btn_start = wx.Button(ctrl_panel, -1, 'Start')
        self._btn_stop = wx.Button(ctrl_panel, -1, 'Stop')
        self._btn_start.Bind(wx.EVT_BUTTON, self._on_start)
        self._btn_stop.Bind(wx.EVT_BUTTON, self._on_stop)
        btn_sizer.Add(self._btn_start, 1, wx.ALL, 4)
        btn_sizer.Add(self._btn_stop, 1, wx.ALL, 4)
        controls.Add(btn_sizer, 0, wx.EXPAND | wx.LEFT | wx.RIGHT, 4)

        controls.AddSpacer(8)
        controls.Add(wx.StaticLine(ctrl_panel, style=wx.LI_HORIZONTAL), 0, wx.EXPAND | wx.ALL, 4)

        # Render config
        self._cb_annotations = wx.CheckBox(ctrl_panel, -1, 'Show annotations')
        self._cb_labels = wx.CheckBox(ctrl_panel, -1, 'Show labels')
        self._cb_annotations.SetValue(self._shared.show_annotations)
        self._cb_labels.SetValue(self._shared.show_labels)
        self._cb_annotations.Bind(wx.EVT_CHECKBOX, self._on_render_config_change)
        self._cb_labels.Bind(wx.EVT_CHECKBOX, self._on_render_config_change)
        controls.Add(self._cb_annotations, 0, wx.ALL, 6)
        controls.Add(self._cb_labels, 0, wx.ALL, 6)

        controls.AddStretchSpacer()

        # Download status
        self._download_status = wx.StaticText(ctrl_panel, -1, '')
        self._download_status.SetForegroundColour(wx.Colour(150, 150, 150))
        controls.Add(self._download_status, 0, wx.ALL, 8)

        ctrl_panel.SetSizer(controls)
        main_sizer.Add(ctrl_panel, 0, wx.EXPAND | wx.ALL, 4)

        self.SetSizer(main_sizer)

    # ---- Slot selection ----

    def _on_slot_selected(self, evt):
        self._select_slot(evt.slot_id)

    def _select_slot(self, idx):
        self._selected_slot = idx
        self._last_controls_state = None  # force re-enable when slot changes
        for i, panel in enumerate(self._slot_panels):
            panel.set_selected(i == idx)
        self._slot_label.SetLabel(f'Selected: Slot {idx}')
        self._update_controls()

    # ---- Controls ----

    def _get_selected_source_string(self) -> str:
        sel = self._source_choice.GetSelection()
        if sel == wx.NOT_FOUND:
            return ''
        n_usb = len(self._source_list)
        if sel < n_usb:
            return self._source_list[sel]
        media_idx = sel - n_usb
        return self._media_sources[media_idx][1]

    def _update_controls(self):
        info = self._shared.get_slot(self._selected_slot)
        is_idle = info.state == SlotState.IDLE
        is_running = info.state == SlotState.RUNNING
        net_sel = self._network_choice.GetSelection()
        needs_download = net_sel != wx.NOT_FOUND and not self._network_list[net_sel][2]
        state_key = (is_idle, is_running, needs_download)
        if state_key == self._last_controls_state:
            return
        self._last_controls_state = state_key
        self._btn_start.Enable(is_idle)
        self._btn_start.SetLabel('Download' if (is_idle and needs_download) else 'Start')
        self._btn_stop.Enable(is_running)
        self._network_choice.Enable(is_idle)
        self._source_choice.Enable(is_idle)
        self._preproc_choice.Enable(is_idle)

    def _on_start(self, _evt):
        idx = self._selected_slot
        info = self._shared.get_slot(idx)
        if info.state != SlotState.IDLE:
            return

        # Get selections
        net_sel = self._network_choice.GetSelection()
        if net_sel == wx.NOT_FOUND:
            return
        _label, network, deployed = self._network_list[net_sel]
        source = self._get_selected_source_string()
        if not source:
            wx.MessageBox('Please select a source.', 'No Source', wx.OK | wx.ICON_WARNING)
            return

        pre_sel = self._preproc_choice.GetSelection()
        preproc = IMAGE_PREPROCESSORS[pre_sel] if pre_sel != wx.NOT_FOUND else ''

        # Check if model is deployed
        if not deployed:
            if network in self._downloading:
                self._download_status.SetLabel(f'{network} is already downloading...')
                return

            def _do_download():
                self._start_download(network, idx, source, preproc)

            _ConfirmDlg(
                self,
                'Model Not Deployed',
                f'Model "{network}" is not deployed.\n\n' f'Would you like to download it?\n',
                on_yes=_do_download,
            )
            return

        self._shared.pending_add.put((idx, network, source, preproc))
        self._update_controls()

    def _on_stop(self, _evt):
        idx = self._selected_slot
        info = self._shared.get_slot(idx)
        if info.state != SlotState.RUNNING:
            return
        self._shared.pending_remove.put(idx)
        self._update_controls()

    def _on_render_config_change(self, _evt):
        show_annotations = self._cb_annotations.GetValue()
        show_labels = self._cb_labels.GetValue()
        self._shared.show_annotations = show_annotations
        self._shared.show_labels = show_labels
        # Collect running pipelines under the lock, then apply render config outside it
        with self._shared.lock:
            running_pipelines = [
                slot.pipeline
                for slot in self._shared.slots
                if slot.state == SlotState.RUNNING and slot.pipeline is not None
            ]
        for pipeline in running_pipelines:
            pipeline.set_render(show_labels, show_annotations)

    def _on_network_choice(self, _evt):
        self._last_controls_state = None  # force button label refresh
        self._update_controls()

    # ---- Model download ----

    def _start_download(self, model_name: str, slot_idx: int, source: str, preproc: str):
        self._downloading.add(model_name)
        self._download_pending[model_name] = (slot_idx, source, preproc)
        self._download_status.SetLabel(f'Downloading {model_name}...')
        threading.Thread(
            target=_download_model_thread,
            args=(model_name, self._on_download_complete),
            daemon=True,
        ).start()

    def _on_download_complete(self, model_name: str, success: bool, msg: str):
        pending = self._download_pending.pop(model_name, None)
        self._downloading.discard(model_name)
        if success:
            # Refresh network list, keeping the just-downloaded model selected
            self._network_list = _build_network_list()
            new_sel = next(
                (i for i, (_, n, _) in enumerate(self._network_list) if n == model_name),
                0,
            )
            self._network_choice.Set([n[0] for n in self._network_list])
            self._network_choice.SetSelection(new_sel)
            self._last_controls_state = None

            if pending:
                slot_idx, source, preproc = pending

                def _start_now():
                    self._shared.pending_add.put((slot_idx, model_name, source, preproc))
                    self._update_controls()

                self._download_status.SetLabel(f'{model_name} ready.')
                _ConfirmDlg(
                    self,
                    'Download Complete',
                    f'"{model_name}" has been downloaded.\n\nStart the stream now?',
                    on_yes=_start_now,
                )
            else:
                self._download_status.SetLabel(f'{model_name} downloaded successfully.')
        else:
            last_line = next((l for l in reversed(msg.splitlines()) if l.strip()), 'unknown error')
            self._download_status.SetForegroundColour(wx.Colour(255, 80, 80))
            self._download_status.SetLabel(f'Failed: {last_line[:70]}')

    # ---- Timer / polling ----

    def _on_timer(self, _evt):
        if self._shared.stop.is_set():
            self._on_close(None)
            return

        now = time.monotonic()

        for i in range(MAX_SLOTS):
            info = self._shared.get_slot(i)
            frame = None

            # Clear pending-remove tracking when slot goes idle
            if info.state == SlotState.IDLE:
                self._pending_remove_slots.discard(i)

            if info.state == SlotState.RUNNING:
                new = self._surfaces[i].pop_latest()
                if new is not None:
                    frame = new

                # Completion timeout: no frame for COMPLETION_TIMEOUT_S while not modifying
                # Guard: only queue one remove per slot
                if (
                    i not in self._pending_remove_slots
                    and info.last_frame_time > 0
                    and info.frames_done > 0
                    and (now - info.last_frame_time) > COMPLETION_TIMEOUT_S
                    and not self._shared.is_modifying.is_set()
                ):
                    self._shared.pending_remove.put(i)
                    self._pending_remove_slots.add(i)

            self._slot_panels[i].update_display(info, frame)

        self._update_controls()
        wx.YieldIfNeeded()

    # ---- Shutdown ----

    def _on_close(self, evt):
        try:
            if self._timer.IsRunning():
                self._timer.Stop()
        finally:
            self._shared.stop.set()
            self.Destroy()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    shared = SharedState()

    with display.App(renderer='opencv') as app:
        surfaces = [app.create_surface(SURFACE_SIZE) for _ in range(MAX_SLOTS)]
        wx_app = wx.App(False)
        _viewer = WxViewer(app, (1200, 700), shared, surfaces)

        app.start_thread(
            inference_main,
            (shared, surfaces),
            name='InferenceThread',
        )

        wx_app.MainLoop()
        shared.stop.set()  # signal before display.App tears down


if __name__ == '__main__':
    main()

#!/usr/bin/env python
# Copyright Axelera AI, 2026

try:
    import wx
except ImportError:
    raise ImportError(
        "wxPython is required to run this application. To install it, call..."
        "\npip install wxpython"
    )

try:
    import cv2
except ImportError:
    raise ImportError(
        "OpenCV is required to run this application. To install it, call..."
        "\npip install opencv-python"
    )

import json
import copy
import re
import os
import threading
import queue
from typing import Any
import time
import camera_scan

version = "1.0.4"
video_panel_width = 890

# This will set the initial size of the video panel to be 1.77 ratio
initial_frame_size = wx.Size(1350, 851)
min_video_size = wx.Size(video_panel_width, int(video_panel_width / 1.777))
initial_model_size = wx.Size(640, 640)

TILE_RESIZE_CORNER_SIZE = 5
MIN_TILE_SIZE = 100

EVT_CAMERAS_READ_ID = wx.NewEventType()
EVT_CAMERAS_READ = wx.PyEventBinder(EVT_CAMERAS_READ_ID, 1)


class AXTileCreatorException(Exception):
    pass


def _get_available_cameras() -> list[dict[str, Any]]:
    """Queries the system for available cameras and their supported configurations.
    Returns a list of dictionaries, each containing information about a camera and its supported
    configurations.

    Returns:
        list[dict[str, Any]]: List of camera info dictionaries
    """
    found_cameras: list[dict[str, Any]] = camera_scan.scan_all_cameras(100)
    return [i for i in found_cameras if i['supported_configs']]


def _extract_resolution_tuple(res_str: str) -> tuple[int, int] | None:
    """Extracts a resolution tuple (width, height) from a string in the format "WIDTHxHEIGHT".

    Args:
        res_str (str): Resolution string in the format "WIDTHxHEIGHT"

    Returns:
        tuple[int, int] | None: Tuple of (width, height) if the string is valid, otherwise None
    """
    match = re.match(r"(\d{1,5})x(\d{1,5})", res_str)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def _resolution_tuple_to_str(res: tuple[int, int]) -> str:
    """Convert a resolution tuple (width, height) to a string in the format "WIDTHxHEIGHT".

    Args:
        res (tuple[int, int]): Tuple of (width, height)

    Returns:
        str: Resolution string in the format "WIDTHxHEIGHT"
    """
    return f"{res[0]}x{res[1]}"


def _get_string_from_tile(tile: list[int]) -> str:
    """Convert a list of four integers representing a tile to a string.

    Args:
        tile (list[int]): List of four integers representing a tile

    Returns:
        str: String representation of the tile in the format "x1, y1, x2, y2"
    """
    return f"{tile[0]}, {tile[1]}, {tile[2]}, {tile[3]}"


class VideoCaptureThread(threading.Thread):
    """Thread that captures video frames from a video file via an opencv VideoCaptureThread
    and pushes them to a queue.
    """

    def __init__(self, video_path: Any, frame_queue: queue.Queue, max_queue_size: int = 30):
        super().__init__(daemon=True)
        self.video_path = video_path
        self.frame_queue = frame_queue
        self.max_queue_size = max_queue_size
        self.running = False
        self.pause = False
        # Suppress FFmpeg/OpenCV stderr noise (e.g. RTSP 404) during VideoCapture open
        saved_stderr_fd = os.dup(2)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, 2)
        os.close(devnull_fd)
        try:
            self.cap = cv2.VideoCapture(self.video_path)
        finally:
            os.dup2(saved_stderr_fd, 2)
            os.close(saved_stderr_fd)
        if not self.cap.isOpened():
            raise RuntimeError(f"Error: Could not open stream or file: {self.video_path}")

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = round(self.cap.get(cv2.CAP_PROP_FPS))
        self.condition = threading.Condition()

    def __del__(self):
        self.cap.release()

    def get_resolution(self) -> tuple[int, int]:
        return self.width, self.height

    def get_fps(self) -> float:
        return self.fps

    def is_playing(self) -> bool:
        return self.pause is False

    def run(self) -> None:
        """Read frames from video and push to queue."""
        self.running = True
        while self.running:
            if not self.pause:
                ret, frame = self.cap.read()
                if not ret:
                    # Video ended, loop back to beginning
                    self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Queue is full, delete the frame to free memory immediately
                    del frame
                    pass
            else:
                with self.condition:
                    self.condition.wait_for(self.is_playing, timeout=0.5)
        self.cap.release()

    def stop(self) -> None:
        self.pause = False
        self.running = False

    def _pause(self) -> None:
        self.pause = True

    def _restart(self) -> None:
        self.pause = False


class CameraCaptureThread(threading.Thread):
    """Thread that captures video frames from a USB camera via an opencv VideoCaptureThread and
    pushes them to a queue.
    """

    def __init__(
        self,
        camera_id: int,
        camera_res: tuple[int, int],
        camera_format: str,
        camera_fps: int,
        frame_queue: queue.Queue,
        max_queue_size: int = 30,
    ):
        super().__init__(daemon=True)
        self.camera_id = camera_id
        self.frame_queue = frame_queue
        self.max_queue_size = max_queue_size
        self.running = False
        self.pause = False
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise RuntimeError(f"Error: Could not open camera {self.camera_id}")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*camera_format))
        self.cap.set(cv2.CAP_PROP_FPS, camera_fps)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, camera_res[0])
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, camera_res[1])

        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = round(self.cap.get(cv2.CAP_PROP_FPS))
        self.condition = threading.Condition()

    def __del__(self):
        self.cap.release()

    def get_resolution(self) -> tuple[int, int]:
        return self.width, self.height

    def get_fps(self) -> float:
        return self.fps

    def is_playing(self) -> bool:
        return self.pause is False

    def run(self) -> None:
        """Read frames from camera and push to queue."""
        self.running = True
        while self.running:
            if not self.pause:
                ret, frame = self.cap.read()
                if not ret:
                    # Failed to read frame, wait a bit and retry
                    time.sleep(0.1)
                    continue
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                try:
                    self.frame_queue.put_nowait(frame)
                except queue.Full:
                    # Queue is full, delete the frame to free memory immediately
                    del frame
                    pass
            else:
                with self.condition:
                    self.condition.wait_for(self.is_playing, timeout=0.5)
        self.cap.release()

    def stop(self) -> None:
        self.pause = False
        self.running = False

    def _pause(self) -> None:
        self.pause = True

    def _restart(self) -> None:
        self.pause = False


class CameraDiscoveryUpdateEvent(wx.PyEvent):
    def __init__(self, camera_info: list[dict[str, Any]]):
        wx.PyEvent.__init__(self)
        self.SetEventType(EVT_CAMERAS_READ_ID)
        self.camera_info = copy.deepcopy(camera_info)


class CameraDiscoveryThread(threading.Thread):
    """Discover all cameras attached to the system and rretrieve their supported configurations.
    This is done in a separate thread because it can take a long time to query all cameras,
    especially if there are mAny or if some are slow to respond.
    """

    def __init__(self, notify_window: wx.Window):
        super().__init__(daemon=True)
        self.notify_window = notify_window

    def run(self) -> None:
        self.camera_info: list[dict[str, Any]] = _get_available_cameras()
        if self.notify_window:
            wx.PostEvent(self.notify_window, CameraDiscoveryUpdateEvent(self.camera_info))


def _convert_string_to_tile(line: str) -> tuple[list[int], str]:
    parts = line.split(',')
    if len(parts) == 4:
        try:
            return [int(part.strip()) for part in parts], ''
        except ValueError:
            return [], f"Invalid line: {line}"
    else:
        return [], f"Invalid format: {line}"


def _convert_string_to_tiles(tiles_list: list[str]) -> tuple[list[list[int]], str]:
    """Takes the list of string lines and converts them to a list of tiles.
    Each tile is represented as a list of four integers: [x, y, w, h].

    Args:
        tiles_list (list[str]): List of strings representing tiles in "x,y,w,h" format.

    Returns:
        tuple: (list of tiles as list of int, error message)
    """
    tiles = []
    for line in tiles_list:
        tile, error = _convert_string_to_tile(line)
        if error:
            return [], error
        tiles.append(tile)

    return tiles, ''


def _get_rect_from_x1y1(xy: wx.Point, x1y1: wx.Point) -> wx.Rect:
    """Given two points, return a wx.Rect that represents the rectangle defined by those points.

    Args:
        xy (wx.Point): First point.
        x1y1 (wx.Point): Second point.

    Returns:
        wx.Rect: Rectangle defined by the two points.
    """
    x = min(xy.x, x1y1.x)
    y = min(xy.y, x1y1.y)
    w = abs(x1y1.x - xy.x)
    h = abs(x1y1.y - xy.y)
    return wx.Rect(x, y, w, h)


class TileHistory:
    """Class to manage undo/redo history of tile edits."""

    def __init__(self):
        self.history: list[list[list[int]]] = []
        self.future: list[list[list[int]]] = []

    def _tiles_cmp(self, lhs: list[list[int]], rhs: list[list[int]]) -> bool:
        if len(lhs) != len(rhs):
            return False
        for tile_lhs, tile_rhs in zip(lhs, rhs):
            if tile_lhs != tile_rhs:
                return False
        return True

    def add_state(self, tiles: list[list[int]]) -> None:
        if self.history and self._tiles_cmp(self.history[-1], tiles):
            return
        self.history.append(copy.deepcopy(tiles))
        self.future.clear()

    def undo(self, current_tiles: list[list[int]]) -> list[list[int]] | None:
        if not self.history:
            return None
        last_state = self.history.pop()
        self.future.append(copy.deepcopy(current_tiles))
        return last_state

    def redo(self, current_tiles: list[list[int]]) -> list[list[int]] | None:
        if not self.future:
            return None
        next_state = self.future.pop()
        self.history.append(copy.deepcopy(current_tiles))
        return next_state

    def clear(self) -> None:
        self.history.clear()
        self.future.clear()

    def canundo(self) -> bool:
        return len(self.history) > 0

    def canredo(self) -> bool:
        return len(self.future) > 0


class TileListBox(wx.ListBox):
    """Specialised listbox for displaying tiles."""

    def __init__(
        self,
        parent: wx.Window,
        id: int = wx.ID_ANY,
        size: wx.Size = wx.DefaultSize,
        style: int = 0,
    ):
        super().__init__(parent, id, size=size, style=style)
        self.parent = parent
        self._bind()

    def _bind(self) -> None:
        self.Bind(wx.EVT_LISTBOX, self._on_selection_changed)
        self.Bind(wx.EVT_KEY_DOWN, self._on_key_down)

    def _on_selection_changed(self, event: wx.Event) -> None:
        self.parent.create_move_tiles_list()
        self.parent.update_video_panel()
        event.Skip()

    def _on_key_down(self, event: wx.Event) -> None:
        key_code = event.GetKeyCode()
        if key_code == wx.WXK_DELETE:
            selections = self.GetSelections()
            self.parent.tile_history.add_state(self.parent.tiles)
            if selections:
                for index in reversed(selections):
                    del self.parent.tiles[index]
                self.parent.update_tiles_listbox(persist_selections=False)


class TileTextCtrl(wx.TextCtrl):
    """Specialised text control for editing tiles."""

    def __init__(
        self,
        parent: wx.Window,
        id: int = wx.ID_ANY,
        size: wx.Size = wx.DefaultSize,
        style: int = 0,
    ):
        super().__init__(parent, id, size=size, style=style)
        self.parent = parent
        self._bind()

    def _bind(self) -> None:
        self.Bind(wx.EVT_TEXT, self._on_text_changed)
        self.Bind(wx.EVT_KEY_DOWN, self._on_key_down)

    def _on_text_changed(self, event: wx.Event) -> None:
        tiles_str = self.GetValue().splitlines()
        _tiles, error = _convert_string_to_tiles(tiles_str)
        self.parent.tile_edit_error.SetLabel(error)
        self.parent.tile_apply_button.Enable(error == '')
        event.Skip()

    def _on_key_down(self, event: wx.Event) -> None:
        key_code = event.GetKeyCode()
        if key_code == wx.WXK_ESCAPE:
            self.parent._quit_edit_tiles()
        event.Skip()


class VideoPanel(wx.Panel):
    """Panel that displays the video stream."""

    def __init__(self, parent: wx.Window, ctrl_dim: wx.Size):
        super().__init__(parent, size=ctrl_dim, style=wx.BORDER_SIMPLE)
        self.parent = parent
        self.SetBackgroundColour(wx.BLACK)
        self.bitmap: wx.Bitmap | None = None
        self.black_pen: wx.Pen = wx.Pen(wx.Colour(0, 0, 0), 2, wx.PENSTYLE_SOLID)
        self.yellow_pen: wx.Pen = wx.Pen(wx.Colour(255, 255, 0), 2, wx.PENSTYLE_SOLID)
        self.green_pen: wx.Pen = wx.Pen(wx.Colour(0, 255, 0), 2, wx.PENSTYLE_SOLID)
        self.red_pen: wx.Pen = wx.Pen(wx.Colour(255, 0, 0), 2, wx.PENSTYLE_SOLID)
        self.select_pen: wx.Pen = wx.Pen(wx.Colour(255, 255, 255), 4, wx.PENSTYLE_DOT)
        self.transparent_brush: wx.Brush = wx.Brush("black", wx.BRUSHSTYLE_TRANSPARENT)

        self.corner_status: str = ''
        self.current_tile: list[int] | None = None
        self.resizing_tile: list[int] | None = None
        self.resizing_corner: str = ''
        self.frame: Any = None
        self._clear_drag()
        self.left_mouse_down: bool = False
        self.drawing_tile: bool = False
        self.dragging_tile: bool = False
        self._bind()

    def cleanup(self) -> None:
        """Explicitly cleanup resources to prevent memory leaks."""
        if self.frame is not None:
            del self.frame
            self.frame = None
        if self.bitmap:
            del self.bitmap
            self.bitmap = None

    def __del__(self):
        # Clean up frame and bitmap first
        if self.frame is not None:
            del self.frame
        if self.bitmap:
            self.bitmap.Destroy()
        # Clean up GDI objects
        self.black_pen.Destroy()
        self.yellow_pen.Destroy()
        self.green_pen.Destroy()
        self.red_pen.Destroy()
        self.select_pen.Destroy()
        self.transparent_brush.Destroy()

    def _clear_drag(self) -> None:
        self.xy = wx.Point(0, 0)
        self.x1y1 = wx.Point(-1, -1)

    def _bind(self) -> None:
        self.Bind(wx.EVT_MOTION, self._on_mouse_move)
        self.Bind(wx.EVT_LEAVE_WINDOW, self._on_mouse_leave)
        self.Bind(wx.EVT_LEFT_DOWN, self._on_left_mouse_down)
        self.Bind(wx.EVT_RIGHT_DOWN, self._on_right_mouse_down)
        self.Bind(wx.EVT_LEFT_UP, self._on_mouse_up)
        self.Bind(wx.EVT_MOUSEWHEEL, self._on_mouse_wheel)
        self.Bind(wx.EVT_KEY_DOWN, self._on_key_down)
        self.Bind(wx.EVT_PAINT, self.on_paint)
        self.Bind(wx.EVT_SIZE, self.on_size)

    def _on_key_down(self, event: wx.Event) -> None:
        key = event.GetKeyCode()
        if key == wx.WXK_SHIFT:
            self.parent.create_move_tiles_list()
        if key == wx.WXK_DELETE:
            selections = self.parent.get_selections()
            self.parent.tile_history.add_state(self.parent.tiles)
            if selections:
                for index in reversed(selections):
                    del self.parent.tiles[index]
                self.parent.update_tiles_listbox(persist_selections=False)
        elif key == ord('Z') and event.ControlDown():
            self.parent.undo()
        elif key == ord('Y') and event.ControlDown():
            self.parent.redo()
        event.Skip()

    def _on_right_mouse_down(self, event: wx.Event) -> None:
        if self.HasFocus() is False:
            self.SetFocus()
        if not self.left_mouse_down and not self.drawing_tile:
            size = (-1, -1)
            if event.ShiftDown():
                size = self.parent.get_custom_tile_size()
            elif event.CmdDown():
                size = self.parent.get_selected_tile_size()
            else:
                size = self.parent.get_model_size()
            if size != (-1, -1):
                self.parent.create_tile(wx.Point(event.GetPosition()), size)
        event.Skip()

    def _on_left_mouse_down(self, event: wx.Event) -> None:
        if self.HasFocus() is False:
            self.SetFocus()
        self.dragging_tile = False
        self.drawing_tile = False
        self.left_mouse_down = True
        self.xy = wx.Point(event.GetPosition())
        if self.corner_status:
            self.resizing_tile = self.current_tile
            self.resizing_corner = self.corner_status
            self.parent.add_current_tiles_to_history()
        elif not event.ShiftDown() and (
            bool(event.CmdDown()) or not self.parent.are_any_tiles_at_point_selected(self.xy)
        ):
            self.parent.select_tiles_at_point(self.xy, bool(event.CmdDown()))

        if self.parent.are_selections():
            self.parent.create_move_tiles_list()
        self.parent.update_video_panel()
        event.Skip()

    def _on_mouse_move(self, event: wx.Event) -> None:
        if self.HasFocus() is False:
            self.SetFocus()
        current_point = wx.Point(event.GetPosition())
        if self.left_mouse_down:
            if not self.drawing_tile:
                if self.corner_status:
                    if not self.dragging_tile:
                        # Drag the corner and resize the tile
                        hit_min_limit = self.parent.size_corner(
                            self.resizing_tile, current_point, self.corner_status
                        )
                        if hit_min_limit:
                            self.left_mouse_down = False
                            self.parent.update_video_panel()
                elif self.parent.are_selections() and self.x1y1 != wx.Point(-1, -1):
                    # drag (move) the selected tiles
                    self.parent.move_selected_titles(
                        current_point.x - self.xy.x, current_point.y - self.xy.y
                    )
                    self.dragging_tile = True
                elif event.ShiftDown():
                    # Manually drawing a new tile with the mouse
                    self.drawing_tile = True
            self.x1y1 = current_point
        if not self.drawing_tile and not self.dragging_tile:
            self.current_tile, self.corner_status = self.parent.get_corner_on_mouse(current_point)
        self.parent.set_current_mouse_pos(current_point)
        self.parent.update_video_panel()
        event.Skip()

    def _on_mouse_up(self, event: wx.Event) -> None:
        if self.HasFocus() is False:
            self.SetFocus()
        if self.drawing_tile:
            self.x1y1 = wx.Point(event.GetPosition())
            new_rect = self.get_drag_rectangle()
            if new_rect.width > 0 and new_rect.height > 0:
                self.parent.add_tile_from_screen_coords(new_rect)
        self.left_mouse_down = False
        self.drawing_tile = False
        self.dragging_tile = False
        self._clear_drag()
        self.parent.update_video_panel()

    def _on_mouse_leave(self, event: wx.Event) -> None:
        if self.HasFocus() is False:
            self.SetFocus()
        if self.corner_status:
            self.left_mouse_down = False
            self.corner_status = ''
            self.parent.update_video_panel()
        event.Skip()

    def _on_mouse_wheel(self, event: wx.Event) -> None:
        if self.HasFocus() is False:
            self.SetFocus()
        delta = 1 if event.GetWheelRotation() > 0 else -1
        self.parent.size_selected_tiles(delta, delta)
        self.parent.update_video_panel()
        event.Skip()

    def get_drag_rectangle(self) -> wx.Rect:
        return _get_rect_from_x1y1(self.x1y1, self.xy)

    def on_paint(self, _event: wx.Event) -> None:
        """Paint the current bitmap."""
        dc = wx.PaintDC(self)
        if self.bitmap:
            dc.DrawBitmap(self.bitmap, 0, 0, True)
        # Cleanup device context resources
        dc.SetPen(wx.NullPen)
        dc.SetBrush(wx.NullBrush)
        del dc

    def on_size(self, event: wx.Event) -> None:
        """Handle resize events."""
        self.parent.set_screen_to_video_ratios()
        self.parent.update_video_panel()
        event.Skip()

    def set_frame(self, frame: Any) -> None:
        # Delete old frame to prevent memory leak
        if self.frame is not None:
            del self.frame
        self.frame = frame
        self.draw_frame()

    def draw_frame(self) -> None:
        """Convert OpenCV frame to bitmap and overlay the tiles."""

        if self.frame is not None:
            panel_width, panel_height = self.GetSize()
            frame_height, frame_width = self.frame.shape[:2]

            # Calculate scaling to fit panel while maintaining aspect ratio
            scale_w = panel_width / frame_width
            scale_h = panel_height / frame_height
            new_width = int(frame_width * scale_w)
            new_height = int(frame_height * scale_h)

            # Track if we created a resized frame that needs cleanup
            resized_frame = None
            if scale_w != 1.0 or scale_h != 1.0:
                resized_frame = cv2.resize(self.frame, (new_width, new_height))
                frame = resized_frame
            else:
                frame = self.frame

            height, width = frame.shape[:2]
            # Delete old bitmap to prevent memory leak
            if self.bitmap:
                del self.bitmap
            self.bitmap = wx.Bitmap.FromBuffer(width, height, frame)

            # Clean up resized frame after bitmap creation
            if resized_frame is not None:
                del resized_frame
                del frame

            dc = wx.MemoryDC()
            dc.SelectObject(self.bitmap)
            dc.SetBrush(self.transparent_brush)

            for tile in self.parent.tiles:
                x = int(tile[0] * scale_w)
                y = int(tile[1] * scale_h)
                w = int(tile[2] * scale_w)
                h = int(tile[3] * scale_h)

                if self.parent.is_tile_selected(tile):
                    dc.SetPen(self.select_pen)
                elif tile[2] < self.parent.model_width or tile[3] < self.parent.model_height:
                    dc.SetPen(self.red_pen)
                elif self.parent.model_width == tile[2] and self.parent.model_height == tile[3]:
                    dc.SetPen(self.green_pen)
                else:
                    dc.SetPen(self.black_pen)
                dc.DrawRectangle(x, y, w, h)
                if tile == self.current_tile and not self.drawing_tile and not self.dragging_tile:
                    corner_size = TILE_RESIZE_CORNER_SIZE
                    dc.SetPen(self.yellow_pen)
                    if self.corner_status == 'TL':
                        dc.DrawRectangle(
                            x - corner_size, y - corner_size, corner_size * 2, corner_size * 2
                        )
                    elif self.corner_status == 'TR':
                        dc.DrawRectangle(
                            x + w - corner_size, y - corner_size, corner_size * 2, corner_size * 2
                        )
                    elif self.corner_status == 'BL':
                        dc.DrawRectangle(
                            x - corner_size, y + h - corner_size, corner_size * 2, corner_size * 2
                        )
                    elif self.corner_status == 'BR':
                        dc.DrawRectangle(
                            x + w - corner_size,
                            y + h - corner_size,
                            corner_size * 2,
                            corner_size * 2,
                        )

            if self.drawing_tile:
                drag_rect = self.get_drag_rectangle()
                if drag_rect.width > 0 and drag_rect.height > 0:
                    dc.SetPen(self.yellow_pen)
                    dc.DrawRectangle(drag_rect.x, drag_rect.y, drag_rect.width, drag_rect.height)

            dc.SelectObject(wx.NullBitmap)
            dc.SetBrush(wx.NullBrush)
            dc.SetPen(wx.NullPen)
            del dc
        else:
            # Draw a black rectangle when no frame is available
            panel_width, panel_height = self.GetSize()
            # Delete old bitmap to prevent memory leak
            if self.bitmap:
                del self.bitmap
            self.bitmap = wx.Bitmap(panel_width, panel_height)
            dc = wx.MemoryDC()
            dc.SelectObject(self.bitmap)
            black_brush = wx.Brush(wx.BLACK)
            dc.SetBackground(black_brush)
            dc.Clear()
            dc.SelectObject(wx.NullBitmap)
            dc.SetBackground(wx.NullBrush)
            del black_brush
            del dc

        self.Refresh(False)


class AxTileGenPanel(wx.Panel):
    """Main panel for AX Tile Creator application."""

    def __init__(
        self,
        parent: wx.Window,
        size: wx.Size,
        version_str: str,
    ) -> None:
        super().__init__(parent, size=size)
        self.parent: wx.Window = parent
        self.version_str: str = version_str
        self.tiles: list[list[int]] = []
        self.move_tiles: list[list[int]] = []
        self.current_x: int = 0
        self.current_y: int = 0
        self.x_ratio: float = 1.0
        self.y_ratio: float = 1.0
        self.model_width: int = initial_model_size.width
        self.model_height: int = initial_model_size.height
        self.min_video_size: wx.Size = min_video_size
        self.framerate: float = 0.0
        self.json_filename: str = ''
        self.editing_tiles: bool = False
        self.resolution: wx.Size = self.min_video_size
        self.ratio: float = 1.0
        self.tile_history: TileHistory = TileHistory()

        # Create frame queue with limited size to prevent memory buildup
        # Keep only 2-3 frames buffered to minimize memory usage
        self.frame_queue: queue.Queue = queue.Queue(maxsize=3)
        self.timer: wx.Timer = wx.Timer(self)

        # Create and start video capture thread
        self.capture_thread: VideoCaptureThread | CameraCaptureThread | None = None

        self.camera_discovery_thread: CameraDiscoveryThread | None = None
        self.camera_info: list[dict[str, Any]] = []

        self._create_widgets()
        self._layout()
        self._bind()
        self._set_video_resolutiuon(wx.Size(0, 0))
        self.Show()
        self._updateundoredo_buttons()

    def _create_widgets(self) -> None:
        right_hand_side_width = 280

        self.video_filename_picker = wx.FilePickerCtrl(
            self,
            path="",
            style=wx.FLP_DEFAULT_STYLE | wx.FLP_SMALL | wx.FLP_FILE_MUST_EXIST,
            size=(650, -1),
        )
        # self.video_filename_picker.SetPath(
        #     "\\\\wsl.localhost\\Ubuntu\\home\\tony\\branches"
        #     "\\app\\tile_creator\\application.framework\\tools"
        #     "\\8K_show_full_iscw - EDITED V5.mp4"
        # )
        pickerfont = self.video_filename_picker.GetFont()
        pickerfont.SetPointSize(pickerfont.GetPointSize() + 2)
        pickerfont.SetWeight(wx.FONTWEIGHT_BOLD)

        edit_size = self.video_filename_picker.GetSize()
        self.ctrl_button_size = wx.Size(140, -1)

        self.start_video_button = wx.Button(self, label="Start Video", size=self.ctrl_button_size)
        self.start_video_button.Enable(False)

        self.rtsp_url_static = wx.StaticText(
            self,
            label="RTSP URL",
            # size=(-1, edit_size.height),
            style=wx.ALIGN_CENTRE_VERTICAL,
        )
        self.rtsp_url_edit = wx.TextCtrl(self, size=(50, edit_size.height))
        self.start_rtsp_button = wx.Button(self, label="Start RTSP", size=self.ctrl_button_size)
        self.start_rtsp_button.Enable(False)

        self.camera_num_static = wx.StaticText(self, label="Cam#", style=wx.ALIGN_CENTRE_VERTICAL)
        self.camera_num_combo = wx.ComboBox(self, style=wx.CB_READONLY, size=(70, -1))
        self.camera_num_combo.Enable(False)

        self.camera_res_static = wx.StaticText(self, label="Res", style=wx.ALIGN_CENTRE_VERTICAL)
        self.camera_res_combo = wx.ComboBox(self, style=wx.CB_READONLY, size=(128, -1))
        self.camera_res_combo.Enable(False)
        self.camera_fmt_static = wx.StaticText(self, label="Fmt", style=wx.ALIGN_CENTRE_VERTICAL)
        self.camera_fmt_combo = wx.ComboBox(self, style=wx.CB_READONLY, size=(94, -1))
        self.camera_fmt_combo.Enable(False)
        self.camera_fps_static = wx.StaticText(self, label="FPS", style=wx.ALIGN_CENTRE_VERTICAL)
        self.camera_fps_combo = wx.ComboBox(self, style=wx.CB_READONLY, size=(80, -1))
        self.camera_fps_combo.Enable(False)

        self.scan_camera_button = wx.Button(self, label="Scan for USB Cameras", size=(170, -1))
        self.start_camera_button = wx.Button(
            self, label="Start USB Camera", size=self.ctrl_button_size
        )
        self.start_camera_button.Enable(False)
        self.pause_button = wx.Button(self, label="Pause", size=self.ctrl_button_size)
        self.pause_button.Enable(False)
        self.stop_button = wx.Button(self, label="Stop", size=self.ctrl_button_size)
        self.stop_button.Enable(False)

        self.video_panel = VideoPanel(self, self.min_video_size)
        self.video_panel.SetMinSize(self.min_video_size)

        self.undo_button = wx.Button(self, label="Undo")
        self.redo_button = wx.Button(self, label="Redo")
        self.correct_button = wx.Button(self, label="Correct too small")

        self.custom_tile_size_static = wx.StaticText(
            self,
            label="Custom tile (w,h)",
            style=wx.ALIGN_CENTRE_VERTICAL | wx.ST_NO_AUTORESIZE,
        )

        self.custom_tile_width_edit = wx.TextCtrl(
            self,
            value=str(self.model_width * 2),
            size=(60, edit_size.height),
        )

        self.custom_tile_height_edit = wx.TextCtrl(
            self,
            value=str(self.model_height * 2),
            size=(60, edit_size.height),
        )

        self.playback_ratio_static = wx.StaticText(
            self,
            label="Playback ratio: 1.00",
            style=wx.ALIGN_CENTRE_VERTICAL,
        )
        self.playback_ratio_static.SetFont(pickerfont)

        self.position_static = wx.StaticText(
            self, style=wx.ALIGN_CENTRE_VERTICAL | wx.ST_NO_AUTORESIZE, size=(-1, edit_size.height)
        )

        self.position_static.SetFont(pickerfont)
        self.set_current_mouse_pos(wx.Point(0, 0))

        self.error_static = wx.StaticText(
            self,
            style=wx.ALIGN_CENTRE_VERTICAL,
        )
        self.error_static.SetFont(pickerfont)
        self.error_static.SetForegroundColour(wx.Colour(255, 0, 0))

        arrow_button_size = wx.Size(60, -1)
        self.up_button = wx.Button(self, label="Up", size=arrow_button_size)
        self.left_button = wx.Button(self, label="Left", size=arrow_button_size)
        self.right_button = wx.Button(self, label="Right", size=arrow_button_size)
        self.down_button = wx.Button(self, label="Down", size=arrow_button_size)

        # control panel
        self.resolution_static = wx.StaticText(
            self,
            label="Video Res",
            style=wx.ALIGN_CENTRE_VERTICAL | wx.ST_NO_AUTORESIZE,
            size=(-1, edit_size.height),
        )
        self.resolution_static.SetFont(pickerfont)

        self.resolution_edit = wx.TextCtrl(
            self, value="0x0", style=wx.TE_PROCESS_ENTER, size=(-1, edit_size.height)
        )
        self.resolution_edit.SetFont(pickerfont)
        self.resolution_edit.SetEditable(False)

        self.fps_static = wx.StaticText(
            self,
            label="Video FPS",
            style=wx.ALIGN_CENTRE_VERTICAL | wx.ST_NO_AUTORESIZE,
            size=(-1, edit_size.height),
        )
        self.fps_static.SetFont(pickerfont)

        self.fps_edit = wx.TextCtrl(
            self,
            value=f"{self.framerate:.2f}",
            style=wx.TE_PROCESS_ENTER,
            size=(-1, edit_size.height),
        )
        self.fps_edit.SetFont(pickerfont)
        self.fps_edit.SetEditable(False)

        self.ratio_static = wx.StaticText(
            self,
            label="Video Ratio",
            style=wx.ALIGN_CENTRE_VERTICAL | wx.ST_NO_AUTORESIZE,
            size=(-1, edit_size.height),
        )
        self.ratio_static.SetFont(pickerfont)

        self.ratio_edit = wx.TextCtrl(
            self, value=f"{self.ratio:.3f}", style=wx.TE_PROCESS_ENTER, size=(-1, edit_size.height)
        )
        self.ratio_edit.SetFont(pickerfont)
        self.ratio_edit.SetEditable(False)

        self.help_button = wx.Button(self, label="Help")

        self.model_width_static = wx.StaticText(
            self,
            label="Model width",
            style=wx.ALIGN_CENTRE_VERTICAL | wx.ST_NO_AUTORESIZE,
            size=(-1, edit_size.height),
        )
        self.model_width_static.SetFont(pickerfont)

        self.model_width_edit = wx.TextCtrl(
            self,
            value=str(self.model_width),
            style=wx.TE_PROCESS_ENTER,
            size=(-1, edit_size.height),
        )
        self.model_width_edit.SetFont(pickerfont)

        self.model_height_static = wx.StaticText(
            self,
            label="Model height",
            style=wx.ALIGN_CENTRE_VERTICAL | wx.ST_NO_AUTORESIZE,
            size=(-1, edit_size.height),
        )
        self.model_height_static.SetFont(pickerfont)

        self.model_height_edit = wx.TextCtrl(
            self,
            value=str(self.model_height),
            style=wx.TE_PROCESS_ENTER,
            size=(-1, edit_size.height),
        )
        self.model_height_edit.SetFont(pickerfont)

        self.load_json_button = wx.Button(
            self, label="Load JSON", size=(right_hand_side_width, -1)
        )

        self.tile_text_listbox = TileListBox(
            self,
            style=wx.BORDER_SIMPLE | wx.LB_ALWAYS_SB | wx.LB_HSCROLL | wx.LB_EXTENDED,
            size=wx.Size(-1, 200),
        )
        self.tile_text_listbox.SetFont(pickerfont)

        self.update_tiles_listbox()

        self.tile_text_ctrl = TileTextCtrl(self, style=wx.BORDER_SIMPLE | wx.TE_MULTILINE)
        self.tile_text_ctrl.SetFont(pickerfont)
        self.tile_text_ctrl.SetForegroundColour(wx.Colour(255, 127, 0))

        self.tile_edit_error = wx.StaticText(
            self,
            style=wx.ALIGN_CENTRE_VERTICAL | wx.BORDER_SIMPLE | wx.ST_NO_AUTORESIZE,
            size=(-1, edit_size[1]),
        )
        self.tile_edit_error.SetFont(pickerfont)
        self.tile_edit_error.SetForegroundColour(wx.Colour(255, 0, 0))

        self.tile_edit_button = wx.Button(self, label="Edit Tiles")
        self.tile_apply_button = wx.Button(self, label="Apply Tiles")
        self.tile_cancel_button = wx.Button(self, label="Cancel Edit")
        self.save_json_button = wx.Button(self, label="Save JSON")

    def _layout(self) -> None:
        self.SetSizer(None)

        outer_hsizer = wx.BoxSizer(wx.HORIZONTAL)
        outer_hsizer.AddSpacer(10)

        outer_vsizer_left = wx.BoxSizer(wx.VERTICAL)

        media_file_hsizer = wx.BoxSizer(wx.HORIZONTAL)
        media_file_hsizer.Add(self.video_filename_picker, 1, wx.EXPAND | wx.ALL)
        media_file_hsizer.AddSpacer(10)
        media_file_hsizer.Add(self.start_video_button, 0)
        media_file_hsizer.AddSpacer(10)
        media_file_hsizer.Add(self.stop_button, 0)

        rtsp_sizer = wx.BoxSizer(wx.HORIZONTAL)
        rtsp_sizer.Add(self.rtsp_url_static, 0, wx.ALIGN_CENTRE_VERTICAL)
        rtsp_sizer.AddSpacer(5)
        rtsp_sizer.Add(self.rtsp_url_edit, 1, wx.EXPAND | wx.ALL)
        rtsp_sizer.AddSpacer(10)
        rtsp_sizer.Add(self.start_rtsp_button, 0)
        rtsp_sizer.AddSpacer(self.ctrl_button_size.width + 10)

        camera_hsizer = wx.BoxSizer(wx.HORIZONTAL)
        camera_hsizer.Add(self.scan_camera_button, 0)
        camera_hsizer.AddSpacer(10)
        camera_hsizer.Add(self.camera_num_static, 0, wx.ALIGN_CENTRE_VERTICAL)
        camera_hsizer.AddSpacer(5)
        camera_hsizer.Add(self.camera_num_combo, 0)
        camera_hsizer.AddSpacer(7)
        camera_hsizer.Add(self.camera_res_static, 0, wx.ALIGN_CENTRE_VERTICAL)
        camera_hsizer.AddSpacer(5)
        camera_hsizer.Add(self.camera_res_combo, 0)
        camera_hsizer.AddSpacer(7)
        camera_hsizer.Add(self.camera_fmt_static, 0, wx.ALIGN_CENTRE_VERTICAL)
        camera_hsizer.AddSpacer(5)
        camera_hsizer.Add(self.camera_fmt_combo, 0)
        camera_hsizer.AddSpacer(7)
        camera_hsizer.Add(self.camera_fps_static, 0, wx.ALIGN_CENTRE_VERTICAL)
        camera_hsizer.AddSpacer(5)
        camera_hsizer.Add(self.camera_fps_combo, 0)
        camera_hsizer.AddStretchSpacer()
        camera_hsizer.Add(self.start_camera_button, 0, wx.EXPAND | wx.ALL)
        camera_hsizer.AddSpacer(10)
        camera_hsizer.Add(self.pause_button, 0)

        outer_vsizer_left.AddSpacer(10)
        outer_vsizer_left.Add(media_file_hsizer, 0, wx.EXPAND | wx.ALL)
        outer_vsizer_left.AddSpacer(10)
        outer_vsizer_left.Add(rtsp_sizer, 0, wx.EXPAND | wx.ALL)
        outer_vsizer_left.AddSpacer(10)
        outer_vsizer_left.Add(camera_hsizer, 0, wx.EXPAND | wx.ALL)
        outer_vsizer_left.AddSpacer(10)
        outer_vsizer_left.Add(self.video_panel, 1, wx.EXPAND | wx.ALL)

        position_hsizer = wx.BoxSizer(wx.HORIZONTAL)
        position_hsizer.Add(self.playback_ratio_static, 0)
        position_hsizer.AddSpacer(20)
        position_hsizer.Add(self.error_static, 1, wx.EXPAND | wx.ALL)
        position_hsizer.AddStretchSpacer()
        position_hsizer.Add(self.position_static, 0)
        outer_vsizer_left.AddSpacer(5)
        outer_vsizer_left.Add(position_hsizer, 0, wx.EXPAND | wx.ALL)

        arrow_buttons = wx.BoxSizer(wx.HORIZONTAL)
        arrow_buttons.Add(self.help_button, 0)
        arrow_buttons.AddSpacer(20)
        arrow_buttons.Add(self.custom_tile_size_static, 0, wx.ALIGN_CENTRE_VERTICAL)
        arrow_buttons.AddSpacer(2)
        arrow_buttons.Add(self.custom_tile_width_edit, 0)
        arrow_buttons.Add(self.custom_tile_height_edit, 0)
        arrow_buttons.AddSpacer(10)
        arrow_buttons.Add(self.correct_button, 0)
        arrow_buttons.AddSpacer(20)
        arrow_buttons.Add(self.up_button, 0)
        arrow_buttons.AddSpacer(5)
        arrow_buttons.Add(self.left_button, 0)
        arrow_buttons.AddSpacer(5)
        arrow_buttons.Add(self.right_button, 0)
        arrow_buttons.AddSpacer(5)
        arrow_buttons.Add(self.down_button, 0)
        arrow_buttons.AddStretchSpacer()
        arrow_buttons.Add(self.undo_button, 0)
        arrow_buttons.AddSpacer(10)
        arrow_buttons.Add(self.redo_button, 0)
        outer_vsizer_left.Add(arrow_buttons, 0, wx.EXPAND | wx.ALL)
        outer_vsizer_left.AddSpacer(10)

        model_size_gridsizer = wx.GridSizer(5, 2, 5, 5)
        model_size_gridsizer.Add(self.resolution_static, 0, wx.ALIGN_CENTRE_VERTICAL)
        model_size_gridsizer.Add(self.resolution_edit, 0, wx.EXPAND | wx.ALL)
        model_size_gridsizer.Add(self.fps_static, 0, wx.ALIGN_CENTRE_VERTICAL)
        model_size_gridsizer.Add(self.fps_edit, 0, wx.EXPAND | wx.ALL)
        model_size_gridsizer.Add(self.ratio_static, 0, wx.ALIGN_CENTRE_VERTICAL)
        model_size_gridsizer.Add(self.ratio_edit, 0, wx.EXPAND | wx.ALL)
        model_size_gridsizer.Add(self.model_width_static, 0, wx.ALIGN_CENTRE_VERTICAL)
        model_size_gridsizer.Add(self.model_width_edit, 0, wx.EXPAND | wx.ALL)
        model_size_gridsizer.Add(self.model_height_static, 0, wx.ALIGN_CENTRE_VERTICAL)
        model_size_gridsizer.Add(self.model_height_edit, 0, wx.EXPAND | wx.ALL)

        if self.editing_tiles:
            self.tile_text_listbox.Hide()
            self.tile_text_ctrl.Show()
        else:
            self.tile_text_ctrl.Hide()
            self.tile_text_listbox.Show()

        outer_vsizer_right = wx.BoxSizer(wx.VERTICAL)
        outer_vsizer_right.AddSpacer(10)
        outer_vsizer_right.Add(model_size_gridsizer, 0, wx.EXPAND | wx.ALL)
        outer_vsizer_right.AddSpacer(10)
        outer_vsizer_right.Add(self.load_json_button, 0, wx.EXPAND | wx.RIGHT)
        outer_vsizer_right.AddSpacer(10)
        outer_vsizer_right.Add(
            self.tile_text_ctrl if self.editing_tiles else self.tile_text_listbox,
            1,
            wx.EXPAND | wx.ALL,
        )

        if not self.editing_tiles:
            self.tile_edit_button.Show()
            self.tile_apply_button.Hide()
            self.tile_cancel_button.Hide()
            self.tile_edit_error.Hide()
            outer_vsizer_right.Add(self.tile_edit_button, 0, wx.EXPAND | wx.ALL)
        else:
            self.tile_edit_button.Hide()
            self.tile_apply_button.Show()
            self.tile_cancel_button.Show()
            self.tile_edit_error.Show()
            outer_vsizer_right.Add(self.tile_edit_error, 0, wx.EXPAND | wx.ALL)
            button_hsizer = wx.BoxSizer(wx.HORIZONTAL)
            button_hsizer.Add(self.tile_cancel_button, 1, wx.EXPAND | wx.ALL)
            button_hsizer.AddSpacer(10)
            button_hsizer.Add(self.tile_apply_button, 1, wx.EXPAND | wx.ALL)
            outer_vsizer_right.Add(button_hsizer, 0, wx.EXPAND | wx.ALL)
        outer_vsizer_right.AddSpacer(10)
        outer_vsizer_right.Add(self.save_json_button, 0, wx.EXPAND | wx.ALL)
        outer_vsizer_right.AddSpacer(10)

        outer_hsizer.Add(outer_vsizer_left, 1, wx.EXPAND | wx.ALL)
        outer_hsizer.AddSpacer(10)
        outer_hsizer.Add(outer_vsizer_right, 0, wx.EXPAND | wx.ALL)
        outer_hsizer.AddSpacer(10)

        final_sizer = wx.BoxSizer(wx.VERTICAL)
        final_sizer.Add(outer_hsizer, 1, wx.EXPAND | wx.ALL)

        self.SetSizer(final_sizer)
        self.Layout()
        self.Refresh()
        self.parent.Fit()

    def _bind(self) -> None:
        self.start_video_button.Bind(wx.EVT_BUTTON, self._start_video)
        self.start_rtsp_button.Bind(wx.EVT_BUTTON, self._start_rtsp)
        self.start_camera_button.Bind(wx.EVT_BUTTON, self._start_camera)
        self.pause_button.Bind(wx.EVT_BUTTON, self._pause_play)
        self.stop_button.Bind(wx.EVT_BUTTON, self._stop)
        self.rtsp_url_edit.Bind(wx.EVT_TEXT, self._rtsp_url_changed)
        self.model_width_edit.Bind(wx.EVT_TEXT, self._model_width_changed)
        self.model_height_edit.Bind(wx.EVT_TEXT, self._model_height_changed)
        self.tile_edit_button.Bind(wx.EVT_BUTTON, self._edit_tiles)
        self.tile_apply_button.Bind(wx.EVT_BUTTON, self._apply_tiles)
        self.tile_cancel_button.Bind(wx.EVT_BUTTON, self._quit_edit_tiles)
        self.load_json_button.Bind(wx.EVT_BUTTON, self._load_json)
        self.save_json_button.Bind(wx.EVT_BUTTON, self._save_json)
        self.up_button.Bind(wx.EVT_BUTTON, self._move_selected_up)
        self.left_button.Bind(wx.EVT_BUTTON, self._move_selected_left)
        self.right_button.Bind(wx.EVT_BUTTON, self._move_selected_right)
        self.down_button.Bind(wx.EVT_BUTTON, self._move_selected_down)
        self.help_button.Bind(wx.EVT_BUTTON, self._show_help)
        self.Bind(wx.EVT_TIMER, self.on_timer)
        self.camera_res_combo.Bind(wx.EVT_COMBOBOX, lambda event: self._fill_camera_combos('res'))
        self.camera_fmt_combo.Bind(wx.EVT_COMBOBOX, lambda event: self._fill_camera_combos('fmt'))
        self.camera_num_combo.Bind(wx.EVT_COMBOBOX, lambda event: self._fill_camera_combos('num'))
        self.scan_camera_button.Bind(wx.EVT_BUTTON, self._scan_cameras)
        self.Bind(EVT_CAMERAS_READ, self._notify_cameras_read)
        self.correct_button.Bind(wx.EVT_BUTTON, self._correct_too_small_tiles)
        self.undo_button.Bind(wx.EVT_BUTTON, self.undo)
        self.redo_button.Bind(wx.EVT_BUTTON, self.redo)
        self.video_filename_picker.Bind(wx.EVT_FILEPICKER_CHANGED, self._picker_changed)

    def on_timer(self, _event: wx.Event) -> None:
        """Pull frame from queue and display it."""
        try:
            frame = self.frame_queue.get(block=False)
            self.video_panel.set_frame(frame)
            # Explicitly delete frame reference to free memory
            del frame
            self._check_tiles()
        except queue.Empty:
            # No frame available yet
            pass

    def _scan_cameras(self, _event: wx.Event | None = None) -> None:
        """Scan for available cameras."""
        self.scan_camera_button.Enable(False)
        self.scan_camera_button.SetLabel("Scanning...")
        self.camera_discovery_thread = CameraDiscoveryThread(self)
        self.camera_discovery_thread.start()

    def _notify_cameras_read(self, event: CameraDiscoveryUpdateEvent) -> None:
        """Handle completion of camera scan."""
        self.camera_info = event.camera_info
        camera_choices = [str(i) for i in range(len(self.camera_info))]
        self.camera_num_combo.SetItems(camera_choices)
        if camera_choices:
            self.camera_num_combo.SetSelection(0)
            self._fill_camera_combos('num')
        else:
            self.camera_res_combo.SetItems([])
            self.camera_fmt_combo.SetItems([])
            self.camera_fps_combo.SetItems([])
        self.scan_camera_button.Enable(True)
        self.scan_camera_button.SetLabel("Scan Cameras")
        enable = self.camera_num_combo.GetStringSelection() != ""
        self.start_camera_button.Enable(enable)
        self.camera_num_combo.Enable(enable)
        self.camera_res_combo.Enable(enable)
        self.camera_fmt_combo.Enable(enable)
        self.camera_fps_combo.Enable(enable)

    def stop_capture_thread(self) -> None:
        if self.capture_thread:
            if self.capture_thread.running:
                self.capture_thread.stop()
                self.capture_thread.join(timeout=1.0)
            self.capture_thread = None
            # Clear queue and delete all frames to free memory
            while not self.frame_queue.empty():
                try:
                    frame = self.frame_queue.get_nowait()  # noqa: F841
                    del frame
                except queue.Empty:
                    break
        # Clear current frame in video panel
        if hasattr(self, 'video_panel') and self.video_panel.frame is not None:
            del self.video_panel.frame
            self.video_panel.frame = None

    def on_close(self, _event: wx.Event) -> None:
        """Clean up when closing."""
        self.timer.Stop()
        self.stop_capture_thread()
        # Explicitly cleanup video panel resources
        if hasattr(self, 'video_panel'):
            self.video_panel.cleanup()

    def _picker_changed(self, _event: wx.Event) -> None:
        self.start_video_button.Enable(self.video_filename_picker.GetPath() != "")

    def _get_all_resolutions_for_camera(self, camera_index: int) -> list[str]:
        """Get all supported resolutions for the given camera index."""
        if camera_index < 0 or camera_index >= len(self.camera_info):
            return []
        camera = self.camera_info[camera_index]
        resolutions = set()
        for cam_config in camera['supported_configs']:
            resolutions.add(cam_config['resolution'])

        s_resolutions = sorted(list(resolutions), key=lambda x: (x[0], x[1]), reverse=True)
        result = [_resolution_tuple_to_str(res) for res in s_resolutions]
        return result

    def _get_all_formats_for_camera_and_resolution(
        self, camera_index: int, resolution: tuple[int, int] | None
    ) -> list[str]:
        """Get all supported formats for the given camera index and resolution."""
        if camera_index >= len(self.camera_info) or resolution is None:
            return []
        camera = self.camera_info[camera_index]
        formats = set()
        for cam_config in camera['supported_configs']:
            if cam_config['resolution'] == resolution:
                formats.add(cam_config['format'])
        result = sorted(list(formats))
        return result

    def _get_all_fps_for_camera_resolution_and_format(
        self, camera_index: int, resolution: tuple[int, int] | None, fmt: str
    ) -> list[int]:
        """Get all supported fps for the given camera index, resolution, and format."""
        if camera_index >= len(self.camera_info) or resolution is None or fmt == "":
            return []
        camera = self.camera_info[camera_index]
        fps_values = set()
        for cam_config in camera['supported_configs']:
            if cam_config['resolution'] == resolution and cam_config['format'] == fmt:
                for fps in cam_config['supported_fps']:
                    try:
                        fps_values.add(int(float(fps)))
                    except ValueError:
                        continue
        result: list[int] = [i for i in sorted(list(fps_values), reverse=True)]
        return result

    def _fill_camera_combos(self, changed: str) -> None:
        cam_num_selection_str = self.camera_num_combo.GetStringSelection()
        if cam_num_selection_str:
            cam_num_selection = int(cam_num_selection_str)
            if changed == 'num':
                last_cam_res_selection = self.camera_res_combo.GetStringSelection()
                self.camera_res_combo.SetItems(
                    self._get_all_resolutions_for_camera(cam_num_selection)
                )
                if self.camera_res_combo.FindString(last_cam_res_selection) != wx.NOT_FOUND:
                    self.camera_res_combo.SetSelection(
                        self.camera_res_combo.FindString(last_cam_res_selection)
                    )
                else:
                    self.camera_res_combo.SetSelection(0)

            if changed in ['num', 'res']:
                cam_res_selection = self.camera_res_combo.GetStringSelection()
                last_cam_fmt_selection = self.camera_fmt_combo.GetStringSelection()
                formats = self._get_all_formats_for_camera_and_resolution(
                    cam_num_selection, _extract_resolution_tuple(cam_res_selection)
                )
                self.camera_fmt_combo.SetItems(formats)
                if self.camera_fmt_combo.FindString(last_cam_fmt_selection) != wx.NOT_FOUND:
                    self.camera_fmt_combo.SetSelection(
                        self.camera_fmt_combo.FindString(last_cam_fmt_selection)
                    )
                else:
                    self.camera_fmt_combo.SetSelection(0)

            if changed in ['num', 'res', 'fmt']:
                cam_res_selection = self.camera_res_combo.GetStringSelection()
                cam_fmt_selection = self.camera_fmt_combo.GetStringSelection()
                last_cam_fps_selection = self.camera_fps_combo.GetStringSelection()
                fpss = self._get_all_fps_for_camera_resolution_and_format(
                    cam_num_selection,
                    _extract_resolution_tuple(cam_res_selection),
                    cam_fmt_selection,
                )
                self.camera_fps_combo.SetItems([str(fps) for fps in fpss])
                if self.camera_fps_combo.FindString(last_cam_fps_selection) != wx.NOT_FOUND:
                    self.camera_fps_combo.SetSelection(
                        self.camera_fps_combo.FindString(last_cam_fps_selection)
                    )
                else:
                    self.camera_fps_combo.SetSelection(0)
        else:
            self.camera_res_combo.SetItems([])
            self.camera_fmt_combo.SetItems([])
            self.camera_fps_combo.SetItems([])

    def _check_tiles(self) -> None:
        warning = False
        for tile in self.tiles:
            if tile[2] < self.model_width or tile[3] < self.model_height:
                warning = True

        current_warning = self.error_static.GetLabel()
        if warning and current_warning == "":
            self.error_static.SetLabel("Warning: Some tiles are smaller than model size")
        elif not warning and current_warning != "":
            self.error_static.SetLabel("")

    def _actual_to_screen(self, actual_value: int, ratio: float) -> int:
        if ratio == 0.0:
            return actual_value
        return int(float(actual_value) / ratio)

    def _actual_to_screen_x(self, actual_value: int) -> int:
        return self._actual_to_screen(actual_value, self.x_ratio)

    def _actual_to_screen_y(self, actual_value: int) -> int:
        return self._actual_to_screen(actual_value, self.y_ratio)

    def _screen_to_actual(self, screen_value: int, ratio: float) -> int:
        if ratio == 0.0:
            return screen_value
        return int(float(screen_value) * ratio)

    def _screen_to_actual_x(self, screen_value: int) -> int:
        return self._screen_to_actual(screen_value, self.x_ratio)

    def _screen_to_actual_y(self, screen_value: int) -> int:
        return self._screen_to_actual(screen_value, self.y_ratio)

    def _load_json(self, _event: wx.Event) -> None:
        open_dialog = wx.FileDialog(
            self,
            message="Open JSON file",
            wildcard="JSON files (*.json)|*.json",
            style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST,
        )
        open_dialog.ShowModal()
        self.json_filename = open_dialog.GetPath()
        self.tile_history.add_state(self.tiles)
        with open(self.json_filename, 'r') as f:
            self.tiles = json.loads(f.read())

        # if the filename contains the model size then set it to that
        match = re.search(r"model(\d{1,5})x(\d{1,5})", self.json_filename)
        if match:
            self.model_width_edit.SetValue(match.group(1))
            self.model_height_edit.SetValue(match.group(2))
        self.update_tiles_listbox()
        just_filename = os.path.basename(self.json_filename)
        self.parent.set_title(just_filename)

    def _save_json(self, _event: wx.Event) -> None:
        save_dialog = wx.FileDialog(
            self,
            message="Save JSON file",
            wildcard="JSON files (*.json)|*.json",
            style=wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT,
        )
        save_dialog.SetPath(self.json_filename if self.json_filename != '' else 'tiles.json')
        save_dialog.ShowModal()
        filename = save_dialog.GetPath()
        with open(filename, 'w') as f:
            f.write(json.dumps(self.tiles, indent=4))

    def _edit_tiles(self, _event: wx.Event) -> None:
        self.editing_tiles = True
        tiles_str = self.tile_text_listbox.GetStrings()
        self.tile_text_ctrl.SetValue('\n'.join(tiles_str))
        self._layout()
        wx.CallLater(100, self.tile_text_ctrl.SetFocus)

    def _apply_tiles(self, _event: wx.Event) -> None:
        self.editing_tiles = False
        self.tile_history.add_state(self.tiles)
        tiles_strs = self.tile_text_ctrl.GetValue().splitlines()
        self.tiles, error = _convert_string_to_tiles(tiles_strs)
        if error == '':
            self.update_tiles_listbox(persist_selections=False)
        self._layout()

    def _quit_edit_tiles(self, _event: wx.Event | None = None) -> None:
        self.editing_tiles = False
        self.update_tiles_listbox(persist_selections=False)
        self._layout()

    def _model_width_changed(self, _event: wx.Event) -> None:
        try:
            self.model_width = int(self.model_width_edit.GetValue())
        except ValueError:
            self.model_width = 0

    def _model_height_changed(self, _event: wx.Event) -> None:
        try:
            self.model_height = int(self.model_height_edit.GetValue())
        except ValueError:
            self.model_height = 0

    def set_current_mouse_pos(self, point: wx.Point) -> None:
        self.current_x = self._screen_to_actual_x(point.x)
        self.current_y = self._screen_to_actual_y(point.y)
        self.position_static.SetLabel(f"Mouse x={self.current_x:>04}, y={self.current_y:>04}")

    def _set_video_resolutiuon(self, resolution: wx.Size) -> None:
        self.resolution = resolution
        self.ratio = (
            self.resolution.width / self.resolution.height if self.resolution.height != 0 else 1
        )
        self.resolution_edit.SetValue(f"{self.resolution.width}x{self.resolution.height}")
        self.ratio_edit.SetValue(f"{self.ratio:.3f}")
        self.set_screen_to_video_ratios()

    def set_screen_to_video_ratios(self) -> None:
        playback_size = self.video_panel.GetSize()
        playback_ratio = (
            playback_size.width / playback_size.height if playback_size.height != 0 else 1.0
        )
        self.x_ratio = self.resolution.width / playback_size.width
        self.y_ratio = self.resolution.height / playback_size.height
        self.playback_ratio_static.SetLabel(f"Playback ratio: {playback_ratio:.2f}")

    def get_selections(self) -> list[int]:
        return self.tile_text_listbox.GetSelections()

    def are_selections(self) -> bool:
        return len(self.get_selections()) > 0

    def is_tile_selected(self, tile: list[int]) -> bool:
        tile_str = _get_string_from_tile(tile)
        selected_indices = self.get_selections()
        for index in selected_indices:
            if self.tile_text_listbox.GetString(index) == tile_str:
                return True
        return False

    def _start_video(self, _event: wx.Event | None = None) -> None:
        filename = self.video_filename_picker.GetPath()
        if not os.path.isfile(filename):
            wx.MessageBox(f"Video file not found: {filename}", "Error", wx.OK | wx.ICON_ERROR)
            self.pause_button.Enable(False)
            self.stop_button.Enable(False)
            return

        self.timer.Stop()
        self.stop_capture_thread()
        self.scan_camera_button.Enable(True)

        self.capture_thread = VideoCaptureThread(filename, self.frame_queue)
        video_width, video_height = self.capture_thread.get_resolution()
        self.framerate = self.capture_thread.get_fps()

        self.fps_edit.SetValue(f"{self.framerate:.2f}")
        ratio = video_width / video_height
        self.video_panel.SetMinSize(wx.Size(video_panel_width, int(video_panel_width // ratio)))
        self._set_video_resolutiuon(wx.Size(video_width, video_height))
        self._layout()
        self.Refresh()

        # Create and start video capture thread
        self.capture_thread.start()
        # Create timer to pull frames from queue at the framerate of the video
        self.timer.Start(int(1000 // self.framerate))

        self.stop_button.Enable(True)
        self.pause_button.Enable(True)
        self.pause_button.SetLabel("Pause")

    def _start_rtsp(self, _event: wx.Event | None = None) -> None:
        url = self.rtsp_url_edit.GetValue().strip()
        if not url:
            wx.MessageBox("Please enter an RTSP URL.", "Error", wx.OK | wx.ICON_ERROR)
            return

        self.timer.Stop()
        self.stop_capture_thread()
        self.scan_camera_button.Enable(True)

        try:
            self.capture_thread = VideoCaptureThread(url, self.frame_queue)
        except RuntimeError as e:
            wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)
            self.pause_button.Enable(False)
            self.stop_button.Enable(False)
            return

        video_width, video_height = self.capture_thread.get_resolution()
        if video_width == 0 or video_height == 0:
            wx.MessageBox(
                f"Could not retrieve resolution from RTSP stream: {url}",
                "Error",
                wx.OK | wx.ICON_ERROR,
            )
            self.stop_capture_thread()
            self.pause_button.Enable(False)
            self.stop_button.Enable(False)
            return

        self.framerate = self.capture_thread.get_fps()
        if self.framerate <= 0:
            self.framerate = 30.0

        self.fps_edit.SetValue(f"{self.framerate:.2f}")
        ratio = video_width / video_height
        self.video_panel.SetMinSize(wx.Size(video_panel_width, int(video_panel_width // ratio)))
        self._set_video_resolutiuon(wx.Size(video_width, video_height))
        self._layout()
        self.Refresh()

        # Create and start RTSP capture thread
        self.capture_thread.start()
        # Create timer to pull frames from queue at the stream framerate
        self.timer.Start(int(1000 // self.framerate))

        self.stop_button.Enable(True)
        self.pause_button.Enable(True)
        self.pause_button.SetLabel("Pause")

    def _start_camera(self, _event: wx.Event | None = None) -> None:
        try:
            camera_num = int(self.camera_num_combo.GetStringSelection())
        except ValueError:
            wx.MessageBox(
                f"Invalid camera ID: {self.camera_num_combo.GetStringSelection()}, must be "
                "an integer",
                "Error",
                wx.OK | wx.ICON_ERROR,
            )
            self.pause_button.Enable(False)
            self.stop_button.Enable(False)
            return

        self.timer.Stop()
        self.stop_capture_thread()

        camera_res = _extract_resolution_tuple(self.camera_res_combo.GetStringSelection())
        camera_fmt = self.camera_fmt_combo.GetStringSelection()
        camera_fps = int(self.camera_fps_combo.GetStringSelection())

        if camera_res is None:
            wx.MessageBox(
                f"Invalid camera resolution: {self.camera_res_combo.GetStringSelection()}",
                "Error",
                wx.OK | wx.ICON_ERROR,
            )
            self.pause_button.Enable(False)
            self.stop_button.Enable(False)
            return

        if camera_fmt == "":
            wx.MessageBox(
                f"Invalid camera format: {camera_fmt}",
                "Error",
                wx.OK | wx.ICON_ERROR,
            )
            self.pause_button.Enable(False)
            self.stop_button.Enable(False)
            return

        try:
            self.capture_thread = CameraCaptureThread(
                camera_num, camera_res, camera_fmt, camera_fps, self.frame_queue
            )
        except RuntimeError as e:
            wx.MessageBox(str(e), "Error", wx.OK | wx.ICON_ERROR)
            self.pause_button.Enable(False)
            self.stop_button.Enable(False)
            return

        self.scan_camera_button.Enable(False)

        video_width, video_height = self.capture_thread.get_resolution()
        self.framerate = self.capture_thread.get_fps()

        self.fps_edit.SetValue(f"{self.framerate:.2f}")
        ratio = video_width / video_height
        self.video_panel.SetMinSize(wx.Size(video_panel_width, int(video_panel_width // ratio)))
        self._set_video_resolutiuon(wx.Size(video_width, video_height))
        self._layout()
        self.Refresh()

        # Create and start video capture thread
        self.capture_thread.start()
        # Create timer to pull frames from queue at the framerate of the video
        self.timer.Start(int(1000 // self.framerate))

        self.stop_button.Enable(True)
        self.pause_button.Enable(True)
        self.pause_button.SetLabel("Pause")

    def _pause_play(self, _event: wx.Event | None = None) -> None:
        if self.capture_thread is not None:
            if self.timer.IsRunning():
                self.capture_thread._pause()
                self.timer.Stop()
                self.pause_button.SetLabel("Play")
            else:
                self.timer.Start(int(1000 // self.framerate))
                self.pause_button.SetLabel("Pause")
                self.capture_thread._restart()

    def _stop(self, _event: wx.Event | None = None) -> None:
        self.timer.Stop()
        self.stop_capture_thread()
        self.video_panel.set_frame(None)
        self._set_video_resolutiuon(wx.Size(0, 0))
        self.scan_camera_button.Enable(True)
        self.pause_button.Enable(False)
        self.stop_button.Enable(False)
        self.pause_button.SetLabel("Pause")

    def add_tile(self, tile: list[int]) -> None:
        if self.resolution.width > 0 and self.resolution.height > 0:
            self.tile_history.add_state(self.tiles)
            self.tiles.append(tile)
            self.update_tiles_listbox()

    def add_tile_from_screen_coords(self, new_rect: wx.Rect) -> None:
        x = self._screen_to_actual_x(new_rect.x)
        y = self._screen_to_actual_y(new_rect.y)
        w = self._screen_to_actual_x(new_rect.width)
        h = self._screen_to_actual_y(new_rect.height)
        self.add_tile([x, y, w, h])

    def _correct_too_small_tiles(self, _event: wx.Event) -> None:
        self.tile_history.add_state(self.tiles)
        for tile in self.tiles:
            if tile[2] < self.model_width or tile[3] < self.model_height:
                tile[2] = self.model_width
                tile[3] = self.model_height
        self.update_tiles_listbox()

    def _move_selected_tiles_by_absolute(self, x_delta: int, y_delta: int) -> None:
        self.tile_history.add_state(self.tiles)
        for index, tile in enumerate(self.tiles):
            tile_str = _get_string_from_tile(tile)
            selected_indices = self.get_selections()
            for index in selected_indices:
                if self.tile_text_listbox.GetString(index) == tile_str:
                    new_x = tile[0] + x_delta
                    new_y = tile[1] + y_delta

                    if new_x < 0:
                        new_x = 0
                    elif new_x + tile[2] > self.resolution.width:
                        new_x = self.resolution.width - tile[2]

                    if new_y < 0:
                        new_y = 0
                    elif new_y + tile[3] > self.resolution.height:
                        new_y = self.resolution.height - tile[3]
                    tile[0] = new_x
                    tile[1] = new_y
        self.update_tiles_listbox()

    def _move_selected_up(self, event: wx.Event) -> None:
        self._move_selected_tiles_by_absolute(0, -1)

    def _move_selected_down(self, event: wx.Event) -> None:
        self._move_selected_tiles_by_absolute(0, 1)

    def _move_selected_left(self, event: wx.Event) -> None:
        self._move_selected_tiles_by_absolute(-1, 0)

    def _move_selected_right(self, event: wx.Event) -> None:
        self._move_selected_tiles_by_absolute(1, 0)

    def _show_help(self, _event: wx.Event) -> None:
        # print(self.parent.GetSize())
        dialog_size = wx.Size(800, 600)
        dlg = help_dialog(self, dialog_size, self.version_str)
        dlg.ShowModal()
        dlg.Destroy()

    def _correct_deltas(
        self, xdelta: int, ydelta: int, delta_from_original_point: bool = True
    ) -> tuple[int, int]:
        new_xdelta = xdelta
        new_ydelta = ydelta
        for index, tile in enumerate(self.tiles):
            tile_str = _get_string_from_tile(tile)
            selected_indices = self.get_selections()
            for index in selected_indices:
                if self.tile_text_listbox.GetString(index) == tile_str:
                    if delta_from_original_point:
                        new_x = self.move_tiles[index][0] + self._screen_to_actual_x(xdelta)
                        new_y = self.move_tiles[index][1] + self._screen_to_actual_y(ydelta)
                    else:
                        new_x = tile[0] + self._screen_to_actual_x(xdelta)
                        new_y = tile[1] + self._screen_to_actual_y(ydelta)
                    if new_x < 0:
                        new_xdelta = xdelta + self._actual_to_screen_x(abs(new_x))
                        new_x = 0
                    elif new_x + tile[2] > self.resolution.width:
                        new_xdelta = xdelta - self._actual_to_screen_x(
                            new_x + tile[2] - self.resolution.width
                        )
                        new_x = self.resolution.width - tile[2]

                    if new_y < 0:
                        new_ydelta = ydelta + self._actual_to_screen_y(abs(new_y))
                        new_y = 0
                    elif new_y + tile[3] > self.resolution.height:
                        new_ydelta = ydelta - self._actual_to_screen_y(
                            (new_y + tile[3] - self.resolution.height)
                        )
                        new_y = self.resolution.height - tile[3]
        return (new_xdelta, new_ydelta)

    def move_selected_titles(
        self, xdelta: int, ydelta: int, delta_from_original_point: bool = True
    ) -> None:
        self.tile_history.add_state(self.tiles)

        xdelta, ydelta = self._correct_deltas(xdelta, ydelta, delta_from_original_point)

        for index, tile in enumerate(self.tiles):
            tile_str = _get_string_from_tile(tile)
            selected_indices = self.get_selections()
            for index in selected_indices:
                if self.tile_text_listbox.GetString(index) == tile_str:
                    if delta_from_original_point:
                        new_x = self.move_tiles[index][0] + self._screen_to_actual_x(xdelta)
                        new_y = self.move_tiles[index][1] + self._screen_to_actual_y(ydelta)
                    else:
                        new_x = tile[0] + self._screen_to_actual_x(xdelta)
                        new_y = tile[1] + self._screen_to_actual_y(ydelta)

                    if new_x < 0:
                        new_x = 0
                    elif new_x + tile[2] > self.resolution.width:
                        new_x = self.resolution.width - tile[2]

                    if new_y < 0:
                        new_y = 0
                    elif new_y + tile[3] > self.resolution.height:
                        new_y = self.resolution.height - tile[3]
                    tile[0] = new_x
                    tile[1] = new_y
        self.update_tiles_listbox()

    def _get_tile_index(self, tile: list[int]) -> int:
        return self.tile_text_listbox.FindString(_get_string_from_tile(tile))

    def size_corner(
        self, resizing_tile: list[int] | None, current_screen_point: wx.Point, corner_status: str
    ) -> bool:
        hit_min_limit = False
        if corner_status and resizing_tile is not None:
            x = resizing_tile[0]
            y = resizing_tile[1]
            x1 = x + resizing_tile[2]
            y1 = y + resizing_tile[3]

            if corner_status == 'BR':
                x1 = self._screen_to_actual_x(current_screen_point.x)
                y1 = self._screen_to_actual_y(current_screen_point.y)
                if x1 - x < MIN_TILE_SIZE:
                    x1 = x + MIN_TILE_SIZE
                    hit_min_limit = True
                if y1 - y < MIN_TILE_SIZE:
                    y1 = y + MIN_TILE_SIZE
                    hit_min_limit = True
            elif corner_status == 'BL':
                x = self._screen_to_actual_x(current_screen_point.x)
                y1 = self._screen_to_actual_y(current_screen_point.y)
                if x1 - x < MIN_TILE_SIZE:
                    x = x1 - MIN_TILE_SIZE
                    hit_min_limit = True
                if y1 - y < MIN_TILE_SIZE:
                    y1 = y + MIN_TILE_SIZE
                    hit_min_limit = True
            elif corner_status == 'TR':
                x1 = self._screen_to_actual_x(current_screen_point.x)
                y = self._screen_to_actual_y(current_screen_point.y)
                if x1 - x < MIN_TILE_SIZE:
                    x1 = x + MIN_TILE_SIZE
                    hit_min_limit = True
                if y1 - y < MIN_TILE_SIZE:
                    y = y1 - MIN_TILE_SIZE
                    hit_min_limit = True
            elif corner_status == 'TL':
                x = self._screen_to_actual_x(current_screen_point.x)
                y = self._screen_to_actual_y(current_screen_point.y)
                if x1 - x < MIN_TILE_SIZE:
                    x = x1 - MIN_TILE_SIZE
                    hit_min_limit = True
                if y1 - y < MIN_TILE_SIZE:
                    y = y1 - MIN_TILE_SIZE
                    hit_min_limit = True

            if x < 0:
                x = 0
            if y < 0:
                y = 0
            if x1 > self.resolution.width:
                x1 = self.resolution.width
            if y1 > self.resolution.height:
                y1 = self.resolution.height

            for tile in self.tiles:
                if tile == resizing_tile:
                    tile[0] = x
                    tile[1] = y
                    tile[2] = x1 - x
                    tile[3] = y1 - y
            self.update_tiles_listbox()
        return hit_min_limit

    def size_selected_tiles(self, wdelta: int, hdelta: int, corner: str = 'BR') -> None:
        self.tile_history.add_state(self.tiles)
        for index, tile in enumerate(self.tiles):
            tile_str = _get_string_from_tile(tile)
            selected_indices = self.get_selections()
            for index in selected_indices:
                if self.tile_text_listbox.GetString(index) == tile_str:
                    if corner == 'BR':
                        tile[2] += self._screen_to_actual_x(wdelta)
                        tile[3] += self._screen_to_actual_y(hdelta)
                    elif corner == 'BL':
                        tile[0] += self._screen_to_actual_x(wdelta)
                        tile[2] -= self._screen_to_actual_x(wdelta)
                        tile[3] += self._screen_to_actual_y(hdelta)
                    elif corner == 'TR':
                        tile[1] += self._screen_to_actual_y(hdelta)
                        tile[2] += self._screen_to_actual_x(wdelta)
                        tile[3] -= self._screen_to_actual_y(hdelta)
                    elif corner == 'TL':
                        tile[0] += self._screen_to_actual_x(wdelta)
                        tile[1] += self._screen_to_actual_y(hdelta)
                        tile[2] -= self._screen_to_actual_x(wdelta)
                        tile[3] -= self._screen_to_actual_y(hdelta)
        self.update_tiles_listbox()

    def get_tiles_at_point(self, point: wx.Point) -> list[list[int]]:
        x = self._screen_to_actual_x(point.x)
        y = self._screen_to_actual_y(point.y)
        tiles_at_point = []
        for tile in self.tiles:
            if x >= tile[0] and x <= tile[0] + tile[2] and y >= tile[1] and y <= tile[1] + tile[3]:
                tiles_at_point.append(tile)
        return tiles_at_point

    def are_any_tiles_at_point_selected(self, point: wx.Point) -> bool:
        tiles_at_point = self.get_tiles_at_point(point)
        for tile in tiles_at_point:
            if self.is_tile_selected(tile):
                return True
        return False

    def select_tiles_at_point(self, point: wx.Point, shift_down: bool) -> None:
        if not shift_down:
            self.tile_text_listbox.SetSelection(wx.NOT_FOUND)
        tiles = self.get_tiles_at_point(point)
        for tile in tiles:
            index = self._get_tile_index(tile)
            if index != wx.NOT_FOUND:
                if self.tile_text_listbox.IsSelected(index):
                    self.tile_text_listbox.Deselect(index)
                else:
                    self.tile_text_listbox.SetSelection(index)
        self.create_move_tiles_list()

    def create_move_tiles_list(self) -> None:
        self.move_tiles = copy.deepcopy(self.tiles)

    def update_tiles_listbox(self, persist_selections: bool = True) -> None:
        if persist_selections:
            selections = self.get_selections()
        else:
            selections = []
        self.tile_text_listbox.Clear()
        for tile in self.tiles:
            self.tile_text_listbox.Append(_get_string_from_tile(tile))

        if persist_selections:
            for index in selections:
                self.tile_text_listbox.SetSelection(index)
        self.update_video_panel()
        self._updateundoredo_buttons()

    def _updateundoredo_buttons(self) -> None:
        self.undo_button.Enable(self.tile_history.canundo())
        self.redo_button.Enable(self.tile_history.canredo())

    def undo(self, _event: wx.Event | None = None) -> None:
        new_tiles = self.tile_history.undo(self.tiles)
        if new_tiles is not None:
            self.tiles = new_tiles
        self.update_tiles_listbox(persist_selections=False)
        self._updateundoredo_buttons()

    def redo(self, _event: wx.Event | None = None) -> None:
        new_tiles = self.tile_history.redo(self.tiles)
        if new_tiles is not None:
            self.tiles = new_tiles
        self.update_tiles_listbox(persist_selections=False)
        self._updateundoredo_buttons()

    def update_video_panel(self) -> None:
        if not self.timer.IsRunning():
            self.video_panel.draw_frame()

    def create_tile(self, point: wx.Point, size: tuple[int, int]) -> None:
        self.add_tile(
            [
                self._screen_to_actual_x(point.x),
                self._screen_to_actual_y(point.y),
                size[0],
                size[1],
            ]
        )

    def get_corner_on_mouse(self, mouse_point: wx.Point) -> tuple[list[int] | None, str]:
        for tile in self.tiles:
            screen_x = self._actual_to_screen_x(tile[0])
            screen_y = self._actual_to_screen_y(tile[1])
            screen_w = self._actual_to_screen_x(tile[2])
            screen_h = self._actual_to_screen_y(tile[3])
            corner_size = TILE_RESIZE_CORNER_SIZE
            if screen_w > corner_size or screen_h > corner_size:
                if (
                    mouse_point.x >= screen_x - corner_size
                    and mouse_point.x <= screen_x + corner_size
                    and mouse_point.y >= screen_y - corner_size
                    and mouse_point.y <= screen_y + corner_size
                ):
                    return tile, 'TL'
                elif (
                    mouse_point.x >= screen_x + screen_w - corner_size
                    and mouse_point.x <= screen_x + screen_w + corner_size
                    and mouse_point.y >= screen_y - corner_size
                    and mouse_point.y <= screen_y + corner_size
                ):
                    return tile, 'TR'
                elif (
                    mouse_point.x >= screen_x - corner_size
                    and mouse_point.x <= screen_x + corner_size
                    and mouse_point.y >= screen_y + screen_h - corner_size
                    and mouse_point.y <= screen_y + screen_h + corner_size
                ):
                    return tile, 'BL'
                elif (
                    mouse_point.x >= screen_x + screen_w - corner_size
                    and mouse_point.x <= screen_x + screen_w + corner_size
                    and mouse_point.y >= screen_y + screen_h - corner_size
                    and mouse_point.y <= screen_y + screen_h + corner_size
                ):
                    return tile, 'BR'
        return None, ''

    def add_current_tiles_to_history(self) -> None:
        self.tile_history.add_state(self.tiles)

    def get_model_size(self) -> tuple[int, int]:
        return self.model_width, self.model_height

    def get_custom_tile_size(self) -> tuple[int, int]:
        try:
            w = int(self.custom_tile_width_edit.GetValue())
            h = int(self.custom_tile_height_edit.GetValue())
            return w, h
        except ValueError:
            return -1, -1

    def get_selected_tile_size(self) -> tuple[int, int]:
        selected_indices = self.get_selections()
        if len(selected_indices) == 1:
            index = selected_indices[0]
            tile_str = self.tile_text_listbox.GetString(index)
            tile, error = _convert_string_to_tile(tile_str)
            if error == '':
                return tile[2], tile[3]
        return -1, -1

    def _rtsp_url_changed(self, event: wx.CommandEvent) -> None:
        url = self.rtsp_url_edit.GetValue().strip()
        self.start_rtsp_button.Enable(bool(url))


class help_dialog(wx.Dialog):
    def __init__(self, parent: wx.Window, size: wx.Size, version_str: str) -> None:
        super(help_dialog, self).__init__(
            parent=parent,
            title=f"Tile_Creator Help (version {version_str})",
            style=wx.RESIZE_BORDER | wx.DEFAULT_DIALOG_STYLE,
            size=size,
        )
        self._text = (
            "Video\n"
            "-----\n"
            "Select your video file and press Load Video to load it and it will start playing \n"
            "automatically. Videos will constantly loop.\n"
            "Or use the Scan Cameras button to find connected cameras. Select a camera, resolution"
            ", format\n"
            "and framerate and press Start Camera it will start streaming.\n"
            "the camera list.\n"
            "Pause/Play button is self explanatory\n"
            "Resize the app to make the video area larger or smaller.\n\n"
            "Set Model Size\n"
            "--------------\n"
            "Set the model width and height to the size your model expects.\n"
            "If you load a JSON file that contains the model size in the filename `model640x640`\n"
            "e.g. 8k_iscw_v5-4230x4320-model640x640-11-tiles.json, then the model size will be"
            " set\n"
            "automatically.\n"
            "Tiles that are the same size as the model will be shown in green.\n"
            "Tiles that are larger than the model will be shown in black.\n"
            "Tiles that are smaller then the model will be shown in red.\n\n"
            "Create Tile\n"
            "-----------\n"
            "To create a tile...\n"
            "    * Hold SHIFT and use the mouse to left click and drag to hand draw the tile.\n"
            "    * Use the mouse right click to create a tile at the mouse point.\n"
            "      Size will be the model size.\n"
            "    * Use SHIFT + the mouse right click to create a tile at the mouse point.\n"
            "      Size will be the custom size.\n"
            "    * Use CTRL/CMD + the mouse right click to create a tile at the mouse point.\n"
            "      Size will be the currently selected tile size.\n"
            "      Note: This only works with 1 tile selected.\n"
            "    * Enter it manually using the Edit button of the tiles list.\n"
            "To correct the size of all tiles to be at least the model size, press the Correct Too"
            " Small\n"
            "button\n\n"
            "Selecting Tiles\n"
            "---------------\n"
            "Click on tiles to select them.\n"
            "If there are multiple tiles on the same point, then all the tiles that share that "
            "point\n"
            "will be selected.\n"
            "To select/unselect multiple tiles, hold down the CMD (Mac) or CTRL (Windows/Linux) "
            "key\n"
            "whilst clicking.\n"
            "To select tiles from the tile list, use the mouse to click on them.\n"
            "To select multiple tiles in the list, hold down the CMD (Mac) or CTRL (Windows/"
            "Linux)\n"
            "whilst clicking.\n"
            "Tiles in the tile list are in the format x, y, width, height.\n\n"
            "Move Tiles\n"
            "----------\n"
            "Click and drag the mouse to move the selected tiles.\n"
            "To move selected tiles more finely, use the Up Left Right Down buttons.\n\n"
            "Resize Tiles\n"
            "------------\n"
            "To resize a specific tile, drag the corner of the tile.\n"
            "To resize the selected tiles with the mouse, use the mouse wheel.\n"
            "Note: you can resize multipe selected tiles\n"
            "To resize selected tiles more finely, manually edit the tiles list.\n\n"
            "Editing Tiles\n"
            "-------------\n"
            "To manually edit all the tiles, as if it was JSON listed in an editor, select the "
            "Edit\n"
            "Tiles button.\n"
            "The text will change to orange to depict you are in edit mode.\n"
            "Make your changes and then press Apply Tiles to apply them and changes will be\n"
            "reflected immediately in the video area.\n"
            "Any errors in the tile definitions will be shown in red below the text area.\n"
            "If you wish to discard your changes, press Cancel Edit or press ESC.\n\n"
            "Undo/Redo\n"
            "---------\n"
            "You can undo and redo Any changes to the tiles using the Undo and Redo buttons.\n"
            "Also you can user CTRL+Z (CMD+Z on Mac) to undo and CTRL+Y (CMD+Y on Mac) to "
            "redo.\n\n"
            "Loading and Saving Tiles\n"
            "------------------------\n"
            "To load tiles from a JSON file, press the Load JSON button and select the file to "
            "load.\n"
            "If the JSON filename contains the model size e.g. tiles_640x480.json, then the "
            "model\n"
            "size will be set automatically.\n"
            "To save the current tiles to a JSON file, press the Save JSON button and select "
            "a file\n"
            "name.\n"
        )
        self.SetMinSize(size)
        self.create_ctrls()
        self.fill_help_listbox()
        self.layout()

    def create_ctrls(self) -> None:
        self.help_listbox = wx.ListBox(self, style=wx.LB_HSCROLL)
        self.fw_font = wx.Font(
            10,
            wx.FONTFAMILY_DEFAULT,
            wx.FONTSTYLE_NORMAL,
            wx.FONTWEIGHT_NORMAL,
            faceName='Courier New',
        )
        self.help_listbox.SetFont(self.fw_font)
        self.ok_button = wx.Button(self, wx.ID_OK, label='OK')
        self.ok_button.SetDefault()

    def layout(self) -> None:
        vsizer = wx.BoxSizer(wx.VERTICAL)
        vsizer.AddSpacer(10)

        hsizer = wx.BoxSizer(wx.HORIZONTAL)
        hsizer.AddSpacer(10)
        hsizer.Add(self.help_listbox, 1, wx.EXPAND)
        hsizer.AddSpacer(10)

        h1sizer = wx.BoxSizer(wx.HORIZONTAL)
        h1sizer.AddStretchSpacer()
        h1sizer.Add(self.ok_button, 0)
        h1sizer.AddSpacer(10)

        vsizer.Add(hsizer, 1, wx.EXPAND)
        vsizer.AddSpacer(10)
        vsizer.Add(h1sizer, 0, wx.EXPAND)
        vsizer.AddSpacer(10)

        self.SetSizer(vsizer)
        self.Layout()

    def fill_help_listbox(self) -> None:
        for line in self._text.splitlines():
            self.help_listbox.Append(line)


class TileGenFrame(wx.Frame):
    def __init__(
        self,
        initial_frame_size: wx.Size,
    ) -> None:
        super().__init__(parent=None, size=initial_frame_size)
        self.SetMinSize(initial_frame_size)
        self.set_title()
        self.panel = AxTileGenPanel(self, initial_frame_size, version)
        self.Show()
        self.Bind(wx.EVT_CLOSE, self.on_close)

    def set_title(self, loaded_json: str = '') -> None:
        title = f"Tile Config (v{version})"
        if loaded_json != '':
            title += f" - Currently loaded : {loaded_json}"
        super().SetTitle(title)

    def on_close(self, event: wx.Event) -> None:
        self.panel.on_close(event)
        self.Destroy()


def main() -> None:
    app = wx.App()
    TileGenFrame(initial_frame_size)
    app.MainLoop()


if __name__ == '__main__':
    main()

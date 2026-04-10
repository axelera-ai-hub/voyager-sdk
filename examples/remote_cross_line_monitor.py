#!/usr/bin/env python
# Copyright Axelera AI, 2025
# Vehicle line-cross counting example with remote streaming support
#
# Usage:
#   python examples/remote_cross_line_monitor.py
#
# Connect from another terminal to observe the JSON feed:
#   nc localhost 8765


from __future__ import annotations

from dataclasses import dataclass
import json
import socket
import socketserver
import threading
from typing import Iterable

import cv2

from axelera import types
from axelera.app import config, logging_utils
from axelera.app.display import App
from axelera.app.stream import create_inference_stream

LOG = logging_utils.getLogger(__name__)


class BroadcastTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, handler_cls):
        super().__init__(server_address, handler_cls)
        self._clients: set[socket.socket] = set()
        self._lock = threading.Lock()

    # -- client management -------------------------------------------------

    def register_client(self, sock: socket.socket) -> None:
        with self._lock:
            self._clients.add(sock)
        LOG.info("Remote client connected (%s:%s)", *sock.getpeername())

    def unregister_client(self, sock: socket.socket) -> None:
        with self._lock:
            self._clients.discard(sock)
        LOG.info("Remote client disconnected")

    def broadcast(self, message: dict) -> None:
        payload = (json.dumps(message) + "\n").encode("utf-8")
        dead: list[socket.socket] = []
        with self._lock:
            for client in self._clients:
                try:
                    client.sendall(payload)
                except OSError:
                    dead.append(client)
            for client in dead:
                self._clients.discard(client)


class BroadcastHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:  # pragma: no cover - network shim
        self.server.register_client(self.request)
        try:
            while self.request.recv(1024):
                pass
        except OSError:
            pass
        finally:
            self.server.unregister_client(self.request)


@dataclass
class RemoteBroadcaster:
    host: str = "0.0.0.0"
    port: int = 8765

    def __post_init__(self) -> None:
        self._server = BroadcastTCPServer((self.host, self.port), BroadcastHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()
        LOG.info("Broadcast server listening on %s:%s", self.host, self.port)

    def broadcast(self, message: dict) -> None:
        self._server.broadcast(message)

    def shutdown(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        LOG.info("Broadcast server stopped")


network = "yolov5m-v7-coco-tracker"
source = config.env.framework / "media/traffic1_480p.mp4"
vehicles = {2, 5, 7}  # car, bus, truck

stream = create_inference_stream(
    network=network,
    sources=[source],
    pipe_type='gst',
    log_level=logging_utils.INFO,
)


def objects_for(vehicle_tracks: Iterable) -> list[dict]:
    serialized = []
    for veh in vehicle_tracks:
        bbox_history = veh.history
        bbox = bbox_history[-1].tolist() if bbox_history.size else []
        serialized.append(
            {
                "track_id": veh.track_id,
                "class_id": veh.class_id,
                "bbox": bbox,
            }
        )
    return serialized


def main(window, stream, broadcaster: RemoteBroadcaster):
    mid_line_start = None
    mid_line_end = None
    mid_line_slope = None
    mid_line_intercept = None

    def is_below_line(point):
        if mid_line_slope == float("inf"):
            return point[0] > mid_line_start[0]
        return point[1] > (mid_line_slope * point[0] + mid_line_intercept)

    n_frames_to_track = 90
    crossed_up = crossed_down = 0
    recent_up: list[tuple[int, int]] = []
    recent_down: list[tuple[int, int]] = []
    counted: set[int] = set()

    up_label = window.text('10px, 50px', 'Vehicles Up: 0', color=(255, 165, 0, 255), stream_id=0)
    down_label = window.text(
        '10px, 100px', 'Vehicles Down: 0', color=(255, 165, 0, 255), stream_id=0
    )

    frame_count = 0
    for frame_result in stream:
        frame_count += 1
        image = frame_result.image.asarray(types.ColorFormat.BGR).copy()

        if mid_line_start is None:
            height, width, _ = image.shape
            mid_line_start = (0, (3 * height) // 4)
            mid_line_end = (width, (3 * height) // 4)
            if mid_line_end[0] != mid_line_start[0]:
                mid_line_slope = (mid_line_end[1] - mid_line_start[1]) / (
                    mid_line_end[0] - mid_line_start[0]
                )
                mid_line_intercept = mid_line_start[1] - (mid_line_slope * mid_line_start[0])
            else:
                mid_line_slope = float("inf")
                mid_line_intercept = mid_line_start[0]

        cv2.line(image, mid_line_start, mid_line_end, (0, 255, 0), 2)

        detections = [
            veh for veh in frame_result.pedestrian_and_vehicle_tracker if veh.class_id in vehicles
        ]
        for veh in detections:
            if veh.track_id in counted or len(veh.history) <= 1:
                continue

            last_bbox = veh.history[-1]
            prev_bbox = veh.history[0]
            last_center = ((last_bbox[0] + last_bbox[2]) / 2, (last_bbox[1] + last_bbox[3]) / 2)
            prev_center = ((prev_bbox[0] + prev_bbox[2]) / 2, (prev_bbox[1] + prev_bbox[3]) / 2)

            if is_below_line(prev_center) and not is_below_line(last_center):
                crossed_down += 1
                counted.add(veh.track_id)
                recent_down.append((veh.track_id, frame_count))
            elif not is_below_line(prev_center) and is_below_line(last_center):
                crossed_up += 1
                counted.add(veh.track_id)
                recent_up.append((veh.track_id, frame_count))

        recent_up[:] = [item for item in recent_up if frame_count - item[1] <= n_frames_to_track]
        recent_down[:] = [
            item for item in recent_down if frame_count - item[1] <= n_frames_to_track
        ]

        up_label["text"] = "Vehicles Up: {} ({})".format(
            crossed_up,
            ", ".join(str(track_id) for track_id, _ in recent_up),
        )
        down_label["text"] = "Vehicles Down: {} ({})".format(
            crossed_down,
            ", ".join(str(track_id) for track_id, _ in recent_down),
        )

        broadcaster.broadcast(
            {
                "frame": frame_count,
                "crossed_up": crossed_up,
                "crossed_down": crossed_down,
                "recent_up": [track_id for track_id, _ in recent_up],
                "recent_down": [track_id for track_id, _ in recent_down],
                "objects": objects_for(detections),
            }
        )

        window.show(
            types.Image.fromarray(image, types.ColorFormat.BGR),
            frame_result.meta,
            frame_result.stream_id,
        )

        if window.is_closed:
            break


broadcaster = RemoteBroadcaster()


with App(renderer=True, opengl=stream.hardware_caps.opengl) as app:
    wnd = app.create_window("Remote cross line monitor", (900, 600))
    try:
        app.start_thread(main, (wnd, stream, broadcaster), name='InferenceThread')
        app.run()
    finally:
        broadcaster.shutdown()

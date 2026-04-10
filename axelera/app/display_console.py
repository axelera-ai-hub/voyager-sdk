# Copyright Axelera AI, 2025
from __future__ import annotations

import base64
import collections
import os
import queue
import time
from typing import TYPE_CHECKING

import cv2
import numpy as np
import tqdm

from axelera import types

from . import display, display_cv, logging_utils, utils

if TYPE_CHECKING:
    import uuid

    from . import inf_tracers
    from .meta import AxMeta

LOG = logging_utils.getLogger(__name__)


def _count_as_bar(value, width=20, max_value=100):
    full = '\u2588'
    blocks = ' \u258f\u258e\u258d\u258c\u258b\u258a\u2589'
    if max_value == 0:
        return full * width
    whole = int(width * value / max_value)
    remainder = int(((width * value / max_value) - whole) * 8)
    rem = blocks[remainder] if remainder else ''
    return (full * whole + rem).ljust(width)


_RESET = "\x1b[0m"
_LTGREY = "\x1b[37;2m"


class ConsoleDraw(display_cv.CVDraw):
    def __init__(
        self,
        stream_id: int,
        num_streams: int,
        composite: types.Image,
        image: types.Image,
        layers: list,
        metrics,
        labels,
        options: ConsoleOptions,
        window_options=None,
        speedometer_smoothing: display.SpeedometerSmoothing = None,
    ):
        super().__init__(
            stream_id=stream_id,
            num_streams=num_streams,
            composite=composite,
            image=image,
            layers=layers,
            options=options,
            window_options=window_options,
            speedometer_smoothing=speedometer_smoothing,
        )
        self.metrics = metrics
        self.labels = labels
        self._options = options

    @property
    def options(self) -> ConsoleOptions:
        return self._options

    def text(
        self,
        p,
        text: str,
        txt_color: display.Color,
        back_color: display.OptionalColor = None,
        font: display.Font = display.Font(),
    ):
        x = int(self._canvas.left + p[0] * self._canvas.scale)
        y = int(self._canvas.top / 2 + p[1] * self._canvas.scale / 2) + 2
        self.labels.append((x, y, text, txt_color))

    def textsize(self, s, font):
        return (len(s), 1)

    def draw_speedometer(self, metric: inf_tracers.TraceMetric):
        if self._speedometer_smoothing:
            self._speedometer_smoothing.update(metric)
        text = display.calculate_speedometer_text(metric, self._speedometer_smoothing)
        value = (
            metric.value
            if not self._speedometer_smoothing
            else self._speedometer_smoothing.value(metric)
        )
        bar = _count_as_bar(value, 10, metric.max_scale_value)
        self.metrics.append(f"{metric.title} [{bar}{_LTGREY}{text}{_RESET}]")

    def draw_statistics(self, stats):
        cells = []
        cells.append(f'min:{int(stats.min):<5d}')
        cells.append(f'mean:{int(stats.mean):<5d}')
        cells.append(f'max:{int(stats.max):<5d}')
        cells.append(f'stddev:{int(stats.stddev):<5d}')
        self.metrics.append(f"{stats.title:<9s} [{_LTGREY}{' '.join(cells)}{_RESET}]")


def _reset():
    print('\033[m', end='')


def _moveto(x, y):
    print(f'\033[{y};{x}H', end='')


def _out(s):
    print(s, end='')


def _flush():
    print(end='', flush=True)


class ConsoleWindow(display.Window):
    def wait_for_close(self):
        self._queue.put(display.THREAD_COMPLETED)
        while not self.is_closed or not self._queue.empty():
            time.sleep(0.1)


class ConsoleOptions(display.Options):
    pass  # No additional options for console... yet


def _read_new_data(queues):
    frames = {}
    others = {}
    for q in queues:
        try:
            while True:
                msg = q.get(block=False)
                if msg is display.SHUTDOWN or msg is display.THREAD_COMPLETED:
                    return msg
                if isinstance(msg, display._Frame):
                    frames[msg.stream_id] = msg  # keep only the latest frame data
                else:
                    others.setdefault(msg.stream_id, []).append(
                        msg
                    )  # don't throw away other messages
        except queue.Empty:
            pass
    msgs = []
    for source_id, source_msgs in others.items():
        msgs.extend([(source_id, m) for m in source_msgs])
    msgs.extend(frames.items())
    return msgs


class ConsoleApp(display.App):
    Window = ConsoleWindow
    SupportedOptions = ConsoleOptions

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cols, self._rows = 0, 0
        self._current: dict[int, tuple[types.Image, AxMeta | None]] = {}
        self._meta_cache = display.MetaCache()
        self._speedometer_smoothing = display.SpeedometerSmoothing()
        self._layers: dict[uuid.UUID, display._Layer] = collections.defaultdict()
        self._current: dict[int, tuple[types.Image, AxMeta | None]] = {}
        self._options: dict[int, ConsoleOptions] = collections.defaultdict(ConsoleOptions)
        self._buffer = None
        self._pending_updates = set()
        self._closed_sources = set()
        self._frame_sink = None
        self._pane_size = None
        self._title = ''

    @property
    def num_sources(self) -> int:
        return len(self._current)

    def _create_new_window(self, q, frame_sink, title, size):
        del q  # unused
        self._frame_sink = frame_sink
        self._pane_size = size
        self._title = title
        return title

    def _handle_message(self, source_id, msg):
        if isinstance(msg, display._OpenSource):
            self._closed_sources.discard(msg.stream_id)
            return
        blocking = isinstance(msg, display._BlockingFrame)
        if getattr(msg, 'stream_id', None) in self._closed_sources:
            if blocking:
                LOG.error(f"Received blocking frame from closed source {msg.stream_id}")
            return  # ignore messages from closed sources
        if isinstance(msg, display._CloseSource):
            if not msg.reopen:
                self._closed_sources.add(msg.stream_id)
            self._current.pop(msg.stream_id, None)
            self._pending_updates.discard(msg.stream_id)
            return
        if isinstance(msg, display._SetOptions):
            self._options[source_id].update(**msg.options)
            return
        if isinstance(msg, display._Layer):
            self._layers[msg.id] = msg
            return
        if not isinstance(msg, display._Frame):
            LOG.debug(f"Unknown message: {msg}")
            return
        self._current[source_id] = (msg.image, msg.meta, blocking)
        self._pending_updates.add(source_id)

    def _output(self):
        if self._title:
            self._show()
        if self._frame_sink:
            self._frame_sink.push(*self._render_to_image())

    def _run(self, interval=1 / 10):
        last_frame = time.time()
        while 1:
            self._create_new_windows()
            new = _read_new_data(self._queues)
            if new is display.SHUTDOWN or new is display.THREAD_COMPLETED:
                return  # never linger on shutdown or thread completed

            for source_id, msg in new:
                self._handle_message(source_id, msg)
            if new:
                self._output()

            now = time.time()
            remaining = max(1 / 1000, interval - (now - last_frame))
            last_frame = now
            time.sleep(remaining)

            if self.has_thread_completed:
                return

    def _render_to_image(self):
        w, h = self._pane_size
        composite = types.Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
        pending_blocking_capture = False
        # Always render in descending stream ID order - window layers are rendered by stream 0.
        # If stream 0 is not rendered last, other streams may render on top of the window layers
        for source_id, (img, meta_data, blocking) in sorted(
            self._current.items(), key=lambda x: x[0], reverse=True
        ):
            draw = display_cv.CVDraw(
                source_id,
                len(self._current),
                composite,
                img,
                display.get_layers(self._layers, source_id),
                self._options[source_id],
                self._options[-1],
                self._speedometer_smoothing,
            )
            _, meta_map = self._meta_cache.get(source_id, meta_data)
            for m in meta_map.values():
                m.visit(lambda m: m.draw(draw))
            draw.draw()
            if blocking:
                pending_blocking_capture = True
        return composite, pending_blocking_capture

    def _show(self):
        SPACE_FOR_SPEEDOMETER = 7
        with tqdm.tqdm.external_write_mode():
            console = os.get_terminal_size()
            cols, rows = console.columns - 1, ((console.lines - SPACE_FOR_SPEEDOMETER) // 2) * 2
            if self._cols == 0 or self._cols != cols or self._rows != rows:
                self._cols, self._rows = cols, rows
                _out('\n' * rows)  # make space

            labels, metrics = [], []
            x = np.zeros((self._rows * 2, self._cols, 3), dtype=np.uint8)
            composite = types.Image.fromarray(x)
            for source_id, (img, meta, _) in sorted(
                self._current.items(), key=lambda x: x[0], reverse=True
            ):
                draw = ConsoleDraw(
                    stream_id=source_id,
                    num_streams=self.num_sources,
                    composite=composite,
                    image=img,
                    layers=display.get_layers(self._layers, source_id),
                    metrics=metrics,
                    labels=labels,
                    options=self._options[source_id],
                    window_options=self._options[-1],
                    speedometer_smoothing=self._speedometer_smoothing,
                )
                _, meta_map = self._meta_cache.get(source_id, meta)
                for m in meta_map.values():
                    m.visit(lambda m: m.draw(draw))
                draw.draw()

            x = composite.asarray('RGB')

            import climage

            _moveto(0, 0)
            _out(climage.convert_array(x, is_unicode=True, is_256color=False, is_truecolor=True))
            short_metrics = [m for m in metrics if len(m) < 40]
            long_metrics = [m for m in metrics if len(m) >= 40]
            _out("        ".join(short_metrics))
            _out("".join(f'\n{x}' for x in long_metrics))

            for x, y, label, (r, g, b, _) in sorted(labels):
                _moveto(x, y)
                _out(f'\x1b[38;2;{r};{g};{b}m{label}')

            _moveto(0, 0)
            _flush()
            _reset()

    def _get_buffer(self, *shape):
        if self._buffer is None or self._buffer.shape != shape:
            self._buffer = np.zeros(shape, dtype=np.uint8)
            self._pending_updates.update(self._current.keys())
        return self._buffer

    def _destroy_all_windows(self):
        _reset()

    def __del__(self):
        try:
            _reset()
        except Exception:
            # In case of an error during reset, we just ignore it.
            # This can happen if the console is closed before the app is destroyed.
            pass


def _as_iterm2_image(
    img,  # cv::Mat
    size=(640, 480),  # $1 (size of the image)
    quality=80,
):
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
    result, encimg = cv2.imencode('.jpg', img, encode_param)
    data = base64.b64encode(encimg).decode('ascii')
    return (
        f"\033]1337;File=inline=1;preserveAspectRatio=1;type=image/jpeg"
        f";size={len(encimg)};width={size[0]}px;height={size[1]}px:{data}\a\n"
    )


class iTerm2App(ConsoleApp):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._desired_size = display.FULL_SCREEN

    def _create_new_window(self, q, frame_sink, title, size):
        del q  # unused
        self._frame_sink = frame_sink
        self._title = title
        self._desired_size = size
        return title

    def _output(self):
        self._show()

    def _show(self):
        with tqdm.tqdm.external_write_mode():
            console = utils.get_terminal_size_ex()
            if self._desired_size == display.FULL_SCREEN:
                cols, rows = console.width, console.height
            else:
                cols, rows = self._desired_size
            if self._cols == 0 or self._cols != cols or self._rows != rows:
                self._cols, self._rows = cols, rows
                if self._title:
                    cell_height = console.height // console.lines
                    _out('\n' * (rows // cell_height))  # make space

            x = self._get_buffer(self._rows, self._cols, 3)
            composite = types.Image.fromarray(x)
            pending_blocking_capture = False
            for source_id, (img, meta, blocking) in sorted(
                self._current.items(), key=lambda x: x[0], reverse=True
            ):
                if source_id not in self._pending_updates:
                    continue
                self._pending_updates.remove(source_id)
                draw = display_cv.CVDraw(
                    source_id,
                    self.num_sources,
                    composite,
                    img,
                    display.get_layers(self._layers, source_id),
                    options=self._options[source_id],
                    window_options=self._options[-1],
                    speedometer_smoothing=self._speedometer_smoothing,
                )
                _, meta_map = self._meta_cache.get(source_id, meta)
                for m in meta_map.values():
                    m.visit(lambda m: m.draw(draw))
                draw.draw()
                if blocking:
                    pending_blocking_capture = True

            if self._frame_sink:
                self._frame_sink.push(composite, block=pending_blocking_capture)
            if self._title:
                x = composite.asarray('BGR')

                _moveto(0, 0)
                _out(_as_iterm2_image(x, (self._cols, self._rows)))
                _moveto(0, 0)
                _flush()
                _reset()

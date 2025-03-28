# Copyright Axelera AI, 2023
from __future__ import annotations

import collections
import math
import os
import time
from typing import TYPE_CHECKING, Tuple

import cv2
import numpy as np
import tqdm

from . import display, display_cv, logging_utils

if TYPE_CHECKING:
    from . import inf_tracers

LOG = logging_utils.getLogger(__name__)


def _count_as_bar(value, width=20, max_value=100):
    full = '\u2588'
    blocks = ' \u258F\u258E\u258D\u258C\u258B\u258A\u2589'
    if max_value == 0:
        return full * width
    whole = int(width * value / max_value)
    remainder = int(((width * value / max_value) - whole) * 8)
    rem = blocks[remainder] if remainder else ''
    return (full * whole + rem).ljust(width)


_RESET = "\x1b[0m"
_LTGREY = "\x1b[37;2m"


class ConsoleDraw(display_cv.CVDraw):
    def __init__(self, img, pane_pos, metrics, labels, options: ConsoleOptions, **kwargs):
        super().__init__(img)
        self.metrics = metrics
        self.labels = labels
        self.pane_pos = pane_pos
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
        w, h = self.canvas_size
        x = int(self.pane_pos[0] + p[0] / w * self.pane_pos[2])
        y = int(self.pane_pos[1] / 2 + p[1] / h * self.pane_pos[3] / 2) + 2
        self.labels.append((x, y, text, txt_color))

    def textsize(self, s, font):
        return (len(s), 1)

    def draw_speedometer(self, metric: inf_tracers.TraceMetric):
        text = display.calculate_speedometer_text(metric)
        bar = _count_as_bar(metric.value, 10, metric.max)
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


def image_support_available():
    try:
        import climage  # noqa

        return True
    except ImportError:
        return False


def _get_layout(stream_id: int, num_streams: int) -> Tuple[float, float, float, float]:
    layouts = {
        3: (0.5, 0.5, [(0.0, 0.0), (0.5, 0.25), (0.0, 0.5)]),
        5: (0.4, 0.4, [(0.0, 0.0), (0.6, 0.0), (0.3, 0.3), (0.0, 0.6), (0.6, 0.6)]),
    }
    try:
        w, h, positions = layouts[num_streams]
        return positions[stream_id] + (w, h)
    except KeyError:
        cols = math.ceil(math.sqrt(num_streams))
        rows = math.ceil(num_streams / cols)
        x, y = (stream_id % cols) / cols, (stream_id // cols) / rows
        return x, y, 1 / cols, 1 / rows


def _fit_within_rect(image: Tuple[int, int], bounding: Tuple[int, int]):
    imgr = image[0] / image[1]
    wndr = bounding[0] / bounding[1]
    if imgr > wndr:
        new_width = bounding[0]
        new_height = int(new_width / imgr)
    else:
        new_height = bounding[1]
        new_width = int(new_height * imgr)
    return new_width, new_height


def _pane_position(
    stream_id: int, num_streams: int, image: Tuple[int, int], window: Tuple[int, int]
) -> Tuple[int, int, int, int]:
    x, y, w, h = _get_layout(stream_id, num_streams)
    x *= window[0]
    y *= window[1]
    bounding_box = w * window[0], h * window[1]
    w, h = _fit_within_rect(image, bounding_box)
    x += (bounding_box[0] - w) / 2
    y += (bounding_box[1] - h) / 2
    return int(x), int(y), int(w), int(h)


class ConsoleWindow(display.Window):
    def wait_for_close(self):
        self._queue.put(display.THREAD_COMPLETED)
        while not self.is_closed or not self._queue.empty():
            time.sleep(0.1)


class ConsoleOptions(display.Options):
    pass  # No additional options for console... yet


class ConsoleApp(display.App):
    Window = ConsoleWindow
    SupportedOptions = ConsoleOptions

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cols, self._rows = 0, 0
        self._current = {}
        self._meta_cache = display.MetaCache()
        self._options: dict[int, ConsoleOptions] = collections.defaultdict(ConsoleOptions)

    def _create_new_window(self, q, title, size):
        del q  # unused
        return title

    def _run(self, interval=1 / 10):
        last_frame = time.time()
        while 1:
            self._create_new_windows()
            new = display_cv._read_new_data(self._wnds, self._queues)
            if new is display.SHUTDOWN or new is display.THREAD_COMPLETED:
                return  # never linger on shutdown or thread completed

            for wndname, msg in new:
                if isinstance(msg, display._SetOptions):
                    self._options[msg.stream_id].update(**msg.options)
                elif isinstance(msg, display._Frame):
                    if wndname not in self._wnds:
                        self._create_new_window(None, wndname, msg.image.size)
                    self._current[wndname] = (msg.image, msg.meta)
            if new:
                self._show()

            now = time.time()
            remaining = max(1, interval - (now - last_frame))
            last_frame = now
            time.sleep(remaining)

            if self.has_thread_completed:
                return

    def _show(self):
        SPACE_FOR_SPEEDOMETER = 7
        with tqdm.tqdm.external_write_mode():
            console = os.get_terminal_size()
            cols, rows = console.columns - 1, ((console.lines - SPACE_FOR_SPEEDOMETER) // 2) * 2
            if self._cols == 0 or self._cols != cols or self._rows != rows:
                self._cols, self._rows = cols, rows
                _out('\n' * rows)  # make space

            labels, metrics = [], []
            x = np.zeros((self._rows * 2, self._cols, 4), dtype=np.uint8)
            for stream_id, (img, meta) in enumerate(self._current.values()):
                opts = self._options[stream_id]
                w, h = img.size
                pos = _pane_position(
                    stream_id, len(self._current), (w, h), (self._cols, self._rows * 2)
                )
                draw = ConsoleDraw(img, pos, metrics=metrics, labels=labels, options=opts)
                _, meta_map = self._meta_cache.get(stream_id, meta)
                for m in meta_map.values():
                    m.draw(draw)
                draw.draw()

                img = img.asarray('RGB')
                img = cv2.resize(img, (pos[2], pos[3]), interpolation=cv2.INTER_AREA)
                img = display_cv.rgb_to_grayscale_rgb(img, opts.grayscale)

                if img.shape[2] == 4:
                    x[pos[1] : pos[1] + img.shape[0], pos[0] : pos[0] + img.shape[1], :] = img
                else:  # 3 channels
                    x[pos[1] : pos[1] + img.shape[0], pos[0] : pos[0] + img.shape[1], :3] = img
                if title := self._options[stream_id].title:
                    draw.text(pos, title, (255, 255, 255, 255))

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

    def _destroy_all_windows(self):
        _reset()

    def __del__(self):
        _reset()

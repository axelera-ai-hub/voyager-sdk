# Copyright Axelera AI, 2025
from __future__ import annotations

import collections
import dataclasses
import functools
import os
import queue
import time
from typing import TYPE_CHECKING, Sequence, Tuple

import PIL
import PIL.ImageDraw
import PIL.ImageFont
import cv2
import numpy as np

from axelera import types

from . import display, logging_utils, meta

if TYPE_CHECKING:
    from . import inf_tracers

LOG = logging_utils.getLogger(__name__)

SegmentationMask = tuple[int, int, int, int, int, int, int, int, np.ndarray]


def _read_new_data(wnds, queues):
    frames = {}
    others = {}
    for wnd, q in zip(wnds, queues):
        try:
            while True:
                msg = q.get(block=False)
                if msg is display.SHUTDOWN or msg is display.THREAD_COMPLETED:
                    return msg
                wndname = f"{wnd} {msg.stream_id}" if msg.stream_id > 0 else wnd
                if isinstance(msg, display._Frame):
                    frames[wndname] = msg  # keep only the latest frame data
                else:
                    others.setdefault(wndname, []).append(msg)  # don't throw away other messages
        except queue.Empty:
            pass
    msgs = []
    for wndname, wndmsgs in others.items():
        msgs.extend([(wndname, m) for m in wndmsgs])
    msgs.extend(frames.items())
    return msgs


def _make_splash(window_size):
    for ico_sz, path in reversed(display.ICONS.items()):
        if ico_sz < min(window_size):
            ico = cv2.imread(path)
            break
    top = int((window_size[1] - ico.shape[0]) / 2)
    bottom = window_size[1] - top - ico.shape[0]
    left = int((window_size[0] - ico.shape[1]) / 2)
    right = window_size[0] - left - ico.shape[1]
    return cv2.copyMakeBorder(ico, top, bottom, left, right, cv2.BORDER_CONSTANT, (0, 0, 0))


def rgb_to_grayscale_rgb(img: np.ndarray, grayness: float) -> np.ndarray:
    if grayness > 0.0:
        grey = np.dot(img[..., :3], [0.2989, 0.5870, 0.1140])
        orig, img = img, np.stack((grey, grey, grey), axis=-1)
        if grayness < 1.0:
            img *= grayness
            img += orig * (1.0 - grayness)
        img = img.astype('uint8')
    return img


@dataclasses.dataclass
class CVOptions(display.Options):
    pass  # No additional options for opencv


class CVApp(display.App):
    SupportedOptions = CVOptions

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._stream_wnds = set()
        self._meta_cache = display.MetaCache()

    def _create_new_window(self, q, wndname, size):
        del q  # unused
        if size == display.FULL_SCREEN:
            cv2.namedWindow(wndname, cv2.WINDOW_FULLSCREEN)
        else:
            cv2.namedWindow(wndname, cv2.WINDOW_NORMAL)
            cv2.imshow(wndname, _make_splash(size))
        cv2.setWindowProperty(wndname, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
        self._stream_wnds.add(wndname)
        return wndname

    def _destroy_all_windows(self):
        cv2.destroyAllWindows()

    def _run(self, interval=1 / 30):
        last_frame = time.time()
        options: dict[str, CVOptions] = collections.defaultdict(CVOptions)
        pending_titles: dict[str, CVOptions] = {}
        while 1:
            self._create_new_windows()
            new_msgs = _read_new_data(self._wnds, self._queues)
            if new_msgs is display.SHUTDOWN:
                return
            if new_msgs is display.THREAD_COMPLETED:
                continue  # ignore, just wait for user to close

            for wndname, msg in new_msgs:
                opts = options[wndname]
                if isinstance(msg, display._SetOptions):
                    oldtitle = opts.title
                    opts.update(**msg.options)
                    if oldtitle != opts.title:
                        pending_titles[wndname] = options[wndname].title
                    continue
                elif not isinstance(msg, display._Frame):
                    LOG.debug(f"Unknown message: {msg}")
                    continue

                if wndname not in self._wnds:
                    self._create_new_window(None, wndname, msg.image.size)
                draw = CVDraw(msg.image, opts)
                _, meta_map = self._meta_cache.get(msg.stream_id, msg.meta)
                for m in meta_map.values():
                    m.visit(lambda m: m.draw(draw))
                draw.draw()
                bgr = msg.image.asarray(types.ColorFormat.BGRA)
                if pending := pending_titles.pop(wndname, None):
                    cv2.setWindowTitle(wndname, pending)
                cv2.imshow(wndname, bgr)

            if any(cv2.getWindowProperty(t, cv2.WND_PROP_VISIBLE) <= 0.0 for t in self._wnds):
                return

            now = time.time()
            remaining = max(1, interval - (now - last_frame))
            last_frame = now
            if cv2.waitKey(remaining) in (ord("q"), ord("Q"), 27, 32):
                return
            if self.has_thread_completed:
                return


_FONT_FAMILIES = {
    display.FontFamily.sans_serif: "sans-serif",
    display.FontFamily.serif: "serif",
    # 'cursive', 'fantasy', or 'monospace'
}


def _coords(centre, length):
    return centre[0] - length, centre[1] - length, centre[0] + length, centre[1] + length


@functools.lru_cache
def _get_speedometer(diameter):
    here = os.path.dirname(__file__)
    x = PIL.Image.open(f'{here}/speedo-alpha-transparent.png')
    return x.resize((diameter, diameter))


class _DrawList(list):
    def __getattr__(self, name):
        def _draw(*args):
            self.append((name,) + args)

        return _draw


class CVDraw(display.Draw):
    def __init__(self, image: types.Image, options: CVOptions = CVOptions()):
        self._canvas_size = (image.width, image.height)
        self._img = image
        rgb = image.asarray('RGB')
        self._pil = PIL.Image.fromarray(rgb_to_grayscale_rgb(rgb, options.grayscale))
        self._draw = PIL.ImageDraw.Draw(self._pil)
        self._dlist = _DrawList()
        self._font_cache = {}
        self._speedometer_index = 0
        self._options = options

    @property
    def options(self) -> CVOptions:
        return self._options

    @property
    def canvas_size(self) -> display.Point:
        return self._canvas_size

    @property
    def _speedometer_dlist_calls(self):
        return 4 * self._speedometer_index

    def draw_speedometer(self, metric: inf_tracers.TraceMetric):
        text = display.calculate_speedometer_text(metric)
        needle_pos = display.calculate_speedometer_needle_pos(metric)
        m = display.SpeedometerMetrics(self._canvas_size, self._speedometer_index)
        font = display.Font(size=m.text_size)

        speedometer = _get_speedometer(m.diameter)
        C = m.center
        self._dlist.paste(speedometer, m.top_left, speedometer)
        pos = (C[0], C[1] + m.text_offset)
        self._dlist.text(pos, text, m.text_color, self._load_font(font), "mb")
        font = dataclasses.replace(font, size=round(0.8 * font.size))
        pos = (C[0], C[1] + m.radius * 75 // 100)
        self._dlist.text(pos, metric.title, m.text_color, self._load_font(font), "mb")
        needle_coords = _coords(C, m.needle_radius)
        self._dlist.pieslice(needle_coords, needle_pos - 2, needle_pos + 2, m.needle_color)

        self._speedometer_index += 1

    def draw_statistics(self, stats):
        pass

    def polylines(
        self,
        lines: Sequence[Sequence[display.Point]],
        closed: bool = False,
        color: display.Color = (255, 255, 255, 255),
        width: int = 1,
    ) -> None:
        import itertools

        # flatten the points into `[[x1, y1, x2, y2, ...], ...]` because PIL
        # insists on tuple for the points if given.
        lines = [list(itertools.chain.from_iterable(line)) for line in lines]
        for line in lines:
            if closed:
                line = line + line[:2]  # make a copy with the first point at the end
            self._dlist.line(line, color, width)

    def _load_font(self, font: display.Font):
        args = dataclasses.astuple(font)
        try:
            return self._font_cache[args]
        except KeyError:
            path = os.path.join(os.path.dirname(__file__), "axelera-sans.ttf")
            f = self._font_cache[args] = PIL.ImageFont.truetype(path, size=font.size)
            return f

    def rectangle(self, p1, p2, fill=None, outline=None, width=1):
        self._dlist.rectangle((p1, p2), fill, outline, int(width))

    def textsize(self, text, font: display.Font = display.Font()):
        font = self._load_font(font)
        x1, y1, x2, y2 = font.getbbox(text)
        return (x2 - x1, y2 - y1)

    def text(
        self,
        p,
        text: str,
        txt_color: display.Color,
        back_color: display.OptionalColor = None,
        font: display.Font = display.Font(),
    ):
        if back_color is not None:
            w, h = self.textsize(text, font)
            self.rectangle(p, (p[0] + w, p[1] + h), back_color)
        self._dlist.text(p, text, txt_color, self._load_font(font))

    def keypoint(
        self, p: display.Point, color: display.Color = (255, 255, 255, 255), size=2
    ) -> None:
        r = size / 2
        p1, p2 = (p[0] - r, p[1] - r), (p[0] + r, p[1] + r)
        self._dlist.ellipse((p1, p2), color)

    def draw(self):
        def call_draw(d):
            if d[0] == 'paste':
                self._pil.paste(*d[1:])
            else:
                getattr(self._draw, d[0])(*d[1:])

        speedometers = self._dlist[: self._speedometer_dlist_calls]
        for d in self._dlist[self._speedometer_dlist_calls :]:
            call_draw(d)
        for d in speedometers:
            call_draw(d)
        self._img.update(pil=self._pil, color_format=self._img.color_format)

    def heatmap(self, data: np.ndarray, color_map: np.ndarray):
        indices = np.clip((data * len(color_map) - 1).astype(int), 0, len(color_map) - 1)
        rgba_mask = color_map[indices]
        mask_pil = PIL.Image.fromarray(rgba_mask)
        self._dlist.paste(mask_pil, (0, 0), mask_pil)

    def segmentation_mask(self, mask_data: SegmentationMask, color: Tuple[int]) -> None:
        x_adjust = y_adjust = 0
        mask, mbox = mask_data[-1], mask_data[4:8]
        img_size = (mbox[2] - mbox[0], mbox[3] - mbox[1])
        resized_image = cv2.resize(mask, img_size, interpolation=cv2.INTER_CUBIC)

        mid_point = np.iinfo(np.uint8).max // 2
        bool_array = resized_image > mid_point
        colored_mask = np.zeros((*bool_array.shape, 4), dtype=np.uint8)
        colored_mask[bool_array] = color

        mask_pil = PIL.Image.fromarray(colored_mask)
        offset = (mbox[0], mbox[1])
        self._dlist.paste(mask_pil, offset, mask_pil)

    def class_map_mask(self, class_map: np.ndarray, color_map: np.ndarray) -> None:
        colored_mask = color_map[class_map]
        colored_mask = cv2.resize(colored_mask, self._canvas_size)
        mask_pil = PIL.Image.fromarray(colored_mask)
        self._dlist.paste(mask_pil, (0, 0), mask_pil)

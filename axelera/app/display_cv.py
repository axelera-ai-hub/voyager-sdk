# Copyright Axelera AI, 2023
from __future__ import annotations

import collections
import dataclasses
import functools
import os
import queue
import time
from typing import TYPE_CHECKING, Sequence, Tuple
import uuid

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

# Named Draw List layers for the most commonly used indexes
FRAME = 0
USER = 1
TOPMOST = -1
SPEEDOS = -2


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


class CVWindow:
    def __init__(self, q, frame_sink, title, size):
        self._pane_size = (
            size if size != display.FULL_SCREEN else display.get_primary_screen_resolution()
        )
        self.title = title or None
        self._queue = q
        self._frame_sink = frame_sink
        self._meta_cache = display.MetaCache()
        self._speedometer_smoothing = display.SpeedometerSmoothing()
        self._layers: dict[uuid.UUID, display._Layer] = collections.defaultdict()
        self._current: dict[int, tuple[types.Image, meta.AxMeta | None]] = {}
        self._options: dict[int, CVOptions] = collections.defaultdict(CVOptions)
        self._num_streams = 0
        self._closed_sources = set()
        self._invalid = False
        if self.title:
            cv2.namedWindow(self.title, cv2.WINDOW_NORMAL)
            if size == display.FULL_SCREEN:
                cv2.setWindowProperty(self.title, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
            else:
                cv2.setWindowProperty(self.title, cv2.WND_PROP_ASPECT_RATIO, cv2.WINDOW_KEEPRATIO)
                cv2.resizeWindow(self.title, *self._pane_size)
            cv2.imshow(self.title, _make_splash(self._pane_size))

    def _num_sources(self, new_source_id: int) -> int:
        return (
            max(
                max(self._current.keys(), default=0),
                new_source_id,
            )
            + 1
        )

    def _handle_message(self, msg):
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
            return
        if isinstance(msg, display._SetOptions):
            self._options[msg.stream_id].update(**msg.options)
            return
        if isinstance(msg, display._Layer):
            self._layers[msg.id] = msg
            return
        if not isinstance(msg, display._Frame):
            LOG.debug(f"Unknown message: {msg}")
            return
        self._current[msg.stream_id] = (msg.image, msg.meta, blocking)
        self._invalid = True

    def update(self):
        try:
            while True:
                msg = self._queue.get(block=False)
                if msg is display.SHUTDOWN:
                    return
                if msg is display.THREAD_COMPLETED:
                    continue  # ignore, just wait for user to close
                self._handle_message(msg)
        except queue.Empty:
            pass

        if self._invalid and self._current:
            w, h = self._pane_size
            composite = types.Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8))
            pending_blocking_capture = False
            # Always render in descending stream ID order - window layers are rendered by stream 0.
            # If stream 0 is not rendered last, other streams may render on top of the window layers
            for source_id, (img, meta_data, blocking_flag) in sorted(
                self._current.items(), key=lambda x: x[0], reverse=True
            ):
                draw = CVDraw(
                    source_id,
                    self._num_sources(source_id),
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
                if blocking_flag:
                    pending_blocking_capture = True
            if self.title:
                cv2.imshow(self.title, composite.asarray(types.ColorFormat.BGR))
            if self._frame_sink:
                self._frame_sink.push(composite, block=pending_blocking_capture)
            self._invalid = False


class CVApp(display.App):
    SupportedOptions = CVOptions

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def _create_new_window(self, q, frame_sink, title, size):
        return CVWindow(q, frame_sink, title, size)

    def _run(self, interval=1 / 30):
        last_frame = time.time()
        while 1:
            self._create_new_windows()
            for wnd in self._wnds:
                wnd.update()
            now = time.time()
            remaining = max(1, int((interval - (now - last_frame)) * 1000))
            last_frame = now
            if all(
                wnd.title and cv2.getWindowProperty(wnd.title, cv2.WND_PROP_VISIBLE) <= 0.0
                for wnd in self._wnds
            ):
                return
            if cv2.waitKey(remaining) in (ord("q"), ord("Q"), 27, 32):  # ESC, q, Q, space
                return
            if self.has_thread_completed:
                return

    def _destroy_all_windows(self):
        cv2.destroyAllWindows()


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
    x = PIL.Image.open(f'{here}/render_assets/speedo-alpha-transparent.png')
    return x.resize((diameter, diameter))


@functools.lru_cache(maxsize=128)
def _load_pil_image_from_file(filename):
    return PIL.Image.open(filename)


def _normalize_anchor_point(
    x: int,
    y: int,
    width: int,
    height: int,
    anchor_x: str,
    anchor_y: str,
) -> Tuple[int, int]:
    '''
    Default pillow anchor is the top left corner. If we are using a different anchor point,
    calculate the x and y co-ordinate of the top-left corner from the given point, anchor and
    renderable size, and use that.
    '''
    if anchor_x == 'center':
        x -= width / 2
    elif anchor_x == 'right':
        x -= width
    if anchor_y == 'center':
        y -= height / 2
    elif anchor_y == 'bottom':
        y -= height
    return int(x), int(y)


class _DrawList(list):
    def __getattr__(self, name):
        def _draw(*args):
            self.append((name,) + args)

        return _draw


class _LayeredDrawList(collections.defaultdict):
    def __init__(self):
        super().__init__(_DrawList)

    def __getattr__(self, name):
        return getattr(self[USER], name)

    def __iter__(self):
        dlists = list(self.keys())
        bottom = sorted([i for i in dlists if i >= 0])
        top = sorted([i for i in dlists if i < 0])
        for dlist in bottom + top:
            yield from self[dlist]

    def __len__(self):
        return sum(len(layer) for layer in self.values())


class CVCanvas(display.Canvas):
    '''
    CVCanvas uses the information from display.Canvas to scale and position
    image coordinates correctly in the OpenCV window.

    A bounding box of (100, 100, 200, 200) in the image space will be drawn as a rectangle in the
    CV space at
    (0 + 100*0.5859, 100*0.5859, 200*0.5859, 200*0.5859) = (59, 59, 117, 117)
    Given a scale factor of 0.5859, and rounding to integer pixel values.
    '''

    @property
    def top(self):
        return self.bottom - self.height

    def cvp(self, p: Tuple[int, int]) -> Tuple[int, int]:
        '''Convert a logical point to a cv point.'''
        return (round(self.left + p[0] * self.scale), round(self.top + p[1] * self.scale))


def _create_canvas(
    source_id: int, num_sources: int, image_size: Tuple[int, int], pane_size: Tuple[int, int]
) -> CVCanvas:
    (x, y, w, h) = display.pane_position(source_id, num_sources, image_size, pane_size)
    return CVCanvas(int(x), int(y + h), int(w), int(h), w / image_size[0], *pane_size)


def _normalize_img_color(img: types.Image) -> types.Image:
    return img.asarray(types.ColorFormat.RGB)


class CVDraw(display.Draw):
    def __init__(
        self,
        stream_id: int,
        num_streams: int,
        composite: types.Image,
        image: types.Image,
        layers: list[display._Layer],
        options: CVOptions = CVOptions(),
        window_options: CVOptions = CVOptions(),
        speedometer_smoothing: display.SpeedometerSmoothing = None,
    ):
        if stream_id >= num_streams:
            LOG.warning(
                f"Source ID {stream_id} is too high for the number of sources ({num_streams}), expect rendering glitches."
            )
        self._source_id = stream_id
        self._img = composite
        self._image_size = (image.width, image.height)
        self._canvas = _create_canvas(
            self._source_id,
            num_streams,
            self._image_size,
            self.composite_size,
        )
        rgb = self._img.asarray('RGB')
        preprocessed = rgb_to_grayscale_rgb(_normalize_img_color(image), options.grayscale)
        x0, y0 = self._canvas.cvp((0, 0))
        x1, y1 = x0 + self.canvas_size[0], y0 + self.canvas_size[1]
        cv2.resize(preprocessed, self.canvas_size, dst=rgb[y0:y1, x0:x1])
        self._pil = PIL.Image.fromarray(rgb_to_grayscale_rgb(rgb, options.grayscale))
        self._draw = PIL.ImageDraw.Draw(self._pil, "RGBA")
        self._dlist = _LayeredDrawList()
        self._font_cache = {}
        self._font_offset_cache = {}
        self._speedometer_index = 0
        self._options = options
        self._window_options = window_options
        self._speedometer_smoothing = (
            speedometer_smoothing
            if (self._window_options.speedometer_smoothing or self._options.speedometer_smoothing)
            else None
        )

        if options.title:
            layers.append(display.gen_title_message(self._source_id, options))
        if window_options.title and self._source_id == 0:
            layers.append(display.gen_title_message(-1, window_options))

        for x in layers:
            if x.stream_id == -1:
                pt_transform = lambda pt: pt
                canvas_size = self.composite_size
            else:
                pt_transform = self._canvas.cvp
                canvas_size = (image.width, image.height)

            pt = pt_transform(x.position.as_px(canvas_size))
            if isinstance(x, display._Text):
                font = display.Font(size=x.font_size)
                text_size = self.textsize(x.text, font)
                pt = _normalize_anchor_point(
                    *pt,
                    text_size[0],
                    text_size[1],
                    x.anchor_x,
                    x.anchor_y,
                )
                color = x.color[:3] + (int(x.color[3] * x.visibility),)
                bgcolor = x.bgcolor[:3] + (int(x.bgcolor[3] * x.visibility),)
                txt = PIL.Image.new('RGBA', text_size, bgcolor)
                txt_draw = PIL.ImageDraw.Draw(txt, "RGBA")
                txt_draw.text((0, 0), x.text, color, self._load_font(font), "lt")
                self._dlist[TOPMOST].paste(txt, pt, txt)
            elif isinstance(x, display._Image):
                img = _load_pil_image_from_file(x.path)
                scale = display.canvas_scale_to_img_scale(x.scale, img.size, canvas_size)
                img = img.resize((int(scale * img.width), int(scale * img.height)))
                if x.visibility < 1.0:
                    img = img.convert("RGBA")
                    alpha = np.array(img.split()[-1])
                    alpha = (alpha * x.visibility).astype(np.uint8)
                    img.putalpha(PIL.Image.fromarray(alpha))
                pt = _normalize_anchor_point(
                    *pt,
                    img.width,
                    img.height,
                    x.anchor_x,
                    x.anchor_y,
                )

                self._dlist[TOPMOST].paste(img, pt, img)
            elif isinstance(x, display._Rectangle):
                x2, y2 = pt_transform(x.bottom_right.as_px(canvas_size))
                x1, x2 = sorted((int(pt[0]), int(x2)))
                y1, y2 = sorted((int(pt[1]), int(y2)))
                color = x.color[:3] + (int(x.color[3] * x.visibility),)
                # Draw the outline onto a transparent tile and paste with its own
                # alpha so visibility blends like the other layers.
                rect = PIL.Image.new('RGBA', (x2 - x1 + 1, y2 - y1 + 1), (0, 0, 0, 0))
                PIL.ImageDraw.Draw(rect, "RGBA").rectangle(
                    [0, 0, x2 - x1, y2 - y1], outline=color, width=1
                )
                self._dlist[TOPMOST].paste(rect, (x1, y1), rect)
            else:
                LOG.debug(f"Unknown layer type {x.__class__.__name__} ignoring...")

    @property
    def options(self) -> CVOptions:
        return self._options

    @property
    def canvas_size(self) -> display.Point:
        return self._canvas.size

    @property
    def image_size(self) -> display.Point:
        '''Return the original, unscaled size of the input image'''
        return self._image_size

    @property
    def composite_size(self) -> display.Point:
        return (self._img.width, self._img.height)

    def draw_speedometer(self, metric: inf_tracers.TraceMetric):
        if self._speedometer_smoothing:
            self._speedometer_smoothing.update(metric)
        text = display.calculate_speedometer_text(metric, self._speedometer_smoothing)
        needle_pos = display.calculate_speedometer_needle_pos(metric, self._speedometer_smoothing)
        m = display.SpeedometerMetrics(self.composite_size, self._speedometer_index)
        font = display.Font(size=m.text_size)

        speedometer = _get_speedometer(m.diameter)
        C = m.center
        self._dlist[SPEEDOS].paste(speedometer, m.top_left, speedometer)
        pos = (C[0], C[1] + m.text_offset)
        self._dlist[SPEEDOS].text(pos, text, m.text_color, self._load_font(font), "mb")
        font = dataclasses.replace(font, size=round(0.8 * font.size))
        pos = (C[0], C[1] + m.radius * 75 // 100)
        self._dlist[SPEEDOS].text(pos, metric.title, m.text_color, self._load_font(font), "mb")
        needle_coords = _coords(C, m.needle_radius)
        # Interpret RGB color as BGR in the OpenCV renderer, so there is clear
        # visual feedback that we are rendering with OpenCV.
        needle_color = m.needle_color[2::-1] + (m.needle_color[3],)
        self._dlist[SPEEDOS].pieslice(needle_coords, needle_pos - 2, needle_pos + 2, needle_color)

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
        lines = [
            list(itertools.chain.from_iterable([self._canvas.cvp(pt) for pt in line]))
            for line in lines
        ]
        width = round(width * self._canvas.scale)
        for line in lines:
            if closed:
                line = line + line[:2]  # make a copy with the first point at the end
            self._dlist.line(line, color, width)

    def _get_offset(self, font: PIL.ImageFont):
        # We only need the y offset, so just use any text with max descender.
        (_, _), (_, offset_y) = font.font.getsize("Agypq")
        return offset_y

    def _load_font(self, font: display.Font):
        args = dataclasses.astuple(font)
        try:
            return self._font_cache[args]
        except KeyError:
            path = os.path.join(os.path.dirname(__file__), "render_assets", "axelera-sans.ttf")
            f = self._font_cache[args] = PIL.ImageFont.truetype(path, size=font.size)
            self._font_offset_cache[args] = self._get_offset(f)
            return f

    def rectangle(self, p1, p2, fill=None, outline=None, width=1):
        self._dlist.rectangle(
            (self._canvas.cvp(p1), self._canvas.cvp(p2)), fill, outline, int(width)
        )

    def textsize(self, text, font: display.Font = display.Font()):
        font = self._load_font(font)
        x1, y1, x2, y2 = font.getbbox(text)
        return (x2 - x1, y2 - y1)

    def fontoffset(self, font: display.Font = display.Font()):
        try:
            return self._font_offset_cache[dataclasses.astuple(font)]
        except KeyError:
            self._load_font(font)
            return self._font_offset_cache[dataclasses.astuple(font)]

    def text(
        self,
        p,
        text: str,
        txt_color: display.Color,
        back_color: display.OptionalColor = None,
        font: display.Font = display.Font(),
    ):
        p = self._canvas.cvp(p)
        if back_color is not None:
            # We don't use regular rectangle method here - as text scales differently.
            w, h = self.textsize(text, font)
            offset = self.fontoffset(font)
            pt0 = (p[0], p[1] + offset)
            pt1 = (pt0[0] + w, pt0[1] + h)
            self._dlist.rectangle((pt0, pt1), back_color, None, 1)
        self._dlist.text(p, text, txt_color, self._load_font(font))

    def keypoint(
        self, p: display.Point, color: display.Color = (255, 255, 255, 255), size=2
    ) -> None:
        size = round(size * self._canvas.scale)
        r = size / 2
        p1, p2 = (p[0] - r, p[1] - r), (p[0] + r, p[1] + r)
        self._dlist.ellipse((self._canvas.cvp(p1), self._canvas.cvp(p2)), color)

    def draw(self):
        def call_draw(d):
            if d[0] == 'paste':
                self._pil.paste(*d[1:])
            else:
                getattr(self._draw, d[0])(*d[1:])

        for d in self._dlist:
            call_draw(d)
        self._img.update(pil=self._pil, color_format=types.ColorFormat.RGBA)

    def heatmap(self, data: np.ndarray, color_map: np.ndarray):
        indices = np.clip((data * len(color_map) - 1).astype(int), 0, len(color_map) - 1)
        rgba_mask = color_map[indices]
        mask_pil = PIL.Image.fromarray(rgba_mask)
        self._dlist.paste(mask_pil, self._canvas.cvp((0, 0)), mask_pil)

    def segmentation_mask(self, mask_data: SegmentationMask, color: Tuple[int]) -> None:
        mask, mbox = mask_data[-1], mask_data[4:8]

        p1 = self._canvas.cvp((mbox[0], mbox[1]))
        p2 = self._canvas.cvp((mbox[2], mbox[3]))
        img_size = (p2[0] - p1[0], p2[1] - p1[1])
        if 0 in img_size or 0 in mask.shape:
            return
        resized_image = cv2.resize(mask, img_size, interpolation=cv2.INTER_LINEAR)

        mid_point = np.iinfo(np.uint8).max // 2
        bool_array = resized_image > mid_point
        colored_mask = np.zeros((*bool_array.shape, 4), dtype=np.uint8)
        colored_mask[bool_array] = color

        mask_pil = PIL.Image.fromarray(colored_mask)
        self._dlist.paste(mask_pil, p1, mask_pil)

    def class_map_mask(self, class_map: np.ndarray, color_map: np.ndarray) -> None:
        colored_mask = color_map[class_map]
        colored_mask = cv2.resize(colored_mask, self.canvas_size)
        mask_pil = PIL.Image.fromarray(colored_mask)
        self._dlist.paste(mask_pil, self._canvas.cvp((0, 0)), mask_pil)

    def draw_image(self, image: np.ndarray):
        if image.dtype not in [np.uint8, np.float32, np.float64]:
            raise ValueError("draw_image: image dtype must be np.uint8, np.float32, or np.float64")

        def float_to_uint8_image(float_img):
            d_min = np.min(float_img)
            d_max = np.max(float_img)

            if np.isclose(d_min, d_max):
                uint8_img = np.zeros_like(image, dtype=np.uint8)
            else:
                uint8_img = np.clip((image - d_min) / (d_max - d_min) * 255.0, 0, 255).astype(
                    np.uint8
                )

            return uint8_img

        if image.ndim == 4 and image.shape[0] == 1:
            image = image.squeeze(axis=0)  # Remove batch dimension
        if image.ndim == 3 and image.shape[0] == 1:
            image = image.squeeze(axis=0)  # Remove channel dimension if single channel

        if image.dtype == np.float32 or image.dtype == np.float64:
            image = float_to_uint8_image(image)

        # Convert CHW to HWC format
        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))

        image = cv2.resize(image, self.canvas_size)

        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGBA)
        else:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2RGBA)

        pil_image = PIL.Image.fromarray(image)
        self._dlist.paste(pil_image, self._canvas.cvp((0, 0)), pil_image)

# Copyright Axelera AI, 2023
from __future__ import annotations

import collections
import contextlib
import ctypes
from dataclasses import dataclass, field
import functools
import math
import operator
import os
import queue
import sys
import time
from typing import TYPE_CHECKING, Any, Mapping, Optional, Sequence, Tuple
import uuid
import weakref

import numpy as np
import pyglet
from pyglet.gl import (
    glBindTexture,
    GL_TEXTURE_2D,
    GL_RED,
    GL_RG,
    glTexParameteri,
    GL_TEXTURE_MIN_FILTER,
    GL_TEXTURE_MAG_FILTER,
    GL_LINEAR,
    GL_NEAREST,
    glTexImage2D,
    GL_UNSIGNED_BYTE,
    glActiveTexture,
    GL_TEXTURE0,
    GL_TEXTURE1,
    GL_TEXTURE2,
    GL_TEXTURE3,
    GL_RGBA,
    GL_RGB,
    glPixelStorei,
    GL_UNPACK_ROW_LENGTH,
)

from axelera import types

from . import config, display, logging_utils, meta
from .utils import catchtime, get_backend_opengl_version

if TYPE_CHECKING:
    from . import inf_tracers

_GL_API, _GL_MAJOR, _GL_MINOR = get_backend_opengl_version(config.env.opengl_backend)
# If using gles, shadow_window must be disabled before any pyglet imports other than
# `import pyglet`. Otherwise, importing pyglet.gl (either by us or by pyglet) will
# cause an error. Hence, all pyglet.* imports should be below here.
if _GL_API == "gles":
    pyglet.options.shadow_window = False


LOG = logging_utils.getLogger(__name__)
KEYPOINT_6 = np.array([[0, 0, 1], [0, 1, 1], [1, 1, 1]], np.float64)
KEYPOINT_4 = np.array([[0, 1], [1, 1]], np.float64)
KEYPOINT_2 = np.full((4, 4), 1, np.float64)

# some env based configs for tweaking performance
_LOW_LATENCY_STREAMS = config.env.render_low_latency_streams
_RENDER_FONT_SCALE = config.env.render_font_scale
_RENDER_LINE_WIDTH = config.env.render_line_width
_RENDER_FPS = config.env.render_fps
_SHOW_BUFFER_STATUS = config.env.render_show_buffer_status
_SHOW_RENDER_FPS = config.env.render_show_fps
_STREAM_QUEUE_SIZE = config.env.render_queue_size
_BOX_KEYPOINTS = config.env.render_box_keypoints

_HIGH_RES_STREAM_COUNT = 2  # first N streams kept at full resolution
_SECONDARY_STREAM_SCALE = 0.10  # scale factor applied to the remaining streams
# Primary streams (source_id < _HIGH_RES_STREAM_COUNT) occupy the main area;
# secondary streams share the remaining right-hand strip.
_SECONDARY_PANEL_WIDTH = 0.25  # fraction of window width given to the secondary side panel

# Soft as won't fail, but may have visual glitches if exceeded
SOFT_MAX_STREAMS = 100

# Offsets for OpenGL groups used for rendering - higher groups
# are rendered on top of lower groups.
GR_BACK_OFFSET = 0  # Bottom
GR_FORE_OFFSET = GR_BACK_OFFSET + SOFT_MAX_STREAMS
GR_SPEEDO_OFFSET = GR_FORE_OFFSET + SOFT_MAX_STREAMS
GR_LAYER_OFFSET = GR_SPEEDO_OFFSET + SOFT_MAX_STREAMS
GR_PROG_OFFSET = GR_LAYER_OFFSET + SOFT_MAX_STREAMS  # Top


class Box(pyglet.shapes.MultiLine):
    def __init__(self, x, y, width, height, **kwargs):
        x1, y1 = x + width, y + height
        pts = [(x, y), (x1, y), (x1, y1), (x, y1)]
        super().__init__(*pts, closed=True, **kwargs)


@contextlib.contextmanager
def _render_to_texture(width, height):
    fb = pyglet.image.Framebuffer()
    t = pyglet.image.Texture.create(width, height)
    fb.attach_texture(t)
    fb.bind()
    try:
        yield t
    finally:
        fb.unbind()


class _SpriteProxy:
    def __init__(self, sprite: pyglet.sprite.Sprite):
        super().__setattr__("sprite", sprite)

    def __getattr__(self, x):
        return getattr(self.sprite, x)

    def __setattr__(self, x, v):
        setattr(self.sprite, x, v)


_label_argnames = (
    "text",
    "font_name",
    "font_size",
    "weight",
    "italic",
    "stretch",
    "color",
    "align",
    "multiline",
    "dpi",
    "back_color",
)


# ============================================================================
# GLSL SHADER FUNCTION LIBRARY - Composable color conversion and effects
# ============================================================================
# Format: {function_name: (function_body, uniforms, tex_size_expr)}
# All functions share: in vec4 vertex_colors; in vec3 texture_coords; out vec4 final_colors;

_shader_functions = {
    "yuv_to_rgb": (
        """
vec4 yuv_to_rgb(float y, float u, float v) {
    float y_scaled = 1.164 * (y - 0.0627);
    float u_shifted = u - 0.5020;
    float v_shifted = v - 0.5020;

    float r = y_scaled + 1.596 * v_shifted;
    float g = y_scaled - 0.391 * u_shifted - 0.813 * v_shifted;
    float b = y_scaled + 2.018 * u_shifted;
    return vec4(r, g, b, 1.0);
}""",
        [],
        "",
    ),
    "convert_nv12": (
        """
vec4 convert_nv12(vec2 coords) {
    float y = texture(y_tex, coords).r;
    vec2 uv = texture(uv_tex, coords).rg;
    return yuv_to_rgb(y, uv.x, uv.y);
}""",
        ["uniform sampler2D y_tex;", "uniform sampler2D uv_tex;"],
        "vec2(textureSize(y_tex, 0))",
    ),
    "convert_i420": (
        """
vec4 convert_i420(vec2 coords) {
    float y = texture(y_tex, coords).r;
    float u = texture(u_tex, coords).r;
    float v = texture(v_tex, coords).r;
    return yuv_to_rgb(y, u, v);
}""",
        ["uniform sampler2D y_tex;", "uniform sampler2D u_tex;", "uniform sampler2D v_tex;"],
        "vec2(textureSize(y_tex, 0))",
    ),
    "convert_nv16": (
        """
vec4 convert_nv16(vec2 coords) {
    float y = texture(y_tex, coords).r;
    vec2 uv = texture(uv_tex, coords).rg;
    return yuv_to_rgb(y, uv.x, uv.y);
}""",
        ["uniform sampler2D y_tex;", "uniform sampler2D uv_tex;"],
        "vec2(textureSize(y_tex, 0))",
    ),
    "convert_yuy2": (
        """
vec4 convert_yuy2(vec2 coords) {
    vec2 pixel_pos = coords * tex_size;
    float x_mod = mod(floor(pixel_pos.x), 2.0);
    vec2 sample_pos = vec2(floor(pixel_pos.x * 0.5) + 0.5, floor(pixel_pos.y) + 0.5);
    vec2 sample_coord = sample_pos / vec2(tex_size.x * 0.5, tex_size.y);
    vec4 yuv_packed = texture(tex, sample_coord);
    float y = mix(yuv_packed.r, yuv_packed.b, x_mod);
    float u = yuv_packed.g;
    float v = yuv_packed.a;
    return yuv_to_rgb(y, u, v);
}""",
        ["uniform sampler2D tex;", "uniform vec2 tex_size;"],
        "tex_size",
    ),
    "convert_gray": (
        """
vec4 convert_gray(vec2 coords) {
    float gray = texture(tex, coords).r;
    return vec4(gray, gray, gray, 1.0);
}""",
        ["uniform sampler2D tex;"],
        "vec2(textureSize(tex, 0))",
    ),
    "convert_rgb": (
        """
vec4 convert_rgb(vec2 coords) {{
    return vec4(texture(tex, coords).{swizzle}, 1.0);
}}""",
        ["uniform sampler2D tex;"],
        "vec2(textureSize(tex, 0))",
    ),
    "convert_rgba": (
        """
vec4 convert_rgba(vec2 coords) {{
    return texture(tex, coords).{swizzle};
}}""",
        ["uniform sampler2D tex;"],
        "vec2(textureSize(tex, 0))",
    ),
    "convert_fallback": (
        """
vec4 convert_fallback(vec2 coords) {
    return texture(tex, coords);
}""",
        ["uniform sampler2D tex;"],
        "vec2(textureSize(tex, 0))",
    ),
    "apply_pixelate": (
        """
vec2 apply_pixelate(vec2 coords, vec2 tex_size, float factor) {
    vec2 pixel_size = vec2(factor) / tex_size;
    return floor(coords / pixel_size) * pixel_size;
}""",
        [],
        "",
    ),
    "apply_grayscale": (
        """
vec4 apply_grayscale(vec4 color, float amount) {{
    float grey = dot(color.rgb, vec3(0.299, 0.587, 0.114));
    vec4 grey_color = vec4(grey, grey, grey, color.a);
    return mix(color, grey_color, {grayness});
}}""",
        [],
        "",
    ),
}


def _compose_shader(
    format_name: str,
    blur: float | None = None,
    grayscale: float | None = None,
    grayscale_area: str = 'all',
) -> str:
    """Compose a complete shader from color conversion and effect functions."""

    format_key = format_name.upper()

    # Determine converter function and any format-specific substitutions
    if format_key in ['RGB', 'BGR']:
        swizzle = 'bgr' if format_key == 'BGR' else 'rgb'
        converter_name = "convert_rgb"
        func_body, uniforms, tex_size_expr = _shader_functions[converter_name]
        converter_body = func_body.format(swizzle=swizzle)
    elif format_key in ['RGBA', 'BGRA', 'RGBX', 'BGRX']:
        swizzle = 'bgra' if format_key in ['BGRA', 'BGRX'] else 'rgba'
        converter_name = "convert_rgba"
        func_body, uniforms, tex_size_expr = _shader_functions[converter_name]
        converter_body = func_body.format(swizzle=swizzle)
    else:
        # Direct lookup for NV12, I420, etc.
        converter_name = f"convert_{format_key.lower()}"
        if converter_name in _shader_functions:
            converter_body, uniforms, tex_size_expr = _shader_functions[converter_name]
        else:
            converter_name = "convert_fallback"
            converter_body, uniforms, tex_size_expr = _shader_functions[converter_name]

    # Collect function bodies
    functions = []

    # Add yuv_to_rgb helper if using a YUV format
    yuv_formats = {'convert_nv12', 'convert_i420', 'convert_nv16', 'convert_yuy2'}
    if converter_name in yuv_formats:
        functions.append(_shader_functions["yuv_to_rgb"][0])

    functions.append(converter_body)

    if blur is not None:
        functions.append(_shader_functions["apply_pixelate"][0])
    if grayscale is not None and grayscale > 0.0:
        # Format grayscale function with area-based expression like the original
        grayscale_expr = {
            'all': str(grayscale),
            'left': f'texture_coords.x < 0.5 ? {grayscale} : 0.0',
            'right': f'texture_coords.x > 0.5 ? {grayscale} : 0.0',
            'top': f'texture_coords.y < 0.5 ? {grayscale} : 0.0',
            'bottom': f'texture_coords.y > 0.5 ? {grayscale} : 0.0',
        }[grayscale_area]
        grayscale_func = _shader_functions["apply_grayscale"][0].format(grayness=grayscale_expr)
        functions.append(grayscale_func)

    # Build main() function body
    main_body = []

    # Apply pixelation if needed
    if blur is not None:
        main_body.append(
            f"    vec2 coords = apply_pixelate(texture_coords.xy, {tex_size_expr}, {blur});"
        )
    else:
        main_body.append("    vec2 coords = texture_coords.xy;")

    # Convert color
    main_body.append(f"    vec4 color = {converter_name}(coords);")

    # Apply grayscale if needed
    if grayscale is not None and grayscale > 0.0:
        main_body.append("    color = apply_grayscale(color, 0.0);")

    # Final output
    main_body.append("    final_colors = color * vertex_colors;")

    # Assemble complete shader
    shader_parts = (
        [
            "#version 150 core",
            "",
            "// Uniforms",
        ]
        + uniforms
        + [
            "",
            "// Varyings",
            "in vec4 vertex_colors;",
            "in vec3 texture_coords;",
            "out vec4 final_colors;",
            "",
            "// Functions",
        ]
        + functions
        + [
            "",
            "void main() {",
        ]
        + main_body
        + [
            "}",
        ]
    )

    return "\n".join(shader_parts)


@functools.lru_cache(maxsize=100)
def _get_shader(
    format_name: str,
    blur: float | None = None,
    grayscale: float | None = None,
    grayscale_area: str = 'all',
) -> pyglet.program.ShaderProgram:
    """Get a shader with color conversion and optional effects.

    Args:
        format_name: Color format (NV12, I420, RGB, etc.)
        blur: Pixelation factor (None = no blur)
        grayscale: Grayscale amount 0.0-1.0 (None = no grayscale)
        grayscale_area: Area to apply grayscale ('all', 'left', 'right', 'top', 'bottom')

    Returns:
        Compiled shader program
    """
    shader_source = _compose_shader(format_name, blur, grayscale, grayscale_area)
    return pyglet.gl.current_context.create_program(
        (pyglet.sprite.vertex_source, 'vertex'),
        (shader_source, 'fragment'),
    )


class SpritePool:
    def __init__(self):
        self._pool = []

    def create_sprite(self, image, x, y, z, rotation=None, batch=None, group=None, opacity=255):
        try:
            # TODO possible opt is to select sprites based on t.owner
            s = self._pool.pop()
            s.image = image
            s.batch = batch
            s.group = group
            s.visible = True
            s.opacity = opacity
            s.update(x=x, y=y, z=z, rotation=rotation)
        except IndexError:
            s = pyglet.sprite.Sprite(image, x, y, z, batch=batch, group=group)
            s.opacity = opacity
            if rotation is not None:
                s.rotation = rotation
        proxy = _SpriteProxy(s)
        weakref.finalize(proxy, self._remove, s)
        return proxy

    def _remove(self, s):
        self._pool.append(s)
        s.visible = False


_ANCHOR_ADJUST_X = dict(left=0, center=0.5, right=1.0)
_ANCHOR_ADJUST_Y = dict(top=1.0, center=0.5, baseline=0.2, bottom=0.0)


class LabelPool:
    def __init__(self, pixel_ratio):
        self._texture_bin = pyglet.image.atlas.TextureBin()
        self._textures = {}
        self._sprites = SpritePool()
        # 1.0 for normal DPI, 2.0 for retina/HiDPI.  On HiDPI we need to create
        # a sprite twice as big and scale it to normal size because the rendering
        # of text (and other line primitives) is effectively done at 2x resolution.
        self._pixel_ratio = pixel_ratio

    def create_label(
        self,
        text="",
        font_name=None,
        font_size=None,
        weight='normal',
        italic=False,
        stretch=False,
        color=(255, 255, 255, 255),
        x=0,
        y=0,
        z=0,
        width=None,
        height=None,
        anchor_x="left",
        anchor_y="baseline",
        align="left",
        multiline=False,
        dpi=None,
        rotation=0,
        batch=None,
        group=None,
        back_color=None,
        opacity=255,
    ):
        all_args = locals()
        args = tuple(all_args[k] for k in _label_argnames)
        try:
            t = self._textures[args]
        except KeyError:
            t = self._textures[args] = self._new_texture(args)
        x -= _ANCHOR_ADJUST_X[anchor_x] * t.width
        y -= _ANCHOR_ADJUST_Y[anchor_y] * t.height
        s = self._sprites.create_sprite(t, x, y, z, rotation, batch, group, opacity)
        # for HiDPI scale the texture on rendering
        s.scale = _RENDER_FONT_SCALE / self._pixel_ratio
        return s

    def _new_texture(self, args):
        kwargs = dict(zip(_label_argnames, args))
        back_color = kwargs.pop("back_color")
        BORDER = 1
        label = pyglet.text.Label(x=BORDER, y=BORDER, **kwargs, anchor_y="bottom")
        width, height = label.content_width, label.content_height
        # for HiDPI create a texture x2 size
        width = math.ceil((width + BORDER * 2) * self._pixel_ratio)
        height = math.ceil((height + BORDER * 2) * self._pixel_ratio)
        with _render_to_texture(width, height) as texture:
            if back_color is not None:
                pyglet.shapes.Rectangle(0, 0, width, height, back_color).draw()
            label.draw()
        t = self._texture_bin.add(texture.get_image_data())
        return t


@functools.lru_cache(maxsize=1)
def _load_fonts():
    # Barlow Regular:
    pyglet.font.add_file(
        os.path.join(os.path.dirname(__file__), "render_assets", "axelera-sans.ttf")
    )


@functools.lru_cache(maxsize=10000)
def _textsize(text, name, pts):
    label = pyglet.text.Label(text, font_name=name, font_size=pts)
    return label.content_width, label.content_height


def _determine_font_params(font: display.Font()) -> Tuple[str, float]:
    '''Convert font.name/size (in pixels high) to font_name/font_size (in points)'''
    _load_fonts()
    name = "Barlow Regular" if font.family == display.FontFamily.sans_serif else "Times"
    pts = font.size * 96 / 72
    _, height = _textsize('Iy', name, pts)
    while abs(font.size - height) > 0.5:
        pts += (font.size - height) / 4
        _, height = _textsize('Iy', name, pts)
    return (name, pts)


def _add_alpha(c):
    if c is not None:
        r, g, b, *a = c
        c = r, g, b, a[0] if a else 255
    return c


@dataclass
class GLCanvas(display.Canvas):
    '''
    GLCanvas uses the information from display.Canvas to scale and position
    image coordinates correctly in the OpenGL GL coordinate system. GL Y direction
    is inverted (0 is the bottom of the screen).

    A bounding box of (100, 100, 200, 200) in the image space will be drawn as a rectangle in the
    GL space at
    (0 + 100*0.5859, 475 - 100*0.5859, 200*0.5859, 200*0.5859) = (59, 416, 117, 117)
    Given a scale factor of 0.5859, and rounding to integer pixel values.
    '''

    def glp(self, p: Tuple[int, int]) -> Tuple[int, int]:
        '''Convert a logical point to a gl point.'''
        return (
            round(self.left + p[0] * self.scale),
            round(self.window_height - self.bottom - p[1] * self.scale),
        )

    def contains(self, mx: int, my: int) -> bool:
        '''Return True if the mouse position (in pyglet/GL coords) is inside this pane.'''
        gl_top = self.window_height - self.bottom
        return self.left <= mx <= self.left + self.width and gl_top - self.height <= my <= gl_top


class ProgressDraw:
    def __init__(self, source_id: int, layout_slot: int, num_slots: int, window_size):
        self._source_id = source_id
        self._layout_slot = layout_slot
        self._canvas = _create_canvas(self._layout_slot, num_slots, window_size, window_size)
        self._p = ProgressBar(
            *self._canvas.glp((window_size[0] / 2, window_size[1] - 14)),
            100,
            10,
            (255, 255, 255, 255),
            (0, 0, 0, 255),
        )

    def resize(self, num_slots: int, window_size: Tuple[int, int], layout_slot: int):
        self._layout_slot = layout_slot
        self._canvas = _create_canvas(self._layout_slot, num_slots, window_size, window_size)
        self._p.move(*self._canvas.glp((window_size[0] / 2, window_size[1] - 14)))

    def set_position(self, value):
        self._p.set_position(value)

    def draw(self):
        self._p.draw()


def _move_sprite(sprite: pyglet.sprite.Sprite, canvas: GLCanvas):
    sprite.scale_x = canvas.scale
    sprite.scale_y = -canvas.scale
    sprite.x, sprite.y = canvas.glp((0, 0))


@functools.lru_cache(maxsize=128)
def _load_image_from_file(filename: str):
    return pyglet.image.load(filename)


def _load_sprite_from_file(
    filename: str, scale, canvas_size, batch, group
) -> pyglet.sprite.Sprite:
    i = _load_image_from_file(filename)
    s = pyglet.sprite.Sprite(i, 0, 0, batch=batch, group=group)
    s.scale = display.canvas_scale_to_img_scale(scale, (s.width, s.height), canvas_size)
    return s


def _create_canvas(
    layout_slot: int,
    num_slots: int,
    image_size: Tuple[int, int],
    window_size: Tuple[int, int],
    multi_res_layout: bool = False,
):
    if multi_res_layout:
        num_primary = min(_HIGH_RES_STREAM_COUNT, num_slots)
        num_secondary = max(0, num_slots - _HIGH_RES_STREAM_COUNT)
        is_secondary = layout_slot >= _HIGH_RES_STREAM_COUNT
        if num_secondary > 0:
            win_w, win_h = window_size
            side_x = round(win_w * (1 - _SECONDARY_PANEL_WIDTH))
            if not is_secondary:
                x, y, w, h = display.pane_position(
                    layout_slot, num_primary, image_size, (side_x, win_h)
                )
            else:
                sec_id = layout_slot - _HIGH_RES_STREAM_COUNT
                sub_win_w = win_w - side_x
                x, y, w, h = display.pane_position(
                    sec_id, num_secondary, image_size, (sub_win_w, win_h)
                )
                x += side_x
            return GLCanvas(x, y, w, h, w / image_size[0], *window_size)
    (x, y, w, h) = display.pane_position(layout_slot, num_slots, image_size, window_size)
    return GLCanvas(x, y, w, h, w / image_size[0], *window_size)


class MasterDraw:
    def __init__(self, window: pyglet.window.Window, label_pool: LabelPool):
        self._label_pool = label_pool
        self._batch = pyglet.graphics.Batch()
        self._window = window
        self._draws: dict[int, GLDraw] = {}
        self._progresses: dict[int, ProgressDraw] = {}
        self._meta_cache = display.MetaCache()
        self._speedometer_smoothing = display.SpeedometerSmoothing()
        self._options: dict[int, GLOptions] = collections.defaultdict(GLOptions)
        self._layers: dict[uuid.UUID, display._Layer] = collections.defaultdict()
        self._render_state: dict[int, dict[Any, Any]] = collections.defaultdict(dict)
        self._highlight_display_id: Optional[int] = None
        self._primary_sources: list[int] = list(range(_HIGH_RES_STREAM_COUNT))

    def clear_state(self, source_id: int):
        if 'grid_sprites' in self._render_state[source_id]:
            for sprite in self._render_state[source_id]['grid_sprites'].values():
                sprite.delete()

        # `_` denotes keys which are internally managed and should not be user clear-able
        preserved = {k: v for k, v in self._render_state[source_id].items() if k.startswith('_')}
        self._render_state[source_id] = preserved

    def _num_sources(self, new_source_id: int) -> int:
        return (
            max(
                max(self._draws.keys(), default=0),
                max(self._progresses.keys(), default=0),
                new_source_id,
            )
            + 1
        )

    def set_primary_sources(self, primary_sources: list[int]) -> None:
        self._primary_sources = list(primary_sources)

    def _layout_slot(self, source_id: int) -> int:
        if not self._options[-1].multi_res_layout:
            return source_id
        if source_id in self._primary_sources:
            return self._primary_sources.index(source_id)
        # Secondary stream: stable ordering among all known secondary streams
        all_known = sorted(set(self._draws.keys()) | {source_id})
        secondaries = [s for s in all_known if s not in self._primary_sources]
        sec_index = secondaries.index(source_id) if source_id in secondaries else 0
        return _HIGH_RES_STREAM_COUNT + sec_index

    def has_anything_to_draw(self):
        return bool(self._draws) or bool(self._progresses)

    def draw(self):
        self._batch.draw()
        for p in self._progresses.values():
            p.draw()

    def hit_test(self, mx: int, my: int) -> Optional[int]:
        for source_id, draw in self._draws.items():
            if draw.canvas.contains(mx, my):
                return source_id
        return None

    def set_highlight(self, display_id: Optional[int]) -> None:
        if self._highlight_display_id is not None and self._highlight_display_id in self._draws:
            self._draws[self._highlight_display_id].set_highlight(False)
        self._highlight_display_id = display_id
        if display_id is not None and display_id in self._draws:
            self._draws[display_id].set_highlight(True)

    def pop_source(self, source_id: int):
        if self._options[-1].multi_res_layout:
            if source_id == self._highlight_display_id:
                self.set_highlight(None)
        self._draws.pop(source_id, None)
        self._progresses.pop(source_id, None)

    def new_frame(
        self,
        source_id: int,
        image: types.Image,
        axmeta: Optional[meta.AxMeta],
        buf_state: float,
    ):
        cached, meta_map = self._meta_cache.get(source_id, axmeta)
        layers = display.get_layers(self._layers, source_id)
        speedometer_smoothing = (
            self._speedometer_smoothing if self._options[-1].speedometer_smoothing else None
        )

        # Only allow Grid and SideBySide frame styles when there is one source
        if (
            self._options[source_id].style
            in (
                display.FrameStyle.GRID,
                display.FrameStyle.SIDE_BY_SIDE,
            )
            and self._num_sources(source_id) > 1
        ):
            self._options[source_id].style = display.FrameStyle.NORMAL
            LOG.warning(
                f"Stream {source_id}: Grid and SideBySide frame styles are only supported when "
                "there is one source. Defaulting to Normal frame style."
            )

        num_slots = self._num_sources(source_id)
        layout_slot = self._layout_slot(source_id)
        multi_res_layout = self._options[-1].multi_res_layout
        self._draws[source_id] = GLDraw(
            source_id,
            layout_slot,
            num_slots,
            self._window.size,
            self._label_pool,
            self._batch,
            image,
            meta_map,
            self._render_state[source_id],
            self._options[source_id],
            self._options[-1],  # window options is source_id -1
            layers,
            speedometer_smoothing,
            multi_res_layout,
            highlighted=(source_id == self._highlight_display_id),
        )
        if _SHOW_BUFFER_STATUS:
            self.set_buffering(source_id, buf_state)
        else:
            self._progresses.pop(source_id, None)

    def options(self, source_id: int, options: dict[str, Any]) -> None:
        self._options[source_id].update(**options)

    def layer(self, msg: display._Text):
        self._layers[msg.id] = msg

    def set_buffering(self, source_id: int, buf_state: float):
        layout_slot = self._layout_slot(source_id)
        num_slots = self._num_sources(source_id)
        try:
            p = self._progresses[source_id]
            p.resize(num_slots, self._window.size, layout_slot)
        except KeyError:
            p = self._progresses[source_id] = ProgressDraw(
                source_id, layout_slot, num_slots, self._window.size
            )
        p.set_position(buf_state)

    def on_resize(self, width: int, height: int):
        window_size = (width, height)
        for source_id, draw in self._draws.items():
            layout_slot = self._layout_slot(source_id)
            num_slots = self._num_sources(source_id)
            draw.resize(layout_slot, num_slots, window_size)
        for source_id, p in self._progresses.items():
            layout_slot = self._layout_slot(source_id)
            num_slots = self._num_sources(source_id)
            p.resize(num_slots, window_size, layout_slot)

    def new_label_pool(self, label_pool: LabelPool):
        self._label_pool = label_pool
        for draw in self._draws.values():
            draw.new_label_pool(label_pool)


warned_style = False


def _find_frame_class(style: display.FrameStyle):
    global warned_style
    if style == display.FrameStyle.NORMAL:
        return NormalFrame
    elif style == display.FrameStyle.GRID:
        return GridFrame
    elif style == display.FrameStyle.SIDE_BY_SIDE:
        return SideBySideFrame
    if not warned_style:
        LOG.warning(f"Unknown frame style {style}, defaulting to NormalFrame")
        warned_style = True
    return NormalFrame


def get_ptr(data, offset):
    base_address = ctypes.cast(data, ctypes.c_void_p).value
    return base_address + offset


def _upload_nv12(width, height, data, sprite, textures, strides, offsets, row_step=1):
    """Handle NV12 format: Y plane + interleaved UV plane (4:2:0)"""
    display_height = height // row_step
    uv_height = height // (row_step * 2)
    if 'y' not in textures:
        textures['y'] = pyglet.image.Texture.create(width, display_height, GL_TEXTURE_2D, GL_RED)
        textures['uv'] = pyglet.image.Texture.create(width // 2, uv_height, GL_TEXTURE_2D, GL_RG)

        # Y plane: linear filtering for smooth luma
        glBindTexture(GL_TEXTURE_2D, textures['y'].id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # UV plane: nearest-neighbor to avoid color bleeding at sharp transitions
        glBindTexture(GL_TEXTURE_2D, textures['uv'].id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    y_stride = strides[0]
    y_offset = offsets[0]
    uv_stride = strides[1]
    uv_offset = offsets[1]

    glBindTexture(GL_TEXTURE_2D, textures['y'].id)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, y_stride * row_step)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RED,
        width,
        display_height,
        0,
        GL_RED,
        GL_UNSIGNED_BYTE,
        get_ptr(data, y_offset),
    )
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0)

    glBindTexture(GL_TEXTURE_2D, textures['uv'].id)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, (uv_stride // 2) * row_step)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RG,
        width // 2,
        uv_height,
        0,
        GL_RG,
        GL_UNSIGNED_BYTE,
        get_ptr(data, uv_offset),
    )
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0)

    _configure_sprite_uniforms(sprite, 'NV12', width, display_height)


def _bind_nv12(textures):
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, textures['y'].id)
    glActiveTexture(GL_TEXTURE2)
    glBindTexture(GL_TEXTURE_2D, textures['uv'].id)


def _upload_i420(width, height, data, sprite, textures, strides, offsets, row_step=1):
    """Handle I420 format: Y plane + separate U and V planes (4:2:0)"""
    display_height = height // row_step
    uv_height = height // (row_step * 2)
    if 'y' not in textures:
        textures['y'] = pyglet.image.Texture.create(width, display_height, GL_TEXTURE_2D, GL_RED)
        textures['u'] = pyglet.image.Texture.create(width // 2, uv_height, GL_TEXTURE_2D, GL_RED)
        textures['v'] = pyglet.image.Texture.create(width // 2, uv_height, GL_TEXTURE_2D, GL_RED)

        # Y plane: linear filtering for smooth luma
        glBindTexture(GL_TEXTURE_2D, textures['y'].id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # U and V planes: nearest-neighbor to avoid color bleeding at sharp transitions
        for tex in [textures['u'], textures['v']]:
            glBindTexture(GL_TEXTURE_2D, tex.id)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
            glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    y_stride = strides[0]
    y_offset = offsets[0]
    u_stride = strides[1]
    u_offset = offsets[1]
    v_stride = strides[2]
    v_offset = offsets[2]

    glBindTexture(GL_TEXTURE_2D, textures['y'].id)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, y_stride * row_step)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RED,
        width,
        display_height,
        0,
        GL_RED,
        GL_UNSIGNED_BYTE,
        get_ptr(data, y_offset),
    )
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0)

    glBindTexture(GL_TEXTURE_2D, textures['u'].id)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, u_stride * row_step)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RED,
        width // 2,
        uv_height,
        0,
        GL_RED,
        GL_UNSIGNED_BYTE,
        get_ptr(data, u_offset),
    )
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0)

    glBindTexture(GL_TEXTURE_2D, textures['v'].id)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, v_stride * row_step)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RED,
        width // 2,
        uv_height,
        0,
        GL_RED,
        GL_UNSIGNED_BYTE,
        get_ptr(data, v_offset),
    )
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0)

    _configure_sprite_uniforms(sprite, 'I420', width, display_height)


def _bind_i420(textures):
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, textures['y'].id)
    glActiveTexture(GL_TEXTURE2)
    glBindTexture(GL_TEXTURE_2D, textures['u'].id)
    glActiveTexture(GL_TEXTURE3)
    glBindTexture(GL_TEXTURE_2D, textures['v'].id)


def _upload_nv16(width, height, data, sprite, textures, strides, offsets, row_step=1):
    """Handle NV16 format: Y plane + interleaved UV plane (4:2:2)"""
    display_height = height // row_step
    if 'y' not in textures:
        textures['y'] = pyglet.image.Texture.create(width, display_height, GL_TEXTURE_2D, GL_RED)
        textures['uv'] = pyglet.image.Texture.create(
            width // 2, display_height, GL_TEXTURE_2D, GL_RG
        )

        # Y plane: linear filtering for smooth luma
        glBindTexture(GL_TEXTURE_2D, textures['y'].id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)

        # UV plane: nearest-neighbor to avoid color bleeding at sharp transitions
        glBindTexture(GL_TEXTURE_2D, textures['uv'].id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

    y_stride = strides[0]
    y_offset = offsets[0]
    uv_stride = strides[1]
    uv_offset = offsets[1]

    glBindTexture(GL_TEXTURE_2D, textures['y'].id)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, y_stride * row_step)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RED,
        width,
        display_height,
        0,
        GL_RED,
        GL_UNSIGNED_BYTE,
        get_ptr(data, y_offset),
    )
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0)

    glBindTexture(GL_TEXTURE_2D, textures['uv'].id)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, (uv_stride // 2) * row_step)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RG,
        width // 2,
        display_height,
        0,
        GL_RG,
        GL_UNSIGNED_BYTE,
        get_ptr(data, uv_offset),
    )
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0)

    _configure_sprite_uniforms(sprite, 'NV16', width, display_height)


def _bind_nv16(textures):
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, textures['y'].id)
    glActiveTexture(GL_TEXTURE2)
    glBindTexture(GL_TEXTURE_2D, textures['uv'].id)


def _upload_yuy2(width, height, data, sprite, textures, strides, offsets, row_step=1):
    """Handle YUY2 format: Packed 4:2:2 (Y0 U Y1 V pattern)"""
    display_height = height // row_step
    if 'tex' not in textures:
        textures['tex'] = pyglet.image.Texture.create(
            width // 2, display_height, GL_TEXTURE_2D, GL_RGBA
        )

        # Use nearest-neighbor filtering to avoid interpolation artifacts
        glBindTexture(GL_TEXTURE_2D, textures['tex'].id)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)

        print(f"Created YUY2 texture: {width//2}x{display_height} (RGBA)")

    y_stride = strides[0]
    y_offset = offsets[0]

    glBindTexture(GL_TEXTURE_2D, textures['tex'].id)
    glPixelStorei(
        GL_UNPACK_ROW_LENGTH, (y_stride // 4) * row_step
    )  # RGBA texels are 4 bytes each; YUY2 stride is 2 bytes/pixel

    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        width // 2,
        display_height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        get_ptr(data, y_offset),
    )
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0)

    _configure_sprite_uniforms(sprite, 'YUY2', width, display_height)


def _bind_yuy2(textures):
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, textures['tex'].id)


def _upload_gray8(width, height, data, sprite, textures, strides, offsets, row_step=1):
    """Handle GRAY8 format: single channel grayscale"""
    display_height = height // row_step
    if 'tex' not in textures:
        textures['tex'] = pyglet.image.Texture.create(width, display_height, GL_TEXTURE_2D, GL_RED)
        print(f"Created GRAY8 texture: {width}x{display_height}")

    y_stride = strides[0]
    y_offset = offsets[0]

    glBindTexture(GL_TEXTURE_2D, textures['tex'].id)
    glPixelStorei(GL_UNPACK_ROW_LENGTH, y_stride * row_step)
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RED,
        width,
        display_height,
        0,
        GL_RED,
        GL_UNSIGNED_BYTE,
        get_ptr(data, y_offset),
    )
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0)

    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, textures['tex'].id)

    _configure_sprite_uniforms(sprite, 'GRAY', width, display_height)


def _bind_gray8(textures):
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, textures['tex'].id)


def _upload_rgb(width, height, data, sprite, textures, strides, offsets, row_step=1):
    """Handle RGB/BGR format"""
    display_height = height // row_step
    if 'tex' not in textures:
        textures['tex'] = pyglet.image.Texture.create(width, display_height, GL_TEXTURE_2D, GL_RGB)

    stride = strides[0]
    offset = offsets[0]
    glBindTexture(GL_TEXTURE_2D, textures['tex'].id)
    glPixelStorei(
        GL_UNPACK_ROW_LENGTH, (stride // 3) * row_step
    )  # Each pixel is 3 bytes in RGB/BGR
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGB,
        width,
        display_height,
        0,
        GL_RGB,
        GL_UNSIGNED_BYTE,
        get_ptr(data, offset),
    )
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0)

    _configure_sprite_uniforms(sprite, 'RGB', width, display_height)


def _bind_rgb(textures):
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, textures['tex'].id)


def _upload_rgba(width, height, data, sprite, textures, strides, offsets, row_step=1):
    """Handle RGBA/BGRA/RGBx/BGRx format"""
    display_height = height // row_step
    if 'tex' not in textures:
        textures['tex'] = pyglet.image.Texture.create(
            width, display_height, GL_TEXTURE_2D, GL_RGBA
        )

    glBindTexture(GL_TEXTURE_2D, textures['tex'].id)
    glPixelStorei(
        GL_UNPACK_ROW_LENGTH, (strides[0] // 4) * row_step
    )  # Each pixel is 4 bytes in RGBA/BGRA/RGBx/BGRx
    glTexImage2D(
        GL_TEXTURE_2D,
        0,
        GL_RGBA,
        width,
        display_height,
        0,
        GL_RGBA,
        GL_UNSIGNED_BYTE,
        get_ptr(data, offsets[0]),
    )
    glPixelStorei(GL_UNPACK_ROW_LENGTH, 0)

    _configure_sprite_uniforms(sprite, 'RGBA', width, display_height)


def _bind_rgba(textures):
    glActiveTexture(GL_TEXTURE1)
    glBindTexture(GL_TEXTURE_2D, textures['tex'].id)


def _configure_sprite_uniforms(sprite, format_name: str, width: int, height: int):
    """Configure shader uniforms for a sprite based on the color format.

    This is needed for grid cell sprites and other sprites that share textures
    but have their own shader program instances.
    """
    format_key = format_name.upper()

    if format_key in ['NV12', 'NV16']:
        sprite.program['y_tex'] = 1
        sprite.program['uv_tex'] = 2
    elif format_key == 'I420':
        sprite.program['y_tex'] = 1
        sprite.program['u_tex'] = 2
        sprite.program['v_tex'] = 3
    elif format_key == 'YUY2':
        sprite.program['tex'] = 1
        sprite.program['tex_size'] = (float(width), float(height))
    elif format_key in ['GRAY', 'RGB', 'BGR', 'RGBA', 'BGRA', 'RGBX', 'BGRX']:
        sprite.program['tex'] = 1
    # Fallback formats also use 'tex'
    else:
        sprite.program['tex'] = 1


def _new_sprite_from_image(
    image,
    texture,
    canvas: GLCanvas,
    batch,
    bind_group,
    grayscale,
    grayscale_area,
    opacity,
    render_state=None,
    row_step=1,
):
    pt = canvas.glp((0, 0))

    format_name = image.color_format.name
    width, height = image.size

    shader = _get_shader(
        format_name, blur=None, grayscale=1.0 if grayscale else None, grayscale_area=grayscale_area
    )
    sprite = pyglet.sprite.Sprite(texture, *pt, batch=batch, group=bind_group, program=shader)
    sprite.scale_x = canvas.scale
    sprite.scale_y = -canvas.scale
    sprite.opacity = opacity

    textures = render_state.get("_textures", {})
    with image.as_c_void_p() as data:
        if format_name == 'NV12':
            _upload_nv12(
                width, height, data, sprite, textures, image.strides, image.offsets, row_step
            )
        elif format_name == 'I420':
            _upload_i420(
                width, height, data, sprite, textures, image.strides, image.offsets, row_step
            )
        elif format_name == 'NV16':
            _upload_nv16(
                width, height, data, sprite, textures, image.strides, image.offsets, row_step
            )
        elif format_name == 'YUY2':
            _upload_yuy2(
                width, height, data, sprite, textures, image.strides, image.offsets, row_step
            )
        elif format_name == 'GRAY':
            _upload_gray8(
                width, height, data, sprite, textures, image.strides, image.offsets, row_step
            )
        elif format_name in ['RGB', 'BGR']:
            _upload_rgb(
                width, height, data, sprite, textures, image.strides, image.offsets, row_step
            )
        elif format_name in ['RGBA', 'BGRA', 'RGBx', 'BGRx']:
            _upload_rgba(
                width, height, data, sprite, textures, image.strides, image.offsets, row_step
            )

    if "_textures" not in render_state:
        render_state["_textures"] = textures

    return sprite


class BindGroup(pyglet.graphics.Group):
    def __init__(self, render_state, format_name, order=0, parent=None):
        super().__init__(order, parent)
        self._render_state = render_state
        self.format_name = format_name

    def set_state(self):
        textures = self._render_state.get('_textures', {})
        if self.format_name == 'NV12':
            _bind_nv12(textures)
        elif self.format_name == 'I420':
            _bind_i420(textures)
        elif self.format_name == 'NV16':
            _bind_nv16(textures)
        elif self.format_name == 'YUY2':
            _bind_yuy2(textures)
        elif self.format_name == 'GRAY8':
            _bind_gray8(textures)
        elif self.format_name in ['RGB', 'BGR']:
            _bind_rgb(textures)
        elif self.format_name in ['RGBA', 'BGRA', 'RGBx', 'BGRx']:
            _bind_rgba(textures)
        glActiveTexture(GL_TEXTURE0)


class Frame:
    def __init__(self, image, canvas, batch, group, render_state, bind_group):
        self._image = image
        self._canvas = canvas
        self._batch = batch
        self._group = group
        self._render_state = render_state
        self._bind_group = bind_group

    def resize(self, canvas):
        self._canvas = canvas

    @property
    def canvas(self) -> GLCanvas:
        return self._canvas

    @property
    def layer_canvas(self) -> GLCanvas:
        return self.canvas

    @property
    def _main_texture(self):
        """Get or create the main texture from render_state."""
        if "_main_texture" not in self._render_state:
            width, height = self._image.size
            self._render_state["_main_texture"] = pyglet.image.Texture.create(
                width, height, GL_TEXTURE_2D, GL_RGBA
            )
        return self._render_state["_main_texture"]


class NormalFrame(Frame):
    def __init__(
        self,
        image,
        canvas,
        batch,
        group,
        foreground,
        shapes,
        render_state,
        bind_group,
        grayscale=False,
        grayscale_area='all',
        opacity=255,
        row_step=1,
        **style_opts,
    ):
        del style_opts, foreground, shapes
        super().__init__(image, canvas, batch, group, render_state, bind_group)
        self._row_step = row_step
        self._sprite = _new_sprite_from_image(
            self._image,
            self._main_texture,
            self._canvas,
            self._batch,
            self._bind_group,
            grayscale,
            grayscale_area,
            opacity,
            self._render_state,
            row_step,
        )

    def resize(self, canvas):
        super().resize(canvas)
        _move_sprite(self._sprite, self._canvas)

    def delete(self) -> None:
        self._sprite.delete()


class GridFrame(Frame):
    PIXELATE_FACTOR = 10.0

    def __init__(
        self,
        texture,
        canvas,
        batch,
        group,
        foreground,
        shapes,
        render_state,
        bind_group,
        crops={},
        grid_dims=(8, 8),
        blurs=set(),
        fades=dict(),
        left=0,
        source_sprite=None,
        **style_opts,
    ):
        del style_opts, foreground, shapes
        super().__init__(
            texture,
            GLCanvas(left, 0, *canvas.window_size, 1.0, *canvas.window_size),
            batch,
            group,
            render_state,
            bind_group,
        )

        if source_sprite is None:
            source_sprite = _new_sprite_from_image(
                texture,
                self._main_texture,
                self._canvas,
                None,  # No batch - invisible sprite just for texture upload
                self._bind_group,
                False,
                'all',
                255,
                self._render_state,
            )
        self._source_sprite = source_sprite

        self._rows, self._cols = grid_dims

        self._batch = batch
        self._group = group

        if blurs == 'all':
            self._blurs = set(range(self._rows * self._cols))
        else:
            self._blurs = blurs

        if "grid_sprites" not in self._render_state:
            self._render_state['grid_sprites'] = {}
            self._render_state['last_dims'] = grid_dims
            self._render_state['last_blurs'] = self._blurs
            self._render_state['last_left'] = left
        self._sprites = self._render_state['grid_sprites']

        self._cell_width = self._canvas.width // self._cols
        self._cell_height = self._canvas.height // self._rows

        new_blurs = self._blurs - self._render_state.get('last_blurs', set())
        manual_blurs = new_blurs - set(crops.keys())  # sprites not being overwritten automatically

        for cell_id in manual_blurs:
            sprite = self._sprites.get(cell_id)
            if sprite is not None:
                format_name = self._image.color_format.name
                sprite.program = _get_shader(
                    format_name, blur=self.PIXELATE_FACTOR, grayscale=None
                )

        self._render_state['last_blurs'] = self._blurs

        # Set opacity for fading sprites, and delete fully faded ones (opacity=0)
        for cell_id, opacity in fades.items():
            if opacity == 0:
                if sprite := self._sprites.pop(cell_id, None):
                    sprite.delete()
            elif sprite := self._sprites.get(cell_id):
                sprite.opacity = opacity

        for cell_id, crop_rect in crops.items():
            self._create_cell_sprite(cell_id, crop_rect)
        if self._render_state['last_dims'] != grid_dims or self._render_state['last_left'] != left:
            # Move existing sprites if dims have changed
            cell_count = self._rows * self._cols
            deleted_cells = [cell_id for cell_id in self._sprites if cell_id >= cell_count]
            for cell_id in deleted_cells:
                sprite = self._sprites.pop(cell_id)
                sprite.delete()
            for cell_id, sprite in self._sprites.items():
                pt, scale_x, scale_y = self._get_pt_and_scale(
                    cell_id, sprite.image.width, sprite.image.height
                )

                sprite.x = pt[0]
                sprite.y = pt[1]

                sprite.scale_x = scale_x
                sprite.scale_y = scale_y
            self._render_state['last_dims'] = grid_dims
            self._render_state['last_left'] = left

    def _get_pt_and_scale(self, cell_id, w, h):
        cell_row = cell_id // self._cols
        cell_col = cell_id % self._cols
        pt = self._canvas.glp(
            (
                (cell_col / self._cols) * self._canvas.width,
                (cell_row / self._rows) * self._canvas.height,
            )
        )
        scale_x = self._cell_width / w
        scale_y = -(self._cell_height / h)
        return pt, scale_x, scale_y

    def _create_cell_sprite(self, cell_id: int, crop_rect: tuple[int, int, int, int]):
        x0, y0, x1, y1 = crop_rect
        w, h = x1 - x0, y1 - y0

        pt, scale_x, scale_y = self._get_pt_and_scale(cell_id, w, h)

        cell_texture_region = self._main_texture.get_region(x0, y0, w, h)

        format_name = self._image.color_format.name
        img_width, img_height = self._image.size

        shader = _get_shader(
            format_name,
            blur=self.PIXELATE_FACTOR if cell_id in self._blurs else None,
            grayscale=None,
        )

        sprite = pyglet.sprite.Sprite(
            cell_texture_region, *pt, batch=self._batch, group=self._bind_group, program=shader
        )
        sprite.scale_x = scale_x
        sprite.scale_y = scale_y

        # Configure shader uniforms to point to the correct texture units
        _configure_sprite_uniforms(sprite, format_name, img_width, img_height)

        self._sprites[cell_id] = sprite

    def resize(self, canvas):
        pass


class SideBySideFrame(Frame):
    LINE_COLOR = (255, 255, 255, 255)
    HIGHLIGHTED_LINE_COLOR = (0, 255, 0, 255)
    LINE_WIDTH = _RENDER_LINE_WIDTH
    HIGHLIGHTED_LINE_WIDTH = _RENDER_LINE_WIDTH * 2

    def __init__(
        self,
        texture,
        canvas,
        batch,
        group,
        foreground,
        shapes,
        render_state,
        bind_group,
        **style_opts,
    ):
        super().__init__(texture, canvas, batch, group, render_state, bind_group)
        lines = style_opts.pop("lines", None)
        highlighted_lines = style_opts.pop("highlighted_lines", None)
        split_pct = style_opts.pop("split_pct", 0.5)

        img_w = self._canvas.width / self._canvas.scale
        img_h = self._canvas.height / self._canvas.scale
        self._main_frame = NormalFrame(
            texture,
            _create_canvas(
                0,
                1,
                (img_w, img_h),
                (int(self._canvas.window_width * split_pct), self._canvas.window_height),
            ),
            batch,
            group,
            foreground,
            shapes,
            render_state,
            bind_group,
            **style_opts,
        )

        self._grid_frame = GridFrame(
            texture,
            GLCanvas(
                left=None,
                bottom=None,
                width=None,
                height=None,
                scale=None,
                window_width=int(self._canvas.window_width * (1 - split_pct)),
                window_height=self._canvas.window_height,
            ),  # This is a mock canvas - only window_width/height are important
            batch,
            group,
            foreground,
            shapes,
            render_state,
            bind_group,
            left=int(self._canvas.window_width * split_pct),
            source_sprite=self._main_frame._sprite,
            **style_opts,
        )

        # for cell_id, real_point in lines.items, draw line from midpoint of cell to real_point
        if lines is not None:
            parsed_lines = [(cell_id, real_point, False) for cell_id, real_point in lines.items()]
            if highlighted_lines is not None:
                parsed_lines += [
                    (cell_id, real_point, True)
                    for cell_id, real_point in highlighted_lines.items()
                ]
            for cell_id, real_point, highlighted in parsed_lines:
                cell_row = cell_id // self._grid_frame._cols
                cell_col = cell_id % self._grid_frame._cols
                cell_x0 = (
                    (cell_col / self._grid_frame._cols)
                    * self._canvas.window_width
                    * (1 - split_pct)
                )
                cell_y0 = (cell_row / self._grid_frame._rows) * self._canvas.window_height
                cell_x1 = (
                    ((cell_col + 1) / self._grid_frame._cols)
                    * self._canvas.window_width
                    * (1 - split_pct)
                )
                cell_y1 = ((cell_row + 1) / self._grid_frame._rows) * self._canvas.window_height
                mid_x = (cell_x0 + cell_x1) / 2
                mid_y = (cell_y0 + cell_y1) / 2

                p1 = self._grid_frame.canvas.glp((mid_x, mid_y))
                p2 = self._main_frame.canvas.glp(real_point)
                shapes.append(
                    pyglet.shapes.Line(
                        p1[0],
                        p1[1],
                        p2[0],
                        p2[1],
                        self.HIGHLIGHTED_LINE_WIDTH if highlighted else self.LINE_WIDTH,
                        color=self.HIGHLIGHTED_LINE_COLOR if highlighted else self.LINE_COLOR,
                        batch=batch,
                        group=foreground,
                    )
                )

    def resize(self, canvas):
        pass

    @property
    def canvas(self) -> GLCanvas:
        return self._main_frame.canvas

    @property
    def layer_canvas(self) -> GLCanvas:
        return self._grid_frame.canvas


class GLDraw(display.Draw):
    def __init__(
        self,
        source_id: int,
        layout_slot: int,
        num_slots: int,
        window_size: tuple[int, int],
        label_pool: LabelPool,
        batch: pyglet.graphics.Batch,
        image: types.Image,
        meta_map: Mapping[str, meta.AxTaskMeta],
        render_state: dict[Any, Any],
        options: GLOptions,
        window_options: GLOptions,
        layers: list[display._Layer],
        speedometer_smoothing: display.SpeedometerSmoothing = None,
        multi_res_layout: bool = False,
        highlighted: bool = False,
    ):
        self._source_id = source_id
        self._window_size = window_size
        self._label_pool = label_pool
        self._batch = batch
        self._shapes = []
        self._back = pyglet.graphics.Group(GR_BACK_OFFSET + self._source_id)
        self._fore = pyglet.graphics.Group(GR_FORE_OFFSET + self._source_id)
        self._speedo0 = pyglet.graphics.Group(GR_SPEEDO_OFFSET + 0)
        self._speedo1 = pyglet.graphics.Group(GR_SPEEDO_OFFSET + 1)
        self._layer_gr = pyglet.graphics.Group(GR_LAYER_OFFSET + self._source_id)
        self._highlighted = highlighted
        self._multi_res_layout = multi_res_layout
        if self._multi_res_layout:
            _is_secondary = layout_slot >= _HIGH_RES_STREAM_COUNT
            self._show_labels = not _is_secondary
            self._scale_down = _SECONDARY_STREAM_SCALE if _is_secondary else 1.0
        else:
            self._show_labels = True
            self._scale_down = 1.0
        self._speedometer_index = 0
        self._meta_map = meta_map
        self._render_state = render_state
        self._speedometer_smoothing = speedometer_smoothing
        self._options = options
        self._image_size = image.size

        format_name = image.color_format.name
        bind_group = BindGroup(
            self._render_state,
            format_name,
            self._back.order if hasattr(self._back, 'order') else 0,
            self._back,
        )

        row_step = max(1, round(1.0 / self._scale_down)) if self._scale_down < 1.0 else 1
        self._frame = _find_frame_class(self._options.style)(
            image,
            _create_canvas(
                layout_slot,
                num_slots,
                self._image_size,
                window_size,
                multi_res_layout=self._multi_res_layout,
            ),
            self._batch,
            self._back,
            self._fore,
            self._shapes,
            self._render_state,
            bind_group,
            grayscale=options.grayscale,
            grayscale_area=options.grayscale_area,
            row_step=row_step,
            **self._options.style_opts,
        )
        self._canvas = self._frame.canvas

        if self._options.show_metadata:
            self._render_meta()

        if self._multi_res_layout:
            stream_id_options = GLOptions(
                title=str(self._source_id),
                title_size=12,
                title_color=(255, 255, 255, 255),
                title_bgcolor=(0, 0, 0, 160),
            )
            layers.append(display.gen_title_message(self._source_id, stream_id_options))
        elif options.title:
            layers.append(display.gen_title_message(self._source_id, options))
        if window_options.title and self._source_id == 0:
            layers.append(display.gen_title_message(-1, window_options))

        layer_canvas = self._frame.layer_canvas

        for x in layers:
            if x.stream_id == -1:
                pt_transform = lambda pt: (pt[0], window_size[1] - pt[1])
                canvas_size = window_size
            else:
                pt_transform = layer_canvas.glp
                canvas_size = self._image_size
            opacity = int(255 * x.visibility)  # TODO: 255 should be x.opacity once added
            if isinstance(x, display._Text):
                self._text(
                    pt_transform(x.position.as_px(canvas_size)),
                    x.text,
                    self._layer_gr,
                    x.color,
                    x.bgcolor,
                    display.Font(size=x.font_size),
                    x.anchor_x,
                    x.anchor_y,
                    opacity,
                )
            elif isinstance(x, display._Image):
                s = _load_sprite_from_file(
                    x.path, x.scale, canvas_size, self._batch, self._layer_gr
                )
                s.x, s.y = pt_transform(x.position.as_px(canvas_size))
                s.opacity = opacity
                if x.anchor_x == 'center':
                    s.x -= s.width / 2
                elif x.anchor_x == 'right':
                    s.x -= s.width
                if x.anchor_y == 'center':
                    s.y -= s.height / 2
                elif x.anchor_y == 'top':
                    s.y -= s.height
                self._shapes.append(s)
            elif isinstance(x, display._Rectangle):
                p1 = pt_transform(x.position.as_px(canvas_size))
                p2 = pt_transform(x.bottom_right.as_px(canvas_size))
                self._rectangle(
                    p1,
                    p2,
                    self._layer_gr,
                    outline=x.color,
                    width=1,
                )
            else:
                LOG.debug(f"Unknown layer type {x.__class__.__name__} ignoring...")

        if self._highlighted:
            img_w, img_h = self._image_size
            self._rectangle(
                self._canvas.glp((0, 0)),
                self._canvas.glp((img_w, img_h)),
                self._fore,
                outline=(255, 200, 0, 255),
            )

    @property
    def options(self) -> GLOptions:
        return self._options

    @property
    def canvas(self) -> GLCanvas:
        return self._canvas

    def _render_meta(self):
        if self._meta_map:
            for m in self._meta_map.values():
                m.visit(lambda m: m.draw(self))

    def delete(self) -> None:
        for shape in self._shapes:
            if not isinstance(shape, _SpriteProxy):
                shape.delete()
        self._shapes.clear()
        self._frame.delete()

    def set_highlight(self, active: bool) -> None:
        self._highlighted = active

    @property
    def is_highlighted(self) -> bool:
        return self._highlighted

    def resize(self, layout_slot: int, num_slots: int, window_size: Tuple[int, int]):
        self._window_size = window_size
        self._canvas = _create_canvas(
            layout_slot,
            num_slots,
            self._image_size,
            window_size,
            multi_res_layout=self._multi_res_layout,
        )
        self._frame.resize(self._canvas)
        self._render_meta()

    def new_label_pool(self, label_pool: LabelPool):
        self._label_pool = label_pool
        self._render_meta()

    @property
    def canvas_size(self) -> display.Point:
        return self._canvas.size

    @property
    def image_size(self) -> display.Point:
        '''Return the original, unscaled size of the input image'''
        return self._image_size

    def polylines(
        self,
        lines: Sequence[Sequence[display.Point]],
        closed: bool = False,
        color: display.Color = (255, 255, 255, 255),
        width: int = 0,
    ) -> None:
        # though width is given here, it is ignored because glLineWidth is not portable, and
        # since the lines are drawn with Screen width of 1 this is usually sufficient.
        # the reason that width is provided is that in CV drawing the line is given in image pixels
        del width
        converted = [[self._canvas.glp(p) for p in pts] for pts in lines]
        for pts in converted:
            self._shapes.append(
                pyglet.shapes.MultiLine(
                    *pts, closed=closed, color=color, batch=self._batch, group=self._fore
                )
            )

    def _rectangle(self, p1, p2, group, fill=None, outline=None, width=1):
        x0, y0 = p1
        x1, y1 = p2
        w, h = x1 - x0, y1 - y0
        if fill and outline:
            kwargs = dict(border=width, color=fill, border_color=outline)
            cls = pyglet.shapes.BorderedRectangle
        elif fill:
            kwargs = dict(color=fill)
            cls = pyglet.shapes.Rectangle
        elif outline:
            kwargs = dict(color=outline)
            cls = Box
        self._shapes.append(cls(x0, y0, w, h, **kwargs, batch=self._batch, group=group))

    def rectangle(self, p1, p2, fill=None, outline=None, width=1):
        self._rectangle(
            self._canvas.glp(p1),
            self._canvas.glp(p2),
            self._fore,
            fill=fill,
            outline=outline,
            width=width,
        )

    def keypoint(
        self, p: display.Point, color: display.Color = (255, 255, 255, 255), size=0.003
    ) -> None:
        size = round(max(self.canvas_size) / 2 * size)
        x, y = self._canvas.glp(p)
        if _BOX_KEYPOINTS:
            point = pyglet.shapes.Rectangle(
                x - size,
                y - size,
                size * 2,
                size * 2,
                color=color,
                batch=self._batch,
                group=self._fore,
            )
        else:
            point = pyglet.shapes.Circle(
                x=x, y=y, radius=size, color=color, batch=self._batch, group=self._fore
            )
        self._shapes.append(point)

    def textsize(self, text, font=display.Font()):
        w, h = _textsize(text, *_determine_font_params(font))
        return w / self._canvas.scale, h / self._canvas.scale

    def text(
        self,
        p,
        text,
        txt_color,
        back_color: display.OptionalColor = None,
        font=display.Font(),
    ):
        if not self._show_labels:
            return
        self._text(self._canvas.glp(p), text, self._fore, txt_color, back_color, font)

    def _text(
        self,
        p,
        text,
        group,
        txt_color,
        back_color: display.OptionalColor = None,
        font=display.Font(),
        anchor_x='left',
        anchor_y='top',
        opacity=255,
    ):
        txt_color = _add_alpha(txt_color)
        back_color = _add_alpha(back_color)
        name, size = _determine_font_params(font)
        self._shapes.append(
            self._label_pool.create_label(
                text,
                name,
                size,
                weight=font.weight,
                italic=font.italic,
                color=txt_color,
                back_color=back_color,
                x=p[0],
                y=p[1],
                anchor_x=anchor_x,
                anchor_y=anchor_y,
                batch=self._batch,
                group=group,
                opacity=opacity,
            )
        )

    def draw_speedometer(self, metric: inf_tracers.TraceMetric):
        if self._speedometer_smoothing:
            self._speedometer_smoothing.update(metric)
        text = display.calculate_speedometer_text(metric, self._speedometer_smoothing)
        needle_pos = display.calculate_speedometer_needle_pos(metric, self._speedometer_smoothing)
        m = display.SpeedometerMetrics(self._window_size, self._speedometer_index)

        image = _get_speedometer()
        top_left = (m.top_left[0], self._window_size[1] - m.bottom_left[1])
        sprite = pyglet.sprite.Sprite(image, *top_left, batch=self._batch, group=self._speedo0)
        sprite.anchor_y = -image.height
        sprite.scale = m.diameter / image.width
        self._shapes.append(sprite)

        C = (m.center[0], self._window_size[1] - m.center[1])
        x1 = round(C[0] + math.cos(_to_radians(needle_pos)) * m.needle_radius)
        y1 = round(C[1] - math.sin(_to_radians(needle_pos)) * m.needle_radius)
        self._shapes.append(
            pyglet.shapes.Line(
                *C,
                x1,
                y1,
                thickness=3,
                color=m.needle_color,
                batch=self._batch,
                group=self._speedo1,
            )
        )
        self._shapes.append(
            pyglet.text.Label(
                text,
                x=C[0],
                y=C[1] - m.text_offset,
                anchor_x='center',
                anchor_y='center',
                color=m.text_color,
                batch=self._batch,
                group=self._speedo1,
                font_size=m.text_size * 0.7,
            )
        )
        self._shapes.append(
            pyglet.text.Label(
                metric.title,
                x=C[0],
                y=C[1] - m.text_offset * 1.75,
                anchor_x='center',
                anchor_y='center',
                color=m.text_color,
                batch=self._batch,
                group=self._speedo1,
                font_size=m.text_size * 0.5,
            )
        )
        self._speedometer_index += 1

    def draw(self):
        pass  # Drawing is handled by the batch

    def heatmap(self, data: np.ndarray, color_map: np.ndarray) -> None:
        indices = np.clip((data * len(color_map) - 1).astype(int), 0, len(color_map) - 1)
        rgba_mask = color_map[indices]
        image = pyglet.image.ImageData(data.shape[1], data.shape[0], 'RGBA', rgba_mask.tobytes())
        sprite = pyglet.sprite.Sprite(
            image, *self._canvas.glp((0, 0)), batch=self._batch, group=self._fore
        )
        sprite.anchor_y = -image.height
        sprite.scale_x = self._canvas.scale
        sprite.scale_y = -self._canvas.scale
        self._shapes.append(sprite)

    def segmentation_mask(self, mask_data, color: Tuple[int]) -> None:
        mask, mbox = mask_data[-1], mask_data[4:8]
        mid_point = np.iinfo(np.uint8).max // 2
        bool_array = mask > mid_point

        colored_mask = np.zeros((*bool_array.shape, 4), dtype=np.uint8)
        colored_mask[bool_array] = color
        buf = colored_mask.ctypes.data_as(ctypes.c_void_p)

        img_size = (mbox[2] - mbox[0], mbox[3] - mbox[1])
        image = pyglet.image.ImageData(mask.shape[1], mask.shape[0], 'RGBA', buf)
        scale_x = img_size[0] / mask.shape[1] * self._canvas.scale
        scale_y = img_size[1] / mask.shape[0] * self._canvas.scale
        sprite = pyglet.sprite.Sprite(
            image, *self._canvas.glp(mbox[:2]), batch=self._batch, group=self._fore
        )

        sprite.anchor_y = -image.height
        sprite.scale_x = scale_x
        sprite.scale_y = -scale_y
        self._shapes.append(sprite)

    def class_map_mask(self, class_map: np.ndarray, color_map: np.ndarray) -> None:
        colored_mask = color_map[class_map]
        image = pyglet.image.ImageData(
            class_map.shape[1], class_map.shape[0], 'RGBA', colored_mask.tobytes()
        )
        sprite = pyglet.sprite.Sprite(
            image, *self._canvas.glp((0, 0)), batch=self._batch, group=self._fore
        )
        sprite.anchor_y = -image.height
        sprite.scale_x = self._canvas.width / image.width
        sprite.scale_y = -(self._canvas.height / image.height)
        self._shapes.append(sprite)

    def draw_image(self, image: np.ndarray) -> None:
        if image.dtype not in [np.uint8, np.float32, np.float64]:
            raise ValueError("draw_image: image dtype must be np.uint8, np.float32, or np.float64")

        if image.ndim == 4 and image.shape[0] == 1:
            image = image.squeeze(axis=0)
        if image.ndim == 3 and image.shape[0] == 1:
            image = image.squeeze(axis=0)

        if image.dtype in (np.float32, np.float64):
            d_min, d_max = image.min(), image.max()
            if np.isclose(d_min, d_max):
                image = np.zeros_like(image, dtype=np.uint8)
            else:
                image = np.clip((image - d_min) / (d_max - d_min) * 255.0, 0, 255).astype(np.uint8)

        if image.ndim == 3 and image.shape[0] == 3:
            image = np.transpose(image, (1, 2, 0))

        if image.ndim == 2:
            rgba = np.stack([image, image, image, np.full_like(image, 255)], axis=-1)
        elif image.ndim == 3 and image.shape[2] == 3:
            alpha = np.full((*image.shape[:2], 1), 255, dtype=np.uint8)
            rgba = np.concatenate([image, alpha], axis=-1)
        elif image.ndim == 3 and image.shape[2] == 4:
            rgba = image
        else:
            return

        h, w = rgba.shape[:2]
        rgba_contiguous = np.ascontiguousarray(rgba)
        gl_image = pyglet.image.ImageData(w, h, 'RGBA', rgba_contiguous.tobytes())
        sprite = pyglet.sprite.Sprite(
            gl_image, *self._canvas.glp((0, 0)), batch=self._batch, group=self._fore
        )
        sprite.anchor_y = -gl_image.height
        sprite.scale_x = self._canvas.width / gl_image.width
        sprite.scale_y = -(self._canvas.height / gl_image.height)
        self._shapes.append(sprite)


@functools.lru_cache
def _get_speedometer():
    here = os.path.dirname(__file__)
    return pyglet.image.load(f'{here}/render_assets/speedo-alpha-transparent.png')


_to_radians = functools.partial(operator.mul, math.pi / 180)


class ProgressBar:
    def __init__(self, x, y, w, h, color, back_color):
        self._width = w
        b = self._border = 2
        assert h > self._border * 2, "Border is too small"
        g0 = pyglet.graphics.Group(GR_PROG_OFFSET + 0)
        g1 = pyglet.graphics.Group(GR_PROG_OFFSET + 1)
        self._outer = pyglet.shapes.BorderedRectangle(
            x, y, w, h, border=1, color=back_color, border_color=color, group=g0
        )
        self._outer.anchor_position = (w // 2, h // 2)
        self._inner = pyglet.shapes.Rectangle(
            x + b, y + b, w - b * 2, h - b * 2, color=color, group=g1
        )
        self._inner.anchor_position = (w // 2 - b, h // 2)
        self.set_position(0.0)

    def move(self, x, y):
        self._outer.position = x, y
        self._inner.position = x + 2, y + 2

    def set_position(self, value):
        value = min(1.0, max(0.0, value))
        self._inner.width = int((self._width - self._border * 3) * value)

    def draw(self):
        self._outer.draw()
        self._inner.draw()


class HighLowQueue(collections.deque):
    def __init__(self, *, maxlen):
        super().__init__(maxlen=maxlen)
        self.low = maxlen // 3
        self.high = max(1, 2 * self.low)
        self.low_water_reached = False


def noexcept(f):
    '''Decorator to catch all exceptions and log them, suitable for event handlers'''

    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            LOG.report_recoverable_exception(e)

    return wrapper


@dataclass
class GLOptions(display.Options):
    multi_res_layout: bool = False
    '''Enable multi-stream display layout with primary/secondary panel split,
    stream ID labels, and interactive stream swapping.
    '''

    grayscale_area: str = 'all'
    '''When grayscale is enabled this specifies the area to grayscale.

    One of 'all' for the entire image, or 'left', 'right', 'top', 'bottom' for
    the respective half of the image. This is useful when tiling only part of
    the screen for example.
    '''

    show_metadata: bool = True
    '''Global toggle whether to show any metadata at all on the surface.'''

    style: display.FrameStyle = display.FrameStyle.NORMAL
    '''The frame style to use when rendering the image.'''

    style_opts: dict = field(default_factory=dict)
    '''Additional style options passed to the frame style.'''


class GLWindow(pyglet.window.Window):
    def __init__(
        self, q: queue.Queue, title, size, buffering, frame_sink, hard_stop=True, borderless=False
    ):
        self._master = None
        self._gles = False
        self._frame_sink = frame_sink
        self._hard_stop = hard_stop
        w, h = (None, None) if size == display.FULL_SCREEN else size

        _display = pyglet.display.get_display()
        screen = _display.get_default_screen()
        gl_config = screen.get_best_config()
        gl_config.opengl_api = _GL_API
        gl_config.major_version = _GL_MAJOR
        gl_config.minor_version = _GL_MINOR
        if gl_config.opengl_api == "gles":
            self._gles = True

        super().__init__(
            w,
            h,
            caption=title,
            fullscreen=size == display.FULL_SCREEN,
            resizable=True,
            config=gl_config,
            visible=bool(title),
            style='default' if not borderless else 'borderless',
        )
        w = w or self.width
        h = h or self.height
        icons = [pyglet.image.load(i) for i in display.ICONS.values()]
        icons[-1].anchor_x = icons[-1].width // 2
        icons[-1].anchor_y = icons[-1].height // 2
        self._start_time = time.time()
        self._logo = pyglet.sprite.Sprite(icons[-1], 0, 0)
        self._progress = ProgressBar(
            w // 2, h // 2 - icons[-1].height, 200, 20, (255, 255, 255, 255), (0, 0, 0, 255)
        )
        self.set_icon(*icons)
        self._queue = q
        self._old_pixel_ratio = self.get_pixel_ratio()
        self._pool = LabelPool(self._old_pixel_ratio)
        self._master = MasterDraw(self, self._pool)
        self._stream_queues: dict[int, HighLowQueue] = {}
        # during initial logo spin have redraws at 30fps. Once we are going drop to 10
        pyglet.clock.schedule_interval(self._redraw, 1 / 30)
        pyglet.clock.schedule_interval(self.on_update, 1 / _RENDER_FPS)
        self._fps_counter = pyglet.window.FPSDisplay(self)
        self.buffering = buffering
        self._closed_sources = set()
        self._pending_blocking_capture = False
        self._primary_sources: list[int] = list(range(_HIGH_RES_STREAM_COUNT))
        self._selected_primary: Optional[int] = None

    @property
    def _multi_res_layout(self) -> bool:
        return self._master._options[-1].multi_res_layout if self._master else False

    def _do_swap(self, secondary_source_id: int, primary_source_id: int) -> None:
        if secondary_source_id == primary_source_id:
            LOG.debug(f"Source {secondary_source_id} is already in the primary slot")
            return
        primary_slot = self._primary_sources.index(primary_source_id)
        self._primary_sources[primary_slot] = secondary_source_id
        self._master.set_primary_sources(self._primary_sources)
        self._master.pop_source(secondary_source_id)
        self._master.pop_source(primary_source_id)
        LOG.debug(
            f"Swapped source {secondary_source_id} into primary slot {primary_slot},"
            f" source {primary_source_id} becomes secondary"
        )

    @noexcept
    def on_mouse_press(self, x, y, button, modifiers):
        if not self._multi_res_layout:
            return
        if button != pyglet.window.mouse.LEFT or not self._master:
            return
        hit = self._master.hit_test(x, y)
        if hit is None:
            self._selected_primary = None
            self._master.set_highlight(None)
            return
        if hit in self._primary_sources:
            if self._selected_primary == hit:
                self._selected_primary = None
                self._master.set_highlight(None)
            else:
                self._selected_primary = hit
                self._master.set_highlight(hit)
        else:
            if self._selected_primary is not None:
                self._do_swap(hit, self._selected_primary)
                self._selected_primary = None

    def on_key_press(self, symbol, modifiers):
        del modifiers
        if symbol in (pyglet.window.key.Q, pyglet.window.key.ESCAPE, pyglet.window.key.SPACE):
            # Just calling pyglet.app.exit() stops the pyglet event loop but doesn't destroy
            # anything. Allowing execution to be resumed.
            if self._hard_stop:
                pyglet.app.platform_event_loop.post_event(self, "on_close")
            else:
                pyglet.app.exit()

    @noexcept
    def on_update(self, dt):
        del dt

        with catchtime('update', LOG.trace):
            try:
                while True:
                    msg = self._queue.get(block=False)
                    if msg is display.SHUTDOWN:
                        self._queue.clear()
                        pyglet.app.platform_event_loop.post_event(self, "on_close")
                        return
                    if msg is display.THREAD_COMPLETED:
                        continue  # ignore, just wait for user to close
                    if isinstance(msg, display._OpenSource):
                        self._closed_sources.discard(msg.stream_id)
                        continue
                    blocking = isinstance(msg, display._BlockingFrame)
                    if (
                        isinstance(msg, display._StreamMessage)
                        and msg.stream_id in self._closed_sources
                    ):
                        if blocking:
                            LOG.error(
                                f"Received blocking frame from closed source {msg.stream_id}"
                            )
                        continue  # ignore messages from closed sources
                    if isinstance(msg, display._CloseSource):
                        if not msg.reopen:
                            self._closed_sources.add(msg.stream_id)
                        self._stream_queues.pop(msg.stream_id, None)
                        self._master.pop_source(msg.stream_id)
                        self._master.clear_state(msg.stream_id)
                    elif isinstance(msg, display._SetOptions):
                        self._master.options(msg.stream_id, msg.options)
                    elif isinstance(msg, display._Layer):
                        self._master.layer(msg)
                    elif isinstance(msg, display._ClearState):
                        self._master.clear_state(msg.stream_id)
                    elif isinstance(msg, display._Frame):
                        pyglet.clock.unschedule(self._redraw)
                        try:
                            q = self._stream_queues[msg.stream_id]
                        except KeyError:
                            maxlen = (
                                2
                                if msg.stream_id in _LOW_LATENCY_STREAMS or not self.buffering
                                else _STREAM_QUEUE_SIZE
                            )
                            q = self._stream_queues[msg.stream_id] = HighLowQueue(maxlen=maxlen)
                        q.append((msg.image, msg.meta, blocking))
                    else:
                        LOG.debug(f"Unexpected render message {msg}")
            except queue.Empty:
                pass

            self.invalid = False
            for source_id, q in self._stream_queues.items():
                self.invalid = True
                if q.low_water_reached or len(q) > q.low:
                    q.low_water_reached = True
                    if len(q) > q.high:
                        # we're falling behind, drop an extra frame
                        image, axmeta, blocking_flag = q.popleft()
                        if blocking_flag:
                            # never drop a blocking frame
                            self._pending_blocking_capture = True
                            self._master.new_frame(source_id, image, axmeta, len(q) / q.maxlen)
                    if len(q):
                        image, axmeta, blocking_flag = q.popleft()
                        if blocking_flag:
                            self._pending_blocking_capture = True
                        self._master.new_frame(source_id, image, axmeta, len(q) / q.maxlen)
                else:
                    # still buffering, don't pop anything but do redraw progress
                    self._master.set_buffering(source_id, len(q) / q.maxlen)

            self._redraw()

    def _redraw(self, dt=None):
        self.dispatch_event('on_draw')
        self.flip()

    def on_resize(self, width: int, height: int):
        # on a resize we need to redo all the scale calculations
        if self._master:
            self._master.on_resize(width, height)
        return super().on_resize(width, height)

    def on_move(self, x, y):
        new_pixel_ratio = self.get_pixel_ratio()
        if self._old_pixel_ratio != new_pixel_ratio:
            # If HiDPI setting has changed in some way then dump the label xfipool
            self._old_pixel_ratio = new_pixel_ratio
            self._pool = LabelPool(new_pixel_ratio)
            self._master.new_label_pool(self._pool)

    @noexcept
    def on_draw(self):
        self.clear()
        if _RENDER_LINE_WIDTH > 1 and not self._gles:
            pyglet.gl.glLineWidth(_RENDER_LINE_WIDTH)
        if self._master.has_anything_to_draw():
            if not self._gles:
                pyglet.gl.glEnable(pyglet.gl.GL_LINE_SMOOTH)
            with catchtime('draw', LOG.trace):
                self._master.draw()
            self.invalid = False
        else:
            self._show_logo()
        if _SHOW_RENDER_FPS:
            self._fps_counter.draw()
        if self._frame_sink:
            color_buf = pyglet.image.get_buffer_manager().get_color_buffer()
            img_data = color_buf.get_image_data()
            row_stride = self.width * 4
            raw = img_data.get_data('RGBA', row_stride)
            arr = np.frombuffer(raw, dtype=np.uint8).reshape(self.height, self.width, 4)
            arr = np.flipud(arr)
            rgb = arr[:, :, :3]
            try:
                self._frame_sink.push(
                    types.Image.fromarray(rgb), block=self._pending_blocking_capture
                )
            except RuntimeError as e:
                LOG.error(f"Blocking frame sink not updated: {e}")
            finally:
                self._pending_blocking_capture = False

    def _show_logo(self):
        # silly bit of code to make the logo pulsate during startup whilst we warm up pipelines
        self._logo.position = self.width // 2, self.height // 2, 0.0
        elapsed = time.time() - self._start_time
        if elapsed < 1.0:
            self._logo.scale = 1.5 * math.sin(math.pi * elapsed)
        else:
            elapsed -= 1.0
            startup_time = 10.0
            # pulsate the logo whilst we show a progress bar counting to arbitrary 10s
            self._logo.opacity = int(180 + 75 * math.sin(math.pi * elapsed * 1.1))
            if elapsed < startup_time:
                self._progress.set_position(elapsed / startup_time)
                self._progress.draw()
        self._logo.draw()


class GLApp(display.App):
    SupportedOptions = GLOptions

    def __init__(self, *args, **kwargs):
        self.buffering = kwargs.pop('buffering', True)
        super().__init__(*args, **kwargs)

    def _idle(self, dt):
        del dt
        self._create_new_windows()
        if self.has_thread_completed:
            pyglet.app.exit()

    def _create_new_window(self, q, frame_sink, title, size):
        return GLWindow(q, title, size, self.buffering, frame_sink)

    def _run_background(self, interval=1 / 30):
        del interval
        if self._running_in_main:
            return
        raise RuntimeError(
            "Implicit OpenGL rendering in the background is not supported. Either: "
            "1. Start the renderer in your application with `display.App.run()` or "
            "2. Use OpenCV rendering with `display.App(renderer='opencv')`"
        )

    def _run(self, interval=1 / 60):
        pyglet.clock.schedule_interval(self._idle, 0.3)
        pyglet.app.run(interval=None if not sys.platform == 'darwin' else 1 / 10)

    def _destroy_all_windows(self):
        pyglet.app.exit()

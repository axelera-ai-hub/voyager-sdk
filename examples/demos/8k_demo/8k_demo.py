#!/usr/bin/env python
# Copyright Axelera AI, 2025

import os
import time

from axelera.app import config, display, inf_tracers
from axelera.app.stream import create_inference_stream

NETWORK = 'yolov8l-coco-onnx'
TITLE = "Axelera 8K Demo YOLOV8L Tiling"
TITLE_FONT = 78
UNTILED_GRAYSCALE = 0.7  # 0.0 = no grayscale, 1.0 = full grayscale

LOGO1 = os.path.join(config.env.framework, "axelera/app/render_assets/voyager-sdk-logo-white.png")
LOGO2 = os.path.join(config.env.framework, "axelera/app/render_assets/axelera-ai-logo.png")


def main(window, stream, args):
    if args.tile_position != 'none':
        opposites = {'left': 'right', 'right': 'left', 'top': 'bottom', 'bottom': 'top'}
        grayscale_area = opposites.get(args.tile_position, 'all')
        grayscale = UNTILED_GRAYSCALE
        if args.tile_position in ('left', 'right'):
            anchors = {'anchor_x': 'center', 'anchor_y': 'bottom'}
            labels = ['25%, 10%', '75%, 10%']
        elif args.tile_position in ('top', 'bottom'):
            anchors = {'anchor_x': 'right', 'anchor_y': 'center'}
            labels = ['98%, 25%', '98%, 75%']
        if args.tile_position in ('right', 'bottom'):
            labels.reverse()
    else:
        grayscale, grayscale_area, labels = 0.0, 'all', []

    window.options(
        -1,
        title=TITLE,
        title_position=('50%', '0%'),
        title_size=TITLE_FONT,
        title_anchor_x='center',
    )

    window.options(
        0,
        grayscale=grayscale,
        grayscale_area=grayscale_area,
        bbox_label_format="{label} {scorep:.0f}%",
    )
    if labels:
        window.text(labels[0], "Tiling enabled", **anchors)
        window.text(labels[1], "No tiling", **anchors)

    logo1 = window.image('98%, 98%', LOGO1, anchor_x='right', anchor_y='bottom', scale=0.45)
    logo2 = window.image(
        '98%, 98%', LOGO2, anchor_x='right', anchor_y='bottom', scale=0.4, fadeout_from=0.0
    )
    start = time.time()
    period = 10.0

    for frame_result in stream:
        now = time.time()
        if (now - start) > period:
            logo1.hide(now, 1.0)
            logo2.show(now, 1.0)
            logo1, logo2 = logo2, logo1
            start = now

        window.show(frame_result.image, frame_result.meta, frame_result.stream_id)

        if window.is_closed:
            break


if __name__ == '__main__':
    parser = config.create_inference_argparser(
        default_network=NETWORK, description='Perform inference on an Axelera platform'
    )
    args = parser.parse_args()
    tracers = inf_tracers.create_tracers('core_temp', 'end_to_end_fps', 'end_to_end_infs')
    stream = create_inference_stream(
        config.SystemConfig.from_parsed_args(args),
        config.InferenceStreamConfig.from_parsed_args(args),
        config.PipelineConfig.from_parsed_args(args),
        config.LoggingConfig.from_parsed_args(args),
        tracers=tracers,
        # low_latency=True,
    )

    with display.App(
        renderer=args.display,
        opengl=stream.hardware_caps.opengl,
        buffering=not stream.is_single_image(),
    ) as app:
        wnd = app.create_window(TITLE, args.window_size)
        app.start_thread(main, (wnd, stream, args), name='InferenceThread')
        app.run(interval=1 / 10)
    stream.stop()

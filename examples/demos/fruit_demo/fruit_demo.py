#!/usr/bin/env python
# Copyright Axelera AI, 2025
#
# This example shows how to run a complex pipeline involving 3 different models
#
# │ └─master_detections   - yolov8lpose-coco-onnx
# │ └─    │ segmentations - yolov8sseg-coco-onnx
#   └─object_detections   - yolov8s-fruit (this is yolov8s-coco-onnx with a filter for fruit)
#
# The purpose of the example is to show how using the ROI of the master_detections model makes
# detection of sub object (fruit in this case) more effective.
#
# The example also shows some other functionality, for example:
#   * how to configure rendering options tuned for this application
#   * how to use the inference.py argparser with a default network
#
import os
import time

from axelera.app import config, display, inf_tracers, logging_utils
from axelera.app.stream import create_inference_stream

NETWORK = 'fruit-demo'

LOGO1 = os.path.join(config.env.framework, "axelera/app/render_assets/voyager-sdk-logo-white.png")
LOGO2 = os.path.join(config.env.framework, "axelera/app/render_assets/axelera-ai-logo.png")


def main(window, stream):
    for n, source in stream.sources.items():
        window.options(
            n,
            title=f"Video {n} {source}" if len(stream.sources) > 1 else "Fruit Demo",
            title_size=24,
            grayscale=0.0,  # 0.9 works well for gray scale
            bbox_class_colors={
                'banana': (0, 0, 255, 125),
                'apple': (0, 255, 0, 125),
                'orange': (255, 0, 0, 125),
            },
        )
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
    # Reuse the inference.py command line parameters
    parser = config.create_inference_argparser(
        default_network=NETWORK, description='Perform inference on an Axelera platform'
    )
    args = parser.parse_args()
    tracers = inf_tracers.create_tracers(
        'latency', 'end_to_end_fps', 'end_to_end_infs', pipe_type='gst'
    )
    stream = create_inference_stream(
        network=NETWORK,
        sources=args.sources,
        pipe_type='gst',
        log_level=logging_utils.get_config_from_args(args).console_level,
        hardware_caps=config.HardwareCaps.from_parsed_args(args),
        tracers=tracers,
        rtsp_latency=args.rtsp_latency,
        allow_hardware_codec=args.enable_hardware_codec,
        specified_frame_rate=args.frame_rate,
        aipu_cores=args.aipu_cores,
        device_selector=args.devices,
        low_latency=True,
    )

    with display.App(
        renderer=args.display,
        opengl=stream.hardware_caps.opengl,
        buffering=not stream.is_single_image(),
    ) as app:
        wnd = app.create_window("Fruit Demo", args.window_size)
        app.start_thread(main, (wnd, stream), name='InferenceThread')
        app.run()
    stream.stop()

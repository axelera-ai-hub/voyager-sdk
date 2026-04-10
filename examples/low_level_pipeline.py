#!/usr/bin/env python
# Copyright Axelera AI, 2025
# Show how to pass a low level yaml file to the framework, and modify it
# to use a specific device.
import argparse
import os
from pathlib import Path
import re
import sys

from axelera.runtime import Context

if __name__ == '__main__':
    # Application Framework is not a package, so add it to the path to import it
    sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from axelera.app import config, display, inf_tracers, logging_utils
from axelera.app.stream import create_inference_stream


def first_device_name() -> str:
    with Context() as ctx:
        return ctx.list_devices()[0].name


def main(window, stream):
    for frame_result in stream:
        window.show(frame_result.image, frame_result.meta, frame_result.stream_id)

        if window.is_closed:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Low-level pipeline demo")
    config.HardwareCaps.add_to_argparser(parser)
    config.add_display_arguments(parser)
    args = parser.parse_args()
    hwcaps = config.HardwareCaps.from_parsed_args(args)
    framework = config.env.framework
    examples_dir = os.path.dirname(__file__)
    low_level = Path(f'{examples_dir}/low-level-fruit-demo.yaml').read_text()
    low_level = re.sub(r'^(\s+devices:\s+).*$', f'\\1{first_device_name()}', low_level, flags=re.M)

    stream = create_inference_stream(
        log_level=logging_utils.TRACE,  # INFO, DEBUG, TRACE
        hardware_caps=hwcaps,
        network='fruit-demo',
        sources=['fakevideo'],
        ax_precompiled_gst=low_level,
    )

    with display.App(renderer=args.display, opengl=stream.hardware_caps.opengl) as app:
        wnd = app.create_window("Low level pipeline demo", args.window_size)
        app.start_thread(main, (wnd, stream), name='InferenceThread')
        app.run()
    stream.stop()

#!/usr/bin/env python
# Copyright Axelera AI, 2025

# gi.repository.Gst is requred to set mainloop configs
import gi

gi.require_version('Gst', '1.0')

from gi.repository import Gst
from tqdm import tqdm

from axelera.app import config, display, inf_tracers, logging_utils, pipe, statistics, yaml_parser
from axelera.app.stream import InferenceStream

LOG = logging_utils.getLogger(__name__)
PBAR = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"


def init(args, tracers):
    hardware_caps = config.HardwareCaps.from_parsed_args(args)
    pipemanager = pipe.PipeManager(
        args.network,
        args.sources,
        args.pipe,
        ax_precompiled_gst=args.ax_precompiled_gst,
        num_cal_images=args.num_cal_images,
        batch=args.calibration_batch,
        data_root=args.data_root,
        build_root=args.build_root,
        save_output=args.save_output,
        frames=args.frames,
        hardware_caps=hardware_caps,
        allow_hardware_codec=args.enable_hardware_codec,
        tracers=tracers,
        metis=args.metis,
        rtsp_latency=args.rtsp_latency,
        device_selector=args.devices,
        specified_frame_rate=args.frame_rate,
    )
    return InferenceStream(pipemanager, frames=args.frames, timeout=args.timeout)


def inference_loop(args, log_file_path, stream, app, wnd, tracers=None):
    if len(stream.manager.sources) > 1:
        for sid, source in stream.manager.sources.items():
            wnd.options(sid, title=f"#{sid} - {source}")

    for frame_result in tqdm(
        stream,
        desc=f"Detecting... {' ':>30}",
        unit='frames',
        leave=False,
        bar_format=PBAR,
        disable=None,
    ):
        image, meta = frame_result.image, frame_result.meta
        if image is None and meta is None:
            if wnd.is_closed:
                break
            continue

        if image:
            wnd.show(image, meta, frame_result.stream_id)

        if wnd.is_closed:
            break

    if stream.is_single_image() and args.display:
        LOG.debug("stream has a single frame, close the window or press Q to exit...")
        wnd.wait_for_close()

    if log_file_path:
        print(statistics.format_table(log_file_path, tracers))
    else:
        for tracer in tracers:
            for m in tracer.get_metrics():
                LOG.info(f"{m.title} : {m.value:.1f}{m.unit}")


if __name__ == "__main__":
    parser = config.create_inference_argparser(
        description='Perform inference on an Axelera platform'
    )
    args = parser.parse_args()
    logging_utils.configure_logging(logging_utils.get_config_from_args(args))
    logging_utils.configure_compiler_level(args)

    tracers = inf_tracers.create_tracers_from_args(args)
    try:
        log_file, log_file_path = None, None
        if args.show_stats:
            log_file, log_file_path = statistics.initialise_logging()
        stream = init(args, tracers)

        with display.App(
            visible=args.display,
            opengl=stream.manager.hardware_caps.opengl,
            buffering=not stream.is_single_image(),
        ) as app:
            wnd = app.create_window('Inference demo', size=args.window_size)
            app.start_thread(
                inference_loop,
                (args, log_file_path, stream, app, wnd, tracers),
                name='InferenceThread',
            )
            app.run(interval=1 / 10)
    except KeyboardInterrupt:
        LOG.exit_with_error_log()
    except logging_utils.UserError as e:
        LOG.exit_with_error_log(e.format())
    except Exception as e:
        LOG.exit_with_error_log(e)
    finally:
        if 'stream' in locals():
            stream.stop()

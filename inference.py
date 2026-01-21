#!/usr/bin/env python
# Copyright Axelera AI, 2025

import os
import sys
import time

if not os.environ.get('AXELERA_FRAMEWORK'):
    sys.exit("Please activate the Axelera environment with source venv/bin/activate and run again")

from tqdm import tqdm

from axelera.app import (
    config,
    create_inference_stream,
    display,
    inf_tracers,
    logging_utils,
    statistics,
    yaml_parser,
)

try:
    import gi

    gi.require_version('Gst', '1.0')
    from gi.repository import Gst
except ImportError:
    pass

LOG = logging_utils.getLogger(__name__)
PBAR = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

LOGO1 = os.path.join(config.env.framework, "axelera/app/voyager-sdk-logo-white.png")
LOGO2 = os.path.join(config.env.framework, "axelera/app/axelera-ai-logo.png")
LOGO_POS = '95%, 95%'


def inference_loop(args, log_file_path, stream, app, wnd, tracers=None):
    if len(stream.sources) > 1:
        for sid, source in stream.sources.items():
            wnd.options(sid, title=f"#{sid} - {source}")

    wnd.options(-1, speedometer_smoothing=args.speedometer_smoothing)
    logo1 = wnd.image(LOGO_POS, LOGO1, anchor_x='right', anchor_y='bottom', scale=0.3)
    logo2 = wnd.image(
        LOGO_POS, LOGO2, anchor_x='right', anchor_y='bottom', scale=0.3, fadeout_from=0.0
    )
    supported = logo1 and logo2
    logo_start = save_start = time.time()
    logo_period = 10.0

    savef, save_period = None, 10.0  # don't save until we have enough data to be meaningful

    for event in tqdm(
        stream.with_events(),
        desc=f"Detecting... {' ':>30}",
        unit='frames',
        leave=False,
        bar_format=PBAR,
        disable=None,
    ):
        if not event.result:
            LOG.warning(f"Unknown event received: {event!r}")
            continue
        frame_result = event.result

        now = time.time()
        if supported and ((now - logo_start) > logo_period):
            logo1.hide(now, 1.0)
            logo2.show(now, 1.0)
            logo1, logo2 = logo2, logo1
            logo_start = now

        if args.save_tracers and ((now - save_start) > save_period):
            metrics = sorted(stream.get_all_metrics().items())
            if not savef:
                path = args.save_tracers.lstrip('+')
                append = args.save_tracers.startswith('+')
                savef = open(path, 'a' if append else 'w')
                if not append:
                    savef.write("timestamp," + ",".join(m[0] for m in metrics) + "\n")
                LOG.info(f"Saving tracer data to {path}")
            save_start = now
            save_period = 1.0
            savef.write(f"{now:.3f},{','.join('%.1f' % (m[1].value,) for m in metrics)}\n")

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
    inf_tracers.display_tracers(tracers)
    if savef:
        savef.close()
        LOG.info(f"Tracer data saved to {savef.name}")


if __name__ == "__main__":
    network_yaml_info = yaml_parser.get_network_yaml_info()
    parser = config.create_inference_argparser(
        network_yaml_info, description='Perform inference on an Axelera platform'
    )
    parser.add_argument(
        '--save-tracers',
        type=str,
        default=None,
        help="Save tracer data to a file as CSV, prefix with `+` to append to an existing file",
    )
    args = parser.parse_args()
    # early exit if the network is a LLM
    if network_yaml_info.has_llm(args.network):
        raise ValueError("inference.py currently supports vision models only")

    tracers = inf_tracers.create_tracers_from_args(args)
    try:
        log_file, log_file_path = None, None
        if args.show_stats:
            log_file, log_file_path = statistics.initialise_logging()
        stream = create_inference_stream(
            config.SystemConfig.from_parsed_args(args),
            config.InferenceStreamConfig.from_parsed_args(args),
            config.PipelineConfig.from_parsed_args(args),
            config.LoggingConfig.from_parsed_args(args),
            config.DeployConfig.from_parsed_args(args),
            tracers=tracers,
        )

        with display.App(
            renderer=args.display,
            opengl=stream.hardware_caps.opengl,
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

    if Gst.is_initialized():
        Gst.deinit()

#!/usr/bin/env python
# Copyright Axelera AI, 2025
# Benchmark chip-level and end-to-end performance for a single model

import inference

from axelera.app import config, display, inf_tracers, logging_utils, statistics, yaml_parser

LOG = logging_utils.getLogger(__name__)


def _unsupported_yaml_condition(info):
    if isinstance(info, yaml_parser.NetworkYamlBase):
        return info.cascaded
    else:
        raise ValueError("info must be an instance of NetworkYamlBase")


if __name__ == "__main__":
    network_yaml_info = yaml_parser.get_network_yaml_info()

    default_caps = config.HardwareCaps.AIPU

    parser = config.create_inference_argparser(
        network_yaml_info,
        default_caps,
        default_display=False,
        default_show_stats=True,
        unsupported_yaml_cond=_unsupported_yaml_condition,
        unsupported_reason='cascaded models are not supported in benchmark mode',
        description='Benchmark a single model on an Axelera platform',
    )

    # TODO: support profiling various opt modes on and off to see which yields the best performance
    # parser.add_argument('--exploratory',
    #                     action='store_true',
    #                     help='Run in exploratory mode')
    args = parser.parse_args()
    logging_utils.configure_logging(logging_utils.get_config_from_args(args))
    logging_utils.configure_compiler_level(args)

    tracers = inf_tracers.create_tracers_from_args(args)

    try:
        log_file, log_file_path = None, None
        if args.show_stats:
            log_file, log_file_path = statistics.initialise_logging()
        stream = inference.init(args, tracers)
        with display.App(visible=args.display, opengl=stream.manager.hardware_caps.opengl) as app:
            wnd = app.create_window('Benchmark demo', size=(900, 600))
            app.start_thread(
                inference.inference_loop,
                (args, log_file_path, stream, app, wnd, tracers),
                name='BenchmarkThread',
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

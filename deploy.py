#!/usr/bin/env python
# Copyright Axelera AI, 2024
# Deployment tool (compilation flow)

import sys

from axelera.app import config, logging_utils, pipeline, utils, yaml_parser

LOG = logging_utils.getLogger(__name__)


def main():
    network_yaml_info = yaml_parser.get_network_yaml_info()
    parser = config.create_deploy_argparser(network_yaml_info)
    args = parser.parse_args()
    logging_utils.configure_logging(logging_utils.get_config_from_args(args))
    logging_utils.configure_compiler_level(args)
    hardware_caps = config.HardwareCaps.from_parsed_args(args)
    nn_info = network_yaml_info.get_info(args.network)
    nn_name = nn_info.yaml_name

    deploy_info = f'{nn_name}: {args.model}' if args.model else nn_name
    verb = (
        'Quantizing'
        if args.mode in (config.DeployMode.QUANTIZE, config.DeployMode.QUANTIZE_DEBUG)
        else 'Compiling'
    )
    with utils.catchtime(f"{verb} {deploy_info}", LOG.info):
        success = pipeline.deploy_from_yaml(
            nn_name,
            args.network,
            args.pipeline_only,
            args.models_only,
            args.model,
            args.pipe,
            args.mode,
            args.num_cal_images,
            args.calibration_batch,
            args.data_root,
            args.build_root,
            args.export,
            hardware_caps,
            args.emulate,
            args.metis,
        )
    if success:
        if args.mode not in (config.DeployMode.QUANTIZE, config.DeployMode.QUANTIZE_DEBUG):
            LOG.info("Successfully deployed network")
        sys.exit(0)

    LOG.error("Failed to deploy network")
    sys.exit(1)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        LOG.exit_with_error_log()
    except logging_utils.UserError as e:
        LOG.exit_with_error_log(e.format())
    except Exception as e:
        LOG.exit_with_error_log(e)

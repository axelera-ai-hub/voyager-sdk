#!/usr/bin/env python
# Copyright Axelera AI, 2022
# Deployment tool (compilation flow)

import sys

from axelera.app import config, logging_utils, pipeline, torch_utils, utils, yaml_parser

LOG = logging_utils.getLogger(__name__)


def main():
    network_yaml_info = yaml_parser.get_network_yaml_info()
    parser = config.create_deploy_argparser(network_yaml_info)
    args = parser.parse_args()
    logging_utils.configure_logging(logging_utils.get_config_from_args(args))
    logging_utils.configure_compiler_level(args)
    nn_info = network_yaml_info.get_info(args.network)
    nn_name = nn_info.yaml_name
    if args.cal_seed is not None:
        torch_utils.set_random_seed(args.cal_seed)

    deploy_info = f'{nn_name}: {args.model}' if args.model else nn_name
    verb = (
        'Quantizing'
        if args.mode in (config.DeployMode.QUANTIZE, config.DeployMode.QUANTIZE_DEBUG)
        else 'Compiling'
    )
    system_config = config.SystemConfig.from_parsed_args(args)
    pipeline_config = config.PipelineConfig(
        network=args.network,
        pipe_type=args.pipe,
        aipu_cores=args.aipu_cores,
    )
    deploy_config = config.DeployConfig.from_parsed_args(args)
    with utils.catchtime(f"{verb} {deploy_info}", LOG.info):
        success = pipeline.deploy_from_yaml(
            nn_name,
            args.pipeline_only,
            args.models_only,
            args.model,
            system_config,
            pipeline_config,
            deploy_config,
            args.mode,
            args.export,
            args.board,
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

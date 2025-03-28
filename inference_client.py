#!/usr/bin/env python
# Copyright Axelera AI, 2023
# Inference POC (interpreter flow)

import argparse
import logging
from pathlib import Path
import sys
import time

import grpc

from axelera.app import display, logging_utils, utils
from axelera.app.config import HardwareEnable
from axelera.app.inference_client import remote_stream
import axelera.app.inference_pb2_grpc as inference_pb_grpc

LOG = logging_utils.getLogger(__name__)


def _is_single_image(source):
    return Path(source).is_file() and utils.get_media_type(source) == 'image'


def inference_loop(args, cb, stream, wnd):
    try:
        frames = 0
        for image, meta in stream:
            if image is not None:
                cb(image, meta, 0)
            frames += 1

        if _is_single_image(args.input):
            LOG.debug("stream has a single frame, close the window or press Q to exit...")
            while not wnd.is_closed:
                time.sleep(0.1)

    except Exception as e:
        LOG.error(f'Unable to connect to server: {repr(e)}')
        raise e


def decode_location(input):
    return ' (decode on server)' if input.startswith('rtsp:') else ' (decode on client)'


def run(args):
    input = args.input
    network = args.network
    with grpc.insecure_channel(args.server) as channel:
        LOG.info('Attempting to connect to server')
        try:
            stub = inference_pb_grpc.InferenceStub(channel)
            with display.App(visible=True, opengl=HardwareEnable.detect, buffering=False) as app:
                wnd = app.create_window('Inference demo' + decode_location(input), size=(960, 540))
                stream = remote_stream(stub, input, network)
                app.start_thread(
                    inference_loop,
                    (args, wnd.show, stream, wnd),
                    name='InferenceThread',
                )
                app.run(interval=1 / 10)
        except Exception as e:
            LOG.error(f'Unable to connect to server: {e.details()}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Inference client')
    parser.add_argument(
        'server', type=str, default='', help='Ip address:port of the server to connect to'
    )
    parser.add_argument('network', type=str, default='', help='The network to run')
    parser.add_argument('input', type=str, default='', help='Source to run inference on')

    args = parser.parse_args()

    logging.basicConfig()
    run(args)
    sys.exit(0)

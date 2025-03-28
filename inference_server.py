#!/usr/bin/env python
# Copyright Axelera AI, 2024
# Inference server POC

import argparse
from concurrent import futures
import logging
import os
import pathlib
import signal
import sys
from typing import TYPE_CHECKING, Any, Tuple

import cv2
import grpc
import numpy as np
from tqdm import tqdm

from axelera import types
from axelera.app import config, inf_tracers, logging_utils, pipe, statistics, utils, yaml_parser
from axelera.app.config import HardwareCaps, add_nn_and_network_arguments
import axelera.app.inference_pb2 as pb
import axelera.app.inference_pb2_grpc as inference_pb2_grpc
from axelera.app.meta import ObjectDetectionMeta
from axelera.app.stream import InferenceStream

if TYPE_CHECKING:
    from axelera import types

LOG = logging_utils.getLogger(__name__)
PBAR = "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"


def resolve_network(
    args: argparse.Namespace,
    network: str,
    network_yaml_info: yaml_parser.NetworkYamlInfo,
) -> Tuple[str, pathlib.Path]:
    try:
        yaml_info = network_yaml_info.get_info(network)
    except KeyError as e:
        raise ValueError(f"Network {network} not found in yaml files")

    nn_dir = pathlib.Path(f"{args.build_root}/{yaml_info.name}")
    return yaml_info.yaml_path, nn_dir


def inference_loop(args, return_image, stream, interrupt_handler, server, wnd, tracers=None):
    log_file, log_file_path = None, None
    if args.show_stats:
        log_file, log_file_path = statistics.initialise_logging()

    interrupt_handler.stream = stream
    face_wnd = None
    with utils.catchtime("Inference") as taken:
        pbar = tqdm(
            stream,
            desc="Detecting... {:>30}".format(""),
            unit='frames',
            leave=False,
            bar_format=PBAR,
        )
        frames = 0
        for frame_result in pbar:
            if frames == 1:
                taken.reset()
            frames += 1
            if frame_result is None:
                break

            image, meta = frame_result.image, frame_result.meta
            if image is None and meta is None:
                break

            img = image.asarray()
            for m in meta.values():
                inf_image = pb.Image(
                    width=img.shape[1],
                    height=img.shape[0],
                    channels=img.shape[2],
                    image=img.tobytes(),
                    color_format=image.color_format.name,
                )
                if isinstance(m, ObjectDetectionMeta):
                    result = pb.ObjectMeta(
                        boxes=m.boxes.tobytes(),
                        scores=m.scores.tobytes(),
                        classes=m.class_ids.tobytes(),
                    )
                    if return_image:
                        yield pb.Inferenceresult(obj=result, image=inf_image)
                    else:
                        yield pb.Inferenceresult(obj=result)

    if log_file_path:
        print(statistics.format_table(log_file_path, tracers))
    else:
        for tracer in tracers:
            for m in tracer.get_metrics():
                LOG.info(f"{m.title} : {m.value:.1f}{m.unit}")

    if stream._interrupt_raised:
        server.stop(None)


class Inference(inference_pb2_grpc.InferenceServicer):
    def __init__(self, ns, network_yaml_info, server, *args: Any, **kwds: Any) -> Any:
        self.ns = ns
        self.network_yaml_info = network_yaml_info
        self.server = server
        self.pipe_mgr = None
        self.network = None
        return super().__init__(*args, **kwds)

    def init(self, args, num_frames, interrupt_handler, tracers, request_iter):
        try:
            hardware_caps = config.HardwareCaps.from_parsed_args(args)
            if not self.pipe_mgr:
                self.pipe_mgr = pipe.PipeManager(
                    args.network,
                    ["server"],
                    args.pipe,
                    ax_precompiled_gst=args.ax_precompiled_gst,
                    num_cal_images=args.num_cal_images,
                    batch=args.calibration_batch,
                    data_root=args.data_root,
                    build_root=args.build_root,
                    save_output=args.save_output,
                    frames=num_frames,
                    hardware_caps=hardware_caps,
                    tracers=tracers,
                    server_loader=InferenceServerLoader(args.sources[0]),
                    metis=args.metis,
                    rtsp_latency=args.rtsp_latency,
                )
            self.pipe_mgr.server_loader.set_source(request_iter)

            stream = InferenceStream(self.pipe_mgr, frames=args.frames, timeout=args.timeout)
            interrupt_handler.stream = stream
            return stream

        except Exception as e:
            LOG.error(f'Error initializing pipe manager: {str(e)}')

    def Infer(self, request, context):
        self.ns.network, self.ns.output_dir = resolve_network(
            self.ns, request.network, self.network_yaml_info
        )
        if self.pipes[0] != self.ns.network:
            num_frames = 1000
            pipe_mgr = self.init(self.ns, num_frames, interrupt_handler, tracers)
        else:
            pipe_mgr = self.pipes[1]
        pipe_mgr.server_loader.set_source(request.input)

        try:
            for response in inference_loop(
                args, True, pipe_mgr, interrupt_handler, self.server, None, tracers
            ):
                yield response
        finally:
            interrupt_handler.stream = None

    def StreamInit(self, request, context):
        try:
            self.ns.network, self.ns.output_dir = resolve_network(
                self.ns, request.network, self.network_yaml_info
            )
            if self.ns.network != self.network:
                self.network = self.ns.network
                if self.pipe_mgr:
                    self.pipe_mgr._pipeline.pipeline_cache = None
                self.pipe_mgr = None

        except Exception as e:
            error = f'Error initializing network: {repr(e)}'
            return pb.InitInferenceResponse(status=error)
        return pb.InitInferenceResponse(status="")

    def StreamInfer(self, request_iter, context):
        # Get network name and set as network and generate if needed
        # Override the current feed function with the new input
        try:
            stream = self.init(self.ns, 0, interrupt_handler, tracers, request_iter)

            for response in inference_loop(
                args, False, stream, interrupt_handler, self.server, None, tracers
            ):
                yield response
        finally:
            interrupt_handler.stream = None


def serve(port, args, network_yaml_info):
    port = str(port)
    MAX_MESSAGE_LENGTH = 8400000  # Enough for a 1080p image
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=10),
        options=[('grpc.max_receive_message_length', MAX_MESSAGE_LENGTH)],
    )
    inference_pb2_grpc.add_InferenceServicer_to_server(
        Inference(args, network_yaml_info, server), server
    )
    server.add_insecure_port("[::]:" + port)
    server.start()
    print("Server started, listening on " + port)
    server.wait_for_termination()


class InterruptHandler:
    def __init__(self):
        self.stream = None
        signal.signal(signal.SIGINT, self)

    def __call__(self, *args):
        if self.stream is not None:
            return self.stream.handle_interrupt()
        else:
            LOG.warning('Stream is not available to stop')
            sys.exit(1)


def request_reader(src):
    """
    Reads the image from the request stream

    Args:
        src : Request iterator

    Yields:
        dict : Simply contains the image as RGB
    """
    for request in src:
        image = np.frombuffer(request.image.image, dtype=np.uint8)
        image = np.reshape(
            image, (request.image.height, request.image.width, request.image.channels)
        )
        image = types.Image.fromarray(image, request.image.color_format)
        yield types.FrameInput(img=image, img_id='', ground_truth=None)


class InferenceServerLoader:
    def __init__(self, src):
        self.src = src
        self.cap = None
        self.num_frames = 0
        self.max_frames = 500 if isinstance(src, str) else 3000

    def set_source(self, src):
        self.src = src
        if self.cap is not None:
            self.cap.release()
        if isinstance(src, str):
            self.cap = cv2.VideoCapture(src)

    def __iter__(self):
        self.num_frames = 0
        if isinstance(self.src, str):
            self.cap = cv2.VideoCapture(self.src)
            if not self.cap.isOpened():
                LOG.error("Error opening video stream or file")
            return self

        return request_reader(self.src)

    def __next__(self):
        got, frame = self.cap.read()
        if not got or self.num_frames > self.max_frames:
            raise StopIteration
        self.num_frames += 1
        return types.FrameInput(
            img=types.Image.fromany(frame, types.ColorFormat.BGR), img_id='', ground_truth=None
        )


if __name__ == "__main__":
    args = sys.argv[1:]
    network_yaml_info = yaml_parser.get_network_yaml_info()
    parser = config.create_inference_argparser(
        network_yaml_info, description='Perform inference on an Axelera platform', port=5000
    )
    # Add in a dummy network
    args.insert(1, 'server')
    args.append('--no-display')
    args.append('--aipu-cores=1')

    args = parser.parse_args(args)
    logging_utils.configure_logging(logging_utils.get_config_from_args(args))
    logging_utils.configure_compiler_level(args)
    interrupt_handler = InterruptHandler()
    tracers = inf_tracers.create_tracers_from_args(args)

    logging.basicConfig()
    serve(args.port, args, network_yaml_info)

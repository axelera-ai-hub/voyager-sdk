#!/usr/bin/env python
# Copyright Axelera AI, 2024
# Inference client POC utilities

from pathlib import Path
import queue

import PIL
import cv2
import numpy as np

from axelera import types
from axelera.app import logging_utils, meta, utils
import axelera.app.inference_pb2 as inference_pb
import axelera.app.inference_pb2_grpc as inference_pb_grpc
from axelera.app.meta import ObjectDetectionMeta

LOG = logging_utils.getLogger(__name__)


def decode_bbox(server_response):
    object = server_response.obj

    bbox_size = 4
    bbox = object.boxes
    boxes1d = np.frombuffer(bbox, dtype=np.float32)
    boxes2d = np.reshape(boxes1d, (-1, bbox_size))
    return boxes2d


def decode_classes(server_response):
    object = server_response.obj
    classes = object.classes
    return np.frombuffer(classes, dtype=np.int32)


def decode_scores(server_response):
    object = server_response.obj
    scores = object.scores
    return np.frombuffer(scores, dtype=np.float32)


def createAxMeta(ax_meta, gst_meta_key, task_meta):
    if type(task_meta) == meta.ClassificationMeta:
        ax_meta.get_instance(
            gst_meta_key,
            type(task_meta),
            # TODO: see how to get top_k automatically
            num_classes=1000,
            # label is not important for measurement
            class_ids=task_meta.class_ids,
            scores=task_meta.scores,
            boxes=task_meta.boxes,
        )
    elif type(task_meta) == meta.ObjectDetectionMeta:
        ax_meta.get_instance(
            gst_meta_key,
            type(task_meta),
            boxes=task_meta.boxes,
            scores=task_meta.scores,
            class_ids=task_meta.class_ids,
            # TODO: using where instead of overwrite_labels
            labels=None,
            # labels=self.labels,
            # We will write flags to extra_info
            extra_info=dict(),
            make_extra_info_mutable=True,
        )
    elif type(task_meta) == meta.TrackerMeta:
        ax_meta.get_instance(
            gst_meta_key,
            type(task_meta),
            tracking_history=task_meta.tracking_history,
            class_ids=task_meta.class_ids,
        )
    else:
        raise NotImplementedError(f"Implement {type(task_meta)}")


def asAxMeta(server_response):
    boxes = decode_bbox(server_response)
    classes = decode_classes(server_response)
    scores = decode_scores(server_response)

    axmeta = meta.AxMeta('Inferences')
    createAxMeta(axmeta, 'object_meta', ObjectDetectionMeta(boxes, scores, classes))
    return axmeta


def _is_single_image(source):
    return Path(source).is_file() and utils.get_media_type(source) == 'image'


def stream_loader(input, queue):
    """
    Generates a stream of images from a video file or camera to send to the server
    optionally puts the image into a queue for later use

    Args:
        input (str): Path to the video file, camera or rtsp stream
        queue (Queue): Queue to put the image into

    Yields:
       StreamInferenceRequest : The image itself
    """

    if _is_single_image(input):
        print('Single image')
        image = PIL.Image.open(input)
        w, h = image.size
        frame = image.convert('RGBA').tobytes()
        num_channels = 4
        inf_image = inference_pb.Image(width=w, height=h, channels=num_channels, image=frame)
        if queue:
            queue.put((w, h, num_channels, frame))
        yield inference_pb.StreamInferenceRequest(image=inf_image)
        return

    cap = cv2.VideoCapture(input)
    if not cap.isOpened():
        raise ValueError(f'Error opening video stream or file:{input}')

    num_frames = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                LOG.info('Stream ended')
                break
            image = types.Image.fromarray(frame, types.ColorFormat.BGR)
            w, h = image.size
            frame = image.tobytes(types.ColorFormat.RGBA)
            num_channels = 4

            inf_image = inference_pb.Image(width=w, height=h, channels=num_channels, image=frame)
            num_frames += 1
            if queue:
                queue.put((w, h, num_channels, frame))
            yield inference_pb.StreamInferenceRequest(image=inf_image)
    finally:
        cap.release()


def remote_stream(stub, input, network):
    try:
        # for video files we read the stream and send the images to the server
        result = stub.StreamInit(inference_pb.InitInferenceRequest(network=network))
        if result.status:
            raise ValueError(f'{result.status}')
        q = queue.Queue()
        stream_iter = stream_loader(input, q)
        stream = stub.StreamInfer(stream_iter)

        for response in stream:
            meta = asAxMeta(response)
            image = None
            if (
                response.image.width > 0
                and response.image.height > 0
                and response.image.channels > 0
            ):
                frame = response.image.image
                w = response.image.width
                h = response.image.height
                num_channels = response.image.channels
            elif q:
                w, h, num_channels, frame = q.get()

            color_format = types.ColorFormat.RGBA if num_channels == 4 else types.ColorFormat.RGB
            image = np.frombuffer(frame, dtype=np.uint8)
            image = np.reshape(image, (h, w, num_channels))
            image = types.Image.fromarray(image, color_format)

            yield image, meta
    except Exception as e:
        LOG.error(e)

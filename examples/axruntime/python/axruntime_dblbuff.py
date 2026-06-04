#!/usr/bin/env python
# Copyright Axelera AI, 2025
#
# Double buffering example for the axelera.runtime API.
#
# This example extends the basic usage pattern to demonstrate double buffering,
# a hardware feature that overlaps DMA data transfer with AIPU computation for
# higher throughput. Use --repeat to process the same images multiple times
# (useful for benchmarking) and --no-double-buffer to compare performance.
#
# See double_buffering.md for a detailed explanation of the technique.
#
# Usage:
#   python axruntime_dblbuff.py build/resnet50-imagenet-onnx/model.json images/ --repeat 10
#   python axruntime_dblbuff.py build/resnet50-imagenet-onnx/model.json images/ --repeat 10 --no-double-buffer
#
from __future__ import annotations

import argparse
import collections
import functools
import logging
from logging import getLogger
import os
from pathlib import Path
import queue
import threading
import time

from axelera.runtime import Context, TensorInfo
import cv2  # noqa
import numpy as np

LOG = getLogger(__name__)

# ImageNet normalization parameters (model-specific)
mean = [0.485, 0.456, 0.406]
stddev = [0.229, 0.224, 0.225]

parser = argparse.ArgumentParser(
    description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
)

parser.add_argument(
    "path",
    type=str,
    help="Path to model to test. This should be a model.json file for an imagenet classification model",
)
parser.add_argument(
    "input_paths", type=Path, nargs='+', help="Path(s) to images or directories containing images"
)
parser.add_argument("--aipu-cores", type=int, default=4, help="Number of AIPU cores to use")
_DEFAULT_LABELS = os.path.expandvars(
    "$AXELERA_FRAMEWORK/ax_datasets/labels/imagenet1000_clsidx_to_labels.txt"
)
_DEFAULT_LABELS = os.path.relpath(_DEFAULT_LABELS)
parser.add_argument(
    "--labels",
    type=Path,
    default=_DEFAULT_LABELS,
    help="Path to text file containing labels (default:%(default)s)",
)
parser.add_argument(
    '--repeat',
    type=int,
    default=1,
    help='Number of times to repeat the input paths (for performance testing)',
)
parser.add_argument(
    '--no-double-buffer',
    dest='double_buffer',
    action='store_false',
    help='Disable double buffering in AIPU',
)
parser.add_argument(
    "-v",
    "--verbose",
    default=0,
    action="count",
    help="be more verbose; use repeatedly for more info",
)


@functools.lru_cache()
def _preproc_cacheable(image_path: Path, unpadded_shape, scale, zero_point, padding):
    # Cache preprocessing results by path - useful when --repeat > 1
    batch, height, width, _ = unpadded_shape
    image = cv2.imread(image_path)
    # Note: this is a simplified resize; proper ImageNet preprocessing uses
    # resize-to-256 then center-crop-to-224 for best accuracy
    image = cv2.resize(image, (width, height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = image / 255.0
    image = image - np.array(mean)
    image = image / np.array(stddev)
    # Quantize float32 → int8 using TensorInfo parameters
    quantized = np.round(image / scale + zero_point).clip(-128, 127).astype(np.int8)
    # Pad for hardware alignment (skip batch dimension)
    padded = np.pad(quantized, padding[1:], mode="constant", constant_values=zero_point)
    if batch > 1:
        padded = np.repeat(padded[np.newaxis, ...], batch, axis=0)
    return padded


def _preproc(image_path: Path, info: TensorInfo):
    return _preproc_cacheable(
        image_path, info.unpadded_shape, info.scale, info.zero_point, tuple(info.padding)
    )


def _postproc(
    frame_no: int,
    count: int,
    image_path: Path,
    output: np.array,
    labels: list[str],
    info: TensorInfo,
):
    '''Top-1 classification postprocessing for ImageNet models.'''
    # Depad then dequantize: int8 → float32
    out = output[tuple(slice(b, -e if e else None) for b, e in info.padding)]
    out = out.squeeze()
    out = (out.astype(np.float32) - info.zero_point) * info.scale

    cls = np.argmax(out)
    label = labels[cls] if cls < len(labels) else " (no label)"
    score = out[cls]
    if frame_no < 20 or frame_no > count - 20:  # avoid spamming stdout for large batches
        print(f"{frame_no:-5d}/{count}: {image_path} : classified as {cls=} {label=} {score=}%")


def _get_inputs(input_paths: list[Path]) -> collections.abc.Generator[Path, None, None]:
    for input_path in input_paths:
        if not input_path.exists():
            raise FileNotFoundError(input_path)

    for input_path in input_paths:
        if input_path.is_dir():
            for image_path in input_path.glob("*"):
                yield image_path
        else:
            yield input_path


class Worker(threading.Thread):
    """Runs inference on a dedicated thread.

    instance.run() is a blocking call. Running each ModelInstance on its own
    thread lets the OS schedule other workers while one is waiting for the AIPU.
    """

    def __init__(self, instance):
        self.instance = instance
        self.inqueue = queue.Queue()
        self.outqueue = queue.Queue()
        super().__init__()
        self.start()

    def run(self):
        while True:
            x = self.inqueue.get()
            if x is None:
                break
            frame_id, *inputs_outputs = x
            try:
                self.instance.run(*inputs_outputs)
            except Exception as e:
                self.outqueue.put(e)
                break
            else:
                self.outqueue.put((frame_id, inputs_outputs[1]))

    def push(self, frame_id, inputs, outputs):
        self.inqueue.put([frame_id, inputs, outputs])

    def pop(self):
        x = self.outqueue.get()
        if isinstance(x, Exception):
            raise x
        return x


def run_model(
    model_path: Path,
    aipu_cores: int,
    input_paths: list[Path],
    labels: list[str],
    double_buffer: bool,
):
    with Context() as ctx:
        model = ctx.load_model(model_path)

        input_infos, output_infos = model.inputs(), model.outputs()
        output_shapes = [i.shape for i in output_infos]
        assert len(output_shapes) == 1, "Only one output shape supported"
        assert output_shapes[0][1:-1] == (1, 1), "Only 1000 classes supported"
        batch_size = input_infos[0].shape[0]

        input_paths = list(_get_inputs(input_paths))
        if len(input_paths) < aipu_cores:
            # Fewer inputs than cores: only create as many instances as needed
            aipu_cores = len(input_paths)

        num_instances = aipu_cores // batch_size
        if aipu_cores % batch_size:
            LOG.warning(
                f"Number of AIPU cores ({aipu_cores}) is not a multiple of batch size ({batch_size})"
            )

        connections = [ctx.device_connect(None, batch_size) for _ in range(num_instances)]
        LOG.info(f"Creating {num_instances} model instances each with batch size of {batch_size}")
        instances = [
            c.load_model_instance(
                model,
                num_sub_devices=batch_size,
                aipu_cores=batch_size,
                double_buffer=double_buffer,  # Key difference from basic usage
            )
            for c in connections
        ]

        inputs = [[np.zeros(t.shape, np.int8) for t in input_infos] for _ in instances]
        outputs = [[np.zeros(t.shape, np.int8) for t in output_infos] for _ in instances]
        workers = [Worker(instance) for instance in instances]

        count = len(input_paths)

        start = time.perf_counter()
        try:
            prefill = len(workers)

            # ====================================================================
            # Double buffering result delay
            #
            # With double buffering enabled, each worker returns *stale* results:
            #   run(input0) → dummy0   ← warm-up, no valid result yet
            #   run(input1) → dummy1   ← warm-up, no valid result yet
            #   run(input2) → result0  ← now returning input0's result
            #   run(input3) → result1
            #
            # With N workers in round-robin, the delay multiplies: the first 2×N
            # results are dummies and must be dropped. We must also push 2×N extra
            # dummy frames at the end to flush those real results out of the pipeline.
            # ====================================================================
            num_to_drop = len(workers) * 2 if double_buffer else 0
            out_frameno = 0
            input_paths += [None] * num_to_drop  # dummy frames to flush pipeline

            # Because results are delayed, we can't directly match a popped result
            # to the input that produced it. We track this association with a deque:
            # push the input path when we submit, pop it when we have a valid result.
            awaiting_result = collections.deque()

            for in_frameno, image_path in enumerate(input_paths):
                next_available = in_frameno % len(workers)
                if image_path is None:
                    # Dummy frame to flush the pipeline. Zeros are easier to debug
                    # than garbage data if the last few results look wrong.
                    inputs[next_available][0][:] = np.zeros_like(inputs[next_available][0])
                else:
                    input = _preproc(image_path, input_infos[0])
                    inputs[next_available][0][:] = input
                workers[next_available].push(
                    image_path, inputs[next_available], outputs[next_available]
                )

                if in_frameno >= prefill:
                    next_ready = out_frameno % len(workers)
                    out_path, outs = workers[next_ready].pop()
                    awaiting_result.append(out_path)

                    if out_frameno >= num_to_drop:
                        # Enough dummy results dropped - now we have valid output.
                        # Match it with the correct input via the deque.
                        out_path = awaiting_result.popleft()
                        _postproc(
                            out_frameno - num_to_drop,
                            count,
                            out_path,
                            outs[0],
                            labels,
                            output_infos[0],
                        )
                    out_frameno += 1

            # Drain: collect the last N results still in the worker queues
            for _ in range(prefill):
                next_ready = out_frameno % len(workers)
                out_path, outs = workers[next_ready].pop()
                awaiting_result.append(out_path)
                if out_frameno >= num_to_drop:
                    out_path = awaiting_result.popleft()
                    _postproc(
                        out_frameno - num_to_drop,
                        count,
                        out_path,
                        outs[0],
                        labels,
                        output_infos[0],
                    )
                out_frameno += 1

        finally:
            duration = time.perf_counter() - start
            num = len(input_paths) - num_to_drop
            print(f"Processed {num} images in {duration:.2f}s, {num / duration:.1f} images/s")

            for worker in workers:
                worker.inqueue.put(None)
            for worker in workers:
                worker.join()


def main(args: argparse.Namespace):
    levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
    desired = levels.get(args.verbose, logging.DEBUG)
    logging.basicConfig(level=desired)

    model_path = Path(args.path)
    labels = args.labels.read_text().splitlines()
    if model_path.is_dir():
        model_path /= "model.json"
    try:
        input_paths = args.input_paths * args.repeat
        run_model(
            model_path,
            args.aipu_cores,
            input_paths,
            labels,
            args.double_buffer,
        )
    except Exception as e:
        if args.verbose:
            raise
        print(f'FAIL: {e}')
        return 1
    else:
        return 0


def entrypoint_main():
    args = parser.parse_args()
    try:
        main(args)
        return 0
    except RuntimeError as e:
        if args.verbose:
            raise
        return f'ERROR: {e}'


if __name__ == '__main__':
    entrypoint_main()

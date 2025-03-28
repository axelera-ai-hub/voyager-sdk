#!/usr/bin/env python
# Copyright Axelera AI, 2025
#
from __future__ import annotations

import argparse
import collections
import logging
from logging import getLogger
import os
from pathlib import Path
import queue
import threading

from axelera.runtime import Context, TensorInfo
import cv2  # noqa
import numpy as np

LOG = getLogger(__name__)

# these values are an artefact of the model
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
    "-v",
    "--verbose",
    default=0,
    action="count",
    help="be more verbose; use repeatedly for more info",
)


def _preproc(image_path: Path, info: TensorInfo):
    batch, height, width, _ = info.unpadded_shape
    image = cv2.imread(image_path)
    # note this is a naive preproc that is functional for imagenet models,
    # but at a loss of some accuracy.  The proper preproc should perform a resize to
    # say 256 square and then center crop to 224 square.
    image = cv2.resize(image, (width, height))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = image.astype(np.float32)
    image = image / 255.0
    image = image - np.array(mean)
    image = image / np.array(stddev)
    quantized = np.round(image / info.scale + info.zero_point).clip(-128, 127).astype(np.int8)
    padded = np.pad(quantized, info.padding[1:], mode="constant", constant_values=info.zero_point)
    if batch > 1:
        padded = np.repeat(padded[np.newaxis, ...], batch, axis=0)
    return padded


def _postproc(image_path: Path, output: np.array, labels: list[str], info: TensorInfo):
    '''Simple postproc for imagenet derived models.

    It simply performs a topk(1) after dequantizing the output, then prints the
    result to stdout.
    '''
    # dequantize the result
    out = output[tuple(slice(b, -e if e else None) for b, e in info.padding)]
    out = out.squeeze()
    out = (out.astype(np.float32) - info.zero_point) * info.scale

    cls = np.argmax(out)
    label = labels[cls] if cls < len(labels) else " (no label)"
    score = out[cls]
    print(f"{image_path} : classified as {cls=} {label=} {score=}%")


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
            # if we have fewer input paths than aipu cores then the end state is a bit fiddly
            # so just avoid it by only executing initialising the cores we need
            aipu_cores = len(input_paths)

        # depending on whether the model was compiled for batch mode or not then we will need to
        # adjust how many instances of the model should be created.  For example the AIPU has 4
        # cores and if the model was compiled with batch size of 2 then we will need to create 2
        # instances of the model to fully utilize the AIPU.
        num_instances = aipu_cores // batch_size
        if aipu_cores % batch_size:
            LOG.warning(
                f"Number of AIPU cores ({aipu_cores}) is not a multiple of batch size ({batch_size})"
            )

        # Create connections to the aipu cores, we need one connection per instance of the model
        connections = [ctx.device_connect(None, batch_size) for _ in range(num_instances)]
        LOG.info(f"Creating {num_instances} model instances each with batch size of {batch_size}")
        instances = [
            c.load_model_instance(
                model,
                num_sub_devices=batch_size,
                aipu_cores=batch_size,
            )
            for c in connections
        ]

        # create input and output buffers for each instance
        inputs = [[np.zeros(t.shape, np.int8) for t in input_infos] for _ in instances]
        outputs = [[np.zeros(t.shape, np.int8) for t in output_infos] for _ in instances]
        workers = [Worker(instance) for instance in instances]

        try:
            prefill = len(workers)
            out_frameno = 0
            for in_frameno, image_path in enumerate(input_paths):
                input = _preproc(image_path, input_infos[0])
                next_available = in_frameno % len(workers)
                inputs[next_available][0][:] = input
                workers[next_available].push(
                    image_path, inputs[next_available], outputs[next_available]
                )
                if in_frameno >= prefill:
                    next_ready = out_frameno % len(workers)
                    out_path, outs = workers[next_ready].pop()
                    _postproc(out_path, outs[0], labels, output_infos[0])
                    out_frameno += 1

            # drain the remaining workers to extract the last N frames
            for _ in range(prefill):
                next_ready = out_frameno % len(workers)
                out_path, outs = workers[next_ready].pop()
                _postproc(out_path, outs[0], labels, output_infos[0])
                out_frameno += 1

        finally:
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
        run_model(
            model_path,
            args.aipu_cores,
            args.input_paths,
            labels,
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

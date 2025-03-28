# Copyright Axelera AI, 2024
# Construct torch application pipeline
from __future__ import annotations

from copy import deepcopy
import time
from typing import TYPE_CHECKING, Callable

from . import base, frame_data
from .. import logging_utils, torch_utils, utils
from ..meta import AxMeta
from ..torch_utils import torch

if TYPE_CHECKING:
    from axelera import types

LOG = logging_utils.getLogger(__name__)


class TorchPipe(base.Pipe):
    def gen_end2end_pipe(self, input, output):
        self.input_generator = input.create_generator()
        self.pipeout = output
        self.pipeout.initialize_writer(input)
        self.batched_data_reformatter = input.batched_data_reformatter

    def init_loop(self) -> Callable[[], None]:
        return self._loop

    def _loop(self):
        self.device = torch.device(torch_utils.device_name('auto'))
        try:
            for data in self.input_generator:
                ts = time.time()
                if self._stop_event.is_set() or not data:
                    break
                if self.batched_data_reformatter:
                    batched_data = self.batched_data_reformatter(data)
                    assert len(batched_data) == 1, "batch_size > 1 is not supported"
                    data = batched_data[0]
                image_source = [data.img] if data.img else data.imgs
                is_pair_validation = data.imgs is not None

                with utils.catchtime('The network', logger=LOG.trace) as t:
                    meta = AxMeta(data.img_id, ground_truth=data.ground_truth)
                    for image in image_source:
                        for model_pipe in self.nn.tasks:
                            # should start from Input Operator
                            image, image_list, meta = model_pipe.input.exec_torch(
                                image, [], meta, data.stream_id
                            )
                            for result in image_list:
                                for op in model_pipe.preprocess:
                                    result = op.exec_torch(result)
                                image, result, meta = model_pipe.inference.exec_torch(
                                    image, result, meta
                                )
                                for op in model_pipe.postprocess:
                                    image, result, meta = op.exec_torch(image, result, meta)

                            if not is_pair_validation:
                                # always return tensor from the last model if having multiple models in a network
                                tensor = [
                                    deepcopy(
                                        element.cpu().detach().numpy()
                                        if hasattr(element, 'cpu')
                                        else element
                                    )
                                    for element in result
                                ]
                                if len(tensor) == 1:
                                    tensor = tensor[0]
                            else:
                                tensor = None
                    now = time.time()
                    fr = frame_data.FrameResult(image, tensor, meta, data.stream_id, ts, now)
                    self.pipeout.sink(fr)
                    self._callback(self.pipeout.result)
        except Exception as e:
            import traceback

            error_type = type(e).__name__
            error_message = str(e)
            tb = traceback.extract_tb(e.__traceback__)
            filename, line_number, _, _ = tb[-1]
            LOG.error(
                f'TorchPipe terminated due to {error_type} at {filename}:{line_number}: {error_message}'
            )
            LOG.error(f'Full traceback:\n{"".join(traceback.format_tb(e.__traceback__))}')
        finally:
            self._callback(None)  # no further messages
            self.pipeout.close_writer()
            self.nn.cleanup()


class TorchAipuPipe(TorchPipe):
    """Torch pipe for quantized model running on AIPU"""

    pass


class QuantizedPipe(TorchPipe):
    """Torch pipe for quantized model running on the host"""

    pass

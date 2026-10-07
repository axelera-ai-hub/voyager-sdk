# Copyright Axelera AI, 2023
# Construct torch application pipeline
from __future__ import annotations

from copy import deepcopy
import time
import traceback
from typing import Callable

from axelera import types

from . import base, frame_data
from .. import logging_utils, torch_utils, utils
from ..meta import AxMeta
from ..torch_utils import torch

LOG = logging_utils.getLogger(__name__)


class TorchPipe(base.Pipe):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._frame_generator = self._pipein.frame_generator()

    def init_loop(self) -> Callable[[], None]:
        return self._loop

    def _loop(self):
        self.device = torch.device(torch_utils.device_name('auto'))
        try:
            for data in self._frame_generator:
                assert isinstance(data, types.FrameInput), f'Expected FrameInput, got {type(data)}'
                ts = time.time()
                if self._stop_event.is_set() or not data:
                    break
                image_source = [data.img] if data.img else data.imgs
                is_pair_validation = data.imgs is not None

                inferences = 0
                with utils.catchtime('The network', logger=LOG.trace):
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
                                inferences += 1
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
                    fr = frame_data.FrameResult(
                        image, tensor, meta, data.stream_id, ts, now, inferences
                    )
                    event = frame_data.FrameEvent.from_result(fr)
                    if self._on_event(event) is False:
                        LOG.debug("_on_event requested to stop the pipeline")
                        break

        except Exception as e:
            LOG.error(f"Pipeline error occurred: {str(e)}\n{traceback.format_exc()}")
            self._on_event(frame_data.FrameEvent.from_end_of_pipeline(0, str(e)))
            self.nn.cleanup()
            raise
        else:
            self._on_event(frame_data.FrameEvent.from_end_of_pipeline(0, 'Normal termination'))
            self.nn.cleanup()


class TorchAipuPipe(TorchPipe):
    """Torch pipe for quantized model running on AIPU"""

    pass


class QuantizedPipe(TorchPipe):
    """Torch pipe for quantized model running on the host"""

    pass

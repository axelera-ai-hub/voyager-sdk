# Copyright Axelera AI, 2025
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as TF

from axelera import types
from axelera.app import gst_builder, logging_utils, meta
from axelera.app.operators import AxOperator, PipelineContext

LOG = logging_utils.getLogger(__name__)


class TopKDecoder(AxOperator):
    k: int = 1
    largest: bool = True
    sorted: bool = True
    softmax: bool = False

    def _post_init(self):
        pass

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        raise NotImplementedError("This is a dummy implementation")

    def exec_torch(self, image, predict, axmeta):
        if self.softmax:
            predict = TF.softmax(predict, dim=1)
        top_scores, top_ids = torch.topk(
            predict, k=self.k, largest=self.largest, sorted=self.sorted
        )
        top_scores = top_scores.cpu().detach().numpy()[0]
        top_ids = top_ids.cpu().detach().numpy()[0]
        LOG.info(f"Output tensor shape: {predict.shape}")
        LOG.info(f"Top {self.k} results: classified as {top_ids} with score {top_scores}")

        return image, predict, axmeta


# Tutorial-3: fill the results into AxTaskMeta to enable measurement and built-in (CV or GL) visualization
class TopKDecoderOutputMeta(AxOperator):
    k: int = 1
    largest: bool = True
    sorted: bool = True
    softmax: bool = False

    def _post_init(self):
        pass

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        where: str,
        compiled_model_dir: Path,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, where, compiled_model_dir
        )
        # here you can get infomation from the compiled model_info
        self.labels = model_info.labels
        self.num_classes = model_info.num_classes
        assert len(self.labels) > 0, "labels should not be empty"

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        raise NotImplementedError("This is a dummy implementation")

    def exec_torch(self, image, predict, axmeta):
        # For different vision tasks, Voyager SDK offers different meta classes.
        # Here we use ClassificationMeta as this is a classification task.
        # The built-in meta classes are well integrated with our visualizers and evaluators.
        model_meta = meta.ClassificationMeta(
            labels=self.labels,
            num_classes=self.num_classes,
        )

        if self.softmax:
            predict = TF.softmax(predict, dim=1)
        top_scores, top_ids = torch.topk(
            predict, k=self.k, largest=self.largest, sorted=self.sorted
        )
        top_scores = top_scores.cpu().detach().numpy()[0]
        top_ids = top_ids.cpu().detach().numpy()[0]

        model_meta.add_result(top_ids, top_scores)
        for i in range(self.k):
            LOG.debug(
                f"Top {i+1} result: classified as {self.labels[top_ids[i]]} with score {top_scores[i]}"
            )

        axmeta.add_instance(self.task_name, model_meta)
        return image, predict, axmeta


# Tutorial-6a: Implement your own GST decoder
class TopKDecoderWithMySimplifiedGstPlugin(AxOperator):
    def _post_init(self):
        pass

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        where: str,
        compiled_model_dir: Path,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, where, compiled_model_dir
        )
        self.labels = model_info.labels
        self.num_classes = model_info.num_classes
        assert len(self.labels) > 0, "labels should not be empty"

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        gst.decode_muxer(lib='libdecode_tu6a.so', options='')

    def exec_torch(self, image, predict, axmeta):
        model_meta = meta.ClassificationMeta(
            labels=self.labels,
            num_classes=self.num_classes,
        )

        top_scores, top_ids = torch.topk(predict, k=1)
        top_scores = top_scores.cpu().detach().numpy()[0]
        top_ids = top_ids.cpu().detach().numpy()[0]

        model_meta.add_result(top_ids, top_scores)

        axmeta.add_instance(self.task_name, model_meta)
        return image, predict, axmeta


# Tutorial-6b: Implement your own GST decoder
class TopKDecoderWithMyGstPlugin(AxOperator):
    k: int = 1

    def _post_init(self):
        pass

    def configure_model_and_context_info(
        self,
        model_info: types.ModelInfo,
        context: PipelineContext,
        task_name: str,
        taskn: int,
        where: str,
        compiled_model_dir: Path,
    ):
        super().configure_model_and_context_info(
            model_info, context, task_name, taskn, where, compiled_model_dir
        )
        self.labels = model_info.labels
        self.num_classes = model_info.num_classes

    def build_gst(self, gst: gst_builder.Builder, stream_idx: str):
        gst.decode_muxer(
            lib='libdecode_tu6b.so',
            options=f'meta_key:{str(self.task_name)};' f'top_k:{self.k};',
        )

    def exec_torch(self, image, predict, axmeta):
        model_meta = meta.ClassificationMeta(
            labels=self.labels,
            num_classes=self.num_classes,
        )

        if self.softmax:
            predict = TF.softmax(predict, dim=1)
        top_scores, top_ids = torch.topk(predict, k=self.k)
        top_scores = top_scores.cpu().detach().numpy()[0]
        top_ids = top_ids.cpu().detach().numpy()[0]

        model_meta.add_result(top_ids, top_scores)
        for i in range(self.k):
            LOG.debug(
                f"Top {i+1} result: classified as {self.labels[top_ids[i]]} with score {top_scores[i]}"
            )

        axmeta.add_instance(self.task_name, model_meta)
        return image, predict, axmeta

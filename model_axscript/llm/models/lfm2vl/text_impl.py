# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""LFM2.5-VL decoder on Metis (registered as lfm2vl-metis-opt).

The device side of the text tower is the LFM2 hybrid layer stack, unchanged --
LFM2.5-VL's decoder IS an LFM2 decoder (short-conv token mixers with grouped-query
full-attention layers interleaved, SwiGLU MLPs, RMSNorm), so both layer kernels and
both buffer packers come straight from ``models/lfm2``. Two things are wrapped:

  * **the config is nested.** A VLM bundle's config is ``{"text": ..., "vision": ...}``,
    so the builders are handed ``config.text`` and the packers a bundle view whose
    config is the text sub-config. Nothing else about them changes.
  * **image embeddings.** The host writes the vision tower's rows over the
    image-placeholder positions of ``inputs_embeds``.

LFM2 rotary positions are plain 1-D, so the decode loop's tables are the backbone's
(design-P padded head, from ``Lfm2OptimizedImplementation``) and nothing extra is
spliced into the layer spine -- the vision side reaches the decoder only through the
embeddings. That also means **chunked prefill works**: the scatter is indexed by
absolute position, so each chunk writes the vision rows that fall inside it (see
:meth:`_splice`).
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging

import numpy as np

from ...implementation import register_implementation
from ..lfm2.optimized_impl import (
    Lfm2OptimizedImplementation,
    attention_layer_buffers,
    conv_layer_buffers,
    make_attention_layer_kernel_multicore,
    make_conv_layer_kernel_multicore,
)
from ...implementation import LayerKernel

logger = logging.getLogger(__name__)


#: The graph that ingests a whole prompt in one full-context call. A chunked artifact
#: carries BOTH this and the chunk-sized "prefill", because neither is best for every
#: prompt length -- see :func:`plan_prefill`.
FULL_PREFILL_GRAPH = "prefill_full"

#: How many chunked positions a REPLAYED token costs. A chunk runs at ~0.29 ms/position
#: and a replayed token is a whole decode step (~13.4 ms) -- measured on LFM2.5-VL-450M at
#: 8 cores. Only the ratio matters, and it is stable because both scale with the same
#: weights: it is what makes the choice below arithmetic rather than a guess.
REPLAY_COST_IN_POSITIONS = 46


def plan_prefill(n_tokens: int, chunk_len: int | None, context_len: int) -> int | None:
    """The chunk length to ingest an ``n_tokens`` prompt with, or None for one full call.

    A chunked artifact can ingest a prompt either way, and which is cheaper depends on the
    length. Counting everything in "positions a chunk would process":

    * the chunked path costs ``(n - r)`` for its whole chunks plus
      ``r * REPLAY_COST_IN_POSITIONS`` for the remainder it has to replay a token at a
      time, where ``r = n % chunk_len``;
    * the full-context graph costs ``context_len`` whatever the prompt's length, since it
      is compiled for every position.

    So chunk when ``n + (REPLAY_COST_IN_POSITIONS - 1) * r < context_len``. That single
    inequality reproduces every measured decision: it chunks an aligned 512- or
    1024-token prompt and a 273-token one (17 left over), and takes the full graph for 100
    or 1000 tokens, where the replay would cost 3-4x the whole prefill.

    Returning None rather than clamping is deliberate: the fallback is the graph a
    non-chunked artifact would have used, so a chunked build is never slower than not
    chunking at all.
    """
    if not chunk_len or n_tokens < chunk_len:
        # Shorter than one chunk means every token is replayed; the full graph wins unless
        # the prompt is tiny, which the same inequality decides.
        remainder = n_tokens
    else:
        remainder = n_tokens % chunk_len
    if n_tokens + (REPLAY_COST_IN_POSITIONS - 1) * remainder < context_len:
        return chunk_len
    return None


@dataclass
class Lfm2VlPrompt:
    """Everything about the current multimodal prompt the decode loop needs.

    Built once per prompt by the pipeline (see pipeline.build_prompt) and handed to the
    implementation for the duration of one generation.
    """

    #: token ids of the full prompt (image placeholders included).
    input_ids: np.ndarray
    #: indices of the image-placeholder tokens, in order.
    image_positions: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    #: (num_image_tokens, hidden) image embeddings for those positions.
    image_embeds: np.ndarray | None = None


# --- kernel/packer adapters: the LFM2 ones over the TEXT half of a VL config ------------


def make_vl_attention_kernel(config, ctx_len, phase, chunk_len=None, cores: int = 1):
    """The full-attention layer kernel, built from the text sub-config.

    The sharded builder by name: lfm2vl is the configuration these shards were measured
    at, and binding it here keeps the text-only ``lfm2-metis-opt`` goldens on the
    unsharded program they describe.
    """
    return make_attention_layer_kernel_multicore(
        config.text, ctx_len, phase, chunk_len=chunk_len, cores=cores
    )


def make_vl_conv_kernel(config, ctx_len, phase, chunk_len=None, cores: int = 1):
    """The short-conv layer kernel, built from the text sub-config (sharded builder)."""
    return make_conv_layer_kernel_multicore(
        config.text, ctx_len, phase, chunk_len=chunk_len, cores=cores
    )


def _text_view(bundle):
    """The bundle with its config narrowed to the text sub-config.

    The LFM2 packers read ``bundle.config["num_key_value_heads"]``, and a VLM bundle's
    config is nested. Everything else they read is keyed by ``.layers.<i>.`` and so
    already selects the decoder only -- the vision tower's parameters live under
    ``visual.blocks.<i>.`` / ``visual.connector.``.
    """
    import dataclasses

    return dataclasses.replace(bundle, config=bundle.config["text"])


def vl_attention_layer_buffers(bundle, spec, layer):
    return attention_layer_buffers(_text_view(bundle), spec, layer)


def vl_conv_layer_buffers(bundle, spec, layer):
    return conv_layer_buffers(_text_view(bundle), spec, layer)


class Lfm2VlTextImplementation(Lfm2OptimizedImplementation):
    """Host embed (+ image splice) -> the LFM2 hybrid device stack -> host lm_head.

    Everything about the device side -- the two layer kinds, their phases, their state,
    the per-call ``state_scale`` conv reset gate and the design-P rotary tables -- is
    inherited from ``lfm2-metis-opt``. What this adds is the nested-config adapters and
    the vision splice.

    The registry hands out one implementation instance, so the prompt context is set for
    the duration of a generation and cleared after (one generation at a time per process).
    """

    name = "lfm2vl-metis-opt"
    architecture = "lfm2vl"

    def __init__(self) -> None:
        self._prompt: Lfm2VlPrompt | None = None
        self._prefill_graph = "prefill"

    # --- prompt context -------------------------------------------------------------

    def set_prompt(self, prompt: Lfm2VlPrompt | None) -> None:
        self._prompt = prompt

    def prompt(self) -> Lfm2VlPrompt:
        if self._prompt is None:
            raise RuntimeError(
                "no prompt context set; call set_prompt() (pipeline.Lfm2VlPipeline does) "
                "before generating -- the image embeddings live there"
            )
        return self._prompt

    # --- the layer stack ------------------------------------------------------------

    def layer_kernels(self, model_config):
        base = super().layer_kernels(model_config)
        adapters = {
            "full_attention": (make_vl_attention_kernel, vl_attention_layer_buffers),
            "conv": (make_vl_conv_kernel, vl_conv_layer_buffers),
        }
        # A third phase, built only when build_artifacts asks for it (chunked builds):
        # DecoderLayerImplementation._compile_phase forwards chunk_len to the phase named
        # exactly "prefill", so this one compiles at the FULL context with no kernel change
        # -- the builders derive seq_len as `chunk_len or ctx_len`.
        phases = ("prefill", FULL_PREFILL_GRAPH, "decode")
        return {
            kind: _rebuild(base[kind], build=build, layer_buffers=buffers, phases=phases)
            for kind, (build, buffers) in adapters.items()
        }

    def select_prefill(self, n_tokens, chunk_len, context_len, available=()) -> int | None:
        """Choose how to ingest a prompt of this length; returns the chunk length or None.

        Also records which graph :meth:`_prefill_graph_name` will name, because the shared
        decode loop asks for the graph and reads ``config.chunk_len`` separately, and the
        two have to agree.
        """
        planned = plan_prefill(n_tokens, chunk_len, context_len)
        if planned is not None:
            self._prefill_graph = "prefill"
            return planned
        if chunk_len and FULL_PREFILL_GRAPH not in set(available):
            # A chunked artifact built before the full-context graph existed: chunking is
            # all there is, so use it, but say what the remainder will cost.
            logger.warning(
                "%d-token prompt leaves %d tokens to replay one at a time, which costs more "
                "than a full-context prefill would -- but this artifact has no %s graph. "
                "Rebuild it (current builds carry both) or drop --chunk-len.",
                n_tokens,
                n_tokens % chunk_len,
                FULL_PREFILL_GRAPH,
            )
            self._prefill_graph = "prefill"
            return chunk_len
        self._prefill_graph = FULL_PREFILL_GRAPH if chunk_len else "prefill"
        return None

    def _prefill_graph_name(self) -> str:
        """The prefill graph the last :meth:`select_prefill` chose (default: the only one)."""
        return self._prefill_graph

    def layer_kinds(self, model_config):
        return super().layer_kinds(model_config.text)

    # --- host hooks -----------------------------------------------------------------

    def _splice(self, embeds: np.ndarray, base_position: int, count: int) -> np.ndarray:
        """Write the vision rows falling in ``[base_position, base_position + count)``.

        Indexed by ABSOLUTE prompt position rather than by "is this the whole prompt", so
        one call serves the full prefill, a chunked-prefill chunk, and the token-by-token
        decode replay a phases=("decode",) artifact falls back to.
        """
        prompt = self._prompt
        if prompt is None or prompt.image_embeds is None:
            return embeds
        positions = np.asarray(prompt.image_positions, dtype=np.int64)
        inside = (positions >= base_position) & (positions < base_position + count)
        if not inside.any():
            return embeds
        rows = np.asarray(prompt.image_embeds, dtype=np.float32)[inside]
        embeds[0, positions[inside] - base_position] = rows.reshape(rows.shape[0], -1, 64)
        return np.ascontiguousarray(embeds)

    def prefill_inputs(self, host_params, token_ids, context_len):
        inputs = super().prefill_inputs(host_params, token_ids, context_len)
        inputs["inputs_embeds"] = self._splice(inputs["inputs_embeds"], 0, context_len)
        return inputs

    def _chunk_inputs(self, host_params, chunk_tokens, base_pos, context_len, chunk_len):
        inputs = super()._chunk_inputs(host_params, chunk_tokens, base_pos, context_len, chunk_len)
        inputs["inputs_embeds"] = self._splice(inputs["inputs_embeds"], base_pos, chunk_len)
        return inputs

    def step_inputs(self, host_params, token_id, position, context_len):
        inputs = super().step_inputs(host_params, token_id, position, context_len)
        # A generated token is never an image placeholder, so this is a no-op during
        # decode; it is not during the token-by-token prompt replay of an artifact
        # compiled without a prefill graph, which walks the prompt through this path.
        inputs["inputs_embeds"] = self._splice(inputs["inputs_embeds"], position, 1)
        return inputs


def _rebuild(kernel: LayerKernel, **overrides) -> LayerKernel:
    """A copy of ``kernel`` with some fields replaced (LayerKernel is frozen)."""
    import dataclasses

    return dataclasses.replace(kernel, **overrides)


register_implementation(Lfm2VlTextImplementation())

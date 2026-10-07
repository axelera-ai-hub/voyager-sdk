# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL decoder on Metis (registered as qwen3vl-metis-opt).

The device side of the text tower is the Qwen3 optimized layer kernel, unchanged
-- Qwen3-VL's decoder IS a Qwen3 decoder (GQA, QK-Norm, SwiGLU, RMSNorm), and the
two things it adds live on the host:

  * **interleaved MRoPE.** The kernel consumes rotary tables, not positions, so a
    3-D rope costs nothing on device: the host builds cos/sin from the prompt's
    (t, h, w) position ids (qwen3vl.mrope_cos_sin). After the prompt every token
    is text, where t == h == w, and the interleaving collapses to ordinary 1-D
    RoPE at ``position + delta`` -- so decode reuses the standard tables.
  * **image embeddings.** ``embed_tokens`` splices the vision tower's merged
    embeddings over the image-placeholder rows of ``inputs_embeds``.

  * **DeepStack.** A different merged feature map is added into the output of each
    of the first three decoder layers, at the image-token positions. It runs on
    device as one extra kind (``deepstack_add``) spliced into the activation spine
    after each injected layer:

        [layer 0, ds add, layer 1, ds add, layer 2, ds add, layer 3, ...]

    An elementwise kernel of its own rather than a variant of the layer kernel: that
    keeps both Qwen3 layer kernels untouched and costs 2 extra codegen units instead
    of 6. Each add reads its own host buffer (``deepstack_L<position>`` in the graph
    descriptor's ``per_layer_inputs``), which the host pre-scatters into the
    image-placeholder rows of a context-sized buffer, so the device side needs no
    indexing. The features come from the vision tower's tap outputs (see vision_impl).
"""

# flake8: noqa: F821  -- A.Buffer dtype-string annotations look like F821 names
#
# NB: no `from __future__ import annotations` here -- this module now authors a
# prim_func (the DeepStack add), and TVMScript reads its parameter annotations at
# runtime; PEP 563 would hand it strings instead of the A.Buffer objects it parses
# (same reason vision_impl and the qwen3/dino kernel modules omit it).

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from ...implementation import (
    HostKernel,
    LayerKernel,
    UniformDecoderImplementation,
    make_rope_cos_sin,
    register_implementation,
)
from ..multicore_layer import MulticoreDecoderImplementation
from ..qwen3.multicore_impl import QWEN3_SPEC, make_qwen3_layer_kernel, qwen3_layer_buffers
from ..qwen3.optimized_impl import make_optimized_layer_kernel, optimized_decode_layer_buffers

if TYPE_CHECKING:
    from .qwen3vl import Qwen3VLConfig


#: The spine kind that adds one DeepStack feature map into the activation.
_DEEPSTACK_KIND = "deepstack_add"


def deepstack_kinds(layer_kind: str, num_layers: int, num_deepstack: int) -> list[str]:
    """The decoder's spine kinds: a DeepStack add after each of the first N layers.

    ``[layer0, ds, layer1, ds, layer2, ds, layer3, ...]`` -- HF adds feature k into the
    output of decoder layer k, so the add sits directly after that layer.
    """
    kinds: list[str] = []
    for layer in range(num_layers):
        kinds.append(layer_kind)
        if layer < num_deepstack:
            kinds.append(_DEEPSTACK_KIND)
    return kinds


def deepstack_positions(num_deepstack: int) -> list[int]:
    """Spine positions of the DeepStack adds -- the ``deepstack_L<position>`` suffixes.

    Add k follows layer k, so it lands at 2k + 1.
    """
    return [2 * layer + 1 for layer in range(num_deepstack)]


def make_deepstack_add_kernel(config: "Qwen3VLConfig", ctx_len: int, phase: str):
    """DeepStack injection: hidden += this layer's merged vision features.

    The host writes the feature map into a context-sized buffer (zeros outside the
    image-placeholder rows), so the device side is one elementwise add over the whole
    activation -- no indexing, no per-core work, identical for every core count.
    """
    from axelera.compiler.alto import axol
    from axelera.compiler.alto import script as A

    hidden_pw = config.text.hidden_size // 64
    seq_len = 1 if phase == "decode" else ctx_len

    @A.ir_module
    class Qwen3VLDeepStackAdd:
        @A.prim_func
        def main(
            hidden_states: A.Buffer((1, seq_len, hidden_pw), "float32x64", scope=A.DDR),
            deepstack: A.Buffer((1, seq_len, hidden_pw), "float32x64", scope=A.DDR),
        ):  # noqa: F821
            A.output(axol.atomic.add(hidden_states, deepstack))

    return Qwen3VLDeepStackAdd


def no_layer_buffers(bundle, spec, layer):
    """The DeepStack add owns no constants: its addend is a host input."""
    return {}


def make_text_layer_kernel(config: "Qwen3VLConfig", ctx_len: int, phase: str):
    """The single-core decoder layer kernel: Qwen3's, built from the text sub-config."""
    return make_optimized_layer_kernel(config.text, ctx_len, phase)


def make_text_layer_kernel_multicore(
    config: "Qwen3VLConfig", ctx_len: int, phase: str, chunk_len: int | None = None, cores: int = 1
):
    """The multicore decoder layer kernel (SDPA heads + MLP matmuls across cores).

    Qwen3's, one layer per kernel: a DeepStack add may follow any of the first layers. The
    signature has to carry ``chunk_len``/``cores`` because
    DecoderLayerImplementation._compile_phase forwards them by inspecting it.
    """
    return make_qwen3_layer_kernel(
        config.text, ctx_len, phase, chunk_len=chunk_len, cores=cores, layers_per_kernel=1
    )


def multicore_layer_buffers(bundle, spec, layer):
    """Qwen3's multicore packer, over the TEXT half of a Qwen3-VL bundle.

    The packer reads `bundle.config["hidden_size"]`, and a VLM bundle's config is nested,
    so hand it a view whose config is the text sub-config. Everything else it reads is
    keyed by ".layers.<i>." and already selects the decoder only.
    """
    import dataclasses

    text_view = dataclasses.replace(bundle, config=bundle.config["text"])
    return qwen3_layer_buffers(text_view, spec, layer)


@dataclass
class Qwen3VLPrompt:
    """Everything about the current multimodal prompt the decode loop needs.

    Built once per prompt by the pipeline (see pipeline.build_prompt) and handed
    to the implementation for the duration of one generation.
    """

    #: token ids of the full prompt (image placeholders included).
    input_ids: np.ndarray
    #: indices of the image-placeholder tokens, in order.
    image_positions: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    #: (num_image_tokens, hidden) merged vision embeddings for those positions.
    image_embeds: np.ndarray | None = None
    #: One (num_image_tokens, hidden) DeepStack feature map per injected decoder layer,
    #: in tower order -- added into those layers' outputs at the image positions.
    deepstack: list[np.ndarray] | None = None
    #: (prompt_len, head_dim/2) MRoPE tables for the prompt positions.
    rope_cos: np.ndarray | None = None
    rope_sin: np.ndarray | None = None
    #: absolute rotary position of generated token i == len(input_ids) + i + delta.
    delta: int = 0


class Qwen3VLHostMixin:
    """The VLM half of the host loop: the prompt context, MRoPE, image splice, DeepStack.

    Mixed into both decoder implementations (single- and multi-core) because the two
    differ only in the device kernel and the shape of the rotary buffer it wants.
    """

    def __init__(self) -> None:
        self._prompt: Qwen3VLPrompt | None = None

    # --- the layer stack: decoder layers + the DeepStack adds between them -----------

    def layer_kernels(self, model_config):
        from ...implementation import LayerKernel

        return {
            **super().layer_kernels(model_config),
            _DEEPSTACK_KIND: LayerKernel(
                _DEEPSTACK_KIND,
                build=make_deepstack_add_kernel,
                layer_buffers=no_layer_buffers,
                phases=("prefill", "decode"),
                # One host buffer per injected layer: "deepstack_L<position>".
                per_layer_inputs=frozenset({"deepstack"}),
            ),
        }

    def layer_kinds(self, model_config):
        return deepstack_kinds(
            self._layer_kernel().name,
            model_config.num_hidden_layers,
            len(model_config.vision.deepstack_visual_indexes),
        )

    def codegen_jobs(self, model_config, config):
        """The (kind, phase) codegen units -- refusing a chunked-prefill build outright.

        Every compile route reaches codegen through here (``compile`` -> ``compile_codegen``
        -> this, and the parallel eval path calls it directly), which is why the refusal
        sits here rather than in ``compile``.

        Only ONE of the two spliced kinds can honour ``chunk_len``: the decoder layer
        kernel's builder takes it (Qwen3's does, and ``_compile_phase`` forwards it by
        inspecting the signature) while the DeepStack add's cannot -- its addend is a host
        buffer covering the whole prompt, scattered at the image-placeholder rows. A build
        that accepted the flag would give the layer kernel a ``chunk_len``-row prefill
        activation and the DeepStack add a ``context_len``-row one on the SAME spine tensor,
        and would only say so on device. So it is refused at the flag, for the reason
        ``_chunk_inputs`` refuses the run.
        """
        chunk_len = getattr(config, "chunk_len", None)
        if chunk_len:
            raise ValueError(
                f"chunk_len={chunk_len} is not supported for Qwen3-VL: the MRoPE tables and "
                "the DeepStack scatter are built for the whole prompt at once, and the "
                f"{_DEEPSTACK_KIND} kernel spliced into the decoder spine has no chunked "
                "form -- it would compile at the full context while the layer kernel "
                "compiled at the chunk. Build with chunk_len unset."
            )
        return super().codegen_jobs(model_config, config)

    def bundle_layers(self, kinds):
        """Map spine position -> decoder layer; the DeepStack adds hold no weights."""
        layers = []
        layer = 0
        for kind in kinds:
            if kind == _DEEPSTACK_KIND:
                layers.append(None)
                continue
            layers.append(layer)
            layer += 1
        return layers

    def _pipeline_stages(self, kernels, used_kinds, config):
        stages = super()._pipeline_stages(kernels, used_kinds, config)
        if _DEEPSTACK_KIND in kernels:
            stages.append(
                {
                    "name": _DEEPSTACK_KIND,
                    "placement": "device",
                    "graphs": list(self._phases(kernels[_DEEPSTACK_KIND], config)),
                }
            )
        return stages

    # --- prompt context -----------------------------------------------------------
    # The registry hands out one implementation instance, so the prompt is set for the
    # duration of a generation and cleared after (one generation at a time per process).

    def set_prompt(self, prompt: Qwen3VLPrompt | None) -> None:
        self._prompt = prompt

    def prompt(self) -> Qwen3VLPrompt:
        if self._prompt is None:
            raise RuntimeError(
                "no prompt context set; call set_prompt() (pipeline.Qwen3VLPipeline does) "
                "before generating -- the MRoPE tables and image embeddings live there"
            )
        return self._prompt

    # --- host hooks ---------------------------------------------------------------

    def _rope_tables(self, positions, head_dim: int, rope_theta: float):
        """MRoPE inside the prompt, plain 1-D RoPE (shifted by delta) beyond it.

        Text tokens carry t == h == w, so the interleaved sections all read the same
        position and the tables equal the 1-D ones: decode needs only the position offset
        the images introduced.
        """
        prompt = self._prompt
        pos = np.atleast_1d(np.asarray(positions)).astype(np.int64)
        if prompt is None or prompt.rope_cos is None:
            return make_rope_cos_sin(pos, head_dim, rope_theta)
        rope_pw = (head_dim // 2) // 64
        cos = np.empty((pos.size, head_dim // 2), dtype=np.float32)
        sin = np.empty_like(cos)
        prompt_len = prompt.rope_cos.shape[0]
        inside = pos < prompt_len
        cos[inside] = prompt.rope_cos[pos[inside]]
        sin[inside] = prompt.rope_sin[pos[inside]]
        if (~inside).any():
            tail_cos, tail_sin = make_rope_cos_sin(
                pos[~inside] + prompt.delta, head_dim, rope_theta
            )
            cos[~inside] = tail_cos.reshape(-1, head_dim // 2)
            sin[~inside] = tail_sin.reshape(-1, head_dim // 2)
        return (
            cos.reshape(1, pos.size, rope_pw, 64),
            sin.reshape(1, pos.size, rope_pw, 64),
        )

    def _deepstack_inputs(self, host_params: dict, seq_len: int, *, prefill: bool) -> dict:
        """The per-layer DeepStack buffers: ``{"deepstack_L<position>": (1, seq, hidden)}``.

        Scattered here rather than on device: the feature map covers only the image
        placeholder rows, so the host writes those rows of an otherwise-zero context-sized
        buffer and the device kernel stays a plain add. A decode step feeds zeros -- a
        generated token is never an image position -- so one kernel serves both phases.
        """
        count = int(host_params.get("deepstack_layers", 0))
        if not count:
            return {}
        hidden_pw = host_params["embed_weight"].shape[-1] // 64
        prompt = self._prompt
        features = prompt.deepstack if (prefill and prompt is not None) else None
        inputs = {}
        for slot, position in enumerate(deepstack_positions(count)):
            buffer = np.zeros((1, seq_len, hidden_pw, 64), dtype=np.float32)
            if features is not None:
                rows = np.asarray(features[slot], dtype=np.float32)
                buffer[0, prompt.image_positions] = rows.reshape(rows.shape[0], -1, 64)
            inputs[f"deepstack_L{position}"] = buffer
        return inputs

    def prefill_inputs(self, host_params, token_ids, context_len):
        inputs = super().prefill_inputs(host_params, token_ids, context_len)
        inputs.update(self._deepstack_inputs(host_params, context_len, prefill=True))
        return inputs

    def step_inputs(self, host_params, token_id, position, context_len):
        inputs = super().step_inputs(host_params, token_id, position, context_len)
        inputs.update(self._deepstack_inputs(host_params, 1, prefill=False))
        return inputs

    def _chunk_inputs(self, host_params, chunk_tokens, base_pos, context_len, chunk_len):
        raise NotImplementedError(
            "chunked prefill is not supported for Qwen3-VL: the MRoPE tables and the "
            "DeepStack scatter are built for the whole prompt at once"
        )

    def embed_tokens(self, token_ids: np.ndarray, host_params: dict) -> np.ndarray:  # noqa: D401
        """Token embeddings with the vision embeddings spliced over image placeholders.

        Only the prefill call covers the image positions (decode feeds one generated
        text token at a time), and it passes the whole prompt, so the recorded
        placeholder indices line up with the returned rows.
        """
        embeds = super().embed_tokens(token_ids, host_params)
        prompt = self._prompt
        if prompt is None or prompt.image_embeds is None:
            return embeds
        ids = np.asarray(token_ids).reshape(-1)
        if ids.size != prompt.input_ids.size:  # a decode step, not the prompt
            return embeds
        rows = np.asarray(prompt.image_embeds, dtype=np.float32)
        embeds[0, prompt.image_positions] = rows.reshape(rows.shape[0], -1, 64)
        return np.ascontiguousarray(embeds)


class Qwen3VLTextImplementation(Qwen3VLHostMixin, UniformDecoderImplementation):
    """Single-core: host embed (+ image splice) -> device decoder stack -> host lm_head."""

    name = "qwen3vl-metis-opt"
    architecture = "qwen3vl"

    def kernels(self):
        return [
            HostKernel("embedding", params=("embed_weight",)),
            LayerKernel(
                "decode_layer",
                build=make_text_layer_kernel,
                layer_buffers=optimized_decode_layer_buffers,
                phases=("prefill", "decode"),
                shared_inputs=self.shared_inputs,
                state=frozenset({"k_out", "v_out"}),
                enable_buffer_promotion=True,
            ),
            HostKernel("lm_head", params=("final_norm_weight", "embed_weight", "rms_eps")),
        ]


class Qwen3VLMulticoreImplementation(Qwen3VLHostMixin, MulticoreDecoderImplementation):
    """Multicore: the Qwen3 multicore decoder layer (SDPA heads + MLP matmuls sharded).

    The device kernel and its packer are Qwen3's, unchanged. Its host contract differs
    from the single-core one only in an int8 1-head causal mask and one combined
    ``rope_trig`` (sin||cos) instead of four rotary buffers, so the VLM feeds MRoPE
    through ``_trig`` and takes the rest from the mixin.
    """

    name = "qwen3vl-metis-multicore"
    architecture = "qwen3vl"
    spec = QWEN3_SPEC
    layer_build = staticmethod(make_text_layer_kernel_multicore)
    layer_buffers_fn = staticmethod(multicore_layer_buffers)

    def _trig(self, positions, host_params):
        """MRoPE, packed as the multicore kernel's single sin||cos buffer."""
        cos, sin = self._rope_tables(
            positions, int(host_params["head_dim"]), float(host_params["rope_theta"])
        )
        return np.ascontiguousarray(np.concatenate([sin, cos], axis=2))


register_implementation(Qwen3VLTextImplementation())
register_implementation(Qwen3VLMulticoreImplementation())

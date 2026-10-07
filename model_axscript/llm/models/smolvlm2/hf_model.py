# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""SmolVLM2 on Metis behind the HuggingFace image-text-to-text API.

``AxSmolVLMForConditionalGeneration`` IS a ``transformers.SmolVLMForConditionalGeneration``
(subclassed, not re-declared) whose weights live in a compiled artifact instead of in
torch: the SigLIP tower and the SmolLM2 decoder run on the device, the host does the
embedding splice and the lm_head. Everything above the model -- the processor, chat
templates, sampling parameters, streamers -- is stock transformers, so a customer drives
it with the code they already have:

    >>> from transformers import AutoModelForImageTextToText, AutoProcessor
    >>> from model_axscript.llm.models.smolvlm2.hf_model import register_auto
    >>> register_auto()   # teaches the Auto classes our model type; see its docstring
    >>> model = AutoModelForImageTextToText.from_pretrained("build/smolvlm2-256m")
    >>> processor = AutoProcessor.from_pretrained("build/smolvlm2-256m")
    >>> messages = [{"role": "user", "content": [
    ...     {"type": "image", "image": image}, {"type": "text", "text": "Describe it."}]}]
    >>> inputs = processor.apply_chat_template(messages, tokenize=True,
    ...     add_generation_prompt=True, return_dict=True, return_tensors="pt")
    >>> ids = model.generate(**inputs, max_new_tokens=64, do_sample=True, min_p=0.05)
    >>> processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

The artifact directory IS a HuggingFace model directory: ``save_hf_metadata`` writes the
config, the processor (chat template included) and the generation defaults into it at
build time, so the Auto classes load one with no hub and no network.
``AxSmolVLMForConditionalGeneration.from_pretrained(dir)`` is the same load spelled
directly; importing this module is what registers the artifact's ``ax_smolvlm`` model
type, which keeps ordinary SmolVLM checkpoints going to stock transformers.

Generation is ``GenerationMixin.generate`` itself, not a lookalike: the model implements
``forward`` (prefill when the step starts at position 0, one decode step otherwise) and
``prepare_inputs_for_generation``, and transformers does the rest -- sampling and every
sampler parameter (temperature, top_k, top_p, min_p, repetition penalties), stopping
criteria, streamers, the returned id tensor. Nothing about min-p, ``TextIteratorStreamer``
or ``GenerationConfig`` is reimplemented here, so they behave as they do for any other HF
model, including the parts nobody has asked for yet.

The device's KV cache is what forces the two limits: it lives on the accelerator and
cannot be inspected, reordered or duplicated, so ``past_key_values`` comes back as None
(HF only threads it through), and beam search / num_return_sequences > 1 / batches > 1
raise instead of quietly misbehaving. ``forward`` with ``logits_to_keep=0`` computes every
position, which is the teacher-forced scoring path.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from ...shared.run_stats import RunStats
from ...huggingface import row_ids

logger = logging.getLogger(__name__)


def context_refusal(prompt_tokens: int, max_new_tokens: int | None, context_len: int):
    """Why a turn of this shape cannot fit ``context_len``, or None if it can.

    The prompt and the answer share ONE compiled KV cache, so a turn is impossible as soon
    as their sum exceeds it -- known before any device work, from three integers. Said up
    front because the alternative is where it used to be said: the prompt prefills, the
    answer streams, and generation dies at token ``context_len - prompt_tokens`` with a
    part-written reply and the tower's work already paid for. A 20-frame clip and
    ``--max-new-tokens 500`` against a 2048 context is that case.

    Answers ``None`` when there is nothing to compare: an unset ``max_new_tokens`` leaves
    the length to HF's own ``max_length``/stopping criteria, and guessing a bound for it
    here would refuse turns that fit. ``forward`` still checks each position it writes, so
    this is the earlier of two gates, not the only one.
    """
    if not max_new_tokens or prompt_tokens + max_new_tokens <= context_len:
        return None
    room = context_len - prompt_tokens
    if room <= 0:
        return (
            f"the prompt alone is {prompt_tokens} tokens against this artifact's "
            f"{context_len}-token context: use fewer images/frames, shorten the prompt, or "
            "rebuild the artifact with a larger --seq-len"
        )
    return (
        f"{prompt_tokens} prompt tokens + {max_new_tokens} max_new_tokens is "
        f"{prompt_tokens + max_new_tokens} against this artifact's {context_len}-token "
        f"context, so the answer would run out of cache after {room} token(s): lower "
        f"max_new_tokens to {room} or less, use fewer images/frames, or rebuild the "
        "artifact with a larger --seq-len"
    )


def _build_model_cls():  # noqa: C901 - one nested class; splitting it would only move lines
    """Define the shim against transformers (imported lazily, like evaluation.hf_model)."""
    from transformers import (
        AutoConfig,
        AutoModelForImageTextToText,
        AutoProcessor,
        GenerationConfig as HFGenerationConfig,
        PreTrainedModel,
    )
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.smolvlm.configuration_smolvlm import SmolVLMConfig
    from transformers.models.smolvlm.modeling_smolvlm import SmolVLMForConditionalGeneration
    from transformers.models.smolvlm.processing_smolvlm import SmolVLMProcessor

    class AxSmolVLMConfig(SmolVLMConfig):
        """SmolVLM's own config under a model type of ours.

        Everything in it is stock SmolVLM; only the name differs, and only so that the
        Auto classes can route an ARTIFACT directory to the device-backed model without
        hijacking plain SmolVLM checkpoints -- registering against ``smolvlm`` itself
        would send every ordinary ``from_pretrained`` in the process here.
        """

        model_type = "ax_smolvlm"

    class AxSmolVLMForConditionalGeneration(SmolVLMForConditionalGeneration):
        """SmolVLM2 backed by a compiled Axelera artifact (vision + connector + decoder)."""

        config_class = AxSmolVLMConfig

        def __init__(self, config):
            # Deliberately NOT SmolVLMForConditionalGeneration.__init__: that allocates
            # every torch parameter we would immediately throw away -- the weights are in
            # the artifact. PreTrainedModel.__init__ gives the HF plumbing (config,
            # generation_config, dtype/device resolution) without the tensors.
            PreTrainedModel.__init__(self, config)
            # One parameter so .device / .dtype resolve; the compute is off the torch graph.
            self._marker = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self._runner = None
            self._processor = None
            self._manifest: dict = {}
            #: the device's KV write position -- the only decode state the host tracks.
            self._position = 0
            #: what the last (or running) generation cost; filled in by `forward`.
            self.stats = RunStats()
            #: every finished generation, for session averages.
            self.history: list[RunStats] = []
            #: the one-off artifact load, paid before the first prompt. nan until loaded.
            self._load_s = float("nan")

        # ----- construction -----------------------------------------------------------

        @classmethod
        def from_artifact(cls, artifact_dir, repo_id: str | None = None, *, device=None):
            """Load a built artifact directory (text/ + vision/ + vlm.json).

            The config, the processor and the generation defaults come from the artifact
            itself (``save_hf_metadata`` writes them at build time), so a load touches no
            network. ``repo_id`` overrides that -- and is the fallback for artifacts built
            before those files were written. How MANY sub-devices is not asked for: the
            connection claims the whole generation's, which is the only useful answer (see
            ``multicore_impl.load_runner``). WHICH part they come from is ``device``, named
            the way ``--device`` spells it, defaulting to the first of the host.
            """
            from .multicore_impl import VLM_MANIFEST, load_runner

            artifact_dir = Path(artifact_dir)
            manifest = json.loads((artifact_dir / VLM_MANIFEST).read_text())
            recorded = manifest.get("repo_id")
            if repo_id is not None and recorded is not None and repo_id != recorded:
                # The checkpoint is not a preference. Its processor decides the tile size
                # the vision graphs were COMPILED at and the chat template the decoder was
                # calibrated on, so another one is a different prompt through the same
                # weights: on an artifact carrying its own processor the override could not
                # be honoured anyway (everything below reads `source`), and on one built
                # before that it IS honoured and quietly mis-tiles. Both are worth a word.
                raise ValueError(
                    f"this artifact was built from {recorded}, so it cannot be loaded as "
                    f"{repo_id}: the geometry its towers were compiled at and the chat "
                    "template its decoder was calibrated on are that checkpoint's. Drop "
                    "the override to use the artifact's own, or build an artifact for "
                    f"{repo_id}."
                )
            repo_id = repo_id or recorded
            if repo_id is None:
                raise ValueError(
                    f"no repo id recorded in {artifact_dir}; pass repo_id= explicitly"
                )
            # A config.json means save_hf_metadata wrote the HF files in (so no hub is
            # touched); artifacts built before that fall back to the recorded repo id.
            source = artifact_dir if (artifact_dir / "config.json").exists() else repo_id
            model = cls(as_ax_config(AutoConfig.from_pretrained(source)))
            model.generation_config = HFGenerationConfig.from_pretrained(source)
            # from_pretrained sets name_or_path to wherever it loaded from (the artifact
            # directory); the checkpoint it was BUILT from is the useful answer.
            model.config.name_or_path = repo_id
            model._repo_id = repo_id
            model._manifest = manifest
            model._processor = AutoProcessor.from_pretrained(source)
            # Timed: both towers' weights crossing PCIe is the one cost a demo watcher sees
            # before anything happens, and a report row carries it as `load_s`.
            started = perf_counter()
            try:
                model._runner = load_runner(artifact_dir, device=device)
            except Exception as failed:  # noqa: BLE001 - the driver's message names nothing
                raise RuntimeError(
                    f"could not open {artifact_dir} on the device: {failed}\n"
                    "The connection claims the whole generation's sub-devices, and the L2 "
                    "arena is shared across exactly those -- so an out-of-memory pool here "
                    "means the compiled context_len needs more L2 than the whole device has: "
                    "rebuild with a smaller --seq-len. Another process "
                    "already holding the device gives the same failure, since a device is "
                    "claimed whole by one connection."
                ) from failed
            model._load_s = perf_counter() - started
            model.stats.load_s = model._load_s
            model.eval()
            return model

        # `from_pretrained` on a directory that is an artifact, not a checkpoint: the
        # familiar spelling, routed to the loader that fits.
        @classmethod
        def from_pretrained(cls, path, *args, **kwargs):
            from .multicore_impl import VLM_MANIFEST

            if not (Path(path) / VLM_MANIFEST).exists():
                raise ValueError(
                    f"{path} is not a compiled SmolVLM2 artifact (no vlm.json). Build "
                    "one with the model's own entry (--build), or use "
                    "transformers.SmolVLMForConditionalGeneration for a plain HF checkpoint."
                )
            # AutoModelForImageTextToText.from_pretrained hands the class a config it
            # already parsed plus its whole hub/loading kwarg set (cache_dir, revision,
            # dtype, device_map, adapter_kwargs, ...). The artifact carries all of that
            # itself, so the ones this loader understands are kept and the rest dropped.
            # An ALLOWLIST, deliberately: a denylist goes stale the next time transformers
            # adds a kwarg, and it fails as a TypeError from inside the Auto factory.
            kept = {k: kwargs[k] for k in ("repo_id", "device") if k in kwargs}
            return cls.from_artifact(path, *args, **kept)

        #: What the other Axelera model wrappers call it; `from_pretrained` is the HF
        #: spelling of the same load.
        load_compiled = from_pretrained

        @property
        def runner(self):
            if self._runner is None:
                raise RuntimeError(
                    "construct via AxSmolVLMForConditionalGeneration.from_pretrained(artifact_dir)"
                )
            return self._runner

        @property
        def processor(self):
            """The model's own AutoProcessor -- chat template, tokenizer, image/video."""
            return self._processor

        @property
        def tokenizer(self):
            return self.runner.tokenizer or self._processor.tokenizer

        @property
        def context_len(self) -> int:
            """The compiled KV-cache capacity: prompt + answer must fit in it."""
            return int(self._manifest["context_len"])

        #: What the OTHER families' shims call the opened artifact, and what anything
        #: family-agnostic reaches through: `axzoo report` finds the device a row was
        #: measured on at `model.pipeline.text_runtime`. `runner` stays the name this
        #: family's own code uses; this is the shared spelling of it, not a second object.
        @property
        def pipeline(self):
            return self.runner

        @property
        def load_s(self) -> float:
            """Seconds the one-off artifact load cost, or nan if it was not timed."""
            return self._load_s

        @property
        def num_cores(self) -> int:
            """AI cores the compiled build spans, off the artifact's own manifest.

            Read, never asked for: the reservation is derived from the decoder build and a
            row that named a different count would be a row about another artifact.
            """
            return int(self._manifest.get("num_cores", 1))

        @property
        def generation(self) -> str:
            """The part this artifact was compiled for, off its manifest."""
            return str(self._manifest.get("generation", "OMEGA"))

        @property
        def repo_id(self) -> str | None:
            """The checkpoint the artifact was BUILT from, not where it was loaded."""
            return getattr(self, "_repo_id", None) or self._manifest.get("repo_id")

        def can_generate(self) -> bool:
            return True

        def get_input_embeddings(self):
            """None: the embedding table is in the artifact, not in torch.

            ``PreTrainedModel``'s version reaches for ``self.model.text_model.embed_tokens``,
            which this class never allocates, so the HF plumbing that merely ASKS
            (tied-weight bookkeeping, ``resize_token_embeddings``' guard) would hit an
            AttributeError instead of the answer "there is no torch embedding here".
            """
            return None

        def set_input_embeddings(self, value):
            raise NotImplementedError(
                "the embedding table lives in the compiled artifact; change it in the "
                "checkpoint and rebuild"
            )

        def save_pretrained(self, *args, **kwargs):
            """Refused: there are no torch weights here to save.

            The inherited version would happily write a checkpoint containing this class's
            single placeholder parameter -- a directory that looks like a model and is not
            one. The artifact IS the saved form; copy the directory.
            """
            raise NotImplementedError(
                "this model's weights are the compiled artifact, not torch parameters: "
                "copy the artifact directory, or rebuild with "
                "llm.models.smolvlm2.smolvlm2_compile"
            )

        # ----- prompt assembly from processor outputs ---------------------------------

        @staticmethod
        def _tiles(pixel_values):
            """A processor pixel stack -> the (blocks, 3, H, W) the vision graph encodes
            one dispatch at a time.

            The processor emits (batch, tiles, 3, H, W) for stills and (batch, frames,
            3, H, W) for video; both flatten to the same stack, which is why the device
            path is identical for the two.
            """
            return pixel_values.reshape(-1, *pixel_values.shape[-3:])

        def _turn_refusal(self, args, kwargs, config):
            """Why this ``generate`` call cannot fit, or None -- asked BEFORE any device work.

            The prompt length is measured the way ``forward`` will measure it (``_row_ids``,
            padding dropped), so the number in the message is the number that will be
            prefilled. A call that carries no ids to count -- ``inputs_embeds``, or a
            streamed continuation -- is not diagnosed here and falls back to the per-position
            check in ``forward``.
            """
            ids = kwargs.get("input_ids")
            if ids is None and args:
                ids = args[0]  # GenerationMixin.generate(inputs, ...)
            if not (torch.is_tensor(ids) or isinstance(ids, (list, tuple, np.ndarray))):
                return None
            asked = kwargs.get("max_new_tokens") or getattr(config, "max_new_tokens", None)
            _, row = row_ids(ids, kwargs.get("attention_mask"))
            return context_refusal(len(row), asked, self.context_len)

        def _check_room(self, length: int) -> None:
            if length > self.context_len:
                raise ValueError(
                    f"{length} tokens against the artifact's {self.context_len}-token "
                    "context: shorten the prompt, use fewer images/frames, lower "
                    "max_new_tokens, or rebuild the artifact with a larger --seq-len"
                )

        # ----- forward: the device, in the shape GenerationMixin.generate expects ------

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            pixel_values=None,
            pixel_attention_mask=None,
            image_hidden_states=None,
            cache_position=None,
            logits_to_keep=0,
            labels=None,
            **kwargs,
        ):
            """Logits for ``input_ids``, prefilling or stepping the device as HF asks.

            The whole model is this one method, which is what lets transformers' own
            ``generate()`` drive: it decides prefill vs decode from ``cache_position`` (the
            position the new tokens start at), exactly as it does for a torch model with a
            KV cache -- except the cache lives on the device, so nothing has to be handed
            back and forth. ``logits_to_keep=1`` (what generation asks for) computes the
            lm_head at the last position only; 0 computes every position, which is the
            teacher-forced scoring path.

            Image tokens enter only at prefill: the tower encodes every tile once, the host
            splices the rows into the ``<image>`` placeholders, and every decode step after
            that is pure text.

            The returned ``past_key_values`` is None on purpose: HF only needs it to be
            threaded through, and this device's cache cannot be inspected, reordered or
            duplicated (hence no beam search).
            """
            runner = self.runner
            impl, host = runner.text_impl, runner.host_params
            _, row = row_ids(input_ids, attention_mask)
            prefill = cache_position is None or int(cache_position[0]) == 0

            if prefill:
                self._check_room(len(row))
                stats = self.stats
                stats.prompt_tokens, stats.context_len = len(row), self.context_len
                embeds = None
                if pixel_values is not None:
                    # `_tiles` is a host reshape; `_image_embeds` is the dispatch loop, one
                    # per tile. Timed apart so the tower's cost is the device's and not the
                    # host's rearranging of the processor's stack.
                    started = perf_counter()
                    tiles = self._tiles(pixel_values)
                    dispatched = perf_counter()
                    embeds = runner._image_embeds(tiles)
                    finished = perf_counter()
                    stats.vision_dispatches += int(tiles.shape[0])
                    stats.vision_tokens += int(embeds.shape[0])
                    stats.vision_device_s += finished - dispatched
                    stats.vision_s += finished - started
                impl.set_image_embeds(embeds)
                # logits_to_keep=1 is generation asking for the next token; 0 is teacher-forced
                # scoring, which wants every position and so has to keep each chunk's hidden.
                want_all = logits_to_keep != 1
                kept = [] if want_all else None
                chunk_len = impl.select_prefill(
                    len(row),
                    self._manifest.get("chunk_len"),
                    self.context_len,
                    runner.text_runtime.graph_names,
                )
                started = perf_counter()
                try:
                    hidden, last_row, self._position = impl.ingest_prompt(
                        runner.text_runtime,
                        host,
                        row,
                        self.context_len,
                        chunk_len,
                        keep_hidden=kept,
                    )
                finally:
                    impl.set_image_embeds(None)  # decode steps are pure text
                ingested = perf_counter()
                logits = (
                    impl.compute_logits_seq(np.concatenate(kept, axis=1), host)
                    if want_all
                    else impl.compute_logits(hidden, host, position=last_row)
                )
                # `ingest_prompt` does its own host prep (embedding gather, mask, scatter)
                # inside the same call, so this family cannot split that out yet and
                # `prefill_prep_s` stays zero rather than being guessed at. `prefill_s` --
                # the sum, and what TTFT is built from -- is right either way; only the
                # device/host split WITHIN prefill is coarser than the other families'.
                stats.prefill_device_s += ingested - started
                stats.prefill_host_s += perf_counter() - ingested
                # logits_to_keep=1 is generation asking for a token; 0 is teacher-forced
                # scoring, which produces none.
                stats.prefill_generated = stats.prefill_generated or not want_all
            else:
                graph = impl._decode_graph_name()
                for token in row:  # generation feeds one; nothing stops a caller feeding more
                    self._check_room(self._position + 1)
                    # `step_inputs` is host work and is built BEFORE the timer rather than
                    # inline as an argument: evaluated in the call it would land inside the
                    # dispatch measurement and read as device time.
                    prepared = perf_counter()
                    step = impl.step_inputs(host, token, self._position, self.context_len)
                    started = perf_counter()
                    device_out = runner.text_runtime.run(graph, step)
                    dispatched = perf_counter()
                    logits = impl.compute_logits(device_out, host)
                    self._position += 1
                    self.stats.step_device_s.append(dispatched - started)
                    self.stats.step_host_s.append(
                        (started - prepared) + (perf_counter() - dispatched)
                    )

            logits = np.asarray(logits, dtype=np.float32)
            out = torch.from_numpy(np.ascontiguousarray(logits.reshape(1, -1, logits.shape[-1])))
            loss = None
            if labels is not None:
                loss = torch.nn.functional.cross_entropy(
                    out[:, :-1].reshape(-1, out.shape[-1]),
                    labels[:, 1:].reshape(-1).to(torch.long),
                    ignore_index=-100,
                )
            return CausalLMOutputWithPast(loss=loss, logits=out, past_key_values=None)

        # ----- generate: HuggingFace's, with only the device bookkeeping added ---------

        def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            cache_position=None,
            pixel_values=None,
            pixel_attention_mask=None,
            image_hidden_states=None,
            **kwargs,
        ):
            """What each generate() step passes to forward: the NEW tokens only.

            The stock SmolVLM version slices against a torch cache and re-attaches the
            image hidden states; here the device holds the cache and the tower ran with the
            prompt, so this only has to say "prefill" or "one more token".
            """
            prefill = cache_position is None or int(cache_position[0]) == 0
            inputs = {
                "input_ids": input_ids if prefill else input_ids[:, -1:],
                "cache_position": cache_position,
                "logits_to_keep": 1,  # generation only ever reads the last position
            }
            if prefill:  # the pictures ride on the prompt, and only on the prompt
                inputs.update(
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    pixel_attention_mask=pixel_attention_mask,
                )
            return inputs

        @torch.no_grad()
        def generate(self, *args, **kwargs):
            """``GenerationMixin.generate``, unchanged.

            Sampling (do_sample, temperature, top_k, top_p, min_p, repetition_penalty),
            stopping criteria, streamers and the returned id tensor are all transformers'
            own -- this override exists only to refuse what one device cache cannot do, and
            to reset the KV write position afterwards.
            """
            config = kwargs.get("generation_config") or self.generation_config
            for flag in ("num_beams", "num_return_sequences"):
                asked = int(kwargs.get(flag, getattr(config, flag, 1)) or 1)
                if asked > 1:
                    raise NotImplementedError(
                        f"{flag}={asked} needs several KV caches; the device has one, and "
                        "it cannot be reordered or duplicated"
                    )
            refusal = self._turn_refusal(args, kwargs, config)
            if refusal is not None:
                raise ValueError(refusal)
            # A fresh block BEFORE the run: `forward` accumulates into it, and a second
            # generate() on the same instance must not add to the first one's totals.
            self.stats = RunStats(load_s=self._load_s, context_len=self.context_len)
            started = perf_counter()
            try:
                return super().generate(*args, **kwargs)
            finally:
                # Measured around generate() and not summed from the stages: transformers
                # samples, runs the logits processors and drives the streamer between the
                # forwards, and `other_s` is what that costs.
                self.stats.wall_s = perf_counter() - started
                self.history.append(self.stats)
                self._position = 0

    def as_ax_config(config):
        """A SmolVLM config (from anywhere) as this model's config type."""
        if isinstance(config, AxSmolVLMConfig):
            return config
        data = config.to_dict()
        data.pop("model_type", None)  # the class attribute is the one that counts
        return AxSmolVLMConfig(**data)

    # The Auto classes, so an artifact directory loads the way any HF model does:
    # `AutoModelForImageTextToText.from_pretrained(artifact_dir)` (this module has to be
    # imported first -- that is the documented registration route, and it is why nothing
    # here needs trust_remote_code). exist_ok: a re-import must not explode.
    AutoConfig.register(AxSmolVLMConfig.model_type, AxSmolVLMConfig, exist_ok=True)
    AutoModelForImageTextToText.register(
        AxSmolVLMConfig, AxSmolVLMForConditionalGeneration, exist_ok=True
    )
    AutoProcessor.register(AxSmolVLMConfig, SmolVLMProcessor, exist_ok=True)

    return AxSmolVLMConfig, AxSmolVLMForConditionalGeneration, as_ax_config


try:
    AxSmolVLMConfig, AxSmolVLMForConditionalGeneration, as_ax_config = _build_model_cls()
except ImportError:  # transformers absent: the rest of the suite still imports
    AxSmolVLMConfig = AxSmolVLMForConditionalGeneration = as_ax_config = None


def register_auto() -> None:
    """Teach the transformers Auto classes about SmolVLM2's model type, so an artifact resolves.

    Call it once before `AutoModelForImageTextToText.from_pretrained(artifact_dir)`. One
    spelling for every AxScript family, text included: a customer moving between them should
    not have to learn a second one, and the failure when they use the other family's is
    "Transformers does not recognize this architecture", which names nothing they can act on.

    Importing this module registers too, wherever transformers is installed. Where it is not,
    the import leaves the names None and this call raises the ImportError that says so.
    """
    global AxSmolVLMConfig, AxSmolVLMForConditionalGeneration, as_ax_config

    if AxSmolVLMForConditionalGeneration is None:
        AxSmolVLMConfig, AxSmolVLMForConditionalGeneration, as_ax_config = _build_model_cls()


def artifact_config(repo_id: str):
    """The config to write into an artifact: the checkpoint's, under our model type.

    Same fields as the HF one -- ``save_hf_metadata`` writes this so the artifact loads
    through the Auto classes without claiming to be a plain SmolVLM checkpoint.
    """
    from transformers import AutoConfig

    config = as_ax_config(AutoConfig.from_pretrained(repo_id))
    # Provenance for a reader of the saved config.json. It does NOT survive a load --
    # from_pretrained resets name_or_path to wherever it loaded from -- so from_artifact
    # restores it from vlm.json's repo_id, which is the authoritative record.
    config.name_or_path = repo_id
    return config


def save_hf_metadata(out_dir, repo_id: str) -> None:
    """Write the HuggingFace-side files into a built artifact.

    This is what makes the artifact directory something the stock Auto classes understand
    -- ``AutoConfig.from_pretrained(dir)``, ``AutoProcessor.from_pretrained(dir)``,
    ``AutoModelForImageTextToText.from_pretrained(dir)`` -- and what lets a built model
    load with no hub and no network at all.

    The saved processor is pinned to the tile resolution the vision tower was COMPILED for.
    That is not a free parameter: the exported tower has a static (image_size // patch_size)
    patch count and a learned position table of exactly that many entries, so a processor
    left at a different ``max_image_size`` would hand the device a stack it cannot encode.
    Note the knob is ``max_image_size`` (the per-tile resolution) -- ``size.longest_edge`` is
    only the pre-tile resize cap, so setting that instead silently leaves the tile alone.
    """
    from transformers import AutoConfig, AutoProcessor, GenerationConfig as HFGenerationConfig

    out_dir = Path(out_dir)
    processor = AutoProcessor.from_pretrained(repo_id)
    tile_px = int(AutoConfig.from_pretrained(repo_id).vision_config.image_size)
    if processor.image_processor.max_image_size["longest_edge"] != tile_px:
        processor.image_processor.max_image_size = {"longest_edge": tile_px}
    processor.save_pretrained(out_dir)
    artifact_config(repo_id).save_pretrained(out_dir)
    # Sampling defaults (temperature, top_p, the eos ids) travel with the artifact too:
    # generate() reads them, and a customer's script never mentions the hub repo. Exactly
    # what the checkpoint ships, unmodified -- the artifact should behave like the model it
    # was built from.
    HFGenerationConfig.from_pretrained(repo_id).save_pretrained(out_dir)

# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""The HuggingFace side of a compiled AxScript LLM: what makes an artifact a model directory.

A customer runs an AxScript LLM with the code they already have, which means the artifact
directory has to load through the stock Auto classes with no hub, no network and no
`trust_remote_code`:

    >>> from model_axscript.llm.huggingface import register_auto
    >>> register_auto()   # teaches the Auto classes our model type; see its docstring
    >>> from transformers import AutoModelForCausalLM, AutoTokenizer
    >>> model = AutoModelForCausalLM.from_pretrained("build/qwen3-0.6b/axscript")
    >>> tokenizer = AutoTokenizer.from_pretrained("build/qwen3-0.6b/axscript")

`save_hf_metadata` is what puts the three files that needs into a built artifact: the config
under our own model type, the tokenizer with its chat template, and the checkpoint's own
generation defaults. It is called once, where `pipeline.json` is written, so every text
implementation gets it without per-model work.

Our own model type (`ax_causal_lm`), not the checkpoint's, on purpose. An artifact is not a
Qwen3 checkpoint: its weights are compiled device blobs, and a directory claiming
`model_type: qwen3` would be handed to stock transformers, which would then look for
safetensors that are not there. Registering a type of our own is also what removes any need
for `trust_remote_code`, because the Auto classes resolve the class from the registry rather
than by importing code out of the artifact.

This module is customer-facing and deliberately sits beside the engine rather than under
`evaluation/`: the same presentation serves a script, a notebook and the eval suite.
`transformers` is imported inside the functions, so the module can be imported (and the
engine built) in an environment that does not carry it.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

#: The model type an AxScript text artifact declares. Stable: a customer's directory keeps
#: working across releases, and stock transformers keeps its own `qwen3` / `llama` types.
MODEL_TYPE = "ax_causal_lm"

_UNKNOWN_ARCH = "does not recognize this architecture"

_ONE_SEQUENCE = (
    "one sequence at a time: the device holds a single KV cache, so a batch above 1 would "
    "have the rows overwrite each other. Loop the prompts, or use batch_size=1."
)


def kept_positions(attention_mask, bi: int, seq: int) -> list[int]:
    """The positions of row `bi` that the mask keeps, in order (every one if there is no mask).

    `torch` is imported here rather than at module scope, as everywhere else in this module,
    so the engine can be built where transformers and torch are not installed.
    """
    import torch

    if attention_mask is None:
        return list(range(seq))
    row = attention_mask.reshape(-1, seq)[bi]
    return torch.nonzero(row > 0, as_tuple=False).flatten().tolist()


def row_ids(input_ids, attention_mask):
    """The single sequence in the batch, as (ids, its tokens with the padding dropped).

    Shared by every adapter. A batch above one has to be refused the same way whichever model
    the caller loaded, and the prompt has to be counted the same way, because that count is
    what the context refusal quotes and what the prefill then ingests.
    """
    import torch

    ids = input_ids if torch.is_tensor(input_ids) else torch.as_tensor(input_ids)
    ids = ids.unsqueeze(0) if ids.dim() == 1 else ids
    if ids.shape[0] != 1:
        raise NotImplementedError(f"batch of {ids.shape[0]}, but {_ONE_SEQUENCE}")
    return ids, [int(ids[0, i]) for i in kept_positions(attention_mask, 0, ids.shape[1])]


def hf_metadata_dir(artifact_dir) -> Path | None:
    """The artifact directory itself when `save_hf_metadata` wrote HF files into it.

    One check ("is there a config.json?") shared by everything that would otherwise reach
    for the hub. An artifact built before this returns None, and those callers fall back to
    the repo id recorded in `pipeline.json`.
    """
    path = Path(artifact_dir)
    return path if (path / "config.json").is_file() else None


def text_config(repo_id: str, fallback=None):
    """A `PretrainedConfig` for `repo_id`, even when transformers cannot read its config.

    Two problems in one call. A multimodal-style config (Qwen3.5) nests the fields a causal
    LM consumer reads under `text_config`, and `get_text_config()` returns that sub-config or
    self. Separately, `AutoConfig.from_pretrained` RAISES for a checkpoint whose `model_type`
    the pinned transformers does not carry, which is the case for `qwen3_5` and is expected
    to keep happening: supporting architectures ahead of that pin is the point of this tree.

    `fallback` is the model tree's own config object, used only for that second case. Without
    one an unreadable config stays an error rather than becoming a guess.

    `repo_id` can be an ARTIFACT directory, which declares our own model type, so that type
    is registered before the read. Unregistered, it reads as a `model_type` transformers does
    not carry, and a full artifact config would take the fallback path above and lose every
    field the fallback does not name.
    """
    from transformers import AutoConfig, PretrainedConfig

    config_cls()  # registers `ax_causal_lm`, so an artifact directory reads back in full

    try:
        return AutoConfig.from_pretrained(repo_id).get_text_config()
    except (ValueError, KeyError) as e:
        if _UNKNOWN_ARCH not in str(e) or fallback is None:
            raise
    # Only the fields a HuggingFace consumer reads off a causal-LM config. Anything else
    # would be a guess dressed as data. Said out loud, because every field the checkpoint
    # carried and this does not is now absent from the config a caller holds, and the
    # absence otherwise surfaces far from here as an `AttributeError` inside transformers.
    logger.warning(
        "[axscript] %s could not be read by the installed transformers, so its config is "
        "reduced to the fields the model tree states: vocab_size, hidden_size, "
        "num_hidden_layers, max_position_embeddings, tie_word_embeddings.",
        repo_id,
    )
    # A field the model tree does not state is OMITTED, never set to None. A consumer
    # reads these as `getattr(config, name, DEFAULT)`, and a None present under the name
    # wins over that default: the caller then computes with None instead of with the value
    # it declared as its fallback, which is worse than the attribute being absent.
    stated = {
        name: int(value)
        for name, value in (
            ("hidden_size", getattr(fallback, "hidden_size", 0)),
            ("num_hidden_layers", getattr(fallback, "num_hidden_layers", 0)),
            # transformers reads this one DIRECTLY in places, not through `getattr`, so
            # stating it is what keeps a reduced config usable at all.
            ("max_position_embeddings", getattr(fallback, "max_position_embeddings", 0)),
        )
        if value
    }
    return PretrainedConfig(
        vocab_size=int(fallback.vocab_size),
        tie_word_embeddings=bool(getattr(fallback, "tie_word_embeddings", False)),
        use_cache=False,
        **stated,
    )


def artifact_fallback_config(artifact_dir, pipeline: dict):
    """The config the artifact was BUILT from, or None when it does not carry one.

    `quantized_model/quantized_model.json` holds the config dict the kernels were compiled
    against, and the architecture reconstructs its own config object from it. Preferred over
    the hub as a fallback precisely because it cannot drift from the graphs being executed: a
    checkpoint can be re-tagged upstream, the artifact cannot.

    Returns None rather than raising, so a missing or older artifact degrades to the existing
    behaviour instead of gaining a new failure mode.
    """
    from .models.architecture import get_architecture

    path = Path(artifact_dir) / "quantized_model" / "quantized_model.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text())
        arch = get_architecture(data.get("architecture") or pipeline["architecture"])
        return arch.config_from_dict(data["config"])
    except (KeyError, ValueError, OSError):
        return None


_CONFIG_CLS = None


def config_cls():
    """The config CLASS an AxScript text artifact declares, defined on first use.

    A class, not an instance with `model_type` assigned: `save_pretrained` serialises
    `model_type` off the CLASS attribute, so an instance-level assignment is dropped on the
    way to disk and the written config.json carries no type at all. Registering it on
    `AutoConfig` is what lets `AutoConfig.from_pretrained(artifact_dir)` resolve without
    `trust_remote_code`, and registering our own type rather than hijacking `llama` or
    `qwen3` keeps ordinary checkpoints going to stock transformers.

    Built lazily, like the adapter classes, so this module imports without transformers.
    """
    global _CONFIG_CLS
    if _CONFIG_CLS is not None:
        return _CONFIG_CLS

    from transformers import AutoConfig, PretrainedConfig

    class AxCausalLMConfig(PretrainedConfig):
        """A compiled AxScript causal LM: the checkpoint's shape, under our own type."""

        model_type = MODEL_TYPE

    AutoConfig.register(MODEL_TYPE, AxCausalLMConfig, exist_ok=True)
    _CONFIG_CLS = AxCausalLMConfig
    return _CONFIG_CLS


def artifact_config(repo_id: str, fallback=None):
    """The config to WRITE into an artifact: the checkpoint's fields, under our model type.

    Same fields a HuggingFace consumer expects. `model_type` comes from the class (see
    `config_cls`), and `use_cache` is off because the device owns its own KV cache and never
    hands one back, so a caller must not be told that one is available.
    """
    data = text_config(repo_id, fallback=fallback).to_dict()
    # The class attribute is the one that counts, and the rest describe the CHECKPOINT: an
    # artifact is not a `LlamaForCausalLM` and was not written by a transformers version.
    for dropped in ("model_type", "_name_or_path", "architectures", "transformers_version"):
        data.pop(dropped, None)
    config = config_cls()(**data)
    config.use_cache = False
    config.name_or_path = repo_id  # provenance: what this artifact was built from
    return config


def save_hf_metadata(out_dir, repo_id: str | None) -> bool:
    """Write the HuggingFace-side files into a built artifact. Returns whether it wrote.

    Three files, each answering one thing a caller would otherwise need the hub for:
    `config.json` (the architecture's shape, under our model type), the tokenizer files
    (including the chat template, so `apply_chat_template` works offline), and
    `generation_config.json` (the checkpoint's own sampling defaults, so a script that never
    names the hub repo still behaves like the model it was built from).

    Best effort by design, and best effort against EVERY failure rather than the ones that
    were foreseen. A build whose `repo_id` is unknown, or an environment without
    `transformers`, writes nothing and says so at INFO. Anything else that goes wrong here is
    caught too and reported at WARNING: a checkpoint whose `model_type` the pinned
    transformers cannot read raises out of `text_config` (`qwen3_5`, and this is expected to
    keep happening), and an unreachable hub raises `OSError` out of `load_tokenizer`. The
    artifact is still complete for every existing caller, which resolves the tokenizer from
    `pipeline.json`'s `repo_id`; only the offline HuggingFace entry point is missing. Failing
    the compile here would throw away a multi-minute build over its last, optional step.
    """
    if not repo_id:
        logger.info("[axscript] no repo_id recorded, so no HuggingFace metadata was written")
        return False
    out_dir = Path(out_dir)
    try:
        from transformers import GenerationConfig as HFGenerationConfig

        from .generate import load_tokenizer

        # `load_tokenizer`, not `AutoTokenizer`: it owns the fallback for a checkpoint whose
        # `tokenizer_config.json` names a `tokenizer_class` the pinned transformers cannot
        # import (LFM2.5's `TokenizersBackend`), and it carries the chat template through.
        # Saving what AutoTokenizer would have raised on is the whole point.
        load_tokenizer(repo_id).save_pretrained(out_dir)
        artifact_config(repo_id).save_pretrained(out_dir)
        try:
            HFGenerationConfig.from_pretrained(repo_id).save_pretrained(out_dir)
        except OSError:  # not every checkpoint ships one
            logger.info("[axscript] %s ships no generation_config", repo_id)
    except ImportError as e:
        logger.info("[axscript] no HuggingFace metadata written (%s)", e)
        return False
    except Exception as e:  # noqa: BLE001 - the step is optional, the build behind it is not
        logger.warning(
            "[axscript] no HuggingFace metadata written (%s: %s). The artifact is complete; "
            "only `from_pretrained` on the directory is unavailable.",
            type(e).__name__,
            e,
        )
        return False
    logger.info("[axscript] wrote the HuggingFace metadata into %s", out_dir)
    return True


_MODEL_CLS = None


def model_cls():
    """The `PreTrainedModel` an AxScript text artifact presents itself as, defined on first use.

    Generation is `GenerationMixin.generate` itself, not a lookalike. This class implements two
    methods, `forward` (prefill when the step starts at position 0, one decode step otherwise)
    and `prepare_inputs_for_generation`, and transformers does everything else: temperature,
    top-k, top-p, min-p, repetition penalty, any caller-supplied logits processor, stopping
    criteria, streamers, seeding, and the returned id tensor. Re-implementing those would mean
    tracking a library we already depend on for the tokenizer and the chat template.

    The device's KV cache sets the limits. It cannot be inspected, reordered or duplicated, so
    `past_key_values` comes back None, and beam search, `num_return_sequences` above 1, a batch
    above 1 and assisted or contrastive decoding raise rather than quietly misbehave.

    Built lazily so this module imports without torch or transformers.
    """
    global _MODEL_CLS
    if _MODEL_CLS is not None:
        return _MODEL_CLS

    import numpy as np
    import torch
    from torch import nn
    from transformers import AutoModelForCausalLM, PreTrainedModel
    from transformers.generation import GenerationMixin
    from transformers.modeling_outputs import CausalLMOutputWithPast

    class AxeleraForCausalLM(PreTrainedModel, GenerationMixin):
        """A causal LM whose weights live in a compiled artifact instead of in torch."""

        config_class = config_cls()
        main_input_name = "input_ids"
        _supports_cache_class = False
        supports_gradient_checkpointing = False

        def __init__(self, config):
            super().__init__(config)
            # One tiny parameter so `.device` and `.dtype` resolve and a harness can "move"
            # the model. The real compute is on the AIPU, off the torch graph entirely.
            self._marker = nn.Parameter(torch.zeros(1), requires_grad=False)
            self._runtime_ready = False
            # Where the device will write next. Every prefill sets it; see `forward`.
            self._position = 0

        # ----- construction ---------------------------------------------------------

        @classmethod
        def from_artifact(
            cls, artifact_dir, repo_id=None, *, num_sub_devices=None, device=None, **gen_kwargs
        ):
            """Build from a PREBUILT artifact directory.

            `repo_id` (for the config and the tokenizer) defaults to the one recorded in the
            artifact's pipeline.json. `num_sub_devices` defaults to the cores the build itself
            spans, never to a constant: a 4-core artifact given one sub-device dies as
            DEVICE_LOST, and an 8-core Europa build needs eight.

            `device` names the Metis part to run on, by name (`metis-0:2d:0`) or index, the way
            the command line's `--device` takes it. It defaults to `$AXELERA_METIS_DEVICE` and
            then to the FIRST part, which is what a host with one wants and what a host with
            several does not: two processes cannot otherwise be placed apart.
            """
            from ..shared.artifact_contract import decoder_tower
            from ..shared.runtime import RuntimeModel
            from .shared.config import GenerationConfig
            from .generate import artifact_repo_id, load_tokenizer
            from .implementation import get_implementation

            artifact_dir = decoder_tower(artifact_dir)
            pipeline = json.loads((artifact_dir / "pipeline.json").read_text())
            if repo_id is None:
                repo_id = artifact_repo_id(artifact_dir)
                if repo_id is None:
                    raise ValueError(
                        f"no repo_id recorded in {artifact_dir}; pass repo_id= explicitly"
                    )

            # The artifact's own config.json when the build wrote one, so a load needs no hub
            # and cannot disagree with the graphs being run; the checkpoint otherwise.
            local = hf_metadata_dir(artifact_dir)
            source = str(local) if local else repo_id
            config = artifact_config(
                source, fallback=artifact_fallback_config(artifact_dir, pipeline)
            )
            model = cls(config)
            model._artifact_dir = artifact_dir
            model._pipeline = pipeline
            model._context_len = int(pipeline["context_len"])
            model._chunk_len = pipeline.get("chunk_len")
            model._impl = get_implementation(pipeline["implementation"])
            model._tokenizer = load_tokenizer(source)
            model._gen_config = GenerationConfig(
                implementation=pipeline["implementation"],
                repo_id=repo_id,
                context_len=model._context_len,
                chunk_len=model._chunk_len,
                **gen_kwargs,
            )
            # Sampling defaults and the stop-token set travel with the artifact, so stopping
            # is HuggingFace's own and a script that never names the hub repo still behaves
            # like the model it was built from.
            model.generation_config = _generation_config(source, model.generation_config)
            # Then whatever the load named, onto the config `generate()` reads. `_gen_config`
            # above is the AxScript dataclass and only the regression harness reads it, so
            # setting that alone would take a temperature and sample at the artifact's own.
            _apply_generation_defaults(model.generation_config, **gen_kwargs)

            from .generate import _load_host_params

            model._host_params = _load_host_params(artifact_dir, pipeline)
            model._runtime = RuntimeModel(
                artifact_dir / pipeline.get("device_model_json", "model.json"),
                num_sub_devices=None if num_sub_devices is None else int(num_sub_devices),
                device=device,
            )
            model._runtime_ready = True
            model.eval()
            return model

        @classmethod
        def from_pretrained(cls, path, *args, **kwargs):
            """`from_pretrained` on a directory that is an ARTIFACT, not a checkpoint.

            The familiar spelling, routed to the loader that fits. Everything this loader does
            not itself declare is dropped rather than made into a second code path: the
            artifact already carries what the hub kwargs would resolve.

            The kept set is stated as an ALLOW list, `_LOAD_KWARGS`. A deny list would have to
            name every keyword the installed transformers happens to inject, and
            `AutoModelForCausalLM` adds `adapter_kwargs`, `code_revision` and `_from_auto` on
            top of the hub ones, so a deny list breaks again on the next version. The cost of
            the allow list is that a misspelled keyword is dropped rather than reported: a
            caller's typo and a library's injection are indistinguishable here.
            """
            if not (Path(path) / "pipeline.json").is_file():
                raise ValueError(
                    f"{path} is not a compiled AxScript LLM artifact (no pipeline.json). "
                    "Build one with `python -m model_axscript.llm build <model>`, or use "
                    "transformers.AutoModelForCausalLM for a plain HuggingFace checkpoint."
                )
            kept = {k: v for k, v in kwargs.items() if k in _LOAD_KWARGS}
            return cls.from_artifact(path, *args, **kept)

        #: What the other Axelera model wrappers call it; `from_pretrained` is the
        #: HuggingFace spelling of the same load.
        load_compiled = from_pretrained

        @property
        def tokenizer(self):
            return self._tokenizer

        def release(self) -> None:
            """Give back the sub-devices this model reserved. Idempotent.

            The adapter holds a `RuntimeModel`, and a `RuntimeModel` holds a device
            reservation that NOTHING frees on garbage collection: there is no finaliser on
            the runtime context. A caller that scores two artifacts in one process (a
            multi-size `axzoo report`, a 1-core baseline beside a 4-core shard) therefore
            asks for the second reservation while the first is still held, and gets
            `requested num_sub_devices(4) > available (4)`, which reads as a part too small
            for the build rather than a model still holding the part, exactly as
            `shared.runtime.release_runtime` warns.

            Not a `__del__`: collection order is not a lifecycle, and the caller that
            finished scoring is the one that knows the device is free to give back.
            """
            from ..shared.runtime import release_runtime

            runtime, self._runtime = getattr(self, "_runtime", None), None
            self._runtime_ready = False
            if runtime is not None:
                release_runtime(runtime)

        def can_generate(self) -> bool:
            return True

        def get_input_embeddings(self):
            # The embedding is an fp16 host table inside the artifact, not an nn.Module, and
            # nothing HuggingFace does with it here would reach the device.
            return None

        def set_input_embeddings(self, value):
            raise NotImplementedError(
                "the embedding table lives in the compiled artifact; rebuild to change it"
            )

        def save_pretrained(self, *args, **kwargs):
            raise NotImplementedError(
                "there are no torch weights to save: the model IS the artifact directory. "
                "Copy it, or rebuild with `python -m model_axscript.llm build <model>`."
            )

        # ----- the two methods that let HuggingFace drive ---------------------------

        def _check_room(self, length: int) -> None:
            if length > self._context_len:
                raise ValueError(
                    f"{length} tokens against the artifact's {self._context_len}-token "
                    "context: shorten the prompt, lower max_new_tokens, or build the "
                    "artifact with a larger context_len"
                )

        def _row_logits(self, ids: list[int]) -> np.ndarray:
            """Per-position logits [len(ids), vocab] for one sequence (left-truncated to ctx).

            Uses the PREFILL graph: one device call per window, every position's hidden at
            once. `$AXSCRIPT_SCORE_GRAPH=decode` scores through the DECODE graph instead, one
            device call per POSITION. Far slower, and a diagnostic rather than a second way to
            report a number: the two graphs are meant to agree, so a number that differs
            between them is itself the finding. It exists because a build can have a sound
            decode kernel and a broken prefill one, and then this is the only way to learn
            what the quantized weights are worth.
            """
            import os

            ids = ids[-self._context_len :]
            if os.environ.get("AXSCRIPT_SCORE_GRAPH", "").strip().lower() == "decode":
                return np.stack(
                    [
                        np.asarray(
                            self._impl.decode_step(
                                self._runtime, self._host_params, int(t), pos, self._context_len
                            )
                        ).reshape(-1)
                        for pos, t in enumerate(ids)
                    ]
                )
            hidden = self._impl.prefill_hidden(
                self._runtime, self._host_params, ids, self._context_len, self._chunk_len
            )
            return np.asarray(
                self._impl.compute_logits_seq(
                    hidden[:, : len(ids)], self._host_params, runtime=self._runtime
                )
            )

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            cache_position=None,
            logits_to_keep=0,
            labels=None,
            return_dict=None,
            **kwargs,
        ):
            """Logits for `input_ids`, prefilling or stepping the device as HuggingFace asks.

            Prefill against decode comes from `cache_position`, exactly as for a torch model
            with a KV cache, except the cache lives on the device so nothing is handed back and
            forth. The ROUTING is all `cache_position` decides: the write cursor a decode step
            uses is `self._position`, what the prefill reported writing, because the two differ
            whenever `attention_mask` drops a submitted token. `logits_to_keep=1` (what
            generation asks for) computes the lm_head at the last position only; 0 computes
            every position, which is the teacher-forced scoring path a log-likelihood harness
            reads.

            `past_key_values` comes back None on purpose: HuggingFace only threads it through,
            and this cache cannot be inspected, reordered or duplicated.
            """
            if not self._runtime_ready:
                raise RuntimeError("construct via AxeleraForCausalLM.from_artifact(...)")
            ids = input_ids if torch.is_tensor(input_ids) else torch.as_tensor(input_ids)
            if ids.dim() == 1:
                ids = ids.unsqueeze(0)
            batch, seq = ids.shape
            prefill = cache_position is None or int(cache_position[0]) == 0

            if not prefill:
                # One decode step. `prepare_inputs_for_generation` sends the new token only,
                # and the host tracks the write cursor. NOT from `cache_position`, which counts
                # every position HuggingFace SUBMITTED: a mask that drops padding leaves the
                # device holding fewer, so writing at it runs past the prompt's real tokens and
                # attends over slots nothing ever wrote. Every buffer is still the right size,
                # so no shape check sees it. What the prefill returned is the cursor.
                if batch != 1:
                    raise NotImplementedError(_ONE_SEQUENCE)
                self._check_room(self._position + 1)
                logits = self._impl.decode_step(
                    self._runtime,
                    self._host_params,
                    int(ids[0, -1]),
                    self._position,
                    self._context_len,
                )
                self._position += 1
                out = torch.from_numpy(
                    np.asarray(logits, dtype=np.float32).reshape(1, 1, -1)
                ).clone()
                return CausalLMOutputWithPast(loss=None, logits=out, past_key_values=None)

            if logits_to_keep == 1:  # generation's prefill: the next token, from one prompt
                if batch != 1:
                    raise NotImplementedError(_ONE_SEQUENCE)
                prompt = [int(ids[0, i]) for i in kept_positions(attention_mask, 0, seq)]
                if not prompt:
                    # Checked here rather than left to `ingest_prompt`, so the message names
                    # the mask. `_check_room` cannot catch it: the kept length is 0, which
                    # fits any context.
                    raise ValueError(
                        f"attention_mask leaves none of the {seq} submitted tokens, so there "
                        "is no prompt to ingest"
                    )
                self._check_room(len(prompt))
                hidden, row, self._position = self._impl.ingest_prompt(
                    self._runtime, self._host_params, prompt, self._context_len, self._chunk_len
                )
                logits = self._impl.compute_logits(
                    hidden, self._host_params, position=row, runtime=self._runtime
                )
                out = torch.from_numpy(
                    np.asarray(logits, dtype=np.float32).reshape(1, 1, -1)
                ).clone()
                return CausalLMOutputWithPast(loss=None, logits=out, past_key_values=None)

            # Teacher-forced scoring: every position, one prefill call per sequence. Padding
            # positions (attention_mask == 0) stay zero; callers gather the real ones by index.
            vocab = int(self.config.vocab_size)
            out = torch.zeros((batch, seq, vocab), dtype=torch.float32)
            for bi in range(batch):
                keep = kept_positions(attention_mask, bi, seq)
                if not keep:
                    continue
                kept_ids = [int(ids[bi, i]) for i in keep]
                keep = keep[-self._context_len :]  # mirror the left-truncation in _row_logits
                out[bi, keep, :] = torch.from_numpy(self._row_logits(kept_ids).astype(np.float32))

            loss = None
            if labels is not None:
                loss = nn.functional.cross_entropy(
                    out[:, :-1, :].reshape(-1, vocab),
                    labels[:, 1:].reshape(-1).to(torch.long),
                    ignore_index=-100,
                )
            return CausalLMOutputWithPast(loss=loss, logits=out, past_key_values=None)

        def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            cache_position=None,
            **kwargs,
        ):
            """What each generate() step passes to forward: the NEW tokens only.

            The stock version slices against a torch cache. Here the device holds the cache, so
            this only has to say "prefill" or "one more token".
            """
            prefill = cache_position is None or int(cache_position[0]) == 0
            inputs = {
                "input_ids": input_ids if prefill else input_ids[:, -1:],
                "cache_position": cache_position,
                "logits_to_keep": 1,  # generation only ever reads the last position
            }
            if prefill:
                inputs["attention_mask"] = attention_mask
            return inputs

        @torch.no_grad()
        def generate(self, *args, **kwargs):
            """`GenerationMixin.generate`, unchanged.

            Sampling, stopping criteria, streamers and the returned id tensor are all
            transformers' own. This override only refuses what one device cache cannot do,
            rather than accepting the parameter and ignoring it.
            """
            config = kwargs.get("generation_config") or self.generation_config
            for flag in ("num_beams", "num_return_sequences"):
                asked = int(kwargs.get(flag, getattr(config, flag, 1)) or 1)
                if asked > 1:
                    raise NotImplementedError(
                        f"{flag}={asked} needs several KV caches; the device has one, and it "
                        "cannot be reordered or duplicated"
                    )
            if kwargs.get("assistant_model") is not None:
                raise NotImplementedError(
                    "assisted and speculative decoding need a cache that can be rolled back; "
                    "this one cannot"
                )
            alpha = kwargs.get("penalty_alpha", getattr(config, "penalty_alpha", None))
            if alpha:
                raise NotImplementedError(
                    "contrastive search needs the hidden states of every candidate and a cache "
                    "that can be reordered; this device gives neither"
                )
            ids = kwargs.get("input_ids", args[0] if args else kwargs.get("inputs"))
            if torch.is_tensor(ids) and ids.dim() > 1 and ids.shape[0] > 1:
                raise NotImplementedError(_ONE_SEQUENCE)
            return super().generate(*args, **kwargs)

    AutoModelForCausalLM.register(config_cls(), AxeleraForCausalLM, exist_ok=True)
    _MODEL_CLS = AxeleraForCausalLM
    return _MODEL_CLS


#: What a LOAD may set. Everything else about a built artifact is fixed at build time, so the
#: rest of `GenerationConfig` (the calibration set, the core count, the phases) would be
#: accepted and then do nothing. Listed rather than derived from the dataclass for that reason:
#: the two sets are different, and deriving one from the other says they are the same.
_LOAD_KWARGS = frozenset(
    {
        "repo_id",  # the config and tokenizer to read, when not the one the artifact records
        "num_sub_devices",  # the reservation to open, when wider than the build needs
        "device",  # the Metis part, as `--device` takes it
        # The three generation defaults a caller may carry in. Each one reaches
        # `generation_config`, so `generate()` honours it; see `_apply_generation_defaults`.
        "max_new_tokens",
        "temperature",
        "stop_token_ids",
    }
)


def _apply_generation_defaults(
    config, max_new_tokens=None, temperature=None, stop_token_ids=None
) -> None:
    """Write a load's generation defaults onto the config `generate()` reads.

    `temperature` also decides `do_sample`, because the two libraries spell greedy differently:
    this family spells it temperature 0, and transformers spells it `do_sample=False` and warns
    about a temperature it is not going to use. A temperature of 0 therefore sets `do_sample`
    and leaves the temperature alone, rather than writing a 0 that the validator then objects
    to. An empty `stop_token_ids` means "the artifact's own", not "no stop token".

    Whatever the caller then passes to `generate()` wins, which is the precedence HuggingFace
    already has between a loaded configuration and a call.

    One keyword does not arrive by the Auto route: `AutoModelForCausalLM.from_pretrained`
    resolves the config first, and `PretrainedConfig.from_dict` consumes any keyword naming a
    config attribute, which `temperature` is (a legacy generation attribute). It reaches this
    function through `model_cls().from_pretrained` and not through the Auto class. Sampling
    parameters belong on `generate()`, which is what the tutorial documents.
    """
    if max_new_tokens is not None:
        config.max_new_tokens = int(max_new_tokens)
    if temperature is not None:
        config.do_sample = float(temperature) > 0
        if config.do_sample:
            config.temperature = float(temperature)
    if stop_token_ids:
        config.eos_token_id = list(stop_token_ids)


def register_auto() -> None:
    """Teach the transformers Auto classes about `ax_causal_lm`, so an artifact resolves.

    Call it once before `AutoModelForCausalLM.from_pretrained(artifact_dir)`. Importing this
    module is not enough, and `model_cls` says why.
    """
    model_cls()  # its class body registers the config, and its last line the model


def _generation_config(source: str, current):
    """The checkpoint's own generation defaults, or what `PreTrainedModel` already built.

    Carries the full stop-token set, which matters: `tokenizer.eos_token_id` is a single id
    and a model may end a turn with any of several (Qwen3 uses both `<|im_end|>` and
    `<|endoftext|>`). With this in place, stopping is HuggingFace's own.
    """
    from transformers import GenerationConfig as HFGenerationConfig

    try:
        return HFGenerationConfig.from_pretrained(source)
    except (OSError, ValueError):
        return current

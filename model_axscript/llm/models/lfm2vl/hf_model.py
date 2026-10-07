# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""LFM2.5-VL on an Axelera accelerator, behind the HuggingFace image-text-to-text API.

``AxLfm2VlForConditionalGeneration`` IS a ``transformers.Lfm2VlForConditionalGeneration``
(subclassed, not re-declared) whose weights live in a compiled artifact instead of in
torch: the vision tower and the decoder run on the device, the host does the embedding
scatter and the lm_head. Everything above the model -- the processor, chat templates,
sampling parameters, streamers -- is stock transformers, so a customer drives it with the
code they already have::

    >>> from transformers import AutoModelForImageTextToText, AutoProcessor
    >>> from model_axscript.llm.models.lfm2vl.hf_model import register_auto
    >>> register_auto()   # teaches the Auto classes our model type; see its docstring
    >>> model = AutoModelForImageTextToText.from_pretrained("build/lfm2.5-vl-450m")
    >>> processor = AutoProcessor.from_pretrained("build/lfm2.5-vl-450m")
    >>> image = model.fit_images(image)          # the compiled square
    >>> messages = [{"role": "user", "content": [
    ...     {"type": "image", "image": image}, {"type": "text", "text": "Describe it."}]}]
    >>> inputs = processor.apply_chat_template(messages, tokenize=True,
    ...     add_generation_prompt=True, return_dict=True, return_tensors="pt")
    >>> ids = model.generate(**inputs.to(model.device), max_new_tokens=64)
    >>> processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)

The artifact directory IS a HuggingFace model directory: ``build_artifacts`` saves the
config, the processor (chat template included) and the generation defaults into it, so the
Auto classes load one with no hub and no network. Importing this module registers the
artifact's ``ax_lfm2_vl`` model type, which keeps ordinary LFM2-VL checkpoints going to
stock transformers.

``fit_images`` is the one Axelera-specific line: the vision tower is compiled for ONE
patch grid, so pictures are resized to it before the processor sizes them its own way.
Skip it and the grid check raises, naming the size to use.

**Tiling** -- the report's resolution knob -- needs a prompt shape a plain
``processor(...)`` call cannot produce (row/column markers around several dispatches), so
it has an entry point of its own: :meth:`AxLfm2VlForConditionalGeneration.process`, which
returns the same ``BatchFeature`` a processor would and goes straight into ``generate``.

Generation is ``GenerationMixin.generate`` itself, not a lookalike: this class implements
``forward`` (prefill when the step starts at position 0, one decode step otherwise) and
``prepare_inputs_for_generation``, and transformers does the rest -- sampling, stopping
criteria, streamers, the returned id tensor.

The device's KV cache forces the two limits: it cannot be inspected, reordered or
duplicated, so ``past_key_values`` comes back as None, and beam search /
``num_return_sequences > 1`` / batches > 1 raise rather than quietly misbehave.
``forward`` with ``logits_to_keep=0`` computes every position -- the teacher-forced
scoring path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
from time import perf_counter

import numpy as np
import torch

from .pipeline import Lfm2VlPipeline, VisionInputs, check_grid, fit_image, open_artifact
from ...huggingface import row_ids

logger = logging.getLogger(__name__)


@dataclass
class RunStats:
    """What one generation cost, split the way the machine is split.

    Three stages, each timed where the work happens: the **vision tower** (one dispatch
    per tile, plus host prep), the **prefill** (device decoder, its host input prep, and
    the host lm_head), and **decode** (the same, per token). Filled in by ``forward``, so
    the CLI and a caller's own ``generate()`` read the same numbers -- and readable
    mid-stream, since the per-step lists grow as tokens land.

    A timer covers a device dispatch OR host work, never a mix. That split is the point:
    the lm_head is a ``[vocab x hidden]`` matmul on the host, a real share of a small
    model's decode step, and mixing it in would blame the accelerator for it.

    The stages do NOT add up to the run: transformers samples, runs the logits processors
    and drives the streamer between calls into ``forward``. So ``wall_s`` is measured
    around ``generate()`` and the difference shows up as ``other_s`` rather than quietly
    flattering the device.
    """

    prompt_tokens: int = 0
    #: Positions the prefill graph ingests whatever the prompt's length -- it is compiled
    #: for the context, and the prompt is padded up to it.
    context_len: int = 0
    #: vision: one dispatch per tile (and one more for a split picture's thumbnail)
    vision_dispatches: int = 0
    vision_tokens: int = 0
    vision_device_s: float = 0.0
    vision_s: float = 0.0  # ... plus the host prep around it
    #: prefill: the decoder ingesting the whole prompt in one call
    prefill_device_s: float = 0.0
    prefill_prep_s: float = 0.0  # host: the embedding gather, the causal mask, the scatter
    prefill_host_s: float = 0.0  # final norm + lm_head at the last position
    #: Whether the prefill produced a token: it does in a generate() run, not on the
    #: scoring path (logits_to_keep=0).
    prefill_generated: bool = False
    #: decode: one entry per generated token, device and host (prep + lm_head)
    step_device_s: list[float] = field(default_factory=list)
    step_host_s: list[float] = field(default_factory=list)
    #: Wall clock of the whole ``generate()`` call. nan until the run ends.
    wall_s: float = float("nan")
    #: Not part of this generation's cost: the one-off artifact load, copied in so every
    #: reader can show what was paid before the first prompt. nan if unknown.
    load_s: float = float("nan")

    # --- vision ---------------------------------------------------------------------

    @property
    def vision_host_s(self) -> float:
        return max(0.0, self.vision_s - self.vision_device_s)

    @property
    def ms_per_dispatch(self) -> float:
        if not self.vision_dispatches:
            return 0.0
        return self.vision_device_s * 1e3 / self.vision_dispatches

    # --- prefill --------------------------------------------------------------------

    @property
    def prefill_s(self) -> float:
        return self.prefill_device_s + self.prefill_prep_s + self.prefill_host_s

    @property
    def prefill_tok_s(self) -> float:
        """Prompt tokens per second of prefill -- a size, not a speed.

        The prefill graph is compiled for ``context_len`` positions and the prompt is
        padded up to it, so the latency is the same whatever the prompt's length: this
        number rises with the prompt while nothing gets quicker.
        """
        return self.prompt_tokens / self.prefill_s if self.prefill_s else float("nan")

    # --- decode ---------------------------------------------------------------------

    @property
    def decode_tokens(self) -> int:
        """Decode STEPS -- one short of the answer, whose first token came off the prefill."""
        return len(self.step_device_s)

    @property
    def generated_tokens(self) -> int:
        """Tokens the run produced: the prefill's first one plus the decode steps."""
        return self.decode_tokens + int(self.prefill_generated)

    @property
    def step_s(self) -> list[float]:
        return [d + h for d, h in zip(self.step_device_s, self.step_host_s)]

    @property
    def decode_s(self) -> float:
        return sum(self.step_s)

    @property
    def decode_host_s(self) -> float:
        return sum(self.step_host_s)

    @property
    def decode_tok_s(self) -> float:
        return self.decode_tokens / self.decode_s if self.decode_s else float("nan")

    @property
    def ms_per_token(self) -> float:
        return self.decode_s * 1e3 / self.decode_tokens if self.decode_tokens else float("nan")

    @property
    def decode_wall_tok_s(self) -> float:
        """Decode tok/s against the CLOCK after the first token, framework included."""
        elapsed = self.total_s - self.ttft_s
        return self.decode_tokens / elapsed if self.decode_tokens and elapsed > 0 else float("nan")

    @property
    def host_share(self) -> float:
        """Fraction of decode spent on the host (the lm_head, mostly)."""
        return self.decode_host_s / self.decode_s if self.decode_s else float("nan")

    def step_percentile(self, pct: float) -> float:
        """Per-token latency in ms at a percentile -- jitter, not just the mean."""
        steps = sorted(self.step_s)
        if not steps:
            return float("nan")
        index = min(len(steps) - 1, max(0, round(pct / 100 * (len(steps) - 1))))
        return steps[index] * 1e3

    # --- totals ---------------------------------------------------------------------

    @property
    def ttft_s(self) -> float:
        """Time to first token: the vision tower plus the prefill."""
        return self.vision_s + self.prefill_s

    @property
    def stages_s(self) -> float:
        return self.ttft_s + self.decode_s

    @property
    def total_s(self) -> float:
        """The run's wall clock, measured around ``generate()`` when there is one."""
        return self.stages_s if math.isnan(self.wall_s) else self.wall_s

    @property
    def other_s(self) -> float:
        """Wall clock the stages do not account for: transformers between the forwards."""
        return max(0.0, self.total_s - self.stages_s)

    @property
    def end_to_end_tok_s(self) -> float:
        """Tokens per second counting the wait for the first one -- what a user feels."""
        return self.generated_tokens / self.total_s if self.total_s else float("nan")

    def summary(self) -> str:
        """One line: what a demo watcher wants without reading a table."""
        return (
            f"vision {self.vision_s * 1e3:.0f} ms ({self.vision_dispatches} dispatch"
            f"{'es' if self.vision_dispatches != 1 else ''}) | "
            f"prefill {self.prefill_s * 1e3:.0f} ms ({self.prompt_tokens} tokens) | "
            f"TTFT {self.ttft_s * 1e3:.0f} ms | {self.generated_tokens} tokens at "
            f"{self.decode_tok_s:.1f} tok/s ({self.ms_per_token:.0f} ms/token, "
            f"{self.host_share * 100:.0f}% host) | total {self.total_s:.2f} s"
        )

    def rows(self) -> list[tuple[str, str, str]]:
        """(section, metric, value) -- the one source the CLI report uses."""

        def per(amount: float, count: float) -> float:
            return amount / count if count else 0.0

        vision = [
            # A split picture is several dispatches, so this counts what the tower
            # encoded, which is the thing the timings are per.
            ("vision", "dispatches", f"{self.vision_dispatches}"),
            ("vision", "tokens produced", f"{self.vision_tokens}"),
            (
                "vision",
                "tokens per dispatch",
                f"{per(self.vision_tokens, self.vision_dispatches):.0f}",
            ),
            ("vision", "device", f"{self.vision_device_s * 1e3:.0f} ms"),
            ("vision", "per dispatch", f"{self.ms_per_dispatch:.0f} ms"),
            (
                "vision",
                "throughput",
                f"{per(self.vision_dispatches, self.vision_device_s):.1f} dispatch/s",
            ),
            ("vision", "host prep", f"{self.vision_host_s * 1e3:.0f} ms"),
        ]
        ingest = f"{self.prefill_tok_s:.0f} tok/s"
        if self.context_len:
            ingest += f" ({self.prompt_tokens} of {self.context_len} positions)"
        prefill = [
            ("prefill", "prompt", f"{self.prompt_tokens} tokens"),
            ("prefill", "device", f"{self.prefill_device_s * 1e3:.0f} ms"),
            ("prefill", "host prep", f"{self.prefill_prep_s * 1e3:.0f} ms"),
            ("prefill", "host lm_head", f"{self.prefill_host_s * 1e3:.0f} ms"),
            ("prefill", "ingest rate", ingest),
        ]
        decode = [
            # generated counts the prefill's token; the rates below are per decode STEP,
            # which is one fewer -- see decode_tokens.
            ("decode", "generated", f"{self.generated_tokens} tokens"),
            ("decode", "steps", f"{self.decode_tokens}"),
            ("decode", "speed", f"{self.decode_tok_s:.2f} tok/s"),
            ("decode", "per token", f"{self.ms_per_token:.1f} ms"),
            *(
                [("decode", "speed (wall)", f"{self.decode_wall_tok_s:.2f} tok/s")]
                if not math.isnan(self.wall_s) and self.decode_tokens
                else []
            ),
            (
                "decode",
                "device / host",
                f"{sum(self.step_device_s) * 1e3:.0f} / {self.decode_host_s * 1e3:.0f} ms",
            ),
            ("decode", "host share", f"{self.host_share * 100:.0f} %"),
        ]
        if self.decode_tokens >= 4:  # a percentile of three samples says nothing
            decode.append(
                (
                    "decode",
                    "median / p95 token",
                    f"{self.step_percentile(50):.0f} / {self.step_percentile(95):.0f} ms",
                )
            )
        total = [
            ("total", "time to first token", f"{self.ttft_s:.2f} s ({self.ttft_s * 1e3:.0f} ms)"),
            ("total", "wall clock", f"{self.total_s:.2f} s"),
            ("total", "end to end", f"{self.end_to_end_tok_s:.2f} tok/s"),
        ]
        if not math.isnan(self.load_s):
            total.insert(0, ("total", "first load", f"{self.load_s:.2f} s"))
        if not math.isnan(self.wall_s) and self.decode_tokens:
            per_token = self.other_s * 1e3 / self.decode_tokens
            total.insert(
                -2,  # after TTFT, above the wall clock it is part of
                ("total", "framework", f"{self.other_s * 1e3:.0f} ms ({per_token:.1f} ms/token)"),
            )
        return [*vision, *prefill, *decode, *total]

    def stages(self) -> list[tuple[str, float]]:
        """Where the wall clock went, in the order the work happens."""
        stages = [
            ("vision", self.vision_s),
            ("prefill", self.prefill_s),
            ("decode", self.decode_s),
        ]
        if self.other_s:
            stages.append(("other", self.other_s))
        return stages

    def time_bar(self, width: int = 44) -> str:
        """The stage split as one bar."""
        total = max(self.total_s, 1e-9)
        blocks = {"vision": "█", "prefill": "▓", "decode": "░", "other": "·"}
        stages = self.stages()
        # Every stage that took any time keeps a block, so a fast one stays visible; the
        # widest then gives back what that rounding borrowed, since a bar wider than
        # `width` wraps the terminal and stops being a bar.
        cells = [max(1, round(share / total * width)) if share else 0 for _, share in stages]
        while sum(cells) > width:
            cells[cells.index(max(cells))] -= 1
        bar = "".join(blocks[name] * cell for (name, _), cell in zip(stages, cells))
        legend = "  ".join(f"{blocks[name]} {name} {share * 1e3:.0f} ms" for name, share in stages)
        return f"{bar}\n{legend}"

    def report(self) -> str:
        """The rows as an aligned block, for a terminal."""
        width = max(len(f"{section} {metric}") for section, metric, _ in self.rows())
        lines, last = [], None
        for section, metric, value in self.rows():
            if section != last:
                lines.append(f"  {section}")
                last = section
            lines.append(f"    {metric:<{width}} {value}")
        return "\n".join(lines)


def session_rows(history: list[RunStats]) -> list[tuple[str, str]]:
    """Averages over every run so far -- what the numbers settle at, not one sample."""
    if not history:
        return []
    n = len(history)

    def mean(pick) -> float:
        return sum(pick(s) for s in history) / n

    return [
        ("runs", f"{n}"),
        ("avg vision", f"{mean(lambda s: s.vision_s) * 1e3:.0f} ms"),
        ("avg prefill", f"{mean(lambda s: s.prefill_s) * 1e3:.0f} ms"),
        ("avg TTFT", f"{mean(lambda s: s.ttft_s) * 1e3:.0f} ms"),
        ("avg decode", f"{mean(lambda s: s.decode_tok_s):.2f} tok/s"),
        ("avg per token", f"{mean(lambda s: s.ms_per_token):.1f} ms"),
        ("tokens generated", f"{sum(s.generated_tokens for s in history)}"),
        ("time on device", f"{sum(s.total_s for s in history):.1f} s"),
    ]


def _build_model_cls():  # noqa: C901
    """Define the shim against transformers (imported lazily)."""
    from transformers import (
        AutoConfig,
        AutoModelForImageTextToText,
        AutoProcessor,
        GenerationConfig as HFGenerationConfig,
        PreTrainedModel,
    )
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.lfm2_vl.configuration_lfm2_vl import Lfm2VlConfig
    from transformers.models.lfm2_vl.modeling_lfm2_vl import Lfm2VlForConditionalGeneration
    from transformers.models.lfm2_vl.processing_lfm2_vl import Lfm2VlProcessor

    class AxLfm2VlConfig(Lfm2VlConfig):
        """LFM2-VL's own config under a model type of ours.

        Only the name differs, and only so the Auto classes route an ARTIFACT directory
        here: registering against the checkpoint's own model type would hijack every
        ordinary ``from_pretrained`` in the process.
        """

        model_type = "ax_lfm2_vl"

    class AxLfm2VlForConditionalGeneration(Lfm2VlForConditionalGeneration):
        """LFM2.5-VL backed by a compiled Axelera artifact pair (vision + decoder)."""

        config_class = AxLfm2VlConfig

        def __init__(self, config):
            # Deliberately NOT Lfm2VlForConditionalGeneration.__init__: that allocates
            # hundreds of millions of torch parameters we would immediately throw away --
            # the weights are in the artifact. PreTrainedModel.__init__ gives the HF
            # plumbing (config, generation_config, dtype/device resolution) without them.
            PreTrainedModel.__init__(self, config)
            # One parameter so .device / .dtype resolve; the compute is off the torch graph.
            self._marker = torch.nn.Parameter(torch.zeros(1), requires_grad=False)
            self._pipeline = None
            #: the device's KV write position -- the only decode state the host tracks.
            self._position = 0
            #: what the last (or running) generation cost; see RunStats.
            self.stats = RunStats()
            #: every finished generation, for the session averages (session_rows).
            self.history: list[RunStats] = []

        # ----- construction -----------------------------------------------------------

        @classmethod
        def from_artifact(
            cls, artifact_dir, repo_id: str | None = None, *, generation=None, device=None
        ):
            """Load a built artifact directory (vision/ + text/ + quantized_model/).

            The config, the processor and the generation defaults come from the artifact
            itself, so a load touches no network. ``repo_id`` overrides that, and is the
            fallback for artifacts built before those files were written. ``generation``
            defaults to what the artifact records and must match it when given; the core
            count is read off the artifact too, never asked for. ``device`` names the Metis
            part, the way ``--device`` spells it, and defaults to the first one.
            """
            from .pipeline import hf_metadata_dir

            artifacts, repo_id = open_artifact(artifact_dir, repo_id)
            source = hf_metadata_dir(artifact_dir) or repo_id
            model = cls(as_ax_config(AutoConfig.from_pretrained(source)))
            model.generation_config = HFGenerationConfig.from_pretrained(source)
            # from_pretrained sets name_or_path to wherever it loaded from (the artifact
            # directory); the checkpoint it was BUILT from is the useful answer.
            model.config.name_or_path = repo_id
            model._repo_id = repo_id
            # Timed: both towers' weights crossing PCIe is the one cost a demo watcher
            # sees before anything happens.
            started = perf_counter()
            model._pipeline = Lfm2VlPipeline(
                artifacts, repo_id, generation=generation, device=device
            )
            model._load_s = perf_counter() - started
            model.eval()
            return model

        @classmethod
        def from_pretrained(cls, path, *args, **kwargs):
            """``from_pretrained`` on a directory that is an artifact, not a checkpoint."""
            if not (Path(path) / "text" / "pipeline.json").exists():
                raise ValueError(
                    f"{path} is not a compiled LFM2-VL artifact (no text/pipeline.json). "
                    "Build one with model_axscript.llm.models.lfm2vl.pipeline.build_artifacts, or use "
                    "transformers.Lfm2VlForConditionalGeneration for a plain HF checkpoint."
                )
            # AutoModelForImageTextToText.from_pretrained hands over a config it already
            # parsed plus its whole hub kwarg set; the artifact carries all of that, so what
            # this loader declares is kept and the rest dropped. An ALLOW list: the Auto
            # factory also injects keywords of its own (adapter_kwargs, code_revision,
            # _from_auto), so a deny list goes stale and the miss lands as a TypeError from
            # inside `from_artifact`, which takes no **kwargs.
            kept = {k: kwargs[k] for k in ("repo_id", "generation", "device") if k in kwargs}
            return cls.from_artifact(path, *args, **kept)

        #: The spelling the other Axelera model wrappers use for the same load.
        load_compiled = from_pretrained

        @property
        def pipeline(self) -> Lfm2VlPipeline:
            if self._pipeline is None:
                raise RuntimeError("construct via AxLfm2VlForConditionalGeneration.from_artifact")
            return self._pipeline

        @property
        def processor(self):
            """The model's own processor -- chat template, tokenizer, image processor."""
            return self.pipeline.processor

        @property
        def tokenizer(self):
            return self.pipeline.tokenizer

        @property
        def num_cores(self) -> int:
            """AI cores the artifact places work on (and sub-devices it claims)."""
            return self.pipeline.num_cores

        @property
        def context_len(self) -> int:
            """The compiled KV-cache capacity: prompt + answer must fit in it."""
            return self.pipeline.context_len

        @property
        def load_s(self) -> float:
            """Seconds the artifact took to load onto the device, or nan if not measured."""
            return getattr(self, "_load_s", float("nan"))

        def can_generate(self) -> bool:
            return True

        def get_input_embeddings(self):
            """None: the embedding table is in the artifact, not in torch.

            ``PreTrainedModel``'s version reaches for a submodule this class never
            allocates, so HF plumbing that merely asks would hit an AttributeError.
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
                "model_axscript.llm.models.lfm2vl.pipeline.build_artifacts"
            )

        # ----- the Axelera-specific steps ---------------------------------------------

        @property
        def image_shape(self) -> tuple[int, int]:
            """(height, width) in pixels the vision tower was compiled for."""
            return self.pipeline.config.vision.image_shape

        @property
        def image_geometry(self) -> str:
            """The compiled geometry for a log line: "512" if square, else "288x512"."""
            return self.pipeline.config.vision.image_geometry

        def fit_images(self, images):
            """Resize picture(s) to the compiled geometry -- run before the processor.

            The tower is built for one geometry, so this is what pins the processor's
            patch grid to the one the device can encode. To split a high-resolution
            picture across several dispatches instead, use :meth:`process`.
            """
            single = not isinstance(images, (list, tuple))
            fitted = [fit_image(self.pipeline.config, i) for i in ([images] if single else images)]
            return fitted[0] if single else fitted

        def process(self, messages, images=None, *, split=None, return_tensors="pt"):
            """A conversation + its pictures as the inputs ``generate`` takes.

            The tiling entry point: the picture is reshaped to the closest supported tile
            aspect, cut into compiled-grid squares and given the processor's own
            row/column markers, thumbnail token and placeholder runs. The result is the
            same ``BatchFeature`` a processor call produces, so it goes straight into
            ``generate(**inputs)``.

            ``split`` overrides the artifact's ``do_image_splitting``: False encodes each
            picture whole (one dispatch), which is cheaper and less detailed.
            """
            from transformers.feature_extraction_utils import BatchFeature

            from .pipeline import render

            pipeline = self.pipeline
            config = pipeline.config
            inputs = render(pipeline.processor, config, messages, images, split=split)
            data = {"input_ids": np.asarray(inputs.input_ids)[None]}
            if inputs.patches is not None:
                n_patches = config.vision.num_patches
                dispatches = inputs.patches.shape[0] // n_patches
                data["pixel_values"] = inputs.patches.reshape(dispatches, n_patches, -1)
                data["spatial_shapes"] = np.asarray(
                    [[config.vision.grid_h, config.vision.grid_w]] * dispatches
                )
                data["pixel_attention_mask"] = np.ones((dispatches, n_patches), dtype=np.int32)
            return BatchFeature(data=data, tensor_type=return_tensors)

        # ----- prompt assembly from processor outputs ---------------------------------

        def _vision_inputs(self, row_ids, pixel_values, spatial_shapes, mask) -> VisionInputs:
            """A processor batch -> VisionInputs, one compiled-grid dispatch per row."""
            config = self.pipeline.config
            ids = np.asarray(row_ids, dtype=np.int64).reshape(-1)
            if pixel_values is None:
                return VisionInputs(input_ids=ids, patches=None)
            values = np.asarray(pixel_values, dtype=np.float32)
            n_patches = config.vision.num_patches
            if spatial_shapes is None:
                raise ValueError("pixel_values were passed without spatial_shapes")
            rows = []
            for i in range(values.shape[0]):
                check_grid(config, np.asarray(spatial_shapes)[i])
                if mask is not None and int(np.asarray(mask)[i].sum()) != n_patches:
                    raise ValueError(
                        f"image {i} carries {int(np.asarray(mask)[i].sum())} real patches "
                        f"but the tower is compiled for {n_patches}"
                    )
                rows.append(values[i, :n_patches])
            return VisionInputs(
                input_ids=ids,
                patches=np.concatenate(rows, axis=0),
                layouts=[(1, 1)] * values.shape[0],
            )

        # ----- forward: the device, in the shape GenerationMixin.generate expects ------

        def forward(
            self,
            input_ids=None,
            pixel_values=None,
            spatial_shapes=None,
            pixel_attention_mask=None,
            attention_mask=None,
            cache_position=None,
            logits_to_keep=0,
            labels=None,
            **kwargs,
        ):
            """Logits for ``input_ids``, prefilling or stepping the device as HF asks.

            The whole model is this one method, which is what lets transformers' own
            ``generate()`` drive: prefill vs decode comes from ``cache_position``, exactly
            as for a torch model with a KV cache -- except the cache lives on the device,
            so nothing is handed back and forth. ``logits_to_keep=1`` (what generation asks
            for) computes the lm_head at the last position only; 0 computes every position.

            ``past_key_values`` comes back None on purpose: HF only threads it through, and
            this device's cache cannot be inspected, reordered or duplicated.
            """
            pipeline = self.pipeline
            impl, host = pipeline.impl, pipeline.host_params
            _, row = row_ids(input_ids, attention_mask)
            prefill = cache_position is None or int(cache_position[0]) == 0

            if prefill:
                # Vision geometry first: a picture the processor sized its own way blows
                # the context too, and "resize it" is the useful half of that news.
                vision_inputs = self._vision_inputs(
                    row, pixel_values, spatial_shapes, pixel_attention_mask
                )
                self._check_room(len(row))
                stats = self.stats = RunStats(
                    prompt_tokens=len(row), context_len=pipeline.context_len, load_s=self.load_s
                )
                # The tower writes `last_encode` only when it runs, so clear it: a prompt
                # with no picture must report no vision, not the last picture's cost.
                tower = getattr(pipeline, "vision", None)
                if tower is not None:
                    tower.last_encode = (0, 0.0)
                # No cache reset between generations: the decoder's state is the
                # position-indexed KV cache and the prefill rewrites it from position 0.
                start = perf_counter()
                prompt = pipeline.prepare_inputs(vision_inputs)
                stats.vision_s = perf_counter() - start
                # The tower times its own dispatches; the rest of the above is host prep.
                stats.vision_dispatches, stats.vision_device_s = getattr(
                    pipeline.vision, "last_encode", (0, 0.0)
                )
                stats.vision_tokens = (
                    0 if prompt.image_embeds is None else len(prompt.image_embeds)
                )
                impl.set_prompt(prompt)

                # The input prep (the embedding gather, the causal mask, the image
                # scatter) is HOST work, and timing it inside the dispatch would bill the
                # accelerator for it -- hence the split rather than a call to
                # impl.prefill_hidden.
                logits = self._ingest_prompt(row, stats, logits_to_keep)
                # logits_to_keep=1 is generation asking for the next token: this prefill
                # produced the answer's first one. 0 is teacher-forced scoring.
                stats.prefill_generated = logits_to_keep == 1
            else:
                graph = impl._decode_graph_name()
                for token in row:  # generation feeds one; nothing stops a caller feeding more
                    self._check_room(self._position + 1)
                    start = perf_counter()
                    step_inputs = impl.step_inputs(
                        host, token, self._position, pipeline.context_len
                    )
                    prep_s = perf_counter() - start
                    start = perf_counter()
                    device_out = pipeline.text_runtime.run(graph, step_inputs)
                    self.stats.step_device_s.append(perf_counter() - start)
                    start = perf_counter()
                    logits = impl.compute_logits(device_out, host)
                    self.stats.step_host_s.append(prep_s + perf_counter() - start)
                    self._position += 1

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

        def _ingest_prompt(self, row, stats, logits_to_keep):
            """Run the prompt through the decoder; returns the logits asked for.

            The routes are `Implementation.ingest_prompt`'s, shared with every other AxScript
            LLM, so this model cannot answer a prompt differently from the way the decode loop
            and the text families answer it. Two things stay here. Choosing between the two
            prefill graphs a chunked artifact carries is this model's own, because the choice
            is arithmetic on a measured replay cost (see `text_impl.plan_prefill`). And the
            lm_head is host work, so its seconds are billed apart from the dispatches.

            Teacher-forced scoring needs every position, and the chunked routes overwrite one
            hidden state per chunk, so `keep_hidden` collects the slices to concatenate.
            """
            pipeline = self.pipeline
            impl, host, ctx = pipeline.impl, pipeline.host_params, pipeline.context_len
            # logits_to_keep=1 is generation asking for the next token; 0 is teacher-forced
            # scoring, which wants every position.
            want_all = logits_to_keep != 1
            # A chunked artifact carries a full-context graph too; which one is cheaper
            # depends on this prompt's length, so the choice is made per prompt.
            chunk_len = impl.select_prefill(
                len(row), pipeline.chunk_len, ctx, pipeline.text_runtime.graph_names
            )
            kept = [] if want_all else None
            hidden, last_row, self._position = impl.ingest_prompt(
                pipeline.text_runtime,
                host,
                row,
                ctx,
                chunk_len,
                timing=stats,
                keep_hidden=kept,
            )
            start = perf_counter()
            logits = (
                impl.compute_logits_seq(np.concatenate(kept, axis=1), host)
                if want_all
                else impl.compute_logits(hidden, host, position=last_row)
            )
            stats.prefill_host_s += perf_counter() - start
            return logits

        def _check_room(self, length: int) -> None:
            if length > self.context_len:
                raise ValueError(
                    f"{length} tokens against the artifact's {self.context_len}-token "
                    "context: shorten the prompt, split fewer tiles, lower max_new_tokens, "
                    "or build the artifact with a larger context_len"
                )

        # ----- generate: HuggingFace's, with only the device bookkeeping added ---------

        def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            cache_position=None,
            pixel_values=None,
            spatial_shapes=None,
            pixel_attention_mask=None,
            **kwargs,
        ):
            """What each generate() step passes to forward: the NEW tokens only.

            The device holds the cache, so this only has to say "prefill" or "one more
            token"; the pictures ride on the prompt, and only on the prompt.
            """
            prefill = cache_position is None or int(cache_position[0]) == 0
            inputs = {
                "input_ids": input_ids if prefill else input_ids[:, -1:],
                "cache_position": cache_position,
                "logits_to_keep": 1,  # generation only ever reads the last position
            }
            if prefill:
                inputs.update(
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    spatial_shapes=spatial_shapes,
                    pixel_attention_mask=pixel_attention_mask,
                )
            return inputs

        @torch.no_grad()
        def generate(self, *args, **kwargs):
            """``GenerationMixin.generate``, unchanged.

            Sampling, stopping criteria, streamers and the returned id tensor are all
            transformers' own; this override only refuses what one device cache cannot do,
            and drops the prompt context afterwards.
            """
            config = kwargs.get("generation_config") or self.generation_config
            for flag in ("num_beams", "num_return_sequences"):
                asked = int(kwargs.get(flag, getattr(config, flag, 1)) or 1)
                if asked > 1:
                    raise NotImplementedError(
                        f"{flag}={asked} needs several KV caches; the device has one, and "
                        "it cannot be reordered or duplicated"
                    )
            # The run's own clock: the stage timers only cover the calls into `forward`, so
            # what transformers spends sampling between them lands in `other_s` rather than
            # nowhere. `previous` tells a run that reached `forward` from one that died
            # before it -- the latter must not stamp its time onto the previous run's stats.
            started, previous = perf_counter(), self.stats
            try:
                return super().generate(*args, **kwargs)
            finally:
                self.pipeline.impl.set_prompt(None)
                self._position = 0
                if self.stats is not previous:  # a real run, not a refusal
                    self.stats.wall_s = perf_counter() - started
                    if self.stats.generated_tokens:
                        self.history.append(self.stats)

    def as_ax_config(config):
        """An LFM2-VL config (from anywhere) as this model's config type."""
        if isinstance(config, AxLfm2VlConfig):
            return config
        data = config.to_dict()
        data.pop("model_type", None)  # the class attribute is the one that counts
        return AxLfm2VlConfig(**data)

    # The Auto classes, so an artifact directory loads the way any HF model does.
    # Importing this module is the registration route, which is why nothing here needs
    # trust_remote_code; exist_ok so a re-import does not explode.
    AutoConfig.register(AxLfm2VlConfig.model_type, AxLfm2VlConfig, exist_ok=True)
    AutoModelForImageTextToText.register(
        AxLfm2VlConfig, AxLfm2VlForConditionalGeneration, exist_ok=True
    )
    AutoProcessor.register(AxLfm2VlConfig, Lfm2VlProcessor, exist_ok=True)

    return AxLfm2VlConfig, AxLfm2VlForConditionalGeneration, as_ax_config


try:
    AxLfm2VlConfig, AxLfm2VlForConditionalGeneration, as_ax_config = _build_model_cls()
except ImportError:  # transformers absent: the rest of the suite still imports
    AxLfm2VlConfig = AxLfm2VlForConditionalGeneration = as_ax_config = None


def register_auto() -> None:
    """Teach the transformers Auto classes about LFM2.5-VL's model type, so an artifact resolves.

    Call it once before `AutoModelForImageTextToText.from_pretrained(artifact_dir)`. One
    spelling for every AxScript family, text included: a customer moving between them should
    not have to learn a second one, and the failure when they use the other family's is
    "Transformers does not recognize this architecture", which names nothing they can act on.

    Importing this module registers too, wherever transformers is installed. Where it is not,
    the import leaves the names None and this call raises the ImportError that says so.
    """
    global AxLfm2VlConfig, AxLfm2VlForConditionalGeneration, as_ax_config

    if AxLfm2VlForConditionalGeneration is None:
        AxLfm2VlConfig, AxLfm2VlForConditionalGeneration, as_ax_config = _build_model_cls()


def artifact_config(repo_id: str):
    """The config to write into an artifact: the checkpoint's, under our model type.

    Same fields as the HF one -- ``save_hf_metadata`` writes this so the artifact loads
    through the Auto classes without claiming to be a plain LFM2-VL checkpoint.
    """
    from transformers import AutoConfig

    config = as_ax_config(AutoConfig.from_pretrained(repo_id))
    config.name_or_path = repo_id  # provenance: what it was built from
    return config

# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Qwen3-VL on Metis behind the HuggingFace image-text-to-text API.

`AxQwen3VLForConditionalGeneration` IS a `transformers.Qwen3VLForConditionalGeneration`
(subclassed, not re-declared) whose weights live in a compiled artifact instead of in
torch: the vision tower and the decoder run on the device, the host does the embedding
splice, MRoPE and the lm_head. Everything above the model -- the processor, chat
templates, sampling parameters, streamers -- is stock transformers, so a customer drives
it with the code they already have:

    >>> from transformers import AutoModelForImageTextToText, AutoProcessor
    >>> import llm.models.qwen3vl.hf_model  # registers the artifact's model type
    >>> model = AutoModelForImageTextToText.from_pretrained("build/qwen3vl-2b")
    >>> processor = AutoProcessor.from_pretrained("build/qwen3vl-2b")
    >>> image = model.fit_images(image)          # the compiled square
    >>> messages = [{"role": "user", "content": [
    ...     {"type": "image", "image": image}, {"type": "text", "text": "Describe it."}]}]
    >>> inputs = processor.apply_chat_template(messages, tokenize=True,
    ...     add_generation_prompt=True, return_dict=True, return_tensors="pt")
    >>> ids = model.generate(**inputs.to(model.device), max_new_tokens=64)
    >>> processor.batch_decode(ids[:, inputs["input_ids"].shape[1]:],
    ...                        skip_special_tokens=True)

The artifact directory IS a HuggingFace model directory: `build_artifacts` saves the
config, the processor (chat template included) and the generation defaults into it, so the
Auto classes load one with no hub and no network. Importing this module registers the
artifact's `ax_qwen3vl` model type, which keeps ordinary Qwen3-VL checkpoints going to
stock transformers.

`fit_images` is the one Axelera-specific line: the vision tower is compiled for ONE patch
grid, so pictures are resized to it before the processor sizes them its own way (Qwen's
smart-resize always preserves aspect ratio, so no processor setting can do it). Skip it
and the grid check raises, naming the size to use.

Generation is `GenerationMixin.generate` itself, not a lookalike: this class implements
`forward` (prefill when the step starts at position 0, one decode step otherwise) and
`prepare_inputs_for_generation`, and transformers does the rest -- sampling, stopping
criteria, streamers, the returned id tensor.

The device's KV cache forces the two limits: it cannot be inspected, reordered or
duplicated, so `past_key_values` comes back as None, and beam search /
num_return_sequences > 1 / batches > 1 raise rather than quietly misbehave. `forward` with
`logits_to_keep=0` computes every position -- the teacher-forced scoring path.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import statistics
from time import perf_counter

import numpy as np
import torch

from .pipeline import Qwen3VLPipeline, VisionInputs, check_grids, fit_image, open_artifact
from ...huggingface import row_ids

logger = logging.getLogger(__name__)


@dataclass
class RunStats:
    """What one generation cost, split the way the machine is split.

    Three stages, each timed where the work happens: the **vision tower** (one dispatch per
    picture / per temporal patch, plus host prep), the **prefill** (device decoder, its host
    input prep, and the host lm_head), and **decode** (the same, per token). Filled in by
    `forward`, so the CLI, the UI and a caller's own `generate()` read the same numbers --
    and readable mid-stream, since `step_s` grows as tokens land.

    A timer covers a device dispatch OR host work, never a mix. That split is the point: the
    lm_head is an fp16 [vocab x hidden] gemv on the host (~15 ms, a real share of a small
    model's decode step), and mixing it in would blame the accelerator for it.

    The stages do NOT add up to the run: transformers samples, runs the logits processors
    and drives the streamer between calls into `forward` (~5 ms/token when sampling, ~0.1 ms
    greedy). So `wall_s` is measured around `generate()` and the difference shows up as
    `other_s` rather than quietly flattering the device.
    """

    prompt_tokens: int = 0
    #: Positions the prefill graph ingests whatever the prompt's length -- it is compiled
    #: for the context, and the prompt is padded up to it. Needed to read `prefill_tok_s`
    #: for what it is; 0 for a RunStats built without one.
    context_len: int = 0
    #: vision: one dispatch per picture, or per temporal patch of a video
    vision_dispatches: int = 0
    vision_tokens: int = 0  # rows the tower produced == placeholder rows in the prompt
    vision_device_s: float = 0.0  # the dispatches themselves
    vision_s: float = 0.0  # ... plus the host prep around them
    #: Host int8 patch quantize, which happens between the dispatches and so used to be
    #: counted inside `vision_device_s`. Broken out because a frame is reported as the
    #: accelerator's cost and anything host-side in that number is a mis-attribution.
    vision_quantize_s: float = 0.0
    #: One entry per dispatch, so a multi-image prompt can be read on a median instead of
    #: a mean that the cold first dispatch drags around.
    vision_dispatch_s: list[float] = field(default_factory=list)
    #: What the tower was shown, off the processor's grids: pictures, and video FRAMES --
    #: not dispatches, since one swallows temporal_patch_size frames. A clip padded up to a
    #: whole temporal patch reads one frame longer than the source, because that is what the
    #: tower was handed.
    images: int = 0
    frames: int = 0
    #: prefill: the decoder ingesting the whole prompt in one call
    prefill_device_s: float = 0.0
    prefill_prep_s: float = 0.0  # host: embeddings, causal mask, MRoPE, DeepStack scatter
    prefill_host_s: float = 0.0  # final norm + lm_head at the last position
    #: Whether the prefill produced a token: it does in a generate() run (the answer's first
    #: token comes off its lm_head), not on the scoring path (logits_to_keep=0). Without it
    #: every count of "tokens generated" is one short.
    prefill_generated: bool = False
    #: decode: one entry per generated token, device and host (prep + lm_head)
    step_device_s: list[float] = field(default_factory=list)
    step_host_s: list[float] = field(default_factory=list)
    #: ... and that host number split in two. `step_host_s` stays the sum, so existing
    #: readers are unaffected. The halves are nothing alike -- sub-millisecond input prep
    #: against a [vocab x hidden] gemv -- and fusing them hid the largest host cost.
    step_prep_s: list[float] = field(default_factory=list)
    step_lmhead_s: list[float] = field(default_factory=list)
    #: Wall clock of the whole `generate()` call, stages and framework alike. nan until the
    #: run ends (mid-stream readers fall back to the stages' sum; see total_s).
    wall_s: float = float("nan")
    #: Not part of this generation's cost: the one-off artifact load, copied in so every
    #: reader of a RunStats can show what was paid before the first prompt. nan if unknown.
    load_s: float = float("nan")

    # --- vision ---------------------------------------------------------------------

    @property
    def vision_host_s(self) -> float:
        """MRoPE tables, the DeepStack scatter, the embedding splice's bookkeeping."""
        return max(0.0, self.vision_s - self.vision_device_s)

    @property
    def ms_per_dispatch(self) -> float:
        if not self.vision_dispatches:
            return 0.0
        return self.vision_device_s * 1e3 / self.vision_dispatches

    @property
    def vision_dispatch_only_s(self) -> float:
        """The dispatches with the host patch quantize taken back out.

        `vision_device_s` brackets the whole encode loop, quantize included; this is the
        part that actually ran on the accelerator.
        """
        return max(0.0, self.vision_device_s - self.vision_quantize_s)

    @property
    def ms_per_dispatch_median(self) -> float:
        """Median dispatch, in ms -- the figure to quote for a frame."""
        if not self.vision_dispatch_s:
            return self.ms_per_dispatch
        return statistics.median(self.vision_dispatch_s) * 1e3

    # --- prefill --------------------------------------------------------------------

    @property
    def prefill_s(self) -> float:
        return self.prefill_device_s + self.prefill_prep_s + self.prefill_host_s

    @property
    def prefill_host_total_s(self) -> float:
        """Everything in the prefill that runs on the host: the input prep and the lm_head."""
        return self.prefill_prep_s + self.prefill_host_s

    @property
    def prefill_tok_s(self) -> float:
        """Prompt tokens per second of prefill -- a size, not a speed.

        The prefill graph is compiled for `context_len` positions and the prompt is padded
        up to it, so the latency is the same whatever the prompt's length: this number rises
        with the prompt while nothing gets quicker. `prefill_s` is the figure that moves.
        """
        return self.prompt_tokens / self.prefill_s if self.prefill_s else float("nan")

    # --- decode ---------------------------------------------------------------------

    @property
    def decode_tokens(self) -> int:
        """Decode STEPS -- one short of the answer, whose first token came off the prefill.

        The right denominator for a per-step rate (`decode_tok_s`, `ms_per_token`), and the
        wrong one for "how many tokens did it write": that is `generated_tokens`.
        """
        return len(self.step_device_s)

    @property
    def generated_tokens(self) -> int:
        """Tokens the run actually produced: the prefill's first one plus the decode steps.

        transformers calls `forward` once per new token and the first of those is the
        prefill, so an N-token answer is one prefill and N-1 decode steps.
        """
        return self.decode_tokens + int(self.prefill_generated)

    @property
    def step_s(self) -> list[float]:
        """Per-token wall time: what the latency plot draws."""
        return [d + h for d, h in zip(self.step_device_s, self.step_host_s)]

    @property
    def decode_s(self) -> float:
        return sum(self.step_s)

    @property
    def decode_host_s(self) -> float:
        return sum(self.step_host_s)

    @property
    def decode_prep_s(self) -> float:
        """Host input prep over the whole decode: token embed, MRoPE row, seq_pos."""
        return sum(self.step_prep_s)

    @property
    def decode_lmhead_s(self) -> float:
        """Host lm_head over the whole decode -- the [vocab x hidden] gemv, per token."""
        return sum(self.step_lmhead_s)

    @property
    def ms_per_lmhead(self) -> float:
        """What one token's lm_head costs on the host, in ms."""
        if not self.step_lmhead_s:
            return float("nan")
        return statistics.median(self.step_lmhead_s) * 1e3

    @property
    def lmhead_share(self) -> float:
        """Fraction of a decode step spent in the host lm_head.

        A fixed per-token cost, so it grows as a share every time the decoder gets faster.
        Measured, it is not worth relocating: the host runs it at ~41 GB/s (DRAM-bound,
        ~15 ms), against ~20 GB/s for the same projection as a device matmul.
        """
        return self.decode_lmhead_s / self.decode_s if self.decode_s else float("nan")

    @property
    def decode_tok_s(self) -> float:
        return self.decode_tokens / self.decode_s if self.decode_s else float("nan")

    @property
    def decode_tok_s_without_lmhead(self) -> float:
        """Decode rate if the lm_head cost nothing -- the ceiling for any work on it.

        An upper bound, not a forecast: the weights still have to be read wherever it runs.
        """
        without = self.decode_s - self.decode_lmhead_s
        return self.decode_tokens / without if self.decode_tokens and without > 0 else float("nan")

    @property
    def ms_per_token(self) -> float:
        return self.decode_s * 1e3 / self.decode_tokens if self.decode_tokens else float("nan")

    @property
    def decode_wall_tok_s(self) -> float:
        """Decode tok/s against the CLOCK after the first token, framework included.

        `decode_tok_s` divides by the stage timers ("how fast are the towers"); this divides
        by the time that actually passed ("how fast do tokens arrive"), so the sampling and
        the streamer are in it. It is the same quantity `generate.benchmark_generation`
        reports for every other model, so this is the number to compare across models.
        """
        elapsed = self.total_s - self.ttft_s
        return self.decode_tokens / elapsed if self.decode_tokens and elapsed > 0 else float("nan")

    @property
    def host_share(self) -> float:
        """Fraction of decode spent on the host (the fp32 lm_head, mostly)."""
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
        """The stages added up -- everything this class timed itself."""
        return self.ttft_s + self.decode_s

    @property
    def total_s(self) -> float:
        """The run's wall clock, measured around `generate()` when there is one.

        Falls back to the stages' sum mid-stream, where nothing has been measured yet: a
        live panel reads a little low and lands on the real figure when the answer does.
        """
        return self.stages_s if math.isnan(self.wall_s) else self.wall_s

    @property
    def other_s(self) -> float:
        """Wall clock the stages do not account for: transformers between the forwards.

        Sampling over the whole vocabulary, the logits processors, the stopping criteria,
        the streamer's decode. Not the accelerator's problem, but it is in the run.
        """
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
            f"prefill {self.prefill_s * 1e3:.0f} ms ({self.prompt_tokens} tokens, "
            f"{self.prefill_tok_s:.0f} tok/s) | TTFT {self.ttft_s * 1e3:.0f} ms | "
            f"{self.generated_tokens} tokens at {self.decode_tok_s:.1f} tok/s "
            f"({self.ms_per_token:.0f} ms/token, {self.host_share * 100:.0f}% host) | "
            f"total {self.total_s:.2f} s"
        )

    def rows(self) -> list[tuple[str, str, str]]:
        """(section, metric, value) -- the one source both the CLI report and the UI use."""

        def per(amount: float, count: float) -> float:
            return amount / count if count else 0.0

        # Pictures plus frames: everything the tower saw, counted as a viewer would. A frame
        # is cheaper than a picture, since paired frames share one temporal patch's tokens.
        visuals = self.images + self.frames
        vision = [
            ("vision", "pictures / frames", f"{self.images} / {self.frames}"),
            ("vision", "dispatches", f"{self.vision_dispatches}"),
            ("vision", "tokens produced", f"{self.vision_tokens}"),
            ("vision", "tokens per picture/frame", f"{per(self.vision_tokens, visuals):.0f}"),
            ("vision", "device", f"{self.vision_dispatch_only_s * 1e3:.0f} ms"),
            ("vision", "host patch quantize", f"{self.vision_quantize_s * 1e3:.1f} ms"),
            ("vision", "per dispatch", f"{self.ms_per_dispatch_median:.0f} ms"),
            ("vision", "per picture/frame", f"{per(self.vision_device_s * 1e3, visuals):.0f} ms"),
            (
                "vision",
                "throughput",
                f"{per(self.vision_dispatches, self.vision_device_s):.1f} dispatch/s",
            ),
            ("vision", "host prep", f"{self.vision_host_s * 1e3:.0f} ms"),
        ]
        if self.frames:  # only a clip has a frame rate to compare against its own fps
            fps = per(self.frames, self.vision_device_s)
            vision.append(("vision", "frames/s", f"{fps:.1f}"))

        # "of N positions": the graph ingests the whole context however long the prompt is,
        # so the rate is the prompt's size and the latency is the speed.
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
            # Only once the run has ended: mid-stream there is no measured clock to divide
            # by, and the stage sum would make this a duplicate of the row above.
            *(
                [("decode", "speed (wall)", f"{self.decode_wall_tok_s:.2f} tok/s")]
                if not math.isnan(self.wall_s) and self.decode_tokens
                else []
            ),
            (
                "decode",
                "device / host",
                f"{sum(self.step_device_s) * 1e3:.0f} / " f"{self.decode_host_s * 1e3:.0f} ms",
            ),
            ("decode", "host share", f"{self.host_share * 100:.0f} %"),
            # The host half, split. The lm_head is a fixed per-token cost that no amount
            # of decoder sharding touches, so it is quoted per token and as a share, with
            # the rate it would leave behind if it went away.
            (
                "decode",
                "host prep / lm_head",
                f"{self.decode_prep_s * 1e3:.0f} / " f"{self.decode_lmhead_s * 1e3:.0f} ms",
            ),
            (
                "decode",
                "lm_head per token",
                f"{self.ms_per_lmhead:.1f} ms " f"({self.lmhead_share * 100:.0f} % of a step)",
            ),
            ("decode", "speed without lm_head", f"{self.decode_tok_s_without_lmhead:.2f} tok/s"),
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
        if not math.isnan(self.load_s):  # a model whose load was timed
            total.insert(0, ("total", "first load", f"{self.load_s:.2f} s"))
        if not math.isnan(self.wall_s) and self.decode_tokens:  # a finished, timed run
            per_token = self.other_s * 1e3 / self.decode_tokens
            total.insert(
                -2,  # after TTFT, above the wall clock it is part of
                ("total", "framework", f"{self.other_s * 1e3:.0f} ms ({per_token:.1f} ms/token)"),
            )
        return [*vision, *prefill, *decode, *total]

    def stages(self) -> list[tuple[str, float]]:
        """Where the wall clock went, in the order the work happens.

        `other` (transformers' per-token work) only appears once the run has ended and
        there is a measured wall clock to subtract the stages from.
        """
        stages = [
            ("vision", self.vision_s),
            ("prefill", self.prefill_s),
            ("decode", self.decode_s),
        ]
        if self.other_s:
            stages.append(("other", self.other_s))
        return stages

    def time_bar(self, width: int = 44) -> str:
        """The stage split as one bar -- the terminal twin of the UI's coloured one."""
        total = max(self.total_s, 1e-9)
        blocks = {"vision": "█", "prefill": "▓", "decode": "░", "other": "·"}
        stages = self.stages()
        # Every stage that took any time keeps a block, so a fast one stays visible; the
        # widest then gives back what that rounding borrowed, since a bar wider than `width`
        # wraps the terminal and stops being a bar.
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
    """Define the shim against transformers (imported lazily, like evaluation.hf_model)."""
    from transformers import (
        AutoConfig,
        AutoModelForImageTextToText,
        AutoProcessor,
        GenerationConfig as HFGenerationConfig,
        PreTrainedModel,
    )
    from transformers.modeling_outputs import CausalLMOutputWithPast
    from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLConfig
    from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLForConditionalGeneration
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor

    class AxQwen3VLConfig(Qwen3VLConfig):
        """Qwen3-VL's own config under a model type of ours.

        Only the name differs, and only so the Auto classes route an ARTIFACT directory
        here: registering against `qwen3_vl` itself would hijack every ordinary
        `from_pretrained` in the process.
        """

        model_type = "ax_qwen3vl"

    class AxQwen3VLForConditionalGeneration(Qwen3VLForConditionalGeneration):
        """Qwen3-VL backed by a compiled Axelera artifact pair (vision + decoder)."""

        config_class = AxQwen3VLConfig

        def __init__(self, config):
            # Deliberately NOT Qwen3VLForConditionalGeneration.__init__: that allocates
            # billions of torch parameters we would immediately throw away -- the weights
            # are in the artifact. PreTrainedModel.__init__ gives the HF plumbing
            # (config, generation_config, dtype/device resolution) without the tensors.
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
            cls,
            artifact_dir,
            repo_id: str | None = None,
            *,
            generation: str | None = None,
            device=None,
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
            # directory); the checkpoint it was BUILT from is the useful answer, and what
            # the UI puts on screen.
            model.config.name_or_path = repo_id
            model._repo_id = repo_id
            # Timed: both towers' weights crossing PCIe is the one cost a demo watcher sees
            # before anything happens.
            started = perf_counter()
            model._pipeline = Qwen3VLPipeline(
                artifacts, repo_id, generation=generation, device=device
            )
            model._load_s = perf_counter() - started
            model.eval()
            return model

        # `from_pretrained` on a directory that is an artifact, not a checkpoint: the
        # familiar spelling, routed to the loader that fits.
        @classmethod
        def from_pretrained(cls, path, *args, **kwargs):
            if not (Path(path) / "text" / "pipeline.json").exists():
                raise ValueError(
                    f"{path} is not a compiled Qwen3-VL artifact (no text/pipeline.json). "
                    "Build one with llm.models.qwen3vl.pipeline.build_artifacts, or use "
                    "transformers.Qwen3VLForConditionalGeneration for a plain HF checkpoint."
                )
            # AutoModelForImageTextToText.from_pretrained hands over a config it already
            # parsed plus its whole hub kwarg set; the artifact carries all of that, so what
            # this loader declares is kept and the rest dropped. An ALLOW list: the Auto
            # factory also injects keywords of its own (adapter_kwargs, code_revision,
            # _from_auto), so a deny list goes stale and the miss lands as a TypeError from
            # inside `from_artifact`, which takes no **kwargs.
            kept = {k: kwargs[k] for k in ("repo_id", "generation", "device") if k in kwargs}
            return cls.from_artifact(path, *args, **kept)

        #: What the other Axelera model wrappers call it; `from_pretrained` is the HF
        #: spelling of the same load.
        load_compiled = from_pretrained

        def release(self) -> None:
            """Give back the sub-devices BOTH towers reserved. Idempotent.

            The name `AxeleraForCausalLM.release` uses, because the zoo's eval driver
            duck-types on it (`llm_runner._release_model`): a model that holds no device
            simply has no `release`, and a VLM that did not answer would have been a silent
            no-op there -- the leak reappearing only as
            `requested num_sub_devices(4) > available (4)` on the next load, which reads as
            a part too small for the build rather than a model still holding it.

            A device reservation is not freed by garbage collection, and this class holds
            TWO: the decoder's `RuntimeModel` and the vision tower's, the latter on a
            connection SHARED from the former. Order is therefore load-bearing -- closing
            the owner tears the context down under the tower it opened (see
            `RuntimeModel.close`), so the shared instance goes first. That ordering lives
            here rather than in the caller because this class is what knows there are two.

            Not a `__del__`, for `AxeleraForCausalLM.release`'s reason: collection order is
            not a lifecycle, and the caller that finished scoring is the one that knows the
            device is free to give back.
            """
            pipeline, self._pipeline = getattr(self, "_pipeline", None), None
            if pipeline is None:
                return
            for holder in (
                getattr(getattr(pipeline, "vision", None), "runtime", None),
                getattr(pipeline, "text_runtime", None),
            ):
                close = getattr(holder, "close", None)
                if callable(close):
                    close()

        @property
        def pipeline(self) -> Qwen3VLPipeline:
            if self._pipeline is None:
                raise RuntimeError("construct via AxQwen3VLForConditionalGeneration.from_artifact")
            return self._pipeline

        @property
        def processor(self):
            """The model's own AutoProcessor -- chat template, tokenizer, image/video."""
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
            """Seconds the artifact took to load onto the device, or nan if not measured.

            A one-off cost paid before the first prompt, so no RunStats owns it.
            `from_artifact` records it; a model built another way has no figure.
            """
            return getattr(self, "_load_s", float("nan"))

        def can_generate(self) -> bool:
            return True

        def get_input_embeddings(self):
            """None: the embedding table is in the artifact, not in torch.

            `PreTrainedModel`'s version reaches for `self.model.embed_tokens`, which this
            class never allocates -- so HF plumbing that merely asks (tied-weight
            bookkeeping, `resize_token_embeddings`' guard) would hit an AttributeError.
            """
            return None

        def set_input_embeddings(self, value):
            raise NotImplementedError(
                "the embedding table lives in the compiled artifact; change it in the "
                "checkpoint and rebuild"
            )

        def save_pretrained(self, *args, **kwargs):
            """Refused: there are no torch weights here to save.

            The inherited version would happily write a checkpoint containing this
            class's single placeholder parameter -- a directory that looks like a model
            and is not one. The artifact IS the saved form; copy the directory.
            """
            raise NotImplementedError(
                "this model's weights are the compiled artifact, not torch parameters: "
                "copy the artifact directory, or rebuild with "
                "llm.models.qwen3vl.pipeline.build_artifacts"
            )

        # ----- the one Axelera-specific step ------------------------------------------

        @property
        def image_shape(self) -> tuple[int, int]:
            """(height, width) in pixels the vision tower was compiled for."""
            return self.pipeline.config.vision.image_shape

        @property
        def image_geometry(self) -> str:
            """The compiled geometry for a log line: "256" if square, else "288x512" (HxW)."""
            return self.pipeline.config.vision.image_geometry

        def fit_images(self, images):
            """Resize picture(s) to the compiled geometry -- run before the processor.

            The tower is built for one geometry (square or not), so this is what pins the
            processor's patch grid to the one the device can encode. Videos: `fit_video`.
            """
            single = not isinstance(images, (list, tuple))
            fitted = [fit_image(self.pipeline.config, i) for i in ([images] if single else images)]
            return fitted[0] if single else fitted

        def fit_video(self, frames, metadata=None, fps: float = 2.0):
            """Frames -> (clip, metadata) for ``processor(videos=[clip], ...)``.

            Same geometry as the images (a video's temporal patches go through the same
            tower), padded up to a whole number of temporal patches. The metadata carries the
            frame times the processor writes into the prompt as ``<X.X seconds>`` labels:
            pass what `pipeline.load_video` returned to keep the video's own clock, or
            nothing to have `fps` invent one. Feed both to the processor with
            ``do_sample_frames=False`` -- these frames are already sampled and sized.
            """
            from .pipeline import fit_metadata, fit_video

            clip = fit_video(self.pipeline.config, frames)
            return clip, fit_metadata(metadata, clip, fps)

        # ----- prompt assembly from processor outputs ---------------------------------

        def _vision_inputs(self, input_ids, kwargs) -> VisionInputs:
            """A processor batch (input_ids + pixel_values + grids) -> VisionInputs."""

            def _grids(key):
                value = kwargs.get(key)
                if value is None:
                    return []
                return check_grids(
                    self.pipeline.config, [tuple(int(v) for v in row) for row in value]
                )

            def _patches(key):
                value = kwargs.get(key)
                return None if value is None else np.asarray(value, dtype=np.float32)

            return VisionInputs(
                input_ids=np.asarray(input_ids, dtype=np.int64).reshape(-1),
                patches=_patches("pixel_values"),
                grids=_grids("image_grid_thw"),
                video_patches=_patches("pixel_values_videos"),
                video_grids=_grids("video_grid_thw"),
            )

        # ----- forward: the device, in the shape GenerationMixin.generate expects ------

        def forward(
            self,
            input_ids=None,
            attention_mask=None,
            pixel_values=None,
            image_grid_thw=None,
            pixel_values_videos=None,
            video_grid_thw=None,
            cache_position=None,
            logits_to_keep=0,
            labels=None,
            **kwargs,
        ):
            """Logits for `input_ids`, prefilling or stepping the device as HF asks.

            The whole model is this one method, which is what lets transformers' own
            `generate()` drive: prefill vs decode comes from `cache_position`, exactly as
            for a torch model with a KV cache -- except the cache lives on the device, so
            nothing is handed back and forth. `logits_to_keep=1` (what generation asks for)
            computes the lm_head at the last position only; 0 computes every position.

            ``past_key_values`` comes back None on purpose: HF only threads it through, and
            this device's cache cannot be inspected, reordered or duplicated.
            """
            pipeline = self.pipeline
            impl, host = pipeline.impl, pipeline.host_params
            ids, row = row_ids(input_ids, attention_mask)
            prefill = cache_position is None or int(cache_position[0]) == 0

            if prefill:
                # Vision geometry first: a picture the processor sized its own way blows
                # the context too, and "resize it" is the useful half of that news.
                vision_inputs = self._vision_inputs(
                    row,
                    {
                        "pixel_values": pixel_values,
                        "image_grid_thw": image_grid_thw,
                        "pixel_values_videos": pixel_values_videos,
                        "video_grid_thw": video_grid_thw,
                    },
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
                    tower.last_quantize_s = 0.0
                    tower.last_dispatch_s = []
                # No cache reset between generations: the decoder's state is the
                # position-indexed KV cache and the prefill rewrites it from position 0.
                start = perf_counter()
                prompt = pipeline.prepare_inputs(vision_inputs)
                stats.vision_s = perf_counter() - start
                # The tower times its own dispatches; the rest of the above is host prep.
                stats.vision_dispatches, stats.vision_device_s = getattr(
                    pipeline.vision, "last_encode", (0, 0.0)
                )
                # ... and the host quantize that ran between those dispatches, so the
                # frame can be split into what the accelerator did and what the host did.
                stats.vision_quantize_s = getattr(pipeline.vision, "last_quantize_s", 0.0)
                stats.vision_dispatch_s = list(
                    getattr(pipeline.vision, "last_dispatch_s", []) or []
                )
                stats.vision_tokens = (
                    0 if prompt.image_embeds is None else len(prompt.image_embeds)
                )
                # Off the processor's (t, h, w) grids: t is 1 per picture, and for a clip it
                # counts temporal patches, so frames is t * temporal_patch_size. getattr all
                # the way down -- a stats count must never break a generation.
                vision_config = getattr(getattr(pipeline, "config", None), "vision", None)
                per_patch = getattr(vision_config, "temporal_patch_size", 1) or 1
                stats.images = sum(grid[0] for grid in vision_inputs.grids)
                stats.frames = sum(grid[0] for grid in vision_inputs.video_grids) * per_patch
                impl.set_prompt(prompt)

                # The shared ingestion, with its timing sink: the input prep (embedding
                # gather, causal mask, MRoPE tables, DeepStack scatter, tens of ms over a
                # whole context) is HOST work, and billing the accelerator for it would
                # mis-blame the device. chunk_len is None because chunked prefill is refused
                # for Qwen3-VL (see _chunk_inputs), so this is always the one-call route.
                hidden, last_row, self._position = impl.ingest_prompt(
                    pipeline.text_runtime, host, row, pipeline.context_len, None, timing=stats
                )
                start = perf_counter()
                logits = (
                    impl.compute_logits(hidden, host, position=last_row)
                    if logits_to_keep == 1
                    else impl.compute_logits_seq(hidden[:, : len(row)], host)
                )
                stats.prefill_host_s = perf_counter() - start
                # logits_to_keep=1 is generation asking for the next token: this prefill
                # produced the answer's first one. 0 is teacher-forced scoring, which
                # generates nothing.
                stats.prefill_generated = logits_to_keep == 1
            else:
                # decode_step inlined so the device call and the host lm_head are timed
                # apart -- on a small model the fp32 [vocab x hidden] matmul is a real
                # share of a step, and a demo that hides that mis-blames the accelerator.
                graph = impl._decode_graph_name()
                for token in row:  # generation feeds one; nothing stops a caller feeding more
                    self._check_room(self._position + 1)
                    # Sub-millisecond, but host work all the same, so it is timed with the
                    # lm_head rather than folded into the dispatch.
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
                    lmhead_s = perf_counter() - start
                    # The sum stays what it was; the two halves are recorded beside it,
                    # because "host" here is ~0.2 ms of bookkeeping plus a gemv two orders
                    # of magnitude bigger, and only the gemv is worth a kernel.
                    self.stats.step_prep_s.append(prep_s)
                    self.stats.step_lmhead_s.append(lmhead_s)
                    self.stats.step_host_s.append(prep_s + lmhead_s)
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

        def _check_room(self, length: int) -> None:
            if length > self.context_len:
                raise ValueError(
                    f"{length} tokens against the artifact's {self.context_len}-token "
                    "context: shorten the prompt, lower max_new_tokens, or build the "
                    "artifact with a larger context_len"
                )

        # ----- generate: HuggingFace's, with only the device bookkeeping added ---------

        def prepare_inputs_for_generation(
            self,
            input_ids,
            past_key_values=None,
            attention_mask=None,
            cache_position=None,
            pixel_values=None,
            image_grid_thw=None,
            pixel_values_videos=None,
            video_grid_thw=None,
            **kwargs,
        ):
            """What each generate() step passes to forward: the NEW tokens only.

            The stock Qwen3-VL version slices against a torch cache and recomputes the
            MRoPE deltas; here the device holds the cache and the host built the MRoPE
            tables with the prompt, so this only has to say "prefill" or "one more token".
            """
            prefill = cache_position is None or int(cache_position[0]) == 0
            inputs = {
                "input_ids": input_ids if prefill else input_ids[:, -1:],
                "cache_position": cache_position,
                "logits_to_keep": 1,  # generation only ever reads the last position
            }
            if prefill:  # the images ride on the prompt, and only on the prompt
                inputs.update(
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_grid_thw=image_grid_thw,
                    pixel_values_videos=pixel_values_videos,
                    video_grid_thw=video_grid_thw,
                )
            return inputs

        @torch.no_grad()
        def generate(self, *args, **kwargs):
            """`GenerationMixin.generate`, unchanged.

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
        """A Qwen3-VL config (from anywhere) as this model's config type."""
        if isinstance(config, AxQwen3VLConfig):
            return config
        data = config.to_dict()
        data.pop("model_type", None)  # the class attribute is the one that counts
        return AxQwen3VLConfig(**data)

    # The Auto classes, so an artifact directory loads the way any HF model does. Importing
    # this module is the registration route, which is why nothing here needs
    # trust_remote_code; exist_ok so a re-import does not explode.
    AutoConfig.register(AxQwen3VLConfig.model_type, AxQwen3VLConfig, exist_ok=True)
    AutoModelForImageTextToText.register(
        AxQwen3VLConfig, AxQwen3VLForConditionalGeneration, exist_ok=True
    )
    AutoProcessor.register(AxQwen3VLConfig, Qwen3VLProcessor, exist_ok=True)

    return AxQwen3VLConfig, AxQwen3VLForConditionalGeneration, as_ax_config


try:
    AxQwen3VLConfig, AxQwen3VLForConditionalGeneration, as_ax_config = _build_model_cls()
except ImportError:  # transformers absent: the rest of the suite still imports
    AxQwen3VLConfig = AxQwen3VLForConditionalGeneration = as_ax_config = None


def register_auto() -> None:
    """Teach the transformers Auto classes about Qwen3-VL's model type, so an artifact resolves.

    Call it once before `AutoModelForImageTextToText.from_pretrained(artifact_dir)`. One
    spelling for every AxScript family, text included: a customer moving between them should
    not have to learn a second one, and the failure when they use the other family's is
    "Transformers does not recognize this architecture", which names nothing they can act on.

    Importing this module registers too, wherever transformers is installed. Where it is not,
    the import leaves the names None and this call raises the ImportError that says so.
    """
    global AxQwen3VLConfig, AxQwen3VLForConditionalGeneration, as_ax_config

    if AxQwen3VLForConditionalGeneration is None:
        AxQwen3VLConfig, AxQwen3VLForConditionalGeneration, as_ax_config = _build_model_cls()


def artifact_config(repo_id: str):
    """The config to write into an artifact: the checkpoint's, under our model type.

    Same fields as the HF one -- `save_hf_metadata` writes this so the artifact loads
    through the Auto classes without claiming to be a plain Qwen3-VL checkpoint.
    """
    from transformers import AutoConfig

    config = as_ax_config(AutoConfig.from_pretrained(repo_id))
    config.name_or_path = repo_id  # provenance: what it was built from
    return config

# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""What one generation cost, split the way the machine is split.

One definition for every vision-language family here. The split -- vision tower, prefill,
decode, and the wall clock around them -- is a property of how these artifacts run and not
of any one architecture, so a family that timed its runs differently would be a family whose
numbers cannot go in a table beside the others'.

`lfm2vl` and `qwen3vl` each still carry their own copy of this dataclass, from before there
was a shared one. They are measured and released against those copies, so they are not
migrated here in the same change that introduces it; consuming this from all three is the
follow-up that removes the duplication.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import math


@dataclass
class RunStats:
    """One generation's cost, filled in by the shim's ``forward`` as the work happens.

    Three stages, each timed where the work is done: the **vision tower** (one dispatch per
    tile, plus host prep), the **prefill** (device decoder, its host input prep, and the host
    lm_head), and **decode** (the same, per token). Readable mid-stream, since the per-step
    lists grow as tokens land.

    A timer covers a device dispatch OR host work, never a mix. That split is the point: the
    lm_head is a ``[vocab x hidden]`` matmul on the host and a real share of a small model's
    decode step, and mixing it in would blame the accelerator for it.

    The stages do NOT add up to the run: transformers samples, runs the logits processors and
    drives the streamer between calls into ``forward``. So ``wall_s`` is measured around
    ``generate()`` and the difference surfaces as ``other_s`` rather than quietly flattering
    the device.
    """

    prompt_tokens: int = 0
    #: Positions the prefill graph ingests whatever the prompt's length -- it is compiled for
    #: the context, and the prompt is padded up to it.
    context_len: int = 0
    #: vision: one dispatch per tile (and one more for a split picture's thumbnail)
    vision_dispatches: int = 0
    vision_tokens: int = 0
    vision_device_s: float = 0.0
    vision_s: float = 0.0  # ... plus the host prep around it
    #: prefill: the decoder ingesting the whole prompt
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

    # --- vision -------------------------------------------------------------------------

    @property
    def vision_host_s(self) -> float:
        return max(0.0, self.vision_s - self.vision_device_s)

    @property
    def ms_per_dispatch(self) -> float:
        if not self.vision_dispatches:
            return 0.0
        return self.vision_device_s * 1e3 / self.vision_dispatches

    # --- prefill ------------------------------------------------------------------------

    @property
    def prefill_s(self) -> float:
        return self.prefill_device_s + self.prefill_prep_s + self.prefill_host_s

    @property
    def prefill_tok_s(self) -> float:
        """Prompt tokens per second of prefill -- a size, not a speed.

        The prefill graph is compiled for ``context_len`` positions and the prompt is padded
        up to it, so the latency is the same whatever the prompt's length: this number rises
        with the prompt while nothing gets quicker.
        """
        return self.prompt_tokens / self.prefill_s if self.prefill_s else float("nan")

    # --- decode -------------------------------------------------------------------------

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

    # --- totals -------------------------------------------------------------------------

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

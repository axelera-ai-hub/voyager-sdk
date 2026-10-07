# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""The build+run config a vision-language model in this family declares.

The seven text-only models return a :class:`~model_axscript.llm.generate.GenerationConfig`.
A VLM cannot: its build compiles TWO towers, and the vision one is pinned to a patch grid
that has no counterpart in a decoder-only config (``image_size``, ``fit_mode``). Rather
than widen ``GenerationConfig`` with fields six of seven models must leave unset, a VLM
declares this one.

Every field the rest of the stack reads off a model config keeps its ``GenerationConfig``
spelling -- ``repo_id``, ``artifact_dir``, ``context_len``, ``supported_context_lens``,
``chunk_len``, ``chunk_lens``, ``num_cores``, ``generation``, ``target``, ``cal_device`` -- because the zoo
reads them structurally and does not care which class it was handed:
``llm_runner.axscript_target_for`` takes the checkpoint id off ``repo_id``, and the entry's
own ``--help`` prints the declared sizes. A renamed field here would be a silent hole
there, so the overlap is a contract, not a coincidence.

Stdlib plus one leaf module that itself imports nothing, so asking a model package what it
declares costs no torch.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

# The one definition of the mapping, in a module that imports nothing: repeating it here
# is what let the two families' defaults drift.
from model_axscript.shared.hardware import CORES_PER_GENERATION, default_cal_device


@dataclass
class VlmConfig:
    """One VLM artifact: which checkpoint, at which geometry, over how many cores."""

    #: The decoder implementation the build registers. ``build_artifacts`` picks between the
    #: single-core and multicore kernels from ``num_cores``, so this names what a build at
    #: this core count resolves to; it is what lands in the artifact's ``pipeline.json``.
    implementation: str = "qwen3vl-metis-multicore"
    repo_id: str = "Qwen/Qwen3-VL-2B-Instruct"
    artifact_dir: Path = Path("build/qwen3-vl-2b-instruct/axscript")

    #: Compiled KV-cache capacity: prompt (text + image placeholder rows) + answer must fit.
    #: TTFT is a function of THIS, not of the prompt -- the prefill graph ingests every
    #: compiled position whatever it is sent -- so it is a latency decision as much as a
    #: capacity one.
    context_len: int = 512
    #: The context lengths this model has been measured at on chip. ``context_len`` must be
    #: one of them (``artifact_contract.validate_run_size`` refuses the rest before the
    #: compile starts): an unmeasured size does not fail at compile, it hangs at prefill
    #: minutes later and far from the cause. Widening the set is an on-chip measurement.
    supported_context_lens: tuple[int, ...] = (512,)
    #: Chunked prefill, as the text-only family spells it. The VLM path ingests the prompt in
    #: one call, so this is always None; it is declared because the entry's ``--help`` and the
    #: eval adapter both read it off a model config and must not special-case the family.
    chunk_len: int | None = None
    #: The text family's per-size chunk widths, aligned with ``supported_context_lens``.
    #: Always None here for the same reason ``chunk_len`` always is -- the VLM path ingests
    #: the prompt in one call -- and declared for the same reason too: the overlap with
    #: ``GenerationConfig`` is what lets the shared readers (the entry's ``--help``, the eval
    #: adapter, the context sweep) take a model config of either family without asking which.
    chunk_lens: tuple[int, ...] | None = None

    #: The patch grid the vision tower is pinned to, in pixels: (height, width). ONE artifact
    #: serves ONE geometry -- the patch count, the interpolated position embeddings, the 2-D
    #: RoPE tables and the attention mask are all baked at it -- so a second resolution is a
    #: second build, not a runtime flag.
    image_size: tuple[int, int] = (256, 256)
    #: The geometries measured on chip, the vision-side counterpart of
    #: ``supported_context_lens`` and gated the same way.
    supported_image_sizes: tuple[tuple[int, int], ...] = ((256, 256),)
    #: How a picture is made to fit the compiled grid, baked into the artifact so inference
    #: reproduces the geometry calibration saw. "stretch" uses every patch and distorts;
    #: "letterbox" keeps the trained aspect ratio and pads; "tile" is SmolVLM2's, which cuts
    #: native-size tiles so no single fit is chosen. Each pipeline refuses what it cannot do.
    fit_mode: str = "stretch"

    #: The (geometry, context) PAIRS this model has been measured at on chip, as
    #: ``run_sizes.INPUTS_DECLARED_ATTR`` mappings. A VLM is the model that paragraph is
    #: written for: it is compiled for a patch grid AND a context length, so neither
    #: ``supported_context_lens`` nor ``supported_image_sizes`` identifies an artifact on its
    #: own, and a report row may state only ONE identity kind, never both.
    #:
    #: Stated as PAIRS and not derived, because the two axes are gated independently
    #: (``validate_run_size`` admits each on its own) while a report row is one artifact. The
    #: cartesian product of the two sets is what the gates admit BETWEEN them, which is a
    #: superset of what anyone has run: a card at ``validated`` or above needs a measured row
    #: for every declared identity -- widening the set is a measurement, not an edit -- so
    #: declaring the product would claim combinations nobody compiled.
    #:
    #: Left empty, the product is used, which is right for a model measured at every
    #: combination and wrong the moment one axis grows without the other. Narrow it by
    #: stating the pairs.
    supported_inputs: tuple[dict[str, list[int]], ...] = ()

    @staticmethod
    def inputs_for(image_size: tuple[int, int], context_len: int) -> dict[str, list[int]]:
        """One compiled identity, spelled the way a report row spells it.

        Logical shapes with the dims explicit: `image` is NCHW at the compiled geometry and
        `tokens` is the decoder's compiled window. The one place the spelling lives, so the
        declared set and `axscript_vlm.row_inputs` cannot disagree about it.
        """
        height, width = image_size
        return {"image": [1, 3, int(height), int(width)], "tokens": [1, int(context_len)]}

    #: AI cores both towers shard over. A build reserves one sub-device per core at load, so
    #: this names a hardware requirement and not only a speed. None means "the whole part",
    #: resolved from ``generation`` through :data:`CORES_PER_GENERATION` -- so a model states which
    #: generation it targets and gets that generation's core count without repeating it.
    num_cores: int | None = None
    #: Hardware generation the artifact targets: OMEGA (Metis) or EUROPA. Both are supported
    #: by every VLM in this family; which one a model DEFAULTS to is its own declaration,
    #: because it follows from whether its resident weights fit that part.
    generation: str = "OMEGA"
    #: "device" for a real part, "sim" for AICoreSim.
    target: str = "device"

    #: DocVQA (document image, question) pairs the W8A8 post-training quantization is
    #: calibrated on. Both towers are calibrated together, which is what lets the decoder's
    #: activation scales see the embeddings the vision tower actually produces.
    num_calibration_samples: int = 48
    #: Calibrating on pictures of your own instead of DocVQA is a build-time choice with no
    #: shipped default, so it is NOT a field here: pass a sample list built by
    #: ``pipeline.multimodal_calibration`` to ``pipeline.build_artifacts(calibration=...)``.
    #: No entry point in this tree exposes that as a flag -- a shipped config declares the
    #: calibration a released artifact was actually built with, and a custom set is a caller
    #: reaching for the builder directly.
    smooth_alpha: float = 0.8
    qk_alpha: float = 0.5

    #: The torch device build-time calibration runs on ("cpu" / "cuda"). Distinct from the
    #: Metis device a run selects, and from ``generation``/``target``: calibration is the bulk
    #: of a build, so a GPU is worth having, and the artifact it produces is identical either
    #: way. Empty resolves to a GPU when there is one, mirroring ``GenerationConfig`` -- eager,
    #: so every consumer sees a concrete device rather than one path defaulting and another
    #: raising.
    cal_device: str = ""

    def __post_init__(self) -> None:
        # Normalized and checked here rather than at the compile, so a typo is a message
        # naming the two generations instead of a KeyError minutes into a build.
        self.generation = str(self.generation).upper()
        if self.generation not in CORES_PER_GENERATION:
            raise ValueError(
                f"unsupported hardware generation {self.generation!r}; supported: "
                f"{', '.join(CORES_PER_GENERATION)}"
            )
        if self.num_cores is None:
            self.num_cores = CORES_PER_GENERATION[self.generation]
        self.num_cores = int(self.num_cores)
        if not self.cal_device:
            self.cal_device = default_cal_device()
        # A square spelling is normalized here so nothing downstream carries two
        # representations of one geometry.
        if isinstance(self.image_size, int):
            self.image_size = (self.image_size, self.image_size)
        self.image_size = tuple(self.image_size)
        self.supported_image_sizes = tuple(
            (size, size) if isinstance(size, int) else tuple(size)
            for size in self.supported_image_sizes
        )
        # The product only when a model states no pairs. Filled HERE rather than on read so
        # `supported_inputs` is one value however it was arrived at: a reader (and the report
        # gate that checks the declared set against measured rows) never has to ask whether
        # it was declared or derived.
        if not self.supported_inputs:
            self.supported_inputs = tuple(
                self.inputs_for(size, context)
                for context in self.supported_context_lens
                for size in self.supported_image_sizes
            )
        self.artifact_dir = Path(self.artifact_dir)


def geometry_label(shape: tuple[int, int]) -> str:
    """``(256, 256)`` -> "256", ``(288, 512)`` -> "288x512": a geometry as a person writes it.

    The spelling a person writes and ``pipeline.build_artifacts``'s ``image_size`` accepts, so
    a geometry printed by ``--help`` reads the same way it is passed to a build.
    """
    height, width = shape
    return str(height) if height == width else f"{height}x{width}"

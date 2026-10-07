# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Evaluation metrics for a registered model implementation.

A Metric scores an implementation given an EvalContext (the shared, lazily-built
resources: the quantized reference model, the held-out corpus, the compiled
runtime). New metrics subclass Metric and read what they need from the context.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import json
import logging
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

from ..generate import (
    NUM_SUB_DEVICES,
    GenerationConfig,
    RuntimeModel,
    _tokenize_calibration,
    benchmark_generation,
    build_quantized_model,
    decode_prefill_consistency,
    device_perplexity,
)
from ..implementation import get_implementation
from ..models.architecture import get_architecture
from ..quantized_model import QuantizedModel
from .scoring import perplexity

logger = logging.getLogger(__name__)


@dataclass
class EvalContext:
    """Shared, lazily-built resources for evaluating one implementation.

    The quantized model, the exported bundle, and the compiled runtime are built at
    most once and reused across metrics: the quantize/calibration runs once and
    feeds both the quantized-model perplexity and the compiled artifact. The
    factory/override fields let tests inject tiny stand-ins (a random-weight model,
    a fake runtime) instead of downloading a checkpoint or needing a device.
    """

    config: GenerationConfig
    #: Performance-run seed text. None takes the model's own `GenerationConfig.prompt`,
    #: which is the one place a default prompt is declared: this dataclass already carries
    #: the config, and holding a second literal here gave a model that overrode the field
    #: two different `ttft_ms` numbers under one name.
    prompt: Optional[str] = None
    prompt_tokens: Optional[int] = None  # if set, benchmark a prompt of this many tokens
    eval_text: Optional[str] = None  # held-out perplexity corpus; None => WikiText-2 test
    eval_max_tokens: int = 16384  # cap the corpus so perplexity stays tractable
    tokenizer: object = None
    eval_ids: Optional[torch.Tensor] = None  # [1, T]; overrides eval_text when set
    calibration: Optional[list] = None  # calibration batches; None => from config
    reference_factory: Optional[Callable[[], torch.nn.Module]] = None
    runtime_override: Optional[tuple] = None  # (runtime, artifact_dir)
    # Both products are cached by default under artifact_dir and auto-reused when their cache key
    # (the inputs that produced them) matches the request; rebuild forces a fresh build of a layer.
    rebuild: frozenset = frozenset()  # subset of {"quant", "artifact"}; force-rebuild those layers
    reuse_artifact: bool = False  # deprecated: reuse is automatic + keyed; no-op
    reuse_quant: bool = False  # deprecated: reuse is automatic + keyed; no-op
    save_quantized_model: bool = True
    _quantized: object = field(default=None, init=False, repr=False)
    _bundle: object = field(default=None, init=False, repr=False)
    _reference: object = field(default=None, init=False, repr=False)

    @property
    def architecture(self) -> str:
        return get_implementation(self.config.implementation).architecture

    @property
    def device_lm_head(self) -> bool:
        """Whether this build actually puts the head on device: the torch reference and the
        bundle cache key both key off what the build does, not off the request."""
        impl = get_implementation(self.config.implementation)
        return impl.uses_device_head(self.config)

    @property
    def num_sub_devices(self) -> int:
        """Sub-devices to claim, from the config's target hardware generation."""
        return NUM_SUB_DEVICES[self.config.generation]

    def get_tokenizer(self):
        if self.tokenizer is None:
            from ..generate import load_tokenizer

            self.tokenizer = load_tokenizer(self.config.repo_id)
        return self.tokenizer

    def get_eval_ids(self) -> torch.Tensor:
        """The held-out corpus tokenized to a single [1, T] tensor (capped)."""
        if self.eval_ids is None:
            if self.eval_text is None:
                import datasets

                ds = datasets.load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
                self.eval_text = "\n\n".join(t for t in ds["text"] if t.strip())
            ids = self.get_tokenizer()(self.eval_text, return_tensors="pt").input_ids
            self.eval_ids = ids[:, : self.eval_max_tokens]
        return self.eval_ids

    def _get_calibration(self) -> list:
        if self.calibration is None:
            self.calibration = _tokenize_calibration(self.config)
        return self.calibration

    @staticmethod
    def _calibration_on(model, calibration: list) -> list:
        """The calibration samples on the MODEL's device, not `cal_device`.

        `_tokenize_calibration` places them on `config.cal_device` (auto-cuda wherever a GPU
        exists), while `load_reference` loads on `config.device` (default cpu, and the
        regression harness's fp leg wants it there). Those agree only on a CPU-only host, so on
        a GPU box every calibrate pass raised `index is on cuda:0, different from other tensors
        on cpu`. The BUILD path sidesteps it by loading the reference on `cal_device`
        (`generate.export_quantized_model`); here the reference's device is deliberate, so the
        data moves instead. Shared by both passes -- an architecture that records extra scales
        in a second pass (DeltaNet) needs it in both places or only the first one works.
        """
        target = next(model.parameters(), torch.empty(0)).device
        return [b.to(target) if hasattr(b, "to") else b for b in calibration]

    def perf_prompt(self):
        """The performance-run prompt: the seed text, or a token-id sequence of
        prompt_tokens tokens (tiled from the seed) for benchmarking TTFT vs length.

        Clamped to context_len - 1 so at least one token can be generated.
        """
        seed_text = self.prompt if self.prompt is not None else self.config.prompt
        if self.prompt_tokens is None:
            return seed_text
        n = min(self.prompt_tokens, self.config.context_len - 1)
        if n != self.prompt_tokens:
            logger.warning("prompt_tokens %d clamped to %d (context_len)", self.prompt_tokens, n)
        seed = self.get_tokenizer()(seed_text, return_tensors="np").input_ids[0]
        return np.tile(seed, n // len(seed) + 1)[:n]

    def load_reference(self) -> torch.nn.Module:
        """The float reference model, loaded once and cached.
        quantize() mutates it in place; the cache lets model_config() read its config
        without a second load before calibration."""
        if self._reference is None:
            if self.reference_factory is not None:
                self._reference = self.reference_factory().eval()
            else:
                arch = get_architecture(self.architecture)
                # quantize_lm_head only when asked: architectures without a device head have no
                # such config field, and passing it would fail their config override.
                overrides = {"quantize_attention": True}
                if self.device_lm_head:
                    overrides["quantize_lm_head"] = True
                self._reference = arch.load_pretrained(
                    self.config.repo_id,
                    dtype=getattr(torch, getattr(self.config, "calib_dtype", "float32")),
                    device=self.config.device,
                    **overrides,
                ).eval()
        return self._reference

    def model_config(self):
        """The architecture's model config (shapes). Available before calibration;
        device codegen needs only this, not the quantized values."""
        arch = get_architecture(self.architecture)
        return arch.config_from_dict(arch.config_to_dict(self.load_reference().config))

    def quantized_model(self) -> torch.nn.Module:
        """The W8A8 fake-quantized reference model (calibrated once, then cached).

        Shared by the quantized-model perplexity and the compiled artifact, so the
        calibration pass is not repeated.
        """
        if self._quantized is None:
            calib = self._get_calibration()
            logger.info(
                "calibrating + quantizing reference model (%d calibration samples)", len(calib)
            )
            model = self.load_reference()
            get_architecture(self.architecture).quantize(
                model,
                self._calibration_on(model, calib),
                smooth_alpha=self.config.smooth_alpha,
                qk_alpha=self.config.qk_alpha,
            )
            self._quantized = model
        return self._quantized

    def _quant_key(self) -> dict:
        """The inputs that determine the quantized bundle (calibration + smoothing)."""
        c = self.config
        return {
            "repo_id": c.repo_id,
            "architecture": self.architecture,
            "calib_budget": c.calib_budget,
            "smooth_alpha": c.smooth_alpha,
            "qk_alpha": c.qk_alpha,
            # Quantizes + unties the head, so the bundle itself differs.
            "device_lm_head": self.device_lm_head,
        }

    def _artifact_key(self) -> dict:
        """The inputs that determine the compiled artifact (the bundle + the compile shape)."""
        c = self.config
        return {
            "quant": self._quant_key(),
            "implementation": c.implementation,
            "context_len": c.context_len,
            "chunk_len": c.chunk_len,
            "num_cores": getattr(c, "num_cores", 1),
            # A compile-time memory budget: changing it changes the buffer placement, so
            # a cached artifact built at another value must not be reused.
            "resources": getattr(c, "resources", 1.0),
            "phases": sorted(c.phases),
            "target": c.target,
            "generation": c.generation,
        }

    @staticmethod
    def _cache_ok(d: Path, key: dict) -> bool:
        """A cached product is reusable if its recorded key matches, OR it predates keyed caching
        (no cache_key.json -> legacy, trust it)."""
        f = d / "cache_key.json"
        return (not f.exists()) or json.loads(f.read_text()) == key

    @staticmethod
    def _write_cache_key(d: Path, key: dict) -> None:
        (d / "cache_key.json").write_text(json.dumps(key, indent=2, sort_keys=True, default=str))

    def bundle(self):
        """The exported QuantizedModel bundle (calibrate + quantize + record LUT ranges + export).

        Cached at <artifact_dir>/quantized_model and persisted by default (see
        `save_quantized_model`); auto-reused when its cache key (repo / calib_budget /
        smoothing / device_lm_head) matches the request. 'quant' in rebuild forces a fresh
        calibration. (The fp32 / fake-quant PerplexityMetric re-quantizes independently; this
        cache only feeds the device compile.)
        """
        if self._bundle is None:
            bundle_dir = Path(self.config.artifact_dir) / "quantized_model"
            key = self._quant_key()
            if (
                "quant" not in self.rebuild
                and (bundle_dir / "quantized_model.json").exists()
                and self._cache_ok(bundle_dir, key)
            ):
                logger.info("reusing quantized bundle at %s (cache key matches)", bundle_dir)
                self._bundle = QuantizedModel.load(bundle_dir)
            else:
                arch = get_architecture(self.architecture)
                if getattr(self.config, "parallel", False) and self._quantized is None:
                    # Fold LUT-range (+ decode-scale) recording INTO the calibration passes.
                    arch.set_calibration_recording(self.load_reference(), True)
                    model = self.quantized_model()
                    arch.set_calibration_recording(model, False)
                else:
                    model = self.quantized_model()
                    logger.info("recording LUT ranges + exporting quantized bundle")
                    arch.record_lut_ranges(
                        model, self._calibration_on(model, self._get_calibration())
                    )
                self._bundle = build_quantized_model(self.architecture, model)
                if self.save_quantized_model:
                    self._bundle.save(bundle_dir)
                    self._write_cache_key(bundle_dir, key)
                    logger.info("saved quantized bundle to %s", bundle_dir)
        return self._bundle

    def get_runtime(self) -> tuple:
        """The compiled artifact, built and loaded once. Returns (runtime, artifact_dir).

        Cached at config.artifact_dir and auto-reused when its cache key (bundle + implementation
        + context_len + chunk_len + phases + target) matches the request -- changing any of these
        auto-recompiles instead of silently reusing a stale graph. 'artifact' in rebuild forces
        a recompile.
        """
        if self.runtime_override is None:
            artifact_dir = Path(self.config.artifact_dir)
            if self.reuse_artifact and not (artifact_dir / "model.json").exists():
                # Reuse is load-only: fail fast on a missing artifact BEFORE building the
                # bundle. Otherwise get_runtime falls through to the compile branch and
                # triggers a full real-model download + multi-thousand-sample calibration
                # just to report that the artifact is missing.
                raise FileNotFoundError(
                    f"reuse_artifact set but no compiled model at {artifact_dir / 'model.json'}; "
                    "build it first"
                )
            key = self._artifact_key()
            if (
                "artifact" not in self.rebuild
                and (artifact_dir / "model.json").exists()
                and self._cache_ok(artifact_dir, key)
            ):
                self.runtime_override = self._load_existing_artifact(artifact_dir)
            elif getattr(self.config, "parallel", False):
                # Build via the job scheduler (codegen ∥ calibration); prepare() sets
                # runtime_override (and writes the cache key) itself.
                self.prepare(build_runtime=True)
            else:
                impl = get_implementation(self.config.implementation)
                logger.info(
                    "compiling artifact (context_len=%d, chunk_len=%s, phases=%s) — "
                    "can take a while",
                    self.config.context_len,
                    self.config.chunk_len,
                    tuple(self.config.phases),
                )
                artifact_dir = impl.compile(self.bundle(), self.config)
                self._write_cache_key(artifact_dir, key)
                logger.info("loading runtime from %s", artifact_dir)
                self.runtime_override = (
                    RuntimeModel(artifact_dir / "model.json", self.num_sub_devices),
                    artifact_dir,
                )
        return self.runtime_override

    def resource_jobs(self, build_runtime: bool = True) -> list:
        """The shared-resource job graph for --parallel (evaluation.scheduler).

        Returns jobs that build the quantized ``"bundle"`` (calibration -- the long pole) and,
        when ``build_runtime``, the device artifact + ``"runtime"`` with codegen running in a
        spawn process pool *concurrent with* calibration. Side effects populate the caches the
        metrics read (``_bundle`` / ``_quantized`` via bundle(); ``runtime_override`` via the
        runtime job), so a metric job can simply depend on ``"bundle"`` or ``"runtime"`` and
        read the result off the context. ``build_runtime`` False (no device metric) builds only
        the bundle.
        """
        from functools import partial

        from ..implementation import codegen_job
        from .scheduler import Job

        impl = get_implementation(self.config.implementation)
        arch = get_architecture(self.architecture)
        artifact_dir = Path(self.config.artifact_dir)

        def _set_runtime(rt: tuple) -> tuple:
            self.runtime_override = rt
            return rt

        # Reference + shapes are needed before we can enumerate the codegen units; load them
        # eagerly -- nothing overlaps with them anyway (every stage depends on them).
        reference = self.load_reference()
        mc_dict = arch.config_to_dict(reference.config)
        model_config = self.model_config()

        jobs = [Job("bundle", (), lambda _i: self.bundle())]
        if build_runtime and self.runtime_override is None:
            have_artifact = (artifact_dir / "model.json").exists()
            if self.reuse_artifact and not have_artifact:
                raise FileNotFoundError(
                    f"reuse_artifact set but no compiled model at {artifact_dir / 'model.json'}; "
                    "build it first"
                )
            key = self._artifact_key()
            cached = (
                "artifact" not in self.rebuild
                and have_artifact
                and self._cache_ok(artifact_dir, key)
            )
            if cached or self.reuse_artifact:
                jobs.append(
                    Job(
                        "runtime",
                        (),
                        lambda _i: _set_runtime(self._load_existing_artifact(artifact_dir)),
                    )
                )
            else:
                codegen_names = []
                for kind, phase in impl.codegen_jobs(model_config, self.config):
                    name = f"codegen:{phase}:{kind}"
                    jobs.append(
                        Job(
                            name,
                            (),
                            partial(
                                codegen_job,
                                impl_name=self.config.implementation,
                                arch_name=self.architecture,
                                mc_dict=mc_dict,
                                config=self.config,
                                kind=kind,
                                phase=phase,
                            ),
                            in_process=True,
                        )
                    )
                    codegen_names.append(name)
                jobs.append(
                    Job(
                        "artifact",
                        tuple(["bundle"] + codegen_names),
                        lambda i: self._bind_from_codegen(impl, model_config, i, key),
                    )
                )
                jobs.append(
                    Job(
                        "runtime",
                        ("artifact",),
                        lambda i: _set_runtime(
                            (
                                RuntimeModel(i["artifact"] / "model.json", self.num_sub_devices),
                                i["artifact"],
                            )
                        ),
                    )
                )
        return jobs

    def prepare(self, build_runtime: bool = True) -> None:
        """Build the shared resources via the job scheduler, populating the caches the metrics
        read. Used by get_runtime's parallel path; run_evaluation folds these jobs into a larger
        graph (metrics included) instead of calling this."""
        from .scheduler import run_jobs

        jobs = self.resource_jobs(build_runtime)
        logger.info(
            "parallel build: %d codegen unit(s) ∥ calibration via job scheduler",
            sum(1 for j in jobs if j.in_process),
        )
        run_jobs(jobs)

    def _bind_from_codegen(self, impl, model_config, inputs: dict, key: dict) -> Path:
        """Artifact job: gather the codegen results from ``inputs`` and bind + author the
        device artifact against the (now-built) bundle, then persist the cache key."""
        from ..layout import extract_layer_buffer_spec

        device_atex: dict = {}
        specs: dict = {}
        for name, value in inputs.items():
            if name.startswith("codegen:"):
                phase, kind, model_json, atex = value
                device_atex[(phase, kind)] = Path(atex)
                specs[(phase, kind)] = extract_layer_buffer_spec(Path(model_json))
        artifact_dir = impl.bind_and_author(
            inputs["bundle"], model_config, device_atex, specs, self.config
        )
        self._write_cache_key(artifact_dir, key)
        return artifact_dir

    def _load_existing_artifact(self, artifact_dir: Path) -> tuple:
        """Load a pre-compiled artifact, adopting its context length."""
        model_json = artifact_dir / "model.json"
        if not model_json.exists():
            raise FileNotFoundError(
                f"reuse_artifact set but no compiled model at {model_json}; build it first"
            )
        built_ctx = int(json.loads((artifact_dir / "pipeline.json").read_text())["context_len"])
        if built_ctx != self.config.context_len:
            logger.warning(
                "reusing artifact at %s: its context_len=%d overrides --context-len %d",
                artifact_dir,
                built_ctx,
                self.config.context_len,
            )
            self.config.context_len = built_ctx
        logger.info("reusing compiled artifact at %s (skipping quantize + compile)", artifact_dir)
        return (RuntimeModel(model_json, self.num_sub_devices), artifact_dir)


class Metric(ABC):
    """A named evaluation that scores an implementation via its EvalContext."""

    name: str
    # The resource this metric needs, and thus the scheduler job it waits on under
    # --parallel: "runtime" (the compiled on-device artifact) or "bundle" (just the
    # quantized model -- so it can run concurrently with the device build).
    requires: str = "runtime"

    @abstractmethod
    def evaluate(self, ctx: EvalContext) -> dict[str, float]:
        """Return {result_name: value} for this metric."""


class PerplexityMetric(Metric):
    """Teacher-forced perplexity of the float reference and the W8A8-quantized
    model over the held-out corpus, at the config's context length."""

    name = "perplexity"
    requires = "bundle"  # only needs the quantized model -> overlaps the device build

    def evaluate(self, ctx: EvalContext) -> dict[str, float]:
        import gc

        window = ctx.config.context_len
        ids = ctx.get_eval_ids()
        # Score fp32 and W8A8 with only one set of weights resident at a time: each is
        # ~the full model size (e.g. ~7.5 GB for the 2B), so holding both OOMs a 12 GB GPU.
        # quantize() is in-place, so the fp32 baseline must be scored first on a throwaway
        # reference; after freeing it, build and score the shared quantized model (cached
        # for the device metrics). If the quantized model is already cached, this just no-ops.
        fp32 = perplexity(ctx.load_reference(), ids, window=window)
        gc.collect()
        if str(ctx.config.device).startswith("cuda"):
            torch.cuda.empty_cache()
        quant = perplexity(ctx.quantized_model(), ids, window=window)
        return {"perplexity": quant, "perplexity_fp32": fp32}


class PerformanceMetric(Metric):
    """Time-to-first-token (s) and steady-state decode throughput (tok/s) of the
    compiled artifact on the device."""

    name = "performance"

    def evaluate(self, ctx: EvalContext) -> dict[str, float]:
        runtime, artifact_dir = ctx.get_runtime()
        return benchmark_generation(
            runtime, ctx.get_tokenizer(), artifact_dir, ctx.perf_prompt(), ctx.config
        )


class OnDevicePerplexityMetric(Metric):
    """Teacher-forced perplexity of the compiled (on-device) model.

    Reflects the real device numerics (LUTs + float18), unlike the fake-quant
    PerplexityMetric. Runs one device step per token, so cap the corpus with
    eval_max_tokens to keep it tractable.
    """

    name = "device_perplexity"

    def evaluate(self, ctx: EvalContext) -> dict[str, float]:
        runtime, artifact_dir = ctx.get_runtime()
        ppl = device_perplexity(runtime, artifact_dir, ctx.get_eval_ids(), ctx.config)
        return {"device_perplexity": ppl}


class DeviceDecodeConsistencyMetric(Metric):
    """Prefill<->decode logit consistency (decode-path numerical soundness).

    device_perplexity only validates the prefill kernel (it takes the prefill graph when
    present). This drives the decode kernel over the same window and compares its
    per-position next-token logits against prefill's -- reference-free, so any divergence
    is purely the decode path (sdpa_decode + incremental int8 KV cache). Cost is one
    decode step per position, so it caps at DECODE_CONSISTENCY_POS positions (default 1024,
    ~100s at ~10 tok/s) to stay tractable on the one device.
    """

    name = "device_decode_consistency"

    def evaluate(self, ctx: EvalContext) -> dict[str, float]:
        import os

        runtime, artifact_dir = ctx.get_runtime()
        max_pos = int(os.environ.get("DECODE_CONSISTENCY_POS", "1024"))
        return decode_prefill_consistency(
            runtime,
            artifact_dir,
            ctx.get_eval_ids(),
            ctx.config,
            max_positions=max_pos,
            verbose=True,
        )

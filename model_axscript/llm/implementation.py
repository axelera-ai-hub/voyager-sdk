# Copyright Axelera AI, 2026
# SPDX-License-Identifier: Apache-2.0
"""Implementation interface + registry: how to run a model on Metis.

An Implementation reads a quantized_model.QuantizedModel, produces
a runnable artifact, and drives it (prefill + decode), decoupled from the
architecture (see models.architecture).

DecoderLayerImplementation is a reusable base for the common decoder-LLM
shape (decode layer on a device graph called once per layer; embedding + lm_head on
the host). A concrete implementation supplies only its
kernels.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import importlib
import inspect
import json
import os
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Any, Callable, Iterator

import numpy as np

# Stdlib only, by that module's own design, so it costs the run path nothing and stays at
# module scope while the build-side imports below do not.
from ..shared.artifact_contract import write_artifact_metadata
from .graph_descriptor import KindContract, TailGraph, author_graph_model, per_layer_spec

if TYPE_CHECKING:
    from axelera.compiler.alto.compiler.config import SyncMode

    # Build-side only. `quantized_model` imports torch at module scope, and the run path
    # must import without it, so the annotation must not become a real import.
    from .quantized_model import QuantizedModel


@dataclass(frozen=True)
class LayerKernel:
    """One decoder-layer kind (e.g. "full_attn" vs "deltanet"): how to build it,
    pack its per-layer constants, what host inputs it consumes, and what
    persistent state it owns. The actual compiled graph is the (phase x kind)
    cross product. A uniform stack is the degenerate single-kind case.

    name is the kind name (also the descriptor kernel-symbol stem);
    build(model_config, ctx_len, phase) -> IRModule produces the alto-script
    module; layer_buffers(bundle, buffer_spec, layer) -> {name: tensor} produces
    that layer's input buffers from the quantized model; phases are the runtime
    graphs to emit (each called once per layer of this kind, sharing the
    weights/state).
    """

    name: str
    build: Callable
    layer_buffers: Callable
    phases: tuple[str, ...] = ("decode",)
    #: Logical host inputs THIS kind's graph consumes each step (a full-attention
    #: layer needs rope + mask + seq_pos; a deltanet layer needs none). A subset of
    #: the implementation's shared_inputs; drives the descriptor's IO classification.
    shared_inputs: frozenset = field(default_factory=frozenset)
    #: This kind's persistent per-layer buffers (a KV cache, or a deltanet
    #: recurrent/conv state) -- codegen-allocated, one instance per layer.
    state: frozenset = field(default_factory=frozenset)
    #: Run the PromoteBuffers pass (promote workspace buffers to L2) for this kind.
    enable_buffer_promotion: bool = False
    #: Run the DatapathFusion pass for this kind. Off by default (as in AltoConfig): a
    #: kind opts in only once fusion is verified on its op chain. A kind whose kernel
    #: hand-builds its datapaths in a specific order (e.g. the deltanet mixer's IMC /
    #: IFD datapaths) must keep it off -- fusion reorders/merges them into device code
    #: that diverges from the interpreter (verified: fused deltanet output is garbage).
    enable_datapath_fusion: bool = False
    #: Logical host inputs supplied PER LAYER instead of once per step: bound to
    #: "<logical>_L<position>", so one kind can appear at several spine positions each
    #: reading its own buffer (Qwen3-VL's DeepStack adds). NOT part of shared_inputs --
    #: the runtime must produce the suffixed names (see KindContract.per_layer_inputs).
    per_layer_inputs: frozenset = field(default_factory=frozenset)
    #: Sync-injection mode for this kind, or None to keep the compiler default
    #: (SyncMode.HARDWARE). SyncMode.SOFTWARE mediates the cross-unit syncs through
    #: the AICore CPU instead of hardware token lines, which skips the ATEX
    #: InjectHWTokens pass -- needed by a kernel that pass gets wrong (see the
    #: qwen3.5 deltanet kernels, which are numerically wrong above cores=4 under
    #: HARDWARE and bit-exact under SOFTWARE, at no measured cost).
    auto_sync_mode: SyncMode | None = None


@dataclass(frozen=True)
class HostKernel:
    """A kernel executed on the host.

    Its behaviour is a method on the implementation (e.g.
    DecoderLayerImplementation.embed_tokens); this record declares the
    stage's name and the host-param keys it reads (params) for the manifest.
    """

    name: str
    params: tuple[str, ...] = field(default_factory=tuple)


Kernel = LayerKernel | HostKernel


class Implementation(ABC):
    """The per-model Step-2 contract: a QuantizedModel -> a runnable artifact."""

    #: registry key.
    name: str
    #: the architecture (registry key) this implementation realises.
    architecture: str

    @abstractmethod
    def compile(self, bundle: QuantizedModel, config) -> Path:
        """Produce the runnable artifact directory from bundle."""

    @abstractmethod
    def generate(
        self, runtime, tokenizer, prompt: str, config, *, host_params: dict, timing=None
    ) -> Iterator[str]:
        """Drive the compiled artifact, yielding decoded text as tokens arrive.

        runtime is a loaded generate.RuntimeModel; host_params is
        the loaded host_params.npz. timing is an optional `shared.run_stats.RunStats`
        sink for the prefill cost; None times nothing."""


class DecoderLayerImplementation(Implementation):
    """Reusable base for 'N x decode-layer (device) + host embedding/lm_head'.

    Subclasses supply kernels; compile compiles + authors the
    artifact, generate drives it through prefill + decode. The runtime hooks
    (embed_tokens, compute_logits, step_inputs) default to
    the LLaMA/Qwen-style convention.
    """

    @abstractmethod
    def layer_kernels(self, model_config) -> dict[str, "LayerKernel"]:
        """The layer kinds this implementation realises, keyed by kind name."""

    @abstractmethod
    def layer_kinds(self, model_config) -> list[str]:
        """The kind of each of the num_layers layers (length == num_layers).

        Uniform stacks repeat a single kind; hybrids read the architecture's
        model_config (e.g. config.layer_type(i)) to vary the kind per index.
        """

    def host_kernels(self) -> list[HostKernel]:
        """The host-side kernels (embedding / lm_head). Default: none."""
        return []

    def bundle_layers(self, kinds: list[str]) -> list[int | None]:
        """Which decoder layer each graph position binds its constants from.

        Default: position i IS decoder layer i. An implementation that splices auxiliary
        kernels into the activation spine (Qwen3-VL's DeepStack adds, which sit between
        decoder layers) returns None at those positions: they own no per-layer constants
        and do not consume a decoder layer, so the ones after them keep counting from
        where the trunk left off.
        """
        return list(range(len(kinds)))

    def has_device_head(self, runtime) -> bool:
        """Whether the LOADED ARTIFACT carries the device lm_head graph.

        The artifact decides, not the config: a build without the head graph falls back to
        the host gemv, so the two can never disagree about what was compiled. A caller may
        still force the host gemv for one run by setting ``force_host_head`` on the runtime;
        the artifact keeps its head graph, this run just never dispatches it. The host path
        needs only what host_params already carries (lm_head_weight, final_norm_weight,
        rms_eps), so no rebuild is involved -- which is what makes it usable to halve the
        dispatches per token when bisecting a device fault.
        """
        if getattr(runtime, "force_host_head", False):
            return False
        if not self.provides_device_head:
            return False
        from .models.lm_head_kernel import LM_HEAD_KIND

        return LM_HEAD_KIND in getattr(runtime, "graph_names", ())

    def _device_head_logits(self, rows: np.ndarray, runtime) -> np.ndarray:
        """Run the device head over every position of ``rows`` (1, n, hidden_pw, 64) -> [n, vocab].

        One call per position: the head graph is compiled at decode width, since decode is
        the path it exists to shorten.
        """
        from .models.lm_head_kernel import LM_HEAD_INPUT, LM_HEAD_KIND

        return np.stack(
            [
                np.asarray(
                    runtime.run(
                        LM_HEAD_KIND, {LM_HEAD_INPUT: np.ascontiguousarray(rows[:, i : i + 1])}
                    ),
                    dtype=np.float32,
                ).reshape(-1)
                for i in range(rows.shape[1])
            ]
        )

    #: Whether this implementation can build the lm_head as a device graph (subclasses that
    #: return one from tail_kernel set it).
    provides_device_head: bool = False

    def uses_device_head(self, config) -> bool:
        """Whether the build puts the lm_head on the device. Asking for one this
        implementation cannot build is an error rather than a silent host gemv, which would
        quantize the head into the bundle and then not use it."""
        requested = bool(getattr(config, "device_lm_head", False))
        if requested and not self.provides_device_head:
            raise ValueError(f"{self.name} provides no device lm_head")
        return requested

    def tail_kernel(self, config) -> "LayerKernel | None":
        """A standalone device graph run after the layer stack (the lm_head), or None.

        Not a layer kind: it is called once, owns no persistent state, and is fed the stack's
        hidden state rather than the previous layer's. It must declare exactly one phase.
        """
        self.uses_device_head(config)  # rejects a request this implementation cannot serve
        return None

    #: Host inputs the runtime supplies each decode/prefill step: the union of every
    #: layer kind's shared_inputs (each kind's graph binds only its own subset).
    #: Single source of truth shared with the runtime step_inputs and prefill_inputs.
    shared_inputs: frozenset = frozenset(
        {"attn_mask", "seq_pos", "q_rope_cos", "q_rope_sin", "k_rope_cos", "k_rope_sin"}
    )
    #: Activations threaded layer-to-layer (the previous layer's output). Uniform
    #: across kinds -- the inter-layer interface that lets one kind's output feed the next.
    threaded_inputs: frozenset = frozenset({"hidden_states", "residual_in"})
    #: Chunked prefill: may the final (partial) chunk be ingested as one PADDED prefill
    #: chunk instead of replaying its tokens through the decode graph? Safe ONLY for
    #: attention-only models (KV cache is the sole threaded state -- padded K/V lands at
    #: positions real tokens never attend to and decode overwrites). Recurrent/conv models
    #: (e.g. deltanet) MUST keep the conservative token-by-token default, since padding
    #: would corrupt the recurrent state decode continues from.
    chunked_remainder_via_padded_chunk: bool = False
    #: Shortest remainder that may go through that padded chunk. A mixer carrying a
    #: K-1 window needs at least K-1 real rows in the chunk, or the window it persists
    #: straddles the carried halo; a shorter remainder falls back to decode replay.
    min_padded_remainder: int = 1

    #: A padded remainder chunk costs a FULL chunk however few of its slots are real, so it
    #: beats decode replay only past chunk_len // this: the decode-step rate over the prefill
    #: rate, a quarter of a chunk by default. A model that measured its own rate declares it
    #: here (see the LFM2 implementation); `padded_chunk_breakeven` is the hook for one that
    #: has to key the ratio on the loaded artifact instead.
    padded_chunk_breakeven_ratio: int = 4

    def padded_chunk_breakeven(self, runtime) -> int:
        """`padded_chunk_breakeven_ratio`, resolved against the loaded artifact.

        A method as well as an attribute so an implementation whose ratio is not one number
        can derive it from what only the loaded artifact knows -- its hardware generation,
        core count or compiled chunk width. None needs to today, which is why `runtime` goes
        unread here: every family that measured a figure declares it as the attribute.

        An override must read the attribute, or defer to `super()`, rather than answering
        with a literal. One that answers unconditionally silently drops whatever a subclass
        declared, which is exactly how LFM2's measured ratio became dead code.
        """
        return self.padded_chunk_breakeven_ratio

    #: Chunked prefill: how many of the chunk's trailing slots must repeat the LAST real
    #: tokens' embeddings, so a depthwise conv's carried window (written from the last
    #: K-1 chunk inputs) lands on real data instead of padding. in_proj is a
    #: position-independent per-token map, so a copy produces the identical window.
    #: 0 = no conv window to preserve. Only consulted on the padded-remainder path.
    conv_window_replay: int = 0

    #: Shape/dtype of the attn_mask host input, matching what the layer kernel declares.
    #: The historical form is one fp32 row per query head; a kernel whose attn_mask buffer
    #: is (1, 1, seq, ctx/64) int8x64 sets mask_num_heads=1 / mask_dtype=np.int8 and lets
    #: the SDPA op broadcast across heads, which is 4*num_q_heads less mask to build on
    #: the host, upload, and stream from DDR once per attention layer. None => num_q_heads.
    #: See make_causal_mask for the int8 sentinel.
    mask_num_heads: int | None = None
    mask_dtype: Any = np.float32

    def causal_mask(self, positions, host_params, context_len: int) -> np.ndarray:
        """The attn_mask host input for these query positions, in the kernel's form."""
        heads = self.mask_num_heads or int(host_params["num_q_heads"])
        return make_causal_mask(positions, heads, context_len, dtype=self.mask_dtype)

    def compile(self, bundle: QuantizedModel, config) -> Path:
        # Lazy import: avoids an architecture <-> implementation import cycle.
        from .models.architecture import get_architecture

        model_config = get_architecture(bundle.architecture).config_from_dict(bundle.config)
        # Codegen (quant-independent) then bind+author (needs the bundle). Split so --parallel
        # can overlap the codegen with calibration (see EvalContext.get_runtime).
        device_atex, specs = self.compile_codegen(model_config, config)
        return self.bind_and_author(bundle, model_config, device_atex, specs, config)

    def codegen_jobs(self, model_config, config) -> list[tuple[str, str]]:
        """The (kind, phase) device-codegen units to compile (one .atex each)."""
        kernels = self.layer_kernels(model_config)
        self._check_shared_inputs(kernels)
        used_kinds = list(dict.fromkeys(self.layer_kinds(model_config)))
        jobs = [
            (kind, phase) for kind in used_kinds for phase in self._phases(kernels[kind], config)
        ]
        tail = self.tail_kernel(config)
        if tail is not None:
            jobs += [(tail.name, phase) for phase in self._phases(tail, config)]
        return jobs

    def kernels_by_kind(self, model_config, config) -> dict[str, "LayerKernel"]:
        """Every device kernel by kind: the layer kinds, plus the tail graph if there is one."""
        kernels = dict(self.layer_kernels(model_config))
        tail = self.tail_kernel(config)
        if tail is not None:
            if tail.name in kernels:
                raise ValueError(
                    f"tail kernel name {tail.name!r} collides with a layer kind; the two share "
                    "the compiled-artifact namespace"
                )
            kernels[tail.name] = tail
        return kernels

    def _tail_graph(self, bundle, specs, config) -> TailGraph | None:
        """Pack the tail graph's constants against its compiled spec."""
        tail = self.tail_kernel(config)
        if tail is None:
            return None
        phases = self._phases(tail, config)
        if len(phases) != 1:
            raise ValueError(
                f"tail kernel {tail.name!r} must declare exactly one phase; got {phases}"
            )
        phase = phases[0]
        return TailGraph(
            kind=tail.name,
            phase=phase,
            buffers=tail.layer_buffers(bundle, specs[(phase, tail.name)], 0),
        )

    def compile_codegen(self, model_config, config) -> tuple[dict, dict]:
        """Device codegen for every (kind, phase): returns {(phase,kind): atex} and {(phase,kind):
        spec}. Depends only on model_config (shapes/phases), not the quantized values -- so it can
        run before/concurrently with calibration. Sequential here; --parallel runs the units in
        a process pool (see EvalContext.get_runtime)."""
        from .layout import extract_layer_buffer_spec

        kernels = self.kernels_by_kind(model_config, config)
        device_atex: dict[tuple[str, str], Path] = {}
        specs: dict[tuple[str, str], dict] = {}
        for kind, phase in self.codegen_jobs(model_config, config):
            model_json, atex = self._compile_phase(kernels[kind], model_config, config, phase)
            device_atex[(phase, kind)] = Path(atex)
            specs[(phase, kind)] = extract_layer_buffer_spec(model_json)
        return device_atex, specs

    def bind_and_author(self, bundle, model_config, device_atex, specs, config) -> Path:
        """Bind the per-layer quantized constants from the bundle against the codegen specs, author
        model.json + blobs, write host_params + pipeline. The bundle-dependent half of compile().
        """
        out_dir = Path(config.artifact_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        kernels = self.layer_kernels(model_config)
        kinds = self.layer_kinds(model_config)
        layer_of = self.bundle_layers(kinds)
        if sum(layer is not None for layer in layer_of) != bundle.num_layers:
            raise ValueError(
                f"layer_kinds returned {len(kinds)} kinds covering "
                f"{sum(layer is not None for layer in layer_of)} decoder layers but the "
                f"model has {bundle.num_layers}"
            )
        used_kinds = list(dict.fromkeys(kinds))
        specs = {key: per_layer_spec(spec) for key, spec in specs.items()}
        # A kind's phases can declare different constant IO, so bind+merge every phase's buffers.
        buffers: dict[int, dict] = {}
        for position, kind in enumerate(kinds):
            merged: dict = {}
            for phase in self._phases(kernels[kind], config):
                merged.update(
                    kernels[kind].layer_buffers(bundle, specs[(phase, kind)], layer_of[position])
                )
            buffers[position] = merged

        if device_atex:
            self._author_device_model(
                device_atex,
                buffers,
                bundle,
                kinds,
                kernels,
                out_dir,
                config,
                self._tail_graph(bundle, specs, config),
            )

        np.savez(out_dir / "host_params.npz", **_host_params(bundle))
        pipeline = {
            "architecture": bundle.architecture,
            "implementation": self.name,
            # The HF checkpoint this artifact was built from -- recorded so downstream tools
            # (regression / evaluate / chat) can pick up the tokenizer/config without --repo-id.
            "repo_id": getattr(config, "repo_id", None),
            # Target hardware generation this artifact was compiled for ("OMEGA"/"EUROPA") --
            # recorded for the same reason: so downstream tools pick the right sub-device count
            # and device without --generation, and RuntimeModel can refuse a mismatched card.
            "generation": getattr(config, "generation", None),
            "num_layers": bundle.num_layers,
            "layer_kinds": kinds,
            "context_len": config.context_len,
            "chunk_len": getattr(
                config, "chunk_len", None
            ),  # chunked prefill: query width of the prefill graph
            # The core count is a measurement condition, not a detail: the same model at a
            # different shard width produces different numbers, so a report that does not
            # carry it invites a comparison that fabricates a regression.
            "num_cores": int(getattr(config, "num_cores", 1) or 1),
            "host_params": "host_params.npz",
            "device_model_json": "model.json",
            "stages": self._pipeline_stages(kernels, used_kinds, config),
        }
        (out_dir / "pipeline.json").write_text(json.dumps(pipeline, indent=2))
        # What this directory was COMPILED at, beside the pipeline that describes it. Authored
        # here rather than only in `generate.build_artifact`, because that is one of four
        # callers of `compile()` -- `shared.context_sweep._build_one`, `llm.build` and
        # `EvalContext.get_runtime` are the others, and each mirrors that body without this
        # step. An artifact from any of them is one `llm_runner._built_seq_len` REFUSES for
        # want of this file, and `axzoo report` then files a perf-only row and still exits 0,
        # so the omission reads as a model that has no accuracy rather than a build that
        # forgot to say what it is. Every path authors its artifact through here, so this is
        # the one place the file cannot go missing. `build_artifact` still writes it too:
        # same keys, same values, and the test that pins its orchestration keeps it honest.
        write_artifact_metadata(
            out_dir,
            compiled_seq_lens=[config.context_len],
            model_name=getattr(config, "repo_id", None) or "",
        )
        # The artifact becomes a HuggingFace model directory here, once, rather than in each
        # implementation: `AutoModelForCausalLM.from_pretrained(artifact_dir)` is the
        # customer-facing entry point, and it needs a config, a tokenizer and the generation
        # defaults sitting beside model.json. Imported inside the function because this
        # module is on the run path and that one reaches transformers. Last, and best effort:
        # it is the one optional step here, so the artifact's own record is already on disk
        # if a checkpoint the pinned transformers cannot read makes this one give up.
        from .huggingface import save_hf_metadata

        save_hf_metadata(out_dir, getattr(config, "repo_id", None))
        return out_dir

    def _pipeline_stages(self, kernels, used_kinds, config) -> list[dict]:
        """The ordered pipeline manifest (informational). Default: host kernels,
        then one device stage per distinct kind. UniformDecoderImplementation
        overrides this to preserve its declared embed -> layer -> lm_head order.
        """
        # A same-named tail graph REPLACES its host stage rather than adding a second one:
        # an implementation declares the host lm_head so a --device-lm-head false build
        # still names it, and the manifest is the only place anyone reads to find out
        # where the head actually runs.
        tail = self.tail_kernel(config)
        host = [
            {"name": k.name, "placement": "host", "params": list(k.params)}
            for k in self.host_kernels()
            if tail is None or k.name != tail.name
        ]
        device = [
            {
                "name": kind,
                "placement": "device",
                "graphs": list(self._phases(kernels[kind], config)),
            }
            for kind in used_kinds
        ]
        return host + device + self._tail_stage(config)

    def _tail_stage(self, config) -> list[dict]:
        """The tail graph's manifest entry (empty when there is no tail)."""
        tail = self.tail_kernel(config)
        if tail is None:
            return []
        return [{"name": tail.name, "placement": "device", "graphs": [tail.name]}]

    def _check_shared_inputs(self, kernels: dict[str, "LayerKernel"]) -> None:
        """Guard against a kind requesting a host input not in shared_inputs.
        Each kind's graph binds a subset of what the runtime supplies; a mismatch
        would only surface as a missing-input error at run time."""
        union: frozenset = frozenset()
        for kernel in kernels.values():
            union |= kernel.shared_inputs
        extra = union - self.shared_inputs
        if extra:
            raise ValueError(
                f"layer kinds consume host inputs {sorted(extra)} not in "
                f"{type(self).__name__}.shared_inputs; override shared_inputs (and "
                f"step_inputs/prefill_inputs to produce them)"
            )

    def _phases(self, kernel: "LayerKernel", config) -> tuple[str, ...]:
        """The phases to build for a kernel, honouring config.phases (if set)."""
        wanted = getattr(config, "phases", None)
        phases = (
            kernel.phases if wanted is None else tuple(p for p in kernel.phases if p in wanted)
        )
        if not phases:
            raise ValueError(
                f"no phases to compile for kind {kernel.name!r}: kernel declares "
                f"{kernel.phases}, config.phases={wanted}"
            )
        return phases

    def _compile_phase(
        self, kernel: "LayerKernel", model_config, config, phase: str
    ) -> tuple[Path, Path]:
        """Compile one phase; return (model.json, model.atex) paths.

        model.atex is what the graph renderer consumes; model.json supplies
        the per-layer buffer spec used to pack the constants.
        """
        from axelera.compiler.alto.compiler.config import (
            AltoConfig,
            HardwareGeneration,
            Target,
            alto_config_context,
        )
        from axelera.compiler.alto.compiler.driver import compile as alto_compile

        layer_dir = Path(config.artifact_dir) / f"{kernel.name}_{phase}"
        # LLM_REUSE_PHASES=1: reuse an already-compiled phase in layer_dir, skipping
        # recompile. Safe only when the kernel codegen is unchanged (host-side edits only).
        if os.environ.get("LLM_REUSE_PHASES"):
            mj = sorted(layer_dir.rglob("model.json"))
            ax = sorted(layer_dir.rglob("model.atex"))
            if mj and ax:
                return mj[0], ax[0]
        alto_target = Target.DEVICE if config.target == "device" else Target.AICORE_SIM
        build_kwargs = {}
        # Chunked prefill: pass chunk_len to builders that accept it; the prefill graph
        # then processes chunk_len queries against the full context_len KV cache.
        chunk_len = getattr(config, "chunk_len", None)
        if chunk_len and phase == "prefill":
            if "chunk_len" in inspect.signature(kernel.build).parameters:
                build_kwargs["chunk_len"] = chunk_len
            else:
                import logging

                logging.getLogger(__name__).warning(
                    "chunk_len=%s requested but builder %r does not accept it; compiling "
                    "FULL-context prefill -- chunked prefill is NOT applied (no compile-time "
                    "saving, and the host chunk-loop will mismatch the graph).",
                    chunk_len,
                    getattr(kernel, "name", phase),
                )
        # Multi-core: pass the requested core count to builders that accept a `cores`
        # arg (they distribute the SDPA heads across cores). AltoConfig.num_cores below
        # configures the system for that many cores so the per-core AICore regions are
        # valid; builders that don't take `cores` simply compile single-core.
        num_cores = int(getattr(config, "num_cores", 1) or 1)
        if num_cores > 1 and "cores" in inspect.signature(kernel.build).parameters:
            build_kwargs["cores"] = num_cores

        # Compiler pass flags a kind opts in/out of: the kernel declares the default (a
        # pass may assert on, or miscompile, a hand-built op chain), and the same-named
        # config field overrides it globally when set (None => keep the kernel default).
        def pass_flag(name: str) -> bool:
            override = getattr(config, name, None)
            return getattr(kernel, name) if override is None else bool(override)

        # Only forwarded when the kernel asks for a mode: AltoConfig's own None means
        # "inject no syncs at all", which is not the default (SyncMode.HARDWARE).
        sync_kw = (
            {} if kernel.auto_sync_mode is None else {"auto_sync_mode": kernel.auto_sync_mode}
        )
        alto_config = AltoConfig(
            output_dir=layer_dir,
            generation=HardwareGeneration[getattr(config, "generation", "OMEGA") or "OMEGA"],
            target=alto_target,
            num_cores=num_cores,
            # See GenerationConfig.resources: scales the compile's L2 budget to the
            # share the runtime will actually be given for this artifact.
            resources=float(getattr(config, "resources", 1.0) or 1.0),
            enable_tiling_injection=True,
            enable_buffer_promotion=pass_flag("enable_buffer_promotion"),
            enable_llm_support=True,
            assign_unique_input_pools=True,
            unpack_axm_archive=True,
            clean_output_dir=True,
            emit_atex_artifacts=True,
            emit_compilation_profile=getattr(config, "emit_compilation_profile", False),
            enable_datapath_fusion=pass_flag("enable_datapath_fusion"),
            **sync_kw,
        )
        # Same as `cores`: a builder that places a buffer in a specific DDR region has to be
        # told the target explicitly, because the trace is an ARGUMENT to alto_compile and so
        # runs before AltoConfig is consulted (see upper_shard_scope).
        if "generation" in inspect.signature(kernel.build).parameters:
            build_kwargs["generation"] = alto_config.generation
        # NB the ambient config makes an op's own unset `generation` Arg resolve to THIS
        # generation rather than to the OMEGA default, so it is not cosmetic. The qwen3.5
        # Europa branch dropped this wrap deliberately, reporting that its decode MLP tail
        # faulted ATEX's InjectHWTokens under generation=EUROPA at cores=8; it is restored
        # here because the shipped Europa VL artifacts are built through it, and because the
        # op that report named (fused_gate_up_swiglu) is passed generation=EUROPA explicitly
        # at both call sites, so the wrap cannot be what changes it. What the wrap DOES move
        # is the _nshard_qma projections, which pass no generation of their own. If that
        # fault returns, the per-kernel escape hatch is auto_sync_mode=SyncMode.SOFTWARE,
        # which models/qwen3_5/europa_multicore_impl.py already sets on three kernels.
        with alto_config_context(generation=alto_config.generation):
            mod = kernel.build(model_config, config.context_len, phase, **build_kwargs)
        alto_compile(mod, alto_config)
        model_jsons = sorted(layer_dir.rglob("model.json"))
        atexes = sorted(layer_dir.rglob("model.atex"))
        if not model_jsons or not atexes:
            raise RuntimeError(f"compile produced no model.json/model.atex under {layer_dir}")
        return model_jsons[0], atexes[0]

    def _author_device_model(
        self,
        device_atex: dict,
        buffers: dict,
        bundle: QuantizedModel,
        kinds: list[str],
        kernels: dict[str, "LayerKernel"],
        out_dir: Path,
        config,
        tail: TailGraph | None = None,
    ) -> None:
        """Author model.json (+ merged source/ELF + blobs) via the graph renderer.

        Delegates to graph_descriptor.author_graph_model, passing the per-layer
        kind list and each kind's IO contract (shared inputs + persistent state).
        """
        contracts = {
            kind: KindContract(
                shared_inputs=k.shared_inputs,
                state=k.state,
                per_layer_inputs=k.per_layer_inputs,
            )
            for kind, k in kernels.items()
        }
        author_graph_model(
            out_dir,
            bundle.architecture,
            device_atex,
            kinds,
            contracts,
            buffers,
            threaded_inputs=self.threaded_inputs,
            generation=(getattr(config, "generation", "OMEGA") or "OMEGA").lower(),
            tail=tail,
        )

    # Runtime hooks: drive the compiled artifact through prefill + decode.
    # The hooks below default to the LLaMA/Qwen-style convention.

    def select_prefill(self, n_tokens, chunk_len, context_len, available=()) -> int | None:
        """How to ingest a prompt of this length: the chunk length, or None for one call.

        The default has no choice to make, because an artifact carries one prefill graph. A
        build that compiles a second one overrides this and `_prefill_graph_name` together,
        so the two always name the same graph (see models.lfm2vl.text_impl). That is why the
        call order is part of the contract: an override records its choice here, and
        `ingest_prompt` reads it back through `_prefill_graph_name`, so every caller that
        ingests a prompt asks this first. `generate_token_ids` below, `forward` in
        `huggingface.py`, and the two vision adapters are the four of them.

        `prefill_hidden` deliberately does NOT ask, and is the one reader of
        `_prefill_graph_name` that does not. It is the per-window primitive behind
        teacher-forced scoring: its caller has already decided which graph and which chunk
        width the window is being scored through, and re-planning here would silently score
        a different graph than the one asked for.
        """
        return chunk_len

    def ingest_prompt(
        self,
        runtime,
        host_params,
        prompt: list[int],
        context_len: int,
        chunk_len: int | None,
        base_position: int = 0,
        timing=None,
        keep_hidden: list | None = None,
    ):
        """Seed the device cache with `prompt`; returns (hidden, row, position).

        `row` indexes the returned hidden at the prompt's LAST real token, which is the row a
        next-token logit comes from. It is always a valid index into `hidden`: the routes that
        end on a single decode step return -1, the last row, which is what they mean and what
        `compute_logits` defaults to. Never None: numpy would read `[0, None]` as an axis
        insertion rather than an error, so a null row would reshape the logits silently and
        fail two calls later.  `position` is where the next token will be written, which the
        caller must carry because the host, not the device, tracks the write cursor.

        `base_position` is the absolute position of the FIRST token of `prompt`. 0 is a fresh
        sequence; a non-zero value continues one the executor has already been fed, so
        `prompt` is only that turn's delta. That is what makes multi-turn chat linear in the
        conversation rather than quadratic, and it is the only way a mixer carrying a
        sequential accumulator can serve multi-turn at all, because its state cannot be
        rewound and re-feeding the history would double-count it.

        One method rather than one per caller: the decode loop and the HuggingFace `forward`
        must ingest a prompt identically, or the same artifact answers the same prompt two
        ways. Four routes, and which one applies is a property of the artifact and the model:

        * a prefill graph and no chunking: one call over the whole prompt. A non-zero
          `base_position` cannot take it, because that graph is compiled to start at 0.
        * chunked prefill, prompt a multiple of chunk_len: chunk after chunk, state threaded.
        * chunked prefill with a remainder, on an attention-only model: the remainder goes in
          as ONE right-padded chunk. The padding writes K/V past the real tokens, but causal
          masking means the real tokens never attend to it and decode overwrites those slots
          as it advances, so it is safe while the write stays inside the cache. This is the
          large time-to-first-token win when chunk_len does not divide the prompt, and it is
          taken only once the remainder is long enough to pay for a full-cost chunk.
        * chunked prefill with a remainder, on a recurrent or conv model: the remainder is
          replayed token by token through the decode graph. Padding would corrupt the
          threaded state that decode continues from, which is what
          `chunked_remainder_via_padded_chunk` decides.

        With no prefill graph at all, the whole prompt is replayed through decode.

        An empty prompt raises. Every route reports the hidden state of a real position, and
        there is none, so the chunked route would otherwise return the `hidden` it never
        assigned and `compute_logits` would fail on its shape two frames later.

        Two optional sinks, for callers that need more than the last position. `timing`
        accumulates the seconds of every dispatch into its `prefill_prep_s` (building the
        inputs, which is host work) and `prefill_device_s` (the dispatch itself); billing
        the accelerator for the host's embedding gather and causal mask is the reason they
        are split rather than timed together. `keep_hidden` collects each dispatch's hidden
        state, sliced to the positions that dispatch really wrote, so a caller scoring EVERY
        position can concatenate them along axis 1: the chunked routes overwrite `hidden`
        per chunk, and without this every chunk but the last is dropped.
        """
        if not prompt:
            raise ValueError("an empty prompt has no last token to read a next-token logit from")
        prefill_graph = self._prefill_graph_name()
        decode_graph = self._decode_graph_name()
        have_prefill = prefill_graph in getattr(runtime, "graph_names", ())
        fits = base_position + len(prompt) <= context_len

        def dispatch(graph, build, written: int):
            start = perf_counter()
            inputs = build()
            built = perf_counter()
            out = runtime.run(graph, inputs)
            if timing is not None:
                timing.prefill_prep_s += built - start
                timing.prefill_device_s += perf_counter() - built
            if keep_hidden is not None:
                keep_hidden.append(np.asarray(out)[:, :written])
            return out

        if have_prefill and not chunk_len and base_position == 0 and fits:
            hidden = dispatch(
                prefill_graph,
                lambda: self.prefill_inputs(host_params, prompt, context_len),
                len(prompt),
            )
            return hidden, len(prompt) - 1, len(prompt)

        if have_prefill and chunk_len and fits:
            n_full = (len(prompt) // chunk_len) * chunk_len  # the chunk-aligned prefix
            hidden = None
            for c0 in range(0, n_full, chunk_len):
                hidden = dispatch(
                    prefill_graph,
                    lambda c=c0: self._chunk_inputs(
                        host_params,
                        prompt[c : c + chunk_len],
                        base_position + c,
                        context_len,
                        chunk_len,
                    ),
                    chunk_len,
                )
            position = base_position + n_full
            remainder = prompt[n_full:]
            if not remainder:
                return hidden, chunk_len - 1, position
            if (
                self.chunked_remainder_via_padded_chunk
                and position + chunk_len <= context_len
                # Two independent floors: enough real rows for the mixer to be correct,
                # and enough of them for a full-cost padded chunk to beat decode replay.
                and len(remainder) >= self.min_padded_remainder
                and len(remainder) > chunk_len // self.padded_chunk_breakeven(runtime)
            ):
                hidden = dispatch(
                    prefill_graph,
                    lambda: self._chunk_inputs(
                        host_params, remainder, position, context_len, chunk_len
                    ),
                    len(remainder),
                )
                return hidden, len(remainder) - 1, position + len(remainder)
            for tok in remainder:
                hidden = dispatch(
                    decode_graph,
                    lambda t=tok, p=position: self.step_inputs(host_params, t, p, context_len),
                    1,
                )
                position += 1
            return hidden, -1, position

        position = base_position
        hidden = None
        for tok in prompt:
            hidden = dispatch(
                decode_graph,
                lambda t=tok, p=position: self.step_inputs(host_params, t, p, context_len),
                1,
            )
            position += 1
        return hidden, -1, position

    def generate_token_ids(
        self,
        runtime,
        tokenizer,
        prompt,
        config,
        *,
        host_params,
        base_position: int = 0,
        timing=None,
    ):
        """Autoregressive decode over the compiled artifact; yields generated token ids.

        The token-level core shared by generate() (text) and the HF shim. Ingests `prompt`
        (seeding the executor's persistent KV/recurrent state), then decodes one token per step,
        tracking the KV write position on the host and supplying the embeddings, RoPE tables, and
        causal mask each step. Stops at the eos set or config.max_new_tokens.

        base_position is the absolute position of the FIRST prompt token. 0 (the default) is a
        fresh sequence. A non-zero value continues one the executor has already been fed, so
        `prompt` is just that turn's delta -- which is what makes multi-turn chat linear in the
        conversation length instead of quadratic, and is the only way a mixer carrying a
        sequential accumulator (not a position-indexed cache) can serve multi-turn at all: its
        state cannot be rewound, so re-feeding the history would double-count it.

        `timing` is the optional `RunStats` sink `ingest_prompt` already accepts, threaded
        through so the TEXT path can report a prefill cost the way the vision-language shims
        already do (`models/qwen3vl/hf_model.py`). None -- the default -- times nothing, so
        every existing caller is unaffected. Only the prefill fields are filled: the decode
        loop below is a generator whose steps are consumed on the caller's clock, and billing
        `step_device_s` from in here would time the caller's own work between `next()` calls.
        """
        # prompt may be text to tokenize, or a pre-tokenized id sequence (e.g. a
        # synthesized fixed-length prompt for benchmarking).
        if isinstance(prompt, str):
            input_ids = tokenizer(prompt, return_tensors="np").input_ids[0]  # [T]
        else:
            input_ids = np.asarray(prompt).reshape(-1)
        # Stop on the full eos set: a model may end a turn with any of several tokens
        # (Qwen3 uses <|im_end|> and <|endoftext|>); honour config.stop_token_ids in addition
        # to tokenizer.eos_token_id, which is only a single id.
        eos_ids = {tokenizer.eos_token_id, *config.stop_token_ids}
        eos_ids.discard(None)
        ctx = config.context_len
        # Fixed-capacity KV cache; cap generation to fit what is left after the prefix
        # already fed (base_position > 0 on an incremental turn). max_new_tokens None asks
        # for everything that still fits, so the cache is the only limit.
        budget = ctx - base_position - len(input_ids)
        if config.max_new_tokens is not None:
            budget = min(budget, config.max_new_tokens)

        # Seed the KV cache and take the first logits. `ingest_prompt` owns every prefill
        # route, so this loop and HuggingFace's `forward` cannot ingest a prompt differently,
        # and `select_prefill` runs first for the same reason: it is where an override records
        # which prefill graph `ingest_prompt` will then name. The default returns `chunk_len`
        # unchanged, so a model with one prefill graph ingests exactly as it did before.
        chunk_len = self.select_prefill(
            len(input_ids),
            getattr(config, "chunk_len", None),
            ctx,
            getattr(runtime, "graph_names", ()),
        )
        hidden, row, position = self.ingest_prompt(
            runtime,
            host_params,
            [int(t) for t in input_ids],
            ctx,
            chunk_len,
            base_position=base_position,
            timing=timing,
        )
        # The lm_head at the last prefill position is host work and belongs to the prefill
        # bracket, not to decode: it is what turns the ingested prompt into the FIRST token.
        # Timed here rather than inside `ingest_prompt` because that method returns a hidden
        # state and knows nothing about logits, and `prefill_host_s` is documented as exactly
        # this ("final norm + lm_head at the last position").
        logits_start = perf_counter()
        logits = self.compute_logits(hidden, host_params, position=row, runtime=runtime)
        if timing is not None:
            timing.prefill_host_s += perf_counter() - logits_start
            # Sizes the prefill, so `prefill_tok_s` means something to a reader; the prompt
            # is this turn's delta, which is what the prefill actually ingested.
            timing.prompt_tokens = len(input_ids)
            timing.context_len = ctx
            timing.prefill_generated = True

        # Decode one token per step; the executor appends to the persistent cache.
        for _ in range(budget):
            token_id = sample_next_token(logits, config.temperature)
            if token_id in eos_ids:
                break
            yield token_id
            logits = self.decode_step(runtime, host_params, token_id, position, ctx)
            position += 1

    def generate(self, runtime, tokenizer, prompt, config, *, host_params, timing=None):
        """Autoregressive decode yielding cumulative decoded text -- a thin wrapper over
        generate_token_ids() (the shared token-level prefill+decode core)."""
        generated: list[int] = []
        for token_id in self.generate_token_ids(
            runtime, tokenizer, prompt, config, host_params=host_params, timing=timing
        ):
            generated.append(token_id)
            yield tokenizer.decode(generated, skip_special_tokens=True)

    def decode_step(self, runtime, host_params, token_id: int, position: int, context_len: int):
        """One decode step: next-token logits for token_id at position.

        The single-step primitive the decode loop and teacher-forced scoring share.
        """
        hidden = runtime.run(
            self._decode_graph_name(),
            self.step_inputs(host_params, token_id, position, context_len),
        )
        return self.compute_logits(hidden, host_params, runtime=runtime)

    def prefill_hidden(
        self,
        runtime,
        host_params,
        token_ids: list[int],
        context_len: int,
        chunk_len: int | None = None,
    ):
        """Run the prefill graph over token_ids; returns the hidden state for every
        position (1, len, hidden_pw, 64). The per-window primitive for teacher-forced perplexity.

        chunk_len: if set, the prefill graph was compiled for chunk_len query positions; the
        prompt is ingested in chunk_len-token chunks, threading recurrent/conv/KV state across
        chunks (the cache is zero-initialised at instance load, so chunk 0 starts fresh)."""
        if not chunk_len:
            return runtime.run(
                self._prefill_graph_name(),
                self.prefill_inputs(host_params, token_ids, context_len),
            )
        outs = []
        for c0 in range(0, len(token_ids), chunk_len):
            chunk = list(token_ids[c0 : c0 + chunk_len])
            h = np.asarray(
                runtime.run(
                    self._prefill_graph_name(),
                    self._chunk_inputs(host_params, chunk, c0, context_len, chunk_len),
                )
            )
            outs.append(h[:, : len(chunk)])  # drop the right-pad of the last chunk
        return np.concatenate(outs, axis=1)

    def _mask_inputs(self, positions, host_params, context_len: int) -> dict:
        """The host input(s) carrying causal-mask / query-position info for one call.

        One hook for all three builders (chunk / prefill / step), so a model whose attention
        consumes something other than a dense mask overrides this in ONE place instead of
        reimplementing every builder (and keeps them from drifting apart). Must return exactly
        the mask-related subset of ``shared_inputs``. Default: the dense additive causal mask
        in the form `causal_mask` declares (see mask_num_heads / mask_dtype).
        """
        return {"attn_mask": self.causal_mask(positions, host_params, context_len)}

    def _chunk_inputs(self, host_params, chunk_tokens, base_pos, context_len, chunk_len) -> dict:
        """Prefill inputs for one chunk starting at absolute position base_pos.
        Produces chunk_len queries (right-padded), each attending the full context_len
        KV cache causally; KV write at seq_pos=base_pos."""
        head_dim = int(host_params["head_dim"])
        rope_theta = float(host_params["rope_theta"])
        positions = np.arange(base_pos, base_pos + chunk_len)
        cos, sin = self._rope_tables(positions, head_dim, rope_theta)
        embeds = self.embed_tokens(np.array([chunk_tokens]), host_params)
        pad = chunk_len - len(chunk_tokens)
        if pad:
            embeds = np.pad(embeds, ((0, 0), (0, pad), (0, 0), (0, 0)))
            # Repeat the last real embeddings in the FINAL slots so a depthwise conv's
            # carried window -- written from the last K-1 chunk inputs -- sees real data
            # rather than padding. in_proj is per-token and position-independent, so the
            # copy yields the identical window the unpadded chunk would have left.
            # Clipped to `pad`: the destination is the last `w` slots, so a window wider
            # than the padding would reach back over REAL tokens and overwrite them with
            # earlier ones. Neither pad_ok floor bounds `pad` from below (both are floors
            # on len(remainder)), so pad < conv_window_replay is reachable.
            w = min(self.conv_window_replay, len(chunk_tokens), pad)
            if w:
                embeds[:, chunk_len - w :] = embeds[:, len(chunk_tokens) - w : len(chunk_tokens)]
        inputs = {
            "inputs_embeds": np.ascontiguousarray(embeds),
            "q_rope_cos": cos,
            "q_rope_sin": sin,
            "k_rope_cos": cos,
            "k_rope_sin": sin,
            "seq_pos": make_seq_pos(base_pos),
        }
        inputs.update(self._mask_inputs(positions, host_params, context_len))
        inputs.update(self._mixer_state_inputs(base_pos, len(chunk_tokens)))
        inputs.update(self._token_mask_inputs(len(chunk_tokens), chunk_len))
        assert set(inputs) == {"inputs_embeds"} | set(self.shared_inputs)
        return inputs

    def _decode_graph_name(self) -> str:
        """The device graph the decode loop drives (the decode-phase graph)."""
        return "decode"

    def _prefill_graph_name(self) -> str:
        """The graph that ingests the whole prompt in one call (if compiled)."""
        return "prefill"

    def prefill_inputs(self, host_params, token_ids: list[int], context_len: int) -> dict:
        """The {graph_input_name: array} set for a single prefill call.

        The prefill graph processes context_len query positions at once, so the
        embeddings are padded to context_len and the mask is full causal; the KV
        write starts at position 0.
        """
        if len(token_ids) > context_len:
            raise ValueError(f"prompt length {len(token_ids)} exceeds context_len {context_len}")
        head_dim = int(host_params["head_dim"])
        rope_theta = float(host_params["rope_theta"])
        positions = np.arange(context_len)
        cos, sin = self._rope_tables(positions, head_dim, rope_theta)
        embeds = self.embed_tokens(np.array([token_ids]), host_params)
        pad = context_len - len(token_ids)
        if pad:
            embeds = np.pad(embeds, ((0, 0), (0, pad), (0, 0), (0, 0)))
        inputs = {
            "inputs_embeds": np.ascontiguousarray(embeds),
            "q_rope_cos": cos,
            "q_rope_sin": sin,
            "k_rope_cos": cos,
            "k_rope_sin": sin,
            "seq_pos": make_seq_pos(0),
        }
        inputs.update(self._mask_inputs(positions, host_params, context_len))
        inputs.update(self._mixer_state_inputs(0, len(token_ids)))
        inputs.update(self._token_mask_inputs(len(token_ids), context_len))
        assert set(inputs) == {"inputs_embeds"} | set(self.shared_inputs)
        return inputs

    def step_inputs(self, host_params, token_id: int, position: int, context_len: int) -> dict:
        """The full {graph_input_name: array} set the host supplies for one step.

        The host owns the embedding lookup, RoPE tables, causal mask, and KV write
        position. Keys are exactly inputs_embeds plus shared_inputs (matching the
        compile-time IO classification), so the runtime and the manifest cannot drift.
        """
        head_dim = int(host_params["head_dim"])
        rope_theta = float(host_params["rope_theta"])
        cos, sin = self._rope_tables(position, head_dim, rope_theta)
        inputs = {
            "inputs_embeds": self.embed_tokens(np.array([[token_id]]), host_params),
            "q_rope_cos": cos,
            "q_rope_sin": sin,
            "k_rope_cos": cos,
            "k_rope_sin": sin,
            "seq_pos": make_seq_pos(position),
        }
        inputs.update(self._mask_inputs(position, host_params, context_len))
        inputs.update(self._mixer_state_inputs(position, 1))
        inputs.update(self._token_mask_inputs(1, 1))
        assert set(inputs) == {"inputs_embeds"} | set(self.shared_inputs), (
            "step_inputs must produce exactly the embeddings + declared shared_inputs; "
            "a model overriding shared_inputs must override step_inputs to match"
        )
        return inputs

    def _token_mask_inputs(self, valid_len: int, width: int) -> dict:
        """Per-call mask marking which of a call's `width` token slots are real.

        Default: none. A mixer whose state is a sequential accumulator (not a
        position-indexed cache) overrides this so a PADDED final prefill chunk can be
        made an exact no-op at the pad positions -- otherwise the padding advances the
        accumulator and the remainder has to be replayed through the decode graph.
        Keys must be a subset of shared_inputs.
        """
        return {}

    def _mixer_state_inputs(self, base_position: int, n_real: int) -> dict:
        """Per-call inputs for stateful token mixers (e.g. LFM2 conv / Qwen3.5 deltanet
        accumulators). Default: none.

        base_position is the absolute position of the first query in this call; 0 means a
        fresh sequence, so a model whose mixer carries a sequential accumulator (not a
        position-indexed cache like the KV cache) uses this to reset that accumulator at a
        sequence start -- which is exactly the multi-prompt reset (each new prompt/window
        prefills from position 0) and the hook a future incremental chat sets only at true
        sequence boundaries. Keys must be a subset of shared_inputs.
        """
        return {}

    def _rope_tables(self, position: int, head_dim: int, rope_theta: float):
        """RoPE cos/sin for one position (half-rotation; the LLaMA/Qwen convention)."""
        return make_rope_cos_sin(position, head_dim, rope_theta)

    def embed_tokens(self, token_ids: np.ndarray, host_params: dict) -> np.ndarray:
        """Host-side embedding lookup, returning float32 x64-packed embeddings.

        Gathers the rows of embed_weight for token_ids [1, T] and
        x64-packs them to the device's inputs_embeds layout (1, T, hidden_pw, 64).
        """
        ids = np.asarray(token_ids)
        rows = host_params["embed_weight"][ids].astype(np.float32)  # [1, T, hidden]
        batch, seq, hidden = rows.shape
        return np.ascontiguousarray(rows.reshape(batch, seq, hidden // 64, 64))

    def _cached_cast(self, weight: np.ndarray, slot: str, cast: Callable) -> Any:
        """`cast(weight)`, remembered so a decode step does not redo it.

        The lm_head is [vocab, hidden] and every conversion of it walks the whole matrix, which
        for Qwen3-0.6B is 151936 x 1024. `compute_logits` runs once per token, so a per-call
        cast is the largest host cost in the step rather than a rounding error on it. Keyed by
        the npz array's IDENTITY rather than its contents, so loading a second artifact into
        the same process rebuilds instead of answering with the first one's weights.
        """
        cache = getattr(self, slot, None)
        if cache is None or cache[0] is not weight:
            cache = (weight, cast(weight))
            setattr(self, slot, cache)
        return cache[1]

    def _lm_head_w(self, weight: np.ndarray):
        """The lm_head weight as a cached torch fp16 tensor [vocab, hidden].

        Kept as fp16 so torch's CPU gemv (fp16 in, fp32 accumulation) can be used directly,
        which halves memory bandwidth against an fp32 cast (~14 vs ~29 ms/step) with
        argmax-identical results (max logit error ~2e-3). The tensor is a zero-copy view of
        the npz array.
        """
        import torch

        return self._cached_cast(
            weight, "_lm_head_w_cache", lambda w: torch.from_numpy(np.ascontiguousarray(w))
        )

    def _lm_head_matmul(self, normed: np.ndarray, host_params: dict) -> np.ndarray:
        """`normed` [.., hidden] against the lm_head; logits [.., vocab]. Torch when present.

        The largest host cost per token, and the ONE place the run path wants torch. Torch's
        CPU gemv takes fp16 in with fp32 accumulation, which halves memory bandwidth against an
        fp32 cast: about 14 ms per step rather than about 29, argmax-identical, maximum logit
        error around 2e-3. So torch stays the fast path wherever it is installed.

        It is not a requirement, though. numpy computes the same product in fp32, and an edge
        target with no torch then runs slower rather than not at all, which is the difference
        between a dependency and an accelerator. numpy is not asked to do it in fp16: it has no
        fp16 GEMM and would upcast internally, paying the same bandwidth as fp32 while looking
        like the cheap path. Both paths CACHE their cast of the weight, because the artifact
        ships it as fp16 and neither of the two per-step figures above includes re-materialising
        [vocab, hidden] once per token.

        Untied models (e.g. Llama-3.1-8B) export a separate "lm_head_weight"; tied models reuse
        "embed_weight" (lm_head == embeddings), so the lookup below covers both.
        """
        weight = host_params.get("lm_head_weight", host_params["embed_weight"])
        try:
            import torch
        except ImportError:
            W = self._cached_cast(
                weight, "_lm_head_w32_cache", lambda w: np.asarray(w, dtype=np.float32)
            )
            return np.asarray(normed, dtype=np.float32) @ W.T
        W = self._lm_head_w(weight)  # cached torch fp16 tensor [vocab, hidden]
        normed_t = torch.from_numpy(np.ascontiguousarray(normed, dtype=np.float16))
        if normed_t.ndim == 1:
            return torch.mv(W, normed_t).numpy().astype(np.float32)
        return torch.matmul(normed_t, W.t()).numpy().astype(np.float32)

    def _final_norm(self, rows: np.ndarray, host_params: dict) -> np.ndarray:
        """The model's final norm, on the host, over the last axis of ``rows`` [.., hidden].

        RMSNorm -- no mean subtraction, no bias -- which is what every decoder in this
        tree wanted until Nemotron. An architecture whose final norm is a full LayerNorm
        overrides THIS rather than compute_logits: the two callers below differ only in
        which rows they select and whether the artifact carries a device head, and a
        subclass that re-implemented them would have to re-derive both. Shapes: a single
        row [hidden] and a block [seq, hidden] both work, since the reduction is over the
        last axis.
        """
        eps = float(host_params["rms_eps"])
        norm_weight = host_params["final_norm_weight"].astype(np.float32)
        return rows / np.sqrt((rows**2).mean(-1, keepdims=True) + eps) * norm_weight

    def compute_logits(
        self,
        device_hidden: np.ndarray,
        host_params: dict,
        position: int = -1,
        *,
        runtime=None,
    ) -> np.ndarray:
        """Final norm (see _final_norm) + lm_head at position (default: last), on the host.

        device_hidden is the graph's output hidden state, x64-packed
        (1, seq, hidden_pw, 64); returns next-token logits [vocab].

        Goes through the device head instead whenever the loaded artifact carries one, which
        is what ``runtime`` is for; it is unused on the host path.
        """
        h = np.asarray(device_hidden, dtype=np.float32)
        if self.has_device_head(runtime):
            row = position if position >= 0 else h.shape[1] + position
            return self._device_head_logits(h[:, row : row + 1], runtime)[0]
        batch, seq, hidden_pw, lane = h.shape
        last = h.reshape(batch, seq, hidden_pw * lane)[0, position]  # [hidden]
        return self._lm_head_matmul(self._final_norm(last, host_params), host_params)

    def compute_logits_seq(
        self, device_hidden: np.ndarray, host_params: dict, *, runtime=None
    ) -> np.ndarray:
        """Final norm (see _final_norm) + lm_head at every position; logits [seq, vocab].

        The all-positions form for teacher-forced (prefill) perplexity. Call on a
        slice of the prefill hidden to bound the [seq, vocab] matmul's memory.

        Goes through the device head when the artifact has one, as compute_logits does, so a
        decode-vs-prefill comparison keeps measuring the decode path rather than the head's
        quantization.
        """
        h = np.asarray(device_hidden, dtype=np.float32)
        if self.has_device_head(runtime):
            return self._device_head_logits(h, runtime)
        batch, seq, hidden_pw, lane = h.shape
        rows = h.reshape(batch, seq, hidden_pw * lane)[0]  # [seq, hidden]
        normed = self._final_norm(rows, host_params)
        return self._lm_head_matmul(normed, host_params)  # [seq, vocab]


class UniformDecoderImplementation(DecoderLayerImplementation):
    """A stack of identical layers -- the single-kind (non-hybrid) case.

    Subclasses implement kernels() returning the host kernels plus exactly one
    LayerKernel (the decoder layer); every layer uses that one kind, routing
    through the per-kind machinery with a single-entry kind map.
    """

    @abstractmethod
    def kernels(self) -> list[Kernel]:
        """The ordered kernels: host kernels + exactly one LayerKernel."""

    def _layer_kernel(self) -> "LayerKernel":
        layer = [k for k in self.kernels() if isinstance(k, LayerKernel)]
        if len(layer) != 1:
            raise ValueError(
                f"{type(self).__name__}.kernels() must return exactly one LayerKernel "
                f"(a uniform stack); got {len(layer)}. Use layer_kernels/layer_kinds "
                f"directly for a hybrid stack."
            )
        return layer[0]

    def host_kernels(self) -> list[HostKernel]:
        return [k for k in self.kernels() if isinstance(k, HostKernel)]

    def layer_kernels(self, model_config) -> dict[str, "LayerKernel"]:
        k = self._layer_kernel()
        return {k.name: k}

    def layer_kinds(self, model_config) -> list[str]:
        return [self._layer_kernel().name] * model_config.num_hidden_layers

    def _pipeline_stages(self, kernels, used_kinds, config) -> list[dict]:
        # Preserve the declared kernels() order: embed -> device layer -> lm_head.
        stages = []
        tail = self.tail_kernel(config)
        for k in self.kernels():
            if isinstance(k, HostKernel):
                if tail is not None and k.name == tail.name:  # see _pipeline_stages
                    continue
                stages.append({"name": k.name, "placement": "host", "params": list(k.params)})
            else:  # LayerKernel
                stages.append(
                    {
                        "name": k.name,
                        "placement": "device",
                        "graphs": list(self._phases(k, config)),
                    }
                )
        return stages + self._tail_stage(config)


# Decoder runtime defaults: per-step host inputs and sampling (pure helpers).

# Additive mask sentinel: -30000 (not float32-min) so masked positions underflow
# exp to 0 while staying float18-representable (float32-min NaNs the device exp).
_MASK_NEG = np.float32(-30000.0)


def make_seq_pos(position: int) -> np.ndarray:
    """The seq_pos buffer (KV-cache write index) as one uint64 PWORD (1, 64)."""
    return np.full((1, 64), position, dtype=np.uint64)


def make_valid_len(n_real: int) -> np.ndarray:
    """How many of a call's query rows are REAL, as one uint64 PWORD (1, 64).

    A right-padded prefill chunk hands causal_conv1d a T-row buffer of which only the
    first n_real are tokens. Without this the op persists the window from the tail of
    that buffer -- pad -- and the next decode step, plus every KV entry it writes,
    is computed from it. See ops/causal_conv1d.valid_len.
    """
    return np.full((1, 64), n_real, dtype=np.uint64)


def make_state_scale(base_position: int) -> np.ndarray:
    """Multiplicative gate on a mixer's carried accumulator window (causal_conv1d's
    state_scale): 0.0 (reset) at a sequence start (base_position 0), else 1.0 (continue).

    One (1, 1, 1) PWORD broadcast scalar (all lanes equal); the op broadcasts it over the
    K-1 window. Deriving it from position means each new prompt/window (which prefills from
    position 0) resets the accumulator automatically, and a future incremental chat that
    ingests new tokens at a running position > 0 continues it -- with no extra plumbing.
    """
    return np.full((1, 1, 1, 64), 0.0 if base_position == 0 else 1.0, dtype=np.float32)


def make_token_mask(valid_len: int, width: int) -> np.ndarray:
    """Per-token validity mask for one call, lane-packed (1, ceil(width/64), 64).

    1.0 for the first valid_len token slots, 0.0 for the rest. A mixer whose state is a
    sequential accumulator multiplies its gates by this, so padding a short final prefill
    chunk up to the compiled width is an exact no-op on that state rather than corrupting
    it -- see DecoderLayerImplementation._token_mask_inputs.
    """
    PWORD = 64
    blocks = max(1, -(-width // PWORD))
    mask = np.zeros(blocks * PWORD, dtype=np.float32)
    mask[: min(valid_len, blocks * PWORD)] = 1.0
    return mask.reshape(1, blocks, PWORD)


def make_rope_cos_sin(
    positions, head_dim: int, rope_theta: float
) -> tuple[np.ndarray, np.ndarray]:
    """RoPE cos/sin for one or more positions, x64-packed.

    positions is a single absolute position (decode) or a sequence (prefill); the
    result is (1, n_positions, rope_pw, 64). Half-rotation RoPE uses the
    head_dim/2 unique frequencies (applied to both halves).
    """
    if (head_dim // 2) % 64 != 0:
        raise ValueError(f"head_dim/2 must be a multiple of 64, got head_dim={head_dim}")
    pos = np.atleast_1d(np.asarray(positions, dtype=np.float64))
    inv_freq = 1.0 / (rope_theta ** (np.arange(0, head_dim, 2, dtype=np.float64) / head_dim))
    angles = pos[:, None] * inv_freq[None, :]
    rope_pw = (head_dim // 2) // 64
    cos = np.cos(angles).astype(np.float32).reshape(1, pos.size, rope_pw, 64)
    sin = np.sin(angles).astype(np.float32).reshape(1, pos.size, rope_pw, 64)
    return cos, sin


def make_causal_mask(
    positions, num_q_heads: int, context_len: int, dtype=np.float32
) -> np.ndarray:
    """Additive causal mask for one or more query positions, x64-packed.

    Each query attends cache positions 0..its_position and is masked from the
    future; same for every head. Result is (1, num_q_heads, n_positions,
    context_len/64, 64).

    dtype: mask element dtype. float* uses the -30000 additive sentinel; int8 uses
    -127 (int8 can't hold -30000, and -127 is exp-negligible after score scaling),
    halving the mask's DDR/L1 footprint. The SDPA op casts the int8 mask to float18.
    """
    if context_len % 64 != 0:
        raise ValueError(f"context_len must be a multiple of 64, got {context_len}")
    masked = -127 if np.dtype(dtype) == np.dtype(np.int8) else _MASK_NEG
    pos = np.atleast_1d(np.asarray(positions))
    keys = np.arange(context_len)
    rows = np.where(keys[None, :] <= pos[:, None], 0, masked).astype(dtype)
    mask = np.broadcast_to(rows[None, None, :, :], (1, num_q_heads, pos.size, context_len))
    return np.ascontiguousarray(mask).reshape(1, num_q_heads, pos.size, context_len // 64, 64)


def make_mask_query_positions(positions) -> np.ndarray:
    """Split-form absolute query positions, x64-packed: (n_positions, 2, 64).

    Column 0 = -64*(pos//64), column 1 = pos%64, each broadcast across the 64 lanes. This is
    what llm.sdpa_flash_europa's ``mask_query_positions`` takes: it REGENERATES the causal mask
    on-chip from these, instead of reading a dense (heads, seq, ctx/64) mask -- 2 MB rather than
    128 MB of DDR per layer call at chunk 1024 / ctx 4096, and no strided per-row mask DMA.

    Split rather than the raw position because the device adds a per-KV-tile term and evaluates
    the comparison in float18 (10-bit mantissa, exact only to 2048): both terms are multiples of
    64 bounded by the context, and the leftover lane offset stays in (-64, 64), so the sum is
    exact where it matters. Passing raw positions would break that.
    """
    pos = np.atleast_1d(np.asarray(positions, dtype=np.int64))
    out = np.zeros((pos.size, 2, 64), dtype=np.float32)
    out[:, 0, :] = (-64 * (pos // 64)).astype(np.float32)[:, None]
    out[:, 1, :] = (pos % 64).astype(np.float32)[:, None]
    return np.ascontiguousarray(out)


def sample_next_token(logits: np.ndarray, temperature: float) -> int:
    """Greedy (temperature 0) or temperature sampling over the logits."""
    logits = np.asarray(logits).astype(np.float32).reshape(-1)
    if temperature == 0:
        return int(np.argmax(logits))
    scaled = logits / temperature
    probs = np.exp(scaled - scaled.max())
    probs /= probs.sum()
    return int(np.random.choice(len(probs), p=probs))


def _host_params(bundle: QuantizedModel) -> dict:
    """The {name: np.ndarray} host params (embedding / final-norm + scalars)."""
    params = {
        name: (tensor.detach().cpu().numpy() if hasattr(tensor, "detach") else np.asarray(tensor))
        for name, tensor in bundle.host_tensors.items()
    }
    params.update(bundle.host_scalars)
    return params


_REGISTRY: dict[str, Implementation] = {}


def register_implementation(impl: Implementation) -> Implementation:
    """Register an implementation instance under its name (idempotent)."""
    _REGISTRY[impl.name] = impl
    return impl


#: The kernel modules that register an Implementation, named individually because a family
#: package registers nothing (see models.qwen3).
_BUNDLED_KERNELS = (
    "domyn.metis_impl",
    "lfm2.optimized_impl",
    "lfm2vl.text_impl",
    "llama.multicore_impl",
    "llama.naive_impl",
    "llama.optimized_impl",
    "qwen3.multicore_impl",
    "qwen3.naive_impl",
    "qwen3.optimized_impl",
    "qwen3_5.deltanet_impl",
    "qwen3_5.europa_multicore_impl",
    "qwen3vl.text_impl",
    "smolvlm2.multicore_impl",
)

_KERNELS_REGISTERED = False


def register_kernels() -> None:
    """Import the bundled device kernels once, so their registrations run. Idempotent.

    The kernel modules only, never their family package and never the torch reference beside
    it: the run path resolves an implementation by name on a machine that may have no torch
    installed, and a build reaches the reference through `architecture.register_architectures`
    instead. `tests/axscript/llm/test_core_import_boundary.py` holds that line.
    """
    global _KERNELS_REGISTERED
    if _KERNELS_REGISTERED:
        return
    _KERNELS_REGISTERED = True
    for module in _BUNDLED_KERNELS:
        importlib.import_module(f".models.{module}", __package__)


def get_implementation(name: str) -> Implementation:
    register_kernels()
    try:
        return _REGISTRY[name]
    except KeyError:
        raise KeyError(
            f"unknown implementation {name!r}; registered: {sorted(_REGISTRY)}"
        ) from None


def codegen_phase_worker(impl_name, arch_name, model_config_dict, config, kind, phase):
    """Process-pool entry point: compile one (kind, phase) device kernel in a fresh process.
    TVM codegen is not safe to run concurrently in-process. Args are picklable (names, a
    config dict, the GenerationConfig dataclass); returns the written (model.json, model.atex)
    paths as strings. Used by --parallel."""
    # Lazy import: avoids the architecture <-> implementation cycle at module import; relative so
    # it works whatever absolute package name this was launched under.
    from .models.architecture import get_architecture

    # A fresh worker process has an empty registry; the two lookups below fill their own.

    impl = get_implementation(impl_name)
    model_config = get_architecture(arch_name).config_from_dict(model_config_dict)
    kernel = impl.kernels_by_kind(model_config, config)[kind]
    model_json, atex = impl._compile_phase(kernel, model_config, config, phase)
    return (phase, kind, str(model_json), str(atex))


def codegen_job(inputs, *, impl_name, arch_name, mc_dict, config, kind, phase):
    """scheduler.Job adapter for codegen_phase_worker: a picklable callable taking the
    scheduler's (here unused) inputs dict, with everything else bound via functools.partial
    so the spawn process pool can ship it. See evaluation.scheduler / EvalContext.prepare."""
    return codegen_phase_worker(impl_name, arch_name, mc_dict, config, kind, phase)
